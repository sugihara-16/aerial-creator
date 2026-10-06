from dataclasses import replace
from copy import deepcopy
import math
import torch
import pytest
from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.policies.request_actor_critic import RequestActorCritic, sample_request_index
from amsrr.training.request_ppo import (
    serialize_encoding,
    collate_transitions,
    evaluate_batch,
    ppo_update,
    discounted_event_returns,
    categorical_kl_from_logits,
    branch_weighted_mean,
    request_runtime_contracts,
    generalized_event_advantages,
)
from amsrr.utils.hashing import hash_file


def test_event_gae_matches_mc_and_independent_irregular_time_td_reference():
    # Rewards actually occur inside intervals of unequal length.
    trajectory = [
        dict(time_s=0., end_time_s=2., reward=2., reward_events=[(1., 2.)],
             reward_timing='observed_reward_time_v1', value=3., done=False),
        dict(time_s=2., end_time_s=5., reward=4., reward_events=[(4., 4.)],
             reward_timing='observed_reward_time_v1', value=5., done=False),
        dict(time_s=5., end_time_s=6., reward=-1., reward_events=[(6., -1.)],
             reward_timing='observed_reward_time_v1', value=7., done=True),
    ]
    gamma = .9
    returns = discounted_event_returns(trajectory, gamma)
    assert generalized_event_advantages(trajectory, gamma, 1.) == pytest.approx(
        [r-e['value'] for r, e in zip(returns, trajectory)])
    td = [2*gamma + gamma**2*5 - 3, 4*gamma**2 + gamma**3*7 - 5, -gamma - 7]
    assert generalized_event_advantages(trajectory, gamma, 0.) == pytest.approx(td)
    expected = [td[0]+gamma**2*.8*(td[1]+gamma**3*.8*td[2]),
                td[1]+gamma**3*.8*td[2], td[2]]
    assert generalized_event_advantages(trajectory, gamma, .8) == pytest.approx(expected)
    trajectory[-1]['done'] = False
    with pytest.raises(ValueError, match='complete episode'):
        generalized_event_advantages(trajectory, gamma, .95)


def test_actor_validates_once_and_matches_independently_checked_features(request_scene, monkeypatch):
    from amsrr.policies.contact_group_geometry import contact_geometry
    from amsrr.schemas.common import SchemaValidationError
    context = decision(request_scene)
    model = RequestActorCritic().eval()
    expected_features = model.ranker.request_features(context)
    expected_candidates, expected_membership = contact_geometry(context)
    original = type(context).validate_snapshot
    calls = []
    def validate(value):
        calls.append(1)
        original(value)
    monkeypatch.setattr(type(context), 'validate_snapshot', validate)
    encoded = model.encode(context)
    assert len(calls) == 1
    for actual, expected in [(encoded['features'], expected_features),
        (encoded['candidates'], expected_candidates), (encoded['membership'], expected_membership)]:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    original(context)
    context.scene.runtime_observation.time_s += .25
    with pytest.raises(SchemaValidationError, match='modified'):
        model.encode(context)


@pytest.mark.parametrize("fault", [None, "missing_manifest", "changed_dataset", "wrong_record", "validation"])
def test_contact_groups_respect_hash_bound_canonical_split(tmp_path, fault):
    """An approved whole-episode reassignment stays explicit and identity-bound."""
    import json
    from amsrr.training.request_ppo import grouped_contact_baseline
    from amsrr.utils.hashing import stable_hash
    record = tmp_path / "record.json"
    record.write_text(json.dumps(dict(episode_id="case", dataset_split="validation")))
    event = dict(encoding={"initial": torch.tensor(True)})
    rollout = tmp_path / "rollout.pt"
    torch.save(dict(environment_seeds=[10, 11], episodes=[[event], [event]]), rollout)
    source = dict(episode_id="case", binding=dict(sha256=hash_file(record)),
                  original_split="validation", split="train")
    if fault == "wrong_record":
        source["binding"]["sha256"] = "wrong"
    if fault == "validation":
        source["split"] = "validation"
    dataset = tmp_path / "dataset.pt"
    torch.save(dict(sources=[source]), dataset)
    identity = dict(record_sha256=hash_file(record), snapshot_hash="same_initial_state")
    data = dict(version="same_initial_condition_rollout_groups_v1", group_size=2,
                rollouts=[dict(path=str(rollout), sha256=hash_file(rollout),
                               record_path=str(record), **identity,
                               condition_hash=stable_hash(identity))])
    if fault != "missing_manifest":
        data["canonical_dataset"] = dict(path=str(dataset), sha256=hash_file(dataset))
    if fault == "changed_dataset":
        torch.save(dict(sources=[]), dataset)
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps(data))
    args = ([event, event], torch.tensor([1., -1.]), [rollout], manifest)
    if fault is not None:
        with pytest.raises(ValueError):
            grouped_contact_baseline(*args)
    else:
        indices, advantages, audit = grouped_contact_baseline(*args)
        assert indices == [0, 1]
        torch.testing.assert_close(advantages, torch.tensor([2., -2.]))
        assert audit["canonical_dataset"] == data["canonical_dataset"]


def test_initial_actor_excludes_transitions_and_train_replay_is_identical(
    request_scene,
):
    torch.manual_seed(7)
    context = decision(request_scene)
    model = RequestActorCritic()
    result = model.decide(context)
    assert result["request"].transition_id is None
    data = collate_transitions([{"encoding": serialize_encoding(result["encoded"])}])
    model.train()
    scores, value = evaluate_batch(model, data, torch.tensor([0]), "cpu")
    log = torch.distributions.Categorical(logits=scores[0]).log_prob(
        torch.tensor(result["action"])
    )
    assert abs(float(log.detach()) - result["log_prob"]) < 1e-6
    restored = RequestActorCritic.from_checkpoint(model.checkpoint())
    torch.testing.assert_close(
        scores, evaluate_batch(restored, data, torch.tensor([0]), "cpu")[0]
    )


def test_ppo_rejects_teacher_and_greedy_and_updates_actor_from_sampled_events(
    request_scene, tmp_path
):
    torch.manual_seed(7)
    model = RequestActorCritic()
    checkpoint = tmp_path / "bc.pt"
    torch.save(model.checkpoint(), checkpoint)
    context = decision(request_scene)
    events = []
    for i in range(6):
        x = model.decide(context)
        events.append(
            dict(
                encoding=serialize_encoding(x["encoded"]),
                action=x["action"],
                value=x["value"],
                log_prob=x["log_prob"],
                reward=float(i % 2),
                done=i == 5,
                time_s=float(i),
            )
        )
    rollout = dict(
        checkpoint_sha256=hash_file(checkpoint),
        sampling="categorical",
        teacher_phase_supervision=False,
        complete=True, runtime_contracts=request_runtime_contracts(),
        episodes=[events],
    )
    path = tmp_path / "rollout.pt"
    torch.save(rollout, path)
    report = ppo_update(checkpoint, [path], tmp_path / "ppo")
    assert report["reload_exact"] and report["replay_log_probability_max_error"] < 2e-5
    assert report["parameter_max_changes"]["ranker.contact_head.3.weight"] > 0
    for key, value in [
        ("sampling", "greedy"),
        ("teacher_phase_supervision", True),
        ("diagnostic_only", True),
        ("evaluation_only", True),
        ("teacher_contact_supervision", True),
    ]:
        bad = {**rollout, key: value}
        torch.save(bad, path)
        with pytest.raises(ValueError, match="on-policy"):
            ppo_update(checkpoint, [path], tmp_path / key)
    # A genuine behavior-probability mismatch must still fail the individual
    # replay check, rather than being excused as batched floating-point error.
    bad = deepcopy(rollout)
    bad['episodes'][0][0]['log_prob'] += .01
    torch.save(bad, path)
    with pytest.raises(ValueError, match='on-policy replay probabilities differ'):
        ppo_update(checkpoint, [path], tmp_path/'wrong_behavior_probability')


def test_initial_input_geometry_changes_logits_but_label_has_no_channel(request_scene):
    model = RequestActorCritic()
    context = decision(request_scene)
    enc = model.encode(context)
    first, _ = model.evaluate_encoding(enc)
    modified = deepcopy(enc)
    modified["candidates"][0, 0] += 0.2
    second, _ = model.evaluate_encoding(modified)
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("changed", [None, "admission", "contact_velocity", "grasp_slip", "goal_completion"])
def test_ppo_rejects_old_runtime_despite_matching_checkpoint(tmp_path, changed):
    checkpoint = tmp_path / "model.pt"
    torch.save(RequestActorCritic().checkpoint(), checkpoint)
    rollout = dict(
        checkpoint_sha256=hash_file(checkpoint), sampling="categorical",
        teacher_phase_supervision=False, complete=True, episodes=[],
    )
    if changed is not None:
        rollout["runtime_contracts"] = request_runtime_contracts()
        rollout["runtime_contracts"][changed] = "obsolete"
    path = tmp_path / "rollout.pt"
    torch.save(rollout, path)
    with pytest.raises(ValueError, match="on-policy runtime contracts"):
        ppo_update(checkpoint, [path], tmp_path / "update")


def test_morphology_actor_replay_and_candidate_permutation(request_scene):
    torch.manual_seed(19)
    model = RequestActorCritic(morphology_aware=True).eval()
    encoded = model.encode(decision(request_scene))
    before = model.evaluate_encoding(encoded)
    perm = torch.arange(len(encoded["candidates"]) - 1, -1, -1)
    changed = deepcopy(encoded)
    changed["candidates"] = encoded["candidates"][perm]
    changed["owners"] = encoded["owners"][perm]
    changed["membership"] = encoded["membership"][:, perm]
    after = model.evaluate_encoding(changed)
    torch.testing.assert_close(before[0], after[0])
    replay = collate_transitions([{"encoding": serialize_encoding(encoded)}])
    restored = RequestActorCritic.from_checkpoint(model.checkpoint()).eval()
    scores, _ = evaluate_batch(restored, replay, torch.tensor([0]), "cpu")
    torch.testing.assert_close(before[0], scores)
    with pytest.raises(ValueError, match="owners"):
        restored.evaluate_encoding({k: v for k, v in encoded.items() if k != "owners"})


def test_morphology_actor_node_permutation(request_scene):
    model = RequestActorCritic(morphology_aware=True).eval()
    encoded = model.encode(decision(request_scene))
    graph = encoded["graph"]
    perm = torch.arange(graph.node_features.shape[1] - 1, -1, -1)
    inverse = perm.argsort()
    edges = graph.edge_index.clone()
    present = edges >= 0
    edges[present] = inverse[edges[present]]
    changed = deepcopy(encoded)
    changed["graph"] = replace(graph, node_features=graph.node_features[:, perm],
                               node_mask=graph.node_mask[:, perm], module_ids=graph.module_ids[:, perm], edge_index=edges)
    changed["owners"] = inverse[encoded["owners"]]
    torch.testing.assert_close(model.evaluate_encoding(encoded)[0], model.evaluate_encoding(changed)[0])


def test_event_discount_depends_on_elapsed_time_not_number_of_decisions():
    direct = [{"time_s": 0.0, "reward": 0.0}, {"time_s": 10.0, "reward": 1.0}]
    segmented = [direct[0], {"time_s": 4.0, "reward": 0.0}, direct[1]]
    assert discounted_event_returns(direct, 0.9)[0] == pytest.approx(0.9**10)
    assert discounted_event_returns(segmented, 0.9)[0] == pytest.approx(0.9**10)
    assert discounted_event_returns([{"reward": -5.0}], 0.9) == [-5.0]


def test_categorical_kl_uses_log_probabilities_when_float32_softmax_underflows():
    # Uniform binary p and q proportional to (1, exp(-110)):
    # KL(p || q) = 55 - log(2) + log(1 + exp(-110)). Padding has no mass.
    reference = torch.tensor([[0.0, 0.0, -torch.inf]])
    updated = torch.tensor([[0.0, -110.0, -torch.inf]])
    assert updated.softmax(-1)[0, 1] == 0
    actual = categorical_kl_from_logits(reference, updated)
    assert float(actual[0]) == pytest.approx(55 - math.log(2), abs=1e-12)
    assert float(categorical_kl_from_logits(reference, reference)[0]) == 0


def test_branch_weighting_is_independent_of_other_branch_frequency():
    # One contact loss of 4 and two temporal losses of 0,2 have means 4,1.
    values = torch.tensor([4., 0., 2.], requires_grad=True)
    mask = torch.tensor([True, False, False])
    loss = branch_weighted_mean(values, mask, .5)
    assert float(loss.detach()) == pytest.approx(2.5)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([.5, .25, .25]))
    repeated = torch.tensor([4., 0., 2., 0., 2., 0., 2.])
    assert float(branch_weighted_mean(repeated, torch.tensor([True] + [False] * 6), .5)) == 2.5
    with pytest.raises(ValueError, match="contact and temporal"):
        branch_weighted_mean(values, torch.ones(3, dtype=torch.bool), .5)


def test_balanced_ppo_restores_entire_model_when_either_branch_exceeds_limit(request_scene, tmp_path):
    torch.manual_seed(31)
    model = RequestActorCritic()
    checkpoint = tmp_path / "start.pt"
    torch.save(model.checkpoint(), checkpoint)
    context = decision(request_scene)
    episodes = []
    for i in range(6):
        trajectory = []
        for step in range(3):
            encoded = model.encode(context)
            encoded["initial"] = torch.tensor(step == 0)
            with torch.no_grad():
                logits, value = model.evaluate_encoding(encoded)
                dist = torch.distributions.Categorical(logits=logits[0])
                action = dist.sample()
            trajectory.append(dict(encoding=serialize_encoding(encoded), action=int(action),
                                   log_prob=float(dist.log_prob(action)), value=float(value[0]),
                                   time_s=float(step), reward=float((i + step) % 3 - 1), done=step == 2))
        episodes.append(trajectory)
    rollout = tmp_path / "rollout.pt"
    torch.save(dict(checkpoint_sha256=hash_file(checkpoint), sampling="categorical",
                    teacher_phase_supervision=False, complete=True, runtime_contracts=request_runtime_contracts(), episodes=episodes), rollout)
    settings = dict(learning_rate=.001, entropy_coefficient=.02, contact_loss_weight=.5,
                    contact_target_kl=100., temporal_target_kl=100.)
    wide = ppo_update(checkpoint, [rollout], tmp_path / "wide", epochs=3, **settings)
    first = ppo_update(checkpoint, [rollout], tmp_path / "first", epochs=1, **settings)
    # Use the measured first two unconstrained optimizer steps to put the
    # strict bound between them; the rejected second step must leave exactly
    # the first step's shared encoder, both heads, and critic parameters.
    for branch in ("contact", "temporal"):
        a, b = [row["branch_kl"][branch] for row in wide["updates"][:2]]
        assert b > a
        limited = {**settings, f"{branch}_target_kl": (a + b) / 2}
        report = ppo_update(checkpoint, [rollout], tmp_path / branch, epochs=3, **limited)
        assert report["optimization"]["completed_epochs"] == 1
        assert report["optimization"]["rejected_step"]["epoch"] == 2
        assert report["updates"][-1]["branch_kl"][branch] <= limited[f"{branch}_target_kl"]
        assert report["checkpoint_sha256"] == first["checkpoint_sha256"]
        actual = RequestActorCritic.load(tmp_path / branch / "checkpoint.pt")
        expected = RequestActorCritic.load(tmp_path / "first/checkpoint.pt")
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], tensor, rtol=0, atol=0)


def test_branch_limits_require_explicit_balanced_all_actor_configuration(tmp_path):
    for args in (dict(contact_target_kl=.05),
                 dict(contact_loss_weight=.5),
                 dict(contact_loss_weight=.5, contact_target_kl=.05, temporal_target_kl=.05,
                      actor_scope="temporal")):
        with pytest.raises(ValueError, match="balanced PPO|branch KL"):
            ppo_update(tmp_path / "unused.pt", [], tmp_path / "unused", **args)


def test_independent_temporal_budget_preserves_contact_and_rolls_back_adam(request_scene, tmp_path, monkeypatch):
    torch.manual_seed(31)
    model = RequestActorCritic().eval()
    checkpoint = tmp_path / 'initial.pt'
    torch.save(model.checkpoint(), checkpoint)
    episodes = []
    for i in range(12):
        episode = []
        for step in range(2):
            encoded = model.encode(decision(request_scene))
            encoded['initial'] = torch.tensor(step == 0)
            with torch.no_grad():
                logits, value = model.evaluate_encoding(encoded)
                dist = torch.distributions.Categorical(logits=logits[0])
                action = dist.sample()
            reward = float((i + step) % 3 - 1)
            episode.append(dict(encoding=serialize_encoding(encoded), action=int(action),
                log_prob=float(dist.log_prob(action)), value=float(value[0]),
                time_s=float(step), end_time_s=float(step+1), reward=reward, done=step == 1,
                reward_timing='observed_reward_time_v1', reward_events=[(float(step+1), reward)]))
        episodes.append(episode)
    path = tmp_path / 'rollout.pt'
    torch.save(dict(checkpoint_sha256=hash_file(checkpoint), sampling='categorical',
        teacher_phase_supervision=False, complete=True, runtime_contracts=request_runtime_contracts(),
        episodes=episodes), path)
    settings = dict(training_profile='timed_mc_v1', epochs=1, value_epochs=2,
        learning_rate=.001, entropy_coefficient=.02, contact_loss_weight=.5,
        contact_target_kl=100., temporal_target_kl=100.)
    base = ppo_update(checkpoint, [path], tmp_path/'base', **settings)
    extra = ppo_update(checkpoint, [path], tmp_path/'extra', temporal_epochs=3, **settings)
    a = RequestActorCritic.load(tmp_path/'base/checkpoint.pt')
    b = RequestActorCritic.load(tmp_path/'extra/checkpoint.pt')
    for name, parameter in a.named_parameters():
        if not name.startswith(('ranker.request_head.', 'ranker.hold_head.')):
            torch.testing.assert_close(parameter, dict(b.named_parameters())[name], rtol=0., atol=0.)
    assert extra['optimization']['completed_temporal_epochs'] == 3
    first, second = [r['temporal_kl'] for r in extra['temporal_updates'][:2]]
    assert second > first > base['updates'][-1]['branch_kl']['temporal']
    bounded = ppo_update(checkpoint, [path], tmp_path/'bounded', temporal_epochs=3,
        **{**settings, 'temporal_target_kl': (first+second)/2})
    direct = ppo_update(checkpoint, [path], tmp_path/'direct', temporal_epochs=1, **settings)
    assert bounded['optimization']['completed_temporal_epochs'] == 1
    assert bounded['optimization']['temporal_rejected_step']['epoch'] == 2
    assert bounded['checkpoint_sha256'] == direct['checkpoint_sha256']
    x = torch.load(tmp_path/'bounded/training_state.pt', weights_only=True)['actor_optimizer']
    y = torch.load(tmp_path/'direct/training_state.pt', weights_only=True)['actor_optimizer']
    torch.testing.assert_close(x, y, rtol=0., atol=0.)
    # Rejection of every shared/contact proposal must still permit a real
    # independent temporal update, and its receipt must count that update.
    monkeypatch.setattr('amsrr.training.request_ppo.preserve_observed_contact_preferences',
                        lambda *a, **k: dict(accepted=False))
    independent = ppo_update(checkpoint, [path], tmp_path/'only_temporal', temporal_epochs=3, **settings)
    assert independent['optimization']['completed_epochs'] == 0
    assert independent['optimization']['completed_actor_steps'] == 3
    actor = RequestActorCritic.load(tmp_path/'only_temporal/checkpoint.pt')
    for name, parameter in model.ranker.named_parameters():
        if not name.startswith(('request_head.', 'hold_head.')):
            torch.testing.assert_close(parameter, dict(actor.ranker.named_parameters())[name], rtol=0., atol=0.)


def test_timed_first_step_backtracking_restores_adam_before_retry(request_scene, tmp_path):
    torch.manual_seed(31)
    model = RequestActorCritic().eval()
    checkpoint = tmp_path / "initial.pt"
    torch.save(model.checkpoint(), checkpoint)
    episodes = []
    for i in range(8):
        episode = []
        for step in range(2):
            encoded = model.encode(decision(request_scene))
            encoded["initial"] = torch.tensor(step == 0)
            with torch.no_grad():
                logits, value = model.evaluate_encoding(encoded)
                distribution = torch.distributions.Categorical(logits=logits[0])
                action = distribution.sample()
            reward = float((i + step) % 3 - 1)
            episode.append(dict(encoding=serialize_encoding(encoded), action=int(action),
                log_prob=float(distribution.log_prob(action)), value=float(value[0]),
                time_s=float(step), end_time_s=float(step + 1), reward=reward, done=step == 1,
                reward_timing="observed_reward_time_v1", reward_events=[(float(step + 1), reward)]))
        episodes.append(episode)
    path = tmp_path / "rollout.pt"
    torch.save(dict(checkpoint_sha256=hash_file(checkpoint), sampling="categorical",
        teacher_phase_supervision=False, complete=True, runtime_contracts=request_runtime_contracts(), episodes=episodes), path)
    settings = dict(training_profile="timed_mc_v1", epochs=1, value_epochs=2,
                    learning_rate=.02, entropy_coefficient=.02, contact_loss_weight=.5,
                    contact_target_kl=100., temporal_target_kl=100.)
    wide = ppo_update(checkpoint, [path], tmp_path / "wide", **settings)
    limit = max(wide["updates"][0]["branch_kl"].values()) / 16
    report = ppo_update(checkpoint, [path], tmp_path / "bounded",
        **{**settings, "contact_target_kl": limit, "temporal_target_kl": limit})
    assert report["optimization"]["initial_step_backtracks"]
    assert report["optimization"]["completed_epochs"] == 1
    assert max(report["updates"][0]["branch_kl"].values()) <= limit
    rate = report["optimization"]["effective_actor_learning_rates"][0]
    assert 0 < rate < settings["learning_rate"]
    # Compare actor parameters/moments against the same pre-step state taking
    # exactly one smaller Adam step. Rejected attempts must not accumulate.
    direct = ppo_update(checkpoint, [path], tmp_path / "direct",
                       **{**settings, "learning_rate": rate})
    bounded_state = torch.load(tmp_path / "bounded/training_state.pt", weights_only=True)
    direct_state = torch.load(tmp_path / "direct/training_state.pt", weights_only=True)
    for key, state in bounded_state["actor_optimizer"]["state"].items():
        assert int(state["step"]) == 1
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, direct_state["actor_optimizer"]["state"][key][name], rtol=0, atol=0)
    a = RequestActorCritic.load(tmp_path / "bounded/checkpoint.pt")
    b = RequestActorCritic.load(tmp_path / "direct/checkpoint.pt")
    for key, tensor in a.ranker.state_dict().items():
        torch.testing.assert_close(tensor, b.ranker.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("updated", [[0.0, -torch.inf], [0.0, float("nan")]])
def test_categorical_kl_rejects_changed_support_or_nonfinite_valid_logits(updated):
    with pytest.raises(ValueError, match="valid action support"):
        categorical_kl_from_logits(torch.tensor([[0.0, -1.0]]), torch.tensor([updated]))


def test_temporal_ppo_preserves_contact_and_encoder_and_bounds_update(request_scene, tmp_path):
    torch.manual_seed(31)
    model = RequestActorCritic()
    checkpoint = tmp_path / "start.pt"
    torch.save(model.checkpoint(), checkpoint)
    context = decision(request_scene)
    events = []
    for i in range(8):
        encoded = model.encode(context)
        # Synthetic decision batches exercise both independent output branches;
        # physical catalog/phase behavior is covered by execution tests.
        encoded["initial"] = torch.tensor(i < 2)
        with torch.no_grad():
            logits, value = model.evaluate_encoding(encoded)
            dist = torch.distributions.Categorical(logits=logits[0])
            action = dist.sample()
        events.append(dict(encoding=serialize_encoding(encoded), action=int(action),
                           log_prob=float(dist.log_prob(action)), value=float(value[0]),
                           time_s=float(i), reward=5.0 if i % 2 else -5.0, done=i == 7))
    path = tmp_path / "rollout.pt"
    torch.save(dict(checkpoint_sha256=hash_file(checkpoint), sampling="categorical",
                    teacher_phase_supervision=False, complete=True, runtime_contracts=request_runtime_contracts(), episodes=[events]), path)
    report = ppo_update(checkpoint, [path], tmp_path / "fit", actor_scope="temporal",
                        epochs=20, learning_rate=.001, entropy_coefficient=.02, target_kl=1e-8)
    assert report["eligible_actor_events"] == 6
    assert report["optimization"]["completed_epochs"] == 1
    assert report["optimization"]["separate_actor_critic_gradient_clipping"]
    assert report["reload_exact"]
    delta = report["parameter_max_changes"]
    assert max(v for k, v in delta.items() if k.startswith("ranker.request_head.")) > 0
    assert all(v == 0 for k, v in delta.items()
               if not k.startswith(("ranker.request_head.", "value_head.")))
    restored = RequestActorCritic.load(tmp_path / "fit/checkpoint.pt")
    torch.testing.assert_close(model.decide(context, deterministic=True)["logits"],
                               restored.decide(context, deterministic=True)["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("second", [-1.0, float("nan"), float("inf")])
def test_event_discount_rejects_invalid_decision_clock(second):
    with pytest.raises(ValueError, match="timestamps"):
        discounted_event_returns(
            [{"time_s": 0.0, "reward": 0.0}, {"time_s": second, "reward": 1.0}], 0.99
        )


def test_seeded_sampling_is_invariant_to_catalog_order_and_subgoal_hash(request_scene):
    context = decision(request_scene)
    model = RequestActorCritic()
    scores, _ = model.evaluate_encoding(model.encode(context))
    entries = context.catalog.entries
    permutation = list(reversed(range(len(entries))))
    reordered = [deepcopy(entries[i]) for i in permutation]
    for entry in reordered:
        entry.request.subgoal_id += ":new_snapshot_hash"
    for seed in range(10):
        first = sample_request_index(
            entries, scores[0], generator=torch.Generator().manual_seed(seed)
        )
        second = sample_request_index(
            reordered,
            scores[0, permutation],
            generator=torch.Generator().manual_seed(seed),
        )
        assert (
            entries[first].request.contact_group_id
            == reordered[second].request.contact_group_id
        )
        assert entries[first].phase_id == reordered[second].phase_id


def test_physical_reward_discount_is_invariant_to_empty_intermediate_decisions():
    def event(start, end, occurrences):
        return dict(time_s=start, end_time_s=end, reward=sum(r for _, r in occurrences),
                    reward_timing='observed_reward_time_v1', reward_events=occurrences)
    direct = [event(0, 10, [(10, 5.)])]
    segmented = [event(0, 5, []), event(5, 10, [(10, 5.)])]
    expected = 5 * .99**10
    assert discounted_event_returns(direct, .99)[0] == pytest.approx(expected)
    assert discounted_event_returns(segmented, .99)[0] == pytest.approx(expected)
    assert discounted_event_returns([event(0, 0, [(0, -5.)])], .99) == [-5.]
    bad = event(0, 5, [(6, 1.)])
    with pytest.raises(ValueError, match='outside causal interval'):
        discounted_event_returns([bad], .99)
    bad = event(0, 5, [(4, 1.)]); bad['reward'] = 2.
    with pytest.raises(ValueError, match='raw sum'):
        discounted_event_returns([bad], .99)
    with pytest.raises(ValueError, match='mixed reward timing'):
        discounted_event_returns([event(0, 5, []), dict(time_s=5., reward=1.)], .99)


@pytest.mark.parametrize("profile", ["timed_mc_v1", "timed_value_v1", "timed_value_grouped",
                                     "timed_value_lr_change", "timed_value_gae"])
def test_timed_ppo_learns_known_reward_and_resumes_adam(request_scene, tmp_path, profile, monkeypatch):
    """Fresh on-policy bandit episodes have a known optimum, independent of code loss."""
    active_profile = "timed_value_v1" if profile.startswith("timed_value") else profile
    fitted_representations = []
    original_features = RequestActorCritic.frozen_value_features
    def capture_representation(model, *args, **kwargs):
        fitted_representations.append(deepcopy(model.ranker.state_dict()))
        return original_features(model, *args, **kwargs)
    monkeypatch.setattr(RequestActorCritic, 'frozen_value_features', capture_representation)
    torch.manual_seed(432)
    model = RequestActorCritic().eval()
    torch.nn.init.zeros_(model.ranker.contact_head[3].weight)
    torch.nn.init.zeros_(model.ranker.contact_head[3].bias)
    context = decision(request_scene)
    encoded = model.encode(context)
    support = torch.nonzero(encoded['mask']).flatten()
    assert len(support) >= 2
    desired = int(support[0])
    initial_probability = 1. / len(support)
    initial_path = tmp_path/'initial.pt'
    torch.save(model.checkpoint(), initial_path)
    path = initial_path
    final_report = None
    for update in range(4):
        model = RequestActorCritic.load(path).eval()
        with torch.no_grad():
            scores, value = model.evaluate_encoding(encoded)
            distribution = torch.distributions.Categorical(logits=scores[0])
        episodes = []
        for action in distribution.sample((96,)):
            reward = 1. if int(action) == desired else -1.
            episodes.append([dict(encoding=serialize_encoding(encoded), action=int(action),
                                  value=float(value[0]), log_prob=float(distribution.log_prob(action)),
                                  time_s=0., end_time_s=1., reward=reward, done=True,
                                  reward_timing='observed_reward_time_v1', reward_events=[(1., reward)])])
        rollout_path = tmp_path/f'rollout_{update}.pt'
        torch.save(dict(checkpoint_sha256=hash_file(path), sampling='categorical',
                        teacher_phase_supervision=False, complete=True, runtime_contracts=request_runtime_contracts(), episodes=episodes), rollout_path)
        out = tmp_path/f'update_{update}'
        extra = {}
        learning_rate = .0015 if profile == 'timed_value_lr_change' and update >= 2 else .003
        if profile == 'timed_value_gae' and update > 0:
            extra['gae_lambda'] = .95
        if profile == 'timed_value_lr_change' and update == 2:
            with pytest.raises(ValueError, match='explicit previous value'):
                ppo_update(path, [rollout_path], tmp_path/'undeclared_rate', training_profile=active_profile,
                           learning_rate=learning_rate)
            extra['previous_learning_rate'] = .003
        if profile == 'timed_value_grouped' and update >= 2:
            import json
            from amsrr.utils.hashing import stable_hash
            archive = torch.load(rollout_path, weights_only=True)
            archive['environment_seeds'] = list(range(96))
            torch.save(archive, rollout_path)
            record = tmp_path/'training_record.json'
            record.write_text(json.dumps(dict(dataset_split='train')))
            identity = dict(record_sha256=hash_file(record), snapshot_hash='synthetic_initial')
            manifest = tmp_path/f'groups_{update}.json'
            manifest.write_text(json.dumps(dict(version='same_initial_condition_rollout_groups_v1',
                group_size=96, rollouts=[dict(path=str(rollout_path), sha256=hash_file(rollout_path),
                record_path=str(record), **identity, condition_hash=stable_hash(identity))])))
            extra['contact_group_manifest'] = manifest
            if update == 2:
                with pytest.raises(ValueError, match='explicit previous contract'):
                    ppo_update(path, [rollout_path], tmp_path/'undeclared_change', training_profile=active_profile,
                               learning_rate=.003, **extra)
                extra['contact_baseline_transition_from'] = 'value_v1'
        final_report = ppo_update(path, [rollout_path], out, training_profile=active_profile, **extra,
                                  epochs=8, value_epochs=5, learning_rate=learning_rate,
                                  entropy_coefficient=.001)
        assert final_report['optimization']['optimizer_updates_resumed'] == update
        if 'contact_group_manifest' in extra:
            assert final_report['contact_baseline_kind'] == 'condition_loo_v1'
            assert final_report['contact_baseline']['condition_count'] == 1
            assert (final_report['contact_baseline_transition'] is not None) == (update == 2)
        if profile == 'timed_value_lr_change':
            assert (final_report['optimization']['learning_rate_transition'] is not None) == (update == 2)
        if profile == 'timed_value_gae':
            assert final_report['optimization']['gae_lambda_per_decision'] == (.95 if update > 0 else None)
        expected_baseline = 'collected_value' if active_profile == 'timed_value_v1' and update > 0 else 'batch_constant'
        assert final_report['value_diagnostics']['baseline'] == expected_baseline
        assert math.isfinite(final_report['value_diagnostics']['collected_value_mse'])
        path = out/'checkpoint.pt'
        # The critic must describe the representation actually saved for the
        # next rollout; later actor updates must not invalidate that fit.
        assert final_report['optimization']['value_fit_order'] == 'after_actor'
        saved_ranker = RequestActorCritic.load(path).ranker.state_dict()
        for name, tensor in fitted_representations[-1].items():
            torch.testing.assert_close(tensor, saved_ranker[name], rtol=0, atol=0)
    state = torch.load(path.parent/'training_state.pt', weights_only=True)
    assert state['completed_updates'] == 4
    actor_steps = [int(v['step']) for v in state['actor_optimizer']['state'].values()]
    assert max(actor_steps) == 32  # eight steps in each of four updates, not reset
    assert {int(v['step']) for v in state['value_optimizer']['state'].values()} == {20}
    with torch.no_grad():
        logits, _ = RequestActorCritic.load(path).evaluate_encoding(encoded)
    assert int(logits[0].argmax()) == desired
    assert float(logits[0].softmax(-1)[desired]) > initial_probability + .2


@pytest.mark.parametrize("morphology_aware", [False, True])
def test_frozen_value_feature_reuse_preserves_every_adam_step(request_scene, morphology_aware):
    from amsrr.training.request_imitation import graph_batch
    torch.manual_seed(317)
    model = RequestActorCritic(morphology_aware=morphology_aware).eval()
    encoded = model.encode(decision(request_scene))
    data = collate_transitions([dict(encoding=serialize_encoding(encoded)) for _ in range(3)])
    ids = torch.arange(3)
    with pytest.raises(ValueError, match="frozen ranker"):
        model.frozen_value_features(data['features'], graph_batch(data['graphs'], ids, 'cpu'), data['mask'])
    for p in model.ranker.parameters():
        p.requires_grad_(False)
    reference = deepcopy(model)
    features = model.frozen_value_features(data['features'], graph_batch(data['graphs'], ids, 'cpu'), data['mask'])
    assert not features.requires_grad
    optimizers = [torch.optim.Adam(m.value_head.parameters(), lr=.001) for m in (model, reference)]
    target = torch.tensor([3., -5., 1.])
    for _ in range(8):
        predictions = [model.value_head(features).squeeze(-1), evaluate_batch(reference, data, ids, 'cpu')[1]]
        torch.testing.assert_close(*predictions, rtol=0, atol=0)
        for m, optimizer, prediction in zip((model, reference), optimizers, predictions):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(prediction, target).backward()
            torch.nn.utils.clip_grad_norm_(m.value_head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in optimizers[0].state_dict()['state'].items():
            for name, actual in value.items():
                torch.testing.assert_close(actual, optimizers[1].state_dict()['state'][key][name], rtol=0, atol=0)


def test_contact_baseline_is_independent_of_own_reward_and_case_offset():
    from amsrr.training.request_ppo import leave_one_out_contact_returns
    values=torch.tensor([8.,8.,-5.,3.,4.,5.])
    groups=['a']*3+['b']*3
    raw,base=leave_one_out_contact_returns(values,groups)
    torch.testing.assert_close(raw,torch.tensor([6.5,6.5,-13.,-1.5,0.,1.5]))
    shifted=values+torch.tensor([100.]*3+[-200.]*3)
    torch.testing.assert_close(leave_one_out_contact_returns(shifted,groups)[0],raw)
    changed=values.clone();changed[0]=500.
    assert leave_one_out_contact_returns(changed,groups)[1][0] == base[0]
    perm=torch.tensor([5,2,0,4,1,3])
    reordered,_=leave_one_out_contact_returns(values[perm],[groups[i] for i in perm])
    torch.testing.assert_close(reordered,raw[perm])
    with pytest.raises(ValueError,match='other independent'):
        leave_one_out_contact_returns(values[:1],['alone'])


def test_equal_outcomes_give_zero_contact_signal():
    from amsrr.training.request_ppo import leave_one_out_contact_returns
    raw,_=leave_one_out_contact_returns(torch.tensor([9.,9.,9.,-5.,-5.,-5.]),['a']*3+['b']*3)
    assert torch.equal(raw,torch.zeros_like(raw))
