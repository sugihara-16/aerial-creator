from dataclasses import replace
from copy import deepcopy
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
)
from amsrr.utils.hashing import hash_file


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
        complete=True,
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
    ]:
        bad = {**rollout, key: value}
        torch.save(bad, path)
        with pytest.raises(ValueError, match="on-policy"):
            ppo_update(checkpoint, [path], tmp_path / key)


def test_initial_input_geometry_changes_logits_but_label_has_no_channel(request_scene):
    model = RequestActorCritic()
    context = decision(request_scene)
    enc = model.encode(context)
    first, _ = model.evaluate_encoding(enc)
    modified = deepcopy(enc)
    modified["candidates"][0, 0] += 0.2
    second, _ = model.evaluate_encoding(modified)
    assert not torch.allclose(first, second)


def test_event_discount_depends_on_elapsed_time_not_number_of_decisions():
    direct = [{"time_s": 0.0, "reward": 0.0}, {"time_s": 10.0, "reward": 1.0}]
    segmented = [direct[0], {"time_s": 4.0, "reward": 0.0}, direct[1]]
    assert discounted_event_returns(direct, 0.9)[0] == pytest.approx(0.9**10)
    assert discounted_event_returns(segmented, 0.9)[0] == pytest.approx(0.9**10)
    assert discounted_event_returns([{"reward": -5.0}], 0.9) == [-5.0]


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
