from copy import deepcopy
import pytest
import torch

from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.request_q_policy import RequestQPolicy, clock_features
from amsrr.training.request_ppo import serialize_encoding, collate_transitions, evaluate_batch
from amsrr.training.request_q_learning import episode_transitions, double_q_target, conservative_q_loss, q_regression_loss


def q_model():
    actor = RequestActorCritic(morphology_aware=True, object_condition_aware=True).eval()
    actor.enable_temporal_options()
    actor.enable_motion_feedback()
    actor.enable_temporal_object_conditions()
    return RequestQPolicy(actor).eval()


def test_replay_rejects_mixed_physical_controllers_and_missing_contract(tmp_path):
    from amsrr.training.request_q_learning import load_compatible_replay
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    payload = dict(version='request_physical_q_replay_v1', split='train',
                   runtime_contracts={'nominal_body_tracking': 'old'}, transitions=[{'action': 0}])
    torch.save(payload, first)
    torch.save(payload, second)
    rows, sources, contract = load_compatible_replay([first, second])
    assert len(rows) == len(sources) == 2 and contract == payload['runtime_contracts']
    torch.save(dict(payload, runtime_contracts={'nominal_body_tracking': 'new'}), second)
    with pytest.raises(ValueError, match='mixes physical execution contracts'):
        load_compatible_replay([first, second])
    torch.save({k: v for k, v in payload.items() if k != 'runtime_contracts'}, second)
    with pytest.raises(ValueError, match='lacks physical execution contracts'):
        load_compatible_replay([second])


def test_q_causal_clock_roundtrip_and_variable_actions(request_scene):
    scene, task, physical, state = request_scene
    task.metadata.update(estimated_mass_kg=1., estimated_com_object=[0., 0., 0.])
    context = decision((scene, task, physical, state))
    model = q_model()
    encoded = model.encode(context)
    q, _ = model.q_values(encoded['features'][None], encoded['graph'], encoded['mask'][None],
        candidates=encoded['candidates'][None], membership=encoded['membership'][None],
        initial=encoded['initial'][None], owners=encoded['owners'][None],
        object_features=encoded['object_features'][None], motion_features=encoded['motion_features'][None],
        hold_indices=encoded['hold_indices'][None], clock_features=encoded['clock_features'][None])
    logits, _ = model.evaluate_encoding(encoded)
    assert int(q.argmax()) == int(logits.argmax())
    probability = logits.softmax(-1)[0]
    assert (probability[encoded['mask']] >= .2/encoded['mask'].sum()-1e-7).all()
    saved = serialize_encoding(encoded)
    batch = collate_transitions([dict(encoding=saved)])
    torch.testing.assert_close(model.evaluate_encoding(encoded), evaluate_batch(model, batch, torch.arange(1), 'cpu'))
    restored = RequestActorCritic.from_checkpoint(model.checkpoint()).eval()
    torch.testing.assert_close(model.evaluate_encoding(encoded), restored.evaluate_encoding(encoded), rtol=0, atol=0)
    shifted = deepcopy(encoded)
    shifted['clock_features'] = clock_features(100., [150.])
    assert not torch.equal(model.evaluate_encoding(encoded)[0], model.evaluate_encoding(shifted)[0])


def test_double_q_selects_online_but_values_target_and_stops_at_terminal():
    online = torch.tensor([[2., 1.], [2., 1.]])
    target = torch.tensor([[3., 9.], [3., 9.]])
    result = double_q_target(torch.tensor([1., 7.]), torch.tensor([.5, 0.]), online, target)
    torch.testing.assert_close(result, torch.tensor([2.5, 7.]))


def test_q_checkpoint_cannot_silently_enter_the_ppo_optimizer(tmp_path):
    from amsrr.training.request_ppo import ppo_update
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(q_model().checkpoint(), checkpoint)
    output = tmp_path / 'wrong_optimizer'
    with pytest.raises(ValueError, match='Q policy checkpoints require'):
        ppo_update(checkpoint, [], output)
    assert not output.exists()


def test_bellman_updates_reject_a_reward_free_waiting_loop():
    # Exact control problem: advance earns ten and ends; holding leaves the
    # state unchanged and costs time. The optimal greedy action must advance,
    # even if waiting was initially overestimated.
    q = torch.tensor([[9., 12.]])
    for _ in range(2000):
        next_values = q.expand(2, -1)
        q = double_q_target(torch.tensor([10., -.001]), torch.tensor([0., .995]),
                            next_values, next_values).reshape(1, 2)
    assert int(q.argmax()) == 0
    torch.testing.assert_close(q, torch.tensor([[10., 9.949]]), atol=1e-5, rtol=0)


def test_conservative_objective_lowers_unobserved_overestimates_and_excludes_mask():
    q = torch.tensor([[1., 5., -torch.inf]], requires_grad=True)
    conservative_q_loss(q, torch.tensor([0])).backward()
    assert q.grad[0, 0] < 0 and q.grad[0, 1] > 0 and q.grad[0, 2] == 0
    assert conservative_q_loss(torch.tensor([[1., -torch.inf]]), torch.tensor([0])) == 0


def test_observed_rare_success_can_overcome_the_conservative_frequency_prior():
    # A simple known control problem: four actions failed (return -1), one
    # succeeded (return 5). At equal values the successful action must rise.
    # Huber+CQL instead pushes it down because its TD derivative saturates.
    actions = torch.tensor([0, 0, 0, 0, 1])
    targets = torch.tensor([-1., -1., -1., -1., 5.])
    q = torch.tensor([[0., 0.]], requires_grad=True)
    values = q.expand(5, -1)
    selected = values.gather(-1, actions[:, None]).squeeze(-1)
    loss = q_regression_loss(selected, targets, 'double_q') + conservative_q_loss(values, actions)
    loss.backward()
    assert q.grad[0, 0] > 0 and q.grad[0, 1] < 0


def test_planner_prefix_replay_preserves_reward_and_only_removes_failed_choices():
    enc = dict(mask=torch.ones(3, dtype=torch.bool), initial=torch.tensor(True))
    event = dict(encoding=enc, ranking_prefix=[2, 0, 1], action=1, planning_attempts=3,
        planning_cost=.75, time_s=0., end_time_s=2., done=True, reward=4.25,
        reward_timing='observed_reward_time_v1', reward_events=[(0., -.75), (2., 5.)])
    rows = episode_transitions([event], deadlines=[150.], gamma=.9)
    assert len(rows) == 3
    assert [r['action'] for r in rows] == [2, 0, 1]
    assert rows[0]['next_encoding']['mask'].tolist() == [True, True, False]
    assert rows[1]['next_encoding']['mask'].tolist() == [False, True, False]
    assert rows[-1]['next_encoding'] is None and rows[-1]['discount'] == 0
    assert rows[0]['mc_return'] == pytest.approx(-.75 + .9**2 * 5.)
    assert enc['mask'].all()
    event['planning_cost'] = 0.
    with pytest.raises(ValueError, match='charge mismatch'):
        episode_transitions([event], deadlines=[150.])
