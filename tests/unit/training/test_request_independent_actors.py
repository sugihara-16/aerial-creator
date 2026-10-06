"""Independent policy copies must isolate behavior, not only head weights."""
from copy import deepcopy
import pytest
import torch

from tests.unit.policies.test_high_level_requests import request_scene, decision
from tests.unit.training.test_request_learning_signals import committed
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.training.request_ppo import ppo_update, serialize_encoding, request_runtime_contracts
from amsrr.utils.hashing import hash_file


@pytest.mark.parametrize('branch', ['contact', 'temporal'])
def test_shared_encoder_changes_cannot_move_fixed_policy(request_scene, branch):
    torch.manual_seed(42)
    reference = RequestActorCritic().eval()
    reference.enable_temporal_options()
    learner = RequestActorCritic.from_checkpoint(reference.checkpoint()).eval()
    learner.enable_fixed_branch(reference, trainable_branch=branch)
    contexts = [decision(request_scene), committed(request_scene)]
    encodings = [learner.encode(c) for c in contexts]
    before = [reference.evaluate_encoding(e)[0] for e in encodings]
    for e, logits in zip(encodings, before):
        torch.testing.assert_close(learner.evaluate_encoding(e)[0], logits, rtol=0, atol=0)
    # Deliberately move shared representations and both heads of the learner.
    with torch.no_grad():
        for p in learner.ranker.parameters():
            p.add_(torch.randn_like(p) * .03)
    frozen_index = 1 if branch == 'contact' else 0
    active_index = 1 - frozen_index
    after = [learner.evaluate_encoding(e)[0] for e in encodings]
    torch.testing.assert_close(after[frozen_index], before[frozen_index], rtol=0, atol=0)
    assert not torch.allclose(after[active_index], before[active_index])
    restored = RequestActorCritic.from_checkpoint(learner.checkpoint()).eval()
    for e, expected in zip(encodings, after):
        torch.testing.assert_close(restored.evaluate_encoding(e)[0], expected, rtol=0, atol=0)
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(restored.fixed_actor.state_dict()[name], tensor, rtol=0, atol=0)
    assert all(not p.requires_grad for p in restored.fixed_actor.parameters())


@pytest.mark.parametrize('branch', ['contact', 'temporal'])
def test_three_fresh_ppo_updates_keep_reference_and_continue_adam(request_scene, tmp_path, branch):
    torch.manual_seed(8)
    reference = RequestActorCritic().eval()
    reference.enable_temporal_options()
    learner = RequestActorCritic.from_checkpoint(reference.checkpoint()).eval()
    learner.enable_fixed_branch(reference, trainable_branch=branch)
    checkpoint = tmp_path/'initial.pt'
    torch.save(learner.checkpoint(), checkpoint)
    contexts = [decision(request_scene), committed(request_scene)]
    reference_state = deepcopy(reference.state_dict())
    active_index = 0 if branch == 'contact' else 1
    fixed_index = 1 - active_index
    desired = int(torch.nonzero(learner.encode(contexts[active_index])['mask'])[0])
    initial_probability = float(learner.decide(contexts[active_index])['logits'].softmax(-1)[desired])
    for update in range(3):
        learner = RequestActorCritic.load(checkpoint).eval()
        episodes = []
        for seed in range(48):
            generator = torch.Generator().manual_seed(1000*update + seed)
            draws = [learner.decide(c, generator=generator) for c in contexts]
            reward = 1. if draws[active_index]['action'] == desired else -1.
            trajectory = []
            for i, draw in enumerate(draws):
                amount = 0. if i == 0 else reward
                trajectory.append(dict(encoding=serialize_encoding(draw['encoded']),
                    action=draw['action'], log_prob=draw['log_prob'], value=draw['value'],
                    hold_duration_s=draw['hold_duration_s'],
                    time_s=float(i), end_time_s=float(i+1), reward=amount,
                    reward_events=[(float(i+1), amount)], reward_timing='observed_reward_time_v1',
                    done=i == 1))
            episodes.append(trajectory)
        path = tmp_path/f'rollout_{update}.pt'
        torch.save(dict(checkpoint_sha256=hash_file(checkpoint), sampling='categorical',
                        teacher_phase_supervision=False, complete=True,
                        runtime_contracts=request_runtime_contracts(), episodes=episodes), path)
        report = ppo_update(checkpoint, [path], tmp_path/f'update_{update}',
            training_profile='timed_value_v1', epochs=8, learning_rate=.001,
            value_epochs=3, target_kl=.05, gae_lambda=None if update == 0 else .95)
        assert report['eligible_actor_events'] == 48
        assert report['optimization']['optimizer_updates_resumed'] == update
        assert report['optimization']['immutable_actor_unchanged']
        checkpoint = tmp_path/f'update_{update}/checkpoint.pt'
        result = RequestActorCritic.load(checkpoint).eval()
        for key, value in result.fixed_actor.state_dict().items():
            torch.testing.assert_close(value, reference_state[key], rtol=0, atol=0)
        encoding = result.encode(contexts[fixed_index])
        torch.testing.assert_close(result.evaluate_encoding(encoding)[0],
                                   reference.evaluate_encoding(encoding)[0], rtol=0, atol=0)
    probability = float(result.decide(contexts[active_index])['logits'].softmax(-1)[desired])
    assert probability > initial_probability
