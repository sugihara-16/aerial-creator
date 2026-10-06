import torch
import pytest

from amsrr.policies.request_actor_critic import RequestActorCritic
from tests.unit.policies.test_high_level_requests import request_scene, decision


@pytest.mark.parametrize('morphology_aware', [False, True])
def test_replica_draws_match_independent_decisions_and_generator_states(request_scene, monkeypatch, morphology_aware):
    context = decision(request_scene)
    actor = RequestActorCritic(morphology_aware=morphology_aware).eval()
    seeds = range(371, 387)
    reference_rng = [torch.Generator().manual_seed(seed) for seed in seeds]
    reference = [actor.decide(context, generator=rng) for rng in reference_rng]
    rngs = [torch.Generator().manual_seed(seed) for seed in seeds]
    calls = []
    encode = actor.encode
    def counted(value):
        calls.append(value)
        return encode(value)
    monkeypatch.setattr(actor, 'encode', counted)
    actual = actor.decide_many(context, generators=rngs)
    assert len(calls) == 1
    for before, after, left_rng, right_rng in zip(reference, actual, reference_rng, rngs):
        for key in ['action', 'log_prob', 'value', 'snapshot_hash', 'requests', 'sampling_order']:
            assert before[key] == after[key]
        assert before['request'].to_dict() == after['request'].to_dict()
        assert torch.equal(before['logits'], after['logits'])
        assert torch.equal(left_rng.get_state(), right_rng.get_state())
    actor.decide_many(context, generators=[torch.Generator().manual_seed(371)])
    assert len(calls) == 2  # No inference cache survives the observation call.
