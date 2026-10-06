from copy import deepcopy

import pytest
import torch
from torch import nn

from tests.unit.training.test_request_q_learning import q_model
from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.request_q_policy import ActionSetQHead, CONTEXTUAL_RANKING_VERSION
from amsrr.training.request_q_learning import migrate_action_context, TRAINABLE_PREFIXES
from amsrr.training.request_ppo import serialize_encoding
from amsrr.training.request_collection import replay_log_probability_errors
from amsrr.utils.hashing import hash_file


def test_context_zero_migration_preserves_predictions_and_checkpoint(request_scene):
    scene, task, physical, state = request_scene
    task.metadata.update(estimated_mass_kg=1., estimated_com_object=[0., 0., 0.])
    model = q_model()
    model.ranked_contact = {'version': 'first_feasible_plackett_luce_v1', 'attempt_cost': .25}
    encoded = model.encode(decision(request_scene))
    before = model.evaluate_encoding(encoded)
    model.enable_candidate_context()
    torch.testing.assert_close(before, model.evaluate_encoding(encoded), rtol=0, atol=0)
    restored = RequestActorCritic.from_checkpoint(model.checkpoint()).eval()
    torch.testing.assert_close(before, restored.evaluate_encoding(encoded), rtol=0, atol=0)
    assert restored.ranked_contact['version'] == CONTEXTUAL_RANKING_VERSION
    with pytest.raises(ValueError, match='already enabled'):
        model.enable_candidate_context()


def test_remaining_actions_resolve_known_planning_chain_and_permutation():
    # All plans fail. Each attempt costs .25; exhaustion additionally costs5.
    # The same first action has exact value -5-.25*N, where N is remaining
    # candidates. A purely local action scorer cannot fit these four targets.
    torch.manual_seed(53)
    local = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 12), nn.SiLU(),
                          nn.Linear(12, 8), nn.SiLU(), nn.Linear(8, 1))
    head = ActionSetQHead(local)
    features = torch.randn(1, 4, 3).expand(4, -1, -1)
    mask = torch.arange(4)[None] < torch.arange(1, 5)[:, None]
    expected = -5. - .25 * torch.arange(1, 5)
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(400):
        predicted = head(features, mask)[:, 0, 0]
        loss = (predicted - expected).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.testing.assert_close(head(features, mask)[:, 0, 0], expected, atol=.04, rtol=0)
    order = torch.tensor([2, 0, 3, 1])
    original = head(features, mask)
    torch.testing.assert_close(head(features[:, order], mask[:, order]), original[:, order])
    padded = torch.cat((features, torch.randn(4, 3, 3)), 1)
    padded_mask = torch.cat((mask, torch.zeros(4, 3, dtype=torch.bool)), 1)
    torch.testing.assert_close(head(padded, padded_mask)[:, :4], original)


def test_conditional_ranking_replay_and_corruption_detection(request_scene):
    scene, task, physical, state = request_scene
    task.metadata.update(estimated_mass_kg=1., estimated_com_object=[0., 0., 0.])
    model = q_model(); model.ranked_contact = {'version': 'first_feasible_plackett_luce_v1', 'attempt_cost': .25}; model.enable_candidate_context()
    with torch.no_grad():
        model.q_contact.context.weight.normal_(0, .3)
    context = decision(request_scene)
    encoded = model.encode(context)
    model.ranking_prefix_length = int(encoded['mask'].sum())
    assert model.ranking_prefix_length > 1
    selected = model.decide(context, generator=torch.Generator().manual_seed(9))
    repeated = model.decide(context, generator=torch.Generator().manual_seed(9))
    assert repeated['ranking_order'] == selected['ranking_order']
    mask = encoded['mask'].clone(); expected = 0.
    for action in selected['ranking_prefix']:
        with torch.no_grad():
            logits, _ = model.evaluate_encoding(dict(encoded, mask=mask))
        expected += float(logits[0].log_softmax(-1)[action])
        mask = mask.clone(); mask[action] = False
    assert selected['log_prob'] == pytest.approx(expected, abs=2e-6)
    event = dict(encoding=serialize_encoding(encoded), action=selected['action'],
                 log_prob=selected['log_prob'], ranking_prefix=selected['ranking_prefix'],
                 ranking_policy_version=CONTEXTUAL_RANKING_VERSION)
    assert replay_log_probability_errors(model, [event, event]).max() < 2e-5
    bad = dict(event, log_prob=event['log_prob']+.1)
    assert replay_log_probability_errors(model, [bad])[0] > .09
    with pytest.raises(ValueError, match='contract mismatch'):
        replay_log_probability_errors(model, [dict(event, ranking_policy_version='old')])
    with pytest.raises(ValueError, match='repeated'):
        model.ranking_log_probability(event['encoding'], [0, 0])
    assert torch.equal(encoded['mask'], selected['encoded']['mask'])
    greedy = model.decide(context, deterministic=True)
    requests, scores = model.rank_with_scores(context)
    assert [r.to_dict() for r in requests] == [greedy['requests'][i] for i in greedy['ranking_order']]
    assert scores == greedy['ranking_conditional_log_probs']


def test_equal_count_different_alternatives_can_have_different_values():
    # Trying a known rejected plan0 costs .25. If plan1 remains, it succeeds
    # for reward10; if only rejected plan2 remains, exhaustion costs5.
    # Both states have two candidates: cardinality alone is insufficient.
    torch.manual_seed(29)
    head = ActionSetQHead(nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 12), nn.SiLU(),
        nn.Linear(12, 8), nn.SiLU(), nn.Linear(8, 1)))
    features = torch.eye(3)[None].expand(2, -1, -1)
    masks = torch.tensor([[True, True, False], [True, False, True]])
    expected = torch.tensor([9.5, -5.5])
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(250):
        loss = (head(features, masks)[:, 0, 0] - expected).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.testing.assert_close(head(features, masks)[:, 0, 0], expected, atol=.03, rtol=0)


def test_optimizer_migration_preserves_old_moments_and_target_lag(tmp_path):
    model = q_model(); model.ranked_contact = {'version': 'first_feasible_plackett_luce_v1', 'attempt_cost': .25}
    named = [(n, p) for n, p in model.named_parameters() if n.startswith(TRAINABLE_PREFIXES)]
    optimizer = torch.optim.Adam([p for _, p in named], lr=1e-4)
    for _, p in named:
        p.grad = torch.ones_like(p)
    optimizer.step()
    target = deepcopy(model)
    with torch.no_grad():
        for p in target.parameters():
            p.add_(.03)
    source = tmp_path/'source'; source.mkdir()
    checkpoint = source/'checkpoint.pt'; torch.save(model.checkpoint(), checkpoint)
    old = dict(checkpoint_sha256=hash_file(checkpoint), optimizer=optimizer.state_dict(),
               target=target.state_dict(), steps=2000, learning_rate=1e-4, target_tau=.005,
               mode='double_q', conservative_weight=1., regression_contract='squared_bellman_huber_initialization_v1')
    torch.save(old, source/'q_training_state.pt')
    output = tmp_path/'migrated'
    report = migrate_action_context(checkpoint, output)
    migrated = RequestActorCritic.load(output/'checkpoint.pt')
    saved = torch.load(output/'q_training_state.pt', weights_only=True)
    new = [(n, p) for n, p in migrated.named_parameters() if n.startswith(TRAINABLE_PREFIXES)]
    loaded_optimizer = torch.optim.Adam([p for _, p in new], lr=1e-4)
    loaded_optimizer.load_state_dict(saved['optimizer'])
    old_moments = {n: optimizer.state[p] for n, p in named}
    for name, p in new:
        if name in old_moments:
            for key, value in old_moments[name].items():
                torch.testing.assert_close(value, loaded_optimizer.state[p][key], rtol=0, atol=0)
            torch.testing.assert_close(saved['target'][name], old['target'][name], rtol=0, atol=0)
        else:
            assert not loaded_optimizer.state[p] and torch.count_nonzero(p) == 0
            assert torch.count_nonzero(saved['target'][name]) == 0
    assert len(report['added_parameters']) == 2 and saved['steps'] == 2000
