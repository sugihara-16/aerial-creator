from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from amsrr.policies.anchor_preference import (
    AnchorPreference, assigned_anchor_distance, request_anchor_distances,
)
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler, ContactCandidateSamplerConfig
from amsrr.robot_model.gripper_surfaces import with_neighbor_grasp_anchors, free_dock_boundary_cycles
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.training.request_ppo import (
    serialize_encoding, collate_transitions, evaluate_batch, anchor_preference_loss,
    configure_anchor_preference, ppo_update, request_runtime_contracts,
)
from amsrr.utils.hashing import hash_file
from tests.unit.policies.test_high_level_requests import request_scene, decision


@pytest.fixture
def neighbor_context(request_scene):
    scene, task, physical, execution = deepcopy(request_scene)
    morphology = with_neighbor_grasp_anchors(scene.morphology_graph, physical,
        slot_ids=[0], max_force_n=30., max_torque_nm=4.)
    candidates = ContactCandidateSampler(ContactCandidateSamplerConfig(
        max_grasp_contacts=2, grasp_neighborhoods=True)).sample(task_spec=task,
        irg=scene.irg, interaction_envelope=scene.interaction_envelope,
        morphology_graph=morphology, geometry_descriptors=IRGBuilder().build_with_scene_graph(task).scene_graph.geometry_descriptors)
    observation = deepcopy(scene.runtime_observation)
    observation.morphology_graph = morphology
    return decision((replace(scene, morphology_graph=morphology,
        contact_candidate_set=candidates, runtime_observation=observation), task, physical, execution))


def test_distance_uses_distinct_upstream_groups_and_not_independent_nearest_centers():
    distances = {10: {'0': 0, '1': 2}, 11: {'0': 1, '1': 2}, 12: {'1': 0}}
    assert assigned_anchor_distance((10, 12), distances) == 0
    assert assigned_anchor_distance((11, 12), distances) == 1
    assert assigned_anchor_distance((10, 11), distances) == 2  # Not 0 + 1.
    assert assigned_anchor_distance((11, 10), distances) == 2
    assert assigned_anchor_distance((4, 2, 9), {4:{'a':0}, 2:{'b':1}, 9:{'c':2}}) == 3
    with pytest.raises(ValueError, match='distinct'):
        assigned_anchor_distance((10, 10), distances)
    with pytest.raises(ValueError, match='cover'):
        assigned_anchor_distance((10, 11), {10:{'a':0},11:{'a':1},12:{'b':0}})


def test_generated_distances_are_measured_from_original_centers(request_scene):
    scene, _, physical, _ = request_scene
    source = scene.morphology_graph
    result = with_neighbor_grasp_anchors(source, physical, slot_ids=[0], max_force_n=30., max_torque_nm=4.)
    cycles = free_dock_boundary_cycles(source, physical)
    centers = [a for a in source.robot_anchors if a.anchor_type == 'grasp']
    for i, center in enumerate(centers):
        port = center.capability.get('surface_port_id', center.capability.get('dock_port_global_id'))
        cycle = next(c for c in cycles if port in c)
        actual = {a.capability['surface_port_id']:a.capability['grasp_neighborhood_distances'][str(i)]
                  for a in result.robot_anchors if str(i) in a.capability.get('grasp_neighborhood_distances', {})}
        expected = {cycle[(cycle.index(port)+delta)%len(cycle)]: abs(delta) for delta in (-2,-1,0,1,2)}
        assert actual == expected
    assert all('grasp_neighborhood_distances' not in a.capability for a in source.robot_anchors)


@pytest.mark.parametrize('field,value', [('distance_decay',0),('distance_decay',float('nan')),
    ('kl_coefficient',-1),('contract','unknown')])
def test_invalid_preference_fails_closed(field, value):
    with pytest.raises(ValueError):
        AnchorPreference(**{field:value})


def test_prior_has_full_support_and_halves_odds_per_boundary_step():
    prior = AnchorPreference()
    p = prior.logits(torch.tensor([[0.,1.,2.,4.,0.]]), torch.tensor([[True]*4+[False]])).softmax(-1)[0]
    assert p[0]/p[1] == pytest.approx(2)
    assert p[1]/p[2] == pytest.approx(2)
    assert p[0]/p[3] == pytest.approx(16)
    assert (p[:4]>0).all() and p[4] == 0
    # A disallowed center does not remove the neighboring alternatives.
    no_center = prior.logits(torch.tensor([[0.,1.,2.]]),torch.tensor([[False,True,True]])).softmax(-1)
    torch.testing.assert_close(no_center,torch.tensor([[0.,2/3,1/3]]))


def test_center_preference_persists_in_inference_replay_and_reload(neighbor_context):
    model = RequestActorCritic(morphology_aware=True).eval()
    with torch.no_grad():
        model.ranker.contact_head[-1].weight.zero_()
        model.ranker.contact_head[-1].bias.zero_()
    old = RequestActorCritic.from_checkpoint(model.checkpoint()).eval()
    model.enable_anchor_preference()
    result = model.decide(neighbor_context, deterministic=True)
    enc = result['encoded']; distance = enc['anchor_distances']; mask = enc['mask']
    assert distance[result['action']] == 0 and distance[mask].max() > 0
    before = model.evaluate_encoding(enc)
    expected = model.anchor_preference.logits(distance, mask)
    torch.testing.assert_close(before[0][0], expected)
    restored = RequestActorCritic.from_checkpoint(model.checkpoint()).eval()
    data = collate_transitions([dict(encoding=serialize_encoding(enc))])
    replay = evaluate_batch(restored, data, torch.tensor([0]), 'cpu')
    torch.testing.assert_close(before, replay, rtol=0, atol=0)
    actual_log = torch.distributions.Categorical(logits=replay[0][0]).log_prob(torch.tensor(result['action']))
    assert abs(actual_log.item()-result['log_prob']) < 2e-5
    # Same temporal input has identical scores and critic before/after migration.
    temporal = dict(enc, initial=torch.tensor(False))
    torch.testing.assert_close(old.evaluate_encoding(temporal), model.evaluate_encoding(temporal), rtol=0, atol=0)
    with pytest.raises(ValueError, match='distances'):
        restored.evaluate_encoding({k:v for k,v in enc.items() if k!='anchor_distances'})


def test_task_evidence_can_override_prior_and_select_a_peripheral_dock(neighbor_context):
    model = RequestActorCritic().eval();model.enable_anchor_preference()
    encoded = model.encode(neighbor_context)
    distance = encoded['anchor_distances']; mask = encoded['mask']
    target = int(distance.masked_fill(~mask,-1).argmax())
    assert distance[target] > 0
    class KnownTaskScores(torch.nn.Module):
        def forward(self, group):
            scores = torch.zeros((*group.shape[:2],1))
            scores[:,target,0] = model.anchor_preference.distance_decay * distance[target] + 1.
            return scores
    model.ranker.contact_head = KnownTaskScores()
    assert model.decide(neighbor_context, deterministic=True)['action'] == target


def test_prior_regularizer_changes_gradients_and_never_penalizes_temporal_events():
    model = RequestActorCritic();model.enable_anchor_preference()
    logits = torch.tensor([[0.,0.,0.,-torch.inf],[1.,2.,3.,4.]],requires_grad=True)
    mask = torch.isfinite(logits)
    data = dict(mask=mask, anchor_distances=torch.tensor([[0.,1.,2.,0.],[0.,0.,0.,0.]]))
    loss = anchor_preference_loss(model, logits, data, torch.tensor([True,False]))
    loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,2]>0  # Descent favors the center.
    assert torch.equal(logits.grad[1],torch.zeros(4)) and torch.isfinite(logits.grad).all()
    prior = model.anchor_preference.logits(data['anchor_distances'],mask)
    assert abs(anchor_preference_loss(model,prior,data,torch.tensor([True,False])).item()) < 1e-12


def test_migration_preserves_parameters_and_optimizer_and_rejects_stale_rollouts(tmp_path):
    model = RequestActorCritic(morphology_aware=True)
    path=tmp_path/'old';path.mkdir();torch.save(model.checkpoint(),path/'checkpoint.pt')
    state=dict(profile='timed_value_v1',checkpoint_sha256=hash_file(path/'checkpoint.pt'),completed_updates=22,
        actor_optimizer={'test':torch.tensor([1.,2.])}, value_optimizer={'test':torch.tensor([3.])})
    torch.save(state,path/'training_state.pt')
    configure_anchor_preference(path/'checkpoint.pt',tmp_path/'new')
    new=RequestActorCritic.load(tmp_path/'new/checkpoint.pt')
    torch.testing.assert_close(model.state_dict(),new.state_dict(),rtol=0,atol=0)
    saved=torch.load(tmp_path/'new/training_state.pt',weights_only=True)
    assert saved['completed_updates']==22 and saved['anchor_preference']==new.anchor_preference.to_dict()
    torch.testing.assert_close(saved['actor_optimizer'],state['actor_optimizer'])
    stale=tmp_path/'stale.pt';torch.save(dict(runtime_contracts=request_runtime_contracts(),
        checkpoint_sha256=hash_file(path/'checkpoint.pt'),sampling='categorical',teacher_phase_supervision=False,
        complete=True,episodes=[]),stale)
    with pytest.raises(ValueError,match='on-policy'):
        ppo_update(tmp_path/'new/checkpoint.pt',[stale],tmp_path/'bad')


def test_actual_ppo_update_keeps_prior_and_learns_peripheral_success(neighbor_context, tmp_path):
    """The real optimizer can trade prior preference for better task outcomes."""
    torch.manual_seed(19)
    model = RequestActorCritic(morphology_aware=True).eval()
    model.enable_anchor_preference(AnchorPreference(kl_coefficient=.05))
    start = tmp_path/'start';start.mkdir();checkpoint=start/'checkpoint.pt'
    torch.save(model.checkpoint(),checkpoint)
    encoded = model.encode(neighbor_context)
    distances = encoded['anchor_distances'];mask=encoded['mask']
    def probabilities(actor):
        scores,_=actor.evaluate_encoding(encoded)
        return scores[0].softmax(-1).detach()
    before = probabilities(model)
    events=[]
    for seed in range(96):
        draw=model.decide(neighbor_context,generator=torch.Generator().manual_seed(seed))
        # A controlled task where the distant choice, not center, succeeds.
        reward=5. if distances[draw['action']]>=2 else -5.
        events.append([dict(encoding=serialize_encoding(draw['encoded']),action=draw['action'],
            value=draw['value'],log_prob=draw['log_prob'],reward=reward,done=True,
            time_s=0.,end_time_s=0.,reward_timing='observed_reward_time_v1',reward_events=[(0.,reward)])])
    assert {e[0]['reward'] for e in events}=={-5.,5.}
    rollout=tmp_path/'rollout.pt';torch.save(dict(checkpoint_sha256=hash_file(checkpoint),
        sampling='categorical',teacher_phase_supervision=False,complete=True,
        runtime_contracts=request_runtime_contracts(),episodes=events),rollout)
    report=ppo_update(checkpoint,[rollout],tmp_path/'updated',epochs=8,
        learning_rate=.001,entropy_coefficient=0.,training_profile='timed_mc_v1',value_epochs=2)
    updated=RequestActorCritic.load(tmp_path/'updated/checkpoint.pt').eval()
    after=probabilities(updated)
    assert after[mask & (distances>=2)].sum()>before[mask & (distances>=2)].sum()
    assert updated.anchor_preference==model.anchor_preference
    assert report['optimization']['anchor_preference']==model.anchor_preference.to_dict()
    assert report['replay_log_probability_max_error']<2e-5 and report['reload_exact']
    assert all(x['anchor_preference_loss']>=-1e-8 for x in report['updates'])
    assert report['anchor_preference_diagnostics']['after']['expected_total_steps'] > 0
    state=torch.load(tmp_path/'updated/training_state.pt',weights_only=True)
    assert state['anchor_preference']==model.anchor_preference.to_dict()


def test_regularization_repeatedly_returns_equal_reward_policy_to_fixed_prior():
    # Isolate the fixed preference from task gradients; unlike initialization,
    # its gradient continues to restore center preference after parameter drift.
    model=RequestActorCritic();model.enable_anchor_preference()
    distance=torch.tensor([[0.,1.,2.]])
    data=dict(mask=torch.ones_like(distance,dtype=torch.bool),anchor_distances=distance)
    logits=torch.nn.Parameter(torch.tensor([[-2.,0.,2.]]))
    optimizer=torch.optim.Adam([logits],lr=.1)
    for _ in range(160):
        optimizer.zero_grad();loss=anchor_preference_loss(model,logits,data,torch.tensor([True]))
        loss.backward();optimizer.step()
    expected=torch.tensor([[4/7,2/7,1/7]])
    torch.testing.assert_close(logits.softmax(-1),expected,atol=.001,rtol=0)


def test_collection_reports_actual_selected_distances_separately_from_reward():
    from amsrr.training.request_collection import aggregate
    rows=[dict(passed=True,safety_failure=False,reward=7.,anchor_distance=0.),
          dict(passed=False,safety_failure=False,reward=-5.,anchor_distance=3.)]
    report=aggregate(rows)
    assert report['center_selections']==1 and report['mean_anchor_distance']==1.5
    assert report['mean_reward']==1. and report['successes']==1
    with pytest.raises(ValueError,match='mixed'):
        aggregate([rows[0],dict(passed=False,safety_failure=False,reward=-5.)])
