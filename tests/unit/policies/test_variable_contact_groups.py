from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler, ContactCandidateSamplerConfig
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.robot_model.gripper_surfaces import with_available_grasp_anchors
from amsrr.training.request_ppo import serialize_encoding, collate_transitions, evaluate_batch


def test_contact_demonstration_choice_does_not_change_predecision_inputs(request_scene):
    from amsrr.policies.request_actor_critic import InitialContactTeacher
    from amsrr.simulation.request_event_execution import validate_initial_encoding
    context=decision(request_scene)
    model=RequestActorCritic().cpu().eval()
    groups=list(dict.fromkeys(e.request.contact_group_id for e in context.catalog.entries
        if e.request.contact_group_id is not None and e.request.transition_id is None))
    assert len(groups)>=2
    outputs=[InitialContactTeacher(model,g).cpu().eval().decide(context,deterministic=True) for g in groups[:2]]
    assert outputs[0]['request'].contact_group_id != outputs[1]['request'].contact_group_id
    validate_initial_encoding(serialize_encoding(outputs[0]['encoded']),serialize_encoding(outputs[1]['encoded']))


@pytest.mark.parametrize('maximum', [2,3,4])
def test_variable_groups_have_bounded_distinct_members_and_ppo_gradients(request_scene, maximum):
    scene, task, physical, execution = request_scene
    morphology = with_available_grasp_anchors(scene.morphology_graph, physical,
        slot_ids=[0], max_force_n=30., max_torque_nm=5.)
    built = IRGBuilder().build_with_scene_graph(task)
    candidates = ContactCandidateSampler(ContactCandidateSamplerConfig(max_grasp_contacts=maximum)).sample(
        task_spec=task, irg=scene.irg, interaction_envelope=scene.interaction_envelope,
        morphology_graph=morphology, geometry_descriptors=built.scene_graph.geometry_descriptors)
    groups = [g for g in candidates.group_proposals if g.group_type in ('grasp_pair','multi_grasp')]
    assert len(groups) <= 8
    assert {len(g.candidate_ids) for g in groups} == set(range(2,maximum+1))
    by_id = {c.candidate_id:c for c in candidates.candidates}
    for g in groups:
        assert len({by_id[i].anchor_id for i in g.candidate_ids}) == len(g.candidate_ids)
    observation=deepcopy(scene.runtime_observation)
    observation.morphology_graph=morphology
    context=decision((replace(scene,morphology_graph=morphology,
        contact_candidate_set=candidates,runtime_observation=observation),task,physical,execution))
    actor=RequestActorCritic(morphology_aware=True)
    encoded=actor.encode(context)
    count=encoded['membership'].sum(-1)
    indices=torch.where((count==maximum)&encoded['mask'])[0]
    assert indices.numel() > 0
    scores, _ = actor.evaluate_encoding(encoded)
    target=indices[0]
    loss=-torch.log_softmax(scores,dim=-1)[0,target]
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.ranker.contact_head.parameters())
    batch=collate_transitions([{'encoding':serialize_encoding(encoded)}])
    replay, _ = evaluate_batch(actor,batch,torch.tensor([0]),'cpu')
    torch.testing.assert_close(replay,scores)


def test_two_virtual_anchors_on_same_physical_surface_conflict():
    from amsrr.policies.contact_candidate_set import _candidates_conflict
    from types import SimpleNamespace
    left=SimpleNamespace(candidate_id=1,anchor_id=2,slot_id=0,candidate_scores={'surface_port_id':7})
    right=SimpleNamespace(candidate_id=3,anchor_id=4,slot_id=1,candidate_scores={'surface_port_id':7})
    assert _candidates_conflict(left,right)


def test_contact_expansion_preserves_preceding_object_condition_observation(request_scene, monkeypatch):
    from dataclasses import dataclass
    from amsrr.training import request_object_conditions as conditions
    scene,task,physical,_=request_scene
    @dataclass
    class Bundle:
        morphology: object
        contact_candidate_set: object
    observed=deepcopy(scene.runtime_observation)
    state=observed.object_states[0]
    state.pose_world=tuple(v+(.01 if i==2 else 0.) for i,v in enumerate(state.pose_world))
    expanded_scene=replace(scene,runtime_observation=observed)
    episode=dict(task=task,bundle=Bundle(scene.morphology_graph,scene.contact_candidate_set),
        initial_scene=expanded_scene)
    def forbidden(*args,**kwargs):
        raise AssertionError('must not reload stale donor observations')
    monkeypatch.setattr(conditions,'decision_context',forbidden)
    expanded,context=conditions.expand_episode_grasp_contacts(episode,physical)
    assert context.observation.object_states[0].pose_world==state.pose_world
    assert scene.runtime_observation.object_states[0].pose_world!=state.pose_world


def test_defined_anchors_preserve_upstream_geometry_and_expanded_observation(request_scene, monkeypatch):
    from amsrr.training import request_object_conditions as conditions
    from amsrr.utils.hashing import stable_hash
    scene, task, physical, execution = request_scene
    observed = deepcopy(scene.runtime_observation)
    state = observed.object_states[0]
    state.pose_world = tuple(v + (.01 if i == 2 else 0.) for i, v in enumerate(state.pose_world))
    scene = replace(scene, runtime_observation=observed)
    episode = dict(initial_scene=scene, task=task, phase_ids={'approach': execution.phase_id})
    before = stable_hash(episode)
    def forbidden(*args, **kwargs):
        raise AssertionError('defined anchors must not be expanded or reset to donor observation')
    monkeypatch.setattr(conditions, 'decision_context', forbidden)
    monkeypatch.setattr('amsrr.robot_model.gripper_surfaces.with_available_grasp_anchors', forbidden)
    result, context = conditions.expand_episode_grasp_contacts(episode, physical, max_grasp_contacts=None)
    assert result is episode and stable_hash(episode) == before
    assert context.scene.morphology_graph.robot_anchors == scene.morphology_graph.robot_anchors
    assert context.observation.object_states[0].pose_world == state.pose_world


def test_anchor_source_is_separate_from_contact_cardinality():
    from amsrr.training.request_object_conditions import grasp_contact_expansion_limit
    assert grasp_contact_expansion_limit({'grasp_anchor_source': 'defined', 'max_grasp_contacts': 2}) is None
    assert grasp_contact_expansion_limit({'grasp_anchor_source': 'all_free', 'max_grasp_contacts': 2}) == 2
    assert grasp_contact_expansion_limit({}) == 2  # Existing experiment compatibility.
    with pytest.raises(ValueError, match='unknown grasp_anchor_source'):
        grasp_contact_expansion_limit({'grasp_anchor_source': 'typo'})


def test_native_three_contact_ik_matches_independent_fk_with_shuffled_assignments(request_scene):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from amsrr.feasibility.articulated_reachability import (
        ArticulatedContactIKSolver, ArticulatedIKConfig, resolve_mesh_backed_anchor_references)
    from amsrr.robot_model.whole_structure_kinematics import WholeStructureKinematics, ordered_global_dock_joint_ids
    from amsrr.schemas.policies import ContactAssignment
    from amsrr.schemas.common import ContactMode
    scene,_,physical,_=request_scene
    morphology=with_available_grasp_anchors(scene.morphology_graph,physical,
        slot_ids=[0],max_force_n=30.,max_torque_nm=5.)
    owners={a.module_id:a for a in morphology.robot_anchors if a.anchor_type=='grasp'}
    anchors=list(owners.values())[:3]
    assert len(anchors)==3
    refs=resolve_mesh_backed_anchor_references(morphology,physical,[a.anchor_id for a in anchors])
    ids=ordered_global_dock_joint_ids(morphology,physical)
    q={j:.1 for j in ids}
    base=(.5,.2,1.0,0.,0.,0.,1.)
    independent=WholeStructureKinematics().forward(morphology,physical,q,base,refs)
    candidates={}
    assignments=[]
    for index,anchor in enumerate(anchors):
        c=deepcopy(scene.contact_candidate_set.candidates[0])
        c.candidate_id=index;c.anchor_id=anchor.anchor_id;c.contact_mode=ContactMode.GRASP
        c.contact_pose_world=independent.anchor_poses_world[anchor.anchor_id]
        c.normal_world=tuple(-Rotation.from_quat(c.contact_pose_world[3:]).as_matrix()[:,0])
        candidates[index]=c
        assignments.append(ContactAssignment(c.slot_id,c.anchor_id,index,c.contact_mode,'maintain'))
    solution=ArticulatedContactIKSolver(physical,config=ArticulatedIKConfig(maximum_iterations=40)).solve(
        morphology=morphology,assignments=list(reversed(assignments)),candidates=candidates,
        initial_joint_positions_rad=q,initial_base_pose_world=base)
    assert solution.feasible
    assert solution.maximum_position_error_m < 1e-6
    assert solution.maximum_normal_error_rad < 1e-6


def test_neighborhood_pairs_cover_cartesian_choices_and_preserve_observation(request_scene):
    from amsrr.robot_model.gripper_surfaces import with_neighbor_grasp_anchors
    scene, task, physical, execution = request_scene
    morphology = with_neighbor_grasp_anchors(scene.morphology_graph, physical,
        slot_ids=[0], max_force_n=30., max_torque_nm=5.)
    anchors = [a for a in morphology.robot_anchors if a.anchor_type == 'grasp']
    sets = [{a.anchor_id for a in anchors if a.capability['grasp_neighborhood_mask'] & bit} for bit in (1,2)]
    expected = {tuple(sorted((a,b))) for a in sets[0] for b in sets[1] if a != b}
    built = IRGBuilder().build_with_scene_graph(task)
    sampled = ContactCandidateSampler(ContactCandidateSamplerConfig(max_grasp_contacts=2,grasp_neighborhoods=True)).sample(
        task_spec=task, irg=scene.irg, interaction_envelope=scene.interaction_envelope,
        morphology_graph=morphology, geometry_descriptors=built.scene_graph.geometry_descriptors)
    cs = {c.candidate_id:c for c in sampled.candidates}
    groups = [g for g in sampled.group_proposals if g.group_type=='grasp_pair']
    actual = {tuple(sorted(cs[i].anchor_id for i in g.candidate_ids)) for g in groups}
    assert actual == expected
    assert len(groups) > 8
    assert len({g.group_id for g in groups}) == len(groups)
    assert all(len({cs[i].anchor_id for i in g.candidate_ids})==2 for g in groups)
    observation=deepcopy(scene.runtime_observation)
    observation.morphology_graph=morphology
    context=decision((replace(scene,morphology_graph=morphology,
        contact_candidate_set=sampled,runtime_observation=observation),task,physical,execution))
    actor=RequestActorCritic(morphology_aware=True)
    out=actor.decide(context,deterministic=True)
    assert torch.isfinite(out['logits']).any()


def test_neighborhood_membership_is_not_a_numeric_policy_feature(request_scene):
    from amsrr.policies.contact_group_geometry import contact_geometry
    context = decision(request_scene)
    before, membership = contact_geometry(context)
    copied = deepcopy(context.scene.contact_candidate_set)
    for index, candidate in enumerate(copied.candidates):
        candidate.candidate_scores['grasp_neighborhood_id'] = float(index % 3 + 1)
    modified = decision((replace(context.scene, contact_candidate_set=copied),
        context.task_spec, context.physical_model, context.execution_state))
    after, after_membership = contact_geometry(modified)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.equal(membership, after_membership)
