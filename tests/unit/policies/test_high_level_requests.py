from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.high_level_requests import (
    RequestCatalogBuilder,
    HeuristicHighLevelPolicy,
    contact_surface_hash,
)
from amsrr.policies.request_high_level_policy import (
    RequestHighLevelPolicy,
    request_ranking_loss,
    REQUEST_FEATURE_NAMES,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import (
    ActiveExecutionState,
    HighLevelRequest,
    RequestRankingLabel,
    PreviousRequestOutcome,
)
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.task_spec import TaskSpec


def test_catalog_budget_preserves_current_uncommitted_choices(request_scene):
    from amsrr.policies.request_actor_critic import RequestActorCritic
    scene, task, physical, execution = request_scene
    full = RequestCatalogBuilder(maximum_entries=10000).build(
        scene, task_spec=task, physical_model=physical, execution_state=execution)
    expected = {e.request.stable_hash() for e in full.catalog.entries
                if e.phase_id == execution.phase_id and e.request.transition_id is None}
    assert expected
    bounded = RequestCatalogBuilder(maximum_entries=len(expected)).build(
        scene, task_spec=task, physical_model=physical, execution_state=execution)
    assert {e.request.stable_hash() for e in bounded.catalog.entries} == expected
    assert bool(RequestActorCritic().encode(bounded)['mask'].all())


def test_snapshot_hash_matches_explicit_serialization_and_recomputes_after_mutation(request_scene):
    from amsrr.utils.hashing import stable_hash
    context = decision(request_scene)
    def reference():
        fields = dict(irg=context.scene.irg, envelope=context.scene.interaction_envelope,
            morphology=context.scene.morphology_graph, candidates=context.scene.contact_candidate_set,
            task=context.task_spec, model=context.physical_model, observation=context.observation,
            execution=context.execution_state, encoder_runtime=context.scene.runtime_observation,
            previous_outcome=context.previous_outcome)
        return stable_hash({key: value.to_dict() if value is not None else None
                            for key, value in fields.items()})
    original = context.compute_snapshot_hash()
    assert original == reference()
    context.scene.runtime_observation.time_s += .125
    assert context.compute_snapshot_hash() == reference() != original
    with pytest.raises(SchemaValidationError, match='modified'):
        context.validate_snapshot()


@pytest.fixture
def request_scene(grasp_carry_dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_grasp_carry_variant_design_output(
        task,
        built.irg,
        model,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).target_morphology
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    observation = RuntimeObservation(
        0.0,
        morphology,
        [
            ModuleRuntimeState(
                m.module_id,
                m.pose_in_design_frame,
                [0.0] * 6,
                {j.joint_id: 0.0 for j in model.joints},
            )
            for m in morphology.modules
        ],
        [
            ObjectRuntimeState(o.object_id, o.pose_world, [0.0] * 6)
            for o in task.scene.objects
        ],
        [],
        ControllerStatus("ok", True),
        TaskProgressState(),
    )
    scene = HighLevelPolicyContext(
        built.irg, envelope, morphology, candidates, observation
    )
    phase = next(n.node_id for n in built.irg.nodes if n.node_type == IRGNodeType.PHASE)
    return scene, task, model, ActiveExecutionState(phase, 0.0)


def decision(parts):
    scene, task, model, execution = parts
    return RequestCatalogBuilder().build(
        scene, task_spec=task, physical_model=model, execution_state=execution
    )


def test_request_action_is_only_three_ids_and_roundtrips(request_scene):
    context = decision(request_scene)
    assert context.catalog.entries
    assert [f.name for f in fields(HighLevelRequest)] == [
        "contact_group_id",
        "transition_id",
        "subgoal_id",
    ]
    for request in HeuristicHighLevelPolicy().rank(context):
        assert HighLevelRequest.from_json(request.to_json()) == request
        context.catalog.resolve(request)
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        HighLevelRequest.from_dict(
            dict(
                contact_group_id=None,
                transition_id=None,
                subgoal_id="x",
                wrench_target=[0.0] * 6,
            )
        )


def test_teacher_phase_and_future_labels_never_change_features(request_scene):
    torch.manual_seed(7)
    policy = RequestHighLevelPolicy().eval()
    context = decision(request_scene)
    before = policy.request_features(context)
    request_scene[0].runtime_observation.task_progress = TaskProgressState(
        phase_label="release",
        success=True,
        progress_ratio=1.0,
        metrics={"next_phase": 99.0, "future_reward": 1000.0},
    )
    request_scene[0].irg.active_phase_id = 999
    request_scene[0].runtime_observation.controller_status.active_mode = (
        "teacher_release"
    )
    request_scene[0].runtime_observation.controller_status.metrics = {
        "future_reward": 1000.0
    }
    changed = decision(request_scene)
    assert changed.catalog.snapshot_hash == context.catalog.snapshot_hash
    torch.testing.assert_close(policy.request_features(changed), before, rtol=0, atol=0)
    torch.testing.assert_close(policy(changed), policy(context), rtol=0, atol=0)
    assert changed.execution_state.phase_id == context.execution_state.phase_id


def test_object_only_ten_mm_shift_regrounds_candidates(request_scene):
    context = decision(request_scene)
    obj = request_scene[0].runtime_observation.object_states[0]
    obj.pose_world = (obj.pose_world[0] + 0.01, *obj.pose_world[1:])
    shifted = decision(request_scene)
    for before, after in zip(
        context.scene.contact_candidate_set.candidates,
        shifted.scene.contact_candidate_set.candidates,
    ):
        assert after.candidate_id == before.candidate_id
        assert after.contact_pose_world[0] == pytest.approx(
            before.contact_pose_world[0] + 0.01
        )
        assert after.contact_pose_world[1:] == pytest.approx(
            before.contact_pose_world[1:]
        )
    assert shifted.task_spec.scene.environment == context.task_spec.scene.environment
    assert shifted.catalog.snapshot_hash != context.catalog.snapshot_hash


def test_snapshot_and_contact_binding_mutations_are_rejected(request_scene):
    context = decision(request_scene)
    context.observation.object_states[0].pose_world = (
        9.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    with pytest.raises(SchemaValidationError, match="modified"):
        HeuristicHighLevelPolicy().rank(context)
    scene, _, _, execution = request_scene
    group = scene.contact_candidate_set.group_proposals[0]
    execution.contact_group_id = group.group_id
    candidate = scene.contact_candidate_set.candidates[0]
    execution.contact_bindings = {
        candidate.candidate_id: (
            candidate.slot_id,
            candidate.anchor_id + 99,
            candidate.target_entity_id,
        )
    }
    with pytest.raises(SchemaValidationError, match="binding changed"):
        decision(request_scene)


def test_empty_catalog_does_not_invent_fallback(request_scene):
    request_scene[0].contact_candidate_set.group_proposals.clear()
    context = decision(request_scene)
    assert HeuristicHighLevelPolicy().rank(context) == []
    assert RequestHighLevelPolicy().rank(context) == []


@pytest.mark.parametrize(
    "mode,normal_dot,friction,admitted",
    [
        ("grasp", -1.0, 0.6, True),
        ("grasp", 1.0, 0.6, False),
        ("grasp", -1.0, 0.01, False),
        ("support", 1.0, 0.6, True),
        ("push", 1.0, 0.6, True),
    ],
)
def test_catalog_shares_planner_assignment_prerequisites(
    request_scene, mode, normal_dot, friction, admitted
):
    from amsrr.schemas.task_spec import ContactMode

    context = decision(request_scene)
    selected = context.catalog.entries[0]
    candidates = context.scene.contact_candidate_set
    group = next(
        g for g in candidates.group_proposals
        if g.group_id == selected.request.contact_group_id
    )
    candidates.group_proposals = [group]
    by_id = {c.candidate_id: c for c in candidates.candidates}
    assert len(group.candidate_ids) == 2
    for i, cid in enumerate(group.candidate_ids):
        candidate = by_id[cid]
        candidate.contact_mode = ContactMode(mode)
        candidate.normal_world = (1.0 if i == 0 else normal_dot, 0.0, 0.0)
        candidate.friction = friction
    cache_before = deepcopy(candidates.assignment_feasibility_cache)
    entries = RequestCatalogBuilder()._entries(context)
    assert bool(entries) is admitted
    assert candidates.assignment_feasibility_cache == cache_before


def test_request_ranking_backward_and_strict_checkpoint(request_scene):
    context = decision(request_scene)
    policy = RequestHighLevelPolicy()
    label = RequestRankingLabel(
        context.catalog.entries[0].request,
        context.catalog.snapshot_hash,
        "TEST_ONLY_plan",
        "first_choice_success",
    )
    loss = request_ranking_loss(policy, context, label)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.request_head[-1].weight.grad is not None
    clone = RequestHighLevelPolicy.from_checkpoint(policy.checkpoint())
    assert clone.rank(context) == policy.rank(context)
    assert not any(
        "wrench" in n or "timing" in n or "guard_head" in n
        for n, _ in policy.named_parameters()
    )
    checkpoint = policy.checkpoint()
    checkpoint["contract_version"] = "order9_autoregressive_full_trajectory_pi_h_v1"
    with pytest.raises(SchemaValidationError, match="compatible"):
        RequestHighLevelPolicy.from_checkpoint(checkpoint)
    label.execution_outcome = "fallback_success"
    with pytest.raises(SchemaValidationError, match="fallback/failure"):
        request_ranking_loss(policy, context, label)


def test_previous_outcome_must_precede_current_decision(request_scene):
    scene, task, model, state = request_scene
    context = RequestCatalogBuilder().build(
        scene,
        task_spec=task,
        physical_model=model,
        execution_state=state,
        previous_outcome=PreviousRequestOutcome("previous-decision", 1.0, "timeout"),
    )
    with pytest.raises(SchemaValidationError, match="future outcome"):
        RequestHighLevelPolicy().rank(context)


def test_declared_subgoal_pose_and_tolerances_reach_actual_actor_features(
    request_scene,
):
    import math

    phase = next(
        n
        for n in request_scene[0].irg.nodes
        if n.feature.get("phase_label") == "transport_object"
    )
    request_scene[3].phase_id = phase.node_id
    target = next(
        n
        for n in request_scene[0].irg.nodes
        if n.feature.get("pose_target_world") is not None
    )
    pose = target.feature["pose_target_world"]
    target.feature["pose_target_world"] = [
        *pose[:3],
        0.0,
        0.0,
        math.sqrt(0.5),
        math.sqrt(0.5),
    ]
    context = decision(request_scene)
    policy = RequestHighLevelPolicy()
    rows = policy.request_features(context)
    index = next(
        i for i, e in enumerate(context.catalog.entries) if e.transition is None
    )
    feature = dict(zip(REQUEST_FEATURE_NAMES, rows[index].tolist()))
    assert feature["target_position_known"] == 1.0
    assert feature["target_dx_m"] == pytest.approx(
        pose[0] - context.observation.object_states[0].pose_world[0]
    )
    assert feature["target_rz_rad"] == pytest.approx(math.pi / 2)
    assert feature["target_pos_tolerance_m"] == pytest.approx(0.05)


def test_out_of_catalog_combinations_are_rejected(request_scene):
    context = decision(request_scene)
    request = deepcopy(context.catalog.entries[0].request)
    request.subgoal_id = "unbound"
    with pytest.raises(SchemaValidationError, match="absent"):
        context.catalog.resolve(request)


def test_encoder_runtime_and_catalog_tampering_are_detected(request_scene):
    context = decision(request_scene)
    context.scene.runtime_observation.module_states[0].joint_positions[
        "pitch_dock_mech_joint1"
    ] = 0.1
    with pytest.raises(SchemaValidationError, match="modified"):
        RequestHighLevelPolicy().rank(context)
    context = decision(request_scene)
    context.catalog.entries[0].heuristic_score += 1.0
    with pytest.raises(SchemaValidationError, match="catalog was modified"):
        RequestHighLevelPolicy().rank(context)


def test_active_contact_point_identity_survives_object_motion_but_not_resampling(
    request_scene,
):
    context = decision(request_scene)
    entry = context.catalog.entries[0]
    candidates = {
        c.candidate_id: c for c in context.scene.contact_candidate_set.candidates
    }
    state = request_scene[3]
    state.contact_group_id = entry.request.contact_group_id
    state.contact_bindings = {
        cid: (
            candidates[cid].slot_id,
            candidates[cid].anchor_id,
            candidates[cid].target_entity_id,
        )
        for cid in entry.candidate_ids
    }
    state.contact_surface_hashes = {
        cid: contact_surface_hash(candidates[cid], context.observation)
        for cid in entry.candidate_ids
    }
    obj = request_scene[0].runtime_observation.object_states[0]
    obj.pose_world = (obj.pose_world[0] + 0.01, *obj.pose_world[1:])
    assert decision(request_scene).catalog.entries
    candidate = next(
        c
        for c in request_scene[0].contact_candidate_set.candidates
        if c.candidate_id in state.contact_bindings
    )
    candidate.contact_pose_world = (
        candidate.contact_pose_world[0] + 0.01,
        *candidate.contact_pose_world[1:],
    )
    with pytest.raises(SchemaValidationError, match="surface identity"):
        decision(request_scene)


@pytest.mark.parametrize('yaw', [0.0, 0.73])
@pytest.mark.parametrize('local_point,local_normal,blocked', [
    ((0., 0., 0.99), (0., 0., -1.), True),
    ((0., 0., 1.0), (0., 0., -1.), True),
    ((0., 0., 1.0), (0., 0., 1.), False),
    ((0., 0., 1.01), (0., 0., -1.), False),
    ((1.01, 0., 0.99), (0., 0., -1.), False),
    ((1., 0., 1.), (0., 0., -1.), False),
])
def test_contact_occlusion_respects_solid_pose_boundary_and_overhang(
    request_scene, yaw, local_point, local_normal, blocked,
):
    import math
    from amsrr.geometry.pose_math import transform_from_pose
    from amsrr.policies.high_level_requests import _occluded_contact_points
    from amsrr.schemas.task_spec import GeometrySpec, GeometryType, CollisionModel, ObstacleSpec
    context = decision(request_scene)
    scene = context.task_spec.scene
    scene.objects = []
    scene.environment.support_surfaces = []
    pose = (3., -2., 4., 0., 0., math.sin(yaw / 2), math.cos(yaw / 2))
    scene.geometry_library.append(GeometrySpec('occluder', GeometryType.BOX,
        {'size_m': [1., 2., 2.]}, None, CollisionModel.PRIMITIVE, scale=(2., 1., 1.)))
    scene.environment.obstacles = [ObstacleSpec('other_body', 'occluder', pose)]
    transform = transform_from_pose(pose)
    candidate = context.scene.contact_candidate_set.candidates[0]
    point = tuple(pose[i] + sum(transform.rotation[i][j] * local_point[j] for j in range(3)) for i in range(3))
    candidate.contact_pose_world = (*point, 0., 0., 0., 1.)
    candidate.normal_world = tuple(sum(transform.rotation[i][j] * local_normal[j] for j in range(3)) for i in range(3))
    assert (candidate.candidate_id in _occluded_contact_points(context)) == blocked
    # The intended contact partner must never occlude its own surface.
    candidate.target_entity_id = 'other_body'
    assert candidate.candidate_id not in _occluded_contact_points(context)


def test_catalog_excludes_new_contacts_inside_other_obstacle(request_scene):
    from amsrr.schemas.task_spec import GeometrySpec, GeometryType, CollisionModel, ObstacleSpec
    context = decision(request_scene)
    selected = context.catalog.entries[0]
    candidate = next(c for c in context.scene.contact_candidate_set.candidates if c.candidate_id in selected.candidate_ids)
    task = request_scene[1]
    task.scene.geometry_library.append(GeometrySpec('blocker', GeometryType.BOX,
        {'size_m': [.02, .02, .02]}, None, CollisionModel.PRIMITIVE))
    task.scene.environment.obstacles.append(ObstacleSpec('blocker', 'blocker', candidate.contact_pose_world))
    changed = decision(request_scene)
    assert all(candidate.candidate_id not in entry.candidate_ids for entry in changed.catalog.entries)


def test_contact_occlusion_uses_current_other_object_pose(request_scene):
    from amsrr.policies.high_level_requests import _occluded_contact_points
    context = decision(request_scene)
    candidate = context.scene.contact_candidate_set.candidates[0]
    other = deepcopy(context.task_spec.scene.objects[0])
    other.object_id = 'moving_occluder'
    other.pose_world = candidate.contact_pose_world
    context.task_spec.scene.objects.append(other)
    observation = deepcopy(context.observation.object_states[0])
    observation.object_id = other.object_id
    observation.pose_world = candidate.contact_pose_world
    context.observation.object_states.append(observation)
    assert candidate.candidate_id in _occluded_contact_points(context)
    observation.pose_world = (other.pose_world[0] + 10., *other.pose_world[1:])
    assert candidate.candidate_id not in _occluded_contact_points(context)
