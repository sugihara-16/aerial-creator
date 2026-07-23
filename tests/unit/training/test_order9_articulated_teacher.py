from __future__ import annotations

from amsrr.feasibility.articulated_reachability import (
    REACHABILITY_JOINT_LIMIT_CODE,
    ArticulatedTrajectoryReachabilityEvaluator,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTrajectoryTeacher,
)
from amsrr.training.order9_c3_teacher import _neutral_runtime_observation


def _system(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    design = build_grasp_carry_variant_design_output(
        task,
        built.irg,
        physical,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    morphology = design.target_morphology
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    context = HighLevelPolicyContext(
        built.irg,
        envelope,
        morphology,
        candidates,
    )
    return task, physical, context


def test_articulated_teacher_emits_complete_joint_com_and_anchor_targets(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    plan = Order9ArticulatedTrajectoryTeacher(physical).plan(
        context,
        initial_object_poses_world={
            obj.object_id: obj.pose_world for obj in task.scene.objects
        },
    )

    assert plan.ik_solution.feasible
    assert plan.ik_solution.maximum_position_error_m <= 0.005
    expected_joint_ids = set(plan.ik_solution.joint_positions_rad)
    for knot in plan.trajectory.knots:
        assert knot.centroidal_target is not None
        assert knot.centroidal_target.com_pos_world is not None
        assert knot.centroidal_target.body_orientation_world is not None
        assert knot.posture_target is not None
        assert set(knot.posture_target.joint_pos_target or {}) == expected_joint_ids
        assert set(knot.posture_target.joint_vel_target or {}) == expected_joint_ids

    checked_context = HighLevelPolicyContext(
        context.irg,
        context.interaction_envelope,
        context.morphology_graph,
        context.contact_candidate_set,
        runtime_observation=_neutral_runtime_observation(
            context.morphology_graph,
            physical,
            task,
        ),
    )
    evaluations = ArticulatedTrajectoryReachabilityEvaluator(
        physical
    ).evaluate_trajectory(
        context=checked_context,
        trajectory=plan.trajectory,
    )

    assert all(value.feasible for value in evaluations)
    assert all(value.evaluator_version for value in evaluations)


def test_reachability_checker_rejects_mutated_teacher_joint_target(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    plan = Order9ArticulatedTrajectoryTeacher(physical).plan(
        context,
        initial_object_poses_world={
            obj.object_id: obj.pose_world for obj in task.scene.objects
        },
    )
    mutated = ContactWrenchTrajectory.from_dict(plan.trajectory.to_dict())
    active = next(
        knot
        for knot in mutated.knots
        if any(
            assignment.schedule_state == "maintain"
            for assignment in knot.contact_assignments
        )
    )
    joint_id = next(iter(active.posture_target.joint_pos_target))
    active.posture_target.joint_pos_target[joint_id] = 10.0
    checked_context = HighLevelPolicyContext(
        context.irg,
        context.interaction_envelope,
        context.morphology_graph,
        context.contact_candidate_set,
        runtime_observation=_neutral_runtime_observation(
            context.morphology_graph,
            physical,
            task,
        ),
    )

    evaluations = ArticulatedTrajectoryReachabilityEvaluator(
        physical
    ).evaluate_trajectory(
        context=checked_context,
        trajectory=mutated,
    )

    assert any(
        REACHABILITY_JOINT_LIMIT_CODE in value.violation_codes
        for value in evaluations
    )
