from __future__ import annotations

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import P4_2DeterministicGraspCarryPlanner, p4_2_phase_from_knot
from amsrr.policies.design_policy_base import DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import BaselineLowLevelPolicy, LowLevelPolicyContext
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import TaskSpec


def _p4_2_context(grasp_carry_dict: dict) -> tuple[TaskSpec, HighLevelPolicyContext, object]:
    task = TaskSpec.from_dict(grasp_carry_dict)
    builder_result = IRGBuilder().build_with_scene_graph(task)
    irg = builder_result.irg
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = FixedSimpleDesignPolicy().design(
        DesignPolicyContext(
            task_spec=task,
            irg=irg,
            interaction_envelope=envelope,
            physical_model=physical_model,
        )
    )
    candidate_set = ContactCandidateSampler().sample(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=builder_result.scene_graph.geometry_descriptors,
    )
    runtime = RuntimeObservation(
        time_s=0.0,
        morphology_graph=design.target_morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions={},
                joint_velocities={},
            )
            for module in design.target_morphology.modules
        ],
        object_states=[
            ObjectRuntimeState(
                object_id=obj.object_id,
                pose_world=obj.pose_world,
                twist_world=[0.0] * 6,
            )
            for obj in task.scene.objects
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(phase_label="approach", progress_ratio=0.0),
    )
    return (
        task,
        HighLevelPolicyContext(
            irg=irg,
            interaction_envelope=envelope,
            morphology_graph=design.target_morphology,
            contact_candidate_set=candidate_set,
            runtime_observation=runtime,
        ),
        physical_model,
    )


def test_p4_2_planner_outputs_explicit_deterministic_rollout_phases(grasp_carry_dict: dict) -> None:
    _, context, _ = _p4_2_context(grasp_carry_dict)

    trajectory = P4_2DeterministicGraspCarryPlanner().plan(context)

    assert trajectory.derived_mode_label == "p4_2_deterministic_grasp_carry"
    assert [p4_2_phase_from_knot(knot) for knot in trajectory.knots] == [
        "approach",
        "pregrasp_align",
        "attach_attempt",
        "attached_maintain",
        "transport",
        "release",
    ]
    assert [knot.contact_assignments[0].schedule_state for knot in trajectory.knots] == [
        "approach",
        "approach",
        "attach",
        "maintain",
        "maintain",
        "release",
    ]
    assert trajectory.knots[2].object_targets == []
    assert trajectory.knots[2].priority_weights["attach_attempt"] == 1.0
    assert trajectory.knots[4].object_targets[0].pose_target_world == (2.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
    assert trajectory.knots[4].centroidal_target is not None
    assert trajectory.knots[4].centroidal_target.com_pos_world is not None
    assert not hasattr(trajectory, "rotor_thrusts_n")


def test_p4_2_low_level_policy_marks_phase_intent_without_actuator_commands(grasp_carry_dict: dict) -> None:
    _, high_context, physical_model = _p4_2_context(grasp_carry_dict)
    trajectory = P4_2DeterministicGraspCarryPlanner().plan(high_context)
    runtime = high_context.runtime_observation
    assert runtime is not None

    attach_command = BaselineLowLevelPolicy().command(
        LowLevelPolicyContext(
            runtime_observation=runtime,
            morphology_graph=high_context.morphology_graph,
            physical_model=physical_model,
            contact_wrench_trajectory=trajectory,
            active_knot=trajectory.knots[2],
            controller_status=runtime.controller_status,
        )
    )
    transport_command = BaselineLowLevelPolicy().command(
        LowLevelPolicyContext(
            runtime_observation=runtime,
            morphology_graph=high_context.morphology_graph,
            physical_model=physical_model,
            contact_wrench_trajectory=trajectory,
            active_knot=trajectory.knots[4],
            controller_status=runtime.controller_status,
        )
    )

    assert attach_command.priority_weights["p4_2_phase_attach_attempt"] == 1.0
    assert attach_command.priority_weights["attach_condition_gate"] == 1.0
    assert attach_command.desired_body_pose is not None
    assert transport_command.priority_weights["p4_2_phase_transport"] == 1.0
    assert transport_command.priority_weights["attached_object_tracking"] == 1.0
    assert transport_command.residual_wrench_body is not None
    assert not hasattr(attach_command, "rotor_thrusts_n")
    assert not hasattr(transport_command, "vectoring_joint_targets")
