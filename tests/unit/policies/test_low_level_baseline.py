from __future__ import annotations

import pytest

from amsrr.controllers.policy_command_builder import PolicyCommandBiasBuilder
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.design_policy_base import DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    BaselineLowLevelPolicyConfig,
    LowLevelPolicyContext,
    select_active_knot,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ContactWrenchTrajectory,
    ControllerStatus,
    PostureTarget,
)
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import TaskSpec


def _high_level_inputs(grasp_carry_dict: dict) -> tuple[TaskSpec, HighLevelPolicyContext, object]:
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
    return (
        task,
        HighLevelPolicyContext(
            irg=irg,
            interaction_envelope=envelope,
            morphology_graph=design.target_morphology,
            contact_candidate_set=candidate_set,
        ),
        physical_model,
    )


def _runtime_observation(
    task: TaskSpec,
    context: HighLevelPolicyContext,
    *,
    time_s: float,
    status: ControllerStatus | None = None,
) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=context.morphology_graph,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions={},
                joint_velocities={},
            )
            for module in context.morphology_graph.modules
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
        controller_status=status or ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(phase_label="transport_object", progress_ratio=0.5),
    )


def _low_level_context(
    grasp_carry_dict: dict,
    *,
    active_knot_index: int | None = 3,
    status: ControllerStatus | None = None,
) -> LowLevelPolicyContext:
    task, high_context, physical_model = _high_level_inputs(grasp_carry_dict)
    trajectory = GraspCarryBaselinePlanner().plan(high_context)
    active_knot = trajectory.knots[active_knot_index] if active_knot_index is not None else None
    time_s = trajectory.knots[3].t_rel_s if active_knot is None else active_knot.t_rel_s
    runtime = _runtime_observation(task, high_context, time_s=time_s, status=status)
    return LowLevelPolicyContext(
        runtime_observation=runtime,
        morphology_graph=high_context.morphology_graph,
        physical_model=physical_model,
        contact_wrench_trajectory=trajectory,
        active_knot=active_knot,
        controller_status=status,
    )


def test_baseline_low_level_policy_outputs_policy_command(grasp_carry_dict: dict) -> None:
    context = _low_level_context(grasp_carry_dict, active_knot_index=3)

    command = BaselineLowLevelPolicy().command(context)
    active_knot = select_active_knot(context)
    refs = PolicyCommandBiasBuilder().build(command, active_knot)

    assert command.residual_wrench_body is not None
    assert command.residual_wrench_body[0] == pytest.approx(4.0)
    assert command.desired_body_twist == [0.0] * 6
    assert set(command.desired_anchor_pose_offsets) == {
        assignment.anchor_id for assignment in active_knot.contact_assignments
    }
    assert set(command.contact_tracking_bias) == {
        assignment.candidate_id for assignment in active_knot.contact_assignments
    }
    assert refs.desired_wrench_body is not None
    assert refs.desired_wrench_body[0] == pytest.approx(4.0)
    assert refs.desired_wrench_body[2] == pytest.approx(5.0)
    assert command.priority_weights["low_level_tracking"] == 1.0
    assert not hasattr(command, "rotor_thrusts_n")
    assert type(command).from_json(command.to_json()).to_dict() == command.to_dict()


def test_baseline_low_level_policy_selects_knot_from_runtime_time(grasp_carry_dict: dict) -> None:
    context = _low_level_context(grasp_carry_dict, active_knot_index=None)

    active_knot = select_active_knot(context)
    command = BaselineLowLevelPolicy().command(context)

    assert active_knot.t_rel_s == context.contact_wrench_trajectory.knots[3].t_rel_s
    assert active_knot.object_targets
    assert command.residual_wrench_body is not None
    assert command.residual_wrench_body[0] > 0.0


def test_centroidal_baseline_emits_absolute_posture_targets_without_contact_bias(
    grasp_carry_dict: dict,
) -> None:
    context = _low_level_context(grasp_carry_dict, active_knot_index=3)
    assert context.active_knot is not None
    context.active_knot.posture_target = PostureTarget(
        joint_pos_target={"module_0:pitch_dock_mech_joint1": 0.2},
        joint_vel_target={"module_0:pitch_dock_mech_joint1": 0.0},
    )
    policy = BaselineLowLevelPolicy(
        BaselineLowLevelPolicyConfig(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        )
    )

    command = policy.command(context)

    assert command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert command.joint_position_targets == {"module_0:pitch_dock_mech_joint1": 0.2}
    assert command.joint_velocity_targets == {"module_0:pitch_dock_mech_joint1": 0.0}
    assert command.contact_tracking_bias == {}


def test_baseline_low_level_policy_suppresses_residual_when_controller_infeasible(
    grasp_carry_dict: dict,
) -> None:
    status = ControllerStatus(status="infeasible", qp_feasible=False, message="allocation failed")
    context = _low_level_context(grasp_carry_dict, active_knot_index=3, status=status)

    command = BaselineLowLevelPolicy().command(context)

    assert command.residual_wrench_body == [0.0] * 6
    assert command.priority_weights["residual_wrench"] == 0.0
    assert command.priority_weights["controller_safety"] == 2.0


def test_select_active_knot_rejects_empty_trajectory(grasp_carry_dict: dict) -> None:
    context = _low_level_context(grasp_carry_dict, active_knot_index=0)
    empty_context = LowLevelPolicyContext(
        runtime_observation=context.runtime_observation,
        morphology_graph=context.morphology_graph,
        physical_model=context.physical_model,
        contact_wrench_trajectory=ContactWrenchTrajectory(horizon_s=1.0, dt_s=0.1, knots=[]),
    )

    with pytest.raises(ValueError, match="must contain knots"):
        select_active_knot(empty_context)
