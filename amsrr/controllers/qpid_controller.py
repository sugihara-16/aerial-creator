from __future__ import annotations

from dataclasses import dataclass

from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.policy_command_builder import DesiredBiasReferences, PolicyCommandBiasBuilder
from amsrr.controllers.qp_allocator_interface import (
    BoundedVerticalRotorAllocator,
    QPAllocationProblem,
    QPAllocatorInterface,
    QPAllocationResult,
    RotorAllocationSpec,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.physical_model import JointModel, PhysicalModel
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import RuntimeObservation


@dataclass(frozen=True)
class QPIDControllerConfig:
    gravity_mps2: float = 9.80665
    default_hover_when_no_wrench: bool = True
    joint_kp: float = 4.0
    joint_kd: float = 0.4
    tracking_warning_residual_norm: float = 1.0e-3
    vertical_tolerance_n: float = 1.0e-6
    unsupported_wrench_tolerance: float = 1.0e-6


class QPIDController:
    """P1 controller scaffold: QP-style rotor allocation plus PD joint bias."""

    def __init__(
        self,
        *,
        allocator: QPAllocatorInterface | None = None,
        bias_builder: PolicyCommandBiasBuilder | None = None,
        config: QPIDControllerConfig | None = None,
    ) -> None:
        self.allocator = allocator or BoundedVerticalRotorAllocator()
        self.bias_builder = bias_builder or PolicyCommandBiasBuilder()
        self.config = config or QPIDControllerConfig()

    def compute(self, context: ControllerContext) -> ControllerCommand:
        refs = context.desired_references or self._build_references(context)
        allocation = self.allocator.allocate(self._allocation_problem(context, refs))
        vectoring_targets = _vectoring_joint_targets(refs, context.physical_model)
        joint_torques = _pd_joint_torques(refs, context.runtime_observation, context.physical_model, self.config)
        dock_commands = _dock_mechanism_hold_commands(context.runtime_observation, context.physical_model)
        return ControllerCommand(
            rotor_thrusts_n=allocation.rotor_thrusts_n,
            vectoring_joint_targets=vectoring_targets,
            joint_torque_commands=joint_torques,
            dock_mechanism_commands=dock_commands,
            controller_status=_controller_status(allocation, self.config),
        )

    def _build_references(self, context: ControllerContext) -> DesiredBiasReferences:
        current_q = _current_joint_positions(context.runtime_observation)
        current_qdot = _current_joint_velocities(context.runtime_observation)
        nominal_q = dict(current_q)
        nominal_qdot = dict(current_qdot)
        if context.active_knot.posture_target is not None:
            if context.active_knot.posture_target.joint_pos_target is not None:
                nominal_q.update(context.active_knot.posture_target.joint_pos_target)
            if context.active_knot.posture_target.joint_vel_target is not None:
                nominal_qdot.update(context.active_knot.posture_target.joint_vel_target)
        return self.bias_builder.build(
            context.policy_command,
            context.active_knot,
            nominal_joint_positions=nominal_q,
            nominal_joint_velocities=nominal_qdot,
            joint_limits=_joint_position_limits(context.physical_model),
            velocity_limits=_joint_velocity_limits(context.physical_model),
        )

    def _allocation_problem(
        self,
        context: ControllerContext,
        refs: DesiredBiasReferences,
    ) -> QPAllocationProblem:
        desired_wrench = refs.desired_wrench_body
        if desired_wrench is None and self.config.default_hover_when_no_wrench:
            desired_wrench = [0.0, 0.0, context.physical_model.aggregate_mass_kg * self.config.gravity_mps2, 0.0, 0.0, 0.0]
        return QPAllocationProblem(
            desired_wrench_body=desired_wrench,
            rotors=[
                RotorAllocationSpec(
                    rotor_id=rotor.rotor_id,
                    thrust_axis_body=rotor.thrust_axis_local,
                    thrust_min_n=rotor.thrust_min_n,
                    thrust_max_n=rotor.thrust_max_n,
                )
                for rotor in context.physical_model.rotors
            ],
            previous_rotor_thrusts_n=(context.previous_command.rotor_thrusts_n if context.previous_command is not None else {}),
            vertical_tolerance_n=self.config.vertical_tolerance_n,
            unsupported_wrench_tolerance=self.config.unsupported_wrench_tolerance,
        )


def _controller_status(allocation: QPAllocationResult, config: QPIDControllerConfig) -> ControllerStatus:
    status = "ok"
    message = "allocation feasible"
    if not allocation.feasible:
        status = "infeasible"
        message = "vertical thrust allocation infeasible"
    elif allocation.residual_norm > config.tracking_warning_residual_norm:
        status = "warning"
        message = "allocation feasible with tracking residual"
    return ControllerStatus(
        status=status,  # type: ignore[arg-type]
        qp_feasible=allocation.feasible,
        active_mode="qpid_baseline",
        message=message,
        metrics={
            **allocation.metrics,
            "residual_norm": allocation.residual_norm,
            "clipped": 1.0 if allocation.clipped else 0.0,
            "violation_count": float(len(allocation.violation_codes)),
        },
    )


def _vectoring_joint_targets(
    refs: DesiredBiasReferences,
    physical_model: PhysicalModel,
) -> dict[str, float]:
    vectoring_joint_ids = {
        joint_id
        for rotor in physical_model.rotors
        for joint_id in rotor.vectoring_joint_ids
    }
    limits = _joint_position_limits(physical_model)
    targets: dict[str, float] = {}
    for joint_id in sorted(vectoring_joint_ids):
        if joint_id not in refs.joint_position_ref:
            continue
        targets[joint_id] = _clip_to_limit(refs.joint_position_ref[joint_id], limits.get(joint_id))
    return targets


def _pd_joint_torques(
    refs: DesiredBiasReferences,
    runtime_observation: RuntimeObservation,
    physical_model: PhysicalModel,
    config: QPIDControllerConfig,
) -> dict[str, float]:
    vectoring_joint_ids = {
        joint_id
        for rotor in physical_model.rotors
        for joint_id in rotor.vectoring_joint_ids
    }
    current_q = _current_joint_positions(runtime_observation)
    current_qdot = _current_joint_velocities(runtime_observation)
    effort_limits = _joint_effort_limits(physical_model)
    torques: dict[str, float] = {}
    for joint_id in sorted(refs.joint_position_ref):
        if joint_id in vectoring_joint_ids:
            continue
        q_ref = refs.joint_position_ref[joint_id]
        qdot_ref = refs.joint_velocity_ref.get(joint_id, 0.0)
        torque = config.joint_kp * (q_ref - current_q.get(joint_id, 0.0))
        torque += config.joint_kd * (qdot_ref - current_qdot.get(joint_id, 0.0))
        torques[joint_id] = _clip_symmetric(torque, effort_limits.get(joint_id))
    return torques


def _dock_mechanism_hold_commands(
    runtime_observation: RuntimeObservation,
    physical_model: PhysicalModel,
) -> dict[str, float]:
    current_q = _current_joint_positions(runtime_observation)
    commands: dict[str, float] = {}
    joint_by_id = {joint.joint_id: joint for joint in physical_model.joints}
    for port in physical_model.dock_ports:
        mechanism_joint_id = port.mechanical_limits.get("mechanism_joint_id")
        if not mechanism_joint_id:
            continue
        joint = joint_by_id.get(str(mechanism_joint_id))
        value = current_q.get(str(mechanism_joint_id), 0.0)
        if joint is not None:
            value = _clip_to_limit(value, _limit_tuple(joint))
        commands[str(mechanism_joint_id)] = value
    return commands


def _current_joint_positions(runtime_observation: RuntimeObservation) -> dict[str, float]:
    values: dict[str, float] = {}
    for state in runtime_observation.module_states:
        values.update({joint_id: float(value) for joint_id, value in state.joint_positions.items()})
    return values


def _current_joint_velocities(runtime_observation: RuntimeObservation) -> dict[str, float]:
    values: dict[str, float] = {}
    for state in runtime_observation.module_states:
        values.update({joint_id: float(value) for joint_id, value in state.joint_velocities.items()})
    return values


def _joint_position_limits(physical_model: PhysicalModel) -> dict[str, tuple[float, float]]:
    return {
        joint.joint_id: limit
        for joint in physical_model.joints
        if (limit := _limit_tuple(joint)) is not None
    }


def _joint_velocity_limits(physical_model: PhysicalModel) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for joint in physical_model.joints:
        if joint.velocity_limit is None:
            continue
        limit = abs(float(joint.velocity_limit))
        limits[joint.joint_id] = (-limit, limit)
    return limits


def _joint_effort_limits(physical_model: PhysicalModel) -> dict[str, float]:
    return {
        joint.joint_id: abs(float(joint.effort_limit))
        for joint in physical_model.joints
        if joint.effort_limit is not None
    }


def _limit_tuple(joint: JointModel) -> tuple[float, float] | None:
    if joint.limit_lower is None or joint.limit_upper is None:
        return None
    lower = float(joint.limit_lower)
    upper = float(joint.limit_upper)
    if lower > upper:
        raise SchemaValidationError(f"Joint {joint.joint_id!r} has lower limit above upper limit")
    return lower, upper


def _clip_to_limit(value: float, limit: tuple[float, float] | None) -> float:
    if limit is None:
        return float(value)
    lower, upper = limit
    return min(max(float(value), lower), upper)


def _clip_symmetric(value: float, limit: float | None) -> float:
    if limit is None:
        return float(value)
    return min(max(float(value), -limit), limit)
