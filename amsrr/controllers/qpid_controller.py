from __future__ import annotations

import math
from dataclasses import dataclass

from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.policy_command_builder import DesiredBiasReferences, PolicyCommandBiasBuilder
from amsrr.controllers.qp_allocator_interface import (
    BoundedVerticalRotorAllocator,
    QPAllocationProblem,
    QPAllocatorInterface,
    QPAllocationResult,
    RigidBodyPseudoinverseAllocator,
    RotorAllocationSpec,
    VirtualThrustQPAllocator,
)
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.physical_model import JointModel, PhysicalModel
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import RuntimeObservation


@dataclass(frozen=True)
class QPIDControllerConfig:
    allocation_mode: str = "bounded_vertical"
    control_dt_s: float = 0.005
    gravity_mps2: float = 9.80665
    default_hover_when_no_wrench: bool = True
    xy_p_gain: float = 3.0
    xy_i_gain: float = 0.05
    xy_d_gain: float = 2.0
    z_p_gain: float = 5.0
    z_i_gain: float = 1.0
    z_d_gain: float = 2.5
    roll_pitch_p_gain: float = 22.0
    roll_pitch_i_gain: float = 1.0
    roll_pitch_d_gain: float = 14.0
    yaw_p_gain: float = 5.0
    yaw_i_gain: float = 1.0
    yaw_d_gain: float = 4.0
    joint_kp: float = 4.0
    joint_kd: float = 0.4
    tracking_warning_residual_norm: float = 1.0e-3
    vertical_tolerance_n: float = 1.0e-6
    unsupported_wrench_tolerance: float = 1.0e-2


class QPIDController:
    """P1 controller scaffold: QP-style rotor allocation plus PD joint bias."""

    def __init__(
        self,
        *,
        allocator: QPAllocatorInterface | None = None,
        bias_builder: PolicyCommandBiasBuilder | None = None,
        rigid_body_model_builder: RigidBodyControlModelBuilder | None = None,
        config: QPIDControllerConfig | None = None,
    ) -> None:
        self.config = config or QPIDControllerConfig()
        self.allocator = allocator or self._default_allocator(self.config)
        self.bias_builder = bias_builder or PolicyCommandBiasBuilder()
        self.rigid_body_model_builder = rigid_body_model_builder or RigidBodyControlModelBuilder()
        self._position_error_integral_world = [0.0, 0.0, 0.0]
        self._attitude_error_integral_body = [0.0, 0.0, 0.0]
        self._pending_position_error_integral_world: list[float] | None = None
        self._pending_attitude_error_integral_body: list[float] | None = None
        self._reference_metrics: dict[str, float] = {}

    def compute(self, context: ControllerContext) -> ControllerCommand:
        refs = context.desired_references or self._build_references(context)
        allocation = self.allocator.allocate(self._allocation_problem(context, refs))
        self._commit_or_freeze_integrators(allocation)
        vectoring_targets = _vectoring_joint_targets(refs, context.physical_model)
        vectoring_targets.update(allocation.vectoring_joint_targets)
        joint_torques = _pd_joint_torques(refs, context.runtime_observation, context.physical_model, self.config)
        dock_commands = _dock_mechanism_hold_commands(context.runtime_observation, context.physical_model)
        controller_status = _controller_status(allocation, self.config)
        controller_status.metrics.update(self._reference_metrics)
        return ControllerCommand(
            rotor_thrusts_n=allocation.rotor_thrusts_n,
            vectoring_joint_targets=vectoring_targets,
            joint_torque_commands=joint_torques,
            dock_mechanism_commands=dock_commands,
            controller_status=controller_status,
        )

    def reset_integrators(self) -> None:
        self._position_error_integral_world = [0.0, 0.0, 0.0]
        self._attitude_error_integral_body = [0.0, 0.0, 0.0]
        self._pending_position_error_integral_world = None
        self._pending_attitude_error_integral_body = None

    @staticmethod
    def _default_allocator(config: QPIDControllerConfig) -> QPAllocatorInterface:
        if config.allocation_mode == "rigid_body_qp":
            return VirtualThrustQPAllocator()
        if config.allocation_mode == "rigid_body_pseudoinverse":
            return RigidBodyPseudoinverseAllocator()
        return BoundedVerticalRotorAllocator()

    def _build_references(self, context: ControllerContext) -> DesiredBiasReferences:
        self._reference_metrics = {}
        self._pending_position_error_integral_world = None
        self._pending_attitude_error_integral_body = None
        current_q = _current_joint_positions(context.runtime_observation)
        current_qdot = _current_joint_velocities(context.runtime_observation)
        nominal_q = dict(current_q)
        nominal_qdot = dict(current_qdot)
        if context.active_knot.posture_target is not None:
            if context.active_knot.posture_target.joint_pos_target is not None:
                nominal_q.update(context.active_knot.posture_target.joint_pos_target)
            if context.active_knot.posture_target.joint_vel_target is not None:
                nominal_qdot.update(context.active_knot.posture_target.joint_vel_target)
        refs = self.bias_builder.build(
            context.policy_command,
            context.active_knot,
            nominal_joint_positions=nominal_q,
            nominal_joint_velocities=nominal_qdot,
            joint_limits=_joint_position_limits(context.physical_model),
            velocity_limits=_joint_velocity_limits(context.physical_model),
        )
        target_wrench = self._target_wrench_from_body_reference(context, refs)
        if target_wrench is not None:
            refs.desired_wrench_body = _sum_wrenches(target_wrench, refs.desired_wrench_body)
        return refs

    def _target_wrench_from_body_reference(
        self,
        context: ControllerContext,
        refs: DesiredBiasReferences,
    ) -> list[float] | None:
        if refs.desired_body_pose is None and refs.desired_body_twist is None:
            return None
        if not context.runtime_observation.module_states:
            return None
        state = context.runtime_observation.module_states[0]
        current_pose = state.pose_world
        target_pose = refs.desired_body_pose or current_pose
        target_twist = refs.desired_body_twist or [0.0] * 6
        dt = max(float(context.control_dt_s or self.config.control_dt_s), 1.0e-9)

        position_error_world = [
            float(target_pose[idx]) - float(current_pose[idx])
            for idx in range(3)
        ]
        current_twist = list(state.twist_world or [0.0] * 6)
        current_twist = (current_twist + [0.0] * 6)[:6]
        target_twist = (list(target_twist) + [0.0] * 6)[:6]
        velocity_error_world = [
            float(target_twist[idx]) - float(current_twist[idx])
            for idx in range(3)
        ]
        position_integral = [
            self._position_error_integral_world[idx] + position_error_world[idx] * dt
            for idx in range(3)
        ]
        self._pending_position_error_integral_world = position_integral

        current_quat = _normalize_quat(_pose_quat(current_pose))
        target_quat = _normalize_quat(_pose_quat(target_pose))
        attitude_error_body = _quat_error_vector_body(current_quat, target_quat)
        body_from_world = _transpose(_quat_to_matrix(current_quat))
        current_angular_velocity_body = _matvec(body_from_world, tuple(float(value) for value in current_twist[3:6]))
        target_angular_velocity_body = tuple(float(value) for value in target_twist[3:6])
        angular_velocity_error_body = [
            target_angular_velocity_body[idx] - current_angular_velocity_body[idx]
            for idx in range(3)
        ]
        attitude_integral = [
            self._attitude_error_integral_body[idx] + attitude_error_body[idx] * dt
            for idx in range(3)
        ]
        self._pending_attitude_error_integral_body = attitude_integral

        desired_acc_world = [
            self.config.xy_p_gain * position_error_world[0]
            + self.config.xy_i_gain * position_integral[0]
            + self.config.xy_d_gain * velocity_error_world[0],
            self.config.xy_p_gain * position_error_world[1]
            + self.config.xy_i_gain * position_integral[1]
            + self.config.xy_d_gain * velocity_error_world[1],
            self.config.z_p_gain * position_error_world[2]
            + self.config.z_i_gain * position_integral[2]
            + self.config.z_d_gain * velocity_error_world[2],
        ]
        rigid_body_model = self.rigid_body_model_builder.build(
            context.morphology_graph,
            context.physical_model,
            context.runtime_observation,
        )
        control_mass_kg = rigid_body_model.total_mass_kg
        desired_force_world = (
            control_mass_kg * desired_acc_world[0],
            control_mass_kg * desired_acc_world[1],
            control_mass_kg * (self.config.gravity_mps2 + desired_acc_world[2]),
        )
        desired_force_body = _matvec(body_from_world, desired_force_world)
        desired_ang_acc_body = (
            self.config.roll_pitch_p_gain * attitude_error_body[0]
            + self.config.roll_pitch_i_gain * attitude_integral[0]
            + self.config.roll_pitch_d_gain * angular_velocity_error_body[0],
            self.config.roll_pitch_p_gain * attitude_error_body[1]
            + self.config.roll_pitch_i_gain * attitude_integral[1]
            + self.config.roll_pitch_d_gain * angular_velocity_error_body[1],
            self.config.yaw_p_gain * attitude_error_body[2]
            + self.config.yaw_i_gain * attitude_integral[2]
            + self.config.yaw_d_gain * angular_velocity_error_body[2],
        )
        desired_torque_body = _matvec(_inertia_matrix_from_inertia6(rigid_body_model.inertia_body), desired_ang_acc_body)
        self._reference_metrics = {
            "target_pos_error_m": math.sqrt(sum(value * value for value in position_error_world)),
            "target_rot_error_rad": math.sqrt(sum(value * value for value in attitude_error_body)),
            "target_velocity_error_norm": math.sqrt(sum(value * value for value in velocity_error_world)),
            "target_angular_velocity_error_norm": math.sqrt(sum(value * value for value in angular_velocity_error_body)),
            "pid_target_builder_active": 1.0,
        }
        return [*desired_force_body, *desired_torque_body]

    def _commit_or_freeze_integrators(self, allocation: QPAllocationResult) -> None:
        if allocation.feasible and not allocation.clipped:
            if self._pending_position_error_integral_world is not None:
                self._position_error_integral_world = list(self._pending_position_error_integral_world)
            if self._pending_attitude_error_integral_body is not None:
                self._attitude_error_integral_body = list(self._pending_attitude_error_integral_body)
        self._pending_position_error_integral_world = None
        self._pending_attitude_error_integral_body = None

    def _allocation_problem(
        self,
        context: ControllerContext,
        refs: DesiredBiasReferences,
    ) -> QPAllocationProblem:
        rigid_body_model = None
        if self._uses_rigid_body_qp():
            rigid_body_model = self.rigid_body_model_builder.build(
                context.morphology_graph,
                context.physical_model,
                context.runtime_observation,
            )
        desired_wrench = refs.desired_wrench_body
        if desired_wrench is None and self.config.default_hover_when_no_wrench:
            hover_mass_kg = rigid_body_model.total_mass_kg if rigid_body_model is not None else context.physical_model.aggregate_mass_kg
            desired_wrench = [0.0, 0.0, hover_mass_kg * self.config.gravity_mps2, 0.0, 0.0, 0.0]
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
            rigid_body_model=rigid_body_model,
            previous_rotor_thrusts_n=(context.previous_command.rotor_thrusts_n if context.previous_command is not None else {}),
            previous_vectoring_joint_targets=(
                context.previous_command.vectoring_joint_targets if context.previous_command is not None else {}
            ),
            control_dt_s=self.config.control_dt_s,
            vertical_tolerance_n=self.config.vertical_tolerance_n,
            unsupported_wrench_tolerance=self.config.unsupported_wrench_tolerance,
        )

    def _uses_rigid_body_qp(self) -> bool:
        return self.config.allocation_mode in {"rigid_body_qp", "rigid_body_pseudoinverse"} or isinstance(
            self.allocator,
            (VirtualThrustQPAllocator, RigidBodyPseudoinverseAllocator),
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
        active_mode=_active_mode(allocation),
        message=message,
        metrics={
            **allocation.metrics,
            "residual_norm": allocation.residual_norm,
            "clipped": 1.0 if allocation.clipped else 0.0,
            "violation_count": float(len(allocation.violation_codes)),
        },
    )


def _active_mode(allocation: QPAllocationResult) -> str:
    if allocation.metrics.get("qp_primary_path") == 1.0:
        return "qpid_rigid_body_qp"
    if allocation.metrics.get("pseudoinverse_path") == 1.0:
        return "qpid_rigid_body_pseudoinverse"
    if allocation.metrics.get("degraded_fallback") == 1.0:
        return "qpid_degraded_fallback"
    return "qpid_baseline"


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


def _sum_wrenches(left: list[float] | None, right: list[float] | None) -> list[float] | None:
    if left is None and right is None:
        return None
    left_values = left or [0.0] * 6
    right_values = right or [0.0] * 6
    return [float(left_values[idx]) + float(right_values[idx]) for idx in range(6)]


def _pose_quat(pose: tuple[float, float, float, float, float, float, float]) -> tuple[float, float, float, float]:
    return (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))


def _normalize_quat(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise SchemaValidationError("Pose quaternion norm must be positive")
    return x / norm, y / norm, z / norm, w / norm


def _quat_error_vector_body(
    current_quat_xyzw: tuple[float, float, float, float],
    target_quat_xyzw: tuple[float, float, float, float],
) -> list[float]:
    error = _quat_multiply(_quat_conjugate(current_quat_xyzw), target_quat_xyzw)
    if error[3] < 0.0:
        error = tuple(-value for value in error)  # type: ignore[assignment]
    vector_norm = math.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
    if vector_norm <= 1.0e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(min(error[3], 1.0), -1.0))
    scale = angle / vector_norm
    return [error[0] * scale, error[1] * scale, error[2] * scale]


def _quat_conjugate(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (-quat_xyzw[0], -quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3])


def _quat_multiply(
    left_xyzw: tuple[float, float, float, float],
    right_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left_xyzw
    rx, ry, rz, rw = right_xyzw
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_to_matrix(quat_xyzw: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = _normalize_quat(quat_xyzw)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _inertia_matrix_from_inertia6(values: list[float]) -> tuple[tuple[float, float, float], ...]:
    if len(values) != 6:
        raise SchemaValidationError("inertia_body must have 6 values")
    ixx, ixy, ixz, iyy, iyz, izz = (float(value) for value in values)
    return (
        (ixx, ixy, ixz),
        (ixy, iyy, iyz),
        (ixz, iyz, izz),
    )
