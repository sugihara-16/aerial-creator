from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import ObjectRuntimeState, RuntimeObservation


IDENTITY_POSE: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class LowLevelPolicyContext:
    runtime_observation: RuntimeObservation
    morphology_graph: MorphologyGraph
    physical_model: PhysicalModel
    contact_wrench_trajectory: ContactWrenchTrajectory
    active_knot: InteractionKnot | None = None
    controller_status: ControllerStatus | None = None


class LowLevelPolicyBase(Protocol):
    """pi_L interface: emits short-horizon tracking intent, not actuator commands."""

    def command(self, context: LowLevelPolicyContext) -> PolicyCommand:
        ...


@dataclass(frozen=True)
class BaselineLowLevelPolicyConfig:
    object_position_gain_n_per_m: float = 8.0
    object_velocity_gain_n_per_mps: float = 1.5
    object_angular_velocity_gain_nm_per_radps: float = 0.2
    residual_force_limit_n: float = 4.0
    residual_torque_limit_nm: float = 0.5
    contact_wrench_bias_scale: float = 0.02
    contact_bias_limit: float = 0.25
    controller_warning_residual_scale: float = 0.5
    controller_infeasible_residual_scale: float = 0.0


class BaselineLowLevelPolicy:
    """Deterministic P1 pi_L baseline conditioned on the active pi_H knot.

    The policy keeps learned-residual fields conservative: it passes zero anchor
    offsets for active assignments, emits small contact-tracking biases, and uses
    object-target pose error only as a residual wrench intent for the controller.
    """

    def __init__(self, config: BaselineLowLevelPolicyConfig | None = None) -> None:
        self.config = config or BaselineLowLevelPolicyConfig()

    def command(self, context: LowLevelPolicyContext) -> PolicyCommand:
        active_knot = select_active_knot(context)
        residual = _object_tracking_residual(active_knot, context.runtime_observation, self.config)
        residual = _scale_residual_for_controller_status(residual, _controller_status(context), self.config)
        return PolicyCommand(
            desired_body_twist=_desired_body_twist(active_knot),
            desired_body_pose=_desired_body_pose(active_knot),
            desired_anchor_pose_offsets=_anchor_pose_offsets(active_knot.contact_assignments),
            joint_position_bias={},
            joint_velocity_bias={},
            residual_wrench_body=residual,
            contact_tracking_bias=_contact_tracking_bias(active_knot.contact_assignments, self.config),
            priority_weights=_low_level_priority_weights(active_knot, _controller_status(context)),
        )


def select_active_knot(context: LowLevelPolicyContext) -> InteractionKnot:
    if context.active_knot is not None:
        return context.active_knot
    knots = sorted(context.contact_wrench_trajectory.knots, key=lambda item: item.t_rel_s)
    if not knots:
        raise SchemaValidationError("LowLevelPolicyContext.contact_wrench_trajectory must contain knots")
    time_s = context.runtime_observation.time_s
    active = knots[0]
    for knot in knots:
        if knot.t_rel_s <= time_s:
            active = knot
        else:
            break
    return active


def _controller_status(context: LowLevelPolicyContext) -> ControllerStatus:
    return context.controller_status or context.runtime_observation.controller_status


def _desired_body_twist(active_knot: InteractionKnot) -> list[float] | None:
    if active_knot.centroidal_target is None:
        return None
    if active_knot.centroidal_target.com_vel_world is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vx, vy, vz = active_knot.centroidal_target.com_vel_world
    return [float(vx), float(vy), float(vz), 0.0, 0.0, 0.0]


def _desired_body_pose(active_knot: InteractionKnot) -> Pose7D | None:
    if active_knot.centroidal_target is None:
        return None
    target = active_knot.centroidal_target
    if target.com_pos_world is None and target.body_orientation_world is None:
        return None
    pos = target.com_pos_world or (0.0, 0.0, 0.0)
    quat = target.body_orientation_world or (0.0, 0.0, 0.0, 1.0)
    return (
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def _anchor_pose_offsets(assignments: list[ContactAssignment]) -> dict[int, Pose7D]:
    return {assignment.anchor_id: IDENTITY_POSE for assignment in assignments}


def _contact_tracking_bias(
    assignments: list[ContactAssignment],
    config: BaselineLowLevelPolicyConfig,
) -> dict[int, list[float]]:
    refs: dict[int, list[float]] = {}
    for assignment in assignments:
        if assignment.schedule_state in {"approach", "release"} or assignment.wrench_target is None:
            refs[assignment.candidate_id] = [0.0] * 6
            continue
        refs[assignment.candidate_id] = [
            _clip(float(value) * config.contact_wrench_bias_scale, config.contact_bias_limit)
            for value in assignment.wrench_target
        ]
    return refs


def _object_tracking_residual(
    active_knot: InteractionKnot,
    runtime_observation: RuntimeObservation,
    config: BaselineLowLevelPolicyConfig,
) -> list[float] | None:
    if not active_knot.object_targets:
        return None
    object_by_id = {state.object_id: state for state in runtime_observation.object_states}
    residual = [0.0] * 6
    contributed = False
    for target in active_knot.object_targets:
        if target.pose_target_world is None:
            continue
        state = object_by_id.get(target.object_id)
        if state is None:
            continue
        contributed = True
        target_twist = target.twist_target_world or [0.0] * 6
        _accumulate_object_target_residual(residual, state, target.pose_target_world, target_twist, config)
    if not contributed:
        return None
    return [
        _clip(residual[0], config.residual_force_limit_n),
        _clip(residual[1], config.residual_force_limit_n),
        _clip(residual[2], config.residual_force_limit_n),
        _clip(residual[3], config.residual_torque_limit_nm),
        _clip(residual[4], config.residual_torque_limit_nm),
        _clip(residual[5], config.residual_torque_limit_nm),
    ]


def _accumulate_object_target_residual(
    residual: list[float],
    state: ObjectRuntimeState,
    pose_target_world: Pose7D,
    twist_target_world: list[float],
    config: BaselineLowLevelPolicyConfig,
) -> None:
    for idx in range(3):
        pos_error = float(pose_target_world[idx]) - float(state.pose_world[idx])
        vel_error = float(twist_target_world[idx]) - float(state.twist_world[idx])
        residual[idx] += (
            config.object_position_gain_n_per_m * pos_error
            + config.object_velocity_gain_n_per_mps * vel_error
        )
    for idx in range(3, 6):
        vel_error = float(twist_target_world[idx]) - float(state.twist_world[idx])
        residual[idx] += config.object_angular_velocity_gain_nm_per_radps * vel_error


def _scale_residual_for_controller_status(
    residual: list[float] | None,
    status: ControllerStatus,
    config: BaselineLowLevelPolicyConfig,
) -> list[float] | None:
    if residual is None:
        return None
    scale = 1.0
    if status.status == "warning":
        scale = config.controller_warning_residual_scale
    if status.status in {"infeasible", "fault"} or not status.qp_feasible:
        scale = config.controller_infeasible_residual_scale
    return [float(value) * scale for value in residual]


def _low_level_priority_weights(
    active_knot: InteractionKnot,
    status: ControllerStatus,
) -> dict[str, float]:
    weights = dict(active_knot.priority_weights)
    safety_weight = 2.0 if status.status in {"warning", "infeasible", "fault"} or not status.qp_feasible else 1.0
    weights.update(
        {
            "low_level_tracking": 1.0,
            "residual_wrench": 0.0 if status.status in {"infeasible", "fault"} or not status.qp_feasible else 1.0,
            "controller_safety": safety_weight,
        }
    )
    return weights


def _clip(value: float, limit: float) -> float:
    if limit < 0.0:
        raise SchemaValidationError("BaselineLowLevelPolicyConfig limits must be non-negative")
    return min(max(value, -limit), limit)
