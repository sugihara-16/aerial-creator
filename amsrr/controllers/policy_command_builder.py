from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len
from amsrr.schemas.policies import InteractionKnot, PolicyCommand


@dataclass
class DesiredBiasReferences(SchemaBase):
    joint_position_ref: dict[str, float] = field(default_factory=dict)
    joint_velocity_ref: dict[str, float] = field(default_factory=dict)
    desired_wrench_body: list[float] | None = None
    desired_body_twist: list[float] | None = None
    desired_body_pose: Pose7D | None = None
    anchor_pose_refs: dict[int, Pose7D] = field(default_factory=dict)
    contact_tracking_refs: dict[int, dict[str, Any]] = field(default_factory=dict)
    priority_weights: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.desired_wrench_body is not None:
            require_len(self.desired_wrench_body, 6, "DesiredBiasReferences.desired_wrench_body")
        if self.desired_body_twist is not None:
            require_len(self.desired_body_twist, 6, "DesiredBiasReferences.desired_body_twist")
        if self.desired_body_pose is not None:
            require_len(self.desired_body_pose, 7, "DesiredBiasReferences.desired_body_pose")


class PolicyCommandBiasBuilder:
    """Convert π_L intent into QP/PID reference inputs, not actuator commands."""

    def build(
        self,
        policy_command: PolicyCommand,
        active_knot: InteractionKnot,
        *,
        nominal_joint_positions: dict[str, float] | None = None,
        nominal_joint_velocities: dict[str, float] | None = None,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        velocity_limits: dict[str, tuple[float, float]] | None = None,
    ) -> DesiredBiasReferences:
        q_nom = nominal_joint_positions or {}
        qdot_nom = nominal_joint_velocities or {}
        q_ref = _apply_bias_and_clip(q_nom, policy_command.joint_position_bias, joint_limits or {})
        qdot_ref = _apply_bias_and_clip(qdot_nom, policy_command.joint_velocity_bias, velocity_limits or {})
        desired_wrench = _sum_wrenches(
            _centroidal_wrench_from_knot(active_knot),
            policy_command.residual_wrench_body,
        )
        priority_weights = {
            **active_knot.priority_weights,
            **policy_command.priority_weights,
        }
        return DesiredBiasReferences(
            joint_position_ref=q_ref,
            joint_velocity_ref=qdot_ref,
            desired_wrench_body=desired_wrench,
            desired_body_twist=policy_command.desired_body_twist,
            desired_body_pose=policy_command.desired_body_pose,
            anchor_pose_refs=policy_command.desired_anchor_pose_offsets,
            contact_tracking_refs=_contact_tracking_refs(policy_command, active_knot),
            priority_weights=priority_weights,
        )


def _apply_bias_and_clip(
    nominal: dict[str, float],
    bias: dict[str, float],
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    keys = sorted(set(nominal) | set(bias))
    refs: dict[str, float] = {}
    for key in keys:
        value = float(nominal.get(key, 0.0)) + float(bias.get(key, 0.0))
        if key in limits:
            lower, upper = limits[key]
            if lower > upper:
                raise SchemaValidationError(f"Invalid limits for {key!r}: lower > upper")
            value = min(max(value, lower), upper)
        refs[key] = value
    return refs


def _centroidal_wrench_from_knot(active_knot: InteractionKnot) -> list[float] | None:
    if active_knot.centroidal_target is None:
        return None
    if active_knot.centroidal_target.centroidal_wrench_preference is None:
        return None
    return list(active_knot.centroidal_target.centroidal_wrench_preference)


def _sum_wrenches(base: list[float] | None, residual: list[float] | None) -> list[float] | None:
    if base is None and residual is None:
        return None
    base_values = base or [0.0] * 6
    residual_values = residual or [0.0] * 6
    require_len(base_values, 6, "base_wrench")
    require_len(residual_values, 6, "residual_wrench")
    return [float(left) + float(right) for left, right in zip(base_values, residual_values)]


def _contact_tracking_refs(
    policy_command: PolicyCommand,
    active_knot: InteractionKnot,
) -> dict[int, dict[str, Any]]:
    refs: dict[int, dict[str, Any]] = {}
    for assignment in active_knot.contact_assignments:
        bias = list(policy_command.contact_tracking_bias.get(assignment.candidate_id, []))
        refs[assignment.candidate_id] = {
            "slot_id": assignment.slot_id,
            "anchor_id": assignment.anchor_id,
            "candidate_id": assignment.candidate_id,
            "contact_mode": assignment.contact_mode.value,
            "schedule_state": assignment.schedule_state,
            "wrench_target": assignment.wrench_target,
            "tracking_bias": bias,
        }
    return refs
