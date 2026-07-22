from __future__ import annotations

"""Shared Order 9 ``pi_L`` reference and centroidal-pose action semantics."""

import math
from typing import Sequence

from amsrr.geometry.pose_math import normalize_quat
from amsrr.policies.low_level_policy_base import (
    LowLevelPolicyContext,
    select_active_knot,
)
from amsrr.policies.order9_low_level_policy import Order9LowLevelPolicyConfig
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    PolicyCommand,
)


def order9_pi_l_reference_command(context: LowLevelPolicyContext) -> PolicyCommand:
    """Build the upstream pi_H reference without running deterministic pi_L.

    This command is an actor parameterization reference, not a command that is
    applied alongside the learned output.  The learned actor emits the complete
    controller-facing ``PolicyCommand``; a deterministic low-level policy is
    consulted only when that learned command is rejected or unavailable.
    """

    knot = select_active_knot(context)
    target = knot.centroidal_target
    if target is None:
        raise SchemaValidationError("Order9 pi_L requires an active centroidal target")
    if target.com_pos_world is None and target.body_orientation_world is None:
        raise SchemaValidationError("Order9 pi_L centroidal target has no pose")
    position = target.com_pos_world or (0.0, 0.0, 0.0)
    orientation = target.body_orientation_world or (0.0, 0.0, 0.0, 1.0)
    pose: Pose7D = (
        float(position[0]),
        float(position[1]),
        float(position[2]),
        *normalize_quat(tuple(float(value) for value in orientation)),
    )
    linear = target.com_vel_world or (0.0, 0.0, 0.0)
    posture = knot.posture_target
    return PolicyCommand(
        desired_body_pose=pose,
        desired_body_twist=[
            float(linear[0]),
            float(linear[1]),
            float(linear[2]),
            0.0,
            0.0,
            0.0,
        ],
        residual_wrench_body=[0.0] * 6,
        priority_weights=dict(knot.priority_weights),
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        joint_position_targets=(
            dict(posture.joint_pos_target or {}) if posture is not None else {}
        ),
        joint_velocity_targets=(
            dict(posture.joint_vel_target or {}) if posture is not None else {}
        ),
        joint_torque_bias={},
    )


def encode_order9_centroidal_pose_action(
    reference_pose: Sequence[float],
    target_pose: Sequence[float],
    config: Order9LowLevelPolicyConfig,
) -> tuple[float, float, float, float, float, float]:
    """Express a world-pose target as bounded reference-relative coordinates."""

    reference = _pose(reference_pose, "reference pose")
    target = _pose(target_pose, "target pose")
    position_scale = float(config.centroidal_position_correction_limit_m)
    orientation_scale = float(config.centroidal_orientation_correction_limit_rad)
    relative = _quat_multiply(_quat_conjugate(reference[3:7]), target[3:7])
    rotation_vector = _quaternion_to_rotation_vector(relative)
    return (
        *((target[index] - reference[index]) / position_scale for index in range(3)),
        *(value / orientation_scale for value in rotation_vector),
    )


def decode_order9_centroidal_pose_action(
    reference_pose: Sequence[float],
    normalized_action: Sequence[float],
    config: Order9LowLevelPolicyConfig,
) -> Pose7D:
    """Apply world-position and body-frame rotation-vector corrections."""

    reference = _pose(reference_pose, "reference pose")
    values = tuple(float(value) for value in normalized_action)
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise SchemaValidationError(
            "Order9 centroidal pose action must contain six finite values"
        )
    delta_quaternion = _rotation_vector_to_quaternion(
        tuple(
            values[3 + index]
            * float(config.centroidal_orientation_correction_limit_rad)
            for index in range(3)
        )
    )
    orientation = _quat_multiply(reference[3:7], delta_quaternion)
    return (
        *(
            reference[index]
            + values[index]
            * float(config.centroidal_position_correction_limit_m)
            for index in range(3)
        ),
        *orientation,
    )


def order9_joint_reference(
    command: PolicyCommand,
    *,
    global_joint_id: str,
    local_joint_id: str,
    current_position_rad: float,
) -> tuple[float, float]:
    """Resolve an absolute pi_H posture reference with current-hold fallback."""

    position = command.joint_position_targets.get(
        global_joint_id,
        command.joint_position_targets.get(local_joint_id, current_position_rad),
    )
    velocity = command.joint_velocity_targets.get(
        global_joint_id,
        command.joint_velocity_targets.get(local_joint_id, 0.0),
    )
    resolved = (float(position), float(velocity))
    if not all(math.isfinite(value) for value in resolved):
        raise SchemaValidationError("Order9 posture reference must be finite")
    return resolved


def _pose(values: Sequence[float], label: str) -> Pose7D:
    resolved = tuple(float(value) for value in values)
    if len(resolved) != 7 or not all(math.isfinite(value) for value in resolved):
        raise SchemaValidationError(f"Order9 {label} must contain seven finite values")
    return (*resolved[:3], *normalize_quat(resolved[3:7]))


def _quat_conjugate(values: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = normalize_quat(tuple(float(value) for value in values))
    return (-x, -y, -z, w)


def _quat_multiply(
    left_values: Sequence[float], right_values: Sequence[float]
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = normalize_quat(tuple(float(value) for value in left_values))
    rx, ry, rz, rw = normalize_quat(tuple(float(value) for value in right_values))
    return normalize_quat(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _quaternion_to_rotation_vector(
    values: Sequence[float],
) -> tuple[float, float, float]:
    x, y, z, w = normalize_quat(tuple(float(value) for value in values))
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(w, 0.0))
    scale = angle / vector_norm
    return (x * scale, y * scale, z * scale)


def _rotation_vector_to_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    x, y, z = (float(value) for value in values)
    angle = math.sqrt(x * x + y * y + z * z)
    if angle <= 1.0e-12:
        return normalize_quat((0.5 * x, 0.5 * y, 0.5 * z, 1.0))
    scale = math.sin(0.5 * angle) / angle
    return normalize_quat((x * scale, y * scale, z * scale, math.cos(0.5 * angle)))


__all__ = [
    "decode_order9_centroidal_pose_action",
    "encode_order9_centroidal_pose_action",
    "order9_joint_reference",
    "order9_pi_l_reference_command",
]
