from __future__ import annotations

"""Tensor hot-path decoder from Order 9 ``pi_L`` actions to controller intent."""

from dataclasses import dataclass

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9LowLevelPolicyConfig,
)
from amsrr.schemas.physical_model import PhysicalModel


@dataclass(frozen=True)
class Order9TensorPolicyCommand:
    desired_body_pose_world: torch.Tensor
    desired_body_twist: torch.Tensor
    residual_wrench_body: torch.Tensor
    joint_position_targets_rad: torch.Tensor
    joint_velocity_targets_radps: torch.Tensor
    joint_torque_bias_nm: torch.Tensor
    joint_target_mask: torch.Tensor
    module_ids: tuple[int, ...]
    local_joint_ids: tuple[str, ...]


class Order9TensorPolicyCommandDecoder:
    """Batched equivalent of ``MorphologyConditionedLowLevelPolicy._decode_command``."""

    def __init__(
        self,
        *,
        module_ids: tuple[int, ...],
        physical_model: PhysicalModel,
        config: Order9LowLevelPolicyConfig | None = None,
    ) -> None:
        if not module_ids or tuple(sorted(set(module_ids))) != module_ids:
            raise ValueError("Order9 tensor decoder module ids must be sorted/unique")
        physical_model.validate()
        self.module_ids = module_ids
        self.physical_model = PhysicalModel.from_dict(physical_model.to_dict())
        self.config = config or Order9LowLevelPolicyConfig()
        self.config.validate()
        self.local_joint_ids = tuple(
            sorted(
                {
                    str(port.mechanical_limits["mechanism_joint_id"])
                    for port in physical_model.dock_ports
                    if port.mechanical_limits.get("mechanism_joint_id")
                }
            )
        )
        if len(self.local_joint_ids) > self.config.max_local_joint_slots:
            raise ValueError("Order9 tensor decoder has too few local joint slots")
        joints_by_id = {joint.joint_id: joint for joint in physical_model.joints}
        self._effort_limits = tuple(
            abs(float(joints_by_id[joint_id].effort_limit or 0.0))
            for joint_id in self.local_joint_ids
        )
        self._policy_identity_validated = False

    def decode(
        self,
        *,
        reference_body_pose_world: torch.Tensor,
        reference_body_twist: torch.Tensor,
        normalized_global_action: torch.Tensor,
        normalized_joint_action: torch.Tensor,
        policy_module_ids: torch.Tensor,
        reference_local_joint_positions_rad: torch.Tensor,
        reference_local_joint_velocities_radps: torch.Tensor,
        reference_local_joint_mask: torch.Tensor,
        total_mass_kg: torch.Tensor,
    ) -> Order9TensorPolicyCommand:
        batch_size = reference_body_pose_world.shape[0]
        module_count = len(self.module_ids)
        slot_count = len(self.local_joint_ids)
        expected = {
            "reference_body_pose_world": (batch_size, 7),
            "reference_body_twist": (batch_size, 6),
            "normalized_global_action": (batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            "policy_module_ids": (
                batch_size,
                module_count,
            ),
            "reference_local_joint_positions_rad": (
                batch_size,
                module_count,
                slot_count,
            ),
            "reference_local_joint_velocities_radps": (
                batch_size,
                module_count,
                slot_count,
            ),
            "reference_local_joint_mask": (
                batch_size,
                module_count,
                slot_count,
            ),
            "total_mass_kg": (batch_size,),
        }
        for name, shape in expected.items():
            value = locals()[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        joint_width = 3 * self.config.max_local_joint_slots
        if tuple(normalized_joint_action.shape) != (
            batch_size,
            module_count,
            joint_width,
        ):
            raise ValueError(
                "normalized_joint_action has invalid Order9 policy shape"
            )
        for value in (
            normalized_global_action,
            normalized_joint_action,
            reference_body_pose_world,
            reference_body_twist,
            reference_local_joint_positions_rad,
            reference_local_joint_velocities_radps,
            total_mass_kg,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Order9 tensor command input must be finite")
        desired_pose = _apply_centroidal_pose_action(
            reference_body_pose_world,
            normalized_global_action[:, :6],
            position_limit_m=float(
                self.config.centroidal_position_correction_limit_m
            ),
            orientation_limit_rad=float(
                self.config.centroidal_orientation_correction_limit_rad
            ),
        )
        twist_limits = torch.tensor(
            [
                *([self.config.linear_twist_correction_limit_mps] * 3),
                *([self.config.angular_twist_correction_limit_radps] * 3),
            ],
            device=reference_body_twist.device,
            dtype=reference_body_twist.dtype,
        )
        desired_twist = (
            reference_body_twist + normalized_global_action[:, 6:12] * twist_limits
        )
        module_count_tensor = torch.full_like(total_mass_kg, float(module_count))
        force_scale = (
            total_mass_kg
            * 9.81
            * float(self.config.residual_force_weight_fraction)
        )
        torque_scale = (
            module_count_tensor * float(self.config.residual_torque_per_module_nm)
        )
        wrench_scale = torch.cat(
            (
                force_scale.unsqueeze(-1).expand(-1, 3),
                torque_scale.unsqueeze(-1).expand(-1, 3),
            ),
            dim=-1,
        )
        residual_wrench = normalized_global_action[:, 12:18] * wrench_scale

        if not self._policy_identity_validated:
            expected_ids = list(self.module_ids)
            actual_ids = policy_module_ids.detach().cpu().tolist()
            if any(row != expected_ids for row in actual_ids):
                raise ValueError(
                    "Order9 policy module-id tensor differs from topology bucket"
                )
            self._policy_identity_validated = True
        effort = torch.tensor(
            self._effort_limits,
            device=reference_body_pose_world.device,
            dtype=reference_body_pose_world.dtype,
        )
        q_delta = (
            normalized_joint_action[:, :, :slot_count]
            * float(self.config.joint_position_delta_limit_rad)
        )
        output_q = reference_local_joint_positions_rad + q_delta
        output_qdot = (
            reference_local_joint_velocities_radps
            + normalized_joint_action[
                :,
                :,
                self.config.max_local_joint_slots : (
                    self.config.max_local_joint_slots + slot_count
                ),
            ]
            * float(self.config.joint_velocity_limit_rad_s)
        )
        output_torque = (
            normalized_joint_action[
                :,
                :,
                2 * self.config.max_local_joint_slots : (
                    2 * self.config.max_local_joint_slots + slot_count
                ),
            ]
            * effort.reshape(1, 1, -1)
            * float(self.config.joint_torque_fraction)
        )
        output_mask = reference_local_joint_mask.clone()
        return Order9TensorPolicyCommand(
            desired_body_pose_world=desired_pose,
            desired_body_twist=desired_twist,
            residual_wrench_body=residual_wrench,
            joint_position_targets_rad=output_q,
            joint_velocity_targets_radps=output_qdot,
            joint_torque_bias_nm=output_torque,
            joint_target_mask=output_mask,
            module_ids=self.module_ids,
            local_joint_ids=self.local_joint_ids,
        )


def _apply_centroidal_pose_action(
    reference_pose_world: torch.Tensor,
    normalized_pose_action: torch.Tensor,
    *,
    position_limit_m: float,
    orientation_limit_rad: float,
) -> torch.Tensor:
    position = reference_pose_world[:, :3] + (
        normalized_pose_action[:, :3] * position_limit_m
    )
    reference_quaternion = _normalize_quaternion(reference_pose_world[:, 3:7])
    rotation_vector = normalized_pose_action[:, 3:6] * orientation_limit_rad
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    small = angle <= 1.0e-8
    vector_scale = torch.where(
        small,
        0.5 - angle.square() / 48.0,
        torch.sin(half_angle) / angle.clamp_min(1.0e-12),
    )
    delta = torch.cat(
        (rotation_vector * vector_scale, torch.cos(half_angle)), dim=-1
    )
    orientation = _quaternion_multiply(reference_quaternion, delta)
    return torch.cat((position, orientation), dim=-1)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    value = torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )
    return _normalize_quaternion(value)


def _normalize_quaternion(value: torch.Tensor) -> torch.Tensor:
    normalized = value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return torch.where(normalized[:, 3:4] < 0.0, -normalized, normalized)


__all__ = [
    "Order9TensorPolicyCommand",
    "Order9TensorPolicyCommandDecoder",
]
