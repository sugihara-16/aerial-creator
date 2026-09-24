from __future__ import annotations

"""Tensor hot-path decoder from Order 9 ``pi_L`` actions to controller intent."""

from dataclasses import dataclass

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9LowLevelPolicyConfig,
)
from amsrr.schemas.physical_model import PhysicalModel


ORDER9_CONTACT_COMPRESSION_ACTION_ADAPTER_VERSION = (
    "order9_contact_compression_scalar_action_adapter_v1"
)
ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION = (
    "order9_morphology_invariant_compression_action_adapter_v2"
)
ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION = (
    "order9_independent_compression_action_adapter_v3"
)


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
    contact_compression_legacy_joint_action: torch.Tensor
    contact_compression_residual_action: torch.Tensor
    contact_compression_action: torch.Tensor
    contact_compression_joint_delta_rad: torch.Tensor


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
        self._position_lower_limits = tuple(
            float(joints_by_id[joint_id].limit_lower)
            for joint_id in self.local_joint_ids
        )
        self._position_upper_limits = tuple(
            float(joints_by_id[joint_id].limit_upper)
            for joint_id in self.local_joint_ids
        )
        self._policy_identity_validated = False

    def decode_nominal(self, *, reference_body_pose_world, reference_body_twist,
                       reference_local_joint_positions_rad, reference_local_joint_velocities_radps,
                       total_mass_kg):
        """Zero-residual specialization, retaining normalization and hard bounds.

        The request controller has no pi_L action. Avoid manufacturing and then
        decoding hundreds of zeros on every tick; the general learned decoder
        remains the independent reference for this specialization.
        """
        pose, twist = reference_body_pose_world, reference_body_twist
        q, qdot = reference_local_joint_positions_rad, reference_local_joint_velocities_radps
        batch = pose.shape[0]
        shape = (batch, len(self.module_ids), len(self.local_joint_ids))
        if pose.shape != (batch, 7) or twist.shape != (batch, 6) or q.shape != shape or qdot.shape != shape or total_mass_kg.shape != (batch,):
            raise ValueError('nominal command input shape differs')
        if not bool(torch.stack([torch.isfinite(v).all() for v in (pose, twist, q, qdot, total_mass_kg)]).all()):
            raise ValueError('Order9 tensor command input must be finite')
        # General decode normalizes the reference, then its identity product.
        orientation = _normalize_quaternion(_normalize_quaternion(pose[:, 3:7]))
        lower = torch.as_tensor(self._position_lower_limits, device=q.device, dtype=q.dtype)
        upper = torch.as_tensor(self._position_upper_limits, device=q.device, dtype=q.dtype)
        scalar_zero = torch.zeros(batch, device=q.device, dtype=q.dtype)
        return Order9TensorPolicyCommand(
            desired_body_pose_world=torch.cat((pose[:, :3], orientation), dim=-1),
            desired_body_twist=twist.clone(), residual_wrench_body=torch.zeros_like(twist),
            joint_position_targets_rad=torch.maximum(torch.minimum(q, upper), lower),
            joint_velocity_targets_radps=qdot.clone(), joint_torque_bias_nm=torch.zeros_like(q),
            joint_target_mask=torch.ones_like(q, dtype=torch.bool),
            module_ids=self.module_ids, local_joint_ids=self.local_joint_ids,
            contact_compression_legacy_joint_action=scalar_zero.clone(),
            contact_compression_residual_action=scalar_zero.clone(),
            contact_compression_action=scalar_zero,
            contact_compression_joint_delta_rad=torch.zeros_like(q))

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
        contact_compression_joint_direction_rad: torch.Tensor | None = None,
        contact_compression_action_mask: torch.Tensor | None = None,
        normalized_contact_compression_residual_action: torch.Tensor | None = None,
        contact_compression_action_limit: float | torch.Tensor | None = None,
        contact_compression_action_adapter_version: str = (
            ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION
        ),
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
        (
            q_delta,
            legacy_compression_action,
            residual_compression_action,
            compression_action,
            compression_joint_delta,
        ) = _joint_position_delta_with_contact_compression_adapter(
            normalized_position_action=(
                normalized_joint_action[:, :, :slot_count]
            ),
            joint_position_delta_limit_rad=float(
                self.config.joint_position_delta_limit_rad
            ),
            contact_compression_joint_direction_rad=(
                contact_compression_joint_direction_rad
            ),
            contact_compression_action_mask=(
                contact_compression_action_mask
            ),
            normalized_contact_compression_residual_action=(
                normalized_contact_compression_residual_action
            ),
            contact_compression_action_limit=contact_compression_action_limit,
            reference_local_joint_mask=reference_local_joint_mask,
            contact_compression_action_adapter_version=(
                contact_compression_action_adapter_version
            ),
        )
        lower = torch.tensor(
            self._position_lower_limits,
            device=reference_body_pose_world.device,
            dtype=reference_body_pose_world.dtype,
        ).reshape(1, 1, -1)
        upper = torch.tensor(
            self._position_upper_limits,
            device=reference_body_pose_world.device,
            dtype=reference_body_pose_world.dtype,
        ).reshape(1, 1, -1)
        # Deployable hard shield: learned residuals and the virtual nominal
        # compression may approach a mechanism bound, but the command passed
        # to the actuator may never cross the PhysicalModel joint interval.
        output_q = torch.maximum(
            torch.minimum(reference_local_joint_positions_rad + q_delta, upper),
            lower,
        )
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
            contact_compression_legacy_joint_action=legacy_compression_action,
            contact_compression_residual_action=residual_compression_action,
            contact_compression_action=compression_action,
            contact_compression_joint_delta_rad=compression_joint_delta,
        )


def _joint_position_delta_with_contact_compression_adapter(
    *,
    normalized_position_action: torch.Tensor,
    joint_position_delta_limit_rad: float,
    contact_compression_joint_direction_rad: torch.Tensor | None,
    contact_compression_action_mask: torch.Tensor | None,
    normalized_contact_compression_residual_action: torch.Tensor | None,
    contact_compression_action_limit: float | torch.Tensor | None,
    reference_local_joint_mask: torch.Tensor,
    contact_compression_action_adapter_version: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Decode direct joint residuals and coherent grasp compression.

    Adapter v2 preserves the historical selected-coordinate behavior for old
    checkpoints.  Adapter v3 keeps every per-joint position action independent
    and uses only the dedicated scalar head for the morphology-specific IK
    compression direction.
    """

    if normalized_position_action.ndim != 3:
        raise ValueError("Order9 normalized position action must be [B, M, J]")
    batch_size, module_count, slot_count = normalized_position_action.shape
    if reference_local_joint_mask.shape != normalized_position_action.shape:
        raise ValueError("Order9 contact adapter joint mask shape differs")
    if contact_compression_action_adapter_version not in {
        ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION,
        ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION,
    }:
        raise ValueError("Order9 contact-compression adapter version differs")
    zero_action = torch.zeros(
        (batch_size,),
        device=normalized_position_action.device,
        dtype=normalized_position_action.dtype,
    )
    zero_delta = torch.zeros_like(normalized_position_action)
    residual_action = (
        zero_action
        if normalized_contact_compression_residual_action is None
        else normalized_contact_compression_residual_action.to(
            device=normalized_position_action.device,
            dtype=normalized_position_action.dtype,
        )
    )
    if residual_action.shape != (batch_size,):
        raise ValueError(
            "Order9 contact-compression residual action shape differs"
        )
    if not bool(torch.isfinite(residual_action).all()) or bool(
        (residual_action.abs() > 1.0 + 1.0e-6).any()
    ):
        raise ValueError(
            "Order9 contact-compression residual action must be finite in [-1, 1]"
        )
    if contact_compression_joint_direction_rad is None:
        if contact_compression_action_mask is not None:
            raise ValueError("Order9 contact adapter mask lacks a direction")
        return (
            normalized_position_action * float(joint_position_delta_limit_rad),
            zero_action,
            zero_action,
            zero_action,
            zero_delta,
        )
    if contact_compression_action_mask is None:
        raise ValueError("Order9 contact adapter direction lacks an active mask")
    direction = contact_compression_joint_direction_rad.to(
        device=normalized_position_action.device,
        dtype=normalized_position_action.dtype,
    )
    if direction.ndim == 2:
        direction = direction.unsqueeze(0).expand(batch_size, -1, -1)
    if direction.shape != (batch_size, module_count, slot_count):
        raise ValueError("Order9 contact adapter direction shape differs")
    if contact_compression_action_mask.shape != (batch_size,):
        raise ValueError("Order9 contact adapter active-mask shape differs")
    if not bool(torch.isfinite(direction).all()):
        raise ValueError("Order9 contact adapter direction is non-finite")
    active = contact_compression_action_mask.to(
        device=normalized_position_action.device, dtype=torch.bool
    )
    if not bool(active.any()):
        return (
            normalized_position_action * float(joint_position_delta_limit_rad),
            zero_action,
            zero_action,
            zero_action,
            zero_delta,
        )
    # Every environment in one topology bucket shares this local IK direction.
    if not bool(torch.allclose(direction, direction[:1].expand_as(direction))):
        raise ValueError("Order9 contact adapter direction differs within bucket")
    flat_direction = direction[0].reshape(-1)
    control_index = int(flat_direction.abs().argmax().item())
    if float(flat_direction[control_index].abs().item()) <= 1.0e-9:
        raise ValueError("Order9 contact adapter direction is zero")
    flat_joint_mask = reference_local_joint_mask.reshape(batch_size, -1)
    if not bool(flat_joint_mask[:, control_index].all()):
        raise ValueError("Order9 contact adapter control coordinate is masked")
    independent = (
        contact_compression_action_adapter_version
        == ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION
    )
    direct_action = normalized_position_action.clone().reshape(batch_size, -1)
    if independent:
        # The dedicated scalar must not steal or suppress any per-joint action.
        active_legacy_action = zero_action
    else:
        legacy_action = direct_action[:, control_index].clone()
        direct_action[:, control_index] = torch.where(
            active,
            torch.zeros_like(legacy_action),
            legacy_action,
        )
        active_legacy_action = torch.where(
            active, legacy_action, torch.zeros_like(legacy_action)
        )
    direct_delta = direct_action.reshape_as(normalized_position_action) * float(
        joint_position_delta_limit_rad
    )
    active_residual_action = torch.where(
        active, residual_action, torch.zeros_like(residual_action)
    )
    compression_action = torch.clamp(
        active_residual_action
        if independent
        else active_legacy_action + active_residual_action,
        min=-1.0,
        max=1.0,
    )
    if contact_compression_action_limit is not None:
        action_limit = torch.as_tensor(
            contact_compression_action_limit,
            device=compression_action.device,
            dtype=compression_action.dtype,
        )
        if action_limit.ndim == 0:
            action_limit = action_limit.expand(batch_size)
        if action_limit.shape != (batch_size,):
            raise ValueError(
                "Order9 contact-compression action limit shape differs"
            )
        if not bool(torch.isfinite(action_limit).all()) or bool(
            ((action_limit <= 0.0) | (action_limit > 1.0)).any()
        ):
            raise ValueError(
                "Order9 contact-compression action limit must be finite in (0, 1]"
            )
        compression_action = torch.maximum(
            torch.minimum(compression_action, action_limit), -action_limit
        )
    compression_joint_delta = (
        compression_action[:, None, None]
        * direction
        * active[:, None, None].to(direction.dtype)
    )
    return (
        direct_delta + compression_joint_delta,
        active_legacy_action,
        active_residual_action,
        compression_action,
        compression_joint_delta,
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
    "ORDER9_CONTACT_COMPRESSION_ACTION_ADAPTER_VERSION",
    "ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION",
    "ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION",
    "Order9TensorPolicyCommand",
    "Order9TensorPolicyCommandDecoder",
]
