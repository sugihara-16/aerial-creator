from __future__ import annotations

"""Tensorized Order 9 QPID controller for topology-bucketed Isaac rollout."""

from dataclasses import dataclass

import torch

from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModel,
)
from amsrr.controllers.batched_virtual_thrust_qp import (
    BatchedVirtualThrustQPConfig,
    BatchedVirtualThrustQPResult,
    solve_batched_virtual_thrust_qp,
)
from amsrr.controllers.qpid_controller import QPIDControllerConfig


@dataclass(frozen=True)
class BatchedQPIDTrackingProfile:
    proportional_gain_scale: torch.Tensor
    integral_gain_scale: torch.Tensor
    derivative_gain_scale: torch.Tensor
    integrator_accumulation_scale: torch.Tensor
    integrator_decay_rate_per_s: torch.Tensor

    @classmethod
    def ones(
        cls, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> "BatchedQPIDTrackingProfile":
        one = torch.ones((batch_size,), device=device, dtype=dtype)
        zero = torch.zeros_like(one)
        return cls(one, one.clone(), one.clone(), one.clone(), zero)


@dataclass(frozen=True)
class BatchedQPIDState:
    position_error_integral_world: torch.Tensor
    attitude_error_integral_body: torch.Tensor
    previous_rotor_thrusts_n: torch.Tensor
    previous_vectoring_targets_rad: torch.Tensor


@dataclass(frozen=True)
class BatchedQPIDResult:
    desired_wrench_body: torch.Tensor
    desired_acceleration_world: torch.Tensor
    desired_angular_acceleration_body: torch.Tensor
    position_error_world: torch.Tensor
    attitude_error_body: torch.Tensor
    allocation: BatchedVirtualThrustQPResult
    next_state: BatchedQPIDState
    integrator_committed: torch.Tensor


class BatchedQPIDController:
    """Same PID/load/QP ownership as ``QPIDController``, batched in torch."""

    def __init__(
        self,
        *,
        config: QPIDControllerConfig | None = None,
        qp_config: BatchedVirtualThrustQPConfig | None = None,
    ) -> None:
        self.config = config or QPIDControllerConfig(
            allocation_mode="rigid_body_qp", control_dt_s=0.02
        )
        if self.config.allocation_mode != "rigid_body_qp":
            raise ValueError("batched QPID requires rigid_body_qp allocation mode")
        self.qp_config = qp_config or BatchedVirtualThrustQPConfig()
        self.qp_config.validate()

    def initial_state(
        self,
        batch_size: int,
        rotor_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchedQPIDState:
        if batch_size < 1 or rotor_count < 1:
            raise ValueError("batched QPID state dimensions must be positive")
        return BatchedQPIDState(
            position_error_integral_world=torch.zeros(
                (batch_size, 3), device=device, dtype=dtype
            ),
            attitude_error_integral_body=torch.zeros(
                (batch_size, 3), device=device, dtype=dtype
            ),
            previous_rotor_thrusts_n=torch.zeros(
                (batch_size, rotor_count), device=device, dtype=dtype
            ),
            previous_vectoring_targets_rad=torch.zeros(
                (batch_size, rotor_count), device=device, dtype=dtype
            ),
        )

    def compute(
        self,
        *,
        control_model: BatchedRigidBodyControlModel,
        desired_body_pose_world: torch.Tensor,
        desired_body_twist: torch.Tensor,
        residual_wrench_body: torch.Tensor,
        state: BatchedQPIDState,
        tracking_profile: BatchedQPIDTrackingProfile | None = None,
        payload_active: torch.Tensor | None = None,
        payload_mass_kg: torch.Tensor | None = None,
        payload_inertia_body: torch.Tensor | None = None,
        payload_com_offset_body: torch.Tensor | None = None,
    ) -> BatchedQPIDResult:
        current_pose = control_model.body_pose_world
        current_twist = control_model.body_twist_world
        batch_size = current_pose.shape[0]
        device = current_pose.device
        dtype = current_pose.dtype
        profile = tracking_profile or BatchedQPIDTrackingProfile.ones(
            batch_size, device=device, dtype=dtype
        )
        self._validate_inputs(
            control_model=control_model,
            desired_body_pose_world=desired_body_pose_world,
            desired_body_twist=desired_body_twist,
            residual_wrench_body=residual_wrench_body,
            state=state,
            tracking_profile=profile,
            payload_active=payload_active,
            payload_mass_kg=payload_mass_kg,
            payload_inertia_body=payload_inertia_body,
            payload_com_offset_body=payload_com_offset_body,
        )
        dt = float(self.config.control_dt_s)
        position_error = desired_body_pose_world[:, :3] - current_pose[:, :3]
        velocity_error = desired_body_twist[:, :3] - current_twist[:, :3]
        retention = torch.exp(-profile.integrator_decay_rate_per_s * dt)
        pending_position_integral = (
            state.position_error_integral_world * retention.unsqueeze(-1)
            + position_error
            * dt
            * profile.integrator_accumulation_scale.unsqueeze(-1)
        )
        current_quaternion = current_pose[:, 3:7]
        target_quaternion = desired_body_pose_world[:, 3:7]
        attitude_error = _orientation_error_body(
            current_quaternion, target_quaternion
        )
        body_from_world = _quaternion_to_matrix(current_quaternion).transpose(-1, -2)
        current_angular_velocity_body = (
            body_from_world @ current_twist[:, 3:6].unsqueeze(-1)
        ).squeeze(-1)
        angular_velocity_error = (
            desired_body_twist[:, 3:6] - current_angular_velocity_body
        )
        pending_attitude_integral = (
            state.attitude_error_integral_body * retention.unsqueeze(-1)
            + attitude_error
            * dt
            * profile.integrator_accumulation_scale.unsqueeze(-1)
        )
        p_scale = profile.proportional_gain_scale.unsqueeze(-1)
        i_scale = profile.integral_gain_scale.unsqueeze(-1)
        d_scale = profile.derivative_gain_scale.unsqueeze(-1)
        p_gain = torch.tensor(
            [
                self.config.xy_p_gain,
                self.config.xy_p_gain,
                self.config.z_p_gain,
            ],
            device=device,
            dtype=dtype,
        )
        i_gain = torch.tensor(
            [
                self.config.xy_i_gain,
                self.config.xy_i_gain,
                self.config.z_i_gain,
            ],
            device=device,
            dtype=dtype,
        )
        d_gain = torch.tensor(
            [
                self.config.xy_d_gain,
                self.config.xy_d_gain,
                self.config.z_d_gain,
            ],
            device=device,
            dtype=dtype,
        )
        desired_acceleration_world = (
            p_scale * p_gain * position_error
            + i_scale * i_gain * pending_position_integral
            + d_scale * d_gain * velocity_error
        )
        force_world = control_model.total_mass_kg.unsqueeze(-1) * (
            desired_acceleration_world
            + torch.tensor(
                [0.0, 0.0, self.config.gravity_mps2],
                device=device,
                dtype=dtype,
            )
        )
        force_body = (body_from_world @ force_world.unsqueeze(-1)).squeeze(-1)
        angular_p_gain = torch.tensor(
            [
                self.config.roll_pitch_p_gain,
                self.config.roll_pitch_p_gain,
                self.config.yaw_p_gain,
            ],
            device=device,
            dtype=dtype,
        )
        angular_i_gain = torch.tensor(
            [
                self.config.roll_pitch_i_gain,
                self.config.roll_pitch_i_gain,
                self.config.yaw_i_gain,
            ],
            device=device,
            dtype=dtype,
        )
        angular_d_gain = torch.tensor(
            [
                self.config.roll_pitch_d_gain,
                self.config.roll_pitch_d_gain,
                self.config.yaw_d_gain,
            ],
            device=device,
            dtype=dtype,
        )
        desired_angular_acceleration_body = (
            p_scale * angular_p_gain * attitude_error
            + i_scale * angular_i_gain * pending_attitude_integral
            + d_scale * angular_d_gain * angular_velocity_error
        )
        torque_body = (
            control_model.inertia_body_matrix
            @ desired_angular_acceleration_body.unsqueeze(-1)
        ).squeeze(-1)
        desired_wrench = torch.cat((force_body, torque_body), dim=-1)
        if payload_active is not None:
            assert payload_mass_kg is not None
            assert payload_inertia_body is not None
            assert payload_com_offset_body is not None
            active = payload_active.to(device=device, dtype=dtype).unsqueeze(-1)
            payload_force_world = payload_mass_kg.unsqueeze(-1) * (
                desired_acceleration_world
                + torch.tensor(
                    [0.0, 0.0, self.config.gravity_mps2],
                    device=device,
                    dtype=dtype,
                )
            )
            payload_force_body = (
                body_from_world @ payload_force_world.unsqueeze(-1)
            ).squeeze(-1)
            payload_inertia_matrix = _inertia6_to_matrix(payload_inertia_body)
            payload_inertia_torque = (
                payload_inertia_matrix
                @ desired_angular_acceleration_body.unsqueeze(-1)
            ).squeeze(-1)
            payload_torque = torch.cross(
                payload_com_offset_body, payload_force_body, dim=-1
            ) + payload_inertia_torque
            desired_wrench = desired_wrench + active * torch.cat(
                (payload_force_body, payload_torque), dim=-1
            )
        desired_wrench = desired_wrench + residual_wrench_body
        allocation = solve_batched_virtual_thrust_qp(
            desired_wrench_body=desired_wrench,
            virtual_x_wrench_columns=control_model.virtual_x_wrench_columns,
            virtual_z_wrench_columns=control_model.virtual_z_wrench_columns,
            current_vectoring_angles_rad=control_model.current_vectoring_angles_rad,
            previous_rotor_thrusts_n=state.previous_rotor_thrusts_n,
            previous_vectoring_targets_rad=state.previous_vectoring_targets_rad,
            thrust_min_n=control_model.thrust_min_n,
            thrust_max_n=control_model.thrust_max_n,
            vectoring_lower_rad=control_model.vectoring_lower_rad,
            vectoring_upper_rad=control_model.vectoring_upper_rad,
            vectoring_velocity_limit_radps=(
                control_model.vectoring_velocity_limit_radps
            ),
            control_dt_s=dt,
            unsupported_wrench_tolerance=self.config.unsupported_wrench_tolerance,
            config=self.qp_config,
        )
        clipped = allocation.thrust_clipped.any(dim=-1) | (
            allocation.vectoring_clipped.any(dim=-1)
        )
        commit = allocation.feasible & ~clipped
        next_position_integral = torch.where(
            commit.unsqueeze(-1),
            pending_position_integral,
            state.position_error_integral_world,
        )
        next_attitude_integral = torch.where(
            commit.unsqueeze(-1),
            pending_attitude_integral,
            state.attitude_error_integral_body,
        )
        return BatchedQPIDResult(
            desired_wrench_body=desired_wrench,
            desired_acceleration_world=desired_acceleration_world,
            desired_angular_acceleration_body=desired_angular_acceleration_body,
            position_error_world=position_error,
            attitude_error_body=attitude_error,
            allocation=allocation,
            next_state=BatchedQPIDState(
                position_error_integral_world=next_position_integral,
                attitude_error_integral_body=next_attitude_integral,
                previous_rotor_thrusts_n=allocation.rotor_thrusts_n,
                previous_vectoring_targets_rad=(
                    allocation.vectoring_joint_targets_rad
                ),
            ),
            integrator_committed=commit,
        )

    @staticmethod
    def reset_state_subset(
        state: BatchedQPIDState,
        env_ids: torch.Tensor,
        *,
        current_vectoring_angles_rad: torch.Tensor | None = None,
    ) -> BatchedQPIDState:
        position = state.position_error_integral_world.clone()
        attitude = state.attitude_error_integral_body.clone()
        thrust = state.previous_rotor_thrusts_n.clone()
        vectoring = state.previous_vectoring_targets_rad.clone()
        position[env_ids] = 0.0
        attitude[env_ids] = 0.0
        thrust[env_ids] = 0.0
        if current_vectoring_angles_rad is None:
            vectoring[env_ids] = 0.0
        else:
            vectoring[env_ids] = current_vectoring_angles_rad[env_ids]
        return BatchedQPIDState(position, attitude, thrust, vectoring)

    @staticmethod
    def _validate_inputs(**values) -> None:
        model = values["control_model"]
        batch_size = model.body_pose_world.shape[0]
        device = model.body_pose_world.device
        dtype = model.body_pose_world.dtype
        expected = {
            "desired_body_pose_world": (batch_size, 7),
            "desired_body_twist": (batch_size, 6),
            "residual_wrench_body": (batch_size, 6),
        }
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.device != device or value.dtype != dtype:
                raise ValueError(f"{name} must share model dtype/device")
        state = values["state"]
        if state.position_error_integral_world.shape != (batch_size, 3):
            raise ValueError("batched QPID position integral shape differs")
        if state.attitude_error_integral_body.shape != (batch_size, 3):
            raise ValueError("batched QPID attitude integral shape differs")
        rotor_shape = model.current_vectoring_angles_rad.shape
        if state.previous_rotor_thrusts_n.shape != rotor_shape or (
            state.previous_vectoring_targets_rad.shape != rotor_shape
        ):
            raise ValueError("batched QPID previous allocation shape differs")
        profile = values["tracking_profile"]
        for value in (
            profile.proportional_gain_scale,
            profile.integral_gain_scale,
            profile.derivative_gain_scale,
            profile.integrator_accumulation_scale,
            profile.integrator_decay_rate_per_s,
        ):
            if value.shape != (batch_size,):
                raise ValueError("batched QPID tracking profile shape differs")
            if bool((value < 0.0).any()) or not bool(torch.isfinite(value).all()):
                raise ValueError("batched QPID tracking profile is invalid")
        for value in (
            profile.proportional_gain_scale,
            profile.integral_gain_scale,
            profile.derivative_gain_scale,
            profile.integrator_accumulation_scale,
        ):
            if bool((value > 1.0).any()):
                raise ValueError("batched QPID tracking scale exceeds one")
        payload_active = values["payload_active"]
        payload_values = (
            values["payload_mass_kg"],
            values["payload_inertia_body"],
            values["payload_com_offset_body"],
        )
        if payload_active is None:
            if any(value is not None for value in payload_values):
                raise ValueError("batched QPID payload fields must be supplied together")
        else:
            if payload_active.shape != (batch_size,) or any(
                value is None for value in payload_values
            ):
                raise ValueError("batched QPID payload fields are incomplete")
            mass, inertia, offset = payload_values
            assert mass is not None and inertia is not None and offset is not None
            if mass.shape != (batch_size,) or inertia.shape != (
                batch_size,
                6,
            ) or offset.shape != (batch_size, 3):
                raise ValueError("batched QPID payload tensor shape differs")


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _orientation_error_body(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    current_xyz, current_w = current[:, :3], current[:, 3:4]
    target_xyz, target_w = target[:, :3], target[:, 3:4]
    inverse_xyz = -current_xyz
    relative_xyz = (
        current_w * target_xyz
        + target_w * inverse_xyz
        + torch.cross(inverse_xyz, target_xyz, dim=-1)
    )
    relative_w = current_w * target_w - (
        inverse_xyz * target_xyz
    ).sum(dim=-1, keepdim=True)
    sign = torch.where(relative_w < 0.0, -1.0, 1.0)
    relative_xyz = relative_xyz * sign
    relative_w = relative_w * sign
    axis_norm = relative_xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(axis_norm, relative_w.clamp_min(1.0e-12))
    return torch.where(
        axis_norm > 1.0e-12,
        relative_xyz / axis_norm.clamp_min(1.0e-12) * angle,
        torch.zeros_like(relative_xyz),
    )


def _inertia6_to_matrix(values: torch.Tensor) -> torch.Tensor:
    ixx, ixy, ixz, iyy, iyz, izz = values.unbind(dim=-1)
    return torch.stack(
        (
            ixx,
            ixy,
            ixz,
            ixy,
            iyy,
            iyz,
            ixz,
            iyz,
            izz,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


__all__ = [
    "BatchedQPIDController",
    "BatchedQPIDResult",
    "BatchedQPIDState",
    "BatchedQPIDTrackingProfile",
]
