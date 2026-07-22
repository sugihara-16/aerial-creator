from __future__ import annotations

"""Tensorized task-adapter targets for Order 9 object grasp-and-carry."""

from dataclasses import dataclass

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
    Order9ObjectTaskRuntimeConfig,
)


ORDER9_CONTACT_SCHEDULE_INACTIVE = 0
ORDER9_CONTACT_SCHEDULE_APPROACH = 1
ORDER9_CONTACT_SCHEDULE_ATTACH = 2
ORDER9_CONTACT_SCHEDULE_MAINTAIN = 3
ORDER9_CONTACT_SCHEDULE_RELEASE = 4


@dataclass(frozen=True)
class Order9TensorObjectTaskTarget:
    desired_robot_root_pose_world: torch.Tensor
    desired_robot_root_twist_world: torch.Tensor
    nominal_joint_positions_rad: torch.Tensor
    nominal_joint_velocities_radps: torch.Tensor
    desired_object_pose_world: torch.Tensor
    phase_goal_robot_root_pose_world: torch.Tensor
    phase_goal_object_pose_world: torch.Tensor
    phase_progress: torch.Tensor
    contact_schedule_index: torch.Tensor


class Order9TensorObjectTaskRuntime:
    def __init__(
        self, config: Order9ObjectTaskRuntimeConfig | None = None
    ) -> None:
        self.config = config or Order9ObjectTaskRuntimeConfig()
        self.config.validate()
        self._durations = tuple(
            float(self.config.phase_duration_s[phase.value])
            for phase in ORDER9_OBJECT_TASK_PHASES
        )

    @property
    def phase_count(self) -> int:
        return len(ORDER9_OBJECT_TASK_PHASES)

    def target(
        self,
        *,
        phase_index: torch.Tensor,
        phase_elapsed_s: torch.Tensor,
        reset_robot_root_pose_world: torch.Tensor,
        reset_object_pose_world: torch.Tensor,
        reset_joint_positions_rad: torch.Tensor,
        phase_end_joint_positions_rad: torch.Tensor,
        lift_clearance_m: torch.Tensor,
        transport_distance_m: torch.Tensor,
        phase_end_robot_pose_world: torch.Tensor | None = None,
        phase_end_object_pose_world: torch.Tensor | None = None,
    ) -> Order9TensorObjectTaskTarget:
        batch_size = phase_index.shape[0]
        expected = {
            "phase_elapsed_s": (batch_size,),
            "reset_robot_root_pose_world": (batch_size, 7),
            "reset_object_pose_world": (batch_size, 7),
            "lift_clearance_m": (batch_size,),
            "transport_distance_m": (batch_size,),
        }
        for name, shape in expected.items():
            value = locals()[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if (
            reset_joint_positions_rad.ndim != 3
            or phase_end_joint_positions_rad.shape
            != reset_joint_positions_rad.shape
            or reset_joint_positions_rad.shape[0] != batch_size
        ):
            raise ValueError(
                "Order9 joint posture references must share shape [batch, module, joint]"
            )
        if phase_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("Order9 phase_index must be integral")
        if bool((phase_index < 0).any()) or bool(
            (phase_index >= self.phase_count).any()
        ):
            raise ValueError("Order9 tensor task phase index is invalid")
        if bool((phase_elapsed_s < 0.0).any()):
            raise ValueError("Order9 tensor task elapsed time is negative")
        device = phase_elapsed_s.device
        dtype = phase_elapsed_s.dtype
        durations = torch.tensor(
            self._durations, device=device, dtype=dtype
        )[phase_index.long()]
        progress = (phase_elapsed_s / durations).clamp(min=0.0, max=1.0)
        smooth = progress.square() * (3.0 - 2.0 * progress)
        derivative = 6.0 * progress * (1.0 - progress)
        displacement = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        object_displacement = torch.zeros_like(displacement)
        phase = phase_index.long()
        displacement[:, 0] = torch.where(
            phase == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.APPROACH),
            torch.full_like(progress, self.config.approach_offset_m),
            displacement[:, 0],
        )
        lift_mask = phase == ORDER9_OBJECT_TASK_PHASES.index(
            Order9ObjectTaskPhase.LIFT
        )
        displacement[:, 2] = torch.where(
            lift_mask, lift_clearance_m, displacement[:, 2]
        )
        object_displacement[:, 2] = torch.where(
            lift_mask, lift_clearance_m, object_displacement[:, 2]
        )
        transport_mask = phase == ORDER9_OBJECT_TASK_PHASES.index(
            Order9ObjectTaskPhase.TRANSPORT
        )
        displacement[:, 0] = torch.where(
            transport_mask, transport_distance_m, displacement[:, 0]
        )
        object_displacement[:, 0] = torch.where(
            transport_mask, transport_distance_m, object_displacement[:, 0]
        )
        place_mask = phase == ORDER9_OBJECT_TASK_PHASES.index(
            Order9ObjectTaskPhase.PLACE
        )
        displacement[:, 2] = torch.where(
            place_mask, -lift_clearance_m, displacement[:, 2]
        )
        object_displacement[:, 2] = torch.where(
            place_mask, -lift_clearance_m, object_displacement[:, 2]
        )
        retreat_mask = phase == ORDER9_OBJECT_TASK_PHASES.index(
            Order9ObjectTaskPhase.RETREAT
        )
        displacement[:, 0] = torch.where(
            retreat_mask,
            torch.full_like(progress, -self.config.retreat_offset_m),
            displacement[:, 0],
        )
        desired_root = reset_robot_root_pose_world.clone()
        if phase_end_robot_pose_world is not None:
            if phase_end_robot_pose_world.shape != (batch_size, 7):
                raise ValueError(
                    "phase_end_robot_pose_world must have shape [batch, 7]"
                )
            displacement = (
                phase_end_robot_pose_world[:, :3]
                - reset_robot_root_pose_world[:, :3]
            )
        desired_root[:, :3] += smooth.unsqueeze(-1) * displacement
        if phase_end_robot_pose_world is not None:
            desired_root[:, 3:7] = _normalized_lerp_quaternion(
                reset_robot_root_pose_world[:, 3:7],
                phase_end_robot_pose_world[:, 3:7],
                smooth,
            )
        desired_object = reset_object_pose_world.clone()
        if phase_end_object_pose_world is not None:
            if phase_end_object_pose_world.shape != (batch_size, 7):
                raise ValueError(
                    "phase_end_object_pose_world must have shape [batch, 7]"
                )
            object_displacement = (
                phase_end_object_pose_world[:, :3]
                - reset_object_pose_world[:, :3]
            )
        desired_object[:, :3] += smooth.unsqueeze(-1) * object_displacement
        if phase_end_object_pose_world is not None:
            desired_object[:, 3:7] = _normalized_lerp_quaternion(
                reset_object_pose_world[:, 3:7],
                phase_end_object_pose_world[:, 3:7],
                smooth,
            )
        linear_velocity = (
            displacement / durations.unsqueeze(-1) * derivative.unsqueeze(-1)
        ).clamp(
            min=-self.config.command_translation_speed_limit_mps,
            max=self.config.command_translation_speed_limit_mps,
        )
        desired_twist = torch.cat(
            (linear_velocity, torch.zeros_like(linear_velocity)), dim=-1
        )
        joint_displacement = (
            phase_end_joint_positions_rad - reset_joint_positions_rad
        )
        joint_positions = reset_joint_positions_rad + (
            smooth[:, None, None] * joint_displacement
        )
        joint_velocity_limit = torch.where(
            phase
            == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.RELEASE),
            torch.full_like(progress, self.config.release_joint_velocity_limit_radps),
            torch.full_like(progress, self.config.contact_joint_velocity_limit_radps),
        )
        joint_velocities = (
            joint_displacement
            / durations[:, None, None]
            * derivative[:, None, None]
        ).clamp(
            min=-joint_velocity_limit[:, None, None],
            max=joint_velocity_limit[:, None, None],
        )
        schedule = torch.full(
            (batch_size,),
            ORDER9_CONTACT_SCHEDULE_INACTIVE,
            device=device,
            dtype=torch.long,
        )
        schedule = torch.where(
            phase == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.APPROACH),
            torch.full_like(schedule, ORDER9_CONTACT_SCHEDULE_APPROACH),
            schedule,
        )
        schedule = torch.where(
            phase
            == ORDER9_OBJECT_TASK_PHASES.index(
                Order9ObjectTaskPhase.CONTACT_ACQUISITION
            ),
            torch.full_like(schedule, ORDER9_CONTACT_SCHEDULE_ATTACH),
            schedule,
        )
        maintain = (
            (phase == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.LIFT))
            | (
                phase
                == ORDER9_OBJECT_TASK_PHASES.index(
                    Order9ObjectTaskPhase.TRANSPORT
                )
            )
            | (phase == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.PLACE))
        )
        schedule = torch.where(
            maintain,
            torch.full_like(schedule, ORDER9_CONTACT_SCHEDULE_MAINTAIN),
            schedule,
        )
        schedule = torch.where(
            phase == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.RELEASE),
            torch.full_like(schedule, ORDER9_CONTACT_SCHEDULE_RELEASE),
            schedule,
        )
        return Order9TensorObjectTaskTarget(
            desired_robot_root_pose_world=desired_root,
            desired_robot_root_twist_world=desired_twist,
            nominal_joint_positions_rad=joint_positions,
            nominal_joint_velocities_radps=joint_velocities,
            desired_object_pose_world=desired_object,
            phase_goal_robot_root_pose_world=(
                phase_end_robot_pose_world
                if phase_end_robot_pose_world is not None
                else _phase_end_pose(reset_robot_root_pose_world, displacement)
            ),
            phase_goal_object_pose_world=(
                phase_end_object_pose_world
                if phase_end_object_pose_world is not None
                else _phase_end_pose(reset_object_pose_world, object_displacement)
            ),
            phase_progress=progress,
            contact_schedule_index=schedule,
        )


def _normalized_lerp_quaternion(
    start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    dot = (start * end).sum(dim=-1, keepdim=True)
    aligned_end = torch.where(dot < 0.0, -end, end)
    value = (1.0 - alpha.unsqueeze(-1)) * start + alpha.unsqueeze(-1) * aligned_end
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _phase_end_pose(start: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    value = start.clone()
    value[:, :3] += displacement
    return value


__all__ = [
    "ORDER9_CONTACT_SCHEDULE_APPROACH",
    "ORDER9_CONTACT_SCHEDULE_ATTACH",
    "ORDER9_CONTACT_SCHEDULE_INACTIVE",
    "ORDER9_CONTACT_SCHEDULE_MAINTAIN",
    "ORDER9_CONTACT_SCHEDULE_RELEASE",
    "Order9TensorObjectTaskRuntime",
    "Order9TensorObjectTaskTarget",
]
