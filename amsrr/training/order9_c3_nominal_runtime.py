from __future__ import annotations

"""Tensor replay of hash-bound C3 nominal IK trajectories.

The deterministic configuration-space planner is an offline teacher tool.  A
production C3 rollout loads its accepted dense trajectory once, then samples
that fixed reference on the 50 Hz control clock; it never runs the planner.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.simulation.order9_tensor_object_task import (
    Order9TensorObjectTaskTarget,
)


ORDER9_C3_NOMINAL_RUNTIME_VERSION = "order9_c3_nominal_tensor_replay_v1"


@dataclass(frozen=True)
class _PhaseReference:
    times_s: torch.Tensor
    body_pose_world: torch.Tensor
    body_twist_world: torch.Tensor
    joint_positions_rad: torch.Tensor
    joint_velocities_radps: torch.Tensor


class Order9C3NominalTensorReference:
    """GPU-resident 10 Hz reference with interpolated 50 Hz sampling."""

    runtime_version = ORDER9_C3_NOMINAL_RUNTIME_VERSION

    def __init__(
        self,
        *,
        phase_trajectories: Mapping[str, ContactWrenchTrajectory],
        module_ids: Sequence[int],
        joint_ids: Sequence[str],
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.module_ids = tuple(int(value) for value in module_ids)
        self.joint_ids = tuple(str(value) for value in joint_ids)
        if not self.module_ids or not self.joint_ids or not dtype.is_floating_point:
            raise ValueError("C3 nominal tensor reference identity is invalid")
        expected = {
            Order9ObjectTaskPhase.APPROACH.value,
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value,
        }
        if set(phase_trajectories) != expected:
            raise SchemaValidationError(
                "C3 nominal runtime requires approach and contact trajectories"
            )
        self._references = {
            phase: self._materialize(trajectory)
            for phase, trajectory in phase_trajectories.items()
        }
        self.provenance = {
            **dict(provenance or {}),
            "runtime_version": self.runtime_version,
            "configuration_space_planner_runtime_enabled": False,
            "dense_reference_rate_hz": self.nominal_rate_hz,
            "control_sampling_semantics": "linear_interpolation_at_control_rate",
        }

    @property
    def nominal_rate_hz(self) -> float:
        values = []
        for reference in self._references.values():
            deltas = reference.times_s[1:] - reference.times_s[:-1]
            values.append(float(torch.median(deltas).item()))
        return 1.0 / min(values)

    @property
    def phase_durations_s(self) -> dict[str, float]:
        return {
            phase: float(reference.times_s[-1].item())
            for phase, reference in self._references.items()
        }

    def initial_joint_positions(self) -> torch.Tensor:
        return self._references[
            Order9ObjectTaskPhase.APPROACH.value
        ].joint_positions_rad[0].clone()

    def initial_joint_velocities(self) -> torch.Tensor:
        return self._references[
            Order9ObjectTaskPhase.APPROACH.value
        ].joint_velocities_radps[0].clone()

    def initial_body_pose(self) -> torch.Tensor:
        return self._references[
            Order9ObjectTaskPhase.APPROACH.value
        ].body_pose_world[0].clone()

    def initial_body_twist(self) -> torch.Tensor:
        return self._references[
            Order9ObjectTaskPhase.APPROACH.value
        ].body_twist_world[0].clone()

    def final_joint_positions(self) -> torch.Tensor:
        return self._references[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ].joint_positions_rad[-1].clone()

    @torch.no_grad()
    def condition(
        self,
        target: Order9TensorObjectTaskTarget,
        *,
        phase_index: torch.Tensor,
        phase_elapsed_s: torch.Tensor,
        scene_origins: torch.Tensor,
    ) -> Order9TensorObjectTaskTarget:
        batch = phase_index.shape[0]
        if (
            phase_index.shape != (batch,)
            or phase_elapsed_s.shape != (batch,)
            or scene_origins.shape != (batch, 3)
        ):
            raise ValueError("C3 nominal replay batch shapes are invalid")
        desired_pose = target.desired_robot_root_pose_world.clone()
        desired_twist = target.desired_robot_root_twist_world.clone()
        joint_position = target.nominal_joint_positions_rad.clone()
        joint_velocity = target.nominal_joint_velocities_radps.clone()
        phase_goal = target.phase_goal_robot_root_pose_world.clone()
        phase_progress = target.phase_progress.clone()
        for phase, runtime_phase in (
            (
                Order9ObjectTaskPhase.APPROACH.value,
                ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.APPROACH),
            ),
            (
                Order9ObjectTaskPhase.CONTACT_ACQUISITION.value,
                ORDER9_OBJECT_TASK_PHASES.index(
                    Order9ObjectTaskPhase.CONTACT_ACQUISITION
                ),
            ),
        ):
            ids = torch.nonzero(
                phase_index == runtime_phase, as_tuple=False
            ).flatten()
            if not ids.numel():
                continue
            reference = self._references[phase]
            sampled = self._sample(
                reference,
                phase_elapsed_s.index_select(0, ids),
            )
            origins = scene_origins.index_select(0, ids).to(
                device=self.device, dtype=self.dtype
            )
            pose = sampled[0]
            pose[:, :3] += origins
            goal = reference.body_pose_world[-1].unsqueeze(0).expand(
                ids.numel(), -1
            ).clone()
            goal[:, :3] += origins
            desired_pose.index_copy_(0, ids, pose)
            desired_twist.index_copy_(0, ids, sampled[1])
            joint_position.index_copy_(0, ids, sampled[2])
            joint_velocity.index_copy_(0, ids, sampled[3])
            phase_goal.index_copy_(0, ids, goal)
            duration = max(float(reference.times_s[-1].item()), 1.0e-6)
            phase_progress.index_copy_(
                0,
                ids,
                (phase_elapsed_s.index_select(0, ids) / duration).clamp(
                    0.0, 1.0
                ),
            )
        return replace(
            target,
            desired_robot_root_pose_world=desired_pose,
            desired_robot_root_twist_world=desired_twist,
            nominal_joint_positions_rad=joint_position,
            nominal_joint_velocities_radps=joint_velocity,
            phase_goal_robot_root_pose_world=phase_goal,
            phase_progress=phase_progress,
        )

    def _materialize(
        self, trajectory: ContactWrenchTrajectory
    ) -> _PhaseReference:
        trajectory.validate()
        times = []
        body_pose = []
        body_twist = []
        joint_position = []
        joint_velocity = []
        previous = -1.0
        for knot in trajectory.knots:
            time_s = float(knot.t_rel_s)
            if time_s <= previous:
                raise SchemaValidationError(
                    "C3 nominal replay times must increase strictly"
                )
            previous = time_s
            pose, twist, q, qdot = self._knot_values(knot)
            times.append(time_s)
            body_pose.append(pose)
            body_twist.append(twist)
            joint_position.append(q)
            joint_velocity.append(qdot)
        if len(times) < 2 or abs(times[0]) > 1.0e-9:
            raise SchemaValidationError(
                "C3 nominal phase replay must begin at zero with two knots"
            )
        return _PhaseReference(
            times_s=torch.tensor(times, device=self.device, dtype=self.dtype),
            body_pose_world=torch.tensor(
                body_pose, device=self.device, dtype=self.dtype
            ),
            body_twist_world=torch.tensor(
                body_twist, device=self.device, dtype=self.dtype
            ),
            joint_positions_rad=torch.tensor(
                joint_position, device=self.device, dtype=self.dtype
            ),
            joint_velocities_radps=torch.tensor(
                joint_velocity, device=self.device, dtype=self.dtype
            ),
        )

    def _knot_values(
        self, knot: InteractionKnot
    ) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
        centroidal = knot.centroidal_target
        posture = knot.posture_target
        if (
            centroidal is None
            or centroidal.com_pos_world is None
            or centroidal.body_orientation_world is None
            or posture is None
            or posture.joint_pos_target is None
            or posture.joint_vel_target is None
        ):
            raise SchemaValidationError(
                "C3 nominal replay knot lacks centroidal/joint targets"
            )
        pose = [
            *[float(value) for value in centroidal.com_pos_world],
            *[float(value) for value in centroidal.body_orientation_world],
        ]
        twist = [
            *[
                float(value)
                for value in (centroidal.com_vel_world or (0.0, 0.0, 0.0))
            ],
            0.0,
            0.0,
            0.0,
        ]
        positions = []
        velocities = []
        for module_id in self.module_ids:
            q_row = []
            qdot_row = []
            for joint_id in self.joint_ids:
                global_id = f"module_{module_id}:{joint_id}"
                if (
                    global_id not in posture.joint_pos_target
                    or global_id not in posture.joint_vel_target
                ):
                    raise SchemaValidationError(
                        f"C3 nominal replay lacks joint {global_id}"
                    )
                q_row.append(float(posture.joint_pos_target[global_id]))
                qdot_row.append(float(posture.joint_vel_target[global_id]))
            positions.append(q_row)
            velocities.append(qdot_row)
        return pose, twist, positions, velocities

    def _sample(
        self,
        reference: _PhaseReference,
        elapsed_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        elapsed = elapsed_s.to(device=self.device, dtype=self.dtype).clamp(
            0.0, float(reference.times_s[-1].item())
        )
        upper = torch.searchsorted(reference.times_s, elapsed, right=True)
        upper = upper.clamp(1, reference.times_s.numel() - 1)
        lower = upper - 1
        t0 = reference.times_s.index_select(0, lower)
        t1 = reference.times_s.index_select(0, upper)
        alpha = ((elapsed - t0) / (t1 - t0).clamp_min(1.0e-12)).clamp(
            0.0, 1.0
        )

        def lerp(values: torch.Tensor) -> torch.Tensor:
            start = values.index_select(0, lower)
            end = values.index_select(0, upper)
            shape = (alpha.shape[0],) + (1,) * (values.ndim - 1)
            weight = alpha.reshape(shape)
            return start + weight * (end - start)

        pose_start = reference.body_pose_world.index_select(0, lower)
        pose_end = reference.body_pose_world.index_select(0, upper)
        pose = lerp(reference.body_pose_world)
        quaternion = _normalized_lerp_quaternion(
            pose_start[:, 3:7], pose_end[:, 3:7], alpha
        )
        pose[:, 3:7] = quaternion
        return (
            pose,
            lerp(reference.body_twist_world),
            lerp(reference.joint_positions_rad),
            lerp(reference.joint_velocities_radps),
        )


def _normalized_lerp_quaternion(
    start: torch.Tensor,
    end: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    dot = (start * end).sum(dim=-1, keepdim=True)
    aligned_end = torch.where(dot < 0.0, -end, end)
    value = (1.0 - alpha.unsqueeze(-1)) * start + alpha.unsqueeze(-1) * aligned_end
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


__all__ = [
    "ORDER9_C3_NOMINAL_RUNTIME_VERSION",
    "Order9C3NominalTensorReference",
]
