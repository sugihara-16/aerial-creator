from __future__ import annotations

"""Tensor replay of hash-bound C3 nominal IK trajectories.

The deterministic configuration-space planner is an offline teacher tool.  A
production C3 rollout loads its accepted dense trajectory once, then samples
that fixed reference on the 50 Hz control clock; it never runs the planner.
"""

import math
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


ORDER9_C3_NOMINAL_RUNTIME_VERSION = (
    "order9_c3_nominal_complete_task_tensor_replay_v5_phase_completion"
)
ORDER9_C3_NOMINAL_BOUNDARY_COMPLETION_CONTRACT = (
    "order9_c3_nominal_successor_reference_completion_gate_v3"
)
# Kept for readers of older artifacts; new runtime metadata uses the expanded
# boundary name below.
ORDER9_C3_NOMINAL_APPROACH_COMPLETION_CONTRACT = (
    ORDER9_C3_NOMINAL_BOUNDARY_COMPLETION_CONTRACT
)
ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE = (
    "order9_accepted_nominal_approach_t0_v1"
)
ORDER9_C3_PHASE_RESET_REFERENCE_VERSION = (
    "order9_c3_nominal_phase_reset_reference_v4_complete_task"
)


@dataclass(frozen=True)
class _PhaseReference:
    times_s: torch.Tensor
    body_pose_world: torch.Tensor
    body_twist_world: torch.Tensor
    joint_positions_rad: torch.Tensor
    joint_velocities_radps: torch.Tensor
    object_pose_world: torch.Tensor
    object_twist_world: torch.Tensor


@dataclass(frozen=True)
class Order9C3PhaseStartReference:
    """Exact start of one accepted nominal task phase."""

    body_pose_local: torch.Tensor
    body_twist: torch.Tensor
    joint_positions_rad: torch.Tensor
    joint_velocities_radps: torch.Tensor
    object_pose_local: torch.Tensor
    object_twist: torch.Tensor

    def validate(self) -> None:
        if self.body_pose_local.shape != (7,) or self.body_twist.shape != (6,):
            raise ValueError("C3 phase-start body reference shape differs")
        if (
            self.joint_positions_rad.ndim != 2
            or self.joint_velocities_radps.shape != self.joint_positions_rad.shape
        ):
            raise ValueError("C3 phase-start joint reference shape differs")
        if self.object_pose_local.shape != (7,) or self.object_twist.shape != (6,):
            raise ValueError("C3 phase-start object reference shape differs")
        for value in (
            self.body_pose_local,
            self.body_twist,
            self.joint_positions_rad,
            self.joint_velocities_radps,
            self.object_pose_local,
            self.object_twist,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("C3 phase-start reference must be finite")


@dataclass(frozen=True)
class Order9C3PhaseResetReferences:
    """Arbitrary-morphology phase strata derived from accepted nominal IK.

    Poses are local to one copied Isaac environment.  The simulator-specific
    articulation root pose is resolved later because the accepted trajectory
    specifies the assembled-body/CoM frame rather than the USD root frame.
    """

    body_pose_local: torch.Tensor
    body_twist: torch.Tensor
    joint_positions_rad: torch.Tensor
    joint_velocities_radps: torch.Tensor
    object_pose_local: torch.Tensor
    object_twist: torch.Tensor
    phase_elapsed_s: torch.Tensor
    phase_progress: torch.Tensor
    phase_start_body_pose_local: torch.Tensor
    phase_start_object_pose_local: torch.Tensor

    def validate(self) -> None:
        phase_count = len(ORDER9_OBJECT_TASK_PHASES)
        if self.body_pose_local.ndim != 3 or self.body_pose_local.shape[:1] != (
            phase_count,
        ) or self.body_pose_local.shape[-1] != 7:
            raise ValueError(
                "C3 phase-reset body poses must be [phase, stratum, 7]"
            )
        stratum_count = self.body_pose_local.shape[1]
        if self.body_twist.shape != (phase_count, stratum_count, 6):
            raise ValueError(
                "C3 phase-reset body twists must be [phase, stratum, 6]"
            )
        if (
            self.joint_positions_rad.ndim != 4
            or self.joint_positions_rad.shape[0] != phase_count
            or self.joint_positions_rad.shape[1] != stratum_count
        ):
            raise ValueError(
                "C3 phase-reset joints must be [phase, stratum, module, joint]"
            )
        if self.joint_velocities_radps.shape != self.joint_positions_rad.shape:
            raise ValueError("C3 phase-reset joint velocity shape differs")
        if self.object_pose_local.shape != (phase_count, stratum_count, 7):
            raise ValueError(
                "C3 phase-reset object poses must be [phase, stratum, 7]"
            )
        if self.object_twist.shape != (phase_count, stratum_count, 6):
            raise ValueError(
                "C3 phase-reset object twists must be [phase, stratum, 6]"
            )
        if self.phase_elapsed_s.shape != (phase_count, stratum_count) or (
            self.phase_progress.shape != (phase_count, stratum_count)
        ):
            raise ValueError("C3 phase-reset progress shapes differ")
        if self.phase_start_body_pose_local.shape != (phase_count, 7) or (
            self.phase_start_object_pose_local.shape != (phase_count, 7)
        ):
            raise ValueError("C3 phase-reset phase-start poses differ")
        for value in (
            self.body_pose_local,
            self.body_twist,
            self.joint_positions_rad,
            self.joint_velocities_radps,
            self.object_pose_local,
            self.object_twist,
            self.phase_elapsed_s,
            self.phase_progress,
            self.phase_start_body_pose_local,
            self.phase_start_object_pose_local,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("C3 phase-reset references must be finite")
        for pose in (
            self.body_pose_local,
            self.object_pose_local,
            self.phase_start_body_pose_local,
            self.phase_start_object_pose_local,
        ):
            norm = torch.linalg.vector_norm(pose[..., 3:7], dim=-1)
            if not bool(torch.allclose(norm, torch.ones_like(norm), atol=1.0e-5)):
                raise ValueError("C3 phase-reset quaternion is not normalized")
        if (
            bool((self.phase_elapsed_s < 0.0).any())
            or bool((self.phase_progress < 0.0).any())
            or bool((self.phase_progress >= 1.0).any())
        ):
            raise ValueError("C3 phase-reset progress is outside its range")

    @property
    def stratum_count(self) -> int:
        return int(self.body_pose_local.shape[1])


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
        expected = tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES)
        if tuple(phase_trajectories) != expected:
            raise SchemaValidationError(
                "C3 nominal runtime requires all task phases in order"
            )
        self._references = {
            phase: self._materialize(trajectory)
            for phase, trajectory in phase_trajectories.items()
        }
        self.provenance = {
            **dict(provenance or {}),
            "runtime_version": self.runtime_version,
            "phase_reset_reference_version": (
                ORDER9_C3_PHASE_RESET_REFERENCE_VERSION
            ),
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

    def phase_start_reference(
        self,
        phase: Order9ObjectTaskPhase = Order9ObjectTaskPhase.APPROACH,
    ) -> Order9C3PhaseStartReference:
        """Return the exact accepted state at ``t=0`` of ``phase``."""

        reference = self._references[phase.value]
        result = Order9C3PhaseStartReference(
            body_pose_local=reference.body_pose_world[0].clone(),
            body_twist=reference.body_twist_world[0].clone(),
            joint_positions_rad=reference.joint_positions_rad[0].clone(),
            joint_velocities_radps=reference.joint_velocities_radps[0].clone(),
            object_pose_local=reference.object_pose_world[0].clone(),
            object_twist=reference.object_twist_world[0].clone(),
        )
        result.validate()
        return result

    def final_joint_positions(self) -> torch.Tensor:
        return self._references[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ].joint_positions_rad[-1].clone()

    def phase_goal_joint_positions(
        self,
        phase_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return each environment's accepted nominal phase-end posture."""

        if phase_index.ndim != 1 or phase_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("C3 phase indices must be integral [batch]")
        if bool(
            ((phase_index < 0) | (phase_index >= len(ORDER9_OBJECT_TASK_PHASES))).any()
        ):
            raise ValueError("C3 phase indices are out of range")
        goals = torch.stack(
            [
                self._references[phase.value].joint_positions_rad[-1]
                for phase in ORDER9_OBJECT_TASK_PHASES
            ]
        )
        return goals.index_select(0, phase_index.to(device=self.device))

    def phase_reset_references(
        self,
        *,
        object_pose_local: torch.Tensor,
        lift_clearance_m: float,
        transport_distance_m: float,
        retreat_offset_m: float,
        phase_duration_s: Mapping[str, float],
        progress_fractions: Sequence[float],
    ) -> Order9C3PhaseResetReferences:
        """Sample all eight persisted phase references without a planner."""

        if object_pose_local.shape != (7,) or not bool(
            torch.isfinite(object_pose_local).all()
        ):
            raise ValueError("C3 phase-reset object pose must be finite [7]")
        values = (
            float(lift_clearance_m),
            float(transport_distance_m),
            float(retreat_offset_m),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("C3 phase-reset task translations are invalid")
        fractions = tuple(float(value) for value in progress_fractions)
        if (
            len(fractions) < 3
            or fractions != tuple(sorted(set(fractions)))
            or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions)
        ):
            raise ValueError("C3 phase-reset progress fractions are invalid")
        if set(phase_duration_s) != {
            phase.value for phase in ORDER9_OBJECT_TASK_PHASES
        } or any(float(value) <= 0.0 for value in phase_duration_s.values()):
            raise ValueError("C3 phase-reset durations are invalid")
        phase_start_body = torch.stack(
            [self._references[phase.value].body_pose_world[0]
             for phase in ORDER9_OBJECT_TASK_PHASES]
        )
        phase_start_objects = torch.stack(
            [self._references[phase.value].object_pose_world[0]
             for phase in ORDER9_OBJECT_TASK_PHASES]
        )
        body_rows = []
        body_twist_rows = []
        joint_rows = []
        joint_velocity_rows = []
        object_rows = []
        object_twist_rows = []
        elapsed_rows = []
        progress_rows = []
        for phase in ORDER9_OBJECT_TASK_PHASES:
            reference = self._references[phase.value]
            progress = torch.tensor(
                fractions, device=self.device, dtype=self.dtype
            )
            elapsed = progress * float(reference.times_s[-1].item())
            sampled = self._sample(reference, elapsed)
            body_rows.append(sampled[0])
            body_twist_rows.append(sampled[1])
            joint_rows.append(sampled[2])
            joint_velocity_rows.append(sampled[3])
            object_rows.append(sampled[4])
            object_twist_rows.append(sampled[5])
            elapsed_rows.append(elapsed)
            progress_rows.append(progress)
        result = Order9C3PhaseResetReferences(
            body_pose_local=torch.stack(body_rows),
            body_twist=torch.stack(body_twist_rows),
            joint_positions_rad=torch.stack(joint_rows),
            joint_velocities_radps=torch.stack(joint_velocity_rows),
            object_pose_local=torch.stack(object_rows),
            object_twist=torch.stack(object_twist_rows),
            phase_elapsed_s=torch.stack(elapsed_rows),
            phase_progress=torch.stack(progress_rows),
            phase_start_body_pose_local=phase_start_body,
            phase_start_object_pose_local=phase_start_objects,
        )
        result.validate()
        return result

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
        desired_object_pose = target.desired_object_pose_world.clone()
        phase_goal = target.phase_goal_robot_root_pose_world.clone()
        phase_goal_object = target.phase_goal_object_pose_world.clone()
        phase_progress = target.phase_progress.clone()
        for runtime_phase, phase_value in enumerate(ORDER9_OBJECT_TASK_PHASES):
            phase = phase_value.value
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
            object_pose = sampled[4]
            object_pose[:, :3] += origins
            object_goal = reference.object_pose_world[-1].unsqueeze(0).expand(
                ids.numel(), -1
            ).clone()
            object_goal[:, :3] += origins
            desired_pose.index_copy_(0, ids, pose)
            desired_twist.index_copy_(0, ids, sampled[1])
            joint_position.index_copy_(0, ids, sampled[2])
            joint_velocity.index_copy_(0, ids, sampled[3])
            desired_object_pose.index_copy_(0, ids, object_pose)
            phase_goal.index_copy_(0, ids, goal)
            phase_goal_object.index_copy_(0, ids, object_goal)
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
            desired_object_pose_world=desired_object_pose,
            phase_goal_robot_root_pose_world=phase_goal,
            phase_goal_object_pose_world=phase_goal_object,
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
        object_pose = []
        object_twist = []
        previous = -1.0
        for knot in trajectory.knots:
            time_s = float(knot.t_rel_s)
            if time_s <= previous:
                raise SchemaValidationError(
                    "C3 nominal replay times must increase strictly"
                )
            previous = time_s
            pose, twist, q, qdot, object_target, object_target_twist = (
                self._knot_values(knot)
            )
            times.append(time_s)
            body_pose.append(pose)
            body_twist.append(twist)
            joint_position.append(q)
            joint_velocity.append(qdot)
            object_pose.append(object_target)
            object_twist.append(object_target_twist)
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
            object_pose_world=torch.tensor(
                object_pose, device=self.device, dtype=self.dtype
            ),
            object_twist_world=torch.tensor(
                object_twist, device=self.device, dtype=self.dtype
            ),
        )

    def _knot_values(
        self, knot: InteractionKnot
    ) -> tuple[
        list[float],
        list[float],
        list[list[float]],
        list[list[float]],
        list[float],
        list[float],
    ]:
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
        object_targets = [
            value
            for value in knot.object_targets
            if value.pose_target_world is not None
        ]
        if len(object_targets) != 1:
            raise SchemaValidationError(
                "C3 nominal replay knot requires one object pose target"
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
        object_target = object_targets[0]
        return (
            pose,
            twist,
            positions,
            velocities,
            [float(value) for value in object_target.pose_target_world],
            [
                float(value)
                for value in (
                    object_target.twist_target_world
                    or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )
            ],
        )

    def _sample(
        self,
        reference: _PhaseReference,
        elapsed_s: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        object_pose_start = reference.object_pose_world.index_select(0, lower)
        object_pose_end = reference.object_pose_world.index_select(0, upper)
        object_pose = lerp(reference.object_pose_world)
        object_pose[:, 3:7] = _normalized_lerp_quaternion(
            object_pose_start[:, 3:7], object_pose_end[:, 3:7], alpha
        )
        return (
            pose,
            lerp(reference.body_twist_world),
            lerp(reference.joint_positions_rad),
            lerp(reference.joint_velocities_radps),
            object_pose,
            lerp(reference.object_twist_world),
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


def gate_order9_c3_nominal_boundary_completion(
    phase_success: torch.Tensor,
    *,
    phase_index: torch.Tensor,
    phase_progress: torch.Tensor,
) -> torch.Tensor:
    """Prevent unsafe early transitions across discontinuous phase boundaries.

    Physical success may become true before the accepted nominal phase reaches
    its endpoint.  The successor reference starts at that endpoint, so an
    early transition skips part of the collision-checked path and creates a
    discontinuous command.  C3 teacher replay therefore requires both the
    physical success condition and reference completion for every phase that
    has a successor.  The final settle phase has no successor and retains its
    physical outcome-based completion semantics.
    """

    if (
        phase_success.dtype != torch.bool
        or phase_success.shape != phase_index.shape
        or phase_success.shape != phase_progress.shape
    ):
        raise ValueError("C3 nominal boundary gate tensor identity differs")
    if not bool(torch.isfinite(phase_progress).all()):
        raise ValueError("C3 nominal boundary progress is non-finite")
    has_successor = phase_index.long() < len(ORDER9_OBJECT_TASK_PHASES) - 1
    reference_complete = phase_progress >= 1.0 - 1.0e-6
    return phase_success & (~has_successor | reference_complete)


def gate_order9_c3_nominal_approach_completion(
    phase_success: torch.Tensor,
    *,
    phase_index: torch.Tensor,
    phase_progress: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for the expanded nominal boundary gate."""

    return gate_order9_c3_nominal_boundary_completion(
        phase_success,
        phase_index=phase_index,
        phase_progress=phase_progress,
    )


def parse_order9_c3_diagnostic_phase_strata(
    raw: str,
    *,
    environment_count: int,
    stratum_count: int = 4,
) -> tuple[tuple[int, int], ...]:
    """Parse one explicit ``phase:stratum`` pair per diagnostic environment."""

    pairs: list[tuple[int, int]] = []
    for token in str(raw).split(","):
        phase_raw, separator, stratum_raw = token.strip().partition(":")
        if not separator:
            raise ValueError(
                "diagnostic phase strata must use PHASE:STRATUM pairs"
            )
        try:
            phase = int(phase_raw)
            stratum = int(stratum_raw)
        except ValueError as exc:
            raise ValueError(
                "diagnostic phase strata must contain integers"
            ) from exc
        if not 0 <= phase < len(ORDER9_OBJECT_TASK_PHASES):
            raise ValueError("diagnostic phase index is outside [0, 7]")
        if not 0 <= stratum < int(stratum_count):
            raise ValueError("diagnostic stratum index is outside its range")
        pairs.append((phase, stratum))
    if len(pairs) != int(environment_count):
        raise ValueError(
            "diagnostic phase strata count must equal the environment count"
        )
    return tuple(pairs)


__all__ = [
    "ORDER9_C3_NOMINAL_APPROACH_COMPLETION_CONTRACT",
    "ORDER9_C3_NOMINAL_BOUNDARY_COMPLETION_CONTRACT",
    "ORDER9_C3_NOMINAL_RUNTIME_VERSION",
    "ORDER9_C3_PHASE_RESET_REFERENCE_VERSION",
    "Order9C3PhaseResetReferences",
    "Order9C3NominalTensorReference",
    "gate_order9_c3_nominal_approach_completion",
    "gate_order9_c3_nominal_boundary_completion",
    "parse_order9_c3_diagnostic_phase_strata",
]
