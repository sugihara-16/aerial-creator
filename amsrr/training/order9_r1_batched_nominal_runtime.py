from __future__ import annotations

"""Environment-wise R1 nominal and wrench references for one Isaac scene."""

from dataclasses import replace
from typing import Any, Sequence

import torch

from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.training.order9_c3_nominal_runtime import (
    Order9C3NominalTensorReference,
)
from amsrr.training.order9_contact_wrench_reward import (
    Order9TensorWrenchRange,
    Order9TensorWrenchRangeReference,
)

ORDER9_R1_BATCHED_NOMINAL_RUNTIME_VERSION = (
    "order9_r1_environment_wise_nominal_runtime_v1"
)

_ACTIVE_BUNDLES: tuple[Any, ...] = ()
_ACTIVE_TASKS: tuple[TaskSpec, ...] = ()
_ACTIVE_WRENCH_TRAJECTORIES: tuple[Any, ...] = ()
_ACTIVE_REPLAY_COUNT = 0


def configure_order9_r1_batched_nominal_runtime(
    *,
    bundles: Sequence[Any],
    tasks: Sequence[TaskSpec],
    wrench_trajectories: Sequence[Any],
    replay_count: int,
) -> None:
    global _ACTIVE_BUNDLES, _ACTIVE_TASKS, _ACTIVE_WRENCH_TRAJECTORIES
    global _ACTIVE_REPLAY_COUNT
    if (
        not bundles
        or len(bundles) != len(tasks)
        or len(bundles) != len(wrench_trajectories)
        or replay_count < 1
    ):
        raise ValueError("R1 batched nominal runtime inputs are not aligned")
    _ACTIVE_BUNDLES = tuple(bundles)
    _ACTIVE_TASKS = tuple(tasks)
    _ACTIVE_WRENCH_TRAJECTORIES = tuple(wrench_trajectories)
    _ACTIVE_REPLAY_COUNT = int(replay_count)


class Order9R1BatchedNominalTensorReference:
    """Sample a different accepted nominal path in every replay pair."""

    runtime_version = ORDER9_R1_BATCHED_NOMINAL_RUNTIME_VERSION

    def __init__(
        self,
        *,
        phase_trajectories,
        module_ids,
        joint_ids,
        device,
        dtype=torch.float32,
        provenance=None,
    ) -> None:
        del phase_trajectories, provenance
        if not _ACTIVE_BUNDLES or _ACTIVE_REPLAY_COUNT < 1:
            raise RuntimeError("R1 batched nominal runtime is not configured")
        self.device = torch.device(device)
        self.dtype = dtype
        self.module_ids = tuple(int(value) for value in module_ids)
        self.joint_ids = tuple(str(value) for value in joint_ids)
        self._references = tuple(
            Order9C3NominalTensorReference(
                phase_trajectories=bundle.phase_trajectories,
                module_ids=self.module_ids,
                joint_ids=self.joint_ids,
                device=self.device,
                dtype=self.dtype,
                provenance=bundle.provenance,
            )
            for bundle in _ACTIVE_BUNDLES
        )
        durations = [reference.phase_durations_s for reference in self._references]
        self._phase_durations_s = {
            phase.value: max(value[phase.value] for value in durations)
            for phase in ORDER9_OBJECT_TASK_PHASES
        }
        self.provenance = dict(self._references[0].provenance)
        self._replay_count = _ACTIVE_REPLAY_COUNT
        self._case_count = len(self._references)
        self._environment_count = self._case_count * self._replay_count
        self._environment_case_index = torch.arange(
            self._case_count, device=self.device, dtype=torch.long
        ).repeat_interleave(self._replay_count)
        self._banks = {
            phase.value: self._materialize_phase_bank(phase.value)
            for phase in ORDER9_OBJECT_TASK_PHASES
        }
        self._phase_goal_joint_bank = torch.stack(
            [
                torch.stack(
                    [
                        reference._references[phase.value].joint_positions_rad[-1]
                        for phase in ORDER9_OBJECT_TASK_PHASES
                    ]
                )
                for reference in self._references
            ]
        )

    @property
    def phase_durations_s(self) -> dict[str, float]:
        return dict(self._phase_durations_s)

    def phase_start_reference(
        self, phase: Order9ObjectTaskPhase = Order9ObjectTaskPhase.APPROACH
    ):
        # The protected reset-bank admission path uses this only before formal
        # phase-zero restoration. The formal batched restore below supplies
        # each environment's actual accepted state.
        return self._references[0].phase_start_reference(phase)

    def phase_reset_references(self, **kwargs):
        return self._references[0].phase_reset_references(**kwargs)

    def batched_phase_start_reference(
        self, phase: Order9ObjectTaskPhase = Order9ObjectTaskPhase.APPROACH
    ) -> dict[str, torch.Tensor]:
        rows = [
            reference.phase_start_reference(phase) for reference in self._references
        ]

        def repeated(name: str) -> torch.Tensor:
            return torch.stack([getattr(row, name) for row in rows]).repeat_interleave(
                self._replay_count, dim=0
            )

        return {
            "body_pose_local": repeated("body_pose_local"),
            "body_twist": repeated("body_twist"),
            "joint_positions_rad": repeated("joint_positions_rad"),
            "joint_velocities_radps": repeated("joint_velocities_radps"),
            "object_pose_local": repeated("object_pose_local"),
            "object_twist": repeated("object_twist"),
        }

    def phase_goal_joint_positions(self, phase_index: torch.Tensor) -> torch.Tensor:
        self._validate_environment_batch(phase_index)
        cases = self._environment_case_index.to(device=phase_index.device)
        return self._phase_goal_joint_bank[cases, phase_index.long()]

    @torch.no_grad()
    def condition(
        self,
        target,
        *,
        phase_index: torch.Tensor,
        phase_elapsed_s: torch.Tensor,
        scene_origins: torch.Tensor,
    ):
        self._validate_environment_batch(phase_index)
        if phase_elapsed_s.shape != phase_index.shape or scene_origins.shape != (
            self._environment_count,
            3,
        ):
            raise ValueError("R1 batched nominal replay shapes differ")
        desired_pose = target.desired_robot_root_pose_world.clone()
        desired_twist = target.desired_robot_root_twist_world.clone()
        joint_position = target.nominal_joint_positions_rad.clone()
        joint_velocity = target.nominal_joint_velocities_radps.clone()
        desired_object_pose = target.desired_object_pose_world.clone()
        phase_goal = target.phase_goal_robot_root_pose_world.clone()
        phase_goal_object = target.phase_goal_object_pose_world.clone()
        phase_progress = target.phase_progress.clone()
        cases_by_environment = self._environment_case_index.to(
            device=phase_index.device
        )
        for runtime_phase, phase_value in enumerate(ORDER9_OBJECT_TASK_PHASES):
            ids = torch.nonzero(phase_index == runtime_phase, as_tuple=False).flatten()
            if not ids.numel():
                continue
            cases = cases_by_environment.index_select(0, ids)
            elapsed = phase_elapsed_s.index_select(0, ids)
            sampled = self._sample(self._banks[phase_value.value], cases, elapsed)
            origins = scene_origins.index_select(0, ids).to(
                device=self.device, dtype=self.dtype
            )
            pose = sampled[0]
            pose[:, :3] += origins
            object_pose = sampled[4]
            object_pose[:, :3] += origins
            goal = self._banks[phase_value.value]["body_pose"][cases, -1].clone()
            goal[:, :3] += origins
            object_goal = self._banks[phase_value.value]["object_pose"][
                cases, -1
            ].clone()
            object_goal[:, :3] += origins
            desired_pose.index_copy_(0, ids, pose)
            desired_twist.index_copy_(0, ids, sampled[1])
            joint_position.index_copy_(0, ids, sampled[2])
            joint_velocity.index_copy_(0, ids, sampled[3])
            desired_object_pose.index_copy_(0, ids, object_pose)
            phase_goal.index_copy_(0, ids, goal)
            phase_goal_object.index_copy_(0, ids, object_goal)
            duration = self._banks[phase_value.value]["duration"].index_select(0, cases)
            phase_progress.index_copy_(
                0, ids, (elapsed / duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
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

    def _validate_environment_batch(self, value: torch.Tensor) -> None:
        if value.shape != (self._environment_count,):
            raise ValueError("R1 batched nominal environment count differs")

    def _materialize_phase_bank(self, phase: str) -> dict[str, torch.Tensor]:
        rows = [reference._references[phase] for reference in self._references]
        maximum = max(int(row.times_s.numel()) for row in rows)

        def pad(value: torch.Tensor) -> torch.Tensor:
            missing = maximum - int(value.shape[0])
            if missing == 0:
                return value
            return torch.cat((value, value[-1:].expand(missing, *value.shape[1:])))

        times = []
        for row in rows:
            missing = maximum - int(row.times_s.numel())
            times.append(
                torch.cat(
                    (
                        row.times_s,
                        torch.full(
                            (missing,),
                            torch.inf,
                            device=self.device,
                            dtype=self.dtype,
                        ),
                    )
                )
            )
        return {
            "times": torch.stack(times),
            "length": torch.tensor(
                [row.times_s.numel() for row in rows],
                device=self.device,
                dtype=torch.long,
            ),
            "duration": torch.tensor(
                [float(row.times_s[-1].item()) for row in rows],
                device=self.device,
                dtype=self.dtype,
            ),
            "body_pose": torch.stack([pad(row.body_pose_world) for row in rows]),
            "body_twist": torch.stack([pad(row.body_twist_world) for row in rows]),
            "joint_position": torch.stack(
                [pad(row.joint_positions_rad) for row in rows]
            ),
            "joint_velocity": torch.stack(
                [pad(row.joint_velocities_radps) for row in rows]
            ),
            "object_pose": torch.stack([pad(row.object_pose_world) for row in rows]),
            "object_twist": torch.stack([pad(row.object_twist_world) for row in rows]),
        }

    def _sample(
        self,
        bank: dict[str, torch.Tensor],
        cases: torch.Tensor,
        elapsed_s: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        elapsed = elapsed_s.to(device=self.device, dtype=self.dtype).clamp_min(0.0)
        duration = bank["duration"].index_select(0, cases)
        elapsed = torch.minimum(elapsed, duration)
        times = bank["times"].index_select(0, cases)
        upper = torch.searchsorted(times, elapsed.unsqueeze(1), right=True).squeeze(1)
        maximum_upper = bank["length"].index_select(0, cases) - 1
        upper = torch.minimum(upper.clamp_min(1), maximum_upper)
        lower = upper - 1
        row = torch.arange(cases.numel(), device=self.device)
        t0 = times[row, lower]
        t1 = times[row, upper]
        alpha = ((elapsed - t0) / (t1 - t0).clamp_min(1.0e-12)).clamp(0.0, 1.0)

        def endpoints(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            values = bank[name].index_select(0, cases)
            return values[row, lower], values[row, upper]

        def lerp(name: str) -> torch.Tensor:
            start, end = endpoints(name)
            weight = alpha.reshape((alpha.shape[0],) + (1,) * (start.ndim - 1))
            return start + weight * (end - start)

        body_start, body_end = endpoints("body_pose")
        body_pose = lerp("body_pose")
        body_pose[:, 3:7] = _normalized_lerp_quaternion(
            body_start[:, 3:7], body_end[:, 3:7], alpha
        )
        object_start, object_end = endpoints("object_pose")
        object_pose = lerp("object_pose")
        object_pose[:, 3:7] = _normalized_lerp_quaternion(
            object_start[:, 3:7], object_end[:, 3:7], alpha
        )
        return (
            body_pose,
            lerp("body_twist"),
            lerp("joint_position"),
            lerp("joint_velocity"),
            object_pose,
            lerp("object_twist"),
        )


class Order9R1BatchedWrenchRangeReference:
    """Resolve each environment against its candidate-specific contact frame."""

    contract_version = ORDER9_R1_BATCHED_NOMINAL_RUNTIME_VERSION

    def __init__(
        self,
        *,
        trajectory,
        contact_candidate_set,
        selected_anchor_ids,
        object_id,
        authored_object_pose_world,
        batch_size,
        device,
        dtype=torch.float32,
    ) -> None:
        del trajectory, contact_candidate_set, object_id, authored_object_pose_world
        if batch_size != len(_ACTIVE_BUNDLES) * _ACTIVE_REPLAY_COUNT:
            raise ValueError("R1 batched wrench environment count differs")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.selected_anchor_ids = tuple(int(value) for value in selected_anchor_ids)
        references = []
        for bundle, task, wrench_trajectory in zip(
            _ACTIVE_BUNDLES, _ACTIVE_TASKS, _ACTIVE_WRENCH_TRAJECTORIES
        ):
            target = next(
                obj
                for obj in task.scene.objects
                if any(goal.target_entity_id == obj.object_id for goal in task.goals)
            )
            reference = Order9TensorWrenchRangeReference(
                trajectory=wrench_trajectory,
                contact_candidate_set=bundle.contact_candidate_set,
                selected_anchor_ids=self.selected_anchor_ids,
                object_id=target.object_id,
                authored_object_pose_world=target.pose_world,
                batch_size=1,
                device=self.device,
                dtype=self.dtype,
            )
            if reference.selected_anchor_ids != self.selected_anchor_ids:
                raise ValueError("R1 batched wrench anchors differ")
            references.append(reference)
        self._lower_bank = torch.stack([value._lower_bank for value in references])
        self._upper_bank = torch.stack([value._upper_bank for value in references])
        self._mask_bank = torch.stack([value._mask_bank for value in references])
        self._relative_pose_bank = torch.stack(
            [value._relative_pose_bank for value in references]
        )
        self._relative_normal_bank = torch.stack(
            [value._relative_normal_bank for value in references]
        )
        self._environment_case_index = torch.arange(
            len(references), device=self.device, dtype=torch.long
        ).repeat_interleave(_ACTIVE_REPLAY_COUNT)

    @torch.no_grad()
    def resolve(
        self,
        *,
        contact_schedule_index: torch.Tensor,
        object_pose_world: torch.Tensor,
    ) -> Order9TensorWrenchRange:
        if contact_schedule_index.shape != (self.batch_size,) or (
            object_pose_world.shape != (self.batch_size, 7)
        ):
            raise ValueError("R1 batched wrench input shapes differ")
        schedule = contact_schedule_index.to(device=self.device, dtype=torch.long)
        cases = self._environment_case_index
        relative_pose = self._relative_pose_bank[cases, schedule]
        object_pose = object_pose_world.to(device=self.device, dtype=self.dtype)
        pose_world = _compose_pose_tensor(object_pose[:, None, :], relative_pose)
        relative_normal = self._relative_normal_bank[cases, schedule]
        object_quaternion = _normalized_quaternion(object_pose[..., 3:7])
        normal_world = _quaternion_rotate(
            object_quaternion[:, None, :].expand(-1, len(self.selected_anchor_ids), -1),
            relative_normal,
        )
        normal_world = normal_world / torch.linalg.vector_norm(
            normal_world, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        return Order9TensorWrenchRange(
            lower_contact=self._lower_bank[cases, schedule],
            upper_contact=self._upper_bank[cases, schedule],
            bound_mask=self._mask_bank[cases, schedule],
            contact_frame_pose_world=pose_world,
            contact_normal_world=normal_world,
        )


def order9_r1_batched_task_for_environment(index: int) -> TaskSpec:
    if not 0 <= index < len(_ACTIVE_TASKS) * _ACTIVE_REPLAY_COUNT:
        raise IndexError("R1 batched task environment index is out of range")
    return _ACTIVE_TASKS[index // _ACTIVE_REPLAY_COUNT]


def _normalized_lerp_quaternion(
    start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    dot = (start * end).sum(dim=-1, keepdim=True)
    aligned_end = torch.where(dot < 0.0, -end, end)
    value = (1.0 - alpha.unsqueeze(-1)) * start + alpha.unsqueeze(-1) * aligned_end
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _normalized_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quaternion[..., :3]
    scalar = quaternion[..., 3:4]
    first = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector + scalar * first + torch.linalg.cross(q_vector, first, dim=-1)


def _compose_pose_tensor(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
    parent_position = parent[..., :3]
    parent_quaternion = _normalized_quaternion(parent[..., 3:7])
    child_position = child[..., :3]
    child_quaternion = _normalized_quaternion(child[..., 3:7])
    position = parent_position + _quaternion_rotate(parent_quaternion, child_position)
    ax, ay, az, aw = parent_quaternion.unbind(dim=-1)
    bx, by, bz, bw = child_quaternion.unbind(dim=-1)
    quaternion = torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )
    return torch.cat((position, _normalized_quaternion(quaternion)), dim=-1)


__all__ = [
    "ORDER9_R1_BATCHED_NOMINAL_RUNTIME_VERSION",
    "Order9R1BatchedNominalTensorReference",
    "Order9R1BatchedWrenchRangeReference",
    "configure_order9_r1_batched_nominal_runtime",
    "order9_r1_batched_task_for_environment",
]
