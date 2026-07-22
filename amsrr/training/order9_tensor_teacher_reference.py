from __future__ import annotations

"""Hash-bound C0 active-knot references for the fixed-nominal C1 runtime."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import LowLevelControlRecord
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
)
from amsrr.training.order9_teacher_collection import (
    load_order9_teacher_low_level_records,
)
from amsrr.utils.hashing import hash_file


ORDER9_TENSOR_TEACHER_REFERENCE_VERSION = (
    "order9_c0_nominal_active_knot_tensor_reference_v2_initial_physical_state"
)


@dataclass(frozen=True)
class Order9TensorTeacherReferenceSample:
    desired_body_pose_world: torch.Tensor
    desired_body_twist: torch.Tensor
    nominal_joint_positions_rad: torch.Tensor
    nominal_joint_velocities_radps: torch.Tensor
    desired_object_pose_world: torch.Tensor
    phase_goal_body_pose_world: torch.Tensor
    phase_goal_object_pose_world: torch.Tensor


class Order9TensorTeacherReference:
    """Vectorized interpolation over one verified nominal C0 episode."""

    reference_version = ORDER9_TENSOR_TEACHER_REFERENCE_VERSION

    def __init__(
        self,
        *,
        phase_progress: torch.Tensor,
        phase_lengths: torch.Tensor,
        desired_body_pose_world: torch.Tensor,
        desired_body_twist: torch.Tensor,
        nominal_joint_positions_rad: torch.Tensor,
        nominal_joint_velocities_radps: torch.Tensor,
        desired_object_pose_world: torch.Tensor,
        phase_goal_body_pose_world: torch.Tensor,
        phase_goal_object_pose_world: torch.Tensor,
        initial_module_pose_world: torch.Tensor,
        initial_module_twist_world: torch.Tensor,
        initial_object_pose_world: torch.Tensor,
        initial_object_twist_world: torch.Tensor,
        initial_joint_positions_rad: Mapping[str, float],
        initial_joint_velocities_radps: Mapping[str, float],
        module_ids: tuple[int, ...],
        joint_ids: tuple[str, ...],
        provenance: Mapping[str, Any],
    ) -> None:
        self.phase_progress = phase_progress
        self.phase_lengths = phase_lengths
        self.desired_body_pose_world = desired_body_pose_world
        self.desired_body_twist = desired_body_twist
        self.nominal_joint_positions_rad = nominal_joint_positions_rad
        self.nominal_joint_velocities_radps = nominal_joint_velocities_radps
        self.desired_object_pose_world = desired_object_pose_world
        self.phase_goal_body_pose_world = phase_goal_body_pose_world
        self.phase_goal_object_pose_world = phase_goal_object_pose_world
        self.initial_module_pose_world = initial_module_pose_world
        self.initial_module_twist_world = initial_module_twist_world
        self.initial_object_pose_world = initial_object_pose_world
        self.initial_object_twist_world = initial_object_twist_world
        self.initial_joint_positions_rad = {
            str(name): float(value)
            for name, value in initial_joint_positions_rad.items()
        }
        self.initial_joint_velocities_radps = {
            str(name): float(value)
            for name, value in initial_joint_velocities_radps.items()
        }
        self.module_ids = module_ids
        self.joint_ids = joint_ids
        self.provenance = dict(provenance)
        self._validate()

    @classmethod
    def from_records(
        cls,
        records: Sequence[LowLevelControlRecord],
        *,
        module_ids: Sequence[int],
        joint_ids: Sequence[str],
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        provenance: Mapping[str, Any] | None = None,
    ) -> "Order9TensorTeacherReference":
        if not records:
            raise SchemaValidationError("Order9 teacher reference records are empty")
        resolved_modules = tuple(int(value) for value in module_ids)
        resolved_joints = tuple(str(value) for value in joint_ids)
        if (
            not resolved_modules
            or len(set(resolved_modules)) != len(resolved_modules)
            or not resolved_joints
            or len(set(resolved_joints)) != len(resolved_joints)
        ):
            raise SchemaValidationError(
                "Order9 teacher reference module/joint ids must be unique"
            )
        phase_index = {
            phase.value: index for index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES)
        }
        grouped: list[list[LowLevelControlRecord]] = [
            [] for _ in ORDER9_OBJECT_TASK_PHASES
        ]
        graph_hash = records[0].runtime_observation.morphology_graph.stable_hash()
        for record in records:
            record.validate()
            label = record.runtime_observation.task_progress.phase_label
            if label not in phase_index:
                continue
            if record.runtime_observation.morphology_graph.stable_hash() != graph_hash:
                raise SchemaValidationError(
                    "Order9 teacher reference mixes morphology graphs"
                )
            guards = [
                str(guard.get("phase_label"))
                for guard in record.active_knot.guard_conditions
                if guard.get("type") == "order9_task_phase"
            ]
            if guards != [label]:
                raise SchemaValidationError(
                    "Order9 teacher reference active-knot phase differs"
                )
            grouped[phase_index[label]].append(record)
        if any(not rows for rows in grouped):
            missing = [
                ORDER9_OBJECT_TASK_PHASES[index].value
                for index, rows in enumerate(grouped)
                if not rows
            ]
            raise SchemaValidationError(
                f"Order9 teacher reference lacks phases: {missing}"
            )

        maximum_rows = max(len(rows) for rows in grouped)
        phase_count = len(grouped)
        module_count = len(resolved_modules)
        joint_count = len(resolved_joints)
        target_device = torch.device(device)
        progress = torch.full(
            (phase_count, maximum_rows),
            float("inf"),
            device=target_device,
            dtype=dtype,
        )
        body_pose = torch.zeros(
            (phase_count, maximum_rows, 7), device=target_device, dtype=dtype
        )
        body_twist = torch.zeros(
            (phase_count, maximum_rows, 6), device=target_device, dtype=dtype
        )
        joint_position = torch.zeros(
            (phase_count, maximum_rows, module_count, joint_count),
            device=target_device,
            dtype=dtype,
        )
        joint_velocity = torch.zeros_like(joint_position)
        object_pose = torch.zeros_like(body_pose)
        lengths = torch.tensor(
            [len(rows) for rows in grouped],
            device=target_device,
            dtype=torch.long,
        )
        for phase, rows in enumerate(grouped):
            previous_progress = -math.inf
            for row_index, record in enumerate(rows):
                ratio = float(record.runtime_observation.task_progress.progress_ratio)
                if not math.isfinite(ratio) or ratio < previous_progress:
                    raise SchemaValidationError(
                        "Order9 teacher reference phase progress is not monotonic"
                    )
                previous_progress = ratio
                target = record.active_knot.centroidal_target
                posture = record.active_knot.posture_target
                if (
                    target is None
                    or target.com_pos_world is None
                    or target.body_orientation_world is None
                    or posture is None
                ):
                    raise SchemaValidationError(
                        "Order9 teacher reference lacks centroidal/posture targets"
                    )
                linear = target.com_vel_world or (0.0, 0.0, 0.0)
                objects = record.runtime_observation.object_states
                if len(objects) != 1:
                    raise SchemaValidationError(
                        "Order9 nominal teacher reference requires one object state"
                    )
                progress[phase, row_index] = ratio
                body_pose[phase, row_index] = torch.tensor(
                    (*target.com_pos_world, *target.body_orientation_world),
                    device=target_device,
                    dtype=dtype,
                )
                body_twist[phase, row_index, :3] = torch.tensor(
                    linear, device=target_device, dtype=dtype
                )
                object_pose[phase, row_index] = torch.tensor(
                    objects[0].pose_world, device=target_device, dtype=dtype
                )
                for module_index, module_id in enumerate(resolved_modules):
                    for joint_index, joint_id in enumerate(resolved_joints):
                        global_id = f"module_{module_id}:{joint_id}"
                        try:
                            q = posture.joint_pos_target[global_id]
                            qdot = posture.joint_vel_target[global_id]
                        except KeyError as exc:
                            raise SchemaValidationError(
                                "Order9 teacher reference does not cover "
                                f"{global_id}"
                            ) from exc
                        joint_position[
                            phase, row_index, module_index, joint_index
                        ] = float(q)
                        joint_velocity[
                            phase, row_index, module_index, joint_index
                        ] = float(qdot)
            last = len(rows) - 1
            if last + 1 < maximum_rows:
                body_pose[phase, last + 1 :] = body_pose[phase, last]
                body_twist[phase, last + 1 :] = body_twist[phase, last]
                joint_position[phase, last + 1 :] = joint_position[phase, last]
                joint_velocity[phase, last + 1 :] = joint_velocity[phase, last]
                object_pose[phase, last + 1 :] = object_pose[phase, last]
        body_pose[..., 3:7] = _normalize_quaternion(body_pose[..., 3:7])
        object_pose[..., 3:7] = _normalize_quaternion(object_pose[..., 3:7])
        initial_record = grouped[0][0]
        module_state_by_id = {
            state.module_id: state
            for state in initial_record.runtime_observation.module_states
        }
        if set(module_state_by_id) != set(resolved_modules):
            raise SchemaValidationError(
                "Order9 teacher initial module identity differs"
            )
        initial_module_pose = torch.tensor(
            [module_state_by_id[module_id].pose_world for module_id in resolved_modules],
            device=target_device,
            dtype=dtype,
        )
        initial_module_twist = torch.tensor(
            [module_state_by_id[module_id].twist_world for module_id in resolved_modules],
            device=target_device,
            dtype=dtype,
        )
        initial_module_pose[:, 3:7] = _normalize_quaternion(
            initial_module_pose[:, 3:7]
        )
        initial_joint_position: dict[str, float] = {}
        initial_joint_velocity: dict[str, float] = {}
        for module_id in resolved_modules:
            state = module_state_by_id[module_id]
            if set(state.joint_positions) != set(state.joint_velocities):
                raise SchemaValidationError(
                    "Order9 teacher initial joint position/velocity identity differs"
                )
            for joint_id, value in state.joint_positions.items():
                name = f"module_{module_id}__{joint_id}"
                initial_joint_position[name] = float(value)
                initial_joint_velocity[name] = float(state.joint_velocities[joint_id])
        initial_objects = initial_record.runtime_observation.object_states
        if len(initial_objects) != 1:
            raise SchemaValidationError(
                "Order9 teacher initial state requires one object"
            )
        initial_object_pose = torch.tensor(
            initial_objects[0].pose_world,
            device=target_device,
            dtype=dtype,
        )
        initial_object_pose[3:7] = _normalize_quaternion(
            initial_object_pose[3:7]
        )
        initial_object_twist = torch.tensor(
            initial_objects[0].twist_world,
            device=target_device,
            dtype=dtype,
        )
        row = torch.arange(phase_count, device=target_device)
        last = lengths - 1
        return cls(
            phase_progress=progress,
            phase_lengths=lengths,
            desired_body_pose_world=body_pose,
            desired_body_twist=body_twist,
            nominal_joint_positions_rad=joint_position,
            nominal_joint_velocities_radps=joint_velocity,
            desired_object_pose_world=object_pose,
            phase_goal_body_pose_world=body_pose[row, last],
            phase_goal_object_pose_world=object_pose[row, last],
            initial_module_pose_world=initial_module_pose,
            initial_module_twist_world=initial_module_twist,
            initial_object_pose_world=initial_object_pose,
            initial_object_twist_world=initial_object_twist,
            initial_joint_positions_rad=initial_joint_position,
            initial_joint_velocities_radps=initial_joint_velocity,
            module_ids=resolved_modules,
            joint_ids=resolved_joints,
            provenance={
                "reference_version": ORDER9_TENSOR_TEACHER_REFERENCE_VERSION,
                "source_graph_hash": graph_hash,
                "terminal_progress_by_phase": {
                    ORDER9_OBJECT_TASK_PHASES[index].value: float(
                        progress[index, lengths[index] - 1].item()
                    )
                    for index in range(phase_count)
                },
                **dict(provenance or {}),
            },
        )

    def sample(
        self,
        *,
        phase_index: torch.Tensor,
        phase_progress: torch.Tensor,
        position_offset_world: torch.Tensor | None = None,
    ) -> Order9TensorTeacherReferenceSample:
        batch = phase_index.shape[0]
        if phase_index.shape != (batch,) or phase_progress.shape != (batch,):
            raise ValueError("Order9 teacher reference phase inputs must be [batch]")
        if phase_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("Order9 teacher reference phase index must be integral")
        if bool((phase_index < 0).any()) or bool(
            (phase_index >= len(ORDER9_OBJECT_TASK_PHASES)).any()
        ):
            raise ValueError("Order9 teacher reference phase index is invalid")
        if phase_progress.device != self.phase_progress.device:
            raise ValueError("Order9 teacher reference device differs")
        phase = phase_index.long()
        boundaries = self.phase_progress.index_select(0, phase)
        lengths = self.phase_lengths.index_select(0, phase)
        raw_upper = torch.searchsorted(
            boundaries.contiguous(), phase_progress.unsqueeze(-1), right=False
        ).squeeze(-1)
        upper = torch.minimum(raw_upper, lengths - 1)
        lower = (upper - 1).clamp_min(0)
        batch_index = torch.arange(batch, device=phase.device)
        lower_progress = boundaries[batch_index, lower]
        upper_progress = boundaries[batch_index, upper]
        denominator = upper_progress - lower_progress
        alpha = torch.where(
            (upper == lower) | (denominator.abs() <= 1.0e-12),
            torch.zeros_like(phase_progress),
            (phase_progress - lower_progress) / denominator,
        ).clamp(0.0, 1.0)

        def interpolate(values: torch.Tensor) -> torch.Tensor:
            start = values[phase, lower]
            end = values[phase, upper]
            weight = alpha.reshape((batch,) + (1,) * (start.ndim - 1))
            return start + weight * (end - start)

        body_pose = interpolate(self.desired_body_pose_world)
        body_pose[:, 3:7] = _interpolate_quaternion(
            self.desired_body_pose_world[phase, lower, 3:7],
            self.desired_body_pose_world[phase, upper, 3:7],
            alpha,
        )
        object_pose = interpolate(self.desired_object_pose_world)
        object_pose[:, 3:7] = _interpolate_quaternion(
            self.desired_object_pose_world[phase, lower, 3:7],
            self.desired_object_pose_world[phase, upper, 3:7],
            alpha,
        )
        goal_body = self.phase_goal_body_pose_world.index_select(0, phase).clone()
        goal_object = self.phase_goal_object_pose_world.index_select(0, phase).clone()
        if position_offset_world is not None:
            if position_offset_world.shape != (batch, 3):
                raise ValueError(
                    "Order9 teacher reference position offset must be [batch, 3]"
                )
            for pose in (body_pose, object_pose, goal_body, goal_object):
                pose[:, :3] += position_offset_world
        return Order9TensorTeacherReferenceSample(
            desired_body_pose_world=body_pose,
            desired_body_twist=interpolate(self.desired_body_twist),
            nominal_joint_positions_rad=interpolate(
                self.nominal_joint_positions_rad
            ),
            nominal_joint_velocities_radps=interpolate(
                self.nominal_joint_velocities_radps
            ),
            desired_object_pose_world=object_pose,
            phase_goal_body_pose_world=goal_body,
            phase_goal_object_pose_world=goal_object,
        )

    def _validate(self) -> None:
        phase_count = len(ORDER9_OBJECT_TASK_PHASES)
        maximum_rows = self.phase_progress.shape[1]
        expected_prefix = (phase_count, maximum_rows)
        expected = {
            "phase_lengths": (phase_count,),
            "desired_body_pose_world": (*expected_prefix, 7),
            "desired_body_twist": (*expected_prefix, 6),
            "nominal_joint_positions_rad": (
                *expected_prefix,
                len(self.module_ids),
                len(self.joint_ids),
            ),
            "nominal_joint_velocities_radps": (
                *expected_prefix,
                len(self.module_ids),
                len(self.joint_ids),
            ),
            "desired_object_pose_world": (*expected_prefix, 7),
            "phase_goal_body_pose_world": (phase_count, 7),
            "phase_goal_object_pose_world": (phase_count, 7),
            "initial_module_pose_world": (len(self.module_ids), 7),
            "initial_module_twist_world": (len(self.module_ids), 6),
            "initial_object_pose_world": (7,),
            "initial_object_twist_world": (6,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise SchemaValidationError(
                    f"Order9 teacher reference {name} shape differs"
                )
        if bool((self.phase_lengths < 1).any()) or bool(
            (self.phase_lengths > maximum_rows).any()
        ):
            raise SchemaValidationError("Order9 teacher reference lengths are invalid")
        finite_tensors = (
            self.desired_body_pose_world,
            self.desired_body_twist,
            self.nominal_joint_positions_rad,
            self.nominal_joint_velocities_radps,
            self.desired_object_pose_world,
            self.phase_goal_body_pose_world,
            self.phase_goal_object_pose_world,
            self.initial_module_pose_world,
            self.initial_module_twist_world,
            self.initial_object_pose_world,
            self.initial_object_twist_world,
        )
        if any(not bool(torch.isfinite(value).all()) for value in finite_tensors):
            raise SchemaValidationError("Order9 teacher reference contains non-finite data")
        if set(self.initial_joint_positions_rad) != set(
            self.initial_joint_velocities_radps
        ) or any(
            not math.isfinite(value)
            for values in (
                self.initial_joint_positions_rad,
                self.initial_joint_velocities_radps,
            )
            for value in values.values()
        ):
            raise SchemaValidationError(
                "Order9 teacher reference initial joint state is invalid"
            )


def load_order9_nominal_tensor_teacher_reference(
    dataset_manifest_path: str | Path,
    *,
    expected_dataset_manifest_sha256: str,
    repository_root: str | Path,
    module_ids: Sequence[int],
    joint_ids: Sequence[str],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Order9TensorTeacherReference:
    """Resolve the unique exact-nominal C0 source bound by a C1 checkpoint."""

    source = Path(dataset_manifest_path).resolve()
    actual_dataset_sha256 = hash_file(source)
    if actual_dataset_sha256 != expected_dataset_manifest_sha256:
        raise SchemaValidationError("Order9 C0 dataset manifest hash mismatch")
    try:
        dataset = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"failed to load Order9 C0 manifest: {exc}") from exc
    if not isinstance(dataset, Mapping):
        raise SchemaValidationError("Order9 C0 dataset manifest must be a mapping")
    raw_paths = dataset.get("source_archive_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise SchemaValidationError("Order9 C0 manifest has no source episodes")
    root = Path(repository_root).resolve()
    nominal_paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaValidationError(
                f"failed to load Order9 C0 episode manifest: {exc}"
            ) from exc
        if _is_exact_nominal_episode(payload):
            nominal_paths.append(path.resolve())
    if len(nominal_paths) != 1:
        raise SchemaValidationError(
            "Order9 C0 dataset must contain exactly one exact nominal episode"
        )
    episode_path = nominal_paths[0]
    manifest, records = load_order9_teacher_low_level_records(episode_path)
    if not manifest.success or manifest.random_seed != 9009:
        raise SchemaValidationError(
            "Order9 nominal teacher reference must be successful seed 9009"
        )
    return Order9TensorTeacherReference.from_records(
        records,
        module_ids=module_ids,
        joint_ids=joint_ids,
        device=device,
        dtype=dtype,
        provenance={
            "dataset_manifest_path": str(source),
            "dataset_manifest_sha256": actual_dataset_sha256,
            "episode_manifest_path": str(episode_path),
            "episode_manifest_sha256": hash_file(episode_path),
            "episode_id": manifest.episode_id,
            "task_id": manifest.task_spec.task_id,
            "random_seed": manifest.random_seed,
            "low_level_shard_sha256": manifest.low_level_shard_sha256,
            "record_count": len(records),
        },
    )


def _is_exact_nominal_episode(payload: object) -> bool:
    if not isinstance(payload, Mapping) or payload.get("success") is not True:
        return False
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    size = metadata.get("object_size_m")
    return (
        int(payload.get("random_seed", -1)) == 9009
        and _close(metadata.get("object_mass_kg"), 1.0)
        and isinstance(size, list)
        and len(size) == 3
        and all(_close(value, expected) for value, expected in zip(size, (0.3, 0.4, 0.15)))
        and _close(metadata.get("object_friction"), 0.6)
        and _close(metadata.get("initial_object_standoff_m"), 0.5)
        and _close(metadata.get("selected_gripper_friction"), 4.5)
        and _close(
            metadata.get("selected_gripper_contact_stiffness_n_per_m"), 7500.0
        )
        and _close(
            metadata.get("selected_gripper_contact_damping_n_s_per_m"), 75.0
        )
    )


def _close(value: object, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1.0e-9
    )


def _normalize_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _interpolate_quaternion(
    start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    aligned = torch.where(
        (start * end).sum(dim=-1, keepdim=True) < 0.0, -end, end
    )
    return _normalize_quaternion(
        start + alpha.unsqueeze(-1) * (aligned - start)
    )


__all__ = [
    "ORDER9_TENSOR_TEACHER_REFERENCE_VERSION",
    "Order9TensorTeacherReference",
    "Order9TensorTeacherReferenceSample",
    "load_order9_nominal_tensor_teacher_reference",
]
