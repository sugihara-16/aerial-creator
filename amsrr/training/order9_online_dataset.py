from __future__ import annotations

"""Hash-bound writer for one fresh Order 9 on-policy rollout generation."""

import gzip
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    P4_3DatasetManifest,
    SequentialDesignTrajectoryRecord,
    TrajectorySourceKind,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_ON_POLICY_DATASET_VERSION = (
    "order9_on_policy_rollout_generation_v2_deterministic_fast_gzip"
)
ORDER9_ON_POLICY_GZIP_COMPRESSLEVEL = 1


def write_order9_on_policy_dataset(
    output_dir: str | Path,
    *,
    generation_id: str,
    low_level_records: Sequence[LowLevelControlRecord] = (),
    trajectory_records: Sequence[InteractionTrajectoryRecord] = (),
    design_records: Sequence[SequentialDesignTrajectoryRecord] = (),
    task_specs: Mapping[str, TaskSpec],
    behavior_checkpoint_sha256_by_family: Mapping[str, str],
    source_isaac_artifact_paths: Sequence[str | Path],
    on_policy_environment_step_count: int,
    random_seeds: Sequence[int],
    config_hash: str,
    robot_model_hash: str,
    urdf_hash: str,
    thrust_model_hash: str,
    simulator_version: str,
    simulator_hash: str,
    metadata: Mapping[str, object] | None = None,
) -> P4_3DatasetManifest:
    """Persist exactly one stochastic generation without mixing checkpoints."""

    if not generation_id:
        raise SchemaValidationError("Order9 rollout generation_id must be non-empty")
    if on_policy_environment_step_count < 1:
        raise SchemaValidationError("Order9 rollout environment-step count must be positive")
    records: list[object] = [
        *low_level_records,
        *trajectory_records,
        *design_records,
    ]
    if not records:
        raise SchemaValidationError("Order9 on-policy generation has no policy records")
    sources = [Path(path) for path in source_isaac_artifact_paths]
    if not sources:
        raise SchemaValidationError("Order9 on-policy generation requires raw Isaac artifacts")
    source_hashes = {str(path): hash_file(path) for path in sources}
    _validate_behavior_lineage(
        low_level_records,
        trajectory_records,
        design_records,
        behavior_checkpoint_sha256_by_family,
    )
    metadata_values = dict(metadata or {})
    topology_randomized = bool(metadata_values.get("topology_randomized", True))
    split_by_task: dict[str, DatasetSplit] = {}
    structural_owner: dict[str, DatasetSplit] = {}
    episode_ids: set[str] = set()
    for record in records:
        task_id = str(getattr(record, "task_id"))
        split = DatasetSplit(getattr(record, "split"))
        previous = split_by_task.setdefault(task_id, split)
        if previous != split:
            raise SchemaValidationError("Order9 on-policy task crosses dataset splits")
        if task_id not in task_specs:
            raise SchemaValidationError(
                f"Order9 on-policy task spec is missing for {task_id!r}"
            )
        if task_specs[task_id].task_id != task_id:
            raise SchemaValidationError("Order9 on-policy task-spec identity mismatch")
        episode_ids.add(str(getattr(record, "episode_id")))
        morphology = _record_morphology(record)
        if morphology is not None and topology_randomized:
            structural_hash = morphology_structural_hash(morphology)
            owner = structural_owner.setdefault(structural_hash, split)
            if owner != split:
                raise SchemaValidationError(
                    "Order9 on-policy morphology structural hash crosses splits"
                )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    shards: list[DatasetShard] = []
    record_groups = (
        (DatasetKind.LOW_LEVEL_CONTROL, low_level_records),
        (DatasetKind.INTERACTION_TRAJECTORY, trajectory_records),
        (DatasetKind.DESIGN_ACTION_TRAJECTORY, design_records),
    )
    for kind, values in record_groups:
        for split in DatasetSplit:
            selected = [record for record in values if record.split == split]
            if not selected:
                continue
            shard_path = target / f"{kind.value}_{split.value}.jsonl.gz"
            _atomic_write_jsonl_gzip(shard_path, selected)
            shards.append(
                DatasetShard(
                    dataset_kind=kind,
                    split=split,
                    path=shard_path.name,
                    record_count=len(selected),
                    sha256=hash_file(shard_path),
                )
            )
    split_tasks = {
        split: sorted(
            task_id for task_id, owner in split_by_task.items() if owner == split
        )
        for split in DatasetSplit
    }
    used_tasks = {task_id: task_specs[task_id] for task_id in split_by_task}
    record_counts = {kind.value: 0 for kind in DatasetKind}
    for kind, values in record_groups:
        record_counts[kind.value] = len(values)
    manifest = P4_3DatasetManifest(
        dataset_id=f"order9-on-policy-{generation_id}",
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=[str(path) for path in sources],
        source_episode_ids=sorted(episode_ids),
        train_task_ids=split_tasks[DatasetSplit.TRAIN],
        validation_task_ids=split_tasks[DatasetSplit.VALIDATION],
        held_out_task_ids=split_tasks[DatasetSplit.HELD_OUT],
        shards=shards,
        record_counts=record_counts,
        source_hash=stable_hash(source_hashes),
        config_hash=config_hash,
        robot_model_hash=robot_model_hash,
        urdf_hash=urdf_hash,
        thrust_model_hash=thrust_model_hash,
        task_hashes={task_id: task.stable_hash() for task_id, task in used_tasks.items()},
        geometry_hashes={
            f"{task_id}:{geometry.geometry_id}": geometry.stable_hash()
            for task_id, task in used_tasks.items()
            for geometry in task.scene.geometry_library
        },
        random_seeds=list(random_seeds),
        simulator_version=simulator_version,
        simulator_hash=simulator_hash,
        metadata={
            "dataset_version": ORDER9_ON_POLICY_DATASET_VERSION,
            "generation_id": generation_id,
            "one_fresh_generation": True,
            "on_policy_environment_step_count": on_policy_environment_step_count,
            "behavior_checkpoint_sha256_by_family": dict(
                behavior_checkpoint_sha256_by_family
            ),
            "source_isaac_artifact_sha256": source_hashes,
            "task_disjoint_splits": True,
            "structural_hash_disjoint_splits": topology_randomized,
            "fixed_topology_task_split": not topology_randomized,
            "raw_contact_actor_input": False,
            "privileged_contact_role": "critic_reward_safety_only",
            "gzip_shards": True,
            "gzip_compresslevel": ORDER9_ON_POLICY_GZIP_COMPRESSLEVEL,
            "gzip_mtime_unix_s": 0,
            "gzip_original_filename_stored": False,
            **metadata_values,
        },
    )
    manifest.validate()
    _atomic_write_text(target / "manifest.json", manifest.to_json(indent=2) + "\n")
    return manifest


def _validate_behavior_lineage(
    low_level: Sequence[LowLevelControlRecord],
    high_level: Sequence[InteractionTrajectoryRecord],
    design: Sequence[SequentialDesignTrajectoryRecord],
    expected: Mapping[str, str],
) -> None:
    groups: tuple[tuple[str, Iterable[object]], ...] = (
        ("pi_l", (record.behavior_trace for record in low_level)),
        ("pi_h", (record.behavior_trace for record in high_level)),
        (
            "pi_d",
            (
                step.behavior_trace
                for record in design
                for step in record.steps
            ),
        ),
    )
    for family, traces in groups:
        values = list(traces)
        if not values:
            continue
        checkpoint = expected.get(family)
        if checkpoint is None:
            raise SchemaValidationError(
                f"Order9 on-policy behavior hash is missing for {family}"
            )
        if len(checkpoint) != 64:
            raise SchemaValidationError("Order9 on-policy behavior hash is invalid")
        for trace in values:
            if (
                trace is None
                or not trace.stochastic
                or trace.policy_family != family
                or trace.policy_checkpoint_sha256 != checkpoint
            ):
                raise SchemaValidationError(
                    f"Order9 on-policy {family} trace/checkpoint contract mismatch"
                )
    for family, records in (("pi_h", high_level), ("pi_d", design)):
        checkpoint = expected.get(family)
        for record in records:
            provenance = record.trajectory_provenance
            if (
                provenance is None
                or provenance.source_kind != TrajectorySourceKind.LEARNED_POLICY
                or provenance.policy_checkpoint_sha256 != checkpoint
            ):
                raise SchemaValidationError(
                    f"Order9 on-policy {family} provenance/checkpoint mismatch"
                )


def _record_morphology(record: object):
    if isinstance(record, LowLevelControlRecord):
        return record.runtime_observation.morphology_graph
    if isinstance(record, InteractionTrajectoryRecord):
        return record.morphology_graph
    if isinstance(record, SequentialDesignTrajectoryRecord):
        return (
            None
            if record.design_output is None
            else record.design_output.target_morphology
        )
    raise TypeError(type(record).__name__)


def _atomic_write_jsonl_gzip(path: Path, records: Sequence[SchemaBase]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=ORDER9_ON_POLICY_GZIP_COMPRESSLEVEL,
                fileobj=raw_handle,
                mtime=0,
            ) as handle:
                for record in records:
                    record.validate()
                    handle.write(record.to_json().encode("utf-8"))
                    handle.write(b"\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "ORDER9_ON_POLICY_GZIP_COMPRESSLEVEL",
    "ORDER9_ON_POLICY_DATASET_VERSION",
    "write_order9_on_policy_dataset",
]
