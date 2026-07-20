from __future__ import annotations

"""Verified dataset I/O and stage-specific replay contracts for Order 9."""

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from amsrr.logging.episode_archive import EpisodeArchive
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetSplit,
    DesignOutcomeRecord,
    HighLevelTransitionKind,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    P4_3DatasetManifest,
    SequentialDesignTrajectoryRecord,
    TrajectorySourceKind,
)
from amsrr.training.order9_curriculum import (
    Order9CurriculumStage,
    Order9LearningMode,
    Order9LearningTarget,
)
from amsrr.utils.hashing import hash_file


ORDER9_DATASET_IO_VERSION = "order9_verified_dataset_io_v1"


@dataclass(frozen=True)
class Order9DatasetBundle:
    manifest: P4_3DatasetManifest
    manifest_path: str
    manifest_sha256: str
    low_level_records: tuple[LowLevelControlRecord, ...]
    trajectory_records: tuple[InteractionTrajectoryRecord, ...]
    sequential_design_records: tuple[SequentialDesignTrajectoryRecord, ...]
    design_outcome_records: tuple[DesignOutcomeRecord, ...]
    rollout_archives: tuple[EpisodeArchive, ...]
    verified_shard_sha256: dict[str, str]


@dataclass
class Order9DatasetStageValidation(SchemaBase):
    stage_id: str
    valid: bool
    failures: list[str]
    record_count: int
    episode_count: int
    task_count: int
    stochastic_record_count: int
    deterministic_teacher_record_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.valid == bool(self.failures):
            raise SchemaValidationError(
                "Order9DatasetStageValidation valid flag conflicts with failures"
            )
        for name in (
            "record_count",
            "episode_count",
            "task_count",
            "stochastic_record_count",
            "deterministic_teacher_record_count",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaValidationError(
                    f"Order9DatasetStageValidation.{name} must be non-negative"
                )


def load_order9_dataset(path: str | Path) -> Order9DatasetBundle:
    manifest_path = _manifest_path(path)
    manifest = P4_3DatasetManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.schema_version != P4_3_DATASET_SCHEMA_VERSION:
        raise SchemaValidationError(
            "Order9 training rejects the legacy P4.3 dataset schema"
        )
    low_level: list[LowLevelControlRecord] = []
    trajectories: list[InteractionTrajectoryRecord] = []
    designs: list[SequentialDesignTrajectoryRecord] = []
    outcomes: list[DesignOutcomeRecord] = []
    archives: list[EpisodeArchive] = []
    verified: dict[str, str] = {}
    record_ids: set[tuple[str, str]] = set()
    for shard in manifest.shards:
        source = _resolve_shard_path(shard.path, manifest_path.parent)
        digest = hash_file(source)
        if digest != shard.sha256:
            raise SchemaValidationError(
                f"Order9 dataset shard SHA-256 mismatch: {source}"
            )
        rows = _jsonl_rows(source)
        if len(rows) != shard.record_count:
            raise SchemaValidationError(
                f"Order9 dataset shard count mismatch: {source}"
            )
        verified[str(source)] = digest
        for raw in rows:
            record = _record_from_payload(shard.dataset_kind, raw)
            record_split = getattr(record, "split", None)
            if (
                shard.split is not None
                and record_split is not None
                and record_split != shard.split
            ):
                raise SchemaValidationError(
                    f"Order9 dataset record split mismatch in {source}"
                )
            record_id = getattr(record, "record_id", None)
            if record_id is not None:
                identity = (shard.dataset_kind.value, str(record_id))
                if identity in record_ids:
                    raise SchemaValidationError(
                        f"Order9 dataset duplicate record id: {identity}"
                    )
                record_ids.add(identity)
            if isinstance(record, LowLevelControlRecord):
                low_level.append(record)
            elif isinstance(record, InteractionTrajectoryRecord):
                trajectories.append(record)
            elif isinstance(record, SequentialDesignTrajectoryRecord):
                designs.append(record)
            elif isinstance(record, DesignOutcomeRecord):
                outcomes.append(record)
            elif isinstance(record, EpisodeArchive):
                archives.append(record)
    _validate_manifest_counts(
        manifest,
        low_level=low_level,
        trajectories=trajectories,
        designs=designs,
        outcomes=outcomes,
        archives=archives,
    )
    _validate_task_split_membership(
        manifest,
        [*low_level, *trajectories, *designs, *outcomes],
    )
    return Order9DatasetBundle(
        manifest=manifest,
        manifest_path=str(manifest_path),
        manifest_sha256=hash_file(manifest_path),
        low_level_records=tuple(low_level),
        trajectory_records=tuple(trajectories),
        sequential_design_records=tuple(designs),
        design_outcome_records=tuple(outcomes),
        rollout_archives=tuple(archives),
        verified_shard_sha256=verified,
    )


def validate_order9_dataset_for_stage(
    bundle: Order9DatasetBundle,
    stage: Order9CurriculumStage,
    *,
    behavior_checkpoint_sha256: str | Mapping[str, str] | None = None,
) -> Order9DatasetStageValidation:
    records = _records_for_stage(bundle, stage)
    failures: list[str] = []
    if not records:
        failures.append("required_record_kind_empty")
    train_records = [
        record for record in records if getattr(record, "split", None) == DatasetSplit.TRAIN
    ]
    validation_records = [
        record
        for record in records
        if getattr(record, "split", None) == DatasetSplit.VALIDATION
    ]
    held_out_records = [
        record
        for record in records
        if getattr(record, "split", None) == DatasetSplit.HELD_OUT
    ]
    if stage.held_out_only:
        if not held_out_records or train_records or validation_records:
            failures.append("held_out_only_split_contract")
    elif stage.learning_mode != Order9LearningMode.COLLECTION:
        if not train_records:
            failures.append("train_split_empty")
        if not validation_records:
            failures.append("validation_split_empty")

    deterministic = 0
    stochastic = 0
    for record in records:
        behavior = _behavior_trace(record)
        provenance = _provenance(record)
        module_count = _record_module_count(record)
        if module_count is not None and not (
            stage.min_modules <= module_count <= stage.max_modules
        ):
            failures.append("morphology_module_count_outside_stage")
        if behavior is not None and behavior.stochastic:
            stochastic += 1
        if provenance is not None and provenance.source_kind == TrajectorySourceKind.DETERMINISTIC_TEACHER:
            deterministic += 1
        elif behavior is not None and not behavior.stochastic:
            deterministic += 1

        if stage.learning_mode == Order9LearningMode.BEHAVIOR_CLONING:
            if behavior is not None and behavior.stochastic:
                failures.append("bc_contains_stochastic_behavior")
            if isinstance(record, LowLevelControlRecord) and behavior is None:
                failures.append("pi_l_bc_missing_teacher_behavior")
            if isinstance(record, LowLevelControlRecord):
                phase_fields = (
                    record.task_type,
                    record.task_adapter_id,
                    record.phase_index,
                    record.phase_count,
                )
                if stage.phase_conditioned_actor_required and any(
                    value is None for value in phase_fields
                ):
                    failures.append("pi_l_bc_missing_task_phase_context")
                if (
                    record.task_adapter_id is not None
                    and record.task_adapter_id not in stage.task_adapter_ids
                ):
                    failures.append("task_adapter_outside_stage")
                if record.reward is None:
                    failures.append("pi_l_bc_missing_reward")
            if isinstance(record, InteractionTrajectoryRecord):
                if provenance is None or provenance.source_kind != TrajectorySourceKind.DETERMINISTIC_TEACHER:
                    failures.append("pi_h_bc_missing_teacher_provenance")
                result = record.trajectory_feasibility_result
                if result is None or not result.feasible:
                    failures.append("pi_h_bc_missing_feasible_c_h_result")
            if isinstance(record, SequentialDesignTrajectoryRecord):
                if record.trajectory_provenance.source_kind != TrajectorySourceKind.DETERMINISTIC_TEACHER:
                    failures.append("pi_d_bc_missing_teacher_provenance")
        elif stage.learning_mode == Order9LearningMode.PPO:
            if behavior is None or not behavior.stochastic:
                failures.append("ppo_missing_stochastic_behavior")
            elif behavior_checkpoint_sha256 is not None:
                expected = (
                    behavior_checkpoint_sha256.get(behavior.policy_family)
                    if isinstance(behavior_checkpoint_sha256, Mapping)
                    else behavior_checkpoint_sha256
                )
                if expected is None:
                    failures.append("ppo_behavior_checkpoint_family_missing")
                elif behavior.policy_checkpoint_sha256 != expected:
                    failures.append("ppo_behavior_checkpoint_mismatch")

    _validate_episode_boundaries(records, failures)
    unique_failures = sorted(set(failures))
    episode_ids = {
        str(getattr(record, "episode_id"))
        for record in records
        if getattr(record, "episode_id", None) is not None
    }
    task_ids = {str(getattr(record, "task_id")) for record in records}
    return Order9DatasetStageValidation(
        stage_id=stage.stage_id,
        valid=not unique_failures,
        failures=unique_failures,
        record_count=len(records),
        episode_count=len(episode_ids),
        task_count=len(task_ids),
        stochastic_record_count=stochastic,
        deterministic_teacher_record_count=deterministic,
        metadata={
            "dataset_io_version": ORDER9_DATASET_IO_VERSION,
            "manifest_sha256": bundle.manifest_sha256,
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "held_out_record_count": len(held_out_records),
        },
    )


def _record_module_count(record: object) -> int | None:
    if isinstance(record, LowLevelControlRecord):
        return len(record.runtime_observation.morphology_graph.modules)
    if isinstance(record, InteractionTrajectoryRecord):
        return len(record.morphology_graph.modules)
    if isinstance(record, SequentialDesignTrajectoryRecord):
        if record.design_output is not None:
            return len(record.design_output.target_morphology.modules)
        return None
    return None


def _records_for_stage(
    bundle: Order9DatasetBundle,
    stage: Order9CurriculumStage,
) -> list[object]:
    if stage.learning_target == Order9LearningTarget.DATASET:
        return [*bundle.low_level_records, *bundle.trajectory_records]
    if stage.learning_target == Order9LearningTarget.PI_L:
        return list(bundle.low_level_records)
    if stage.learning_target in {
        Order9LearningTarget.PI_H_ASSIGNMENT,
        Order9LearningTarget.PI_H_TRAJECTORY,
    }:
        return list(bundle.trajectory_records)
    if stage.learning_target == Order9LearningTarget.PI_D:
        return list(bundle.sequential_design_records)
    if stage.learning_target in {
        Order9LearningTarget.JOINT_OBJECT_TASK,
        Order9LearningTarget.FULL_SYSTEM,
    }:
        return [
            *bundle.low_level_records,
            *bundle.trajectory_records,
            *bundle.sequential_design_records,
        ]
    raise SchemaValidationError(
        f"unsupported Order9 learning target {stage.learning_target.value!r}"
    )


def _behavior_trace(record: object):
    if isinstance(record, SequentialDesignTrajectoryRecord):
        traces = [step.behavior_trace for step in record.steps]
        present = [trace for trace in traces if trace is not None]
        if present and len(present) != len(traces):
            raise SchemaValidationError(
                "Order9 sequential design trajectory mixes traced/untraced steps"
            )
        if not present:
            return None
        stochastic_values = {trace.stochastic for trace in present}
        checkpoints = {trace.policy_checkpoint_sha256 for trace in present}
        if len(stochastic_values) != 1 or len(checkpoints) != 1:
            raise SchemaValidationError(
                "Order9 sequential design behavior contract changes within an episode"
            )
        return present[0]
    return getattr(record, "behavior_trace", None)


def _provenance(record: object):
    return getattr(record, "trajectory_provenance", None)


def _validate_episode_boundaries(records: list[object], failures: list[str]) -> None:
    temporal = [
        record
        for record in records
        if isinstance(record, (LowLevelControlRecord, InteractionTrajectoryRecord))
    ]
    by_episode: dict[tuple[str, type], list[object]] = {}
    for record in temporal:
        by_episode.setdefault((record.episode_id, type(record)), []).append(record)
    for episode_records in by_episode.values():
        key = "step_index" if isinstance(episode_records[0], LowLevelControlRecord) else "decision_index"
        ordered = sorted(episode_records, key=lambda record: int(getattr(record, key)))
        indices = [int(getattr(record, key)) for record in ordered]
        if len(indices) != len(set(indices)):
            failures.append("duplicate_episode_index")
        boundaries = [
            index
            for index, record in enumerate(ordered)
            if bool(getattr(record, "terminal")) or bool(getattr(record, "truncated"))
        ]
        if not boundaries:
            continue
        for boundary in boundaries:
            if boundary == len(ordered) - 1:
                continue
            record = ordered[boundary]
            if not (
                isinstance(record, InteractionTrajectoryRecord)
                and record.transition_kind
                == HighLevelTransitionKind.CHECKER_REJECTION
                and record.terminal
                and not record.truncated
            ):
                failures.append("nonfinal_episode_boundary")
                break


def _manifest_path(path: str | Path) -> Path:
    value = Path(path)
    return value / "manifest.json" if value.is_dir() else value


def _resolve_shard_path(raw: str, manifest_dir: Path) -> Path:
    source = Path(raw)
    if source.is_file():
        return source
    candidate = manifest_dir / source
    if candidate.is_file():
        return candidate
    basename = manifest_dir / source.name
    if basename.is_file():
        return basename
    raise FileNotFoundError(f"Order9 dataset shard does not exist: {raw}")


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SchemaValidationError(
                    f"Order9 dataset {path}:{line_number} must contain an object"
                )
            rows.append(value)
    return rows


def _record_from_payload(kind: DatasetKind, raw: dict[str, object]):
    classes = {
        DatasetKind.ISAAC_ROLLOUT: EpisodeArchive,
        DatasetKind.LOW_LEVEL_CONTROL: LowLevelControlRecord,
        DatasetKind.INTERACTION_TRAJECTORY: InteractionTrajectoryRecord,
        DatasetKind.DESIGN_OUTCOME: DesignOutcomeRecord,
        DatasetKind.DESIGN_ACTION_TRAJECTORY: SequentialDesignTrajectoryRecord,
    }
    return classes[kind].from_dict(raw)


def _validate_manifest_counts(
    manifest: P4_3DatasetManifest,
    *,
    low_level: list[LowLevelControlRecord],
    trajectories: list[InteractionTrajectoryRecord],
    designs: list[SequentialDesignTrajectoryRecord],
    outcomes: list[DesignOutcomeRecord],
    archives: list[EpisodeArchive],
) -> None:
    actual = {
        DatasetKind.ISAAC_ROLLOUT.value: len(archives),
        DatasetKind.LOW_LEVEL_CONTROL.value: len(low_level),
        DatasetKind.INTERACTION_TRAJECTORY.value: len(trajectories),
        DatasetKind.DESIGN_OUTCOME.value: len(outcomes),
        DatasetKind.DESIGN_ACTION_TRAJECTORY.value: len(designs),
    }
    for kind, expected in manifest.record_counts.items():
        if actual[kind] != expected:
            raise SchemaValidationError(
                f"Order9 dataset manifest count mismatch for {kind}: "
                f"{actual[kind]} != {expected}"
            )


def _validate_task_split_membership(
    manifest: P4_3DatasetManifest,
    records: Iterable[object],
) -> None:
    allowed = {
        DatasetSplit.TRAIN: set(manifest.train_task_ids),
        DatasetSplit.VALIDATION: set(manifest.validation_task_ids),
        DatasetSplit.HELD_OUT: set(manifest.held_out_task_ids),
    }
    for record in records:
        split = getattr(record, "split")
        task_id = str(getattr(record, "task_id"))
        if task_id not in allowed[split]:
            raise SchemaValidationError(
                f"Order9 dataset task {task_id!r} is not declared in {split.value}"
            )
