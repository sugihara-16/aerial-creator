from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from amsrr.policies.learned_low_level_policy import (
    PI_L_FEATURE_NAMES,
    PI_L_FEATURE_OOD_MIN_SCALES,
    PI_L_OUTPUT_MODE,
    PI_L_POLICY_CHECKPOINT_VERSION,
    PI_L_TARGET_LOWER_BOUNDS,
    PI_L_TARGET_NAMES,
    PI_L_TARGET_UPPER_BOUNDS,
    TinyPiLDeltaMLP,
    pi_l_feature_vector,
)
from amsrr.policies.low_level_policy_base import BaselineLowLevelPolicy, LowLevelPolicyContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import (
    DatasetKind,
    DatasetSplit,
    LowLevelControlRecord,
    P4_3DatasetManifest,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactWrenchTrajectory, PolicyCommand
from amsrr.utils.config import load_config


P4_3_PI_L_TRAINING_VERSION = "p4_3_pi_l_training_v1"


@dataclass
class P4_3PiLTrainingConfig(SchemaBase):
    epochs: int = 20
    learning_rate: float = 0.001
    hidden_dim: int = 64
    batch_size: int = 256
    seed: int = 11
    output_mode: str = PI_L_OUTPUT_MODE
    checkpoint_dir: str = "artifacts/p4_3/pi_l"
    ood_z_score_limit: float = 8.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise SchemaValidationError("P4_3PiLTrainingConfig.epochs must be positive")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise SchemaValidationError(
                "P4_3PiLTrainingConfig.learning_rate must be finite and positive"
            )
        if self.hidden_dim < 1:
            raise SchemaValidationError("P4_3PiLTrainingConfig.hidden_dim must be positive")
        if self.batch_size < 1:
            raise SchemaValidationError("P4_3PiLTrainingConfig.batch_size must be positive")
        if self.seed < 0:
            raise SchemaValidationError("P4_3PiLTrainingConfig.seed must be non-negative")
        if self.output_mode != PI_L_OUTPUT_MODE:
            raise SchemaValidationError(
                f"P4_3PiLTrainingConfig.output_mode must be {PI_L_OUTPUT_MODE!r}"
            )
        require_non_empty(self.checkpoint_dir, "P4_3PiLTrainingConfig.checkpoint_dir")
        if self.ood_z_score_limit <= 0.0 or not math.isfinite(self.ood_z_score_limit):
            raise SchemaValidationError(
                "P4_3PiLTrainingConfig.ood_z_score_limit must be finite and positive"
            )


@dataclass(frozen=True)
class P4_3PiLTrainingManifest:
    output_dir: str
    checkpoint_path: str
    metrics_path: str
    loss_curve_path: str
    reward_curve_path: str
    rollout_evaluation_path: str
    fallback_metadata_path: str
    metrics: dict[str, Any]


def load_p4_3_pi_l_training_config(
    path: str | Path = "configs/training/p4_3_learning_bootstrap.yaml",
) -> P4_3PiLTrainingConfig:
    data = load_config(path)
    return P4_3PiLTrainingConfig.from_dict(data.get("pi_l", {}))


def load_low_level_control_records(
    path: str | Path | Iterable[str | Path],
) -> list[LowLevelControlRecord]:
    records: list[LowLevelControlRecord] = []
    paths = _resolve_low_level_dataset_paths(path)
    for source_path in paths:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(LowLevelControlRecord.from_json(stripped))
                except (json.JSONDecodeError, SchemaValidationError) as exc:
                    raise ValueError(
                        "invalid LowLevelControlRecord JSONL row at "
                        f"{source_path}:{line_number}: {exc}"
                    ) from exc
    if not records:
        raise ValueError("LowLevelControlRecord dataset must not be empty")
    duplicate_ids = _duplicates(record.record_id for record in records)
    if duplicate_ids:
        raise ValueError(f"LowLevelControlRecord record_id values must be unique: {duplicate_ids}")
    _require_task_disjoint_splits(records)
    return records


def low_level_record_feature_vector(record: LowLevelControlRecord) -> list[float]:
    return pi_l_feature_vector(_context_for_record(record))


def low_level_record_target_delta(record: LowLevelControlRecord) -> list[float]:
    target, _, _ = _target_delta_with_stats(record)
    return target


def train_p4_3_pi_l(
    *,
    dataset_path: str | Path | Iterable[str | Path],
    output_dir: str | Path | None = None,
    config_path: str | Path = "configs/training/p4_3_learning_bootstrap.yaml",
    config: P4_3PiLTrainingConfig | None = None,
    source_is_real_isaac: bool | None = None,
) -> P4_3PiLTrainingManifest:
    training_config = config or load_p4_3_pi_l_training_config(config_path)
    target_dir = Path(output_dir or training_config.checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _resolve_low_level_dataset_paths(dataset_path)
    records = load_low_level_control_records(dataset_paths)
    real_isaac_source, source_evidence = _resolve_real_isaac_source_claim(
        dataset_path,
        source_is_real_isaac,
    )
    eligible = [record for record in records if _eligible_for_learned_delta(record)]
    train_records = [record for record in eligible if record.split == DatasetSplit.TRAIN]
    validation_records = [
        record for record in eligible if record.split == DatasetSplit.VALIDATION
    ]
    held_out_records = [record for record in eligible if record.split == DatasetSplit.HELD_OUT]
    if not train_records:
        raise ValueError("pi_L training requires controller-feasible train records")
    if not validation_records:
        raise ValueError("pi_L training requires controller-feasible validation records")

    torch.manual_seed(training_config.seed)
    generator = torch.Generator().manual_seed(training_config.seed)
    x_train, y_train, train_stats = _record_tensors(train_records)
    x_validation, y_validation, validation_stats = _record_tensors(validation_records)
    held_tensors = _record_tensors(held_out_records) if held_out_records else None

    feature_mean = x_train.mean(dim=0)
    empirical_std = x_train.std(dim=0, unbiased=False)
    feature_std = torch.clamp(empirical_std, min=1.0e-4)
    ood_floor = torch.tensor(PI_L_FEATURE_OOD_MIN_SCALES, dtype=torch.float32)
    feature_ood_scale = torch.maximum(empirical_std, ood_floor)
    normalized_train = (x_train - feature_mean) / feature_std
    normalized_validation = (x_validation - feature_mean) / feature_std
    normalized_held = (
        (held_tensors[0] - feature_mean) / feature_std if held_tensors is not None else None
    )

    model = TinyPiLDeltaMLP(
        input_dim=len(PI_L_FEATURE_NAMES),
        hidden_dim=training_config.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    target_scale = torch.tensor(
        [
            (upper - lower) * 0.5
            for lower, upper in zip(
                PI_L_TARGET_LOWER_BOUNDS,
                PI_L_TARGET_UPPER_BOUNDS,
            )
        ],
        dtype=torch.float32,
    )
    loss_curve: list[dict[str, float]] = []
    for epoch in range(training_config.epochs):
        model.train()
        order = torch.randperm(len(train_records), generator=generator)
        for start in range(0, len(train_records), training_config.batch_size):
            batch_indices = order[start : start + training_config.batch_size]
            prediction = model(normalized_train[batch_indices])
            loss = _normalized_mse(prediction, y_train[batch_indices], target_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = _normalized_mse(model(normalized_train), y_train, target_scale)
            validation_loss = _normalized_mse(
                model(normalized_validation),
                y_validation,
                target_scale,
            )
        loss_curve.append(
            {
                "epoch": float(epoch + 1),
                "train_normalized_mse": float(train_loss.item()),
                "validation_normalized_mse": float(validation_loss.item()),
            }
        )

    model.eval()
    with torch.no_grad():
        train_prediction = model(normalized_train)
        validation_prediction = model(normalized_validation)
        held_prediction = model(normalized_held) if normalized_held is not None else None

    reward_rows = _offline_reward_curve(records, source_is_real_isaac=real_isaac_source)
    dataset_hash = _combined_dataset_sha256(dataset_paths)
    config_hash = training_config.stable_hash()
    metrics: dict[str, Any] = {
        "training_version": P4_3_PI_L_TRAINING_VERSION,
        "output_mode": PI_L_OUTPUT_MODE,
        "evaluation_mode": "offline_dataset_evaluation",
        "online_rollout_evaluation": False,
        "reward_curve_semantics": "offline_episode_returns_not_online_training_curve",
        "source_is_real_isaac": real_isaac_source,
        "source_evidence": source_evidence,
        "dataset_sha256": dataset_hash,
        "dataset_paths": [str(path) for path in dataset_paths],
        "config_hash": config_hash,
        "record_count": len(records),
        "eligible_record_count": len(eligible),
        "controller_fallback_record_count": len(records) - len(eligible),
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "held_out_record_count": len(held_out_records),
        "train_task_ids": sorted({record.task_id for record in train_records}),
        "validation_task_ids": sorted({record.task_id for record in validation_records}),
        "held_out_task_ids": sorted({record.task_id for record in held_out_records}),
        "train_normalized_mse": _normalized_mse_value(
            train_prediction,
            y_train,
            target_scale,
        ),
        "validation_normalized_mse": _normalized_mse_value(
            validation_prediction,
            y_validation,
            target_scale,
        ),
        "held_out_normalized_mse": (
            _normalized_mse_value(held_prediction, held_tensors[1], target_scale)
            if held_prediction is not None and held_tensors is not None
            else None
        ),
        "train_raw_mse": _raw_mse_value(train_prediction, y_train),
        "validation_raw_mse": _raw_mse_value(validation_prediction, y_validation),
        "held_out_raw_mse": (
            _raw_mse_value(held_prediction, held_tensors[1])
            if held_prediction is not None and held_tensors is not None
            else None
        ),
        "target_clipped_value_count": (
            train_stats["target_clipped_value_count"]
            + validation_stats["target_clipped_value_count"]
            + (held_tensors[2]["target_clipped_value_count"] if held_tensors is not None else 0)
        ),
        "unrepresentable_target_value_count": (
            train_stats["unrepresentable_target_value_count"]
            + validation_stats["unrepresentable_target_value_count"]
            + (
                held_tensors[2]["unrepresentable_target_value_count"]
                if held_tensors is not None
                else 0
            )
        ),
        "offline_reward_episode_count": len(reward_rows),
        "offline_mean_episode_return": (
            sum(float(row["offline_episode_return"]) for row in reward_rows) / len(reward_rows)
            if reward_rows
            else None
        ),
        "deterministic_fallback_available": True,
        "deterministic_fallback_path": (
            "amsrr.policies.low_level_policy_base.BaselineLowLevelPolicy"
        ),
        "controller_authority": "controller_qp_safety_layer_only",
        "p4_full_completion_claim": False,
    }

    checkpoint_path = target_dir / "checkpoint.pt"
    metrics_path = target_dir / "metrics.json"
    loss_curve_path = target_dir / "loss_curve.csv"
    reward_curve_path = target_dir / "reward_curve.csv"
    rollout_evaluation_path = target_dir / "rollout_evaluation.json"
    fallback_metadata_path = target_dir / "fallback_metadata.json"
    fallback_metadata = {
        "policy_family": "pi_L",
        "fallback_available": True,
        "path": "amsrr.policies.low_level_policy_base.BaselineLowLevelPolicy",
        "fallback_path": "amsrr.policies.low_level_policy_base.BaselineLowLevelPolicy",
        "triggers": [
            "feature_extraction_error",
            "non_finite_features",
            "feature_shape",
            "feature_ood",
            "non_finite_normalized_features",
            "model_inference_error",
            "model_output_shape",
            "non_finite_model_output",
            "non_finite_merged_command",
            "controller_infeasible",
        ],
        "output_mode": PI_L_OUTPUT_MODE,
        "target_names": list(PI_L_TARGET_NAMES),
        "output_lower_bounds": list(PI_L_TARGET_LOWER_BOUNDS),
        "output_upper_bounds": list(PI_L_TARGET_UPPER_BOUNDS),
        "controller_command_output": False,
        "actuator_target_output": False,
        "controller_authority": "controller_qp_safety_layer_only",
    }
    checkpoint: dict[str, Any] = {
        "checkpoint_version": PI_L_POLICY_CHECKPOINT_VERSION,
        "training_version": P4_3_PI_L_TRAINING_VERSION,
        "model_type": "TinyPiLDeltaMLP",
        "task": "pi_l_bounded_policy_command_delta_imitation",
        "output_mode": PI_L_OUTPUT_MODE,
        "state_dict": model.state_dict(),
        "input_dim": len(PI_L_FEATURE_NAMES),
        "hidden_dim": training_config.hidden_dim,
        "output_dim": len(PI_L_TARGET_NAMES),
        "feature_names": list(PI_L_FEATURE_NAMES),
        "target_names": list(PI_L_TARGET_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "feature_ood_scale": feature_ood_scale.tolist(),
        "output_lower_bounds": list(PI_L_TARGET_LOWER_BOUNDS),
        "output_upper_bounds": list(PI_L_TARGET_UPPER_BOUNDS),
        "ood_z_score_limit": training_config.ood_z_score_limit,
        "dataset_sha256": dataset_hash,
        "config_hash": config_hash,
        "metrics": metrics,
        "reward_curve_semantics": "offline_episode_returns_not_online_training_curve",
        "source_is_real_isaac": real_isaac_source,
        "learned_output_contract": (
            "bounded PolicyCommand desired_body_twist/body_position/residual_wrench delta subset"
        ),
        "controller_command_output": False,
        "actuator_target_output": False,
        "deterministic_fallback": fallback_metadata,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    _write_json(metrics_path, metrics)
    _write_loss_curve(loss_curve_path, loss_curve)
    _write_reward_curve(reward_curve_path, reward_rows)
    _write_json(
        rollout_evaluation_path,
        {
            "policy_family": "pi_L",
            "evaluation_mode": "offline_dataset_evaluation",
            "artifact_semantics": (
                "offline evaluation on Isaac-backed LowLevelControlRecord data; "
                "not a learned-policy online Isaac rollout"
            ),
            "online_rollout_executed": False,
            "learned_policy_deployed_in_isaac": False,
            "source_is_real_isaac": real_isaac_source,
            "dataset_sha256": dataset_hash,
            "train_normalized_mse": metrics["train_normalized_mse"],
            "validation_normalized_mse": metrics["validation_normalized_mse"],
            "held_out_normalized_mse": metrics["held_out_normalized_mse"],
            "offline_reward_episode_count": metrics["offline_reward_episode_count"],
            "offline_mean_episode_return": metrics["offline_mean_episode_return"],
            "p4_full_completion_claim": False,
        },
    )
    _write_json(fallback_metadata_path, fallback_metadata)
    return P4_3PiLTrainingManifest(
        output_dir=str(target_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        loss_curve_path=str(loss_curve_path),
        reward_curve_path=str(reward_curve_path),
        rollout_evaluation_path=str(rollout_evaluation_path),
        fallback_metadata_path=str(fallback_metadata_path),
        metrics=metrics,
    )


def _context_for_record(record: LowLevelControlRecord) -> LowLevelPolicyContext:
    # BaselineLowLevelPolicy does not consume detailed PhysicalModel fields, but
    # LowLevelPolicyContext requires the schema. Keep this record-only adapter
    # explicit so the feature/target construction never reaches actuators.
    physical_model = PhysicalModel(
        model_id="p4-3-pi-l-record-context",
        urdf_path="record-context://not-used-for-pi-l-features",
        links=[],
        joints=[],
        rotors=[],
        dock_ports=[],
        collision_primitives=[],
        aggregate_mass_kg=0.0,
        aggregate_inertia_body=[0.0] * 6,
    )
    horizon_s = max(0.01, float(record.active_knot.t_rel_s) + 0.01)
    trajectory = ContactWrenchTrajectory(
        horizon_s=horizon_s,
        dt_s=0.01,
        knots=[record.active_knot],
        derived_mode_label="p4_3_pi_l_record_context",
    )
    return LowLevelPolicyContext(
        runtime_observation=record.runtime_observation,
        morphology_graph=record.runtime_observation.morphology_graph,
        physical_model=physical_model,
        contact_wrench_trajectory=trajectory,
        active_knot=record.active_knot,
        controller_status=record.runtime_observation.controller_status,
    )


def _target_delta_with_stats(
    record: LowLevelControlRecord,
) -> tuple[list[float], int, int]:
    baseline = BaselineLowLevelPolicy().command(_context_for_record(record))
    teacher = record.policy_command
    raw: list[float] = []
    unrepresentable = 0

    twist, missing = _optional_delta(teacher.desired_body_twist, baseline.desired_body_twist, 6)
    raw.extend(twist)
    unrepresentable += missing

    teacher_position = teacher.desired_body_pose[:3] if teacher.desired_body_pose is not None else None
    baseline_position = baseline.desired_body_pose[:3] if baseline.desired_body_pose is not None else None
    position, missing = _optional_delta(teacher_position, baseline_position, 3)
    raw.extend(position)
    unrepresentable += missing

    residual, missing = _optional_delta(
        teacher.residual_wrench_body,
        baseline.residual_wrench_body,
        6,
    )
    raw.extend(residual)
    unrepresentable += missing

    bounded: list[float] = []
    clipped_count = 0
    for value, lower, upper in zip(
        raw,
        PI_L_TARGET_LOWER_BOUNDS,
        PI_L_TARGET_UPPER_BOUNDS,
    ):
        clipped = min(max(float(value), lower), upper)
        clipped_count += int(not math.isclose(clipped, float(value), rel_tol=0.0, abs_tol=1.0e-12))
        bounded.append(clipped)
    return bounded, clipped_count, unrepresentable


def _optional_delta(
    teacher: Iterable[float] | None,
    baseline: Iterable[float] | None,
    size: int,
) -> tuple[list[float], int]:
    if teacher is None and baseline is None:
        return [0.0] * size, 0
    if teacher is None and baseline is not None:
        baseline_values = [float(value) for value in baseline]
        if len(baseline_values) != size:
            raise ValueError("pi_L baseline target shape mismatch")
        return [-value for value in baseline_values], 0
    if baseline is None:
        return [0.0] * size, size
    teacher_values = [float(value) for value in teacher]
    baseline_values = [float(value) for value in baseline]
    if len(teacher_values) != size or len(baseline_values) != size:
        raise ValueError("pi_L teacher/baseline target shape mismatch")
    return [left - right for left, right in zip(teacher_values, baseline_values)], 0


def _record_tensors(
    records: list[LowLevelControlRecord],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    clipped_count = 0
    unrepresentable_count = 0
    for record in records:
        feature = low_level_record_feature_vector(record)
        target, clipped, unrepresentable = _target_delta_with_stats(record)
        if not all(math.isfinite(value) for value in feature + target):
            raise ValueError(f"record {record.record_id!r} produced non-finite pi_L tensors")
        features.append(feature)
        targets.append(target)
        clipped_count += clipped
        unrepresentable_count += unrepresentable
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        {
            "target_clipped_value_count": clipped_count,
            "unrepresentable_target_value_count": unrepresentable_count,
        },
    )


def _eligible_for_learned_delta(record: LowLevelControlRecord) -> bool:
    statuses = (
        record.runtime_observation.controller_status,
        record.controller_command.controller_status,
    )
    return all(
        status.qp_feasible and status.status not in {"infeasible", "fault"}
        for status in statuses
    )


def _require_task_disjoint_splits(records: list[LowLevelControlRecord]) -> None:
    tasks: dict[DatasetSplit, set[str]] = {split: set() for split in DatasetSplit}
    for record in records:
        tasks[record.split].add(record.task_id)
    splits = list(DatasetSplit)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = sorted(tasks[left].intersection(tasks[right]))
            if overlap:
                raise ValueError(
                    "LowLevelControlRecord task splits must be disjoint; "
                    f"{left.value}/{right.value} overlap: {overlap}"
                )


def _offline_reward_curve(
    records: list[LowLevelControlRecord],
    *,
    source_is_real_isaac: bool,
) -> list[dict[str, Any]]:
    by_episode: dict[str, list[LowLevelControlRecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
    rows: list[dict[str, Any]] = []
    running_sum = 0.0
    for episode_index, episode_id in enumerate(sorted(by_episode), start=1):
        episode_records = sorted(by_episode[episode_id], key=lambda item: item.step_index)
        reward_values = [float(item.reward) for item in episode_records if item.reward is not None]
        if not reward_values:
            continue
        split_values = {item.split.value for item in episode_records}
        task_values = {item.task_id for item in episode_records}
        if len(split_values) != 1 or len(task_values) != 1:
            raise ValueError(f"episode {episode_id!r} crosses task or split boundaries")
        episode_return = sum(reward_values)
        running_sum += episode_return
        rows.append(
            {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "task_id": next(iter(task_values)),
                "split": next(iter(split_values)),
                "offline_episode_return": episode_return,
                "offline_running_mean_return": running_sum / (len(rows) + 1),
                "reward_step_count": len(reward_values),
                "source_is_real_isaac": int(source_is_real_isaac),
                "evaluation_mode": "offline_dataset_evaluation",
            }
        )
    return rows


def _normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(torch.square((prediction - target) / scale))


def _normalized_mse_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    return float(_normalized_mse(prediction, target, scale).item())


def _raw_mse_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(prediction, target).item())


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _resolve_low_level_dataset_paths(
    source: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(source, (str, Path)):
        candidates = [Path(source)]
    else:
        candidates = [Path(value) for value in source]
    resolved: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            resolved.extend(sorted(candidate.glob("low_level_control_*.jsonl")))
            continue
        if candidate.name == "manifest.json":
            manifest = P4_3DatasetManifest.from_json(candidate.read_text(encoding="utf-8"))
            for shard in manifest.shards:
                if shard.dataset_kind != DatasetKind.LOW_LEVEL_CONTROL:
                    continue
                shard_path = Path(shard.path)
                if not shard_path.is_absolute() and not shard_path.exists():
                    manifest_relative = candidate.parent / shard_path
                    sibling = candidate.parent / shard_path.name
                    shard_path = (
                        manifest_relative if manifest_relative.exists() else sibling
                    )
                resolved.append(shard_path)
            continue
        resolved.append(candidate)
    unique = sorted(set(resolved), key=lambda path: str(path))
    if not unique:
        raise ValueError("no low_level_control JSONL shards were found")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise ValueError(f"low_level_control JSONL shards do not exist: {missing}")
    return unique


def _resolve_real_isaac_source_claim(
    source: str | Path | Iterable[str | Path],
    explicit_claim: bool | None,
) -> tuple[bool, str]:
    manifest_paths = _candidate_manifest_paths(source)
    manifests = [
        P4_3DatasetManifest.from_json(path.read_text(encoding="utf-8"))
        for path in manifest_paths
    ]
    if manifests:
        source_count = sum(
            int(manifest.metadata.get("source_episode_count", 0))
            for manifest in manifests
        )
        isaac_count = sum(
            int(manifest.metadata.get("isaac_backed_episode_count", 0))
            for manifest in manifests
        )
        manifest_claim = source_count > 0 and isaac_count == source_count
        if explicit_claim is True and not manifest_claim:
            raise ValueError(
                "source_is_real_isaac=True conflicts with P4.3 dataset manifest provenance"
            )
        if explicit_claim is None:
            return (
                manifest_claim,
                "p4_3_dataset_manifest_all_episodes_real_isaac"
                if manifest_claim
                else "p4_3_dataset_manifest_not_all_episodes_real_isaac",
            )
        return bool(explicit_claim), "explicit_claim_verified_against_p4_3_dataset_manifest"
    if explicit_claim is True:
        return True, "caller_attested_p4_3a_real_isaac_dataset_without_manifest"
    return False, "no_real_isaac_dataset_manifest_or_attestation"


def _candidate_manifest_paths(
    source: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(source, (str, Path)):
        candidates = [Path(source)]
    else:
        candidates = [Path(value) for value in source]
    manifests: set[Path] = set()
    for candidate in candidates:
        if candidate.name == "manifest.json" and candidate.is_file():
            manifests.add(candidate)
        elif candidate.is_dir() and (candidate / "manifest.json").is_file():
            manifests.add(candidate / "manifest.json")
        elif candidate.is_file() and (candidate.parent / "manifest.json").is_file():
            manifests.add(candidate.parent / "manifest.json")
    return sorted(manifests, key=lambda path: str(path))


def _combined_dataset_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _write_loss_curve(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_normalized_mse", "validation_normalized_mse"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_reward_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "episode_id",
        "task_id",
        "split",
        "offline_episode_return",
        "offline_running_mean_return",
        "reward_step_count",
        "source_is_real_isaac",
        "evaluation_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the P4.3b bounded learned pi_L head.")
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="LowLevelControlRecord JSONL file(s), P4.3 dataset directory, or manifest.json.",
    )
    parser.add_argument(
        "--config",
        default="configs/training/p4_3_learning_bootstrap.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--source-is-real-isaac",
        action="store_true",
        default=None,
        help="Attest that the LowLevelControlRecord shard came from real P4.3a Isaac rollout archives.",
    )
    args = parser.parse_args(argv)
    manifest = train_p4_3_pi_l(
        dataset_path=args.dataset,
        config_path=args.config,
        output_dir=args.output_dir,
        source_is_real_isaac=args.source_is_real_isaac,
    )
    print(f"checkpoint: {manifest.checkpoint_path}")
    print(f"metrics: {manifest.metrics_path}")
    print(f"loss curve: {manifest.loss_curve_path}")
    print(f"reward curve: {manifest.reward_curve_path}")
    print(f"rollout evaluation: {manifest.rollout_evaluation_path}")
    print(f"fallback metadata: {manifest.fallback_metadata_path}")
    print(json.dumps(manifest.metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
