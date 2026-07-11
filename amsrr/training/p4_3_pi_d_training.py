from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from amsrr.policies.design_policy_p2 import P2DesignCandidateEvaluation
from amsrr.schemas.datasets import DatasetSplit, DesignOutcomeRecord
from amsrr.training.p2_learned_scorer import TinyP2MLP
from amsrr.training.p2_learning_dataset import P2_LEARNING_FEATURE_NAMES, p2_learning_feature_vector
from amsrr.utils.hashing import hash_file, stable_hash


P4_3_PI_D_CHECKPOINT_TASK = "p4_3_pi_d_outcome_safety_ranking_v1"


@dataclass(frozen=True)
class P4_3PiDTargetWeights:
    success_bonus: float = 2.0
    object_drop_penalty: float = 4.0
    hard_collision_penalty: float = 4.0
    controller_infeasible_terminal_penalty: float = 3.0


@dataclass(frozen=True)
class P4_3PiDTrainingManifest:
    output_dir: str
    checkpoint_path: str
    metrics_path: str
    loss_curve_path: str
    rollout_outcome_evaluation_path: str
    fallback_metadata_path: str
    metrics: dict[str, Any]


def load_design_outcome_records(
    dataset_paths: str | Path | Iterable[str | Path],
) -> list[DesignOutcomeRecord]:
    paths = _normalize_paths(dataset_paths)
    records: list[DesignOutcomeRecord] = []
    seen_ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = DesignOutcomeRecord.from_json(text)
                except Exception as exc:
                    raise ValueError(f"invalid DesignOutcomeRecord at {path}:{line_number}: {exc}") from exc
                if record.record_id in seen_ids:
                    raise ValueError(f"duplicate DesignOutcomeRecord.record_id: {record.record_id}")
                seen_ids.add(record.record_id)
                records.append(record)
    if not records:
        raise ValueError("P4.3 pi_D training requires at least one DesignOutcomeRecord")
    split_by_task: dict[str, DatasetSplit] = {}
    for record in records:
        previous = split_by_task.setdefault(record.task_id, record.split)
        if previous != record.split:
            raise ValueError(
                f"task {record.task_id!r} appears in both {previous.value} and {record.split.value}"
            )
    return records


def design_outcome_feature_vector(record: DesignOutcomeRecord) -> list[float]:
    """Build inference-safe P2 features without reading any rollout outcome label."""

    return p2_candidate_feature_vector(
        candidate_id=record.candidate_id,
        variant_name=_variant_name(record.design_output),
        design_output=record.design_output,
        feasibility_result=record.feasibility_result,
    )


def p2_candidate_feature_vector(
    candidate: P2DesignCandidateEvaluation | None = None,
    *,
    candidate_id: int | None = None,
    variant_name: str | None = None,
    design_output: Any | None = None,
    feasibility_result: Any | None = None,
) -> list[float]:
    """Map a deterministic P2 candidate to the established P2.5 feature layout."""

    if candidate is not None:
        candidate_id = candidate.candidate_id
        variant_name = candidate.variant
        design_output = candidate.design_output
        feasibility_result = candidate.feasibility_result
    if candidate_id is None or design_output is None or feasibility_result is None:
        raise ValueError("candidate id, DesignOutput, and FeasibilityResult are required")
    variant_name = variant_name or _variant_name(design_output)
    morphology = design_output.target_morphology
    slot_ids = {
        binding.slot_id for binding in design_output.slot_anchor_binding_prior
    } | {
        slot_id
        for anchor in morphology.robot_anchors
        for slot_id in anchor.associated_contact_slot_ids
    }
    margins = feasibility_result.margins
    record = {
        "variant_name": variant_name,
        "candidate_source": (
            "closed_loop_invalid_probe" if morphology.is_closed_loop else "p4_3_design_outcome"
        ),
        "candidate_id": candidate_id,
        "required_slot_coverage": margins.get("required_slot_coverage_ratio", 0.0),
        "anchor_coverage": margins.get("required_slot_anchor_coverage_ratio", 0.0),
        "capability_coverage": margins.get(
            "required_slot_anchor_capability_coverage_ratio", 0.0
        ),
        "thrust_margin": margins.get("thrust_margin_ratio", 0.0),
        "payload_margin": margins.get("payload_margin_ratio", 0.0),
        "reachability_margin": margins.get("coarse_reachability_ratio", 0.0),
        "module_count": len(morphology.modules),
        "dock_edge_count": len(morphology.dock_edges),
        "robot_anchor_ids": [anchor.anchor_id for anchor in morphology.robot_anchors],
        "contact_slot_ids": sorted(slot_ids),
        "control_group_ids": [group.group_id for group in morphology.control_groups],
        "feasibility_margins": margins,
    }
    features = p2_learning_feature_vector(record)
    if len(features) != len(P2_LEARNING_FEATURE_NAMES) or not all(math.isfinite(value) for value in features):
        raise ValueError("P4.3 pi_D produced an invalid P2 feature vector")
    return features


def outcome_safety_target(
    record: DesignOutcomeRecord,
    *,
    weights: P4_3PiDTargetWeights | None = None,
) -> float:
    """Return target used for training; this function is never called by inference."""

    weights = weights or P4_3PiDTargetWeights()
    if not record.rollout_executed or record.episode_return is None:
        raise ValueError("outcome target requires an executed rollout with episode_return")
    if any(
        value is None
        for value in (
            record.task_success,
            record.object_dropped,
            record.hard_collision,
            record.controller_infeasible_terminal,
        )
    ):
        raise ValueError("outcome target requires complete safety labels")
    target = float(record.episode_return)
    target += weights.success_bonus if record.task_success else 0.0
    target -= weights.object_drop_penalty if record.object_dropped else 0.0
    target -= weights.hard_collision_penalty if record.hard_collision else 0.0
    target -= (
        weights.controller_infeasible_terminal_penalty
        if record.controller_infeasible_terminal
        else 0.0
    )
    if not math.isfinite(target):
        raise ValueError("outcome target must be finite")
    return target


def train_p4_3_pi_d(
    *,
    dataset_paths: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    p2_checkpoint_path: str | Path | None = None,
    epochs: int = 40,
    lr: float = 0.01,
    seed: int = 0,
    hidden_dim: int = 24,
    target_weights: P4_3PiDTargetWeights | None = None,
) -> P4_3PiDTrainingManifest:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    target_weights = target_weights or P4_3PiDTargetWeights()
    torch.manual_seed(seed)

    source_paths = _normalize_paths(dataset_paths)
    source_dataset_hashes = {str(path): hash_file(path) for path in source_paths}
    training_config = {
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "target_weights": target_weights.__dict__,
        "p2_initializer_checkpoint": (
            str(p2_checkpoint_path) if p2_checkpoint_path is not None else None
        ),
    }
    training_config_hash = stable_hash(training_config)
    dataset_hash = stable_hash(source_dataset_hashes)
    records = load_design_outcome_records(source_paths)
    eligible = [
        record
        for record in records
        if record.rollout_executed and record.feasibility_result.feasible
    ]
    train_records = [record for record in eligible if record.split == DatasetSplit.TRAIN]
    validation_records = [
        record for record in eligible if record.split == DatasetSplit.VALIDATION
    ]
    held_out_records = [record for record in eligible if record.split == DatasetSplit.HELD_OUT]
    if not train_records:
        raise ValueError("P4.3 pi_D training split has no executed hard-feasible records")
    if not validation_records:
        raise ValueError("P4.3 pi_D validation split has no executed hard-feasible records")

    x_train, raw_y_train = _tensors(train_records, target_weights)
    x_validation, raw_y_validation = _tensors(validation_records, target_weights)
    target_mean = float(raw_y_train.mean().item())
    target_scale = float(raw_y_train.std(unbiased=False).item())
    if target_scale < 1.0e-6:
        target_scale = 1.0
    y_train = (raw_y_train - target_mean) / target_scale
    y_validation = (raw_y_validation - target_mean) / target_scale

    model, p2_initialization = _model_with_optional_p2_initialization(
        input_dim=len(P2_LEARNING_FEATURE_NAMES),
        hidden_dim=hidden_dim,
        checkpoint_path=p2_checkpoint_path,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(x_validation), y_validation)
        curve.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss.item()),
                "validation_loss": float(validation_loss.item()),
            }
        )

    metrics = _evaluation_metrics(
        model,
        train_records=train_records,
        validation_records=validation_records,
        held_out_records=held_out_records,
        target_mean=target_mean,
        target_scale=target_scale,
        target_weights=target_weights,
    )
    metrics.update(
        {
            "train_target_std": float(raw_y_train.std(unbiased=False).item()),
            "validation_target_std": float(raw_y_validation.std(unbiased=False).item()),
            "train_unique_target_count": float(len(set(float(value) for value in raw_y_train.tolist()))),
            "task_with_multiple_candidates_count": float(
                sum(
                    1
                    for task_id in {record.task_id for record in train_records}
                    if sum(1 for record in train_records if record.task_id == task_id) > 1
                )
            ),
            "training_config_hash": training_config_hash,
            "dataset_hash": dataset_hash,
            "training_seed": float(seed),
            "training_epoch_count": float(epochs),
        }
    )
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / "checkpoint.pt"
    metrics_path = target_dir / "metrics.json"
    loss_curve_path = target_dir / "loss_curve.csv"
    rollout_evaluation_path = target_dir / "rollout_outcome_evaluation.json"
    fallback_metadata_path = target_dir / "fallback_metadata.json"

    checkpoint = {
        "model_type": "TinyP2MLP",
        "task": P4_3_PI_D_CHECKPOINT_TASK,
        "state_dict": model.state_dict(),
        "feature_names": list(P2_LEARNING_FEATURE_NAMES),
        "feature_min": x_train.min(dim=0).values.tolist(),
        "feature_max": x_train.max(dim=0).values.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "target_weights": target_weights.__dict__,
        "metrics": metrics,
        "training_seed": seed,
        "training_config": training_config,
        "training_config_hash": training_config_hash,
        "dataset_hash": dataset_hash,
        "p2_initialization": p2_initialization,
        "source_dataset_hashes": source_dataset_hashes,
        "source_of_truth": "deterministic FeasibilityChecker hard gate",
        "inference_contract": "design and deterministic feasibility features only",
        "outcome_target_is_inference_feature": False,
    }
    torch.save(checkpoint, checkpoint_path)
    _write_json(metrics_path, metrics)
    _write_curve(loss_curve_path, curve)
    rollout_evaluation = _rollout_evaluation(
        model,
        held_out_records,
        target_mean=target_mean,
        target_scale=target_scale,
        target_weights=target_weights,
    )
    rollout_evaluation.update(
        {
            "training_config_hash": training_config_hash,
            "dataset_hash": dataset_hash,
            "p4_full_completion_claim": False,
        }
    )
    _write_json(rollout_evaluation_path, rollout_evaluation)
    _write_json(
        fallback_metadata_path,
        {
            "policy_family": "pi_D",
            "checkpoint_task": P4_3_PI_D_CHECKPOINT_TASK,
            "deterministic_fallback_available": True,
            "hard_filter": "FeasibilityChecker before learned ranking",
            "selected_candidate_recheck": "FeasibilityChecker after learned ranking",
            "deterministic_fallback": "P2DesignPolicy",
            "fallback_triggers": [
                "checkpoint_invalid",
                "feature_layout_mismatch",
                "non_finite_prediction",
                "out_of_distribution_feature",
                "no_hard_feasible_candidate",
                "selected_candidate_recheck_failed",
            ],
            "outcome_labels_are_training_targets_only": True,
            "learned_feasibility_replaces_hard_gate": False,
        },
    )
    return P4_3PiDTrainingManifest(
        output_dir=str(target_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        loss_curve_path=str(loss_curve_path),
        rollout_outcome_evaluation_path=str(rollout_evaluation_path),
        fallback_metadata_path=str(fallback_metadata_path),
        metrics=metrics,
    )


train_p4_3_pi_d_scorer = train_p4_3_pi_d


def _normalize_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        normalized = [Path(paths)]
    else:
        normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("at least one DesignOutcomeRecord shard is required")
    return normalized


def _variant_name(design_output: Any) -> str:
    for group in design_output.target_morphology.control_groups:
        variant = group.metadata.get("morphology_variant")
        if isinstance(variant, str) and variant:
            return variant
    variant_id = int(design_output.design_scores.get("p2_design_policy_variant_id", -1.0))
    variants = (
        "chain_grasp",
        "symmetric_two_anchor_grasp",
        "tri_anchor_support_grasp",
        "central_base_plus_two_grasp_arms",
    )
    if 0 <= variant_id < len(variants):
        return variants[variant_id]
    return "unknown"


def _tensors(
    records: list[DesignOutcomeRecord],
    weights: P4_3PiDTargetWeights,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [design_outcome_feature_vector(record) for record in records],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [outcome_safety_target(record, weights=weights) for record in records],
        dtype=torch.float32,
    )
    return features, targets


def _model_with_optional_p2_initialization(
    *,
    input_dim: int,
    hidden_dim: int,
    checkpoint_path: str | Path | None,
) -> tuple[TinyP2MLP, dict[str, Any]]:
    if checkpoint_path is None:
        return TinyP2MLP(input_dim=input_dim, hidden_dim=hidden_dim), {
            "used": False,
            "reason": "not_requested",
        }
    path = Path(checkpoint_path)
    try:
        payload = _load_checkpoint(path)
        if payload.get("feature_names") != P2_LEARNING_FEATURE_NAMES:
            raise ValueError("feature_names mismatch")
        state_dict = payload["state_dict"]
        first_weight = state_dict["net.0.weight"]
        inferred_hidden_dim = int(first_weight.shape[0])
        if int(first_weight.shape[1]) != input_dim:
            raise ValueError("input dimension mismatch")
        model = TinyP2MLP(input_dim=input_dim, hidden_dim=inferred_hidden_dim)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        return TinyP2MLP(input_dim=input_dim, hidden_dim=hidden_dim), {
            "used": False,
            "reason": f"incompatible:{type(exc).__name__}:{exc}",
            "checkpoint_path": str(path),
        }
    return model, {
        "used": True,
        "reason": "compatible_p2_feature_layout_and_state_dict",
        "checkpoint_path": str(path),
        "checkpoint_hash": hash_file(path),
        "source_task": str(payload.get("task", "unknown")),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def _predict_raw(
    model: nn.Module,
    records: list[DesignOutcomeRecord],
    *,
    target_mean: float,
    target_scale: float,
) -> list[float]:
    if not records:
        return []
    features = torch.tensor(
        [design_outcome_feature_vector(record) for record in records],
        dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        values = model(features) * target_scale + target_mean
    return [float(value) for value in values.tolist()]


def _evaluation_metrics(
    model: nn.Module,
    *,
    train_records: list[DesignOutcomeRecord],
    validation_records: list[DesignOutcomeRecord],
    held_out_records: list[DesignOutcomeRecord],
    target_mean: float,
    target_scale: float,
    target_weights: P4_3PiDTargetWeights,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "num_train_samples": float(len(train_records)),
        "num_validation_samples": float(len(validation_records)),
        "num_held_out_samples": float(len(held_out_records)),
    }
    for prefix, records in (
        ("train", train_records),
        ("validation", validation_records),
        ("held_out", held_out_records),
    ):
        predictions = _predict_raw(
            model,
            records,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        targets = [outcome_safety_target(record, weights=target_weights) for record in records]
        metrics[f"{prefix}_mae"] = _mae(predictions, targets)
        accuracy, pair_count = _pairwise_ranking_accuracy(records, predictions, targets)
        metrics[f"{prefix}_pairwise_ranking_accuracy"] = accuracy
        metrics[f"{prefix}_ranking_pair_count"] = float(pair_count)
    return metrics


def _pairwise_ranking_accuracy(
    records: list[DesignOutcomeRecord],
    predictions: list[float],
    targets: list[float],
) -> tuple[float, int]:
    correct = 0
    total = 0
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left].task_id != records[right].task_id:
                continue
            target_delta = targets[left] - targets[right]
            if abs(target_delta) <= 1.0e-9:
                continue
            prediction_delta = predictions[left] - predictions[right]
            total += 1
            if prediction_delta * target_delta > 0.0:
                correct += 1
    return (float(correct) / float(total), total) if total else (0.0, 0)


def _rollout_evaluation(
    model: nn.Module,
    records: list[DesignOutcomeRecord],
    *,
    target_mean: float,
    target_scale: float,
    target_weights: P4_3PiDTargetWeights,
) -> dict[str, Any]:
    predictions = _predict_raw(
        model,
        records,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    rows = []
    for record, prediction in zip(records, predictions, strict=True):
        rows.append(
            {
                "record_id": record.record_id,
                "task_id": record.task_id,
                "candidate_id": record.candidate_id,
                "predicted_outcome_score": prediction,
                "target_outcome_score": outcome_safety_target(record, weights=target_weights),
                "task_success": record.task_success,
                "object_dropped": record.object_dropped,
                "hard_collision": record.hard_collision,
                "controller_infeasible_terminal": record.controller_infeasible_terminal,
            }
        )
    return {
        "split": DatasetSplit.HELD_OUT.value,
        "record_count": len(rows),
        "records": rows,
        "outcome_fields_are_targets_not_inference_features": True,
        "evaluation_hash": stable_hash(rows),
    }


def _mae(predictions: list[float], targets: list[float]) -> float:
    if not targets:
        return 0.0
    return sum(abs(prediction - target) for prediction, target in zip(predictions, targets, strict=True)) / len(targets)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_curve(path: Path, curve: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "validation_loss"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(curve)
