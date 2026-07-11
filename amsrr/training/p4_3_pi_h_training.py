from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from amsrr.policies.contact_candidate_encoder import ContactCandidateEncoder
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.learned_high_level_policy import (
    LearnedHighLevelPolicy,
    LearnedHighLevelPolicyConfig,
    P4_3HighLevelRanker,
    evaluate_selected_candidate_ids,
    policy_config_dict,
)
from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    require_non_empty,
)
from amsrr.schemas.datasets import DatasetSplit, InteractionTrajectoryRecord
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file, stable_hash


P4_3_PI_H_TRAINING_VERSION = "p4_3_pi_h_imitation_v1"


@dataclass
class P4_3PiHTrainingConfig(SchemaBase):
    epochs: int = 20
    learning_rate: float = 1.0e-3
    hidden_dim: int = 64
    batch_size: int = 64
    seed: int = 13
    update_rate_hz: float = 2.0
    checkpoint_dir: str = "artifacts/p4_3/pi_h"
    encoder_d_model: int = 48
    max_timing_residual_s: float = 0.05
    timing_loss_weight: float = 0.25

    def validate(self) -> None:
        if self.epochs <= 0:
            raise SchemaValidationError("P4_3PiHTrainingConfig.epochs must be positive")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.learning_rate must be finite and positive"
            )
        if self.hidden_dim <= 0:
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.hidden_dim must be positive"
            )
        if self.batch_size <= 0:
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.batch_size must be positive"
            )
        if self.seed < 0:
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.seed must be non-negative"
            )
        if self.update_rate_hz <= 0.0 or not math.isfinite(self.update_rate_hz):
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.update_rate_hz must be finite and positive"
            )
        require_non_empty(
            self.checkpoint_dir,
            "P4_3PiHTrainingConfig.checkpoint_dir",
        )
        if self.encoder_d_model <= 0:
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.encoder_d_model must be positive"
            )
        if self.max_timing_residual_s < 0.0 or not math.isfinite(
            self.max_timing_residual_s
        ):
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.max_timing_residual_s must be finite and non-negative"
            )
        if self.timing_loss_weight < 0.0 or not math.isfinite(
            self.timing_loss_weight
        ):
            raise SchemaValidationError(
                "P4_3PiHTrainingConfig.timing_loss_weight must be finite and non-negative"
            )


@dataclass(frozen=True)
class P4_3PiHTrainingManifest:
    output_dir: str
    checkpoint_path: str
    metrics_path: str
    loss_curve_path: str
    rollout_evaluation_path: str
    fallback_metadata_path: str
    metrics: dict[str, float]


def load_p4_3_pi_h_training_config(
    path: str | Path = "configs/training/p4_3_learning_bootstrap.yaml",
) -> P4_3PiHTrainingConfig:
    data = load_config(path)
    return P4_3PiHTrainingConfig.from_dict(data.get("pi_h", {}))


def load_interaction_trajectory_records(
    shard_paths: str | Path | Sequence[str | Path],
) -> list[InteractionTrajectoryRecord]:
    """Load direct ``InteractionTrajectoryRecord`` JSONL shards.

    Wrapper objects and generic rollout archives are deliberately rejected so
    the P4.3c training boundary remains schema-first and unambiguous.
    """

    paths = _normalize_paths(shard_paths)
    records: list[InteractionTrajectoryRecord] = []
    for path in paths:
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"pi_H dataset shard must be JSONL: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = InteractionTrajectoryRecord.from_json(line)
                except (json.JSONDecodeError, SchemaValidationError) as exc:
                    raise SchemaValidationError(
                        f"invalid InteractionTrajectoryRecord at {path}:{line_number}: {exc}"
                    ) from exc
                records.append(record)
    if not records:
        raise ValueError("pi_H dataset shards contain no InteractionTrajectoryRecord rows")
    _validate_record_collection(records)
    return records


def train_p4_3_pi_h(
    *,
    shard_paths: str | Path | Sequence[str | Path],
    output_dir: str | Path | None = None,
    config_path: str | Path = "configs/training/p4_3_learning_bootstrap.yaml",
    config: P4_3PiHTrainingConfig | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    hidden_dim: int | None = None,
    encoder_d_model: int | None = None,
    max_timing_residual_s: float | None = None,
    timing_loss_weight: float | None = None,
) -> P4_3PiHTrainingManifest:
    base_config = config or load_p4_3_pi_h_training_config(config_path)
    epochs = base_config.epochs if epochs is None else epochs
    learning_rate = (
        base_config.learning_rate if learning_rate is None else learning_rate
    )
    seed = base_config.seed if seed is None else seed
    hidden_dim = base_config.hidden_dim if hidden_dim is None else hidden_dim
    encoder_d_model = (
        base_config.encoder_d_model
        if encoder_d_model is None
        else encoder_d_model
    )
    max_timing_residual_s = (
        base_config.max_timing_residual_s
        if max_timing_residual_s is None
        else max_timing_residual_s
    )
    timing_loss_weight = (
        base_config.timing_loss_weight
        if timing_loss_weight is None
        else timing_loss_weight
    )
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if timing_loss_weight < 0.0 or not math.isfinite(timing_loss_weight):
        raise ValueError("timing_loss_weight must be finite and non-negative")

    paths = _normalize_paths(shard_paths)
    records = load_interaction_trajectory_records(paths)
    train_records = [record for record in records if record.split == DatasetSplit.TRAIN]
    validation_records = [
        record for record in records if record.split == DatasetSplit.VALIDATION
    ]
    held_out_records = [record for record in records if record.split == DatasetSplit.HELD_OUT]
    if not train_records:
        raise ValueError("pi_H training requires at least one train record")
    if not validation_records:
        raise ValueError("pi_H training requires at least one validation record")

    torch.manual_seed(seed)
    policy_config = LearnedHighLevelPolicyConfig(
        encoder_d_model=encoder_d_model,
        hidden_dim=hidden_dim,
        max_timing_residual_s=max_timing_residual_s,
        timing_residual_enabled=True,
    )
    policy_config.validate()
    encoder = ContactCandidateEncoder(d_model=encoder_d_model)
    model = P4_3HighLevelRanker(
        d_model=encoder_d_model,
        hidden_dim=hidden_dim,
        max_timing_residual_s=max_timing_residual_s,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    curve: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_components = [
            _record_losses(
                model,
                encoder,
                record,
                timing_loss_weight=timing_loss_weight,
                max_timing_residual_s=max_timing_residual_s,
            )
            for record in train_records
        ]
        train_loss = torch.stack([item["total"] for item in train_components]).mean()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_components = [
                _record_losses(
                    model,
                    encoder,
                    record,
                    timing_loss_weight=timing_loss_weight,
                    max_timing_residual_s=max_timing_residual_s,
                )
                for record in validation_records
            ]
        train_summary = _mean_loss_components(train_components)
        validation_summary = _mean_loss_components(validation_components)
        curve.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss.item()),
                "validation_loss": validation_summary["total"],
                "train_candidate_loss": train_summary["candidate"],
                "validation_candidate_loss": validation_summary["candidate"],
                "train_group_loss": train_summary["group"],
                "validation_group_loss": validation_summary["group"],
                "train_timing_loss": train_summary["timing"],
                "validation_timing_loss": validation_summary["timing"],
            }
        )

    model.eval()
    train_ranking_metrics = _ranking_metrics(model, encoder, train_records)
    validation_ranking_metrics = _ranking_metrics(model, encoder, validation_records)
    validation_rollout_evaluation = _offline_rollout_evaluation(
        model,
        encoder,
        policy_config,
        validation_records,
    )
    rollout_records = validation_records + held_out_records
    rollout_evaluation = _offline_rollout_evaluation(
        model,
        encoder,
        policy_config,
        rollout_records,
    )
    final_curve = curve[-1]
    metrics: dict[str, float] = {
        "train_loss": final_curve["train_loss"],
        "validation_loss": final_curve["validation_loss"],
        "train_candidate_topk_recall": train_ranking_metrics[
            "candidate_topk_recall"
        ],
        "validation_candidate_topk_recall": validation_ranking_metrics[
            "candidate_topk_recall"
        ],
        "train_group_top1_accuracy": train_ranking_metrics[
            "group_top1_accuracy"
        ],
        "validation_group_top1_accuracy": validation_ranking_metrics[
            "group_top1_accuracy"
        ],
        "validation_timing_mae_s": validation_ranking_metrics["timing_mae_s"],
        "validation_exact_selection_rate": float(
            validation_rollout_evaluation["exact_teacher_selection_rate"]
        ),
        "validation_schema_valid_rate": float(
            validation_rollout_evaluation["schema_valid_rate"]
        ),
        "validation_assignment_feasible_rate": float(
            validation_rollout_evaluation["assignment_feasible_rate"]
        ),
        "validation_fallback_rate": float(
            validation_rollout_evaluation["fallback_rate"]
        ),
        "num_train_records": float(len(train_records)),
        "num_validation_records": float(len(validation_records)),
        "num_held_out_records": float(len(held_out_records)),
    }

    target_dir = Path(output_dir or base_config.checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / "checkpoint.pt"
    metrics_path = target_dir / "metrics.json"
    loss_curve_path = target_dir / "loss_curve.csv"
    rollout_evaluation_path = target_dir / "rollout_evaluation.json"
    fallback_metadata_path = target_dir / "fallback_metadata.json"
    source_shards = [
        {
            "path": str(path),
            "sha256": hash_file(path),
        }
        for path in paths
    ]
    training_config = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "batch_size": base_config.batch_size,
        "update_rate_hz": base_config.update_rate_hz,
        "encoder_d_model": encoder_d_model,
        "max_timing_residual_s": max_timing_residual_s,
        "timing_loss_weight": timing_loss_weight,
    }
    training_config_hash = stable_hash(training_config)
    dataset_hash = stable_hash(source_shards)
    rollout_evaluation["training_config_hash"] = training_config_hash
    rollout_evaluation["dataset_hash"] = dataset_hash
    torch.save(
        {
            "model_type": "P4_3HighLevelRanker",
            "training_version": P4_3_PI_H_TRAINING_VERSION,
            "training_stage": "P4.3c",
            "state_dict": model.state_dict(),
            "policy_config": policy_config_dict(policy_config),
            "training_config": training_config,
            "training_config_hash": training_config_hash,
            "dataset_hash": dataset_hash,
            "metrics": metrics,
            "source_shards": source_shards,
            "teacher": "InteractionTrajectoryRecord.trajectory + selected_candidate_ids",
            "output_contract": "ContactWrenchTrajectory",
            "actuator_command_output": False,
            "deterministic_assignment_feasibility_gate": True,
            "deterministic_fallback": "P4_2DeterministicGraspCarryPlanner",
        },
        checkpoint_path,
    )
    _write_json(
        metrics_path,
        {
            **metrics,
            "training_version": P4_3_PI_H_TRAINING_VERSION,
            "training_stage": "P4.3c",
            "source_shards": source_shards,
            "dataset_hash": dataset_hash,
            "training_config": training_config,
            "training_config_hash": training_config_hash,
            "policy_config": asdict(policy_config),
            "output_contract": "ContactWrenchTrajectory",
            "actuator_command_output": False,
        },
    )
    _write_loss_curve(loss_curve_path, curve)
    _write_json(rollout_evaluation_path, rollout_evaluation)
    _write_json(
        fallback_metadata_path,
        {
            "fallback_class": "P4_2DeterministicGraspCarryPlanner",
            "fallback_available": True,
            "fallback_triggers": [
                "non_finite_score_or_timing",
                "unknown_or_invalid_candidate_or_group_id",
                "missing_candidate_or_group_score",
                "pairwise_conflict_or_unary_invalid_selection",
                "assignment_feasibility_failure",
                "schema_or_monotonic_timing_validation_failure",
            ],
            "hard_safety_source": "evaluate_selected_assignment_feasibility",
            "learned_output_contract": "ContactWrenchTrajectory",
            "actuator_command_output": False,
            "evaluation_fallback_count": rollout_evaluation["fallback_count"],
            "evaluation_record_count": rollout_evaluation["record_count"],
            "training_config_hash": training_config_hash,
            "dataset_hash": dataset_hash,
        },
    )
    return P4_3PiHTrainingManifest(
        output_dir=str(target_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        loss_curve_path=str(loss_curve_path),
        rollout_evaluation_path=str(rollout_evaluation_path),
        fallback_metadata_path=str(fallback_metadata_path),
        metrics=metrics,
    )


def _record_losses(
    model: P4_3HighLevelRanker,
    encoder: ContactCandidateEncoder,
    record: InteractionTrajectoryRecord,
    *,
    timing_loss_weight: float,
    max_timing_residual_s: float,
) -> dict[str, torch.Tensor]:
    context = _context_from_record(record)
    feasibility = evaluate_selected_candidate_ids(
        context,
        list(record.selected_candidate_ids),
        update_cache=False,
    )
    if not feasibility.feasible:
        raise SchemaValidationError(
            "pi_H teacher selection failed deterministic assignment feasibility: "
            + ",".join(feasibility.violation_codes)
        )
    encoding = encoder.encode(record.contact_candidate_set)
    candidate_features = torch.tensor(
        encoding.candidate_tokens(), dtype=torch.float32
    )
    group_features = torch.tensor(
        encoding.group_tokens(), dtype=torch.float32
    ).reshape((-1, model.d_model))
    candidate_mask = torch.tensor(
        encoding.candidate_valid_mask(), dtype=torch.bool
    )
    group_mask = torch.tensor(encoding.group_valid_mask(), dtype=torch.bool)
    candidate_logits, group_logits, timing_residual = model(
        candidate_features,
        group_features,
        candidate_mask=candidate_mask,
        group_mask=group_mask,
    )
    candidate_ids = encoding.candidate_ids[0][: encoding.candidate_counts[0]]
    teacher_ids = set(record.selected_candidate_ids)
    invalid_teacher_ids = {
        candidate_id
        for index, candidate_id in enumerate(candidate_ids)
        if candidate_id in teacher_ids and not candidate_mask[index]
    }
    if invalid_teacher_ids:
        raise SchemaValidationError(
            f"pi_H teacher selected masked/unary-invalid candidates: {sorted(invalid_teacher_ids)}"
        )
    candidate_targets = torch.tensor(
        [1.0 if candidate_id in teacher_ids else 0.0 for candidate_id in candidate_ids],
        dtype=torch.float32,
    )
    candidate_loss = nn.functional.binary_cross_entropy_with_logits(
        candidate_logits[candidate_mask],
        candidate_targets[candidate_mask],
    )

    valid_group_indices = [
        index for index, valid in enumerate(group_mask.tolist()) if valid
    ]
    if valid_group_indices:
        target_group_index = _teacher_group_index(
            record,
            encoding,
            valid_group_indices,
        )
        local_target = valid_group_indices.index(target_group_index)
        group_loss = nn.functional.cross_entropy(
            group_logits[group_mask].unsqueeze(0),
            torch.tensor([local_target], dtype=torch.long),
        )
    else:
        group_loss = candidate_loss.new_zeros(())

    timing_target = torch.tensor(
        _teacher_timing_residual(record.trajectory, max_timing_residual_s),
        dtype=torch.float32,
    )
    timing_loss = nn.functional.mse_loss(timing_residual, timing_target)
    total_loss = candidate_loss + group_loss + timing_loss_weight * timing_loss
    return {
        "total": total_loss,
        "candidate": candidate_loss,
        "group": group_loss,
        "timing": timing_loss,
    }


def _teacher_group_index(
    record: InteractionTrajectoryRecord,
    encoding,
    valid_group_indices: list[int],
) -> int:
    teacher = set(record.selected_candidate_ids)

    def overlap_key(index: int) -> tuple[float, int, str]:
        members = set(encoding.group_candidate_ids[0][index])
        union = teacher | members
        jaccard = len(teacher & members) / float(max(1, len(union)))
        exact = 1 if members == teacher else 0
        return (float(exact), jaccard, str(encoding.group_ids[0][index]))

    return max(valid_group_indices, key=overlap_key)


def _teacher_timing_residual(
    trajectory: ContactWrenchTrajectory,
    max_timing_residual_s: float,
) -> float:
    if len(trajectory.knots) <= 2:
        return 0.0
    residuals = [
        knot.t_rel_s - float(index) * trajectory.dt_s
        for index, knot in enumerate(trajectory.knots[1:-1], start=1)
    ]
    mean_residual = sum(residuals) / float(len(residuals))
    return max(-max_timing_residual_s, min(max_timing_residual_s, mean_residual))


def _mean_loss_components(
    components: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    return {
        name: float(
            torch.stack([component[name].detach() for component in components])
            .mean()
            .item()
        )
        for name in ("total", "candidate", "group", "timing")
    }


def _ranking_metrics(
    model: P4_3HighLevelRanker,
    encoder: ContactCandidateEncoder,
    records: list[InteractionTrajectoryRecord],
) -> dict[str, float]:
    candidate_recalls: list[float] = []
    group_hits: list[float] = []
    timing_errors: list[float] = []
    with torch.no_grad():
        for record in records:
            encoding = encoder.encode(record.contact_candidate_set)
            prediction = model.predict(encoding)
            teacher_ids = set(record.selected_candidate_ids)
            top_k = sorted(
                prediction.candidate_scores,
                key=lambda candidate_id: (
                    -prediction.candidate_scores[candidate_id],
                    candidate_id,
                ),
            )[: len(teacher_ids)]
            candidate_recalls.append(
                len(teacher_ids.intersection(top_k)) / float(max(1, len(teacher_ids)))
            )
            valid_group_indices = [
                index
                for index, valid in enumerate(encoding.group_valid_mask())
                if valid
            ]
            if valid_group_indices:
                target_index = _teacher_group_index(
                    record,
                    encoding,
                    valid_group_indices,
                )
                target_group_id = str(encoding.group_ids[0][target_index])
                predicted_group_id = max(
                    sorted(prediction.group_scores),
                    key=lambda group_id: prediction.group_scores[group_id],
                )
                group_hits.append(float(predicted_group_id == target_group_id))
            timing_target = _teacher_timing_residual(
                record.trajectory,
                model.max_timing_residual_s,
            )
            timing_errors.append(abs(prediction.timing_residual_s - timing_target))
    return {
        "candidate_topk_recall": sum(candidate_recalls)
        / float(max(1, len(candidate_recalls))),
        "group_top1_accuracy": sum(group_hits) / float(max(1, len(group_hits))),
        "timing_mae_s": sum(timing_errors) / float(max(1, len(timing_errors))),
    }


def _offline_rollout_evaluation(
    model: P4_3HighLevelRanker,
    encoder: ContactCandidateEncoder,
    policy_config: LearnedHighLevelPolicyConfig,
    records: list[InteractionTrajectoryRecord],
) -> dict[str, Any]:
    exact_count = 0
    schema_valid_count = 0
    feasible_count = 0
    fallback_count = 0
    evaluations: list[dict[str, Any]] = []
    policy = LearnedHighLevelPolicy(
        model,
        config=policy_config,
        encoder=encoder,
    )
    for record in records:
        trajectory = policy.plan(_context_from_record(record))
        ContactWrenchTrajectory.from_dict(trajectory.to_dict())
        schema_valid_count += 1
        output_ids = sorted(
            {
                assignment.candidate_id
                for knot in trajectory.knots
                for assignment in knot.contact_assignments
            }
        )
        exact = output_ids == sorted(record.selected_candidate_ids)
        exact_count += int(exact)
        decision = policy.last_decision
        if decision is None:
            raise RuntimeError("LearnedHighLevelPolicy did not expose decision metadata")
        feasible_count += int(decision.assignment_feasible)
        fallback_count += int(decision.used_fallback)
        evaluations.append(
            {
                "record_id": record.record_id,
                "task_id": record.task_id,
                "split": record.split.value,
                "teacher_candidate_ids": sorted(record.selected_candidate_ids),
                "output_candidate_ids": output_ids,
                "exact_teacher_selection": exact,
                "schema_valid": True,
                "assignment_feasible": decision.assignment_feasible,
                "used_fallback": decision.used_fallback,
                "fallback_reason": decision.fallback_reason,
                "timing_residual_s": decision.timing_residual_s,
            }
        )
    count = len(records)
    return {
        "evaluation_type": "offline_teacher_record_decode",
        "training_stage": "P4.3c",
        "record_count": count,
        "exact_teacher_selection_count": exact_count,
        "exact_teacher_selection_rate": exact_count / float(max(1, count)),
        "schema_valid_count": schema_valid_count,
        "schema_valid_rate": schema_valid_count / float(max(1, count)),
        "assignment_feasible_count": feasible_count,
        "assignment_feasible_rate": feasible_count / float(max(1, count)),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / float(max(1, count)),
        "deterministic_fallback_available": True,
        "deterministic_safety_gate_used": True,
        "output_contract": "ContactWrenchTrajectory",
        "actuator_command_output": False,
        "isaac_rollout_claim": False,
        "p4_full_completion_claim": False,
        "records": evaluations,
    }


def _context_from_record(record: InteractionTrajectoryRecord) -> HighLevelPolicyContext:
    return HighLevelPolicyContext(
        irg=record.irg,
        interaction_envelope=record.interaction_envelope,
        morphology_graph=record.morphology_graph,
        contact_candidate_set=record.contact_candidate_set,
        runtime_observation=record.runtime_observation,
    )


def _validate_record_collection(records: list[InteractionTrajectoryRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise SchemaValidationError(
            "pi_H InteractionTrajectoryRecord record_id values must be unique"
        )
    tasks_by_split: dict[DatasetSplit, set[str]] = {
        split: {record.task_id for record in records if record.split == split}
        for split in DatasetSplit
    }
    split_order = list(DatasetSplit)
    for index, left in enumerate(split_order):
        for right in split_order[index + 1 :]:
            overlap = sorted(tasks_by_split[left].intersection(tasks_by_split[right]))
            if overlap:
                raise SchemaValidationError(
                    "pi_H dataset task splits must be disjoint; "
                    f"{left.value}/{right.value} overlap: {overlap}"
                )


def _normalize_paths(
    shard_paths: str | Path | Sequence[str | Path],
) -> list[Path]:
    raw_paths: Iterable[str | Path]
    if isinstance(shard_paths, (str, Path)):
        raw_paths = [shard_paths]
    else:
        raw_paths = shard_paths
    paths = [Path(path) for path in raw_paths]
    if not paths:
        raise ValueError("at least one pi_H JSONL shard path is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pi_H dataset shards do not exist: {missing}")
    return paths


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _write_loss_curve(path: Path, curve: list[dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_candidate_loss",
        "validation_candidate_loss",
        "train_group_loss",
        "validation_group_loss",
        "train_timing_loss",
        "validation_timing_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the minimum P4.3c learned pi_H imitation ranker."
    )
    parser.add_argument("shards", nargs="+", help="InteractionTrajectoryRecord JSONL shards")
    parser.add_argument(
        "--config",
        default="configs/training/p4_3_learning_bootstrap.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--encoder-d-model", type=int)
    parser.add_argument("--max-timing-residual-s", type=float)
    args = parser.parse_args(argv)
    manifest = train_p4_3_pi_h(
        shard_paths=args.shards,
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        encoder_d_model=args.encoder_d_model,
        max_timing_residual_s=args.max_timing_residual_s,
    )
    print(f"checkpoint: {manifest.checkpoint_path}")
    print(f"metrics: {manifest.metrics_path}")
    print(f"loss curve: {manifest.loss_curve_path}")
    print(f"rollout evaluation: {manifest.rollout_evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
