from __future__ import annotations

"""Hash-bound episode evidence and promotion metrics for Order 9 stages."""

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_curriculum import (
    Order9CurriculumStage,
    Order9LearningMode,
    Order9ProductionRuntimeConfig,
    Order9StageMetrics,
)
from amsrr.utils.hashing import hash_file


ORDER9_STAGE_EVALUATION_VERSION = "order9_stage_episode_evaluation_v1"


@dataclass
class Order9EvaluationEpisode(SchemaBase):
    episode_id: str
    task_id: str
    split: DatasetSplit
    random_seed: int
    task_success: bool
    no_fallback_success: bool
    safety_failure: bool
    high_level_decision_count: int
    fallback_decision_count: int
    environment_step_count: int
    isaac_backed: bool
    full_mesh_evaluation: bool
    source_artifact_path: str
    source_artifact_sha256: str
    failure_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("episode_id", "task_id", "source_artifact_path"):
            require_non_empty(
                str(getattr(self, name)), f"Order9EvaluationEpisode.{name}"
            )
        if self.random_seed < 0:
            raise SchemaValidationError(
                "Order9 evaluation random_seed must be non-negative"
            )
        for name in (
            "high_level_decision_count",
            "fallback_decision_count",
            "environment_step_count",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaValidationError(
                    f"Order9EvaluationEpisode.{name} must be non-negative"
                )
        if self.environment_step_count < 1:
            raise SchemaValidationError(
                "Order9 evaluation episode requires positive environment steps"
            )
        if self.fallback_decision_count > self.high_level_decision_count:
            raise SchemaValidationError(
                "Order9 evaluation fallback decisions exceed high-level decisions"
            )
        if self.no_fallback_success and (
            not self.task_success or self.fallback_decision_count != 0
        ):
            raise SchemaValidationError(
                "no-fallback success requires task success and zero fallback decisions"
            )
        if self.task_success and self.safety_failure:
            raise SchemaValidationError(
                "a safety-failed Order9 episode cannot be successful"
            )
        if self.task_success and self.failure_reason is not None:
            raise SchemaValidationError(
                "a successful Order9 episode cannot have a failure reason"
            )
        if not self.task_success and self.failure_reason is None:
            raise SchemaValidationError(
                "a failed Order9 episode requires a failure reason"
            )
        _require_sha256(
            self.source_artifact_sha256,
            "Order9EvaluationEpisode.source_artifact_sha256",
        )
        for key, value in self.metrics.items():
            require_non_empty(key, "Order9EvaluationEpisode.metrics.key")
            if not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"Order9 evaluation metric {key!r} must be finite"
                )


@dataclass
class Order9StageEvaluationReport(SchemaBase):
    evaluation_version: str
    stage_id: str
    schedule_hash: str
    episodes: list[Order9EvaluationEpisode]
    policy_checkpoint_sha256_by_family: dict[str, str]
    training_rollout_environment_step_count: int
    training_rollout_wall_elapsed_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.evaluation_version != ORDER9_STAGE_EVALUATION_VERSION:
            raise SchemaValidationError("Order9 stage evaluation version mismatch")
        require_non_empty(self.stage_id, "Order9StageEvaluationReport.stage_id")
        _require_sha256(
            self.schedule_hash, "Order9StageEvaluationReport.schedule_hash"
        )
        if not self.episodes:
            raise SchemaValidationError(
                "Order9 stage evaluation requires episode evidence"
            )
        identities = [episode.episode_id for episode in self.episodes]
        if len(identities) != len(set(identities)):
            raise SchemaValidationError(
                "Order9 stage evaluation episode ids must be unique"
            )
        for family, digest in self.policy_checkpoint_sha256_by_family.items():
            if family not in Order9PolicyFamily.values():
                raise SchemaValidationError(
                    f"Order9 stage evaluation has unknown policy family {family!r}"
                )
            _require_sha256(
                digest,
                "Order9StageEvaluationReport.policy_checkpoint_sha256_by_family"
                f"[{family!r}]",
            )
        if self.training_rollout_environment_step_count < 0:
            raise SchemaValidationError(
                "Order9 training rollout steps must be non-negative"
            )
        if not math.isfinite(float(self.training_rollout_wall_elapsed_s)) or (
            self.training_rollout_wall_elapsed_s < 0.0
        ):
            raise SchemaValidationError(
                "Order9 training rollout wall time must be finite/non-negative"
            )
        if bool(self.training_rollout_environment_step_count) != bool(
            self.training_rollout_wall_elapsed_s
        ):
            raise SchemaValidationError(
                "Order9 training rollout steps and wall time must both be zero or positive"
            )

    @property
    def aggregate_env_steps_per_s(self) -> float:
        if self.training_rollout_wall_elapsed_s <= 0.0:
            return 0.0
        return (
            self.training_rollout_environment_step_count
            / self.training_rollout_wall_elapsed_s
        )

    def stage_metrics(self) -> Order9StageMetrics:
        return Order9StageMetrics(
            episode_count=len(self.episodes),
            success_count=sum(episode.task_success for episode in self.episodes),
            no_fallback_success_count=sum(
                episode.no_fallback_success for episode in self.episodes
            ),
            safety_failure_episode_count=sum(
                episode.safety_failure for episode in self.episodes
            ),
            high_level_decision_count=sum(
                episode.high_level_decision_count for episode in self.episodes
            ),
            fallback_decision_count=sum(
                episode.fallback_decision_count for episode in self.episodes
            ),
            aggregate_env_steps_per_s=self.aggregate_env_steps_per_s,
        )


def build_order9_stage_evaluation_report(
    *,
    stage: Order9CurriculumStage,
    schedule_hash: str,
    episodes: Sequence[Order9EvaluationEpisode],
    policy_checkpoint_sha256_by_family: Mapping[
        Order9PolicyFamily | str, str
    ] | None = None,
    training_rollout_environment_step_count: int = 0,
    training_rollout_wall_elapsed_s: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> Order9StageEvaluationReport:
    report = Order9StageEvaluationReport(
        evaluation_version=ORDER9_STAGE_EVALUATION_VERSION,
        stage_id=stage.stage_id,
        schedule_hash=schedule_hash,
        episodes=list(episodes),
        policy_checkpoint_sha256_by_family={
            Order9PolicyFamily(family).value: str(digest)
            for family, digest in (policy_checkpoint_sha256_by_family or {}).items()
        },
        training_rollout_environment_step_count=int(
            training_rollout_environment_step_count
        ),
        training_rollout_wall_elapsed_s=float(training_rollout_wall_elapsed_s),
        metadata={
            "metrics_derived_from_episode_rows": True,
            "aggregate_throughput_excludes_full_mesh_evaluation": True,
            **dict(metadata or {}),
        },
    )
    report.validate()
    return report


def validate_order9_stage_evaluation_report(
    report: Order9StageEvaluationReport,
    *,
    stage: Order9CurriculumStage,
    schedule_hash: str,
    runtime: Order9ProductionRuntimeConfig,
    verify_source_artifacts: bool = True,
) -> Order9StageMetrics:
    report.validate()
    if report.stage_id != stage.stage_id or report.schedule_hash != schedule_hash:
        raise SchemaValidationError(
            "Order9 evaluation report does not match the curriculum stage"
        )
    expected_split = (
        DatasetSplit.HELD_OUT if stage.held_out_only else DatasetSplit.VALIDATION
    )
    if stage.learning_mode != Order9LearningMode.COLLECTION and any(
        episode.split != expected_split for episode in report.episodes
    ):
        raise SchemaValidationError(
            f"Order9 promotion evidence must use {expected_split.value} tasks"
        )
    if any(not episode.isaac_backed for episode in report.episodes):
        raise SchemaValidationError(
            "Order9 promotion evidence must be Isaac-backed"
        )
    if stage.learning_mode in {
        Order9LearningMode.PPO,
        Order9LearningMode.EVALUATION,
    }:
        full_mesh_count = sum(
            episode.full_mesh_evaluation for episode in report.episodes
        )
        if full_mesh_count < runtime.full_mesh_evaluation_episode_count:
            raise SchemaValidationError(
                "Order9 promotion report lacks periodic unchanged full-mesh episodes"
            )
    if stage.learning_mode == Order9LearningMode.PPO and (
        report.training_rollout_environment_step_count < 1
        or report.training_rollout_wall_elapsed_s <= 0.0
    ):
        raise SchemaValidationError(
            "Order9 PPO promotion requires measured training-rollout throughput"
        )
    if verify_source_artifacts:
        verified: dict[str, str] = {}
        for episode in report.episodes:
            path = episode.source_artifact_path
            prior = verified.get(path)
            if prior is not None and prior != episode.source_artifact_sha256:
                raise SchemaValidationError(
                    "Order9 evaluation source path has conflicting hashes"
                )
            if prior is None:
                actual = hash_file(path)
                if actual != episode.source_artifact_sha256:
                    raise SchemaValidationError(
                        "Order9 evaluation source artifact hash mismatch"
                    )
                verified[path] = actual
    metrics = report.stage_metrics()
    metrics.validate()
    return metrics


def load_order9_stage_evaluation_report(
    path: str | Path,
) -> Order9StageEvaluationReport:
    return Order9StageEvaluationReport.from_json(
        Path(path).read_text(encoding="utf-8")
    )


def write_order9_stage_evaluation_report(
    path: str | Path,
    report: Order9StageEvaluationReport,
) -> None:
    report.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(report.to_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")


__all__ = [
    "ORDER9_STAGE_EVALUATION_VERSION",
    "Order9EvaluationEpisode",
    "Order9StageEvaluationReport",
    "build_order9_stage_evaluation_report",
    "load_order9_stage_evaluation_report",
    "validate_order9_stage_evaluation_report",
    "write_order9_stage_evaluation_report",
]
