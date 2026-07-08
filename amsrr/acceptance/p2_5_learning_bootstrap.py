from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from amsrr.acceptance.p2_5_inspection import P2_5InspectionCriteria, run_p2_5_inspection
from amsrr.reporting.p2_5_inspection_report import (
    P2_5_LEARNED_NON_PRODUCTION_NOTE,
    generate_p2_5_inspection_report,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.training.p2_feasibility_head_training import train_p2_feasibility_head
from amsrr.training.p2_learned_scorer import train_p2_learned_scorer
from amsrr.training.p2_learning_dataset import build_p2_learning_dataset


@dataclass
class P2_5LearningBootstrapCriteria(SchemaBase):
    config_path: str = "configs/training/p2_design_grasp_carry.yaml"
    output_root: str = "outputs/p2_5"
    dataset_sample_count: int = 64
    inspection_sample_count: int = 1
    seed: int = 0
    epochs: int = 40

    def validate(self) -> None:
        if not self.config_path:
            raise SchemaValidationError("P2_5LearningBootstrapCriteria.config_path must be non-empty")
        if not self.output_root:
            raise SchemaValidationError("P2_5LearningBootstrapCriteria.output_root must be non-empty")
        if self.dataset_sample_count <= 0 or self.inspection_sample_count <= 0:
            raise SchemaValidationError("P2_5LearningBootstrapCriteria sample counts must be positive")
        if self.epochs <= 0:
            raise SchemaValidationError("P2_5LearningBootstrapCriteria.epochs must be positive")


@dataclass
class P2_5LearningBootstrapReport(SchemaBase):
    passed: bool
    dataset_path: str
    dataset_summary_path: str
    train_ids_path: str
    val_ids_path: str
    pi_d_scorer_checkpoint_path: str
    pi_d_scorer_metrics_path: str
    feasibility_head_checkpoint_path: str
    feasibility_head_metrics_path: str
    inspection_report_path: str
    record_count: int
    train_count: int
    val_count: int
    pi_d_scorer_metrics: dict[str, float] = field(default_factory=dict)
    feasibility_head_metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.record_count <= 0 or self.train_count <= 0 or self.val_count <= 0:
            raise SchemaValidationError("P2_5LearningBootstrapReport counts must be positive")
        if self.passed and self.failure_reasons:
            raise SchemaValidationError("P2_5LearningBootstrapReport cannot pass with failure_reasons")


def run_p2_5_learning_bootstrap(
    *,
    criteria: P2_5LearningBootstrapCriteria | None = None,
) -> P2_5LearningBootstrapReport:
    """Run the supervised P2.5 learning bootstrap without replacing deterministic P2 logic."""

    criteria = criteria or P2_5LearningBootstrapCriteria()
    root = Path(criteria.output_root)
    inspection = run_p2_5_inspection(
        criteria=P2_5InspectionCriteria(
            config_path=criteria.config_path,
            output_root=str(root),
            sample_count=criteria.inspection_sample_count,
            seed=criteria.seed,
        )
    )
    dataset = build_p2_learning_dataset(
        config_path=criteria.config_path,
        output_dir=root / "datasets",
        sample_count=criteria.dataset_sample_count,
        seed=criteria.seed,
    )
    scorer = train_p2_learned_scorer(
        dataset_path=dataset.dataset_path,
        train_ids_path=dataset.train_ids_path,
        val_ids_path=dataset.val_ids_path,
        output_dir=root / "training" / "pi_d_scorer",
        epochs=criteria.epochs,
        seed=criteria.seed,
    )
    feasibility = train_p2_feasibility_head(
        dataset_path=dataset.dataset_path,
        train_ids_path=dataset.train_ids_path,
        val_ids_path=dataset.val_ids_path,
        output_dir=root / "training" / "feasibility_head",
        epochs=criteria.epochs,
        seed=criteria.seed + 1,
    )
    report = generate_p2_5_inspection_report(
        trace_dir=root / "candidate_traces",
        visualization_dir=root / "visualization",
        output_dir=root / "report",
        config_path=criteria.config_path,
        dataset_dir=root / "datasets",
        training_dir=root / "training",
    )
    failure_reasons = _failure_reasons(
        inspection_passed=inspection.passed,
        dataset_path=Path(dataset.dataset_path),
        summary_path=Path(dataset.summary_path),
        train_ids_path=Path(dataset.train_ids_path),
        val_ids_path=Path(dataset.val_ids_path),
        scorer_checkpoint=Path(scorer.checkpoint_path),
        scorer_metrics_path=Path(scorer.metrics_path),
        feasibility_checkpoint=Path(feasibility.checkpoint_path),
        feasibility_metrics_path=Path(feasibility.metrics_path),
        report_path=Path(report.report_path),
    )
    return P2_5LearningBootstrapReport(
        passed=not failure_reasons,
        dataset_path=dataset.dataset_path,
        dataset_summary_path=dataset.summary_path,
        train_ids_path=dataset.train_ids_path,
        val_ids_path=dataset.val_ids_path,
        pi_d_scorer_checkpoint_path=scorer.checkpoint_path,
        pi_d_scorer_metrics_path=scorer.metrics_path,
        feasibility_head_checkpoint_path=feasibility.checkpoint_path,
        feasibility_head_metrics_path=feasibility.metrics_path,
        inspection_report_path=report.report_path,
        record_count=dataset.record_count,
        train_count=dataset.train_count,
        val_count=dataset.val_count,
        pi_d_scorer_metrics=scorer.metrics,
        feasibility_head_metrics=feasibility.metrics,
        failure_reasons=failure_reasons,
    )


def _failure_reasons(
    *,
    inspection_passed: bool,
    dataset_path: Path,
    summary_path: Path,
    train_ids_path: Path,
    val_ids_path: Path,
    scorer_checkpoint: Path,
    scorer_metrics_path: Path,
    feasibility_checkpoint: Path,
    feasibility_metrics_path: Path,
    report_path: Path,
) -> list[str]:
    reasons: list[str] = []
    if not inspection_passed:
        reasons.append("P2.5 inspection gate did not pass before learning bootstrap")
    for label, path in (
        ("dataset jsonl", dataset_path),
        ("dataset summary", summary_path),
        ("train ids", train_ids_path),
        ("val ids", val_ids_path),
        ("pi_d scorer checkpoint", scorer_checkpoint),
        ("pi_d scorer metrics", scorer_metrics_path),
        ("feasibility head checkpoint", feasibility_checkpoint),
        ("feasibility head metrics", feasibility_metrics_path),
        ("inspection report", report_path),
    ):
        if not path.exists():
            reasons.append(f"{label} missing: {path}")
    for label, path in (
        ("pi_d scorer", scorer_metrics_path),
        ("feasibility head", feasibility_metrics_path),
    ):
        if path.exists():
            metrics = _read_json(path)
            for key in ("train_loss", "val_loss"):
                if key not in metrics:
                    reasons.append(f"{label} metrics missing {key}")
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        if P2_5_LEARNED_NON_PRODUCTION_NOTE not in report_text:
            reasons.append("report missing learned-model non-production note")
        if "deterministic P2DesignPolicy / FeasibilityChecker remain source of truth" not in report_text:
            reasons.append("report missing deterministic source-of-truth note")
    return reasons


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
