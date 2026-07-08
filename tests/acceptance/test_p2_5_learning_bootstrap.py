from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P2_5LearningBootstrapCriteria, run_p2_5_learning_bootstrap
from amsrr.reporting.p2_5_inspection_report import P2_5_LEARNED_NON_PRODUCTION_NOTE


def test_p2_5_learning_bootstrap_acceptance_gate(tmp_path: Path) -> None:
    report = run_p2_5_learning_bootstrap(
        criteria=P2_5LearningBootstrapCriteria(
            output_root=str(tmp_path / "p2_5"),
            dataset_sample_count=8,
            inspection_sample_count=1,
            epochs=5,
            seed=5,
        )
    )

    assert report.passed, report.failure_reasons
    assert report.record_count == 40
    assert report.train_count > 0
    assert report.val_count > 0
    for path in (
        report.dataset_path,
        report.dataset_summary_path,
        report.train_ids_path,
        report.val_ids_path,
        report.pi_d_scorer_checkpoint_path,
        report.pi_d_scorer_metrics_path,
        report.feasibility_head_checkpoint_path,
        report.feasibility_head_metrics_path,
        report.inspection_report_path,
    ):
        assert Path(path).exists()
    assert "train_loss" in report.pi_d_scorer_metrics
    assert "val_loss" in report.pi_d_scorer_metrics
    assert "train_loss" in report.feasibility_head_metrics
    assert "val_loss" in report.feasibility_head_metrics
    report_text = Path(report.inspection_report_path).read_text(encoding="utf-8")
    assert P2_5_LEARNED_NON_PRODUCTION_NOTE in report_text
    assert "deterministic P2DesignPolicy / FeasibilityChecker remain source of truth" in report_text
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
