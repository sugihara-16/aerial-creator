from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P2_5InspectionCriteria, run_p2_5_inspection
from amsrr.reporting.p2_5_inspection_report import P2_5_LEARNED_NON_PRODUCTION_NOTE, P2_5_NON_EXECUTION_NOTE


def test_p2_5_inspection_acceptance_gate(tmp_path: Path) -> None:
    report = run_p2_5_inspection(
        criteria=P2_5InspectionCriteria(
            output_root=str(tmp_path / "p2_5"),
            sample_count=1,
            seed=0,
        )
    )

    assert report.passed, report.failure_reasons
    assert len(report.visualization_files) == 8
    for path in report.visualization_files:
        assert Path(path).exists()
    assert Path(report.candidate_trace_jsonl_path).exists()
    assert Path(report.candidate_trace_csv_path).exists()
    assert Path(report.inspection_report_path).exists()
    assert report.candidate_record_count == 5
    assert report.accepted_count >= 1
    assert report.rejected_count >= 1
    assert report.selected_count == 1
    text = Path(report.inspection_report_path).read_text(encoding="utf-8")
    assert P2_5_NON_EXECUTION_NOTE in text
    assert P2_5_LEARNED_NON_PRODUCTION_NOTE in text
    assert "actuator command" in text
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
