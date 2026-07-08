from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from amsrr.reporting.p2_5_inspection_report import (
    P2_5_LEARNED_NON_PRODUCTION_NOTE,
    P2_5_NON_EXECUTION_NOTE,
    generate_p2_5_inspection_report,
)
from amsrr.training.p2_candidate_trace_export import export_p2_candidate_traces
from amsrr.visualization.p2_morphology import render_p2_morphology_visualizations


def test_p2_5_inspection_report_contains_summary_and_scope_notes(tmp_path: Path) -> None:
    trace_dir = tmp_path / "candidate_traces"
    visualization_dir = tmp_path / "visualization"
    report_dir = tmp_path / "report"
    export_p2_candidate_traces(output_dir=trace_dir, sample_count=1, seed=0)
    render_p2_morphology_visualizations(output_dir=visualization_dir, seed=0)

    manifest = generate_p2_5_inspection_report(
        trace_dir=trace_dir,
        visualization_dir=visualization_dir,
        output_dir=report_dir,
        generated_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
    )

    report_path = Path(manifest.report_path)
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "# P2.5 Inspection Report" in text
    assert "Variant Counts" in text
    assert "Violation Code Histogram" in text
    assert "Representative Selected Designs" in text
    assert "Representative Rejected Designs" in text
    assert "Learning Bootstrap Artifacts" in text
    assert P2_5_NON_EXECUTION_NOTE in text
    assert P2_5_LEARNED_NON_PRODUCTION_NOTE in text
    assert "Isaac" in text
    assert "π_H / π_L" in text
    assert "QP/PID" in text
    assert "supervised training" in text
    assert "deterministic P2DesignPolicy / FeasibilityChecker remain source of truth" in text
    assert manifest.record_count == 5
    assert manifest.accepted_count == 4
    assert manifest.rejected_count == 1
    assert manifest.selected_count == 1
