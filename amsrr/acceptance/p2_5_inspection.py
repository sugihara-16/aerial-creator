from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.morphology.grasp_carry_designs import GRASP_CARRY_VARIANT_ORDER
from amsrr.reporting.p2_5_inspection_report import (
    P2_5_LEARNED_NON_PRODUCTION_NOTE,
    P2_5_NON_EXECUTION_NOTE,
    generate_p2_5_inspection_report,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.training.p2_candidate_trace_export import export_p2_candidate_traces
from amsrr.visualization.p2_morphology import render_p2_morphology_visualizations


@dataclass
class P2_5InspectionCriteria(SchemaBase):
    config_path: str = "configs/training/p2_design_grasp_carry.yaml"
    output_root: str = "outputs/p2_5"
    sample_count: int = 1
    seed: int = 0

    def validate(self) -> None:
        if not self.config_path:
            raise SchemaValidationError("P2_5InspectionCriteria.config_path must be non-empty")
        if not self.output_root:
            raise SchemaValidationError("P2_5InspectionCriteria.output_root must be non-empty")
        if self.sample_count <= 0:
            raise SchemaValidationError("P2_5InspectionCriteria.sample_count must be positive")


@dataclass
class P2_5InspectionReport(SchemaBase):
    passed: bool
    visualization_files: list[str] = field(default_factory=list)
    candidate_trace_jsonl_path: str = ""
    candidate_trace_csv_path: str = ""
    inspection_report_path: str = ""
    candidate_record_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    selected_count: int = 0
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.candidate_record_count < 0:
            raise SchemaValidationError("P2_5InspectionReport.candidate_record_count must be non-negative")
        if min(self.accepted_count, self.rejected_count, self.selected_count) < 0:
            raise SchemaValidationError("P2_5InspectionReport counts must be non-negative")
        if self.passed and self.failure_reasons:
            raise SchemaValidationError("P2_5InspectionReport cannot pass with failure_reasons")


def run_p2_5_inspection(
    *,
    criteria: P2_5InspectionCriteria | None = None,
) -> P2_5InspectionReport:
    """Run the P2.5 pre-P3 inspection and debugging gate."""

    criteria = criteria or P2_5InspectionCriteria()
    root = Path(criteria.output_root)
    visualization_dir = root / "visualization"
    trace_dir = root / "candidate_traces"
    report_dir = root / "report"
    visualization = render_p2_morphology_visualizations(
        config_path=criteria.config_path,
        output_dir=visualization_dir,
        seed=criteria.seed,
        sample_index=0,
    )
    traces = export_p2_candidate_traces(
        config_path=criteria.config_path,
        output_dir=trace_dir,
        sample_count=criteria.sample_count,
        seed=criteria.seed,
        include_closed_loop_probe=True,
    )
    report = generate_p2_5_inspection_report(
        trace_dir=trace_dir,
        visualization_dir=visualization_dir,
        output_dir=report_dir,
        config_path=criteria.config_path,
        dataset_dir=root / "datasets",
        training_dir=root / "training",
    )
    records = _read_jsonl(Path(traces.jsonl_path))
    visualization_files = sorted(list(visualization.graph_files.values()) + list(visualization.layout_files.values()))
    failure_reasons = _failure_reasons(
        visualization_files=visualization_files,
        records=records,
        jsonl_path=Path(traces.jsonl_path),
        csv_path=Path(traces.csv_path),
        report_path=Path(report.report_path),
    )
    return P2_5InspectionReport(
        passed=not failure_reasons,
        visualization_files=visualization_files,
        candidate_trace_jsonl_path=traces.jsonl_path,
        candidate_trace_csv_path=traces.csv_path,
        inspection_report_path=report.report_path,
        candidate_record_count=len(records),
        accepted_count=sum(1 for record in records if record["accepted"]),
        rejected_count=sum(1 for record in records if record["rejected"]),
        selected_count=sum(1 for record in records if record["selected"]),
        failure_reasons=failure_reasons,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _failure_reasons(
    *,
    visualization_files: list[str],
    records: list[dict[str, Any]],
    jsonl_path: Path,
    csv_path: Path,
    report_path: Path,
) -> list[str]:
    reasons: list[str] = []
    variant_values = {variant.value for variant in GRASP_CARRY_VARIANT_ORDER}
    expected_visualization_count = len(variant_values) * 2
    if len(visualization_files) != expected_visualization_count:
        reasons.append(
            f"expected {expected_visualization_count} visualization files, got {len(visualization_files)}"
        )
    for variant in variant_values:
        expected = {f"{variant}_graph.svg", f"{variant}_layout.svg"}
        present = {Path(path).name for path in visualization_files if Path(path).name.startswith(variant)}
        if not expected <= present:
            reasons.append(f"visualization files missing for variant {variant}")
    if not jsonl_path.exists():
        reasons.append(f"candidate trace JSONL missing: {jsonl_path}")
    if not csv_path.exists():
        reasons.append(f"candidate trace CSV missing: {csv_path}")
    if not records:
        reasons.append("candidate trace has no records")
    if records and not any(record.get("selected") for record in records):
        reasons.append("candidate trace has no selected candidate")
    if records and not any(record.get("accepted") for record in records):
        reasons.append("candidate trace has no accepted candidate")
    if records and not any(record.get("rejected") for record in records):
        reasons.append("candidate trace has no rejected candidate")
    for index, record in enumerate(records):
        missing = _missing_trace_fields(record)
        if missing:
            reasons.append(f"candidate trace record {index} missing fields: {missing}")
    if not report_path.exists():
        reasons.append(f"inspection report missing: {report_path}")
    else:
        report_text = report_path.read_text(encoding="utf-8")
        if P2_5_NON_EXECUTION_NOTE not in report_text:
            reasons.append("inspection report missing non-execution note")
        for phrase in ("Isaac", "π_H", "π_L", "QP/PID", "actuator command"):
            if phrase not in report_text:
                reasons.append(f"inspection report missing scope phrase: {phrase}")
        if P2_5_LEARNED_NON_PRODUCTION_NOTE not in report_text:
            reasons.append("inspection report missing learned-model non-production note")
    return reasons


def _missing_trace_fields(record: dict[str, Any]) -> list[str]:
    required_fields = [
        "task_id",
        "episode_id",
        "variant_name",
        "candidate_id",
        "selected",
        "accepted",
        "rejected",
        "design_score",
        "design_scores",
        "feasible",
        "hard_violation_codes",
        "feasibility_proxy_labels",
        "feasibility_margins",
        "required_slot_coverage",
        "anchor_coverage",
        "capability_coverage",
        "thrust_margin",
        "payload_margin",
        "reachability_margin",
        "module_count",
        "dock_edge_count",
        "base_module_id",
        "robot_anchor_ids",
        "contact_slot_ids",
        "control_group_ids",
    ]
    missing = [field for field in required_fields if field not in record]
    if "feasibility_proxy_labels" in record and not record["feasibility_proxy_labels"]:
        missing.append("feasibility_proxy_labels(non-empty)")
    if "feasibility_margins" in record and not record["feasibility_margins"]:
        missing.append("feasibility_margins(non-empty)")
    if "design_scores" in record and not record["design_scores"]:
        missing.append("design_scores(non-empty)")
    return missing
