from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amsrr.training.p2_candidate_trace_export import TRACE_JSONL_NAME


REPORT_NAME = "p2_5_inspection_report.md"
P2_5_NON_EXECUTION_NOTE = "P2.5 inspection path は Isaac / π_H / π_L / QP/PID / actuator command execution を実行しない"
P2_5_LEARNED_NON_PRODUCTION_NOTE = "learned models are NOT used in production path"


@dataclass(frozen=True)
class P2_5InspectionReportManifest:
    output_dir: str
    report_path: str
    record_count: int
    accepted_count: int
    rejected_count: int
    selected_count: int


def generate_p2_5_inspection_report(
    *,
    trace_dir: str | Path = "outputs/p2_5/candidate_traces",
    visualization_dir: str | Path = "outputs/p2_5/visualization",
    output_dir: str | Path = "outputs/p2_5/report",
    config_path: str | Path = "configs/training/p2_design_grasp_carry.yaml",
    dataset_dir: str | Path = "outputs/p2_5/datasets",
    training_dir: str | Path = "outputs/p2_5/training",
    generated_at: datetime | None = None,
) -> P2_5InspectionReportManifest:
    trace_path = Path(trace_dir) / TRACE_JSONL_NAME
    records = _read_trace_records(trace_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / REPORT_NAME
    generated = generated_at or datetime.now(timezone.utc)
    report_path.write_text(
        _report_markdown(
            records,
            config_path=str(config_path),
            generated_at=generated,
            trace_path=trace_path,
            visualization_dir=Path(visualization_dir),
            dataset_dir=Path(dataset_dir),
            training_dir=Path(training_dir),
            report_dir=target_dir,
        ),
        encoding="utf-8",
    )
    return P2_5InspectionReportManifest(
        output_dir=str(target_dir),
        report_path=str(report_path),
        record_count=len(records),
        accepted_count=sum(1 for record in records if record["accepted"]),
        rejected_count=sum(1 for record in records if record["rejected"]),
        selected_count=sum(1 for record in records if record["selected"]),
    )


def _read_trace_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _report_markdown(
    records: list[dict[str, Any]],
    *,
    config_path: str,
    generated_at: datetime,
    trace_path: Path,
    visualization_dir: Path,
    dataset_dir: Path,
    training_dir: Path,
    report_dir: Path,
) -> str:
    variant_counts = Counter(record["variant_name"] for record in records)
    selected_counts = Counter(record["variant_name"] for record in records if record["selected"])
    accepted_count = sum(1 for record in records if record["accepted"])
    rejected_count = sum(1 for record in records if record["rejected"])
    selected_count = sum(1 for record in records if record["selected"])
    sample_count = len({record["sample_id"] for record in records})
    valid_design_rate = accepted_count / max(1.0, float(accepted_count + rejected_count))
    violation_histogram = Counter(
        code
        for record in records
        for code in record.get("hard_violation_codes", [])
    )
    coverage_summary = _numeric_summary(records, "required_slot_coverage")
    margin_summaries = {
        "anchor_coverage": _numeric_summary(records, "anchor_coverage"),
        "capability_coverage": _numeric_summary(records, "capability_coverage"),
        "thrust_margin": _numeric_summary(records, "thrust_margin"),
        "payload_margin": _numeric_summary(records, "payload_margin"),
        "reachability_margin": _numeric_summary(records, "reachability_margin"),
    }
    learning = _learning_artifacts(dataset_dir, training_dir)
    selected_examples = [record for record in records if record["selected"]][:3]
    rejected_examples = [record for record in records if record["rejected"]][:3]
    lines = [
        "# P2.5 Inspection Report",
        "",
        f"- Config path: `{config_path}`",
        f"- Generated at: `{generated_at.isoformat()}`",
        f"- Trace JSONL: `{_relative_link(trace_path, report_dir)}`",
        f"- Sample count: `{sample_count}`",
        f"- Candidate records: `{len(records)}`",
        f"- Accepted / rejected / selected: `{accepted_count}` / `{rejected_count}` / `{selected_count}`",
        f"- Valid design rate over exported candidates: `{valid_design_rate:.3f}`",
        "",
        "P2 は learned design ではなく deterministic scaffold です。",
        f"{P2_5_NON_EXECUTION_NOTE}。",
        f"{P2_5_LEARNED_NON_PRODUCTION_NOTE}。",
        "deterministic P2DesignPolicy / FeasibilityChecker remain source of truth.",
        "",
        "## Variant Counts",
        "",
        _counter_table(variant_counts, ["variant", "count"]),
        "",
        "## Selected Variant Distribution",
        "",
        _counter_table(selected_counts, ["variant", "selected_count"]),
        "",
        "## Required Slot Coverage Summary",
        "",
        _summary_table({"required_slot_coverage": coverage_summary}),
        "",
        "## Feasibility Margin Summary",
        "",
        _summary_table(margin_summaries),
        "",
        "## Violation Code Histogram",
        "",
        _counter_table(violation_histogram, ["violation_code", "count"]),
        "",
        "## Visualization Files",
        "",
        _visualization_table(visualization_dir, report_dir),
        "",
        "## Representative Selected Designs",
        "",
        _representative_table(selected_examples),
        "",
        "## Representative Rejected Designs",
        "",
        _representative_table(rejected_examples),
        "",
        "## Learning Bootstrap Artifacts",
        "",
        _learning_section(learning, report_dir),
        "",
        "## Scope Notes",
        "",
        "- P2 completion gate は変更していません。",
        "- P2.5 は P3 に進む前の inspection / debugging phase です。",
        "- P2.5 learning bootstrap は supervised training のみであり、full RL ではありません。",
        "- P2.5 では Isaac、π_H / π_L、QP/PID、actuator command execution は未実行です。",
        "- learned models are NOT used in production path.",
        "- deterministic P2DesignPolicy / FeasibilityChecker remain source of truth.",
        "- P3 に進む前に、人間が visualization とこの report を確認してください。",
        "",
    ]
    return "\n".join(lines)


def _numeric_summary(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(record.get(key, 0.0)) for record in records]
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": sum(values) / float(len(values)),
        "max": max(values),
    }


def _counter_table(counter: Counter, headers: list[str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | ---: |"]
    if not counter:
        lines.append("| none | 0 |")
        return "\n".join(lines)
    for key, value in sorted(counter.items()):
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def _summary_table(summaries: dict[str, dict[str, float]]) -> str:
    lines = ["| metric | min | mean | max |", "| --- | ---: | ---: | ---: |"]
    for key, summary in sorted(summaries.items()):
        lines.append(f"| `{key}` | {summary['min']:.3f} | {summary['mean']:.3f} | {summary['max']:.3f} |")
    return "\n".join(lines)


def _visualization_table(visualization_dir: Path, report_dir: Path) -> str:
    graph_files = sorted(visualization_dir.glob("*_graph.svg"))
    layout_files = {path.name.replace("_layout.svg", ""): path for path in visualization_dir.glob("*_layout.svg")}
    lines = ["| variant | graph view | simple 3D layout view |", "| --- | --- | --- |"]
    for graph_path in graph_files:
        variant = graph_path.name.replace("_graph.svg", "")
        layout_path = layout_files.get(variant)
        graph_link = _relative_link(graph_path, report_dir)
        layout_link = _relative_link(layout_path, report_dir) if layout_path is not None else "missing"
        lines.append(f"| `{variant}` | [{graph_path.name}]({graph_link}) | [{layout_path.name if layout_path else 'missing'}]({layout_link}) |")
    if len(lines) == 2:
        lines.append("| none | missing | missing |")
    return "\n".join(lines)


def _representative_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| episode | variant | source | selected | accepted | score | violations |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not records:
        lines.append("| none | none | none | 0 | 0 | 0.000 | none |")
        return "\n".join(lines)
    for record in records:
        violations = ", ".join(record["hard_violation_codes"]) if record["hard_violation_codes"] else "none"
        lines.append(
            f"| `{record['episode_id']}` | `{record['variant_name']}` | `{record['candidate_source']}` | "
            f"{int(record['selected'])} | {int(record['accepted'])} | {float(record['design_score']):.3f} | {violations} |"
        )
    return "\n".join(lines)


def _learning_artifacts(dataset_dir: Path, training_dir: Path) -> dict[str, Any]:
    dataset_summary_path = dataset_dir / "p2_candidate_dataset_summary.json"
    scorer_metrics_path = training_dir / "pi_d_scorer" / "metrics.json"
    feasibility_metrics_path = training_dir / "feasibility_head" / "metrics.json"
    return {
        "dataset_path": dataset_dir / "p2_candidate_dataset.jsonl",
        "dataset_summary_path": dataset_summary_path,
        "train_ids_path": dataset_dir / "train_ids.json",
        "val_ids_path": dataset_dir / "val_ids.json",
        "dataset_summary": _read_optional_json(dataset_summary_path),
        "pi_d_scorer_checkpoint_path": training_dir / "pi_d_scorer" / "checkpoint.pt",
        "pi_d_scorer_metrics_path": scorer_metrics_path,
        "pi_d_scorer_metrics": _read_optional_json(scorer_metrics_path),
        "feasibility_head_checkpoint_path": training_dir / "feasibility_head" / "checkpoint.pt",
        "feasibility_head_metrics_path": feasibility_metrics_path,
        "feasibility_head_metrics": _read_optional_json(feasibility_metrics_path),
    }


def _learning_section(learning: dict[str, Any], report_dir: Path) -> str:
    summary = learning["dataset_summary"] or {}
    scorer_metrics = learning["pi_d_scorer_metrics"] or {}
    feasibility_metrics = learning["feasibility_head_metrics"] or {}
    lines = [
        f"- Dataset path: `{_relative_link(learning['dataset_path'], report_dir)}`",
        f"- Dataset summary: `{_relative_link(learning['dataset_summary_path'], report_dir)}`",
        f"- Train IDs: `{_relative_link(learning['train_ids_path'], report_dir)}`",
        f"- Val IDs: `{_relative_link(learning['val_ids_path'], report_dir)}`",
        f"- Train / val sample count: `{summary.get('train_count', 'missing')}` / `{summary.get('val_count', 'missing')}`",
        f"- π_D scorer checkpoint: `{_relative_link(learning['pi_d_scorer_checkpoint_path'], report_dir)}`",
        f"- π_D scorer metrics: `{_relative_link(learning['pi_d_scorer_metrics_path'], report_dir)}`",
        f"- Feasibility head checkpoint: `{_relative_link(learning['feasibility_head_checkpoint_path'], report_dir)}`",
        f"- Feasibility head metrics: `{_relative_link(learning['feasibility_head_metrics_path'], report_dir)}`",
        "",
        "π_D scorer metrics:",
        "",
        _metrics_table(scorer_metrics),
        "",
        "Feasibility head metrics:",
        "",
        _metrics_table(feasibility_metrics),
        "",
        f"{P2_5_LEARNED_NON_PRODUCTION_NOTE}.",
        "deterministic P2DesignPolicy / FeasibilityChecker remain source of truth.",
    ]
    return "\n".join(lines)


def _metrics_table(metrics: dict[str, Any]) -> str:
    lines = ["| metric | value |", "| --- | ---: |"]
    if not metrics:
        lines.append("| missing | 0 |")
        return "\n".join(lines)
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            lines.append(f"| `{key}` | {float(value):.6f} |")
        else:
            lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative_link(path: Path | None, base_dir: Path) -> str:
    if path is None:
        return "missing"
    return os.path.relpath(path.resolve(), base_dir.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a P2.5 human inspection report.")
    parser.add_argument("--trace-dir", default="outputs/p2_5/candidate_traces")
    parser.add_argument("--visualization-dir", default="outputs/p2_5/visualization")
    parser.add_argument("--output-dir", default="outputs/p2_5/report")
    parser.add_argument("--config", default="configs/training/p2_design_grasp_carry.yaml")
    parser.add_argument("--dataset-dir", default="outputs/p2_5/datasets")
    parser.add_argument("--training-dir", default="outputs/p2_5/training")
    args = parser.parse_args(argv)
    manifest = generate_p2_5_inspection_report(
        trace_dir=args.trace_dir,
        visualization_dir=args.visualization_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        training_dir=args.training_dir,
    )
    print(f"report: {manifest.report_path}")
    print(
        "records: "
        f"{manifest.record_count}, accepted={manifest.accepted_count}, "
        f"rejected={manifest.rejected_count}, selected={manifest.selected_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
