from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.design_policy_p2 import P2DesignCandidateEvaluation, P2DesignPolicy
from amsrr.schemas.common import to_plain_data
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignOutput
from amsrr.training.p2_inspection_context import build_p2_inspection_context


TRACE_JSONL_NAME = "p2_candidate_trace.jsonl"
TRACE_CSV_NAME = "p2_candidate_summary.csv"


@dataclass(frozen=True)
class P2CandidateTraceExportManifest:
    output_dir: str
    jsonl_path: str
    csv_path: str
    record_count: int
    accepted_count: int
    rejected_count: int
    selected_count: int


def export_p2_candidate_traces(
    *,
    config_path: str | Path = "configs/training/p2_design_grasp_carry.yaml",
    output_dir: str | Path = "outputs/p2_5/candidate_traces",
    sample_count: int = 1,
    seed: int = 0,
    include_closed_loop_probe: bool = True,
) -> P2CandidateTraceExportManifest:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        context = build_p2_inspection_context(
            config_path=config_path,
            seed=seed + sample_index,
            sample_index=sample_index,
        )
        policy = P2DesignPolicy(config=context.policy_config)
        candidate_designs = [
            (candidate.variant, candidate.design_output)
            for candidate in context.selection.candidates
        ]
        source_labels = ["policy_variant" for _ in candidate_designs]
        if include_closed_loop_probe:
            candidate_designs.append(
                (
                    "tri_anchor_support_grasp_closed_loop_probe",
                    _closed_loop_probe_design(context.task_spec, context.irg, context.physical_model),
                )
            )
            source_labels.append("closed_loop_invalid_probe")
        expanded_selection = policy.evaluate_design_outputs(context.design_context, candidate_designs)
        selected_candidate_id = expanded_selection.selected_candidate.candidate_id
        for candidate, source_label in zip(expanded_selection.candidates, source_labels):
            records.append(
                _candidate_trace_record(
                    candidate,
                    task_id=context.task_spec.task_id,
                    episode_id=f"p2_5_trace_{sample_index:04d}",
                    sample_id=sample_index,
                    selected=candidate.candidate_id == selected_candidate_id,
                    candidate_source=source_label,
                )
            )
    jsonl_path = target_dir / TRACE_JSONL_NAME
    csv_path = target_dir / TRACE_CSV_NAME
    _write_jsonl(jsonl_path, records)
    _write_csv(csv_path, records)
    accepted_count = sum(1 for record in records if record["accepted"])
    rejected_count = sum(1 for record in records if record["rejected"])
    selected_count = sum(1 for record in records if record["selected"])
    return P2CandidateTraceExportManifest(
        output_dir=str(target_dir),
        jsonl_path=str(jsonl_path),
        csv_path=str(csv_path),
        record_count=len(records),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        selected_count=selected_count,
    )


def _closed_loop_probe_design(task_spec, irg, physical_model) -> DesignOutput:
    good_design = build_grasp_carry_variant_design_output(
        task_spec,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP,
    )
    return replace(
        good_design,
        target_morphology=replace(good_design.target_morphology, is_closed_loop=True),
    )


def _candidate_trace_record(
    candidate: P2DesignCandidateEvaluation,
    *,
    task_id: str,
    episode_id: str,
    sample_id: int,
    selected: bool,
    candidate_source: str,
) -> dict[str, Any]:
    design = candidate.design_output
    morphology = design.target_morphology
    feasibility = candidate.feasibility_result
    hard_codes = [violation.code for violation in feasibility.hard_violations]
    contact_slot_ids = sorted(
        {
            int(slot_id)
            for anchor in morphology.robot_anchors
            for slot_id in anchor.associated_contact_slot_ids
        }
    )
    robot_anchor_ids = [anchor.anchor_id for anchor in sorted(morphology.robot_anchors, key=lambda item: item.anchor_id)]
    control_group_ids = [group.group_id for group in sorted(morphology.control_groups, key=lambda item: item.group_id)]
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "sample_id": sample_id,
        "variant_name": candidate.variant,
        "candidate_source": candidate_source,
        "candidate_id": candidate.candidate_id,
        "selected": selected,
        "accepted": candidate.accepted,
        "rejected": not candidate.accepted,
        "rejection_reason": candidate.rejection_reason,
        "design_score": candidate.soft_score,
        "design_scores": to_plain_data(design.design_scores),
        "feasible": feasibility.feasible,
        "hard_violation_codes": hard_codes,
        "feasibility_proxy_labels": _label_scores(feasibility),
        "feasibility_margins": to_plain_data(feasibility.margins),
        "required_slot_coverage": feasibility.margins.get("required_slot_coverage_ratio", 0.0),
        "anchor_coverage": feasibility.margins.get("required_slot_anchor_coverage_ratio", 0.0),
        "capability_coverage": feasibility.margins.get("required_slot_anchor_capability_coverage_ratio", 0.0),
        "thrust_margin": feasibility.margins.get("thrust_margin_ratio", 0.0),
        "payload_margin": feasibility.margins.get("payload_margin_ratio", 0.0),
        "reachability_margin": feasibility.margins.get("coarse_reachability_ratio", 0.0),
        "module_count": len(morphology.modules),
        "dock_edge_count": len(morphology.dock_edges),
        "base_module_id": morphology.base_module_id,
        "robot_anchor_ids": robot_anchor_ids,
        "contact_slot_ids": contact_slot_ids,
        "control_group_ids": control_group_ids,
    }


def _label_scores(feasibility: FeasibilityResult) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in sorted(feasibility.proxy_scores.items())
        if key.startswith("L_")
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True))
            handle.write("\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_id",
        "sample_id",
        "task_id",
        "variant_name",
        "candidate_source",
        "candidate_id",
        "selected",
        "accepted",
        "rejected",
        "design_score",
        "feasible",
        "hard_violation_codes",
        "required_slot_coverage",
        "anchor_coverage",
        "capability_coverage",
        "thrust_margin",
        "payload_margin",
        "reachability_margin",
        "module_count",
        "dock_edge_count",
        "base_module_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {name: record[name] for name in fieldnames}
            row["hard_violation_codes"] = ";".join(record["hard_violation_codes"])
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export P2 per-candidate evaluation traces.")
    parser.add_argument("--config", default="configs/training/p2_design_grasp_carry.yaml")
    parser.add_argument("--output-dir", default="outputs/p2_5/candidate_traces")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-closed-loop-probe",
        action="store_true",
        help="Do not append the explicit closed-loop invalid rejection probe.",
    )
    args = parser.parse_args(argv)
    manifest = export_p2_candidate_traces(
        config_path=args.config,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        include_closed_loop_probe=not args.no_closed_loop_probe,
    )
    print(f"jsonl: {manifest.jsonl_path}")
    print(f"csv: {manifest.csv_path}")
    print(
        "records: "
        f"{manifest.record_count}, accepted={manifest.accepted_count}, "
        f"rejected={manifest.rejected_count}, selected={manifest.selected_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
