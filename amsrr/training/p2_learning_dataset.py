from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amsrr.morphology.grasp_carry_designs import GRASP_CARRY_VARIANT_ORDER
from amsrr.training.p2_candidate_trace_export import build_p2_candidate_trace_records


DATASET_JSONL_NAME = "p2_candidate_dataset.jsonl"
DATASET_SUMMARY_NAME = "p2_candidate_dataset_summary.json"
TRAIN_IDS_NAME = "train_ids.json"
VAL_IDS_NAME = "val_ids.json"
P2_LEARNING_FEATURE_NAMES = [
    "variant_chain_grasp",
    "variant_symmetric_two_anchor_grasp",
    "variant_tri_anchor_support_grasp",
    "variant_central_base_plus_two_grasp_arms",
    "candidate_source_closed_loop_probe",
    "candidate_id",
    "required_slot_coverage",
    "anchor_coverage",
    "capability_coverage",
    "thrust_margin",
    "payload_margin",
    "reachability_margin",
    "module_count",
    "dock_edge_count",
    "robot_anchor_count",
    "contact_slot_count",
    "control_group_count",
    "port_conflict_count",
    "closed_loop_rejected",
]


@dataclass(frozen=True)
class P2LearningDatasetManifest:
    output_dir: str
    dataset_path: str
    summary_path: str
    train_ids_path: str
    val_ids_path: str
    record_count: int
    train_count: int
    val_count: int
    accepted_count: int
    rejected_count: int
    selected_count: int


def build_p2_learning_dataset(
    *,
    config_path: str | Path = "configs/training/p2_design_grasp_carry.yaml",
    output_dir: str | Path = "outputs/p2_5/datasets",
    sample_count: int = 64,
    seed: int = 0,
    val_fraction: float = 0.20,
) -> P2LearningDatasetManifest:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_records = build_p2_candidate_trace_records(
        config_path=config_path,
        sample_count=sample_count,
        seed=seed,
        include_closed_loop_probe=True,
    )
    records = [_dataset_record(record, row_index=index) for index, record in enumerate(raw_records)]
    train_ids, val_ids = _train_val_split(records, val_fraction=val_fraction)
    dataset_path = target_dir / DATASET_JSONL_NAME
    summary_path = target_dir / DATASET_SUMMARY_NAME
    train_ids_path = target_dir / TRAIN_IDS_NAME
    val_ids_path = target_dir / VAL_IDS_NAME
    _write_jsonl(dataset_path, records)
    _write_json(train_ids_path, train_ids)
    _write_json(val_ids_path, val_ids)
    summary = _summary(records, train_ids, val_ids, config_path=str(config_path), sample_count=sample_count)
    _write_json(summary_path, summary)
    return P2LearningDatasetManifest(
        output_dir=str(target_dir),
        dataset_path=str(dataset_path),
        summary_path=str(summary_path),
        train_ids_path=str(train_ids_path),
        val_ids_path=str(val_ids_path),
        record_count=len(records),
        train_count=len(train_ids),
        val_count=len(val_ids),
        accepted_count=sum(1 for record in records if record["accepted_label"] == 1),
        rejected_count=sum(1 for record in records if record["accepted_label"] == 0),
        selected_count=sum(1 for record in records if record["selected_label"] == 1),
    )


def load_p2_learning_dataset(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_id_split(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(json.load(handle))


def p2_learning_feature_vector(record: dict[str, Any]) -> list[float]:
    variant = str(record["variant_name"])
    margins = record.get("feasibility_margins", {})
    return [
        1.0 if variant == "chain_grasp" else 0.0,
        1.0 if variant == "symmetric_two_anchor_grasp" else 0.0,
        1.0 if variant == "tri_anchor_support_grasp" else 0.0,
        1.0 if variant == "central_base_plus_two_grasp_arms" else 0.0,
        1.0 if record.get("candidate_source") == "closed_loop_invalid_probe" else 0.0,
        float(record["candidate_id"]) / 8.0,
        float(record.get("required_slot_coverage", 0.0)),
        float(record.get("anchor_coverage", 0.0)),
        float(record.get("capability_coverage", 0.0)),
        _scaled(float(record.get("thrust_margin", 0.0)), 5.0),
        _scaled(float(record.get("payload_margin", 0.0)), 10.0),
        float(record.get("reachability_margin", 0.0)),
        _scaled(float(record.get("module_count", 0.0)), 10.0),
        _scaled(float(record.get("dock_edge_count", 0.0)), 10.0),
        _scaled(float(len(record.get("robot_anchor_ids", []))), 10.0),
        _scaled(float(len(record.get("contact_slot_ids", []))), 10.0),
        _scaled(float(len(record.get("control_group_ids", []))), 10.0),
        _scaled(float(margins.get("port_conflict_count", 0.0)), 10.0),
        float(margins.get("closed_loop_rejected", 0.0)),
    ]


def _dataset_record(record: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    output = dict(record)
    output["record_id"] = f"{record['episode_id']}:candidate:{record['candidate_id']:02d}:{row_index:05d}"
    output["selected_label"] = 1 if record["selected"] else 0
    output["accepted_label"] = 1 if record["accepted"] else 0
    output["feasible_label"] = 1 if record["feasible"] else 0
    output["teacher_score"] = float(record["design_score"])
    output["violation_labels"] = dict(record["feasibility_proxy_labels"])
    output["feature_names"] = list(P2_LEARNING_FEATURE_NAMES)
    output["features"] = p2_learning_feature_vector(output)
    return output


def _train_val_split(records: list[dict[str, Any]], *, val_fraction: float) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    val_period = max(2, round(1.0 / val_fraction))
    train_ids: list[str] = []
    val_ids: list[str] = []
    buckets: dict[tuple[str, str, int, int, int], list[str]] = defaultdict(list)
    for record in records:
        key = (
            str(record["variant_name"]),
            str(record["candidate_source"]),
            int(record["selected_label"]),
            int(record["accepted_label"]),
            int(record["feasible_label"]),
        )
        buckets[key].append(str(record["record_id"]))
    for key in sorted(buckets):
        bucket_ids = sorted(buckets[key])
        if len(bucket_ids) == 1:
            train_ids.append(bucket_ids[0])
            continue
        bucket_train: list[str] = []
        bucket_val: list[str] = []
        for index, record_id in enumerate(bucket_ids):
            if index % val_period == 0:
                bucket_val.append(record_id)
            else:
                bucket_train.append(record_id)
        if not bucket_train:
            bucket_train.append(bucket_val.pop())
        train_ids.extend(bucket_train)
        val_ids.extend(bucket_val)
    if not val_ids and len(train_ids) > 1:
        val_ids.append(train_ids.pop())
    if not train_ids or not val_ids:
        raise ValueError("train/val split produced an empty split")
    return sorted(train_ids), sorted(val_ids)


def _summary(
    records: list[dict[str, Any]],
    train_ids: list[str],
    val_ids: list[str],
    *,
    config_path: str,
    sample_count: int,
) -> dict[str, Any]:
    variant_counts = Counter(record["variant_name"] for record in records)
    hard_violation_counts = Counter(code for record in records for code in record["hard_violation_codes"])
    return {
        "config_path": config_path,
        "sample_count": sample_count,
        "record_count": len(records),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "accepted_count": sum(1 for record in records if record["accepted_label"] == 1),
        "rejected_count": sum(1 for record in records if record["accepted_label"] == 0),
        "selected_count": sum(1 for record in records if record["selected_label"] == 1),
        "feature_names": P2_LEARNING_FEATURE_NAMES,
        "variant_counts": dict(sorted(variant_counts.items())),
        "hard_violation_counts": dict(sorted(hard_violation_counts.items())),
        "normal_variant_names": [variant.value for variant in GRASP_CARRY_VARIANT_ORDER],
        "source_of_truth": "P2DesignPolicy + deterministic FeasibilityChecker",
        "production_path": "learned bootstrap models are not used in production path",
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True))
            handle.write("\n")


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _scaled(value: float, scale: float) -> float:
    return value / max(scale, 1.0e-9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the P2.5 candidate learning dataset.")
    parser.add_argument("--config", default="configs/training/p2_design_grasp_carry.yaml")
    parser.add_argument("--output-dir", default="outputs/p2_5/datasets")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    manifest = build_p2_learning_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    print(f"dataset: {manifest.dataset_path}")
    print(f"summary: {manifest.summary_path}")
    print(f"train ids: {manifest.train_ids_path}")
    print(f"val ids: {manifest.val_ids_path}")
    print(
        "records: "
        f"{manifest.record_count}, train={manifest.train_count}, val={manifest.val_count}, "
        f"accepted={manifest.accepted_count}, rejected={manifest.rejected_count}, selected={manifest.selected_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
