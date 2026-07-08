from __future__ import annotations

import json
from pathlib import Path

from amsrr.training.p2_learning_dataset import (
    P2_LEARNING_FEATURE_NAMES,
    build_p2_learning_dataset,
    load_id_split,
    load_p2_learning_dataset,
)


def test_p2_learning_dataset_builds_records_and_split(tmp_path: Path) -> None:
    output_dir = tmp_path / "datasets"

    manifest = build_p2_learning_dataset(output_dir=output_dir, sample_count=8, seed=0)

    assert manifest.record_count == 40
    assert manifest.train_count > 0
    assert manifest.val_count > 0
    assert manifest.accepted_count == 32
    assert manifest.rejected_count == 8
    assert manifest.selected_count == 8
    records = load_p2_learning_dataset(manifest.dataset_path)
    train_ids = load_id_split(manifest.train_ids_path)
    val_ids = load_id_split(manifest.val_ids_path)
    train_records = [record for record in records if record["record_id"] in set(train_ids)]
    val_records = [record for record in records if record["record_id"] in set(val_ids)]
    assert len(records) == manifest.record_count
    assert set(train_ids).isdisjoint(val_ids)
    assert any(record["accepted_label"] == 0 for record in train_records)
    assert any(record["accepted_label"] == 0 for record in val_records)
    assert any(record["selected_label"] == 1 for record in train_records)
    assert any(record["selected_label"] == 1 for record in val_records)
    first = records[0]
    assert first["record_id"]
    assert first["selected_label"] in (0, 1)
    assert first["accepted_label"] in (0, 1)
    assert first["feasible_label"] in (0, 1)
    assert first["teacher_score"] == first["design_score"]
    assert first["feature_names"] == P2_LEARNING_FEATURE_NAMES
    assert len(first["features"]) == len(P2_LEARNING_FEATURE_NAMES)
    with Path(manifest.summary_path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["record_count"] == manifest.record_count
    assert summary["production_path"] == "learned bootstrap models are not used in production path"
