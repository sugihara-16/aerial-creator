from __future__ import annotations

import csv
import json
from pathlib import Path

from amsrr.training.p2_candidate_trace_export import export_p2_candidate_traces


def test_p2_candidate_trace_export_writes_all_candidates_and_probe(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate_traces"

    manifest = export_p2_candidate_traces(output_dir=output_dir, sample_count=1, seed=0)

    jsonl_path = Path(manifest.jsonl_path)
    csv_path = Path(manifest.csv_path)
    assert jsonl_path.exists()
    assert csv_path.exists()
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert manifest.record_count == len(records) == 5
    assert manifest.accepted_count == 4
    assert manifest.rejected_count == 1
    assert manifest.selected_count == 1
    assert any(record["selected"] for record in records)
    assert any(record["accepted"] for record in records)
    rejected = [record for record in records if record["rejected"]]
    assert len(rejected) == 1
    assert rejected[0]["candidate_source"] == "closed_loop_invalid_probe"
    assert "F_CLOSED_LOOP_REJECT_V1" in rejected[0]["hard_violation_codes"]
    for record in records:
        assert record["design_scores"]
        assert "L_FEASIBLE" in record["feasibility_proxy_labels"]
        assert "required_slot_coverage_ratio" in record["feasibility_margins"]
        assert "required_slot_anchor_capability_coverage_ratio" in record["feasibility_margins"]
        assert isinstance(record["robot_anchor_ids"], list)
        assert isinstance(record["contact_slot_ids"], list)
        assert isinstance(record["control_group_ids"], list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records)
    assert {row["candidate_source"] for row in rows} >= {"policy_variant", "closed_loop_invalid_probe"}

