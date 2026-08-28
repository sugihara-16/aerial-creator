from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
    Order9R1IsaacCaseManifest,
)
from amsrr.training.order9_r1_nominal_calibration_v7 import (
    ORDER9_R1_NOMINAL_COMPRESSION_OPTION,
    coalesce_order9_r1_v7_same_morphology_collectors,
    load_order9_r1_nominal_calibration_v7_contract,
    order9_r1_v7_additional_compression_mm,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def test_v7_contract_and_bucket_local_compression_are_hash_bound() -> None:
    protocol, approval = load_order9_r1_nominal_calibration_v7_contract(REPOSITORY)
    assert approval["decision"] == "approved"
    repaired = _materialized("train-000004-190f3a425b3e")
    ordinary = _materialized("train-000003-3b95da871f01")
    assert order9_r1_v7_additional_compression_mm(repaired, protocol) == 5.0
    assert order9_r1_v7_additional_compression_mm(ordinary, protocol) == 0.0


def test_v7_coalescer_strips_wrapper_option_from_persistent_jobs(
    tmp_path: Path,
) -> None:
    commands = {
        "case_a": _command("case_a", "5.0,5.0"),
        "case_b": _command("case_b", "5.0,5.0"),
    }
    combined, logs, members = coalesce_order9_r1_v7_same_morphology_collectors(
        commands,
        log_paths={
            "case_a": tmp_path / "case_a.log",
            "case_b": tmp_path / "case_b.log",
        },
        morphology_hash_by_name={"case_a": "abc123", "case_b": "abc123"},
        batch_root=tmp_path / "batches",
    )
    assert tuple(combined) == ("morphology_abc123_chunk_00",)
    batch_command = combined["morphology_abc123_chunk_00"]
    assert batch_command.count(ORDER9_R1_NOMINAL_COMPRESSION_OPTION) == 1
    manifest = Path(batch_command[-1]).read_text(encoding="utf-8")
    assert ORDER9_R1_NOMINAL_COMPRESSION_OPTION not in manifest
    assert members["morphology_abc123_chunk_00"] == ("case_a", "case_b")
    assert logs["morphology_abc123_chunk_00"].name == "persistent_process.log"


def test_v7_coalescer_rejects_mixed_repairs_within_one_morphology(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="differ in executable or repair"):
        coalesce_order9_r1_v7_same_morphology_collectors(
            {
                "case_a": _command("case_a", "5.0,5.0"),
                "case_b": _command("case_b", "0.0,0.0"),
            },
            log_paths={
                "case_a": tmp_path / "case_a.log",
                "case_b": tmp_path / "case_b.log",
            },
            morphology_hash_by_name={"case_a": "abc123", "case_b": "abc123"},
            batch_root=tmp_path / "batches",
        )


def _command(candidate_id: str, sweep: str) -> list[str]:
    return [
        "/python",
        "/repo/scripts/order9_r1_nominal_compression_sweep_rollout.py",
        "--generation-id",
        candidate_id,
        "--output-raw",
        f"/{candidate_id}.pt",
        "--evaluation-jsonl",
        f"/{candidate_id}.jsonl",
        ORDER9_R1_NOMINAL_COMPRESSION_OPTION,
        sweep,
    ]


def _materialized(source_bucket_id: str) -> MaterializedOrder9R1IsaacCase:
    manifest = object.__new__(Order9R1IsaacCaseManifest)
    manifest.source_bucket_id = source_bucket_id
    return MaterializedOrder9R1IsaacCase(Path("case_manifest.json"), manifest)
