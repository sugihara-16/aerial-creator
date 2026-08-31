#!/usr/bin/env python3
from __future__ import annotations

"""Finalize the stopped-on-first-failure R1 train-side range selection."""

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts.order9_run_r1_range_selection_v10 import (  # noqa: E402
    APPROVAL,
    PARENT_RESULT,
    PROTOCOL,
    _load_contract,
)

CORNER_RESULT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_v10/"
    "corner_probe/r1_l2_20mm_10deg/result.json"
)
ARTIFACT_RESULT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_v10/"
    "range_selection_result.json"
)
RESULT = REPOSITORY / "for_codex/R1_RANGE_SELECTION_V10_RESULT.md"
LEDGER = REPOSITORY / "for_codex/R1_RANGE_SELECTION_V10_RESULT_LEDGER.json"

FINALIZER_VERSION = "order9_r1_range_selection_finalizer_v10"
RESULT_VERSION = "order9_r1_range_selection_result_v10"
LEDGER_VERSION = "order9_r1_range_selection_result_ledger_v10"
FAILED_LEVEL_ID = "r1_l2_20mm_10deg"
SELECTED_LEVEL_ID = "r1_l1_10mm_5deg"
DECISIVE_CANDIDATE_ID = "r1_l2_20mm_10deg__train__train-000000-5fecfad4f44f__lattice_02"


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": source.relative_to(REPOSITORY).as_posix(),
        "sha256": hash_file(source),
    }


def _validate_binding(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 range binding is missing: {label}")
    path = REPOSITORY / str(value.get("path", ""))
    if not path.is_file() or hash_file(path) != value.get("sha256"):
        raise SchemaValidationError(f"R1 range binding changed: {label}")


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_validated_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, _approval = _load_contract()
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    if (
        parent.get("decision") != "accepted_minimum_level"
        or parent.get("level_id") != SELECTED_LEVEL_ID
        or int(parent.get("candidate_count", -1)) != 682
        or int(parent.get("episode_count", -1)) != 1364
        or int(parent.get("success_count", -1)) != 1364
        or int(parent.get("safety_failure_count", -1)) != 0
        or int(parent.get("fallback_count", -1)) != 0
        or parent.get("pi_l_actor_command_applied") is not False
        or parent.get("qpid_qp_applied") is not True
        or parent.get("local_servo_applied") is not True
    ):
        raise SchemaValidationError("accepted R1 minimum-level result differs")

    corner = json.loads(CORNER_RESULT.read_text(encoding="utf-8"))
    entries = corner.get("entries") if isinstance(corner, dict) else None
    if (
        corner.get("result_version") != "order9_r1_range_corner_probe_result_v10"
        or corner.get("level_id") != FAILED_LEVEL_ID
        or corner.get("status") != "failed"
        or corner.get("lightweight_passed") is not True
        or corner.get("control_passed") is not False
        or int(corner.get("candidate_count", -1)) != 176
        or corner.get("pi_l_actor_command_applied") is not False
        or corner.get("qpid_qp_applied") is not True
        or corner.get("local_servo_applied") is not True
        or not isinstance(entries, list)
        or len(entries) != 176
    ):
        raise SchemaValidationError("R1 level-two corner result differs")
    for label, binding in corner.get("bindings", {}).items():
        _validate_binding(binding, label=f"corner:{label}")
    decisive = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("candidate_id") == DECISIVE_CANDIDATE_ID
    ]
    if len(decisive) != 1:
        raise SchemaValidationError("R1 decisive candidate identity differs")
    failure = decisive[0]
    if (
        failure.get("valid_evidence") is not True
        or failure.get("evidence_group") != "single_confirmation"
        or failure.get("sample_kind") != "lattice"
        or failure.get("candidate_passed") is not False
        or int(failure.get("episode_count", -1)) != 2
        or int(failure.get("success_count", -1)) != 0
        or int(failure.get("safety_failure_count", -1)) != 2
        or int(failure.get("fallback_count", -1)) != 0
    ):
        raise SchemaValidationError("R1 decisive single confirmation differs")
    _validate_binding(failure.get("raw_rollout"), label="decisive:raw")
    _validate_binding(failure.get("episodes"), label="decisive:episodes")
    return protocol, parent, {"corner": corner, "failure": failure}


def _summary(protocol: Mapping[str, Any], failure: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "result_version": RESULT_VERSION,
        "decision": "accepted_largest_consecutive_passing_level",
        "selected_level_id": SELECTED_LEVEL_ID,
        "selected_position_half_span_m": 0.010,
        "selected_yaw_half_span_rad": 0.08726646259971647,
        "first_failed_level_id": FAILED_LEVEL_ID,
        "stop_reason": "confirmed_single_lattice_safety_failure",
        "not_attempted_level_ids": ["r1_l3_30mm_15deg", "r1_l4_40mm_20deg"],
        "level_two_lightweight_corner_candidate_count": 176,
        "level_two_lightweight_corner_pass_count": 176,
        "decisive_failure": {
            "candidate_id": failure["candidate_id"],
            "source_bucket_id": failure["source_bucket_id"],
            "module_count": failure["module_count"],
            "sample_kind": failure["sample_kind"],
            "sample_index": failure["sample_index"],
            "episode_count": failure["episode_count"],
            "success_count": failure["success_count"],
            "safety_failure_count": failure["safety_failure_count"],
            "fallback_count": failure["fallback_count"],
            "evidence_group": failure["evidence_group"],
            "raw_rollout": failure["raw_rollout"],
            "episodes": failure["episodes"],
        },
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "calibration_evidence_training_eligible": False,
        "held_out_confirmation_in_scope": False,
        "formal_teacher_collection_authorized": False,
        "learning_started": False,
        "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
    }


def main() -> int:
    protocol, _parent, evidence = _read_validated_inputs()
    summary = _summary(protocol, evidence["failure"])
    _write(ARTIFACT_RESULT, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    markdown = f"""# R1 train-side range selection v10 result

- 採用範囲: 位置 ±10 mm、向き ±5度（`{SELECTED_LEVEL_ID}`）
- 第2段階: 位置 ±20 mm、向き ±10度は不合格
- 一次判定: 第2段階の176隅点すべて合格（Isaac・制御器なし）
- 確定不合格: `{DECISIVE_CANDIDATE_ID}` を単独で2回実行し、成功0、安全不合格2、代替処理0
- 停止規則: 最初の不合格段階で停止したため、第3・第4段階は未実行
- 制御契約: π_Lの学習器出力は0、QPID/QPと局所サーボは有効
- 範囲選定用証跡は学習データではなく、未使用14機体の確認・正式教師軌道収集・模倣学習はいずれも未開始
"""
    _write(RESULT, markdown)

    ledger = {
        **summary,
        "ledger_version": LEDGER_VERSION,
        "finalizer_version": FINALIZER_VERSION,
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "approval": _binding(APPROVAL),
            "parent_minimum_level_result": _binding(PARENT_RESULT),
            "level_two_corner_result": _binding(CORNER_RESULT),
            "artifact_result": _binding(ARTIFACT_RESULT),
            "result_document": _binding(RESULT),
            "finalizer_implementation": _binding(Path(__file__)),
            "protected_c3_rollout": protocol["protected_c3_rollout"],
            "protected_c3_curriculum": protocol["protected_c3_curriculum"],
            "protected_c3_release_ledger": protocol["protected_c3_release_ledger"],
        },
    }
    _write(LEDGER, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": ledger["decision"],
                "selected_level_id": SELECTED_LEVEL_ID,
                "ledger": _binding(LEDGER),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
