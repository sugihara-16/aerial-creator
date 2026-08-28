#!/usr/bin/env python3
from __future__ import annotations

"""Finalize the R1 minimum-level v8 overlay from hash-bound evidence."""

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v7 import (  # noqa: E402
    load_order9_r1_nominal_calibration_v7_contract,
    validate_order9_r1_nominal_v7_isaac_result,
)
from amsrr.training.order9_r1_nominal_calibration_v8 import (  # noqa: E402
    load_order9_r1_nominal_calibration_v8_contract,
    validate_order9_r1_nominal_v8_repair_evidence,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts.order9_run_r1_batched_nominal_calibration_v7 import (  # noqa: E402
    _candidate_manifest,
    _select_source_bucket_manifests,
)

RESULT_VERSION = "order9_r1_nominal_calibration_v8_result_v1"
LEDGER_VERSION = "order9_r1_nominal_calibration_v8_result_ledger_v1"
RESULT_RELATIVE = Path("for_codex/R1_NOMINAL_CALIBRATION_V8_RESULT.md")
LEDGER_RELATIVE = Path("for_codex/R1_NOMINAL_CALIBRATION_V8_RESULT_LEDGER.json")


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source.relative_to(REPOSITORY)), "sha256": hash_file(source)}


def _validate_unchanged_cases(
    *, protocol_v8: dict[str, Any], excluded: frozenset[str]
) -> tuple[dict[str, Any], ...]:
    protocol_v7, _approval_v7 = load_order9_r1_nominal_calibration_v7_contract(
        REPOSITORY
    )
    root = REPOSITORY / str(protocol_v7["output_root"])
    source_manifests = _select_source_bucket_manifests(
        root / "persistent_process_batches"
    )
    wrapper_sha = protocol_v8["batched_wrapper_implementation"]["sha256"]
    runtime_sha = protocol_v8["batched_runtime_implementation"]["sha256"]
    entries = []
    all_source_ids = set()
    for source_id, source_manifest in sorted(source_manifests.items()):
        all_source_ids.add(source_id)
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        for job in payload["jobs"]:
            candidate_id = job.get("name")
            if candidate_id in excluded:
                continue
            manifest_path = _candidate_manifest(job)
            case = load_order9_r1_isaac_case(manifest_path, REPOSITORY)
            results = validate_order9_r1_nominal_v7_isaac_result(
                case, repository_root=REPOSITORY, protocol=protocol_v7
            )
            raw_path = manifest_path.parent / "isaac/evaluation_rollout.pt"
            episodes_path = manifest_path.parent / "isaac/evaluation_episodes.jsonl"
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            metadata = raw.get("metadata") if isinstance(raw, dict) else None
            if (
                candidate_id != case.manifest.candidate_id
                or case.manifest.source_bucket_id != source_id
                or len(results) != 2
                or not all(result.success for result in results)
                or any(result.safety_failure for result in results)
                or any(result.fallback_used for result in results)
                or not isinstance(metadata, dict)
                or metadata.get("r1_batched_nominal_wrapper_sha256") != wrapper_sha
                or metadata.get("r1_batched_nominal_runtime_sha256") != runtime_sha
                or metadata.get("pi_l_actor_command_applied") is not False
                or not 1 <= int(metadata.get("r1_batched_candidate_count", -1)) <= 31
            ):
                raise SchemaValidationError(
                    f"R1 nominal v8 unchanged evidence failed: {candidate_id}"
                )
            entries.append(
                {
                    "candidate_id": candidate_id,
                    "source_bucket_id": source_id,
                    "trajectory_kind": "unchanged_v7",
                    "episode_count": 2,
                    "success_count": 2,
                    "safety_failure_count": 0,
                    "fallback_count": 0,
                    "pi_l_actor_command_applied": False,
                    "case_manifest": _binding(manifest_path),
                    "raw_rollout": _binding(raw_path),
                    "episodes": _binding(episodes_path),
                }
            )
    if len(all_source_ids) != 22 or len(entries) != 673:
        raise SchemaValidationError(
            f"R1 nominal v8 unchanged candidate count differs: {len(entries)}"
        )
    return tuple(entries)


def _repair_entries(
    protocol_v8: dict[str, Any], repairs: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    ledger_path = REPOSITORY / str(protocol_v8["repair_evidence_ledger"]["path"])
    ledger_entries = {
        value["candidate_id"]: value
        for value in json.loads(ledger_path.read_text(encoding="utf-8"))["entries"]
    }
    output = []
    for repair in repairs:
        bound = ledger_entries[repair["candidate_id"]]
        output.append(
            {
                **repair,
                "source_bucket_id": "train-000002-0e86a4ff3e71",
                "trajectory_kind": "yaw_branch_repair_v1",
                "case_manifest": bound["case_manifest"],
                "lightweight_admission": bound["lightweight_admission"],
                "raw_rollout": bound["raw_rollout"],
                "episodes": bound["episodes"],
                "selected_execution": bound["selected_execution"],
            }
        )
    return tuple(output)


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    protocol_v8, _approval_v8 = load_order9_r1_nominal_calibration_v8_contract(
        REPOSITORY
    )
    exclusion_path = REPOSITORY / str(
        protocol_v8["repaired_candidate_exclusion"]["path"]
    )
    excluded = frozenset(
        json.loads(exclusion_path.read_text(encoding="utf-8"))["candidate_ids"]
    )
    repairs = validate_order9_r1_nominal_v8_repair_evidence(
        REPOSITORY, protocol=protocol_v8
    )
    unchanged = _validate_unchanged_cases(protocol_v8=protocol_v8, excluded=excluded)
    repaired_entries = _repair_entries(protocol_v8, repairs)
    entries = sorted(
        (*unchanged, *repaired_entries), key=lambda value: value["candidate_id"]
    )
    candidate_ids = [entry["candidate_id"] for entry in entries]
    if (
        len(entries) != 682
        or len(candidate_ids) != len(set(candidate_ids))
        or set(candidate_ids).intersection(excluded) != excluded
        or sum(entry["episode_count"] for entry in entries) != 1364
        or sum(entry["success_count"] for entry in entries) != 1364
        or sum(entry["safety_failure_count"] for entry in entries) != 0
        or sum(entry["fallback_count"] for entry in entries) != 0
        or any(entry["pi_l_actor_command_applied"] for entry in entries)
    ):
        raise SchemaValidationError("R1 nominal v8 aggregate gate failed")

    protocol_path = (
        REPOSITORY / "configs/training/order9_r1_nominal_calibration_protocol_v8.yaml"
    )
    approval_path = (
        REPOSITORY / "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V8_APPROVAL.json"
    )
    subset_result = (
        REPOSITORY
        / "artifacts/p4_full/order9/r1_teacher/calibration_v9_nominal_compression_v7"
        / "selection/r1_l1_10mm_5deg/batched_run_result_v7_subset_for_v8.json"
    )
    ledger_path = REPOSITORY / LEDGER_RELATIVE
    ledger = {
        "ledger_version": LEDGER_VERSION,
        "result_version": RESULT_VERSION,
        "status": "accepted",
        "level_id": "r1_l1_10mm_5deg",
        "source_bucket_count": 22,
        "candidate_count": 682,
        "episode_count": 1364,
        "success_count": 1364,
        "safety_failure_count": 0,
        "fallback_count": 0,
        "unchanged_v7_candidate_count": 673,
        "yaw_repaired_candidate_count": 9,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "minimum_level_result_authorizes_collection": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v8_protocol": _binding(protocol_path),
            "v8_approval": _binding(approval_path),
            "v7_subset_result": _binding(subset_result),
            "repair_evidence_ledger": _binding(
                REPOSITORY / str(protocol_v8["repair_evidence_ledger"]["path"])
            ),
            "finalizer_implementation": _binding(Path(__file__)),
        },
        "entries": entries,
    }
    write_order9_r1_clearance_diagnostic(ledger, ledger_path)
    result_path = REPOSITORY / RESULT_RELATIVE
    _write_text_new(
        result_path,
        "# R1 最小条件・第8版 結果\n\n"
        "- 判定: 合格\n"
        "- 対象: 学習側22機体、各31軌道\n"
        "- 実行: 682軌道 x 2回 = 1364回\n"
        "- 成功: 1364 / 1364\n"
        "- 安全違反: 0\n"
        "- 代替制御: 0\n"
        "- 元の第7版軌道: 673本\n"
        "- 軽量判定後に差し替えた軌道: 9本\n"
        "- π_L: システムには保持、出力は全実行で0\n"
        "- 実行制御: QPID/QP/local servoを使用\n"
        "- 注意: この最小条件の合格だけでは正式教師収集を許可しない\n\n"
        f"結果台帳: `{LEDGER_RELATIVE}`\n",
    )
    print(
        "ORDER9_R1_NOMINAL_V8_COMPLETE="
        + json.dumps(
            {
                "candidate_count": 682,
                "episode_count": 1364,
                "success_count": 1364,
                "ledger_path": str(ledger_path),
                "ledger_sha256": hash_file(ledger_path),
                "result_path": str(result_path),
                "result_sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
