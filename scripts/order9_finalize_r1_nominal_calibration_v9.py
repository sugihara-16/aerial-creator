#!/usr/bin/env python3
from __future__ import annotations

"""Finalize the bounded-four R1 minimum-level evidence as a hash-bound result."""

from collections import Counter
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
from amsrr.training.order9_r1_nominal_calibration_v8 import (  # noqa: E402
    validate_order9_r1_nominal_v8_repair_evidence,
)
from amsrr.utils.config import load_config  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts.order9_run_r1_yaw_repair_bounded_validation_v9 import (  # noqa: E402
    RESULT_LEDGER as YAW_BOUNDED_RESULT_LEDGER,
    validate_result_ledger as validate_yaw_bounded_result_ledger,
)

PROTOCOL = Path("configs/training/order9_r1_nominal_calibration_protocol_v9.yaml")
APPROVAL = Path("for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V9_APPROVAL.json")
RUNNER_LEDGER = Path(
    "artifacts/p4_full/order9/r1_teacher/validation_v10_bounded_four/"
    "r1_l1_10mm_5deg/result_ledger.json"
)
V8_PROTOCOL = Path("configs/training/order9_r1_nominal_calibration_protocol_v8.yaml")
RESULT = Path("for_codex/R1_NOMINAL_CALIBRATION_V9_RESULT.md")
LEDGER = Path("for_codex/R1_NOMINAL_CALIBRATION_V9_RESULT_LEDGER.json")

PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v9"
APPROVAL_VERSION = "order9_r1_nominal_calibration_approval_record_v9"
RESULT_VERSION = "order9_r1_nominal_calibration_v9_result_v1"
LEDGER_VERSION = "order9_r1_nominal_calibration_v9_result_ledger_v1"
RUNNER_VERSION = (
    "order9_r1_bounded_four_minimum_level_runner_v3_existing_evidence_first"
)

EXPECTED_SOURCE_COUNT = 22
EXPECTED_SOURCE_CANDIDATES = 31
EXPECTED_RUNNER_CANDIDATES = 673
EXPECTED_REPAIRED_CANDIDATES = 9
EXPECTED_TOTAL_CANDIDATES = 682
EXPECTED_TOTAL_EPISODES = 1364

_REQUIRED_PROTOCOL_BINDINGS = (
    "protected_c3_checkpoint",
    "protected_c3_rollout",
    "protected_c3_curriculum",
    "protected_c3_release_ledger",
    "batched_wrapper_implementation",
    "batched_runtime_implementation",
    "bounded_four_runner_implementation",
    "release_clearance_repair_implementation",
    "train003_repair_manifest",
    "repaired_candidate_exclusion",
    "yaw_repair_evidence_ledger",
    "yaw_repair_evidence_validator_implementation",
    "yaw_bounded_revalidation_runner_implementation",
    "yaw_bounded_revalidation_result_ledger",
    "finalizer_implementation",
)


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source.relative_to(REPOSITORY)),
        "sha256": hash_file(source),
    }


def _validate_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 v9 binding missing: {label}")
    path = REPOSITORY / str(value.get("path", ""))
    if not path.is_file() or hash_file(path) != value.get("sha256"):
        raise SchemaValidationError(f"R1 v9 binding changed: {label}")
    return path


def _write_exact(path: Path, text: str) -> Path:
    destination = (REPOSITORY / path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"R1 v9 result already differs: {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = REPOSITORY / PROTOCOL
    approval_path = REPOSITORY / APPROVAL
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "approved"
        or protocol.get("level_id") != "r1_l1_10mm_5deg"
        or int(protocol.get("source_bucket_count", -1)) != EXPECTED_SOURCE_COUNT
        or int(protocol.get("candidate_count", -1)) != EXPECTED_TOTAL_CANDIDATES
        or int(protocol.get("replay_count_per_candidate", -1)) != 2
        or int(protocol.get("maximum_candidates_per_isaac_scene", -1)) != 4
        or protocol.get("restart_reuses_validated_retry_evidence_before_isaac")
        is not True
        or abs(float(protocol.get("isaac_environment_spacing_m", -1.0)) - 3.0) > 1.0e-12
        or abs(float(protocol.get("train003_grasp_compression_m", -1.0)) - 0.005)
        > 1.0e-12
        or abs(float(protocol.get("train003_release_height_offset_m", -1.0)) - 0.03)
        > 1.0e-12
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_applied") is not True
        or protocol.get("local_servo_applied") is not True
        or protocol.get("minimum_level_result_authorizes_collection") is not False
    ):
        raise SchemaValidationError("R1 nominal v9 protocol contract differs")
    if (
        approval.get("record_version") != APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path") != str(PROTOCOL)
        or approval.get("approved_protocol", {}).get("sha256")
        != hash_file(protocol_path)
    ):
        raise SchemaValidationError("R1 nominal v9 approval record differs")
    for key in _REQUIRED_PROTOCOL_BINDINGS:
        _validate_binding(protocol.get(key), label=key)
    return protocol, approval


def _validate_runner_ledger(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    path = REPOSITORY / RUNNER_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if (
        payload.get("runner_version") != RUNNER_VERSION
        or payload.get("status") != "passed"
        or int(payload.get("source_bucket_count", -1)) != EXPECTED_SOURCE_COUNT
        or int(payload.get("candidate_count", -1)) != EXPECTED_RUNNER_CANDIDATES
        or int(payload.get("episode_count", -1)) != 2 * EXPECTED_RUNNER_CANDIDATES
        or int(payload.get("success_count", -1)) != 2 * EXPECTED_RUNNER_CANDIDATES
        or int(payload.get("safety_failure_count", -1)) != 0
        or int(payload.get("fallback_count", -1)) != 0
        or payload.get("pi_l_actor_command_applied") is not False
        or payload.get("qpid_qp_local_servo_applied") is not True
        or abs(float(payload.get("environment_spacing_m", -1.0)) - 3.0) > 1.0e-12
        or payload.get("wrapper_sha256")
        != protocol["batched_wrapper_implementation"]["sha256"]
        or payload.get("runtime_sha256")
        != protocol["batched_runtime_implementation"]["sha256"]
        or int(payload.get("repaired_candidate_count_validated_separately", -1))
        != EXPECTED_REPAIRED_CANDIDATES
        or int(payload.get("legacy_v7_candidate_count", -1))
        != int(protocol.get("expected_legacy_v7_evidence_reuse_count", -2))
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_RUNNER_CANDIDATES
    ):
        raise SchemaValidationError("R1 nominal v9 runner result differs")

    candidate_ids: list[str] = []
    validated: list[dict[str, Any]] = []
    for entry in entries:
        candidate_id = entry.get("candidate_id") if isinstance(entry, dict) else None
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or int(entry.get("episode_count", -1)) != 2
            or int(entry.get("success_count", -1)) != 2
            or int(entry.get("safety_failure_count", -1)) != 0
            or int(entry.get("fallback_count", -1)) != 0
        ):
            raise SchemaValidationError("R1 nominal v9 runner entry differs")
        _validate_binding(entry.get("raw_rollout"), label=f"{candidate_id}:raw")
        _validate_binding(entry.get("episodes"), label=f"{candidate_id}:episodes")
        if entry.get("legacy_v7_evidence_reused"):
            _validate_binding(
                entry.get("case_manifest"), label=f"{candidate_id}:case_manifest"
            )
        if (
            sum(
                bool(entry.get(key))
                for key in (
                    "legacy_v7_evidence_reused",
                    "bounded_four_candidate_confirmation",
                    "single_candidate_confirmation",
                )
            )
            != 1
        ):
            raise SchemaValidationError(
                f"R1 nominal v9 evidence route is ambiguous: {candidate_id}"
            )
        candidate_ids.append(candidate_id)
        validated.append({**entry, "evidence_group": "bounded_four_main_set"})
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SchemaValidationError("R1 nominal v9 runner candidate IDs repeat")
    return payload, tuple(validated)


def _validate_repaired_entries(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    v8_protocol = load_config(REPOSITORY / V8_PROTOCOL)
    exclusion_path = _validate_binding(
        protocol.get("repaired_candidate_exclusion"), label="yaw_exclusion"
    )
    evidence_path = _validate_binding(
        protocol.get("yaw_repair_evidence_ledger"), label="yaw_repair_ledger"
    )
    if v8_protocol.get("repaired_candidate_exclusion", {}).get("sha256") != hash_file(
        exclusion_path
    ) or v8_protocol.get("repair_evidence_ledger", {}).get("sha256") != hash_file(
        evidence_path
    ):
        raise SchemaValidationError("R1 nominal v9 yaw repair parent binding differs")
    results = validate_order9_r1_nominal_v8_repair_evidence(
        REPOSITORY, protocol=v8_protocol
    )
    bindings = {
        entry["candidate_id"]: entry
        for entry in json.loads(evidence_path.read_text(encoding="utf-8"))["entries"]
    }
    output = []
    for result in results:
        candidate_id = result["candidate_id"]
        output.append(
            {
                **result,
                "source_bucket_id": "train-000002-0e86a4ff3e71",
                "evidence_group": "yaw_branch_repair",
                **{
                    key: value
                    for key, value in bindings[candidate_id].items()
                    if key != "candidate_id"
                },
            }
        )
    if len(output) != EXPECTED_REPAIRED_CANDIDATES:
        raise SchemaValidationError("R1 nominal v9 yaw repair count differs")
    expected_by_id = {entry["candidate_id"]: entry for entry in output}
    bounded_path = _validate_binding(
        protocol.get("yaw_bounded_revalidation_result_ledger"),
        label="yaw_bounded_revalidation_result_ledger",
    )
    if bounded_path.resolve() != YAW_BOUNDED_RESULT_LEDGER.resolve():
        raise SchemaValidationError("R1 nominal v9 yaw bounded result path differs")
    bounded = validate_yaw_bounded_result_ledger(bounded_path)
    if {entry["candidate_id"] for entry in bounded} != set(expected_by_id):
        raise SchemaValidationError("R1 nominal v9 bounded yaw candidate set differs")
    return tuple(
        {
            **entry,
            "minimum_collision_clearance_m": expected_by_id[entry["candidate_id"]][
                "minimum_collision_clearance_m"
            ],
            "minimum_normalized_joint_limit_reserve": expected_by_id[
                entry["candidate_id"]
            ]["minimum_normalized_joint_limit_reserve"],
            "evidence_group": "yaw_branch_repair_bounded_four",
        }
        for entry in bounded
    )


def main() -> int:
    protocol, _approval = _validate_protocol()
    runner, main_entries = _validate_runner_ledger(protocol)
    repaired_entries = _validate_repaired_entries(protocol)
    entries = sorted(
        (*main_entries, *repaired_entries), key=lambda value: value["candidate_id"]
    )
    candidate_ids = [entry["candidate_id"] for entry in entries]
    source_counts = Counter(entry["source_bucket_id"] for entry in entries)
    if (
        len(entries) != EXPECTED_TOTAL_CANDIDATES
        or len(candidate_ids) != len(set(candidate_ids))
        or len(source_counts) != EXPECTED_SOURCE_COUNT
        or set(source_counts.values()) != {EXPECTED_SOURCE_CANDIDATES}
        or sum(entry["episode_count"] for entry in entries) != EXPECTED_TOTAL_EPISODES
        or sum(entry["success_count"] for entry in entries) != EXPECTED_TOTAL_EPISODES
        or sum(entry["safety_failure_count"] for entry in entries) != 0
        or sum(entry["fallback_count"] for entry in entries) != 0
        or any(entry.get("pi_l_actor_command_applied") for entry in entries)
    ):
        raise SchemaValidationError("R1 nominal v9 aggregate gate failed")

    ledger = {
        "ledger_version": LEDGER_VERSION,
        "result_version": RESULT_VERSION,
        "decision": "accepted_minimum_level",
        "level_id": "r1_l1_10mm_5deg",
        "source_bucket_count": EXPECTED_SOURCE_COUNT,
        "candidate_count": EXPECTED_TOTAL_CANDIDATES,
        "episode_count": EXPECTED_TOTAL_EPISODES,
        "success_count": EXPECTED_TOTAL_EPISODES,
        "safety_failure_count": 0,
        "fallback_count": 0,
        "bounded_four_main_candidate_count": EXPECTED_RUNNER_CANDIDATES,
        "yaw_repaired_candidate_count": EXPECTED_REPAIRED_CANDIDATES,
        "legacy_v7_evidence_reused_candidate_count": int(
            runner["legacy_v7_candidate_count"]
        ),
        "maximum_candidates_per_isaac_scene": 4,
        "isaac_environment_spacing_m": 3.0,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "minimum_level_result_authorizes_collection": False,
        "formal_teacher_collection_authorized": False,
        "training_eligible": False,
        "invalid_large_scene_evidence_accepted": False,
        "bindings": {
            "protocol": _binding(REPOSITORY / PROTOCOL),
            "approval": _binding(REPOSITORY / APPROVAL),
            "runner_result_ledger": _binding(REPOSITORY / RUNNER_LEDGER),
            "runner_implementation": protocol["bounded_four_runner_implementation"],
            "yaw_repair_evidence_ledger": protocol["yaw_repair_evidence_ledger"],
            "yaw_repair_evidence_validator_implementation": protocol[
                "yaw_repair_evidence_validator_implementation"
            ],
            "yaw_bounded_revalidation_runner_implementation": protocol[
                "yaw_bounded_revalidation_runner_implementation"
            ],
            "yaw_bounded_revalidation_result_ledger": protocol[
                "yaw_bounded_revalidation_result_ledger"
            ],
            "finalizer_implementation": _binding(Path(__file__)),
        },
        "source_candidate_counts": dict(sorted(source_counts.items())),
        "entries": entries,
    }
    ledger_text = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    ledger_path = _write_exact(LEDGER, ledger_text)
    result_text = (
        "# R1 最小条件・第9版 結果\n\n"
        "- 判定: 合格\n"
        "- 対象: 学習側22機体、各31軌道\n"
        "- 実行: 682軌道を各2回、合計1364回\n"
        "- 成功: 1364 / 1364\n"
        "- 安全違反: 0\n"
        "- 代替制御: 0\n"
        "- π_L: システムには保持し、出力は全実行で0\n"
        "- 実行制御: QPID、QP、局所サーボを使用\n"
        "- 同一Isaac場面の候補数: 最大4、機体間隔3 m\n"
        "- 大規模な同時配置で得た不安定な結果: 正式証拠には不採用\n"
        "- 注意: この結果だけでは教師軌道の正式収集を許可しない\n\n"
        f"結果台帳: `{LEDGER}`\n"
    )
    result_path = _write_exact(RESULT, result_text)
    print(
        "ORDER9_R1_NOMINAL_V9_COMPLETE="
        + json.dumps(
            {
                "candidate_count": EXPECTED_TOTAL_CANDIDATES,
                "episode_count": EXPECTED_TOTAL_EPISODES,
                "success_count": EXPECTED_TOTAL_EPISODES,
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
