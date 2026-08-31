#!/usr/bin/env python3
from __future__ import annotations

"""Preserve independently repeated early robot-object collisions."""

import argparse
import json
from pathlib import Path
import re
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_range_selection_v13 as runtime_v13


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--single-manifest", required=True)
    parser.add_argument("--single-root", required=True)
    parser.add_argument("--expected-body-name", required=True)
    parser.add_argument("--expected-body-index", type=int, required=True)
    parser.add_argument("--expected-phase-index", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--expected-selected-anchor-body",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--replacement-ports", required=True)
    parser.add_argument("--replacement-group-id", required=True)
    parser.add_argument("--replacement-height-m", type=float, required=True)
    parser.add_argument("--replacement-pregrasp-clearance-m", type=float, default=0.08)
    parser.add_argument("--replacement-collision-margin-m", type=float, default=0.005)
    parser.add_argument("--replacement-tangent-world-m", default="0,0,0")
    return parser


def _numbers(value: str, *, count: int, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != count:
        raise SchemaValidationError(f"expected {count} comma-separated values")
    return result


def _diagnostics(path: Path, *, environment_indices: set[int]) -> list[dict]:
    prefix = "ORDER9_COLLISION_DIAGNOSTIC="
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        value = json.loads(line[len(prefix) :])
        if int(value.get("environment_index", -1)) in environment_indices:
            records.append(value)
    return records


def _require_collision(
    records: list[dict],
    *,
    label: str,
    body_name: str,
    body_index: int,
    phase_index: int,
    selected_anchor_body: bool,
) -> None:
    if len(records) != 2:
        raise SchemaValidationError(f"{label} collision count differs")
    for record in records:
        bodies = record.get("active_bodies")
        matches = [
            value
            for value in bodies or []
            if value.get("body_name") == body_name
            and int(value.get("body_index", -1)) == body_index
        ]
        body = matches[0] if len(matches) == 1 else None
        if (
            int(record.get("runtime_phase_index", -1)) != phase_index
            or record.get("prohibited_environment_contact") is not False
            or record.get("prohibited_object_contact") is not True
            or body is None
            or body.get("selected_anchor_body") is not selected_anchor_body
            or float(body.get("object_force_norm_n", 0.0)) <= 0.0
            or float(body.get("environment_force_norm_n", -1.0)) != 0.0
        ):
            raise SchemaValidationError(f"{label} collision diagnosis differs")


def _require_episodes(
    path: Path,
    *,
    candidate_id: str,
    label: str,
    phase_index: int,
) -> list[dict]:
    episodes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generation_id = f"r1_nominal_calibration:{candidate_id}"
    if len(episodes) != 2 or any(
        value.get("task_success") is not False
        or value.get("safety_failure") is not True
        or value.get("failure_reason") != "hard_collision"
        or int(value.get("fallback_decision_count", -1)) != 0
        or value.get("isaac_backed") is not True
        or (value.get("metadata") or {}).get("generation_id") != generation_id
        or float((value.get("metrics") or {}).get("hard_collision", 0.0)) != 1.0
        or int(float((value.get("metrics") or {}).get("terminal_phase_index", -1.0)))
        != phase_index
        for value in episodes
    ):
        raise SchemaValidationError(f"{label} collision episodes differ")
    return episodes


def _quarantine(candidate_id: str, evidence_root: Path, *, reason: str) -> Path:
    binding = runtime_v13.runtime_v10._quarantine_invalid_candidate_output(
        candidate_id=candidate_id,
        evidence_root=evidence_root,
        preparation_root=(
            runtime_v13.OUTPUT_ROOT / "prepared" / "r1_l2_20mm_10deg"
        ),
        failure_reason=reason,
    )
    if binding is None:
        raise SchemaValidationError("early object-collision evidence is absent")
    manifest = json.loads(
        (REPOSITORY / binding["path"]).read_text(encoding="utf-8")
    )
    episodes = (
        REPOSITORY
        / manifest["quarantine_path"]
        / "isaac/evaluation_episodes.jsonl"
    )
    if not episodes.is_file():
        raise SchemaValidationError("quarantined collision episodes are absent")
    return episodes


def _prefix(candidate_id: str) -> str:
    match = re.search(r"__(train|validation)-0*([0-9]+)-[^_]+__(.+)$", candidate_id)
    if match is None:
        raise SchemaValidationError("early collision candidate identity differs")
    return f"{match.group(1)}{int(match.group(2)):03d}_{match.group(3)}"


def main() -> int:
    arguments = _parser().parse_args()
    runtime_v13._load_contract()
    candidate_id = arguments.candidate_id
    batch_manifest = Path(arguments.batch_manifest).resolve()
    single_manifest = Path(arguments.single_manifest).resolve()
    batch_root = Path(arguments.batch_root).resolve()
    single_root = Path(arguments.single_root).resolve()
    batch_payload = json.loads(batch_manifest.read_text(encoding="utf-8"))
    single_payload = json.loads(single_manifest.read_text(encoding="utf-8"))
    batch_names = [str(value.get("name")) for value in batch_payload.get("jobs", [])]
    single_names = [str(value.get("name")) for value in single_payload.get("jobs", [])]
    if candidate_id not in batch_names or single_names != [candidate_id]:
        raise SchemaValidationError("early collision manifest membership differs")
    batch_index = batch_names.index(candidate_id)
    batch_records = _diagnostics(
        batch_manifest.with_suffix(".log"),
        environment_indices={2 * batch_index, 2 * batch_index + 1},
    )
    single_log = single_manifest.with_suffix(".log")
    single_records = (
        _diagnostics(single_log, environment_indices={0, 1})
        if single_log.is_file()
        else []
    )
    collision_keywords = {
        "body_name": arguments.expected_body_name,
        "body_index": arguments.expected_body_index,
        "phase_index": arguments.expected_phase_index,
        "selected_anchor_body": arguments.expected_selected_anchor_body,
    }
    _require_collision(batch_records, label="bounded batch", **collision_keywords)
    if single_records:
        _require_collision(
            single_records, label="single replay", **collision_keywords
        )

    batch_source = batch_root / candidate_id / "isaac/evaluation_episodes.jsonl"
    single_source = single_root / candidate_id / "isaac/evaluation_episodes.jsonl"
    _require_episodes(
        batch_source,
        candidate_id=candidate_id,
        label="bounded batch",
        phase_index=arguments.expected_phase_index,
    )
    _require_episodes(
        single_source,
        candidate_id=candidate_id,
        label="single replay",
        phase_index=arguments.expected_phase_index,
    )

    prefix = _prefix(candidate_id)
    repair_root = runtime_v13.OUTPUT_ROOT / "repair_sources"
    immutable_batch = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_early_object_collision_batch_manifest_v1.json",
        batch_payload,
    )
    immutable_single = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_early_object_collision_single_manifest_v1.json",
        single_payload,
    )
    batch_episodes = _quarantine(
        candidate_id,
        batch_root,
        reason="confirmed_early_selected_anchor_object_collision_bounded_batch",
    )
    single_episodes = _quarantine(
        candidate_id,
        single_root,
        reason="confirmed_early_selected_anchor_object_collision_single_replay",
    )

    prepared = (
        runtime_v13.OUTPUT_ROOT
        / "prepared/r1_l2_20mm_10deg"
        / candidate_id
    )
    case_manifest = json.loads(
        (prepared / "case_manifest.json").read_text(encoding="utf-8")
    )
    nominal = json.loads(
        (REPOSITORY / case_manifest["nominal_artifact"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    replacement_ports = list(
        _numbers(arguments.replacement_ports, count=2, cast=int)
    )
    replacement_tangent = list(
        _numbers(arguments.replacement_tangent_world_m, count=3, cast=float)
    )
    phase_name = ("approach", "contact_acquisition")[arguments.expected_phase_index]
    evidence = {
        "record_version": "order9_r1_confirmed_isaac_early_object_collision_v1",
        "candidate_id": candidate_id,
        "failure_kind": (
            "independently_repeated_selected_anchor_body_early_object_hard_collision"
            if arguments.expected_selected_anchor_body
            else "repeated_non_grasp_body_object_hard_collision"
        ),
        "first_failure_phase": phase_name,
        "first_failure_phase_index": arguments.expected_phase_index,
        "colliding_body_name": arguments.expected_body_name,
        "colliding_body_index": arguments.expected_body_index,
        "selected_anchor_body": arguments.expected_selected_anchor_body,
        "prohibited_object_contact": True,
        "prohibited_environment_contact": False,
        "original_selected_surface_port_ids": [
            int(value) for value in nominal["selected_surface_port_ids"]
        ],
        "original_selected_candidate_group_id": str(
            nominal["selected_candidate_group_id"]
        ),
        "replacement_teacher_option": {
            "selected_surface_port_ids": replacement_ports,
            "candidate_group_id": arguments.replacement_group_id,
            "pregrasp_clearance_m": arguments.replacement_pregrasp_clearance_m,
            "collision_margin_m": arguments.replacement_collision_margin_m,
            "grasp_contact_height_offset_m": arguments.replacement_height_m,
            "tangent_offset_world_m": replacement_tangent,
        },
        "independent_single_candidate_replay": True,
        "single_replay_body_diagnostic_available": bool(single_records),
        "bounded_batch_failure": {
            "path": str(batch_episodes.relative_to(REPOSITORY)),
            "sha256": hash_file(batch_episodes),
        },
        "single_replay_failure": {
            "path": str(single_episodes.relative_to(REPOSITORY)),
            "sha256": hash_file(single_episodes),
        },
        "bounded_batch_manifest": {
            "path": str(immutable_batch.relative_to(REPOSITORY)),
            "sha256": hash_file(immutable_batch),
        },
        "single_replay_manifest": {
            "path": str(immutable_single.relative_to(REPOSITORY)),
            "sha256": hash_file(immutable_single),
        },
        "controller_contract": {
            "pi_l_system_retained": True,
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "manifest_switch": "--diagnostic-nominal-qpid-only",
        },
        "repair_scope": "teacher_candidate_reselection_only",
        "contact_ik_constraints_changed": False,
        "acceptance_or_safety_gate_changed": False,
        "collision_phase_elapsed_s": [
            float(value["phase_elapsed_s"])
            for value in (single_records or batch_records)
        ],
    }
    evidence_path = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_confirmed_early_object_collision_v1.json",
        evidence,
    )
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "evidence_path": str(evidence_path.relative_to(REPOSITORY)),
                "evidence_sha256": hash_file(evidence_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
