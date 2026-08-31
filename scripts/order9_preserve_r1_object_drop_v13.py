#!/usr/bin/env python3
from __future__ import annotations

"""Preserve independently repeated R1 release-entry object-drop evidence."""

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
    return parser


def _read_episodes(path: Path, *, label: str) -> list[dict]:
    episodes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successful = [value for value in episodes if value.get("task_success") is True]
    failed = [value for value in episodes if value.get("task_success") is False]
    if len(episodes) != 2 or len(successful) != 1 or len(failed) != 1:
        raise SchemaValidationError(f"{label} object-drop episode count differs")
    episode = failed[0]
    metrics = episode.get("metrics") or {}
    if (
        episode.get("failure_reason") != "object_dropped"
        or episode.get("safety_failure") is not True
        or int(episode.get("fallback_decision_count", -1)) != 0
        or float(metrics.get("terminal_phase_index", -1.0)) != 5.0
        or float(metrics.get("object_dropped", -1.0)) != 1.0
        or float(metrics.get("timeout", -1.0)) != 0.0
        or float(metrics.get("hard_collision", -1.0)) != 0.0
        or float(metrics.get("qp_infeasible_terminal", -1.0)) != 0.0
    ):
        raise SchemaValidationError(f"{label} object-drop diagnosis differs")
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
        raise SchemaValidationError("object-drop evidence is absent")
    manifest = json.loads(
        (REPOSITORY / binding["path"]).read_text(encoding="utf-8")
    )
    episodes = REPOSITORY / manifest["quarantine_path"] / "isaac/evaluation_episodes.jsonl"
    if not episodes.is_file():
        raise SchemaValidationError("quarantined object-drop episodes are absent")
    return episodes


def _prefix(candidate_id: str) -> str:
    match = re.search(r"__(train|validation)-0*([0-9]+)-[^_]+__(.+)$", candidate_id)
    if match is None:
        raise SchemaValidationError("object-drop candidate identity differs")
    return f"{match.group(1)}{int(match.group(2)):03d}_{match.group(3)}"


def main() -> int:
    arguments = _parser().parse_args()
    runtime_v13._load_contract()
    candidate_id = arguments.candidate_id
    prefix = _prefix(candidate_id)
    repair_root = runtime_v13.OUTPUT_ROOT / "repair_sources"
    batch_manifest = Path(arguments.batch_manifest).resolve()
    single_manifest = Path(arguments.single_manifest).resolve()
    batch_payload = json.loads(batch_manifest.read_text(encoding="utf-8"))
    single_payload = json.loads(single_manifest.read_text(encoding="utf-8"))
    batch_names = [str(value.get("name")) for value in batch_payload.get("jobs", [])]
    single_names = [str(value.get("name")) for value in single_payload.get("jobs", [])]
    if candidate_id not in batch_names or single_names != [candidate_id]:
        raise SchemaValidationError("object-drop manifest membership differs")

    batch_source = (
        Path(arguments.batch_root).resolve()
        / candidate_id
        / "isaac/evaluation_episodes.jsonl"
    )
    single_source = (
        Path(arguments.single_root).resolve()
        / candidate_id
        / "isaac/evaluation_episodes.jsonl"
    )
    batch_episodes = _read_episodes(batch_source, label="bounded batch")
    single_episodes = _read_episodes(single_source, label="single replay")
    immutable_batch = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_object_drop_bounded_batch_manifest_v1.json",
        batch_payload,
    )
    immutable_single = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_object_drop_single_replay_manifest_v1.json",
        single_payload,
    )
    quarantined_batch = _quarantine(
        candidate_id,
        Path(arguments.batch_root).resolve(),
        reason="confirmed_release_entry_object_drop_bounded_batch",
    )
    quarantined_single = _quarantine(
        candidate_id,
        Path(arguments.single_root).resolve(),
        reason="confirmed_release_entry_object_drop_single_replay",
    )

    prepared_root = (
        runtime_v13.OUTPUT_ROOT / "prepared/r1_l2_20mm_10deg" / candidate_id
    )
    case_manifest = json.loads(
        (prepared_root / "case_manifest.json").read_text(encoding="utf-8")
    )
    nominal_path = REPOSITORY / case_manifest["nominal_artifact"]["path"]
    nominal = json.loads(nominal_path.read_text(encoding="utf-8"))
    evidence = {
        "record_version": "order9_r1_confirmed_isaac_object_drop_v1",
        "candidate_id": candidate_id,
        "failure_kind": "independently_repeated_release_entry_object_drop",
        "terminal_phase": "release",
        "terminal_phase_index": 5,
        "selected_surface_port_ids": [
            int(value) for value in nominal["selected_surface_port_ids"]
        ],
        "selected_candidate_group_id": str(
            nominal["selected_candidate_group_id"]
        ),
        "independent_single_candidate_replay": True,
        "bounded_batch_failure": {
            "path": str(quarantined_batch.relative_to(REPOSITORY)),
            "sha256": hash_file(quarantined_batch),
            "episode_count": len(batch_episodes),
            "success_count": 1,
            "object_drop_count": 1,
        },
        "single_replay_failure": {
            "path": str(quarantined_single.relative_to(REPOSITORY)),
            "sha256": hash_file(quarantined_single),
            "episode_count": len(single_episodes),
            "success_count": 1,
            "object_drop_count": 1,
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
        "proposed_repair_scope": "place_phase_duration_only",
        "proposed_phase_duration_multipliers": {
            "approach": 1.0,
            "contact_acquisition": 1.0,
            "lift": 1.0,
            "transport": 1.0,
            "place": 2.0,
            "release": 1.0,
            "retreat": 1.0,
            "settle": 1.0,
        },
        "spatial_or_joint_path_change_proposed": False,
        "acceptance_or_safety_gate_changed": False,
    }
    evidence_path = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_confirmed_isaac_object_drop_v1.json",
        evidence,
    )
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "object_drop_evidence_path": str(
                    evidence_path.relative_to(REPOSITORY)
                ),
                "object_drop_evidence_sha256": hash_file(evidence_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
