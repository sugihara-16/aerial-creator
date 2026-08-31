#!/usr/bin/env python3
from __future__ import annotations

"""Preserve independently repeated R1 phase-timeout evidence before repair."""

import argparse
import json
from pathlib import Path
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
    parser.add_argument("--replacement-group-id", required=True)
    return parser


def _read_episodes(path: Path, *, label: str) -> list[dict]:
    episodes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episodes) != 2:
        raise SchemaValidationError(f"{label} episode count differs")
    for episode in episodes:
        metrics = episode.get("metrics") or {}
        if (
            episode.get("task_success") is not False
            or episode.get("failure_reason") != "phase_timeout"
            or episode.get("safety_failure") is not False
            or int(episode.get("fallback_decision_count", -1)) != 0
            or float(metrics.get("terminal_phase_index", -1.0)) != 4.0
            or float(metrics.get("timeout", -1.0)) != 1.0
            or float(metrics.get("hard_collision", -1.0)) != 0.0
            or float(metrics.get("object_dropped", -1.0)) != 0.0
            or float(metrics.get("qp_infeasible_terminal", -1.0)) != 0.0
        ):
            raise SchemaValidationError(f"{label} timeout diagnosis differs")
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
        raise SchemaValidationError("phase-timeout evidence is absent")
    manifest = json.loads(
        (REPOSITORY / binding["path"]).read_text(encoding="utf-8")
    )
    episodes = REPOSITORY / manifest["quarantine_path"] / "isaac/evaluation_episodes.jsonl"
    if not episodes.is_file():
        raise SchemaValidationError("quarantined phase-timeout episodes are absent")
    return episodes


def main() -> int:
    arguments = _parser().parse_args()
    runtime_v13._load_contract()
    candidate_id = arguments.candidate_id
    sample = candidate_id.rsplit("__", 1)[-1]
    prefix = f"train027_{sample}"
    repair_root = runtime_v13.OUTPUT_ROOT / "repair_sources"
    batch_manifest = Path(arguments.batch_manifest).resolve()
    single_manifest = Path(arguments.single_manifest).resolve()
    batch_payload = json.loads(batch_manifest.read_text(encoding="utf-8"))
    single_payload = json.loads(single_manifest.read_text(encoding="utf-8"))
    batch_names = [str(value.get("name")) for value in batch_payload.get("jobs", [])]
    single_names = [str(value.get("name")) for value in single_payload.get("jobs", [])]
    if candidate_id not in batch_names or single_names != [candidate_id]:
        raise SchemaValidationError("phase-timeout manifest membership differs")

    batch_source = Path(arguments.batch_root).resolve() / candidate_id / "isaac/evaluation_episodes.jsonl"
    single_source = Path(arguments.single_root).resolve() / candidate_id / "isaac/evaluation_episodes.jsonl"
    batch_episodes = _read_episodes(batch_source, label="bounded batch")
    single_episodes = _read_episodes(single_source, label="single replay")
    immutable_batch = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_timeout_bounded_batch_manifest_v1.json",
        batch_payload,
    )
    immutable_single = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_timeout_single_replay_manifest_v1.json",
        single_payload,
    )
    quarantined_batch = _quarantine(
        candidate_id,
        Path(arguments.batch_root).resolve(),
        reason="confirmed_transport_phase_timeout_bounded_batch",
    )
    quarantined_single = _quarantine(
        candidate_id,
        Path(arguments.single_root).resolve(),
        reason="confirmed_transport_phase_timeout_single_replay",
    )

    prepared_root = (
        runtime_v13.OUTPUT_ROOT / "prepared/r1_l2_20mm_10deg" / candidate_id
    )
    case_manifest = json.loads(
        (prepared_root / "case_manifest.json").read_text(encoding="utf-8")
    )
    nominal_path = REPOSITORY / case_manifest["nominal_artifact"]["path"]
    nominal = json.loads(nominal_path.read_text(encoding="utf-8"))
    selected_ports = [int(value) for value in nominal["selected_surface_port_ids"]]
    selected_group = str(nominal["selected_candidate_group_id"])

    evidence = {
        "record_version": "order9_r1_confirmed_isaac_phase_timeout_v1",
        "candidate_id": candidate_id,
        "failure_kind": "repeated_transport_phase_timeout",
        "first_failure_phase": "transport",
        "terminal_phase_index": 4,
        "selected_surface_port_ids": selected_ports,
        "selected_candidate_group_id": selected_group,
        "replacement_candidate_group_id": arguments.replacement_group_id,
        "independent_single_candidate_replay": True,
        "bounded_batch_failure": {
            "path": str(quarantined_batch.relative_to(REPOSITORY)),
            "sha256": hash_file(quarantined_batch),
            "episode_count": len(batch_episodes),
        },
        "single_replay_failure": {
            "path": str(quarantined_single.relative_to(REPOSITORY)),
            "sha256": hash_file(quarantined_single),
            "episode_count": len(single_episodes),
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
        "repair_scope": "teacher_posture_branch_and_configuration_space_goal_initialization",
        "contact_goal_seed_applied": False,
        "contact_ik_constraints_changed": False,
        "acceptance_or_safety_gate_changed": False,
    }
    evidence_path = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_confirmed_isaac_transport_timeout_v1.json",
        evidence,
    )
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "timeout_evidence_path": str(evidence_path.relative_to(REPOSITORY)),
                "timeout_evidence_sha256": hash_file(evidence_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
