#!/usr/bin/env python3
from __future__ import annotations

"""Preserve repeated R1 support-collision evidence before teacher repair."""

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
    parser.add_argument("--reference-candidate-id", required=True)
    parser.add_argument("--replacement-height-m", type=float, default=0.03)
    return parser


def _diagnostics(path: Path, *, environment_indices: set[int]):
    prefix = "ORDER9_COLLISION_DIAGNOSTIC="
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        value = json.loads(line[len(prefix) :])
        if int(value.get("environment_index", -1)) in environment_indices:
            records.append(value)
    return records


def _require_collision(records, *, label: str):
    if len(records) != 2:
        raise SchemaValidationError(f"{label} collision count differs")
    for record in records:
        bodies = record.get("active_bodies")
        body = next(
            (
                value
                for value in bodies or []
                if value.get("body_name") == "module_3__yaw_dock_mech2"
            ),
            None,
        )
        if (
            int(record.get("runtime_phase_index", -1)) != 2
            or record.get("prohibited_environment_contact") is not True
            or record.get("prohibited_object_contact") is not False
            or body is None
            or int(body.get("body_index", -1)) != 133
            or float(body.get("environment_force_norm_n", 0.0)) <= 0.0
            or float(body.get("object_force_norm_n", -1.0)) != 0.0
        ):
            raise SchemaValidationError(f"{label} collision diagnosis differs")


def _quarantine(candidate_id: str, evidence_root: Path, *, reason: str):
    binding = runtime_v13.runtime_v10._quarantine_invalid_candidate_output(
        candidate_id=candidate_id,
        evidence_root=evidence_root,
        preparation_root=(
            runtime_v13.OUTPUT_ROOT / "prepared" / "r1_l2_20mm_10deg"
        ),
        failure_reason=reason,
    )
    if binding is None:
        raise SchemaValidationError("support-collision evidence is absent")
    manifest = json.loads(
        (REPOSITORY / binding["path"]).read_text(encoding="utf-8")
    )
    root = REPOSITORY / manifest["quarantine_path"]
    episodes = root / "isaac/evaluation_episodes.jsonl"
    if not episodes.is_file():
        raise SchemaValidationError("quarantined collision episodes are absent")
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
        raise SchemaValidationError("support-collision manifest membership differs")
    batch_index = batch_names.index(candidate_id)
    batch_records = _diagnostics(
        batch_manifest.with_suffix(".log"),
        environment_indices={2 * batch_index, 2 * batch_index + 1},
    )
    single_records = _diagnostics(
        single_manifest.with_suffix(".log"), environment_indices={0, 1}
    )
    _require_collision(batch_records, label="bounded batch")
    _require_collision(single_records, label="single replay")

    immutable_batch = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_bounded_batch_manifest_v1.json", batch_payload
    )
    immutable_single = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_single_replay_manifest_v1.json", single_payload
    )
    batch_episodes = _quarantine(
        candidate_id,
        Path(arguments.batch_root).resolve(),
        reason="confirmed_fixed_support_collision_bounded_batch",
    )
    single_episodes = _quarantine(
        candidate_id,
        Path(arguments.single_root).resolve(),
        reason="confirmed_fixed_support_collision_single_replay",
    )

    prepared_root = (
        runtime_v13.OUTPUT_ROOT
        / "prepared/r1_l2_20mm_10deg"
        / candidate_id
    )
    case_manifest = json.loads(
        (prepared_root / "case_manifest.json").read_text(encoding="utf-8")
    )
    nominal_manifest_path = REPOSITORY / case_manifest["nominal_artifact"]["path"]
    nominal = json.loads(nominal_manifest_path.read_text(encoding="utf-8"))
    selected_ports = [int(value) for value in nominal["selected_surface_port_ids"]]
    selected_group = str(nominal["selected_candidate_group_id"])
    if selected_ports != [21, 29] or selected_group != "slot_0:grasp_pair:3":
        raise SchemaValidationError("support-collision selected teacher differs")

    collision = {
        "record_version": "order9_r1_confirmed_isaac_teacher_collision_v1",
        "candidate_id": candidate_id,
        "failure_kind": "repeated_non_grasp_body_object_hard_collision",
        "first_failure_phase": "lift",
        "colliding_body_name": "module_3__yaw_dock_mech2",
        "colliding_body_index": 133,
        "prohibited_object_contact": False,
        "prohibited_environment_contact": True,
        "selected_surface_port_ids": selected_ports,
        "selected_candidate_group_id": selected_group,
        "replacement_candidate_group_id": selected_group,
        "replacement_grasp_contact_height_offset_m": float(
            arguments.replacement_height_m
        ),
        "independent_single_candidate_replay": True,
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
        "repair_scope": (
            "teacher_candidate_reselection_and_configuration_space_goal_initialization"
        ),
        "contact_goal_seed_applied": False,
        "contact_ik_constraints_changed": False,
        "acceptance_or_safety_gate_changed": False,
        "diagnosis": {
            "bounded_batch_collision_count": 2,
            "single_replay_collision_count": 2,
            "collision_runtime_phase_index": 2,
            "collision_phase_elapsed_s": [
                float(value["phase_elapsed_s"]) for value in single_records
            ],
            "same_surface_pair_success_reference_candidate_id": (
                arguments.reference_candidate_id
            ),
            "same_surface_pair_success_reference_candidate_group_id": selected_group,
            "reason": (
                "Both the bounded batch and independent replay hit the fixed "
                "support with module_3__yaw_dock_mech2 shortly after lift began. "
                "Keep the [21,29] group 3 grasp, raise the contact height by 10 mm, "
                "and initialize configuration-space search from the hash-bound "
                "nearby group 3 trajectory. The complete CPU screen accepted "
                "this option without changing any gate."
            ),
        },
    }
    collision_path = runtime_v13.runtime_v10._atomic_json(
        repair_root / f"{prefix}_confirmed_isaac_support_collision_v1.json",
        collision,
    )
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "collision_evidence_path": str(collision_path.relative_to(REPOSITORY)),
                "collision_evidence_sha256": hash_file(collision_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
