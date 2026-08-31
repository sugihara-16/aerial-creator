#!/usr/bin/env python3
from __future__ import annotations

"""Promote a proven R1 early-collision repair without changing v13."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_isaac_calibration import load_order9_r1_isaac_case
from amsrr.training.order9_r1_phase_duration_repair import _replace_portable_root
from amsrr.training.order9_r1_yaw_branch_repair import _portable, _write_json_text
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_range_selection_v13 as runtime_v13


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-evidence", required=True)
    parser.add_argument("--diagnostic-case-manifest")
    parser.add_argument("--diagnostic-manifest")
    parser.add_argument("--diagnostic-episodes")
    parser.add_argument(
        "--promotion-case-manifest",
        help="Optional materialized derivative of the diagnostic case to promote.",
    )
    parser.add_argument("--destination-case-root", required=True)
    return parser


def _binding(value: object, *, label: str) -> Path:
    binding = value if isinstance(value, dict) else {}
    path = (REPOSITORY / str(binding.get("path", ""))).resolve()
    if (
        REPOSITORY not in path.parents
        or not path.is_file()
        or binding.get("sha256") != hash_file(path)
    ):
        raise ValueError(f"R1 early-collision {label} binding differs")
    return path


def _episodes(path: Path) -> tuple[dict, ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _validated_external_path(value: str | None, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"R1 early-collision {label} is required")
    path = Path(value).resolve()
    if REPOSITORY not in path.parents or not path.is_file():
        raise ValueError(f"R1 early-collision {label} differs")
    return path


def _validate_evidence(
    path: Path,
    *,
    diagnostic_case_manifest: str | None,
    diagnostic_manifest: str | None,
    diagnostic_episodes: str | None,
) -> tuple[dict, Path, dict]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    candidate_id = evidence.get("candidate_id")
    selected = evidence.get("selected_anchor_body")
    expected_kind = (
        "independently_repeated_selected_anchor_body_early_object_hard_collision"
        if selected is True
        else "repeated_non_grasp_body_object_hard_collision"
    )
    contract = evidence.get("controller_contract") or {}
    option = evidence.get("replacement_teacher_option")
    if (
        evidence.get("record_version")
        != "order9_r1_confirmed_isaac_early_object_collision_v1"
        or not isinstance(candidate_id, str)
        or evidence.get("failure_kind") != expected_kind
        or selected not in (True, False)
        or int(evidence.get("first_failure_phase_index", -1)) not in (0, 1)
        or evidence.get("prohibited_object_contact") is not True
        or evidence.get("prohibited_environment_contact") is not False
        or evidence.get("independent_single_candidate_replay") is not True
        or evidence.get("repair_scope") != "teacher_candidate_reselection_only"
        or evidence.get("contact_ik_constraints_changed") is not False
        or evidence.get("acceptance_or_safety_gate_changed") is not False
        or not isinstance(option, dict)
        or contract.get("pi_l_actor_command_applied") is not False
        or contract.get("qpid_qp_applied") is not True
        or contract.get("local_servo_applied") is not True
    ):
        raise ValueError("R1 early-collision repair evidence differs")

    phase_index = int(evidence["first_failure_phase_index"])
    generation_id = f"r1_nominal_calibration:{candidate_id}"
    for label in ("bounded_batch_failure", "single_replay_failure"):
        episodes = _episodes(_binding(evidence.get(label), label=label))
        if len(episodes) != 2 or any(
            episode.get("task_success") is not False
            or episode.get("safety_failure") is not True
            or episode.get("failure_reason") != "hard_collision"
            or int(episode.get("fallback_decision_count", -1)) != 0
            or episode.get("isaac_backed") is not True
            or (episode.get("metadata") or {}).get("generation_id")
            != generation_id
            or int(
                float((episode.get("metrics") or {}).get("terminal_phase_index", -1))
            )
            != phase_index
            for episode in episodes
        ):
            raise ValueError("R1 early-collision failure replay differs")

    if evidence.get("replacement_diagnostic_preparation") is not None:
        preparation_path = _binding(
            evidence.get("replacement_diagnostic_preparation"),
            label="replacement_diagnostic_preparation",
        )
    else:
        preparation_path = (
            _validated_external_path(
                diagnostic_case_manifest,
                label="diagnostic_case_manifest",
            ).parent
            / "diagnostic_teacher_option_v1.json"
        )
        if not preparation_path.is_file():
            raise ValueError("R1 early-collision diagnostic preparation is absent")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if (
        preparation.get("candidate_id") != candidate_id
        or preparation.get("teacher_option") != option
        or preparation.get("teacher_trajectory_complete") is not True
        or preparation.get("screen_accepted") is not True
        or preparation.get("eligible_for_full_control_test") is not True
        or preparation.get("isaac_invoked") is not False
        or preparation.get("controller_layers_invoked") is not False
        or preparation.get("acceptance_or_safety_gate_changed") is not False
    ):
        raise ValueError("R1 early-collision diagnostic preparation differs")

    success_path = (
        _binding(
            evidence.get("replacement_diagnostic_success"),
            label="replacement_diagnostic_success",
        )
        if evidence.get("replacement_diagnostic_success") is not None
        else _validated_external_path(
            diagnostic_episodes,
            label="diagnostic_episodes",
        )
    )
    success = _episodes(success_path)
    if len(success) != 2 or any(
        episode.get("task_success") is not True
        or episode.get("safety_failure") is not False
        or episode.get("failure_reason") is not None
        or int(episode.get("fallback_decision_count", -1)) != 0
        or episode.get("isaac_backed") is not True
        or (episode.get("metadata") or {}).get("generation_id") != generation_id
        for episode in success
    ):
        raise ValueError("R1 early-collision diagnostic replay failed")

    manifest_paths = {
        label: _binding(evidence.get(label), label=label)
        for label in ("bounded_batch_manifest", "single_replay_manifest")
    }
    manifest_paths["replacement_diagnostic_manifest"] = (
        _binding(
            evidence.get("replacement_diagnostic_manifest"),
            label="replacement_diagnostic_manifest",
        )
        if evidence.get("replacement_diagnostic_manifest") is not None
        else _validated_external_path(
            diagnostic_manifest,
            label="diagnostic_manifest",
        )
    )
    for label, manifest_path in manifest_paths.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = [job for job in manifest.get("jobs", ()) if job.get("name") == candidate_id]
        argv = jobs[0].get("argv") if len(jobs) == 1 else None
        if (
            not isinstance(argv, list)
            or "--diagnostic-nominal-qpid-only" not in argv
            or "--evaluation-episode-count" not in argv
            or argv[argv.index("--evaluation-episode-count") + 1] != "2"
        ):
            raise ValueError("R1 early-collision controller manifest differs")
    diagnostic_bindings = {
        "preparation": {
            "path": str(preparation_path.relative_to(REPOSITORY)),
            "sha256": hash_file(preparation_path),
        },
        "manifest": {
            "path": str(
                manifest_paths["replacement_diagnostic_manifest"].relative_to(
                    REPOSITORY
                )
            ),
            "sha256": hash_file(
                manifest_paths["replacement_diagnostic_manifest"]
            ),
        },
        "episodes": {
            "path": str(success_path.relative_to(REPOSITORY)),
            "sha256": hash_file(success_path),
        },
    }
    return evidence, preparation_path.parent, diagnostic_bindings


def _relocate(
    *,
    source: Path,
    destination: Path,
    evidence_path: Path,
    diagnostic_bindings: dict,
) -> dict:
    source_relative = _portable(source, REPOSITORY)
    destination_relative = _portable(destination, REPOSITORY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(source, temporary, ignore=shutil.ignore_patterns("isaac"))
        set_path = temporary / "nominal_set/manifest.json"
        nominal_set = Order9C3NominalTrajectorySetManifest.from_json(
            set_path.read_text(encoding="utf-8")
        )
        nominal_set.validate()
        if len(nominal_set.entries) != 1:
            raise ValueError("R1 early-collision diagnostic set is not singular")
        artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
        artifact_payload = _replace_portable_root(
            json.loads(artifact_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        _write_json_text(artifact_path, artifact_payload)
        validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)

        collision_path = set_path.parent / "collision_validation.json"
        collision_payload = _replace_portable_root(
            json.loads(collision_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        _write_json_text(collision_path, collision_payload)
        nominal_set.entries[0].artifact_sha256 = hash_file(artifact_path)
        set_payload = _replace_portable_root(
            nominal_set.to_dict(),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        _write_json_text(set_path, set_payload)

        case_path = temporary / "case_manifest.json"
        case_payload = _replace_portable_root(
            json.loads(case_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        case_payload["nominal_set"]["sha256"] = hash_file(set_path)
        case_payload["nominal_artifact"]["sha256"] = hash_file(artifact_path)
        _write_json_text(case_path, case_payload)

        reset_path = temporary / "reset_bank.pt"
        reset = torch.load(reset_path, map_location="cpu", weights_only=False)
        reference = (reset.get("contract") or {}).get("c3_nominal_reference")
        if not isinstance(reference, dict):
            raise ValueError("R1 early-collision reset reference is absent")
        reference.update(
            {
                "set_manifest_path": _portable(
                    destination / "nominal_set/manifest.json", REPOSITORY
                ),
                "set_manifest_sha256": hash_file(set_path),
                "artifact_path": _portable(
                    destination / "nominal_set" / nominal_set.entries[0].artifact_path,
                    REPOSITORY,
                ),
                "artifact_sha256": hash_file(artifact_path),
                "timeline_sha256": hash_file(
                    artifact_path.parent / "nominal_timeline.json"
                ),
            }
        )
        torch.save(reset, reset_path)
        admission = {
            "admission_version": "order9_r1_early_object_collision_repair_v1",
            "status": "accepted",
            "candidate_id": case_payload["candidate_id"],
            "source_failure_evidence": {
                "path": str(evidence_path.relative_to(REPOSITORY)),
                "sha256": hash_file(evidence_path),
            },
            "source_diagnostic_case": {
                "path": source_relative + "/case_manifest.json",
                "sha256": hash_file(source / "case_manifest.json"),
            },
            "diagnostic_success": diagnostic_bindings,
            "case_manifest_sha256": hash_file(case_path),
            "nominal_set_sha256": hash_file(set_path),
            "nominal_artifact_sha256": hash_file(artifact_path),
            "acceptance_or_safety_gate_changed": False,
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
        }
        _write_json_text(
            temporary / "early_object_collision_repair_admission.json", admission
        )
        os.rename(temporary, destination)
        load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
        return admission
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = _parser().parse_args()
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    evidence_path = Path(args.failure_evidence).resolve()
    evidence, source, diagnostic_bindings = _validate_evidence(
        evidence_path,
        diagnostic_case_manifest=args.diagnostic_case_manifest,
        diagnostic_manifest=args.diagnostic_manifest,
        diagnostic_episodes=args.diagnostic_episodes,
    )
    if args.promotion_case_manifest is not None:
        promotion_manifest = _validated_external_path(
            args.promotion_case_manifest,
            label="promotion_case_manifest",
        )
        promotion_payload = json.loads(
            promotion_manifest.read_text(encoding="utf-8")
        )
        promotion_admission_path = (
            promotion_manifest.parent
            / "release_clearance_materialization_admission_v1.json"
        )
        if not promotion_admission_path.is_file():
            raise ValueError("R1 early-collision promotion admission is absent")
        promotion_admission = json.loads(
            promotion_admission_path.read_text(encoding="utf-8")
        )
        if (
            promotion_payload.get("candidate_id") != evidence["candidate_id"]
            or promotion_admission.get("status") != "accepted"
            or promotion_admission.get("candidate_id") != evidence["candidate_id"]
            or promotion_admission.get("acceptance_or_safety_gate_changed")
            is not False
            or promotion_admission.get("diagnostic_manifest_sha256")
            != diagnostic_bindings["manifest"]["sha256"]
            or promotion_admission.get("diagnostic_episodes_sha256")
            != diagnostic_bindings["episodes"]["sha256"]
        ):
            raise ValueError("R1 early-collision promotion case differs")
        source = promotion_manifest.parent
    destination = Path(args.destination_case_root).resolve()
    expected_destination = (
        runtime_v13.OUTPUT_ROOT
        / "prepared/r1_l2_20mm_10deg"
        / evidence["candidate_id"]
    ).resolve()
    if destination != expected_destination:
        raise ValueError("R1 early-collision destination differs")
    case = next(
        case
        for case in runtime_v13.runtime_v11._corrected_cases(
            "r1_l2_20mm_10deg", "validation"
        )
        if case.candidate_id == evidence["candidate_id"]
    )
    runtime_v13._validate_prepared_case(case, source)
    if destination.exists():
        runtime_v13._quarantine_preparation_destination(
            destination, reason="superseded_early_object_collision_repair"
        )
    admission = _relocate(
        source=source,
        destination=destination,
        evidence_path=evidence_path,
        diagnostic_bindings=diagnostic_bindings,
    )
    runtime_v13._validate_prepared_case(case, destination)
    print(
        "ORDER9_R1_EARLY_COLLISION_REPAIR_PROMOTED="
        + json.dumps(admission, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
