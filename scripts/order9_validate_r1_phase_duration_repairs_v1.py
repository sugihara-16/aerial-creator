#!/usr/bin/env python3
from __future__ import annotations

"""Validate supplemental R1 place-duration repairs without mutating v13."""

import json
from pathlib import Path
import sys

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_isaac_calibration import load_order9_r1_isaac_case
from amsrr.training.order9_r1_phase_duration_repair import (
    ORDER9_R1_PHASE_DURATION_REPAIRS_RELATIVE,
    load_order9_r1_phase_duration_repairs,
)
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_range_selection_v13 as runtime_v13

APPROVAL = REPOSITORY / "for_codex/R1_PHASE_DURATION_REPAIR_V1_APPROVAL.json"
LEVEL = "r1_l2_20mm_10deg"


def _require_binding(value: object, *, label: str) -> Path:
    binding = value if isinstance(value, dict) else {}
    path = (REPOSITORY / str(binding.get("path", ""))).resolve()
    if (
        REPOSITORY not in path.parents
        or not path.is_file()
        or binding.get("sha256") != hash_file(path)
    ):
        raise SchemaValidationError(f"R1 phase repair {label} binding differs")
    return path


def main() -> int:
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if (
        approval.get("record_version")
        != "order9_r1_phase_duration_repair_approval_v1"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or int(approval.get("candidate_count", -1)) != 3
        or float(approval.get("place_source_duration_s", -1.0)) != 3.0
        or float(approval.get("place_repaired_duration_s", -1.0)) != 6.0
        or approval.get("spatial_path_changed") is not False
        or approval.get("joint_path_changed") is not False
        or approval.get("acceptance_or_safety_gate_changed") is not False
        or approval.get("learning_authorized") is not False
        or approval.get("teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 phase-duration supplemental approval differs")
    for label in (
        "base_range_protocol",
        "base_range_approval",
        "repair_config",
        "repair_implementation",
        "failure_preservation_implementation",
        "validator_implementation",
    ):
        _require_binding(approval.get(label), label=label)

    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    repairs = load_order9_r1_phase_duration_repairs(
        ORDER9_R1_PHASE_DURATION_REPAIRS_RELATIVE,
        repository_root=REPOSITORY,
    )
    cases = {
        case.candidate_id: case
        for case in runtime_v13.runtime_v11._corrected_cases(LEVEL, "validation")
    }
    preparation_root = runtime_v13.OUTPUT_ROOT / "prepared" / LEVEL
    isaac_root = runtime_v13.OUTPUT_ROOT / "isaac" / LEVEL
    records = []
    for candidate_id, repair in sorted(repairs.items()):
        case = cases.get(candidate_id)
        if case is None:
            raise SchemaValidationError("R1 phase repair candidate is not validation")
        case_root = preparation_root / candidate_id
        runtime_v13._validate_prepared_case(case, case_root)
        materialized = load_order9_r1_isaac_case(
            case_root / "case_manifest.json", REPOSITORY
        )
        artifact_path = REPOSITORY / materialized.manifest.nominal_artifact.path
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        provenance = (artifact.get("selection_evidence") or {}).get(
            "r1_phase_duration_repair"
        )
        admission_path = case_root / "phase_duration_repair_admission.json"
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        multipliers = repair["phase_duration_multipliers"]
        if (
            not isinstance(provenance, dict)
            or provenance.get("candidate_id") != candidate_id
            or provenance.get("phase_duration_multipliers") != multipliers
            or provenance.get("spatial_path_changed") is not False
            or provenance.get("joint_path_changed") is not False
            or admission.get("status") != "accepted"
            or admission.get("candidate_id") != candidate_id
            or admission.get("phase_duration_multipliers") != multipliers
            or admission.get("case_manifest_sha256")
            != hash_file(case_root / "case_manifest.json")
            or admission.get("formal_teacher_collection_authorized") is not False
        ):
            raise SchemaValidationError("R1 phase repair prepared case differs")

        reset = torch.load(
            case_root / "reset_bank.pt", map_location="cpu", weights_only=False
        )
        reference = (reset.get("contract") or {}).get("c3_nominal_reference")
        timeline_path = artifact_path.parent / "nominal_timeline.json"
        if (
            not isinstance(reference, dict)
            or reference.get("set_manifest_path")
            != materialized.manifest.nominal_set.path
            or reference.get("set_manifest_sha256")
            != materialized.manifest.nominal_set.sha256
            or reference.get("artifact_path")
            != materialized.manifest.nominal_artifact.path
            or reference.get("artifact_sha256")
            != materialized.manifest.nominal_artifact.sha256
            or reference.get("timeline_sha256") != hash_file(timeline_path)
        ):
            raise SchemaValidationError("R1 phase repair reset-bank binding differs")

        outputs = list(
            (isaac_root / "batches" / case.source_bucket.bucket_id).glob(
                f"chunk_*/{candidate_id}/isaac/evaluation_episodes.jsonl"
            )
        )
        if len(outputs) != 1:
            raise SchemaValidationError("R1 phase repair formal output count differs")
        evidence_root = outputs[0].parents[2]
        evidence = runtime_v13.runtime_v10._inspect_candidate_output(
            candidate_id,
            evidence_root,
            preparation_root=preparation_root,
            quarantine_invalid=False,
        )
        if not evidence.get("valid_evidence") or not evidence.get("candidate_passed"):
            raise SchemaValidationError("R1 phase repair formal replay failed")
        records.append(
            {
                "candidate_id": candidate_id,
                "place_duration_multiplier": multipliers["place"],
                "success_count": evidence["success_count"],
                "episode_count": evidence["episode_count"],
                "safety_failure_count": evidence["safety_failure_count"],
                "fallback_count": evidence["fallback_count"],
                "formal_episodes": evidence["episodes"],
            }
        )
    print(
        "ORDER9_R1_PHASE_DURATION_REPAIRS_VALID="
        + json.dumps(
            {
                "candidate_count": len(records),
                "episode_count": sum(value["episode_count"] for value in records),
                "success_count": sum(value["success_count"] for value in records),
                "safety_failure_count": sum(
                    value["safety_failure_count"] for value in records
                ),
                "fallback_count": sum(value["fallback_count"] for value in records),
                "records": records,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
