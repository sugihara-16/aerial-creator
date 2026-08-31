#!/usr/bin/env python3
from __future__ import annotations

"""Verify the R1 v11 geometry correction on the previously failed candidate."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.contact_candidates import ContactCandidateSet  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.policies import ContactWrenchTrajectory  # noqa: E402
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_complete_task_geometry_v11 import (  # noqa: E402
    ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_FILENAME,
    audit_order9_r1_complete_task_geometry_v11,
)
from amsrr.training.order9_r1_complete_task_materialization_v11 import (  # noqa: E402
    finalize_order9_r1_v11_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    run_order9_r1_nominal_isaac_cases,
    write_order9_r1_nominal_case_contract,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    build_order9_r1_geometry_variants_v11,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v11 import (  # noqa: E402
    Order9R1RangeTeacherScreenPipelineV11,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

RUNNER_VERSION = "order9_r1_nominal_geometry_v11_smoke"
BASE_PROTOCOL = REPOSITORY / "configs/training/order9_r1_calibration_protocol_v1.yaml"
DEFAULT_CANDIDATE = "r1_l2_20mm_10deg__train__train-000000-5fecfad4f44f__lattice_02"
OLD_CASE_ROOT = (
    REPOSITORY / "artifacts/p4_full/order9/r1_teacher/range_selection_v10/prepared/"
    "r1_l2_20mm_10deg" / DEFAULT_CANDIDATE
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY / "artifacts/p4_full/order9/r1_teacher/nominal_geometry_v11_smoke"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be positive")
    started = perf_counter()
    contract = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    case = next(
        (
            value
            for value in enumerate_order9_r1_calibration_level_cases(
                protocol_path=BASE_PROTOCOL,
                repository_root=REPOSITORY,
                level_id="r1_l2_20mm_10deg",
                split="train",
            )
            if value.candidate_id == args.candidate_id
        ),
        None,
    )
    if case is None:
        raise SchemaValidationError("R1 v11 requested candidate is unknown")
    variants = build_order9_r1_geometry_variants_v11(case, contract=contract)
    corrected = variants.nominal_case
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rejection_path = output_root / "small_object_pre_isaac_rejection.json"
    if rejection_path.is_file():
        small_audit = json.loads(rejection_path.read_text(encoding="utf-8"))
    else:
        small_audit = _audit_historical_small_object(
            task_spec=variants.small_object_task_spec,
            contract=contract,
        )
        small_audit.update(
            {
                "candidate_id": args.candidate_id,
                "source_case_root": str(OLD_CASE_ROOT),
                "source_case_manifest_sha256": hash_file(
                    OLD_CASE_ROOT / "case_manifest.json"
                ),
                "expected_disposition": "rejected_before_isaac",
                "small_object_role": "difficult_condition_only",
                "formal_range_selection_eligible": False,
            }
        )
        write_order9_r1_clearance_diagnostic(small_audit, rejection_path)
    if small_audit.get("accepted") is not False:
        raise SchemaValidationError(
            "R1 v11 failed to reject the historical small-object path"
        )

    case_root = output_root / "prepared" / corrected.level_id / corrected.candidate_id
    geometry_path = case_root / ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_FILENAME
    if case_root.is_dir() and geometry_path.is_file():
        geometry_audit = json.loads(geometry_path.read_text(encoding="utf-8"))
        if geometry_audit.get("accepted") is not True:
            raise SchemaValidationError("existing R1 v11 geometry audit is rejected")
        materialized = load_order9_r1_isaac_case(
            case_root / "case_manifest.json", REPOSITORY
        )
    else:
        if case_root.exists():
            raise FileExistsError(
                f"incomplete R1 v11 case exists and was preserved: {case_root}"
            )
        base = load_order9_r1_calibration_protocol(
            BASE_PROTOCOL,
            repository_root=REPOSITORY,
        )
        pipeline = Order9R1RangeTeacherScreenPipelineV11(
            repository_root=REPOSITORY,
            source_bucket_manifest_path=(REPOSITORY / base.source_bucket_manifest.path),
            minimum_normalized_joint_limit_reserve=(
                base.minimum_normalized_joint_limit_reserve
            ),
            maximum_body_tilt_rad=base.maximum_body_tilt_rad,
            anchor_position_tolerance_m=0.030,
            enforce_joint_limit_reserve_during_ik=True,
            geometry_contract=contract,
        )
        prepared = pipeline.prepare(corrected)
        prepared.validate_for(corrected)
        if not prepared.teacher_trajectory_complete:
            raise SchemaValidationError(
                "R1 v11 corrected teacher failed: " + str(prepared.failure_reason)
            )
        screen = pipeline.screen(corrected, prepared)
        if not screen.accepted or not screen.eligible_for_full_control_test:
            raise SchemaValidationError(
                "R1 v11 corrected fast screen failed: "
                + ",".join(screen.violation_codes)
            )
        materialized = materialize_order9_r1_isaac_case(
            case=corrected,
            prepared=prepared,
            screen=screen,
            output_dir=case_root,
            repository_root=REPOSITORY,
            approved_protocol_path=BASE_PROTOCOL,
        )
        materialized = finalize_order9_r1_v11_materialized_case(
            materialized,
            task_spec=corrected.task_spec,
            phase_time_scales=order9_r1_safe_phase_time_scales(
                corrected.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=REPOSITORY,
            geometry_contract=contract,
        )
        geometry_audit = json.loads(geometry_path.read_text(encoding="utf-8"))

    results = ()
    if not args.prepare_only:
        contract_path = case_root / "nominal_contract.json"
        if not contract_path.is_file():
            write_order9_r1_nominal_case_contract(
                materialized,
                repository_root=REPOSITORY,
            )
        result_map = run_order9_r1_nominal_isaac_cases(
            [materialized],
            repository_root=REPOSITORY,
            python_executable=args.isaac_python,
            rollout_steps=args.rollout_steps,
            maximum_parallel_process_count=1,
            persistent_morphology_coalescing=False,
        )
        results = result_map[corrected.candidate_id]

    payload = {
        "result_version": RUNNER_VERSION,
        "status": (
            "prepared"
            if args.prepare_only
            else ("passed" if all(value.success for value in results) else "failed")
        ),
        "candidate_id": corrected.candidate_id,
        "historical_small_object_rejected_before_isaac": True,
        "corrected_nominal_geometry_admitted_before_isaac": (
            geometry_audit.get("accepted") is True
        ),
        "object_height_increment_m": contract.object_height_increment_m,
        "grasp_contact_height_offset_m": contract.grasp_contact_height_offset_m,
        "required_robot_support_clearance_m": (
            contract.required_robot_support_clearance_m
        ),
        "checked_phases": geometry_audit.get("checked_phases"),
        "checked_knot_count": geometry_audit.get("checked_knot_count"),
        "minimum_support_or_self_clearance_m": geometry_audit.get(
            "minimum_support_or_self_clearance_m"
        ),
        "episode_count": len(results),
        "success_count": sum(value.success for value in results),
        "safety_failure_count": sum(value.safety_failure for value in results),
        "fallback_count": sum(value.fallback_used for value in results),
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "isaac_invoked": not args.prepare_only,
        "formal_teacher_collection_authorized": False,
        "wall_time_s": perf_counter() - started,
        "bindings": {
            "geometry_config": _binding(contract.config_path),
            "small_object_rejection": _binding(rejection_path),
            "case_manifest": _binding(case_root / "case_manifest.json"),
            "complete_task_geometry_audit": _binding(geometry_path),
            "runner": _binding(Path(__file__)),
        },
    }
    result_path = write_order9_r1_clearance_diagnostic(
        payload,
        output_root / "result.json",
    )
    print(
        "ORDER9_R1_NOMINAL_GEOMETRY_V11="
        + json.dumps(
            {
                "status": payload["status"],
                "result": str(result_path),
                "sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if payload["status"] in {"prepared", "passed"} else 1


def _audit_historical_small_object(*, task_spec, contract):
    artifact_path = (
        OLD_CASE_ROOT / "nominal_set/buckets" / DEFAULT_CANDIDATE / "manifest.json"
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    root = artifact_path.parent
    phases = {
        value.phase: ContactWrenchTrajectory.from_json(
            (root / value.trajectory_path).read_text(encoding="utf-8")
        )
        for value in artifact.phase_trajectories
    }
    morphology = MorphologyGraph.from_json(
        (root / artifact.task_conditioned_morphology_path).read_text(encoding="utf-8")
    )
    candidates = ContactCandidateSet.from_json(
        (root / artifact.contact_candidate_set_path).read_text(encoding="utf-8")
    )
    return audit_order9_r1_complete_task_geometry_v11(
        phases=phases,
        task_spec=task_spec,
        morphology=morphology,
        contact_candidate_set=candidates,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
        contract=contract,
        fail_fast=True,
    )


def _binding(path: Path):
    resolved = path.resolve()
    try:
        portable = str(resolved.relative_to(REPOSITORY))
    except ValueError:
        portable = str(resolved)
    return {"path": portable, "sha256": hash_file(resolved)}


if __name__ == "__main__":
    raise SystemExit(main())
