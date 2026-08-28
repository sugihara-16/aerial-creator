#!/usr/bin/env python3
from __future__ import annotations

"""Validate the R1 six-module grasp-rotation repair in two Isaac replays."""

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

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
from amsrr.training.order9_r1_complete_task_materialization_v7 import (  # noqa: E402
    ORDER9_R1_GRASP_ROTATION_AUDIT_FILENAME,
    finalize_order9_r1_v7_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v3 import (  # noqa: E402
    run_order9_r1_nominal_v3_isaac_cases,
)
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (  # noqa: E402
    ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE,
    Order9R1RotationalTeacherScreenPipelineV7,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

_BASE_PROTOCOL = Path("configs/training/order9_r1_calibration_protocol_v1.yaml")
_SOURCE_CANDIDATE_ID = "r1_l1_10mm_5deg__train__train-000004-190f3a425b3e__lattice_02"
_OUTPUT_ROOT = Path(
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_v7_x_face_rotation_stability_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.rollout_steps < 1:
        raise ValueError("rollout steps must be positive")
    repository = REPOSITORY.resolve()
    base_path = repository / _BASE_PROTOCOL
    protocol = load_order9_r1_calibration_protocol(
        base_path,
        repository_root=repository,
    )
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=base_path,
        repository_root=repository,
        level_id="r1_l1_10mm_5deg",
        split="train",
    )
    case = next(value for value in cases if value.candidate_id == _SOURCE_CANDIDATE_ID)
    if (
        case.support_audit.minimum_projected_com_margin_m
        < protocol.minimum_support_projected_com_margin_m
    ):
        raise RuntimeError("diagnostic case failed the support gate")

    destination = repository / _OUTPUT_ROOT / case.candidate_id
    if destination.exists():
        raise FileExistsError(f"diagnostic output already exists: {destination}")
    pipeline = Order9R1RotationalTeacherScreenPipelineV7(
        repository_root=repository,
        source_bucket_manifest_path=(repository / protocol.source_bucket_manifest.path),
        minimum_normalized_joint_limit_reserve=(
            protocol.minimum_normalized_joint_limit_reserve
        ),
        maximum_body_tilt_rad=protocol.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
    )
    prepared = pipeline.prepare(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete:
        audit = pipeline.rotation_audits.get(case.candidate_id)
        if audit is not None:
            diagnostic = write_order9_r1_clearance_diagnostic(
                {
                    **audit,
                    "result_version": "order9_r1_grasp_rotation_cpu_rejection_v1",
                    "failure_reason": prepared.failure_reason,
                    "training_eligible": False,
                },
                repository
                / _OUTPUT_ROOT
                / "cpu_rejections"
                / f"{case.candidate_id}.json",
            )
            raise RuntimeError(
                "v7 teacher rotation admission failed; evidence: " f"{diagnostic}"
            )
        raise RuntimeError(f"v7 teacher preparation failed: {prepared.failure_reason}")
    screen = pipeline.screen(case, prepared)
    if not screen.accepted:
        raise RuntimeError(
            "v7 exact-tracking screen failed: " + ",".join(screen.violation_codes)
        )

    try:
        materialized = materialize_order9_r1_isaac_case(
            case=case,
            prepared=prepared,
            screen=screen,
            output_dir=destination,
            repository_root=repository,
            approved_protocol_path=base_path,
        )
        materialized = finalize_order9_r1_v7_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=repository,
        )
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise

    artifact_path = repository / materialized.manifest.nominal_artifact.path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    contract_path = write_order9_r1_clearance_diagnostic(
        {
            "contract_version": "order9_r1_grasp_rotation_repair_diagnostic_v1",
            "candidate_id": case.candidate_id,
            "source_bucket_id": case.source_bucket.bucket_id,
            "module_count": case.source_bucket.module_count,
            "selected_surface_port_ids": list(artifact.selected_surface_port_ids),
            "selected_candidate_group_id": artifact.selected_candidate_group_id,
            "replay_count": 2,
            "pi_l_system_retained": True,
            "pi_l_actor_command_applied": False,
            "nominal_preload_applied": True,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "isaac_required": True,
            "training_eligible": False,
            "formal_calibration_result": False,
            "bindings": {
                "base_protocol": _binding(base_path, repository),
                "rotation_overrides": _binding(
                    repository / ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE,
                    repository,
                ),
                "rotation_gate": _binding(
                    repository / "amsrr/training/order9_r1_grasp_rotation.py",
                    repository,
                ),
                "teacher_pipeline": _binding(
                    repository
                    / "amsrr/training/order9_r1_rotational_teacher_pipeline_v7.py",
                    repository,
                ),
                "materializer": _binding(
                    repository
                    / "amsrr/training/order9_r1_complete_task_materialization_v7.py",
                    repository,
                ),
                "rotation_audit": _binding(
                    destination / ORDER9_R1_GRASP_ROTATION_AUDIT_FILENAME,
                    repository,
                ),
                "nominal_artifact": _binding(artifact_path, repository),
            },
        },
        destination / "grasp_rotation_diagnostic_contract.json",
    )
    results_by_case = run_order9_r1_nominal_v3_isaac_cases(
        [materialized],
        repository_root=repository,
        python_executable=args.isaac_python,
        rollout_steps=args.rollout_steps,
        maximum_parallel_process_count=1,
        persistent_morphology_coalescing=False,
    )
    results = results_by_case[case.candidate_id]
    accepted = len(results) == 2 and all(
        value.success and not value.safety_failure and not value.fallback_used
        for value in results
    )
    result_path = write_order9_r1_clearance_diagnostic(
        {
            "result_version": "order9_r1_grasp_rotation_repair_result_v1",
            "status": "accepted" if accepted else "rejected",
            "candidate_id": case.candidate_id,
            "episode_count": len(results),
            "success_count": sum(value.success for value in results),
            "safety_failure_count": sum(value.safety_failure for value in results),
            "fallback_count": sum(value.fallback_used for value in results),
            "replays": [asdict(value) for value in results],
            "pi_l_actor_command_applied": False,
            "training_eligible": False,
            "formal_calibration_result": False,
            "bindings": {
                "diagnostic_contract": _binding(contract_path, repository),
                "base_v3_result": _binding(
                    destination / "nominal_result_v3.json", repository
                ),
                "raw_rollout": _binding(
                    destination / "isaac/evaluation_rollout.pt", repository
                ),
                "episodes": _binding(
                    destination / "isaac/evaluation_episodes.jsonl", repository
                ),
            },
        },
        destination / "grasp_rotation_diagnostic_result.json",
    )
    print(f"ORDER9_R1_GRASP_ROTATION_RESULT={result_path}", flush=True)
    return 0 if accepted else 2


def _binding(path: Path, repository: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(repository).as_posix(),
        "sha256": hash_file(resolved),
    }


if __name__ == "__main__":
    raise SystemExit(main())
