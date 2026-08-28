#!/usr/bin/env python3
from __future__ import annotations

"""Prepare one lightweight-admitted R1 grasp-rotation repair for Isaac."""

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
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
from amsrr.training.order9_r1_complete_task_materialization_v6 import (  # noqa: E402
    finalize_order9_r1_v6_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_rotation_stability_repair import (  # noqa: E402
    ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE,
    ORDER9_R1_ROTATION_STABILITY_REPAIR_VERSION,
    Order9R1RotationStabilityRepairPipeline,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

BASE_PROTOCOL = Path("configs/training/order9_r1_calibration_protocol_v1.yaml")
DEFAULT_CANDIDATE_ID = "r1_l1_10mm_5deg__train__train-000003-3b95da871f01__lattice_09"
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_v8_train000003_rotation_stability_repair_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=10000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = REPOSITORY.resolve()
    protocol_path = repository / BASE_PROTOCOL
    protocol = load_order9_r1_calibration_protocol(
        protocol_path,
        repository_root=repository,
    )
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=protocol_path,
        repository_root=repository,
        level_id="r1_l1_10mm_5deg",
        split="train",
    )
    matching = [value for value in cases if value.candidate_id == args.candidate_id]
    if len(matching) != 1:
        raise SchemaValidationError("R1 rotation repair candidate is not unique")
    case = matching[0]
    destination = (repository / args.output_root / case.candidate_id).resolve()
    if destination.exists():
        raise FileExistsError(destination)

    pipeline = Order9R1RotationStabilityRepairPipeline(
        repository_root=repository,
        source_bucket_manifest_path=repository / protocol.source_bucket_manifest.path,
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
        raise SchemaValidationError(
            "R1 rotation repair teacher preparation failed: "
            + str(prepared.failure_reason)
        )
    screen = pipeline.screen(case, prepared)
    if not screen.accepted:
        raise SchemaValidationError(
            "R1 rotation repair lightweight screen failed: "
            + ",".join(screen.violation_codes)
        )
    rotation_audit = pipeline.rotation_audits.get(case.candidate_id)
    if (
        not isinstance(rotation_audit, dict)
        or rotation_audit.get("accepted") is not True
    ):
        raise SchemaValidationError("R1 rotation repair audit is missing")

    materialized = materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=destination,
        repository_root=repository,
        approved_protocol_path=protocol_path,
    )
    materialized = finalize_order9_r1_v6_materialized_case(
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
    artifact_path = repository / materialized.manifest.nominal_artifact.path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    if artifact.selected_candidate_group_id not in {
        "slot_0:grasp_pair:0",
        "slot_0:grasp_pair:1",
    }:
        raise SchemaValidationError("R1 rotation repair selected the wrong grasp")

    admission_path = write_order9_r1_clearance_diagnostic(
        {
            "admission_version": ORDER9_R1_ROTATION_STABILITY_REPAIR_VERSION,
            "status": "accepted",
            "candidate_id": case.candidate_id,
            "source_bucket_id": case.source_bucket.bucket_id,
            "selected_surface_port_ids": list(artifact.selected_surface_port_ids),
            "selected_candidate_group_id": artifact.selected_candidate_group_id,
            "minimum_collision_clearance_m": (screen.minimum_collision_clearance_m),
            "minimum_normalized_joint_limit_reserve": (
                screen.minimum_normalized_joint_limit_reserve
            ),
            "maximum_body_tilt_rad": screen.maximum_body_tilt_rad,
            "rotation_stability_audit": rotation_audit,
            "case_manifest_sha256": hash_file(materialized.manifest_path),
            "nominal_artifact_sha256": hash_file(artifact_path),
            "rotation_overrides_sha256": hash_file(
                repository / ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE
            ),
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "ik_invoked_during_screen": False,
            "trajectory_optimization_invoked_during_screen": False,
            "pi_l_actor_command_applied": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
        },
        destination / "rotation_stability_repair_admission.json",
    )

    command, log_path = order9_r1_nominal_case_command(
        materialized,
        repository_root=repository,
        python_executable=args.isaac_python,
        rollout_steps=args.rollout_steps,
    )
    if (
        len(command) < 3
        or Path(command[1]).name != "order9_vectorized_isaac_rollout.py"
    ):
        raise SchemaValidationError("R1 rotation repair Isaac command differs")
    jobs_path = write_order9_r1_clearance_diagnostic(
        {
            "manifest_version": "order9_r1_rotation_stability_repair_jobs_v1",
            "pi_l_actor_command_applied": False,
            "r1_nominal_additional_compression_sweep_mm": "0.0,0.0",
            "jobs": [
                {
                    "name": case.candidate_id,
                    "argv": command[2:],
                    "log_path": str(log_path),
                }
            ],
        },
        destination.parent / f"{case.candidate_id}_jobs.json",
    )
    print(
        "ORDER9_R1_ROTATION_REPAIR_PREPARED="
        + json.dumps(
            {
                "candidate_id": case.candidate_id,
                "case_manifest": str(materialized.manifest_path),
                "admission": str(admission_path),
                "jobs": str(jobs_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
