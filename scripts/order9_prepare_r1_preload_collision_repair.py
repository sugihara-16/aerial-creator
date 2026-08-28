#!/usr/bin/env python3
from __future__ import annotations

"""Materialize collision-clear alternate R1 teachers before Isaac replay."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    load_order9_c3_accepted_nominal_bundle,
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
from amsrr.training.order9_r1_early_tilt_pipeline import (  # noqa: E402
    Order9R1EarlyTiltTeacherScreenPipeline,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_nominal_preload_admission import (  # noqa: E402
    Order9R1NominalPreloadAdmissionConfig,
    admit_order9_r1_nominal_preload,
)
from amsrr.training.order9_r1_preload_collision_repair import (  # noqa: E402
    ORDER9_R1_PRELOAD_COLLISION_REPAIR_VERSION,
    load_order9_r1_preload_collision_repair_rules,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.training.order9_curriculum import (  # noqa: E402
    load_order9_learning_config,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

BASE_PROTOCOL = Path("configs/training/order9_r1_calibration_protocol_v1.yaml")
DEFAULT_REPAIR_CONFIG = Path(
    "configs/training/order9_r1_preload_collision_repair_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_v10_preload_collision_repair_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-config", default=str(DEFAULT_REPAIR_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--candidate-id",
        action="append",
        help="Prepare only the named bound candidate; repeat to select several.",
    )
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.rollout_steps < 1:
        raise ValueError("R1 preload-collision rollout steps must be positive")
    repository = REPOSITORY.resolve()
    protocol_path = repository / BASE_PROTOCOL
    protocol = load_order9_r1_calibration_protocol(
        protocol_path,
        repository_root=repository,
    )
    config_path = (repository / args.repair_config).resolve()
    reason, rules = load_order9_r1_preload_collision_repair_rules(config_path)
    if args.candidate_id:
        selected_ids = tuple(args.candidate_id)
        if (
            len(selected_ids) != len(set(selected_ids))
            or any(candidate_id not in rules for candidate_id in selected_ids)
        ):
            raise SchemaValidationError(
                "R1 preload-collision selected candidate set is invalid"
            )
        rules = {candidate_id: rules[candidate_id] for candidate_id in selected_ids}
    output_root = (repository / args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    all_cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=protocol_path,
        repository_root=repository,
        level_id="r1_l1_10mm_5deg",
        split="train",
    )
    cases = {case.candidate_id: case for case in all_cases}
    if any(candidate_id not in cases for candidate_id in rules):
        raise SchemaValidationError("R1 preload-collision repair candidate is absent")

    pipeline = Order9R1EarlyTiltTeacherScreenPipeline(
        repository_root=repository,
        source_bucket_manifest_path=repository
        / protocol.source_bucket_manifest.path,
        minimum_normalized_joint_limit_reserve=(
            protocol.minimum_normalized_joint_limit_reserve
        ),
        maximum_body_tilt_rad=protocol.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
        strict_preferred_candidate_options=True,
    )
    learning = load_order9_learning_config(
        repository / "configs/training/order9_learning_curriculum.yaml"
    )
    runtime = learning.production_runtime
    preload_config = Order9R1NominalPreloadAdmissionConfig(
        minimum_inward_lead_m=float(runtime.c3_virtual_contact_inward_lead_m),
        maximum_inward_lead_m=float(
            runtime.c3_actuator_aware_nominal_preload_maximum_m
        ),
        inward_lead_quantization_m=float(
            runtime.c3_actuator_aware_nominal_preload_quantization_m
        ),
        support_safety_factor=float(
            runtime.c3_actuator_aware_nominal_preload_support_safety_factor
        ),
        maximum_peak_effort_utilization=float(
            runtime.c3_actuator_aware_nominal_preload_maximum_peak_utilization
        ),
        minimum_anchor_force_fraction=float(
            runtime.c3_actuator_aware_nominal_preload_minimum_anchor_fraction
        ),
        model_error_margin_m=float(
            runtime.c3_actuator_aware_nominal_preload_model_error_margin_m
        ),
    )
    jobs = []
    entries = []
    for candidate_id, rule in sorted(rules.items()):
        case = cases[candidate_id]
        attempt_failures = []
        destination = output_root / candidate_id
        selected = None
        for option_index, option in enumerate(rule.teacher_options):
            pipeline.teacher_candidate_overrides[candidate_id] = (option,)
            prepared = pipeline.prepare(case)
            prepared.validate_for(case)
            if not prepared.teacher_trajectory_complete:
                attempt_failures.append(
                    f"option_{option_index}:teacher:{prepared.failure_reason}"
                )
                continue
            selection = prepared.payload.selection_bundle
            if (
                tuple(selection.selected_surface_port_ids) != option[0]
                or selection.trajectory_plan.candidate_group_id != option[1]
            ):
                raise SchemaValidationError(
                    "R1 preload-collision repair did not select the bound contact"
                )
            screen = pipeline.screen(case, prepared)
            if not screen.accepted:
                attempt_failures.append(
                    f"option_{option_index}:screen:"
                    + ",".join(screen.violation_codes)
                )
                continue
            try:
                preliminary_preload_admission = admit_order9_r1_nominal_preload(
                    nominal=prepared.payload,
                    task_spec=case.task_spec,
                    physical_model=pipeline.physical_model,
                    contact_friction=float(
                        case.source_bucket.selected_gripper_friction
                    ),
                    contact_stiffness_n_per_m=float(
                        case.source_bucket.contact_stiffness_n_per_m
                    ),
                    config=preload_config,
                )
            except (SchemaValidationError, RuntimeError, ValueError) as error:
                attempt_failures.append(
                    f"option_{option_index}:nominal_preload:{error}"
                )
                continue
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
            try:
                persisted = load_order9_c3_accepted_nominal_bundle(
                    repository / materialized.manifest.nominal_set.path,
                    bucket_id=candidate_id,
                    repository_root=repository,
                    expected_set_sha256=materialized.manifest.nominal_set.sha256,
                    expected_artifact_sha256=(
                        materialized.manifest.nominal_artifact.sha256
                    ),
                    expected_structural_hash=case.source_bucket.structural_hash,
                    expected_task_spec_sha256=hash_file(
                        repository / materialized.manifest.task_spec.path
                    ),
                    expected_physical_model_hash=pipeline.physical_model.stable_hash(),
                )
                preload_admission = admit_order9_r1_nominal_preload(
                    nominal=persisted,
                    task_spec=case.task_spec,
                    physical_model=pipeline.physical_model,
                    contact_friction=float(
                        case.source_bucket.selected_gripper_friction
                    ),
                    contact_stiffness_n_per_m=float(
                        case.source_bucket.contact_stiffness_n_per_m
                    ),
                    config=preload_config,
                )
            except (SchemaValidationError, RuntimeError, ValueError) as error:
                rejected = (
                    output_root
                    / "rejected_materialized_options"
                    / candidate_id
                    / f"option_{option_index}"
                )
                rejected.parent.mkdir(parents=True, exist_ok=True)
                if rejected.exists():
                    raise FileExistsError(rejected)
                os.rename(destination, rejected)
                attempt_failures.append(
                    f"option_{option_index}:persisted_nominal_preload:{error}"
                )
                continue
            selected = (
                option_index,
                prepared,
                selection,
                screen,
                preliminary_preload_admission,
                preload_admission,
                materialized,
            )
            break
        if selected is None:
            raise SchemaValidationError(
                "R1 preload-collision alternate teacher failed: "
                + candidate_id
                + ":"
                + "|".join(attempt_failures)
            )
        (
            selected_option_index,
            prepared,
            selection,
            screen,
            preliminary_preload_admission,
            preload_admission,
            materialized,
        ) = selected
        artifact_path = repository / materialized.manifest.nominal_artifact.path
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            artifact_path
        )
        if (
            tuple(artifact.selected_surface_port_ids)
            != tuple(selection.selected_surface_port_ids)
            or artifact.selected_candidate_group_id
            != selection.trajectory_plan.candidate_group_id
        ):
            raise SchemaValidationError(
                "R1 preload-collision materialized contact differs"
            )
        command, log_path = order9_r1_nominal_case_command(
            materialized,
            repository_root=repository,
            python_executable=args.isaac_python,
            rollout_steps=args.rollout_steps,
        )
        if len(command) < 3 or Path(command[1]).name != (
            "order9_vectorized_isaac_rollout.py"
        ):
            raise SchemaValidationError("R1 preload-collision Isaac command differs")
        jobs.append(
            {
                "name": candidate_id,
                "argv": command[2:],
                "log_path": str(log_path),
            }
        )
        entries.append(
            {
                "candidate_id": candidate_id,
                "source_bucket_id": case.source_bucket.bucket_id,
                "candidate_task_hash": case.task_spec.stable_hash(),
                "selected_surface_port_ids": list(
                    selection.selected_surface_port_ids
                ),
                "selected_candidate_group_id": (
                    selection.trajectory_plan.candidate_group_id
                ),
                "selected_repair_option_index": selected_option_index,
                "minimum_collision_clearance_m": (
                    screen.minimum_collision_clearance_m
                ),
                "minimum_normalized_joint_limit_reserve": (
                    screen.minimum_normalized_joint_limit_reserve
                ),
                "maximum_body_tilt_rad": screen.maximum_body_tilt_rad,
                "nominal_preload_admission": preload_admission.to_dict(),
                "pre_materialization_nominal_preload_admission": (
                    preliminary_preload_admission.to_dict()
                ),
                "case_manifest": {
                    "path": str(materialized.manifest_path.relative_to(repository)),
                    "sha256": hash_file(materialized.manifest_path),
                },
                "nominal_artifact": {
                    "path": str(artifact_path.relative_to(repository)),
                    "sha256": hash_file(artifact_path),
                },
            }
        )

    jobs_path = write_order9_r1_clearance_diagnostic(
        {
            "manifest_version": "order9_r1_preload_collision_repair_jobs_v1",
            "repair_version": ORDER9_R1_PRELOAD_COLLISION_REPAIR_VERSION,
            "repair_config": {
                "path": str(config_path.relative_to(repository)),
                "sha256": hash_file(config_path),
            },
            "reason": reason,
            "candidate_count": len(jobs),
            "r1_nominal_additional_compression_sweep_mm": "0.0,0.0",
            "pi_l_actor_command_applied": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
            "jobs": jobs,
        },
        output_root / "jobs.json",
    )
    admission_path = write_order9_r1_clearance_diagnostic(
        {
            "admission_version": ORDER9_R1_PRELOAD_COLLISION_REPAIR_VERSION,
            "status": "accepted_before_isaac",
            "repair_config_sha256": hash_file(config_path),
            "candidate_count": len(entries),
            "teacher_and_ik_screen_passed": True,
            "nominal_preload_collision_admission_passed": True,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "pi_l_actor_command_applied": False,
            "nominal_preload_collision_admission_must_remain_active": True,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
            "entries": entries,
        },
        output_root / "admission.json",
    )
    print(
        "ORDER9_R1_PRELOAD_COLLISION_REPAIR_PREPARED="
        + json.dumps(
            {
                "candidate_count": len(entries),
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
