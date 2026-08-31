#!/usr/bin/env python3
from __future__ import annotations

"""Materialize one hash-bound, CPU-confirmed R1 v13 teacher option."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

for _thread_environment_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_calibration import load_order9_r1_calibration_protocol
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v13 import (
    Order9R1ExactMarginTeacherScreenPipelineV13,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
)
from scripts import order9_run_r1_range_selection_v13 as runtime_v13


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-id", default="r1_l2_20mm_10deg")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--candidate-id", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "R1 v13 confirmed-option materialization requires PYTHONHASHSEED=0"
        )
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    case = next(
        case
        for case in runtime_v13.runtime_v11._corrected_cases(
            arguments.level_id, arguments.split
        )
        if case.candidate_id == arguments.candidate_id
    )
    base = load_order9_r1_calibration_protocol(
        runtime_v13.BASE_PROTOCOL,
        repository_root=REPOSITORY,
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    pipeline = Order9R1ExactMarginTeacherScreenPipelineV13(
        repository_root=REPOSITORY,
        source_bucket_manifest_path=REPOSITORY / base.source_bucket_manifest.path,
        minimum_normalized_joint_limit_reserve=(
            base.minimum_normalized_joint_limit_reserve
        ),
        maximum_body_tilt_rad=base.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
        geometry_contract=geometry,
        clearance_contract=clearance,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
    )
    option = pipeline.confirmed_teacher_option_overrides.get(case.candidate_id)
    if option is None:
        raise RuntimeError("candidate lacks a hash-bound confirmed teacher option")
    pipeline.teacher_fast_paths[case.candidate_id] = (option,)
    pipeline.teacher_candidate_overrides[case.candidate_id] = (option,)
    pipeline.strict_preferred_candidate_options = True

    started = time.time()
    prepared = pipeline.prepare(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete:
        raise RuntimeError(str(prepared.failure_reason))
    screen = pipeline.screen(case, prepared)
    if not screen.accepted or not screen.eligible_for_full_control_test:
        raise RuntimeError(
            "confirmed teacher option failed CPU screen: "
            + ",".join(screen.violation_codes)
        )

    destination = (
        runtime_v13.OUTPUT_ROOT
        / "prepared"
        / case.level_id
        / case.candidate_id
    )
    if destination.exists():
        runtime_v13._quarantine_preparation_destination(
            destination,
            reason="superseded_confirmed_option_materialization",
        )
    materialized = runtime_v13.materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=destination,
        repository_root=REPOSITORY,
        approved_protocol_path=runtime_v13.BASE_PROTOCOL,
    )
    certificate = pipeline.clearance_generation_certificates.get(case.candidate_id)
    runtime_v13.finalize_order9_r1_v14_materialized_case(
        materialized,
        task_spec=case.task_spec,
        phase_time_scales=runtime_v13.runtime_v11.order9_r1_safe_phase_time_scales(
            case.source_bucket.module_count
        ),
        joint_rate_limit_rad_s=(
            float(screen.maximum_joint_rate_rad_s)
            + float(screen.minimum_joint_rate_margin_rad_s)
        ),
        repository_root=REPOSITORY,
        geometry_contract=geometry,
        clearance_contract=clearance,
        clearance_generation_certificate=certificate,
    )
    seed_evidence = pipeline.configuration_goal_seed_evidence.get(case.candidate_id)
    if seed_evidence is not None:
        runtime_v13.runtime_v10._atomic_json(
            destination / "configuration_goal_seed_evidence_v1.json",
            seed_evidence,
        )
    for filename, values in (
        (
            "automatic_rotation_contact_retry_v1.json",
            pipeline.automatic_rotation_contact_retries,
        ),
        (
            "automatic_lightweight_contact_retry_v1.json",
            pipeline.automatic_lightweight_contact_retries,
        ),
    ):
        evidence = values.get(case.candidate_id)
        if evidence is not None:
            runtime_v13.runtime_v10._atomic_json(destination / filename, evidence)
    record = runtime_v13._validate_prepared_case(case, destination)
    print(
        "ORDER9_R1_V13_CONFIRMED_OPTION_MATERIALIZED="
        + json.dumps(
            {
                "candidate_id": case.candidate_id,
                "elapsed_s": time.time() - started,
                "prepared": bool(record["prepared"]),
                "teacher_feasible": bool(record["teacher_feasible"]),
                "fast_screen_passed": bool(record["fast_screen_passed"]),
                "pre_isaac_geometry_passed": bool(
                    record["pre_isaac_geometry_passed"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
