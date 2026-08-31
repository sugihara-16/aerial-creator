#!/usr/bin/env python3
from __future__ import annotations

"""Prepare missing validation cases from hash-bound lower-level solutions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

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
from amsrr.utils.hashing import hash_file, stable_hash
from scripts import order9_run_r1_range_selection_v13 as runtime_v13

LEVEL = "r1_l2_20mm_10deg"
LOWER_LEVEL = "r1_l1_10mm_5deg"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-process-count", type=int, default=12)
    parser.add_argument("--maximum-source-attempts", type=int, default=4)
    parser.add_argument("--candidate-id", action="append", default=[])
    return parser


def _pipeline():
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
    return (
        Order9R1ExactMarginTeacherScreenPipelineV13(
            repository_root=REPOSITORY,
            source_bucket_manifest_path=REPOSITORY
            / base.source_bucket_manifest.path,
            minimum_normalized_joint_limit_reserve=(
                base.minimum_normalized_joint_limit_reserve
            ),
            maximum_body_tilt_rad=base.maximum_body_tilt_rad,
            anchor_position_tolerance_m=0.030,
            enforce_joint_limit_reserve_during_ik=True,
            geometry_contract=geometry,
            clearance_contract=clearance,
            physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
        ),
        geometry,
        clearance,
    )


def _lower_sources(case):
    lower_cases = [
        value
        for value in runtime_v13.runtime_v11._corrected_cases(
            LOWER_LEVEL, "validation"
        )
        if value.source_bucket.bucket_id == case.source_bucket.bucket_id
    ]
    lower_root = runtime_v13.OUTPUT_ROOT / "prepared" / LOWER_LEVEL
    current_cases = [
        value
        for value in runtime_v13.runtime_v11._corrected_cases(LEVEL, "validation")
        if value.source_bucket.bucket_id == case.source_bucket.bucket_id
        and value.candidate_id != case.candidate_id
    ]
    current_root = runtime_v13.OUTPUT_ROOT / "prepared" / LEVEL

    def distance(value):
        kind_penalty = 0.0 if value.sample_kind == case.sample_kind else 4.0
        exact_penalty = (
            -100.0
            if value.sample_kind == case.sample_kind
            and value.sample_index == case.sample_index
            else 0.0
        )
        return (
            exact_penalty
            + kind_penalty
            + (value.x_offset_m / 0.010 - case.x_offset_m / 0.020) ** 2
            + (value.y_offset_m / 0.010 - case.y_offset_m / 0.020) ** 2
            + (
                value.yaw_offset_rad / math.radians(5.0)
                - case.yaw_offset_rad / math.radians(10.0)
            )
            ** 2
        )

    sources = [
        (value, current_root / value.candidate_id, 0.0)
        for value in current_cases
        if (current_root / value.candidate_id / "case_manifest.json").is_file()
    ]
    sources.extend(
        (value, lower_root / value.candidate_id, 0.25)
        for value in lower_cases
        if (lower_root / value.candidate_id / "case_manifest.json").is_file()
    )
    sources.sort(key=lambda value: distance(value[0]) + value[2])
    return [(value, root) for value, root, _penalty in sources]


def _source_option(source_case, source_root: Path):
    case_manifest = json.loads(
        (source_root / "case_manifest.json").read_text(encoding="utf-8")
    )
    nominal_manifest_path = REPOSITORY / case_manifest["nominal_artifact"]["path"]
    nominal_manifest = json.loads(nominal_manifest_path.read_text(encoding="utf-8"))
    trajectories = sorted(
        nominal_manifest_path.parent.glob("windows/*/resolved_nominal_trajectory.json")
    )
    if not trajectories:
        raise RuntimeError("lower-level source lacks resolved trajectories")
    trajectory_path = trajectories[-1]
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    knots = trajectory.get("knots")
    posture = knots[-1].get("posture_target") if knots else None
    positions = posture.get("joint_pos_target") if isinstance(posture, dict) else None
    if not isinstance(positions, dict) or not positions:
        raise RuntimeError("lower-level source lacks a configuration seed")
    option = (
        tuple(int(value) for value in nominal_manifest["selected_surface_port_ids"]),
        str(nominal_manifest["selected_candidate_group_id"]),
        0.12,
        0.007,
        float(source_case.task_spec.metadata["r1_nominal_grasp_contact_height_offset_m"]),
        (0.0, 0.0, 0.0),
    )
    return (
        option,
        {str(key): float(value) for key, value in positions.items()},
        trajectory_path,
        nominal_manifest_path,
    )


def _prepare_one(arguments):
    candidate_id, maximum_source_attempts = arguments
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    case = next(
        value
        for value in runtime_v13.runtime_v11._corrected_cases(LEVEL, "validation")
        if value.candidate_id == candidate_id
    )
    destination = runtime_v13.OUTPUT_ROOT / "prepared" / LEVEL / candidate_id
    if destination.is_dir():
        try:
            record = runtime_v13._validate_prepared_case(case, destination)
            return {"candidate_id": candidate_id, "prepared": True, "reused": True}
        except Exception:
            runtime_v13._quarantine_preparation_destination(
                destination,
                reason="superseded_lower_level_initialization",
            )

    failures = []
    started = time.time()
    attempted_options = set()
    for source_case, source_root in _lower_sources(case):
        try:
            option, seed, trajectory_path, nominal_manifest_path = _source_option(
                source_case, source_root
            )
            option_key = (option[0], option[1], option[4], option[5])
            if option_key in attempted_options:
                continue
            if len(attempted_options) >= maximum_source_attempts:
                break
            attempted_options.add(option_key)
            pipeline, geometry, clearance = _pipeline()
            pipeline.teacher_fast_paths[candidate_id] = (option,)
            pipeline.teacher_candidate_overrides[candidate_id] = (option,)
            pipeline.strict_preferred_candidate_options = True
            pipeline.configuration_goal_seeds[candidate_id] = seed
            prepared = pipeline.prepare(case)
            prepared.validate_for(case)
            if not prepared.teacher_trajectory_complete:
                failures.append(
                    f"{source_case.candidate_id}:teacher:{prepared.failure_reason}"
                )
                continue
            screen = pipeline.screen(case, prepared)
            if not screen.accepted or not screen.eligible_for_full_control_test:
                failures.append(
                    f"{source_case.candidate_id}:screen:"
                    + ",".join(screen.violation_codes)
                )
                continue
            materialized = runtime_v13.materialize_order9_r1_isaac_case(
                case=case,
                prepared=prepared,
                screen=screen,
                output_dir=destination,
                repository_root=REPOSITORY,
                approved_protocol_path=runtime_v13.BASE_PROTOCOL,
            )
            certificate = pipeline.clearance_generation_certificates.get(candidate_id)
            runtime_v13.finalize_order9_r1_v14_materialized_case(
                materialized,
                task_spec=case.task_spec,
                phase_time_scales=(
                    runtime_v13.runtime_v11.order9_r1_safe_phase_time_scales(
                        case.source_bucket.module_count
                    )
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
            provenance = {
                "record_version": (
                    "order9_r1_lower_level_initialization_v1"
                    if source_case.level_id == LOWER_LEVEL
                    else "order9_r1_same_level_initialization_v1"
                ),
                "candidate_id": candidate_id,
                "source_candidate_id": source_case.candidate_id,
                "scope": "configuration_space_initialization_and_teacher_option_only",
                "source_trajectory": {
                    "path": str(trajectory_path.relative_to(REPOSITORY)),
                    "sha256": hash_file(trajectory_path),
                },
                "source_nominal_manifest": {
                    "path": str(nominal_manifest_path.relative_to(REPOSITORY)),
                    "sha256": hash_file(nominal_manifest_path),
                },
                "configuration_seed_hash": stable_hash(seed),
                "teacher_option": {
                    "selected_surface_port_ids": list(option[0]),
                    "candidate_group_id": option[1],
                    "pregrasp_clearance_m": option[2],
                    "collision_margin_m": option[3],
                    "grasp_contact_height_offset_m": option[4],
                    "tangent_offset_world_m": list(option[5]),
                },
                "teacher_trajectory_complete": True,
                "screen_accepted": True,
                "eligible_for_full_control_test": True,
                "isaac_invoked": False,
                "controller_layers_invoked": False,
                "acceptance_or_safety_gate_changed": False,
            }
            runtime_v13.runtime_v10._atomic_json(
                destination / "lower_level_initialization_v1.json", provenance
            )
            runtime_v13._validate_prepared_case(case, destination)
            return {
                "candidate_id": candidate_id,
                "prepared": True,
                "reused": False,
                "source_candidate_id": source_case.candidate_id,
                "elapsed_s": time.time() - started,
            }
        except Exception as error:
            if destination.exists():
                runtime_v13._quarantine_preparation_destination(
                    destination,
                    reason="failed_lower_level_initialization",
                )
            failures.append(
                f"{source_case.candidate_id}:{type(error).__name__}:{error}"
            )
    return {
        "candidate_id": candidate_id,
        "prepared": False,
        "elapsed_s": time.time() - started,
        "failures": failures,
    }


def main() -> int:
    arguments = _parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("lower-level initialization requires PYTHONHASHSEED=0")
    if not 1 <= arguments.maximum_process_count <= 12:
        raise ValueError("maximum process count differs")
    if not 1 <= arguments.maximum_source_attempts <= 8:
        raise ValueError("maximum source attempts differs")
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    cases = list(runtime_v13.runtime_v11._corrected_cases(LEVEL, "validation"))
    selected = set(arguments.candidate_id)
    if selected and selected.difference(case.candidate_id for case in cases):
        raise ValueError("unknown candidate id")
    pending = []
    for case in cases:
        if selected and case.candidate_id not in selected:
            continue
        destination = runtime_v13.OUTPUT_ROOT / "prepared" / LEVEL / case.candidate_id
        try:
            runtime_v13._validate_prepared_case(case, destination)
        except Exception:
            pending.append(case.candidate_id)
    results = []
    with ProcessPoolExecutor(
        max_workers=min(arguments.maximum_process_count, len(pending) or 1),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                _prepare_one, (candidate_id, arguments.maximum_source_attempts)
            ): candidate_id
            for candidate_id in pending
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                "ORDER9_R1_V13_LOWER_LEVEL_PREPARATION_PROGRESS="
                + json.dumps(result, sort_keys=True),
                flush=True,
            )
    failed = [value for value in results if not value["prepared"]]
    print(
        "ORDER9_R1_V13_LOWER_LEVEL_PREPARATION_RESULT="
        + json.dumps(
            {
                "pending_count": len(pending),
                "prepared_count": len(results) - len(failed),
                "failed_count": len(failed),
                "failed_candidate_ids": [value["candidate_id"] for value in failed],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
