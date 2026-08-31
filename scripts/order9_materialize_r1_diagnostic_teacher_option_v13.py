#!/usr/bin/env python3
from __future__ import annotations

"""Materialize one CPU-screened R1 v13 option for diagnostic Isaac replay."""

import argparse
import json
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


DIAGNOSTIC_ROOT = (
    REPOSITORY
    / "artifacts/p4_full/order9/r1_teacher/range_selection_v13/diagnostics"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-id", default="r1_l2_20mm_10deg")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--tangent-world-m", default="0,0,0")
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.12)
    parser.add_argument("--collision-margin-m", type=float, default=0.007)
    parser.add_argument("--bounded-contact-search", action="store_true")
    parser.add_argument("--configuration-goal-seed-trajectory")
    parser.add_argument("--output-dir", required=True)
    return parser


def _tuple(value: str, *, count: int, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != count:
        raise ValueError(f"expected {count} comma-separated values")
    return result


def _configuration_seed(path: str | None) -> tuple[dict[str, float] | None, Path | None]:
    if path is None:
        return None, None
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    knots = payload.get("knots") if isinstance(payload, dict) else None
    posture = knots[-1].get("posture_target") if isinstance(knots, list) and knots else None
    positions = posture.get("joint_pos_target") if isinstance(posture, dict) else None
    if not isinstance(positions, dict) or not positions:
        raise ValueError("configuration seed lacks final joint targets")
    return {str(key): float(value) for key, value in positions.items()}, source


def main() -> int:
    args = _parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("diagnostic materialization requires PYTHONHASHSEED=0")
    destination = Path(args.output_dir).resolve()
    try:
        destination.relative_to(DIAGNOSTIC_ROOT)
    except ValueError as error:
        raise ValueError("diagnostic output must remain beneath the R1 diagnostic root") from error
    if destination.exists():
        raise FileExistsError(f"diagnostic output already exists: {destination}")

    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    case = next(
        case
        for case in runtime_v13.runtime_v11._corrected_cases(args.level_id, args.split)
        if case.candidate_id == args.candidate_id
    )
    base = load_order9_r1_calibration_protocol(
        runtime_v13.BASE_PROTOCOL, repository_root=REPOSITORY
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE, repository_root=REPOSITORY
    )
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    pipeline = Order9R1ExactMarginTeacherScreenPipelineV13(
        repository_root=REPOSITORY,
        source_bucket_manifest_path=REPOSITORY / base.source_bucket_manifest.path,
        minimum_normalized_joint_limit_reserve=base.minimum_normalized_joint_limit_reserve,
        maximum_body_tilt_rad=base.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
        geometry_contract=geometry,
        clearance_contract=clearance,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
    )
    option = (
        _tuple(args.ports, count=2, cast=int),
        args.group_id,
        float(args.pregrasp_clearance_m),
        float(args.collision_margin_m),
        float(args.height_m),
        _tuple(args.tangent_world_m, count=3, cast=float),
    )
    pipeline.teacher_fast_paths[case.candidate_id] = (option,)
    pipeline.teacher_candidate_overrides[case.candidate_id] = (option,)
    pipeline.strict_preferred_candidate_options = True
    if not args.bounded_contact_search:
        pipeline.bounded_local_contact_repairs = frozenset(
            candidate_id
            for candidate_id in pipeline.bounded_local_contact_repairs
            if candidate_id != case.candidate_id
        )
    seed, seed_path = _configuration_seed(args.configuration_goal_seed_trajectory)
    if seed is not None:
        pipeline.configuration_goal_seeds[case.candidate_id] = seed

    started = time.time()
    prepared = pipeline.prepare(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete:
        raise RuntimeError(str(prepared.failure_reason))
    screen = pipeline.screen(case, prepared)
    if not screen.accepted or not screen.eligible_for_full_control_test:
        raise RuntimeError("diagnostic teacher failed CPU screen: " + ",".join(screen.violation_codes))
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
    evidence = {
        "record_version": "order9_r1_diagnostic_teacher_option_materialization_v1",
        "candidate_id": case.candidate_id,
        "teacher_option": {
            "selected_surface_port_ids": list(option[0]),
            "candidate_group_id": option[1],
            "pregrasp_clearance_m": option[2],
            "collision_margin_m": option[3],
            "grasp_contact_height_offset_m": option[4],
            "tangent_offset_world_m": list(option[5]),
        },
        "configuration_goal_seed": (
            None
            if seed is None or seed_path is None
            else {
                "path": str(seed_path.relative_to(REPOSITORY)),
                "sha256": hash_file(seed_path),
                "value_hash": stable_hash(seed),
            }
        ),
        "teacher_trajectory_complete": True,
        "screen_accepted": True,
        "eligible_for_full_control_test": True,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "acceptance_or_safety_gate_changed": False,
        "elapsed_s": time.time() - started,
    }
    runtime_v13.runtime_v10._atomic_json(
        destination / "diagnostic_teacher_option_v1.json", evidence
    )
    runtime_v13._validate_prepared_case(case, destination)
    print("ORDER9_R1_DIAGNOSTIC_TEACHER_OPTION_MATERIALIZED=" + json.dumps(evidence, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
