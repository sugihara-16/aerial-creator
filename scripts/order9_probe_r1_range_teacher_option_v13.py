#!/usr/bin/env python3
from __future__ import annotations

"""Probe one R1 v13 contact option without writing formal evidence."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

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
from amsrr.training import order9_r1_early_tilt_pipeline as early_tilt
from amsrr.training import order9_articulated_teacher as articulated_teacher
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
)
from scripts import order9_run_r1_range_selection_v13 as runtime_v13


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-id", default="r1_l2_20mm_10deg")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--tangent-world-m", default="0,0,0")
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.12)
    parser.add_argument("--collision-margin-m", type=float, default=0.007)
    parser.add_argument("--bounded-contact-search", action="store_true")
    parser.add_argument(
        "--contact-goal-seed-trajectory",
        help=(
            "Resolved nominal trajectory whose final knot supplies only the "
            "deterministic contact-IK initial joint posture."
        ),
    )
    parser.add_argument(
        "--configuration-goal-seed-trajectory",
        help=(
            "Resolved nominal trajectory whose final knot supplies only the "
            "collision-free configuration-goal search initial posture."
        ),
    )
    parser.add_argument("--extended-centroidal-offsets", action="store_true")
    return parser


def _tuple(value: str, *, count: int, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != count:
        raise ValueError(f"expected {count} comma-separated values")
    return result


def _contact_goal_seed(path: str | None) -> dict[str, float] | None:
    if path is None:
        return None
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    knots = payload.get("knots") if isinstance(payload, dict) else None
    posture = (
        knots[-1].get("posture_target")
        if isinstance(knots, list) and knots and isinstance(knots[-1], dict)
        else None
    )
    values = posture.get("joint_pos_target") if isinstance(posture, dict) else None
    if (
        not isinstance(values, dict)
        or not values
        or any(not isinstance(key, str) or not isinstance(value, (int, float)) for key, value in values.items())
    ):
        raise ValueError("contact-goal seed trajectory lacks final joint targets")
    return {key: float(value) for key, value in values.items()}


def main() -> int:
    args = _parser().parse_args()
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    case = next(
        case
        for case in runtime_v13.runtime_v11._corrected_cases(
            args.level_id, "train"
        )
        if case.candidate_id == args.candidate_id
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
    contact_seed = _contact_goal_seed(args.contact_goal_seed_trajectory)
    configuration_seed = _contact_goal_seed(
        args.configuration_goal_seed_trajectory
    )
    original_generate = early_tilt.generate_order9_c3_nominal_grasp_trajectory

    def generate_with_seed(*values, **keywords):
        if contact_seed is not None:
            keywords["contact_goal_joint_seed_positions_rad"] = contact_seed
        if configuration_seed is not None:
            keywords["configuration_goal_joint_seed_positions_rad"] = (
                configuration_seed
            )
        return original_generate(*values, **keywords)

    started = time.time()
    offsets = articulated_teacher._CONFIGURATION_GOAL_CENTROIDAL_OFFSETS_M
    if args.extended_centroidal_offsets:
        offsets = (*offsets, *(
            (0.02, 0.02, 0.02),
            (0.02, -0.02, 0.02),
            (-0.02, 0.02, 0.02),
            (-0.02, -0.02, 0.02),
            (0.03, 0.0, 0.02),
            (-0.03, 0.0, 0.02),
            (0.0, 0.03, 0.02),
            (0.0, -0.03, 0.02),
        ))
    with patch.object(
        early_tilt,
        "generate_order9_c3_nominal_grasp_trajectory",
        generate_with_seed,
    ), patch.object(
        articulated_teacher,
        "_CONFIGURATION_GOAL_CENTROIDAL_OFFSETS_M",
        offsets,
    ):
        prepared = pipeline.prepare(case)
    result = {
        "candidate_id": case.candidate_id,
        "option": option,
        "bounded_contact_search": bool(args.bounded_contact_search),
        "contact_goal_seed_trajectory": args.contact_goal_seed_trajectory,
        "configuration_goal_seed_trajectory": (
            args.configuration_goal_seed_trajectory
        ),
        "extended_centroidal_offsets": bool(args.extended_centroidal_offsets),
        "teacher_trajectory_complete": bool(prepared.teacher_trajectory_complete),
        "failure_reason": prepared.failure_reason,
        "elapsed_s": time.time() - started,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
    }
    if prepared.teacher_trajectory_complete:
        screen = pipeline.screen(case, prepared)
        result.update(
            {
                "screen_accepted": bool(screen.accepted),
                "eligible_for_full_control_test": bool(
                    screen.eligible_for_full_control_test
                ),
                "violation_codes": list(screen.violation_codes),
            }
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("eligible_for_full_control_test") else 1


if __name__ == "__main__":
    raise SystemExit(main())
