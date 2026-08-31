#!/usr/bin/env python3
from __future__ import annotations

"""CPU-only probe for one finite R1 range teacher contact option."""

import argparse
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_range_selection_v10 import (  # noqa: E402
    ORDER9_R1_RANGE_LEVEL_IDS,
    Order9R1RangeTeacherScreenPipelineV10,
)
from scripts.order9_run_r1_range_selection_v10 import (  # noqa: E402
    _ensure_python_hash_seed_zero,
)

BASE_PROTOCOL = REPOSITORY / "configs/training/order9_r1_calibration_protocol_v1.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-id", required=True, choices=ORDER9_R1_RANGE_LEVEL_IDS)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--ports", required=True, help="Two comma-separated port ids")
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--height-m", required=True, type=float)
    parser.add_argument(
        "--tangent-world-m",
        required=True,
        help="Three comma-separated world-frame offsets",
    )
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.12)
    parser.add_argument("--collision-margin-m", type=float, default=0.007)
    return parser


def _numbers(value: str, *, count: int, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != count:
        raise ValueError(f"expected {count} comma-separated values")
    return result


def main() -> int:
    _ensure_python_hash_seed_zero()
    args = _parser().parse_args()
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=BASE_PROTOCOL,
        repository_root=REPOSITORY,
        level_id=args.level_id,
        split="train",
    )
    matching = [case for case in cases if case.candidate_id == args.candidate_id]
    if len(matching) != 1:
        raise ValueError("candidate id does not resolve exactly once")
    base = load_order9_r1_calibration_protocol(
        BASE_PROTOCOL, repository_root=REPOSITORY
    )
    pipeline = Order9R1RangeTeacherScreenPipelineV10(
        repository_root=REPOSITORY,
        source_bucket_manifest_path=REPOSITORY / base.source_bucket_manifest.path,
        minimum_normalized_joint_limit_reserve=(
            base.minimum_normalized_joint_limit_reserve
        ),
        maximum_body_tilt_rad=base.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
    )
    option = (
        _numbers(args.ports, count=2, cast=int),
        args.group_id,
        float(args.pregrasp_clearance_m),
        float(args.collision_margin_m),
        float(args.height_m),
        _numbers(args.tangent_world_m, count=3, cast=float),
    )
    pipeline.teacher_fast_paths[args.candidate_id] = (option,)
    started = time.time()
    prepared = pipeline.prepare(matching[0])
    payload = {
        "candidate_id": args.candidate_id,
        "option": option,
        "teacher_trajectory_complete": prepared.teacher_trajectory_complete,
        "failure_reason": prepared.failure_reason,
        "elapsed_s": time.time() - started,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
    }
    if prepared.teacher_trajectory_complete:
        screen = pipeline.screen(matching[0], prepared)
        payload.update(
            {
                "screen_accepted": screen.accepted,
                "eligible_for_full_control_test": (
                    screen.eligible_for_full_control_test
                ),
                "violation_codes": list(screen.violation_codes),
            }
        )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload.get("eligible_for_full_control_test") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
