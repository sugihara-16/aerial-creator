#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight-only diagnosis of one R1 rotation-stable gripper pair."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_rotation_stability_repair import (  # noqa: E402
    Order9R1RotationStabilityRepairPipeline,
)

BASE_PROTOCOL = Path("configs/training/order9_r1_calibration_protocol_v1.yaml")
DEFAULT_CANDIDATE_ID = "r1_l1_10mm_5deg__train__train-000003-3b95da871f01__lattice_09"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--surface-ports", required=True, nargs=2, type=int)
    parser.add_argument("--candidate-group-id", default="slot_0:grasp_pair:0")
    parser.add_argument("--grasp-contact-height-offset-m", type=float, default=0.0)
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
    case = next(value for value in cases if value.candidate_id == args.candidate_id)
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
    source_id = case.source_bucket.bucket_id
    rule = pipeline.rotation_stability_rules[source_id]
    pipeline.rotation_stability_rules[source_id] = replace(
        rule,
        selected_surface_port_ids=tuple(args.surface_ports),
        preferred_candidate_group_ids=(args.candidate_group_id,),
        grasp_contact_height_offset_m=args.grasp_contact_height_offset_m,
        reason="lightweight_surface_pair_diagnosis",
    )
    prepared = pipeline.prepare(case)
    payload = {
        "candidate_id": case.candidate_id,
        "source_bucket_id": source_id,
        "requested_surface_port_ids": args.surface_ports,
        "requested_candidate_group_id": args.candidate_group_id,
        "grasp_contact_height_offset_m": args.grasp_contact_height_offset_m,
        "teacher_trajectory_complete": prepared.teacher_trajectory_complete,
        "failure_reason": prepared.failure_reason,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "training_eligible": False,
    }
    if prepared.teacher_trajectory_complete:
        screen = pipeline.screen(case, prepared)
        selection = prepared.payload.selection_bundle
        payload.update(
            {
                "selected_surface_port_ids": list(selection.selected_surface_port_ids),
                "selected_candidate_group_id": (
                    selection.trajectory_plan.candidate_group_id
                ),
                "lightweight_screen_accepted": screen.accepted,
                "violation_codes": list(screen.violation_codes),
                "minimum_collision_clearance_m": (screen.minimum_collision_clearance_m),
                "minimum_normalized_joint_limit_reserve": (
                    screen.minimum_normalized_joint_limit_reserve
                ),
                "maximum_body_tilt_rad": screen.maximum_body_tilt_rad,
                "rotation_stability_audit": pipeline.rotation_audits.get(
                    case.candidate_id
                ),
            }
        )
    print("ORDER9_R1_ROTATION_PAIR_DIAGNOSIS=" + json.dumps(payload, sort_keys=True))
    return 0 if payload.get("lightweight_screen_accepted") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
