from __future__ import annotations

import json
import math
from pathlib import Path

from amsrr.geometry.pose_math import pose_to_xyz_rpy
from amsrr.schemas.task_spec import TaskSpec
from scripts.order9_run_r1_teacher_collection_v19 import build_target_identity_task

REPOSITORY = Path(__file__).resolve().parents[3]


def test_collection_target_identity_moves_only_requested_planar_object_pose() -> None:
    case = next(
        (
            REPOSITORY
            / "artifacts/p4_full/order9/r1_teacher/range_selection_v13/prepared/"
            "r1_l2_20mm_10deg"
        ).glob("*lattice_13/task_spec.json")
    )
    reference = TaskSpec.from_json(case.read_text(encoding="utf-8"))
    entry = {
        "task_id": "unit-r1-task",
        "episode_id": "unit-r1-episode",
        "split": "train",
        "seed": 29009,
        "pose_condition_hash": "1" * 64,
        "x_offset_m": 0.04,
        "y_offset_m": -0.02,
        "yaw_offset_rad": math.radians(20.0),
    }
    target = build_target_identity_task(
        reference, entry, fraction=0.5, candidate_id="unit-midpoint"
    )
    object_id = next(
        goal.target_entity_id
        for goal in reference.goals
        if goal.goal_type == "object_pose"
    )
    source_object = next(
        value for value in reference.scene.objects if value.object_id == object_id
    )
    target_object = next(
        value for value in target.scene.objects if value.object_id == object_id
    )
    source_xyz, source_rpy = pose_to_xyz_rpy(source_object.pose_world)
    target_xyz, target_rpy = pose_to_xyz_rpy(target_object.pose_world)
    assert math.isclose(target_xyz[0] - source_xyz[0], 0.02, abs_tol=1.0e-9)
    assert math.isclose(target_xyz[1] - source_xyz[1], -0.01, abs_tol=1.0e-9)
    assert math.isclose(target_xyz[2], source_xyz[2], abs_tol=1.0e-9)
    assert math.isclose(target_rpy[2] - source_rpy[2], math.radians(10), abs_tol=1.0e-9)
    assert target.metadata["r1_calibration_candidate_id"] == "unit-midpoint"
    assert target.metadata["order9_rollout_bucket_id"] == "unit-midpoint"
    assert target.metadata["r1_formal_teacher_collection_eligible"] is False


def test_collection_final_identity_uses_the_planned_task_and_dataset_split() -> None:
    case = next(
        (
            REPOSITORY
            / "artifacts/p4_full/order9/r1_teacher/range_selection_v13/prepared/"
            "r1_l2_20mm_10deg"
        ).glob("*lattice_13/task_spec.json")
    )
    reference = TaskSpec.from_json(case.read_text(encoding="utf-8"))
    entry = {
        "task_id": "unit-r1-final-task",
        "episode_id": "unit-r1-final-episode",
        "split": "held_out",
        "seed": 29010,
        "pose_condition_hash": "2" * 64,
        "x_offset_m": 0.01,
        "y_offset_m": 0.02,
        "yaw_offset_rad": math.radians(5.0),
    }
    target = build_target_identity_task(
        reference, entry, fraction=1.0, candidate_id="unit-final"
    )
    assert target.task_id == entry["task_id"]
    assert target.metadata["dataset_split"] == "held_out"
    assert target.metadata["order9_rollout_bucket_id"] == "unit-final"
