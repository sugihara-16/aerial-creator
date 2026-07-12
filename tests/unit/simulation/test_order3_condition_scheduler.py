from __future__ import annotations

import math

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order3_rollout_condition import build_order3_rollout_condition
from amsrr.simulation.order3_rollout_condition import (
    Order3ConditionTargetScheduler,
    order3_tracking_window_start_s,
    quaternion_multiply_xyzw,
    rpy_to_quat_xyzw,
)


IDENTITY_POSE = (1.0, -2.0, 1.5, 0.0, 0.0, 0.0, 1.0)


def test_waypoint_scheduler_ramps_translation_and_full_rpy() -> None:
    condition = build_order3_rollout_condition(
        stage_id="3c_randomized",
        task_mode="waypoint",
        seed=13,
        waypoint_position_offset_world=(0.3, -0.2, 0.1),
        waypoint_orientation_rpy_rad=(0.2, -0.1, 0.4),
        waypoint_ramp_s=2.0,
    )
    scheduler = Order3ConditionTargetScheduler(
        condition,
        nominal_hover_pose_world=IDENTITY_POSE,
    )

    start = scheduler.target_at(0.0)
    midpoint = scheduler.target_at(1.0)
    final = scheduler.target_at(2.0)

    assert start.desired_pose_world == IDENTITY_POSE
    assert midpoint.ramp_progress == pytest.approx(0.5)
    assert midpoint.desired_pose_world[:3] == pytest.approx((1.15, -2.1, 1.55))
    assert final.desired_pose_world[:3] == pytest.approx((1.3, -2.2, 1.6))
    assert final.desired_pose_world[3:7] == pytest.approx(
        rpy_to_quat_xyzw((0.2, -0.1, 0.4))
    )
    assert final.hold_active is True
    assert math.sqrt(sum(value * value for value in midpoint.desired_pose_world[3:7])) == pytest.approx(1.0)


def test_hover_scheduler_holds_unperturbed_reference() -> None:
    condition = build_order3_rollout_condition(
        stage_id="3b_hover",
        task_mode="hover",
        seed=4,
        initial_position_offset_world=(0.1, 0.0, -0.1),
        initial_orientation_rpy_rad=(0.1, 0.2, 0.3),
    )
    scheduler = Order3ConditionTargetScheduler(
        condition,
        nominal_hover_pose_world=IDENTITY_POSE,
    )
    assert scheduler.target_at(7.0).desired_pose_world == IDENTITY_POSE


def test_terminal_dwell_starts_after_waypoint_and_disturbance() -> None:
    disturbed = build_order3_rollout_condition(
        stage_id="disturbed_waypoint",
        task_mode="waypoint",
        seed=8,
        waypoint_ramp_s=1.5,
        external_wrench_body=(1.0, 0.0, 0.0, 0.0, 0.0, 0.1),
        disturbance_start_s=3.0,
        disturbance_duration_s=1.0,
    )
    scheduler = Order3ConditionTargetScheduler(
        disturbed,
        nominal_hover_pose_world=IDENTITY_POSE,
    )
    assert scheduler.terminal_evidence_start_s == pytest.approx(4.0)
    assert order3_tracking_window_start_s(disturbed) == pytest.approx(3.0)

    persistent = build_order3_rollout_condition(
        stage_id="persistent_hover",
        task_mode="hover",
        seed=9,
        external_wrench_body=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        disturbance_start_s=2.0,
        disturbance_duration_s=0.0,
    )
    persistent_scheduler = Order3ConditionTargetScheduler(
        persistent,
        nominal_hover_pose_world=IDENTITY_POSE,
    )
    assert persistent_scheduler.terminal_evidence_start_s == pytest.approx(2.0)
    assert order3_tracking_window_start_s(persistent) == pytest.approx(2.0)


def test_target_scheduler_rejects_takeoff_and_invalid_time() -> None:
    takeoff = build_order3_rollout_condition(
        stage_id="3d_takeoff",
        task_mode="takeoff",
        seed=0,
    )
    with pytest.raises(SchemaValidationError, match="hover/waypoint"):
        Order3ConditionTargetScheduler(takeoff, nominal_hover_pose_world=IDENTITY_POSE)

    hover = build_order3_rollout_condition(stage_id="hover", task_mode="hover", seed=0)
    scheduler = Order3ConditionTargetScheduler(hover, nominal_hover_pose_world=IDENTITY_POSE)
    with pytest.raises(SchemaValidationError, match="elapsed_s"):
        scheduler.target_at(-0.1)


def test_rpy_composes_on_nominal_orientation() -> None:
    yaw = rpy_to_quat_xyzw((0.0, 0.0, 0.3))
    roll = rpy_to_quat_xyzw((0.2, 0.0, 0.0))
    composed = quaternion_multiply_xyzw(yaw, roll)
    assert math.sqrt(sum(value * value for value in composed)) == pytest.approx(1.0)
    assert composed != pytest.approx(yaw)
