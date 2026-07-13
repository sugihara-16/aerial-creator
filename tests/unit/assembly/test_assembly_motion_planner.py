from __future__ import annotations

import pytest

from amsrr.assembly.assembly_motion_planner import (
    AssemblyMotionPlannerConfig,
    AssemblyMotionPlanningError,
    DeterministicAssemblyMotionPlanner,
)


IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def test_direct_se3_path_is_selected_when_every_sample_is_free() -> None:
    planner = DeterministicAssemblyMotionPlanner(
        AssemblyMotionPlannerConfig(sample_spacing_m=0.05, via_offset_m=0.25)
    )
    goal = (0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)

    plan = planner.plan(IDENTITY, goal, is_pose_collision_free=lambda _pose: True)

    assert plan.method == "direct_se3"
    assert plan.waypoints_world == [goal]
    assert plan.collision_check_count >= 11
    assert type(plan).from_json(plan.to_json()).to_dict() == plan.to_dict()


def test_deterministic_via_path_routes_around_blocked_midpoint() -> None:
    planner = DeterministicAssemblyMotionPlanner(
        AssemblyMotionPlannerConfig(sample_spacing_m=0.025, via_offset_m=0.30)
    )
    goal = (0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def collision_free(pose) -> bool:
        x, y, z = pose[:3]
        return not (0.20 <= x <= 0.40 and abs(y) < 0.10 and abs(z) < 0.10)

    plan = planner.plan(IDENTITY, goal, is_pose_collision_free=collision_free)

    assert plan.method == "single_via_se3"
    assert len(plan.waypoints_world) == 2
    assert plan.waypoints_world[0][2] > 0.0
    assert plan.waypoints_world[-1] == goal


def test_collision_oracle_failure_and_blocked_candidates_fail_closed() -> None:
    planner = DeterministicAssemblyMotionPlanner(
        AssemblyMotionPlannerConfig(sample_spacing_m=0.1, via_offset_m=0.1, maximum_via_depth=1)
    )
    goal = (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    with pytest.raises(AssemblyMotionPlanningError, match="no collision-free"):
        planner.plan(IDENTITY, goal, is_pose_collision_free=lambda pose: pose == IDENTITY)

    def broken_oracle(_pose):
        raise RuntimeError("sensor unavailable")

    with pytest.raises(AssemblyMotionPlanningError, match="oracle failed closed"):
        planner.plan(IDENTITY, goal, is_pose_collision_free=broken_oracle)
