from __future__ import annotations

import pytest

from amsrr.training.order9_configuration_space_planner import (
    DeterministicOrder9ConfigurationSpacePlanner,
    Order9ConfigurationSpacePlannerConfig,
    Order9ConfigurationSpacePlanningError,
    Order9ConfigurationState,
    order9_lightweight_configuration_planning,
)


def _state(x: float, y: float, joint: float) -> Order9ConfigurationState:
    return Order9ConfigurationState(
        base_pose_world=(x, y, 0.0, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"module_0:yaw": joint},
    )


def test_configuration_space_planner_finds_deterministic_detour() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner(
        Order9ConfigurationSpacePlannerConfig(
            maximum_iterations=400,
            deterministic_seed=1,
        )
    )

    def collision_free(state: Order9ConfigurationState) -> bool:
        x, y = state.base_pose_world[:2]
        return not (0.40 < x < 0.60 and abs(y) < 0.15)

    kwargs = {
        "start": _state(0.0, 0.0, 0.0),
        "goal": _state(1.0, 0.0, 1.0),
        "joint_limits_rad": {"module_0:yaw": (-2.0, 2.0)},
        "is_collision_free": collision_free,
        "sampling_center_world": (0.5, 0.0, 0.0),
    }
    first = planner.plan(**kwargs)
    second = planner.plan(**kwargs)

    assert first.method == "deterministic_birrt_connect"
    assert first.states == second.states
    assert first.collision_check_count == second.collision_check_count
    assert len(first.states) >= 3
    assert all(collision_free(state) for state in first.states)


def test_lightweight_admission_refuses_sampled_configuration_search() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()

    def collision_free(state: Order9ConfigurationState) -> bool:
        x, y = state.base_pose_world[:2]
        return not (0.40 < x < 0.60 and abs(y) < 0.15)

    with order9_lightweight_configuration_planning(), pytest.raises(
        Order9ConfigurationSpacePlanningError,
        match="sampled configuration search is disabled",
    ):
        planner.plan(
            start=_state(0.0, 0.0, 0.0),
            goal=_state(1.0, 0.0, 1.0),
            joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
            is_collision_free=collision_free,
            sampling_center_world=(0.5, 0.0, 0.0),
        )


def test_local_contact_planner_uses_bounded_coordinate_detour() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()

    def collision_free(state: Order9ConfigurationState) -> bool:
        x, y = state.base_pose_world[:2]
        return not (0.40 < x < 0.60 and abs(y) < 0.05)

    plan = planner.plan_local_contact(
        start=_state(0.0, 0.0, 0.0),
        goal=_state(1.0, 0.0, 1.0),
        joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
        is_collision_free=collision_free,
        corridor_radius_m=0.25,
    )

    assert plan.method == "deterministic_local_coordinate_bridge"
    assert plan.sampled_state_count == 0
    assert len(plan.states) == 3
    assert all(collision_free(state) for state in plan.states)
    assert max(abs(state.base_pose_world[1]) for state in plan.states) <= 0.25


def test_local_contact_planner_fails_after_bounded_local_sampling() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner(
        Order9ConfigurationSpacePlannerConfig(
            local_contact_maximum_iterations=20,
        )
    )

    with pytest.raises(
        Order9ConfigurationSpacePlanningError,
        match="local contact search exhausted 20 samples",
    ):
        planner.plan_local_contact(
            start=_state(0.0, 0.0, 0.0),
            goal=_state(1.0, 0.0, 1.0),
            joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
            is_collision_free=lambda state: not (
                0.40 < state.base_pose_world[0] < 0.60
            ),
            corridor_radius_m=0.10,
        )


def test_configuration_space_planner_fails_closed_on_colliding_goal() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()

    with pytest.raises(
        Order9ConfigurationSpacePlanningError,
        match="goal state is in collision",
    ):
        planner.plan(
            start=_state(0.0, 0.0, 0.0),
            goal=_state(1.0, 0.0, 1.0),
            joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
            is_collision_free=lambda state: (state.base_pose_world[0] < 0.9),
        )


def test_configuration_space_planner_revalidates_cached_direct_edge() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()
    start = _state(0.25, 0.0, 0.25)
    goal = _state(1.0, 0.0, 1.0)

    accepted, checks = planner.validate_direct_edge(
        start=start,
        goal=goal,
        is_collision_free=lambda state: state.base_pose_world[0] < 0.90,
    )

    assert accepted is False
    assert checks > 1

    accepted, checks = planner.validate_direct_edge(
        start=start,
        goal=goal,
        is_collision_free=lambda _state: True,
    )

    assert accepted is True
    assert checks > 2


def test_configuration_space_planner_prefers_overhead_bridge() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner(
        Order9ConfigurationSpacePlannerConfig(sampling_margin_m=0.65)
    )

    def collision_free(state: Order9ConfigurationState) -> bool:
        x, z = state.base_pose_world[0], state.base_pose_world[2]
        return not (0.40 < x < 0.60 and z < 0.40)

    plan = planner.plan(
        start=_state(0.0, 0.0, 0.0),
        goal=_state(1.0, 0.0, 1.0),
        joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
        is_collision_free=collision_free,
        sampling_center_world=(0.5, 0.0, 0.0),
        overhead_required=True,
    )

    assert plan.method == "deterministic_overhead_configuration_bridge"
    assert max(state.base_pose_world[2] for state in plan.states) == (
        pytest.approx(0.65)
    )
    assert plan.sampled_state_count == 0
    assert all(collision_free(state) for state in plan.states)


def test_overhead_planner_finds_joint_detour_before_descent() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner(
        Order9ConfigurationSpacePlannerConfig(
            maximum_iterations=800,
            deterministic_seed=3,
        )
    )
    start = Order9ConfigurationState(
        base_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"joint_a": -1.0, "joint_b": -1.0},
    )
    goal = Order9ConfigurationState(
        base_pose_world=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"joint_a": 1.0, "joint_b": 1.0},
    )

    def collision_free(state: Order9ConfigurationState) -> bool:
        x, _, z = state.base_pose_world[:3]
        if 0.40 < x < 0.60 and z < 0.40:
            return False
        first = float(state.joint_positions_rad["joint_a"])
        second = float(state.joint_positions_rad["joint_b"])
        return not (
            abs(first) < 0.80 and abs(second) < 0.80 and abs(first - second) < 0.25
        )

    plan = planner.plan(
        start=start,
        goal=goal,
        joint_limits_rad={
            "joint_a": (-2.0, 2.0),
            "joint_b": (-2.0, 2.0),
        },
        is_collision_free=collision_free,
        sampling_center_world=(0.5, 0.0, 0.0),
        overhead_required=True,
    )

    assert plan.method in {
        "deterministic_overhead_coordinate_bridge",
        "deterministic_overhead_birrt_bridge",
    }
    assert all(collision_free(state) for state in plan.states)
    assert all(state.base_pose_world[2] >= 0.65 - 1.0e-9 for state in plan.states[1:-1])


def test_overhead_bridge_snaps_numerically_converged_waypoint() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()
    start = _state(1.0, 0.0, 0.001)
    goal = _state(1.0, 0.0, 0.0)

    plan = planner.plan(
        start=start,
        goal=goal,
        joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
        is_collision_free=lambda _state: True,
        sampling_center_world=(1.0, 0.0, -0.65),
        overhead_required=True,
    )

    assert plan.method == "direct_configuration_edge"
    assert plan.states == (start, goal)


def test_overhead_replan_continues_aligned_vertical_descent() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()
    start = Order9ConfigurationState(
        base_pose_world=(1.0, 0.0, 0.65, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"module_0:yaw": 0.001},
    )
    goal = Order9ConfigurationState(
        base_pose_world=(1.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"module_0:yaw": 0.0},
    )

    plan = planner.plan(
        start=start,
        goal=goal,
        joint_limits_rad={"module_0:yaw": (-2.0, 2.0)},
        is_collision_free=lambda _state: True,
        sampling_center_world=(1.0, 0.0, 0.20),
        overhead_required=True,
    )

    assert plan.method == "direct_configuration_edge"
    assert plan.states == (start, goal)
