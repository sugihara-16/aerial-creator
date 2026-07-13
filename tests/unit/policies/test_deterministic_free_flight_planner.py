from __future__ import annotations

import math

import pytest

from amsrr.policies.deterministic_free_flight_planner import (
    DeterministicFreeFlightPlanner,
    Order4FreeFlightContextFactory,
    Order4FreeFlightTrajectoryRuntime,
)
from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    BaselineLowLevelPolicyConfig,
    LowLevelPolicyContext,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order4 import (
    Order4DeterministicPlannerConfig,
    Order4FreeFlightWaypoint,
    build_order4_free_flight_mission,
)
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ContactState,
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.simulation.p4_control_controller_smoke import build_single_module_morphology


def _mission():
    return build_order4_free_flight_mission(
        mission_id="unit-multi-waypoint",
        waypoints=[
            Order4FreeFlightWaypoint(
                waypoint_id="first",
                position_offset_world=[0.1, 0.0, 0.0],
                orientation_rpy_rad=[0.0, 0.0, 0.0],
                transition_duration_s=0.5,
                dwell_s=0.25,
                timeout_s=3.0,
            ),
            Order4FreeFlightWaypoint(
                waypoint_id="second",
                position_offset_world=[0.0, 0.1, 0.0],
                orientation_rpy_rad=[0.0, 0.0, 0.1],
                transition_duration_s=0.5,
                dwell_s=0.25,
                timeout_s=3.0,
            ),
        ],
        hover_height_delta_m=0.5,
        hover_acquisition_dwell_s=0.25,
        final_hover_hold_s=0.5,
        mission_timeout_s=10.0,
    )


def _config() -> Order4DeterministicPlannerConfig:
    return Order4DeterministicPlannerConfig(
        update_rate_hz=4.0,
        horizon_s=1.0,
        knot_dt_s=0.25,
        floor_settle_duration_s=0.5,
        floor_settle_dwell_s=0.25,
        takeoff_duration_s=0.5,
        hover_acquisition_timeout_s=2.0,
        position_tolerance_m=0.02,
        attitude_tolerance_rad=0.05,
        linear_speed_tolerance_mps=0.05,
        angular_speed_tolerance_rad_s=0.05,
        max_tilt_rad=1.2,
        trajectory_expiry_grace_s=0.1,
    )


def _observation(
    morphology,
    *,
    time_s: float,
    pose,
    floor_contact: bool = False,
    qp_feasible: bool = True,
) -> RuntimeObservation:
    contacts = []
    if floor_contact:
        contacts.append(
            ContactState(
                contact_id="floor",
                entity_a="morphology",
                entity_b="floor:/World/defaultGroundPlane",
                active=True,
            )
        )
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=0,
                pose_world=pose,
                twist_world=[0.0] * 6,
            )
        ],
        object_states=[],
        contact_states=contacts,
        controller_status=ControllerStatus(
            status="ok" if qp_feasible else "infeasible",
            qp_feasible=qp_feasible,
        ),
        task_progress=TaskProgressState(progress_ratio=0.0),
    )


def _yaw_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def test_deterministic_planner_runs_state_guards_and_multiple_waypoints() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_single_module_morphology(physical_model)
    mission = _mission()
    config = _config()
    factory = Order4FreeFlightContextFactory(
        mission=mission,
        morphology_graph=morphology,
        planner_config=config,
    )
    planner = DeterministicFreeFlightPlanner(
        physical_model=physical_model,
        config=config,
    )
    runtime = Order4FreeFlightTrajectoryRuntime(planner=planner)
    floor_pose = (0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)
    hover_pose = (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0)
    first_pose = (0.1, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0)
    second_pose = (0.0, 0.1, 0.7, *_yaw_quaternion(0.1))

    sequence = [
        (0.00, floor_pose, True),
        (0.25, floor_pose, True),
        (0.50, floor_pose, True),
        (0.75, hover_pose, False),
        (1.00, hover_pose, False),
        (1.25, hover_pose, False),
        (1.50, hover_pose, False),
        (1.75, first_pose, False),
        (2.00, first_pose, False),
        (2.25, first_pose, False),
        (2.50, second_pose, False),
        (2.75, second_pose, False),
        (3.00, second_pose, False),
        (3.25, second_pose, False),
        (3.50, second_pose, False),
        (3.75, second_pose, False),
    ]
    steps = []
    for time_s, pose, floor_contact in sequence:
        steps.append(
            runtime.step(
                factory.context(
                    _observation(
                        morphology,
                        time_s=time_s,
                        pose=pose,
                        floor_contact=floor_contact,
                    )
                )
            )
        )

    phases = [step.phase for step in steps]
    assert phases[0] == "floor_settle"
    assert "takeoff" in phases
    assert "hover_acquisition" in phases
    assert phases.count("waypoint") >= 4
    assert planner.phase == "complete"
    assert steps[-1].mission_progress_ratio == 1.0
    assert all(
        step.reachability_status == "not_applicable_no_active_assignments"
        and not step.active_knot.contact_assignments
        for step in steps
    )
    assert [
        transition.waypoint_index
        for transition in planner.transitions
        if transition.to_phase == "waypoint"
    ] == [0, 1]
    assert all(
        record["plan_start_time_s"] == pytest.approx(
            sequence[index][0]
        )
        for index, record in enumerate(runtime.plan_records)
    )

    active_trajectory = runtime.active_trajectory
    assert active_trajectory is not None
    low_level = BaselineLowLevelPolicy(
        BaselineLowLevelPolicyConfig(
            control_contract_version="centroidal_local_joint_v2"
        )
    )
    command = low_level.command(
        LowLevelPolicyContext(
            runtime_observation=_observation(
                morphology,
                time_s=sequence[-1][0],
                pose=second_pose,
            ),
            morphology_graph=morphology,
            physical_model=physical_model,
            contact_wrench_trajectory=active_trajectory,
            active_knot=steps[-1].active_knot,
        )
    )
    assert planner.final_target_pose is not None
    assert command.desired_body_pose == pytest.approx(planner.final_target_pose)
    assert command.contact_tracking_bias == {}
    assert command.joint_position_targets
    assert set(command.joint_position_targets.values()) == {0.0}


def test_deterministic_planner_fails_closed_on_controller_infeasibility() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_single_module_morphology(physical_model)
    mission = _mission()
    config = _config()
    factory = Order4FreeFlightContextFactory(
        mission=mission,
        morphology_graph=morphology,
        planner_config=config,
    )
    planner = DeterministicFreeFlightPlanner(
        physical_model=physical_model,
        config=config,
    )
    runtime = Order4FreeFlightTrajectoryRuntime(planner=planner)
    pose = (0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)
    initial_step = runtime.step(
        factory.context(
            _observation(
                morphology,
                time_s=0.0,
                pose=pose,
                floor_contact=True,
            )
        )
    )

    step = runtime.step(
        factory.context(
            _observation(
                morphology,
                time_s=0.25,
                pose=pose,
                floor_contact=True,
                qp_feasible=False,
            )
        )
    )

    assert step.phase == "safe_hold"
    assert step.safe_hold_active
    assert step.failure_reason == "controller_not_feasible"
    assert step.active_knot.centroidal_target.com_pos_world == pytest.approx(
        initial_step.active_knot.centroidal_target.com_pos_world
    )


def test_order4_context_rejects_nonzero_contact_requirement() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_single_module_morphology(physical_model)
    mission = _mission()
    config = _config()
    factory = Order4FreeFlightContextFactory(
        mission=mission,
        morphology_graph=morphology,
        planner_config=config,
    )
    factory.envelope.required_contact_count_range = (1, 1)
    planner = DeterministicFreeFlightPlanner(
        physical_model=physical_model,
        config=config,
    )
    with pytest.raises(SchemaValidationError, match="require zero contacts"):
        planner.plan(
            factory.context(
                _observation(
                    morphology,
                    time_s=0.0,
                    pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
                    floor_contact=True,
                )
            )
        )
