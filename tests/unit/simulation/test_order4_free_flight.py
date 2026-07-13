from __future__ import annotations

import pytest

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order4 import (
    ORDER4_FREE_FLIGHT_REPORT_VERSION,
    ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
    Order4DeterministicPlannerConfig,
    Order4TrajectoryRuntimeStep,
    default_order4_free_flight_mission,
)
from amsrr.schemas.policies import CentroidalTarget, ContactWrenchTrajectory, InteractionKnot
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.order4_free_flight import (
    Order4IsaacFreeFlightConfig,
    Order4IsaacFreeFlightEnv,
    order4_free_flight_report_failures,
)
from amsrr.simulation.p4_control_controller_smoke import build_single_module_morphology
from amsrr.simulation.random_morphology_takeoff import (
    RandomMorphologyTakeoffConfig,
    RandomMorphologyTakeoffEnv,
)
from amsrr.utils.hashing import hash_file, stable_hash


def _takeoff_env():
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    takeoff_config = RandomMorphologyTakeoffConfig(
        mesh_search_dirs=["module_urdf"],
        control_contract_version="centroidal_local_joint_v2",
        hover_hold_duration_s=5.0,
        hover_acquisition_timeout_s=5.0,
    )
    backend = IsaacLabBackend(
        load_isaac_lab_backend_config(takeoff_config.backend_config_path)
    )
    return RandomMorphologyTakeoffEnv(
        config=takeoff_config,
        backend=backend,
        physical_model=physical_model,
    )


def _config():
    return Order4IsaacFreeFlightConfig(
        mission=default_order4_free_flight_mission(),
        planner=Order4DeterministicPlannerConfig(),
    )


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_order4_probe_command_binds_mission_runtime_and_gui() -> None:
    takeoff_env = _takeoff_env()
    morphology = build_single_module_morphology(takeoff_env.physical_model)
    env = Order4IsaacFreeFlightEnv(
        config=_config(),
        takeoff_env=takeoff_env,
        viewer="kit",
        realtime_playback=True,
        keep_open_after_rollout_s=20.0,
    )

    command = env.build_probe_command(morphology)

    assert _argument(command, "--steps") == str(env.requested_steps)
    assert _argument(command, "--order4-free-flight-mission-json") == env.config.mission.to_canonical_json()
    assert _argument(command, "--order4-planner-config-json") == env.config.planner.to_json()
    assert _argument(command, "--hover-hold-duration-s") == "5.0"
    assert _argument(command, "--viz") == "kit"
    assert "--realtime-playback" in command
    assert _argument(command, "--keep-open-after-smoke-s") == "20.0"
    assert "--order3-rollout-condition-json" not in command


def test_order4_optional_pi_l_checkpoint_is_hash_bound_and_propagated(tmp_path) -> None:
    takeoff_env = _takeoff_env()
    morphology = build_single_module_morphology(takeoff_env.physical_model)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"order4-checkpoint-contract")
    checkpoint_sha256 = hash_file(checkpoint)
    config = Order4IsaacFreeFlightConfig(
        mission=default_order4_free_flight_mission(),
        planner=Order4DeterministicPlannerConfig(),
        pi_l_checkpoint_path=str(checkpoint),
        expected_pi_l_checkpoint_sha256=checkpoint_sha256,
    )

    command = Order4IsaacFreeFlightEnv(
        config=config,
        takeoff_env=takeoff_env,
    ).build_probe_command(morphology)

    assert _argument(command, "--order4-pi-l-checkpoint-path") == str(checkpoint)
    with pytest.raises(SchemaValidationError, match="sha256 mismatch"):
        Order4IsaacFreeFlightEnv(
            config=Order4IsaacFreeFlightConfig(
                mission=config.mission,
                planner=config.planner,
                pi_l_checkpoint_path=str(checkpoint),
                expected_pi_l_checkpoint_sha256="0" * 64,
            ),
            takeoff_env=takeoff_env,
        )


def test_order4_fake_real_report_passes_and_tampering_fails() -> None:
    takeoff_env = _takeoff_env()
    morphology = build_single_module_morphology(takeoff_env.physical_model)
    config = _config()
    probe_env = Order4IsaacFreeFlightEnv(
        config=config,
        takeoff_env=takeoff_env,
    )
    report = _valid_report(
        morphology=morphology,
        config=config,
        takeoff_env=takeoff_env,
        requested_steps=probe_env.requested_steps,
    )
    env = Order4IsaacFreeFlightEnv(
        config=config,
        takeoff_env=takeoff_env,
        command_executor=lambda command, timeout: report,
    )

    result = env.run(morphology, dry_run=False, check_availability=False)

    assert result.passed
    assert result.report_validation_failures == []

    report["order4_free_flight_time_origin_valid"] = False
    failures = order4_free_flight_report_failures(
        report,
        morphology_graph=morphology,
        config=config,
        takeoff_env=takeoff_env,
        requested_steps=probe_env.requested_steps,
    )
    assert "mismatch:order4_free_flight_time_origin_valid" in failures


def _valid_report(*, morphology, config, takeoff_env, requested_steps):
    pose = (0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0)
    first = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=pose[:3],
            com_vel_world=(0.0, 0.0, 0.0),
            body_orientation_world=pose[3:],
        ),
    )
    second = InteractionKnot(
        t_rel_s=2.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=pose[:3],
            com_vel_world=(0.0, 0.0, 0.0),
            body_orientation_world=pose[3:],
        ),
    )
    trajectory = ContactWrenchTrajectory(
        horizon_s=2.0,
        dt_s=0.25,
        knots=[first, second],
        derived_mode_label="order4_free_flight_complete",
    )
    runtime_step = Order4TrajectoryRuntimeStep(
        runtime_version=ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
        time_s=10.0,
        mission_hash=config.mission.mission_hash,
        phase="complete",
        waypoint_index=len(config.mission.waypoints) - 1,
        mission_progress_ratio=1.0,
        plan_sequence=1,
        plan_start_time_s=10.0,
        plan_elapsed_s=0.0,
        active_knot_index=0,
        next_knot_index=1,
        interpolation_ratio=0.0,
        replanned=True,
        safe_hold_active=False,
        failure_reason=None,
        reachability_status="not_applicable_no_active_assignments",
        active_knot=first,
    )
    transitions = [
        {
            "from_phase": None,
            "to_phase": "floor_settle",
            "time_s": 0.0,
            "reason": "mission_initialized",
            "waypoint_index": None,
        },
        {
            "from_phase": "floor_settle",
            "to_phase": "takeoff",
            "time_s": 1.0,
            "reason": "guard",
            "waypoint_index": None,
        },
        {
            "from_phase": "takeoff",
            "to_phase": "hover_acquisition",
            "time_s": 3.0,
            "reason": "guard",
            "waypoint_index": None,
        },
    ]
    previous = "hover_acquisition"
    for index in range(len(config.mission.waypoints)):
        transitions.append(
            {
                "from_phase": previous,
                "to_phase": "waypoint",
                "time_s": 4.0 + index,
                "reason": "guard",
                "waypoint_index": index,
            }
        )
        previous = "waypoint"
    transitions.extend(
        [
            {
                "from_phase": "waypoint",
                "to_phase": "final_hover",
                "time_s": 8.0,
                "reason": "guard",
                "waypoint_index": len(config.mission.waypoints) - 1,
            },
            {
                "from_phase": "final_hover",
                "to_phase": "complete",
                "time_s": 13.0,
                "reason": "guard",
                "waypoint_index": len(config.mission.waypoints) - 1,
            },
        ]
    )
    return {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": True,
        "command_probe_passed": True,
        "command_returncode": 0,
        "order4_free_flight_enabled": True,
        "order4_free_flight_passed": True,
        "order4_free_flight_report_version": ORDER4_FREE_FLIGHT_REPORT_VERSION,
        "order4_free_flight_mission": config.mission.to_dict(),
        "order4_free_flight_mission_hash": config.mission.mission_hash,
        "order4_free_flight_planner_config": config.planner.to_dict(),
        "order4_free_flight_planner_config_hash": stable_hash(config.planner),
        "order4_free_flight_deterministic_pi_h": True,
        "order4_free_flight_pi_h_scope": "free_flight_only_no_contact_planning",
        "order4_free_flight_trajectory_runtime_version": ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
        "order4_free_flight_final_phase": "complete",
        "order4_free_flight_progress_ratio": 1.0,
        "order4_free_flight_waypoint_count": len(config.mission.waypoints),
        "order4_free_flight_completed_waypoint_count": len(config.mission.waypoints),
        "order4_free_flight_time_origin_valid": True,
        "order4_free_flight_reachability_status": "not_applicable_no_active_assignments",
        "order4_free_flight_max_active_assignment_count": 0,
        "order4_free_flight_safe_hold_active": False,
        "order4_free_flight_failure_reason": None,
        "order4_free_flight_existing_actor_progress_unchanged": True,
        "order4_free_flight_final_hover_hold_time_s": config.mission.final_hover_hold_s,
        "order4_free_flight_final_hover_hold_required_s": config.mission.final_hover_hold_s,
        "order4_free_flight_plan_records": [
            {
                "plan_sequence": 1,
                "plan_start_time_s": 10.0,
                "phase": "complete",
                "waypoint_index": len(config.mission.waypoints) - 1,
                "trajectory": trajectory.to_dict(),
            }
        ],
        "order4_free_flight_runtime_steps": [runtime_step.to_dict()],
        "order4_free_flight_phase_transitions": transitions,
        "order4_free_flight_low_level_source": "deterministic_baseline_pi_l",
        "order4_pi_l_checkpoint_sha256": None,
        "random_morphology_takeoff_smoke_passed": True,
        "random_morphology_takeoff_settle_passed": True,
        "random_morphology_takeoff_ramp_passed": True,
        "random_morphology_takeoff_hover_passed": True,
        "random_morphology_takeoff_fixed_dock_neutral_hold_passed": True,
        "random_morphology_takeoff_exact_cross_module_collision_passed": True,
        "random_morphology_takeoff_finite_state": True,
        "random_morphology_takeoff_logging_passed": True,
        "random_morphology_takeoff_graph_id": morphology.graph_id,
        "random_morphology_takeoff_morphology_hash": morphology.stable_hash(),
        "random_morphology_takeoff_backend_config_hash": takeoff_env.backend.config.stable_hash(),
        "random_morphology_takeoff_physical_model_hash": takeoff_env.physical_model.stable_hash(),
        "random_morphology_takeoff_collision_geometry_hash": collision_geometry_content_hash(
            takeoff_env.physical_model,
            mesh_search_dirs=takeoff_env.config.mesh_search_dirs,
        ),
        "random_morphology_takeoff_requested_steps": requested_steps,
        "random_morphology_takeoff_control_contract_version": "centroidal_local_joint_v2",
        "random_morphology_takeoff_qp_infeasible_count": 0,
        "random_morphology_takeoff_controller_clipped_count": 0,
        "random_morphology_takeoff_missing_actuator_count": 0,
        "random_morphology_takeoff_unsupported_actuator_count": 0,
        "random_morphology_takeoff_clipped_target_count": 0,
        "random_morphology_takeoff_application_unresolved_target_count": 0,
        "random_morphology_takeoff_dynamic_exact_contact_violation_step_count": 0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_observation_count": 0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_saturation_step_count": 0,
        "random_morphology_takeoff_max_abs_dock_position_target_rad": 0.0,
        "random_morphology_takeoff_max_abs_dock_velocity_target_rad_s": 0.0,
        "random_morphology_takeoff_max_abs_dock_torque_bias_nm": 0.0,
        "random_morphology_takeoff_artifacts": {
            "order4_learned_pi_h_claim": False,
            "order4_contact_planning_claim": False,
            "is_p4_full_completion": False,
        },
    }
