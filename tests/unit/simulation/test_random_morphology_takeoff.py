from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.random_morphology_takeoff import (
    DeterministicTakeoffScheduler,
    RandomMorphologyTakeoffConfig,
    RandomMorphologyTakeoffEnv,
    TakeoffPhase,
    compute_floor_contact_placement,
    intended_dock_body_link_pairs,
    random_morphology_takeoff_result_from_report,
)


def _sample(*, seed: int = 3, module_count: int = 3):
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=seed,
        module_count=module_count,
    )
    return physical_model, morphology


def _backend_config_hash() -> str:
    return load_isaac_lab_backend_config(
        "configs/env/isaac_lab.yaml"
    ).stable_hash()


def test_floor_contact_placement_uses_order1_collision_bounds() -> None:
    physical_model, morphology = _sample(module_count=4)

    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
        floor_z_m=0.0,
        clearance_m=0.002,
    )

    assert placement.floor_gap_m == pytest.approx(0.002)
    assert placement.root_pose_world[2] > 0.0
    assert placement.collision_bounds_root.method == "order1_morphology_collision_aabbs_v1"
    assert placement.collision_bounds_root.collision_geometry_count == (
        len(physical_model.collision_primitives) * len(morphology.modules)
    )
    assert placement.collision_bounds_root.mesh_geometry_count > 0


def test_intended_dock_body_pairs_bind_graph_ports_to_physical_links() -> None:
    physical_model, morphology = _sample(module_count=4)
    ports_by_id = {port.port_global_id: port for port in morphology.ports}
    dock_specs_by_id = {port.port_id: port for port in physical_model.dock_ports}

    pairs = intended_dock_body_link_pairs(morphology, physical_model)

    assert pairs == [
        (
            edge.src_module_id,
            dock_specs_by_id[ports_by_id[edge.src_port_id].port_local_id].parent_link,
            edge.dst_module_id,
            dock_specs_by_id[ports_by_id[edge.dst_port_id].port_local_id].parent_link,
        )
        for edge in sorted(morphology.dock_edges, key=lambda item: item.edge_id)
    ]


def test_takeoff_scheduler_is_settle_ramp_hover_complete() -> None:
    config = RandomMorphologyTakeoffConfig(
        settle_duration_s=1.0,
        takeoff_ramp_duration_s=2.0,
        hover_hold_duration_s=1.0,
        hover_height_delta_m=0.6,
    )
    scheduler = DeterministicTakeoffScheduler(config)
    settled_pose = (0.2, -0.1, 0.08, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5)

    settle = scheduler.target_at(0.75, settled_pose_world=settled_pose)
    ramp_start = scheduler.target_at(1.0, settled_pose_world=settled_pose)
    ramp_mid = scheduler.target_at(2.0, settled_pose_world=settled_pose)
    hover = scheduler.target_at(3.0, settled_pose_world=settled_pose)
    complete = scheduler.target_at(4.0, settled_pose_world=settled_pose)

    assert settle.phase == TakeoffPhase.SETTLE
    assert settle.thrust_enabled is False
    assert settle.desired_pose_world is None
    assert ramp_start.phase == TakeoffPhase.TAKEOFF_RAMP
    assert ramp_start.ramp_progress == 0.0
    assert ramp_start.desired_pose_world == pytest.approx(settled_pose)
    assert ramp_mid.ramp_progress == pytest.approx(0.5)
    assert ramp_mid.desired_pose_world[:3] == pytest.approx((0.2, -0.1, 0.38))
    assert hover.phase == TakeoffPhase.HOVER_HOLD
    assert hover.desired_pose_world == pytest.approx((0.2, -0.1, 0.68, 0.0, 0.0, 0.0, 1.0))
    assert complete.phase == TakeoffPhase.COMPLETE


def test_takeoff_env_dry_contract_is_graph_specific_and_not_physics_success() -> None:
    _, morphology = _sample(seed=11, module_count=2)
    config = RandomMorphologyTakeoffConfig()
    backend = IsaacLabBackend(load_isaac_lab_backend_config(config.backend_config_path))
    env = RandomMorphologyTakeoffEnv(config=config, backend=backend)

    result = env.run(morphology, dry_run=True)
    command = result.report["probe_command"]

    assert result.attempted is False
    assert result.unit_contract_passed is True
    assert result.real_isaac_passed is False
    assert result.isaac_backed is False
    assert result.metrics["module_count"] == 2
    assert "--random-morphology-takeoff" in command
    assert "--random-morphology-graph-json" in command
    assert morphology.graph_id in command[command.index("--random-morphology-graph-json") + 1]
    assert "--spawn-height" in command
    assert "--takeoff-settle-duration-s" in command
    assert "--takeoff-ramp-duration-s" in command
    assert "--takeoff-hover-height-delta-m" in command
    assert "--takeoff-exact-cross-module-contact-force-threshold-n" in command
    assert "--takeoff-exact-cross-module-contact-max-patches-per-body-pair" in command
    assert command[command.index("--dt") + 1] == str(config.simulation_dt_s)
    assert "--takeoff-initial-root-position-tolerance-m" in command
    assert "--takeoff-initial-root-attitude-tolerance-rad" in command
    assert "--random-morphology-mesh-search-dir" in command
    assert command[command.index("--random-morphology-mesh-search-dir") + 1] == "module_urdf"
    assert command[command.index("--steps") + 1] == str(config.required_steps)


def test_takeoff_env_rejects_physical_model_backend_urdf_mismatch() -> None:
    physical_model, _ = _sample(module_count=2)
    config = RandomMorphologyTakeoffConfig()
    backend_config = load_isaac_lab_backend_config(config.backend_config_path)
    backend = IsaacLabBackend(
        replace(backend_config, holon_urdf_path="module_urdf/holon.urdf.xacro")
    )

    with pytest.raises(SchemaValidationError, match="same URDF"):
        RandomMorphologyTakeoffEnv(
            config=config,
            backend=backend,
            physical_model=physical_model,
        )


def test_takeoff_acceptance_rejects_pseudoinverse_mode() -> None:
    with pytest.raises(SchemaValidationError, match="requires allocation_mode='rigid_body_qp'"):
        RandomMorphologyTakeoffConfig(allocation_mode="rigid_body_pseudoinverse").validate()


def _valid_takeoff_report(*, physical_model, morphology, placement) -> dict[str, object]:
    steps = 6
    cross_module_pair_count = (
        len(morphology.modules) * (len(morphology.modules) - 1) // 2
    )
    nonadjacent_pair_count = (
        cross_module_pair_count - len(morphology.dock_edges)
    )
    module_ids = sorted(module.module_id for module in morphology.modules)
    cross_module_pair_max_forces = {
        f"{src_module_id}-{dst_module_id}": 0.0
        for src_index, src_module_id in enumerate(module_ids)
        for dst_module_id in module_ids[src_index + 1 :]
    }
    dock_link_pairs = intended_dock_body_link_pairs(morphology, physical_model)
    dock_path_pairs = [
        [
            f"/World/Holon/module_{src_module_id}__{src_link}",
            f"/World/Holon/module_{dst_module_id}__{dst_link}",
        ]
        for src_module_id, src_link, dst_module_id, dst_link in dock_link_pairs
    ]
    runtime_observations = []
    policy_commands = []
    controller_commands = []
    actuator_records = []
    for index in range(steps):
        time_s = index * 0.005
        status = ControllerStatus(status="ok", qp_feasible=True)
        runtime_observations.append(
            RuntimeObservation(
                time_s=time_s,
                morphology_graph=morphology,
                module_states=[
                    ModuleRuntimeState(
                        module_id=module.module_id,
                        pose_world=module.pose_in_design_frame,
                        twist_world=[0.0] * 6,
                    )
                    for module in morphology.modules
                ],
                object_states=[],
                contact_states=[],
                controller_status=status,
                task_progress=TaskProgressState(),
            ).to_dict()
        )
        policy_commands.append(PolicyCommand().to_dict())
        controller_commands.append(
            ControllerCommand(
                rotor_thrusts_n={},
                vectoring_joint_targets={},
                joint_torque_commands={},
                dock_mechanism_commands={},
                controller_status=status,
            ).to_dict()
        )
        actuator_records.append(
            IsaacActuatorTargetRecord(
                time_s=time_s,
                backend="isaac_lab",
                morphology_graph_id=morphology.graph_id,
                command_index=index,
                actuator_targets=[],
                qp_status="ok",
            ).to_dict()
        )
    return {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": True,
        "command_probe_passed": True,
        "command_returncode": 0,
        "allocation_mode": "rigid_body_qp",
        "random_morphology_takeoff_smoke": True,
        "random_morphology_takeoff_smoke_passed": True,
        "random_morphology_takeoff_graph_id": morphology.graph_id,
        "random_morphology_takeoff_morphology_hash": morphology.stable_hash(),
        "random_morphology_takeoff_backend_config_hash": _backend_config_hash(),
        "random_morphology_takeoff_physical_model_hash": physical_model.stable_hash(),
        "random_morphology_takeoff_collision_geometry_hash": collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        "random_morphology_takeoff_allocation_mode": "rigid_body_qp",
        "random_morphology_takeoff_module_count": len(morphology.modules),
        "random_morphology_takeoff_dock_edge_count": len(morphology.dock_edges),
        "random_morphology_takeoff_single_articulation": True,
        "random_morphology_takeoff_assembly_representation": "reset_time_fixed_dock_tree",
        "random_morphology_takeoff_learned_policy_used": False,
        "random_morphology_takeoff_controller": "deterministic_qpid",
        "random_morphology_takeoff_sim_dt_s": 0.005,
        "random_morphology_takeoff_sim_dt_matches_config": True,
        "random_morphology_takeoff_floor_spawned": True,
        "random_morphology_takeoff_floor_pose_evidenced": True,
        "random_morphology_takeoff_floor_placement": placement.to_dict(),
        "random_morphology_takeoff_initial_root_pose_world": list(placement.root_pose_world),
        "random_morphology_takeoff_initial_root_position_error_m": 0.001,
        "random_morphology_takeoff_initial_root_position_tolerance_m": 0.002,
        "random_morphology_takeoff_initial_root_attitude_error_rad": 0.0005,
        "random_morphology_takeoff_initial_root_attitude_tolerance_rad": 0.001,
        "random_morphology_takeoff_floor_contact_evidenced": True,
        "random_morphology_takeoff_floor_contact_force_threshold_n": 0.5,
        "random_morphology_takeoff_floor_contact_max_aggregate_force_n": 12.0,
        "random_morphology_takeoff_floor_contact_dwell_time_s": 0.15,
        "random_morphology_takeoff_floor_contact_dwell_required_s": 0.10,
        "random_morphology_takeoff_contact_sensor_body_count": len(morphology.modules),
        "random_morphology_takeoff_contact_external_collider_scope": "floor_only",
        "random_morphology_takeoff_self_collisions_enabled": True,
        "random_morphology_takeoff_exact_cross_module_collision_passed": True,
        "random_morphology_takeoff_exact_nonadjacent_collision_passed": True,
        "random_morphology_takeoff_exact_collision_rigid_body_count": len(
            morphology.modules
        ),
        "random_morphology_takeoff_exact_collision_filtered_body_pair_count": (
            1 + len(morphology.dock_edges)
        ),
        "random_morphology_takeoff_exact_collision_same_module_filtered_body_pair_count": 1,
        "random_morphology_takeoff_exact_collision_intended_dock_body_pair_count": len(
            morphology.dock_edges
        ),
        "random_morphology_takeoff_exact_collision_intended_dock_body_link_pairs": [
            list(pair) for pair in dock_link_pairs
        ],
        "random_morphology_takeoff_exact_collision_intended_dock_body_pairs": dock_path_pairs,
        "random_morphology_takeoff_exact_collision_adjacent_module_pair_count": len(
            morphology.dock_edges
        ),
        "random_morphology_takeoff_exact_collision_nonadjacent_module_pair_count": (
            nonadjacent_pair_count
        ),
        "random_morphology_takeoff_exact_nonadjacent_contact_count": 0,
        "random_morphology_takeoff_exact_nonadjacent_contact_pairs": [],
        "random_morphology_takeoff_filtered_scope_contact_count": 0,
        "random_morphology_takeoff_unclassified_robot_contact_count": 0,
        "random_morphology_takeoff_exact_collision_check_method": "isaac_physx_get_initial_collider_pairs_v1",
        "random_morphology_takeoff_exact_collision_fixed_module_root_pose_invariant": True,
        "random_morphology_takeoff_exact_collision_raw_pair_count": 0,
        "random_morphology_takeoff_exact_collision_robot_pair_count": 0,
        "random_morphology_takeoff_dynamic_exact_collision_check_method": "omni_physics_tensors_force_matrix_and_contact_data_v2",
        "random_morphology_takeoff_dynamic_exact_contact_scope": "all_cross_module_except_intended_dock_body_pairs",
        "random_morphology_takeoff_dynamic_exact_contact_view_count": cross_module_pair_count,
        "random_morphology_takeoff_dynamic_exact_contact_view_update_count": (
            steps * cross_module_pair_count
        ),
        "random_morphology_takeoff_dynamic_exact_contact_force_threshold_n": 0.001,
        "random_morphology_takeoff_dynamic_exact_contact_max_force_n": 0.0,
        "random_morphology_takeoff_dynamic_exact_contact_violation_step_count": 0,
        "random_morphology_takeoff_dynamic_exact_pair_max_forces_n": (
            cross_module_pair_max_forces
        ),
        "random_morphology_takeoff_dynamic_exact_raw_contact_method": "omni_physics_tensors_get_contact_data_v1",
        "random_morphology_takeoff_dynamic_exact_raw_contact_max_patches_per_body_pair": 8,
        "random_morphology_takeoff_dynamic_exact_raw_contact_capacity": (
            cross_module_pair_count * 8
        ),
        "random_morphology_takeoff_dynamic_exact_raw_contact_view_update_count": (
            steps * cross_module_pair_count
        ),
        "random_morphology_takeoff_dynamic_exact_raw_contact_observation_count": 0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_observed_step_count": 0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_max_force_n": 0.0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_min_separation_m": 0.0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_saturation_step_count": 0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_observed": False,
        "random_morphology_takeoff_dynamic_exact_raw_contact_buffer_saturated": False,
        "random_morphology_takeoff_dynamic_exact_pair_raw_contact_counts": {
            key: 0 for key in cross_module_pair_max_forces
        },
        "random_morphology_takeoff_exact_adjacent_unintended_contact_count": 0,
        "random_morphology_takeoff_exact_adjacent_unintended_contact_pairs": [],
        "random_morphology_takeoff_resolved_fc_body_count": len(morphology.modules),
        "random_morphology_takeoff_settle_zero_thrust": True,
        "random_morphology_takeoff_settle_duration_s": 1.0,
        "random_morphology_takeoff_settle_passed": True,
        "random_morphology_takeoff_settled_pose_world": list(
            placement.root_pose_world
        ),
        "random_morphology_takeoff_settled_linear_speed_mps": 0.05,
        "random_morphology_takeoff_settle_linear_speed_threshold_mps": 0.20,
        "random_morphology_takeoff_settled_angular_speed_rad_s": 0.10,
        "random_morphology_takeoff_settle_angular_speed_threshold_rad_s": 0.50,
        "random_morphology_takeoff_settle_low_speed_dwell_time_s": 0.30,
        "random_morphology_takeoff_settle_low_speed_dwell_required_s": 0.25,
        "random_morphology_takeoff_ramp_passed": True,
        "random_morphology_takeoff_takeoff_ramp_duration_s": 2.0,
        "random_morphology_takeoff_ramp_max_progress": 1.0,
        "random_morphology_takeoff_height_gain_ratio": 0.95,
        "random_morphology_takeoff_min_height_gain_ratio": 0.80,
        "random_morphology_takeoff_hover_passed": True,
        "random_morphology_takeoff_hover_height_delta_m": 0.5,
        "random_morphology_takeoff_stop_on_hover_hold": True,
        "random_morphology_takeoff_hover_target_pose_world": [
            placement.root_pose_world[0],
            placement.root_pose_world[1],
            placement.root_pose_world[2] + 0.5,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "random_morphology_takeoff_final_position_error_m": 0.05,
        "random_morphology_takeoff_position_error_threshold_m": 0.20,
        "random_morphology_takeoff_final_attitude_error_rad": 0.10,
        "random_morphology_takeoff_attitude_error_threshold_rad": 0.25,
        "random_morphology_takeoff_final_linear_speed_mps": 0.05,
        "random_morphology_takeoff_hover_linear_speed_threshold_mps": 0.15,
        "random_morphology_takeoff_final_angular_speed_rad_s": 0.10,
        "random_morphology_takeoff_hover_angular_speed_threshold_rad_s": 0.25,
        "random_morphology_takeoff_hover_hold_time_s": 1.0,
        "random_morphology_takeoff_hover_hold_required_s": 1.0,
        "random_morphology_takeoff_hover_acquisition_timeout_s": 2.0,
        "random_morphology_takeoff_max_vertical_speed_mps": 1.0,
        "random_morphology_takeoff_max_vertical_speed_threshold_mps": 3.0,
        "random_morphology_takeoff_finite_state": True,
        "random_morphology_takeoff_logging_passed": True,
        "random_morphology_takeoff_qp_infeasible_count": 0,
        "random_morphology_takeoff_controller_clipped_count": 0,
        "random_morphology_takeoff_missing_actuator_count": 0,
        "random_morphology_takeoff_unsupported_actuator_count": 0,
        "random_morphology_takeoff_clipped_target_count": 0,
        "random_morphology_takeoff_application_requested_target_count": 32,
        "random_morphology_takeoff_application_applied_target_count": 32,
        "random_morphology_takeoff_application_unresolved_target_count": 0,
        "random_morphology_takeoff_reaction_torque_target_count": 16,
        "random_morphology_takeoff_reaction_torque_abs_sum_nm": 1.5,
        "random_morphology_takeoff_steps": steps,
        "random_morphology_takeoff_requested_steps": RandomMorphologyTakeoffConfig().required_steps,
        "random_morphology_takeoff_phase_counts": {
            "settle": 2,
            "takeoff_ramp": 2,
            "hover_hold": 2,
            "complete": 0,
        },
        "random_morphology_takeoff_phase_transitions": [
            {"from_phase": None, "to_phase": "settle", "time_s": 0.0},
            {"from_phase": "settle", "to_phase": "takeoff_ramp", "time_s": 1.0},
            {"from_phase": "takeoff_ramp", "to_phase": "hover_hold", "time_s": 3.0},
        ],
        "random_morphology_takeoff_runtime_observations": runtime_observations,
        "random_morphology_takeoff_policy_commands": policy_commands,
        "random_morphology_takeoff_controller_commands": controller_commands,
        "random_morphology_takeoff_actuator_target_records": actuator_records,
        "random_morphology_takeoff_root_pose_history": [
            list(placement.root_pose_world) for _ in range(steps)
        ],
        "random_morphology_takeoff_artifacts": {
            "phase": "P4-full-order2",
            "backend": "isaac_lab",
            "isaac_backed": True,
            "dry_run": False,
            "is_p4_full_completion": False,
            "physical_success_claim": "floor_takeoff_hover_only",
            "object_task_claim": False,
            "learned_policy_claim": False,
        },
    }


def test_takeoff_report_accepts_only_complete_graph_bound_real_isaac_evidence() -> None:
    physical_model, morphology = _sample(module_count=2)
    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
    )

    result = random_morphology_takeoff_result_from_report(
        morphology,
        placement=placement,
        report=_valid_takeoff_report(
            physical_model=physical_model,
            morphology=morphology,
            placement=placement,
        ),
        expected_backend_config_hash=_backend_config_hash(),
        expected_physical_model_hash=physical_model.stable_hash(),
        expected_collision_geometry_hash=collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        expected_config=RandomMorphologyTakeoffConfig(),
    )

    assert result.real_isaac_passed is True
    assert result.isaac_backed is True
    assert result.failure_reason is None
    assert result.metrics["random_morphology_takeoff_report_validation_failures"] == []


def test_takeoff_report_rejects_stale_dynamic_contact_pair_keys() -> None:
    physical_model, morphology = _sample(module_count=3)
    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
    )
    report = _valid_takeoff_report(
        physical_model=physical_model,
        morphology=morphology,
        placement=placement,
    )
    report["random_morphology_takeoff_dynamic_exact_pair_max_forces_n"] = {
        "stale-0": 0.0,
        "stale-1": 0.0,
        "stale-2": 0.0,
    }

    result = random_morphology_takeoff_result_from_report(
        morphology,
        placement=placement,
        report=report,
        expected_backend_config_hash=_backend_config_hash(),
        expected_physical_model_hash=physical_model.stable_hash(),
        expected_collision_geometry_hash=collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        expected_config=RandomMorphologyTakeoffConfig(),
    )

    assert result.real_isaac_passed is False
    assert "dynamic_exact_pair_max_forces_n" in (result.failure_reason or "")


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("spawn_passed", False),
        ("isaac_backed", False),
        ("command_returncode", 1),
        ("allocation_mode", "rigid_body_pseudoinverse"),
        ("random_morphology_takeoff_graph_id", "stale-graph"),
        ("random_morphology_takeoff_morphology_hash", "stale-hash"),
        ("random_morphology_takeoff_backend_config_hash", "stale-backend"),
        ("random_morphology_takeoff_physical_model_hash", "stale-model"),
        ("random_morphology_takeoff_collision_geometry_hash", "stale-geometry"),
        ("random_morphology_takeoff_allocation_mode", "rigid_body_pseudoinverse"),
        ("random_morphology_takeoff_module_count", 99),
        ("random_morphology_takeoff_dock_edge_count", 99),
        ("random_morphology_takeoff_floor_pose_evidenced", False),
        ("random_morphology_takeoff_floor_contact_evidenced", False),
        ("random_morphology_takeoff_self_collisions_enabled", False),
        ("random_morphology_takeoff_exact_cross_module_collision_passed", False),
        ("random_morphology_takeoff_exact_nonadjacent_collision_passed", False),
        ("random_morphology_takeoff_exact_nonadjacent_contact_count", 1),
        ("random_morphology_takeoff_exact_adjacent_unintended_contact_count", 1),
        ("random_morphology_takeoff_dynamic_exact_contact_max_force_n", 0.01),
        ("random_morphology_takeoff_dynamic_exact_contact_violation_step_count", 1),
        ("random_morphology_takeoff_dynamic_exact_raw_contact_observation_count", 1),
        ("random_morphology_takeoff_dynamic_exact_raw_contact_observed", True),
        ("random_morphology_takeoff_dynamic_exact_raw_contact_saturation_step_count", 1),
        ("random_morphology_takeoff_dynamic_exact_raw_contact_buffer_saturated", True),
        ("random_morphology_takeoff_dynamic_exact_pair_raw_contact_counts", {"0-1": 1}),
        ("random_morphology_takeoff_floor_contact_max_aggregate_force_n", 0.1),
        ("random_morphology_takeoff_settle_passed", False),
        ("random_morphology_takeoff_settle_duration_s", 0.5),
        ("random_morphology_takeoff_settle_low_speed_dwell_time_s", 0.1),
        ("random_morphology_takeoff_ramp_passed", False),
        ("random_morphology_takeoff_takeoff_ramp_duration_s", 1.5),
        ("random_morphology_takeoff_ramp_max_progress", 0.001),
        ("random_morphology_takeoff_height_gain_ratio", 0.1),
        ("random_morphology_takeoff_hover_passed", False),
        ("random_morphology_takeoff_hover_height_delta_m", 0.4),
        ("random_morphology_takeoff_stop_on_hover_hold", False),
        ("random_morphology_takeoff_final_position_error_m", 0.3),
        ("random_morphology_takeoff_final_attitude_error_rad", 0.3),
        ("random_morphology_takeoff_hover_hold_time_s", 0.5),
        ("random_morphology_takeoff_final_linear_speed_mps", 0.2),
        ("random_morphology_takeoff_max_vertical_speed_mps", 4.0),
        ("random_morphology_takeoff_finite_state", False),
        ("random_morphology_takeoff_logging_passed", False),
        ("random_morphology_takeoff_qp_infeasible_count", 1),
        ("random_morphology_takeoff_application_unresolved_target_count", 1),
        ("random_morphology_takeoff_reaction_torque_target_count", 0),
        ("random_morphology_takeoff_reaction_torque_abs_sum_nm", 0.0),
    ],
)
def test_takeoff_report_rejects_invalid_identity_backend_and_evidence(
    key: str,
    invalid_value: object,
) -> None:
    physical_model, morphology = _sample(module_count=2)
    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
    )
    report = _valid_takeoff_report(
        physical_model=physical_model,
        morphology=morphology,
        placement=placement,
    )
    report[key] = invalid_value

    result = random_morphology_takeoff_result_from_report(
        morphology,
        placement=placement,
        report=report,
        expected_backend_config_hash=_backend_config_hash(),
        expected_physical_model_hash=physical_model.stable_hash(),
        expected_collision_geometry_hash=collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        expected_config=RandomMorphologyTakeoffConfig(),
    )

    assert result.real_isaac_passed is False
    assert result.failure_reason is not None
    assert key in result.failure_reason


@pytest.mark.parametrize(
    "missing_key",
    [
        "isaac_backed",
        "command_returncode",
        "random_morphology_takeoff_morphology_hash",
        "random_morphology_takeoff_backend_config_hash",
        "random_morphology_takeoff_physical_model_hash",
        "random_morphology_takeoff_allocation_mode",
        "random_morphology_takeoff_sim_dt_matches_config",
        "random_morphology_takeoff_floor_contact_evidenced",
        "random_morphology_takeoff_dynamic_exact_raw_contact_method",
        "random_morphology_takeoff_application_requested_target_count",
        "random_morphology_takeoff_reaction_torque_abs_sum_nm",
    ],
)
def test_takeoff_report_rejects_missing_required_keys(missing_key: str) -> None:
    physical_model, morphology = _sample(module_count=2)
    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
    )
    report = _valid_takeoff_report(
        physical_model=physical_model,
        morphology=morphology,
        placement=placement,
    )
    del report[missing_key]

    result = random_morphology_takeoff_result_from_report(
        morphology,
        placement=placement,
        report=report,
        expected_backend_config_hash=_backend_config_hash(),
        expected_physical_model_hash=physical_model.stable_hash(),
        expected_collision_geometry_hash=collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        expected_config=RandomMorphologyTakeoffConfig(),
    )

    assert result.real_isaac_passed is False
    assert result.failure_reason is not None
    assert f"missing:{missing_key}" in result.failure_reason


def test_takeoff_report_rejects_missing_key_phase_count_and_artifact_evidence() -> None:
    physical_model, morphology = _sample(module_count=2)
    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
    )
    report = _valid_takeoff_report(
        physical_model=physical_model,
        morphology=morphology,
        placement=placement,
    )
    del report["random_morphology_takeoff_smoke_passed"]
    phase_counts = report["random_morphology_takeoff_phase_counts"]
    assert isinstance(phase_counts, dict)
    phase_counts["hover_hold"] = 0
    artifacts = report["random_morphology_takeoff_artifacts"]
    assert isinstance(artifacts, dict)
    del artifacts["backend"]

    result = random_morphology_takeoff_result_from_report(
        morphology,
        placement=placement,
        report=report,
        expected_backend_config_hash=_backend_config_hash(),
        expected_physical_model_hash=physical_model.stable_hash(),
        expected_collision_geometry_hash=collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        expected_config=RandomMorphologyTakeoffConfig(),
    )

    assert result.real_isaac_passed is False
    failures = result.metrics["random_morphology_takeoff_report_validation_failures"]
    assert "missing:random_morphology_takeoff_smoke_passed" in failures
    assert "invalid_phase_count:hover_hold" in failures
    assert "missing:random_morphology_takeoff_artifacts.backend" in failures


def test_takeoff_report_rejects_application_count_mismatch_and_log_length() -> None:
    physical_model, morphology = _sample(module_count=2)
    placement = compute_floor_contact_placement(
        morphology,
        physical_model,
        mesh_search_dirs=["module_urdf"],
    )
    report = _valid_takeoff_report(
        physical_model=physical_model,
        morphology=morphology,
        placement=placement,
    )
    report["random_morphology_takeoff_application_applied_target_count"] = 31
    observations = report["random_morphology_takeoff_runtime_observations"]
    assert isinstance(observations, list)
    observations.pop()

    result = random_morphology_takeoff_result_from_report(
        morphology,
        placement=placement,
        report=report,
        expected_backend_config_hash=_backend_config_hash(),
        expected_physical_model_hash=physical_model.stable_hash(),
        expected_collision_geometry_hash=collision_geometry_content_hash(
            physical_model, mesh_search_dirs=["module_urdf"]
        ),
        expected_config=RandomMorphologyTakeoffConfig(),
    )

    assert result.real_isaac_passed is False
    failures = result.metrics["random_morphology_takeoff_report_validation_failures"]
    assert "mismatch:random_morphology_takeoff_application_target_counts" in failures
    assert "length_mismatch:random_morphology_takeoff_runtime_observations" in failures


def test_takeoff_rejects_disconnected_graph_before_floor_or_isaac() -> None:
    physical_model, morphology = _sample(module_count=3)
    disconnected = type(morphology)(
        graph_id="disconnected",
        modules=morphology.modules,
        ports=morphology.ports,
        dock_edges=morphology.dock_edges[:1],
        robot_anchors=morphology.robot_anchors,
        control_groups=morphology.control_groups,
        base_module_id=morphology.base_module_id,
        is_closed_loop=False,
    )

    with pytest.raises(SchemaValidationError, match="N-1 dock edges"):
        compute_floor_contact_placement(
            disconnected,
            physical_model,
            mesh_search_dirs=["module_urdf"],
        )
