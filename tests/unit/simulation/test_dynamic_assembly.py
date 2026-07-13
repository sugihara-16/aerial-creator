from __future__ import annotations

import copy
import math
import sys
from dataclasses import replace

import pytest

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.dynamic_assembly import (
    DYNAMIC_ASSEMBLY_PHYSICAL_ACCEPTANCE_CONTRACT,
    DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
    DYNAMIC_ASSEMBLY_PROGRESS_PREFIX,
    DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION,
    DynamicAssemblyIsaacConfig,
    DynamicAssemblyIsaacEnv,
    _run_json_command,
    dynamic_assembly_progress_due,
    dynamic_assembly_report_failures,
    format_dynamic_assembly_progress,
)
from amsrr.simulation.dynamic_dock_constraint import (
    build_dynamic_dock_constraint_spec,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, IsaacLabBackendConfig


def _graph():
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return RandomConnectedMorphologyDistribution(model).sample(seed=2, module_count=2)


def _physical_config() -> DynamicAssemblyIsaacConfig:
    default = DynamicAssemblyIsaacConfig()
    return replace(
        default,
        mating_contact_mode=DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
        control_bridge=replace(
            default.control_bridge,
            require_selected_pair_contact=True,
        ),
    )


@pytest.mark.parametrize(
    ("event", "live_phase"),
    [
        ("staging", "staging"),
        ("axial_approach", "axial"),
        ("constraint_enabled", "fixed"),
        ("unload_dwell", "unload"),
        ("separation", "separation"),
    ],
)
def test_progress_line_exposes_live_phase_and_simulation_time(
    event: str,
    live_phase: str,
) -> None:
    line = format_dynamic_assembly_progress(event, 12.3456)

    assert line.startswith(DYNAMIC_ASSEMBLY_PROGRESS_PREFIX)
    assert "simulation_time=12.346s" in line
    assert f"phase={live_phase}" in line
    assert f"event={event}" in line


def test_progress_heartbeat_is_due_once_per_simulation_second() -> None:
    assert dynamic_assembly_progress_due(None, 0.0)
    assert not dynamic_assembly_progress_due(5.0, 5.999)
    assert dynamic_assembly_progress_due(5.0, 6.0)


@pytest.mark.parametrize(
    ("last_emit_time_s", "simulation_time_s", "interval_s"),
    [
        (0.0, -0.1, 1.0),
        (-0.1, 0.0, 1.0),
        (1.0, 0.5, 1.0),
        (0.0, 1.0, 0.0),
    ],
)
def test_progress_heartbeat_rejects_invalid_time_inputs(
    last_emit_time_s: float,
    simulation_time_s: float,
    interval_s: float,
) -> None:
    with pytest.raises(SchemaValidationError):
        dynamic_assembly_progress_due(
            last_emit_time_s,
            simulation_time_s,
            interval_s=interval_s,
        )


def test_json_command_forwards_progress_while_preserving_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = format_dynamic_assembly_progress("staging", 1.25)
    code = (
        "import json,sys;"
        f"print({progress!r}, file=sys.stderr, flush=True);"
        "print(json.dumps({'probe_passed': True}), flush=True)"
    )

    report = _run_json_command([sys.executable, "-c", code], timeout_s=5.0)

    assert report["probe_passed"] is True
    assert report["command_returncode"] == 0
    assert progress in report["command_stderr_tail"]
    assert progress in capsys.readouterr().err


def _raw_contact_evidence() -> dict[str, object]:
    return {
        "selected_pair_scope": "selected_dock_body_pair",
        "selected_pair_exact_body_match": True,
        "selected_contact": True,
        "selected_force_n": 1.0,
        "selected_penetration_m": 0.0001,
        "selected_raw_contact_count": 1,
        "selected_physical_patch_count": 1,
        "selected_contact_points_world": [[0.0, 0.0, 1.0]],
        "selected_contact_normals_world": [[0.0, 1.0, 0.0]],
        "selected_patch_forces_n": [1.0],
        "selected_patch_separations_m": [-0.0001],
        "selected_min_separation_m": -0.0001,
        "raw_contact_count": 1,
        "raw_contact_capacity": 8,
        "raw_contact_saturated": False,
        "raw_contact_valid": True,
        "raw_contact_nonfinite": False,
        "raw_contact_layout_invalid": False,
        "unintended_contact": False,
        "unintended_raw_contact_count": 0,
    }


def _guidance_contact_evidence(config) -> dict[str, object]:
    return {
        **_raw_contact_evidence(),
        "guidance_contact_valid": True,
        # Funnel-wall contact is deliberately well outside the 3 mm final gate.
        "guidance_axial_gap_m": 0.046,
        "guidance_transverse_error_m": 0.003,
        "guidance_attitude_error_rad": 0.01,
        "guidance_contact_max_axial_gap_m": config.guidance_contact_max_axial_gap_m,
        "guidance_contact_max_transverse_error_m": (
            config.guidance_contact_max_transverse_error_m
        ),
        "guidance_contact_max_attitude_error_rad": (
            config.guidance_contact_max_attitude_error_rad
        ),
        "time_s": 4.5,
        "leader_connect_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "follower_connect_pose_world": [0.046, 0.003, 1.0, 0.0, 0.0, 1.0, 0.0],
    }


def _final_seated_evidence(config) -> dict[str, object]:
    return {
        **_raw_contact_evidence(),
        "evidence_version": "final_seated_contact_v1",
        "selected_pair_contact_required": True,
        "selected_pair_contact_observed": True,
        "final_seated_valid": True,
        "leader_qp_feasible": True,
        "follower_qp_feasible": True,
        "axial_error_m": 0.001,
        "transverse_error_m": 0.001,
        "position_error_m": 0.0015,
        "attitude_error_rad": 0.001,
        "relative_linear_speed_mps": 0.001,
        "relative_angular_speed_radps": 0.001,
        "continuous_strict_dwell_s": config.control_bridge.selected_contact_dwell_s,
        "required_strict_dwell_s": config.control_bridge.selected_contact_dwell_s,
        "time_s": 4.6,
        "leader_connect_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "follower_connect_pose_world": [0.001, 0.001, 1.0, 0.0, 0.0, 1.0, 0.0],
        "leader_connect_twist_world": [0.0] * 6,
        "follower_connect_twist_world": [0.001, 0.0, 0.0, 0.0, 0.0, 0.001],
    }


def _dock_collision_evidence() -> dict[str, object]:
    return {
        "requested_collision_type": "Convex Decomposition",
        "requested_approximation_token": "convexDecomposition",
        "physx_convex_decomposition_api_verified": True,
        "max_convex_hulls": 128,
        "shrink_wrap": True,
        "authored_prim_count": 4,
        "authored_prim_paths": [f"layer.usd:/dock/collider_{index}" for index in range(4)],
        "composed_prim_count": 4,
        "composed_prim_paths": [f"/World/dock/collider_{index}" for index in range(4)],
        "original_approximation_tokens": {
            f"collider_{index}": "convexHull" for index in range(4)
        },
        "verified": True,
    }


def _axial_selected_joint_evidence(
    *,
    leader_module_id: int,
    follower_module_id: int,
) -> dict[str, object]:
    def module_evidence(module_id: int, role: str) -> dict[str, object]:
        return {
            "role": role,
            "joint_id": f"{role}_dock_mech_joint1",
            "resolved_joint_name": f"module_{module_id}__{role}_dock_mech_joint1",
            "max_abs_joint_position_target_rad": 0.0,
            "max_abs_joint_velocity_target_radps": 0.0,
            "max_abs_joint_effort_bias_target_nm": 0.0,
            "max_abs_measured_joint_position_rad": 0.02,
            "max_abs_measured_joint_velocity_radps": 0.04,
            "max_abs_joint_computed_torque_nm": 0.50,
            "max_abs_joint_applied_torque_nm": 0.40,
            "max_selected_body_minus_root_angular_speed_radps": 0.04,
            "first_root_pose_world": [
                float(module_id),
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "last_root_pose_world": [
                float(module_id) + (0.0 if role == "leader" else 0.1),
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "first_selected_body_pose_in_root": [
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "last_selected_body_pose_in_root": [
                0.2,
                0.0,
                0.0,
                0.0,
                0.01,
                0.0,
                0.99995,
            ],
        }

    return {
        "evidence_version": "axial_selected_dock_joint_zero_target_v1",
        "sample_count": 60,
        "all_targets_zero": True,
        "by_module": {
            str(leader_module_id): module_evidence(leader_module_id, "leader"),
            str(follower_module_id): module_evidence(
                follower_module_id,
                "follower",
            ),
        },
    }


def _report(graph, config):
    phases = [
        "floor_settle",
        "preflight_vectoring",
        "preflight_vectoring_ready",
        "takeoff",
        "hover_acquisition",
        "hover_acquired",
        "staging",
        "prealign_dwell",
        "axial_approach",
        "fix_ready",
        "constraint_enabled",
        "verify",
        "constraint_verified",
        "attach_handover",
        "attached_hold",
        "control_graph_split",
        "split_handover",
        "unload_dwell",
        "constraint_removed",
        "separation",
        "collision_filter_removed",
        "post_release_hold",
        "complete",
    ]
    phase_times = {
        "floor_settle": 0.0,
        "preflight_vectoring": 1.0,
        "preflight_vectoring_ready": 1.1,
        "takeoff": 1.1,
        "hover_acquisition": 3.1,
        "hover_acquired": 3.6,
        "staging": 3.6,
        "prealign_dwell": 4.0,
        "axial_approach": 4.3,
        "fix_ready": 4.6,
        "constraint_enabled": 4.6,
        "verify": 4.605,
        "constraint_verified": 4.85,
        "attach_handover": 4.85,
        "attached_hold": 5.10,
        "control_graph_split": 6.10,
        "split_handover": 6.10,
        "unload_dwell": 6.35,
        "constraint_removed": 6.45,
        "separation": 6.45,
        "collision_filter_removed": 10.45,
        "post_release_hold": 10.45,
        "complete": 11.45,
    }
    edge = graph.dock_edges[0]
    leader_module_id = graph.base_module_id
    follower_module_id = (
        edge.dst_module_id
        if edge.src_module_id == leader_module_id
        else edge.src_module_id
    )
    constraint_spec = build_dynamic_dock_constraint_spec(
        graph,
        build_physical_model_from_config("configs/robot/robot_model.yaml"),
        edge_id=edge.edge_id,
        leader_module_id=leader_module_id,
        follower_module_id=follower_module_id,
        leader_body_path="/World/Assembly/Leader/dock",
        follower_body_path="/World/Assembly/Follower/dock",
    )
    floor_steps = int(config.floor_settle_required_dwell_s / config.simulation_dt_s)
    attached_steps = int(config.attached_hold_s / config.simulation_dt_s)
    post_steps = int(config.post_release_hold_s / config.simulation_dt_s)
    expected_step_types = ["move_to_staging", "align_ports", "dock", "verify_attach"]

    def event_metrics(phase: str) -> dict[str, object]:
        if phase == "complete":
            return {"passed": True}
        if phase == "collision_filter_removed":
            return {
                "selected_body_clearance_m": 0.04,
                "separation_gap_m": config.separation_distance_m,
                "separation_steps": int(
                    math.ceil(
                        config.separation_distance_m
                        / config.separation_speed_mps
                        / config.simulation_dt_s
                    )
                ),
            }
        if phase == "post_release_hold":
            return {
                "stable_dwell_s": post_steps * config.simulation_dt_s,
                "observed_steps": post_steps,
            }
        if phase == "preflight_vectoring_ready":
            return {"dwell_s": config.preflight_feasible_dwell_s}
        if phase == "hover_acquired":
            return {
                "dwell_s": config.hover_acquisition_dwell_s,
                "max_position_error_m": 0.01,
                "max_attitude_error_rad": 0.01,
                "max_linear_speed_mps": 0.01,
                "max_angular_speed_radps": 0.01,
                "max_dock_joint_position_rad": 0.001,
                "max_dock_joint_speed_radps": 0.001,
            }
        return {}

    return {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_probe_passed": True,
        "dynamic_assembly_roundtrip": True,
        "dynamic_assembly_attach_passed": True,
        "dynamic_assembly_detach_passed": True,
        "dynamic_assembly_passed": True,
        "dynamic_assembly_constraint_identity_verified": True,
        "dynamic_assembly_constraint_removed": True,
        "dynamic_assembly_external_fixed_joint": True,
        "dynamic_assembly_constraint_excluded_from_articulation": True,
        "dynamic_assembly_selected_pair_contact_observed": True,
        "dynamic_assembly_guidance_contact_observed": True,
        "dynamic_assembly_physical_mating_contact_claimed": True,
        "dynamic_assembly_physical_attach_passed": True,
        "dynamic_assembly_filter_fallback_attach_passed": False,
        "dynamic_assembly_selected_surface_contact_observed": True,
        "dynamic_assembly_selected_pair_filter_applied": True,
        "dynamic_assembly_selected_pair_filter_apply_verified": True,
        "dynamic_assembly_mating_filter_evidence": None,
        "dynamic_assembly_selected_pair_filter_removed": True,
        "dynamic_assembly_selected_pair_filter_remove_verified": True,
        "dynamic_assembly_selected_pair_filter_removal_clearance_m": 0.04,
        "dynamic_assembly_final_selected_body_clearance_m": 0.05,
        "dynamic_assembly_post_unfilter_selected_contact_count": 0,
        "dynamic_assembly_post_unfilter_raw_invalid_count": 0,
        # Legacy surface keys are retained as diagnostics, not acceptance gates.
        "dynamic_assembly_first_selected_contact_evidence": {"diagnostic": True},
        "dynamic_assembly_first_guidance_contact_evidence": (
            _guidance_contact_evidence(config)
        ),
        "dynamic_assembly_final_seated_evidence": _final_seated_evidence(config),
        "dynamic_assembly_floor_initialization_verified": True,
        "dynamic_assembly_floor_initialization_evidence": {
            "explicit_zero_joint_position_target": True,
            "explicit_zero_joint_velocity_target": True,
            "explicit_zero_joint_effort_bias": True,
            "continuous_dwell_steps": floor_steps,
            "required_dwell_steps": floor_steps,
            "max_contact_force_n_by_module": {"0": 1.0, "1": 1.0},
            "verified": True,
        },
        "dynamic_assembly_attach_handover_completed": True,
        "dynamic_assembly_attached_stability_verified": True,
        "dynamic_assembly_attached_selected_pair_contact_free": True,
        "dynamic_assembly_attached_stable_steps": attached_steps,
        "dynamic_assembly_attached_required_stable_steps": attached_steps,
        "dynamic_assembly_attached_max_metrics": {
            "position_error_m": 0.01,
            "attitude_error_rad": 0.01,
            "linear_speed_mps": 0.01,
            "angular_speed_radps": 0.01,
            "connect_position_error_m": 0.001,
            "connect_axial_error_m": 0.001,
            "connect_transverse_error_m": 0.001,
            "connect_attitude_error_rad": 0.001,
            "connect_relative_linear_speed_mps": 0.001,
            "connect_relative_angular_speed_radps": 0.001,
            "dock_joint_position_rad": 0.001,
            "dock_joint_speed_radps": 0.001,
        },
        "dynamic_assembly_split_handover_completed": True,
        "dynamic_assembly_controller_handover_samples": [
            {
                "direction": "components_to_assembled",
                "time_s": 4.85,
                "alpha": 0.02,
                "source_qp_feasible": True,
                "target_qp_feasible": True,
            },
            {
                "direction": "components_to_assembled",
                "time_s": 5.095,
                "alpha": 1.0,
                "source_qp_feasible": True,
                "target_qp_feasible": True,
            },
            {
                "direction": "assembled_to_components",
                "time_s": 6.10,
                "alpha": 0.02,
                "source_qp_feasible": True,
                "target_qp_feasible": True,
            },
            {
                "direction": "assembled_to_components",
                "time_s": 6.345,
                "alpha": 1.0,
                "source_qp_feasible": True,
                "target_qp_feasible": True,
            },
        ],
        "dynamic_assembly_constraint_disabled_verified": True,
        "dynamic_assembly_unload_ready": True,
        "dynamic_assembly_follower_external_contact_free_during_unload": True,
        "dynamic_assembly_follower_external_contact_max_force_n": 0.0,
        "dynamic_assembly_follower_external_contact_invalid_during_unload_count": 0,
        "dynamic_assembly_follower_external_contact_raw_patch_count_during_unload": 0,
        "dynamic_assembly_follower_external_contact_scope": (
            "follower_component_all_external_contacts"
        ),
        "dynamic_assembly_unload_estimate": {
            "edge_id": edge.edge_id,
            "follower_module_ids": [follower_module_id],
            "valid": True,
            "failure_reason": None,
            "wrench_follower_com_body": [0.0] * 6,
            "wrench_follower_dock_frame": [0.0] * 6,
            "force_norm_n": 0.1,
            "torque_norm_nm": 0.01,
        },
        "dynamic_assembly_unload_decision": {
            "ready_to_release": True,
            "consecutive_unload_steps": config.detach_unload.unload_dwell_steps,
            "failure_reasons": [],
            "metrics": {
                "cut_force_norm_n": 0.1,
                "cut_torque_norm_nm": 0.01,
                "relative_position_error_m": 0.001,
                "relative_rotation_error_rad": 0.001,
                "relative_linear_speed_mps": 0.001,
                "relative_angular_speed_radps": 0.001,
                "required_unload_dwell_steps": float(
                    config.detach_unload.unload_dwell_steps
                ),
            },
        },
        "dynamic_assembly_control_graph_split_before_release": True,
        "dynamic_assembly_post_release_stable": True,
        "dynamic_assembly_final_separation_gap_m": config.separation_distance_m,
        "dynamic_assembly_post_release_min_separation_gap_m": (
            config.separation_distance_m
        ),
        "dynamic_assembly_post_release_min_selected_body_clearance_m": 0.05,
        "dynamic_assembly_post_release_stable_dwell_steps": post_steps,
        "dynamic_assembly_post_release_required_dwell_steps": post_steps,
        "dynamic_assembly_post_release_max_metrics": {
            "position_error_m": 0.01,
            "attitude_error_rad": 0.01,
            "linear_speed_mps": 0.01,
            "angular_speed_radps": 0.01,
            "dock_joint_position_rad": 0.001,
            "dock_joint_speed_radps": 0.001,
        },
        "dynamic_assembly_preflight_vectoring_ready": True,
        "dynamic_assembly_hover_acquired": True,
        "dynamic_assembly_finite_state": True,
        "command_returncode": 0,
        "dynamic_assembly_version": config.version,
        "dynamic_assembly_collision_type": config.collision_type,
        "dynamic_assembly_mating_contact_mode": config.mating_contact_mode,
        "dynamic_assembly_acceptance_contract": (
            DYNAMIC_ASSEMBLY_PHYSICAL_ACCEPTANCE_CONTRACT
        ),
        "dynamic_assembly_dock_collision_approximation_verified": True,
        "dynamic_assembly_dock_collision_approximation_token": "convexDecomposition",
        "dynamic_assembly_dock_collision_composed_prim_count": 4,
        "dynamic_assembly_dock_collision_approximation_evidence": (
            _dock_collision_evidence()
        ),
        "dynamic_assembly_force_usd_conversion": True,
        "dynamic_assembly_acceptance_gate": config.acceptance_gate,
        "dynamic_assembly_solver_position_iteration_count": (
            config.solver_position_iteration_count
        ),
        "dynamic_assembly_solver_velocity_iteration_count": (
            config.solver_velocity_iteration_count
        ),
        "dynamic_assembly_dock_drive_stiffness_nm_per_rad": 200.0,
        "dynamic_assembly_dock_drive_damping_nms_per_rad": 2.0,
        "dynamic_assembly_dock_effort_limit_sim_nm": 4.1,
        "dynamic_assembly_dock_velocity_limit_sim_radps": 3.0,
        "dynamic_assembly_graph_id": graph.graph_id,
        "dynamic_assembly_graph_hash": graph.stable_hash(),
        "dynamic_assembly_config_hash": config.stable_hash(),
        "dynamic_assembly_backend_config_hash": "a" * 64,
        "dynamic_assembly_physical_model_hash": "b" * 64,
        "dynamic_assembly_collision_geometry_content_hash": "d" * 64,
        "generated_urdf_sha256": "c" * 64,
        "generated_usd_sha256": "e" * 64,
        "generated_usd_bundle_hash": "f" * 64,
        "dynamic_assembly_module_count": 2,
        "dynamic_assembly_edge_id": edge.edge_id,
        "dynamic_assembly_leader_module_id": leader_module_id,
        "dynamic_assembly_follower_module_id": follower_module_id,
        "dynamic_assembly_axial_selected_joint_evidence": (
            _axial_selected_joint_evidence(
                leader_module_id=leader_module_id,
                follower_module_id=follower_module_id,
            )
        ),
        "dynamic_assembly_constraint_version": "external_connect_frame_fixed_joint_v1",
        "dynamic_assembly_constraint_spec": constraint_spec.to_dict(),
        "dynamic_assembly_constraint_identity_failures": [],
        "dynamic_assembly_qpid_joint_dynamics_unaware": True,
        "dynamic_assembly_dock_joint_latch_semantics": False,
        "dynamic_assembly_qp_infeasible_count": 0,
        "dynamic_assembly_unintended_contact_count": 0,
        "dynamic_assembly_missing_actuator_count": 0,
        "dynamic_assembly_unsupported_actuator_count": 0,
        "dynamic_assembly_application_unresolved_target_count": 0,
        "dynamic_assembly_clipped_target_count": 0,
        "dynamic_assembly_constraint_identity_failure_count": 0,
        "dynamic_assembly_filter_fallback_selected_contact_violation_count": 0,
        "dynamic_assembly_assembly_run_report": {
            "plan": {
                "target_graph_id": graph.graph_id,
                "steps": [
                    {"step_id": index, "step_type": step_type}
                    for index, step_type in enumerate(expected_step_types)
                ],
            },
            "success": True,
            "state_matches_target": True,
            "aborted": False,
            "failure_reason": None,
            "failures": [],
            "completed_step_count": 4,
            "attached_edge_count": 1,
            "target_edge_count": 1,
            "retry_count": 0,
            "abort_count": 0,
            "executed_step_types": expected_step_types,
            "step_results": [
                {"step_id": index, "success": True}
                for index in range(4)
            ],
            "metrics": {
                "target_module_count": 2.0,
                "assembled_module_count": 2.0,
                "target_edge_count": 1.0,
                "attached_edge_count": 1.0,
                "module_set_matches_target": 1.0,
                "dock_edge_set_matches_target": 1.0,
                "port_occupancy_matches_target": 1.0,
                "state_matches_target": 1.0,
            },
        },
        "dynamic_assembly_events": [
            {
                "time_s": phase_times[phase],
                "phase": phase,
                "metrics": event_metrics(phase),
            }
            for phase in phases
        ],
    }


def test_config_roundtrip_and_dry_probe_command() -> None:
    graph = _graph()
    config = _physical_config()
    backend = IsaacLabBackend(
        IsaacLabBackendConfig(
            isaaclab_path="/opt/IsaacLab",
            holon_urdf_path="assets/robots/holon/holon.urdf",
        )
    )
    env = DynamicAssemblyIsaacEnv(config=config, backend=backend)

    result = env.run(graph, dry_run=True)

    assert type(config).from_json(config.to_json()).to_dict() == config.to_dict()
    assert result.version == DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION
    assert result.dry_run is True
    command = result.report["probe_command"]
    assert "--force-convert" in command
    assert "--convert-if-missing" not in command
    assert "--dynamic-assembly-roundtrip" in command
    assert "--dynamic-assembly-graph-json" in command
    assert "--dynamic-assembly-config-json" in command
    assembly_executor_timeout_s = (
        config.control_bridge.step_timeout_s
        + max(5.0, config.control_bridge.prealign_dwell_s + 2.0)
        + config.control_bridge.axial_approach_timeout_s
        + max(5.0, config.constraint_verify_dwell_s + 2.0)
    )
    expected_steps = math.ceil(
        (
            config.floor_settle_duration_s
            + config.preflight_vectoring_timeout_s
            + config.takeoff_duration_s
            + config.takeoff_hold_s
            + config.hover_acquisition_timeout_s
            + assembly_executor_timeout_s
            + 2.0 * config.controller_handover_blend_s
            + config.constraint_verify_dwell_s
            + config.attached_hold_s
            + 2.0 * config.post_release_hold_s
            + 2.0
            * config.separation_distance_m
            / config.separation_speed_mps
            + 5.0
        )
        / config.simulation_dt_s
    )
    steps_index = command.index("--steps")
    assert int(command[steps_index + 1]) == expected_steps


def test_report_gate_accepts_complete_roundtrip_and_rejects_mislabel() -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)

    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    ) == []

    report["dynamic_assembly_constraint_excluded_from_articulation"] = False
    report["dynamic_assembly_dock_joint_latch_semantics"] = True
    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )
    assert "mismatch:dynamic_assembly_constraint_excluded_from_articulation" in failures
    assert "mismatch:dynamic_assembly_dock_joint_latch_semantics" in failures


@pytest.mark.parametrize(
    "key",
    [
        "dynamic_assembly_guidance_contact_observed",
        "dynamic_assembly_dock_collision_approximation_verified",
        "dynamic_assembly_selected_pair_filter_applied",
        "dynamic_assembly_selected_pair_filter_removed",
        "dynamic_assembly_floor_initialization_verified",
        "dynamic_assembly_attach_handover_completed",
        "dynamic_assembly_split_handover_completed",
        "dynamic_assembly_unload_ready",
        "dynamic_assembly_follower_external_contact_free_during_unload",
    ],
)
def test_report_gate_rejects_missing_required_runtime_evidence(key: str) -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)
    report[key] = False

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert f"mismatch:{key}" in failures


def test_report_gate_rejects_out_of_order_phases() -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)
    events = report["dynamic_assembly_events"]
    fix_index = next(
        index for index, event in enumerate(events) if event["phase"] == "fix_ready"
    )
    enable_index = next(
        index
        for index, event in enumerate(events)
        if event["phase"] == "constraint_enabled"
    )
    events[fix_index], events[enable_index] = events[enable_index], events[fix_index]

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert any(
        failure.startswith("invalid:dynamic_assembly_phase_order:")
        for failure in failures
    )


def test_report_gate_rejects_hash_and_return_code_failures() -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)
    report["dynamic_assembly_config_hash"] = "d" * 64
    report["dynamic_assembly_backend_config_hash"] = "short"
    report["dynamic_assembly_physical_model_hash"] = None
    report["dynamic_assembly_collision_geometry_content_hash"] = "z" * 64
    report["generated_urdf_sha256"] = "not-a-sha256"
    report["generated_usd_sha256"] = "not-a-sha256"
    report["generated_usd_bundle_hash"] = "not-a-sha256"
    report["dynamic_assembly_acceptance_gate"] = "attach_only"
    report["command_returncode"] = 1

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert "mismatch:dynamic_assembly_config_hash" in failures
    assert "invalid:dynamic_assembly_backend_config_hash" in failures
    assert "invalid:dynamic_assembly_physical_model_hash" in failures
    assert "invalid:dynamic_assembly_collision_geometry_content_hash" in failures
    assert "invalid:generated_urdf_sha256" in failures
    assert "invalid:generated_usd_sha256" in failures
    assert "invalid:generated_usd_bundle_hash" in failures
    assert "mismatch:dynamic_assembly_acceptance_gate" in failures
    assert "mismatch:command_returncode" in failures


@pytest.mark.parametrize(
    ("path", "replacement", "expected_failure"),
    [
        (
            ("dynamic_assembly_floor_initialization_evidence", "continuous_dwell_steps"),
            0,
            "invalid:dynamic_assembly_floor_initialization_evidence",
        ),
        (
            ("dynamic_assembly_events", 2, "metrics", "dwell_s"),
            0.0,
            "invalid:dynamic_assembly_preflight_evidence",
        ),
        (
            ("dynamic_assembly_events", 5, "metrics", "max_position_error_m"),
            1.0,
            "invalid:dynamic_assembly_hover_evidence",
        ),
        (
            (
                "dynamic_assembly_first_guidance_contact_evidence",
                "raw_contact_valid",
            ),
            False,
            "invalid:dynamic_assembly_first_guidance_contact_evidence",
        ),
        (
            (
                "dynamic_assembly_final_seated_evidence",
                "continuous_strict_dwell_s",
            ),
            0.0,
            "invalid:dynamic_assembly_final_seated_evidence",
        ),
        (
            ("dynamic_assembly_dock_collision_approximation_evidence", "verified"),
            False,
            "invalid:dynamic_assembly_dock_collision_approximation_evidence",
        ),
        (
            ("dynamic_assembly_constraint_spec", "edge_id"),
            999,
            "invalid:dynamic_assembly_constraint_evidence",
        ),
        (
            ("dynamic_assembly_assembly_run_report", "success"),
            False,
            "invalid:dynamic_assembly_assembly_run_report",
        ),
        (
            ("dynamic_assembly_controller_handover_samples", 1, "alpha"),
            0.9,
            "invalid:dynamic_assembly_controller_handover_samples",
        ),
        (
            ("dynamic_assembly_attached_max_metrics", "position_error_m"),
            1.0,
            "invalid:dynamic_assembly_attached_stability_evidence",
        ),
        (
            ("dynamic_assembly_unload_estimate", "force_norm_n"),
            1.0,
            "invalid:dynamic_assembly_unload_evidence",
        ),
        (
            ("dynamic_assembly_final_separation_gap_m",),
            0.01,
            "invalid:dynamic_assembly_separation_evidence",
        ),
        (
            ("dynamic_assembly_selected_pair_filter_removal_clearance_m",),
            0.0,
            "invalid:dynamic_assembly_filter_clearance_evidence",
        ),
        (
            (
                "dynamic_assembly_events",
                20,
                "metrics",
                "separation_steps",
            ),
            1,
            "invalid:dynamic_assembly_filter_clearance_evidence",
        ),
        (
            ("dynamic_assembly_post_unfilter_selected_contact_count",),
            1,
            "invalid:dynamic_assembly_post_unfilter_contact_evidence",
        ),
        (
            ("dynamic_assembly_post_release_stable_dwell_steps",),
            0,
            "invalid:dynamic_assembly_post_release_stability_evidence",
        ),
        (
            (
                "dynamic_assembly_events",
                21,
                "metrics",
                "stable_dwell_s",
            ),
            0.0,
            "invalid:dynamic_assembly_post_release_stability_evidence",
        ),
        (
            (
                "dynamic_assembly_events",
                21,
                "metrics",
                "observed_steps",
            ),
            401,
            "invalid:dynamic_assembly_post_release_stability_evidence",
        ),
    ],
)
def test_report_gate_rejects_inconsistent_nested_evidence(
    path: tuple[str | int, ...],
    replacement: object,
    expected_failure: str,
) -> None:
    graph = _graph()
    config = _physical_config()
    report = copy.deepcopy(_report(graph, config))
    _replace_path(report, path, replacement)

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert expected_failure in failures


def test_report_gate_rejects_filter_removal_before_nominal_separation() -> None:
    graph = _graph()
    config = _physical_config()
    report = copy.deepcopy(_report(graph, config))
    separation_time = report["dynamic_assembly_events"][19]["time_s"]
    report["dynamic_assembly_events"][20]["time_s"] = separation_time + 0.60

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert "invalid:dynamic_assembly_filter_clearance_evidence" in failures


def test_report_gate_rejects_nonmonotonic_duplicate_and_false_terminal_events() -> None:
    graph = _graph()
    config = _physical_config()

    nonmonotonic = copy.deepcopy(_report(graph, config))
    takeoff_event = next(
        event
        for event in nonmonotonic["dynamic_assembly_events"]
        if event["phase"] == "takeoff"
    )
    takeoff_event["time_s"] = 0.5
    assert "invalid:dynamic_assembly_event_timestamps" in dynamic_assembly_report_failures(
        nonmonotonic,
        morphology_graph=graph,
        config=config,
    )

    duplicate = copy.deepcopy(_report(graph, config))
    constraint_index = next(
        index
        for index, event in enumerate(duplicate["dynamic_assembly_events"])
        if event["phase"] == "constraint_enabled"
    )
    duplicate["dynamic_assembly_events"].insert(
        constraint_index + 1,
        copy.deepcopy(duplicate["dynamic_assembly_events"][constraint_index]),
    )
    assert (
        "invalid:dynamic_assembly_phase_cardinality:constraint_enabled"
        in dynamic_assembly_report_failures(
            duplicate,
            morphology_graph=graph,
            config=config,
        )
    )

    false_terminal = copy.deepcopy(_report(graph, config))
    false_terminal["dynamic_assembly_events"][-1]["metrics"]["passed"] = False
    assert "invalid:dynamic_assembly_complete_event" in dynamic_assembly_report_failures(
        false_terminal,
        morphology_graph=graph,
        config=config,
    )


def test_report_gate_rejects_short_lifecycle_dwell_despite_success_flags() -> None:
    graph = _graph()
    config = _physical_config()
    report = copy.deepcopy(_report(graph, config))
    post_event = next(
        event
        for event in report["dynamic_assembly_events"]
        if event["phase"] == "post_release_hold"
    )
    complete_event = report["dynamic_assembly_events"][-1]
    complete_event["time_s"] = post_event["time_s"] + 0.1

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert (
        "invalid:dynamic_assembly_lifecycle_timing:post_release_hold:complete"
        in failures
    )


def test_first_funnel_contact_can_precede_the_strict_final_gate() -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)
    guidance = report["dynamic_assembly_first_guidance_contact_evidence"]

    assert guidance["guidance_axial_gap_m"] == 0.046
    assert guidance["guidance_axial_gap_m"] > config.control_bridge.fix_axial_tolerance_m
    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    ) == []


def test_legacy_surface_diagnostic_is_not_an_attach_gate() -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)
    report["dynamic_assembly_selected_surface_contact_observed"] = False
    report["dynamic_assembly_first_selected_contact_evidence"] = None

    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    ) == []


@pytest.mark.parametrize(
    ("path", "replacement", "expected_failure"),
    [
        (
            ("dynamic_assembly_first_guidance_contact_evidence", "raw_contact_saturated"),
            True,
            "invalid:dynamic_assembly_first_guidance_contact_evidence",
        ),
        (
            ("dynamic_assembly_first_guidance_contact_evidence", "selected_force_n"),
            31.0,
            "invalid:dynamic_assembly_first_guidance_contact_evidence",
        ),
        (
            ("dynamic_assembly_first_guidance_contact_evidence", "unintended_contact"),
            True,
            "invalid:dynamic_assembly_first_guidance_contact_evidence",
        ),
        (
            ("dynamic_assembly_final_seated_evidence", "axial_error_m"),
            0.004,
            "invalid:dynamic_assembly_final_seated_evidence",
        ),
        (
            ("dynamic_assembly_dock_collision_approximation_token",),
            "convexHull",
            "mismatch:dynamic_assembly_dock_collision_approximation_token",
        ),
        (
            ("dynamic_assembly_dock_collision_composed_prim_count",),
            0,
            "invalid:dynamic_assembly_dock_collision_composed_prim_count",
        ),
    ],
)
def test_funnel_contact_contract_fails_closed(
    path: tuple[str | int, ...],
    replacement: object,
    expected_failure: str,
) -> None:
    graph = _graph()
    config = _physical_config()
    report = copy.deepcopy(_report(graph, config))
    _replace_path(report, path, replacement)

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert expected_failure in failures


def test_report_gate_checks_expected_backend_hash() -> None:
    graph = _graph()
    config = _physical_config()
    report = _report(graph, config)

    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
        backend_config_hash="a" * 64,
    ) == []

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
        backend_config_hash="e" * 64,
    )
    assert "mismatch:dynamic_assembly_backend_config_hash" in failures


def _replace_path(container: object, path: tuple[str | int, ...], value: object) -> None:
    target = container
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
