from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.dynamic_assembly import (
    DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE,
    DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT,
    DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE,
    DYNAMIC_ASSEMBLY_PHYSICAL_ACCEPTANCE_CONTRACT,
    DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
    DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE,
    DynamicAssemblyIsaacConfig,
    DynamicAssemblyIsaacEnv,
    dynamic_assembly_report_failures,
)
from amsrr.simulation.dynamic_dock_constraint import DYNAMIC_DOCK_CONSTRAINT_VERSION
from amsrr.simulation.dynamic_dock_constraint import build_dynamic_dock_constraint_spec
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, IsaacLabBackendConfig


def _graph():
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return RandomConnectedMorphologyDistribution(model).sample(seed=2, module_count=2)


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


def _attach_report(graph, config):
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
        "complete": 6.10,
    }
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    edge = graph.dock_edges[0]
    leader_module_id = graph.base_module_id
    follower_module_id = (
        edge.dst_module_id
        if edge.src_module_id == leader_module_id
        else edge.src_module_id
    )
    constraint_spec = build_dynamic_dock_constraint_spec(
        graph,
        model,
        edge_id=edge.edge_id,
        leader_module_id=leader_module_id,
        follower_module_id=follower_module_id,
        leader_body_path="/World/Assembly/Leader/dock",
        follower_body_path="/World/Assembly/Follower/dock",
    )
    floor_steps = int(config.floor_settle_required_dwell_s / config.simulation_dt_s)
    attached_steps = int(config.attached_hold_s / config.simulation_dt_s)
    expected_step_types = ["move_to_staging", "align_ports", "dock", "verify_attach"]

    def event_metrics(phase: str) -> dict[str, object]:
        if phase == "complete":
            return {"passed": True}
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
        "dynamic_assembly_roundtrip": False,
        "dynamic_assembly_attach_passed": True,
        "dynamic_assembly_detach_passed": False,
        "dynamic_assembly_passed": True,
        "dynamic_assembly_constraint_identity_verified": True,
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
        ],
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
        "dynamic_assembly_backend_config_hash": "b" * 64,
        "dynamic_assembly_physical_model_hash": model.stable_hash(),
        "dynamic_assembly_collision_geometry_content_hash": (
            collision_geometry_content_hash(
                model,
                mesh_search_dirs=("module_urdf", "module_urdf/mesh"),
            )
        ),
        "generated_urdf_sha256": "d" * 64,
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
        "dynamic_assembly_constraint_version": DYNAMIC_DOCK_CONSTRAINT_VERSION,
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


def _fallback_config() -> DynamicAssemblyIsaacConfig:
    return DynamicAssemblyIsaacConfig(
        acceptance_gate=DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE,
    )


def _physical_config(
    *,
    acceptance_gate: str = DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE,
) -> DynamicAssemblyIsaacConfig:
    base = DynamicAssemblyIsaacConfig(acceptance_gate=acceptance_gate)
    return replace(
        base,
        mating_contact_mode=DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
        control_bridge=replace(
            base.control_bridge,
            require_selected_pair_contact=True,
        ),
    )


def _fallback_attach_report(graph, config) -> dict[str, object]:
    report = _attach_report(graph, config)
    constraint = report["dynamic_assembly_constraint_spec"]
    leader_path = constraint["leader_body_path"]
    follower_path = constraint["follower_body_path"]
    report.update(
        {
            "dynamic_assembly_selected_pair_contact_observed": False,
            "dynamic_assembly_guidance_contact_observed": False,
            "dynamic_assembly_selected_surface_contact_observed": False,
            "dynamic_assembly_first_selected_contact_evidence": None,
            "dynamic_assembly_first_guidance_contact_evidence": None,
            "dynamic_assembly_physical_mating_contact_claimed": False,
            "dynamic_assembly_physical_attach_passed": False,
            "dynamic_assembly_filter_fallback_attach_passed": True,
            "dynamic_assembly_acceptance_contract": (
                DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT
            ),
            "dynamic_assembly_config_hash": config.stable_hash(),
            "dynamic_assembly_mating_filter_evidence": {
                "evidence_version": (
                    DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT
                ),
                "mating_contact_mode": DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE,
                "scope": "selected_dock_body_pair_only",
                "apply_phase": "prealign_dwell",
                "time_s": 4.0,
                "command_index": 10,
                "applied_before_first_axial_physics_step": True,
                "apply_verified": True,
                "environment_collisions_preserved": True,
                "other_body_pair_collisions_preserved": True,
                "leader_body_path": leader_path,
                "follower_body_path": follower_path,
                "leader_body_prim_valid": True,
                "follower_body_prim_valid": True,
                "leader_is_rigid_body": True,
                "follower_is_rigid_body": True,
                "leader_targets_before": ["/World/UnrelatedLeaderTarget"],
                "follower_targets_before": ["/World/UnrelatedFollowerTarget"],
                "leader_targets_after": [
                    "/World/UnrelatedLeaderTarget",
                    follower_path,
                ],
                "follower_targets_after": ["/World/UnrelatedFollowerTarget"],
                "added_leader_targets": [follower_path],
                "removed_leader_targets": [],
                "added_follower_targets": [],
                "removed_follower_targets": [],
                "selected_contact_count_after_filter": 0,
                "selected_contact_violation_count": 0,
            },
            "dynamic_assembly_final_seated_evidence": {
                "evidence_version": "final_seated_alignment_v1",
                "selected_pair_scope": "selected_dock_body_pair",
                "selected_pair_contact_required": False,
                "selected_pair_contact_observed": False,
                "final_seated_valid": True,
                "leader_qp_feasible": True,
                "follower_qp_feasible": True,
                "axial_error_m": 0.001,
                "transverse_error_m": 0.001,
                "position_error_m": 0.0015,
                "attitude_error_rad": 0.001,
                "relative_linear_speed_mps": 0.001,
                "relative_angular_speed_radps": 0.001,
                "continuous_strict_dwell_s": (
                    config.control_bridge.selected_contact_dwell_s
                ),
                "required_strict_dwell_s": (
                    config.control_bridge.selected_contact_dwell_s
                ),
                "time_s": 4.6,
                "leader_connect_pose_world": [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "follower_connect_pose_world": [
                    0.001,
                    0.001,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                "leader_connect_twist_world": [0.0] * 6,
                "follower_connect_twist_world": [
                    0.001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.001,
                ],
            },
        }
    )
    return report


def _backend():
    return IsaacLabBackend(
        IsaacLabBackendConfig(
            isaaclab_path="/opt/IsaacLab",
            holon_urdf_path="assets/robots/holon/holon.urdf",
        )
    )


def test_attach_only_gate_passes_without_detach_evidence() -> None:
    graph = _graph()
    config = _physical_config()
    assert DynamicAssemblyIsaacConfig.from_json(config.to_json()).acceptance_gate == (
        DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE
    )
    report = _attach_report(graph, config)
    backend = _backend()
    report["dynamic_assembly_backend_config_hash"] = backend.config.stable_hash()
    env = DynamicAssemblyIsaacEnv(
        config=config,
        backend=backend,
        command_executor=lambda command, timeout_s: report,
        verify_local_artifacts=False,
    )

    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    ) == []
    result = env.run(graph, dry_run=False, check_availability=False)
    result.validate()
    restored = type(result).from_json(result.to_json())
    restored.validate()

    assert result.acceptance_gate == DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE
    assert result.attach_passed is True
    assert result.detach_passed is False
    assert result.passed is True
    assert restored.passed is True


def test_production_attach_only_success_report_includes_assembly_run_report() -> None:
    probe_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "p4_control_holon_spawn_probe.py"
    )
    syntax_tree = ast.parse(
        probe_path.read_text(encoding="utf-8"),
        filename=str(probe_path),
    )
    probe_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_dynamic_assembly_roundtrip_probe"
    )
    attach_only_success_returns: list[dict[str, ast.expr]] = []
    for node in ast.walk(probe_function):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        literal_fields = {
            key.value: value
            for key, value in zip(node.value.keys, node.value.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        roundtrip_value = literal_fields.get("dynamic_assembly_roundtrip")
        if (
            isinstance(roundtrip_value, ast.Constant)
            and roundtrip_value.value is False
        ):
            attach_only_success_returns.append(literal_fields)

    assert len(attach_only_success_returns) == 1
    assembly_report_value = attach_only_success_returns[0].get(
        "dynamic_assembly_assembly_run_report"
    )
    assert isinstance(assembly_report_value, ast.Call)
    assert isinstance(assembly_report_value.func, ast.Attribute)
    assert isinstance(assembly_report_value.func.value, ast.Name)
    assert assembly_report_value.func.value.id == "assembly_report"
    assert assembly_report_value.func.attr == "to_dict"


def test_production_roundtrip_wires_bounded_separation_lifecycle() -> None:
    probe_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "p4_control_holon_spawn_probe.py"
    )
    syntax_tree = ast.parse(
        probe_path.read_text(encoding="utf-8"),
        filename=str(probe_path),
    )
    probe_function = next(
        node
        for node in syntax_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_dynamic_assembly_roundtrip_probe"
    )
    constructor_calls = [
        node
        for node in ast.walk(probe_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DynamicSeparationLifecycle"
    ]
    lifecycle_method_calls = {
        node.func.attr
        for node in ast.walk(probe_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "separation_lifecycle"
    }

    assert len(constructor_calls) == 1
    constructor_keywords = {
        keyword.arg: keyword.value
        for keyword in constructor_calls[0].keywords
        if keyword.arg is not None
    }
    assert {
        "nominal_separation_steps",
        "max_separation_steps",
        "minimum_gap_m",
        "minimum_clearance_m",
        "required_post_release_stable_steps",
        "max_post_release_steps",
    } == set(constructor_keywords)
    assert ast.unparse(constructor_keywords["nominal_separation_steps"]) == (
        "separation_steps"
    )
    assert ast.unparse(constructor_keywords["max_separation_steps"]) == (
        "2 * separation_steps"
    )
    assert ast.unparse(constructor_keywords["minimum_gap_m"]) == (
        "0.8 * config.separation_distance_m"
    )
    assert ast.unparse(constructor_keywords["minimum_clearance_m"]) == (
        "config.release_filter_clearance_m"
    )
    assert ast.unparse(
        constructor_keywords["required_post_release_stable_steps"]
    ) == "post_hold_steps"
    assert ast.unparse(constructor_keywords["max_post_release_steps"]) == (
        "2 * post_hold_steps"
    )
    assert {
        "observe_separation",
        "confirm_filter_removal",
        "observe_post_release",
    } <= lifecycle_method_calls


@pytest.mark.parametrize(
    "target_maximum_field",
    [
        "max_abs_joint_position_target_rad",
        "max_abs_joint_velocity_target_radps",
        "max_abs_joint_effort_bias_target_nm",
    ],
)
def test_axial_selected_joint_evidence_rejects_each_nonzero_target_maximum(
    target_maximum_field: str,
) -> None:
    graph = _graph()
    config = _physical_config()
    report = _attach_report(graph, config)
    leader_key = str(report["dynamic_assembly_leader_module_id"])
    evidence = report["dynamic_assembly_axial_selected_joint_evidence"]
    evidence["by_module"][leader_key][target_maximum_field] = 1.0e-3

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert "invalid:dynamic_assembly_axial_selected_joint_evidence" in failures


@pytest.mark.parametrize(
    "tamper",
    ["zero_sample_count", "missing_follower", "nonfinite_pose"],
)
def test_axial_selected_joint_evidence_rejects_incomplete_or_nonfinite_data(
    tamper: str,
) -> None:
    graph = _graph()
    config = _physical_config()
    report = _attach_report(graph, config)
    evidence = report["dynamic_assembly_axial_selected_joint_evidence"]
    leader_key = str(report["dynamic_assembly_leader_module_id"])
    follower_key = str(report["dynamic_assembly_follower_module_id"])
    if tamper == "zero_sample_count":
        evidence["sample_count"] = 0
    elif tamper == "missing_follower":
        evidence["by_module"].pop(follower_key)
    else:
        evidence["by_module"][leader_key]["first_root_pose_world"][0] = float(
            "nan"
        )

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert "invalid:dynamic_assembly_axial_selected_joint_evidence" in failures


def test_axial_selected_joint_evidence_allows_nonzero_measured_q_and_qdot() -> None:
    graph = _graph()
    config = _physical_config()
    report = _attach_report(graph, config)
    evidence = report["dynamic_assembly_axial_selected_joint_evidence"]
    for module_evidence in evidence["by_module"].values():
        module_evidence["max_abs_measured_joint_position_rad"] = 0.20
        module_evidence["max_abs_measured_joint_velocity_radps"] = 0.50

    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    ) == []


def test_roundtrip_gate_rejects_the_same_attach_only_report() -> None:
    graph = _graph()
    config = _physical_config(acceptance_gate=DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE)
    report = _attach_report(graph, config)

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert "mismatch:dynamic_assembly_detach_passed" in failures
    assert "mismatch:dynamic_assembly_roundtrip" in failures
    assert "missing:dynamic_assembly_constraint_removed" in failures
    assert "missing:dynamic_assembly_split_handover_completed" in failures
    assert (
        "missing:dynamic_assembly_follower_external_contact_free_during_unload"
        in failures
    )
    assert any(failure.startswith("invalid:dynamic_assembly_phase_order:") for failure in failures)


def test_config_rejects_unknown_acceptance_gate() -> None:
    with pytest.raises(SchemaValidationError, match="acceptance_gate"):
        DynamicAssemblyIsaacConfig(acceptance_gate="unknown").validate()


def test_solver_iteration_counts_are_positive_integers_and_round_trip() -> None:
    config = DynamicAssemblyIsaacConfig(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )

    config.validate()
    restored = DynamicAssemblyIsaacConfig.from_json(config.to_json())

    assert restored.solver_position_iteration_count == 16
    assert restored.solver_velocity_iteration_count == 4
    assert type(restored.solver_position_iteration_count) is int
    assert type(restored.solver_velocity_iteration_count) is int


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("solver_position_iteration_count", 0),
        ("solver_position_iteration_count", -1),
        ("solver_position_iteration_count", 1.5),
        ("solver_position_iteration_count", 2.0),
        ("solver_position_iteration_count", True),
        ("solver_velocity_iteration_count", 0),
        ("solver_velocity_iteration_count", -1),
        ("solver_velocity_iteration_count", 1.5),
        ("solver_velocity_iteration_count", 2.0),
        ("solver_velocity_iteration_count", False),
    ],
)
def test_solver_iteration_counts_reject_non_positive_or_non_integer_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(SchemaValidationError, match=field_name):
        replace(
            DynamicAssemblyIsaacConfig(),
            **{field_name: invalid_value},
        )


def test_config_requires_mode_specific_contact_semantics() -> None:
    default = DynamicAssemblyIsaacConfig()
    default.validate()
    assert default.mating_contact_mode == DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE
    assert default.control_bridge.require_selected_pair_contact is False

    with pytest.raises(SchemaValidationError, match="requires selected-pair contact"):
        replace(
            default,
            mating_contact_mode=DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
        ).validate()

    with pytest.raises(SchemaValidationError, match="requires.*false"):
        replace(
            default,
            control_bridge=replace(
                default.control_bridge,
                require_selected_pair_contact=True,
            ),
        ).validate()

    _fallback_config().validate()
    _physical_config().validate()


def test_filter_fallback_has_separate_contactless_acceptance_evidence() -> None:
    graph = _graph()
    config = _fallback_config()
    report = _fallback_attach_report(graph, config)

    assert dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    ) == []


def test_filter_fallback_rejects_physical_contact_claims_and_bad_filter_delta() -> None:
    graph = _graph()
    config = _fallback_config()
    report = _fallback_attach_report(graph, config)
    report["dynamic_assembly_selected_pair_contact_observed"] = True
    report["dynamic_assembly_first_selected_contact_evidence"] = (
        _raw_contact_evidence()
    )
    report["dynamic_assembly_mating_filter_evidence"][
        "added_leader_targets"
    ] = ["/World/Unexpected"]

    failures = dynamic_assembly_report_failures(
        report,
        morphology_graph=graph,
        config=config,
    )

    assert "mismatch:dynamic_assembly_selected_pair_contact_observed" in failures
    assert "invalid:dynamic_assembly_fallback_contact_evidence" in failures
    assert "invalid:dynamic_assembly_mating_filter_evidence" in failures


def test_guidance_envelope_must_match_coarse_prealign_envelope() -> None:
    with pytest.raises(SchemaValidationError, match="transverse envelope"):
        DynamicAssemblyIsaacConfig(
            guidance_contact_max_transverse_error_m=0.02
        ).validate()


def test_config_requires_filter_clearance_before_full_separation() -> None:
    with pytest.raises(SchemaValidationError, match="release filter clearance"):
        DynamicAssemblyIsaacConfig(
            release_filter_clearance_m=0.20,
            separation_distance_m=0.20,
        ).validate()


def test_config_requires_post_release_hold_to_preserve_clearance_margin() -> None:
    with pytest.raises(SchemaValidationError, match="post-release position tolerance"):
        DynamicAssemblyIsaacConfig(
            separation_distance_m=0.20,
            release_filter_clearance_m=0.03,
            post_release_position_tolerance_m=0.17,
        ).validate()
