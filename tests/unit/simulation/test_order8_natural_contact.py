from __future__ import annotations

from dataclasses import replace
import io
import json

import pytest

from amsrr.controllers.actuator_mapping import build_actuator_mapping
from amsrr.controllers.centroidal_admittance import (
    CentroidalAdmittanceConfig,
    CentroidalExternalWrenchEstimatorConfig,
)
from amsrr.controllers.natural_contact_joint_controller import (
    NaturalContactJointControllerConfig,
    position_drive_peak_effort_lead_rad,
)
from amsrr.controllers.qpid_controller import QPIDControllerConfig
from amsrr.policies.deterministic_natural_contact_planner import (
    NaturalContactPlannerConfig,
    ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT,
)

import amsrr.simulation.order8_natural_contact as order8_module

from amsrr.robot_model.gripper_surfaces import (
    select_opposing_gripper_surface_pair,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaValidationError, canonical_json
from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
    ORDER8_NATURAL_CONTACT_MODEL,
    ORDER8_NATURAL_CONTACT_RESULT_VERSION,
    Order8NaturalContactConfig,
    Order8NaturalContactPhase,
    Order8NaturalContactResult,
)
from amsrr.simulation.isaac_lab_backend import (
    IsaacLabAvailability,
    IsaacLabBackend,
    load_isaac_lab_backend_config,
)
from amsrr.simulation.order8_natural_contact import (
    ORDER8_NATURAL_CONTACT_REPORT_VERSION,
    ORDER8_NATURAL_CONTACT_REQUIRED_PHASES,
    ORDER8_NATURAL_CONTACT_SCOPE,
    Order8IsaacNaturalContactEnv,
    build_representative_order8_morphology,
    order8_natural_contact_report_failures,
    validate_representative_order8_morphology,
)
from amsrr.simulation.order8_isaac_runtime import (
    ORDER8_CONTACT_STALL_RATED_TORQUE_FRACTION,
    ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
    ORDER8_SELECTED_GRIPPER_MATERIAL_PATH,
)
from amsrr.utils.hashing import stable_hash


def _env(**kwargs):
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    backend = IsaacLabBackend(
        load_isaac_lab_backend_config("configs/env/isaac_lab.yaml")
    )
    return Order8IsaacNaturalContactEnv(
        config=Order8NaturalContactConfig(),
        backend=backend,
        physical_model=physical_model,
        **kwargs,
    )


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_order8_command_force_converts_and_binds_graph_config_and_gui() -> None:
    env = _env(
        viewer="kit",
        realtime_playback=True,
        keep_open_after_rollout_s=20.0,
    )
    graph = env.representative_morphology()

    command = env.build_probe_command(graph)

    assert "--force-convert" in command
    assert "--convert-if-missing" not in command
    assert "--order8-natural-contact" in command
    assert _argument(command, "--steps") == str(env.requested_steps)
    assert _argument(command, "--order8-morphology-graph-json") == canonical_json(graph)
    assert _argument(command, "--order8-config-json") == env.config.to_json()
    assert _argument(command, "--order8-seed") == "0"
    assert _argument(command, "--generated-usd-dir") == env.generated_usd_dir
    assert _argument(command, "--viz") == "kit"
    assert "--realtime-playback" in command
    assert _argument(command, "--keep-open-after-smoke-s") == "20.0"


def test_order9_teacher_command_is_explicit_and_can_reuse_generated_asset() -> None:
    env = _env(
        force_convert=False,
        generated_usd_path="artifacts/isaac/order8-prepared/holon.usda",
        order9_teacher_output="artifacts/order9/c0/episode-1",
        order9_teacher_episode_id="episode-1",
        order9_teacher_task_id="task-1",
        order9_teacher_split="validation",
    )

    command = env.build_probe_command(env.representative_morphology())

    assert "--convert-if-missing" in command
    assert "--force-convert" not in command
    assert _argument(command, "--generated-usd-path") == (
        "artifacts/isaac/order8-prepared/holon.usda"
    )
    assert _argument(command, "--order9-teacher-output") == (
        "artifacts/order9/c0/episode-1"
    )
    assert _argument(command, "--order9-teacher-episode-id") == "episode-1"
    assert _argument(command, "--order9-teacher-task-id") == "task-1"
    assert _argument(command, "--order9-teacher-split") == "validation"
    assert _argument(command, "--order9-teacher-high-level-stride") == "5"


def test_representative_graph_is_exact_symmetric_two_anchor_three_module_design() -> (
    None
):
    env = _env()
    graph = build_representative_order8_morphology(env.physical_model)

    validate_representative_order8_morphology(graph, physical_model=env.physical_model)
    assert len(graph.modules) == 3
    assert len(graph.robot_anchors) == 2
    assert {anchor.module_id for anchor in graph.robot_anchors} == {1, 2}

    changed = replace(graph, graph_id=graph.graph_id + ":tampered")
    with pytest.raises(SchemaValidationError, match="exactly match"):
        validate_representative_order8_morphology(
            changed, physical_model=env.physical_model
        )


def test_fake_real_report_passes_and_no_root_write_or_phase_tampering_can_pass() -> (
    None
):
    base = _env()
    graph = base.representative_morphology()
    report = _valid_report(base, graph)
    env = _env(command_executor=lambda command, timeout: report)

    result = env.run(graph, dry_run=False, check_availability=False)

    assert result.passed is True
    assert result.report_validation_failures == []

    report["order8_natural_contact_object_root_pose_write_count"] = 1
    report["order8_natural_contact_object_constraint_reference_count"] = 1
    report["order8_natural_contact_object_constraint_prim_paths"] = [
        "/World/Order8/ForbiddenObjectJoint"
    ]
    report["order8_natural_contact_ordered_phase_trace"] = [
        phase for phase in ORDER8_NATURAL_CONTACT_REQUIRED_PHASES if phase != "release"
    ]
    report["order8_natural_contact_seed_applied"]["torch"] = False
    report["order8_natural_contact_dock_joint_observed_ids"][-1] = "not-a-dock-joint"
    report["order8_natural_contact_contact_closure_raw_contact_input"] = True
    report["order8_natural_contact_contact_yield_raw_contact_input"] = True
    report[
        "order8_natural_contact_contact_yield_joint_drive_raw_contact_input"
    ] = True
    report[
        "order8_natural_contact_contact_yield_joint_drive_final_stiffness_nm_per_rad"
    ] = 50.0
    report[
        "order8_natural_contact_contact_yield_joint_drive_final_blend"
    ] = 0.5
    report["order8_natural_contact_contact_yield_grasp_pose_rebased"] = False
    report["order8_natural_contact_contact_yield_final_blend"] = 0.5
    report["order8_natural_contact_contact_yield_maximum_translation_offset_m"] = (
        2.0 * base.config.contact_admittance_max_translation_offset_m
    )
    report["order8_natural_contact_contact_wrench_application_mapping"] = "tampered"
    report["order8_natural_contact_contact_configuration_latched"] = False
    report["order8_natural_contact_selected_gripper_material_binding_audit_passed"] = (
        False
    )
    report["order8_natural_contact_selected_gripper_dynamic_friction"] = 0.6
    report["order8_natural_contact_payload_load_observer_raw_contact_input"] = True
    report[
        "order8_natural_contact_payload_feedforward_max_lead_over_observed_scale"
    ] = 1.1
    report[
        "order8_natural_contact_payload_commanded_lift_progress_peak_scale"
    ] = 0.5
    report[
        "order8_natural_contact_payload_feedforward_max_lag_behind_commanded_"
        "progress_scale"
    ] = 0.1
    report[
        "order8_natural_contact_lift_acceleration_bias_raw_contact_input"
    ] = True
    report[
        "order8_natural_contact_lift_acceleration_bias_non_lift_active_count"
    ] = 1
    report[
        "order8_natural_contact_lift_acceleration_bias_peak_residual_force_"
        "body_norm_n"
    ] = 0.5
    report["order8_natural_contact_last_lift_acceleration_bias_scale"] = 0.1
    report[
        "order8_natural_contact_lift_acceleration_bias_removal_complete_time_s"
    ] = 20.1
    report["order8_natural_contact_monitor_result"][
        "max_force_per_selected_contact_n"
    ] = 31.0
    failures = order8_natural_contact_report_failures(
        report,
        morphology_graph=graph,
        config=base.config,
        physical_model=base.physical_model,
        expected_backend_config_hash=base.backend.config.stable_hash(),
        expected_collision_geometry_hash=base.collision_geometry_hash,
        expected_source_urdf_hash=base.source_urdf_hash,
        requested_steps=base.requested_steps,
    )
    assert "mismatch:order8_natural_contact_object_root_pose_write_count" in failures
    assert (
        "mismatch:order8_natural_contact_object_constraint_reference_count" in failures
    )
    assert "mismatch:order8_natural_contact_object_constraint_prim_paths" in failures
    assert "mismatch:order8_natural_contact_ordered_phase_trace" in failures
    assert "invalid:order8_natural_contact_seed_applied" in failures
    assert "mismatch:order8_natural_contact_dock_joint_observed_ids" in failures
    assert (
        "mismatch:order8_natural_contact_contact_closure_raw_contact_input" in failures
    )
    assert (
        "mismatch:order8_natural_contact_contact_yield_raw_contact_input" in failures
    )
    assert (
        "mismatch:order8_natural_contact_contact_yield_joint_drive_raw_contact_input"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_contact_yield_joint_drive_final_stiffness_nm_per_rad"
        in failures
    )
    assert (
        "invalid:order8_natural_contact_contact_yield_joint_drive_final_blend"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_contact_yield_grasp_pose_rebased" in failures
    )
    assert (
        "invalid:order8_natural_contact_contact_yield_final_blend" in failures
    )
    assert (
        "invalid:order8_natural_contact_contact_yield_maximum_translation_offset_m"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_contact_wrench_application_mapping" in failures
    )
    assert "mismatch:order8_natural_contact_contact_configuration_latched" in failures
    assert (
        "mismatch:order8_natural_contact_selected_gripper_material_binding_audit_passed"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_selected_gripper_dynamic_friction" in failures
    )
    assert (
        "mismatch:order8_natural_contact_payload_load_observer_raw_contact_input"
        in failures
    )
    assert (
        "invalid:order8_natural_contact_payload_feedforward_max_lead_over_observed_scale"
        in failures
    )
    assert (
        "invalid:order8_natural_contact_payload_commanded_lift_progress_peak_scale"
        in failures
    )
    assert (
        "invalid:order8_natural_contact_payload_feedforward_max_lag_behind_"
        "commanded_progress_scale"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_lift_acceleration_bias_raw_contact_input"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_lift_acceleration_bias_non_lift_active_count"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_lift_acceleration_bias_peak_residual_"
        "force_body_norm_n"
        in failures
    )
    assert (
        "mismatch:order8_natural_contact_last_lift_acceleration_bias_scale"
        in failures
    )
    assert (
        "invalid:order8_natural_contact_lift_acceleration_bias_removal_complete_"
        "time_s"
        in failures
    )
    assert (
        "above:order8_natural_contact_monitor_result.max_force_per_selected_contact_n"
        in failures
    )


def test_optional_all_arrest_shape_hold_may_remain_inactive() -> None:
    env = _env()
    graph = env.representative_morphology()
    report = _valid_report(env, graph)
    report["order8_natural_contact_contact_centering_active_step_count"] = 0
    report["order8_natural_contact_contact_centering_cycle_count"] = 0

    failures = order8_natural_contact_report_failures(
        report,
        morphology_graph=graph,
        config=env.config,
        physical_model=env.physical_model,
        expected_backend_config_hash=env.backend.config.stable_hash(),
        expected_collision_geometry_hash=env.collision_geometry_hash,
        expected_source_urdf_hash=env.source_urdf_hash,
        requested_steps=env.requested_steps,
    )

    assert failures == []


def test_teacher_collection_may_not_need_clearance_sync_activation() -> None:
    env = _env()
    graph = env.representative_morphology()
    report = _valid_report(env, graph)
    report["order9_teacher_collection_enabled"] = True
    report["order8_natural_contact_contact_clearance_sync_active_step_count"] = 0

    failures = order8_natural_contact_report_failures(
        report,
        morphology_graph=graph,
        config=env.config,
        physical_model=env.physical_model,
        expected_backend_config_hash=env.backend.config.stable_hash(),
        expected_collision_geometry_hash=env.collision_geometry_hash,
        expected_source_urdf_hash=env.source_urdf_hash,
        requested_steps=env.requested_steps,
        expected_order9_teacher_collection_enabled=True,
    )

    assert failures == []

    failures = order8_natural_contact_report_failures(
        report,
        morphology_graph=graph,
        config=env.config,
        physical_model=env.physical_model,
        expected_backend_config_hash=env.backend.config.stable_hash(),
        expected_collision_geometry_hash=env.collision_geometry_hash,
        expected_source_urdf_hash=env.source_urdf_hash,
        requested_steps=env.requested_steps,
    )
    assert "mismatch:order9_teacher_collection_enabled" in failures
    assert (
        "invalid:order8_natural_contact_contact_clearance_sync_active_step_count"
        in failures
    )


def test_external_isaac_launcher_does_not_require_host_python_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _env()
    graph = base.representative_morphology()
    report = _valid_report(base, graph)
    env = _env(command_executor=lambda command, timeout: report)
    monkeypatch.setattr(
        env.backend,
        "availability",
        lambda: IsaacLabAvailability(
            available=False,
            isaaclab_path_exists=True,
            launch_script_exists=True,
            urdf_exists=True,
            generated_usd_exists=False,
            python_modules_available=False,
            missing_reasons=["isaac_python_modules_unavailable_in_current_interpreter"],
        ),
    )

    result = env.run(graph, dry_run=False)

    assert result.attempted is True
    assert result.passed is True

    monkeypatch.setattr(
        env.backend,
        "availability",
        lambda: IsaacLabAvailability(
            available=False,
            isaaclab_path_exists=True,
            launch_script_exists=False,
            urdf_exists=True,
            generated_usd_exists=False,
            python_modules_available=False,
            missing_reasons=[
                "launch_script_missing",
                "isaac_python_modules_unavailable_in_current_interpreter",
            ],
        ),
    )
    blocked = env.run(graph, dry_run=False)

    assert blocked.attempted is False
    assert blocked.report_validation_failures == ["launch_script_missing"]


def test_dry_run_is_never_real_acceptance_and_viewing_contract_fails_closed() -> None:
    env = _env()
    result = env.run(dry_run=True)

    assert result.dry_run is True
    assert result.attempted is False
    assert result.isaac_backed is False
    assert result.passed is False
    assert "--order8-natural-contact" in result.report["probe_command"]

    with pytest.raises(SchemaValidationError, match="requires viewer"):
        _env(realtime_playback=True)


def test_subprocess_wrapper_preserves_nonzero_probe_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO(json.dumps({"spawn_passed": False}) + "\n")
            self.stderr = io.StringIO(
                "[order8-natural-contact] simulation_time=1.000s phase=approach\n"
            )

        def wait(self, timeout=None):
            return 7

        def kill(self) -> None:
            pass

    monkeypatch.setattr(
        order8_module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    report = order8_module._run_json_command(["fake-order8-probe"], 1.0)

    assert report["spawn_passed"] is False
    assert report["command_returncode"] == 7
    assert "phase=approach" in report["command_stderr_tail"]


def test_subprocess_wrapper_timeout_preserves_progress_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(
                "[order8-natural-contact] simulation_time=63.100s "
                "phase=contact_acquisition\n"
            )
            self.killed = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise order8_module.subprocess.TimeoutExpired(
                    cmd="fake-order8-probe", timeout=timeout
                )
            return -9

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(
        order8_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(RuntimeError, match="simulation_time=63.100s"):
        order8_module._run_json_command(["fake-order8-probe"], 1.0)
    assert process.killed is True


def _valid_report(env: Order8IsaacNaturalContactEnv, graph) -> dict:
    surface_pair = select_opposing_gripper_surface_pair(graph, env.physical_model)
    selected_link_ids = sorted(
        f"module_{surface.module_id}:{surface.mechanism_link_id}"
        for surface in (surface_pair.first, surface_pair.second)
    )
    monitor_result = Order8NaturalContactResult(
        result_version=ORDER8_NATURAL_CONTACT_RESULT_VERSION,
        config_version=ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
        config_hash=env.config.stable_hash(),
        contact_model=ORDER8_NATURAL_CONTACT_MODEL,
        final_phase=Order8NaturalContactPhase.COMPLETE,
        attempted=True,
        passed=True,
        step_count=4000,
        duration_s=20.0,
        selected_dock_link_ids=selected_link_ids,
        grasp_acquired=True,
        lift_acquired=True,
        transport_acquired=True,
        release_contact_free_acquired=True,
        retreat_clearance_acquired=True,
        settle_acquired=True,
        object_dropped=False,
        unintended_contact_count=0,
        max_force_per_selected_contact_n=11.5,
        max_torque_per_selected_contact_nm=0.5,
        max_penetration_m=0.001,
        max_tangential_slip_speed_mps=0.01,
        max_provisional_acquisition_slip_speed_mps=0.03,
        max_contact_point_slip_displacement_m_by_link={
            link_id: 0.005 for link_id in selected_link_ids
        },
        failure_reasons=[],
    )
    geometry_refs = sorted(
        primitive.geometry_ref
        for surface in (surface_pair.first, surface_pair.second)
        for primitive in surface.collision_primitives
        if primitive.geometry_ref is not None
    )
    selected_surfaces = (surface_pair.first, surface_pair.second)
    material_body_paths = sorted(
        f"/World/Order8/Module_{surface.module_id}/{surface.mechanism_link_id}"
        for surface in selected_surfaces
    )
    material_collision_paths = [
        f"{body_path}/collisions/mesh" for body_path in material_body_paths
    ]
    dock_joint_ids = list(ordered_global_dock_joint_ids(graph, env.physical_model))
    qpid_config = QPIDControllerConfig(
        allocation_mode="rigid_body_qp",
        control_dt_s=env.simulation_dt_s,
    )
    contact_centering_qpid_config = replace(
        qpid_config,
        xy_p_gain=env.config.contact_centering_xy_p_gain,
        xy_d_gain=env.config.contact_centering_xy_d_gain,
        roll_pitch_p_gain=(env.config.contact_centering_roll_pitch_p_gain),
        roll_pitch_d_gain=(env.config.contact_centering_roll_pitch_d_gain),
    )
    external_wrench_estimator_config = CentroidalExternalWrenchEstimatorConfig(
        gravity_mps2=qpid_config.gravity_mps2,
        wrench_filter_time_constant_s=(
            env.config.contact_external_wrench_filter_time_constant_s
        ),
        bias_filter_time_constant_s=(
            env.config.contact_external_wrench_bias_time_constant_s
        ),
    )
    contact_admittance_config = CentroidalAdmittanceConfig(
        force_deadband_n=env.config.contact_admittance_force_deadband_n,
        torque_deadband_nm=env.config.contact_admittance_torque_deadband_nm,
        linear_admittance_mps_per_n=(
            env.config.contact_admittance_linear_gain_mps_per_n
        ),
        angular_admittance_radps_per_nm=(
            env.config.contact_admittance_angular_gain_radps_per_nm
        ),
        maximum_linear_speed_mps=(
            env.config.contact_admittance_max_linear_speed_mps
        ),
        maximum_angular_speed_radps=(
            env.config.contact_admittance_max_angular_speed_radps
        ),
        maximum_translation_offset_m=(
            env.config.contact_admittance_max_translation_offset_m
        ),
    )
    joint_controller_config = NaturalContactJointControllerConfig(
        control_dt_s=env.simulation_dt_s,
        max_position_command_lead_rad=position_drive_peak_effort_lead_rad(
            stiffness_nm_per_rad=200.0,
            peak_effort_nm=4.1,
        ),
        reachability_absolute_tolerance=(
            env.config.simultaneous_reachability_absolute_tolerance
        ),
    )
    planner_config = NaturalContactPlannerConfig(
        contact_acquisition_timeout_s=(env.config.contact_acquisition_timeout_s),
        normal_force_target_per_contact_n=(
            env.config.normal_force_target_per_contact_n
        ),
    )
    return {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": True,
        "command_probe_passed": True,
        "command_returncode": 0,
        "order8_natural_contact_enabled": True,
        "order8_natural_contact_diagnostic_only": False,
        "order8_natural_contact_diagnostic_force_fixture": False,
        "order8_natural_contact_diagnostic_precontact_fixture": False,
        "order8_natural_contact_diagnostic_precontact_base_pose": None,
        "order8_natural_contact_diagnostic_world_fixed_base": False,
        "order8_natural_contact_diagnostic_world_fixed_body_path": None,
        "order8_natural_contact_diagnostic_world_fixed_pose": None,
        "order8_natural_contact_diagnostic_world_fixed_object": False,
        "order8_natural_contact_diagnostic_world_fixed_object_pose": None,
        "order8_natural_contact_diagnostic_object_width_padding_m": 0.0,
        "order8_natural_contact_runtime_object_size_m": list(env.config.object_size_m),
        "order8_natural_contact_object_support_method": (
            "free_object_on_fixed_raised_platform_without_pose_constraint_v1"
        ),
        "order8_natural_contact_object_support_path": (
            "/World/Order8/ObjectSupport"
        ),
        "order8_natural_contact_object_support_height_m": (
            env.config.object_support_height_m
        ),
        "order8_natural_contact_object_support_size_m": [
            env.config.object_size_m[0]
            + env.config.required_transport_distance_m
            + 0.05,
            env.config.object_size_m[1] - 0.04,
            env.config.object_support_height_m,
        ],
        "order8_natural_contact_object_support_pose_world": [
            0.0,
            0.0,
            0.5 * env.config.object_support_height_m,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_object_support_covers_planned_place_pose": True,
        "order8_natural_contact_robot_environment_contact_method": (
            "all_robot_rigid_bodies_against_floor_and_object_support_v1"
        ),
        "order8_natural_contact_robot_environment_contact_step_count": 0,
        "order8_natural_contact_robot_environment_unsafe_contact_step_count": 0,
        "order8_natural_contact_robot_environment_first_unsafe_contact_time_s": None,
        "order8_natural_contact_acceptance_eligible": True,
        "order8_natural_contact_diagnostic_mode": "disabled",
        "order8_natural_contact_diagnostic_stop_force_scale": None,
        "order8_natural_contact_diagnostic_stop_reached": False,
        "order8_natural_contact_diagnostic_observed_monitor_passed": True,
        "order8_natural_contact_solver_position_iteration_count": 8,
        "order8_natural_contact_solver_velocity_iteration_count": 8,
        "order8_natural_contact_passed": True,
        "order8_natural_contact_report_version": ORDER8_NATURAL_CONTACT_REPORT_VERSION,
        "order8_natural_contact_contact_model": ORDER8_NATURAL_CONTACT_MODEL,
        "order8_natural_contact_scope": ORDER8_NATURAL_CONTACT_SCOPE,
        "order8_natural_contact_p4_full_completion_claim": False,
        "order8_natural_contact_order9_full_taskspec_claim": False,
        "order8_natural_contact_learned_policy_success_claim": False,
        "order8_natural_contact_config": env.config.to_dict(),
        "order8_natural_contact_config_hash": env.config.stable_hash(),
        "order8_natural_contact_graph_id": graph.graph_id,
        "order8_natural_contact_graph_hash": graph.stable_hash(),
        "order8_natural_contact_module_count": 3,
        "order8_natural_contact_robot_anchor_count": 2,
        "order8_natural_contact_seed": env.seed,
        "order8_natural_contact_seed_applied": {
            "seed": env.seed,
            "python_random": True,
            "torch": True,
            "torch_cuda": False,
            "numpy": True,
        },
        "order8_natural_contact_backend_config_hash": env.backend.config.stable_hash(),
        "order8_natural_contact_physical_model_hash": env.physical_model.stable_hash(),
        "order8_natural_contact_collision_geometry_content_hash": env.collision_geometry_hash,
        "order8_natural_contact_source_urdf_sha256": env.source_urdf_hash,
        "order8_natural_contact_generated_usd_sha256": "a" * 64,
        "order8_natural_contact_generated_usd_bundle_hash": "b" * 64,
        "order8_natural_contact_force_usd_conversion": True,
        "order8_natural_contact_dock_collision_type": "Convex Decomposition",
        "order8_natural_contact_dock_collision_approximation_token": "convexDecomposition",
        "order8_natural_contact_dock_collision_approximation_verified": True,
        "order8_natural_contact_dock_collision_composed_prim_count": 6,
        "order8_natural_contact_requested_steps": env.requested_steps,
        "order8_natural_contact_simulation_dt_s": env.simulation_dt_s,
        "order8_natural_contact_qpid_config": {
            field: getattr(qpid_config, field)
            for field in qpid_config.__dataclass_fields__
        },
        "order8_natural_contact_qpid_config_hash": stable_hash(qpid_config),
        "order8_natural_contact_contact_centering_qpid_config": {
            field: getattr(contact_centering_qpid_config, field)
            for field in contact_centering_qpid_config.__dataclass_fields__
        },
        "order8_natural_contact_contact_centering_qpid_config_hash": (
            stable_hash(contact_centering_qpid_config)
        ),
        "order8_natural_contact_external_wrench_estimator_config": {
            field: getattr(external_wrench_estimator_config, field)
            for field in external_wrench_estimator_config.__dataclass_fields__
        },
        "order8_natural_contact_external_wrench_estimator_config_hash": (
            stable_hash(external_wrench_estimator_config)
        ),
        "order8_natural_contact_contact_admittance_config": {
            field: getattr(contact_admittance_config, field)
            for field in contact_admittance_config.__dataclass_fields__
        },
        "order8_natural_contact_contact_admittance_config_hash": stable_hash(
            contact_admittance_config
        ),
        "order8_natural_contact_contact_yield_method": (
            "first_damping_compensated_terminal_joint_surface_load_enables_"
            "contact_axis_centroidal_admittance_with_full_height_attitude_"
            "pose_tracking_v9"
        ),
        "order8_natural_contact_contact_yield_trigger_method": (
            "any_selected_terminal_joint_damping_compensated_load_plus_mesh_"
            "proximity_after_closure_armed_latched_until_verified_grasp_v7"
        ),
        "order8_natural_contact_contact_yield_raw_contact_input": False,
        "order8_natural_contact_contact_yield_per_contact_wrench_input": False,
        "order8_natural_contact_contact_yield_external_wrench_scope": (
            "aggregate_centroidal_only_v1"
        ),
        "order8_natural_contact_contact_yield_triggered_time_s": 10.0,
        "order8_natural_contact_contact_yield_trigger_anchor_ids": [
            graph.robot_anchors[0].anchor_id
        ],
        "order8_natural_contact_contact_yield_active_step_count": 100,
        "order8_natural_contact_contact_yield_full_step_count": 50,
        "order8_natural_contact_contact_yield_restore_step_count": 25,
        "order8_natural_contact_contact_yield_final_blend": 0.0,
        "order8_natural_contact_contact_yield_minimum_pi_scale": 1.0,
        "order8_natural_contact_contact_yield_estimator_valid_step_count": 500,
        "order8_natural_contact_contact_yield_estimator_invalid_step_count": 0,
        "order8_natural_contact_contact_yield_maximum_external_force_n": 12.0,
        "order8_natural_contact_contact_yield_maximum_external_torque_nm": 0.5,
        "order8_natural_contact_contact_yield_maximum_translation_offset_m": 0.01,
        "order8_natural_contact_contact_yield_last_admittance_twist": [0.0] * 6,
        "order8_natural_contact_contact_yield_last_translation_offset_world": [
            0.0,
            0.0,
            0.0,
        ],
        "order8_natural_contact_contact_yield_grasp_pose_rebased": True,
        "order8_natural_contact_contact_yield_grasp_pose_rebase_time_s": 20.0,
        "order8_natural_contact_contact_yield_grasp_centroidal_pose": [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_contact_yield_grasp_rebase_method": (
            "nonprivileged_load_limited_position_preload_measured_full_6d_"
            "centroidal_with_zero_offset_torque_then_pi_restore_v6"
        ),
        "order8_natural_contact_contact_yield_joint_drive_method": (
            "disabled_nominal_dock_implicit_impedance_preserved_v7"
        ),
        "order8_natural_contact_contact_yield_joint_drive_trigger_method": (
            "disabled_v7"
        ),
        "order8_natural_contact_contact_yield_joint_drive_raw_contact_input": False,
        "order8_natural_contact_contact_yield_joint_drive_scope": (
            "none_v2"
        ),
        "order8_natural_contact_post_qclose_geometric_preload_complete": True,
        "order8_natural_contact_post_qclose_geometric_preload_method": (
            "not_applicable_replaced_by_joint_space_load_limited_preload_v5"
        ),
        "order8_natural_contact_post_qclose_geometric_preload_distance_m": (
            env.config.contact_closure_inward_overtravel_m
        ),
        "order8_natural_contact_post_qclose_geometric_preload_terminal_error_m": 0.0,
        "order8_natural_contact_post_qclose_geometric_preload_settle_dwell_s": (
            env.config.contact_stall_dwell_s
        ),
        "order8_natural_contact_post_qclose_geometric_preload_active_step_count": 20,
        "order8_natural_contact_contact_force_position_preload_method": (
            "fixed_closure_ratio_previous_target_integration_per_anchor_"
            "damping_compensated_load_dwell_and_freeze_v3"
        ),
        "order8_natural_contact_contact_force_position_preload_speed_limit_mps": (
            0.0
        ),
        "order8_natural_contact_contact_position_preload_joint_speed_radps": (
            env.config.contact_position_preload_joint_speed_radps
        ),
        "order8_natural_contact_contact_position_preload_load_threshold_nm": (
            env.config.contact_position_preload_load_threshold_nm
        ),
        "order8_natural_contact_contact_position_preload_complete": True,
        "order8_natural_contact_contact_position_preload_completion_source": (
            "per_anchor_damping_compensated_moving_chain_load_dwell"
        ),
        "order8_natural_contact_contact_position_preload_joint_ids_by_anchor": {
            str(anchor.anchor_id): [
                dock_joint_ids[index % len(dock_joint_ids)]
            ]
            for index, anchor in enumerate(graph.robot_anchors)
        },
        "order8_natural_contact_contact_position_preload_velocity_targets_radps": {
            joint_id: 0.0 for joint_id in dock_joint_ids
        },
        "order8_natural_contact_contact_position_preload_position_targets_rad": {
            joint_id: 0.0 for joint_id in dock_joint_ids
        },
        "order8_natural_contact_contact_position_preload_load_nm_by_anchor": {
            str(anchor.anchor_id): env.config.contact_position_preload_load_threshold_nm
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_position_preload_max_load_nm_by_anchor": {
            str(anchor.anchor_id): env.config.contact_position_preload_load_threshold_nm
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_position_preload_load_dwell_s_by_anchor": {
            str(anchor.anchor_id): env.config.contact_stall_dwell_s
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_position_preload_frozen_anchor_ids": sorted(
            anchor.anchor_id for anchor in graph.robot_anchors
        ),
        "order8_natural_contact_contact_position_preload_frozen_time_s_by_anchor": {
            str(anchor.anchor_id): 20.0 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_position_preload_active_step_count": 20,
        "order8_natural_contact_contact_yield_joint_drive_triggered_time_s": None,
        "order8_natural_contact_contact_yield_joint_drive_final_blend": 0.0,
        "order8_natural_contact_contact_yield_joint_drive_nominal_stiffness_nm_per_rad": (
            200.0
        ),
        "order8_natural_contact_contact_yield_joint_drive_nominal_damping_nms_per_rad": (
            5.0
        ),
        "order8_natural_contact_contact_yield_joint_drive_stiffness_scale": (
            env.config.contact_yield_joint_drive_stiffness_scale
        ),
        "order8_natural_contact_contact_yield_joint_drive_target_damping_nms_per_rad": (
            env.config.contact_yield_joint_drive_damping_nms_per_rad
        ),
        "order8_natural_contact_contact_yield_joint_drive_active_step_count": 0,
        "order8_natural_contact_contact_yield_joint_drive_write_count": 0,
        "order8_natural_contact_contact_yield_joint_drive_restore_write_count": 0,
        "order8_natural_contact_contact_yield_joint_drive_minimum_stiffness_nm_per_rad": (
            200.0
        ),
        "order8_natural_contact_contact_yield_joint_drive_maximum_damping_nms_per_rad": (
            5.0
        ),
        "order8_natural_contact_contact_yield_joint_drive_final_stiffness_nm_per_rad": (
            200.0
        ),
        "order8_natural_contact_contact_yield_joint_drive_final_damping_nms_per_rad": (
            5.0
        ),
        "order8_natural_contact_contact_yield_joint_drive_stiffness_targets_nm_per_rad": {
            joint_id: 200.0 for joint_id in dock_joint_ids
        },
        "order8_natural_contact_contact_yield_joint_drive_damping_targets_nms_per_rad": {
            joint_id: 5.0 for joint_id in dock_joint_ids
        },
        "order8_natural_contact_contact_axial_qpid_gain_schedule": (
            "mesh_open_axial_insert_uses_centering_horizontal_gain_bank_v1"
        ),
        "order8_natural_contact_contact_axial_gain_scheduled_step_count": 100,
        "order8_natural_contact_joint_controller_config": {
            field: getattr(joint_controller_config, field)
            for field in joint_controller_config.__dataclass_fields__
        },
        "order8_natural_contact_joint_controller_config_hash": stable_hash(
            joint_controller_config
        ),
        "order8_natural_contact_contact_joint_velocity_limit_command_rad_s": (
            env.config.contact_joint_velocity_limit_radps
        ),
        "order8_natural_contact_contact_joint_velocity_limit_basis": (
            "fixed_whole_structure_previous_target_integrated_velocity_and_"
            "simulator_consistent_below_ak40_10_configured_speed_limit_v2"
        ),
        "order8_natural_contact_joint_position_reference_mode": (
            "one_shot_whole_structure_ik_direction_previous_target_integrated_"
            "fixed_velocity_ratio_with_diagnostic_absolute_pitch_hold_until_"
            "load_qclose_then_slow_load_limited_previous_target_preload_and_"
            "measured_qopen_direct_return_v12"
        ),
        "order8_natural_contact_max_joint_position_command_lead_rad": 0.0205,
        "order8_natural_contact_max_joint_velocity_command_radps": (
            env.config.contact_joint_velocity_limit_radps
        ),
        "order8_natural_contact_planner_config": {
            field: getattr(planner_config, field)
            for field in planner_config.__dataclass_fields__
        },
        "order8_natural_contact_planner_config_hash": stable_hash(planner_config),
        "order8_natural_contact_base_target_speed_limit_mps": (
            env.config.base_translation_speed_limit_mps
        ),
        "order8_natural_contact_contact_base_target_speed_limit_mps": (
            env.config.contact_base_translation_speed_limit_mps
        ),
        "order8_natural_contact_contact_axial_min_mesh_overlap_m": (
            env.config.contact_axial_min_mesh_overlap_m
        ),
        "order8_natural_contact_contact_axial_overlap_method": (
            "selected_urdf_mesh_world_aabb_approach_axis_projection_v1"
        ),
        "order8_natural_contact_contact_axial_overlap_at_latch_m": (
            env.config.contact_axial_min_mesh_overlap_m
        ),
        "order8_natural_contact_grasp_base_pose_method": (
            "normal_aligned_floor_clear_tangential_contact_region_v1"
        ),
        "order8_natural_contact_floor_base_pose": [
            0.0,
            0.0,
            0.143,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_unconstrained_grasp_base_pose": [
            0.5,
            0.0,
            0.122,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_grasp_base_pose": [
            0.5,
            0.0,
            0.153,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_grasp_base_vertical_correction_m": 0.031,
        "order8_natural_contact_grasp_additional_floor_clearance_m": 0.01,
        "order8_natural_contact_grasp_base_normal_correction_m_by_anchor": {
            "0": 0.0,
            "1": 0.0,
        },
        "order8_natural_contact_grasp_base_tangential_correction_m_by_anchor": {
            "0": [0.0, 0.031],
            "1": [0.0, 0.031],
        },
        "order8_natural_contact_contact_axial_hold_method": (
            "measured_free_object_relative_floor_clear_contact_region_base_pose_"
            "with_rate_limited_retarget_v4"
        ),
        "order8_natural_contact_contact_axial_hold_base_pose": [
            0.25,
            0.0,
            0.50,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "order8_natural_contact_contact_axial_settle_dwell_s": (
            env.config.contact_stall_dwell_s
        ),
        "order8_natural_contact_contact_axial_settle_position_tolerance_m": (
            env.config.contact_tangential_tolerance_m
        ),
        "order8_natural_contact_contact_axial_settle_base_speed_tolerance_mps": (
            env.config.pregrasp_linear_speed_tolerance_mps
        ),
        "order8_natural_contact_contact_side_closure_enabled": True,
        "order8_natural_contact_contact_anchor_target_speed_limit_mps": (
            env.config.anchor_translation_speed_limit_mps
        ),
        "order8_natural_contact_contact_near_anchor_target_speed_limit_mps": (
            0.2 * env.config.anchor_translation_speed_limit_mps
        ),
        "order8_natural_contact_contact_near_anchor_slowdown_error_m": (
            env.config.contact_near_surface_slowdown_m
        ),
        "order8_natural_contact_contact_surface_anchor_target_speed_limit_mps": (
            env.config.contact_surface_creep_speed_limit_mps
        ),
        "order8_natural_contact_contact_surface_anchor_speed_boundary_m": (
            env.config.contact_surface_arm_clearance_m
        ),
        "order8_natural_contact_contact_anchor_target_speed_schedule": (
            "nonprivileged_three_tier_precenter_then_symmetric_creep_close_"
            "with_opposing_clearance_synchronization_v8"
        ),
        "order8_natural_contact_contact_clearance_sync_method": (
            "closer_surface_linear_slowdown_farther_surface_full_tier_speed_v1"
        ),
        "order8_natural_contact_contact_clearance_sync_deadband_m": (
            env.config.contact_clearance_sync_deadband_m
        ),
        "order8_natural_contact_contact_clearance_sync_full_slowdown_m": (
            env.config.contact_clearance_sync_full_slowdown_m
        ),
        "order8_natural_contact_contact_clearance_sync_minimum_speed_scale": (
            env.config.contact_clearance_sync_minimum_speed_scale
        ),
        "order8_natural_contact_contact_clearance_sync_active_step_count": 100,
        "order9_teacher_collection_enabled": False,
        "order8_natural_contact_post_first_arrest_creep_active_step_count": 0,
        "order8_natural_contact_post_first_arrest_centroidal_transfer_active_step_count": 0,
        "order8_natural_contact_max_contact_clearance_imbalance_m": 0.020,
        "order8_natural_contact_pregrasp_staging_method": (
            "selected_urdf_mesh_aabb_axial_retreat_bisection_v1"
        ),
        "order8_natural_contact_pregrasp_mesh_clearance_target_m": (
            env.config.pregrasp_mesh_clearance_m
        ),
        "order8_natural_contact_pregrasp_mesh_clearance_predicted_m": (
            env.config.pregrasp_mesh_clearance_m
        ),
        "order8_natural_contact_pregrasp_staging_retreat_distance_m": 0.26,
        "order8_natural_contact_pregrasp_approach_axis_world": [1.0, 0.0, 0.0],
        "order8_natural_contact_pregrasp_anchor_target_source": (
            "selected_urdf_mesh_aabb_outward_opening_in_base_frame_v1"
        ),
        "order8_natural_contact_pregrasp_opening_distance_m_by_anchor": {
            str(anchor.anchor_id): 0.05 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_pregrasp_opening_clearance_m_by_anchor": {
            str(anchor.anchor_id): env.config.pregrasp_mesh_clearance_m
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_pregrasp_minimum_achieved_mesh_clearance_m": (
            env.config.pregrasp_mesh_clearance_m
            - env.config.anchor_command_tracking_tolerance_m
        ),
        "order8_natural_contact_pregrasp_achieved_mesh_clearance_m": (
            env.config.pregrasp_mesh_clearance_m
        ),
        "order8_natural_contact_pregrasp_reachability_gate_passed": True,
        "order8_natural_contact_pregrasp_reachability_gate_source": (
            "achieved_mesh_clear_endpoint"
        ),
        "order8_natural_contact_pregrasp_open_configuration_latched": True,
        "order8_natural_contact_contact_axial_alignment_latched": True,
        "order8_natural_contact_contact_motion_sequence": (
            "mesh_open_then_floor_clear_object_relative_base_settle_then_"
            "known_grasp_ready_pose_then_one_shot_whole_structure_direction_"
            "fixed_velocity_close_until_simultaneous_load_qclose_then_slow_"
            "per_side_load_limited_position_preload_v25"
        ),
        "order8_natural_contact_contact_mesh_precenter_method": (
            "one_shot_direction_seed_only_not_completion_gate_v3"
        ),
        "order8_natural_contact_contact_mesh_precenter_clearance_m": (
            env.config.contact_near_surface_slowdown_m
        ),
        "order8_natural_contact_contact_mesh_precenter_tangential_tolerance_m": (
            env.config.contact_tangential_tolerance_m
        ),
        "order8_natural_contact_mesh_pair_base_centering_method": (
            "horizontal_approach_axis_mean_authored_mesh_patch_centering_v1"
        ),
        "order8_natural_contact_mesh_pair_base_centering_correction_world": [
            0.01,
            0.0,
            0.0,
        ],
        "order8_natural_contact_contact_mesh_precenter_complete": False,
        "order8_natural_contact_contact_mesh_precenter_dwell_s": 0.0,
        "order8_natural_contact_contact_mesh_precenter_completed_time_s": None,
        "order8_natural_contact_contact_centering_method": (
            "known_object_relative_centroidal_pose_hold_without_closure_mesh_"
            "feedback_v3"
        ),
        "order8_natural_contact_contact_centering_joint_motion_mode": (
            "all_docks_fixed_one_shot_velocity_ratio_without_receding_"
            "geometry_feedback_v4"
        ),
        "order8_natural_contact_contact_individual_arrest_centroidal_hold": (
            "disabled_provisional_contact_may_separate_until_simultaneous_qclose_v1"
        ),
        "order8_natural_contact_contact_post_arrest_shape_hold_activation": (
            "simultaneous_qclose_only_v1"
        ),
        "order8_natural_contact_contact_centering_settle_gate": (
            "object_relative_final_base_pose_and_speed_dwell_before_joint_close_v1"
        ),
        "order8_natural_contact_contact_centering_raw_contact_input": False,
        "order8_natural_contact_contact_centering_max_offset_limit_m": (
            env.config.contact_centering_max_offset_m
        ),
        "order8_natural_contact_contact_centering_max_tilt_limit_rad": (
            env.config.contact_centering_max_tilt_rad
        ),
        "order8_natural_contact_contact_centering_tilt_source": (
            "not_used_in_surface_region_joint_only_close_v1"
        ),
        "order8_natural_contact_contact_centering_active_step_count": 0,
        "order8_natural_contact_contact_continuous_balance_active_step_count": 0,
        "order8_natural_contact_contact_sequential_reacquire_active_step_count": 0,
        "order8_natural_contact_contact_sequential_centroidal_nudge_active_step_count": 0,
        "order8_natural_contact_contact_sequential_latched_transfer_active_step_count": 0,
        "order8_natural_contact_contact_sequential_joint_position_hold_step_count": 0,
        "order8_natural_contact_contact_centering_cycle_count": 0,
        "order8_natural_contact_contact_centering_max_observed_offset_m": 0.0,
        "order8_natural_contact_contact_centering_max_observed_tilt_rad": 0.0,
        "order8_natural_contact_contact_centering_max_measured_tilt_rad": 0.0,
        "order8_natural_contact_contact_centering_latched_offset_world": [
            0.0,
            0.020,
            0.0,
        ],
        "order8_natural_contact_anchor_reference_frame": (
            "measured_free_object_relative_authored_mesh_contact_rebased_through_"
            "measured_base_v4"
        ),
        "order8_natural_contact_contact_tangential_region_method": (
            "authored_mesh_sample_componentwise_tangential_region_with_"
            "pair_mean_base_centering_v3"
        ),
        "order8_natural_contact_contact_tangential_tolerance_m": (
            env.config.contact_tangential_tolerance_m
        ),
        "order8_natural_contact_provisional_contact_separation_allowed": True,
        "order8_natural_contact_contact_slip_enforcement_phase": (
            "grasp_latched_object_frame_contact_point_displacement_v1"
        ),
        "order8_natural_contact_contact_slip_measurement_method": (
            "force_weighted_selected_contact_centroid_object_frame_"
            "displacement_norm_from_grasp_confirmation_v1"
        ),
        "order8_natural_contact_contact_slip_speed_safe_hold_enabled": False,
        "order8_natural_contact_contact_break_enforcement_phase": (
            "after_verified_two_contact_grasp_dwell_until_planned_release_v2"
        ),
        "order8_natural_contact_max_provisional_acquisition_slip_speed_mps": (
            monitor_result.max_provisional_acquisition_slip_speed_mps
        ),
        "order8_natural_contact_object_motion_retargeting_enabled": True,
        "order8_natural_contact_object_motion_retarget_source": (
            "measured_free_object_pose_read_only_v1"
        ),
        "order8_natural_contact_object_follow_active_step_count": 100,
        "order8_natural_contact_max_observed_pre_qclose_object_translation_m": 0.01,
        "order8_natural_contact_max_observed_base_retarget_translation_m": 0.01,
        "order8_natural_contact_object_follow_pose_write_count": 0,
        "order8_natural_contact_morphology_aware_module_root_targets": True,
        "order8_natural_contact_module_root_target_source": (
            "whole_structure_fk_of_measured_absolute_dock_state_and_"
            "planner_base_pose_v4"
        ),
        "order8_natural_contact_module_frame_link_id": "fc",
        "order8_natural_contact_spawn_pose_conversion": (
            "graph_module_frame_to_urdf_root_v1"
        ),
        "order8_natural_contact_runtime_module_pose_source": (
            "isaac_named_module_frame_link_pose_and_twist_v1"
        ),
        "order8_natural_contact_qpid_centroidal_target_source": (
            "single_full_morphology_rigid_body_model_from_planner_base_pose_"
            "and_measured_absolute_dock_state_v5"
        ),
        "order8_natural_contact_qpid_joint_motion_assumption": (
            "quasi_static_measured_shape_without_commanded_joint_motion_"
            "compensation_even_during_slow_preload_v2"
        ),
        "order8_natural_contact_qpid_unreached_joint_target_compensation": False,
        "order8_natural_contact_morphology_aware_module_root_target_count": 4000,
        "order8_natural_contact_max_base_target_step_m": (
            env.config.base_translation_speed_limit_mps * env.simulation_dt_s
        ),
        "order8_natural_contact_max_contact_base_target_step_m": (
            env.config.contact_base_translation_speed_limit_mps * env.simulation_dt_s
        ),
        "order8_natural_contact_actuator_mapping_hash": build_actuator_mapping(
            graph,
            env.physical_model,
        ).stable_hash(),
        "order8_natural_contact_component_actuator_mapping_hashes": {
            str(module.module_id): build_actuator_mapping(
                order8_module._order8_component_graph(
                    graph,
                    module.module_id,
                ),
                env.physical_model,
            ).stable_hash()
            for module in graph.modules
        },
        "order8_natural_contact_free_object": True,
        "order8_natural_contact_object_kinematic": False,
        "order8_natural_contact_object_root_pose_write_count": 0,
        "order8_natural_contact_object_constraint_created": False,
        "order8_natural_contact_object_root_pose_write_audit_method": (
            "instrumented_post_spawn_object_pose_write_counter_v1"
        ),
        "order8_natural_contact_object_constraint_stage_audit_method": (
            "usd_physics_joint_body_target_scan_v1"
        ),
        "order8_natural_contact_object_constraint_reference_count": 0,
        "order8_natural_contact_object_constraint_prim_paths": [],
        "order8_natural_contact_pre_contact_object_pose_hold": False,
        "order8_natural_contact_kinematic_payload_attach_used": False,
        "order8_natural_contact_dynamic_assembly_filter_fallback_used": False,
        "order8_natural_contact_selected_surface_actual_dock_mesh": True,
        "order8_natural_contact_debug_command_mask_enabled": False,
        "order8_natural_contact_selected_surface_module_ids": sorted(
            [surface_pair.first.module_id, surface_pair.second.module_id]
        ),
        "order8_natural_contact_selected_surface_port_global_ids": sorted(
            [surface_pair.first.port_global_id, surface_pair.second.port_global_id]
        ),
        "order8_natural_contact_selected_surface_geometry_refs": geometry_refs,
        "order8_natural_contact_selected_gripper_material_method": (
            "selected_authored_dock_mesh_compliant_material_v3"
        ),
        "order8_natural_contact_selected_gripper_material_path": (
            ORDER8_SELECTED_GRIPPER_MATERIAL_PATH
        ),
        "order8_natural_contact_selected_gripper_static_friction": (
            env.config.selected_gripper_friction
        ),
        "order8_natural_contact_selected_gripper_dynamic_friction": (
            env.config.selected_gripper_friction
        ),
        "order8_natural_contact_selected_gripper_friction_combine_mode": (
            ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE
        ),
        "order8_natural_contact_selected_gripper_compliant_contact_enabled": True,
        "order8_natural_contact_selected_gripper_compliant_contact_stiffness_n_per_m": (
            env.config.selected_gripper_compliant_contact_stiffness_n_per_m
        ),
        "order8_natural_contact_selected_gripper_compliant_contact_damping_n_s_per_m": (
            env.config.selected_gripper_compliant_contact_damping_n_s_per_m
        ),
        "order8_natural_contact_selected_gripper_compliant_contact_audit_passed": True,
        "order8_natural_contact_selected_gripper_material_binding_strength": (
            "strongerThanDescendants"
        ),
        "order8_natural_contact_selected_gripper_material_body_paths": (
            material_body_paths
        ),
        "order8_natural_contact_selected_gripper_material_collision_prim_paths": (
            material_collision_paths
        ),
        "order8_natural_contact_selected_gripper_material_collision_prim_count": len(
            material_collision_paths
        ),
        "order8_natural_contact_selected_gripper_material_binding_audit_passed": True,
        "order8_natural_contact_gripper_clearance_geometry": (
            "urdf_collision_mesh_local_aabb_world_aabb_v1"
        ),
        "order8_natural_contact_gripper_clearance_mesh_aabb_count": len(geometry_refs),
        "order8_natural_contact_contact_report_body_counts": {
            "0": 20,
            "1": 20,
            "2": 20,
        },
        "order8_natural_contact_object_contact_report_body_count": 1,
        "order8_natural_contact_robot_object_contact_view_sensor_count": 60,
        "order8_natural_contact_robot_object_contact_view_filter_count": 1,
        "order8_natural_contact_selected_dock_link_ids": selected_link_ids,
        "order8_natural_contact_selected_contact_pair_count": 2,
        "order8_natural_contact_last_selected_normal_force_n_by_link": {
            link_id: 0.0 for link_id in selected_link_ids
        },
        "order8_natural_contact_max_selected_normal_force_n_by_link": {
            link_id: 11.5 for link_id in selected_link_ids
        },
        "order8_natural_contact_contact_closure_detection": (
            "simultaneous_selected_terminal_joint_load_dwell_then_measured_"
            "qclose_and_privileged_contact_validation_v18"
        ),
        "order8_natural_contact_contact_anchor_orientation_task_weight": (
            ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
        ),
        "order8_natural_contact_contact_anchor_task_hierarchy": (
            "contact_translation_primary_measured_orientation_zero_error_then_"
            "verified_absolute_joint_hold_v2"
        ),
        "order8_natural_contact_contact_terminal_inward_overtravel_m": (
            env.config.contact_closure_inward_overtravel_m
        ),
        "order8_natural_contact_provisional_surface_load_settle_method": (
            "one_sided_contact_may_separate_continuous_bounded_creep_until_"
            "simultaneous_nonprivileged_surface_load_qclose_v3"
        ),
        "order8_natural_contact_provisional_surface_load_settle_raw_contact_input": False,
        "order8_natural_contact_provisional_surface_load_settle_active_step_count": 0,
        "order8_natural_contact_contact_closure_raw_contact_input": False,
        "order8_natural_contact_contact_terminal_target_snapshotted": True,
        "order8_natural_contact_release_terminal_target_snapshotted": True,
        "order8_natural_contact_release_terminal_target_source": (
            "measured_closure_start_qopen_anchor_poses_base_v2"
        ),
        "order8_natural_contact_contact_configuration_latched": True,
        "order8_natural_contact_contact_closure_reason": (
            "dynamic_simultaneous_surface_region_arrest_then_"
            "load_limited_position_preload"
        ),
        "order8_natural_contact_contact_stall_latched": True,
        "order8_natural_contact_contact_stall_dwell_s": 0.0,
        "order8_natural_contact_contact_configuration_dwell_s": (
            env.config.contact_stall_dwell_s
        ),
        "order8_natural_contact_contact_stall_dwell_s_by_anchor": {
            str(anchor.anchor_id): 0.0
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_stall_latched_anchor_ids": sorted(
            anchor.anchor_id for anchor in graph.robot_anchors
        ),
        "order8_natural_contact_contact_stall_command_error_m_by_anchor": {
            str(anchor.anchor_id): 0.001
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_stall_anchor_speed_mps_by_anchor": {
            str(anchor.anchor_id): 0.02 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_stall_selected_joint_id_by_anchor": {
            str(anchor.anchor_id): (
                f"module_{anchor.module_id}:"
                f"{anchor.capability['dock_mechanism_joint_id']}"
            )
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_stall_selected_joint_load_nm_by_anchor": {
            str(anchor.anchor_id): 0.13 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_stall_selected_joint_load_threshold_nm": (
            ORDER8_CONTACT_STALL_RATED_TORQUE_FRACTION * 1.3
        ),
        "order8_natural_contact_contact_stall_selected_joint_load_source": (
            "absolute_per_anchor_terminal_mechanism_joint_isaac_applied_"
            "torque_minus_estimated_virtual_drive_damping_torque_v4"
        ),
        "order8_natural_contact_contact_stall_speed_reference_frame": (
            "first_order_low_pass_selected_mesh_sample_point_object_normal_"
            "relative_speed_v2"
        ),
        "order8_natural_contact_contact_configuration_base_speed_tolerance_mps": (
            env.config.pregrasp_linear_speed_tolerance_mps
        ),
        "order8_natural_contact_contact_configuration_base_speed_gate": (
            "world_base_linear_speed_with_object_relative_target_follow_v1"
        ),
        "order8_natural_contact_contact_mesh_clearance_arm_threshold_m": (
            env.config.contact_surface_arm_clearance_m
        ),
        "order8_natural_contact_contact_mesh_clearance_reacquire_tolerance_m": (
            env.config.contact_penetration_noise_floor_m
        ),
        "order8_natural_contact_contact_mesh_surface_distance_method": (
            "sampled_urdf_collision_mesh_surface_to_observed_object_obb_v1"
        ),
        "order8_natural_contact_contact_wrench_application_mapping": (
            "high_level_semantic_only_local_joint_offset_torque_forced_zero_v4"
        ),
        "order8_natural_contact_contact_wrench_application_raw_contact_input": False,
        "order8_natural_contact_contact_mesh_surface_sample_count": 8192,
        "order8_natural_contact_contact_mesh_surface_clearance_at_latch_m_by_anchor": {
            str(anchor.anchor_id): 0.0 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_contact_tangential_offset_at_latch_m_by_anchor": {
            str(anchor.anchor_id): [0.04, -0.03]
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_grasp_hold_anchor_target_source": (
            "simultaneous_surface_region_qclose_measured_anchor_poses_in_base_frame_v3"
        ),
        "order8_natural_contact_grasp_hold_anchor_count": len(graph.robot_anchors),
        "order8_natural_contact_contact_force_ramp_elapsed_s": (
            env.config.contact_force_ramp_s
        ),
        "order8_natural_contact_contact_force_ramp_elapsed_s_by_anchor": {
            str(anchor.anchor_id): env.config.contact_force_ramp_s
            for anchor in graph.robot_anchors
        },
        "order8_natural_contact_last_contact_force_scale": 1.0,
        "order8_natural_contact_last_contact_force_scale_by_anchor": {
            str(anchor.anchor_id): 1.0 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_max_contact_force_scale_by_anchor": {
            str(anchor.anchor_id): 1.0 for anchor in graph.robot_anchors
        },
        "order8_natural_contact_dock_joint_structural_lock_count": 0,
        "order8_natural_contact_whole_structure_kinematics_used": True,
        "order8_natural_contact_anchor_jacobian_column_count": len(dock_joint_ids),
        "order8_natural_contact_anchor_jacobian_ids": sorted(
            anchor.anchor_id for anchor in graph.robot_anchors
        ),
        "order8_natural_contact_dock_joint_physical_dof_count": len(dock_joint_ids),
        "order8_natural_contact_dock_joint_expected_ids": dock_joint_ids,
        "order8_natural_contact_dock_joint_observed_ids": dock_joint_ids,
        "order8_natural_contact_dock_joint_position_commanded_ids": dock_joint_ids,
        "order8_natural_contact_dock_joint_velocity_commanded_ids": dock_joint_ids,
        "order8_natural_contact_dock_joint_torque_bias_commanded_ids": dock_joint_ids,
        "order8_natural_contact_dock_torque_bias_limit_nm": 1.3,
        "order8_natural_contact_dock_torque_bias_limit_basis": (
            "ak40_10_continuous_torque_limit_v1"
        ),
        "order8_natural_contact_dock_continuous_torque_nm": 1.3,
        "order8_natural_contact_dock_peak_torque_nm": 4.1,
        "order8_natural_contact_dock_peak_current_a": 7.3,
        "order8_natural_contact_dock_actuator_telemetry_method": (
            "requested_unclipped_limited_isaac_target_computed_applied_speed_"
            "and_linear_current_estimate_v2"
        ),
        "order8_natural_contact_dock_velocity_limit_sim_rad_s": 3.0,
        "order8_natural_contact_dock_actuator_envelope_audit_passed": True,
        "order8_natural_contact_dock_actuator_envelope_violation_step_count": 0,
        "order8_natural_contact_dock_actuator_telemetry_maxima": {
            "abs_position_error_rad": 0.01,
            "abs_measured_velocity_radps": 0.5,
            "abs_requested_unclipped_torque_bias_nm": 0.0,
            "abs_requested_limited_torque_bias_nm": 0.0,
            "abs_isaac_effort_target_nm": 0.0,
            "abs_estimated_position_drive_torque_nm": 2.0,
            "abs_estimated_total_drive_torque_nm": 5.0,
            "abs_isaac_computed_torque_nm": 5.0,
            "abs_isaac_applied_torque_nm": 4.1,
            "estimated_current_a": 7.3,
        },
        "order8_natural_contact_ordered_phase_trace": list(
            ORDER8_NATURAL_CONTACT_REQUIRED_PHASES
        ),
        "order8_natural_contact_raw_contact_truth_role": "privileged_diagnostic_only",
        "order8_natural_contact_raw_contact_truth_actor_input": False,
        "order8_natural_contact_raw_contact_truth_qpid_command": False,
        "order8_natural_contact_raw_contact_failure_reasons": [],
        "order8_natural_contact_payload_feedforward_active_count": 500,
        "order8_natural_contact_payload_feedforward_method": (
            "verified_grasp_shared_commanded_lift_progress_and_centroidal_"
            "load_observer_known_payload_qpid_coupling_v7"
        ),
        "order8_natural_contact_payload_load_observer_method": (
            "aggregate_centroidal_external_vertical_force_delta_from_lift_start_"
            "normalized_by_known_payload_weight_v1"
        ),
        "order8_natural_contact_payload_load_observer_raw_contact_input": False,
        "order8_natural_contact_payload_load_transfer_driver": (
            "slew_limited_max_commanded_lift_progress_observed_load_after_"
            "verified_grasp_v3"
        ),
        "order8_natural_contact_payload_commanded_lift_progress_method": (
            "shared_lift_phase_elapsed_over_payload_transfer_duration_v1"
        ),
        "order8_natural_contact_last_payload_commanded_lift_progress_scale": 0.0,
        "order8_natural_contact_payload_commanded_lift_progress_peak_scale": 1.0,
        "order8_natural_contact_contact_motion_entry_speed_ramp_method": (
            "immediate_linear_lift_and_maintained_contact_phase_entry_ramp_v6"
        ),
        "order8_natural_contact_payload_feedforward_transition_duration_s": (
            env.config.payload_load_transfer_s
        ),
        "order8_natural_contact_payload_load_observer_valid_step_count": 100,
        "order8_natural_contact_payload_load_observer_invalid_step_count": 0,
        "order8_natural_contact_estimated_payload_lift_transfer_peak_scale": 1.0,
        "order8_natural_contact_payload_lift_off_confirmed_time_s": 20.0,
        "order8_natural_contact_payload_feedforward_max_lead_over_observed_scale": 0.0,
        "order8_natural_contact_payload_feedforward_max_lag_behind_commanded_"
        "progress_scale": 0.0,
        "order8_natural_contact_payload_feedforward_object_constraint": False,
        "order8_natural_contact_lift_acceleration_bias_method": (
            "known_payload_mass_times_shared_lift_progress_world_vertical_"
            "policy_command_residual_wrench_v1"
        ),
        "order8_natural_contact_lift_acceleration_bias_qpid_application": (
            "policy_command_residual_wrench_body_centroidal_only_v1"
        ),
        "order8_natural_contact_lift_acceleration_bias_raw_contact_input": False,
        "order8_natural_contact_lift_acceleration_bias_payload_mass_kg": (
            env.config.object_mass_kg
        ),
        "order8_natural_contact_lift_payload_acceleration_mps2": (
            env.config.lift_payload_acceleration_mps2
        ),
        "order8_natural_contact_lift_acceleration_bias_removal_s": (
            env.config.lift_acceleration_bias_removal_s
        ),
        "order8_natural_contact_lift_acceleration_bias_removal_method": (
            "cubic_smoothstep_zero_endpoint_slope_v1"
        ),
        "order8_natural_contact_lift_acceleration_bias_active_count": 100,
        "order8_natural_contact_lift_acceleration_bias_non_lift_active_count": 0,
        "order8_natural_contact_lift_acceleration_bias_policy_command_active_count": 100,
        "order8_natural_contact_lift_acceleration_bias_peak_scale": 0.75,
        "order8_natural_contact_last_lift_acceleration_bias_scale": 0.0,
        "order8_natural_contact_lift_acceleration_bias_lift_off_scale": 0.75,
        "order8_natural_contact_lift_acceleration_bias_removal_complete_time_s": 20.5,
        "order8_natural_contact_lift_acceleration_bias_peak_force_world_z_n": 0.75,
        "order8_natural_contact_lift_acceleration_bias_peak_residual_force_"
        "body_norm_n": 0.75,
        "order8_natural_contact_last_lift_acceleration_bias_force_world_z_n": 0.0,
        "order8_natural_contact_last_lift_acceleration_residual_wrench_body": [
            0.0
        ]
        * 6,
        "order8_natural_contact_constraint_identity_failures": [],
        "order8_natural_contact_monitor_result": monitor_result.to_dict(),
        "order8_natural_contact_monitor_result_hash": stable_hash(monitor_result),
        "order8_natural_contact_failure_reason": None,
        **{
            key: 0
            for key in (
                "order8_natural_contact_raw_contact_invalid_count",
                "order8_natural_contact_raw_contact_saturation_count",
                "order8_natural_contact_unintended_contact_count",
                "order8_natural_contact_object_drop_count",
                "order8_natural_contact_post_release_selected_contact_count",
                "order8_natural_contact_qp_infeasible_count",
                "order8_natural_contact_controller_failure_count",
                "order8_natural_contact_missing_actuator_target_count",
                "order8_natural_contact_unsupported_actuator_target_count",
                "order8_natural_contact_clipped_actuator_target_count",
                "order8_natural_contact_unresolved_actuator_target_count",
            )
        },
    }
