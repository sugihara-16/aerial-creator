from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
    ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION,
    ORDER8_NATURAL_CONTACT_RESULT_VERSION,
    Order8NaturalContactConfig,
    Order8NaturalContactObservation,
    Order8NaturalContactPhase,
    Order8NaturalContactResult,
    Order8RawContactPatch,
    load_order8_natural_contact_config,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "training"
    / "order8_natural_contact.yaml"
)


def test_order8_config_loads_the_frozen_natural_contact_thresholds() -> None:
    config = load_order8_natural_contact_config(_CONFIG_PATH)

    assert config.object_mass_kg == 1.0
    assert config.object_size_m == [0.30, 0.40, 0.15]
    assert config.object_support_height_m == 0.15
    assert config.initial_object_standoff_m == 0.50
    assert config.object_friction == 0.6
    assert config.floor_friction == 0.8
    assert config.selected_gripper_friction == 4.5
    assert config.selected_gripper_compliant_contact_stiffness_n_per_m == 7500.0
    assert config.selected_gripper_compliant_contact_damping_n_s_per_m == 75.0
    assert config.required_distinct_dock_links == 2
    assert config.contact_normal_force_threshold_n == 0.5
    assert config.contact_penetration_noise_floor_m == 0.0001
    assert config.contact_dwell_s == 0.25
    assert config.normal_force_target_per_contact_n == 11.0
    assert config.max_force_per_contact_n == 30.0
    assert config.max_torque_per_contact_nm == 5.0
    assert config.max_penetration_m == 0.002
    assert config.max_tangential_slip_speed_mps == 0.02
    assert config.max_contact_point_slip_displacement_m == 0.030
    assert config.contact_break_grace_s == 0.05
    assert config.minimum_lift_clearance_m == 0.100
    assert config.downward_drop_velocity_threshold_mps == 0.25
    assert config.required_transport_distance_m == 0.200
    assert config.base_translation_speed_limit_mps == 0.10
    assert config.contact_base_translation_speed_limit_mps == 0.010
    assert config.payload_load_transfer_s == 1.0
    assert config.lift_payload_acceleration_mps2 == 1.0
    assert config.lift_acceleration_bias_removal_s == 0.5
    assert config.hover_dwell_s == 2.0
    assert config.pregrasp_mesh_clearance_m == 0.050
    assert config.pregrasp_position_tolerance_m == 0.080
    assert config.pregrasp_linear_speed_tolerance_mps == 0.020
    assert config.contact_axial_min_mesh_overlap_m == 0.050
    assert config.anchor_translation_speed_limit_mps == 0.010
    assert config.anchor_reference_terminal_tolerance_m == 0.001
    assert config.anchor_command_tracking_tolerance_m == 0.020
    assert config.contact_near_surface_slowdown_m == 0.015
    assert config.contact_surface_arm_clearance_m == 0.003
    assert config.contact_tangential_tolerance_m == 0.050
    assert config.contact_surface_creep_speed_limit_mps == 0.001
    assert config.contact_clearance_sync_deadband_m == 0.0005
    assert config.contact_clearance_sync_full_slowdown_m == 0.0015
    assert config.contact_clearance_sync_minimum_speed_scale == 0.05
    assert config.contact_closure_inward_overtravel_m == 0.004
    assert config.contact_stall_anchor_speed_threshold_mps == 0.0015
    assert config.contact_stall_dwell_s == 0.10
    assert config.contact_centering_max_offset_m == 0.030
    assert config.contact_centering_max_tilt_rad == 0.020
    assert config.contact_centering_xy_p_gain == 12.0
    assert config.contact_centering_xy_d_gain == 4.0
    assert config.contact_centering_roll_pitch_p_gain == 60.0
    assert config.contact_centering_roll_pitch_d_gain == 24.0
    assert config.contact_yield_ramp_down_s == 0.05
    assert config.contact_yield_ramp_up_s == 0.50
    assert config.contact_yield_integrator_decay_rate_per_s == 12.0
    assert config.contact_external_wrench_filter_time_constant_s == 0.05
    assert config.contact_external_wrench_bias_time_constant_s == 0.50
    assert config.contact_admittance_force_deadband_n == 0.5
    assert config.contact_admittance_torque_deadband_nm == 0.05
    assert config.contact_admittance_linear_gain_mps_per_n == 0.0015
    assert config.contact_admittance_angular_gain_radps_per_nm == 0.03
    assert config.contact_admittance_max_linear_speed_mps == 0.020
    assert config.contact_admittance_max_angular_speed_radps == 0.15
    assert config.contact_admittance_max_translation_offset_m == 0.030
    assert config.contact_yield_joint_drive_stiffness_scale == 0.25
    assert config.contact_yield_joint_drive_damping_nms_per_rad == 8.0
    assert config.contact_joint_velocity_limit_radps == 0.10
    assert config.contact_position_preload_joint_speed_radps == 0.002
    assert config.contact_position_preload_load_threshold_nm == 1.2
    assert config.contact_force_ramp_s == 40.0
    assert config.contact_joint_drive_damping_multiplier == 1.0
    assert config.contact_acquisition_timeout_s == 90.0
    assert config.simultaneous_reachability_absolute_tolerance == 0.010
    assert config.joint_limit_state_tolerance_rad == 0.005
    assert config.release_contact_free_dwell_s == 0.10
    assert config.gripper_retreat_clearance_m == 0.050
    assert config.settle_linear_speed_mps == 0.05
    assert config.settle_angular_speed_rad_s == 0.10
    assert config.post_release_settle_dwell_s == 1.0
    assert config.raw_contact_truth_role == "privileged_diagnostic_only"
    assert config.raw_contact_truth_actor_input is False
    assert config.raw_contact_truth_qpid_command is False
    assert Order8NaturalContactConfig.from_json(config.to_json()) == config


def test_order8_raw_truth_cannot_be_enabled_for_actor_or_qpid() -> None:
    config = Order8NaturalContactConfig().to_dict()
    config["raw_contact_truth_actor_input"] = True
    with pytest.raises(SchemaValidationError, match="one of"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["raw_contact_truth_qpid_command"] = True
    with pytest.raises(SchemaValidationError, match="one of"):
        Order8NaturalContactConfig.from_dict(config)


def test_order8_position_preload_speed_cannot_exceed_joint_limit() -> None:
    payload = Order8NaturalContactConfig().to_dict()
    payload["contact_position_preload_joint_speed_radps"] = 0.2
    with pytest.raises(SchemaValidationError, match="position-preload speed"):
        Order8NaturalContactConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "lift_payload_acceleration_mps2",
        "lift_acceleration_bias_removal_s",
        "selected_gripper_compliant_contact_stiffness_n_per_m",
        "selected_gripper_compliant_contact_damping_n_s_per_m",
    ],
)
def test_order8_lift_acceleration_settings_must_be_positive(
    field_name: str,
) -> None:
    payload = Order8NaturalContactConfig().to_dict()
    payload[field_name] = 0.0
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        Order8NaturalContactConfig.from_dict(payload)


def test_order8_drop_and_penetration_thresholds_are_validated() -> None:
    config = Order8NaturalContactConfig().to_dict()
    config["selected_gripper_friction"] = 0.5 * config["object_friction"]
    with pytest.raises(SchemaValidationError, match="selected-gripper friction"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["downward_drop_velocity_threshold_mps"] = 0.0
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_penetration_noise_floor_m"] = 0.003
    with pytest.raises(SchemaValidationError, match="noise floor"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["base_translation_speed_limit_mps"] = 0.0
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_base_translation_speed_limit_mps"] = 0.20
    with pytest.raises(SchemaValidationError, match="contact base speed"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["pregrasp_mesh_clearance_m"] = config["initial_object_standoff_m"]
    with pytest.raises(SchemaValidationError, match="pregrasp mesh clearance"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_tangential_tolerance_m"] = 0.5 * min(config["object_size_m"])
    with pytest.raises(SchemaValidationError, match="tangential contact tolerance"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["joint_limit_state_tolerance_rad"] = 0.0
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_acquisition_timeout_s"] = 0.0
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_surface_arm_clearance_m"] = config[
        "contact_penetration_noise_floor_m"
    ]
    with pytest.raises(SchemaValidationError, match="arm clearance"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_stall_anchor_speed_threshold_mps"] = (
        1.01 * config["max_tangential_slip_speed_mps"]
    )
    with pytest.raises(SchemaValidationError, match="contact-stall speed"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_surface_creep_speed_limit_mps"] = (
        0.21 * config["anchor_translation_speed_limit_mps"]
    )
    with pytest.raises(SchemaValidationError, match="surface creep speed"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_clearance_sync_full_slowdown_m"] = config[
        "contact_clearance_sync_deadband_m"
    ]
    with pytest.raises(SchemaValidationError, match="clearance synchronization"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_clearance_sync_minimum_speed_scale"] = 1.1
    with pytest.raises(SchemaValidationError, match="minimum speed scale"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_admittance_max_linear_speed_mps"] = (
        1.01 * config["max_tangential_slip_speed_mps"]
    )
    with pytest.raises(SchemaValidationError, match="admittance linear speed"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_admittance_max_translation_offset_m"] = (
        1.01 * config["contact_tangential_tolerance_m"]
    )
    with pytest.raises(SchemaValidationError, match="admittance translation offset"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_joint_drive_damping_multiplier"] = 0.99
    with pytest.raises(SchemaValidationError, match="damping multiplier"):
        Order8NaturalContactConfig.from_dict(config)

    config = Order8NaturalContactConfig().to_dict()
    config["contact_yield_joint_drive_stiffness_scale"] = 1.01
    with pytest.raises(SchemaValidationError, match="stiffness scale"):
        Order8NaturalContactConfig.from_dict(config)


def test_order8_observation_requires_distinct_selected_links_and_valid_patch_magnitudes() -> (
    None
):
    payload = _observation().to_dict()
    payload["selected_dock_link_ids"] = ["dock_a", "dock_a"]
    with pytest.raises(SchemaValidationError, match="must be distinct"):
        Order8NaturalContactObservation.from_dict(payload)

    with pytest.raises(SchemaValidationError, match="must not exceed"):
        Order8RawContactPatch(
            patch_id="bad",
            robot_link_id="dock_a",
            other_body_id="payload",
            normal_force_n=2.0,
            force_magnitude_n=1.0,
            torque_magnitude_nm=0.0,
            penetration_m=0.0,
            tangential_slip_speed_mps=0.0,
        )


def test_order8_observation_accepts_signed_finite_world_vertical_velocity() -> None:
    payload = _observation().to_dict()
    payload["object_vertical_velocity_world_mps"] = -0.25
    observation = Order8NaturalContactObservation.from_dict(payload)
    assert observation.object_vertical_velocity_world_mps == -0.25

    payload["object_vertical_velocity_world_mps"] = float("nan")
    with pytest.raises(SchemaValidationError, match="must be finite"):
        Order8NaturalContactObservation.from_dict(payload)


def test_order8_result_pass_flag_is_recomputed_from_all_gates() -> None:
    common = {
        "result_version": ORDER8_NATURAL_CONTACT_RESULT_VERSION,
        "config_version": ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
        "config_hash": "0" * 64,
        "contact_model": "natural_contact_grasp_v1",
        "final_phase": "complete",
        "attempted": True,
        "step_count": 100,
        "duration_s": 2.0,
        "selected_dock_link_ids": ["dock_a", "dock_b"],
        "grasp_acquired": True,
        "lift_acquired": True,
        "transport_acquired": True,
        "release_contact_free_acquired": True,
        "retreat_clearance_acquired": True,
        "settle_acquired": True,
        "object_dropped": False,
        "unintended_contact_count": 0,
        "max_force_per_selected_contact_n": 11.0,
        "max_torque_per_selected_contact_nm": 0.1,
        "max_penetration_m": 0.0005,
        "max_tangential_slip_speed_mps": 0.001,
        "max_provisional_acquisition_slip_speed_mps": 0.03,
        "max_contact_point_slip_displacement_m_by_link": {
            "dock_a": 0.001,
            "dock_b": 0.001,
        },
        "failure_reasons": [],
    }
    result = Order8NaturalContactResult.from_dict({**common, "passed": True})
    assert result.passed is True

    with pytest.raises(SchemaValidationError, match="passed"):
        Order8NaturalContactResult.from_dict(
            {**common, "passed": True, "settle_acquired": False}
        )


def _observation() -> Order8NaturalContactObservation:
    return Order8NaturalContactObservation(
        observation_version=ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION,
        phase=Order8NaturalContactPhase.APPROACH,
        time_s=0.0,
        step_dt_s=0.01,
        object_id="payload",
        selected_dock_link_ids=["dock_a", "dock_b"],
        raw_contact_patches=[],
        selected_contact_point_object_m_by_link={},
        raw_contact_valid=True,
        raw_contact_saturated=False,
        object_bottom_clearance_m=0.0,
        object_floor_contact=True,
        object_linear_speed_mps=0.0,
        object_vertical_velocity_world_mps=0.0,
        object_angular_speed_rad_s=0.0,
        transport_distance_m=0.0,
        gripper_object_clearance_m=0.0,
        controller_qp_feasible=True,
        simultaneous_qclose_acquired=False,
        grasp_confirmation_ready=False,
    )
