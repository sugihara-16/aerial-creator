from __future__ import annotations

from pathlib import Path

from amsrr.training import order9_r1_nominal_calibration_v4 as nominal_v4

REPOSITORY = Path(__file__).resolve().parents[3]


def test_r1_nominal_v4_contract_binds_early_safe_teacher() -> None:
    protocol, approval = nominal_v4.load_order9_r1_nominal_calibration_v4_contract(
        REPOSITORY
    )

    assert approval["decision"] == "approved"
    assert protocol["contact_solution_tilt_checked_before_dense_path"] is True
    assert (
        protocol["extreme_contact_solution_removed_before_configuration_space_planning"]
        is True
    )
    assert protocol["alternative_unoccupied_surface_pair_allowed"] is True
    assert protocol["maximum_body_tilt_rad"] == 1.0471975511965976


def test_r1_nominal_v4_preserves_numeric_and_control_boundaries() -> None:
    protocol, _approval = nominal_v4.load_order9_r1_nominal_calibration_v4_contract(
        REPOSITORY
    )

    assert protocol["minimum_normalized_joint_limit_reserve"] == 0.01
    assert (
        protocol["grasp_point_position_error_metric"]
        == "three_dimensional_euclidean_distance"
    )
    assert protocol["maximum_grasp_point_position_error_m"] == 0.030
    assert protocol["pi_l_system_retained"] is True
    assert protocol["pi_l_actor_command_applied"] is False
    assert protocol["nominal_preload_applied"] is True
    assert protocol["qpid_qp_applied"] is True
    assert protocol["local_servo_applied"] is True
    assert protocol["isaac_required_after_fast_screen"] is True


def test_r1_nominal_v4_protocol_alone_does_not_authorize_collection() -> None:
    protocol, _approval = nominal_v4.load_order9_r1_nominal_calibration_v4_contract(
        REPOSITORY
    )

    assert protocol["selection_bucket_count"] == 22
    assert protocol["confirmation_bucket_count"] == 14
    assert protocol["candidate_count_per_bucket"] == 31
    assert protocol["replay_count_per_screen_accepted_candidate"] == 2
    assert protocol["formal_teacher_collection_authorized_by_protocol_alone"] is False
