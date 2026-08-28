from __future__ import annotations

from pathlib import Path

from amsrr.training import order9_r1_nominal_calibration_v3 as nominal_v3
from amsrr.training.order9_r1_nominal_calibration import (
    ORDER9_R1_NOMINAL_EXECUTION_CONTRACT,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def test_r1_nominal_v3_contract_binds_30mm_radius_and_joint_reserve() -> None:
    protocol, approval = nominal_v3.load_order9_r1_nominal_calibration_v3_contract(
        REPOSITORY
    )

    assert approval["decision"] == "approved"
    assert protocol["joint_limit_reserve_projection_enabled"] is True
    assert protocol["minimum_normalized_joint_limit_reserve"] == 0.01
    assert (
        protocol["grasp_point_position_error_metric"]
        == "three_dimensional_euclidean_distance"
    )
    assert protocol["maximum_grasp_point_position_error_m"] == 0.030
    assert protocol["per_axis_plus_minus_30mm_contract"] is False


def test_r1_nominal_v3_preserves_nominal_control_boundary() -> None:
    protocol, _approval = nominal_v3.load_order9_r1_nominal_calibration_v3_contract(
        REPOSITORY
    )

    assert protocol["base_nominal_isaac_execution_contract"] == (
        ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
    )
    assert protocol["pi_l_system_retained"] is True
    assert protocol["pi_l_actor_command_applied"] is False
    assert protocol["nominal_preload_applied"] is True
    assert protocol["qpid_qp_applied"] is True
    assert protocol["local_servo_applied"] is True
    assert protocol["isaac_required_after_fast_screen"] is True


def test_r1_nominal_v3_reuses_ladder_but_does_not_authorize_collection() -> None:
    protocol, _approval = nominal_v3.load_order9_r1_nominal_calibration_v3_contract(
        REPOSITORY
    )

    assert protocol["reuse_base_four_level_numeric_ladder"] is True
    assert protocol["selection_bucket_count"] == 22
    assert protocol["confirmation_bucket_count"] == 14
    assert protocol["candidate_count_per_bucket"] == 31
    assert protocol["replay_count_per_screen_accepted_candidate"] == 2
    assert protocol["formal_teacher_collection_authorized_by_protocol_alone"] is False
