from __future__ import annotations

import math

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_calibration import (
    ORDER9_R1_CALIBRATION_APPROVAL_RECORD_VERSION,
    ORDER9_R1_CALIBRATION_PROTOCOL_VERSION,
    Order9R1CalibrationProtocol,
    load_order9_r1_calibration_approval_record,
    load_order9_r1_calibration_protocol,
)

_PROPOSAL = "configs/training/order9_r1_calibration_protocol_v1_proposal.yaml"
_APPROVED = "configs/training/order9_r1_calibration_protocol_v1.yaml"
_APPROVAL = "for_codex/R1_CALIBRATION_PROTOCOL_V1_APPROVAL.json"


def test_r1_calibration_proposal_is_valid_but_not_approved() -> None:
    protocol = load_order9_r1_calibration_protocol(
        _PROPOSAL,
        repository_root=".",
    )

    assert protocol.protocol_version == ORDER9_R1_CALIBRATION_PROTOCOL_VERSION
    assert protocol.calibration_gate_id == "r1-calibration-gate-v1"
    assert protocol.status == "proposed"
    assert protocol.approval_record is None
    assert protocol.source_bucket_count == 36
    assert protocol.source_module_counts == tuple(range(2, 9))
    assert (protocol.selection_split, protocol.selection_bucket_count) == (
        "train",
        22,
    )
    assert (protocol.confirmation_split, protocol.confirmation_bucket_count) == (
        "validation",
        14,
    )
    assert protocol.lattice_point_count_per_bucket == 27
    assert protocol.teacher_lattice_case_count_per_level == 594
    assert protocol.teacher_interior_case_count_per_level == 88
    assert protocol.isaac_episode_count_per_level == 1364
    assert protocol.confirmation_teacher_case_count == 434
    assert protocol.confirmation_isaac_episode_count == 868
    assert protocol.random_interior_seed_derivation == "r1_derived_seed_v1"
    assert protocol.fast_screen_phases == ("approach", "contact_acquisition")
    assert protocol.fast_screen_reuses_resolved_trajectory
    assert protocol.fast_screen_isaac_forbidden
    assert protocol.fast_screen_controller_layers_forbidden
    assert protocol.fast_screen_ik_resolve_forbidden
    assert protocol.fast_screen_trajectory_optimization_forbidden
    assert protocol.fast_screen_applies_to_calibration_and_collection
    assert protocol.full_control_test_requires_fast_screen_pass
    assert protocol.minimum_normalized_joint_limit_reserve == pytest.approx(0.01)
    assert math.degrees(protocol.maximum_body_tilt_rad) == pytest.approx(60.0)
    assert protocol.minimum_fast_screen_pass_rate == pytest.approx(0.95)
    assert protocol.collection_episode_count == 140
    assert (
        protocol.collection_train_episode_count,
        protocol.collection_validation_episode_count,
        protocol.collection_held_out_episode_count,
    ) == (98, 21, 21)
    assert (
        protocol.collection_train_per_module_episode_count,
        protocol.collection_validation_per_module_episode_count,
        protocol.collection_held_out_per_module_episode_count,
    ) == (14, 3, 3)
    assert [level.position_half_span_m for level in protocol.calibration_levels] == [
        0.01,
        0.02,
        0.03,
        0.04,
    ]
    yaw_degrees = [
        math.degrees(level.yaw_half_span_rad) for level in protocol.calibration_levels
    ]
    assert yaw_degrees == pytest.approx([5.0, 10.0, 15.0, 20.0])


def test_r1_calibration_protocol_rejects_skipped_or_weakened_gates() -> None:
    protocol = load_order9_r1_calibration_protocol(
        _PROPOSAL,
        repository_root=".",
    )
    invalid = protocol.to_dict()
    invalid["lattice_point_count_per_bucket"] = 26
    with pytest.raises(SchemaValidationError, match="3x3x3 lattice"):
        Order9R1CalibrationProtocol.from_dict(invalid)

    invalid = protocol.to_dict()
    invalid["minimum_lattice_pass_rate"] = 0.99
    with pytest.raises(SchemaValidationError, match="lattice must pass 100%"):
        Order9R1CalibrationProtocol.from_dict(invalid)

    invalid = protocol.to_dict()
    invalid["calibration_records_training_eligible"] = True
    with pytest.raises(SchemaValidationError, match="not training data"):
        Order9R1CalibrationProtocol.from_dict(invalid)

    invalid = protocol.to_dict()
    invalid["fast_screen_isaac_forbidden"] = False
    with pytest.raises(SchemaValidationError, match="fast-screen/full-control"):
        Order9R1CalibrationProtocol.from_dict(invalid)

    invalid = protocol.to_dict()
    invalid["collection_train_per_module_episode_count"] = 13
    with pytest.raises(SchemaValidationError, match="split-by-module allocation"):
        Order9R1CalibrationProtocol.from_dict(invalid)

    invalid = protocol.to_dict()
    invalid["collection_cross_split_pose_disjoint_required"] = False
    with pytest.raises(SchemaValidationError, match="uniqueness/split-module"):
        Order9R1CalibrationProtocol.from_dict(invalid)


def test_r1_calibration_protocol_cannot_be_approved_without_record() -> None:
    protocol = load_order9_r1_calibration_protocol(
        _PROPOSAL,
        repository_root=".",
    )
    invalid = protocol.to_dict()
    invalid["status"] = "approved"

    with pytest.raises(SchemaValidationError, match="requires an approval record"):
        Order9R1CalibrationProtocol.from_dict(invalid)


def test_r1_calibration_approval_is_hash_bound_to_reviewed_protocol() -> None:
    protocol = load_order9_r1_calibration_protocol(_APPROVED, repository_root=".")
    approval = load_order9_r1_calibration_approval_record(
        _APPROVAL,
        repository_root=".",
    )

    assert protocol.status == "approved"
    assert protocol.approval_record == _APPROVAL
    assert approval.record_version == ORDER9_R1_CALIBRATION_APPROVAL_RECORD_VERSION
    assert approval.decision == "approved"
    assert approval.approved_protocol.path == _APPROVED
    assert set(approval.approval_scope) == {
        "four_level_numeric_ladder",
        "selection_on_22_train_buckets",
        "single_confirmation_on_14_validation_buckets",
        "confirmation_failure_rejects_v1",
        "fast_screen_before_full_control",
        "one_percent_joint_limit_reserve",
        "sixty_degree_body_tilt_ceiling",
        "formal_collection_140_episodes",
    }
