from __future__ import annotations

import math
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_calibration_runner import (
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_complete_task_geometry_v11 import (
    ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_VERSION,
    _obstacle_scene_admission,
    require_order9_r1_complete_task_geometry_v11,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    build_order9_r1_explicit_support_collision_object_v11,
    build_order9_r1_geometry_variants_v11,
    load_order9_r1_nominal_geometry_v11_contract,
)

ROOT = Path(__file__).resolve().parents[3]
BASE_PROTOCOL = ROOT / "configs/training/order9_r1_calibration_protocol_v1.yaml"
FAILED_V10_CANDIDATE = "r1_l2_20mm_10deg__train__train-000000-5fecfad4f44f__lattice_02"


def _case():
    return next(
        value
        for value in enumerate_order9_r1_calibration_level_cases(
            protocol_path=BASE_PROTOCOL,
            repository_root=ROOT,
            level_id="r1_l2_20mm_10deg",
            split="train",
        )
        if value.candidate_id == FAILED_V10_CANDIDATE
    )


def _contract():
    return load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=ROOT,
    )


def _object_and_size(task):
    obj = next(value for value in task.scene.objects if value.movable)
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == obj.geometry_id
    )
    return obj, tuple(geometry.primitive_params["size_m"])


def test_nominal_geometry_raises_payload_and_preserves_bottom_and_mass() -> None:
    variants = build_order9_r1_geometry_variants_v11(_case(), contract=_contract())
    small_object, small_size = _object_and_size(variants.small_object_task_spec)
    nominal_object, nominal_size = _object_and_size(variants.nominal_case.task_spec)
    assert nominal_size[:2] == pytest.approx(small_size[:2])
    assert nominal_size[2] == pytest.approx(small_size[2] + 0.04)
    assert nominal_object.pose_world[2] == pytest.approx(
        small_object.pose_world[2] + 0.02
    )
    assert nominal_object.pose_world[2] - 0.5 * nominal_size[2] == pytest.approx(
        small_object.pose_world[2] - 0.5 * small_size[2]
    )
    assert nominal_object.mass_kg == pytest.approx(small_object.mass_kg)
    assert nominal_object.density_kg_m3 == pytest.approx(
        nominal_object.mass_kg / math.prod(nominal_size)
    )
    assert variants.small_object_task_spec.metadata["r1_geometry_role"] == (
        "small_object_difficult_condition"
    )
    assert (
        variants.small_object_task_spec.metadata["r1_formal_range_selection_eligible"]
        is False
    )
    assert (
        variants.nominal_case.task_spec.metadata["r1_formal_range_selection_eligible"]
        is True
    )


def test_task_support_is_identical_to_hash_bound_isaac_support() -> None:
    contract = _contract()
    variants = build_order9_r1_geometry_variants_v11(_case(), contract=contract)
    collision = build_order9_r1_explicit_support_collision_object_v11(
        variants.nominal_case.task_spec,
        contract=contract,
    )
    assert len(collision.environment_boxes) == 1
    assert collision.environment_boxes[0].size_m == pytest.approx(
        contract.support_size_m
    )
    assert collision.environment_boxes[0].pose_world == pytest.approx(
        contract.support_pose_world
    )
    with pytest.raises(SchemaValidationError, match="Isaac canonical support"):
        build_order9_r1_explicit_support_collision_object_v11(
            _case().task_spec,
            contract=contract,
        )


def test_eight_phase_geometry_admission_is_fail_closed() -> None:
    accepted = {
        "audit_version": ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_VERSION,
        "accepted": True,
        "status": "accepted",
        "checked_phases": [
            "approach",
            "contact_acquisition",
            "lift",
            "transport",
            "place",
            "release",
            "retreat",
            "settle",
        ],
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "formal_isaac_admission": True,
    }
    require_order9_r1_complete_task_geometry_v11(accepted)
    rejected = {**accepted, "accepted": False, "status": "rejected"}
    with pytest.raises(SchemaValidationError, match="geometry admission rejected"):
        require_order9_r1_complete_task_geometry_v11(rejected)


def test_exact_geometry_separates_self_contact_from_support_clearance() -> None:
    close_but_not_intersecting_self = {
        "worst_pairs": [
            {
                "second_kind": "robot",
                "clearance_m": 7.0e-9,
                "colliding": False,
            }
        ],
        "minimum_ground_clearance_m": 0.1,
    }
    accepted, support_lower_bound, violations = _obstacle_scene_admission(
        close_but_not_intersecting_self,
        required_obstacle_clearance_m=0.0195,
        required_ground_clearance_m=0.0,
    )
    assert accepted is True
    assert support_lower_bound == pytest.approx(0.0195)
    assert violations == []

    support_too_close = {
        **close_but_not_intersecting_self,
        "worst_pairs": [
            {
                "second_kind": "object",
                "clearance_m": 0.003,
                "colliding": False,
            }
        ],
    }
    accepted, support_lower_bound, violations = _obstacle_scene_admission(
        support_too_close,
        required_obstacle_clearance_m=0.0195,
        required_ground_clearance_m=0.0,
    )
    assert accepted is False
    assert support_lower_bound == pytest.approx(0.003)
    assert violations[0]["required_clearance_m"] == pytest.approx(0.0195)

    self_intersection = {
        **close_but_not_intersecting_self,
        "worst_pairs": [
            {
                "second_kind": "robot",
                "clearance_m": -0.001,
                "colliding": True,
            }
        ],
    }
    accepted, _, violations = _obstacle_scene_admission(
        self_intersection,
        required_obstacle_clearance_m=0.0195,
        required_ground_clearance_m=0.0,
    )
    assert accepted is False
    assert violations[0]["required_state"] == "not_intersecting"
