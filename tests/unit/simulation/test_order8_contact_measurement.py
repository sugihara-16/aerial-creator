from __future__ import annotations

import math

import pytest

from amsrr.simulation.order8_contact_measurement import (
    Order8AABB,
    Order8ContactMeasurementError,
    aabb_clearance_m,
    gripper_object_clearance_m,
    measure_order8_raw_contacts,
    object_bottom_clearance_m,
    object_transport_distance_m,
    oriented_cuboid_world_aabb,
    relative_point_normal_speed_mps,
    relative_point_normal_velocity_mps,
    relative_point_speed_mps,
)

IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _measurement_kwargs() -> dict[str, object]:
    return {
        "sensor_body_ids": ["sensor_a", "sensor_b", "sensor_other"],
        "sensor_global_link_ids": [
            "module_1::yaw_dock_mech2",
            "module_2::pitch_dock_mech1",
            "module_0::fc",
        ],
        "selected_global_link_ids": [
            "module_1::yaw_dock_mech2",
            "module_2::pitch_dock_mech1",
        ],
        "object_body_id": "payload",
        "contact_counts": [1, 1, 0],
        "start_indices": [0, 1, 2],
        "patch_forces_n": [5.0, -7.0, 0.0, 0.0],
        "patch_points_world": [
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ],
        "patch_normals_world": [
            (0.0, 0.0, 2.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
        "patch_separations_m": [-0.002, 0.001, 1.0, 1.0],
        "raw_capacity": 4,
        "body_com_poses_world": {
            "sensor_a": IDENTITY_POSE,
            "sensor_b": IDENTITY_POSE,
            "sensor_other": IDENTITY_POSE,
        },
        "body_twists_world": {
            "sensor_a": (1.0, 0.0, 0.0, 0.0, 0.0, 2.0),
            "sensor_b": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "sensor_other": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
        "object_com_pose_world": IDENTITY_POSE,
        "object_twist_world": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }


def test_converts_nonaggregated_patch_magnitudes_without_force_cancellation() -> None:
    measurement = measure_order8_raw_contacts(**_measurement_kwargs())

    assert measurement.raw_contact_valid is True
    assert measurement.raw_contact_saturated is False
    assert measurement.selected_contact is True
    assert measurement.selected_raw_contact_count == 2
    assert measurement.selected_force_magnitude_sum_n == 12.0
    assert measurement.unintended_contact is False
    assert len(measurement.raw_contact_patches) == 2
    first, second = measurement.raw_contact_patches
    assert first.robot_link_id == "module_1::yaw_dock_mech2"
    assert first.patch_id == "module_1::yaw_dock_mech2::payload::patch_0000"
    assert first.normal_force_n == 5.0
    assert first.force_magnitude_n == 5.0
    assert first.torque_magnitude_nm == pytest.approx(5.0)
    assert first.penetration_m == 0.002
    # v + omega x r = (1, 0, 0) + (-2, 0, 0); all is tangential to z.
    assert first.tangential_slip_speed_mps == pytest.approx(1.0)
    assert second.normal_force_n == 7.0
    assert second.force_magnitude_n == 7.0
    assert second.penetration_m == 0.0


def test_empty_contact_set_with_empty_sparse_buffers_is_valid() -> None:
    values = _measurement_kwargs()
    values["contact_counts"] = [0, 0, 0]
    # Sparse backends may retain in-capacity offsets even when no range is active.
    values["start_indices"] = [0, 2, 4]
    values["patch_forces_n"] = []
    values["patch_points_world"] = []
    values["patch_normals_world"] = []
    values["patch_separations_m"] = []
    # Inactive sensors need no COM/twist fetch; their stable identity remains
    # covered by the contact-view layout itself.
    values["body_com_poses_world"] = {}
    values["body_twists_world"] = {}

    measurement = measure_order8_raw_contacts(**values)

    assert measurement.raw_contact_valid is True
    assert measurement.raw_contact_saturated is False
    assert measurement.raw_contact_count == 0
    assert measurement.raw_contact_capacity == 4
    assert measurement.raw_contact_patches == []
    assert measurement.selected_contact is False
    assert measurement.unintended_contact is False
    assert measurement.failure_reasons == ()


def test_relative_slip_includes_both_body_angular_velocities_at_patch() -> None:
    values = _measurement_kwargs()
    values["contact_counts"] = [1, 0, 0]
    values["body_twists_world"] = {
        "sensor_a": (0.0, 0.0, 0.0, 0.0, 0.0, 2.0),
        "sensor_b": (0.0,) * 6,
        "sensor_other": (0.0,) * 6,
    }
    values["object_twist_world"] = (0.0, 0.0, 0.0, 0.0, 0.0, -1.0)

    measurement = measure_order8_raw_contacts(**values)

    # At r=(0,1,0): robot omega x r=(-2,0,0), object=(-omega) gives (1,0,0).
    assert measurement.raw_contact_patches[
        0
    ].tangential_slip_speed_mps == pytest.approx(3.0)


def test_nonprivileged_relative_point_speed_includes_both_angular_velocities() -> (
    None
):
    speed = relative_point_speed_mps(
        body_reference_pose_world=IDENTITY_POSE,
        body_twist_world=(0.0, 0.0, 0.0, 0.0, 0.0, 2.0),
        object_reference_pose_world=IDENTITY_POSE,
        object_twist_world=(0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        point_world=(0.0, 1.0, 0.0),
    )

    assert speed == pytest.approx(3.0)


def test_nonprivileged_relative_point_speed_rejects_mismatched_state() -> None:
    with pytest.raises(Order8ContactMeasurementError, match="six finite"):
        relative_point_speed_mps(
            body_reference_pose_world=IDENTITY_POSE,
            body_twist_world=(0.0,) * 5,
            object_reference_pose_world=IDENTITY_POSE,
            object_twist_world=(0.0,) * 6,
            point_world=(0.0, 1.0, 0.0),
        )


def test_nonprivileged_normal_speed_excludes_tangential_motion() -> None:
    values = {
        "body_reference_pose_world": IDENTITY_POSE,
        "body_twist_world": (0.003, 0.004, 0.012, 0.0, 0.0, 0.0),
        "object_reference_pose_world": IDENTITY_POSE,
        "object_twist_world": (0.0,) * 6,
        "point_world": (0.0, 1.0, 0.0),
    }

    assert relative_point_speed_mps(**values) == pytest.approx(0.013)
    assert relative_point_normal_speed_mps(
        **values,
        surface_normal_world=(0.0, 0.0, 7.0),
    ) == pytest.approx(0.012)
    assert relative_point_normal_speed_mps(
        **values,
        surface_normal_world=(0.0, 1.0, 0.0),
    ) == pytest.approx(0.004)
    assert relative_point_normal_velocity_mps(
        **values,
        surface_normal_world=(0.0, -1.0, 0.0),
    ) == pytest.approx(-0.004)


def test_nonprivileged_normal_speed_rejects_zero_normal() -> None:
    with pytest.raises(Order8ContactMeasurementError, match="non-zero"):
        relative_point_normal_speed_mps(
            body_reference_pose_world=IDENTITY_POSE,
            body_twist_world=(0.0,) * 6,
            object_reference_pose_world=IDENTITY_POSE,
            object_twist_world=(0.0,) * 6,
            point_world=(0.0, 1.0, 0.0),
            surface_normal_world=(0.0, 0.0, 0.0),
        )


def test_unselected_sensor_patch_is_reported_as_unintended_global_link() -> None:
    values = _measurement_kwargs()
    values["contact_counts"] = [1, 0, 1]
    values["start_indices"] = [0, 1, 1]
    values["patch_forces_n"] = [5.0, 3.0, 0.0, 0.0]
    values["patch_points_world"] = [
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    values["patch_normals_world"] = [(0.0, 0.0, 1.0)] * 4
    values["patch_separations_m"] = [-0.001, -0.001, 1.0, 1.0]

    measurement = measure_order8_raw_contacts(**values)

    assert measurement.raw_contact_valid is True
    assert measurement.selected_raw_contact_count == 1
    assert measurement.unintended_contact is True
    assert measurement.unintended_raw_contact_count == 1
    assert measurement.unintended_contact_link_ids == ("module_0::fc",)
    assert measurement.unintended_force_magnitude_sum_n == 3.0


def test_patch_identity_and_output_order_use_global_links_deterministically() -> None:
    values = _measurement_kwargs()
    first = measure_order8_raw_contacts(**values)
    second = measure_order8_raw_contacts(**values)

    assert first == second
    assert first.sensor_body_to_global_link_id == (
        ("sensor_a", "module_1::yaw_dock_mech2"),
        ("sensor_b", "module_2::pitch_dock_mech1"),
        ("sensor_other", "module_0::fc"),
    )
    assert len({patch.patch_id for patch in first.raw_contact_patches}) == 2


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("contact_counts", [1, 1], "contact_pair_layout_length_mismatch"),
        ("contact_counts", [1, -1, 0], "contact_count_not_nonnegative_integer"),
        ("start_indices", [0, 4, 2], "active_patch_range_exceeds_capacity"),
        ("start_indices", [0, 0, 2], "active_patch_ranges_overlap"),
        ("patch_forces_n", [5.0], "raw_patch_buffer_length_mismatch"),
        (
            "selected_global_link_ids",
            ["module_1::yaw_dock_mech2", "missing"],
            "selected_global_link_id_not_in_sensor_view",
        ),
    ],
)
def test_invalid_layout_and_ranges_fail_closed(
    field: str,
    value: object,
    failure: str,
) -> None:
    values = _measurement_kwargs()
    values[field] = value

    measurement = measure_order8_raw_contacts(**values)

    assert measurement.raw_contact_valid is False
    assert measurement.raw_contact_patches == []
    assert failure in measurement.failure_reasons


def test_active_patch_range_beyond_sparse_buffers_fails_closed() -> None:
    values = _measurement_kwargs()
    values["contact_counts"] = [1, 1, 0]
    values["patch_forces_n"] = [5.0]
    values["patch_points_world"] = [(0.0, 1.0, 0.0)]
    values["patch_normals_world"] = [(0.0, 0.0, 1.0)]
    values["patch_separations_m"] = [-0.001]

    measurement = measure_order8_raw_contacts(**values)

    assert measurement.raw_contact_valid is False
    assert measurement.raw_contact_out_of_range is True
    assert measurement.raw_contact_patches == []
    assert "active_patch_range_exceeds_buffer" in measurement.failure_reasons


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("patch_forces_n", [math.nan, -7.0, 0.0, 0.0]),
        (
            "patch_points_world",
            [(math.inf, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0,) * 3, (0.0,) * 3],
        ),
        ("object_twist_world", (math.nan, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_nonfinite_active_contact_or_com_state_fails_closed(
    field: str,
    value: object,
) -> None:
    values = _measurement_kwargs()
    values[field] = value

    measurement = measure_order8_raw_contacts(**values)

    assert measurement.raw_contact_valid is False
    assert measurement.raw_contact_nonfinite is True
    assert measurement.raw_contact_patches == []


def test_zero_normal_and_exact_capacity_saturation_fail_closed() -> None:
    zero_normal = _measurement_kwargs()
    zero_normal["patch_normals_world"] = [(0.0, 0.0, 0.0)] * 4
    saturated = _measurement_kwargs()
    saturated["contact_counts"] = [2, 1, 1]
    saturated["start_indices"] = [0, 2, 3]

    invalid_normal = measure_order8_raw_contacts(**zero_normal)
    full_buffer = measure_order8_raw_contacts(**saturated)

    assert invalid_normal.raw_contact_valid is False
    assert invalid_normal.raw_contact_layout_invalid is True
    assert "active_patch_normal_zero" in invalid_normal.failure_reasons
    assert full_buffer.raw_contact_valid is False
    assert full_buffer.raw_contact_saturated is True
    assert full_buffer.raw_contact_count == 4
    assert full_buffer.raw_contact_patches == []


def test_geometry_helpers_cover_oriented_cuboid_transport_and_aabb_clearance() -> None:
    half_turn_about_y = (
        0.0,
        0.0,
        0.30,
        0.0,
        math.sin(math.pi / 4.0),
        0.0,
        math.cos(math.pi / 4.0),
    )
    bounds = oriented_cuboid_world_aabb(
        pose_world=half_turn_about_y,
        size_m=(0.40, 0.20, 0.10),
    )
    assert bounds.minimum_world[2] == pytest.approx(0.10)
    assert object_bottom_clearance_m(
        object_pose_world=half_turn_about_y,
        object_size_m=(0.40, 0.20, 0.10),
    ) == pytest.approx(0.10)
    assert (
        object_transport_distance_m(
            IDENTITY_POSE,
            (3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        == 5.0
    )

    gripper = Order8AABB(
        minimum_world=(0.15, -0.05, 0.25),
        maximum_world=(0.25, 0.05, 0.35),
    )
    assert aabb_clearance_m(gripper, bounds) == pytest.approx(0.10)
    assert gripper_object_clearance_m(
        gripper_aabbs_world=[gripper],
        object_pose_world=half_turn_about_y,
        object_size_m=(0.40, 0.20, 0.10),
    ) == pytest.approx(0.10)


def test_geometry_helpers_reject_invalid_bounds() -> None:
    invalid = Order8AABB(
        minimum_world=(1.0, 0.0, 0.0),
        maximum_world=(0.0, 1.0, 1.0),
    )
    valid = Order8AABB(
        minimum_world=(0.0, 0.0, 0.0),
        maximum_world=(1.0, 1.0, 1.0),
    )

    with pytest.raises(Order8ContactMeasurementError, match="minimum"):
        aabb_clearance_m(invalid, valid)
    with pytest.raises(Order8ContactMeasurementError, match="at least one"):
        gripper_object_clearance_m(
            gripper_aabbs_world=[],
            object_pose_world=IDENTITY_POSE,
            object_size_m=(1.0, 1.0, 1.0),
        )
