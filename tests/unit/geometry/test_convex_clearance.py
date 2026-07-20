import math

import pytest

from amsrr.geometry.convex_clearance import (
    OrientedBox,
    capsule_oriented_box_clearance,
    circumscribed_cylinder_polytope,
    oriented_box_clearance,
    oriented_box_polytope_clearance,
    sphere_oriented_box_clearance,
)


def test_axis_aligned_box_clearance_is_exact() -> None:
    left = OrientedBox.axis_aligned((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    right = OrientedBox.axis_aligned((4.0, 5.0, 1.0), (1.0, 1.0, 1.0))
    assert oriented_box_clearance(left, right) == pytest.approx(math.sqrt(13.0))


def test_overlapping_and_touching_boxes_have_zero_clearance() -> None:
    left = OrientedBox.axis_aligned((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    touching = OrientedBox.axis_aligned((2.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlapping = OrientedBox.axis_aligned((1.5, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert oriented_box_clearance(left, touching) == 0.0
    assert oriented_box_clearance(left, overlapping) == 0.0


def test_rotated_box_near_finite_support_edge_avoids_world_aabb_false_zero() -> None:
    angle = math.pi / 4.0
    rotated = OrientedBox.from_pose_and_local_bounds(
        (2.4, 0.0, 0.4, 0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
        (-0.5, -0.1, -0.1),
        (0.5, 0.1, 0.1),
    )
    support = OrientedBox.axis_aligned((0.0, 0.0, 0.0), (2.0, 2.0, 0.2))
    assert oriented_box_clearance(rotated, support) == pytest.approx(0.1)


def test_pose_local_center_offset_is_rotated() -> None:
    angle = math.pi / 2.0
    box = OrientedBox.from_pose_and_local_bounds(
        (1.0, 2.0, 3.0, 0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
        (1.0, -1.0, -1.0),
        (3.0, 1.0, 1.0),
    )
    assert box.center == pytest.approx((1.0, 4.0, 3.0))
    assert box.half_extents == pytest.approx((1.0, 1.0, 1.0))


def test_sphere_and_capsule_clearance_are_shape_exact() -> None:
    box = OrientedBox.axis_aligned((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert sphere_oriented_box_clearance((3.0, 0.0, 0.0), 0.5, box) == pytest.approx(1.5)
    assert capsule_oriented_box_clearance(
        (3.0, 0.0, -1.0),
        (3.0, 0.0, 1.0),
        0.5,
        box,
    ) == pytest.approx(1.5)
    assert capsule_oriented_box_clearance(
        (0.0, 0.0, -2.0),
        (0.0, 0.0, 2.0),
        0.5,
        box,
    ) == 0.0


def test_circumscribed_cylinder_clearance_is_tight_and_conservative() -> None:
    box = OrientedBox.axis_aligned((3.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    cylinder = circumscribed_cylinder_polytope(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        radius=1.0,
        height=2.0,
        side_count=64,
    )
    clearance = oriented_box_polytope_clearance(box, cylinder)
    assert clearance <= 1.5
    assert clearance == pytest.approx(1.5, abs=0.002)


def test_box_polytope_overlap_detects_edge_face_crossing() -> None:
    box = OrientedBox.axis_aligned((0.95, 0.0, 0.0), (0.2, 0.2, 2.0))
    cylinder = circumscribed_cylinder_polytope(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        radius=1.0,
        height=1.0,
    )
    assert oriented_box_polytope_clearance(box, cylinder) == 0.0
