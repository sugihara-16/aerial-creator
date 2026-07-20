from __future__ import annotations

"""Small deterministic convex-clearance primitives used by hard safety gates.

The functions in this module deliberately avoid simulator APIs.  Isaac owns
the poses, while this module evaluates the geometry supplied by the copied
runtime.  In particular, an oriented box is never expanded into a world AABB:
that expansion can turn a harmless rotation near an obstacle edge into a false
zero-clearance result.
"""

import math
from dataclasses import dataclass
from typing import Sequence

from amsrr.geometry.pose_math import Matrix3, matvec, transform_from_pose
from amsrr.schemas.common import Pose7D, Vector3


_BOX_EDGE_VERTEX_INDICES = tuple(
    (index, index ^ (1 << axis))
    for index in range(8)
    for axis in range(3)
    if (index & (1 << axis)) == 0
)


@dataclass(frozen=True)
class OrientedBox:
    center: Vector3
    axes: Matrix3
    half_extents: Vector3

    def __post_init__(self) -> None:
        values = (*self.center, *self.half_extents, *(value for row in self.axes for value in row))
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("oriented-box values must be finite")
        if any(float(value) < 0.0 for value in self.half_extents):
            raise ValueError("oriented-box half extents must be non-negative")
        for axis in self.axes:
            if not math.isclose(_dot(axis, axis), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError("oriented-box axes must be unit length")
        if any(
            not math.isclose(_dot(self.axes[left], self.axes[right]), 0.0, rel_tol=0.0, abs_tol=1.0e-6)
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("oriented-box axes must be orthogonal")

    @classmethod
    def from_pose_and_local_bounds(
        cls,
        pose_world: Sequence[float],
        minimum_local: Sequence[float],
        maximum_local: Sequence[float],
    ) -> "OrientedBox":
        if len(pose_world) != 7 or len(minimum_local) != 3 or len(maximum_local) != 3:
            raise ValueError("oriented-box pose/bounds dimensions are invalid")
        minimum = tuple(float(value) for value in minimum_local)
        maximum = tuple(float(value) for value in maximum_local)
        if any(left > right for left, right in zip(minimum, maximum)):
            raise ValueError("oriented-box local bounds are inverted")
        local_center = tuple(
            0.5 * (minimum[axis] + maximum[axis]) for axis in range(3)
        )
        half_extents = tuple(
            0.5 * (maximum[axis] - minimum[axis]) for axis in range(3)
        )
        transform = transform_from_pose(tuple(float(value) for value in pose_world))
        rotated_center = matvec(transform.rotation, local_center)
        center = tuple(
            float(transform.translation[axis]) + float(rotated_center[axis])
            for axis in range(3)
        )
        # Matrix rows encode world-coordinate components.  The local basis
        # vectors expressed in world are therefore the matrix columns.
        axes = tuple(
            tuple(float(transform.rotation[row][column]) for row in range(3))
            for column in range(3)
        )
        return cls(center=center, axes=axes, half_extents=half_extents)  # type: ignore[arg-type]

    @classmethod
    def axis_aligned(
        cls,
        center: Sequence[float],
        half_extents: Sequence[float],
    ) -> "OrientedBox":
        if len(center) != 3 or len(half_extents) != 3:
            raise ValueError("axis-aligned box dimensions are invalid")
        return cls(
            center=tuple(float(value) for value in center),  # type: ignore[arg-type]
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            half_extents=tuple(float(value) for value in half_extents),  # type: ignore[arg-type]
        )

    def vertices(self) -> tuple[Vector3, ...]:
        result: list[Vector3] = []
        for index in range(8):
            point = list(self.center)
            for axis in range(3):
                sign = 1.0 if index & (1 << axis) else -1.0
                for component in range(3):
                    point[component] += (
                        sign
                        * float(self.half_extents[axis])
                        * float(self.axes[axis][component])
                    )
            result.append((float(point[0]), float(point[1]), float(point[2])))
        return tuple(result)

    def edges(self) -> tuple[tuple[Vector3, Vector3], ...]:
        vertices = self.vertices()
        return tuple(
            (vertices[left], vertices[right])
            for left, right in _BOX_EDGE_VERTEX_INDICES
        )

    def triangles(self) -> tuple[tuple[Vector3, Vector3, Vector3], ...]:
        vertices = self.vertices()
        # Two triangles per face.  Winding is irrelevant for distance and
        # segment-intersection queries.
        faces = (
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        )
        return tuple(
            triangle
            for a, b, c, d in faces
            for triangle in (
                (vertices[a], vertices[b], vertices[c]),
                (vertices[a], vertices[c], vertices[d]),
            )
        )

    def contains(self, point: Sequence[float], *, tolerance: float = 1.0e-12) -> bool:
        delta = _sub(point, self.center)
        return all(
            abs(_dot(delta, self.axes[axis]))
            <= float(self.half_extents[axis]) + float(tolerance)
            for axis in range(3)
        )


@dataclass(frozen=True)
class ConvexPolytope:
    """Closed convex polytope with explicit surface features and halfspaces."""

    vertices: tuple[Vector3, ...]
    triangles: tuple[tuple[int, int, int], ...]
    edges: tuple[tuple[int, int], ...]
    plane_normals: tuple[Vector3, ...]
    plane_offsets: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 4 or not self.triangles or not self.edges:
            raise ValueError("convex polytope requires vertices, faces, and edges")
        if len(self.plane_normals) != len(self.plane_offsets) or not self.plane_normals:
            raise ValueError("convex polytope halfspace layout is invalid")
        if any(
            index < 0 or index >= len(self.vertices)
            for item in (*self.triangles, *self.edges)
            for index in item
        ):
            raise ValueError("convex polytope feature index is out of range")
        values = (
            *(value for vertex in self.vertices for value in vertex),
            *(value for normal in self.plane_normals for value in normal),
            *self.plane_offsets,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("convex polytope values must be finite")
        if any(
            not math.isclose(_dot(normal, normal), 1.0, rel_tol=0.0, abs_tol=1.0e-6)
            for normal in self.plane_normals
        ):
            raise ValueError("convex polytope plane normals must be unit length")

    def contains(self, point: Sequence[float], *, tolerance: float = 1.0e-12) -> bool:
        return all(
            _dot(normal, point) <= float(offset) + float(tolerance)
            for normal, offset in zip(self.plane_normals, self.plane_offsets)
        )

    def triangle_points(self) -> tuple[tuple[Vector3, Vector3, Vector3], ...]:
        return tuple(
            (self.vertices[first], self.vertices[second], self.vertices[third])
            for first, second, third in self.triangles
        )

    def edge_points(self) -> tuple[tuple[Vector3, Vector3], ...]:
        return tuple(
            (self.vertices[first], self.vertices[second])
            for first, second in self.edges
        )


def oriented_box_clearance(left: OrientedBox, right: OrientedBox) -> float:
    """Return the Euclidean distance between two closed oriented boxes."""

    if _oriented_boxes_overlap(left, right):
        return 0.0
    minimum_squared = math.inf
    for point in left.vertices():
        minimum_squared = min(minimum_squared, _point_box_distance_squared(point, right))
    for point in right.vertices():
        minimum_squared = min(minimum_squared, _point_box_distance_squared(point, left))
    for left_start, left_end in left.edges():
        for right_start, right_end in right.edges():
            minimum_squared = min(
                minimum_squared,
                _segment_segment_distance_squared(
                    left_start,
                    left_end,
                    right_start,
                    right_end,
                ),
            )
    if not math.isfinite(minimum_squared):  # pragma: no cover - boxes always have features.
        raise RuntimeError("oriented-box clearance could not be evaluated")
    return math.sqrt(max(0.0, minimum_squared))


def point_oriented_box_clearance(point: Sequence[float], box: OrientedBox) -> float:
    if len(point) != 3:
        raise ValueError("point must contain three values")
    return math.sqrt(_point_box_distance_squared(tuple(float(value) for value in point), box))


def segment_oriented_box_clearance(
    start: Sequence[float],
    end: Sequence[float],
    box: OrientedBox,
) -> float:
    """Return the exact distance between a finite segment and an OBB."""

    first = tuple(float(value) for value in start)
    second = tuple(float(value) for value in end)
    if len(first) != 3 or len(second) != 3:
        raise ValueError("segment endpoints must contain three values")
    if _segment_intersects_box(first, second, box):
        return 0.0
    minimum_squared = min(
        _point_box_distance_squared(first, box),
        _point_box_distance_squared(second, box),
    )
    for edge_start, edge_end in box.edges():
        minimum_squared = min(
            minimum_squared,
            _segment_segment_distance_squared(first, second, edge_start, edge_end),
        )
    return math.sqrt(max(0.0, minimum_squared))


def sphere_oriented_box_clearance(
    center: Sequence[float],
    radius: float,
    box: OrientedBox,
) -> float:
    if not math.isfinite(float(radius)) or radius <= 0.0:
        raise ValueError("sphere radius must be positive")
    return max(0.0, point_oriented_box_clearance(center, box) - float(radius))


def capsule_oriented_box_clearance(
    segment_start: Sequence[float],
    segment_end: Sequence[float],
    radius: float,
    box: OrientedBox,
) -> float:
    if not math.isfinite(float(radius)) or radius <= 0.0:
        raise ValueError("capsule radius must be positive")
    return max(
        0.0,
        segment_oriented_box_clearance(segment_start, segment_end, box)
        - float(radius),
    )


def circumscribed_cylinder_polytope(
    pose_world: Sequence[float],
    *,
    radius: float,
    height: float,
    side_count: int = 64,
) -> ConvexPolytope:
    """Build a deterministic prism that safely contains a z-axis cylinder.

    The radial excess is bounded by ``1/cos(pi/N)-1`` (about 0.12 percent for
    the default 64 sides).  Thus the result is conservative for a hard
    clearance gate while avoiding the large corner inflation of a square OBB.
    """

    if not math.isfinite(float(radius)) or radius <= 0.0:
        raise ValueError("cylinder radius must be positive")
    if not math.isfinite(float(height)) or height <= 0.0:
        raise ValueError("cylinder height must be positive")
    if side_count < 8:
        raise ValueError("cylinder prism requires at least eight sides")
    transform = transform_from_pose(tuple(float(value) for value in pose_world))
    outer_radius = float(radius) / math.cos(math.pi / side_count)
    half_height = 0.5 * float(height)
    local_vertices = tuple(
        (
            outer_radius * math.cos(2.0 * math.pi * index / side_count),
            outer_radius * math.sin(2.0 * math.pi * index / side_count),
            z,
        )
        for z in (-half_height, half_height)
        for index in range(side_count)
    )
    vertices = tuple(
        _add(transform.translation, matvec(transform.rotation, vertex))
        for vertex in local_vertices
    )
    triangles: list[tuple[int, int, int]] = []
    edges: set[tuple[int, int]] = set()
    for index in range(1, side_count - 1):
        triangles.append((0, index + 1, index))
        triangles.append(
            (side_count, side_count + index, side_count + index + 1)
        )
    for index in range(side_count):
        next_index = (index + 1) % side_count
        lower = index
        lower_next = next_index
        upper = side_count + index
        upper_next = side_count + next_index
        triangles.extend(
            ((lower, lower_next, upper_next), (lower, upper_next, upper))
        )
        for pair in (
            (lower, lower_next),
            (upper, upper_next),
            (lower, upper),
        ):
            edges.add(tuple(sorted(pair)))
    local_normals = [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
    local_offsets = [half_height, half_height]
    for index in range(side_count):
        angle = 2.0 * math.pi * (index + 0.5) / side_count
        local_normals.append((math.cos(angle), math.sin(angle), 0.0))
        local_offsets.append(float(radius))
    normals = tuple(matvec(transform.rotation, normal) for normal in local_normals)
    offsets = tuple(
        float(local_offset) + _dot(normal, transform.translation)
        for normal, local_offset in zip(normals, local_offsets)
    )
    return ConvexPolytope(
        vertices=vertices,
        triangles=tuple(triangles),
        edges=tuple(sorted(edges)),
        plane_normals=normals,
        plane_offsets=offsets,
    )


def oriented_box_polytope_clearance(
    box: OrientedBox,
    polytope: ConvexPolytope,
) -> float:
    """Return feature-exact clearance between an OBB and convex polytope."""

    box_vertices = box.vertices()
    polytope_triangles = polytope.triangle_points()
    polytope_edges = polytope.edge_points()
    if any(box.contains(point) for point in polytope.vertices):
        return 0.0
    if any(polytope.contains(point) for point in box_vertices):
        return 0.0
    box_edges = box.edges()
    box_triangles = box.triangles()
    if any(
        _segment_intersects_triangle(start, end, triangle)
        for start, end in box_edges
        for triangle in polytope_triangles
    ) or any(
        _segment_intersects_triangle(start, end, triangle)
        for start, end in polytope_edges
        for triangle in box_triangles
    ):
        return 0.0
    minimum_squared = min(
        _point_box_distance_squared(point, box) for point in polytope.vertices
    )
    for point in box_vertices:
        for triangle in polytope_triangles:
            minimum_squared = min(
                minimum_squared,
                _point_triangle_distance_squared(point, triangle),
            )
    for poly_start, poly_end in polytope_edges:
        for box_start, box_end in box_edges:
            minimum_squared = min(
                minimum_squared,
                _segment_segment_distance_squared(
                    poly_start,
                    poly_end,
                    box_start,
                    box_end,
                ),
            )
    return math.sqrt(max(0.0, minimum_squared))


def _oriented_boxes_overlap(left: OrientedBox, right: OrientedBox) -> bool:
    center_delta = _sub(right.center, left.center)
    axes = [*left.axes, *right.axes]
    axes.extend(
        _cross(left_axis, right_axis)
        for left_axis in left.axes
        for right_axis in right.axes
    )
    for raw_axis in axes:
        norm_squared = _dot(raw_axis, raw_axis)
        if norm_squared <= 1.0e-20:
            continue
        inv_norm = 1.0 / math.sqrt(norm_squared)
        axis = tuple(float(value) * inv_norm for value in raw_axis)
        center_separation = abs(_dot(center_delta, axis))
        left_radius = sum(
            float(left.half_extents[index]) * abs(_dot(left.axes[index], axis))
            for index in range(3)
        )
        right_radius = sum(
            float(right.half_extents[index]) * abs(_dot(right.axes[index], axis))
            for index in range(3)
        )
        if center_separation > left_radius + right_radius + 1.0e-12:
            return False
    return True


def _point_box_distance_squared(point: Vector3, box: OrientedBox) -> float:
    delta = _sub(point, box.center)
    squared = 0.0
    for axis in range(3):
        coordinate = _dot(delta, box.axes[axis])
        excess = max(abs(coordinate) - float(box.half_extents[axis]), 0.0)
        squared += excess * excess
    return squared


def _segment_intersects_box(start: Vector3, end: Vector3, box: OrientedBox) -> bool:
    local_start_delta = _sub(start, box.center)
    local_end_delta = _sub(end, box.center)
    local_start = tuple(_dot(local_start_delta, axis) for axis in box.axes)
    local_end = tuple(_dot(local_end_delta, axis) for axis in box.axes)
    direction = _sub(local_end, local_start)
    lower_parameter = 0.0
    upper_parameter = 1.0
    for axis in range(3):
        lower = -float(box.half_extents[axis])
        upper = float(box.half_extents[axis])
        if abs(direction[axis]) <= 1.0e-20:
            if local_start[axis] < lower or local_start[axis] > upper:
                return False
            continue
        first = (lower - local_start[axis]) / direction[axis]
        second = (upper - local_start[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        lower_parameter = max(lower_parameter, first)
        upper_parameter = min(upper_parameter, second)
        if lower_parameter > upper_parameter:
            return False
    return True


def _segment_intersects_triangle(
    start: Vector3,
    end: Vector3,
    triangle: tuple[Vector3, Vector3, Vector3],
) -> bool:
    direction = _sub(end, start)
    first_edge = _sub(triangle[1], triangle[0])
    second_edge = _sub(triangle[2], triangle[0])
    cross_direction = _cross(direction, second_edge)
    determinant = _dot(first_edge, cross_direction)
    if abs(determinant) <= 1.0e-14:
        return False
    inverse = 1.0 / determinant
    offset = _sub(start, triangle[0])
    first_barycentric = _dot(offset, cross_direction) * inverse
    if first_barycentric < -1.0e-12 or first_barycentric > 1.0 + 1.0e-12:
        return False
    cross_offset = _cross(offset, first_edge)
    second_barycentric = _dot(direction, cross_offset) * inverse
    if (
        second_barycentric < -1.0e-12
        or first_barycentric + second_barycentric > 1.0 + 1.0e-12
    ):
        return False
    segment_parameter = _dot(second_edge, cross_offset) * inverse
    return -1.0e-12 <= segment_parameter <= 1.0 + 1.0e-12


def _point_triangle_distance_squared(
    point: Vector3,
    triangle: tuple[Vector3, Vector3, Vector3],
) -> float:
    """Closest point-region test from *Real-Time Collision Detection*."""

    first, second, third = triangle
    first_edge = _sub(second, first)
    second_edge = _sub(third, first)
    first_to_point = _sub(point, first)
    d1 = _dot(first_edge, first_to_point)
    d2 = _dot(second_edge, first_to_point)
    if d1 <= 0.0 and d2 <= 0.0:
        return _dot(first_to_point, first_to_point)
    second_to_point = _sub(point, second)
    d3 = _dot(first_edge, second_to_point)
    d4 = _dot(second_edge, second_to_point)
    if d3 >= 0.0 and d4 <= d3:
        return _dot(second_to_point, second_to_point)
    first_edge_region = d1 * d4 - d3 * d2
    if first_edge_region <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        parameter = d1 / (d1 - d3)
        closest = _add(first, _scale(first_edge, parameter))
        delta = _sub(point, closest)
        return _dot(delta, delta)
    third_to_point = _sub(point, third)
    d5 = _dot(first_edge, third_to_point)
    d6 = _dot(second_edge, third_to_point)
    if d6 >= 0.0 and d5 <= d6:
        return _dot(third_to_point, third_to_point)
    second_edge_region = d5 * d2 - d1 * d6
    if second_edge_region <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        parameter = d2 / (d2 - d6)
        closest = _add(first, _scale(second_edge, parameter))
        delta = _sub(point, closest)
        return _dot(delta, delta)
    opposite_edge_region = d3 * d6 - d5 * d4
    if (
        opposite_edge_region <= 0.0
        and d4 - d3 >= 0.0
        and d5 - d6 >= 0.0
    ):
        parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        closest = _add(second, _scale(_sub(third, second), parameter))
        delta = _sub(point, closest)
        return _dot(delta, delta)
    denominator = 1.0 / (
        first_edge_region + second_edge_region + opposite_edge_region
    )
    second_weight = second_edge_region * denominator
    third_weight = first_edge_region * denominator
    closest = _add(
        first,
        _add(
            _scale(first_edge, second_weight),
            _scale(second_edge, third_weight),
        ),
    )
    delta = _sub(point, closest)
    return _dot(delta, delta)


def _segment_segment_distance_squared(
    first_start: Vector3,
    first_end: Vector3,
    second_start: Vector3,
    second_end: Vector3,
) -> float:
    """Closest squared distance for two finite 3-D segments.

    This is the clamped two-parameter solution from *Real-Time Collision
    Detection*.  Degenerate segments are handled explicitly because very thin
    authored collision bounds are legal.
    """

    first_direction = _sub(first_end, first_start)
    second_direction = _sub(second_end, second_start)
    offset = _sub(first_start, second_start)
    first_length_squared = _dot(first_direction, first_direction)
    second_length_squared = _dot(second_direction, second_direction)
    first_dot_offset = _dot(first_direction, offset)
    if first_length_squared <= 1.0e-20 and second_length_squared <= 1.0e-20:
        return _dot(offset, offset)
    if first_length_squared <= 1.0e-20:
        first_parameter = 0.0
        second_parameter = _clamp(
            _dot(second_direction, offset) / second_length_squared,
            0.0,
            1.0,
        )
    else:
        if second_length_squared <= 1.0e-20:
            second_parameter = 0.0
            first_parameter = _clamp(-first_dot_offset / first_length_squared, 0.0, 1.0)
        else:
            direction_dot = _dot(first_direction, second_direction)
            second_dot_offset = _dot(second_direction, offset)
            denominator = (
                first_length_squared * second_length_squared
                - direction_dot * direction_dot
            )
            first_parameter = (
                _clamp(
                    (direction_dot * second_dot_offset - first_dot_offset * second_length_squared)
                    / denominator,
                    0.0,
                    1.0,
                )
                if denominator > 1.0e-20
                else 0.0
            )
            second_parameter = (
                direction_dot * first_parameter + second_dot_offset
            ) / second_length_squared
            if second_parameter < 0.0:
                second_parameter = 0.0
                first_parameter = _clamp(-first_dot_offset / first_length_squared, 0.0, 1.0)
            elif second_parameter > 1.0:
                second_parameter = 1.0
                first_parameter = _clamp(
                    (direction_dot - first_dot_offset) / first_length_squared,
                    0.0,
                    1.0,
                )
    closest_offset = tuple(
        float(offset[axis])
        + first_parameter * float(first_direction[axis])
        - second_parameter * float(second_direction[axis])
        for axis in range(3)
    )
    return _dot(closest_offset, closest_offset)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _sub(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))  # type: ignore[return-value]


def _add(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(float(left[index]) + float(right[index]) for index in range(3))  # type: ignore[return-value]


def _scale(value: Sequence[float], scalar: float) -> Vector3:
    return tuple(float(value[index]) * float(scalar) for index in range(3))  # type: ignore[return-value]


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


__all__ = [
    "ConvexPolytope",
    "OrientedBox",
    "capsule_oriented_box_clearance",
    "circumscribed_cylinder_polytope",
    "oriented_box_clearance",
    "oriented_box_polytope_clearance",
    "point_oriented_box_clearance",
    "segment_oriented_box_clearance",
    "sphere_oriented_box_clearance",
]
