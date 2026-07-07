from __future__ import annotations

import math

from amsrr.schemas.common import Vector3


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vector3, factor: float) -> Vector3:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vector3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector3, *, default: Vector3 = (0.0, 0.0, 1.0)) -> Vector3:
    length = norm(a)
    if length <= 1.0e-12:
        return default
    return (a[0] / length, a[1] / length, a[2] / length)


def distance(a: Vector3, b: Vector3) -> float:
    return norm(sub(a, b))


def angle_between(a: Vector3, b: Vector3) -> float:
    an = normalize(a)
    bn = normalize(b)
    value = max(-1.0, min(1.0, dot(an, bn)))
    return math.acos(value)


def triangle_area(a: Vector3, b: Vector3, c: Vector3) -> float:
    return 0.5 * norm(cross(sub(b, a), sub(c, a)))


def triangle_normal(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    return normalize(cross(sub(b, a), sub(c, a)))


def triangle_centroid(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    return ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0)


def tetra_signed_volume(a: Vector3, b: Vector3, c: Vector3) -> float:
    return dot(a, cross(b, c)) / 6.0


def bbox_from_points(points: list[Vector3]) -> tuple[Vector3, Vector3, Vector3]:
    if not points:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    mins = (
        min(point[0] for point in points),
        min(point[1] for point in points),
        min(point[2] for point in points),
    )
    maxs = (
        max(point[0] for point in points),
        max(point[1] for point in points),
        max(point[2] for point in points),
    )
    return mins, maxs, (maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2])


def center_from_bbox(mins: Vector3, maxs: Vector3) -> Vector3:
    return ((mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5)


def orthonormal_basis(normal: Vector3) -> tuple[Vector3, Vector3]:
    n = normalize(normal)
    helper = (0.0, 0.0, 1.0)
    if abs(dot(n, helper)) > 0.9:
        helper = (0.0, 1.0, 0.0)
    tangent_u = normalize(cross(helper, n), default=(1.0, 0.0, 0.0))
    tangent_v = normalize(cross(n, tangent_u), default=(0.0, 1.0, 0.0))
    return tangent_u, tangent_v


def dominant_normal_cluster(normal: Vector3) -> str:
    n = normalize(normal)
    values = [abs(n[0]), abs(n[1]), abs(n[2])]
    axis = values.index(max(values))
    sign = "pos" if n[axis] >= 0.0 else "neg"
    axis_name = ("x", "y", "z")[axis]
    return f"{sign}_{axis_name}"


def principal_axes_identity_flat() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

