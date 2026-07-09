from __future__ import annotations

import math
from dataclasses import dataclass

from amsrr.schemas.common import Pose7D, Vector3


Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

FACE_TO_FACE_DOCK_RELATION: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class Transform3D:
    rotation: Matrix3
    translation: Vector3


def compose_pose(left: Pose7D, right: Pose7D) -> Pose7D:
    return pose_from_transform(compose_transform(transform_from_pose(left), transform_from_pose(right)))


def inverse_pose(pose: Pose7D) -> Pose7D:
    return pose_from_transform(inverse_transform(transform_from_pose(pose)))


def dock_module_relative_pose(
    src_port_pose: Pose7D,
    dst_port_pose: Pose7D,
    *,
    port_relation: Pose7D = FACE_TO_FACE_DOCK_RELATION,
) -> Pose7D:
    """Return source-module to destination-module pose satisfying the dock port relation."""

    src_port = transform_from_pose(src_port_pose)
    relation = transform_from_pose(port_relation)
    dst_port = transform_from_pose(dst_port_pose)
    return pose_from_transform(compose_transform(compose_transform(src_port, relation), inverse_transform(dst_port)))


def transform_from_pose(pose: Pose7D) -> Transform3D:
    return Transform3D(
        rotation=quat_to_matrix((float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))),
        translation=(float(pose[0]), float(pose[1]), float(pose[2])),
    )


def transform_from_xyz_rpy(xyz: Vector3, rpy: Vector3) -> Transform3D:
    return Transform3D(rotation=rpy_to_matrix(rpy), translation=xyz)


def pose_from_transform(transform: Transform3D) -> Pose7D:
    qx, qy, qz, qw = quat_from_matrix(transform.rotation)
    return (
        float(transform.translation[0]),
        float(transform.translation[1]),
        float(transform.translation[2]),
        qx,
        qy,
        qz,
        qw,
    )


def pose_to_xyz_rpy(pose: Pose7D) -> tuple[Vector3, Vector3]:
    transform = transform_from_pose(pose)
    return transform.translation, rpy_from_matrix(transform.rotation)


def compose_transform(left: Transform3D, right: Transform3D) -> Transform3D:
    return Transform3D(
        rotation=matmul(left.rotation, right.rotation),
        translation=add3(left.translation, matvec(left.rotation, right.translation)),
    )


def inverse_transform(transform: Transform3D) -> Transform3D:
    rotation_inv = transpose(transform.rotation)
    return Transform3D(
        rotation=rotation_inv,
        translation=matvec(rotation_inv, scale3(transform.translation, -1.0)),
    )


def quat_to_matrix(quat_xyzw: tuple[float, float, float, float]) -> Matrix3:
    x, y, z, w = normalize_quat(quat_xyzw)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def quat_from_matrix(matrix: Matrix3) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    return normalize_quat((qx, qy, qz, qw))


def normalize_quat(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:
        return -x, -y, -z, -w
    return x, y, z, w


def rpy_to_matrix(rpy: Vector3) -> Matrix3:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def rpy_from_matrix(matrix: Matrix3) -> Vector3:
    pitch = math.atan2(-matrix[2][0], math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0]))
    if abs(math.cos(pitch)) <= 1.0e-12:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    else:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    return (roll, pitch, yaw)


def matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][idx] * right[idx][col] for idx in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def add3(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def scale3(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)
