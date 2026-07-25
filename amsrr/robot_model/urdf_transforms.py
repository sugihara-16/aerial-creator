from __future__ import annotations

import math
from collections.abc import Mapping

from amsrr.geometry.pose_math import (
    Transform3D,
    compose_transform,
    inverse_transform,
    pose_from_transform,
    scale3,
    transform_from_pose,
    transform_from_xyz_rpy,
)
from amsrr.robot_model.urdf_loader import URDFModel
from amsrr.schemas.common import Pose7D, SchemaValidationError


def module_base_link_name(urdf_model: URDFModel) -> str:
    baselink_meta = urdf_model.metadata.get("baselink")
    if isinstance(baselink_meta, dict) and isinstance(baselink_meta.get("name"), str):
        name = str(baselink_meta["name"])
        if any(link.name == name for link in urdf_model.links):
            return name
    if any(link.name == "fc" for link in urdf_model.links):
        return "fc"
    if urdf_model.root_links:
        return urdf_model.root_links[0]
    raise SchemaValidationError("URDF model has no base/root link")


def link_poses_in_root_frame(urdf_model: URDFModel) -> dict[str, Pose7D]:
    return {link_id: pose_from_transform(transform) for link_id, transform in link_transforms_in_root_frame(urdf_model).items()}


def link_poses_at_joint_positions(
    urdf_model: URDFModel,
    joint_positions: Mapping[str, float],
    *,
    root_pose_world: Pose7D = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
) -> dict[str, Pose7D]:
    """Evaluate a tree URDF at the requested generalized coordinates.

    ``root_pose_world`` is the pose of the URDF's unique root link.  Missing
    joint positions use zero, matching URDF/Isaac reset semantics.  Extra
    names are rejected so diagnostics cannot silently apply coordinates from
    a different articulation.
    """

    if len(urdf_model.root_links) != 1:
        raise SchemaValidationError(
            "joint-position URDF FK requires exactly one root link"
        )
    joints_by_name = {joint.name: joint for joint in urdf_model.joints}
    unknown = sorted(set(joint_positions) - set(joints_by_name))
    if unknown:
        raise SchemaValidationError(
            f"joint-position URDF FK received unknown joints: {unknown}"
        )
    joints_by_parent = {}
    for joint in urdf_model.joints:
        joints_by_parent.setdefault(joint.parent_link, []).append(joint)

    root_link = urdf_model.root_links[0]
    transforms = {root_link: transform_from_pose(root_pose_world)}
    pending = [root_link]
    while pending:
        parent = pending.pop(0)
        parent_transform = transforms[parent]
        for joint in sorted(
            joints_by_parent.get(parent, []),
            key=lambda item: item.name,
        ):
            joint_frame = compose_transform(
                parent_transform,
                transform_from_xyz_rpy(
                    joint.origin_xyz,
                    joint.origin_rpy,
                ),
            )
            child_transform = compose_transform(
                joint_frame,
                _joint_motion_transform(
                    joint.joint_type,
                    joint.axis_xyz,
                    float(joint_positions.get(joint.name, 0.0)),
                ),
            )
            transforms[joint.child_link] = child_transform
            pending.append(joint.child_link)
    missing = sorted(
        {link.name for link in urdf_model.links} - set(transforms)
    )
    if missing:
        raise SchemaValidationError(
            f"URDF model has links missing from dynamic transform tree: {missing}"
        )
    return {
        link_id: pose_from_transform(transform)
        for link_id, transform in transforms.items()
    }


def link_poses_in_module_frame(urdf_model: URDFModel, *, base_link: str | None = None) -> dict[str, Pose7D]:
    transforms = link_transforms_in_root_frame(urdf_model)
    base = base_link or module_base_link_name(urdf_model)
    if base not in transforms:
        raise SchemaValidationError(f"URDF base link {base!r} is missing from transforms")
    root_from_base = inverse_transform(transforms[base])
    return {
        link_id: pose_from_transform(compose_transform(root_from_base, transform))
        for link_id, transform in transforms.items()
    }


def link_transforms_in_root_frame(urdf_model: URDFModel) -> dict[str, Transform3D]:
    if not urdf_model.root_links:
        raise SchemaValidationError("URDF model has no root links")
    joints_by_parent = {}
    for joint in urdf_model.joints:
        joints_by_parent.setdefault(joint.parent_link, []).append(joint)

    transforms = {
        root: Transform3D(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
        )
        for root in urdf_model.root_links
    }
    pending = list(urdf_model.root_links)
    while pending:
        parent = pending.pop(0)
        parent_transform = transforms[parent]
        for joint in sorted(joints_by_parent.get(parent, []), key=lambda item: item.name):
            child_transform = compose_transform(
                parent_transform,
                transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy),
            )
            transforms[joint.child_link] = child_transform
            pending.append(joint.child_link)
    missing = sorted({link.name for link in urdf_model.links} - set(transforms))
    if missing:
        raise SchemaValidationError(f"URDF model has links missing from transform tree: {missing}")
    return transforms


def _joint_motion_transform(
    joint_type: str,
    axis: tuple[float, float, float],
    position: float,
) -> Transform3D:
    if joint_type in {"revolute", "continuous"}:
        return Transform3D(
            rotation=_axis_angle_to_matrix(axis, position),
            translation=(0.0, 0.0, 0.0),
        )
    if joint_type == "prismatic":
        norm = math.sqrt(sum(float(value) ** 2 for value in axis))
        if norm <= 0.0:
            raise SchemaValidationError(
                "prismatic URDF joint axis must be non-zero"
            )
        return Transform3D(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=scale3(
                tuple(float(value) / norm for value in axis),  # type: ignore[arg-type]
                position,
            ),
        )
    return Transform3D(
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
    )


def _axis_angle_to_matrix(
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    norm = math.sqrt(sum(float(value) ** 2 for value in axis))
    if norm <= 0.0:
        raise SchemaValidationError(
            "revolute URDF joint axis must be non-zero"
        )
    x, y, z = (float(value) / norm for value in axis)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return (
        (
            c + x * x * one_c,
            x * y * one_c - z * s,
            x * z * one_c + y * s,
        ),
        (
            y * x * one_c + z * s,
            c + y * y * one_c,
            y * z * one_c - x * s,
        ),
        (
            z * x * one_c - y * s,
            z * y * one_c + x * s,
            c + z * z * one_c,
        ),
    )
