from __future__ import annotations

from amsrr.geometry.pose_math import (
    Transform3D,
    compose_transform,
    inverse_transform,
    pose_from_transform,
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
