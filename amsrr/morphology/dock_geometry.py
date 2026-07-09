from __future__ import annotations

from dataclasses import replace

from amsrr.geometry.pose_math import compose_pose, dock_module_relative_pose, inverse_pose
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.morphology import DockEdge, ModuleNode, PortNode


IDENTITY_POSE: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def relative_pose_for_dock_ports(src_port: PortNode, dst_port: PortNode) -> Pose7D:
    return dock_module_relative_pose(src_port.local_pose, dst_port.local_pose)


def modules_with_dock_aligned_poses(
    modules: list[ModuleNode],
    dock_edges: list[DockEdge],
    *,
    base_module_id: int = 0,
) -> list[ModuleNode]:
    if not dock_edges:
        return modules
    poses = module_poses_from_dock_edges(
        [module.module_id for module in modules],
        dock_edges,
        base_module_id=base_module_id,
    )
    return [
        replace(module, pose_in_design_frame=poses.get(module.module_id, module.pose_in_design_frame))
        for module in modules
    ]


def module_poses_from_dock_edges(
    module_ids: list[int],
    dock_edges: list[DockEdge],
    *,
    base_module_id: int = 0,
) -> dict[int, Pose7D]:
    if base_module_id not in set(module_ids):
        raise SchemaValidationError("base_module_id is missing from module ids")
    poses: dict[int, Pose7D] = {base_module_id: IDENTITY_POSE}
    remaining = list(dock_edges)
    progress = True
    while remaining and progress:
        progress = False
        next_remaining: list[DockEdge] = []
        for edge in remaining:
            if edge.src_module_id in poses and edge.dst_module_id not in poses:
                poses[edge.dst_module_id] = compose_pose(poses[edge.src_module_id], edge.relative_pose_src_to_dst)
                progress = True
            elif edge.dst_module_id in poses and edge.src_module_id not in poses:
                poses[edge.src_module_id] = compose_pose(poses[edge.dst_module_id], inverse_pose(edge.relative_pose_src_to_dst))
                progress = True
            elif edge.src_module_id in poses and edge.dst_module_id in poses:
                progress = True
            else:
                next_remaining.append(edge)
        remaining = next_remaining
    if remaining:
        missing = sorted({edge.src_module_id for edge in remaining} | {edge.dst_module_id for edge in remaining})
        raise SchemaValidationError(f"Cannot resolve dock-aligned module poses for modules {missing}")
    return poses
