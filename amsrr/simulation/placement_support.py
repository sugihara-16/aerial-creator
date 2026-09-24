"""Privileged placement outcome evidence; isolated from deployable control.

The existing object sensor separates robot contacts from total contacts. A
non-robot upward reaction only counts when the object intersects the finite top
face of an allowed support and satisfies its authored object-pose goal. This
does not declare success: release, retreat and settle must still execute.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from amsrr.geometry.convex_clearance import OrientedBox, oriented_box_clearance
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES, Order9ObjectTaskPhase

PLACEMENT_SUPPORT_CONTRACT = "finite_allowed_support_reaction_place_outcome_v1"


class PlacementSupportEvidence:
    def __init__(self, task, *, force_threshold_n):
        self.force_threshold_n = float(force_threshold_n)
        scene = task["scene"]
        library = {g["geometry_id"]: g for g in scene["geometry_library"]}
        objects = {o["object_id"]: o for o in scene["objects"]}
        self.bindings = []
        self.trace = []
        if len(objects) != 1:
            return  # No ambiguous binding of one tracked pose to several objects.
        for goal in task["goals"]:
            if goal["goal_type"] != "object_pose":
                continue
            obj = objects[goal["target_entity_id"]]
            geometry = library[obj["geometry_id"]]
            # Unsupported geometry is conservative: no exemption. New shapes
            # can reuse the corresponding native convex collision primitive.
            if geometry["geometry_type"] != "box":
                continue
            half = np.asarray(geometry["primitive_params"]["size_m"]) * np.asarray(geometry["scale"]) / 2
            supports = []
            for surface in scene["environment"]["support_surfaces"]:
                if not surface["contact_allowed"] or "support" not in surface["allowed_contact_modes"]:
                    continue
                sg = library[surface["geometry_id"]]
                if sg["geometry_type"] != "box":
                    continue
                sh = np.asarray(sg["primitive_params"]["size_m"]) * np.asarray(sg["scale"]) / 2
                lower = -sh.copy(); lower[2] = sh[2]
                top = OrientedBox.from_pose_and_local_bounds(surface["pose_world"], lower, sh)
                supports.append((surface["surface_id"], top, np.asarray(top.axes[2])))
            self.bindings.append((goal, half, supports))

    def sample(self, *, phase_index, object_pose_world, scene_origins,
               net_force_world, robot_contact_force_world, time_s=None):
        """All tensors refer to the same post-physics sample, with env axis.

        net: (B,3); robot: (B,N,3), both forces ON the object in world frame.
        """
        batch = phase_index.shape[0]
        if net_force_world.shape != (batch, 3) or robot_contact_force_world.ndim != 3 or (
            robot_contact_force_world.shape[0] != batch or robot_contact_force_world.shape[-1] != 3
        ):
            raise ValueError("placement support force batch/shape mismatch")
        residual = net_force_world - robot_contact_force_world.sum(dim=1)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("nonfinite placement support force")
        result = torch.zeros_like(phase_index, dtype=torch.bool)
        place = ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.PLACE)
        for index in torch.nonzero(phase_index == place, as_tuple=False).flatten().tolist():
            pose = object_pose_world[index].detach().cpu().double().numpy().copy()
            pose[:3] -= scene_origins[index].detach().cpu().double().numpy()
            force = residual[index].detach().cpu().double().numpy()
            supported = False
            for goal, half, supports in self.bindings:
                target = np.asarray(goal["target_pose_world"])
                if np.linalg.norm(pose[:3] - target[:3]) > goal["tolerance_pos_m"]:
                    continue
                if (Rotation.from_quat(pose[3:]) * Rotation.from_quat(target[3:]).inv()).magnitude() > goal["tolerance_rot_rad"]:
                    continue
                box = OrientedBox.from_pose_and_local_bounds(pose, -half, half)
                for _, top, normal in supports:
                    # One micrometre only covers geometric floating-point
                    # roundoff; positive force alone never proves location.
                    if np.dot(force, normal) >= self.force_threshold_n and oriented_box_clearance(box, top) <= 1e-6:
                        supported = True
                        break
            result[index] = supported
            self.trace.append(dict(environment=index, pose=pose.tolist(),
                                   time_s=None if time_s is None else float(time_s[index]),
                                   net_force_world=net_force_world[index].detach().cpu().double().tolist(),
                                   robot_force_world=robot_contact_force_world[index].sum(dim=0).detach().cpu().double().tolist(),
                                   non_robot_force_world=force.tolist(), supported=supported))
        return result
