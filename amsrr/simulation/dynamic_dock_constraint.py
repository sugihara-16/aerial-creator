from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    compose_pose,
    inverse_pose,
    pose_from_transform,
    transform_from_xyz_rpy,
)
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph, PortNode
from amsrr.schemas.physical_model import DockPortSpec, PhysicalModel


DYNAMIC_DOCK_CONSTRAINT_VERSION = "external_connect_frame_fixed_joint_v1"


@dataclass
class DynamicDockConstraintSpec(SchemaBase):
    constraint_id: str
    prim_path: str
    edge_id: int
    leader_module_id: int
    follower_module_id: int
    leader_port_id: int
    follower_port_id: int
    leader_body_path: str
    follower_body_path: str
    leader_body_local_connect_pose: Pose7D
    follower_body_local_constraint_pose: Pose7D
    version: str = DYNAMIC_DOCK_CONSTRAINT_VERSION

    def validate(self) -> None:
        for path, value in (
            ("constraint_id", self.constraint_id),
            ("prim_path", self.prim_path),
            ("leader_body_path", self.leader_body_path),
            ("follower_body_path", self.follower_body_path),
        ):
            require_non_empty(value, f"DynamicDockConstraintSpec.{path}")
        if not self.prim_path.startswith("/"):
            raise SchemaValidationError("DynamicDockConstraintSpec.prim_path must be absolute")
        if min(
            self.edge_id,
            self.leader_module_id,
            self.follower_module_id,
            self.leader_port_id,
            self.follower_port_id,
        ) < 0:
            raise SchemaValidationError("DynamicDockConstraintSpec ids must be non-negative")
        require_len(
            self.leader_body_local_connect_pose,
            7,
            "DynamicDockConstraintSpec.leader_body_local_connect_pose",
        )
        require_len(
            self.follower_body_local_constraint_pose,
            7,
            "DynamicDockConstraintSpec.follower_body_local_constraint_pose",
        )
        if self.version != DYNAMIC_DOCK_CONSTRAINT_VERSION:
            raise SchemaValidationError("DynamicDockConstraintSpec.version mismatch")


@dataclass
class DynamicDockConstraintRecord(SchemaBase):
    spec: DynamicDockConstraintSpec
    preauthored_disabled: bool
    enabled: bool
    identity_verified: bool
    collision_pair_filtered: bool
    created_time_s: float | None = None
    enabled_time_s: float | None = None
    disabled_time_s: float | None = None
    removed_time_s: float | None = None
    verification_failures: list[str] = field(default_factory=list)

    def validate(self) -> None:
        for name in (
            "created_time_s",
            "enabled_time_s",
            "disabled_time_s",
            "removed_time_s",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise SchemaValidationError(
                    f"DynamicDockConstraintRecord.{name} must be finite and non-negative"
                )
        if self.identity_verified and self.verification_failures:
            raise SchemaValidationError(
                "DynamicDockConstraintRecord cannot verify identity with failures"
            )


@dataclass
class DynamicDockConstraintResidual(SchemaBase):
    position_error_m: float
    attitude_error_rad: float
    relative_linear_speed_mps: float
    relative_angular_speed_radps: float

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise SchemaValidationError(
                    f"DynamicDockConstraintResidual.{name} must be finite and non-negative"
                )


def build_dynamic_dock_constraint_spec(
    morphology_graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    edge_id: int,
    leader_module_id: int,
    follower_module_id: int,
    leader_body_path: str,
    follower_body_path: str,
    constraint_root_path: str = "/World/AssemblyConstraints",
) -> DynamicDockConstraintSpec:
    edge = next((item for item in morphology_graph.dock_edges if item.edge_id == edge_id), None)
    if edge is None:
        raise SchemaValidationError(f"MorphologyGraph has no DockEdge {edge_id}")
    if {leader_module_id, follower_module_id} != {
        edge.src_module_id,
        edge.dst_module_id,
    }:
        raise SchemaValidationError("constraint leader/follower must be DockEdge endpoints")
    port_by_id = {port.port_global_id: port for port in morphology_graph.ports}
    endpoint_ports = {
        edge.src_module_id: port_by_id.get(edge.src_port_id),
        edge.dst_module_id: port_by_id.get(edge.dst_port_id),
    }
    leader_port = endpoint_ports.get(leader_module_id)
    follower_port = endpoint_ports.get(follower_module_id)
    if leader_port is None or follower_port is None:
        raise SchemaValidationError("DockEdge endpoint port is missing")
    leader_local = connect_frame_pose_in_parent_body(leader_port, physical_model)
    follower_connect_local = connect_frame_pose_in_parent_body(follower_port, physical_model)
    # FixedJoint makes its two local frames equal.  The raw connector frames
    # are face-to-face, so the follower joint frame includes Rz(pi).
    follower_constraint_local = compose_pose(
        follower_connect_local,
        FACE_TO_FACE_DOCK_RELATION,
    )
    constraint_id = f"dock_edge_{edge_id}_{leader_module_id}_{follower_module_id}"
    return DynamicDockConstraintSpec(
        constraint_id=constraint_id,
        prim_path=f"{constraint_root_path.rstrip('/')}/{constraint_id}",
        edge_id=edge_id,
        leader_module_id=leader_module_id,
        follower_module_id=follower_module_id,
        leader_port_id=leader_port.port_global_id,
        follower_port_id=follower_port.port_global_id,
        leader_body_path=leader_body_path,
        follower_body_path=follower_body_path,
        leader_body_local_connect_pose=leader_local,
        follower_body_local_constraint_pose=follower_constraint_local,
    )


def connect_frame_pose_in_parent_body(
    graph_port: PortNode,
    physical_model: PhysicalModel,
) -> Pose7D:
    port = _physical_port(graph_port, physical_model)
    joint = next(
        (candidate for candidate in physical_model.joints if candidate.joint_id == port.port_id),
        None,
    )
    if joint is None:
        raise SchemaValidationError(
            f"PhysicalModel has no connect-point JointModel {port.port_id!r}"
        )
    if joint.parent_link != port.parent_link:
        raise SchemaValidationError(
            f"Connect-point joint {joint.joint_id!r} parent does not match DockPortSpec"
        )
    return pose_from_transform(
        transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
    )


def constraint_residual(
    leader_connect_pose_world: Pose7D,
    follower_connect_pose_world: Pose7D,
    *,
    leader_connect_twist_world: list[float] | tuple[float, ...],
    follower_connect_twist_world: list[float] | tuple[float, ...],
) -> DynamicDockConstraintResidual:
    require_len(leader_connect_twist_world, 6, "leader_connect_twist_world")
    require_len(follower_connect_twist_world, 6, "follower_connect_twist_world")
    relative = compose_pose(
        inverse_pose(leader_connect_pose_world),
        follower_connect_pose_world,
    )
    error = compose_pose(inverse_pose(FACE_TO_FACE_DOCK_RELATION), relative)
    position_error = math.sqrt(sum(float(value) ** 2 for value in error[:3]))
    attitude_error = 2.0 * math.acos(min(1.0, abs(float(error[6]))))
    linear_error = math.sqrt(
        sum(
            (float(follower_connect_twist_world[index]) - float(leader_connect_twist_world[index]))
            ** 2
            for index in range(3)
        )
    )
    angular_error = math.sqrt(
        sum(
            (float(follower_connect_twist_world[index]) - float(leader_connect_twist_world[index]))
            ** 2
            for index in range(3, 6)
        )
    )
    return DynamicDockConstraintResidual(
        position_error_m=position_error,
        attitude_error_rad=attitude_error,
        relative_linear_speed_mps=linear_error,
        relative_angular_speed_radps=angular_error,
    )


def preauthor_disabled_fixed_joint(stage: Any, spec: DynamicDockConstraintSpec):
    """Author a disabled external FixedJoint before SimulationContext.reset."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    spec.validate()
    root_path = spec.prim_path.rsplit("/", 1)[0]
    UsdGeom.Scope.Define(stage, Sdf.Path(root_path))
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(spec.prim_path))
    joint.CreateJointEnabledAttr(False).Set(False)
    joint.CreateExcludeFromArticulationAttr(True).Set(True)
    joint.CreateCollisionEnabledAttr(True).Set(True)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(spec.leader_body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(spec.follower_body_path)])
    _set_local_frame(joint, 0, spec.leader_body_local_connect_pose, Gf)
    _set_local_frame(joint, 1, spec.follower_body_local_constraint_pose, Gf)
    return joint


def set_fixed_joint_enabled(joint: Any, enabled: bool) -> None:
    joint.GetJointEnabledAttr().Set(bool(enabled))
    from omni.physx import get_physx_simulation_interface

    get_physx_simulation_interface().flush_changes()


def selected_body_pair_filter_state(
    stage: Any,
    spec: DynamicDockConstraintSpec,
) -> dict[str, object]:
    """Return exact forward/reverse FilteredPairs targets for the Dock bodies."""

    from pxr import Sdf, UsdPhysics

    leader_prim = stage.GetPrimAtPath(Sdf.Path(spec.leader_body_path))
    follower_prim = stage.GetPrimAtPath(Sdf.Path(spec.follower_body_path))
    if not leader_prim.IsValid() or not follower_prim.IsValid():
        raise RuntimeError("selected dock body prim is missing for collision filtering")

    def targets(prim: Any) -> list[str]:
        relationship = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
        if not relationship:
            return []
        return sorted(target.pathString for target in relationship.GetTargets())

    return {
        "leader_body_path": spec.leader_body_path,
        "follower_body_path": spec.follower_body_path,
        "leader_body_prim_valid": True,
        "follower_body_prim_valid": True,
        "leader_is_rigid_body": bool(
            leader_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        "follower_is_rigid_body": bool(
            follower_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        "leader_targets": targets(leader_prim),
        "follower_targets": targets(follower_prim),
    }


def filter_selected_body_pair(
    stage: Any,
    spec: DynamicDockConstraintSpec,
) -> dict[str, object]:
    """Filter only the selected Dock-body pair and return an ownership delta.

    A pre-existing forward or reverse selected-pair filter is rejected so this
    runtime never claims ownership of, or later removes, another subsystem's
    filter.  Unrelated targets are preserved exactly.
    """

    from pxr import Sdf, UsdPhysics

    before = selected_body_pair_filter_state(stage, spec)
    if not before["leader_is_rigid_body"] or not before["follower_is_rigid_body"]:
        raise RuntimeError("selected dock body prim is not a rigid body")
    leader_before = set(before["leader_targets"])
    follower_before = set(before["follower_targets"])
    if (
        spec.follower_body_path in leader_before
        or spec.leader_body_path in follower_before
    ):
        raise RuntimeError("selected dock body pair was already collision-filtered")

    leader_prim = stage.GetPrimAtPath(Sdf.Path(spec.leader_body_path))
    follower_prim = stage.GetPrimAtPath(Sdf.Path(spec.follower_body_path))
    filtered = UsdPhysics.FilteredPairsAPI.Apply(leader_prim)
    filtered.CreateFilteredPairsRel().AddTarget(follower_prim.GetPath())
    from omni.physx import get_physx_simulation_interface

    get_physx_simulation_interface().flush_changes()
    after = selected_body_pair_filter_state(stage, spec)
    leader_after = set(after["leader_targets"])
    follower_after = set(after["follower_targets"])
    added_leader = sorted(leader_after - leader_before)
    removed_leader = sorted(leader_before - leader_after)
    added_follower = sorted(follower_after - follower_before)
    removed_follower = sorted(follower_before - follower_after)
    if (
        added_leader != [spec.follower_body_path]
        or removed_leader
        or added_follower
        or removed_follower
    ):
        raise RuntimeError("selected dock body filter changed an unexpected target")
    return {
        "leader_body_path": spec.leader_body_path,
        "follower_body_path": spec.follower_body_path,
        "leader_body_prim_valid": True,
        "follower_body_prim_valid": True,
        "leader_is_rigid_body": True,
        "follower_is_rigid_body": True,
        "leader_targets_before": sorted(leader_before),
        "follower_targets_before": sorted(follower_before),
        "leader_targets_after": sorted(leader_after),
        "follower_targets_after": sorted(follower_after),
        "added_leader_targets": added_leader,
        "removed_leader_targets": removed_leader,
        "added_follower_targets": added_follower,
        "removed_follower_targets": removed_follower,
    }


def unfilter_selected_body_pair(stage: Any, spec: DynamicDockConstraintSpec) -> None:
    from pxr import Sdf, UsdPhysics

    leader_prim = stage.GetPrimAtPath(Sdf.Path(spec.leader_body_path))
    if not leader_prim.IsValid():
        raise RuntimeError("leader dock body prim is missing for collision unfiltering")
    filtered = UsdPhysics.FilteredPairsAPI.Apply(leader_prim)
    filtered.GetFilteredPairsRel().RemoveTarget(Sdf.Path(spec.follower_body_path))
    from omni.physx import get_physx_simulation_interface

    get_physx_simulation_interface().flush_changes()


def selected_body_pair_filter_failures(
    stage: Any,
    spec: DynamicDockConstraintSpec,
    *,
    expected_filtered: bool,
) -> list[str]:
    from pxr import Sdf, UsdPhysics

    leader_prim = stage.GetPrimAtPath(Sdf.Path(spec.leader_body_path))
    follower_prim = stage.GetPrimAtPath(Sdf.Path(spec.follower_body_path))
    if not leader_prim.IsValid() or not follower_prim.IsValid():
        return ["selected_filter_body_prim_missing"]
    filtered = UsdPhysics.FilteredPairsAPI(leader_prim)
    relationship = filtered.GetFilteredPairsRel()
    if not relationship:
        return [] if not expected_filtered else ["selected_filter_relationship_missing"]
    targets = {target.pathString for target in relationship.GetTargets()}
    present = spec.follower_body_path in targets
    return (
        []
        if present is bool(expected_filtered)
        else ["selected_filter_target_state_mismatch"]
    )


def fixed_joint_identity_failures(
    stage: Any,
    spec: DynamicDockConstraintSpec,
    *,
    expected_enabled: bool = True,
) -> list[str]:
    """Return identity failures for the authored external fixed joint.

    Identity includes the runtime ``physics:jointEnabled`` state.  In
    particular, a correctly-authored but still-preauthored-disabled joint is
    not evidence that the dock constraint was created.  Callers that verify
    the pre-reset authored state can opt into ``expected_enabled=False``.
    A removed joint always fails identity verification because there is no
    constraint left whose identity can be established.
    """

    from pxr import Sdf, UsdPhysics

    failures: list[str] = []
    prim = stage.GetPrimAtPath(Sdf.Path(spec.prim_path))
    if not prim.IsValid():
        return ["constraint_prim_missing"]
    if not prim.IsA(UsdPhysics.FixedJoint):
        failures.append("constraint_prim_not_fixed_joint")
        return failures
    joint = UsdPhysics.FixedJoint(prim)
    enabled = joint.GetJointEnabledAttr().Get()
    if enabled is None or bool(enabled) != expected_enabled:
        failures.append("constraint_joint_enabled_mismatch")
    if [target.pathString for target in joint.GetBody0Rel().GetTargets()] != [spec.leader_body_path]:
        failures.append("constraint_body0_mismatch")
    if [target.pathString for target in joint.GetBody1Rel().GetTargets()] != [spec.follower_body_path]:
        failures.append("constraint_body1_mismatch")
    if joint.GetExcludeFromArticulationAttr().Get() is not True:
        failures.append("constraint_not_excluded_from_articulation")
    if joint.GetCollisionEnabledAttr().Get() is not True:
        failures.append("constraint_collision_enabled_mismatch")
    failures.extend(_local_frame_failures(joint, 0, spec.leader_body_local_connect_pose))
    failures.extend(_local_frame_failures(joint, 1, spec.follower_body_local_constraint_pose))
    return failures


def disable_and_remove_fixed_joint(stage: Any, spec: DynamicDockConstraintSpec) -> None:
    from pxr import Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(Sdf.Path(spec.prim_path))
    if prim.IsValid() and prim.IsA(UsdPhysics.FixedJoint):
        set_fixed_joint_enabled(UsdPhysics.FixedJoint(prim), False)
    stage.RemovePrim(Sdf.Path(spec.prim_path))
    from omni.physx import get_physx_simulation_interface

    get_physx_simulation_interface().flush_changes()


def _physical_port(graph_port: PortNode, physical_model: PhysicalModel) -> DockPortSpec:
    matches = [port for port in physical_model.dock_ports if port.port_id == graph_port.port_local_id]
    if len(matches) != 1:
        raise SchemaValidationError(
            f"PhysicalModel does not uniquely define DockPortSpec {graph_port.port_local_id!r}"
        )
    return matches[0]


def _set_local_frame(joint: Any, index: int, pose: Pose7D, gf: Any) -> None:
    position = gf.Vec3f(*(float(value) for value in pose[:3]))
    x, y, z, w = (float(value) for value in pose[3:7])
    rotation = gf.Quatf(w, x, y, z)
    if index == 0:
        joint.CreateLocalPos0Attr(position).Set(position)
        joint.CreateLocalRot0Attr(rotation).Set(rotation)
    else:
        joint.CreateLocalPos1Attr(position).Set(position)
        joint.CreateLocalRot1Attr(rotation).Set(rotation)


def _local_frame_failures(joint: Any, index: int, expected: Pose7D) -> list[str]:
    if index == 0:
        position = joint.GetLocalPos0Attr().Get()
        rotation = joint.GetLocalRot0Attr().Get()
    else:
        position = joint.GetLocalPos1Attr().Get()
        rotation = joint.GetLocalRot1Attr().Get()
    if position is None or rotation is None:
        return [f"constraint_local_frame_{index}_missing"]
    try:
        imaginary = rotation.GetImaginary()
        real = rotation.GetReal()
    except (AttributeError, TypeError, ValueError):
        return [f"constraint_local_frame_{index}_invalid"]
    try:
        actual = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(real),
        )
    except (IndexError, TypeError, ValueError):
        return [f"constraint_local_frame_{index}_invalid"]
    return [] if _pose_close(actual, expected) else [f"constraint_local_frame_{index}_mismatch"]


def _pose_close(left: Pose7D, right: Pose7D, tolerance: float = 1.0e-6) -> bool:
    position_close = all(abs(float(left[index]) - float(right[index])) <= tolerance for index in range(3))
    direct = all(abs(float(left[index]) - float(right[index])) <= tolerance for index in range(3, 7))
    negated = all(abs(float(left[index]) + float(right[index])) <= tolerance for index in range(3, 7))
    return position_close and (direct or negated)


__all__ = [
    "DYNAMIC_DOCK_CONSTRAINT_VERSION",
    "DynamicDockConstraintRecord",
    "DynamicDockConstraintResidual",
    "DynamicDockConstraintSpec",
    "build_dynamic_dock_constraint_spec",
    "connect_frame_pose_in_parent_body",
    "constraint_residual",
    "disable_and_remove_fixed_joint",
    "filter_selected_body_pair",
    "fixed_joint_identity_failures",
    "preauthor_disabled_fixed_joint",
    "selected_body_pair_filter_failures",
    "selected_body_pair_filter_state",
    "set_fixed_joint_enabled",
    "unfilter_selected_body_pair",
]
