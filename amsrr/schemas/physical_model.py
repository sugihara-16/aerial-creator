from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import Pose7D, SchemaBase, Vector3, require_len, require_non_empty, require_non_negative


@dataclass
class CollisionPrimitive(SchemaBase):
    primitive_id: str
    link_id: str
    primitive_type: Literal["box", "sphere", "cylinder", "capsule", "mesh", "convex", "sdf"]
    local_pose: Pose7D | None = None
    params: dict[str, Any] = field(default_factory=dict)
    geometry_ref: str | None = None

    def validate(self) -> None:
        require_non_empty(self.primitive_id, "CollisionPrimitive.primitive_id")
        require_non_empty(self.link_id, "CollisionPrimitive.link_id")
        if self.local_pose is not None:
            require_len(self.local_pose, 7, "CollisionPrimitive.local_pose")


@dataclass
class LinkModel(SchemaBase):
    link_id: str
    parent_joint_id: str | None
    mass_kg: float
    inertia_kgm2: list[float]
    local_com: Vector3
    visual_geometry_ref: str | None
    collision_geometry_ref: str | None

    def validate(self) -> None:
        require_non_empty(self.link_id, "LinkModel.link_id")
        require_non_negative(self.mass_kg, "LinkModel.mass_kg")
        require_len(self.inertia_kgm2, 6, "LinkModel.inertia_kgm2")
        require_len(self.local_com, 3, "LinkModel.local_com")


@dataclass
class JointModel(SchemaBase):
    joint_id: str
    joint_type: Literal["fixed", "revolute", "continuous", "prismatic"]
    parent_link: str
    child_link: str
    origin_xyz: Vector3
    origin_rpy: Vector3
    axis_xyz: Vector3
    limit_lower: float | None
    limit_upper: float | None
    effort_limit: float | None
    velocity_limit: float | None

    def validate(self) -> None:
        require_non_empty(self.joint_id, "JointModel.joint_id")
        require_non_empty(self.parent_link, "JointModel.parent_link")
        require_non_empty(self.child_link, "JointModel.child_link")
        require_len(self.origin_xyz, 3, "JointModel.origin_xyz")
        require_len(self.origin_rpy, 3, "JointModel.origin_rpy")
        require_len(self.axis_xyz, 3, "JointModel.axis_xyz")


@dataclass
class RotorModel(SchemaBase):
    rotor_id: str
    thrust_frame_link: str
    thrust_axis_local: Vector3
    thrust_min_n: float
    thrust_max_n: float
    reaction_torque_coeff_nm_per_n: float
    vectoring_joint_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.rotor_id, "RotorModel.rotor_id")
        require_non_empty(self.thrust_frame_link, "RotorModel.thrust_frame_link")
        require_len(self.thrust_axis_local, 3, "RotorModel.thrust_axis_local")
        require_non_negative(self.thrust_min_n, "RotorModel.thrust_min_n")
        require_non_negative(self.thrust_max_n, "RotorModel.thrust_max_n")
        if self.thrust_max_n < self.thrust_min_n:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("RotorModel.thrust_max_n must be >= thrust_min_n")


@dataclass
class DockPortSpec(SchemaBase):
    port_id: str
    parent_link: str
    local_pose: Pose7D
    port_type: Literal["pitch_dock", "yaw_dock", "generic_dock"]
    compatible_port_types: list[str]
    latch_axis_local: Vector3 | None = None
    mechanical_limits: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.port_id, "DockPortSpec.port_id")
        require_non_empty(self.parent_link, "DockPortSpec.parent_link")
        require_len(self.local_pose, 7, "DockPortSpec.local_pose")
        if self.latch_axis_local is not None:
            require_len(self.latch_axis_local, 3, "DockPortSpec.latch_axis_local")


@dataclass
class ModuleCapabilityToken(SchemaBase):
    module_type: str
    aggregate_mass_norm: float
    aggregate_inertia_features: list[float]
    rotor_count: int
    port_count: int
    thrust_min_features: list[float]
    thrust_max_features: list[float]
    thrust_to_weight_ratio_est: float
    dock_port_type_counts: list[int]
    has_vectoring: bool
    has_dock_mechanism: bool

    def validate(self) -> None:
        require_non_empty(self.module_type, "ModuleCapabilityToken.module_type")
        require_non_negative(self.aggregate_mass_norm, "ModuleCapabilityToken.aggregate_mass_norm")
        require_non_negative(self.rotor_count, "ModuleCapabilityToken.rotor_count")
        require_non_negative(self.port_count, "ModuleCapabilityToken.port_count")
        require_non_negative(self.thrust_to_weight_ratio_est, "ModuleCapabilityToken.thrust_to_weight_ratio_est")


@dataclass
class PhysicalModel(SchemaBase):
    model_id: str
    urdf_path: str
    links: list[LinkModel]
    joints: list[JointModel]
    rotors: list[RotorModel]
    dock_ports: list[DockPortSpec]
    collision_primitives: list[CollisionPrimitive]
    aggregate_mass_kg: float
    aggregate_inertia_body: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.model_id, "PhysicalModel.model_id")
        require_non_empty(self.urdf_path, "PhysicalModel.urdf_path")
        require_non_negative(self.aggregate_mass_kg, "PhysicalModel.aggregate_mass_kg")
        require_len(self.aggregate_inertia_body, 6, "PhysicalModel.aggregate_inertia_body")
        link_ids = [link.link_id for link in self.links]
        if len(link_ids) != len(set(link_ids)):
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("PhysicalModel.links has duplicate link_id values")
        known_links = set(link_ids)
        for joint in self.joints:
            if joint.parent_link not in known_links or joint.child_link not in known_links:
                from amsrr.schemas.common import SchemaValidationError

                raise SchemaValidationError(f"JointModel {joint.joint_id!r} references an unknown link")

