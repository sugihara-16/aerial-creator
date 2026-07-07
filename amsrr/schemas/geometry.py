from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.schemas.common import (
    ContactMode,
    Pose7D,
    SchemaBase,
    Vector3,
    require_len,
    require_non_empty,
    require_non_negative,
)


@dataclass
class GlobalShapeFeatures(SchemaBase):
    bbox_m: Vector3
    volume_m3: float
    surface_area_m2: float
    approximate_com_object: Vector3
    approximate_inertia_diag: Vector3
    principal_axes_flat: list[float]
    compactness: float
    symmetry_features: list[float]

    def validate(self) -> None:
        require_len(self.bbox_m, 3, "GlobalShapeFeatures.bbox_m")
        require_len(self.approximate_com_object, 3, "GlobalShapeFeatures.approximate_com_object")
        require_len(self.approximate_inertia_diag, 3, "GlobalShapeFeatures.approximate_inertia_diag")
        require_len(self.principal_axes_flat, 9, "GlobalShapeFeatures.principal_axes_flat")
        require_non_negative(self.volume_m3, "GlobalShapeFeatures.volume_m3")
        require_non_negative(self.surface_area_m2, "GlobalShapeFeatures.surface_area_m2")
        require_non_negative(self.compactness, "GlobalShapeFeatures.compactness")


@dataclass
class SurfacePatchToken(SchemaBase):
    patch_id: int
    entity_id: str
    position_object: Vector3
    normal_object: Vector3
    tangent_u_object: Vector3
    tangent_v_object: Vector3
    patch_area_m2: float
    mean_curvature: float
    gaussian_curvature: float
    local_thickness_m: float | None
    friction: float | None
    contact_allowed: bool
    allowed_contact_modes: list[ContactMode]

    def validate(self) -> None:
        require_non_empty(self.entity_id, "SurfacePatchToken.entity_id")
        require_len(self.position_object, 3, "SurfacePatchToken.position_object")
        require_len(self.normal_object, 3, "SurfacePatchToken.normal_object")
        require_len(self.tangent_u_object, 3, "SurfacePatchToken.tangent_u_object")
        require_len(self.tangent_v_object, 3, "SurfacePatchToken.tangent_v_object")
        require_non_negative(self.patch_area_m2, "SurfacePatchToken.patch_area_m2")
        if self.local_thickness_m is not None:
            require_non_negative(self.local_thickness_m, "SurfacePatchToken.local_thickness_m")
        if self.friction is not None:
            require_non_negative(self.friction, "SurfacePatchToken.friction")


@dataclass
class SurfacePatchEdge(SchemaBase):
    src_patch_id: int
    dst_patch_id: int
    edge_type: Literal["adjacent", "same_region", "rim_neighbor", "opposite_patch", "normal_cluster"]
    distance_m: float
    normal_angle_rad: float

    def validate(self) -> None:
        require_non_negative(self.distance_m, "SurfacePatchEdge.distance_m")
        require_non_negative(self.normal_angle_rad, "SurfacePatchEdge.normal_angle_rad")


@dataclass
class SurfacePatchGraph(SchemaBase):
    nodes: list[SurfacePatchToken]
    edges: list[SurfacePatchEdge] = field(default_factory=list)


@dataclass
class ContactRegion(SchemaBase):
    region_id: str
    entity_id: str
    region_type: Literal["face", "rim", "edge", "pipe", "floor", "wall", "curved_patch", "mesh_patch_cluster"]
    patch_ids: list[int]
    pose_object: Pose7D | None
    normal_summary_object: Vector3
    area_m2: float
    curvature_summary: list[float]
    friction: float | None
    allowed_contact_modes: list[ContactMode]
    task_relevance_features: list[float] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.region_id, "ContactRegion.region_id")
        require_non_empty(self.entity_id, "ContactRegion.entity_id")
        require_len(self.normal_summary_object, 3, "ContactRegion.normal_summary_object")
        if self.pose_object is not None:
            require_len(self.pose_object, 7, "ContactRegion.pose_object")
        require_non_negative(self.area_m2, "ContactRegion.area_m2")
        if self.friction is not None:
            require_non_negative(self.friction, "ContactRegion.friction")


@dataclass
class ContactRegionEdge(SchemaBase):
    src_region_id: str
    dst_region_id: str
    edge_type: Literal[
        "adjacent_region",
        "opposite_region",
        "same_object",
        "spatially_near",
        "supports_moment_arm",
        "mutually_exclusive_contact",
    ]
    distance_m: float | None = None
    normal_angle_rad: float | None = None
    params: dict[str, float | str | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.src_region_id, "ContactRegionEdge.src_region_id")
        require_non_empty(self.dst_region_id, "ContactRegionEdge.dst_region_id")
        if self.distance_m is not None:
            require_non_negative(self.distance_m, "ContactRegionEdge.distance_m")
        if self.normal_angle_rad is not None:
            require_non_negative(self.normal_angle_rad, "ContactRegionEdge.normal_angle_rad")


@dataclass
class ContactRegionGraph(SchemaBase):
    nodes: list[ContactRegion]
    edges: list[ContactRegionEdge] = field(default_factory=list)

    def validate(self) -> None:
        ids = [node.region_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("ContactRegionGraph.nodes has duplicate region_id values")


@dataclass
class GeometryDescriptor(SchemaBase):
    geometry_id: str
    global_shape_features: GlobalShapeFeatures
    surface_patch_graph: SurfacePatchGraph
    contact_region_graph: ContactRegionGraph
    collision_ref: str
    exact_geometry_ref: str

    def validate(self) -> None:
        require_non_empty(self.geometry_id, "GeometryDescriptor.geometry_id")
        require_non_empty(self.collision_ref, "GeometryDescriptor.collision_ref")
        require_non_empty(self.exact_geometry_ref, "GeometryDescriptor.exact_geometry_ref")

