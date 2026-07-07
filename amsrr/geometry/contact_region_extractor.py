from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from amsrr.geometry.asset_resolver import GeometryAssetReference
from amsrr.geometry.surface_patch_graph import (
    add,
    angle_between,
    bbox_from_points,
    center_from_bbox,
    dominant_normal_cluster,
    distance,
    dot,
    normalize,
    orthonormal_basis,
    principal_axes_identity_flat,
    scale as vec_scale,
    tetra_signed_volume,
    triangle_area,
    triangle_centroid,
    triangle_normal,
)
from amsrr.schemas.common import ContactMode, SchemaValidationError, Vector3
from amsrr.schemas.geometry import (
    ContactRegion,
    ContactRegionEdge,
    ContactRegionGraph,
    GlobalShapeFeatures,
    SurfacePatchEdge,
    SurfacePatchGraph,
    SurfacePatchToken,
)
from amsrr.schemas.task_spec import GeometrySpec, GeometryType


DEFAULT_CONTACT_MODES = [ContactMode.GRASP, ContactMode.SUPPORT, ContactMode.PUSH]


@dataclass
class GeometryExtraction:
    global_shape_features: GlobalShapeFeatures
    surface_patch_graph: SurfacePatchGraph
    contact_region_graph: ContactRegionGraph


@dataclass
class _MeshCluster:
    key: str
    area_m2: float = 0.0
    weighted_centroid: Vector3 = (0.0, 0.0, 0.0)
    weighted_normal: Vector3 = (0.0, 0.0, 0.0)
    triangle_count: int = 0

    def add_triangle(self, centroid: Vector3, normal: Vector3, area: float) -> None:
        self.area_m2 += area
        self.weighted_centroid = add(self.weighted_centroid, vec_scale(centroid, area))
        self.weighted_normal = add(self.weighted_normal, vec_scale(normal, area))
        self.triangle_count += 1

    def centroid(self) -> Vector3:
        if self.area_m2 <= 1.0e-12:
            return (0.0, 0.0, 0.0)
        return vec_scale(self.weighted_centroid, 1.0 / self.area_m2)

    def normal(self) -> Vector3:
        return normalize(self.weighted_normal)


@dataclass
class MeshSummary:
    bbox_m: Vector3
    bbox_min: Vector3
    bbox_max: Vector3
    surface_area_m2: float
    volume_m3: float
    centroid_object: Vector3
    clusters: list[_MeshCluster]
    triangle_count: int


def _modes(allowed_contact_modes: list[ContactMode] | None) -> list[ContactMode]:
    return allowed_contact_modes or list(DEFAULT_CONTACT_MODES)


def _scaled_tuple(values: Iterable[float], scale: Vector3) -> Vector3:
    x, y, z = values
    return (float(x) * scale[0], float(y) * scale[1], float(z) * scale[2])


def _global_features(bbox: Vector3, volume: float, area: float, com: Vector3 | None = None) -> GlobalShapeFeatures:
    bbox_volume = max(bbox[0] * bbox[1] * bbox[2], 1.0e-12)
    inertia_diag = (
        (bbox[1] * bbox[1] + bbox[2] * bbox[2]) / 12.0,
        (bbox[0] * bbox[0] + bbox[2] * bbox[2]) / 12.0,
        (bbox[0] * bbox[0] + bbox[1] * bbox[1]) / 12.0,
    )
    compactness = max(0.0, min(1.0, volume / bbox_volume)) if bbox_volume > 0.0 else 0.0
    symmetry = [
        1.0 if abs(bbox[0] - bbox[1]) <= 1.0e-9 else 0.0,
        1.0 if abs(bbox[1] - bbox[2]) <= 1.0e-9 else 0.0,
        1.0 if abs(bbox[0] - bbox[2]) <= 1.0e-9 else 0.0,
    ]
    return GlobalShapeFeatures(
        bbox_m=bbox,
        volume_m3=max(0.0, volume),
        surface_area_m2=max(0.0, area),
        approximate_com_object=com or (0.0, 0.0, 0.0),
        approximate_inertia_diag=inertia_diag,
        principal_axes_flat=principal_axes_identity_flat(),
        compactness=compactness,
        symmetry_features=symmetry,
    )


def _patch(
    patch_id: int,
    entity_id: str,
    position: Vector3,
    normal: Vector3,
    area: float,
    *,
    thickness: float | None,
    friction: float | None,
    contact_allowed: bool,
    allowed_contact_modes: list[ContactMode],
) -> SurfacePatchToken:
    tangent_u, tangent_v = orthonormal_basis(normal)
    return SurfacePatchToken(
        patch_id=patch_id,
        entity_id=entity_id,
        position_object=position,
        normal_object=normalize(normal),
        tangent_u_object=tangent_u,
        tangent_v_object=tangent_v,
        patch_area_m2=area,
        mean_curvature=0.0,
        gaussian_curvature=0.0,
        local_thickness_m=thickness,
        friction=friction,
        contact_allowed=contact_allowed,
        allowed_contact_modes=allowed_contact_modes,
    )


def _region(
    region_id: str,
    entity_id: str,
    region_type: str,
    patch_id: int,
    normal: Vector3,
    area: float,
    *,
    friction: float | None,
    allowed_contact_modes: list[ContactMode],
) -> ContactRegion:
    return ContactRegion(
        region_id=region_id,
        entity_id=entity_id,
        region_type=region_type,  # type: ignore[arg-type]
        patch_ids=[patch_id],
        pose_object=None,
        normal_summary_object=normalize(normal),
        area_m2=area,
        curvature_summary=[0.0, 0.0],
        friction=friction,
        allowed_contact_modes=allowed_contact_modes,
        task_relevance_features=[],
    )


def _graph_edges_for_patches(patches: list[SurfacePatchToken]) -> list[SurfacePatchEdge]:
    edges: list[SurfacePatchEdge] = []
    for src in patches:
        for dst in patches:
            if src.patch_id >= dst.patch_id:
                continue
            normal_angle = angle_between(src.normal_object, dst.normal_object)
            if abs(normal_angle - math.pi) <= 1.0e-6:
                edge_type = "opposite_patch"
            elif normal_angle <= 1.0e-6:
                edge_type = "normal_cluster"
            else:
                edge_type = "adjacent"
            edges.append(
                SurfacePatchEdge(
                    src_patch_id=src.patch_id,
                    dst_patch_id=dst.patch_id,
                    edge_type=edge_type,  # type: ignore[arg-type]
                    distance_m=distance(src.position_object, dst.position_object),
                    normal_angle_rad=normal_angle,
                )
            )
    return edges


def _graph_edges_for_regions(regions: list[ContactRegion]) -> list[ContactRegionEdge]:
    edges: list[ContactRegionEdge] = []
    for src in regions:
        for dst in regions:
            if src.region_id >= dst.region_id:
                continue
            normal_angle = angle_between(src.normal_summary_object, dst.normal_summary_object)
            edge_type = "opposite_region" if abs(normal_angle - math.pi) <= 1.0e-6 else "adjacent_region"
            edges.append(
                ContactRegionEdge(
                    src_region_id=src.region_id,
                    dst_region_id=dst.region_id,
                    edge_type=edge_type,  # type: ignore[arg-type]
                    normal_angle_rad=normal_angle,
                    params={"entity_relation": "same_object"},
                )
            )
    return edges


def _box_params(params: dict, scale: Vector3) -> Vector3:
    if "size_m" in params:
        size = params["size_m"]
    elif all(key in params for key in ("x_m", "y_m", "z_m")):
        size = [params["x_m"], params["y_m"], params["z_m"]]
    else:
        raise SchemaValidationError("box primitive requires size_m or x_m/y_m/z_m")
    if len(size) != 3:
        raise SchemaValidationError("box size_m must have length 3")
    return _scaled_tuple(size, scale)


def _sphere_radius(params: dict, scale: Vector3) -> float:
    radius = params.get("radius_m", params.get("radius"))
    if radius is None:
        raise SchemaValidationError("sphere primitive requires radius_m")
    return float(radius) * max(scale)


def _cylinder_params(params: dict, scale: Vector3) -> tuple[float, float]:
    radius = params.get("radius_m", params.get("radius"))
    height = params.get("height_m", params.get("height"))
    if radius is None or height is None:
        raise SchemaValidationError("cylinder primitive requires radius_m and height_m")
    return float(radius) * max(scale[0], scale[1]), float(height) * scale[2]


def extract_primitive_geometry(
    geometry_spec: GeometrySpec,
    *,
    entity_id: str | None = None,
    friction: float | None = None,
    contact_allowed: bool = True,
    allowed_contact_modes: list[ContactMode] | None = None,
) -> GeometryExtraction:
    if geometry_spec.primitive_params is None:
        raise SchemaValidationError(f"{geometry_spec.geometry_id} requires primitive_params")
    entity = entity_id or geometry_spec.geometry_id
    modes = _modes(allowed_contact_modes)

    if geometry_spec.geometry_type == GeometryType.BOX:
        sx, sy, sz = _box_params(geometry_spec.primitive_params, geometry_spec.scale)
        face_specs = [
            ("pos_x", (sx * 0.5, 0.0, 0.0), (1.0, 0.0, 0.0), sy * sz, sx),
            ("neg_x", (-sx * 0.5, 0.0, 0.0), (-1.0, 0.0, 0.0), sy * sz, sx),
            ("pos_y", (0.0, sy * 0.5, 0.0), (0.0, 1.0, 0.0), sx * sz, sy),
            ("neg_y", (0.0, -sy * 0.5, 0.0), (0.0, -1.0, 0.0), sx * sz, sy),
            ("pos_z", (0.0, 0.0, sz * 0.5), (0.0, 0.0, 1.0), sx * sy, sz),
            ("neg_z", (0.0, 0.0, -sz * 0.5), (0.0, 0.0, -1.0), sx * sy, sz),
        ]
        patches = [
            _patch(
                idx,
                entity,
                position,
                normal,
                area,
                thickness=thickness,
                friction=friction,
                contact_allowed=contact_allowed,
                allowed_contact_modes=modes,
            )
            for idx, (_, position, normal, area, thickness) in enumerate(face_specs)
        ]
        regions = [
            _region(
                f"{entity}_face_{name}",
                entity,
                "face",
                idx,
                normal,
                area,
                friction=friction,
                allowed_contact_modes=modes,
            )
            for idx, (name, _, normal, area, _) in enumerate(face_specs)
        ]
        features = _global_features((sx, sy, sz), sx * sy * sz, 2.0 * (sx * sy + sx * sz + sy * sz))
        return GeometryExtraction(features, SurfacePatchGraph(patches, _graph_edges_for_patches(patches)), ContactRegionGraph(regions, _graph_edges_for_regions(regions)))

    if geometry_spec.geometry_type == GeometryType.SPHERE:
        radius = _sphere_radius(geometry_spec.primitive_params, geometry_spec.scale)
        directions = [
            ("pos_x", (1.0, 0.0, 0.0)),
            ("neg_x", (-1.0, 0.0, 0.0)),
            ("pos_y", (0.0, 1.0, 0.0)),
            ("neg_y", (0.0, -1.0, 0.0)),
            ("pos_z", (0.0, 0.0, 1.0)),
            ("neg_z", (0.0, 0.0, -1.0)),
        ]
        patch_area = 4.0 * math.pi * radius * radius / len(directions)
        patches = [
            _patch(idx, entity, vec_scale(normal, radius), normal, patch_area, thickness=radius * 2.0, friction=friction, contact_allowed=contact_allowed, allowed_contact_modes=modes)
            for idx, (_, normal) in enumerate(directions)
        ]
        regions = [
            _region(f"{entity}_sphere_cluster_{name}", entity, "curved_patch", idx, normal, patch_area, friction=friction, allowed_contact_modes=modes)
            for idx, (name, normal) in enumerate(directions)
        ]
        features = _global_features((2.0 * radius, 2.0 * radius, 2.0 * radius), 4.0 / 3.0 * math.pi * radius**3, 4.0 * math.pi * radius**2)
        return GeometryExtraction(features, SurfacePatchGraph(patches, _graph_edges_for_patches(patches)), ContactRegionGraph(regions, _graph_edges_for_regions(regions)))

    if geometry_spec.geometry_type == GeometryType.CYLINDER:
        radius, height = _cylinder_params(geometry_spec.primitive_params, geometry_spec.scale)
        side_area = 2.0 * math.pi * radius * height
        cap_area = math.pi * radius * radius
        patch_specs = [
            ("side", (radius, 0.0, 0.0), (1.0, 0.0, 0.0), side_area, "curved_patch"),
            ("top", (0.0, 0.0, height * 0.5), (0.0, 0.0, 1.0), cap_area, "face"),
            ("bottom", (0.0, 0.0, -height * 0.5), (0.0, 0.0, -1.0), cap_area, "face"),
        ]
        patches = [
            _patch(idx, entity, position, normal, area, thickness=height if name == "side" else radius * 2.0, friction=friction, contact_allowed=contact_allowed, allowed_contact_modes=modes)
            for idx, (name, position, normal, area, _) in enumerate(patch_specs)
        ]
        regions = [
            _region(f"{entity}_cylinder_{name}", entity, region_type, idx, normal, area, friction=friction, allowed_contact_modes=modes)
            for idx, (name, _, normal, area, region_type) in enumerate(patch_specs)
        ]
        features = _global_features((2.0 * radius, 2.0 * radius, height), math.pi * radius * radius * height, side_area + 2.0 * cap_area)
        return GeometryExtraction(features, SurfacePatchGraph(patches, _graph_edges_for_patches(patches)), ContactRegionGraph(regions, _graph_edges_for_regions(regions)))

    if geometry_spec.geometry_type == GeometryType.CAPSULE:
        radius, cylinder_height = _cylinder_params(geometry_spec.primitive_params, geometry_spec.scale)
        side_area = 2.0 * math.pi * radius * cylinder_height
        hemi_area = 2.0 * math.pi * radius * radius
        patch_specs = [
            ("side", (radius, 0.0, 0.0), (1.0, 0.0, 0.0), side_area),
            ("pos_end", (0.0, 0.0, cylinder_height * 0.5 + radius), (0.0, 0.0, 1.0), hemi_area),
            ("neg_end", (0.0, 0.0, -cylinder_height * 0.5 - radius), (0.0, 0.0, -1.0), hemi_area),
        ]
        patches = [
            _patch(idx, entity, position, normal, area, thickness=radius * 2.0, friction=friction, contact_allowed=contact_allowed, allowed_contact_modes=modes)
            for idx, (_, position, normal, area) in enumerate(patch_specs)
        ]
        regions = [
            _region(f"{entity}_capsule_{name}", entity, "curved_patch", idx, normal, area, friction=friction, allowed_contact_modes=modes)
            for idx, (name, _, normal, area) in enumerate(patch_specs)
        ]
        total_height = cylinder_height + 2.0 * radius
        volume = math.pi * radius * radius * cylinder_height + 4.0 / 3.0 * math.pi * radius**3
        area = side_area + 2.0 * hemi_area
        features = _global_features((2.0 * radius, 2.0 * radius, total_height), volume, area)
        return GeometryExtraction(features, SurfacePatchGraph(patches, _graph_edges_for_patches(patches)), ContactRegionGraph(regions, _graph_edges_for_regions(regions)))

    raise SchemaValidationError(f"Unsupported primitive type: {geometry_spec.geometry_type}")


def _read_binary_stl(path: Path, scale: Vector3) -> MeshSummary:
    clusters: dict[str, _MeshCluster] = {}
    points: list[Vector3] = []
    surface_area = 0.0
    volume = 0.0
    triangle_count = 0
    with path.open("rb") as handle:
        header = handle.read(80)
        count_data = handle.read(4)
        if len(header) != 80 or len(count_data) != 4:
            raise SchemaValidationError(f"Invalid binary STL: {path}")
        triangle_total = struct.unpack("<I", count_data)[0]
        for _ in range(triangle_total):
            payload = handle.read(50)
            if len(payload) != 50:
                raise SchemaValidationError(f"Truncated binary STL: {path}")
            values = struct.unpack("<12fH", payload)
            a = _scaled_tuple(values[3:6], scale)
            b = _scaled_tuple(values[6:9], scale)
            c = _scaled_tuple(values[9:12], scale)
            area = triangle_area(a, b, c)
            if area <= 1.0e-12:
                continue
            normal = triangle_normal(a, b, c)
            centroid = triangle_centroid(a, b, c)
            key = dominant_normal_cluster(normal)
            clusters.setdefault(key, _MeshCluster(key)).add_triangle(centroid, normal, area)
            points.extend([a, b, c])
            surface_area += area
            volume += tetra_signed_volume(a, b, c)
            triangle_count += 1
    mins, maxs, bbox = bbox_from_points(points)
    bbox_volume = bbox[0] * bbox[1] * bbox[2]
    mesh_volume = abs(volume)
    if mesh_volume <= 1.0e-12 and bbox_volume > 0.0:
        mesh_volume = bbox_volume
    return MeshSummary(bbox, mins, maxs, surface_area, mesh_volume, center_from_bbox(mins, maxs), sorted(clusters.values(), key=lambda item: item.key), triangle_count)


def _read_ascii_stl(path: Path, scale: Vector3) -> MeshSummary:
    clusters: dict[str, _MeshCluster] = {}
    points: list[Vector3] = []
    current: list[Vector3] = []
    surface_area = 0.0
    volume = 0.0
    triangle_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("vertex "):
                continue
            current.append(_scaled_tuple([float(item) for item in line.split()[1:4]], scale))
            if len(current) == 3:
                a, b, c = current
                current = []
                area = triangle_area(a, b, c)
                if area <= 1.0e-12:
                    continue
                normal = triangle_normal(a, b, c)
                centroid = triangle_centroid(a, b, c)
                key = dominant_normal_cluster(normal)
                clusters.setdefault(key, _MeshCluster(key)).add_triangle(centroid, normal, area)
                points.extend([a, b, c])
                surface_area += area
                volume += tetra_signed_volume(a, b, c)
                triangle_count += 1
    mins, maxs, bbox = bbox_from_points(points)
    mesh_volume = abs(volume) or bbox[0] * bbox[1] * bbox[2]
    return MeshSummary(bbox, mins, maxs, surface_area, mesh_volume, center_from_bbox(mins, maxs), sorted(clusters.values(), key=lambda item: item.key), triangle_count)


def _is_binary_stl(path: Path) -> bool:
    size = path.stat().st_size
    if size < 84:
        return False
    with path.open("rb") as handle:
        header = handle.read(80)
        count_data = handle.read(4)
    if len(count_data) != 4:
        return False
    triangle_total = struct.unpack("<I", count_data)[0]
    if 84 + triangle_total * 50 == size:
        return True
    return not header.lstrip().lower().startswith(b"solid")


def _read_obj(path: Path, scale: Vector3) -> MeshSummary:
    vertices: list[Vector3] = []
    triangles: list[tuple[Vector3, Vector3, Vector3]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                vertices.append(_scaled_tuple([float(item) for item in line.split()[1:4]], scale))
            elif line.startswith("f "):
                indices = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                for idx in range(1, len(indices) - 1):
                    triangles.append((vertices[indices[0]], vertices[indices[idx]], vertices[indices[idx + 1]]))
    clusters: dict[str, _MeshCluster] = {}
    points: list[Vector3] = []
    surface_area = 0.0
    volume = 0.0
    for a, b, c in triangles:
        area = triangle_area(a, b, c)
        if area <= 1.0e-12:
            continue
        normal = triangle_normal(a, b, c)
        centroid = triangle_centroid(a, b, c)
        key = dominant_normal_cluster(normal)
        clusters.setdefault(key, _MeshCluster(key)).add_triangle(centroid, normal, area)
        points.extend([a, b, c])
        surface_area += area
        volume += tetra_signed_volume(a, b, c)
    mins, maxs, bbox = bbox_from_points(points)
    mesh_volume = abs(volume) or bbox[0] * bbox[1] * bbox[2]
    return MeshSummary(bbox, mins, maxs, surface_area, mesh_volume, center_from_bbox(mins, maxs), sorted(clusters.values(), key=lambda item: item.key), len(triangles))


def load_mesh_summary(path: Path, scale: Vector3) -> MeshSummary:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return _read_binary_stl(path, scale) if _is_binary_stl(path) else _read_ascii_stl(path, scale)
    if suffix == ".obj":
        return _read_obj(path, scale)
    raise SchemaValidationError(f"Unsupported mesh format for P0 smoke: {path.suffix}")


def extract_mesh_geometry(
    geometry_spec: GeometrySpec,
    asset_ref: GeometryAssetReference,
    *,
    entity_id: str | None = None,
    friction: float | None = None,
    contact_allowed: bool = True,
    allowed_contact_modes: list[ContactMode] | None = None,
) -> GeometryExtraction:
    if asset_ref.asset_path is None:
        raise SchemaValidationError("mesh extraction requires an asset path")
    entity = entity_id or geometry_spec.geometry_id
    modes = _modes(allowed_contact_modes)
    summary = load_mesh_summary(asset_ref.asset_path, geometry_spec.scale)
    patches: list[SurfacePatchToken] = []
    regions: list[ContactRegion] = []
    for idx, cluster in enumerate(summary.clusters):
        normal = cluster.normal()
        patches.append(
            _patch(
                idx,
                entity,
                cluster.centroid(),
                normal,
                cluster.area_m2,
                thickness=None,
                friction=friction,
                contact_allowed=contact_allowed,
                allowed_contact_modes=modes,
            )
        )
        regions.append(
            _region(
                f"{entity}_mesh_cluster_{cluster.key}",
                entity,
                "mesh_patch_cluster",
                idx,
                normal,
                cluster.area_m2,
                friction=friction,
                allowed_contact_modes=modes,
            )
        )
    features = _global_features(summary.bbox_m, summary.volume_m3, summary.surface_area_m2, summary.centroid_object)
    return GeometryExtraction(
        features,
        SurfacePatchGraph(patches, _graph_edges_for_patches(patches)),
        ContactRegionGraph(regions, _graph_edges_for_regions(regions)),
    )

