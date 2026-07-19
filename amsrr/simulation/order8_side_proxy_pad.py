from __future__ import annotations

"""Pure geometry for the Order-8 Dock-cone micro-pad preview.

The preview samples only the conical contact shell of each selected yaw Dock
collision mesh in an axial/circumferential grid.  Every occupied surface cell
gets its own small, thin box whose orientation is fitted to nearby authored
STL triangles.  Coarser cells merge visually similar neighbours without
extending onto the cylindrical/mounting structure.  Pads are fixed in the
Dock-link frame; neither an object pose nor simulator contact data is used to
place them.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import struct
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.config import load_yaml


ORDER8_SIDE_PROXY_PAD_PREVIEW_VERSION = "order8_side_proxy_pad_preview_v4"
_TARGET_LINK_IDS = ("yaw_dock_mech1", "yaw_dock_mech2")
_LINK_AXIAL_AXIS = np.asarray((1.0, 0.0, 0.0), dtype=float)


@dataclass(frozen=True)
class Order8SideProxyPadPreviewConfig:
    version: str
    acceptance_eligible: bool
    visual_approval_recorded: bool
    contact_runtime_enabled: bool
    thickness_m: float
    mesh_clearance_m: float
    axial_band_count: int
    circumferential_segment_count: int
    cone_axial_min_m: float
    cone_axial_max_m: float
    cone_normal_axial_min: float
    cone_normal_axial_max: float
    side_min_radial_alignment: float
    side_max_axial_normal_abs: float
    outer_envelope_depth_m: float
    local_plane_tolerance_m: float
    normal_similarity_cosine: float
    tile_coverage_scale: float
    tile_size_min_m: float
    tile_size_max_m: float
    display_color_rgb: tuple[float, float, float]
    display_opacity: float
    mesh_search_paths: tuple[str, ...]
    link_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.version != ORDER8_SIDE_PROXY_PAD_PREVIEW_VERSION:
            raise SchemaValidationError(
                "unsupported Order-8 side-proxy preview version: "
                f"{self.version!r}"
            )
        if self.acceptance_eligible:
            raise SchemaValidationError(
                "Order-8 side-proxy preview must remain acceptance-ineligible"
            )
        if self.contact_runtime_enabled and not self.visual_approval_recorded:
            raise SchemaValidationError(
                "Order-8 side-proxy contact runtime requires recorded visual approval"
            )
        for label, value in (
            ("thickness_m", self.thickness_m),
            ("mesh_clearance_m", self.mesh_clearance_m),
            ("outer_envelope_depth_m", self.outer_envelope_depth_m),
            ("local_plane_tolerance_m", self.local_plane_tolerance_m),
            ("tile_size_min_m", self.tile_size_min_m),
            ("tile_size_max_m", self.tile_size_max_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(f"{label} must be finite and positive")
        if self.axial_band_count < 2:
            raise SchemaValidationError("axial_band_count must be at least 2")
        if self.circumferential_segment_count < 8:
            raise SchemaValidationError(
                "circumferential_segment_count must be at least 8"
            )
        if (
            not math.isfinite(self.cone_axial_min_m)
            or not math.isfinite(self.cone_axial_max_m)
            or self.cone_axial_min_m >= self.cone_axial_max_m
        ):
            raise SchemaValidationError(
                "cone axial bounds must be finite and strictly increasing"
            )
        if not (
            0.0 < self.cone_normal_axial_min
            < self.cone_normal_axial_max
            < 1.0
        ):
            raise SchemaValidationError(
                "cone normal axial bounds must be strictly increasing in (0, 1)"
            )
        for label, value in (
            ("side_min_radial_alignment", self.side_min_radial_alignment),
            ("side_max_axial_normal_abs", self.side_max_axial_normal_abs),
            ("normal_similarity_cosine", self.normal_similarity_cosine),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise SchemaValidationError(f"{label} must be in (0, 1)")
        if (
            not math.isfinite(self.tile_coverage_scale)
            or not 1.0 <= self.tile_coverage_scale <= 1.5
        ):
            raise SchemaValidationError("tile_coverage_scale must be in [1, 1.5]")
        if self.tile_size_min_m > self.tile_size_max_m:
            raise SchemaValidationError(
                "tile_size_min_m must not exceed tile_size_max_m"
            )
        if len(self.display_color_rgb) != 3 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.display_color_rgb
        ):
            raise SchemaValidationError(
                "display_color_rgb must contain three finite values in [0, 1]"
            )
        if not math.isfinite(self.display_opacity) or not (
            0.0 < self.display_opacity <= 1.0
        ):
            raise SchemaValidationError("display_opacity must be in (0, 1]")
        if not self.mesh_search_paths:
            raise SchemaValidationError("mesh_search_paths must not be empty")
        if self.link_ids != _TARGET_LINK_IDS:
            raise SchemaValidationError(
                "side-proxy preview links must be yaw_dock_mech1 and "
                "yaw_dock_mech2 in canonical order"
            )


@dataclass(frozen=True)
class DockSideProxyPadSpec:
    pad_id: str
    link_id: str
    geometry_refs: tuple[str, ...]
    axial_band_index: int
    circumferential_segment_index: int
    center_local: tuple[float, float, float]
    representative_surface_point_local: tuple[float, float, float]
    orientation_local_xyzw: tuple[float, float, float, float]
    size_m: tuple[float, float, float]
    outward_normal_local: tuple[float, float, float]
    axial_axis_local: tuple[float, float, float]
    circumferential_axis_local: tuple[float, float, float]
    inner_face_surface_gap_m: float
    surface_fit_max_gap_m: float
    candidate_triangle_count: int
    surface_triangle_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_order8_side_proxy_pad_preview_config(
    path: str | Path,
) -> Order8SideProxyPadPreviewConfig:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml(config_path)
    if not isinstance(raw, dict):
        raise SchemaValidationError("side-proxy preview config must be a map")
    mesh_search_paths_raw = raw.get("mesh_search_paths")
    if not isinstance(mesh_search_paths_raw, list) or not all(
        isinstance(value, str) and value for value in mesh_search_paths_raw
    ):
        raise SchemaValidationError("mesh_search_paths must be a non-empty string list")
    link_ids_raw = raw.get("link_ids")
    if not isinstance(link_ids_raw, list) or not all(
        isinstance(value, str) and value for value in link_ids_raw
    ):
        raise SchemaValidationError("link_ids must be a non-empty string list")
    config = Order8SideProxyPadPreviewConfig(
        version=str(raw.get("version", "")),
        acceptance_eligible=_strict_bool(
            raw.get("acceptance_eligible"), label="acceptance_eligible"
        ),
        visual_approval_recorded=_strict_bool(
            raw.get("visual_approval_recorded"),
            label="visual_approval_recorded",
        ),
        contact_runtime_enabled=_strict_bool(
            raw.get("contact_runtime_enabled"), label="contact_runtime_enabled"
        ),
        thickness_m=_finite_float(raw.get("thickness_m"), label="thickness_m"),
        mesh_clearance_m=_finite_float(
            raw.get("mesh_clearance_m"), label="mesh_clearance_m"
        ),
        axial_band_count=_strict_int(
            raw.get("axial_band_count"), label="axial_band_count"
        ),
        circumferential_segment_count=_strict_int(
            raw.get("circumferential_segment_count"),
            label="circumferential_segment_count",
        ),
        cone_axial_min_m=_finite_float(
            raw.get("cone_axial_min_m"), label="cone_axial_min_m"
        ),
        cone_axial_max_m=_finite_float(
            raw.get("cone_axial_max_m"), label="cone_axial_max_m"
        ),
        cone_normal_axial_min=_finite_float(
            raw.get("cone_normal_axial_min"), label="cone_normal_axial_min"
        ),
        cone_normal_axial_max=_finite_float(
            raw.get("cone_normal_axial_max"), label="cone_normal_axial_max"
        ),
        side_min_radial_alignment=_finite_float(
            raw.get("side_min_radial_alignment"),
            label="side_min_radial_alignment",
        ),
        side_max_axial_normal_abs=_finite_float(
            raw.get("side_max_axial_normal_abs"),
            label="side_max_axial_normal_abs",
        ),
        outer_envelope_depth_m=_finite_float(
            raw.get("outer_envelope_depth_m"), label="outer_envelope_depth_m"
        ),
        local_plane_tolerance_m=_finite_float(
            raw.get("local_plane_tolerance_m"),
            label="local_plane_tolerance_m",
        ),
        normal_similarity_cosine=_finite_float(
            raw.get("normal_similarity_cosine"),
            label="normal_similarity_cosine",
        ),
        tile_coverage_scale=_finite_float(
            raw.get("tile_coverage_scale"), label="tile_coverage_scale"
        ),
        tile_size_min_m=_finite_float(
            raw.get("tile_size_min_m"), label="tile_size_min_m"
        ),
        tile_size_max_m=_finite_float(
            raw.get("tile_size_max_m"), label="tile_size_max_m"
        ),
        display_color_rgb=_vector3(
            raw.get("display_color_rgb"), label="display_color_rgb"
        ),
        display_opacity=_finite_float(
            raw.get("display_opacity"), label="display_opacity"
        ),
        mesh_search_paths=tuple(
            str(
                (
                    Path(value).expanduser()
                    if Path(value).expanduser().is_absolute()
                    else config_path.parent / Path(value).expanduser()
                ).resolve()
            )
            for value in mesh_search_paths_raw
        ),
        link_ids=tuple(link_ids_raw),
    )
    config.validate()
    return config


def build_order8_side_proxy_pad_specs(
    *,
    urdf_path: str | Path,
    config: Order8SideProxyPadPreviewConfig,
) -> tuple[DockSideProxyPadSpec, ...]:
    """Build small surface-following pads for both selected Dock links."""

    config.validate()
    source_path = Path(urdf_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Holon URDF was not found: {source_path}")
    root = ET.parse(source_path).getroot()
    links = {str(link.attrib.get("name", "")): link for link in root.findall("link")}
    specs: list[DockSideProxyPadSpec] = []
    for link_id in config.link_ids:
        link = links.get(link_id)
        if link is None:
            raise SchemaValidationError(
                f"Holon URDF has no side-proxy link {link_id!r}"
            )
        triangles, geometry_refs = _collision_mesh_triangles_link_local(
            link=link,
            urdf_path=source_path,
            mesh_search_paths=tuple(Path(value) for value in config.mesh_search_paths),
        )
        specs.extend(
            _surface_following_pad_specs(
                link_id=link_id,
                triangles_local=triangles,
                geometry_refs=geometry_refs,
                config=config,
            )
        )
    return tuple(
        sorted(
            specs,
            key=lambda spec: (
                spec.link_id,
                spec.axial_band_index,
                spec.circumferential_segment_index,
            ),
        )
    )


def _surface_following_pad_specs(
    *,
    link_id: str,
    triangles_local: np.ndarray,
    geometry_refs: tuple[str, ...],
    config: Order8SideProxyPadPreviewConfig,
) -> list[DockSideProxyPadSpec]:
    if (
        triangles_local.ndim != 3
        or triangles_local.shape[1:] != (3, 3)
        or len(triangles_local) < 4
        or not np.isfinite(triangles_local).all()
    ):
        raise SchemaValidationError(
            f"{link_id} collision mesh has invalid triangle samples"
        )

    edge_1 = triangles_local[:, 1] - triangles_local[:, 0]
    edge_2 = triangles_local[:, 2] - triangles_local[:, 0]
    cross = np.cross(edge_1, edge_2)
    cross_norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * cross_norm
    normals = cross / np.maximum(cross_norm[:, None], 1.0e-15)
    centroids = np.mean(triangles_local, axis=1)

    radial_vectors = np.array(centroids, copy=True)
    radial_vectors[:, 0] = 0.0
    radii = np.linalg.norm(radial_vectors, axis=1)
    radial_units = radial_vectors / np.maximum(radii[:, None], 1.0e-15)
    inward = np.sum(normals * radial_units, axis=1) < 0.0
    normals[inward] *= -1.0
    radial_alignment = np.sum(normals * radial_units, axis=1)

    mesh_axial_min = float(np.min(triangles_local[:, :, 0]))
    mesh_axial_max = float(np.max(triangles_local[:, :, 0]))
    axial_min = config.cone_axial_min_m
    axial_max = config.cone_axial_max_m
    if axial_min < mesh_axial_min or axial_max > mesh_axial_max:
        raise SchemaValidationError(
            f"{link_id} configured cone axial range is outside the collision mesh"
        )
    eligible = (
        (areas > 1.0e-10)
        & (radii > 0.004)
        & (radial_alignment >= config.side_min_radial_alignment)
        & (np.abs(normals[:, 0]) <= config.side_max_axial_normal_abs)
        & (normals[:, 0] >= config.cone_normal_axial_min)
        & (normals[:, 0] <= config.cone_normal_axial_max)
        & (centroids[:, 0] >= axial_min)
        & (centroids[:, 0] <= axial_max)
    )
    if not np.any(eligible):
        raise SchemaValidationError(
            f"{link_id} has no eligible lateral collision triangles"
        )

    triangle_angles = np.mod(
        np.arctan2(centroids[:, 2], centroids[:, 1]),
        2.0 * math.pi,
    )
    axial_edges = np.linspace(
        axial_min,
        axial_max,
        config.axial_band_count + 1,
    )
    angular_step = 2.0 * math.pi / config.circumferential_segment_count
    specs: list[DockSideProxyPadSpec] = []
    for axial_index in range(config.axial_band_count):
        lower = float(axial_edges[axial_index])
        upper = float(axial_edges[axial_index + 1])
        axial_midpoint = 0.5 * (lower + upper)
        axial_mask = (centroids[:, 0] >= lower) & (centroids[:, 0] <= upper)
        for segment_index in range(config.circumferential_segment_count):
            angular_midpoint = (segment_index + 0.5) * angular_step
            angular_distance = np.abs(
                _wrapped_angle_delta(triangle_angles, angular_midpoint)
            )
            candidate_indices = np.flatnonzero(
                eligible & axial_mask & (angular_distance <= 0.5 * angular_step)
            )
            if len(candidate_indices) == 0:
                # An empty cell means the authored collision mesh has no
                # eligible conical surface there.
                continue
            specs.append(
                _fit_surface_cell_pad(
                    link_id=link_id,
                    triangles_local=triangles_local,
                    centroids=centroids,
                    normals=normals,
                    areas=areas,
                    radial_units=radial_units,
                    triangle_angles=triangle_angles,
                    candidate_indices=candidate_indices,
                    geometry_refs=geometry_refs,
                    axial_band_index=axial_index,
                    segment_index=segment_index,
                    axial_midpoint=axial_midpoint,
                    axial_cell_width=upper - lower,
                    angular_midpoint=angular_midpoint,
                    angular_step=angular_step,
                    config=config,
                )
            )
    if not specs:
        raise SchemaValidationError(f"{link_id} produced no side-proxy micro-pads")
    return specs


def _fit_surface_cell_pad(
    *,
    link_id: str,
    triangles_local: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    radial_units: np.ndarray,
    triangle_angles: np.ndarray,
    candidate_indices: np.ndarray,
    geometry_refs: tuple[str, ...],
    axial_band_index: int,
    segment_index: int,
    axial_midpoint: float,
    axial_cell_width: float,
    angular_midpoint: float,
    angular_step: float,
    config: Order8SideProxyPadPreviewConfig,
) -> DockSideProxyPadSpec:
    radial_direction = np.asarray(
        (0.0, math.cos(angular_midpoint), math.sin(angular_midpoint)),
        dtype=float,
    )
    candidate_support = np.max(
        triangles_local[candidate_indices] @ radial_direction,
        axis=1,
    )
    maximum_support = float(np.max(candidate_support))
    envelope_mask = (
        candidate_support >= maximum_support - config.outer_envelope_depth_m
    )
    envelope_indices = candidate_indices[envelope_mask]
    envelope_areas = areas[envelope_indices]
    area_scale = max(float(np.max(envelope_areas)), 1.0e-15)
    axial_distance = np.abs(centroids[envelope_indices, 0] - axial_midpoint)
    angular_distance = np.abs(
        _wrapped_angle_delta(
            triangle_angles[envelope_indices],
            angular_midpoint,
        )
    )
    # Prefer a real, reasonably sized outer triangle near the cell centre;
    # every candidate remains inside the configured outer-envelope depth.
    seed_score = (
        envelope_areas / area_scale
        - 0.35 * axial_distance / max(axial_cell_width, 1.0e-12)
        - 0.25 * angular_distance / max(0.5 * angular_step, 1.0e-12)
    )
    seed_index = int(envelope_indices[int(np.argmax(seed_score))])
    seed_normal = normals[seed_index]
    normal_similarity = normals[candidate_indices] @ seed_normal
    plane_distance = (
        centroids[candidate_indices] - centroids[seed_index]
    ) @ seed_normal
    local_support_mask = (
        candidate_support
        >= maximum_support - 2.0 * config.outer_envelope_depth_m
    )
    local_mask = (
        (normal_similarity >= config.normal_similarity_cosine)
        & (np.abs(plane_distance) <= config.local_plane_tolerance_m)
        & local_support_mask
    )
    surface_indices = candidate_indices[local_mask]
    if len(surface_indices) == 0:
        surface_indices = np.asarray((seed_index,), dtype=int)

    weights = areas[surface_indices]
    fitted_normal = np.average(normals[surface_indices], axis=0, weights=weights)
    fitted_normal /= np.linalg.norm(fitted_normal)
    if float(fitted_normal @ radial_units[seed_index]) < 0.0:
        fitted_normal *= -1.0

    fitted_centroid = np.average(
        centroids[surface_indices],
        axis=0,
        weights=weights,
    )
    local_vertices = triangles_local[surface_indices].reshape(-1, 3)
    surface_projection = float(np.max(local_vertices @ fitted_normal))
    representative_point = fitted_centroid + fitted_normal * (
        surface_projection - float(fitted_centroid @ fitted_normal)
    )
    surface_fit_max_gap = surface_projection - float(
        np.min(local_vertices @ fitted_normal)
    )
    center = representative_point + fitted_normal * (
        config.mesh_clearance_m + 0.5 * config.thickness_m
    )

    axial_axis = _LINK_AXIAL_AXIS - fitted_normal * float(
        _LINK_AXIAL_AXIS @ fitted_normal
    )
    axial_axis /= np.linalg.norm(axial_axis)
    circumferential_axis = np.cross(fitted_normal, axial_axis)
    circumferential_axis /= np.linalg.norm(circumferential_axis)
    # axial x circumferential = outward normal by construction.
    rotation = np.column_stack(
        (axial_axis, circumferential_axis, fitted_normal)
    )
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8):
        raise SchemaValidationError(
            f"{link_id} micro-pad orientation is not right-handed"
        )

    required_axial_length = (
        config.tile_coverage_scale
        * axial_cell_width
        / max(abs(float(axial_axis[0])), 0.5)
    )
    local_radius = max(float(np.linalg.norm(representative_point[1:])), 0.01)
    required_circumferential_width = (
        config.tile_coverage_scale * local_radius * angular_step
    )
    size = (
        min(
            config.tile_size_max_m,
            max(config.tile_size_min_m, required_axial_length),
        ),
        min(
            config.tile_size_max_m,
            max(config.tile_size_min_m, required_circumferential_width),
        ),
        config.thickness_m,
    )
    return DockSideProxyPadSpec(
        pad_id=(
            f"{link_id}_side_micro_pad_"
            f"a{axial_band_index:02d}_c{segment_index:02d}"
        ),
        link_id=link_id,
        geometry_refs=geometry_refs,
        axial_band_index=axial_band_index,
        circumferential_segment_index=segment_index,
        center_local=tuple(float(value) for value in center),
        representative_surface_point_local=tuple(
            float(value) for value in representative_point
        ),
        orientation_local_xyzw=_quaternion_xyzw_from_rotation(rotation),
        size_m=tuple(float(value) for value in size),
        outward_normal_local=tuple(float(value) for value in fitted_normal),
        axial_axis_local=tuple(float(value) for value in axial_axis),
        circumferential_axis_local=tuple(
            float(value) for value in circumferential_axis
        ),
        inner_face_surface_gap_m=config.mesh_clearance_m,
        surface_fit_max_gap_m=float(surface_fit_max_gap),
        candidate_triangle_count=int(len(candidate_indices)),
        surface_triangle_count=int(len(surface_indices)),
    )


def _collision_mesh_triangles_link_local(
    *,
    link: ET.Element,
    urdf_path: Path,
    mesh_search_paths: tuple[Path, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    transformed: list[np.ndarray] = []
    geometry_refs: list[str] = []
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None or not mesh.attrib.get("filename"):
            continue
        geometry_ref = str(mesh.attrib["filename"])
        mesh_path = _resolve_urdf_mesh_path(
            geometry_ref,
            urdf_path.parent,
            mesh_search_paths,
        )
        scale = _space_vector3(
            mesh.attrib.get("scale", "1 1 1"),
            label=f"{link.attrib.get('name', '')} collision mesh scale",
        )
        origin = collision.find("origin")
        xyz = _space_vector3(
            origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            label=f"{link.attrib.get('name', '')} collision origin xyz",
        )
        rpy = _space_vector3(
            origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            label=f"{link.attrib.get('name', '')} collision origin rpy",
        )
        triangles = _stl_triangles(mesh_path) * np.asarray(scale, dtype=float)
        rotation = _rotation_matrix_from_rpy(rpy)
        transformed.append(
            triangles @ rotation.T + np.asarray(xyz, dtype=float)
        )
        geometry_refs.append(geometry_ref)
    if not transformed:
        raise SchemaValidationError(
            f"link {link.attrib.get('name', '')!r} has no mesh collision geometry"
        )
    return np.concatenate(transformed, axis=0), tuple(sorted(geometry_refs))


def _resolve_urdf_mesh_path(
    reference: str,
    urdf_directory: Path,
    search_directories: tuple[Path, ...],
) -> Path:
    value = str(reference)
    if value.startswith("file://"):
        value = value[len("file://") :]
    if value.startswith("package://"):
        raise SchemaValidationError(
            f"package:// mesh references are unsupported in side-proxy preview: {reference}"
        )
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [urdf_directory / path]
    if not path.is_absolute():
        for directory in search_directories:
            candidates.extend((directory / path, directory / path.name))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "collision mesh was not found; searched: "
        + ", ".join(str(candidate.resolve()) for candidate in candidates)
    )


def _stl_triangles(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected_size = 84 + 50 * int(triangle_count)
        if expected_size == len(payload):
            triangles = np.empty((triangle_count, 3, 3), dtype=float)
            offset = 84
            for triangle_index in range(triangle_count):
                values = struct.unpack_from("<12fH", payload, offset)
                offset += 50
                triangles[triangle_index] = np.asarray(
                    values[3:12], dtype=float
                ).reshape(3, 3)
            return triangles
    vertices: list[tuple[float, float, float]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaValidationError(f"invalid STL encoding: {path}") from exc
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in fields[1:4]))
    if not vertices or len(vertices) % 3 != 0:
        raise SchemaValidationError(f"STL contains no complete triangles: {path}")
    return np.asarray(vertices, dtype=float).reshape(-1, 3, 3)


def _rotation_matrix_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def _wrapped_angle_delta(values: np.ndarray, reference: float) -> np.ndarray:
    return (values - reference + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_xyzw_from_rotation(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal_index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray((qx, qy, qz, qw), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise SchemaValidationError(f"{label} must be bool")
    return bool(value)


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{label} must be int")
    return int(value)


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{label} must be finite")
    return result


def _vector3(value: object, *, label: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        values: Iterable[object] = value.split()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise SchemaValidationError(f"{label} must contain three values")
    result = tuple(_finite_float(item, label=label) for item in values)
    if len(result) != 3:
        raise SchemaValidationError(f"{label} must contain exactly three values")
    return result


def _space_vector3(value: str, *, label: str) -> tuple[float, float, float]:
    fields = value.split()
    if len(fields) != 3:
        raise SchemaValidationError(f"{label} must contain exactly three values")
    try:
        result = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise SchemaValidationError(f"{label} must be numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise SchemaValidationError(f"{label} must be finite")
    return result
