"""Convex-only collision hard gate for the native Order 9 posture IK.

The online IK penalty and final admission use a convex hull of each authored
URDF collision mesh.  Exact FCL BVHs are optional and retained only for
offline comparison.  The proxy-only path loads authored-overlap exclusions
from a hash-bound manifest and fails closed on identity mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from amsrr.geometry.pose_math import rpy_to_matrix
from amsrr.feasibility.articulated_reachability import (
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.physical_model import PhysicalModel

from amsrr.feasibility.order9_batched_kinematics import _pose_arrays
from amsrr.feasibility.order9_native_posture_ik import (
    NativeCentroidalPostureIKSolver,
)


@dataclass(frozen=True)
class CollisionAwareIKConfig:
    collision_margin_m: float = 0.005
    collision_activation_distance_m: float = 0.020
    collision_weight: float = 4.0
    collision_refinement_iterations: int = 20
    collision_line_search_steps: int = 8

    def __post_init__(self) -> None:
        for name in (
            "collision_margin_m",
            "collision_activation_distance_m",
            "collision_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.collision_activation_distance_m < self.collision_margin_m:
            raise ValueError(
                "collision_activation_distance_m must be at least the "
                "collision margin"
            )
        if self.collision_refinement_iterations < 0:
            raise ValueError(
                "collision_refinement_iterations must be non-negative"
            )
        if self.collision_line_search_steps < 1:
            raise ValueError("collision_line_search_steps must be positive")

    def to_native_dict(self) -> dict[str, float | int]:
        return {
            "collision_margin_m": float(self.collision_margin_m),
            "collision_activation_distance_m": float(
                self.collision_activation_distance_m
            ),
            "collision_weight": float(self.collision_weight),
            "collision_refinement_iterations": int(
                self.collision_refinement_iterations
            ),
            "collision_line_search_steps": int(
                self.collision_line_search_steps
            ),
        }


@dataclass(frozen=True)
class CollisionGeometryArrays:
    urdf_sha256: str
    link_ids: tuple[str, ...]
    link_indices: np.ndarray
    local_rotations: np.ndarray
    local_translations: np.ndarray
    proxy_centers: np.ndarray
    proxy_half_extents: np.ndarray
    convex_vertices: tuple[np.ndarray, ...]
    convex_faces: tuple[np.ndarray, ...]
    mesh_paths: tuple[str, ...]
    mesh_digests: tuple[str, ...]
    mesh_scales: np.ndarray


def _vector(
    text: str | None,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if text is None:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"expected three-vector, got {text!r}")
    return values


def _mesh_path(
    reference: str,
    *,
    urdf_path: Path,
    search_directories: Iterable[Path],
) -> Path:
    raw = Path(reference)
    candidates = [raw] if raw.is_absolute() else [urdf_path.parent / raw]
    if not raw.is_absolute():
        for directory in search_directories:
            candidates.extend((directory / raw, directory / raw.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"collision mesh {reference!r} not found; searched "
        + ", ".join(str(value) for value in candidates)
    )


def _binary_stl_bounds(
    path: Path,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"binary STL is truncated: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if 84 + triangle_count * 50 != len(data):
        raise ValueError(
            f"posture collision mesh must be binary STL: {path}"
        )
    records = np.ndarray(
        shape=(triangle_count,),
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ],
            align=False,
        ),
        buffer=data,
        offset=84,
    )
    vertices = np.asarray(records["vertices"], dtype=np.float64)
    vertices *= scale.reshape(1, 1, 3)
    return (
        np.min(vertices, axis=(0, 1)),
        np.max(vertices, axis=(0, 1)),
    )


def _binary_stl_convex_hull(
    path: Path,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    records = np.ndarray(
        shape=(triangle_count,),
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ],
            align=False,
        ),
        buffer=data,
        offset=84,
    )
    points = np.asarray(records["vertices"], dtype=np.float64).reshape(-1, 3)
    points *= scale.reshape(1, 3)
    points = np.unique(points, axis=0)
    hull = ConvexHull(points)
    simplices = np.asarray(hull.simplices, dtype=np.int32).copy()
    for index, simplex in enumerate(simplices):
        first, second, third = points[simplex]
        normal = np.cross(second - first, third - first)
        if float(np.dot(normal, hull.equations[index, :3])) < 0.0:
            simplices[index, 1], simplices[index, 2] = (
                simplices[index, 2],
                simplices[index, 1],
            )
    used = np.unique(simplices)
    remap = np.full(points.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return (
        np.ascontiguousarray(points[used], dtype=np.float64),
        np.ascontiguousarray(remap[simplices], dtype=np.int32),
    )


def build_collision_geometry_arrays(
    physical_model: PhysicalModel,
    *,
    link_index: Mapping[str, int],
    repository_root: Path,
) -> CollisionGeometryArrays:
    urdf_path = Path(physical_model.urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    link_ids: list[str] = []
    link_indices: list[int] = []
    local_rotations: list[np.ndarray] = []
    local_translations: list[np.ndarray] = []
    proxy_centers: list[np.ndarray] = []
    proxy_half_extents: list[np.ndarray] = []
    convex_vertices: list[np.ndarray] = []
    convex_faces: list[np.ndarray] = []
    mesh_paths: list[str] = []
    mesh_digests: list[str] = []
    mesh_scales: list[np.ndarray] = []
    search_directories = (
        repository_root / "module_urdf",
        repository_root / "module_urdf" / "mesh",
    )
    hull_cache: dict[
        tuple[str, tuple[float, float, float]],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    canonical_mesh_by_digest: dict[str, Path] = {}
    for link in root.findall("link"):
        link_id = link.attrib.get("name")
        if link_id not in link_index:
            continue
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = None if geometry is None else geometry.find("mesh")
            if mesh is None or not mesh.attrib.get("filename"):
                raise ValueError(
                    "posture collision currently requires mesh-backed "
                    f"URDF collisions ({link_id!r})"
                )
            origin = collision.find("origin")
            xyz = _vector(
                None if origin is None else origin.attrib.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _vector(
                None if origin is None else origin.attrib.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            scale = np.asarray(
                _vector(mesh.attrib.get("scale"), (1.0, 1.0, 1.0)),
                dtype=np.float64,
            )
            path = _mesh_path(
                mesh.attrib["filename"],
                urdf_path=urdf_path,
                search_directories=search_directories,
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path = canonical_mesh_by_digest.setdefault(digest, path)
            lower, upper = _binary_stl_bounds(path, scale)
            hull_key = (
                str(path),
                tuple(float(value) for value in scale),
            )
            if hull_key not in hull_cache:
                hull_cache[hull_key] = _binary_stl_convex_hull(
                    path, scale
                )
            hull_vertices, hull_faces = hull_cache[hull_key]
            center = 0.5 * (lower + upper)
            link_ids.append(link_id)
            link_indices.append(link_index[link_id])
            local_rotations.append(
                np.asarray(rpy_to_matrix(rpy), dtype=np.float64)
            )
            local_translations.append(np.asarray(xyz, dtype=np.float64))
            proxy_centers.append(center)
            proxy_half_extents.append(0.5 * (upper - lower))
            convex_vertices.append(
                np.ascontiguousarray(
                    hull_vertices - center.reshape(1, 3),
                    dtype=np.float64,
                )
            )
            convex_faces.append(hull_faces)
            mesh_paths.append(str(path))
            mesh_digests.append(digest)
            mesh_scales.append(scale)
    if not link_ids:
        raise ValueError("URDF has no mesh collision geometry")
    return CollisionGeometryArrays(
        urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        link_ids=tuple(link_ids),
        link_indices=np.asarray(link_indices, dtype=np.int32),
        local_rotations=np.asarray(local_rotations, dtype=np.float64),
        local_translations=np.asarray(local_translations, dtype=np.float64),
        proxy_centers=np.asarray(proxy_centers, dtype=np.float64),
        proxy_half_extents=np.asarray(proxy_half_extents, dtype=np.float64),
        convex_vertices=tuple(convex_vertices),
        convex_faces=tuple(convex_faces),
        mesh_paths=tuple(mesh_paths),
        mesh_digests=tuple(mesh_digests),
        mesh_scales=np.asarray(mesh_scales, dtype=np.float64),
    )


def configure_kernel_collision_geometry(
    *,
    kernel,
    physical_model: PhysicalModel,
    link_index: Mapping[str, int],
    repository_root: Path,
    load_exact_geometry: bool,
    nominal_collision_pair_manifest: Path | None,
) -> CollisionGeometryArrays:
    arrays = build_collision_geometry_arrays(
        physical_model,
        link_index=link_index,
        repository_root=repository_root,
    )
    kernel.configure_collision_geometry(
        arrays.link_indices,
        arrays.local_rotations,
        arrays.local_translations,
        arrays.proxy_centers,
        arrays.proxy_half_extents,
        list(arrays.convex_vertices),
        list(arrays.convex_faces),
        list(arrays.mesh_paths),
        arrays.mesh_scales,
        bool(load_exact_geometry),
    )
    if not load_exact_geometry:
        if nominal_collision_pair_manifest is None:
            raise ValueError(
                "proxy-only geometry requires a nominal collision-pair "
                "manifest"
            )
        manifest = json.loads(
            nominal_collision_pair_manifest.read_text(encoding="utf-8")
        )
        if (
            manifest.get("manifest_version")
            != "holon_nominal_collision_pairs_v1"
            or manifest.get("urdf_sha256") != arrays.urdf_sha256
            or tuple(manifest.get("collision_link_ids", ()))
            != arrays.link_ids
            or tuple(manifest.get("mesh_sha256", ()))
            != arrays.mesh_digests
        ):
            raise ValueError(
                "nominal collision-pair manifest does not match the "
                "current URDF collision geometry"
            )
        pairs = np.asarray(
            manifest.get("authored_overlap_geometry_pairs", ()),
            dtype=np.int32,
        ).reshape((-1, 2))
        kernel.configure_nominal_collision_pairs(pairs)
    return arrays


def set_kernel_collision_scene(
    *,
    kernel,
    object_pose_world: Pose7D | None,
    object_size_m: tuple[float, float, float] | None,
    allowed_contact_module_indices: Iterable[int],
    allowed_contact_link_indices: Iterable[int],
) -> None:
    modules = np.asarray(
        tuple(int(value) for value in allowed_contact_module_indices),
        dtype=np.int32,
    )
    links = np.asarray(
        tuple(int(value) for value in allowed_contact_link_indices),
        dtype=np.int32,
    )
    if modules.shape != links.shape:
        raise ValueError("allowed contact module/link arrays must match")
    enabled = object_pose_world is not None and object_size_m is not None
    if enabled:
        object_r, object_p = _pose_arrays(object_pose_world)
        object_size = np.asarray(object_size_m, dtype=np.float64)
    else:
        object_r = np.eye(3, dtype=np.float64)
        object_p = np.zeros(3, dtype=np.float64)
        object_size = np.ones(3, dtype=np.float64)
    kernel.set_collision_scene(
        enabled,
        object_r,
        object_p,
        object_size,
        modules,
        links,
    )


class CollisionAwareNativeCentroidalPostureIKSolver(
    NativeCentroidalPostureIKSolver
):
    """Native IK with convex-hull collision avoidance."""

    solver_version = "centroidal_posture_ik_cpp_eigen_fcl_convex_v1"

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        kinematics,
        config=None,
        collision_config: CollisionAwareIKConfig | None = None,
    ) -> None:
        if not kinematics._enable_collision_geometry:
            raise ValueError(
                "kinematics must enable collision geometry"
            )
        super().__init__(
            physical_model,
            kinematics=kinematics,
            config=config,
            collision_config=collision_config or CollisionAwareIKConfig(),
        )
        self.collision_config = (
            collision_config or CollisionAwareIKConfig()
        )
        self._scene_morphology = None

    def set_collision_scene(
        self,
        *,
        morphology,
        object_pose_world: Pose7D | None,
        object_size_m: tuple[float, float, float] | None,
        allowed_anchor_ids: Iterable[int],
    ) -> None:
        self.kinematics._ensure_graph(morphology)
        references = resolve_mesh_backed_anchor_references(
            morphology,
            self.physical_model,
            tuple(sorted(int(value) for value in allowed_anchor_ids)),
        )
        set_kernel_collision_scene(
            kernel=self.kinematics._cpp_kernel,
            object_pose_world=object_pose_world,
            object_size_m=object_size_m,
            allowed_contact_module_indices=(
                self.kinematics._module_index[
                    reference.anchor.module_id
                ]
                for reference in references
            ),
            allowed_contact_link_indices=(
                self.kinematics._link_index[
                    reference.surface.mechanism_link_id
                ]
                for reference in references
            ),
        )
        self._solution_cache.clear()
        self._scene_morphology = morphology

    def check_configuration(
        self,
        *,
        morphology,
        centroidal_pose_world: Pose7D,
        joint_positions_rad: Mapping[str, float],
        exact: bool,
        margin_m: float,
    ) -> dict[str, object]:
        self.kinematics._ensure_graph(morphology)
        if morphology is not self._scene_morphology:
            raise RuntimeError(
                "collision scene must be installed for this morphology"
            )
        ordered_ids = ordered_global_dock_joint_ids(
            morphology, self.physical_model
        )
        q = np.asarray(
            [joint_positions_rad[joint_id] for joint_id in ordered_ids],
            dtype=np.float64,
        ).reshape(
            len(self.kinematics._module_ids),
            len(self.kinematics._joint_ids),
        )
        centroidal_r, centroidal_p = _pose_arrays(
            centroidal_pose_world
        )
        result = dict(
            # The proxy-only runtime intentionally fails closed if a caller
            # accidentally requests the disabled exact geometry path.
            self.kinematics._cpp_kernel.check_collisions(
                q,
                centroidal_r,
                centroidal_p,
                bool(exact),
                float(margin_m),
            )
        )
        geometry = self.kinematics.collision_geometry
        assert geometry is not None
        geometry_count = len(geometry.link_ids)
        details = []
        for pair in result.get("worst_pairs", []):
            first_instance = int(pair["first_instance"])
            detail = {
                **dict(pair),
                "first_module_index": (
                    first_instance // geometry_count
                ),
                "first_link_id": geometry.link_ids[
                    first_instance % geometry_count
                ],
            }
            if pair["second_kind"] == "robot":
                second_instance = int(pair["second_instance"])
                detail.update(
                    {
                        "second_module_index": (
                            second_instance // geometry_count
                        ),
                        "second_link_id": geometry.link_ids[
                            second_instance % geometry_count
                        ],
                    }
                )
            details.append(detail)
        result["worst_pairs"] = details
        return result
