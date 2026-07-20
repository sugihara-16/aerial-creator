from __future__ import annotations

"""Geometry-derived rigid-body mass properties for task randomization."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.schemas.common import SchemaBase, SchemaValidationError, Vector3, require_len, require_non_empty
from amsrr.schemas.task_spec import GeometrySpec, GeometryType


MASS_PROPERTIES_VERSION = "geometry_mass_properties_v1"


@dataclass
class RigidBodyMassProperties(SchemaBase):
    mass_kg: float
    center_of_mass_object: Vector3
    inertia_kgm2: list[float]
    volume_m3: float
    density_kg_m3: float
    source: str

    def validate(self) -> None:
        for name in ("mass_kg", "volume_m3", "density_kg_m3"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"RigidBodyMassProperties.{name} must be finite and positive"
                )
        require_len(
            self.center_of_mass_object,
            3,
            "RigidBodyMassProperties.center_of_mass_object",
        )
        require_len(self.inertia_kgm2, 6, "RigidBodyMassProperties.inertia_kgm2")
        if not all(math.isfinite(float(value)) for value in self.inertia_kgm2):
            raise SchemaValidationError(
                "RigidBodyMassProperties.inertia_kgm2 must be finite"
            )
        ixx, ixy, ixz, iyy, iyz, izz = self.inertia_kgm2
        if min(ixx, iyy, izz) <= 0.0:
            raise SchemaValidationError(
                "RigidBodyMassProperties diagonal inertia must be positive"
            )
        # Positive definiteness of the symmetric 3x3 tensor by Sylvester's
        # criterion. Products of inertia use the repository's direct matrix
        # entry convention [Ixx, Ixy, Ixz, Iyy, Iyz, Izz].
        minor_2 = ixx * iyy - ixy * ixy
        determinant = (
            ixx * (iyy * izz - iyz * iyz)
            - ixy * (ixy * izz - iyz * ixz)
            + ixz * (ixy * iyz - iyy * ixz)
        )
        if minor_2 <= 0.0 or determinant <= 0.0:
            raise SchemaValidationError(
                "RigidBodyMassProperties inertia tensor must be positive definite"
            )
        require_non_empty(self.source, "RigidBodyMassProperties.source")


def mass_properties_from_geometry(
    geometry: GeometrySpec,
    *,
    density_kg_m3: float,
    project_root: str | Path | None = None,
) -> RigidBodyMassProperties:
    density = _positive(density_kg_m3, "density_kg_m3")
    params = geometry.primitive_params or {}
    scale = tuple(float(value) for value in geometry.scale)
    if geometry.geometry_type == GeometryType.BOX:
        size = _scaled_vector(params.get("size_m"), scale, "box.size_m")
        return cuboid_mass_properties(size, density_kg_m3=density)
    if geometry.geometry_type == GeometryType.SPHERE:
        radius = _positive(params.get("radius_m"), "sphere.radius_m")
        if not math.isclose(scale[0], scale[1]) or not math.isclose(scale[1], scale[2]):
            return _trimesh_primitive_mass_properties(
                "sphere",
                params,
                scale,
                density,
            )
        return sphere_mass_properties(radius * scale[0], density_kg_m3=density)
    if geometry.geometry_type == GeometryType.CYLINDER:
        radius = _positive(params.get("radius_m"), "cylinder.radius_m")
        height = _positive(params.get("height_m"), "cylinder.height_m")
        if not math.isclose(scale[0], scale[1]):
            return _trimesh_primitive_mass_properties(
                "cylinder",
                params,
                scale,
                density,
            )
        return cylinder_mass_properties(
            radius * scale[0],
            height * scale[2],
            density_kg_m3=density,
        )
    if geometry.geometry_type == GeometryType.CAPSULE:
        return _trimesh_primitive_mass_properties(
            "capsule",
            params,
            scale,
            density,
        )
    if geometry.geometry_type == GeometryType.MESH:
        if not geometry.asset_path:
            raise SchemaValidationError("mesh mass properties require asset_path")
        path = Path(geometry.asset_path)
        if not path.is_absolute() and project_root is not None:
            path = Path(project_root) / path
        return mesh_mass_properties(path, scale=scale, density_kg_m3=density)
    raise SchemaValidationError(
        f"mass properties are not implemented for {geometry.geometry_type.value!r}"
    )


def cuboid_mass_properties(
    size_m: Sequence[float],
    *,
    density_kg_m3: float,
) -> RigidBodyMassProperties:
    x, y, z = _vector3_positive(size_m, "size_m")
    density = _positive(density_kg_m3, "density_kg_m3")
    volume = x * y * z
    mass = density * volume
    return RigidBodyMassProperties(
        mass_kg=mass,
        center_of_mass_object=(0.0, 0.0, 0.0),
        inertia_kgm2=[
            mass * (y * y + z * z) / 12.0,
            0.0,
            0.0,
            mass * (x * x + z * z) / 12.0,
            0.0,
            mass * (x * x + y * y) / 12.0,
        ],
        volume_m3=volume,
        density_kg_m3=density,
        source=f"{MASS_PROPERTIES_VERSION}:analytic_cuboid",
    )


def sphere_mass_properties(
    radius_m: float,
    *,
    density_kg_m3: float,
) -> RigidBodyMassProperties:
    radius = _positive(radius_m, "radius_m")
    density = _positive(density_kg_m3, "density_kg_m3")
    volume = 4.0 * math.pi * radius**3 / 3.0
    mass = density * volume
    inertia = 2.0 * mass * radius * radius / 5.0
    return RigidBodyMassProperties(
        mass_kg=mass,
        center_of_mass_object=(0.0, 0.0, 0.0),
        inertia_kgm2=[inertia, 0.0, 0.0, inertia, 0.0, inertia],
        volume_m3=volume,
        density_kg_m3=density,
        source=f"{MASS_PROPERTIES_VERSION}:analytic_sphere",
    )


def cylinder_mass_properties(
    radius_m: float,
    height_m: float,
    *,
    density_kg_m3: float,
) -> RigidBodyMassProperties:
    radius = _positive(radius_m, "radius_m")
    height = _positive(height_m, "height_m")
    density = _positive(density_kg_m3, "density_kg_m3")
    volume = math.pi * radius * radius * height
    mass = density * volume
    transverse = mass * (3.0 * radius * radius + height * height) / 12.0
    axial = mass * radius * radius / 2.0
    return RigidBodyMassProperties(
        mass_kg=mass,
        center_of_mass_object=(0.0, 0.0, 0.0),
        inertia_kgm2=[transverse, 0.0, 0.0, transverse, 0.0, axial],
        volume_m3=volume,
        density_kg_m3=density,
        source=f"{MASS_PROPERTIES_VERSION}:analytic_cylinder_z",
    )


def mesh_mass_properties(
    asset_path: str | Path,
    *,
    scale: Sequence[float] = (1.0, 1.0, 1.0),
    density_kg_m3: float,
) -> RigidBodyMassProperties:
    path = Path(asset_path)
    if not path.is_file():
        raise SchemaValidationError(f"mesh mass-properties asset does not exist: {path}")
    trimesh = _import_trimesh()
    loaded = trimesh.load_mesh(path, process=True)
    if hasattr(loaded, "dump") and not hasattr(loaded, "vertices"):
        loaded = loaded.dump(concatenate=True)
    if not getattr(loaded, "is_watertight", False):
        raise SchemaValidationError(
            f"mesh mass properties require a watertight mesh: {path}"
        )
    mesh = loaded.copy()
    mesh.apply_scale(_vector3_positive(scale, "scale"))
    mesh.density = _positive(density_kg_m3, "density_kg_m3")
    return _from_trimesh_mass_properties(
        mesh.mass_properties,
        source=f"{MASS_PROPERTIES_VERSION}:trimesh:{path}",
    )


def _trimesh_primitive_mass_properties(
    primitive: str,
    params: Mapping[str, Any],
    scale: Sequence[float],
    density: float,
) -> RigidBodyMassProperties:
    trimesh = _import_trimesh()
    if primitive == "sphere":
        mesh = trimesh.creation.icosphere(
            subdivisions=4,
            radius=_positive(params.get("radius_m"), "sphere.radius_m"),
        )
    elif primitive == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=_positive(params.get("radius_m"), "cylinder.radius_m"),
            height=_positive(params.get("height_m"), "cylinder.height_m"),
            sections=128,
        )
    elif primitive == "capsule":
        mesh = trimesh.creation.capsule(
            radius=_positive(params.get("radius_m"), "capsule.radius_m"),
            height=_positive(params.get("height_m"), "capsule.height_m"),
            count=[32, 64],
        )
    else:  # pragma: no cover - internal call contract.
        raise AssertionError(primitive)
    mesh.apply_scale(_vector3_positive(scale, "scale"))
    mesh.density = density
    return _from_trimesh_mass_properties(
        mesh.mass_properties,
        source=f"{MASS_PROPERTIES_VERSION}:trimesh_{primitive}",
    )


def _from_trimesh_mass_properties(properties: Any, *, source: str) -> RigidBodyMassProperties:
    inertia = properties.inertia
    return RigidBodyMassProperties(
        mass_kg=float(properties.mass),
        center_of_mass_object=tuple(
            float(value) for value in properties.center_mass
        ),  # type: ignore[arg-type]
        inertia_kgm2=[
            float(inertia[0][0]),
            float(inertia[0][1]),
            float(inertia[0][2]),
            float(inertia[1][1]),
            float(inertia[1][2]),
            float(inertia[2][2]),
        ],
        volume_m3=float(properties.volume),
        density_kg_m3=float(properties.density),
        source=source,
    )


def _import_trimesh():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - environment contract.
        raise RuntimeError(
            "trimesh is required for mesh, capsule, or anisotropically-scaled primitive mass properties"
        ) from exc
    return trimesh


def _scaled_vector(
    values: object,
    scale: Sequence[float],
    path: str,
) -> tuple[float, float, float]:
    vector = _vector3_positive(values, path)
    scale_vector = _vector3_positive(scale, "scale")
    return tuple(
        value * factor for value, factor in zip(vector, scale_vector)
    )  # type: ignore[return-value]


def _vector3_positive(values: object, path: str) -> tuple[float, float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SchemaValidationError(f"{path} must be a three-vector")
    require_len(values, 3, path)
    parsed = tuple(_positive(value, f"{path}[]") for value in values)
    return parsed  # type: ignore[return-value]


def _positive(value: object, path: str) -> float:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{path} must be finite and positive")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{path} must be finite and positive") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise SchemaValidationError(f"{path} must be finite and positive")
    return parsed
