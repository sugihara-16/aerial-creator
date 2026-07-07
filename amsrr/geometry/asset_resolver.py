from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.task_spec import GeometrySpec, GeometryType
from amsrr.utils.hashing import hash_file, stable_hash


@dataclass(frozen=True)
class GeometryAssetReference:
    geometry_id: str
    geometry_type: GeometryType
    asset_path: Path | None
    asset_hash: str
    collision_ref: str
    exact_geometry_ref: str
    metadata: dict[str, Any]


def _resolve_asset_path(path_value: str, *, project_root: Path | None, search_roots: list[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / path)
    candidates.append(Path.cwd() / path)
    candidates.extend(root / path for root in search_roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SchemaValidationError(f"Geometry asset path does not exist: {path_value}")


def resolve_geometry_asset(
    geometry_spec: GeometrySpec,
    *,
    project_root: str | Path | None = None,
    search_roots: list[str | Path] | None = None,
) -> GeometryAssetReference:
    root = Path(project_root) if project_root is not None else None
    roots = [Path(item) for item in (search_roots or [])]
    if geometry_spec.geometry_type in {
        GeometryType.BOX,
        GeometryType.SPHERE,
        GeometryType.CYLINDER,
        GeometryType.CAPSULE,
    }:
        digest = stable_hash(
            {
                "geometry_id": geometry_spec.geometry_id,
                "geometry_type": geometry_spec.geometry_type.value,
                "primitive_params": geometry_spec.primitive_params,
                "scale": geometry_spec.scale,
                "collision_model": geometry_spec.collision_model.value,
            }
        )
        return GeometryAssetReference(
            geometry_id=geometry_spec.geometry_id,
            geometry_type=geometry_spec.geometry_type,
            asset_path=None,
            asset_hash=digest,
            collision_ref=f"primitive://sha256:{digest}",
            exact_geometry_ref=f"primitive://sha256:{digest}",
            metadata={"source": "primitive"},
        )

    if geometry_spec.asset_path is None:
        raise SchemaValidationError(f"{geometry_spec.geometry_id} requires asset_path")
    asset_path = _resolve_asset_path(geometry_spec.asset_path, project_root=root, search_roots=roots)
    digest = hash_file(asset_path)
    return GeometryAssetReference(
        geometry_id=geometry_spec.geometry_id,
        geometry_type=geometry_spec.geometry_type,
        asset_path=asset_path,
        asset_hash=digest,
        collision_ref=f"{geometry_spec.collision_model.value}://sha256:{digest}",
        exact_geometry_ref=f"{geometry_spec.geometry_type.value}://sha256:{digest}",
        metadata={"source": "asset", "suffix": asset_path.suffix.lower(), "size_bytes": asset_path.stat().st_size},
    )

