from __future__ import annotations

from pathlib import Path

from amsrr.geometry.asset_resolver import resolve_geometry_asset
from amsrr.geometry.contact_region_extractor import extract_mesh_geometry, extract_primitive_geometry
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.geometry import GeometryDescriptor
from amsrr.schemas.task_spec import GeometrySpec, GeometryType


class GeometryProcessor:
    """Deterministic geometry compiler for P0 primitives and mesh smoke."""

    def __init__(self, *, project_root: str | Path | None = None, search_roots: list[str | Path] | None = None) -> None:
        self.project_root = Path(project_root) if project_root is not None else None
        self.search_roots = [Path(item) for item in (search_roots or [])]

    def process_geometry(
        self,
        geometry_spec: GeometrySpec,
        *,
        entity_id: str | None = None,
        friction: float | None = None,
        contact_allowed: bool = True,
        allowed_contact_modes: list[ContactMode] | None = None,
    ) -> GeometryDescriptor:
        asset_ref = resolve_geometry_asset(
            geometry_spec,
            project_root=self.project_root,
            search_roots=self.search_roots,
        )
        if geometry_spec.geometry_type in {
            GeometryType.BOX,
            GeometryType.SPHERE,
            GeometryType.CYLINDER,
            GeometryType.CAPSULE,
        }:
            extraction = extract_primitive_geometry(
                geometry_spec,
                entity_id=entity_id,
                friction=friction,
                contact_allowed=contact_allowed,
                allowed_contact_modes=allowed_contact_modes,
            )
        elif geometry_spec.geometry_type == GeometryType.MESH:
            extraction = extract_mesh_geometry(
                geometry_spec,
                asset_ref,
                entity_id=entity_id,
                friction=friction,
                contact_allowed=contact_allowed,
                allowed_contact_modes=allowed_contact_modes,
            )
        else:
            raise SchemaValidationError(f"Geometry type {geometry_spec.geometry_type.value!r} is not implemented in P0")

        return GeometryDescriptor(
            geometry_id=geometry_spec.geometry_id,
            global_shape_features=extraction.global_shape_features,
            surface_patch_graph=extraction.surface_patch_graph,
            contact_region_graph=extraction.contact_region_graph,
            collision_ref=asset_ref.collision_ref,
            exact_geometry_ref=asset_ref.exact_geometry_ref,
        )

