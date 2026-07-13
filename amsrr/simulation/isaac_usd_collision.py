from __future__ import annotations

from pathlib import Path
from typing import Any


ISAAC_COLLISION_TYPE_TO_USD_TOKEN = {
    "Convex Hull": "convexHull",
    "Convex Decomposition": "convexDecomposition",
    "Bounding Sphere": "boundingSphere",
    "Bounding Cube": "boundingCube",
}

# Match the installed Isaac physics utility's high-fidelity mesh-collider
# recipe.  Merely authoring the USD ``convexDecomposition`` token leaves the
# PhysX decomposition API/settings implicit and did not preserve the Holon
# pitch-funnel insertion path in the first physical attach attempt.
DEFAULT_DOCK_CONVEX_DECOMPOSITION_MAX_HULLS = 128
DEFAULT_DOCK_CONVEX_DECOMPOSITION_SHRINK_WRAP = True


def isaac_collision_type_to_usd_token(collision_type: str) -> str:
    """Map the Isaac URDF-importer label to the authored USD token."""

    try:
        return ISAAC_COLLISION_TYPE_TO_USD_TOKEN[collision_type]
    except KeyError as exc:
        supported = ", ".join(sorted(ISAAC_COLLISION_TYPE_TO_USD_TOKEN))
        raise ValueError(
            f"unsupported Isaac collision type {collision_type!r}; expected one of {supported}"
        ) from exc


def is_holon_dock_collision_path(prim_path: str) -> bool:
    """Return whether a USD prim path belongs to a Holon Dock-mechanism mesh."""

    return any(
        component.startswith(("pitch_dock_mech", "yaw_dock_mech"))
        for component in str(prim_path).split("/")
        if component
    )


def enforce_holon_dock_mesh_collision_approximation(
    usd_path: str | Path,
    *,
    collision_type: str,
    max_convex_hulls: int = DEFAULT_DOCK_CONVEX_DECOMPOSITION_MAX_HULLS,
    shrink_wrap: bool = DEFAULT_DOCK_CONVEX_DECOMPOSITION_SHRINK_WRAP,
) -> dict[str, Any]:
    """Author and verify the requested approximation on every Holon Dock collider.

    Isaac Sim's URDF importer 3.0 currently authors ``convexHull`` for explicit
    URDF ``<collision>`` meshes.  Its ``collision_type`` option is applied only
    when collision geometry is synthesized from visuals.  Holon intentionally
    supplies collision meshes, so dynamic assembly must repair the generated USD
    package before it is referenced into the simulation stage.

    The asset-transformer output keeps collision API opinions in a referenced
    ``instances`` layer.  Editing the composed instance proxies is not legal USD;
    this function therefore locates the owning used layer, authors the requested
    token there, saves it, then reopens the root asset and verifies every composed
    Dock collision instance with instance-proxy traversal.

    ``pxr`` is imported lazily so normal non-Isaac unit-test collection remains
    independent of the Isaac environment.
    """

    from pxr import Sdf, Usd, UsdPhysics

    root_path = Path(usd_path).expanduser().resolve()
    if not root_path.is_file():
        raise FileNotFoundError(f"generated USD is missing: {root_path}")
    requested_token = isaac_collision_type_to_usd_token(collision_type)
    if requested_token == "convexDecomposition":
        if not 1 <= int(max_convex_hulls) <= 2048:
            raise ValueError("max_convex_hulls must be in [1, 2048]")
        if type(shrink_wrap) is not bool:
            raise TypeError("shrink_wrap must be bool")

    root_stage = Usd.Stage.Open(str(root_path))
    if root_stage is None:
        raise RuntimeError(f"failed to open generated USD stage: {root_path}")

    authored_paths: list[str] = []
    original_tokens: dict[str, str | None] = {}
    editable_layer_ids = []
    for layer in root_stage.GetUsedLayers():
        if bool(getattr(layer, "anonymous", False)):
            continue
        real_path = str(getattr(layer, "realPath", ""))
        if not real_path:
            continue
        resolved_layer_path = Path(real_path).expanduser().resolve()
        if not resolved_layer_path.is_relative_to(root_path.parent):
            continue
        editable_layer_ids.append(str(layer.identifier))
    editable_layer_ids = sorted(set(editable_layer_ids))
    for layer_id in editable_layer_ids:
        layer_stage = Usd.Stage.Open(layer_id)
        if layer_stage is None:
            continue
        layer_changed = False
        for prim in layer_stage.TraverseAll():
            path = str(prim.GetPath())
            if not is_holon_dock_collision_path(path):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                continue
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
            original = approximation.Get()
            evidence_key = f"{layer_id}:{path}"
            original_tokens[evidence_key] = None if original is None else str(original)
            if str(original) != requested_token:
                if not approximation.Set(requested_token):
                    raise RuntimeError(
                        "failed to author Dock mesh collision approximation on "
                        f"{evidence_key}"
                    )
                layer_changed = True
            if requested_token == "convexDecomposition":
                prim.AddAppliedSchema(
                    "PhysxConvexDecompositionCollisionAPI"
                )
                max_hulls_attr = prim.CreateAttribute(
                    "physxConvexDecompositionCollision:maxConvexHulls",
                    Sdf.ValueTypeNames.Int,
                    custom=False,
                )
                shrink_wrap_attr = prim.CreateAttribute(
                    "physxConvexDecompositionCollision:shrinkWrap",
                    Sdf.ValueTypeNames.Bool,
                    custom=False,
                )
                if int(max_hulls_attr.Get() or 0) != int(max_convex_hulls):
                    if not max_hulls_attr.Set(int(max_convex_hulls)):
                        raise RuntimeError(
                            "failed to author maxConvexHulls on "
                            f"{evidence_key}"
                        )
                    layer_changed = True
                if shrink_wrap_attr.Get() is not bool(shrink_wrap):
                    if not shrink_wrap_attr.Set(bool(shrink_wrap)):
                        raise RuntimeError(
                            "failed to author shrinkWrap on "
                            f"{evidence_key}"
                        )
                    layer_changed = True
            authored_paths.append(evidence_key)
        if layer_changed and not layer_stage.GetRootLayer().Save():
            raise RuntimeError(f"failed to save Dock collision USD layer: {layer_id}")

    if not authored_paths:
        raise RuntimeError(
            "generated USD contains no editable Holon Dock mesh collision prims"
        )

    verified_stage = Usd.Stage.Open(str(root_path))
    if verified_stage is None:
        raise RuntimeError(f"failed to reopen generated USD stage: {root_path}")
    composed_paths: list[str] = []
    mismatches: list[dict[str, str | None]] = []
    prim_range = Usd.PrimRange.Stage(
        verified_stage,
        Usd.TraverseInstanceProxies(),
    )
    for prim in prim_range:
        path = str(prim.GetPath())
        if not is_holon_dock_collision_path(path):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        composed_paths.append(path)
        value = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        actual_token = None if value is None else str(value)
        if actual_token != requested_token:
            mismatches.append(
                {
                    "prim_path": path,
                    "actual_token": actual_token,
                }
            )
            continue
        if requested_token == "convexDecomposition":
            if (
                "PhysxConvexDecompositionCollisionAPI"
                not in str(prim.GetMetadata("apiSchemas"))
            ):
                mismatches.append(
                    {
                        "prim_path": path,
                        "actual_token": actual_token,
                        "failure": "missing_physx_convex_decomposition_api",
                    }
                )
                continue
            actual_max_hulls = prim.GetAttribute(
                "physxConvexDecompositionCollision:maxConvexHulls"
            ).Get()
            actual_shrink_wrap = prim.GetAttribute(
                "physxConvexDecompositionCollision:shrinkWrap"
            ).Get()
            if (
                int(actual_max_hulls or 0) != int(max_convex_hulls)
                or actual_shrink_wrap is not bool(shrink_wrap)
            ):
                mismatches.append(
                    {
                        "prim_path": path,
                        "actual_token": actual_token,
                        "actual_max_convex_hulls": actual_max_hulls,
                        "actual_shrink_wrap": actual_shrink_wrap,
                    }
                )

    if not composed_paths:
        raise RuntimeError(
            "generated USD contains no composed Holon Dock mesh collision prims"
        )
    if mismatches:
        raise RuntimeError(
            "generated USD Dock collision approximation verification failed: "
            f"{mismatches}"
        )

    return {
        "requested_collision_type": collision_type,
        "requested_approximation_token": requested_token,
        "physx_convex_decomposition_api_verified": (
            requested_token == "convexDecomposition"
        ),
        "max_convex_hulls": (
            int(max_convex_hulls)
            if requested_token == "convexDecomposition"
            else None
        ),
        "shrink_wrap": (
            bool(shrink_wrap)
            if requested_token == "convexDecomposition"
            else None
        ),
        "authored_prim_count": len(authored_paths),
        "authored_prim_paths": sorted(authored_paths),
        "composed_prim_count": len(composed_paths),
        "composed_prim_paths": sorted(composed_paths),
        "original_approximation_tokens": {
            key: original_tokens[key] for key in sorted(original_tokens)
        },
        "verified": True,
    }
