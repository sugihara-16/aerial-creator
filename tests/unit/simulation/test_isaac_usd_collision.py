from __future__ import annotations

import importlib.util

import pytest

from amsrr.simulation.isaac_usd_collision import (
    enforce_holon_dock_mesh_collision_approximation,
    isaac_collision_type_to_usd_token,
    is_holon_dock_collision_path,
)


def test_isaac_collision_type_to_usd_token() -> None:
    assert (
        isaac_collision_type_to_usd_token("Convex Decomposition")
        == "convexDecomposition"
    )
    assert isaac_collision_type_to_usd_token("Convex Hull") == "convexHull"
    with pytest.raises(ValueError, match="unsupported Isaac collision type"):
        isaac_collision_type_to_usd_token("triangle mesh")


@pytest.mark.parametrize(
    ("prim_path", "expected"),
    [
        ("/Robot/pitch_dock_mech1/collision", True),
        ("/Instances/yaw_dock_mech_1_1/yaw_dock_mech_1", True),
        ("/Robot/rotor_arm_pitch/collision", False),
        ("/Robot/not_a_pitch_dock_mechanism/collision", False),
    ],
)
def test_is_holon_dock_collision_path(prim_path: str, expected: bool) -> None:
    assert is_holon_dock_collision_path(prim_path) is expected


@pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="pxr is available only in the Isaac environment",
)
def test_enforce_holon_dock_mesh_collision_approximation(tmp_path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    instances_path = tmp_path / "instances.usda"
    instances = Usd.Stage.CreateNew(str(instances_path))
    UsdGeom.Scope.Define(instances, "/Instances")
    for dock_name in ("pitch_dock_mech_1", "yaw_dock_mech_1"):
        instance_name = f"{dock_name}_1"
        UsdGeom.Xform.Define(instances, f"/Instances/{instance_name}")
        mesh = UsdGeom.Mesh.Define(
            instances,
            f"/Instances/{instance_name}/{dock_name}",
        )
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
    instances.GetRootLayer().Save()

    root_path = tmp_path / "robot.usda"
    root = Usd.Stage.CreateNew(str(root_path))
    UsdGeom.Xform.Define(root, "/Robot")
    for index, dock_name in enumerate(
        (
            "pitch_dock_mech_1",
            "yaw_dock_mech_1",
            "yaw_dock_mech_1",
            "pitch_dock_mech_1",
        ),
        start=1,
    ):
        instance_name = f"{dock_name}_1"
        prim = UsdGeom.Xform.Define(
            root,
            f"/Robot/dock_{index}_{dock_name}",
        ).GetPrim()
        prim.GetReferences().AddReference(
            str(instances_path),
            f"/Instances/{instance_name}",
        )
        prim.SetInstanceable(True)
    root.SetDefaultPrim(root.GetPrimAtPath("/Robot"))
    root.GetRootLayer().Save()

    evidence = enforce_holon_dock_mesh_collision_approximation(
        root_path,
        collision_type="Convex Decomposition",
    )

    assert evidence["verified"] is True
    assert evidence["requested_approximation_token"] == "convexDecomposition"
    assert evidence["physx_convex_decomposition_api_verified"] is True
    assert evidence["max_convex_hulls"] == 128
    assert evidence["shrink_wrap"] is True
    assert evidence["authored_prim_count"] == 2
    assert evidence["composed_prim_count"] == 4
    assert set(evidence["original_approximation_tokens"].values()) == {
        "convexHull"
    }

    reopened = Usd.Stage.Open(str(root_path))
    composed = []
    for prim in Usd.PrimRange.Stage(reopened, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        composed.append(
            str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        )
    assert composed == ["convexDecomposition"] * 4
    for prim in Usd.PrimRange.Stage(reopened, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        assert "PhysxConvexDecompositionCollisionAPI" in str(
            prim.GetMetadata("apiSchemas")
        )
        assert (
            prim.GetAttribute(
                "physxConvexDecompositionCollision:maxConvexHulls"
            ).Get()
            == 128
        )
        assert (
            prim.GetAttribute(
                "physxConvexDecompositionCollision:shrinkWrap"
            ).Get()
            is True
        )


@pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="pxr is available only in the Isaac environment",
)
def test_enforce_fails_closed_without_dock_collision_prims(tmp_path) -> None:
    from pxr import Usd, UsdGeom

    root_path = tmp_path / "robot.usda"
    root = Usd.Stage.CreateNew(str(root_path))
    UsdGeom.Xform.Define(root, "/Robot")
    root.SetDefaultPrim(root.GetPrimAtPath("/Robot"))
    root.GetRootLayer().Save()

    with pytest.raises(RuntimeError, match="no editable Holon Dock"):
        enforce_holon_dock_mesh_collision_approximation(
            root_path,
            collision_type="Convex Decomposition",
        )
