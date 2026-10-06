from pxr import Usd, UsdGeom, UsdPhysics
import pytest
from amsrr.simulation.request_cpu_physics import aggregate_pair_capacity


def test_capacity_covers_all_found_and_lost_pairs_and_counts_disabled_shapes():
    stage = Usd.Stage.CreateInMemory()
    for i in range(300):
        p = UsdGeom.Cube.Define(stage, f"/box_{i}").GetPrim()
        UsdPhysics.CollisionAPI.Apply(p).GetCollisionEnabledAttr().Set(i % 2 == 0)
    value = aggregate_pair_capacity(stage, 2**25)
    assert value["shape_count"] == 300
    assert value["chosen_capacity"] == 2**17
    assert value["chosen_capacity"] >= 2 * (300 * 299 // 2)


@pytest.mark.parametrize("kind", ["decomposition", "instancer"])
def test_expanding_geometry_keeps_original_capacity(kind):
    stage = Usd.Stage.CreateInMemory()
    p = UsdGeom.Mesh.Define(stage, "/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(p)
    if kind == "decomposition":
        UsdPhysics.MeshCollisionAPI.Apply(p).GetApproximationAttr().Set(
            "convexDecomposition"
        )
    else:
        UsdGeom.PointInstancer.Define(stage, "/instances")
    value = aggregate_pair_capacity(stage, 2**25)
    assert value["chosen_capacity"] == 2**25 and value["unsupported"]


def test_empty_collision_scene_rejected():
    with pytest.raises(ValueError, match="collision scene"):
        aggregate_pair_capacity(Usd.Stage.CreateInMemory(), 2**25)
