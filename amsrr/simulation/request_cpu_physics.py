"""CPU dynamics with optional GPU broadphase for a static request scene.

Only pair finding and its reserved memory change. Contact solving, integration,
materials, geometry and all control tensors stay on the existing CPU path.
"""


def aggregate_pair_capacity(stage, original):
    """Conservative all-shape-pairs reservation; keep defaults for expansions.

    This caller creates the entire scene before reset and adds no colliders
    during the episode. Two reports per unordered pair cover found plus lost.
    Mesh decomposition and point instancers do not have one shape per prim.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    count = 0
    unsupported = []
    single_mesh = {
        "none",
        "convexHull",
        "boundingCube",
        "boundingSphere",
        "meshSimplification",
        "sdf",
    }
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.PointInstancer):
            unsupported.append(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        count += 1
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mode = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if mode not in single_mesh:
                unsupported.append(str(prim.GetPath()))
    if not count:
        raise ValueError("GPU broadphase requires an authored collision scene")
    pairs = max(65536, count * (count - 1))
    bound = 1 << (pairs - 1).bit_length()
    capacity = original if unsupported else min(original, bound)
    return dict(
        shape_count=count,
        unsupported=unsupported,
        original_capacity=original,
        chosen_capacity=capacity,
    )


def configure_cpu_broadphase(sim, mode):
    from pxr import PhysxSchema
    import torch

    if str(sim.device) != "cpu" or mode not in ("MBP", "GPU"):
        raise ValueError("CPU broadphase requires the CPU dynamics backend")
    api = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
    if api.GetEnableGPUDynamicsAttr().Get():
        raise ValueError("CPU dynamics unexpectedly enabled GPU solving")
    reservation = None
    if mode == "GPU":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU broadphase requested without a CUDA device")
        sim.set_setting("/physics/cudaDevice", 0)
        sim.set_setting("/physics/suppressReadback", False)
        attr = api.GetGpuFoundLostAggregatePairsCapacityAttr()
        reservation = aggregate_pair_capacity(sim.stage, int(attr.Get()))
        attr.Set(reservation["chosen_capacity"])
        api.GetBroadphaseTypeAttr().Set(mode)
    if api.GetBroadphaseTypeAttr().Get() != mode:
        raise RuntimeError("CPU broadphase readback differs from request")
    return dict(
        tensor_device=str(sim.device),
        gpu_dynamics=api.GetEnableGPUDynamicsAttr().Get(),
        broadphase=api.GetBroadphaseTypeAttr().Get(),
        physics_threads=sim.get_setting("/persistent/physics/numThreads"),
        aggregate_pair_reservation=reservation,
    )
