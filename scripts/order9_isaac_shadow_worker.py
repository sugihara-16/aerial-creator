#!/usr/bin/env python3
from __future__ import annotations

"""Persistent real-Isaac worker for the production Order 9 ``C_H`` gate.

One process owns one topology/object-geometry rollout bucket and one immutable
``pi_L`` checkpoint.  Requests arrive through the prefixed JSON-line protocol;
ordinary Kit output is ignored by the client transport.
"""

import argparse
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    # Keep this non-required while AppLauncher performs its early
    # ``parse_known_args`` pass; enforce it after the complete parser exists so
    # ``--help`` remains usable without a fake checkpoint path.
    parser.add_argument("--pi-l-checkpoint")
    parser.add_argument("--pi-l-checkpoint-sha256")
    parser.add_argument(
        "--morphology-graph-json",
        help=(
            "Path to the exact MorphologyGraph owned by this topology bucket. "
            "Omitting it retains the canonical Order 8 three-module bucket."
        ),
    )
    parser.add_argument(
        "--robot-usd",
        default=(
            "artifacts/isaac/robots/holon/holon_p4_2_graph/"
            "holon_p4_2_graph.usda"
        ),
    )
    parser.add_argument(
        "--object-geometry",
        choices=("box", "sphere", "cylinder", "capsule"),
        default="box",
    )
    parser.add_argument("--object-id", default="order8_object")
    parser.add_argument("--object-size", nargs=3, type=float, default=(0.30, 0.40, 0.15))
    parser.add_argument("--object-mass-kg", type=float, default=1.0)
    parser.add_argument("--object-friction", type=float, default=0.6)
    parser.add_argument("--selected-gripper-friction", type=float, default=4.5)
    parser.add_argument("--contact-stiffness", type=float, default=7500.0)
    parser.add_argument("--contact-damping", type=float, default=75.0)
    parser.add_argument("--support-top-z", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.02)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
if not args_cli.pi_l_checkpoint:
    raise ValueError("--pi-l-checkpoint is required")
if any(not math.isfinite(value) or value <= 0.0 for value in args_cli.object_size):
    raise ValueError("--object-size values must be positive")
for name in (
    "object_mass_kg",
    "object_friction",
    "selected_gripper_friction",
    "contact_stiffness",
    "contact_damping",
    "dt",
):
    value = float(getattr(args_cli, name))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"--{name.replace('_', '-')} must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from amsrr.controllers.actuator_mapping import build_actuator_mapping
from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.geometry.mass_properties import (
    cuboid_mass_properties,
    cylinder_mass_properties,
    mass_properties_from_geometry,
    sphere_mass_properties,
)
from amsrr.policies.order9_low_level_runtime import Order9LowLevelRuntimePolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_actuator_runtime import (
    order9_actuator_runtime_values,
    validate_order9_actuator_readback,
)
from amsrr.simulation.order9_isaac_scene_adapter import (
    IsaacLabOrder9SceneAdapter,
    Order9IsaacContactViewLayout,
)
from amsrr.simulation.order9_isaac_shadow_runtime import Order9IsaacCopiedRuntime
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.simulation.order9_shadow_executor import (
    Order9IsaacShadowExecutor,
    Order9IsaacShadowExecutorConfig,
)
from amsrr.simulation.order9_shadow_worker import run_order9_shadow_worker_rpc
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import CollisionModel, GeometrySpec, GeometryType
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pipeline import order9_schedule_hash


ROBOT_USD = str(Path(args_cli.robot_usd).resolve())
if not Path(ROBOT_USD).is_file():
    raise FileNotFoundError(f"Order9 robot USD is missing: {ROBOT_USD}")
CONFIG = load_order9_learning_config(args_cli.config)
CONFIG.validate()
PHYSICAL_MODEL = build_physical_model_from_config(
    CONFIG.production_runtime.robot_model_config_path
)
ACTUATOR_RUNTIME = order9_actuator_runtime_values(PHYSICAL_MODEL)
CANONICAL_RESET = load_order9_canonical_reset(
    CONFIG.production_runtime.canonical_order8_report_path,
    expected_sha256=CONFIG.production_runtime.canonical_order8_report_sha256,
)
if args_cli.morphology_graph_json:
    _MORPHOLOGY_PATH = Path(args_cli.morphology_graph_json).resolve()
    if not _MORPHOLOGY_PATH.is_file():
        raise FileNotFoundError(
            f"Order9 morphology graph is missing: {_MORPHOLOGY_PATH}"
        )
    MORPHOLOGY = MorphologyGraph.from_json(
        _MORPHOLOGY_PATH.read_text(encoding="utf-8")
    )
    MORPHOLOGY.validate()
else:
    MORPHOLOGY = None
OBJECT_SIZE = tuple(float(value) for value in args_cli.object_size)
_SUPPORT_SIZE_RAW = CANONICAL_RESET.metadata.get("object_support_size_m")
_SUPPORT_POSE_RAW = CANONICAL_RESET.metadata.get("object_support_pose_world")
if not (
    isinstance(_SUPPORT_SIZE_RAW, list)
    and len(_SUPPORT_SIZE_RAW) == 3
    and isinstance(_SUPPORT_POSE_RAW, list)
    and len(_SUPPORT_POSE_RAW) == 7
):
    raise RuntimeError("Order9 canonical reset lacks support geometry")
SUPPORT_SIZE = tuple(float(value) for value in _SUPPORT_SIZE_RAW)
SUPPORT_CENTER = (
    float(_SUPPORT_POSE_RAW[0]),
    float(_SUPPORT_POSE_RAW[1]),
    float(args_cli.support_top_z) - 0.5 * SUPPORT_SIZE[2],
)


def _object_spawn_cfg():
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.object_mass_kg)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(args_cli.object_friction),
            dynamic_friction=float(args_cli.object_friction),
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.72, 0.38, 0.12)
        ),
        activate_contact_sensors=True,
    )
    if args_cli.object_geometry == "box":
        return sim_utils.CuboidCfg(size=OBJECT_SIZE, **common)
    if args_cli.object_geometry == "sphere":
        if not math.isclose(OBJECT_SIZE[0], OBJECT_SIZE[1], rel_tol=0.0, abs_tol=1.0e-9) or not math.isclose(OBJECT_SIZE[1], OBJECT_SIZE[2], rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("sphere object size must be an equal-diameter vector")
        return sim_utils.SphereCfg(radius=0.5 * OBJECT_SIZE[0], **common)
    if not math.isclose(OBJECT_SIZE[0], OBJECT_SIZE[1], rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("round object x/y size must be the common diameter")
    if args_cli.object_geometry == "cylinder":
        return sim_utils.CylinderCfg(
            radius=0.5 * OBJECT_SIZE[0], height=OBJECT_SIZE[2], **common
        )
    return sim_utils.CapsuleCfg(
        radius=0.5 * OBJECT_SIZE[0], height=OBJECT_SIZE[2], **common
    )


@configclass
class Order9ShadowSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.0,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "gimbal_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*gimbal.*"],
                stiffness=ACTUATOR_RUNTIME.gimbal_stiffness,
                damping=ACTUATOR_RUNTIME.gimbal_damping,
                armature=ACTUATOR_RUNTIME.gimbal_armature,
                effort_limit_sim=ACTUATOR_RUNTIME.gimbal_effort_limit,
                velocity_limit_sim=ACTUATOR_RUNTIME.gimbal_velocity_limit,
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=ACTUATOR_RUNTIME.dock_stiffness,
                damping=ACTUATOR_RUNTIME.dock_damping,
                armature=ACTUATOR_RUNTIME.dock_armature,
                effort_limit_sim=ACTUATOR_RUNTIME.dock_effort_limit,
                velocity_limit_sim=ACTUATOR_RUNTIME.dock_velocity_limit,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"], stiffness=0.0, damping=0.0
            ),
        },
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_object_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3267, 0.0, 0.225)),
    )
    support: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=SUPPORT_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SUPPORT_CENTER
        ),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        ),
    )


def _mass_properties() -> list[float]:
    mass = float(args_cli.object_mass_kg)
    if args_cli.object_geometry == "box":
        volume = OBJECT_SIZE[0] * OBJECT_SIZE[1] * OBJECT_SIZE[2]
        return cuboid_mass_properties(
            OBJECT_SIZE, density_kg_m3=mass / volume
        ).inertia_kgm2
    if args_cli.object_geometry == "sphere":
        radius = 0.5 * OBJECT_SIZE[0]
        volume = 4.0 * math.pi * radius**3 / 3.0
        return sphere_mass_properties(
            radius, density_kg_m3=mass / volume
        ).inertia_kgm2
    radius = 0.5 * OBJECT_SIZE[0]
    if args_cli.object_geometry == "cylinder":
        volume = math.pi * radius * radius * OBJECT_SIZE[2]
        return cylinder_mass_properties(
            radius, OBJECT_SIZE[2], density_kg_m3=mass / volume
        ).inertia_kgm2
    geometry = GeometrySpec(
        geometry_id="order9_shadow_capsule",
        geometry_type=GeometryType.CAPSULE,
        primitive_params={"radius_m": radius, "height_m": OBJECT_SIZE[2]},
        asset_path=None,
        collision_model=CollisionModel.PRIMITIVE,
    )
    unit_density = mass_properties_from_geometry(
        geometry,
        density_kg_m3=1.0,
    )
    return mass_properties_from_geometry(
        geometry,
        density_kg_m3=mass / unit_density.volume_m3,
    ).inertia_kgm2


def _rigid_body_path_by_name(stage, robot_root: str) -> dict[str, str]:
    root = stage.GetPrimAtPath(Sdf.Path(robot_root))
    if not root.IsValid():
        raise RuntimeError(f"Order9 robot prim is invalid: {robot_root}")
    result: dict[str, str] = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            result[prim.GetName()] = prim.GetPath().pathString
    return result


def _bind_selected_material(stage, selected_paths: list[str]) -> None:
    material_path = "/World/Order9SelectedGripperMaterial"
    cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(args_cli.selected_gripper_friction),
        dynamic_friction=float(args_cli.selected_gripper_friction),
        restitution=0.0,
        compliant_contact_stiffness=float(args_cli.contact_stiffness),
        compliant_contact_damping=float(args_cli.contact_damping),
        friction_combine_mode="max",
    )
    cfg.func(material_path, cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(Sdf.Path(material_path)))
    if not material.GetPrim().IsValid():
        raise RuntimeError("Order9 selected gripper material is invalid")
    for path in selected_paths:
        body = stage.GetPrimAtPath(Sdf.Path(path))
        api = (
            UsdShade.MaterialBindingAPI(body)
            if body.HasAPI(UsdShade.MaterialBindingAPI)
            else UsdShade.MaterialBindingAPI.Apply(body)
        )
        api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        collisions = [
            prim
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not collisions:
            raise RuntimeError(f"Order9 selected body has no collision: {path}")
        for collision in collisions:
            bound, _ = UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial(
                materialPurpose="physics"
            )
            if bound.GetPath().pathString != material_path:
                raise RuntimeError(
                    f"Order9 selected material binding failed: {collision.GetPath()}"
                )


def _body_local_aabbs(
    stage, paths: dict[str, str]
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )
    result: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = {}
    for name, path in paths.items():
        body = stage.GetPrimAtPath(Sdf.Path(path))
        lower = [math.inf, math.inf, math.inf]
        upper = [-math.inf, -math.inf, -math.inf]
        for collision in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not collision.HasAPI(UsdPhysics.CollisionAPI):
                continue
            owner = collision
            while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner.IsValid() or owner.GetPath() != body.GetPath():
                continue
            extent = cache.ComputeRelativeBound(collision, body).ComputeAlignedRange()
            minimum = extent.GetMin()
            maximum = extent.GetMax()
            for axis in range(3):
                lower[axis] = min(lower[axis], float(minimum[axis]))
                upper[axis] = max(upper[axis], float(maximum[axis]))
        if all(math.isfinite(value) for value in (*lower, *upper)) and all(
            lower[axis] <= upper[axis] for axis in range(3)
        ):
            result[name] = (tuple(lower), tuple(upper))
    return result


def _activate_nested_contact_reports(stage, *, root_prim_path: str) -> int:
    """Apply zero-threshold reporting to every rigid body in the articulation."""

    root_prefix = root_prim_path.rstrip("/") + "/"
    applied = 0
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path != root_prim_path and not path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        applied += 1
    if applied == 0:
        raise RuntimeError("Order9 worker found no robot rigid body for contact reporting")
    return applied


def _require_contact_view_layout(
    view,
    *,
    label: str,
    sensor_count: int,
    filter_count: int,
) -> None:
    actual_sensors = int(view.sensor_count)
    actual_filters = int(view.filter_count)
    if actual_sensors != sensor_count or actual_filters != filter_count:
        raise RuntimeError(
            f"Order9 {label} contact-view layout mismatch: "
            f"sensors={actual_sensors}/{sensor_count}, "
            f"filters={actual_filters}/{filter_count}"
        )


def main() -> int:
    config = CONFIG
    reset = CANONICAL_RESET
    physical_model = PHYSICAL_MODEL
    morphology = MORPHOLOGY or build_representative_order8_morphology(physical_model)
    if MORPHOLOGY is None and morphology.graph_id != reset.source_graph_id:
        raise RuntimeError("Order9 worker morphology differs from canonical Order8")
    policy = Order9LowLevelRuntimePolicy.from_checkpoint(
        args_cli.pi_l_checkpoint,
        physical_model=physical_model,
        expected_sha256=args_cli.pi_l_checkpoint_sha256,
        expected_schedule_hash=order9_schedule_hash(config),
        deterministic=True,
        device=str(args_cli.device),
    )
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=float(args_cli.dt),
            device=str(args_cli.device),
            use_fabric=True,
        )
    )
    scene = InteractiveScene(
        Order9ShadowSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    )
    robot = scene["robot"]
    object_asset = scene["object"]
    robot_root = "/World/envs/env_0/Robot"
    paths_by_name = _rigid_body_path_by_name(sim.stage, robot_root)
    anchor_by_id = {anchor.anchor_id: anchor for anchor in morphology.robot_anchors}
    selected_anchors = sorted(anchor_by_id.values(), key=lambda item: item.anchor_id)
    selected_names = [
        f"module_{anchor.module_id}__{anchor.link_id}" for anchor in selected_anchors
    ]
    if set(selected_names) - set(paths_by_name):
        raise RuntimeError("Order9 worker could not resolve selected anchor bodies")
    selected_paths = [paths_by_name[name] for name in selected_names]
    _bind_selected_material(sim.stage, selected_paths)
    _activate_nested_contact_reports(sim.stage, root_prim_path=robot_root)
    sim.reset()
    scene.reset()
    effort_limits = robot.data.joint_effort_limits
    velocity_limits = robot.data.joint_velocity_limits
    if hasattr(effort_limits, "torch"):
        effort_limits = effort_limits.torch
    if hasattr(velocity_limits, "torch"):
        velocity_limits = velocity_limits.torch
    actuator_readback = validate_order9_actuator_readback(
        robot.joint_names,
        effort_limits[0].tolist(),
        velocity_limits[0].tolist(),
        expected=ACTUATOR_RUNTIME,
    )
    if set(robot.body_names) - set(paths_by_name):
        raise RuntimeError("Order9 worker could not resolve every robot rigid body")
    physics_view = sim.physics_manager.get_physics_sim_view()
    object_path = "/World/envs/env_0/Object"
    support_path = "/World/envs/env_0/Support"
    selected_view = physics_view.create_rigid_contact_view(
        selected_paths,
        filter_patterns=[[object_path] for _ in selected_paths],
        max_contact_data_count=max(128, 32 * len(selected_paths)),
    )
    _require_contact_view_layout(
        selected_view,
        label="selected_robot_object",
        sensor_count=len(selected_paths),
        filter_count=1,
    )
    all_names = list(robot.body_names)
    all_paths = [paths_by_name[name] for name in all_names]
    all_view = physics_view.create_rigid_contact_view(
        all_paths,
        filter_patterns=[[object_path, support_path] for _ in all_paths],
        max_contact_data_count=max(512, 16 * len(all_paths)),
    )
    _require_contact_view_layout(
        all_view,
        label="all_robot_object_support",
        sensor_count=len(all_paths),
        filter_count=2,
    )
    layout = Order9IsaacContactViewLayout(
        selected_sensor_body_names=tuple(selected_names),
        selected_anchor_ids=tuple(anchor.anchor_id for anchor in selected_anchors),
        all_sensor_body_names=tuple(all_names),
        all_sensor_entity_ids=tuple(
            name.replace("__", ":", 1) for name in all_names
        ),
        all_filter_entity_ids=(str(args_cli.object_id), "support"),
    )
    adapter = IsaacLabOrder9SceneAdapter(
        sim=sim,
        robot=robot,
        object_asset=object_asset,
        morphology_graph=morphology,
        physical_model=physical_model,
        selected_contact_view=selected_view,
        all_contact_view=all_view,
        contact_layout=layout,
        torch_module=torch,
        warp_module=wp,
        device=str(args_cli.device),
        object_id=str(args_cli.object_id),
        object_mass_kg=float(args_cli.object_mass_kg),
        object_inertia_body=_mass_properties(),
        object_friction=float(args_cli.object_friction),
        selected_gripper_friction=float(args_cli.selected_gripper_friction),
        contact_stiffness_n_per_m=float(args_cli.contact_stiffness),
        contact_damping_n_s_per_m=float(args_cli.contact_damping),
        object_geometry_type=str(args_cli.object_geometry),
        object_size_m=OBJECT_SIZE,
        support_top_z_m=float(args_cli.support_top_z),
        support_center_world_m=SUPPORT_CENTER,
        support_half_extents_m=tuple(0.5 * value for value in SUPPORT_SIZE),
        body_local_aabb_m=_body_local_aabbs(sim.stage, paths_by_name),
        actuator_readback=actuator_readback,
    )
    runtime = Order9IsaacCopiedRuntime(
        scene_adapter=adapter,
        morphology_graph=morphology,
        physical_model=physical_model,
        pi_l_policy=policy,
        controller=QPIDController(
            config=QPIDControllerConfig(
                allocation_mode="rigid_body_qp",
                control_dt_s=float(args_cli.dt),
            )
        ),
        actuator_mapping=build_actuator_mapping(morphology, physical_model),
        force_scale_n=config.hard_checker.qp_force_scale_n,
        torque_scale_nm=config.hard_checker.qp_torque_scale_nm,
    )
    executor = Order9IsaacShadowExecutor(
        runtime,
        config=Order9IsaacShadowExecutorConfig(
            control_dt_s=config.hard_checker.shadow_control_dt_s,
            maximum_horizon_s=config.hard_checker.shadow_rollout_horizon_s,
            force_scale_n=config.hard_checker.qp_force_scale_n,
            torque_scale_nm=config.hard_checker.qp_torque_scale_nm,
            maximum_control_steps=(
                math.ceil(
                    config.hard_checker.shadow_rollout_horizon_s
                    / config.hard_checker.shadow_control_dt_s
                )
                + 1
            ),
        ),
    )
    print(
        "ORDER9_SHADOW_READY="
        + json.dumps(
            {
                "worker_version": executor.worker_version,
                "pi_l_checkpoint_sha256": executor.pi_l_checkpoint_sha256,
                "object_geometry": str(args_cli.object_geometry),
                "actuator_readback": actuator_readback,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        return run_order9_shadow_worker_rpc(executor)
    finally:
        sim.stop()
        sim.clear_instance()

_exit_code = 1
try:
    _exit_code = main()
except BaseException as exc:
    print(
        "ORDER9_SHADOW_STARTUP_ERROR="
        + json.dumps(
            {"error_type": type(exc).__name__, "error": str(exc)},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    raise
finally:
    simulation_app.close()
raise SystemExit(_exit_code)
