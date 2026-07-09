from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

from amsrr.geometry.pose_math import compose_pose, inverse_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn the generated Holon USD as an Isaac Lab articulation.")
    parser.add_argument("--config", default="configs/env/isaac_lab.yaml", help="Isaac Lab backend config path.")
    parser.add_argument("--convert-if-missing", action="store_true", help="Convert the Holon URDF if USD is absent.")
    parser.add_argument("--force-convert", action="store_true", help="Force URDF-to-USD conversion before spawn.")
    parser.add_argument("--generated-usd-dir", default=None, help="Override generated USD output directory.")
    parser.add_argument("--generated-usd-path", default=None, help="Override generated USD path to spawn.")
    parser.add_argument("--steps", type=int, default=5, help="Number of physics steps after reset.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation time step in seconds.")
    parser.add_argument("--spawn-height", type=float, default=0.5, help="Initial Holon root height in meters.")
    parser.add_argument("--force-per-rotor-n", type=float, default=0.0, help="World +z force per thrust body.")
    parser.add_argument(
        "--hover-force-scale",
        type=float,
        default=None,
        help="Override force per rotor using total mass * gravity / rotor count times this scale.",
    )
    parser.add_argument("--gimbal-target-rad", type=float, default=0.0, help="Position target for gimbal joints.")
    parser.add_argument("--gimbal-tolerance-rad", type=float, default=0.02, help="Probe pass tolerance for gimbal joints.")
    parser.add_argument("--gimbal-stiffness", type=float, default=20.0, help="Implicit actuator stiffness.")
    parser.add_argument("--gimbal-damping", type=float, default=1.0, help="Implicit actuator damping.")
    parser.add_argument("--dock-stiffness", type=float, default=20.0, help="Implicit dock mechanism hold stiffness.")
    parser.add_argument("--dock-damping", type=float, default=1.0, help="Implicit dock mechanism hold damping.")
    parser.add_argument(
        "--controller-command-smoke",
        action="store_true",
        help="Use A-MSRR QPIDController and IsaacControllerBridge outputs as the command source.",
    )
    parser.add_argument(
        "--single-module-hover-smoke",
        action="store_true",
        help="Run a closed-loop single-module hover smoke with QPIDController commands.",
    )
    parser.add_argument(
        "--single-module-articulated-hover-smoke",
        action="store_true",
        help="Run a closed-loop single-module hover smoke while dock mechanism joints move.",
    )
    parser.add_argument(
        "--fixed-morphology-hover-smoke",
        action="store_true",
        help="Run a closed-loop fixed-morphology hover smoke with a rigid combined URDF.",
    )
    parser.add_argument(
        "--fixed-morphology-articulated-hover-smoke",
        action="store_true",
        help="Run a closed-loop fixed-morphology hover smoke while dock mechanism joints move.",
    )
    parser.add_argument(
        "--fixed-morphology-waypoint-smoke",
        action="store_true",
        help="Run a closed-loop fixed-morphology waypoint smoke with a rigid combined URDF.",
    )
    parser.add_argument(
        "--p4-1-full-scene-backend-smoke",
        action="store_true",
        help="Run the P4.1 robot+object+floor backend smoke and emit short per-step records.",
    )
    parser.add_argument(
        "--p4-1-uses-p2-p3",
        action="store_true",
        help="Mark this P4.1 case as sourced from P2 selected design and P3 assembled morphology.",
    )
    parser.add_argument(
        "--p4-1-object-size-m",
        type=float,
        nargs=3,
        default=(0.30, 0.20, 0.15),
        metavar=("X", "Y", "Z"),
        help="P4.1 box object dimensions in meters.",
    )
    parser.add_argument("--p4-1-object-mass-kg", type=float, default=1.0, help="P4.1 object mass in kg.")
    parser.add_argument(
        "--p4-1-object-pose-world",
        type=float,
        nargs=7,
        default=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="P4.1 object initial pose in world coordinates.",
    )
    parser.add_argument(
        "--p4-2-deterministic-rollout",
        action="store_true",
        help="Run the P4.2 deterministic graph-specific grasp/carry rollout probe.",
    )
    parser.add_argument(
        "--p4-2-uses-p2-p3",
        action="store_true",
        help="Mark this P4.2 case as sourced from P2 selected design and P3 assembled morphology.",
    )
    parser.add_argument(
        "--p4-2-morphology-graph-json",
        default=None,
        help="Serialized P3 assembled MorphologyGraph JSON for graph-specific reset asset generation.",
    )
    parser.add_argument(
        "--p4-2-object-size-m",
        type=float,
        nargs=3,
        default=(0.30, 0.20, 0.15),
        metavar=("X", "Y", "Z"),
        help="P4.2 box object dimensions in meters.",
    )
    parser.add_argument("--p4-2-object-mass-kg", type=float, default=1.0, help="P4.2 object mass in kg.")
    parser.add_argument(
        "--p4-2-object-pose-world",
        type=float,
        nargs=7,
        default=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="P4.2 object initial pose in world coordinates.",
    )
    parser.add_argument(
        "--p4-2-contact-model",
        default="kinematic_payload_coupled_attach_v1",
        help="P4.2 contact model label.",
    )
    parser.add_argument(
        "--p4-2-attach-distance-threshold-m",
        type=float,
        default=0.06,
        help="P4.2 object attach distance threshold.",
    )
    parser.add_argument(
        "--p4-2-attach-relative-velocity-threshold-mps",
        type=float,
        default=0.20,
        help="P4.2 object attach relative velocity threshold.",
    )
    parser.add_argument(
        "--p4-2-attach-snap-distance-threshold-m",
        type=float,
        default=0.03,
        help="P4.2 object attach snap distance threshold.",
    )
    parser.add_argument("--fixed-module-count", type=int, default=2, help="Module count for fixed-morphology smokes.")
    parser.add_argument("--fixed-module-spacing-m", type=float, default=0.45, help="Rigid spacing between fixed modules.")
    parser.add_argument(
        "--allocation-mode",
        choices=("rigid_body_qp", "rigid_body_pseudoinverse"),
        default="rigid_body_qp",
        help="Controller allocation mode for closed-loop P4-control smokes.",
    )
    parser.add_argument(
        "--vectoring-velocity-limit-rad-s",
        type=float,
        default=None,
        help="Override gimbal/vectoring joint velocity limits in the generated conversion URDF.",
    )
    parser.add_argument("--hover-target-height", type=float, default=None, help="Closed-loop hover target z in meters.")
    parser.add_argument(
        "--waypoint-target-position-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed-morphology waypoint target position in meters.",
    )
    parser.add_argument("--waypoint-target-yaw-rad", type=float, default=0.0, help="Fixed-morphology waypoint target yaw.")
    parser.add_argument("--waypoint-ramp-duration-s", type=float, default=0.1, help="Ramp duration for waypoint targets.")
    parser.add_argument("--hover-position-tolerance-m", type=float, default=0.20, help="Closed-loop hover position tolerance.")
    parser.add_argument("--hover-attitude-tolerance-rad", type=float, default=0.25, help="Closed-loop hover attitude tolerance.")
    parser.add_argument("--hover-hold-duration-s", type=float, default=1.0, help="Required final hold duration for hover pass.")
    parser.add_argument(
        "--articulated-joint-amplitude-rad",
        type=float,
        default=0.12,
        help="Sinusoidal dock mechanism joint target amplitude for articulated hover smokes.",
    )
    parser.add_argument(
        "--articulated-joint-period-s",
        type=float,
        default=8.0,
        help="Sinusoidal dock mechanism joint target period for articulated hover smokes.",
    )
    parser.add_argument(
        "--articulated-joint-warmup-s",
        type=float,
        default=1.0,
        help="Initial zero-target warmup before articulated joint motion starts.",
    )
    parser.add_argument(
        "--articulated-joint-tracking-tolerance-rad",
        type=float,
        default=0.20,
        help="Allowed max dock mechanism target tracking error for articulated hover smokes.",
    )
    parser.add_argument(
        "--articulated-joint-names",
        nargs="*",
        default=None,
        help="Optional local dock mechanism joint names to move; defaults to all dock mechanism joints.",
    )
    parser.add_argument(
        "--hover-stop-on-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop closed-loop hover smoke once the required hold duration is achieved.",
    )
    parser.add_argument(
        "--realtime-playback",
        action="store_true",
        help="Sleep one physics dt after each step so GUI playback is easier to inspect.",
    )
    parser.add_argument(
        "--keep-open-after-smoke-s",
        type=float,
        default=0.0,
        help="Keep the Kit app open for this many seconds after the smoke finishes.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> int:
    try:
        report = run_probe(args_cli)
    except Exception as exc:  # pragma: no cover - exercised through real Isaac smoke commands.
        report = {
            "spawn_passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    print(json.dumps(report, sort_keys=True))
    if not report.get("spawn_passed"):
        return 1
    for key in (
        "command_probe_passed",
        "single_module_hover_smoke_passed",
        "single_module_articulated_hover_smoke_passed",
        "fixed_morphology_hover_smoke_passed",
        "fixed_morphology_articulated_hover_smoke_passed",
        "fixed_morphology_waypoint_smoke_passed",
        "p4_1_full_scene_backend_smoke_passed",
        "p4_2_deterministic_rollout_passed",
    ):
        if report.get(key) is False:
            return 1
    return 0


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    import torch

    from amsrr.robot_model.fixed_morphology_urdf import (
        articulated_morphology_connections,
        fixed_morphology_module_poses,
        morphology_graph_module_poses,
        split_fixed_module_name,
        write_articulated_morphology_urdf,
        write_fixed_morphology_urdf,
        write_fixed_morphology_graph_urdf,
        write_joint_velocity_override_urdf,
        write_resolved_mesh_urdf,
    )
    from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
    from amsrr.schemas.morphology import MorphologyGraph
    from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
    from amsrr.simulation.p4_control_controller_smoke import (
        bridge_supported_controller_command,
        build_fixed_morphology,
        build_runtime_observation,
        build_single_module_controller_command_smoke,
        build_single_module_morphology,
    )

    backend_config = load_isaac_lab_backend_config(args.config)
    backend = IsaacLabBackend(backend_config)
    physical_model = build_physical_model_from_config(backend_config.robot_model_config_path)
    urdf_path = _expand_path(backend_config.holon_urdf_path)
    usd_dir = _expand_path(args.generated_usd_dir or backend_config.generated_usd_dir)
    usd_path = _expand_path(args.generated_usd_path or backend_config.generated_usd_path)
    fixed_control_smoke_requested = bool(
        args.fixed_morphology_hover_smoke
        or args.fixed_morphology_articulated_hover_smoke
        or args.fixed_morphology_waypoint_smoke
    )
    fixed_smoke_requested = bool(
        fixed_control_smoke_requested
        or args.p4_1_full_scene_backend_smoke
        or args.p4_2_deterministic_rollout
    )
    p4_2_morphology_graph = (
        MorphologyGraph.from_json(args.p4_2_morphology_graph_json)
        if args.p4_2_deterministic_rollout and args.p4_2_morphology_graph_json
        else None
    )
    fixed_module_poses = None
    articulated_connections = []
    converted = False

    if args.force_convert or fixed_smoke_requested or (args.convert_if_missing and not usd_path.exists()):
        mesh_search_dirs = _holon_mesh_search_dirs()
        if fixed_smoke_requested:
            if args.p4_2_deterministic_rollout:
                if p4_2_morphology_graph is None:
                    raise RuntimeError("P4.2 deterministic rollout requires --p4-2-morphology-graph-json")
                fixed_module_poses = morphology_graph_module_poses(p4_2_morphology_graph)
                graph_urdf_path = usd_dir / "graph_morphology_urdf" / "holon_p4_2_graph.urdf"
                urdf_path = write_fixed_morphology_graph_urdf(
                    urdf_path,
                    graph_urdf_path,
                    morphology_graph=p4_2_morphology_graph,
                    mesh_search_dirs=mesh_search_dirs,
                )
            else:
                fixed_module_poses = fixed_morphology_module_poses(
                    urdf_path,
                    module_count=int(args.fixed_module_count),
                    module_spacing_m=float(args.fixed_module_spacing_m),
                )
            if args.p4_2_deterministic_rollout:
                pass
            elif args.fixed_morphology_articulated_hover_smoke:
                articulated_connections = articulated_morphology_connections(
                    urdf_path,
                    module_count=int(args.fixed_module_count),
                )
                articulated_urdf_path = (
                    usd_dir
                    / "articulated_morphology_urdf"
                    / f"holon_articulated_{int(args.fixed_module_count)}.urdf"
                )
                urdf_path = write_articulated_morphology_urdf(
                    urdf_path,
                    articulated_urdf_path,
                    module_count=int(args.fixed_module_count),
                    mesh_search_dirs=mesh_search_dirs,
                )
            else:
                fixed_urdf_path = usd_dir / "fixed_morphology_urdf" / f"holon_fixed_{int(args.fixed_module_count)}.urdf"
                urdf_path = write_fixed_morphology_urdf(
                    urdf_path,
                    fixed_urdf_path,
                    module_count=int(args.fixed_module_count),
                    module_spacing_m=float(args.fixed_module_spacing_m),
                    mesh_search_dirs=mesh_search_dirs,
                )
        else:
            urdf_path = write_resolved_mesh_urdf(
                urdf_path,
                usd_dir / "resolved_urdf" / "holon.urdf",
                mesh_search_dirs=mesh_search_dirs,
            )
        if args.vectoring_velocity_limit_rad_s is not None:
            urdf_path = write_joint_velocity_override_urdf(
                urdf_path,
                urdf_path,
                joint_velocity_overrides=_vectoring_velocity_overrides(
                    physical_model,
                    float(args.vectoring_velocity_limit_rad_s),
                ),
            )
        converter_cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(usd_dir),
            fix_base=False,
            merge_fixed_joints=False,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=100.0,
                    damping=1.0,
                ),
                target_type="position",
            ),
        )
        converter = UrdfConverter(converter_cfg)
        usd_path = Path(converter.usd_path)
        converted = True

    if not usd_path.exists():
        raise FileNotFoundError(f"Generated Holon USD is missing: {usd_path}")

    sim_utils.create_new_stage()
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, args.spawn_height])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Holon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args.spawn_height),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "gimbal_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*gimbal.*"],
                stiffness=args.gimbal_stiffness,
                damping=args.gimbal_damping,
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=args.dock_stiffness,
                damping=args.dock_damping,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    p4_1_object = None
    if args.p4_1_full_scene_backend_smoke or args.p4_2_deterministic_rollout:
        object_pose = tuple(
            float(value)
            for value in (args.p4_2_object_pose_world if args.p4_2_deterministic_rollout else args.p4_1_object_pose_world)
        )
        object_size = tuple(
            float(value)
            for value in (args.p4_2_object_size_m if args.p4_2_deterministic_rollout else args.p4_1_object_size_m)
        )
        object_mass = float(args.p4_2_object_mass_kg if args.p4_2_deterministic_rollout else args.p4_1_object_mass_kg)
        object_cfg = RigidObjectCfg(
            prim_path="/World/Object/box_01",
            spawn=sim_utils.CuboidCfg(
                size=object_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=object_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 0.85)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=object_pose[:3],
                rot=object_pose[3:7],
            ),
        )
        p4_1_object = RigidObject(object_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()
    if p4_1_object is not None:
        p4_1_object.update(sim_dt)
    thrust_body_ids, thrust_body_names = robot.find_bodies(".*thrust_.*")
    gimbal_joint_ids, gimbal_joint_names = robot.find_joints(".*gimbal.*")
    robot_mass = float(robot.data.body_mass.torch[0].sum().detach().cpu())
    gravity = float(torch.tensor(sim.cfg.gravity, device=sim.device).norm().detach().cpu())
    force_per_rotor_n = float(args.force_per_rotor_n)
    if args.hover_force_scale is not None:
        if not thrust_body_ids:
            raise RuntimeError("Cannot compute hover force without thrust bodies.")
        force_per_rotor_n = robot_mass * gravity * float(args.hover_force_scale) / len(thrust_body_ids)
    command_applied = force_per_rotor_n != 0.0 or args.gimbal_target_rad != 0.0
    expected_thrust_bodies = 4 * int(args.fixed_module_count) if fixed_smoke_requested else 4
    if force_per_rotor_n != 0.0 and len(thrust_body_ids) != expected_thrust_bodies:
        raise RuntimeError(
            f"Expected {expected_thrust_bodies} thrust bodies, found {len(thrust_body_ids)}: {thrust_body_names}"
        )
    if args.gimbal_target_rad != 0.0 and not gimbal_joint_ids:
        raise RuntimeError("Cannot command gimbal target without gimbal joints.")

    thrust_body_ids_tensor = torch.tensor(thrust_body_ids, dtype=torch.int32, device=sim.device)
    gimbal_joint_ids_tensor = torch.tensor(gimbal_joint_ids, dtype=torch.int32, device=sim.device)
    initial_root_pos_w = _tensor_row(robot.data.root_pos_w.torch)
    controller_bundle = None
    hover_smoke_report = None
    p4_1_smoke_report = None
    p4_2_rollout_report = None
    if args.controller_command_smoke:
        controller_bundle = build_single_module_controller_command_smoke(
            physical_model,
            time_s=0.0,
            command_index=0,
            control_dt_s=args.dt,
            pose_world=tuple(_tensor_row(robot.data.root_pose_w.torch)),  # type: ignore[arg-type]
            twist_world=_tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch),
            joint_positions=_joint_state_dict(robot.joint_names, robot.data.joint_pos.torch),
            joint_velocities=_joint_state_dict(robot.joint_names, robot.data.joint_vel.torch),
        )
        _apply_actuator_record(robot, controller_bundle.actuator_target_record, physical_model, sim.device)
    if args.single_module_hover_smoke or args.single_module_articulated_hover_smoke:
        single_articulated = bool(args.single_module_articulated_hover_smoke)
        hover_smoke_report = _run_single_module_hover_smoke(
            robot=robot,
            sim=sim,
            sim_dt=sim_dt,
            physical_model=physical_model,
            device=sim.device,
            steps=max(0, args.steps),
            target_height=float(args.hover_target_height if args.hover_target_height is not None else args.spawn_height),
            position_tolerance_m=float(args.hover_position_tolerance_m),
            attitude_tolerance_rad=float(args.hover_attitude_tolerance_rad),
            hold_duration_s=float(args.hover_hold_duration_s),
            stop_on_hold=bool(args.hover_stop_on_hold),
            control_dt_s=float(args.dt),
            build_runtime_observation=build_runtime_observation,
            build_single_module_morphology=build_single_module_morphology,
            bridge_supported_controller_command=bridge_supported_controller_command,
            realtime_playback=bool(args.realtime_playback),
            allocation_mode=str(args.allocation_mode),
            report_prefix=(
                "single_module_articulated_hover"
                if single_articulated
                else "single_module_hover"
            ),
            articulated=single_articulated,
            articulated_joint_names=args.articulated_joint_names,
            articulated_joint_amplitude_rad=float(args.articulated_joint_amplitude_rad),
            articulated_joint_period_s=float(args.articulated_joint_period_s),
            articulated_joint_warmup_s=float(args.articulated_joint_warmup_s),
            articulated_joint_tracking_tolerance_rad=float(args.articulated_joint_tracking_tolerance_rad),
        )
    if fixed_control_smoke_requested:
        waypoint_position = args.waypoint_target_position_m
        target_height = float(args.hover_target_height if args.hover_target_height is not None else args.spawn_height)
        if waypoint_position is None:
            waypoint_position = [0.25, 0.0, target_height]
        if args.fixed_morphology_waypoint_smoke:
            target_position = tuple(float(value) for value in waypoint_position)
            target_yaw = float(args.waypoint_target_yaw_rad)
            report_prefix = "fixed_morphology_waypoint"
            waypoint_ramp_duration_s = float(args.waypoint_ramp_duration_s)
            fixed_articulated = False
            fixed_articulated_joint_names = None
        elif args.fixed_morphology_articulated_hover_smoke:
            target_position = (0.0, 0.0, target_height)
            target_yaw = 0.0
            report_prefix = "fixed_morphology_articulated_hover"
            waypoint_ramp_duration_s = 0.0
            fixed_articulated = True
            fixed_articulated_joint_names = args.articulated_joint_names or [
                connection.parent_mechanism_joint_id
                for connection in articulated_connections
                if connection.parent_mechanism_joint_id is not None
            ]
        else:
            target_position = (0.0, 0.0, target_height)
            target_yaw = 0.0
            report_prefix = "fixed_morphology_hover"
            waypoint_ramp_duration_s = 0.0
            fixed_articulated = False
            fixed_articulated_joint_names = args.articulated_joint_names
        hover_smoke_report = _run_fixed_morphology_smoke(
            robot=robot,
            sim=sim,
            sim_dt=sim_dt,
            physical_model=physical_model,
            device=sim.device,
            steps=max(0, args.steps),
            module_count=int(args.fixed_module_count),
            module_spacing_m=float(args.fixed_module_spacing_m),
            module_poses=fixed_module_poses,
            target_position=target_position,  # type: ignore[arg-type]
            target_yaw_rad=target_yaw,
            position_tolerance_m=float(args.hover_position_tolerance_m),
            attitude_tolerance_rad=float(args.hover_attitude_tolerance_rad),
            hold_duration_s=float(args.hover_hold_duration_s),
            stop_on_hold=bool(args.hover_stop_on_hold),
            control_dt_s=float(args.dt),
            build_fixed_morphology=build_fixed_morphology,
            bridge_supported_controller_command=bridge_supported_controller_command,
            split_fixed_module_name=split_fixed_module_name,
            report_prefix=report_prefix,
            waypoint_ramp_duration_s=waypoint_ramp_duration_s,
            realtime_playback=bool(args.realtime_playback),
            allocation_mode=str(args.allocation_mode),
            articulated=fixed_articulated,
            articulated_joint_names=fixed_articulated_joint_names,
            articulated_joint_amplitude_rad=float(args.articulated_joint_amplitude_rad),
            articulated_joint_period_s=float(args.articulated_joint_period_s),
            articulated_joint_warmup_s=float(args.articulated_joint_warmup_s),
            articulated_joint_tracking_tolerance_rad=float(args.articulated_joint_tracking_tolerance_rad),
            articulated_assembly=bool(args.fixed_morphology_articulated_hover_smoke),
        )
    if args.p4_1_full_scene_backend_smoke:
        if p4_1_object is None:
            raise RuntimeError("P4.1 full-scene backend smoke requested without a spawned object.")
        p4_1_smoke_report = _run_p4_1_full_scene_backend_smoke(
            robot=robot,
            p4_1_object=p4_1_object,
            sim=sim,
            sim_dt=sim_dt,
            physical_model=physical_model,
            device=sim.device,
            steps=max(0, args.steps),
            module_count=int(args.fixed_module_count),
            module_spacing_m=float(args.fixed_module_spacing_m),
            module_poses=fixed_module_poses,
            object_id="box_01",
            target_height=float(args.hover_target_height if args.hover_target_height is not None else args.spawn_height),
            control_dt_s=float(args.dt),
            build_fixed_morphology=build_fixed_morphology,
            bridge_supported_controller_command=bridge_supported_controller_command,
            split_fixed_module_name=split_fixed_module_name,
            realtime_playback=bool(args.realtime_playback),
            allocation_mode=str(args.allocation_mode),
            uses_p2_p3=bool(args.p4_1_uses_p2_p3),
        )
    if args.p4_2_deterministic_rollout:
        if p4_1_object is None:
            raise RuntimeError("P4.2 deterministic rollout requested without a spawned object.")
        if p4_2_morphology_graph is None:
            raise RuntimeError("P4.2 deterministic rollout requested without a morphology graph.")
        p4_2_rollout_report = _run_p4_2_deterministic_rollout_probe(
            robot=robot,
            p4_2_object=p4_1_object,
            sim=sim,
            sim_dt=sim_dt,
            physical_model=physical_model,
            device=sim.device,
            steps=max(0, args.steps),
            morphology_graph=p4_2_morphology_graph,
            module_poses=fixed_module_poses,
            object_id="box_01",
            target_height=float(args.hover_target_height if args.hover_target_height is not None else args.spawn_height),
            control_dt_s=float(args.dt),
            bridge_supported_controller_command=bridge_supported_controller_command,
            split_fixed_module_name=split_fixed_module_name,
            realtime_playback=bool(args.realtime_playback),
            allocation_mode=str(args.allocation_mode),
            uses_p2_p3=bool(args.p4_2_uses_p2_p3),
            contact_model=str(args.p4_2_contact_model),
            attach_snap_distance_threshold_m=float(args.p4_2_attach_snap_distance_threshold_m),
        )
    for _ in range(
        0
        if hover_smoke_report is not None or p4_1_smoke_report is not None or p4_2_rollout_report is not None
        else max(0, args.steps)
    ):
        if controller_bundle is not None:
            _apply_actuator_record(robot, controller_bundle.actuator_target_record, physical_model, sim.device)
        elif force_per_rotor_n != 0.0:
            forces = torch.zeros(robot.num_instances, len(thrust_body_ids), 3, device=sim.device)
            torques = torch.zeros_like(forces)
            forces[..., 2] = force_per_rotor_n
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces,
                torques=torques,
                body_ids=thrust_body_ids_tensor,
                is_global=True,
            )
        if args.gimbal_target_rad != 0.0:
            gimbal_targets = torch.full(
                (robot.num_instances, len(gimbal_joint_ids)),
                float(args.gimbal_target_rad),
                dtype=torch.float32,
                device=sim.device,
            )
            robot.set_joint_position_target_index(target=gimbal_targets, joint_ids=gimbal_joint_ids_tensor)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if p4_1_object is not None:
            p4_1_object.update(sim_dt)
        if args.realtime_playback:
            time.sleep(max(0.0, sim_dt))
    final_root_pos_w = _tensor_row(robot.data.root_pos_w.torch)
    gimbal_joint_pos = _tensor_indices(robot.data.joint_pos.torch, gimbal_joint_ids)
    gimbal_joint_pos_target = _tensor_indices(robot.data.joint_pos_target.torch, gimbal_joint_ids)
    gimbal_target_error_rad = 0.0
    if args.gimbal_target_rad != 0.0:
        gimbal_target_error_rad = max(
            abs(position - float(args.gimbal_target_rad))
            for position in gimbal_joint_pos
        )
    force_command_ok = force_per_rotor_n == 0.0 or len(thrust_body_ids) == 4
    gimbal_command_ok = args.gimbal_target_rad == 0.0 or gimbal_target_error_rad <= args.gimbal_tolerance_rad
    controller_command_ok = True
    if controller_bundle is not None:
        controller_command_ok = (
            controller_bundle.actuator_target_record.metrics["missing_actuator_count"] == 0.0
            and controller_bundle.actuator_target_record.metrics["unsupported_actuator_count"] == 0.0
            and controller_bundle.controller_command.controller_status.metrics.get("qp_primary_path", 0.0) == 1.0
        )
    hover_command_ok = True
    if hover_smoke_report is not None:
        hover_command_ok = any(
            bool(hover_smoke_report.get(key))
            for key in (
                "single_module_hover_smoke_passed",
                "single_module_articulated_hover_smoke_passed",
                "fixed_morphology_hover_smoke_passed",
                "fixed_morphology_articulated_hover_smoke_passed",
                "fixed_morphology_waypoint_smoke_passed",
            )
        )

    report = {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": (
            command_applied
            or controller_bundle is not None
            or hover_smoke_report is not None
            or p4_1_smoke_report is not None
            or p4_2_rollout_report is not None
        ),
        "command_probe_passed": (
            force_command_ok and gimbal_command_ok and controller_command_ok and hover_command_ok
            if command_applied
            or controller_bundle is not None
            or hover_smoke_report is not None
            or p4_1_smoke_report is not None
            or p4_2_rollout_report is not None
            else None
        ),
        "controller_command_smoke": controller_bundle is not None,
        "converted": converted,
        "usd_path": str(usd_path),
        "urdf_path": str(urdf_path),
        "prim_path": robot_cfg.prim_path,
        "steps": max(0, args.steps),
        "sim_dt": sim_dt,
        "device": args.device,
        "robot_mass_kg": robot_mass,
        "gravity_mps2": gravity,
        "num_instances": int(robot.num_instances),
        "num_bodies": int(robot.num_bodies),
        "num_joints": int(robot.num_joints),
        "body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
        "thrust_body_ids": list(thrust_body_ids),
        "thrust_body_names": list(thrust_body_names),
        "gimbal_joint_ids": list(gimbal_joint_ids),
        "gimbal_joint_names": list(gimbal_joint_names),
        "force_per_rotor_n": force_per_rotor_n,
        "total_commanded_force_n": force_per_rotor_n * len(thrust_body_ids),
        "gimbal_target_rad": float(args.gimbal_target_rad),
        "gimbal_tolerance_rad": float(args.gimbal_tolerance_rad),
        "gimbal_target_error_rad": gimbal_target_error_rad,
        "gimbal_joint_pos": gimbal_joint_pos,
        "gimbal_joint_pos_target": gimbal_joint_pos_target,
        "allocation_mode": str(args.allocation_mode),
        "vectoring_velocity_limit_rad_s": (
            float(args.vectoring_velocity_limit_rad_s)
            if args.vectoring_velocity_limit_rad_s is not None
            else None
        ),
        "initial_root_pos_w": initial_root_pos_w,
        "root_pos_w": final_root_pos_w,
        "root_delta_w": [final - initial for final, initial in zip(final_root_pos_w, initial_root_pos_w, strict=True)],
        "root_quat_w": _tensor_row(robot.data.root_quat_w.torch),
        "root_lin_vel_w": _tensor_row(robot.data.root_lin_vel_w.torch),
        "root_ang_vel_w": _tensor_row(robot.data.root_ang_vel_w.torch),
        "joint_pos_sample": _tensor_row(robot.data.joint_pos.torch, limit=8),
    }
    if controller_bundle is not None:
        report.update(_controller_bundle_report(controller_bundle))
    if hover_smoke_report is not None:
        report.update(hover_smoke_report)
    if p4_1_smoke_report is not None:
        report.update(p4_1_smoke_report)
    if p4_2_rollout_report is not None:
        report.update(p4_2_rollout_report)
    report["realtime_playback"] = bool(args.realtime_playback)
    report["keep_open_after_smoke_s"] = float(args.keep_open_after_smoke_s)
    if args.keep_open_after_smoke_s > 0.0:
        _keep_viewer_open(float(args.keep_open_after_smoke_s))
    sim.stop()
    sim.clear_instance()
    return report


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def _holon_mesh_search_dirs() -> list[Path]:
    return [
        REPO_ROOT / "module_urdf",
        REPO_ROOT / "module_urdf" / "mesh",
    ]


def _vectoring_velocity_overrides(physical_model, velocity_limit_rad_s: float) -> dict[str, float]:
    if velocity_limit_rad_s < 0.0:
        raise ValueError("vectoring velocity limit must be non-negative")
    return {
        joint_id: velocity_limit_rad_s
        for rotor in physical_model.rotors
        for joint_id in rotor.vectoring_joint_ids
    }


def _articulated_joint_ids(physical_model, requested_joint_names: list[str] | None) -> list[str]:
    available = sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
            if port.mechanical_limits.get("mechanism_joint_id")
        }
    )
    if requested_joint_names is None:
        return available
    requested = [str(joint_id) for joint_id in requested_joint_names]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown dock mechanism joints for articulated smoke: {unknown}")
    return requested


def _articulated_joint_targets(
    joint_ids: list[str],
    *,
    time_s: float,
    amplitude_rad: float,
    period_s: float,
    warmup_s: float,
) -> dict[str, float]:
    if amplitude_rad < 0.0:
        raise ValueError("articulated joint amplitude must be non-negative")
    if period_s <= 0.0:
        raise ValueError("articulated joint period must be positive")
    if warmup_s < 0.0:
        raise ValueError("articulated joint warmup must be non-negative")
    if time_s < warmup_s or not joint_ids:
        return {joint_id: 0.0 for joint_id in joint_ids}
    active_t = float(time_s) - float(warmup_s)
    value = float(amplitude_rad) * math.sin(2.0 * math.pi * active_t / float(period_s))
    return {joint_id: value for joint_id in joint_ids}


def _joint_position_for_module_joint(
    joint_names: list[str],
    joint_positions_tensor,
    *,
    module_id: int,
    local_id: str,
) -> float | None:
    joint_name = _resolve_module_name(joint_names, module_id, local_id)
    if joint_name is None:
        return None
    return _tensor_row(joint_positions_tensor)[joint_names.index(joint_name)]


def _joint_positions_for_command_key(
    joint_names: list[str],
    joint_positions_tensor,
    *,
    command_key: str,
    module_count: int,
) -> list[float]:
    parsed = _split_global_command_key(command_key)
    if parsed is not None:
        module_id, local_id = parsed
        value = _joint_position_for_module_joint(
            joint_names,
            joint_positions_tensor,
            module_id=module_id,
            local_id=local_id,
        )
        return [] if value is None else [value]
    values = []
    for module_id in range(module_count):
        value = _joint_position_for_module_joint(
            joint_names,
            joint_positions_tensor,
            module_id=module_id,
            local_id=command_key,
        )
        if value is not None:
            values.append(value)
    return values


def _module_scoped_joint_targets(joint_targets: dict[str, float], *, module_id: int) -> dict[str, float]:
    return {f"module_{module_id}:{joint_id}": float(value) for joint_id, value in joint_targets.items()}


def _split_global_command_key(command_key: str) -> tuple[int, str] | None:
    if not command_key.startswith("module_"):
        return None
    module_text, separator, local_id = command_key.partition(":")
    if separator == "" or not local_id:
        return None
    module_id_text = module_text[len("module_") :]
    if not module_id_text.isdigit():
        return None
    return int(module_id_text), local_id


def _module_body_pose(robot, *, module_id: int, local_body_name: str):
    body_name = _resolve_module_name(robot.body_names, module_id, local_body_name)
    if body_name is None:
        return None
    body_id = robot.body_names.index(body_name)
    pos = _tensor_body_row(robot.data.body_pos_w.torch, body_id)
    quat = _tensor_body_row(robot.data.body_quat_w.torch, body_id)
    return (pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3])


def _module_body_twist(robot, *, module_id: int, local_body_name: str):
    body_name = _resolve_module_name(robot.body_names, module_id, local_body_name)
    if body_name is None:
        return None
    body_id = robot.body_names.index(body_name)
    if not hasattr(robot.data, "body_lin_vel_w") or not hasattr(robot.data, "body_ang_vel_w"):
        return None
    linear = _tensor_body_row(robot.data.body_lin_vel_w.torch, body_id)
    angular = _tensor_body_row(robot.data.body_ang_vel_w.torch, body_id)
    return [*linear, *angular]


def _tensor_body_row(tensor, body_id: int) -> list[float]:
    row = tensor[0, body_id]
    return [float(value) for value in row.detach().cpu().tolist()]


def _max_vector_dict_delta(
    current: dict[str, tuple[float, float, float]],
    initial: dict[str, tuple[float, float, float]],
) -> float:
    max_delta = 0.0
    for key, current_vector in current.items():
        initial_vector = initial.get(key)
        if initial_vector is None:
            continue
        max_delta = max(
            max_delta,
            sum((float(current_vector[idx]) - float(initial_vector[idx])) ** 2 for idx in range(3)) ** 0.5,
        )
    return max_delta


def _max_matrix_delta(current: list[list[float]], initial: list[list[float]]) -> float:
    max_delta = 0.0
    for row_idx, row in enumerate(current):
        if row_idx >= len(initial):
            continue
        for col_idx, value in enumerate(row):
            if col_idx >= len(initial[row_idx]):
                continue
            max_delta = max(max_delta, abs(float(value) - float(initial[row_idx][col_idx])))
    return max_delta


def _run_single_module_hover_smoke(
    *,
    robot,
    sim,
    sim_dt: float,
    physical_model,
    device: str,
    steps: int,
    target_height: float,
    position_tolerance_m: float,
    attitude_tolerance_rad: float,
    hold_duration_s: float,
    stop_on_hold: bool,
    control_dt_s: float,
    build_runtime_observation,
    build_single_module_morphology,
    bridge_supported_controller_command,
    realtime_playback: bool,
    allocation_mode: str,
    report_prefix: str,
    articulated: bool,
    articulated_joint_names: list[str] | None,
    articulated_joint_amplitude_rad: float,
    articulated_joint_period_s: float,
    articulated_joint_warmup_s: float,
    articulated_joint_tracking_tolerance_rad: float,
) -> dict[str, object]:
    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand, PostureTarget

    morphology_graph = build_single_module_morphology(
        physical_model,
        graph_id="single-module-hover-smoke",
    )
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    bridge = IsaacControllerBridge()
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode=allocation_mode,
            control_dt_s=control_dt_s,
        )
    )
    target_pose = (0.0, 0.0, target_height, 0.0, 0.0, 0.0, 1.0)
    target_twist = [0.0] * 6
    previous_command = None
    max_position_error = 0.0
    max_attitude_error = 0.0
    final_position_error = 0.0
    final_attitude_error = 0.0
    qp_infeasible_count = 0
    clipped_count = 0
    missing_actuator_count = 0
    unsupported_actuator_count = 0
    clipped_target_count = 0
    finite_state = True
    hold_steps = 0
    max_hold_steps = 0
    hold_steps_required = max(1, int(round(hold_duration_s / max(sim_dt, 1.0e-9))))
    min_height = float("inf")
    max_height = float("-inf")
    last_controller_status = None
    last_bridge_metrics: dict[str, float] = {}
    executed_steps = 0
    articulated_joint_ids = _articulated_joint_ids(physical_model, articulated_joint_names) if articulated else []
    max_joint_target_abs = 0.0
    max_joint_position_abs = 0.0
    max_joint_tracking_error = 0.0
    observed_joint_count = 0
    last_joint_targets: dict[str, float] = {}

    for step_idx in range(max(0, steps)):
        executed_steps = step_idx + 1
        time_s = step_idx * sim_dt
        joint_targets = (
            _articulated_joint_targets(
                articulated_joint_ids,
                time_s=time_s,
                amplitude_rad=articulated_joint_amplitude_rad,
                period_s=articulated_joint_period_s,
                warmup_s=articulated_joint_warmup_s,
            )
            if articulated
            else {}
        )
        last_joint_targets = dict(joint_targets)
        posture_target = (
            PostureTarget(joint_pos_target=joint_targets, joint_vel_target={joint_id: 0.0 for joint_id in joint_targets})
            if joint_targets
            else None
        )
        runtime_observation = build_runtime_observation(
            morphology_graph,
            time_s=time_s,
            pose_world=tuple(_tensor_row(robot.data.root_pose_w.torch)),
            twist_world=_tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch),
            joint_positions=_joint_state_dict(robot.joint_names, robot.data.joint_pos.torch),
            joint_velocities=_joint_state_dict(robot.joint_names, robot.data.joint_vel.torch),
        )
        controller_command = controller.compute(
            ControllerContext(
                runtime_observation=runtime_observation,
                morphology_graph=morphology_graph,
                physical_model=physical_model,
                active_knot=InteractionKnot(
                    t_rel_s=time_s,
                    contact_assignments=[],
                    posture_target=posture_target,
                ),
                policy_command=PolicyCommand(
                    desired_body_pose=target_pose,
                    desired_body_twist=target_twist,
                ),
                previous_command=previous_command,
                control_dt_s=control_dt_s,
            )
        )
        bridged_command = bridge_supported_controller_command(controller_command)
        actuator_record = bridge.convert(
            bridged_command,
            actuator_mapping,
            time_s=step_idx * sim_dt,
            command_index=step_idx,
        )
        _apply_actuator_record(robot, actuator_record, physical_model, device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if realtime_playback:
            time.sleep(max(0.0, sim_dt))

        root_pose = _tensor_row(robot.data.root_pose_w.torch)
        root_pos = root_pose[:3]
        position_error = _position_error_norm(root_pos, target_pose[:3])
        attitude_error = _quat_error_norm(root_pose[3:7], target_pose[3:7])
        final_position_error = position_error
        final_attitude_error = attitude_error
        max_position_error = max(max_position_error, position_error)
        max_attitude_error = max(max_attitude_error, attitude_error)
        min_height = min(min_height, root_pos[2])
        max_height = max(max_height, root_pos[2])
        finite_state = finite_state and all(_is_finite(value) for value in root_pose)
        if articulated:
            for joint_id, target in joint_targets.items():
                actual = _joint_position_for_module_joint(
                    robot.joint_names,
                    robot.data.joint_pos.torch,
                    module_id=0,
                    local_id=joint_id,
                )
                if actual is None:
                    continue
                observed_joint_count += 1
                max_joint_target_abs = max(max_joint_target_abs, abs(float(target)))
                max_joint_position_abs = max(max_joint_position_abs, abs(float(actual)))
                max_joint_tracking_error = max(max_joint_tracking_error, abs(float(actual) - float(target)))
        if position_error <= position_tolerance_m and attitude_error <= attitude_tolerance_rad:
            hold_steps += 1
        else:
            hold_steps = 0
        max_hold_steps = max(max_hold_steps, hold_steps)

        status = bridged_command.controller_status
        last_controller_status = status.to_dict()
        last_bridge_metrics = dict(actuator_record.metrics)
        if not status.qp_feasible:
            qp_infeasible_count += 1
        if status.metrics.get("clipped", 0.0) > 0.0:
            clipped_count += 1
        missing_actuator_count += len(actuator_record.missing_actuators)
        unsupported_actuator_count += len(actuator_record.unsupported_actuators)
        clipped_target_count += len(actuator_record.clipped_targets)
        previous_command = bridged_command
        motion_observed_for_stop = (
            not articulated
            or max_joint_position_abs + 1.0e-9 >= 0.5 * abs(float(articulated_joint_amplitude_rad))
        )
        if stop_on_hold and hold_steps >= hold_steps_required and motion_observed_for_stop:
            break

    hold_time_s = max_hold_steps * sim_dt
    expected_joint_motion = 0.5 * abs(float(articulated_joint_amplitude_rad))
    joint_motion_passed = (
        not articulated
        or (
            bool(articulated_joint_ids)
            and observed_joint_count > 0
            and max_joint_target_abs + 1.0e-9 >= expected_joint_motion
            and max_joint_position_abs + 1.0e-9 >= expected_joint_motion
            and max_joint_tracking_error <= articulated_joint_tracking_tolerance_rad
        )
    )
    passed = (
        executed_steps > 0
        and finite_state
        and qp_infeasible_count == 0
        and missing_actuator_count == 0
        and unsupported_actuator_count == 0
        and clipped_target_count == 0
        and final_position_error <= position_tolerance_m
        and final_attitude_error <= attitude_tolerance_rad
        and hold_time_s + 1.0e-9 >= hold_duration_s
        and joint_motion_passed
    )
    return {
        f"{report_prefix}_smoke": True,
        f"{report_prefix}_smoke_passed": passed,
        f"{report_prefix}_target_pose": list(target_pose),
        f"{report_prefix}_steps": int(executed_steps),
        f"{report_prefix}_requested_steps": int(max(0, steps)),
        f"{report_prefix}_duration_s": float(executed_steps * sim_dt),
        f"{report_prefix}_hold_time_s": hold_time_s,
        f"{report_prefix}_hold_required_s": hold_duration_s,
        f"{report_prefix}_stopped_on_hold": bool(stop_on_hold and executed_steps < max(0, steps)),
        f"{report_prefix}_position_tolerance_m": position_tolerance_m,
        f"{report_prefix}_attitude_tolerance_rad": attitude_tolerance_rad,
        f"{report_prefix}_final_position_error_m": final_position_error,
        f"{report_prefix}_final_attitude_error_rad": final_attitude_error,
        f"{report_prefix}_max_position_error_m": max_position_error,
        f"{report_prefix}_max_attitude_error_rad": max_attitude_error,
        f"{report_prefix}_min_height_m": min_height if executed_steps > 0 else None,
        f"{report_prefix}_max_height_m": max_height if executed_steps > 0 else None,
        f"{report_prefix}_finite_state": finite_state,
        f"{report_prefix}_qp_infeasible_count": qp_infeasible_count,
        f"{report_prefix}_controller_clipped_count": clipped_count,
        f"{report_prefix}_missing_actuator_count": missing_actuator_count,
        f"{report_prefix}_unsupported_actuator_count": unsupported_actuator_count,
        f"{report_prefix}_clipped_target_count": clipped_target_count,
        f"{report_prefix}_articulated": bool(articulated),
        f"{report_prefix}_articulated_joint_ids": list(articulated_joint_ids),
        f"{report_prefix}_articulated_joint_amplitude_rad": float(articulated_joint_amplitude_rad),
        f"{report_prefix}_articulated_joint_period_s": float(articulated_joint_period_s),
        f"{report_prefix}_articulated_joint_warmup_s": float(articulated_joint_warmup_s),
        f"{report_prefix}_articulated_joint_tracking_tolerance_rad": float(
            articulated_joint_tracking_tolerance_rad
        ),
        f"{report_prefix}_articulated_joint_motion_passed": bool(joint_motion_passed),
        f"{report_prefix}_articulated_joint_observed_count": int(observed_joint_count),
        f"{report_prefix}_articulated_max_joint_target_abs_rad": float(max_joint_target_abs),
        f"{report_prefix}_articulated_max_joint_position_abs_rad": float(max_joint_position_abs),
        f"{report_prefix}_articulated_max_joint_tracking_error_rad": float(max_joint_tracking_error),
        f"{report_prefix}_articulated_last_joint_targets": dict(last_joint_targets),
        f"{report_prefix}_last_controller_status": last_controller_status,
        f"{report_prefix}_last_bridge_metrics": last_bridge_metrics,
    }


def _run_fixed_morphology_smoke(
    *,
    robot,
    sim,
    sim_dt: float,
    physical_model,
    device: str,
    steps: int,
    module_count: int,
    module_spacing_m: float,
    module_poses: dict[int, tuple[float, float, float, float, float, float, float]] | None,
    target_position: tuple[float, float, float],
    target_yaw_rad: float,
    position_tolerance_m: float,
    attitude_tolerance_rad: float,
    hold_duration_s: float,
    stop_on_hold: bool,
    control_dt_s: float,
    build_fixed_morphology,
    bridge_supported_controller_command,
    split_fixed_module_name,
    report_prefix: str,
    waypoint_ramp_duration_s: float,
    realtime_playback: bool,
    allocation_mode: str,
    articulated: bool,
    articulated_joint_names: list[str] | None,
    articulated_joint_amplitude_rad: float,
    articulated_joint_period_s: float,
    articulated_joint_warmup_s: float,
    articulated_joint_tracking_tolerance_rad: float,
    articulated_assembly: bool,
) -> dict[str, object]:
    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand, PostureTarget

    morphology_graph = build_fixed_morphology(
        physical_model,
        graph_id=f"{report_prefix}-smoke",
        module_count=module_count,
        module_spacing_m=module_spacing_m,
        module_poses=module_poses,
    )
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    bridge = IsaacControllerBridge()
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode=allocation_mode,
            control_dt_s=control_dt_s,
        )
    )
    target_quat = _yaw_quat_xyzw(target_yaw_rad)
    target_pose = (
        float(target_position[0]),
        float(target_position[1]),
        float(target_position[2]),
        target_quat[0],
        target_quat[1],
        target_quat[2],
        target_quat[3],
    )
    initial_root_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
    tracked_initial_pose = initial_root_pose
    if articulated_assembly:
        initial_base_pose = _module_body_pose(robot, module_id=0, local_body_name="fc")
        if initial_base_pose is not None:
            height_offset = float(initial_base_pose[2]) - float(initial_root_pose[2])
            target_pose = (
                float(initial_base_pose[0]),
                float(initial_base_pose[1]),
                float(target_position[2]) + height_offset,
                target_pose[3],
                target_pose[4],
                target_pose[5],
                target_pose[6],
            )
            tracked_initial_pose = initial_base_pose
    target_twist = [0.0] * 6
    previous_command = None
    max_position_error = 0.0
    max_attitude_error = 0.0
    final_position_error = 0.0
    final_attitude_error = 0.0
    qp_infeasible_count = 0
    clipped_count = 0
    missing_actuator_count = 0
    unsupported_actuator_count = 0
    clipped_target_count = 0
    finite_state = True
    hold_steps = 0
    max_hold_steps = 0
    hold_steps_required = max(1, int(round(hold_duration_s / max(sim_dt, 1.0e-9))))
    min_height = float("inf")
    max_height = float("-inf")
    last_controller_status = None
    last_bridge_metrics: dict[str, float] = {}
    executed_steps = 0
    articulated_joint_ids = _articulated_joint_ids(physical_model, articulated_joint_names) if articulated else []
    max_joint_target_abs = 0.0
    max_joint_position_abs = 0.0
    max_joint_tracking_error = 0.0
    observed_joint_count = 0
    last_joint_targets: dict[str, float] = {}
    initial_relative_module_pose = None
    max_relative_module_position_change = 0.0
    max_relative_module_attitude_change = 0.0
    initial_model_rotor_origins = None
    initial_model_allocation_matrix = None
    max_model_rotor_origin_change = 0.0
    max_model_allocation_change = 0.0

    for step_idx in range(max(0, steps)):
        executed_steps = step_idx + 1
        time_s = step_idx * sim_dt
        root_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
        root_twist = _tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch)
        command_target_pose = _ramped_target_pose(
            tracked_initial_pose,  # type: ignore[arg-type]
            target_pose,
            elapsed_s=time_s,
            ramp_duration_s=waypoint_ramp_duration_s,
        )
        local_joint_targets = (
            _articulated_joint_targets(
                articulated_joint_ids,
                time_s=time_s,
                amplitude_rad=articulated_joint_amplitude_rad,
                period_s=articulated_joint_period_s,
                warmup_s=articulated_joint_warmup_s,
            )
            if articulated
            else {}
        )
        joint_targets = (
            _module_scoped_joint_targets(local_joint_targets, module_id=0)
            if articulated_assembly
            else local_joint_targets
        )
        last_joint_targets = dict(joint_targets)
        posture_target = (
            PostureTarget(joint_pos_target=joint_targets, joint_vel_target={joint_id: 0.0 for joint_id in joint_targets})
            if joint_targets
            else None
        )
        if articulated_assembly:
            runtime_observation = _build_articulated_runtime_observation(
                morphology_graph,
                time_s=time_s,
                robot=robot,
                root_pose_world=root_pose,  # type: ignore[arg-type]
                root_twist_world=root_twist,
                joint_names=robot.joint_names,
                joint_positions_tensor=robot.data.joint_pos.torch,
                joint_velocities_tensor=robot.data.joint_vel.torch,
                module_count=module_count,
                split_fixed_module_name=split_fixed_module_name,
            )
        else:
            runtime_observation = _build_fixed_runtime_observation(
                morphology_graph,
                time_s=time_s,
                root_pose_world=root_pose,  # type: ignore[arg-type]
                root_twist_world=root_twist,
                joint_names=robot.joint_names,
                joint_positions_tensor=robot.data.joint_pos.torch,
                joint_velocities_tensor=robot.data.joint_vel.torch,
                module_count=module_count,
                module_spacing_m=module_spacing_m,
                module_poses=module_poses,
                split_fixed_module_name=split_fixed_module_name,
            )
        if articulated_assembly and len(runtime_observation.module_states) > 1:
            base_pose = runtime_observation.module_states[0].pose_world
            child_pose = runtime_observation.module_states[1].pose_world
            relative_pose = compose_pose(inverse_pose(base_pose), child_pose)
            if initial_relative_module_pose is None:
                initial_relative_module_pose = relative_pose
            max_relative_module_position_change = max(
                max_relative_module_position_change,
                _position_error_norm(list(relative_pose[:3]), initial_relative_module_pose[:3]),
            )
            max_relative_module_attitude_change = max(
                max_relative_module_attitude_change,
                _quat_error_norm(list(relative_pose[3:7]), initial_relative_module_pose[3:7]),
            )
            diagnostic_model = controller.rigid_body_model_builder.build(
                morphology_graph,
                physical_model,
                runtime_observation,
            )
            if initial_model_rotor_origins is None:
                initial_model_rotor_origins = dict(diagnostic_model.rotor_origins_body)
                initial_model_allocation_matrix = [list(row) for row in diagnostic_model.allocation_matrix_body]
            else:
                max_model_rotor_origin_change = max(
                    max_model_rotor_origin_change,
                    _max_vector_dict_delta(diagnostic_model.rotor_origins_body, initial_model_rotor_origins),
                )
                max_model_allocation_change = max(
                    max_model_allocation_change,
                    _max_matrix_delta(
                        diagnostic_model.allocation_matrix_body,
                        initial_model_allocation_matrix or diagnostic_model.allocation_matrix_body,
                    ),
                )
        controller_command = controller.compute(
            ControllerContext(
                runtime_observation=runtime_observation,
                morphology_graph=morphology_graph,
                physical_model=physical_model,
                active_knot=InteractionKnot(
                    t_rel_s=time_s,
                    contact_assignments=[],
                    posture_target=posture_target,
                ),
                policy_command=PolicyCommand(
                    desired_body_pose=command_target_pose,
                    desired_body_twist=target_twist,
                ),
                previous_command=previous_command,
                control_dt_s=control_dt_s,
            )
        )
        bridged_command = bridge_supported_controller_command(controller_command)
        actuator_record = bridge.convert(
            bridged_command,
            actuator_mapping,
            time_s=step_idx * sim_dt,
            command_index=step_idx,
        )
        _apply_actuator_record(robot, actuator_record, physical_model, device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        if realtime_playback:
            time.sleep(max(0.0, sim_dt))

        root_pose_after = _tensor_row(robot.data.root_pose_w.torch)
        tracked_pose_after = (
            list(_module_body_pose(robot, module_id=0, local_body_name="fc") or tuple(root_pose_after))
            if articulated_assembly
            else root_pose_after
        )
        root_pos = tracked_pose_after[:3]
        position_error = _position_error_norm(root_pos, target_pose[:3])
        attitude_error = _quat_error_norm(tracked_pose_after[3:7], target_pose[3:7])
        final_position_error = position_error
        final_attitude_error = attitude_error
        max_position_error = max(max_position_error, position_error)
        max_attitude_error = max(max_attitude_error, attitude_error)
        min_height = min(min_height, root_pos[2])
        max_height = max(max_height, root_pos[2])
        finite_state = finite_state and all(_is_finite(value) for value in root_pose_after)
        if articulated:
            for command_key, target in joint_targets.items():
                actuals = _joint_positions_for_command_key(
                    robot.joint_names,
                    robot.data.joint_pos.torch,
                    command_key=command_key,
                    module_count=module_count,
                )
                for actual in actuals:
                    observed_joint_count += 1
                    max_joint_target_abs = max(max_joint_target_abs, abs(float(target)))
                    max_joint_position_abs = max(max_joint_position_abs, abs(float(actual)))
                    max_joint_tracking_error = max(max_joint_tracking_error, abs(float(actual) - float(target)))
        ramp_complete = time_s + 1.0e-9 >= waypoint_ramp_duration_s
        if ramp_complete and position_error <= position_tolerance_m and attitude_error <= attitude_tolerance_rad:
            hold_steps += 1
        else:
            hold_steps = 0
        max_hold_steps = max(max_hold_steps, hold_steps)

        status = bridged_command.controller_status
        last_controller_status = status.to_dict()
        last_bridge_metrics = dict(actuator_record.metrics)
        if not status.qp_feasible:
            qp_infeasible_count += 1
        if status.metrics.get("clipped", 0.0) > 0.0:
            clipped_count += 1
        missing_actuator_count += len(actuator_record.missing_actuators)
        unsupported_actuator_count += len(actuator_record.unsupported_actuators)
        clipped_target_count += len(actuator_record.clipped_targets)
        previous_command = bridged_command
        motion_observed_for_stop = (
            not articulated
            or max_joint_position_abs + 1.0e-9 >= 0.5 * abs(float(articulated_joint_amplitude_rad))
        )
        if stop_on_hold and hold_steps >= hold_steps_required and motion_observed_for_stop:
            break

    hold_time_s = max_hold_steps * sim_dt
    expected_joint_motion = 0.5 * abs(float(articulated_joint_amplitude_rad))
    joint_motion_passed = (
        not articulated
        or (
            bool(articulated_joint_ids)
            and observed_joint_count > 0
            and max_joint_target_abs + 1.0e-9 >= expected_joint_motion
            and max_joint_position_abs + 1.0e-9 >= expected_joint_motion
            and max_joint_tracking_error <= articulated_joint_tracking_tolerance_rad
        )
    )
    module_motion_passed = (
        not articulated_assembly
        or max_relative_module_position_change >= 5.0e-3
        or max_relative_module_attitude_change >= 2.0e-2
    )
    model_update_passed = (
        not articulated_assembly
        or max_model_rotor_origin_change >= 5.0e-3
        or max_model_allocation_change >= 1.0e-3
    )
    passed = (
        executed_steps > 0
        and finite_state
        and qp_infeasible_count == 0
        and missing_actuator_count == 0
        and unsupported_actuator_count == 0
        and clipped_target_count == 0
        and final_position_error <= position_tolerance_m
        and final_attitude_error <= attitude_tolerance_rad
        and hold_time_s + 1.0e-9 >= hold_duration_s
        and joint_motion_passed
        and module_motion_passed
        and model_update_passed
    )
    return {
        f"{report_prefix}_smoke": True,
        f"{report_prefix}_smoke_passed": passed,
        f"{report_prefix}_target_pose": list(target_pose),
        f"{report_prefix}_ramp_duration_s": float(waypoint_ramp_duration_s),
        f"{report_prefix}_module_count": int(module_count),
        f"{report_prefix}_module_spacing_m": float(module_spacing_m),
        f"{report_prefix}_steps": int(executed_steps),
        f"{report_prefix}_requested_steps": int(max(0, steps)),
        f"{report_prefix}_duration_s": float(executed_steps * sim_dt),
        f"{report_prefix}_hold_time_s": hold_time_s,
        f"{report_prefix}_hold_required_s": hold_duration_s,
        f"{report_prefix}_stopped_on_hold": bool(stop_on_hold and executed_steps < max(0, steps)),
        f"{report_prefix}_position_tolerance_m": position_tolerance_m,
        f"{report_prefix}_attitude_tolerance_rad": attitude_tolerance_rad,
        f"{report_prefix}_final_position_error_m": final_position_error,
        f"{report_prefix}_final_attitude_error_rad": final_attitude_error,
        f"{report_prefix}_max_position_error_m": max_position_error,
        f"{report_prefix}_max_attitude_error_rad": max_attitude_error,
        f"{report_prefix}_min_height_m": min_height if executed_steps > 0 else None,
        f"{report_prefix}_max_height_m": max_height if executed_steps > 0 else None,
        f"{report_prefix}_finite_state": finite_state,
        f"{report_prefix}_qp_infeasible_count": qp_infeasible_count,
        f"{report_prefix}_controller_clipped_count": clipped_count,
        f"{report_prefix}_missing_actuator_count": missing_actuator_count,
        f"{report_prefix}_unsupported_actuator_count": unsupported_actuator_count,
        f"{report_prefix}_clipped_target_count": clipped_target_count,
        f"{report_prefix}_articulated": bool(articulated),
        f"{report_prefix}_articulated_joint_ids": list(articulated_joint_ids),
        f"{report_prefix}_articulated_joint_amplitude_rad": float(articulated_joint_amplitude_rad),
        f"{report_prefix}_articulated_joint_period_s": float(articulated_joint_period_s),
        f"{report_prefix}_articulated_joint_warmup_s": float(articulated_joint_warmup_s),
        f"{report_prefix}_articulated_joint_tracking_tolerance_rad": float(
            articulated_joint_tracking_tolerance_rad
        ),
        f"{report_prefix}_articulated_joint_motion_passed": bool(joint_motion_passed),
        f"{report_prefix}_articulated_assembly": bool(articulated_assembly),
        f"{report_prefix}_articulated_module_motion_passed": bool(module_motion_passed),
        f"{report_prefix}_articulated_model_update_passed": bool(model_update_passed),
        f"{report_prefix}_articulated_joint_observed_count": int(observed_joint_count),
        f"{report_prefix}_articulated_max_joint_target_abs_rad": float(max_joint_target_abs),
        f"{report_prefix}_articulated_max_joint_position_abs_rad": float(max_joint_position_abs),
        f"{report_prefix}_articulated_max_joint_tracking_error_rad": float(max_joint_tracking_error),
        f"{report_prefix}_articulated_max_relative_module_position_change_m": float(
            max_relative_module_position_change
        ),
        f"{report_prefix}_articulated_max_relative_module_attitude_change_rad": float(
            max_relative_module_attitude_change
        ),
        f"{report_prefix}_articulated_max_model_rotor_origin_change_m": float(max_model_rotor_origin_change),
        f"{report_prefix}_articulated_max_model_allocation_change": float(max_model_allocation_change),
        f"{report_prefix}_articulated_last_joint_targets": dict(last_joint_targets),
        f"{report_prefix}_last_controller_status": last_controller_status,
        f"{report_prefix}_last_bridge_metrics": last_bridge_metrics,
    }


def _run_p4_1_full_scene_backend_smoke(
    *,
    robot,
    p4_1_object,
    sim,
    sim_dt: float,
    physical_model,
    device: str,
    steps: int,
    module_count: int,
    module_spacing_m: float,
    module_poses: dict[int, tuple[float, float, float, float, float, float, float]] | None,
    object_id: str,
    target_height: float,
    control_dt_s: float,
    build_fixed_morphology,
    bridge_supported_controller_command,
    split_fixed_module_name,
    realtime_playback: bool,
    allocation_mode: str,
    uses_p2_p3: bool,
) -> dict[str, object]:
    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand
    from amsrr.simulation.p4_1_backend_smoke import evaluate_runtime_observation_joint_state

    morphology_graph = build_fixed_morphology(
        physical_model,
        graph_id="p4-1-full-scene-backend-smoke",
        module_count=module_count,
        module_spacing_m=module_spacing_m,
        module_poses=module_poses,
    )
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    bridge = IsaacControllerBridge()
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode=allocation_mode,
            control_dt_s=control_dt_s,
        )
    )
    target_pose = (0.0, 0.0, target_height, 0.0, 0.0, 0.0, 1.0)
    target_twist = [0.0] * 6
    previous_command = None
    runtime_observation_objects = []
    runtime_observations: list[dict[str, object]] = []
    controller_commands: list[dict[str, object]] = []
    actuator_target_records: list[dict[str, object]] = []
    object_pose_history: list[list[float]] = []
    qp_infeasible_count = 0
    clipped_count = 0
    missing_actuator_count = 0
    unsupported_actuator_count = 0
    clipped_target_count = 0
    finite_state = True
    executed_steps = 0
    last_controller_status = None
    last_bridge_metrics: dict[str, float] = {}

    for step_idx in range(max(0, steps)):
        executed_steps = step_idx + 1
        time_s = step_idx * sim_dt
        root_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
        root_twist = _tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch)
        runtime_observation = _build_fixed_runtime_observation(
            morphology_graph,
            time_s=time_s,
            root_pose_world=root_pose,  # type: ignore[arg-type]
            root_twist_world=root_twist,
            joint_names=robot.joint_names,
            joint_positions_tensor=robot.data.joint_pos.torch,
            joint_velocities_tensor=robot.data.joint_vel.torch,
            module_count=module_count,
            module_spacing_m=module_spacing_m,
            module_poses=module_poses,
            split_fixed_module_name=split_fixed_module_name,
        )
        object_state = _build_p4_1_object_runtime_state(p4_1_object, object_id=object_id)
        runtime_observation.object_states = [object_state]
        runtime_observation_objects.append(runtime_observation)
        runtime_observations.append(runtime_observation.to_dict())
        controller_command = controller.compute(
            ControllerContext(
                runtime_observation=runtime_observation,
                morphology_graph=morphology_graph,
                physical_model=physical_model,
                active_knot=InteractionKnot(t_rel_s=time_s, contact_assignments=[]),
                policy_command=PolicyCommand(
                    desired_body_pose=target_pose,
                    desired_body_twist=target_twist,
                ),
                previous_command=previous_command,
                control_dt_s=control_dt_s,
            )
        )
        bridged_command = bridge_supported_controller_command(controller_command)
        actuator_record = bridge.convert(
            bridged_command,
            actuator_mapping,
            time_s=time_s,
            command_index=step_idx,
        )
        controller_commands.append(bridged_command.to_dict())
        actuator_target_records.append(actuator_record.to_dict())
        _apply_actuator_record(robot, actuator_record, physical_model, device)
        robot.write_data_to_sim()
        _write_asset_data_to_sim(p4_1_object)
        sim.step()
        robot.update(sim_dt)
        p4_1_object.update(sim_dt)
        if realtime_playback:
            time.sleep(max(0.0, sim_dt))

        object_pose, object_twist = _p4_1_object_pose_and_twist(p4_1_object)
        object_pose_history.append(list(object_pose))
        finite_state = finite_state and all(_is_finite(value) for value in root_pose)
        finite_state = finite_state and all(_is_finite(value) for value in object_pose)
        finite_state = finite_state and all(_is_finite(value) for value in object_twist)

        status = bridged_command.controller_status
        last_controller_status = status.to_dict()
        last_bridge_metrics = dict(actuator_record.metrics)
        if not status.qp_feasible:
            qp_infeasible_count += 1
        if status.metrics.get("clipped", 0.0) > 0.0:
            clipped_count += 1
        missing_actuator_count += len(actuator_record.missing_actuators)
        unsupported_actuator_count += len(actuator_record.unsupported_actuators)
        clipped_target_count += len(actuator_record.clipped_targets)
        previous_command = bridged_command

    joint_state_metrics = evaluate_runtime_observation_joint_state(
        runtime_observation_objects,
        articulated_morphology=False,
    )
    logged_step_count_ok = (
        executed_steps > 0
        and len(runtime_observations) == executed_steps
        and len(controller_commands) == executed_steps
        and len(actuator_target_records) == executed_steps
        and len(object_pose_history) == executed_steps
    )
    object_pose_history_ok = all(len(pose) == 7 for pose in object_pose_history)
    passed = (
        executed_steps > 0
        and finite_state
        and logged_step_count_ok
        and object_pose_history_ok
        and joint_state_metrics.passed
        and qp_infeasible_count == 0
        and missing_actuator_count == 0
        and unsupported_actuator_count == 0
        and clipped_target_count == 0
    )
    return {
        "p4_1_full_scene_backend_smoke": True,
        "p4_1_full_scene_backend_smoke_passed": passed,
        "p4_1_full_scene_spawned": True,
        "p4_1_robot_spawned": True,
        "p4_1_object_spawned": True,
        "p4_1_floor_spawned": True,
        "p4_1_uses_p2_p3": bool(uses_p2_p3),
        "p4_1_articulated_morphology": False,
        "p4_1_module_count": int(module_count),
        "p4_1_module_spacing_m": float(module_spacing_m),
        "p4_1_steps": int(executed_steps),
        "p4_1_requested_steps": int(max(0, steps)),
        "p4_1_duration_s": float(executed_steps * sim_dt),
        "p4_1_runtime_observations": runtime_observations,
        "p4_1_controller_commands": controller_commands,
        "p4_1_actuator_target_records": actuator_target_records,
        "p4_1_object_pose_history": object_pose_history,
        "p4_1_runtime_observation_count": len(runtime_observations),
        "p4_1_controller_command_count": len(controller_commands),
        "p4_1_actuator_target_record_count": len(actuator_target_records),
        "p4_1_object_pose_count": len(object_pose_history),
        "p4_1_logged_step_count_ok": bool(logged_step_count_ok),
        "p4_1_object_pose_history_ok": bool(object_pose_history_ok),
        "p4_1_finite_state": bool(finite_state),
        "p4_1_qp_infeasible_count": int(qp_infeasible_count),
        "p4_1_controller_clipped_count": int(clipped_count),
        "p4_1_missing_actuator_count": int(missing_actuator_count),
        "p4_1_unsupported_actuator_count": int(unsupported_actuator_count),
        "p4_1_clipped_target_count": int(clipped_target_count),
        "p4_1_joint_state_preservation_passed": bool(joint_state_metrics.passed),
        "p4_1_joint_state_failure_reasons": list(joint_state_metrics.failure_reasons),
        "p4_1_module_state_count": int(joint_state_metrics.module_state_count),
        "p4_1_modules_with_pose": int(joint_state_metrics.modules_with_pose),
        "p4_1_modules_with_twist": int(joint_state_metrics.modules_with_twist),
        "p4_1_modules_with_joint_positions": int(joint_state_metrics.modules_with_joint_positions),
        "p4_1_modules_with_joint_velocities": int(joint_state_metrics.modules_with_joint_velocities),
        "p4_1_vectoring_joint_key_count": int(joint_state_metrics.vectoring_joint_key_count),
        "p4_1_dock_joint_key_count": int(joint_state_metrics.dock_joint_key_count),
        "p4_1_vectoring_joint_value_count": int(joint_state_metrics.vectoring_joint_value_count),
        "p4_1_dock_joint_value_count": int(joint_state_metrics.dock_joint_value_count),
        "p4_1_max_model_rotor_origin_change_m": 0.0,
        "p4_1_max_model_allocation_change": 0.0,
        "p4_1_last_controller_status": last_controller_status,
        "p4_1_last_bridge_metrics": last_bridge_metrics,
    }


def _run_p4_2_deterministic_rollout_probe(
    *,
    robot,
    p4_2_object,
    sim,
    sim_dt: float,
    physical_model,
    device: str,
    steps: int,
    morphology_graph,
    module_poses: dict[int, tuple[float, float, float, float, float, float, float]] | None,
    object_id: str,
    target_height: float,
    control_dt_s: float,
    bridge_supported_controller_command,
    split_fixed_module_name,
    realtime_playback: bool,
    allocation_mode: str,
    uses_p2_p3: bool,
    contact_model: str,
    attach_snap_distance_threshold_m: float,
) -> dict[str, object]:
    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand
    from amsrr.simulation.p4_2_rollout import (
        P4_2RolloutPhase,
        P4_2PhaseTransitionRecord,
        p4_2_failure_metrics,
        p4_2_no_mislabeling_artifacts,
    )

    module_count = len(morphology_graph.modules)
    graph_module_poses = module_poses or {
        module.module_id: module.pose_in_design_frame
        for module in morphology_graph.modules
    }
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    bridge = IsaacControllerBridge()
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode=allocation_mode,
            control_dt_s=control_dt_s,
        )
    )
    object_pose_initial, _ = _p4_1_object_pose_and_twist(p4_2_object)
    approach_pose = (
        float(object_pose_initial[0]) - 0.25,
        float(object_pose_initial[1]),
        max(float(target_height), float(object_pose_initial[2]) + 0.20),
        0.0,
        0.0,
        0.0,
        1.0,
    )
    target_twist = [0.0] * 6
    previous_command = None
    runtime_observation_objects = []
    runtime_observations: list[dict[str, object]] = []
    policy_commands: list[dict[str, object]] = []
    controller_commands: list[dict[str, object]] = []
    actuator_target_records: list[dict[str, object]] = []
    object_pose_history: list[list[float]] = []
    qp_infeasible_count = 0
    clipped_count = 0
    missing_actuator_count = 0
    unsupported_actuator_count = 0
    clipped_target_count = 0
    finite_state = True
    executed_steps = 0
    last_controller_status = None
    last_bridge_metrics: dict[str, float] = {}
    selected_contact_candidates_available = False
    phase_transitions = [
        P4_2PhaseTransitionRecord(
            from_phase=P4_2RolloutPhase.RESET,
            to_phase=P4_2RolloutPhase.APPROACH,
            time_s=0.0,
            phase_elapsed_s=0.0,
            reason="reset_complete_graph_specific_fixed_morphology_spawned",
            entry_condition_results={
                "p2_p3_design_claim": bool(uses_p2_p3),
                "morphology_graph_available": True,
            },
            exit_condition_results={
                "morphology_asset_reflected": True,
                "module_placement_reflected": len(graph_module_poses) == module_count,
                "actuator_mapping_reflected": len(actuator_mapping.channels) > 0,
            },
        )
    ]

    for step_idx in range(max(0, steps)):
        executed_steps = step_idx + 1
        time_s = step_idx * sim_dt
        root_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
        root_twist = _tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch)
        runtime_observation = _build_fixed_runtime_observation(
            morphology_graph,
            time_s=time_s,
            root_pose_world=root_pose,  # type: ignore[arg-type]
            root_twist_world=root_twist,
            joint_names=robot.joint_names,
            joint_positions_tensor=robot.data.joint_pos.torch,
            joint_velocities_tensor=robot.data.joint_vel.torch,
            module_count=module_count,
            module_spacing_m=0.45,
            module_poses=graph_module_poses,
            split_fixed_module_name=split_fixed_module_name,
        )
        object_state = _build_p4_1_object_runtime_state(p4_2_object, object_id=object_id)
        runtime_observation.object_states = [object_state]
        runtime_observation.task_progress.phase_label = P4_2RolloutPhase.APPROACH.value
        runtime_observation.task_progress.metrics.update(
            {
                "selected_contact_candidates_available": 0.0,
                "unconditional_attach_allowed": 0.0,
            }
        )
        runtime_observation_objects.append(runtime_observation)
        runtime_observations.append(runtime_observation.to_dict())
        policy_command = PolicyCommand(
            desired_body_pose=approach_pose,
            desired_body_twist=target_twist,
            priority_weights={
                "p4_2_phase_approach": 1.0,
                "attach_condition_gate": 0.0,
            },
        )
        active_knot = InteractionKnot(
            t_rel_s=time_s,
            contact_assignments=[],
            priority_weights={"p4_2_phase_approach": 1.0},
            guard_conditions=[
                {
                    "type": "p4_2_phase",
                    "phase": P4_2RolloutPhase.APPROACH.value,
                    "contact_model": contact_model,
                },
                {
                    "type": "p4_2_attach_gate",
                    "selected_contact_candidates_available": False,
                    "robot_anchors_available": len(morphology_graph.robot_anchors) > 0,
                    "unconditional_attach_allowed": False,
                    "attach_snap_distance_threshold_m": float(attach_snap_distance_threshold_m),
                },
            ],
        )
        policy_commands.append(policy_command.to_dict())
        controller_command = controller.compute(
            ControllerContext(
                runtime_observation=runtime_observation,
                morphology_graph=morphology_graph,
                physical_model=physical_model,
                active_knot=active_knot,
                policy_command=policy_command,
                previous_command=previous_command,
                control_dt_s=control_dt_s,
            )
        )
        bridged_command = bridge_supported_controller_command(controller_command)
        actuator_record = bridge.convert(
            bridged_command,
            actuator_mapping,
            time_s=time_s,
            command_index=step_idx,
        )
        controller_commands.append(bridged_command.to_dict())
        actuator_target_records.append(actuator_record.to_dict())
        _apply_actuator_record(robot, actuator_record, physical_model, device)
        robot.write_data_to_sim()
        _write_asset_data_to_sim(p4_2_object)
        sim.step()
        robot.update(sim_dt)
        p4_2_object.update(sim_dt)
        if realtime_playback:
            time.sleep(max(0.0, sim_dt))

        object_pose, object_twist = _p4_1_object_pose_and_twist(p4_2_object)
        object_pose_history.append(list(object_pose))
        finite_state = finite_state and all(_is_finite(value) for value in root_pose)
        finite_state = finite_state and all(_is_finite(value) for value in object_pose)
        finite_state = finite_state and all(_is_finite(value) for value in object_twist)

        status = bridged_command.controller_status
        last_controller_status = status.to_dict()
        last_bridge_metrics = dict(actuator_record.metrics)
        if not status.qp_feasible:
            qp_infeasible_count += 1
        if status.metrics.get("clipped", 0.0) > 0.0:
            clipped_count += 1
        missing_actuator_count += len(actuator_record.missing_actuators)
        unsupported_actuator_count += len(actuator_record.unsupported_actuators)
        clipped_target_count += len(actuator_record.clipped_targets)
        previous_command = bridged_command

    controller_terminal = qp_infeasible_count > 0
    final_phase = (
        P4_2RolloutPhase.CONTROLLER_FAILURE
        if controller_terminal
        else P4_2RolloutPhase.TIMEOUT_FAILURE
    )
    phase_transitions.append(
        P4_2PhaseTransitionRecord(
            from_phase=P4_2RolloutPhase.APPROACH,
            to_phase=final_phase,
            time_s=float(executed_steps * sim_dt),
            phase_elapsed_s=float(executed_steps * sim_dt),
            reason=(
                "controller_or_qp_infeasible"
                if controller_terminal
                else "selected_contact_candidate_gate_not_available_before_probe_end"
            ),
            entry_condition_results={
                "selected_contact_candidates_available": selected_contact_candidates_available,
                "unconditional_attach_allowed": False,
            },
            exit_condition_results={
                "attach_attempt_not_entered": True,
                "object_attach_event_recorded": False,
            },
            timeout_s=float(executed_steps * sim_dt) if executed_steps > 0 else None,
        )
    )
    logged_step_count_ok = (
        len(runtime_observations) == executed_steps
        and len(policy_commands) == executed_steps
        and len(controller_commands) == executed_steps
        and len(actuator_target_records) == executed_steps
        and len(object_pose_history) == executed_steps
    )
    object_pose_history_ok = all(len(pose) == 7 for pose in object_pose_history)
    module_placement_reflected = len(graph_module_poses) == module_count
    actuator_mapping_reflected = len(actuator_mapping.channels) > 0 and actuator_mapping.graph_id == morphology_graph.graph_id
    metrics = p4_2_failure_metrics(
        final_phase=final_phase,
        controller_qp_infeasible_terminal=controller_terminal,
    )
    metrics.update(
        {
            "p4_2_runtime_observation_count": float(len(runtime_observations)),
            "p4_2_policy_command_count": float(len(policy_commands)),
            "p4_2_controller_command_count": float(len(controller_commands)),
            "p4_2_actuator_target_record_count": float(len(actuator_target_records)),
            "p4_2_selected_contact_candidates_available": 0.0,
            "p4_2_unconditional_attach_allowed": 0.0,
            "p4_2_attach_snap_distance_threshold_m": float(attach_snap_distance_threshold_m),
            "p4_2_attach_event_count": 0.0,
            "p4_2_actuator_channel_count": float(len(actuator_mapping.channels)),
        }
    )
    artifacts = p4_2_no_mislabeling_artifacts()
    artifacts.update(
        {
            "object_attach_release_only": True,
            "module_attach_detach_claim": False,
            "dynamic_morphology_update_claim": False,
            "asset_generation_semantics": "reset_time_fixed_morphology_not_pi_a_dynamic_construction",
        }
    )
    return {
        "p4_2_deterministic_rollout": True,
        "p4_2_deterministic_rollout_passed": False,
        "p4_2_contact_model": contact_model,
        "p4_2_final_phase": final_phase.value,
        "p4_2_uses_p2_p3": bool(uses_p2_p3),
        "p4_2_graph_id": morphology_graph.graph_id,
        "p4_2_module_count": int(module_count),
        "p4_2_module_ids": [module.module_id for module in morphology_graph.modules],
        "p4_2_dock_edge_count": len(morphology_graph.dock_edges),
        "p4_2_robot_anchor_count": len(morphology_graph.robot_anchors),
        "p4_2_morphology_asset_reflected": True,
        "p4_2_module_placement_reflected": bool(module_placement_reflected),
        "p4_2_module_poses_source": "p3_assembled_morphology_graph",
        "p4_2_module_poses": {str(module_id): list(pose) for module_id, pose in sorted(graph_module_poses.items())},
        "p4_2_actuator_mapping_reflected": bool(actuator_mapping_reflected),
        "p4_2_actuator_mapping_graph_id": actuator_mapping.graph_id,
        "p4_2_actuator_channel_count": len(actuator_mapping.channels),
        "p4_2_object_attach_release_only": True,
        "p4_2_module_attach_detach_claim": False,
        "p4_2_dynamic_morphology_update_claim": False,
        "p4_2_asset_generation_semantics": "reset_time_fixed_morphology_not_pi_a_dynamic_construction",
        "p4_2_attach_gate_input_available": False,
        "p4_2_unconditional_attach_allowed": False,
        "p4_2_selected_contact_candidate_count": 0,
        "p4_2_runtime_observations": runtime_observations,
        "p4_2_policy_commands": policy_commands,
        "p4_2_controller_commands": controller_commands,
        "p4_2_actuator_target_records": actuator_target_records,
        "p4_2_phase_transitions": [transition.to_dict() for transition in phase_transitions],
        "p4_2_attach_events": [],
        "p4_2_object_pose_history": object_pose_history,
        "p4_2_runtime_observation_count": len(runtime_observations),
        "p4_2_policy_command_count": len(policy_commands),
        "p4_2_controller_command_count": len(controller_commands),
        "p4_2_actuator_target_record_count": len(actuator_target_records),
        "p4_2_object_pose_count": len(object_pose_history),
        "p4_2_logged_step_count_ok": bool(logged_step_count_ok),
        "p4_2_object_pose_history_ok": bool(object_pose_history_ok),
        "p4_2_finite_state": bool(finite_state),
        "p4_2_qp_infeasible_count": int(qp_infeasible_count),
        "p4_2_controller_clipped_count": int(clipped_count),
        "p4_2_missing_actuator_count": int(missing_actuator_count),
        "p4_2_unsupported_actuator_count": int(unsupported_actuator_count),
        "p4_2_clipped_target_count": int(clipped_target_count),
        "p4_2_last_controller_status": last_controller_status,
        "p4_2_last_bridge_metrics": last_bridge_metrics,
        "p4_2_rollout_artifacts": artifacts,
        **metrics,
    }


def _build_p4_1_object_runtime_state(p4_1_object, *, object_id: str):
    from amsrr.schemas.runtime import ObjectRuntimeState

    pose_world, twist_world = _p4_1_object_pose_and_twist(p4_1_object)
    return ObjectRuntimeState(
        object_id=object_id,
        pose_world=pose_world,
        twist_world=twist_world,
    )


def _p4_1_object_pose_and_twist(p4_1_object):
    data = p4_1_object.data
    if hasattr(data, "root_pose_w"):
        pose_values = _tensor_row(_isaac_tensor(data.root_pose_w))
        pose_world = tuple(float(value) for value in pose_values[:7])
    else:
        root_pos = _tensor_row(_isaac_tensor(data.root_pos_w))
        root_quat = _tensor_row(_isaac_tensor(data.root_quat_w))
        pose_world = tuple(float(value) for value in [*root_pos[:3], *root_quat[:4]])

    if hasattr(data, "root_lin_vel_w") and hasattr(data, "root_ang_vel_w"):
        linear = _tensor_row(_isaac_tensor(data.root_lin_vel_w))
        angular = _tensor_row(_isaac_tensor(data.root_ang_vel_w))
        twist_world = [float(value) for value in [*linear[:3], *angular[:3]]]
    elif hasattr(data, "root_vel_w"):
        velocity_values = _tensor_row(_isaac_tensor(data.root_vel_w))
        twist_world = [float(value) for value in (velocity_values + [0.0] * 6)[:6]]
    else:
        twist_world = [0.0] * 6
    return pose_world, twist_world


def _isaac_tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _write_asset_data_to_sim(asset) -> None:
    if hasattr(asset, "write_data_to_sim"):
        asset.write_data_to_sim()


def _build_fixed_runtime_observation(
    morphology_graph,
    *,
    time_s: float,
    root_pose_world: tuple[float, float, float, float, float, float, float],
    root_twist_world: list[float],
    joint_names: list[str],
    joint_positions_tensor,
    joint_velocities_tensor,
    module_count: int,
    module_spacing_m: float,
    module_poses: dict[int, tuple[float, float, float, float, float, float, float]] | None,
    split_fixed_module_name,
):
    from amsrr.schemas.policies import ControllerStatus
    from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState

    joint_positions = _fixed_module_joint_state_dicts(
        module_count,
        joint_names,
        joint_positions_tensor,
        split_fixed_module_name=split_fixed_module_name,
    )
    joint_velocities = _fixed_module_joint_state_dicts(
        module_count,
        joint_names,
        joint_velocities_tensor,
        split_fixed_module_name=split_fixed_module_name,
    )
    root_twist = (list(root_twist_world) + [0.0] * 6)[:6]
    module_states = []
    for module_id in range(module_count):
        relative_pose = (
            module_poses[module_id]
            if module_poses is not None and module_id in module_poses
            else (module_spacing_m * module_id, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )
        offset_root = tuple(float(value) for value in relative_pose[:3])
        pose_world = compose_pose(root_pose_world, relative_pose)
        twist_world = _fixed_module_twist(root_pose_world, root_twist, offset_root)
        module_states.append(
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=pose_world,
                twist_world=twist_world,
                joint_positions=joint_positions[module_id],
                joint_velocities=joint_velocities[module_id],
            )
        )
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology_graph,
        module_states=module_states,
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )


def _build_articulated_runtime_observation(
    morphology_graph,
    *,
    time_s: float,
    robot,
    root_pose_world: tuple[float, float, float, float, float, float, float],
    root_twist_world: list[float],
    joint_names: list[str],
    joint_positions_tensor,
    joint_velocities_tensor,
    module_count: int,
    split_fixed_module_name,
):
    from amsrr.schemas.policies import ControllerStatus
    from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState

    joint_positions = _fixed_module_joint_state_dicts(
        module_count,
        joint_names,
        joint_positions_tensor,
        split_fixed_module_name=split_fixed_module_name,
    )
    joint_velocities = _fixed_module_joint_state_dicts(
        module_count,
        joint_names,
        joint_velocities_tensor,
        split_fixed_module_name=split_fixed_module_name,
    )
    root_twist = (list(root_twist_world) + [0.0] * 6)[:6]
    module_states = []
    for module_id in range(module_count):
        pose_world = _module_body_pose(robot, module_id=module_id, local_body_name="fc")
        twist_world = _module_body_twist(robot, module_id=module_id, local_body_name="fc")
        if pose_world is None:
            pose_world = root_pose_world
        if twist_world is None:
            twist_world = root_twist
        module_states.append(
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=pose_world,
                twist_world=twist_world,
                joint_positions=joint_positions[module_id],
                joint_velocities=joint_velocities[module_id],
            )
        )
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology_graph,
        module_states=module_states,
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )


def _ramped_target_pose(
    start_pose: tuple[float, float, float, float, float, float, float],
    final_pose: tuple[float, float, float, float, float, float, float],
    *,
    elapsed_s: float,
    ramp_duration_s: float,
) -> tuple[float, float, float, float, float, float, float]:
    if ramp_duration_s <= 0.0:
        return final_pose
    ratio = min(max(float(elapsed_s) / float(ramp_duration_s), 0.0), 1.0)
    smooth = ratio * ratio * (3.0 - 2.0 * ratio)
    position = tuple(
        float(start_pose[idx]) + (float(final_pose[idx]) - float(start_pose[idx])) * smooth
        for idx in range(3)
    )
    return (
        position[0],
        position[1],
        position[2],
        final_pose[3],
        final_pose[4],
        final_pose[5],
        final_pose[6],
    )


def _fixed_module_joint_state_dicts(
    module_count: int,
    joint_names: list[str],
    tensor,
    *,
    split_fixed_module_name,
) -> list[dict[str, float]]:
    values = tensor[0].detach().cpu().tolist()
    states = [dict() for _ in range(module_count)]
    for index, joint_name in enumerate(joint_names):
        parsed = split_fixed_module_name(joint_name)
        if parsed is None:
            if module_count == 1:
                states[0][joint_name] = float(values[index])
            continue
        module_id, local_name = parsed
        if 0 <= module_id < module_count:
            states[module_id][local_name] = float(values[index])
    return states


def _fixed_module_pose(
    root_pose_world: tuple[float, float, float, float, float, float, float],
    offset_root: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    rotation = _quat_to_matrix(tuple(root_pose_world[3:7]))
    offset_world = _matvec(rotation, offset_root)
    return (
        float(root_pose_world[0]) + offset_world[0],
        float(root_pose_world[1]) + offset_world[1],
        float(root_pose_world[2]) + offset_world[2],
        float(root_pose_world[3]),
        float(root_pose_world[4]),
        float(root_pose_world[5]),
        float(root_pose_world[6]),
    )


def _fixed_module_twist(
    root_pose_world: tuple[float, float, float, float, float, float, float],
    root_twist_world: list[float],
    offset_root: tuple[float, float, float],
) -> list[float]:
    rotation = _quat_to_matrix(tuple(root_pose_world[3:7]))
    offset_world = _matvec(rotation, offset_root)
    linear = tuple(float(value) for value in root_twist_world[:3])
    angular = tuple(float(value) for value in root_twist_world[3:6])
    linear_at_module = _add3(linear, _cross3(angular, offset_world))
    return [*linear_at_module, *angular]


def _yaw_quat_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _tensor_row(tensor, *, limit: int | None = None) -> list[float]:
    row = tensor[0]
    if limit is not None:
        row = row[:limit]
    return [float(value) for value in row.detach().cpu().tolist()]


def _tensor_indices(tensor, indices: list[int]) -> list[float]:
    if not indices:
        return []
    row = tensor[0, indices]
    return [float(value) for value in row.detach().cpu().tolist()]


def _joint_state_dict(joint_names: list[str], tensor) -> dict[str, float]:
    values = tensor[0].detach().cpu().tolist()
    return {name: float(values[index]) for index, name in enumerate(joint_names)}


def _keep_viewer_open(duration_s: float) -> None:
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline and simulation_app.is_running():
        simulation_app.update()
        time.sleep(1.0 / 60.0)


def _position_error_norm(position: list[float], target: tuple[float, float, float]) -> float:
    return sum((float(position[idx]) - float(target[idx])) ** 2 for idx in range(3)) ** 0.5


def _quat_error_norm(current_xyzw: list[float], target_xyzw: tuple[float, float, float, float]) -> float:
    cx, cy, cz, cw = _normalize_quat(tuple(float(value) for value in current_xyzw))
    tx, ty, tz, tw = _normalize_quat(target_xyzw)
    ex, ey, ez, ew = _quat_multiply((-cx, -cy, -cz, cw), (tx, ty, tz, tw))
    if ew < 0.0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    vector_norm = (ex * ex + ey * ey + ez * ez) ** 0.5
    if vector_norm <= 1.0e-12:
        return 0.0
    return 2.0 * math.atan2(vector_norm, max(min(ew, 1.0), -1.0))


def _normalize_quat(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sum(value * value for value in quat_xyzw) ** 0.5
    if norm <= 0.0:
        raise RuntimeError("Cannot normalize zero quaternion")
    return tuple(float(value) / norm for value in quat_xyzw)  # type: ignore[return-value]


def _quat_multiply(
    left_xyzw: tuple[float, float, float, float],
    right_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left_xyzw
    rx, ry, rz, rw = right_xyzw
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_to_matrix(quat_xyzw: tuple[float, float, float, float]):
    x, y, z, w = _normalize_quat(quat_xyzw)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _matvec(matrix, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _add3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _cross3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _apply_actuator_record(robot, actuator_record, physical_model, device: str) -> None:
    import torch

    rotors_by_id = {rotor.rotor_id: rotor for rotor in physical_model.rotors}
    force_body_ids: list[int] = []
    force_rows: list[list[float]] = []
    joint_ids: list[int] = []
    joint_targets: list[float] = []

    for target in actuator_record.actuator_targets:
        local_id = str(target.metadata.get("local_id", target.command_key))
        module_id = int(target.metadata.get("module_id", 0))
        if target.actuator_type == "rotor_thrust":
            rotor = rotors_by_id.get(local_id)
            body_name = _resolve_module_name(robot.body_names, module_id, local_id)
            if rotor is None or body_name is None:
                continue
            force_body_ids.append(robot.body_names.index(body_name))
            force_rows.append([float(axis) * target.target_value for axis in rotor.thrust_axis_local])
        elif target.actuator_type in {"vectoring_joint_position", "dock_joint_position"}:
            joint_name = _resolve_module_name(robot.joint_names, module_id, local_id)
            if joint_name is None:
                continue
            joint_ids.append(robot.joint_names.index(joint_name))
            joint_targets.append(target.target_value)

    if force_body_ids:
        forces = torch.tensor([force_rows], dtype=torch.float32, device=device)
        torques = torch.zeros_like(forces)
        body_ids = torch.tensor(force_body_ids, dtype=torch.int32, device=device)
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=False,
        )
    if joint_ids:
        target_tensor = torch.tensor([joint_targets], dtype=torch.float32, device=device)
        joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.int32, device=device)
        robot.set_joint_position_target_index(target=target_tensor, joint_ids=joint_ids_tensor)


def _resolve_module_name(names: list[str], module_id: int, local_id: str) -> str | None:
    prefixed = f"module_{module_id}__{local_id}"
    if prefixed in names:
        return prefixed
    if local_id in names:
        return local_id
    suffix = f"__{local_id}"
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def _controller_bundle_report(controller_bundle) -> dict[str, object]:
    record = controller_bundle.actuator_target_record
    command = controller_bundle.controller_command
    return {
        "controller_status": command.controller_status.to_dict(),
        "controller_rotor_thrusts_n": dict(command.rotor_thrusts_n),
        "controller_vectoring_joint_targets": dict(command.vectoring_joint_targets),
        "controller_dock_mechanism_commands": dict(command.dock_mechanism_commands),
        "controller_bridge_metrics": dict(record.metrics),
        "controller_bridge_missing_actuators": list(record.missing_actuators),
        "controller_bridge_unsupported_actuators": list(record.unsupported_actuators),
        "controller_bridge_clipped_targets": list(record.clipped_targets),
        "controller_smoke_metrics": dict(controller_bundle.metrics),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
