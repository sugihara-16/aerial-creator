from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


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
    return 0 if report.get("spawn_passed") else 1


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    import torch

    from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config

    backend_config = load_isaac_lab_backend_config(args.config)
    backend = IsaacLabBackend(backend_config)
    urdf_path = _expand_path(backend_config.holon_urdf_path)
    usd_dir = _expand_path(args.generated_usd_dir or backend_config.generated_usd_dir)
    usd_path = _expand_path(args.generated_usd_path or backend_config.generated_usd_path)
    converted = False

    if args.force_convert or (args.convert_if_missing and not usd_path.exists()):
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
                joint_names_expr=["gimbal.*"],
                stiffness=args.gimbal_stiffness,
                damping=args.gimbal_damping,
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=0.0,
                damping=0.0,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=["rotor.*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()
    thrust_body_ids, thrust_body_names = robot.find_bodies("thrust_.*")
    gimbal_joint_ids, gimbal_joint_names = robot.find_joints("gimbal.*")
    robot_mass = float(robot.data.body_mass.torch[0].sum().detach().cpu())
    gravity = float(torch.tensor(sim.cfg.gravity, device=sim.device).norm().detach().cpu())
    force_per_rotor_n = float(args.force_per_rotor_n)
    if args.hover_force_scale is not None:
        if not thrust_body_ids:
            raise RuntimeError("Cannot compute hover force without thrust bodies.")
        force_per_rotor_n = robot_mass * gravity * float(args.hover_force_scale) / len(thrust_body_ids)
    command_applied = force_per_rotor_n != 0.0 or args.gimbal_target_rad != 0.0
    if force_per_rotor_n != 0.0 and len(thrust_body_ids) != 4:
        raise RuntimeError(f"Expected 4 thrust bodies, found {len(thrust_body_ids)}: {thrust_body_names}")
    if args.gimbal_target_rad != 0.0 and not gimbal_joint_ids:
        raise RuntimeError("Cannot command gimbal target without gimbal joints.")

    thrust_body_ids_tensor = torch.tensor(thrust_body_ids, dtype=torch.int32, device=sim.device)
    gimbal_joint_ids_tensor = torch.tensor(gimbal_joint_ids, dtype=torch.int32, device=sim.device)
    initial_root_pos_w = _tensor_row(robot.data.root_pos_w.torch)
    for _ in range(max(0, args.steps)):
        if force_per_rotor_n != 0.0:
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

    report = {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": command_applied,
        "command_probe_passed": force_command_ok and gimbal_command_ok if command_applied else None,
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
        "initial_root_pos_w": initial_root_pos_w,
        "root_pos_w": final_root_pos_w,
        "root_delta_w": [final - initial for final, initial in zip(final_root_pos_w, initial_root_pos_w, strict=True)],
        "root_quat_w": _tensor_row(robot.data.root_quat_w.torch),
        "root_lin_vel_w": _tensor_row(robot.data.root_lin_vel_w.torch),
        "root_ang_vel_w": _tensor_row(robot.data.root_ang_vel_w.torch),
        "joint_pos_sample": _tensor_row(robot.data.joint_pos.torch, limit=8),
    }
    sim.stop()
    sim.clear_instance()
    return report


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
