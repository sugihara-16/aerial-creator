from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--hover-target-height", type=float, default=None, help="Closed-loop hover target z in meters.")
    parser.add_argument("--hover-position-tolerance-m", type=float, default=0.20, help="Closed-loop hover position tolerance.")
    parser.add_argument("--hover-attitude-tolerance-rad", type=float, default=0.25, help="Closed-loop hover attitude tolerance.")
    parser.add_argument("--hover-hold-duration-s", type=float, default=1.0, help="Required final hold duration for hover pass.")
    parser.add_argument(
        "--hover-stop-on-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop closed-loop hover smoke once the required hold duration is achieved.",
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
    for key in ("command_probe_passed", "single_module_hover_smoke_passed"):
        if report.get(key) is False:
            return 1
    return 0


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    import torch

    from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
    from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
    from amsrr.simulation.p4_control_controller_smoke import (
        bridge_supported_controller_command,
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
                stiffness=args.dock_stiffness,
                damping=args.dock_damping,
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
    controller_bundle = None
    hover_smoke_report = None
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
    if args.single_module_hover_smoke:
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
        )
    for _ in range(0 if hover_smoke_report is not None else max(0, args.steps)):
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
        hover_command_ok = bool(hover_smoke_report.get("single_module_hover_smoke_passed"))

    report = {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": command_applied or controller_bundle is not None or hover_smoke_report is not None,
        "command_probe_passed": (
            force_command_ok and gimbal_command_ok and controller_command_ok and hover_command_ok
            if command_applied or controller_bundle is not None or hover_smoke_report is not None
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
    sim.stop()
    sim.clear_instance()
    return report


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


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
) -> dict[str, object]:
    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand

    morphology_graph = build_single_module_morphology(
        physical_model,
        graph_id="single-module-hover-smoke",
    )
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    bridge = IsaacControllerBridge()
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_qp",
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

    for step_idx in range(max(0, steps)):
        executed_steps = step_idx + 1
        runtime_observation = build_runtime_observation(
            morphology_graph,
            time_s=step_idx * sim_dt,
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
                active_knot=InteractionKnot(t_rel_s=step_idx * sim_dt, contact_assignments=[]),
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
        if stop_on_hold and hold_steps >= hold_steps_required:
            break

    hold_time_s = max_hold_steps * sim_dt
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
    )
    return {
        "single_module_hover_smoke": True,
        "single_module_hover_smoke_passed": passed,
        "single_module_hover_target_pose": list(target_pose),
        "single_module_hover_steps": int(executed_steps),
        "single_module_hover_requested_steps": int(max(0, steps)),
        "single_module_hover_duration_s": float(executed_steps * sim_dt),
        "single_module_hover_hold_time_s": hold_time_s,
        "single_module_hover_hold_required_s": hold_duration_s,
        "single_module_hover_stopped_on_hold": bool(stop_on_hold and executed_steps < max(0, steps)),
        "single_module_hover_position_tolerance_m": position_tolerance_m,
        "single_module_hover_attitude_tolerance_rad": attitude_tolerance_rad,
        "single_module_hover_final_position_error_m": final_position_error,
        "single_module_hover_final_attitude_error_rad": final_attitude_error,
        "single_module_hover_max_position_error_m": max_position_error,
        "single_module_hover_max_attitude_error_rad": max_attitude_error,
        "single_module_hover_min_height_m": min_height if executed_steps > 0 else None,
        "single_module_hover_max_height_m": max_height if executed_steps > 0 else None,
        "single_module_hover_finite_state": finite_state,
        "single_module_hover_qp_infeasible_count": qp_infeasible_count,
        "single_module_hover_controller_clipped_count": clipped_count,
        "single_module_hover_missing_actuator_count": missing_actuator_count,
        "single_module_hover_unsupported_actuator_count": unsupported_actuator_count,
        "single_module_hover_clipped_target_count": clipped_target_count,
        "single_module_hover_last_controller_status": last_controller_status,
        "single_module_hover_last_bridge_metrics": last_bridge_metrics,
    }


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
        if target.actuator_type == "rotor_thrust":
            rotor = rotors_by_id.get(local_id)
            if rotor is None or local_id not in robot.body_names:
                continue
            force_body_ids.append(robot.body_names.index(local_id))
            force_rows.append([float(axis) * target.target_value for axis in rotor.thrust_axis_local])
        elif target.actuator_type in {"vectoring_joint_position", "dock_joint_position"}:
            if local_id not in robot.joint_names:
                continue
            joint_ids.append(robot.joint_names.index(local_id))
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
