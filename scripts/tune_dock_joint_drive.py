from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune AK40-10 Dock-joint implicit-drive Kp/Kd in one fixed-base, "
            "contact-free Holon articulation."
        )
    )
    parser.add_argument("--backend-config", default="configs/env/isaac_lab.yaml")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/p4_full/order8_natural_contact/joint_drive_tuning/"
            "dock_joint_drive_tuning_v1.json"
        ),
    )
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--step-amplitude-rad", type=float, default=0.01)
    parser.add_argument("--disturbance-torque-nm", type=float, default=1.20)
    parser.add_argument(
        "--coarse-kp",
        type=float,
        nargs="+",
        default=(75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 650.0),
    )
    parser.add_argument(
        "--coarse-kd",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.5, 5.0, 8.0, 12.0, 20.0, 30.0),
    )
    parser.add_argument(
        "--fine-multipliers",
        type=float,
        nargs="+",
        default=(0.75, 0.875, 1.0, 1.125, 1.25),
    )
    parser.add_argument("--skip-fine", action="store_true")
    parser.add_argument(
        "--maximum-candidates",
        type=int,
        default=None,
        help="Debug-only cap applied before the baseline candidate is restored.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> int:
    output_path = _path(args_cli.output)
    started = time.monotonic()
    try:
        report = run_tuning(args_cli)
        exit_code = 0
    except Exception as error:  # pragma: no cover - real Isaac failure path.
        report = {
            "version": "dock_joint_drive_tuning_v1",
            "attempted": True,
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc(limit=12),
        }
        exit_code = 1
    report["wall_elapsed_s"] = time.monotonic() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_summary(report, output_path), sort_keys=True))
    return exit_code


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sim import SimulationContext
    import torch

    from amsrr.robot_model.physical_model_builder import (
        build_physical_model_from_config,
    )
    from amsrr.simulation.dock_joint_drive_tuning import (
        DOCK_JOINT_DRIVE_TUNING_DEPLOYMENT_GATE,
        DOCK_JOINT_DRIVE_TUNING_METHOD,
        DOCK_JOINT_DRIVE_TUNING_SELECTION_SCOPE,
        DOCK_JOINT_DRIVE_TUNING_VERSION,
        DockJointDriveSample,
        DockJointDriveTuningConfig,
        coarse_gain_candidates,
        evaluate_gain_candidate,
        fine_gain_candidates,
        select_best_gain_candidate,
    )
    from amsrr.simulation.isaac_lab_backend import load_isaac_lab_backend_config
    from amsrr.utils.hashing import hash_file

    backend_config = load_isaac_lab_backend_config(str(_path(args.backend_config)))
    physical_model = build_physical_model_from_config(
        backend_config.robot_model_config_path
    )
    actuator_specs = physical_model.metadata.get("joint_actuator_specs", {})
    dock_spec = actuator_specs.get("dock", {})
    vectoring_spec = actuator_specs.get("vectoring", {})
    dock_drive = dock_spec.get("simulation_drive", {})
    vectoring_drive = vectoring_spec.get("simulation_drive", {})
    effort_limit_nm = float(dock_spec["peak_torque_nm"])
    peak_current_a = float(dock_spec["peak_current_a"])
    velocity_limit_rad_s = float(dock_drive["safe_velocity_limit_rad_s"])
    armature_kg_m2 = float(dock_drive.get("armature_kg_m2", 0.0))
    baseline_kp = float(dock_drive["stiffness"])
    baseline_kd = float(dock_drive["damping"])
    config = DockJointDriveTuningConfig(
        simulation_dt_s=float(args.dt),
        step_amplitude_rad=float(args.step_amplitude_rad),
        disturbance_torque_nm=float(args.disturbance_torque_nm),
        effort_limit_nm=effort_limit_nm,
        peak_current_a=peak_current_a,
        velocity_limit_rad_s=velocity_limit_rad_s,
        coarse_kp_values=tuple(float(value) for value in args.coarse_kp),
        coarse_kd_values=tuple(float(value) for value in args.coarse_kd),
        fine_multipliers=tuple(float(value) for value in args.fine_multipliers),
    )
    config.validate()
    if args.maximum_candidates is not None and int(args.maximum_candidates) <= 0:
        raise ValueError("--maximum-candidates must be positive")

    usd_path = _path(backend_config.generated_usd_path)
    urdf_path = _path(backend_config.holon_urdf_path)
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"generated Holon USD is missing: {usd_path}; regenerate it before tuning"
        )
    dock_joint_specs = tuple(
        joint
        for joint in physical_model.joints
        if joint.joint_type != "fixed" and "dock_mech_joint" in joint.joint_id
    )
    if len(dock_joint_specs) != 4:
        raise RuntimeError(
            f"expected four Dock mechanism joints, found {len(dock_joint_specs)}"
        )

    sim_utils.create_new_stage()
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=config.simulation_dt_s, device=args.device)
    )
    robot_cfg = ArticulationCfg(
        prim_path="/World/DockJointTuning/Holon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.1,
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
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "gimbal_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*gimbal.*"],
                stiffness=float(vectoring_drive["stiffness"]),
                damping=float(vectoring_drive["damping"]),
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=baseline_kp,
                damping=baseline_kd,
                armature=armature_kg_m2,
                effort_limit_sim=effort_limit_nm,
                velocity_limit_sim=velocity_limit_rad_s,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    fixed_root_body_path = _preauthor_fixed_root_constraint(
        stage=sim.stage,
        articulation_root_path="/World/DockJointTuning/Holon",
    )
    sim.reset()
    robot.update(config.simulation_dt_s)

    dock_joint_names = tuple(joint.joint_id for joint in dock_joint_specs)
    dock_joint_ids = tuple(robot.joint_names.index(name) for name in dock_joint_names)
    load_body_names = tuple(joint.child_link for joint in dock_joint_specs)
    load_body_ids = tuple(robot.body_names.index(name) for name in load_body_names)
    load_axes_local = tuple(_unit_vector(joint.axis_xyz) for joint in dock_joint_specs)
    direction_signs = tuple(1.0 if index % 2 == 0 else -1.0 for index in range(4))

    candidates = list(coarse_gain_candidates(config))
    if args.maximum_candidates is not None:
        candidates = candidates[: int(args.maximum_candidates)]
    if (baseline_kp, baseline_kd) not in candidates:
        candidates.append((baseline_kp, baseline_kd))
    results: list[dict[str, object]] = []
    for index, (kp, kd) in enumerate(candidates, start=1):
        samples, readback = _run_candidate(
            sim=sim,
            robot=robot,
            torch=torch,
            kp=kp,
            kd=kd,
            dock_joint_names=dock_joint_names,
            dock_joint_ids=dock_joint_ids,
            load_body_ids=load_body_ids,
            load_axes_local=load_axes_local,
            direction_signs=direction_signs,
            config=config,
        )
        result = evaluate_gain_candidate(
            kp=kp,
            kd=kd,
            joint_names=dock_joint_names,
            samples=samples,
            config=config,
        )
        result["search_stage"] = "coarse"
        result["applied_gain_readback"] = readback
        results.append(result)
        print(
            f"[dock-gain coarse {index}/{len(candidates)}] "
            f"kp={kp:.6g} kd={kd:.6g} score={float(result['score']):.6f} "
            f"feasible={bool(result['feasible'])}",
            flush=True,
        )
    coarse_best = select_best_gain_candidate(results)

    if not args.skip_fine:
        fine_candidates = fine_gain_candidates(
            config,
            center_kp=float(coarse_best["kp_nm_per_rad"]),
            center_kd=float(coarse_best["kd_nms_per_rad"]),
            excluded=candidates,
        )
        for index, (kp, kd) in enumerate(fine_candidates, start=1):
            samples, readback = _run_candidate(
                sim=sim,
                robot=robot,
                torch=torch,
                kp=kp,
                kd=kd,
                dock_joint_names=dock_joint_names,
                dock_joint_ids=dock_joint_ids,
                load_body_ids=load_body_ids,
                load_axes_local=load_axes_local,
                direction_signs=direction_signs,
                config=config,
            )
            result = evaluate_gain_candidate(
                kp=kp,
                kd=kd,
                joint_names=dock_joint_names,
                samples=samples,
                config=config,
            )
            result["search_stage"] = "fine"
            result["applied_gain_readback"] = readback
            results.append(result)
            print(
                f"[dock-gain fine {index}/{len(fine_candidates)}] "
                f"kp={kp:.6g} kd={kd:.6g} score={float(result['score']):.6f} "
                f"feasible={bool(result['feasible'])}",
                flush=True,
            )

    selected = dict(select_best_gain_candidate(results))
    selected_key = (
        float(selected["kp_nm_per_rad"]),
        float(selected["kd_nms_per_rad"]),
    )
    verification_samples, verification_readback = _run_candidate(
        sim=sim,
        robot=robot,
        torch=torch,
        kp=selected_key[0],
        kd=selected_key[1],
        dock_joint_names=dock_joint_names,
        dock_joint_ids=dock_joint_ids,
        load_body_ids=load_body_ids,
        load_axes_local=load_axes_local,
        direction_signs=direction_signs,
        config=config,
    )
    verification = evaluate_gain_candidate(
        kp=selected_key[0],
        kd=selected_key[1],
        joint_names=dock_joint_names,
        samples=verification_samples,
        config=config,
    )
    verification["applied_gain_readback"] = verification_readback
    score_delta = abs(float(verification["score"]) - float(selected["score"]))
    deterministic_repeat = score_delta <= max(1.0e-8, 1.0e-5 * float(selected["score"]))
    baseline = next(
        result
        for result in results
        if math.isclose(float(result["kp_nm_per_rad"]), baseline_kp)
        and math.isclose(float(result["kd_nms_per_rad"]), baseline_kd)
    )
    simulated_duration_per_candidate = sum(config.phase_steps().values()) * config.simulation_dt_s
    report = {
        "version": DOCK_JOINT_DRIVE_TUNING_VERSION,
        "method": DOCK_JOINT_DRIVE_TUNING_METHOD,
        "attempted": True,
        "passed": bool(verification["feasible"] and deterministic_repeat),
        "acceptance_eligible": False,
        "diagnostic_only": True,
        "learning_used": False,
        "selection_scope": DOCK_JOINT_DRIVE_TUNING_SELECTION_SCOPE,
        "deployment_gate": DOCK_JOINT_DRIVE_TUNING_DEPLOYMENT_GATE,
        "deployment_gain_selected": False,
        "deployment_note": (
            "The numerical optimum is valid only for this fixed-base, "
            "contact-free response bench. Keep the configured gain unless a "
            "separate representative contact-task A/B validates a replacement."
        ),
        "environment": {
            "module_count": 1,
            "fixed_root": True,
            "fixed_root_body_path": fixed_root_body_path,
            "gravity_enabled": False,
            "ground_created": False,
            "contact_sensors_created": False,
            "self_collisions_enabled": False,
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 8,
            "usd_reused_without_conversion": True,
            "device": str(args.device),
        },
        "source": {
            "backend_config_path": str(_path(args.backend_config)),
            "backend_config_hash": backend_config.stable_hash(),
            "physical_model_hash": physical_model.stable_hash(),
            "urdf_path": str(urdf_path),
            "urdf_hash": hash_file(urdf_path),
            "usd_path": str(usd_path),
            "usd_hash": hash_file(usd_path),
        },
        "actuator_limits": {
            "continuous_torque_nm": float(dock_spec["continuous_torque_limit_nm"]),
            "peak_torque_nm": effort_limit_nm,
            "peak_current_a": peak_current_a,
            "safe_velocity_limit_rad_s": velocity_limit_rad_s,
            "armature_kg_m2": armature_kg_m2,
        },
        "config": config.as_dict(),
        "dock_joint_names": list(dock_joint_names),
        "disturbance_child_body_names": list(load_body_names),
        "disturbance_axes_local": [list(axis) for axis in load_axes_local],
        "baseline": baseline,
        "coarse_best": coarse_best,
        "bench_selected": selected,
        # Retained as an additive compatibility alias for the first v1 report.
        "selected": selected,
        "selected_repeat_verification": verification,
        "selected_repeat_score_delta": score_delta,
        "selected_repeat_deterministic": deterministic_repeat,
        "selected_trace": [_sample_dict(sample) for sample in verification_samples],
        "candidate_count": len(results),
        "simulated_duration_per_candidate_s": simulated_duration_per_candidate,
        "total_candidate_simulated_duration_s": (
            simulated_duration_per_candidate * (len(results) + 1)
        ),
        "candidates": results,
    }
    sim.stop()
    sim.clear_instance()
    return report


def _run_candidate(
    *,
    sim,
    robot,
    torch,
    kp: float,
    kd: float,
    dock_joint_names: tuple[str, ...],
    dock_joint_ids: tuple[int, ...],
    load_body_ids: tuple[int, ...],
    load_axes_local: tuple[tuple[float, float, float], ...],
    direction_signs: tuple[float, ...],
    config,
):
    from amsrr.simulation.dock_joint_drive_tuning import DockJointDriveSample

    dock_joint_id_tensor = torch.tensor(
        dock_joint_ids, dtype=torch.int32, device=sim.device
    )
    load_body_id_tensor = torch.tensor(
        load_body_ids, dtype=torch.int32, device=sim.device
    )
    robot.write_joint_stiffness_to_sim_index(
        stiffness=float(kp), joint_ids=dock_joint_id_tensor
    )
    robot.write_joint_damping_to_sim_index(
        damping=float(kd), joint_ids=dock_joint_id_tensor
    )
    joint_position = robot.data.default_joint_pos.torch.clone()
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel.torch)
    robot.write_joint_position_to_sim_index(position=joint_position)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
    robot.reset()
    zero_force = torch.zeros((1, len(load_body_ids), 3), device=sim.device)
    zero_torque = torch.zeros_like(zero_force)
    robot.permanent_wrench_composer.set_forces_and_torques_index(
        forces=zero_force,
        torques=zero_torque,
        body_ids=load_body_id_tensor,
        is_global=False,
    )
    robot.set_joint_position_target_index(target=joint_position)
    robot.set_joint_velocity_target_index(target=joint_velocity)
    robot.set_joint_effort_target_index(target=torch.zeros_like(joint_position))
    for _ in range(config.phase_steps()["reset_settle"]):
        robot.write_data_to_sim()
        sim.step()
        robot.update(config.simulation_dt_s)

    selected_target = torch.tensor(
        [[config.step_amplitude_rad * sign for sign in direction_signs]],
        dtype=torch.float32,
        device=sim.device,
    )
    zero_selected_target = torch.zeros_like(selected_target)
    disturbance_torque = torch.tensor(
        [
            [
                [
                    config.disturbance_torque_nm * sign * axis_component
                    for axis_component in axis
                ]
                for sign, axis in zip(direction_signs, load_axes_local, strict=True)
            ]
        ],
        dtype=torch.float32,
        device=sim.device,
    )
    samples: list[DockJointDriveSample] = []
    for phase, target, external_torque in (
        ("step", selected_target, zero_torque),
        ("return", zero_selected_target, zero_torque),
        ("disturbance", zero_selected_target, disturbance_torque),
        ("recovery", zero_selected_target, zero_torque),
    ):
        step_count = config.phase_steps()[phase]
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=zero_force,
            torques=external_torque,
            body_ids=load_body_id_tensor,
            is_global=False,
        )
        robot.set_joint_position_target_index(
            target=target, joint_ids=dock_joint_id_tensor
        )
        robot.set_joint_velocity_target_index(
            target=zero_selected_target, joint_ids=dock_joint_id_tensor
        )
        for step_index in range(step_count):
            robot.write_data_to_sim()
            sim.step()
            robot.update(config.simulation_dt_s)
            positions = robot.data.joint_pos.torch[0]
            velocities = robot.data.joint_vel.torch[0]
            torques = robot.data.applied_torque.torch[0]
            samples.append(
                DockJointDriveSample(
                    phase=phase,
                    phase_time_s=(step_index + 1) * config.simulation_dt_s,
                    position_rad_by_joint={
                        name: float(positions[joint_id].detach().cpu())
                        for name, joint_id in zip(
                            dock_joint_names, dock_joint_ids, strict=True
                        )
                    },
                    velocity_rad_s_by_joint={
                        name: float(velocities[joint_id].detach().cpu())
                        for name, joint_id in zip(
                            dock_joint_names, dock_joint_ids, strict=True
                        )
                    },
                    target_rad_by_joint={
                        name: float(target[0, local_index].detach().cpu())
                        for local_index, name in enumerate(dock_joint_names)
                    },
                    applied_torque_nm_by_joint={
                        name: float(torques[joint_id].detach().cpu())
                        for name, joint_id in zip(
                            dock_joint_names, dock_joint_ids, strict=True
                        )
                    },
                )
            )
    applied_stiffness = robot.data.joint_stiffness.torch[0]
    applied_damping = robot.data.joint_damping.torch[0]
    readback = {
        "stiffness_nm_per_rad_by_joint": {
            name: float(applied_stiffness[joint_id].detach().cpu())
            for name, joint_id in zip(dock_joint_names, dock_joint_ids, strict=True)
        },
        "damping_nms_per_rad_by_joint": {
            name: float(applied_damping[joint_id].detach().cpu())
            for name, joint_id in zip(dock_joint_names, dock_joint_ids, strict=True)
        },
    }
    return samples, readback


def _sample_dict(sample) -> dict[str, object]:
    return {
        "phase": sample.phase,
        "phase_time_s": sample.phase_time_s,
        "position_rad_by_joint": dict(sample.position_rad_by_joint),
        "velocity_rad_s_by_joint": dict(sample.velocity_rad_s_by_joint),
        "target_rad_by_joint": dict(sample.target_rad_by_joint),
        "applied_torque_nm_by_joint": dict(sample.applied_torque_nm_by_joint),
    }


def _preauthor_fixed_root_constraint(*, stage, articulation_root_path: str) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    rigid_body_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(articulation_root_path.rstrip("/") + "/")
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and prim.GetName() == "root"
    ]
    if len(rigid_body_paths) != 1:
        raise RuntimeError(
            "fixed-base tuning requires exactly one rigid root body under "
            f"{articulation_root_path!r}, found {rigid_body_paths!r}"
        )
    root_body_path = rigid_body_paths[0]
    joint_root = "/World/DockJointTuning/Constraints"
    UsdGeom.Scope.Define(stage, Sdf.Path(joint_root))
    joint = UsdPhysics.FixedJoint.Define(
        stage, Sdf.Path(f"{joint_root}/world_to_holon_root")
    )
    joint.CreateJointEnabledAttr(True).Set(True)
    joint.CreateExcludeFromArticulationAttr(True).Set(True)
    joint.CreateCollisionEnabledAttr(False).Set(False)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(root_body_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return root_body_path


def _unit_vector(values) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"invalid Dock joint axis: {values!r}")
    return tuple(value / norm for value in vector)


def _summary(report: dict[str, object], output_path: Path) -> dict[str, object]:
    selected = report.get("bench_selected", report.get("selected", {}))
    baseline = report.get("baseline", {})
    return {
        "attempted": report.get("attempted"),
        "passed": report.get("passed"),
        "output": str(output_path),
        "candidate_count": report.get("candidate_count"),
        "baseline_kp": baseline.get("kp_nm_per_rad") if isinstance(baseline, dict) else None,
        "baseline_kd": baseline.get("kd_nms_per_rad") if isinstance(baseline, dict) else None,
        "baseline_score": baseline.get("score") if isinstance(baseline, dict) else None,
        "bench_selected_kp": (
            selected.get("kp_nm_per_rad") if isinstance(selected, dict) else None
        ),
        "bench_selected_kd": (
            selected.get("kd_nms_per_rad") if isinstance(selected, dict) else None
        ),
        "bench_selected_score": (
            selected.get("score") if isinstance(selected, dict) else None
        ),
        "deployment_gain_selected": report.get("deployment_gain_selected"),
        "error": report.get("error"),
    }


def _path(value: str | os.PathLike[str]) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else REPO_ROOT / expanded


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
