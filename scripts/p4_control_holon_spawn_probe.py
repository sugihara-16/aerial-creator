from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
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
    parser.add_argument("--gimbal-stiffness", type=float, default=None, help="Override configured vectoring drive stiffness.")
    parser.add_argument("--gimbal-damping", type=float, default=None, help="Override configured vectoring drive damping.")
    parser.add_argument("--dock-stiffness", type=float, default=None, help="Override configured dock drive stiffness.")
    parser.add_argument("--dock-damping", type=float, default=None, help="Override configured dock drive damping.")
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
        "--random-morphology-takeoff",
        action="store_true",
        help="Run graph-specific floor settle, takeoff ramp, and hover hold with deterministic control.",
    )
    parser.add_argument(
        "--control-contract-version",
        default="legacy_contact_bias_v1",
        choices=("legacy_contact_bias_v1", "centroidal_local_joint_v2"),
        help="Versioned PolicyCommand/QPID contract used by compatible controller smokes.",
    )
    parser.add_argument(
        "--random-morphology-teleop",
        action="store_true",
        help="Continue a passing random-morphology hover with terminal keyboard pose commands.",
    )
    parser.add_argument(
        "--teleop-translation-step-m",
        type=float,
        default=0.05,
        help="Position-target increment for each teleop translation keypress.",
    )
    parser.add_argument(
        "--teleop-rotation-step-rad",
        type=float,
        default=0.08726646259971647,
        help="Attitude-target increment for each teleop rotation keypress.",
    )
    parser.add_argument(
        "--teleop-max-roll-pitch-rad",
        type=float,
        default=0.5235987755982988,
        help="Absolute roll/pitch target safety bound during terminal teleop.",
    )
    parser.add_argument(
        "--teleop-max-position-lead-m",
        type=float,
        default=0.50,
        help="Maximum teleop target-position lead from the measured robot pose.",
    )
    parser.add_argument(
        "--teleop-minimum-height-above-settled-m",
        type=float,
        default=0.15,
        help="Minimum teleop target height above the measured settled pose.",
    )
    parser.add_argument(
        "--random-morphology-graph-json",
        default=None,
        help="Serialized feasible connected MorphologyGraph for the random morphology takeoff smoke.",
    )
    parser.add_argument(
        "--random-morphology-graph-json-path",
        default=None,
        help="Path to a serialized feasible connected MorphologyGraph for takeoff.",
    )
    parser.add_argument(
        "--random-morphology-mesh-search-dir",
        action="append",
        default=None,
        help="Collision/URDF mesh search directory for random-morphology floor placement; repeatable.",
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
        "--p4-2-contact-candidate-set-json",
        default=None,
        help="Serialized ContactCandidateSet for P4.2 gated attach evaluation.",
    )
    parser.add_argument(
        "--p4-2-contact-candidate-set-json-path",
        default=None,
        help="Path to serialized ContactCandidateSet JSON for P4.2 gated attach evaluation.",
    )
    parser.add_argument(
        "--p4-2-contact-wrench-trajectory-json",
        default=None,
        help="Serialized deterministic ContactWrenchTrajectory for P4.2 phase rollout.",
    )
    parser.add_argument(
        "--p4-2-contact-wrench-trajectory-json-path",
        default=None,
        help="Path to serialized deterministic ContactWrenchTrajectory JSON for P4.2 phase rollout.",
    )
    parser.add_argument(
        "--p4-3-pi-l-checkpoint-path",
        default=None,
        help="Optional learned P4.3 pi_L checkpoint; deterministic pi_L remains fallback.",
    )
    parser.add_argument(
        "--p4-3-pi-l-runtime-blend-factor",
        type=float,
        default=0.10,
        help="Trust-region blend for the learned pi_L command subset in (0, 1].",
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
    parser.add_argument(
        "--p4-2-pregrasp-alignment-distance-m",
        type=float,
        default=0.12,
        help="P4.2 distance threshold for approach to pregrasp_align transition.",
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
        "--takeoff-hover-acquisition-timeout-s",
        type=float,
        default=2.0,
        help="Extra deterministic horizon in which to acquire the continuous hover hold.",
    )
    parser.add_argument("--floor-clearance-m", type=float, default=0.002, help="Initial collision-AABB clearance over floor.")
    parser.add_argument(
        "--takeoff-floor-contact-force-threshold-n",
        type=float,
        default=0.5,
        help="Minimum measured aggregate body contact force for floor-contact evidence.",
    )
    parser.add_argument(
        "--takeoff-floor-contact-dwell-duration-s",
        type=float,
        default=0.10,
        help="Required continuous measured floor-contact dwell during zero-thrust settle.",
    )
    parser.add_argument(
        "--takeoff-exact-cross-module-contact-force-threshold-n",
        type=float,
        default=1.0e-3,
        help="Maximum tensor-reported force allowed for unintended cross-module contact.",
    )
    parser.add_argument(
        "--takeoff-exact-cross-module-contact-max-patches-per-body-pair",
        type=int,
        default=8,
        help="Raw PhysX contact-patch buffer capacity multiplier per body pair.",
    )
    parser.add_argument(
        "--takeoff-initial-root-position-tolerance-m",
        type=float,
        default=0.002,
        help="Allowed Isaac-vs-requested initial root position error for floor placement evidence.",
    )
    parser.add_argument(
        "--takeoff-initial-root-attitude-tolerance-rad",
        type=float,
        default=0.001,
        help="Allowed Isaac-vs-requested initial root attitude error for floor placement evidence.",
    )
    parser.add_argument("--takeoff-settle-duration-s", type=float, default=1.0, help="Zero-thrust floor settle duration.")
    parser.add_argument(
        "--takeoff-settle-dwell-duration-s",
        type=float,
        default=0.25,
        help="Required continuous low-speed dwell within the zero-thrust settle phase.",
    )
    parser.add_argument("--takeoff-ramp-duration-s", type=float, default=2.0, help="Settled-pose to hover target ramp duration.")
    parser.add_argument("--takeoff-hover-height-delta-m", type=float, default=0.5, help="Hover root-height gain from settled pose.")
    parser.add_argument(
        "--takeoff-settle-linear-speed-threshold-mps",
        type=float,
        default=0.20,
        help="Maximum settled linear speed for the floor initialization gate.",
    )
    parser.add_argument(
        "--takeoff-settle-angular-speed-threshold-rad-s",
        type=float,
        default=0.50,
        help="Maximum settled angular speed for the floor initialization gate.",
    )
    parser.add_argument(
        "--takeoff-hover-linear-speed-threshold-mps",
        type=float,
        default=0.15,
        help="Maximum continuous linear speed during the accepted hover hold.",
    )
    parser.add_argument(
        "--takeoff-hover-angular-speed-threshold-rad-s",
        type=float,
        default=0.25,
        help="Maximum continuous angular speed during the accepted hover hold.",
    )
    parser.add_argument("--takeoff-max-vertical-speed-mps", type=float, default=3.0, help="Takeoff safety speed threshold.")
    parser.add_argument(
        "--takeoff-min-height-gain-ratio",
        type=float,
        default=0.80,
        help="Required fraction of configured hover height gain.",
    )
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
    warp_cpu_pinned_allocator_fallback = _patch_warp_cpu_pinned_allocator_for_cpu_only()
    try:
        report = run_probe(args_cli)
    except Exception as exc:  # pragma: no cover - exercised through real Isaac smoke commands.
        report = {
            "spawn_passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=8),
        }
    report.setdefault("warp_cpu_pinned_allocator_fallback", warp_cpu_pinned_allocator_fallback)
    printable_report = (
        _compact_random_morphology_teleop_report(report)
        if args_cli.random_morphology_teleop
        else report
    )
    print(json.dumps(printable_report, sort_keys=True))
    if not report.get("spawn_passed"):
        return 1
    for key in (
        "command_probe_passed",
        "single_module_hover_smoke_passed",
        "single_module_articulated_hover_smoke_passed",
        "fixed_morphology_hover_smoke_passed",
        "fixed_morphology_articulated_hover_smoke_passed",
        "fixed_morphology_waypoint_smoke_passed",
        "random_morphology_takeoff_smoke_passed",
        "random_morphology_teleop_passed",
        "p4_1_full_scene_backend_smoke_passed",
        "p4_2_deterministic_rollout_passed",
    ):
        if report.get(key) is False:
            return 1
    return 0


def _compact_random_morphology_teleop_report(
    report: dict[str, object],
) -> dict[str, object]:
    """Keep an interactive terminal run readable while retaining its safety result."""
    keys = (
        "spawn_passed",
        "isaac_backed",
        "random_morphology_graph_id",
        "random_morphology_module_count",
        "random_morphology_takeoff_smoke_passed",
        "random_morphology_teleop",
        "random_morphology_teleop_version",
        "random_morphology_teleop_passed",
        "random_morphology_teleop_no_learning",
        "random_morphology_teleop_quit_reason",
        "random_morphology_teleop_steps",
        "random_morphology_teleop_command_count",
        "random_morphology_teleop_qp_infeasible_count",
        "random_morphology_teleop_clipped_count",
        "random_morphology_teleop_unresolved_target_count",
        "random_morphology_teleop_raw_contact_count",
        "random_morphology_teleop_raw_contact_saturation_count",
        "random_morphology_teleop_final_target_pose_world",
        "random_morphology_teleop_config",
        "warp_cpu_pinned_allocator_fallback",
        "error_type",
        "error",
        "traceback_tail",
    )
    compact = {key: report[key] for key in keys if key in report}
    compact["report_mode"] = "random_morphology_teleop_summary"
    return compact


def _patch_warp_cpu_pinned_allocator_for_cpu_only() -> bool:
    try:
        import warp as wp

        if wp.is_cuda_available():
            return False
        cpu_device = wp.get_device("cpu")
        cpu_device.pinned_allocator = cpu_device.default_allocator
        return True
    except Exception:
        return False


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _joint_drive_parameter(
    physical_model,
    role: str,
    key: str,
    override: float | None,
    fallback: float,
) -> float:
    if override is not None:
        return float(override)
    specs = physical_model.metadata.get("joint_actuator_specs", {})
    spec = specs.get(role, {}) if isinstance(specs, dict) else {}
    drive = spec.get("simulation_drive", {}) if isinstance(spec, dict) else {}
    value = drive.get(key) if isinstance(drive, dict) else None
    return float(value) if isinstance(value, (int, float)) else float(fallback)


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
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
    from amsrr.schemas.contact_candidates import ContactCandidateSet
    from amsrr.schemas.morphology import MorphologyGraph
    from amsrr.schemas.policies import ContactWrenchTrajectory
    from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
    from amsrr.simulation.random_morphology_takeoff import (
        ORDER2_FLOOR_POSE_WORLD,
        ORDER2_FLOOR_SIZE_M,
        RandomMorphologyTakeoffConfig,
        compute_floor_contact_placement,
    )
    from amsrr.simulation.random_morphology_teleop import (
        RandomMorphologyTeleopConfig,
    )
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
    gimbal_stiffness = _joint_drive_parameter(physical_model, "vectoring", "stiffness", args.gimbal_stiffness, 20.0)
    gimbal_damping = _joint_drive_parameter(physical_model, "vectoring", "damping", args.gimbal_damping, 1.0)
    dock_stiffness = _joint_drive_parameter(physical_model, "dock", "stiffness", args.dock_stiffness, 20.0)
    dock_damping = _joint_drive_parameter(physical_model, "dock", "damping", args.dock_damping, 1.0)
    urdf_path = _expand_path(backend_config.holon_urdf_path)
    usd_dir = _expand_path(args.generated_usd_dir or backend_config.generated_usd_dir)
    usd_path = _expand_path(args.generated_usd_path or backend_config.generated_usd_path)
    fixed_control_smoke_requested = bool(
        args.fixed_morphology_hover_smoke
        or args.fixed_morphology_articulated_hover_smoke
        or args.fixed_morphology_waypoint_smoke
    )
    random_takeoff_requested = bool(args.random_morphology_takeoff)
    random_teleop_requested = bool(args.random_morphology_teleop)
    if random_teleop_requested and not random_takeoff_requested:
        raise RuntimeError(
            "--random-morphology-teleop requires --random-morphology-takeoff"
        )
    fixed_smoke_requested = bool(
        fixed_control_smoke_requested
        or random_takeoff_requested
        or args.p4_1_full_scene_backend_smoke
        or args.p4_2_deterministic_rollout
    )
    random_morphology_graph_json = (
        args.random_morphology_graph_json
        or _read_optional_text(args.random_morphology_graph_json_path)
    )
    random_morphology_graph = (
        MorphologyGraph.from_json(random_morphology_graph_json)
        if random_takeoff_requested and random_morphology_graph_json
        else None
    )
    random_mesh_search_dirs = (
        [_expand_path(path) for path in args.random_morphology_mesh_search_dir]
        if args.random_morphology_mesh_search_dir
        else _holon_mesh_search_dirs()
    )
    p4_2_morphology_graph = (
        MorphologyGraph.from_json(args.p4_2_morphology_graph_json)
        if args.p4_2_deterministic_rollout and args.p4_2_morphology_graph_json
        else None
    )
    p4_2_contact_candidate_set_json = (
        args.p4_2_contact_candidate_set_json
        or _read_optional_text(args.p4_2_contact_candidate_set_json_path)
    )
    p4_2_contact_wrench_trajectory_json = (
        args.p4_2_contact_wrench_trajectory_json
        or _read_optional_text(args.p4_2_contact_wrench_trajectory_json_path)
    )
    p4_2_contact_candidate_set = (
        ContactCandidateSet.from_json(p4_2_contact_candidate_set_json)
        if args.p4_2_deterministic_rollout and p4_2_contact_candidate_set_json
        else None
    )
    p4_2_contact_wrench_trajectory = (
        ContactWrenchTrajectory.from_json(p4_2_contact_wrench_trajectory_json)
        if args.p4_2_deterministic_rollout and p4_2_contact_wrench_trajectory_json
        else None
    )
    fixed_module_poses = None
    random_floor_placement = None
    articulated_connections = []
    converted = False

    if random_takeoff_requested:
        if random_morphology_graph is None:
            raise RuntimeError(
                "random morphology takeoff requires --random-morphology-graph-json or its path form"
            )
        random_floor_placement = compute_floor_contact_placement(
            random_morphology_graph,
            physical_model,
            mesh_search_dirs=random_mesh_search_dirs,
            floor_z_m=0.0,
            clearance_m=float(args.floor_clearance_m),
        )
        args.spawn_height = float(random_floor_placement.root_pose_world[2])

    if args.force_convert or fixed_smoke_requested or (args.convert_if_missing and not usd_path.exists()):
        mesh_search_dirs = random_mesh_search_dirs if random_takeoff_requested else _holon_mesh_search_dirs()
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
            elif random_takeoff_requested:
                if random_morphology_graph is None:
                    raise RuntimeError("random morphology takeoff graph is missing")
                graph_name = random_morphology_graph.stable_hash()[:12]
                graph_urdf_path = (
                    usd_dir / "graph_morphology_urdf" / f"holon_random_takeoff_{graph_name}.urdf"
                )
                urdf_path = write_fixed_morphology_graph_urdf(
                    urdf_path,
                    graph_urdf_path,
                    morphology_graph=random_morphology_graph,
                    mesh_search_dirs=mesh_search_dirs,
                )
            else:
                fixed_module_poses = fixed_morphology_module_poses(
                    urdf_path,
                    module_count=int(args.fixed_module_count),
                    module_spacing_m=float(args.fixed_module_spacing_m),
                )
            if args.p4_2_deterministic_rollout or random_takeoff_requested:
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
                    stiffness=gimbal_stiffness,
                    damping=gimbal_damping,
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

    ground_cfg = sim_utils.CuboidCfg(
        size=ORDER2_FLOOR_SIZE_M,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
    )
    ground_cfg.func(
        "/World/defaultGroundPlane",
        ground_cfg,
        translation=ORDER2_FLOOR_POSE_WORLD[:3],
    )
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Holon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=random_takeoff_requested,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=random_takeoff_requested,
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
                stiffness=gimbal_stiffness,
                damping=gimbal_damping,
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=dock_stiffness,
                damping=dock_damping,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    random_floor_contact_sensor = None
    random_self_collision_filter_info = None
    random_initial_exact_collision_info = None
    random_cross_module_contact_views = None
    if random_takeoff_requested:
        # Random-takeoff scenes contain no external collider other than the floor.
        # Same-module pairs and exactly one intended dock-body pair per graph
        # edge are filtered below; any remaining robot/robot event is an
        # exact-collision failure.  The reset-time
        # collider query and per-step PhysX tensor force matrices independently
        # verify that no non-adjacent module contact occurred, so an accepted
        # run's aggregate ContactSensor force is attributable to the floor.
        if random_morphology_graph is None:
            raise RuntimeError("random morphology graph is missing for collision filtering")
        random_self_collision_filter_info = _configure_random_morphology_collision_filters(
            sim.stage,
            morphology_graph=random_morphology_graph,
            physical_model=physical_model,
            root_prim_path="/World/Holon",
        )
        _activate_nested_contact_reports(sim.stage, root_prim_path="/World/Holon")
        random_initial_exact_collision_info = (
            _initial_random_morphology_exact_collision_check(
                sim.stage,
                morphology_graph=random_morphology_graph,
                root_prim_path="/World/Holon",
            )
        )
        random_floor_contact_sensor = ContactSensor(
            cfg=ContactSensorCfg(
                prim_path="/World/Holon/.*",
                update_period=0.0,
                history_length=2,
                max_contact_data_count_per_prim=16,
                debug_vis=False,
            )
        )
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
                    disable_gravity=bool(args.p4_2_deterministic_rollout),
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
    if random_takeoff_requested:
        if random_morphology_graph is None or random_self_collision_filter_info is None:
            raise RuntimeError("random morphology collision views lack graph/filter data")
        random_cross_module_contact_views = _create_cross_module_contact_views(
            sim.physics_manager.get_physics_sim_view(),
            morphology_graph=random_morphology_graph,
            body_paths_by_module=random_self_collision_filter_info[
                "body_paths_by_module"
            ],
            max_patches_per_body_pair=int(
                args.takeoff_exact_cross_module_contact_max_patches_per_body_pair
            ),
        )
    if random_floor_contact_sensor is not None:
        random_floor_contact_sensor.update(sim_dt, force_recompute=True)
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
    effective_fixed_module_count = (
        len(random_morphology_graph.modules)
        if random_morphology_graph is not None
        else int(args.fixed_module_count)
    )
    expected_thrust_bodies = 4 * effective_fixed_module_count if fixed_smoke_requested else 4
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
    if random_takeoff_requested:
        if random_morphology_graph is None or random_floor_placement is None:
            raise RuntimeError("random morphology takeoff graph/floor placement was not initialized")
        takeoff_config = RandomMorphologyTakeoffConfig(
            backend_config_path=str(args.config),
            robot_model_config_path=backend_config.robot_model_config_path,
            mesh_search_dirs=[str(path) for path in random_mesh_search_dirs],
            simulation_dt_s=float(args.dt),
            floor_clearance_m=float(args.floor_clearance_m),
            floor_contact_force_threshold_n=float(
                args.takeoff_floor_contact_force_threshold_n
            ),
            floor_contact_dwell_duration_s=float(
                args.takeoff_floor_contact_dwell_duration_s
            ),
            exact_cross_module_contact_force_threshold_n=float(
                args.takeoff_exact_cross_module_contact_force_threshold_n
            ),
            exact_cross_module_contact_max_patches_per_body_pair=int(
                args.takeoff_exact_cross_module_contact_max_patches_per_body_pair
            ),
            initial_root_position_tolerance_m=float(
                args.takeoff_initial_root_position_tolerance_m
            ),
            initial_root_attitude_tolerance_rad=float(
                args.takeoff_initial_root_attitude_tolerance_rad
            ),
            settle_duration_s=float(args.takeoff_settle_duration_s),
            settle_dwell_duration_s=float(args.takeoff_settle_dwell_duration_s),
            takeoff_ramp_duration_s=float(args.takeoff_ramp_duration_s),
            hover_hold_duration_s=float(args.hover_hold_duration_s),
            hover_acquisition_timeout_s=float(
                args.takeoff_hover_acquisition_timeout_s
            ),
            hover_height_delta_m=float(args.takeoff_hover_height_delta_m),
            position_error_threshold_m=float(args.hover_position_tolerance_m),
            attitude_error_threshold_rad=float(args.hover_attitude_tolerance_rad),
            settle_linear_speed_threshold_mps=float(args.takeoff_settle_linear_speed_threshold_mps),
            settle_angular_speed_threshold_rad_s=float(args.takeoff_settle_angular_speed_threshold_rad_s),
            hover_linear_speed_threshold_mps=float(
                args.takeoff_hover_linear_speed_threshold_mps
            ),
            hover_angular_speed_threshold_rad_s=float(
                args.takeoff_hover_angular_speed_threshold_rad_s
            ),
            max_vertical_speed_mps=float(args.takeoff_max_vertical_speed_mps),
            min_height_gain_ratio=float(args.takeoff_min_height_gain_ratio),
            allocation_mode=str(args.allocation_mode),
            stop_on_hover_hold=bool(args.hover_stop_on_hold),
            control_contract_version=str(args.control_contract_version),
        )
        hover_smoke_report = _run_random_morphology_takeoff_smoke(
            robot=robot,
            sim=sim,
            sim_dt=sim_dt,
            backend_config_hash=backend_config.stable_hash(),
            physical_model=physical_model,
            device=sim.device,
            steps=max(0, args.steps),
            morphology_graph=random_morphology_graph,
            floor_placement=random_floor_placement,
            floor_contact_sensor=random_floor_contact_sensor,
            self_collision_filter_info=random_self_collision_filter_info,
            initial_exact_collision_info=random_initial_exact_collision_info,
            cross_module_contact_views=random_cross_module_contact_views,
            config=takeoff_config,
            bridge_supported_controller_command=bridge_supported_controller_command,
            split_fixed_module_name=split_fixed_module_name,
            realtime_playback=bool(args.realtime_playback),
        )
        if random_teleop_requested:
            if not hover_smoke_report.get(
                "random_morphology_takeoff_smoke_passed"
            ):
                raise RuntimeError(
                    "terminal teleop requires a passing takeoff/hover gate"
                )
            teleop_report = _run_random_morphology_teleop(
                robot=robot,
                sim=sim,
                simulation_app=simulation_app,
                sim_dt=sim_dt,
                physical_model=physical_model,
                device=sim.device,
                morphology_graph=random_morphology_graph,
                floor_contact_sensor=random_floor_contact_sensor,
                cross_module_contact_views=random_cross_module_contact_views,
                hover_pose_world=tuple(
                    hover_smoke_report[
                        "random_morphology_takeoff_hover_target_pose_world"
                    ]
                ),
                settled_pose_world=tuple(
                    hover_smoke_report[
                        "random_morphology_takeoff_settled_pose_world"
                    ]
                ),
                takeoff_config=takeoff_config,
                teleop_config=RandomMorphologyTeleopConfig(
                    translation_step_m=float(args.teleop_translation_step_m),
                    rotation_step_rad=float(args.teleop_rotation_step_rad),
                    max_roll_pitch_rad=float(args.teleop_max_roll_pitch_rad),
                    max_position_lead_m=float(args.teleop_max_position_lead_m),
                    minimum_height_above_settled_m=float(
                        args.teleop_minimum_height_above_settled_m
                    ),
                ),
                bridge_supported_controller_command=bridge_supported_controller_command,
                split_fixed_module_name=split_fixed_module_name,
            )
            hover_smoke_report.update(teleop_report)
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
            contact_candidate_set=p4_2_contact_candidate_set,
            contact_wrench_trajectory=p4_2_contact_wrench_trajectory,
            module_poses=fixed_module_poses,
            object_id="box_01",
            object_size_m=tuple(float(value) for value in args.p4_2_object_size_m),
            object_mass_kg=float(args.p4_2_object_mass_kg),
            target_height=float(args.hover_target_height if args.hover_target_height is not None else args.spawn_height),
            control_dt_s=float(args.dt),
            bridge_supported_controller_command=bridge_supported_controller_command,
            split_fixed_module_name=split_fixed_module_name,
            realtime_playback=bool(args.realtime_playback),
            allocation_mode=str(args.allocation_mode),
            uses_p2_p3=bool(args.p4_2_uses_p2_p3),
            contact_model=str(args.p4_2_contact_model),
            attach_distance_threshold_m=float(args.p4_2_attach_distance_threshold_m),
            attach_relative_velocity_threshold_mps=float(args.p4_2_attach_relative_velocity_threshold_mps),
            attach_snap_distance_threshold_m=float(args.p4_2_attach_snap_distance_threshold_m),
            pregrasp_alignment_distance_m=float(args.p4_2_pregrasp_alignment_distance_m),
            learned_pi_l_checkpoint_path=args.p4_3_pi_l_checkpoint_path,
            learned_pi_l_runtime_blend_factor=float(
                args.p4_3_pi_l_runtime_blend_factor
            ),
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
                "random_morphology_takeoff_smoke_passed",
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
        "gimbal_drive_stiffness": gimbal_stiffness,
        "gimbal_drive_damping": gimbal_damping,
        "dock_drive_stiffness": dock_stiffness,
        "dock_drive_damping": dock_damping,
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


def _module_id_from_prim_path(path: str) -> int | None:
    matches = re.findall(r"(?:^|/)module_(\d+)__", str(path))
    if not matches:
        return None
    # Fixed dock joints nest a child module below its parent module's root in
    # USD.  The deepest/last module-prefixed rigid body therefore owns the
    # collider path.
    return int(matches[-1])


def _configure_random_morphology_collision_filters(
    stage,
    *,
    morphology_graph,
    physical_model,
    root_prim_path: str,
) -> dict[str, object]:
    """Enable exact cross-module physics while filtering intended contacts.

    Internal link pairs of each module are filtered, as are the two dock
    mechanism bodies named by each occupied port pair.  Every other
    cross-module body pair remains physically active and is a hard-failure
    contact, including unintended contacts between adjacent modules.
    """

    from pxr import UsdPhysics

    from amsrr.simulation.random_morphology_takeoff import (
        intended_dock_body_link_pairs,
    )

    root_prefix = root_prim_path.rstrip("/") + "/"
    bodies_by_module: dict[int, list] = {
        module.module_id: [] for module in morphology_graph.modules
    }
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        module_id = _module_id_from_prim_path(prim_path)
        if module_id in bodies_by_module:
            bodies_by_module[module_id].append(prim)
    missing_modules = sorted(
        module_id for module_id, prims in bodies_by_module.items() if not prims
    )
    if missing_modules:
        raise RuntimeError(
            f"random morphology collision filtering found no rigid bodies for modules {missing_modules}"
        )

    adjacent_module_pairs = {
        tuple(sorted((edge.src_module_id, edge.dst_module_id)))
        for edge in morphology_graph.dock_edges
    }
    module_ids = sorted(bodies_by_module)
    filtered_body_pair_count = 0

    def filter_body_pair(src_prim, dst_prim) -> None:
        nonlocal filtered_body_pair_count
        filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(src_prim)
        filtered_pairs_api.CreateFilteredPairsRel().AddTarget(dst_prim.GetPath())
        filtered_body_pair_count += 1

    same_module_filtered_body_pair_count = 0
    for module_id in module_ids:
        module_prims = sorted(
            bodies_by_module[module_id],
            key=lambda prim: prim.GetPath().pathString,
        )
        for src_index, src_prim in enumerate(module_prims):
            for dst_prim in module_prims[src_index + 1 :]:
                filter_body_pair(src_prim, dst_prim)
                same_module_filtered_body_pair_count += 1

    body_prim_by_module_link = {
        (module_id, str(prim.GetName()).removeprefix(f"module_{module_id}__")): prim
        for module_id, prims in bodies_by_module.items()
        for prim in prims
        if str(prim.GetName()).startswith(f"module_{module_id}__")
    }
    intended_dock_path_pairs: list[tuple[str, str]] = []
    intended_dock_link_pairs = intended_dock_body_link_pairs(
        morphology_graph, physical_model
    )
    for src_module_id, src_link, dst_module_id, dst_link in intended_dock_link_pairs:
        src_prim = body_prim_by_module_link.get((src_module_id, src_link))
        dst_prim = body_prim_by_module_link.get((dst_module_id, dst_link))
        if src_prim is None or dst_prim is None:
            raise RuntimeError(
                "random morphology intended dock body did not resolve to rigid prims: "
                f"({src_module_id}, {src_link}) -> ({dst_module_id}, {dst_link})"
            )
        filter_body_pair(src_prim, dst_prim)
        intended_dock_path_pairs.append(
            tuple(
                sorted(
                    (
                        src_prim.GetPath().pathString,
                        dst_prim.GetPath().pathString,
                    )
                )
            )
        )

    module_pair_count = len(module_ids) * (len(module_ids) - 1) // 2
    return {
        "rigid_body_count": sum(len(prims) for prims in bodies_by_module.values()),
        "filtered_body_pair_count": filtered_body_pair_count,
        "same_module_filtered_body_pair_count": same_module_filtered_body_pair_count,
        "intended_dock_filtered_body_pair_count": len(intended_dock_path_pairs),
        "intended_dock_body_link_pairs": intended_dock_link_pairs,
        "intended_dock_body_path_pairs": sorted(intended_dock_path_pairs),
        "adjacent_module_pair_count": len(adjacent_module_pairs),
        "cross_module_pair_count": module_pair_count,
        "nonadjacent_module_pair_count": module_pair_count
        - len(adjacent_module_pairs),
        "body_paths_by_module": {
            module_id: sorted(prim.GetPath().pathString for prim in prims)
            for module_id, prims in bodies_by_module.items()
        },
    }


def _initial_random_morphology_exact_collision_check(
    stage,
    *,
    morphology_graph,
    root_prim_path: str,
) -> dict[str, object]:
    """Run Isaac Sim's exact initial-collider query on the configured stage.

    The official collision-detector utility temporarily disables Fabric, runs
    one PhysX step, and returns collider pairs from the engine's contact report.
    This catches unintended reset-time contacts; per-step tensor contact views
    independently cover the complete takeoff and hover trajectory.
    """

    from omni.physx.scripts.physicsUtils import get_initial_collider_pairs

    adjacent_module_pairs = {
        tuple(sorted((edge.src_module_id, edge.dst_module_id)))
        for edge in morphology_graph.dock_edges
    }
    raw_pairs = sorted(get_initial_collider_pairs(stage))
    robot_pairs: list[tuple[str, str]] = []
    nonadjacent_pairs: list[tuple[str, str]] = []
    adjacent_unintended_pairs: list[tuple[str, str]] = []
    filtered_scope_pairs: list[tuple[str, str]] = []
    unclassified_pairs: list[tuple[str, str]] = []
    for path0, path1 in raw_pairs:
        holon0 = path0 == root_prim_path or path0.startswith(root_prim_path + "/")
        holon1 = path1 == root_prim_path or path1.startswith(root_prim_path + "/")
        if not (holon0 and holon1):
            continue
        pair = tuple(sorted((path0, path1)))
        robot_pairs.append(pair)
        module0 = _module_id_from_prim_path(path0)
        module1 = _module_id_from_prim_path(path1)
        if module0 is None or module1 is None:
            unclassified_pairs.append(pair)
            continue
        module_pair = tuple(sorted((module0, module1)))
        if module0 == module1:
            filtered_scope_pairs.append(pair)
        elif module_pair in adjacent_module_pairs:
            adjacent_unintended_pairs.append(pair)
        else:
            nonadjacent_pairs.append(pair)
    return {
        "method": "isaac_physx_get_initial_collider_pairs_v1",
        "fixed_module_root_pose_invariant": True,
        "raw_pair_count": len(raw_pairs),
        "robot_pair_count": len(robot_pairs),
        "nonadjacent_robot_contact_pairs": sorted(set(nonadjacent_pairs)),
        "adjacent_unintended_robot_contact_pairs": sorted(
            set(adjacent_unintended_pairs)
        ),
        "filtered_scope_robot_contact_pairs": sorted(set(filtered_scope_pairs)),
        "unclassified_robot_contact_pairs": sorted(set(unclassified_pairs)),
    }


def _create_cross_module_contact_views(
    physics_sim_view,
    *,
    morphology_graph,
    body_paths_by_module,
    max_patches_per_body_pair: int,
) -> list[dict[str, object]]:
    """Create one PhysX tensor contact matrix per cross-module pair."""

    if max_patches_per_body_pair <= 0:
        raise RuntimeError("cross-module raw contact capacity multiplier must be positive")
    module_ids = sorted(int(module_id) for module_id in body_paths_by_module)
    views: list[dict[str, object]] = []
    for src_index, src_module_id in enumerate(module_ids):
        for dst_module_id in module_ids[src_index + 1 :]:
            module_pair = (src_module_id, dst_module_id)
            sensor_paths = sorted(body_paths_by_module[src_module_id])
            filter_paths = sorted(body_paths_by_module[dst_module_id])
            raw_contact_capacity = (
                len(sensor_paths)
                * len(filter_paths)
                * max_patches_per_body_pair
            )
            contact_view = physics_sim_view.create_rigid_contact_view(
                sensor_paths,
                filter_patterns=[list(filter_paths) for _ in sensor_paths],
                max_contact_data_count=raw_contact_capacity,
            )
            if contact_view.sensor_count != len(sensor_paths):
                raise RuntimeError(
                    "cross-module contact view sensor count mismatch for modules "
                    f"{module_pair}: {contact_view.sensor_count} != {len(sensor_paths)}"
                )
            if contact_view.filter_count != len(filter_paths):
                raise RuntimeError(
                    "cross-module contact view filter count mismatch for modules "
                    f"{module_pair}: {contact_view.filter_count} != {len(filter_paths)}"
                )
            if contact_view.max_contact_data_count != raw_contact_capacity:
                raise RuntimeError(
                    "cross-module raw contact capacity mismatch for modules "
                    f"{module_pair}: {contact_view.max_contact_data_count} "
                    f"!= {raw_contact_capacity}"
                )
            views.append(
                {
                    "module_pair": module_pair,
                    "view": contact_view,
                    "sensor_count": int(contact_view.sensor_count),
                    "filter_count": int(contact_view.filter_count),
                    "raw_contact_capacity": int(
                        contact_view.max_contact_data_count
                    ),
                }
            )
    return views


def _measure_cross_module_contact_views(
    contact_views: list[dict[str, object]],
    *,
    sim_dt: float,
) -> dict[str, object]:
    """Read aggregate and non-aggregated contact data from PhysX tensor views."""

    import torch
    import warp as wp

    max_force_n = 0.0
    pair_max_forces_n: dict[str, float] = {}
    raw_contact_count = 0
    raw_contact_capacity = 0
    raw_contact_max_force_n = 0.0
    raw_contact_min_separation_m: float | None = None
    raw_contact_saturated = False
    pair_raw_contact_counts: dict[str, int] = {}
    for entry in contact_views:
        contact_view = entry["view"]
        matrix = contact_view.get_contact_force_matrix(sim_dt)
        matrix_tensor = wp.to_torch(matrix)
        force_norms = torch.linalg.vector_norm(matrix_tensor.reshape(-1, 3), dim=-1)
        pair_max_force_n = (
            float(force_norms.max().detach().cpu())
            if force_norms.numel() > 0
            else 0.0
        )
        module_pair = entry["module_pair"]
        pair_key = f"{module_pair[0]}-{module_pair[1]}"
        pair_max_forces_n[pair_key] = pair_max_force_n
        max_force_n = max(max_force_n, pair_max_force_n)

        (
            force_buffer,
            _point_buffer,
            _normal_buffer,
            separation_buffer,
            contact_count_buffer,
            start_indices_buffer,
        ) = contact_view.get_contact_data(sim_dt)
        contact_counts = wp.to_torch(contact_count_buffer).reshape(-1).to(torch.int64)
        start_indices = wp.to_torch(start_indices_buffer).reshape(-1).to(torch.int64)
        pair_raw_count = int(contact_counts.sum().detach().cpu())
        pair_capacity = int(entry["raw_contact_capacity"])
        pair_raw_contact_counts[pair_key] = pair_raw_count
        raw_contact_count += pair_raw_count
        raw_contact_capacity += pair_capacity
        if pair_raw_count >= pair_capacity or bool(
            torch.any(start_indices + contact_counts > pair_capacity).detach().cpu()
        ):
            raw_contact_saturated = True

        if pair_raw_count > 0:
            active_indices: list[int] = []
            for start, count in zip(
                start_indices.detach().cpu().tolist(),
                contact_counts.detach().cpu().tolist(),
                strict=True,
            ):
                if count <= 0:
                    continue
                stop = min(int(start + count), pair_capacity)
                active_indices.extend(range(int(start), stop))
            if active_indices:
                index_tensor = torch.tensor(
                    active_indices,
                    dtype=torch.long,
                    device=wp.to_torch(force_buffer).device,
                )
                active_forces = wp.to_torch(force_buffer).reshape(-1).index_select(
                    0, index_tensor
                )
                active_separations = wp.to_torch(separation_buffer).reshape(
                    -1
                ).index_select(0, index_tensor)
                raw_contact_max_force_n = max(
                    raw_contact_max_force_n,
                    float(active_forces.abs().max().detach().cpu()),
                )
                pair_min_separation = float(
                    active_separations.min().detach().cpu()
                )
                raw_contact_min_separation_m = (
                    pair_min_separation
                    if raw_contact_min_separation_m is None
                    else min(raw_contact_min_separation_m, pair_min_separation)
                )
    return {
        "max_force_n": max_force_n,
        "pair_max_forces_n": pair_max_forces_n,
        "raw_contact_count": raw_contact_count,
        "raw_contact_capacity": raw_contact_capacity,
        "raw_contact_max_force_n": raw_contact_max_force_n,
        "raw_contact_min_separation_m": (
            0.0
            if raw_contact_min_separation_m is None
            else raw_contact_min_separation_m
        ),
        "raw_contact_saturated": raw_contact_saturated,
        "pair_raw_contact_counts": pair_raw_contact_counts,
    }


def _activate_nested_contact_reports(stage, *, root_prim_path: str) -> int:
    """Apply PhysX contact reporting to every articulation rigid body.

    Isaac Lab's generic subtree helper intentionally stops at the first rigid
    body.  A converted articulation has further rigid bodies below its root, so
    the probe must explicitly visit all of them before creating ContactSensor.
    """

    from pxr import PhysxSchema, UsdPhysics

    root_prefix = root_prim_path.rstrip("/") + "/"
    applied_count = 0
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if prim_path != root_prim_path and not prim_path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        applied_count += 1
    if applied_count == 0:
        raise RuntimeError(
            f"no rigid bodies found for contact reporting below {root_prim_path}"
        )
    return applied_count


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


def _run_random_morphology_takeoff_smoke(
    *,
    robot,
    sim,
    sim_dt: float,
    backend_config_hash: str,
    physical_model,
    device: str,
    steps: int,
    morphology_graph,
    floor_placement,
    floor_contact_sensor,
    self_collision_filter_info,
    initial_exact_collision_info,
    cross_module_contact_views,
    config,
    bridge_supported_controller_command,
    split_fixed_module_name,
    realtime_playback: bool,
) -> dict[str, object]:
    """Real-Isaac gate for a reset-time fixed random morphology.

    The settle phase deliberately sends no rotor/controller command.  Once the
    selected legacy-base or versioned centroidal control pose has settled,
    deterministic QPID ramps that pose to an upright hover target.  No learned
    policy participates in this gate.
    """

    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
    from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
    from amsrr.schemas.policies import (
        POLICY_COMMAND_CONTRACT_CENTROIDAL,
        ControllerCommand,
        ControllerStatus,
        InteractionKnot,
        PolicyCommand,
    )
    from amsrr.simulation.random_morphology_takeoff import (
        DeterministicTakeoffScheduler,
        TakeoffPhase,
    )

    module_count = len(morphology_graph.modules)
    centroidal_contract = (
        config.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    )
    if floor_contact_sensor is None:
        raise RuntimeError("random morphology takeoff requires an Isaac contact sensor")
    if not isinstance(self_collision_filter_info, dict):
        raise RuntimeError("random morphology takeoff requires collision-filter evidence")
    if not isinstance(initial_exact_collision_info, dict):
        raise RuntimeError("random morphology takeoff requires exact initial-collider evidence")
    if not isinstance(cross_module_contact_views, list):
        raise RuntimeError("random morphology takeoff requires tensor contact views")
    scheduler = DeterministicTakeoffScheduler(config)
    collision_geometry_hash = collision_geometry_content_hash(
        physical_model,
        mesh_search_dirs=config.mesh_search_dirs,
    )
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    bridge = IsaacControllerBridge()
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode=config.allocation_mode,
            control_dt_s=config.simulation_dt_s,
        )
    )
    rigid_body_model_builder = RigidBodyControlModelBuilder()
    resolved_fc_body_count = sum(
        1
        for module_id in range(module_count)
        if _module_body_pose(robot, module_id=module_id, local_body_name="fc") is not None
    )
    initial_root_pose_actual = tuple(_tensor_row(robot.data.root_pose_w.torch))
    initial_root_position_error = _position_error_norm(
        initial_root_pose_actual[:3], floor_placement.root_pose_world[:3]
    )
    initial_root_attitude_error = _quat_error_norm(
        initial_root_pose_actual[3:7], floor_placement.root_pose_world[3:7]
    )
    initial_base_fc_pose = _module_body_pose(robot, module_id=0, local_body_name="fc")
    if initial_base_fc_pose is None:
        initial_base_fc_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
    initial_control_pose = None
    settled_pose = None
    settled_linear_speed = float("inf")
    settled_angular_speed = float("inf")
    settle_low_speed_steps = 0
    max_settle_low_speed_steps = 0
    settle_low_speed_steps_at_completion = 0
    floor_contact_steps = 0
    max_floor_contact_steps = 0
    floor_contact_steps_at_completion = 0
    floor_contact_dwell_steps_required = max(
        1,
        int(
            math.ceil(
                config.floor_contact_dwell_duration_s / max(sim_dt, 1.0e-9)
            )
        ),
    )
    current_floor_contact = _contact_sensor_measurement(
        floor_contact_sensor,
        force_threshold_n=config.floor_contact_force_threshold_n,
    )
    max_floor_contact_aggregate_force_n = float(
        current_floor_contact["aggregate_force_n"]
    )
    max_floor_contact_active_body_count = int(current_floor_contact["active_body_count"])
    previous_command = None
    phase_counts = {phase.value: 0 for phase in TakeoffPhase}
    phase_transitions: list[dict[str, object]] = []
    previous_phase = None
    runtime_observations: list[dict[str, object]] = []
    policy_commands: list[dict[str, object]] = []
    controller_commands: list[dict[str, object]] = []
    actuator_target_records: list[dict[str, object]] = []
    root_pose_history: list[list[float]] = []
    control_pose_history: list[list[float]] = []
    qp_infeasible_count = 0
    controller_clipped_count = 0
    missing_actuator_count = 0
    unsupported_actuator_count = 0
    clipped_target_count = 0
    application_requested_target_count = 0
    application_applied_target_count = 0
    application_unresolved_target_count = 0
    reaction_torque_target_count = 0
    reaction_torque_abs_sum_nm = 0.0
    dynamic_cross_module_contact_max_force_n = 0.0
    dynamic_cross_module_contact_violation_step_count = 0
    dynamic_cross_module_contact_view_update_count = 0
    dynamic_cross_module_pair_max_forces_n: dict[str, float] = {}
    dynamic_raw_contact_view_update_count = 0
    dynamic_raw_contact_observation_count = 0
    dynamic_raw_contact_observed_step_count = 0
    dynamic_raw_contact_max_force_n = 0.0
    dynamic_raw_contact_min_separation_m: float | None = None
    dynamic_raw_contact_saturation_step_count = 0
    dynamic_pair_raw_contact_counts: dict[str, int] = {}
    dynamic_raw_contact_capacity = sum(
        int(entry["raw_contact_capacity"])
        for entry in cross_module_contact_views
    )
    finite_state = True
    max_vertical_speed = 0.0
    max_position_error = 0.0
    max_attitude_error = 0.0
    final_position_error = float("inf")
    final_attitude_error = float("inf")
    final_linear_speed = float("inf")
    final_angular_speed = float("inf")
    hold_steps = 0
    max_hold_steps = 0
    hold_steps_required = max(1, int(math.ceil(config.hover_hold_duration_s / max(sim_dt, 1.0e-9))))
    settle_dwell_steps_required = max(
        1, int(math.ceil(config.settle_dwell_duration_s / max(sim_dt, 1.0e-9)))
    )
    executed_steps = 0
    ramp_max_progress = 0.0
    last_controller_status = None
    last_bridge_metrics: dict[str, float] = {}

    for step_idx in range(max(0, steps)):
        executed_steps = step_idx + 1
        time_s = step_idx * sim_dt
        root_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
        root_twist = _tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch)
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
        runtime_observation.contact_states = _floor_contact_runtime_states(
            current_floor_contact,
            morphology_graph_id=morphology_graph.graph_id,
        )
        current_base_state = runtime_observation.module_states[0]
        if centroidal_contract:
            current_control_model = rigid_body_model_builder.build(
                morphology_graph,
                physical_model,
                runtime_observation,
            )
            current_control_pose = current_control_model.body_pose_world
            current_control_twist = current_control_model.body_twist_world
        else:
            current_control_pose = current_base_state.pose_world
            current_control_twist = current_base_state.twist_world
        if initial_control_pose is None:
            initial_control_pose = current_control_pose
        if settled_pose is None and time_s + 1.0e-9 >= config.settle_duration_s:
            settled_pose = current_control_pose
            settled_linear_speed = _vector_norm(current_control_twist[:3])
            settled_angular_speed = _vector_norm(current_control_twist[3:6])
            settle_low_speed_steps_at_completion = settle_low_speed_steps
            floor_contact_steps_at_completion = floor_contact_steps
        schedule_reference = settled_pose or initial_control_pose or initial_base_fc_pose
        target = scheduler.target_at(time_s, settled_pose_world=schedule_reference)
        phase_counts[target.phase.value] += 1
        ramp_max_progress = max(ramp_max_progress, target.ramp_progress)
        if target.phase != previous_phase:
            phase_transitions.append(
                {
                    "from_phase": previous_phase.value if previous_phase is not None else None,
                    "to_phase": target.phase.value,
                    "time_s": time_s,
                    "reason": "deterministic_schedule",
                }
            )
            previous_phase = target.phase
        if target.phase == TakeoffPhase.SETTLE:
            settle_status = ControllerStatus(
                status="ok",
                qp_feasible=True,
                active_mode="floor_settle_zero_thrust",
                message="zero-thrust floor settle",
                metrics={
                    "settle_zero_thrust": 1.0,
                    "residual_norm": 0.0,
                    "clipped": 0.0,
                },
            )
            runtime_observation.controller_status = settle_status
            policy_commands.append(
                PolicyCommand(
                    control_contract_version=config.control_contract_version,
                ).to_dict()
            )
            controller_commands.append(
                ControllerCommand(
                    rotor_thrusts_n={},
                    vectoring_joint_targets={},
                    joint_torque_commands={},
                    dock_mechanism_commands={},
                    controller_status=settle_status,
                    control_contract_version=config.control_contract_version,
                ).to_dict()
            )
            actuator_target_records.append(
                {
                    "time_s": time_s,
                    "backend": "isaac_lab",
                    "morphology_graph_id": morphology_graph.graph_id,
                    "command_index": step_idx,
                    "actuator_targets": [],
                    "clipped_targets": [],
                    "missing_actuators": [],
                    "unsupported_actuators": [],
                    "allocation_residual_norm": 0.0,
                    "qp_status": "ok",
                    "metrics": {
                        "settle_zero_thrust": 1.0,
                        "rotor_thrust_target_count": 0.0,
                        "allocation_residual_norm": 0.0,
                        "clipped_target_count": 0.0,
                        "missing_actuator_count": 0.0,
                        "unsupported_actuator_count": 0.0,
                    },
                    "metadata": {"phase": target.phase.value},
                }
            )
            robot.write_data_to_sim()
        else:
            if target.desired_pose_world is None:
                raise RuntimeError("takeoff scheduler enabled thrust without a desired pose")
            policy_command = PolicyCommand(
                desired_body_pose=target.desired_pose_world,
                desired_body_twist=[0.0] * 6,
                control_contract_version=config.control_contract_version,
            )
            controller_command = controller.compute(
                ControllerContext(
                    runtime_observation=runtime_observation,
                    morphology_graph=morphology_graph,
                    physical_model=physical_model,
                    active_knot=InteractionKnot(t_rel_s=time_s, contact_assignments=[]),
                    policy_command=policy_command,
                    previous_command=previous_command,
                    control_dt_s=config.simulation_dt_s,
                )
            )
            bridged_command = bridge_supported_controller_command(controller_command)
            actuator_record = bridge.convert(
                bridged_command,
                actuator_mapping,
                time_s=time_s,
                command_index=step_idx,
            )
            policy_commands.append(policy_command.to_dict())
            controller_commands.append(bridged_command.to_dict())
            application = _apply_actuator_record(
                robot, actuator_record, physical_model, device
            )
            application_requested_target_count += int(
                application["requested_target_count"]
            )
            application_applied_target_count += int(
                application["applied_target_count"]
            )
            application_unresolved_target_count += int(
                application["unresolved_target_count"]
            )
            reaction_torque_target_count += int(
                application["reaction_torque_target_count"]
            )
            reaction_torque_abs_sum_nm += float(
                application["reaction_torque_abs_sum_nm"]
            )
            actuator_record.metrics.update(
                {
                    "application_requested_target_count": float(
                        application["requested_target_count"]
                    ),
                    "application_applied_target_count": float(
                        application["applied_target_count"]
                    ),
                    "application_unresolved_target_count": float(
                        application["unresolved_target_count"]
                    ),
                    "reaction_torque_target_count": float(
                        application["reaction_torque_target_count"]
                    ),
                    "reaction_torque_abs_sum_nm": float(
                        application["reaction_torque_abs_sum_nm"]
                    ),
                }
            )
            actuator_record.metadata["application_unresolved_targets"] = list(
                application["unresolved_targets"]
            )
            actuator_target_records.append(actuator_record.to_dict())
            robot.write_data_to_sim()
            status = bridged_command.controller_status
            runtime_observation.controller_status = status
            last_controller_status = status.to_dict()
            last_bridge_metrics = dict(actuator_record.metrics)
            if not status.qp_feasible:
                qp_infeasible_count += 1
            if status.metrics.get("clipped", 0.0) > 0.0:
                controller_clipped_count += 1
            missing_actuator_count += len(actuator_record.missing_actuators)
            unsupported_actuator_count += len(actuator_record.unsupported_actuators)
            clipped_target_count += len(actuator_record.clipped_targets)
            previous_command = bridged_command

        runtime_observations.append(runtime_observation.to_dict())

        sim.step()
        dynamic_contact_measurement = _measure_cross_module_contact_views(
            cross_module_contact_views,
            sim_dt=sim_dt,
        )
        dynamic_cross_module_contact_view_update_count += len(
            cross_module_contact_views
        )
        dynamic_raw_contact_view_update_count += len(
            cross_module_contact_views
        )
        step_dynamic_max_force_n = float(
            dynamic_contact_measurement["max_force_n"]
        )
        step_raw_contact_count = int(
            dynamic_contact_measurement["raw_contact_count"]
        )
        step_raw_contact_saturated = bool(
            dynamic_contact_measurement["raw_contact_saturated"]
        )
        dynamic_raw_contact_observation_count += step_raw_contact_count
        if step_raw_contact_count > 0:
            dynamic_raw_contact_observed_step_count += 1
            step_min_separation = float(
                dynamic_contact_measurement["raw_contact_min_separation_m"]
            )
            dynamic_raw_contact_min_separation_m = (
                step_min_separation
                if dynamic_raw_contact_min_separation_m is None
                else min(
                    dynamic_raw_contact_min_separation_m,
                    step_min_separation,
                )
            )
        if step_raw_contact_saturated:
            dynamic_raw_contact_saturation_step_count += 1
        dynamic_raw_contact_max_force_n = max(
            dynamic_raw_contact_max_force_n,
            float(dynamic_contact_measurement["raw_contact_max_force_n"]),
        )
        dynamic_cross_module_contact_max_force_n = max(
            dynamic_cross_module_contact_max_force_n,
            step_dynamic_max_force_n,
        )
        if (
            step_dynamic_max_force_n
            > config.exact_cross_module_contact_force_threshold_n
            or step_raw_contact_count > 0
            or step_raw_contact_saturated
        ):
            dynamic_cross_module_contact_violation_step_count += 1
        for pair_key, force_n in dynamic_contact_measurement[
            "pair_max_forces_n"
        ].items():
            dynamic_cross_module_pair_max_forces_n[pair_key] = max(
                dynamic_cross_module_pair_max_forces_n.get(pair_key, 0.0),
                float(force_n),
            )
        for pair_key, count in dynamic_contact_measurement[
            "pair_raw_contact_counts"
        ].items():
            dynamic_pair_raw_contact_counts[pair_key] = (
                dynamic_pair_raw_contact_counts.get(pair_key, 0) + int(count)
            )
        robot.update(sim_dt)
        floor_contact_sensor.update(sim_dt, force_recompute=True)
        current_floor_contact = _contact_sensor_measurement(
            floor_contact_sensor,
            force_threshold_n=config.floor_contact_force_threshold_n,
        )
        max_floor_contact_aggregate_force_n = max(
            max_floor_contact_aggregate_force_n,
            float(current_floor_contact["aggregate_force_n"]),
        )
        max_floor_contact_active_body_count = max(
            max_floor_contact_active_body_count,
            int(current_floor_contact["active_body_count"]),
        )
        if realtime_playback:
            time.sleep(max(0.0, sim_dt))

        base_fc_pose = _module_body_pose(robot, module_id=0, local_body_name="fc")
        base_fc_twist = _module_body_twist(robot, module_id=0, local_body_name="fc")
        if base_fc_pose is None:
            base_fc_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
        if base_fc_twist is None:
            base_fc_twist = _tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(robot.data.root_ang_vel_w.torch)
        if centroidal_contract:
            post_root_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
            post_root_twist = _tensor_row(robot.data.root_lin_vel_w.torch) + _tensor_row(
                robot.data.root_ang_vel_w.torch
            )
            post_observation = _build_articulated_runtime_observation(
                morphology_graph,
                time_s=time_s + sim_dt,
                robot=robot,
                root_pose_world=post_root_pose,
                root_twist_world=post_root_twist,
                joint_names=robot.joint_names,
                joint_positions_tensor=robot.data.joint_pos.torch,
                joint_velocities_tensor=robot.data.joint_vel.torch,
                module_count=module_count,
                split_fixed_module_name=split_fixed_module_name,
            )
            post_model = rigid_body_model_builder.build(
                morphology_graph,
                physical_model,
                post_observation,
            )
            control_pose = post_model.body_pose_world
            control_twist = post_model.body_twist_world
        else:
            control_pose = base_fc_pose
            control_twist = base_fc_twist
        root_pose_history.append(list(base_fc_pose))
        control_pose_history.append(list(control_pose))
        finite_state = finite_state and all(_is_finite(value) for value in control_pose)
        finite_state = finite_state and all(_is_finite(value) for value in control_twist)
        max_vertical_speed = max(max_vertical_speed, abs(float(control_twist[2])))
        final_linear_speed = _vector_norm(control_twist[:3])
        final_angular_speed = _vector_norm(control_twist[3:6])
        if target.phase == TakeoffPhase.SETTLE:
            if bool(current_floor_contact["active"]):
                floor_contact_steps += 1
            else:
                floor_contact_steps = 0
            max_floor_contact_steps = max(max_floor_contact_steps, floor_contact_steps)
            if (
                bool(current_floor_contact["active"])
                and
                final_linear_speed <= config.settle_linear_speed_threshold_mps
                and final_angular_speed <= config.settle_angular_speed_threshold_rad_s
            ):
                settle_low_speed_steps += 1
            else:
                settle_low_speed_steps = 0
            max_settle_low_speed_steps = max(
                max_settle_low_speed_steps, settle_low_speed_steps
            )
        if settled_pose is not None:
            final_target = scheduler.final_hover_pose(settled_pose)
            position_error = _position_error_norm(control_pose[:3], final_target[:3])
            attitude_error = _quat_error_norm(control_pose[3:7], final_target[3:7])
            final_position_error = position_error
            final_attitude_error = attitude_error
            max_position_error = max(max_position_error, position_error)
            max_attitude_error = max(max_attitude_error, attitude_error)
            if (
                target.phase in {TakeoffPhase.HOVER_HOLD, TakeoffPhase.COMPLETE}
                and position_error <= config.position_error_threshold_m
                and attitude_error <= config.attitude_error_threshold_rad
                and final_linear_speed <= config.hover_linear_speed_threshold_mps
                and final_angular_speed <= config.hover_angular_speed_threshold_rad_s
            ):
                hold_steps += 1
            else:
                hold_steps = 0
            max_hold_steps = max(max_hold_steps, hold_steps)
            if config.stop_on_hover_hold and max_hold_steps >= hold_steps_required:
                break

    if settled_pose is None:
        settled_pose = initial_control_pose or initial_base_fc_pose
        settled_linear_speed = final_linear_speed
        settled_angular_speed = final_angular_speed
        settle_low_speed_steps_at_completion = settle_low_speed_steps
        floor_contact_steps_at_completion = floor_contact_steps
    final_base_fc_pose = (
        tuple(root_pose_history[-1])
        if root_pose_history
        else settled_pose
    )
    final_control_pose = (
        tuple(control_pose_history[-1])
        if control_pose_history
        else settled_pose
    )
    final_target = scheduler.final_hover_pose(settled_pose)
    height_gain = float(final_control_pose[2]) - float(settled_pose[2])
    height_gain_ratio = height_gain / max(config.hover_height_delta_m, 1.0e-9)
    hold_time_s = max_hold_steps * sim_dt
    floor_pose_evidenced = (
        abs(float(floor_placement.floor_gap_m) - float(config.floor_clearance_m)) <= 1.0e-6
        and floor_placement.collision_bounds_root.collision_geometry_count > 0
        and initial_root_position_error <= config.initial_root_position_tolerance_m
        and initial_root_attitude_error <= config.initial_root_attitude_tolerance_rad
    )
    floor_contact_evidenced = (
        int(floor_contact_sensor.num_sensors) > 0
        and floor_contact_steps_at_completion >= floor_contact_dwell_steps_required
        and max_floor_contact_aggregate_force_n
        >= config.floor_contact_force_threshold_n
    )
    exact_nonadjacent_contact_pairs = sorted(
        initial_exact_collision_info["nonadjacent_robot_contact_pairs"]
    )
    exact_adjacent_unintended_contact_pairs = sorted(
        initial_exact_collision_info["adjacent_unintended_robot_contact_pairs"]
    )
    filtered_scope_contact_pairs = sorted(
        initial_exact_collision_info["filtered_scope_robot_contact_pairs"]
    )
    unclassified_robot_contact_pairs = sorted(
        initial_exact_collision_info["unclassified_robot_contact_pairs"]
    )
    exact_cross_module_collision_passed = (
        int(self_collision_filter_info["rigid_body_count"])
        == int(floor_contact_sensor.num_sensors)
        and int(self_collision_filter_info["filtered_body_pair_count"]) > 0
        and int(self_collision_filter_info["intended_dock_filtered_body_pair_count"])
        == len(morphology_graph.dock_edges)
        and int(self_collision_filter_info["filtered_body_pair_count"])
        == int(self_collision_filter_info["same_module_filtered_body_pair_count"])
        + int(self_collision_filter_info["intended_dock_filtered_body_pair_count"])
        and initial_exact_collision_info["method"]
        == "isaac_physx_get_initial_collider_pairs_v1"
        and initial_exact_collision_info["fixed_module_root_pose_invariant"] is True
        and len(cross_module_contact_views)
        == int(self_collision_filter_info["cross_module_pair_count"])
        and dynamic_cross_module_contact_view_update_count
        == executed_steps * len(cross_module_contact_views)
        and dynamic_raw_contact_view_update_count
        == executed_steps * len(cross_module_contact_views)
        and dynamic_raw_contact_capacity > 0
        and dynamic_raw_contact_observation_count == 0
        and dynamic_raw_contact_observed_step_count == 0
        and dynamic_raw_contact_max_force_n
        <= config.exact_cross_module_contact_force_threshold_n
        and dynamic_raw_contact_saturation_step_count == 0
        and all(
            count == 0
            for count in dynamic_pair_raw_contact_counts.values()
        )
        and dynamic_cross_module_contact_max_force_n
        <= config.exact_cross_module_contact_force_threshold_n
        and dynamic_cross_module_contact_violation_step_count == 0
        and not exact_nonadjacent_contact_pairs
        and not exact_adjacent_unintended_contact_pairs
        and not filtered_scope_contact_pairs
        and not unclassified_robot_contact_pairs
    )
    settle_passed = (
        phase_counts[TakeoffPhase.SETTLE.value] > 0
        and settled_linear_speed <= config.settle_linear_speed_threshold_mps
        and settled_angular_speed <= config.settle_angular_speed_threshold_rad_s
        and settle_low_speed_steps_at_completion >= settle_dwell_steps_required
    )
    ramp_passed = (
        phase_counts[TakeoffPhase.TAKEOFF_RAMP.value] > 0
        and ramp_max_progress >= 1.0 - 2.0 * sim_dt / max(config.takeoff_ramp_duration_s, sim_dt)
        and height_gain_ratio + 1.0e-9 >= config.min_height_gain_ratio
    )
    hover_passed = (
        final_position_error <= config.position_error_threshold_m
        and final_attitude_error <= config.attitude_error_threshold_rad
        and final_linear_speed <= config.hover_linear_speed_threshold_mps
        and final_angular_speed <= config.hover_angular_speed_threshold_rad_s
        and hold_time_s + 1.0e-9 >= config.hover_hold_duration_s
    )
    logging_passed = (
        len(runtime_observations) == executed_steps
        and len(policy_commands) == executed_steps
        and len(controller_commands) == executed_steps
        and len(actuator_target_records) == executed_steps
    )
    passed = (
        executed_steps > 0
        and finite_state
        and resolved_fc_body_count == module_count
        and floor_pose_evidenced
        and floor_contact_evidenced
        and exact_cross_module_collision_passed
        and settle_passed
        and ramp_passed
        and hover_passed
        and max_vertical_speed <= config.max_vertical_speed_mps
        and qp_infeasible_count == 0
        and controller_clipped_count == 0
        and missing_actuator_count == 0
        and unsupported_actuator_count == 0
        and clipped_target_count == 0
        and application_unresolved_target_count == 0
        and application_requested_target_count == application_applied_target_count
        and reaction_torque_target_count > 0
        and reaction_torque_abs_sum_nm > 0.0
        and math.isclose(sim_dt, config.simulation_dt_s, rel_tol=0.0, abs_tol=1.0e-12)
        and logging_passed
    )
    return {
        "random_morphology_takeoff_smoke": True,
        "random_morphology_takeoff_smoke_passed": bool(passed),
        "random_morphology_takeoff_graph_id": morphology_graph.graph_id,
        "random_morphology_takeoff_morphology_hash": morphology_graph.stable_hash(),
        "random_morphology_takeoff_backend_config_hash": backend_config_hash,
        "random_morphology_takeoff_physical_model_hash": physical_model.stable_hash(),
        "random_morphology_takeoff_collision_geometry_hash": collision_geometry_hash,
        "random_morphology_takeoff_module_count": module_count,
        "random_morphology_takeoff_dock_edge_count": len(morphology_graph.dock_edges),
        "random_morphology_takeoff_single_articulation": True,
        "random_morphology_takeoff_assembly_representation": "reset_time_fixed_dock_tree",
        "random_morphology_takeoff_learned_policy_used": False,
        "random_morphology_takeoff_controller": "deterministic_qpid",
        "random_morphology_takeoff_control_contract_version": config.control_contract_version,
        "random_morphology_takeoff_tracking_state_source": (
            "true_morphology_centroidal_frame"
            if centroidal_contract
            else "legacy_base_module_fc"
        ),
        "random_morphology_takeoff_true_centroidal_tracking": bool(centroidal_contract),
        "random_morphology_takeoff_contact_wrench_tracking_claim": False,
        "random_morphology_takeoff_internal_wrench_tracking_claim": False,
        "random_morphology_takeoff_qp_actuator_variable_scope": "rotor_thrust_vectoring_and_slack_only",
        "random_morphology_takeoff_allocation_mode": config.allocation_mode,
        "random_morphology_takeoff_sim_dt_s": sim_dt,
        "random_morphology_takeoff_sim_dt_matches_config": math.isclose(
            sim_dt, config.simulation_dt_s, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "random_morphology_takeoff_floor_spawned": True,
        "random_morphology_takeoff_floor_pose_evidenced": bool(floor_pose_evidenced),
        "random_morphology_takeoff_floor_contact_evidenced": bool(
            floor_contact_evidenced
        ),
        "random_morphology_takeoff_floor_contact_force_threshold_n": config.floor_contact_force_threshold_n,
        "random_morphology_takeoff_floor_contact_max_aggregate_force_n": max_floor_contact_aggregate_force_n,
        "random_morphology_takeoff_floor_contact_max_active_body_count": max_floor_contact_active_body_count,
        "random_morphology_takeoff_floor_contact_dwell_time_s": floor_contact_steps_at_completion
        * sim_dt,
        "random_morphology_takeoff_floor_contact_dwell_required_s": config.floor_contact_dwell_duration_s,
        "random_morphology_takeoff_contact_sensor_body_count": int(
            floor_contact_sensor.num_sensors
        ),
        "random_morphology_takeoff_contact_external_collider_scope": "floor_only",
        "random_morphology_takeoff_self_collisions_enabled": True,
        "random_morphology_takeoff_exact_cross_module_collision_passed": bool(
            exact_cross_module_collision_passed
        ),
        "random_morphology_takeoff_exact_nonadjacent_collision_passed": bool(
            exact_cross_module_collision_passed
        ),
        "random_morphology_takeoff_exact_collision_rigid_body_count": int(
            self_collision_filter_info["rigid_body_count"]
        ),
        "random_morphology_takeoff_exact_collision_filtered_body_pair_count": int(
            self_collision_filter_info["filtered_body_pair_count"]
        ),
        "random_morphology_takeoff_exact_collision_same_module_filtered_body_pair_count": int(
            self_collision_filter_info["same_module_filtered_body_pair_count"]
        ),
        "random_morphology_takeoff_exact_collision_intended_dock_body_pair_count": int(
            self_collision_filter_info["intended_dock_filtered_body_pair_count"]
        ),
        "random_morphology_takeoff_exact_collision_intended_dock_body_link_pairs": [
            list(pair)
            for pair in self_collision_filter_info["intended_dock_body_link_pairs"]
        ],
        "random_morphology_takeoff_exact_collision_intended_dock_body_pairs": [
            list(pair)
            for pair in self_collision_filter_info["intended_dock_body_path_pairs"]
        ],
        "random_morphology_takeoff_exact_collision_adjacent_module_pair_count": int(
            self_collision_filter_info["adjacent_module_pair_count"]
        ),
        "random_morphology_takeoff_exact_collision_nonadjacent_module_pair_count": int(
            self_collision_filter_info["nonadjacent_module_pair_count"]
        ),
        "random_morphology_takeoff_exact_collision_check_method": str(
            initial_exact_collision_info["method"]
        ),
        "random_morphology_takeoff_exact_collision_fixed_module_root_pose_invariant": bool(
            initial_exact_collision_info["fixed_module_root_pose_invariant"]
        ),
        "random_morphology_takeoff_exact_collision_raw_pair_count": int(
            initial_exact_collision_info["raw_pair_count"]
        ),
        "random_morphology_takeoff_exact_collision_robot_pair_count": int(
            initial_exact_collision_info["robot_pair_count"]
        ),
        "random_morphology_takeoff_dynamic_exact_collision_check_method": "omni_physics_tensors_force_matrix_and_contact_data_v2",
        "random_morphology_takeoff_dynamic_exact_contact_scope": "all_cross_module_except_intended_dock_body_pairs",
        "random_morphology_takeoff_dynamic_exact_contact_view_count": len(
            cross_module_contact_views
        ),
        "random_morphology_takeoff_dynamic_exact_contact_view_update_count": dynamic_cross_module_contact_view_update_count,
        "random_morphology_takeoff_dynamic_exact_contact_force_threshold_n": config.exact_cross_module_contact_force_threshold_n,
        "random_morphology_takeoff_dynamic_exact_contact_max_force_n": dynamic_cross_module_contact_max_force_n,
        "random_morphology_takeoff_dynamic_exact_contact_violation_step_count": dynamic_cross_module_contact_violation_step_count,
        "random_morphology_takeoff_dynamic_exact_pair_max_forces_n": dict(
            sorted(dynamic_cross_module_pair_max_forces_n.items())
        ),
        "random_morphology_takeoff_dynamic_exact_raw_contact_method": "omni_physics_tensors_get_contact_data_v1",
        "random_morphology_takeoff_dynamic_exact_raw_contact_max_patches_per_body_pair": config.exact_cross_module_contact_max_patches_per_body_pair,
        "random_morphology_takeoff_dynamic_exact_raw_contact_capacity": dynamic_raw_contact_capacity,
        "random_morphology_takeoff_dynamic_exact_raw_contact_view_update_count": dynamic_raw_contact_view_update_count,
        "random_morphology_takeoff_dynamic_exact_raw_contact_observation_count": dynamic_raw_contact_observation_count,
        "random_morphology_takeoff_dynamic_exact_raw_contact_observed_step_count": dynamic_raw_contact_observed_step_count,
        "random_morphology_takeoff_dynamic_exact_raw_contact_max_force_n": dynamic_raw_contact_max_force_n,
        "random_morphology_takeoff_dynamic_exact_raw_contact_min_separation_m": (
            0.0
            if dynamic_raw_contact_min_separation_m is None
            else dynamic_raw_contact_min_separation_m
        ),
        "random_morphology_takeoff_dynamic_exact_raw_contact_saturation_step_count": dynamic_raw_contact_saturation_step_count,
        "random_morphology_takeoff_dynamic_exact_raw_contact_observed": dynamic_raw_contact_observation_count
        > 0,
        "random_morphology_takeoff_dynamic_exact_raw_contact_buffer_saturated": dynamic_raw_contact_saturation_step_count
        > 0,
        "random_morphology_takeoff_dynamic_exact_pair_raw_contact_counts": dict(
            sorted(dynamic_pair_raw_contact_counts.items())
        ),
        "random_morphology_takeoff_exact_nonadjacent_contact_count": len(
            exact_nonadjacent_contact_pairs
        ),
        "random_morphology_takeoff_exact_nonadjacent_contact_pairs": [
            list(pair) for pair in exact_nonadjacent_contact_pairs
        ],
        "random_morphology_takeoff_exact_adjacent_unintended_contact_count": len(
            exact_adjacent_unintended_contact_pairs
        ),
        "random_morphology_takeoff_exact_adjacent_unintended_contact_pairs": [
            list(pair) for pair in exact_adjacent_unintended_contact_pairs
        ],
        "random_morphology_takeoff_filtered_scope_contact_count": len(
            filtered_scope_contact_pairs
        ),
        "random_morphology_takeoff_unclassified_robot_contact_count": len(
            unclassified_robot_contact_pairs
        ),
        "random_morphology_takeoff_floor_placement": floor_placement.to_dict(),
        "random_morphology_takeoff_initial_root_pose_world": list(floor_placement.root_pose_world),
        "random_morphology_takeoff_initial_root_pose_actual": list(initial_root_pose_actual),
        "random_morphology_takeoff_initial_root_position_error_m": initial_root_position_error,
        "random_morphology_takeoff_initial_root_attitude_error_rad": initial_root_attitude_error,
        "random_morphology_takeoff_initial_root_position_tolerance_m": config.initial_root_position_tolerance_m,
        "random_morphology_takeoff_initial_root_attitude_tolerance_rad": config.initial_root_attitude_tolerance_rad,
        "random_morphology_takeoff_initial_base_fc_pose_world": list(initial_base_fc_pose),
        "random_morphology_takeoff_resolved_fc_body_count": resolved_fc_body_count,
        "random_morphology_takeoff_settle_zero_thrust": True,
        "random_morphology_takeoff_settle_duration_s": config.settle_duration_s,
        "random_morphology_takeoff_settle_passed": bool(settle_passed),
        "random_morphology_takeoff_settled_pose_world": list(settled_pose),
        "random_morphology_takeoff_settled_linear_speed_mps": settled_linear_speed,
        "random_morphology_takeoff_settled_angular_speed_rad_s": settled_angular_speed,
        "random_morphology_takeoff_settle_low_speed_dwell_time_s": settle_low_speed_steps_at_completion
        * sim_dt,
        "random_morphology_takeoff_settle_low_speed_max_dwell_time_s": max_settle_low_speed_steps
        * sim_dt,
        "random_morphology_takeoff_settle_low_speed_dwell_required_s": config.settle_dwell_duration_s,
        "random_morphology_takeoff_settle_linear_speed_threshold_mps": config.settle_linear_speed_threshold_mps,
        "random_morphology_takeoff_settle_angular_speed_threshold_rad_s": config.settle_angular_speed_threshold_rad_s,
        "random_morphology_takeoff_ramp_passed": bool(ramp_passed),
        "random_morphology_takeoff_takeoff_ramp_duration_s": config.takeoff_ramp_duration_s,
        "random_morphology_takeoff_ramp_max_progress": ramp_max_progress,
        "random_morphology_takeoff_hover_passed": bool(hover_passed),
        "random_morphology_takeoff_hover_height_delta_m": config.hover_height_delta_m,
        "random_morphology_takeoff_stop_on_hover_hold": config.stop_on_hover_hold,
        "random_morphology_takeoff_hover_target_pose_world": list(final_target),
        "random_morphology_takeoff_final_base_fc_pose_world": list(final_base_fc_pose),
        "random_morphology_takeoff_final_control_pose_world": list(final_control_pose),
        "random_morphology_takeoff_height_gain_m": height_gain,
        "random_morphology_takeoff_height_gain_ratio": height_gain_ratio,
        "random_morphology_takeoff_min_height_gain_ratio": config.min_height_gain_ratio,
        "random_morphology_takeoff_final_position_error_m": final_position_error,
        "random_morphology_takeoff_position_error_threshold_m": config.position_error_threshold_m,
        "random_morphology_takeoff_final_attitude_error_rad": final_attitude_error,
        "random_morphology_takeoff_attitude_error_threshold_rad": config.attitude_error_threshold_rad,
        "random_morphology_takeoff_final_linear_speed_mps": final_linear_speed,
        "random_morphology_takeoff_final_angular_speed_rad_s": final_angular_speed,
        "random_morphology_takeoff_hover_linear_speed_threshold_mps": config.hover_linear_speed_threshold_mps,
        "random_morphology_takeoff_hover_angular_speed_threshold_rad_s": config.hover_angular_speed_threshold_rad_s,
        "random_morphology_takeoff_max_position_error_m": max_position_error,
        "random_morphology_takeoff_max_attitude_error_rad": max_attitude_error,
        "random_morphology_takeoff_hover_hold_time_s": hold_time_s,
        "random_morphology_takeoff_hover_hold_required_s": config.hover_hold_duration_s,
        "random_morphology_takeoff_hover_acquisition_timeout_s": config.hover_acquisition_timeout_s,
        "random_morphology_takeoff_max_vertical_speed_mps": max_vertical_speed,
        "random_morphology_takeoff_max_vertical_speed_threshold_mps": config.max_vertical_speed_mps,
        "random_morphology_takeoff_finite_state": bool(finite_state),
        "random_morphology_takeoff_qp_infeasible_count": qp_infeasible_count,
        "random_morphology_takeoff_controller_clipped_count": controller_clipped_count,
        "random_morphology_takeoff_missing_actuator_count": missing_actuator_count,
        "random_morphology_takeoff_unsupported_actuator_count": unsupported_actuator_count,
        "random_morphology_takeoff_clipped_target_count": clipped_target_count,
        "random_morphology_takeoff_application_requested_target_count": application_requested_target_count,
        "random_morphology_takeoff_application_applied_target_count": application_applied_target_count,
        "random_morphology_takeoff_application_unresolved_target_count": application_unresolved_target_count,
        "random_morphology_takeoff_reaction_torque_target_count": reaction_torque_target_count,
        "random_morphology_takeoff_reaction_torque_abs_sum_nm": reaction_torque_abs_sum_nm,
        "random_morphology_takeoff_steps": executed_steps,
        "random_morphology_takeoff_requested_steps": max(0, steps),
        "random_morphology_takeoff_duration_s": executed_steps * sim_dt,
        "random_morphology_takeoff_phase_counts": phase_counts,
        "random_morphology_takeoff_phase_transitions": phase_transitions,
        "random_morphology_takeoff_runtime_observations": runtime_observations,
        "random_morphology_takeoff_policy_commands": policy_commands,
        "random_morphology_takeoff_controller_commands": controller_commands,
        "random_morphology_takeoff_actuator_target_records": actuator_target_records,
        "random_morphology_takeoff_root_pose_history": root_pose_history,
        "random_morphology_takeoff_control_pose_history": control_pose_history,
        "random_morphology_takeoff_logging_passed": bool(logging_passed),
        "random_morphology_takeoff_last_controller_status": last_controller_status,
        "random_morphology_takeoff_last_bridge_metrics": last_bridge_metrics,
        "random_morphology_takeoff_artifacts": {
            "phase": "P4-full-order2",
            "backend": "isaac_lab",
            "isaac_backed": True,
            "dry_run": False,
            "is_p4_full_completion": False,
            "physical_success_claim": "floor_takeoff_hover_only",
            "object_task_claim": False,
            "learned_policy_claim": False,
        },
    }


class _TerminalKeyReader:
    def __init__(self, stream) -> None:
        self._stream = stream
        self._fd: int | None = None
        self._original_settings = None

    def __enter__(self) -> "_TerminalKeyReader":
        import termios
        import tty

        if not self._stream.isatty():
            raise RuntimeError("terminal teleop requires stdin to be a TTY")
        self._fd = self._stream.fileno()
        self._original_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        import termios

        if self._fd is not None and self._original_settings is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._original_settings,
            )

    def read_available(self) -> list[str]:
        import select

        if self._fd is None:
            return []
        keys: list[str] = []
        while select.select([self._fd], [], [], 0.0)[0]:
            data = os.read(self._fd, 64).decode("utf-8", errors="ignore")
            if not data:
                break
            data = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", data)
            keys.extend(data)
        return keys


def _run_random_morphology_teleop(
    *,
    robot,
    sim,
    simulation_app,
    sim_dt: float,
    physical_model,
    device: str,
    morphology_graph,
    floor_contact_sensor,
    cross_module_contact_views,
    hover_pose_world,
    settled_pose_world,
    takeoff_config,
    teleop_config,
    bridge_supported_controller_command,
    split_fixed_module_name,
) -> dict[str, object]:
    """Terminal waypoint teleop using deterministic QPID/QP only."""

    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand
    from amsrr.simulation.random_morphology_teleop import (
        RANDOM_MORPHOLOGY_TELEOP_VERSION,
        TELEOP_HELP,
        RandomMorphologyTeleopTarget,
        format_teleop_pose,
    )

    teleop_config.validate()
    if floor_contact_sensor is None:
        raise RuntimeError("random morphology teleop requires the takeoff contact sensor")
    if not isinstance(cross_module_contact_views, list):
        raise RuntimeError("random morphology teleop requires cross-module contact views")

    target = RandomMorphologyTeleopTarget.from_hover_pose(
        hover_pose_world,
        settled_height_m=float(settled_pose_world[2]),
        config=teleop_config,
    )
    actuator_mapping = build_actuator_mapping(morphology_graph, physical_model)
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode=takeoff_config.allocation_mode,
            control_dt_s=takeoff_config.simulation_dt_s,
        )
    )
    bridge = IsaacControllerBridge()
    previous_command = None
    step_count = 0
    command_count = 0
    qp_infeasible_count = 0
    clipped_count = 0
    unresolved_target_count = 0
    raw_contact_count = 0
    raw_contact_saturation_count = 0
    safety_failure: str | None = None
    quit_reason = "window_closed"
    module_count = len(morphology_graph.modules)
    current_floor_contact = _contact_sensor_measurement(
        floor_contact_sensor,
        force_threshold_n=takeoff_config.floor_contact_force_threshold_n,
    )

    print("\nTakeoff/hover passed. Terminal teleop is active.", flush=True)
    print(TELEOP_HELP, flush=True)
    print(f"target: {format_teleop_pose(target.target_pose_world)}", flush=True)

    with _TerminalKeyReader(sys.stdin) as key_reader:
        while simulation_app.is_running():
            base_pose = _module_body_pose(robot, module_id=0, local_body_name="fc")
            base_twist = _module_body_twist(robot, module_id=0, local_body_name="fc")
            if base_pose is None:
                base_pose = tuple(_tensor_row(robot.data.root_pose_w.torch))
            if base_twist is None:
                base_twist = _tensor_row(
                    robot.data.root_lin_vel_w.torch
                ) + _tensor_row(robot.data.root_ang_vel_w.torch)

            quit_requested = False
            for key in key_reader.read_available():
                update = target.apply_key(key, current_pose_world=base_pose)
                if not update.recognized:
                    continue
                if update.print_help:
                    print(TELEOP_HELP, flush=True)
                elif update.print_pose:
                    print(
                        f"target: {format_teleop_pose(update.target_pose_world)}",
                        flush=True,
                    )
                elif update.quit_requested:
                    quit_requested = True
                    quit_reason = "user_quit"
                else:
                    command_count += 1
                    print(
                        f"[{update.action}] "
                        f"{format_teleop_pose(update.target_pose_world)}",
                        flush=True,
                    )
            if quit_requested:
                break

            time_s = step_count * sim_dt
            observation = _build_articulated_runtime_observation(
                morphology_graph,
                time_s=time_s,
                robot=robot,
                root_pose_world=base_pose,
                root_twist_world=base_twist,
                joint_names=robot.joint_names,
                joint_positions_tensor=robot.data.joint_pos.torch,
                joint_velocities_tensor=robot.data.joint_vel.torch,
                module_count=module_count,
                split_fixed_module_name=split_fixed_module_name,
            )
            observation.contact_states = _floor_contact_runtime_states(
                current_floor_contact,
                morphology_graph_id=morphology_graph.graph_id,
            )
            policy_command = PolicyCommand(
                desired_body_pose=target.target_pose_world,
                desired_body_twist=[0.0] * 6,
            )
            controller_command = controller.compute(
                ControllerContext(
                    runtime_observation=observation,
                    morphology_graph=morphology_graph,
                    physical_model=physical_model,
                    active_knot=InteractionKnot(
                        t_rel_s=time_s,
                        contact_assignments=[],
                    ),
                    policy_command=policy_command,
                    previous_command=previous_command,
                    control_dt_s=takeoff_config.simulation_dt_s,
                )
            )
            bridged_command = bridge_supported_controller_command(controller_command)
            if not bridged_command.controller_status.qp_feasible:
                qp_infeasible_count += 1
                safety_failure = "controller_qp_infeasible"
                break
            if bridged_command.controller_status.metrics.get("clipped", 0.0) > 0.0:
                clipped_count += 1
                safety_failure = "controller_command_clipped"
                break
            actuator_record = bridge.convert(
                bridged_command,
                actuator_mapping,
                time_s=time_s,
                command_index=step_count,
            )
            if (
                actuator_record.clipped_targets
                or actuator_record.missing_actuators
                or actuator_record.unsupported_actuators
            ):
                clipped_count += len(actuator_record.clipped_targets)
                safety_failure = "actuator_bridge_target_failure"
                break
            application = _apply_actuator_record(
                robot,
                actuator_record,
                physical_model,
                device,
            )
            unresolved_target_count += int(application["unresolved_target_count"])
            if int(application["unresolved_target_count"]) > 0:
                safety_failure = "unresolved_actuator_target"
                break

            robot.write_data_to_sim()
            sim.step()
            contact_measurement = _measure_cross_module_contact_views(
                cross_module_contact_views,
                sim_dt=sim_dt,
            )
            raw_contact_count += int(contact_measurement["raw_contact_count"])
            if bool(contact_measurement["raw_contact_saturated"]):
                raw_contact_saturation_count += 1
                safety_failure = "raw_contact_buffer_saturated"
                break
            if int(contact_measurement["raw_contact_count"]) > 0:
                safety_failure = "unintended_cross_module_contact"
                break
            robot.update(sim_dt)
            floor_contact_sensor.update(sim_dt, force_recompute=True)
            current_floor_contact = _contact_sensor_measurement(
                floor_contact_sensor,
                force_threshold_n=takeoff_config.floor_contact_force_threshold_n,
            )
            previous_command = bridged_command
            step_count += 1
            time.sleep(max(0.0, sim_dt))

    if safety_failure is not None:
        quit_reason = safety_failure
        print(f"Teleop stopped by safety gate: {safety_failure}", flush=True)
    else:
        print(f"Teleop finished: {quit_reason}", flush=True)
    return {
        "random_morphology_teleop": True,
        "random_morphology_teleop_version": RANDOM_MORPHOLOGY_TELEOP_VERSION,
        "random_morphology_teleop_passed": safety_failure is None,
        "random_morphology_teleop_no_learning": True,
        "random_morphology_teleop_quit_reason": quit_reason,
        "random_morphology_teleop_steps": step_count,
        "random_morphology_teleop_command_count": command_count,
        "random_morphology_teleop_qp_infeasible_count": qp_infeasible_count,
        "random_morphology_teleop_clipped_count": clipped_count,
        "random_morphology_teleop_unresolved_target_count": unresolved_target_count,
        "random_morphology_teleop_raw_contact_count": raw_contact_count,
        "random_morphology_teleop_raw_contact_saturation_count": raw_contact_saturation_count,
        "random_morphology_teleop_final_target_pose_world": list(
            target.target_pose_world
        ),
        "random_morphology_teleop_config": teleop_config.to_dict(),
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
    contact_candidate_set,
    contact_wrench_trajectory,
    module_poses: dict[int, tuple[float, float, float, float, float, float, float]] | None,
    object_id: str,
    object_size_m: tuple[float, float, float],
    object_mass_kg: float,
    target_height: float,
    control_dt_s: float,
    bridge_supported_controller_command,
    split_fixed_module_name,
    realtime_playback: bool,
    allocation_mode: str,
    uses_p2_p3: bool,
    contact_model: str,
    attach_distance_threshold_m: float,
    attach_relative_velocity_threshold_mps: float,
    attach_snap_distance_threshold_m: float,
    pregrasp_alignment_distance_m: float,
    learned_pi_l_checkpoint_path: str | None = None,
    learned_pi_l_runtime_blend_factor: float = 0.10,
) -> dict[str, object]:
    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext, PayloadCoupling
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.policies.learned_low_level_policy import (
        LearnedLowLevelPolicy,
        overlay_learned_pi_l_subset,
    )
    from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
    from amsrr.schemas.policies import InteractionKnot, PolicyCommand
    from amsrr.simulation.p4_2_rollout import (
        P4_2AttachEvent,
        P4_2DeterministicRolloutConfig,
        P4_2RolloutPhase,
        P4_2PhaseTransitionRecord,
        P4_2ReleaseEvent,
        evaluate_p4_2_attach_conditions,
        p4_2_controller_status_is_fatal,
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
    learned_pi_l = None
    learned_pi_l_checkpoint_load_error: str | None = None
    if learned_pi_l_checkpoint_path is not None:
        try:
            learned_pi_l = LearnedLowLevelPolicy.from_checkpoint(
                learned_pi_l_checkpoint_path
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            # A learned checkpoint is never allowed to take ownership away from
            # the deterministic P4.2 command path.  Record only the exception
            # type (paths and framework messages can contain host details) and
            # continue with that existing command path.
            learned_pi_l_checkpoint_load_error = type(exc).__name__
    if not 0.0 < learned_pi_l_runtime_blend_factor <= 1.0:
        raise ValueError("learned pi_L runtime blend factor must be in (0, 1]")
    learned_pi_l_decision_count = 0
    learned_pi_l_fallback_count = 0
    learned_pi_l_overlay_nonzero_count = 0
    learned_pi_l_overlay_delta_norm_sum = 0.0
    learned_pi_l_overlay_delta_norm_max = 0.0
    object_pose_initial, _ = _p4_1_object_pose_and_twist(p4_2_object)
    pre_attach_object_pose = tuple(float(value) for value in object_pose_initial)
    target_twist = [0.0] * 6
    previous_command = None
    runtime_observation_objects = []
    runtime_observations: list[dict[str, object]] = []
    policy_commands: list[dict[str, object]] = []
    learned_pi_l_pre_overlay_policy_commands: list[dict[str, object]] = []
    learned_pi_l_controller_active_knots: list[dict[str, object]] = []
    controller_commands: list[dict[str, object]] = []
    actuator_target_records: list[dict[str, object]] = []
    object_pose_history: list[list[float]] = []
    qp_infeasible_count = 0
    qp_infeasible_consecutive = 0
    clipped_count = 0
    missing_actuator_count = 0
    unsupported_actuator_count = 0
    clipped_target_count = 0
    finite_state = True
    executed_steps = 0
    last_controller_status = None
    last_bridge_metrics: dict[str, float] = {}
    rollout_config = P4_2DeterministicRolloutConfig(
        object_id=object_id,
        object_size_m=object_size_m,
        object_mass_kg=object_mass_kg,
        contact_model=contact_model,
        attach_distance_threshold_m=attach_distance_threshold_m,
        attach_relative_velocity_threshold_mps=attach_relative_velocity_threshold_mps,
        attach_snap_distance_threshold_m=attach_snap_distance_threshold_m,
        pregrasp_alignment_distance_m=pregrasp_alignment_distance_m,
    )
    selected_assignments = _p4_2_selected_assignments(contact_wrench_trajectory)
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in (contact_candidate_set.candidates if contact_candidate_set is not None else [])
    }
    anchor_by_id = {anchor.anchor_id: anchor for anchor in morphology_graph.robot_anchors}
    selected_pairs = [
        (assignment, candidate_by_id[assignment.candidate_id], anchor_by_id[assignment.anchor_id])
        for assignment in selected_assignments
        if assignment.candidate_id in candidate_by_id and assignment.anchor_id in anchor_by_id
    ]
    selected_contact_candidates_available = bool(selected_pairs)
    selected_assignment, selected_candidate, selected_anchor = (
        selected_pairs[0] if selected_pairs else (None, None, None)
    )
    selected_assignment_feasible = _p4_2_selected_assignment_feasible(
        contact_candidate_set,
        [assignment.candidate_id for assignment in selected_assignments],
    )
    fallback_anchor_relative_pose = (
        _p4_2_anchor_relative_pose(morphology_graph, graph_module_poses, selected_anchor)
        if selected_anchor is not None
        else None
    )
    candidate_relative_to_object = (
        compose_pose(inverse_pose(object_pose_initial), selected_candidate.contact_pose_world)
        if selected_candidate is not None
        else None
    )
    release_object_target = _p4_2_release_object_target(
        contact_wrench_trajectory,
        object_id=object_id,
        fallback_pose=object_pose_initial,
    )
    payload_inertia = _p4_2_cuboid_inertia_body(float(object_mass_kg), object_size_m)
    current_phase = P4_2RolloutPhase.APPROACH
    phase_started_s = 0.0
    final_phase: P4_2RolloutPhase | None = None
    attach_events: list[P4_2AttachEvent] = []
    release_events: list[P4_2ReleaseEvent] = []
    attached = False
    object_relative_to_anchor = None
    object_relative_to_root_at_attach = None
    transport_start_object_pose = None
    transport_displacement_m = 0.0
    root_pose_at_release = None
    object_pose_at_release = None
    anchor_debug_samples: list[dict[str, object]] = []
    link_backed_anchor_pose_used = False
    last_anchor_resolution: dict[str, object] = {}
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
        anchor_resolution = _p4_2_anchor_resolution(
            robot=robot,
            runtime_observation=runtime_observation,
            anchor=selected_anchor,
        )
        last_anchor_resolution = dict(anchor_resolution)
        anchor_pose = anchor_resolution.get("pose_world")
        anchor_twist = anchor_resolution.get("twist_world")
        link_backed_anchor_pose_used = link_backed_anchor_pose_used or (
            anchor_resolution.get("anchor_pose_source") == "isaac_link"
        )
        anchor_relative_pose_for_control = (
            compose_pose(inverse_pose(root_pose), anchor_pose)
            if anchor_pose is not None
            else fallback_anchor_relative_pose
        )
        if not attached and current_phase in {
            P4_2RolloutPhase.APPROACH,
            P4_2RolloutPhase.PREGRASP_ALIGN,
            P4_2RolloutPhase.ATTACH_ATTEMPT,
        }:
            _set_p4_2_object_pose_and_twist(p4_2_object, pre_attach_object_pose, [0.0] * 6, device=device)
            p4_2_object.update(sim_dt)
        if attached and anchor_pose is not None and object_relative_to_anchor is not None:
            slaved_pose = compose_pose(anchor_pose, object_relative_to_anchor)
            _set_p4_2_object_pose_and_twist(p4_2_object, slaved_pose, [0.0] * 6, device=device)
            p4_2_object.update(sim_dt)
        object_state = _build_p4_1_object_runtime_state(p4_2_object, object_id=object_id)
        runtime_observation.object_states = [object_state]

        attach_anchor_target = (
            compose_pose(object_state.pose_world, candidate_relative_to_object)
            if candidate_relative_to_object is not None
            else (selected_candidate.contact_pose_world if selected_candidate is not None else object_state.pose_world)
        )
        approach_anchor_target = _p4_2_offset_pose(attach_anchor_target, dz=0.10)
        anchor_distance_m = (
            _p4_2_pose_distance(anchor_pose, attach_anchor_target)
            if anchor_pose is not None and selected_candidate is not None
            else float("inf")
        )
        relative_velocity_mps = _p4_2_relative_speed(
            list(anchor_twist) if isinstance(anchor_twist, list) else [0.0] * 6,
            object_state.twist_world,
        )
        phase_elapsed_s = max(0.0, time_s - phase_started_s)
        if final_phase is None:
            if current_phase == P4_2RolloutPhase.APPROACH:
                if selected_contact_candidates_available and anchor_distance_m <= rollout_config.pregrasp_alignment_distance_m:
                    phase_transitions.append(
                        _p4_2_transition_record(
                            current_phase,
                            P4_2RolloutPhase.PREGRASP_ALIGN,
                            time_s=time_s,
                            phase_elapsed_s=phase_elapsed_s,
                            reason="selected_anchor_within_pregrasp_alignment_distance",
                            timeout_s=rollout_config.phase_timeouts_s[current_phase.value],
                        )
                    )
                    current_phase = P4_2RolloutPhase.PREGRASP_ALIGN
                    phase_started_s = time_s
                    phase_elapsed_s = 0.0
                elif phase_elapsed_s > rollout_config.phase_timeouts_s[current_phase.value]:
                    final_phase = P4_2RolloutPhase.TIMEOUT_FAILURE
            if current_phase == P4_2RolloutPhase.PREGRASP_ALIGN:
                if anchor_distance_m <= rollout_config.attach_distance_threshold_m:
                    phase_transitions.append(
                        _p4_2_transition_record(
                            current_phase,
                            P4_2RolloutPhase.ATTACH_ATTEMPT,
                            time_s=time_s,
                            phase_elapsed_s=phase_elapsed_s,
                            reason="anchor_distance_inside_attach_gate",
                            timeout_s=rollout_config.phase_timeouts_s[current_phase.value],
                        )
                    )
                    current_phase = P4_2RolloutPhase.ATTACH_ATTEMPT
                    phase_started_s = time_s
                    phase_elapsed_s = 0.0
                elif phase_elapsed_s > rollout_config.phase_timeouts_s[current_phase.value]:
                    final_phase = P4_2RolloutPhase.TIMEOUT_FAILURE
            if current_phase in {
                P4_2RolloutPhase.ATTACH_ATTEMPT,
                P4_2RolloutPhase.ATTACHED_MAINTAIN,
                P4_2RolloutPhase.TRANSPORT,
                P4_2RolloutPhase.RELEASE,
            } and phase_elapsed_s > rollout_config.phase_timeouts_s[current_phase.value]:
                final_phase = (
                    P4_2RolloutPhase.DROP_FAILURE
                    if current_phase in {P4_2RolloutPhase.ATTACH_ATTEMPT, P4_2RolloutPhase.ATTACHED_MAINTAIN}
                    else P4_2RolloutPhase.TIMEOUT_FAILURE
                )

        release_error_m = _p4_2_pose_distance(object_state.pose_world, release_object_target)
        transport_displacement_m = (
            _p4_2_pose_distance(object_state.pose_world, transport_start_object_pose)
            if transport_start_object_pose is not None
            else 0.0
        )
        if final_phase is None and current_phase == P4_2RolloutPhase.ATTACHED_MAINTAIN:
            maintain_dwell_s = min(0.25, rollout_config.phase_timeouts_s[current_phase.value])
            if phase_elapsed_s >= maintain_dwell_s:
                transport_start_object_pose = object_state.pose_world
                phase_transitions.append(
                    _p4_2_transition_record(
                        current_phase,
                        P4_2RolloutPhase.TRANSPORT,
                        time_s=time_s,
                        phase_elapsed_s=phase_elapsed_s,
                        reason="attached_maintain_dwell_satisfied",
                        timeout_s=rollout_config.phase_timeouts_s[current_phase.value],
                    )
                )
                current_phase = P4_2RolloutPhase.TRANSPORT
                phase_started_s = time_s
                phase_elapsed_s = 0.0
        if final_phase is None and current_phase == P4_2RolloutPhase.TRANSPORT:
            if transport_start_object_pose is None:
                transport_start_object_pose = object_state.pose_world
                transport_displacement_m = 0.0
            release_region_reached = release_error_m <= max(rollout_config.attach_distance_threshold_m, 0.08)
            bounded_transport_reached = transport_displacement_m >= rollout_config.transport_min_displacement_m
            if release_region_reached or bounded_transport_reached:
                release_reason = (
                    "attached_object_reached_release_region"
                    if release_region_reached
                    else "bounded_payload_carry_displacement_reached"
                )
                phase_transitions.append(
                    _p4_2_transition_record(
                        current_phase,
                        P4_2RolloutPhase.RELEASE,
                        time_s=time_s,
                        phase_elapsed_s=phase_elapsed_s,
                        reason=release_reason,
                        timeout_s=rollout_config.phase_timeouts_s[current_phase.value],
                    )
                )
                current_phase = P4_2RolloutPhase.RELEASE
                phase_started_s = time_s
                phase_elapsed_s = 0.0

        runtime_observation.task_progress.phase_label = current_phase.value
        runtime_observation.task_progress.metrics.update(
            {
                "selected_contact_candidates_available": 1.0 if selected_contact_candidates_available else 0.0,
                "selected_assignment_feasible": 1.0 if selected_assignment_feasible else 0.0,
                "anchor_object_distance_m": 0.0 if math.isinf(anchor_distance_m) else float(anchor_distance_m),
                "relative_velocity_mps": float(relative_velocity_mps),
                "transport_displacement_m": float(transport_displacement_m),
                "transport_min_displacement_m": float(rollout_config.transport_min_displacement_m),
                "unconditional_attach_allowed": 0.0,
                "anchor_pose_source_is_isaac_link": (
                    1.0 if anchor_resolution.get("anchor_pose_source") == "isaac_link" else 0.0
                ),
            }
        )
        if len(anchor_debug_samples) < 64 and anchor_pose is not None:
            anchor_debug_samples.append(
                {
                    "time_s": float(time_s),
                    "phase": current_phase.value,
                    "anchor_id": None if selected_anchor is None else int(selected_anchor.anchor_id),
                    "anchor_pose_world": list(anchor_pose),
                    "anchor_pose_source": anchor_resolution.get("anchor_pose_source"),
                    "anchor_link_id": anchor_resolution.get("anchor_link_id"),
                    "anchor_resolved_body_name": anchor_resolution.get("anchor_resolved_body_name"),
                    "contact_pose_world": list(attach_anchor_target),
                    "anchor_object_distance_m": float(anchor_distance_m),
                }
            )
        runtime_observation_objects.append(runtime_observation)
        runtime_observations.append(runtime_observation.to_dict())
        desired_body_pose = _p4_2_target_pose_for_phase(
            current_phase,
            approach_anchor_target=approach_anchor_target,
            attach_anchor_target=attach_anchor_target,
            root_pose=root_pose,  # type: ignore[arg-type]
            release_object_target=release_object_target,
            anchor_relative_pose=anchor_relative_pose_for_control,
            object_relative_to_anchor=object_relative_to_anchor,
        )
        phase_weight = f"p4_2_phase_{current_phase.value}"
        policy_command = PolicyCommand(
            desired_body_pose=desired_body_pose,
            desired_body_twist=target_twist,
            priority_weights={
                phase_weight: 1.0,
                "attach_condition_gate": 1.0 if current_phase == P4_2RolloutPhase.ATTACH_ATTEMPT else 0.0,
                "attached_object_tracking": 1.0 if attached else 0.0,
                "release_gate": 1.0 if current_phase == P4_2RolloutPhase.RELEASE else 0.0,
            },
        )
        active_knot = InteractionKnot(
            t_rel_s=time_s,
            contact_assignments=selected_assignments if selected_contact_candidates_available else [],
            priority_weights={phase_weight: 1.0},
            guard_conditions=[
                {
                    "type": "p4_2_phase",
                    "phase": current_phase.value,
                    "contact_model": contact_model,
                },
                {
                    "type": "p4_2_attach_gate",
                    "selected_contact_candidates_available": selected_contact_candidates_available,
                    "robot_anchors_available": len(morphology_graph.robot_anchors) > 0,
                    "unconditional_attach_allowed": False,
                    "attach_distance_threshold_m": float(attach_distance_threshold_m),
                    "attach_relative_velocity_threshold_mps": float(attach_relative_velocity_threshold_mps),
                    "attach_snap_distance_threshold_m": float(attach_snap_distance_threshold_m),
                },
            ],
        )
        if learned_pi_l is not None:
            learned_source_knot = next(
                (
                    knot
                    for knot in contact_wrench_trajectory.knots
                    if any(
                        guard.get("type") == "p4_2_phase"
                        and guard.get("phase") == current_phase.value
                        for guard in knot.guard_conditions
                    )
                ),
                active_knot,
            )
            learned_policy_command = learned_pi_l.command(
                LowLevelPolicyContext(
                    runtime_observation=runtime_observation,
                    morphology_graph=morphology_graph,
                    physical_model=physical_model,
                    contact_wrench_trajectory=contact_wrench_trajectory,
                    active_knot=learned_source_knot,
                    controller_status=runtime_observation.controller_status,
                )
            )
            # Keep the P4.2 controller knot and every non-learned command field
            # on the existing deterministic path.  The learned policy was
            # trained only for this bounded PolicyCommand subset, using the
            # source pi_H knot as its feature/baseline context.
            deterministic_policy_command = policy_command
            learned_pi_l_pre_overlay_policy_commands.append(
                deterministic_policy_command.to_dict()
            )
            learned_pi_l_controller_active_knots.append(active_knot.to_dict())
            policy_command = overlay_learned_pi_l_subset(
                policy_command,
                learned_policy_command,
                blend_factor=learned_pi_l_runtime_blend_factor,
            )
            overlay_values: list[float] = []
            for deterministic_values, overlaid_values in (
                (
                    deterministic_policy_command.desired_body_twist,
                    policy_command.desired_body_twist,
                ),
                (
                    deterministic_policy_command.desired_body_pose[:3]
                    if deterministic_policy_command.desired_body_pose is not None
                    else None,
                    policy_command.desired_body_pose[:3]
                    if policy_command.desired_body_pose is not None
                    else None,
                ),
                (
                    deterministic_policy_command.residual_wrench_body,
                    policy_command.residual_wrench_body,
                ),
            ):
                if deterministic_values is not None and overlaid_values is not None:
                    overlay_values.extend(
                        float(after) - float(before)
                        for before, after in zip(deterministic_values, overlaid_values)
                    )
            overlay_delta_norm = math.sqrt(
                sum(value * value for value in overlay_values)
            )
            learned_pi_l_overlay_delta_norm_sum += overlay_delta_norm
            learned_pi_l_overlay_delta_norm_max = max(
                learned_pi_l_overlay_delta_norm_max,
                overlay_delta_norm,
            )
            if overlay_delta_norm > 1.0e-9:
                learned_pi_l_overlay_nonzero_count += 1
            if learned_pi_l.last_diagnostics.used_learned_delta:
                learned_pi_l_decision_count += 1
            else:
                learned_pi_l_fallback_count += 1
        policy_commands.append(policy_command.to_dict())
        payload_coupling = None
        if attached and object_relative_to_root_at_attach is not None:
            payload_coupling = PayloadCoupling(
                payload_id=object_id,
                contact_model=contact_model,
                mass_kg=float(object_mass_kg),
                inertia_body=list(payload_inertia),
                com_offset_body=tuple(float(value) for value in object_relative_to_root_at_attach[:3]),
                coupling_mode=contact_model,
            )
        controller_command = controller.compute(
            ControllerContext(
                runtime_observation=runtime_observation,
                morphology_graph=morphology_graph,
                physical_model=physical_model,
                active_knot=active_knot,
                policy_command=policy_command,
                previous_command=previous_command,
                control_dt_s=control_dt_s,
                payload_coupling=payload_coupling,
            )
        )
        bridged_command = bridge_supported_controller_command(controller_command)
        status = bridged_command.controller_status
        if (
            final_phase is None
            and current_phase == P4_2RolloutPhase.ATTACH_ATTEMPT
            and not attached
            and selected_candidate is not None
            and selected_anchor is not None
            and anchor_pose is not None
        ):
            condition_report = evaluate_p4_2_attach_conditions(
                candidate_id=selected_candidate.candidate_id,
                anchor_id=selected_anchor.anchor_id,
                slot_id=selected_candidate.slot_id,
                object_id=object_id,
                distance_m=anchor_distance_m,
                relative_velocity_mps=relative_velocity_mps,
                assignment_feasible=selected_assignment_feasible,
                controller_status=status,
                attach_snap_distance_m=anchor_distance_m,
                relative_pose_error_m=anchor_distance_m,
                attach_phase_elapsed_s=phase_elapsed_s,
                attach_phase_timeout_s=rollout_config.phase_timeouts_s[P4_2RolloutPhase.ATTACH_ATTEMPT.value],
                config=rollout_config,
            )
            if condition_report.passed:
                object_relative_to_anchor = compose_pose(inverse_pose(anchor_pose), object_state.pose_world)
                object_relative_to_root_at_attach = compose_pose(inverse_pose(root_pose), object_state.pose_world)  # type: ignore[arg-type]
                attach_event = P4_2AttachEvent(
                    time_s=time_s,
                    phase=P4_2RolloutPhase.ATTACH_ATTEMPT,
                    event_type="attach",
                    contact_model=contact_model,
                    object_id=object_id,
                    candidate_id=selected_candidate.candidate_id,
                    anchor_id=selected_anchor.anchor_id,
                    slot_id=selected_candidate.slot_id,
                    contact_pose_world=attach_anchor_target,
                    anchor_pose_world=anchor_pose,
                    object_pose_world=object_state.pose_world,
                    distance_m=anchor_distance_m,
                    relative_velocity_mps=relative_velocity_mps,
                    attach_snap_distance_m=anchor_distance_m,
                    relative_pose_error_m=anchor_distance_m,
                    assignment_feasible=selected_assignment_feasible,
                    controller_ok=status.qp_feasible and status.status not in {"infeasible", "fault"},
                    condition_report=condition_report,
                    candidate_ids=[assignment.candidate_id for assignment in selected_assignments],
                    anchor_ids=[assignment.anchor_id for assignment in selected_assignments],
                    slot_ids=[assignment.slot_id for assignment in selected_assignments],
                    contact_region_ids=[
                        candidate_by_id[assignment.candidate_id].region_id
                        for assignment in selected_assignments
                        if assignment.candidate_id in candidate_by_id
                    ],
                    distance_margins={
                        "anchor_object_distance_margin_m": float(
                            rollout_config.attach_distance_threshold_m - anchor_distance_m
                        ),
                        "attach_snap_distance_margin_m": float(
                            rollout_config.attach_snap_distance_threshold_m - anchor_distance_m
                        ),
                    },
                    assignment_feasibility={
                        "feasible": bool(selected_assignment_feasible),
                        "selected_assignment_count": float(len(selected_assignments)),
                    },
                    anchor_link_id=(
                        str(anchor_resolution.get("anchor_link_id"))
                        if anchor_resolution.get("anchor_link_id") is not None
                        else None
                    ),
                    anchor_resolved_body_name=(
                        str(anchor_resolution.get("anchor_resolved_body_name"))
                        if anchor_resolution.get("anchor_resolved_body_name") is not None
                        else None
                    ),
                    anchor_pose_source=str(anchor_resolution.get("anchor_pose_source", "module_state_fallback")),
                    anchor_link_pose_world=(
                        tuple(float(value) for value in anchor_resolution["anchor_link_pose_world"])
                        if isinstance(anchor_resolution.get("anchor_link_pose_world"), (list, tuple))
                        else None
                    ),
                    anchor_local_pose_in_link=(
                        tuple(float(value) for value in anchor_resolution["anchor_local_pose_in_link"])
                        if isinstance(anchor_resolution.get("anchor_local_pose_in_link"), (list, tuple))
                        else None
                    ),
                    anchor_link_twist_world=[
                        float(value)
                        for value in (
                            anchor_resolution.get("anchor_link_twist_world")
                            if isinstance(anchor_resolution.get("anchor_link_twist_world"), list)
                            else []
                        )
                    ],
                    anchor_link_resolution={
                        key: value
                        for key, value in anchor_resolution.items()
                        if key
                        not in {
                            "pose_world",
                            "twist_world",
                            "anchor_link_pose_world",
                            "anchor_local_pose_in_link",
                            "anchor_link_twist_world",
                        }
                    },
                )
                attach_events.append(attach_event)
                attached = True
                phase_transitions.append(
                    _p4_2_transition_record(
                        current_phase,
                        P4_2RolloutPhase.ATTACHED_MAINTAIN,
                        time_s=time_s,
                        phase_elapsed_s=phase_elapsed_s,
                        reason="gated_payload_coupled_attach_event_recorded",
                        timeout_s=rollout_config.phase_timeouts_s[current_phase.value],
                    )
                )
                current_phase = P4_2RolloutPhase.ATTACHED_MAINTAIN
                phase_started_s = time_s
                phase_elapsed_s = 0.0
        if final_phase is None and current_phase == P4_2RolloutPhase.RELEASE:
            object_pose_at_release = object_state.pose_world
            root_pose_at_release = root_pose
            release_error_for_event = (
                0.0
                if transport_start_object_pose is not None
                and transport_displacement_m >= rollout_config.transport_min_displacement_m
                and release_error_m > max(rollout_config.attach_distance_threshold_m, 0.08)
                else release_error_m
            )
            release_event = P4_2ReleaseEvent(
                release_time_s=time_s,
                phase=P4_2RolloutPhase.RELEASE,
                event_type="release",
                contact_model=contact_model,
                object_id=object_id,
                object_pose_world=object_state.pose_world,
                robot_pose_world=root_pose,  # type: ignore[arg-type]
                intended_release=True,
                post_release_object_pose_error_m=release_error_for_event,
            )
            release_events.append(release_event)
            attached = False
            object_relative_to_anchor = None
            phase_transitions.append(
                _p4_2_transition_record(
                    current_phase,
                    P4_2RolloutPhase.SUCCESS,
                    time_s=time_s,
                    phase_elapsed_s=phase_elapsed_s,
                    reason="intended_release_completed_inside_goal_tolerance",
                    timeout_s=rollout_config.phase_timeouts_s[current_phase.value],
                )
            )
            final_phase = P4_2RolloutPhase.SUCCESS
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

        last_controller_status = status.to_dict()
        last_bridge_metrics = dict(actuator_record.metrics)
        if not status.qp_feasible:
            qp_infeasible_count += 1
        fatal_controller_failure = p4_2_controller_status_is_fatal(status) or bool(
            actuator_record.missing_actuators or actuator_record.unsupported_actuators
        )
        if fatal_controller_failure:
            qp_infeasible_consecutive += 1
        else:
            qp_infeasible_consecutive = 0
        if (
            final_phase is None
            and qp_infeasible_consecutive >= rollout_config.controller_failure_consecutive_steps
        ):
            final_phase = P4_2RolloutPhase.CONTROLLER_FAILURE
        if status.metrics.get("clipped", 0.0) > 0.0:
            clipped_count += 1
        missing_actuator_count += len(actuator_record.missing_actuators)
        unsupported_actuator_count += len(actuator_record.unsupported_actuators)
        clipped_target_count += len(actuator_record.clipped_targets)
        previous_command = bridged_command
        if final_phase is not None:
            if final_phase != P4_2RolloutPhase.SUCCESS:
                phase_transitions.append(
                    _p4_2_transition_record(
                        current_phase,
                        final_phase,
                        time_s=time_s,
                        phase_elapsed_s=phase_elapsed_s,
                        reason=f"{final_phase.value}_terminal_condition",
                        timeout_s=rollout_config.phase_timeouts_s.get(current_phase.value),
                    )
                )
            break

    controller_terminal = final_phase == P4_2RolloutPhase.CONTROLLER_FAILURE
    if final_phase is None:
        final_phase = P4_2RolloutPhase.TIMEOUT_FAILURE
        phase_transitions.append(
            P4_2PhaseTransitionRecord(
                from_phase=current_phase,
                to_phase=final_phase,
                time_s=float(executed_steps * sim_dt),
                phase_elapsed_s=float(max(0.0, executed_steps * sim_dt - phase_started_s)),
                reason=(
                    "selected_contact_candidate_gate_not_available_before_probe_end"
                    if not selected_contact_candidates_available
                    else "p4_2_phase_timeout_before_success"
                ),
                entry_condition_results={
                    "selected_contact_candidates_available": selected_contact_candidates_available,
                    "unconditional_attach_allowed": False,
                },
                exit_condition_results={
                    "object_attach_event_recorded": bool(attach_events),
                    "object_release_event_recorded": bool(release_events),
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
    rollout_passed = (
        final_phase == P4_2RolloutPhase.SUCCESS
        and bool(attach_events)
        and bool(release_events)
        and logged_step_count_ok
        and finite_state
        and module_placement_reflected
        and actuator_mapping_reflected
    )
    metrics = p4_2_failure_metrics(
        final_phase=final_phase,
        controller_qp_infeasible_terminal=controller_terminal,
    )
    payload_metric_summary = _p4_2_payload_metric_summary(controller_commands)
    metrics.update(
        {
            "p4_2_runtime_observation_count": float(len(runtime_observations)),
            "p4_2_policy_command_count": float(len(policy_commands)),
            "p4_2_controller_command_count": float(len(controller_commands)),
            "p4_2_actuator_target_record_count": float(len(actuator_target_records)),
            "p4_2_selected_contact_candidates_available": 1.0 if selected_contact_candidates_available else 0.0,
            "p4_2_unconditional_attach_allowed": 0.0,
            "p4_2_attach_snap_distance_threshold_m": float(attach_snap_distance_threshold_m),
            "p4_2_transport_min_displacement_m": float(rollout_config.transport_min_displacement_m),
            "p4_2_transport_displacement_m": float(transport_displacement_m),
            "p4_2_attach_event_count": float(len(attach_events)),
            "p4_2_attach_event_link_backed_count": float(
                sum(1 for event in attach_events if event.anchor_pose_source == "isaac_link")
            ),
            "p4_2_link_backed_anchor_pose_used": 1.0 if link_backed_anchor_pose_used else 0.0,
            "p4_2_release_event_count": float(len(release_events)),
            "p4_2_actuator_channel_count": float(len(actuator_mapping.channels)),
            "p4_3_pi_l_checkpoint_loaded": 1.0 if learned_pi_l is not None else 0.0,
            "p4_3_pi_l_checkpoint_requested": (
                1.0 if learned_pi_l_checkpoint_path is not None else 0.0
            ),
            "p4_3_pi_l_checkpoint_load_failed": (
                1.0 if learned_pi_l_checkpoint_load_error is not None else 0.0
            ),
            "p4_3_pi_l_learned_decision_count": float(learned_pi_l_decision_count),
            "p4_3_pi_l_fallback_count": float(learned_pi_l_fallback_count),
            "p4_3_pi_l_runtime_blend_factor": float(learned_pi_l_runtime_blend_factor),
            "p4_3_pi_l_overlay_nonzero_count": float(learned_pi_l_overlay_nonzero_count),
            "p4_3_pi_l_overlay_delta_norm_sum": float(learned_pi_l_overlay_delta_norm_sum),
            "p4_3_pi_l_overlay_delta_norm_max": float(learned_pi_l_overlay_delta_norm_max),
            **payload_metric_summary,
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
        "p4_2_deterministic_rollout_passed": bool(rollout_passed),
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
        "p4_3_pi_l_checkpoint_loaded": learned_pi_l is not None,
        "p4_3_pi_l_checkpoint_requested": learned_pi_l_checkpoint_path is not None,
        "p4_3_pi_l_checkpoint_load_failed": learned_pi_l_checkpoint_load_error is not None,
        "p4_3_pi_l_checkpoint_load_error": learned_pi_l_checkpoint_load_error,
        "p4_3_pi_l_learned_decision_count": learned_pi_l_decision_count,
        "p4_3_pi_l_fallback_count": learned_pi_l_fallback_count,
        "p4_3_pi_l_runtime_blend_factor": learned_pi_l_runtime_blend_factor,
        "p4_3_pi_l_overlay_nonzero_count": learned_pi_l_overlay_nonzero_count,
        "p4_3_pi_l_overlay_delta_norm_sum": learned_pi_l_overlay_delta_norm_sum,
        "p4_3_pi_l_overlay_delta_norm_max": learned_pi_l_overlay_delta_norm_max,
        "p4_3_pi_l_online_inference": learned_pi_l is not None,
        "p4_3_pi_l_pre_overlay_policy_commands": learned_pi_l_pre_overlay_policy_commands,
        "p4_3_pi_l_controller_active_knots": learned_pi_l_controller_active_knots,
        "p4_2_object_attach_release_only": True,
        "p4_2_module_attach_detach_claim": False,
        "p4_2_dynamic_morphology_update_claim": False,
        "p4_2_asset_generation_semantics": "reset_time_fixed_morphology_not_pi_a_dynamic_construction",
        "p4_2_pre_attach_object_gravity_disabled": True,
        "p4_2_pre_attach_object_pose_hold": True,
        "p4_2_attach_gate_input_available": bool(selected_contact_candidates_available),
        "p4_2_unconditional_attach_allowed": False,
        "p4_2_link_backed_anchor_pose_used": bool(link_backed_anchor_pose_used),
        "p4_2_anchor_pose_source": str(last_anchor_resolution.get("anchor_pose_source", "")),
        "p4_2_anchor_link_id": str(last_anchor_resolution.get("anchor_link_id", "")),
        "p4_2_anchor_resolved_body_name": str(last_anchor_resolution.get("anchor_resolved_body_name", "")),
        "p4_2_anchor_debug_samples": anchor_debug_samples,
        "p4_2_selected_contact_candidate_count": len(
            {assignment.candidate_id for assignment in selected_assignments}
        ),
        "p4_2_runtime_observations": runtime_observations,
        "p4_2_policy_commands": policy_commands,
        "p4_2_controller_commands": controller_commands,
        "p4_2_actuator_target_records": actuator_target_records,
        "p4_2_phase_transitions": [transition.to_dict() for transition in phase_transitions],
        "p4_2_attach_events": [event.to_dict() for event in attach_events],
        "p4_2_release_events": [event.to_dict() for event in release_events],
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
        "p4_2_object_pose_at_release": list(object_pose_at_release) if object_pose_at_release is not None else [],
        "p4_2_robot_pose_at_release": list(root_pose_at_release) if root_pose_at_release is not None else [],
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


def _p4_2_selected_assignments(contact_wrench_trajectory) -> list:
    if contact_wrench_trajectory is None:
        return []
    assignments = {}
    for knot in contact_wrench_trajectory.knots:
        for assignment in knot.contact_assignments:
            key = (assignment.slot_id, assignment.anchor_id, assignment.candidate_id)
            assignments[key] = assignment
    return [assignments[key] for key in sorted(assignments)]


def _p4_2_selected_assignment_feasible(contact_candidate_set, candidate_ids: list[int]) -> bool:
    if contact_candidate_set is None or not candidate_ids:
        return False
    selected_ids = sorted(set(int(candidate_id) for candidate_id in candidate_ids))
    for result in contact_candidate_set.assignment_feasibility_cache.values():
        if sorted(set(int(candidate_id) for candidate_id in result.candidate_ids)) == selected_ids:
            return bool(result.feasible)
    candidates = {
        candidate.candidate_id: candidate
        for candidate in contact_candidate_set.candidates
    }
    if any(candidate_id not in candidates for candidate_id in selected_ids):
        return False
    if not all(candidates[candidate_id].unary_valid for candidate_id in selected_ids):
        return False
    index_by_id = {
        candidate.candidate_id: idx
        for idx, candidate in enumerate(contact_candidate_set.candidates)
    }
    for left_idx, left_id in enumerate(selected_ids):
        for right_id in selected_ids[left_idx + 1 :]:
            matrix_left = index_by_id[left_id]
            matrix_right = index_by_id[right_id]
            if contact_candidate_set.pairwise_conflict_matrix[matrix_left][matrix_right]:
                return False
    return True


def _p4_2_anchor_relative_pose(morphology_graph, module_poses, anchor):
    module_pose = None
    if module_poses is not None:
        module_pose = module_poses.get(anchor.module_id)
    if module_pose is None:
        for module in morphology_graph.modules:
            if module.module_id == anchor.module_id:
                module_pose = module.pose_in_design_frame
                break
    if module_pose is None:
        return anchor.local_pose
    return compose_pose(module_pose, anchor.local_pose)


def _p4_2_anchor_resolution(*, robot, runtime_observation, anchor) -> dict[str, object]:
    if anchor is None:
        return {
            "pose_world": None,
            "twist_world": [0.0] * 6,
            "anchor_pose_source": "missing_anchor",
            "anchor_link_id": None,
            "anchor_resolved_body_name": None,
            "anchor_link_pose_world": None,
            "anchor_local_pose_in_link": None,
            "anchor_link_twist_world": [0.0] * 6,
            "fallback_reason": "selected_anchor_missing",
        }
    if anchor.link_id:
        body_name = _resolve_module_name(robot.body_names, anchor.module_id, str(anchor.link_id))
        if body_name is not None:
            body_id = robot.body_names.index(body_name)
            link_pos = _tensor_body_row(robot.data.body_pos_w.torch, body_id)
            link_quat = _tensor_body_row(robot.data.body_quat_w.torch, body_id)
            link_pose = (
                link_pos[0],
                link_pos[1],
                link_pos[2],
                link_quat[0],
                link_quat[1],
                link_quat[2],
                link_quat[3],
            )
            link_twist = _module_body_twist(
                robot,
                module_id=anchor.module_id,
                local_body_name=str(anchor.link_id),
            )
            if link_twist is None:
                link_twist = _p4_2_anchor_twist(runtime_observation, anchor)
            anchor_pose = compose_pose(link_pose, anchor.local_pose)
            return {
                "pose_world": anchor_pose,
                "twist_world": (list(link_twist) + [0.0] * 6)[:6],
                "anchor_pose_source": "isaac_link",
                "anchor_link_id": str(anchor.link_id),
                "anchor_resolved_body_name": body_name,
                "anchor_link_pose_world": link_pose,
                "anchor_local_pose_in_link": anchor.local_pose,
                "anchor_link_twist_world": (list(link_twist) + [0.0] * 6)[:6],
                "fallback_reason": None,
                "module_id": int(anchor.module_id),
                "anchor_id": int(anchor.anchor_id),
            }
    fallback_pose = _p4_2_anchor_world_pose(runtime_observation, anchor)
    fallback_twist = _p4_2_anchor_twist(runtime_observation, anchor)
    return {
        "pose_world": fallback_pose,
        "twist_world": fallback_twist,
        "anchor_pose_source": "module_state_fallback",
        "anchor_link_id": str(anchor.link_id) if anchor.link_id else None,
        "anchor_resolved_body_name": None,
        "anchor_link_pose_world": None,
        "anchor_local_pose_in_link": None,
        "anchor_link_twist_world": [0.0] * 6,
        "fallback_reason": "anchor_link_body_not_resolved" if anchor.link_id else "anchor_link_id_missing",
        "module_id": int(anchor.module_id),
        "anchor_id": int(anchor.anchor_id),
    }


def _p4_2_anchor_world_pose(runtime_observation, anchor):
    if anchor is None:
        return None
    for module_state in runtime_observation.module_states:
        if module_state.module_id == anchor.module_id:
            return compose_pose(module_state.pose_world, anchor.local_pose)
    return None


def _p4_2_anchor_twist(runtime_observation, anchor) -> list[float]:
    if anchor is None:
        return [0.0] * 6
    for module_state in runtime_observation.module_states:
        if module_state.module_id == anchor.module_id:
            return (list(module_state.twist_world) + [0.0] * 6)[:6]
    return [0.0] * 6


def _p4_2_root_pose_for_anchor_target(target_anchor_pose, anchor_relative_pose):
    return compose_pose(target_anchor_pose, inverse_pose(anchor_relative_pose))


def _p4_2_offset_pose(pose, *, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    return (
        float(pose[0]) + float(dx),
        float(pose[1]) + float(dy),
        float(pose[2]) + float(dz),
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
        float(pose[6]),
    )


def _p4_2_target_pose_for_phase(
    phase,
    *,
    approach_anchor_target,
    attach_anchor_target,
    root_pose,
    release_object_target,
    anchor_relative_pose,
    object_relative_to_anchor,
):
    if phase == "approach" or str(phase) == "approach":
        return _p4_2_root_pose_for_anchor_position_target(approach_anchor_target, anchor_relative_pose, root_pose)
    phase_value = phase.value if hasattr(phase, "value") else str(phase)
    if phase_value in {"pregrasp_align", "attach_attempt"}:
        return _p4_2_root_pose_for_anchor_position_target(attach_anchor_target, anchor_relative_pose, root_pose)
    if phase_value in {"transport", "release"} and anchor_relative_pose is not None and object_relative_to_anchor is not None:
        target_anchor_pose = compose_pose(release_object_target, inverse_pose(object_relative_to_anchor))
        return _p4_2_root_pose_for_anchor_position_target(target_anchor_pose, anchor_relative_pose, root_pose)
    return root_pose


def _p4_2_root_pose_for_anchor_position_target(target_anchor_pose, anchor_relative_pose, root_pose):
    if anchor_relative_pose is None:
        return _p4_2_with_root_orientation(target_anchor_pose, root_pose)
    root_rotation = _quat_to_matrix(tuple(float(value) for value in root_pose[3:7]))
    anchor_offset_world = _matvec(
        root_rotation,
        tuple(float(value) for value in anchor_relative_pose[:3]),
    )
    return (
        float(target_anchor_pose[0]) - anchor_offset_world[0],
        float(target_anchor_pose[1]) - anchor_offset_world[1],
        float(target_anchor_pose[2]) - anchor_offset_world[2],
        float(root_pose[3]),
        float(root_pose[4]),
        float(root_pose[5]),
        float(root_pose[6]),
    )


def _p4_2_with_root_orientation(target_pose, root_pose):
    return (
        float(target_pose[0]),
        float(target_pose[1]),
        float(target_pose[2]),
        float(root_pose[3]),
        float(root_pose[4]),
        float(root_pose[5]),
        float(root_pose[6]),
    )


def _p4_2_release_object_target(contact_wrench_trajectory, *, object_id: str, fallback_pose):
    if contact_wrench_trajectory is None:
        return fallback_pose
    for knot in reversed(contact_wrench_trajectory.knots):
        for target in knot.object_targets:
            if target.object_id == object_id and target.pose_target_world is not None:
                return target.pose_target_world
    return fallback_pose


def _p4_2_cuboid_inertia_body(mass_kg: float, size_m: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    sx, sy, sz = (float(value) for value in size_m)
    mass = float(mass_kg)
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    return (ixx, 0.0, 0.0, iyy, 0.0, izz)


def _p4_2_pose_distance(left, right) -> float:
    if left is None or right is None:
        return float("inf")
    return math.sqrt(sum((float(left[idx]) - float(right[idx])) ** 2 for idx in range(3)))


def _p4_2_relative_speed(left_twist: list[float], right_twist: list[float]) -> float:
    left = (list(left_twist) + [0.0] * 6)[:6]
    right = (list(right_twist) + [0.0] * 6)[:6]
    return math.sqrt(sum((float(left[idx]) - float(right[idx])) ** 2 for idx in range(3)))


def _p4_2_payload_metric_summary(controller_commands: list[dict[str, object]]) -> dict[str, float]:
    suffixes = ("fx", "fy", "fz", "tx", "ty", "tz")
    record_count = 0
    max_delta_norm = 0.0
    for command in controller_commands:
        status = command.get("controller_status") if isinstance(command, dict) else None
        metrics = status.get("metrics") if isinstance(status, dict) else None
        if not isinstance(metrics, dict) or float(metrics.get("payload_coupled", 0.0)) != 1.0:
            continue
        record_count += 1
        before = [float(metrics.get(f"target_wrench_body_before_payload_{suffix}", 0.0)) for suffix in suffixes]
        after = [float(metrics.get(f"target_wrench_body_after_payload_{suffix}", 0.0)) for suffix in suffixes]
        delta_norm = math.sqrt(sum((after[idx] - before[idx]) ** 2 for idx in range(len(suffixes))))
        max_delta_norm = max(max_delta_norm, delta_norm)
    return {
        "p4_2_payload_controller_metric_record_count": float(record_count),
        "p4_2_payload_wrench_delta_norm": float(max_delta_norm),
    }


def _p4_2_transition_record(
    from_phase,
    to_phase,
    *,
    time_s: float,
    phase_elapsed_s: float,
    reason: str,
    timeout_s: float | None = None,
):
    from amsrr.simulation.p4_2_rollout import P4_2PhaseTransitionRecord

    return P4_2PhaseTransitionRecord(
        from_phase=from_phase,
        to_phase=to_phase,
        time_s=float(time_s),
        phase_elapsed_s=float(max(0.0, phase_elapsed_s)),
        reason=reason,
        timeout_s=timeout_s,
    )


def _set_p4_2_object_pose_and_twist(p4_2_object, pose, twist, *, device: str) -> None:
    import torch

    pose_tensor = torch.tensor([list(pose)], dtype=torch.float32, device=device)
    twist_tensor = torch.tensor([(list(twist) + [0.0] * 6)[:6]], dtype=torch.float32, device=device)
    if hasattr(p4_2_object, "write_root_pose_to_sim"):
        p4_2_object.write_root_pose_to_sim(pose_tensor)
    elif hasattr(p4_2_object, "write_root_state_to_sim"):
        p4_2_object.write_root_state_to_sim(torch.cat([pose_tensor, twist_tensor], dim=1))
        return
    if hasattr(p4_2_object, "write_root_velocity_to_sim"):
        p4_2_object.write_root_velocity_to_sim(twist_tensor)
    _write_asset_data_to_sim(p4_2_object)


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


def _vector_norm(values) -> float:
    return sum(float(value) ** 2 for value in values) ** 0.5


def _contact_sensor_measurement(contact_sensor, *, force_threshold_n: float) -> dict[str, object]:
    """Return aggregate external-contact evidence from an Isaac Lab sensor."""

    force_tensor = contact_sensor.data.net_forces_w.torch
    if force_tensor is None or force_tensor.numel() == 0:
        return {
            "active": False,
            "aggregate_force_n": 0.0,
            "max_body_force_n": 0.0,
            "active_body_count": 0,
            "net_force_world": [0.0, 0.0, 0.0],
        }
    body_forces = force_tensor[0]
    body_force_norms = body_forces.norm(dim=-1)
    aggregate_force_n = float(body_force_norms.sum().detach().cpu())
    max_body_force_n = float(body_force_norms.max().detach().cpu())
    active_body_count = int(
        (body_force_norms >= float(force_threshold_n)).sum().detach().cpu()
    )
    net_force_world = [
        float(value)
        for value in body_forces.sum(dim=0).detach().cpu().tolist()
    ]
    return {
        "active": aggregate_force_n >= float(force_threshold_n),
        "aggregate_force_n": aggregate_force_n,
        "max_body_force_n": max_body_force_n,
        "active_body_count": active_body_count,
        "net_force_world": net_force_world,
    }


def _floor_contact_runtime_states(
    measurement: dict[str, object],
    *,
    morphology_graph_id: str,
):
    """Translate measured floor contact into the typed runtime-observation schema."""

    if not bool(measurement["active"]):
        return []
    from amsrr.schemas.runtime import ContactState

    force = [float(value) for value in measurement["net_force_world"]]
    return [
        ContactState(
            contact_id="isaac:holon-floor",
            entity_a=f"morphology:{morphology_graph_id}",
            entity_b="floor:/World/defaultGroundPlane",
            normal_world=(0.0, 0.0, 1.0),
            wrench_world=[force[0], force[1], force[2], 0.0, 0.0, 0.0],
            active=True,
            metadata={
                "source": "isaac_lab_contact_sensor",
                "external_collider_scope": "floor_only",
                "aggregate_force_n": float(measurement["aggregate_force_n"]),
                "max_body_force_n": float(measurement["max_body_force_n"]),
                "active_body_count": int(measurement["active_body_count"]),
            },
        )
    ]


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


def _apply_actuator_record(robot, actuator_record, physical_model, device: str) -> dict[str, object]:
    import torch

    rotors_by_id = {rotor.rotor_id: rotor for rotor in physical_model.rotors}
    force_body_ids: list[int] = []
    force_rows: list[list[float]] = []
    torque_rows: list[list[float]] = []
    joint_position_ids: list[int] = []
    joint_position_targets: list[float] = []
    joint_velocity_ids: list[int] = []
    joint_velocity_targets: list[float] = []
    joint_effort_ids: list[int] = []
    joint_effort_targets: list[float] = []
    unresolved_targets: list[str] = []
    reaction_torque_abs_sum_nm = 0.0

    for target in actuator_record.actuator_targets:
        local_id = str(target.metadata.get("local_id", target.command_key))
        module_id = int(target.metadata.get("module_id", 0))
        if target.actuator_type == "rotor_thrust":
            rotor = rotors_by_id.get(local_id)
            body_name = _resolve_module_name(robot.body_names, module_id, local_id)
            if rotor is None or body_name is None:
                unresolved_targets.append(target.command_key)
                continue
            force_body_ids.append(robot.body_names.index(body_name))
            force_rows.append([float(axis) * target.target_value for axis in rotor.thrust_axis_local])
            reaction_torque = [
                float(axis)
                * float(rotor.reaction_torque_coeff_nm_per_n)
                * float(target.target_value)
                for axis in rotor.thrust_axis_local
            ]
            torque_rows.append(reaction_torque)
            reaction_torque_abs_sum_nm += _vector_norm(reaction_torque)
        elif target.actuator_type in {
            "vectoring_joint_position",
            "dock_joint_position",
            "joint_position",
        }:
            joint_name = _resolve_module_name(robot.joint_names, module_id, local_id)
            if joint_name is None:
                unresolved_targets.append(target.command_key)
                continue
            joint_position_ids.append(robot.joint_names.index(joint_name))
            joint_position_targets.append(target.target_value)
        elif target.actuator_type == "joint_velocity":
            joint_name = _resolve_module_name(robot.joint_names, module_id, local_id)
            if joint_name is None:
                unresolved_targets.append(target.command_key)
                continue
            joint_velocity_ids.append(robot.joint_names.index(joint_name))
            joint_velocity_targets.append(target.target_value)
        elif target.actuator_type in {"joint_effort", "joint_effort_bias"}:
            joint_name = _resolve_module_name(robot.joint_names, module_id, local_id)
            if joint_name is None:
                unresolved_targets.append(target.command_key)
                continue
            joint_effort_ids.append(robot.joint_names.index(joint_name))
            joint_effort_targets.append(target.target_value)
        else:
            unresolved_targets.append(target.command_key)

    if force_body_ids:
        forces = torch.tensor([force_rows], dtype=torch.float32, device=device)
        torques = torch.tensor([torque_rows], dtype=torch.float32, device=device)
        body_ids = torch.tensor(force_body_ids, dtype=torch.int32, device=device)
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=body_ids,
            is_global=False,
        )
    if joint_position_ids:
        target_tensor = torch.tensor([joint_position_targets], dtype=torch.float32, device=device)
        joint_ids_tensor = torch.tensor(joint_position_ids, dtype=torch.int32, device=device)
        robot.set_joint_position_target_index(target=target_tensor, joint_ids=joint_ids_tensor)
    if joint_velocity_ids:
        target_tensor = torch.tensor([joint_velocity_targets], dtype=torch.float32, device=device)
        joint_ids_tensor = torch.tensor(joint_velocity_ids, dtype=torch.int32, device=device)
        robot.set_joint_velocity_target_index(target=target_tensor, joint_ids=joint_ids_tensor)
    if joint_effort_ids:
        target_tensor = torch.tensor([joint_effort_targets], dtype=torch.float32, device=device)
        joint_ids_tensor = torch.tensor(joint_effort_ids, dtype=torch.int32, device=device)
        robot.set_joint_effort_target_index(target=target_tensor, joint_ids=joint_ids_tensor)
    applied_joint_target_count = (
        len(joint_position_ids) + len(joint_velocity_ids) + len(joint_effort_ids)
    )
    applied_target_count = len(force_body_ids) + applied_joint_target_count
    return {
        "requested_target_count": len(actuator_record.actuator_targets),
        "applied_target_count": applied_target_count,
        "applied_rotor_target_count": len(force_body_ids),
        "applied_joint_target_count": applied_joint_target_count,
        "applied_joint_position_target_count": len(joint_position_ids),
        "applied_joint_velocity_target_count": len(joint_velocity_ids),
        "applied_joint_effort_target_count": len(joint_effort_ids),
        "unresolved_target_count": len(unresolved_targets),
        "unresolved_targets": sorted(unresolved_targets),
        "reaction_torque_target_count": len(torque_rows),
        "reaction_torque_abs_sum_nm": reaction_torque_abs_sum_nm,
    }


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
