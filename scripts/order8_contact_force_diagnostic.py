from __future__ import annotations

"""Accelerated, acceptance-ineligible Order 8 contact-force diagnostic."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.robot_model.gripper_surfaces import (
    select_opposing_gripper_surface_pair,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import (
    link_poses_in_root_frame,
    module_base_link_name,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematicsConfig,
    _build_model_context,
    _module_link_poses,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order8 import load_order8_natural_contact_config
from amsrr.schemas.common import Pose7D
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.simulation.isaac_lab_backend import (
    IsaacLabBackend,
    load_isaac_lab_backend_config,
)
from amsrr.simulation.order8_natural_contact import (
    ORDER8_DEFAULT_GENERATED_USD_DIR,
    Order8IsaacNaturalContactEnv,
    _run_json_command,
    build_representative_order8_morphology,
)
from amsrr.simulation.order8_state_trace import load_order8_state_trace
from amsrr.utils.hashing import hash_file

DEFAULT_CONFIG_PATH = "configs/training/order8_natural_contact.yaml"
DEFAULT_BACKEND_CONFIG_PATH = "configs/env/isaac_lab.yaml"
DEFAULT_REPORT_PATH = Path(
    "artifacts/p4_full/order8_natural_contact/diagnostics/" "contact_force_fast.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a temporary accelerated Order-8 contact-force fault-isolation "
            "episode. This command can never satisfy acceptance."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--backend-config", default=DEFAULT_BACKEND_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-scale", type=float, default=1.5)
    parser.add_argument("--force-ramp-s", type=float, default=4.0)
    parser.add_argument(
        "--contact-dwell-s",
        type=float,
        default=None,
        help=(
            "Optional acceptance-ineligible contact-dwell override. Fixture "
            "runs default to 0.05 s; specify the production value when the "
            "dwell/ramp boundary itself is under test."
        ),
    )
    parser.add_argument("--stop-force-scale", type=float, default=0.40)
    parser.add_argument(
        "--force-anchor-id",
        type=int,
        action="append",
        default=None,
        help=(
            "Acceptance-ineligible force-isolation mask. Repeat to excite "
            "more than one selected anchor."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.020,
        help="Temporary diagnostic physics step; acceptance keeps backend dt.",
    )
    parser.add_argument(
        "--object-width-padding-m",
        type=float,
        default=0.040,
        help=(
            "Temporary increase of the opposing-gripper object width so the "
            "fixture starts near contact; this never changes acceptance config."
        ),
    )
    parser.add_argument(
        "--dock-stiffness",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only Dock position-drive stiffness override. "
            "Omit it to retain the configured production value."
        ),
    )
    parser.add_argument(
        "--dock-damping",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only Dock position-drive damping override. "
            "Omit it to retain the configured production value."
        ),
    )
    parser.add_argument(
        "--dock-velocity-limit",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only Dock joint physics velocity limit. "
            "It must not exceed the configured AK40-10 simulation limit."
        ),
    )
    parser.add_argument(
        "--dock-armature-kg-m2",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only reflected Dock-joint armature. This "
            "changes simulator inertia only; AK40-10 torque, speed, and "
            "current limits remain unchanged."
        ),
    )
    parser.add_argument(
        "--object-mass-kg",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only object-mass override. Omit it to "
            "retain the configured 1 kg production payload."
        ),
    )
    parser.add_argument(
        "--object-friction",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only object static/dynamic friction "
            "override. Omit it to retain the configured production value."
        ),
    )
    parser.add_argument(
        "--selected-gripper-friction",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only friction override for the two selected "
            "Dock rigid bodies and any enabled proxy pads. Omit it to retain "
            "the configured production value."
        ),
    )
    parser.add_argument(
        "--selected-gripper-compliant-contact-stiffness",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic override for the PhysX compliant-contact "
            "stiffness on the complete selected authored Dock meshes."
        ),
    )
    parser.add_argument(
        "--selected-gripper-compliant-contact-damping",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic override for the PhysX compliant-contact "
            "damping on the complete selected authored Dock meshes."
        ),
    )
    parser.add_argument(
        "--proxy-pad",
        action="store_true",
        help=(
            "Add acceptance-ineligible 30 x 30 x 2 mm finite-area colliders "
            "3 mm beyond the retained selected Dock meshes."
        ),
    )
    parser.add_argument(
        "--cone-proxy-pad",
        action="store_true",
        help=(
            "Use the visually approved cone-only merged micro-pad colliders "
            "in the acceptance-ineligible live-physics diagnostic."
        ),
    )
    parser.add_argument(
        "--contact-closure-joint-speed-radps",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only fixed joint-space q_close speed. It "
            "may only slow the ordinary closure command."
        ),
    )
    parser.add_argument(
        "--normal-force-target-n",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only normal-force target per selected "
            "contact. Omit it to retain the configured production value."
        ),
    )
    parser.add_argument(
        "--payload-load-transfer-s",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only shared LIFT motion/feed-forward "
            "progress duration. Omit it to retain the configured production "
            "value."
        ),
    )
    parser.add_argument(
        "--contact-stall-speed-threshold-mps",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only non-privileged q_close/force-settle "
            "relative-speed threshold. Raw contact slip safety is unchanged."
        ),
    )
    parser.add_argument(
        "--max-slip-speed-mps",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only selected-contact slip-speed limit. "
            "Use it only to distinguish a short load-transfer transient from "
            "continued sliding; acceptance always keeps the source limit."
        ),
    )
    parser.add_argument(
        "--max-contact-point-slip-displacement-m",
        "--max-cumulative-slip-m",
        dest="max_contact_point_slip_displacement_m",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only limit on each selected contact "
            "centre's object-frame displacement norm from its grasp-"
            "confirmation reference. The old cumulative-slip spelling is a "
            "deprecated command-line alias."
        ),
    )
    parser.add_argument(
        "--peak-torque-window-s",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only AK40-10 peak-torque window after "
            "simultaneous q_close. The runtime ramps the torque-bias limit "
            "back to the 1.3 Nm continuous rating before the window ends; "
            "acceptance never enables this override."
        ),
    )
    parser.add_argument(
        "--post-grasp-joint-torque-bias-nm",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only equal-magnitude offset torque for each "
            "grasp-contributing Dock joint after load preload. The runtime "
            "signs it using the fixed closure direction and retains the "
            "AK40-10 continuous/hard envelope audits."
        ),
    )
    parser.add_argument(
        "--disable-slip-speed-safe-hold",
        action="store_true",
        help=(
            "Temporary diagnostic-only removal of the instantaneous selected-"
            "contact slip-speed safe hold. Slip speed is still measured; the "
            "contact-point displacement and all other safety gates remain active."
        ),
    )
    parser.add_argument(
        "--disable-all-safe-hold",
        action="store_true",
        help=(
            "Temporary diagnostic-only suppression of every Order-8 "
            "SAFE_HOLD transition. Safety violations remain measured and "
            "reported, and the probe runs its complete requested step budget."
        ),
    )
    parser.add_argument(
        "--lock-object-rotation",
        action="store_true",
        help=(
            "Temporary diagnostic-only projection of payload orientation and "
            "angular velocity during LIFT while translation stays free."
        ),
    )
    parser.add_argument(
        "--anchor-hold-joint-correction",
        action="store_true",
        help=(
            "Temporary diagnostic-only full-Dock multi-anchor outer loop. "
            "During LIFT/TRANSPORT/PLACE it integrates DLS corrections into "
            "absolute joint position targets so both measured q_close anchor "
            "poses follow the commanded centroidal manipulation path."
        ),
    )
    parser.add_argument(
        "--loaded-state-rebase",
        action="store_true",
        help=(
            "Temporary diagnostic-only micro-lift/load-capture sequence. At "
            "the first 1 mm geometric lift-off it pauses the centroidal "
            "trajectory, rebases all Dock position targets once to measured "
            "loaded q, waits for relative-motion settle, and resumes LIFT."
        ),
    )
    parser.add_argument(
        "--world-fixed-object",
        action="store_true",
        help=(
            "Acceptance-ineligible collision fault-isolation mode that fixes "
            "only the object while retaining the free robot base."
        ),
    )
    parser.add_argument(
        "--kinematic-base-isolation",
        action="store_true",
        help=(
            "Acceptance-ineligible fault isolation: fix the base module to "
            "world and suppress QPID rotor wrench while retaining Dock "
            "articulation and graph constraints."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument(
        "--full-sequence",
        action="store_true",
        help=(
            "Retain the normal approach/contact-acquisition sequence instead "
            "of jumping directly into the force fixture. The run remains "
            "acceptance-ineligible."
        ),
    )
    parser.add_argument(
        "--continue-after-force-ramp",
        action="store_true",
        help=(
            "Continue an acceptance-ineligible precontact diagnostic through "
            "lift/transport/place/release/settle instead of stopping at the "
            "requested force scale."
        ),
    )
    parser.add_argument(
        "--separate-lift-transition",
        action="store_true",
        help=(
            "Acceptance-ineligible isolation of QPID restore, ordinary "
            "payload LIFT, and the extra acceleration bias."
        ),
    )
    parser.add_argument(
        "--lift-bias-delay-s",
        type=float,
        default=1.25,
        help=(
            "Delay after LIFT entry before the separated diagnostic begins "
            "the extra acceleration-bias ramp."
        ),
    )
    parser.add_argument(
        "--disable-payload-feedforward",
        action="store_true",
        help=(
            "Acceptance-ineligible A/B isolation that keeps the LIFT pose "
            "trajectory but suppresses payload coupling."
        ),
    )
    parser.add_argument(
        "--payload-coupling-component-mode",
        choices=(
            "full",
            "translational_force_only",
            "translational_force_and_com_offset_moment",
        ),
        default="full",
        help=(
            "Acceptance-ineligible A/B isolation of payload translational "
            "force, COM-offset moment, and rotational inertia."
        ),
    )
    parser.add_argument(
        "--precontact-fixture-report",
        default=None,
        help=(
            "Acceptance-ineligible prior diagnostic report whose measured "
            "post-axial base pose initializes dynamic q_close acquisition."
        ),
    )
    parser.add_argument(
        "--near-contact-fixture-report",
        default=None,
        help=(
            "Acceptance-ineligible collision-free prior diagnostic report "
            "whose final base pose, complete Dock state, and object pose "
            "initialize only the last millimetres of q_close acquisition."
        ),
    )
    parser.add_argument(
        "--fixture-height-offset-m",
        type=float,
        default=0.0,
        help=(
            "Acceptance-ineligible upward offset applied to both the restored "
            "near-contact base and free-object poses. Use the configured "
            "support height when replaying an older floor-level fixture."
        ),
    )
    parser.add_argument(
        "--fixture-opening-source-report",
        default=None,
        help=(
            "Prior diagnostic report containing a complete fixed closure "
            "velocity map. With --fixture-opening-duration-s, the near-contact "
            "joint state is moved backward along that map before spawn."
        ),
    )
    parser.add_argument(
        "--fixture-opening-duration-s",
        type=float,
        default=0.0,
        help=(
            "Acceptance-ineligible duration used to move the restored Dock "
            "state opposite the fixed closure direction."
        ),
    )
    parser.add_argument(
        "--qclose-fixture-report",
        default=None,
        help=(
            "Acceptance-ineligible prior report whose measured q_close base "
            "pose and Dock state initialize the short free-base force fixture."
        ),
    )
    parser.add_argument(
        "--qclose-fixture-state-trace",
        default=None,
        help=(
            "Acceptance-ineligible physical state trace whose first lift frame "
            "is converted into an exact free-base q_close checkpoint. This "
            "avoids replaying takeoff and contact acquisition while preserving "
            "the recorded module roots, Dock q/qdot, object state, and measured "
            "anchor hold geometry."
        ),
    )
    parser.add_argument(
        "--zero-qclose-velocities",
        action="store_true",
        help=(
            "Restore q_close poses and joint positions but initialize all "
            "saved velocities to zero. This acceptance-ineligible option "
            "isolates post-grasp force mechanics from checkpoint transients."
        ),
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        help=(
            "Optional cProfile output for the Isaac probe subprocess; used "
            "only to isolate diagnostic runtime bottlenecks."
        ),
    )
    parser.add_argument(
        "--max-simulation-time-s",
        type=float,
        default=None,
        help=(
            "Optional temporary step budget for profiling a prefix of the "
            "normal sequence."
        ),
    )
    parser.add_argument(
        "--state-trace-path",
        default=None,
        help=(
            "Optional acceptance-ineligible state trace captured during this "
            "physical diagnostic for later wall-clock GUI replay."
        ),
    )
    parser.add_argument(
        "--state-trace-frame-stride",
        type=int,
        default=2,
        help="Capture every Nth physics frame when --state-trace-path is set.",
    )
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return parser


def _source_probe_option(
    source_probe_argv: list[str],
    option: str,
) -> str:
    """Return one unambiguous value option from recorded probe argv."""

    values: list[str] = []
    for index, value in enumerate(source_probe_argv):
        if value != option:
            continue
        if index + 1 >= len(source_probe_argv):
            raise ValueError(f"state trace {option} has no value")
        values.append(str(source_probe_argv[index + 1]))
    if not values:
        raise ValueError(f"state trace source argv has no {option}")
    if len(set(values)) != 1:
        raise ValueError(f"state trace source argv has conflicting {option} values")
    return values[-1]


def _resolved_urdf_from_state_trace(trace: dict[str, object]) -> Path:
    source_probe_argv = trace.get("source_probe_argv")
    if not isinstance(source_probe_argv, list) or any(
        not isinstance(value, str) for value in source_probe_argv
    ):
        raise ValueError("state trace has no valid source_probe_argv")
    generated_usd_dir = Path(
        _source_probe_option(source_probe_argv, "--generated-usd-dir")
    ).expanduser()
    if not generated_usd_dir.is_absolute():
        generated_usd_dir = REPO_ROOT / generated_usd_dir
    resolved_dir = generated_usd_dir.resolve() / "resolved_urdf"
    candidates = sorted(resolved_dir.glob("*.urdf"))
    if len(candidates) != 1:
        raise ValueError(
            "state trace generated USD directory must contain exactly one "
            "resolved URDF"
        )
    resolved_urdf = candidates[0].resolve()
    expected_hash = trace.get("source_urdf_sha256")
    if not isinstance(expected_hash, str) or hash_file(resolved_urdf) != expected_hash:
        raise ValueError("state trace resolved URDF hash mismatch")
    return resolved_urdf


def _build_qclose_fixture_from_state_trace(
    trace: dict[str, object],
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    resolved_urdf_path: Path,
) -> dict[str, object]:
    """Convert the first recorded lift state into a diagnostic q_close fixture.

    The saved articulation roots are restored verbatim.  Anchor hold poses are
    reconstructed from each *measured* module root and its recorded local Dock
    state, rather than from the ideal graph constraint, so the short replay does
    not erase compliant constraint offsets present when the grasp gate passed.
    """

    if morphology.stable_hash() != trace.get("graph_hash"):
        raise ValueError("state trace morphology hash mismatch")
    frames = trace.get("frames")
    if not isinstance(frames, list):
        raise ValueError("state trace has no frames")
    frame = next(
        (
            item
            for item in frames
            if isinstance(item, dict) and item.get("phase") == "lift"
        ),
        None,
    )
    if frame is None:
        raise ValueError("state trace has no lift frame after grasp acquisition")

    module_ids = tuple(sorted(module.module_id for module in morphology.modules))
    modules = frame.get("modules")
    joint_names_by_module = trace.get("joint_names_by_module")
    if not isinstance(modules, dict) or not isinstance(
        joint_names_by_module, dict
    ):
        raise ValueError("state trace lift frame has no complete module state")

    urdf_model = load_urdf(resolved_urdf_path)
    root_link_poses = link_poses_in_root_frame(urdf_model)
    module_frame_link_id = module_base_link_name(urdf_model)
    root_to_module_frame = root_link_poses[module_frame_link_id]
    model_context = _build_model_context(
        physical_model,
        WholeStructureKinematicsConfig(),
    )

    module_root_poses: dict[int, Pose7D] = {}
    module_root_velocities: dict[int, tuple[float, ...]] = {}
    module_frame_poses: dict[int, Pose7D] = {}
    local_positions_by_module: dict[int, dict[str, float]] = {}
    local_velocities_by_module: dict[int, dict[str, float]] = {}
    for module_id in module_ids:
        module_key = str(module_id)
        state = modules.get(module_key)
        joint_names = joint_names_by_module.get(module_key)
        if not isinstance(state, dict) or not isinstance(joint_names, list):
            raise ValueError(
                f"state trace lift frame is missing module {module_id}"
            )
        positions = state.get("joint_positions_rad")
        velocities = state.get("joint_velocities_radps")
        root_pose = state.get("root_pose_world")
        root_velocity = state.get("root_twist_world")
        if (
            not isinstance(positions, list)
            or not isinstance(velocities, list)
            or len(positions) != len(joint_names)
            or len(velocities) != len(joint_names)
            or not isinstance(root_pose, list)
            or len(root_pose) != 7
            or not isinstance(root_velocity, list)
            or len(root_velocity) != 6
        ):
            raise ValueError(
                f"state trace lift frame has malformed module {module_id} state"
            )
        local_positions = {
            str(name): float(value)
            for name, value in zip(joint_names, positions, strict=True)
        }
        local_velocities = {
            str(name): float(value)
            for name, value in zip(joint_names, velocities, strict=True)
        }
        missing_dock = set(model_context.dock_joint_ids) - set(local_positions)
        if missing_dock:
            raise ValueError(
                f"state trace module {module_id} lacks Dock joints "
                f"{sorted(missing_dock)}"
            )
        module_root_poses[module_id] = cast(
            Pose7D,
            tuple(float(value) for value in root_pose),
        )
        module_root_velocities[module_id] = tuple(
            float(value) for value in root_velocity
        )
        module_frame_poses[module_id] = compose_pose(
            module_root_poses[module_id],
            root_to_module_frame,
        )
        local_positions_by_module[module_id] = local_positions
        local_velocities_by_module[module_id] = local_velocities

    ordered_joint_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    global_joint_positions: dict[str, float] = {}
    global_joint_velocities: dict[str, float] = {}
    for global_joint_id in ordered_joint_ids:
        module_label, local_joint_id = global_joint_id.split(":", 1)
        module_id = int(module_label.removeprefix("module_"))
        global_joint_positions[global_joint_id] = local_positions_by_module[
            module_id
        ][local_joint_id]
        global_joint_velocities[global_joint_id] = local_velocities_by_module[
            module_id
        ][local_joint_id]

    selected_pair = select_opposing_gripper_surface_pair(
        morphology,
        physical_model,
    )
    selected_surfaces = (selected_pair.first, selected_pair.second)
    grasp_anchor_by_module = {
        anchor.module_id: anchor
        for anchor in morphology.robot_anchors
        if anchor.anchor_type == "grasp"
    }
    anchor_hold_poses_base: dict[int, tuple[float, ...]] = {}
    base_module_pose = module_frame_poses[morphology.base_module_id]
    for surface in selected_surfaces:
        anchor = grasp_anchor_by_module.get(surface.module_id)
        if anchor is None:
            raise ValueError("selected trace gripper has no matching grasp anchor")
        module_link_poses = _module_link_poses(
            model_context,
            {
                joint_id: local_positions_by_module[surface.module_id][joint_id]
                for joint_id in model_context.dock_joint_ids
            },
        )
        if surface.mechanism_link_id not in module_link_poses:
            raise ValueError("selected trace gripper link is absent from module FK")
        mechanism_pose_world = compose_pose(
            module_frame_poses[surface.module_id],
            module_link_poses[surface.mechanism_link_id],
        )
        anchor_pose_world = compose_pose(mechanism_pose_world, anchor.local_pose)
        anchor_hold_poses_base[int(anchor.anchor_id)] = compose_pose(
            inverse_pose(base_module_pose),
            anchor_pose_world,
        )
    if set(anchor_hold_poses_base) != {
        int(anchor.anchor_id)
        for anchor in morphology.robot_anchors
        if anchor.anchor_type == "grasp"
    }:
        raise ValueError("trace q_close fixture must cover both grasp anchors")

    object_pose = frame.get("object_pose_world")
    object_twist = frame.get("object_twist_world")
    if (
        not isinstance(object_pose, list)
        or len(object_pose) != 7
        or not isinstance(object_twist, list)
        or len(object_twist) != 6
    ):
        raise ValueError("state trace lift frame has malformed object state")
    checkpoint_state: dict[str, object] = {
        "schema_version": "order8_qclose_checkpoint_state_v1",
        "module_root_poses": {
            str(module_id): list(pose)
            for module_id, pose in sorted(module_root_poses.items())
        },
        "module_root_velocities": {
            str(module_id): list(velocity)
            for module_id, velocity in sorted(module_root_velocities.items())
        },
        "joint_positions_rad": dict(sorted(global_joint_positions.items())),
        "joint_velocities_radps": dict(sorted(global_joint_velocities.items())),
        "object_pose": [float(value) for value in object_pose],
        "object_twist": [float(value) for value in object_twist],
        "anchor_hold_poses_base": {
            str(anchor_id): list(pose)
            for anchor_id, pose in sorted(anchor_hold_poses_base.items())
        },
    }
    return {
        "order8_natural_contact_qclose_checkpoint_base_pose": list(
            base_module_pose
        ),
        "order8_natural_contact_qclose_checkpoint_joint_positions_rad": dict(
            sorted(global_joint_positions.items())
        ),
        "order8_natural_contact_qclose_checkpoint_state": checkpoint_state,
        "order8_natural_contact_qclose_trace_fixture_provenance": {
            "diagnostic_only": True,
            "acceptance_eligible": False,
            "source_trace_payload_hash": trace.get("trace_payload_hash"),
            "source_trace_graph_hash": trace.get("graph_hash"),
            "source_trace_urdf_sha256": trace.get("source_urdf_sha256"),
            "source_frame_phase": "lift",
            "source_frame_simulation_time_s": float(frame["simulation_time_s"]),
            "anchor_hold_source": (
                "recorded_module_roots_plus_recorded_dock_q_local_fk_v1"
            ),
        },
    }


def _load_qclose_fixture_state_trace(path: Path) -> dict[str, object]:
    trace = load_order8_state_trace(path)
    source_probe_argv = trace["source_probe_argv"]
    assert isinstance(source_probe_argv, list)
    resolved_urdf = _resolved_urdf_from_state_trace(trace)
    backend_config_path = Path(
        _source_probe_option(source_probe_argv, "--config")
    ).expanduser()
    if not backend_config_path.is_absolute():
        backend_config_path = REPO_ROOT / backend_config_path
    backend_config = load_isaac_lab_backend_config(backend_config_path.resolve())
    physical_model = build_physical_model_from_config(
        Path(backend_config.robot_model_config_path).resolve(),
        urdf_path_override=resolved_urdf,
    )
    morphology_json = _source_probe_option(
        source_probe_argv,
        "--order8-morphology-graph-json",
    )
    morphology = MorphologyGraph.from_json(morphology_json)
    fixture = _build_qclose_fixture_from_state_trace(
        trace,
        morphology=morphology,
        physical_model=physical_model,
        resolved_urdf_path=resolved_urdf,
    )
    provenance = fixture[
        "order8_natural_contact_qclose_trace_fixture_provenance"
    ]
    assert isinstance(provenance, dict)
    provenance["source_trace_path"] = str(path.resolve())
    provenance["resolved_urdf_path"] = str(resolved_urdf)
    return fixture


def _load_fixed_closure_velocity_targets(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload.get("report", payload)
    if not isinstance(report, dict):
        raise ValueError("fixture opening source must contain a report map")
    raw = report.get(
        "order8_natural_contact_contact_closure_velocity_targets_radps"
    )
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "fixture opening source has no fixed closure velocity targets"
        )
    result: dict[str, float] = {}
    for joint_id, value in raw.items():
        if (
            not isinstance(joint_id, str)
            or not joint_id
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                "fixture opening velocity targets must map joint ids to finite values"
            )
        result[joint_id] = float(value)
    return result


def _transform_near_contact_fixture(
    state: dict[str, object],
    *,
    height_offset_m: float,
    opening_velocity_targets_radps: dict[str, float] | None,
    opening_duration_s: float,
) -> dict[str, object]:
    if not math.isfinite(float(height_offset_m)) or float(height_offset_m) < 0.0:
        raise ValueError("fixture height offset must be finite and non-negative")
    if not math.isfinite(float(opening_duration_s)) or float(opening_duration_s) < 0.0:
        raise ValueError("fixture opening duration must be finite and non-negative")
    if (opening_velocity_targets_radps is None) != (
        float(opening_duration_s) <= 0.0
    ):
        raise ValueError(
            "positive fixture opening duration requires exactly one velocity source"
        )

    transformed = dict(state)
    base_pose = [float(value) for value in state["base_pose"]]  # type: ignore[index]
    object_pose = [float(value) for value in state["object_pose"]]  # type: ignore[index]
    base_pose[2] += float(height_offset_m)
    object_pose[2] += float(height_offset_m)
    transformed["base_pose"] = base_pose
    transformed["object_pose"] = object_pose

    positions = {
        str(joint_id): float(value)
        for joint_id, value in state["joint_positions_rad"].items()  # type: ignore[union-attr]
    }
    if opening_velocity_targets_radps is not None:
        if set(opening_velocity_targets_radps) != set(positions):
            raise ValueError(
                "fixture opening velocity map must exactly cover the restored Dock state"
            )
        positions = {
            joint_id: positions[joint_id]
            - float(opening_duration_s)
            * float(opening_velocity_targets_radps[joint_id])
            for joint_id in sorted(positions)
        }
    transformed["joint_positions_rad"] = positions
    transformed["height_offset_m"] = float(height_offset_m)
    transformed["opening_duration_s"] = float(opening_duration_s)
    transformed["opening_velocity_targets_radps"] = (
        None
        if opening_velocity_targets_radps is None
        else dict(sorted(opening_velocity_targets_radps.items()))
    )
    return transformed


def _fast_config(
    config: object,
    *,
    speed_scale: float,
    force_ramp_s: float,
    object_width_padding_m: float,
    full_sequence: bool,
    object_friction: float | None,
    contact_dwell_s: float | None,
    object_mass_kg: float | None = None,
    selected_gripper_friction: float | None = None,
    selected_gripper_compliant_contact_stiffness: float | None = None,
    selected_gripper_compliant_contact_damping: float | None = None,
    normal_force_target_n: float | None = None,
    contact_stall_speed_threshold_mps: float | None = None,
    max_slip_speed_mps: float | None = None,
    max_contact_point_slip_displacement_m: float | None = None,
    payload_load_transfer_s: float | None = None,
):
    if speed_scale <= 1.0:
        raise ValueError("diagnostic speed scale must be greater than one")
    if object_width_padding_m < 0.0:
        raise ValueError("diagnostic object width padding must be non-negative")
    return replace(
        config,
        base_translation_speed_limit_mps=(
            float(config.base_translation_speed_limit_mps) * speed_scale
        ),
        contact_base_translation_speed_limit_mps=(
            float(config.contact_base_translation_speed_limit_mps)
            if full_sequence
            else float(config.contact_base_translation_speed_limit_mps) * speed_scale
        ),
        hover_dwell_s=(0.50 if full_sequence else 0.10),
        anchor_translation_speed_limit_mps=(
            float(config.anchor_translation_speed_limit_mps)
            if full_sequence
            else float(config.anchor_translation_speed_limit_mps) * speed_scale
        ),
        contact_surface_creep_speed_limit_mps=(
            float(config.contact_surface_creep_speed_limit_mps)
            if full_sequence
            else float(config.contact_surface_creep_speed_limit_mps) * speed_scale
        ),
        contact_dwell_s=(
            float(contact_dwell_s)
            if contact_dwell_s is not None
            else (float(config.contact_dwell_s) if full_sequence else 0.05)
        ),
        contact_force_ramp_s=float(force_ramp_s),
        contact_stall_dwell_s=(
            float(config.contact_stall_dwell_s) if full_sequence else 0.10
        ),
        contact_acquisition_timeout_s=(
            float(config.contact_acquisition_timeout_s)
            if full_sequence
            else max(35.0, float(force_ramp_s) + 20.0)
        ),
        object_mass_kg=(
            float(config.object_mass_kg)
            if object_mass_kg is None
            else float(object_mass_kg)
        ),
        object_friction=(
            float(config.object_friction)
            if object_friction is None
            else float(object_friction)
        ),
        selected_gripper_friction=(
            float(config.selected_gripper_friction)
            if selected_gripper_friction is None
            else float(selected_gripper_friction)
        ),
        selected_gripper_compliant_contact_stiffness_n_per_m=(
            float(config.selected_gripper_compliant_contact_stiffness_n_per_m)
            if selected_gripper_compliant_contact_stiffness is None
            else float(selected_gripper_compliant_contact_stiffness)
        ),
        selected_gripper_compliant_contact_damping_n_s_per_m=(
            float(config.selected_gripper_compliant_contact_damping_n_s_per_m)
            if selected_gripper_compliant_contact_damping is None
            else float(selected_gripper_compliant_contact_damping)
        ),
        normal_force_target_per_contact_n=(
            float(config.normal_force_target_per_contact_n)
            if normal_force_target_n is None
            else float(normal_force_target_n)
        ),
        contact_stall_anchor_speed_threshold_mps=(
            float(config.contact_stall_anchor_speed_threshold_mps)
            if contact_stall_speed_threshold_mps is None
            else float(contact_stall_speed_threshold_mps)
        ),
        max_tangential_slip_speed_mps=(
            float(config.max_tangential_slip_speed_mps)
            if max_slip_speed_mps is None
            else float(max_slip_speed_mps)
        ),
        max_contact_point_slip_displacement_m=(
            float(config.max_contact_point_slip_displacement_m)
            if max_contact_point_slip_displacement_m is None
            else float(max_contact_point_slip_displacement_m)
        ),
        payload_load_transfer_s=(
            float(config.payload_load_transfer_s)
            if payload_load_transfer_s is None
            else float(payload_load_transfer_s)
        ),
    )


def _supports_phase_continuation(
    *,
    precontact_base_pose: object,
    near_contact_base_pose: object = None,
    qclose_base_pose: object,
    qclose_checkpoint_state: object,
) -> bool:
    """Accept only free-object fixtures with enough state for continuation."""

    return bool(
        precontact_base_pose is not None
        or near_contact_base_pose is not None
        or (qclose_base_pose is not None and qclose_checkpoint_state is not None)
    )


def _load_near_contact_fixture_report(path: Path) -> dict[str, object]:
    """Load a collision-free, whole-structure state for short fault isolation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload.get("report", payload)
    if not isinstance(report, dict):
        raise ValueError("near-contact fixture report must contain a report map")

    def finite_pose(key: str) -> list[float]:
        value = report.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 7
            or not all(
                isinstance(component, (int, float))
                and not isinstance(component, bool)
                and math.isfinite(float(component))
                for component in value
            )
        ):
            raise ValueError(
                f"near-contact fixture report has no finite {key!r} pose"
            )
        quaternion_norm = math.sqrt(
            sum(float(component) ** 2 for component in value[3:7])
        )
        if abs(quaternion_norm - 1.0) > 1.0e-3:
            raise ValueError(
                f"near-contact fixture report {key!r} quaternion is not unit length"
            )
        return [float(component) for component in value]

    recorded_initial_fixture = bool(
        report.get("order8_natural_contact_diagnostic_near_contact_base_pose")
        is not None
        and report.get(
            "order8_natural_contact_diagnostic_near_contact_joint_positions_rad"
        )
        is not None
        and report.get("order8_natural_contact_diagnostic_near_contact_object_pose")
        is not None
        and report.get(
            "order8_natural_contact_diagnostic_near_contact_initial_surface_clearance_m_by_anchor"
        )
        is not None
    )
    joint_positions_key = (
        "order8_natural_contact_diagnostic_near_contact_joint_positions_rad"
        if recorded_initial_fixture
        else "order8_natural_contact_last_joint_positions_rad"
    )
    joint_positions = report.get(joint_positions_key)
    if not isinstance(joint_positions, dict) or not joint_positions:
        raise ValueError(
            "near-contact fixture report has no complete Dock-joint state"
        )
    normalized_joint_positions: dict[str, float] = {}
    for joint_id, value in joint_positions.items():
        if (
            not isinstance(joint_id, str)
            or not joint_id
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                "near-contact fixture Dock state must map non-empty ids to "
                "finite numbers"
            )
        normalized_joint_positions[joint_id] = float(value)

    clearances_key = (
        "order8_natural_contact_diagnostic_near_contact_initial_surface_clearance_m_by_anchor"
        if recorded_initial_fixture
        else "order8_natural_contact_last_contact_mesh_surface_clearance_m_by_anchor"
    )
    clearances = report.get(clearances_key)
    if (
        not isinstance(clearances, dict)
        or len(clearances) < 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in clearances.values()
        )
    ):
        raise ValueError(
            "near-contact fixture must be a measured collision-free state with "
            "positive selected-surface clearances"
        )
    if (
        not recorded_initial_fixture
        and report.get("order8_natural_contact_qclose_checkpoint_base_pose") is not None
    ):
        raise ValueError(
            "near-contact fixture must precede q_close; use q_close fixture "
            "diagnostics for an already arrested state"
        )

    return {
        "base_pose": finite_pose(
            "order8_natural_contact_diagnostic_near_contact_base_pose"
            if recorded_initial_fixture
            else "order8_natural_contact_last_measured_base_module_pose"
        ),
        "joint_positions_rad": normalized_joint_positions,
        "object_pose": finite_pose(
            "order8_natural_contact_diagnostic_near_contact_object_pose"
            if recorded_initial_fixture
            else "order8_natural_contact_last_measured_object_pose"
        ),
        "source_surface_clearance_m_by_anchor": {
            str(anchor_id): float(value)
            for anchor_id, value in clearances.items()
        },
        "source_state_method": (
            "recorded_diagnostic_initial_near_contact_fixture_v1"
            if recorded_initial_fixture
            else "terminal_collision_free_near_contact_state_v1"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not math.isfinite(float(args.lift_bias_delay_s))
        or float(args.lift_bias_delay_s) < 0.0
    ):
        raise ValueError(
            "diagnostic LIFT-bias delay must be finite and non-negative"
        )
    if args.separate_lift_transition and not args.continue_after_force_ramp:
        raise ValueError(
            "--separate-lift-transition requires --continue-after-force-ramp"
        )
    if args.disable_payload_feedforward and not args.separate_lift_transition:
        raise ValueError(
            "--disable-payload-feedforward requires --separate-lift-transition"
        )
    if (
        args.payload_coupling_component_mode != "full"
        and not args.separate_lift_transition
    ):
        raise ValueError(
            "--payload-coupling-component-mode requires "
            "--separate-lift-transition"
        )
    if (
        args.disable_payload_feedforward
        and args.payload_coupling_component_mode != "full"
    ):
        raise ValueError(
            "--disable-payload-feedforward cannot be combined with payload "
            "component isolation"
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.0 < float(args.stop_force_scale) <= 1.0:
        raise ValueError("stop force scale must be in (0, 1]")
    if args.force_anchor_id is not None and (
        not args.force_anchor_id
        or len(set(args.force_anchor_id)) != len(args.force_anchor_id)
        or any(value < 0 for value in args.force_anchor_id)
    ):
        raise ValueError("force anchor ids must be unique and non-negative")
    if args.force_ramp_s <= 0.0 or args.timeout_s <= 0.0 or args.dt <= 0.0:
        raise ValueError("force ramp, timeout, and diagnostic dt must be positive")
    if args.contact_dwell_s is not None and (
        not math.isfinite(float(args.contact_dwell_s))
        or float(args.contact_dwell_s) <= 0.0
    ):
        raise ValueError("diagnostic contact dwell must be finite and positive")
    if args.dock_stiffness is not None and (
        not math.isfinite(float(args.dock_stiffness))
        or float(args.dock_stiffness) <= 0.0
    ):
        raise ValueError("diagnostic Dock stiffness must be finite and positive")
    if args.dock_damping is not None and (
        not math.isfinite(float(args.dock_damping)) or float(args.dock_damping) <= 0.0
    ):
        raise ValueError("diagnostic Dock damping must be finite and positive")
    if args.dock_velocity_limit is not None and (
        not math.isfinite(float(args.dock_velocity_limit))
        or float(args.dock_velocity_limit) <= 0.0
    ):
        raise ValueError(
            "diagnostic Dock velocity limit must be finite and positive"
        )
    if args.dock_armature_kg_m2 is not None and (
        not math.isfinite(float(args.dock_armature_kg_m2))
        or float(args.dock_armature_kg_m2) <= 0.0
    ):
        raise ValueError(
            "diagnostic Dock armature must be finite and positive"
        )
    if args.object_mass_kg is not None and (
        not math.isfinite(float(args.object_mass_kg))
        or float(args.object_mass_kg) <= 0.0
    ):
        raise ValueError("diagnostic object mass must be finite and positive")
    if args.object_friction is not None and (
        not math.isfinite(float(args.object_friction))
        or float(args.object_friction) <= 0.0
    ):
        raise ValueError("diagnostic object friction must be finite and positive")
    if args.selected_gripper_friction is not None and (
        not math.isfinite(float(args.selected_gripper_friction))
        or float(args.selected_gripper_friction) <= 0.0
    ):
        raise ValueError(
            "diagnostic selected-gripper friction must be finite and positive"
        )
    if args.selected_gripper_compliant_contact_stiffness is not None and (
        not math.isfinite(
            float(args.selected_gripper_compliant_contact_stiffness)
        )
        or float(args.selected_gripper_compliant_contact_stiffness) <= 0.0
    ):
        raise ValueError(
            "diagnostic compliant-contact stiffness must be finite and positive"
        )
    if args.selected_gripper_compliant_contact_damping is not None and (
        not math.isfinite(float(args.selected_gripper_compliant_contact_damping))
        or float(args.selected_gripper_compliant_contact_damping) <= 0.0
    ):
        raise ValueError(
            "diagnostic compliant-contact damping must be finite and positive"
        )
    if args.normal_force_target_n is not None and (
        not math.isfinite(float(args.normal_force_target_n))
        or float(args.normal_force_target_n) <= 0.0
    ):
        raise ValueError(
            "diagnostic normal-force target must be finite and positive"
        )
    if args.payload_load_transfer_s is not None and (
        not math.isfinite(float(args.payload_load_transfer_s))
        or float(args.payload_load_transfer_s) <= 0.0
    ):
        raise ValueError(
            "diagnostic payload load-transfer duration must be finite and positive"
        )
    if args.contact_stall_speed_threshold_mps is not None and (
        not math.isfinite(float(args.contact_stall_speed_threshold_mps))
        or float(args.contact_stall_speed_threshold_mps) <= 0.0
    ):
        raise ValueError(
            "diagnostic contact-stall speed threshold must be finite and positive"
        )
    if args.contact_closure_joint_speed_radps is not None and (
        not math.isfinite(float(args.contact_closure_joint_speed_radps))
        or float(args.contact_closure_joint_speed_radps) <= 0.0
    ):
        raise ValueError(
            "diagnostic contact-closure joint speed must be finite and positive"
        )
    if args.max_slip_speed_mps is not None and (
        not math.isfinite(float(args.max_slip_speed_mps))
        or float(args.max_slip_speed_mps) <= 0.0
    ):
        raise ValueError("diagnostic slip-speed limit must be finite and positive")
    if args.max_contact_point_slip_displacement_m is not None and (
        not math.isfinite(float(args.max_contact_point_slip_displacement_m))
        or float(args.max_contact_point_slip_displacement_m) <= 0.0
    ):
        raise ValueError(
            "diagnostic contact-point slip-displacement limit must be finite "
            "and positive"
        )
    if args.peak_torque_window_s is not None and (
        not math.isfinite(float(args.peak_torque_window_s))
        or float(args.peak_torque_window_s) <= 0.0
    ):
        raise ValueError(
            "diagnostic peak-torque window must be finite and positive"
        )
    if args.post_grasp_joint_torque_bias_nm is not None and (
        not math.isfinite(float(args.post_grasp_joint_torque_bias_nm))
        or float(args.post_grasp_joint_torque_bias_nm) <= 0.0
    ):
        raise ValueError(
            "diagnostic post-grasp joint torque bias must be finite and positive"
        )
    if args.max_simulation_time_s is not None and args.max_simulation_time_s <= 0.0:
        raise ValueError("diagnostic max simulation time must be positive")
    if args.proxy_pad and args.cone_proxy_pad:
        raise ValueError("--proxy-pad and --cone-proxy-pad are mutually exclusive")
    if (
        not math.isfinite(float(args.fixture_height_offset_m))
        or float(args.fixture_height_offset_m) < 0.0
    ):
        raise ValueError("fixture height offset must be finite and non-negative")
    if (
        not math.isfinite(float(args.fixture_opening_duration_s))
        or float(args.fixture_opening_duration_s) < 0.0
    ):
        raise ValueError("fixture opening duration must be finite and non-negative")
    if (args.fixture_opening_source_report is None) != (
        float(args.fixture_opening_duration_s) <= 0.0
    ):
        raise ValueError(
            "--fixture-opening-source-report and a positive "
            "--fixture-opening-duration-s must be supplied together"
        )
    if (
        not isinstance(args.state_trace_frame_stride, int)
        or isinstance(args.state_trace_frame_stride, bool)
        or args.state_trace_frame_stride <= 0
    ):
        raise ValueError("state trace frame stride must be a positive integer")
    fixture_mode_count = sum(
        (
            bool(args.full_sequence),
            args.precontact_fixture_report is not None,
            args.near_contact_fixture_report is not None,
            args.qclose_fixture_report is not None,
            args.qclose_fixture_state_trace is not None,
        )
    )
    if fixture_mode_count > 1:
        raise ValueError(
            "--full-sequence, --precontact-fixture-report, "
            "--near-contact-fixture-report, --qclose-fixture-report, and "
            "--qclose-fixture-state-trace are mutually exclusive"
        )
    if (
        float(args.fixture_height_offset_m) > 0.0
        or float(args.fixture_opening_duration_s) > 0.0
    ) and args.near_contact_fixture_report is None:
        raise ValueError(
            "fixture height/opening transforms require --near-contact-fixture-report"
        )
    precontact_source_path = (
        None
        if args.precontact_fixture_report is None
        else Path(args.precontact_fixture_report).resolve()
    )
    precontact_base_pose = None
    if precontact_source_path is not None:
        source_payload = json.loads(precontact_source_path.read_text(encoding="utf-8"))
        source_report = source_payload.get("report", source_payload)
        precontact_base_pose = source_report.get(
            "order8_natural_contact_contact_axial_hold_base_pose"
        )
        if (
            not isinstance(precontact_base_pose, list)
            or len(precontact_base_pose) != 7
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in precontact_base_pose
            )
        ):
            raise ValueError(
                "precontact fixture report has no finite measured axial-hold pose"
            )
    near_contact_source_path = (
        None
        if args.near_contact_fixture_report is None
        else Path(args.near_contact_fixture_report).resolve()
    )
    near_contact_state = (
        None
        if near_contact_source_path is None
        else _load_near_contact_fixture_report(near_contact_source_path)
    )
    fixture_opening_source_path = (
        None
        if args.fixture_opening_source_report is None
        else Path(args.fixture_opening_source_report).resolve()
    )
    fixture_opening_velocity_targets = (
        None
        if fixture_opening_source_path is None
        else _load_fixed_closure_velocity_targets(fixture_opening_source_path)
    )
    if near_contact_state is not None:
        near_contact_state = _transform_near_contact_fixture(
            near_contact_state,
            height_offset_m=float(args.fixture_height_offset_m),
            opening_velocity_targets_radps=fixture_opening_velocity_targets,
            opening_duration_s=float(args.fixture_opening_duration_s),
        )
    near_contact_base_pose = (
        None
        if near_contact_state is None
        else near_contact_state["base_pose"]
    )
    qclose_source_path = (
        None
        if args.qclose_fixture_report is None
        else Path(args.qclose_fixture_report).resolve()
    )
    qclose_trace_source_path = (
        None
        if args.qclose_fixture_state_trace is None
        else Path(args.qclose_fixture_state_trace).resolve()
    )
    qclose_base_pose = None
    qclose_joint_positions = None
    qclose_checkpoint_state = None
    qclose_report = None
    if qclose_trace_source_path is not None:
        qclose_report = _load_qclose_fixture_state_trace(qclose_trace_source_path)
    elif qclose_source_path is not None:
        qclose_payload = json.loads(qclose_source_path.read_text(encoding="utf-8"))
        qclose_report = qclose_payload.get("report", qclose_payload)
    if qclose_report is not None:
        if not isinstance(qclose_report, dict):
            raise ValueError("q_close fixture source must contain a report map")
        qclose_base_pose = qclose_report.get(
            "order8_natural_contact_qclose_checkpoint_base_pose"
        )
        if qclose_base_pose is None:
            axial_pose = qclose_report.get(
                "order8_natural_contact_contact_axial_hold_base_pose"
            )
            centering_offset = qclose_report.get(
                "order8_natural_contact_contact_centering_latched_offset_world"
            )
            if (
                isinstance(axial_pose, list)
                and len(axial_pose) == 7
                and isinstance(centering_offset, list)
                and len(centering_offset) == 3
            ):
                qclose_base_pose = [
                    float(axial_pose[index]) + float(centering_offset[index])
                    for index in range(3)
                ] + [float(value) for value in axial_pose[3:7]]
        qclose_joint_positions = qclose_report.get(
            "order8_natural_contact_qclose_checkpoint_joint_positions_rad",
            qclose_report.get("order8_natural_contact_last_joint_positions_rad"),
        )
        if (
            not isinstance(qclose_base_pose, list)
            or len(qclose_base_pose) != 7
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in qclose_base_pose
            )
        ):
            raise ValueError(
                "q_close fixture report has no finite measured q_close base pose"
            )
        if not isinstance(qclose_joint_positions, dict) or not all(
            isinstance(joint_id, str)
            and joint_id
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for joint_id, value in qclose_joint_positions.items()
        ):
            raise ValueError(
                "q_close fixture report has no finite global Dock-joint state"
            )
        qclose_joint_positions = {
            str(joint_id): float(value)
            for joint_id, value in qclose_joint_positions.items()
        }
        qclose_checkpoint_state = qclose_report.get(
            "order8_natural_contact_qclose_checkpoint_state"
        )
        if qclose_checkpoint_state is not None and (
            not isinstance(qclose_checkpoint_state, dict)
            or qclose_checkpoint_state.get("schema_version")
            != "order8_qclose_checkpoint_state_v1"
        ):
            raise ValueError(
                "q_close fixture report has an invalid exact checkpoint state"
            )
    if args.zero_qclose_velocities and qclose_checkpoint_state is None:
        raise ValueError(
            "--zero-qclose-velocities requires an exact "
            "q_close report or state-trace checkpoint"
        )
    if args.anchor_hold_joint_correction and args.lock_object_rotation:
        raise ValueError(
            "--anchor-hold-joint-correction cannot be combined with "
            "--lock-object-rotation in one causal diagnostic"
        )
    if args.loaded_state_rebase and not args.continue_after_force_ramp:
        raise ValueError(
            "--loaded-state-rebase requires --continue-after-force-ramp"
        )
    if args.loaded_state_rebase and not args.separate_lift_transition:
        raise ValueError(
            "--loaded-state-rebase requires --separate-lift-transition"
        )
    if args.loaded_state_rebase and args.anchor_hold_joint_correction:
        raise ValueError(
            "--loaded-state-rebase cannot be combined with "
            "--anchor-hold-joint-correction"
        )
    if args.loaded_state_rebase and args.lock_object_rotation:
        raise ValueError(
            "--loaded-state-rebase cannot be combined with "
            "--lock-object-rotation"
        )
    contact_sequence = bool(
        args.full_sequence
        or precontact_base_pose is not None
        or near_contact_base_pose is not None
        or qclose_base_pose is not None
    )
    if contact_sequence and float(args.object_width_padding_m) != 0.0:
        raise ValueError("full/precontact sequence requires --object-width-padding-m 0")
    if args.continue_after_force_ramp and (
        not _supports_phase_continuation(
            precontact_base_pose=precontact_base_pose,
            near_contact_base_pose=near_contact_base_pose,
            qclose_base_pose=qclose_base_pose,
            qclose_checkpoint_state=qclose_checkpoint_state,
        )
        or args.world_fixed_object
        or args.force_anchor_id is not None
    ):
        raise ValueError(
            "continuation requires a free-object precontact fixture or an exact "
            "q_close checkpoint and cannot use world-fixed-object or force-anchor "
            "isolation"
        )

    config_path = Path(args.config).resolve()
    backend_path = Path(args.backend_config).resolve()
    source_config = load_order8_natural_contact_config(config_path)
    diagnostic_config = _fast_config(
        source_config,
        speed_scale=float(args.speed_scale),
        force_ramp_s=float(args.force_ramp_s),
        object_width_padding_m=float(args.object_width_padding_m),
        # Measured precontact/q_close fixtures are explicitly temporary fault
        # isolation and may use accelerated closure/dwell settings.  Only the
        # from-reset --full-sequence path preserves nominal contact timing.
        full_sequence=bool(args.full_sequence),
        object_mass_kg=args.object_mass_kg,
        object_friction=args.object_friction,
        contact_dwell_s=args.contact_dwell_s,
        selected_gripper_friction=args.selected_gripper_friction,
        selected_gripper_compliant_contact_stiffness=(
            args.selected_gripper_compliant_contact_stiffness
        ),
        selected_gripper_compliant_contact_damping=(
            args.selected_gripper_compliant_contact_damping
        ),
        normal_force_target_n=args.normal_force_target_n,
        contact_stall_speed_threshold_mps=(
            args.contact_stall_speed_threshold_mps
        ),
        max_slip_speed_mps=args.max_slip_speed_mps,
        max_contact_point_slip_displacement_m=(
            args.max_contact_point_slip_displacement_m
        ),
        payload_load_transfer_s=args.payload_load_transfer_s,
    )
    diagnostic_config.validate()
    backend_config = load_isaac_lab_backend_config(backend_path)
    physical_model = build_physical_model_from_config(
        Path(backend_config.robot_model_config_path).resolve()
    )
    morphology = build_representative_order8_morphology(physical_model)
    env = Order8IsaacNaturalContactEnv(
        config=diagnostic_config,
        backend=IsaacLabBackend(backend_config),
        physical_model=physical_model,
        backend_config_path=backend_path,
        rollout_budget_s=30.0,
        command_timeout_s=float(args.timeout_s),
        generated_usd_dir=ORDER8_DEFAULT_GENERATED_USD_DIR,
        seed=int(args.seed),
    )
    command = env.build_probe_command(morphology)
    dt_index = command.index("--dt")
    command[dt_index + 1] = str(float(args.dt))
    if args.max_simulation_time_s is not None:
        steps_index = command.index("--steps")
        command[steps_index + 1] = str(
            max(
                1,
                int(math.ceil(float(args.max_simulation_time_s) / float(args.dt))),
            )
        )
    if "--force-convert" in command:
        command.remove("--force-convert")
    command.append("--force-convert" if args.force_convert else "--convert-if-missing")
    command.extend(["--order8-diagnostic-only"])
    if args.separate_lift_transition:
        command.extend(
            [
                "--order8-diagnostic-separated-lift-transition",
                "--order8-diagnostic-lift-bias-delay-s",
                str(float(args.lift_bias_delay_s)),
            ]
        )
    if args.disable_payload_feedforward:
        command.append("--order8-diagnostic-disable-payload-feedforward")
    if args.payload_coupling_component_mode != "full":
        command.extend(
            [
                "--order8-diagnostic-payload-coupling-component-mode",
                str(args.payload_coupling_component_mode),
            ]
        )
    if args.proxy_pad:
        command.append("--order8-diagnostic-proxy-pad")
    if args.cone_proxy_pad:
        command.append("--order8-diagnostic-cone-proxy-pad")
    if args.contact_closure_joint_speed_radps is not None:
        command.extend(
            [
                "--order8-diagnostic-contact-closure-joint-speed-radps",
                str(float(args.contact_closure_joint_speed_radps)),
            ]
        )
    if args.continue_after_force_ramp:
        command.append("--order8-diagnostic-continue-after-force-ramp")
    if args.world_fixed_object:
        command.append("--order8-diagnostic-world-fixed-object")
    if args.kinematic_base_isolation:
        command.append("--order8-diagnostic-kinematic-base-isolation")
    if args.dock_stiffness is not None:
        command.extend(["--dock-stiffness", str(float(args.dock_stiffness))])
    if args.dock_damping is not None:
        command.extend(["--dock-damping", str(float(args.dock_damping))])
    if args.dock_velocity_limit is not None:
        command.extend(
            [
                "--order8-diagnostic-dock-velocity-limit-rad-s",
                str(float(args.dock_velocity_limit)),
            ]
        )
    if args.dock_armature_kg_m2 is not None:
        command.extend(
            [
                "--order8-diagnostic-dock-armature-kg-m2",
                str(float(args.dock_armature_kg_m2)),
            ]
        )
    if args.peak_torque_window_s is not None:
        command.extend(
            [
                "--order8-diagnostic-peak-torque-window-s",
                str(float(args.peak_torque_window_s)),
            ]
        )
    if args.post_grasp_joint_torque_bias_nm is not None:
        command.extend(
            [
                "--order8-diagnostic-post-grasp-joint-torque-bias-nm",
                str(float(args.post_grasp_joint_torque_bias_nm)),
            ]
        )
    if args.disable_slip_speed_safe_hold:
        command.append("--order8-diagnostic-disable-slip-speed-safe-hold")
    if args.disable_all_safe_hold:
        command.append("--order8-diagnostic-disable-all-safe-hold")
    if args.lock_object_rotation:
        command.append("--order8-diagnostic-lock-object-rotation")
    if args.anchor_hold_joint_correction:
        command.append("--order8-diagnostic-anchor-hold-joint-correction")
    if args.loaded_state_rebase:
        command.append("--order8-diagnostic-loaded-state-rebase")
    if qclose_base_pose is not None and qclose_joint_positions is not None:
        command.extend(
            [
                "--order8-diagnostic-qclose-base-pose",
                *(str(float(value)) for value in qclose_base_pose),
                "--order8-diagnostic-qclose-joint-positions-json",
                json.dumps(qclose_joint_positions, sort_keys=True),
            ]
        )
        if qclose_checkpoint_state is not None:
            command.extend(
                [
                    "--order8-diagnostic-qclose-state-json",
                    json.dumps(qclose_checkpoint_state, sort_keys=True),
                ]
            )
            if args.zero_qclose_velocities:
                command.append("--order8-diagnostic-qclose-zero-velocities")
    elif near_contact_state is not None:
        command.extend(
            [
                "--order8-diagnostic-near-contact-base-pose",
                *(
                    str(float(value))
                    for value in near_contact_state["base_pose"]
                ),
                "--order8-diagnostic-near-contact-joint-positions-json",
                json.dumps(
                    near_contact_state["joint_positions_rad"],
                    sort_keys=True,
                ),
                "--order8-diagnostic-near-contact-object-pose",
                *(
                    str(float(value))
                    for value in near_contact_state["object_pose"]
                ),
            ]
        )
    elif precontact_base_pose is not None:
        command.extend(
            [
                "--order8-diagnostic-precontact-base-pose",
                *(str(float(value)) for value in precontact_base_pose),
            ]
        )
    elif not args.full_sequence:
        command.extend(
            [
                "--order8-diagnostic-force-fixture",
                "--order8-diagnostic-object-width-padding-m",
                str(float(args.object_width_padding_m)),
            ]
        )
    command.extend(
        [
            "--order8-diagnostic-stop-force-scale",
            str(float(args.stop_force_scale)),
        ]
    )
    for anchor_id in args.force_anchor_id or ():
        command.extend(["--order8-diagnostic-force-anchor-id", str(int(anchor_id))])
    if args.profile_output is not None:
        command.extend(
            [
                "--order8-diagnostic-profile-output",
                str(Path(args.profile_output).resolve()),
            ]
        )
    if args.state_trace_path is not None:
        command.extend(
            [
                "--order8-state-trace-output",
                str(Path(args.state_trace_path).expanduser().resolve()),
                "--order8-state-trace-frame-stride",
                str(int(args.state_trace_frame_stride)),
            ]
        )

    wall_started = time.monotonic()
    raw_report = _run_json_command(command, float(args.timeout_s))
    wall_elapsed_s = time.monotonic() - wall_started
    diagnostic_time_budget_reached = bool(
        args.max_simulation_time_s is not None
        and float(raw_report.get("order8_natural_contact_simulation_time_s", 0.0))
        >= float(args.max_simulation_time_s) - 0.5 * float(args.dt)
    )
    diagnostic = {
        "diagnostic_version": "order8_contact_force_fast_diagnostic_v28",
        "diagnostic_only": True,
        "diagnostic_force_fixture": not contact_sequence,
        "diagnostic_precontact_fixture": precontact_base_pose is not None,
        "diagnostic_near_contact_fixture": near_contact_state is not None,
        "diagnostic_qclose_fixture": qclose_base_pose is not None,
        "diagnostic_full_sequence": bool(args.full_sequence),
        "continue_after_force_ramp": bool(args.continue_after_force_ramp),
        "separated_lift_transition": bool(args.separate_lift_transition),
        "lift_bias_delay_s": float(args.lift_bias_delay_s),
        "payload_feedforward_disabled": bool(
            args.disable_payload_feedforward
        ),
        "payload_coupling_component_mode": str(
            args.payload_coupling_component_mode
        ),
        "acceptance_eligible": False,
        "source_config_path": str(config_path),
        "source_config_hash": source_config.stable_hash(),
        "diagnostic_config": diagnostic_config.to_dict(),
        "diagnostic_config_hash": diagnostic_config.stable_hash(),
        "speed_scale": float(args.speed_scale),
        "force_ramp_s": float(args.force_ramp_s),
        "contact_dwell_override_s": (
            None
            if args.contact_dwell_s is None
            else float(args.contact_dwell_s)
        ),
        "stop_force_scale": float(args.stop_force_scale),
        "force_anchor_ids": (
            None
            if args.force_anchor_id is None
            else sorted(int(value) for value in args.force_anchor_id)
        ),
        "diagnostic_dt_s": float(args.dt),
        "object_width_padding_m": float(args.object_width_padding_m),
        "dock_stiffness_override_nm_per_rad": (
            None if args.dock_stiffness is None else float(args.dock_stiffness)
        ),
        "dock_damping_override_nms_per_rad": (
            None if args.dock_damping is None else float(args.dock_damping)
        ),
        "dock_velocity_limit_override_rad_s": (
            None
            if args.dock_velocity_limit is None
            else float(args.dock_velocity_limit)
        ),
        "dock_armature_override_kg_m2": (
            None
            if args.dock_armature_kg_m2 is None
            else float(args.dock_armature_kg_m2)
        ),
        "object_mass_override_kg": (
            None if args.object_mass_kg is None else float(args.object_mass_kg)
        ),
        "object_friction_override": (
            None if args.object_friction is None else float(args.object_friction)
        ),
        "selected_gripper_friction_override": (
            None
            if args.selected_gripper_friction is None
            else float(args.selected_gripper_friction)
        ),
        "selected_gripper_compliant_contact_stiffness_override_n_per_m": (
            None
            if args.selected_gripper_compliant_contact_stiffness is None
            else float(args.selected_gripper_compliant_contact_stiffness)
        ),
        "selected_gripper_compliant_contact_damping_override_n_s_per_m": (
            None
            if args.selected_gripper_compliant_contact_damping is None
            else float(args.selected_gripper_compliant_contact_damping)
        ),
        "proxy_pad_requested": bool(args.proxy_pad),
        "cone_proxy_pad_requested": bool(args.cone_proxy_pad),
        "contact_closure_joint_speed_override_radps": (
            None
            if args.contact_closure_joint_speed_radps is None
            else float(args.contact_closure_joint_speed_radps)
        ),
        "normal_force_target_override_n": (
            None
            if args.normal_force_target_n is None
            else float(args.normal_force_target_n)
        ),
        "payload_load_transfer_override_s": (
            None
            if args.payload_load_transfer_s is None
            else float(args.payload_load_transfer_s)
        ),
        "contact_stall_speed_threshold_override_mps": (
            None
            if args.contact_stall_speed_threshold_mps is None
            else float(args.contact_stall_speed_threshold_mps)
        ),
        "max_slip_speed_override_mps": (
            None
            if args.max_slip_speed_mps is None
            else float(args.max_slip_speed_mps)
        ),
        "max_contact_point_slip_displacement_override_m": (
            None
            if args.max_contact_point_slip_displacement_m is None
            else float(args.max_contact_point_slip_displacement_m)
        ),
        "peak_torque_window_override_s": (
            None
            if args.peak_torque_window_s is None
            else float(args.peak_torque_window_s)
        ),
        "post_grasp_joint_torque_bias_override_nm": (
            None
            if args.post_grasp_joint_torque_bias_nm is None
            else float(args.post_grasp_joint_torque_bias_nm)
        ),
        "slip_speed_safe_hold_disabled": bool(
            args.disable_slip_speed_safe_hold
        ),
        "all_safe_hold_disabled": bool(args.disable_all_safe_hold),
        "object_rotation_lock_requested": bool(args.lock_object_rotation),
        "anchor_hold_joint_correction_requested": bool(
            args.anchor_hold_joint_correction
        ),
        "loaded_state_rebase_requested": bool(args.loaded_state_rebase),
        "world_fixed_object_requested": bool(args.world_fixed_object),
        "kinematic_base_isolation_requested": bool(
            args.kinematic_base_isolation
        ),
        "precontact_fixture_report": (
            None if precontact_source_path is None else str(precontact_source_path)
        ),
        "precontact_base_pose": precontact_base_pose,
        "near_contact_fixture_report": (
            None
            if near_contact_source_path is None
            else str(near_contact_source_path)
        ),
        "near_contact_state": near_contact_state,
        "fixture_height_offset_m": float(args.fixture_height_offset_m),
        "fixture_opening_duration_s": float(args.fixture_opening_duration_s),
        "fixture_opening_source_report": (
            None
            if fixture_opening_source_path is None
            else str(fixture_opening_source_path)
        ),
        "fixture_opening_velocity_targets_radps": (
            fixture_opening_velocity_targets
        ),
        "state_trace_path": (
            None
            if args.state_trace_path is None
            else str(Path(args.state_trace_path).expanduser().resolve())
        ),
        "state_trace_frame_stride": int(args.state_trace_frame_stride),
        "qclose_fixture_report": (
            None if qclose_source_path is None else str(qclose_source_path)
        ),
        "qclose_fixture_state_trace": (
            None
            if qclose_trace_source_path is None
            else str(qclose_trace_source_path)
        ),
        "qclose_trace_fixture_provenance": (
            None
            if qclose_report is None
            else qclose_report.get(
                "order8_natural_contact_qclose_trace_fixture_provenance"
            )
        ),
        "qclose_base_pose": qclose_base_pose,
        "qclose_joint_positions_rad": qclose_joint_positions,
        "qclose_checkpoint_state": qclose_checkpoint_state,
        "zero_qclose_velocities": bool(args.zero_qclose_velocities),
        "force_convert": bool(args.force_convert),
        "profile_output": (
            None
            if args.profile_output is None
            else str(Path(args.profile_output).resolve())
        ),
        "max_simulation_time_s": args.max_simulation_time_s,
        "diagnostic_time_budget_reached": diagnostic_time_budget_reached,
        "wall_elapsed_s": float(wall_elapsed_s),
        "probe_command": command,
        "report": raw_report,
    }
    output_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(diagnostic, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    summary = {
        "diagnostic_version": diagnostic["diagnostic_version"],
        "acceptance_eligible": False,
        "wall_elapsed_s": wall_elapsed_s,
        "simulation_time_s": raw_report.get("order8_natural_contact_simulation_time_s"),
        "diagnostic_stop_reached": raw_report.get(
            "order8_natural_contact_diagnostic_stop_reached"
        ),
        "continue_after_force_ramp": raw_report.get(
            "order8_natural_contact_diagnostic_continue_after_force_ramp"
        ),
        "precontact_fixture": raw_report.get(
            "order8_natural_contact_diagnostic_precontact_fixture"
        ),
        "near_contact_fixture": raw_report.get(
            "order8_natural_contact_diagnostic_near_contact_fixture"
        ),
        "qclose_fixture": raw_report.get(
            "order8_natural_contact_diagnostic_qclose_fixture"
        ),
        "world_fixed_base": raw_report.get(
            "order8_natural_contact_diagnostic_world_fixed_base"
        ),
        "world_fixed_body_path": raw_report.get(
            "order8_natural_contact_diagnostic_world_fixed_body_path"
        ),
        "world_fixed_object": raw_report.get(
            "order8_natural_contact_diagnostic_world_fixed_object"
        ),
        "diagnostic_time_budget_reached": diagnostic_time_budget_reached,
        "failure_reason": raw_report.get("order8_natural_contact_failure_reason"),
        "monitor_passed": (
            raw_report.get("order8_natural_contact_monitor_result", {}).get(
                "passed"
            )
        ),
        "last_force_scale": raw_report.get(
            "order8_natural_contact_last_contact_force_scale"
        ),
        "last_surface_clearance_m_by_anchor": raw_report.get(
            "order8_natural_contact_last_contact_mesh_surface_clearance_m_by_anchor"
        ),
        "last_actuator_telemetry": raw_report.get(
            "order8_natural_contact_last_dock_actuator_telemetry"
        ),
        "report_path": str(output_path),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    if args.continue_after_force_ramp:
        return (
            0
            if raw_report.get("order8_natural_contact_monitor_result", {}).get(
                "passed"
            )
            is True
            else 1
        )
    return 0 if (
        raw_report.get("order8_natural_contact_diagnostic_stop_reached") is True
        or diagnostic_time_budget_reached
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
