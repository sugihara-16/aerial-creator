from __future__ import annotations

"""Accelerated, acceptance-ineligible Order 8 contact-force diagnostic."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.order8 import load_order8_natural_contact_config
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
            "Dock collision meshes. Omit it to retain the configured production "
            "value."
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
        "--max-cumulative-slip-m",
        type=float,
        default=None,
        help=(
            "Temporary diagnostic-only cumulative selected-contact slip "
            "limit. Use this only to test whether lift pickup settles inside "
            "the approved tangential surface region."
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
    selected_gripper_friction: float | None = None,
    normal_force_target_n: float | None = None,
    contact_stall_speed_threshold_mps: float | None = None,
    max_slip_speed_mps: float | None = None,
    max_cumulative_slip_m: float | None = None,
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
        max_cumulative_tangential_slip_m=(
            float(config.max_cumulative_tangential_slip_m)
            if max_cumulative_slip_m is None
            else float(max_cumulative_slip_m)
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

    joint_positions = report.get(
        "order8_natural_contact_last_joint_positions_rad"
    )
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

    clearances = report.get(
        "order8_natural_contact_last_contact_mesh_surface_clearance_m_by_anchor"
    )
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
    if report.get("order8_natural_contact_qclose_checkpoint_base_pose") is not None:
        raise ValueError(
            "near-contact fixture must precede q_close; use q_close fixture "
            "diagnostics for an already arrested state"
        )

    return {
        "base_pose": finite_pose(
            "order8_natural_contact_last_measured_base_module_pose"
        ),
        "joint_positions_rad": normalized_joint_positions,
        "object_pose": finite_pose(
            "order8_natural_contact_last_measured_object_pose"
        ),
        "source_surface_clearance_m_by_anchor": {
            str(anchor_id): float(value)
            for anchor_id, value in clearances.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if args.normal_force_target_n is not None and (
        not math.isfinite(float(args.normal_force_target_n))
        or float(args.normal_force_target_n) <= 0.0
    ):
        raise ValueError(
            "diagnostic normal-force target must be finite and positive"
        )
    if args.contact_stall_speed_threshold_mps is not None and (
        not math.isfinite(float(args.contact_stall_speed_threshold_mps))
        or float(args.contact_stall_speed_threshold_mps) <= 0.0
    ):
        raise ValueError(
            "diagnostic contact-stall speed threshold must be finite and positive"
        )
    if args.max_slip_speed_mps is not None and (
        not math.isfinite(float(args.max_slip_speed_mps))
        or float(args.max_slip_speed_mps) <= 0.0
    ):
        raise ValueError("diagnostic slip-speed limit must be finite and positive")
    if args.max_cumulative_slip_m is not None and (
        not math.isfinite(float(args.max_cumulative_slip_m))
        or float(args.max_cumulative_slip_m) <= 0.0
    ):
        raise ValueError(
            "diagnostic cumulative-slip limit must be finite and positive"
        )
    if args.peak_torque_window_s is not None and (
        not math.isfinite(float(args.peak_torque_window_s))
        or float(args.peak_torque_window_s) <= 0.0
    ):
        raise ValueError(
            "diagnostic peak-torque window must be finite and positive"
        )
    if args.max_simulation_time_s is not None and args.max_simulation_time_s <= 0.0:
        raise ValueError("diagnostic max simulation time must be positive")
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
        )
    )
    if fixture_mode_count > 1:
        raise ValueError(
            "--full-sequence, --precontact-fixture-report, "
            "--near-contact-fixture-report, and --qclose-fixture-report are "
            "mutually exclusive"
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
    qclose_base_pose = None
    qclose_joint_positions = None
    qclose_checkpoint_state = None
    if qclose_source_path is not None:
        qclose_payload = json.loads(qclose_source_path.read_text(encoding="utf-8"))
        qclose_report = qclose_payload.get("report", qclose_payload)
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
            "--qclose-fixture-report checkpoint"
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
        object_friction=args.object_friction,
        contact_dwell_s=args.contact_dwell_s,
        selected_gripper_friction=args.selected_gripper_friction,
        normal_force_target_n=args.normal_force_target_n,
        contact_stall_speed_threshold_mps=(
            args.contact_stall_speed_threshold_mps
        ),
        max_slip_speed_mps=args.max_slip_speed_mps,
        max_cumulative_slip_m=args.max_cumulative_slip_m,
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
        "diagnostic_version": "order8_contact_force_fast_diagnostic_v17",
        "diagnostic_only": True,
        "diagnostic_force_fixture": not contact_sequence,
        "diagnostic_precontact_fixture": precontact_base_pose is not None,
        "diagnostic_near_contact_fixture": near_contact_state is not None,
        "diagnostic_qclose_fixture": qclose_base_pose is not None,
        "diagnostic_full_sequence": bool(args.full_sequence),
        "continue_after_force_ramp": bool(args.continue_after_force_ramp),
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
        "object_friction_override": (
            None if args.object_friction is None else float(args.object_friction)
        ),
        "selected_gripper_friction_override": (
            None
            if args.selected_gripper_friction is None
            else float(args.selected_gripper_friction)
        ),
        "normal_force_target_override_n": (
            None
            if args.normal_force_target_n is None
            else float(args.normal_force_target_n)
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
        "max_cumulative_slip_override_m": (
            None
            if args.max_cumulative_slip_m is None
            else float(args.max_cumulative_slip_m)
        ),
        "peak_torque_window_override_s": (
            None
            if args.peak_torque_window_s is None
            else float(args.peak_torque_window_s)
        ),
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
