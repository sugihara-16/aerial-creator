#!/usr/bin/env python3
from __future__ import annotations

"""Translate or retime robot-side targets in C3 nominal components.

This is a narrow offline bucket-repair tool.  Translation preserves every
joint target, contact assignment, object target, and phase duration while
moving the centroidal and free-anchor pose targets together.  The output
remains pending until the normal independent collision admission is rerun.

Selected phases can also be uniformly retimed.  Retiming resamples the same
geometric path at the original knot rate and scales velocity targets without
lowering the nominal IK reference rate or changing the final configuration.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
)
from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION,
    ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetEntry,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash


REPAIR_VERSION = "order9_c3_rigid_robot_target_translation_v1"
SMOOTH_OFFSET_VERSION = "order9_c3_smoothstep_robot_target_translation_v1"
RETIME_VERSION = "order9_c3_uniform_phase_retime_v2"
PRELIFT_VERSION = "order9_c3_vertical_prelift_then_phase_motion_v1"
OPEN_THEN_LIFT_VERSION = "order9_c3_partial_open_at_fixed_body_then_lift_v2"
HOLD_POSTURE_VERSION = "order9_c3_hold_reference_posture_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest")
    parser.add_argument("output")
    parser.add_argument(
        "--bucket-id",
        action="append",
        default=[],
        help=(
            "Repair only the named bucket. May be repeated; the source-set "
            "binding is preserved in the derived manifest."
        ),
    )
    parser.add_argument("--offset-x-mm", type=float, default=0.0)
    parser.add_argument("--offset-y-mm", type=float, default=0.0)
    parser.add_argument("--offset-z-mm", type=float, default=0.0)
    parser.add_argument(
        "--phase-offset-z-mm",
        action="append",
        default=[],
        metavar="PHASE=START[:END]",
        help=(
            "Override the global z offset for one phase. A START:END pair "
            "is linearly interpolated and its velocity is included in the "
            "centroidal target."
        ),
    )
    parser.add_argument(
        "--phase-smooth-offset-z-mm",
        action="append",
        default=[],
        metavar="PHASE=START[:END]",
        help=(
            "Override the global z offset for one phase using a smoothstep "
            "profile. Position and velocity remain continuous at both phase "
            "boundaries."
        ),
    )
    parser.add_argument(
        "--phase-time-scale",
        action="append",
        default=[],
        metavar="PHASE=SCALE",
        help=(
            "Uniformly retime one phase while preserving its original knot "
            "rate. Values below 1 shorten it and values above 1 lengthen it."
        ),
    )
    parser.add_argument(
        "--phase-prelift-z-mm",
        action="append",
        default=[],
        metavar="PHASE=LIFT_MM:DURATION_S",
        help=(
            "Prepend a smooth vertical robot-only lift while holding the "
            "phase-start posture, then replay the original phase path at a "
            "constant elevated offset."
        ),
    )
    parser.add_argument(
        "--phase-open-then-lift-z-mm",
        action="append",
        default=[],
        metavar="PHASE=EXTRA_LIFT_MM:DURATION_S",
        help=(
            "Replay the phase joint path while holding the robot body at the "
            "phase-start pose, then move the opened robot to the original "
            "phase-end body pose plus EXTRA_LIFT_MM."
        ),
    )
    parser.add_argument(
        "--phase-open-fraction",
        action="append",
        default=[],
        metavar="PHASE=FRACTION",
        help=(
            "Use only the leading FRACTION of the phase joint path before "
            "the body lift. The default is the complete path (1.0)."
        ),
    )
    parser.add_argument(
        "--phase-hold-posture-from",
        action="append",
        default=[],
        metavar="TARGET_PHASE=SOURCE_PHASE",
        help=(
            "Hold SOURCE_PHASE's final joint and free-anchor posture while "
            "following TARGET_PHASE's centroidal path."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_path = _resolve(args.source_manifest)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    offset = (
        1.0e-3 * float(args.offset_x_mm),
        1.0e-3 * float(args.offset_y_mm),
        1.0e-3 * float(args.offset_z_mm),
    )
    phase_z_offsets = _parse_phase_z_offsets(args.phase_offset_z_mm)
    phase_smooth_z_offsets = _parse_phase_z_offsets(
        args.phase_smooth_offset_z_mm
    )
    phase_time_scales = _parse_phase_time_scales(args.phase_time_scale)
    phase_prelifts = _parse_phase_prelifts(args.phase_prelift_z_mm)
    phase_open_then_lifts = _parse_phase_open_then_lifts(
        args.phase_open_then_lift_z_mm
    )
    phase_open_fractions = _parse_phase_open_fractions(
        args.phase_open_fraction
    )
    phase_hold_postures = _parse_phase_hold_postures(
        args.phase_hold_posture_from
    )
    unknown_fraction_phases = set(phase_open_fractions).difference(
        phase_open_then_lifts
    )
    if unknown_fraction_phases:
        raise ValueError(
            "phase open fraction requires open-then-lift for: "
            + ", ".join(sorted(unknown_fraction_phases))
        )
    if (
        max(abs(value) for value in offset) <= 0.0
        and not phase_z_offsets
        and not phase_smooth_z_offsets
        and not phase_time_scales
        and not phase_prelifts
        and not phase_open_then_lifts
        and not phase_hold_postures
    ):
        raise ValueError("at least one translation or time scale must be set")
    repeated_offset_phases = set(phase_z_offsets).intersection(
        phase_smooth_z_offsets
    )
    if repeated_offset_phases:
        raise ValueError(
            "linear and smooth phase z offset cannot target the same phase: "
            + ", ".join(sorted(repeated_offset_phases))
        )
    all_phase_z_offsets = {
        **phase_z_offsets,
        **phase_smooth_z_offsets,
    }
    ambiguous_phases = set(phase_prelifts).intersection(all_phase_z_offsets)
    ambiguous_phases.update(
        set(phase_open_then_lifts).intersection(
            set(all_phase_z_offsets)
            | set(phase_prelifts)
            | set(phase_time_scales)
        )
    )
    if ambiguous_phases:
        raise ValueError(
            "phase prelift/open and phase z/retime operations conflict for: "
            + ", ".join(sorted(ambiguous_phases))
        )

    source = Order9C3NominalTrajectorySetManifest.from_json(
        source_path.read_text(encoding="utf-8")
    )
    source.validate()
    output.mkdir(parents=True)
    selected_bucket_ids = set(args.bucket_id)
    source_entries = tuple(
        entry
        for entry in source.entries
        if not selected_bucket_ids or entry.bucket_id in selected_bucket_ids
    )
    missing_bucket_ids = selected_bucket_ids.difference(
        entry.bucket_id for entry in source_entries
    )
    if missing_bucket_ids:
        raise ValueError(
            "source manifest does not contain requested buckets: "
            + ", ".join(sorted(missing_bucket_ids))
        )

    entries: list[Order9C3NominalTrajectorySetEntry] = []
    for source_entry in source_entries:
        source_artifact_path = source_path.parent / source_entry.artifact_path
        if hash_file(source_artifact_path) != source_entry.artifact_sha256:
            raise ValueError(f"source artifact changed: {source_entry.bucket_id}")
        source_artifact = Order9C3NominalTrajectoryArtifact.from_json(
            source_artifact_path.read_text(encoding="utf-8")
        )
        source_artifact.validate()
        destination = output / "buckets" / source_entry.bucket_id
        shutil.copytree(source_artifact_path.parent, destination)

        phases: dict[str, ContactWrenchTrajectory] = {}
        phase_entries: list[Order9C3NominalPhaseArtifact] = []
        for phase_entry in source_artifact.phase_trajectories:
            phase_path = destination / phase_entry.trajectory_path
            trajectory = ContactWrenchTrajectory.from_json(
                phase_path.read_text(encoding="utf-8")
            )
            start_z, end_z = all_phase_z_offsets.get(
                phase_entry.phase, (offset[2], offset[2])
            )
            _translate_robot_targets(
                trajectory,
                (offset[0], offset[1], start_z),
                (offset[0], offset[1], end_z),
                smooth=phase_entry.phase in phase_smooth_z_offsets,
            )
            held_from_phase = phase_hold_postures.get(phase_entry.phase)
            if held_from_phase is not None:
                if held_from_phase not in phases:
                    raise ValueError(
                        f"held posture source must precede target: "
                        f"{phase_entry.phase}={held_from_phase}"
                    )
                _hold_posture_from_reference(
                    trajectory,
                    reference=phases[held_from_phase],
                )
            prelift = phase_prelifts.get(phase_entry.phase)
            if prelift is not None:
                trajectory = _prepend_vertical_prelift(
                    trajectory,
                    lift_m=prelift[0],
                    duration_s=prelift[1],
                )
            open_then_lift = phase_open_then_lifts.get(phase_entry.phase)
            if open_then_lift is not None:
                trajectory = _open_at_fixed_body_then_lift(
                    trajectory,
                    extra_lift_m=open_then_lift[0],
                    lift_duration_s=open_then_lift[1],
                    open_fraction=phase_open_fractions.get(
                        phase_entry.phase, 1.0
                    ),
                )
            scale = phase_time_scales.get(phase_entry.phase)
            if scale is not None:
                trajectory = _time_stretch_trajectory(trajectory, scale=scale)
            trajectory.validate()
            phase_path.write_text(
                trajectory.to_json(indent=2) + "\n", encoding="utf-8"
            )
            phases[phase_entry.phase] = trajectory
            phase_entries.append(
                Order9C3NominalPhaseArtifact(
                    phase=phase_entry.phase,
                    trajectory_path=phase_entry.trajectory_path,
                    trajectory_sha256=hash_file(phase_path),
                    trajectory_hash=stable_hash(trajectory.to_dict()),
                    knot_count=len(trajectory.knots),
                    generation_method=(
                        f"{phase_entry.generation_method}+{REPAIR_VERSION}"
                        + (
                            f"+{SMOOTH_OFFSET_VERSION}"
                            if phase_entry.phase in phase_smooth_z_offsets
                            else ""
                        )
                        + (
                            f"+{RETIME_VERSION}"
                            if scale is not None
                            else ""
                        )
                        + (
                            f"+{PRELIFT_VERSION}"
                            if prelift is not None
                            else ""
                        )
                        + (
                            f"+{OPEN_THEN_LIFT_VERSION}"
                            if open_then_lift is not None
                            else ""
                        )
                        + (
                            f"+{HOLD_POSTURE_VERSION}"
                            if held_from_phase is not None
                            else ""
                        )
                    ),
                    collision_validation_status="pending_offline_admission",
                )
            )

        timeline_path = destination / source_artifact.timeline_path
        timeline_path.write_text(
            json.dumps(
                {
                    "timeline_version": ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
                    "semantic_scope": (
                        "offline precomputed complete eight-phase nominal; "
                        "configuration-space planning is never executed by "
                        "the training or inference runtime"
                    ),
                    "proxy_collision_validation_status": (
                        "pending_offline_admission"
                    ),
                    "duration_s": sum(
                        float(value.horizon_s) for value in phases.values()
                    ),
                    "records": _complete_timeline(phases),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        evidence = dict(source_artifact.selection_evidence)
        evidence["rigid_robot_target_translation"] = {
            "repair_version": REPAIR_VERSION,
            "offset_world_m": list(offset),
            "phase_z_offset_start_end_m": {
                phase: list(values)
                for phase, values in phase_z_offsets.items()
            },
            "phase_smooth_z_offset_start_end_m": {
                phase: list(values)
                for phase, values in phase_smooth_z_offsets.items()
            },
            "smooth_offset_boundary_velocity_zero": bool(
                phase_smooth_z_offsets
            ),
            "joint_targets_changed": False,
            "object_targets_changed": False,
            "contact_assignments_changed": False,
            "runtime_planner_enabled": False,
        }
        if phase_time_scales:
            evidence["uniform_phase_time_stretch"] = {
                "repair_version": RETIME_VERSION,
                "phase_time_scales": dict(phase_time_scales),
                "geometric_path_changed": False,
                "final_configuration_changed": False,
                "original_knot_rate_preserved": True,
                "velocity_targets_scaled": True,
                "runtime_planner_enabled": False,
            }
        if phase_prelifts:
            evidence["vertical_prelift_then_phase_motion"] = {
                "repair_version": PRELIFT_VERSION,
                "phase_lift_m_and_duration_s": {
                    phase: list(values)
                    for phase, values in phase_prelifts.items()
                },
                "phase_order_changed": False,
                "source_joint_path_changed": False,
                "final_joint_configuration_changed": False,
                "robot_target_path_changed": True,
                "object_targets_changed": False,
                "runtime_planner_enabled": False,
            }
        if phase_open_then_lifts:
            evidence["open_at_fixed_body_then_lift"] = {
                "repair_version": OPEN_THEN_LIFT_VERSION,
                "phase_extra_lift_m_and_duration_s": {
                    phase: list(values)
                    for phase, values in phase_open_then_lifts.items()
                },
                "phase_open_fractions": {
                    phase: phase_open_fractions.get(phase, 1.0)
                    for phase in phase_open_then_lifts
                },
                "phase_order_changed": False,
                "source_joint_path_truncated": any(
                    fraction < 1.0
                    for fraction in phase_open_fractions.values()
                ),
                "final_joint_configuration_changed": any(
                    fraction < 1.0
                    for fraction in phase_open_fractions.values()
                ),
                "robot_target_path_changed": True,
                "object_targets_changed": False,
                "runtime_planner_enabled": False,
            }
        if phase_hold_postures:
            evidence["hold_reference_posture"] = {
                "repair_version": HOLD_POSTURE_VERSION,
                "target_to_source_phase": dict(phase_hold_postures),
                "centroidal_paths_changed": False,
                "object_targets_changed": False,
                "runtime_planner_enabled": False,
            }
        payload = source_artifact.to_dict()
        payload.update(
            {
                "timeline_sha256": hash_file(timeline_path),
                "phase_trajectories": [
                    value.to_dict() for value in phase_entries
                ],
                "proxy_collision_validation_status": (
                    "pending_offline_admission"
                ),
                "selection_evidence": evidence,
            }
        )
        artifact = Order9C3NominalTrajectoryArtifact.from_dict(payload)
        artifact.validate()
        artifact_path = destination / "manifest.json"
        artifact_path.write_text(
            artifact.to_json(indent=2) + "\n", encoding="utf-8"
        )
        validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
        entries.append(
            Order9C3NominalTrajectorySetEntry(
                bucket_id=source_entry.bucket_id,
                split=source_entry.split,
                module_count=source_entry.module_count,
                structural_hash=source_entry.structural_hash,
                artifact_path=str(artifact_path.relative_to(output)),
                artifact_sha256=hash_file(artifact_path),
            )
        )

    manifest = Order9C3NominalTrajectorySetManifest(
        bucket_manifest_path=source.bucket_manifest_path,
        bucket_manifest_sha256=source.bucket_manifest_sha256,
        physical_model_hash=source.physical_model_hash,
        entries=entries,
        metadata={
            "source_manifest_path": _portable(source_path),
            "source_manifest_sha256": hash_file(source_path),
            "repair_version": REPAIR_VERSION,
            "offset_world_m": list(offset),
            "phase_z_offset_start_end_m": {
                phase: list(values)
                for phase, values in phase_z_offsets.items()
            },
            "phase_smooth_z_offset_start_end_m": {
                phase: list(values)
                for phase, values in phase_smooth_z_offsets.items()
            },
            "phase_time_scales": dict(phase_time_scales),
            "phase_prelift_m_and_duration_s": {
                phase: list(values)
                for phase, values in phase_prelifts.items()
            },
            "phase_open_then_lift_m_and_duration_s": {
                phase: list(values)
                for phase, values in phase_open_then_lifts.items()
            },
            "phase_open_fractions": dict(phase_open_fractions),
            "phase_hold_posture_from": dict(phase_hold_postures),
            "selected_bucket_ids": [
                entry.bucket_id for entry in source_entries
            ],
            "independent_collision_admission": "pending",
        },
        manifest_version=ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION,
    )
    manifest.validate()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        manifest.to_json(indent=2) + "\n", encoding="utf-8"
    )
    print(
        "ORDER9_C3_RIGID_TARGET_TRANSLATION="
        f"path={manifest_path} sha256={hash_file(manifest_path)} "
        f"buckets={len(entries)} offset_m={offset}",
        flush=True,
    )
    return 0


def _translate_robot_targets(
    trajectory: ContactWrenchTrajectory,
    start_offset: tuple[float, float, float],
    end_offset: tuple[float, float, float],
    *,
    smooth: bool = False,
) -> None:
    for knot in trajectory.knots:
        alpha = min(
            max(float(knot.t_rel_s) / float(trajectory.horizon_s), 0.0),
            1.0,
        )
        interpolation = (
            alpha * alpha * (3.0 - 2.0 * alpha)
            if smooth
            else alpha
        )
        interpolation_rate = (
            6.0 * alpha * (1.0 - alpha) / float(trajectory.horizon_s)
            if smooth
            else 1.0 / float(trajectory.horizon_s)
        )
        offset = tuple(
            start + interpolation * (end - start)
            for start, end in zip(start_offset, end_offset)
        )
        centroidal = knot.centroidal_target
        if centroidal is not None and centroidal.com_pos_world is not None:
            centroidal.com_pos_world = tuple(
                float(value) + delta
                for value, delta in zip(centroidal.com_pos_world, offset)
            )
            if centroidal.com_vel_world is not None:
                velocity_offset = tuple(
                    (end - start) * interpolation_rate
                    for start, end in zip(start_offset, end_offset)
                )
                centroidal.com_vel_world = tuple(
                    float(value) + delta
                    for value, delta in zip(
                        centroidal.com_vel_world, velocity_offset
                    )
                )
        posture = knot.posture_target
        if posture is None or posture.free_anchor_pose_targets is None:
            continue
        posture.free_anchor_pose_targets = {
            anchor_id: tuple(
                [
                    float(pose[0]) + offset[0],
                    float(pose[1]) + offset[1],
                    float(pose[2]) + offset[2],
                    *[float(value) for value in pose[3:]],
                ]
            )
            for anchor_id, pose in posture.free_anchor_pose_targets.items()
        }


def _time_stretch_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    scale: float,
) -> ContactWrenchTrajectory:
    if not float(scale) > 0.0:
        raise ValueError("phase time scale must be positive")
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.0)
    executor.install(source, plan_start_time_s=0.0)
    new_horizon_s = float(source.horizon_s) * float(scale)
    dt_s = float(source.dt_s)
    sample_count = int(new_horizon_s / dt_s + 1.0e-9)
    times = [index * dt_s for index in range(sample_count + 1)]
    if not times or new_horizon_s - times[-1] > 1.0e-9:
        times.append(new_horizon_s)
    else:
        times[-1] = new_horizon_s
    knots = []
    for stretched_time_s in times:
        source_time_s = min(stretched_time_s / float(scale), source.horizon_s)
        knot = executor.sample(time_s=source_time_s).active_knot
        knot.t_rel_s = float(stretched_time_s)
        _scale_velocity_targets(knot, scale=float(scale))
        knots.append(knot)
    result = ContactWrenchTrajectory(
        horizon_s=new_horizon_s,
        dt_s=dt_s,
        knots=knots,
        derived_mode_label=source.derived_mode_label,
        contract_version=source.contract_version,
    )
    result.validate()
    return result


def _prepend_vertical_prelift(
    trajectory: ContactWrenchTrajectory,
    *,
    lift_m: float,
    duration_s: float,
) -> ContactWrenchTrajectory:
    if not float(lift_m) > 0.0 or not float(duration_s) > 0.0:
        raise ValueError("phase prelift distance and duration must be positive")
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.0)
    executor.install(source, plan_start_time_s=0.0)
    new_horizon_s = float(duration_s) + float(source.horizon_s)
    dt_s = float(source.dt_s)
    sample_count = int(new_horizon_s / dt_s + 1.0e-9)
    times = [index * dt_s for index in range(sample_count + 1)]
    if not times or new_horizon_s - times[-1] > 1.0e-9:
        times.append(new_horizon_s)
    else:
        times[-1] = new_horizon_s
    knots = []
    for output_time_s in times:
        if output_time_s <= duration_s:
            knot = executor.sample(time_s=0.0).active_knot
            alpha = min(max(output_time_s / duration_s, 0.0), 1.0)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            smooth_rate = 6.0 * alpha * (1.0 - alpha) / duration_s
            _zero_motion_targets(knot)
            _translate_robot_targets_at_knot(
                knot,
                offset=(0.0, 0.0, lift_m * smooth),
                velocity_offset=(0.0, 0.0, lift_m * smooth_rate),
            )
        else:
            source_time_s = min(output_time_s - duration_s, source.horizon_s)
            knot = executor.sample(time_s=source_time_s).active_knot
            _translate_robot_targets_at_knot(
                knot,
                offset=(0.0, 0.0, lift_m),
                velocity_offset=(0.0, 0.0, 0.0),
            )
        knot.t_rel_s = float(output_time_s)
        knots.append(knot)
    result = ContactWrenchTrajectory(
        horizon_s=new_horizon_s,
        dt_s=dt_s,
        knots=knots,
        derived_mode_label=source.derived_mode_label,
        contract_version=source.contract_version,
    )
    result.validate()
    return result


def _open_at_fixed_body_then_lift(
    trajectory: ContactWrenchTrajectory,
    *,
    extra_lift_m: float,
    lift_duration_s: float,
    open_fraction: float = 1.0,
) -> ContactWrenchTrajectory:
    if float(extra_lift_m) < 0.0 or not float(lift_duration_s) > 0.0:
        raise ValueError(
            "open-then-lift extra distance must be non-negative and duration positive"
        )
    if not 0.0 < float(open_fraction) <= 1.0:
        raise ValueError("open fraction must be in (0, 1]")
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.0)
    executor.install(source, plan_start_time_s=0.0)
    start = executor.sample(time_s=0.0).active_knot
    open_horizon_s = float(source.horizon_s) * float(open_fraction)
    opened = executor.sample(time_s=open_horizon_s).active_knot
    body_end = executor.sample(time_s=source.horizon_s).active_knot
    start_pose = _centroidal_pose(start)
    final_pose = list(_centroidal_pose(body_end))
    final_pose[2] += float(extra_lift_m)
    final_pose_tuple = tuple(final_pose)
    new_horizon_s = open_horizon_s + float(lift_duration_s)
    dt_s = float(source.dt_s)
    sample_count = int(new_horizon_s / dt_s + 1.0e-9)
    times = [index * dt_s for index in range(sample_count + 1)]
    if not times or new_horizon_s - times[-1] > 1.0e-9:
        times.append(new_horizon_s)
    else:
        times[-1] = new_horizon_s
    knots = []
    for output_time_s in times:
        if output_time_s <= open_horizon_s:
            knot = executor.sample(time_s=output_time_s).active_knot
            source_pose = _centroidal_pose(knot)
            _replace_robot_body_pose(
                knot,
                source_pose=source_pose,
                target_pose=start_pose,
                target_linear_velocity=(0.0, 0.0, 0.0),
            )
        else:
            knot = InteractionKnot.from_dict(opened.to_dict())
            alpha = min(
                max(
                    (output_time_s - open_horizon_s) / lift_duration_s,
                    0.0,
                ),
                1.0,
            )
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            smooth_rate = 6.0 * alpha * (1.0 - alpha) / lift_duration_s
            target_pose = (
                *(
                    start_pose[index]
                    + smooth * (final_pose_tuple[index] - start_pose[index])
                    for index in range(3)
                ),
                *_normalized_lerp_quaternion_values(
                    start_pose[3:7], final_pose_tuple[3:7], smooth
                ),
            )
            target_velocity = tuple(
                (final_pose_tuple[index] - start_pose[index]) * smooth_rate
                for index in range(3)
            )
            _zero_motion_targets(knot)
            _replace_robot_body_pose(
                knot,
                source_pose=_centroidal_pose(knot),
                target_pose=target_pose,
                target_linear_velocity=target_velocity,
            )
        knot.t_rel_s = float(output_time_s)
        knots.append(knot)
    result = ContactWrenchTrajectory(
        horizon_s=new_horizon_s,
        dt_s=dt_s,
        knots=knots,
        derived_mode_label=source.derived_mode_label,
        contract_version=source.contract_version,
    )
    result.validate()
    return result


def _hold_posture_from_reference(
    trajectory: ContactWrenchTrajectory,
    *,
    reference: ContactWrenchTrajectory,
) -> None:
    trajectory.validate()
    reference.validate()
    reference_knot = reference.knots[-1]
    reference_posture = reference_knot.posture_target
    if reference_posture is None or reference_posture.joint_pos_target is None:
        raise ValueError("held posture source has no joint position target")
    reference_body_pose = _centroidal_pose(reference_knot)
    reference_anchor_poses = reference_posture.free_anchor_pose_targets
    for knot in trajectory.knots:
        posture = knot.posture_target
        if posture is None:
            raise ValueError("held posture target has no posture target")
        posture.joint_pos_target = dict(reference_posture.joint_pos_target)
        posture.joint_vel_target = {
            key: 0.0 for key in reference_posture.joint_pos_target
        }
        if reference_anchor_poses is None:
            posture.free_anchor_pose_targets = None
            continue
        target_body_pose = _centroidal_pose(knot)
        world_delta = compose_pose(
            target_body_pose,
            inverse_pose(reference_body_pose),
        )
        posture.free_anchor_pose_targets = {
            anchor_id: compose_pose(
                world_delta,
                tuple(float(value) for value in pose),
            )
            for anchor_id, pose in reference_anchor_poses.items()
        }


def _centroidal_pose(knot: InteractionKnot) -> tuple[float, ...]:
    centroidal = knot.centroidal_target
    if (
        centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise ValueError("open-then-lift requires a complete centroidal pose")
    return tuple(
        [
            *[float(value) for value in centroidal.com_pos_world],
            *[float(value) for value in centroidal.body_orientation_world],
        ]
    )


def _replace_robot_body_pose(
    knot: InteractionKnot,
    *,
    source_pose: tuple[float, ...],
    target_pose: tuple[float, ...],
    target_linear_velocity: tuple[float, float, float],
) -> None:
    centroidal = knot.centroidal_target
    if centroidal is None:
        raise ValueError("open-then-lift requires a centroidal target")
    centroidal.com_pos_world = tuple(target_pose[:3])
    centroidal.body_orientation_world = tuple(target_pose[3:7])
    centroidal.com_vel_world = tuple(target_linear_velocity)
    posture = knot.posture_target
    if posture is None or posture.free_anchor_pose_targets is None:
        return
    world_delta = compose_pose(target_pose, inverse_pose(source_pose))
    posture.free_anchor_pose_targets = {
        anchor_id: compose_pose(world_delta, tuple(float(value) for value in pose))
        for anchor_id, pose in posture.free_anchor_pose_targets.items()
    }


def _normalized_lerp_quaternion_values(
    start: tuple[float, ...],
    end: tuple[float, ...],
    alpha: float,
) -> tuple[float, float, float, float]:
    aligned = tuple(-value for value in end) if sum(
        left * right for left, right in zip(start, end)
    ) < 0.0 else end
    values = tuple(
        (1.0 - alpha) * left + alpha * right
        for left, right in zip(start, aligned)
    )
    norm = sum(value * value for value in values) ** 0.5
    if norm <= 1.0e-12:
        raise ValueError("open-then-lift quaternion interpolation is singular")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _translate_robot_targets_at_knot(
    knot: InteractionKnot,
    *,
    offset: tuple[float, float, float],
    velocity_offset: tuple[float, float, float],
) -> None:
    centroidal = knot.centroidal_target
    if centroidal is not None and centroidal.com_pos_world is not None:
        centroidal.com_pos_world = tuple(
            float(value) + delta
            for value, delta in zip(centroidal.com_pos_world, offset)
        )
        if centroidal.com_vel_world is not None:
            centroidal.com_vel_world = tuple(
                float(value) + delta
                for value, delta in zip(
                    centroidal.com_vel_world, velocity_offset
                )
            )
    posture = knot.posture_target
    if posture is None or posture.free_anchor_pose_targets is None:
        return
    posture.free_anchor_pose_targets = {
        anchor_id: tuple(
            [
                float(pose[0]) + offset[0],
                float(pose[1]) + offset[1],
                float(pose[2]) + offset[2],
                *[float(value) for value in pose[3:]],
            ]
        )
        for anchor_id, pose in posture.free_anchor_pose_targets.items()
    }


def _zero_motion_targets(knot: InteractionKnot) -> None:
    centroidal = knot.centroidal_target
    if centroidal is not None and centroidal.com_vel_world is not None:
        centroidal.com_vel_world = tuple(0.0 for _ in centroidal.com_vel_world)
    posture = knot.posture_target
    if posture is not None and posture.joint_vel_target is not None:
        posture.joint_vel_target = {
            key: 0.0 for key in posture.joint_vel_target
        }
    for target in knot.object_targets:
        if target.twist_target_world is not None:
            target.twist_target_world = [
                0.0 for _ in target.twist_target_world
            ]
        if target.generalized_qdot_target is not None:
            target.generalized_qdot_target = [
                0.0 for _ in target.generalized_qdot_target
            ]


def _scale_velocity_targets(knot: InteractionKnot, *, scale: float) -> None:
    centroidal = knot.centroidal_target
    if centroidal is not None and centroidal.com_vel_world is not None:
        centroidal.com_vel_world = tuple(
            float(value) / scale for value in centroidal.com_vel_world
        )
    posture = knot.posture_target
    if posture is not None and posture.joint_vel_target is not None:
        posture.joint_vel_target = {
            key: float(value) / scale
            for key, value in posture.joint_vel_target.items()
        }
    for target in knot.object_targets:
        if target.twist_target_world is not None:
            target.twist_target_world = [
                float(value) / scale for value in target.twist_target_world
            ]
        if target.generalized_qdot_target is not None:
            target.generalized_qdot_target = [
                float(value) / scale
                for value in target.generalized_qdot_target
            ]


def _parse_phase_z_offsets(
    values: list[str],
) -> dict[str, tuple[float, float]]:
    valid = {value.value for value in ORDER9_OBJECT_TASK_PHASES}
    result: dict[str, tuple[float, float]] = {}
    for raw in values:
        phase, separator, offset_raw = raw.partition("=")
        if not separator or phase not in valid or phase in result:
            raise ValueError(f"invalid or repeated phase z offset: {raw}")
        start_raw, pair_separator, end_raw = offset_raw.partition(":")
        start = 1.0e-3 * float(start_raw)
        end = 1.0e-3 * float(end_raw) if pair_separator else start
        result[phase] = (start, end)
    return result


def _parse_phase_time_scales(values: list[str]) -> dict[str, float]:
    valid = {value.value for value in ORDER9_OBJECT_TASK_PHASES}
    result: dict[str, float] = {}
    for raw in values:
        phase, separator, scale_raw = raw.partition("=")
        if not separator or phase not in valid or phase in result:
            raise ValueError(f"invalid or repeated phase time scale: {raw}")
        scale = float(scale_raw)
        if not scale > 0.0:
            raise ValueError(f"phase time scale must be positive: {raw}")
        result[phase] = scale
    return result


def _parse_phase_prelifts(
    values: list[str],
) -> dict[str, tuple[float, float]]:
    valid = {value.value for value in ORDER9_OBJECT_TASK_PHASES}
    result: dict[str, tuple[float, float]] = {}
    for raw in values:
        phase, separator, specification = raw.partition("=")
        lift_raw, pair_separator, duration_raw = specification.partition(":")
        if (
            not separator
            or not pair_separator
            or phase not in valid
            or phase in result
        ):
            raise ValueError(f"invalid or repeated phase prelift: {raw}")
        lift_m = 1.0e-3 * float(lift_raw)
        duration_s = float(duration_raw)
        if not lift_m > 0.0 or not duration_s > 0.0:
            raise ValueError(f"phase prelift must be positive: {raw}")
        result[phase] = (lift_m, duration_s)
    return result


def _parse_phase_open_then_lifts(
    values: list[str],
) -> dict[str, tuple[float, float]]:
    valid = {value.value for value in ORDER9_OBJECT_TASK_PHASES}
    result: dict[str, tuple[float, float]] = {}
    for raw in values:
        phase, separator, specification = raw.partition("=")
        lift_raw, pair_separator, duration_raw = specification.partition(":")
        if (
            not separator
            or not pair_separator
            or phase not in valid
            or phase in result
        ):
            raise ValueError(f"invalid or repeated phase open-then-lift: {raw}")
        extra_lift_m = 1.0e-3 * float(lift_raw)
        duration_s = float(duration_raw)
        if extra_lift_m < 0.0 or not duration_s > 0.0:
            raise ValueError(
                f"phase open-then-lift distance/duration is invalid: {raw}"
            )
        result[phase] = (extra_lift_m, duration_s)
    return result


def _parse_phase_open_fractions(values: list[str]) -> dict[str, float]:
    valid = {value.value for value in ORDER9_OBJECT_TASK_PHASES}
    result: dict[str, float] = {}
    for raw in values:
        phase, separator, fraction_raw = raw.partition("=")
        if not separator or phase not in valid or phase in result:
            raise ValueError(f"invalid or repeated phase open fraction: {raw}")
        fraction = float(fraction_raw)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"phase open fraction must be in (0, 1]: {raw}")
        result[phase] = fraction
    return result


def _parse_phase_hold_postures(values: list[str]) -> dict[str, str]:
    ordered = [value.value for value in ORDER9_OBJECT_TASK_PHASES]
    order = {phase: index for index, phase in enumerate(ordered)}
    result: dict[str, str] = {}
    for raw in values:
        target, separator, source = raw.partition("=")
        if (
            not separator
            or target not in order
            or source not in order
            or target in result
            or order[source] >= order[target]
        ):
            raise ValueError(f"invalid or repeated phase posture hold: {raw}")
        result[target] = source
    return result


def _complete_timeline(
    phases: dict[str, ContactWrenchTrajectory],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    offset_s = 0.0
    for phase in (value.value for value in ORDER9_OBJECT_TASK_PHASES):
        trajectory = phases[phase]
        for knot_index, knot in enumerate(trajectory.knots):
            if records and knot_index == 0:
                continue
            records.append(
                {
                    "sample_index": len(records),
                    "global_time_s": offset_s + float(knot.t_rel_s),
                    "phase": phase,
                    "phase_local_time_s": float(knot.t_rel_s),
                    "phase_target_reached": True,
                    "knot": knot.to_dict(),
                }
            )
        offset_s += float(trajectory.horizon_s)
    return records


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
