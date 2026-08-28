from __future__ import annotations

"""Retime an R1 nominal reference without changing its geometric path.

This module is deliberately limited to newly materialized R1 case artifacts.
It never edits the protected C3 release.  The operation is admitted only when
the resampled joint targets remain inside the rate limit already established
by the controller-free R1 screen.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_R1_NOMINAL_RETIME_VERSION = "order9_r1_uniform_nominal_retime_v1"
_PHASES = tuple(value.value for value in ORDER9_OBJECT_TASK_PHASES)


@dataclass
class Order9R1NominalRetimeAdmission(SchemaBase):
    admission_version: str
    time_scale: float
    phase_time_scales: dict[str, float]
    source_artifact_sha256: str
    retimed_artifact_sha256: str
    source_duration_s: float
    retimed_duration_s: float
    joint_rate_limit_rad_s: float
    maximum_commanded_joint_rate_rad_s: float
    minimum_joint_rate_margin_rad_s: float
    maximum_centroidal_speed_m_s: float
    maximum_object_linear_speed_m_s: float
    phase_order_unchanged: bool
    geometric_path_unchanged: bool
    endpoint_configuration_unchanged: bool
    contact_assignments_unchanged: bool
    collision_admission_inherited_from_identical_geometry: bool
    accepted: bool

    def validate(self) -> None:
        if self.admission_version != ORDER9_R1_NOMINAL_RETIME_VERSION:
            raise SchemaValidationError("R1 nominal retime admission version mismatch")
        for name in (
            "time_scale",
            "source_duration_s",
            "retimed_duration_s",
            "joint_rate_limit_rad_s",
            "maximum_commanded_joint_rate_rad_s",
            "minimum_joint_rate_margin_rad_s",
            "maximum_centroidal_speed_m_s",
            "maximum_object_linear_speed_m_s",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise SchemaValidationError(f"R1 nominal retime {name} is not finite")
        if not 0.0 < self.time_scale <= 1.0:
            raise SchemaValidationError("R1 nominal retime scale must be in (0, 1]")
        if (
            set(self.phase_time_scales) != set(_PHASES)
            or any(not 0.0 < float(value) <= 1.0 for value in self.phase_time_scales.values())
            or not math.isclose(
                self.time_scale,
                min(float(value) for value in self.phase_time_scales.values()),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise SchemaValidationError("R1 nominal phase time scales are invalid")
        if self.joint_rate_limit_rad_s <= 0.0:
            raise SchemaValidationError("R1 nominal retime joint-rate limit is invalid")
        if len(self.source_artifact_sha256) != 64 or len(
            self.retimed_artifact_sha256
        ) != 64:
            raise SchemaValidationError("R1 nominal retime hash is invalid")
        if not all(
            (
                self.phase_order_unchanged,
                self.geometric_path_unchanged,
                self.endpoint_configuration_unchanged,
                self.contact_assignments_unchanged,
                self.collision_admission_inherited_from_identical_geometry,
                self.accepted,
            )
        ):
            raise SchemaValidationError("R1 nominal retime admission was not accepted")
        if self.minimum_joint_rate_margin_rad_s < -1.0e-9:
            raise SchemaValidationError("R1 nominal retime exceeds the joint-rate limit")


def retime_order9_r1_nominal_artifact(
    manifest_path: str | Path,
    *,
    time_scale: float | None = None,
    phase_time_scales: Mapping[str, float] | None = None,
    joint_rate_limit_rad_s: float,
) -> Order9R1NominalRetimeAdmission:
    """Retime a private, not-yet-published R1 artifact in place.

    The caller is responsible for invoking this function only inside the
    temporary directory used by R1 case materialization.
    """

    source = Path(manifest_path).resolve()
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(source)
    if (time_scale is None) == (phase_time_scales is None):
        raise SchemaValidationError(
            "R1 nominal retime requires exactly one uniform or per-phase scale"
        )
    scales = (
        {phase: float(time_scale) for phase in _PHASES}
        if time_scale is not None
        else {str(phase): float(value) for phase, value in phase_time_scales.items()}
    )
    rate_limit = float(joint_rate_limit_rad_s)
    if (
        set(scales) != set(_PHASES)
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in scales.values())
        or not math.isfinite(rate_limit)
        or rate_limit <= 0.0
    ):
        raise SchemaValidationError("R1 nominal retime inputs are invalid")

    source_artifact_sha256 = hash_file(source)
    root = source.parent
    retimed: dict[str, ContactWrenchTrajectory] = {}
    phase_entries: list[Order9C3NominalPhaseArtifact] = []
    maximum_joint_rate = 0.0
    maximum_centroidal_speed = 0.0
    maximum_object_speed = 0.0
    endpoint_unchanged = True
    contacts_unchanged = True

    for phase_entry in artifact.phase_trajectories:
        scale = scales[phase_entry.phase]
        path = root / phase_entry.trajectory_path
        original = ContactWrenchTrajectory.from_json(path.read_text(encoding="utf-8"))
        original.validate()
        value = _retime_trajectory(original, scale=scale)
        endpoint_unchanged &= _semantically_close(
            _configuration_endpoint(original),
            _configuration_endpoint(value),
        )
        contacts_unchanged &= _contact_assignments_match_resampling(
            original,
            value,
            scale=scale,
        )
        phase_maximum_joint_rate, phase_centroidal_speed, phase_object_speed = (
            _maximum_commanded_speeds(value)
        )
        maximum_joint_rate = max(maximum_joint_rate, phase_maximum_joint_rate)
        maximum_centroidal_speed = max(
            maximum_centroidal_speed, phase_centroidal_speed
        )
        maximum_object_speed = max(maximum_object_speed, phase_object_speed)
        if phase_maximum_joint_rate > rate_limit + 1.0e-9:
            raise SchemaValidationError(
                "R1 nominal retime exceeds screened joint-rate limit: "
                f"{phase_entry.phase} {phase_maximum_joint_rate:.9g} > "
                f"{rate_limit:.9g} rad/s"
            )
        path.write_text(value.to_json(indent=2) + "\n", encoding="utf-8")
        retimed[phase_entry.phase] = value
        phase_entries.append(
            Order9C3NominalPhaseArtifact(
                phase=phase_entry.phase,
                trajectory_path=phase_entry.trajectory_path,
                trajectory_sha256=hash_file(path),
                trajectory_hash=stable_hash(value.to_dict()),
                knot_count=len(value.knots),
                generation_method=(
                    f"{phase_entry.generation_method}+"
                    f"{ORDER9_R1_NOMINAL_RETIME_VERSION}"
                ),
                collision_validation_status="accepted_offline_admission",
            )
        )

    if tuple(retimed) != _PHASES:
        raise SchemaValidationError("R1 nominal retime phase order changed")
    if not endpoint_unchanged or not contacts_unchanged:
        raise SchemaValidationError("R1 nominal retime changed path semantics")

    duration_s = sum(float(retimed[phase].horizon_s) for phase in _PHASES)
    timeline_path = root / artifact.timeline_path
    timeline_path.write_text(
        json.dumps(
            {
                "timeline_version": ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
                "semantic_scope": (
                    "offline precomputed complete eight-phase nominal; geometric "
                    "path preserved by uniform time reparameterization"
                ),
                "proxy_collision_validation_status": "enforced_during_generation",
                "duration_s": duration_s,
                "records": _complete_timeline(retimed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = dict(artifact.selection_evidence)
    evidence["r1_uniform_nominal_retime"] = {
        "admission_version": ORDER9_R1_NOMINAL_RETIME_VERSION,
        "phase_time_scales": scales,
        "source_artifact_sha256": source_artifact_sha256,
        "source_duration_s": float(artifact.duration_s),
        "retimed_duration_s": duration_s,
        "phase_order_unchanged": True,
        "geometric_path_unchanged": True,
        "endpoint_configuration_unchanged": True,
        "contact_assignments_unchanged": True,
        "joint_rate_limit_rad_s": rate_limit,
        "maximum_commanded_joint_rate_rad_s": maximum_joint_rate,
        "collision_admission_inherited_from_identical_geometry": True,
        "runtime_planner_enabled": False,
    }
    payload = artifact.to_dict()
    payload.update(
        {
            "timeline_sha256": hash_file(timeline_path),
            "duration_s": duration_s,
            "phase_trajectories": [value.to_dict() for value in phase_entries],
            "proxy_collision_validation_status": "enforced_during_generation",
            "selection_evidence": evidence,
        }
    )
    updated = Order9C3NominalTrajectoryArtifact.from_dict(payload)
    updated.validate()
    source.write_text(updated.to_json(indent=2) + "\n", encoding="utf-8")
    validate_order9_c3_nominal_trajectory_artifact_bytes(source)

    admission = Order9R1NominalRetimeAdmission(
        admission_version=ORDER9_R1_NOMINAL_RETIME_VERSION,
        time_scale=min(scales.values()),
        phase_time_scales=scales,
        source_artifact_sha256=source_artifact_sha256,
        retimed_artifact_sha256=hash_file(source),
        source_duration_s=float(artifact.duration_s),
        retimed_duration_s=duration_s,
        joint_rate_limit_rad_s=rate_limit,
        maximum_commanded_joint_rate_rad_s=maximum_joint_rate,
        minimum_joint_rate_margin_rad_s=rate_limit - maximum_joint_rate,
        maximum_centroidal_speed_m_s=maximum_centroidal_speed,
        maximum_object_linear_speed_m_s=maximum_object_speed,
        phase_order_unchanged=True,
        geometric_path_unchanged=True,
        endpoint_configuration_unchanged=True,
        contact_assignments_unchanged=True,
        collision_admission_inherited_from_identical_geometry=True,
        accepted=True,
    )
    admission.validate()
    return admission


def _retime_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    scale: float,
) -> ContactWrenchTrajectory:
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.0)
    executor.install(source, plan_start_time_s=0.0)
    horizon = float(source.horizon_s) * scale
    dt_s = float(source.dt_s)
    sample_count = int(horizon / dt_s + 1.0e-9)
    times = [index * dt_s for index in range(sample_count + 1)]
    if not times or horizon - times[-1] > 1.0e-9:
        times.append(horizon)
    else:
        times[-1] = horizon
    knots: list[InteractionKnot] = []
    for output_time in times:
        source_time = min(output_time / scale, float(source.horizon_s))
        knot = executor.sample(time_s=source_time).active_knot
        knot.t_rel_s = output_time
        _scale_velocity_targets(knot, scale=scale)
        knots.append(knot)
    value = ContactWrenchTrajectory(
        horizon_s=horizon,
        dt_s=dt_s,
        knots=knots,
        derived_mode_label=source.derived_mode_label,
        contract_version=source.contract_version,
    )
    value.validate()
    return value


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
                float(value) / scale for value in target.generalized_qdot_target
            ]


def _configuration_endpoint(trajectory: ContactWrenchTrajectory) -> dict[str, object]:
    knot = trajectory.knots[-1]
    posture = knot.posture_target
    centroidal = knot.centroidal_target
    return {
        "joint_position": (
            None if posture is None else posture.joint_pos_target
        ),
        "free_anchor_pose": (
            None if posture is None else posture.free_anchor_pose_targets
        ),
        "centroidal": (
            None
            if centroidal is None
            else {
                "com_position": centroidal.com_pos_world,
                "body_orientation": centroidal.body_orientation_world,
            }
        ),
        "objects": [
            {
                "object_id": value.object_id,
                "pose": value.pose_target_world,
                "generalized_q": value.generalized_q_target,
            }
            for value in knot.object_targets
        ],
        "contacts": [value.to_dict() for value in knot.contact_assignments],
    }


def _contact_assignments_match_resampling(
    source: ContactWrenchTrajectory,
    retimed: ContactWrenchTrajectory,
    *,
    scale: float,
) -> bool:
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.0)
    executor.install(source, plan_start_time_s=0.0)
    for knot in retimed.knots:
        source_time = min(float(knot.t_rel_s) / scale, float(source.horizon_s))
        expected = executor.sample(time_s=source_time).active_knot
        if [value.to_dict() for value in knot.contact_assignments] != [
            value.to_dict() for value in expected.contact_assignments
        ]:
            return False
    return True


def _maximum_commanded_speeds(
    trajectory: ContactWrenchTrajectory,
) -> tuple[float, float, float]:
    joint = 0.0
    centroidal = 0.0
    object_linear = 0.0
    for knot in trajectory.knots:
        posture = knot.posture_target
        if posture is not None and posture.joint_vel_target is not None:
            joint = max(
                joint,
                *(abs(float(value)) for value in posture.joint_vel_target.values()),
            )
        target = knot.centroidal_target
        if target is not None and target.com_vel_world is not None:
            centroidal = max(
                centroidal,
                math.sqrt(sum(float(value) ** 2 for value in target.com_vel_world)),
            )
        for object_target in knot.object_targets:
            twist = object_target.twist_target_world
            if twist is not None and len(twist) >= 3:
                object_linear = max(
                    object_linear,
                    math.sqrt(sum(float(value) ** 2 for value in twist[:3])),
                )
    return joint, centroidal, object_linear


def _semantically_close(left: object, right: object, *, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return bool(
            isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
        )
    if isinstance(left, dict):
        return bool(
            isinstance(right, dict)
            and set(left) == set(right)
            and all(
                _semantically_close(left[key], right[key], tolerance=tolerance)
                for key in left
            )
        )
    if isinstance(left, (list, tuple)):
        return bool(
            isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _semantically_close(a, b, tolerance=tolerance)
                for a, b in zip(left, right)
            )
        )
    return left == right


def _complete_timeline(
    phases: dict[str, ContactWrenchTrajectory],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    offset_s = 0.0
    for phase in _PHASES:
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


__all__ = [
    "ORDER9_R1_NOMINAL_RETIME_VERSION",
    "Order9R1NominalRetimeAdmission",
    "retime_order9_r1_nominal_artifact",
]
