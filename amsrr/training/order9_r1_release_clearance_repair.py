from __future__ import annotations

"""Raise the R1 place/release edge while preserving the accepted grasp path.

The object is released a small distance above its final support pose and is
allowed to settle under gravity.  Robot and object references receive the same
rigid vertical translation, so the grasp geometry and joint path do not change.
"""

from dataclasses import replace
import math
from typing import Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskPhase

ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION = (
    "order9_r1_rigid_release_height_clearance_v1"
)
ORDER9_R1_MAXIMUM_RELEASE_HEIGHT_OFFSET_M = 0.03


def _smoothstep(value: float) -> tuple[float, float]:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value), 6.0 * value * (1.0 - value)


def _shift_pose_z(pose: tuple[float, ...], offset_m: float) -> tuple[float, ...]:
    values = list(pose)
    values[2] = float(values[2]) + offset_m
    return tuple(values)


def _shift_knot(
    knot: InteractionKnot,
    *,
    offset_m: float,
    vertical_velocity_m_s: float,
) -> InteractionKnot:
    centroidal = knot.centroidal_target
    if centroidal is not None:
        com_position = centroidal.com_pos_world
        com_velocity = centroidal.com_vel_world
        centroidal = replace(
            centroidal,
            com_pos_world=(
                None
                if com_position is None
                else _shift_pose_z(tuple(com_position), offset_m)[:3]
            ),
            com_vel_world=(
                None
                if com_velocity is None
                else (
                    float(com_velocity[0]),
                    float(com_velocity[1]),
                    float(com_velocity[2]) + vertical_velocity_m_s,
                )
            ),
        )
    posture = knot.posture_target
    if posture is not None and posture.free_anchor_pose_targets is not None:
        posture = replace(
            posture,
            free_anchor_pose_targets={
                int(anchor_id): _shift_pose_z(tuple(pose), offset_m)
                for anchor_id, pose in posture.free_anchor_pose_targets.items()
            },
        )
    object_targets = []
    for target in knot.object_targets:
        pose = target.pose_target_world
        twist = target.twist_target_world
        object_targets.append(
            replace(
                target,
                pose_target_world=(
                    None if pose is None else _shift_pose_z(tuple(pose), offset_m)
                ),
                twist_target_world=(
                    None
                    if twist is None
                    else [
                        float(twist[0]),
                        float(twist[1]),
                        float(twist[2]) + vertical_velocity_m_s,
                        *[float(value) for value in twist[3:]],
                    ]
                ),
            )
        )
    return replace(
        knot,
        centroidal_target=centroidal,
        posture_target=posture,
        object_targets=object_targets,
    )


def apply_order9_r1_release_clearance_repair(
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    *,
    height_offset_m: float,
) -> dict[str, ContactWrenchTrajectory]:
    """Apply a smooth rigid rise in place and hold it through settle."""

    offset = float(height_offset_m)
    if (
        not math.isfinite(offset)
        or offset <= 0.0
        or offset > ORDER9_R1_MAXIMUM_RELEASE_HEIGHT_OFFSET_M + 1.0e-12
    ):
        raise SchemaValidationError("R1 release height offset must be in (0, 0.03] m")
    expected = tuple(value.value for value in Order9ObjectTaskPhase)
    if tuple(phase_trajectories) != expected:
        raise SchemaValidationError("R1 release-clearance phase order differs")
    repaired = dict(phase_trajectories)
    place_name = Order9ObjectTaskPhase.PLACE.value
    place = phase_trajectories[place_name]
    place_knots = []
    for knot in place.knots:
        progress = float(knot.t_rel_s) / float(place.horizon_s)
        scale, derivative = _smoothstep(progress)
        place_knots.append(
            _shift_knot(
                knot,
                offset_m=offset * scale,
                vertical_velocity_m_s=offset * derivative / float(place.horizon_s),
            )
        )
    repaired[place_name] = replace(
        place,
        knots=place_knots,
        derived_mode_label=(f"{ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION}:place"),
    )
    for phase in (
        Order9ObjectTaskPhase.RELEASE,
        Order9ObjectTaskPhase.RETREAT,
        Order9ObjectTaskPhase.SETTLE,
    ):
        source = phase_trajectories[phase.value]
        repaired[phase.value] = replace(
            source,
            knots=[
                _shift_knot(knot, offset_m=offset, vertical_velocity_m_s=0.0)
                for knot in source.knots
            ],
            derived_mode_label=(
                f"{ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION}:{phase.value}"
            ),
        )
    for trajectory in repaired.values():
        trajectory.validate()
    return repaired


__all__ = [
    "ORDER9_R1_MAXIMUM_RELEASE_HEIGHT_OFFSET_M",
    "ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION",
    "apply_order9_r1_release_clearance_repair",
]
