from __future__ import annotations

"""R1-only repair for the nominal retreat edge.

The shared C3 complete-task materializer accepts ``retreat_offset_m`` but its
historical implementation discards that value and reverses the entire approach
path.  That can command metre-scale post-release travel where the task runtime
requires only a short separation.  R1 replaces only retreat and settle after
the protected grasp/release construction has completed.
"""

import math
from collections.abc import Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskPhase
from amsrr.training.order9_c3_nominal_trajectory import (
    _build_complete_task_phase_trajectory,
    _complete_task_knot_state,
    materialize_order9_c3_complete_task_phases as _base_materialize_complete_task,
)

ORDER9_R1_NOMINAL_RETREAT_VERSION = "order9_r1_short_task_offset_retreat_v1"


def materialize_order9_r1_complete_task_phases(
    *,
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    lift_clearance_m: float,
    retreat_offset_m: float,
    phase_duration_s: Mapping[str, float] | None = None,
    nominal_dt_s: float = 0.1,
) -> dict[str, ContactWrenchTrajectory]:
    """Keep the accepted task path, replacing only overlong retreat/settle."""

    offset = float(retreat_offset_m)
    if not math.isfinite(offset) or offset <= 0.0:
        raise SchemaValidationError("R1 retreat offset must be positive")
    phases = _base_materialize_complete_task(
        phase_trajectories=phase_trajectories,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=offset,
        phase_duration_s=phase_duration_s,
        nominal_dt_s=nominal_dt_s,
    )
    release = phases[Order9ObjectTaskPhase.RELEASE.value]
    old_retreat = phases[Order9ObjectTaskPhase.RETREAT.value]
    old_settle = phases[Order9ObjectTaskPhase.SETTLE.value]
    release_end = release.knots[-1]
    body_start, _, q_open, _ = _complete_task_knot_state(release_end)
    body_end = (
        float(body_start[0]) - offset,
        *[float(value) for value in body_start[1:]],
    )
    objects = [
        target
        for target in release_end.object_targets
        if target.pose_target_world is not None
    ]
    if len(objects) != 1:
        raise SchemaValidationError("R1 retreat requires one released object target")
    object_target = objects[0]
    object_pose = tuple(float(value) for value in object_target.pose_target_world)
    retreat = _build_complete_task_phase_trajectory(
        template=release_end,
        phase=Order9ObjectTaskPhase.RETREAT.value,
        duration_s=float(old_retreat.horizon_s),
        dt_s=float(nominal_dt_s),
        body_start=body_start,
        body_end=body_end,
        q_start=q_open,
        q_end=q_open,
        object_id=object_target.object_id,
        object_start=object_pose,
        object_end=object_pose,
        contact_schedule_state="inactive",
        anchor_pose_translation_origin=object_pose,
        contract_version=release.contract_version,
    )
    retreat.derived_mode_label = f"{ORDER9_R1_NOMINAL_RETREAT_VERSION}:retreat"
    settle = _build_complete_task_phase_trajectory(
        template=retreat.knots[-1],
        phase=Order9ObjectTaskPhase.SETTLE.value,
        duration_s=float(old_settle.horizon_s),
        dt_s=float(nominal_dt_s),
        body_start=body_end,
        body_end=body_end,
        q_start=q_open,
        q_end=q_open,
        object_id=object_target.object_id,
        object_start=object_pose,
        object_end=object_pose,
        contact_schedule_state="inactive",
        anchor_pose_translation_origin=object_pose,
        contract_version=release.contract_version,
    )
    settle.derived_mode_label = f"{ORDER9_R1_NOMINAL_RETREAT_VERSION}:settle"
    retreat.validate()
    settle.validate()
    phases[Order9ObjectTaskPhase.RETREAT.value] = retreat
    phases[Order9ObjectTaskPhase.SETTLE.value] = settle
    validate_order9_r1_short_retreat(
        phases,
        retreat_offset_m=offset,
    )
    return phases


def validate_order9_r1_short_retreat(
    phases: Mapping[str, ContactWrenchTrajectory],
    *,
    retreat_offset_m: float,
) -> None:
    """Check release continuity, exact task offset, and stationary settle."""

    release = phases[Order9ObjectTaskPhase.RELEASE.value]
    retreat = phases[Order9ObjectTaskPhase.RETREAT.value]
    settle = phases[Order9ObjectTaskPhase.SETTLE.value]
    release_end, _, release_q, _ = _complete_task_knot_state(release.knots[-1])
    retreat_start, _, retreat_start_q, _ = _complete_task_knot_state(retreat.knots[0])
    retreat_end, _, retreat_end_q, _ = _complete_task_knot_state(retreat.knots[-1])
    settle_start, _, settle_start_q, _ = _complete_task_knot_state(settle.knots[0])
    settle_end, _, settle_end_q, _ = _complete_task_knot_state(settle.knots[-1])
    offset = float(retreat_offset_m)
    if (
        any(
            abs(float(a) - float(b)) > 1.0e-9
            for a, b in zip(release_end, retreat_start)
        )
        or release_q != retreat_start_q
        or any(
            abs(float(a) - float(b)) > 1.0e-9 for a, b in zip(retreat_end, settle_start)
        )
        or retreat_end_q != settle_start_q
        or any(
            abs(float(a) - float(b)) > 1.0e-9 for a, b in zip(settle_start, settle_end)
        )
        or settle_start_q != settle_end_q
        or abs((float(retreat_start[0]) - float(retreat_end[0])) - offset) > 1.0e-9
        or any(
            abs(float(retreat_start[index]) - float(retreat_end[index])) > 1.0e-9
            for index in range(1, 7)
        )
    ):
        raise SchemaValidationError("R1 short retreat semantics differ")
    moving = [
        knot
        for knot in retreat.knots[1:-1]
        if knot.centroidal_target is not None
        and knot.centroidal_target.com_vel_world is not None
    ]
    if not moving or not any(
        abs(float(knot.centroidal_target.com_vel_world[0])) > 1.0e-12 for knot in moving
    ):
        raise SchemaValidationError("R1 short retreat lacks velocity feedforward")


__all__ = [
    "ORDER9_R1_NOMINAL_RETREAT_VERSION",
    "materialize_order9_r1_complete_task_phases",
    "validate_order9_r1_short_retreat",
]
