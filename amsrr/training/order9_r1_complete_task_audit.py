from __future__ import annotations

"""Cheap, fail-closed semantic audit for a materialized R1 complete task."""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    _complete_task_knot_state,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION = "order9_r1_complete_task_semantic_audit_v1"
ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION = (
    "order9_r1_complete_task_parameter_effect_v1"
)
ORDER9_R1_COMPLETE_TASK_PHASES = tuple(
    phase.value for phase in ORDER9_OBJECT_TASK_PHASES
)

E_PHASE_ORDER = "E_R1_COMPLETE_TASK_PHASE_ORDER"
E_PHASE_CONTINUITY = "E_R1_COMPLETE_TASK_PHASE_CONTINUITY"
E_OBJECT_SEMANTICS = "E_R1_COMPLETE_TASK_OBJECT_SEMANTICS"
E_CONTACT_SEMANTICS = "E_R1_COMPLETE_TASK_CONTACT_SEMANTICS"
E_CARRY_RIGIDITY = "E_R1_COMPLETE_TASK_CARRY_RIGIDITY"
E_RETREAT_OFFSET = "E_R1_COMPLETE_TASK_RETREAT_OFFSET"
E_RETREAT_PATH = "E_R1_COMPLETE_TASK_RETREAT_PATH"
E_RETREAT_VELOCITY = "E_R1_COMPLETE_TASK_RETREAT_VELOCITY"
E_SETTLE = "E_R1_COMPLETE_TASK_SETTLE"
E_PARAMETER_EFFECT = "E_R1_COMPLETE_TASK_PARAMETER_EFFECT"

CompleteTaskMaterializer = Callable[..., Mapping[str, ContactWrenchTrajectory]]


def audit_order9_r1_complete_task_artifact(
    manifest_path: str | Path,
    *,
    task_spec: TaskSpec,
    lift_clearance_m: float,
    retreat_offset_m: float,
    materializer_id: str,
) -> dict[str, Any]:
    """Load a persisted artifact and audit all eight phase trajectories."""

    source = Path(manifest_path).resolve()
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(source)
    phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (source.parent / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }
    result = audit_order9_r1_complete_task_semantics(
        phases,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=retreat_offset_m,
        materializer_id=materializer_id,
    )
    result["artifact"] = {
        "path": str(source),
        "sha256": hash_file(source),
        "semantic_hash": stable_hash(
            {phase: phases[phase].to_dict() for phase in ORDER9_R1_COMPLETE_TASK_PHASES}
        ),
    }
    return result


def audit_order9_r1_complete_task_semantics(
    phases: Mapping[str, ContactWrenchTrajectory],
    *,
    task_spec: TaskSpec,
    lift_clearance_m: float,
    retreat_offset_m: float,
    materializer_id: str,
) -> dict[str, Any]:
    """Reject task-semantic errors without controllers, solvers, or Isaac."""

    lift = _positive_finite(lift_clearance_m, "lift clearance")
    retreat_offset = _positive_finite(retreat_offset_m, "retreat offset")
    if not materializer_id:
        raise SchemaValidationError("R1 complete-task materializer id is empty")
    if tuple(phases) != ORDER9_R1_COMPLETE_TASK_PHASES:
        _fail(E_PHASE_ORDER, "complete-task phase order differs")
    for trajectory in phases.values():
        trajectory.validate()

    states = {
        phase: (_state(trajectory.knots[0]), _state(trajectory.knots[-1]))
        for phase, trajectory in phases.items()
    }
    for left, right in zip(
        ORDER9_R1_COMPLETE_TASK_PHASES,
        ORDER9_R1_COMPLETE_TASK_PHASES[1:],
    ):
        if not _states_continuous(states[left][1], states[right][0]):
            _fail(E_PHASE_CONTINUITY, f"{left}->{right} is discontinuous")

    object_id, object_start, object_goal = _task_object_transition(task_spec)
    object_lifted = _translated_pose(object_start, z=lift)
    object_goal_lifted = _translated_pose(object_goal, z=lift)
    expected_object_edges = {
        "approach": (object_start, object_start),
        "contact_acquisition": (object_start, object_start),
        "lift": (object_start, object_lifted),
        "transport": (object_lifted, object_goal_lifted),
        "place": (object_goal_lifted, object_goal),
        "release": (object_goal, object_goal),
        "retreat": (object_goal, object_goal),
        "settle": (object_goal, object_goal),
    }
    for phase, (expected_start, expected_end) in expected_object_edges.items():
        trajectory = phases[phase]
        for knot in trajectory.knots:
            knot_object_id, _pose = _object_state(knot)
            if knot_object_id != object_id:
                _fail(E_OBJECT_SEMANTICS, f"{phase} object identity changed")
        if not _pose_close(states[phase][0][2], expected_start) or not _pose_close(
            states[phase][1][2], expected_end
        ):
            _fail(E_OBJECT_SEMANTICS, f"{phase} object endpoints differ")

    _validate_contact_semantics(phases)
    for phase in ("lift", "transport", "place"):
        start, end = states[phase]
        relative_start = compose_pose(inverse_pose(start[2]), start[0])
        relative_end = compose_pose(inverse_pose(end[2]), end[0])
        if not _pose_close(relative_start, relative_end, tolerance=1.0e-7):
            _fail(E_CARRY_RIGIDITY, f"{phase} changed the grasp transform")

    retreat = phases["retreat"]
    retreat_start = states["retreat"][0]
    retreat_end = states["retreat"][1]
    expected_retreat_end = _translated_pose(retreat_start[0], x=-retreat_offset)
    if not _pose_close(retreat_end[0], expected_retreat_end):
        _fail(E_RETREAT_OFFSET, "retreat endpoint does not use the requested offset")
    retreat_path_length = _body_translation_path_length(retreat)
    if (
        retreat_path_length < retreat_offset - 1.0e-9
        or retreat_path_length > retreat_offset + 1.0e-7
    ):
        _fail(E_RETREAT_PATH, "retreat contains extra travel")
    if any(knot.contact_assignments for knot in retreat.knots):
        _fail(E_CONTACT_SEMANTICS, "retreat retained a contact assignment")
    if not any(
        abs(float(value)) > 1.0e-12
        for knot in retreat.knots[1:-1]
        for value in (knot.centroidal_target.com_vel_world or ())
    ):
        _fail(E_RETREAT_VELOCITY, "moving retreat has no velocity target")

    settle = phases["settle"]
    settle_reference = _state(settle.knots[0])
    if any(
        not _states_continuous(settle_reference, _state(knot)) for knot in settle.knots
    ):
        _fail(E_SETTLE, "settle is not stationary")
    if any(
        abs(float(value)) > 1.0e-12
        for knot in settle.knots
        for value in (
            *(knot.centroidal_target.com_vel_world or ()),
            *(knot.posture_target.joint_vel_target or {}).values(),
            *(knot.object_targets[0].twist_target_world or ()),
        )
    ):
        _fail(E_SETTLE, "settle has a nonzero velocity target")

    return {
        "audit_version": ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION,
        "status": "accepted",
        "materializer_id": materializer_id,
        "checked_phases": list(ORDER9_R1_COMPLETE_TASK_PHASES),
        "checked_knot_count": sum(len(value.knots) for value in phases.values()),
        "phase_boundary_count": len(ORDER9_R1_COMPLETE_TASK_PHASES) - 1,
        "retreat_offset_m": retreat_offset,
        "measured_retreat_path_length_m": retreat_path_length,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "ik_resolve_invoked": False,
        "trajectory_optimization_invoked": False,
        "training_eligible": False,
    }


def audit_order9_r1_materializer_parameter_effect(
    materializer: CompleteTaskMaterializer,
    *,
    materializer_id: str,
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    lift_clearance_m: float,
    retreat_offsets_m: Sequence[float] = (0.05, 0.10),
    phase_duration_s: Mapping[str, float] | None = None,
    nominal_dt_s: float = 0.1,
) -> dict[str, Any]:
    """Prove that a critical input changes only its declared output edge."""

    offsets = tuple(float(value) for value in retreat_offsets_m)
    if len(offsets) != 2 or offsets[0] == offsets[1]:
        raise SchemaValidationError("R1 parameter-effect audit needs two offsets")
    outputs = []
    for offset in offsets:
        phases = dict(
            materializer(
                phase_trajectories=phase_trajectories,
                task_spec=task_spec,
                lift_clearance_m=lift_clearance_m,
                retreat_offset_m=offset,
                phase_duration_s=phase_duration_s,
                nominal_dt_s=nominal_dt_s,
            )
        )
        audit_order9_r1_complete_task_semantics(
            phases,
            task_spec=task_spec,
            lift_clearance_m=lift_clearance_m,
            retreat_offset_m=offset,
            materializer_id=materializer_id,
        )
        outputs.append(phases)
    for phase in ORDER9_R1_COMPLETE_TASK_PHASES[:6]:
        if outputs[0][phase].to_dict() != outputs[1][phase].to_dict():
            _fail(E_PARAMETER_EFFECT, f"retreat offset changed {phase}")
    first_end = _state(outputs[0]["retreat"].knots[-1])[0]
    second_end = _state(outputs[1]["retreat"].knots[-1])[0]
    measured_effect = first_end[0] - second_end[0]
    expected_effect = offsets[1] - offsets[0]
    if not math.isclose(measured_effect, expected_effect, rel_tol=0.0, abs_tol=1.0e-9):
        _fail(E_PARAMETER_EFFECT, "retreat output is insensitive to its input")
    return {
        "audit_version": ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION,
        "status": "accepted",
        "materializer_id": materializer_id,
        "retreat_offsets_m": list(offsets),
        "measured_endpoint_effect_m": measured_effect,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "trajectory_optimization_invoked": False,
    }


def write_order9_r1_complete_task_audit(
    payload: Mapping[str, Any], path: str | Path
) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _state(
    knot: InteractionKnot,
) -> tuple[tuple[float, ...], dict[str, float], tuple[float, ...]]:
    body, _twist, joints, _joint_velocities = _complete_task_knot_state(knot)
    _object_id, object_pose = _object_state(knot)
    return body, joints, object_pose


def _object_state(knot: InteractionKnot) -> tuple[str, tuple[float, ...]]:
    targets = [
        value for value in knot.object_targets if value.pose_target_world is not None
    ]
    if len(targets) != 1:
        _fail(E_OBJECT_SEMANTICS, "knot does not contain exactly one object pose")
    return targets[0].object_id, tuple(
        float(value) for value in targets[0].pose_target_world
    )


def _task_object_transition(
    task_spec: TaskSpec,
) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    goals = [
        goal
        for goal in task_spec.goals
        if goal.goal_type == "object_pose"
        and goal.target_entity_id is not None
        and goal.target_pose_world is not None
    ]
    if len(goals) != 1:
        _fail(E_OBJECT_SEMANTICS, "task does not contain one object-pose goal")
    goal = goals[0]
    objects = [
        value
        for value in task_spec.scene.objects
        if value.object_id == goal.target_entity_id
    ]
    if len(objects) != 1:
        _fail(E_OBJECT_SEMANTICS, "task object-pose goal has no unique scene object")
    return (
        objects[0].object_id,
        tuple(float(value) for value in objects[0].pose_world),
        tuple(float(value) for value in goal.target_pose_world),
    )


def _validate_contact_semantics(phases: Mapping[str, ContactWrenchTrajectory]) -> None:
    allowed = {
        "approach": {"approach"},
        "contact_acquisition": {"attach", "maintain"},
        "lift": {"maintain"},
        "transport": {"maintain"},
        "place": {"maintain"},
        "release": {"release"},
        "retreat": set(),
        "settle": set(),
    }
    for phase, trajectory in phases.items():
        for knot in trajectory.knots:
            states = {
                assignment.schedule_state for assignment in knot.contact_assignments
            }
            if states - allowed[phase]:
                _fail(E_CONTACT_SEMANTICS, f"{phase} has an invalid contact state")
            if phase in {"lift", "transport", "place", "release"} and not states:
                _fail(E_CONTACT_SEMANTICS, f"{phase} lost all contact assignments")


def _states_continuous(
    left: tuple[tuple[float, ...], dict[str, float], tuple[float, ...]],
    right: tuple[tuple[float, ...], dict[str, float], tuple[float, ...]],
) -> bool:
    return (
        _pose_close(left[0], right[0])
        and left[1].keys() == right[1].keys()
        and all(
            math.isclose(left[1][key], right[1][key], rel_tol=0.0, abs_tol=1.0e-8)
            for key in left[1]
        )
        and _pose_close(left[2], right[2])
    )


def _pose_close(
    left: Sequence[float], right: Sequence[float], *, tolerance: float = 1.0e-8
) -> bool:
    if len(left) != 7 or len(right) != 7:
        return False
    if any(abs(float(a) - float(b)) > tolerance for a, b in zip(left[:3], right[:3])):
        return False
    dot = abs(sum(float(a) * float(b) for a, b in zip(left[3:7], right[3:7])))
    return abs(dot - 1.0) <= tolerance


def _translated_pose(
    pose: Sequence[float], *, x: float = 0.0, z: float = 0.0
) -> tuple[float, ...]:
    return (
        float(pose[0]) + x,
        float(pose[1]),
        float(pose[2]) + z,
        *[float(value) for value in pose[3:7]],
    )


def _body_translation_path_length(trajectory: ContactWrenchTrajectory) -> float:
    positions = [_state(knot)[0][:3] for knot in trajectory.knots]
    return sum(
        math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(left, right)))
        for left, right in zip(positions, positions[1:])
    )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SchemaValidationError(f"R1 complete-task {name} is invalid")
    return result


def _fail(code: str, detail: str) -> None:
    raise SchemaValidationError(f"{code}: {detail}")


__all__ = [
    "CompleteTaskMaterializer",
    "E_PARAMETER_EFFECT",
    "E_RETREAT_OFFSET",
    "ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION",
    "ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION",
    "ORDER9_R1_COMPLETE_TASK_PHASES",
    "audit_order9_r1_complete_task_artifact",
    "audit_order9_r1_complete_task_semantics",
    "audit_order9_r1_materializer_parameter_effect",
    "write_order9_r1_complete_task_audit",
]
