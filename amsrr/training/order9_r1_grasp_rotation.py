from __future__ import annotations

"""Cheap R1 grasp-rotation admission before closed-loop Isaac execution."""

import math
from collections.abc import Sequence
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec

ORDER9_R1_GRASP_ROTATION_AUDIT_VERSION = (
    "order9_r1_grasp_rotation_force_moment_audit_v1"
)


def audit_order9_r1_grasp_rotation_margin(
    *,
    task_spec: TaskSpec,
    contact_candidate_set: ContactCandidateSet,
    trajectory: ContactWrenchTrajectory,
    selected_candidate_group_id: str | None,
    required_rotation_axis_object: Sequence[float],
    minimum_force_moment_arm_m: float,
) -> dict[str, Any]:
    """Check that contact forces have an arm about one observed slip axis.

    This is deliberately a point-force, conservative screen.  It does not
    claim dynamic success and does not replace Isaac.  It rejects the known
    failure mode where a two-sided pinch has no force-generated moment arm
    about the axis around which the payload slipped.
    """

    axis_object = _unit_vector(required_rotation_axis_object, "rotation axis")
    if not math.isfinite(float(minimum_force_moment_arm_m)) or not (
        minimum_force_moment_arm_m > 0.0
    ):
        raise SchemaValidationError("R1 grasp moment-arm threshold must be positive")
    target_ids = {
        goal.target_entity_id
        for goal in task_spec.goals
        if goal.goal_type == "object_pose" and goal.target_entity_id is not None
    }
    objects = [
        value for value in task_spec.scene.objects if value.object_id in target_ids
    ]
    if len(objects) != 1:
        raise SchemaValidationError("R1 grasp audit requires one target object")
    target = objects[0]
    axis_world = _rotate_vector(target.pose_world[3:7], axis_object)
    com_object = target.center_of_mass_object or (0.0, 0.0, 0.0)
    com_offset_world = _rotate_vector(target.pose_world[3:7], com_object)
    com_world = tuple(
        float(target.pose_world[index]) + com_offset_world[index] for index in range(3)
    )

    assignments = _active_grasp_assignments(trajectory)
    if len(assignments) != 2:
        raise SchemaValidationError("R1 grasp rotation audit requires two contacts")
    candidate_by_id = {
        value.candidate_id: value for value in contact_candidate_set.candidates
    }
    group = next(
        (
            value
            for value in contact_candidate_set.group_proposals
            if value.group_id == selected_candidate_group_id
        ),
        None,
    )
    assignment_ids = {value.candidate_id for value in assignments}
    if group is None or set(group.candidate_ids) != assignment_ids:
        raise SchemaValidationError("R1 grasp audit candidate group differs")

    rows = []
    for assignment in assignments:
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is None:
            raise SchemaValidationError("R1 grasp audit candidate is missing")
        if candidate.friction is None or not float(candidate.friction) > 0.0:
            raise SchemaValidationError("R1 grasp audit friction is missing")
        arm = _force_moment_arm_about_axis(
            contact_position_world=candidate.contact_pose_world[:3],
            com_world=com_world,
            tangent_basis_world=candidate.tangent_basis_world,
            axis_world=axis_world,
        )
        nominal_normal_force_n = _nominal_normal_force(assignment)
        tangential_capacity_n = float(candidate.friction) * nominal_normal_force_n
        rows.append(
            {
                "anchor_id": int(assignment.anchor_id),
                "candidate_id": int(assignment.candidate_id),
                "region_id": candidate.region_id,
                "friction": float(candidate.friction),
                "nominal_normal_force_n": nominal_normal_force_n,
                "tangential_force_capacity_n": tangential_capacity_n,
                "force_moment_arm_m": arm,
                "predicted_torque_capacity_nm": arm * tangential_capacity_n,
            }
        )
    minimum_arm = min(float(value["force_moment_arm_m"]) for value in rows)
    mass_kg = float(target.mass_kg or 0.0)
    gravity = _norm(task_spec.scene.environment.gravity)
    total_tangential_capacity = sum(
        float(value["tangential_force_capacity_n"]) for value in rows
    )
    gravity_support_margin_n = total_tangential_capacity - mass_kg * gravity
    accepted = (
        minimum_arm + 1.0e-12 >= float(minimum_force_moment_arm_m)
        and gravity_support_margin_n >= 0.0
    )
    violations = []
    if minimum_arm + 1.0e-12 < float(minimum_force_moment_arm_m):
        violations.append("E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM")
    if gravity_support_margin_n < 0.0:
        violations.append("E_R1_GRASP_ROTATION_FRICTION_SUPPORT")
    return {
        "audit_version": ORDER9_R1_GRASP_ROTATION_AUDIT_VERSION,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "candidate_group_id": selected_candidate_group_id,
        "required_rotation_axis_object": list(axis_object),
        "required_rotation_axis_world": list(axis_world),
        "minimum_force_moment_arm_required_m": float(minimum_force_moment_arm_m),
        "minimum_force_moment_arm_observed_m": minimum_arm,
        "gravity_support_margin_n": gravity_support_margin_n,
        "contacts": rows,
        "violation_codes": violations,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "dynamic_success_claimed": False,
    }


def _active_grasp_assignments(trajectory: ContactWrenchTrajectory):
    for schedule_state in ("maintain", "attach"):
        for knot in trajectory.knots:
            assignments = tuple(
                value
                for value in knot.contact_assignments
                if value.schedule_state == schedule_state
            )
            if assignments:
                return assignments
    raise SchemaValidationError("R1 grasp audit lacks active grasp contacts")


def _nominal_normal_force(assignment) -> float:
    if assignment.wrench_target is None:
        raise SchemaValidationError("R1 grasp audit lacks nominal contact wrench")
    force = assignment.wrench_target[:3]
    value = _norm(force)
    if not value > 0.0:
        raise SchemaValidationError("R1 grasp audit nominal normal force is zero")
    return value


def _force_moment_arm_about_axis(
    *,
    contact_position_world: Sequence[float],
    com_world: Sequence[float],
    tangent_basis_world: Sequence[float],
    axis_world: Sequence[float],
) -> float:
    if len(contact_position_world) != 3 or len(tangent_basis_world) != 6:
        raise SchemaValidationError("R1 grasp audit contact geometry is invalid")
    arm = tuple(
        float(contact_position_world[index]) - float(com_world[index])
        for index in range(3)
    )
    tangents = (
        _unit_vector(tangent_basis_world[:3], "contact tangent"),
        _unit_vector(tangent_basis_world[3:6], "contact tangent"),
    )
    return max(abs(_dot(axis_world, _cross(arm, tangent))) for tangent in tangents)


def _rotate_vector(quaternion_xyzw, vector):
    if len(quaternion_xyzw) != 4:
        raise SchemaValidationError("R1 grasp audit object quaternion is invalid")
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    magnitude = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not magnitude > 0.0 or not math.isfinite(magnitude):
        raise SchemaValidationError("R1 grasp audit object quaternion is invalid")
    q = (qx / magnitude, qy / magnitude, qz / magnitude)
    qw /= magnitude
    value = tuple(float(item) for item in vector)
    first = tuple(2.0 * item for item in _cross(q, value))
    second = _cross(q, first)
    return tuple(value[index] + qw * first[index] + second[index] for index in range(3))


def _unit_vector(value, label):
    if len(value) != 3:
        raise SchemaValidationError(f"R1 grasp audit {label} must have length three")
    result = tuple(float(item) for item in value)
    magnitude = _norm(result)
    if not magnitude > 0.0 or not math.isfinite(magnitude):
        raise SchemaValidationError(f"R1 grasp audit {label} is invalid")
    return tuple(item / magnitude for item in result)


def _norm(value):
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cross(left, right):
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


__all__ = [
    "ORDER9_R1_GRASP_ROTATION_AUDIT_VERSION",
    "audit_order9_r1_grasp_rotation_margin",
]
