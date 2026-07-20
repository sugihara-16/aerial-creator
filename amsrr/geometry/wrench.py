from __future__ import annotations

"""Frame conversion for the versioned contact-wrench policy contract."""

from amsrr.geometry.pose_math import matvec, transform_from_pose, transpose
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate
from amsrr.schemas.policies import ContactAssignment


def contact_wrench_to_world(
    wrench_contact: list[float] | tuple[float, ...],
    candidate: ContactCandidate,
) -> list[float]:
    """Rotate a wrench to world at the same candidate-frame origin.

    The v2 contract defines the wrench as the robot anchor acting on the target.
    Since both representations use the candidate-frame origin, no moment-arm
    shift is applied here.
    """

    wrench = _validated_wrench(wrench_contact)
    rotation = transform_from_pose(candidate.contact_frame_world).rotation
    force_world = matvec(rotation, (wrench[0], wrench[1], wrench[2]))
    torque_world = matvec(rotation, (wrench[3], wrench[4], wrench[5]))
    return [*force_world, *torque_world]


def world_wrench_to_contact(
    wrench_world: list[float] | tuple[float, ...],
    candidate: ContactCandidate,
) -> list[float]:
    """Rotate a same-origin world wrench into the candidate contact frame."""

    wrench = _validated_wrench(wrench_world)
    world_from_contact = transform_from_pose(candidate.contact_frame_world).rotation
    contact_from_world = transpose(world_from_contact)
    force_contact = matvec(contact_from_world, (wrench[0], wrench[1], wrench[2]))
    torque_contact = matvec(contact_from_world, (wrench[3], wrench[4], wrench[5]))
    return [*force_contact, *torque_contact]


def assignment_wrench_target_world(
    assignment: ContactAssignment,
    candidate: ContactCandidate,
) -> list[float] | None:
    """Resolve either contract version to a world-frame target wrench."""

    if assignment.wrench_target is None:
        return None
    if assignment.wrench_frame == "world":
        return _validated_wrench(assignment.wrench_target)
    if assignment.wrench_frame == "contact":
        return contact_wrench_to_world(assignment.wrench_target, candidate)
    raise SchemaValidationError(
        f"unsupported ContactAssignment.wrench_frame: {assignment.wrench_frame!r}"
    )


def _validated_wrench(values: list[float] | tuple[float, ...]) -> list[float]:
    if len(values) != 6:
        raise SchemaValidationError("wrench must have length 6")
    return [float(value) for value in values]
