from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from amsrr.geometry.pose_math import transform_from_pose
from amsrr.schemas.common import Pose7D, SchemaValidationError, Vector3
from amsrr.schemas.order8 import Order8RawContactPatch


class Order8ContactMeasurementError(SchemaValidationError):
    """Raised when an auxiliary Order 8 geometry input is unusable."""


@dataclass(frozen=True)
class Order8ContactMeasurement:
    raw_contact_patches: list[Order8RawContactPatch]
    raw_contact_valid: bool
    raw_contact_saturated: bool
    raw_contact_nonfinite: bool
    raw_contact_layout_invalid: bool
    raw_contact_out_of_range: bool
    raw_contact_count: int
    raw_contact_capacity: int
    selected_contact: bool
    selected_raw_contact_count: int
    selected_contact_link_ids: tuple[str, ...]
    selected_force_magnitude_sum_n: float
    unintended_contact: bool
    unintended_raw_contact_count: int
    unintended_contact_link_ids: tuple[str, ...]
    unintended_force_magnitude_sum_n: float
    sensor_body_to_global_link_id: tuple[tuple[str, str], ...]
    failure_reasons: tuple[str, ...]
    patch_kinematics: tuple["Order8ContactPatchKinematics", ...]


@dataclass(frozen=True)
class Order8ContactPatchKinematics:
    """Privileged vector kinematics paired with one raw contact patch.

    The versioned Order-8 schema intentionally keeps only the non-negative
    slip-speed magnitude used by safety/acceptance.  These vectors remain in
    the simulator diagnostic layer so a fault-isolation run can distinguish
    signed sliding direction without exposing raw contact truth to a policy or
    QPID command.
    """

    patch_id: str
    robot_link_id: str
    contact_point_world: Vector3
    contact_normal_world: Vector3
    body_contact_velocity_world_mps: Vector3
    object_contact_velocity_world_mps: Vector3
    relative_velocity_world_mps: Vector3
    tangential_velocity_world_mps: Vector3


@dataclass(frozen=True)
class Order8AABB:
    minimum_world: Vector3
    maximum_world: Vector3


def measure_order8_raw_contacts(
    *,
    sensor_body_ids: Sequence[str],
    sensor_global_link_ids: Sequence[str],
    selected_global_link_ids: Sequence[str],
    object_body_id: str,
    contact_counts: Sequence[int],
    start_indices: Sequence[int],
    patch_forces_n: Sequence[float],
    patch_points_world: Sequence[Sequence[float]],
    patch_normals_world: Sequence[Sequence[float]],
    patch_separations_m: Sequence[float],
    raw_capacity: int,
    body_com_poses_world: Mapping[str, Sequence[float]],
    body_twists_world: Mapping[str, Sequence[float]],
    object_com_pose_world: Sequence[float],
    object_twist_world: Sequence[float],
) -> Order8ContactMeasurement:
    """Convert flattened PhysX contact-view buffers into patch-level evidence.

    The function assumes one object filter per sensor body, hence one count/start
    entry per ``sensor_body_id``.  Twists are world-frame ``[v_xyz, omega_xyz]``
    at each body's COM.  No patch forces are vector-summed: every output force and
    torque is a non-negative per-patch magnitude.
    """

    failures: list[str] = []
    layout_invalid = False
    out_of_range = False
    nonfinite = False

    def layout_failure(reason: str) -> None:
        nonlocal layout_invalid
        layout_invalid = True
        _append_unique(failures, reason)

    def range_failure(reason: str) -> None:
        nonlocal out_of_range
        out_of_range = True
        _append_unique(failures, reason)

    try:
        body_ids = list(sensor_body_ids)
        global_ids = list(sensor_global_link_ids)
        selected_ids_input = list(selected_global_link_ids)
        counts = list(contact_counts)
        starts = list(start_indices)
        forces = list(patch_forces_n)
        points = list(patch_points_world)
        normals = list(patch_normals_world)
        separations = list(patch_separations_m)
    except TypeError:
        return _invalid_measurement(
            raw_capacity=raw_capacity,
            failures=("contact_inputs_not_sequences",),
            layout_invalid=True,
        )

    capacity = raw_capacity if _is_nonnegative_int(raw_capacity) else 0
    if not _is_positive_int(raw_capacity):
        layout_failure("raw_capacity_not_positive_integer")
    if not isinstance(object_body_id, str) or not object_body_id:
        layout_failure("object_body_id_invalid")
    if len(body_ids) != len(global_ids):
        layout_failure("sensor_identity_length_mismatch")
    if len(counts) != len(body_ids) or len(starts) != len(body_ids):
        layout_failure("contact_pair_layout_length_mismatch")
    if any(not isinstance(value, str) or not value for value in body_ids):
        layout_failure("sensor_body_id_invalid")
    if any(not isinstance(value, str) or not value for value in global_ids):
        layout_failure("sensor_global_link_id_invalid")
    if len(set(body_ids)) != len(body_ids):
        layout_failure("sensor_body_id_not_unique")
    if len(set(global_ids)) != len(global_ids):
        layout_failure("sensor_global_link_id_not_unique")
    if (
        len(selected_ids_input) < 2
        or any(not isinstance(value, str) or not value for value in selected_ids_input)
        or len(set(selected_ids_input)) != len(selected_ids_input)
    ):
        layout_failure("selected_global_link_ids_invalid")
    selected_ids = set(
        value for value in selected_ids_input if isinstance(value, str) and value
    )
    if not selected_ids.issubset(set(global_ids)):
        layout_failure("selected_global_link_id_not_in_sensor_view")
    if any(not _is_nonnegative_int(value) for value in counts):
        layout_failure("contact_count_not_nonnegative_integer")
    if any(not _is_nonnegative_int(value) for value in starts):
        layout_failure("contact_start_not_nonnegative_integer")
    raw_buffer_lengths = tuple(
        len(buffer) for buffer in (forces, points, normals, separations)
    )
    if len(set(raw_buffer_lengths)) != 1:
        layout_failure("raw_patch_buffer_length_mismatch")
    available_patch_count = min(raw_buffer_lengths, default=0)

    total_count = (
        sum(int(value) for value in counts)
        if all(_is_nonnegative_int(value) for value in counts)
        else 0
    )
    active_records: list[tuple[int, int, int]] = []
    occupied_buffer_indices: set[int] = set()
    if len(counts) == len(starts) == len(body_ids):
        for pair_index, (start, count) in enumerate(zip(starts, counts, strict=True)):
            if not _is_nonnegative_int(start) or not _is_nonnegative_int(count):
                continue
            stop = start + count
            if stop > capacity:
                range_failure("active_patch_range_exceeds_capacity")
                continue
            if count == 0:
                continue
            if stop > available_patch_count:
                range_failure("active_patch_range_exceeds_buffer")
                continue
            for local_patch_index, buffer_index in enumerate(range(start, stop)):
                if buffer_index in occupied_buffer_indices:
                    layout_failure("active_patch_ranges_overlap")
                    continue
                occupied_buffer_indices.add(buffer_index)
                active_records.append((pair_index, local_patch_index, buffer_index))

    saturated = bool(
        not _is_positive_int(raw_capacity)
        or total_count >= capacity > 0
        or out_of_range
    )
    if saturated:
        _append_unique(failures, "raw_contact_buffer_saturated")

    identity_pairs = tuple(sorted(zip(body_ids, global_ids)))
    body_states: dict[str, tuple[Pose7D, tuple[float, ...]]] = {}
    active_pair_indices = sorted({record[0] for record in active_records})
    for pair_index in active_pair_indices:
        body_id = body_ids[pair_index]
        pose = body_com_poses_world.get(body_id)
        twist = body_twists_world.get(body_id)
        if not _valid_pose(pose) or not _valid_twist(twist):
            if pose is not None and not _finite_sequence(pose, 7):
                nonfinite = nonfinite or _contains_nonfinite(pose)
            if twist is not None and not _finite_sequence(twist, 6):
                nonfinite = nonfinite or _contains_nonfinite(twist)
            layout_failure("sensor_body_com_state_invalid")
            continue
        body_states[body_id] = (
            tuple(float(value) for value in pose),  # type: ignore[arg-type]
            tuple(float(value) for value in twist),  # type: ignore[arg-type]
        )
    if not _valid_pose(object_com_pose_world) or not _valid_twist(object_twist_world):
        nonfinite = nonfinite or _contains_nonfinite(object_com_pose_world)
        nonfinite = nonfinite or _contains_nonfinite(object_twist_world)
        layout_failure("object_com_state_invalid")

    candidate_patches: list[
        tuple[
            str,
            int,
            Order8RawContactPatch,
            Order8ContactPatchKinematics,
        ]
    ] = []
    if not layout_invalid and not out_of_range:
        object_pose = tuple(float(value) for value in object_com_pose_world)
        object_twist = tuple(float(value) for value in object_twist_world)
        for pair_index, local_patch_index, buffer_index in active_records:
            force = forces[buffer_index]
            separation = separations[buffer_index]
            point = points[buffer_index]
            normal = normals[buffer_index]
            if (
                not _is_finite_number(force)
                or not _is_finite_number(separation)
                or not _finite_sequence(point, 3)
                or not _finite_sequence(normal, 3)
            ):
                nonfinite = True
                _append_unique(failures, "active_patch_nonfinite")
                continue
            normal_world = _unit_or_none(normal)
            if normal_world is None:
                layout_failure("active_patch_normal_zero")
                continue
            body_id = body_ids[pair_index]
            global_link_id = global_ids[pair_index]
            body_pose, body_twist = body_states[body_id]
            point_world = tuple(float(value) for value in point)
            force_magnitude = abs(float(force))
            force_world = _scale(normal_world, force_magnitude)
            body_radius = _subtract(point_world, body_pose[:3])
            body_contact_velocity = _contact_point_velocity(
                body_pose,
                body_twist,
                point_world,
            )
            object_contact_velocity = _contact_point_velocity(
                object_pose,
                object_twist,
                point_world,
            )
            relative_velocity = _subtract(
                body_contact_velocity,
                object_contact_velocity,
            )
            tangential_velocity = _subtract(
                relative_velocity,
                _scale(normal_world, _dot(relative_velocity, normal_world)),
            )
            patch_id = (
                f"{global_link_id}::{object_body_id}::patch_{local_patch_index:04d}"
            )
            patch = Order8RawContactPatch(
                patch_id=patch_id,
                robot_link_id=global_link_id,
                other_body_id=object_body_id,
                normal_force_n=force_magnitude,
                force_magnitude_n=force_magnitude,
                torque_magnitude_nm=_norm(_cross(body_radius, force_world)),
                penetration_m=max(0.0, -float(separation)),
                tangential_slip_speed_mps=_norm(tangential_velocity),
            )
            kinematics = Order8ContactPatchKinematics(
                patch_id=patch_id,
                robot_link_id=global_link_id,
                contact_point_world=point_world,
                contact_normal_world=normal_world,
                body_contact_velocity_world_mps=body_contact_velocity,
                object_contact_velocity_world_mps=object_contact_velocity,
                relative_velocity_world_mps=relative_velocity,
                tangential_velocity_world_mps=tangential_velocity,
            )
            candidate_patches.append(
                (global_link_id, local_patch_index, patch, kinematics)
            )

    if nonfinite:
        _append_unique(failures, "raw_contact_nonfinite")
    valid = not (layout_invalid or out_of_range or nonfinite or saturated)
    if not valid:
        patches: list[Order8RawContactPatch] = []
        patch_kinematics: tuple[Order8ContactPatchKinematics, ...] = ()
    else:
        candidate_patches.sort(key=lambda value: (value[0], value[1]))
        patches = [value[2] for value in candidate_patches]
        patch_kinematics = tuple(value[3] for value in candidate_patches)

    selected_patches = [
        patch for patch in patches if patch.robot_link_id in selected_ids
    ]
    unintended_patches = [
        patch for patch in patches if patch.robot_link_id not in selected_ids
    ]
    selected_contact_links = tuple(
        sorted({patch.robot_link_id for patch in selected_patches})
    )
    unintended_contact_links = tuple(
        sorted({patch.robot_link_id for patch in unintended_patches})
    )
    return Order8ContactMeasurement(
        raw_contact_patches=patches,
        raw_contact_valid=valid,
        raw_contact_saturated=saturated,
        raw_contact_nonfinite=nonfinite,
        raw_contact_layout_invalid=layout_invalid,
        raw_contact_out_of_range=out_of_range,
        raw_contact_count=total_count,
        raw_contact_capacity=capacity,
        selected_contact=bool(selected_patches),
        selected_raw_contact_count=len(selected_patches),
        selected_contact_link_ids=selected_contact_links,
        selected_force_magnitude_sum_n=sum(
            patch.force_magnitude_n for patch in selected_patches
        ),
        unintended_contact=bool(unintended_patches),
        unintended_raw_contact_count=len(unintended_patches),
        unintended_contact_link_ids=unintended_contact_links,
        unintended_force_magnitude_sum_n=sum(
            patch.force_magnitude_n for patch in unintended_patches
        ),
        sensor_body_to_global_link_id=identity_pairs,
        failure_reasons=tuple(failures),
        patch_kinematics=patch_kinematics,
    )


def oriented_cuboid_world_aabb(
    *,
    pose_world: Sequence[float],
    size_m: Sequence[float],
) -> Order8AABB:
    """Return the conservative world AABB of an oriented cuboid."""

    pose = _require_pose(pose_world, "pose_world")
    size = _require_positive_vector3(size_m, "size_m")
    rotation = transform_from_pose(pose).rotation
    half = tuple(value * 0.5 for value in size)
    extents = tuple(
        sum(abs(rotation[row][column]) * half[column] for column in range(3))
        for row in range(3)
    )
    center = pose[:3]
    return Order8AABB(
        minimum_world=tuple(center[index] - extents[index] for index in range(3)),
        maximum_world=tuple(center[index] + extents[index] for index in range(3)),
    )


def object_bottom_clearance_m(
    *,
    object_pose_world: Sequence[float],
    object_size_m: Sequence[float],
    floor_height_m: float = 0.0,
) -> float:
    if not _is_finite_number(floor_height_m):
        raise Order8ContactMeasurementError("floor_height_m must be finite")
    bounds = oriented_cuboid_world_aabb(
        pose_world=object_pose_world,
        size_m=object_size_m,
    )
    return max(0.0, bounds.minimum_world[2] - float(floor_height_m))


def object_transport_distance_m(
    initial_object_pose_world: Sequence[float],
    current_object_pose_world: Sequence[float],
) -> float:
    initial = _require_pose(initial_object_pose_world, "initial_object_pose_world")
    current = _require_pose(current_object_pose_world, "current_object_pose_world")
    return _norm(_subtract(current[:3], initial[:3]))


def relative_point_speed_mps(
    *,
    body_reference_pose_world: Sequence[float],
    body_twist_world: Sequence[float],
    object_reference_pose_world: Sequence[float],
    object_twist_world: Sequence[float],
    point_world: Sequence[float],
) -> float:
    """Return non-privileged body/object relative speed at a world point.

    The reference pose and twist must describe the same frame for each body.
    This is the kinematic counterpart of contact-patch slip computation, but
    it needs no contact existence, force, normal, impulse, or penetration.
    """

    relative_velocity = _relative_point_velocity_world(
        body_reference_pose_world=body_reference_pose_world,
        body_twist_world=body_twist_world,
        object_reference_pose_world=object_reference_pose_world,
        object_twist_world=object_twist_world,
        point_world=point_world,
    )
    return _norm(relative_velocity)


def relative_point_normal_speed_mps(
    *,
    body_reference_pose_world: Sequence[float],
    body_twist_world: Sequence[float],
    object_reference_pose_world: Sequence[float],
    object_twist_world: Sequence[float],
    point_world: Sequence[float],
    surface_normal_world: Sequence[float],
) -> float:
    """Return absolute normal relative speed at a non-privileged surface point.

    Tangential motion is intentionally excluded.  Order 8 treats each selected
    Dock target as a bounded surface region, so tangential motion inside that
    region must not masquerade as normal separation during the post-``q_close``
    force-settle gate.  The normal comes from observed object geometry, never
    from simulator contact truth.
    """

    return abs(
        relative_point_normal_velocity_mps(
            body_reference_pose_world=body_reference_pose_world,
            body_twist_world=body_twist_world,
            object_reference_pose_world=object_reference_pose_world,
            object_twist_world=object_twist_world,
            point_world=point_world,
            surface_normal_world=surface_normal_world,
        )
    )


def relative_point_normal_velocity_mps(
    *,
    body_reference_pose_world: Sequence[float],
    body_twist_world: Sequence[float],
    object_reference_pose_world: Sequence[float],
    object_twist_world: Sequence[float],
    point_world: Sequence[float],
    surface_normal_world: Sequence[float],
) -> float:
    """Return signed relative velocity along a geometric surface normal."""

    if not _finite_sequence(surface_normal_world, 3):
        raise Order8ContactMeasurementError(
            "surface_normal_world must contain three finite values"
        )
    normal = tuple(float(value) for value in surface_normal_world)
    normal_norm = _norm(normal)
    if normal_norm <= 1.0e-12:
        raise Order8ContactMeasurementError(
            "surface_normal_world must have non-zero length"
        )
    unit_normal = _scale(normal, 1.0 / normal_norm)
    relative_velocity = _relative_point_velocity_world(
        body_reference_pose_world=body_reference_pose_world,
        body_twist_world=body_twist_world,
        object_reference_pose_world=object_reference_pose_world,
        object_twist_world=object_twist_world,
        point_world=point_world,
    )
    return _dot(relative_velocity, unit_normal)


def _relative_point_velocity_world(
    *,
    body_reference_pose_world: Sequence[float],
    body_twist_world: Sequence[float],
    object_reference_pose_world: Sequence[float],
    object_twist_world: Sequence[float],
    point_world: Sequence[float],
) -> Vector3:
    body_pose = _require_pose(body_reference_pose_world, "body_reference_pose_world")
    object_pose = _require_pose(
        object_reference_pose_world,
        "object_reference_pose_world",
    )
    if not _valid_twist(body_twist_world):
        raise Order8ContactMeasurementError(
            "body_twist_world must contain six finite values"
        )
    if not _valid_twist(object_twist_world):
        raise Order8ContactMeasurementError(
            "object_twist_world must contain six finite values"
        )
    if not _finite_sequence(point_world, 3):
        raise Order8ContactMeasurementError(
            "point_world must contain three finite values"
        )
    point = tuple(float(value) for value in point_world)
    return _subtract(
        _contact_point_velocity(body_pose, body_twist_world, point),
        _contact_point_velocity(object_pose, object_twist_world, point),
    )


def aabb_clearance_m(first: Order8AABB, second: Order8AABB) -> float:
    _validate_aabb(first, "first")
    _validate_aabb(second, "second")
    axis_gaps = tuple(
        max(
            first.minimum_world[index] - second.maximum_world[index],
            second.minimum_world[index] - first.maximum_world[index],
            0.0,
        )
        for index in range(3)
    )
    return _norm(axis_gaps)


def gripper_object_clearance_m(
    *,
    gripper_aabbs_world: Sequence[Order8AABB],
    object_pose_world: Sequence[float],
    object_size_m: Sequence[float],
) -> float:
    if not gripper_aabbs_world:
        raise Order8ContactMeasurementError(
            "gripper_aabbs_world must contain at least one bounds record"
        )
    object_bounds = oriented_cuboid_world_aabb(
        pose_world=object_pose_world,
        size_m=object_size_m,
    )
    return min(
        aabb_clearance_m(gripper_bounds, object_bounds)
        for gripper_bounds in gripper_aabbs_world
    )


def _invalid_measurement(
    *,
    raw_capacity: object,
    failures: tuple[str, ...],
    layout_invalid: bool,
) -> Order8ContactMeasurement:
    return Order8ContactMeasurement(
        raw_contact_patches=[],
        raw_contact_valid=False,
        raw_contact_saturated=not _is_positive_int(raw_capacity),
        raw_contact_nonfinite=False,
        raw_contact_layout_invalid=layout_invalid,
        raw_contact_out_of_range=False,
        raw_contact_count=0,
        raw_contact_capacity=(
            int(raw_capacity) if _is_nonnegative_int(raw_capacity) else 0
        ),
        selected_contact=False,
        selected_raw_contact_count=0,
        selected_contact_link_ids=(),
        selected_force_magnitude_sum_n=0.0,
        unintended_contact=False,
        unintended_raw_contact_count=0,
        unintended_contact_link_ids=(),
        unintended_force_magnitude_sum_n=0.0,
        sensor_body_to_global_link_id=(),
        failure_reasons=failures,
        patch_kinematics=(),
    )


def _contact_point_velocity(
    com_pose_world: Sequence[float],
    twist_world: Sequence[float],
    point_world: Vector3,
) -> Vector3:
    radius = _subtract(point_world, com_pose_world[:3])
    return _add(
        tuple(float(value) for value in twist_world[:3]),
        _cross(tuple(float(value) for value in twist_world[3:]), radius),
    )


def _valid_pose(value: object) -> bool:
    if not _finite_sequence(value, 7):
        return False
    quaternion_norm = _norm(tuple(float(component) for component in value[3:7]))
    return quaternion_norm > 1.0e-12


def _valid_twist(value: object) -> bool:
    return _finite_sequence(value, 6)


def _require_pose(value: object, label: str) -> Pose7D:
    if not _valid_pose(value):
        raise Order8ContactMeasurementError(f"{label} must be a finite Pose7D")
    return tuple(float(component) for component in value)  # type: ignore[arg-type,return-value]


def _require_positive_vector3(value: object, label: str) -> Vector3:
    if not _finite_sequence(value, 3) or any(
        float(component) <= 0.0 for component in value
    ):
        raise Order8ContactMeasurementError(
            f"{label} must contain three finite positive values"
        )
    return tuple(float(component) for component in value)  # type: ignore[arg-type,return-value]


def _validate_aabb(bounds: Order8AABB, label: str) -> None:
    if not _finite_sequence(bounds.minimum_world, 3) or not _finite_sequence(
        bounds.maximum_world, 3
    ):
        raise Order8ContactMeasurementError(f"{label} AABB must be finite")
    if any(
        bounds.minimum_world[index] > bounds.maximum_world[index] for index in range(3)
    ):
        raise Order8ContactMeasurementError(
            f"{label} AABB minimum must not exceed maximum"
        )


def _unit_or_none(value: Sequence[float]) -> Vector3 | None:
    vector = tuple(float(component) for component in value)
    magnitude = _norm(vector)
    if magnitude <= 1.0e-12:
        return None
    return _scale(vector, 1.0 / magnitude)


def _finite_sequence(value: object, length: int) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(_is_finite_number(component) for component in value)
    )


def _contains_nonfinite(value: object) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and any(
            isinstance(component, (int, float))
            and not isinstance(component, bool)
            and not math.isfinite(float(component))
            for component in value
        )
    )


def _is_finite_number(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_nonnegative_int(value) and int(value) > 0


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _add(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(float(left[index]) + float(right[index]) for index in range(3))  # type: ignore[return-value]


def _subtract(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))  # type: ignore[return-value]


def _scale(vector: Sequence[float], scalar: float) -> Vector3:
    return tuple(float(vector[index]) * scalar for index in range(3))  # type: ignore[return-value]


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


__all__ = [
    "Order8AABB",
    "Order8ContactMeasurement",
    "Order8ContactMeasurementError",
    "aabb_clearance_m",
    "gripper_object_clearance_m",
    "measure_order8_raw_contacts",
    "object_bottom_clearance_m",
    "object_transport_distance_m",
    "oriented_cuboid_world_aabb",
    "relative_point_normal_speed_mps",
    "relative_point_normal_velocity_mps",
    "relative_point_speed_mps",
]
