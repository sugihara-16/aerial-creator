from __future__ import annotations

"""Isaac-free geometry admission for all eight persisted R1 task phases."""

import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKConfig,
    _global_joint_limits,
    resolve_mesh_backed_anchor_references,
)
from amsrr.feasibility.order9_native_posture_ik import (
    CppWholeStructureKinematics,
)
from amsrr.feasibility.order9_posture_collision import (
    CollisionAwareIKConfig,
    CollisionAwareNativeCentroidalPostureIKSolver,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_r1_complete_task_audit import (
    ORDER9_R1_COMPLETE_TASK_PHASES,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
    build_order9_r1_explicit_support_collision_object_v11,
)

ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_VERSION = (
    "order9_r1_complete_eight_phase_mesh_backed_geometry_v11"
)
ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_FILENAME = "complete_task_geometry_audit_v11.json"
_MINIMUM_NORMALIZED_JOINT_RESERVE = 0.01
_MAXIMUM_BODY_TILT_RAD = math.radians(60.0)
_ANCHOR_POSITION_TOLERANCE_M = 0.030
_ANCHOR_ATTITUDE_TOLERANCE_RAD = math.radians(1.0)


def audit_order9_r1_complete_task_geometry_v11(
    *,
    phases: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    morphology: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    physical_model_config_path: str | Path,
    contract: Order9R1NominalGeometryV11Contract,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Check joint reserve, attitude, self/object/support geometry at every knot."""

    started = perf_counter()
    if tuple(phases) != ORDER9_R1_COMPLETE_TASK_PHASES:
        raise SchemaValidationError("R1 v11 geometry audit phase order differs")
    physical = build_physical_model_from_config(physical_model_config_path)
    collision_object = build_order9_r1_explicit_support_collision_object_v11(
        task_spec,
        contract=contract,
    )
    kinematics = CppWholeStructureKinematics(
        physical,
        enable_collision_geometry=True,
        enable_exact_collision_geometry=True,
    )
    solver = CollisionAwareNativeCentroidalPostureIKSolver(
        physical,
        kinematics=kinematics,
        config=CentroidalPostureIKConfig(
            minimum_normalized_joint_limit_reserve=(_MINIMUM_NORMALIZED_JOINT_RESERVE)
        ),
        collision_config=CollisionAwareIKConfig(
            collision_margin_m=max(0.0005, contract.other_collision_clearance_m),
            collision_activation_distance_m=max(
                contract.required_robot_support_clearance_m,
                contract.other_collision_clearance_m,
            ),
        ),
    )
    ordered_ids = ordered_global_dock_joint_ids(morphology, physical)
    limits = _global_joint_limits(morphology, physical, ordered_ids)
    candidates = {
        int(value.candidate_id): value for value in contact_candidate_set.candidates
    }
    object_pose = tuple(collision_object.initial_pose_world)
    minimum_joint_fraction = math.inf
    maximum_tilt = 0.0
    maximum_anchor_position_error = 0.0
    maximum_anchor_attitude_error = 0.0
    minimum_object_clearance = math.inf
    minimum_support_clearance = math.inf
    checked_knots = 0
    checked_scenes = 0
    reused_scene_results = 0
    failures: list[dict[str, Any]] = []
    phase_records = []
    collision_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    stop = False

    for phase in ORDER9_R1_COMPLETE_TASK_PHASES:
        trajectory = phases[phase]
        trajectory.validate()
        phase_object_clearance = math.inf
        phase_support_clearance = math.inf
        phase_failure_count = 0
        for knot_index, knot in enumerate(trajectory.knots):
            checked_knots += 1
            centroidal = knot.centroidal_target
            posture = knot.posture_target
            if (
                centroidal is None
                or centroidal.com_pos_world is None
                or centroidal.body_orientation_world is None
                or posture is None
                or posture.joint_pos_target is None
                or set(posture.joint_pos_target) != set(ordered_ids)
            ):
                raise SchemaValidationError(
                    f"R1 v11 incomplete robot target at {phase}:{knot_index}"
                )
            q = {
                joint_id: float(posture.joint_pos_target[joint_id])
                for joint_id in ordered_ids
            }
            joint_fraction = min(
                min(
                    q[joint_id] - limits[joint_id][0], limits[joint_id][1] - q[joint_id]
                )
                / (limits[joint_id][1] - limits[joint_id][0])
                for joint_id in ordered_ids
            )
            minimum_joint_fraction = min(minimum_joint_fraction, joint_fraction)
            tilt = _body_tilt_rad(centroidal.body_orientation_world)
            maximum_tilt = max(maximum_tilt, tilt)
            if joint_fraction < _MINIMUM_NORMALIZED_JOINT_RESERVE - 1.0e-12:
                phase_failure_count += 1
                _record_failure(
                    failures,
                    phase=phase,
                    knot_index=knot_index,
                    failure="joint_limit_reserve",
                    measured=joint_fraction,
                    required=_MINIMUM_NORMALIZED_JOINT_RESERVE,
                )
            if tilt > _MAXIMUM_BODY_TILT_RAD + 1.0e-12:
                phase_failure_count += 1
                _record_failure(
                    failures,
                    phase=phase,
                    knot_index=knot_index,
                    failure="extreme_body_attitude",
                    measured=tilt,
                    required=_MAXIMUM_BODY_TILT_RAD,
                )
            centroidal_pose: Pose7D = (
                *tuple(float(value) for value in centroidal.com_pos_world),
                *tuple(float(value) for value in centroidal.body_orientation_world),
            )
            assigned_anchor_ids = {
                int(value.anchor_id) for value in knot.contact_assignments
            }
            if posture.free_anchor_pose_targets and assigned_anchor_ids:
                if not assigned_anchor_ids.issubset(posture.free_anchor_pose_targets):
                    raise SchemaValidationError(
                        "R1 v11 assigned anchor lacks a pose target"
                    )
                references = resolve_mesh_backed_anchor_references(
                    morphology,
                    physical,
                    tuple(sorted(assigned_anchor_ids)),
                )
                base_pose = solver.kinematics.base_pose_for_centroidal_target(
                    morphology=morphology,
                    q=q,
                    com_pos_world=centroidal.com_pos_world,
                    body_orientation_world=centroidal.body_orientation_world,
                )
                forward = solver.kinematics.forward(
                    morphology,
                    physical,
                    q,
                    base_pose,
                    references,
                )
                for anchor_id in assigned_anchor_ids:
                    expected = tuple(posture.free_anchor_pose_targets[anchor_id])
                    actual = forward.anchor_poses_world[anchor_id]
                    position_error = math.dist(expected[:3], actual[:3])
                    attitude_error = _quaternion_distance_rad(expected[3:], actual[3:])
                    maximum_anchor_position_error = max(
                        maximum_anchor_position_error, position_error
                    )
                    maximum_anchor_attitude_error = max(
                        maximum_anchor_attitude_error, attitude_error
                    )
                    if (
                        position_error > _ANCHOR_POSITION_TOLERANCE_M
                        or attitude_error > _ANCHOR_ATTITUDE_TOLERANCE_RAD
                    ):
                        phase_failure_count += 1
                        _record_failure(
                            failures,
                            phase=phase,
                            knot_index=knot_index,
                            failure="anchor_target_mismatch",
                            measured=max(position_error, attitude_error),
                            required=max(
                                _ANCHOR_POSITION_TOLERANCE_M,
                                _ANCHOR_ATTITUDE_TOLERANCE_RAD,
                            ),
                        )

            for target in knot.object_targets:
                if (
                    target.object_id == collision_object.object_id
                    and target.pose_target_world is not None
                ):
                    object_pose = tuple(target.pose_target_world)
            allowed_anchors = []
            for assignment in knot.contact_assignments:
                candidate = candidates.get(int(assignment.candidate_id))
                if candidate is None or int(candidate.anchor_id) != int(
                    assignment.anchor_id
                ):
                    raise SchemaValidationError(
                        "R1 v11 contact assignment is not in its candidate set"
                    )
                if (
                    candidate.target_entity_id == collision_object.object_id
                    and assignment.schedule_state
                    in {"attach", "maintain", "slide", "release"}
                ):
                    allowed_anchors.append(int(assignment.anchor_id))

            scenes = (
                (
                    "object_or_self",
                    object_pose,
                    collision_object.size_m,
                    tuple(sorted(allowed_anchors)),
                    contract.other_collision_clearance_m,
                ),
                *(
                    (
                        "support_or_self",
                        box.pose_world,
                        box.size_m,
                        (),
                        contract.required_robot_support_clearance_m,
                    )
                    for box in collision_object.environment_boxes
                ),
            )
            for kind, pose, size, allowed, margin in scenes:
                key = (
                    kind,
                    tuple(round(value, 12) for value in centroidal_pose),
                    tuple(
                        (joint_id, round(q[joint_id], 12)) for joint_id in ordered_ids
                    ),
                    tuple(round(value, 12) for value in pose),
                    allowed,
                    margin,
                )
                result = collision_cache.get(key)
                if result is None:
                    solver.set_collision_scene(
                        morphology=morphology,
                        object_pose_world=pose,
                        object_size_m=size,
                        allowed_anchor_ids=allowed,
                    )
                    result = solver.check_configuration(
                        morphology=morphology,
                        centroidal_pose_world=centroidal_pose,
                        joint_positions_rad=q,
                        exact=True,
                        margin_m=margin,
                        ground_plane_z_m=collision_object.ground_plane_z_m,
                    )
                    collision_cache[key] = result
                    checked_scenes += 1
                else:
                    reused_scene_results += 1
                clearance = float(result["minimum_clearance_m"])
                if kind == "object_or_self":
                    scene_accepted, obstacle_clearance, scene_violations = (
                        _obstacle_scene_admission(
                            result,
                            required_obstacle_clearance_m=(
                                contract.other_collision_clearance_m
                            ),
                            required_ground_clearance_m=(
                                contract.other_collision_clearance_m
                            ),
                        )
                    )
                    clearance = obstacle_clearance
                    minimum_object_clearance = min(minimum_object_clearance, clearance)
                    phase_object_clearance = min(phase_object_clearance, clearance)
                else:
                    scene_accepted, support_clearance, scene_violations = (
                        _obstacle_scene_admission(
                            result,
                            required_obstacle_clearance_m=(
                                contract.required_robot_support_clearance_m
                            ),
                            required_ground_clearance_m=(
                                contract.other_collision_clearance_m
                            ),
                        )
                    )
                    clearance = support_clearance
                    minimum_support_clearance = min(
                        minimum_support_clearance, clearance
                    )
                    phase_support_clearance = min(phase_support_clearance, clearance)
                rejected = not scene_accepted
                if rejected:
                    phase_failure_count += 1
                    _record_failure(
                        failures,
                        phase=phase,
                        knot_index=knot_index,
                        failure=kind,
                        measured=clearance,
                        required=margin,
                        detail={
                            "violating_pair_count": int(
                                result.get("violating_pair_count", 0)
                            ),
                            "ground_violating_proxy_count": int(
                                result.get("ground_violating_proxy_count", 0)
                            ),
                            "worst_pairs": scene_violations[:4],
                        },
                    )
            if fail_fast and phase_failure_count:
                stop = True
                break
        phase_records.append(
            {
                "phase": phase,
                "knot_count": len(trajectory.knots),
                "failure_count": phase_failure_count,
                "minimum_object_or_self_clearance_m": phase_object_clearance,
                "minimum_support_or_self_clearance_m": phase_support_clearance,
            }
        )
        if stop:
            break

    accepted = not failures
    return {
        "audit_version": ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_VERSION,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "collision_geometry_mode": (
            "urdf_stl_triangle_mesh_robot_plus_exact_task_box_geometry"
        ),
        "checked_phases": [value["phase"] for value in phase_records],
        "checked_knot_count": checked_knots,
        "computed_collision_scene_count": checked_scenes,
        "reused_collision_scene_result_count": reused_scene_results,
        "phase_records": phase_records,
        "minimum_object_or_self_clearance_m": minimum_object_clearance,
        "minimum_support_or_self_clearance_m": minimum_support_clearance,
        "minimum_robot_support_clearance_lower_bound_m": (minimum_support_clearance),
        "minimum_normalized_joint_limit_reserve": minimum_joint_fraction,
        "maximum_body_tilt_rad": maximum_tilt,
        "maximum_anchor_position_error_m": maximum_anchor_position_error,
        "maximum_anchor_attitude_error_rad": maximum_anchor_attitude_error,
        "required_robot_support_clearance_m": (
            contract.required_robot_support_clearance_m
        ),
        "required_other_collision_clearance_m": (contract.other_collision_clearance_m),
        "support_geometry_source_sha256": contract.canonical_report_sha256,
        "failure_count": sum(value["failure_count"] for value in phase_records),
        "failure_examples": failures,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "trajectory_optimization_invoked": False,
        "wall_time_s": perf_counter() - started,
        "formal_isaac_admission": accepted,
    }


def require_order9_r1_complete_task_geometry_v11(audit: Mapping[str, Any]) -> None:
    if (
        audit.get("audit_version") != ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_VERSION
        or audit.get("accepted") is not True
        or audit.get("status") != "accepted"
        or tuple(audit.get("checked_phases", ())) != ORDER9_R1_COMPLETE_TASK_PHASES
        or audit.get("isaac_invoked") is not False
        or audit.get("controller_layers_invoked") is not False
        or audit.get("formal_isaac_admission") is not True
    ):
        examples = audit.get("failure_examples", [])
        suffix = "" if not examples else f": {examples[0]}"
        raise SchemaValidationError(
            "R1 v11 eight-phase geometry admission rejected" + suffix
        )


def _record_failure(
    failures: list[dict[str, Any]],
    *,
    phase: str,
    knot_index: int,
    failure: str,
    measured: float,
    required: float,
    detail: Mapping[str, Any] | None = None,
) -> None:
    if len(failures) >= 16:
        return
    failures.append(
        {
            "phase": phase,
            "knot_index": int(knot_index),
            "failure": failure,
            "measured": float(measured),
            "required": float(required),
            **({} if detail is None else {"detail": dict(detail)}),
        }
    )


def _obstacle_scene_admission(
    result: Mapping[str, Any],
    *,
    required_obstacle_clearance_m: float,
    required_ground_clearance_m: float,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Separate exact self-intersection from obstacle/ground clearance.

    The native checker reports self and active-obstacle pairs together.  Its
    ``worst_pairs`` list contains every pair below the requested margin, so it
    can be separated without another geometry evaluation.
    """

    pairs = [dict(value) for value in (result.get("worst_pairs") or [])]
    violations = []
    obstacle_values = []
    for pair in pairs:
        clearance = float(pair["clearance_m"])
        is_obstacle = pair.get("second_kind") == "object"
        if is_obstacle:
            obstacle_values.append(clearance)
            if (
                pair.get("colliding") is True
                or clearance + 1.0e-12 < required_obstacle_clearance_m
            ):
                violations.append(
                    {
                        **pair,
                        "required_clearance_m": required_obstacle_clearance_m,
                    }
                )
        elif pair.get("colliding") is True:
            # Fixed internal parts may be arbitrarily close by design.  Exact
            # triangle-mesh intersection, not an obstacle clearance margin,
            # is the self-interference criterion.
            violations.append({**pair, "required_state": "not_intersecting"})
    ground = result.get("minimum_ground_clearance_m")
    if ground is not None and float(ground) + 1.0e-12 < required_ground_clearance_m:
        violations.append(
            {
                "second_kind": "ground",
                "clearance_m": float(ground),
                "required_clearance_m": required_ground_clearance_m,
            }
        )
    # If no obstacle pair appears, every one was pruned above its requested
    # margin.  Record that proven lower bound, not infinity.
    obstacle_lower_bound = (
        min(obstacle_values)
        if obstacle_values
        else float(required_obstacle_clearance_m)
    )
    return not violations, obstacle_lower_bound, violations


def _body_tilt_rad(quaternion) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise SchemaValidationError("R1 v11 body quaternion is invalid")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    vertical_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, vertical_z)))


def _quaternion_distance_rad(first, second) -> float:
    a = tuple(float(value) for value in first)
    b = tuple(float(value) for value in second)
    a_norm = math.sqrt(sum(value * value for value in a))
    b_norm = math.sqrt(sum(value * value for value in b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        raise SchemaValidationError("R1 v11 anchor quaternion is invalid")
    dot = abs(sum(a[index] * b[index] for index in range(4)) / (a_norm * b_norm))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_FILENAME",
    "ORDER9_R1_COMPLETE_TASK_GEOMETRY_V11_VERSION",
    "audit_order9_r1_complete_task_geometry_v11",
    "require_order9_r1_complete_task_geometry_v11",
]
