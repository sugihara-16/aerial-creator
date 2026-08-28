from __future__ import annotations

"""Fast, fail-closed joint-limit reserve projection for Order 9 R1.

The C3 nominal generator is intentionally left on its promoted, zero-reserve
path.  R1 applies the stricter reserve only after the complete nominal path is
available, so a boundary clamp cannot perturb rolling replanning or multiply
IK work.  Only changed knots are rechecked in task and collision space.
"""

from dataclasses import replace
import math
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKConfig,
    _global_joint_limits,
    _joint_limit_branch_seeds,
    _joint_limits_with_normalized_reserve,
    _pose_rotation_error_vector,
    resolve_mesh_backed_anchor_references,
)
from amsrr.feasibility.order9_batched_kinematics import (
    _poses_from_arrays,
    _rotation_log,
)
from amsrr.geometry.pose_math import quat_to_matrix
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
    Order9C3NominalWindow,
    _flatten_nominal_windows,
    _ideal_endpoint_observation,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_neutral_runtime_observation,
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_posture_resolver import (
    Order9ResolvedPostureTrajectory,
    _allowed_object_contact_anchors,
    _collision_object_pose,
    _default_posture_ik_solver,
    _maximum_timed_joint_rate,
)
from amsrr.utils.hashing import stable_hash

ORDER9_R1_JOINT_RESERVE_PROJECTION_VERSION = (
    "order9_r1_joint_reserve_projection_v2_anchor_radius_30mm"
)
# Maximum three-dimensional Euclidean distance from each R1 target grasp point.
# This is not an independent +/-30 mm allowance on each Cartesian axis.
ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M = 0.030


def _project_q(
    q: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], bool]:
    if set(q) != set(limits):
        raise SchemaValidationError(
            "R1 joint-reserve projection received a joint identity mismatch"
        )
    projected = {
        joint_id: min(
            max(float(q[joint_id]), float(lower)),
            float(upper),
        )
        for joint_id, (lower, upper) in limits.items()
    }
    changed = any(
        abs(projected[joint_id] - float(q[joint_id])) > 1.0e-15
        for joint_id in projected
    )
    return projected, changed


def _trajectory_with_projected_q(
    trajectory: ContactWrenchTrajectory,
    *,
    ordered_ids: tuple[str, ...],
    reserve_limits: Mapping[str, tuple[float, float]],
) -> tuple[ContactWrenchTrajectory, tuple[int, ...], list[dict[str, float]]]:
    projected = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    q_samples: list[dict[str, float]] = []
    changed_indices: list[int] = []
    for index, knot in enumerate(projected.knots):
        posture = knot.posture_target
        if posture is None or posture.joint_pos_target is None:
            raise SchemaValidationError(
                "R1 joint-reserve projection requires resolved joint targets"
            )
        q, changed = _project_q(posture.joint_pos_target, reserve_limits)
        q_samples.append(q)
        if changed:
            changed_indices.append(index)

    times = tuple(float(knot.t_rel_s) for knot in projected.knots)
    for index, (knot, q) in enumerate(zip(projected.knots, q_samples, strict=True)):
        if len(times) == 1:
            qdot = {joint_id: 0.0 for joint_id in ordered_ids}
        else:
            left = max(0, index - 1)
            right = min(len(times) - 1, index + 1)
            duration = times[right] - times[left]
            if duration <= 0.0:
                raise SchemaValidationError(
                    "R1 projected trajectory times must be increasing"
                )
            qdot = {
                joint_id: (q_samples[right][joint_id] - q_samples[left][joint_id])
                / duration
                for joint_id in ordered_ids
            }
        old_posture = knot.posture_target
        assert old_posture is not None
        knot.posture_target = PostureTarget(
            joint_pos_target=q,
            joint_vel_target=qdot,
            free_anchor_pose_targets=old_posture.free_anchor_pose_targets,
        )
    projected.validate()
    return projected, tuple(changed_indices), q_samples


def _write_q_samples_to_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    ordered_ids: tuple[str, ...],
    q_samples: Sequence[Mapping[str, float]],
) -> None:
    times = tuple(float(knot.t_rel_s) for knot in trajectory.knots)
    if len(times) != len(q_samples):
        raise SchemaValidationError(
            "R1 projected trajectory and joint sample counts differ"
        )
    for index, (knot, q_source) in enumerate(
        zip(trajectory.knots, q_samples, strict=True)
    ):
        q = {joint_id: float(q_source[joint_id]) for joint_id in ordered_ids}
        if len(times) == 1:
            qdot = {joint_id: 0.0 for joint_id in ordered_ids}
        else:
            left = max(0, index - 1)
            right = min(len(times) - 1, index + 1)
            duration = times[right] - times[left]
            if duration <= 0.0:
                raise SchemaValidationError(
                    "R1 projected trajectory times must be increasing"
                )
            qdot = {
                joint_id: (
                    float(q_samples[right][joint_id]) - float(q_samples[left][joint_id])
                )
                / duration
                for joint_id in ordered_ids
            }
        old_posture = knot.posture_target
        assert old_posture is not None
        knot.posture_target = PostureTarget(
            joint_pos_target=q,
            joint_vel_target=qdot,
            free_anchor_pose_targets=old_posture.free_anchor_pose_targets,
        )
    trajectory.validate()


def _centroidal_pose(knot: InteractionKnot) -> Pose7D:
    centroidal = knot.centroidal_target
    if (
        centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise SchemaValidationError(
            "R1 joint-reserve recheck requires a complete centroidal target"
        )
    return (
        *[float(value) for value in centroidal.com_pos_world],
        *[float(value) for value in centroidal.body_orientation_world],
    )


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _batch_task_residuals(
    *,
    solver,
    morphology,
    q_samples: Sequence[Mapping[str, float]],
    target_coms: np.ndarray,
    target_poses_by_anchor: Mapping[int, Sequence[Pose7D]],
    orientation: tuple[float, ...],
    references,
    attitude_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, anchor_rotations, anchor_positions, com_positions = (
        solver.kinematics._evaluate_batch(
            morphology=morphology,
            q_samples=q_samples,
            base_pose_world=(0.0, 0.0, 0.0, *orientation),
            references=references,
        )
    )
    translation = target_coms - np.asarray(com_positions, dtype=np.float64)
    residual_parts: list[np.ndarray] = []
    position_errors: list[np.ndarray] = []
    attitude_errors: list[np.ndarray] = []
    for anchor_id in sorted(target_poses_by_anchor):
        targets = target_poses_by_anchor[anchor_id]
        target_positions = np.asarray([pose[:3] for pose in targets], dtype=np.float64)
        actual_positions = (
            np.asarray(anchor_positions[anchor_id], dtype=np.float64) + translation
        )
        position_residual = target_positions - actual_positions
        target_rotations = np.asarray(
            [quat_to_matrix(tuple(pose[3:7])) for pose in targets],
            dtype=np.float64,
        )
        actual_rotations = np.asarray(anchor_rotations[anchor_id], dtype=np.float64)
        rotation_delta = target_rotations @ np.swapaxes(actual_rotations, -1, -2)
        attitude_residual = _rotation_log(rotation_delta)
        residual_parts.extend(
            (position_residual, float(attitude_weight) * attitude_residual)
        )
        position_errors.append(np.linalg.norm(position_residual, axis=1))
        attitude_errors.append(np.linalg.norm(attitude_residual, axis=1))
    if not residual_parts:
        count = len(q_samples)
        return (
            np.zeros((count, 0), dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        )
    return (
        np.concatenate(residual_parts, axis=1),
        np.max(np.stack(position_errors, axis=1), axis=1),
        np.max(np.stack(attitude_errors, axis=1), axis=1),
    )


def _refine_projected_q_batch(
    *,
    solver,
    morphology,
    physical_model: PhysicalModel,
    trajectory: ContactWrenchTrajectory,
    q_samples: list[dict[str, float]],
    changed_indices: Sequence[int],
    ordered_ids: tuple[str, ...],
    reserve_limits: Mapping[str, tuple[float, float]],
    position_tolerance_m: float,
    attitude_tolerance_rad: float,
    maximum_iterations: int = 60,
    allow_branch_fallback: bool = True,
) -> tuple[int, ...]:
    """Restore task-space accuracy with one vectorized bounded IK pass."""

    grouped: dict[tuple[tuple[float, ...], tuple[int, ...]], list[int]] = {}
    for index in changed_indices:
        knot = trajectory.knots[index]
        posture = knot.posture_target
        targets = (
            {}
            if posture is None or posture.free_anchor_pose_targets is None
            else posture.free_anchor_pose_targets
        )
        if targets:
            centroidal = _centroidal_pose(knot)
            key = (
                tuple(float(value) for value in centroidal[3:7]),
                tuple(sorted(int(value) for value in targets)),
            )
            grouped.setdefault(key, []).append(index)

    lower = np.asarray(
        [reserve_limits[joint_id][0] for joint_id in ordered_ids],
        dtype=np.float64,
    )
    upper = np.asarray(
        [reserve_limits[joint_id][1] for joint_id in ordered_ids],
        dtype=np.float64,
    )
    finite_difference_step = 1.0e-5
    damping = 2.0e-3
    continuity_weight = 1.0e-5
    # Scale the two residual families by their admission tolerances.  The
    # generic IK's 0.35 posture preference can over-refine an already-small
    # attitude error while leaving position just outside R1's hard gate.
    attitude_weight = min(
        1.0,
        float(position_tolerance_m) / float(attitude_tolerance_rad),
    )

    for (orientation, anchor_ids), indices in grouped.items():
        references = resolve_mesh_backed_anchor_references(
            morphology, physical_model, anchor_ids
        )
        active = list(indices)
        for _iteration in range(maximum_iterations):
            if not active:
                break
            active_q = [q_samples[index] for index in active]
            target_coms = np.asarray(
                [_centroidal_pose(trajectory.knots[index])[:3] for index in active],
                dtype=np.float64,
            )
            target_poses = {
                anchor_id: [
                    trajectory.knots[index].posture_target.free_anchor_pose_targets[
                        anchor_id
                    ]
                    for index in active
                ]
                for anchor_id in anchor_ids
            }
            residual, position_error, attitude_error = _batch_task_residuals(
                solver=solver,
                morphology=morphology,
                q_samples=active_q,
                target_coms=target_coms,
                target_poses_by_anchor=target_poses,
                orientation=orientation,
                references=references,
                attitude_weight=attitude_weight,
            )
            converged = (position_error <= float(position_tolerance_m)) & (
                attitude_error <= float(attitude_tolerance_rad)
            )
            if np.all(converged):
                active = []
                break
            solve_rows = np.flatnonzero(~converged)
            solve_indices = [active[int(row)] for row in solve_rows]
            solve_q = [q_samples[index] for index in solve_indices]
            solve_coms = target_coms[solve_rows]
            solve_targets = {
                anchor_id: [target_poses[anchor_id][int(row)] for row in solve_rows]
                for anchor_id in anchor_ids
            }
            base_residual = residual[solve_rows]
            q_matrix = np.asarray(
                [[sample[joint_id] for joint_id in ordered_ids] for sample in solve_q],
                dtype=np.float64,
            )
            perturbations: list[dict[str, float]] = []
            steps = np.zeros((len(solve_q), len(ordered_ids)), dtype=np.float64)
            for column, joint_id in enumerate(ordered_ids):
                for row, sample in enumerate(solve_q):
                    direction = (
                        1.0
                        if q_matrix[row, column] + finite_difference_step
                        <= upper[column]
                        else -1.0
                    )
                    step = direction * finite_difference_step
                    perturbed = dict(sample)
                    perturbed[joint_id] = float(
                        np.clip(
                            q_matrix[row, column] + step,
                            lower[column],
                            upper[column],
                        )
                    )
                    steps[row, column] = perturbed[joint_id] - q_matrix[row, column]
                    perturbations.append(perturbed)
            repeated_coms = np.tile(solve_coms, (len(ordered_ids), 1))
            repeated_targets = {
                anchor_id: solve_targets[anchor_id] * len(ordered_ids)
                for anchor_id in anchor_ids
            }
            perturbed_residual, _, _ = _batch_task_residuals(
                solver=solver,
                morphology=morphology,
                q_samples=perturbations,
                target_coms=repeated_coms,
                target_poses_by_anchor=repeated_targets,
                orientation=orientation,
                references=references,
                attitude_weight=attitude_weight,
            )
            jacobian = (
                perturbed_residual.reshape(
                    len(ordered_ids), len(solve_q), -1
                ).transpose(1, 2, 0)
                - base_residual[:, :, None]
            ) / steps[:, None, :]
            normal = np.einsum("bdi,bdj->bij", jacobian, jacobian)
            normal += (damping**2 + continuity_weight) * np.eye(len(ordered_ids))[
                None, :, :
            ]
            right = -np.einsum("bdi,bd->bi", jacobian, base_residual)
            try:
                delta = np.linalg.solve(normal, right)
            except np.linalg.LinAlgError:
                delta = np.stack(
                    [
                        np.linalg.lstsq(normal[row], right[row], rcond=None)[0]
                        for row in range(len(solve_q))
                    ]
                )
            delta = np.clip(delta, -0.20, 0.20)
            q_matrix = np.clip(q_matrix + delta, lower, upper)
            for row, knot_index in enumerate(solve_indices):
                q_samples[knot_index] = {
                    joint_id: float(q_matrix[row, column])
                    for column, joint_id in enumerate(ordered_ids)
                }
            active = solve_indices
        if active and allow_branch_fallback:
            fallback_indices = []
            for index, knot in enumerate(trajectory.knots):
                posture = knot.posture_target
                targets = (
                    {}
                    if posture is None or posture.free_anchor_pose_targets is None
                    else posture.free_anchor_pose_targets
                )
                if not targets:
                    continue
                centroidal = _centroidal_pose(knot)
                key = (
                    tuple(float(value) for value in centroidal[3:7]),
                    tuple(sorted(int(value) for value in targets)),
                )
                if key == (orientation, anchor_ids):
                    fallback_indices.append(index)
            branch_candidates: list[tuple[float, float, list[dict[str, float]]]] = []
            saturated_ids = tuple(
                joint_id
                for joint_id in ordered_ids
                if any(
                    min(
                        abs(q_samples[index][joint_id] - reserve_limits[joint_id][0]),
                        abs(q_samples[index][joint_id] - reserve_limits[joint_id][1]),
                    )
                    <= 1.0e-7
                    for index in active
                )
            )
            global_seeds = _joint_limit_branch_seeds(reserve_limits)
            branch_initializers: list[tuple[str, Mapping[str, float] | None]] = [
                ("saturated_center", None),
                ("saturated_opposite_quarter", None),
                *[(f"global_{index}", seed) for index, seed in enumerate(global_seeds)],
            ]
            for initializer_name, global_seed in branch_initializers:
                branch_q = [dict(sample) for sample in q_samples]
                for index in fallback_indices:
                    if global_seed is not None:
                        branch_q[index] = dict(global_seed)
                        continue
                    for joint_id in saturated_ids:
                        bound_lower, bound_upper = reserve_limits[joint_id]
                        if initializer_name == "saturated_center":
                            value = 0.5 * (bound_lower + bound_upper)
                        elif abs(q_samples[index][joint_id] - bound_lower) <= abs(
                            q_samples[index][joint_id] - bound_upper
                        ):
                            value = bound_lower + 0.75 * (bound_upper - bound_lower)
                        else:
                            value = bound_lower + 0.25 * (bound_upper - bound_lower)
                        branch_q[index][joint_id] = float(value)
                branch_trajectory = ContactWrenchTrajectory.from_dict(
                    trajectory.to_dict()
                )
                _write_q_samples_to_trajectory(
                    branch_trajectory,
                    ordered_ids=ordered_ids,
                    q_samples=branch_q,
                )
                try:
                    _refine_projected_q_batch(
                        solver=solver,
                        morphology=morphology,
                        physical_model=physical_model,
                        trajectory=branch_trajectory,
                        q_samples=branch_q,
                        changed_indices=fallback_indices,
                        ordered_ids=ordered_ids,
                        reserve_limits=reserve_limits,
                        position_tolerance_m=position_tolerance_m,
                        attitude_tolerance_rad=attitude_tolerance_rad,
                        maximum_iterations=maximum_iterations,
                        allow_branch_fallback=False,
                    )
                except SchemaValidationError:
                    continue
                minimum_inset_fraction = min(
                    min(
                        (branch_q[index][joint_id] - reserve_limits[joint_id][0])
                        / (reserve_limits[joint_id][1] - reserve_limits[joint_id][0]),
                        (reserve_limits[joint_id][1] - branch_q[index][joint_id])
                        / (reserve_limits[joint_id][1] - reserve_limits[joint_id][0]),
                    )
                    for index in fallback_indices
                    for joint_id in ordered_ids
                )
                maximum_step = max(
                    abs(branch_q[index][joint_id] - branch_q[index - 1][joint_id])
                    for index in range(1, len(branch_q))
                    for joint_id in ordered_ids
                )
                branch_candidates.append(
                    (minimum_inset_fraction, -maximum_step, branch_q)
                )
            if branch_candidates:
                _, _, selected_q = max(
                    branch_candidates, key=lambda item: (item[0], item[1])
                )
                q_samples[:] = [dict(sample) for sample in selected_q]
                _write_q_samples_to_trajectory(
                    trajectory,
                    ordered_ids=ordered_ids,
                    q_samples=q_samples,
                )
                changed_indices = tuple(
                    sorted(set(changed_indices) | set(fallback_indices))
                )
                active = []
                continue
        if active:
            remaining_q = [q_samples[index] for index in active]
            remaining_coms = np.asarray(
                [_centroidal_pose(trajectory.knots[index])[:3] for index in active],
                dtype=np.float64,
            )
            remaining_targets = {
                anchor_id: [
                    trajectory.knots[index].posture_target.free_anchor_pose_targets[
                        anchor_id
                    ]
                    for index in active
                ]
                for anchor_id in anchor_ids
            }
            _, remaining_position, remaining_attitude = _batch_task_residuals(
                solver=solver,
                morphology=morphology,
                q_samples=remaining_q,
                target_coms=remaining_coms,
                target_poses_by_anchor=remaining_targets,
                orientation=orientation,
                references=references,
                attitude_weight=attitude_weight,
            )
            raise SchemaValidationError(
                "R1 batch joint-reserve IK could not restore task-space "
                f"accuracy for {len(active)} changed knots "
                f"(indices={active[:8]}, "
                f"maximum_position_error_m={max(remaining_position):.6g}, "
                "maximum_attitude_error_rad="
                f"{max(remaining_attitude):.6g})"
            )

    _write_q_samples_to_trajectory(
        trajectory,
        ordered_ids=ordered_ids,
        q_samples=q_samples,
    )
    return tuple(sorted(set(changed_indices)))


def _recheck_anchor_targets_batch(
    *,
    solver,
    morphology,
    physical_model: PhysicalModel,
    trajectory: ContactWrenchTrajectory,
    q_samples: Sequence[Mapping[str, float]],
    changed_indices: Sequence[int],
    position_tolerance_m: float,
    attitude_tolerance_rad: float,
) -> tuple[float, float]:
    maximum_position_error = 0.0
    maximum_attitude_error = 0.0
    groups: dict[tuple[tuple[float, ...], tuple[int, ...]], list[int]] = {}
    for index in changed_indices:
        knot = trajectory.knots[index]
        posture = knot.posture_target
        targets = (
            {}
            if posture is None or posture.free_anchor_pose_targets is None
            else posture.free_anchor_pose_targets
        )
        if not targets:
            continue
        centroidal = _centroidal_pose(knot)
        key = (
            tuple(float(value) for value in centroidal[3:7]),
            tuple(sorted(int(value) for value in targets)),
        )
        groups.setdefault(key, []).append(index)

    for (orientation, anchor_ids), indices in groups.items():
        references = resolve_mesh_backed_anchor_references(
            morphology,
            physical_model,
            anchor_ids,
        )
        _, _, anchor_rotations, anchor_positions, com_positions = (
            solver.kinematics._evaluate_batch(
                morphology=morphology,
                q_samples=[q_samples[index] for index in indices],
                base_pose_world=(0.0, 0.0, 0.0, *orientation),
                references=references,
            )
        )
        anchor_poses = {
            anchor_id: _poses_from_arrays(
                anchor_rotations[anchor_id], anchor_positions[anchor_id]
            )
            for anchor_id in anchor_ids
        }
        for batch_index, knot_index in enumerate(indices):
            knot = trajectory.knots[knot_index]
            posture = knot.posture_target
            assert posture is not None
            targets = posture.free_anchor_pose_targets or {}
            target_com = _centroidal_pose(knot)[:3]
            translation = tuple(
                float(target_com[axis]) - float(com_positions[batch_index, axis])
                for axis in range(3)
            )
            for anchor_id in anchor_ids:
                pose = anchor_poses[anchor_id][batch_index]
                actual = (
                    float(pose[0]) + translation[0],
                    float(pose[1]) + translation[1],
                    float(pose[2]) + translation[2],
                    *pose[3:7],
                )
                target = targets[anchor_id]
                position_error = _norm(
                    float(target[axis]) - float(actual[axis]) for axis in range(3)
                )
                attitude_error = _norm(_pose_rotation_error_vector(target, actual))
                maximum_position_error = max(maximum_position_error, position_error)
                maximum_attitude_error = max(maximum_attitude_error, attitude_error)
    if maximum_position_error > float(position_tolerance_m) + 1.0e-12:
        raise SchemaValidationError(
            "R1 joint-reserve projection changed an anchor beyond the "
            "position tolerance "
            f"({maximum_position_error:.6g} > {position_tolerance_m:.6g} m)"
        )
    if maximum_attitude_error > float(attitude_tolerance_rad) + 1.0e-12:
        raise SchemaValidationError(
            "R1 joint-reserve projection changed an anchor beyond the "
            "attitude tolerance "
            f"({maximum_attitude_error:.6g} > "
            f"{attitude_tolerance_rad:.6g} rad)"
        )
    return maximum_position_error, maximum_attitude_error


def _recheck_changed_collisions(
    *,
    solver,
    context: HighLevelPolicyContext,
    collision_object,
    trajectory: ContactWrenchTrajectory,
    changed_indices: Sequence[int],
    collision_margin_m: float,
) -> tuple[float | None, int]:
    clearances: list[float] = []
    maximum_violating_pairs = 0
    for index in changed_indices:
        knot = trajectory.knots[index]
        solver.set_collision_scene(
            morphology=context.morphology_graph,
            object_pose_world=_collision_object_pose(
                context=context,
                knot=knot,
                collision_object=collision_object,
            ),
            object_size_m=collision_object.size_m,
            allowed_anchor_ids=_allowed_object_contact_anchors(
                context=context,
                knot=knot,
                object_id=collision_object.object_id,
            ),
        )
        scenes = [(collision_object.object_id, None, None, None)]
        scenes.extend(
            (box.box_id, box.pose_world, box.size_m, ())
            for box in collision_object.environment_boxes
        )
        posture = knot.posture_target
        assert posture is not None and posture.joint_pos_target is not None
        for obstacle_id, pose, size, allowed in scenes:
            if pose is not None:
                solver.set_collision_scene(
                    morphology=context.morphology_graph,
                    object_pose_world=pose,
                    object_size_m=size,
                    allowed_anchor_ids=allowed,
                )
            result = solver.check_configuration(
                morphology=context.morphology_graph,
                centroidal_pose_world=_centroidal_pose(knot),
                joint_positions_rad=posture.joint_pos_target,
                exact=False,
                margin_m=float(collision_margin_m),
                ground_plane_z_m=collision_object.ground_plane_z_m,
            )
            clearances.append(float(result["minimum_clearance_m"]))
            maximum_violating_pairs = max(
                maximum_violating_pairs,
                int(result["violating_pair_count"]),
            )
            if result.get("accepted") is not True:
                raise SchemaValidationError(
                    "R1 joint-reserve projection failed collision recheck "
                    f"at knot {index} against {obstacle_id!r} "
                    f"(clearance={result['minimum_clearance_m']:.6g}, "
                    f"pairs={result['violating_pair_count']})"
                )
    return (min(clearances) if clearances else None), maximum_violating_pairs


def project_order9_r1_nominal_joint_reserve(
    nominal: Order9C3NominalTrajectory,
    *,
    task_spec: TaskSpec,
    physical_model: PhysicalModel,
    minimum_normalized_joint_limit_reserve: float,
    anchor_position_tolerance_m: float = ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M,
    anchor_attitude_tolerance_rad: float = 0.10,
    collision_margin_m: float = 0.005,
) -> Order9C3NominalTrajectory:
    """Project a complete nominal path once and recheck only changed knots."""

    reserve = float(minimum_normalized_joint_limit_reserve)
    if not math.isfinite(reserve) or not 0.0 <= reserve < 0.5:
        raise ValueError("minimum_normalized_joint_limit_reserve must be in [0, 0.5)")
    if (
        not math.isfinite(float(anchor_position_tolerance_m))
        or not 0.0 < float(anchor_position_tolerance_m) <= 0.10
    ):
        raise ValueError("anchor_position_tolerance_m must be finite and in (0, 0.10]")
    if reserve == 0.0:
        return nominal
    morphology = nominal.selection_bundle.design_output.target_morphology
    ordered_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    reserve_limits = _joint_limits_with_normalized_reserve(
        _global_joint_limits(morphology, physical_model, ordered_ids),
        reserve,
    )
    requires_projection = False
    for window in nominal.windows:
        for knot in window.plan.trajectory.knots:
            posture = knot.posture_target
            if posture is None or posture.joint_pos_target is None:
                raise SchemaValidationError(
                    "R1 joint-reserve projection requires resolved joint targets"
                )
            _, changed = _project_q(posture.joint_pos_target, reserve_limits)
            if changed:
                requires_projection = True
                break
        if requires_projection:
            break
    if not requires_projection:
        return nominal

    projected_data = [
        _trajectory_with_projected_q(
            window.plan.trajectory,
            ordered_ids=ordered_ids,
            reserve_limits=reserve_limits,
        )
        for window in nominal.windows
    ]
    if not any(changed for _, changed, _ in projected_data):
        return nominal

    collision_object = build_order9_c3_posture_collision_object(task_spec)
    solver = _default_posture_ik_solver(
        physical_model,
        collision_object=collision_object,
        prefer_native=True,
        require_native=True,
        nominal_collision_pair_manifest=None,
        ik_config=CentroidalPostureIKConfig(
            anchor_position_tolerance_m=float(anchor_position_tolerance_m),
            anchor_attitude_tolerance_rad=float(anchor_attitude_tolerance_rad),
            use_relaxed_seed=False,
        ),
        collision_margin_m=float(collision_margin_m),
    )
    built = IRGBuilder().build_with_scene_graph(task_spec)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    observation = build_order9_c3_neutral_runtime_observation(
        morphology,
        physical_model,
        task_spec,
        phase_label=(nominal.windows[0].phase if nominal.windows else None),
    )
    windows: list[Order9C3NominalWindow] = []

    for window_index, (window, projected_item) in enumerate(
        zip(nominal.windows, projected_data, strict=True)
    ):
        window_projection_start = perf_counter()
        trajectory, changed_indices, q_samples = projected_item
        context = HighLevelPolicyContext(
            built.irg,
            envelope,
            morphology,
            nominal.selection_bundle.contact_candidate_set,
            runtime_observation=observation,
        )
        plan = window.plan
        if changed_indices:
            changed_indices = _refine_projected_q_batch(
                solver=solver,
                morphology=morphology,
                physical_model=physical_model,
                trajectory=trajectory,
                q_samples=q_samples,
                changed_indices=changed_indices,
                ordered_ids=ordered_ids,
                reserve_limits=reserve_limits,
                position_tolerance_m=anchor_position_tolerance_m,
                attitude_tolerance_rad=anchor_attitude_tolerance_rad,
            )
            position_error, attitude_error = _recheck_anchor_targets_batch(
                solver=solver,
                morphology=morphology,
                physical_model=physical_model,
                trajectory=trajectory,
                q_samples=q_samples,
                changed_indices=changed_indices,
                position_tolerance_m=anchor_position_tolerance_m,
                attitude_tolerance_rad=anchor_attitude_tolerance_rad,
            )
            minimum_clearance, violating_pairs = _recheck_changed_collisions(
                solver=solver,
                context=context,
                collision_object=collision_object,
                trajectory=trajectory,
                changed_indices=changed_indices,
                collision_margin_m=collision_margin_m,
            )
            evidence = plan.posture_resolution.evidence
            times = tuple(float(knot.t_rel_s) for knot in trajectory.knots)
            maximum_rate, maximum_segment, required_duration = (
                _maximum_timed_joint_rate(
                    times,
                    q_samples,
                    ordered_ids,
                    joint_rate_limit_rad_s=evidence.joint_rate_limit_rad_s,
                )
            )
            if maximum_rate > evidence.joint_rate_limit_rad_s + 1.0e-9:
                raise SchemaValidationError(
                    "R1 joint-reserve projection exceeds the joint-rate limit "
                    f"({maximum_rate:.6g} > "
                    f"{evidence.joint_rate_limit_rad_s:.6g} rad/s)"
                )
            combined_clearance = minimum_clearance
            if evidence.minimum_collision_clearance_m is not None:
                combined_clearance = (
                    evidence.minimum_collision_clearance_m
                    if combined_clearance is None
                    else min(
                        evidence.minimum_collision_clearance_m,
                        combined_clearance,
                    )
                )
            projected_evidence = replace(
                evidence,
                resolver_version=(
                    f"{evidence.resolver_version}+"
                    f"{ORDER9_R1_JOINT_RESERVE_PROJECTION_VERSION}"
                ),
                resolved_trajectory_hash=stable_hash(trajectory.to_dict()),
                maximum_anchor_position_error_m=max(
                    evidence.maximum_anchor_position_error_m,
                    position_error,
                ),
                maximum_anchor_attitude_error_rad=max(
                    evidence.maximum_anchor_attitude_error_rad,
                    attitude_error,
                ),
                maximum_joint_rate_rad_s=maximum_rate,
                maximum_joint_rate_segment_index=maximum_segment,
                minimum_required_segment_duration_s=required_duration,
                minimum_joint_rate_margin_rad_s=(
                    evidence.joint_rate_limit_rad_s - maximum_rate
                ),
                solve_wall_time_s=(
                    evidence.solve_wall_time_s
                    + (perf_counter() - window_projection_start)
                ),
                minimum_collision_clearance_m=combined_clearance,
                maximum_collision_violating_pair_count=max(
                    evidence.maximum_collision_violating_pair_count,
                    violating_pairs,
                ),
            )
            resolution = Order9ResolvedPostureTrajectory(
                raw_trajectory=plan.posture_resolution.raw_trajectory,
                trajectory=trajectory,
                evidence=projected_evidence,
            )
            plan = replace(
                plan,
                trajectory=trajectory,
                posture_resolution=resolution,
            )
        projected_window = replace(window, plan=plan)
        windows.append(projected_window)
        next_phase = (
            nominal.windows[window_index + 1].phase
            if window_index + 1 < len(nominal.windows)
            else window.phase
        )
        observation = _ideal_endpoint_observation(
            context=context,
            plan=plan,
            physical_model=physical_model,
            next_phase=next_phase,
        )

    result = replace(
        nominal,
        windows=tuple(windows),
        timeline=_flatten_nominal_windows(windows),
        final_observation=observation,
    )
    return result


__all__ = [
    "ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M",
    "ORDER9_R1_JOINT_RESERVE_PROJECTION_VERSION",
    "project_order9_r1_nominal_joint_reserve",
]
