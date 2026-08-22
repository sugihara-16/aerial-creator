from __future__ import annotations

"""Bounded execution-side contact lead for Order 9 nominal IK replay.

The high-level contact pose keeps its physical surface semantics.  This module
only converts a small, inward virtual target displacement into a nearby joint
servo target.  It never reads contact force, collision truth, or any other
simulator-only signal and therefore remains deployable with the production IK
resolver.
"""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKSolver,
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.physical_model_builder import PhysicalModel
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import InteractionKnot
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.simulation.order9_tensor_object_task import (
    Order9TensorObjectTaskTarget,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
)


ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION = (
    "order9_surface_semantic_bounded_virtual_contact_ik_v4_bounded_refinement_achieved_lead"
)
ORDER9_CONTACT_COMPRESSION_COLLISION_SHIELD_VERSION = (
    "order9_convex_plus_local_joint_trust_contact_compression_shield_v3"
)
# Continuous real-mesh replay established that the contact path, rather than
# the compression residual, caused the earlier gimbal/object contact.  After
# retiming that path, 60 mrad admitted the checkpoint's full requested local
# correction and maintained both contacts through lift without collision.
ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD = 0.060
_ORDER9_VIRTUAL_CONTACT_CONTINUATION_STEP_M = 0.002
_ORDER9_VIRTUAL_CONTACT_LINE_SEARCH_STEPS = 12
_ORDER9_VIRTUAL_CONTACT_OBJECTIVE_EPSILON = 1.0e-14


@dataclass(frozen=True)
class Order9VirtualContactCompressionConfig:
    inward_lead_m: float
    damping: float = 2.0e-2
    maximum_iterations: int = 12
    maximum_joint_step_rad: float = 0.04
    maximum_joint_delta_rad: float = 0.15
    convergence_tolerance_m: float = 2.5e-4

    def __post_init__(self) -> None:
        if not math.isfinite(self.inward_lead_m) or self.inward_lead_m < 0.0:
            raise ValueError("virtual contact inward lead must be finite and non-negative")
        if not math.isfinite(self.damping) or self.damping <= 0.0:
            raise ValueError("virtual contact damping must be finite and positive")
        if self.maximum_iterations < 1:
            raise ValueError("virtual contact IK iterations must be positive")
        for name in (
            "maximum_joint_step_rad",
            "maximum_joint_delta_rad",
            "convergence_tolerance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"virtual contact {name} must be finite and positive")


@dataclass(frozen=True)
class Order9VirtualContactCompressionSolution:
    inward_lead_m: float
    joint_delta_rad: Mapping[str, float]
    achieved_inward_displacement_m: Mapping[int, float]
    tangential_error_m: Mapping[int, float]
    maximum_joint_delta_rad: float
    iterations: int
    saturated: bool
    version: str = ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION


def solve_order9_virtual_contact_compression_for_achieved_lead(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    contact_knot: InteractionKnot,
    candidate_set: ContactCandidateSet,
    minimum_achieved_inward_lead_m: float,
    maximum_requested_inward_lead_m: float,
    requested_lead_quantization_m: float,
) -> Order9VirtualContactCompressionSolution:
    """Refine the IK request until every anchor achieves the physical lead.

    The local IK generally achieves slightly less Cartesian displacement than
    requested because multiple contacts and joint regularization are solved
    together.  Actuator compliance is a physical lower bound on *achieved*
    anchor displacement, so compensate that deterministic IK transfer loss
    before applying the collision shield.
    """

    required = float(minimum_achieved_inward_lead_m)
    maximum = float(maximum_requested_inward_lead_m)
    quantum = float(requested_lead_quantization_m)
    if (
        not math.isfinite(required)
        or required <= 0.0
        or not math.isfinite(maximum)
        or maximum < required
        or not math.isfinite(quantum)
        or quantum <= 0.0
    ):
        raise ValueError("achieved-lead virtual contact bounds are invalid")

    def quantize_up(value: float) -> float:
        return min(
            maximum,
            quantum * math.ceil((value - 1.0e-12) / quantum),
        )

    requested = quantize_up(required)
    last_solution: Order9VirtualContactCompressionSolution | None = None
    for _ in range(8):
        solution = solve_order9_virtual_contact_compression(
            morphology=morphology,
            physical_model=physical_model,
            contact_knot=contact_knot,
            candidate_set=candidate_set,
            config=Order9VirtualContactCompressionConfig(
                inward_lead_m=requested
            ),
        )
        last_solution = solution
        achieved = min(
            float(value)
            for value in solution.achieved_inward_displacement_m.values()
        )
        if achieved + 1.0e-9 >= required:
            return solution
        if requested >= maximum - 1.0e-12:
            break
        ratio = required / max(achieved, 1.0e-9)
        requested = quantize_up(
            max(requested + quantum, requested * ratio)
        )
    assert last_solution is not None
    return last_solution


@dataclass(frozen=True)
class Order9ContactCompressionCollisionLimit:
    maximum_action_scale: float
    collision_check_count: int
    limiting_scene_id: str | None
    limiting_result: Mapping[str, object] | None
    collision_margin_m: float
    maximum_action_joint_delta_rad: float
    bisection_iterations: int
    version: str = ORDER9_CONTACT_COMPRESSION_COLLISION_SHIELD_VERSION


def solve_order9_collision_safe_compression_scale(
    evaluator: Callable[[float], tuple[bool, str | None, Mapping[str, object] | None]],
    *,
    collision_margin_m: float = 0.001,
    maximum_scale: float = 1.0,
    maximum_action_joint_delta_rad: float = (
        ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD
    ),
    bisection_iterations: int = 12,
) -> Order9ContactCompressionCollisionLimit:
    """Find the largest safe multiplier for one local compression direction."""

    if not math.isfinite(collision_margin_m) or collision_margin_m <= 0.0:
        raise ValueError("compression collision margin must be finite and positive")
    if not math.isfinite(maximum_scale) or not 0.0 < maximum_scale <= 1.0:
        raise ValueError("compression maximum scale must lie in (0, 1]")
    if (
        not math.isfinite(maximum_action_joint_delta_rad)
        or maximum_action_joint_delta_rad <= 0.0
    ):
        raise ValueError(
            "compression maximum action joint delta must be finite and positive"
        )
    if bisection_iterations < 1:
        raise ValueError("compression collision bisection iterations must be positive")
    check_count = 0

    def evaluate(scale: float):
        nonlocal check_count
        check_count += 1
        return evaluator(scale)

    full_safe, full_scene, full_result = evaluate(maximum_scale)
    if full_safe:
        return Order9ContactCompressionCollisionLimit(
            maximum_action_scale=maximum_scale,
            collision_check_count=check_count,
            limiting_scene_id=(
                None if maximum_scale == 1.0 else "joint_trust_region"
            ),
            limiting_result=(
                None
                if maximum_scale == 1.0
                else {"maximum_action_scale": maximum_scale}
            ),
            collision_margin_m=collision_margin_m,
            maximum_action_joint_delta_rad=maximum_action_joint_delta_rad,
            bisection_iterations=bisection_iterations,
        )
    zero_safe, zero_scene, zero_result = evaluate(0.0)
    if not zero_safe:
        raise ValueError(
            "accepted nominal posture is not collision-free before compression: "
            f"scene={zero_scene!r}, result={zero_result!r}"
        )
    lower = 0.0
    upper = maximum_scale
    limiting_scene = full_scene
    limiting_result = full_result
    for _ in range(bisection_iterations):
        middle = 0.5 * (lower + upper)
        safe, scene, result = evaluate(middle)
        if safe:
            lower = middle
        else:
            upper = middle
            limiting_scene = scene
            limiting_result = result
    return Order9ContactCompressionCollisionLimit(
        maximum_action_scale=lower,
        collision_check_count=check_count,
        limiting_scene_id=limiting_scene,
        limiting_result=limiting_result,
        collision_margin_m=collision_margin_m,
        maximum_action_joint_delta_rad=maximum_action_joint_delta_rad,
        bisection_iterations=bisection_iterations,
    )


def limit_order9_virtual_contact_compression_by_collision(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    contact_knot: InteractionKnot,
    solution: Order9VirtualContactCompressionSolution,
    collision_object: Order9PostureCollisionObject,
    collision_solver,
    collision_margin_m: float = 0.001,
    maximum_action_joint_delta_rad: float = (
        ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD
    ),
    bisection_iterations: int = 12,
) -> Order9ContactCompressionCollisionLimit:
    """Clip a pi_L compression direction against object/support/self proxies.

    The check runs once when a morphology/nominal plan is installed. Selected
    anchor penetration into the grasped object is task-allowed; every other
    object, support, ground, and self collision remains prohibited.
    """

    posture = contact_knot.posture_target
    centroidal = contact_knot.centroidal_target
    if (
        posture is None
        or posture.joint_pos_target is None
        or centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise ValueError("compression collision shield requires a resolved posture")
    centroidal_pose: Pose7D = (
        *tuple(float(value) for value in centroidal.com_pos_world),
        *tuple(float(value) for value in centroidal.body_orientation_world),
    )
    reference_q = {
        joint_id: float(value)
        for joint_id, value in posture.joint_pos_target.items()
    }
    selected_anchor_ids = tuple(
        sorted(int(value.anchor_id) for value in contact_knot.contact_assignments)
    )
    scenes = (
        (
            collision_object.object_id,
            collision_object.initial_pose_world,
            collision_object.size_m,
            selected_anchor_ids,
            True,
        ),
        *tuple(
            (box.box_id, box.pose_world, box.size_m, (), False)
            for box in collision_object.environment_boxes
        ),
    )

    def evaluator(scale: float):
        q = {
            joint_id: reference_q[joint_id]
            + scale * float(solution.joint_delta_rad.get(joint_id, 0.0))
            for joint_id in reference_q
        }
        for scene_id, pose, size, allowed, selected_contact_scene in scenes:
            collision_solver.set_collision_scene(
                morphology=morphology,
                object_pose_world=pose,
                object_size_m=size,
                allowed_anchor_ids=allowed,
            )
            result = collision_solver.check_configuration(
                morphology=morphology,
                centroidal_pose_world=centroidal_pose,
                joint_positions_rad=q,
                exact=False,
                margin_m=collision_margin_m,
                ground_plane_z_m=collision_object.ground_plane_z_m,
            )
            if not _compression_collision_result_accepted(
                result,
                selected_contact_scene=selected_contact_scene,
                collision_margin_m=collision_margin_m,
            ):
                return False, scene_id, result
        return True, None, None

    maximum_scale = min(
        1.0,
        maximum_action_joint_delta_rad
        / max(float(solution.maximum_joint_delta_rad), 1.0e-12),
    )
    return solve_order9_collision_safe_compression_scale(
        evaluator,
        collision_margin_m=collision_margin_m,
        maximum_scale=maximum_scale,
        maximum_action_joint_delta_rad=maximum_action_joint_delta_rad,
        bisection_iterations=bisection_iterations,
    )


def _compression_collision_result_accepted(
    result: Mapping[str, object],
    *,
    selected_contact_scene: bool,
    collision_margin_m: float,
) -> bool:
    if result.get("accepted") is True:
        return True
    if not selected_contact_scene:
        return False
    if int(result.get("selected_contact_penetration_violating_pair_count", 0)) < 1:
        return False
    if int(result.get("ground_violating_proxy_count", 0)) != 0:
        return False
    if int(result.get("colliding_pair_count", 0)) != 0:
        return False
    tolerance = float(result.get("collision_feasibility_tolerance_m", 0.0002))
    minimum_allowed = collision_margin_m - tolerance
    return all(
        pair.get("colliding") is not True
        and float(pair.get("clearance_m", -math.inf)) >= minimum_allowed
        for pair in result.get("worst_pairs", ())
    )


def solve_order9_virtual_contact_compression(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    contact_knot: InteractionKnot,
    candidate_set: ContactCandidateSet,
    config: Order9VirtualContactCompressionConfig,
) -> Order9VirtualContactCompressionSolution:
    """Find the closest bounded q target for an inward surface displacement.

    This is deliberately a local differential IK problem around the reviewed
    nominal grasp.  Unlike the general posture IK, it has no pitch-to-zero
    objective and cannot jump to another articulated branch.
    """

    posture = contact_knot.posture_target
    centroidal = contact_knot.centroidal_target
    if (
        posture is None
        or posture.joint_pos_target is None
        or posture.free_anchor_pose_targets is None
        or centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise ValueError("virtual contact IK requires a resolved contact posture")
    assignments = tuple(contact_knot.contact_assignments)
    if len(assignments) < 2:
        raise ValueError("virtual contact IK requires at least two assignments")
    candidate_by_id = {
        int(candidate.candidate_id): candidate
        for candidate in candidate_set.candidates
    }
    ordered_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    reference_q = {
        joint_id: float(posture.joint_pos_target[joint_id])
        for joint_id in ordered_ids
    }
    q = dict(reference_q)
    selected_anchor_ids = tuple(int(value.anchor_id) for value in assignments)
    references = resolve_mesh_backed_anchor_references(
        morphology, physical_model, selected_anchor_ids
    )
    reference_by_anchor = {
        int(reference.anchor.anchor_id): reference for reference in references
    }
    surfaces: dict[int, np.ndarray] = {}
    normals: dict[int, np.ndarray] = {}
    for assignment in assignments:
        anchor_id = int(assignment.anchor_id)
        candidate = candidate_by_id[int(assignment.candidate_id)]
        normal = np.asarray(candidate.normal_world, dtype=float)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1.0e-9:
            raise ValueError("virtual contact candidate normal is invalid")
        normal /= norm
        surface_pose = posture.free_anchor_pose_targets.get(anchor_id)
        if surface_pose is None:
            raise ValueError("virtual contact posture lacks a selected anchor target")
        surfaces[anchor_id] = np.asarray(surface_pose[:3], dtype=float)
        normals[anchor_id] = normal

    limits = _global_joint_limits(ordered_ids, physical_model)
    solver = CentroidalPostureIKSolver(physical_model)
    centroidal_pose: Pose7D = (
        *tuple(float(value) for value in centroidal.com_pos_world),
        *tuple(float(value) for value in centroidal.body_orientation_world),
    )
    saturated = False
    iterations = 0
    continuation_count = max(
        1,
        int(
            math.ceil(
                float(config.inward_lead_m)
                / _ORDER9_VIRTUAL_CONTACT_CONTINUATION_STEP_M
            )
        ),
    )
    continuation_failed = False
    for continuation_index in range(1, continuation_count + 1):
        stage_lead_m = (
            float(config.inward_lead_m)
            * float(continuation_index)
            / float(continuation_count)
        )
        targets = {
            anchor_id: surfaces[anchor_id] - stage_lead_m * normals[anchor_id]
            for anchor_id in selected_anchor_ids
        }
        stage_converged = False
        for _ in range(config.maximum_iterations):
            iterations += 1
            base_pose = base_pose_for_centroidal_target(
                morphology,
                physical_model,
                q,
                centroidal_pose[:3],
                centroidal_pose[3:7],
                kinematics=solver.kinematics,
            )
            result = solver.kinematics.forward(
                morphology, physical_model, q, base_pose, references
            )
            jacobians = solver._fixed_centroidal_anchor_jacobians(
                morphology=morphology,
                centroidal_pose_world=centroidal_pose,
                q=q,
                limits=limits,
                references=references,
                nominal=result.anchor_poses_world,
            )
            matrix_rows: list[np.ndarray] = []
            residual_rows: list[np.ndarray] = []
            for anchor_id in selected_anchor_ids:
                actual = np.asarray(
                    result.anchor_poses_world[anchor_id][:3], dtype=float
                )
                matrix_rows.append(
                    np.asarray(jacobians[anchor_id][:3], dtype=float)
                )
                residual_rows.append(targets[anchor_id] - actual)
            residual_stack = np.stack(residual_rows)
            residual_norms = np.linalg.norm(residual_stack, axis=1)
            current_maximum_error_m = float(np.max(residual_norms))
            if current_maximum_error_m <= config.convergence_tolerance_m:
                stage_converged = True
                break
            matrix = np.concatenate(matrix_rows, axis=0)
            residual = np.concatenate(residual_rows, axis=0)
            current_objective = float(residual @ residual)
            normal_matrix = matrix.T @ matrix
            normal_matrix += (float(config.damping) ** 2) * np.eye(len(ordered_ids))
            right = matrix.T @ residual
            try:
                step = np.linalg.solve(normal_matrix, right)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(normal_matrix, right, rcond=None)[0]
            step = np.clip(
                step,
                -float(config.maximum_joint_step_rad),
                float(config.maximum_joint_step_rad),
            )
            accepted_q: dict[str, float] | None = None
            accepted_clipped = False
            for line_search_index in range(_ORDER9_VIRTUAL_CONTACT_LINE_SEARCH_STEPS):
                scale = 0.5**line_search_index
                candidate_q: dict[str, float] = {}
                candidate_clipped = False
                for index, joint_id in enumerate(ordered_ids):
                    desired = float(q[joint_id]) + scale * float(step[index])
                    lower_delta = max(
                        -float(config.maximum_joint_delta_rad),
                        limits[joint_id][0] - reference_q[joint_id],
                    )
                    upper_delta = min(
                        float(config.maximum_joint_delta_rad),
                        limits[joint_id][1] - reference_q[joint_id],
                    )
                    clipped_delta = min(
                        max(desired - reference_q[joint_id], lower_delta),
                        upper_delta,
                    )
                    candidate_clipped |= not math.isclose(
                        clipped_delta,
                        desired - reference_q[joint_id],
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    candidate_q[joint_id] = reference_q[joint_id] + clipped_delta
                if max(
                    abs(candidate_q[joint_id] - q[joint_id])
                    for joint_id in ordered_ids
                ) <= 1.0e-10:
                    continue
                candidate_base = base_pose_for_centroidal_target(
                    morphology,
                    physical_model,
                    candidate_q,
                    centroidal_pose[:3],
                    centroidal_pose[3:7],
                    kinematics=solver.kinematics,
                )
                candidate_result = solver.kinematics.forward(
                    morphology,
                    physical_model,
                    candidate_q,
                    candidate_base,
                    references,
                )
                candidate_residuals = np.stack(
                    [
                        targets[anchor_id]
                        - np.asarray(
                            candidate_result.anchor_poses_world[anchor_id][:3],
                            dtype=float,
                        )
                        for anchor_id in selected_anchor_ids
                    ]
                )
                candidate_objective = float(
                    np.ravel(candidate_residuals)
                    @ np.ravel(candidate_residuals)
                )
                if (
                    candidate_objective
                    < current_objective - _ORDER9_VIRTUAL_CONTACT_OBJECTIVE_EPSILON
                ):
                    accepted_q = candidate_q
                    accepted_clipped = candidate_clipped
                    break
            if accepted_q is None:
                break
            q = accepted_q
            saturated |= accepted_clipped
        if not stage_converged:
            continuation_failed = True
            break

    if continuation_failed and float(config.inward_lead_m) > 0.0:
        from scipy.optimize import least_squares

        full_targets = {
            anchor_id: surfaces[anchor_id]
            - float(config.inward_lead_m) * normals[anchor_id]
            for anchor_id in selected_anchor_ids
        }
        active_joint_ids: list[str] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for joint_id in ordered_ids:
            lower = max(
                reference_q[joint_id] - float(config.maximum_joint_delta_rad),
                limits[joint_id][0],
            )
            upper = min(
                reference_q[joint_id] + float(config.maximum_joint_delta_rad),
                limits[joint_id][1],
            )
            if upper - lower > 1.0e-10:
                active_joint_ids.append(joint_id)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

        def refinement_residual(values: np.ndarray) -> np.ndarray:
            candidate_q = dict(q)
            candidate_q.update(
                {
                    joint_id: float(values[index])
                    for index, joint_id in enumerate(active_joint_ids)
                }
            )
            candidate_base = base_pose_for_centroidal_target(
                morphology,
                physical_model,
                candidate_q,
                centroidal_pose[:3],
                centroidal_pose[3:7],
                kinematics=solver.kinematics,
            )
            candidate_result = solver.kinematics.forward(
                morphology,
                physical_model,
                candidate_q,
                candidate_base,
                references,
            )
            return np.concatenate(
                [
                    full_targets[anchor_id]
                    - np.asarray(
                        candidate_result.anchor_poses_world[anchor_id][:3],
                        dtype=float,
                    )
                    for anchor_id in selected_anchor_ids
                ]
            )

        if active_joint_ids:
            initial = np.asarray(
                [q[joint_id] for joint_id in active_joint_ids], dtype=float
            )
            lower_array = np.asarray(lower_bounds, dtype=float)
            upper_array = np.asarray(upper_bounds, dtype=float)
            initial = np.minimum(np.maximum(initial, lower_array), upper_array)
            initial_residual = refinement_residual(initial)
            refined = least_squares(
                refinement_residual,
                initial,
                bounds=(lower_array, upper_array),
                method="trf",
                ftol=1.0e-9,
                xtol=1.0e-9,
                gtol=1.0e-9,
                max_nfev=max(64, config.maximum_iterations * len(active_joint_ids)),
            )
            iterations += int(refined.nfev)
            if (
                np.isfinite(refined.x).all()
                and float(refined.fun @ refined.fun)
                < float(initial_residual @ initial_residual)
                - _ORDER9_VIRTUAL_CONTACT_OBJECTIVE_EPSILON
            ):
                q.update(
                    {
                        joint_id: float(refined.x[index])
                        for index, joint_id in enumerate(active_joint_ids)
                    }
                )

    final_base = base_pose_for_centroidal_target(
        morphology,
        physical_model,
        q,
        centroidal_pose[:3],
        centroidal_pose[3:7],
        kinematics=solver.kinematics,
    )
    final = solver.kinematics.forward(
        morphology, physical_model, q, final_base, references
    )
    achieved: dict[int, float] = {}
    tangential: dict[int, float] = {}
    for assignment in assignments:
        anchor_id = int(assignment.anchor_id)
        surface = np.asarray(
            posture.free_anchor_pose_targets[anchor_id][:3], dtype=float
        )
        displacement = (
            np.asarray(final.anchor_poses_world[anchor_id][:3], dtype=float)
            - surface
        )
        inward = -float(displacement @ normals[anchor_id])
        achieved[anchor_id] = inward
        tangential[anchor_id] = float(
            np.linalg.norm(displacement + inward * normals[anchor_id])
        )
    deltas = {
        joint_id: float(q[joint_id] - reference_q[joint_id])
        for joint_id in ordered_ids
    }
    maximum_delta = max((abs(value) for value in deltas.values()), default=0.0)
    saturated |= maximum_delta >= float(config.maximum_joint_delta_rad) - 1.0e-8
    saturated |= continuation_failed
    return Order9VirtualContactCompressionSolution(
        inward_lead_m=float(config.inward_lead_m),
        joint_delta_rad=deltas,
        achieved_inward_displacement_m=achieved,
        tangential_error_m=tangential,
        maximum_joint_delta_rad=maximum_delta,
        iterations=iterations,
        saturated=saturated,
    )


def order9_virtual_contact_phase_weight(
    phase_index: torch.Tensor,
    phase_progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compression weight and d(weight)/d(progress)."""

    if phase_index.shape != phase_progress.shape:
        raise ValueError("virtual contact phase tensors must have matching shape")
    progress = phase_progress.clamp(0.0, 1.0)
    smooth = progress.square() * (3.0 - 2.0 * progress)
    derivative = 6.0 * progress * (1.0 - progress)
    weight = torch.zeros_like(progress)
    slope = torch.zeros_like(progress)
    contact = phase_index == ORDER9_OBJECT_TASK_PHASES.index(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION
    )
    hold = torch.zeros_like(contact)
    for phase in (
        Order9ObjectTaskPhase.LIFT,
        Order9ObjectTaskPhase.TRANSPORT,
        Order9ObjectTaskPhase.PLACE,
    ):
        hold |= phase_index == ORDER9_OBJECT_TASK_PHASES.index(phase)
    release = phase_index == ORDER9_OBJECT_TASK_PHASES.index(
        Order9ObjectTaskPhase.RELEASE
    )
    weight = torch.where(contact, smooth, weight)
    slope = torch.where(contact, derivative, slope)
    weight = torch.where(hold, torch.ones_like(weight), weight)
    weight = torch.where(release, 1.0 - smooth, weight)
    slope = torch.where(release, -derivative, slope)
    return weight, slope


@torch.no_grad()
def apply_order9_virtual_contact_compression(
    target: Order9TensorObjectTaskTarget,
    *,
    phase_index: torch.Tensor,
    phase_progress: torch.Tensor,
    joint_delta_rad: torch.Tensor,
    phase_duration_s: torch.Tensor,
) -> Order9TensorObjectTaskTarget:
    """Add the bounded virtual-contact target without altering pi_H poses."""

    from dataclasses import replace

    batch = phase_index.shape[0]
    if joint_delta_rad.ndim == 2:
        joint_delta_rad = joint_delta_rad.unsqueeze(0).expand(batch, -1, -1)
    if joint_delta_rad.shape != target.nominal_joint_positions_rad.shape:
        raise ValueError("virtual contact joint-delta tensor shape differs")
    if phase_duration_s.shape != (batch,):
        raise ValueError("virtual contact phase-duration shape differs")
    weight, progress_slope = order9_virtual_contact_phase_weight(
        phase_index, phase_progress
    )
    time_slope = progress_slope / phase_duration_s.clamp_min(1.0e-6)
    return replace(
        target,
        nominal_joint_positions_rad=(
            target.nominal_joint_positions_rad
            + weight[:, None, None] * joint_delta_rad
        ),
        nominal_joint_velocities_radps=(
            target.nominal_joint_velocities_radps
            + time_slope[:, None, None] * joint_delta_rad
        ),
    )


def compression_joint_delta_tensor(
    solution: Order9VirtualContactCompressionSolution,
    *,
    module_ids: Sequence[int],
    local_joint_ids: Sequence[str],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            [
                float(
                    solution.joint_delta_rad.get(
                        f"module_{int(module_id)}:{joint_id}", 0.0
                    )
                )
                for joint_id in local_joint_ids
            ]
            for module_id in module_ids
        ],
        device=device,
        dtype=dtype,
    )


def _global_joint_limits(
    ordered_ids: Sequence[str], physical_model: PhysicalModel
) -> dict[str, tuple[float, float]]:
    local = {
        joint.joint_id: (float(joint.limit_lower), float(joint.limit_upper))
        for joint in physical_model.joints
        if joint.limit_lower is not None and joint.limit_upper is not None
    }
    result = {}
    for joint_id in ordered_ids:
        local_id = joint_id.split(":", 1)[1]
        if local_id not in local:
            raise ValueError(f"virtual contact joint is absent: {joint_id}")
        result[joint_id] = local[local_id]
    return result


__all__ = [
    "ORDER9_CONTACT_COMPRESSION_COLLISION_SHIELD_VERSION",
    "ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD",
    "ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION",
    "Order9ContactCompressionCollisionLimit",
    "Order9VirtualContactCompressionConfig",
    "Order9VirtualContactCompressionSolution",
    "apply_order9_virtual_contact_compression",
    "compression_joint_delta_tensor",
    "limit_order9_virtual_contact_compression_by_collision",
    "order9_virtual_contact_phase_weight",
    "solve_order9_collision_safe_compression_scale",
    "solve_order9_virtual_contact_compression",
]
