from __future__ import annotations

"""Production hybrid physics backend for high-level trajectory checking.

The analytic layer solves a small convex contact-wrench feasibility QP.  It
does not choose contacts, plan motion, or alter the policy proposal.  A
separate shadow backend executes the same proposal in a copied Isaac state and
reports controller, contact, and collision evidence.  The final evaluator is
fail-closed and returns one :class:`KnotPhysicsEvaluation` per original knot.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from amsrr.feasibility.contact_wrench_trajectory import KnotPhysicsEvaluation
from amsrr.geometry.pose_math import transform_from_pose
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate
from amsrr.schemas.irg import IRGEdgeType, IRGNode, IRGNodeType, PhaseType
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)


LIGHTWEIGHT_CONTACT_QP_VERSION = "order9_lightweight_contact_qp_v1"
HYBRID_C_H_EVALUATOR_VERSION = "order9_hybrid_c_h_qp_shadow_v1"
_ACTIVE_CONTACT_STATES = frozenset({"attach", "maintain", "slide"})
_ALLOWED_COLLISION_CONTACT_STATES = frozenset(
    {"attach", "maintain", "slide", "release"}
)


@dataclass(frozen=True)
class LightweightContactQPConfig:
    """Numerical limits for the dimensionless per-knot contact QP."""

    force_scale_n: float = 30.0
    torque_scale_nm: float = 5.0
    regularization: float = 1.0e-7
    solver_absolute_tolerance: float = 1.0e-5
    solver_relative_tolerance: float = 1.0e-5
    solver_max_iterations: int = 4000
    infeasible_residual: float = 1.0
    minimum_patch_radius_m: float = 1.0e-3
    enforce_patch_moments: bool = True

    def __post_init__(self) -> None:
        for name in (
            "force_scale_n",
            "torque_scale_nm",
            "regularization",
            "solver_absolute_tolerance",
            "solver_relative_tolerance",
            "infeasible_residual",
            "minimum_patch_radius_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"LightweightContactQPConfig.{name} must be positive")
        if self.solver_max_iterations < 1:
            raise ValueError(
                "LightweightContactQPConfig.solver_max_iterations must be positive"
            )


@dataclass(frozen=True)
class ShadowCollisionSample:
    """One clearance/contact observation from the copied simulator state.

    ``signed_distance_m`` is positive when separated and non-positive at
    overlap/contact.  Candidate identity is preferred.  Anchor/entity identity
    is a strict fallback for simulators that cannot attach candidate ids to a
    contact report.
    """

    entity_a: str
    entity_b: str
    signed_distance_m: float
    candidate_id: int | None = None
    anchor_id: int | None = None
    target_entity_id: str | None = None
    task_allowed: bool = False
    allowance_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_a or not self.entity_b:
            raise ValueError("shadow collision entities must be non-empty")
        if not math.isfinite(float(self.signed_distance_m)):
            raise ValueError("shadow collision distance must be finite")
        if self.candidate_id is not None and self.candidate_id < 0:
            raise ValueError("shadow collision candidate_id must be non-negative")
        if self.anchor_id is not None and self.anchor_id < 0:
            raise ValueError("shadow collision anchor_id must be non-negative")
        if self.task_allowed and not self.allowance_reason:
            raise ValueError("task-allowed shadow collisions require an allowance reason")


@dataclass(frozen=True)
class ShadowKnotObservation:
    """Raw counterfactual evidence returned by an Isaac shadow worker."""

    controller_qp_residual: float
    contact_wrench_residual: float
    collision_samples: tuple[ShadowCollisionSample, ...] = ()
    collision_free_clearance_m: float = 0.0
    finite_state: bool = True
    main_state_unchanged: bool = True
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "controller_qp_residual",
            "contact_wrench_residual",
            "collision_free_clearance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"ShadowKnotObservation.{name} must be non-negative")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("ShadowKnotObservation.metrics must be finite")


class ShadowTrajectoryRolloutBackend(Protocol):
    """Persistent copied-environment backend; the main state must not advance."""

    @property
    def backend_version(self) -> str:
        ...

    def rollout(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> Sequence[ShadowKnotObservation]:
        ...


@dataclass(frozen=True)
class _QPResult:
    residual: float
    wrench_residual: float
    margins: dict[str, float]


class LightweightContactWrenchQPEvaluator:
    """Small convex witness QP over the already selected contact-wrench boxes."""

    def __init__(self, config: LightweightContactQPConfig | None = None) -> None:
        self.config = config or LightweightContactQPConfig()

    def evaluate_trajectory(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[KnotPhysicsEvaluation, ...]:
        return tuple(
            self.evaluate(
                context=context,
                trajectory=trajectory,
                knot_index=index,
                knot=knot,
            )
            for index, knot in enumerate(trajectory.knots)
        )

    def evaluate(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        knot_index: int,
        knot: InteractionKnot,
    ) -> KnotPhysicsEvaluation:
        del trajectory, knot_index
        try:
            result = self._solve(context=context, knot=knot)
        except Exception:
            result = _failed_qp(
                self.config,
                "unexpected_contact_qp_exception",
            )
        return KnotPhysicsEvaluation(
            qp_residual=result.residual,
            wrench_residual=result.wrench_residual,
            min_collision_margin_m=None,
            margins=result.margins,
            evaluator_version=LIGHTWEIGHT_CONTACT_QP_VERSION,
        )

    def _solve(
        self,
        *,
        context: HighLevelPolicyContext,
        knot: InteractionKnot,
    ) -> _QPResult:
        config = self.config
        active = [
            assignment
            for assignment in knot.contact_assignments
            if assignment.schedule_state in _ACTIVE_CONTACT_STATES
        ]
        requirements = active_numeric_object_wrench_requirements(context, knot)
        if not active:
            if requirements:
                return _failed_qp(config, "required_wrench_without_active_contact")
            return _QPResult(
                residual=0.0,
                wrench_residual=0.0,
                margins={
                    "contact_qp_solved": 1.0,
                    "contact_qp_variable_count": 0.0,
                    "contact_qp_constraint_count": 0.0,
                    "active_numeric_wrench_requirement_count": 0.0,
                },
            )

        candidate_by_id = {
            item.candidate_id: item
            for item in context.contact_candidate_set.candidates
        }
        candidates: list[ContactCandidate] = []
        for assignment in active:
            candidate = candidate_by_id.get(assignment.candidate_id)
            if (
                candidate is None
                or assignment.wrench_target is None
                or assignment.wrench_lower is None
                or assignment.wrench_upper is None
                or candidate.friction is None
            ):
                return _failed_qp(config, "incomplete_contact_qp_input")
            if (
                not math.isfinite(float(candidate.friction))
                or float(candidate.friction) < 0.0
                or any(
                    not math.isfinite(float(value))
                    for values in (
                        assignment.wrench_target,
                        assignment.wrench_lower,
                        assignment.wrench_upper,
                    )
                    for value in values
                )
            ):
                return _failed_qp(config, "non_finite_contact_qp_input")
            candidates.append(candidate)
        target_entities = {candidate.target_entity_id for candidate in candidates}
        if requirements and len(target_entities) != 1:
            # A numeric object-effect requirement needs an unambiguous wrench
            # reference.  Multi-object tasks must expose separate requirements
            # rather than silently summing moments about unrelated origins.
            return _failed_qp(config, "ambiguous_multi_object_wrench_requirement")

        try:
            import numpy as np
            import osqp
            from scipy import sparse
        except Exception:
            return _failed_qp(config, "contact_qp_solver_unavailable")

        variable_count = 6 * len(active)
        target = np.zeros(variable_count, dtype=float)
        lower = np.zeros(variable_count, dtype=float)
        upper = np.zeros(variable_count, dtype=float)
        scale6 = np.asarray(
            [config.force_scale_n] * 3 + [config.torque_scale_nm] * 3,
            dtype=float,
        )
        anchor_by_id = {
            anchor.anchor_id: anchor for anchor in context.morphology_graph.robot_anchors
        }
        max_contact_force = _maximum_contact_force_n(context)
        rows: list[list[float]] = []
        row_lower: list[float] = []
        row_upper: list[float] = []

        for index, (assignment, candidate) in enumerate(zip(active, candidates)):
            offset = 6 * index
            target[offset : offset + 6] = (
                np.asarray(assignment.wrench_target, dtype=float) / scale6
            )
            item_lower = np.asarray(assignment.wrench_lower, dtype=float) / scale6
            item_upper = np.asarray(assignment.wrench_upper, dtype=float) / scale6
            anchor = anchor_by_id.get(assignment.anchor_id)
            capability = {} if anchor is None else anchor.capability
            force_cap = _minimum_positive(
                max_contact_force,
                _optional_positive(capability.get("max_force_n")),
                config.force_scale_n,
            )
            torque_cap = _minimum_positive(
                _optional_positive(capability.get("max_torque_nm")),
                config.torque_scale_nm,
            )
            caps = np.asarray([force_cap] * 3 + [torque_cap] * 3) / scale6
            lower[offset : offset + 6] = np.maximum(item_lower, -caps)
            upper[offset : offset + 6] = np.minimum(item_upper, caps)
            if bool(np.any(lower[offset : offset + 6] > upper[offset : offset + 6])):
                return _failed_qp(config, "empty_assignment_wrench_box")
            _append_friction_and_patch_constraints(
                rows,
                row_lower,
                row_upper,
                variable_count=variable_count,
                variable_offset=offset,
                candidate=candidate,
                config=config,
            )

        for variable_index in range(variable_count):
            row = [0.0] * variable_count
            row[variable_index] = 1.0
            rows.append(row)
            row_lower.append(float(lower[variable_index]))
            row_upper.append(float(upper[variable_index]))

        net_matrix = _net_wrench_matrix(
            candidates,
            context=context,
            force_scale_n=config.force_scale_n,
            torque_scale_nm=config.torque_scale_nm,
        )
        try:
            requirement_lower, requirement_upper = intersect_numeric_wrench_requirements(
                requirements,
                force_scale_n=config.force_scale_n,
                torque_scale_nm=config.torque_scale_nm,
            )
        except (SchemaValidationError, TypeError, ValueError):
            return _failed_qp(config, "invalid_numeric_wrench_requirement")
        if requirement_lower is not None and requirement_upper is not None:
            if any(left > right for left, right in zip(requirement_lower, requirement_upper)):
                return _failed_qp(config, "empty_required_wrench_intersection")
            for axis in range(6):
                rows.append(list(net_matrix[axis]))
                row_lower.append(requirement_lower[axis])
                row_upper.append(requirement_upper[axis])

        matrix = sparse.csc_matrix(np.asarray(rows, dtype=float))
        regularization = float(config.regularization)
        objective = sparse.eye(variable_count, format="csc") * (2.0 * (1.0 + regularization))
        linear = -2.0 * target
        solver = osqp.OSQP()
        try:
            solver.setup(
                P=objective,
                q=linear,
                A=matrix,
                l=np.asarray(row_lower, dtype=float),
                u=np.asarray(row_upper, dtype=float),
                verbose=False,
                eps_abs=config.solver_absolute_tolerance,
                eps_rel=config.solver_relative_tolerance,
                max_iter=config.solver_max_iterations,
                polishing=True,
                adaptive_rho=True,
            )
            solution = solver.solve(raise_error=False)
        except Exception:
            return _failed_qp(config, "contact_qp_solver_error")
        status = str(getattr(solution.info, "status", "")).lower()
        solved = status in {"solved", "solved inaccurate"} and solution.x is not None
        if not solved:
            return _failed_qp(
                config,
                "contact_qp_infeasible",
                margins={
                    "contact_qp_status_value": float(
                        getattr(solution.info, "status_val", -1)
                    )
                },
            )
        values = np.asarray(solution.x, dtype=float)
        if not bool(np.isfinite(values).all()):
            return _failed_qp(config, "non_finite_contact_qp_solution")
        applied = matrix @ values
        explicit_violation = _linear_constraint_violation(
            applied,
            np.asarray(row_lower, dtype=float),
            np.asarray(row_upper, dtype=float),
        )
        primal = abs(float(getattr(solution.info, "prim_res", 0.0)))
        residual = max(primal, explicit_violation)
        wrench_residual = 0.0
        if requirement_lower is not None and requirement_upper is not None:
            achieved = net_matrix @ values
            wrench_residual = _linear_constraint_violation(
                achieved,
                np.asarray(requirement_lower, dtype=float),
                np.asarray(requirement_upper, dtype=float),
            )
        target_delta = float(np.linalg.norm(values - target))
        return _QPResult(
            residual=float(residual),
            wrench_residual=float(wrench_residual),
            margins={
                "contact_qp_solved": 1.0,
                "contact_qp_variable_count": float(variable_count),
                "contact_qp_constraint_count": float(len(rows)),
                "contact_qp_primal_residual": primal,
                "contact_qp_explicit_constraint_violation": explicit_violation,
                "contact_qp_target_witness_delta_norm": target_delta,
                "active_numeric_wrench_requirement_count": float(len(requirements)),
                "contact_qp_objective": float(solution.info.obj_val),
            },
        )


class HybridContactWrenchPhysicsEvaluator:
    """Combine the lightweight QP with one immutable-state shadow rollout."""

    def __init__(
        self,
        *,
        shadow_backend: ShadowTrajectoryRolloutBackend,
        qp_evaluator: LightweightContactWrenchQPEvaluator | None = None,
        fail_closed_residual: float = 1.0,
    ) -> None:
        if not math.isfinite(float(fail_closed_residual)) or fail_closed_residual <= 0.0:
            raise ValueError("hybrid fail_closed_residual must be positive")
        self.shadow_backend = shadow_backend
        self.qp_evaluator = qp_evaluator or LightweightContactWrenchQPEvaluator()
        self.fail_closed_residual = float(fail_closed_residual)

    def evaluate_trajectory(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[KnotPhysicsEvaluation, ...]:
        try:
            analytic = self.qp_evaluator.evaluate_trajectory(
                context=context,
                trajectory=trajectory,
            )
        except Exception:
            analytic = tuple(
                self._analytic_failure() for _ in trajectory.knots
            )
        try:
            shadow = tuple(
                self.shadow_backend.rollout(context=context, trajectory=trajectory)
            )
        except Exception:
            return tuple(self._shadow_failure(item) for item in analytic)
        if len(shadow) != len(trajectory.knots):
            return tuple(self._shadow_failure(item) for item in analytic)
        return tuple(
            self._combine(
                analytic_item,
                shadow_item,
                knot=knot,
                context=context,
            )
            for analytic_item, shadow_item, knot in zip(
                analytic, shadow, trajectory.knots
            )
        )

    def _combine(
        self,
        analytic: KnotPhysicsEvaluation,
        shadow: ShadowKnotObservation,
        *,
        knot: InteractionKnot,
        context: HighLevelPolicyContext,
    ) -> KnotPhysicsEvaluation:
        valid_shadow = bool(shadow.finite_state and shadow.main_state_unchanged)
        (
            collision_margin,
            prohibited_sample_count,
            prohibited_overlap_count,
            allowed_count,
        ) = (
            classify_shadow_collision_margin(knot, context, shadow)
        )
        qp_residual = max(
            float(analytic.qp_residual or 0.0),
            float(shadow.controller_qp_residual),
        )
        wrench_residual = max(
            float(analytic.wrench_residual or 0.0),
            float(shadow.contact_wrench_residual),
        )
        if not valid_shadow:
            qp_residual = max(qp_residual, self.fail_closed_residual)
            wrench_residual = max(wrench_residual, self.fail_closed_residual)
            collision_margin = min(collision_margin, -self.fail_closed_residual)
        margins = {
            **analytic.margins,
            **{f"shadow.{key}": float(value) for key, value in shadow.metrics.items()},
            "shadow_controller_qp_residual": float(shadow.controller_qp_residual),
            "shadow_contact_wrench_residual": float(shadow.contact_wrench_residual),
            "shadow_prohibited_pair_sample_count": float(
                prohibited_sample_count
            ),
            "shadow_prohibited_collision_count": float(
                prohibited_overlap_count
            ),
            "shadow_allowed_contact_count": float(allowed_count),
            "shadow_finite_state": 1.0 if shadow.finite_state else 0.0,
            "shadow_main_state_unchanged": 1.0 if shadow.main_state_unchanged else 0.0,
        }
        return KnotPhysicsEvaluation(
            qp_residual=qp_residual,
            wrench_residual=wrench_residual,
            min_collision_margin_m=collision_margin,
            margins=margins,
            evaluator_version=(
                f"{HYBRID_C_H_EVALUATOR_VERSION}:"
                f"{self.shadow_backend.backend_version}"
            ),
        )

    def _shadow_failure(
        self, analytic: KnotPhysicsEvaluation
    ) -> KnotPhysicsEvaluation:
        return KnotPhysicsEvaluation(
            qp_residual=max(
                float(analytic.qp_residual or 0.0), self.fail_closed_residual
            ),
            wrench_residual=max(
                float(analytic.wrench_residual or 0.0), self.fail_closed_residual
            ),
            min_collision_margin_m=-self.fail_closed_residual,
            margins={
                **analytic.margins,
                "shadow_backend_failure": 1.0,
                "shadow_main_state_unchanged": 0.0,
            },
            evaluator_version=(
                f"{HYBRID_C_H_EVALUATOR_VERSION}:shadow_backend_failure"
            ),
        )

    def _analytic_failure(self) -> KnotPhysicsEvaluation:
        return KnotPhysicsEvaluation(
            qp_residual=self.fail_closed_residual,
            wrench_residual=self.fail_closed_residual,
            min_collision_margin_m=None,
            margins={"analytic_qp_backend_failure": 1.0},
            evaluator_version=(
                f"{HYBRID_C_H_EVALUATOR_VERSION}:analytic_qp_backend_failure"
            ),
        )


def classify_shadow_collision_margin(
    knot: InteractionKnot,
    context: HighLevelPolicyContext,
    observation: ShadowKnotObservation,
) -> tuple[float, int, int, int]:
    """Exclude only explicitly intended/task-allowed pairs from the margin."""

    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    allowed_by_candidate: set[int] = set()
    allowed_by_identity: set[tuple[int, str]] = set()
    for assignment in knot.contact_assignments:
        if assignment.schedule_state not in _ALLOWED_COLLISION_CONTACT_STATES:
            continue
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is None:
            continue
        allowed_by_candidate.add(candidate.candidate_id)
        allowed_by_identity.add((assignment.anchor_id, candidate.target_entity_id))
    prohibited: list[float] = []
    allowed_count = 0
    for sample in observation.collision_samples:
        allowed = sample.task_allowed
        if sample.candidate_id is not None:
            allowed = allowed or sample.candidate_id in allowed_by_candidate
        elif sample.anchor_id is not None and sample.target_entity_id is not None:
            allowed = allowed or (
                sample.anchor_id,
                sample.target_entity_id,
            ) in allowed_by_identity
        if allowed:
            allowed_count += 1
        else:
            prohibited.append(float(sample.signed_distance_m))
    margin = (
        min(prohibited)
        if prohibited
        else float(observation.collision_free_clearance_m)
    )
    return (
        margin,
        len(prohibited),
        sum(1 for value in prohibited if value <= 0.0),
        allowed_count,
    )


def _append_friction_and_patch_constraints(
    rows: list[list[float]],
    lower: list[float],
    upper: list[float],
    *,
    variable_count: int,
    variable_offset: int,
    candidate: ContactCandidate,
    config: LightweightContactQPConfig,
) -> None:
    rotation = transform_from_pose(candidate.contact_frame_world).rotation
    inward = tuple(-float(value) for value in candidate.normal_world)
    tangents = (
        tuple(float(value) for value in candidate.tangent_basis_world[:3]),
        tuple(float(value) for value in candidate.tangent_basis_world[3:6]),
    )
    normal_coeff = _left_multiply(inward, rotation)
    tangent_coeffs = tuple(_left_multiply(item, rotation) for item in tangents)

    def append_local(
        force_coeff: Sequence[float],
        torque_coeff: Sequence[float],
        row_lower: float,
        row_upper: float,
    ) -> None:
        row = [0.0] * variable_count
        for axis in range(3):
            row[variable_offset + axis] = float(force_coeff[axis])
            row[variable_offset + 3 + axis] = float(torque_coeff[axis])
        rows.append(row)
        lower.append(float(row_lower))
        upper.append(float(row_upper))

    append_local(normal_coeff, (0.0, 0.0, 0.0), 0.0, math.inf)
    friction = float(candidate.friction or 0.0)
    for tangent in tangent_coeffs:
        append_local(
            tuple(tangent[i] - friction * normal_coeff[i] for i in range(3)),
            (0.0, 0.0, 0.0),
            -math.inf,
            0.0,
        )
        append_local(
            tuple(-tangent[i] - friction * normal_coeff[i] for i in range(3)),
            (0.0, 0.0, 0.0),
            -math.inf,
            0.0,
        )
    if not config.enforce_patch_moments:
        return
    radius = max(
        config.minimum_patch_radius_m,
        math.sqrt(max(float(candidate.patch_area_m2), 0.0) / math.pi),
    )
    force_to_torque = radius * config.force_scale_n / config.torque_scale_nm
    for tangent in tangent_coeffs:
        append_local(
            tuple(-force_to_torque * value for value in normal_coeff),
            tangent,
            -math.inf,
            0.0,
        )
        append_local(
            tuple(-force_to_torque * value for value in normal_coeff),
            tuple(-value for value in tangent),
            -math.inf,
            0.0,
        )
    torsion_scale = friction * force_to_torque
    append_local(
        tuple(-torsion_scale * value for value in normal_coeff),
        normal_coeff,
        -math.inf,
        0.0,
    )
    append_local(
        tuple(-torsion_scale * value for value in normal_coeff),
        tuple(-value for value in normal_coeff),
        -math.inf,
        0.0,
    )


def _net_wrench_matrix(
    candidates: Sequence[ContactCandidate],
    *,
    context: HighLevelPolicyContext,
    force_scale_n: float,
    torque_scale_nm: float,
):
    import numpy as np

    reference = wrench_reference_world(candidates, context)
    matrix = np.zeros((6, 6 * len(candidates)), dtype=float)
    for index, candidate in enumerate(candidates):
        rotation = np.asarray(
            transform_from_pose(candidate.contact_frame_world).rotation,
            dtype=float,
        )
        position = np.asarray(candidate.contact_pose_world[:3], dtype=float)
        arm = position - np.asarray(reference, dtype=float)
        skew = np.asarray(
            (
                (0.0, -arm[2], arm[1]),
                (arm[2], 0.0, -arm[0]),
                (-arm[1], arm[0], 0.0),
            ),
            dtype=float,
        )
        offset = 6 * index
        matrix[:3, offset : offset + 3] = rotation
        matrix[3:, offset : offset + 3] = (
            force_scale_n / torque_scale_nm
        ) * (skew @ rotation)
        matrix[3:, offset + 3 : offset + 6] = rotation
    return matrix


def wrench_reference_world(
    candidates: Sequence[ContactCandidate], context: HighLevelPolicyContext
) -> tuple[float, float, float]:
    target_ids = {candidate.target_entity_id for candidate in candidates}
    observation = context.runtime_observation
    if observation is not None and len(target_ids) == 1:
        target_id = next(iter(target_ids))
        for state in observation.object_states:
            if state.object_id == target_id:
                return tuple(float(value) for value in state.pose_world[:3])
    count = max(len(candidates), 1)
    return tuple(
        sum(float(candidate.contact_pose_world[axis]) for candidate in candidates)
        / count
        for axis in range(3)
    )  # type: ignore[return-value]


def active_numeric_object_wrench_requirements(
    context: HighLevelPolicyContext,
    knot: InteractionKnot,
) -> list[IRGNode]:
    phase_index = _active_phase_index(context, knot)
    release_indices = sorted(
        int(node.feature.get("phase_index", -1))
        for node in context.irg.nodes
        if node.node_type == IRGNodeType.PHASE
        and node.feature.get("phase_type") == PhaseType.RELEASE_CONTACT.value
        and int(node.feature.get("phase_index", -1)) >= 0
    )
    if phase_index is not None and any(phase_index >= item for item in release_indices):
        return []
    node_by_id = {node.node_id: node for node in context.irg.nodes}
    trigger_by_requirement: dict[int, list[int]] = {}
    for edge in context.irg.edges:
        if edge.edge_type != IRGEdgeType.REQUIRES:
            continue
        source = node_by_id.get(edge.src_id)
        target = node_by_id.get(edge.dst_id)
        if (
            source is not None
            and target is not None
            and source.node_type == IRGNodeType.PHASE
            and target.node_type == IRGNodeType.WRENCH_REQUIREMENT
        ):
            trigger_by_requirement.setdefault(target.node_id, []).append(
                int(source.feature.get("phase_index", -1))
            )
    selected: list[IRGNode] = []
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.WRENCH_REQUIREMENT or not node.is_hard:
            continue
        if node.feature.get("applies_to") != "object_effect":
            continue
        if not any(
            node.feature.get(name) is not None
            for name in ("wrench_lower", "wrench_upper", "target_wrench")
        ):
            continue
        triggers = [value for value in trigger_by_requirement.get(node.node_id, []) if value >= 0]
        if phase_index is None or not triggers or phase_index >= min(triggers):
            selected.append(node)
    return selected


def _active_phase_index(
    context: HighLevelPolicyContext, knot: InteractionKnot
) -> int | None:
    phase_label: str | None = None
    for guard in knot.guard_conditions:
        if guard.get("type") == "order9_task_phase" and guard.get("phase_label"):
            phase_label = str(guard["phase_label"])
            break
    if phase_label is None and context.runtime_observation is not None:
        phase_label = context.runtime_observation.task_progress.phase_label
    if phase_label is None:
        return None
    aliases = {
        "reset": "approach_object",
        "approach": "approach_object",
        "contact_acquisition": "establish_object_contacts",
        "establish_contact": "establish_object_contacts",
        "apply_wrench": "apply_grasp_wrench",
        "lift": "lift_object",
        "transport": "transport_object",
        "place": "place_object",
        "release": "release_contacts",
    }
    normalized = aliases.get(phase_label, phase_label)
    for node in context.irg.nodes:
        if (
            node.node_type == IRGNodeType.PHASE
            and str(node.feature.get("phase_label", "")) == normalized
        ):
            return int(node.feature.get("phase_index", -1))
    return None


def intersect_numeric_wrench_requirements(
    requirements: Sequence[IRGNode],
    *,
    force_scale_n: float,
    torque_scale_nm: float,
) -> tuple[list[float] | None, list[float] | None]:
    if not requirements:
        return None, None
    scale = [force_scale_n] * 3 + [torque_scale_nm] * 3
    lower = [-math.inf] * 6
    upper = [math.inf] * 6
    for node in requirements:
        frame = str(node.feature.get("frame", "world"))
        if frame != "world":
            raise SchemaValidationError(
                "numeric object-effect wrench requirements currently require world frame"
            )
        target = node.feature.get("target_wrench")
        raw_lower = node.feature.get("wrench_lower")
        raw_upper = node.feature.get("wrench_upper")
        if target is not None:
            raw_lower = target
            raw_upper = target
        if raw_lower is not None:
            if len(raw_lower) != 6:
                raise SchemaValidationError("wrench requirement lower bound must have length 6")
            lower = [
                max(lower[index], float(raw_lower[index]) / scale[index])
                for index in range(6)
            ]
        if raw_upper is not None:
            if len(raw_upper) != 6:
                raise SchemaValidationError("wrench requirement upper bound must have length 6")
            upper = [
                min(upper[index], float(raw_upper[index]) / scale[index])
                for index in range(6)
            ]
    return lower, upper


def _maximum_contact_force_n(context: HighLevelPolicyContext) -> float | None:
    values: list[float] = []
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.CONSTRAINT:
            continue
        if node.feature.get("constraint_type") != "max_contact_force":
            continue
        parameters = node.feature.get("parameters", {}) or {}
        value = _optional_positive(parameters.get("max_n"))
        if value is not None:
            values.append(value)
    return min(values) if values else None


def _minimum_positive(*values: float | None) -> float:
    selected = [float(value) for value in values if value is not None and value > 0.0]
    if not selected:
        raise ValueError("at least one positive limit is required")
    return min(selected)


def _optional_positive(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _left_multiply(
    vector: Sequence[float], matrix: Sequence[Sequence[float]]
) -> tuple[float, float, float]:
    return tuple(
        sum(float(vector[row]) * float(matrix[row][column]) for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def _linear_constraint_violation(values, lower, upper) -> float:
    import numpy as np

    below = np.maximum(lower - values, 0.0)
    above = np.maximum(values - upper, 0.0)
    finite = np.concatenate((below[np.isfinite(below)], above[np.isfinite(above)]))
    return float(np.max(finite)) if finite.size else 0.0


def _failed_qp(
    config: LightweightContactQPConfig,
    reason: str,
    *,
    margins: dict[str, float] | None = None,
) -> _QPResult:
    reason_hash = float(sum((index + 1) * ord(char) for index, char in enumerate(reason)))
    return _QPResult(
        residual=config.infeasible_residual,
        wrench_residual=config.infeasible_residual,
        margins={
            "contact_qp_solved": 0.0,
            "contact_qp_failure_reason_hash": reason_hash,
            **(margins or {}),
        },
    )


__all__ = [
    "HYBRID_C_H_EVALUATOR_VERSION",
    "LIGHTWEIGHT_CONTACT_QP_VERSION",
    "HybridContactWrenchPhysicsEvaluator",
    "LightweightContactQPConfig",
    "LightweightContactWrenchQPEvaluator",
    "ShadowCollisionSample",
    "ShadowKnotObservation",
    "ShadowTrajectoryRolloutBackend",
    "active_numeric_object_wrench_requirements",
    "classify_shadow_collision_margin",
    "intersect_numeric_wrench_requirements",
    "wrench_reference_world",
]
