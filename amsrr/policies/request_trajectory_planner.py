"""Bounded constrained optimization and joint-authoritative CWT evaluation.

A task/robot model supplies a numerical problem, not a learned trajectory or
an archived teacher. Missing physical constraints fail closed. The optimizer
never substitutes for the independent runtime C_H check.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from time import monotonic
from typing import Callable, Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from amsrr.feasibility.articulated_reachability import (
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.policies.contact_wrench_trajectory_runtime import (
    _interpolate_interaction_knot,
)
from amsrr.policies.high_level_requests import HighLevelDecisionContext
from amsrr.robot_model.whole_structure_kinematics import WholeStructureKinematics
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import GeneratedExecutionPlan, HighLevelRequest
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.utils.hashing import stable_hash

REQUIRED_PLANNING_CONSTRAINTS = frozenset(
    {
        "whole_body_kinematics",
        "contact_geometry",
        "collision_intervals",
        "joint_limits",
        "joint_velocity",
        "joint_acceleration",
        "joint_effort",
        "contact_wrench",
        "actuator_authority",
        "tracking_margin",
        "remaining_task",
    }
)


@dataclass(frozen=True)
class RequestPlannerConfig:
    max_iterations: int = 80
    feasibility_tolerance: float = 1.0e-6
    objective_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.feasibility_tolerance, self.objective_tolerance)
        ):
            raise ValueError("planner tolerances must be finite and positive")


@dataclass(frozen=True)
class RequestPlanningProblem:
    """A model-built fixed-contact-schedule problem; all margins are >= 0.

    The builder owns the common body/joint/object/wrench/time parameterization
    and declared uncertainty margins. A solver status alone is never evidence.
    """

    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    objective: Callable[[np.ndarray], float]
    constraint_margins: Callable[[np.ndarray], dict[str, np.ndarray]]
    materialize: Callable[[np.ndarray], ContactWrenchTrajectory]
    model_version: str
    config_hash: str


class RequestPlanningModel(Protocol):
    def build_problem(
        self,
        context: HighLevelDecisionContext,
        request: HighLevelRequest,
        *,
        deadline: float,
    ) -> RequestPlanningProblem: ...


@dataclass(frozen=True)
class RequestPlanningResult:
    status: str
    plan: GeneratedExecutionPlan | None
    reason: str
    elapsed_s: float
    iterations: int = 0


class ConstrainedRequestPlanner:
    """One finite request -> one bounded model-constrained solve -> immutable plan."""

    planner_version = "constrained_request_slsqp_v1"

    def __init__(
        self, model: RequestPlanningModel, config: RequestPlannerConfig | None = None
    ):
        self.model = model
        self.config = config or RequestPlannerConfig()

    def plan(
        self,
        context: HighLevelDecisionContext,
        request: HighLevelRequest,
        *,
        deadline: float,
    ) -> RequestPlanningResult:
        start = monotonic()
        iterations = 0

        def remaining() -> None:
            if monotonic() >= deadline:
                raise TimeoutError("request planning deadline exceeded")

        try:
            remaining()
            context.validate_snapshot()
            context.catalog.resolve(request)
            problem = self.model.build_problem(context, request, deadline=deadline)
            remaining()
            x0, lower, upper = (
                np.asarray(x, dtype=float)
                for x in (problem.initial, problem.lower, problem.upper)
            )
            if (
                x0.ndim != 1
                or not x0.size
                or lower.shape != x0.shape
                or upper.shape != x0.shape
            ):
                raise SchemaValidationError("invalid planning variable dimensions")
            if not all(np.isfinite(x).all() for x in (x0, lower, upper)) or np.any(
                lower > upper
            ):
                raise SchemaValidationError("invalid planning bounds")
            if not problem.model_version or not problem.config_hash:
                raise SchemaValidationError(
                    "planning model/config identity is required"
                )
            shape = None

            def margins(x):
                nonlocal shape
                remaining()
                terms = problem.constraint_margins(x)
                if set(terms) != REQUIRED_PLANNING_CONSTRAINTS:
                    raise SchemaValidationError("incomplete planning constraints")
                values = [
                    np.asarray(terms[name], dtype=float).reshape(-1)
                    for name in sorted(terms)
                ]
                if any(not v.size or not np.isfinite(v).all() for v in values):
                    raise SchemaValidationError("empty/nonfinite planning constraint")
                current_shape = tuple(len(v) for v in values)
                if shape is not None and shape != current_shape:
                    raise SchemaValidationError(
                        "planning constraint dimensions changed"
                    )
                shape = current_shape
                remaining()
                return np.concatenate(values)

            def objective(x):
                remaining()
                result = float(problem.objective(x))
                if not math.isfinite(result):
                    raise SchemaValidationError("nonfinite planning objective")
                return result

            def callback(x):
                nonlocal iterations
                iterations += 1
                remaining()

            # Establish the entire constraint surface before launching optimization.
            margins(x0)
            result = minimize(
                objective,
                np.clip(x0, lower, upper),
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=[{"type": "ineq", "fun": margins}],
                callback=callback,
                options={
                    "maxiter": self.config.max_iterations,
                    "ftol": self.config.objective_tolerance,
                },
            )
            remaining()
            valid = (
                np.isfinite(result.x).all()
                and np.all(result.x >= lower - self.config.feasibility_tolerance)
                and np.all(result.x <= upper + self.config.feasibility_tolerance)
                and np.all(margins(result.x) >= -self.config.feasibility_tolerance)
            )
            if not valid:
                return RequestPlanningResult(
                    "no_solution_found",
                    None,
                    str(result.message),
                    monotonic() - start,
                    iterations,
                )
            trajectory = problem.materialize(result.x)
            remaining()
            context.validate_snapshot()
            plan = GeneratedExecutionPlan(
                deepcopy(request),
                context.catalog.snapshot_hash,
                context.physical_model.stable_hash(),
                self.planner_version + ":" + problem.model_version,
                stable_hash(
                    {"model": problem.config_hash, "solver": self.config.__dict__}
                ),
                context.observation.time_s,
                context.observation.time_s + trajectory.horizon_s,
                trajectory,
            )
            plan.solver_constraint_margins = {
                name: np.asarray(values, dtype=float).reshape(-1).tolist()
                for name, values in problem.constraint_margins(result.x).items()
            }
            plan.validate()
            remaining()
            return RequestPlanningResult(
                "solution_found",
                plan,
                str(result.message),
                monotonic() - start,
                iterations,
            )
        except TimeoutError as error:
            return RequestPlanningResult(
                "timeout", None, str(error), monotonic() - start, iterations
            )
        except (SchemaValidationError, ValueError, KeyError, TypeError) as error:
            return RequestPlanningResult(
                "invalid_request", None, str(error), monotonic() - start, iterations
            )


class JointPlanEvaluator:
    """Evaluate the same quintic joint/CoM path in optimizer, C_H and executor.

    Free-anchor poses are recomputed by FK, never independently interpolated
    into a second joint solution. Knot joint velocities must be zero: each
    quintic segment starts and ends at rest with zero acceleration.
    """

    def __init__(self, context: HighLevelDecisionContext):
        self.context = context
        self.kinematics = WholeStructureKinematics()
        self.model_hash = context.physical_model.stable_hash()
        self._references: dict[tuple[int, ...], tuple] = {}

    def sample(self, plan: GeneratedExecutionPlan, *, time_s: float) -> InteractionKnot:
        if plan.model_hash != self.model_hash:
            raise SchemaValidationError("plan model identity mismatch")
        t = time_s - plan.observation_time_s
        if (
            not math.isfinite(t)
            or t < -1.0e-10
            or time_s > plan.valid_until_s + 1.0e-10
        ):
            raise SchemaValidationError(
                "execution plan expired or sampled before its origin"
            )
        knots = plan.trajectory.knots
        for knot in knots:
            if any(
                abs(v) > 1.0e-12 for v in knot.posture_target.joint_vel_target.values()
            ):
                raise SchemaValidationError(
                    "quintic-rest interpolation requires zero knot velocity"
                )
        index = next(
            (i for i in range(len(knots) - 1) if t < knots[i + 1].t_rel_s),
            len(knots) - 2,
        )
        left, right = knots[index : index + 2]
        duration = right.t_rel_s - left.t_rel_s
        u = min(1.0, max(0.0, (t - left.t_rel_s) / duration))
        blend = u**3 * (10 + u * (-15 + 6 * u))
        speed = 30 * u**2 * (1 - u) ** 2 / duration
        knot = _interpolate_interaction_knot(left, right, blend, t)
        q = knot.posture_target.joint_pos_target
        knot.posture_target.joint_vel_target = {
            key: (
                right.posture_target.joint_pos_target[key]
                - left.posture_target.joint_pos_target[key]
            )
            * speed
            for key in q
        }
        centroidal = knot.centroidal_target
        if (
            centroidal is None
            or centroidal.com_pos_world is None
            or centroidal.body_orientation_world is None
        ):
            raise SchemaValidationError(
                "joint plan requires complete centroidal targets"
            )
        centroidal.com_vel_world = tuple(
            (
                right.centroidal_target.com_pos_world[i]
                - left.centroidal_target.com_pos_world[i]
            )
            * speed
            for i in range(3)
        )
        a_rotation = Rotation.from_quat(left.centroidal_target.body_orientation_world)
        delta_rotation = a_rotation.inv() * Rotation.from_quat(
            right.centroidal_target.body_orientation_world
        )
        centroidal.body_orientation_world = tuple(
            (
                a_rotation * Rotation.from_rotvec(delta_rotation.as_rotvec() * blend)
            ).as_quat()
        )
        for target in knot.object_targets:
            a = next(x for x in left.object_targets if x.object_id == target.object_id)
            b = next(x for x in right.object_targets if x.object_id == target.object_id)
            if a.pose_target_world and b.pose_target_world:
                ar = Rotation.from_quat(a.pose_target_world[3:])
                dr = ar.inv() * Rotation.from_quat(b.pose_target_world[3:])
                target.pose_target_world = (
                    *target.pose_target_world[:3],
                    *(ar * Rotation.from_rotvec(dr.as_rotvec() * blend)).as_quat(),
                )
                angular = (
                    Rotation.from_quat(b.pose_target_world[3:])
                    * Rotation.from_quat(a.pose_target_world[3:]).inv()
                ).as_rotvec() * speed
                target.twist_target_world = [
                    (b.pose_target_world[i] - a.pose_target_world[i]) * speed
                    for i in range(3)
                ] + angular.tolist()
        ids = tuple(sorted(a.anchor_id for a in knot.contact_assignments))
        if ids not in self._references:
            self._references[ids] = resolve_mesh_backed_anchor_references(
                self.context.scene.morphology_graph, self.context.physical_model, ids
            )
        base = base_pose_for_centroidal_target(
            self.context.scene.morphology_graph,
            self.context.physical_model,
            q,
            centroidal.com_pos_world,
            centroidal.body_orientation_world,
            kinematics=self.kinematics,
        )
        fk = self.kinematics.forward(
            self.context.scene.morphology_graph,
            self.context.physical_model,
            q,
            base,
            self._references[ids],
        )
        knot.posture_target.free_anchor_pose_targets = dict(fk.anchor_poses_world)
        return knot

    def body_angular_velocity(
        self, plan: GeneratedExecutionPlan, *, time_s: float
    ) -> list[float]:
        """Body-frame angular velocity of the same shortest-arc SLERP path."""
        t = time_s - plan.observation_time_s
        knots = plan.trajectory.knots
        i = next(
            (i for i in range(len(knots) - 1) if t < knots[i + 1].t_rel_s),
            len(knots) - 2,
        )
        a, b = knots[i : i + 2]
        dt = b.t_rel_s - a.t_rel_s
        u = min(1.0, max(0.0, (t - a.t_rel_s) / dt))
        delta = Rotation.from_quat(
            a.centroidal_target.body_orientation_world
        ).inv() * Rotation.from_quat(b.centroidal_target.body_orientation_world)
        return (delta.as_rotvec() * (30 * u * u * (1 - u) ** 2 / dt)).tolist()
