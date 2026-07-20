from __future__ import annotations

"""Hard acceptance boundary for an unmodified high-level policy proposal."""

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from amsrr.geometry.wrench import contact_wrench_to_world
from amsrr.policies.assignment_feasibility import (
    ASSIGNMENT_QP_INFEASIBLE_CODE,
    ASSIGNMENT_WRENCH_INFEASIBLE_CODE,
    COLLISION_MARGIN_FAIL_CODE,
    evaluate_selected_assignment_feasibility,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.feasibility import (
    TrajectoryFeasibilityResult,
    TrajectoryKnotFeasibilityResult,
    Violation,
    ViolationSeverity,
)
from amsrr.schemas.irg import IRGEdgeType, IRGNodeType, PhaseType
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)


CONTACT_WRENCH_TRAJECTORY_CHECKER_VERSION = "contact_wrench_trajectory_checker_v2"
TRAJECTORY_SCHEMA_INVALID_CODE = "E_TRAJECTORY_SCHEMA_INVALID"
TRAJECTORY_TIME_INVALID_CODE = "E_TRAJECTORY_TIME_INVALID"
TRAJECTORY_CONTRACT_INVALID_CODE = "E_TRAJECTORY_CONTRACT_INVALID"
TRAJECTORY_WRENCH_BOUNDS_MISSING_CODE = "E_TRAJECTORY_WRENCH_BOUNDS_MISSING"
TRAJECTORY_WRENCH_CONE_FAIL_CODE = "E_TRAJECTORY_WRENCH_CONE_FAIL"
TRAJECTORY_QP_NOT_EVALUATED_CODE = "E_TRAJECTORY_QP_NOT_EVALUATED"
TRAJECTORY_COLLISION_NOT_EVALUATED_CODE = "E_TRAJECTORY_COLLISION_NOT_EVALUATED"
TRAJECTORY_WRENCH_NOT_EVALUATED_CODE = "E_TRAJECTORY_WRENCH_NOT_EVALUATED"


@dataclass(frozen=True)
class KnotPhysicsEvaluation:
    """Results supplied by the real controller/collision backend for one knot."""

    qp_residual: float | None = None
    wrench_residual: float | None = None
    min_collision_margin_m: float | None = None
    margins: dict[str, float] = field(default_factory=dict)
    evaluator_version: str = "unspecified"

    def __post_init__(self) -> None:
        for name in (
            "qp_residual",
            "wrench_residual",
            "min_collision_margin_m",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"KnotPhysicsEvaluation.{name} must be finite")
        if not self.evaluator_version:
            raise ValueError("KnotPhysicsEvaluation.evaluator_version must be non-empty")
        if any(not math.isfinite(float(value)) for value in self.margins.values()):
            raise ValueError("KnotPhysicsEvaluation.margins must be finite")


class KnotPhysicsEvaluator(Protocol):
    def evaluate(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        knot_index: int,
        knot: InteractionKnot,
    ) -> KnotPhysicsEvaluation:
        ...


class TrajectoryPhysicsEvaluator(Protocol):
    """Batch evaluator used by counterfactual/shadow backends.

    A shadow rollout advances a copied environment once for the complete
    proposal.  Calling it independently for every knot would both waste work
    and evaluate different counterfactual histories, so production evaluators
    may expose this trajectory-level interface.  The legacy per-knot protocol
    remains supported for existing controller adapters and tests.
    """

    def evaluate_trajectory(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> Sequence[KnotPhysicsEvaluation]:
        ...


@dataclass(frozen=True)
class ContactWrenchTrajectoryCheckerConfig:
    evaluation_mode: Literal["production", "warmup_proxy"] = "production"
    allow_legacy_contract: bool = False
    require_active_wrench_bounds: bool = True
    require_qp_evaluation: bool = True
    require_collision_evaluation: bool = True
    require_wrench_evaluation: bool = True
    qp_residual_threshold: float = 1.0e-4
    wrench_residual_threshold: float = 1.0e-3
    collision_margin_threshold_m: float = 0.0
    min_required_friction: float = 0.05
    friction_cone_tolerance_n: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in (
            "qp_residual_threshold",
            "wrench_residual_threshold",
            "collision_margin_threshold_m",
            "min_required_friction",
            "friction_cone_tolerance_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"ContactWrenchTrajectoryCheckerConfig.{name} must be finite and non-negative"
                )
        if self.evaluation_mode == "production" and not (
            self.require_qp_evaluation
            and self.require_collision_evaluation
            and self.require_wrench_evaluation
        ):
            raise ValueError(
                "production trajectory checking must require QP, collision, and wrench evaluation"
            )

    @classmethod
    def warmup_proxy(cls, *, allow_legacy_contract: bool = False) -> "ContactWrenchTrajectoryCheckerConfig":
        return cls(
            evaluation_mode="warmup_proxy",
            allow_legacy_contract=allow_legacy_contract,
            require_qp_evaluation=False,
            require_collision_evaluation=False,
            require_wrench_evaluation=False,
        )


class ContactWrenchTrajectoryFeasibilityChecker:
    """Accept or reject a proposal without projecting or mutating it."""

    def __init__(
        self,
        *,
        config: ContactWrenchTrajectoryCheckerConfig | None = None,
        physics_evaluator: KnotPhysicsEvaluator | TrajectoryPhysicsEvaluator | None = None,
    ) -> None:
        self.config = config or ContactWrenchTrajectoryCheckerConfig()
        self.physics_evaluator = physics_evaluator

    def check(
        self,
        trajectory: ContactWrenchTrajectory,
        context: HighLevelPolicyContext,
    ) -> TrajectoryFeasibilityResult:
        hard: list[Violation] = []
        warnings: list[Violation] = []
        _check_trajectory_structure(trajectory, hard)
        if (
            trajectory.contract_version != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
            and not self.config.allow_legacy_contract
        ):
            _append_violation(
                hard,
                TRAJECTORY_CONTRACT_INVALID_CODE,
                "production checker requires contact_frame_robot_on_target_v2",
            )

        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in context.contact_candidate_set.candidates
        }
        knot_results: list[TrajectoryKnotFeasibilityResult] = []
        all_margins: dict[str, float] = {}
        evaluator_versions: set[str] = set()
        trajectory_physics = self._trajectory_physics_evaluations(
            context=context,
            trajectory=trajectory,
        )
        required_collision_margin_m = max(
            self.config.collision_margin_threshold_m,
            _required_collision_margin_m(context),
        )

        for knot_index, knot in enumerate(trajectory.knots):
            slot_mins, slot_maxes = _slot_count_requirements(context, knot)
            knot_codes: list[str] = []
            cone_margin = _check_wrench_contract_and_cones(
                knot.contact_assignments,
                candidate_by_id,
                config=self.config,
                violation_codes=knot_codes,
            )
            physics = (
                trajectory_physics[knot_index]
                if trajectory_physics is not None
                else self._physics_evaluation(
                    context=context,
                    trajectory=trajectory,
                    knot_index=knot_index,
                    knot=knot,
                )
            )
            if physics is not None:
                evaluator_versions.add(physics.evaluator_version)
            qp_evaluated = physics is not None and physics.qp_residual is not None
            collision_evaluated = (
                physics is not None and physics.min_collision_margin_m is not None
            )
            wrench_evaluated = physics is not None and physics.wrench_residual is not None
            _require_evaluation(
                qp_evaluated,
                self.config.require_qp_evaluation,
                TRAJECTORY_QP_NOT_EVALUATED_CODE,
                knot_codes,
                warnings,
                knot_index,
                self.config.evaluation_mode,
            )
            _require_evaluation(
                collision_evaluated,
                self.config.require_collision_evaluation,
                TRAJECTORY_COLLISION_NOT_EVALUATED_CODE,
                knot_codes,
                warnings,
                knot_index,
                self.config.evaluation_mode,
            )
            _require_evaluation(
                wrench_evaluated,
                self.config.require_wrench_evaluation,
                TRAJECTORY_WRENCH_NOT_EVALUATED_CODE,
                knot_codes,
                warnings,
                knot_index,
                self.config.evaluation_mode,
            )

            qp_residual = None if physics is None else physics.qp_residual
            wrench_residual = None if physics is None else physics.wrench_residual
            collision_margin = (
                None if physics is None else physics.min_collision_margin_m
            )
            assignment_result = evaluate_selected_assignment_feasibility(
                knot.contact_assignments,
                context.contact_candidate_set,
                slot_min_counts=slot_mins,
                slot_max_counts=slot_maxes,
                qp_residual=qp_residual,
                qp_residual_threshold=self.config.qp_residual_threshold,
                wrench_residual=wrench_residual,
                wrench_residual_threshold=self.config.wrench_residual_threshold,
                min_required_friction=self.config.min_required_friction,
                min_collision_margin_m=collision_margin,
                collision_margin_threshold_m=required_collision_margin_m,
                update_cache=False,
            )
            for code in assignment_result.violation_codes:
                _append_unique(knot_codes, code)

            margins = {} if physics is None else dict(physics.margins)
            if cone_margin is not None:
                margins["friction_cone_force_margin_n"] = cone_margin
            if qp_residual is not None:
                margins["qp_residual_margin"] = (
                    self.config.qp_residual_threshold - qp_residual
                )
            if wrench_residual is not None:
                margins["wrench_residual_margin"] = (
                    self.config.wrench_residual_threshold - wrench_residual
                )
            if collision_margin is not None:
                margins["collision_margin_m"] = (
                    collision_margin - required_collision_margin_m
                )
                margins["required_collision_margin_m"] = (
                    required_collision_margin_m
                )
            for name, value in margins.items():
                key = f"knot_{knot_index}.{name}"
                all_margins[key] = float(value)

            for code in knot_codes:
                _append_violation(
                    hard,
                    code,
                    _violation_message(code, knot_index),
                    node_or_edge_ref=f"trajectory.knot[{knot_index}]",
                )
            knot_results.append(
                TrajectoryKnotFeasibilityResult(
                    knot_index=knot_index,
                    t_rel_s=float(knot.t_rel_s),
                    assignment_result=assignment_result,
                    qp_evaluated=qp_evaluated,
                    collision_evaluated=collision_evaluated,
                    wrench_evaluated=wrench_evaluated,
                    margins=margins,
                    violation_codes=knot_codes,
                )
            )

        return TrajectoryFeasibilityResult(
            feasible=not hard,
            hard_violations=hard,
            warnings=warnings,
            knot_results=knot_results,
            margins=all_margins,
            checker_version=CONTACT_WRENCH_TRAJECTORY_CHECKER_VERSION,
            contract_version=trajectory.contract_version,
            metadata={
                "evaluation_mode": self.config.evaluation_mode,
                "proposal_mutated": False,
                "physics_evaluator_versions": sorted(evaluator_versions),
                "required_collision_margin_m": required_collision_margin_m,
            },
        )

    def _trajectory_physics_evaluations(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[KnotPhysicsEvaluation, ...] | None:
        evaluator = self.physics_evaluator
        if evaluator is None:
            return None
        evaluate_trajectory = getattr(evaluator, "evaluate_trajectory", None)
        if evaluate_trajectory is None:
            return None
        values = tuple(
            evaluate_trajectory(context=context, trajectory=trajectory)
        )
        if len(values) != len(trajectory.knots):
            raise ValueError(
                "trajectory physics evaluator must return exactly one result per knot"
            )
        if any(not isinstance(value, KnotPhysicsEvaluation) for value in values):
            raise TypeError(
                "trajectory physics evaluator returned a non-KnotPhysicsEvaluation"
            )
        return values

    def _physics_evaluation(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        knot_index: int,
        knot: InteractionKnot,
    ) -> KnotPhysicsEvaluation | None:
        if self.physics_evaluator is None:
            return None
        return self.physics_evaluator.evaluate(
            context=context,
            trajectory=trajectory,
            knot_index=knot_index,
            knot=knot,
        )


def _check_trajectory_structure(
    trajectory: ContactWrenchTrajectory,
    hard: list[Violation],
) -> None:
    try:
        trajectory.validate()
    except (SchemaValidationError, TypeError, ValueError) as exc:
        _append_violation(hard, TRAJECTORY_SCHEMA_INVALID_CODE, str(exc))
        return
    times = [float(knot.t_rel_s) for knot in trajectory.knots]
    invalid = (
        not times
        or not all(math.isfinite(value) for value in times)
        or abs(times[0]) > 1.0e-9
        or any(right <= left for left, right in zip(times, times[1:]))
        or times[-1] > trajectory.horizon_s + 1.0e-9
        or trajectory.horizon_s - times[-1] > trajectory.dt_s + 1.0e-9
    )
    if invalid:
        _append_violation(
            hard,
            TRAJECTORY_TIME_INVALID_CODE,
            "trajectory knots must start at zero, increase strictly, and cover the horizon",
        )


def _check_wrench_contract_and_cones(
    assignments: list[ContactAssignment],
    candidate_by_id: dict[int, object],
    *,
    config: ContactWrenchTrajectoryCheckerConfig,
    violation_codes: list[str],
) -> float | None:
    minimum_margin: float | None = None
    for assignment in assignments:
        active = assignment.schedule_state in {"attach", "maintain", "slide"}
        if active and config.require_active_wrench_bounds and (
            assignment.wrench_target is None
            or assignment.wrench_lower is None
            or assignment.wrench_upper is None
        ):
            _append_unique(
                violation_codes,
                TRAJECTORY_WRENCH_BOUNDS_MISSING_CODE,
            )
        if assignment.wrench_target is None:
            continue
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is None:
            continue
        friction = getattr(candidate, "friction", None)
        if friction is None:
            continue
        outward = getattr(candidate, "normal_world")
        inward = (-float(outward[0]), -float(outward[1]), -float(outward[2]))
        # The target itself is executed unchanged and therefore must lie in the
        # cone.  Bounds describe a feasible-search region for C_H's witness QP;
        # requiring every box corner to lie in the circular cone would reject
        # valid ranges whose intersection with the cone is non-empty.
        force_values = [tuple(float(value) for value in assignment.wrench_target[:3])]
        for force_value in force_values:
            if assignment.wrench_frame == "contact":
                force_world = contact_wrench_to_world(
                    [*force_value, 0.0, 0.0, 0.0],
                    candidate,
                )[:3]
            else:
                force_world = list(force_value)
            normal_force = sum(
                force_world[index] * inward[index] for index in range(3)
            )
            tangent = [
                force_world[index] - normal_force * inward[index]
                for index in range(3)
            ]
            tangential_force = math.sqrt(sum(value * value for value in tangent))
            margin = float(friction) * normal_force - tangential_force
            minimum_margin = (
                margin if minimum_margin is None else min(minimum_margin, margin)
            )
            if (
                normal_force < -config.friction_cone_tolerance_n
                or margin < -config.friction_cone_tolerance_n
            ):
                _append_unique(
                    violation_codes,
                    TRAJECTORY_WRENCH_CONE_FAIL_CODE,
                )
    return minimum_margin


def _required_collision_margin_m(context: HighLevelPolicyContext) -> float:
    """Resolve the hard IRG collision clearance without task-name branches."""

    values: list[float] = []
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.CONSTRAINT:
            continue
        if node.feature.get("constraint_type") != "collision_margin":
            continue
        parameters = node.feature.get("parameters", {}) or {}
        value = parameters.get("margin_m")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            if math.isfinite(parsed) and parsed >= 0.0:
                values.append(parsed)
    return max(values, default=0.0)


def _slot_count_requirements(
    context: HighLevelPolicyContext,
    knot: InteractionKnot,
) -> tuple[dict[int, int], dict[int, int]]:
    active_phase_index = _active_phase_index(context, knot)
    activation_by_slot: dict[int, list[int]] = {}
    phase_index_by_node_id = {
        node.node_id: int(node.feature.get("phase_index", -1))
        for node in context.irg.nodes
        if node.node_type == IRGNodeType.PHASE
    }
    for edge in context.irg.edges:
        if edge.edge_type != IRGEdgeType.ACTIVATES:
            continue
        phase_index = phase_index_by_node_id.get(edge.src_id)
        if phase_index is not None and phase_index >= 0:
            activation_by_slot.setdefault(edge.dst_id, []).append(phase_index)
    release_indices = sorted(
        int(node.feature.get("phase_index", -1))
        for node in context.irg.nodes
        if node.node_type == IRGNodeType.PHASE
        and node.feature.get("phase_type") == PhaseType.RELEASE_CONTACT.value
        and int(node.feature.get("phase_index", -1)) >= 0
    )
    minimums: dict[int, int] = {}
    maximums: dict[int, int] = {}
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.CONTACT_SLOT:
            continue
        slot_id = int(node.feature.get("slot_id", node.node_id))
        required_now = bool(node.feature.get("required", True))
        activations = activation_by_slot.get(node.node_id, [])
        if active_phase_index is not None and activations:
            activation_index = min(activations)
            release_index = next(
                (value for value in release_indices if value > activation_index),
                None,
            )
            required_now = bool(
                required_now
                and active_phase_index >= activation_index
                and (release_index is None or active_phase_index < release_index)
            )
        if required_now:
            minimums[slot_id] = int(node.feature.get("min_count_group", 1))
        maximums[slot_id] = int(node.feature.get("max_count_group", 1))
    return minimums, maximums


def _active_phase_index(
    context: HighLevelPolicyContext,
    knot: InteractionKnot,
) -> int | None:
    phase_label = _knot_phase_label(knot)
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
    normalized = aliases.get(str(phase_label), str(phase_label))
    phases = [
        node
        for node in context.irg.nodes
        if node.node_type == IRGNodeType.PHASE
    ]
    exact = [
        node
        for node in phases
        if str(node.feature.get("phase_label", "")) == normalized
    ]
    if exact:
        return int(exact[0].feature["phase_index"])
    # Order 8 continues with retreat/settle/complete after the TaskSpec's
    # explicit release phase.  They are post-release free-motion states.
    if normalized in {"retreat", "settle", "complete", "safe_hold"}:
        release = [
            int(node.feature["phase_index"])
            for node in phases
            if node.feature.get("phase_type") == PhaseType.RELEASE_CONTACT.value
        ]
        return (max(release) + 1) if release else None
    return None


def _knot_phase_label(knot: InteractionKnot) -> str | None:
    for guard in knot.guard_conditions:
        if guard.get("type") == "order9_task_phase":
            value = guard.get("phase_label")
            return str(value) if value is not None else None
    return None


def _require_evaluation(
    evaluated: bool,
    required: bool,
    code: str,
    knot_codes: list[str],
    warnings: list[Violation],
    knot_index: int,
    evaluation_mode: str,
) -> None:
    if evaluated:
        return
    message = _violation_message(code, knot_index)
    if required:
        _append_unique(knot_codes, code)
        return
    _append_violation(
        warnings,
        code,
        f"{message}; permitted only by {evaluation_mode}",
        severity=ViolationSeverity.WARNING,
        node_or_edge_ref=f"trajectory.knot[{knot_index}]",
    )


def _violation_message(code: str, knot_index: int) -> str:
    messages = {
        TRAJECTORY_WRENCH_BOUNDS_MISSING_CODE: "active contact is missing target/lower/upper wrench fields",
        TRAJECTORY_WRENCH_CONE_FAIL_CODE: "target wrench lies outside the candidate friction cone",
        TRAJECTORY_QP_NOT_EVALUATED_CODE: "controller QP was not evaluated",
        TRAJECTORY_COLLISION_NOT_EVALUATED_CODE: "multi-contact collision margin was not evaluated",
        TRAJECTORY_WRENCH_NOT_EVALUATED_CODE: "full wrench feasibility was not evaluated",
        ASSIGNMENT_QP_INFEASIBLE_CODE: "controller QP residual exceeded its threshold",
        ASSIGNMENT_WRENCH_INFEASIBLE_CODE: "assignment wrench feasibility failed",
        COLLISION_MARGIN_FAIL_CODE: "collision margin fell below its threshold",
    }
    return f"knot {knot_index}: {messages.get(code, code)}"


def _append_violation(
    violations: list[Violation],
    code: str,
    message: str,
    *,
    severity: ViolationSeverity = ViolationSeverity.HARD,
    node_or_edge_ref: str | None = None,
) -> None:
    if any(item.code == code and item.node_or_edge_ref == node_or_edge_ref for item in violations):
        return
    violations.append(
        Violation(
            code=code,
            severity=severity,
            message=message,
            node_or_edge_ref=node_or_edge_ref,
        )
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
