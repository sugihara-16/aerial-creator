from __future__ import annotations

"""Production hard-gate adapter for raw pi_H and resolved execution paths."""

import math

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.feasibility import (
    TrajectoryFeasibilityResult,
    TrajectoryKnotFeasibilityResult,
    Violation,
    ViolationSeverity,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolutionEvidence,
    Order9PostureTrajectoryResolver,
    Order9ResolvedPostureTrajectory,
)
from amsrr.utils.hashing import stable_hash


ORDER9_POSTURE_RESOLVING_HARD_GATE_VERSION = (
    "order9_posture_resolving_hard_gate_v2_native_convex"
)
POSTURE_RESOLUTION_FAILED_CODE = "E_POSTURE_RESOLUTION_FAILED"


class Order9PostureResolvingHardChecker:
    """Check raw semantics, resolve posture, then check physical execution.

    ``check`` returns a raw-knot-aligned result so a learned pi_H transition
    remains bound to the exact action it sampled.  ``execution_trajectory``
    exposes only the already-checked detached resolved copy for pi_L/QPID.
    """

    checker_version = ORDER9_POSTURE_RESOLVING_HARD_GATE_VERSION

    def __init__(
        self,
        *,
        physical_model: PhysicalModel,
        resolved_checker: ContactWrenchTrajectoryFeasibilityChecker,
        resolver: Order9PostureTrajectoryResolver | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.resolved_checker = resolved_checker
        self.resolver = resolver or Order9PostureTrajectoryResolver(
            physical_model
        )
        self.raw_checker = ContactWrenchTrajectoryFeasibilityChecker(
            config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
        )
        self._last_key: tuple[str, str] | None = None
        self._last_resolution: Order9ResolvedPostureTrajectory | None = None
        self._last_result: TrajectoryFeasibilityResult | None = None

    @property
    def config(self) -> ContactWrenchTrajectoryCheckerConfig:
        return self.resolved_checker.config

    @property
    def physics_evaluator(self) -> object:
        return self.resolved_checker.physics_evaluator

    @property
    def reachability_evaluator(self) -> object:
        return self.resolved_checker.reachability_evaluator

    def check(
        self,
        trajectory: ContactWrenchTrajectory,
        context: HighLevelPolicyContext,
    ) -> TrajectoryFeasibilityResult:
        raw_hash = stable_hash(trajectory.to_dict())
        key = self._key(trajectory, context)
        raw_result = self.raw_checker.check(trajectory, context)
        if not raw_result.feasible:
            result = _with_boundary_metadata(
                raw_result,
                raw_hash=raw_hash,
                resolution=None,
                resolved_result=None,
            )
            self._remember(key, result=result, resolution=None)
            return result
        try:
            initial_q = _runtime_initial_q(
                context,
                self.physical_model,
            )
            resolution = self.resolver.resolve(
                context=context,
                raw_trajectory=trajectory,
                initial_joint_positions_rad=initial_q,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            result = _resolution_failure_result(
                raw_result,
                raw_hash=raw_hash,
                error=error,
            )
            self._remember(key, result=result, resolution=None)
            return result
        resolved_result = self.resolved_checker.check(
            resolution.trajectory,
            context,
        )
        result = _raw_aligned_result(
            raw_trajectory=trajectory,
            raw_result=raw_result,
            resolved_result=resolved_result,
            resolution_evidence=resolution.evidence,
        )
        if stable_hash(trajectory.to_dict()) != raw_hash:
            raise RuntimeError(
                "posture-resolving hard gate mutated the raw pi_H proposal"
            )
        self._remember(key, result=result, resolution=resolution)
        return result

    def execution_trajectory(
        self,
        trajectory: ContactWrenchTrajectory,
        context: HighLevelPolicyContext,
    ) -> ContactWrenchTrajectory:
        if self._last_key != self._key(trajectory, context):
            raise RuntimeError(
                "execution trajectory requested without the matching check"
            )
        if (
            self._last_result is None
            or not self._last_result.feasible
            or self._last_resolution is None
        ):
            raise RuntimeError(
                "execution trajectory requested for a rejected raw proposal"
            )
        return ContactWrenchTrajectory.from_dict(
            self._last_resolution.trajectory.to_dict()
        )

    def resolution_evidence(
        self,
        trajectory: ContactWrenchTrajectory,
        context: HighLevelPolicyContext,
    ) -> Order9PostureResolutionEvidence:
        if (
            self._last_key != self._key(trajectory, context)
            or self._last_resolution is None
        ):
            raise RuntimeError(
                "resolution evidence requested without the matching check"
            )
        return self._last_resolution.evidence

    def _key(
        self,
        trajectory: ContactWrenchTrajectory,
        context: HighLevelPolicyContext,
    ) -> tuple[str, str]:
        observation = context.runtime_observation
        observation_hash = (
            "missing"
            if observation is None
            else stable_hash(observation.to_dict())
        )
        return stable_hash(trajectory.to_dict()), observation_hash

    def _remember(
        self,
        key: tuple[str, str],
        *,
        result: TrajectoryFeasibilityResult,
        resolution: Order9ResolvedPostureTrajectory | None,
    ) -> None:
        self._last_key = key
        self._last_result = result
        self._last_resolution = resolution


def _runtime_initial_q(
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
) -> dict[str, float]:
    observation = context.runtime_observation
    if observation is None:
        raise SchemaValidationError(
            "posture-resolving hard gate requires runtime joint state"
        )
    states = {state.module_id: state for state in observation.module_states}
    ordered_ids = ordered_global_dock_joint_ids(
        context.morphology_graph,
        physical_model,
    )
    output: dict[str, float] = {}
    for global_id in ordered_ids:
        module_label, local_id = global_id.split(":", 1)
        module_id = int(module_label.removeprefix("module_"))
        state = states.get(module_id)
        if state is None or local_id not in state.joint_positions:
            raise SchemaValidationError(
                "posture-resolving hard gate lacks measured joint "
                f"{global_id}"
            )
        output[global_id] = float(state.joint_positions[local_id])
    return output


def _resolution_failure_result(
    raw_result: TrajectoryFeasibilityResult,
    *,
    raw_hash: str,
    error: Exception,
) -> TrajectoryFeasibilityResult:
    result = TrajectoryFeasibilityResult.from_dict(raw_result.to_dict())
    result.feasible = False
    result.hard_violations.append(
        Violation(
            code=POSTURE_RESOLUTION_FAILED_CODE,
            severity=ViolationSeverity.HARD,
            message=f"{type(error).__name__}:{error}",
            node_or_edge_ref="trajectory.posture_resolution",
        )
    )
    for knot in result.knot_results:
        if POSTURE_RESOLUTION_FAILED_CODE not in knot.violation_codes:
            knot.violation_codes.append(POSTURE_RESOLUTION_FAILED_CODE)
    result.checker_version = ORDER9_POSTURE_RESOLVING_HARD_GATE_VERSION
    result.metadata = {
        **result.metadata,
        "raw_pi_h_trajectory_hash": raw_hash,
        "raw_proposal_mutated": False,
        "posture_resolution_succeeded": False,
        "posture_resolution_error_type": type(error).__name__,
    }
    result.validate()
    return result


def _raw_aligned_result(
    *,
    raw_trajectory: ContactWrenchTrajectory,
    raw_result: TrajectoryFeasibilityResult,
    resolved_result: TrajectoryFeasibilityResult,
    resolution_evidence: Order9PostureResolutionEvidence,
) -> TrajectoryFeasibilityResult:
    raw_times = [float(knot.t_rel_s) for knot in raw_trajectory.knots]
    resolved_knots = resolved_result.knot_results
    aligned: list[TrajectoryKnotFeasibilityResult] = []
    for raw_index, raw_time in enumerate(raw_times):
        lower = -math.inf if raw_index == 0 else raw_times[raw_index - 1]
        group = [
            item
            for item in resolved_knots
            if item.t_rel_s > lower + 1.0e-12
            and item.t_rel_s <= raw_time + 1.0e-12
        ]
        exact = min(
            resolved_knots,
            key=lambda item: abs(float(item.t_rel_s) - raw_time),
        )
        if abs(float(exact.t_rel_s) - raw_time) > 1.0e-9:
            raise RuntimeError(
                "resolved posture trajectory omitted a raw knot time"
            )
        codes = sorted(
            {
                code
                for item in group or [exact]
                for code in item.violation_codes
            }
        )
        margins = {
            f"resolved_{item.knot_index}.{name}": float(value)
            for item in group or [exact]
            for name, value in item.margins.items()
        }
        aligned.append(
            TrajectoryKnotFeasibilityResult(
                knot_index=raw_index,
                t_rel_s=raw_time,
                assignment_result=type(exact.assignment_result).from_dict(
                    exact.assignment_result.to_dict()
                ),
                qp_evaluated=all(
                    item.qp_evaluated for item in group or [exact]
                ),
                collision_evaluated=all(
                    item.collision_evaluated for item in group or [exact]
                ),
                wrench_evaluated=all(
                    item.wrench_evaluated for item in group or [exact]
                ),
                margins=margins,
                violation_codes=codes,
            )
        )
    margins = {
        **{
            f"raw.{name}": float(value)
            for name, value in raw_result.margins.items()
        },
        **{
            f"resolved.{name}": float(value)
            for name, value in resolved_result.margins.items()
        },
        "resolver.maximum_anchor_position_error_m": (
            resolution_evidence.maximum_anchor_position_error_m
        ),
        "resolver.maximum_anchor_attitude_error_rad": (
            resolution_evidence.maximum_anchor_attitude_error_rad
        ),
        "resolver.maximum_joint_rate_rad_s": (
            resolution_evidence.maximum_joint_rate_rad_s
        ),
        "resolver.minimum_joint_rate_margin_rad_s": (
            resolution_evidence.minimum_joint_rate_margin_rad_s
        ),
        **(
            {}
            if resolution_evidence.minimum_collision_clearance_m is None
            else {
                "resolver.minimum_collision_clearance_m": (
                    resolution_evidence.minimum_collision_clearance_m
                )
            }
        ),
    }
    result = TrajectoryFeasibilityResult(
        feasible=resolved_result.feasible,
        hard_violations=[
            Violation.from_dict(value.to_dict())
            for value in resolved_result.hard_violations
        ],
        warnings=[
            *(
                Violation.from_dict(value.to_dict())
                for value in raw_result.warnings
            ),
            *(
                Violation.from_dict(value.to_dict())
                for value in resolved_result.warnings
            ),
        ],
        knot_results=aligned,
        margins=margins,
        checker_version=ORDER9_POSTURE_RESOLVING_HARD_GATE_VERSION,
        contract_version=raw_trajectory.contract_version,
        metadata={
            "raw_pi_h_trajectory_hash": (
                resolution_evidence.raw_trajectory_hash
            ),
            "resolved_trajectory_hash": (
                resolution_evidence.resolved_trajectory_hash
            ),
            "posture_resolution_identity_hash": stable_hash(
                resolution_evidence.identity_dict()
            ),
            "posture_resolver_version": (
                resolution_evidence.resolver_version
            ),
            "posture_collision_gate_status": (
                resolution_evidence.collision_gate_status
            ),
            "posture_collision_gate_version": (
                resolution_evidence.collision_gate_version
            ),
            "posture_collision_maximum_violating_pair_count": (
                resolution_evidence.maximum_collision_violating_pair_count
            ),
            "resolved_checker_version": (
                resolved_result.checker_version
            ),
            "raw_proposal_mutated": False,
            "posture_resolution_succeeded": True,
            "resolved_knot_count": len(resolved_result.knot_results),
        },
    )
    result.validate()
    return result


def _with_boundary_metadata(
    result: TrajectoryFeasibilityResult,
    *,
    raw_hash: str,
    resolution: Order9ResolvedPostureTrajectory | None,
    resolved_result: TrajectoryFeasibilityResult | None,
) -> TrajectoryFeasibilityResult:
    output = TrajectoryFeasibilityResult.from_dict(result.to_dict())
    output.checker_version = ORDER9_POSTURE_RESOLVING_HARD_GATE_VERSION
    output.metadata = {
        **output.metadata,
        "raw_pi_h_trajectory_hash": raw_hash,
        "raw_proposal_mutated": False,
        "posture_resolution_succeeded": resolution is not None,
        "resolved_checker_ran": resolved_result is not None,
    }
    output.validate()
    return output


__all__ = [
    "ORDER9_POSTURE_RESOLVING_HARD_GATE_VERSION",
    "POSTURE_RESOLUTION_FAILED_CODE",
    "Order9PostureResolvingHardChecker",
]
