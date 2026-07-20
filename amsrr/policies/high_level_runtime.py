from __future__ import annotations

"""Order 9 high-level responsibility boundary.

``pi_H`` is only the learned proposal policy. The deterministic teacher,
hard checker C_H, fallback, and rolling executor remain separate objects.
"""

from dataclasses import dataclass
from typing import Protocol

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.feasibility import TrajectoryFeasibilityResult
from amsrr.schemas.policies import ContactWrenchTrajectory


class HighLevelProposalPolicy(Protocol):
    """Pure learned pi_H interface: propose, but never check or fall back."""

    def propose(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        ...


class HighLevelTeacher(Protocol):
    """Deterministic supervision source used only for data/warm-up."""

    @property
    def teacher_version(self) -> str:
        ...

    def teach(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        ...


class HighLevelFallback(Protocol):
    """Deterministic safe behavior used after rejected proposals."""

    @property
    def fallback_version(self) -> str:
        ...

    def fallback(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        ...


@dataclass(frozen=True)
class RejectedTrajectoryProposal:
    attempt_index: int
    reason: str
    feasibility_result: TrajectoryFeasibilityResult | None
    trajectory: ContactWrenchTrajectory | None = None


@dataclass(frozen=True)
class HighLevelRuntimeDecision:
    trajectory: ContactWrenchTrajectory
    feasibility_result: TrajectoryFeasibilityResult
    used_fallback: bool
    accepted_proposal_attempt: int | None
    rejected_proposals: tuple[RejectedTrajectoryProposal, ...]
    fallback_version: str | None


class HighLevelTrajectoryRuntime:
    """Resample/reject/fallback coordinator; never projects a pi_H proposal."""

    def __init__(
        self,
        *,
        proposal_policy: HighLevelProposalPolicy,
        checker: ContactWrenchTrajectoryFeasibilityChecker,
        fallback: HighLevelFallback,
        executor: ContactWrenchTrajectoryExecutor | None = None,
        max_proposal_attempts: int = 2,
    ) -> None:
        if max_proposal_attempts < 1:
            raise ValueError("max_proposal_attempts must be positive")
        self.proposal_policy = proposal_policy
        self.checker = checker
        self.fallback_policy = fallback
        self.executor = executor or ContactWrenchTrajectoryExecutor()
        self.max_proposal_attempts = max_proposal_attempts

    def plan_and_install(
        self,
        context: HighLevelPolicyContext,
        *,
        plan_start_time_s: float,
    ) -> HighLevelRuntimeDecision:
        rejected: list[RejectedTrajectoryProposal] = []
        for attempt_index in range(self.max_proposal_attempts):
            try:
                proposal = self.proposal_policy.propose(context)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                rejected.append(
                    RejectedTrajectoryProposal(
                        attempt_index=attempt_index,
                        reason=f"proposal_error:{type(exc).__name__}:{exc}",
                        feasibility_result=None,
                        trajectory=None,
                    )
                )
                continue
            result = self.checker.check(proposal, context)
            if result.feasible:
                self.executor.install(
                    proposal,
                    plan_start_time_s=plan_start_time_s,
                )
                return HighLevelRuntimeDecision(
                    trajectory=proposal,
                    feasibility_result=result,
                    used_fallback=False,
                    accepted_proposal_attempt=attempt_index,
                    rejected_proposals=tuple(rejected),
                    fallback_version=None,
                )
            rejected.append(
                RejectedTrajectoryProposal(
                    attempt_index=attempt_index,
                    reason="hard_checker_rejected",
                    feasibility_result=result,
                    trajectory=proposal,
                )
            )

        fallback_trajectory = self.fallback_policy.fallback(context)
        fallback_result = self.checker.check(fallback_trajectory, context)
        if not fallback_result.feasible:
            codes = sorted(
                {violation.code for violation in fallback_result.hard_violations}
            )
            raise RuntimeError(
                "deterministic high-level fallback failed C_H: " + ",".join(codes)
            )
        self.executor.install(
            fallback_trajectory,
            plan_start_time_s=plan_start_time_s,
        )
        return HighLevelRuntimeDecision(
            trajectory=fallback_trajectory,
            feasibility_result=fallback_result,
            used_fallback=True,
            accepted_proposal_attempt=None,
            rejected_proposals=tuple(rejected),
            fallback_version=self.fallback_policy.fallback_version,
        )


@dataclass(frozen=True)
class PlannerTeacherAdapter:
    planner: object
    teacher_version: str

    def teach(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        plan = getattr(self.planner, "plan", None)
        if plan is None:
            raise TypeError("teacher planner must expose plan(context)")
        return plan(context)


@dataclass(frozen=True)
class PlannerFallbackAdapter:
    planner: object
    fallback_version: str

    def fallback(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        plan = getattr(self.planner, "plan", None)
        if plan is None:
            raise TypeError("fallback planner must expose plan(context)")
        return plan(context)
