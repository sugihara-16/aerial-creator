from __future__ import annotations

"""Fail-closed runtime boundary around the learned sequential Order 9 ``pi_D``."""

from dataclasses import dataclass
from typing import Protocol

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignOutput


ORDER9_DESIGN_RUNTIME_VERSION = "order9_pi_d_retry_checker_fallback_v1"


class LearnedDesignProposal(Protocol):
    """Learned proposal interface; deterministic masks may use ``checker``."""

    def propose(
        self,
        context: DesignPolicyContext,
        *,
        deterministic: bool,
        checker: FeasibilityChecker | None = None,
    ) -> DesignOutput:
        ...


class DeterministicDesignFallback(Protocol):
    @property
    def fallback_version(self) -> str:
        ...

    def design(self, context: DesignPolicyContext) -> DesignOutput:
        ...


@dataclass(frozen=True)
class Order9DesignRuntimeConfig:
    maximum_proposal_attempts: int = 3
    deterministic_proposal: bool = False

    def __post_init__(self) -> None:
        if self.maximum_proposal_attempts < 1:
            raise ValueError("Order9 pi_D runtime requires at least one proposal attempt")


@dataclass(frozen=True)
class Order9DesignRuntimeDecision:
    design_output: DesignOutput
    feasibility_result: FeasibilityResult
    used_fallback: bool
    proposal_attempt_count: int
    rejected_proposal_reasons: tuple[str, ...]
    checker_version: str
    fallback_version: str | None
    runtime_version: str = ORDER9_DESIGN_RUNTIME_VERSION


class Order9DesignRuntime:
    """Retry learned designs, hard-check them, then use a checked fallback.

    No branch edits, repairs, projects, or otherwise mutates a proposal.  A
    rejected proposal is useful as an RL outcome, but is never handed to the
    assembly or object-task runtime.
    """

    def __init__(
        self,
        *,
        proposal: LearnedDesignProposal,
        checker: FeasibilityChecker,
        fallback: DeterministicDesignFallback,
        config: Order9DesignRuntimeConfig | None = None,
    ) -> None:
        self.proposal = proposal
        self.checker = checker
        self.fallback = fallback
        self.config = config or Order9DesignRuntimeConfig()

    def design(self, context: DesignPolicyContext) -> Order9DesignRuntimeDecision:
        rejected: list[str] = []
        attempts = 0
        for _ in range(self.config.maximum_proposal_attempts):
            attempts += 1
            try:
                candidate = self.proposal.propose(
                    context,
                    deterministic=self.config.deterministic_proposal,
                    checker=self.checker,
                )
            except (RuntimeError, SchemaValidationError, TypeError, ValueError) as exc:
                rejected.append(f"proposal_error:{type(exc).__name__}:{exc}")
                continue
            result = self._check(candidate, context)
            if result.feasible:
                return Order9DesignRuntimeDecision(
                    design_output=candidate,
                    feasibility_result=result,
                    used_fallback=False,
                    proposal_attempt_count=attempts,
                    rejected_proposal_reasons=tuple(rejected),
                    checker_version=result.checker_version,
                    fallback_version=None,
                )
            rejected.append(_feasibility_rejection(result))

        fallback_design = self.fallback.design(context)
        fallback_result = self._check(fallback_design, context)
        if not fallback_result.feasible:
            raise SchemaValidationError(
                "deterministic pi_D fallback failed the final hard feasibility check: "
                + _feasibility_rejection(fallback_result)
            )
        return Order9DesignRuntimeDecision(
            design_output=fallback_design,
            feasibility_result=fallback_result,
            used_fallback=True,
            proposal_attempt_count=attempts,
            rejected_proposal_reasons=tuple(rejected),
            checker_version=fallback_result.checker_version,
            fallback_version=self.fallback.fallback_version,
        )

    def _check(
        self,
        design: DesignOutput,
        context: DesignPolicyContext,
    ) -> FeasibilityResult:
        return self.checker.check_design(
            design,
            task_spec=context.task_spec,
            irg=context.irg,
            physical_model=context.physical_model,
        )


@dataclass(frozen=True)
class DesignPolicyFallbackAdapter:
    policy: object
    fallback_version: str

    def design(self, context: DesignPolicyContext) -> DesignOutput:
        method = getattr(self.policy, "design", None)
        if method is None:
            raise TypeError("deterministic pi_D fallback must expose design(context)")
        design = method(context)
        if not isinstance(design, DesignOutput):
            raise TypeError("deterministic pi_D fallback returned an invalid type")
        return design


def _feasibility_rejection(result: FeasibilityResult) -> str:
    codes = sorted({violation.code for violation in result.hard_violations})
    return "hard_feasibility:" + (",".join(codes) if codes else "unknown")
