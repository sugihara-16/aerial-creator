from __future__ import annotations

"""Replay-safe stochastic ``pi_D`` rollouts around the deterministic hard gate.

The learned policy owns only the masked graph-edit choices.  The grammar and
``FeasibilityChecker`` remain deterministic, every sampled edit keeps its exact
behavior-policy probability, and a deterministic fallback never becomes an
actor transition.
"""

import math
from dataclasses import dataclass

import torch

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.policies.design_candidate_generator import DesignActionCandidate
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.order9_design_grammar import Order9DesignGrammar
from amsrr.policies.order9_design_policy import (
    ORDER9_AUTOREGRESSIVE_PI_D_VERSION,
    Order9AutoregressiveDesignPolicy,
)
from amsrr.policies.order9_design_runtime import DeterministicDesignFallback
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    DesignActionCandidateRecord,
    PolicyBehaviorTrace,
    SequentialDesignStepRecord,
    SequentialDesignTrajectoryRecord,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignAction, DesignActionType, DesignOutput
from amsrr.training.order9_ppo import order9_pi_d_behavior_trace


ORDER9_PI_D_ROLLOUT_VERSION = "order9_pi_d_masked_hard_gate_rollout_v1"


@dataclass(frozen=True)
class Order9PiDSampledStep:
    step_index: int
    partial_action_history: tuple[DesignAction, ...]
    candidates: tuple[DesignActionCandidate, ...]
    selected_candidate_index: int
    behavior_trace: PolicyBehaviorTrace


@dataclass(frozen=True)
class Order9PiDProposalSample:
    attempt_index: int
    steps: tuple[Order9PiDSampledStep, ...]
    design_output: DesignOutput | None
    feasibility_result: FeasibilityResult | None
    failure_reason: str | None

    @property
    def accepted(self) -> bool:
        return bool(
            self.design_output is not None
            and self.feasibility_result is not None
            and self.feasibility_result.feasible
            and self.failure_reason is None
        )


@dataclass(frozen=True)
class Order9PiDHardGateDecision:
    samples: tuple[Order9PiDProposalSample, ...]
    accepted_sample: Order9PiDProposalSample | None
    execution_design: DesignOutput
    execution_feasibility_result: FeasibilityResult
    used_fallback: bool
    fallback_version: str | None

    @property
    def rejected_samples(self) -> tuple[Order9PiDProposalSample, ...]:
        return tuple(sample for sample in self.samples if not sample.accepted)


def sample_order9_pi_d_proposal(
    policy: Order9AutoregressiveDesignPolicy,
    context: DesignPolicyContext,
    *,
    checker: FeasibilityChecker,
    checkpoint_sha256: str,
    attempt_index: int = 0,
) -> Order9PiDProposalSample:
    """Sample one complete edit sequence without invoking a fallback."""

    _require_sha256(checkpoint_sha256)
    if attempt_index < 0:
        raise ValueError("Order9 pi_D attempt index must be non-negative")
    if context.interaction_envelope is None:
        raise SchemaValidationError(
            "Order9 pi_D stochastic rollout requires an InteractionEnvelope"
        )
    grammar = Order9DesignGrammar(context, checker=checker)
    state = grammar.initial_state()
    history = policy.initial_history()
    sampled: list[Order9PiDSampledStep] = []
    policy.eval()
    with torch.no_grad():
        for step_index in range(policy.config.maximum_design_steps):
            candidates = grammar.candidates(state)
            output = policy.forward_step(
                context,
                state,
                candidates,
                history=history,
            )
            evaluation = policy.sample_step(output, deterministic=False)
            selected_index = int(evaluation.selected_index.item())
            selected = candidates[selected_index]
            behavior = order9_pi_d_behavior_trace(
                evaluation,
                selected_action=selected.action.to_dict(),
                checkpoint_sha256=checkpoint_sha256,
            )
            sampled.append(
                Order9PiDSampledStep(
                    step_index=step_index,
                    partial_action_history=tuple(state.action_history),
                    candidates=tuple(candidates),
                    selected_candidate_index=selected_index,
                    behavior_trace=behavior,
                )
            )
            history = policy.advance_history(output, selected_index)
            state = grammar.apply(state, selected)
            if selected.action.action_type != DesignActionType.STOP:
                continue
            design = grammar.build_design_output(state)
            feasibility = checker.check_design(
                design,
                task_spec=context.task_spec,
                irg=context.irg,
                physical_model=context.physical_model,
            )
            failure = None
            if not feasibility.feasible:
                codes = sorted(
                    {violation.code for violation in feasibility.hard_violations}
                )
                failure = "hard_feasibility:" + (
                    ",".join(codes) if codes else "unknown"
                )
            return Order9PiDProposalSample(
                attempt_index=attempt_index,
                steps=tuple(sampled),
                design_output=design,
                feasibility_result=feasibility,
                failure_reason=failure,
            )
    if not sampled:
        raise SchemaValidationError("Order9 pi_D sampled no graph-edit action")
    return Order9PiDProposalSample(
        attempt_index=attempt_index,
        steps=tuple(sampled),
        design_output=None,
        feasibility_result=None,
        failure_reason="maximum_design_steps_exceeded",
    )


def sample_order9_pi_d_with_hard_checker(
    policy: Order9AutoregressiveDesignPolicy,
    context: DesignPolicyContext,
    *,
    checker: FeasibilityChecker,
    fallback: DeterministicDesignFallback,
    checkpoint_sha256: str,
    max_proposal_attempts: int = 3,
) -> Order9PiDHardGateDecision:
    """Retry learned edit sequences and execute only a checked complete design."""

    if max_proposal_attempts < 1:
        raise ValueError("Order9 pi_D proposal attempts must be positive")
    samples: list[Order9PiDProposalSample] = []
    for attempt_index in range(max_proposal_attempts):
        sample = sample_order9_pi_d_proposal(
            policy,
            context,
            checker=checker,
            checkpoint_sha256=checkpoint_sha256,
            attempt_index=attempt_index,
        )
        samples.append(sample)
        if sample.accepted:
            assert sample.design_output is not None
            assert sample.feasibility_result is not None
            return Order9PiDHardGateDecision(
                samples=tuple(samples),
                accepted_sample=sample,
                execution_design=sample.design_output,
                execution_feasibility_result=sample.feasibility_result,
                used_fallback=False,
                fallback_version=None,
            )

    fallback_design = fallback.design(context)
    fallback_result = checker.check_design(
        fallback_design,
        task_spec=context.task_spec,
        irg=context.irg,
        physical_model=context.physical_model,
    )
    if not fallback_result.feasible:
        codes = sorted(
            {violation.code for violation in fallback_result.hard_violations}
        )
        raise SchemaValidationError(
            "deterministic pi_D fallback failed the final hard feasibility check: "
            + (",".join(codes) if codes else "unknown")
        )
    return Order9PiDHardGateDecision(
        samples=tuple(samples),
        accepted_sample=None,
        execution_design=fallback_design,
        execution_feasibility_result=fallback_result,
        used_fallback=True,
        fallback_version=fallback.fallback_version,
    )


def order9_pi_d_rejection_record(
    sample: Order9PiDProposalSample,
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    context: DesignPolicyContext,
    rejection_penalty: float,
) -> SequentialDesignTrajectoryRecord:
    """Store one rejected proposal as an independent terminal GAE segment."""

    if sample.accepted:
        raise SchemaValidationError("accepted pi_D sample is not a rejection record")
    if not math.isfinite(float(rejection_penalty)) or rejection_penalty <= 0.0:
        raise ValueError("Order9 pi_D rejection penalty must be finite and positive")
    return _proposal_record(
        sample,
        record_id=record_id,
        episode_id=episode_id,
        split=split,
        context=context,
        episode_return=-float(rejection_penalty),
        task_success=False,
        failure_reason=sample.failure_reason or "hard_checker_rejected",
    )


def order9_pi_d_executed_record(
    sample: Order9PiDProposalSample,
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    context: DesignPolicyContext,
    episode_return: float,
    task_success: bool,
    failure_reason: str | None,
) -> SequentialDesignTrajectoryRecord:
    """Bind one accepted learned design to its actual downstream task outcome."""

    if not sample.accepted:
        raise SchemaValidationError("rejected pi_D sample cannot receive task outcome")
    if task_success == (failure_reason is not None):
        raise SchemaValidationError(
            "successful pi_D execution cannot have failure_reason and failure requires one"
        )
    return _proposal_record(
        sample,
        record_id=record_id,
        episode_id=episode_id,
        split=split,
        context=context,
        episode_return=episode_return,
        task_success=task_success,
        failure_reason=failure_reason,
    )


def _proposal_record(
    sample: Order9PiDProposalSample,
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    context: DesignPolicyContext,
    episode_return: float,
    task_success: bool,
    failure_reason: str | None,
) -> SequentialDesignTrajectoryRecord:
    if context.interaction_envelope is None:
        raise SchemaValidationError("Order9 pi_D record requires InteractionEnvelope")
    if not sample.steps:
        raise SchemaValidationError("Order9 pi_D record requires sampled behavior steps")
    if not math.isfinite(float(episode_return)):
        raise SchemaValidationError("Order9 pi_D episode return must be finite")
    steps: list[SequentialDesignStepRecord] = []
    for index, sampled in enumerate(sample.steps):
        terminal = index == len(sample.steps) - 1
        steps.append(
            SequentialDesignStepRecord(
                step_index=index,
                partial_action_history=list(sampled.partial_action_history),
                candidates=[
                    DesignActionCandidateRecord(
                        candidate_index=candidate_index,
                        action=candidate.action,
                        valid=candidate.valid,
                        reason_code=candidate.reason_code,
                        score_prior=candidate.score_prior,
                    )
                    for candidate_index, candidate in enumerate(sampled.candidates)
                ],
                selected_candidate_index=sampled.selected_candidate_index,
                reward=float(episode_return) if terminal else 0.0,
                terminal=terminal,
                truncated=False,
                behavior_trace=sampled.behavior_trace,
            )
        )
    record = SequentialDesignTrajectoryRecord(
        record_id=record_id,
        episode_id=episode_id,
        task_id=context.task_spec.task_id,
        split=split,
        task_spec=context.task_spec,
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        physical_model_hash=context.physical_model.stable_hash(),
        steps=steps,
        design_output=sample.design_output,
        feasibility_result=sample.feasibility_result,
        episode_return=float(episode_return),
        task_success=bool(task_success),
        failure_reason=failure_reason,
        stage_masks=StageDecisionMasks(design_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.LEARNED_POLICY,
            source_version=ORDER9_AUTOREGRESSIVE_PI_D_VERSION,
            policy_checkpoint_sha256=sample.steps[0].behavior_trace.policy_checkpoint_sha256,
            metadata={
                "rollout_version": ORDER9_PI_D_ROLLOUT_VERSION,
                "proposal_attempt_index": sample.attempt_index,
                "hard_checker_accepted": sample.accepted,
                "executed_in_environment": sample.accepted,
                "fallback_reward_credited": False,
            },
        ),
    )
    record.validate()
    return record


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError("Order9 pi_D checkpoint hash must be SHA-256")


__all__ = [
    "ORDER9_PI_D_ROLLOUT_VERSION",
    "Order9PiDHardGateDecision",
    "Order9PiDProposalSample",
    "Order9PiDSampledStep",
    "order9_pi_d_executed_record",
    "order9_pi_d_rejection_record",
    "sample_order9_pi_d_proposal",
    "sample_order9_pi_d_with_hard_checker",
]
