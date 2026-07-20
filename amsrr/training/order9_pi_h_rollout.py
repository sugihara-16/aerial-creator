from __future__ import annotations

"""Replay-safe stochastic pi_H proposal attempts around the hard C_H gate."""

import math
from dataclasses import dataclass

import torch

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.high_level_runtime import HighLevelFallback
from amsrr.policies.order9_high_level_policy import (
    ORDER9_FULL_PI_H_VERSION,
    Order9AutoregressiveHighLevelPolicy,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    HighLevelTransitionKind,
    InteractionTrajectoryRecord,
    PolicyBehaviorTrace,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.feasibility import TrajectoryFeasibilityResult
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.training.order9_ppo import order9_pi_h_behavior_trace


ORDER9_PI_H_ROLLOUT_VERSION = "order9_pi_h_hard_gate_rollout_v1"


@dataclass(frozen=True)
class Order9PiHProposalSample:
    attempt_index: int
    trajectory: ContactWrenchTrajectory
    feasibility_result: TrajectoryFeasibilityResult
    behavior_trace: PolicyBehaviorTrace

    @property
    def accepted(self) -> bool:
        return bool(self.feasibility_result.feasible)


@dataclass(frozen=True)
class Order9PiHHardGateDecision:
    samples: tuple[Order9PiHProposalSample, ...]
    accepted_sample: Order9PiHProposalSample | None
    execution_trajectory: ContactWrenchTrajectory
    execution_feasibility_result: TrajectoryFeasibilityResult
    used_fallback: bool
    fallback_version: str | None

    @property
    def rejected_samples(self) -> tuple[Order9PiHProposalSample, ...]:
        return tuple(sample for sample in self.samples if not sample.accepted)


@dataclass(frozen=True)
class Order9PiHRolloutResult:
    physical_episode_id: str
    records: tuple[InteractionTrajectoryRecord, ...]
    learned_execution_count: int
    rejected_proposal_count: int
    fallback_execution_count: int
    gae_segment_count: int
    ignored_fallback_reward: float


@dataclass
class _PendingAcceptedProposal:
    sample: Order9PiHProposalSample
    context: HighLevelPolicyContext
    segment_episode_id: str
    decision_index: int
    reward: float = 0.0


class Order9PiHEpisodeCollector:
    """Accumulate accepted task rewards without crossing checker/fallback boundaries."""

    def __init__(
        self,
        *,
        physical_episode_id: str,
        split: DatasetSplit,
        rejection_penalty: float,
        discount_gamma: float = 0.99,
    ) -> None:
        if not physical_episode_id:
            raise SchemaValidationError(
                "Order9 pi_H physical episode id must be non-empty"
            )
        if not math.isfinite(float(rejection_penalty)) or rejection_penalty <= 0.0:
            raise ValueError("Order9 pi_H rejection penalty must be positive")
        if not math.isfinite(float(discount_gamma)) or not 0.0 <= discount_gamma <= 1.0:
            raise ValueError("Order9 pi_H discount gamma must lie in [0, 1]")
        self.physical_episode_id = physical_episode_id
        self.split = split
        self.rejection_penalty = float(rejection_penalty)
        self.discount_gamma = float(discount_gamma)
        self._records: list[InteractionTrajectoryRecord] = []
        self._pending: _PendingAcceptedProposal | None = None
        self._current_segment_id: str | None = None
        self._segment_count = 0
        self._segment_decision_index = 0
        self._rejection_count = 0
        self._accepted_count = 0
        self._fallback_count = 0
        self._ignored_fallback_reward = 0.0
        self._fallback_active = False
        self._finalized = False

    def record_decision(
        self,
        decision: Order9PiHHardGateDecision,
        *,
        context: HighLevelPolicyContext,
    ) -> None:
        """Close the previous interval and install this hard-gated decision."""

        self._require_open()
        if context.runtime_observation is None:
            raise SchemaValidationError("Order9 pi_H decision requires runtime state")
        context.runtime_observation.validate()
        next_is_learned = decision.accepted_sample is not None
        self._close_pending(terminal=not next_is_learned)
        if not next_is_learned:
            self._current_segment_id = None
            self._segment_decision_index = 0
        for sample in decision.rejected_samples:
            rejection_id = (
                f"{self.physical_episode_id}:pi_h_rejection:"
                f"{self._rejection_count:06d}"
            )
            self._records.append(
                order9_pi_h_rejection_record(
                    sample,
                    record_id=f"{rejection_id}:record",
                    episode_id=rejection_id,
                    split=self.split,
                    decision_index=0,
                    context=context,
                    rejection_penalty=self.rejection_penalty,
                )
            )
            self._rejection_count += 1
        if decision.accepted_sample is None:
            if not decision.used_fallback:
                raise SchemaValidationError(
                    "Order9 pi_H decision has neither learned execution nor fallback"
                )
            self._fallback_count += 1
            self._fallback_active = True
            return
        if decision.used_fallback:
            raise SchemaValidationError(
                "Order9 pi_H accepted sample cannot simultaneously use fallback"
            )
        if self._current_segment_id is None:
            self._current_segment_id = (
                f"{self.physical_episode_id}:pi_h_segment:{self._segment_count:04d}"
            )
            self._segment_count += 1
            self._segment_decision_index = 0
        self._pending = _PendingAcceptedProposal(
            sample=decision.accepted_sample,
            context=_copy_context(context),
            segment_episode_id=self._current_segment_id,
            decision_index=self._segment_decision_index,
        )
        self._segment_decision_index += 1
        self._accepted_count += 1
        self._fallback_active = False

    def add_environment_reward(self, reward: float) -> bool:
        """Attribute one executed reward; return false when fallback owns it."""

        self._require_open()
        if not math.isfinite(float(reward)):
            raise SchemaValidationError("Order9 pi_H environment reward must be finite")
        if self._pending is not None:
            self._pending.reward += float(reward)
            return True
        if self._fallback_active:
            self._ignored_fallback_reward += float(reward)
            return False
        raise SchemaValidationError(
            "Order9 pi_H reward arrived before an execution decision"
        )

    def finalize(
        self,
        *,
        terminal: bool,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
        terminal_reward: float = 0.0,
    ) -> Order9PiHRolloutResult:
        self._require_open()
        if terminal == truncated:
            raise SchemaValidationError(
                "Order9 pi_H finalization requires exactly one terminal/truncated boundary"
            )
        for value in (bootstrap_value, terminal_reward):
            if not math.isfinite(float(value)):
                raise SchemaValidationError("Order9 pi_H final values must be finite")
        if terminal and not math.isclose(float(bootstrap_value), 0.0, abs_tol=1.0e-12):
            raise SchemaValidationError("terminal pi_H boundary cannot bootstrap")
        if self._pending is not None:
            self._pending.reward += float(terminal_reward)
            self._close_pending(
                terminal=terminal,
                truncated=truncated,
                bootstrap_value=bootstrap_value,
            )
        elif self._fallback_active:
            self._ignored_fallback_reward += float(terminal_reward)
        self._populate_decision_returns()
        self._finalized = True
        return Order9PiHRolloutResult(
            physical_episode_id=self.physical_episode_id,
            records=tuple(self._records),
            learned_execution_count=self._accepted_count,
            rejected_proposal_count=self._rejection_count,
            fallback_execution_count=self._fallback_count,
            gae_segment_count=self._segment_count + self._rejection_count,
            ignored_fallback_reward=self._ignored_fallback_reward,
        )

    def _close_pending(
        self,
        *,
        terminal: bool,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        pending = self._pending
        if pending is None:
            return
        record = order9_pi_h_executed_record(
            pending.sample,
            record_id=(
                f"{pending.segment_episode_id}:decision:"
                f"{pending.decision_index:06d}"
            ),
            episode_id=pending.segment_episode_id,
            split=self.split,
            decision_index=pending.decision_index,
            context=pending.context,
            decision_reward=pending.reward,
            decision_return=pending.reward,
            terminal=terminal,
            truncated=truncated,
            bootstrap_value=bootstrap_value,
        )
        self._records.append(record)
        self._pending = None

    def _populate_decision_returns(self) -> None:
        by_segment: dict[str, list[InteractionTrajectoryRecord]] = {}
        for record in self._records:
            if record.transition_kind == HighLevelTransitionKind.EXECUTED_TRAJECTORY:
                by_segment.setdefault(record.episode_id, []).append(record)
        for records in by_segment.values():
            running = 0.0
            for record in reversed(
                sorted(records, key=lambda item: item.decision_index)
            ):
                if record.truncated:
                    running = float(record.bootstrap_value)
                elif record.terminal:
                    running = 0.0
                if record.decision_reward is None:
                    raise AssertionError("validated pi_H reward disappeared")
                running = float(record.decision_reward) + self.discount_gamma * running
                record.decision_return = running
                record.validate()

    def _require_open(self) -> None:
        if self._finalized:
            raise SchemaValidationError("Order9 pi_H collector is already finalized")


def sample_order9_pi_h_with_hard_checker(
    policy: Order9AutoregressiveHighLevelPolicy,
    context: HighLevelPolicyContext,
    *,
    checker: ContactWrenchTrajectoryFeasibilityChecker,
    fallback: HighLevelFallback,
    checkpoint_sha256: str,
    max_proposal_attempts: int = 2,
) -> Order9PiHHardGateDecision:
    """Sample every behavior attempt and execute only an unchanged C_H pass."""

    if max_proposal_attempts < 1:
        raise ValueError("Order9 pi_H proposal attempts must be positive")
    _require_sha256(checkpoint_sha256)
    if context.runtime_observation is None:
        raise SchemaValidationError("Order9 stochastic pi_H requires runtime state")
    samples: list[Order9PiHProposalSample] = []
    policy.eval()
    for attempt_index in range(max_proposal_attempts):
        with torch.no_grad():
            output = policy.forward_contexts([context])
            action = policy.sample_action(output, deterministic=False)
            evaluation = policy.evaluate_action(output, action)
            trajectory = policy.decode_action(context, action, batch_index=0)
        behavior = order9_pi_h_behavior_trace(
            action,
            evaluation,
            batch_index=0,
            candidate_count=len(context.contact_candidate_set.candidates),
            object_count=len(context.runtime_observation.object_states),
            checkpoint_sha256=checkpoint_sha256,
        )
        feasibility = checker.check(trajectory, context)
        sample = Order9PiHProposalSample(
            attempt_index=attempt_index,
            trajectory=trajectory,
            feasibility_result=feasibility,
            behavior_trace=behavior,
        )
        samples.append(sample)
        if feasibility.feasible:
            return Order9PiHHardGateDecision(
                samples=tuple(samples),
                accepted_sample=sample,
                execution_trajectory=trajectory,
                execution_feasibility_result=feasibility,
                used_fallback=False,
                fallback_version=None,
            )

    fallback_trajectory = fallback.fallback(context)
    fallback_result = checker.check(fallback_trajectory, context)
    if not fallback_result.feasible:
        codes = sorted(
            violation.code for violation in fallback_result.hard_violations
        )
        raise RuntimeError(
            "deterministic pi_H fallback failed C_H: " + ",".join(codes)
        )
    return Order9PiHHardGateDecision(
        samples=tuple(samples),
        accepted_sample=None,
        execution_trajectory=fallback_trajectory,
        execution_feasibility_result=fallback_result,
        used_fallback=True,
        fallback_version=fallback.fallback_version,
    )


def order9_pi_h_rejection_record(
    sample: Order9PiHProposalSample,
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    decision_index: int,
    context: HighLevelPolicyContext,
    rejection_penalty: float,
) -> InteractionTrajectoryRecord:
    """Persist one rejected action as its own terminal GAE segment."""

    if sample.accepted:
        raise SchemaValidationError("accepted pi_H sample is not a rejection record")
    if not math.isfinite(float(rejection_penalty)) or rejection_penalty <= 0.0:
        raise ValueError("Order9 pi_H rejection penalty must be finite and positive")
    return _proposal_record(
        sample,
        record_id=record_id,
        episode_id=episode_id,
        split=split,
        decision_index=decision_index,
        context=context,
        decision_reward=-float(rejection_penalty),
        decision_return=-float(rejection_penalty),
        terminal=True,
        truncated=False,
        bootstrap_value=0.0,
        transition_kind=HighLevelTransitionKind.CHECKER_REJECTION,
    )


def order9_pi_h_executed_record(
    sample: Order9PiHProposalSample,
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    decision_index: int,
    context: HighLevelPolicyContext,
    decision_reward: float,
    decision_return: float,
    terminal: bool,
    truncated: bool = False,
    bootstrap_value: float = 0.0,
) -> InteractionTrajectoryRecord:
    """Bind an accepted sampled proposal to its actual environment outcome."""

    if not sample.accepted:
        raise SchemaValidationError("rejected pi_H sample cannot be executed")
    return _proposal_record(
        sample,
        record_id=record_id,
        episode_id=episode_id,
        split=split,
        decision_index=decision_index,
        context=context,
        decision_reward=decision_reward,
        decision_return=decision_return,
        terminal=terminal,
        truncated=truncated,
        bootstrap_value=bootstrap_value,
        transition_kind=HighLevelTransitionKind.EXECUTED_TRAJECTORY,
    )


def _proposal_record(
    sample: Order9PiHProposalSample,
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    decision_index: int,
    context: HighLevelPolicyContext,
    decision_reward: float,
    decision_return: float,
    terminal: bool,
    truncated: bool,
    bootstrap_value: float,
    transition_kind: HighLevelTransitionKind,
) -> InteractionTrajectoryRecord:
    observation = context.runtime_observation
    if observation is None:
        raise SchemaValidationError("Order9 pi_H record requires runtime state")
    selected_ids = sorted(
        {
            assignment.candidate_id
            for knot in sample.trajectory.knots
            for assignment in knot.contact_assignments
        }
    )
    record = InteractionTrajectoryRecord(
        record_id=record_id,
        episode_id=episode_id,
        task_id=context.irg.task_id,
        split=split,
        decision_index=decision_index,
        decision_time_s=observation.time_s,
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        morphology_graph=context.morphology_graph,
        contact_candidate_set=context.contact_candidate_set,
        runtime_observation=observation,
        trajectory=sample.trajectory,
        selected_candidate_ids=selected_ids,
        assignment_feasibility_results=[
            result.assignment_result
            for result in sample.feasibility_result.knot_results
        ],
        decision_return=float(decision_return),
        decision_reward=float(decision_reward),
        stage_masks=StageDecisionMasks(high_level_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.LEARNED_POLICY,
            source_version=ORDER9_FULL_PI_H_VERSION,
            policy_checkpoint_sha256=sample.behavior_trace.policy_checkpoint_sha256,
            metadata={
                "rollout_version": ORDER9_PI_H_ROLLOUT_VERSION,
                "checker_version": sample.feasibility_result.checker_version,
                "checker_rejected": transition_kind
                == HighLevelTransitionKind.CHECKER_REJECTION,
                "executed_in_environment": transition_kind
                == HighLevelTransitionKind.EXECUTED_TRAJECTORY,
                "fallback_reward_credited": False,
            },
        ),
        trajectory_feasibility_result=sample.feasibility_result,
        terminal=bool(terminal),
        truncated=bool(truncated),
        bootstrap_value=float(bootstrap_value),
        behavior_trace=sample.behavior_trace,
        transition_kind=transition_kind,
        proposal_attempt_index=sample.attempt_index,
        fallback_reward_credited=False,
    )
    record.validate()
    return record


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError("Order9 pi_H checkpoint hash must be SHA-256")


def _copy_context(context: HighLevelPolicyContext) -> HighLevelPolicyContext:
    observation = context.runtime_observation
    return HighLevelPolicyContext(
        irg=type(context.irg).from_dict(context.irg.to_dict()),
        interaction_envelope=type(context.interaction_envelope).from_dict(
            context.interaction_envelope.to_dict()
        ),
        morphology_graph=type(context.morphology_graph).from_dict(
            context.morphology_graph.to_dict()
        ),
        contact_candidate_set=type(context.contact_candidate_set).from_dict(
            context.contact_candidate_set.to_dict()
        ),
        runtime_observation=(
            None
            if observation is None
            else type(observation).from_dict(observation.to_dict())
        ),
    )


__all__ = [
    "ORDER9_PI_H_ROLLOUT_VERSION",
    "Order9PiHEpisodeCollector",
    "Order9PiHHardGateDecision",
    "Order9PiHProposalSample",
    "Order9PiHRolloutResult",
    "order9_pi_h_executed_record",
    "order9_pi_h_rejection_record",
    "sample_order9_pi_h_with_hard_checker",
]
