from __future__ import annotations

"""Replay-safe PPO utilities and optimizer updates for Order 9 policies."""

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
    Order3PolicyInference,
    order3_actor_feature_vector,
)
from amsrr.policies.order9_design_policy import (
    ORDER9_AUTOREGRESSIVE_PI_D_VERSION,
    Order9AutoregressiveDesignPolicy,
    Order9PiDActionEvaluation,
)
from amsrr.policies.order9_high_level_policy import (
    ORDER9_FULL_PI_H_VERSION,
    Order9AutoregressiveHighLevelPolicy,
    Order9PiHAction,
    Order9PiHActionEvaluation,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    ORDER9_PI_L_POLICY_VERSION,
    Order9LowLevelActorCriticStep,
    Order9PhaseConditionedActorCritic,
    order9_phase_actor_feature_vector,
)
from amsrr.policies.order9_policy_command import order9_pi_l_reference_command
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.datasets import (
    HighLevelTransitionKind,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    PolicyBehaviorTrace,
    SequentialDesignTrajectoryRecord,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.training.order9_curriculum import Order9PPOOptimizationConfig
from amsrr.training.order9_offline_training import (
    reconstruct_order9_pi_d_teacher_trace,
)
from amsrr.training.order9_pi_d_learning import (
    compute_order9_pi_d_masked_ppo_loss,
)
from amsrr.training.order9_pi_h_learning import clipped_ppo_surrogate_loss
from amsrr.training.order9_pi_l_learning import (
    compute_order9_pi_l_ppo_loss,
)


ORDER9_PPO_REPLAY_VERSION = "order9_ppo_exact_behavior_replay_v2_batched_sequence"
ORDER9_PI_L_ACTION_SEMANTICS = (
    "squashed_complete_policy_command_and_module_joint_action_v3_actor_graph_frame"
)
ORDER9_PI_L_GRAPH_JOINT_SUMMARY_ALL = "all_present_joints"
ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED = "non_fixed_joints_only"
ORDER9_PI_H_ACTION_SEMANTICS = "full_autoregressive_trajectory_tensor_action_v1"
ORDER9_PI_D_ACTION_SEMANTICS = "masked_grammar_candidate_index_v1"


@dataclass(frozen=True)
class Order9GAETransition:
    reward: float
    value: float
    terminal: bool
    truncated: bool
    bootstrap_value: float = 0.0


@dataclass(frozen=True)
class Order9GAEResult:
    advantages: tuple[float, ...]
    returns: tuple[float, ...]


@dataclass(frozen=True)
class _PiLReplayCache:
    index_by_record_id: dict[str, int]
    actor_graph_observations: tuple[RuntimeObservation, ...]
    actor_features: torch.Tensor
    phase_features: torch.Tensor
    privileged_disturbance_body: torch.Tensor


@dataclass
class Order9PPOUpdateResult(SchemaBase):
    replay_version: str
    policy_family: str
    sample_count: int
    requested_epoch_count: int
    completed_epoch_count: int
    optimizer_step_count: int
    actor_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approximate_kl: float
    clipped_fraction: float
    early_stopped_for_kl: bool
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.replay_version != ORDER9_PPO_REPLAY_VERSION:
            raise SchemaValidationError("Order9 PPO replay version mismatch")
        if self.policy_family not in {"pi_l", "pi_h", "pi_d"}:
            raise SchemaValidationError("Order9 PPO policy family is invalid")
        if min(
            self.sample_count,
            self.requested_epoch_count,
            self.completed_epoch_count,
            self.optimizer_step_count,
        ) < 1:
            raise SchemaValidationError("Order9 PPO counts must be positive")
        if self.completed_epoch_count > self.requested_epoch_count:
            raise SchemaValidationError("Order9 PPO completed too many epochs")
        for value in (
            self.actor_loss,
            self.value_loss,
            self.entropy,
            self.total_loss,
            self.approximate_kl,
            self.clipped_fraction,
        ):
            if not math.isfinite(float(value)):
                raise SchemaValidationError("Order9 PPO metrics must be finite")
        if not 0.0 <= self.clipped_fraction <= 1.0:
            raise SchemaValidationError("Order9 PPO clipped_fraction must lie in [0, 1]")


def compute_order9_gae(
    transitions: Sequence[Order9GAETransition],
    *,
    gamma: float,
    gae_lambda: float,
) -> Order9GAEResult:
    """Compute GAE without leaking recurrence across terminal/reset boundaries."""

    if not transitions:
        raise ValueError("Order9 GAE requires at least one transition")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("Order9 GAE gamma/lambda must lie in [0, 1]")
    for transition in transitions:
        if transition.terminal and transition.truncated:
            raise SchemaValidationError("Order9 GAE boundary cannot be terminal and truncated")
        values = (
            transition.reward,
            transition.value,
            transition.bootstrap_value,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise SchemaValidationError("Order9 GAE transition contains non-finite values")
        if not transition.truncated and not math.isclose(
            transition.bootstrap_value, 0.0, abs_tol=1.0e-12
        ):
            raise SchemaValidationError(
                "Order9 GAE bootstrap is restricted to truncation boundaries"
            )
    if not (transitions[-1].terminal or transitions[-1].truncated):
        raise SchemaValidationError(
            "Order9 GAE fragment must end in a terminal or truncation boundary"
        )

    advantages = [0.0] * len(transitions)
    next_advantage = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        if transition.terminal:
            next_value = 0.0
            recurrence_mask = 0.0
        elif transition.truncated:
            next_value = float(transition.bootstrap_value)
            recurrence_mask = 0.0
        else:
            if index + 1 >= len(transitions):
                raise SchemaValidationError("Order9 GAE non-boundary fragment tail")
            next_value = float(transitions[index + 1].value)
            recurrence_mask = 1.0
        delta = (
            float(transition.reward)
            + gamma * next_value
            - float(transition.value)
        )
        next_advantage = (
            delta + gamma * gae_lambda * recurrence_mask * next_advantage
        )
        advantages[index] = next_advantage
    returns = [
        advantage + float(transition.value)
        for advantage, transition in zip(advantages, transitions)
    ]
    return Order9GAEResult(tuple(advantages), tuple(returns))


def order9_pi_l_behavior_trace(
    step: Order9LowLevelActorCriticStep,
    *,
    batch_index: int,
    checkpoint_sha256: str,
    recurrent_state_in: torch.Tensor,
    previous_global_action: torch.Tensor,
    privileged_disturbance_body: torch.Tensor | None = None,
    actor_graph_frame_origin_world: Sequence[float] = (0.0, 0.0, 0.0),
    actor_graph_joint_summary_semantics: str = (
        ORDER9_PI_L_GRAPH_JOINT_SUMMARY_ALL
    ),
) -> PolicyBehaviorTrace:
    module_ids = [
        int(value)
        for value in step.graph_encoding.module_ids[batch_index].detach().cpu().tolist()
        if int(value) >= 0
    ]
    joint_rows = step.joint_action[batch_index, : len(module_ids)].detach().cpu().tolist()
    recurrent_in = recurrent_state_in[batch_index].detach().cpu().tolist()
    recurrent_out = step.recurrent_state[batch_index].detach().cpu().tolist()
    privileged = (
        [0.0] * 6
        if privileged_disturbance_body is None
        else privileged_disturbance_body[batch_index].detach().cpu().tolist()
    )
    graph_origin = _finite_vector(
        actor_graph_frame_origin_world,
        width=3,
        label="Order9 pi_L actor graph-frame origin",
    )
    graph_joint_summary = _graph_joint_summary_semantics(
        actor_graph_joint_summary_semantics
    )
    return PolicyBehaviorTrace(
        policy_family="pi_l",
        policy_version=ORDER9_PI_L_POLICY_VERSION,
        action_semantics=ORDER9_PI_L_ACTION_SEMANTICS,
        action_payload={
            "global_action": step.action[batch_index].detach().cpu().tolist(),
            "module_ids": module_ids,
            "joint_action": joint_rows,
            "previous_global_action": previous_global_action[batch_index]
            .detach()
            .cpu()
            .tolist(),
            "privileged_disturbance_body": privileged,
            "actor_graph_frame_origin_world": graph_origin,
            "actor_graph_joint_summary_semantics": graph_joint_summary,
        },
        stochastic=True,
        policy_checkpoint_sha256=checkpoint_sha256,
        old_log_prob=float(step.log_prob[batch_index].detach().cpu().item()),
        old_value=float(step.value[batch_index].detach().cpu().item()),
        recurrent_state_in=[float(value) for value in recurrent_in],
        recurrent_state_out=[float(value) for value in recurrent_out],
    )


def order9_pi_l_behavior_trace_from_inference(
    inference: Order3PolicyInference,
    *,
    checkpoint_sha256: str,
    privileged_disturbance_body: Sequence[float] | None = None,
    actor_graph_frame_origin_world: Sequence[float] = (0.0, 0.0, 0.0),
    actor_graph_joint_summary_semantics: str = (
        ORDER9_PI_L_GRAPH_JOINT_SUMMARY_ALL
    ),
) -> PolicyBehaviorTrace:
    """Serialize the exact action returned by the stateful production wrapper."""

    if not inference.learned_policy_applied:
        raise SchemaValidationError(
            "deterministic pi_L fallback cannot be recorded as a stochastic actor action"
        )
    module_ids = [int(value) for value in inference.module_ids]
    if not module_ids or len(module_ids) != len(set(module_ids)):
        raise SchemaValidationError(
            "Order9 pi_L inference must preserve unique source module ids"
        )
    if len(inference.normalized_joint_action) != len(module_ids):
        raise SchemaValidationError(
            "Order9 pi_L inference joint rows do not match source module ids"
        )
    privileged = [float(value) for value in (privileged_disturbance_body or [0.0] * 6)]
    if len(privileged) != 6 or not all(math.isfinite(value) for value in privileged):
        raise SchemaValidationError(
            "Order9 pi_L privileged critic input must contain six finite values"
        )
    graph_origin = _finite_vector(
        actor_graph_frame_origin_world,
        width=3,
        label="Order9 pi_L actor graph-frame origin",
    )
    graph_joint_summary = _graph_joint_summary_semantics(
        actor_graph_joint_summary_semantics
    )
    return PolicyBehaviorTrace(
        policy_family="pi_l",
        policy_version=ORDER9_PI_L_POLICY_VERSION,
        action_semantics=ORDER9_PI_L_ACTION_SEMANTICS,
        action_payload={
            "global_action": list(inference.normalized_action),
            "module_ids": module_ids,
            "joint_action": [list(row) for row in inference.normalized_joint_action],
            "previous_global_action": list(inference.previous_action),
            "privileged_disturbance_body": privileged,
            "actor_graph_frame_origin_world": graph_origin,
            "actor_graph_joint_summary_semantics": graph_joint_summary,
        },
        stochastic=True,
        policy_checkpoint_sha256=checkpoint_sha256,
        old_log_prob=float(inference.log_prob),
        old_value=float(inference.value),
        recurrent_state_in=list(inference.recurrent_state_in),
        recurrent_state_out=list(inference.recurrent_state_out),
    )


def order9_pi_h_behavior_trace(
    action: Order9PiHAction,
    evaluation: Order9PiHActionEvaluation,
    *,
    batch_index: int,
    candidate_count: int,
    object_count: int,
    checkpoint_sha256: str,
) -> PolicyBehaviorTrace:
    payload = {
        "candidate_count": candidate_count,
        "object_count": object_count,
        "assignment_active": _slice_cpu(
            action.assignment_active[batch_index], candidate_count=candidate_count
        ),
        "schedule_index": _slice_cpu(
            action.schedule_index[batch_index], candidate_count=candidate_count
        ),
        "wrench_raw": _slice_cpu(
            action.wrench_raw[batch_index], candidate_count=candidate_count
        ),
        "anchor_pose_raw": _slice_cpu(
            action.anchor_pose_raw[batch_index], candidate_count=candidate_count
        ),
        "knot_target_raw": action.knot_target_raw[batch_index].detach().cpu().tolist(),
        "priority_raw": action.priority_raw[batch_index].detach().cpu().tolist(),
        "guard_active": action.guard_active[batch_index].detach().cpu().tolist(),
        "guard_threshold_raw": action.guard_threshold_raw[batch_index]
        .detach()
        .cpu()
        .tolist(),
        "interval_raw": action.interval_raw[batch_index].detach().cpu().tolist(),
        "object_target_raw": action.object_target_raw[
            batch_index, :, :object_count
        ]
        .detach()
        .cpu()
        .tolist(),
    }
    return PolicyBehaviorTrace(
        policy_family="pi_h",
        policy_version=ORDER9_FULL_PI_H_VERSION,
        action_semantics=ORDER9_PI_H_ACTION_SEMANTICS,
        action_payload=payload,
        stochastic=True,
        policy_checkpoint_sha256=checkpoint_sha256,
        old_log_prob=float(evaluation.log_prob[batch_index].detach().cpu().item()),
        old_value=float(evaluation.value[batch_index].detach().cpu().item()),
    )


def order9_pi_d_behavior_trace(
    evaluation: Order9PiDActionEvaluation,
    *,
    selected_action: Mapping[str, Any],
    checkpoint_sha256: str,
) -> PolicyBehaviorTrace:
    return PolicyBehaviorTrace(
        policy_family="pi_d",
        policy_version=ORDER9_AUTOREGRESSIVE_PI_D_VERSION,
        action_semantics=ORDER9_PI_D_ACTION_SEMANTICS,
        action_payload={
            "selected_candidate_index": int(evaluation.selected_index.item()),
            "selected_action": dict(selected_action),
        },
        stochastic=True,
        policy_checkpoint_sha256=checkpoint_sha256,
        old_log_prob=float(evaluation.log_prob.squeeze(0).detach().cpu().item()),
        old_value=float(evaluation.value.squeeze(0).detach().cpu().item()),
    )


def update_order9_pi_h_ppo(
    policy: Order9AutoregressiveHighLevelPolicy,
    records: Sequence[InteractionTrajectoryRecord],
    *,
    optimizer: torch.optim.Optimizer,
    config: Order9PPOOptimizationConfig,
    behavior_checkpoint_sha256: str,
    seed: int,
) -> Order9PPOUpdateResult:
    config.validate()
    _require_stochastic_records(
        [record.behavior_trace for record in records],
        family="pi_h",
        checkpoint_sha256=behavior_checkpoint_sha256,
    )
    advantages, returns = _pi_h_advantages(records, config)
    normalized = _normalize_advantages(advantages)
    metric_rows: list[dict[str, float]] = []
    completed_epochs = 0
    optimizer_steps = 0
    early_stop = False
    for epoch in range(config.epochs_per_update):
        indices = list(range(len(records)))
        random.Random(seed + epoch).shuffle(indices)
        for batch_indices in _index_batches(indices, config.minibatch_size):
            batch = [records[index] for index in batch_indices]
            contexts = [_high_level_context(record) for record in batch]
            output = policy.forward_contexts(contexts)
            action = _deserialize_pi_h_action(policy, output, batch)
            evaluation = policy.evaluate_action(output, action)
            old_log_prob = _trace_tensor(batch, "old_log_prob", evaluation.log_prob)
            advantage = torch.tensor(
                [normalized[index] for index in batch_indices],
                device=evaluation.log_prob.device,
                dtype=evaluation.log_prob.dtype,
            )
            return_tensor = torch.tensor(
                [returns[index] for index in batch_indices],
                device=evaluation.value.device,
                dtype=evaluation.value.dtype,
            )
            metrics = _ppo_step(
                policy,
                optimizer,
                new_log_prob=evaluation.log_prob,
                old_log_prob=old_log_prob,
                advantages=advantage,
                new_values=evaluation.value,
                returns=return_tensor,
                entropy=evaluation.entropy,
                config=config,
                family="pi_h",
            )
            metric_rows.append(metrics)
            optimizer_steps += 1
            if metrics["approximate_kl"] > config.target_kl:
                early_stop = True
                break
        completed_epochs += 1
        if early_stop:
            break
    return _ppo_result(
        "pi_h",
        sample_count=len(records),
        requested_epochs=config.epochs_per_update,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        rows=metric_rows,
        early_stop=early_stop,
        behavior_checkpoint_sha256=behavior_checkpoint_sha256,
    )


def update_order9_pi_d_ppo(
    policy: Order9AutoregressiveDesignPolicy,
    records: Sequence[SequentialDesignTrajectoryRecord],
    *,
    physical_model: PhysicalModel,
    optimizer: torch.optim.Optimizer,
    config: Order9PPOOptimizationConfig,
    behavior_checkpoint_sha256: str,
    seed: int,
) -> Order9PPOUpdateResult:
    config.validate()
    traces = [step.behavior_trace for record in records for step in record.steps]
    _require_stochastic_records(
        traces, family="pi_d", checkpoint_sha256=behavior_checkpoint_sha256
    )
    prepared = {
        record.record_id: reconstruct_order9_pi_d_teacher_trace(record, physical_model)
        for record in records
    }
    advantage_by_step, return_by_step = _pi_d_advantages(records, config)
    normalized_values = _normalize_advantages(list(advantage_by_step.values()))
    normalized = {
        key: value for key, value in zip(advantage_by_step, normalized_values)
    }
    metric_rows: list[dict[str, float]] = []
    completed_epochs = 0
    optimizer_steps = 0
    early_stop = False
    for epoch in range(config.epochs_per_update):
        shuffled = list(records)
        random.Random(seed + epoch).shuffle(shuffled)
        for batch in _design_minibatches(shuffled, config.minibatch_size):
            new_log_probs = []
            old_log_probs = []
            values = []
            advantages = []
            returns = []
            entropies = []
            for record in batch:
                context, trace = prepared[record.record_id]
                history = policy.initial_history()
                for index, teacher_step in enumerate(trace):
                    output = policy.forward_step(
                        context,
                        teacher_step.state,
                        teacher_step.candidate_step.candidates,
                        history=history,
                    )
                    selected_index = record.steps[index].selected_candidate_index
                    evaluation = policy.evaluate_selected_step(output, selected_index)
                    behavior = record.steps[index].behavior_trace
                    if behavior is None:
                        raise AssertionError("validated pi_D behavior disappeared")
                    new_log_probs.append(evaluation.log_prob.squeeze(0))
                    old_log_probs.append(float(behavior.old_log_prob))
                    values.append(evaluation.value.squeeze(0))
                    advantages.append(normalized[(record.record_id, index)])
                    returns.append(return_by_step[(record.record_id, index)])
                    entropies.append(evaluation.entropy.squeeze(0))
                    history = policy.advance_history(output, selected_index)
            reference = torch.stack(new_log_probs)
            old = torch.tensor(old_log_probs, device=reference.device, dtype=reference.dtype)
            advantage_tensor = torch.tensor(
                advantages, device=reference.device, dtype=reference.dtype
            )
            value_tensor = torch.stack(values)
            return_tensor = torch.tensor(
                returns, device=value_tensor.device, dtype=value_tensor.dtype
            )
            entropy_tensor = torch.stack(entropies)
            total = compute_order9_pi_d_masked_ppo_loss(
                new_log_prob=reference,
                old_log_prob=old,
                advantages=advantage_tensor,
                new_values=value_tensor,
                returns=return_tensor,
                entropy=entropy_tensor,
                clip_ratio=config.clip_ratio,
                value_loss_weight=config.value_loss_weight,
                entropy_bonus_weight=config.entropy_bonus_weight,
            )
            metrics = _optimizer_metrics_step(
                policy,
                optimizer,
                total=total,
                new_log_prob=reference,
                old_log_prob=old,
                advantages=advantage_tensor,
                new_values=value_tensor,
                returns=return_tensor,
                entropy=entropy_tensor,
                config=config,
            )
            metric_rows.append(metrics)
            optimizer_steps += 1
            if metrics["approximate_kl"] > config.target_kl:
                early_stop = True
                break
        completed_epochs += 1
        if early_stop:
            break
    return _ppo_result(
        "pi_d",
        sample_count=sum(len(record.steps) for record in records),
        requested_epochs=config.epochs_per_update,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        rows=metric_rows,
        early_stop=early_stop,
        behavior_checkpoint_sha256=behavior_checkpoint_sha256,
    )


def update_order9_pi_l_ppo(
    policy: Order9PhaseConditionedActorCritic,
    records: Sequence[LowLevelControlRecord],
    *,
    physical_model: PhysicalModel,
    optimizer: torch.optim.Optimizer,
    config: Order9PPOOptimizationConfig,
    behavior_checkpoint_sha256: str,
    seed: int,
    sequence_length: int = 16,
    progress_callback: Callable[[int, Mapping[str, float]], None] | None = None,
) -> Order9PPOUpdateResult:
    """Run recurrent PPO with current-policy recurrence over stored sequences."""

    config.validate()
    if sequence_length < 1:
        raise ValueError("Order9 recurrent PPO sequence_length must be positive")
    _require_stochastic_records(
        [record.behavior_trace for record in records],
        family="pi_l",
        checkpoint_sha256=behavior_checkpoint_sha256,
    )
    replay_cache = _build_pi_l_replay_cache(
        records,
        policy=policy,
        physical_model=physical_model,
    )
    sequences = _low_level_sequences(records, sequence_length=sequence_length)
    replay_metrics = _validate_pi_l_exact_behavior_replay(
        policy,
        sequences,
        replay_cache=replay_cache,
    )
    advantage_by_id, return_by_id = _pi_l_advantages(records, config)
    normalized_values = _normalize_advantages(list(advantage_by_id.values()))
    normalized = {key: value for key, value in zip(advantage_by_id, normalized_values)}
    sequences_per_minibatch = max(1, config.minibatch_size // sequence_length)
    metric_rows: list[dict[str, float]] = []
    completed_epochs = 0
    optimizer_steps = 0
    early_stop = False
    for epoch in range(config.epochs_per_update):
        shuffled = list(sequences)
        random.Random(seed + epoch).shuffle(shuffled)
        for sequence_batch in _plain_batches(shuffled, sequences_per_minibatch):
            metrics = _pi_l_sequence_ppo_step(
                policy,
                sequence_batch,
                optimizer=optimizer,
                config=config,
                advantages=normalized,
                returns=return_by_id,
                replay_cache=replay_cache,
            )
            metric_rows.append(metrics)
            optimizer_steps += 1
            if progress_callback is not None:
                progress_callback(
                    optimizer_steps,
                    {
                        **metrics,
                        "epoch_index": float(epoch),
                        "optimizer_step": float(optimizer_steps),
                    },
                )
            if metrics["approximate_kl"] > config.target_kl:
                early_stop = True
                break
        completed_epochs += 1
        if early_stop:
            break
    return _ppo_result(
        "pi_l",
        sample_count=len(records),
        requested_epochs=config.epochs_per_update,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        rows=metric_rows,
        early_stop=early_stop,
        behavior_checkpoint_sha256=behavior_checkpoint_sha256,
        metadata={
            "sequence_length": sequence_length,
            "recurrent_replay": True,
            "timestep_batched_active_sequences": True,
            "record_invariant_replay_cache": True,
            "exact_behavior_replay_validated": True,
            **replay_metrics,
        },
    )


def _pi_l_sequence_ppo_step(
    policy: Order9PhaseConditionedActorCritic,
    sequences: Sequence[Sequence[LowLevelControlRecord]],
    *,
    optimizer: torch.optim.Optimizer,
    config: Order9PPOOptimizationConfig,
    advantages: Mapping[str, float],
    returns: Mapping[str, float],
    replay_cache: _PiLReplayCache,
) -> dict[str, float]:
    # Batch all active sequences at the same recurrent timestep.  This is the
    # same independent recurrence as sequence-at-a-time replay, but lets the
    # graph encoder and actor/critic use the GPU instead of issuing thousands
    # of batch-size-one forward calls.
    if not sequences or any(not sequence for sequence in sequences):
        raise SchemaValidationError("Order9 pi_L PPO sequence batch is empty")
    parameter = next(policy.parameters())
    hidden_by_sequence = [
        torch.tensor(
            _pi_l_trace(sequence[0]).recurrent_state_in,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for sequence in sequences
    ]
    previous_by_sequence = [
        torch.tensor(
            _pi_l_trace(sequence[0]).action_payload["previous_global_action"],
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for sequence in sequences
    ]
    new_log_probs = []
    old_log_probs = []
    new_values = []
    advantage_values = []
    return_values = []
    entropies = []
    for timestep in range(max(len(sequence) for sequence in sequences)):
        active_indices = [
            index
            for index, sequence in enumerate(sequences)
            if timestep < len(sequence)
        ]
        records = [sequences[index][timestep] for index in active_indices]
        traces = [_pi_l_trace(record) for record in records]
        actor_features, phase_features, privileged, actor_observations = (
            _pi_l_cached_batch(records, replay_cache)
        )
        global_action, joint_action = _pi_l_actions(records, traces, policy)
        step = policy.step(
            [record.runtime_observation.morphology_graph for record in records],
            actor_observations,
            actor_features,
            torch.stack([previous_by_sequence[index] for index in active_indices]),
            torch.stack([hidden_by_sequence[index] for index in active_indices]),
            phase_features=phase_features,
            privileged_disturbance_body=privileged,
            action=global_action,
            joint_action=joint_action,
        )
        new_log_probs.extend(step.log_prob.unbind(0))
        old_log_probs.extend(float(trace.old_log_prob) for trace in traces)
        new_values.extend(step.value.unbind(0))
        entropies.extend(step.entropy.unbind(0))
        advantage_values.extend(advantages[record.record_id] for record in records)
        return_values.extend(returns[record.record_id] for record in records)
        for row, sequence_index in enumerate(active_indices):
            hidden_by_sequence[sequence_index] = step.recurrent_state[row]
            previous_by_sequence[sequence_index] = step.action[row].detach()
    new_log_prob = torch.stack(new_log_probs)
    old_log_prob = torch.tensor(
        old_log_probs, device=new_log_prob.device, dtype=new_log_prob.dtype
    )
    advantage_tensor = torch.tensor(
        advantage_values, device=new_log_prob.device, dtype=new_log_prob.dtype
    )
    values = torch.stack(new_values)
    return_tensor = torch.tensor(
        return_values, device=values.device, dtype=values.dtype
    )
    entropy = torch.stack(entropies)
    total = compute_order9_pi_l_ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantage_tensor,
        new_values=values,
        returns=return_tensor,
        entropy=entropy,
        clip_ratio=config.clip_ratio,
        value_loss_weight=config.value_loss_weight,
        entropy_bonus_weight=config.entropy_bonus_weight,
    )
    return _optimizer_metrics_step(
        policy,
        optimizer,
        total=total,
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantage_tensor,
        new_values=values,
        returns=return_tensor,
        entropy=entropy,
        config=config,
    )


def _build_pi_l_replay_cache(
    records: Sequence[LowLevelControlRecord],
    *,
    policy: Order9PhaseConditionedActorCritic,
    physical_model: PhysicalModel,
) -> _PiLReplayCache:
    if not records:
        raise SchemaValidationError("Order9 pi_L replay cache requires records")
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    builder = RigidBodyControlModelBuilder()
    index_by_record_id: dict[str, int] = {}
    actor_observations = []
    actor_rows = []
    phase_rows = []
    privileged_rows = []
    for index, record in enumerate(records):
        if record.record_id in index_by_record_id:
            raise SchemaValidationError(
                f"Order9 pi_L replay record id is duplicated: {record.record_id}"
            )
        index_by_record_id[record.record_id] = index
        trace = _pi_l_trace(record)
        context = _pi_l_context(record, physical_model)
        reference = order9_pi_l_reference_command(context)
        if reference.desired_body_pose is None:
            raise SchemaValidationError("Order9 pi_L PPO reference pose is missing")
        control_model = builder.build(
            context.morphology_graph, physical_model, context.runtime_observation
        )
        actor_rows.append(
            order3_actor_feature_vector(
                context.runtime_observation,
                control_model,
                target_pose_world=reference.desired_body_pose,
                target_twist=list(reference.desired_body_twist or [0.0] * 6),
                max_modules=policy.config.max_modules,
            )
        )
        phase_rows.append(order9_phase_actor_feature_vector(context, policy.config))
        privileged = trace.action_payload.get("privileged_disturbance_body")
        if not isinstance(privileged, list) or len(privileged) != 6:
            raise SchemaValidationError("Order9 pi_L behavior lacks privileged critic input")
        privileged_rows.append(privileged)
        actor_observations.append(
            _pi_l_actor_graph_observation(
                context.runtime_observation,
                trace,
                physical_model,
            )
        )
    actor = torch.tensor(actor_rows, device=device, dtype=dtype)
    if actor.shape[1] != len(ORDER3_ACTOR_FEATURE_NAMES):
        raise RuntimeError("Order9 pi_L actor feature layout drifted")
    return _PiLReplayCache(
        index_by_record_id=index_by_record_id,
        actor_graph_observations=tuple(actor_observations),
        actor_features=actor,
        phase_features=torch.tensor(phase_rows, device=device, dtype=dtype),
        privileged_disturbance_body=torch.tensor(
            privileged_rows, device=device, dtype=dtype
        ),
    )


def _pi_l_cached_batch(
    records: Sequence[LowLevelControlRecord],
    cache: _PiLReplayCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[RuntimeObservation]]:
    if not records:
        raise SchemaValidationError("Order9 pi_L cached replay batch is empty")
    try:
        indices = [cache.index_by_record_id[record.record_id] for record in records]
    except KeyError as exc:
        raise SchemaValidationError(
            f"Order9 pi_L replay cache misses record {exc.args[0]}"
        ) from exc
    index = torch.tensor(
        indices,
        device=cache.actor_features.device,
        dtype=torch.long,
    )
    return (
        cache.actor_features.index_select(0, index),
        cache.phase_features.index_select(0, index),
        cache.privileged_disturbance_body.index_select(0, index),
        [cache.actor_graph_observations[value] for value in indices],
    )


def _pi_l_actions(
    records: Sequence[LowLevelControlRecord],
    traces: Sequence[PolicyBehaviorTrace],
    policy: Order9PhaseConditionedActorCritic,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    maximum_modules = max(len(record.runtime_observation.morphology_graph.modules) for record in records)
    joint_width = 3 * policy.config.max_local_joint_slots
    globals_: list[list[float]] = []
    joints = torch.zeros(
        (len(records), maximum_modules, joint_width), device=device, dtype=dtype
    )
    for batch_index, (record, trace) in enumerate(zip(records, traces)):
        payload = trace.action_payload
        global_action = payload.get("global_action")
        module_ids = payload.get("module_ids")
        joint_rows = payload.get("joint_action")
        if (
            not isinstance(global_action, list)
            or len(global_action) != ORDER9_GLOBAL_ACTION_SIZE
            or not isinstance(module_ids, list)
            or not isinstance(joint_rows, list)
            or len(module_ids) != len(joint_rows)
        ):
            raise SchemaValidationError("Order9 pi_L action payload shape is invalid")
        expected_ids = sorted(
            module.module_id
            for module in record.runtime_observation.morphology_graph.modules
        )
        if [int(value) for value in module_ids] != expected_ids:
            raise SchemaValidationError("Order9 pi_L action module order drifted")
        for row_index, row in enumerate(joint_rows):
            if not isinstance(row, list) or len(row) != joint_width:
                raise SchemaValidationError("Order9 pi_L joint action width drifted")
            joints[batch_index, row_index] = torch.tensor(
                row, device=device, dtype=dtype
            )
        globals_.append(global_action)
    return torch.tensor(globals_, device=device, dtype=dtype), joints


def _deserialize_pi_h_action(
    policy: Order9AutoregressiveHighLevelPolicy,
    output,
    records: Sequence[InteractionTrajectoryRecord],
) -> Order9PiHAction:
    device = output.assignment_logits.device
    dtype = output.assignment_logits.dtype
    batch = len(records)
    knots = policy.config.num_knots
    candidates = output.assignment_logits.shape[2]
    objects = output.object_target_raw_mean.shape[2]
    assignment = torch.zeros((batch, knots, candidates), dtype=torch.bool, device=device)
    schedule = torch.zeros((batch, knots, candidates), dtype=torch.long, device=device)
    wrench = torch.zeros_like(output.wrench_raw_mean)
    anchor = torch.zeros_like(output.anchor_pose_raw_mean)
    knot_target = torch.zeros_like(output.knot_target_raw_mean)
    priority = torch.zeros_like(output.priority_raw_mean)
    guard = torch.zeros_like(output.guard_logits, dtype=torch.bool)
    guard_threshold = torch.zeros_like(output.guard_threshold_raw_mean)
    interval = torch.zeros_like(output.interval_raw_mean)
    object_target = torch.zeros_like(output.object_target_raw_mean)
    for index, record in enumerate(records):
        trace = record.behavior_trace
        if trace is None or trace.action_semantics != ORDER9_PI_H_ACTION_SEMANTICS:
            raise SchemaValidationError("Order9 pi_H action semantics mismatch")
        payload = trace.action_payload
        candidate_count = _payload_int(payload, "candidate_count")
        object_count = _payload_int(payload, "object_count")
        if candidate_count != len(record.contact_candidate_set.candidates):
            raise SchemaValidationError("Order9 pi_H candidate width drifted")
        if object_count != len(record.runtime_observation.object_states):
            raise SchemaValidationError("Order9 pi_H object width drifted")
        if candidate_count:
            assignment[index, :, :candidate_count] = _payload_tensor(
                payload, "assignment_active", device=device, dtype=torch.bool,
                shape=(knots, candidate_count),
            )
            schedule[index, :, :candidate_count] = _payload_tensor(
                payload, "schedule_index", device=device, dtype=torch.long,
                shape=(knots, candidate_count),
            )
            wrench[index, :, :candidate_count] = _payload_tensor(
                payload, "wrench_raw", device=device, dtype=dtype,
                shape=(knots, candidate_count, 18),
            )
            anchor[index, :, :candidate_count] = _payload_tensor(
                payload, "anchor_pose_raw", device=device, dtype=dtype,
                shape=(knots, candidate_count, 7),
            )
        else:
            _require_empty_second_axis_payload(
                payload,
                ("assignment_active", "schedule_index", "wrench_raw", "anchor_pose_raw"),
                outer_size=knots,
            )
        knot_target[index] = _payload_tensor(
            payload, "knot_target_raw", device=device, dtype=dtype,
            shape=(knots, 10),
        )
        priority[index] = _payload_tensor(
            payload, "priority_raw", device=device, dtype=dtype,
            shape=tuple(priority[index].shape),
        )
        guard[index] = _payload_tensor(
            payload, "guard_active", device=device, dtype=torch.bool,
            shape=tuple(guard[index].shape),
        )
        guard_threshold[index] = _payload_tensor(
            payload, "guard_threshold_raw", device=device, dtype=dtype,
            shape=tuple(guard_threshold[index].shape),
        )
        interval[index] = _payload_tensor(
            payload, "interval_raw", device=device, dtype=dtype,
            shape=(knots,),
        )
        if object_count:
            object_target[index, :, :object_count] = _payload_tensor(
                payload, "object_target_raw", device=device, dtype=dtype,
                shape=(knots, object_count, 13),
            )
        else:
            _require_empty_second_axis_payload(
                payload, ("object_target_raw",), outer_size=knots
            )
    return Order9PiHAction(
        assignment_active=assignment,
        schedule_index=schedule,
        wrench_raw=wrench,
        anchor_pose_raw=anchor,
        knot_target_raw=knot_target,
        priority_raw=priority,
        guard_active=guard,
        guard_threshold_raw=guard_threshold,
        interval_raw=interval,
        object_target_raw=object_target,
    )


def _pi_h_advantages(
    records: Sequence[InteractionTrajectoryRecord],
    config: Order9PPOOptimizationConfig,
) -> tuple[list[float], list[float]]:
    by_episode: dict[str, list[tuple[int, InteractionTrajectoryRecord]]] = {}
    for index, record in enumerate(records):
        by_episode.setdefault(record.episode_id, []).append((index, record))
    advantages = [0.0] * len(records)
    returns = [0.0] * len(records)
    for episode in by_episode.values():
        ordered = sorted(episode, key=lambda item: item[1].decision_index)
        transitions = []
        for _, record in ordered:
            trace = record.behavior_trace
            if trace is None or record.decision_reward is None:
                raise SchemaValidationError("Order9 pi_H PPO transition is incomplete")
            if record.transition_kind == HighLevelTransitionKind.CHECKER_REJECTION:
                expected_penalty = -float(config.hard_checker_rejection_penalty)
                if not math.isclose(
                    float(record.decision_reward),
                    expected_penalty,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise SchemaValidationError(
                        "Order9 pi_H checker-rejection reward does not match the "
                        "configured credit-assignment contract"
                    )
            transitions.append(
                Order9GAETransition(
                    reward=record.decision_reward,
                    value=float(trace.old_value),
                    terminal=record.terminal,
                    truncated=record.truncated,
                    bootstrap_value=record.bootstrap_value,
                )
            )
        computed = compute_order9_gae(
            transitions, gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        for (source_index, _), advantage, return_ in zip(
            ordered, computed.advantages, computed.returns
        ):
            advantages[source_index] = advantage
            returns[source_index] = return_
    return advantages, returns


def _pi_l_advantages(
    records: Sequence[LowLevelControlRecord],
    config: Order9PPOOptimizationConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    by_episode: dict[str, list[LowLevelControlRecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
    advantages: dict[str, float] = {}
    returns: dict[str, float] = {}
    for episode in by_episode.values():
        ordered = sorted(episode, key=lambda item: item.step_index)
        transitions = []
        for record in ordered:
            trace = _pi_l_trace(record)
            if record.reward is None:
                raise SchemaValidationError("Order9 pi_L PPO reward is missing")
            transitions.append(
                Order9GAETransition(
                    reward=record.reward,
                    value=float(trace.old_value),
                    terminal=record.terminal,
                    truncated=record.truncated,
                    bootstrap_value=record.bootstrap_value,
                )
            )
        computed = compute_order9_gae(
            transitions, gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        for record, advantage, return_ in zip(
            ordered, computed.advantages, computed.returns
        ):
            advantages[record.record_id] = advantage
            returns[record.record_id] = return_
    return advantages, returns


def _pi_d_advantages(
    records: Sequence[SequentialDesignTrajectoryRecord],
    config: Order9PPOOptimizationConfig,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    advantages = {}
    returns = {}
    for record in records:
        transitions = []
        for step in record.steps:
            if step.behavior_trace is None:
                raise SchemaValidationError("Order9 pi_D PPO behavior is missing")
            transitions.append(
                Order9GAETransition(
                    reward=step.reward,
                    value=float(step.behavior_trace.old_value),
                    terminal=step.terminal,
                    truncated=step.truncated,
                    bootstrap_value=step.bootstrap_value,
                )
            )
        computed = compute_order9_gae(
            transitions, gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        for step, advantage, return_ in zip(
            record.steps, computed.advantages, computed.returns
        ):
            key = (record.record_id, step.step_index)
            advantages[key] = advantage
            returns[key] = return_
    return advantages, returns


def _ppo_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    config: Order9PPOOptimizationConfig,
    family: str,
) -> dict[str, float]:
    actor = clipped_ppo_surrogate_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        clip_ratio=config.clip_ratio,
    )
    value = torch.nn.functional.mse_loss(new_values, returns)
    total = actor + config.value_loss_weight * value - config.entropy_bonus_weight * entropy.mean()
    return _optimizer_metrics_step(
        model,
        optimizer,
        total=total,
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        new_values=new_values,
        returns=returns,
        entropy=entropy,
        config=config,
    )


def _optimizer_metrics_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    total: torch.Tensor,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    config: Order9PPOOptimizationConfig,
) -> dict[str, float]:
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("Order9 PPO loss became non-finite")
    actor = clipped_ppo_surrogate_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        clip_ratio=config.clip_ratio,
    )
    value = torch.nn.functional.mse_loss(new_values, returns)
    with torch.no_grad():
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        approximate_kl = float(((ratio - 1.0) - log_ratio).mean().cpu().item())
        clipped_fraction = float(
            ((ratio - 1.0).abs() > config.clip_ratio).float().mean().cpu().item()
        )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    if not bool(torch.isfinite(torch.as_tensor(norm)).item()):
        raise FloatingPointError("Order9 PPO gradient norm became non-finite")
    optimizer.step()
    return {
        "actor_loss": float(actor.detach().cpu().item()),
        "value_loss": float(value.detach().cpu().item()),
        "entropy": float(entropy.mean().detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
        "approximate_kl": max(0.0, approximate_kl),
        "clipped_fraction": clipped_fraction,
    }


def _ppo_result(
    family: str,
    *,
    sample_count: int,
    requested_epochs: int,
    completed_epochs: int,
    optimizer_steps: int,
    rows: Sequence[Mapping[str, float]],
    early_stop: bool,
    behavior_checkpoint_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> Order9PPOUpdateResult:
    if not rows:
        raise SchemaValidationError("Order9 PPO performed no optimizer steps")
    means = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }
    result = Order9PPOUpdateResult(
        replay_version=ORDER9_PPO_REPLAY_VERSION,
        policy_family=family,
        sample_count=sample_count,
        requested_epoch_count=requested_epochs,
        completed_epoch_count=completed_epochs,
        optimizer_step_count=optimizer_steps,
        actor_loss=means["actor_loss"],
        value_loss=means["value_loss"],
        entropy=means["entropy"],
        total_loss=means["total_loss"],
        approximate_kl=means["approximate_kl"],
        clipped_fraction=means["clipped_fraction"],
        early_stopped_for_kl=early_stop,
        metadata={
            "behavior_checkpoint_sha256": behavior_checkpoint_sha256,
            **dict(metadata or {}),
        },
    )
    result.validate()
    return result


def _require_stochastic_records(
    traces: Sequence[PolicyBehaviorTrace | None],
    *,
    family: str,
    checkpoint_sha256: str,
) -> None:
    if not traces:
        raise SchemaValidationError("Order9 PPO rollout batch is empty")
    for trace in traces:
        if (
            trace is None
            or trace.policy_family != family
            or not trace.stochastic
            or trace.policy_checkpoint_sha256 != checkpoint_sha256
            or trace.old_log_prob is None
            or trace.old_value is None
        ):
            raise SchemaValidationError(
                "Order9 PPO rollout does not match the exact behavior checkpoint"
            )


def _normalize_advantages(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("Order9 PPO advantages are empty")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance + 1.0e-8)
    return [(value - mean) / scale for value in values]


def _trace_tensor(
    records: Sequence[InteractionTrajectoryRecord],
    field: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    values = []
    for record in records:
        trace = record.behavior_trace
        if trace is None:
            raise AssertionError("validated behavior trace disappeared")
        values.append(float(getattr(trace, field)))
    return torch.tensor(values, device=reference.device, dtype=reference.dtype)


def _high_level_context(record: InteractionTrajectoryRecord) -> HighLevelPolicyContext:
    return HighLevelPolicyContext(
        irg=record.irg,
        interaction_envelope=record.interaction_envelope,
        morphology_graph=record.morphology_graph,
        contact_candidate_set=record.contact_candidate_set,
        runtime_observation=record.runtime_observation,
    )


def _pi_l_context(
    record: LowLevelControlRecord, physical_model: PhysicalModel
) -> LowLevelPolicyContext:
    from amsrr.training.order9_offline_training import _low_level_context

    return _low_level_context(record, physical_model)


def _pi_l_trace(record: LowLevelControlRecord) -> PolicyBehaviorTrace:
    trace = record.behavior_trace
    if trace is None or trace.action_semantics != ORDER9_PI_L_ACTION_SEMANTICS:
        raise SchemaValidationError("Order9 pi_L action semantics mismatch")
    if (
        len(trace.recurrent_state_in) == 0
        or len(trace.recurrent_state_out) == 0
    ):
        raise SchemaValidationError("Order9 recurrent pi_L trace lacks hidden states")
    _finite_vector(
        trace.action_payload.get("actor_graph_frame_origin_world"),
        width=3,
        label="Order9 pi_L actor graph-frame origin",
    )
    _graph_joint_summary_semantics(
        trace.action_payload.get("actor_graph_joint_summary_semantics")
    )
    return trace


def _pi_l_actor_graph_observation(
    observation: RuntimeObservation,
    trace: PolicyBehaviorTrace,
    physical_model: PhysicalModel,
) -> RuntimeObservation:
    """Recreate the graph-only actor frame used by the tensor collector."""

    origin = _finite_vector(
        trace.action_payload.get("actor_graph_frame_origin_world"),
        width=3,
        label="Order9 pi_L actor graph-frame origin",
    )
    joint_summary = _graph_joint_summary_semantics(
        trace.action_payload.get("actor_graph_joint_summary_semantics")
    )
    active_joint_ids = (
        None
        if joint_summary == ORDER9_PI_L_GRAPH_JOINT_SUMMARY_ALL
        else {
            joint.joint_id
            for joint in physical_model.joints
            if joint.joint_type != "fixed"
        }
    )
    replay = replace(
        observation,
        module_states=[
            replace(
                state,
                pose_world=tuple(
                    [
                        float(state.pose_world[index]) - origin[index]
                        for index in range(3)
                    ]
                    + [float(value) for value in state.pose_world[3:]]
                ),
                joint_positions={
                    joint_id: value
                    for joint_id, value in state.joint_positions.items()
                    if active_joint_ids is None or joint_id in active_joint_ids
                },
                joint_velocities={
                    joint_id: value
                    for joint_id, value in state.joint_velocities.items()
                    if active_joint_ids is None or joint_id in active_joint_ids
                },
            )
            for state in observation.module_states
        ],
    )
    replay.validate()
    return replay


def _validate_pi_l_exact_behavior_replay(
    policy: Order9PhaseConditionedActorCritic,
    sequences: Sequence[Sequence[LowLevelControlRecord]],
    *,
    replay_cache: _PiLReplayCache,
    absolute_tolerance: float = 2.0e-5,
) -> dict[str, float | int]:
    """Fail before optimization unless every stored actor result replays."""

    if (
        not sequences
        or any(not sequence for sequence in sequences)
        or absolute_tolerance <= 0.0
    ):
        raise ValueError("Order9 pi_L exact replay configuration is invalid")
    parameter = next(policy.parameters())
    maximum_log_prob_error = 0.0
    maximum_value_error = 0.0
    maximum_recurrent_error = 0.0
    validated_count = 0
    with torch.no_grad():
        hidden_by_sequence = [
            torch.tensor(
                _pi_l_trace(sequence[0]).recurrent_state_in,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            for sequence in sequences
        ]
        previous_by_sequence = [
            torch.tensor(
                _pi_l_trace(sequence[0]).action_payload["previous_global_action"],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            for sequence in sequences
        ]
        for timestep in range(max(len(sequence) for sequence in sequences)):
            active_indices = [
                index
                for index, sequence in enumerate(sequences)
                if timestep < len(sequence)
            ]
            records = [sequences[index][timestep] for index in active_indices]
            traces = [_pi_l_trace(record) for record in records]
            hidden = torch.stack(
                [hidden_by_sequence[index] for index in active_indices]
            )
            previous = torch.stack(
                [previous_by_sequence[index] for index in active_indices]
            )
            expected_hidden = torch.tensor(
                [trace.recurrent_state_in for trace in traces],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            expected_previous = torch.tensor(
                [trace.action_payload["previous_global_action"] for trace in traces],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            _require_exact_replay_batch(
                hidden,
                expected_hidden,
                tolerance=absolute_tolerance,
                records=records,
                field="recurrent_state_in",
            )
            _require_exact_replay_batch(
                previous,
                expected_previous,
                tolerance=absolute_tolerance,
                records=records,
                field="previous_global_action",
            )
            actor_features, phase_features, privileged, actor_observations = (
                _pi_l_cached_batch(records, replay_cache)
            )
            global_action, joint_action = _pi_l_actions(records, traces, policy)
            step = policy.step(
                [record.runtime_observation.morphology_graph for record in records],
                actor_observations,
                actor_features,
                previous,
                hidden,
                phase_features=phase_features,
                privileged_disturbance_body=privileged,
                action=global_action,
                joint_action=joint_action,
            )
            expected_log_prob = torch.tensor(
                [float(trace.old_log_prob) for trace in traces],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            expected_value = torch.tensor(
                [float(trace.old_value) for trace in traces],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            expected_recurrent = torch.tensor(
                [trace.recurrent_state_out for trace in traces],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            log_prob_error = _require_exact_replay_batch(
                step.log_prob,
                expected_log_prob,
                tolerance=absolute_tolerance,
                records=records,
                field="old_log_prob",
            )
            value_error = _require_exact_replay_batch(
                step.value,
                expected_value,
                tolerance=absolute_tolerance,
                records=records,
                field="old_value",
            )
            recurrent_error = _require_exact_replay_batch(
                step.recurrent_state,
                expected_recurrent,
                tolerance=absolute_tolerance,
                records=records,
                field="recurrent_state_out",
            )
            maximum_log_prob_error = max(maximum_log_prob_error, log_prob_error)
            maximum_value_error = max(maximum_value_error, value_error)
            maximum_recurrent_error = max(
                maximum_recurrent_error, recurrent_error
            )
            validated_count += len(records)
            for row, sequence_index in enumerate(active_indices):
                hidden_by_sequence[sequence_index] = step.recurrent_state[row]
                previous_by_sequence[sequence_index] = step.action[row]
    return {
        "exact_replay_record_count": validated_count,
        "maximum_log_prob_replay_error": maximum_log_prob_error,
        "maximum_value_replay_error": maximum_value_error,
        "maximum_recurrent_replay_error": maximum_recurrent_error,
        "exact_replay_absolute_tolerance": absolute_tolerance,
        "exact_replay_timestep_batched_active_sequences": True,
        "exact_replay_record_invariant_cache": True,
    }


def _require_exact_replay_batch(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float,
    records: Sequence[LowLevelControlRecord],
    field: str,
) -> float:
    if actual.shape != expected.shape or actual.shape[0] != len(records):
        raise SchemaValidationError(
            f"Order9 pi_L exact replay {field} batched shape differs"
        )
    difference = (actual - expected).abs()
    per_record = difference.reshape(len(records), -1).amax(dim=1)
    maximum, row = per_record.max(dim=0)
    error = float(maximum.detach().cpu().item())
    if not math.isfinite(error) or error > tolerance:
        record_id = records[int(row.detach().cpu().item())].record_id
        raise SchemaValidationError(
            "Order9 pi_L exact behavior replay mismatch at "
            f"{record_id}:{field} (max_abs_error={error:.9g}, "
            f"tolerance={tolerance:.9g})"
        )
    return error


def _finite_vector(
    values: object,
    *,
    width: int,
    label: str,
) -> list[float]:
    if (
        not isinstance(values, (list, tuple))
        or len(values) != width
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise SchemaValidationError(
            f"{label} must contain {width} finite numeric values"
        )
    return [float(value) for value in values]


def _graph_joint_summary_semantics(value: object) -> str:
    allowed = {
        ORDER9_PI_L_GRAPH_JOINT_SUMMARY_ALL,
        ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED,
    }
    if not isinstance(value, str) or value not in allowed:
        raise SchemaValidationError(
            "Order9 pi_L actor graph joint-summary semantics is invalid"
        )
    return value


def _low_level_sequences(
    records: Sequence[LowLevelControlRecord], *, sequence_length: int
) -> list[list[LowLevelControlRecord]]:
    by_episode: dict[str, list[LowLevelControlRecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
    sequences = []
    for episode_id in sorted(by_episode):
        ordered = sorted(by_episode[episode_id], key=lambda item: item.step_index)
        sequences.extend(
            ordered[index : index + sequence_length]
            for index in range(0, len(ordered), sequence_length)
        )
    return sequences


def _design_minibatches(
    records: Sequence[SequentialDesignTrajectoryRecord], max_steps: int
) -> Iterable[list[SequentialDesignTrajectoryRecord]]:
    current = []
    count = 0
    for record in records:
        if current and count + len(record.steps) > max_steps:
            yield current
            current = []
            count = 0
        current.append(record)
        count += len(record.steps)
    if current:
        yield current


def _index_batches(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _plain_batches(values: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _slice_cpu(tensor: torch.Tensor, *, candidate_count: int) -> list[Any]:
    return tensor[:, :candidate_count].detach().cpu().tolist()


def _payload_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaValidationError(f"Order9 action payload {key!r} must be non-negative int")
    return value


def _payload_tensor(
    payload: Mapping[str, Any],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    try:
        value = torch.tensor(payload[key], device=device, dtype=dtype)
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError(f"Order9 action payload {key!r} is invalid") from exc
    if tuple(value.shape) != shape:
        raise SchemaValidationError(
            f"Order9 action payload {key!r} must have shape {shape}, got {tuple(value.shape)}"
        )
    return value


def _require_empty_second_axis_payload(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    outer_size: int,
) -> None:
    for key in keys:
        value = payload.get(key)
        if (
            not isinstance(value, list)
            or len(value) != outer_size
            or any(item != [] for item in value)
        ):
            raise SchemaValidationError(
                f"Order9 action payload {key!r} must preserve an empty second axis"
            )
