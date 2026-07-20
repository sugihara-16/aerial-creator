from __future__ import annotations

"""BC/PPO loss boundary for sequential masked Order 9 pi_D."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.order9_design_grammar import Order9DesignTeacherStep
from amsrr.policies.order9_design_policy import Order9AutoregressiveDesignPolicy
from amsrr.training.order9_pi_h_learning import clipped_ppo_surrogate_loss


ORDER9_PI_D_LEARNING_VERSION = "order9_sequential_pi_d_bc_masked_ppo_v1"


@dataclass
class Order9PiDBehaviorCloningLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    step_count: int


def compute_order9_pi_d_behavior_cloning_loss(
    policy: Order9AutoregressiveDesignPolicy,
    context: DesignPolicyContext,
    trace: Sequence[Order9DesignTeacherStep],
    *,
    design_return: float | None = None,
    value_loss_weight: float = 0.5,
    entropy_bonus_weight: float = 0.0,
) -> Order9PiDBehaviorCloningLoss:
    if not trace:
        raise ValueError("pi_D BC requires a non-empty teacher trace")
    if value_loss_weight < 0.0 or entropy_bonus_weight < 0.0:
        raise ValueError("pi_D BC loss weights must be non-negative")
    history = policy.initial_history()
    negative_log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for expected_step, teacher_step in enumerate(trace):
        if teacher_step.candidate_step.step_index != expected_step:
            raise ValueError("pi_D teacher trace step indices must be contiguous")
        candidates = teacher_step.candidate_step.candidates
        selected_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.action.to_dict()
            == teacher_step.candidate_step.selected_action.to_dict()
        ]
        if len(selected_indices) != 1:
            raise ValueError("pi_D teacher action must identify exactly one candidate")
        selected_index = selected_indices[0]
        output = policy.forward_step(
            context,
            teacher_step.state,
            candidates,
            history=history,
        )
        evaluation = policy.evaluate_selected_step(output, selected_index)
        negative_log_probs.append(-evaluation.log_prob.squeeze(0))
        values.append(evaluation.value.squeeze(0))
        entropies.append(evaluation.entropy.squeeze(0))
        history = policy.advance_history(output, selected_index)
    policy_loss = torch.stack(negative_log_probs).mean()
    entropy = torch.stack(entropies).mean()
    if design_return is None:
        value_loss = torch.stack(values).sum() * 0.0
    else:
        target = torch.full_like(torch.stack(values), float(design_return))
        value_loss = F.mse_loss(torch.stack(values), target)
    total = (
        policy_loss
        + value_loss_weight * value_loss
        - entropy_bonus_weight * entropy
    )
    return Order9PiDBehaviorCloningLoss(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
        step_count=len(trace),
    )


def compute_order9_pi_d_masked_ppo_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float = 0.2,
    value_loss_weight: float = 0.5,
    entropy_bonus_weight: float = 0.01,
) -> torch.Tensor:
    if value_loss_weight < 0.0 or entropy_bonus_weight < 0.0:
        raise ValueError("pi_D PPO loss weights must be non-negative")
    actor = clipped_ppo_surrogate_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        clip_ratio=clip_ratio,
    )
    critic = F.mse_loss(new_values, returns)
    return actor + value_loss_weight * critic - entropy_bonus_weight * entropy.mean()
