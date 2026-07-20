from __future__ import annotations

"""Behavior-cloning and PPO utilities for the full Order 9 pi_H actor."""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.order9_high_level_policy import (
    GUARD_TYPES,
    PRIORITY_KEYS,
    SCHEDULE_STATES,
    Order9AutoregressiveHighLevelPolicy,
    decode_wrench_tensors,
)
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactWrenchTrajectory,
    InteractionKnot,
)


ORDER9_PI_H_LEARNING_VERSION = "order9_full_pi_h_bc_ppo_contract_v1"


@dataclass(frozen=True)
class Order9PiHLossWeights:
    assignment: float = 1.0
    schedule: float = 0.5
    wrench: float = 1.0
    timing: float = 0.25
    centroidal: float = 0.5
    posture: float = 0.5
    object_target: float = 0.75
    priority: float = 0.1
    guard: float = 0.1
    value: float = 0.5

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"Order9PiHLossWeights.{name} must be finite and non-negative")


@dataclass
class Order9PiHBehaviorCloningLoss:
    total: torch.Tensor
    assignment: torch.Tensor
    schedule: torch.Tensor
    wrench: torch.Tensor
    timing: torch.Tensor
    centroidal: torch.Tensor
    posture: torch.Tensor
    object_target: torch.Tensor
    priority: torch.Tensor
    guard: torch.Tensor
    value: torch.Tensor
    selected_assignment_count: int
    active_wrench_count: int


def compute_order9_pi_h_behavior_cloning_loss(
    policy: Order9AutoregressiveHighLevelPolicy,
    contexts: Sequence[HighLevelPolicyContext],
    teacher_trajectories: Sequence[ContactWrenchTrajectory],
    *,
    decision_returns: Sequence[float] | None = None,
    weights: Order9PiHLossWeights | None = None,
) -> Order9PiHBehaviorCloningLoss:
    """Train every persisted pi_H field, not only candidate assignment."""

    if not contexts or len(contexts) != len(teacher_trajectories):
        raise ValueError("pi_H BC requires equally sized non-empty context/trajectory batches")
    if decision_returns is not None and len(decision_returns) != len(contexts):
        raise ValueError("pi_H BC decision_returns must match the batch")
    loss_weights = weights or Order9PiHLossWeights()
    loss_weights.validate()
    cfg = policy.config
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    maximum_candidates = max(
        1, max(len(context.contact_candidate_set.candidates) for context in contexts)
    )
    targets = _teacher_targets(
        contexts,
        teacher_trajectories,
        num_knots=cfg.num_knots,
        maximum_candidates=maximum_candidates,
        device=device,
        dtype=dtype,
        policy=policy,
    )
    output = policy.forward_contexts(
        contexts,
        teacher_assignment_mask=targets["assignment"],
    )
    candidate_mask = output.candidate_mask[:, None, :].expand_as(
        targets["assignment"]
    )
    assignment = _masked_mean_loss(
        F.binary_cross_entropy_with_logits(
            output.assignment_logits,
            targets["assignment"],
            reduction="none",
        ),
        candidate_mask,
    )
    selected = targets["assignment"].bool() & candidate_mask
    schedule_raw = F.cross_entropy(
        output.schedule_logits.reshape(-1, len(SCHEDULE_STATES)),
        targets["schedule"].reshape(-1),
        reduction="none",
    ).reshape_as(targets["schedule"])
    schedule = _masked_mean_loss(schedule_raw, selected)

    predicted_target, predicted_lower, predicted_upper = decode_wrench_tensors(
        output.wrench_raw_mean,
        force_limit_n=cfg.force_limit_n,
        torque_limit_nm=cfg.torque_limit_nm,
    )
    wrench_mask = targets["wrench_mask"].unsqueeze(-1)
    wrench_terms = (
        F.smooth_l1_loss(predicted_target, targets["wrench_target"], reduction="none")
        + F.smooth_l1_loss(predicted_lower, targets["wrench_lower"], reduction="none")
        + F.smooth_l1_loss(predicted_upper, targets["wrench_upper"], reduction="none")
    )
    wrench = _masked_mean_loss(wrench_terms, wrench_mask)

    predicted_times = _differentiable_times(
        output.interval_raw_mean,
        horizon_s=cfg.horizon_s,
        minimum_fraction=cfg.minimum_interval_fraction,
    )
    timing = F.smooth_l1_loss(predicted_times, targets["times"])

    knot_raw = output.knot_target_raw_mean
    centroidal_position = F.smooth_l1_loss(
        torch.tanh(knot_raw[..., :3]),
        targets["com_position"],
        reduction="none",
    )
    centroidal_velocity = F.smooth_l1_loss(
        torch.tanh(knot_raw[..., 3:6]),
        targets["com_velocity"],
        reduction="none",
    )
    centroidal_orientation = _quaternion_loss(
        knot_raw[..., 6:10], targets["com_orientation"]
    ).unsqueeze(-1)
    centroidal_components = torch.cat(
        (centroidal_position, centroidal_velocity, centroidal_orientation), dim=-1
    )
    centroidal = _masked_mean_loss(
        centroidal_components,
        targets["centroidal_mask"].unsqueeze(-1),
    )

    anchor_raw = output.anchor_pose_raw_mean
    posture_position = F.smooth_l1_loss(
        torch.tanh(anchor_raw[..., :3]),
        targets["anchor_position"],
        reduction="none",
    )
    posture_orientation = _quaternion_loss(
        anchor_raw[..., 3:7], targets["anchor_orientation"]
    ).unsqueeze(-1)
    posture = _masked_mean_loss(
        torch.cat((posture_position, posture_orientation), dim=-1),
        targets["anchor_mask"].unsqueeze(-1),
    )

    object_raw = output.object_target_raw_mean
    object_position = F.smooth_l1_loss(
        torch.tanh(object_raw[..., :3]),
        targets["object_position"],
        reduction="none",
    )
    object_orientation = _quaternion_loss(
        object_raw[..., 3:7], targets["object_orientation"]
    ).unsqueeze(-1)
    object_twist = F.smooth_l1_loss(
        torch.tanh(object_raw[..., 7:13]),
        targets["object_twist"],
        reduction="none",
    )
    object_target = _masked_mean_loss(
        torch.cat((object_position, object_orientation, object_twist), dim=-1),
        targets["object_mask"].unsqueeze(-1),
    )

    priority = F.smooth_l1_loss(
        F.softplus(output.priority_raw_mean), targets["priority"]
    )
    guard_presence = F.binary_cross_entropy_with_logits(
        output.guard_logits, targets["guard"]
    )
    guard_threshold = _masked_mean_loss(
        F.smooth_l1_loss(
            torch.sigmoid(output.guard_threshold_raw_mean),
            targets["guard_threshold"],
            reduction="none",
        ),
        targets["guard"].bool(),
    )
    guard = guard_presence + guard_threshold
    if decision_returns is None:
        value = output.value.sum() * 0.0
    else:
        value_targets = torch.tensor(decision_returns, dtype=dtype, device=device)
        value = F.mse_loss(output.value, value_targets)

    total = (
        loss_weights.assignment * assignment
        + loss_weights.schedule * schedule
        + loss_weights.wrench * wrench
        + loss_weights.timing * timing
        + loss_weights.centroidal * centroidal
        + loss_weights.posture * posture
        + loss_weights.object_target * object_target
        + loss_weights.priority * priority
        + loss_weights.guard * guard
        + loss_weights.value * value
    )
    return Order9PiHBehaviorCloningLoss(
        total=total,
        assignment=assignment,
        schedule=schedule,
        wrench=wrench,
        timing=timing,
        centroidal=centroidal,
        posture=posture,
        object_target=object_target,
        priority=priority,
        guard=guard,
        value=value,
        selected_assignment_count=int(selected.sum().item()),
        active_wrench_count=int(targets["wrench_mask"].sum().item()),
    )


def clipped_ppo_surrogate_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """Standard clipped objective shared by full pi_H and masked pi_D."""

    if not 0.0 < clip_ratio < 1.0:
        raise ValueError("PPO clip_ratio must be in (0, 1)")
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()


def _teacher_targets(
    contexts: Sequence[HighLevelPolicyContext],
    trajectories: Sequence[ContactWrenchTrajectory],
    *,
    num_knots: int,
    maximum_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
    policy: Order9AutoregressiveHighLevelPolicy,
) -> dict[str, torch.Tensor]:
    batch = len(contexts)
    maximum_objects = max(
        1,
        max(
            0
            if context.runtime_observation is None
            else len(context.runtime_observation.object_states)
            for context in contexts
        ),
    )
    zeros = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)
    assignment = zeros(batch, num_knots, maximum_candidates)
    schedule = torch.zeros(
        (batch, num_knots, maximum_candidates), dtype=torch.long, device=device
    )
    wrench_mask = torch.zeros(
        (batch, num_knots, maximum_candidates), dtype=torch.bool, device=device
    )
    wrench_target = zeros(batch, num_knots, maximum_candidates, 6)
    wrench_lower = zeros(batch, num_knots, maximum_candidates, 6)
    wrench_upper = zeros(batch, num_knots, maximum_candidates, 6)
    times = zeros(batch, num_knots)
    com_position = zeros(batch, num_knots, 3)
    com_velocity = zeros(batch, num_knots, 3)
    com_orientation = zeros(batch, num_knots, 4)
    com_orientation[..., 3] = 1.0
    centroidal_mask = torch.zeros((batch, num_knots), dtype=torch.bool, device=device)
    anchor_position = zeros(batch, num_knots, maximum_candidates, 3)
    anchor_orientation = zeros(batch, num_knots, maximum_candidates, 4)
    anchor_orientation[..., 3] = 1.0
    anchor_mask = torch.zeros(
        (batch, num_knots, maximum_candidates), dtype=torch.bool, device=device
    )
    object_position = zeros(batch, num_knots, maximum_objects, 3)
    object_orientation = zeros(batch, num_knots, maximum_objects, 4)
    object_orientation[..., 3] = 1.0
    object_twist = zeros(batch, num_knots, maximum_objects, 6)
    object_mask = torch.zeros(
        (batch, num_knots, maximum_objects), dtype=torch.bool, device=device
    )
    priority = zeros(batch, num_knots, len(PRIORITY_KEYS))
    guard = zeros(batch, num_knots, len(GUARD_TYPES))
    guard_threshold = zeros(batch, num_knots, len(GUARD_TYPES))

    for batch_index, (context, trajectory) in enumerate(zip(contexts, trajectories)):
        if trajectory.contract_version != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME:
            raise ValueError("Order 9 full pi_H BC requires the v2 contact-frame contract")
        knots, target_times = _resample_knots(trajectory, num_knots)
        candidate_index = {
            candidate.candidate_id: index
            for index, candidate in enumerate(context.contact_candidate_set.candidates)
        }
        candidate_by_index = list(context.contact_candidate_set.candidates)
        com_reference = _centroidal_reference(context)
        object_states = (
            []
            if context.runtime_observation is None
            else context.runtime_observation.object_states
        )
        object_indices = {
            object_state.object_id: index
            for index, object_state in enumerate(object_states)
        }
        for knot_index, (knot, target_time) in enumerate(zip(knots, target_times)):
            times[batch_index, knot_index] = target_time / policy.config.horizon_s
            for teacher_assignment in knot.contact_assignments:
                index = candidate_index.get(teacher_assignment.candidate_id)
                if index is None:
                    raise ValueError("teacher pi_H assignment references an unknown candidate")
                assignment[batch_index, knot_index, index] = 1.0
                schedule[batch_index, knot_index, index] = SCHEDULE_STATES.index(
                    teacher_assignment.schedule_state
                )
                if (
                    teacher_assignment.wrench_target is not None
                    and teacher_assignment.wrench_lower is not None
                    and teacher_assignment.wrench_upper is not None
                ):
                    wrench_mask[batch_index, knot_index, index] = True
                    wrench_target[batch_index, knot_index, index] = torch.tensor(
                        teacher_assignment.wrench_target, dtype=dtype, device=device
                    )
                    wrench_lower[batch_index, knot_index, index] = torch.tensor(
                        teacher_assignment.wrench_lower, dtype=dtype, device=device
                    )
                    wrench_upper[batch_index, knot_index, index] = torch.tensor(
                        teacher_assignment.wrench_upper, dtype=dtype, device=device
                    )
            if knot.centroidal_target is not None:
                centroidal = knot.centroidal_target
                target_pose = (
                    *(centroidal.com_pos_world or com_reference[:3]),
                    *(centroidal.body_orientation_world or com_reference[3:7]),
                )
                relative = compose_pose(inverse_pose(com_reference), target_pose)
                com_position[batch_index, knot_index] = torch.tensor(
                    [
                        max(-1.0, min(1.0, value / policy.config.max_com_offset_m))
                        for value in relative[:3]
                    ],
                    dtype=dtype,
                    device=device,
                )
                velocity = centroidal.com_vel_world or (0.0, 0.0, 0.0)
                com_velocity[batch_index, knot_index] = torch.tensor(
                    [
                        max(-1.0, min(1.0, value / policy.config.max_com_velocity_mps))
                        for value in velocity
                    ],
                    dtype=dtype,
                    device=device,
                )
                com_orientation[batch_index, knot_index] = torch.tensor(
                    relative[3:7], dtype=dtype, device=device
                )
                centroidal_mask[batch_index, knot_index] = True
            free_targets = (
                {}
                if knot.posture_target is None
                or knot.posture_target.free_anchor_pose_targets is None
                else knot.posture_target.free_anchor_pose_targets
            )
            for index, candidate in enumerate(candidate_by_index):
                target_pose = free_targets.get(candidate.anchor_id)
                if target_pose is None:
                    continue
                relative = compose_pose(
                    inverse_pose(candidate.contact_pose_world), target_pose
                )
                anchor_position[batch_index, knot_index, index] = torch.tensor(
                    [
                        max(-1.0, min(1.0, value / policy.config.max_anchor_offset_m))
                        for value in relative[:3]
                    ],
                    dtype=dtype,
                    device=device,
                )
                anchor_orientation[batch_index, knot_index, index] = torch.tensor(
                    relative[3:7], dtype=dtype, device=device
                )
                anchor_mask[batch_index, knot_index, index] = True
            for target in knot.object_targets:
                index = object_indices.get(target.object_id)
                if index is None:
                    raise ValueError("teacher pi_H object target references an unknown runtime object")
                state = object_states[index]
                pose_target = target.pose_target_world or state.pose_world
                relative = compose_pose(inverse_pose(state.pose_world), pose_target)
                object_position[batch_index, knot_index, index] = torch.tensor(
                    [
                        max(-1.0, min(1.0, value / policy.config.max_object_offset_m))
                        for value in relative[:3]
                    ],
                    dtype=dtype,
                    device=device,
                )
                object_orientation[batch_index, knot_index, index] = torch.tensor(
                    relative[3:7], dtype=dtype, device=device
                )
                twist = target.twist_target_world or [0.0] * 6
                object_twist[batch_index, knot_index, index] = torch.tensor(
                    [
                        max(-1.0, min(1.0, value / policy.config.max_object_twist))
                        for value in twist
                    ],
                    dtype=dtype,
                    device=device,
                )
                object_mask[batch_index, knot_index, index] = True
            priority[batch_index, knot_index] = torch.tensor(
                _canonical_priorities(knot), dtype=dtype, device=device
            )
            for condition in knot.guard_conditions:
                index = GUARD_TYPES.index(_canonical_guard(condition))
                guard[batch_index, knot_index, index] = 1.0
                threshold = condition.get("threshold", 0.5)
                guard_threshold[batch_index, knot_index, index] = max(
                    0.0, min(1.0, float(threshold))
                )
    return {
        "assignment": assignment,
        "schedule": schedule,
        "wrench_mask": wrench_mask,
        "wrench_target": wrench_target,
        "wrench_lower": wrench_lower,
        "wrench_upper": wrench_upper,
        "times": times,
        "com_position": com_position,
        "com_velocity": com_velocity,
        "com_orientation": com_orientation,
        "centroidal_mask": centroidal_mask,
        "anchor_position": anchor_position,
        "anchor_orientation": anchor_orientation,
        "anchor_mask": anchor_mask,
        "object_position": object_position,
        "object_orientation": object_orientation,
        "object_twist": object_twist,
        "object_mask": object_mask,
        "priority": priority,
        "guard": guard,
        "guard_threshold": guard_threshold,
    }


def _resample_knots(
    trajectory: ContactWrenchTrajectory,
    count: int,
) -> tuple[list[InteractionKnot], list[float]]:
    if not trajectory.knots:
        raise ValueError("teacher trajectory has no knots")
    target_times = [trajectory.horizon_s * index / float(count - 1) for index in range(count)]
    knots: list[InteractionKnot] = []
    source_index = 0
    for time_s in target_times:
        while (
            source_index + 1 < len(trajectory.knots)
            and trajectory.knots[source_index + 1].t_rel_s <= time_s + 1.0e-9
        ):
            source_index += 1
        knots.append(trajectory.knots[source_index])
    return knots, target_times


def _differentiable_times(
    raw: torch.Tensor,
    *,
    horizon_s: float,
    minimum_fraction: float,
) -> torch.Tensor:
    intervals = F.softplus(raw[:, 1:]) + minimum_fraction
    fractions = intervals / intervals.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return torch.cat(
        (raw.new_zeros((raw.shape[0], 1)), torch.cumsum(fractions, dim=1)),
        dim=1,
    )


def _quaternion_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(predicted, dim=-1, keepdim=True).clamp_min(1.0e-8)
    predicted_unit = predicted / norm
    target_unit = target / torch.linalg.vector_norm(
        target, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    return 1.0 - torch.abs((predicted_unit * target_unit).sum(dim=-1))


def _masked_mean_loss(loss: torch.Tensor, mask: torch.Tensor | float) -> torch.Tensor:
    if isinstance(mask, float):
        return loss.mean() * mask
    expanded = mask.to(loss.dtype)
    while expanded.ndim < loss.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(loss)
    denominator = expanded.sum().clamp_min(1.0)
    return (loss * expanded).sum() / denominator


def _centroidal_reference(context: HighLevelPolicyContext):
    observation = context.runtime_observation
    if observation is not None and observation.module_states:
        count = float(len(observation.module_states))
        position = tuple(
            sum(state.pose_world[axis] for state in observation.module_states) / count
            for axis in range(3)
        )
        return (*position, *observation.module_states[0].pose_world[3:7])
    modules = context.morphology_graph.modules
    count = float(len(modules))
    position = tuple(
        sum(module.pose_in_design_frame[axis] for module in modules) / count
        for axis in range(3)
    )
    return (*position, *modules[0].pose_in_design_frame[3:7])


def _canonical_priorities(knot: InteractionKnot) -> list[float]:
    result = {key: 0.0 for key in PRIORITY_KEYS}
    for source, raw_value in knot.priority_weights.items():
        value = max(0.0, float(raw_value))
        lower = source.lower()
        if any(token in lower for token in ("contact", "anchor", "grasp", "attach")):
            result["contact"] = max(result["contact"], value)
        if any(token in lower for token in ("centroid", "stability", "body", "lift")):
            result["centroidal"] = max(result["centroidal"], value)
        if any(token in lower for token in ("posture", "approach", "align")):
            result["posture"] = max(result["posture"], value)
        if any(token in lower for token in ("object", "transport", "place", "release")):
            result["object"] = max(result["object"], value)
        if any(token in lower for token in ("safety", "collision", "qp")):
            result["safety"] = max(result["safety"], value)
    return [result[key] for key in PRIORITY_KEYS]


def _canonical_guard(condition: dict[str, object]) -> str:
    source = str(condition.get("type", "task_phase_complete")).lower()
    if any(token in source for token in ("elapsed", "dwell", "time")):
        return "elapsed_fraction"
    if any(token in source for token in ("pose", "align", "goal", "transport")):
        return "pose_error_below"
    if any(token in source for token in ("velocity", "speed", "settle")):
        return "velocity_below"
    if any(token in source for token in ("contact", "attach", "grasp")):
        return "contact_estimate_valid"
    if any(token in source for token in ("controller", "qp", "feasible")):
        return "controller_feasible"
    return "task_phase_complete"
