from __future__ import annotations

"""Privileged, training-only warm-start for the C3 common compression head.

This module does not add contact measurements to the deployable actor.  It
uses immutable Isaac rollout tensors to distil a morphology-conditioned
compression correction into the existing contact residual branch before PPO
continues from a fresh on-policy generation.
"""

import copy
import itertools
import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_RESIDUAL_PHASE_INDICES,
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.training.order9_tensor_pi_l_ppo import (
    _PolicyBatch,
    _TensorReplay,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    Order9TensorRolloutArtifact,
)

ORDER9_CONTACT_RESIDUAL_WARM_START_VERSION = (
    "order9_morphology_invariant_compression_privileged_warm_start_v2"
)


@dataclass(frozen=True)
class Order9ContactResidualWarmStartConfig:
    target_module_count: int = 5
    compression_action_step: float = 0.40
    maximum_target_action: float = 0.90
    learning_rate: float = 1.0e-3
    epochs: int = 12
    batch_size: int = 1024
    maximum_samples_per_artifact: int = 8192
    anchor_loss_weight: float = 1.0
    anchor_control_loss_weight: float = 10.0
    target_non_control_loss_weight: float = 0.25
    control_output_row_only: bool = True
    target_rmse_stop: float = 0.025
    anchor_rmse_stop: float = 0.010
    minimum_epochs: int = 3
    seed: int = 0

    def validate(self) -> None:
        if self.target_module_count < 2:
            raise ValueError("Order9 warm-start target module count is invalid")
        if not 0.0 < self.compression_action_step <= 1.0:
            raise ValueError("Order9 warm-start action step is invalid")
        if not 0.0 < self.maximum_target_action < 1.0:
            raise ValueError("Order9 warm-start target action limit is invalid")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("Order9 warm-start learning rate is invalid")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.maximum_samples_per_artifact,
                self.minimum_epochs,
            )
            < 1
        ):
            raise ValueError("Order9 warm-start iteration settings are invalid")
        if self.minimum_epochs > self.epochs:
            raise ValueError("Order9 warm-start minimum epochs exceed its budget")
        if (
            min(
                self.anchor_loss_weight,
                self.anchor_control_loss_weight,
                self.target_non_control_loss_weight,
                self.target_rmse_stop,
                self.anchor_rmse_stop,
            )
            < 0.0
        ):
            raise ValueError("Order9 warm-start loss settings are invalid")
        if self.seed < 0:
            raise ValueError("Order9 warm-start seed is invalid")


@dataclass(frozen=True)
class Order9ContactResidualWarmStartResult:
    optimizer_step_count: int
    completed_epochs: int
    target_sample_count: int
    anchor_sample_count: int
    initial_target_rmse: float
    final_target_rmse: float
    initial_anchor_rmse: float
    final_anchor_rmse: float
    initial_anchor_control_rmse: float
    final_anchor_control_rmse: float
    final_anchor_maximum_absolute_error: float
    maximum_non_contact_parameter_error: float
    stopped_early: bool
    epoch_metrics: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "warm_start_version": ORDER9_CONTACT_RESIDUAL_WARM_START_VERSION,
            "optimizer_step_count": self.optimizer_step_count,
            "completed_epochs": self.completed_epochs,
            "target_sample_count": self.target_sample_count,
            "anchor_sample_count": self.anchor_sample_count,
            "initial_target_rmse": self.initial_target_rmse,
            "final_target_rmse": self.final_target_rmse,
            "initial_anchor_rmse": self.initial_anchor_rmse,
            "final_anchor_rmse": self.final_anchor_rmse,
            "initial_anchor_control_rmse": self.initial_anchor_control_rmse,
            "final_anchor_control_rmse": self.final_anchor_control_rmse,
            "final_anchor_maximum_absolute_error": (
                self.final_anchor_maximum_absolute_error
            ),
            "maximum_non_contact_parameter_error": (
                self.maximum_non_contact_parameter_error
            ),
            "stopped_early": self.stopped_early,
            "epoch_metrics": list(self.epoch_metrics),
            "raw_contact_actor_input": False,
            "privileged_teacher_training_only": True,
        }


@dataclass(frozen=True)
class _WarmStartSource:
    replay: _TensorReplay
    indices: torch.Tensor
    target: bool


def order9_contact_residual_warm_start_target(
    parent_action: torch.Tensor,
    teacher_direction: torch.Tensor,
    *,
    action_step: float,
    maximum_action: float,
) -> torch.Tensor:
    """Return an underforce-only compression target in bounded action space."""

    if parent_action.shape != teacher_direction.shape:
        raise ValueError("Order9 warm-start target tensor shapes differ")
    if not bool(torch.isfinite(parent_action).all()) or not bool(
        torch.isfinite(teacher_direction).all()
    ):
        raise ValueError("Order9 warm-start target tensors are non-finite")
    if not 0.0 < action_step <= 1.0 or not 0.0 < maximum_action < 1.0:
        raise ValueError("Order9 warm-start target limits are invalid")
    correction = action_step * teacher_direction.clamp(min=0.0, max=1.0)
    return (parent_action + correction).clamp(min=-maximum_action, max=maximum_action)


def configure_order9_contact_residual_warm_start_parameters(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    *,
    target_module_count: int = 5,
) -> tuple[str, ...]:
    """Expose only one zero-initialized module-count calibration scalar."""

    if not 0 <= target_module_count <= policy.config.max_modules:
        raise SchemaValidationError(
            "Order9 warm-start target module count exceeds the actor contract"
        )

    trainable = []
    for name, parameter in policy.named_parameters():
        parameter.requires_grad = name == "contact_compression_module_count_bias"
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise SchemaValidationError("Order9 warm-start has no trainable parameters")
    gradient_mask = torch.zeros_like(
        policy.contact_compression_module_count_bias
    )
    gradient_mask[target_module_count] = 1.0
    policy.contact_compression_module_count_bias.register_hook(
        lambda gradient: gradient * gradient_mask
    )
    return tuple(trainable)


def warm_start_order9_contact_residual(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    artifacts: Sequence[Order9TensorRolloutArtifact],
    *,
    physical_model: PhysicalModel,
    config: Order9ContactResidualWarmStartConfig,
) -> Order9ContactResidualWarmStartResult:
    """Fit the 5-module compression residual while anchoring prior morphologies."""

    config.validate()
    if not artifacts:
        raise SchemaValidationError("Order9 warm-start has no rollout artifacts")
    parameter = next(policy.parameters())
    device = parameter.device
    parent = copy.deepcopy(policy).to(device)
    parent.eval()
    for value in parent.parameters():
        value.requires_grad = False
    original_parameters = {
        name: value.detach().cpu().clone() for name, value in policy.named_parameters()
    }
    sources = _build_sources(
        policy,
        artifacts,
        physical_model=physical_model,
        target_module_count=config.target_module_count,
    )
    targets = tuple(source for source in sources if source.target)
    anchors = tuple(source for source in sources if not source.target)
    if not targets:
        raise SchemaValidationError(
            "Order9 warm-start has no target morphology samples"
        )
    if not anchors:
        raise SchemaValidationError("Order9 warm-start has no preservation samples")
    configure_order9_contact_residual_warm_start_parameters(
        policy, target_module_count=config.target_module_count
    )
    policy.train()

    optimizer = torch.optim.Adam(
        [value for value in policy.parameters() if value.requires_grad],
        lr=config.learning_rate,
    )
    initial = _evaluate_sources(policy, parent, targets, anchors, config=config)
    rows: list[dict[str, float]] = []
    optimizer_steps = 0
    stopped_early = False
    for epoch in range(config.epochs):
        target_batches = _epoch_batches(
            targets,
            batch_size=config.batch_size,
            maximum_samples_per_artifact=config.maximum_samples_per_artifact,
            seed=config.seed + epoch * 10_007,
        )
        anchor_batches = _epoch_batches(
            anchors,
            batch_size=config.batch_size,
            maximum_samples_per_artifact=config.maximum_samples_per_artifact,
            seed=config.seed + epoch * 10_007 + 1,
        )
        if not target_batches or not anchor_batches:
            raise SchemaValidationError("Order9 warm-start epoch has no batches")
        random.Random(config.seed + epoch).shuffle(target_batches)
        random.Random(config.seed + epoch + 1).shuffle(anchor_batches)
        target_cycle = itertools.cycle(target_batches)
        anchor_cycle = itertools.cycle(anchor_batches)
        epoch_losses = []
        for _ in range(max(len(target_batches), len(anchor_batches))):
            target_source, target_indices = next(target_cycle)
            anchor_source, anchor_indices = next(anchor_cycle)
            optimizer.zero_grad(set_to_none=True)
            target_loss, target_other_loss = _target_batch_losses(
                policy,
                parent,
                target_source,
                target_indices,
                config=config,
            )
            anchor_loss, anchor_control_loss = _anchor_batch_losses(
                policy, parent, anchor_source, anchor_indices
            )
            total = (
                target_loss
                + config.target_non_control_loss_weight * target_other_loss
                + config.anchor_loss_weight * anchor_loss
                + config.anchor_control_loss_weight * anchor_control_loss
            )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Order9 warm-start loss is non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in policy.parameters() if value.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(total.detach().cpu().item()))
        metrics = _evaluate_sources(policy, parent, targets, anchors, config=config)
        rows.append(
            {
                "epoch": float(epoch),
                "loss_mean": sum(epoch_losses) / len(epoch_losses),
                **metrics,
            }
        )
        if (
            epoch + 1 >= config.minimum_epochs
            and metrics["target_rmse"] <= config.target_rmse_stop
            and metrics["anchor_rmse"] <= config.anchor_rmse_stop
            and metrics["anchor_control_rmse"] <= config.anchor_rmse_stop
        ):
            stopped_early = True
            break

    final = rows[-1]
    frozen_errors = []
    for name, value in policy.named_parameters():
        difference = value.detach().cpu() - original_parameters[name]
        if name == "contact_compression_module_count_bias":
            difference = difference.clone()
            difference[config.target_module_count] = 0.0
        frozen_errors.append(float(difference.abs().max().item()))
    maximum_non_contact_error = max(frozen_errors, default=0.0)
    if maximum_non_contact_error != 0.0:
        raise SchemaValidationError(
            "Order9 warm-start changed a frozen policy parameter"
        )
    return Order9ContactResidualWarmStartResult(
        optimizer_step_count=optimizer_steps,
        completed_epochs=len(rows),
        target_sample_count=sum(len(source.indices) for source in targets),
        anchor_sample_count=sum(len(source.indices) for source in anchors),
        initial_target_rmse=initial["target_rmse"],
        final_target_rmse=final["target_rmse"],
        initial_anchor_rmse=initial["anchor_rmse"],
        final_anchor_rmse=final["anchor_rmse"],
        initial_anchor_control_rmse=initial["anchor_control_rmse"],
        final_anchor_control_rmse=final["anchor_control_rmse"],
        final_anchor_maximum_absolute_error=final["anchor_max_abs"],
        maximum_non_contact_parameter_error=maximum_non_contact_error,
        stopped_early=stopped_early,
        epoch_metrics=tuple(rows),
    )


def _build_sources(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    artifacts: Sequence[Order9TensorRolloutArtifact],
    *,
    physical_model: PhysicalModel,
    target_module_count: int,
) -> tuple[_WarmStartSource, ...]:
    sources = []
    for artifact in artifacts:
        artifact.validate()
        module_count = len(artifact.metadata.get("module_ids", ()))
        replay = _TensorReplay(artifact, policy=policy, physical_model=physical_model)
        phase = artifact.tensors["phase_index"]
        valid = artifact.tensors["valid"].bool()
        contact_phase = torch.zeros_like(valid)
        for phase_index in ORDER9_CONTACT_RESIDUAL_PHASE_INDICES:
            contact_phase |= phase == phase_index
        target = module_count == target_module_count
        if target:
            selected = (
                valid
                & contact_phase
                & replay.compression_teacher_mask
                & (replay.compression_teacher_direction > 1.0e-6)
            )
        elif module_count < target_module_count:
            selected = valid & contact_phase
        else:
            continue
        indices = selected.nonzero(as_tuple=False).cpu()
        if len(indices):
            sources.append(
                _WarmStartSource(replay=replay, indices=indices, target=target)
            )
    return tuple(sources)


def _epoch_batches(
    sources: Sequence[_WarmStartSource],
    *,
    batch_size: int,
    maximum_samples_per_artifact: int,
    seed: int,
) -> list[tuple[_WarmStartSource, torch.Tensor]]:
    result = []
    for ordinal, source in enumerate(sources):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + ordinal * 100_003)
        order = torch.randperm(len(source.indices), generator=generator)
        order = order[:maximum_samples_per_artifact]
        selected = source.indices.index_select(0, order)
        for start in range(0, len(selected), batch_size):
            result.append((source, selected[start : start + batch_size]))
    return result


def _policy_step(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    batch: _PolicyBatch,
):
    return policy.step(
        batch.graph,
        None,
        batch.actor_features,
        batch.previous_global_action,
        batch.recurrent_state_in,
        phase_features=batch.phase_features,
        privileged_disturbance_body=batch.privileged,
        deterministic=True,
        active_knot_features=batch.active_knot_features,
        active_assignment_features=batch.active_assignment_features,
    )


def _materialize_batch(source: _WarmStartSource, indices: torch.Tensor) -> _PolicyBatch:
    return source.replay.batch(indices[:, 0].tolist(), indices[:, 1].tolist())


def _target_batch_losses(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    parent: Order9MorphologyInvariantCompressionActorCritic,
    source: _WarmStartSource,
    indices: torch.Tensor,
    *,
    config: Order9ContactResidualWarmStartConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = _materialize_batch(source, indices)
    with torch.no_grad():
        parent_step = _policy_step(parent, batch)
    current_step = _policy_step(policy, batch)
    parent_scalar = parent_step.contact_compression_residual_action_mean
    desired = order9_contact_residual_warm_start_target(
        parent_scalar,
        batch.compression_teacher_direction,
        action_step=config.compression_action_step,
        maximum_action=config.maximum_target_action,
    )
    target_loss = torch.nn.functional.smooth_l1_loss(
        current_step.contact_compression_residual_action_mean, desired
    )
    other_loss = target_loss * 0.0
    return target_loss, other_loss


def _anchor_batch_losses(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    parent: Order9MorphologyInvariantCompressionActorCritic,
    source: _WarmStartSource,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = _materialize_batch(source, indices)
    with torch.no_grad():
        parent_mean = _policy_step(
            parent, batch
        ).contact_compression_residual_action_mean
    current_mean = _policy_step(
        policy, batch
    ).contact_compression_residual_action_mean
    difference = current_mean - parent_mean
    loss = difference.square().mean()
    return loss, loss


def _evaluate_sources(
    policy: Order9MorphologyInvariantCompressionActorCritic,
    parent: Order9MorphologyInvariantCompressionActorCritic,
    targets: Sequence[_WarmStartSource],
    anchors: Sequence[_WarmStartSource],
    *,
    config: Order9ContactResidualWarmStartConfig,
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    target_errors = []
    anchor_errors = []
    anchor_control_errors = []
    with torch.no_grad():
        for source, indices in _epoch_batches(
            targets,
            batch_size=config.batch_size,
            maximum_samples_per_artifact=config.maximum_samples_per_artifact,
            seed=config.seed + 900_001,
        ):
            batch = _materialize_batch(source, indices)
            parent_mean = _policy_step(
                parent, batch
            ).contact_compression_residual_action_mean
            current_mean = _policy_step(
                policy, batch
            ).contact_compression_residual_action_mean
            desired = order9_contact_residual_warm_start_target(
                parent_mean,
                batch.compression_teacher_direction,
                action_step=config.compression_action_step,
                maximum_action=config.maximum_target_action,
            )
            target_errors.append(current_mean - desired)
        for source, indices in _epoch_batches(
            anchors,
            batch_size=config.batch_size,
            maximum_samples_per_artifact=config.maximum_samples_per_artifact,
            seed=config.seed + 900_002,
        ):
            batch = _materialize_batch(source, indices)
            parent_mean = _policy_step(
                parent, batch
            ).contact_compression_residual_action_mean
            current_mean = _policy_step(
                policy, batch
            ).contact_compression_residual_action_mean
            difference = current_mean - parent_mean
            anchor_errors.append(difference.reshape(-1))
            anchor_control_errors.append(difference)
    if was_training:
        policy.train()
    target_error = torch.cat(target_errors)
    anchor_error = torch.cat(anchor_errors)
    anchor_control_error = torch.cat(anchor_control_errors)
    return {
        "target_rmse": math.sqrt(float(target_error.square().mean().cpu())),
        "target_mean_error": float(target_error.mean().cpu()),
        "anchor_rmse": math.sqrt(float(anchor_error.square().mean().cpu())),
        "anchor_control_rmse": math.sqrt(
            float(anchor_control_error.square().mean().cpu())
        ),
        "anchor_max_abs": float(anchor_error.abs().max().cpu()),
    }


__all__ = [
    "ORDER9_CONTACT_RESIDUAL_WARM_START_VERSION",
    "Order9ContactResidualWarmStartConfig",
    "Order9ContactResidualWarmStartResult",
    "configure_order9_contact_residual_warm_start_parameters",
    "order9_contact_residual_warm_start_target",
    "warm_start_order9_contact_residual",
]
