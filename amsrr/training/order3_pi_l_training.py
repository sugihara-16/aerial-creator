from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.policies.low_level_policy_base import BaselineLowLevelPolicyConfig
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_POLICY_OUTPUT_MODE,
    MorphologyConditionedActorCritic,
    Order3MorphologyConditionedPolicyConfig,
    load_order3_policy_checkpoint,
    order3_actor_feature_schema_hash,
    order3_actor_feature_vector,
    order3_graph_feature_schema_hash,
    save_order3_policy_checkpoint,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3PolicyCheckpointMetadata,
    Order3PolicyTransition,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL
from amsrr.training.order3_dataset import Order3DatasetIOResult, load_order3_dataset
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file, stable_hash


ORDER3_PI_L_TRAINING_VERSION = "order3_morphology_pi_l_training_v1"
DEFAULT_ORDER3_TRAINING_CONFIG_PATH = "configs/training/order3_morphology_pi_l.yaml"
DEFAULT_ORDER3_TRAINING_ROOT = "artifacts/p4_full/order3_pi_l_v2/training"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_P4_3_ROOT = (_REPOSITORY_ROOT / "artifacts" / "p4_3").resolve()
_EPSILON = 1.0e-8


@dataclass
class Order3BCTrainingConfig(SchemaBase):
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    sequence_length: int = 16
    burn_in_steps: int = 4
    value_loss_weight: float = 0.25
    max_grad_norm: float = 0.5

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "sequence_length"):
            if int(getattr(self, name)) <= 0:
                raise SchemaValidationError(f"Order3BCTrainingConfig.{name} must be positive")
        if self.burn_in_steps < 0 or self.burn_in_steps >= self.sequence_length:
            raise SchemaValidationError(
                "Order3BCTrainingConfig.burn_in_steps must be in [0, sequence_length)"
            )
        for name in ("learning_rate", "max_grad_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3BCTrainingConfig.{name} must be finite and positive"
                )
        if not math.isfinite(self.value_loss_weight) or self.value_loss_weight < 0.0:
            raise SchemaValidationError(
                "Order3BCTrainingConfig.value_loss_weight must be finite and non-negative"
            )


@dataclass
class Order3PPOTrainingConfig(SchemaBase):
    updates: int = 40
    rollout_steps_per_update: int = 2048
    epochs_per_update: int = 4
    minibatch_size: int = 256
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.001
    max_grad_norm: float = 0.5
    recurrent_burn_in_steps: int = 4

    def validate(self) -> None:
        for name in (
            "updates",
            "rollout_steps_per_update",
            "epochs_per_update",
            "minibatch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise SchemaValidationError(f"Order3PPOTrainingConfig.{name} must be positive")
        if self.recurrent_burn_in_steps < 0:
            raise SchemaValidationError(
                "Order3PPOTrainingConfig.recurrent_burn_in_steps must be non-negative"
            )
        for name in ("learning_rate", "clip_ratio", "max_grad_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3PPOTrainingConfig.{name} must be finite and positive"
                )
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0 or not math.isfinite(value):
                raise SchemaValidationError(
                    f"Order3PPOTrainingConfig.{name} must be finite and in [0, 1]"
                )
        for name in ("value_loss_weight", "entropy_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order3PPOTrainingConfig.{name} must be finite and non-negative"
                )


@dataclass
class Order3PiLTrainingConfig(SchemaBase):
    seed: int = 3011
    device: str = "cpu"
    artifact_root: str = "artifacts/p4_full/order3_pi_l_v2"
    bc: Order3BCTrainingConfig = field(default_factory=Order3BCTrainingConfig)
    ppo: Order3PPOTrainingConfig = field(default_factory=Order3PPOTrainingConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise SchemaValidationError("Order3PiLTrainingConfig.seed must be non-negative")
        require_non_empty(self.device, "Order3PiLTrainingConfig.device")
        require_non_empty(self.artifact_root, "Order3PiLTrainingConfig.artifact_root")
        _reject_legacy_p4_3_path(Path(self.artifact_root))


@dataclass(frozen=True)
class Order3GAEResult:
    advantages: list[float]
    returns: list[float]


@dataclass(frozen=True)
class Order3PiLTrainingResult:
    output_root: str
    bc_checkpoint_path: str
    ppo_checkpoint_path: str
    bc_loss_curve_path: str
    ppo_loss_curve_path: str
    reward_curve_path: str
    bc_metrics_path: str
    ppo_metrics_path: str
    summary_path: str
    bc_checkpoint_sha256: str
    ppo_checkpoint_sha256: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class Order3BCTrainingResult:
    output_root: str
    checkpoint_path: str
    loss_curve_path: str
    metrics_path: str
    checkpoint_sha256: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class Order3PPOTrainingResult:
    output_root: str
    checkpoint_path: str
    loss_curve_path: str
    reward_curve_path: str
    metrics_path: str
    checkpoint_sha256: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _PreparedTransition:
    record: Order3PolicyTransition
    actor_features: tuple[float, ...]


@dataclass(frozen=True)
class _PPOBehaviorContract:
    policy_version: str
    checkpoint_hash: str


@dataclass(frozen=True)
class _TrainingDataContext:
    dataset: Order3DatasetIOResult
    dataset_hash: str
    urdf_hash: str
    controller_contract_hash: str
    splits: dict[DatasetSplit, list[_PreparedTransition]]


def load_order3_pi_l_training_config(
    path: str | Path = DEFAULT_ORDER3_TRAINING_CONFIG_PATH,
) -> tuple[Order3PiLTrainingConfig, Order3MorphologyConditionedPolicyConfig]:
    data = load_config(path)
    return (
        Order3PiLTrainingConfig.from_dict(data.get("training", {})),
        Order3MorphologyConditionedPolicyConfig.from_dict(data.get("policy", {})),
    )


def compute_order3_gae(
    transitions: Sequence[Order3PolicyTransition],
    *,
    gamma: float,
    gae_lambda: float,
) -> Order3GAEResult:
    """Compute episode-bounded GAE in the caller's transition order."""

    if not 0.0 <= gamma <= 1.0 or not math.isfinite(gamma):
        raise ValueError("Order3 GAE gamma must be finite and in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0 or not math.isfinite(gae_lambda):
        raise ValueError("Order3 GAE lambda must be finite and in [0, 1]")
    records = list(transitions)
    if not records:
        raise ValueError("Order3 GAE requires transitions")
    advantages = [0.0] * len(records)
    returns = [0.0] * len(records)
    episode_indices: dict[str, list[int]] = {}
    episode_contracts: dict[str, tuple[DatasetSplit, str]] = {}
    for index, record in enumerate(records):
        contract = (record.split, record.structural_hash)
        if episode_contracts.setdefault(record.episode_id, contract) != contract:
            raise ValueError("Order3 GAE episode crosses split or morphology boundary")
        episode_indices.setdefault(record.episode_id, []).append(index)
    for episode_id, raw_indices in episode_indices.items():
        indices = sorted(raw_indices, key=lambda index: records[index].step_index)
        if len({records[index].step_index for index in indices}) != len(indices):
            raise ValueError(f"Order3 GAE episode {episode_id!r} has duplicate step indices")
        boundary_positions = [
            position
            for position, index in enumerate(indices)
            if records[index].terminal or records[index].truncated
        ]
        if boundary_positions and boundary_positions != [len(indices) - 1]:
            raise ValueError(
                f"Order3 GAE episode {episode_id!r} terminal/truncated row must be final"
            )
        next_advantage = 0.0
        for position in range(len(indices) - 1, -1, -1):
            index = indices[position]
            record = records[index]
            has_next = position + 1 < len(indices)
            if record.truncated:
                # Bootstrap the TD target at a time-limit boundary, while still
                # preventing GAE from leaking into the next episode.
                next_value = float(record.bootstrap_value)
                value_continuation = 1.0
                gae_continuation = 0.0
            elif record.terminal:
                next_value = 0.0
                value_continuation = 0.0
                gae_continuation = 0.0
            elif has_next:
                next_value = float(records[indices[position + 1]].old_value)
                value_continuation = 1.0
                gae_continuation = 1.0
            else:
                next_value = 0.0
                value_continuation = 0.0
                gae_continuation = 0.0
            delta = (
                float(record.reward)
                + gamma * next_value * value_continuation
                - float(record.old_value)
            )
            advantage = delta + gamma * gae_lambda * gae_continuation * next_advantage
            if not math.isfinite(advantage):
                raise ValueError("Order3 GAE produced a non-finite advantage")
            advantages[index] = advantage
            returns[index] = advantage + float(record.old_value)
            next_advantage = advantage
    return Order3GAEResult(advantages=advantages, returns=returns)


def _resolve_training_configs(
    *,
    training_config: Order3PiLTrainingConfig | None,
    policy_config: Order3MorphologyConditionedPolicyConfig | None,
    config_path: str | Path,
) -> tuple[Order3PiLTrainingConfig, Order3MorphologyConditionedPolicyConfig]:
    loaded_training: Order3PiLTrainingConfig | None = None
    loaded_policy: Order3MorphologyConditionedPolicyConfig | None = None
    if training_config is None or policy_config is None:
        loaded_training, loaded_policy = load_order3_pi_l_training_config(config_path)
    resolved_training = training_config or loaded_training
    resolved_policy = policy_config or loaded_policy
    if resolved_training is None or resolved_policy is None:  # pragma: no cover
        raise RuntimeError("Order3 training configuration resolution failed")
    return resolved_training, resolved_policy


def _load_training_data_context(
    *,
    dataset_path: str | Path,
    physical_model: PhysicalModel,
    model_config: Order3MorphologyConditionedPolicyConfig,
    controller_contract_hash: str | None,
) -> _TrainingDataContext:
    dataset = load_order3_dataset(dataset_path)
    if dataset.manifest.physical_model_hash != physical_model.stable_hash():
        raise SchemaValidationError(
            "Order3 training PhysicalModel hash does not match the dataset manifest"
        )
    urdf_path = Path(physical_model.urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Order3 training URDF does not exist: {urdf_path}")
    contract_hash = controller_contract_hash or stable_hash(
        {
            "policy_contract": POLICY_COMMAND_CONTRACT_CENTROIDAL,
            "qp_scope": "rotor_thrust_vectoring_and_slack_only",
            "local_joint_scope": "absolute_targets_and_bounded_torque_bias",
        }
    )
    prepared = _prepare_transitions(dataset.transitions, physical_model)
    splits = {
        split: [item for item in prepared if item.record.split == split]
        for split in DatasetSplit
    }
    _validate_training_splits(splits, model_config)
    return _TrainingDataContext(
        dataset=dataset,
        dataset_hash=hash_file(dataset.manifest_path),
        urdf_hash=hash_file(urdf_path),
        controller_contract_hash=contract_hash,
        splits=splits,
    )


def train_order3_pi_l_bc(
    *,
    dataset_path: str | Path,
    physical_model: PhysicalModel,
    training_config: Order3PiLTrainingConfig | None = None,
    policy_config: Order3MorphologyConditionedPolicyConfig | None = None,
    config_path: str | Path = DEFAULT_ORDER3_TRAINING_CONFIG_PATH,
    output_root: str | Path | None = None,
    git_revision: str = "unknown",
    controller_contract_hash: str | None = None,
) -> Order3BCTrainingResult:
    """Train the production BC stage from deterministic-teacher rows only."""

    train_cfg, model_cfg = _resolve_training_configs(
        training_config=training_config,
        policy_config=policy_config,
        config_path=config_path,
    )
    target_root = Path(
        output_root or (Path(train_cfg.artifact_root) / "training" / "bc")
    )
    _reject_legacy_p4_3_path(target_root)
    require_non_empty(git_revision, "Order3 BC training git_revision")
    device = _training_device(train_cfg.device)
    context = _load_training_data_context(
        dataset_path=dataset_path,
        physical_model=physical_model,
        model_config=model_cfg,
        controller_contract_hash=controller_contract_hash,
    )
    bc_records, ppo_records, _ = _partition_stage_records(context.splits)
    if any(ppo_records.values()):
        raise SchemaValidationError(
            "Order3 production BC dataset must contain deterministic teacher rows only"
        )
    _require_complete_stage_splits(bc_records, stage="BC")

    _seed_everything(train_cfg.seed)
    model = MorphologyConditionedActorCritic(model_cfg).to(device)
    initial = _evaluate_bc(
        model,
        bc_records,
        device=device,
        value_loss_weight=train_cfg.bc.value_loss_weight,
    )
    loss_rows, gradient_max = _train_bc(
        model,
        bc_records[DatasetSplit.TRAIN],
        config=train_cfg.bc,
        seed=train_cfg.seed,
        device=device,
    )
    final = _evaluate_bc(
        model,
        bc_records,
        device=device,
        value_loss_weight=train_cfg.bc.value_loss_weight,
    )
    metrics: dict[str, Any] = {
        "training_version": ORDER3_PI_L_TRAINING_VERSION,
        "stage": "bc",
        "workflow": "production_staged",
        "dataset_hash": context.dataset_hash,
        "dataset_manifest_path": context.dataset.manifest_path,
        "training_config_hash": train_cfg.stable_hash(),
        "policy_config_hash": model_cfg.stable_hash(),
        "train_transition_count": len(bc_records[DatasetSplit.TRAIN]),
        "validation_transition_count": len(bc_records[DatasetSplit.VALIDATION]),
        "held_out_transition_count": len(bc_records[DatasetSplit.HELD_OUT]),
        "behavior_policy_kind": "deterministic_v2_teacher",
        "behavior_policy_version": ORDER3_FALLBACK_VERSION,
        "action_semantics": "reference_hold",
        "zero_teacher_action_count": sum(
            _is_zero_action(item.record.action)
            for item in bc_records[DatasetSplit.TRAIN]
        ),
        "reference_teacher_action_count": len(bc_records[DatasetSplit.TRAIN]),
        "initial": initial,
        "final": final,
        "max_preclip_gradient_norm": gradient_max,
        "gradient_clip_norm": train_cfg.bc.max_grad_norm,
        "episode_sequence_training": True,
        "recurrent_state_in_used": True,
        "actor_uses_privileged_disturbance": False,
        "critic_uses_privileged_disturbance": True,
        "held_out_used_for_optimization": False,
        "p4_full_completion_claim": False,
    }
    target_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_root / "checkpoint.pt"
    metadata = _checkpoint_metadata(
        stage="bc",
        model_config=model_cfg,
        training_config=train_cfg,
        dataset_manifest=context.dataset.manifest,
        dataset_hash=context.dataset_hash,
        urdf_hash=context.urdf_hash,
        controller_contract_hash=context.controller_contract_hash,
        git_revision=git_revision,
        parent_bc_checkpoint_hash=None,
        stage_metrics=metrics,
    )
    checkpoint_hash = save_order3_policy_checkpoint(
        checkpoint_path,
        model=model,
        metadata=metadata,
    )
    metrics["checkpoint_sha256"] = checkpoint_hash
    loss_path = target_root / "loss_curve.csv"
    metrics_path = target_root / "metrics.json"
    _write_csv(loss_path, loss_rows)
    _write_json(metrics_path, metrics)
    loaded = load_order3_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=checkpoint_hash,
    )
    if loaded.metadata.training_stage != "bc":  # pragma: no cover - strict schema path.
        raise RuntimeError("Order3 BC checkpoint roundtrip stage mismatch")
    return Order3BCTrainingResult(
        output_root=str(target_root),
        checkpoint_path=str(checkpoint_path),
        loss_curve_path=str(loss_path),
        metrics_path=str(metrics_path),
        checkpoint_sha256=checkpoint_hash,
        metrics=metrics,
    )


def train_order3_pi_l_ppo(
    *,
    dataset_path: str | Path,
    parent_bc_checkpoint_path: str | Path,
    parent_bc_checkpoint_sha256: str,
    physical_model: PhysicalModel,
    training_config: Order3PiLTrainingConfig | None = None,
    config_path: str | Path = DEFAULT_ORDER3_TRAINING_CONFIG_PATH,
    output_root: str | Path | None = None,
    git_revision: str = "unknown",
    controller_contract_hash: str | None = None,
) -> Order3PPOTrainingResult:
    """Apply one PPO update to a checkpoint-matched fresh Isaac rollout.

    A production call deliberately consumes exactly one behavior checkpoint and
    one rollout generation.  A later update must collect a fresh dataset from
    the checkpoint returned by this call and invoke this API again.
    """

    if not _is_sha256(parent_bc_checkpoint_sha256):
        raise SchemaValidationError(
            "Order3 PPO expected parent BC hash must be a SHA-256 hex digest"
        )
    if training_config is None:
        train_cfg, _ = load_order3_pi_l_training_config(config_path)
    else:
        train_cfg = training_config
    if train_cfg.ppo.updates != 1:
        raise SchemaValidationError(
            "Order3 production PPO requires updates=1 per fresh Isaac rollout; "
            "the pipeline must recollect before the next update"
        )
    target_root = Path(
        output_root or (Path(train_cfg.artifact_root) / "training" / "ppo")
    )
    _reject_legacy_p4_3_path(target_root)
    require_non_empty(git_revision, "Order3 PPO training git_revision")
    device = _training_device(train_cfg.device)
    parent = load_order3_policy_checkpoint(
        parent_bc_checkpoint_path,
        device=device,
        expected_sha256=parent_bc_checkpoint_sha256,
    )
    if parent.metadata.training_stage not in {"bc", "ppo"}:
        raise SchemaValidationError(
            "Order3 PPO parent checkpoint must be a BC or prior PPO checkpoint"
        )
    context = _load_training_data_context(
        dataset_path=dataset_path,
        physical_model=physical_model,
        model_config=parent.config,
        controller_contract_hash=controller_contract_hash,
    )
    bc_records, ppo_records, ppo_behavior = _partition_stage_records(context.splits)
    if any(bc_records.values()):
        raise SchemaValidationError(
            "Order3 production PPO dataset must contain learned checkpoint rows only"
        )
    _require_complete_stage_splits(ppo_records, stage="PPO")
    if ppo_behavior is None:  # pragma: no cover - complete PPO splits imply provenance.
        raise RuntimeError("Order3 PPO failed to resolve behavior provenance")
    if ppo_behavior.checkpoint_hash != parent.sha256:
        raise SchemaValidationError(
            "Order3 PPO dataset behavior checkpoint hash does not match the immediate parent checkpoint"
        )
    _validate_parent_checkpoint_contract(parent.metadata, context)
    root_bc_checkpoint_hash = (
        parent.sha256
        if parent.metadata.training_stage == "bc"
        else parent.metadata.parent_bc_checkpoint_hash
    )
    if root_bc_checkpoint_hash is None:  # pragma: no cover - schema validates PPO parents.
        raise RuntimeError("Order3 PPO lineage has no root BC checkpoint")

    _seed_everything(train_cfg.seed + 1)
    model = parent.model.to(device)
    gae = compute_order3_gae(
        [item.record for item in ppo_records[DatasetSplit.TRAIN]],
        gamma=train_cfg.ppo.gamma,
        gae_lambda=train_cfg.ppo.gae_lambda,
    )
    loss_rows, reward_rows, gradient_max, rollout_selection = _train_ppo(
        model,
        ppo_records[DatasetSplit.TRAIN],
        gae=gae,
        config=train_cfg.ppo,
        seed=train_cfg.seed + 1,
        device=device,
    )
    evaluation = _evaluate_ppo(
        model,
        ppo_records,
        config=train_cfg.ppo,
        device=device,
    )
    metrics: dict[str, Any] = {
        "training_version": ORDER3_PI_L_TRAINING_VERSION,
        "stage": "ppo",
        "workflow": "production_staged",
        "dataset_hash": context.dataset_hash,
        "dataset_manifest_path": context.dataset.manifest_path,
        "training_config_hash": train_cfg.stable_hash(),
        "policy_config_hash": parent.config.stable_hash(),
        "parent_checkpoint_sha256": parent.sha256,
        "parent_checkpoint_training_stage": parent.metadata.training_stage,
        "parent_bc_checkpoint_sha256": root_bc_checkpoint_hash,
        "train_transition_count": len(ppo_records[DatasetSplit.TRAIN]),
        "validation_transition_count": len(ppo_records[DatasetSplit.VALIDATION]),
        "held_out_transition_count": len(ppo_records[DatasetSplit.HELD_OUT]),
        "behavior_policy_kind": "order3_checkpoint",
        "behavior_policy_version": ppo_behavior.policy_version,
        "behavior_checkpoint_hash": ppo_behavior.checkpoint_hash,
        "behavior_checkpoint_matches_immediate_parent": True,
        "behavior_checkpoint_matches_parent_bc": (
            parent.metadata.training_stage == "bc"
        ),
        "action_semantics": "learned_residual",
        "max_preclip_gradient_norm": gradient_max,
        "gradient_clip_norm": train_cfg.ppo.max_grad_norm,
        "gae_gamma": train_cfg.ppo.gamma,
        "gae_lambda": train_cfg.ppo.gae_lambda,
        "truncated_bootstrap_supported": True,
        "clipped_ppo": True,
        "critic_uses_privileged_disturbance": True,
        "actor_uses_privileged_disturbance": False,
        "recurrent_state_in_used": True,
        "recorded_recurrent_state_used_for_optimization": False,
        "recurrent_sequence_training": True,
        "fresh_online_rollout_update": True,
        "online_update_count": 1,
        "rollout_selection": rollout_selection,
        "held_out_used_for_optimization": False,
        "evaluation": evaluation,
        "all_losses_finite": _rows_are_finite(loss_rows),
        "p4_full_completion_claim": False,
    }
    target_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_root / "checkpoint.pt"
    metadata = _checkpoint_metadata(
        stage="ppo",
        model_config=parent.config,
        training_config=train_cfg,
        dataset_manifest=context.dataset.manifest,
        dataset_hash=context.dataset_hash,
        urdf_hash=context.urdf_hash,
        controller_contract_hash=context.controller_contract_hash,
        git_revision=git_revision,
        parent_bc_checkpoint_hash=root_bc_checkpoint_hash,
        stage_metrics=metrics,
        morphology_hashes_override=parent.metadata.metadata.get(
            "morphology_hashes"
        ),
    )
    checkpoint_hash = save_order3_policy_checkpoint(
        checkpoint_path,
        model=model,
        metadata=metadata,
    )
    metrics["checkpoint_sha256"] = checkpoint_hash
    loss_path = target_root / "loss_curve.csv"
    reward_path = target_root / "reward_curve.csv"
    metrics_path = target_root / "metrics.json"
    _write_csv(loss_path, loss_rows)
    _write_csv(reward_path, reward_rows)
    _write_json(metrics_path, metrics)
    loaded = load_order3_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=checkpoint_hash,
    )
    if loaded.metadata.training_stage != "ppo":  # pragma: no cover - strict schema path.
        raise RuntimeError("Order3 PPO checkpoint roundtrip stage mismatch")
    return Order3PPOTrainingResult(
        output_root=str(target_root),
        checkpoint_path=str(checkpoint_path),
        loss_curve_path=str(loss_path),
        reward_curve_path=str(reward_path),
        metrics_path=str(metrics_path),
        checkpoint_sha256=checkpoint_hash,
        metrics=metrics,
    )


def train_order3_pi_l(
    *,
    dataset_path: str | Path,
    physical_model: PhysicalModel,
    training_config: Order3PiLTrainingConfig | None = None,
    policy_config: Order3MorphologyConditionedPolicyConfig | None = None,
    config_path: str | Path = DEFAULT_ORDER3_TRAINING_CONFIG_PATH,
    output_root: str | Path | None = None,
    git_revision: str = "unknown",
    controller_contract_hash: str | None = None,
) -> Order3PiLTrainingResult:
    """BC-warm-start then offline PPO-update the Order-3 recurrent actor/critic."""

    train_cfg, model_cfg = _resolve_training_configs(
        training_config=training_config,
        policy_config=policy_config,
        config_path=config_path,
    )
    target_root = Path(output_root or (Path(train_cfg.artifact_root) / "training"))
    _reject_legacy_p4_3_path(target_root)
    require_non_empty(git_revision, "Order3 training git_revision")
    device = _training_device(train_cfg.device)

    _seed_everything(train_cfg.seed)
    model = MorphologyConditionedActorCritic(model_cfg).to(device)
    context = _load_training_data_context(
        dataset_path=dataset_path,
        physical_model=physical_model,
        model_config=model_cfg,
        controller_contract_hash=controller_contract_hash,
    )
    dataset = context.dataset
    dataset_hash = context.dataset_hash
    urdf_hash = context.urdf_hash
    contract_hash = context.controller_contract_hash
    split_records = context.splits
    bc_split_records, ppo_split_records, ppo_behavior = _partition_stage_records(
        split_records
    )
    _require_complete_stage_splits(bc_split_records, stage="BC")
    _require_complete_stage_splits(ppo_split_records, stage="PPO")
    if ppo_behavior is None:  # pragma: no cover - complete PPO splits imply provenance.
        raise RuntimeError("Order3 combined training failed to resolve PPO provenance")

    target_root.mkdir(parents=True, exist_ok=True)
    bc_dir = target_root / "bc"
    ppo_dir = target_root / "ppo"
    bc_dir.mkdir(parents=True, exist_ok=True)
    ppo_dir.mkdir(parents=True, exist_ok=True)

    initial_bc = _evaluate_bc(
        model,
        bc_split_records,
        device=device,
        value_loss_weight=train_cfg.bc.value_loss_weight,
    )
    bc_rows, bc_gradient_max = _train_bc(
        model,
        bc_split_records[DatasetSplit.TRAIN],
        config=train_cfg.bc,
        seed=train_cfg.seed,
        device=device,
    )
    final_bc = _evaluate_bc(
        model,
        bc_split_records,
        device=device,
        value_loss_weight=train_cfg.bc.value_loss_weight,
    )
    bc_metrics: dict[str, Any] = {
        "training_version": ORDER3_PI_L_TRAINING_VERSION,
        "stage": "bc",
        "dataset_hash": dataset_hash,
        "dataset_manifest_path": dataset.manifest_path,
        "training_config_hash": train_cfg.stable_hash(),
        "policy_config_hash": model_cfg.stable_hash(),
        "train_transition_count": len(bc_split_records[DatasetSplit.TRAIN]),
        "validation_transition_count": len(bc_split_records[DatasetSplit.VALIDATION]),
        "held_out_transition_count": len(bc_split_records[DatasetSplit.HELD_OUT]),
        "behavior_policy_kind": "deterministic_v2_teacher",
        "behavior_policy_version": ORDER3_FALLBACK_VERSION,
        "action_semantics": "reference_hold",
        "zero_teacher_action_count": sum(
            _is_zero_action(item.record.action)
            for item in bc_split_records[DatasetSplit.TRAIN]
        ),
        "reference_teacher_action_count": len(bc_split_records[DatasetSplit.TRAIN]),
        "initial": initial_bc,
        "final": final_bc,
        "max_preclip_gradient_norm": bc_gradient_max,
        "gradient_clip_norm": train_cfg.bc.max_grad_norm,
        "episode_sequence_training": True,
        "recurrent_state_in_used": True,
        "actor_uses_privileged_disturbance": False,
        "critic_uses_privileged_disturbance": True,
        "held_out_used_for_optimization": False,
        "p4_full_completion_claim": False,
    }
    bc_checkpoint_path = bc_dir / "checkpoint.pt"
    bc_metadata = _checkpoint_metadata(
        stage="bc",
        model_config=model_cfg,
        training_config=train_cfg,
        dataset_manifest=dataset.manifest,
        dataset_hash=dataset_hash,
        urdf_hash=urdf_hash,
        controller_contract_hash=contract_hash,
        git_revision=git_revision,
        parent_bc_checkpoint_hash=None,
        stage_metrics=bc_metrics,
    )
    bc_checkpoint_hash = save_order3_policy_checkpoint(
        bc_checkpoint_path,
        model=model,
        metadata=bc_metadata,
    )
    bc_metrics["checkpoint_sha256"] = bc_checkpoint_hash
    bc_loss_path = bc_dir / "loss_curve.csv"
    bc_metrics_path = bc_dir / "metrics.json"
    _write_csv(bc_loss_path, bc_rows)
    _write_json(bc_metrics_path, bc_metrics)

    gae = compute_order3_gae(
        [item.record for item in ppo_split_records[DatasetSplit.TRAIN]],
        gamma=train_cfg.ppo.gamma,
        gae_lambda=train_cfg.ppo.gae_lambda,
    )
    ppo_rows, reward_rows, ppo_gradient_max, rollout_selection = _train_ppo(
        model,
        ppo_split_records[DatasetSplit.TRAIN],
        gae=gae,
        config=train_cfg.ppo,
        seed=train_cfg.seed + 1,
        device=device,
    )
    ppo_evaluation = _evaluate_ppo(
        model,
        ppo_split_records,
        config=train_cfg.ppo,
        device=device,
    )
    ppo_metrics: dict[str, Any] = {
        "training_version": ORDER3_PI_L_TRAINING_VERSION,
        "stage": "ppo",
        "dataset_hash": dataset_hash,
        "dataset_manifest_path": dataset.manifest_path,
        "training_config_hash": train_cfg.stable_hash(),
        "policy_config_hash": model_cfg.stable_hash(),
        "parent_bc_checkpoint_sha256": bc_checkpoint_hash,
        "train_transition_count": len(ppo_split_records[DatasetSplit.TRAIN]),
        "validation_transition_count": len(ppo_split_records[DatasetSplit.VALIDATION]),
        "held_out_transition_count": len(ppo_split_records[DatasetSplit.HELD_OUT]),
        "behavior_policy_kind": "order3_checkpoint",
        "behavior_policy_version": ppo_behavior.policy_version,
        "behavior_checkpoint_hash": ppo_behavior.checkpoint_hash,
        "action_semantics": "learned_residual",
        "max_preclip_gradient_norm": ppo_gradient_max,
        "gradient_clip_norm": train_cfg.ppo.max_grad_norm,
        "gae_gamma": train_cfg.ppo.gamma,
        "gae_lambda": train_cfg.ppo.gae_lambda,
        "clipped_ppo": True,
        "critic_uses_privileged_disturbance": True,
        "actor_uses_privileged_disturbance": False,
        "recurrent_state_in_used": True,
        "held_out_used_for_optimization": False,
        "rollout_selection": rollout_selection,
        "evaluation": ppo_evaluation,
        "all_losses_finite": _rows_are_finite(ppo_rows),
        "p4_full_completion_claim": False,
    }
    ppo_checkpoint_path = ppo_dir / "checkpoint.pt"
    ppo_metadata = _checkpoint_metadata(
        stage="ppo",
        model_config=model_cfg,
        training_config=train_cfg,
        dataset_manifest=dataset.manifest,
        dataset_hash=dataset_hash,
        urdf_hash=urdf_hash,
        controller_contract_hash=contract_hash,
        git_revision=git_revision,
        parent_bc_checkpoint_hash=bc_checkpoint_hash,
        stage_metrics=ppo_metrics,
    )
    ppo_checkpoint_hash = save_order3_policy_checkpoint(
        ppo_checkpoint_path,
        model=model,
        metadata=ppo_metadata,
    )
    ppo_metrics["checkpoint_sha256"] = ppo_checkpoint_hash
    ppo_loss_path = ppo_dir / "loss_curve.csv"
    reward_curve_path = ppo_dir / "reward_curve.csv"
    ppo_metrics_path = ppo_dir / "metrics.json"
    _write_csv(ppo_loss_path, ppo_rows)
    _write_csv(reward_curve_path, reward_rows)
    _write_json(ppo_metrics_path, ppo_metrics)

    # Strict loader round trips are part of training completion, not a test-only check.
    loaded_bc = load_order3_policy_checkpoint(bc_checkpoint_path, device=device)
    loaded_ppo = load_order3_policy_checkpoint(ppo_checkpoint_path, device=device)
    if loaded_bc.metadata.training_stage != "bc" or loaded_ppo.metadata.training_stage != "ppo":
        raise RuntimeError("Order3 checkpoint roundtrip stage mismatch")
    summary = {
        "training_version": ORDER3_PI_L_TRAINING_VERSION,
        "policy_family": ORDER3_POLICY_FAMILY,
        "policy_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
        "output_mode": ORDER3_POLICY_OUTPUT_MODE,
        "dataset_manifest_path": dataset.manifest_path,
        "dataset_hash": dataset_hash,
        "bc_checkpoint_path": str(bc_checkpoint_path),
        "bc_checkpoint_sha256": bc_checkpoint_hash,
        "ppo_checkpoint_path": str(ppo_checkpoint_path),
        "ppo_checkpoint_sha256": ppo_checkpoint_hash,
        "bc_loss_decreased": (
            final_bc["train_actor_mse"] <= initial_bc["train_actor_mse"]
        ),
        "ppo_losses_finite": ppo_metrics["all_losses_finite"],
        "morphology_split_disjoint": True,
        "actor_uses_privileged_disturbance": False,
        "critic_uses_privileged_disturbance": True,
        "deterministic_fallback_version": ORDER3_FALLBACK_VERSION,
        "bc_behavior_policy_kind": "deterministic_v2_teacher",
        "bc_action_semantics": "reference_hold",
        "ppo_behavior_policy_kind": "order3_checkpoint",
        "ppo_behavior_policy_version": ppo_behavior.policy_version,
        "ppo_behavior_checkpoint_hash": ppo_behavior.checkpoint_hash,
        "ppo_action_semantics": "learned_residual",
        "legacy_p4_3_artifact_reused": False,
        "p4_full_completion_claim": False,
    }
    summary_path = target_root / "training_summary.json"
    _write_json(summary_path, summary)
    return Order3PiLTrainingResult(
        output_root=str(target_root),
        bc_checkpoint_path=str(bc_checkpoint_path),
        ppo_checkpoint_path=str(ppo_checkpoint_path),
        bc_loss_curve_path=str(bc_loss_path),
        ppo_loss_curve_path=str(ppo_loss_path),
        reward_curve_path=str(reward_curve_path),
        bc_metrics_path=str(bc_metrics_path),
        ppo_metrics_path=str(ppo_metrics_path),
        summary_path=str(summary_path),
        bc_checkpoint_sha256=bc_checkpoint_hash,
        ppo_checkpoint_sha256=ppo_checkpoint_hash,
        metrics={"bc": bc_metrics, "ppo": ppo_metrics, "summary": summary},
    )


def _prepare_transitions(
    transitions: Iterable[Order3PolicyTransition],
    physical_model: PhysicalModel,
) -> list[_PreparedTransition]:
    builder = RigidBodyControlModelBuilder()
    prepared: list[_PreparedTransition] = []
    for record in transitions:
        control_model = builder.build(
            record.runtime_observation.morphology_graph,
            physical_model,
            record.runtime_observation,
        )
        features = order3_actor_feature_vector(
            record.runtime_observation,
            control_model,
            target_pose_world=record.target_pose_world,
            target_twist=record.target_twist,
        )
        prepared.append(
            _PreparedTransition(
                record=record,
                actor_features=tuple(float(value) for value in features),
            )
        )
    return prepared


def _validate_training_splits(
    splits: dict[DatasetSplit, list[_PreparedTransition]],
    model_config: Order3MorphologyConditionedPolicyConfig,
) -> None:
    hash_sets: dict[DatasetSplit, set[str]] = {}
    for split in DatasetSplit:
        records = splits[split]
        if not records:
            raise SchemaValidationError(f"Order3 training split {split.value!r} is empty")
        hash_sets[split] = {item.record.structural_hash for item in records}
        for item in records:
            if len(item.record.recurrent_state_in) != model_config.recurrent_hidden_dim:
                raise SchemaValidationError(
                    "Order3 transition recurrent_state_in width does not match policy config"
                )
    ordered = list(DatasetSplit)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if hash_sets[left].intersection(hash_sets[right]):
                raise SchemaValidationError("Order3 training morphology split leakage detected")


def _partition_stage_records(
    splits: dict[DatasetSplit, list[_PreparedTransition]],
) -> tuple[
    dict[DatasetSplit, list[_PreparedTransition]],
    dict[DatasetSplit, list[_PreparedTransition]],
    _PPOBehaviorContract | None,
]:
    bc_records: dict[DatasetSplit, list[_PreparedTransition]] = {
        split: [] for split in DatasetSplit
    }
    ppo_records: dict[DatasetSplit, list[_PreparedTransition]] = {
        split: [] for split in DatasetSplit
    }
    episode_contracts: dict[str, tuple[str, str, str | None, str]] = {}
    ppo_contracts: set[tuple[str, str]] = set()
    for split in DatasetSplit:
        for item in splits[split]:
            record = item.record
            contract = (
                record.behavior_policy_kind,
                record.behavior_policy_version,
                record.behavior_checkpoint_hash,
                record.action_semantics,
            )
            previous = episode_contracts.setdefault(record.episode_id, contract)
            if previous != contract:
                raise SchemaValidationError(
                    "Order3 training episode crosses behavior-policy provenance boundary"
                )
            if record.behavior_policy_kind == "deterministic_v2_teacher":
                if record.behavior_policy_version != ORDER3_FALLBACK_VERSION:
                    raise SchemaValidationError(
                        "Order3 BC rows require the approved deterministic-v2 teacher version"
                    )
                if (
                    record.action_semantics != "reference_hold"
                    or record.behavior_checkpoint_hash is not None
                ):
                    raise SchemaValidationError(
                        "Order3 BC rows require deterministic_v2_teacher/reference_hold provenance"
                    )
                bc_records[split].append(item)
                continue
            if (
                record.behavior_policy_kind != "order3_checkpoint"
                or record.action_semantics != "learned_residual"
            ):
                raise SchemaValidationError(
                    "Order3 PPO rows require order3_checkpoint/learned_residual provenance"
                )
            if record.behavior_policy_version != ORDER3_CHECKPOINT_VERSION:
                raise SchemaValidationError(
                    "Order3 PPO behavior policy version must match the Order3 checkpoint version"
                )
            checkpoint_hash = record.behavior_checkpoint_hash or ""
            if not _is_sha256(checkpoint_hash):
                raise SchemaValidationError(
                    "Order3 PPO behavior checkpoint hash must be a SHA-256 hex digest"
                )
            ppo_contracts.add((record.behavior_policy_version, checkpoint_hash))
            ppo_records[split].append(item)

    for stage, stage_records in (("BC", bc_records), ("PPO", ppo_records)):
        if any(stage_records.values()):
            _validate_stage_episode_boundaries(stage_records, stage=stage)
    if len(ppo_contracts) > 1:
        raise SchemaValidationError(
            "Order3 PPO rows must share one behavior policy version/checkpoint hash"
        )
    ppo_behavior = None
    if ppo_contracts:
        version, checkpoint_hash = next(iter(ppo_contracts))
        ppo_behavior = _PPOBehaviorContract(
            policy_version=version,
            checkpoint_hash=checkpoint_hash,
        )
    return (
        bc_records,
        ppo_records,
        ppo_behavior,
    )


def _require_complete_stage_splits(
    splits: dict[DatasetSplit, list[_PreparedTransition]],
    *,
    stage: str,
) -> None:
    for split in DatasetSplit:
        if not splits[split]:
            raise SchemaValidationError(f"Order3 {stage} split {split.value!r} is empty")


def _validate_stage_episode_boundaries(
    splits: dict[DatasetSplit, list[_PreparedTransition]],
    *,
    stage: str,
) -> None:
    by_episode: dict[str, list[Order3PolicyTransition]] = {}
    for split in DatasetSplit:
        for item in splits[split]:
            by_episode.setdefault(item.record.episode_id, []).append(item.record)
    for episode_id, records in by_episode.items():
        ordered = sorted(records, key=lambda record: record.step_index)
        boundary_positions = [
            position
            for position, record in enumerate(ordered)
            if record.terminal or record.truncated
        ]
        if boundary_positions != [len(ordered) - 1]:
            raise SchemaValidationError(
                f"Order3 {stage} episode {episode_id!r} requires exactly one final "
                "terminal/truncated boundary"
            )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _training_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Order3 training requested CUDA but CUDA is unavailable")
    return device


def _validate_parent_checkpoint_contract(
    metadata: Order3PolicyCheckpointMetadata,
    context: _TrainingDataContext,
) -> None:
    expected = {
        "pool_hash": context.dataset.manifest.pool_hash,
        "physical_model_hash": context.dataset.manifest.physical_model_hash,
        "urdf_hash": context.urdf_hash,
        "controller_contract_hash": context.controller_contract_hash,
    }
    mismatches = [
        name for name, value in expected.items() if getattr(metadata, name) != value
    ]
    if mismatches:
        raise SchemaValidationError(
            "Order3 PPO parent checkpoint contract mismatch: " + ", ".join(mismatches)
        )


def _train_bc(
    model: MorphologyConditionedActorCritic,
    records: list[_PreparedTransition],
    *,
    config: Order3BCTrainingConfig,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    chunks = _episode_chunks(records, config.sequence_length)
    rows: list[dict[str, float]] = []
    maximum_gradient_norm = 0.0
    for epoch in range(config.epochs):
        model.train()
        rng = random.Random(seed + epoch * 104729)
        ordered_chunks = list(chunks)
        rng.shuffle(ordered_chunks)
        actor_sum = 0.0
        value_sum = 0.0
        total_sum = 0.0
        sample_count = 0
        for start in range(0, len(ordered_chunks), config.batch_size):
            batch_chunks = ordered_chunks[start : start + config.batch_size]
            losses: list[torch.Tensor] = []
            actor_losses: list[torch.Tensor] = []
            value_losses: list[torch.Tensor] = []
            for chunk in batch_chunks:
                hidden = _recurrent_state_tensor(chunk[0].record, device=device)
                effective_burn = min(config.burn_in_steps, max(0, len(chunk) - 1))
                for step_index, item in enumerate(chunk):
                    step = _model_step(
                        model,
                        [item],
                        device=device,
                        recurrent_state=hidden,
                        use_recorded_recurrent_state=False,
                        supplied_action=True,
                    )
                    hidden = step.recurrent_state
                    if step_index < effective_burn:
                        continue
                    teacher = torch.tensor(
                        [item.record.action], dtype=step.action_mean.dtype, device=device
                    )
                    actor_loss = nn.functional.mse_loss(step.action_mean, teacher)
                    value_target = torch.tensor(
                        [item.record.old_value], dtype=step.value.dtype, device=device
                    )
                    value_loss = nn.functional.mse_loss(step.value, value_target)
                    losses.append(actor_loss + config.value_loss_weight * value_loss)
                    actor_losses.append(actor_loss)
                    value_losses.append(value_loss)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            gradient_value = float(gradient_norm.detach().cpu().item())
            if not math.isfinite(gradient_value):
                raise RuntimeError("Order3 BC produced a non-finite gradient norm")
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)
            optimizer.step()
            count = len(losses)
            actor_sum += float(torch.stack(actor_losses).sum().detach().cpu().item())
            value_sum += float(torch.stack(value_losses).sum().detach().cpu().item())
            total_sum += float(torch.stack(losses).sum().detach().cpu().item())
            sample_count += count
        if sample_count == 0:
            raise RuntimeError("Order3 BC found no post-burn-in training steps")
        rows.append(
            {
                "epoch": float(epoch + 1),
                "actor_mse": actor_sum / sample_count,
                "value_mse": value_sum / sample_count,
                "total_loss": total_sum / sample_count,
            }
        )
    return rows, maximum_gradient_norm


def _evaluate_bc(
    model: MorphologyConditionedActorCritic,
    split_records: dict[DatasetSplit, list[_PreparedTransition]],
    *,
    device: torch.device,
    value_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    with torch.no_grad():
        for split in DatasetSplit:
            actor_losses: list[float] = []
            value_losses: list[float] = []
            for chunk in _episode_chunks(split_records[split], sequence_length=10**9):
                hidden = _recurrent_state_tensor(chunk[0].record, device=device)
                for item in chunk:
                    step = _model_step(
                        model,
                        [item],
                        device=device,
                        recurrent_state=hidden,
                        use_recorded_recurrent_state=False,
                        supplied_action=True,
                    )
                    hidden = step.recurrent_state
                    teacher = torch.tensor(
                        [item.record.action], dtype=step.action_mean.dtype, device=device
                    )
                    actor_losses.append(
                        float(nn.functional.mse_loss(step.action_mean, teacher).cpu().item())
                    )
                    value_target = torch.tensor(
                        [item.record.old_value], dtype=step.value.dtype, device=device
                    )
                    value_losses.append(
                        float(nn.functional.mse_loss(step.value, value_target).cpu().item())
                    )
            actor = _mean(actor_losses)
            value = _mean(value_losses)
            metrics[f"{split.value}_actor_mse"] = actor
            metrics[f"{split.value}_value_mse"] = value
            metrics[f"{split.value}_total_loss"] = actor + value_loss_weight * value
    return metrics


def _train_ppo(
    model: MorphologyConditionedActorCritic,
    records: list[_PreparedTransition],
    *,
    gae: Order3GAEResult,
    config: Order3PPOTrainingConfig,
    seed: int,
    device: torch.device,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    float,
    dict[str, Any],
]:
    if len(records) != len(gae.advantages) or len(records) != len(gae.returns):
        raise ValueError("Order3 PPO GAE tensors must align with train transitions")
    raw_advantages = torch.tensor(gae.advantages, dtype=torch.float32)
    returns = torch.tensor(gae.returns, dtype=torch.float32)
    if config.updates != 1:
        raise SchemaValidationError(
            "Order3 PPO optimizer consumes one fresh rollout generation per call"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rows: list[dict[str, float]] = []
    reward_rows: list[dict[str, float]] = []
    maximum_gradient_norm = 0.0
    indexed_episodes = _indexed_ppo_episodes(records)
    selected_episodes = _select_complete_ppo_episodes(
        indexed_episodes,
        rollout_step_budget=config.rollout_steps_per_update,
        seed=seed,
    )
    if not selected_episodes:
        raise RuntimeError("Order3 PPO found no complete episode in the rollout batch")
    selected_indices = [
        index for episode in selected_episodes for index, _ in episode
    ]
    selected_count = len(selected_indices)
    selected_advantages = raw_advantages[selected_indices]
    advantage_mean = selected_advantages.mean()
    advantage_std = torch.clamp(
        selected_advantages.std(unbiased=False), min=1.0e-6
    )
    advantages = (raw_advantages - advantage_mean) / advantage_std
    selected_records = [item.record for episode in selected_episodes for _, item in episode]
    selected_module_counts = sorted(
        {
            len(episode[0][1].record.runtime_observation.morphology_graph.modules)
            for episode in selected_episodes
        }
    )
    selected_episode_ids = [episode[0][1].record.episode_id for episode in selected_episodes]
    selected_structural_hashes = sorted(
        {episode[0][1].record.structural_hash for episode in selected_episodes}
    )
    selection_metadata: dict[str, Any] = {
        "selection_strategy": "seeded_module_count_round_robin_complete_episodes_v1",
        "selection_seed": seed,
        "rollout_step_budget": config.rollout_steps_per_update,
        "selected_transition_count": selected_count,
        "selected_episode_count": len(selected_episodes),
        "selected_module_counts": selected_module_counts,
        "selected_structural_hashes": selected_structural_hashes,
        "selected_episode_ids": selected_episode_ids,
        "selection_hash": stable_hash(
            {
                "seed": seed,
                "episode_ids": selected_episode_ids,
                "structural_hashes": selected_structural_hashes,
                "transition_count": selected_count,
            }
        ),
        "advantage_normalization_scope": "selected_transitions_only",
        "discarded_transition_count": len(records) - selected_count,
    }
    episode_return = _mean_episode_return(selected_records)
    model.train()
    actor_values: list[float] = []
    critic_values: list[float] = []
    entropy_values: list[float] = []
    total_values: list[float] = []
    clip_values: list[float] = []
    for epoch in range(config.epochs_per_update):
        rng = random.Random(seed + epoch * 104729)
        ordered = list(selected_episodes)
        rng.shuffle(ordered)
        batch: list[list[tuple[int, _PreparedTransition]]] = []
        batch_steps = 0
        for episode_index, episode in enumerate(ordered):
            batch.append(episode)
            batch_steps += len(episode)
            is_last = episode_index == len(ordered) - 1
            if batch_steps < config.minibatch_size and not is_last:
                continue
            losses: list[torch.Tensor] = []
            for indexed_episode in batch:
                hidden = model.initial_state(1, device=device)
                effective_burn = min(
                    config.recurrent_burn_in_steps,
                    max(0, len(indexed_episode) - 1),
                )
                for sequence_index, (record_index, item) in enumerate(indexed_episode):
                    step = _model_step(
                        model,
                        [item],
                        device=device,
                        recurrent_state=hidden,
                        use_recorded_recurrent_state=False,
                        supplied_action=True,
                    )
                    hidden = step.recurrent_state
                    if sequence_index < effective_burn:
                        continue
                    old_log_prob = torch.tensor(
                        [item.record.old_log_prob],
                        dtype=step.log_prob.dtype,
                        device=device,
                    )
                    advantage = advantages[record_index].to(
                        device=device, dtype=step.log_prob.dtype
                    )
                    return_target = returns[record_index].to(
                        device=device, dtype=step.value.dtype
                    )
                    log_ratio = torch.clamp(
                        step.log_prob - old_log_prob, -20.0, 20.0
                    )
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - config.clip_ratio,
                        1.0 + config.clip_ratio,
                    )
                    actor_loss = -torch.minimum(
                        ratio * advantage,
                        clipped_ratio * advantage,
                    ).mean()
                    critic_loss = nn.functional.mse_loss(
                        step.value,
                        return_target.reshape_as(step.value),
                    )
                    entropy = step.entropy.mean()
                    total_loss = (
                        actor_loss
                        + config.value_loss_weight * critic_loss
                        - config.entropy_weight * entropy
                    )
                    if not bool(torch.isfinite(total_loss).item()):
                        raise RuntimeError("Order3 PPO produced a non-finite loss")
                    losses.append(total_loss)
                    actor_values.append(float(actor_loss.detach().cpu().item()))
                    critic_values.append(float(critic_loss.detach().cpu().item()))
                    entropy_values.append(float(entropy.detach().cpu().item()))
                    total_values.append(float(total_loss.detach().cpu().item()))
                    clip_values.append(
                        float(
                            (
                                (ratio < 1.0 - config.clip_ratio)
                                | (ratio > 1.0 + config.clip_ratio)
                            )
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                            .item()
                        )
                    )
            if not losses:
                raise RuntimeError("Order3 recurrent PPO minibatch has no post-burn-in steps")
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            gradient_value = float(gradient_norm.detach().cpu().item())
            if not math.isfinite(gradient_value):
                raise RuntimeError("Order3 PPO produced a non-finite gradient norm")
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_value)
            optimizer.step()
            batch = []
            batch_steps = 0
    row = {
        "update": 1.0,
        "actor_loss": _mean(actor_values),
        "critic_loss": _mean(critic_values),
        "entropy": _mean(entropy_values),
        "total_loss": _mean(total_values),
        "clip_fraction": _mean(clip_values),
        "fresh_rollout_transition_count": float(selected_count),
    }
    if not all(math.isfinite(value) for value in row.values()):
        raise RuntimeError("Order3 PPO update summary is non-finite")
    rows.append(row)
    reward_rows.append(
        {
            "update": 1.0,
            "fresh_online_mean_train_episode_return": episode_return,
            "fresh_online_rollout_consumed": 1.0,
        }
    )
    return rows, reward_rows, maximum_gradient_norm, selection_metadata


def _indexed_ppo_episodes(
    records: list[_PreparedTransition],
) -> list[list[tuple[int, _PreparedTransition]]]:
    by_episode: dict[str, list[tuple[int, _PreparedTransition]]] = {}
    for index, item in enumerate(records):
        by_episode.setdefault(item.record.episode_id, []).append((index, item))
    episodes: list[list[tuple[int, _PreparedTransition]]] = []
    for episode_id in sorted(by_episode):
        episode = sorted(
            by_episode[episode_id],
            key=lambda indexed: indexed[1].record.step_index,
        )
        if not (episode[-1][1].record.terminal or episode[-1][1].record.truncated):
            raise SchemaValidationError(
                f"Order3 PPO episode {episode_id!r} has no final boundary"
            )
        episodes.append(episode)
    return episodes


def _select_complete_ppo_episodes(
    episodes: list[list[tuple[int, _PreparedTransition]]],
    *,
    rollout_step_budget: int,
    seed: int,
) -> list[list[tuple[int, _PreparedTransition]]]:
    """Select a seeded, module-balanced set of complete recurrent episodes."""

    if rollout_step_budget <= 0:
        raise ValueError("Order3 PPO rollout_step_budget must be positive")
    buckets: dict[int, list[list[tuple[int, _PreparedTransition]]]] = {}
    for episode in episodes:
        if not episode:
            continue
        module_count = len(
            episode[0][1].record.runtime_observation.morphology_graph.modules
        )
        buckets.setdefault(module_count, []).append(episode)
    if not buckets:
        return []
    rng = random.Random(seed)
    for module_count in sorted(buckets):
        rng.shuffle(buckets[module_count])
    module_order = sorted(buckets)
    offset = seed % len(module_order)
    module_order = module_order[offset:] + module_order[:offset]
    selected: list[list[tuple[int, _PreparedTransition]]] = []
    selected_count = 0
    while True:
        added = False
        for module_count in module_order:
            if not buckets[module_count]:
                continue
            episode = buckets[module_count].pop()
            if selected and selected_count + len(episode) > rollout_step_budget:
                continue
            selected.append(episode)
            selected_count += len(episode)
            added = True
            if selected_count >= rollout_step_budget:
                return selected
        if not added:
            return selected


def _evaluate_ppo(
    model: MorphologyConditionedActorCritic,
    split_records: dict[DatasetSplit, list[_PreparedTransition]],
    *,
    config: Order3PPOTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    with torch.no_grad():
        for split in DatasetSplit:
            records = split_records[split]
            gae = compute_order3_gae(
                [item.record for item in records],
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            values: list[float] = []
            log_probs: list[float] = []
            entropies: list[float] = []
            action_mse: list[float] = []
            indexed_values: dict[int, float] = {}
            indexed_log_probs: dict[int, float] = {}
            indexed_entropies: dict[int, float] = {}
            indexed_action_mse: dict[int, float] = {}
            for episode in _indexed_ppo_episodes(records):
                hidden = model.initial_state(1, device=device)
                for index, item in episode:
                    step = _model_step(
                        model,
                        [item],
                        device=device,
                        recurrent_state=hidden,
                        use_recorded_recurrent_state=False,
                        supplied_action=True,
                    )
                    hidden = step.recurrent_state
                    teacher = torch.tensor(
                        [item.record.action],
                        dtype=step.action_mean.dtype,
                        device=device,
                    )
                    indexed_values[index] = float(step.value[0].cpu().item())
                    indexed_log_probs[index] = float(step.log_prob[0].cpu().item())
                    indexed_entropies[index] = float(step.entropy[0].cpu().item())
                    indexed_action_mse[index] = float(
                        torch.mean(torch.square(step.action_mean - teacher)).cpu().item()
                    )
            for index in range(len(records)):
                values.append(indexed_values[index])
                log_probs.append(indexed_log_probs[index])
                entropies.append(indexed_entropies[index])
                action_mse.append(indexed_action_mse[index])
            value_mse = _mean(
                [(value - target) ** 2 for value, target in zip(values, gae.returns)]
            )
            metrics[f"{split.value}_action_mse"] = _mean(action_mse)
            metrics[f"{split.value}_value_return_mse"] = value_mse
            metrics[f"{split.value}_mean_log_prob"] = _mean(log_probs)
            metrics[f"{split.value}_mean_entropy"] = _mean(entropies)
            metrics[f"{split.value}_mean_episode_return"] = _mean_episode_return(
                [item.record for item in records]
            )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Order3 PPO evaluation produced non-finite metrics")
    return metrics


def _model_step(
    model: MorphologyConditionedActorCritic,
    items: list[_PreparedTransition],
    *,
    device: torch.device,
    recurrent_state: torch.Tensor | None,
    use_recorded_recurrent_state: bool,
    supplied_action: bool,
):
    if use_recorded_recurrent_state:
        hidden = torch.tensor(
            [item.record.recurrent_state_in for item in items],
            dtype=torch.float32,
            device=device,
        )
    elif recurrent_state is not None:
        hidden = recurrent_state
    else:
        raise ValueError("Order3 model step requires recurrent state input")
    return model.step(
        [item.record.runtime_observation.morphology_graph for item in items],
        [item.record.runtime_observation for item in items],
        torch.tensor(
            [item.actor_features for item in items],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [item.record.previous_action for item in items],
            dtype=torch.float32,
            device=device,
        ),
        hidden,
        privileged_disturbance_body=torch.tensor(
            [item.record.privileged_disturbance_body for item in items],
            dtype=torch.float32,
            device=device,
        ),
        action=(
            torch.tensor(
                [item.record.action for item in items],
                dtype=torch.float32,
                device=device,
            )
            if supplied_action
            else None
        ),
    )


def _episode_chunks(
    records: list[_PreparedTransition],
    sequence_length: int,
) -> list[list[_PreparedTransition]]:
    by_episode: dict[str, list[_PreparedTransition]] = {}
    for item in records:
        by_episode.setdefault(item.record.episode_id, []).append(item)
    chunks: list[list[_PreparedTransition]] = []
    for episode_id in sorted(by_episode):
        episode = sorted(by_episode[episode_id], key=lambda item: item.record.step_index)
        for start in range(0, len(episode), sequence_length):
            chunks.append(episode[start : start + sequence_length])
    return chunks


def _recurrent_state_tensor(
    record: Order3PolicyTransition,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor([record.recurrent_state_in], dtype=torch.float32, device=device)


def _checkpoint_metadata(
    *,
    stage: str,
    model_config: Order3MorphologyConditionedPolicyConfig,
    training_config: Order3PiLTrainingConfig,
    dataset_manifest,
    dataset_hash: str,
    urdf_hash: str,
    controller_contract_hash: str,
    git_revision: str,
    parent_bc_checkpoint_hash: str | None,
    stage_metrics: dict[str, Any],
    morphology_hashes_override: Any | None = None,
) -> Order3PolicyCheckpointMetadata:
    fallback_config = BaselineLowLevelPolicyConfig(
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL
    )
    return Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        training_stage=stage,
        action_names=list(ORDER3_ACTION_NAMES),
        actor_feature_schema_hash=order3_actor_feature_schema_hash(),
        graph_feature_schema_hash=order3_graph_feature_schema_hash(),
        config_hash=model_config.stable_hash(),
        pool_hash=dataset_manifest.pool_hash,
        dataset_hash=dataset_hash,
        physical_model_hash=dataset_manifest.physical_model_hash,
        urdf_hash=urdf_hash,
        controller_contract_hash=controller_contract_hash,
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash=stable_hash(fallback_config),
        seed=training_config.seed,
        git_revision=git_revision,
        actor_uses_privileged_wrench=False,
        outputs_contact_wrench=False,
        outputs_internal_wrench=False,
        outputs_vectoring_joint_targets=False,
        parent_bc_checkpoint_hash=parent_bc_checkpoint_hash,
        metadata={
            "training_version": ORDER3_PI_L_TRAINING_VERSION,
            "output_mode": ORDER3_POLICY_OUTPUT_MODE,
            "algorithm": (
                "episode_sequence_behavior_cloning"
                if stage == "bc"
                else "fresh_online_recurrent_clipped_ppo_single_update_with_gae"
            ),
            "training_config_hash": training_config.stable_hash(),
            "dataset_version": dataset_manifest.dataset_version,
            "morphology_hashes": (
                morphology_hashes_override
                if morphology_hashes_override is not None
                else dataset_manifest.morphology_hashes
            ),
            "actor_privileged_inputs": [],
            "critic_privileged_inputs": ["privileged_disturbance_body"],
            "recurrent_state_source": "Order3PolicyTransition.recurrent_state_in",
            "behavior_policy_kind": stage_metrics["behavior_policy_kind"],
            "behavior_policy_version": stage_metrics["behavior_policy_version"],
            "behavior_checkpoint_hash": stage_metrics.get("behavior_checkpoint_hash"),
            "immediate_parent_checkpoint_hash": stage_metrics.get(
                "parent_checkpoint_sha256"
            ),
            "action_semantics": stage_metrics["action_semantics"],
            "zero_and_reference_teacher_actions_included": True,
            "held_out_used_for_optimization": False,
            "stage_metrics_hash": stable_hash(stage_metrics),
            "legacy_p4_3_artifact_reused": False,
            "p4_full_completion_claim": False,
        },
    )


def _mean_episode_return(records: Sequence[Order3PolicyTransition]) -> float:
    by_episode: dict[str, float] = {}
    for record in records:
        by_episode[record.episode_id] = by_episode.get(record.episode_id, 0.0) + float(
            record.reward
        )
    return _mean(list(by_episode.values()))


def _is_zero_action(action: Sequence[float]) -> bool:
    return all(abs(float(value)) <= 1.0e-12 for value in action)


def _rows_are_finite(rows: Sequence[dict[str, float]]) -> bool:
    return bool(rows) and all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Order3 metric aggregation requires at least one value")
    return sum(float(value) for value in values) / len(values)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"))


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"Order3 training CSV {path.name!r} requires rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, output.getvalue().encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _reject_legacy_p4_3_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(_LEGACY_P4_3_ROOT)
    except ValueError:
        return
    raise SchemaValidationError(
        "Order3 v2 training must not read from or write into legacy artifacts/p4_3"
    )


__all__ = [
    "DEFAULT_ORDER3_TRAINING_CONFIG_PATH",
    "DEFAULT_ORDER3_TRAINING_ROOT",
    "ORDER3_PI_L_TRAINING_VERSION",
    "Order3BCTrainingConfig",
    "Order3BCTrainingResult",
    "Order3GAEResult",
    "Order3PPOTrainingConfig",
    "Order3PPOTrainingResult",
    "Order3PiLTrainingConfig",
    "Order3PiLTrainingResult",
    "compute_order3_gae",
    "load_order3_pi_l_training_config",
    "train_order3_pi_l",
    "train_order3_pi_l_bc",
    "train_order3_pi_l_ppo",
]
