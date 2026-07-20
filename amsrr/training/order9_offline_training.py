from __future__ import annotations

"""Production offline behavior-cloning runner for Order 9 policy families.

The runner intentionally stops at a hash-bound learned checkpoint.  Curriculum
promotion remains a separate online/full-mesh evaluation decision owned by
``order9_pipeline``.
"""

import csv
import io
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from amsrr.policies.design_candidate_generator import DesignCandidateStep
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.morphology_conditioned_low_level_policy import (
    load_order3_policy_checkpoint,
)
from amsrr.policies.order9_design_grammar import (
    Order9DesignGrammar,
    Order9DesignTeacherStep,
)
from amsrr.policies.order9_design_policy import (
    Order9AutoregressiveDesignPolicy,
    Order9DesignPolicyConfig,
)
from amsrr.policies.order9_high_level_policy import (
    Order9AutoregressiveHighLevelPolicy,
    Order9HighLevelPolicyConfig,
)
from amsrr.policies.order9_low_level_policy import (
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import (
    DatasetSplit,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    SequentialDesignTrajectoryRecord,
)
from amsrr.schemas.order9 import (
    ORDER9_POLICY_CHECKPOINT_VERSION,
    Order9PolicyCheckpointMetadata,
    Order9PolicyFamily,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    order9_model_config_dict,
    order9_policy_identity,
    order9_state_dict_hash,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import (
    Order9BCOptimizationConfig,
    Order9CurriculumStage,
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
)
from amsrr.training.order9_dataset import (
    Order9DatasetBundle,
    load_order9_dataset,
    validate_order9_dataset_for_stage,
)
from amsrr.training.order9_pi_d_learning import (
    compute_order9_pi_d_behavior_cloning_loss,
)
from amsrr.training.order9_pi_h_learning import (
    Order9PiHLossWeights,
    compute_order9_pi_h_behavior_cloning_loss,
)
from amsrr.training.order9_pi_l_learning import (
    compute_order9_pi_l_behavior_cloning_loss,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_OFFLINE_TRAINING_VERSION = "order9_offline_bc_trainer_v1"


@dataclass
class Order9OfflineTrainingResult(SchemaBase):
    training_version: str
    stage_id: str
    policy_family: Order9PolicyFamily
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    metrics_path: str
    metrics_sha256: str
    loss_curve_path: str
    loss_curve_sha256: str
    epoch_count: int
    training_record_count: int
    validation_record_count: int
    best_validation_loss: float
    final_training_loss: float
    parameter_count: int
    random_seed: int
    parent_checkpoint_sha256: str | None = None
    source_order3_checkpoint_sha256: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.training_version != ORDER9_OFFLINE_TRAINING_VERSION:
            raise SchemaValidationError("Order9 offline training version mismatch")
        for name in (
            "stage_id",
            "dataset_manifest_path",
            "checkpoint_path",
            "metrics_path",
            "loss_curve_path",
        ):
            require_non_empty(str(getattr(self, name)), f"Order9OfflineTrainingResult.{name}")
        for name in (
            "dataset_manifest_sha256",
            "checkpoint_sha256",
            "metrics_sha256",
            "loss_curve_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        for name in ("parent_checkpoint_sha256", "source_order3_checkpoint_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        if min(
            self.epoch_count,
            self.training_record_count,
            self.validation_record_count,
            self.parameter_count,
        ) < 1:
            raise SchemaValidationError("Order9 offline training counts must be positive")
        if self.random_seed < 0:
            raise SchemaValidationError("Order9 offline training seed must be non-negative")
        for value in (
            self.best_validation_loss,
            self.final_training_loss,
            *self.metrics.values(),
        ):
            if not math.isfinite(float(value)):
                raise SchemaValidationError("Order9 offline training metrics must be finite")


@dataclass(frozen=True)
class _PiLWindow:
    burn_in: tuple[LowLevelControlRecord, ...]
    training: tuple[LowLevelControlRecord, ...]


def train_order9_behavior_cloning(
    config: Order9LearningConfig,
    *,
    stage_id: str,
    dataset_manifest_path: str | Path,
    physical_model: PhysicalModel,
    output_dir: str | Path,
    git_revision: str,
    device: str | torch.device | None = None,
    parent_checkpoint_path: str | Path | None = None,
    source_order3_checkpoint_path: str | Path | None = None,
    model_config: object | None = None,
    additional_input_artifact_paths: Mapping[str, str | Path] | None = None,
) -> Order9OfflineTrainingResult:
    """Train one C1/C4/C5/C7 stage and emit a strict policy checkpoint."""

    config.validate()
    stage = order9_stage_by_id(config, stage_id)
    if stage.learning_mode != Order9LearningMode.BEHAVIOR_CLONING:
        raise SchemaValidationError("Order9 offline trainer accepts BC stages only")
    require_non_empty(git_revision, "git_revision")
    resolved_device = _resolve_device(device or config.production_runtime.device)
    seed = config.production_runtime.seed + stage.stage_index
    _seed_everything(seed)

    bundle = load_order9_dataset(dataset_manifest_path)
    validation = validate_order9_dataset_for_stage(bundle, stage)
    if not validation.valid:
        raise SchemaValidationError(
            "Order9 BC dataset failed replay contract: " + ",".join(validation.failures)
        )
    model, family, parent_sha, order3_sha = _build_bc_model(
        config,
        stage,
        physical_model=physical_model,
        device=resolved_device,
        parent_checkpoint_path=parent_checkpoint_path,
        source_order3_checkpoint_path=source_order3_checkpoint_path,
        model_config=model_config,
    )
    optimization = _bc_optimization(config, stage)
    optimizer = torch.optim.Adam(model.parameters(), lr=optimization.learning_rate)

    if family == Order9PolicyFamily.PI_L:
        train_records = _split_records(bundle.low_level_records, DatasetSplit.TRAIN)
        validation_records = _split_records(
            bundle.low_level_records, DatasetSplit.VALIDATION
        )
        rows = _train_pi_l(
            model,
            train_records,
            validation_records,
            physical_model=physical_model,
            optimizer=optimizer,
            optimization=optimization,
            seed=seed,
            gamma=config.optimization.pi_l_ppo.gamma,
        )
    elif family == Order9PolicyFamily.PI_H:
        train_records = _split_records(bundle.trajectory_records, DatasetSplit.TRAIN)
        validation_records = _split_records(
            bundle.trajectory_records, DatasetSplit.VALIDATION
        )
        rows = _train_pi_h(
            model,
            train_records,
            validation_records,
            optimizer=optimizer,
            optimization=optimization,
            seed=seed,
            assignment_only=(
                stage.learning_target == Order9LearningTarget.PI_H_ASSIGNMENT
            ),
        )
    elif family == Order9PolicyFamily.PI_D:
        train_records = _split_records(
            bundle.sequential_design_records, DatasetSplit.TRAIN
        )
        validation_records = _split_records(
            bundle.sequential_design_records, DatasetSplit.VALIDATION
        )
        rows = _train_pi_d(
            model,
            train_records,
            validation_records,
            physical_model=physical_model,
            optimizer=optimizer,
            optimization=optimization,
            seed=seed,
        )
    else:  # pragma: no cover - model dispatch is exhaustive.
        raise AssertionError(family)

    best_index = min(range(len(rows)), key=lambda index: rows[index]["validation_total"])
    # Each family trainer restores its best state before returning.
    best_validation = float(rows[best_index]["validation_total"])
    final_training = float(rows[-1]["training_total"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    curve_path = output / "loss_curve.csv"
    _write_loss_curve(curve_path, rows)

    input_hashes = order9_checkpoint_input_hashes(
        bundle,
        parent_checkpoint_path=parent_checkpoint_path,
        source_order3_checkpoint_path=source_order3_checkpoint_path,
        additional_paths=additional_input_artifact_paths or {},
    )
    summary_metrics = {
        "best_validation_loss": best_validation,
        "final_training_loss": final_training,
        "best_epoch": float(best_index),
        "training_record_count": float(len(train_records)),
        "validation_record_count": float(len(validation_records)),
    }
    metadata = build_order9_checkpoint_metadata(
        model,
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        physical_model_hash=physical_model.stable_hash(),
        git_revision=git_revision,
        random_seed=seed,
        input_artifact_hashes=input_hashes,
        parent_checkpoint_sha256=parent_sha,
        source_order3_checkpoint_sha256=order3_sha,
        metrics=summary_metrics,
        trainer_version=ORDER9_OFFLINE_TRAINING_VERSION,
    )
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = save_order9_policy_checkpoint(
        checkpoint_path, model=model, metadata=metadata
    )
    metrics_path = output / "training_metrics.json"
    metrics_payload = {
        "training_version": ORDER9_OFFLINE_TRAINING_VERSION,
        "stage_id": stage.stage_id,
        "stage_index": stage.stage_index,
        "policy_family": family.value,
        "schedule_hash": order9_schedule_hash(config),
        "dataset_manifest_path": bundle.manifest_path,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "physical_model_hash": physical_model.stable_hash(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "loss_curve_path": str(curve_path),
        "loss_curve_sha256": hash_file(curve_path),
        "optimization": optimization.to_dict(),
        "metrics": summary_metrics,
        "epochs": rows,
        "promotion_evaluation_completed": False,
    }
    _atomic_write_text(metrics_path, _json_text(metrics_payload))
    result_path = output / "training_result.json"
    # Result hashes cannot include the result itself; they bind every executable
    # output that promotion/evaluation consumes.
    result = Order9OfflineTrainingResult(
        training_version=ORDER9_OFFLINE_TRAINING_VERSION,
        stage_id=stage.stage_id,
        policy_family=family,
        dataset_manifest_path=bundle.manifest_path,
        dataset_manifest_sha256=bundle.manifest_sha256,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        metrics_path=str(metrics_path),
        metrics_sha256=hash_file(metrics_path),
        loss_curve_path=str(curve_path),
        loss_curve_sha256=hash_file(curve_path),
        epoch_count=len(rows),
        training_record_count=len(train_records),
        validation_record_count=len(validation_records),
        best_validation_loss=best_validation,
        final_training_loss=final_training,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        random_seed=seed,
        parent_checkpoint_sha256=parent_sha,
        source_order3_checkpoint_sha256=order3_sha,
        metrics=summary_metrics,
        metadata={
            "result_path": str(result_path),
            "device": str(resolved_device),
            "best_state_restored": True,
            "promotion_evaluation_completed": False,
        },
    )
    result.validate()
    _atomic_write_text(result_path, result.to_json(indent=2) + "\n")
    return result


def reconstruct_order9_pi_d_teacher_trace(
    record: SequentialDesignTrajectoryRecord,
    physical_model: PhysicalModel,
) -> tuple[DesignPolicyContext, list[Order9DesignTeacherStep]]:
    """Rebuild grammar state and reject any persisted mask/candidate drift."""

    if record.physical_model_hash != physical_model.stable_hash():
        raise SchemaValidationError("Order9 pi_D record physical-model hash mismatch")
    context = DesignPolicyContext(
        task_spec=record.task_spec,
        irg=record.irg,
        physical_model=physical_model,
        interaction_envelope=record.interaction_envelope,
    )
    grammar = Order9DesignGrammar(context)
    state = grammar.initial_state()
    trace: list[Order9DesignTeacherStep] = []
    for persisted in record.steps:
        if [action.to_dict() for action in state.action_history] != [
            action.to_dict() for action in persisted.partial_action_history
        ]:
            raise SchemaValidationError("Order9 pi_D partial state history drifted")
        runtime_candidates = grammar.candidates(state)
        if len(runtime_candidates) != len(persisted.candidates):
            raise SchemaValidationError("Order9 pi_D grammar candidate count drifted")
        for index, (runtime, saved) in enumerate(
            zip(runtime_candidates, persisted.candidates)
        ):
            if (
                runtime.candidate_id != index
                or saved.candidate_index != index
                or runtime.action.to_dict() != saved.action.to_dict()
                or runtime.valid != saved.valid
                or runtime.reason_code != saved.reason_code
                or not math.isclose(
                    runtime.score_prior,
                    saved.score_prior,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise SchemaValidationError(
                    f"Order9 pi_D grammar candidate drifted at step {persisted.step_index}, "
                    f"candidate {index}"
                )
        selected = runtime_candidates[persisted.selected_candidate_index]
        trace.append(
            Order9DesignTeacherStep(
                state=state,
                candidate_step=DesignCandidateStep(
                    step_index=persisted.step_index,
                    selected_action=selected.action,
                    candidates=runtime_candidates,
                ),
            )
        )
        state = grammar.apply(state, selected)
    if not state.stopped:
        raise SchemaValidationError("Order9 pi_D teacher trace did not reach STOP")
    replayed = grammar.build_design_output(state)
    if record.design_output is not None and [
        action.to_dict() for action in replayed.design_actions
    ] != [action.to_dict() for action in record.design_output.design_actions]:
        raise SchemaValidationError("Order9 pi_D final design action trace drifted")
    return context, trace


def _build_bc_model(
    config: Order9LearningConfig,
    stage: Order9CurriculumStage,
    *,
    physical_model: PhysicalModel,
    device: torch.device,
    parent_checkpoint_path: str | Path | None,
    source_order3_checkpoint_path: str | Path | None,
    model_config: object | None,
) -> tuple[nn.Module, Order9PolicyFamily, str | None, str | None]:
    schedule_hash = order9_schedule_hash(config)
    family = _stage_family(stage)
    if parent_checkpoint_path is not None and source_order3_checkpoint_path is not None:
        raise SchemaValidationError(
            "Order9 BC initialization cannot combine Order9 parent and Order3 source"
        )
    if stage.learning_target == Order9LearningTarget.PI_H_TRAJECTORY and parent_checkpoint_path is None:
        raise SchemaValidationError("C5 full pi_H BC requires the promoted C4 parent checkpoint")
    parent_sha = None
    order3_sha = None
    if parent_checkpoint_path is not None:
        loaded = load_order9_policy_checkpoint(
            parent_checkpoint_path,
            device=device,
            expected_family=family,
            expected_schedule_hash=schedule_hash,
        )
        if loaded.metadata.curriculum_stage_index >= stage.stage_index:
            raise SchemaValidationError("Order9 parent checkpoint must come from an earlier stage")
        if loaded.metadata.physical_model_hash != physical_model.stable_hash():
            raise SchemaValidationError("Order9 parent checkpoint physical-model hash mismatch")
        if model_config is not None:
            raise SchemaValidationError("model_config cannot override a parent checkpoint")
        return loaded.model.to(device), family, loaded.sha256, None

    if family == Order9PolicyFamily.PI_L:
        if model_config is not None and not isinstance(model_config, Order9LowLevelPolicyConfig):
            raise TypeError("pi_L model_config must be Order9LowLevelPolicyConfig")
        if source_order3_checkpoint_path is not None:
            source = load_order3_policy_checkpoint(
                source_order3_checkpoint_path, device=device
            )
            raw = source.config.to_dict()
            raw.update(
                {
                    "free_flight_joint_residual_enabled": True,
                    "max_phase_count": 16,
                    "joint_action_log_std_init": -2.0,
                }
            )
            pi_l_config = Order9LowLevelPolicyConfig.from_dict(raw)
            model = Order9PhaseConditionedActorCritic(pi_l_config).to(device)
            model.initialize_from_order3(source.model)
            order3_sha = source.sha256
        else:
            model = Order9PhaseConditionedActorCritic(
                model_config or Order9LowLevelPolicyConfig()
            ).to(device)
    elif family == Order9PolicyFamily.PI_H:
        if model_config is not None and not isinstance(model_config, Order9HighLevelPolicyConfig):
            raise TypeError("pi_H model_config must be Order9HighLevelPolicyConfig")
        model = Order9AutoregressiveHighLevelPolicy(
            model_config or Order9HighLevelPolicyConfig()
        ).to(device)
    else:
        if model_config is not None and not isinstance(model_config, Order9DesignPolicyConfig):
            raise TypeError("pi_D model_config must be Order9DesignPolicyConfig")
        model = Order9AutoregressiveDesignPolicy(
            model_config or Order9DesignPolicyConfig()
        ).to(device)
    return model, family, parent_sha, order3_sha


def _train_pi_l(
    model: nn.Module,
    training_records: Sequence[LowLevelControlRecord],
    validation_records: Sequence[LowLevelControlRecord],
    *,
    physical_model: PhysicalModel,
    optimizer: torch.optim.Optimizer,
    optimization: Order9BCOptimizationConfig,
    seed: int,
    gamma: float,
) -> list[dict[str, float]]:
    if not isinstance(model, Order9PhaseConditionedActorCritic):
        raise TypeError("pi_L BC model type mismatch")
    returns = {
        **_low_level_returns(training_records, gamma=gamma),
        **_low_level_returns(validation_records, gamma=gamma),
    }
    train_windows = _pi_l_windows(
        training_records,
        sequence_length=optimization.sequence_length,
        burn_in_steps=optimization.burn_in_steps,
    )
    validation_windows = _pi_l_windows(
        validation_records,
        sequence_length=optimization.sequence_length,
        burn_in_steps=optimization.burn_in_steps,
    )
    rows: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(optimization.epochs):
        model.train()
        train = _pi_l_epoch(
            model,
            train_windows,
            physical_model=physical_model,
            returns=returns,
            optimizer=optimizer,
            optimization=optimization,
            randomizer=random.Random(seed + epoch),
            training=True,
        )
        model.eval()
        validation = _pi_l_epoch(
            model,
            validation_windows,
            physical_model=physical_model,
            returns=returns,
            optimizer=None,
            optimization=optimization,
            randomizer=None,
            training=False,
        )
        row = _epoch_row(epoch, train, validation)
        rows.append(row)
        if validation["total"] < best_loss:
            best_loss = validation["total"]
            best_state = _cpu_state_dict(model)
    _restore_best(model, best_state)
    return rows


def _pi_l_epoch(
    model: Order9PhaseConditionedActorCritic,
    windows: Sequence[_PiLWindow],
    *,
    physical_model: PhysicalModel,
    returns: Mapping[str, float],
    optimizer: torch.optim.Optimizer | None,
    optimization: Order9BCOptimizationConfig,
    randomizer: random.Random | None,
    training: bool,
) -> dict[str, float]:
    groups: dict[tuple[int, int], list[_PiLWindow]] = defaultdict(list)
    for window in windows:
        groups[(len(window.burn_in), len(window.training))].append(window)
    batches: list[list[_PiLWindow]] = []
    for key in sorted(groups):
        items = groups[key]
        if randomizer is not None:
            randomizer.shuffle(items)
        batches.extend(
            items[index : index + optimization.batch_size]
            for index in range(0, len(items), optimization.batch_size)
        )
    if randomizer is not None:
        randomizer.shuffle(batches)
    totals = defaultdict(float)
    trained_steps = 0
    context_manager = torch.enable_grad if training else torch.no_grad
    for batch in batches:
        burn_length = len(batch[0].burn_in)
        train_length = len(batch[0].training)
        hidden = model.initial_state(len(batch))
        parameter = next(model.parameters())
        hidden = hidden.to(device=parameter.device, dtype=parameter.dtype)
        previous = torch.zeros(
            (len(batch), model.config.action_size),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for offset in range(burn_length):
            records = [window.burn_in[offset] for window in batch]
            contexts = [_low_level_context(record, physical_model) for record in records]
            with torch.no_grad():
                loss = compute_order9_pi_l_behavior_cloning_loss(
                    model,
                    contexts,
                    [record.policy_command for record in records],
                    previous_actions=previous,
                    recurrent_state=hidden,
                    value_loss_weight=optimization.value_loss_weight,
                )
            hidden = loss.recurrent_state_out.detach()
            previous = loss.teacher_global_action.detach()

        losses = []
        with context_manager():
            for offset in range(train_length):
                records = [window.training[offset] for window in batch]
                contexts = [
                    _low_level_context(record, physical_model) for record in records
                ]
                loss = compute_order9_pi_l_behavior_cloning_loss(
                    model,
                    contexts,
                    [record.policy_command for record in records],
                    previous_actions=previous,
                    recurrent_state=hidden,
                    decision_returns=[returns[record.record_id] for record in records],
                    value_loss_weight=optimization.value_loss_weight,
                )
                losses.append(loss)
                hidden = loss.recurrent_state_out
                previous = loss.teacher_global_action.detach()
            batch_loss = torch.stack([item.total for item in losses]).mean()
        if training:
            if optimizer is None:
                raise AssertionError("training pi_L epoch requires optimizer")
            _optimizer_step(
                model, optimizer, batch_loss, max_grad_norm=optimization.max_grad_norm
            )
        weight = len(batch) * train_length
        trained_steps += weight
        totals["total"] += float(batch_loss.detach().cpu().item()) * weight
        for name in ("global_action", "joint_action", "value"):
            value = torch.stack([getattr(item, name) for item in losses]).mean()
            totals[name] += float(value.detach().cpu().item()) * weight
    return _normalize_metrics(totals, trained_steps)


def _train_pi_h(
    model: nn.Module,
    training_records: Sequence[InteractionTrajectoryRecord],
    validation_records: Sequence[InteractionTrajectoryRecord],
    *,
    optimizer: torch.optim.Optimizer,
    optimization: Order9BCOptimizationConfig,
    seed: int,
    assignment_only: bool,
) -> list[dict[str, float]]:
    if not isinstance(model, Order9AutoregressiveHighLevelPolicy):
        raise TypeError("pi_H BC model type mismatch")
    weights = (
        Order9PiHLossWeights(
            assignment=1.0,
            schedule=0.0,
            wrench=0.0,
            timing=0.0,
            centroidal=0.0,
            posture=0.0,
            object_target=0.0,
            priority=0.0,
            guard=0.0,
            value=0.0,
        )
        if assignment_only
        else Order9PiHLossWeights(value=optimization.value_loss_weight)
    )
    rows: list[dict[str, float]] = []
    best_loss = math.inf
    best_state = None
    for epoch in range(optimization.epochs):
        train = _pi_h_epoch(
            model,
            training_records,
            optimizer=optimizer,
            optimization=optimization,
            weights=weights,
            randomizer=random.Random(seed + epoch),
            training=True,
        )
        validation = _pi_h_epoch(
            model,
            validation_records,
            optimizer=None,
            optimization=optimization,
            weights=weights,
            randomizer=None,
            training=False,
        )
        rows.append(_epoch_row(epoch, train, validation))
        if validation["total"] < best_loss:
            best_loss = validation["total"]
            best_state = _cpu_state_dict(model)
    _restore_best(model, best_state)
    return rows


def _pi_h_epoch(
    model: Order9AutoregressiveHighLevelPolicy,
    records: Sequence[InteractionTrajectoryRecord],
    *,
    optimizer: torch.optim.Optimizer | None,
    optimization: Order9BCOptimizationConfig,
    weights: Order9PiHLossWeights,
    randomizer: random.Random | None,
    training: bool,
) -> dict[str, float]:
    items = list(records)
    if randomizer is not None:
        randomizer.shuffle(items)
    model.train(training)
    totals = defaultdict(float)
    count = 0
    for batch in _batches(items, optimization.batch_size):
        contexts = [
            HighLevelPolicyContext(
                irg=record.irg,
                interaction_envelope=record.interaction_envelope,
                morphology_graph=record.morphology_graph,
                contact_candidate_set=record.contact_candidate_set,
                runtime_observation=record.runtime_observation,
            )
            for record in batch
        ]
        with torch.set_grad_enabled(training):
            loss = compute_order9_pi_h_behavior_cloning_loss(
                model,
                contexts,
                [record.trajectory for record in batch],
                decision_returns=[record.decision_return for record in batch],
                weights=weights,
            )
        if training:
            if optimizer is None:
                raise AssertionError("training pi_H epoch requires optimizer")
            _optimizer_step(
                model, optimizer, loss.total, max_grad_norm=optimization.max_grad_norm
            )
        weight = len(batch)
        count += weight
        for name in (
            "total",
            "assignment",
            "schedule",
            "wrench",
            "timing",
            "centroidal",
            "posture",
            "object_target",
            "priority",
            "guard",
            "value",
        ):
            totals[name] += float(getattr(loss, name).detach().cpu().item()) * weight
    return _normalize_metrics(totals, count)


def _train_pi_d(
    model: nn.Module,
    training_records: Sequence[SequentialDesignTrajectoryRecord],
    validation_records: Sequence[SequentialDesignTrajectoryRecord],
    *,
    physical_model: PhysicalModel,
    optimizer: torch.optim.Optimizer,
    optimization: Order9BCOptimizationConfig,
    seed: int,
) -> list[dict[str, float]]:
    if not isinstance(model, Order9AutoregressiveDesignPolicy):
        raise TypeError("pi_D BC model type mismatch")
    prepared = {
        record.record_id: reconstruct_order9_pi_d_teacher_trace(record, physical_model)
        for record in (*training_records, *validation_records)
    }
    rows: list[dict[str, float]] = []
    best_loss = math.inf
    best_state = None
    for epoch in range(optimization.epochs):
        train = _pi_d_epoch(
            model,
            training_records,
            prepared=prepared,
            optimizer=optimizer,
            optimization=optimization,
            randomizer=random.Random(seed + epoch),
            training=True,
        )
        validation = _pi_d_epoch(
            model,
            validation_records,
            prepared=prepared,
            optimizer=None,
            optimization=optimization,
            randomizer=None,
            training=False,
        )
        rows.append(_epoch_row(epoch, train, validation))
        if validation["total"] < best_loss:
            best_loss = validation["total"]
            best_state = _cpu_state_dict(model)
    _restore_best(model, best_state)
    return rows


def _pi_d_epoch(
    model: Order9AutoregressiveDesignPolicy,
    records: Sequence[SequentialDesignTrajectoryRecord],
    *,
    prepared: Mapping[
        str, tuple[DesignPolicyContext, list[Order9DesignTeacherStep]]
    ],
    optimizer: torch.optim.Optimizer | None,
    optimization: Order9BCOptimizationConfig,
    randomizer: random.Random | None,
    training: bool,
) -> dict[str, float]:
    items = list(records)
    if randomizer is not None:
        randomizer.shuffle(items)
    model.train(training)
    totals = defaultdict(float)
    count = 0
    for batch in _batches(items, optimization.batch_size):
        losses = []
        with torch.set_grad_enabled(training):
            for record in batch:
                context, trace = prepared[record.record_id]
                losses.append(
                    compute_order9_pi_d_behavior_cloning_loss(
                        model,
                        context,
                        trace,
                        design_return=record.episode_return,
                        value_loss_weight=optimization.value_loss_weight,
                    )
                )
            batch_loss = torch.stack([loss.total for loss in losses]).mean()
        if training:
            if optimizer is None:
                raise AssertionError("training pi_D epoch requires optimizer")
            _optimizer_step(
                model, optimizer, batch_loss, max_grad_norm=optimization.max_grad_norm
            )
        weight = len(batch)
        count += weight
        totals["total"] += float(batch_loss.detach().cpu().item()) * weight
        for name in ("policy", "value", "entropy"):
            value = torch.stack([getattr(loss, name) for loss in losses]).mean()
            totals[name] += float(value.detach().cpu().item()) * weight
    return _normalize_metrics(totals, count)


def _low_level_context(
    record: LowLevelControlRecord,
    physical_model: PhysicalModel,
) -> LowLevelPolicyContext:
    if any(
        value is None
        for value in (
            record.task_type,
            record.task_adapter_id,
            record.phase_index,
            record.phase_count,
        )
    ):
        raise SchemaValidationError("Order9 pi_L record lacks task/phase context")
    knot = InteractionKnot.from_dict(record.active_knot.to_dict())
    knot.t_rel_s = 0.0
    trajectory = ContactWrenchTrajectory(
        horizon_s=0.02,
        dt_s=0.02,
        knots=[knot],
        derived_mode_label="order9_low_level_record_active_knot",
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )
    return LowLevelPolicyContext(
        runtime_observation=record.runtime_observation,
        morphology_graph=record.runtime_observation.morphology_graph,
        physical_model=physical_model,
        contact_wrench_trajectory=trajectory,
        active_knot=knot,
        task_type=record.task_type,
        task_adapter_id=record.task_adapter_id,
        phase_index=record.phase_index,
        phase_count=record.phase_count,
    )


def _pi_l_windows(
    records: Sequence[LowLevelControlRecord],
    *,
    sequence_length: int,
    burn_in_steps: int,
) -> list[_PiLWindow]:
    by_episode: dict[str, list[LowLevelControlRecord]] = defaultdict(list)
    for record in records:
        by_episode[record.episode_id].append(record)
    windows: list[_PiLWindow] = []
    for episode_id in sorted(by_episode):
        ordered = sorted(by_episode[episode_id], key=lambda item: item.step_index)
        for start in range(0, len(ordered), sequence_length):
            training = tuple(ordered[start : start + sequence_length])
            burn_start = max(0, start - burn_in_steps)
            burn = tuple(ordered[burn_start:start])
            if training:
                windows.append(_PiLWindow(burn_in=burn, training=training))
    if not windows:
        raise SchemaValidationError("Order9 pi_L split produced no sequence windows")
    return windows


def _low_level_returns(
    records: Sequence[LowLevelControlRecord], *, gamma: float
) -> dict[str, float]:
    by_episode: dict[str, list[LowLevelControlRecord]] = defaultdict(list)
    for record in records:
        by_episode[record.episode_id].append(record)
    result: dict[str, float] = {}
    for episode in by_episode.values():
        running = 0.0
        for record in reversed(sorted(episode, key=lambda item: item.step_index)):
            if record.truncated:
                running = float(record.bootstrap_value)
            if record.terminal:
                running = 0.0
            if record.reward is None:
                raise SchemaValidationError("Order9 pi_L BC record reward is missing")
            running = float(record.reward) + gamma * running
            result[record.record_id] = running
    return result


def _stage_family(stage: Order9CurriculumStage) -> Order9PolicyFamily:
    if stage.learning_target == Order9LearningTarget.PI_L:
        return Order9PolicyFamily.PI_L
    if stage.learning_target in {
        Order9LearningTarget.PI_H_ASSIGNMENT,
        Order9LearningTarget.PI_H_TRAJECTORY,
    }:
        return Order9PolicyFamily.PI_H
    if stage.learning_target == Order9LearningTarget.PI_D:
        return Order9PolicyFamily.PI_D
    raise SchemaValidationError(
        f"Order9 BC stage target {stage.learning_target.value!r} is unsupported"
    )


def _bc_optimization(
    config: Order9LearningConfig, stage: Order9CurriculumStage
) -> Order9BCOptimizationConfig:
    if stage.learning_target == Order9LearningTarget.PI_L:
        return config.optimization.pi_l_bc
    if stage.learning_target == Order9LearningTarget.PI_H_ASSIGNMENT:
        return config.optimization.pi_h_assignment_bc
    if stage.learning_target == Order9LearningTarget.PI_H_TRAJECTORY:
        return config.optimization.pi_h_full_bc
    if stage.learning_target == Order9LearningTarget.PI_D:
        return config.optimization.pi_d_bc
    raise SchemaValidationError("Order9 stage has no BC optimization block")


def build_order9_checkpoint_metadata(
    model: nn.Module,
    *,
    stage: Order9CurriculumStage,
    schedule_hash: str,
    physical_model_hash: str,
    git_revision: str,
    random_seed: int,
    input_artifact_hashes: dict[str, str],
    parent_checkpoint_sha256: str | None,
    source_order3_checkpoint_sha256: str | None,
    metrics: dict[str, float],
    trainer_version: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Order9PolicyCheckpointMetadata:
    family, policy_version = order9_policy_identity(model)
    contracts = {
        Order9PolicyFamily.PI_L: (
            "task_phase_morphology_centroidal_no_raw_contact_v1",
            "actor_plus_privileged_disturbance_v1",
            "bounded_global_and_masked_local_joint_residual_v1",
        ),
        Order9PolicyFamily.PI_H: (
            "irg_envelope_morphology_candidates_runtime_object_no_raw_contact_v1",
            "high_level_context_value_v1",
            "full_contact_wrench_trajectory_proposal_v2",
        ),
        Order9PolicyFamily.PI_D: (
            "taskspec_irg_envelope_partial_design_mask_v1",
            "design_context_value_v1",
            "masked_autoregressive_graph_edit_v1",
        ),
    }
    actor, critic, action = contracts[family]
    model_config = order9_model_config_dict(model)
    return Order9PolicyCheckpointMetadata(
        checkpoint_version=ORDER9_POLICY_CHECKPOINT_VERSION,
        policy_family=family,
        policy_version=policy_version,
        curriculum_schedule_hash=schedule_hash,
        curriculum_stage_id=stage.stage_id,
        curriculum_stage_index=stage.stage_index,
        learning_mode=stage.learning_mode.value,
        model_config_hash=stable_hash(model_config),
        state_dict_hash=order9_state_dict_hash(model.state_dict()),
        physical_model_hash=physical_model_hash,
        actor_observation_contract=actor,
        critic_observation_contract=critic,
        action_contract=action,
        git_revision=git_revision,
        random_seed=random_seed,
        input_artifact_hashes=input_artifact_hashes,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        source_order3_checkpoint_sha256=source_order3_checkpoint_sha256,
        metrics=metrics,
        metadata={
            "trainer_version": trainer_version,
            "promotion_evaluation_completed": False,
            **dict(extra_metadata or {}),
        },
    )


def order9_checkpoint_input_hashes(
    bundle: Order9DatasetBundle,
    *,
    parent_checkpoint_path: str | Path | None,
    source_order3_checkpoint_path: str | Path | None,
    additional_paths: Mapping[str, str | Path],
) -> dict[str, str]:
    result = {"dataset_manifest": bundle.manifest_sha256}
    for index, (_, digest) in enumerate(sorted(bundle.verified_shard_sha256.items())):
        result[f"dataset_shard_{index:03d}"] = digest
    if parent_checkpoint_path is not None:
        result["parent_checkpoint"] = hash_file(parent_checkpoint_path)
    if source_order3_checkpoint_path is not None:
        result["source_order3_checkpoint"] = hash_file(source_order3_checkpoint_path)
    for name, path in sorted(additional_paths.items()):
        key = f"input_{name}"
        if key in result:
            raise SchemaValidationError(f"duplicate Order9 input artifact key {key!r}")
        result[key] = hash_file(path)
    return result


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    max_grad_norm: float,
) -> None:
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Order9 training loss became non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not bool(torch.isfinite(torch.as_tensor(norm)).item()):
        raise FloatingPointError("Order9 training gradient norm became non-finite")
    optimizer.step()


def _split_records(records: Iterable[Any], split: DatasetSplit) -> list[Any]:
    values = [record for record in records if record.split == split]
    if not values:
        raise SchemaValidationError(f"Order9 dataset {split.value} split is empty")
    return values


def _batches(values: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _normalize_metrics(
    totals: Mapping[str, float], count: int
) -> dict[str, float]:
    if count < 1:
        raise SchemaValidationError("Order9 training epoch consumed no records")
    result = {key: float(value) / count for key, value in totals.items()}
    if "total" not in result or not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Order9 epoch metrics are incomplete or non-finite")
    return result


def _epoch_row(
    epoch: int,
    training: Mapping[str, float],
    validation: Mapping[str, float],
) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        **{f"training_{key}": float(value) for key, value in training.items()},
        **{f"validation_{key}": float(value) for key, value in validation.items()},
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _restore_best(
    model: nn.Module, state: Mapping[str, torch.Tensor] | None
) -> None:
    if state is None:
        raise RuntimeError("Order9 training did not produce a best checkpoint")
    model.load_state_dict(state, strict=True)
    model.eval()


def _resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Order9 production config requests CUDA but CUDA is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _write_loss_curve(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    if not rows:
        raise ValueError("Order9 loss curve cannot be empty")
    keys = sorted({key for row in rows for key in row})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, stream.getvalue())


def _json_text(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"Order9OfflineTrainingResult.{label} is not SHA-256")
