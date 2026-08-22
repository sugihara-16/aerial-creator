from __future__ import annotations

"""Training-only contact-normal labels from short signed-action probes.

This module deliberately keeps PhysX contact force out of the deployable
policy input.  Force is used only offline to select a target category for a
state whose deployable observation is already present in the rollout.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from amsrr.policies.order9_low_level_policy import (
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_tensor_pi_l_ppo import (
    _TensorReplay,
    _training_preload_deficit_categorical_teacher_artifact_signal,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    Order9TensorRolloutArtifact,
)


ORDER9_CONTACT_NORMAL_PROBE_TEACHER_VERSION = (
    "order9_contact_normal_short_probe_teacher_v1"
)
ORDER9_CONTACT_NORMAL_PRELOAD_DEFICIT_TEACHER_VERSION = (
    "order9_contact_normal_preload_deficit_teacher_v1"
)


@dataclass(frozen=True)
class Order9ContactNormalProbeCandidate:
    residual_mm: float
    environment_index: int
    evaluated_step_count: int
    mean_reward: float
    weak_anchor_normal_force_n: float
    qp_feasible_fraction: float
    prohibited_collision: bool
    rotor_saturation: bool
    terminal: bool

    @property
    def safe(self) -> bool:
        return (
            self.evaluated_step_count > 0
            and self.qp_feasible_fraction == 1.0
            and not self.prohibited_collision
            and not self.rotor_saturation
            and not self.terminal
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "residual_mm": self.residual_mm,
            "environment_index": self.environment_index,
            "evaluated_step_count": self.evaluated_step_count,
            "mean_reward": self.mean_reward,
            "weak_anchor_normal_force_n": self.weak_anchor_normal_force_n,
            "qp_feasible_fraction": self.qp_feasible_fraction,
            "prohibited_collision": self.prohibited_collision,
            "rotor_saturation": self.rotor_saturation,
            "terminal": self.terminal,
            "safe": self.safe,
        }


def select_order9_contact_normal_probe_teacher_candidate(
    candidates: Sequence[Order9ContactNormalProbeCandidate],
    *,
    required_weak_anchor_normal_force_n: float,
) -> Order9ContactNormalProbeCandidate | None:
    """Choose the smallest nonnegative safe residual meeting force support."""

    eligible = [
        candidate
        for candidate in candidates
        if candidate.safe
        and candidate.residual_mm >= 0.0
        and candidate.weak_anchor_normal_force_n
        >= required_weak_anchor_normal_force_n
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            candidate.residual_mm,
            -candidate.mean_reward,
            candidate.environment_index,
        ),
    )


def build_order9_contact_normal_probe_teacher_dataset(
    artifact: Order9TensorRolloutArtifact,
    *,
    source_rollout_sha256: str,
    maximum_probe_steps: int = 64,
) -> dict[str, object]:
    """Extract one categorical target per independently reset probe state."""

    artifact.validate()
    metadata = artifact.metadata
    residuals_raw = metadata.get("diagnostic_contact_normal_residual_sweep_mm")
    strata_raw = metadata.get("diagnostic_initial_phase_strata")
    if not isinstance(residuals_raw, list) or len(residuals_raw) < 2:
        raise SchemaValidationError("contact-normal probe residual sweep is missing")
    if not isinstance(strata_raw, list) or len(strata_raw) != artifact.environment_count:
        raise SchemaValidationError("contact-normal probe phase strata are missing")
    residuals = tuple(float(value) for value in residuals_raw)
    if artifact.environment_count % len(residuals) != 0:
        raise SchemaValidationError("contact-normal probe environment groups differ")
    states_per_action = artifact.environment_count // len(residuals)
    state_schedule = tuple(tuple(int(value) for value in row) for row in strata_raw)
    base_states = state_schedule[:states_per_action]
    for action_index in range(len(residuals)):
        start = action_index * states_per_action
        if state_schedule[start : start + states_per_action] != base_states:
            raise SchemaValidationError("contact-normal probe state schedule differs")

    preload = metadata.get("actuator_aware_nominal_preload")
    if not isinstance(preload, Mapping):
        raise SchemaValidationError("actuator-aware nominal preload metadata is missing")
    target_forces = preload.get("target_normal_force_n_by_anchor")
    if not isinstance(target_forces, list) or not target_forces:
        raise SchemaValidationError("target normal forces are missing")
    required_force = max(float(value) for value in target_forces)

    tensors = artifact.tensors
    records: list[dict[str, object]] = []
    confident: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    category_min_mm = -20.0
    category_step_mm = 0.5
    for state_index, (phase_index, stratum_index) in enumerate(base_states):
        candidates: list[Order9ContactNormalProbeCandidate] = []
        for action_index, residual_mm in enumerate(residuals):
            environment = action_index * states_per_action + state_index
            mask = (
                tensors["valid"][:, environment]
                & tensors["episode_serial"][:, environment].eq(0)
            )
            times = mask.nonzero(as_tuple=False).flatten()[:maximum_probe_steps]
            if times.numel() == 0:
                continue
            normal_force = tensors["selected_contact_wrenches_contact"][
                times, environment, :2, 0
            ].abs()
            weak_force = normal_force.min(dim=-1).values.mean()
            candidate = Order9ContactNormalProbeCandidate(
                residual_mm=float(residual_mm),
                environment_index=environment,
                evaluated_step_count=int(times.numel()),
                mean_reward=float(tensors["reward"][times, environment].mean()),
                weak_anchor_normal_force_n=float(weak_force),
                qp_feasible_fraction=float(
                    tensors["qp_feasible"][times, environment].float().mean()
                ),
                prohibited_collision=bool(
                    tensors["prohibited_collision"][times, environment].any()
                ),
                rotor_saturation=bool(
                    tensors["rotor_saturation"][times, environment].any()
                ),
                terminal=bool(tensors["terminal"][times, environment].any()),
            )
            candidates.append(candidate)
        selected = select_order9_contact_normal_probe_teacher_candidate(
            candidates,
            required_weak_anchor_normal_force_n=required_force,
        )
        record: dict[str, object] = {
            "state_index": state_index,
            "phase_index": phase_index,
            "stratum_index": stratum_index,
            "required_weak_anchor_normal_force_n": required_force,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "confident": selected is not None,
        }
        if selected is None:
            record["exclusion_reason"] = (
                "no_nonnegative_safe_candidate_met_required_weak_anchor_force"
            )
            excluded.append(record)
        else:
            category_index = round(
                (selected.residual_mm - category_min_mm) / category_step_mm
            )
            record.update(
                {
                    "source_environment_index": selected.environment_index,
                    "source_time_index": 0,
                    "target_residual_mm": selected.residual_mm,
                    "target_category_index": category_index,
                    "selected_candidate": selected.to_dict(),
                }
            )
            confident.append(record)
        records.append(record)

    return {
        "version": ORDER9_CONTACT_NORMAL_PROBE_TEACHER_VERSION,
        "source_rollout_sha256": source_rollout_sha256,
        "source_checkpoint_sha256": str(metadata["pi_l_checkpoint_sha256"]),
        "task_id": str(metadata["task_specs"][0]["task_id"]),
        "module_count": len(metadata["module_ids"]),
        "maximum_probe_steps": maximum_probe_steps,
        "required_weak_anchor_normal_force_n": required_force,
        "category_min_mm": category_min_mm,
        "category_max_mm": 20.0,
        "category_step_mm": category_step_mm,
        "state_count": len(records),
        "confident_example_count": len(confident),
        "excluded_state_count": len(excluded),
        "examples": confident,
        "excluded_states": excluded,
        "states": records,
    }


def validate_order9_contact_normal_probe_teacher_dataset(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("version") not in {
        ORDER9_CONTACT_NORMAL_PROBE_TEACHER_VERSION,
        ORDER9_CONTACT_NORMAL_PRELOAD_DEFICIT_TEACHER_VERSION,
    }:
        raise SchemaValidationError("contact-normal probe teacher version differs")
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise SchemaValidationError("contact-normal probe teacher has no examples")
    if int(payload.get("confident_example_count", -1)) != len(examples):
        raise SchemaValidationError("contact-normal probe teacher count differs")
    for example in examples:
        if (
            not isinstance(example, Mapping)
            or not 0 <= int(example["target_category_index"]) < 81
            or int(example["source_time_index"]) < 0
            or int(example["source_environment_index"]) < 0
        ):
            raise SchemaValidationError("contact-normal probe teacher example is invalid")


def select_order9_contact_normal_preload_deficit_teacher_rows(
    *,
    valid: torch.Tensor,
    phase_index: torch.Tensor,
    category_index_by_environment: torch.Tensor,
    deficit_environment_mask: torch.Tensor,
    target_phase_indices: Sequence[int],
) -> tuple[list[dict[str, int | float]], list[dict[str, int]]]:
    """Select one representative observed state per environment and phase.

    Positive-deficit environments become categorical teacher examples.  The
    otherwise identical zero-deficit environments become retention examples,
    so head-only pretraining cannot collapse every morphology state to a
    positive compression category.
    """

    if (
        valid.ndim != 2
        or phase_index.shape != valid.shape
        or category_index_by_environment.shape != (valid.shape[1],)
        or deficit_environment_mask.shape != (valid.shape[1],)
        or valid.dtype != torch.bool
        or deficit_environment_mask.dtype != torch.bool
    ):
        raise SchemaValidationError(
            "preload-deficit teacher selection tensors differ"
        )
    phases = tuple(dict.fromkeys(int(value) for value in target_phase_indices))
    if not phases or any(value < 0 for value in phases):
        raise SchemaValidationError(
            "preload-deficit teacher target phases are invalid"
        )
    examples: list[dict[str, int | float]] = []
    retention: list[dict[str, int]] = []
    for environment in range(valid.shape[1]):
        for phase in phases:
            times = torch.nonzero(
                valid[:, environment]
                & phase_index[:, environment].eq(phase),
                as_tuple=False,
            ).flatten()
            if times.numel() == 0:
                continue
            time_index = int(times[times.numel() // 2])
            if bool(deficit_environment_mask[environment]):
                target_index = int(category_index_by_environment[environment])
                examples.append(
                    {
                        "source_time_index": time_index,
                        "source_environment_index": environment,
                        "phase_index": phase,
                        "target_category_index": target_index,
                        "target_residual_mm": (target_index - 40) * 0.5,
                    }
                )
            else:
                retention.append(
                    {
                        "source_time_index": time_index,
                        "source_environment_index": environment,
                        "phase_index": phase,
                    }
                )
    if not examples:
        raise SchemaValidationError(
            "preload-deficit teacher selection has no positive examples"
        )
    if not retention:
        raise SchemaValidationError(
            "preload-deficit teacher selection has no zero-deficit retention"
        )
    return examples, retention


def build_order9_contact_normal_preload_deficit_teacher_dataset(
    artifact: Order9TensorRolloutArtifact,
    *,
    source_rollout_sha256: str,
    target_phase_labels: Sequence[str],
) -> dict[str, object]:
    """Build head-only labels from authored train-only nominal deficits."""

    artifact.validate()
    actor_phase_labels = artifact.metadata.get("actor_phase_labels")
    if not isinstance(actor_phase_labels, list) or not actor_phase_labels:
        raise SchemaValidationError(
            "preload-deficit teacher actor phase labels are missing"
        )
    try:
        target_phase_indices = tuple(
            actor_phase_labels.index(str(label)) for label in target_phase_labels
        )
    except ValueError as exc:
        raise SchemaValidationError(
            "preload-deficit teacher target phase label is unavailable"
        ) from exc
    category, deficit_mask = (
        _training_preload_deficit_categorical_teacher_artifact_signal(artifact)
    )
    examples, retention = (
        select_order9_contact_normal_preload_deficit_teacher_rows(
            valid=artifact.tensors["valid"].bool(),
            phase_index=artifact.tensors["phase_index"].long(),
            category_index_by_environment=category,
            deficit_environment_mask=deficit_mask,
            target_phase_indices=target_phase_indices,
        )
    )
    return {
        "version": ORDER9_CONTACT_NORMAL_PRELOAD_DEFICIT_TEACHER_VERSION,
        "source_rollout_sha256": source_rollout_sha256,
        "source_checkpoint_sha256": str(
            artifact.metadata["pi_l_checkpoint_sha256"]
        ),
        "task_id": str(artifact.metadata["task_specs"][0]["task_id"]),
        "module_count": len(artifact.metadata["module_ids"]),
        "target_phase_labels": [str(value) for value in target_phase_labels],
        "target_phase_indices": list(target_phase_indices),
        "confident_example_count": len(examples),
        "retention_example_count": len(retention),
        "examples": examples,
        "retention_examples": retention,
        "excluded_states": [],
    }


def validate_order9_contact_normal_probe_teacher_target_lineage(
    *,
    probe_source_checkpoint_sha256: str,
    target_checkpoint_sha256: str,
    target_parent_checkpoint_sha256: str | None,
) -> None:
    """Allow the source policy itself or its direct on-policy child.

    The second case is the intended combined update: first perform ordinary
    PPO from the probe source, then apply the training-only auxiliary pass to
    that direct child.  More distant or unrelated checkpoints fail closed.
    """

    if probe_source_checkpoint_sha256 == target_checkpoint_sha256:
        return
    if probe_source_checkpoint_sha256 == target_parent_checkpoint_sha256:
        return
    raise SchemaValidationError(
        "probe teacher target is neither its source policy nor a direct child"
    )


@dataclass(frozen=True)
class Order9ContactNormalProbeTeacherUpdateResult:
    optimizer_step_count: int
    example_count: int
    active_slot_example_count: int
    initial_teacher_loss: float
    final_teacher_loss: float
    initial_target_match_fraction: float
    final_target_match_fraction: float
    retention_kl: float
    changed_parameter_names: tuple[str, ...]
    target_residual_mm_by_example: tuple[float, ...]
    predicted_residual_mm_before: tuple[tuple[float, ...], ...]
    predicted_residual_mm_after: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer_step_count": self.optimizer_step_count,
            "example_count": self.example_count,
            "active_slot_example_count": self.active_slot_example_count,
            "initial_teacher_loss": self.initial_teacher_loss,
            "final_teacher_loss": self.final_teacher_loss,
            "initial_target_match_fraction": self.initial_target_match_fraction,
            "final_target_match_fraction": self.final_target_match_fraction,
            "retention_kl": self.retention_kl,
            "changed_parameter_names": list(self.changed_parameter_names),
            "target_residual_mm_by_example": list(
                self.target_residual_mm_by_example
            ),
            "predicted_residual_mm_before": [
                list(row) for row in self.predicted_residual_mm_before
            ],
            "predicted_residual_mm_after": [
                list(row) for row in self.predicted_residual_mm_after
            ],
        }


def order9_contact_normal_head_named_parameters(
    policy: Order9CategoricalContactNormalPhaseConditionedActorCritic,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    values = tuple(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if name.startswith("contact_normal_category_logits.")
    )
    if not values:
        raise SchemaValidationError("categorical contact-normal head is missing")
    return values


def train_order9_contact_normal_probe_teacher_head(
    policy: Order9CategoricalContactNormalPhaseConditionedActorCritic,
    artifact: Order9TensorRolloutArtifact,
    dataset: Mapping[str, Any],
    *,
    physical_model: PhysicalModel,
    learning_rate: float = 1.0e-3,
    maximum_optimizer_steps: int = 128,
    retention_kl_weight: float = 1.0,
    parameter_drift_weight: float = 1.0e-5,
) -> Order9ContactNormalProbeTeacherUpdateResult:
    """Fit only the categorical normal head to short-probe labels.

    All other parameters are frozen and verified bit-for-bit after the pass.
    Excluded contact-acquisition states retain their parent distribution via a
    KL term, preventing the six lift/transport labels from becoming a global
    unconditional +compression bias.
    """

    validate_order9_contact_normal_probe_teacher_dataset(dataset)
    if learning_rate <= 0.0 or maximum_optimizer_steps < 1:
        raise ValueError("probe-teacher optimizer settings are invalid")
    if retention_kl_weight < 0.0 or parameter_drift_weight < 0.0:
        raise ValueError("probe-teacher regularization weights are invalid")
    if dataset["source_checkpoint_sha256"] != artifact.metadata[
        "pi_l_checkpoint_sha256"
    ]:
        raise SchemaValidationError("probe teacher/checkpoint lineage differs")

    replay = _TensorReplay(artifact, policy=policy, physical_model=physical_model)
    examples = list(dataset["examples"])
    teacher_times = [int(value["source_time_index"]) for value in examples]
    teacher_environments = [
        int(value["source_environment_index"]) for value in examples
    ]
    targets = torch.tensor(
        [int(value["target_category_index"]) for value in examples],
        device=next(policy.parameters()).device,
        dtype=torch.long,
    )

    retention_times: list[int] = []
    retention_environments: list[int] = []
    explicit_retention = list(dataset.get("retention_examples", []))
    if explicit_retention:
        retention_times = [
            int(value["source_time_index"]) for value in explicit_retention
        ]
        retention_environments = [
            int(value["source_environment_index"])
            for value in explicit_retention
        ]
    else:
        excluded = list(dataset.get("excluded_states", []))
        for state in excluded:
            zero_candidate = next(
                (
                    value
                    for value in state.get("candidates", [])
                    if float(value["residual_mm"]) == 0.0
                ),
                None,
            )
            if zero_candidate is not None:
                retention_times.append(0)
                retention_environments.append(
                    int(zero_candidate["environment_index"])
                )

    before_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    head = order9_contact_normal_head_named_parameters(policy)
    head_names = {name for name, _ in head}
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name in head_names)
    initial_head = {
        name: parameter.detach().clone() for name, parameter in head
    }
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in head],
        lr=learning_rate,
        weight_decay=0.0,
    )

    def forward_logits(times: list[int], environments: list[int]):
        batch = replay.batch(times, environments)
        step = policy.step(
            batch.graph,
            None,
            batch.actor_features,
            batch.previous_global_action,
            batch.recurrent_state_in,
            phase_features=batch.phase_features,
            privileged_disturbance_body=batch.privileged,
            active_knot_features=batch.active_knot_features,
            active_assignment_features=batch.active_assignment_features,
            contact_slot_features=batch.contact_slot_features,
            contact_slot_owner_module_indices=(
                batch.contact_slot_owner_module_indices
            ),
            contact_slot_mask=batch.contact_slot_mask,
            deterministic=True,
        )
        if step.contact_normal_category_logits is None:
            raise SchemaValidationError("categorical contact-normal logits are missing")
        return step.contact_normal_category_logits, batch.contact_slot_mask

    with torch.no_grad():
        initial_logits, teacher_mask = forward_logits(
            teacher_times, teacher_environments
        )
        retention_parent_logits = None
        retention_mask = None
        if retention_environments:
            retention_parent_logits, retention_mask = forward_logits(
                retention_times, retention_environments
            )
        expanded_targets = targets[:, None].expand_as(teacher_mask)
        initial_loss = functional.cross_entropy(
            initial_logits[teacher_mask], expanded_targets[teacher_mask]
        )
        initial_match = (
            initial_logits.argmax(dim=-1)[teacher_mask]
            == expanded_targets[teacher_mask]
        ).float().mean()
        initial_prediction = _category_logits_to_mm(initial_logits, teacher_mask)

    optimizer_steps = 0
    for optimizer_steps in range(1, maximum_optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, mask = forward_logits(teacher_times, teacher_environments)
        expanded_targets = targets[:, None].expand_as(mask)
        teacher_loss = functional.cross_entropy(
            logits[mask], expanded_targets[mask]
        )
        retention_kl = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if retention_parent_logits is not None and retention_mask is not None:
            current_retention, current_mask = forward_logits(
                retention_times, retention_environments
            )
            if not torch.equal(current_mask, retention_mask):
                raise SchemaValidationError("probe retention contact mask changed")
            retention_kl = functional.kl_div(
                current_retention[current_mask],
                retention_parent_logits[current_mask].exp(),
                reduction="batchmean",
            )
        drift = torch.zeros((), device=logits.device, dtype=logits.dtype)
        for name, parameter in head:
            drift = drift + (parameter - initial_head[name]).square().mean()
        loss = (
            teacher_loss
            + retention_kl_weight * retention_kl
            + parameter_drift_weight * drift
        )
        loss.backward()
        optimizer.step()
        if (
            logits.argmax(dim=-1)[mask]
            == expanded_targets[mask]
        ).all() and optimizer_steps >= 8:
            break

    with torch.no_grad():
        final_logits, final_mask = forward_logits(
            teacher_times, teacher_environments
        )
        expanded_targets = targets[:, None].expand_as(final_mask)
        final_loss = functional.cross_entropy(
            final_logits[final_mask], expanded_targets[final_mask]
        )
        final_match = (
            final_logits.argmax(dim=-1)[final_mask]
            == expanded_targets[final_mask]
        ).float().mean()
        final_prediction = _category_logits_to_mm(final_logits, final_mask)
        final_retention_kl = torch.zeros(())
        if retention_parent_logits is not None and retention_mask is not None:
            current_retention, current_mask = forward_logits(
                retention_times, retention_environments
            )
            final_retention_kl = functional.kl_div(
                current_retention[current_mask],
                retention_parent_logits[current_mask].exp(),
                reduction="batchmean",
            ).cpu()

    changed: list[str] = []
    for name, value in policy.state_dict().items():
        if not torch.equal(before_state[name], value.detach().cpu()):
            changed.append(name)
            if not name.startswith("contact_normal_category_logits."):
                raise SchemaValidationError(
                    f"probe teacher changed forbidden parameter {name!r}"
                )
    if not changed:
        raise SchemaValidationError("probe teacher did not update its head")
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.eval()
    return Order9ContactNormalProbeTeacherUpdateResult(
        optimizer_step_count=optimizer_steps,
        example_count=len(examples),
        active_slot_example_count=int(final_mask.sum()),
        initial_teacher_loss=float(initial_loss),
        final_teacher_loss=float(final_loss),
        initial_target_match_fraction=float(initial_match),
        final_target_match_fraction=float(final_match),
        retention_kl=float(final_retention_kl),
        changed_parameter_names=tuple(sorted(changed)),
        target_residual_mm_by_example=tuple(
            float(value["target_residual_mm"]) for value in examples
        ),
        predicted_residual_mm_before=initial_prediction,
        predicted_residual_mm_after=final_prediction,
    )


def _category_logits_to_mm(
    logits: torch.Tensor, mask: torch.Tensor
) -> tuple[tuple[float, ...], ...]:
    indices = logits.argmax(dim=-1)
    values = (indices.to(torch.float32) - 40.0) * 0.5
    return tuple(
        tuple(float(value) for value in values[row][mask[row]].cpu())
        for row in range(values.shape[0])
    )
