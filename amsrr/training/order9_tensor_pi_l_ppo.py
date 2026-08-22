from __future__ import annotations

"""Tensor-native recurrent PPO replay for Order 9 ``pi_L`` rollouts."""

import hashlib
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import torch

from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS,
    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION,
    Order9ActiveKnotPhaseConditionedActorCritic,
    Order9ContactResidualPhaseConditionedActorCritic,
    Order9ContactSpacePhaseConditionedActorCritic,
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
    Order9MorphologyInvariantCompressionActorCritic,
    Order9PhaseConditionedActorCritic,
)
from amsrr.policies.order9_tensor_command_decoder import (
    ORDER9_CONTACT_COMPRESSION_ACTION_ADAPTER_VERSION,
    ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskType
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ADAPTER_ID,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
)
from amsrr.training.order9_curriculum import (
    Order9C3BoundaryFineTuneConfig,
    Order9PPOOptimizationConfig,
)
from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACTS,
    order9_c3_action_contract_global_dimension,
    order9_c3_action_contract_uses_contact_space,
    order9_c3_action_contract_uses_full_policy,
)
from amsrr.training.order9_contact_space_action import (
    ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION,
)
from amsrr.training.order9_factorized_actor_credit import (
    ORDER9_CONTACT_COORDINATE_ACTION_INDICES,
    ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS,
    ORDER9_CONTACT_COORDINATE_CREDIT_TERM_WEIGHTS,
    ORDER9_CONTACT_COORDINATE_CREDIT_VERSION,
    ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS,
    ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS,
    ORDER9_FACTORIZED_ACTOR_CREDIT_VERSION,
    factorize_order9_contact_coordinate_reward,
    factorize_order9_actor_reward,
)
from amsrr.training.order9_pi_l_learning import compute_order9_pi_l_ppo_loss
from amsrr.training.order9_pi_h_learning import clipped_ppo_surrogate_loss
from amsrr.training.order9_ppo import (
    ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED,
    Order9PPOUpdateResult,
    _ppo_result,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    Order9TensorRolloutArtifact,
)
from amsrr.training.order9_tensor_runtime import (
    Order9CentroidalTensorObservation,
    Order9TensorizedTopologyBucket,
    order9_low_level_actor_features_from_tensors,
)
from amsrr.training.order9_training_nominal_preload_deficit import (
    ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION,
    order9_training_preload_deficit_category_indices,
)
from amsrr.utils.hashing import stable_hash


ORDER9_TENSOR_PI_L_PPO_VERSION = (
    "order9_tensor_native_pi_l_ppo_v16_preload_deficit_categorical_teacher"
)
ORDER9_C3_BOUNDARY_FINE_TUNE_CONTRACT = (
    "order9_c3_boundary_fine_tune_v10_preload_deficit_categorical_teacher"
)
ORDER9_CONTACT_HEAD_EXTRA_OPTIMIZER_CONTRACT = (
    "order9_contact_head_extra_optimizer_v2_coordinate_factorized_advantage"
)
ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES = (
    "contact_space_actor_log_std",
    "contact_space_actor_mean.",
    "contact_space_feature_encoder.",
    "contact_space_slot_embedding.",
)
ORDER9_TOPOLOGY_MINIBATCH_PACKING = (
    "phase_contiguous_sequences_by_transition_budget_v1"
)


@dataclass(frozen=True)
class _Sequence:
    environment: int
    start: int
    length: int
    phase_index: int


@dataclass(frozen=True)
class _PolicyBatch:
    graph: Any
    actor_features: torch.Tensor
    phase_features: torch.Tensor
    privileged: torch.Tensor
    active_knot_features: torch.Tensor
    active_assignment_features: torch.Tensor
    global_action: torch.Tensor
    applied_global_action: torch.Tensor
    joint_action: torch.Tensor
    contact_compression_residual_action: torch.Tensor
    contact_space_residual_action: torch.Tensor
    contact_slot_features: torch.Tensor
    contact_slot_owner_module_indices: torch.Tensor
    contact_slot_mask: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    recurrent_state_in: torch.Tensor
    recurrent_state_out: torch.Tensor
    previous_global_action: torch.Tensor
    phase_index: torch.Tensor
    compression_teacher_direction: torch.Tensor
    compression_teacher_mask: torch.Tensor
    compression_teacher_satisfied_mask: torch.Tensor
    contact_normal_preload_deficit_teacher_category_index: torch.Tensor
    contact_normal_preload_deficit_teacher_mask: torch.Tensor
    compression_control_module_index: int
    compression_control_joint_index: int
    advantage: torch.Tensor | None = None
    return_: torch.Tensor | None = None


@dataclass(frozen=True)
class _TopologyReplayData:
    key: str
    replay: "_TensorReplay"
    sequences: tuple[_Sequence, ...]
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True)
class _SequencePPOTerms:
    total: torch.Tensor
    new_log_prob: torch.Tensor
    old_log_prob: torch.Tensor
    new_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    entropy: torch.Tensor
    phase_index: torch.Tensor
    target_phase_mask: torch.Tensor
    non_target_parent_kl: torch.Tensor
    privileged_compression_teacher_loss: torch.Tensor
    privileged_compression_teacher_sample_count: int
    privileged_wrench_satisfied_parent_kl: torch.Tensor
    privileged_wrench_satisfied_sample_count: int
    contact_normal_preload_deficit_teacher_loss: torch.Tensor
    contact_normal_preload_deficit_teacher_sample_count: int
    actor_loss: torch.Tensor | None = None
    actor_component_losses: Mapping[str, torch.Tensor] | None = None
    actor_component_advantages: Mapping[str, torch.Tensor] | None = None
    actor_component_new_log_probs: Mapping[str, torch.Tensor] | None = None
    actor_component_old_log_probs: Mapping[str, torch.Tensor] | None = None
    contact_coordinate_losses: Mapping[str, torch.Tensor] | None = None
    contact_coordinate_advantages: Mapping[str, torch.Tensor] | None = None
    contact_coordinate_new_log_probs: Mapping[str, torch.Tensor] | None = None
    contact_coordinate_old_log_probs: Mapping[str, torch.Tensor] | None = None


def _squashed_coordinate_log_prob_and_entropy(
    *,
    bounded_action: torch.Tensor,
    bounded_mean: torch.Tensor,
    log_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row log probability and raw-Normal entropy for coordinates."""

    if bounded_action.shape != bounded_mean.shape or bounded_action.ndim != 2:
        raise ValueError("Order9 contracted action/mean shapes differ")
    if log_std.shape not in {
        (bounded_action.shape[1],),
        tuple(bounded_action.shape),
    }:
        raise ValueError("Order9 contracted log-std shape differs")
    action = bounded_action.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    mean = bounded_mean.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    raw_action = torch.atanh(action)
    raw_mean = torch.atanh(mean)
    std_values = torch.exp(torch.clamp(log_std, min=-6.0, max=1.0))
    std = (
        std_values.reshape(1, -1).expand_as(raw_mean)
        if log_std.ndim == 1
        else std_values
    )
    distribution = torch.distributions.Normal(raw_mean, std)
    log_prob = (
        distribution.log_prob(raw_action)
        - torch.log1p(-action.square() + 1.0e-6)
    ).sum(dim=-1)
    return log_prob, distribution.entropy().sum(dim=-1)


def _contact_coordinate_group_log_probs(
    *,
    bounded_action: torch.Tensor,
    bounded_mean: torch.Tensor,
    log_std: torch.Tensor,
    slot_mask: torch.Tensor,
    categorical_contact_log_prob: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return contact log probability separated by physical action role."""

    if (
        bounded_action.shape != bounded_mean.shape
        or bounded_action.ndim != 3
        or bounded_action.shape[-1] != 6
        or slot_mask.shape != bounded_action.shape[:2]
        or slot_mask.dtype != torch.bool
    ):
        raise ValueError("Order9 contact-coordinate density shapes differ")
    if log_std.shape != (bounded_action.shape[-1],):
        raise ValueError("Order9 contact-coordinate log-std shape differs")
    action = bounded_action.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    mean = bounded_mean.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    raw_action = torch.atanh(action)
    raw_mean = torch.atanh(mean)
    std = torch.exp(torch.clamp(log_std, min=-6.0, max=1.0)).reshape(
        1, 1, -1
    )
    distribution = torch.distributions.Normal(raw_mean, std)
    per_coordinate = distribution.log_prob(raw_action) - torch.log1p(
        -action.square() + 1.0e-6
    )
    per_coordinate = per_coordinate * slot_mask.unsqueeze(-1).to(
        per_coordinate.dtype
    )
    rows = {
        name: per_coordinate[..., list(indices)].sum(dim=(1, 2))
        for name, indices in ORDER9_CONTACT_COORDINATE_ACTION_INDICES.items()
    }
    if categorical_contact_log_prob is not None:
        if categorical_contact_log_prob.shape != rows["normal_translation"].shape:
            raise ValueError("Order9 categorical contact log-prob shape differs")
        # Tangential translation and rotation remain Gaussian.  The residual
        # required to recover the exact aggregate contact likelihood is the
        # direct categorical normal-coordinate likelihood.
        rows["normal_translation"] = (
            categorical_contact_log_prob
            - rows["tangential_translation"]
            - rows["rotation"]
        )
    return rows


def _named_factorized_clipped_actor_loss(
    *,
    channel_names: Sequence[str],
    new_log_probs: Mapping[str, torch.Tensor],
    old_log_probs: Mapping[str, torch.Tensor],
    advantages: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    clip_ratio: float,
    label: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected = set(channel_names)
    if (
        set(new_log_probs) != expected
        or set(old_log_probs) != expected
        or set(advantages) != expected
        or mask.dtype != torch.bool
        or not bool(mask.any())
    ):
        raise SchemaValidationError(
            f"Order9 {label} actor loss channels or mask differ"
        )
    losses: dict[str, torch.Tensor] = {}
    for name in channel_names:
        if (
            new_log_probs[name].shape != mask.shape
            or old_log_probs[name].shape != mask.shape
            or advantages[name].shape != mask.shape
        ):
            raise SchemaValidationError(
                f"Order9 {label} actor loss tensor shapes differ"
            )
        losses[name] = clipped_ppo_surrogate_loss(
            new_log_prob=new_log_probs[name][mask],
            old_log_prob=old_log_probs[name][mask],
            advantages=advantages[name][mask],
            clip_ratio=clip_ratio,
        )
    return torch.stack([losses[name] for name in channel_names]).mean(), losses


def _factorized_clipped_actor_loss(
    *,
    new_log_probs: Mapping[str, torch.Tensor],
    old_log_probs: Mapping[str, torch.Tensor],
    advantages: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply each role-specific advantage only to its matching density."""

    return _named_factorized_clipped_actor_loss(
        channel_names=ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS,
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        mask=mask,
        clip_ratio=clip_ratio,
        label="factorized",
    )


def _weighted_contact_coordinate_actor_loss(
    losses: Mapping[str, torch.Tensor],
    *,
    normal_translation_weight: float,
) -> torch.Tensor:
    """Combine contact-coordinate losses while preserving weight-one behavior."""

    expected = set(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
    if set(losses) != expected:
        raise SchemaValidationError(
            "Order9 contact-coordinate actor loss channels differ"
        )
    weight = float(normal_translation_weight)
    if not math.isfinite(weight) or weight <= 0.0:
        raise SchemaValidationError(
            "Order9 contact normal-translation actor-loss weight must be "
            "finite and positive"
        )
    return (
        weight * losses["normal_translation"]
        + losses["tangential_translation"]
        + losses["rotation"]
    ) / float(len(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS))


def _categorical_preload_deficit_teacher_loss(
    *,
    category_logits: torch.Tensor,
    category_index: torch.Tensor,
    environment_mask: torch.Tensor,
    contact_slot_mask: torch.Tensor,
    target_phase_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return NLL for positive authored deficits on active contact slots."""

    if (
        category_logits.ndim != 3
        or category_index.shape != category_logits.shape[:1]
        or environment_mask.shape != category_index.shape
        or target_phase_mask.shape != category_index.shape
        or contact_slot_mask.shape != category_logits.shape[:2]
        or environment_mask.dtype != torch.bool
        or contact_slot_mask.dtype != torch.bool
        or target_phase_mask.dtype != torch.bool
    ):
        raise SchemaValidationError(
            "Order9 preload-deficit categorical teacher tensors differ"
        )
    slot_mask = (
        environment_mask[:, None]
        & target_phase_mask[:, None]
        & contact_slot_mask
    )
    sample_count = int(slot_mask.sum().item())
    if sample_count == 0:
        return category_logits.sum() * 0.0, 0
    expanded_index = category_index[:, None].expand(
        -1, category_logits.shape[1]
    )
    selected_log_prob = category_logits.gather(
        -1, expanded_index.unsqueeze(-1)
    ).squeeze(-1)
    return -selected_log_prob[slot_mask].mean(), sample_count


def _order9_contact_head_named_parameters(
    policy: Order9PhaseConditionedActorCritic,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Return the strictly contact-specific v7/v8 actor parameters.

    The shared graph/recurrent trunk is deliberately excluded.  Updating that
    trunk from contact-only credit would also move the centroidal and posture
    outputs, defeating the extra pass's single-responsibility contract.
    """

    if not isinstance(policy, Order9ContactSpacePhaseConditionedActorCritic):
        raise SchemaValidationError(
            "Order9 contact-head extra optimization requires the contact-space actor"
        )
    prefixes = ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES + (
        ("contact_normal_category_logits.",)
        if isinstance(
            policy, Order9CategoricalContactNormalPhaseConditionedActorCritic
        )
        else ()
    )
    result = tuple(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
        and name.startswith(prefixes)
    )
    observed_prefixes = {
        prefix
        for prefix in prefixes
        if any(name.startswith(prefix) for name, _ in result)
    }
    if observed_prefixes != set(prefixes):
        raise SchemaValidationError(
            "Order9 contact-head parameter boundary is incomplete"
        )
    return result


def order9_privileged_compression_teacher_direction(
    *,
    selected_contact_wrenches_contact: torch.Tensor,
    wrench_lower_contact: torch.Tensor,
    wrench_upper_contact: torch.Tensor,
    wrench_bound_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a bounded training-only correction for compression action.

    The signed center of each force box identifies its authored grasp-normal
    direction.  A measured projection below the projected lower bound asks
    for more inward compression; a projection above the upper bound asks for
    less.  The exact PhysX wrench is only used to build this supervised loss
    target and is never returned to the actor observation.
    """

    if (
        selected_contact_wrenches_contact.ndim < 3
        or selected_contact_wrenches_contact.shape[-1] != 6
        or wrench_lower_contact.shape
        != selected_contact_wrenches_contact.shape
        or wrench_upper_contact.shape != wrench_lower_contact.shape
        or wrench_bound_mask.shape
        != selected_contact_wrenches_contact.shape[:-1]
    ):
        raise ValueError("Order9 privileged compression teacher shapes differ")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            selected_contact_wrenches_contact,
            wrench_lower_contact,
            wrench_upper_contact,
        )
    ):
        raise ValueError("Order9 privileged compression teacher is non-finite")
    if bool((wrench_lower_contact > wrench_upper_contact).any()):
        raise ValueError("Order9 privileged compression wrench box is inverted")

    measured_force = selected_contact_wrenches_contact[..., :3]
    lower_force = wrench_lower_contact[..., :3]
    upper_force = wrench_upper_contact[..., :3]
    center = 0.5 * (lower_force + upper_force)
    half_width = 0.5 * (upper_force - lower_force)
    center_norm = torch.linalg.vector_norm(center, dim=-1)
    anchor_valid = wrench_bound_mask.bool() & (center_norm > 1.0e-6)
    direction_contact = center / center_norm.clamp_min(1.0e-6).unsqueeze(-1)
    projected_center = (center * direction_contact).sum(dim=-1)
    projected_half_width = (
        half_width * direction_contact.abs()
    ).sum(dim=-1).clamp_min(1.0)
    projected_actual = (measured_force * direction_contact).sum(dim=-1)
    projected_lower = projected_center - projected_half_width
    projected_upper = projected_center + projected_half_width
    normalized_error = torch.where(
        projected_actual < projected_lower,
        (projected_lower - projected_actual) / projected_half_width,
        torch.where(
            projected_actual > projected_upper,
            (projected_upper - projected_actual) / projected_half_width,
            torch.zeros_like(projected_actual),
        ),
    )
    violated = anchor_valid & (normalized_error.abs() > 1.0e-6)
    anchor_direction = torch.tanh(normalized_error) * anchor_valid.to(
        normalized_error.dtype
    )
    divisor = anchor_valid.sum(dim=-1).clamp_min(1).to(anchor_direction.dtype)
    direction = anchor_direction.sum(dim=-1) / divisor
    sample_mask = violated.any(dim=-1)
    return torch.where(sample_mask, direction, torch.zeros_like(direction)), sample_mask


def _privileged_compression_teacher_artifact_signal(
    artifact: Order9TensorRolloutArtifact,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    step_count = artifact.step_count
    environment_count = artifact.environment_count
    zero_direction = torch.zeros((step_count, environment_count))
    zero_mask = torch.zeros(
        (step_count, environment_count), dtype=torch.bool
    )
    adapter_version = artifact.metadata.get(
        "contact_compression_action_adapter_version"
    )
    if adapter_version is None:
        return zero_direction, zero_mask, zero_mask.clone(), 0, 0
    if adapter_version == ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION:
        # v7 has no grasp-specific scalar and receives contact credit through
        # its native per-contact action distribution.
        return zero_direction, zero_mask, zero_mask.clone(), 0, 0
    if adapter_version not in {
        ORDER9_CONTACT_COMPRESSION_ACTION_ADAPTER_VERSION,
        ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION,
        ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION,
    }:
        raise SchemaValidationError(
            "Order9 privileged compression teacher adapter version differs"
        )
    if artifact.metadata.get("raw_contact_actor_input") is not False:
        raise SchemaValidationError(
            "Order9 privileged compression teacher requires contact-free actor input"
        )
    control_id = artifact.metadata.get(
        "contact_compression_action_control_joint_id"
    )
    if not isinstance(control_id, str) or ":" not in control_id:
        raise SchemaValidationError(
            "Order9 privileged compression teacher lacks its control joint"
        )
    raw_module_id, joint_id = control_id.split(":", 1)
    try:
        module_id = int(raw_module_id.removeprefix("module_"))
        module_index = list(artifact.metadata["module_ids"]).index(module_id)
        joint_index = list(
            artifact.metadata["command_local_joint_ids"]
        ).index(joint_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            "Order9 privileged compression teacher control joint is invalid"
        ) from exc
    required = (
        "selected_contact_wrenches_contact",
        "wrench_lower_contact",
        "wrench_upper_contact",
        "wrench_bound_mask",
        "valid",
    )
    if any(name not in artifact.tensors for name in required):
        raise SchemaValidationError(
            "Order9 privileged compression teacher tensors are incomplete"
        )
    direction, sample_mask = order9_privileged_compression_teacher_direction(
        selected_contact_wrenches_contact=artifact.tensors[
            "selected_contact_wrenches_contact"
        ],
        wrench_lower_contact=artifact.tensors["wrench_lower_contact"],
        wrench_upper_contact=artifact.tensors["wrench_upper_contact"],
        wrench_bound_mask=artifact.tensors["wrench_bound_mask"],
    )
    sample_mask &= artifact.tensors["valid"].bool()
    force_center = 0.5 * (
        artifact.tensors["wrench_lower_contact"][..., :3]
        + artifact.tensors["wrench_upper_contact"][..., :3]
    )
    force_bound_mask = artifact.tensors["wrench_bound_mask"].bool() & (
        torch.linalg.vector_norm(force_center, dim=-1) > 1.0e-6
    )
    satisfied_mask = (
        force_bound_mask.any(dim=-1)
        & ~sample_mask
        & artifact.tensors["valid"].bool()
    )
    return (
        direction.cpu(),
        sample_mask.cpu(),
        satisfied_mask.cpu(),
        module_index,
        joint_index,
    )


def _training_preload_deficit_categorical_teacher_artifact_signal(
    artifact: Order9TensorRolloutArtifact,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment categorical targets for authored under-preload.

    The authored deficit is training metadata, not an actor observation.  A
    zero-deficit environment remains an ordinary on-policy sample and is not
    included in this auxiliary supervised objective.
    """

    environment_count = artifact.environment_count
    zero_category = torch.zeros(environment_count, dtype=torch.long)
    zero_mask = torch.zeros(environment_count, dtype=torch.bool)
    version = artifact.metadata.get("training_nominal_preload_deficit_version")
    if version is None:
        return zero_category, zero_mask
    if version != ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION:
        raise SchemaValidationError(
            "Order9 training nominal-preload deficit version differs"
        )
    if (
        artifact.metadata.get("pi_l_policy_version")
        != ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION
    ):
        raise SchemaValidationError(
            "Order9 preload-deficit categorical teacher requires the "
            "categorical contact-normal behavior policy"
        )
    total_lead = artifact.metadata.get(
        "training_nominal_preload_total_lead_m_by_environment"
    )
    if not isinstance(total_lead, list) or len(total_lead) != environment_count:
        raise SchemaValidationError(
            "Order9 preload-deficit categorical teacher lead metadata differs"
        )
    try:
        category_indices = order9_training_preload_deficit_category_indices(
            nominal_lead_m=float(
                artifact.metadata["virtual_contact_inward_lead_m"]
            ),
            total_lead_m_by_environment=[float(value) for value in total_lead],
            category_step_m=float(
                artifact.metadata["contact_normal_action_category_step_m"]
            ),
            category_count=int(
                artifact.metadata["contact_normal_action_category_count"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            "Order9 preload-deficit categorical teacher metadata is invalid"
        ) from exc
    category = torch.tensor(category_indices, dtype=torch.long)
    zero_index = int(
        artifact.metadata["contact_normal_action_category_count"]
    ) // 2
    return category, category != zero_index


class _TensorReplay:
    def __init__(
        self,
        artifact: Order9TensorRolloutArtifact,
        *,
        policy: Order9PhaseConditionedActorCritic,
        physical_model: PhysicalModel,
    ) -> None:
        artifact.validate()
        if artifact.artifact_version not in {
            ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
            ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
            ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
        }:
            raise SchemaValidationError(
                "Order9 tensor-native PPO requires the current rollout artifact"
            )
        if not isinstance(policy, Order9ActiveKnotPhaseConditionedActorCritic):
            raise TypeError("Order9 tensor-native PPO requires active-knot pi_L")
        if (
            artifact.metadata.get("pi_l_policy_version")
            not in {
                ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION,
                ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION,
                ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION,
                ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
                ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
                ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
            }
        ):
            raise SchemaValidationError("Order9 tensor behavior policy version differs")
        if (
            artifact.metadata.get("pi_l_policy_version")
            == ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION
            and not isinstance(
                policy, Order9ContactResidualPhaseConditionedActorCritic
            )
        ):
            raise SchemaValidationError(
                "Order9 contact-residual behavior requires its v5 actor"
            )
        if (
            artifact.metadata.get("pi_l_policy_version")
            == ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION
            and not isinstance(
                policy, Order9MorphologyInvariantCompressionActorCritic
            )
        ):
            raise SchemaValidationError(
                "Order9 morphology-invariant behavior requires its v6 actor"
            )
        if (
            artifact.metadata.get("pi_l_policy_version")
            in ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS
            and not isinstance(policy, Order9ContactSpacePhaseConditionedActorCritic)
        ):
            raise SchemaValidationError(
                "Order9 contact-space behavior requires its v7/v8 actor"
            )
        if artifact.metadata.get(
            "actor_graph_joint_summary_semantics",
            ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED,
        ) != ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED:
            raise SchemaValidationError("Order9 tensor graph joint summary differs")
        self.artifact = artifact
        self.tensors = artifact.tensors
        self.policy = policy
        self.morphology = MorphologyGraph.from_dict(
            artifact.metadata["morphology_graph"]
        )
        self.builder = BatchedRigidBodyControlModelBuilder(
            self.morphology, physical_model
        )
        if tuple(int(value) for value in artifact.metadata["module_ids"]) != tuple(
            self.builder.module_ids
        ):
            raise SchemaValidationError("Order9 tensor module order differs")
        if tuple(str(value) for value in artifact.metadata["local_joint_ids"]) != tuple(
            self.builder.local_joint_ids
        ):
            raise SchemaValidationError("Order9 tensor local-joint order differs")
        parameter = next(policy.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.origins = torch.tensor(
            [
                task["metadata"]["isaac_environment_origin_world"]
                for task in artifact.metadata["task_specs"]
            ],
            dtype=torch.float32,
        )
        if self.origins.shape != (artifact.environment_count, 3) or not bool(
            torch.isfinite(self.origins).all()
        ):
            raise SchemaValidationError("Order9 tensor policy-frame origins are invalid")
        joint_types = {
            joint.joint_id: joint.joint_type for joint in physical_model.joints
        }
        self.active_joint_mask = torch.tensor(
            [
                joint_types[joint_id] != "fixed"
                for joint_id in self.builder.local_joint_ids
            ],
            device=self.device,
            dtype=torch.bool,
        )
        self._buckets: dict[int, Order9TensorizedTopologyBucket] = {}
        self._phase_templates: dict[int, torch.Tensor] = {}
        self.behavior_component_log_probs: dict[str, torch.Tensor] | None = None
        self.factorized_actor_advantages: dict[str, torch.Tensor] | None = None
        self.behavior_contact_coordinate_log_probs: (
            dict[str, torch.Tensor] | None
        ) = None
        self.contact_coordinate_advantages: dict[str, torch.Tensor] | None = None
        (
            self.compression_teacher_direction,
            self.compression_teacher_mask,
            self.compression_teacher_satisfied_mask,
            self.compression_control_module_index,
            self.compression_control_joint_index,
        ) = _privileged_compression_teacher_artifact_signal(artifact)
        (
            self.contact_normal_preload_deficit_teacher_category_index,
            self.contact_normal_preload_deficit_teacher_mask,
        ) = _training_preload_deficit_categorical_teacher_artifact_signal(
            artifact
        )

    def batch(
        self,
        times: Sequence[int],
        environments: Sequence[int],
        *,
        advantages: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
    ) -> _PolicyBatch:
        if not times or len(times) != len(environments):
            raise ValueError("Order9 tensor PPO batch indices are empty or misaligned")
        time_index = torch.tensor(times, dtype=torch.long)
        environment_index = torch.tensor(environments, dtype=torch.long)

        def take(name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
            value = self.tensors[name][time_index, environment_index]
            return value.to(
                device=self.device,
                dtype=self.dtype if dtype is None and value.is_floating_point() else dtype,
                non_blocking=True,
            )

        module_pose = take("module_pose_world")
        module_twist = take("module_twist_world")
        joint_position = take("local_joint_positions_rad")
        joint_velocity = take("local_joint_velocities_radps")
        control = self.builder.build(
            module_pose_world=module_pose,
            module_twist_world=module_twist,
            local_joint_positions_rad=joint_position,
        )
        batch_size = len(times)
        bucket = self._bucket(batch_size)
        policy_pose = module_pose.clone()
        policy_pose[..., :3].sub_(
            self.origins.index_select(0, environment_index).to(
                device=self.device, dtype=self.dtype
            )[:, None, :]
        )
        joint_mask = self.active_joint_mask.reshape(1, 1, -1).expand(
            batch_size, self.builder.module_count, self.builder.local_joint_count
        )
        cached_graph = bucket.update_runtime_(
            module_pose_world=policy_pose,
            module_twist_world=module_twist,
            module_health=torch.ones(
                (batch_size, self.builder.module_count),
                device=self.device,
                dtype=self.dtype,
            ),
            joint_positions=joint_position,
            joint_velocities=joint_velocity,
            joint_mask=joint_mask,
        )
        # The topology cache is updated in place on the next recurrent
        # timestep.  Autograd must retain an immutable node-feature snapshot
        # for backpropagation through the complete sequence.
        graph = replace(
            cached_graph, node_features=cached_graph.node_features.clone()
        )
        phase_index = take("phase_index", dtype=torch.long)
        phase_progress = take("phase_progress")
        actor_features = order9_low_level_actor_features_from_tensors(
            Order9CentroidalTensorObservation(
                time_s=take("time_s"),
                module_count=torch.full(
                    (batch_size,),
                    float(self.builder.module_count),
                    device=self.device,
                    dtype=self.dtype,
                ),
                total_mass_kg=control.total_mass_kg,
                inertia_body=control.inertia_body,
                body_pose_world=control.body_pose_world,
                body_twist_world=control.body_twist_world,
                target_pose_world=take("desired_body_pose_world"),
                target_twist=take("desired_body_twist_reference"),
                controller_qp_feasible=take(
                    "actor_controller_qp_feasible", dtype=torch.bool
                ),
                controller_status_one_hot=take(
                    "actor_controller_status_one_hot"
                ),
                allocation_residual_norm=take("actor_allocation_residual_norm"),
                task_progress_ratio=phase_progress,
                task_success=take("actor_task_success", dtype=torch.bool),
            ),
            max_modules=self.policy.config.max_modules,
        )
        phase_features = self._phase_template(batch_size).clone()
        phase_offset = len(TaskType)
        progress_offset = phase_offset + self.policy.config.max_phase_count
        phase_features.scatter_(
            1, (phase_offset + phase_index).unsqueeze(1), 1.0
        )
        phase_features[:, progress_offset] = phase_progress
        selected_advantage = None
        selected_return = None
        if advantages is not None:
            selected_advantage = advantages[time_index, environment_index].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
        if returns is not None:
            selected_return = returns[time_index, environment_index].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
        return _PolicyBatch(
            graph=graph,
            actor_features=actor_features,
            phase_features=phase_features,
            privileged=take("privileged_disturbance_body"),
            active_knot_features=take("actor_active_knot_features"),
            active_assignment_features=take("actor_active_assignment_features"),
            global_action=take("global_action"),
            applied_global_action=(
                take("applied_global_action")
                if "applied_global_action" in self.tensors
                else take("global_action")
            ),
            joint_action=take("joint_action"),
            contact_compression_residual_action=(
                take("contact_compression_residual_action")
                if "contact_compression_residual_action" in self.tensors
                else torch.zeros(
                    (batch_size,), device=self.device, dtype=self.dtype
                )
            ),
            contact_space_residual_action=(
                take("contact_space_residual_action")
                if "contact_space_residual_action" in self.tensors
                else torch.zeros(
                    (
                        batch_size,
                        int(self.policy.config.max_contact_slots),
                        6,
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
            ),
            contact_slot_features=(
                take("actor_contact_slot_features")
                if "actor_contact_slot_features" in self.tensors
                else torch.zeros(
                    (
                        batch_size,
                        int(self.policy.config.max_contact_slots),
                        int(self.policy.config.contact_space_feature_dim),
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
            ),
            contact_slot_owner_module_indices=(
                take("actor_contact_slot_owner_indices", dtype=torch.long)
                if "actor_contact_slot_owner_indices" in self.tensors
                else torch.full(
                    (batch_size, int(self.policy.config.max_contact_slots)),
                    -1,
                    device=self.device,
                    dtype=torch.long,
                )
            ),
            contact_slot_mask=(
                take("actor_contact_slot_mask", dtype=torch.bool)
                if "actor_contact_slot_mask" in self.tensors
                else torch.zeros(
                    (batch_size, int(self.policy.config.max_contact_slots)),
                    device=self.device,
                    dtype=torch.bool,
                )
            ),
            old_log_prob=take("old_log_prob"),
            old_value=take("old_value"),
            recurrent_state_in=take("recurrent_state_in"),
            recurrent_state_out=take("recurrent_state_out"),
            previous_global_action=take("previous_global_action"),
            phase_index=phase_index,
            compression_teacher_direction=(
                self.compression_teacher_direction[
                    time_index, environment_index
                ].to(device=self.device, dtype=self.dtype, non_blocking=True)
            ),
            compression_teacher_mask=(
                self.compression_teacher_mask[
                    time_index, environment_index
                ].to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
            compression_teacher_satisfied_mask=(
                self.compression_teacher_satisfied_mask[
                    time_index, environment_index
                ].to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
            contact_normal_preload_deficit_teacher_category_index=(
                self.contact_normal_preload_deficit_teacher_category_index[
                    environment_index
                ].to(device=self.device, dtype=torch.long, non_blocking=True)
            ),
            contact_normal_preload_deficit_teacher_mask=(
                self.contact_normal_preload_deficit_teacher_mask[
                    environment_index
                ].to(device=self.device, dtype=torch.bool, non_blocking=True)
            ),
            compression_control_module_index=(
                self.compression_control_module_index
            ),
            compression_control_joint_index=(
                self.compression_control_joint_index
            ),
            advantage=selected_advantage,
            return_=selected_return,
        )

    def _bucket(self, batch_size: int) -> Order9TensorizedTopologyBucket:
        if batch_size not in self._buckets:
            bucket = Order9TensorizedTopologyBucket(
                self.morphology,
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
            bucket.batch.metadata.update(
                {
                    "runtime_pose_translation_frame": "world_minus_policy_frame_origin",
                    "runtime_joint_summary_semantics": (
                        ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED
                    ),
                    "runtime_active_local_joint_count": int(
                        self.active_joint_mask.sum().item()
                    ),
                }
            )
            self._buckets[batch_size] = bucket
        return self._buckets[batch_size]

    def _phase_template(self, batch_size: int) -> torch.Tensor:
        if batch_size not in self._phase_templates:
            values = torch.zeros(
                (batch_size, self.policy.config.phase_feature_dim),
                device=self.device,
                dtype=self.dtype,
            )
            values[:, list(TaskType).index(TaskType.OBJECT_GRASP_CARRY)] = 1.0
            progress_offset = len(TaskType) + self.policy.config.max_phase_count
            adapter = int.from_bytes(
                hashlib.sha256(
                    ORDER9_OBJECT_TASK_ADAPTER_ID.encode("utf-8")
                ).digest()[:8],
                "big",
            ) / float(2**64 - 1)
            values[:, progress_offset + 1] = math.sin(2.0 * math.pi * adapter)
            values[:, progress_offset + 2] = math.cos(2.0 * math.pi * adapter)
            self._phase_templates[batch_size] = values
        return self._phase_templates[batch_size]


def update_order9_tensor_pi_l_ppo(
    policy: Order9PhaseConditionedActorCritic,
    artifact: Order9TensorRolloutArtifact | Sequence[Order9TensorRolloutArtifact],
    *,
    physical_model: PhysicalModel,
    optimizer: torch.optim.Optimizer,
    config: Order9PPOOptimizationConfig,
    behavior_checkpoint_sha256: str,
    seed: int,
    sequence_length: int = 16,
    exact_replay_sequence_batch_size: int = 1024,
    boundary_fine_tune: Order9C3BoundaryFineTuneConfig | None = None,
    compression_only_actor_objective: bool = False,
    joint_head_only_actor_update: bool = False,
    contact_residual_only_actor_update: bool = False,
    c3_action_contract: str | None = None,
    progress_callback: Callable[[int, Mapping[str, float]], None] | None = None,
) -> Order9PPOUpdateResult:
    """Run exact recurrent replay over topology-homogeneous PPO subbatches."""

    config.validate()
    if sequence_length < 1 or exact_replay_sequence_batch_size < 1:
        raise ValueError("Order9 tensor PPO sequence settings must be positive")
    artifacts = (
        (artifact,)
        if isinstance(artifact, Order9TensorRolloutArtifact)
        else tuple(artifact)
    )
    if not artifacts:
        raise SchemaValidationError("Order9 tensor PPO has no train topology shards")
    rollout_modes = [
        str(
            item.metadata.get(
                "c3_train_rollout_mode", "phase_reset"
            )
        )
        for item in artifacts
    ]
    if any(
        value not in {"phase_reset", "continuous_state_inheritance"}
        for value in rollout_modes
    ):
        raise SchemaValidationError("Order9 tensor PPO rollout mode is invalid")
    state_inheritance_artifacts = [
        item
        for item, mode in zip(artifacts, rollout_modes)
        if mode == "continuous_state_inheritance"
    ]
    if any(
        item.metadata.get("state_inheritance_preserves_phase_transitions")
        is not True
        or item.metadata.get(
            "state_inheritance_recurrent_state_reset_only_at_episode_boundary"
        )
        is not True
        for item in state_inheritance_artifacts
    ):
        raise SchemaValidationError(
            "Order9 tensor PPO state-inheritance contract differs"
        )
    fine_tune = boundary_fine_tune
    if fine_tune is not None:
        fine_tune.validate()
        if not fine_tune.enabled:
            fine_tune = None
    if joint_head_only_actor_update and not compression_only_actor_objective:
        raise SchemaValidationError(
            "Order9 joint-head-only update requires compression-only credit"
        )
    if (
        contact_residual_only_actor_update
        and not compression_only_actor_objective
    ):
        raise SchemaValidationError(
            "Order9 contact-residual-only update requires compression-only "
            "credit"
        )
    if joint_head_only_actor_update and contact_residual_only_actor_update:
        raise SchemaValidationError(
            "Order9 joint-head-only and contact-residual-only updates are "
            "mutually exclusive"
        )
    if contact_residual_only_actor_update and not isinstance(
        policy, Order9ContactResidualPhaseConditionedActorCritic
    ):
        raise SchemaValidationError(
            "Order9 contact-residual-only update requires the v5 actor"
        )
    if c3_action_contract is not None and c3_action_contract not in (
        ORDER9_C3_ACTION_CONTRACTS
    ):
        raise SchemaValidationError("Order9 C3 PPO action contract is invalid")
    if (
        c3_action_contract is not None
        and not order9_c3_action_contract_uses_full_policy(c3_action_contract)
        and not compression_only_actor_objective
    ):
        raise SchemaValidationError(
            "Order9 C3 action contract requires action-specific actor credit"
        )
    factorized_actor_credit_enabled = bool(
        fine_tune is not None and fine_tune.factorized_actor_credit_enabled
    )
    contact_head_extra_optimizer_passes = int(
        0
        if fine_tune is None
        else fine_tune.contact_head_extra_optimizer_passes
    )
    contact_coordinate_credit_enabled = bool(
        fine_tune is not None and fine_tune.contact_coordinate_credit_enabled
    )
    preload_deficit_teacher_enabled = bool(
        fine_tune is not None
        and fine_tune.contact_normal_preload_deficit_teacher_weight > 0.0
    )
    if factorized_actor_credit_enabled and (
        c3_action_contract is None
        or not order9_c3_action_contract_uses_contact_space(c3_action_contract)
        or not isinstance(policy, Order9ContactSpacePhaseConditionedActorCritic)
    ):
        raise SchemaValidationError(
            "Order9 factorized actor credit requires the contact-space policy contract"
        )
    if factorized_actor_credit_enabled and (
        compression_only_actor_objective
        or joint_head_only_actor_update
        or contact_residual_only_actor_update
    ):
        raise SchemaValidationError(
            "Order9 factorized actor credit cannot mix with a legacy head-only update"
        )
    if contact_head_extra_optimizer_passes > 0 and (
        not factorized_actor_credit_enabled
        or c3_action_contract is None
        or not order9_c3_action_contract_uses_contact_space(
            c3_action_contract
        )
    ):
        raise SchemaValidationError(
            "Order9 contact-head extra optimization requires the factorized "
            "contact-space contract"
        )
    if contact_coordinate_credit_enabled and (
        not factorized_actor_credit_enabled
    ):
        raise SchemaValidationError(
            "Order9 contact-coordinate credit requires factorized actor credit"
        )
    if preload_deficit_teacher_enabled and not isinstance(
        policy, Order9CategoricalContactNormalPhaseConditionedActorCritic
    ):
        raise SchemaValidationError(
            "Order9 preload-deficit categorical teacher requires the "
            "categorical contact-normal policy"
        )
    target_phase_indices = _resolve_boundary_target_phase_indices(
        artifacts, fine_tune
    )
    topology_data: list[_TopologyReplayData] = []
    exact_rows: dict[str, dict[str, float | int | bool]] = {}
    advantage_rows: dict[str, dict[str, object]] = {}
    factorized_advantage_rows: dict[str, dict[str, object]] = {}
    contact_coordinate_advantage_rows: dict[str, dict[str, object]] = {}
    for topology_index, topology_artifact in enumerate(artifacts):
        if (
            topology_artifact.metadata.get("pi_l_checkpoint_sha256")
            != behavior_checkpoint_sha256
        ):
            raise SchemaValidationError("Order9 tensor PPO behavior checkpoint differs")
        key = _topology_key(topology_artifact, topology_index)
        replay = _TensorReplay(
            topology_artifact, policy=policy, physical_model=physical_model
        )
        sequences = tuple(
            _sequences(topology_artifact, sequence_length=sequence_length)
        )
        exact_rows[key] = _validate_exact_replay(
            policy,
            replay,
            sequences,
            sequence_batch_size=exact_replay_sequence_batch_size,
        )
        advantages, returns = _tensor_gae(topology_artifact, config)
        normalized, advantage_metadata = _normalize_advantages(
            advantages,
            valid=topology_artifact.tensors["valid"],
            phase_index=topology_artifact.tensors["phase_index"],
            phase_local=config.phase_normalized_advantages,
        )
        advantage_rows[key] = advantage_metadata
        if factorized_actor_credit_enabled:
            component_rewards, reconstruction_error = (
                factorize_order9_actor_reward(
                    reward=topology_artifact.tensors["reward"].float(),
                    reward_terms=topology_artifact.tensors["reward_terms"].float(),
                    reward_term_names=topology_artifact.metadata["reward_term_names"],
                    valid=topology_artifact.tensors["valid"],
                )
            )
            component_advantages: dict[str, torch.Tensor] = {}
            component_metadata: dict[str, object] = {
                "maximum_reward_reconstruction_error": reconstruction_error,
                "channels": {},
            }
            raw_component_advantages: list[torch.Tensor] = []
            raw_component_advantages_by_name: dict[str, torch.Tensor] = {}
            for channel in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS:
                channel_advantage, _ = _tensor_gae(
                    topology_artifact,
                    config,
                    reward_override=component_rewards[channel],
                    value_scale=(1.0 / len(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)),
                )
                channel_normalized, channel_metadata = _normalize_advantages(
                    channel_advantage,
                    valid=topology_artifact.tensors["valid"],
                    phase_index=topology_artifact.tensors["phase_index"],
                    phase_local=config.phase_normalized_advantages,
                )
                component_advantages[channel] = channel_normalized
                raw_component_advantages.append(channel_advantage)
                raw_component_advantages_by_name[channel] = channel_advantage
                component_metadata["channels"][channel] = channel_metadata
            component_sum_error = (
                torch.stack(raw_component_advantages, dim=0).sum(dim=0)
                - advantages
            ).abs()[topology_artifact.tensors["valid"]]
            maximum_component_advantage_sum_error = (
                float(component_sum_error.max().item())
                if component_sum_error.numel()
                else 0.0
            )
            if maximum_component_advantage_sum_error > 2.0e-4:
                raise SchemaValidationError(
                    "Order9 factorized actor advantages do not reconstruct total GAE"
                )
            component_metadata["maximum_advantage_sum_error"] = (
                maximum_component_advantage_sum_error
            )
            replay.factorized_actor_advantages = component_advantages
            factorized_advantage_rows[key] = component_metadata
            if contact_coordinate_credit_enabled:
                coordinate_rewards, coordinate_reward_error = (
                    factorize_order9_contact_coordinate_reward(
                        contact_reward=component_rewards["contact"],
                        reward_terms=(
                            topology_artifact.tensors["reward_terms"].float()
                        ),
                        reward_term_names=(
                            topology_artifact.metadata["reward_term_names"]
                        ),
                        valid=topology_artifact.tensors["valid"],
                    )
                )
                coordinate_advantages: dict[str, torch.Tensor] = {}
                coordinate_metadata: dict[str, object] = {
                    "maximum_contact_reward_reconstruction_error": (
                        coordinate_reward_error
                    ),
                    "channels": {},
                }
                raw_coordinate_advantages: list[torch.Tensor] = []
                for coordinate_channel in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS:
                    coordinate_advantage, _ = _tensor_gae(
                        topology_artifact,
                        config,
                        reward_override=coordinate_rewards[coordinate_channel],
                        value_scale=(
                            1.0
                            / (
                                len(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
                                * len(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
                            )
                        ),
                    )
                    coordinate_normalized, coordinate_channel_metadata = (
                        _normalize_advantages(
                            coordinate_advantage,
                            valid=topology_artifact.tensors["valid"],
                            phase_index=topology_artifact.tensors["phase_index"],
                            phase_local=config.phase_normalized_advantages,
                        )
                    )
                    coordinate_advantages[coordinate_channel] = (
                        coordinate_normalized
                    )
                    raw_coordinate_advantages.append(coordinate_advantage)
                    coordinate_metadata["channels"][coordinate_channel] = (
                        coordinate_channel_metadata
                    )
                coordinate_sum_error = (
                    torch.stack(raw_coordinate_advantages, dim=0).sum(dim=0)
                    - raw_component_advantages_by_name["contact"]
                ).abs()[topology_artifact.tensors["valid"]]
                maximum_coordinate_advantage_sum_error = (
                    float(coordinate_sum_error.max().item())
                    if coordinate_sum_error.numel()
                    else 0.0
                )
                if maximum_coordinate_advantage_sum_error > 2.0e-4:
                    raise SchemaValidationError(
                        "Order9 contact-coordinate advantages do not reconstruct "
                        "contact GAE"
                    )
                coordinate_metadata["maximum_advantage_sum_error"] = (
                    maximum_coordinate_advantage_sum_error
                )
                replay.contact_coordinate_advantages = coordinate_advantages
                contact_coordinate_advantage_rows[key] = coordinate_metadata
        topology_data.append(
            _TopologyReplayData(
                key=key,
                replay=replay,
                sequences=sequences,
                advantages=normalized,
                returns=returns,
            )
        )

    topology_count = len(topology_data)
    transition_budget_per_topology_minibatch = max(
        1,
        config.minibatch_size // topology_count,
    )
    rows: list[dict[str, float]] = []
    optimizer_steps = 0
    optimizer_attempts = 0
    completed_epochs = 0
    early_stop = False
    early_stop_reason: str | None = None
    requested_epochs = (
        config.epochs_per_update
        if fine_tune is None
        else fine_tune.epochs_per_update
    )
    for epoch in range(requested_epochs):
        batches_by_topology: dict[str, list[list[_Sequence]]] = {}
        for topology_index, data in enumerate(topology_data):
            topology_seed = seed + epoch * 1009 + topology_index * 100_003
            ordered = (
                _phase_balanced_sequences(data.sequences, seed=topology_seed)
                if config.phase_balanced_sampling
                else list(data.sequences)
            )
            if not config.phase_balanced_sampling:
                random.Random(topology_seed).shuffle(ordered)
            packed = _transition_budget_sequence_batches(
                ordered,
                transition_budget=transition_budget_per_topology_minibatch,
            )
            if fine_tune is not None:
                packed = _merge_boundary_incomplete_sequence_batches(
                    packed,
                    target_phase_indices=target_phase_indices,
                    allow_target_only=(
                        data.replay.artifact.metadata.get(
                            "c3_train_rollout_mode"
                        )
                        == "continuous_state_inheritance"
                    ),
                )
            batches_by_topology[data.key] = packed
        batch_count = max(len(values) for values in batches_by_topology.values())
        for batch_index in range(batch_count):
            topology_batches = []
            for data in topology_data:
                packed = batches_by_topology[data.key]
                selected = packed[batch_index % len(packed)]
                topology_batches.append(
                    (
                        data.key,
                        data.replay,
                        selected,
                        data.advantages,
                        data.returns,
                    )
                )
            metrics = _topology_stratified_sequence_step(
                policy,
                optimizer,
                topology_batches,
                config=config,
                boundary_fine_tune=fine_tune,
                target_phase_indices=target_phase_indices,
                compression_only_actor_objective=(
                    compression_only_actor_objective
                ),
                c3_action_contract=c3_action_contract,
            )
            rows.append(metrics)
            optimizer_attempts += 1
            optimizer_steps += int(metrics["optimizer_step_applied"])
            if progress_callback is not None:
                progress_callback(
                    optimizer_attempts,
                    {
                        **metrics,
                        "epoch_index": float(epoch),
                        "optimizer_step": float(optimizer_steps),
                        "optimizer_attempt": float(optimizer_attempts),
                    },
                )
            if metrics["optimizer_step_rolled_back"] > 0.5:
                early_stop = True
                early_stop_reason = (
                    "non_target_parent_kl"
                    if metrics["boundary_non_target_kl_violation"] > 0.5
                    else "maximum_topology_phase_kl"
                )
                break
            stop_kl = metrics[
                "maximum_phase_kl" if config.phase_local_kl else "approximate_kl"
            ]
            if stop_kl > config.target_kl:
                early_stop = True
                early_stop_reason = "phase_kl"
                break
        completed_epochs += 1
        if early_stop:
            break
    contact_head_extra_rows: list[dict[str, float]] = []
    completed_contact_head_extra_passes = 0
    if not early_stop and contact_head_extra_optimizer_passes > 0:
        assert fine_tune is not None and c3_action_contract is not None
        for pass_index in range(contact_head_extra_optimizer_passes):
            batches_by_topology = {}
            for topology_index, data in enumerate(topology_data):
                topology_seed = (
                    seed
                    + 10_000_019
                    + pass_index * 1009
                    + topology_index * 100_003
                )
                ordered = (
                    _phase_balanced_sequences(
                        data.sequences, seed=topology_seed
                    )
                    if config.phase_balanced_sampling
                    else list(data.sequences)
                )
                if not config.phase_balanced_sampling:
                    random.Random(topology_seed).shuffle(ordered)
                packed = _transition_budget_sequence_batches(
                    ordered,
                    transition_budget=(
                        transition_budget_per_topology_minibatch
                    ),
                )
                packed = _merge_boundary_incomplete_sequence_batches(
                    packed,
                    target_phase_indices=target_phase_indices,
                    allow_target_only=(
                        data.replay.artifact.metadata.get(
                            "c3_train_rollout_mode"
                        )
                        == "continuous_state_inheritance"
                    ),
                )
                batches_by_topology[data.key] = packed
            batch_count = max(
                len(values) for values in batches_by_topology.values()
            )
            for batch_index in range(batch_count):
                topology_batches = []
                for data in topology_data:
                    packed = batches_by_topology[data.key]
                    selected = packed[batch_index % len(packed)]
                    topology_batches.append(
                        (
                            data.key,
                            data.replay,
                            selected,
                            data.advantages,
                            data.returns,
                        )
                    )
                metrics = _topology_stratified_contact_head_sequence_step(
                    policy,
                    optimizer,
                    topology_batches,
                    config=config,
                    boundary_fine_tune=fine_tune,
                    target_phase_indices=target_phase_indices,
                    c3_action_contract=c3_action_contract,
                )
                contact_head_extra_rows.append(metrics)
                optimizer_attempts += 1
                optimizer_steps += int(metrics["optimizer_step_applied"])
                if progress_callback is not None:
                    progress_callback(
                        optimizer_attempts,
                        {
                            **metrics,
                            "epoch_index": float(requested_epochs),
                            "contact_head_extra_pass_index": float(
                                pass_index
                            ),
                            "optimizer_step": float(optimizer_steps),
                            "optimizer_attempt": float(optimizer_attempts),
                        },
                    )
                if metrics["optimizer_step_rolled_back"] > 0.5:
                    early_stop = True
                    early_stop_reason = (
                        "contact_head_extra_non_target_parent_kl"
                        if metrics[
                            "boundary_non_target_kl_violation"
                        ]
                        > 0.5
                        else "contact_head_extra_maximum_topology_phase_kl"
                    )
                    break
                stop_kl = metrics[
                    "maximum_phase_kl"
                    if config.phase_local_kl
                    else "approximate_kl"
                ]
                if stop_kl > config.target_kl:
                    early_stop = True
                    early_stop_reason = "contact_head_extra_phase_kl"
                    break
            completed_contact_head_extra_passes += 1
            if early_stop:
                break
    all_optimizer_rows = [*rows, *contact_head_extra_rows]
    observed_phases = sorted(
        {
            sequence.phase_index
            for data in topology_data
            for sequence in data.sequences
        }
    )
    phase_kl_means = {}
    for phase in observed_phases:
        name = f"approximate_kl_phase_{phase}"
        phase_rows = [row[name] for row in rows if name in row]
        if not phase_rows:
            raise SchemaValidationError(
                f"Order9 tensor PPO never sampled actor phase {phase}"
            )
        phase_kl_means[str(phase)] = sum(phase_rows) / len(phase_rows)
    common_metric_names = set(rows[0]).intersection(
        *(set(row) for row in rows[1:])
    )
    summary_rows = [
        {name: row[name] for name in common_metric_names}
        for row in rows
    ]
    exact = _aggregate_exact_replay(exact_rows)
    advantage_metadata = _aggregate_advantage_metadata(advantage_rows)
    return _ppo_result(
        "pi_l",
        sample_count=sum(item.environment_step_count for item in artifacts),
        requested_epochs=requested_epochs,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        rows=summary_rows,
        early_stop=early_stop,
        behavior_checkpoint_sha256=behavior_checkpoint_sha256,
        metadata={
            "tensor_native_ppo_version": ORDER9_TENSOR_PI_L_PPO_VERSION,
            "tensor_native_rollout": True,
            "record_schema_materialization": False,
            "jsonl_materialization": False,
            "sequence_length": sequence_length,
            "recurrent_replay": True,
            "timestep_batched_active_sequences": True,
            "phase_balanced_sampling": bool(config.phase_balanced_sampling),
            "phase_normalized_advantages": bool(
                config.phase_normalized_advantages
            ),
            "phase_local_kl": bool(config.phase_local_kl),
            "critic_actor_state_gradient_detached": True,
            "autograd_multithreading_enabled": bool(
                torch.autograd.is_multithreading_enabled()
            ),
            "topology_stratified_update": topology_count > 1,
            "compression_only_actor_objective": bool(
                compression_only_actor_objective
            ),
            "joint_head_only_actor_update": bool(
                joint_head_only_actor_update
            ),
            "contact_residual_only_actor_update": bool(
                contact_residual_only_actor_update
            ),
            "c3_action_contract": c3_action_contract,
            "factorized_actor_credit_enabled": factorized_actor_credit_enabled,
            "factorized_actor_credit_version": (
                ORDER9_FACTORIZED_ACTOR_CREDIT_VERSION
                if factorized_actor_credit_enabled
                else None
            ),
            "factorized_actor_credit_channels": (
                list(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
                if factorized_actor_credit_enabled
                else []
            ),
            "factorized_actor_credit_term_weights": (
                {
                    name: list(weights)
                    for name, weights in (
                        ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS.items()
                    )
                }
                if factorized_actor_credit_enabled
                else {}
            ),
            "factorized_actor_credit_component_baseline_share": (
                1.0 / len(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
                if factorized_actor_credit_enabled
                else None
            ),
            "factorized_actor_credit_advantage_metadata_by_topology": (
                factorized_advantage_rows
            ),
            "contact_coordinate_credit_enabled": (
                contact_coordinate_credit_enabled
            ),
            "contact_coordinate_credit_version": (
                ORDER9_CONTACT_COORDINATE_CREDIT_VERSION
                if contact_coordinate_credit_enabled
                else None
            ),
            "contact_coordinate_credit_channels": (
                list(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
                if contact_coordinate_credit_enabled
                else []
            ),
            "contact_coordinate_credit_action_indices": (
                {
                    name: list(indices)
                    for name, indices in (
                        ORDER9_CONTACT_COORDINATE_ACTION_INDICES.items()
                    )
                }
                if contact_coordinate_credit_enabled
                else {}
            ),
            "contact_coordinate_credit_term_weights": (
                {
                    name: list(weights)
                    for name, weights in (
                        ORDER9_CONTACT_COORDINATE_CREDIT_TERM_WEIGHTS.items()
                    )
                }
                if contact_coordinate_credit_enabled
                else {}
            ),
            "contact_coordinate_credit_advantage_metadata_by_topology": (
                contact_coordinate_advantage_rows
            ),
            "contact_head_extra_optimizer_contract": (
                ORDER9_CONTACT_HEAD_EXTRA_OPTIMIZER_CONTRACT
                if contact_head_extra_optimizer_passes > 0
                else None
            ),
            "contact_head_extra_optimizer_passes_requested": (
                contact_head_extra_optimizer_passes
            ),
            "contact_head_extra_optimizer_passes_completed": (
                completed_contact_head_extra_passes
            ),
            "contact_head_extra_optimizer_parameter_prefixes": (
                list(ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES)
                if contact_head_extra_optimizer_passes > 0
                else []
            ),
            "contact_head_extra_optimizer_step_count": sum(
                int(row["optimizer_step_applied"])
                for row in contact_head_extra_rows
            ),
            "contact_head_extra_optimizer_attempt_count": len(
                contact_head_extra_rows
            ),
            "contact_head_extra_actor_loss_mean": (
                sum(
                    row["actor_loss_contact_extra"]
                    for row in contact_head_extra_rows
                )
                / len(contact_head_extra_rows)
                if contact_head_extra_rows
                else None
            ),
            "contact_head_extra_gradient_norm_mean": (
                sum(
                    row["contact_head_extra_gradient_norm"]
                    for row in contact_head_extra_rows
                )
                / len(contact_head_extra_rows)
                if contact_head_extra_rows
                else None
            ),
            "contact_head_extra_coordinate_actor_loss_mean": {
                name: (
                    sum(
                        row[f"actor_loss_contact_extra_{name}"]
                        for row in contact_head_extra_rows
                    )
                    / len(contact_head_extra_rows)
                    if contact_head_extra_rows
                    else None
                )
                for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
            },
            "topology_gradient_surgery_enabled": bool(
                fine_tune is not None
                and fine_tune.topology_gradient_surgery_enabled
            ),
            "topology_gradient_surgery_method": (
                "deterministic_all_pairs_pcgrad"
                if fine_tune is not None
                and fine_tune.topology_gradient_surgery_enabled
                else None
            ),
            "topology_gradient_conflict_pair_count_mean": sum(
                row["topology_gradient_conflict_pair_count"] for row in rows
            )
            / len(rows),
            "topology_gradient_conflict_pair_count_observed": max(
                row["topology_gradient_conflict_pair_count"] for row in rows
            ),
            "topology_shard_count": topology_count,
            "phase_reset_shard_count": (
                topology_count - len(state_inheritance_artifacts)
            ),
            "state_inheritance_shard_count": len(
                state_inheritance_artifacts
            ),
            "state_inheritance_environment_step_count": sum(
                item.environment_step_count
                for item in state_inheritance_artifacts
            ),
            "state_inheritance_gae_cross_phase_preserved": bool(
                state_inheritance_artifacts
            ),
            "state_inheritance_recurrent_inputs_preserved": bool(
                state_inheritance_artifacts
            ),
            "state_inheritance_target_only_minibatches_anchored_by_phase_reset_shards": bool(
                state_inheritance_artifacts
            ),
            "topology_keys": [data.key for data in topology_data],
            "topology_module_counts": [
                data.replay.builder.module_count for data in topology_data
            ],
            "topology_minibatch_packing": (
                ORDER9_TOPOLOGY_MINIBATCH_PACKING
            ),
            "transition_budget_per_topology_minibatch": (
                transition_budget_per_topology_minibatch
            ),
            "sequence_phase_count": {
                str(phase): sum(
                    sequence.phase_index == phase
                    for data in topology_data
                    for sequence in data.sequences
                )
                for phase in observed_phases
            },
            "phase_kl_mean_by_actor_phase": phase_kl_means,
            "maximum_phase_kl_observed": max(
                row["maximum_phase_kl"] for row in all_optimizer_rows
            ),
            "maximum_phase_kl_mean": sum(
                row["maximum_phase_kl"] for row in all_optimizer_rows
            )
            / len(all_optimizer_rows),
            "maximum_topology_phase_kl_observed": max(
                row["maximum_topology_phase_kl"]
                for row in all_optimizer_rows
            ),
            "maximum_topology_phase_kl_mean": sum(
                row["maximum_topology_phase_kl"]
                for row in all_optimizer_rows
            )
            / len(all_optimizer_rows),
            "optimizer_attempt_count": optimizer_attempts,
            "optimizer_rollback_count": sum(
                int(row["optimizer_step_rolled_back"])
                for row in all_optimizer_rows
            ),
            "privileged_compression_teacher_loss_mean": sum(
                row["privileged_compression_teacher_loss"] for row in rows
            )
            / len(rows),
            "privileged_compression_teacher_sample_count_mean": sum(
                row["privileged_compression_teacher_sample_count"]
                for row in rows
            )
            / len(rows),
            "privileged_wrench_satisfied_parent_kl_mean": sum(
                row["privileged_wrench_satisfied_parent_kl"] for row in rows
            )
            / len(rows),
            "privileged_wrench_satisfied_sample_count_mean": sum(
                row["privileged_wrench_satisfied_sample_count"]
                for row in rows
            )
            / len(rows),
            "contact_normal_preload_deficit_teacher_loss_mean": sum(
                row["contact_normal_preload_deficit_teacher_loss"]
                for row in rows
            )
            / len(rows),
            "contact_normal_preload_deficit_teacher_sample_count_mean": sum(
                row["contact_normal_preload_deficit_teacher_sample_count"]
                for row in rows
            )
            / len(rows),
            "early_stop_reason": early_stop_reason,
            **_boundary_fine_tune_metadata(
                fine_tune=fine_tune,
                target_phase_indices=target_phase_indices,
                artifacts=artifacts,
                rows=all_optimizer_rows,
                behavior_checkpoint_sha256=behavior_checkpoint_sha256,
            ),
            **advantage_metadata,
            "exact_behavior_replay_validated": True,
            **exact,
        },
    )


def _sequence_ppo_step(
    policy: Order9PhaseConditionedActorCritic,
    replay: _TensorReplay,
    sequences: Sequence[_Sequence],
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: Order9PPOOptimizationConfig,
) -> dict[str, float]:
    return _topology_stratified_sequence_step(
        policy,
        optimizer,
        [("single", replay, sequences, advantages, returns)],
        config=config,
    )


def _sequence_ppo_terms(
    policy: Order9PhaseConditionedActorCritic,
    replay: _TensorReplay,
    sequences: Sequence[_Sequence],
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: Order9PPOOptimizationConfig,
    boundary_fine_tune: Order9C3BoundaryFineTuneConfig | None = None,
    target_phase_indices: Sequence[int] = (),
    compression_only_actor_objective: bool = False,
    c3_action_contract: str | None = None,
) -> _SequencePPOTerms:
    if not sequences:
        raise SchemaValidationError("Order9 tensor PPO sequence batch is empty")
    first = replay.batch(
        [item.start for item in sequences],
        [item.environment for item in sequences],
    )
    hidden = list(first.recurrent_state_in.unbind(0))
    previous = list(first.previous_global_action.unbind(0))
    log_probs: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    advantage_rows: list[torch.Tensor] = []
    return_rows: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    phase_rows: list[torch.Tensor] = []
    compression_mean_rows: list[torch.Tensor] = []
    compression_log_prob_rows: list[torch.Tensor] = []
    contract_global_log_prob_rows: list[torch.Tensor] = []
    contract_entropy_rows: list[torch.Tensor] = []
    compression_behavior_rows: list[torch.Tensor] = []
    compression_teacher_direction_rows: list[torch.Tensor] = []
    compression_teacher_mask_rows: list[torch.Tensor] = []
    compression_teacher_satisfied_mask_rows: list[torch.Tensor] = []
    preload_deficit_category_logits_rows: list[torch.Tensor] = []
    preload_deficit_category_index_rows: list[torch.Tensor] = []
    preload_deficit_teacher_mask_rows: list[torch.Tensor] = []
    preload_deficit_contact_slot_mask_rows: list[torch.Tensor] = []
    component_new_log_prob_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    }
    component_old_log_prob_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    }
    component_advantage_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    }
    contact_coordinate_new_log_prob_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
    }
    contact_coordinate_old_log_prob_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
    }
    contact_coordinate_advantage_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
    }
    factorized_actor_credit_enabled = bool(
        boundary_fine_tune is not None
        and boundary_fine_tune.factorized_actor_credit_enabled
    )
    contact_coordinate_credit_enabled = bool(
        boundary_fine_tune is not None
        and boundary_fine_tune.contact_coordinate_credit_enabled
    )
    preload_deficit_teacher_enabled = bool(
        boundary_fine_tune is not None
        and boundary_fine_tune.contact_normal_preload_deficit_teacher_weight
        > 0.0
    )
    if factorized_actor_credit_enabled and (
        replay.behavior_component_log_probs is None
        or replay.factorized_actor_advantages is None
    ):
        raise SchemaValidationError(
            "Order9 factorized actor credit replay cache is unavailable"
        )
    if contact_coordinate_credit_enabled and (
        replay.behavior_contact_coordinate_log_probs is None
        or replay.contact_coordinate_advantages is None
        or not isinstance(policy, Order9ContactSpacePhaseConditionedActorCritic)
    ):
        raise SchemaValidationError(
            "Order9 contact-coordinate credit replay cache is unavailable"
        )
    if preload_deficit_teacher_enabled and not isinstance(
        policy, Order9CategoricalContactNormalPhaseConditionedActorCritic
    ):
        raise SchemaValidationError(
            "Order9 preload-deficit categorical teacher requires the "
            "categorical contact-normal policy"
        )
    for timestep in range(max(item.length for item in sequences)):
        active = [index for index, item in enumerate(sequences) if timestep < item.length]
        times = [sequences[index].start + timestep for index in active]
        environments = [sequences[index].environment for index in active]
        batch = replay.batch(
            times,
            environments,
            advantages=advantages,
            returns=returns,
        )
        step = policy.step(
            batch.graph,
            None,
            batch.actor_features,
            torch.stack([previous[index] for index in active]),
            torch.stack([hidden[index] for index in active]),
            phase_features=batch.phase_features,
            privileged_disturbance_body=batch.privileged,
            action=batch.global_action,
            joint_action=batch.joint_action,
            **(
                {
                    "contact_compression_residual_action": (
                        batch.contact_compression_residual_action
                    )
                }
                if isinstance(
                    policy, Order9MorphologyInvariantCompressionActorCritic
                )
                else {}
            ),
            **(
                {
                    "contact_space_residual_action": (
                        batch.contact_space_residual_action
                    ),
                    "contact_slot_features": batch.contact_slot_features,
                    "contact_slot_owner_module_indices": (
                        batch.contact_slot_owner_module_indices
                    ),
                    "contact_slot_mask": batch.contact_slot_mask,
                }
                if isinstance(
                    policy, Order9ContactSpacePhaseConditionedActorCritic
                )
                else {}
            ),
            active_knot_features=batch.active_knot_features,
            active_assignment_features=batch.active_assignment_features,
        )
        log_probs.extend(step.log_prob.unbind(0))
        old_log_probs.extend(batch.old_log_prob.unbind(0))
        values.extend(step.value.unbind(0))
        entropies.extend(step.entropy.unbind(0))
        phase_rows.extend(batch.phase_index.unbind(0))
        if factorized_actor_credit_enabled:
            current_component_rows = {
                "contact": step.contact_log_prob,
                "centroidal": step.centroidal_log_prob,
                "posture": step.posture_log_prob,
            }
            time_index = torch.tensor(times, dtype=torch.long)
            environment_index = torch.tensor(environments, dtype=torch.long)
            assert replay.behavior_component_log_probs is not None
            assert replay.factorized_actor_advantages is not None
            for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS:
                component_new_log_prob_rows[name].extend(
                    current_component_rows[name].unbind(0)
                )
                component_old_log_prob_rows[name].extend(
                    replay.behavior_component_log_probs[name][
                        time_index, environment_index
                    ].to(
                        device=replay.device,
                        dtype=replay.dtype,
                        non_blocking=True,
                    ).unbind(0)
                )
                component_advantage_rows[name].extend(
                    replay.factorized_actor_advantages[name][
                        time_index, environment_index
                    ].to(
                        device=replay.device,
                        dtype=replay.dtype,
                        non_blocking=True,
                    ).unbind(0)
                )
            if contact_coordinate_credit_enabled:
                current_coordinate_rows = _contact_coordinate_group_log_probs(
                    bounded_action=batch.contact_space_residual_action,
                    bounded_mean=step.contact_space_residual_action_mean,
                    log_std=policy.contact_space_actor_log_std,
                    slot_mask=batch.contact_slot_mask,
                    categorical_contact_log_prob=(
                        step.contact_log_prob
                        if isinstance(
                            policy,
                            Order9CategoricalContactNormalPhaseConditionedActorCritic,
                        )
                        else None
                    ),
                )
                assert replay.behavior_contact_coordinate_log_probs is not None
                assert replay.contact_coordinate_advantages is not None
                for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS:
                    contact_coordinate_new_log_prob_rows[name].extend(
                        current_coordinate_rows[name].unbind(0)
                    )
                    contact_coordinate_old_log_prob_rows[name].extend(
                        replay.behavior_contact_coordinate_log_probs[name][
                            time_index, environment_index
                        ].to(
                            device=replay.device,
                            dtype=replay.dtype,
                            non_blocking=True,
                        ).unbind(0)
                    )
                    contact_coordinate_advantage_rows[name].extend(
                        replay.contact_coordinate_advantages[name][
                            time_index, environment_index
                        ].to(
                            device=replay.device,
                            dtype=replay.dtype,
                            non_blocking=True,
                        ).unbind(0)
                    )
        morphology_invariant = isinstance(
            policy, Order9MorphologyInvariantCompressionActorCritic
        )
        compression_mean = (
            step.contact_compression_residual_action_mean
            if morphology_invariant
            else step.joint_action_mean[
                :,
                batch.compression_control_module_index,
                batch.compression_control_joint_index,
            ]
        )
        compression_mean_rows.extend(compression_mean.unbind(0))
        compression_action = (
            batch.contact_compression_residual_action
            if morphology_invariant
            else batch.joint_action[
                :,
                batch.compression_control_module_index,
                batch.compression_control_joint_index,
            ]
        ).clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
        compression_log_std = (
            policy.contact_compression_actor_log_std.reshape(1)
            if morphology_invariant
            else policy.joint_actor_log_std[
                batch.compression_control_joint_index
            ].reshape(1)
        )
        compression_log_prob, compression_entropy = (
            _squashed_coordinate_log_prob_and_entropy(
                bounded_action=compression_action.reshape(-1, 1),
                bounded_mean=compression_mean.reshape(-1, 1),
                log_std=compression_log_std,
            )
        )
        compression_log_prob_rows.extend(compression_log_prob.unbind(0))
        global_dimension = (
            0
            if c3_action_contract is None
            else order9_c3_action_contract_global_dimension(c3_action_contract)
        )
        if global_dimension:
            bounded_global = batch.global_action[:, :global_dimension]
            bounded_global_mean = step.action_mean[:, :global_dimension]
            global_log_prob, global_entropy = (
                _squashed_coordinate_log_prob_and_entropy(
                    bounded_action=bounded_global,
                    bounded_mean=bounded_global_mean,
                    log_std=policy.actor_log_std[:global_dimension],
                )
            )
            contract_global_log_prob_rows.extend(global_log_prob.unbind(0))
            contract_entropy_rows.extend(
                (compression_entropy + global_entropy).unbind(0)
            )
        else:
            contract_global_log_prob_rows.extend(
                torch.zeros_like(compression_entropy).unbind(0)
            )
            contract_entropy_rows.extend(compression_entropy.unbind(0))
        compression_behavior_rows.extend(compression_action.unbind(0))
        compression_teacher_direction_rows.extend(
            batch.compression_teacher_direction.unbind(0)
        )
        compression_teacher_mask_rows.extend(
            batch.compression_teacher_mask.unbind(0)
        )
        compression_teacher_satisfied_mask_rows.extend(
            batch.compression_teacher_satisfied_mask.unbind(0)
        )
        if preload_deficit_teacher_enabled:
            if step.contact_normal_category_logits is None:
                raise SchemaValidationError(
                    "Order9 categorical contact-normal logits are unavailable"
                )
            preload_deficit_category_logits_rows.extend(
                step.contact_normal_category_logits.unbind(0)
            )
            preload_deficit_category_index_rows.extend(
                batch.contact_normal_preload_deficit_teacher_category_index.unbind(0)
            )
            preload_deficit_teacher_mask_rows.extend(
                batch.contact_normal_preload_deficit_teacher_mask.unbind(0)
            )
            preload_deficit_contact_slot_mask_rows.extend(
                batch.contact_slot_mask.unbind(0)
            )
        assert batch.advantage is not None and batch.return_ is not None
        advantage_rows.extend(batch.advantage.unbind(0))
        return_rows.extend(batch.return_.unbind(0))
        for row, sequence_index in enumerate(active):
            hidden[sequence_index] = step.recurrent_state[row]
            previous[sequence_index] = batch.applied_global_action[row].detach()
    new_log_prob = torch.stack(log_probs)
    old_log_prob = torch.stack(old_log_probs)
    new_values = torch.stack(values)
    advantage_tensor = torch.stack(advantage_rows)
    return_tensor = torch.stack(return_rows)
    entropy = torch.stack(entropies)
    phase_tensor = torch.stack(phase_rows).long()
    compression_mean = torch.stack(compression_mean_rows)
    compression_log_prob = torch.stack(compression_log_prob_rows)
    contract_global_log_prob = torch.stack(contract_global_log_prob_rows)
    contract_entropy = torch.stack(contract_entropy_rows)
    compression_behavior = torch.stack(compression_behavior_rows)
    compression_teacher_direction = torch.stack(
        compression_teacher_direction_rows
    )
    compression_teacher_mask = torch.stack(
        compression_teacher_mask_rows
    ).bool()
    compression_teacher_satisfied_mask = torch.stack(
        compression_teacher_satisfied_mask_rows
    ).bool()
    component_new_log_probs = (
        {
            name: torch.stack(values)
            for name, values in component_new_log_prob_rows.items()
        }
        if factorized_actor_credit_enabled
        else None
    )
    component_old_log_probs = (
        {
            name: torch.stack(values)
            for name, values in component_old_log_prob_rows.items()
        }
        if factorized_actor_credit_enabled
        else None
    )
    component_advantages = (
        {
            name: torch.stack(values)
            for name, values in component_advantage_rows.items()
        }
        if factorized_actor_credit_enabled
        else None
    )
    contact_coordinate_new_log_probs = (
        {
            name: torch.stack(values)
            for name, values in contact_coordinate_new_log_prob_rows.items()
        }
        if contact_coordinate_credit_enabled
        else None
    )
    contact_coordinate_old_log_probs = (
        {
            name: torch.stack(values)
            for name, values in contact_coordinate_old_log_prob_rows.items()
        }
        if contact_coordinate_credit_enabled
        else None
    )
    contact_coordinate_advantages = (
        {
            name: torch.stack(values)
            for name, values in contact_coordinate_advantage_rows.items()
        }
        if contact_coordinate_credit_enabled
        else None
    )
    component_actor_losses: dict[str, torch.Tensor] | None = None
    contact_coordinate_losses: dict[str, torch.Tensor] | None = None
    target_mask = torch.ones_like(phase_tensor, dtype=torch.bool)
    non_target_parent_kl = new_log_prob.sum() * 0.0
    privileged_compression_teacher_loss = new_log_prob.sum() * 0.0
    privileged_compression_teacher_sample_count = 0
    privileged_wrench_satisfied_parent_kl = new_log_prob.sum() * 0.0
    privileged_wrench_satisfied_sample_count = 0
    contact_normal_preload_deficit_teacher_loss = new_log_prob.sum() * 0.0
    contact_normal_preload_deficit_teacher_sample_count = 0
    if boundary_fine_tune is None:
        actor = clipped_ppo_surrogate_loss(
            new_log_prob=new_log_prob,
            old_log_prob=old_log_prob,
            advantages=advantage_tensor,
            clip_ratio=config.clip_ratio,
        )
        total = compute_order9_pi_l_ppo_loss(
            new_log_prob=new_log_prob,
            old_log_prob=old_log_prob,
            advantages=advantage_tensor,
            new_values=new_values,
            returns=return_tensor,
            entropy=entropy,
            clip_ratio=config.clip_ratio,
            value_loss_weight=config.value_loss_weight,
            entropy_bonus_weight=config.entropy_bonus_weight,
        )
    else:
        target_mask, non_target_mask = _boundary_phase_masks(
            phase_tensor,
            target_phase_indices,
            allow_target_only=(
                replay.artifact.metadata.get("c3_train_rollout_mode")
                == "continuous_state_inheritance"
            ),
        )
        if contact_coordinate_credit_enabled:
            assert contact_coordinate_new_log_probs is not None
            assert contact_coordinate_old_log_probs is not None
            assert contact_coordinate_advantages is not None
            _, contact_coordinate_losses = _named_factorized_clipped_actor_loss(
                channel_names=ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS,
                new_log_probs=contact_coordinate_new_log_probs,
                old_log_probs=contact_coordinate_old_log_probs,
                advantages=contact_coordinate_advantages,
                mask=target_mask,
                clip_ratio=config.clip_ratio,
                label="contact-coordinate",
            )
        if factorized_actor_credit_enabled:
            assert component_new_log_probs is not None
            assert component_old_log_probs is not None
            assert component_advantages is not None
            actor, component_actor_losses = _factorized_clipped_actor_loss(
                new_log_probs=component_new_log_probs,
                old_log_probs=component_old_log_probs,
                advantages=component_advantages,
                mask=target_mask,
                clip_ratio=config.clip_ratio,
            )
            if contact_coordinate_credit_enabled:
                assert contact_coordinate_losses is not None
                # Replace the coarse all-contact density objective with its
                # normal/tangent/rotation objectives.  This is the ordinary
                # PPO pass: no extra update or duplicated rollout is needed.
                component_actor_losses["contact"] = (
                    _weighted_contact_coordinate_actor_loss(
                        contact_coordinate_losses,
                        normal_translation_weight=(
                            boundary_fine_tune.
                            contact_normal_translation_actor_loss_weight
                        ),
                    )
                )
                actor = torch.stack(
                    [
                        component_actor_losses[name]
                        for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
                    ]
                ).mean()
        elif c3_action_contract is not None:
            if order9_c3_action_contract_uses_full_policy(c3_action_contract):
                actor = clipped_ppo_surrogate_loss(
                    new_log_prob=new_log_prob[target_mask],
                    old_log_prob=old_log_prob[target_mask],
                    advantages=advantage_tensor[target_mask],
                    clip_ratio=config.clip_ratio,
                )
            else:
                contracted_log_prob = (
                    compression_log_prob + contract_global_log_prob
                )
                actor = -(
                    contracted_log_prob[target_mask]
                    * advantage_tensor[target_mask].detach()
                ).mean()
        elif compression_only_actor_objective:
            # One on-policy epoch and the existing topology/phase KL gates make
            # this an unclipped, action-specific policy-gradient objective.
            # It removes unrelated PolicyCommand coordinates from wrench-range
            # credit assignment without adding any actor observation.
            actor = -(
                compression_log_prob[target_mask]
                * advantage_tensor[target_mask].detach()
            ).mean()
        else:
            actor = clipped_ppo_surrogate_loss(
                new_log_prob=new_log_prob[target_mask],
                old_log_prob=old_log_prob[target_mask],
                advantages=advantage_tensor[target_mask],
                clip_ratio=config.clip_ratio,
            )
        critic = torch.nn.functional.mse_loss(new_values, return_tensor)
        if bool(non_target_mask.any()):
            log_ratio = (
                new_log_prob[non_target_mask] - old_log_prob[non_target_mask]
            )
            ratio = torch.exp(log_ratio)
            non_target_parent_kl = ((ratio - 1.0) - log_ratio).mean()
        total = (
            actor
            + config.value_loss_weight * critic
            - config.entropy_bonus_weight
            * (
                entropy[target_mask].mean()
                if (
                    c3_action_contract is None
                    or order9_c3_action_contract_uses_full_policy(
                        c3_action_contract
                    )
                )
                else contract_entropy[target_mask].mean()
            )
            + boundary_fine_tune.non_target_parent_kl_weight
            * non_target_parent_kl
        )
        satisfied_mask = target_mask & compression_teacher_satisfied_mask
        if (
            boundary_fine_tune.privileged_wrench_satisfied_parent_kl_weight
            > 0.0
            and bool(satisfied_mask.any())
        ):
            satisfied_log_ratio = new_log_prob[satisfied_mask] - old_log_prob[
                satisfied_mask
            ]
            satisfied_ratio = torch.exp(satisfied_log_ratio)
            privileged_wrench_satisfied_parent_kl = (
                (satisfied_ratio - 1.0) - satisfied_log_ratio
            ).mean()
            privileged_wrench_satisfied_sample_count = int(
                satisfied_mask.sum().item()
            )
            total = total + (
                float(
                    boundary_fine_tune.
                    privileged_wrench_satisfied_parent_kl_weight
                )
                * privileged_wrench_satisfied_parent_kl
            )
        teacher_mask = target_mask & compression_teacher_mask
        teacher_direction = compression_teacher_direction
        if boundary_fine_tune.privileged_compression_teacher_underforce_only:
            teacher_direction = teacher_direction.clamp_min(0.0)
            teacher_mask &= teacher_direction > 1.0e-6
        if (
            boundary_fine_tune.privileged_compression_teacher_weight > 0.0
            and bool(teacher_mask.any())
        ):
            teacher_target = torch.clamp(
                compression_behavior
                + float(
                    boundary_fine_tune.
                    privileged_compression_teacher_action_step
                )
                * teacher_direction,
                min=-1.0,
                max=1.0,
            )
            privileged_compression_teacher_loss = (
                torch.nn.functional.smooth_l1_loss(
                    compression_mean[teacher_mask],
                    teacher_target[teacher_mask],
                )
            )
            privileged_compression_teacher_sample_count = int(
                teacher_mask.sum().item()
            )
            total = total + (
                float(
                    boundary_fine_tune.
                    privileged_compression_teacher_weight
                )
                * privileged_compression_teacher_loss
            )
        if preload_deficit_teacher_enabled:
            (
                contact_normal_preload_deficit_teacher_loss,
                contact_normal_preload_deficit_teacher_sample_count,
            ) = _categorical_preload_deficit_teacher_loss(
                category_logits=torch.stack(
                    preload_deficit_category_logits_rows
                ),
                category_index=torch.stack(
                    preload_deficit_category_index_rows
                ).long(),
                environment_mask=torch.stack(
                    preload_deficit_teacher_mask_rows
                ).bool(),
                contact_slot_mask=torch.stack(
                    preload_deficit_contact_slot_mask_rows
                ).bool(),
                target_phase_mask=target_mask,
            )
            total = total + (
                float(
                    boundary_fine_tune.
                    contact_normal_preload_deficit_teacher_weight
                )
                * contact_normal_preload_deficit_teacher_loss
            )
    return _SequencePPOTerms(
        total=total,
        actor_loss=actor,
        actor_component_losses=component_actor_losses,
        actor_component_advantages=component_advantages,
        actor_component_new_log_probs=component_new_log_probs,
        actor_component_old_log_probs=component_old_log_probs,
        contact_coordinate_losses=contact_coordinate_losses,
        contact_coordinate_advantages=contact_coordinate_advantages,
        contact_coordinate_new_log_probs=contact_coordinate_new_log_probs,
        contact_coordinate_old_log_probs=contact_coordinate_old_log_probs,
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        new_values=new_values,
        advantages=advantage_tensor,
        returns=return_tensor,
        entropy=entropy,
        phase_index=phase_tensor,
        target_phase_mask=target_mask,
        non_target_parent_kl=non_target_parent_kl,
        privileged_compression_teacher_loss=(
            privileged_compression_teacher_loss
        ),
        privileged_compression_teacher_sample_count=(
            privileged_compression_teacher_sample_count
        ),
        privileged_wrench_satisfied_parent_kl=(
            privileged_wrench_satisfied_parent_kl
        ),
        privileged_wrench_satisfied_sample_count=(
            privileged_wrench_satisfied_sample_count
        ),
        contact_normal_preload_deficit_teacher_loss=(
            contact_normal_preload_deficit_teacher_loss
        ),
        contact_normal_preload_deficit_teacher_sample_count=(
            contact_normal_preload_deficit_teacher_sample_count
        ),
    )


def _boundary_phase_masks(
    phase_index: torch.Tensor,
    target_phase_indices: Sequence[int],
    *,
    allow_target_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(phase_index, dtype=torch.bool)
    for phase in target_phase_indices:
        target |= phase_index == int(phase)
    anchor = ~target
    if not bool(target.any()) or (not bool(anchor.any()) and not allow_target_only):
        raise SchemaValidationError(
            "Order9 C3 boundary minibatch requires target and anchor phases"
        )
    return target, anchor


def _mean_pcgrad(
    topology_gradients: Sequence[Sequence[torch.Tensor]],
    *,
    projection_parameter_mask: Sequence[bool] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Project conflicting topology gradients, then return their mean.

    This is the deterministic all-pairs form of PCGrad.  It changes only the
    training update; actor observations, actions, and deployment remain
    unchanged.
    """

    if not topology_gradients:
        raise ValueError("Order9 PCGrad requires topology gradients")
    parameter_count = len(topology_gradients[0])
    if parameter_count < 1 or any(
        len(values) != parameter_count for values in topology_gradients
    ):
        raise ValueError("Order9 PCGrad topology gradient widths differ")
    projection_mask = (
        tuple(True for _ in range(parameter_count))
        if projection_parameter_mask is None
        else tuple(bool(value) for value in projection_parameter_mask)
    )
    if len(projection_mask) != parameter_count or not any(projection_mask):
        raise ValueError("Order9 PCGrad projection mask is invalid")
    projected_rows: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for source in topology_gradients:
            projected = [value.clone() for value in source]
            for reference in topology_gradients:
                if reference is source:
                    continue
                dot = sum(
                    (left * right).sum()
                    for index, (left, right) in enumerate(
                        zip(projected, reference)
                    )
                    if projection_mask[index]
                )
                if bool(dot < 0.0):
                    norm_squared = sum(
                        right.square().sum()
                        for index, right in enumerate(reference)
                        if projection_mask[index]
                    ).clamp_min(1.0e-12)
                    coefficient = dot / norm_squared
                    projected = [
                        left - coefficient * right
                        if projection_mask[index]
                        else left
                        for index, (left, right) in enumerate(
                            zip(projected, reference)
                        )
                    ]
            projected_rows.append(projected)
        return tuple(
            torch.stack(
                [row[index] for row in projected_rows], dim=0
            ).mean(dim=0)
            for index in range(parameter_count)
        )


def _pcgrad_conflict_pair_count(
    topology_gradients: Sequence[Sequence[torch.Tensor]],
    *,
    projection_parameter_mask: Sequence[bool],
) -> int:
    mask = tuple(bool(value) for value in projection_parameter_mask)
    count = 0
    with torch.no_grad():
        for left_index, left in enumerate(topology_gradients):
            for right in topology_gradients[left_index + 1 :]:
                dot = sum(
                    (first * second).sum()
                    for index, (first, second) in enumerate(zip(left, right))
                    if mask[index]
                )
                count += int(bool(dot < 0.0))
    return count


def _topology_stratified_sequence_step(
    policy: Order9PhaseConditionedActorCritic,
    optimizer: torch.optim.Optimizer,
    topology_batches: Sequence[
        tuple[
            str,
            _TensorReplay,
            Sequence[_Sequence],
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    *,
    config: Order9PPOOptimizationConfig,
    boundary_fine_tune: Order9C3BoundaryFineTuneConfig | None = None,
    target_phase_indices: Sequence[int] = (),
    compression_only_actor_objective: bool = False,
    c3_action_contract: str | None = None,
) -> dict[str, float]:
    if not topology_batches:
        raise SchemaValidationError(
            "Order9 topology-stratified PPO batches are empty"
        )
    optimizer.zero_grad(set_to_none=True)
    topology_count = len(topology_batches)
    named_parameters = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    pcgrad_projection_mask = tuple(
        not name.startswith("critic.") for name, _ in named_parameters
    )
    topology_gradients: list[tuple[torch.Tensor, ...]] = []
    detached_new: list[torch.Tensor] = []
    detached_old: list[torch.Tensor] = []
    detached_phase: list[torch.Tensor] = []
    base_rows: list[dict[str, float]] = []
    topology_phase_metrics: dict[str, dict[str, float]] = {}
    pcgrad_conflict_pair_count = 0
    for key, replay, sequences, advantages, returns in topology_batches:
        item = _sequence_ppo_terms(
            policy,
            replay,
            sequences,
            advantages=advantages,
            returns=returns,
            config=config,
            boundary_fine_tune=boundary_fine_tune,
            target_phase_indices=target_phase_indices,
            compression_only_actor_objective=compression_only_actor_objective,
            c3_action_contract=c3_action_contract,
        )
        if not bool(torch.isfinite(item.total).item()):
            raise FloatingPointError("Order9 topology PPO loss became non-finite")
        if (
            boundary_fine_tune is not None
            and boundary_fine_tune.topology_gradient_surgery_enabled
        ):
            raw_gradients = torch.autograd.grad(
                item.total,
                parameters,
                allow_unused=True,
            )
            topology_gradients.append(
                tuple(
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach()
                    for parameter, gradient in zip(parameters, raw_gradients)
                )
            )
        else:
            (item.total / topology_count).backward()
        detached_new.append(item.new_log_prob.detach())
        detached_old.append(item.old_log_prob.detach())
        detached_phase.append(item.phase_index.detach())
        base_rows.append(_detached_ppo_metrics(item, config=config))
        topology_phase_metrics[key] = _phase_kl_metrics(
            new_log_prob=item.new_log_prob.detach(),
            old_log_prob=item.old_log_prob.detach(),
            phase_index=item.phase_index,
        )
        del item
    if topology_gradients:
        pcgrad_conflict_pair_count = _pcgrad_conflict_pair_count(
            topology_gradients,
            projection_parameter_mask=pcgrad_projection_mask,
        )
        projected = _mean_pcgrad(
            topology_gradients,
            projection_parameter_mask=pcgrad_projection_mask,
        )
        for parameter, gradient in zip(parameters, projected):
            parameter.grad = gradient
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
    if not bool(torch.isfinite(torch.as_tensor(norm)).item()):
        raise FloatingPointError("Order9 topology PPO gradient norm became non-finite")
    metrics = {
        name: sum(row[name] for row in base_rows) / len(base_rows)
        for name in base_rows[0]
    }
    new_log_prob = torch.cat(detached_new)
    old_log_prob = torch.cat(detached_old)
    phase_index = torch.cat(detached_phase)
    phase_kl = _phase_kl_metrics(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        phase_index=phase_index,
    )
    metrics.update(phase_kl)
    topology_phase_maxima: list[float] = []
    for key, local in topology_phase_metrics.items():
        topology_phase_maxima.append(local["maximum_phase_kl"])
        for name, value in local.items():
            metrics[f"{name}_topology_{key}"] = value
    metrics["maximum_topology_phase_kl"] = max(topology_phase_maxima)
    metrics["maximum_target_topology_phase_kl"] = _maximum_topology_phase_kl(
        topology_phase_metrics, target_phase_indices
    )
    metrics["maximum_non_target_topology_phase_kl"] = (
        _maximum_topology_phase_kl(
            topology_phase_metrics,
            tuple(
                phase
                for phase in sorted(int(value) for value in phase_index.unique())
                if phase not in set(int(value) for value in target_phase_indices)
            ),
        )
        if boundary_fine_tune is not None
        else 0.0
    )
    metrics["optimizer_step_applied"] = 1.0
    metrics["optimizer_step_rolled_back"] = 0.0
    metrics["boundary_non_target_kl_violation"] = 0.0
    metrics["boundary_topology_phase_kl_violation"] = 0.0
    metrics["topology_gradient_conflict_pair_count"] = float(
        pcgrad_conflict_pair_count
    )
    if boundary_fine_tune is None:
        optimizer.step()
        return metrics

    snapshot = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    post_metrics = _post_step_topology_kl_metrics(
        policy,
        topology_batches,
        config=config,
        boundary_fine_tune=boundary_fine_tune,
        target_phase_indices=target_phase_indices,
        compression_only_actor_objective=compression_only_actor_objective,
        c3_action_contract=c3_action_contract,
    )
    metrics.update(post_metrics)
    non_target_violation = (
        metrics["maximum_non_target_topology_phase_kl"]
        > boundary_fine_tune.non_target_parent_kl_limit
    )
    topology_violation = (
        metrics["maximum_topology_phase_kl"]
        > boundary_fine_tune.maximum_topology_phase_kl
    )
    if non_target_violation or topology_violation:
        with torch.no_grad():
            for parameter, prior in zip(parameters, snapshot):
                parameter.copy_(prior)
        optimizer.zero_grad(set_to_none=True)
        metrics["optimizer_step_applied"] = 0.0
        metrics["optimizer_step_rolled_back"] = 1.0
        metrics["boundary_non_target_kl_violation"] = float(
            non_target_violation
        )
        metrics["boundary_topology_phase_kl_violation"] = float(
            topology_violation
        )
    return metrics


def _topology_stratified_contact_head_sequence_step(
    policy: Order9PhaseConditionedActorCritic,
    optimizer: torch.optim.Optimizer,
    topology_batches: Sequence[
        tuple[
            str,
            _TensorReplay,
            Sequence[_Sequence],
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    *,
    config: Order9PPOOptimizationConfig,
    boundary_fine_tune: Order9C3BoundaryFineTuneConfig,
    target_phase_indices: Sequence[int],
    c3_action_contract: str,
) -> dict[str, float]:
    """Apply one extra PPO step to contact-specific parameters only."""

    if not topology_batches:
        raise SchemaValidationError(
            "Order9 contact-head topology batches are empty"
        )
    if not boundary_fine_tune.factorized_actor_credit_enabled:
        raise SchemaValidationError(
            "Order9 contact-head extra optimization requires factorized credit"
        )
    named_contact_parameters = _order9_contact_head_named_parameters(policy)
    contact_parameters = [parameter for _, parameter in named_contact_parameters]
    contact_parameter_ids = {id(parameter) for parameter in contact_parameters}
    optimizer.zero_grad(set_to_none=True)
    topology_count = len(topology_batches)
    detached_new: list[torch.Tensor] = []
    detached_old: list[torch.Tensor] = []
    detached_phase: list[torch.Tensor] = []
    contact_losses: list[float] = []
    contact_advantage_abs_means: list[float] = []
    coordinate_loss_rows: dict[str, list[float]] = {
        name: [] for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
    }
    coordinate_advantage_rows: dict[str, list[float]] = {
        name: [] for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
    }
    for _, replay, sequences, advantages, returns in topology_batches:
        item = _sequence_ppo_terms(
            policy,
            replay,
            sequences,
            advantages=advantages,
            returns=returns,
            config=config,
            boundary_fine_tune=boundary_fine_tune,
            target_phase_indices=target_phase_indices,
            compression_only_actor_objective=False,
            c3_action_contract=c3_action_contract,
        )
        if boundary_fine_tune.contact_coordinate_credit_enabled:
            if (
                item.contact_coordinate_losses is None
                or item.contact_coordinate_advantages is None
                or set(item.contact_coordinate_losses)
                != set(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
                or set(item.contact_coordinate_advantages)
                != set(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
            ):
                raise SchemaValidationError(
                    "Order9 contact-head extra optimization lacks coordinate "
                    "advantages"
                )
            contact_loss = _weighted_contact_coordinate_actor_loss(
                item.contact_coordinate_losses,
                normal_translation_weight=(
                    boundary_fine_tune.
                    contact_normal_translation_actor_loss_weight
                ),
            )
            for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS:
                coordinate_loss_rows[name].append(
                    float(
                        item.contact_coordinate_losses[name]
                        .detach()
                        .cpu()
                        .item()
                    )
                )
                coordinate_advantage_rows[name].append(
                    float(
                        item.contact_coordinate_advantages[name][
                            item.target_phase_mask
                        ]
                        .abs()
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )
                )
        else:
            if (
                item.actor_component_losses is None
                or item.actor_component_advantages is None
                or "contact" not in item.actor_component_losses
                or "contact" not in item.actor_component_advantages
            ):
                raise SchemaValidationError(
                    "Order9 contact-head extra optimization lacks contact advantage"
                )
            contact_loss = item.actor_component_losses["contact"]
            contact_advantage_abs_means.append(
                float(
                    item.actor_component_advantages["contact"][
                        item.target_phase_mask
                    ]
                    .abs()
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
            )
        if not bool(torch.isfinite(contact_loss).item()):
            raise FloatingPointError(
                "Order9 contact-head PPO loss became non-finite"
            )
        (contact_loss / topology_count).backward()
        contact_losses.append(float(contact_loss.detach().cpu().item()))
        detached_new.append(item.new_log_prob.detach())
        detached_old.append(item.old_log_prob.detach())
        detached_phase.append(item.phase_index.detach())
        del item

    # Backpropagation traverses the shared trunk to reach the contact head.
    # Remove those incidental gradients before Adam so only the explicitly
    # owned contact parameters can move during this pass.
    for parameter in policy.parameters():
        if id(parameter) not in contact_parameter_ids:
            parameter.grad = None
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        contact_parameters, config.max_grad_norm
    )
    if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).item()):
        raise FloatingPointError(
            "Order9 contact-head PPO gradient norm became non-finite"
        )
    if not any(parameter.grad is not None for parameter in contact_parameters):
        raise SchemaValidationError(
            "Order9 contact-head extra optimization produced no gradient"
        )
    snapshot = [parameter.detach().clone() for parameter in contact_parameters]
    optimizer.step()
    post_metrics = _post_step_topology_kl_metrics(
        policy,
        topology_batches,
        config=config,
        boundary_fine_tune=boundary_fine_tune,
        target_phase_indices=target_phase_indices,
        compression_only_actor_objective=False,
        c3_action_contract=c3_action_contract,
    )
    non_target_violation = (
        post_metrics["maximum_non_target_topology_phase_kl"]
        > boundary_fine_tune.non_target_parent_kl_limit
    )
    topology_violation = (
        post_metrics["maximum_topology_phase_kl"]
        > boundary_fine_tune.maximum_topology_phase_kl
    )
    rolled_back = non_target_violation or topology_violation
    if rolled_back:
        with torch.no_grad():
            for parameter, prior in zip(contact_parameters, snapshot):
                parameter.copy_(prior)
        optimizer.zero_grad(set_to_none=True)
    actor_loss = sum(contact_losses) / len(contact_losses)
    new_log_prob = torch.cat(detached_new)
    old_log_prob = torch.cat(detached_old)
    phase_index = torch.cat(detached_phase)
    pre_metrics = _phase_kl_metrics(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        phase_index=phase_index,
    )
    return {
        "actor_loss": actor_loss,
        "actor_loss_contact_extra": actor_loss,
        "actor_advantage_abs_mean_contact_extra": (
            sum(contact_advantage_abs_means) / len(contact_advantage_abs_means)
            if contact_advantage_abs_means
            else (
                sum(
                    value
                    for rows in coordinate_advantage_rows.values()
                    for value in rows
                )
                / sum(len(rows) for rows in coordinate_advantage_rows.values())
            )
        ),
        **{
            f"actor_loss_contact_extra_{name}": (
                sum(rows) / len(rows) if rows else 0.0
            )
            for name, rows in coordinate_loss_rows.items()
        },
        **{
            f"actor_advantage_abs_mean_contact_extra_{name}": (
                sum(rows) / len(rows) if rows else 0.0
            )
            for name, rows in coordinate_advantage_rows.items()
        },
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": actor_loss,
        "contact_head_extra_gradient_norm": float(
            torch.as_tensor(gradient_norm).detach().cpu().item()
        ),
        "contact_head_extra_optimizer_step": 1.0,
        "optimizer_step_applied": float(not rolled_back),
        "optimizer_step_rolled_back": float(rolled_back),
        "boundary_non_target_kl_violation": float(non_target_violation),
        "boundary_topology_phase_kl_violation": float(topology_violation),
        "topology_gradient_conflict_pair_count": 0.0,
        **{
            f"pre_step_{name}": value
            for name, value in pre_metrics.items()
        },
        **post_metrics,
    }


def _post_step_topology_kl_metrics(
    policy: Order9PhaseConditionedActorCritic,
    topology_batches: Sequence[
        tuple[
            str,
            _TensorReplay,
            Sequence[_Sequence],
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    *,
    config: Order9PPOOptimizationConfig,
    boundary_fine_tune: Order9C3BoundaryFineTuneConfig,
    target_phase_indices: Sequence[int],
    compression_only_actor_objective: bool = False,
    c3_action_contract: str | None = None,
) -> dict[str, float]:
    detached_new: list[torch.Tensor] = []
    detached_old: list[torch.Tensor] = []
    detached_phase: list[torch.Tensor] = []
    topology_phase_metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for key, replay, sequences, advantages, returns in topology_batches:
            item = _sequence_ppo_terms(
                policy,
                replay,
                sequences,
                advantages=advantages,
                returns=returns,
                config=config,
                boundary_fine_tune=boundary_fine_tune,
                target_phase_indices=target_phase_indices,
                compression_only_actor_objective=(
                    compression_only_actor_objective
                ),
                c3_action_contract=c3_action_contract,
            )
            detached_new.append(item.new_log_prob)
            detached_old.append(item.old_log_prob)
            detached_phase.append(item.phase_index)
            topology_phase_metrics[key] = _phase_kl_metrics(
                new_log_prob=item.new_log_prob,
                old_log_prob=item.old_log_prob,
                phase_index=item.phase_index,
            )
    new_log_prob = torch.cat(detached_new)
    old_log_prob = torch.cat(detached_old)
    phase_index = torch.cat(detached_phase)
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    metrics = _phase_kl_metrics(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        phase_index=phase_index,
    )
    metrics["approximate_kl"] = max(
        0.0, float(((ratio - 1.0) - log_ratio).mean().cpu().item())
    )
    for key, local in topology_phase_metrics.items():
        for name, value in local.items():
            metrics[f"{name}_topology_{key}"] = value
    observed = sorted(int(value) for value in phase_index.unique())
    non_target = tuple(
        phase
        for phase in observed
        if phase not in set(int(value) for value in target_phase_indices)
    )
    metrics["maximum_topology_phase_kl"] = _maximum_topology_phase_kl(
        topology_phase_metrics, observed
    )
    metrics["maximum_target_topology_phase_kl"] = _maximum_topology_phase_kl(
        topology_phase_metrics, target_phase_indices
    )
    metrics["maximum_non_target_topology_phase_kl"] = (
        _maximum_topology_phase_kl(topology_phase_metrics, non_target)
    )
    return metrics


def _maximum_topology_phase_kl(
    topology_phase_metrics: Mapping[str, Mapping[str, float]],
    phase_indices: Sequence[int],
) -> float:
    names = tuple(f"approximate_kl_phase_{int(phase)}" for phase in phase_indices)
    values = [
        float(row[name])
        for row in topology_phase_metrics.values()
        for name in names
        if name in row
    ]
    return max(values, default=0.0)


def _detached_ppo_metrics(
    item: _SequencePPOTerms, *, config: Order9PPOOptimizationConfig
) -> dict[str, float]:
    actor = (
        item.actor_loss
        if item.actor_loss is not None
        else clipped_ppo_surrogate_loss(
            new_log_prob=item.new_log_prob[item.target_phase_mask],
            old_log_prob=item.old_log_prob[item.target_phase_mask],
            advantages=item.advantages[item.target_phase_mask],
            clip_ratio=config.clip_ratio,
        )
    )
    value = torch.nn.functional.mse_loss(item.new_values, item.returns)
    with torch.no_grad():
        log_ratio = item.new_log_prob - item.old_log_prob
        ratio = torch.exp(log_ratio)
        approximate_kl = float(((ratio - 1.0) - log_ratio).mean().cpu().item())
        clipped_fraction = float(
            ((ratio - 1.0).abs() > config.clip_ratio).float().mean().cpu().item()
        )
    metrics = {
        "actor_loss": float(actor.detach().cpu().item()),
        "value_loss": float(value.detach().cpu().item()),
        "entropy": float(
            item.entropy[item.target_phase_mask].mean().detach().cpu().item()
        ),
        "total_loss": float(item.total.detach().cpu().item()),
        "approximate_kl": max(0.0, approximate_kl),
        "clipped_fraction": clipped_fraction,
        "non_target_parent_kl": float(
            item.non_target_parent_kl.detach().cpu().item()
        ),
        "privileged_compression_teacher_loss": float(
            item.privileged_compression_teacher_loss.detach().cpu().item()
        ),
        "privileged_compression_teacher_sample_count": float(
            item.privileged_compression_teacher_sample_count
        ),
        "privileged_wrench_satisfied_parent_kl": float(
            item.privileged_wrench_satisfied_parent_kl.detach().cpu().item()
        ),
        "privileged_wrench_satisfied_sample_count": float(
            item.privileged_wrench_satisfied_sample_count
        ),
        "contact_normal_preload_deficit_teacher_loss": float(
            item.contact_normal_preload_deficit_teacher_loss.detach().cpu().item()
        ),
        "contact_normal_preload_deficit_teacher_sample_count": float(
            item.contact_normal_preload_deficit_teacher_sample_count
        ),
    }
    if item.actor_component_losses is not None:
        assert item.actor_component_advantages is not None
        assert item.actor_component_new_log_probs is not None
        assert item.actor_component_old_log_probs is not None
        for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS:
            metrics[f"actor_loss_{name}"] = float(
                item.actor_component_losses[name].detach().cpu().item()
            )
            component_advantage = item.actor_component_advantages[name][
                item.target_phase_mask
            ]
            metrics[f"actor_advantage_abs_mean_{name}"] = float(
                component_advantage.abs().mean().detach().cpu().item()
            )
            component_log_ratio = (
                item.actor_component_new_log_probs[name]
                - item.actor_component_old_log_probs[name]
            )
            component_ratio = torch.exp(component_log_ratio)
            metrics[f"approximate_kl_{name}"] = max(
                0.0,
                float(
                    (
                        (component_ratio - 1.0) - component_log_ratio
                    ).mean().detach().cpu().item()
                ),
            )
    return metrics


def _topology_key(
    artifact: Order9TensorRolloutArtifact, topology_index: int
) -> str:
    morphology = artifact.metadata.get("morphology_graph")
    if not isinstance(morphology, dict):
        raise SchemaValidationError("Order9 tensor PPO morphology graph is missing")
    module_count = len(morphology.get("modules", []))
    if module_count < 1:
        raise SchemaValidationError("Order9 tensor PPO morphology is empty")
    return f"m{module_count:02d}_{topology_index:02d}_{stable_hash(morphology)[:8]}"


def _aggregate_exact_replay(
    rows: Mapping[str, Mapping[str, float | int | bool]],
) -> dict[str, object]:
    if not rows:
        raise SchemaValidationError("Order9 exact replay topology rows are empty")
    maximum_names = (
        "maximum_log_prob_replay_error",
        "maximum_component_log_prob_sum_replay_error",
        "maximum_contact_coordinate_log_prob_sum_replay_error",
        "maximum_value_replay_error",
        "maximum_recurrent_replay_error",
        "maximum_stored_recurrent_continuity_error",
        "maximum_stored_previous_action_continuity_error",
    )
    result: dict[str, object] = {
        "exact_replay_record_count": sum(
            int(row["exact_replay_record_count"]) for row in rows.values()
        ),
        "exact_replay_topology_count": len(rows),
        "exact_replay_by_topology": {key: dict(value) for key, value in rows.items()},
    }
    for name in maximum_names:
        result[name] = max(float(row[name]) for row in rows.values())
    for name in (
        "exact_replay_absolute_tolerance",
        "exact_replay_log_prob_tolerance",
        "exact_replay_value_tolerance",
        "exact_replay_recurrent_tolerance",
        "exact_replay_continuity_tolerance",
    ):
        values = {float(row[name]) for row in rows.values()}
        if len(values) != 1:
            raise SchemaValidationError(
                f"Order9 exact replay topology tolerance differs at {name}"
            )
        result[name] = values.pop()
    for name in (
        "exact_replay_collection_shaped_time_slices",
        "exact_replay_timestep_batched_active_sequences",
        "exact_replay_record_invariant_cache",
    ):
        values = {bool(row[name]) for row in rows.values()}
        if len(values) != 1:
            raise SchemaValidationError(
                f"Order9 exact replay topology contract differs at {name}"
            )
        result[name] = values.pop()
    return result


def _aggregate_advantage_metadata(
    rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise SchemaValidationError("Order9 advantage topology rows are empty")
    counts: dict[str, int] = {}
    positive_weighted: dict[str, float] = {}
    for row in rows.values():
        row_counts = row["advantage_phase_sample_count"]
        row_positive = row["advantage_phase_positive_fraction_before_normalization"]
        assert isinstance(row_counts, dict) and isinstance(row_positive, dict)
        for phase, raw_count in row_counts.items():
            count = int(raw_count)
            counts[phase] = counts.get(phase, 0) + count
            positive_weighted[phase] = positive_weighted.get(phase, 0.0) + (
                count * float(row_positive[phase])
            )
    return {
        "advantage_normalization_scope": (
            "topology_actor_phase" if len(rows) > 1 else "actor_phase"
        ),
        "advantage_phase_sample_count": counts,
        "advantage_phase_positive_fraction_before_normalization": {
            phase: positive_weighted[phase] / counts[phase]
            for phase in counts
        },
        "advantage_metadata_by_topology": {
            key: dict(value) for key, value in rows.items()
        },
    }


def _validate_exact_replay(
    policy: Order9PhaseConditionedActorCritic,
    replay: _TensorReplay,
    sequences: Sequence[_Sequence],
    *,
    sequence_batch_size: int,
    log_prob_tolerance: float = 2.5e-3,
    value_tolerance: float = 5.0e-5,
    recurrent_tolerance: float = 5.0e-4,
    continuity_tolerance: float = 2.0e-5,
) -> dict[str, float | int | bool]:
    # Re-evaluate the exact collection-shaped [environment] slices.  Long
    # articulated rollouts amplify a few-float reconstruction difference in
    # centroidal features; field-specific tolerances keep value and temporal
    # continuity at the original strict threshold without pretending the
    # action log-probability is bitwise invariant to that reconstruction.
    del sequence_batch_size
    maxima = {"log_prob": 0.0, "value": 0.0, "recurrent": 0.0}
    component_log_probs = {
        name: torch.full_like(
            replay.tensors["old_log_prob"],
            float("nan"),
            device="cpu",
        )
        for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    }
    contact_coordinate_log_probs = (
        {
            name: torch.full_like(
                replay.tensors["old_log_prob"],
                float("nan"),
                device="cpu",
            )
            for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
        }
        if isinstance(policy, Order9ContactSpacePhaseConditionedActorCritic)
        else None
    )
    maximum_component_sum_error = 0.0
    maximum_contact_coordinate_sum_error = 0.0
    validated = 0
    with torch.no_grad():
        for time_index in range(replay.artifact.step_count):
            environments = torch.nonzero(
                replay.tensors["valid"][time_index], as_tuple=False
            ).flatten().tolist()
            if not environments:
                continue
            batch = replay.batch(
                [time_index] * len(environments), environments
            )
            step = policy.step(
                batch.graph,
                None,
                batch.actor_features,
                batch.previous_global_action,
                batch.recurrent_state_in,
                phase_features=batch.phase_features,
                privileged_disturbance_body=batch.privileged,
                action=batch.global_action,
                joint_action=batch.joint_action,
                **(
                    {
                        "contact_compression_residual_action": (
                            batch.contact_compression_residual_action
                        )
                    }
                    if isinstance(
                        policy, Order9MorphologyInvariantCompressionActorCritic
                    )
                    else {}
                ),
                **(
                    {
                        "contact_space_residual_action": (
                            batch.contact_space_residual_action
                        ),
                        "contact_slot_features": batch.contact_slot_features,
                        "contact_slot_owner_module_indices": (
                            batch.contact_slot_owner_module_indices
                        ),
                        "contact_slot_mask": batch.contact_slot_mask,
                    }
                    if isinstance(
                        policy, Order9ContactSpacePhaseConditionedActorCritic
                    )
                    else {}
                ),
                active_knot_features=batch.active_knot_features,
                active_assignment_features=batch.active_assignment_features,
            )
            maxima["log_prob"] = max(
                maxima["log_prob"],
                _require_close(
                    step.log_prob,
                    batch.old_log_prob,
                    tolerance=log_prob_tolerance,
                    label="old_log_prob",
                ),
            )
            maxima["value"] = max(
                maxima["value"],
                _require_close(
                    step.value,
                    batch.old_value,
                    tolerance=value_tolerance,
                    label="old_value",
                ),
            )
            maxima["recurrent"] = max(
                maxima["recurrent"],
                _require_close(
                    step.recurrent_state,
                    batch.recurrent_state_out,
                    tolerance=recurrent_tolerance,
                    label="recurrent_state_out",
                ),
            )
            component_rows = {
                "contact": step.contact_log_prob,
                "centroidal": step.centroidal_log_prob,
                "posture": step.posture_log_prob,
            }
            component_sum = torch.stack(
                [component_rows[name] for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS],
                dim=0,
            ).sum(dim=0)
            maximum_component_sum_error = max(
                maximum_component_sum_error,
                _require_close(
                    component_sum,
                    batch.old_log_prob,
                    tolerance=log_prob_tolerance,
                    label="factorized_old_log_prob",
                ),
            )
            for name, value in component_rows.items():
                component_log_probs[name][time_index, environments] = (
                    value.detach().cpu()
                )
            if contact_coordinate_log_probs is not None:
                coordinate_rows = _contact_coordinate_group_log_probs(
                    bounded_action=batch.contact_space_residual_action,
                    bounded_mean=step.contact_space_residual_action_mean,
                    log_std=policy.contact_space_actor_log_std,
                    slot_mask=batch.contact_slot_mask,
                    categorical_contact_log_prob=(
                        step.contact_log_prob
                        if isinstance(
                            policy,
                            Order9CategoricalContactNormalPhaseConditionedActorCritic,
                        )
                        else None
                    ),
                )
                coordinate_sum = torch.stack(
                    [
                        coordinate_rows[name]
                        for name in ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
                    ],
                    dim=0,
                ).sum(dim=0)
                maximum_contact_coordinate_sum_error = max(
                    maximum_contact_coordinate_sum_error,
                    _require_close(
                        coordinate_sum,
                        step.contact_log_prob,
                        tolerance=log_prob_tolerance,
                        label="contact_coordinate_old_log_prob",
                    ),
                )
                for name, value in coordinate_rows.items():
                    contact_coordinate_log_probs[name][
                        time_index, environments
                    ] = value.detach().cpu()
            validated += len(environments)
    expected_count = sum(item.length for item in sequences)
    if validated != expected_count:
        raise SchemaValidationError("Order9 tensor exact replay count differs")
    valid = replay.tensors["valid"]
    if any(
        not bool(torch.isfinite(value[valid]).all())
        for value in component_log_probs.values()
    ):
        raise SchemaValidationError(
            "Order9 factorized behavior log probability cache is incomplete"
        )
    replay.behavior_component_log_probs = component_log_probs
    if contact_coordinate_log_probs is not None:
        if any(
            not bool(torch.isfinite(value[valid]).all())
            for value in contact_coordinate_log_probs.values()
        ):
            raise SchemaValidationError(
                "Order9 contact-coordinate behavior log probability cache is "
                "incomplete"
            )
        replay.behavior_contact_coordinate_log_probs = (
            contact_coordinate_log_probs
        )
    continuity = _validate_stored_recurrence_continuity(
        replay.artifact, tolerance=continuity_tolerance
    )
    maximum_tolerance = max(
        log_prob_tolerance, value_tolerance, recurrent_tolerance
    )
    return {
        "exact_replay_record_count": validated,
        "maximum_log_prob_replay_error": maxima["log_prob"],
        "maximum_component_log_prob_sum_replay_error": (
            maximum_component_sum_error
        ),
        "maximum_contact_coordinate_log_prob_sum_replay_error": (
            maximum_contact_coordinate_sum_error
        ),
        "maximum_value_replay_error": maxima["value"],
        "maximum_recurrent_replay_error": maxima["recurrent"],
        "maximum_stored_recurrent_continuity_error": continuity["recurrent"],
        "maximum_stored_previous_action_continuity_error": continuity[
            "previous_action"
        ],
        "exact_replay_absolute_tolerance": maximum_tolerance,
        "exact_replay_log_prob_tolerance": log_prob_tolerance,
        "exact_replay_value_tolerance": value_tolerance,
        "exact_replay_recurrent_tolerance": recurrent_tolerance,
        "exact_replay_continuity_tolerance": continuity_tolerance,
        "exact_replay_collection_shaped_time_slices": True,
        "exact_replay_timestep_batched_active_sequences": False,
        "exact_replay_record_invariant_cache": True,
    }


def _validate_stored_recurrence_continuity(
    artifact: Order9TensorRolloutArtifact, *, tolerance: float
) -> dict[str, float]:
    tensors = artifact.tensors
    valid = tensors["valid"]
    maximum_recurrent = 0.0
    maximum_previous = 0.0
    c3_action_contract = artifact.metadata.get("c3_action_contract")
    global_dimension = (
        None
        if c3_action_contract is None
        else order9_c3_action_contract_global_dimension(
            str(c3_action_contract)
        )
    )
    masked_phase_labels = set(
        str(value)
        for value in artifact.metadata.get(
            "c3_nominal_only_action_mask_phase_labels", []
        )
    )
    actor_phase_labels = tuple(
        str(value) for value in artifact.metadata.get("actor_phase_labels", [])
    )
    for time_index in range(artifact.step_count - 1):
        same_episode = (
            valid[time_index]
            & valid[time_index + 1]
            & (tensors["episode_serial"][time_index]
               == tensors["episode_serial"][time_index + 1])
        )
        if bool(same_episode.any()):
            maximum_recurrent = max(
                maximum_recurrent,
                float(
                    (
                        tensors["recurrent_state_out"][time_index, same_episode]
                        - tensors["recurrent_state_in"][time_index + 1, same_episode]
                    )
                    .abs()
                    .amax()
                    .item()
                ),
            )
            expected_previous = tensors[
                "applied_global_action"
                if "applied_global_action" in tensors
                else "global_action"
            ][time_index, same_episode].clone()
            if global_dimension is not None and "applied_global_action" not in tensors:
                expected_previous[:, global_dimension:] = 0.0
                phase_rows = tensors["phase_index"][time_index, same_episode]
                if actor_phase_labels and masked_phase_labels:
                    nominal_only = torch.tensor(
                        [
                            actor_phase_labels[int(value)]
                            in masked_phase_labels
                            for value in phase_rows.tolist()
                        ],
                        device=expected_previous.device,
                        dtype=torch.bool,
                    )
                    expected_previous[nominal_only] = 0.0
            maximum_previous = max(
                maximum_previous,
                float(
                    (
                        expected_previous
                        - tensors["previous_global_action"][
                            time_index + 1, same_episode
                        ]
                    )
                    .abs()
                    .amax()
                    .item()
                ),
            )
    if maximum_recurrent > tolerance or maximum_previous > tolerance:
        raise SchemaValidationError(
            "Order9 stored recurrent/action continuity mismatch "
            f"(recurrent={maximum_recurrent:.9g}, "
            f"previous_action={maximum_previous:.9g}, tolerance={tolerance:.9g})"
        )
    return {"recurrent": maximum_recurrent, "previous_action": maximum_previous}


def _require_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float,
    label: str,
) -> float:
    if actual.shape != expected.shape:
        raise SchemaValidationError(
            f"Order9 tensor exact replay {label} shape differs"
        )
    maximum = float((actual - expected).abs().amax().detach().cpu().item())
    if not math.isfinite(maximum) or maximum > tolerance:
        raise SchemaValidationError(
            f"Order9 tensor exact replay {label} mismatch "
            f"(max_abs_error={maximum:.9g}, tolerance={tolerance:.9g})"
        )
    return maximum


def _sequences(
    artifact: Order9TensorRolloutArtifact, *, sequence_length: int
) -> list[_Sequence]:
    tensors = artifact.tensors
    result: list[_Sequence] = []
    for environment in range(artifact.environment_count):
        valid_times = torch.nonzero(
            tensors["valid"][:, environment], as_tuple=False
        ).flatten().tolist()
        cursor = 0
        while cursor < len(valid_times):
            start_offset = cursor
            serial = int(
                tensors["episode_serial"][valid_times[cursor], environment]
            )
            expected_step = 0
            while cursor < len(valid_times):
                time_index = valid_times[cursor]
                if int(tensors["episode_serial"][time_index, environment]) != serial:
                    break
                if int(tensors["step_index"][time_index, environment]) != expected_step:
                    raise SchemaValidationError(
                        "Order9 tensor episode steps are not contiguous"
                    )
                expected_step += 1
                cursor += 1
            episode_times = valid_times[start_offset:cursor]
            phase_start = 0
            while phase_start < len(episode_times):
                phase = int(
                    tensors["phase_index"][
                        episode_times[phase_start], environment
                    ]
                )
                phase_end = phase_start + 1
                while (
                    phase_end < len(episode_times)
                    and int(
                        tensors["phase_index"][
                            episode_times[phase_end], environment
                        ]
                    )
                    == phase
                ):
                    phase_end += 1
                phase_times = episode_times[phase_start:phase_end]
                for offset in range(0, len(phase_times), sequence_length):
                    chunk = phase_times[offset : offset + sequence_length]
                    if chunk != list(range(chunk[0], chunk[0] + len(chunk))):
                        raise SchemaValidationError(
                            "Order9 tensor PPO requires contiguous rollout time indices"
                        )
                    result.append(
                        _Sequence(
                            environment=environment,
                            start=chunk[0],
                            length=len(chunk),
                            phase_index=phase,
                        )
                    )
                phase_start = phase_end
    if not result:
        raise SchemaValidationError("Order9 tensor PPO has no sequences")
    return result


def _normalize_advantages(
    advantages: torch.Tensor,
    *,
    valid: torch.Tensor,
    phase_index: torch.Tensor,
    phase_local: bool,
) -> tuple[torch.Tensor, dict[str, object]]:
    normalized = torch.zeros_like(advantages)
    observed = sorted(
        int(value) for value in phase_index[valid].unique().tolist()
    )
    if not observed:
        raise SchemaValidationError("Order9 tensor PPO has no valid phase")
    groups = observed if phase_local else [-1]
    metadata: dict[str, object] = {
        "advantage_normalization_scope": (
            "actor_phase" if phase_local else "global"
        ),
        "advantage_phase_sample_count": {},
        "advantage_phase_positive_fraction_before_normalization": {},
        "advantage_singleton_phases_zero_centered": [],
    }
    counts = metadata["advantage_phase_sample_count"]
    positives = metadata[
        "advantage_phase_positive_fraction_before_normalization"
    ]
    assert isinstance(counts, dict) and isinstance(positives, dict)
    for phase in observed:
        mask = valid & (phase_index == phase)
        values = advantages[mask]
        counts[str(phase)] = int(values.numel())
        positives[str(phase)] = float((values > 0.0).float().mean().item())
    for phase in groups:
        mask = valid if phase < 0 else valid & (phase_index == phase)
        values = advantages[mask]
        if values.numel() < 2:
            if phase >= 0 and values.numel() == 1:
                # A continuous-state-inheritance shard may end exactly one
                # transition after crossing a phase boundary.  Phase-local
                # centering of that singleton is mathematically zero; retain
                # it for critic/recurrent replay without inventing an actor
                # gradient or borrowing statistics from another topology.
                normalized[mask] = 0.0
                singleton_phases = metadata[
                    "advantage_singleton_phases_zero_centered"
                ]
                assert isinstance(singleton_phases, list)
                singleton_phases.append(phase)
                continue
            raise SchemaValidationError(
                f"Order9 tensor PPO phase {phase} lacks advantage support"
            )
        mean = values.mean()
        scale = torch.sqrt(values.var(unbiased=False) + 1.0e-8)
        normalized[mask] = (values - mean) / scale
    return normalized, metadata


def _phase_balanced_sequences(
    sequences: Sequence[_Sequence], *, seed: int
) -> list[_Sequence]:
    by_phase: dict[int, list[_Sequence]] = {}
    for sequence in sequences:
        by_phase.setdefault(sequence.phase_index, []).append(sequence)
    if len(by_phase) < 2 or any(not values for values in by_phase.values()):
        raise SchemaValidationError(
            "Order9 phase-balanced PPO requires multiple populated phases"
        )
    generator = random.Random(seed)
    for values in by_phase.values():
        generator.shuffle(values)
    phase_order = sorted(by_phase)
    target_per_phase = math.ceil(len(sequences) / len(phase_order))
    balanced: list[_Sequence] = []
    for offset in range(target_per_phase):
        cycle = list(phase_order)
        generator.shuffle(cycle)
        for phase in cycle:
            values = by_phase[phase]
            balanced.append(values[offset % len(values)])
    return balanced


def _transition_budget_sequence_batches(
    sequences: Sequence[_Sequence], *, transition_budget: int
) -> list[list[_Sequence]]:
    """Pack complete recurrent sequences by transition rather than list count.

    Phase and terminal boundaries create short sequences.  A fixed number of
    sequences therefore makes a highly fragmented shard dictate excessive
    optimizer steps while contributing fewer transitions per loss.  Packing
    whole sequences up to a per-topology transition budget keeps recurrent
    boundaries intact and makes topology-stratified minibatches comparable in
    actual sample count.
    """

    if transition_budget < 1:
        raise ValueError("Order9 topology transition budget must be positive")
    if not sequences:
        raise SchemaValidationError(
            "Order9 topology transition packing requires sequences"
        )
    result: list[list[_Sequence]] = []
    current: list[_Sequence] = []
    current_transition_count = 0
    for sequence in sequences:
        if sequence.length < 1:
            raise SchemaValidationError(
                "Order9 topology transition packing found an empty sequence"
            )
        if current and current_transition_count + sequence.length > transition_budget:
            result.append(current)
            current = []
            current_transition_count = 0
        current.append(sequence)
        current_transition_count += sequence.length
        if current_transition_count >= transition_budget:
            result.append(current)
            current = []
            current_transition_count = 0
    if current:
        result.append(current)
    return result


def _merge_boundary_incomplete_sequence_batches(
    batches: Sequence[Sequence[_Sequence]],
    *,
    target_phase_indices: Sequence[int],
    allow_target_only: bool,
) -> list[list[_Sequence]]:
    """Merge packing spill so every boundary batch has usable actor support.

    Transition-budget packing can leave a short final batch containing only
    one phase even when the phase-balanced source covers the complete phase
    set.  Such a spill is not an independent PPO minibatch: a boundary actor
    objective needs at least one target phase and phase-reset shards also need
    a non-target parent-policy anchor.  Merge incomplete spills with an
    adjacent batch without dropping or duplicating any recurrent sequence.

    Continuous state-inheritance shards may legitimately contain only target
    phases because their parent anchor is supplied by the phase-reset shards
    in the same topology-stratified optimizer step.
    """

    if not batches:
        raise SchemaValidationError(
            "Order9 boundary batch repair requires packed sequences"
        )
    targets = {int(value) for value in target_phase_indices}
    if not targets:
        raise SchemaValidationError(
            "Order9 boundary batch repair requires target phases"
        )

    def complete(batch: Sequence[_Sequence]) -> bool:
        phases = {int(sequence.phase_index) for sequence in batch}
        has_target = bool(phases & targets)
        has_anchor = bool(phases - targets)
        return has_target and (has_anchor or allow_target_only)

    repaired: list[list[_Sequence]] = []
    pending: list[_Sequence] = []
    for source in batches:
        current = [*pending, *source]
        if complete(current):
            repaired.append(current)
            pending = []
        else:
            pending = current
    if pending:
        if not repaired:
            # Preserve the original data for the existing fail-closed mask
            # validation, which provides the more specific contract error.
            return [pending]
        repaired[-1].extend(pending)
    return repaired


def _phase_kl_metrics(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    phase_index: torch.Tensor,
) -> dict[str, float]:
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    pointwise = (ratio - 1.0) - log_ratio
    result: dict[str, float] = {}
    values = []
    for phase in sorted(int(value) for value in phase_index.unique().tolist()):
        mask = phase_index == phase
        value = max(0.0, float(pointwise[mask].mean().cpu().item()))
        result[f"approximate_kl_phase_{phase}"] = value
        values.append(value)
    result["maximum_phase_kl"] = max(values, default=0.0)
    return result


def _resolve_boundary_target_phase_indices(
    artifacts: Sequence[Order9TensorRolloutArtifact],
    fine_tune: Order9C3BoundaryFineTuneConfig | None,
) -> tuple[int, ...]:
    if fine_tune is None:
        return ()
    actor_labels: tuple[str, ...] | None = None
    observed: set[int] = set()
    for artifact in artifacts:
        labels = tuple(str(value) for value in artifact.metadata["actor_phase_labels"])
        if actor_labels is None:
            actor_labels = labels
        elif labels != actor_labels:
            raise SchemaValidationError(
                "Order9 C3 boundary actor phase labels differ by topology"
            )
        observed.update(
            int(value)
            for value in artifact.tensors["phase_index"][
                artifact.tensors["valid"]
            ].unique()
        )
    assert actor_labels is not None
    missing = [
        label
        for label in fine_tune.target_actor_phase_labels
        if label not in actor_labels
    ]
    if missing:
        raise SchemaValidationError(
            "Order9 C3 boundary target actor phases are missing: "
            + ",".join(missing)
        )
    result = tuple(
        actor_labels.index(label)
        for label in fine_tune.target_actor_phase_labels
    )
    if any(phase not in observed for phase in result):
        raise SchemaValidationError(
            "Order9 C3 boundary target phases lack on-policy support"
        )
    if not (observed - set(result)):
        raise SchemaValidationError(
            "Order9 C3 boundary fine-tune lacks non-target anchor phases"
        )
    return result


def _boundary_fine_tune_metadata(
    *,
    fine_tune: Order9C3BoundaryFineTuneConfig | None,
    target_phase_indices: Sequence[int],
    artifacts: Sequence[Order9TensorRolloutArtifact],
    rows: Sequence[Mapping[str, float]],
    behavior_checkpoint_sha256: str,
) -> dict[str, object]:
    if fine_tune is None:
        return {}
    actor_labels = tuple(
        str(value) for value in artifacts[0].metadata["actor_phase_labels"]
    )
    observed = sorted(
        {
            int(value)
            for artifact in artifacts
            for value in artifact.tensors["phase_index"][
                artifact.tensors["valid"]
            ].unique()
        }
    )
    target_set = set(int(value) for value in target_phase_indices)
    non_target = [phase for phase in observed if phase not in target_set]
    applied = [row for row in rows if row["optimizer_step_applied"] > 0.5]
    return {
        "c3_boundary_fine_tune_contract": (
            ORDER9_C3_BOUNDARY_FINE_TUNE_CONTRACT
        ),
        "c3_boundary_fine_tune_parent_checkpoint_sha256": (
            behavior_checkpoint_sha256
        ),
        "c3_boundary_target_actor_phase_labels": list(
            fine_tune.target_actor_phase_labels
        ),
        "c3_boundary_target_actor_phase_indices": list(target_phase_indices),
        "c3_boundary_non_target_actor_phase_labels": [
            actor_labels[phase] for phase in non_target
        ],
        "c3_boundary_non_target_actor_phase_indices": non_target,
        "c3_boundary_target_only_actor_objective": True,
        "c3_boundary_parent_anchor_source": (
            "sha_bound_exact_behavior_log_probability"
        ),
        "c3_boundary_non_target_parent_kl_limit": (
            fine_tune.non_target_parent_kl_limit
        ),
        "c3_boundary_learning_rate_scale": fine_tune.learning_rate_scale,
        "c3_boundary_non_target_parent_kl_weight": (
            fine_tune.non_target_parent_kl_weight
        ),
        "c3_privileged_compression_teacher_weight": (
            fine_tune.privileged_compression_teacher_weight
        ),
        "c3_privileged_compression_teacher_action_step": (
            fine_tune.privileged_compression_teacher_action_step
        ),
        "c3_privileged_compression_teacher_underforce_only": (
            fine_tune.privileged_compression_teacher_underforce_only
        ),
        "c3_privileged_wrench_satisfied_parent_kl_weight": (
            fine_tune.privileged_wrench_satisfied_parent_kl_weight
        ),
        "c3_topology_gradient_surgery_enabled": (
            fine_tune.topology_gradient_surgery_enabled
        ),
        "c3_contact_coordinate_credit_enabled": (
            fine_tune.contact_coordinate_credit_enabled
        ),
        "c3_contact_normal_translation_actor_loss_weight": (
            fine_tune.contact_normal_translation_actor_loss_weight
        ),
        "c3_contact_normal_preload_deficit_teacher_weight": (
            fine_tune.contact_normal_preload_deficit_teacher_weight
        ),
        "c3_contact_normal_preload_deficit_teacher_version": (
            ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION
            if fine_tune.contact_normal_preload_deficit_teacher_weight > 0.0
            else None
        ),
        "c3_contact_normal_preload_deficit_teacher_source": (
            "authored_training_nominal_preload_deficit"
            if fine_tune.contact_normal_preload_deficit_teacher_weight > 0.0
            else None
        ),
        "c3_contact_normal_preload_deficit_teacher_raw_actor_input": False,
        "c3_contact_normal_preload_deficit_teacher_training_only": True,
        "c3_privileged_compression_teacher_raw_contact_actor_input": False,
        "c3_privileged_compression_teacher_training_only": True,
        "c3_boundary_maximum_topology_phase_kl_limit": (
            fine_tune.maximum_topology_phase_kl
        ),
        "c3_boundary_maximum_non_target_topology_phase_kl_observed": max(
            row["maximum_non_target_topology_phase_kl"] for row in rows
        ),
        "c3_boundary_maximum_topology_phase_kl_observed": max(
            row["maximum_topology_phase_kl"] for row in rows
        ),
        "c3_boundary_maximum_applied_non_target_topology_phase_kl": max(
            (
                row["maximum_non_target_topology_phase_kl"]
                for row in applied
            ),
            default=0.0,
        ),
        "c3_boundary_maximum_applied_topology_phase_kl": max(
            (row["maximum_topology_phase_kl"] for row in applied),
            default=0.0,
        ),
        "c3_boundary_optimizer_step_rollback": True,
    }


def _tensor_gae(
    artifact: Order9TensorRolloutArtifact,
    config: Order9PPOOptimizationConfig,
    *,
    reward_override: torch.Tensor | None = None,
    value_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = artifact.tensors
    valid = tensors["valid"]
    reward = (
        tensors["reward"].float()
        if reward_override is None
        else reward_override.float()
    )
    if reward.shape != tensors["reward"].shape:
        raise SchemaValidationError("Order9 tensor GAE reward shape differs")
    if not math.isfinite(float(value_scale)) or value_scale < 0.0:
        raise ValueError("Order9 tensor GAE value scale must be finite and non-negative")
    value = tensors["old_value"].float() * float(value_scale)
    terminal = tensors["terminal"]
    truncated = tensors["truncated"]
    bootstrap = tensors["bootstrap_value"].float() * float(value_scale)
    serial = tensors["episode_serial"]
    advantages = torch.zeros_like(value)
    next_advantage = torch.zeros(value.shape[1], dtype=value.dtype)
    for time_index in range(value.shape[0] - 1, -1, -1):
        row_valid = valid[time_index]
        has_same_next = torch.zeros_like(row_valid)
        next_value = torch.zeros_like(value[time_index])
        if time_index + 1 < value.shape[0]:
            has_same_next = (
                valid[time_index + 1]
                & (serial[time_index + 1] == serial[time_index])
            )
            next_value = torch.where(
                has_same_next, value[time_index + 1], next_value
            )
        non_boundary = row_valid & ~terminal[time_index] & ~truncated[time_index]
        if bool((non_boundary & ~has_same_next).any()):
            raise SchemaValidationError(
                "Order9 tensor GAE fragment lacks a terminal/truncation boundary"
            )
        next_value = torch.where(
            truncated[time_index], bootstrap[time_index], next_value
        )
        recurrence = non_boundary.to(value.dtype)
        delta = reward[time_index] + config.gamma * next_value - value[time_index]
        current = delta + config.gamma * config.gae_lambda * recurrence * next_advantage
        current = torch.where(row_valid, current, torch.zeros_like(current))
        advantages[time_index] = current
        next_advantage = current
    returns = advantages + value
    return advantages, returns


__all__ = [
    "ORDER9_C3_BOUNDARY_FINE_TUNE_CONTRACT",
    "ORDER9_CONTACT_HEAD_EXTRA_OPTIMIZER_CONTRACT",
    "ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES",
    "ORDER9_TENSOR_PI_L_PPO_VERSION",
    "ORDER9_TOPOLOGY_MINIBATCH_PACKING",
    "order9_privileged_compression_teacher_direction",
    "update_order9_tensor_pi_l_ppo",
]
