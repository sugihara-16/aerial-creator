from __future__ import annotations

"""Task/phase-conditioned morphology actor-critic for Order 9 pi_L."""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch.distributions import Normal

from amsrr.encoders.morphology_graph_encoder import MorphologyGraphBatch
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
    MorphologyConditionedActorCritic,
    Order3MorphologyConditionedPolicyConfig,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.task_spec import TaskType


ORDER9_PI_L_POLICY_VERSION = "order9_phase_conditioned_policy_command_pi_l_v2"
ORDER9_MAX_PHASE_COUNT = 16
ORDER9_GLOBAL_ACTION_NAMES: tuple[str, ...] = (
    "centroidal_position_correction_world.x",
    "centroidal_position_correction_world.y",
    "centroidal_position_correction_world.z",
    "centroidal_orientation_correction_body.rx",
    "centroidal_orientation_correction_body.ry",
    "centroidal_orientation_correction_body.rz",
    "centroidal_twist_correction.linear_world.x",
    "centroidal_twist_correction.linear_world.y",
    "centroidal_twist_correction.linear_world.z",
    "centroidal_twist_correction.angular_body.x",
    "centroidal_twist_correction.angular_body.y",
    "centroidal_twist_correction.angular_body.z",
    "centroidal_wrench_bias_body.force.x",
    "centroidal_wrench_bias_body.force.y",
    "centroidal_wrench_bias_body.force.z",
    "centroidal_wrench_bias_body.torque.x",
    "centroidal_wrench_bias_body.torque.y",
    "centroidal_wrench_bias_body.torque.z",
)
ORDER9_GLOBAL_ACTION_SIZE = len(ORDER9_GLOBAL_ACTION_NAMES)
_EPSILON = 1.0e-6


@dataclass
class Order9LowLevelPolicyConfig(Order3MorphologyConditionedPolicyConfig):
    action_size: int = ORDER9_GLOBAL_ACTION_SIZE
    centroidal_position_correction_limit_m: float = 0.05
    centroidal_orientation_correction_limit_rad: float = 0.25
    # Order 9 owns a complete PolicyCommand action.  Its bounds are applied
    # directly; the legacy Order-3 blend must not shrink or mix the command
    # with a deterministic pi_L output.
    trust_region_blend: float = 1.0
    max_phase_count: int = ORDER9_MAX_PHASE_COUNT
    joint_action_log_std_init: float = -2.0
    # The inherited field controlled whether the Order-3 joint decoder was
    # safety-masked.  Order 9 contact tasks intentionally enable the same
    # bounded local-joint output path.
    free_flight_joint_residual_enabled: bool = True

    @property
    def expected_action_size(self) -> int:
        return ORDER9_GLOBAL_ACTION_SIZE

    def validate(self) -> None:
        super().validate()
        if self.max_phase_count < 1:
            raise SchemaValidationError("Order9 max_phase_count must be positive")
        if not math.isfinite(self.joint_action_log_std_init):
            raise SchemaValidationError("Order9 joint_action_log_std_init must be finite")
        for name in (
            "centroidal_position_correction_limit_m",
            "centroidal_orientation_correction_limit_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9 {name} must be finite and positive"
                )
        if not math.isclose(self.trust_region_blend, 1.0, abs_tol=1.0e-12):
            raise SchemaValidationError(
                "Order9 complete PolicyCommand must use trust_region_blend=1"
            )
        if not self.free_flight_joint_residual_enabled:
            raise SchemaValidationError(
                "Order9 contact-task pi_L requires the bounded joint action path"
            )

    @property
    def phase_feature_dim(self) -> int:
        return len(TaskType) + self.max_phase_count + 3


@dataclass(frozen=True)
class Order9LowLevelActorCriticStep:
    action: torch.Tensor
    action_mean: torch.Tensor
    joint_action: torch.Tensor
    joint_action_mean: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    recurrent_state: torch.Tensor
    graph_encoding: object

    @property
    def joint_residuals(self) -> torch.Tensor:
        """Compatibility alias consumed by the v2 PolicyCommand decoder."""

        return self.joint_action


class Order9PhaseConditionedActorCritic(MorphologyConditionedActorCritic):
    """Morphology trunk with a complete phase-conditioned PolicyCommand head."""

    def __init__(self, config: Order9LowLevelPolicyConfig | None = None) -> None:
        resolved = config or Order9LowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved
        self.phase_encoder = torch.nn.Sequential(
            torch.nn.Linear(resolved.phase_feature_dim, resolved.graph_hidden_dim),
            torch.nn.LayerNorm(resolved.graph_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(resolved.graph_hidden_dim, resolved.graph_hidden_dim),
        )
        self.joint_actor_log_std = torch.nn.Parameter(
            torch.full(
                (3 * resolved.max_local_joint_slots,),
                float(resolved.joint_action_log_std_init),
            )
        )

    def initialize_from_order3(
        self,
        source: MorphologyConditionedActorCritic,
    ) -> tuple[list[str], list[str]]:
        """Warm-start shape-compatible encoders while keeping the v2 head fresh."""

        source_state = source.state_dict()
        target_state = self.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        incompatible = self.load_state_dict(compatible, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise SchemaValidationError(
                f"Order3 initialization has unexpected parameters: {unexpected}"
            )
        missing = sorted(incompatible.missing_keys)
        required_fresh_prefixes = (
            "actor_log_std",
            "actor_mean.",
            "fusion.0.",
            "joint_actor_log_std",
            "phase_encoder.",
        )
        if any(
            not key.startswith(required_fresh_prefixes)
            for key in missing
        ):
            raise SchemaValidationError(
                f"Order3 initialization missing unexpected parameters: {missing}"
            )
        return missing, unexpected

    def phase_feature_tensor(
        self,
        context: LowLevelPolicyContext,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(
            [order9_phase_actor_feature_vector(context, self.config)],
            dtype=dtype,
            device=device,
        )

    def step(
        self,
        morphologies,
        runtime_observations,
        actor_features: torch.Tensor,
        previous_action: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        phase_features: torch.Tensor,
        privileged_disturbance_body: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        joint_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Order9LowLevelActorCriticStep:
        graph_encoding = self.graph_encoder(
            morphologies,
            runtime_observations=(
                None
                if isinstance(morphologies, MorphologyGraphBatch)
                else runtime_observations
            ),
        )
        batch_size = graph_encoding.global_embedding.shape[0]
        device = graph_encoding.global_embedding.device
        dtype = graph_encoding.global_embedding.dtype
        actor_features = actor_features.to(device=device, dtype=dtype)
        phase_features = phase_features.to(device=device, dtype=dtype)
        previous_action = previous_action.to(device=device, dtype=dtype)
        recurrent_state = recurrent_state.to(device=device, dtype=dtype)
        _require_shape(
            actor_features,
            (batch_size, len(ORDER3_ACTOR_FEATURE_NAMES)),
            "actor_features",
        )
        _require_shape(
            phase_features,
            (batch_size, self.config.phase_feature_dim),
            "phase_features",
        )
        _require_shape(
            previous_action,
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            "previous_action",
        )
        _require_shape(
            recurrent_state,
            (batch_size, self.config.recurrent_hidden_dim),
            "recurrent_state",
        )
        feature_embedding = self.actor_feature_encoder(actor_features)
        feature_embedding = feature_embedding + self.phase_encoder(phase_features)
        fused = self.fusion(
            torch.cat(
                (graph_encoding.global_embedding, feature_embedding, previous_action),
                dim=-1,
            )
        )
        next_state = self.recurrent(fused, recurrent_state)
        raw_mean = self.actor_mean(next_state)
        std = torch.exp(
            torch.clamp(self.actor_log_std, min=-6.0, max=1.0)
        ).expand_as(raw_mean)
        distribution = Normal(raw_mean, std)
        bounded_action, raw_action = _sample_or_evaluate_squashed(
            distribution,
            raw_mean,
            action,
            expected_shape=(batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            deterministic=deterministic,
            label="action",
        )
        global_log_prob = _squashed_log_prob(
            distribution, raw_action, bounded_action
        ).sum(dim=-1)
        global_entropy = distribution.entropy().sum(dim=-1)

        recurrent_tokens = next_state.unsqueeze(1).expand(
            -1, graph_encoding.node_embeddings.shape[1], -1
        )
        joint_raw_mean = self.joint_decoder(
            torch.cat((graph_encoding.node_embeddings, recurrent_tokens), dim=-1)
        )
        joint_std = torch.exp(
            torch.clamp(self.joint_actor_log_std, min=-6.0, max=1.0)
        ).reshape(1, 1, -1).expand_as(joint_raw_mean)
        joint_distribution = Normal(joint_raw_mean, joint_std)
        bounded_joint, raw_joint = _sample_or_evaluate_squashed(
            joint_distribution,
            joint_raw_mean,
            joint_action,
            expected_shape=tuple(joint_raw_mean.shape),
            deterministic=deterministic,
            label="joint_action",
        )
        node_mask = graph_encoding.mask.unsqueeze(-1).to(dtype)
        bounded_joint = bounded_joint * node_mask
        joint_log_prob = (
            _squashed_log_prob(joint_distribution, raw_joint, bounded_joint)
            * node_mask
        ).sum(dim=(1, 2))
        joint_entropy = (joint_distribution.entropy() * node_mask).sum(dim=(1, 2))

        privileged = (
            torch.zeros((batch_size, 6), device=device, dtype=dtype)
            if privileged_disturbance_body is None
            else privileged_disturbance_body.to(device=device, dtype=dtype)
        )
        _require_shape(privileged, (batch_size, 6), "privileged_disturbance_body")
        value = self.critic(torch.cat((next_state, privileged), dim=-1)).squeeze(-1)
        return Order9LowLevelActorCriticStep(
            action=bounded_action,
            action_mean=torch.tanh(raw_mean),
            joint_action=bounded_joint,
            joint_action_mean=torch.tanh(joint_raw_mean) * node_mask,
            log_prob=global_log_prob + joint_log_prob,
            entropy=global_entropy + joint_entropy,
            value=value,
            recurrent_state=next_state,
            graph_encoding=graph_encoding,
        )


def order9_phase_actor_feature_vector(
    context: LowLevelPolicyContext,
    config: Order9LowLevelPolicyConfig,
) -> list[float]:
    if context.task_type is None or context.task_adapter_id is None:
        raise SchemaValidationError("Order9 pi_L requires task_type and task_adapter_id")
    if context.phase_index is None or context.phase_count is None:
        raise SchemaValidationError("Order9 pi_L requires explicit phase index/count")
    try:
        task_type = TaskType(context.task_type)
    except ValueError as exc:
        raise SchemaValidationError("Order9 pi_L task_type is invalid") from exc
    if (
        context.phase_count < 1
        or context.phase_count > config.max_phase_count
        or not 0 <= context.phase_index < context.phase_count
    ):
        raise SchemaValidationError("Order9 pi_L phase index/count is invalid")
    phase_one_hot = [0.0] * config.max_phase_count
    phase_one_hot[context.phase_index] = 1.0
    progress = min(
        max(float(context.runtime_observation.task_progress.progress_ratio), 0.0),
        1.0,
    )
    adapter = _stable_scalar(context.task_adapter_id)
    return [
        *[1.0 if task_type == item else 0.0 for item in TaskType],
        *phase_one_hot,
        progress,
        math.sin(2.0 * math.pi * adapter),
        math.cos(2.0 * math.pi * adapter),
    ]


def _sample_or_evaluate_squashed(
    distribution: Normal,
    raw_mean: torch.Tensor,
    supplied: torch.Tensor | None,
    *,
    expected_shape: tuple[int, ...],
    deterministic: bool,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if supplied is None:
        raw = raw_mean if deterministic else distribution.rsample()
        return torch.tanh(raw), raw
    bounded = supplied.to(device=raw_mean.device, dtype=raw_mean.dtype)
    _require_shape(bounded, expected_shape, label)
    if bool((bounded.abs() > 1.0 + 1.0e-6).any().item()):
        raise ValueError(f"{label} must be normalized to [-1, 1]")
    raw = torch.atanh(torch.clamp(bounded, -1.0 + _EPSILON, 1.0 - _EPSILON))
    return bounded, raw


def _squashed_log_prob(
    distribution: Normal,
    raw: torch.Tensor,
    bounded: torch.Tensor,
) -> torch.Tensor:
    return distribution.log_prob(raw) - torch.log(
        torch.clamp(1.0 - bounded.square(), min=_EPSILON)
    )


def _require_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    label: str,
) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{label} must have shape {expected}, got {tuple(tensor.shape)}")


def _stable_scalar(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)
