from __future__ import annotations

"""Task/phase-conditioned morphology actor-critic for Order 9 pi_L."""

import hashlib
import math
from dataclasses import dataclass, replace

import torch
from torch.distributions import Categorical, Normal

from amsrr.encoders.morphology_graph_encoder import MorphologyGraphBatch
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
    MorphologyConditionedActorCritic,
    Order3MorphologyConditionedPolicyConfig,
)
from amsrr.policies.order9_active_knot_features import (
    ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES,
    ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION,
    ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES,
    order9_active_knot_feature_vectors,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.task_spec import TaskType


ORDER9_PI_L_POLICY_VERSION = "order9_phase_conditioned_policy_command_pi_l_v2"
ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION = (
    "order9_active_knot_policy_command_pi_l_v4"
)
ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION = (
    "order9_active_knot_contact_residual_pi_l_v5"
)
ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION = (
    "order9_morphology_invariant_contact_compression_pi_l_v6"
)
ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION = (
    "order9_contact_space_projected_policy_command_pi_l_v7"
)
ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION = (
    "order9_contact_feedback_policy_command_pi_l_v8"
)
ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION = (
    "order9_categorical_contact_normal_policy_command_pi_l_v9"
)
ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS: tuple[str, ...] = (
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
)
ORDER9_MAX_CONTACT_SLOTS = 16
ORDER9_CONTACT_SPACE_ACTION_SIZE = 6
ORDER9_CONTACT_SPACE_ACTION_NAMES: tuple[str, ...] = (
    "translation.inward_normal",
    "translation.tangent_1",
    "translation.tangent_2",
    "rotation.inward_normal",
    "rotation.tangent_1",
    "rotation.tangent_2",
)
ORDER9_CONTACT_SPACE_FEATURE_NAMES: tuple[str, ...] = (
    "contact.normal_world.x",
    "contact.normal_world.y",
    "contact.normal_world.z",
    "contact.tangent_1_world.x",
    "contact.tangent_1_world.y",
    "contact.tangent_1_world.z",
    "contact.tangent_2_world.x",
    "contact.tangent_2_world.y",
    "contact.tangent_2_world.z",
    "anchor.local_position.x",
    "anchor.local_position.y",
    "anchor.local_position.z",
    "contact.relative_position.x",
    "contact.relative_position.y",
    "contact.relative_position.z",
    "wrench.target.force.x",
    "wrench.target.force.y",
    "wrench.target.force.z",
    "wrench.target.torque.x",
    "wrench.target.torque.y",
    "wrench.target.torque.z",
    "wrench.half_range.force.x",
    "wrench.half_range.force.y",
    "wrench.half_range.force.z",
    "wrench.half_range.torque.x",
    "wrench.half_range.torque.y",
    "wrench.half_range.torque.z",
    "contact.friction",
    "contact.patch_area_m2",
    "contact.priority",
)
ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES: tuple[str, ...] = (
    "feedback.signed_surface_distance_over_normal_span",
    "feedback.relative_linear_velocity_contact.x",
    "feedback.relative_linear_velocity_contact.y",
    "feedback.relative_linear_velocity_contact.z",
    "feedback.relative_angular_velocity_contact.x",
    "feedback.relative_angular_velocity_contact.y",
    "feedback.relative_angular_velocity_contact.z",
    "feedback.signed_log_compression_load_proxy_nm",
)
ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES: tuple[str, ...] = (
    *ORDER9_CONTACT_SPACE_FEATURE_NAMES,
    *ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES,
)
# Runtime actor phases: establish_contact, lift, transport, and place.
# The residual is identically zero during approach/release/retreat/settle.
ORDER9_CONTACT_RESIDUAL_PHASE_INDICES: tuple[int, ...] = (1, 3, 4, 5)
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


@dataclass
class Order9ActiveKnotLowLevelPolicyConfig(Order9LowLevelPolicyConfig):
    active_knot_feature_contract_version: str = (
        ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION
    )
    active_knot_global_feature_dim: int = len(
        ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
    )
    active_assignment_feature_dim: int = len(
        ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES
    )
    max_contact_slots: int = ORDER9_MAX_CONTACT_SLOTS
    contact_space_feature_dim: int = len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
    contact_space_action_log_std_init: float = -2.0

    @property
    def expected_contact_space_feature_dim(self) -> int:
        return len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)

    def validate(self) -> None:
        super().validate()
        if (
            self.active_knot_feature_contract_version
            != ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION
        ):
            raise SchemaValidationError(
                "Order9 active-knot feature contract version mismatch"
            )
        if self.active_knot_global_feature_dim != len(
            ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
        ):
            raise SchemaValidationError(
                "Order9 active-knot global feature width mismatch"
            )
        if self.active_assignment_feature_dim != len(
            ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES
        ):
            raise SchemaValidationError(
                "Order9 active-assignment feature width mismatch"
            )
        if self.max_contact_slots < 1:
            raise SchemaValidationError(
                "Order9 max_contact_slots must be positive"
            )
        if self.contact_space_feature_dim != self.expected_contact_space_feature_dim:
            raise SchemaValidationError(
                "Order9 contact-space feature width mismatch"
            )
        if not math.isfinite(self.contact_space_action_log_std_init):
            raise SchemaValidationError(
                "Order9 contact-space action log std must be finite"
            )


@dataclass
class Order9ContactFeedbackLowLevelPolicyConfig(
    Order9ActiveKnotLowLevelPolicyConfig
):
    """v8 contact actor configuration with deployable closed-loop feedback."""

    contact_space_feature_dim: int = len(
        ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES
    )

    @property
    def expected_contact_space_feature_dim(self) -> int:
        return len(ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES)


@dataclass
class Order9CategoricalContactNormalLowLevelPolicyConfig(
    Order9ContactFeedbackLowLevelPolicyConfig
):
    """v9 direct categorical inward-normal action configuration."""

    contact_normal_category_count: int = 81
    contact_normal_category_min_normalized: float = -1.0
    contact_normal_category_max_normalized: float = 1.0
    contact_normal_category_initial_std_normalized: float = 0.30
    contact_normal_category_prior_mode: str = "gaussian_centered"
    contact_normal_category_uniform_window_min_normalized: float = 0.50
    contact_normal_category_uniform_window_max_normalized: float = 1.00
    contact_normal_category_uniform_outside_logit_penalty: float = 4.0

    def validate(self) -> None:
        super().validate()
        if self.contact_normal_category_count != 81:
            raise SchemaValidationError(
                "Order9 categorical contact-normal action requires 81 bins"
            )
        if (
            self.contact_normal_category_min_normalized != -1.0
            or self.contact_normal_category_max_normalized != 1.0
        ):
            raise SchemaValidationError(
                "Order9 categorical contact-normal bounds must be [-1, 1]"
            )
        if not 0.0 < self.contact_normal_category_initial_std_normalized <= 1.0:
            raise SchemaValidationError(
                "Order9 categorical contact-normal initial std is invalid"
            )
        if self.contact_normal_category_prior_mode not in {
            "gaussian_centered",
            "uniform_window",
        }:
            raise SchemaValidationError(
                "Order9 categorical contact-normal prior mode is invalid"
            )
        window_min = self.contact_normal_category_uniform_window_min_normalized
        window_max = self.contact_normal_category_uniform_window_max_normalized
        if (
            not math.isfinite(window_min)
            or not math.isfinite(window_max)
            or window_min < self.contact_normal_category_min_normalized
            or window_max > self.contact_normal_category_max_normalized
            or window_min >= window_max
        ):
            raise SchemaValidationError(
                "Order9 categorical contact-normal uniform window is invalid"
            )
        outside_penalty = (
            self.contact_normal_category_uniform_outside_logit_penalty
        )
        if not math.isfinite(outside_penalty) or outside_penalty <= 0.0:
            raise SchemaValidationError(
                "Order9 categorical contact-normal outside penalty is invalid"
            )


@dataclass(frozen=True)
class Order9LowLevelActorCriticStep:
    action: torch.Tensor
    action_mean: torch.Tensor
    joint_action: torch.Tensor
    joint_action_mean: torch.Tensor
    contact_compression_residual_action: torch.Tensor
    contact_compression_residual_action_mean: torch.Tensor
    contact_space_residual_action: torch.Tensor
    contact_space_residual_action_mean: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    centroidal_log_prob: torch.Tensor
    centroidal_entropy: torch.Tensor
    posture_log_prob: torch.Tensor
    posture_entropy: torch.Tensor
    contact_log_prob: torch.Tensor
    contact_entropy: torch.Tensor
    value: torch.Tensor
    recurrent_state: torch.Tensor
    graph_encoding: object
    # Present only for the v9 categorical inward-normal actor.  This exposes
    # its normalized category logits to training without changing deployment.
    contact_normal_category_logits: torch.Tensor | None = None

    @property
    def joint_residuals(self) -> torch.Tensor:
        """Compatibility alias consumed by the v2 PolicyCommand decoder."""

        return self.joint_action


class Order9PhaseConditionedActorCritic(MorphologyConditionedActorCritic):
    """Morphology trunk with a complete phase-conditioned PolicyCommand head."""

    policy_version = ORDER9_PI_L_POLICY_VERSION

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
        active_knot_features: torch.Tensor | None = None,
        active_assignment_features: torch.Tensor | None = None,
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
        feature_embedding, node_embeddings = self._augment_active_knot_context(
            feature_embedding=feature_embedding,
            node_embeddings=graph_encoding.node_embeddings,
            node_mask=graph_encoding.mask,
            active_knot_features=active_knot_features,
            active_assignment_features=active_assignment_features,
        )
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
            -1, node_embeddings.shape[1], -1
        )
        joint_decoder_input = torch.cat(
            (node_embeddings, recurrent_tokens), dim=-1
        )
        joint_raw_mean = self._joint_raw_mean(
            joint_decoder_input=joint_decoder_input,
            phase_features=phase_features,
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
        # The privileged value fit must not move the deployable actor's graph,
        # feature, fusion, or recurrent representation.  The critic head still
        # learns from the exact actor state, but its gradient terminates here.
        value = self.critic(
            torch.cat((next_state.detach(), privileged), dim=-1)
        ).squeeze(-1)
        return Order9LowLevelActorCriticStep(
            action=bounded_action,
            action_mean=torch.tanh(raw_mean),
            joint_action=bounded_joint,
            joint_action_mean=torch.tanh(joint_raw_mean) * node_mask,
            contact_compression_residual_action=torch.zeros(
                (batch_size,), device=device, dtype=dtype
            ),
            contact_compression_residual_action_mean=torch.zeros(
                (batch_size,), device=device, dtype=dtype
            ),
            contact_space_residual_action=torch.zeros(
                (
                    batch_size,
                    int(getattr(self.config, "max_contact_slots", ORDER9_MAX_CONTACT_SLOTS)),
                    ORDER9_CONTACT_SPACE_ACTION_SIZE,
                ),
                device=device,
                dtype=dtype,
            ),
            contact_space_residual_action_mean=torch.zeros(
                (
                    batch_size,
                    int(getattr(self.config, "max_contact_slots", ORDER9_MAX_CONTACT_SLOTS)),
                    ORDER9_CONTACT_SPACE_ACTION_SIZE,
                ),
                device=device,
                dtype=dtype,
            ),
            log_prob=global_log_prob + joint_log_prob,
            entropy=global_entropy + joint_entropy,
            centroidal_log_prob=global_log_prob,
            centroidal_entropy=global_entropy,
            posture_log_prob=joint_log_prob,
            posture_entropy=joint_entropy,
            contact_log_prob=torch.zeros_like(global_log_prob),
            contact_entropy=torch.zeros_like(global_entropy),
            value=value,
            recurrent_state=next_state,
            graph_encoding=graph_encoding,
        )

    def _augment_active_knot_context(
        self,
        *,
        feature_embedding: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        active_knot_features: torch.Tensor | None,
        active_assignment_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del node_mask
        if (
            active_knot_features is not None
            or active_assignment_features is not None
        ):
            raise ValueError(
                "legacy Order9 pi_L does not accept active-knot feature tensors"
            )
        return feature_embedding, node_embeddings

    def _joint_raw_mean(
        self,
        *,
        joint_decoder_input: torch.Tensor,
        phase_features: torch.Tensor,
    ) -> torch.Tensor:
        del phase_features
        return self.joint_decoder(joint_decoder_input)


class Order9ActiveKnotPhaseConditionedActorCritic(
    Order9PhaseConditionedActorCritic
):
    """C3 successor actor with an explicit deployable active-knot branch."""

    policy_version = ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION

    def __init__(
        self, config: Order9ActiveKnotLowLevelPolicyConfig | None = None
    ) -> None:
        resolved = config or Order9ActiveKnotLowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved
        self.active_knot_encoder = _zero_residual_encoder(
            resolved.active_knot_global_feature_dim,
            resolved.graph_hidden_dim,
        )
        self.active_assignment_encoder = _zero_residual_encoder(
            resolved.active_assignment_feature_dim,
            resolved.graph_hidden_dim,
        )

    def initialize_from_legacy_order9(
        self, source: Order9PhaseConditionedActorCritic
    ) -> tuple[list[str], list[str]]:
        if isinstance(source, Order9ActiveKnotPhaseConditionedActorCritic):
            raise SchemaValidationError(
                "Order9 active-knot migration source must be the legacy v2 actor"
            )
        source_state = source.state_dict()
        target_state = self.state_dict()
        for key, value in source_state.items():
            if key not in target_state or target_state[key].shape != value.shape:
                raise SchemaValidationError(
                    f"Order9 active-knot migration cannot copy {key!r}"
                )
        incompatible = self.load_state_dict(source_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = sorted(incompatible.missing_keys)
        if unexpected or any(
            not key.startswith(
                ("active_knot_encoder.", "active_assignment_encoder.")
            )
            for key in missing
        ):
            raise SchemaValidationError(
                "Order9 active-knot migration parameter boundary differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def active_knot_feature_tensors(
        self,
        context: LowLevelPolicyContext,
        *,
        device: torch.device,
        dtype: torch.dtype,
        control_model: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        body_pose = getattr(control_model, "body_pose_world", None)
        body_twist = getattr(control_model, "body_twist_world", None)
        features = order9_active_knot_feature_vectors(
            context,
            body_pose_world=body_pose,
            body_twist_world=body_twist,
        )
        return (
            torch.tensor(
                [features.global_features], device=device, dtype=dtype
            ),
            torch.tensor(
                [features.assignment_features], device=device, dtype=dtype
            ),
        )

    def _augment_active_knot_context(
        self,
        *,
        feature_embedding: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        active_knot_features: torch.Tensor | None,
        active_assignment_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            active_knot_features is None
            or active_assignment_features is None
        ):
            raise ValueError(
                "Order9 active-knot pi_L requires both active feature tensors"
            )
        batch_size, node_count, _ = node_embeddings.shape
        device, dtype = node_embeddings.device, node_embeddings.dtype
        active_global = active_knot_features.to(device=device, dtype=dtype)
        active_nodes = active_assignment_features.to(
            device=device, dtype=dtype
        )
        _require_shape(
            active_global,
            (batch_size, self.config.active_knot_global_feature_dim),
            "active_knot_features",
        )
        _require_shape(
            active_nodes,
            (
                batch_size,
                node_count,
                self.config.active_assignment_feature_dim,
            ),
            "active_assignment_features",
        )
        encoded_global = self.active_knot_encoder(active_global)
        encoded_nodes = self.active_assignment_encoder(active_nodes)
        mask = node_mask.unsqueeze(-1).to(dtype)
        encoded_nodes = encoded_nodes * mask
        pooled = encoded_nodes.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return (
            feature_embedding + encoded_global + pooled,
            node_embeddings + encoded_nodes,
        )


class Order9ContactResidualPhaseConditionedActorCritic(
    Order9ActiveKnotPhaseConditionedActorCritic
):
    """Active-knot actor with an exactly phase-gated contact residual head.

    The inherited actor is a frozen behavior prior during C3 boundary
    fine-tuning.  This zero-initialized branch can alter node-local joint
    action means only in phases that require grasp wrench regulation.
    """

    policy_version = ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION

    def __init__(
        self, config: Order9ActiveKnotLowLevelPolicyConfig | None = None
    ) -> None:
        resolved = config or Order9ActiveKnotLowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved
        input_dim = (
            resolved.graph_hidden_dim + resolved.recurrent_hidden_dim
        )
        output_dim = 3 * resolved.max_local_joint_slots
        self.contact_residual_decoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, resolved.recurrent_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(resolved.recurrent_hidden_dim, output_dim),
        )
        final = self.contact_residual_decoder[-1]
        if not isinstance(final, torch.nn.Linear):  # pragma: no cover
            raise RuntimeError("Order9 contact residual final layer is not linear")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def initialize_from_active_knot(
        self, source: Order9ActiveKnotPhaseConditionedActorCritic
    ) -> tuple[list[str], list[str]]:
        if isinstance(
            source, Order9ContactResidualPhaseConditionedActorCritic
        ):
            raise SchemaValidationError(
                "Order9 contact-residual migration source must be the v4 actor"
            )
        source_state = source.state_dict()
        target_state = self.state_dict()
        for key, value in source_state.items():
            if key not in target_state or target_state[key].shape != value.shape:
                raise SchemaValidationError(
                    f"Order9 contact-residual migration cannot copy {key!r}"
                )
        incompatible = self.load_state_dict(source_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = sorted(incompatible.missing_keys)
        if unexpected or any(
            not key.startswith("contact_residual_decoder.")
            for key in missing
        ):
            raise SchemaValidationError(
                "Order9 contact-residual migration parameter boundary differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def _joint_raw_mean(
        self,
        *,
        joint_decoder_input: torch.Tensor,
        phase_features: torch.Tensor,
    ) -> torch.Tensor:
        base = super()._joint_raw_mean(
            joint_decoder_input=joint_decoder_input,
            phase_features=phase_features,
        )
        phase_offset = len(TaskType)
        phase_gate = sum(
            phase_features[:, phase_offset + phase_index]
            for phase_index in ORDER9_CONTACT_RESIDUAL_PHASE_INDICES
        ).clamp(min=0.0, max=1.0)
        residual = self.contact_residual_decoder(joint_decoder_input)
        return base + residual * phase_gate.reshape(-1, 1, 1)


class Order9MorphologyInvariantCompressionActorCritic(
    Order9ContactResidualPhaseConditionedActorCritic
):
    """v5 actor with one topology-invariant grasp-compression action.

    The inherited joint action keeps its v5 semantics.  This additional scalar
    is decoded along the morphology-specific IK compression direction, so its
    action coordinate does not change when the module count or selected joint
    changes.  The mean head is zero-output initialized for deterministic
    behavior-preserving migration from v5.
    """

    policy_version = ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION

    def __init__(
        self, config: Order9ActiveKnotLowLevelPolicyConfig | None = None
    ) -> None:
        resolved = config or Order9ActiveKnotLowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved
        self.contact_compression_actor_mean = torch.nn.Sequential(
            torch.nn.Linear(
                resolved.recurrent_hidden_dim + resolved.graph_hidden_dim + 1,
                resolved.recurrent_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(resolved.recurrent_hidden_dim, 1),
        )
        final = self.contact_compression_actor_mean[-1]
        if not isinstance(final, torch.nn.Linear):  # pragma: no cover
            raise RuntimeError("Order9 common compression final layer is not linear")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        self.contact_compression_actor_log_std = torch.nn.Parameter(
            torch.tensor(float(resolved.joint_action_log_std_init))
        )
        self.contact_compression_module_count_bias = torch.nn.Parameter(
            torch.zeros(resolved.max_modules + 1)
        )

    def initialize_from_contact_residual(
        self, source: Order9ContactResidualPhaseConditionedActorCritic
    ) -> tuple[list[str], list[str]]:
        if type(source) is not Order9ContactResidualPhaseConditionedActorCritic:
            raise SchemaValidationError(
                "Order9 morphology-invariant compression migration source must "
                "be the exact v5 actor"
            )
        source_state = source.state_dict()
        target_state = self.state_dict()
        for key, value in source_state.items():
            if key not in target_state or target_state[key].shape != value.shape:
                raise SchemaValidationError(
                    "Order9 morphology-invariant compression migration cannot "
                    f"copy {key!r}"
                )
        incompatible = self.load_state_dict(source_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = sorted(incompatible.missing_keys)
        allowed_prefixes = (
            "contact_compression_actor_log_std",
            "contact_compression_actor_mean.",
            "contact_compression_module_count_bias",
        )
        if unexpected or any(
            not key.startswith(allowed_prefixes) for key in missing
        ):
            raise SchemaValidationError(
                "Order9 morphology-invariant compression migration parameter "
                f"boundary differs: missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def step(
        self,
        morphologies,
        runtime_observations,
        actor_features: torch.Tensor,
        previous_action: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        phase_features: torch.Tensor,
        active_knot_features: torch.Tensor | None = None,
        active_assignment_features: torch.Tensor | None = None,
        privileged_disturbance_body: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        joint_action: torch.Tensor | None = None,
        contact_compression_residual_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Order9LowLevelActorCriticStep:
        base = super().step(
            morphologies,
            runtime_observations,
            actor_features,
            previous_action,
            recurrent_state,
            phase_features=phase_features,
            active_knot_features=active_knot_features,
            active_assignment_features=active_assignment_features,
            privileged_disturbance_body=privileged_disturbance_body,
            action=action,
            joint_action=joint_action,
            deterministic=deterministic,
        )
        module_fraction = (
            base.graph_encoding.mask.sum(dim=1, keepdim=True).to(
                dtype=base.recurrent_state.dtype
            )
            / float(self.config.max_modules)
        )
        module_count = base.graph_encoding.mask.sum(dim=1).long().clamp(
            min=0, max=self.config.max_modules
        )
        raw_mean = self.contact_compression_actor_mean(
            torch.cat(
                (
                    base.recurrent_state,
                    base.graph_encoding.global_embedding,
                    module_fraction,
                ),
                dim=-1,
            )
        ).squeeze(-1) + self.contact_compression_module_count_bias.index_select(
            0, module_count
        )
        std = torch.exp(
            torch.clamp(
                self.contact_compression_actor_log_std, min=-6.0, max=1.0
            )
        ).expand_as(raw_mean)
        distribution = Normal(raw_mean, std)
        bounded, raw = _sample_or_evaluate_squashed(
            distribution,
            raw_mean,
            contact_compression_residual_action,
            expected_shape=tuple(raw_mean.shape),
            deterministic=deterministic,
            label="contact_compression_residual_action",
        )
        phase_offset = len(TaskType)
        phase_gate = sum(
            phase_features[:, phase_offset + phase_index]
            for phase_index in ORDER9_CONTACT_RESIDUAL_PHASE_INDICES
        ).clamp(min=0.0, max=1.0)
        bounded = bounded * phase_gate
        bounded_mean = torch.tanh(raw_mean) * phase_gate
        scalar_log_prob = _squashed_log_prob(
            distribution, raw, torch.tanh(raw)
        ) * phase_gate
        scalar_entropy = distribution.entropy() * phase_gate
        return replace(
            base,
            contact_compression_residual_action=bounded,
            contact_compression_residual_action_mean=bounded_mean,
            log_prob=base.log_prob + scalar_log_prob,
            entropy=base.entropy + scalar_entropy,
            contact_log_prob=scalar_log_prob,
            contact_entropy=scalar_entropy,
        )


class Order9ContactSpacePhaseConditionedActorCritic(
    Order9ActiveKnotPhaseConditionedActorCritic
):
    """Task-generic contact-space residual actor used by the v7 adapter.

    Contact closure/slip corrections have one unambiguous learned path: a
    padded set of contact-frame residual twists.  Centroidal pose/twist and
    per-joint posture residuals remain available, while the execution adapter
    removes contact-incompatible centroidal components and projects posture
    residuals into the active-contact nullspace.  Residual wrench and learned
    joint-torque coordinates intentionally do not contribute to the v7 policy
    density and are forced to zero by the decoder contract.
    """

    policy_version = ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION

    def __init__(
        self, config: Order9ActiveKnotLowLevelPolicyConfig | None = None
    ) -> None:
        resolved = config or Order9ActiveKnotLowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved
        hidden = resolved.recurrent_hidden_dim
        graph = resolved.graph_hidden_dim
        self.contact_space_feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(resolved.contact_space_feature_dim, graph),
            torch.nn.LayerNorm(graph),
            torch.nn.SiLU(),
        )
        self.contact_space_slot_embedding = torch.nn.Embedding(
            resolved.max_contact_slots, graph
        )
        self.contact_space_actor_mean = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3 * graph, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, ORDER9_CONTACT_SPACE_ACTION_SIZE),
        )
        final = self.contact_space_actor_mean[-1]
        if not isinstance(final, torch.nn.Linear):  # pragma: no cover
            raise RuntimeError("Order9 contact-space final layer is not linear")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        self.contact_space_actor_log_std = torch.nn.Parameter(
            torch.full(
                (ORDER9_CONTACT_SPACE_ACTION_SIZE,),
                float(resolved.contact_space_action_log_std_init),
            )
        )

    def initialize_from_morphology_invariant_compression(
        self, source: "Order9MorphologyInvariantCompressionActorCritic"
    ) -> tuple[list[str], list[str]]:
        """Copy the common v4 actor exactly and create a fresh contact head."""

        if type(source) is not Order9MorphologyInvariantCompressionActorCritic:
            raise SchemaValidationError(
                "Order9 v7 migration source must be the exact v6 actor"
            )
        source_state = source.state_dict()
        target_state = self.state_dict()
        contact_prefixes = (
            "contact_space_actor_log_std",
            "contact_space_actor_mean.",
            "contact_space_feature_encoder.",
            "contact_space_slot_embedding.",
        )
        copied = {
            key: source_state[key]
            for key in target_state
            if key in source_state and source_state[key].shape == target_state[key].shape
        }
        incompatible = self.load_state_dict(copied, strict=False)
        missing = sorted(incompatible.missing_keys)
        unexpected = sorted(incompatible.unexpected_keys)
        if unexpected or any(
            not key.startswith(contact_prefixes) for key in missing
        ):
            raise SchemaValidationError(
                "Order9 v7 migration parameter boundary differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def step(
        self,
        morphologies,
        runtime_observations,
        actor_features: torch.Tensor,
        previous_action: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        phase_features: torch.Tensor,
        active_knot_features: torch.Tensor | None = None,
        active_assignment_features: torch.Tensor | None = None,
        contact_slot_features: torch.Tensor | None = None,
        contact_slot_owner_module_indices: torch.Tensor | None = None,
        contact_slot_mask: torch.Tensor | None = None,
        privileged_disturbance_body: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        joint_action: torch.Tensor | None = None,
        contact_space_residual_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Order9LowLevelActorCriticStep:
        base = super().step(
            morphologies,
            runtime_observations,
            actor_features,
            previous_action,
            recurrent_state,
            phase_features=phase_features,
            active_knot_features=active_knot_features,
            active_assignment_features=active_assignment_features,
            privileged_disturbance_body=privileged_disturbance_body,
            action=action,
            joint_action=joint_action,
            deterministic=deterministic,
        )
        if (
            contact_slot_features is None
            or contact_slot_owner_module_indices is None
            or contact_slot_mask is None
        ):
            raise ValueError(
                "Order9 v7 pi_L requires contact features, owners, and mask"
            )
        batch = base.action.shape[0]
        slots = self.config.max_contact_slots
        device = base.action.device
        dtype = base.action.dtype
        features = contact_slot_features.to(device=device, dtype=dtype)
        owners = contact_slot_owner_module_indices.to(device=device, dtype=torch.long)
        mask = contact_slot_mask.to(device=device, dtype=torch.bool)
        _require_shape(
            features,
            (batch, slots, self.config.contact_space_feature_dim),
            "contact_slot_features",
        )
        _require_shape(owners, (batch, slots), "contact_slot_owner_module_indices")
        _require_shape(mask, (batch, slots), "contact_slot_mask")
        if bool((mask & ((owners < 0) | (owners >= base.graph_encoding.node_embeddings.shape[1]))).any()):
            raise ValueError("Order9 active contact owner index is invalid")
        safe_owners = owners.clamp(
            min=0, max=base.graph_encoding.node_embeddings.shape[1] - 1
        )
        owner_embedding = torch.gather(
            base.graph_encoding.node_embeddings,
            1,
            safe_owners.unsqueeze(-1).expand(
                -1, -1, base.graph_encoding.node_embeddings.shape[-1]
            ),
        )
        encoded = self.contact_space_feature_encoder(features)
        slot_ids = torch.arange(slots, device=device)
        slot_embedding = self.contact_space_slot_embedding(slot_ids).unsqueeze(0).expand(
            batch, -1, -1
        )
        recurrent = base.recurrent_state.unsqueeze(1).expand(-1, slots, -1)
        raw_mean = self.contact_space_actor_mean(
            torch.cat((recurrent, owner_embedding, encoded, slot_embedding), dim=-1)
        )
        std = torch.exp(
            torch.clamp(self.contact_space_actor_log_std, min=-6.0, max=1.0)
        ).reshape(1, 1, -1).expand_as(raw_mean)
        distribution = Normal(raw_mean, std)
        bounded, raw = _sample_or_evaluate_squashed(
            distribution,
            raw_mean,
            contact_space_residual_action,
            expected_shape=(batch, slots, ORDER9_CONTACT_SPACE_ACTION_SIZE),
            deterministic=deterministic,
            label="contact_space_residual_action",
        )
        slot_mask = mask.unsqueeze(-1).to(dtype)
        bounded = bounded * slot_mask
        bounded_mean = torch.tanh(raw_mean) * slot_mask

        # v7 density contains only coordinates that reach the physical command:
        # centroidal pose/twist, joint position/velocity, and active contacts.
        global_raw = torch.atanh(
            base.action[:, :12].clamp(-1.0 + _EPSILON, 1.0 - _EPSILON)
        )
        global_distribution = Normal(
            torch.atanh(
                base.action_mean[:, :12].clamp(
                    -1.0 + _EPSILON, 1.0 - _EPSILON
                )
            ),
            torch.exp(torch.clamp(self.actor_log_std[:12], min=-6.0, max=1.0)).expand(
                batch, -1
            ),
        )
        global_log_prob = _squashed_log_prob(
            global_distribution, global_raw, base.action[:, :12]
        ).sum(dim=-1)
        global_entropy = global_distribution.entropy().sum(dim=-1)
        joint_width = 2 * self.config.max_local_joint_slots
        joint_raw = torch.atanh(
            base.joint_action[:, :, :joint_width].clamp(
                -1.0 + _EPSILON, 1.0 - _EPSILON
            )
        )
        joint_distribution = Normal(
            torch.atanh(
                base.joint_action_mean[:, :, :joint_width].clamp(
                    -1.0 + _EPSILON, 1.0 - _EPSILON
                )
            ),
            torch.exp(
                torch.clamp(self.joint_actor_log_std[:joint_width], min=-6.0, max=1.0)
            ).reshape(1, 1, -1).expand_as(joint_raw),
        )
        node_mask = base.graph_encoding.mask.unsqueeze(-1).to(dtype)
        joint_log_prob = (
            _squashed_log_prob(
                joint_distribution,
                joint_raw,
                base.joint_action[:, :, :joint_width],
            )
            * node_mask
        ).sum(dim=(1, 2))
        joint_entropy = (
            joint_distribution.entropy() * node_mask
        ).sum(dim=(1, 2))
        contact_log_prob = (
            _squashed_log_prob(distribution, raw, torch.tanh(raw)) * slot_mask
        ).sum(dim=(1, 2))
        contact_entropy = (
            distribution.entropy() * slot_mask
        ).sum(dim=(1, 2))
        return replace(
            base,
            contact_space_residual_action=bounded,
            contact_space_residual_action_mean=bounded_mean,
            log_prob=global_log_prob + joint_log_prob + contact_log_prob,
            entropy=global_entropy + joint_entropy + contact_entropy,
            centroidal_log_prob=global_log_prob,
            centroidal_entropy=global_entropy,
            posture_log_prob=joint_log_prob,
            posture_entropy=joint_entropy,
            contact_log_prob=contact_log_prob,
            contact_entropy=contact_entropy,
        )


class Order9ContactFeedbackPhaseConditionedActorCritic(
    Order9ContactSpacePhaseConditionedActorCritic
):
    """v8 contact actor with deployable geometric/load feedback.

    The execution action remains exactly the v7 task-generic contact-space
    residual.  v8 only appends runtime geometry, relative motion, and a
    motor-load proxy to each contact slot.  No raw simulator contact force is
    admitted to the actor.
    """

    policy_version = ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION

    def __init__(
        self, config: Order9ContactFeedbackLowLevelPolicyConfig | None = None
    ) -> None:
        resolved = config or Order9ContactFeedbackLowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved

    def initialize_from_contact_space(
        self, source: Order9ContactSpacePhaseConditionedActorCritic
    ) -> tuple[list[str], list[str]]:
        """Copy v7 exactly and zero only the newly appended input columns."""

        if type(source) is not Order9ContactSpacePhaseConditionedActorCritic:
            raise SchemaValidationError(
                "Order9 v8 migration source must be the exact v7 actor"
            )
        source_state = source.state_dict()
        target_state = self.state_dict()
        expanded_name = "contact_space_feature_encoder.0.weight"
        source_width = len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
        target_width = len(ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES)
        if (
            source_state[expanded_name].shape[1] != source_width
            or target_state[expanded_name].shape[1] != target_width
            or source_state[expanded_name].shape[0]
            != target_state[expanded_name].shape[0]
        ):
            raise SchemaValidationError(
                "Order9 v8 contact feedback encoder migration shape differs"
            )
        migrated = {}
        for name, target_value in target_state.items():
            source_value = source_state.get(name)
            if name == expanded_name:
                value = torch.zeros_like(target_value)
                value[:, :source_width].copy_(source_state[name])
                migrated[name] = value
            elif source_value is not None and source_value.shape == target_value.shape:
                migrated[name] = source_value
            else:
                raise SchemaValidationError(
                    f"Order9 v8 migration cannot copy parameter {name!r}"
                )
        incompatible = self.load_state_dict(migrated, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise SchemaValidationError(
                "Order9 v8 migration produced an incomplete state_dict"
            )
        return [
            f"{expanded_name}[:, {source_width}:{target_width}]"
        ], []


class Order9CategoricalContactNormalPhaseConditionedActorCritic(
    Order9ContactFeedbackPhaseConditionedActorCritic
):
    """v9 contact actor with a direct 81-bin inward-normal distribution.

    The other five contact coordinates retain the v8 squashed-Normal density.
    Each active contact slot independently selects one normalized value from
    ``[-1, 1]`` at an exact interval of ``1 / 40``.  With the production
    20 mm contact-normal authority this is exactly 0.5 mm per category.
    """

    policy_version = ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION

    def __init__(
        self,
        config: Order9CategoricalContactNormalLowLevelPolicyConfig | None = None,
    ) -> None:
        resolved = config or Order9CategoricalContactNormalLowLevelPolicyConfig()
        resolved.validate()
        super().__init__(resolved)
        self.config = resolved
        hidden = resolved.recurrent_hidden_dim
        graph = resolved.graph_hidden_dim
        self.contact_normal_category_logits = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3 * graph, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, resolved.contact_normal_category_count),
        )
        final = self.contact_normal_category_logits[-1]
        if not isinstance(final, torch.nn.Linear):  # pragma: no cover
            raise RuntimeError("Order9 categorical contact final layer is not linear")
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def initialize_from_contact_feedback(
        self, source: Order9ContactFeedbackPhaseConditionedActorCritic
    ) -> tuple[list[str], list[str]]:
        """Copy v8 exactly and initialize only the categorical residual logits."""

        if type(source) is not Order9ContactFeedbackPhaseConditionedActorCritic:
            raise SchemaValidationError(
                "Order9 v9 migration source must be the exact v8 actor"
            )
        source_state = source.state_dict()
        copied = {
            name: value
            for name, value in source_state.items()
            if name in self.state_dict()
            and self.state_dict()[name].shape == value.shape
        }
        incompatible = self.load_state_dict(copied, strict=False)
        missing = sorted(incompatible.missing_keys)
        unexpected = sorted(incompatible.unexpected_keys)
        if unexpected or any(
            not name.startswith("contact_normal_category_logits.")
            for name in missing
        ):
            raise SchemaValidationError(
                "Order9 v9 migration parameter boundary differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def _contact_normal_prior_logits(
        self,
        category_values: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        """Return the configured full-support prior over normal-action bins."""

        if self.config.contact_normal_category_prior_mode == "gaussian_centered":
            sigma = float(
                self.config.contact_normal_category_initial_std_normalized
            )
            return -0.5 * ((category_values - center) / sigma).square()
        in_window = (
            category_values
            >= self.config.contact_normal_category_uniform_window_min_normalized
        ) & (
            category_values
            <= self.config.contact_normal_category_uniform_window_max_normalized
        )
        return torch.where(
            in_window,
            torch.zeros_like(category_values),
            torch.full_like(
                category_values,
                -float(
                    self.config.contact_normal_category_uniform_outside_logit_penalty
                ),
            ),
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
        active_knot_features: torch.Tensor | None = None,
        active_assignment_features: torch.Tensor | None = None,
        contact_slot_features: torch.Tensor | None = None,
        contact_slot_owner_module_indices: torch.Tensor | None = None,
        contact_slot_mask: torch.Tensor | None = None,
        privileged_disturbance_body: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        joint_action: torch.Tensor | None = None,
        contact_space_residual_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Order9LowLevelActorCriticStep:
        base = super().step(
            morphologies,
            runtime_observations,
            actor_features,
            previous_action,
            recurrent_state,
            phase_features=phase_features,
            active_knot_features=active_knot_features,
            active_assignment_features=active_assignment_features,
            contact_slot_features=contact_slot_features,
            contact_slot_owner_module_indices=contact_slot_owner_module_indices,
            contact_slot_mask=contact_slot_mask,
            privileged_disturbance_body=privileged_disturbance_body,
            action=action,
            joint_action=joint_action,
            contact_space_residual_action=contact_space_residual_action,
            deterministic=deterministic,
        )
        if (
            contact_slot_features is None
            or contact_slot_owner_module_indices is None
            or contact_slot_mask is None
        ):
            raise ValueError(
                "Order9 v9 pi_L requires contact features, owners, and mask"
            )
        batch = base.action.shape[0]
        slots = self.config.max_contact_slots
        device = base.action.device
        dtype = base.action.dtype
        features = contact_slot_features.to(device=device, dtype=dtype)
        owners = contact_slot_owner_module_indices.to(
            device=device, dtype=torch.long
        )
        mask = contact_slot_mask.to(device=device, dtype=torch.bool)
        safe_owners = owners.clamp(
            min=0, max=base.graph_encoding.node_embeddings.shape[1] - 1
        )
        owner_embedding = torch.gather(
            base.graph_encoding.node_embeddings,
            1,
            safe_owners.unsqueeze(-1).expand(
                -1, -1, base.graph_encoding.node_embeddings.shape[-1]
            ),
        )
        encoded = self.contact_space_feature_encoder(features)
        slot_ids = torch.arange(slots, device=device)
        slot_embedding = self.contact_space_slot_embedding(slot_ids).unsqueeze(0).expand(
            batch, -1, -1
        )
        recurrent = base.recurrent_state.unsqueeze(1).expand(-1, slots, -1)
        decoder_input = torch.cat(
            (recurrent, owner_embedding, encoded, slot_embedding), dim=-1
        )
        continuous_raw_mean = self.contact_space_actor_mean(decoder_input)
        continuous_std = torch.exp(
            torch.clamp(self.contact_space_actor_log_std, min=-6.0, max=1.0)
        ).reshape(1, 1, -1).expand_as(continuous_raw_mean)
        learned_logits = self.contact_normal_category_logits(decoder_input)
        category_values = torch.linspace(
            self.config.contact_normal_category_min_normalized,
            self.config.contact_normal_category_max_normalized,
            self.config.contact_normal_category_count,
            device=device,
            dtype=dtype,
        )
        center = base.contact_space_residual_action_mean[..., 0].unsqueeze(-1)
        prior_logits = self._contact_normal_prior_logits(category_values, center)
        distribution = Categorical(logits=prior_logits + learned_logits)

        if contact_space_residual_action is None:
            category_index = (
                distribution.logits.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
        else:
            supplied_normal = contact_space_residual_action[..., 0].to(
                device=device, dtype=dtype
            )
            category_step = (
                self.config.contact_normal_category_max_normalized
                - self.config.contact_normal_category_min_normalized
            ) / (self.config.contact_normal_category_count - 1)
            category_index = torch.round(
                (
                    supplied_normal
                    - self.config.contact_normal_category_min_normalized
                )
                / category_step
            ).to(torch.long)
            category_index = category_index.clamp(
                0, self.config.contact_normal_category_count - 1
            )
            reconstructed = category_values[category_index]
            if bool(
                (
                    mask
                    & (
                        (reconstructed - supplied_normal).abs()
                        > 8.0 * torch.finfo(dtype).eps
                    )
                ).any()
            ):
                raise ValueError(
                    "Order9 v9 replay contact-normal action is off category grid"
                )

        selected_normal = category_values[category_index] * mask.to(dtype)
        deterministic_index = distribution.logits.argmax(dim=-1)
        deterministic_normal = (
            category_values[deterministic_index] * mask.to(dtype)
        )
        bounded = base.contact_space_residual_action.clone()
        bounded[..., 0] = selected_normal
        bounded_mean = base.contact_space_residual_action_mean.clone()
        bounded_mean[..., 0] = deterministic_normal

        mask_float = mask.to(dtype)
        remaining_action = base.contact_space_residual_action[..., 1:]
        remaining_raw = torch.atanh(
            remaining_action.clamp(-1.0 + _EPSILON, 1.0 - _EPSILON)
        )
        remaining_distribution = Normal(
            continuous_raw_mean[..., 1:],
            continuous_std[..., 1:],
        )
        remaining_log_prob = (
            _squashed_log_prob(
                remaining_distribution,
                remaining_raw,
                remaining_action,
            )
            * mask_float.unsqueeze(-1)
        ).sum(dim=(1, 2))
        remaining_entropy = (
            remaining_distribution.entropy() * mask_float.unsqueeze(-1)
        ).sum(dim=(1, 2))
        categorical_log_prob = (
            distribution.log_prob(category_index) * mask_float
        ).sum(dim=1)
        categorical_entropy = (distribution.entropy() * mask_float).sum(dim=1)
        # Compute the mixed density directly.  Subtracting the provisional
        # continuous-normal density from ``base.contact_log_prob`` is not
        # replay-stable when a sampled action is close to tanh saturation:
        # the unsaved provisional sample cannot be reconstructed from the
        # categorical action.  Coordinates 1:6 are the actions that actually
        # reach the decoder, so their squashed-Normal density plus the direct
        # categorical density is the exact behavior likelihood.
        contact_log_prob = remaining_log_prob + categorical_log_prob
        contact_entropy = remaining_entropy + categorical_entropy
        return replace(
            base,
            contact_space_residual_action=bounded,
            contact_space_residual_action_mean=bounded_mean,
            log_prob=(
                base.centroidal_log_prob
                + base.posture_log_prob
                + contact_log_prob
            ),
            entropy=(
                base.centroidal_entropy
                + base.posture_entropy
                + contact_entropy
            ),
            contact_log_prob=contact_log_prob,
            contact_entropy=contact_entropy,
            contact_normal_category_logits=distribution.logits,
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


def _zero_residual_encoder(
    input_dim: int, hidden_dim: int
) -> torch.nn.Sequential:
    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
    )
    final = encoder[-1]
    if not isinstance(final, torch.nn.Linear):  # pragma: no cover
        raise RuntimeError("Order9 residual encoder final layer is not linear")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)
    return encoder


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
