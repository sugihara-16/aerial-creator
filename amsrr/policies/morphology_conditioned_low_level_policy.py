from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence

import torch
from torch import nn
from torch.distributions import Normal

from amsrr.controllers.rigid_body_model import RigidBodyControlModel, RigidBodyControlModelBuilder
from amsrr.encoders.morphology_graph_encoder import (
    MORPHOLOGY_EDGE_FEATURE_NAMES,
    MORPHOLOGY_GRAPH_ENCODER_VERSION,
    MORPHOLOGY_GRAPH_TENSORIZER_VERSION,
    MORPHOLOGY_NODE_FEATURE_NAMES,
    MorphologyGraphEncoder,
    MorphologyGraphEncoderOutput,
)
from amsrr.geometry.pose_math import matvec, quat_to_matrix, transpose
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    BaselineLowLevelPolicyConfig,
    LowLevelPolicyContext,
)
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_ACTION_SIZE,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3PolicyCheckpointMetadata,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.utils.hashing import hash_file, stable_hash


ORDER3_POLICY_OUTPUT_MODE = "bounded_centroidal_twist_wrench_residual_v1"
MORPHOLOGY_POLICY_RUNTIME_STATE_VERSION = (
    "morphology_conditioned_policy_runtime_state_v1"
)
_GRAVITY_MPS2 = 9.81
_EPSILON = 1.0e-6

ORDER3_ACTOR_FEATURE_NAMES: tuple[str, ...] = (
    "time.sin",
    "time.cos",
    "morphology.module_count_fraction",
    "centroidal.signed_log_mass",
    *(f"centroidal.signed_log_inertia.{index}" for index in range(6)),
    "target.position_error_world.x",
    "target.position_error_world.y",
    "target.position_error_world.z",
    "target.orientation_error_body.x",
    "target.orientation_error_body.y",
    "target.orientation_error_body.z",
    "state.linear_velocity_world.x",
    "state.linear_velocity_world.y",
    "state.linear_velocity_world.z",
    "state.angular_velocity_body.x",
    "state.angular_velocity_body.y",
    "state.angular_velocity_body.z",
    "target.linear_velocity_world.x",
    "target.linear_velocity_world.y",
    "target.linear_velocity_world.z",
    "target.angular_velocity_body.x",
    "target.angular_velocity_body.y",
    "target.angular_velocity_body.z",
    "target.linear_velocity_error_world.x",
    "target.linear_velocity_error_world.y",
    "target.linear_velocity_error_world.z",
    "target.angular_velocity_error_body.x",
    "target.angular_velocity_error_body.y",
    "target.angular_velocity_error_body.z",
    "controller.qp_feasible",
    "controller.status.ok",
    "controller.status.warning",
    "controller.status.infeasible",
    "controller.status.fault",
    "controller.signed_log_allocation_residual",
    "task.progress_ratio",
    "task.success",
)


@dataclass
class Order3MorphologyConditionedPolicyConfig(SchemaBase):
    graph_hidden_dim: int = 96
    graph_message_layers: int = 3
    recurrent_hidden_dim: int = 128
    max_modules: int = 8
    action_size: int = ORDER3_ACTION_SIZE
    update_rate_hz: float = 50.0
    linear_twist_correction_limit_mps: float = 0.5
    angular_twist_correction_limit_radps: float = 0.6
    residual_force_weight_fraction: float = 0.15
    residual_torque_per_module_nm: float = 0.25
    max_local_joint_slots: int = 4
    joint_position_delta_limit_rad: float = 0.15
    joint_velocity_limit_rad_s: float = 0.5
    joint_torque_fraction: float = 0.20
    free_flight_joint_residual_enabled: bool = False
    trust_region_blend: float = 0.10
    initial_log_std: float = -2.0
    ood_absolute_feature_limit: float = 1.0e6

    def validate(self) -> None:
        for name in (
            "graph_hidden_dim",
            "graph_message_layers",
            "recurrent_hidden_dim",
            "max_modules",
            "max_local_joint_slots",
        ):
            if int(getattr(self, name)) <= 0:
                raise SchemaValidationError(
                    f"Order3MorphologyConditionedPolicyConfig.{name} must be positive"
                )
        if self.max_modules != 8:
            raise SchemaValidationError(
                "Order3MorphologyConditionedPolicyConfig.max_modules must be 8"
            )
        if self.action_size != ORDER3_ACTION_SIZE:
            raise SchemaValidationError(
                f"Order3 policy action_size must be {ORDER3_ACTION_SIZE}"
            )
        for name in (
            "update_rate_hz",
            "linear_twist_correction_limit_mps",
            "angular_twist_correction_limit_radps",
            "residual_force_weight_fraction",
            "residual_torque_per_module_nm",
            "joint_position_delta_limit_rad",
            "joint_velocity_limit_rad_s",
            "joint_torque_fraction",
            "ood_absolute_feature_limit",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3MorphologyConditionedPolicyConfig.{name} must be finite and positive"
                )
        if not 0.0 < self.trust_region_blend <= 1.0:
            raise SchemaValidationError(
                "Order3MorphologyConditionedPolicyConfig.trust_region_blend must be in (0, 1]"
            )
        if not math.isfinite(self.initial_log_std):
            raise SchemaValidationError(
                "Order3MorphologyConditionedPolicyConfig.initial_log_std must be finite"
            )


@dataclass(frozen=True)
class Order3ActorCriticStep:
    action: torch.Tensor
    action_mean: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    recurrent_state: torch.Tensor
    joint_residuals: torch.Tensor
    graph_encoding: MorphologyGraphEncoderOutput


class MorphologyConditionedActorCritic(nn.Module):
    """Edge-aware recurrent actor/critic for Order-3 low-level intent.

    The actor emits only normalized twist/wrench residual coordinates and a
    masked non-vectoring-joint residual head. It cannot represent rotor thrust,
    vectoring targets, contact wrench, internal wrench, or actuator commands.
    """

    def __init__(
        self,
        config: Order3MorphologyConditionedPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Order3MorphologyConditionedPolicyConfig()
        self.graph_encoder = MorphologyGraphEncoder(
            d_model=self.config.graph_hidden_dim,
            message_passing_steps=self.config.graph_message_layers,
        )
        self.actor_feature_encoder = nn.Sequential(
            nn.Linear(len(ORDER3_ACTOR_FEATURE_NAMES), self.config.graph_hidden_dim),
            nn.LayerNorm(self.config.graph_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.graph_hidden_dim, self.config.graph_hidden_dim),
            nn.SiLU(),
        )
        fusion_width = (
            2 * self.config.graph_hidden_dim
            + ORDER3_ACTION_SIZE
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_width, self.config.recurrent_hidden_dim),
            nn.LayerNorm(self.config.recurrent_hidden_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(
            self.config.recurrent_hidden_dim,
            self.config.recurrent_hidden_dim,
        )
        self.actor_mean = nn.Linear(self.config.recurrent_hidden_dim, ORDER3_ACTION_SIZE)
        self.actor_log_std = nn.Parameter(
            torch.full((ORDER3_ACTION_SIZE,), float(self.config.initial_log_std))
        )
        self.critic = nn.Sequential(
            nn.Linear(self.config.recurrent_hidden_dim + 6, self.config.recurrent_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.recurrent_hidden_dim, 1),
        )
        self.joint_decoder = nn.Sequential(
            nn.Linear(
                self.config.graph_hidden_dim + self.config.recurrent_hidden_dim,
                self.config.recurrent_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.config.recurrent_hidden_dim,
                3 * self.config.max_local_joint_slots,
            ),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("Order3 recurrent batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            (batch_size, self.config.recurrent_hidden_dim),
            device=device or parameter.device,
            dtype=dtype or parameter.dtype,
        )

    def step(
        self,
        morphologies,
        runtime_observations: Sequence[RuntimeObservation],
        actor_features: torch.Tensor,
        previous_action: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        privileged_disturbance_body: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Order3ActorCriticStep:
        graph_encoding = self.graph_encoder(
            morphologies,
            runtime_observations=runtime_observations,
        )
        batch_size = graph_encoding.global_embedding.shape[0]
        device = graph_encoding.global_embedding.device
        dtype = graph_encoding.global_embedding.dtype
        actor_features = actor_features.to(device=device, dtype=dtype)
        previous_action = previous_action.to(device=device, dtype=dtype)
        recurrent_state = recurrent_state.to(device=device, dtype=dtype)
        _require_tensor_shape(
            actor_features,
            (batch_size, len(ORDER3_ACTOR_FEATURE_NAMES)),
            "actor_features",
        )
        _require_tensor_shape(
            previous_action,
            (batch_size, ORDER3_ACTION_SIZE),
            "previous_action",
        )
        _require_tensor_shape(
            recurrent_state,
            (batch_size, self.config.recurrent_hidden_dim),
            "recurrent_state",
        )
        feature_embedding = self.actor_feature_encoder(actor_features)
        fused = self.fusion(
            torch.cat(
                (graph_encoding.global_embedding, feature_embedding, previous_action),
                dim=-1,
            )
        )
        next_state = self.recurrent(fused, recurrent_state)
        raw_mean = self.actor_mean(next_state)
        std = torch.exp(torch.clamp(self.actor_log_std, min=-6.0, max=1.0)).expand_as(raw_mean)
        distribution = Normal(raw_mean, std)
        if action is None:
            raw_action = raw_mean if deterministic else distribution.rsample()
            bounded_action = torch.tanh(raw_action)
        else:
            bounded_action = action.to(device=device, dtype=dtype)
            _require_tensor_shape(
                bounded_action,
                (batch_size, ORDER3_ACTION_SIZE),
                "action",
            )
            if bool((bounded_action.abs() > 1.0 + 1.0e-6).any().item()):
                raise ValueError("Order3 action must be normalized to [-1, 1]")
            raw_action = torch.atanh(torch.clamp(bounded_action, -1.0 + _EPSILON, 1.0 - _EPSILON))
        log_prob = (
            distribution.log_prob(raw_action)
            - torch.log(torch.clamp(1.0 - bounded_action.square(), min=_EPSILON))
        ).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        privileged = (
            torch.zeros((batch_size, 6), device=device, dtype=dtype)
            if privileged_disturbance_body is None
            else privileged_disturbance_body.to(device=device, dtype=dtype)
        )
        _require_tensor_shape(privileged, (batch_size, 6), "privileged_disturbance_body")
        value = self.critic(torch.cat((next_state, privileged), dim=-1)).squeeze(-1)
        recurrent_tokens = next_state.unsqueeze(1).expand(
            -1,
            graph_encoding.node_embeddings.shape[1],
            -1,
        )
        joint_residuals = torch.tanh(
            self.joint_decoder(
                torch.cat((graph_encoding.node_embeddings, recurrent_tokens), dim=-1)
            )
        )
        joint_residuals = joint_residuals * graph_encoding.mask.unsqueeze(-1).to(dtype)
        return Order3ActorCriticStep(
            action=bounded_action,
            action_mean=torch.tanh(raw_mean),
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            recurrent_state=next_state,
            joint_residuals=joint_residuals,
            graph_encoding=graph_encoding,
        )


@dataclass(frozen=True)
class Order3PolicyInference:
    command: PolicyCommand
    previous_action: list[float]
    normalized_action: list[float]
    action_mean: list[float]
    log_prob: float
    value: float
    recurrent_state_in: list[float]
    recurrent_state_out: list[float]
    learned_policy_applied: bool
    fallback_reason: str | None
    normalized_joint_action: list[list[float]] = field(default_factory=list)
    joint_action_mean: list[list[float]] = field(default_factory=list)
    module_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Order3PolicyDiagnostics:
    learned_policy_applied: bool
    fallback_reason: str | None
    graph_id: str | None
    normalized_action: dict[str, float] = field(default_factory=dict)


class MorphologyConditionedLowLevelPolicy:
    """Stateful deployable pi_L wrapper with a strict deterministic v2 fallback."""

    def __init__(
        self,
        *,
        model: MorphologyConditionedActorCritic,
        physical_model: PhysicalModel,
        config: Order3MorphologyConditionedPolicyConfig | None = None,
        deterministic: bool = True,
        baseline_policy: BaselineLowLevelPolicy | None = None,
        allowed_structural_hashes: set[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.physical_model = physical_model
        self.config = config or model.config
        if self.config.to_dict() != model.config.to_dict():
            raise ValueError("Order3 wrapper/model policy configs must match")
        self.deterministic = bool(deterministic)
        self.baseline_policy = baseline_policy or BaselineLowLevelPolicy(
            BaselineLowLevelPolicyConfig(
                control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            )
        )
        self.allowed_structural_hashes = (
            frozenset(allowed_structural_hashes)
            if allowed_structural_hashes is not None
            else None
        )
        self.device = torch.device(device)
        self.rigid_body_builder = RigidBodyControlModelBuilder()
        self._hidden = self.model.initial_state(1, device=self.device)
        self._previous_action = torch.zeros(
            (1, ORDER3_ACTION_SIZE),
            dtype=self._hidden.dtype,
            device=self.device,
        )
        self._last_graph_id: str | None = None
        self._last_time_s: float | None = None
        self.checkpoint_metadata: Order3PolicyCheckpointMetadata | None = None
        self.checkpoint_sha256: str | None = None
        self.last_diagnostics = Order3PolicyDiagnostics(
            learned_policy_applied=False,
            fallback_reason="not_run",
            graph_id=None,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        physical_model: PhysicalModel,
        expected_sha256: str | None = None,
        deterministic: bool = True,
        device: torch.device | str = "cpu",
        baseline_policy: BaselineLowLevelPolicy | None = None,
    ) -> "MorphologyConditionedLowLevelPolicy":
        payload = load_order3_policy_checkpoint(
            checkpoint_path,
            device=device,
            expected_sha256=expected_sha256,
        )
        if payload.metadata.physical_model_hash != physical_model.stable_hash():
            raise SchemaValidationError(
                "Order3 checkpoint PhysicalModel hash does not match the runtime model"
            )
        if payload.metadata.urdf_hash != hash_file(physical_model.urdf_path):
            raise SchemaValidationError(
                "Order3 checkpoint URDF hash does not match the runtime robot"
            )
        resolved_baseline = baseline_policy or BaselineLowLevelPolicy(
            BaselineLowLevelPolicyConfig(
                control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            )
        )
        if stable_hash(resolved_baseline.config) != payload.metadata.fallback_config_hash:
            raise SchemaValidationError(
                "Order3 checkpoint fallback config hash does not match the runtime fallback"
            )
        instance = cls(
            model=payload.model,
            physical_model=physical_model,
            config=payload.config,
            deterministic=deterministic,
            baseline_policy=resolved_baseline,
            allowed_structural_hashes=_checkpoint_morphology_allowlist(
                payload.metadata
            ),
            device=device,
        )
        instance.checkpoint_metadata = payload.metadata
        instance.checkpoint_sha256 = payload.sha256
        return instance

    def reset(self) -> None:
        self._hidden = self.model.initial_state(1, device=self.device)
        self._previous_action.zero_()
        self._last_graph_id = None
        self._last_time_s = None

    def export_runtime_state(self) -> dict[str, object]:
        """Export recurrent behavior state for an isolated shadow rollout."""

        return {
            "runtime_state_version": MORPHOLOGY_POLICY_RUNTIME_STATE_VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "hidden": self._hidden.detach().cpu().tolist(),
            "previous_action": self._previous_action.detach().cpu().tolist(),
            "last_graph_id": self._last_graph_id,
            "last_time_s": self._last_time_s,
        }

    def restore_runtime_state(self, payload: dict[str, object]) -> None:
        """Restore recurrent behavior state after validating the whole payload."""

        if payload.get("runtime_state_version") != MORPHOLOGY_POLICY_RUNTIME_STATE_VERSION:
            raise SchemaValidationError("morphology policy runtime state version mismatch")
        checkpoint = payload.get("checkpoint_sha256")
        if checkpoint != self.checkpoint_sha256:
            raise SchemaValidationError(
                "morphology policy runtime state checkpoint mismatch"
            )
        hidden = _runtime_tensor(
            payload.get("hidden"),
            expected_shape=tuple(self._hidden.shape),
            device=self.device,
            dtype=self._hidden.dtype,
            name="hidden",
        )
        previous = _runtime_tensor(
            payload.get("previous_action"),
            expected_shape=tuple(self._previous_action.shape),
            device=self.device,
            dtype=self._previous_action.dtype,
            name="previous_action",
        )
        graph_id = payload.get("last_graph_id")
        if graph_id is not None and (not isinstance(graph_id, str) or not graph_id):
            raise SchemaValidationError("morphology policy last_graph_id is invalid")
        last_time = payload.get("last_time_s")
        if last_time is not None:
            if (
                not isinstance(last_time, (int, float))
                or not math.isfinite(float(last_time))
                or float(last_time) < 0.0
            ):
                raise SchemaValidationError("morphology policy last_time_s is invalid")
            last_time = float(last_time)
        self._hidden = hidden
        self._previous_action = previous
        self._last_graph_id = graph_id
        self._last_time_s = last_time

    def command(self, context: LowLevelPolicyContext) -> PolicyCommand:
        return self.command_with_trace(context).command

    def bootstrap_value(
        self,
        context: LowLevelPolicyContext,
        *,
        privileged_disturbance_body: Sequence[float] | None = None,
    ) -> float:
        """Evaluate the critic at a time-limit boundary without mutating policy state."""

        baseline = _strict_v2_fallback_command(
            self.baseline_policy.command(context),
            context.runtime_observation,
            self.physical_model,
        )
        reason = _runtime_fallback_reason(
            context,
            self.config,
            allowed_structural_hashes=self.allowed_structural_hashes,
        )
        if reason is not None or baseline.desired_body_pose is None:
            raise SchemaValidationError(
                f"Order3 bootstrap value is unavailable at an unsafe boundary: {reason}"
            )
        control_model = self.rigid_body_builder.build(
            context.morphology_graph,
            self.physical_model,
            context.runtime_observation,
        )
        features = order3_actor_feature_vector(
            context.runtime_observation,
            control_model,
            target_pose_world=baseline.desired_body_pose,
            target_twist=list(baseline.desired_body_twist or [0.0] * 6),
            max_modules=self.config.max_modules,
        )
        privileged = list(privileged_disturbance_body or [0.0] * 6)
        if len(privileged) != 6 or not all(math.isfinite(float(value)) for value in privileged):
            raise SchemaValidationError("Order3 bootstrap privileged disturbance must be finite")
        with torch.no_grad():
            step = self.model.step(
                [context.morphology_graph],
                [context.runtime_observation],
                torch.tensor([features], dtype=self._hidden.dtype, device=self.device),
                self._previous_action,
                self._hidden,
                privileged_disturbance_body=torch.tensor(
                    [privileged], dtype=self._hidden.dtype, device=self.device
                ),
                deterministic=True,
                **_optional_phase_step_kwargs(
                    self.model,
                    context,
                    device=self.device,
                    dtype=self._hidden.dtype,
                ),
            )
        value = float(step.value[0].detach().cpu().item())
        if not math.isfinite(value):
            raise SchemaValidationError("Order3 bootstrap critic value is non-finite")
        return value

    def command_with_trace(
        self,
        context: LowLevelPolicyContext,
        *,
        privileged_disturbance_body: Sequence[float] | None = None,
    ) -> Order3PolicyInference:
        baseline = _strict_v2_fallback_command(
            self.baseline_policy.command(context),
            context.runtime_observation,
            self.physical_model,
        )
        fallback_reason = _runtime_fallback_reason(
            context,
            self.config,
            allowed_structural_hashes=self.allowed_structural_hashes,
        )
        if fallback_reason is not None:
            return self._fallback(baseline, fallback_reason, context.morphology_graph.graph_id)
        if baseline.desired_body_pose is None:
            return self._fallback(baseline, "missing_centroidal_pose_target", context.morphology_graph.graph_id)
        target_twist = list(baseline.desired_body_twist or [0.0] * 6)
        self._reset_recurrent_if_needed(
            context.morphology_graph.graph_id,
            context.runtime_observation.time_s,
        )
        try:
            control_model = self.rigid_body_builder.build(
                context.morphology_graph,
                self.physical_model,
                context.runtime_observation,
            )
            features = order3_actor_feature_vector(
                context.runtime_observation,
                control_model,
                target_pose_world=baseline.desired_body_pose,
                target_twist=target_twist,
                max_modules=self.config.max_modules,
            )
        except (SchemaValidationError, TypeError, ValueError, ZeroDivisionError):
            return self._fallback(baseline, "feature_extraction_error", context.morphology_graph.graph_id)
        if (
            len(features) != len(ORDER3_ACTOR_FEATURE_NAMES)
            or not all(math.isfinite(value) for value in features)
        ):
            return self._fallback(baseline, "non_finite_actor_features", context.morphology_graph.graph_id)
        if any(abs(value) > self.config.ood_absolute_feature_limit for value in features):
            return self._fallback(baseline, "actor_feature_ood", context.morphology_graph.graph_id)

        privileged = list(privileged_disturbance_body or [0.0] * 6)
        if len(privileged) != 6 or not all(
            math.isfinite(float(value)) for value in privileged
        ):
            return self._fallback(
                baseline,
                "invalid_privileged_critic_input",
                context.morphology_graph.graph_id,
            )

        hidden_in = self._hidden.detach().clone()
        previous_action = [
            float(value) for value in self._previous_action[0].detach().cpu().tolist()
        ]
        try:
            with torch.no_grad():
                step = self.model.step(
                    [context.morphology_graph],
                    [context.runtime_observation],
                    torch.tensor([features], dtype=self._hidden.dtype, device=self.device),
                    self._previous_action,
                    self._hidden,
                    privileged_disturbance_body=torch.tensor(
                        [privileged], dtype=self._hidden.dtype, device=self.device
                    ),
                    deterministic=self.deterministic,
                    **_optional_phase_step_kwargs(
                        self.model,
                        context,
                        device=self.device,
                        dtype=self._hidden.dtype,
                    ),
                )
        except (RuntimeError, SchemaValidationError, TypeError, ValueError):
            return self._fallback(baseline, "model_inference_error", context.morphology_graph.graph_id)
        action = [float(value) for value in step.action[0].detach().cpu().tolist()]
        action_mean = [float(value) for value in step.action_mean[0].detach().cpu().tolist()]
        if len(action) != ORDER3_ACTION_SIZE or not all(math.isfinite(value) for value in action):
            return self._fallback(baseline, "invalid_model_action", context.morphology_graph.graph_id)
        try:
            command = self._decode_command(
                baseline,
                context.runtime_observation,
                control_model,
                action,
                step,
            )
        except (SchemaValidationError, TypeError, ValueError, IndexError):
            return self._fallback(baseline, "command_decode_error", context.morphology_graph.graph_id)
        self._hidden = step.recurrent_state.detach()
        self._previous_action = step.action.detach()
        self._last_graph_id = context.morphology_graph.graph_id
        self._last_time_s = context.runtime_observation.time_s
        self.last_diagnostics = Order3PolicyDiagnostics(
            learned_policy_applied=True,
            fallback_reason=None,
            graph_id=context.morphology_graph.graph_id,
            normalized_action=dict(zip(ORDER3_ACTION_NAMES, action)),
        )
        return Order3PolicyInference(
            command=command,
            previous_action=previous_action,
            normalized_action=action,
            action_mean=action_mean,
            log_prob=float(step.log_prob[0].detach().cpu().item()),
            value=float(step.value[0].detach().cpu().item()),
            recurrent_state_in=[float(value) for value in hidden_in[0].cpu().tolist()],
            recurrent_state_out=[
                float(value) for value in step.recurrent_state[0].detach().cpu().tolist()
            ],
            learned_policy_applied=True,
            fallback_reason=None,
            normalized_joint_action=(
                [
                    [float(value) for value in row]
                    for row in step.joint_action[0].detach().cpu().tolist()
                ]
                if hasattr(step, "joint_action")
                else []
            ),
            joint_action_mean=(
                [
                    [float(value) for value in row]
                    for row in step.joint_action_mean[0].detach().cpu().tolist()
                ]
                if hasattr(step, "joint_action_mean")
                else []
            ),
            module_ids=[
                int(value)
                for value in step.graph_encoding.module_ids[0]
                .detach()
                .cpu()
                .tolist()
                if int(value) >= 0
            ],
        )

    def _decode_command(
        self,
        baseline: PolicyCommand,
        observation: RuntimeObservation,
        control_model: RigidBodyControlModel,
        action: list[float],
        step: Order3ActorCriticStep,
    ) -> PolicyCommand:
        blend = self.config.trust_region_blend
        twist_limits = [
            *([self.config.linear_twist_correction_limit_mps] * 3),
            *([self.config.angular_twist_correction_limit_radps] * 3),
        ]
        base_twist = list(baseline.desired_body_twist or [0.0] * 6)
        desired_twist = [
            float(base_twist[index]) + blend * action[index] * twist_limits[index]
            for index in range(6)
        ]
        wrench_scales = [
            *([control_model.total_mass_kg * _GRAVITY_MPS2 * self.config.residual_force_weight_fraction] * 3),
            *([len(observation.morphology_graph.modules) * self.config.residual_torque_per_module_nm] * 3),
        ]
        base_wrench = list(baseline.residual_wrench_body or [0.0] * 6)
        residual_wrench = [
            float(base_wrench[index]) + blend * action[6 + index] * wrench_scales[index]
            for index in range(6)
        ]
        joint_positions, joint_velocities, joint_torque_bias = self._decode_joint_targets(
            observation,
            step,
            blend=blend,
        )
        return PolicyCommand(
            desired_body_pose=baseline.desired_body_pose,
            desired_body_twist=desired_twist,
            residual_wrench_body=residual_wrench,
            priority_weights=dict(baseline.priority_weights),
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=joint_positions,
            joint_velocity_targets=joint_velocities,
            joint_torque_bias=joint_torque_bias,
        )

    def _decode_joint_targets(
        self,
        observation: RuntimeObservation,
        step: Order3ActorCriticStep,
        *,
        blend: float,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        joint_ids = _dock_mechanism_joint_ids(self.physical_model)
        if len(joint_ids) > self.config.max_local_joint_slots:
            raise SchemaValidationError(
                "Order3 joint decoder has fewer slots than the physical model"
            )
        state_by_id = {state.module_id: state for state in observation.module_states}
        module_ids = step.graph_encoding.module_ids[0].detach().cpu().tolist()
        raw = step.joint_residuals[0].detach().cpu()
        enable = 1.0 if self.config.free_flight_joint_residual_enabled else 0.0
        q_targets: dict[str, float] = {}
        qdot_targets: dict[str, float] = {}
        torque_bias: dict[str, float] = {}
        joint_by_id = {joint.joint_id: joint for joint in self.physical_model.joints}
        for node_index, module_id_value in enumerate(module_ids):
            module_id = int(module_id_value)
            if module_id < 0:
                continue
            state = state_by_id[module_id]
            for slot, joint_id in enumerate(joint_ids):
                if joint_id not in state.joint_positions:
                    continue
                global_id = f"module_{module_id}:{joint_id}"
                current = float(state.joint_positions[joint_id])
                # Order-3 fixed-morphology free flight has no structural-joint
                # degree of freedom.  Holding the latest measured position
                # ratchets passive drift into the next command and lets the
                # visible dock frames separate even though module roots are
                # fixed.  Use the URDF neutral position whenever the joint
                # residual head is disabled; the articulated/dynamic assembly
                # paths use their own explicit posture targets.
                nominal_position = current if enable > 0.0 else 0.0
                q_delta = (
                    enable
                    * blend
                    * float(raw[node_index, slot].item())
                    * self.config.joint_position_delta_limit_rad
                )
                velocity = (
                    enable
                    * blend
                    * float(raw[node_index, self.config.max_local_joint_slots + slot].item())
                    * self.config.joint_velocity_limit_rad_s
                )
                effort_limit = float(joint_by_id[joint_id].effort_limit or 0.0)
                torque = (
                    enable
                    * blend
                    * float(raw[node_index, 2 * self.config.max_local_joint_slots + slot].item())
                    * effort_limit
                    * self.config.joint_torque_fraction
                )
                q_targets[global_id] = nominal_position + q_delta
                qdot_targets[global_id] = velocity
                torque_bias[global_id] = torque
        return q_targets, qdot_targets, torque_bias

    def _reset_recurrent_if_needed(self, graph_id: str, time_s: float) -> None:
        if (
            self._last_graph_id is not None
            and (
                graph_id != self._last_graph_id
                or (self._last_time_s is not None and time_s < self._last_time_s)
            )
        ):
            self.reset()

    def _fallback(
        self,
        command: PolicyCommand,
        reason: str,
        graph_id: str | None,
    ) -> Order3PolicyInference:
        self.last_diagnostics = Order3PolicyDiagnostics(
            learned_policy_applied=False,
            fallback_reason=reason,
            graph_id=graph_id,
        )
        hidden = [float(value) for value in self._hidden[0].detach().cpu().tolist()]
        previous_action = [
            float(value) for value in self._previous_action[0].detach().cpu().tolist()
        ]
        return Order3PolicyInference(
            command=command,
            previous_action=previous_action,
            normalized_action=[0.0] * ORDER3_ACTION_SIZE,
            action_mean=[0.0] * ORDER3_ACTION_SIZE,
            log_prob=0.0,
            value=0.0,
            recurrent_state_in=hidden,
            recurrent_state_out=hidden,
            learned_policy_applied=False,
            fallback_reason=reason,
        )


def order3_actor_feature_vector(
    observation: RuntimeObservation,
    control_model: RigidBodyControlModel,
    *,
    target_pose_world: Pose7D,
    target_twist: Sequence[float],
    max_modules: int = 8,
) -> list[float]:
    if len(target_pose_world) != 7 or len(target_twist) != 6:
        raise SchemaValidationError("Order3 centroidal target pose/twist shape mismatch")
    current_pose = control_model.body_pose_world
    current_twist = list(control_model.body_twist_world)
    body_from_world = transpose(quat_to_matrix(tuple(float(value) for value in current_pose[3:7])))
    current_angular_body = matvec(
        body_from_world,
        tuple(float(value) for value in current_twist[3:6]),
    )
    position_error = [
        float(target_pose_world[index]) - float(current_pose[index])
        for index in range(3)
    ]
    orientation_error = _orientation_error_body(current_pose, target_pose_world)
    target_values = [float(value) for value in target_twist]
    linear_velocity_error = [
        target_values[index] - float(current_twist[index])
        for index in range(3)
    ]
    angular_velocity_error = [
        target_values[3 + index] - float(current_angular_body[index])
        for index in range(3)
    ]
    status = observation.controller_status
    status_one_hot = [1.0 if status.status == name else 0.0 for name in ("ok", "warning", "infeasible", "fault")]
    residual = status.metrics.get(
        "allocation_residual_norm",
        status.metrics.get("residual_norm", 0.0),
    )
    phase = float(observation.time_s) * 0.2
    features = [
        math.sin(phase),
        math.cos(phase),
        len(observation.morphology_graph.modules) / float(max_modules),
        _signed_log1p(control_model.total_mass_kg),
        *[_signed_log1p(value) for value in control_model.inertia_body],
        *position_error,
        *orientation_error,
        *[float(value) for value in current_twist[:3]],
        *[float(value) for value in current_angular_body],
        *target_values,
        *linear_velocity_error,
        *angular_velocity_error,
        1.0 if status.qp_feasible else 0.0,
        *status_one_hot,
        _signed_log1p(float(residual)),
        float(observation.task_progress.progress_ratio),
        1.0 if observation.task_progress.success else 0.0,
    ]
    if len(features) != len(ORDER3_ACTOR_FEATURE_NAMES):
        raise RuntimeError("Order3 actor feature layout mismatch")
    if not all(math.isfinite(value) for value in features):
        raise SchemaValidationError("Order3 actor features must be finite")
    return features


@dataclass(frozen=True)
class LoadedOrder3PolicyCheckpoint:
    model: MorphologyConditionedActorCritic
    config: Order3MorphologyConditionedPolicyConfig
    metadata: Order3PolicyCheckpointMetadata
    path: str
    sha256: str


def save_order3_policy_checkpoint(
    path: str | Path,
    *,
    model: MorphologyConditionedActorCritic,
    metadata: Order3PolicyCheckpointMetadata,
) -> str:
    _validate_checkpoint_runtime_contract(model.config, metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": ORDER3_CHECKPOINT_VERSION,
        "metadata": metadata.to_dict(),
        "policy_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hash_file(destination)


def load_order3_policy_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_sha256: str | None = None,
) -> LoadedOrder3PolicyCheckpoint:
    source = Path(path)
    actual_sha256 = hash_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise SchemaValidationError("Order3 policy checkpoint sha256 mismatch")
    try:
        payload = torch.load(source, map_location=device, weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SchemaValidationError(f"failed to load Order3 policy checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError("Order3 policy checkpoint must be a mapping")
    if payload.get("checkpoint_version") != ORDER3_CHECKPOINT_VERSION:
        raise SchemaValidationError(
            "checkpoint is not the centroidal morphology-conditioned Order3 version"
        )
    required = {"checkpoint_version", "metadata", "policy_config", "state_dict"}
    if set(payload) != required:
        raise SchemaValidationError("Order3 policy checkpoint keys do not match the v1 contract")
    metadata = Order3PolicyCheckpointMetadata.from_dict(payload["metadata"])
    config = Order3MorphologyConditionedPolicyConfig.from_dict(payload["policy_config"])
    _validate_checkpoint_runtime_contract(config, metadata)
    model = MorphologyConditionedActorCritic(config).to(device)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SchemaValidationError(f"Order3 policy state_dict is incompatible: {exc}") from exc
    model.eval()
    return LoadedOrder3PolicyCheckpoint(
        model=model,
        config=config,
        metadata=metadata,
        path=str(source),
        sha256=actual_sha256,
    )


def order3_actor_feature_schema_hash() -> str:
    return stable_hash(
        {
            "names": ORDER3_ACTOR_FEATURE_NAMES,
            "previous_action_names": ORDER3_ACTION_NAMES,
            "privileged_wrench": False,
            "contact_state_features": False,
        }
    )


def order3_graph_feature_schema_hash() -> str:
    return stable_hash(
        {
            "tensorizer_version": MORPHOLOGY_GRAPH_TENSORIZER_VERSION,
            "encoder_version": MORPHOLOGY_GRAPH_ENCODER_VERSION,
            "node_features": MORPHOLOGY_NODE_FEATURE_NAMES,
            "edge_features": MORPHOLOGY_EDGE_FEATURE_NAMES,
        }
    )


def _validate_checkpoint_runtime_contract(
    config: Order3MorphologyConditionedPolicyConfig,
    metadata: Order3PolicyCheckpointMetadata,
) -> None:
    if metadata.config_hash != config.stable_hash():
        raise SchemaValidationError("Order3 checkpoint policy config hash mismatch")
    if metadata.actor_feature_schema_hash != order3_actor_feature_schema_hash():
        raise SchemaValidationError("Order3 checkpoint actor feature schema hash mismatch")
    if metadata.graph_feature_schema_hash != order3_graph_feature_schema_hash():
        raise SchemaValidationError("Order3 checkpoint graph feature schema hash mismatch")
    if metadata.tensorizer_version != MORPHOLOGY_GRAPH_TENSORIZER_VERSION:
        raise SchemaValidationError("Order3 checkpoint tensorizer implementation mismatch")
    if metadata.encoder_version != MORPHOLOGY_GRAPH_ENCODER_VERSION:
        raise SchemaValidationError("Order3 checkpoint graph encoder implementation mismatch")
    _checkpoint_morphology_allowlist(metadata)


def _strict_v2_fallback_command(
    command: PolicyCommand,
    observation: RuntimeObservation,
    physical_model: PhysicalModel,
) -> PolicyCommand:
    dock_joint_ids = _dock_mechanism_joint_ids(physical_model)
    q_targets: dict[str, float] = {}
    qdot_targets: dict[str, float] = {}
    torque_bias: dict[str, float] = {}
    for state in observation.module_states:
        for joint_id in dock_joint_ids:
            if joint_id not in state.joint_positions:
                continue
            global_id = f"module_{state.module_id}:{joint_id}"
            q_targets[global_id] = 0.0
            qdot_targets[global_id] = 0.0
            torque_bias[global_id] = 0.0
    return PolicyCommand(
        desired_body_pose=command.desired_body_pose,
        desired_body_twist=command.desired_body_twist,
        residual_wrench_body=command.residual_wrench_body,
        priority_weights=dict(command.priority_weights),
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        joint_position_targets=q_targets,
        joint_velocity_targets=qdot_targets,
        joint_torque_bias=torque_bias,
    )


def _runtime_fallback_reason(
    context: LowLevelPolicyContext,
    config: Order3MorphologyConditionedPolicyConfig,
    *,
    allowed_structural_hashes: frozenset[str] | set[str] | None = None,
) -> str | None:
    observation = context.runtime_observation
    status = context.controller_status or observation.controller_status
    if status.status in {"infeasible", "fault"} or not status.qp_feasible:
        return "controller_infeasible"
    module_count = len(context.morphology_graph.modules)
    if not 2 <= module_count <= config.max_modules:
        return "module_count_ood"
    graph_ids = {module.module_id for module in context.morphology_graph.modules}
    state_ids = [state.module_id for state in observation.module_states]
    if len(state_ids) != len(set(state_ids)) or set(state_ids) != graph_ids:
        return "runtime_module_identity_mismatch"
    if observation.morphology_graph.graph_id != context.morphology_graph.graph_id:
        return "runtime_graph_identity_mismatch"
    if (
        allowed_structural_hashes is not None
        and morphology_structural_hash(context.morphology_graph)
        not in allowed_structural_hashes
    ):
        return "structural_hash_ood"
    return None


def _checkpoint_morphology_allowlist(
    metadata: Order3PolicyCheckpointMetadata,
) -> set[str]:
    raw = metadata.metadata.get("morphology_hashes")
    if not isinstance(raw, dict) or not raw:
        raise SchemaValidationError(
            "Order3 checkpoint metadata requires morphology_hashes by split"
        )
    allowed: set[str] = set()
    for split, values in raw.items():
        if not isinstance(split, str) or not isinstance(values, list):
            raise SchemaValidationError(
                "Order3 checkpoint morphology_hashes must map split names to lists"
            )
        for value in values:
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaValidationError(
                    "Order3 checkpoint morphology allowlist contains an invalid hash"
                )
            if value in allowed:
                raise SchemaValidationError(
                    "Order3 checkpoint morphology allowlist contains duplicate hashes"
                )
            allowed.add(value)
    if not allowed:
        raise SchemaValidationError(
            "Order3 checkpoint morphology allowlist must not be empty"
        )
    return allowed


def _dock_mechanism_joint_ids(physical_model: PhysicalModel) -> list[str]:
    return sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
            if port.mechanical_limits.get("mechanism_joint_id")
        }
    )


def _optional_phase_step_kwargs(
    model: nn.Module,
    context: LowLevelPolicyContext,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Supply the additive Order-9 phase contract without changing Order-3 calls."""

    builder = getattr(model, "phase_feature_tensor", None)
    if builder is None:
        return {}
    return {
        "phase_features": builder(context, device=device, dtype=dtype),
    }


def _orientation_error_body(current_pose: Pose7D, target_pose: Pose7D) -> list[float]:
    current_rotation = quat_to_matrix(tuple(float(value) for value in current_pose[3:7]))
    target_rotation = quat_to_matrix(tuple(float(value) for value in target_pose[3:7]))
    current_from_target = _matmul3(transpose(current_rotation), target_rotation)
    trace = sum(current_from_target[index][index] for index in range(3))
    angle = math.acos(max(-1.0, min(1.0, 0.5 * (trace - 1.0))))
    if angle <= 1.0e-8:
        return [0.0, 0.0, 0.0]
    scale = angle / max(2.0 * math.sin(angle), 1.0e-8)
    return [
        scale * (current_from_target[2][1] - current_from_target[1][2]),
        scale * (current_from_target[0][2] - current_from_target[2][0]),
        scale * (current_from_target[1][0] - current_from_target[0][1]),
    ]


def _matmul3(left, right):
    return tuple(
        tuple(
            sum(float(left[row][index]) * float(right[index][column]) for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _signed_log1p(value: float) -> float:
    numeric = float(value)
    return math.copysign(math.log1p(abs(numeric)), numeric)


def _require_tensor_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"Order3 {name} shape must be {shape}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"Order3 {name} must be finite")


def _runtime_tensor(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(
            f"morphology policy runtime {name} is not numeric"
        ) from exc
    if tuple(tensor.shape) != expected_shape:
        raise SchemaValidationError(
            f"morphology policy runtime {name} has shape {tuple(tensor.shape)}, "
            f"expected {expected_shape}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise SchemaValidationError(
            f"morphology policy runtime {name} must be finite"
        )
    return tensor.clone()


__all__ = [
    "ORDER3_ACTOR_FEATURE_NAMES",
    "ORDER3_POLICY_OUTPUT_MODE",
    "MORPHOLOGY_POLICY_RUNTIME_STATE_VERSION",
    "LoadedOrder3PolicyCheckpoint",
    "MorphologyConditionedActorCritic",
    "MorphologyConditionedLowLevelPolicy",
    "Order3ActorCriticStep",
    "Order3MorphologyConditionedPolicyConfig",
    "Order3PolicyDiagnostics",
    "Order3PolicyInference",
    "load_order3_policy_checkpoint",
    "order3_actor_feature_schema_hash",
    "order3_actor_feature_vector",
    "order3_graph_feature_schema_hash",
    "save_order3_policy_checkpoint",
]
