from __future__ import annotations

"""Validated Order 9 curriculum and promotion/throughput gates."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from amsrr.schemas.common import SchemaBase, SchemaValidationError, StrEnum, require_non_empty
from amsrr.schemas.policies import CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
from amsrr.training.p4_3_reward import P4_3RewardConfig
from amsrr.training.order9_c3_boundary_sampling import (
    order9_c3_fixed_reset_stratum_index,
    order9_c3_reset_strata_by_phase,
)
from amsrr.training.order9_randomization import (
    Order9ConservativeRandomizationConfig,
    Order9ExpandedObjectRandomizationConfig,
)
from amsrr.utils.config import load_config


ORDER9_CURRICULUM_VERSION = "order9_curriculum_v3_progressive_object_conditions"
ORDER9_LEGACY_CURRICULUM_VERSION = "order9_curriculum_v2"
ORDER9_C0_COLLECTION_PROFILE_VERSION = "order9_c0_bounded_diversity_v1"


class Order9LearningMode(StrEnum):
    COLLECTION = "collection"
    BEHAVIOR_CLONING = "behavior_cloning"
    PPO = "ppo"
    EVALUATION = "evaluation"


class Order9LearningTarget(StrEnum):
    DATASET = "dataset"
    PI_L = "pi_l"
    PI_H_ASSIGNMENT = "pi_h_assignment"
    PI_H_TRAJECTORY = "pi_h_trajectory"
    PI_D = "pi_d"
    JOINT_OBJECT_TASK = "joint_object_task"
    FULL_SYSTEM = "full_system"


class PiHOutputScope(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ASSIGNMENT_ONLY_WARMUP = "assignment_only_warmup"
    FULL_CONTACT_WRENCH_TRAJECTORY = "full_contact_wrench_trajectory"


class ObjectDistributionLevel(StrEnum):
    NOMINAL = "nominal"
    CONSERVATIVE_ORDER8_ANCHOR = "conservative_order8_anchor"
    REACHABLE_POSE_EXPANSION = "reachable_pose_expansion"
    BOX_PROPERTY_EXPANSION = "box_property_expansion"
    EXPANDED_PRIMITIVES = "expanded_primitives"
    EXPANDED_CROSS_PRODUCT = "expanded_cross_product"
    HELD_OUT_SHAPES_AND_INERTIA = "held_out_shapes_and_inertia"


class AssemblyEvaluationMode(StrEnum):
    SEPARATE = "separate"
    SUBSET_END_TO_END = "subset_end_to_end"


@dataclass
class Order9TeacherCollectionRuntimeConfig(SchemaBase):
    """High-fidelity C0 collection budget, split, and execution profile."""

    profile_version: str = ORDER9_C0_COLLECTION_PROFILE_VERSION
    episode_count: int = 20
    validation_episode_count: int = 3
    held_out_episode_count: int = 3
    low_level_stride: int = 5
    high_level_stride: int = 5
    parallel_process_count: int = 2
    condition_distribution: ObjectDistributionLevel = (
        ObjectDistributionLevel.CONSERVATIVE_ORDER8_ANCHOR
    )

    def validate(self) -> None:
        if self.profile_version != ORDER9_C0_COLLECTION_PROFILE_VERSION:
            raise SchemaValidationError(
                "Order9 C0 teacher collection profile version mismatch"
            )
        if self.episode_count < 5:
            raise SchemaValidationError(
                "Order9 C0 requires nominal plus four conservative boundary episodes"
            )
        if min(self.validation_episode_count, self.held_out_episode_count) < 1:
            raise SchemaValidationError(
                "Order9 C0 validation and held-out episode counts must be positive"
            )
        if (
            self.validation_episode_count + self.held_out_episode_count
            >= self.episode_count
        ):
            raise SchemaValidationError(
                "Order9 C0 split must leave at least one training episode"
            )
        if min(
            self.low_level_stride,
            self.high_level_stride,
            self.parallel_process_count,
        ) < 1:
            raise SchemaValidationError(
                "Order9 C0 strides and parallel process count must be positive"
            )
        if (
            self.condition_distribution
            != ObjectDistributionLevel.CONSERVATIVE_ORDER8_ANCHOR
        ):
            raise SchemaValidationError(
                "Order9 C0 conditions must remain in the conservative Order8 anchor"
            )


@dataclass
class Order9CurriculumStage(SchemaBase):
    stage_id: str
    stage_index: int
    learning_mode: Order9LearningMode
    learning_target: Order9LearningTarget
    min_modules: int
    max_modules: int
    topology_randomized: bool
    object_distribution: ObjectDistributionLevel
    deterministic_teacher_required: bool
    pi_h_output_scope: PiHOutputScope
    phase_conditioned_actor_required: bool
    design_action_mask_required: bool
    assembly_evaluation_mode: AssemblyEvaluationMode
    end_to_end_episode_fraction: float
    environment_steps: int
    minimum_episodes: int
    minimum_success_rate: float
    minimum_no_fallback_success_rate: float
    maximum_fallback_rate: float
    maximum_safety_failure_episodes: int
    held_out_only: bool
    task_adapter_ids: list[str] = field(default_factory=lambda: ["object_grasp_carry_v1"])
    parallel_environment_count: int | None = None
    rollout_steps_per_environment: int | None = None

    def validate(self) -> None:
        require_non_empty(self.stage_id, "Order9CurriculumStage.stage_id")
        if self.stage_index < 0:
            raise SchemaValidationError("Order9 stage_index must be non-negative")
        if not 1 <= self.min_modules <= self.max_modules <= 8:
            raise SchemaValidationError("Order9 module range must stay within [1, 8]")
        if self.environment_steps < 0 or self.minimum_episodes < 1:
            raise SchemaValidationError(
                "Order9 environment_steps must be non-negative and minimum_episodes positive"
            )
        for name in (
            "end_to_end_episode_fraction",
            "minimum_success_rate",
            "minimum_no_fallback_success_rate",
            "maximum_fallback_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(f"Order9CurriculumStage.{name} must be in [0, 1]")
        if self.maximum_safety_failure_episodes < 0:
            raise SchemaValidationError(
                "Order9 maximum_safety_failure_episodes must be non-negative"
            )
        rollout_override_values = (
            self.parallel_environment_count,
            self.rollout_steps_per_environment,
        )
        if any(value is not None for value in rollout_override_values):
            if self.learning_mode != Order9LearningMode.PPO:
                raise SchemaValidationError(
                    "Order9 stage rollout overrides are restricted to PPO stages"
                )
            if any(value is None or value < 1 for value in rollout_override_values):
                raise SchemaValidationError(
                    "Order9 PPO stage rollout overrides must specify positive "
                    "parallel_environment_count and rollout_steps_per_environment"
                )
        if not self.task_adapter_ids or len(self.task_adapter_ids) != len(set(self.task_adapter_ids)):
            raise SchemaValidationError(
                "Order9 task_adapter_ids must be non-empty and unique"
            )
        if self.assembly_evaluation_mode == AssemblyEvaluationMode.SEPARATE:
            if self.end_to_end_episode_fraction != 0.0:
                raise SchemaValidationError(
                    "separate assembly evaluation requires zero end-to-end episode fraction"
                )
        elif self.end_to_end_episode_fraction <= 0.0:
            raise SchemaValidationError(
                "subset end-to-end assembly evaluation requires a positive episode fraction"
            )
        if self.learning_mode == Order9LearningMode.BEHAVIOR_CLONING:
            if not self.deterministic_teacher_required:
                raise SchemaValidationError("Order9 BC stages require a deterministic teacher")
            if self.environment_steps != 0:
                raise SchemaValidationError("Order9 BC stages use records, not environment_steps")
        if self.learning_mode == Order9LearningMode.PPO and self.environment_steps <= 0:
            raise SchemaValidationError("Order9 PPO stages require a positive environment_steps budget")
        if self.learning_mode == Order9LearningMode.EVALUATION:
            if not self.held_out_only or self.environment_steps != 0:
                raise SchemaValidationError(
                    "Order9 evaluation must be held-out-only with no training steps"
                )
        if self.pi_h_output_scope == PiHOutputScope.ASSIGNMENT_ONLY_WARMUP:
            if self.learning_target != Order9LearningTarget.PI_H_ASSIGNMENT:
                raise SchemaValidationError(
                    "assignment-only output is restricted to the pi_H warm-up target"
                )
        if self.learning_target in {
            Order9LearningTarget.PI_H_TRAJECTORY,
            Order9LearningTarget.JOINT_OBJECT_TASK,
            Order9LearningTarget.FULL_SYSTEM,
        } and self.pi_h_output_scope != PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY:
            raise SchemaValidationError(
                "trajectory/joint/full-system stages require full pi_H trajectory output"
            )
        if self.learning_target == Order9LearningTarget.PI_D and not self.design_action_mask_required:
            raise SchemaValidationError("pi_D learning requires deterministic action masking")


@dataclass
class Order9CurriculumSchedule(SchemaBase):
    schedule_version: str = ORDER9_CURRICULUM_VERSION
    contact_wrench_contract_version: str = CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
    stages: list[Order9CurriculumStage] = field(default_factory=list)
    pi_h_is_learned_proposal_only: bool = True
    deterministic_checker_projects_actions: bool = False
    dynamic_assembly_policy_learned: bool = False

    def validate(self) -> None:
        require_non_empty(self.schedule_version, "Order9CurriculumSchedule.schedule_version")
        if self.schedule_version not in {
            ORDER9_LEGACY_CURRICULUM_VERSION,
            ORDER9_CURRICULUM_VERSION,
        }:
            raise SchemaValidationError(
                "Order9 curriculum schedule version is unsupported"
            )
        if self.contact_wrench_contract_version != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME:
            raise SchemaValidationError("Order9 requires the v2 contact-frame wrench contract")
        if not self.pi_h_is_learned_proposal_only:
            raise SchemaValidationError("Order9 pi_H must denote only the learned proposal policy")
        if self.deterministic_checker_projects_actions:
            raise SchemaValidationError("Order9 C_H must accept/reject and never project pi_H actions")
        if self.dynamic_assembly_policy_learned:
            raise SchemaValidationError("P4-full dynamic assembly remains deterministic")
        if not self.stages:
            raise SchemaValidationError("Order9 curriculum must contain stages")
        if [stage.stage_index for stage in self.stages] != list(range(len(self.stages))):
            raise SchemaValidationError("Order9 stage indices must be contiguous and ordered")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise SchemaValidationError("Order9 stage ids must be unique")
        if self.stages[0].learning_mode != Order9LearningMode.COLLECTION:
            raise SchemaValidationError("Order9 curriculum must begin with teacher collection")
        if self.stages[-1].learning_mode != Order9LearningMode.EVALUATION:
            raise SchemaValidationError("Order9 curriculum must end with held-out evaluation")
        _require_bc_before_ppo(self.stages, Order9LearningTarget.PI_L)
        _require_bc_before_ppo(self.stages, Order9LearningTarget.PI_H_TRAJECTORY)
        _require_bc_before_ppo(self.stages, Order9LearningTarget.PI_D)
        if not any(
            stage.learning_target == Order9LearningTarget.PI_D
            and stage.learning_mode == Order9LearningMode.PPO
            for stage in self.stages
        ):
            raise SchemaValidationError("Order9 must complete pi_D masked PPO")
        if not any(
            stage.pi_h_output_scope == PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY
            and stage.learning_mode == Order9LearningMode.PPO
            for stage in self.stages
        ):
            raise SchemaValidationError("Order9 must train full-trajectory pi_H with PPO")
        arbitrary_morphology = [
            stage
            for stage in self.stages
            if stage.topology_randomized and stage.min_modules == 2 and stage.max_modules == 8
        ]
        if not arbitrary_morphology:
            raise SchemaValidationError("Order9 must include 2--8 module topology-randomized training")
        if self.schedule_version == ORDER9_CURRICULUM_VERSION:
            _validate_progressive_object_condition_schedule(self.stages)


@dataclass
class Order9RuntimeBenchmarkConfig(SchemaBase):
    environment_count_candidates: list[int] = field(default_factory=lambda: [32, 64, 128])
    initial_environment_count: int = 64
    control_dt_s: float = 0.02
    minimum_aggregate_env_steps_per_s: float = 500.0
    warmup_steps: int = 256
    measurement_steps: int = 2048
    maximum_wall_time_s: float = 900.0
    topology_bucketed: bool = True
    phase_specific_resets: bool = True
    per_step_json_logging: bool = False
    require_tensorized_pi_l_inference: bool = True

    def validate(self) -> None:
        if (
            not self.environment_count_candidates
            or any(value < 1 for value in self.environment_count_candidates)
            or len(self.environment_count_candidates)
            != len(set(self.environment_count_candidates))
        ):
            raise SchemaValidationError(
                "Order9 benchmark environment counts must be positive and unique"
            )
        if self.initial_environment_count not in self.environment_count_candidates:
            raise SchemaValidationError(
                "Order9 initial_environment_count must be benchmarked"
            )
        for name in (
            "control_dt_s",
            "minimum_aggregate_env_steps_per_s",
            "maximum_wall_time_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(f"Order9RuntimeBenchmarkConfig.{name} must be positive")
        if self.warmup_steps < 0 or self.measurement_steps < 1:
            raise SchemaValidationError(
                "Order9 benchmark warmup must be non-negative and measurement positive"
            )
        if not self.topology_bucketed or not self.phase_specific_resets:
            raise SchemaValidationError(
                "Order9 production throughput requires topology buckets and phase-specific resets"
            )
        if self.per_step_json_logging:
            raise SchemaValidationError("Order9 training must not emit per-step JSON")
        if not self.require_tensorized_pi_l_inference:
            raise SchemaValidationError(
                "Order9 production throughput must include tensorized pi_L inference"
            )


@dataclass
class Order9BCOptimizationConfig(SchemaBase):
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    value_loss_weight: float = 0.5
    max_grad_norm: float = 0.5
    sequence_length: int = 16
    burn_in_steps: int = 4
    phase_balanced_sampling: bool = False

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.sequence_length < 1:
            raise SchemaValidationError("Order9 BC epochs/batch_size must be positive")
        if self.burn_in_steps < 0 or self.burn_in_steps >= self.sequence_length:
            raise SchemaValidationError(
                "Order9 BC burn_in_steps must lie in [0, sequence_length)"
            )
        for name in ("learning_rate", "max_grad_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(f"Order9 BC {name} must be positive")
        if not math.isfinite(self.value_loss_weight) or self.value_loss_weight < 0.0:
            raise SchemaValidationError(
                "Order9 BC value_loss_weight must be finite and non-negative"
            )


@dataclass
class Order9PPOOptimizationConfig(SchemaBase):
    rollout_steps_per_environment: int = 256
    epochs_per_update: int = 4
    minibatch_size: int = 4096
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_loss_weight: float = 0.5
    entropy_bonus_weight: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    hard_checker_rejection_penalty: float = 1.0
    phase_balanced_sampling: bool = False
    phase_normalized_advantages: bool = False
    phase_local_kl: bool = False

    def validate(self) -> None:
        for name in (
            "rollout_steps_per_environment",
            "epochs_per_update",
            "minibatch_size",
        ):
            if int(getattr(self, name)) < 1:
                raise SchemaValidationError(f"Order9 PPO {name} must be positive")
        for name in (
            "learning_rate",
            "max_grad_norm",
            "target_kl",
            "hard_checker_rejection_penalty",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(f"Order9 PPO {name} must be positive")
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(f"Order9 PPO {name} must lie in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise SchemaValidationError("Order9 PPO clip_ratio must lie in (0, 1)")
        for name in ("value_loss_weight", "entropy_bonus_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order9 PPO {name} must be finite and non-negative"
                )


@dataclass
class Order9C3BoundaryFineTuneConfig(SchemaBase):
    enabled: bool = False
    target_actor_phase_labels: list[str] = field(
        default_factory=lambda: ["release", "retreat", "settle"]
    )
    epochs_per_update: int = 1
    learning_rate_scale: float = 0.25
    non_target_parent_kl_limit: float = 0.01
    non_target_parent_kl_weight: float = 1.0
    maximum_topology_phase_kl: float = 0.04
    # Training-only proportional teacher for the deployable compression
    # scalar.  Exact simulator wrench is used only to construct the loss
    # target; it never enters actor observation or the deployed command path.
    privileged_compression_teacher_weight: float = 0.0
    privileged_compression_teacher_action_step: float = 0.10
    privileged_compression_teacher_underforce_only: bool = True
    privileged_wrench_satisfied_parent_kl_weight: float = 0.0
    topology_gradient_surgery_enabled: bool = False
    # Assign the contact/lift actor advantage only to the morphology-specific
    # compression scalar, instead of every PolicyCommand coordinate.
    compression_only_actor_objective: bool = False
    # Freeze the global/shared actor and train only the node-wise joint decoder
    # (plus the detached critic) during the boundary update.
    joint_head_only_actor_update: bool = False
    # Freeze the complete inherited actor and train only the exactly
    # phase-gated contact residual decoder (plus the detached critic).
    contact_residual_only_actor_update: bool = False
    # Training-initializer-only exploration width for the contact-space
    # inward-normal coordinate.  It changes stochastic rollout collection but
    # not the deterministic actor mean, deployed command, or action bound.
    contact_normal_exploration_initial_std: float | None = None
    # Route the existing scalar reward through the contact, centroidal, and
    # posture action densities according to their physical responsibilities.
    # The critic continues to fit the unchanged total return.
    factorized_actor_credit_enabled: bool = False
    # After the ordinary all-head PPO epoch, replay the same on-policy data
    # through the contact density only.  During these extra passes only the
    # contact-specific feature/slot/mean/std parameters may change; the shared
    # trunk, centroidal head, posture head, and critic remain fixed.
    contact_head_extra_optimizer_passes: int = 0
    # Refine contact credit in the ordinary PPO pass into normal-translation,
    # tangential-translation, and rotation likelihoods.  The three reward
    # channels reconstruct the existing contact reward exactly; this is
    # training-only credit routing, not a new deployed action or reward.
    contact_coordinate_credit_enabled: bool = False
    # Relative multiplier for the inward-normal coordinate in the ordinary
    # contact-coordinate PPO objective.  A value of one preserves the equal
    # normal/tangential/rotational weighting used by the original contract.
    contact_normal_translation_actor_loss_weight: float = 1.0
    # Training-only negative-log-likelihood weight that maps an authored
    # nominal-preload deficit to the corresponding categorical inward-normal
    # residual.  The deficit is never added to deployed observations.
    contact_normal_preload_deficit_teacher_weight: float = 0.0
    # Optional physical quantization of the deployed inward-normal contact
    # residual.  PPO retains its exact continuous latent action likelihood;
    # only the deterministic environment command is quantized.
    contact_normal_action_quantization_step_m: float | None = None

    def validate(self) -> None:
        labels = [str(value) for value in self.target_actor_phase_labels]
        if (
            not labels
            or len(set(labels)) != len(labels)
            or any(not value for value in labels)
        ):
            raise SchemaValidationError(
                "Order9 C3 boundary fine-tune phase labels must be unique"
            )
        if self.epochs_per_update != 1:
            raise SchemaValidationError(
                "Order9 C3 boundary fine-tune is restricted to one epoch"
            )
        for name in (
            "learning_rate_scale",
            "non_target_parent_kl_limit",
            "non_target_parent_kl_weight",
            "maximum_topology_phase_kl",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9 C3 boundary fine-tune {name} must be positive"
                )
        teacher_weight = float(
            self.privileged_compression_teacher_weight
        )
        if not math.isfinite(teacher_weight) or teacher_weight < 0.0:
            raise SchemaValidationError(
                "Order9 C3 privileged compression teacher weight must be "
                "finite and non-negative"
            )
        teacher_step = float(
            self.privileged_compression_teacher_action_step
        )
        if not math.isfinite(teacher_step) or not 0.0 < teacher_step <= 1.0:
            raise SchemaValidationError(
                "Order9 C3 privileged compression teacher action step must "
                "lie in (0, 1]"
            )
        satisfied_kl_weight = float(
            self.privileged_wrench_satisfied_parent_kl_weight
        )
        if not math.isfinite(satisfied_kl_weight) or satisfied_kl_weight < 0.0:
            raise SchemaValidationError(
                "Order9 C3 privileged wrench-satisfied parent KL weight "
                "must be finite and non-negative"
            )
        if self.learning_rate_scale > 1.0:
            raise SchemaValidationError(
                "Order9 C3 boundary fine-tune learning-rate scale must not exceed one"
            )
        if self.non_target_parent_kl_limit >= self.maximum_topology_phase_kl:
            raise SchemaValidationError(
                "Order9 C3 non-target KL limit must be below the topology-phase cap"
            )
        if (
            self.joint_head_only_actor_update
            and not self.compression_only_actor_objective
        ):
            raise SchemaValidationError(
                "Order9 C3 joint-head-only update requires the compression-only "
                "actor objective"
            )
        if (
            self.contact_residual_only_actor_update
            and not self.compression_only_actor_objective
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-residual-only update requires the "
                "compression-only actor objective"
            )
        if (
            self.joint_head_only_actor_update
            and self.contact_residual_only_actor_update
        ):
            raise SchemaValidationError(
                "Order9 C3 joint-head-only and contact-residual-only updates "
                "are mutually exclusive"
            )
        if self.factorized_actor_credit_enabled and (
            self.compression_only_actor_objective
            or self.joint_head_only_actor_update
            or self.contact_residual_only_actor_update
        ):
            raise SchemaValidationError(
                "Order9 C3 factorized actor credit is incompatible with legacy "
                "single-head actor objectives"
            )
        if (
            isinstance(self.contact_head_extra_optimizer_passes, bool)
            or not isinstance(self.contact_head_extra_optimizer_passes, int)
            or not 0 <= self.contact_head_extra_optimizer_passes <= 4
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-head extra optimizer passes must lie in "
                "[0, 4]"
            )
        if (
            self.contact_head_extra_optimizer_passes > 0
            and not self.factorized_actor_credit_enabled
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-head extra optimizer passes require "
                "factorized actor credit"
            )
        if self.contact_coordinate_credit_enabled and not self.factorized_actor_credit_enabled:
            raise SchemaValidationError(
                "Order9 C3 contact-coordinate credit requires factorized actor credit"
            )
        normal_actor_weight = float(
            self.contact_normal_translation_actor_loss_weight
        )
        if not math.isfinite(normal_actor_weight) or normal_actor_weight <= 0.0:
            raise SchemaValidationError(
                "Order9 C3 contact normal-translation actor-loss weight must "
                "be finite and positive"
            )
        preload_teacher_weight = float(
            self.contact_normal_preload_deficit_teacher_weight
        )
        if (
            not math.isfinite(preload_teacher_weight)
            or preload_teacher_weight < 0.0
        ):
            raise SchemaValidationError(
                "Order9 C3 contact normal preload-deficit teacher weight must "
                "be finite and non-negative"
            )
        if (
            preload_teacher_weight > 0.0
            and not self.contact_coordinate_credit_enabled
        ):
            raise SchemaValidationError(
                "Order9 C3 contact normal preload-deficit teacher requires "
                "contact-coordinate credit"
            )
        if self.contact_normal_exploration_initial_std is not None:
            exploration_std = float(
                self.contact_normal_exploration_initial_std
            )
            if (
                not math.isfinite(exploration_std)
                or not 0.0 < exploration_std <= 1.0
            ):
                raise SchemaValidationError(
                    "Order9 C3 contact-normal exploration initial std must "
                    "lie in (0, 1]"
                )
        if self.contact_normal_action_quantization_step_m is not None:
            quantization_step = float(
                self.contact_normal_action_quantization_step_m
            )
            if (
                not math.isfinite(quantization_step)
                or not 0.0 < quantization_step <= 0.020
            ):
                raise SchemaValidationError(
                    "Order9 C3 contact-normal action quantization step must "
                    "lie in (0, 0.020] m"
                )


@dataclass
class Order9OptimizationConfig(SchemaBase):
    pi_l_bc: Order9BCOptimizationConfig = field(
        default_factory=Order9BCOptimizationConfig
    )
    pi_h_assignment_bc: Order9BCOptimizationConfig = field(
        default_factory=lambda: Order9BCOptimizationConfig(batch_size=32)
    )
    pi_h_full_bc: Order9BCOptimizationConfig = field(
        default_factory=lambda: Order9BCOptimizationConfig(batch_size=32)
    )
    pi_d_bc: Order9BCOptimizationConfig = field(
        default_factory=lambda: Order9BCOptimizationConfig(batch_size=16)
    )
    pi_l_ppo: Order9PPOOptimizationConfig = field(
        default_factory=Order9PPOOptimizationConfig
    )
    c3_boundary_fine_tune: Order9C3BoundaryFineTuneConfig = field(
        default_factory=Order9C3BoundaryFineTuneConfig
    )
    pi_h_ppo: Order9PPOOptimizationConfig = field(
        default_factory=lambda: Order9PPOOptimizationConfig(
            rollout_steps_per_environment=64,
            minibatch_size=512,
            entropy_bonus_weight=0.005,
        )
    )
    pi_d_ppo: Order9PPOOptimizationConfig = field(
        default_factory=lambda: Order9PPOOptimizationConfig(
            rollout_steps_per_environment=16,
            minibatch_size=256,
            entropy_bonus_weight=0.01,
        )
    )
    joint_ppo: Order9PPOOptimizationConfig = field(
        default_factory=lambda: Order9PPOOptimizationConfig(
            learning_rate=5.0e-5,
            entropy_bonus_weight=0.001,
            target_kl=0.01,
        )
    )


@dataclass
class Order9ProductionRuntimeConfig(SchemaBase):
    seed: int = 9009
    device: str = "cuda:0"
    artifact_root: str = "artifacts/p4_full/order9"
    selected_environment_count: int = 128
    runtime_load_sample_interval_s: float = 1.0
    runtime_benchmark_report_path: str = (
        "artifacts/p4_full/order9/runtime_benchmark.json"
    )
    runtime_benchmark_report_sha256: str = ""
    checkpoint_interval_updates: int = 10
    metrics_flush_interval_updates: int = 1
    full_mesh_evaluation_interval_updates: int = 25
    full_mesh_evaluation_episode_count: int = 8
    canonical_order8_report_path: str = (
        "artifacts/p4_full/order8_natural_contact/"
        "order8_mu4p5_dt20ms_full_v406.json"
    )
    canonical_order8_report_sha256: str = (
        "d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49"
    )
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    tensorized_rollout_hot_path: bool = True
    raw_contact_actor_input: bool = False
    full_mesh_acceptance_replaced: bool = False
    c3_phase_reset_progress_fractions: list[float] = field(
        default_factory=lambda: [
            1.0 / 6.0,
            1.0 / 2.0,
            2.0 / 3.0,
            9.0 / 10.0,
        ]
    )
    c3_boundary_tail_phase_labels: list[str] = field(
        default_factory=lambda: ["place", "release", "retreat"]
    )
    c3_boundary_tail_progress_fraction: float = 0.9
    c3_contact_reset_preload_steps: int = 13
    c3_contact_reset_min_selected_contacts: int = 2
    c3_contact_reset_max_object_displacement_m: float = 0.03
    c3_contact_reset_max_downward_speed_mps: float = 0.25
    # Execution-side servo lead.  pi_H contact poses remain on the physical
    # object surface; the IK resolver alone offsets its bounded joint target.
    c3_virtual_contact_inward_lead_m: float = 0.0
    # Replace the fixed lead by a common actuator/compliance calculation at
    # nominal-plan installation time.  The calculation uses only the reviewed
    # contact geometry, object estimate, friction, actuator provenance, and
    # authored contact stiffness; it is deployable and module-count agnostic.
    c3_actuator_aware_nominal_preload_enabled: bool = False
    c3_actuator_aware_nominal_preload_maximum_m: float = 0.040
    c3_actuator_aware_nominal_preload_quantization_m: float = 0.001
    c3_actuator_aware_nominal_preload_support_safety_factor: float = 1.25
    c3_actuator_aware_nominal_preload_maximum_peak_utilization: float = 0.80
    c3_actuator_aware_nominal_preload_minimum_anchor_fraction: float = 0.50
    # Common additive uncertainty margin calibrated from train-split
    # late-contact rollout geometry.  This covers the systematic difference
    # between the linear compliance model and the realized articulated/mesh
    # contact geometry; it must never be fitted on held-out validation data.
    c3_actuator_aware_nominal_preload_model_error_margin_m: float = 0.0
    # One existing normalized joint-position action is repurposed during
    # attach/maintain as a morphology-conditioned scalar along a local
    # task-space compression IK direction.  This changes no actor input and
    # consumes no contact-force measurement.
    c3_contact_compression_action_adapter_enabled: bool = False
    c3_contact_compression_action_span_m: float = 0.0
    # Newly introduced morphology sizes may initially execute only the
    # morphology-conditioned contact-compression scalar on top of the
    # accepted nominal IK trajectory.  Other pi_L action coordinates remain
    # sampled for exact on-policy replay but are not applied to QPID.
    c3_contact_compression_only_module_counts: list[int] = field(
        default_factory=list
    )
    # A morphology may graduate from scalar-only entry to the complete joint
    # residual head while retaining the conservative nominal centroidal
    # command.  This masks pose/twist/residual-wrench corrections but applies
    # all bounded joint outputs plus the compression adapter.
    c3_joint_only_module_counts: list[int] = field(default_factory=list)
    c3_force_estimator_baseline_update_alpha: float = 0.10
    c3_force_estimator_filter_alpha: float = 0.25
    c3_force_estimator_ridge_damping_m2: float = 1.0e-5
    c3_force_estimator_minimum_jacobian_norm_m: float = 1.0e-3
    c3_force_estimator_fit_residual_scale_nm: float = 0.25
    c3_force_estimator_minimum_confidence: float = 0.25
    c3_force_estimator_support_safety_factor: float = 1.0
    c3_wrench_gate_curriculum_start_update_index: int = 4
    c3_wrench_gate_curriculum_scales: list[float] = field(
        default_factory=lambda: [4.0, 3.0, 2.0, 1.5, 1.25, 1.0]
    )
    c3_topology_stratified_updates: bool = True
    c3_topologies_per_module_count_per_update: int = 1
    c3_topology_shard_parallel_process_count: int = 2
    c3_state_inheritance_rollouts_enabled: bool = False
    c3_state_inheritance_rollout_steps: int = 1280
    c3_state_inheritance_environment_count_per_module: int = 12
    # Number of topology-homogeneous continuous shards collected for each
    # module-count stratum.  A focused stratum may temporarily receive wider
    # topology coverage during incremental morphology expansion; this changes
    # only the training distribution, never the policy/action contract.
    c3_state_inheritance_topologies_per_module_count: int = 1
    c3_state_inheritance_focus_module_counts: list[int] = field(
        default_factory=list
    )
    c3_state_inheritance_focus_topology_count: int = 1
    c3_state_inheritance_initial_phase_indices: list[int] = field(
        default_factory=lambda: [1, 3]
    )
    # Optional exact reset-bank fraction for a transition-targeted backward
    # curriculum.  When set, every inherited-state episode starts from this
    # same persisted intra-phase state; evidence-gated later stages may move
    # this value earlier without changing policy, reward, or controller
    # semantics.
    c3_state_inheritance_fixed_reset_progress_fraction: float | None = None

    def c3_state_inheritance_topology_count(self, module_count: int) -> int:
        """Resolve continuous topology coverage for one morphology stratum."""

        return (
            int(self.c3_state_inheritance_focus_topology_count)
            if int(module_count) in {
                int(value) for value in self.c3_state_inheritance_focus_module_counts
            }
            else int(self.c3_state_inheritance_topologies_per_module_count)
        )

    def validate(self) -> None:
        if self.seed < 0:
            raise SchemaValidationError("Order9 production seed must be non-negative")
        for name in (
            "device",
            "artifact_root",
            "runtime_benchmark_report_path",
            "canonical_order8_report_path",
            "robot_model_config_path",
        ):
            require_non_empty(
                str(getattr(self, name)), f"Order9ProductionRuntimeConfig.{name}"
            )
        if self.selected_environment_count < 1:
            raise SchemaValidationError(
                "Order9 selected_environment_count must be positive"
            )
        if (
            not math.isfinite(self.runtime_load_sample_interval_s)
            or self.runtime_load_sample_interval_s <= 0.0
        ):
            raise SchemaValidationError(
                "Order9 runtime_load_sample_interval_s must be positive"
            )
        for name in (
            "checkpoint_interval_updates",
            "metrics_flush_interval_updates",
            "full_mesh_evaluation_interval_updates",
            "full_mesh_evaluation_episode_count",
        ):
            if int(getattr(self, name)) < 1:
                raise SchemaValidationError(
                    f"Order9ProductionRuntimeConfig.{name} must be positive"
                )
        if self.c3_wrench_gate_curriculum_start_update_index < 0:
            raise SchemaValidationError(
                "Order9 C3 wrench-gate curriculum start must be non-negative"
            )
        wrench_gate_scales = [
            float(value) for value in self.c3_wrench_gate_curriculum_scales
        ]
        if (
            not wrench_gate_scales
            or any(not math.isfinite(value) or value < 1.0 for value in wrench_gate_scales)
            or any(
                later > earlier
                for earlier, later in zip(wrench_gate_scales, wrench_gate_scales[1:])
            )
            or wrench_gate_scales[-1] != 1.0
        ):
            raise SchemaValidationError(
                "Order9 C3 wrench-gate curriculum must be finite, non-increasing, "
                "at least one, and terminate at exactly one"
            )
        for name in (
            "runtime_benchmark_report_sha256",
            "canonical_order8_report_sha256",
        ):
            value = str(getattr(self, name))
            if value and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaValidationError(
                    f"Order9ProductionRuntimeConfig.{name} must be a SHA-256 digest"
                )
        if not self.tensorized_rollout_hot_path:
            raise SchemaValidationError("Order9 production rollout must be tensorized")
        if self.raw_contact_actor_input:
            raise SchemaValidationError("Order9 actor input must exclude raw contact truth")
        if self.full_mesh_acceptance_replaced:
            raise SchemaValidationError(
                "Order9 training approximation cannot replace full-mesh acceptance"
            )
        fractions = [
            float(value) for value in self.c3_phase_reset_progress_fractions
        ]
        if (
            len(fractions) < 4
            or fractions != sorted(set(fractions))
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in fractions
            )
        ):
            raise SchemaValidationError(
                "Order9 C3 reset progress must be unique ordered values strictly "
                "inside (0, 1)"
            )
        try:
            order9_c3_reset_strata_by_phase(
                phase_labels=(
                    "approach",
                    "contact_acquisition",
                    "lift",
                    "transport",
                    "place",
                    "release",
                    "retreat",
                    "settle",
                ),
                progress_fractions=fractions,
                boundary_tail_phase_labels=(
                    self.c3_boundary_tail_phase_labels
                ),
                boundary_tail_progress_fraction=(
                    self.c3_boundary_tail_progress_fraction
                ),
            )
        except ValueError as exc:
            raise SchemaValidationError(str(exc)) from exc
        if self.c3_contact_reset_preload_steps < 1:
            raise SchemaValidationError(
                "Order9 C3 contact reset preload must be positive"
            )
        if self.c3_contact_reset_min_selected_contacts < 2:
            raise SchemaValidationError(
                "Order9 C3 contact reset requires at least two contacts"
            )
        for name in (
            "c3_contact_reset_max_object_displacement_m",
            "c3_contact_reset_max_downward_speed_mps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9ProductionRuntimeConfig.{name} must be positive"
                )
        if (
            not math.isfinite(self.c3_virtual_contact_inward_lead_m)
            or not 0.0 <= self.c3_virtual_contact_inward_lead_m <= 0.02
        ):
            raise SchemaValidationError(
                "Order9 C3 virtual contact lead must be in [0, 0.02] m"
            )
        for name in (
            "c3_actuator_aware_nominal_preload_maximum_m",
            "c3_actuator_aware_nominal_preload_quantization_m",
            "c3_actuator_aware_nominal_preload_support_safety_factor",
            "c3_actuator_aware_nominal_preload_maximum_peak_utilization",
            "c3_actuator_aware_nominal_preload_minimum_anchor_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9ProductionRuntimeConfig.{name} must be positive"
                )
        if (
            not math.isfinite(
                self.c3_actuator_aware_nominal_preload_model_error_margin_m
            )
            or self.c3_actuator_aware_nominal_preload_model_error_margin_m < 0.0
        ):
            raise SchemaValidationError(
                "Order9 actuator-aware nominal preload model-error margin "
                "must be non-negative"
            )
        if (
            self.c3_actuator_aware_nominal_preload_maximum_m
            < self.c3_virtual_contact_inward_lead_m
            or self.c3_actuator_aware_nominal_preload_maximum_m > 0.05
            or self.c3_actuator_aware_nominal_preload_quantization_m
            > self.c3_actuator_aware_nominal_preload_maximum_m
            or self.c3_actuator_aware_nominal_preload_model_error_margin_m
            > self.c3_actuator_aware_nominal_preload_maximum_m
            or not 0.0
            < self.c3_actuator_aware_nominal_preload_maximum_peak_utilization
            <= 1.0
            or not 0.0
            < self.c3_actuator_aware_nominal_preload_minimum_anchor_fraction
            <= 1.0
        ):
            raise SchemaValidationError(
                "Order9 actuator-aware nominal preload bounds are invalid"
            )
        if (
            not math.isfinite(self.c3_contact_compression_action_span_m)
            or not 0.0 <= self.c3_contact_compression_action_span_m <= 0.02
            or (
                self.c3_contact_compression_action_adapter_enabled
                and self.c3_contact_compression_action_span_m <= 0.0
            )
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-compression action span must be in "
                "(0, 0.02] m when enabled"
            )
        compression_only_counts = [
            int(value)
            for value in self.c3_contact_compression_only_module_counts
        ]
        if (
            compression_only_counts != sorted(set(compression_only_counts))
            or any(not 2 <= value <= 8 for value in compression_only_counts)
            or (
                compression_only_counts
                and not self.c3_contact_compression_action_adapter_enabled
            )
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-compression-only module counts must be "
                "unique, ordered, lie in [2, 8], and require the adapter"
            )
        joint_only_counts = [
            int(value) for value in self.c3_joint_only_module_counts
        ]
        if (
            joint_only_counts != sorted(set(joint_only_counts))
            or any(not 2 <= value <= 8 for value in joint_only_counts)
            or set(joint_only_counts).intersection(compression_only_counts)
        ):
            raise SchemaValidationError(
                "Order9 C3 joint-only module counts must be unique, ordered, "
                "lie in [2, 8], and not overlap compression-only counts"
            )
        for name in (
            "c3_force_estimator_baseline_update_alpha",
            "c3_force_estimator_filter_alpha",
            "c3_force_estimator_minimum_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise SchemaValidationError(
                    f"Order9ProductionRuntimeConfig.{name} must be in (0, 1]"
                )
        for name in (
            "c3_force_estimator_ridge_damping_m2",
            "c3_force_estimator_minimum_jacobian_norm_m",
            "c3_force_estimator_fit_residual_scale_nm",
            "c3_force_estimator_support_safety_factor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9ProductionRuntimeConfig.{name} must be positive"
                )
        if not self.c3_topology_stratified_updates:
            raise SchemaValidationError(
                "Order9 C3 production PPO requires topology-stratified updates"
            )
        if self.c3_topologies_per_module_count_per_update < 1:
            raise SchemaValidationError(
                "Order9 C3 topologies per module count must be positive"
            )
        if self.c3_topology_shard_parallel_process_count < 1:
            raise SchemaValidationError(
                "Order9 C3 topology shard parallel process count must be positive"
            )
        inheritance_phases = [
            int(value) for value in self.c3_state_inheritance_initial_phase_indices
        ]
        inheritance_focus_modules = [
            int(value) for value in self.c3_state_inheritance_focus_module_counts
        ]
        if self.c3_state_inheritance_rollouts_enabled:
            topology_counts = [
                self.c3_state_inheritance_topology_count(module_count)
                for module_count in range(2, 9)
            ]
            fixed_progress = (
                None
                if self.c3_state_inheritance_fixed_reset_progress_fraction is None
                else float(
                    self.c3_state_inheritance_fixed_reset_progress_fraction
                )
            )
            common_invalid = (
                self.c3_state_inheritance_rollout_steps < 1
                or self.c3_state_inheritance_environment_count_per_module < 1
                or self.c3_state_inheritance_topologies_per_module_count < 0
                or self.c3_state_inheritance_focus_topology_count < 1
                or self.c3_state_inheritance_focus_topology_count
                < self.c3_state_inheritance_topologies_per_module_count
                or not any(value > 0 for value in topology_counts)
                or len(inheritance_focus_modules)
                != len(set(inheritance_focus_modules))
                or any(value < 2 or value > 8 for value in inheritance_focus_modules)
                or self.c3_state_inheritance_environment_count_per_module
                < len(inheritance_phases)
                or len(inheritance_phases) != len(set(inheritance_phases))
                or any(value not in {1, 2, 3} for value in inheritance_phases)
            )
            phase_invalid = (
                set(inheritance_phases) != {1, 2}
                if fixed_progress is not None
                else not {1, 3}.issubset(inheritance_phases)
            )
            fixed_progress_invalid = False
            if fixed_progress is not None:
                try:
                    order9_c3_fixed_reset_stratum_index(
                        progress_fractions=fractions,
                        reset_progress_fraction=fixed_progress,
                    )
                except ValueError:
                    fixed_progress_invalid = True
            if common_invalid or phase_invalid or fixed_progress_invalid:
                raise SchemaValidationError(
                    "Order9 C3 state-inheritance rollout requires a positive "
                    "runtime, non-negative base topology coverage with at least "
                    "one selected topology, valid focus coverage, unique seed "
                    "phases, and an exact reset-bank stratum. A fixed-progress "
                    "backward curriculum must seed contact acquisition and lift."
                )


@dataclass
class Order9HardCheckerConfig(SchemaBase):
    backend: str = "hybrid_lightweight_qp_persistent_isaac_shadow"
    max_proposal_attempts: int = 2
    qp_residual_threshold: float = 1.0e-4
    wrench_residual_threshold: float = 1.0e-3
    qp_force_scale_n: float = 30.0
    qp_torque_scale_nm: float = 5.0
    qp_solver_absolute_tolerance: float = 1.0e-5
    qp_solver_relative_tolerance: float = 1.0e-5
    qp_solver_max_iterations: int = 4000
    shadow_rollout_horizon_s: float = 3.0
    shadow_control_dt_s: float = 0.02
    require_isolated_persistent_worker: bool = True
    require_current_pi_l_checkpoint: bool = True
    main_state_digest_required: bool = True
    projection_allowed: bool = False
    allowed_contact_semantics: str = "explicit_active_assignment_pairs_only"

    def validate(self) -> None:
        if self.backend != "hybrid_lightweight_qp_persistent_isaac_shadow":
            raise SchemaValidationError(
                "Order9 production C_H requires the approved hybrid backend"
            )
        if self.max_proposal_attempts != 2:
            raise SchemaValidationError(
                "Order9 pi_H uses exactly two checked learned attempts"
            )
        for name in (
            "qp_residual_threshold",
            "wrench_residual_threshold",
            "qp_force_scale_n",
            "qp_torque_scale_nm",
            "qp_solver_absolute_tolerance",
            "qp_solver_relative_tolerance",
            "shadow_rollout_horizon_s",
            "shadow_control_dt_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9HardCheckerConfig.{name} must be positive"
                )
        if self.qp_solver_max_iterations < 1:
            raise SchemaValidationError(
                "Order9 C_H QP iteration limit must be positive"
            )
        if self.shadow_rollout_horizon_s < self.shadow_control_dt_s:
            raise SchemaValidationError(
                "Order9 C_H shadow horizon must cover one control step"
            )
        if not (
            self.require_isolated_persistent_worker
            and self.require_current_pi_l_checkpoint
            and self.main_state_digest_required
        ):
            raise SchemaValidationError(
                "Order9 production shadow must be isolated, checkpoint-bound, and digest-audited"
            )
        if self.projection_allowed:
            raise SchemaValidationError("Order9 C_H must not project pi_H output")
        if self.allowed_contact_semantics != "explicit_active_assignment_pairs_only":
            raise SchemaValidationError(
                "Order9 collision exceptions must be explicit assignment pairs"
            )


@dataclass
class Order9LearningConfig(SchemaBase):
    curriculum: Order9CurriculumSchedule
    runtime_benchmark: Order9RuntimeBenchmarkConfig
    randomization: Order9ConservativeRandomizationConfig
    teacher_collection: Order9TeacherCollectionRuntimeConfig = field(
        default_factory=Order9TeacherCollectionRuntimeConfig
    )
    expanded_randomization: Order9ExpandedObjectRandomizationConfig = field(
        default_factory=Order9ExpandedObjectRandomizationConfig
    )
    optimization: Order9OptimizationConfig = field(
        default_factory=Order9OptimizationConfig
    )
    reward: P4_3RewardConfig = field(default_factory=P4_3RewardConfig)
    production_runtime: Order9ProductionRuntimeConfig = field(
        default_factory=Order9ProductionRuntimeConfig
    )
    hard_checker: Order9HardCheckerConfig = field(
        default_factory=Order9HardCheckerConfig
    )

    def validate(self) -> None:
        self.teacher_collection.validate()
        self.optimization.c3_boundary_fine_tune.validate()
        c0 = self.curriculum.stages[0]
        if c0.minimum_episodes != self.teacher_collection.episode_count:
            raise SchemaValidationError(
                "Order9 C0 stage minimum_episodes must match teacher collection count"
            )
        if c0.object_distribution != self.teacher_collection.condition_distribution:
            raise SchemaValidationError(
                "Order9 C0 stage distribution must match teacher collection conditions"
            )
        if not self.optimization.pi_l_bc.phase_balanced_sampling:
            raise SchemaValidationError(
                "Order9 C1 pi_L BC requires phase-balanced teacher sampling"
            )
        for stage in self.curriculum.stages:
            runtime = resolve_order9_stage_runtime(self, stage)
            if runtime.generation_environment_steps is None:
                continue
            optimization = order9_ppo_optimization(self, stage)
            if runtime.generation_environment_steps < optimization.minibatch_size:
                raise SchemaValidationError(
                    f"Order9 stage {stage.stage_id!r} generation is smaller than "
                    "its PPO minibatch"
                )
            if runtime.generation_environment_steps % optimization.minibatch_size:
                raise SchemaValidationError(
                    f"Order9 stage {stage.stage_id!r} generation size must be "
                    "divisible by its PPO minibatch"
                )


@dataclass
class Order9ResolvedStageRuntime(SchemaBase):
    environment_count: int
    rollout_steps_per_environment: int | None
    generation_environment_steps: int | None
    environment_count_source: str
    rollout_steps_source: str | None

    def validate(self) -> None:
        if self.environment_count < 1:
            raise SchemaValidationError(
                "Order9 resolved environment_count must be positive"
            )
        if self.rollout_steps_per_environment is None:
            if self.generation_environment_steps is not None:
                raise SchemaValidationError(
                    "Order9 non-PPO runtime cannot declare a generation size"
                )
        elif (
            self.rollout_steps_per_environment < 1
            or self.generation_environment_steps
            != self.environment_count * self.rollout_steps_per_environment
        ):
            raise SchemaValidationError(
                "Order9 resolved PPO generation size is inconsistent"
            )
        require_non_empty(
            self.environment_count_source,
            "Order9ResolvedStageRuntime.environment_count_source",
        )
        if self.rollout_steps_per_environment is not None:
            require_non_empty(
                str(self.rollout_steps_source or ""),
                "Order9ResolvedStageRuntime.rollout_steps_source",
            )


def order9_ppo_optimization(
    config: Order9LearningConfig,
    stage: Order9CurriculumStage,
) -> Order9PPOOptimizationConfig:
    if stage.learning_target == Order9LearningTarget.PI_L:
        return config.optimization.pi_l_ppo
    if stage.learning_target == Order9LearningTarget.PI_H_TRAJECTORY:
        return config.optimization.pi_h_ppo
    if stage.learning_target == Order9LearningTarget.PI_D:
        return config.optimization.pi_d_ppo
    if stage.learning_target == Order9LearningTarget.JOINT_OBJECT_TASK:
        return config.optimization.joint_ppo
    raise SchemaValidationError("Order9 stage has no PPO optimization block")


def resolve_order9_stage_runtime(
    config: Order9LearningConfig,
    stage: Order9CurriculumStage,
) -> Order9ResolvedStageRuntime:
    environment_count = (
        stage.parallel_environment_count
        if stage.parallel_environment_count is not None
        else config.production_runtime.selected_environment_count
    )
    environment_source = (
        "curriculum_stage_override"
        if stage.parallel_environment_count is not None
        else "production_runtime_default"
    )
    if stage.learning_mode != Order9LearningMode.PPO:
        return Order9ResolvedStageRuntime(
            environment_count=environment_count,
            rollout_steps_per_environment=None,
            generation_environment_steps=None,
            environment_count_source=environment_source,
            rollout_steps_source=None,
        )
    optimization = order9_ppo_optimization(config, stage)
    rollout_steps = (
        stage.rollout_steps_per_environment
        if stage.rollout_steps_per_environment is not None
        else optimization.rollout_steps_per_environment
    )
    return Order9ResolvedStageRuntime(
        environment_count=environment_count,
        rollout_steps_per_environment=rollout_steps,
        generation_environment_steps=environment_count * rollout_steps,
        environment_count_source=environment_source,
        rollout_steps_source=(
            "curriculum_stage_override"
            if stage.rollout_steps_per_environment is not None
            else "policy_family_optimization_default"
        ),
    )


def require_order9_stage_execution_allowed(
    config: Order9LearningConfig,
    stage: Order9CurriculumStage,
) -> None:
    if (
        config.curriculum.schedule_version == ORDER9_CURRICULUM_VERSION
        and stage.stage_index <= 2
    ):
        raise SchemaValidationError(
            "Order9 v3 C0--C2 are immutable imported history; use the v2 "
            "schedule only to validate their original artifacts"
        )


@dataclass
class Order9StageMetrics(SchemaBase):
    episode_count: int
    success_count: int
    no_fallback_success_count: int
    safety_failure_episode_count: int
    high_level_decision_count: int
    fallback_decision_count: int
    aggregate_env_steps_per_s: float

    def validate(self) -> None:
        integer_fields = (
            "episode_count",
            "success_count",
            "no_fallback_success_count",
            "safety_failure_episode_count",
            "high_level_decision_count",
            "fallback_decision_count",
        )
        if any(getattr(self, name) < 0 for name in integer_fields):
            raise SchemaValidationError("Order9StageMetrics counts must be non-negative")
        if self.success_count > self.episode_count:
            raise SchemaValidationError("Order9 success_count cannot exceed episode_count")
        if self.no_fallback_success_count > self.success_count:
            raise SchemaValidationError(
                "Order9 no_fallback_success_count cannot exceed success_count"
            )
        if self.fallback_decision_count > self.high_level_decision_count:
            raise SchemaValidationError(
                "Order9 fallback_decision_count cannot exceed high_level_decision_count"
            )
        if not math.isfinite(self.aggregate_env_steps_per_s) or self.aggregate_env_steps_per_s < 0.0:
            raise SchemaValidationError("Order9 aggregate throughput must be finite and non-negative")

    @property
    def success_rate(self) -> float:
        return self.success_count / self.episode_count if self.episode_count else 0.0

    @property
    def no_fallback_success_rate(self) -> float:
        return (
            self.no_fallback_success_count / self.episode_count
            if self.episode_count
            else 0.0
        )

    @property
    def fallback_rate(self) -> float:
        """Fraction of high-level decisions executed by deterministic fallback."""

        return (
            self.fallback_decision_count / self.high_level_decision_count
            if self.high_level_decision_count
            else 0.0
        )


@dataclass
class Order9PromotionDecision(SchemaBase):
    promote: bool
    failed_gates: list[str]
    measured_success_rate: float
    measured_no_fallback_success_rate: float
    measured_fallback_rate: float
    measured_aggregate_env_steps_per_s: float


def evaluate_stage_promotion(
    stage: Order9CurriculumStage,
    metrics: Order9StageMetrics,
    benchmark: Order9RuntimeBenchmarkConfig,
) -> Order9PromotionDecision:
    failed: list[str] = []
    if metrics.episode_count < stage.minimum_episodes:
        failed.append("minimum_episodes")
    if metrics.success_rate < stage.minimum_success_rate:
        failed.append("minimum_success_rate")
    if metrics.no_fallback_success_rate < stage.minimum_no_fallback_success_rate:
        failed.append("minimum_no_fallback_success_rate")
    if metrics.fallback_rate > stage.maximum_fallback_rate:
        failed.append("maximum_fallback_rate")
    if metrics.safety_failure_episode_count > stage.maximum_safety_failure_episodes:
        failed.append("maximum_safety_failure_episodes")
    if (
        stage.learning_mode == Order9LearningMode.PPO
        and metrics.aggregate_env_steps_per_s
        < benchmark.minimum_aggregate_env_steps_per_s
    ):
        failed.append("minimum_aggregate_env_steps_per_s")
    return Order9PromotionDecision(
        promote=not failed,
        failed_gates=failed,
        measured_success_rate=metrics.success_rate,
        measured_no_fallback_success_rate=metrics.no_fallback_success_rate,
        measured_fallback_rate=metrics.fallback_rate,
        measured_aggregate_env_steps_per_s=metrics.aggregate_env_steps_per_s,
    )


def load_order9_learning_config(
    path: str | Path = "configs/training/order9_learning_curriculum.yaml",
) -> Order9LearningConfig:
    return Order9LearningConfig.from_dict(load_config(path))


_PROGRESSIVE_RING_STAGE_SPECS = (
    (
        "teacher_trajectory_collection",
        Order9LearningMode.COLLECTION,
        Order9LearningTarget.DATASET,
        PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY,
        True,
        False,
    ),
    (
        "pi_l_bc_teacher_trajectory",
        Order9LearningMode.BEHAVIOR_CLONING,
        Order9LearningTarget.PI_L,
        PiHOutputScope.NOT_APPLICABLE,
        True,
        False,
    ),
    (
        "pi_l_ppo_teacher_trajectory",
        Order9LearningMode.PPO,
        Order9LearningTarget.PI_L,
        PiHOutputScope.NOT_APPLICABLE,
        True,
        False,
    ),
    (
        "pi_h_assignment_bc",
        Order9LearningMode.BEHAVIOR_CLONING,
        Order9LearningTarget.PI_H_ASSIGNMENT,
        PiHOutputScope.ASSIGNMENT_ONLY_WARMUP,
        True,
        False,
    ),
    (
        "pi_h_full_trajectory_bc",
        Order9LearningMode.BEHAVIOR_CLONING,
        Order9LearningTarget.PI_H_TRAJECTORY,
        PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY,
        True,
        False,
    ),
    (
        "pi_h_ppo_frozen_pi_l",
        Order9LearningMode.PPO,
        Order9LearningTarget.PI_H_TRAJECTORY,
        PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY,
        False,
        False,
    ),
    (
        "pi_l_readaptation_frozen_pi_h",
        Order9LearningMode.PPO,
        Order9LearningTarget.PI_L,
        PiHOutputScope.NOT_APPLICABLE,
        False,
        False,
    ),
)


def _validate_progressive_object_condition_schedule(
    stages: list[Order9CurriculumStage],
) -> None:
    prefix_ids = (
        "c0_order8_teacher_collection",
        "c1_pi_l_bc_fixed_nominal",
        "c2_pi_l_ppo_fixed_conservative",
        "c3_pi_l_ppo_arbitrary_morphology",
    )
    if tuple(stage.stage_id for stage in stages[:4]) != prefix_ids:
        raise SchemaValidationError(
            "Order9 v3 must preserve C0--C3 as the curriculum prefix"
        )
    if any(
        stage.object_distribution
        != ObjectDistributionLevel.CONSERVATIVE_ORDER8_ANCHOR
        for stage in stages[:4]
    ):
        raise SchemaValidationError(
            "Order9 v3 C0--C3 must retain the conservative Order8 anchor"
        )

    ring_specs = (
        ("r1", ObjectDistributionLevel.REACHABLE_POSE_EXPANSION),
        ("r2", ObjectDistributionLevel.BOX_PROPERTY_EXPANSION),
        ("r3", ObjectDistributionLevel.EXPANDED_PRIMITIVES),
        ("r4", ObjectDistributionLevel.EXPANDED_CROSS_PRODUCT),
    )
    cursor = 4
    for ring_id, distribution in ring_specs:
        for (
            suffix,
            mode,
            target,
            output_scope,
            teacher_required,
            design_mask_required,
        ) in _PROGRESSIVE_RING_STAGE_SPECS:
            if cursor >= len(stages):
                raise SchemaValidationError(
                    f"Order9 v3 is missing the {ring_id} progressive cycle"
                )
            stage = stages[cursor]
            expected_id = f"{ring_id}_{suffix}"
            if (
                stage.stage_id != expected_id
                or stage.learning_mode != mode
                or stage.learning_target != target
                or stage.pi_h_output_scope != output_scope
                or stage.deterministic_teacher_required != teacher_required
                or stage.design_action_mask_required != design_mask_required
                or stage.object_distribution != distribution
            ):
                raise SchemaValidationError(
                    f"Order9 v3 stage {cursor} does not match {expected_id!r}"
                )
            if (
                not stage.topology_randomized
                or stage.min_modules != 2
                or stage.max_modules != 8
                or stage.held_out_only
            ):
                raise SchemaValidationError(
                    f"Order9 v3 stage {expected_id!r} must train 2--8-module "
                    "topology-randomized non-held-out tasks"
                )
            cursor += 1
        cumulative_gate = stages[cursor - 1]
        if (
            cumulative_gate.minimum_no_fallback_success_rate <= 0.0
            or cumulative_gate.maximum_fallback_rate > 0.10
        ):
            raise SchemaValidationError(
                f"Order9 v3 {ring_id} readaptation must retain a cumulative "
                "no-fallback physical gate"
            )

    tail_specs = (
        (
            "post_r4_pi_d_structured_bc",
            Order9LearningMode.BEHAVIOR_CLONING,
            Order9LearningTarget.PI_D,
            ObjectDistributionLevel.EXPANDED_CROSS_PRODUCT,
            PiHOutputScope.NOT_APPLICABLE,
            True,
            True,
        ),
        (
            "post_r4_pi_d_masked_ppo",
            Order9LearningMode.PPO,
            Order9LearningTarget.PI_D,
            ObjectDistributionLevel.EXPANDED_CROSS_PRODUCT,
            PiHOutputScope.NOT_APPLICABLE,
            False,
            True,
        ),
        (
            "joint_object_task_ppo",
            Order9LearningMode.PPO,
            Order9LearningTarget.JOINT_OBJECT_TASK,
            ObjectDistributionLevel.EXPANDED_CROSS_PRODUCT,
            PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY,
            False,
            True,
        ),
        (
            "held_out_full_system_evaluation",
            Order9LearningMode.EVALUATION,
            Order9LearningTarget.FULL_SYSTEM,
            ObjectDistributionLevel.HELD_OUT_SHAPES_AND_INERTIA,
            PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY,
            False,
            True,
        ),
    )
    if len(stages) != cursor + len(tail_specs):
        raise SchemaValidationError(
            "Order9 v3 contains an unexpected stage outside the approved "
            "progressive rings and post-R4 tail"
        )
    for (
        expected_id,
        mode,
        target,
        distribution,
        output_scope,
        teacher_required,
        design_mask_required,
    ) in tail_specs:
        stage = stages[cursor]
        if (
            stage.stage_id != expected_id
            or stage.learning_mode != mode
            or stage.learning_target != target
            or stage.object_distribution != distribution
            or stage.pi_h_output_scope != output_scope
            or stage.deterministic_teacher_required != teacher_required
            or stage.design_action_mask_required != design_mask_required
        ):
            raise SchemaValidationError(
                f"Order9 v3 stage {cursor} does not match {expected_id!r}"
            )
        cursor += 1


def _require_bc_before_ppo(
    stages: list[Order9CurriculumStage],
    target: Order9LearningTarget,
) -> None:
    indices = {
        mode: [
            stage.stage_index
            for stage in stages
            if stage.learning_target == target and stage.learning_mode == mode
        ]
        for mode in (Order9LearningMode.BEHAVIOR_CLONING, Order9LearningMode.PPO)
    }
    if not indices[Order9LearningMode.PPO]:
        return
    if not indices[Order9LearningMode.BEHAVIOR_CLONING]:
        raise SchemaValidationError(f"Order9 {target.value} PPO requires a prior BC stage")
    if min(indices[Order9LearningMode.PPO]) < min(indices[Order9LearningMode.BEHAVIOR_CLONING]):
        raise SchemaValidationError(f"Order9 {target.value} BC must precede PPO")
