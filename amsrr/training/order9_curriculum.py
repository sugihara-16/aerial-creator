from __future__ import annotations

"""Validated Order 9 curriculum and promotion/throughput gates."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from amsrr.schemas.common import SchemaBase, SchemaValidationError, StrEnum, require_non_empty
from amsrr.schemas.policies import CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
from amsrr.training.order9_randomization import (
    Order9ConservativeRandomizationConfig,
    Order9ExpandedObjectRandomizationConfig,
)
from amsrr.utils.config import load_config


ORDER9_CURRICULUM_VERSION = "order9_curriculum_v2"
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
    EXPANDED_PRIMITIVES = "expanded_primitives"
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
    shadow_rollout_horizon_s: float = 2.0
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
    production_runtime: Order9ProductionRuntimeConfig = field(
        default_factory=Order9ProductionRuntimeConfig
    )
    hard_checker: Order9HardCheckerConfig = field(
        default_factory=Order9HardCheckerConfig
    )

    def validate(self) -> None:
        self.teacher_collection.validate()
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
