from __future__ import annotations

"""Production orchestration for the staged Order-3 morphology-conditioned pi_L.

The runner deliberately keeps Isaac execution behind process boundaries.  It
also keeps the deterministic-v2 behavior-cloning source separate from learned
online PPO traces: the two report kinds have distinct collection methods and
cannot be mixed into one dataset.
"""

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from amsrr.acceptance.order3_acceptance import (
    ORDER3_AGGREGATE_SUCCESS_THRESHOLD,
    ORDER3_ID_MAX_FALLBACK_RATE,
    ORDER3_NOMINAL_MAX_RELATIVE_DEGRADATION,
    ORDER3_PER_MODULE_SUCCESS_THRESHOLD,
    ORDER3_RANDOMIZED_MIN_RELATIVE_IMPROVEMENT,
    ORDER3_RANDOMIZED_MIN_SUCCESS_GAIN,
    ORDER3_ACCEPTANCE_ARTIFACT_VERSION,
    Order3AcceptanceArtifactMetadata,
    Order3AcceptanceReport,
    run_order3_acceptance_from_paths,
)
from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    StrEnum,
    require_non_empty,
)
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import (
    Order3MorphologyPoolEntry,
    Order3MorphologyPoolManifest,
    Order3PolicyTransition,
)
from amsrr.schemas.order3_rollout_condition import (
    ORDER3_ROLLOUT_CONDITION_VERSION,
    Order3RolloutCondition,
    build_order3_rollout_condition,
)
from amsrr.simulation.isaac_lab_backend import (
    IsaacLabBackend,
    load_isaac_lab_backend_config,
)
from amsrr.simulation.order3_policy_rollout import (
    Order3DeterministicBaselineRolloutConfig,
    Order3DeterministicBaselineRolloutEnv,
    Order3IsaacPolicyRolloutConfig,
    Order3IsaacPolicyRolloutEnv,
    Order3IsaacPolicyRolloutResult,
    order3_condition_report_failures,
    order3_provenance_report_failures,
)
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffEnv
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffResult
from amsrr.training.order3_dataset import (
    Order3DatasetIOResult,
    load_order3_dataset,
    write_order3_dataset,
)
from amsrr.training.order3_morphology_pool import (
    Order3MorphologyPoolConfig,
    build_order3_morphology_pool,
    write_order3_morphology_pool,
)
from amsrr.training.order3_pi_l_training import (
    Order3BCTrainingResult,
    Order3PPOTrainingResult,
    load_order3_pi_l_training_config,
    train_order3_pi_l_bc,
    train_order3_pi_l_ppo,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    load_order3_policy_checkpoint,
)
from amsrr.training.order3_free_flight import (
    ORDER3_FREE_FLIGHT_VERSION,
    Order3EvaluationEpisode,
    Order3TaskMode,
    Order3TerminalMetrics,
)
from amsrr.training.order3_takeoff_collector import (
    Order3TakeoffBCCollectorConfig,
    collect_order3_takeoff_bc_transitions,
)
from amsrr.training.random_morphology_takeoff_runner import (
    RandomMorphologyTakeoffRunnerResult,
    load_random_morphology_takeoff_runner_config,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file, stable_hash


ORDER3_PIPELINE_RUNNER_VERSION = "order3_morphology_pi_l_pipeline_v1"
DEFAULT_ORDER3_PIPELINE_CONFIG_PATH = "configs/training/order3_morphology_pi_l.yaml"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_P4_3_ROOT = (_REPOSITORY_ROOT / "artifacts" / "p4_3").resolve()
_SHA256_LENGTH = 64


class Order3PipelineMode(StrEnum):
    FULL = "full"
    SMOKE = "smoke"


class Order3PipelineStage(StrEnum):
    BUILD_POOL = "build-pool"
    BC_ROLLOUTS = "bc-rollouts"
    COLLECT_BC = "collect-bc"
    TRAIN_BC = "train-bc"
    LEARNED_ROLLOUTS = "learned-rollouts"
    COLLECT_PPO = "collect-ppo"
    TRAIN_PPO = "train-ppo"
    EVALUATE_LEARNED = "evaluate-learned"
    EVALUATE_BASELINE = "evaluate-baseline"
    BUILD_EVALUATION_EPISODES = "build-evaluation-episodes"
    BUILD_ACCEPTANCE_ARTIFACT = "build-acceptance-artifact"
    ACCEPT = "accept"


@dataclass
class Order3PipelinePathsConfig(SchemaBase):
    artifact_root: str = "artifacts/p4_full/order3_pi_l_v2"
    pool_manifest_path: str = (
        "artifacts/p4_full/order3_pi_l_v2/morphology_pool.json"
    )
    bc_dataset_dir: str = "artifacts/p4_full/order3_pi_l_v2/datasets/bc"
    ppo_dataset_dir: str = "artifacts/p4_full/order3_pi_l_v2/datasets/ppo"
    evaluation_path: str = (
        "artifacts/p4_full/order3_pi_l_v2/evaluation/episodes.jsonl"
    )
    takeoff_config_path: str = "configs/training/order2_5_centroidal_control.yaml"
    report_dir: str = "artifacts/p4_full/order3_pi_l_v2/rollouts"
    command_timeout_s: float = 300.0
    dataset_shard_size: int = 100_000

    def validate(self) -> None:
        for name in (
            "artifact_root",
            "pool_manifest_path",
            "bc_dataset_dir",
            "ppo_dataset_dir",
            "evaluation_path",
            "takeoff_config_path",
            "report_dir",
        ):
            value = str(getattr(self, name))
            require_non_empty(value, f"Order3PipelinePathsConfig.{name}")
            _reject_legacy_p4_3_path(Path(value))
        if not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0.0:
            raise SchemaValidationError(
                "Order3PipelinePathsConfig.command_timeout_s must be finite and positive"
            )
        if self.dataset_shard_size <= 0:
            raise SchemaValidationError(
                "Order3PipelinePathsConfig.dataset_shard_size must be positive"
            )


@dataclass
class Order3PipelineAcceptanceConfig(SchemaBase):
    held_out_aggregate_success_min: float = 0.95
    held_out_per_module_count_success_min: float = 0.90
    nominal_baseline_degradation_max: float = 0.05
    disturbed_tracking_error_improvement_min: float = 0.15
    disturbed_success_improvement_min: float = 0.10
    in_distribution_fallback_rate_max: float = 0.01
    require_zero_qp_infeasible_terminal: bool = True
    require_zero_hard_collision_terminal: bool = True
    require_zero_nonfinite_terminal: bool = True
    require_zero_unsupported_actuator_terminal: bool = True
    require_ood_fallback: bool = True
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        bounded = (
            "held_out_aggregate_success_min",
            "held_out_per_module_count_success_min",
            "nominal_baseline_degradation_max",
            "disturbed_tracking_error_improvement_min",
            "disturbed_success_improvement_min",
            "in_distribution_fallback_rate_max",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(
                    f"Order3PipelineAcceptanceConfig.{name} must be in [0, 1]"
                )
        supported = {
            "held_out_aggregate_success_min": ORDER3_AGGREGATE_SUCCESS_THRESHOLD,
            "held_out_per_module_count_success_min": ORDER3_PER_MODULE_SUCCESS_THRESHOLD,
            "nominal_baseline_degradation_max": ORDER3_NOMINAL_MAX_RELATIVE_DEGRADATION,
            "disturbed_tracking_error_improvement_min": (
                ORDER3_RANDOMIZED_MIN_RELATIVE_IMPROVEMENT
            ),
            "disturbed_success_improvement_min": ORDER3_RANDOMIZED_MIN_SUCCESS_GAIN,
            "in_distribution_fallback_rate_max": ORDER3_ID_MAX_FALLBACK_RATE,
        }
        for name, expected in supported.items():
            if not math.isclose(
                float(getattr(self, name)), expected, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise SchemaValidationError(
                    f"Order3PipelineAcceptanceConfig.{name} must match the "
                    f"implemented acceptance gate value {expected}"
                )
        required_true = (
            "require_zero_qp_infeasible_terminal",
            "require_zero_hard_collision_terminal",
            "require_zero_nonfinite_terminal",
            "require_zero_unsupported_actuator_terminal",
            "require_ood_fallback",
        )
        if any(not bool(getattr(self, name)) for name in required_true):
            raise SchemaValidationError(
                "Order3 production acceptance safety/OOD gates cannot be disabled"
            )
        if self.p4_full_completion_claim:
            raise SchemaValidationError(
                "Order3 free-flight pipeline is not P4 full completion"
            )


@dataclass
class Order3ConfiguredCurriculumStage(SchemaBase):
    name: str
    floor_takeoff: bool
    translation_waypoints: bool
    attitude_waypoints: bool
    model_randomization_scale: float
    disturbance_scale: float
    initial_state_randomization_scale: float = 0.0

    def validate(self) -> None:
        require_non_empty(self.name, "Order3ConfiguredCurriculumStage.name")
        if self.floor_takeoff and (
            self.translation_waypoints or self.attitude_waypoints
        ):
            raise SchemaValidationError(
                "Order3 configured floor-takeoff and waypoint modes must be separate stages"
            )
        if (
            not math.isfinite(self.model_randomization_scale)
            or not 0.0 <= self.model_randomization_scale <= 0.5
        ):
            raise SchemaValidationError(
                "Order3 model_randomization_scale must be in [0, 0.5]"
            )
        if not math.isfinite(self.disturbance_scale) or self.disturbance_scale < 0.0:
            raise SchemaValidationError(
                "Order3 disturbance_scale must be finite and non-negative"
            )
        if (
            not math.isfinite(self.initial_state_randomization_scale)
            or not 0.0 <= self.initial_state_randomization_scale <= 1.0
        ):
            raise SchemaValidationError(
                "Order3 initial_state_randomization_scale must be in [0, 1]"
            )
        if self.floor_takeoff and self.initial_state_randomization_scale > 0.0:
            raise SchemaValidationError(
                "Order3 floor-takeoff stage cannot randomize its floor initial state"
            )


@dataclass
class Order3ConfiguredCurriculum(SchemaBase):
    stages: list[Order3ConfiguredCurriculumStage]

    def validate(self) -> None:
        if not self.stages:
            raise SchemaValidationError("Order3 configured curriculum must not be empty")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise SchemaValidationError(
                "Order3 configured curriculum stage names must be unique"
            )
        if not any(stage.floor_takeoff for stage in self.stages):
            raise SchemaValidationError(
                "Order3 configured curriculum must contain floor takeoff"
            )
        if not any(
            stage.translation_waypoints or stage.attitude_waypoints
            for stage in self.stages
        ):
            raise SchemaValidationError(
                "Order3 configured curriculum must contain waypoint tracking"
            )


@dataclass(frozen=True)
class Order3PipelineRunnerConfig:
    phase: str
    pipeline: Order3PipelinePathsConfig
    pool: Order3MorphologyPoolConfig
    curriculum: Order3ConfiguredCurriculum
    acceptance: Order3PipelineAcceptanceConfig
    recurrent_hidden_dim: int
    training_seed: int
    ppo_orchestration_updates: int
    config_path: str
    config_sha256: str
    runner_version: str = ORDER3_PIPELINE_RUNNER_VERSION

    def __post_init__(self) -> None:
        if self.phase != "P4-full-order3":
            raise SchemaValidationError(
                "Order3 pipeline config phase must be 'P4-full-order3'"
            )
        if self.recurrent_hidden_dim <= 0:
            raise SchemaValidationError(
                "Order3 pipeline recurrent_hidden_dim must be positive"
            )
        if self.training_seed < 0:
            raise SchemaValidationError(
                "Order3 pipeline training_seed must be non-negative"
            )
        if self.ppo_orchestration_updates <= 0:
            raise SchemaValidationError(
                "Order3 pipeline ppo_orchestration_updates must be positive"
            )
        if not _is_sha256(self.config_sha256):
            raise SchemaValidationError(
                "Order3 pipeline config_sha256 must be a lowercase SHA-256 digest"
            )
        if self.runner_version != ORDER3_PIPELINE_RUNNER_VERSION:
            raise SchemaValidationError("Order3 pipeline runner version mismatch")


@dataclass
class Order3PipelineCommand(SchemaBase):
    stage: Order3PipelineStage
    mode: Order3PipelineMode
    split: DatasetSplit
    module_count: int
    structural_hash: str
    graph_path: str
    report_path: str
    argv: list[str]
    real_isaac_requested: bool
    condition_hash: str | None = None
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError("Order3 pipeline command module_count must be in [2, 8]")
        if not _is_sha256(self.structural_hash):
            raise SchemaValidationError("Order3 pipeline command structural_hash must be sha256")
        for name in ("graph_path", "report_path"):
            require_non_empty(str(getattr(self, name)), f"Order3PipelineCommand.{name}")
            _reject_legacy_p4_3_path(Path(getattr(self, name)))
        if not self.argv or not all(isinstance(value, str) and value for value in self.argv):
            raise SchemaValidationError("Order3 pipeline command argv must be non-empty strings")
        if self.condition_hash is not None and not _is_sha256(self.condition_hash):
            raise SchemaValidationError(
                "Order3 pipeline command condition_hash must be sha256 when present"
            )
        if self.p4_full_completion_claim:
            raise SchemaValidationError("Order3 command cannot claim P4 full completion")


@dataclass
class Order3PipelineCommandPlan(SchemaBase):
    runner_version: str
    stage: Order3PipelineStage
    mode: Order3PipelineMode
    config_sha256: str
    pool_hash: str
    commands: list[Order3PipelineCommand]
    condition_hashes: list[str]
    full_pool_coverage: bool
    execution_requested: bool
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if self.runner_version != ORDER3_PIPELINE_RUNNER_VERSION:
            raise SchemaValidationError("Order3 command plan runner version mismatch")
        if not _is_sha256(self.config_sha256) or not _is_sha256(self.pool_hash):
            raise SchemaValidationError("Order3 command plan hashes must be SHA-256")
        if not self.commands:
            raise SchemaValidationError("Order3 command plan must not be empty")
        if any(command.stage != self.stage for command in self.commands):
            raise SchemaValidationError("Order3 command plan contains a different stage")
        if any(command.mode != self.mode for command in self.commands):
            raise SchemaValidationError("Order3 command plan contains a different mode")
        expected_conditions = sorted(
            {
                command.condition_hash
                for command in self.commands
                if command.condition_hash is not None
            }
        )
        if self.condition_hashes != expected_conditions:
            raise SchemaValidationError(
                "Order3 command plan condition hashes do not match commands"
            )
        if self.mode == Order3PipelineMode.SMOKE and self.full_pool_coverage:
            raise SchemaValidationError("Order3 smoke plan cannot claim full-pool coverage")
        if self.p4_full_completion_claim:
            raise SchemaValidationError("Order3 command plan cannot claim P4 full completion")


@dataclass
class Order3PPOOrchestrationResult(SchemaBase):
    runner_version: str
    mode: Order3PipelineMode
    requested_update_count: int
    completed_update_count: int
    initial_checkpoint_sha256: str
    final_checkpoint_path: str
    final_checkpoint_sha256: str
    updates: list[dict[str, Any]]
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if self.runner_version != ORDER3_PIPELINE_RUNNER_VERSION:
            raise SchemaValidationError("Order3 PPO orchestration runner version mismatch")
        if self.requested_update_count <= 0:
            raise SchemaValidationError("Order3 PPO orchestration update count must be positive")
        if self.completed_update_count != len(self.updates):
            raise SchemaValidationError(
                "Order3 PPO orchestration completed count must match update records"
            )
        if self.completed_update_count > self.requested_update_count:
            raise SchemaValidationError(
                "Order3 PPO orchestration completed too many updates"
            )
        for value in (
            self.initial_checkpoint_sha256,
            self.final_checkpoint_sha256,
        ):
            if not _is_sha256(value):
                raise SchemaValidationError(
                    "Order3 PPO orchestration checkpoint hashes must be sha256"
                )
        require_non_empty(
            self.final_checkpoint_path,
            "Order3PPOOrchestrationResult.final_checkpoint_path",
        )
        if self.p4_full_completion_claim:
            raise SchemaValidationError(
                "Order3 PPO orchestration cannot claim P4 full completion"
            )


CommandExecutor = Callable[[Sequence[str], float], None]
OnlineCollector = Callable[..., Any]


@dataclass(frozen=True)
class _EvaluationGraph:
    split: DatasetSplit
    module_count: int
    structural_hash: str
    graph_path: Path


def load_order3_pipeline_runner_config(
    path: str | Path = DEFAULT_ORDER3_PIPELINE_CONFIG_PATH,
) -> Order3PipelineRunnerConfig:
    config_path = Path(path)
    _reject_legacy_p4_3_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Order3 pipeline config does not exist: {config_path}")
    data = load_config(config_path)
    if not isinstance(data, dict):
        raise SchemaValidationError("Order3 pipeline config root must be a mapping")
    pool_data = dict(data.get("pool", {}))
    if "mesh_search_dirs" in pool_data:
        raw_mesh_dirs = pool_data["mesh_search_dirs"]
        if not isinstance(raw_mesh_dirs, list):
            raise SchemaValidationError("Order3 pool mesh_search_dirs must be a list")
        pool_data["mesh_search_dirs"] = tuple(str(value) for value in raw_mesh_dirs)
    try:
        pool_config = Order3MorphologyPoolConfig(**pool_data)
    except TypeError as exc:
        raise SchemaValidationError(f"invalid Order3 pool config: {exc}") from exc
    policy_data = data.get("policy", {})
    if not isinstance(policy_data, dict):
        raise SchemaValidationError("Order3 policy config must be a mapping")
    recurrent_hidden_dim = policy_data.get("recurrent_hidden_dim", 128)
    if not isinstance(recurrent_hidden_dim, int) or isinstance(recurrent_hidden_dim, bool):
        raise SchemaValidationError("Order3 policy recurrent_hidden_dim must be an integer")
    training_data = data.get("training", {})
    if not isinstance(training_data, dict):
        raise SchemaValidationError("Order3 training config must be a mapping")
    training_seed = training_data.get("seed", 3011)
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        raise SchemaValidationError("Order3 training seed must be an integer")
    ppo_data = training_data.get("ppo", {})
    if not isinstance(ppo_data, dict):
        raise SchemaValidationError("Order3 PPO training config must be a mapping")
    ppo_orchestration_updates = ppo_data.get("updates", 40)
    if not isinstance(ppo_orchestration_updates, int) or isinstance(
        ppo_orchestration_updates, bool
    ):
        raise SchemaValidationError("Order3 PPO updates must be an integer")
    return Order3PipelineRunnerConfig(
        phase=str(data.get("phase", "")),
        pipeline=Order3PipelinePathsConfig.from_dict(data.get("pipeline", {})),
        pool=pool_config,
        curriculum=Order3ConfiguredCurriculum.from_dict(data.get("curriculum", {})),
        acceptance=Order3PipelineAcceptanceConfig.from_dict(data.get("acceptance", {})),
        recurrent_hidden_dim=recurrent_hidden_dim,
        training_seed=training_seed,
        ppo_orchestration_updates=ppo_orchestration_updates,
        config_path=str(config_path),
        config_sha256=hash_file(config_path),
    )


class Order3PipelineRunner:
    """Run independently resumable, hash-bound stages of the Order-3 pipeline."""

    def __init__(
        self,
        config: Order3PipelineRunnerConfig,
        *,
        command_executor: CommandExecutor | None = None,
        online_collector: OnlineCollector | None = None,
    ) -> None:
        self.config = config
        self.command_executor = command_executor or _execute_command
        self.online_collector = online_collector

    @classmethod
    def from_config_path(
        cls,
        path: str | Path = DEFAULT_ORDER3_PIPELINE_CONFIG_PATH,
        **kwargs: Any,
    ) -> "Order3PipelineRunner":
        return cls(load_order3_pipeline_runner_config(path), **kwargs)

    @property
    def physical_model(self):
        return build_physical_model_from_config(
            self.config.pool.robot_model_config_path
        )

    def curriculum_conditions(
        self,
        *,
        replicates_per_stage: int = 1,
        stage_ids: Sequence[str] = (),
    ) -> list[Order3RolloutCondition]:
        """Expand YAML curriculum stages into deterministic hash-bound conditions."""

        if replicates_per_stage <= 0:
            raise SchemaValidationError(
                "Order3 curriculum replicates_per_stage must be positive"
            )
        requested = set(stage_ids)
        known = {stage.name for stage in self.config.curriculum.stages}
        if requested - known:
            raise SchemaValidationError(
                f"unknown Order3 curriculum stages: {sorted(requested - known)}"
            )
        conditions: list[Order3RolloutCondition] = []
        for stage_index, stage in enumerate(self.config.curriculum.stages):
            if requested and stage.name not in requested:
                continue
            for replicate_index in range(replicates_per_stage):
                seed = (
                    self.config.training_seed
                    + stage_index * 100_003
                    + replicate_index * 10_007
                )
                conditions.append(_condition_from_curriculum_stage(stage, seed=seed))
        hashes = [condition.condition_hash for condition in conditions]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError("Order3 curriculum produced duplicate condition hashes")
        return conditions

    def evaluation_conditions(
        self,
        *,
        replicates_per_cell: int = 1,
    ) -> list[Order3RolloutCondition]:
        """Build the required task-mode x nominal/randomized evaluation matrix."""

        if replicates_per_cell <= 0:
            raise SchemaValidationError(
                "Order3 evaluation replicates_per_cell must be positive"
            )
        max_model_scale = max(
            stage.model_randomization_scale
            for stage in self.config.curriculum.stages
        )
        max_disturbance_scale = max(
            stage.disturbance_scale for stage in self.config.curriculum.stages
        )
        conditions: list[Order3RolloutCondition] = []
        task_modes = ("hover", "waypoint", "takeoff")
        for task_index, task_mode in enumerate(task_modes):
            for randomized_index, randomized in enumerate((False, True)):
                for replicate_index in range(replicates_per_cell):
                    seed = (
                        self.config.training_seed
                        + 700_001
                        + task_index * 100_003
                        + randomized_index * 10_007
                        + replicate_index * 1_009
                    )
                    direction = _signed_unit(seed, "evaluation_direction")
                    conditions.append(
                        build_order3_rollout_condition(
                            stage_id=(
                                f"evaluation_{task_mode}_"
                                f"{'randomized' if randomized else 'nominal'}"
                            ),
                            task_mode=task_mode,
                            seed=seed,
                            initial_position_offset_world=(
                                0.05 * direction
                                if randomized and task_mode != "takeoff"
                                else 0.0,
                                0.0,
                                0.0,
                            ),
                            initial_orientation_rpy_rad=(
                                0.0,
                                0.05 * direction
                                if randomized and task_mode != "takeoff"
                                else 0.0,
                                0.0,
                            ),
                            initial_linear_velocity_world=(
                                0.05 * direction
                                if randomized and task_mode != "takeoff"
                                else 0.0,
                                0.0,
                                0.0,
                            ),
                            initial_angular_velocity_body=(
                                0.0,
                                0.0,
                                0.05 * direction
                                if randomized and task_mode != "takeoff"
                                else 0.0,
                            ),
                            waypoint_position_offset_world=(
                                0.25 * direction if task_mode == "waypoint" else 0.0,
                                0.125 if task_mode == "waypoint" else 0.0,
                                0.10 if task_mode == "waypoint" else 0.0,
                            ),
                            waypoint_orientation_rpy_rad=(
                                0.0,
                                0.0,
                                0.25 * direction if task_mode == "waypoint" else 0.0,
                            ),
                            external_wrench_body=(
                                max_disturbance_scale * direction
                                if randomized
                                else 0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.1 * max_disturbance_scale * direction
                                if randomized
                                else 0.0,
                            ),
                            disturbance_start_s=3.0,
                            disturbance_duration_s=(
                                1.0 if randomized and max_disturbance_scale > 0.0 else 0.0
                            ),
                            mass_scale=(
                                1.0 + max_model_scale * direction
                                if randomized
                                else 1.0
                            ),
                            inertia_scale=(
                                1.0
                                + max_model_scale
                                * _signed_unit(seed, "evaluation_inertia")
                                if randomized
                                else 1.0
                            ),
                            thrust_scale=(
                                1.0
                                + max_model_scale
                                * _signed_unit(seed, "evaluation_thrust")
                                if randomized
                                else 1.0
                            ),
                        )
                    )
        _validate_unique_conditions(conditions)
        return conditions

    def build_pool(
        self,
        *,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Order3MorphologyPoolManifest:
        destination = Path(output_path or self.config.pipeline.pool_manifest_path)
        _reject_legacy_p4_3_path(destination)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Order3 morphology pool already exists: {destination}; use overwrite explicitly"
            )
        manifest = build_order3_morphology_pool(
            self.config.pool,
            physical_model=self.physical_model,
        )
        if manifest.config_hash != stable_hash(self.config.pool):
            raise RuntimeError("Order3 morphology pool config hash changed during construction")
        write_order3_morphology_pool(manifest, destination)
        persisted = self.load_pool(destination)
        # ``sampling_metadata`` and feasibility metadata intentionally allow
        # JSON-compatible ``Any`` values; tuple/list normalization can change a
        # pre-serialization schema hash. Bind all downstream stages to the
        # persisted canonical manifest and compare its typed identities here.
        if (
            persisted.config_hash != manifest.config_hash
            or persisted.physical_model_hash != manifest.physical_model_hash
            or [entry.structural_hash for entry in persisted.entries]
            != [entry.structural_hash for entry in manifest.entries]
        ):
            raise RuntimeError("Order3 morphology pool atomic roundtrip mismatch")
        return persisted

    def load_pool(
        self,
        path: str | Path | None = None,
    ) -> Order3MorphologyPoolManifest:
        source = Path(path or self.config.pipeline.pool_manifest_path)
        _reject_legacy_p4_3_path(source)
        if not source.is_file():
            raise FileNotFoundError(f"Order3 morphology pool does not exist: {source}")
        manifest = Order3MorphologyPoolManifest.from_json(source.read_text(encoding="utf-8"))
        if manifest.physical_model_hash != self.physical_model.stable_hash():
            raise SchemaValidationError("Order3 pool PhysicalModel hash mismatch")
        if manifest.config_hash != stable_hash(self.config.pool):
            raise SchemaValidationError("Order3 pool config hash mismatch")
        for entry in manifest.entries:
            if morphology_structural_hash(entry.morphology_graph) != entry.structural_hash:
                raise SchemaValidationError("Order3 pool entry structural hash mismatch")
        return manifest

    def plan_bc_rollouts(
        self,
        *,
        mode: Order3PipelineMode,
        pool_manifest_path: str | Path | None = None,
        graph_paths: Sequence[str | Path] = (),
        real: bool = False,
    ) -> Order3PipelineCommandPlan:
        manifest = self.load_pool(pool_manifest_path)
        entries = self._select_entries(manifest, mode=mode, graph_paths=graph_paths)
        commands: list[Order3PipelineCommand] = []
        for entry, graph_path in entries:
            report_path = self._report_path("bc", entry)
            archive_path = report_path.with_suffix(".jsonl")
            staged_report_path = _staging_path(report_path)
            staged_archive_path = _staging_path(archive_path)
            argv = [
                sys.executable,
                str(_REPOSITORY_ROOT / "scripts" / "random_morphology_takeoff.py"),
                "--config",
                self.config.pipeline.takeoff_config_path,
                "--morphology-graph-json-path",
                str(graph_path),
                "--report-path",
                str(staged_report_path),
                "--archive-path",
                str(staged_archive_path),
            ]
            if real:
                argv.append("--real")
            commands.append(
                Order3PipelineCommand(
                    stage=Order3PipelineStage.BC_ROLLOUTS,
                    mode=mode,
                    split=entry.split,
                    module_count=entry.module_count,
                    structural_hash=entry.structural_hash,
                    graph_path=str(graph_path),
                    report_path=str(report_path),
                    argv=argv,
                    real_isaac_requested=real,
                )
            )
        return self._command_plan(
            stage=Order3PipelineStage.BC_ROLLOUTS,
            mode=mode,
            manifest=manifest,
            commands=commands,
            execution_requested=real,
        )

    def plan_learned_rollouts(
        self,
        *,
        mode: Order3PipelineMode,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        pool_manifest_path: str | Path | None = None,
        graph_paths: Sequence[str | Path] = (),
        real: bool = False,
        external_wrench_body: Sequence[float] = (0.0,) * 6,
        disturbance_start_s: float = 3.0,
        disturbance_duration_s: float = 0.0,
        stochastic: bool = True,
        conditions: Sequence[Order3RolloutCondition] = (),
        condition_seed_namespace: str | None = None,
    ) -> Order3PipelineCommandPlan:
        if not stochastic:
            raise SchemaValidationError(
                "Order3 PPO collection rollouts must use stochastic behavior; "
                "use the learned evaluation stage for deterministic evaluation"
            )
        self._validate_checkpoint(checkpoint_path, checkpoint_sha256, require_file=real)
        rollout_conditions = list(conditions) or [
            build_order3_rollout_condition(
                stage_id="manual_ppo_takeoff",
                task_mode="takeoff",
                seed=self.config.training_seed,
                external_wrench_body=external_wrench_body,
                disturbance_start_s=disturbance_start_s,
                disturbance_duration_s=disturbance_duration_s,
            )
        ]
        _validate_unique_conditions(rollout_conditions)
        manifest = self.load_pool(pool_manifest_path)
        entries = self._select_entries(manifest, mode=mode, graph_paths=graph_paths)
        commands: list[Order3PipelineCommand] = []
        for entry, graph_path in entries:
            for base_condition in rollout_conditions:
                condition = (
                    self._condition_for_episode(
                        base_condition,
                        structural_hash=entry.structural_hash,
                        seed_namespace=condition_seed_namespace,
                    )
                    if condition_seed_namespace is not None
                    else base_condition
                )
                report_path = self._report_path(
                    "ppo",
                    entry,
                    condition=condition,
                    behavior_checkpoint_sha256=checkpoint_sha256,
                )
                staged_report_path = _staging_path(report_path)
                argv = [
                    sys.executable,
                    str(_REPOSITORY_ROOT / "scripts" / "order3_morphology_pi_l.py"),
                    "--config",
                    self.config.config_path,
                    "learned-rollout-one",
                    "--mode",
                    mode.value,
                    "--graph-path",
                    str(graph_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--checkpoint-sha256",
                    checkpoint_sha256,
                    "--report-path",
                    str(staged_report_path),
                    "--rollout-condition-json",
                    condition.to_canonical_json(),
                    "--stochastic",
                ]
                if real:
                    argv.append("--real")
                commands.append(
                    Order3PipelineCommand(
                        stage=Order3PipelineStage.LEARNED_ROLLOUTS,
                        mode=mode,
                        split=entry.split,
                        module_count=entry.module_count,
                        structural_hash=entry.structural_hash,
                        graph_path=str(graph_path),
                        report_path=str(report_path),
                        argv=argv,
                        real_isaac_requested=real,
                        condition_hash=condition.condition_hash,
                    )
                )
        return self._command_plan(
            stage=Order3PipelineStage.LEARNED_ROLLOUTS,
            mode=mode,
            manifest=manifest,
            commands=commands,
            execution_requested=real,
        )

    def plan_learned_evaluation_rollouts(
        self,
        *,
        mode: Order3PipelineMode,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        pool_manifest_path: str | Path | None = None,
        graph_paths: Sequence[str | Path] = (),
        ood_graph_paths: Sequence[str | Path] = (),
        conditions: Sequence[Order3RolloutCondition] = (),
        real: bool = False,
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
    ) -> Order3PipelineCommandPlan:
        _validate_order3_visualization_options(
            real=real,
            viewer=viewer,
            realtime_playback=realtime_playback,
            keep_open_after_rollout_s=keep_open_after_rollout_s,
        )
        self._validate_checkpoint(checkpoint_path, checkpoint_sha256, require_file=real)
        rollout_conditions = list(conditions) or self.evaluation_conditions()
        _validate_unique_conditions(rollout_conditions)
        manifest = self.load_pool(pool_manifest_path)
        graphs = self._select_evaluation_graphs(
            manifest,
            mode=mode,
            graph_paths=graph_paths,
            ood_graph_paths=ood_graph_paths,
        )
        commands: list[Order3PipelineCommand] = []
        for graph in graphs:
            for condition in rollout_conditions:
                report_path = self._evaluation_report_path(
                    "learned",
                    graph,
                    condition,
                    behavior_checkpoint_sha256=checkpoint_sha256,
                )
                argv = [
                    sys.executable,
                    str(_REPOSITORY_ROOT / "scripts" / "order3_morphology_pi_l.py"),
                    "--config",
                    self.config.config_path,
                    "learned-rollout-one",
                    "--mode",
                    mode.value,
                    "--graph-path",
                    str(graph.graph_path),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--checkpoint-sha256",
                    checkpoint_sha256,
                    "--report-path",
                    str(_staging_path(report_path)),
                    "--rollout-condition-json",
                    condition.to_canonical_json(),
                    "--raw-report",
                ]
                if real:
                    argv.append("--real")
                if viewer is not None:
                    argv.extend(["--viewer", viewer])
                if realtime_playback:
                    argv.append("--realtime-playback")
                if keep_open_after_rollout_s > 0.0:
                    argv.extend(
                        [
                            "--keep-open-after-rollout-s",
                            str(keep_open_after_rollout_s),
                        ]
                    )
                commands.append(
                    Order3PipelineCommand(
                        stage=Order3PipelineStage.EVALUATE_LEARNED,
                        mode=mode,
                        split=graph.split,
                        module_count=graph.module_count,
                        structural_hash=graph.structural_hash,
                        graph_path=str(graph.graph_path),
                        report_path=str(report_path),
                        argv=argv,
                        real_isaac_requested=real,
                        condition_hash=condition.condition_hash,
                    )
                )
        return self._command_plan(
            stage=Order3PipelineStage.EVALUATE_LEARNED,
            mode=mode,
            manifest=manifest,
            commands=commands,
            execution_requested=real,
        )

    def plan_baseline_evaluation_rollouts(
        self,
        *,
        mode: Order3PipelineMode,
        pool_manifest_path: str | Path | None = None,
        graph_paths: Sequence[str | Path] = (),
        ood_graph_paths: Sequence[str | Path] = (),
        conditions: Sequence[Order3RolloutCondition] = (),
        real: bool = False,
    ) -> Order3PipelineCommandPlan:
        rollout_conditions = list(conditions) or self.evaluation_conditions()
        _validate_unique_conditions(rollout_conditions)
        manifest = self.load_pool(pool_manifest_path)
        graphs = self._select_evaluation_graphs(
            manifest,
            mode=mode,
            graph_paths=graph_paths,
            ood_graph_paths=ood_graph_paths,
        )
        commands: list[Order3PipelineCommand] = []
        for graph in graphs:
            for condition in rollout_conditions:
                report_path = self._evaluation_report_path(
                    "baseline", graph, condition
                )
                argv = [
                    sys.executable,
                    str(_REPOSITORY_ROOT / "scripts" / "order3_morphology_pi_l.py"),
                    "--config",
                    self.config.config_path,
                    "baseline-rollout-one",
                    "--mode",
                    mode.value,
                    "--graph-path",
                    str(graph.graph_path),
                    "--report-path",
                    str(_staging_path(report_path)),
                    "--rollout-condition-json",
                    condition.to_canonical_json(),
                    "--raw-report",
                ]
                if real:
                    argv.append("--real")
                commands.append(
                    Order3PipelineCommand(
                        stage=Order3PipelineStage.EVALUATE_BASELINE,
                        mode=mode,
                        split=graph.split,
                        module_count=graph.module_count,
                        structural_hash=graph.structural_hash,
                        graph_path=str(graph.graph_path),
                        report_path=str(report_path),
                        argv=argv,
                        real_isaac_requested=real,
                        condition_hash=condition.condition_hash,
                    )
                )
        return self._command_plan(
            stage=Order3PipelineStage.EVALUATE_BASELINE,
            mode=mode,
            manifest=manifest,
            commands=commands,
            execution_requested=real,
        )

    def execute_plan(
        self,
        plan: Order3PipelineCommandPlan,
        *,
        plan_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if plan_path is not None:
            _atomic_write_text(Path(plan_path), plan.to_json(indent=2) + "\n")
        if not plan.execution_requested:
            return {
                "executed": False,
                "command_count": len(plan.commands),
                "report_hashes": {},
                "p4_full_completion_claim": False,
            }
        report_hashes: dict[str, str] = {}
        for command in plan.commands:
            staged_report_path = Path(
                _command_argument(command.argv, "--report-path")
            )
            if staged_report_path.exists():
                staged_report_path.unlink()
            staged_archive_path: Path | None = None
            if "--archive-path" in command.argv:
                staged_archive_path = Path(
                    _command_argument(command.argv, "--archive-path")
                )
                if staged_archive_path.exists():
                    staged_archive_path.unlink()
            self.command_executor(command.argv, self.config.pipeline.command_timeout_s)
            report_path = Path(command.report_path)
            if not staged_report_path.is_file():
                raise RuntimeError(
                    "Order3 rollout command did not create its staged report: "
                    f"{staged_report_path}"
                )
            if command.stage == Order3PipelineStage.BC_ROLLOUTS:
                RandomMorphologyTakeoffRunnerResult.from_json(
                    staged_report_path.read_text(encoding="utf-8")
                )
            elif command.stage in {
                Order3PipelineStage.EVALUATE_BASELINE,
                Order3PipelineStage.EVALUATE_LEARNED,
            }:
                typed_report = json.loads(staged_report_path.read_text(encoding="utf-8"))
                if not isinstance(typed_report, dict):
                    raise SchemaValidationError(
                        "Order3 evaluation raw report must be a JSON object"
                    )
                condition = Order3RolloutCondition.from_json(
                    _command_argument(command.argv, "--rollout-condition-json")
                )
                _validate_report_condition(typed_report, condition)
            else:
                typed_report = Order3IsaacPolicyRolloutResult.from_json(
                    staged_report_path.read_text(encoding="utf-8")
                )
                condition = Order3RolloutCondition.from_json(
                    _command_argument(command.argv, "--rollout-condition-json")
                )
                _validate_report_condition(typed_report.takeoff_result.report, condition)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_report_path, report_path)
            if staged_archive_path is not None and staged_archive_path.is_file():
                archive_path = report_path.with_suffix(".jsonl")
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_archive_path, archive_path)
            report_hashes[str(report_path)] = hash_file(report_path)
        return {
            "executed": True,
            "command_count": len(plan.commands),
            "report_hashes": report_hashes,
            "p4_full_completion_claim": False,
        }

    def run_learned_rollout_one(
        self,
        *,
        graph_path: str | Path,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        report_path: str | Path,
        real: bool,
        external_wrench_body: Sequence[float] = (0.0,) * 6,
        disturbance_start_s: float = 3.0,
        disturbance_duration_s: float = 0.0,
        stochastic: bool = False,
        rollout_condition: Order3RolloutCondition | None = None,
        raw_report: bool = False,
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
    ) -> Order3IsaacPolicyRolloutResult:
        _validate_order3_visualization_options(
            real=real,
            viewer=viewer,
            realtime_playback=realtime_playback,
            keep_open_after_rollout_s=keep_open_after_rollout_s,
        )
        self._validate_checkpoint(checkpoint_path, checkpoint_sha256, require_file=True)
        morphology = _load_graph(graph_path)
        condition = rollout_condition or build_order3_rollout_condition(
            stage_id="manual_learned_rollout",
            task_mode="takeoff",
            seed=self.config.training_seed,
            external_wrench_body=external_wrench_body,
            disturbance_start_s=disturbance_start_s,
            disturbance_duration_s=disturbance_duration_s,
        )
        _, takeoff_config = load_random_morphology_takeoff_runner_config(
            self.config.pipeline.takeoff_config_path
        )
        physical_model = build_physical_model_from_config(
            takeoff_config.robot_model_config_path
        )
        backend = IsaacLabBackend(
            load_isaac_lab_backend_config(takeoff_config.backend_config_path)
        )
        takeoff_env = RandomMorphologyTakeoffEnv(
            config=takeoff_config,
            backend=backend,
            physical_model=physical_model,
        )
        rollout_kwargs: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "expected_checkpoint_sha256": checkpoint_sha256,
            "stochastic": stochastic,
            "external_wrench_body": list(condition.external_wrench_body),
            "disturbance_start_s": condition.disturbance_start_s,
            "disturbance_duration_s": condition.disturbance_duration_s,
        }
        if "rollout_condition" in Order3IsaacPolicyRolloutConfig.__dataclass_fields__:
            rollout_kwargs["rollout_condition"] = condition
        else:  # Temporary compatibility until the runtime consumes the typed field.
            base_executor = takeoff_env.command_executor

            def condition_executor(argv: list[str], timeout_s: float) -> dict[str, Any]:
                return base_executor(
                    [
                        *argv,
                        "--order3-rollout-condition-json",
                        condition.to_canonical_json(),
                    ],
                    timeout_s,
                )

            takeoff_env.command_executor = condition_executor
        result = Order3IsaacPolicyRolloutEnv(
            config=Order3IsaacPolicyRolloutConfig(**rollout_kwargs),
            takeoff_env=takeoff_env,
            viewer=viewer,
            realtime_playback=realtime_playback,
            keep_open_after_rollout_s=keep_open_after_rollout_s,
        ).run(morphology, dry_run=not real)
        if real:
            _validate_report_condition(result.takeoff_result.report, condition)
        payload = (
            json.dumps(result.takeoff_result.report, sort_keys=True, indent=2) + "\n"
            if raw_report
            else result.to_json(indent=2) + "\n"
        )
        _atomic_write_text(Path(report_path), payload)
        if raw_report:
            return result
        persisted = Order3IsaacPolicyRolloutResult.from_json(
            Path(report_path).read_text(encoding="utf-8")
        )
        if persisted.checkpoint_sha256 != checkpoint_sha256:
            raise RuntimeError("Order3 learned rollout report checkpoint roundtrip mismatch")
        return persisted

    def run_baseline_rollout_one(
        self,
        *,
        graph_path: str | Path,
        report_path: str | Path,
        real: bool,
        rollout_condition: Order3RolloutCondition,
        raw_report: bool = False,
    ) -> RandomMorphologyTakeoffResult:
        morphology = _load_graph(graph_path)
        _, takeoff_config = load_random_morphology_takeoff_runner_config(
            self.config.pipeline.takeoff_config_path
        )
        physical_model = build_physical_model_from_config(
            takeoff_config.robot_model_config_path
        )
        takeoff_env = RandomMorphologyTakeoffEnv(
            config=takeoff_config,
            backend=IsaacLabBackend(
                load_isaac_lab_backend_config(takeoff_config.backend_config_path)
            ),
            physical_model=physical_model,
        )
        baseline_kwargs: dict[str, Any] = {
            "external_wrench_body": list(rollout_condition.external_wrench_body),
            "disturbance_start_s": rollout_condition.disturbance_start_s,
            "disturbance_duration_s": rollout_condition.disturbance_duration_s,
        }
        if (
            "rollout_condition"
            in Order3DeterministicBaselineRolloutConfig.__dataclass_fields__
        ):
            baseline_kwargs["rollout_condition"] = rollout_condition
        else:  # Temporary compatibility until runtime consumes the typed field.
            base_executor = takeoff_env.command_executor

            def condition_executor(argv: list[str], timeout_s: float) -> dict[str, Any]:
                return base_executor(
                    [
                        *argv,
                        "--order3-rollout-condition-json",
                        rollout_condition.to_canonical_json(),
                    ],
                    timeout_s,
                )

            takeoff_env.command_executor = condition_executor
        result = Order3DeterministicBaselineRolloutEnv(
            config=Order3DeterministicBaselineRolloutConfig(**baseline_kwargs),
            takeoff_env=takeoff_env,
        ).run(morphology, dry_run=not real)
        if real:
            _validate_report_condition(result.report, rollout_condition)
        payload = (
            json.dumps(result.report, sort_keys=True, indent=2) + "\n"
            if raw_report
            else result.to_json(indent=2) + "\n"
        )
        _atomic_write_text(Path(report_path), payload)
        if raw_report:
            return result
        return RandomMorphologyTakeoffResult.from_json(
            Path(report_path).read_text(encoding="utf-8")
        )

    def collect_bc_dataset(
        self,
        *,
        report_paths: Sequence[str | Path],
        mode: Order3PipelineMode,
        pool_manifest_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> Order3DatasetIOResult:
        manifest = self.load_pool(pool_manifest_path)
        entries_by_hash = {entry.structural_hash: entry for entry in manifest.entries}
        transitions: list[Order3PolicyTransition] = []
        seen: set[str] = set()
        source_hashes: dict[str, str] = {}
        seen_report_hashes: set[str] = set()
        collector_config = Order3TakeoffBCCollectorConfig(
            physical_model_config_path=self.config.pool.robot_model_config_path,
            recurrent_state_dim=self.config.recurrent_hidden_dim,
        )
        for raw_path in _unique_paths(report_paths):
            report_path = Path(raw_path)
            _reject_legacy_p4_3_path(report_path)
            result = RandomMorphologyTakeoffRunnerResult.from_json(
                report_path.read_text(encoding="utf-8")
            )
            report_sha256 = hash_file(report_path)
            if report_sha256 in seen_report_hashes:
                raise SchemaValidationError(
                    "duplicate Order3 BC source report content is not a fresh episode"
                )
            seen_report_hashes.add(report_sha256)
            structural_hash = morphology_structural_hash(result.morphology_graph)
            entry = entries_by_hash.get(structural_hash)
            if entry is None:
                raise SchemaValidationError(
                    "Order3 BC report morphology is not assigned by the pool manifest"
                )
            seen.add(structural_hash)
            collected = collect_order3_takeoff_bc_transitions(
                result.takeoff_result,
                split=entry.split,
                expected_structural_hash=entry.structural_hash,
                episode_id=(
                    f"order3-bc-{entry.structural_hash[:12]}-{report_sha256[:16]}"
                ),
                physical_model=self.physical_model,
                config=collector_config,
            )
            transitions.extend(collected.transitions)
            source_hashes[str(report_path)] = report_sha256
        self._validate_collection_coverage(manifest, seen, mode=mode, stage="BC")
        destination = Path(output_dir or self.config.pipeline.bc_dataset_dir)
        return write_order3_dataset(
            transitions,
            pool_hash=manifest.stable_hash(),
            physical_model_hash=manifest.physical_model_hash,
            config_hash=self.config.config_sha256,
            output_dir=destination,
            shard_size=self.config.pipeline.dataset_shard_size,
            overwrite=overwrite,
            metadata={
                "pipeline_runner_version": ORDER3_PIPELINE_RUNNER_VERSION,
                "pipeline_stage": Order3PipelineStage.COLLECT_BC.value,
                "pipeline_mode": mode.value,
                "pipeline_config_sha256": self.config.config_sha256,
                "pool_manifest_sha256": hash_file(
                    pool_manifest_path or self.config.pipeline.pool_manifest_path
                ),
                "source_report_sha256": source_hashes,
                "full_pool_coverage": mode == Order3PipelineMode.FULL,
            },
        )

    def collect_ppo_dataset(
        self,
        *,
        report_paths: Sequence[str | Path],
        mode: Order3PipelineMode,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        pool_manifest_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> Order3DatasetIOResult:
        self._validate_checkpoint(checkpoint_path, checkpoint_sha256, require_file=True)
        manifest = self.load_pool(pool_manifest_path)
        entries_by_hash = {entry.structural_hash: entry for entry in manifest.entries}
        collector = self.online_collector or _load_order3_online_collector()
        transitions: list[Order3PolicyTransition] = []
        seen: set[str] = set()
        source_bindings: dict[str, dict[str, str]] = {}
        seen_report_hashes: set[str] = set()
        for raw_path in _unique_paths(report_paths):
            report_path = Path(raw_path)
            _reject_legacy_p4_3_path(report_path)
            result = Order3IsaacPolicyRolloutResult.from_json(
                report_path.read_text(encoding="utf-8")
            )
            report_sha256 = hash_file(report_path)
            if report_sha256 in seen_report_hashes:
                raise SchemaValidationError(
                    "duplicate Order3 PPO source report content is not a fresh online episode"
                )
            seen_report_hashes.add(report_sha256)
            report = result.takeoff_result.report
            condition = _condition_from_report(report)
            runtime_rows = report.get("random_morphology_takeoff_runtime_observations")
            if not isinstance(runtime_rows, list) or not runtime_rows:
                raise SchemaValidationError(
                    "Order3 learned rollout report lacks runtime observations"
                )
            graph = MorphologyGraph.from_dict(runtime_rows[0]["morphology_graph"])
            structural_hash = morphology_structural_hash(graph)
            entry = entries_by_hash.get(structural_hash)
            if entry is None:
                raise SchemaValidationError(
                    "Order3 PPO report morphology is not assigned by the pool manifest"
                )
            if result.checkpoint_sha256 != checkpoint_sha256:
                raise SchemaValidationError(
                    "Order3 PPO report checkpoint hash does not match requested behavior checkpoint"
                )
            seen.add(structural_hash)
            collected = _invoke_online_collector(
                collector,
                source=result.takeoff_result,
                split=entry.split,
                expected_structural_hash=entry.structural_hash,
                expected_checkpoint_sha256=checkpoint_sha256,
                behavior_checkpoint_path=checkpoint_path,
                episode_id=(
                    f"order3-online-{entry.structural_hash[:12]}-{report_sha256[:16]}"
                ),
                physical_model=self.physical_model,
            )
            if self.online_collector is None and getattr(
                collected, "metadata", {}
            ).get("behavior_replay_verified") is not True:
                raise SchemaValidationError(
                    "Order3 production PPO collection requires behavior replay verification"
                )
            rows = getattr(collected, "transitions", collected)
            if not isinstance(rows, list) or not rows or not all(
                isinstance(row, Order3PolicyTransition) for row in rows
            ):
                raise SchemaValidationError(
                    "Order3 online collector must return transitions or a result with transitions"
                )
            transitions.extend(rows)
            source_bindings[str(report_path)] = {
                "report_sha256": report_sha256,
                "condition_hash": condition.condition_hash,
                "checkpoint_sha256": checkpoint_sha256,
                "structural_hash": structural_hash,
            }
        self._validate_collection_coverage(manifest, seen, mode=mode, stage="PPO")
        destination = Path(output_dir or self.config.pipeline.ppo_dataset_dir)
        return write_order3_dataset(
            transitions,
            pool_hash=manifest.stable_hash(),
            physical_model_hash=manifest.physical_model_hash,
            config_hash=self.config.config_sha256,
            output_dir=destination,
            shard_size=self.config.pipeline.dataset_shard_size,
            overwrite=overwrite,
            metadata={
                "pipeline_runner_version": ORDER3_PIPELINE_RUNNER_VERSION,
                "pipeline_stage": Order3PipelineStage.COLLECT_PPO.value,
                "pipeline_mode": mode.value,
                "pipeline_config_sha256": self.config.config_sha256,
                "pool_manifest_sha256": hash_file(
                    pool_manifest_path or self.config.pipeline.pool_manifest_path
                ),
                "source_report_bindings": source_bindings,
                "behavior_checkpoint_sha256": checkpoint_sha256,
                "behavior_replay_verified": self.online_collector is None,
                "full_pool_coverage": mode == Order3PipelineMode.FULL,
            },
        )

    def train_bc(
        self,
        *,
        dataset_path: str | Path | None = None,
        output_root: str | Path | None = None,
        git_revision: str | None = None,
    ) -> Order3BCTrainingResult:
        dataset = self._load_bound_dataset(dataset_path or self.config.pipeline.bc_dataset_dir)
        training_config, policy_config = load_order3_pi_l_training_config(
            self.config.config_path
        )
        return train_order3_pi_l_bc(
            dataset_path=dataset.manifest_path,
            physical_model=self.physical_model,
            training_config=training_config,
            policy_config=policy_config,
            config_path=self.config.config_path,
            output_root=output_root,
            git_revision=git_revision or _git_revision(),
        )

    def train_ppo(
        self,
        *,
        dataset_path: str | Path | None = None,
        parent_checkpoint_path: str | Path,
        parent_checkpoint_sha256: str,
        update_index: int,
        output_root: str | Path | None = None,
        git_revision: str | None = None,
    ) -> Order3PPOTrainingResult:
        if not 0 <= update_index < self.config.ppo_orchestration_updates:
            raise SchemaValidationError(
                "Order3 PPO update_index is outside the configured orchestration range"
            )
        dataset = self._load_bound_dataset(dataset_path or self.config.pipeline.ppo_dataset_dir)
        training_config, _ = load_order3_pi_l_training_config(self.config.config_path)
        one_fresh_update = replace(
            training_config,
            seed=training_config.seed + update_index,
            ppo=replace(training_config.ppo, updates=1),
        )
        target_root = output_root or (
            Path(training_config.artifact_root)
            / "training"
            / "ppo"
            / f"update_{update_index:04d}"
        )
        return train_order3_pi_l_ppo(
            dataset_path=dataset.manifest_path,
            # The training API keeps its original parameter names for backward
            # compatibility, but accepts an immediate BC *or PPO* parent.
            parent_bc_checkpoint_path=parent_checkpoint_path,
            parent_bc_checkpoint_sha256=parent_checkpoint_sha256,
            physical_model=self.physical_model,
            training_config=one_fresh_update,
            config_path=self.config.config_path,
            output_root=target_root,
            git_revision=git_revision or _git_revision(),
        )

    def run_ppo_orchestration(
        self,
        *,
        mode: Order3PipelineMode,
        initial_checkpoint_path: str | Path,
        initial_checkpoint_sha256: str,
        start_update_index: int = 0,
        update_count: int | None = None,
        pool_manifest_path: str | Path | None = None,
        graph_paths: Sequence[str | Path] = (),
        conditions: Sequence[Order3RolloutCondition] = (),
        git_revision: str | None = None,
    ) -> Order3PPOOrchestrationResult:
        """Alternate fresh stochastic Isaac collection and exactly one PPO update."""

        count = (
            self.config.ppo_orchestration_updates - start_update_index
            if update_count is None
            else update_count
        )
        if count <= 0 or start_update_index < 0:
            raise SchemaValidationError(
                "Order3 PPO orchestration update range must be positive"
            )
        if start_update_index + count > self.config.ppo_orchestration_updates:
            raise SchemaValidationError(
                "Order3 PPO orchestration exceeds configured YAML update count"
            )
        self._validate_checkpoint(
            initial_checkpoint_path,
            initial_checkpoint_sha256,
            require_file=True,
        )
        current_path = str(initial_checkpoint_path)
        current_hash = initial_checkpoint_sha256
        updates: list[dict[str, Any]] = []
        rollout_conditions = list(conditions) or self.curriculum_conditions()
        for update_index in range(start_update_index, start_update_index + count):
            plan = self.plan_learned_rollouts(
                mode=mode,
                checkpoint_path=current_path,
                checkpoint_sha256=current_hash,
                pool_manifest_path=pool_manifest_path,
                graph_paths=graph_paths,
                real=True,
                stochastic=True,
                conditions=rollout_conditions,
                condition_seed_namespace=f"ppo_update_{update_index:04d}",
            )
            plan_path = (
                Path(self.config.pipeline.artifact_root)
                / "plans"
                / f"ppo_update_{update_index:04d}_rollouts.json"
            )
            execution = self.execute_plan(plan, plan_path=plan_path)
            report_paths = [command.report_path for command in plan.commands]
            dataset_dir = (
                Path(self.config.pipeline.artifact_root)
                / "datasets"
                / "ppo"
                / f"update_{update_index:04d}"
            )
            dataset = self.collect_ppo_dataset(
                report_paths=report_paths,
                mode=mode,
                checkpoint_path=current_path,
                checkpoint_sha256=current_hash,
                pool_manifest_path=pool_manifest_path,
                output_dir=dataset_dir,
                overwrite=False,
            )
            parent_hash = current_hash
            training = self.train_ppo(
                dataset_path=dataset.manifest_path,
                parent_checkpoint_path=current_path,
                parent_checkpoint_sha256=current_hash,
                update_index=update_index,
                git_revision=git_revision,
            )
            current_path = training.checkpoint_path
            current_hash = training.checkpoint_sha256
            updates.append(
                {
                    "update_index": update_index,
                    "behavior_checkpoint_sha256": parent_hash,
                    "rollout_plan_path": str(plan_path),
                    "report_count": len(report_paths),
                    "report_sha256": execution["report_hashes"],
                    "dataset_manifest_path": dataset.manifest_path,
                    "dataset_manifest_sha256": hash_file(dataset.manifest_path),
                    "output_checkpoint_path": current_path,
                    "output_checkpoint_sha256": current_hash,
                    "fresh_online_rollout_update": True,
                    "online_update_count": 1,
                }
            )
        return Order3PPOOrchestrationResult(
            runner_version=ORDER3_PIPELINE_RUNNER_VERSION,
            mode=mode,
            requested_update_count=count,
            completed_update_count=len(updates),
            initial_checkpoint_sha256=initial_checkpoint_sha256,
            final_checkpoint_path=current_path,
            final_checkpoint_sha256=current_hash,
            updates=updates,
        )

    def evaluate_acceptance(
        self,
        *,
        mode: Order3PipelineMode,
        pool_manifest_path: str | Path | None = None,
        dataset_manifest_path: str | Path,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        episodes_path: str | Path,
        artifact_metadata_path: str | Path | None,
        output_path: str | Path | None = None,
    ) -> Order3AcceptanceReport:
        if mode != Order3PipelineMode.FULL:
            raise SchemaValidationError(
                "Order3 smoke mode cannot run or claim the statistical acceptance gate"
            )
        resolved_artifact_metadata_path = artifact_metadata_path
        if resolved_artifact_metadata_path is None:
            resolved_artifact_metadata_path = self.build_acceptance_artifact_metadata(
                pool_manifest_path=pool_manifest_path,
                dataset_manifest_path=dataset_manifest_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                episodes_path=episodes_path,
            )[1]
        canonical_episodes_path = Path(episodes_path).with_name(
            f"{Path(episodes_path).stem}.canonical.json"
        )
        canonical_episodes = _load_evaluation_episodes(Path(episodes_path))
        _atomic_write_text(
            canonical_episodes_path,
            json.dumps(
                [episode.to_dict() for episode in canonical_episodes],
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        for value in (
            pool_manifest_path or self.config.pipeline.pool_manifest_path,
            dataset_manifest_path,
            checkpoint_path,
            episodes_path,
            resolved_artifact_metadata_path,
        ):
            _reject_legacy_p4_3_path(Path(value))
        report = run_order3_acceptance_from_paths(
            pool_manifest_path=(
                pool_manifest_path or self.config.pipeline.pool_manifest_path
            ),
            dataset_manifest_path=dataset_manifest_path,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=checkpoint_sha256,
            episodes_path=canonical_episodes_path,
            artifact_metadata_path=resolved_artifact_metadata_path,
        )
        destination = Path(
            output_path
            or Path(self.config.pipeline.evaluation_path).with_name(
                "acceptance_report.json"
            )
        )
        _reject_legacy_p4_3_path(destination)
        _atomic_write_text(destination, report.to_json(indent=2) + "\n")
        return report

    def build_acceptance_artifact_metadata(
        self,
        *,
        dataset_manifest_path: str | Path,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        episodes_path: str | Path,
        pool_manifest_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[Order3AcceptanceArtifactMetadata, str]:
        pool_path = Path(
            pool_manifest_path or self.config.pipeline.pool_manifest_path
        )
        dataset_path = Path(dataset_manifest_path)
        checkpoint = Path(checkpoint_path)
        episodes_source = Path(episodes_path)
        for path in (pool_path, dataset_path, checkpoint, episodes_source):
            _reject_legacy_p4_3_path(path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Order3 acceptance metadata source does not exist: {path}"
                )
        self._validate_checkpoint(checkpoint, checkpoint_sha256, require_file=True)
        pool = self.load_pool(pool_path)
        dataset = load_order3_dataset(dataset_path)
        if dataset.manifest.pool_hash != pool.stable_hash():
            raise SchemaValidationError(
                "Order3 acceptance dataset pool hash does not match persisted pool"
            )
        loaded = load_order3_policy_checkpoint(
            checkpoint,
            expected_sha256=checkpoint_sha256,
        )
        episodes = _load_evaluation_episodes(episodes_source)
        _validate_evaluation_episode_bindings(
            episodes,
            checkpoint_sha256=checkpoint_sha256,
        )
        metadata = loaded.metadata
        if metadata.pool_hash != pool.stable_hash():
            raise SchemaValidationError(
                "Order3 acceptance checkpoint pool hash mismatch"
            )
        if metadata.dataset_hash != hash_file(dataset.manifest_path):
            raise SchemaValidationError(
                "Order3 acceptance checkpoint dataset hash mismatch"
            )
        artifact = Order3AcceptanceArtifactMetadata(
            artifact_version=ORDER3_ACCEPTANCE_ARTIFACT_VERSION,
            evaluation_scope_version=ORDER3_FREE_FLIGHT_VERSION,
            evaluation_source="real_isaac_paired_learned_and_deterministic_v2",
            checkpoint_sha256=checkpoint_sha256,
            dataset_manifest_sha256=hash_file(dataset.manifest_path),
            policy_family=metadata.policy_family,
            policy_contract_version=metadata.policy_contract_version,
            architecture_version=metadata.architecture_version,
            tensorizer_version=metadata.tensorizer_version,
            encoder_version=metadata.encoder_version,
            graph_encoder_used=True,
            recurrent_gru_used=True,
            actor_uses_privileged_wrench=metadata.actor_uses_privileged_wrench,
            deterministic_fallback_available=True,
            pool_hash=pool.stable_hash(),
            evaluation_episode_set_hash=stable_hash(
                [episode.to_dict() for episode in episodes]
            ),
            rollout_condition_version=ORDER3_ROLLOUT_CONDITION_VERSION,
            raw_report_hashes_bound=True,
            paired_deterministic_baseline=True,
            required_task_modes=["hover", "takeoff", "waypoint"],
            object_task_claim=False,
            contact_task_claim=False,
            p4_full_completion_claim=False,
        )

        destination = Path(
            output_path
            or Path(self.config.pipeline.evaluation_path).with_name(
                "artifact_metadata.json"
            )
        )
        _atomic_write_text(destination, artifact.to_json(indent=2) + "\n")
        persisted = Order3AcceptanceArtifactMetadata.from_json(
            destination.read_text(encoding="utf-8")
        )
        if persisted.stable_hash() != artifact.stable_hash():
            raise RuntimeError("Order3 acceptance metadata atomic roundtrip mismatch")
        return persisted, str(destination)

    def _condition_for_episode(
        self,
        condition: Order3RolloutCondition,
        *,
        structural_hash: str,
        seed_namespace: str,
    ) -> Order3RolloutCondition:
        """Derive a reproducible fresh condition for one graph/update pair."""

        seed = int(
            stable_hash(
                {
                    "base_condition_hash": condition.condition_hash,
                    "structural_hash": structural_hash,
                    "seed_namespace": seed_namespace,
                }
            )[:16],
            16,
        ) % (2**31)
        configured = next(
            (
                stage
                for stage in self.config.curriculum.stages
                if stage.name == condition.stage_id
            ),
            None,
        )
        if configured is not None:
            return _condition_from_curriculum_stage(configured, seed=seed)
        return _reseed_order3_condition(condition, seed=seed)

    def build_evaluation_episodes(
        self,
        *,
        mode: Order3PipelineMode,
        learned_report_paths: Sequence[str | Path],
        baseline_report_paths: Sequence[str | Path],
        checkpoint_sha256: str,
        pool_manifest_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[list[Order3EvaluationEpisode], str]:
        if not _is_sha256(checkpoint_sha256):
            raise SchemaValidationError(
                "Order3 evaluation episode checkpoint sha256 is invalid"
        )
        pool = self.load_pool(pool_manifest_path)
        pool_entries = {entry.structural_hash: entry for entry in pool.entries}
        _, takeoff_config = load_random_morphology_takeoff_runner_config(
            self.config.pipeline.takeoff_config_path
        )
        physical_model = self.physical_model
        expected_backend_config_hash = load_isaac_lab_backend_config(
            takeoff_config.backend_config_path
        ).stable_hash()
        expected_physical_model_hash = physical_model.stable_hash()
        expected_collision_geometry_hash = collision_geometry_content_hash(
            physical_model,
            mesh_search_dirs=takeoff_config.mesh_search_dirs,
        )
        learned = _load_raw_evaluation_reports(
            learned_report_paths,
            expected_policy="learned",
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_backend_config_hash=expected_backend_config_hash,
            expected_physical_model_hash=expected_physical_model_hash,
            expected_collision_geometry_hash=expected_collision_geometry_hash,
        )
        baseline = _load_raw_evaluation_reports(
            baseline_report_paths,
            expected_policy="baseline",
            expected_checkpoint_sha256=None,
            expected_backend_config_hash=expected_backend_config_hash,
            expected_physical_model_hash=expected_physical_model_hash,
            expected_collision_geometry_hash=expected_collision_geometry_hash,
        )
        if set(learned) != set(baseline):
            raise SchemaValidationError(
                "Order3 learned/baseline evaluation reports must pair by "
                "structural hash and condition hash"
            )
        episodes: list[Order3EvaluationEpisode] = []
        for key in sorted(learned):
            learned_path, learned_hash, learned_report, condition = learned[key]
            baseline_path, baseline_hash, baseline_report, baseline_condition = baseline[key]
            if condition.to_canonical_json() != baseline_condition.to_canonical_json():
                raise SchemaValidationError(
                    "Order3 learned/baseline evaluation condition payload mismatch"
                )
            structural_hash, condition_hash = key
            entry = pool_entries.get(structural_hash)
            split = DatasetSplit.HELD_OUT if entry is None else entry.split
            module_count = int(
                learned_report.get("random_morphology_takeoff_module_count", 0)
            )
            if entry is not None and module_count != entry.module_count:
                raise SchemaValidationError(
                    "Order3 evaluation report module count differs from pool"
                )
            learned_metrics = Order3TerminalMetrics.from_dict(
                learned_report.get("order3_terminal_metrics")
            )
            baseline_metrics = Order3TerminalMetrics.from_dict(
                baseline_report.get("order3_terminal_metrics")
            )
            episode = Order3EvaluationEpisode(
                episode_id=(
                    "order3-evaluation-"
                    + stable_hash(
                        {
                            "structural_hash": structural_hash,
                            "condition_hash": condition_hash,
                            "checkpoint_sha256": checkpoint_sha256,
                            "learned_report_sha256": learned_hash,
                            "baseline_report_sha256": baseline_hash,
                        }
                    )[:20]
                ),
                structural_hash=structural_hash,
                module_count=module_count,
                split=split,
                success=bool(learned_report["order3_free_flight_success"]),
                tracking_cost=float(
                    learned_report["order3_free_flight_tracking_cost"]
                ),
                deterministic_baseline_tracking_cost=float(
                    baseline_report["order3_free_flight_tracking_cost"]
                ),
                randomized=_condition_is_randomized(condition),
                fallback_used=bool(learned_report["order3_fallback_used"]),
                qp_infeasible=bool(learned_report["order3_qp_infeasible"]),
                hard_collision=bool(learned_report["order3_hard_collision"]),
                non_finite_state=bool(learned_report["order3_non_finite_state"]),
                unsupported_actuator=bool(
                    learned_report["order3_unsupported_actuator"]
                ),
                task_mode=Order3TaskMode(condition.task_mode),
                terminal_metrics=learned_metrics,
                deterministic_baseline_terminal_metrics=baseline_metrics,
                condition_hash=condition_hash,
                condition_seed=condition.seed,
                checkpoint_sha256=checkpoint_sha256,
                learned_report_path=str(learned_path),
                learned_report_sha256=learned_hash,
                deterministic_baseline_report_path=str(baseline_path),
                deterministic_baseline_report_sha256=baseline_hash,
                fallback_reason=learned_report.get("order3_fallback_reason"),
                isaac_backed=True,
            )
            episodes.append(episode)
        if mode == Order3PipelineMode.FULL:
            _validate_full_evaluation_episode_matrix(episodes, pool)
        destination = Path(output_path or self.config.pipeline.evaluation_path)
        _atomic_write_text(
            destination,
            json.dumps(
                [episode.to_dict() for episode in episodes],
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        persisted = _load_evaluation_episodes(destination)
        if [episode.to_dict() for episode in persisted] != [
            episode.to_dict() for episode in episodes
        ]:
            raise RuntimeError("Order3 evaluation episode atomic roundtrip mismatch")
        return persisted, str(destination)

    def _select_entries(
        self,
        manifest: Order3MorphologyPoolManifest,
        *,
        mode: Order3PipelineMode,
        graph_paths: Sequence[str | Path],
    ) -> list[tuple[Order3MorphologyPoolEntry, Path]]:
        entries_by_hash = {entry.structural_hash: entry for entry in manifest.entries}
        if mode == Order3PipelineMode.FULL:
            if graph_paths:
                raise SchemaValidationError(
                    "Order3 full mode selects the manifest pool; explicit graph paths are smoke-only"
                )
            selected: list[tuple[Order3MorphologyPoolEntry, Path]] = []
            for entry in manifest.entries:
                graph_path = self._graph_path(entry)
                _atomic_write_text(graph_path, entry.morphology_graph.to_json(indent=2) + "\n")
                selected.append((entry, graph_path))
            return selected
        if not graph_paths:
            raise SchemaValidationError(
                "Order3 smoke mode requires at least one explicit graph path"
            )
        selected = []
        seen: set[str] = set()
        for raw_path in graph_paths:
            graph_path = Path(raw_path)
            graph = _load_graph(graph_path)
            structural_hash = morphology_structural_hash(graph)
            entry = entries_by_hash.get(structural_hash)
            if entry is None:
                raise SchemaValidationError(
                    "Order3 smoke graph is not assigned by the morphology pool"
                )
            if structural_hash in seen:
                raise SchemaValidationError("duplicate Order3 smoke morphology graph")
            seen.add(structural_hash)
            selected.append((entry, graph_path))
        return selected

    def _select_evaluation_graphs(
        self,
        manifest: Order3MorphologyPoolManifest,
        *,
        mode: Order3PipelineMode,
        graph_paths: Sequence[str | Path],
        ood_graph_paths: Sequence[str | Path],
    ) -> list[_EvaluationGraph]:
        entries_by_hash = {entry.structural_hash: entry for entry in manifest.entries}
        selected: list[_EvaluationGraph] = []
        if mode == Order3PipelineMode.FULL:
            if graph_paths:
                raise SchemaValidationError(
                    "Order3 full evaluation selects held-out pool graphs; "
                    "use ood_graph_paths only for additional out-of-distribution evidence"
                )
            for entry in manifest.entries:
                if entry.split != DatasetSplit.HELD_OUT:
                    continue
                graph_path = self._graph_path(entry)
                _atomic_write_text(
                    graph_path, entry.morphology_graph.to_json(indent=2) + "\n"
                )
                selected.append(
                    _EvaluationGraph(
                        split=entry.split,
                        module_count=entry.module_count,
                        structural_hash=entry.structural_hash,
                        graph_path=graph_path,
                    )
                )
        else:
            if not graph_paths:
                raise SchemaValidationError(
                    "Order3 smoke evaluation requires explicit graph paths"
                )
            for raw_path in graph_paths:
                path = Path(raw_path)
                graph = _load_graph(path)
                structural_hash = morphology_structural_hash(graph)
                entry = entries_by_hash.get(structural_hash)
                if entry is None:
                    raise SchemaValidationError(
                        "Order3 smoke ID evaluation graph is not assigned by the pool; "
                        "pass it as an OOD graph instead"
                    )
                selected.append(
                    _EvaluationGraph(
                        split=entry.split,
                        module_count=len(graph.modules),
                        structural_hash=structural_hash,
                        graph_path=path,
                    )
                )
        for raw_path in ood_graph_paths:
            path = Path(raw_path)
            graph = _load_graph(path)
            structural_hash = morphology_structural_hash(graph)
            if structural_hash in entries_by_hash:
                raise SchemaValidationError(
                    "Order3 OOD evaluation graph is already in the morphology pool"
                )
            selected.append(
                _EvaluationGraph(
                    split=DatasetSplit.HELD_OUT,
                    module_count=len(graph.modules),
                    structural_hash=structural_hash,
                    graph_path=path,
                )
            )
        identities = [
            (item.structural_hash, str(item.graph_path)) for item in selected
        ]
        if len(identities) != len(set(identities)):
            raise SchemaValidationError("duplicate Order3 evaluation graph")
        return selected

    def _command_plan(
        self,
        *,
        stage: Order3PipelineStage,
        mode: Order3PipelineMode,
        manifest: Order3MorphologyPoolManifest,
        commands: list[Order3PipelineCommand],
        execution_requested: bool,
    ) -> Order3PipelineCommandPlan:
        hashes = {command.structural_hash for command in commands}
        all_hashes = {entry.structural_hash for entry in manifest.entries}
        return Order3PipelineCommandPlan(
            runner_version=ORDER3_PIPELINE_RUNNER_VERSION,
            stage=stage,
            mode=mode,
            config_sha256=self.config.config_sha256,
            pool_hash=manifest.stable_hash(),
            commands=commands,
            condition_hashes=sorted(
                {
                    command.condition_hash
                    for command in commands
                    if command.condition_hash is not None
                }
            ),
            full_pool_coverage=(mode == Order3PipelineMode.FULL and hashes == all_hashes),
            execution_requested=execution_requested,
        )

    def _graph_path(self, entry: Order3MorphologyPoolEntry) -> Path:
        return (
            Path(self.config.pipeline.artifact_root)
            / "graphs"
            / entry.split.value
            / f"n{entry.module_count}_{entry.structural_hash}.json"
        )

    def _report_path(
        self,
        kind: str,
        entry: Order3MorphologyPoolEntry,
        *,
        condition: Order3RolloutCondition | None = None,
        behavior_checkpoint_sha256: str | None = None,
    ) -> Path:
        suffix = "" if condition is None else f"_{condition.condition_hash}"
        if behavior_checkpoint_sha256 is not None:
            if not _is_sha256(behavior_checkpoint_sha256):
                raise SchemaValidationError(
                    "Order3 report behavior checkpoint hash must be sha256"
                )
            suffix += f"_{behavior_checkpoint_sha256}"
        return (
            Path(self.config.pipeline.report_dir)
            / kind
            / entry.split.value
            / f"n{entry.module_count}_{entry.structural_hash}{suffix}.json"
        )

    def _evaluation_report_path(
        self,
        kind: str,
        graph: _EvaluationGraph,
        condition: Order3RolloutCondition,
        *,
        behavior_checkpoint_sha256: str | None = None,
    ) -> Path:
        checkpoint_suffix = ""
        if behavior_checkpoint_sha256 is not None:
            if kind != "learned" or not _is_sha256(behavior_checkpoint_sha256):
                raise SchemaValidationError(
                    "Order3 learned evaluation report checkpoint hash is invalid"
                )
            checkpoint_suffix = f"_{behavior_checkpoint_sha256}"
        return (
            Path(self.config.pipeline.report_dir)
            / "evaluation"
            / kind
            / graph.split.value
            / (
                f"n{graph.module_count}_{graph.structural_hash}_"
                f"{condition.condition_hash}{checkpoint_suffix}.json"
            )
        )

    def _validate_collection_coverage(
        self,
        manifest: Order3MorphologyPoolManifest,
        seen: set[str],
        *,
        mode: Order3PipelineMode,
        stage: str,
    ) -> None:
        expected = {entry.structural_hash for entry in manifest.entries}
        if mode == Order3PipelineMode.FULL and seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise SchemaValidationError(
                f"Order3 {stage} full collection must cover the exact pool; "
                f"missing={missing}, extra={extra}"
            )
        split_by_hash = {entry.structural_hash: entry.split for entry in manifest.entries}
        present_splits = {split_by_hash[value] for value in seen}
        if present_splits != set(DatasetSplit):
            raise SchemaValidationError(
                f"Order3 {stage} collection requires train/validation/held_out evidence"
            )

    def _load_bound_dataset(self, path: str | Path) -> Order3DatasetIOResult:
        dataset = load_order3_dataset(path)
        pool = self.load_pool()
        if dataset.manifest.pool_hash != pool.stable_hash():
            raise SchemaValidationError("Order3 dataset morphology-pool hash mismatch")
        if dataset.manifest.physical_model_hash != pool.physical_model_hash:
            raise SchemaValidationError("Order3 dataset PhysicalModel hash mismatch")
        return dataset

    @staticmethod
    def _validate_checkpoint(
        path: str | Path,
        expected_sha256: str,
        *,
        require_file: bool,
    ) -> None:
        checkpoint = Path(path)
        _reject_legacy_p4_3_path(checkpoint)
        if not _is_sha256(expected_sha256):
            raise SchemaValidationError("Order3 checkpoint expected sha256 is invalid")
        if require_file and not checkpoint.is_file():
            raise FileNotFoundError(f"Order3 checkpoint does not exist: {checkpoint}")
        if checkpoint.is_file() and hash_file(checkpoint) != expected_sha256:
            raise SchemaValidationError("Order3 checkpoint sha256 mismatch")


def _invoke_online_collector(
    collector: OnlineCollector,
    **kwargs: Any,
) -> Any:
    """Typed seam used by unit tests; production resolves the strict collector."""

    return collector(**kwargs)


def _load_order3_online_collector() -> OnlineCollector:
    try:
        from amsrr.training.order3_online_collector import (
            collect_order3_online_transitions,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Order3 online PPO collector is unavailable; deterministic takeoff reports "
            "must never be substituted for learned online traces"
        ) from exc
    return collect_order3_online_transitions


def _load_graph(path: str | Path) -> MorphologyGraph:
    source = Path(path)
    _reject_legacy_p4_3_path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Order3 morphology graph does not exist: {source}")
    return MorphologyGraph.from_json(source.read_text(encoding="utf-8"))


def _condition_from_curriculum_stage(
    stage: Order3ConfiguredCurriculumStage,
    *,
    seed: int,
) -> Order3RolloutCondition:
    task_mode = (
        "takeoff"
        if stage.floor_takeoff
        else (
            "waypoint"
            if stage.translation_waypoints or stage.attitude_waypoints
            else "hover"
        )
    )
    direction = _signed_unit(seed, "direction")
    initial_scale = (
        0.0
        if stage.floor_takeoff
        else float(stage.initial_state_randomization_scale)
    )
    model_scale = float(stage.model_randomization_scale)
    disturbance_scale = float(stage.disturbance_scale)
    return build_order3_rollout_condition(
        stage_id=stage.name,
        task_mode=task_mode,
        seed=seed,
        initial_position_offset_world=(
            0.10 * initial_scale * _signed_unit(seed, "initial_position_x"),
            0.10 * initial_scale * _signed_unit(seed, "initial_position_y"),
            0.10 * initial_scale * _signed_unit(seed, "initial_position_z"),
        ),
        initial_orientation_rpy_rad=(
            0.10 * initial_scale * _signed_unit(seed, "initial_roll"),
            0.10 * initial_scale * _signed_unit(seed, "initial_pitch"),
            0.10 * initial_scale * _signed_unit(seed, "initial_yaw"),
        ),
        initial_linear_velocity_world=(
            0.10 * initial_scale * _signed_unit(seed, "initial_velocity_x"),
            0.10 * initial_scale * _signed_unit(seed, "initial_velocity_y"),
            0.10 * initial_scale * _signed_unit(seed, "initial_velocity_z"),
        ),
        initial_angular_velocity_body=(
            0.10 * initial_scale * _signed_unit(seed, "initial_omega_x"),
            0.10 * initial_scale * _signed_unit(seed, "initial_omega_y"),
            0.10 * initial_scale * _signed_unit(seed, "initial_omega_z"),
        ),
        waypoint_position_offset_world=(
            0.25 * direction if stage.translation_waypoints else 0.0,
            0.125 * _signed_unit(seed, "waypoint_y")
            if stage.translation_waypoints
            else 0.0,
            0.10 if stage.translation_waypoints else 0.0,
        ),
        waypoint_orientation_rpy_rad=(
            0.0,
            0.0,
            0.25 * _signed_unit(seed, "waypoint_yaw")
            if stage.attitude_waypoints
            else 0.0,
        ),
        waypoint_ramp_s=1.0,
        hold_s=1.0,
        external_wrench_body=(
            disturbance_scale * direction,
            0.25 * disturbance_scale * _signed_unit(seed, "wrench_y"),
            0.0,
            0.0,
            0.0,
            0.10 * disturbance_scale * _signed_unit(seed, "wrench_yaw"),
        ),
        disturbance_start_s=3.0,
        disturbance_duration_s=1.0 if disturbance_scale > 0.0 else 0.0,
        mass_scale=1.0 + model_scale * _signed_unit(seed, "mass"),
        inertia_scale=1.0 + model_scale * _signed_unit(seed, "inertia"),
        thrust_scale=1.0 + model_scale * _signed_unit(seed, "thrust"),
    )


def _reseed_order3_condition(
    condition: Order3RolloutCondition,
    *,
    seed: int,
) -> Order3RolloutCondition:
    """Clone an explicit condition with a fresh, hash-bound episode seed."""

    return build_order3_rollout_condition(
        stage_id=condition.stage_id,
        task_mode=condition.task_mode,
        seed=seed,
        initial_position_offset_world=condition.initial_position_offset_world,
        initial_orientation_rpy_rad=condition.initial_orientation_rpy_rad,
        initial_linear_velocity_world=condition.initial_linear_velocity_world,
        initial_angular_velocity_body=condition.initial_angular_velocity_body,
        waypoint_position_offset_world=condition.waypoint_position_offset_world,
        waypoint_orientation_rpy_rad=condition.waypoint_orientation_rpy_rad,
        waypoint_ramp_s=condition.waypoint_ramp_s,
        hold_s=condition.hold_s,
        external_wrench_body=condition.external_wrench_body,
        disturbance_start_s=condition.disturbance_start_s,
        disturbance_duration_s=condition.disturbance_duration_s,
        mass_scale=condition.mass_scale,
        inertia_scale=condition.inertia_scale,
        thrust_scale=condition.thrust_scale,
    )


def _validate_unique_conditions(
    conditions: Sequence[Order3RolloutCondition],
) -> None:
    if not conditions:
        raise SchemaValidationError("Order3 rollout conditions must not be empty")
    hashes = [condition.condition_hash for condition in conditions]
    if len(hashes) != len(set(hashes)):
        raise SchemaValidationError("Order3 rollout condition hashes must be unique")


def _validate_order3_visualization_options(
    *,
    real: bool,
    viewer: str | None,
    realtime_playback: bool,
    keep_open_after_rollout_s: float,
) -> None:
    if viewer not in {None, "kit"}:
        raise SchemaValidationError("Order3 viewer must be None or 'kit'")
    if not math.isfinite(keep_open_after_rollout_s) or keep_open_after_rollout_s < 0.0:
        raise SchemaValidationError(
            "Order3 keep_open_after_rollout_s must be finite and non-negative"
        )
    visualization_requested = (
        viewer is not None or realtime_playback or keep_open_after_rollout_s > 0.0
    )
    if visualization_requested and not real:
        raise SchemaValidationError(
            "Order3 visualization requires real Isaac execution"
        )
    if viewer is None and (realtime_playback or keep_open_after_rollout_s > 0.0):
        raise SchemaValidationError(
            "Order3 realtime playback and post-rollout hold require viewer='kit'"
        )


def _validate_report_condition(
    report: Mapping[str, Any],
    condition: Order3RolloutCondition,
) -> None:
    if report.get("order3_rollout_condition_hash") != condition.condition_hash:
        raise SchemaValidationError(
            "Order3 rollout report condition hash does not match its planned condition"
        )
    raw_condition = report.get("order3_rollout_condition")
    try:
        reported = (
            Order3RolloutCondition.from_json(raw_condition)
            if isinstance(raw_condition, str)
            else Order3RolloutCondition.from_dict(raw_condition)
        )
    except (SchemaValidationError, TypeError) as exc:
        raise SchemaValidationError(
            "Order3 rollout report condition payload is missing or invalid"
        ) from exc
    if reported.to_canonical_json() != condition.to_canonical_json():
        raise SchemaValidationError(
            "Order3 rollout report condition payload differs from its planned condition"
        )


def _condition_from_report(report: Mapping[str, Any]) -> Order3RolloutCondition:
    raw = report.get("order3_rollout_condition")
    try:
        condition = (
            Order3RolloutCondition.from_json(raw)
            if isinstance(raw, str)
            else Order3RolloutCondition.from_dict(raw)
        )
    except (SchemaValidationError, TypeError) as exc:
        raise SchemaValidationError(
            "Order3 rollout report lacks a valid hash-bound rollout condition"
        ) from exc
    _validate_report_condition(report, condition)
    return condition


def _load_evaluation_episodes(path: Path) -> list[Order3EvaluationEpisode]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("episodes")
    if not isinstance(payload, list) or not payload:
        raise SchemaValidationError(
            "Order3 evaluation episode artifact must contain a non-empty list"
        )
    return [Order3EvaluationEpisode.from_dict(value) for value in payload]


def _validate_evaluation_episode_bindings(
    episodes: Sequence[Order3EvaluationEpisode],
    *,
    checkpoint_sha256: str,
) -> None:
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise SchemaValidationError("Order3 evaluation episode ids must be unique")
    for episode in episodes:
        if episode.checkpoint_sha256 != checkpoint_sha256:
            raise SchemaValidationError(
                "Order3 evaluation episode checkpoint binding mismatch"
            )
        if not episode.isaac_backed:
            raise SchemaValidationError(
                "Order3 acceptance metadata requires real-Isaac episodes"
            )
        if episode.condition_hash is None or episode.condition_seed is None:
            raise SchemaValidationError(
                "Order3 evaluation episode condition binding is incomplete"
            )
        report_bindings = (
            (episode.learned_report_path, episode.learned_report_sha256),
            (
                episode.deterministic_baseline_report_path,
                episode.deterministic_baseline_report_sha256,
            ),
        )
        for raw_path, expected_hash in report_bindings:
            if raw_path is None or expected_hash is None:
                raise SchemaValidationError(
                    "Order3 evaluation episode raw report binding is incomplete"
                )
            path = Path(raw_path)
            _reject_legacy_p4_3_path(path)
            if not path.is_file() or hash_file(path) != expected_hash:
                raise SchemaValidationError(
                    "Order3 evaluation episode raw report hash mismatch"
                )


def _load_raw_evaluation_reports(
    paths: Sequence[str | Path],
    *,
    expected_policy: str,
    expected_checkpoint_sha256: str | None,
    expected_backend_config_hash: str,
    expected_physical_model_hash: str,
    expected_collision_geometry_hash: str,
) -> dict[
    tuple[str, str],
    tuple[Path, str, dict[str, Any], Order3RolloutCondition],
]:
    if expected_policy not in {"learned", "baseline"}:
        raise ValueError("Order3 expected_policy must be learned or baseline")
    output: dict[
        tuple[str, str],
        tuple[Path, str, dict[str, Any], Order3RolloutCondition],
    ] = {}
    for raw_path in _unique_paths(paths):
        path = Path(raw_path)
        _reject_legacy_p4_3_path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise SchemaValidationError("Order3 evaluation raw report must be a JSON object")
        if report.get("isaac_backed") is not True:
            raise SchemaValidationError("Order3 evaluation report must be real-Isaac backed")
        if report.get("order3_report_validation_failures", []) != []:
            raise SchemaValidationError(
                "Order3 evaluation report contains validation failures"
            )
        condition = _condition_from_report(report)
        condition_failures = order3_condition_report_failures(
            report,
            expected_condition=condition,
        )
        if condition_failures:
            raise SchemaValidationError(
                "Order3 evaluation condition realization is invalid: "
                + ",".join(condition_failures)
            )
        provenance_failures = order3_provenance_report_failures(
            report,
            expected_backend_config_hash=expected_backend_config_hash,
            expected_physical_model_hash=expected_physical_model_hash,
            expected_collision_geometry_hash=expected_collision_geometry_hash,
        )
        if provenance_failures:
            raise SchemaValidationError(
                "Order3 evaluation simulator provenance is invalid: "
                + ",".join(provenance_failures)
            )
        structural_hash = report.get("order3_structural_hash")
        if not isinstance(structural_hash, str) or not _is_sha256(structural_hash):
            raise SchemaValidationError(
                "Order3 evaluation report structural hash is invalid"
            )
        if report.get("order3_task_mode") != condition.task_mode:
            raise SchemaValidationError(
                "Order3 evaluation report task mode differs from condition"
            )
        if expected_policy == "learned":
            if report.get("order3_pi_l_rollout") is not True:
                raise SchemaValidationError(
                    "Order3 learned evaluation report lacks learned-policy marker"
                )
            if (
                report.get("order3_pi_l_checkpoint_sha256")
                != expected_checkpoint_sha256
            ):
                raise SchemaValidationError(
                    "Order3 learned evaluation report checkpoint mismatch"
                )
        elif report.get("order3_deterministic_baseline_rollout") is not True:
            raise SchemaValidationError(
                "Order3 baseline evaluation report lacks deterministic marker"
            )
        key = (structural_hash, condition.condition_hash)
        if key in output:
            raise SchemaValidationError(
                "duplicate Order3 evaluation report structural/condition pair"
            )
        output[key] = (path, hash_file(path), report, condition)
    return output


def _condition_is_randomized(condition: Order3RolloutCondition) -> bool:
    return any(
        abs(float(value)) > 1.0e-12
        for value in (
            *condition.initial_position_offset_world,
            *condition.initial_orientation_rpy_rad,
            *condition.initial_linear_velocity_world,
            *condition.initial_angular_velocity_body,
            *condition.external_wrench_body,
            condition.mass_scale - 1.0,
            condition.inertia_scale - 1.0,
            condition.thrust_scale - 1.0,
        )
    )


def _validate_full_evaluation_episode_matrix(
    episodes: Sequence[Order3EvaluationEpisode],
    pool: Order3MorphologyPoolManifest,
) -> None:
    held_hashes = {
        entry.structural_hash
        for entry in pool.entries
        if entry.split == DatasetSplit.HELD_OUT
    }
    all_pool_hashes = {entry.structural_hash for entry in pool.entries}
    required_cells = {
        (task_mode, randomized)
        for task_mode in Order3TaskMode
        for randomized in (False, True)
    }
    for structural_hash in held_hashes:
        cells = {
            (episode.task_mode, episode.randomized)
            for episode in episodes
            if episode.structural_hash == structural_hash
        }
        if cells != required_cells:
            raise SchemaValidationError(
                "Order3 full evaluation lacks the held-out task-mode/randomization matrix"
            )
    unexpected_id = [
        episode
        for episode in episodes
        if episode.structural_hash in all_pool_hashes
        and episode.structural_hash not in held_hashes
    ]
    if unexpected_id:
        raise SchemaValidationError(
            "Order3 full evaluation ID episodes must use held-out morphologies"
        )
    ood = [
        episode
        for episode in episodes
        if episode.structural_hash not in all_pool_hashes
    ]
    if not ood or any(
        not episode.fallback_used or episode.fallback_reason != "structural_hash_ood"
        for episode in ood
    ):
        raise SchemaValidationError(
            "Order3 full evaluation requires OOD structural_hash_ood fallback evidence"
        )


def _signed_unit(seed: int, label: str) -> float:
    digest = stable_hash({"seed": seed, "label": label})
    numerator = int(digest[:16], 16)
    unit = numerator / float(0xFFFFFFFFFFFFFFFF)
    return 2.0 * unit - 1.0


def _unique_paths(paths: Iterable[str | Path]) -> list[str]:
    values = [str(Path(value)) for value in paths]
    if not values:
        raise SchemaValidationError("Order3 collection requires report paths")
    if len(values) != len(set(values)):
        raise SchemaValidationError("Order3 collection report paths must be unique")
    return values


def _staging_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.order3-staging")


def _command_argument(argv: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise SchemaValidationError(
            f"Order3 pipeline command must contain exactly one {flag} value"
        )
    return argv[indices[0] + 1]


def _execute_command(argv: Sequence[str], timeout_s: float) -> None:
    completed = subprocess.run(
        list(argv),
        cwd=_REPOSITORY_ROOT,
        check=False,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Order3 pipeline command failed with exit code {completed.returncode}: {list(argv)!r}"
        )


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = completed.stdout.strip()
    return value or "unknown"


def _atomic_write_text(path: Path, payload: str) -> None:
    _reject_legacy_p4_3_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
    resolved = path.expanduser().resolve(strict=False)
    if resolved == _LEGACY_P4_3_ROOT or _LEGACY_P4_3_ROOT in resolved.parents:
        raise SchemaValidationError(
            "Order3 pipeline must not read from or write to legacy artifacts/p4_3"
        )


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )
