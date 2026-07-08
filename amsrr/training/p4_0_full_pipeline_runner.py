from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from amsrr.assembly import AssemblyRunner, AssemblyRunnerConfig, SimplifiedAssemblyExecutor
from amsrr.controllers import ControllerContext
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.morphology.grasp_carry_designs import GraspCarryMorphologyVariant
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignPolicyConfig, P2DesignSelection
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext, select_active_knot
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerCommand, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.simplified_grasp_carry_env import SimplifiedGraspCarryEnv, SimplifiedGraspCarryEnvConfig
from amsrr.training.p2_design_distribution import (
    P2DesignDistributionConfig,
    P2DesignTaskSample,
    P2GraspCarryDesignDistribution,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


P4_0_FULL_PIPELINE_RUNNER_VERSION = "p4_0_full_pipeline_runner_v1"
P4_0_SIMPLIFIED_BACKEND_NOTE = (
    "P4.0 metrics are simplified backend indicators and are not Isaac-backed physical success rates."
)


@dataclass
class P4_0FullPipelineRunnerConfig(SchemaBase):
    episode_count: int = 1000
    seed: int = 0
    source_hash: str = "unknown"
    runner_version: str = P4_0_FULL_PIPELINE_RUNNER_VERSION
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    archive_success_only: bool = False
    max_retries_per_step: int = 1
    simulator_version: str = "simplified_grasp_carry_env_v1"

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P4_0FullPipelineRunnerConfig.episode_count must be positive")
        if self.max_retries_per_step < 0:
            raise SchemaValidationError("P4_0FullPipelineRunnerConfig.max_retries_per_step must be non-negative")
        require_non_empty(self.source_hash, "P4_0FullPipelineRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P4_0FullPipelineRunnerConfig.runner_version")
        require_non_empty(self.robot_model_config_path, "P4_0FullPipelineRunnerConfig.robot_model_config_path")
        require_non_empty(self.simulator_version, "P4_0FullPipelineRunnerConfig.simulator_version")


@dataclass
class P4_0FullPipelineRunnerResult(SchemaBase):
    episode_count: int
    success_count: int
    failure_count: int
    crash_count: int
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P4_0FullPipelineRunnerResult.episode_count must be positive")
        for name in ("success_count", "failure_count", "crash_count"):
            if getattr(self, name) < 0:
                raise SchemaValidationError(f"P4_0FullPipelineRunnerResult.{name} must be non-negative")


@dataclass
class _RolloutRecord:
    policy_commands: list[PolicyCommand] = field(default_factory=list)
    controller_commands: list[ControllerCommand] = field(default_factory=list)
    runtime_observations: list[RuntimeObservation] = field(default_factory=list)
    rewards: list[dict[str, float]] = field(default_factory=list)
    crashed: bool = False
    failure_reason: str | None = None


def load_p4_0_full_pipeline_runner_config(
    path: str | Path,
) -> tuple[
    P4_0FullPipelineRunnerConfig,
    P2DesignDistributionConfig,
    P2DesignPolicyConfig,
    SimplifiedGraspCarryEnvConfig,
]:
    data = load_config(path)
    return (
        P4_0FullPipelineRunnerConfig.from_dict(data.get("runner", {})),
        P2DesignDistributionConfig.from_dict(data.get("distribution", {})),
        _p2_design_policy_config_from_dict(data.get("policy", {})),
        SimplifiedGraspCarryEnvConfig(**data.get("env", {})),
    )


class P4_0FullPipelineRunner:
    """Run P4.0 simplified full-pipeline wiring over P2/P3 outputs."""

    def __init__(
        self,
        base_task_spec: TaskSpec,
        *,
        runner_config: P4_0FullPipelineRunnerConfig | None = None,
        distribution_config: P2DesignDistributionConfig | None = None,
        policy_config: P2DesignPolicyConfig | None = None,
        env_config: SimplifiedGraspCarryEnvConfig | None = None,
        design_policy: P2DesignPolicy | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> None:
        self.base_task_spec = base_task_spec
        self.runner_config = runner_config or P4_0FullPipelineRunnerConfig()
        self.distribution_config = distribution_config or P2DesignDistributionConfig()
        self.policy_config = policy_config or (
            design_policy.config if design_policy is not None else P2DesignPolicyConfig()
        )
        self.env_config = env_config or SimplifiedGraspCarryEnvConfig(
            robot_model_config_path=self.runner_config.robot_model_config_path
        )
        self.physical_model = physical_model or build_physical_model_from_config(
            self.runner_config.robot_model_config_path
        )
        self.distribution = P2GraspCarryDesignDistribution(base_task_spec, self.distribution_config)
        self.irg_builder = IRGBuilder()
        self.envelope_extractor = InteractionEnvelopeExtractor()
        self.design_policy = design_policy or P2DesignPolicy(config=self.policy_config)

    def run(self, *, archive_path: str | Path | None = None) -> P4_0FullPipelineRunnerResult:
        success_count = 0
        failure_count = 0
        crash_count = 0
        object_drop_count = 0.0
        hard_collision_count = 0.0
        qp_infeasible_terminal_count = 0.0
        policy_command_count_sum = 0.0
        controller_command_count_sum = 0.0
        contact_candidate_episode_count = 0.0
        trajectory_episode_count = 0.0
        archives: list[EpisodeArchive] = []

        for index in range(self.runner_config.episode_count):
            seed = self.runner_config.seed + index
            sample = self._sample(seed=seed, index=index)
            try:
                archive = self._run_one(sample, seed=seed, episode_id=f"p4_0_full_pipeline_{index:04d}")
            except Exception:
                crash_count += 1
                continue

            if archive.success:
                success_count += 1
            else:
                failure_count += 1
            object_drop_count += archive.metrics.get("object_drop", 0.0)
            hard_collision_count += archive.metrics.get("hard_collision", 0.0)
            qp_infeasible_terminal_count += archive.metrics.get("qp_infeasible_terminal", 0.0)
            policy_command_count_sum += archive.metrics.get("policy_command_count", 0.0)
            controller_command_count_sum += archive.metrics.get("controller_command_count", 0.0)
            contact_candidate_episode_count += 1.0 if archive.metrics.get("contact_candidate_count", 0.0) > 0.0 else 0.0
            trajectory_episode_count += 1.0 if archive.metrics.get("trajectory_knot_count", 0.0) > 0.0 else 0.0
            if archive.success or not self.runner_config.archive_success_only:
                archives.append(archive)

        result = P4_0FullPipelineRunnerResult(
            episode_count=self.runner_config.episode_count,
            success_count=success_count,
            failure_count=failure_count,
            crash_count=crash_count,
            archives=archives,
            metrics={
                "success_rate": float(success_count) / float(self.runner_config.episode_count),
                "failure_rate": float(failure_count) / float(self.runner_config.episode_count),
                "crash_rate": float(crash_count) / float(self.runner_config.episode_count),
                "object_drop_rate": object_drop_count / float(self.runner_config.episode_count),
                "hard_collision_rate": hard_collision_count / float(self.runner_config.episode_count),
                "qp_infeasible_terminal_rate": qp_infeasible_terminal_count / float(self.runner_config.episode_count),
                "contact_candidate_episode_rate": contact_candidate_episode_count / float(self.runner_config.episode_count),
                "trajectory_episode_rate": trajectory_episode_count / float(self.runner_config.episode_count),
                "mean_policy_command_count": policy_command_count_sum / float(self.runner_config.episode_count),
                "mean_controller_command_count": controller_command_count_sum / float(self.runner_config.episode_count),
                "archive_count": float(len(archives)),
                "simplified_backend": 1.0,
                "isaac_backed": 0.0,
                "p4_full_completion": 0.0,
            },
        )
        if archive_path is not None:
            write_episode_archives_jsonl(archive_path, archives)
        return result

    def _sample(self, *, seed: int, index: int) -> P2DesignTaskSample:
        sample = self.distribution.sample(seed=seed, sample_index=index)
        task_data = sample.task_spec.to_dict()
        task_data["task_id"] = f"{self.base_task_spec.task_id}_p4_0_{index:04d}"
        metadata = dict(task_data.get("metadata", {}) or {})
        metadata["p4_phase"] = "P4.0"
        metadata["p4_0_backend"] = "simplified"
        metadata["p4_full_completion"] = False
        task_data["metadata"] = metadata
        return P2DesignTaskSample(
            task_spec=TaskSpec.from_dict(task_data),
            seed=sample.seed,
            sample_index=sample.sample_index,
            sampled_values=sample.sampled_values,
        )

    def _run_one(self, sample: P2DesignTaskSample, *, seed: int, episode_id: str) -> EpisodeArchive:
        builder_result = self.irg_builder.build_with_scene_graph(sample.task_spec)
        irg = builder_result.irg
        envelope = self.envelope_extractor.extract(irg)
        context = DesignPolicyContext(
            task_spec=sample.task_spec,
            irg=irg,
            physical_model=self.physical_model,
            interaction_envelope=envelope,
        )
        selection = self.design_policy.evaluate_candidates(context)
        selected = selection.selected_candidate
        target_graph = selected.design_output.target_morphology
        assembly_report = AssemblyRunner(
            config=AssemblyRunnerConfig(max_retries_per_step=self.runner_config.max_retries_per_step)
        ).run(
            target_graph,
            SimplifiedAssemblyExecutor(target_graph=target_graph),
        )
        rollout = _RolloutRecord()
        env: SimplifiedGraspCarryEnv | None = None
        if selected.feasibility_result.feasible and assembly_report.success:
            env = SimplifiedGraspCarryEnv(
                sample.task_spec,
                config=self.env_config,
                design_output=selected.design_output,
                assembled_morphology=assembly_report.final_state.physical_graph,
            )
            rollout = self._execute_rollout(
                env,
                sample.task_spec,
                selected_design=selected.design_output,
                assembled_morphology=assembly_report.final_state.physical_graph,
                seed=seed,
                episode_id=episode_id,
            )

        success = (
            selected.feasibility_result.feasible
            and assembly_report.success
            and bool(rollout.runtime_observations)
            and bool(rollout.runtime_observations[-1].task_progress.success)
            and not rollout.crashed
        )
        failure_reason = _failure_reason(selection, assembly_report, rollout, success=success)
        return _episode_archive(
            sample,
            irg=irg,
            envelope=envelope,
            selection=selection,
            assembly_report=assembly_report,
            env=env,
            rollout=rollout,
            physical_model=self.physical_model,
            episode_id=episode_id,
            seed=seed,
            success=success,
            failure_reason=failure_reason,
            config_hash=self._config_hash(),
            runner_config=self.runner_config,
        )

    def _execute_rollout(
        self,
        env: SimplifiedGraspCarryEnv,
        task_spec: TaskSpec,
        *,
        selected_design,
        assembled_morphology,
        seed: int,
        episode_id: str,
    ) -> _RolloutRecord:
        record = _RolloutRecord()
        try:
            observation = env.reset(
                task_spec=task_spec,
                design_output=selected_design,
                assembled_morphology=assembled_morphology,
                seed=seed,
                episode_id=episode_id,
            )
            record.runtime_observations.append(observation)
            previous_command: ControllerCommand | None = None
            for _ in range(env.config.max_episode_steps):
                if observation.task_progress.success or observation.task_progress.failure_reason is not None:
                    break
                low_context = LowLevelPolicyContext(
                    runtime_observation=observation,
                    morphology_graph=env.artifacts.design_output.target_morphology,
                    physical_model=env.artifacts.physical_model,
                    contact_wrench_trajectory=env.artifacts.contact_wrench_trajectory,
                    controller_status=observation.controller_status,
                )
                active_knot = select_active_knot(low_context)
                policy_command = env.low_level_policy.command(
                    LowLevelPolicyContext(
                        runtime_observation=observation,
                        morphology_graph=env.artifacts.design_output.target_morphology,
                        physical_model=env.artifacts.physical_model,
                        contact_wrench_trajectory=env.artifacts.contact_wrench_trajectory,
                        active_knot=active_knot,
                        controller_status=observation.controller_status,
                    )
                )
                controller_command = env.controller.compute(
                    ControllerContext(
                        runtime_observation=observation,
                        morphology_graph=env.artifacts.design_output.target_morphology,
                        physical_model=env.artifacts.physical_model,
                        active_knot=active_knot,
                        policy_command=policy_command,
                        previous_command=previous_command,
                        control_dt_s=env.config.dt_s,
                    )
                )
                record.policy_commands.append(policy_command)
                record.controller_commands.append(controller_command)
                observation = env.step(controller_command)
                previous_command = controller_command
                record.runtime_observations.append(observation)
                record.rewards.append(_reward_from_observation(observation))
        except Exception as exc:  # pragma: no cover - regression smoke path.
            record.crashed = True
            record.failure_reason = f"{type(exc).__name__}: {exc}"
            if not record.runtime_observations:
                record.runtime_observations.append(env.get_runtime_observation())
        if record.runtime_observations and record.runtime_observations[-1].task_progress.failure_reason is not None:
            record.failure_reason = record.runtime_observations[-1].task_progress.failure_reason
        return record

    def _config_hash(self) -> str:
        return stable_hash(
            {
                "runner": self.runner_config,
                "distribution": self.distribution_config,
                "policy": self.policy_config,
                "env": self.env_config,
            }
        )


def _episode_archive(
    sample: P2DesignTaskSample,
    *,
    irg: InteractionRequirementGraph,
    envelope: InteractionEnvelope,
    selection: P2DesignSelection,
    assembly_report,
    env: SimplifiedGraspCarryEnv | None,
    rollout: _RolloutRecord,
    physical_model: PhysicalModel,
    episode_id: str,
    seed: int,
    success: bool,
    failure_reason: str | None,
    config_hash: str,
    runner_config: P4_0FullPipelineRunnerConfig,
) -> EpisodeArchive:
    selected = selection.selected_candidate
    metrics = _episode_metrics(selection, assembly_report, env, rollout, success=success)
    return EpisodeArchive(
        episode_id=episode_id,
        task_spec=sample.task_spec,
        task_hash=sample.task_spec.stable_hash(),
        geometry_hashes={
            geometry.geometry_id: stable_hash(geometry)
            for geometry in sample.task_spec.scene.geometry_library
        },
        robot_model_hash=physical_model.stable_hash(),
        config_hash=config_hash,
        irg=irg,
        interaction_envelope=envelope,
        design_output=selected.design_output,
        feasibility_result=selected.feasibility_result,
        assembly_plan=assembly_report.plan.to_dict(),
        trajectory_records=[env.artifacts.contact_wrench_trajectory] if env is not None else [],
        policy_commands=rollout.policy_commands,
        controller_commands=rollout.controller_commands,
        rewards=rollout.rewards,
        metrics=metrics,
        success=success,
        failure_reason=failure_reason,
        runtime_observations=rollout.runtime_observations,
        actuator_target_records=[],
        rollout_artifacts={
            "phase": "P4.0",
            "backend": "simplified",
            "is_p4_full_completion": False,
            "isaac_backed": False,
            "physical_success_claim": False,
            "note": P4_0_SIMPLIFIED_BACKEND_NOTE,
        },
        learning_artifacts={},
        reproducibility={
            "source_hash": runner_config.source_hash,
            "random_seed": seed,
            "runner_version": runner_config.runner_version,
            "simulator_version": runner_config.simulator_version,
            "policy_version": selection.policy_version,
            "robot_model_config_path": runner_config.robot_model_config_path,
            "urdf_hash": str(physical_model.metadata.get("urdf_hash", "")),
            "thrust_model_hash": str(physical_model.metadata.get("thrust_model_hash", "")),
        },
    )


def _episode_metrics(
    selection: P2DesignSelection,
    assembly_report,
    env: SimplifiedGraspCarryEnv | None,
    rollout: _RolloutRecord,
    *,
    success: bool,
) -> dict[str, float]:
    selected = selection.selected_candidate
    controller_infeasible_steps = sum(
        1 for command in rollout.controller_commands if not command.controller_status.qp_feasible
    )
    terminal_qp_infeasible = (
        1.0
        if rollout.controller_commands and not rollout.controller_commands[-1].controller_status.qp_feasible
        else 0.0
    )
    failure_reason = rollout.failure_reason or ""
    object_drop = 1.0 if failure_reason in {"contact_break_force"} else 0.0
    trajectory_knot_count = (
        float(len(env.artifacts.contact_wrench_trajectory.knots))
        if env is not None
        else 0.0
    )
    contact_candidate_count = (
        float(len(env.artifacts.contact_candidate_set.candidates))
        if env is not None
        else 0.0
    )
    assignment_feasibility_cache_count = (
        float(len(env.artifacts.contact_candidate_set.assignment_feasibility_cache))
        if env is not None
        else 0.0
    )
    metrics = {
        "success": 1.0 if success else 0.0,
        "crashed": 1.0 if rollout.crashed else 0.0,
        "selected_feasible": 1.0 if selected.feasibility_result.feasible else 0.0,
        "selected_candidate_id": float(selected.candidate_id),
        "selected_soft_score": float(selected.soft_score),
        "p2_selected_design_used": 1.0,
        "fixed_simple_design_policy_used": 0.0,
        "p3_assembly_result_used": 1.0 if assembly_report.success else 0.0,
        "assembly_success": 1.0 if assembly_report.success else 0.0,
        "assembly_state_matches_target": 1.0 if assembly_report.state_matches_target else 0.0,
        "assembly_plan_step_count": float(len(assembly_report.plan.steps)),
        "assembly_executed_step_count": float(len(assembly_report.step_results)),
        "assembly_retry_count": float(assembly_report.retry_count),
        "assembly_abort_count": float(assembly_report.abort_count),
        "contact_candidate_count": contact_candidate_count,
        "assignment_feasibility_cache_count": assignment_feasibility_cache_count,
        "trajectory_count": 1.0 if trajectory_knot_count > 0.0 else 0.0,
        "trajectory_knot_count": trajectory_knot_count,
        "policy_command_count": float(len(rollout.policy_commands)),
        "controller_command_count": float(len(rollout.controller_commands)),
        "runtime_observation_count": float(len(rollout.runtime_observations)),
        "reward_count": float(len(rollout.rewards)),
        "qp_infeasible_step_count": float(controller_infeasible_steps),
        "qp_infeasible_terminal": terminal_qp_infeasible,
        "object_drop": object_drop,
        "hard_collision": 0.0,
        "simplified_backend": 1.0,
        "isaac_backed": 0.0,
        "p4_full_completion": 0.0,
    }
    for key, value in assembly_report.metrics.items():
        metrics[f"assembly_{key}"] = float(value)
    if rollout.runtime_observations:
        progress = rollout.runtime_observations[-1].task_progress
        metrics["final_progress_ratio"] = float(progress.progress_ratio)
        for key, value in progress.metrics.items():
            metrics[f"final_{key}"] = float(value)
    return metrics


def _failure_reason(
    selection: P2DesignSelection,
    assembly_report,
    rollout: _RolloutRecord,
    *,
    success: bool,
) -> str | None:
    if success:
        return None
    selected = selection.selected_candidate
    if not selected.feasibility_result.feasible:
        return selected.rejection_reason
    if not assembly_report.success:
        return assembly_report.failure_reason
    if rollout.crashed:
        return rollout.failure_reason or "simplified rollout crashed"
    if rollout.failure_reason is not None:
        return rollout.failure_reason
    return "simplified full-pipeline rollout failed"


def _reward_from_observation(observation: RuntimeObservation) -> dict[str, float]:
    goal_distance = observation.task_progress.metrics.get("goal_distance_m", 0.0)
    return {
        "success": 1.0 if observation.task_progress.success else 0.0,
        "progress": observation.task_progress.progress_ratio,
        "negative_goal_distance": -float(goal_distance),
    }


def _p2_design_policy_config_from_dict(data: dict[str, Any]) -> P2DesignPolicyConfig:
    if not isinstance(data, dict):
        raise SchemaValidationError("P2DesignPolicyConfig expects a dict")
    raw = dict(data)
    policy_fields = {item.name for item in fields(P2DesignPolicyConfig)}
    unknown = set(raw) - policy_fields
    if unknown:
        raise SchemaValidationError(f"P2DesignPolicyConfig got unknown fields: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    if "variants" in raw:
        kwargs["variants"] = tuple(GraspCarryMorphologyVariant(item) for item in raw.pop("variants"))
    for key, value in raw.items():
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        kwargs[key] = value
    return P2DesignPolicyConfig(**kwargs)
