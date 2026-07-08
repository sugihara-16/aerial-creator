from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amsrr.controllers.controller_base import ControllerContext
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext, select_active_knot
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.policies import ControllerCommand, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.simplified_grasp_carry_env import SimplifiedGraspCarryEnv, SimplifiedGraspCarryEnvConfig
from amsrr.training.p1_task_distribution import P1GraspCarryTaskDistribution, P1TaskDistributionConfig, P1TaskSample
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


@dataclass
class P1RunnerConfig(SchemaBase):
    episode_count: int = 1000
    seed: int = 0
    source_hash: str = "unknown"
    simulator_version: str = "simplified_grasp_carry_env_v1"
    archive_success_only: bool = False

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P1RunnerConfig.episode_count must be positive")


@dataclass
class P1RunnerResult(SchemaBase):
    episode_count: int
    success_count: int
    crash_count: int
    failure_count: int
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_p1_runner_config(path: str) -> tuple[P1RunnerConfig, P1TaskDistributionConfig, SimplifiedGraspCarryEnvConfig]:
    data = load_config(path)
    return (
        P1RunnerConfig.from_dict(data.get("runner", {})),
        P1TaskDistributionConfig.from_dict(data.get("distribution", {})),
        SimplifiedGraspCarryEnvConfig(**data.get("env", {})),
    )


class P1SimplifiedRunner:
    def __init__(
        self,
        base_task_spec: TaskSpec,
        *,
        runner_config: P1RunnerConfig | None = None,
        distribution_config: P1TaskDistributionConfig | None = None,
        env_config: SimplifiedGraspCarryEnvConfig | None = None,
    ) -> None:
        self.base_task_spec = base_task_spec
        self.runner_config = runner_config or P1RunnerConfig()
        self.distribution_config = distribution_config or P1TaskDistributionConfig()
        self.env_config = env_config or SimplifiedGraspCarryEnvConfig()
        self.distribution = P1GraspCarryTaskDistribution(base_task_spec, self.distribution_config)

    def run(self, *, archive_path: str | Path | None = None) -> P1RunnerResult:
        success_count = 0
        crash_count = 0
        failure_count = 0
        total_steps = 0
        archives: list[EpisodeArchive] = []

        first_sample = self.distribution.sample(seed=self.runner_config.seed, sample_index=0)
        env = SimplifiedGraspCarryEnv(first_sample.task_spec, config=self.env_config)
        for index in range(self.runner_config.episode_count):
            seed = self.runner_config.seed + index
            sample = self.distribution.sample(seed=seed, sample_index=index)
            archive = self._run_one(env, sample, seed=seed, episode_id=f"p1_runner_{index:04d}")
            success_count += 1 if archive.success else 0
            crashed = archive.metrics.get("crashed", 0.0) > 0.0
            crash_count += 1 if crashed else 0
            failure_count += 1 if not archive.success else 0
            total_steps += int(archive.metrics.get("steps", 0.0))
            if archive.success or not self.runner_config.archive_success_only:
                archives.append(archive)

        result = P1RunnerResult(
            episode_count=self.runner_config.episode_count,
            success_count=success_count,
            crash_count=crash_count,
            failure_count=failure_count,
            archives=archives,
            metrics={
                "success_rate": float(success_count) / float(self.runner_config.episode_count),
                "crash_rate": float(crash_count) / float(self.runner_config.episode_count),
                "failure_rate": float(failure_count) / float(self.runner_config.episode_count),
                "mean_steps": float(total_steps) / float(self.runner_config.episode_count),
                "archive_count": float(len(archives)),
            },
        )
        if archive_path is not None:
            write_episode_archives_jsonl(archive_path, archives)
        return result

    def _run_one(
        self,
        env: SimplifiedGraspCarryEnv,
        sample: P1TaskSample,
        *,
        seed: int,
        episode_id: str,
    ) -> EpisodeArchive:
        policy_commands: list[PolicyCommand] = []
        controller_commands: list[ControllerCommand] = []
        rewards: list[dict[str, float]] = []
        crashed = False
        failure_reason: str | None = None
        try:
            observation = env.reset(task_spec=sample.task_spec, seed=seed, episode_id=episode_id)
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
                policy_commands.append(policy_command)
                controller_commands.append(controller_command)
                observation = env.step(controller_command)
                previous_command = controller_command
                rewards.append(_reward_from_observation(observation))
        except Exception as exc:  # pragma: no cover - regression smoke path.
            crashed = True
            failure_reason = f"{type(exc).__name__}: {exc}"
            observation = env.get_runtime_observation()
        if observation.task_progress.failure_reason is not None:
            failure_reason = observation.task_progress.failure_reason
        success = bool(observation.task_progress.success) and not crashed
        return _episode_archive(
            env,
            sample,
            episode_id=episode_id,
            seed=seed,
            policy_commands=policy_commands,
            controller_commands=controller_commands,
            rewards=rewards,
            success=success,
            crashed=crashed,
            failure_reason=failure_reason,
            config_hash=self._config_hash(),
            runner_config=self.runner_config,
        )

    def _config_hash(self) -> str:
        return stable_hash(
            {
                "runner": self.runner_config,
                "distribution": self.distribution_config,
                "env": self.env_config,
            }
        )


def _episode_archive(
    env: SimplifiedGraspCarryEnv,
    sample: P1TaskSample,
    *,
    episode_id: str,
    seed: int,
    policy_commands: list[PolicyCommand],
    controller_commands: list[ControllerCommand],
    rewards: list[dict[str, float]],
    success: bool,
    crashed: bool,
    failure_reason: str | None,
    config_hash: str,
    runner_config: P1RunnerConfig,
) -> EpisodeArchive:
    observation = env.get_runtime_observation()
    physical_model = env.artifacts.physical_model
    metrics = dict(observation.task_progress.metrics)
    metrics.update(
        {
            "success": 1.0 if success else 0.0,
            "crashed": 1.0 if crashed else 0.0,
            "steps": float(len(controller_commands)),
            "policy_command_count": float(len(policy_commands)),
            "controller_command_count": float(len(controller_commands)),
        }
    )
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
        irg=env.artifacts.irg,
        interaction_envelope=env.artifacts.interaction_envelope,
        design_output=env.artifacts.design_output,
        feasibility_result=None,
        assembly_plan=None,
        trajectory_records=[env.artifacts.contact_wrench_trajectory],
        policy_commands=policy_commands,
        controller_commands=controller_commands,
        rewards=rewards,
        metrics=metrics,
        success=success,
        failure_reason=failure_reason,
        reproducibility={
            "source_hash": runner_config.source_hash,
            "random_seed": seed,
            "simulator_version": runner_config.simulator_version,
            "urdf_hash": str(physical_model.metadata.get("urdf_hash", "")),
            "thrust_model_hash": str(physical_model.metadata.get("thrust_model_hash", "")),
        },
    )


def _reward_from_observation(observation: RuntimeObservation) -> dict[str, float]:
    goal_distance = observation.task_progress.metrics.get("goal_distance_m", 0.0)
    return {
        "success": 1.0 if observation.task_progress.success else 0.0,
        "progress": observation.task_progress.progress_ratio,
        "negative_goal_distance": -float(goal_distance),
    }
