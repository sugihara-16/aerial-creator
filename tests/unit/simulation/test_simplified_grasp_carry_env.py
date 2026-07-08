from __future__ import annotations

from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation import (
    SimplifiedGraspCarryEnv,
    SimplifiedGraspCarryEnvConfig,
    SimulationEnvBase,
    run_crash_free_episodes,
)


def _env(grasp_carry_dict: dict) -> SimplifiedGraspCarryEnv:
    task = TaskSpec.from_dict(grasp_carry_dict)
    return SimplifiedGraspCarryEnv(
        task,
        config=SimplifiedGraspCarryEnvConfig(
            dt_s=0.1,
            max_episode_steps=40,
            object_tracking_speed_mps=10.0,
            initial_position_jitter_m=0.02,
        ),
    )


def test_simplified_grasp_carry_env_matches_base_protocol(grasp_carry_dict: dict) -> None:
    env: SimulationEnvBase = _env(grasp_carry_dict)

    observation = env.reset(seed=0, episode_id="protocol_smoke")

    assert isinstance(observation, RuntimeObservation)
    assert env.get_runtime_observation().task_progress.phase_label == "establish_contact"
    assert callable(env.step)


def test_simplified_grasp_carry_env_runs_policy_controller_episode(grasp_carry_dict: dict) -> None:
    env = _env(grasp_carry_dict)

    result = env.run_episode(seed=3, episode_id="single_episode")

    assert result.episode_id == "single_episode"
    assert result.crashed is False
    assert result.success is True
    assert result.failure_reason is None
    assert result.metrics["goal_distance_m"] <= env.config.goal_tolerance_m
    assert result.metrics["ever_attached"] == 1.0
    assert result.metrics["policy_command_count"] == result.metrics["controller_command_count"]
    assert result.metrics["controller_command_count"] > 0
    assert env.artifacts.contact_candidate_set.candidates
    assert env.artifacts.contact_wrench_trajectory.knots
    assert env.get_runtime_observation().task_progress.success is True
    assert isinstance(env._last_controller_command, ControllerCommand)
    assert type(result).from_json(result.to_json()).to_dict() == result.to_dict()


def test_simplified_grasp_carry_1000_episodes_crash_free(grasp_carry_dict: dict) -> None:
    env = _env(grasp_carry_dict)

    batch = run_crash_free_episodes(env, episode_count=1000, seed=100)

    assert batch.episode_count == 1000
    assert batch.crash_count == 0
    assert batch.metrics["crash_rate"] == 0.0
    assert batch.metrics["success_rate"] >= 0.60
    assert batch.success_count == 1000
    assert type(batch).from_json(batch.to_json()).to_dict() == batch.to_dict()
