from __future__ import annotations

from pathlib import Path

from amsrr.logging import EpisodeArchive, read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P1GraspCarryTaskDistribution,
    P1RunnerConfig,
    P1SimplifiedRunner,
    load_p1_runner_config,
    load_p1_task_distribution_config,
)


def test_p1_distribution_randomizes_configured_fields(grasp_carry_dict: dict) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    config = load_p1_task_distribution_config("configs/training/p1_grasp_carry_distribution.yaml")
    distribution = P1GraspCarryTaskDistribution(base_task, config)

    sample_a = distribution.sample(seed=10, sample_index=0)
    sample_b = distribution.sample(seed=11, sample_index=1)

    object_a = sample_a.task_spec.scene.objects[0]
    object_b = sample_b.task_spec.scene.objects[0]
    geometry_a = sample_a.task_spec.scene.geometry_library[0]
    assert sample_a.task_spec.task_id.endswith("_p1_0000")
    assert object_a.mass_kg != object_b.mass_kg
    assert object_a.friction != object_b.friction
    assert geometry_a.primitive_params is not None
    assert config.object_size_x_m[0] <= geometry_a.primitive_params["size_m"][0] <= config.object_size_x_m[1]
    assert config.object_mass_kg[0] <= object_a.mass_kg <= config.object_mass_kg[1]
    assert config.object_friction[0] <= object_a.friction <= config.object_friction[1]
    assert config.target_x_m[0] <= sample_a.task_spec.goals[0].target_pose_world[0] <= config.target_x_m[1]
    assert sample_a.sampled_values["object_mass_kg"] == object_a.mass_kg
    assert type(sample_a).from_json(sample_a.to_json()).to_dict() == sample_a.to_dict()


def test_p1_runner_collects_metrics_and_archives(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    _, distribution_config, env_config = load_p1_runner_config("configs/training/p1_grasp_carry_distribution.yaml")
    runner = P1SimplifiedRunner(
        base_task,
        runner_config=P1RunnerConfig(episode_count=8, seed=20, source_hash="unit-test"),
        distribution_config=distribution_config,
        env_config=env_config,
    )
    archive_path = tmp_path / "episodes.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.episode_count == 8
    assert result.crash_count == 0
    assert result.metrics["success_rate"] >= 0.60
    assert result.metrics["archive_count"] == 8.0
    assert len(result.archives) == 8
    assert archive_path.exists()
    first = result.archives[0]
    assert first.episode_id == "p1_runner_0000"
    assert first.policy_commands
    assert first.controller_commands
    assert first.runtime_observations == []
    assert first.actuator_target_records == []
    assert first.rollout_artifacts == {}
    assert first.learning_artifacts == {}
    assert first.rewards
    assert first.trajectory_records
    assert first.task_hash == first.task_spec.stable_hash()
    assert first.reproducibility["source_hash"] == "unit-test"
    assert first.reproducibility["simulator_version"] == "simplified_grasp_carry_env_v1"
    assert first.geometry_hashes
    loaded = read_episode_archives_jsonl(archive_path)
    assert len(loaded) == 8
    assert loaded[0].to_dict() == first.to_dict()
    legacy_archive = first.to_dict()
    for key in ("runtime_observations", "actuator_target_records", "rollout_artifacts", "learning_artifacts"):
        legacy_archive.pop(key)
    restored = EpisodeArchive.from_dict(legacy_archive)
    assert restored.runtime_observations == []
    assert restored.actuator_target_records == []
    assert restored.rollout_artifacts == {}
    assert restored.learning_artifacts == {}
    assert type(result).from_json(result.to_json()).to_dict() == result.to_dict()


def test_p1_runner_config_loader() -> None:
    runner_config, distribution_config, env_config = load_p1_runner_config(
        "configs/training/p1_grasp_carry_distribution.yaml"
    )

    assert runner_config.episode_count == 1000
    assert distribution_config.object_mass_kg == (0.5, 2.0)
    assert env_config.max_episode_steps == 40
