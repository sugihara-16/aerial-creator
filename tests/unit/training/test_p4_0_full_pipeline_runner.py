from __future__ import annotations

from pathlib import Path

from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P4_0FullPipelineRunner,
    P4_0FullPipelineRunnerConfig,
    load_p4_0_full_pipeline_runner_config,
)


def test_p4_0_full_pipeline_runner_archives_full_simplified_pipeline(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    _, distribution_config, policy_config, env_config = load_p4_0_full_pipeline_runner_config(
        "configs/training/p4_0_grasp_carry.yaml"
    )
    runner = P4_0FullPipelineRunner(
        base_task,
        runner_config=P4_0FullPipelineRunnerConfig(
            episode_count=4,
            seed=900,
            source_hash="unit-test",
        ),
        distribution_config=distribution_config,
        policy_config=policy_config,
        env_config=env_config,
    )
    archive_path = tmp_path / "p4_0_episodes.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.episode_count == 4
    assert result.crash_count == 0
    assert result.success_count == 4
    assert result.metrics["success_rate"] == 1.0
    assert result.metrics["object_drop_rate"] == 0.0
    assert result.metrics["hard_collision_rate"] == 0.0
    assert result.metrics["qp_infeasible_terminal_rate"] == 0.0
    assert result.metrics["simplified_backend"] == 1.0
    assert result.metrics["isaac_backed"] == 0.0
    assert result.metrics["p4_full_completion"] == 0.0
    assert result.metrics["archive_count"] == 4.0

    first = result.archives[0]
    assert first.episode_id == "p4_0_full_pipeline_0000"
    assert first.task_spec.task_id.endswith("_p4_0_0000")
    assert first.task_spec.metadata["p4_phase"] == "P4.0"
    assert first.task_spec.metadata["p4_0_backend"] == "simplified"
    assert first.task_spec.metadata["p4_full_completion"] is False
    assert first.design_output is not None
    assert first.design_output.design_scores["p2_design_policy_selected"] == 1.0
    assert first.feasibility_result is not None
    assert first.feasibility_result.feasible is True
    assert first.assembly_plan is not None
    assert first.trajectory_records
    assert first.policy_commands
    assert first.controller_commands
    assert first.rewards
    assert first.runtime_observations
    assert first.actuator_target_records == []
    assert first.learning_artifacts == {}
    assert first.metrics["p2_selected_design_used"] == 1.0
    assert first.metrics["fixed_simple_design_policy_used"] == 0.0
    assert first.metrics["p3_assembly_result_used"] == 1.0
    assert first.metrics["assembly_success"] == 1.0
    assert first.metrics["assembly_state_matches_target"] == 1.0
    assert first.metrics["contact_candidate_count"] > 0.0
    assert first.metrics["assignment_feasibility_cache_count"] > 0.0
    assert first.metrics["trajectory_count"] == 1.0
    assert first.metrics["trajectory_knot_count"] > 0.0
    assert first.metrics["policy_command_count"] == first.metrics["controller_command_count"]
    assert first.metrics["policy_command_count"] > 0.0
    assert first.metrics["reward_count"] == first.metrics["controller_command_count"]
    assert first.metrics["runtime_observation_count"] == first.metrics["controller_command_count"] + 1.0
    assert first.metrics["simplified_backend"] == 1.0
    assert first.metrics["isaac_backed"] == 0.0
    assert first.metrics["p4_full_completion"] == 0.0
    assert first.rollout_artifacts["phase"] == "P4.0"
    assert first.rollout_artifacts["backend"] == "simplified"
    assert first.rollout_artifacts["is_p4_full_completion"] is False
    assert first.rollout_artifacts["isaac_backed"] is False
    assert first.rollout_artifacts["physical_success_claim"] is False
    assert "not Isaac-backed physical success rates" in first.rollout_artifacts["note"]
    assert first.reproducibility["source_hash"] == "unit-test"
    assert first.reproducibility["runner_version"] == "p4_0_full_pipeline_runner_v1"
    assert first.reproducibility["simulator_version"] == "simplified_grasp_carry_env_v1"

    loaded = read_episode_archives_jsonl(archive_path)
    assert len(loaded) == 4
    assert loaded[0].to_dict() == first.to_dict()
    assert type(result).from_json(result.to_json()).to_dict() == result.to_dict()


def test_p4_0_full_pipeline_runner_config_loader() -> None:
    runner_config, distribution_config, policy_config, env_config = load_p4_0_full_pipeline_runner_config(
        "configs/training/p4_0_grasp_carry.yaml"
    )

    assert runner_config.episode_count == 1000
    assert runner_config.runner_version == "p4_0_full_pipeline_runner_v1"
    assert runner_config.simulator_version == "simplified_grasp_carry_env_v1"
    assert distribution_config.object_mass_kg == (0.4, 2.5)
    assert len(policy_config.variants) == 4
    assert env_config.max_episode_steps == 40
