from __future__ import annotations

from pathlib import Path

from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P3AssemblyEvaluationRunner,
    P3AssemblyRunnerConfig,
    load_p3_assembly_runner_config,
)


def test_p3_assembly_runner_collects_successful_assembly_archives(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    _, distribution_config, policy_config = load_p3_assembly_runner_config(
        "configs/training/p3_assembly_grasp_carry.yaml"
    )
    runner = P3AssemblyEvaluationRunner(
        base_task,
        runner_config=P3AssemblyRunnerConfig(episode_count=8, seed=60, source_hash="unit-test"),
        distribution_config=distribution_config,
        policy_config=policy_config,
    )
    archive_path = tmp_path / "p3_assembly_episodes.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.episode_count == 8
    assert result.crash_count == 0
    assert result.assembly_success_count == 8
    assert result.metrics["assembly_success_rate"] == 1.0
    assert result.metrics["state_match_rate"] == 1.0
    assert result.metrics["archive_count"] == 8.0
    assert len(result.archives) == 8
    first = result.archives[0]
    assert first.episode_id == "p3_assembly_0000"
    assert first.task_spec.task_id.endswith("_p3_0000")
    assert first.task_spec.metadata["assembly_evaluation_phase"] == "P3"
    assert first.design_output is not None
    assert first.feasibility_result is not None
    assert first.assembly_plan is not None
    assert first.trajectory_records == []
    assert first.policy_commands == []
    assert first.controller_commands == []
    assert first.metrics["assembly_success"] == 1.0
    assert first.metrics["assembly_state_matches_target"] == 1.0
    assert first.metrics["assembly_abort_count"] == 0.0
    assert first.metrics["assembly_plan_step_count"] > 0.0
    assert first.metrics["assembly_executed_step_count"] == first.metrics["assembly_plan_step_count"]
    assert first.reproducibility["source_hash"] == "unit-test"
    assert first.reproducibility["runner_version"] == "p3_assembly_eval_runner_v1"
    loaded = read_episode_archives_jsonl(archive_path)
    assert len(loaded) == 8
    assert loaded[0].to_dict() == first.to_dict()
    assert type(result).from_json(result.to_json()).to_dict() == result.to_dict()


def test_p3_assembly_runner_config_loader() -> None:
    runner_config, distribution_config, policy_config = load_p3_assembly_runner_config(
        "configs/training/p3_assembly_grasp_carry.yaml"
    )

    assert runner_config.episode_count == 1000
    assert runner_config.max_retries_per_step == 1
    assert distribution_config.object_mass_kg == (0.4, 2.5)
    assert len(policy_config.variants) == 4
