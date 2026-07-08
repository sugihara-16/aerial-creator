from __future__ import annotations

from pathlib import Path

from amsrr.logging import read_episode_archives_jsonl
from amsrr.morphology.grasp_carry_designs import GRASP_CARRY_VARIANT_ORDER
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P2DesignEvaluationRunner,
    P2DesignRunnerConfig,
    P2GraspCarryDesignDistribution,
    load_p2_design_distribution_config,
    load_p2_design_runner_config,
)


def test_p2_design_distribution_randomizes_and_marks_metadata(grasp_carry_dict: dict) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    config = load_p2_design_distribution_config("configs/training/p2_design_grasp_carry.yaml")
    distribution = P2GraspCarryDesignDistribution(base_task, config)

    sample_a = distribution.sample(seed=30, sample_index=0)
    sample_b = distribution.sample(seed=31, sample_index=1)

    object_a = sample_a.task_spec.scene.objects[0]
    object_b = sample_b.task_spec.scene.objects[0]
    geometry_a = sample_a.task_spec.scene.geometry_library[0]
    assert sample_a.task_spec.task_id.endswith("_p2_0000")
    assert sample_a.task_spec.metadata["randomization_family"] == "p2_design_grasp_carry"
    assert sample_a.task_spec.metadata["design_evaluation_phase"] == "P2"
    assert object_a.mass_kg != object_b.mass_kg
    assert geometry_a.primitive_params is not None
    assert config.object_size_x_m[0] <= geometry_a.primitive_params["size_m"][0] <= config.object_size_x_m[1]
    assert config.object_mass_kg[0] <= object_a.mass_kg <= config.object_mass_kg[1]
    assert config.object_friction[0] <= object_a.friction <= config.object_friction[1]
    assert sample_a.sampled_values["object_mass_kg"] == object_a.mass_kg
    assert type(sample_a).from_json(sample_a.to_json()).to_dict() == sample_a.to_dict()


def test_p2_design_runner_collects_feasibility_archives(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    _, distribution_config, policy_config = load_p2_design_runner_config(
        "configs/training/p2_design_grasp_carry.yaml"
    )
    runner = P2DesignEvaluationRunner(
        base_task,
        runner_config=P2DesignRunnerConfig(episode_count=8, seed=40, source_hash="unit-test"),
        distribution_config=distribution_config,
        policy_config=policy_config,
    )
    archive_path = tmp_path / "p2_design_episodes.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.episode_count == 8
    assert result.crash_count == 0
    assert result.valid_design_count == 8
    assert result.metrics["valid_design_rate"] == 1.0
    assert result.metrics["mean_accepted_candidate_count"] == float(len(GRASP_CARRY_VARIANT_ORDER))
    assert result.metrics["archive_count"] == 8.0
    assert len(result.archives) == 8
    assert archive_path.exists()
    first = result.archives[0]
    assert first.episode_id == "p2_design_0000"
    assert first.design_output is not None
    assert first.feasibility_result is not None
    assert first.feasibility_result.proxy_scores["L_FEASIBLE"] == 1.0
    assert first.feasibility_result.margins["required_slot_coverage_ratio"] == 1.0
    assert first.metrics["label_L_FEASIBLE"] == 1.0
    assert first.metrics["candidate_count"] == float(len(GRASP_CARRY_VARIANT_ORDER))
    assert first.trajectory_records == []
    assert first.policy_commands == []
    assert first.controller_commands == []
    assert first.reproducibility["source_hash"] == "unit-test"
    assert first.reproducibility["runner_version"] == "p2_design_eval_runner_v1"
    assert first.geometry_hashes
    loaded = read_episode_archives_jsonl(archive_path)
    assert len(loaded) == 8
    assert loaded[0].to_dict() == first.to_dict()
    assert type(result).from_json(result.to_json()).to_dict() == result.to_dict()


def test_p2_design_runner_config_loader() -> None:
    runner_config, distribution_config, policy_config = load_p2_design_runner_config(
        "configs/training/p2_design_grasp_carry.yaml"
    )

    assert runner_config.episode_count == 1000
    assert distribution_config.object_mass_kg == (0.4, 2.5)
    assert policy_config.variants == GRASP_CARRY_VARIANT_ORDER
