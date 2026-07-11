from __future__ import annotations

from amsrr.training.p4_3_rollout_runner import (
    P4_3RolloutRunner,
    load_p4_3_rollout_runner_config,
)


def test_p4_3_rollout_config_loads_approved_minimum() -> None:
    config = load_p4_3_rollout_runner_config(
        "configs/training/p4_3_learning_bootstrap.yaml"
    )
    assert config.task_count == 6
    assert config.task_start_index == 0
    assert config.candidates_per_task == 2
    assert config.candidate_offset == 0
    assert config.learned_pi_l_runtime_blend_factor == 0.10
    assert config.dry_run is True
    assert set(config.split_fractions) == {"train", "validation", "held_out"}


def test_p4_3_rollout_dry_run_never_claims_isaac_or_learning() -> None:
    config = load_p4_3_rollout_runner_config(
        "configs/training/p4_3_learning_bootstrap.yaml"
    )
    config.task_count = 1
    config.candidates_per_task = 1
    result = P4_3RolloutRunner(config).run()
    assert result.dry_run is True
    assert not result.archives
    assert result.metrics["candidate_rollout_count"] == 1.0
    assert result.metrics["isaac_backed_count"] == 0.0
    assert result.candidate_results[0].deterministic_feasible is True


def test_p4_3_rollout_uses_distinct_randomization_seed_per_task() -> None:
    config = load_p4_3_rollout_runner_config(
        "configs/training/p4_3_learning_bootstrap.yaml"
    )
    runner = P4_3RolloutRunner(config)
    first = runner._p4_2_runner(0).build_p2_p3_rollout_case()
    second = runner._p4_2_runner(1).build_p2_p3_rollout_case()
    assert first.sample.seed == config.seed
    assert second.sample.seed == config.seed + 1
    assert first.sample.sampled_values != second.sample.sampled_values


def test_p4_3_rollout_candidate_window_collects_additional_feasible_designs() -> None:
    config = load_p4_3_rollout_runner_config(
        "configs/training/p4_3_learning_bootstrap.yaml"
    )
    config.candidates_per_task = 2
    runner = P4_3RolloutRunner(config)
    selection = runner._p4_2_runner(0).build_p2_p3_rollout_case().selection
    first_ids = [item.candidate_id for item in runner._rollout_candidates(selection)]
    config.candidate_offset = 2
    second_ids = [item.candidate_id for item in runner._rollout_candidates(selection)]

    assert len(first_ids) == len(second_ids) == 2
    assert set(first_ids).isdisjoint(second_ids)
    assert all(
        item.feasibility_result.feasible
        for item in selection.accepted_candidates
        if item.candidate_id in second_ids
    )
