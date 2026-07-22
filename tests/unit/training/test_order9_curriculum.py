from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_curriculum import (
    ObjectDistributionLevel,
    Order9LearningMode,
    Order9LearningTarget,
    Order9StageMetrics,
    PiHOutputScope,
    evaluate_stage_promotion,
    load_order9_learning_config,
    resolve_order9_stage_runtime,
)


def test_order9_config_loads_complete_pi_l_pi_h_pi_d_curriculum() -> None:
    config = load_order9_learning_config()
    stages = config.curriculum.stages

    assert stages[0].learning_mode == Order9LearningMode.COLLECTION
    assert stages[-1].learning_mode == Order9LearningMode.EVALUATION
    assert stages[3].stage_id == "c3_pi_l_ppo_arbitrary_morphology"
    assert (stages[3].min_modules, stages[3].max_modules) == (2, 8)
    assignment_warmup = stages[4]
    assert assignment_warmup.pi_h_output_scope == PiHOutputScope.ASSIGNMENT_ONLY_WARMUP
    assert assignment_warmup.learning_target == Order9LearningTarget.PI_H_ASSIGNMENT
    assert any(
        stage.learning_target == Order9LearningTarget.PI_H_TRAJECTORY
        and stage.learning_mode == Order9LearningMode.PPO
        and stage.pi_h_output_scope == PiHOutputScope.FULL_CONTACT_WRENCH_TRAJECTORY
        for stage in stages
    )
    assert any(
        stage.learning_target == Order9LearningTarget.PI_D
        and stage.learning_mode == Order9LearningMode.PPO
        and stage.design_action_mask_required
        for stage in stages
    )
    assert stages[-1].object_distribution == ObjectDistributionLevel.HELD_OUT_SHAPES_AND_INERTIA
    assert config.hard_checker.backend == (
        "hybrid_lightweight_qp_persistent_isaac_shadow"
    )
    assert config.hard_checker.max_proposal_attempts == 2
    assert config.hard_checker.projection_allowed is False
    assert config.hard_checker.require_current_pi_l_checkpoint is True
    assert config.teacher_collection.episode_count == 20
    assert config.teacher_collection.validation_episode_count == 3
    assert config.teacher_collection.held_out_episode_count == 3
    assert config.teacher_collection.low_level_stride == 5
    assert config.teacher_collection.parallel_process_count == 2
    assert stages[0].minimum_episodes == 20
    assert stages[1].minimum_episodes == 100
    assert stages[0].object_distribution == (
        ObjectDistributionLevel.CONSERVATIVE_ORDER8_ANCHOR
    )
    assert stages[1].object_distribution == (
        ObjectDistributionLevel.CONSERVATIVE_ORDER8_ANCHOR
    )
    assert config.optimization.pi_l_bc.phase_balanced_sampling is True


def test_c2_uses_provisional_2048_runtime_without_changing_c3_default() -> None:
    config = load_order9_learning_config()
    c2 = next(stage for stage in config.curriculum.stages if stage.stage_index == 2)
    c3 = next(stage for stage in config.curriculum.stages if stage.stage_index == 3)

    c2_runtime = resolve_order9_stage_runtime(config, c2)
    c3_runtime = resolve_order9_stage_runtime(config, c3)

    assert c2_runtime.environment_count == 2048
    assert c2_runtime.rollout_steps_per_environment == 16
    assert c2_runtime.generation_environment_steps == 32768
    assert c2_runtime.environment_count_source == "curriculum_stage_override"
    assert c3_runtime.environment_count == 128
    assert c3_runtime.rollout_steps_per_environment == 256
    assert c3_runtime.generation_environment_steps == 32768
    assert c3_runtime.environment_count_source == "production_runtime_default"


def test_stage_runtime_override_requires_an_explicit_environment_step_pair() -> None:
    config = load_order9_learning_config()
    c2 = next(stage for stage in config.curriculum.stages if stage.stage_index == 2)

    with pytest.raises(SchemaValidationError, match="must specify positive"):
        replace(c2, rollout_steps_per_environment=None)


def test_fallback_rate_is_decision_fraction_and_blocks_promotion() -> None:
    config = load_order9_learning_config()
    stage = next(
        value
        for value in config.curriculum.stages
        if value.stage_id == "c6_pi_h_full_trajectory_ppo"
    )
    metrics = Order9StageMetrics(
        episode_count=500,
        success_count=450,
        no_fallback_success_count=400,
        safety_failure_episode_count=0,
        high_level_decision_count=2000,
        fallback_decision_count=300,
        aggregate_env_steps_per_s=600.0,
    )

    decision = evaluate_stage_promotion(stage, metrics, config.runtime_benchmark)

    assert metrics.fallback_rate == pytest.approx(0.15)
    assert decision.promote is False
    assert "maximum_fallback_rate" in decision.failed_gates


def test_promotion_requires_throughput_and_no_fallback_success() -> None:
    config = load_order9_learning_config()
    stage = next(
        value
        for value in config.curriculum.stages
        if value.stage_id == "c6_pi_h_full_trajectory_ppo"
    )
    passing = Order9StageMetrics(
        episode_count=500,
        success_count=425,
        no_fallback_success_count=390,
        safety_failure_episode_count=0,
        high_level_decision_count=2000,
        fallback_decision_count=100,
        aggregate_env_steps_per_s=550.0,
    )
    failing_throughput = replace(passing, aggregate_env_steps_per_s=499.0)

    assert evaluate_stage_promotion(stage, passing, config.runtime_benchmark).promote
    decision = evaluate_stage_promotion(
        stage,
        failing_throughput,
        config.runtime_benchmark,
    )
    assert decision.promote is False
    assert "minimum_aggregate_env_steps_per_s" in decision.failed_gates


def test_final_pi_h_stage_cannot_be_relabelled_assignment_only() -> None:
    config = load_order9_learning_config()
    stage = next(
        value
        for value in config.curriculum.stages
        if value.stage_id == "c6_pi_h_full_trajectory_ppo"
    )

    with pytest.raises(SchemaValidationError, match="pi_H"):
        replace(stage, pi_h_output_scope=PiHOutputScope.ASSIGNMENT_ONLY_WARMUP)
