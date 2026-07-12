from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    POLICY_COMMAND_CONTRACT_LEGACY,
)
from amsrr.training.order3_free_flight import (
    ORDER3_REQUIRED_MODULE_COUNTS,
    ORDER3_TWO_MODULE_STRUCTURAL_CAPACITY,
    TRUE_CENTROIDAL_TRACKING_SOURCE,
    Order3CurriculumSchedule,
    Order3EvaluationEpisode,
    Order3FreeFlightStep,
    Order3LearningMode,
    Order3PrivilegedRewardSignals,
    Order3TaskMode,
    compute_order3_free_flight_reward,
    default_order3_curriculum_schedule,
    order3_scope_metadata,
    recommended_order3_morphology_split_counts,
    summarize_order3_module_coverage,
)


IDENTITY_POSE = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)


def _step(**overrides) -> Order3FreeFlightStep:
    values = {
        "module_count": 3,
        "task_mode": Order3TaskMode.HOVER,
        "centroidal_pose_world": IDENTITY_POSE,
        "centroidal_twist_world": [0.0] * 6,
        "target_pose_world": IDENTITY_POSE,
        "target_twist_world": [0.0] * 6,
        "within_tolerance_duration_s": 1.0,
    }
    values.update(overrides)
    return Order3FreeFlightStep(**values)


def test_default_curriculum_is_bc_then_ppo_then_held_out_evaluation() -> None:
    schedule = default_order3_curriculum_schedule()

    assert schedule.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert schedule.tracking_state_source == TRUE_CENTROIDAL_TRACKING_SOURCE
    assert schedule.stages[0].learning_mode == Order3LearningMode.BEHAVIOR_CLONING
    assert [stage.learning_mode for stage in schedule.stages[1:4]] == [
        Order3LearningMode.PPO,
        Order3LearningMode.PPO,
        Order3LearningMode.PPO,
    ]
    assert schedule.stages[-1].learning_mode == Order3LearningMode.EVALUATION
    assert schedule.stages[-1].held_out_only is True
    assert schedule.stages[-1].min_modules == 2
    assert schedule.stages[-1].max_modules == 8
    assert all(stage.residual_policy for stage in schedule.stages)
    assert all(not stage.object_task_claim for stage in schedule.stages)
    assert all(not stage.contact_task_claim for stage in schedule.stages)
    assert schedule.p4_full_completion_claim is False

    restored = Order3CurriculumSchedule.from_json(schedule.to_json())
    assert restored == schedule


def test_curriculum_rejects_contact_scope_and_missing_ppo() -> None:
    schedule = default_order3_curriculum_schedule()
    with pytest.raises(SchemaValidationError, match="object/contact"):
        replace(schedule.stages[2], contact_task_claim=True)

    with pytest.raises(SchemaValidationError, match="PPO"):
        Order3CurriculumSchedule(
            stages=[schedule.stages[0], replace(schedule.stages[-1], stage_index=1)]
        )


def test_true_centroidal_hover_success_is_terminal_and_scope_limited() -> None:
    result = compute_order3_free_flight_reward(_step())

    assert result.success is True
    assert result.failure is False
    assert result.terminal is True
    assert result.tracking_cost == pytest.approx(0.0)
    assert result.terms["terminal_success"] == 1.0
    assert result.reward > 10.0
    assert result.tracking_state_source == TRUE_CENTROIDAL_TRACKING_SOURCE
    assert result.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert result.privileged_reward_used is False
    assert result.privileged_actor_observation_allowed is False
    assert result.object_task_claim is False
    assert result.contact_task_claim is False
    assert result.p4_full_completion_claim is False


def test_reward_fails_closed_for_safety_terminals() -> None:
    result = compute_order3_free_flight_reward(
        _step(
            qp_feasible=False,
            hard_collision=True,
            unsupported_actuator=True,
            within_tolerance_duration_s=0.0,
        )
    )

    assert result.success is False
    assert result.failure is True
    assert result.terminal is True
    assert result.failure_reasons == [
        "qp_infeasible",
        "hard_collision",
        "unsupported_actuator",
    ]
    assert result.terms["terminal_failure"] == 1.0
    assert result.reward < 0.0


def test_privileged_reward_compares_disturbed_policy_with_baseline_only() -> None:
    privileged = Order3PrivilegedRewardSignals(
        applied_external_wrench_body=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mass_scale=1.1,
        deterministic_baseline_tracking_cost=2.0,
    )
    improved = compute_order3_free_flight_reward(
        _step(
            centroidal_pose_world=(0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            within_tolerance_duration_s=0.0,
            privileged=privileged,
        )
    )
    regressed = compute_order3_free_flight_reward(
        _step(
            centroidal_pose_world=(2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            within_tolerance_duration_s=0.0,
            privileged=privileged,
        )
    )

    assert improved.privileged_reward_used is True
    assert improved.terms["privileged_disturbance_severity"] > 0.0
    assert improved.terms["privileged_baseline_improvement"] > 0.0
    assert regressed.terms["privileged_baseline_improvement"] < 0.0
    assert improved.privileged_actor_observation_allowed is False
    with pytest.raises(SchemaValidationError, match="actor"):
        Order3PrivilegedRewardSignals(actor_observation_allowed=True)


def test_takeoff_success_requires_height_gate() -> None:
    below = compute_order3_free_flight_reward(
        _step(task_mode=Order3TaskMode.TAKEOFF, takeoff_height_gain_ratio=0.79)
    )
    passing = compute_order3_free_flight_reward(
        _step(task_mode=Order3TaskMode.TAKEOFF, takeoff_height_gain_ratio=0.80)
    )

    assert below.success is False
    assert below.terminal is False
    assert passing.success is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tracking_state_source": "base_module_fc"}, "true morphology centroidal"),
        ({"control_contract_version": POLICY_COMMAND_CONTRACT_LEGACY}, "centroidal_local_joint_v2"),
        ({"object_task_active": True}, "exclude object"),
        ({"contact_assignment_count": 1}, "exclude object"),
        ({"dock_motion_commanded": True}, "exclude object"),
    ],
)
def test_free_flight_step_rejects_out_of_scope_inputs(override, message) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        _step(**override)


def test_module_coverage_aggregates_two_through_eight_and_baseline_comparison() -> None:
    episodes = [
        Order3EvaluationEpisode(
            episode_id=f"held-{module_count}",
            structural_hash=f"hash-{module_count}",
            module_count=module_count,
            split=DatasetSplit.HELD_OUT,
            success=module_count != 8,
            tracking_cost=0.5,
            deterministic_baseline_tracking_cost=1.0,
            randomized=True,
            fallback_used=module_count == 7,
            hard_collision=module_count == 8,
        )
        for module_count in ORDER3_REQUIRED_MODULE_COUNTS
    ]

    summary = summarize_order3_module_coverage(episodes)

    assert summary.coverage_complete is True
    assert summary.covered_module_counts == list(range(2, 9))
    assert summary.missing_module_counts == []
    assert summary.episode_count == 7
    assert summary.aggregate_success_rate == pytest.approx(6.0 / 7.0)
    assert summary.randomized_mean_relative_improvement == pytest.approx(0.5)
    assert summary.per_module_count["2"].unique_structural_hash_count == 1
    assert summary.per_module_count["8"].success_rate == 0.0
    assert summary.safety_failure_episode_count == 1
    assert summary.safety_passed is False
    assert summary.fallback_rate == pytest.approx(1.0 / 7.0)
    assert summary.split_episode_counts[DatasetSplit.HELD_OUT.value] == 7
    assert summary.object_task_claim is False
    assert summary.contact_task_claim is False
    assert summary.p4_full_completion_claim is False

    incomplete = summarize_order3_module_coverage(episodes[:-1])
    assert incomplete.coverage_complete is False
    assert incomplete.missing_module_counts == [8]


def test_morphology_split_recommendation_respects_two_module_capacity() -> None:
    two_module = recommended_order3_morphology_split_counts(2)
    three_module = recommended_order3_morphology_split_counts(3)

    assert sum(two_module.values()) == ORDER3_TWO_MODULE_STRUCTURAL_CAPACITY
    assert two_module == {
        DatasetSplit.TRAIN: 4,
        DatasetSplit.VALIDATION: 2,
        DatasetSplit.HELD_OUT: 2,
    }
    assert three_module == {
        DatasetSplit.TRAIN: 8,
        DatasetSplit.VALIDATION: 2,
        DatasetSplit.HELD_OUT: 2,
    }


def test_scope_metadata_never_claims_contact_object_or_p4_full() -> None:
    metadata = order3_scope_metadata()

    assert metadata["module_count_min"] == 2
    assert metadata["module_count_max"] == 8
    assert metadata["object_task_claim"] is False
    assert metadata["contact_task_claim"] is False
    assert metadata["p4_full_completion_claim"] is False
    assert "dock_joint_motion" in metadata["excluded_claims"]
