from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_evaluation import (
    Order9EvaluationEpisode,
    build_order9_stage_evaluation_report,
    order9_evaluation_horizon_timeout_outcome,
    validate_order9_stage_evaluation_report,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.utils.hashing import hash_file


def test_stage_metrics_are_derived_from_hash_bound_episode_rows(
    tmp_path: Path,
) -> None:
    config = load_order9_learning_config()
    stage = order9_stage_by_id(config, "r1_pi_h_ppo_frozen_pi_l")
    raw = tmp_path / "isaac-rollout.json"
    raw.write_text("{}\n", encoding="utf-8")
    digest = hash_file(raw)
    episodes = [
        _episode(
            index,
            raw,
            digest,
            fallback_count=1 if index == 0 else 0,
        )
        for index in range(8)
    ]
    report = build_order9_stage_evaluation_report(
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        episodes=episodes,
        policy_checkpoint_sha256_by_family={
            Order9PolicyFamily.PI_H: "a" * 64
        },
        training_rollout_environment_step_count=6400,
        training_rollout_wall_elapsed_s=10.0,
    )

    metrics = validate_order9_stage_evaluation_report(
        report,
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        runtime=config.production_runtime,
    )

    assert metrics.episode_count == 8
    assert metrics.success_count == 8
    assert metrics.no_fallback_success_count == 7
    assert metrics.high_level_decision_count == 80
    assert metrics.fallback_decision_count == 1
    assert metrics.aggregate_env_steps_per_s == pytest.approx(640.0)


def test_stage_evaluation_rejects_tampered_raw_artifact(tmp_path: Path) -> None:
    config = load_order9_learning_config()
    stage = order9_stage_by_id(config, "r1_pi_h_ppo_frozen_pi_l")
    raw = tmp_path / "isaac-rollout.json"
    raw.write_text("{}\n", encoding="utf-8")
    digest = hash_file(raw)
    report = build_order9_stage_evaluation_report(
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        episodes=[_episode(index, raw, digest) for index in range(8)],
        policy_checkpoint_sha256_by_family={"pi_h": "a" * 64},
        training_rollout_environment_step_count=6400,
        training_rollout_wall_elapsed_s=10.0,
    )
    raw.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="artifact hash mismatch"):
        validate_order9_stage_evaluation_report(
            report,
            stage=stage,
            schedule_hash=order9_schedule_hash(config),
            runtime=config.production_runtime,
        )


def test_evaluation_horizon_exhaustion_is_a_task_failure_not_missing_evidence() -> None:
    outcome = order9_evaluation_horizon_timeout_outcome(
        environment=3,
        environment_step_count=15000,
        episode_return=123.5,
        terminal_phase_index=6,
    )

    assert outcome["environment"] == 3
    assert outcome["task_success"] is False
    assert outcome["safety_failure"] is False
    assert outcome["failure_reason"] == "evaluation_horizon_timeout"
    assert outcome["timeout"] is True
    assert outcome["evaluation_horizon_timeout"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"environment": -1}, "environment"),
        ({"environment_step_count": 0}, "positive steps"),
        ({"episode_return": float("nan")}, "finite"),
        ({"terminal_phase_index": -1}, "phase index"),
    ],
)
def test_evaluation_horizon_timeout_rejects_invalid_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "environment": 0,
        "environment_step_count": 10,
        "episode_return": 1.0,
        "terminal_phase_index": 2,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        order9_evaluation_horizon_timeout_outcome(**values)


def _episode(
    index: int,
    raw: Path,
    digest: str,
    *,
    fallback_count: int = 0,
) -> Order9EvaluationEpisode:
    return Order9EvaluationEpisode(
        episode_id=f"evaluation-{index}",
        task_id=f"task-{index}",
        split=DatasetSplit.VALIDATION,
        random_seed=100 + index,
        task_success=True,
        no_fallback_success=fallback_count == 0,
        safety_failure=False,
        high_level_decision_count=10,
        fallback_decision_count=fallback_count,
        environment_step_count=100,
        isaac_backed=True,
        full_mesh_evaluation=True,
        source_artifact_path=str(raw),
        source_artifact_sha256=digest,
    )
