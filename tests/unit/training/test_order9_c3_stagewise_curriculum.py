from __future__ import annotations

from pathlib import Path
import json

import pytest

from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_c3_stagewise_curriculum import (
    cleanup_order9_intermediate_raw_artifacts,
    compare_order9_c3_incremental_forgetting,
    select_order9_c3_new_module_validation_bucket_ids,
    select_order9_c3_stagewise_validation_bucket_ids,
    should_run_order9_c3_continuous_validation,
    summarize_order9_c3_phase_reset_validation,
    summarize_order9_c3_stagewise_validation,
)
from amsrr.training.order9_evaluation import (
    Order9EvaluationEpisode,
    write_order9_evaluation_episodes_jsonl,
)
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
)


def _bucket(module_count: int, sample_index: int) -> Order9PiLRolloutBucket:
    return Order9PiLRolloutBucket(
        bucket_id=f"validation-{sample_index:06d}-m{module_count}",
        split=DatasetSplit.VALIDATION,
        seed=100 + sample_index,
        sample_index=sample_index,
        task_id=f"task-{sample_index}",
        task_spec_path=f"task-{sample_index}.json",
        task_spec_sha256="a" * 64,
        morphology_graph_path=f"graph-{sample_index}.json",
        morphology_graph_sha256="b" * 64,
        morphology_hash="c" * 64,
        structural_hash=f"{sample_index:064x}",
        module_count=module_count,
        robot_usd_path="robot.usda",
        robot_usd_sha256="d" * 64,
        selected_gripper_friction=4.5,
        contact_stiffness_n_per_m=7500.0,
        contact_damping_n_s_per_m=75.0,
        estimated_mass_kg=1.0,
        estimated_inertia_body=[0.01, 0.0, 0.0, 0.02, 0.0, 0.03],
        estimated_com_object=[0.0, 0.0, 0.0],
        randomization_version="unit",
        topology_source="unit",
    )


def _manifest() -> Order9PiLRolloutBucketManifest:
    validation = [
        _bucket(module_count, 100 * module_count + replica)
        for module_count in range(2, 9)
        for replica in range(2)
    ]
    train = _bucket(2, 1)
    train.split = DatasetSplit.TRAIN
    train.bucket_id = "train-000001-m2"
    return Order9PiLRolloutBucketManifest(
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        stage_config_hash="1" * 64,
        curriculum_schedule_hash="2" * 64,
        config_hash="3" * 64,
        physical_model_hash="4" * 64,
        topology_randomized=False,
        buckets=[train, *validation],
    )


def _episode(index: int, *, success: bool, safety: bool) -> Order9EvaluationEpisode:
    return Order9EvaluationEpisode(
        episode_id=f"episode-{index}",
        task_id="task",
        split=DatasetSplit.VALIDATION,
        random_seed=index,
        task_success=success,
        no_fallback_success=success,
        safety_failure=safety,
        high_level_decision_count=0,
        fallback_decision_count=0,
        environment_step_count=100,
        isaac_backed=True,
        full_mesh_evaluation=True,
        source_artifact_path="raw.pt",
        source_artifact_sha256="e" * 64,
        failure_reason=None if success else "task_failure",
        metrics={},
        metadata={},
    )


def test_stagewise_validation_selects_two_fixed_buckets_per_seen_count() -> None:
    ids = select_order9_c3_stagewise_validation_bucket_ids(
        _manifest(), maximum_module_count=4
    )
    assert len(ids) == 6
    assert ids == (
        "validation-000200-m2",
        "validation-000201-m2",
        "validation-000300-m3",
        "validation-000301-m3",
        "validation-000400-m4",
        "validation-000401-m4",
    )


def test_new_module_validation_selects_only_new_size() -> None:
    assert select_order9_c3_new_module_validation_bucket_ids(
        _manifest(), module_count=5
    ) == (
        "validation-000500-m5",
        "validation-000501-m5",
    )


def test_incremental_forgetting_compares_same_bucket_cycle() -> None:
    def summary(*, cycle: int, sha: str, reward_delta: float, phase_delta: float):
        return {
            "bucket_cycle_index": cycle,
            "evaluated_behavior_checkpoint_sha256": sha,
            "aggregates": {
                f"module_{module_count:02d}_train": {
                    "reward_mean": 4.0 + reward_delta,
                    "phase_success_rate": 0.5 + phase_delta,
                }
                for module_count in range(2, 5)
            },
        }

    passed = compare_order9_c3_incremental_forgetting(
        baseline_summary=summary(
            cycle=0, sha="a" * 64, reward_delta=0.0, phase_delta=0.0
        ),
        candidate_summary=summary(
            cycle=0, sha="b" * 64, reward_delta=-0.1, phase_delta=-0.02
        ),
        existing_maximum_module_count=4,
    )
    assert passed["passed"]
    failed = compare_order9_c3_incremental_forgetting(
        baseline_summary=summary(
            cycle=0, sha="a" * 64, reward_delta=0.0, phase_delta=0.0
        ),
        candidate_summary=summary(
            cycle=0, sha="b" * 64, reward_delta=-0.3, phase_delta=-0.04
        ),
        existing_maximum_module_count=4,
    )
    assert not failed["passed"]
    with pytest.raises(ValueError, match="different bucket cycles"):
        compare_order9_c3_incremental_forgetting(
            baseline_summary=summary(
                cycle=0, sha="a" * 64, reward_delta=0.0, phase_delta=0.0
            ),
            candidate_summary=summary(
                cycle=1, sha="b" * 64, reward_delta=0.0, phase_delta=0.0
            ),
            existing_maximum_module_count=4,
        )


def test_stagewise_validation_requires_every_episode_and_zero_safety_failures(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    bucket_ids = select_order9_c3_stagewise_validation_bucket_ids(
        manifest, maximum_module_count=3
    )
    for bucket_id in bucket_ids:
        write_order9_evaluation_episodes_jsonl(
            tmp_path / "buckets" / bucket_id / "evaluation_episodes.jsonl",
            [_episode(index, success=True, safety=False) for index in range(4)],
        )
    passed = summarize_order9_c3_stagewise_validation(
        manifest=manifest,
        maximum_module_count=3,
        expected_episode_count_per_bucket=4,
        evaluation_root=tmp_path,
    )
    assert passed.passed
    assert passed.success_count == 16

    failed_bucket = bucket_ids[-1]
    write_order9_evaluation_episodes_jsonl(
        tmp_path / "buckets" / failed_bucket / "evaluation_episodes.jsonl",
        [
            *[_episode(index, success=True, safety=False) for index in range(3)],
            _episode(3, success=False, safety=True),
        ],
    )
    failed = summarize_order9_c3_stagewise_validation(
        manifest=manifest,
        maximum_module_count=3,
        expected_episode_count_per_bucket=4,
        evaluation_root=tmp_path,
    )
    assert not failed.passed
    assert failed.success_count == 15
    assert failed.safety_failure_count == 1


def test_stagewise_validation_matches_c3_aggregate_gate_and_can_fail_fast(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    bucket_ids = select_order9_c3_stagewise_validation_bucket_ids(
        manifest, maximum_module_count=3
    )
    for bucket_index, bucket_id in enumerate(bucket_ids):
        failures = 3 if bucket_index == 0 else 0
        write_order9_evaluation_episodes_jsonl(
            tmp_path / "buckets" / bucket_id / "evaluation_episodes.jsonl",
            [
                _episode(
                    index,
                    success=index >= failures,
                    safety=False,
                )
                for index in range(4)
            ],
        )
    aggregate = summarize_order9_c3_stagewise_validation(
        manifest=manifest,
        maximum_module_count=3,
        expected_episode_count_per_bucket=4,
        minimum_success_rate=0.80,
        evaluation_root=tmp_path,
    )
    assert aggregate.success_count == 13
    assert aggregate.passed

    partial_root = tmp_path / "partial"
    write_order9_evaluation_episodes_jsonl(
        partial_root / "buckets" / bucket_ids[0] / "evaluation_episodes.jsonl",
        [
            _episode(index, success=index != 0, safety=index == 0)
            for index in range(4)
        ],
    )
    partial = summarize_order9_c3_stagewise_validation(
        manifest=manifest,
        maximum_module_count=3,
        expected_episode_count_per_bucket=4,
        minimum_success_rate=0.80,
        evaluation_root=partial_root,
        allow_mathematically_failed_partial=True,
    )
    assert not partial.complete
    assert partial.mathematically_failed
    assert not partial.passed


def test_continuous_validation_runs_periodically_and_at_stage_end() -> None:
    assert not should_run_order9_c3_continuous_validation(
        attempt_ordinal=1,
        interval_updates=3,
        maximum_attempt_count=13,
    )
    assert should_run_order9_c3_continuous_validation(
        attempt_ordinal=3,
        interval_updates=3,
        maximum_attempt_count=13,
    )
    assert should_run_order9_c3_continuous_validation(
        attempt_ordinal=13,
        interval_updates=3,
        maximum_attempt_count=13,
    )
    with pytest.raises(ValueError, match="schedule"):
        should_run_order9_c3_continuous_validation(
            attempt_ordinal=0,
            interval_updates=3,
            maximum_attempt_count=13,
        )


def test_quick_validation_is_bound_to_pre_update_parent_without_raw(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "already-cleaned-validation.pt"
    raw_sha = "9" * 64
    parent_sha = "8" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "behavior_checkpoint_sha256": parent_sha,
                "generation_id": "generation-2",
                "shards": [
                    {
                        "split": "validation",
                        "path": str(raw_path),
                        "sha256": raw_sha,
                        "module_count": 3,
                        "environment_count": 16,
                        "environment_step_count": 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "validation.log"
    log_path.write_text(
        "ORDER9_ROLLOUT_JSON="
        + json.dumps(
            {
                "split": "validation",
                "raw_artifact_sha256": raw_sha,
                "environment_steps": 64,
                "promotion_evidence_eligible": False,
                "terminal_count": 10,
                "successful_terminal_count": 7,
                "phase_transition_counts": {"lift->transport": 3},
                "aggregate_env_steps_per_s": 1234.0,
                "collection_wall_elapsed_s": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = summarize_order9_c3_phase_reset_validation(
        dataset_manifest_path=manifest_path,
        rollout_log_path=log_path,
    )
    assert summary["evaluated_checkpoint_sha256"] == parent_sha
    assert summary["evaluated_checkpoint_role"] == "pre_update_behavior_parent"
    assert summary["terminal_success_rate"] == pytest.approx(0.7)
    assert summary["raw_tensor_metrics_available"] is False
    assert summary["promotion_evidence_eligible"] is False


def test_intermediate_cleanup_is_scoped_and_records_hashes(tmp_path: Path) -> None:
    root = tmp_path / "lineage"
    raw = root / "generations" / "generation_000000" / "raw"
    raw.mkdir(parents=True)
    artifact = raw / "train.pt"
    artifact.write_bytes(b"raw")
    evidence = root / "cleanup.json"

    payload = cleanup_order9_intermediate_raw_artifacts(
        roots=(raw,),
        allowed_ancestor=root,
        evidence_path=evidence,
    )
    assert payload["file_count"] == 1
    assert len(payload["files"][0]["sha256"]) == 64
    assert not artifact.exists()
    assert evidence.is_file()

    with pytest.raises(ValueError, match="unsafe"):
        cleanup_order9_intermediate_raw_artifacts(
            roots=(tmp_path,),
            allowed_ancestor=root,
            evidence_path=evidence,
        )
