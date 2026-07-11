from __future__ import annotations

import json
from pathlib import Path

from amsrr.acceptance.p4_3_acceptance import P4_3AcceptanceReport
from amsrr.logging.episode_archive import (
    read_episode_archives_jsonl,
    write_episode_archives_jsonl,
)
from amsrr.schemas.datasets import P4_3DatasetManifest
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.training.p4_0_full_pipeline_runner import (
    P4_0FullPipelineRunner,
    P4_0FullPipelineRunnerConfig,
)
from amsrr.training.p4_3_learning_archive import (
    validate_p4_3_learning_summary_archive,
    write_p4_3_learning_summary_archive,
)


def test_p4_3_learning_summary_is_sanitized_and_self_validating(tmp_path: Path) -> None:
    source_archive = P4_0FullPipelineRunner(
        default_grasp_carry_task_spec(),
        runner_config=P4_0FullPipelineRunnerConfig(episode_count=1, seed=3),
    ).run().archives[0]
    source_path = tmp_path / "source.jsonl"
    write_episode_archives_jsonl(source_path, [source_archive])
    manifest = P4_3DatasetManifest(
        dataset_id="summary-unit",
        schema_version="p4_3_dataset_v1",
        source_archive_paths=[str(source_path)],
        source_episode_ids=[source_archive.episode_id],
        train_task_ids=[source_archive.task_spec.task_id],
        validation_task_ids=[],
        held_out_task_ids=[],
        shards=[],
        record_counts={},
        source_hash="source",
        config_hash="config",
        robot_model_hash=source_archive.robot_model_hash,
        urdf_hash="urdf",
        thrust_model_hash="thrust",
        task_hashes={source_archive.task_spec.task_id: source_archive.task_hash},
        geometry_hashes=source_archive.geometry_hashes,
        random_seeds=[3],
        simulator_version="unit",
        simulator_hash="sim",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")

    policy_dirs = {name: tmp_path / name for name in ("pi_l", "pi_h", "pi_d")}
    for directory in policy_dirs.values():
        directory.mkdir()
        (directory / "checkpoint.pt").write_bytes(b"unit-checkpoint")
        (directory / "metrics.json").write_text("{}\n", encoding="utf-8")
        (directory / "fallback_metadata.json").write_text("{}\n", encoding="utf-8")
    (policy_dirs["pi_l"] / "reward_curve.csv").write_text(
        "episode,return\n1,0\n", encoding="utf-8"
    )
    (policy_dirs["pi_h"] / "rollout_evaluation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (policy_dirs["pi_d"] / "rollout_outcome_evaluation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    online_archive = policy_dirs["pi_l"] / "online.jsonl"
    online_archive.write_text('{"episode_id":"unit-online"}\n', encoding="utf-8")
    (policy_dirs["pi_l"] / "online_rollout_evaluation.json").write_text(
        json.dumps(
            {
                "archive_path": str(online_archive),
                "learned_decision_count": 2,
                "fallback_count": 0,
                "overlay_nonzero_count": 2,
                "object_drop_count": 0,
                "hard_collision_count": 0,
                "controller_qp_infeasible_terminal_count": 0,
            }
        ),
        encoding="utf-8",
    )
    acceptance = P4_3AcceptanceReport(
        dataset_passed=True,
        pi_l_passed=True,
        pi_h_passed=True,
        pi_d_passed=True,
        deterministic_fallbacks_passed=True,
        no_mislabeling_passed=True,
        completion_passed=True,
    )
    output_path = tmp_path / "summary.jsonl"

    write_p4_3_learning_summary_archive(
        source_rollout_archive_path=source_path,
        output_path=output_path,
        dataset_manifest_path=manifest_path,
        pi_l_dir=policy_dirs["pi_l"],
        pi_h_dir=policy_dirs["pi_h"],
        pi_d_dir=policy_dirs["pi_d"],
        acceptance=acceptance,
    )

    summary = read_episode_archives_jsonl(output_path)[0]
    assert validate_p4_3_learning_summary_archive(output_path) is True
    assert summary.task_spec.metadata["p4_phase"] == "P4.3"
    assert summary.rollout_artifacts["p4_3_dataset_collection"] is False
    assert summary.metrics["p4_3_pi_l_checkpoint_loaded"] == 1.0
    assert summary.policy_commands == []
    assert summary.runtime_observations == []
