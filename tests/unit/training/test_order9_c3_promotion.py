from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import (
    ORDER9_STAGE_RUN_VERSION,
    Order9ArtifactBinding,
    Order9StageRunManifest,
    Order9StageRunStatus,
)
from amsrr.training.order9_c3_promotion import (
    load_order9_evaluation_episode_jsonl,
    measured_order9_training_rollout_throughput,
    order9_c3_promotion_success_upper_bound,
    resolve_order9_training_generation_lineage,
    select_order9_c3_promotion_buckets,
)
from amsrr.utils.hashing import hash_file
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucketManifest,
    load_order9_pi_l_rollout_bucket_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPOSITORY_ROOT / (
    "artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/"
    "rollout_buckets_current_lineage_v5/manifest_c3_nominal_replay_v1.json"
)


def test_c3_promotion_selects_two_validation_buckets_per_topology() -> None:
    manifest = load_order9_pi_l_rollout_bucket_manifest(MANIFEST)
    selected = select_order9_c3_promotion_buckets(manifest)
    assert len(selected) == 14
    assert [bucket.module_count for bucket in selected] == [
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
    ]
    assert all(bucket.split == DatasetSplit.VALIDATION for bucket in selected)


def test_c3_promotion_rejects_incomplete_validation_matrix() -> None:
    manifest = load_order9_pi_l_rollout_bucket_manifest(MANIFEST)
    incomplete = Order9PiLRolloutBucketManifest.from_dict(manifest.to_dict())
    incomplete.buckets = incomplete.buckets[:-1]
    with pytest.raises(SchemaValidationError, match="exactly two"):
        select_order9_c3_promotion_buckets(incomplete)


def test_evaluation_jsonl_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="empty"):
        load_order9_evaluation_episode_jsonl(path)


def test_c3_promotion_success_upper_bound_fails_fast_when_unreachable() -> None:
    result = order9_c3_promotion_success_upper_bound(
        completed_success_count=122,
        completed_episode_count=224,
        total_episode_count=448,
        minimum_success_rate=0.80,
    )
    assert result is not None
    assert result["maximum_possible_success_count"] == 346
    assert result["maximum_possible_success_rate"] == pytest.approx(346 / 448)


def test_c3_promotion_success_upper_bound_keeps_reachable_run() -> None:
    assert order9_c3_promotion_success_upper_bound(
        completed_success_count=90,
        completed_episode_count=96,
        total_episode_count=448,
        minimum_success_rate=0.80,
    ) is None


def test_training_generation_lineage_follows_parent_across_branches(
    tmp_path: Path,
) -> None:
    checkpoint_paths: list[Path] = []
    manifest_paths: list[Path] = []
    for update, branch in enumerate(("base", "branch_a", "branch_b")):
        generation = tmp_path / branch / "generations" / f"generation_{update:06d}"
        dataset = generation / "dataset" / "manifest.json"
        dataset.parent.mkdir(parents=True)
        dataset.write_text(f'{{"update": {update}}}\n', encoding="utf-8")
        update_root = tmp_path / branch / f"update_{update:06d}"
        update_root.mkdir(parents=True)
        checkpoint = update_root / f"checkpoint_update_{update:06d}.pt"
        checkpoint.write_bytes(f"checkpoint-{update}".encode())
        inputs = [
            Order9ArtifactBinding(
                artifact_kind="dataset_manifest",
                path=str(dataset),
                sha256=hash_file(dataset),
            )
        ]
        if update:
            inputs.append(
                Order9ArtifactBinding(
                    artifact_kind="parent_checkpoint",
                    path=str(checkpoint_paths[-1]),
                    sha256=hash_file(checkpoint_paths[-1]),
                )
            )
        manifest = Order9StageRunManifest(
            run_version=ORDER9_STAGE_RUN_VERSION,
            run_id=f"run-{update}",
            stage_id="c3_pi_l_ppo_arbitrary_morphology",
            stage_index=3,
            status=Order9StageRunStatus.RUNNING,
            schedule_hash="1" * 64,
            stage_config_hash="2" * 64,
            runtime_config_hash="3" * 64,
            random_seed=update,
            device="cpu",
            environment_count=1,
            input_artifacts=inputs,
            output_artifacts=[
                Order9ArtifactBinding(
                    artifact_kind="policy_checkpoint",
                    path=str(checkpoint),
                    sha256=hash_file(checkpoint),
                )
            ],
            policy_checkpoint_sha256_by_family={"pi_l": hash_file(checkpoint)},
        )
        manifest_path = update_root / "stage_training_complete.json"
        manifest_path.write_text(manifest.to_json(indent=2) + "\n", encoding="utf-8")
        checkpoint_paths.append(checkpoint)
        manifest_paths.append(manifest_path)

    resolved = resolve_order9_training_generation_lineage(
        manifest_paths[-1],
        through_update=2,
    )
    assert resolved == tuple(
        (
            tmp_path
            / branch
            / "generations"
            / f"generation_{update:06d}"
        ).resolve()
        for update, branch in enumerate(("base", "branch_a", "branch_b"))
    )


def test_training_generation_lineage_follows_hash_bound_checkpoint_transforms(
    tmp_path: Path,
) -> None:
    checkpoint_paths: list[Path] = []
    manifest_paths: list[Path] = []
    for update in range(2):
        generation = tmp_path / "ppo" / "generations" / f"generation_{update:06d}"
        dataset = generation / "dataset" / "manifest.json"
        dataset.parent.mkdir(parents=True)
        dataset.write_text(f'{{"update": {update}}}\n', encoding="utf-8")
        update_root = tmp_path / "ppo" / f"update_{update:06d}"
        update_root.mkdir(parents=True)
        checkpoint = update_root / f"checkpoint_update_{update:06d}.pt"
        checkpoint.write_bytes(f"checkpoint-{update}".encode())
        inputs = [
            Order9ArtifactBinding(
                artifact_kind="dataset_manifest",
                path=str(dataset),
                sha256=hash_file(dataset),
            )
        ]
        if update:
            transformed = tmp_path / "teacher_b" / "checkpoint.pt"
            inputs.append(
                Order9ArtifactBinding(
                    artifact_kind="parent_checkpoint",
                    path=str(transformed),
                    sha256=hash_file(transformed),
                )
            )
        manifest = Order9StageRunManifest(
            run_version=ORDER9_STAGE_RUN_VERSION,
            run_id=f"run-{update}",
            stage_id="c3_pi_l_ppo_arbitrary_morphology",
            stage_index=3,
            status=Order9StageRunStatus.RUNNING,
            schedule_hash="1" * 64,
            stage_config_hash="2" * 64,
            runtime_config_hash="3" * 64,
            random_seed=update,
            device="cpu",
            environment_count=1,
            input_artifacts=inputs,
            output_artifacts=[
                Order9ArtifactBinding(
                    artifact_kind="policy_checkpoint",
                    path=str(checkpoint),
                    sha256=hash_file(checkpoint),
                )
            ],
            policy_checkpoint_sha256_by_family={"pi_l": hash_file(checkpoint)},
        )
        manifest_path = update_root / "stage_training_complete.json"
        manifest_path.write_text(manifest.to_json(indent=2) + "\n", encoding="utf-8")
        checkpoint_paths.append(checkpoint)
        manifest_paths.append(manifest_path)

        if update == 0:
            parent = checkpoint
            parent_sha256 = hash_file(parent)
            for name in ("teacher_a", "teacher_b"):
                root = tmp_path / name
                root.mkdir()
                output = root / "checkpoint.pt"
                output.write_bytes(f"{name}-checkpoint".encode())
                report = {
                    "output_checkpoint": str(output),
                    "output_checkpoint_sha256": hash_file(output),
                    "parent_checkpoint": str(parent),
                    "parent_checkpoint_sha256": parent_sha256,
                }
                (root / "teacher_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                parent = output
                parent_sha256 = hash_file(parent)

    resolved = resolve_order9_training_generation_lineage(
        manifest_paths[-1], through_update=1
    )
    assert resolved == tuple(
        (
            tmp_path
            / "ppo"
            / "generations"
            / f"generation_{update:06d}"
        ).resolve()
        for update in range(2)
    )


def _write_rollout_log(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ORDER9_ROLLOUT_JSON=" + json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_training_throughput_uses_only_production_width_collectors(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "generation_000000"
    unrelated = generation / "logs" / "reward_summary.log"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('{"mean_reward": 1.0}\n', encoding="utf-8")
    _write_rollout_log(
        generation / "logs" / "validation.log",
        environment_count=1024,
        environment_steps=262_144,
        collection_wall_elapsed_s=128.0,
    )
    _write_rollout_log(
        generation / "logs" / "train_inherit_m08.log",
        environment_count=12,
        environment_steps=15_360,
        collection_wall_elapsed_s=320.0,
    )

    steps, wall = measured_order9_training_rollout_throughput(
        [generation],
        required_environment_count=1024,
    )

    assert steps == 262_144
    assert wall == pytest.approx(128.0)
    assert steps / wall == pytest.approx(2048.0)


def test_training_throughput_requires_production_width_evidence(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "generation_000000"
    _write_rollout_log(
        generation / "logs" / "train_inherit_m08.log",
        environment_count=12,
        environment_steps=15_360,
        collection_wall_elapsed_s=320.0,
    )

    with pytest.raises(SchemaValidationError, match="production-width"):
        measured_order9_training_rollout_throughput(
            [generation],
            required_environment_count=1024,
        )
