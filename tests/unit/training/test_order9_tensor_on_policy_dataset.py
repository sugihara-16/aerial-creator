from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_tensor_on_policy_dataset import (
    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION,
    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V1,
    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2,
    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3,
    ORDER9_STATE_INHERITANCE_ROLLOUT_MODE,
    Order9TensorPiLDatasetManifest,
    Order9TensorRolloutShard,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
)


def _shard(split: str, marker: str, *, suffix: str = "") -> Order9TensorRolloutShard:
    return Order9TensorRolloutShard(
        split=split,
        path=f"/{split}{suffix}.pt",
        sha256=marker if len(marker) == 64 else marker * 64,
        artifact_version=ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
        environment_count=4,
        rollout_step_count=8,
        environment_step_count=32,
        episode_count=4,
        task_count=4,
        morphology_hash="c" * 64,
        random_seed=1,
        simulator_version="unit",
        simulator_hash="d" * 64,
        collector_version="unit",
        robot_usd_sha256="e" * 64,
    )


def _manifest() -> Order9TensorPiLDatasetManifest:
    return Order9TensorPiLDatasetManifest(
        dataset_version=ORDER9_TENSOR_ON_POLICY_DATASET_VERSION,
        generation_id="generation-0",
        stage_id="c3",
        stage_config_hash="1" * 64,
        curriculum_schedule_hash="2" * 64,
        config_hash="3" * 64,
        physical_model_hash="4" * 64,
        urdf_hash="5" * 64,
        thrust_model_hash="6" * 64,
        behavior_checkpoint_sha256="7" * 64,
        environment_step_count=64,
        shards=[_shard("train", "a"), _shard("validation", "b")],
        metadata={
            "one_fresh_generation": True,
            "tensor_native_ppo": True,
            "record_schema_materialization": False,
            "jsonl_materialization": False,
        },
    )


def test_tensor_native_dataset_manifest_roundtrip() -> None:
    manifest = _manifest()
    manifest.validate()

    loaded = Order9TensorPiLDatasetManifest.from_json(manifest.to_json())

    assert loaded == manifest


def test_tensor_native_dataset_requires_train_validation_pair() -> None:
    manifest = _manifest()
    manifest.shards[1].split = "train"

    with pytest.raises(SchemaValidationError, match="train and validation"):
        manifest.validate()


def test_tensor_native_dataset_accepts_topology_stratified_train_shards() -> None:
    manifest = _manifest()
    manifest.dataset_version = ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2
    train = [
        _shard("train", format(module_count, "x"), suffix=f"_m{module_count}")
        for module_count in range(2, 9)
    ]
    validation = _shard("validation", "b")
    manifest.shards = [*train, validation]
    manifest.environment_step_count = sum(
        shard.environment_step_count for shard in manifest.shards
    )
    manifest.metadata.update(
        {
            "topology_stratified_train": True,
            "train_module_counts": list(range(2, 9)),
            "topologies_per_module_count": 1,
        }
    )

    manifest.validate()


def test_tensor_native_dataset_accepts_replicated_topology_strata() -> None:
    manifest = _manifest()
    manifest.dataset_version = ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3
    train = [
        _shard(
            "train",
            format(module_count * 2 + replica, "x")[-1],
            suffix=f"_m{module_count}_r{replica}",
        )
        for module_count in range(2, 9)
        for replica in range(2)
    ]
    manifest.shards = [*train, _shard("validation", "2")]
    manifest.environment_step_count = sum(
        shard.environment_step_count for shard in manifest.shards
    )
    manifest.metadata.update(
        {
            "topology_stratified_train": True,
            "train_module_counts": [
                value for value in range(2, 9) for _ in range(2)
            ],
            "topologies_per_module_count": 2,
        }
    )

    manifest.validate()


def test_tensor_native_dataset_accepts_mixed_state_inheritance_strata() -> None:
    manifest = _manifest()
    train = []
    marker_index = 1
    for module_count in range(2, 9):
        for replica in range(2):
            shard = _shard(
                "train",
                format(marker_index, "064x"),
                suffix=f"_m{module_count}_r{replica}",
            )
            marker_index += 1
            shard.module_count = module_count
            train.append(shard)
        inherited = _shard(
            "train",
            format(marker_index, "064x"),
            suffix=f"_m{module_count}_inherit",
        )
        marker_index += 1
        inherited.module_count = module_count
        inherited.rollout_mode = ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
        train.append(inherited)
    manifest.shards = [
        *train,
        _shard("validation", format(marker_index, "064x")),
    ]
    manifest.environment_step_count = sum(
        shard.environment_step_count for shard in manifest.shards
    )
    manifest.metadata.update(
        {
            "topology_stratified_train": True,
            "train_module_counts": [
                value for value in range(2, 9) for _ in range(3)
            ],
            "topologies_per_module_count": 2,
            "phase_reset_train_shard_count": 14,
            "state_inheritance_train_shard_count": 7,
            "phase_reset_topologies_per_module_count": 2,
            "state_inheritance_shards_per_module_count": 1,
        }
    )

    manifest.validate()


def test_tensor_native_dataset_accepts_focused_state_inheritance_strata() -> None:
    manifest = _manifest()
    train = []
    marker_index = 1
    inheritance_counts = {2: 0, 3: 0, 4: 0, 5: 4}
    for module_count in range(2, 6):
        for replica in range(2):
            shard = _shard(
                "train",
                format(marker_index, "064x"),
                suffix=f"_m{module_count}_r{replica}",
            )
            marker_index += 1
            shard.module_count = module_count
            train.append(shard)
        for replica in range(inheritance_counts[module_count]):
            inherited = _shard(
                "train",
                format(marker_index, "064x"),
                suffix=f"_m{module_count}_inherit_r{replica}",
            )
            marker_index += 1
            inherited.module_count = module_count
            inherited.rollout_mode = ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
            train.append(inherited)
    manifest.shards = [
        *train,
        _shard("validation", format(marker_index, "064x")),
    ]
    manifest.environment_step_count = sum(
        shard.environment_step_count for shard in manifest.shards
    )
    manifest.metadata.update(
        {
            "topology_stratified_train": True,
            "train_module_counts": [shard.module_count for shard in train],
            "topologies_per_module_count": 2,
            "phase_reset_train_shard_count": 8,
            "state_inheritance_train_shard_count": 4,
            "phase_reset_topologies_per_module_count": 2,
            "state_inheritance_shards_per_module_count": {
                str(key): value for key, value in inheritance_counts.items()
            },
        }
    )

    manifest.validate()


def test_tensor_native_v1_rejects_multiple_train_shards() -> None:
    manifest = _manifest()
    manifest.dataset_version = ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V1
    manifest.shards.insert(1, _shard("train", "c", suffix="_extra"))
    manifest.environment_step_count += manifest.shards[1].environment_step_count

    with pytest.raises(SchemaValidationError, match="v1"):
        manifest.validate()
