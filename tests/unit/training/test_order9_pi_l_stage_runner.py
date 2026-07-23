from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_ROLLOUT_RESULT_PREFIX,
    load_order9_rollout_result,
    required_order9_update_count,
    resolve_order9_extended_update_budget,
    select_order9_pi_l_rollout_buckets,
)
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
)


def _bucket(split: DatasetSplit, sample_index: int) -> Order9PiLRolloutBucket:
    return Order9PiLRolloutBucket(
        bucket_id=f"{split.value}-{sample_index}",
        split=split,
        seed=100 + sample_index,
        sample_index=sample_index,
        task_id=f"task-{split.value}-{sample_index}",
        task_spec_path=f"task-{sample_index}.json",
        task_spec_sha256="a" * 64,
        morphology_graph_path=f"graph-{sample_index}.json",
        morphology_graph_sha256="b" * 64,
        morphology_hash="c" * 64,
        structural_hash="d" * 64,
        module_count=3,
        robot_usd_path="robot.usda",
        robot_usd_sha256="e" * 64,
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
    config = load_order9_learning_config()
    return Order9PiLRolloutBucketManifest(
        stage_id="c2_pi_l_ppo_fixed_conservative",
        stage_config_hash="1" * 64,
        curriculum_schedule_hash="2" * 64,
        config_hash="3" * 64,
        physical_model_hash="4" * 64,
        topology_randomized=False,
        buckets=[
            _bucket(DatasetSplit.TRAIN, 2),
            _bucket(DatasetSplit.VALIDATION, 9),
            _bucket(DatasetSplit.TRAIN, 0),
            _bucket(DatasetSplit.VALIDATION, 8),
            _bucket(DatasetSplit.TRAIN, 1),
        ],
        metadata={"config_version": config.curriculum.schedule_version},
    )


def test_required_update_count_covers_c2_target() -> None:
    assert required_order9_update_count(3_000_000, 65_536) == 46
    assert 45 * 65_536 < 3_000_000 <= 46 * 65_536
    with pytest.raises(ValueError, match="positive"):
        required_order9_update_count(0, 65_536)


def test_extended_update_budget_preserves_configured_stage_target() -> None:
    assert resolve_order9_extended_update_budget(
        3_000_000,
        65_536,
        0,
    ) == (46, 46, 3_014_656)
    assert resolve_order9_extended_update_budget(
        3_000_000,
        65_536,
        4,
    ) == (46, 50, 3_276_800)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_order9_extended_update_budget(3_000_000, 65_536, -1)


def test_bucket_selection_rotates_each_split_independently() -> None:
    manifest = _manifest()
    selected = [select_order9_pi_l_rollout_buckets(manifest, index) for index in range(4)]
    assert [train.sample_index for train, _ in selected] == [0, 1, 2, 0]
    assert [validation.sample_index for _, validation in selected] == [8, 9, 8, 9]
    with pytest.raises(ValueError, match="non-negative"):
        select_order9_pi_l_rollout_buckets(manifest, -1)


def test_rollout_result_parser_uses_last_structured_result(tmp_path: Path) -> None:
    log = tmp_path / "rollout.log"
    expected = {"passed": True, "generation_id": "generation-1"}
    log.write_text(
        "startup\n"
        + ORDER9_ROLLOUT_RESULT_PREFIX
        + json.dumps({"passed": False})
        + "\n"
        + ORDER9_ROLLOUT_RESULT_PREFIX
        + json.dumps(expected)
        + "\n",
        encoding="utf-8",
    )
    assert load_order9_rollout_result(log) == expected
    missing = tmp_path / "missing.log"
    missing.write_text("startup only\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="missing"):
        load_order9_rollout_result(missing)
