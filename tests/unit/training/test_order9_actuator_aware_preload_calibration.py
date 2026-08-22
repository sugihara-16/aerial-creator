from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from amsrr.training.order9_actuator_aware_preload_calibration import (
    Order9ActuatorAwarePreloadCalibrationConfig,
    calibrate_order9_actuator_aware_preload_margin,
)


def _write_shard(path: Path, *, split: str, gaps_m: tuple[float, float]) -> None:
    feature_names = [f"feature_{index}" for index in range(30)] + [
        "feedback.signed_surface_distance_over_normal_span"
    ]
    features = torch.zeros(2, 2, 2, len(feature_names))
    features[-1, 0, :, -1] = gaps_m[0] / 0.020
    features[-1, 1, :, -1] = gaps_m[1] / 0.020
    artifact = {
        "metadata": {
            "evaluation_mode": split != "train",
            "environment_splits": [split, split],
            "contact_space_feature_names": feature_names,
            "contact_normal_action_category_count": 81,
            "contact_normal_action_category_step_m": 0.0005,
            "selected_anchor_ids": [0, 1],
        },
        "tensors": {
            "valid": torch.ones(2, 2, dtype=torch.bool),
            "phase_index": torch.ones(2, 2, dtype=torch.long),
            "phase_progress": torch.tensor([[0.8, 0.8], [0.95, 0.95]]),
            "actor_contact_slot_features": features,
            "contact_space_residual_action": torch.zeros(2, 2, 2, 6),
        },
    }
    torch.save(artifact, path)


def test_calibration_uses_only_train_shards_and_quantizes_up(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    validation = tmp_path / "validation.pt"
    _write_shard(train, split="train", gaps_m=(0.0041, 0.0062))
    _write_shard(validation, split="validation", gaps_m=(0.019, 0.019))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "generation_id": "generation:test",
                "behavior_checkpoint_sha256": "actor",
                "shards": [
                    {
                        "path": str(train),
                        "sha256": "train-hash",
                        "split": "train",
                        "module_count": 2,
                        "rollout_mode": "phase_reset",
                    },
                    {
                        "path": str(validation),
                        "sha256": "validation-hash",
                        "split": "validation",
                        "module_count": 2,
                        "rollout_mode": "evaluation",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = calibrate_order9_actuator_aware_preload_margin(
        dataset_manifest_path=manifest,
        expected_module_counts=(2,),
        config=Order9ActuatorAwarePreloadCalibrationConfig(
            calibration_quantile=0.90,
            quantization_m=0.001,
        ),
    )

    assert report["split_contract"] == "train_only_no_held_out_validation"
    assert report["sample_count"] == 2
    assert report["raw_quantile_m"] == pytest.approx(0.00599)
    assert report["calibrated_model_error_margin_m"] == pytest.approx(0.006)
    assert len(report["train_shards"]) == 1


def test_calibration_rejects_missing_module_stratum(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    _write_shard(train, split="train", gaps_m=(0.004, 0.004))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "path": str(train),
                        "sha256": "train-hash",
                        "split": "train",
                        "module_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="module strata differ"):
        calibrate_order9_actuator_aware_preload_margin(
            dataset_manifest_path=manifest,
            expected_module_counts=(2, 3),
        )


def test_calibration_uses_worst_module_stratum_quantile(tmp_path: Path) -> None:
    module_2 = tmp_path / "module_2.pt"
    module_8 = tmp_path / "module_8.pt"
    _write_shard(module_2, split="train", gaps_m=(0.001, 0.002))
    _write_shard(module_8, split="train", gaps_m=(0.010, 0.012))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "path": str(module_2),
                        "sha256": "module-2-hash",
                        "split": "train",
                        "module_count": 2,
                    },
                    {
                        "path": str(module_8),
                        "sha256": "module-8-hash",
                        "split": "train",
                        "module_count": 8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = calibrate_order9_actuator_aware_preload_margin(
        dataset_manifest_path=manifest,
        expected_module_counts=(2, 8),
        config=Order9ActuatorAwarePreloadCalibrationConfig(
            calibration_quantile=0.90,
            quantization_m=0.001,
        ),
    )

    assert report["aggregation_contract"] == (
        "maximum_within_module_stratum_quantile"
    )
    assert report["limiting_module_count"] == 8
    assert report["pooled_quantile_m_diagnostic_only"] == pytest.approx(0.0114)
    assert report["raw_quantile_m"] == pytest.approx(0.0118)
    assert report["calibrated_model_error_margin_m"] == pytest.approx(0.012)
