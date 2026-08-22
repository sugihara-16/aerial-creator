from __future__ import annotations

"""Train-split calibration for the Order 9 nominal contact preload model."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from amsrr.utils.hashing import hash_file


ORDER9_ACTUATOR_AWARE_PRELOAD_CALIBRATION_VERSION = (
    "order9_actuator_aware_preload_train_gap_calibration_v2_stratified"
)
_SIGNED_DISTANCE_FEATURE = "feedback.signed_surface_distance_over_normal_span"


@dataclass(frozen=True)
class Order9ActuatorAwarePreloadCalibrationConfig:
    contact_phase_index: int = 1
    minimum_phase_progress: float = 0.90
    residual_history_steps: int = 10
    calibration_quantile: float = 0.90
    quantization_m: float = 0.001
    clipped_feature_threshold: float = 1.999

    def validate(self) -> None:
        if self.contact_phase_index < 0:
            raise ValueError("preload calibration phase index must be non-negative")
        if not 0.0 <= self.minimum_phase_progress <= 1.0:
            raise ValueError("preload calibration progress must be in [0, 1]")
        if self.residual_history_steps <= 0:
            raise ValueError("preload calibration history must be positive")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ValueError("preload calibration quantile must be in (0, 1)")
        if not math.isfinite(self.quantization_m) or self.quantization_m <= 0.0:
            raise ValueError("preload calibration quantization must be positive")
        if (
            not math.isfinite(self.clipped_feature_threshold)
            or not 0.0 < self.clipped_feature_threshold <= 2.0
        ):
            raise ValueError("preload calibration clipping threshold is invalid")


def calibrate_order9_actuator_aware_preload_margin(
    *,
    dataset_manifest_path: str | Path,
    expected_module_counts: Sequence[int] = tuple(range(2, 9)),
    config: Order9ActuatorAwarePreloadCalibrationConfig | None = None,
) -> dict[str, object]:
    """Estimate one common additive lead margin from train rollout shards.

    For each train environment, the final late-contact sample is converted to
    the equivalent surface gap at zero learned normal residual:

        realized signed gap + recent mean inward residual.

    The quantile is computed independently for every module-count stratum and
    the largest stratum quantile is rounded upward to the configured lead
    quantum.  This keeps one common runtime margin while preventing the more
    numerous/easier strata from hiding a high-module-count closure error.
    Clipped distance observations are discarded because they carry no metric
    information about the missing closure distance.
    """

    resolved = config or Order9ActuatorAwarePreloadCalibrationConfig()
    resolved.validate()
    manifest_path = Path(dataset_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("preload calibration dataset manifest lacks shards")
    train_shards = [shard for shard in shards if shard.get("split") == "train"]
    if not train_shards:
        raise ValueError("preload calibration dataset has no train shards")
    if any(shard.get("split") != "train" for shard in train_shards):  # pragma: no cover
        raise AssertionError("preload calibration selected a non-train shard")

    expected = tuple(sorted({int(value) for value in expected_module_counts}))
    observed = tuple(sorted({int(shard["module_count"]) for shard in train_shards}))
    if observed != expected:
        raise ValueError(
            "preload calibration module strata differ: "
            f"observed={observed}, expected={expected}"
        )

    candidates_by_module: dict[int, list[float]] = {
        module_count: [] for module_count in observed
    }
    shard_records: list[dict[str, object]] = []
    for shard in train_shards:
        path = Path(str(shard["path"])).resolve()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        metadata = artifact.get("metadata")
        tensors = artifact.get("tensors")
        if not isinstance(metadata, dict) or not isinstance(tensors, dict):
            raise ValueError(f"preload calibration shard is malformed: {path}")
        if bool(metadata.get("evaluation_mode")):
            raise ValueError("preload calibration refuses evaluation rollout data")
        splits = metadata.get("environment_splits")
        if not isinstance(splits, list) or not splits or set(splits) != {"train"}:
            raise ValueError("preload calibration shard is not train-only")
        feature_names = metadata.get("contact_space_feature_names")
        if not isinstance(feature_names, list) or _SIGNED_DISTANCE_FEATURE not in feature_names:
            raise ValueError("preload calibration lacks signed-distance feedback")
        feature_index = feature_names.index(_SIGNED_DISTANCE_FEATURE)
        category_count = int(metadata["contact_normal_action_category_count"])
        category_step_m = float(metadata["contact_normal_action_category_step_m"])
        normal_span_m = category_step_m * float(category_count - 1) / 2.0
        if not math.isfinite(normal_span_m) or normal_span_m <= 0.0:
            raise ValueError("preload calibration contact-normal span is invalid")
        selected_anchor_ids = metadata.get("selected_anchor_ids")
        if not isinstance(selected_anchor_ids, list) or len(selected_anchor_ids) < 2:
            raise ValueError("preload calibration requires selected contacts")
        selected_count = len(selected_anchor_ids)

        valid = tensors["valid"].bool()
        phase = tensors["phase_index"]
        progress = tensors["phase_progress"]
        features = tensors["actor_contact_slot_features"]
        actions = tensors["contact_space_residual_action"]
        if valid.ndim != 2 or phase.shape != valid.shape or progress.shape != valid.shape:
            raise ValueError("preload calibration rollout axes differ")
        shard_candidates: list[float] = []
        clipped_count = 0
        for environment_index in range(valid.shape[1]):
            indices = torch.where(
                valid[:, environment_index]
                & (phase[:, environment_index] == resolved.contact_phase_index)
                & (
                    progress[:, environment_index]
                    >= resolved.minimum_phase_progress
                )
            )[0]
            if indices.numel() == 0:
                continue
            final_index = int(indices[-1])
            normalized_gap = features[
                final_index,
                environment_index,
                :selected_count,
                feature_index,
            ]
            if bool(
                (normalized_gap.abs() >= resolved.clipped_feature_threshold).any()
            ):
                clipped_count += 1
                continue
            history = indices[-min(resolved.residual_history_steps, indices.numel()) :]
            mean_normal_residual_m = actions[
                history,
                environment_index,
                :selected_count,
                0,
            ].mean(dim=0) * normal_span_m
            zero_residual_equivalent_gap_m = torch.max(
                normalized_gap * normal_span_m + mean_normal_residual_m
            )
            candidate = max(0.0, float(zero_residual_equivalent_gap_m))
            if not math.isfinite(candidate):
                raise ValueError("preload calibration produced a non-finite gap")
            shard_candidates.append(candidate)
        module_count = int(shard["module_count"])
        candidates_by_module[module_count].extend(shard_candidates)
        shard_records.append(
            {
                "path": str(path),
                "sha256": str(shard["sha256"]),
                "module_count": module_count,
                "rollout_mode": str(shard.get("rollout_mode")),
                "eligible_sample_count": len(shard_candidates),
                "clipped_sample_count": clipped_count,
            }
        )

    pooled = [
        value
        for module_count in observed
        for value in candidates_by_module[module_count]
    ]
    if not pooled or any(not values for values in candidates_by_module.values()):
        raise ValueError("preload calibration has an empty module stratum")
    module_summary: dict[str, dict[str, object]] = {}
    quantile_by_module: dict[int, float] = {}
    for module_count, values in candidates_by_module.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        stratum_quantile_m = float(
            torch.quantile(tensor, resolved.calibration_quantile)
        )
        quantile_by_module[module_count] = stratum_quantile_m
        module_summary[str(module_count)] = {
            "sample_count": len(values),
            "median_zero_residual_equivalent_gap_m": float(torch.quantile(tensor, 0.5)),
            "calibration_quantile_zero_residual_equivalent_gap_m": (
                stratum_quantile_m
            ),
        }
    pooled_tensor = torch.tensor(pooled, dtype=torch.float64)
    pooled_quantile_m = float(
        torch.quantile(pooled_tensor, resolved.calibration_quantile)
    )
    limiting_module_count = max(quantile_by_module, key=quantile_by_module.get)
    raw_quantile_m = quantile_by_module[limiting_module_count]
    quantum = float(resolved.quantization_m)
    calibrated_margin_m = quantum * math.ceil(
        (raw_quantile_m - 1.0e-12) / quantum
    )
    return {
        "calibration_version": ORDER9_ACTUATOR_AWARE_PRELOAD_CALIBRATION_VERSION,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": hash_file(manifest_path),
        "dataset_generation_id": manifest.get("generation_id"),
        "behavior_checkpoint_sha256": manifest.get("behavior_checkpoint_sha256"),
        "split_contract": "train_only_no_held_out_validation",
        "expected_module_counts": list(expected),
        "calibration_quantile": resolved.calibration_quantile,
        "minimum_phase_progress": resolved.minimum_phase_progress,
        "residual_history_steps": resolved.residual_history_steps,
        "quantization_m": quantum,
        "sample_count": len(pooled),
        "aggregation_contract": "maximum_within_module_stratum_quantile",
        "pooled_quantile_m_diagnostic_only": pooled_quantile_m,
        "limiting_module_count": limiting_module_count,
        "raw_quantile_m": raw_quantile_m,
        "calibrated_model_error_margin_m": calibrated_margin_m,
        "summary_by_module_count": module_summary,
        "train_shards": shard_records,
    }


__all__ = [
    "ORDER9_ACTUATOR_AWARE_PRELOAD_CALIBRATION_VERSION",
    "Order9ActuatorAwarePreloadCalibrationConfig",
    "calibrate_order9_actuator_aware_preload_margin",
]
