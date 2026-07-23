from __future__ import annotations

"""C3 runtime selection from production collector and concurrent-load evidence."""

import math
from pathlib import Path
from typing import Sequence

import torch

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_curriculum import Order9LearningConfig
from amsrr.training.order9_production_benchmark import (
    build_order9_production_benchmark_report,
)
from amsrr.training.order9_runtime_benchmark import (
    Order9RuntimeBenchmarkReport,
    write_order9_runtime_benchmark_report,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    Order9TensorRolloutArtifact,
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


ORDER9_C3_RUNTIME_SELECTION_VERSION = (
    "order9_c3_concurrent_production_runtime_selection_v1"
)


def build_order9_c3_runtime_benchmark_report(
    config: Order9LearningConfig,
    *,
    stage_id: str,
    pi_l_checkpoint_sha256: str,
    throughput_artifact_paths: Sequence[str | Path],
    concurrent_artifact_paths: Sequence[str | Path],
    rejected_capacity_artifact_paths: Sequence[str | Path],
    bucket_manifest_path: str | Path,
    order8_report_path: str | Path,
    output_path: str | Path,
    maximum_gpu_memory_fraction: float = 0.85,
) -> Order9RuntimeBenchmarkReport:
    """Bind single-process throughput and production-sized concurrent capacity."""

    if (
        not math.isfinite(maximum_gpu_memory_fraction)
        or not 0.0 < maximum_gpu_memory_fraction < 1.0
    ):
        raise ValueError("maximum_gpu_memory_fraction must lie in (0, 1)")
    report = build_order9_production_benchmark_report(
        config.runtime_benchmark,
        raw_artifact_paths=throughput_artifact_paths,
        expected_stage_id=stage_id,
        expected_checkpoint_sha256=pi_l_checkpoint_sha256,
        order8_report_path=order8_report_path,
    )
    selected = report.selected_environment_count
    if (
        selected is None
        or selected != config.production_runtime.selected_environment_count
    ):
        raise SchemaValidationError(
            "Order9 C3 throughput selection differs from production runtime"
        )
    production_steps = config.optimization.pi_l_ppo.rollout_steps_per_environment
    concurrent = _concurrent_capacity_evidence(
        concurrent_artifact_paths,
        expected_stage_id=stage_id,
        expected_checkpoint_sha256=pi_l_checkpoint_sha256,
        expected_environment_count=selected,
        expected_rollout_steps=production_steps,
        maximum_gpu_memory_fraction=maximum_gpu_memory_fraction,
    )
    rejected = [
        _artifact_capacity_summary(
            path,
            expected_stage_id=stage_id,
            expected_checkpoint_sha256=pi_l_checkpoint_sha256,
        )
        for path in rejected_capacity_artifact_paths
    ]
    if any(
        int(value["environment_count"]) <= selected
        for value in rejected
    ):
        raise SchemaValidationError(
            "Order9 C3 rejected capacity must exceed the selected count"
        )
    bucket_manifest = Path(bucket_manifest_path).resolve()
    report.metadata.update(
        {
            "c3_runtime_selection_version": (
                ORDER9_C3_RUNTIME_SELECTION_VERSION
            ),
            "selection_scope": (
                "eight_module_worst_case_single_throughput_plus_"
                "train_validation_concurrent_production_rollout"
            ),
            "selected_environment_count_per_process": selected,
            "concurrent_process_count": 2,
            "concurrent_capacity_evidence": concurrent,
            "maximum_gpu_memory_fraction": maximum_gpu_memory_fraction,
            "rejected_capacity_diagnostics": rejected,
            "bucket_manifest_path": str(bucket_manifest),
            "bucket_manifest_sha256": hash_file(bucket_manifest),
            "selection_reason": (
                "maximum throughput-eligible count with a completed "
                "production-sized two-process capacity check"
            ),
        }
    )
    report.validate()
    write_order9_runtime_benchmark_report(output_path, report)
    return report


def _concurrent_capacity_evidence(
    paths: Sequence[str | Path],
    *,
    expected_stage_id: str,
    expected_checkpoint_sha256: str,
    expected_environment_count: int,
    expected_rollout_steps: int,
    maximum_gpu_memory_fraction: float,
) -> dict[str, object]:
    if len(paths) != 2:
        raise SchemaValidationError(
            "Order9 C3 concurrent capacity requires exactly two artifacts"
        )
    summaries = [
        _artifact_capacity_summary(
            path,
            expected_stage_id=expected_stage_id,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        for path in paths
    ]
    if {
        str(summary["split"]) for summary in summaries
    } != {DatasetSplit.TRAIN.value, DatasetSplit.VALIDATION.value}:
        raise SchemaValidationError(
            "Order9 C3 concurrent capacity requires train and validation"
        )
    for summary in summaries:
        if (
            int(summary["environment_count"])
            != expected_environment_count
            or int(summary["rollout_steps"]) != expected_rollout_steps
            or int(summary["module_count"]) != 8
        ):
            raise SchemaValidationError(
                "Order9 C3 concurrent artifact differs from production capacity"
            )
    if len({str(value["structural_hash"]) for value in summaries}) != 2:
        raise SchemaValidationError(
            "Order9 C3 concurrent capacity must use distinct topologies"
        )
    peak_memory = max(
        float(summary["global_gpu_memory_used_peak_mib"])
        for summary in summaries
    )
    total_memory = min(
        float(summary["global_gpu_memory_total_mib"])
        for summary in summaries
    )
    fraction = peak_memory / total_memory
    if fraction > maximum_gpu_memory_fraction:
        raise SchemaValidationError(
            "Order9 C3 concurrent capacity exceeds the GPU memory gate"
        )
    slower_collection = max(
        float(summary["collection_wall_elapsed_s"]) for summary in summaries
    )
    slower_rollout = max(
        float(summary["rollout_wall_elapsed_s"]) for summary in summaries
    )
    total_environment_steps = sum(
        int(summary["environment_steps"]) for summary in summaries
    )
    return {
        "passed": True,
        "artifacts": summaries,
        "global_gpu_memory_used_peak_mib": peak_memory,
        "global_gpu_memory_total_mib": total_memory,
        "global_gpu_memory_fraction_peak": fraction,
        "global_gpu_memory_headroom_mib": total_memory - peak_memory,
        "conservative_pair_end_to_end_env_steps_per_s": (
            total_environment_steps / slower_collection
        ),
        "conservative_pair_rollout_env_steps_per_s": (
            total_environment_steps / slower_rollout
        ),
        "total_environment_steps": total_environment_steps,
    }


def _artifact_capacity_summary(
    path: str | Path,
    *,
    expected_stage_id: str,
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    source = Path(path).resolve()
    digest = hash_file(source)
    artifact = load_order9_tensor_rollout_artifact(
        source,
        expected_sha256=digest,
    )
    metadata = artifact.metadata
    if (
        metadata.get("stage_id") != expected_stage_id
        or metadata.get("pi_l_checkpoint_sha256")
        != expected_checkpoint_sha256
        or metadata.get("topology_randomized") is not True
    ):
        raise SchemaValidationError(
            "Order9 C3 capacity artifact identity differs"
        )
    morphology = MorphologyGraph.from_dict(metadata["morphology_graph"])
    splits = {
        str(value) for value in metadata.get("environment_splits", [])
    }
    if len(splits) != 1:
        raise SchemaValidationError(
            "Order9 C3 capacity artifact must contain one split"
        )
    _require_finite_critical_state(artifact)
    runtime_load = metadata.get("runtime_load")
    if not isinstance(runtime_load, dict):
        raise SchemaValidationError(
            "Order9 C3 capacity artifact lacks runtime load"
        )
    required_load = (
        "gpu_memory_used_mib_peak",
        "gpu_memory_total_mib_peak",
        "gpu_utilization_percent_peak",
        "process_rss_mib_peak",
    )
    if any(name not in runtime_load for name in required_load):
        raise SchemaValidationError(
            "Order9 C3 capacity artifact lacks required load fields"
        )
    return {
        "path": str(source),
        "sha256": digest,
        "generation_id": str(metadata["generation_id"]),
        "split": next(iter(splits)),
        "environment_count": artifact.environment_count,
        "rollout_steps": artifact.step_count,
        "environment_steps": artifact.environment_step_count,
        "module_count": len(morphology.modules),
        "structural_hash": morphology_structural_hash(morphology),
        "setup_wall_elapsed_s": float(metadata["setup_wall_elapsed_s"]),
        "rollout_wall_elapsed_s": float(metadata["rollout_wall_elapsed_s"]),
        "collection_wall_elapsed_s": float(
            metadata["collection_wall_elapsed_s"]
        ),
        "aggregate_env_steps_per_s": float(
            metadata["aggregate_env_steps_per_s"]
        ),
        "end_to_end_env_steps_per_s": float(
            metadata["end_to_end_env_steps_per_s"]
        ),
        "global_gpu_memory_used_peak_mib": float(
            runtime_load["gpu_memory_used_mib_peak"]
        ),
        "global_gpu_memory_total_mib": float(
            runtime_load["gpu_memory_total_mib_peak"]
        ),
        "global_gpu_utilization_peak_percent": float(
            runtime_load["gpu_utilization_percent_peak"]
        ),
        "process_rss_peak_mib": float(runtime_load["process_rss_mib_peak"]),
        "finite_critical_state": True,
    }


def _require_finite_critical_state(
    artifact: Order9TensorRolloutArtifact,
) -> None:
    valid = artifact.tensors["valid"]
    for name in (
        "module_pose_world",
        "object_pose_world",
        "local_joint_positions_rad",
        "global_action",
    ):
        values = artifact.tensors[name][valid]
        if not bool(torch.isfinite(values).all()):
            raise SchemaValidationError(
                f"Order9 C3 capacity artifact contains non-finite {name}"
            )


__all__ = [
    "ORDER9_C3_RUNTIME_SELECTION_VERSION",
    "build_order9_c3_runtime_benchmark_report",
]
