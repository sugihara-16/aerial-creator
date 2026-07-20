from __future__ import annotations

"""Build the runtime gate from complete production-collector artifacts."""

import math
from pathlib import Path
from typing import Sequence

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_curriculum import Order9RuntimeBenchmarkConfig
from amsrr.training.order9_runtime_benchmark import (
    ORDER9_RUNTIME_BENCHMARK_VERSION,
    Order9RuntimeBenchmarkReport,
    Order9RuntimeBenchmarkSample,
    load_order8_timing_reference,
    write_order9_runtime_benchmark_report,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_PRODUCTION_COLLECTOR_VERSION,
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PRODUCTION_BENCHMARK_BUILDER_VERSION = (
    "order9_production_collector_benchmark_builder_v1"
)


def build_order9_production_benchmark_report(
    config: Order9RuntimeBenchmarkConfig,
    *,
    raw_artifact_paths: Sequence[str | Path],
    expected_stage_id: str,
    expected_checkpoint_sha256: str,
    order8_report_path: str | Path,
    output_path: str | Path | None = None,
) -> Order9RuntimeBenchmarkReport:
    """Verify measured raw shards and produce the environment-count gate."""

    config.validate()
    if config.warmup_steps != 0:
        raise SchemaValidationError(
            "Order9 production artifact benchmark currently includes cold start; "
            "warmup_steps must be zero"
        )
    sources = tuple(Path(value).resolve() for value in raw_artifact_paths)
    if not sources or len(sources) != len(set(sources)):
        raise SchemaValidationError(
            "Order9 production benchmark requires unique raw artifacts"
        )
    by_environment_count: dict[int, Order9RuntimeBenchmarkSample] = {}
    source_hashes: dict[str, str] = {}
    generation_ids: dict[str, str] = {}
    for source in sources:
        digest = hash_file(source)
        artifact = load_order9_tensor_rollout_artifact(
            source, expected_sha256=digest
        )
        metadata = artifact.metadata
        if str(metadata.get("collector_version", "")) != (
            ORDER9_PRODUCTION_COLLECTOR_VERSION
        ):
            raise SchemaValidationError(
                "Order9 production benchmark collector version differs"
            )
        actuator_readback = metadata.get("actuator_readback")
        if not isinstance(actuator_readback, dict) or (
            actuator_readback.get("matches_physical_model") is not True
        ):
            raise SchemaValidationError(
                "Order9 production benchmark lacks actuator-limit readback"
            )
        object_readback = metadata.get("object_mass_properties_readback")
        if not isinstance(object_readback, dict) or (
            object_readback.get("matches_task_spec") is not True
        ):
            raise SchemaValidationError(
                "Order9 production benchmark lacks object-property readback"
            )
        if str(metadata["stage_id"]) != expected_stage_id:
            raise SchemaValidationError(
                "Order9 production benchmark stage identity differs"
            )
        if str(metadata["pi_l_checkpoint_sha256"]) != expected_checkpoint_sha256:
            raise SchemaValidationError(
                "Order9 production benchmark policy checkpoint differs"
            )
        environment_count = int(metadata.get("environment_count", 0))
        rollout_steps = int(metadata.get("rollout_steps", 0))
        elapsed = float(metadata.get("rollout_wall_elapsed_s", 0.0))
        recorded_throughput = float(
            metadata.get("aggregate_env_steps_per_s", 0.0)
        )
        if environment_count != artifact.environment_count:
            raise SchemaValidationError(
                "Order9 production benchmark environment count differs"
            )
        if rollout_steps != artifact.step_count or rollout_steps != config.measurement_steps:
            raise SchemaValidationError(
                "Order9 production benchmark measurement length differs"
            )
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise SchemaValidationError(
                "Order9 production benchmark elapsed time is invalid"
            )
        derived = artifact.environment_step_count / elapsed
        if not math.isclose(
            recorded_throughput,
            derived,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise SchemaValidationError(
                "Order9 production benchmark throughput metadata differs"
            )
        if environment_count in by_environment_count:
            raise SchemaValidationError(
                "Order9 production benchmark repeats an environment count"
            )
        by_environment_count[environment_count] = Order9RuntimeBenchmarkSample(
            environment_count=environment_count,
            attempted=True,
            isaac_backed=True,
            backend_version=str(metadata["simulator_version"]),
            device=str(metadata["device"]),
            warmup_steps=config.warmup_steps,
            measurement_steps=rollout_steps,
            wall_elapsed_s=elapsed,
            aggregate_env_steps_per_s=derived,
            per_environment_steps_per_s=derived / environment_count,
            passed_throughput_gate=(
                derived >= config.minimum_aggregate_env_steps_per_s
            ),
            topology_bucketed=True,
            phase_specific_resets=True,
            metadata={
                "tensorized_pi_l_inference": True,
                "production_collector": True,
                "real_contact_sensors": True,
                "batched_qpid_qp": True,
                "phase_aware_reward": True,
                "raw_tensor_artifact_written": True,
                "collector_version": ORDER9_PRODUCTION_COLLECTOR_VERSION,
                "actuator_limits_bound_from_physical_model": True,
                "object_mass_properties_match_task_spec": True,
                "raw_contact_actor_input": False,
                "raw_artifact_path": str(source),
                "raw_artifact_sha256": digest,
                "terminal_count": int(metadata.get("terminal_count", 0)),
                "successful_terminal_count": int(
                    metadata.get("successful_terminal_count", 0)
                ),
            },
        )
        source_hashes[str(source)] = digest
        generation_ids[str(source)] = str(metadata["generation_id"])
    expected_counts = set(config.environment_count_candidates)
    if set(by_environment_count) != expected_counts:
        raise SchemaValidationError(
            "Order9 production benchmark artifacts do not cover configured counts"
        )
    samples = [
        by_environment_count[count]
        for count in config.environment_count_candidates
    ]
    passing = [sample for sample in samples if sample.passed_throughput_gate]
    selected = (
        max(
            passing,
            key=lambda sample: (
                sample.aggregate_env_steps_per_s,
                -abs(
                    sample.environment_count - config.initial_environment_count
                ),
            ),
        ).environment_count
        if passing
        else None
    )
    report = Order9RuntimeBenchmarkReport(
        benchmark_version=ORDER9_RUNTIME_BENCHMARK_VERSION,
        config_hash=stable_hash(config.to_dict()),
        samples=samples,
        selected_environment_count=selected,
        minimum_aggregate_env_steps_per_s=(
            config.minimum_aggregate_env_steps_per_s
        ),
        passed=selected is not None,
        order8_reference=load_order8_timing_reference(order8_report_path),
        metadata={
            "builder_version": ORDER9_PRODUCTION_BENCHMARK_BUILDER_VERSION,
            "initial_environment_count": config.initial_environment_count,
            "selection_rule": (
                "maximum_measured_aggregate_throughput_then_initial_count_proximity"
            ),
            "measurement_scope": (
                "real_isaac_contact_qpid_qp_reward_policy_and_gpu_artifact_buffer"
            ),
            "cold_first_step_included": True,
            "post_simulation_serialization_excluded": True,
            "source_raw_artifact_sha256": source_hashes,
            "source_generation_ids": generation_ids,
            "old_inference_only_probe_replaced": True,
            "full_order8_acceptance_replaced": False,
            "per_step_json_logging": False,
        },
    )
    report.validate()
    if output_path is not None:
        write_order9_runtime_benchmark_report(output_path, report)
    return report


__all__ = [
    "ORDER9_PRODUCTION_BENCHMARK_BUILDER_VERSION",
    "build_order9_production_benchmark_report",
]
