from __future__ import annotations

from pathlib import Path

from amsrr.training.order9_curriculum import (
    Order9StageMetrics,
    load_order9_learning_config,
)
from amsrr.training.order9_pipeline import (
    finalize_order9_stage,
    load_order9_stage_manifest,
    preflight_order9_stage,
)
from amsrr.training.order9_runtime_benchmark import (
    ORDER9_RUNTIME_BENCHMARK_VERSION,
    Order9RuntimeBenchmarkReport,
    Order9RuntimeBenchmarkSample,
)
from amsrr.utils.hashing import hash_file, stable_hash


def test_order9_pipeline_preflights_and_promotes_collection_atomically(
    tmp_path: Path,
) -> None:
    config = _config_with_bound_runtime(tmp_path)
    source = tmp_path / "order8-source.json"
    source.write_text("{}\n", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"

    prepared, dataset_validation = preflight_order9_stage(
        config,
        stage_id="c0_order8_teacher_collection",
        input_artifact_paths={"order8_teacher_source": source},
        output_path=prepared_path,
    )

    assert dataset_validation is None
    assert prepared.status.value == "prepared"
    assert load_order9_stage_manifest(prepared_path).to_dict() == prepared.to_dict()

    dataset = tmp_path / "dataset-manifest.json"
    dataset.write_text("{}\n", encoding="utf-8")
    final_path = tmp_path / "promoted.json"
    final = finalize_order9_stage(
        prepared,
        config,
        metrics=Order9StageMetrics(
            episode_count=100,
            success_count=96,
            no_fallback_success_count=0,
            safety_failure_episode_count=0,
            high_level_decision_count=0,
            fallback_decision_count=0,
            aggregate_env_steps_per_s=0.0,
        ),
        output_artifact_paths={"dataset_manifest": dataset},
        output_path=final_path,
    )

    assert final.promoted
    assert final.status.value == "promoted"
    assert not final.promotion_failed_gates
    assert final.metadata["promotion_evaluation_completed"] is True
    assert load_order9_stage_manifest(final_path).to_dict() == final.to_dict()


def test_order9_pipeline_records_failed_gates_without_promoting(tmp_path: Path) -> None:
    config = _config_with_bound_runtime(tmp_path)
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    prepared, _ = preflight_order9_stage(
        config,
        stage_id="c0_order8_teacher_collection",
        input_artifact_paths={"order8_teacher_source": source},
    )
    output = tmp_path / "dataset.json"
    output.write_text("{}\n", encoding="utf-8")

    final = finalize_order9_stage(
        prepared,
        config,
        metrics=Order9StageMetrics(
            episode_count=10,
            success_count=10,
            no_fallback_success_count=0,
            safety_failure_episode_count=0,
            high_level_decision_count=0,
            fallback_decision_count=0,
            aggregate_env_steps_per_s=0.0,
        ),
        output_artifact_paths={"dataset_manifest": output},
    )

    assert not final.promoted
    assert "minimum_episodes" in final.promotion_failed_gates


def _config_with_bound_runtime(tmp_path: Path):
    config = load_order9_learning_config()
    benchmark_path = tmp_path / "benchmark.json"
    raw_path = tmp_path / "benchmark-raw.pt"
    raw_path.write_bytes(b"unit production benchmark raw artifact")
    report = Order9RuntimeBenchmarkReport(
        benchmark_version=ORDER9_RUNTIME_BENCHMARK_VERSION,
        config_hash=stable_hash(config.runtime_benchmark.to_dict()),
        samples=[
            Order9RuntimeBenchmarkSample(
                environment_count=128,
                attempted=True,
                isaac_backed=True,
                backend_version="unit-isaac",
                device="cuda:0",
                warmup_steps=1,
                measurement_steps=1,
                wall_elapsed_s=0.1,
                aggregate_env_steps_per_s=1280.0,
                per_environment_steps_per_s=10.0,
                passed_throughput_gate=True,
                topology_bucketed=True,
                phase_specific_resets=True,
                metadata={
                    "tensorized_pi_l_inference": True,
                    "production_collector": True,
                    "real_contact_sensors": True,
                    "batched_qpid_qp": True,
                    "phase_aware_reward": True,
                    "raw_tensor_artifact_written": True,
                    "actuator_limits_bound_from_physical_model": True,
                    "object_mass_properties_match_task_spec": True,
                    "raw_artifact_path": str(raw_path),
                    "raw_artifact_sha256": hash_file(raw_path),
                },
            )
        ],
        selected_environment_count=128,
        minimum_aggregate_env_steps_per_s=500.0,
        passed=True,
    )
    benchmark_path.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
    order8_path = tmp_path / "order8.json"
    order8_path.write_text("{}\n", encoding="utf-8")
    config.production_runtime.runtime_benchmark_report_path = str(benchmark_path)
    config.production_runtime.runtime_benchmark_report_sha256 = hash_file(benchmark_path)
    config.production_runtime.canonical_order8_report_path = str(order8_path)
    config.production_runtime.canonical_order8_report_sha256 = hash_file(order8_path)
    config.production_runtime.selected_environment_count = 128
    return config
