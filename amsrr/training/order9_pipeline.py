from __future__ import annotations

"""Fail-closed stage ledger for the production Order 9 curriculum."""

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import (
    ORDER9_STAGE_RUN_VERSION,
    Order9ArtifactBinding,
    Order9PolicyFamily,
    Order9StageRunManifest,
    Order9StageRunStatus,
)
from amsrr.training.order9_curriculum import (
    Order9CurriculumStage,
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
    Order9StageMetrics,
    evaluate_stage_promotion,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_dataset import (
    Order9DatasetStageValidation,
    load_order9_dataset,
    load_order9_dataset_index,
    validate_order9_dataset_for_stage,
    validate_order9_pi_l_dataset_for_stage_streaming,
)
from amsrr.training.order9_evaluation import (
    load_order9_stage_evaluation_report,
    validate_order9_stage_evaluation_report,
)
from amsrr.training.order9_runtime_benchmark import (
    ORDER9_RUNTIME_BENCHMARK_VERSION,
    Order9RuntimeBenchmarkReport,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PIPELINE_VERSION = "order9_production_stage_pipeline_v1"


def order9_schedule_hash(config: Order9LearningConfig) -> str:
    return stable_hash(config.curriculum.to_dict())


def order9_stage_by_id(
    config: Order9LearningConfig,
    stage_id: str,
) -> Order9CurriculumStage:
    matching = [stage for stage in config.curriculum.stages if stage.stage_id == stage_id]
    if len(matching) != 1:
        raise SchemaValidationError(f"unknown Order9 curriculum stage {stage_id!r}")
    return matching[0]


def expected_order9_checkpoint_families(
    stage: Order9CurriculumStage,
) -> tuple[Order9PolicyFamily, ...]:
    if stage.learning_target == Order9LearningTarget.PI_L:
        return (Order9PolicyFamily.PI_L,)
    if stage.learning_target in {
        Order9LearningTarget.PI_H_ASSIGNMENT,
        Order9LearningTarget.PI_H_TRAJECTORY,
    }:
        return (Order9PolicyFamily.PI_H,)
    if stage.learning_target == Order9LearningTarget.PI_D:
        return (Order9PolicyFamily.PI_D,)
    if stage.learning_target in {
        Order9LearningTarget.JOINT_OBJECT_TASK,
        Order9LearningTarget.FULL_SYSTEM,
    }:
        return (
            Order9PolicyFamily.PI_L,
            Order9PolicyFamily.PI_H,
            Order9PolicyFamily.PI_D,
        )
    return ()


def preflight_order9_stage(
    config: Order9LearningConfig,
    *,
    stage_id: str,
    input_artifact_paths: Mapping[str, str | Path],
    prior_stage_manifests: Iterable[Order9StageRunManifest] = (),
    dataset_manifest_path: str | Path | None = None,
    behavior_checkpoint_sha256: str | None = None,
    behavior_checkpoint_sha256_by_family: Mapping[str, str] | None = None,
    output_path: str | Path | None = None,
) -> tuple[Order9StageRunManifest, Order9DatasetStageValidation | None]:
    """Verify every immutable input before a stage may start."""

    config.validate()
    stage = order9_stage_by_id(config, stage_id)
    stage_runtime = resolve_order9_stage_runtime(config, stage)
    schedule_hash = order9_schedule_hash(config)
    _validate_runtime_binding(config)
    _validate_prior_stage_chain(
        config,
        stage,
        list(prior_stage_manifests),
        schedule_hash=schedule_hash,
    )
    bindings = [
        _artifact_binding(kind, path)
        for kind, path in sorted(input_artifact_paths.items())
    ]
    dataset_validation = None
    if dataset_manifest_path is not None:
        if (
            behavior_checkpoint_sha256 is not None
            and behavior_checkpoint_sha256_by_family is not None
        ):
            raise SchemaValidationError(
                "Order9 preflight accepts either one behavior hash or a family map"
            )
        if (
            stage.learning_target == Order9LearningTarget.PI_L
            and stage.learning_mode == Order9LearningMode.BEHAVIOR_CLONING
        ):
            dataset_index = load_order9_dataset_index(dataset_manifest_path)
            dataset_validation = validate_order9_pi_l_dataset_for_stage_streaming(
                dataset_index, stage
            )
            dataset_manifest = dataset_index.manifest_path
        else:
            dataset = load_order9_dataset(dataset_manifest_path)
            dataset_validation = validate_order9_dataset_for_stage(
                dataset,
                stage,
                behavior_checkpoint_sha256=(
                    behavior_checkpoint_sha256_by_family
                    if behavior_checkpoint_sha256_by_family is not None
                    else behavior_checkpoint_sha256
                ),
            )
            dataset_manifest = dataset.manifest_path
        if not dataset_validation.valid:
            raise SchemaValidationError(
                "Order9 dataset failed stage replay contract: "
                + ",".join(dataset_validation.failures)
            )
        dataset_binding = _artifact_binding(
            "dataset_manifest", dataset_manifest
        )
        if all(item.path != dataset_binding.path for item in bindings):
            bindings.append(dataset_binding)
    elif stage.learning_mode not in {
        Order9LearningMode.COLLECTION,
        Order9LearningMode.EVALUATION,
    }:
        raise SchemaValidationError(
            f"Order9 stage {stage.stage_id!r} requires a verified dataset manifest"
        )
    if not bindings:
        raise SchemaValidationError("Order9 stage preflight requires bound input artifacts")
    run_id = (
        f"{stage.stage_id}-"
        + stable_hash(
            {
                "schedule": schedule_hash,
                "stage": stage.to_dict(),
                "seed": config.production_runtime.seed,
                "inputs": [item.to_dict() for item in bindings],
            }
        )[:16]
    )
    manifest = Order9StageRunManifest(
        run_version=ORDER9_STAGE_RUN_VERSION,
        run_id=run_id,
        stage_id=stage.stage_id,
        stage_index=stage.stage_index,
        status=Order9StageRunStatus.PREPARED,
        schedule_hash=schedule_hash,
        stage_config_hash=stable_hash(stage.to_dict()),
        runtime_config_hash=stable_hash(config.production_runtime.to_dict()),
        random_seed=config.production_runtime.seed + stage.stage_index,
        device=config.production_runtime.device,
        environment_count=stage_runtime.environment_count,
        input_artifacts=sorted(bindings, key=lambda item: (item.artifact_kind, item.path)),
        metadata={
            "pipeline_version": ORDER9_PIPELINE_VERSION,
            "learning_mode": stage.learning_mode.value,
            "learning_target": stage.learning_target.value,
            "resolved_stage_runtime": stage_runtime.to_dict(),
            "expected_checkpoint_families": [
                family.value for family in expected_order9_checkpoint_families(stage)
            ],
            "dataset_stage_validation": (
                None if dataset_validation is None else dataset_validation.to_dict()
            ),
            "full_mesh_acceptance_replaced": False,
        },
    )
    if output_path is not None:
        write_order9_stage_manifest(output_path, manifest)
    return manifest, dataset_validation


def finalize_order9_stage(
    prepared: Order9StageRunManifest,
    config: Order9LearningConfig,
    *,
    metrics: Order9StageMetrics,
    output_artifact_paths: Mapping[str, str | Path],
    checkpoint_sha256_by_family: Mapping[Order9PolicyFamily | str, str] | None = None,
    output_path: str | Path | None = None,
) -> Order9StageRunManifest:
    if prepared.status not in {
        Order9StageRunStatus.PREPARED,
        Order9StageRunStatus.RUNNING,
        Order9StageRunStatus.EVALUATED,
    }:
        raise SchemaValidationError("Order9 stage can only finalize from an active status")
    if prepared.schedule_hash != order9_schedule_hash(config):
        raise SchemaValidationError("Order9 stage schedule changed after preflight")
    stage = order9_stage_by_id(config, prepared.stage_id)
    if stable_hash(stage.to_dict()) != prepared.stage_config_hash:
        raise SchemaValidationError("Order9 stage config changed after preflight")
    decision = evaluate_stage_promotion(stage, metrics, config.runtime_benchmark)
    outputs = [
        _artifact_binding(kind, path)
        for kind, path in sorted(output_artifact_paths.items())
    ]
    checkpoint_hashes = {
        Order9PolicyFamily(family).value: digest
        for family, digest in (checkpoint_sha256_by_family or {}).items()
    }
    expected = {family.value for family in expected_order9_checkpoint_families(stage)}
    if stage.learning_mode != Order9LearningMode.COLLECTION and set(checkpoint_hashes) != expected:
        raise SchemaValidationError(
            "Order9 stage checkpoint families do not match its learning target: "
            f"expected={sorted(expected)}, actual={sorted(checkpoint_hashes)}"
        )
    for family, digest in checkpoint_hashes.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise SchemaValidationError(
                f"Order9 checkpoint digest for {family!r} is invalid"
            )
    final = replace(
        prepared,
        status=(
            Order9StageRunStatus.PROMOTED
            if decision.promote
            else Order9StageRunStatus.REJECTED
        ),
        output_artifacts=outputs,
        policy_checkpoint_sha256_by_family=checkpoint_hashes,
        metrics={
            "episode_count": float(metrics.episode_count),
            "success_count": float(metrics.success_count),
            "success_rate": decision.measured_success_rate,
            "no_fallback_success_rate": decision.measured_no_fallback_success_rate,
            "fallback_rate": decision.measured_fallback_rate,
            "safety_failure_episode_count": float(
                metrics.safety_failure_episode_count
            ),
            "aggregate_env_steps_per_s": decision.measured_aggregate_env_steps_per_s,
        },
        promotion_failed_gates=decision.failed_gates,
        promoted=decision.promote,
        metadata={
            **prepared.metadata,
            "promotion_evaluation_completed": True,
            "promotion_decision": decision.to_dict(),
        },
    )
    final.validate()
    if output_path is not None:
        write_order9_stage_manifest(output_path, final)
    return final


def finalize_order9_stage_from_evaluation(
    active: Order9StageRunManifest,
    config: Order9LearningConfig,
    *,
    evaluation_report_path: str | Path,
    output_artifact_paths: Mapping[str, str | Path],
    checkpoint_paths_by_family: Mapping[
        Order9PolicyFamily | str, str | Path
    ] | None = None,
    output_path: str | Path | None = None,
) -> Order9StageRunManifest:
    """Promote/reject only from verified episode rows and checkpoint bytes."""

    stage = order9_stage_by_id(config, active.stage_id)
    report = load_order9_stage_evaluation_report(evaluation_report_path)
    metrics = validate_order9_stage_evaluation_report(
        report,
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        runtime=config.production_runtime,
    )
    supplied = {
        Order9PolicyFamily(family): Path(path)
        for family, path in (checkpoint_paths_by_family or {}).items()
    }
    expected = set(expected_order9_checkpoint_families(stage))
    if set(supplied) != expected:
        raise SchemaValidationError(
            "Order9 evaluation checkpoint families do not match the stage"
        )
    checkpoint_hashes = {
        family: hash_file(path) for family, path in supplied.items()
    }
    report_hashes = {
        Order9PolicyFamily(family): digest
        for family, digest in report.policy_checkpoint_sha256_by_family.items()
    }
    if report_hashes != checkpoint_hashes:
        raise SchemaValidationError(
            "Order9 evaluation report is not bound to the supplied checkpoints"
        )
    if active.policy_checkpoint_sha256_by_family:
        active_hashes = {
            Order9PolicyFamily(family): digest
            for family, digest in active.policy_checkpoint_sha256_by_family.items()
        }
        if active_hashes != checkpoint_hashes:
            raise SchemaValidationError(
                "Order9 evaluated checkpoint differs from the stage training output"
            )
    outputs = dict(output_artifact_paths)
    evaluation_path = str(Path(evaluation_report_path))
    existing_evaluation = outputs.get("stage_evaluation_report")
    if existing_evaluation is not None and str(existing_evaluation) != evaluation_path:
        raise SchemaValidationError(
            "Order9 output artifacts contain a conflicting evaluation report"
        )
    outputs["stage_evaluation_report"] = evaluation_report_path
    return finalize_order9_stage(
        active,
        config,
        metrics=metrics,
        output_artifact_paths=outputs,
        checkpoint_sha256_by_family=checkpoint_hashes,
        output_path=output_path,
    )


def record_order9_stage_training_outputs(
    prepared: Order9StageRunManifest,
    config: Order9LearningConfig,
    *,
    output_artifact_paths: Mapping[str, str | Path],
    checkpoint_paths_by_family: Mapping[Order9PolicyFamily | str, str | Path],
    output_path: str | Path | None = None,
) -> Order9StageRunManifest:
    """Bind completed training bytes without treating training as promotion."""

    if prepared.status != Order9StageRunStatus.PREPARED:
        raise SchemaValidationError(
            "Order9 training outputs can only attach to a prepared stage"
        )
    if prepared.schedule_hash != order9_schedule_hash(config):
        raise SchemaValidationError("Order9 schedule changed after stage preflight")
    stage = order9_stage_by_id(config, prepared.stage_id)
    if stable_hash(stage.to_dict()) != prepared.stage_config_hash:
        raise SchemaValidationError("Order9 stage config changed after preflight")
    expected = set(expected_order9_checkpoint_families(stage))
    supplied = {
        Order9PolicyFamily(family): Path(path)
        for family, path in checkpoint_paths_by_family.items()
    }
    if set(supplied) != expected:
        raise SchemaValidationError(
            "Order9 training checkpoint families do not match stage target"
        )
    from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint

    checkpoint_hashes: dict[str, str] = {}
    for family, path in supplied.items():
        loaded = load_order9_policy_checkpoint(
            path,
            expected_family=family,
            expected_schedule_hash=prepared.schedule_hash,
        )
        if (
            loaded.metadata.curriculum_stage_id != stage.stage_id
            or loaded.metadata.curriculum_stage_index != stage.stage_index
        ):
            raise SchemaValidationError(
                "Order9 checkpoint metadata does not identify the prepared stage"
            )
        checkpoint_hashes[family.value] = loaded.sha256
    outputs = [
        _artifact_binding(kind, path)
        for kind, path in sorted(output_artifact_paths.items())
    ]
    checkpoint_output_paths = {str(path) for path in supplied.values()}
    if not checkpoint_output_paths.issubset({item.path for item in outputs}):
        raise SchemaValidationError(
            "Order9 checkpoint paths must also appear in output artifacts"
        )
    running = replace(
        prepared,
        status=Order9StageRunStatus.RUNNING,
        output_artifacts=outputs,
        policy_checkpoint_sha256_by_family=checkpoint_hashes,
        metadata={
            **prepared.metadata,
            "offline_training_completed": True,
            "promotion_evaluation_completed": False,
        },
    )
    running.validate()
    if output_path is not None:
        write_order9_stage_manifest(output_path, running)
    return running


def load_order9_stage_manifest(path: str | Path) -> Order9StageRunManifest:
    return Order9StageRunManifest.from_json(Path(path).read_text(encoding="utf-8"))


def write_order9_stage_manifest(
    path: str | Path,
    manifest: Order9StageRunManifest,
) -> None:
    manifest.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(manifest.to_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_runtime_binding(config: Order9LearningConfig) -> None:
    runtime = config.production_runtime
    benchmark_path = Path(runtime.runtime_benchmark_report_path)
    order8_path = Path(runtime.canonical_order8_report_path)
    if not runtime.runtime_benchmark_report_sha256:
        raise SchemaValidationError(
            "Order9 production runtime must bind a completed benchmark hash"
        )
    if hash_file(benchmark_path) != runtime.runtime_benchmark_report_sha256:
        raise SchemaValidationError("Order9 runtime benchmark artifact hash mismatch")
    if hash_file(order8_path) != runtime.canonical_order8_report_sha256:
        raise SchemaValidationError("Order9 canonical Order8 report hash mismatch")
    report = Order9RuntimeBenchmarkReport.from_json(
        benchmark_path.read_text(encoding="utf-8")
    )
    if report.benchmark_version != ORDER9_RUNTIME_BENCHMARK_VERSION:
        raise SchemaValidationError("Order9 runtime benchmark version is stale")
    if report.config_hash != stable_hash(config.runtime_benchmark.to_dict()):
        raise SchemaValidationError("Order9 runtime benchmark config changed")
    if (
        not report.passed
        or report.selected_environment_count != runtime.selected_environment_count
    ):
        raise SchemaValidationError(
            "Order9 selected environment count is not benchmark-authorized"
        )
    if config.runtime_benchmark.require_tensorized_pi_l_inference:
        sample = next(
            (
                item
                for item in report.samples
                if item.environment_count == report.selected_environment_count
            ),
            None,
        )
        if sample is None:
            raise SchemaValidationError(
                "Order9 benchmark omits its selected environment-count sample"
            )
        if sample.metadata.get("tensorized_pi_l_inference") is not True:
            raise SchemaValidationError(
                "Order9 benchmark did not include tensorized pi_L inference"
            )
        required_production_evidence = (
            "production_collector",
            "real_contact_sensors",
            "batched_qpid_qp",
            "phase_aware_reward",
            "raw_tensor_artifact_written",
            "actuator_limits_bound_from_physical_model",
            "object_mass_properties_match_task_spec",
        )
        if any(
            sample.metadata.get(name) is not True
            for name in required_production_evidence
        ):
            raise SchemaValidationError(
                "Order9 benchmark omitted production collector evidence"
            )
        raw_path_value = sample.metadata.get("raw_artifact_path")
        raw_sha256 = sample.metadata.get("raw_artifact_sha256")
        if not isinstance(raw_path_value, str) or not isinstance(raw_sha256, str):
            raise SchemaValidationError(
                "Order9 benchmark omitted raw artifact provenance"
            )
        if hash_file(Path(raw_path_value)) != raw_sha256:
            raise SchemaValidationError(
                "Order9 benchmark raw artifact hash mismatch"
            )


def _validate_prior_stage_chain(
    config: Order9LearningConfig,
    stage: Order9CurriculumStage,
    manifests: list[Order9StageRunManifest],
    *,
    schedule_hash: str,
) -> None:
    required_ids = {
        item.stage_id
        for item in config.curriculum.stages
        if item.stage_index < stage.stage_index
    }
    by_id = {manifest.stage_id: manifest for manifest in manifests}
    if len(by_id) != len(manifests):
        raise SchemaValidationError("Order9 prior stage manifests contain duplicates")
    missing = sorted(required_ids - set(by_id))
    if missing:
        raise SchemaValidationError(
            "Order9 stage requires all earlier promoted stages: " + ",".join(missing)
        )
    for stage_id in sorted(required_ids):
        manifest = by_id[stage_id]
        if not manifest.promoted or manifest.status != Order9StageRunStatus.PROMOTED:
            raise SchemaValidationError(
                f"Order9 prerequisite stage {stage_id!r} was not promoted"
            )
        if manifest.schedule_hash != schedule_hash:
            raise SchemaValidationError(
                f"Order9 prerequisite stage {stage_id!r} uses a different schedule"
            )


def _artifact_binding(kind: str, path: str | Path) -> Order9ArtifactBinding:
    source = Path(path)
    return Order9ArtifactBinding(
        artifact_kind=str(kind),
        path=str(source),
        sha256=hash_file(source),
    )
