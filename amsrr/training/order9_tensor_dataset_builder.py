from __future__ import annotations

"""Merge real-Isaac tensor rollout shards into one exact on-policy dataset."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit, P4_3DatasetManifest
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_curriculum import (
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_dataset import (
    Order9DatasetBundle,
    load_order9_dataset_index,
)
from amsrr.training.order9_online_dataset import write_order9_on_policy_dataset
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    load_order9_tensor_rollout_artifact,
    order9_pi_l_records_from_tensor_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_TENSOR_DATASET_BUILDER_VERSION = (
    "order9_tensor_pi_l_on_policy_dataset_builder_v2_exact_actor_graph_replay"
)


@dataclass(frozen=True)
class Order9PiLDatasetBuildResult:
    manifest: P4_3DatasetManifest
    bundle: Order9DatasetBundle


def build_order9_pi_l_on_policy_dataset(
    output_dir: str | Path,
    *,
    raw_artifact_paths: Sequence[str | Path],
    generation_id: str,
    stage_id: str,
    pi_l_checkpoint_path: str | Path,
    config: Order9LearningConfig,
    physical_model: PhysicalModel | None = None,
    require_train_and_validation: bool = True,
    _bundle_sink: list[Order9DatasetBundle] | None = None,
) -> P4_3DatasetManifest:
    """Validate and merge one fresh ``pi_L`` behavior-policy generation.

    Every source artifact must identify the same configuration, stage,
    behavior checkpoint, robot model, and simulator backend.  Morphology,
    object task, and split may differ by bucket.  Source-byte hashes namespace
    episode identifiers so independently collected shards cannot collide.
    """

    config.validate()
    stage = order9_stage_by_id(config, stage_id)
    stage_runtime = resolve_order9_stage_runtime(config, stage)
    if (
        stage.learning_mode != Order9LearningMode.PPO
        or stage.learning_target != Order9LearningTarget.PI_L
    ):
        raise SchemaValidationError(
            "Order9 tensor dataset builder requires a pi_L PPO stage"
        )
    if not generation_id:
        raise SchemaValidationError("Order9 tensor dataset generation is empty")
    sources = tuple(Path(value).resolve() for value in raw_artifact_paths)
    if not sources or len(sources) != len(set(sources)):
        raise SchemaValidationError(
            "Order9 tensor dataset requires unique raw artifact paths"
        )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"Order9 on-policy output directory is not empty: {target}"
        )

    model = physical_model or build_physical_model_from_config(
        config.production_runtime.robot_model_config_path
    )
    model.validate()
    checkpoint_path = Path(pi_l_checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = hash_file(checkpoint_path)
    expected = {
        "generation_id": generation_id,
        "stage_id": stage.stage_id,
        "stage_config_hash": stable_hash(stage.to_dict()),
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "config_hash": stable_hash(config.to_dict()),
        "physical_model_hash": model.stable_hash(),
        "urdf_hash": hash_file(model.urdf_path),
        "thrust_model_hash": str(model.metadata.get("thrust_model_hash", "")),
        "pi_l_checkpoint_sha256": checkpoint_sha256,
        "topology_randomized": bool(stage.topology_randomized),
    }
    if not expected["thrust_model_hash"]:
        raise SchemaValidationError(
            "Order9 PhysicalModel lacks thrust-model provenance"
        )

    artifacts = []
    source_hashes: dict[str, str] = {}
    records = []
    tasks: dict[str, TaskSpec] = {}
    task_hashes: dict[str, str] = {}
    splits_present: set[DatasetSplit] = set()
    random_seeds: set[int] = set()
    simulator_versions: set[str] = set()
    simulator_hashes: set[str] = set()
    collector_versions: set[str] = set()
    robot_usd_hashes: dict[str, str] = {}
    morphology_hashes: dict[str, str] = {}
    source_collection_runtime: dict[str, dict[str, object]] = {}
    total_rollout_wall_elapsed_s = 0.0
    total_setup_wall_elapsed_s = 0.0
    runtime_metric_source_count = 0

    for source in sources:
        digest = hash_file(source)
        if digest in source_hashes.values():
            raise SchemaValidationError(
                "Order9 tensor dataset contains duplicate raw artifact bytes"
            )
        source_hashes[str(source)] = digest
        artifact = load_order9_tensor_rollout_artifact(
            source, expected_sha256=digest
        )
        artifacts.append(artifact)
        metadata = artifact.metadata
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise SchemaValidationError(
                    f"Order9 raw artifact {source} differs at {name}"
                )
        if "runtime_override_used" in metadata:
            if metadata.get("runtime_override_used") is not False:
                raise SchemaValidationError(
                    "Order9 production dataset cannot consume a diagnostic "
                    "collector runtime override"
                )
            if (
                int(metadata.get("environment_count", 0))
                != stage_runtime.environment_count
                or int(metadata.get("rollout_steps", 0))
                != stage_runtime.rollout_steps_per_environment
            ):
                raise SchemaValidationError(
                    "Order9 raw artifact does not match the configured stage runtime"
                )
        setup_elapsed = metadata.get("setup_wall_elapsed_s")
        rollout_elapsed = metadata.get("rollout_wall_elapsed_s")
        if _positive_number(setup_elapsed) and _positive_number(rollout_elapsed):
            setup_value = float(setup_elapsed)
            rollout_value = float(rollout_elapsed)
            total_setup_wall_elapsed_s += setup_value
            total_rollout_wall_elapsed_s += rollout_value
            runtime_metric_source_count += 1
            source_collection_runtime[str(source)] = {
                "environment_count": artifact.environment_count,
                "rollout_steps_per_environment": artifact.step_count,
                "environment_steps": artifact.environment_step_count,
                "setup_wall_elapsed_s": setup_value,
                "rollout_wall_elapsed_s": rollout_value,
                "collection_wall_elapsed_s": setup_value + rollout_value,
                "aggregate_env_steps_per_s": (
                    artifact.environment_step_count / rollout_value
                ),
                "end_to_end_env_steps_per_s": (
                    artifact.environment_step_count / (setup_value + rollout_value)
                ),
                "runtime_load": _runtime_load_summary(metadata.get("runtime_load")),
            }
        namespace = digest[:20]
        records.extend(
            order9_pi_l_records_from_tensor_artifact(
                artifact,
                record_namespace=namespace,
            )
        )
        for raw_task, raw_split in zip(
            metadata["task_specs"], metadata["environment_splits"]
        ):
            task = TaskSpec.from_dict(raw_task)
            split = DatasetSplit(raw_split)
            digest_task = task.stable_hash()
            previous = task_hashes.setdefault(task.task_id, digest_task)
            if previous != digest_task:
                raise SchemaValidationError(
                    f"Order9 task id {task.task_id!r} has conflicting payloads"
                )
            tasks.setdefault(task.task_id, task)
            splits_present.add(split)
        random_seeds.add(int(metadata["random_seed"]))
        simulator_versions.add(str(metadata["simulator_version"]))
        simulator_hashes.add(str(metadata["simulator_hash"]))
        collector_versions.add(str(metadata.get("collector_version", "unknown")))
        robot_usd_hashes[str(source)] = str(metadata["robot_usd_sha256"])
        morphology_hashes[str(source)] = stable_hash(metadata["morphology_graph"])

    if require_train_and_validation and not {
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
    }.issubset(splits_present):
        raise SchemaValidationError(
            "Order9 pi_L PPO generation requires train and validation shards"
        )
    if DatasetSplit.HELD_OUT in splits_present:
        raise SchemaValidationError(
            "Order9 pi_L training generation cannot consume held-out rollouts"
        )
    if len(simulator_versions) != 1 or len(simulator_hashes) != 1:
        raise SchemaValidationError(
            "Order9 raw artifacts use different simulator backends"
        )

    total_environment_steps = sum(
        artifact.environment_step_count for artifact in artifacts
    )
    collection_runtime_complete = runtime_metric_source_count == len(artifacts)
    manifest = write_order9_on_policy_dataset(
        target,
        generation_id=generation_id,
        low_level_records=tuple(records),
        task_specs=tasks,
        behavior_checkpoint_sha256_by_family={
            Order9PolicyFamily.PI_L.value: checkpoint_sha256
        },
        source_isaac_artifact_paths=sources,
        on_policy_environment_step_count=total_environment_steps,
        random_seeds=sorted(random_seeds),
        config_hash=str(expected["config_hash"]),
        robot_model_hash=str(expected["physical_model_hash"]),
        urdf_hash=str(expected["urdf_hash"]),
        thrust_model_hash=str(expected["thrust_model_hash"]),
        simulator_version=next(iter(simulator_versions)),
        simulator_hash=next(iter(simulator_hashes)),
        metadata={
            "builder_version": ORDER9_TENSOR_DATASET_BUILDER_VERSION,
            "tensor_rollout_artifact_version": (
                ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION
            ),
            "stage_id": stage.stage_id,
            "stage_config_hash": expected["stage_config_hash"],
            "curriculum_schedule_hash": expected[
                "curriculum_schedule_hash"
            ],
            "topology_randomized": bool(stage.topology_randomized),
            "source_raw_artifact_sha256": source_hashes,
            "source_robot_usd_sha256": robot_usd_hashes,
            "source_morphology_hash": morphology_hashes,
            "collector_versions": sorted(collector_versions),
            "source_split_values": sorted(value.value for value in splits_present),
            "pi_l_checkpoint_path": str(checkpoint_path),
            "one_behavior_checkpoint_only": True,
            "episode_id_namespaced_by_source_bytes": True,
            "configured_stage_runtime": stage_runtime.to_dict(),
            "source_collection_runtime": source_collection_runtime,
            "collection_runtime_complete": collection_runtime_complete,
            "aggregate_collection_env_steps_per_s": (
                total_environment_steps / total_rollout_wall_elapsed_s
                if collection_runtime_complete
                else None
            ),
            "end_to_end_collection_env_steps_per_s": (
                total_environment_steps
                / (total_setup_wall_elapsed_s + total_rollout_wall_elapsed_s)
                if collection_runtime_complete
                else None
            ),
            "total_setup_wall_elapsed_s": (
                total_setup_wall_elapsed_s if collection_runtime_complete else None
            ),
            "total_rollout_wall_elapsed_s": (
                total_rollout_wall_elapsed_s if collection_runtime_complete else None
            ),
        },
    )
    if _bundle_sink is not None:
        index = load_order9_dataset_index(target / "manifest.json")
        _bundle_sink.append(
            Order9DatasetBundle(
                manifest=index.manifest,
                manifest_path=index.manifest_path,
                manifest_sha256=index.manifest_sha256,
                low_level_records=tuple(records),
                trajectory_records=(),
                sequential_design_records=(),
                design_outcome_records=(),
                rollout_archives=(),
                verified_shard_sha256=index.verified_shard_sha256,
            )
        )
    return manifest


def build_order9_pi_l_on_policy_dataset_with_bundle(
    output_dir: str | Path,
    *,
    raw_artifact_paths: Sequence[str | Path],
    generation_id: str,
    stage_id: str,
    pi_l_checkpoint_path: str | Path,
    config: Order9LearningConfig,
    physical_model: PhysicalModel | None = None,
    require_train_and_validation: bool = True,
) -> Order9PiLDatasetBuildResult:
    """Build canonical shards and retain the same verified records for PPO.

    The bundle is an in-process handoff only.  The canonical manifest and
    compressed JSONL shards are still written and hash-verified exactly as in
    :func:`build_order9_pi_l_on_policy_dataset`.
    """

    bundles: list[Order9DatasetBundle] = []
    manifest = build_order9_pi_l_on_policy_dataset(
        output_dir,
        raw_artifact_paths=raw_artifact_paths,
        generation_id=generation_id,
        stage_id=stage_id,
        pi_l_checkpoint_path=pi_l_checkpoint_path,
        config=config,
        physical_model=physical_model,
        require_train_and_validation=require_train_and_validation,
        _bundle_sink=bundles,
    )
    if len(bundles) != 1:
        raise RuntimeError("Order9 tensor dataset builder did not retain one bundle")
    return Order9PiLDatasetBuildResult(manifest=manifest, bundle=bundles[0])


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _runtime_load_summary(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {
        str(name): item
        for name, item in value.items()
        if name != "samples"
    }


__all__ = [
    "ORDER9_TENSOR_DATASET_BUILDER_VERSION",
    "Order9PiLDatasetBuildResult",
    "build_order9_pi_l_on_policy_dataset",
    "build_order9_pi_l_on_policy_dataset_with_bundle",
]
