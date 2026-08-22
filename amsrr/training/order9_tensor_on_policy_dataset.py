from __future__ import annotations

"""Hash-bound tensor-native dataset index for Order 9 ``pi_L`` PPO.

The real-Isaac collector already emits every tensor required by PPO.  This
module deliberately does not reconstruct ``LowLevelControlRecord`` objects or
write JSONL shards; it validates the immutable rollout tensors and writes only
the small lineage manifest consumed by the tensor-native trainer.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.training.order9_curriculum import (
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_dataset import Order9DatasetStageValidation
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    Order9TensorRolloutArtifact,
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V1 = (
    "order9_tensor_native_pi_l_on_policy_dataset_v1"
)
ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2 = (
    "order9_tensor_native_pi_l_on_policy_dataset_v2_topology_stratified"
)
ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3 = (
    "order9_tensor_native_pi_l_on_policy_dataset_v3_replicated_topologies"
)
ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V4 = (
    "order9_tensor_native_pi_l_on_policy_dataset_v4_mixed_state_inheritance"
)
ORDER9_TENSOR_ON_POLICY_DATASET_VERSION = (
    "order9_tensor_native_pi_l_on_policy_dataset_v5_transition_backward_curriculum"
)
ORDER9_PHASE_RESET_ROLLOUT_MODE = "phase_reset"
ORDER9_STATE_INHERITANCE_ROLLOUT_MODE = "continuous_state_inheritance"
_SUPPORTED_PPO_ARTIFACT_VERSIONS = frozenset(
    {
        ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
        ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
        ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    }
)


@dataclass
class Order9TensorRolloutShard(SchemaBase):
    split: str
    path: str
    sha256: str
    artifact_version: str
    environment_count: int
    rollout_step_count: int
    environment_step_count: int
    episode_count: int
    task_count: int
    morphology_hash: str
    random_seed: int
    simulator_version: str
    simulator_hash: str
    collector_version: str
    robot_usd_sha256: str
    rollout_mode: str = ORDER9_PHASE_RESET_ROLLOUT_MODE
    module_count: int = 0

    def validate(self) -> None:
        if self.split not in {DatasetSplit.TRAIN.value, DatasetSplit.VALIDATION.value}:
            raise SchemaValidationError("Order9 tensor shard split is invalid")
        require_non_empty(self.path, "Order9TensorRolloutShard.path")
        for name in (
            "sha256",
            "morphology_hash",
            "simulator_hash",
            "robot_usd_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if self.artifact_version not in _SUPPORTED_PPO_ARTIFACT_VERSIONS:
            raise SchemaValidationError("Order9 tensor shard artifact version differs")
        if min(
            self.environment_count,
            self.rollout_step_count,
            self.environment_step_count,
            self.episode_count,
            self.task_count,
        ) < 1:
            raise SchemaValidationError("Order9 tensor shard counts must be positive")
        if self.random_seed < 0:
            raise SchemaValidationError("Order9 tensor shard seed must be non-negative")
        for name in ("simulator_version", "collector_version"):
            require_non_empty(str(getattr(self, name)), name)
        if self.rollout_mode not in {
            ORDER9_PHASE_RESET_ROLLOUT_MODE,
            ORDER9_STATE_INHERITANCE_ROLLOUT_MODE,
        }:
            raise SchemaValidationError("Order9 tensor shard rollout mode is invalid")
        if self.module_count < 0:
            raise SchemaValidationError("Order9 tensor shard module count is invalid")


@dataclass
class Order9TensorPiLDatasetManifest(SchemaBase):
    dataset_version: str
    generation_id: str
    stage_id: str
    stage_config_hash: str
    curriculum_schedule_hash: str
    config_hash: str
    physical_model_hash: str
    urdf_hash: str
    thrust_model_hash: str
    behavior_checkpoint_sha256: str
    environment_step_count: int
    shards: list[Order9TensorRolloutShard]
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dataset_version not in {
            ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V1,
            ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2,
            ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3,
            ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V4,
            ORDER9_TENSOR_ON_POLICY_DATASET_VERSION,
        }:
            raise SchemaValidationError("Order9 tensor dataset version differs")
        for name in ("generation_id", "stage_id"):
            require_non_empty(str(getattr(self, name)), name)
        for name in (
            "stage_config_hash",
            "curriculum_schedule_hash",
            "config_hash",
            "physical_model_hash",
            "urdf_hash",
            "thrust_model_hash",
            "behavior_checkpoint_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        split_counts = {
            split: sum(item.split == split for item in self.shards)
            for split in (DatasetSplit.TRAIN.value, DatasetSplit.VALIDATION.value)
        }
        if len(self.shards) < 2 or any(value < 1 for value in split_counts.values()):
            raise SchemaValidationError(
                "Order9 tensor dataset requires train and validation shards"
            )
        if (
            self.dataset_version == ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V1
            and split_counts
            != {
                DatasetSplit.TRAIN.value: 1,
                DatasetSplit.VALIDATION.value: 1,
            }
        ):
            raise SchemaValidationError(
                "Order9 v1 tensor dataset requires exactly one train and validation shard"
            )
        if {item.split for item in self.shards} != {
            DatasetSplit.TRAIN.value,
            DatasetSplit.VALIDATION.value,
        }:
            raise SchemaValidationError(
                "Order9 tensor dataset requires exactly one train and validation shard"
            )
        for shard in self.shards:
            shard.validate()
        if len({item.path for item in self.shards}) != len(self.shards) or len(
            {item.sha256 for item in self.shards}
        ) != len(self.shards):
            raise SchemaValidationError("Order9 tensor dataset shards are duplicated")
        if self.environment_step_count != sum(
            item.environment_step_count for item in self.shards
        ):
            raise SchemaValidationError("Order9 tensor dataset step count differs")
        if self.metadata.get("one_fresh_generation") is not True:
            raise SchemaValidationError("Order9 tensor dataset is not one fresh generation")
        if self.metadata.get("tensor_native_ppo") is not True:
            raise SchemaValidationError("Order9 tensor dataset is not tensor-native")
        if self.metadata.get("topology_stratified_train") is True:
            train = [
                shard
                for shard in self.shards
                if shard.split == DatasetSplit.TRAIN.value
            ]
            module_counts = self.metadata.get("train_module_counts")
            common_invalid = (
                self.dataset_version
                not in {
                    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2,
                    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3,
                    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V4,
                    ORDER9_TENSOR_ON_POLICY_DATASET_VERSION,
                }
                or len(train) < 2
                or not isinstance(module_counts, list)
                or len(module_counts) != len(train)
            )
            unique_v2_invalid = (
                self.dataset_version
                == ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2
                and sorted(set(int(value) for value in module_counts or []))
                != sorted(int(value) for value in module_counts or [])
            )
            replicas = self.metadata.get("topologies_per_module_count")
            replicated_v3_invalid = (
                self.dataset_version == ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3
                and (
                    not isinstance(replicas, int)
                    or replicas < 1
                    or any(
                        list(module_counts or []).count(value) != replicas
                        for value in set(module_counts or [])
                    )
                )
            )
            mixed_v4_invalid = False
            if self.dataset_version in {
                ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V4,
                ORDER9_TENSOR_ON_POLICY_DATASET_VERSION,
            }:
                reset_count = self.metadata.get("phase_reset_train_shard_count")
                inheritance_count = self.metadata.get(
                    "state_inheritance_train_shard_count"
                )
                reset_replicas = self.metadata.get(
                    "phase_reset_topologies_per_module_count"
                )
                inheritance_replicas = self.metadata.get(
                    "state_inheritance_shards_per_module_count"
                )
                reset_rows = [
                    shard
                    for shard in train
                    if shard.rollout_mode == ORDER9_PHASE_RESET_ROLLOUT_MODE
                ]
                inheritance_rows = [
                    shard
                    for shard in train
                    if shard.rollout_mode
                    == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
                ]
                module_values = {int(value) for value in module_counts or []}
                if isinstance(inheritance_replicas, int):
                    inheritance_replicas_by_module = {
                        value: inheritance_replicas for value in module_values
                    }
                elif isinstance(inheritance_replicas, dict):
                    try:
                        inheritance_replicas_by_module = {
                            int(key): int(value)
                            for key, value in inheritance_replicas.items()
                        }
                    except (TypeError, ValueError):
                        inheritance_replicas_by_module = {}
                else:
                    inheritance_replicas_by_module = {}
                mixed_v4_invalid = (
                    not isinstance(reset_count, int)
                    or not isinstance(inheritance_count, int)
                    or not isinstance(reset_replicas, int)
                    or reset_count != len(reset_rows)
                    or inheritance_count != len(inheritance_rows)
                    or reset_count + inheritance_count != len(train)
                    or reset_replicas < 1
                    or set(inheritance_replicas_by_module) != module_values
                    or any(
                        value < 0
                        for value in inheritance_replicas_by_module.values()
                    )
                    or any(
                        sum(
                            row.rollout_mode == ORDER9_PHASE_RESET_ROLLOUT_MODE
                            and row.module_count == value
                            for row in train
                        )
                        != reset_replicas
                        or sum(
                            row.rollout_mode
                            == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
                            and row.module_count == value
                            for row in train
                        )
                        != inheritance_replicas_by_module.get(int(value), 0)
                        for value in module_values
                    )
                )
            if (
                common_invalid
                or unique_v2_invalid
                or replicated_v3_invalid
                or mixed_v4_invalid
            ):
                raise SchemaValidationError(
                    "Order9 topology-stratified tensor dataset metadata is invalid"
                )


@dataclass(frozen=True)
class Order9TensorPiLDatasetBundle:
    manifest: Order9TensorPiLDatasetManifest
    manifest_path: str
    manifest_sha256: str
    artifacts_by_split: dict[str, tuple[Order9TensorRolloutArtifact, ...]]
    verified_shard_sha256: dict[str, str]

    @property
    def train_artifact(self) -> Order9TensorRolloutArtifact:
        values = self.train_artifacts
        if len(values) != 1:
            raise SchemaValidationError(
                "Order9 tensor dataset has multiple train topology shards"
            )
        return values[0]

    @property
    def train_artifacts(self) -> tuple[Order9TensorRolloutArtifact, ...]:
        return self.artifacts_by_split[DatasetSplit.TRAIN.value]

    @property
    def validation_artifacts(self) -> tuple[Order9TensorRolloutArtifact, ...]:
        return self.artifacts_by_split[DatasetSplit.VALIDATION.value]


def build_order9_tensor_pi_l_dataset(
    output_dir: str | Path,
    *,
    raw_artifact_paths: Sequence[str | Path],
    generation_id: str,
    stage_id: str,
    behavior_checkpoint_path: str | Path,
    config: Order9LearningConfig,
    physical_model: PhysicalModel,
    known_sha256_by_path: Mapping[str | Path, str] | None = None,
    c3_training_module_counts: Sequence[int] | None = None,
) -> Order9TensorPiLDatasetBundle:
    """Validate fresh rollout tensors and atomically publish their small index."""

    target = Path(output_dir).resolve()
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Order9 tensor dataset already exists: {manifest_path}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Order9 tensor dataset directory is not empty: {target}")
    config.validate()
    stage = order9_stage_by_id(config, stage_id)
    if (
        stage.learning_mode != Order9LearningMode.PPO
        or stage.learning_target != Order9LearningTarget.PI_L
    ):
        raise SchemaValidationError("Order9 tensor dataset requires a pi_L PPO stage")
    runtime = resolve_order9_stage_runtime(config, stage)
    training_module_counts = _resolve_c3_training_module_counts(
        stage_id=stage.stage_id,
        stage_minimum_module_count=stage.min_modules,
        stage_maximum_module_count=stage.max_modules,
        requested=c3_training_module_counts,
    )
    sources = tuple(Path(value).resolve() for value in raw_artifact_paths)
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise SchemaValidationError(
            "Order9 tensor dataset requires unique train and validation shards"
        )
    checkpoint = Path(behavior_checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = hash_file(checkpoint)
    known = {
        str(Path(path).resolve()): digest
        for path, digest in dict(known_sha256_by_path or {}).items()
    }
    expected = {
        "generation_id": generation_id,
        "stage_id": stage.stage_id,
        "stage_config_hash": stable_hash(stage.to_dict()),
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "config_hash": stable_hash(config.to_dict()),
        "physical_model_hash": physical_model.stable_hash(),
        "urdf_hash": hash_file(physical_model.urdf_path),
        "thrust_model_hash": str(physical_model.metadata.get("thrust_model_hash", "")),
        "pi_l_checkpoint_sha256": checkpoint_sha,
        "topology_randomized": bool(stage.topology_randomized),
    }
    if not expected["thrust_model_hash"]:
        raise SchemaValidationError("Order9 PhysicalModel lacks thrust-model provenance")

    artifacts: dict[str, list[Order9TensorRolloutArtifact]] = {
        DatasetSplit.TRAIN.value: [],
        DatasetSplit.VALIDATION.value: [],
    }
    shard_rows: list[Order9TensorRolloutShard] = []
    verified: dict[str, str] = {}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = known.get(str(source)) or hash_file(source)
        artifact = load_order9_tensor_rollout_artifact(
            source, expected_sha256=digest
        )
        metadata = artifact.metadata
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise SchemaValidationError(
                    f"Order9 raw artifact {source} differs at {name}"
                )
        if metadata.get("runtime_override_used") is not False:
            raise SchemaValidationError("Order9 tensor dataset rejects runtime overrides")
        rollout_mode = str(
            metadata.get(
                "c3_train_rollout_mode", ORDER9_PHASE_RESET_ROLLOUT_MODE
            )
        )
        if rollout_mode not in {
            ORDER9_PHASE_RESET_ROLLOUT_MODE,
            ORDER9_STATE_INHERITANCE_ROLLOUT_MODE,
        }:
            raise SchemaValidationError("Order9 raw artifact rollout mode is invalid")
        expected_rollout_steps = (
            config.production_runtime.c3_state_inheritance_rollout_steps
            if rollout_mode == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
            else runtime.rollout_steps_per_environment
        )
        if int(metadata.get("rollout_steps", 0)) != expected_rollout_steps:
            raise SchemaValidationError("Order9 raw artifact runtime differs from stage")
        raw_splits = {str(value) for value in metadata["environment_splits"]}
        if len(raw_splits) != 1:
            raise SchemaValidationError("Order9 raw shard mixes dataset splits")
        split = next(iter(raw_splits))
        if split not in {DatasetSplit.TRAIN.value, DatasetSplit.VALIDATION.value}:
            raise SchemaValidationError("Order9 raw shard split is not train/validation")
        if (
            rollout_mode == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
            and split != DatasetSplit.TRAIN.value
        ):
            raise SchemaValidationError(
                "Order9 state-inheritance artifact must use the train split"
            )
        if rollout_mode == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE and (
            metadata.get("state_inheritance_initial_phase_indices")
            != list(
                config.production_runtime
                .c3_state_inheritance_initial_phase_indices
            )
            or metadata.get(
                "state_inheritance_fixed_reset_progress_fraction"
            )
            != config.production_runtime
            .c3_state_inheritance_fixed_reset_progress_fraction
        ):
            raise SchemaValidationError(
                "Order9 state-inheritance artifact curriculum differs"
            )
        artifacts[split].append(artifact)
        verified[str(source)] = digest
        tasks = metadata["task_specs"]
        episode_count = sum(
            len(set(int(value) for value in artifact.tensors["episode_serial"][:, env][artifact.tensors["valid"][:, env]].tolist()))
            for env in range(artifact.environment_count)
        )
        shard_rows.append(
            Order9TensorRolloutShard(
                split=split,
                path=str(source),
                sha256=digest,
                artifact_version=artifact.artifact_version,
                environment_count=artifact.environment_count,
                rollout_step_count=artifact.step_count,
                environment_step_count=artifact.environment_step_count,
                episode_count=episode_count,
                task_count=len({stable_hash(value) for value in tasks}),
                morphology_hash=stable_hash(metadata["morphology_graph"]),
                random_seed=int(metadata["random_seed"]),
                simulator_version=str(metadata["simulator_version"]),
                simulator_hash=str(metadata["simulator_hash"]),
                collector_version=str(metadata.get("collector_version", "unknown")),
                robot_usd_sha256=str(metadata["robot_usd_sha256"]),
                rollout_mode=rollout_mode,
                module_count=len(metadata["morphology_graph"]["modules"]),
            )
        )
    if any(not values for values in artifacts.values()):
        raise SchemaValidationError("Order9 tensor dataset split pair is incomplete")
    if len({row.simulator_version for row in shard_rows}) != 1 or len(
        {row.simulator_hash for row in shard_rows}
    ) != 1:
        raise SchemaValidationError("Order9 tensor dataset simulator lineage differs")
    train_artifacts = artifacts[DatasetSplit.TRAIN.value]
    validation_artifacts = artifacts[DatasetSplit.VALIDATION.value]
    phase_reset_train_artifacts = [
        item
        for item in train_artifacts
        if item.metadata.get(
            "c3_train_rollout_mode", ORDER9_PHASE_RESET_ROLLOUT_MODE
        )
        == ORDER9_PHASE_RESET_ROLLOUT_MODE
    ]
    state_inheritance_train_artifacts = [
        item
        for item in train_artifacts
        if item.metadata.get(
            "c3_train_rollout_mode", ORDER9_PHASE_RESET_ROLLOUT_MODE
        )
        == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
    ]
    phase_reset_train_environment_count = sum(
        item.environment_count for item in phase_reset_train_artifacts
    )
    validation_environment_count = sum(
        item.environment_count for item in validation_artifacts
    )
    if (
        phase_reset_train_environment_count != runtime.environment_count
        or validation_environment_count != runtime.environment_count
    ):
        raise SchemaValidationError(
            "Order9 tensor dataset split environment budget differs from stage"
        )
    inheritance_enabled = bool(
        stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
        and config.production_runtime.c3_state_inheritance_rollouts_enabled
    )
    expected_inheritance_count = (
        sum(
            config.production_runtime
            .c3_state_inheritance_topology_count(module_count)
            for module_count in training_module_counts
        )
        if inheritance_enabled
        else 0
    )
    expected_inheritance_module_counts = sorted(
        module_count
        for module_count in training_module_counts
        for _ in range(
            config.production_runtime
            .c3_state_inheritance_topology_count(module_count)
            if inheritance_enabled
            else 0
        )
    )
    if (
        len(state_inheritance_train_artifacts) != expected_inheritance_count
        or any(
            artifact.environment_count
            != config.production_runtime
            .c3_state_inheritance_environment_count_per_module
            for artifact in state_inheritance_train_artifacts
        )
        or sorted(
            len(artifact.metadata["morphology_graph"]["modules"])
            for artifact in state_inheritance_train_artifacts
        )
        != expected_inheritance_module_counts
    ):
        raise SchemaValidationError(
            "Order9 state-inheritance train coverage is incomplete"
        )
    topology_stratified = len(train_artifacts) > 1
    topologies_per_module_count = 1
    train_module_counts = [
        len(artifact.metadata["morphology_graph"]["modules"])
        for artifact in train_artifacts
    ]
    if topology_stratified:
        topologies_per_module_count = (
            config.production_runtime.c3_topologies_per_module_count_per_update
        )
        expected_phase_reset_module_counts = sorted(
            module_count
            for module_count in training_module_counts
            for _ in range(topologies_per_module_count)
        )
        shard_indices: list[int] = []
        for artifact in train_artifacts:
            contract = artifact.metadata.get("topology_stratified_shard")
            if (
                not isinstance(contract, dict)
                or contract.get("enabled") is not True
                or int(contract.get("count", 0)) != len(train_artifacts)
                or int(contract.get("environment_count", 0))
                != artifact.environment_count
                or int(contract.get("configured_total_environment_count", 0))
                != runtime.environment_count
            ):
                raise SchemaValidationError(
                    "Order9 train artifact lacks production topology-shard contract"
                )
            shard_indices.append(int(contract.get("index", -1)))
        if (
            sorted(shard_indices) != list(range(len(train_artifacts)))
            or sorted(
                len(artifact.metadata["morphology_graph"]["modules"])
                for artifact in phase_reset_train_artifacts
            )
            != expected_phase_reset_module_counts
        ):
            raise SchemaValidationError(
                "Order9 topology-stratified train coverage is incomplete"
            )
    elif int(train_artifacts[0].metadata.get("environment_count", 0)) != runtime.environment_count:
        raise SchemaValidationError("Order9 raw train artifact runtime differs from stage")
    if any(
        int(artifact.metadata.get("environment_count", 0)) != artifact.environment_count
        for artifact in validation_artifacts
    ):
        raise SchemaValidationError("Order9 raw validation artifact runtime differs")
    manifest = Order9TensorPiLDatasetManifest(
        dataset_version=ORDER9_TENSOR_ON_POLICY_DATASET_VERSION,
        generation_id=generation_id,
        stage_id=stage.stage_id,
        stage_config_hash=str(expected["stage_config_hash"]),
        curriculum_schedule_hash=str(expected["curriculum_schedule_hash"]),
        config_hash=str(expected["config_hash"]),
        physical_model_hash=str(expected["physical_model_hash"]),
        urdf_hash=str(expected["urdf_hash"]),
        thrust_model_hash=str(expected["thrust_model_hash"]),
        behavior_checkpoint_sha256=checkpoint_sha,
        environment_step_count=sum(row.environment_step_count for row in shard_rows),
        shards=sorted(shard_rows, key=lambda row: row.split),
        metadata={
            "one_fresh_generation": True,
            "tensor_native_ppo": True,
            "record_schema_materialization": False,
            "jsonl_materialization": False,
            "topology_stratified_train": topology_stratified,
            "train_shard_count": len(train_artifacts),
            "validation_shard_count": len(validation_artifacts),
            "train_module_counts": train_module_counts,
            "topologies_per_module_count": topologies_per_module_count,
            "phase_reset_train_shard_count": len(
                phase_reset_train_artifacts
            ),
            "state_inheritance_train_shard_count": len(
                state_inheritance_train_artifacts
            ),
            "phase_reset_topologies_per_module_count": (
                topologies_per_module_count
            ),
            "state_inheritance_shards_per_module_count": (
                {
                    str(module_count): config.production_runtime
                    .c3_state_inheritance_topology_count(module_count)
                    for module_count in training_module_counts
                }
                if inheritance_enabled
                else {}
            ),
            "c3_training_module_counts": list(training_module_counts),
            "c3_module_subcurriculum": list(training_module_counts)
            != list(range(stage.min_modules, stage.max_modules + 1)),
            "phase_reset_train_environment_step_count": sum(
                artifact.environment_step_count
                for artifact in phase_reset_train_artifacts
            ),
            "state_inheritance_train_environment_step_count": sum(
                artifact.environment_step_count
                for artifact in state_inheritance_train_artifacts
            ),
            "state_inheritance_initial_phase_indices": (
                list(
                    config.production_runtime
                    .c3_state_inheritance_initial_phase_indices
                )
                if inheritance_enabled
                else []
            ),
            "state_inheritance_fixed_reset_progress_fraction": (
                config.production_runtime
                .c3_state_inheritance_fixed_reset_progress_fraction
                if inheritance_enabled
                else None
            ),
            "state_inheritance_preserves_phase_transitions": (
                inheritance_enabled
            ),
            "state_inheritance_recurrent_state_reset_only_at_episode_boundary": (
                inheritance_enabled
            ),
            "topology_environment_count_by_module": {
                str(module_count): sum(
                    artifact.environment_count
                    for artifact in train_artifacts
                    if len(artifact.metadata["morphology_graph"]["modules"])
                    == module_count
                )
                for module_count in sorted(set(train_module_counts))
            },
        },
    )
    manifest.validate()
    _atomic_write_text(manifest_path, manifest.to_json(indent=2) + "\n")
    return Order9TensorPiLDatasetBundle(
        manifest=manifest,
        manifest_path=str(manifest_path),
        manifest_sha256=hash_file(manifest_path),
        artifacts_by_split={
            split: tuple(values) for split, values in artifacts.items()
        },
        verified_shard_sha256=verified,
    )


def load_order9_tensor_pi_l_dataset(
    path: str | Path,
) -> Order9TensorPiLDatasetBundle:
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = Order9TensorPiLDatasetManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest.validate()
    artifacts: dict[str, list[Order9TensorRolloutArtifact]] = {
        DatasetSplit.TRAIN.value: [],
        DatasetSplit.VALIDATION.value: [],
    }
    verified: dict[str, str] = {}
    for shard in manifest.shards:
        source = Path(shard.path).resolve()
        artifact = load_order9_tensor_rollout_artifact(
            source, expected_sha256=shard.sha256
        )
        _validate_shard_against_artifact(shard, artifact)
        artifacts[shard.split].append(artifact)
        verified[str(source)] = shard.sha256
    return Order9TensorPiLDatasetBundle(
        manifest=manifest,
        manifest_path=str(manifest_path),
        manifest_sha256=hash_file(manifest_path),
        artifacts_by_split={
            split: tuple(values) for split, values in artifacts.items()
        },
        verified_shard_sha256=verified,
    )


def validate_order9_tensor_pi_l_dataset_for_stage(
    bundle: Order9TensorPiLDatasetBundle,
    *,
    config: Order9LearningConfig,
    stage_id: str,
    behavior_checkpoint_sha256: str,
) -> Order9DatasetStageValidation:
    stage = order9_stage_by_id(config, stage_id)
    failures: list[str] = []
    manifest = bundle.manifest
    expected = {
        "stage_id": stage.stage_id,
        "stage_config_hash": stable_hash(stage.to_dict()),
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "config_hash": stable_hash(config.to_dict()),
        "behavior_checkpoint_sha256": behavior_checkpoint_sha256,
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            failures.append(f"{name}_mismatch")
    train = bundle.train_artifacts
    if any(
        artifact.metadata.get("pi_l_checkpoint_sha256")
        != behavior_checkpoint_sha256
        for artifact in train
    ):
        failures.append("train_behavior_checkpoint_mismatch")
    if any(
        artifact.artifact_version not in _SUPPORTED_PPO_ARTIFACT_VERSIONS
        for artifact in train
    ):
        failures.append("train_artifact_version_mismatch")
    inheritance_required = bool(
        stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
        and config.production_runtime.c3_state_inheritance_rollouts_enabled
    )
    inheritance_artifacts = [
        artifact
        for artifact in train
        if artifact.metadata.get(
            "c3_train_rollout_mode", ORDER9_PHASE_RESET_ROLLOUT_MODE
        )
        == ORDER9_STATE_INHERITANCE_ROLLOUT_MODE
    ]
    try:
        training_module_counts = _resolve_c3_training_module_counts(
            stage_id=stage.stage_id,
            stage_minimum_module_count=stage.min_modules,
            stage_maximum_module_count=stage.max_modules,
            requested=manifest.metadata.get("c3_training_module_counts"),
        )
    except (TypeError, ValueError):
        training_module_counts = tuple()
        failures.append("c3_training_module_counts_invalid")
    if inheritance_required and (
        manifest.dataset_version != ORDER9_TENSOR_ON_POLICY_DATASET_VERSION
        or len(inheritance_artifacts)
        != sum(
            config.production_runtime
            .c3_state_inheritance_topology_count(module_count)
            for module_count in training_module_counts
        )
        or sorted(
            int(artifact.metadata["morphology_graph"]["module_count"])
            if "module_count" in artifact.metadata["morphology_graph"]
            else len(artifact.metadata["morphology_graph"]["modules"])
            for artifact in inheritance_artifacts
        )
        != sorted(
            module_count
            for module_count in training_module_counts
            for _ in range(
                config.production_runtime
                .c3_state_inheritance_topology_count(module_count)
            )
        )
        or manifest.metadata.get(
            "state_inheritance_preserves_phase_transitions"
        )
        is not True
        or manifest.metadata.get(
            "state_inheritance_recurrent_state_reset_only_at_episode_boundary"
        )
        is not True
        or manifest.metadata.get(
            "state_inheritance_fixed_reset_progress_fraction"
        )
        != config.production_runtime
        .c3_state_inheritance_fixed_reset_progress_fraction
    ):
        failures.append("state_inheritance_contract_mismatch")
    episode_count = sum(row.episode_count for row in manifest.shards)
    task_count = sum(row.task_count for row in manifest.shards)
    result = Order9DatasetStageValidation(
        stage_id=stage.stage_id,
        valid=not failures,
        failures=failures,
        record_count=manifest.environment_step_count,
        episode_count=episode_count,
        task_count=task_count,
        stochastic_record_count=manifest.environment_step_count,
        deterministic_teacher_record_count=0,
        metadata={
            "tensor_native_ppo": True,
            "train_environment_step_count": sum(
                artifact.environment_step_count for artifact in train
            ),
            "validation_environment_step_count": sum(
                artifact.environment_step_count
                for artifact in bundle.validation_artifacts
            ),
            "train_topology_shard_count": len(train),
            "state_inheritance_shard_count": len(inheritance_artifacts),
            "state_inheritance_environment_step_count": sum(
                artifact.environment_step_count
                for artifact in inheritance_artifacts
            ),
            "topology_stratified_train": (
                manifest.metadata.get("topology_stratified_train") is True
            ),
            "exact_behavior_fields_validated_in_artifact": True,
            "c3_training_module_counts": list(training_module_counts),
        },
    )
    result.validate()
    return result


def _validate_shard_against_artifact(
    shard: Order9TensorRolloutShard, artifact: Order9TensorRolloutArtifact
) -> None:
    values = {
        "artifact_version": artifact.artifact_version,
        "environment_count": artifact.environment_count,
        "rollout_step_count": artifact.step_count,
        "environment_step_count": artifact.environment_step_count,
        "morphology_hash": stable_hash(artifact.metadata["morphology_graph"]),
        "random_seed": int(artifact.metadata["random_seed"]),
        "simulator_version": str(artifact.metadata["simulator_version"]),
        "simulator_hash": str(artifact.metadata["simulator_hash"]),
        "collector_version": str(artifact.metadata.get("collector_version", "unknown")),
        "robot_usd_sha256": str(artifact.metadata["robot_usd_sha256"]),
        "rollout_mode": str(
            artifact.metadata.get(
                "c3_train_rollout_mode", ORDER9_PHASE_RESET_ROLLOUT_MODE
            )
        ),
        "module_count": len(artifact.metadata["morphology_graph"]["modules"]),
    }
    for name, value in values.items():
        if getattr(shard, name) != value:
            raise SchemaValidationError(f"Order9 tensor shard differs at {name}")
    if {str(value) for value in artifact.metadata["environment_splits"]} != {
        shard.split
    }:
        raise SchemaValidationError("Order9 tensor shard split metadata differs")


def _resolve_c3_training_module_counts(
    *,
    stage_id: str,
    stage_minimum_module_count: int,
    stage_maximum_module_count: int,
    requested: Sequence[int] | object | None,
) -> tuple[int, ...]:
    full = tuple(range(stage_minimum_module_count, stage_maximum_module_count + 1))
    if stage_id != "c3_pi_l_ppo_arbitrary_morphology":
        if requested is not None and tuple(int(value) for value in requested) != full:
            raise ValueError("Order9 module subcurriculum is restricted to C3")
        return full
    if requested is None:
        return full
    if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
        raise TypeError("Order9 C3 training module counts must be a sequence")
    counts = tuple(int(value) for value in requested)
    if (
        not counts
        or counts != tuple(range(counts[0], counts[-1] + 1))
        or counts[0] < stage_minimum_module_count
        or counts[-1] > stage_maximum_module_count
    ):
        raise ValueError("Order9 C3 training module counts must be contiguous and in-stage")
    return counts


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"{label} must be a lowercase SHA-256 digest")


__all__ = [
    "ORDER9_TENSOR_ON_POLICY_DATASET_VERSION",
    "ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V1",
    "ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V2",
    "ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V3",
    "ORDER9_TENSOR_ON_POLICY_DATASET_VERSION_V4",
    "ORDER9_PHASE_RESET_ROLLOUT_MODE",
    "ORDER9_STATE_INHERITANCE_ROLLOUT_MODE",
    "Order9TensorPiLDatasetBundle",
    "Order9TensorPiLDatasetManifest",
    "Order9TensorRolloutShard",
    "build_order9_tensor_pi_l_dataset",
    "load_order9_tensor_pi_l_dataset",
    "validate_order9_tensor_pi_l_dataset_for_stage",
]
