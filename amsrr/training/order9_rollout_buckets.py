from __future__ import annotations

"""Immutable topology/object buckets for real-Isaac Order 9 pi_L rollout."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
    validate_order9_morphology_asset_manifest_bytes,
)
from amsrr.training.order9_curriculum import (
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_randomization import Order9ConservativeRandomizer
from amsrr.training.order9_topology_provider import Order9CurriculumTopologyProvider
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_ROLLOUT_BUCKET_MANIFEST_VERSION = "order9_pi_l_rollout_buckets_v1"


@dataclass
class Order9PiLRolloutBucket(SchemaBase):
    bucket_id: str
    split: DatasetSplit
    seed: int
    sample_index: int
    task_id: str
    task_spec_path: str
    task_spec_sha256: str
    morphology_graph_path: str
    morphology_graph_sha256: str
    morphology_hash: str
    structural_hash: str
    module_count: int
    robot_usd_path: str
    robot_usd_sha256: str
    selected_gripper_friction: float
    contact_stiffness_n_per_m: float
    contact_damping_n_s_per_m: float
    estimated_mass_kg: float
    estimated_inertia_body: list[float]
    estimated_com_object: list[float]
    randomization_version: str
    topology_source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in (
            "bucket_id",
            "task_id",
            "task_spec_path",
            "morphology_graph_path",
            "robot_usd_path",
            "randomization_version",
            "topology_source",
        ):
            require_non_empty(
                str(getattr(self, name)), f"Order9PiLRolloutBucket.{name}"
            )
        for name in (
            "task_spec_sha256",
            "morphology_graph_sha256",
            "morphology_hash",
            "structural_hash",
            "robot_usd_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if self.seed < 0 or self.sample_index < 0:
            raise SchemaValidationError("Order9 rollout bucket seed/index is invalid")
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError("Order9 rollout bucket module count is invalid")
        for name in (
            "selected_gripper_friction",
            "contact_stiffness_n_per_m",
            "contact_damping_n_s_per_m",
            "estimated_mass_kg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9 rollout bucket {name} must be positive"
                )
        for name, values, width in (
            ("estimated_inertia_body", self.estimated_inertia_body, 6),
            ("estimated_com_object", self.estimated_com_object, 3),
        ):
            if len(values) != width or any(
                not math.isfinite(float(value)) for value in values
            ):
                raise SchemaValidationError(
                    f"Order9 rollout bucket {name} is invalid"
                )


@dataclass
class Order9PiLRolloutBucketManifest(SchemaBase):
    stage_id: str
    stage_config_hash: str
    curriculum_schedule_hash: str
    config_hash: str
    physical_model_hash: str
    topology_randomized: bool
    buckets: list[Order9PiLRolloutBucket]
    source_pool_path: str | None = None
    source_pool_sha256: str | None = None
    source_asset_manifest_path: str | None = None
    source_asset_manifest_sha256: str | None = None
    manifest_version: str = ORDER9_ROLLOUT_BUCKET_MANIFEST_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.manifest_version != ORDER9_ROLLOUT_BUCKET_MANIFEST_VERSION:
            raise SchemaValidationError("Order9 rollout bucket version mismatch")
        require_non_empty(self.stage_id, "Order9PiLRolloutBucketManifest.stage_id")
        for name in (
            "stage_config_hash",
            "curriculum_schedule_hash",
            "config_hash",
            "physical_model_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if not self.buckets:
            raise SchemaValidationError("Order9 rollout bucket manifest is empty")
        ids = [bucket.bucket_id for bucket in self.buckets]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Order9 rollout bucket ids repeat")
        splits = {bucket.split for bucket in self.buckets}
        if not {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}.issubset(splits):
            raise SchemaValidationError(
                "Order9 rollout buckets require train and validation splits"
            )
        if DatasetSplit.HELD_OUT in splits:
            raise SchemaValidationError(
                "Order9 pi_L training buckets cannot contain held-out tasks"
            )
        task_owners: dict[str, DatasetSplit] = {}
        structural_owners: dict[str, DatasetSplit] = {}
        for bucket in self.buckets:
            owner = task_owners.setdefault(bucket.task_id, bucket.split)
            if owner != bucket.split:
                raise SchemaValidationError("Order9 bucket task crosses splits")
            if self.topology_randomized:
                structural_owner = structural_owners.setdefault(
                    bucket.structural_hash, bucket.split
                )
                if structural_owner != bucket.split:
                    raise SchemaValidationError(
                        "Order9 bucket topology crosses splits"
                    )
        provenance = (
            self.source_pool_path,
            self.source_pool_sha256,
            self.source_asset_manifest_path,
            self.source_asset_manifest_sha256,
        )
        if self.topology_randomized and any(value is None for value in provenance):
            raise SchemaValidationError(
                "Order9 randomized topology buckets require pool/asset provenance"
            )
        for name in ("source_pool_sha256", "source_asset_manifest_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(str(value), name)


def prepare_order9_pi_l_rollout_buckets(
    output_dir: str | Path,
    *,
    config: Order9LearningConfig,
    stage_id: str,
    physical_model: PhysicalModel,
    base_task_spec: TaskSpec,
    train_bucket_count: int,
    validation_bucket_count: int,
    repository_root: str | Path,
    fixed_robot_usd_path: str | Path | None = None,
    morphology_pool_path: str | Path | None = None,
    morphology_asset_manifest_path: str | Path | None = None,
    seed: int | None = None,
) -> Order9PiLRolloutBucketManifest:
    """Materialize split-safe homogeneous simulator buckets before collection."""

    config.validate()
    physical_model.validate()
    base_task_spec.validate()
    stage = order9_stage_by_id(config, stage_id)
    if (
        stage.learning_mode != Order9LearningMode.PPO
        or stage.learning_target != Order9LearningTarget.PI_L
    ):
        raise SchemaValidationError(
            "Order9 rollout bucket preparation requires a pi_L PPO stage"
        )
    if train_bucket_count < 1 or validation_bucket_count < 1:
        raise ValueError("Order9 rollout bucket split counts must be positive")
    repository = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Order9 rollout bucket output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    pool_path: Path | None = None
    asset_path: Path | None = None
    pool_digest: str | None = None
    asset_digest: str | None = None
    provider: Order9CurriculumTopologyProvider | None = None
    asset_manifest: Order9MorphologyAssetManifest | None = None
    fixed_usd: Path | None = None
    try:
        if stage.topology_randomized:
            if morphology_pool_path is None or morphology_asset_manifest_path is None:
                raise SchemaValidationError(
                    "Order9 arbitrary-morphology buckets require pool and assets"
                )
            pool_path = _resolve(morphology_pool_path, repository)
            asset_path = _resolve(morphology_asset_manifest_path, repository)
            pool_digest = hash_file(pool_path)
            asset_digest = hash_file(asset_path)
            pool = Order3MorphologyPoolManifest.from_json(
                pool_path.read_text(encoding="utf-8")
            )
            provider = Order9CurriculumTopologyProvider(
                pool,
                physical_model=physical_model,
                pool_path=pool_path,
                pool_sha256=pool_digest,
            )
            asset_manifest = Order9MorphologyAssetManifest.from_json(
                asset_path.read_text(encoding="utf-8")
            )
            validate_order9_morphology_asset_manifest_bytes(
                asset_manifest,
                repository_root=repository,
                expected_pool_sha256=pool_digest,
            )
        else:
            if fixed_robot_usd_path is None:
                raise SchemaValidationError(
                    "Order9 fixed-topology buckets require a robot USD"
                )
            fixed_usd = _resolve(fixed_robot_usd_path, repository)
            if not fixed_usd.is_file():
                raise FileNotFoundError(fixed_usd)

        randomizer = Order9ConservativeRandomizer(config.randomization)
        fixed_graph = build_representative_order8_morphology(physical_model)
        requested = (
            (DatasetSplit.TRAIN, train_bucket_count),
            (DatasetSplit.VALIDATION, validation_bucket_count),
        )
        base_seed = (
            int(config.production_runtime.seed + stage.stage_index)
            if seed is None
            else int(seed)
        )
        if base_seed < 0:
            raise ValueError("Order9 rollout bucket seed must be non-negative")
        buckets: list[Order9PiLRolloutBucket] = []
        global_index = 0
        for split, count in requested:
            for split_index in range(count):
                sample_seed = base_seed + global_index
                randomization = randomizer.sample(
                    base_task_spec,
                    seed=sample_seed,
                    sample_index=global_index,
                )
                task = TaskSpec.from_dict(randomization.task_spec.to_dict())
                if stage.topology_randomized:
                    assert provider is not None and asset_manifest is not None
                    topology = provider.sample(
                        task,
                        split=split,
                        seed=sample_seed,
                        sample_index=global_index,
                        min_modules=stage.min_modules,
                        max_modules=stage.max_modules,
                    )
                    graph = topology.morphology_graph
                    asset_entry = asset_manifest.entry_for(graph)
                    if asset_entry.split != split:
                        raise SchemaValidationError(
                            "Order9 topology asset split differs from provider"
                        )
                    robot_usd = _resolve(asset_entry.usd_path, repository)
                    topology_source = "split_safe_pool_task_conditioned_teacher"
                    topology_metadata = topology.metadata
                else:
                    graph = fixed_graph
                    assert fixed_usd is not None
                    robot_usd = fixed_usd
                    topology_source = "canonical_order8_fixed_morphology"
                    topology_metadata = {"learned_pi_d_used": False}
                if not robot_usd.is_file():
                    raise FileNotFoundError(robot_usd)
                bucket_id = (
                    f"{split.value}-{global_index:06d}-"
                    f"{morphology_structural_hash(graph)[:12]}"
                )
                task.metadata = {
                    **task.metadata,
                    "order9_rollout_bucket_id": bucket_id,
                    "dataset_split": split.value,
                    "estimated_mass_kg": randomization.estimated_mass_properties.mass_kg,
                    "estimated_inertia_body": list(
                        randomization.estimated_mass_properties.inertia_kgm2
                    ),
                    "estimated_com_object": list(
                        randomization.estimated_mass_properties.center_of_mass_object
                    ),
                }
                task.validate()
                bucket_dir = temporary / "buckets" / bucket_id
                bucket_dir.mkdir(parents=True)
                task_path = bucket_dir / "task_spec.json"
                graph_path = bucket_dir / "morphology_graph.json"
                _write_new_text(task_path, task.to_json(indent=2) + "\n")
                _write_new_text(graph_path, graph.to_json(indent=2) + "\n")
                buckets.append(
                    Order9PiLRolloutBucket(
                        bucket_id=bucket_id,
                        split=split,
                        seed=sample_seed,
                        sample_index=global_index,
                        task_id=task.task_id,
                        task_spec_path=str(task_path.relative_to(temporary)),
                        task_spec_sha256=hash_file(task_path),
                        morphology_graph_path=str(graph_path.relative_to(temporary)),
                        morphology_graph_sha256=hash_file(graph_path),
                        morphology_hash=graph.stable_hash(),
                        structural_hash=morphology_structural_hash(graph),
                        module_count=len(graph.modules),
                        robot_usd_path=_portable(robot_usd, repository),
                        robot_usd_sha256=hash_file(robot_usd),
                        selected_gripper_friction=(
                            randomization.selected_gripper_friction
                        ),
                        contact_stiffness_n_per_m=(
                            randomization.contact_stiffness_n_per_m
                        ),
                        contact_damping_n_s_per_m=(
                            randomization.contact_damping_n_s_per_m
                        ),
                        estimated_mass_kg=(
                            randomization.estimated_mass_properties.mass_kg
                        ),
                        estimated_inertia_body=list(
                            randomization.estimated_mass_properties.inertia_kgm2
                        ),
                        estimated_com_object=list(
                            randomization.estimated_mass_properties.center_of_mass_object
                        ),
                        randomization_version=randomization.randomization_version,
                        topology_source=topology_source,
                        metadata={
                            "split_bucket_index": split_index,
                            "sampled_values": randomization.sampled_values,
                            "true_mass_properties": (
                                randomization.true_mass_properties.to_dict()
                            ),
                            "estimated_mass_properties": (
                                randomization.estimated_mass_properties.to_dict()
                            ),
                            "topology_provider": topology_metadata,
                        },
                    )
                )
                global_index += 1
        manifest = Order9PiLRolloutBucketManifest(
            stage_id=stage.stage_id,
            stage_config_hash=stable_hash(stage.to_dict()),
            curriculum_schedule_hash=order9_schedule_hash(config),
            config_hash=stable_hash(config.to_dict()),
            physical_model_hash=physical_model.stable_hash(),
            topology_randomized=bool(stage.topology_randomized),
            source_pool_path=(
                None if pool_path is None else _portable(pool_path, repository)
            ),
            source_pool_sha256=pool_digest,
            source_asset_manifest_path=(
                None if asset_path is None else _portable(asset_path, repository)
            ),
            source_asset_manifest_sha256=asset_digest,
            buckets=buckets,
            metadata={
                "base_task_hash": base_task_spec.stable_hash(),
                "base_seed": base_seed,
                "train_bucket_count": train_bucket_count,
                "validation_bucket_count": validation_bucket_count,
                "one_object_and_topology_per_simulator_bucket": True,
                "learned_pi_d_used": False,
            },
        )
        manifest.validate()
        _write_new_text(temporary / "manifest.json", manifest.to_json(indent=2) + "\n")
        os.rename(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_order9_pi_l_rollout_bucket_manifest(
    path: str | Path,
) -> Order9PiLRolloutBucketManifest:
    source = Path(path)
    manifest_path = source / "manifest.json" if source.is_dir() else source
    manifest = Order9PiLRolloutBucketManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest.validate()
    return manifest


def validate_order9_pi_l_rollout_bucket_bytes(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
) -> None:
    source = Path(manifest_path).resolve()
    if source.is_dir():
        source = source / "manifest.json"
    manifest = load_order9_pi_l_rollout_bucket_manifest(source)
    repository = Path(repository_root).resolve()
    for bucket in manifest.buckets:
        task_path = source.parent / bucket.task_spec_path
        graph_path = source.parent / bucket.morphology_graph_path
        robot_usd = _resolve(bucket.robot_usd_path, repository)
        for path, expected in (
            (task_path, bucket.task_spec_sha256),
            (graph_path, bucket.morphology_graph_sha256),
            (robot_usd, bucket.robot_usd_sha256),
        ):
            if hash_file(path) != expected:
                raise SchemaValidationError(
                    f"Order9 rollout bucket bytes changed: {path}"
                )
        task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
        graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
        if task.task_id != bucket.task_id or graph.stable_hash() != bucket.morphology_hash:
            raise SchemaValidationError("Order9 rollout bucket semantic hash changed")
        if morphology_structural_hash(graph) != bucket.structural_hash:
            raise SchemaValidationError("Order9 rollout bucket structure changed")


def order9_pi_l_collector_arguments(
    bucket: Order9PiLRolloutBucket,
    *,
    bucket_manifest_path: str | Path,
    repository_root: str | Path,
) -> list[str]:
    manifest_path = Path(bucket_manifest_path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    repository = Path(repository_root).resolve()
    task_path = manifest_path.parent / bucket.task_spec_path
    graph_path = manifest_path.parent / bucket.morphology_graph_path
    robot_usd = _resolve(bucket.robot_usd_path, repository)
    return [
        "--split",
        bucket.split.value,
        "--seed",
        str(bucket.seed),
        "--task-spec-json",
        str(task_path),
        "--morphology-graph-json",
        str(graph_path),
        "--robot-usd",
        str(robot_usd),
        "--selected-gripper-friction",
        repr(float(bucket.selected_gripper_friction)),
        "--contact-stiffness",
        repr(float(bucket.contact_stiffness_n_per_m)),
        "--contact-damping",
        repr(float(bucket.contact_damping_n_s_per_m)),
        "--estimated-mass-kg",
        repr(float(bucket.estimated_mass_kg)),
        "--estimated-inertia-body",
        *(repr(float(value)) for value in bucket.estimated_inertia_body),
        "--estimated-com-object",
        *(repr(float(value)) for value in bucket.estimated_com_object),
    ]


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


def _portable(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError:
        return str(path.resolve())


def _write_new_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"Order9 rollout bucket {name} is not SHA-256")


__all__ = [
    "ORDER9_ROLLOUT_BUCKET_MANIFEST_VERSION",
    "Order9PiLRolloutBucket",
    "Order9PiLRolloutBucketManifest",
    "load_order9_pi_l_rollout_bucket_manifest",
    "order9_pi_l_collector_arguments",
    "prepare_order9_pi_l_rollout_buckets",
    "validate_order9_pi_l_rollout_bucket_bytes",
]
