from __future__ import annotations

"""Fail-closed orchestration helpers for Order 9 ``pi_L`` PPO stages.

The simulator, dataset builder, and PPO trainer remain separate production
programs.  This module only binds their immutable artifacts into a resumable
one-generation/one-update sequence.
"""

import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_curriculum import (
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_dataset import load_order9_dataset_index
from amsrr.training.order9_online_training import Order9OnlineTrainingResult
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_randomization import Order9ConservativeRandomizer
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
    load_order9_pi_l_rollout_bucket_manifest,
    order9_pi_l_collector_arguments,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PI_L_STAGE_RUNNER_VERSION = "order9_pi_l_ppo_stage_runner_v1"
ORDER9_ROLLOUT_RESULT_PREFIX = "ORDER9_ROLLOUT_JSON="


@dataclass(frozen=True)
class Order9PiLStagePlan:
    stage_id: str
    target_environment_steps: int
    generation_environment_steps: int
    target_update_count: int
    next_update_index: int
    completed_environment_steps: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str


@dataclass(frozen=True)
class Order9CommandResult:
    command: tuple[str, ...]
    wall_elapsed_s: float
    return_code: int


def required_order9_update_count(
    target_environment_steps: int, generation_environment_steps: int
) -> int:
    if target_environment_steps < 1 or generation_environment_steps < 1:
        raise ValueError("Order9 PPO stage/generation step counts must be positive")
    return math.ceil(target_environment_steps / generation_environment_steps)


def select_order9_pi_l_rollout_buckets(
    manifest: Order9PiLRolloutBucketManifest,
    update_index: int,
) -> tuple[Order9PiLRolloutBucket, Order9PiLRolloutBucket]:
    if update_index < 0:
        raise ValueError("Order9 PPO update index must be non-negative")
    train = sorted(
        (bucket for bucket in manifest.buckets if bucket.split == DatasetSplit.TRAIN),
        key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
    )
    validation = sorted(
        (
            bucket
            for bucket in manifest.buckets
            if bucket.split == DatasetSplit.VALIDATION
        ),
        key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
    )
    if not train or not validation:
        raise SchemaValidationError(
            "Order9 pi_L runner requires train and validation rollout buckets"
        )
    return train[update_index % len(train)], validation[update_index % len(validation)]


def resolve_order9_pi_l_stage_plan(
    config: Order9LearningConfig,
    *,
    stage_id: str,
    stage_root: str | Path,
    initial_checkpoint_path: str | Path,
    repository_root: str | Path,
) -> Order9PiLStagePlan:
    config.validate()
    stage = order9_stage_by_id(config, stage_id)
    if (
        stage.learning_mode != Order9LearningMode.PPO
        or stage.learning_target != Order9LearningTarget.PI_L
    ):
        raise SchemaValidationError("Order9 pi_L stage runner requires a pi_L PPO stage")
    runtime = resolve_order9_stage_runtime(config, stage)
    split_generation_steps = runtime.generation_environment_steps
    if split_generation_steps is None:
        raise SchemaValidationError("Order9 pi_L PPO generation size is missing")
    # The tensor runtime size is per split.  One immutable production
    # generation deliberately contains one train and one validation shard, and
    # the existing dataset/checkpoint lineage counts both shards as consumed
    # environment steps.
    generation_steps = 2 * split_generation_steps
    target_updates = required_order9_update_count(
        stage.environment_steps, generation_steps
    )
    repository = Path(repository_root).resolve()
    root = _resolve(stage_root, repository)
    initial = _resolve(initial_checkpoint_path, repository)
    if not initial.is_file():
        raise FileNotFoundError(initial)
    initial_loaded = load_order9_policy_checkpoint(
        initial,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=order9_schedule_hash(config),
    )
    expected_parent_sha = initial_loaded.sha256
    parent_path = initial
    completed_steps = 0
    next_index = 0
    for update_index in range(target_updates):
        result_path = (
            root
            / f"update_{update_index:06d}"
            / f"training_result_update_{update_index:06d}.json"
        )
        if not result_path.is_file():
            break
        result = Order9OnlineTrainingResult.from_json(
            result_path.read_text(encoding="utf-8")
        )
        result.validate()
        if result.stage_id != stage_id or result.update_index != update_index:
            raise SchemaValidationError(
                f"Order9 completed update {update_index} identity mismatch"
            )
        if result.policy_family != Order9PolicyFamily.PI_L:
            raise SchemaValidationError(
                f"Order9 completed update {update_index} is not pi_L"
            )
        if result.parent_checkpoint_sha256 != expected_parent_sha:
            raise SchemaValidationError(
                f"Order9 completed update {update_index} parent lineage mismatch"
            )
        checkpoint = _resolve(result.checkpoint_path, repository)
        if hash_file(checkpoint) != result.checkpoint_sha256:
            raise SchemaValidationError(
                f"Order9 completed update {update_index} checkpoint hash mismatch"
            )
        if result.consumed_environment_steps != generation_steps:
            raise SchemaValidationError(
                f"Order9 completed update {update_index} generation size mismatch"
            )
        completed_steps += result.consumed_environment_steps
        expected_parent_sha = result.checkpoint_sha256
        parent_path = checkpoint
        next_index = update_index + 1
    return Order9PiLStagePlan(
        stage_id=stage_id,
        target_environment_steps=stage.environment_steps,
        generation_environment_steps=generation_steps,
        target_update_count=target_updates,
        next_update_index=next_index,
        completed_environment_steps=completed_steps,
        parent_checkpoint_path=str(parent_path),
        parent_checkpoint_sha256=expected_parent_sha,
    )


def validate_order9_pi_l_stage_runner_inputs(
    config: Order9LearningConfig,
    *,
    stage_id: str,
    bucket_manifest_path: str | Path,
    repository_root: str | Path,
) -> Order9PiLRolloutBucketManifest:
    repository = Path(repository_root).resolve()
    manifest_path = _resolve(bucket_manifest_path, repository)
    validate_order9_pi_l_rollout_bucket_bytes(
        manifest_path,
        repository_root=repository,
    )
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    stage = order9_stage_by_id(config, stage_id)
    physical = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path, repository)
    )
    expected = {
        "stage_id": stage.stage_id,
        "stage_config_hash": stable_hash(stage.to_dict()),
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "physical_model_hash": physical.stable_hash(),
        "topology_randomized": stage.topology_randomized,
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            raise SchemaValidationError(
                f"Order9 rollout bucket manifest differs at {name}"
            )
    _validate_current_bucket_randomization(
        config,
        manifest,
        manifest_path=manifest_path,
        repository=repository,
    )
    return manifest


def order9_pi_l_collector_command(
    *,
    python_executable: str | Path,
    repository_root: str | Path,
    config_path: str | Path,
    stage_id: str,
    parent_checkpoint_path: str | Path,
    parent_checkpoint_sha256: str,
    generation_id: str,
    output_raw_path: str | Path,
    bucket: Order9PiLRolloutBucket,
    bucket_manifest_path: str | Path,
) -> list[str]:
    repository = Path(repository_root).resolve()
    return [
        str(python_executable),
        str(repository / "scripts/order9_vectorized_isaac_rollout.py"),
        "--config",
        str(_resolve(config_path, repository)),
        "--stage",
        stage_id,
        "--pi-l-checkpoint",
        str(_resolve(parent_checkpoint_path, repository)),
        "--pi-l-checkpoint-sha256",
        parent_checkpoint_sha256,
        "--generation-id",
        generation_id,
        "--output-raw",
        str(_resolve(output_raw_path, repository)),
        *order9_pi_l_collector_arguments(
            bucket,
            bucket_manifest_path=_resolve(bucket_manifest_path, repository),
            repository_root=repository,
        ),
    ]


def run_logged_order9_command(
    command: Sequence[str],
    *,
    repository_root: str | Path,
    log_path: str | Path,
    append: bool = False,
) -> Order9CommandResult:
    repository = Path(repository_root).resolve()
    log = _resolve(log_path, repository)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        _write_command_header(handle, command)
        completed = subprocess.run(
            list(command),
            cwd=repository,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Order9 command failed with {completed.returncode}: {shlex.join(command)}; "
            f"log={log}"
        )
    return Order9CommandResult(tuple(command), elapsed, completed.returncode)


def run_parallel_order9_collectors(
    commands: Mapping[str, Sequence[str]],
    *,
    repository_root: str | Path,
    log_paths: Mapping[str, str | Path],
) -> tuple[float, dict[str, int]]:
    if not commands or set(commands) != set(log_paths):
        raise ValueError("Order9 collector commands/log paths must be non-empty and aligned")
    repository = Path(repository_root).resolve()
    handles: dict[str, TextIO] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    started = time.perf_counter()
    try:
        for split, command in commands.items():
            log = _resolve(log_paths[split], repository)
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            handles[split] = handle
            _write_command_header(handle, command)
            processes[split] = subprocess.Popen(
                list(command),
                cwd=repository,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        return_codes = {split: process.wait() for split, process in processes.items()}
    finally:
        for handle in handles.values():
            handle.close()
    elapsed = time.perf_counter() - started
    failures = {split: code for split, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"Order9 parallel collectors failed: {failures}")
    return elapsed, return_codes


def load_order9_rollout_result(log_path: str | Path) -> dict[str, Any]:
    path = Path(log_path)
    payload: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(ORDER9_ROLLOUT_RESULT_PREFIX):
                raw = json.loads(line[len(ORDER9_ROLLOUT_RESULT_PREFIX) :])
                if not isinstance(raw, dict):
                    raise SchemaValidationError("Order9 rollout result is not an object")
                payload = raw
    if payload is None:
        raise SchemaValidationError(f"Order9 rollout result is missing from {path}")
    return payload


def validate_order9_rollout_result(
    payload: Mapping[str, Any],
    *,
    stage_id: str,
    generation_id: str,
    split: DatasetSplit,
    expected_environment_steps: int,
    raw_artifact_path: str | Path,
    parent_checkpoint_sha256: str,
) -> str:
    expected = {
        "stage_id": stage_id,
        "generation_id": generation_id,
        "split": split.value,
        "environment_steps": expected_environment_steps,
        "passed": True,
        "finite_state": True,
        "runtime_override_used": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise SchemaValidationError(f"Order9 rollout result differs at {name}")
    raw = Path(raw_artifact_path).resolve()
    if Path(str(payload.get("raw_artifact_path", ""))).resolve() != raw:
        raise SchemaValidationError("Order9 rollout result raw artifact path mismatch")
    digest = hash_file(raw)
    if payload.get("raw_artifact_sha256") != digest:
        raise SchemaValidationError("Order9 rollout result raw artifact hash mismatch")
    artifact = load_order9_tensor_rollout_artifact(raw, expected_sha256=digest)
    metadata = artifact.metadata
    if metadata.get("pi_l_checkpoint_sha256") != parent_checkpoint_sha256:
        raise SchemaValidationError("Order9 rollout behavior checkpoint mismatch")
    if metadata.get("generation_id") != generation_id:
        raise SchemaValidationError("Order9 raw rollout generation mismatch")
    if metadata.get("runtime_override_used") is not False:
        raise SchemaValidationError("Order9 raw rollout used a runtime override")
    return digest


def validate_order9_generation_dataset(
    manifest_path: str | Path,
    *,
    stage_id: str,
    generation_id: str,
    parent_checkpoint_sha256: str,
    expected_environment_steps: int,
) -> str:
    index = load_order9_dataset_index(manifest_path)
    metadata = index.manifest.metadata
    expected = {
        "stage_id": stage_id,
        "generation_id": generation_id,
        "on_policy_environment_step_count": expected_environment_steps,
        "one_fresh_generation": True,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise SchemaValidationError(f"Order9 generation dataset differs at {name}")
    behavior = metadata.get("behavior_checkpoint_sha256_by_family")
    if not isinstance(behavior, dict) or behavior.get("pi_l") != parent_checkpoint_sha256:
        raise SchemaValidationError("Order9 generation dataset behavior checkpoint mismatch")
    return index.manifest_sha256


def validate_order9_completed_update(
    result_path: str | Path,
    *,
    repository_root: str | Path,
    stage_id: str,
    update_index: int,
    parent_checkpoint_sha256: str,
    rollout_manifest_sha256: str,
    expected_environment_steps: int,
) -> Order9OnlineTrainingResult:
    repository = Path(repository_root).resolve()
    path = _resolve(result_path, repository)
    result = Order9OnlineTrainingResult.from_json(path.read_text(encoding="utf-8"))
    result.validate()
    expected = {
        "stage_id": stage_id,
        "update_index": update_index,
        "policy_family": Order9PolicyFamily.PI_L,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "rollout_manifest_sha256": rollout_manifest_sha256,
        "consumed_environment_steps": expected_environment_steps,
    }
    for name, value in expected.items():
        if getattr(result, name) != value:
            raise SchemaValidationError(f"Order9 PPO update differs at {name}")
    checkpoint = _resolve(result.checkpoint_path, repository)
    metrics = _resolve(result.metrics_path, repository)
    if hash_file(checkpoint) != result.checkpoint_sha256:
        raise SchemaValidationError("Order9 PPO child checkpoint hash mismatch")
    if hash_file(metrics) != result.metrics_sha256:
        raise SchemaValidationError("Order9 PPO metrics hash mismatch")
    replay = result.ppo_update.metadata
    if replay.get("exact_behavior_replay_validated") is not True:
        raise SchemaValidationError("Order9 PPO exact behavior replay was not validated")
    tolerance = float(replay.get("exact_replay_absolute_tolerance", 0.0))
    for name in (
        "maximum_log_prob_replay_error",
        "maximum_recurrent_replay_error",
        "maximum_value_replay_error",
    ):
        value = float(replay.get(name, math.inf))
        if not math.isfinite(value) or value > tolerance:
            raise SchemaValidationError(f"Order9 PPO exact replay failed at {name}")
    for value in (
        result.ppo_update.actor_loss,
        result.ppo_update.value_loss,
        result.ppo_update.total_loss,
        result.ppo_update.approximate_kl,
        result.ppo_update.clipped_fraction,
    ):
        if not math.isfinite(float(value)):
            raise SchemaValidationError("Order9 PPO update contains non-finite metrics")
    return result


def write_order9_stage_runner_state(
    path: str | Path,
    *,
    stage_id: str,
    status: str,
    plan: Order9PiLStagePlan,
    current_update_index: int | None,
    completed_updates: Sequence[Mapping[str, Any]],
    failure: str | None = None,
) -> None:
    if status not in {"running", "completed", "failed"}:
        raise ValueError("Order9 stage runner status is invalid")
    payload = {
        "runner_version": ORDER9_PI_L_STAGE_RUNNER_VERSION,
        "stage_id": stage_id,
        "status": status,
        "target_environment_steps": plan.target_environment_steps,
        "generation_environment_steps": plan.generation_environment_steps,
        "target_update_count": plan.target_update_count,
        "initial_next_update_index": plan.next_update_index,
        "initial_completed_environment_steps": plan.completed_environment_steps,
        "current_update_index": current_update_index,
        "completed_updates_this_run": list(completed_updates),
        "failure": failure,
        "updated_unix_time_s": time.time(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _write_command_header(handle: TextIO, command: Sequence[str]) -> None:
    handle.write(f"ORDER9_COMMAND={shlex.join(command)}\n")
    handle.flush()


def _validate_current_bucket_randomization(
    config: Order9LearningConfig,
    manifest: Order9PiLRolloutBucketManifest,
    *,
    manifest_path: Path,
    repository: Path,
) -> None:
    canonical = load_order9_canonical_reset(
        _resolve(config.production_runtime.canonical_order8_report_path, repository),
        expected_sha256=config.production_runtime.canonical_order8_report_sha256,
    )
    base_task = build_order8_grasp_carry_task_spec(
        object_pose_world=tuple(canonical.object_pose_world),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=canonical.transport_distance_m,
        support_height_m=config.randomization.support_top_z_m,
        max_contact_force_n=config.hard_checker.qp_force_scale_n,
        max_contact_torque_nm=config.hard_checker.qp_torque_scale_nm,
        selected_gripper_friction=(
            config.randomization.nominal_selected_gripper_friction
        ),
        task_id="order9-vectorized-base",
    )
    if manifest.metadata.get("base_task_hash") != base_task.stable_hash():
        raise SchemaValidationError("Order9 rollout bucket base task changed")
    randomizer = Order9ConservativeRandomizer(config.randomization)
    for bucket in manifest.buckets:
        sample = randomizer.sample(
            base_task,
            seed=bucket.seed,
            sample_index=bucket.sample_index,
        )
        task = TaskSpec.from_dict(sample.task_spec.to_dict())
        task.metadata = {
            **task.metadata,
            "order9_rollout_bucket_id": bucket.bucket_id,
            "dataset_split": bucket.split.value,
            "estimated_mass_kg": sample.estimated_mass_properties.mass_kg,
            "estimated_inertia_body": list(
                sample.estimated_mass_properties.inertia_kgm2
            ),
            "estimated_com_object": list(
                sample.estimated_mass_properties.center_of_mass_object
            ),
        }
        persisted_task = TaskSpec.from_json(
            (manifest_path.parent / bucket.task_spec_path).read_text(encoding="utf-8")
        )
        expected = {
            "task_hash": task.stable_hash(),
            "selected_gripper_friction": sample.selected_gripper_friction,
            "contact_stiffness_n_per_m": sample.contact_stiffness_n_per_m,
            "contact_damping_n_s_per_m": sample.contact_damping_n_s_per_m,
            "estimated_mass_kg": sample.estimated_mass_properties.mass_kg,
            "estimated_inertia_body": list(
                sample.estimated_mass_properties.inertia_kgm2
            ),
            "estimated_com_object": list(
                sample.estimated_mass_properties.center_of_mass_object
            ),
            "randomization_version": sample.randomization_version,
            "sampled_values": sample.sampled_values,
            "true_mass_properties": sample.true_mass_properties.to_dict(),
            "estimated_mass_properties": (
                sample.estimated_mass_properties.to_dict()
            ),
        }
        actual = {
            "task_hash": persisted_task.stable_hash(),
            "selected_gripper_friction": bucket.selected_gripper_friction,
            "contact_stiffness_n_per_m": bucket.contact_stiffness_n_per_m,
            "contact_damping_n_s_per_m": bucket.contact_damping_n_s_per_m,
            "estimated_mass_kg": bucket.estimated_mass_kg,
            "estimated_inertia_body": bucket.estimated_inertia_body,
            "estimated_com_object": bucket.estimated_com_object,
            "randomization_version": bucket.randomization_version,
            "sampled_values": bucket.metadata.get("sampled_values"),
            "true_mass_properties": bucket.metadata.get("true_mass_properties"),
            "estimated_mass_properties": bucket.metadata.get(
                "estimated_mass_properties"
            ),
        }
        if actual != expected:
            differing = sorted(
                name for name in expected if actual.get(name) != expected[name]
            )
            raise SchemaValidationError(
                "Order9 rollout bucket current randomization mismatch: "
                f"{bucket.bucket_id}:{','.join(differing)}"
            )


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


__all__ = [
    "ORDER9_PI_L_STAGE_RUNNER_VERSION",
    "ORDER9_ROLLOUT_RESULT_PREFIX",
    "Order9CommandResult",
    "Order9PiLStagePlan",
    "load_order9_rollout_result",
    "order9_pi_l_collector_command",
    "required_order9_update_count",
    "resolve_order9_pi_l_stage_plan",
    "run_logged_order9_command",
    "run_parallel_order9_collectors",
    "select_order9_pi_l_rollout_buckets",
    "validate_order9_completed_update",
    "validate_order9_generation_dataset",
    "validate_order9_pi_l_stage_runner_inputs",
    "validate_order9_rollout_result",
    "write_order9_stage_runner_state",
]
