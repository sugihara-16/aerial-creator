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
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
    ORDER9_OBJECT_TASK_PHASES,
)
from amsrr.training.order9_curriculum import (
    Order9C3BoundaryFineTuneConfig,
    Order9LearningConfig,
    Order9LearningMode,
    Order9LearningTarget,
    require_order9_stage_execution_allowed,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_curriculum_lineage import (
    load_order9_stage_parent_checkpoint,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_c3_nominal_runtime import (
    ORDER9_C3_PHASE_RESET_REFERENCE_VERSION,
)
from amsrr.training.order9_c3_boundary_sampling import (
    ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT,
    ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT,
)
from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACTS,
    order9_c3_action_contract_global_dimension,
    order9_c3_action_contract_uses_full_policy,
)
from amsrr.training.order9_dataset import load_order9_dataset_index
from amsrr.training.order9_factorized_actor_credit import (
    ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS,
    ORDER9_CONTACT_COORDINATE_CREDIT_VERSION,
    ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS,
    ORDER9_FACTORIZED_ACTOR_CREDIT_VERSION,
)
from amsrr.training.order9_online_training import Order9OnlineTrainingResult
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_randomization import Order9ConservativeRandomizer
from amsrr.training.order9_rollout_buckets import (
    ORDER9_C3_BUCKET_PRECHECK_VERSION,
    ORDER9_C3_STAGE_ID,
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
    load_order9_pi_l_rollout_bucket_manifest,
    order9_pi_l_collector_arguments,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    load_order9_tensor_rollout_artifact,
)
from amsrr.training.order9_tensor_pi_l_ppo import (
    ORDER9_C3_BOUNDARY_FINE_TUNE_CONTRACT,
    ORDER9_CONTACT_HEAD_EXTRA_OPTIMIZER_CONTRACT,
    ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES,
    ORDER9_TOPOLOGY_MINIBATCH_PACKING,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PI_L_STAGE_RUNNER_VERSION = (
    "order9_pi_l_ppo_stage_runner_v19_contact_coordinate_credit"
)
ORDER9_ROLLOUT_RESULT_PREFIX = "ORDER9_ROLLOUT_JSON="
ORDER9_C3_PHASE_RESET_ROLLOUT_MODE = "phase_reset"
ORDER9_C3_STATE_INHERITANCE_ROLLOUT_MODE = "continuous_state_inheritance"
ORDER9_PERSISTENT_BUCKET_JOBS_VERSION = (
    "order9_persistent_same_morphology_bucket_jobs_v1"
)


@dataclass(frozen=True)
class Order9PiLStagePlan:
    stage_id: str
    configured_target_environment_steps: int
    target_environment_steps: int
    generation_environment_steps: int
    base_target_update_count: int
    additional_update_count: int
    target_update_count: int
    next_update_index: int
    completed_environment_steps: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    branch_parent_update_index: int | None = None
    training_minimum_module_count: int | None = None
    training_maximum_module_count: int | None = None


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


def resolve_order9_extended_update_budget(
    configured_target_environment_steps: int,
    generation_environment_steps: int,
    additional_update_count: int,
) -> tuple[int, int, int]:
    """Resolve a post-quota extension without changing the schedule hash."""

    if additional_update_count < 0:
        raise ValueError("Order9 additional update count must be non-negative")
    base_update_count = required_order9_update_count(
        configured_target_environment_steps,
        generation_environment_steps,
    )
    target_update_count = base_update_count + additional_update_count
    target_environment_steps = target_update_count * generation_environment_steps
    return base_update_count, target_update_count, target_environment_steps


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


def select_order9_pi_l_topology_stratified_buckets(
    manifest: Order9PiLRolloutBucketManifest,
    update_index: int,
    *,
    minimum_module_count: int = 2,
    maximum_module_count: int = 8,
    topologies_per_module_count: int = 1,
) -> tuple[tuple[Order9PiLRolloutBucket, ...], Order9PiLRolloutBucket]:
    """Select fresh train topologies within every module-count stratum."""

    if update_index < 0:
        raise ValueError("Order9 PPO update index must be non-negative")
    if (
        minimum_module_count < 1
        or maximum_module_count < minimum_module_count
        or topologies_per_module_count < 1
    ):
        raise ValueError("Order9 topology stratum range is invalid")
    selected: list[Order9PiLRolloutBucket] = []
    for module_count in range(minimum_module_count, maximum_module_count + 1):
        candidates = sorted(
            (
                bucket
                for bucket in manifest.buckets
                if bucket.split == DatasetSplit.TRAIN
                and bucket.module_count == module_count
            ),
            key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
        )
        if not candidates:
            raise SchemaValidationError(
                "Order9 topology-stratified runner lacks train bucket for "
                f"module_count={module_count}"
            )
        if len(candidates) < topologies_per_module_count:
            raise SchemaValidationError(
                "Order9 topology-stratified runner lacks distinct train "
                f"buckets for module_count={module_count}"
            )
        start = (update_index * topologies_per_module_count) % len(candidates)
        selected.extend(
            candidates[(start + offset) % len(candidates)]
            for offset in range(topologies_per_module_count)
        )
    validation = sorted(
        (
            bucket
            for bucket in manifest.buckets
            if bucket.split == DatasetSplit.VALIDATION
            and minimum_module_count <= bucket.module_count <= maximum_module_count
        ),
        key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
    )
    if not validation:
        raise SchemaValidationError(
            "Order9 topology-stratified runner requires validation buckets"
        )
    return tuple(selected), validation[update_index % len(validation)]


def select_order9_c3_state_inheritance_buckets(
    manifest: Order9PiLRolloutBucketManifest,
    update_index: int,
    *,
    minimum_module_count: int,
    maximum_module_count: int,
    topology_count_by_module: Mapping[int, int],
) -> tuple[Order9PiLRolloutBucket, ...]:
    """Select independently rotating continuous topologies per module count."""

    if update_index < 0 or minimum_module_count < 1 or (
        maximum_module_count < minimum_module_count
    ):
        raise ValueError("Order9 state-inheritance topology range is invalid")
    expected_modules = tuple(range(minimum_module_count, maximum_module_count + 1))
    if (
        set(int(value) for value in topology_count_by_module)
        != set(expected_modules)
        or any(int(value) < 0 for value in topology_count_by_module.values())
        or not any(int(value) > 0 for value in topology_count_by_module.values())
    ):
        raise ValueError("Order9 state-inheritance topology counts are invalid")
    selected: list[Order9PiLRolloutBucket] = []
    for module_count in expected_modules:
        candidates = sorted(
            (
                bucket
                for bucket in manifest.buckets
                if bucket.split == DatasetSplit.TRAIN
                and bucket.module_count == module_count
            ),
            key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
        )
        requested = int(topology_count_by_module[module_count])
        if requested == 0:
            continue
        if len(candidates) < requested:
            raise SchemaValidationError(
                "Order9 state-inheritance runner lacks distinct train buckets "
                f"for module_count={module_count}: {len(candidates)} < {requested}"
            )
        start = (update_index * requested) % len(candidates)
        selected.extend(
            candidates[(start + offset) % len(candidates)]
            for offset in range(requested)
        )
    return tuple(selected)


def allocate_order9_topology_stratified_environments(
    total_environment_count: int,
    module_counts: Sequence[int],
    *,
    update_index: int,
) -> dict[int, int]:
    """Balance an immutable per-update environment budget over topology strata."""

    ordered = tuple(sorted(set(int(value) for value in module_counts)))
    if (
        total_environment_count < len(ordered)
        or not ordered
        or len(ordered) != len(module_counts)
        or update_index < 0
    ):
        raise ValueError("Order9 topology-stratified environment budget is invalid")
    base, remainder = divmod(total_environment_count, len(ordered))
    allocation = {module_count: base for module_count in ordered}
    for offset in range(remainder):
        module_count = ordered[(update_index + offset) % len(ordered)]
        allocation[module_count] += 1
    if sum(allocation.values()) != total_environment_count:
        raise AssertionError("Order9 topology allocation lost environments")
    return allocation


def allocate_order9_topology_shard_environments(
    total_environment_count: int,
    topology_shard_count: int,
    *,
    update_index: int,
) -> tuple[int, ...]:
    """Balance one immutable environment budget over replicated topologies."""

    if (
        topology_shard_count < 1
        or total_environment_count < topology_shard_count
        or update_index < 0
    ):
        raise ValueError("Order9 topology-shard environment budget is invalid")
    base, remainder = divmod(total_environment_count, topology_shard_count)
    allocation = [base] * topology_shard_count
    for offset in range(remainder):
        allocation[(update_index + offset) % topology_shard_count] += 1
    if sum(allocation) != total_environment_count:
        raise AssertionError("Order9 topology-shard allocation lost environments")
    return tuple(allocation)


def order9_c3_state_inheritance_environment_steps(
    config: Order9LearningConfig,
    *,
    minimum_module_count: int,
    maximum_module_count: int,
) -> int:
    """Return the extra fresh train steps contributed by inherited-state shards."""

    runtime = config.production_runtime
    if not runtime.c3_state_inheritance_rollouts_enabled:
        return 0
    if minimum_module_count < 1 or maximum_module_count < minimum_module_count:
        raise ValueError("Order9 C3 state-inheritance module range is invalid")
    topology_shard_count = sum(
        runtime.c3_state_inheritance_topology_count(module_count)
        for module_count in range(minimum_module_count, maximum_module_count + 1)
    )
    return (
        topology_shard_count
        * runtime.c3_state_inheritance_environment_count_per_module
        * runtime.c3_state_inheritance_rollout_steps
    )


def order9_c3_state_inheritance_phase_indices(
    environment_ids: Sequence[int],
    episode_serials: Sequence[int],
    initial_phase_indices: Sequence[int],
) -> tuple[int, ...]:
    """Deterministically balance continuous episodes over approved seed phases."""

    ids = tuple(int(value) for value in environment_ids)
    serials = tuple(int(value) for value in episode_serials)
    phases = tuple(int(value) for value in initial_phase_indices)
    if (
        len(ids) != len(serials)
        or not phases
        or len(set(phases)) != len(phases)
        or any(value < 0 or value >= len(ORDER9_OBJECT_TASK_PHASES) for value in phases)
        or any(value < 0 for value in ids)
        or any(value < 0 for value in serials)
    ):
        raise ValueError("Order9 C3 state-inheritance phase selection is invalid")
    return tuple(
        phases[(environment_id + episode_serial) % len(phases)]
        for environment_id, episode_serial in zip(ids, serials)
    )


def resolve_order9_pi_l_stage_plan(
    config: Order9LearningConfig,
    *,
    stage_id: str,
    stage_root: str | Path,
    initial_checkpoint_path: str | Path,
    repository_root: str | Path,
    additional_update_count: int = 0,
    branch_parent_update_index: int | None = None,
    expected_physical_model_hash: str | None = None,
    training_minimum_module_count: int | None = None,
    training_maximum_module_count: int | None = None,
) -> Order9PiLStagePlan:
    config.validate()
    stage = order9_stage_by_id(config, stage_id)
    require_order9_stage_execution_allowed(config, stage)
    if (
        stage.learning_mode != Order9LearningMode.PPO
        or stage.learning_target != Order9LearningTarget.PI_L
    ):
        raise SchemaValidationError("Order9 pi_L stage runner requires a pi_L PPO stage")
    runtime = resolve_order9_stage_runtime(config, stage)
    if (training_minimum_module_count is None) != (
        training_maximum_module_count is None
    ):
        raise ValueError("Order9 training module range must provide both bounds")
    training_minimum = (
        stage.min_modules
        if training_minimum_module_count is None
        else int(training_minimum_module_count)
    )
    training_maximum = (
        stage.max_modules
        if training_maximum_module_count is None
        else int(training_maximum_module_count)
    )
    if (
        training_minimum < stage.min_modules
        or training_maximum > stage.max_modules
        or training_maximum < training_minimum
    ):
        raise ValueError("Order9 training module range lies outside the stage")
    if (
        (training_minimum, training_maximum)
        != (stage.min_modules, stage.max_modules)
        and stage.stage_id != ORDER9_C3_STAGE_ID
    ):
        raise ValueError("Order9 module subcurriculum is restricted to C3")
    split_generation_steps = runtime.generation_environment_steps
    if split_generation_steps is None:
        raise SchemaValidationError("Order9 pi_L PPO generation size is missing")
    # The tensor runtime size is per split.  One immutable production
    # generation deliberately contains one train and one validation shard, and
    # the existing dataset/checkpoint lineage counts both shards as consumed
    # environment steps.
    generation_steps = 2 * split_generation_steps
    if stage.stage_id == ORDER9_C3_STAGE_ID:
        generation_steps += order9_c3_state_inheritance_environment_steps(
            config,
            minimum_module_count=training_minimum,
            maximum_module_count=training_maximum,
        )
    (
        base_target_updates,
        target_updates,
        target_environment_steps,
    ) = resolve_order9_extended_update_budget(
        stage.environment_steps,
        generation_steps,
        additional_update_count,
    )
    repository = Path(repository_root).resolve()
    root = _resolve(stage_root, repository)
    initial = _resolve(initial_checkpoint_path, repository)
    if not initial.is_file():
        raise FileNotFoundError(initial)
    initial_loaded = load_order9_stage_parent_checkpoint(
        config,
        stage,
        initial,
        expected_family=Order9PolicyFamily.PI_L,
        update_index=0,
    )
    if (
        expected_physical_model_hash is not None
        and initial_loaded.metadata.physical_model_hash
        != expected_physical_model_hash
    ):
        raise SchemaValidationError(
            "Order9 pi_L stage initializer physical-model hash mismatch"
        )
    if branch_parent_update_index is not None:
        if branch_parent_update_index < 0:
            raise ValueError("Order9 branch parent update index must be non-negative")
        if branch_parent_update_index >= target_updates:
            raise SchemaValidationError(
                "Order9 branch parent is outside the requested update budget"
            )
        metadata = initial_loaded.metadata
        observed_update = metadata.metadata.get("ppo_update_index")
        if (
            metadata.curriculum_stage_id != stage.stage_id
            or metadata.curriculum_stage_index != stage.stage_index
            or observed_update != branch_parent_update_index
        ):
            raise SchemaValidationError(
                "Order9 branch initializer identity/update index mismatch"
            )
    expected_parent_sha = initial_loaded.sha256
    parent_path = initial
    first_update_index = (
        0
        if branch_parent_update_index is None
        else branch_parent_update_index + 1
    )
    completed_steps = first_update_index * generation_steps
    next_index = first_update_index
    for update_index in range(first_update_index, target_updates):
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
        configured_target_environment_steps=stage.environment_steps,
        target_environment_steps=target_environment_steps,
        generation_environment_steps=generation_steps,
        base_target_update_count=base_target_updates,
        additional_update_count=additional_update_count,
        target_update_count=target_updates,
        next_update_index=next_index,
        completed_environment_steps=completed_steps,
        parent_checkpoint_path=str(parent_path),
        parent_checkpoint_sha256=expected_parent_sha,
        branch_parent_update_index=branch_parent_update_index,
        training_minimum_module_count=training_minimum,
        training_maximum_module_count=training_maximum,
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
    if stage.stage_id == ORDER9_C3_STAGE_ID:
        if (
            manifest.metadata.get("c3_articulated_teacher_prechecked") is not True
            or manifest.metadata.get("c3_bucket_precheck_version")
            != ORDER9_C3_BUCKET_PRECHECK_VERSION
        ):
            raise SchemaValidationError(
                "Order9 C3 rollout buckets lack the current articulated-teacher "
                "precheck"
            )
        nominal_path_raw = manifest.metadata.get(
            "c3_nominal_set_manifest_path"
        )
        nominal_sha256 = manifest.metadata.get(
            "c3_nominal_set_manifest_sha256"
        )
        review_path_raw = manifest.metadata.get(
            "c3_human_review_manifest_path"
        )
        review_sha256 = manifest.metadata.get(
            "c3_human_review_manifest_sha256"
        )
        if (
            manifest.metadata.get("c3_accepted_nominal_trajectory_bound")
            is not True
            or manifest.metadata.get(
                "c3_configuration_space_planner_runtime_enabled"
            )
            is not False
            or not all(
                isinstance(value, str) and value
                for value in (
                    nominal_path_raw,
                    nominal_sha256,
                    review_path_raw,
                    review_sha256,
                )
            )
        ):
            raise SchemaValidationError(
                "Order9 C3 rollout buckets lack accepted nominal replay binding"
            )
        nominal_path = _resolve(str(nominal_path_raw), repository)
        nominal = validate_order9_c3_nominal_trajectory_set_bytes(
            nominal_path,
            repository_root=repository,
            expected_sha256=str(nominal_sha256),
        )
        review_path = _resolve(str(review_path_raw), repository)
        if hash_file(review_path) != review_sha256:
            raise SchemaValidationError("Order9 C3 human review bytes changed")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review_entries = review.get("entries")
        if (
            review.get("nominal_set_manifest_sha256") != nominal_sha256
            or review.get("accepted_count") != len(manifest.buckets)
            or review.get("rejected_count") != 0
            or not isinstance(review_entries, list)
            or {entry.get("bucket_id") for entry in review_entries}
            != {bucket.bucket_id for bucket in manifest.buckets}
            or any(entry.get("decision") != "accepted" for entry in review_entries)
        ):
            raise SchemaValidationError(
                "Order9 C3 human review does not accept the complete nominal set"
            )
        nominal_by_bucket = {entry.bucket_id: entry for entry in nominal.entries}
        for bucket in manifest.buckets:
            topology_metadata = bucket.metadata.get("topology_provider")
            evidence = (
                topology_metadata.get("articulated_teacher_precheck")
                if isinstance(topology_metadata, dict)
                else None
            )
            if not isinstance(evidence, dict) or not evidence.get(
                "trajectory_hash"
            ):
                raise SchemaValidationError(
                    "Order9 C3 rollout bucket lacks articulated-teacher evidence"
                )
            accepted = bucket.metadata.get("accepted_nominal_trajectory")
            nominal_entry = nominal_by_bucket.get(bucket.bucket_id)
            if (
                not isinstance(accepted, dict)
                or nominal_entry is None
                or accepted.get("set_manifest_path") != nominal_path_raw
                or accepted.get("set_manifest_sha256") != nominal_sha256
                or accepted.get("artifact_sha256")
                != nominal_entry.artifact_sha256
                or accepted.get("animation_scene_sha256")
                != nominal_entry.animation_scene_sha256
                or accepted.get(
                    "configuration_space_planner_runtime_enabled"
                )
                is not False
                or float(accepted.get("nominal_reference_rate_hz", 0.0))
                != 10.0
            ):
                raise SchemaValidationError(
                    f"Order9 C3 nominal replay binding differs: {bucket.bucket_id}"
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
    c3_reset_bank_path: str | Path | None = None,
    generation_id: str,
    output_raw_path: str | Path,
    bucket: Order9PiLRolloutBucket,
    bucket_manifest_path: str | Path,
    tensorboard_log_dir: str | Path | None = None,
    topology_shard_index: int | None = None,
    topology_shard_count: int | None = None,
    topology_shard_environment_count: int | None = None,
    c3_wrench_gate_range_scale: float | None = None,
    c3_state_inheritance: bool = False,
    c3_action_contract: str | None = None,
    c3_training_nominal_preload_deficit_mm: str | None = None,
    c3_training_nominal_preload_deficit_module_count: int | None = None,
) -> list[str]:
    repository = Path(repository_root).resolve()
    command = [
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
    if c3_reset_bank_path is not None:
        command.extend(
            [
                "--c3-reset-bank",
                str(_resolve(c3_reset_bank_path, repository)),
            ]
        )
    if c3_wrench_gate_range_scale is not None:
        if not math.isfinite(c3_wrench_gate_range_scale) or c3_wrench_gate_range_scale < 1.0:
            raise ValueError("C3 wrench-gate range scale must be finite and at least one")
        command.extend(
            [
                "--c3-wrench-gate-range-scale",
                str(float(c3_wrench_gate_range_scale)),
            ]
        )
    topology_values = (
        topology_shard_index,
        topology_shard_count,
        topology_shard_environment_count,
    )
    if any(value is not None for value in topology_values):
        if any(value is None for value in topology_values):
            raise ValueError("topology shard arguments must be provided together")
        assert topology_shard_index is not None
        assert topology_shard_count is not None
        assert topology_shard_environment_count is not None
        if (
            topology_shard_count < 2
            or not 0 <= topology_shard_index < topology_shard_count
            or topology_shard_environment_count < 1
        ):
            raise ValueError("topology shard arguments are invalid")
        command.extend(
            [
                "--production-topology-shard-index",
                str(topology_shard_index),
                "--production-topology-shard-count",
                str(topology_shard_count),
                "--production-topology-shard-environment-count",
                str(topology_shard_environment_count),
            ]
        )
    if tensorboard_log_dir is not None:
        command.extend(
            [
                "--tensorboard-log-dir",
                str(_resolve(tensorboard_log_dir, repository)),
            ]
        )
    if c3_state_inheritance:
        if stage_id != ORDER9_C3_STAGE_ID or not all(
            value is not None for value in topology_values
        ):
            raise ValueError(
                "state-inheritance collection requires a C3 production topology shard"
            )
        command.append("--production-state-inheritance-shard")
    if c3_action_contract is not None:
        command.extend(["--c3-action-contract", c3_action_contract])
    deficit_values = (
        c3_training_nominal_preload_deficit_mm,
        c3_training_nominal_preload_deficit_module_count,
    )
    if any(value is not None for value in deficit_values):
        if any(value is None for value in deficit_values):
            raise ValueError(
                "training nominal-preload deficit arguments must be provided together"
            )
        if stage_id != ORDER9_C3_STAGE_ID or bucket.split != DatasetSplit.TRAIN:
            raise ValueError(
                "training nominal-preload deficit is restricted to C3 train buckets"
            )
        assert c3_training_nominal_preload_deficit_mm is not None
        assert c3_training_nominal_preload_deficit_module_count is not None
        command.extend(
            [
                "--training-nominal-preload-deficit-mm",
                c3_training_nominal_preload_deficit_mm,
                "--training-nominal-preload-deficit-module-count",
                str(c3_training_nominal_preload_deficit_module_count),
            ]
        )
    return command


def order9_c3_training_wrench_gate_range_scale(
    config: Order9LearningConfig, update_index: int
) -> float:
    """Resolve the exact-reward / relaxed-admission curriculum for one update."""

    if update_index < 0:
        raise ValueError("C3 wrench-gate curriculum update index must be non-negative")
    runtime = config.production_runtime
    offset = update_index - runtime.c3_wrench_gate_curriculum_start_update_index
    if offset < 0:
        return 1.0
    scales = runtime.c3_wrench_gate_curriculum_scales
    return float(scales[min(offset, len(scales) - 1)])


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
    maximum_parallel_process_count: int | None = None,
    process_start_stagger_s: float = 0.0,
) -> tuple[float, dict[str, int]]:
    if not commands or set(commands) != set(log_paths):
        raise ValueError("Order9 collector commands/log paths must be non-empty and aligned")
    repository = Path(repository_root).resolve()
    if maximum_parallel_process_count is None:
        maximum_parallel_process_count = len(commands)
    if maximum_parallel_process_count < 1:
        raise ValueError("Order9 collector parallel process count must be positive")
    if not math.isfinite(process_start_stagger_s) or process_start_stagger_s < 0.0:
        raise ValueError("Order9 collector process-start stagger is invalid")
    handles: dict[str, TextIO] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    return_codes: dict[str, int] = {}
    started = time.perf_counter()
    try:
        pending = list(commands.items())
        while pending or processes:
            while pending and len(processes) < maximum_parallel_process_count:
                if processes and process_start_stagger_s > 0.0:
                    time.sleep(process_start_stagger_s)
                split, command = pending.pop(0)
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
            completed_name = next(iter(processes))
            process = processes.pop(completed_name)
            return_codes[completed_name] = process.wait()
            handles.pop(completed_name).close()
    finally:
        for handle in handles.values():
            handle.close()
    elapsed = time.perf_counter() - started
    failures = {split: code for split, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"Order9 parallel collectors failed: {failures}")
    return elapsed, return_codes


def coalesce_order9_same_morphology_collectors(
    commands: Mapping[str, Sequence[str]],
    *,
    log_paths: Mapping[str, str | Path],
    morphology_hash_by_name: Mapping[str, str],
    batch_root: str | Path,
) -> tuple[dict[str, list[str]], dict[str, Path], dict[str, tuple[str, ...]]]:
    """Reuse one Kit process for pending buckets with identical morphology.

    Bucket simulations remain sequential and independently hash-bound: each
    call creates and clears its own SimulationContext and writes the original
    raw artifact, episode JSONL, and log.  Only the expensive AppLauncher/Kit
    process lifetime is shared.  Distinct morphologies remain independent
    processes so no heterogeneous robot asset is silently mixed into a scene.
    """

    names = tuple(commands)
    if (
        not names
        or set(names) != set(log_paths)
        or set(names) != set(morphology_hash_by_name)
    ):
        raise ValueError(
            "Order9 persistent collector inputs must be non-empty and aligned"
        )
    groups: dict[str, list[str]] = {}
    for name in names:
        morphology_hash = morphology_hash_by_name[name]
        if not isinstance(morphology_hash, str) or not morphology_hash:
            raise ValueError("Order9 collector morphology hash is invalid")
        groups.setdefault(morphology_hash, []).append(name)

    root = Path(batch_root).resolve()
    output_commands: dict[str, list[str]] = {}
    output_logs: dict[str, Path] = {}
    members: dict[str, tuple[str, ...]] = {}
    for morphology_hash, grouped_names in groups.items():
        if len(grouped_names) == 1:
            name = grouped_names[0]
            output_commands[name] = list(commands[name])
            output_logs[name] = Path(log_paths[name]).resolve()
            members[name] = (name,)
            continue

        batch_name = f"morphology_{morphology_hash[:16]}"
        if batch_name in output_commands:
            raise ValueError("Order9 persistent collector batch names collide")
        first = list(commands[grouped_names[0]])
        script_index = next(
            (
                index
                for index, value in enumerate(first)
                if Path(value).name == "order9_vectorized_isaac_rollout.py"
            ),
            None,
        )
        if script_index is None:
            raise ValueError("Order9 collector command lacks its rollout script")
        jobs: list[dict[str, object]] = []
        for name in grouped_names:
            command = list(commands[name])
            candidate_index = next(
                (
                    index
                    for index, value in enumerate(command)
                    if Path(value).name == "order9_vectorized_isaac_rollout.py"
                ),
                None,
            )
            if (
                candidate_index != script_index
                or command[: script_index + 1] != first[: script_index + 1]
            ):
                raise ValueError(
                    "same-morphology collectors must share one executable prefix"
                )
            jobs.append(
                {
                    "name": name,
                    "argv": command[script_index + 1 :],
                    "log_path": str(Path(log_paths[name]).resolve()),
                }
            )
        batch_dir = root / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = batch_dir / "persistent_bucket_jobs.json"
        payload = {
            "version": ORDER9_PERSISTENT_BUCKET_JOBS_VERSION,
            "morphology_hash": morphology_hash,
            "jobs": jobs,
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        batch_command = [
            *first,
            "--persistent-bucket-jobs",
            str(manifest_path),
        ]
        output_commands[batch_name] = batch_command
        output_logs[batch_name] = batch_dir / "persistent_process.log"
        members[batch_name] = tuple(grouped_names)
    return output_commands, output_logs, members


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
    c3_wrench_gate_range_scale: float | None = None,
    c3_state_inheritance: bool = False,
    c3_state_inheritance_initial_phase_indices: Sequence[int] | None = None,
    c3_state_inheritance_fixed_reset_progress_fraction: float | None = None,
    c3_action_contract: str | None = None,
    c3_training_nominal_preload_deficit_mm: str | None = None,
    c3_training_nominal_preload_deficit_module_count: int | None = None,
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
    if stage_id == ORDER9_C3_STAGE_ID:
        if metadata.get("c3_action_contract") != c3_action_contract:
            raise SchemaValidationError("Order9 C3 rollout action contract differs")
        expected_deficits = (
            None
            if c3_training_nominal_preload_deficit_mm is None
            else [
                float(value.strip())
                for value in c3_training_nominal_preload_deficit_mm.split(",")
            ]
        )
        if (
            metadata.get("training_nominal_preload_deficit_mm")
            != expected_deficits
            or metadata.get("training_nominal_preload_deficit_module_count")
            != c3_training_nominal_preload_deficit_module_count
        ):
            raise SchemaValidationError(
                "Order9 C3 training nominal-preload deficit contract differs"
            )
        expected_rollout_mode = (
            ORDER9_C3_STATE_INHERITANCE_ROLLOUT_MODE
            if c3_state_inheritance
            else ORDER9_C3_PHASE_RESET_ROLLOUT_MODE
        )
        expected_scale = float(c3_wrench_gate_range_scale or 1.0)
        if (
            float(metadata.get("wrench_range_gate_scale", math.nan))
            != expected_scale
            or float(payload.get("wrench_range_gate_scale", math.nan))
            != expected_scale
            or metadata.get("wrench_range_hard_gate_enabled") is not False
            or payload.get("wrench_range_hard_gate_enabled") is not False
            or metadata.get("exact_wrench_reward_bounds_preserved") is not True
            or metadata.get(
                "c3_train_rollout_mode", ORDER9_C3_PHASE_RESET_ROLLOUT_MODE
            )
            != expected_rollout_mode
            or payload.get(
                "c3_train_rollout_mode", ORDER9_C3_PHASE_RESET_ROLLOUT_MODE
            )
            != expected_rollout_mode
        ):
            raise SchemaValidationError(
                "Order9 C3 rollout wrench diagnostic/outcome-gate contract differs"
            )
    if stage_id == ORDER9_C3_STAGE_ID:
        expected_selectable_strata = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2],
        ]
        transition_counts = metadata.get("phase_transition_counts")
        result_transition_counts = payload.get("phase_transition_counts")
        target_boundary_keys = (
            "place->release",
            "release->retreat",
            "retreat->settle",
        )
        valid_transition_counts = (
            isinstance(transition_counts, dict)
            and all(
                isinstance(transition_counts.get(key), int)
                and not isinstance(transition_counts.get(key), bool)
                and int(transition_counts[key]) >= 0
                for key in target_boundary_keys
            )
        )
        if (
            artifact.artifact_version
            not in {
                ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
                ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
                ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
            }
            or metadata.get("canonical_phase_resets") is not False
            or metadata.get("phase_specific_resets_available") is not True
            or metadata.get("phase_reset_reference_version")
            != ORDER9_C3_PHASE_RESET_REFERENCE_VERSION
            or int(metadata.get("phase_reset_stratum_count", 0)) != 4
            or metadata.get("c3_boundary_tail_sampling_contract")
            != ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            or metadata.get("c3_boundary_tail_phase_labels")
            != ["place", "release", "retreat"]
            or metadata.get("c3_boundary_tail_progress_fraction") != 0.9
            or metadata.get("phase_reset_selectable_strata_by_phase")
            != expected_selectable_strata
            or not isinstance(
                metadata.get("c3_contact_reset_stabilization"), dict
            )
            or metadata["c3_contact_reset_stabilization"].get("passed")
            is not True
            or metadata["c3_contact_reset_stabilization"].get("reset_source")
            != "hash_bound_human_accepted_nominal_trajectory"
            or metadata["c3_contact_reset_stabilization"].get(
                "dynamic_stability_gate_required"
            )
            is not False
            or metadata.get("c3_reset_stabilization_checkpoint_sha256")
            is not None
            or not _is_sha256(metadata.get("c3_reset_bank_sha256"))
            or metadata["c3_contact_reset_stabilization"].get(
                "reset_bank_sha256"
            )
            != metadata.get("c3_reset_bank_sha256")
            or payload.get("phase_specific_resets_available") is not True
            or int(payload.get("phase_reset_stratum_count", 0)) != 4
            or payload.get("c3_boundary_tail_sampling_contract")
            != ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            or payload.get("c3_boundary_tail_phase_labels")
            != ["place", "release", "retreat"]
            or payload.get("c3_boundary_tail_progress_fraction") != 0.9
            or payload.get("phase_reset_selectable_strata_by_phase")
            != expected_selectable_strata
            or not isinstance(transition_counts, dict)
            or not isinstance(result_transition_counts, dict)
            or not valid_transition_counts
            or result_transition_counts != transition_counts
            or not isinstance(
                payload.get("c3_contact_reset_stabilization"), dict
            )
            or payload["c3_contact_reset_stabilization"].get("passed")
            is not True
            or payload.get("unlocked_phase_indices")
            != list(range(len(ORDER9_OBJECT_TASK_PHASES)))
        ):
            raise SchemaValidationError(
                "Order9 C3 rollout lacks morphology-specific phase resets"
            )
        observed = set(int(value) for value in artifact.tensors["phase_index"].unique())
        required = set(ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME)
        if c3_state_inheritance:
            expected_initial = [
                int(value)
                for value in (
                    c3_state_inheritance_initial_phase_indices
                    if c3_state_inheritance_initial_phase_indices is not None
                    else (1, 3)
                )
            ]
            observed_initial = metadata.get(
                "state_inheritance_initial_phase_indices", []
            )
            fixed_progress = (
                None
                if c3_state_inheritance_fixed_reset_progress_fraction is None
                else float(c3_state_inheritance_fixed_reset_progress_fraction)
            )
            expected_stratum_policy = (
                "fixed_progress_fraction_on_every_episode"
                if fixed_progress is not None
                else "latest_selectable_then_reverse_cycle_on_terminal"
            )
            if (
                metadata.get("state_inheritance_preserves_phase_transitions")
                is not True
                or metadata.get("state_inheritance_initial_stratum_policy")
                != expected_stratum_policy
                or metadata.get(
                    "state_inheritance_recurrent_state_reset_only_at_episode_boundary"
                )
                is not True
                or observed_initial != expected_initial
                or metadata.get(
                    "state_inheritance_fixed_reset_progress_fraction"
                )
                != fixed_progress
                or metadata.get(
                    "state_inheritance_transition_backward_curriculum_contract"
                )
                != (
                    ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT
                    if fixed_progress is not None
                    else None
                )
                or payload.get("state_inheritance_initial_phase_indices")
                != expected_initial
                or payload.get("state_inheritance_initial_stratum_policy")
                != expected_stratum_policy
                or payload.get(
                    "state_inheritance_fixed_reset_progress_fraction"
                )
                != fixed_progress
                or payload.get(
                    "state_inheritance_transition_backward_curriculum_contract"
                )
                != (
                    ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT
                    if fixed_progress is not None
                    else None
                )
                or not observed.intersection(
                    {
                        ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME[index]
                        for index in expected_initial
                    }
                )
            ):
                raise SchemaValidationError(
                    "Order9 C3 rollout lacks continuous state-inheritance evidence"
                )
        elif observed != required:
            raise SchemaValidationError(
                "Order9 C3 rollout does not cover every runtime phase"
            )
    return digest


def validate_order9_c3_generation_boundary_coverage(
    rollout_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Report real successor exposure across the aggregate train generation.

    A morphology that fails before crossing the boundary is useful on-policy
    terminal evidence and must not be rejected before PPO can consume it.  The
    phase-reset strata already guarantee direct release/retreat tail samples;
    requiring an untrained initializer to *succeed* at either transition would
    discard precisely the failures PPO needs to improve.
    """

    train = [
        payload
        for name, payload in rollout_summaries.items()
        if str(name).startswith("train_")
    ]
    if not train:
        raise SchemaValidationError("Order9 C3 generation has no train shards")
    totals = {
        "place->release": 0,
        "release->retreat": 0,
        "retreat->settle": 0,
    }
    for payload in train:
        counts = payload.get("phase_transition_counts")
        if not isinstance(counts, dict):
            raise SchemaValidationError(
                "Order9 C3 train shard lacks phase-transition counts"
            )
        for key in totals:
            value = counts.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SchemaValidationError(
                    f"Order9 C3 train shard has invalid transition count: {key}"
                )
            totals[key] += value
    return totals


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


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
    c3_boundary_fine_tune: Order9C3BoundaryFineTuneConfig | None = None,
    c3_action_contract: str | None = None,
    c3_state_inheritance_required: bool = False,
    expected_state_inheritance_shard_count: int = 0,
    expected_topology_module_counts: Sequence[int] = tuple(range(2, 9)),
    expected_topology_shard_count: int | None = None,
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
    if c3_action_contract is not None:
        if c3_action_contract not in ORDER9_C3_ACTION_CONTRACTS:
            raise SchemaValidationError(
                "Order9 C3 completed-update action contract is invalid"
            )
        if replay.get("c3_action_contract") != c3_action_contract:
            raise SchemaValidationError(
                "Order9 C3 completed update action contract mismatch"
            )
    if replay.get("exact_behavior_replay_validated") is not True:
        raise SchemaValidationError("Order9 PPO exact behavior replay was not validated")
    if stage_id == ORDER9_C3_STAGE_ID and any(
        replay.get(name) is not True
        for name in (
            "phase_balanced_sampling",
            "phase_normalized_advantages",
            "phase_local_kl",
            "critic_actor_state_gradient_detached",
        )
    ):
        raise SchemaValidationError(
            "Order9 C3 PPO lacks the phase-balanced detached-critic contract"
        )
    expected_modules = sorted({int(value) for value in expected_topology_module_counts})
    if not expected_modules:
        raise ValueError("Order9 expected topology module counts are empty")
    observed_topology_shard_count = int(replay.get("topology_shard_count", 0))
    if stage_id == ORDER9_C3_STAGE_ID and (
        replay.get("topology_stratified_update") is not True
        or (
            expected_topology_shard_count is None
            and observed_topology_shard_count < len(expected_modules)
        )
        or (
            expected_topology_shard_count is not None
            and observed_topology_shard_count != expected_topology_shard_count
        )
        or sorted(
            {int(value) for value in replay.get("topology_module_counts", [])}
        )
        != expected_modules
    ):
        raise SchemaValidationError(
            "Order9 C3 PPO lacks complete topology-stratified coverage"
        )
    if c3_state_inheritance_required and (
        expected_state_inheritance_shard_count < 1
        or replay.get("topology_minibatch_packing")
        != ORDER9_TOPOLOGY_MINIBATCH_PACKING
        or int(replay.get("transition_budget_per_topology_minibatch", 0)) < 1
        or replay.get("state_inheritance_gae_cross_phase_preserved") is not True
        or replay.get("state_inheritance_recurrent_inputs_preserved") is not True
        or replay.get(
            "state_inheritance_target_only_minibatches_anchored_by_phase_reset_shards"
        )
        is not True
        or int(replay.get("state_inheritance_shard_count", 0))
        != expected_state_inheritance_shard_count
        or int(replay.get("state_inheritance_environment_step_count", 0)) < 1
    ):
        raise SchemaValidationError(
            "Order9 C3 PPO lacks mixed state-inheritance replay"
        )
    if stage_id == ORDER9_C3_STAGE_ID and c3_boundary_fine_tune is not None:
        target_labels = [
            str(label)
            for label in c3_boundary_fine_tune.target_actor_phase_labels
        ]
        missing = [
            label
            for label in target_labels
            if label not in ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS
        ]
        if missing:
            raise SchemaValidationError(
                "Order9 C3 fine-tune validation has unknown actor phases: "
                + ",".join(missing)
            )
        target_indices = [
            ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index(label)
            for label in target_labels
        ]
        expected_contact_residual_only = bool(
            c3_boundary_fine_tune.contact_residual_only_actor_update
        )
        expected_compression_only = bool(
            c3_boundary_fine_tune.compression_only_actor_objective
        )
        expected_joint_head_only = bool(
            c3_boundary_fine_tune.joint_head_only_actor_update
        )
        if c3_action_contract is not None:
            # An explicit action contract supersedes the legacy actor-head
            # update switch.  Compression-only trains the scalar compression
            # decoder; contracts with a global prefix also train that prefix.
            if order9_c3_action_contract_uses_full_policy(c3_action_contract):
                expected_compression_only = False
                expected_joint_head_only = False
                expected_contact_residual_only = False
            else:
                expected_contact_residual_only = (
                    order9_c3_action_contract_global_dimension(
                        c3_action_contract
                    )
                    == 0
                )
        expected_boundary = {
            "c3_boundary_fine_tune_contract": (
                ORDER9_C3_BOUNDARY_FINE_TUNE_CONTRACT
            ),
            "c3_boundary_fine_tune_parent_checkpoint_sha256": (
                parent_checkpoint_sha256
            ),
            "c3_boundary_target_actor_phase_labels": target_labels,
            "c3_boundary_target_actor_phase_indices": target_indices,
            "c3_boundary_target_only_actor_objective": True,
            "compression_only_actor_objective": expected_compression_only,
            "joint_head_only_actor_update": expected_joint_head_only,
            "contact_residual_only_actor_update": (
                expected_contact_residual_only
            ),
            "factorized_actor_credit_enabled": bool(
                c3_boundary_fine_tune.factorized_actor_credit_enabled
            ),
            "contact_coordinate_credit_enabled": bool(
                c3_boundary_fine_tune.contact_coordinate_credit_enabled
            ),
            "c3_contact_coordinate_credit_enabled": bool(
                c3_boundary_fine_tune.contact_coordinate_credit_enabled
            ),
            "contact_head_extra_optimizer_contract": (
                ORDER9_CONTACT_HEAD_EXTRA_OPTIMIZER_CONTRACT
                if c3_boundary_fine_tune.contact_head_extra_optimizer_passes
                > 0
                else None
            ),
            "contact_head_extra_optimizer_passes_requested": int(
                c3_boundary_fine_tune.contact_head_extra_optimizer_passes
            ),
            "contact_head_extra_optimizer_parameter_prefixes": (
                list(ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES)
                if c3_boundary_fine_tune.contact_head_extra_optimizer_passes
                > 0
                else []
            ),
            "c3_boundary_parent_anchor_source": (
                "sha_bound_exact_behavior_log_probability"
            ),
            "c3_boundary_learning_rate_scale": (
                c3_boundary_fine_tune.learning_rate_scale
            ),
            "c3_boundary_optimizer_step_rollback": True,
        }
        for name, value in expected_boundary.items():
            if replay.get(name) != value:
                raise SchemaValidationError(
                    f"Order9 C3 boundary fine-tune differs at {name}"
                )
        if c3_boundary_fine_tune.factorized_actor_credit_enabled and (
            replay.get("factorized_actor_credit_version")
            != ORDER9_FACTORIZED_ACTOR_CREDIT_VERSION
            or replay.get("factorized_actor_credit_channels")
            != list(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
        ):
            raise SchemaValidationError(
                "Order9 C3 factorized actor-credit provenance differs"
            )
        if c3_boundary_fine_tune.contact_coordinate_credit_enabled and (
            replay.get("contact_coordinate_credit_version")
            != ORDER9_CONTACT_COORDINATE_CREDIT_VERSION
            or replay.get("contact_coordinate_credit_channels")
            != list(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-coordinate credit provenance differs"
            )
        requested_contact_passes = int(
            c3_boundary_fine_tune.contact_head_extra_optimizer_passes
        )
        completed_contact_passes = int(
            replay.get(
                "contact_head_extra_optimizer_passes_completed", -1
            )
        )
        contact_step_count = int(
            replay.get("contact_head_extra_optimizer_step_count", -1)
        )
        if requested_contact_passes > 0 and (
            completed_contact_passes != requested_contact_passes
            or contact_step_count < 1
        ):
            raise SchemaValidationError(
                "Order9 C3 contact-head extra optimizer pass is incomplete"
            )
        non_target_limit = float(
            replay.get("c3_boundary_non_target_parent_kl_limit", math.nan)
        )
        topology_limit = float(
            replay.get(
                "c3_boundary_maximum_topology_phase_kl_limit", math.nan
            )
        )
        applied_non_target = float(
            replay.get(
                "c3_boundary_maximum_applied_non_target_topology_phase_kl",
                math.inf,
            )
        )
        applied_topology = float(
            replay.get(
                "c3_boundary_maximum_applied_topology_phase_kl", math.inf
            )
        )
        if (
            result.ppo_update.requested_epoch_count != 1
            or not math.isfinite(non_target_limit)
            or not math.isfinite(topology_limit)
            or non_target_limit
            != c3_boundary_fine_tune.non_target_parent_kl_limit
            or topology_limit
            != c3_boundary_fine_tune.maximum_topology_phase_kl
            or topology_limit <= non_target_limit
            or not math.isfinite(applied_non_target)
            or not math.isfinite(applied_topology)
            or applied_non_target > non_target_limit + 1.0e-9
            or applied_topology > topology_limit + 1.0e-9
        ):
            raise SchemaValidationError(
                "Order9 C3 boundary fine-tune exceeded an applied KL authority"
            )
    tolerance = float(replay.get("exact_replay_absolute_tolerance", 0.0))
    replay_fields = {
        "maximum_log_prob_replay_error": "exact_replay_log_prob_tolerance",
        "maximum_recurrent_replay_error": "exact_replay_recurrent_tolerance",
        "maximum_value_replay_error": "exact_replay_value_tolerance",
    }
    for name, tolerance_name in replay_fields.items():
        value = float(replay.get(name, math.inf))
        field_tolerance = float(replay.get(tolerance_name, tolerance))
        if (
            not math.isfinite(value)
            or not math.isfinite(field_tolerance)
            or field_tolerance <= 0.0
            or field_tolerance > tolerance
            or value > field_tolerance
        ):
            raise SchemaValidationError(f"Order9 PPO exact replay failed at {name}")
    continuity_tolerance = replay.get("exact_replay_continuity_tolerance")
    if continuity_tolerance is not None:
        continuity_tolerance = float(continuity_tolerance)
        for name in (
            "maximum_stored_recurrent_continuity_error",
            "maximum_stored_previous_action_continuity_error",
        ):
            value = float(replay.get(name, math.inf))
            if (
                not math.isfinite(value)
                or not math.isfinite(continuity_tolerance)
                or continuity_tolerance <= 0.0
                or value > continuity_tolerance
            ):
                raise SchemaValidationError(
                    f"Order9 PPO exact replay failed at {name}"
                )
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
        "configured_target_environment_steps": (
            plan.configured_target_environment_steps
        ),
        "target_environment_steps": plan.target_environment_steps,
        "generation_environment_steps": plan.generation_environment_steps,
        "base_target_update_count": plan.base_target_update_count,
        "additional_update_count": plan.additional_update_count,
        "branch_parent_update_index": plan.branch_parent_update_index,
        "training_minimum_module_count": plan.training_minimum_module_count,
        "training_maximum_module_count": plan.training_maximum_module_count,
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
    # The accepted C3 bucket set predates the patch-derived moment envelope.
    # Preserve its immutable TaskSpec lineage here; the rollout collector
    # calibrates only the active trajectory's torque bounds at runtime.
    source_task = TaskSpec.from_json(
        (manifest_path.parent / manifest.buckets[0].task_spec_path).read_text(
            encoding="utf-8"
        )
    )
    source_teacher_version = source_task.metadata.get("teacher_version")
    if not isinstance(source_teacher_version, str) or not source_teacher_version:
        raise SchemaValidationError(
            "Order9 rollout bucket source teacher version is missing"
        )
    base_task.metadata["teacher_version"] = source_teacher_version
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
        topology_metadata = bucket.metadata.get("topology_provider")
        precheck = (
            topology_metadata.get("articulated_teacher_precheck")
            if isinstance(topology_metadata, dict)
            else None
        )
        if precheck is not None:
            task.metadata["order9_c3_articulated_teacher_precheck"] = precheck
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
    "allocate_order9_topology_shard_environments",
    "allocate_order9_topology_stratified_environments",
    "order9_c3_state_inheritance_environment_steps",
    "order9_c3_state_inheritance_phase_indices",
    "load_order9_rollout_result",
    "order9_pi_l_collector_command",
    "required_order9_update_count",
    "resolve_order9_extended_update_budget",
    "resolve_order9_pi_l_stage_plan",
    "run_logged_order9_command",
    "run_parallel_order9_collectors",
    "select_order9_c3_state_inheritance_buckets",
    "select_order9_pi_l_rollout_buckets",
    "select_order9_pi_l_topology_stratified_buckets",
    "validate_order9_completed_update",
    "validate_order9_c3_generation_boundary_coverage",
    "validate_order9_generation_dataset",
    "validate_order9_pi_l_stage_runner_inputs",
    "validate_order9_rollout_result",
    "write_order9_stage_runner_state",
]
