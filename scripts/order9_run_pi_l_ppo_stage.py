#!/usr/bin/env python3
from __future__ import annotations

"""Run or resume a complete fail-closed Order 9 ``pi_L`` PPO stage."""

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.datasets import DatasetSplit
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_curriculum import (
    load_order9_learning_config,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_c3_action_contract import ORDER9_C3_ACTION_CONTRACTS
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_PI_L_STAGE_RUNNER_VERSION,
    allocate_order9_topology_shard_environments,
    load_order9_rollout_result,
    order9_pi_l_collector_command,
    order9_c3_training_wrench_gate_range_scale,
    resolve_order9_pi_l_stage_plan,
    run_logged_order9_command,
    run_parallel_order9_collectors,
    select_order9_c3_state_inheritance_buckets,
    select_order9_pi_l_rollout_buckets,
    select_order9_pi_l_topology_stratified_buckets,
    validate_order9_completed_update,
    validate_order9_c3_generation_boundary_coverage,
    validate_order9_pi_l_stage_runner_inputs,
    validate_order9_rollout_result,
    write_order9_stage_runner_state,
)
from amsrr.training.order9_tensor_on_policy_dataset import (
    build_order9_tensor_pi_l_dataset,
    load_order9_tensor_pi_l_dataset,
    validate_order9_tensor_pi_l_dataset_for_stage,
)
from scripts.order9_train_ppo import run_order9_tensor_pi_l_ppo_training


@dataclass(frozen=True)
class _SelectedRollout:
    split: DatasetSplit
    bucket: object
    topology_shard_index: int | None
    environment_count: int
    rollout_steps: int
    state_inheritance: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", default="c2_pi_l_ppo_fixed_conservative")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--prior-stage-manifest", action="append", required=True)
    parser.add_argument(
        "--stage-root",
        help=(
            "Optional explicit output root for a distinct immutable training "
            "lineage; defaults to the configured canonical stage directory."
        ),
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--stop-after-update-index",
        type=int,
        help="Inclusive operational stop for a bounded run; stage target is unchanged.",
    )
    parser.add_argument(
        "--additional-update-count",
        type=int,
        default=0,
        help=(
            "Complete-generation extension beyond the hash-bound stage quota; "
            "optimization settings and curriculum hashes remain unchanged."
        ),
    )
    parser.add_argument(
        "--branch-parent-update-index",
        type=int,
        help=(
            "Begin a fresh immutable lineage from a checkpoint produced by "
            "this stage at the specified PPO update index."
        ),
    )
    parser.add_argument(
        "--training-min-module-count",
        type=int,
        help="Inclusive C3 module-count subcurriculum lower bound.",
    )
    parser.add_argument(
        "--training-max-module-count",
        type=int,
        help="Inclusive C3 module-count subcurriculum upper bound.",
    )
    parser.add_argument(
        "--c3-action-contract",
        choices=ORDER9_C3_ACTION_CONTRACTS,
        help="Matched C3 action contract applied during rollout and PPO update.",
    )
    parser.add_argument(
        "--maximum-parallel-collector-process-count",
        type=int,
        choices=(1, 2),
        help=(
            "Operational Isaac collector concurrency override. It changes only "
            "wall-clock execution; rollout contents and training budgets remain "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--collector-process-start-stagger-s",
        type=float,
        default=0.0,
        help=(
            "Delay between concurrent Isaac process starts to avoid the shared "
            "Kit/Hub startup lock. This changes wall time only."
        ),
    )
    parser.add_argument(
        "--c3-training-nominal-preload-deficit-mm",
        help=(
            "Comma-separated training-only nominal-preload deficits in mm. "
            "The unmodified 0 mm case must be included."
        ),
    )
    parser.add_argument(
        "--c3-training-nominal-preload-deficit-module-count",
        type=int,
        help="Only this C3 train topology receives the preload deficits.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = REPOSITORY_ROOT.resolve()
    config_path = _resolve(args.config, repository)
    config = load_order9_learning_config(config_path)
    stage = next(
        value for value in config.curriculum.stages if value.stage_id == args.stage
    )
    if (args.training_min_module_count is None) != (
        args.training_max_module_count is None
    ):
        raise ValueError("both C3 training module-count bounds are required")
    training_minimum_module_count = (
        stage.min_modules
        if args.training_min_module_count is None
        else args.training_min_module_count
    )
    training_maximum_module_count = (
        stage.max_modules
        if args.training_max_module_count is None
        else args.training_max_module_count
    )
    training_module_counts = tuple(
        range(training_minimum_module_count, training_maximum_module_count + 1)
    )
    if args.c3_action_contract is not None and (
        args.stage != "c3_pi_l_ppo_arbitrary_morphology"
    ):
        raise ValueError("explicit C3 action contract requires the C3 stage")
    deficit_arguments = (
        args.c3_training_nominal_preload_deficit_mm,
        args.c3_training_nominal_preload_deficit_module_count,
    )
    if any(value is not None for value in deficit_arguments):
        if any(value is None for value in deficit_arguments):
            raise ValueError(
                "both C3 training nominal-preload deficit arguments are required"
            )
        if (
            args.stage != "c3_pi_l_ppo_arbitrary_morphology"
            or not training_minimum_module_count
            <= args.c3_training_nominal_preload_deficit_module_count
            <= training_maximum_module_count
        ):
            raise ValueError(
                "C3 training nominal-preload deficit module count is outside "
                "the active training curriculum"
            )
    physical_model = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path, repository)
    )
    stage_root = (
        _resolve(args.stage_root, repository)
        if args.stage_root is not None
        else (
            repository
            / config.production_runtime.artifact_root
            / "stages"
            / args.stage
        ).resolve()
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    state_path = stage_root / "stage_runner_state.json"
    plan = resolve_order9_pi_l_stage_plan(
        config,
        stage_id=args.stage,
        stage_root=stage_root,
        initial_checkpoint_path=args.initial_checkpoint,
        repository_root=repository,
        additional_update_count=args.additional_update_count,
        branch_parent_update_index=args.branch_parent_update_index,
        expected_physical_model_hash=physical_model.stable_hash(),
        training_minimum_module_count=training_minimum_module_count,
        training_maximum_module_count=training_maximum_module_count,
    )
    buckets = validate_order9_pi_l_stage_runner_inputs(
        config,
        stage_id=args.stage,
        bucket_manifest_path=args.bucket_manifest,
        repository_root=repository,
    )
    for path in args.prior_stage_manifest:
        if not _resolve(path, repository).is_file():
            raise FileNotFoundError(_resolve(path, repository))
    final_update_index = plan.target_update_count - 1
    if args.stop_after_update_index is not None:
        if args.stop_after_update_index < plan.next_update_index:
            raise ValueError("--stop-after-update-index precedes the resume point")
        final_update_index = min(final_update_index, args.stop_after_update_index)
    completed: list[dict[str, object]] = []
    write_order9_stage_runner_state(
        state_path,
        stage_id=args.stage,
        status="running",
        plan=plan,
        current_update_index=(
            None if plan.next_update_index >= plan.target_update_count else plan.next_update_index
        ),
        completed_updates=completed,
    )
    print(
        "ORDER9_STAGE_RUNNER="
        + json.dumps(
            {
                "runner_version": ORDER9_PI_L_STAGE_RUNNER_VERSION,
                "stage_id": args.stage,
                "resume_update_index": plan.next_update_index,
                "final_update_index_this_run": final_update_index,
                "configured_target_environment_steps": (
                    plan.configured_target_environment_steps
                ),
                "target_update_count": plan.target_update_count,
                "base_target_update_count": plan.base_target_update_count,
                "additional_update_count": plan.additional_update_count,
                "branch_parent_update_index": (
                    plan.branch_parent_update_index
                ),
                "completed_environment_steps": plan.completed_environment_steps,
                "target_environment_steps": plan.target_environment_steps,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                "training_module_counts": list(training_module_counts),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    parent_path = Path(plan.parent_checkpoint_path)
    parent_sha = plan.parent_checkpoint_sha256
    current_index: int | None = None
    try:
        for update_index in range(plan.next_update_index, final_update_index + 1):
            current_index = update_index
            generation_started = time.perf_counter()
            generation_id = f"{args.stage}:generation:{update_index:06d}"
            generation_root = (
                stage_root
                / "generations"
                / f"generation_{update_index:06d}"
            )
            raw_root = generation_root / "raw"
            log_root = generation_root / "logs"
            raw_root.mkdir(parents=True, exist_ok=True)
            log_root.mkdir(parents=True, exist_ok=True)
            c3_wrench_gate_range_scale = (
                order9_c3_training_wrench_gate_range_scale(config, update_index)
                if stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
                else None
            )
            if (
                stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
                and config.production_runtime.c3_topology_stratified_updates
            ):
                train_buckets, validation_bucket = (
                    select_order9_pi_l_topology_stratified_buckets(
                        buckets,
                        update_index,
                        minimum_module_count=training_minimum_module_count,
                        maximum_module_count=training_maximum_module_count,
                        topologies_per_module_count=(
                            config.production_runtime.
                            c3_topologies_per_module_count_per_update
                        ),
                    )
                )
            else:
                train_bucket, validation_bucket = select_order9_pi_l_rollout_buckets(
                    buckets,
                    update_index,
                )
                train_buckets = (train_bucket,)
            configured_stage_runtime = resolve_order9_stage_runtime(
                config, stage
            )
            split_environment_count = configured_stage_runtime.environment_count
            phase_reset_rollout_steps = (
                configured_stage_runtime.rollout_steps_per_environment
            )
            if phase_reset_rollout_steps is None:
                raise ValueError("Order9 PPO stage lacks rollout steps")
            topology_environment_allocation = (
                allocate_order9_topology_shard_environments(
                    split_environment_count,
                    len(train_buckets),
                    update_index=update_index,
                )
            )
            module_replica_index: dict[int, int] = {}
            selected: dict[str, _SelectedRollout] = {}
            for shard_index, bucket in enumerate(train_buckets):
                replica_index = module_replica_index.get(
                    bucket.module_count, 0
                )
                module_replica_index[bucket.module_count] = replica_index + 1
                key = (
                    f"train_m{bucket.module_count:02d}"
                    if config.production_runtime.
                    c3_topologies_per_module_count_per_update == 1
                    else (
                        f"train_m{bucket.module_count:02d}_"
                        f"r{replica_index:02d}"
                    )
                )
                selected[key] = _SelectedRollout(
                    split=DatasetSplit.TRAIN,
                    bucket=bucket,
                    topology_shard_index=shard_index,
                    environment_count=(
                        topology_environment_allocation[shard_index]
                    ),
                    rollout_steps=phase_reset_rollout_steps,
                )
            if len(selected) != len(train_buckets):
                raise AssertionError(
                    "Order9 topology shard selection keys are not unique"
                )
            if (
                stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
                and config.production_runtime.c3_state_inheritance_rollouts_enabled
            ):
                inheritance_count_by_module = {
                    module_count: config.production_runtime
                    .c3_state_inheritance_topology_count(module_count)
                    for module_count in training_module_counts
                }
                inheritance_buckets = select_order9_c3_state_inheritance_buckets(
                    buckets,
                    update_index,
                    minimum_module_count=training_minimum_module_count,
                    maximum_module_count=training_maximum_module_count,
                    topology_count_by_module=inheritance_count_by_module,
                )
                inheritance_replica_index: dict[int, int] = {}
                for offset, bucket in enumerate(inheritance_buckets):
                    replica_index = inheritance_replica_index.get(
                        bucket.module_count, 0
                    )
                    inheritance_replica_index[bucket.module_count] = (
                        replica_index + 1
                    )
                    replica_count = inheritance_count_by_module[
                        bucket.module_count
                    ]
                    key = f"train_inherit_m{bucket.module_count:02d}"
                    if replica_count > 1:
                        key += f"_r{replica_index:02d}"
                    selected[key] = _SelectedRollout(
                        split=DatasetSplit.TRAIN,
                        bucket=bucket,
                        topology_shard_index=len(train_buckets) + offset,
                        environment_count=(
                            config.production_runtime
                            .c3_state_inheritance_environment_count_per_module
                        ),
                        rollout_steps=(
                            config.production_runtime
                            .c3_state_inheritance_rollout_steps
                        ),
                        state_inheritance=True,
                    )
            train_shard_count = sum(
                value.split == DatasetSplit.TRAIN for value in selected.values()
            )
            selected["validation"] = _SelectedRollout(
                split=DatasetSplit.VALIDATION,
                bucket=validation_bucket,
                topology_shard_index=None,
                environment_count=split_environment_count,
                rollout_steps=phase_reset_rollout_steps,
            )
            expected_generation_environment_steps = sum(
                value.environment_count * value.rollout_steps
                for value in selected.values()
            )
            if expected_generation_environment_steps != plan.generation_environment_steps:
                raise AssertionError(
                    "Order9 selected rollout budget differs from the stage plan: "
                    f"{expected_generation_environment_steps} != "
                    f"{plan.generation_environment_steps}"
                )
            raw_paths = {
                key: raw_root / f"{key}.pt" for key in selected
            }
            log_paths = {
                key: log_root / f"{key}.log" for key in selected
            }
            missing: list[str] = []
            for key, rollout in selected.items():
                raw_path = raw_paths[key]
                log_path = log_paths[key]
                if not raw_path.is_file() or not log_path.is_file():
                    missing.append(key)
                    continue
                try:
                    existing_payload = load_order9_rollout_result(log_path)
                    validate_order9_rollout_result(
                        existing_payload,
                        stage_id=args.stage,
                        generation_id=generation_id,
                        split=rollout.split,
                        expected_environment_steps=(
                            rollout.environment_count * rollout.rollout_steps
                        ),
                        raw_artifact_path=raw_path,
                        parent_checkpoint_sha256=parent_sha,
                        c3_wrench_gate_range_scale=c3_wrench_gate_range_scale,
                        c3_state_inheritance=rollout.state_inheritance,
                        c3_state_inheritance_initial_phase_indices=(
                            config.production_runtime.
                            c3_state_inheritance_initial_phase_indices
                        ),
                        c3_state_inheritance_fixed_reset_progress_fraction=(
                            config.production_runtime.
                            c3_state_inheritance_fixed_reset_progress_fraction
                        ),
                        c3_action_contract=args.c3_action_contract,
                        c3_training_nominal_preload_deficit_mm=(
                            args.c3_training_nominal_preload_deficit_mm
                            if rollout.split == DatasetSplit.TRAIN
                            else None
                        ),
                        c3_training_nominal_preload_deficit_module_count=(
                            args.c3_training_nominal_preload_deficit_module_count
                            if rollout.split == DatasetSplit.TRAIN
                            else None
                        ),
                    )
                except (OSError, ValueError) as error:
                    print(
                        "ORDER9_STALE_ROLLOUT_RECOLLECT "
                        f"update={update_index} shard={key} reason={error}",
                        flush=True,
                    )
                    stale_suffix = f".stale-{time.time_ns()}"
                    for stale_path in (raw_path, log_path):
                        if not stale_path.exists():
                            continue
                        archived_path = stale_path.with_name(
                            stale_path.name + stale_suffix
                        )
                        stale_path.replace(archived_path)
                        print(
                            "ORDER9_STALE_ROLLOUT_ARCHIVED "
                            f"source={stale_path} archived={archived_path}",
                            flush=True,
                        )
                    missing.append(key)
            collection_wall_s = 0.0
            if missing:
                commands = {
                    key: order9_pi_l_collector_command(
                        python_executable=sys.executable,
                        repository_root=repository,
                        config_path=config_path,
                        stage_id=args.stage,
                        parent_checkpoint_path=parent_path,
                        parent_checkpoint_sha256=parent_sha,
                        c3_reset_bank_path=(
                            stage_root
                            / "phase_reset_banks"
                            / f"{selected[key].bucket.bucket_id}.pt"
                            if stage.stage_id
                            == "c3_pi_l_ppo_arbitrary_morphology"
                            else None
                        ),
                        generation_id=generation_id,
                        output_raw_path=raw_paths[key],
                        bucket=selected[key].bucket,
                        bucket_manifest_path=args.bucket_manifest,
                        c3_wrench_gate_range_scale=c3_wrench_gate_range_scale,
                        tensorboard_log_dir=(
                            stage_root / "tensorboard" / "rollout_shards" / key
                        ),
                        topology_shard_index=(
                            selected[key].topology_shard_index
                            if train_shard_count > 1
                            else None
                        ),
                        topology_shard_count=(
                            train_shard_count
                            if selected[key].split == DatasetSplit.TRAIN
                            and train_shard_count > 1
                            else None
                        ),
                        topology_shard_environment_count=(
                            selected[key].environment_count
                            if selected[key].split == DatasetSplit.TRAIN
                            and train_shard_count > 1
                            else None
                        ),
                        c3_state_inheritance=(
                            selected[key].state_inheritance
                        ),
                        c3_action_contract=args.c3_action_contract,
                        c3_training_nominal_preload_deficit_mm=(
                            args.c3_training_nominal_preload_deficit_mm
                            if selected[key].split == DatasetSplit.TRAIN
                            else None
                        ),
                        c3_training_nominal_preload_deficit_module_count=(
                            args.c3_training_nominal_preload_deficit_module_count
                            if selected[key].split == DatasetSplit.TRAIN
                            else None
                        ),
                    )
                    for key in missing
                }
                print(
                    f"ORDER9_GENERATION_START update={update_index} "
                    f"collect={','.join(missing)}",
                    flush=True,
                )
                if len(commands) == 1:
                    key, command = next(iter(commands.items()))
                    result = run_logged_order9_command(
                        command,
                        repository_root=repository,
                        log_path=log_paths[key],
                    )
                    collection_wall_s = result.wall_elapsed_s
                else:
                    collection_wall_s, _ = run_parallel_order9_collectors(
                        commands,
                        repository_root=repository,
                        log_paths={key: log_paths[key] for key in missing},
                        maximum_parallel_process_count=(
                            args.maximum_parallel_collector_process_count
                            or config.production_runtime.
                            c3_topology_shard_parallel_process_count
                        ),
                        process_start_stagger_s=(
                            args.collector_process_start_stagger_s
                        ),
                    )
            rollout_summaries: dict[str, dict[str, object]] = {}
            raw_hashes: dict[str, str] = {}
            for key, rollout in selected.items():
                payload = load_order9_rollout_result(log_paths[key])
                digest = validate_order9_rollout_result(
                    payload,
                    stage_id=args.stage,
                    generation_id=generation_id,
                    split=rollout.split,
                    expected_environment_steps=(
                        rollout.environment_count * rollout.rollout_steps
                    ),
                    raw_artifact_path=raw_paths[key],
                    parent_checkpoint_sha256=parent_sha,
                    c3_wrench_gate_range_scale=c3_wrench_gate_range_scale,
                    c3_state_inheritance=rollout.state_inheritance,
                    c3_state_inheritance_initial_phase_indices=(
                        config.production_runtime.
                        c3_state_inheritance_initial_phase_indices
                    ),
                    c3_state_inheritance_fixed_reset_progress_fraction=(
                        config.production_runtime.
                        c3_state_inheritance_fixed_reset_progress_fraction
                    ),
                    c3_action_contract=args.c3_action_contract,
                    c3_training_nominal_preload_deficit_mm=(
                        args.c3_training_nominal_preload_deficit_mm
                        if rollout.split == DatasetSplit.TRAIN
                        else None
                    ),
                    c3_training_nominal_preload_deficit_module_count=(
                        args.c3_training_nominal_preload_deficit_module_count
                        if rollout.split == DatasetSplit.TRAIN
                        else None
                    ),
                )
                rollout_summaries[key] = payload
                raw_hashes[key] = digest
            boundary_transition_coverage = None
            if args.stage == "c3_pi_l_ppo_arbitrary_morphology":
                boundary_transition_coverage = (
                    validate_order9_c3_generation_boundary_coverage(
                        rollout_summaries
                    )
                )
            dataset_root = generation_root / "dataset"
            dataset_manifest = dataset_root / "manifest.json"
            dataset_build_wall_s = 0.0
            tensor_bundle = None
            if not dataset_manifest.is_file():
                build_log = log_root / "dataset_build.log"
                _write_operation_log(
                    build_log,
                    operation="build_tensor_native_dataset_index",
                    status="started",
                    payload={"generation_id": generation_id},
                )
                build_started = time.perf_counter()
                tensor_bundle = build_order9_tensor_pi_l_dataset(
                    dataset_root,
                    raw_artifact_paths=tuple(raw_paths.values()),
                    generation_id=generation_id,
                    stage_id=args.stage,
                    behavior_checkpoint_path=parent_path,
                    config=config,
                    physical_model=physical_model,
                    known_sha256_by_path={
                        raw_paths[key]: raw_hashes[key] for key in selected
                    },
                    c3_training_module_counts=(
                        training_module_counts
                        if stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
                        else None
                    ),
                )
                dataset_build_wall_s = time.perf_counter() - build_started
                _write_operation_log(
                    build_log,
                    operation="build_tensor_native_dataset_index",
                    status="completed",
                    payload={
                        "generation_id": generation_id,
                        "manifest_sha256": tensor_bundle.manifest_sha256,
                        "environment_step_count": (
                            tensor_bundle.manifest.environment_step_count
                        ),
                        "record_schema_materialization": False,
                        "jsonl_materialization": False,
                        "wall_elapsed_s": dataset_build_wall_s,
                    },
                    append=True,
                )
            if tensor_bundle is None:
                tensor_bundle = load_order9_tensor_pi_l_dataset(dataset_manifest)
            dataset_validation = validate_order9_tensor_pi_l_dataset_for_stage(
                tensor_bundle,
                config=config,
                stage_id=args.stage,
                behavior_checkpoint_sha256=parent_sha,
            )
            if not dataset_validation.valid:
                raise ValueError(
                    "Order9 tensor generation dataset is invalid: "
                    + ",".join(dataset_validation.failures)
                )
            if (
                tensor_bundle.manifest.generation_id != generation_id
                or tensor_bundle.manifest.environment_step_count
                != plan.generation_environment_steps
            ):
                raise ValueError("Order9 tensor generation identity/count differs")
            dataset_sha = tensor_bundle.manifest_sha256
            result_path = (
                stage_root
                / f"update_{update_index:06d}"
                / f"training_result_update_{update_index:06d}.json"
            )
            training_command_wall_s = 0.0
            preloaded_bundle_used = False
            if not result_path.is_file():
                training_log = stage_root / "logs" / f"update_{update_index:06d}.log"
                preloaded_bundle_used = True
                _write_operation_log(
                    training_log,
                    operation="execute_ppo_update",
                    status="started",
                    payload={
                        "update_index": update_index,
                        "preloaded_bundle_used": preloaded_bundle_used,
                    },
                )
                training_started = time.perf_counter()
                executed, _, _ = run_order9_tensor_pi_l_ppo_training(
                    config_path=config_path,
                    stage_id=args.stage,
                    rollout_dataset_path=dataset_manifest,
                    rollout_bundle=tensor_bundle,
                    parent_checkpoint_path=parent_path,
                    update_index=update_index,
                    prior_stage_manifest_paths=tuple(
                        _resolve(prior, repository)
                        for prior in args.prior_stage_manifest
                    ),
                    device=args.device or config.production_runtime.device,
                    output_dir=stage_root,
                    tensorboard_log_dir=stage_root / "tensorboard",
                    c3_action_contract=args.c3_action_contract,
                )
                training_command_wall_s = time.perf_counter() - training_started
                _write_operation_log(
                    training_log,
                    operation="execute_ppo_update",
                    status="completed",
                    payload={
                        "update_index": update_index,
                        "checkpoint_sha256": executed.checkpoint_sha256,
                        "preloaded_bundle_used": preloaded_bundle_used,
                        "wall_elapsed_s": training_command_wall_s,
                    },
                    append=True,
                )
            training = validate_order9_completed_update(
                result_path,
                repository_root=repository,
                stage_id=args.stage,
                update_index=update_index,
                parent_checkpoint_sha256=parent_sha,
                rollout_manifest_sha256=dataset_sha,
                expected_environment_steps=plan.generation_environment_steps,
                c3_boundary_fine_tune=(
                    config.optimization.c3_boundary_fine_tune
                    if (
                        args.stage == "c3_pi_l_ppo_arbitrary_morphology"
                        and config.optimization.c3_boundary_fine_tune.enabled
                    )
                    else None
                ),
                c3_action_contract=args.c3_action_contract,
                c3_state_inheritance_required=(
                    stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
                    and config.production_runtime
                    .c3_state_inheritance_rollouts_enabled
                ),
                expected_state_inheritance_shard_count=(
                    sum(
                        config.production_runtime
                        .c3_state_inheritance_topology_count(module_count)
                        for module_count in training_module_counts
                    )
                    if stage.stage_id
                    == "c3_pi_l_ppo_arbitrary_morphology"
                    and config.production_runtime
                    .c3_state_inheritance_rollouts_enabled
                    else 0
                ),
                expected_topology_module_counts=training_module_counts,
                expected_topology_shard_count=train_shard_count,
            )
            metrics_payload = json.loads(
                _resolve(training.metrics_path, repository).read_text(encoding="utf-8")
            )
            summary = {
                "update_index": update_index,
                "generation_id": generation_id,
                "train_bucket_ids": [bucket.bucket_id for bucket in train_buckets],
                "state_inheritance_bucket_ids": [
                    rollout.bucket.bucket_id
                    for rollout in selected.values()
                    if rollout.state_inheritance
                ],
                "state_inheritance_topology_count_by_module": {
                    str(module_count): config.production_runtime
                    .c3_state_inheritance_topology_count(module_count)
                    for module_count in training_module_counts
                },
                "train_module_counts": [
                    bucket.module_count for bucket in train_buckets
                ],
                "training_module_counts": list(training_module_counts),
                "train_environment_count_by_module": {
                    str(module_count): sum(
                        rollout.environment_count
                        for rollout in selected.values()
                        if rollout.split == DatasetSplit.TRAIN
                        and rollout.bucket.module_count == module_count
                    )
                    for module_count in sorted(
                        {bucket.module_count for bucket in train_buckets}
                    )
                },
                "validation_bucket_id": validation_bucket.bucket_id,
                "parent_checkpoint_sha256": parent_sha,
                "raw_artifact_sha256": raw_hashes,
                "boundary_transition_coverage": (
                    boundary_transition_coverage
                ),
                "rollout_manifest_sha256": dataset_sha,
                "child_checkpoint_sha256": training.checkpoint_sha256,
                "environment_steps": training.consumed_environment_steps,
                "phase_reset_train_environment_steps": sum(
                    rollout.environment_count * rollout.rollout_steps
                    for rollout in selected.values()
                    if rollout.split == DatasetSplit.TRAIN
                    and not rollout.state_inheritance
                ),
                "state_inheritance_train_environment_steps": sum(
                    rollout.environment_count * rollout.rollout_steps
                    for rollout in selected.values()
                    if rollout.state_inheritance
                ),
                "collection_command_wall_elapsed_s": collection_wall_s,
                "dataset_build_command_wall_elapsed_s": dataset_build_wall_s,
                "training_command_wall_elapsed_s": training_command_wall_s,
                "preloaded_bundle_used": preloaded_bundle_used,
                "tensor_native_dataset": True,
                "ppo_update_wall_elapsed_s": metrics_payload["update_wall_elapsed_s"],
                "generation_wall_elapsed_s": time.perf_counter() - generation_started,
                "approximate_kl": training.ppo_update.approximate_kl,
                "clipped_fraction": training.ppo_update.clipped_fraction,
                "actor_loss": training.ppo_update.actor_loss,
                "value_loss": training.ppo_update.value_loss,
                "total_loss": training.ppo_update.total_loss,
                "early_stopped_for_kl": training.ppo_update.early_stopped_for_kl,
                "maximum_phase_kl_observed": training.ppo_update.metadata.get(
                    "maximum_phase_kl_observed"
                ),
                "phase_kl_mean_by_actor_phase": training.ppo_update.metadata.get(
                    "phase_kl_mean_by_actor_phase"
                ),
                "completed_epoch_count": training.ppo_update.completed_epoch_count,
                "optimizer_step_count": training.ppo_update.optimizer_step_count,
                "runtime_load": _compact_runtime_load(metrics_payload["runtime_load"]),
                "rollout_summary": {
                    split: {
                        key: payload.get(key)
                        for key in (
                            "aggregate_env_steps_per_s",
                            "end_to_end_env_steps_per_s",
                            "setup_wall_elapsed_s",
                            "wall_elapsed_s",
                            "terminal_count",
                            "successful_terminal_count",
                            "runtime_load",
                        )
                    }
                    for split, payload in rollout_summaries.items()
                },
            }
            del tensor_bundle
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            completed.append(summary)
            parent_path = _resolve(training.checkpoint_path, repository)
            parent_sha = training.checkpoint_sha256
            write_order9_stage_runner_state(
                state_path,
                stage_id=args.stage,
                status="running",
                plan=plan,
                current_update_index=(
                    update_index + 1
                    if update_index + 1 < plan.target_update_count
                    else None
                ),
                completed_updates=completed,
            )
            console_summary = {
                "checkpoint_sha256": summary.get("child_checkpoint_sha256"),
                "generation_wall_elapsed_s": summary.get(
                    "generation_wall_elapsed_s"
                ),
                "optimizer_step_count": summary.get("optimizer_step_count"),
                "update_index": summary.get("update_index"),
            }
            print(
                "ORDER9_UPDATE_COMPLETE="
                + json.dumps(console_summary, sort_keys=True),
                flush=True,
            )
        fully_complete = final_update_index == plan.target_update_count - 1
        write_order9_stage_runner_state(
            state_path,
            stage_id=args.stage,
            status="completed" if fully_complete else "running",
            plan=plan,
            current_update_index=None if fully_complete else final_update_index + 1,
            completed_updates=completed,
        )
        print(
            "ORDER9_STAGE_TRAINING_COMPLETE="
            + json.dumps(
                {
                    "stage_id": args.stage,
                    "target_reached": fully_complete,
                    "additional_update_count": plan.additional_update_count,
                    "last_update_index": (
                        final_update_index
                        if final_update_index >= plan.next_update_index
                        else plan.next_update_index - 1
                    ),
                    "last_checkpoint_path": str(parent_path),
                    "last_checkpoint_sha256": parent_sha,
                    "state_path": str(state_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        write_order9_stage_runner_state(
            state_path,
            stage_id=args.stage,
            status="failed",
            plan=plan,
            current_update_index=current_index,
            completed_updates=completed,
            failure=f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
        return 1


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


def _compact_runtime_load(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {name: item for name, item in value.items() if name != "samples"}


def _write_operation_log(
    path: Path,
    *,
    operation: str,
    status: str,
    payload: dict[str, object],
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(
            "ORDER9_OPERATION="
            + json.dumps(
                {
                    "operation": operation,
                    "status": status,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()


if __name__ == "__main__":
    raise SystemExit(main())
