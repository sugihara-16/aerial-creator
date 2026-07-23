#!/usr/bin/env python3
from __future__ import annotations

"""Run or resume a complete fail-closed Order 9 ``pi_L`` PPO stage."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.datasets import DatasetSplit
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_PI_L_STAGE_RUNNER_VERSION,
    load_order9_rollout_result,
    order9_pi_l_collector_command,
    resolve_order9_pi_l_stage_plan,
    run_logged_order9_command,
    run_parallel_order9_collectors,
    select_order9_pi_l_rollout_buckets,
    validate_order9_completed_update,
    validate_order9_generation_dataset,
    validate_order9_pi_l_stage_runner_inputs,
    validate_order9_rollout_result,
    write_order9_stage_runner_state,
)
from amsrr.training.order9_tensor_dataset_builder import (
    build_order9_pi_l_on_policy_dataset_with_bundle,
)
from scripts.order9_train_ppo import run_order9_ppo_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", default="c2_pi_l_ppo_fixed_conservative")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--prior-stage-manifest", action="append", required=True)
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = REPOSITORY_ROOT.resolve()
    config_path = _resolve(args.config, repository)
    config = load_order9_learning_config(config_path)
    physical_model = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path, repository)
    )
    stage_root = (
        repository
        / config.production_runtime.artifact_root
        / "stages"
        / args.stage
    ).resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    state_path = stage_root / "stage_runner_state.json"
    plan = resolve_order9_pi_l_stage_plan(
        config,
        stage_id=args.stage,
        stage_root=stage_root,
        initial_checkpoint_path=args.initial_checkpoint,
        repository_root=repository,
        additional_update_count=args.additional_update_count,
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
                "completed_environment_steps": plan.completed_environment_steps,
                "target_environment_steps": plan.target_environment_steps,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
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
            train_bucket, validation_bucket = select_order9_pi_l_rollout_buckets(
                buckets, update_index
            )
            selected = {
                DatasetSplit.TRAIN: train_bucket,
                DatasetSplit.VALIDATION: validation_bucket,
            }
            raw_paths = {
                split: raw_root / f"{split.value}.pt" for split in selected
            }
            log_paths = {
                split: log_root / f"{split.value}.log" for split in selected
            }
            missing = [split for split, path in raw_paths.items() if not path.is_file()]
            collection_wall_s = 0.0
            if missing:
                commands = {
                    split.value: order9_pi_l_collector_command(
                        python_executable=sys.executable,
                        repository_root=repository,
                        config_path=config_path,
                        stage_id=args.stage,
                        parent_checkpoint_path=parent_path,
                        parent_checkpoint_sha256=parent_sha,
                        generation_id=generation_id,
                        output_raw_path=raw_paths[split],
                        bucket=selected[split],
                        bucket_manifest_path=args.bucket_manifest,
                    )
                    for split in missing
                }
                print(
                    f"ORDER9_GENERATION_START update={update_index} "
                    f"collect={','.join(split.value for split in missing)}",
                    flush=True,
                )
                if len(commands) == 1:
                    split_name, command = next(iter(commands.items()))
                    result = run_logged_order9_command(
                        command,
                        repository_root=repository,
                        log_path=log_paths[DatasetSplit(split_name)],
                    )
                    collection_wall_s = result.wall_elapsed_s
                else:
                    collection_wall_s, _ = run_parallel_order9_collectors(
                        commands,
                        repository_root=repository,
                        log_paths={
                            split.value: log_paths[split] for split in missing
                        },
                    )
            rollout_summaries: dict[str, dict[str, object]] = {}
            raw_hashes: dict[str, str] = {}
            split_steps = plan.generation_environment_steps // 2
            for split in selected:
                payload = load_order9_rollout_result(log_paths[split])
                digest = validate_order9_rollout_result(
                    payload,
                    stage_id=args.stage,
                    generation_id=generation_id,
                    split=split,
                    expected_environment_steps=split_steps,
                    raw_artifact_path=raw_paths[split],
                    parent_checkpoint_sha256=parent_sha,
                )
                rollout_summaries[split.value] = payload
                raw_hashes[split.value] = digest
            dataset_root = generation_root / "dataset"
            dataset_manifest = dataset_root / "manifest.json"
            dataset_build_wall_s = 0.0
            preloaded_bundle = None
            if not dataset_manifest.is_file():
                build_log = log_root / "dataset_build.log"
                _write_operation_log(
                    build_log,
                    operation="build_dataset_with_preloaded_bundle",
                    status="started",
                    payload={"generation_id": generation_id},
                )
                build_started = time.perf_counter()
                built = build_order9_pi_l_on_policy_dataset_with_bundle(
                    dataset_root,
                    raw_artifact_paths=(
                        raw_paths[DatasetSplit.TRAIN],
                        raw_paths[DatasetSplit.VALIDATION],
                    ),
                    generation_id=generation_id,
                    stage_id=args.stage,
                    pi_l_checkpoint_path=parent_path,
                    config=config,
                    physical_model=physical_model,
                )
                dataset_build_wall_s = time.perf_counter() - build_started
                preloaded_bundle = built.bundle
                _write_operation_log(
                    build_log,
                    operation="build_dataset_with_preloaded_bundle",
                    status="completed",
                    payload={
                        "generation_id": generation_id,
                        "manifest_sha256": built.bundle.manifest_sha256,
                        "record_count": len(built.bundle.low_level_records),
                        "wall_elapsed_s": dataset_build_wall_s,
                    },
                    append=True,
                )
                del built
            dataset_sha = validate_order9_generation_dataset(
                dataset_manifest,
                stage_id=args.stage,
                generation_id=generation_id,
                parent_checkpoint_sha256=parent_sha,
                expected_environment_steps=plan.generation_environment_steps,
            )
            result_path = (
                stage_root
                / f"update_{update_index:06d}"
                / f"training_result_update_{update_index:06d}.json"
            )
            training_command_wall_s = 0.0
            preloaded_bundle_used = False
            if not result_path.is_file():
                training_log = stage_root / "logs" / f"update_{update_index:06d}.log"
                preloaded_bundle_used = preloaded_bundle is not None
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
                executed, _, _ = run_order9_ppo_training(
                    config_path=config_path,
                    stage_id=args.stage,
                    rollout_dataset_path=dataset_manifest,
                    rollout_bundle=preloaded_bundle,
                    parent_checkpoint_path=parent_path,
                    update_index=update_index,
                    prior_stage_manifest_paths=tuple(
                        _resolve(prior, repository)
                        for prior in args.prior_stage_manifest
                    ),
                    device=args.device or config.production_runtime.device,
                    output_dir=stage_root,
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
            )
            metrics_payload = json.loads(
                _resolve(training.metrics_path, repository).read_text(encoding="utf-8")
            )
            summary = {
                "update_index": update_index,
                "generation_id": generation_id,
                "train_bucket_id": train_bucket.bucket_id,
                "validation_bucket_id": validation_bucket.bucket_id,
                "parent_checkpoint_sha256": parent_sha,
                "raw_artifact_sha256": raw_hashes,
                "rollout_manifest_sha256": dataset_sha,
                "child_checkpoint_sha256": training.checkpoint_sha256,
                "environment_steps": training.consumed_environment_steps,
                "collection_command_wall_elapsed_s": collection_wall_s,
                "dataset_build_command_wall_elapsed_s": dataset_build_wall_s,
                "training_command_wall_elapsed_s": training_command_wall_s,
                "preloaded_bundle_used": preloaded_bundle_used,
                "ppo_update_wall_elapsed_s": metrics_payload["update_wall_elapsed_s"],
                "generation_wall_elapsed_s": time.perf_counter() - generation_started,
                "approximate_kl": training.ppo_update.approximate_kl,
                "clipped_fraction": training.ppo_update.clipped_fraction,
                "actor_loss": training.ppo_update.actor_loss,
                "value_loss": training.ppo_update.value_loss,
                "total_loss": training.ppo_update.total_loss,
                "early_stopped_for_kl": training.ppo_update.early_stopped_for_kl,
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
            del preloaded_bundle
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
            print("ORDER9_UPDATE_COMPLETE=" + json.dumps(summary, sort_keys=True), flush=True)
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
