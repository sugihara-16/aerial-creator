#!/usr/bin/env python3
from __future__ import annotations

"""Run and finalize the resumable formal C3 promotion evaluation."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_c3_action_contract import ORDER9_C3_ACTION_CONTRACTS
from amsrr.training.order9_c3_promotion import (
    ORDER9_C3_PROMOTION_RUNNER_VERSION,
    load_order9_evaluation_episode_jsonl,
    measured_order9_training_rollout_throughput,
    order9_c3_promotion_success_upper_bound,
    recover_order9_c3_horizon_timeout_evidence,
    resolve_order9_training_generation_lineage,
    select_order9_c3_promotion_buckets,
    validate_order9_c3_promotion_bucket_evidence,
    write_order9_c3_promotion_state,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_evaluation import (
    build_order9_stage_evaluation_report,
    write_order9_stage_evaluation_report,
)
from amsrr.training.order9_pi_l_stage_runner import (
    coalesce_order9_same_morphology_collectors,
    order9_pi_l_collector_command,
    run_parallel_order9_collectors,
    validate_order9_pi_l_stage_runner_inputs,
)
from amsrr.training.order9_pipeline import (
    finalize_order9_stage_from_evaluation,
    load_order9_stage_manifest,
    order9_schedule_hash,
    order9_stage_by_id,
)
from amsrr.utils.hashing import hash_file


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--active-manifest", required=True)
    parser.add_argument("--training-lineage-root", required=True)
    parser.add_argument(
        "--reset-bank-root",
        help=(
            "Optional reset-bank directory. Defaults to "
            "<training-lineage-root>/phase_reset_banks."
        ),
    )
    parser.add_argument("--through-update", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-count-per-bucket", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument(
        "--bucket-limit",
        type=int,
        default=None,
        help="Collect only the leading validation buckets, without finalizing.",
    )
    parser.add_argument(
        "--bucket-id",
        action="append",
        default=[],
        help=(
            "Collect only the named fixed validation bucket(s), without "
            "finalizing. May be repeated."
        ),
    )
    parser.add_argument(
        "--diagnostic-action-ablation",
        choices=("zero_global", "zero_joint", "contact_compression_only"),
        help=(
            "Run an acceptance-ineligible deterministic pi_L action ablation. "
            "This requires a partial bucket selection and never finalizes C3."
        ),
    )
    parser.add_argument(
        "--diagnostic-nominal-qpid-only",
        action="store_true",
        help=(
            "Run an acceptance-ineligible nominal-trajectory plus QPID "
            "diagnostic. This requires a partial bucket selection and never "
            "finalizes C3."
        ),
    )
    parser.add_argument(
        "--diagnostic-virtual-contact-lead-mm",
        type=float,
        help=(
            "Run an acceptance-ineligible full-sequence evaluation with an "
            "explicit total inward IK servo lead. This requires a partial "
            "bucket selection and never finalizes C3."
        ),
    )
    parser.add_argument(
        "--c3-action-contract",
        choices=ORDER9_C3_ACTION_CONTRACTS,
        help=(
            "Apply one explicit C3 action contract throughout the formal "
            "rollout. This also disables the legacy release/retreat action mask. "
            "Nominal-QPID diagnostics may retain this argument so a checkpoint "
            "with a contract-specific decoder can be loaded, although no policy "
            "action is applied."
        ),
    )
    parser.add_argument("--maximum-parallel-process-count", type=int, default=2)
    parser.add_argument(
        "--enable-persistent-morphology-coalescing",
        action="store_true",
        help=(
            "Reuse one Isaac process for multiple validation buckets with the "
            "same morphology. Disabled by default because a second PhysX scene "
            "initialization can retain the first scene's GPU buffers."
        ),
    )
    parser.add_argument(
        "--process-start-stagger-s",
        type=float,
        default=0.0,
        help=(
            "Delay consecutive Isaac starts within a parallel group to avoid "
            "the shared Kit/Hub startup lock. This changes wall time only."
        ),
    )
    parser.add_argument(
        "--disable-fail-fast",
        action="store_true",
        help=(
            "Collect every selected bucket even after the promotion gate is "
            "mathematically unreachable.  This is intended for fixed "
            "per-update diagnostic matrices; formal promotion retains "
            "fail-fast by default."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--cpu-list", default="0-7,9-31")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.episode_count_per_bucket < 1 or args.rollout_steps < 1:
        raise ValueError("evaluation episode count and rollout steps must be positive")
    if not 1 <= args.maximum_parallel_process_count <= 4:
        raise ValueError("C3 formal evaluation permits one to four Isaac processes")
    if (
        not math.isfinite(args.process_start_stagger_s)
        or args.process_start_stagger_s < 0.0
    ):
        raise ValueError("C3 formal evaluation process-start stagger is invalid")
    if args.bucket_limit is not None and not 1 <= args.bucket_limit <= 14:
        raise ValueError("bucket limit must be between 1 and 14")
    if args.bucket_limit is not None and args.bucket_id:
        raise ValueError("bucket limit and explicit bucket ids are mutually exclusive")
    repository = REPOSITORY_ROOT
    config_path = _resolve(args.config, repository)
    manifest_path = _resolve(args.bucket_manifest, repository)
    checkpoint_path = _resolve(args.checkpoint, repository)
    active_manifest_path = _resolve(args.active_manifest, repository)
    lineage_root = _resolve(args.training_lineage_root, repository)
    reset_bank_root = (
        lineage_root / "phase_reset_banks"
        if args.reset_bank_root is None
        else _resolve(args.reset_bank_root, repository)
    )
    output = _resolve(args.output_dir, repository)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "promotion_runner_state.json"
    report_path = output / "stage_evaluation.json"
    final_path = output / "stage_promoted_or_rejected.json"

    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, STAGE_ID)
    manifest = validate_order9_pi_l_stage_runner_inputs(
        config,
        stage_id=STAGE_ID,
        bucket_manifest_path=manifest_path,
        repository_root=repository,
    )
    buckets = select_order9_c3_promotion_buckets(manifest)
    partial_collection = args.bucket_limit is not None or bool(args.bucket_id)
    if (
        args.diagnostic_action_ablation is not None
        or args.diagnostic_nominal_qpid_only
        or args.diagnostic_virtual_contact_lead_mm is not None
    ) and not partial_collection:
        raise ValueError(
            "diagnostic evaluation requires --bucket-id or --bucket-limit"
        )
    if (
        args.diagnostic_action_ablation is not None
        and args.diagnostic_nominal_qpid_only
    ):
        raise ValueError(
            "diagnostic action ablation and nominal-QPID-only are mutually exclusive"
        )
    if args.diagnostic_virtual_contact_lead_mm is not None and (
        not math.isfinite(args.diagnostic_virtual_contact_lead_mm)
        or not 0.0 <= args.diagnostic_virtual_contact_lead_mm <= 20.0
    ):
        raise ValueError("diagnostic virtual-contact lead must be in [0, 20] mm")
    if args.diagnostic_virtual_contact_lead_mm is not None and (
        args.diagnostic_action_ablation is not None
        or args.diagnostic_nominal_qpid_only
    ):
        raise ValueError(
            "diagnostic virtual-contact lead cannot be combined with another "
            "diagnostic action mode"
        )
    if args.c3_action_contract is not None and (
        args.diagnostic_action_ablation is not None
    ):
        raise ValueError(
            "explicit C3 action contract and diagnostic action ablation are "
            "mutually exclusive"
        )
    if partial_collection:
        if args.bucket_id:
            requested = set(args.bucket_id)
            available = {bucket.bucket_id for bucket in buckets}
            if requested - available:
                raise ValueError(
                    "unknown fixed validation bucket ids: "
                    + ",".join(sorted(requested - available))
                )
            buckets = tuple(
                bucket for bucket in buckets if bucket.bucket_id in requested
            )
        else:
            buckets = buckets[: args.bucket_limit]
    checkpoint_sha256 = hash_file(checkpoint_path)
    active = load_order9_stage_manifest(active_manifest_path)
    if active.policy_checkpoint_sha256_by_family != {
        Order9PolicyFamily.PI_L.value: checkpoint_sha256
    }:
        raise SchemaValidationError(
            "active C3 manifest is not bound to the evaluated pi_L checkpoint"
        )

    completed: dict[str, tuple] = {}
    pending_commands: dict[str, list[str]] = {}
    pending_logs: dict[str, Path] = {}
    for bucket in buckets:
        bucket_root = output / "buckets" / bucket.bucket_id
        raw = bucket_root / "evaluation_rollout.pt"
        episode_jsonl = bucket_root / "evaluation_episodes.jsonl"
        log = bucket_root / "evaluation.log"
        generation_id = f"c3_formal_update_{args.through_update:06d}:{bucket.bucket_id}"
        try:
            completed[bucket.bucket_id] = validate_order9_c3_promotion_bucket_evidence(
                log_path=log,
                raw_artifact_path=raw,
                episode_jsonl_path=episode_jsonl,
                stage_id=STAGE_ID,
                generation_id=generation_id,
                checkpoint_sha256=checkpoint_sha256,
                expected_episode_count=args.episode_count_per_bucket,
            )
            continue
        except (OSError, ValueError, SchemaValidationError):
            try:
                recover_order9_c3_horizon_timeout_evidence(
                    log_path=log,
                    raw_artifact_path=raw,
                    episode_jsonl_path=episode_jsonl,
                    stage_id=STAGE_ID,
                    generation_id=generation_id,
                    checkpoint_sha256=checkpoint_sha256,
                    expected_episode_count=args.episode_count_per_bucket,
                )
                completed[bucket.bucket_id] = (
                    validate_order9_c3_promotion_bucket_evidence(
                        log_path=log,
                        raw_artifact_path=raw,
                        episode_jsonl_path=episode_jsonl,
                        stage_id=STAGE_ID,
                        generation_id=generation_id,
                        checkpoint_sha256=checkpoint_sha256,
                        expected_episode_count=args.episode_count_per_bucket,
                    )
                )
                continue
            except (OSError, ValueError, SchemaValidationError):
                _archive_partial(bucket_root)
        reset_bank = reset_bank_root / f"{bucket.bucket_id}.pt"
        command = order9_pi_l_collector_command(
            python_executable=args.python_executable,
            repository_root=repository,
            config_path=config_path,
            stage_id=STAGE_ID,
            parent_checkpoint_path=checkpoint_path,
            parent_checkpoint_sha256=checkpoint_sha256,
            c3_reset_bank_path=reset_bank,
            generation_id=generation_id,
            output_raw_path=raw,
            bucket=bucket,
            bucket_manifest_path=manifest_path,
            c3_action_contract=args.c3_action_contract,
        )
        command.extend(
            [
                "--num-envs",
                str(args.episode_count_per_bucket),
                "--rollout-steps",
                str(args.rollout_steps),
                "--evaluation-jsonl",
                str(episode_jsonl),
                "--evaluation-episode-count",
                str(args.episode_count_per_bucket),
                "--formal-phase-zero-start",
                "--diagnostic-collision-evidence",
                "--no-tensorboard",
            ]
        )
        if args.diagnostic_action_ablation is not None:
            command.extend(
                [
                    "--diagnostic-action-ablation",
                    args.diagnostic_action_ablation,
                ]
            )
        if args.diagnostic_nominal_qpid_only:
            command.append("--diagnostic-nominal-qpid-only")
        if args.diagnostic_virtual_contact_lead_mm is not None:
            command.extend(
                [
                    "--virtual-contact-lead-mm",
                    str(args.diagnostic_virtual_contact_lead_mm),
                ]
            )
        if args.cpu_list:
            command = ["taskset", "-c", args.cpu_list, *command]
        pending_commands[bucket.bucket_id] = command
        pending_logs[bucket.bucket_id] = log

    _write_state(
        state_path,
        buckets=buckets,
        completed=completed,
        checkpoint_sha256=checkpoint_sha256,
        status="collecting" if pending_commands else "aggregating",
    )
    if pending_commands:
        bucket_by_id = {bucket.bucket_id: bucket for bucket in buckets}
        if args.enable_persistent_morphology_coalescing:
            process_commands, process_logs, process_members = (
                coalesce_order9_same_morphology_collectors(
                    pending_commands,
                    log_paths=pending_logs,
                    morphology_hash_by_name={
                        name: bucket_by_id[name].morphology_hash
                        for name in pending_commands
                    },
                    batch_root=output / "persistent_process_batches",
                )
            )
        else:
            process_commands = dict(pending_commands)
            process_logs = dict(pending_logs)
            process_members = {
                name: (name,) for name in pending_commands
            }
        process_names = list(process_commands)
        for start in range(
            0, len(process_names), args.maximum_parallel_process_count
        ):
            impossibility = None
            if not args.disable_fail_fast:
                impossibility = _promotion_success_impossibility(
                    stage=stage,
                    buckets=buckets,
                    completed=completed,
                    episodes_per_bucket=args.episode_count_per_bucket,
                )
            if impossibility is not None:
                _write_impossibility_state(
                    state_path,
                    output / "promotion_mathematically_impossible.json",
                    buckets=buckets,
                    completed=completed,
                    checkpoint_sha256=checkpoint_sha256,
                    impossibility=impossibility,
                )
                print(
                    "ORDER9_C3_PROMOTION_MATHEMATICALLY_IMPOSSIBLE="
                    + json.dumps(impossibility, sort_keys=True),
                    flush=True,
                )
                return 2
            group = process_names[
                start : start + args.maximum_parallel_process_count
            ]
            run_parallel_order9_collectors(
                {name: process_commands[name] for name in group},
                repository_root=repository,
                log_paths={name: process_logs[name] for name in group},
                maximum_parallel_process_count=args.maximum_parallel_process_count,
                process_start_stagger_s=args.process_start_stagger_s,
            )
            completed_names = tuple(
                name
                for process_name in group
                for name in process_members[process_name]
            )
            for name in completed_names:
                bucket_root = output / "buckets" / name
                generation_id = f"c3_formal_update_{args.through_update:06d}:{name}"
                log = bucket_root / "evaluation.log"
                raw = bucket_root / "evaluation_rollout.pt"
                episode_jsonl = bucket_root / "evaluation_episodes.jsonl"
                try:
                    completed[name] = validate_order9_c3_promotion_bucket_evidence(
                        log_path=log,
                        raw_artifact_path=raw,
                        episode_jsonl_path=episode_jsonl,
                        stage_id=STAGE_ID,
                        generation_id=generation_id,
                        checkpoint_sha256=checkpoint_sha256,
                        expected_episode_count=args.episode_count_per_bucket,
                    )
                except (OSError, ValueError, SchemaValidationError):
                    recover_order9_c3_horizon_timeout_evidence(
                        log_path=log,
                        raw_artifact_path=raw,
                        episode_jsonl_path=episode_jsonl,
                        stage_id=STAGE_ID,
                        generation_id=generation_id,
                        checkpoint_sha256=checkpoint_sha256,
                        expected_episode_count=args.episode_count_per_bucket,
                    )
                    completed[name] = validate_order9_c3_promotion_bucket_evidence(
                        log_path=log,
                        raw_artifact_path=raw,
                        episode_jsonl_path=episode_jsonl,
                        stage_id=STAGE_ID,
                        generation_id=generation_id,
                        checkpoint_sha256=checkpoint_sha256,
                        expected_episode_count=args.episode_count_per_bucket,
                    )
            _write_state(
                state_path,
                buckets=buckets,
                completed=completed,
                checkpoint_sha256=checkpoint_sha256,
                status="collecting" if len(completed) < len(buckets) else "aggregating",
            )
            print(
                "ORDER9_C3_PROMOTION_PROGRESS="
                + json.dumps(
                    {"completed_bucket_count": len(completed), "bucket_count": len(buckets)},
                    sort_keys=True,
                ),
                flush=True,
            )

    if partial_collection:
        _write_state(
            state_path,
            buckets=buckets,
            completed=completed,
            checkpoint_sha256=checkpoint_sha256,
            status="partial_complete",
        )
        print(
            "ORDER9_C3_PROMOTION_PARTIAL="
            + json.dumps(
                {
                    "completed_bucket_count": len(completed),
                    "completed_episode_count": sum(
                        len(rows) for rows in completed.values()
                    ),
                    "bucket_ids": [bucket.bucket_id for bucket in buckets],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    episodes = tuple(
        episode
        for bucket in buckets
        for episode in load_order9_evaluation_episode_jsonl(
            output / "buckets" / bucket.bucket_id / "evaluation_episodes.jsonl"
        )
    )
    generation_roots = resolve_order9_training_generation_lineage(
        active_manifest_path,
        through_update=args.through_update,
    )
    training_steps, training_wall = measured_order9_training_rollout_throughput(
        generation_roots,
        required_environment_count=(
            config.runtime_benchmark.initial_environment_count
        ),
    )
    report = build_order9_stage_evaluation_report(
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        episodes=episodes,
        policy_checkpoint_sha256_by_family={
            Order9PolicyFamily.PI_L: checkpoint_sha256
        },
        training_rollout_environment_step_count=training_steps,
        training_rollout_wall_elapsed_s=training_wall,
        metadata={
            "runner_version": ORDER9_C3_PROMOTION_RUNNER_VERSION,
            "checkpoint_path": str(checkpoint_path),
            "bucket_manifest_path": str(manifest_path),
            "bucket_manifest_sha256": hash_file(manifest_path),
            "bucket_ids": [bucket.bucket_id for bucket in buckets],
            "episodes_per_bucket": args.episode_count_per_bucket,
            "deterministic_policy": True,
            "initial_phase_index": 0,
            "first_terminal_only": True,
            "full_mesh_evaluation": True,
            "raw_contact_actor_input": False,
            "training_rollout_throughput_contract": (
                "production_width_collectors_conservative_serial_wall_v2"
            ),
            "training_rollout_environment_count": (
                config.runtime_benchmark.initial_environment_count
            ),
            "training_rollout_wall_is_conservative_serial_sum": True,
        },
    )
    write_order9_stage_evaluation_report(report_path, report)
    artifact_paths = {
        "c3_rollout_bucket_manifest": manifest_path,
    }
    for bucket in buckets:
        bucket_root = output / "buckets" / bucket.bucket_id
        artifact_paths[f"evaluation_raw_{bucket.bucket_id}"] = (
            bucket_root / "evaluation_rollout.pt"
        )
        artifact_paths[f"evaluation_episodes_{bucket.bucket_id}"] = (
            bucket_root / "evaluation_episodes.jsonl"
        )
    finalized = finalize_order9_stage_from_evaluation(
        active,
        config,
        evaluation_report_path=report_path,
        output_artifact_paths=artifact_paths,
        checkpoint_paths_by_family={Order9PolicyFamily.PI_L: checkpoint_path},
        output_path=final_path,
    )
    _write_state(
        state_path,
        buckets=buckets,
        completed=completed,
        checkpoint_sha256=checkpoint_sha256,
        status="promoted" if finalized.promoted else "rejected",
        final_manifest=final_path,
    )
    metrics = report.stage_metrics()
    print(
        "ORDER9_C3_PROMOTION_COMPLETE="
        + json.dumps(
            {
                "status": finalized.status.value,
                "promoted": finalized.promoted,
                "episode_count": metrics.episode_count,
                "success_rate": metrics.success_rate,
                "safety_failure_episode_count": metrics.safety_failure_episode_count,
                "aggregate_env_steps_per_s": metrics.aggregate_env_steps_per_s,
                "failed_gates": finalized.promotion_failed_gates,
                "manifest": str(final_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


def _archive_partial(bucket_root: Path) -> None:
    if not bucket_root.exists() or not any(bucket_root.iterdir()):
        return
    archive = bucket_root.parent.parent / "stale" / (
        f"{bucket_root.name}-{time.time_ns()}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(bucket_root), str(archive))


def _write_state(
    path: Path,
    *,
    buckets,
    completed,
    checkpoint_sha256: str,
    status: str,
    final_manifest: Path | None = None,
) -> None:
    write_order9_c3_promotion_state(
        path,
        {
            "runner_version": ORDER9_C3_PROMOTION_RUNNER_VERSION,
            "stage_id": STAGE_ID,
            "status": status,
            "checkpoint_sha256": checkpoint_sha256,
            "bucket_count": len(buckets),
            "completed_bucket_count": len(completed),
            "completed_episode_count": sum(len(rows) for rows in completed.values()),
            "completed_bucket_ids": sorted(completed),
            "pending_bucket_ids": [
                bucket.bucket_id
                for bucket in buckets
                if bucket.bucket_id not in completed
            ],
            "final_manifest": None if final_manifest is None else str(final_manifest),
        },
    )


def _promotion_success_impossibility(
    *,
    stage,
    buckets,
    completed,
    episodes_per_bucket: int,
) -> dict[str, object] | None:
    """Return fail-fast evidence once the success gate is unreachable.

    Every uncollected episode is optimistically counted as a success.  If that
    upper bound is still below the configured promotion threshold, further
    Isaac collection cannot change the decision and would only waste runtime.
    """

    completed_safety_failure_count = sum(
        int(episode.safety_failure)
        for rows in completed.values()
        for episode in rows
    )
    if completed_safety_failure_count > int(
        stage.maximum_safety_failure_episodes
    ):
        return {
            "contract": "order9_c3_promotion_safety_fail_fast_v1",
            "failed_gate": "maximum_safety_failure_episodes",
            "maximum_safety_failure_episodes": int(
                stage.maximum_safety_failure_episodes
            ),
            "completed_safety_failure_episode_count": (
                completed_safety_failure_count
            ),
            "completed_episode_count": sum(
                len(rows) for rows in completed.values()
            ),
        }
    total_episode_count = len(buckets) * episodes_per_bucket
    completed_episode_count = sum(len(rows) for rows in completed.values())
    completed_success_count = sum(
        int(episode.task_success)
        for rows in completed.values()
        for episode in rows
    )
    return order9_c3_promotion_success_upper_bound(
        completed_success_count=completed_success_count,
        completed_episode_count=completed_episode_count,
        total_episode_count=total_episode_count,
        minimum_success_rate=float(stage.minimum_success_rate),
    )


def _write_impossibility_state(
    state_path: Path,
    report_path: Path,
    *,
    buckets,
    completed,
    checkpoint_sha256: str,
    impossibility: dict[str, object],
) -> None:
    write_order9_c3_promotion_state(report_path, impossibility)
    _write_state(
        state_path,
        buckets=buckets,
        completed=completed,
        checkpoint_sha256=checkpoint_sha256,
        status="promotion_mathematically_impossible",
        final_manifest=report_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
