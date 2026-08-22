#!/usr/bin/env python3
from __future__ import annotations

"""Paired deterministic all-phase C3 comparison on fixed validation buckets."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
)
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_c3_action_contract import ORDER9_C3_ACTION_CONTRACTS
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pi_l_stage_runner import (
    load_order9_rollout_result,
    order9_pi_l_collector_command,
    run_logged_order9_command,
    run_parallel_order9_collectors,
    validate_order9_pi_l_stage_runner_inputs,
)
from amsrr.training.order9_pipeline import order9_schedule_hash
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    Order9TensorRolloutArtifact,
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


ORDER9_PI_L_FIXED_CHECKPOINT_COMPARISON_VERSION = (
    "order9_pi_l_fixed_validation_checkpoint_comparison_v4_reusable_baseline"
)

ORDER9_C3_FIXED_VALIDATION_PHASE_RESET_STRATUM_COUNT = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument(
        "--stage", default="c3_pi_l_ppo_arbitrary_morphology"
    )
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument(
        "--reuse-baseline-evaluation-dir",
        help=(
            "Optional completed comparison directory whose initializer "
            "rollouts and immutable phase-reset banks are reused."
        ),
    )
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--c3-action-contract",
        choices=ORDER9_C3_ACTION_CONTRACTS,
        help=(
            "Explicit C3 action contract used by both checkpoints. Current "
            "v7 policies require this binding."
        ),
    )
    parser.add_argument(
        "--candidate-label",
        default="candidate",
        help="Stable report key for the candidate checkpoint.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument(
        "--limit-buckets",
        type=int,
        help="Diagnostic prefix of validation buckets; omit for the complete set.",
    )
    parser.add_argument(
        "--module-count",
        type=int,
        help="Diagnostic filter for one validation morphology module count.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="IsaacLab Python used by the two paired collector processes.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate_label = str(args.candidate_label)
    if (
        not candidate_label
        or candidate_label == "initializer"
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in candidate_label
        )
    ):
        raise ValueError("candidate label must be a non-initializer file-safe token")
    if args.num_envs < 8 or args.num_envs % 8 != 0:
        raise ValueError("comparison environment count must be a positive multiple of 8")
    if args.rollout_steps < 1:
        raise ValueError("comparison rollout steps must be positive")
    if args.limit_buckets is not None and args.limit_buckets < 1:
        raise ValueError("comparison bucket limit must be positive")
    if args.module_count is not None and not 2 <= args.module_count <= 8:
        raise ValueError("comparison module count must lie in [2, 8]")

    repository = REPOSITORY_ROOT
    config_path = _resolve(args.config, repository)
    manifest_path = _resolve(args.bucket_manifest, repository)
    output = _resolve(args.output_dir, repository)
    reusable_baseline = (
        _resolve(args.reuse_baseline_evaluation_dir, repository)
        if args.reuse_baseline_evaluation_dir is not None
        else None
    )
    reusable_report: Mapping[str, Any] | None = None
    if reusable_baseline is not None:
        reusable_report_path = reusable_baseline / "comparison_report.json"
        if not reusable_report_path.is_file():
            raise FileNotFoundError(reusable_report_path)
        reusable_report = json.loads(
            reusable_report_path.read_text(encoding="utf-8")
        )
    report_path = output / "comparison_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("ORDER9_PI_L_COMPARISON=" + json.dumps(report["summary"], sort_keys=True))
        print(f"report: {report_path}")
        return 0
    output.mkdir(parents=True, exist_ok=True)

    config = load_order9_learning_config(config_path)
    manifest = validate_order9_pi_l_stage_runner_inputs(
        config,
        stage_id=args.stage,
        bucket_manifest_path=manifest_path,
        repository_root=repository,
    )
    validation = sorted(
        (
            bucket
            for bucket in manifest.buckets
            if bucket.split == DatasetSplit.VALIDATION
        ),
        key=lambda value: (value.sample_index, value.bucket_id),
    )
    if args.module_count is not None:
        validation = [
            bucket
            for bucket in validation
            if bucket.module_count == args.module_count
        ]
    if args.limit_buckets is not None:
        validation = validation[: args.limit_buckets]
    if not validation:
        raise SchemaValidationError("comparison has no validation buckets")

    schedule_hash = order9_schedule_hash(config)
    checkpoints = {}
    for label, raw_path in (
        ("initializer", args.baseline_checkpoint),
        (candidate_label, args.candidate_checkpoint),
    ):
        path = _resolve(raw_path, repository)
        loaded = load_order9_policy_checkpoint(
            path,
            device="cpu",
            expected_family=Order9PolicyFamily.PI_L,
            expected_schedule_hash=schedule_hash,
        )
        if loaded.metadata.physical_model_hash != manifest.physical_model_hash:
            raise SchemaValidationError(
                f"comparison checkpoint physical model differs: {label}"
            )
        checkpoints[label] = {
            "path": path,
            "sha256": loaded.sha256,
            "ppo_update_index": loaded.metadata.metadata.get("ppo_update_index"),
        }
        del loaded

    if reusable_report is not None:
        _validate_reusable_baseline_report(
            reusable_report,
            config_sha256=hash_file(config_path),
            bucket_manifest_sha256=hash_file(manifest_path),
            baseline_checkpoint_sha256=str(checkpoints["initializer"]["sha256"]),
            bucket_ids=[bucket.bucket_id for bucket in validation],
            environment_count=args.num_envs,
            rollout_steps=args.rollout_steps,
            c3_action_contract=args.c3_action_contract,
        )

    started = time.perf_counter()
    bucket_rows = []
    for ordinal, bucket in enumerate(validation):
        reset_bank_root = (
            reusable_baseline if reusable_baseline is not None else output
        )
        reset_bank = (
            reset_bank_root
            / "phase_reset_banks"
            / f"{bucket.bucket_id}.pt"
        )
        artifacts: dict[str, Order9TensorRolloutArtifact] = {}
        results: dict[str, Mapping[str, Any]] = {}
        pending_commands: dict[str, list[str]] = {}
        pending_logs: dict[str, Path] = {}
        paths: dict[str, tuple[Path, Path, str]] = {}
        for label, checkpoint in checkpoints.items():
            artifact_root = (
                reusable_baseline
                if label == "initializer" and reusable_baseline is not None
                else output
            )
            raw = (
                artifact_root
                / "rollouts"
                / label
                / f"{bucket.bucket_id}.pt"
            )
            log = (
                artifact_root
                / "logs"
                / label
                / f"{bucket.bucket_id}.log"
            )
            generation_id = (
                f"fixed_validation_compare:{label}:{bucket.bucket_id}"
            )
            paths[label] = (raw, log, generation_id)
            if label == "initializer" and reusable_baseline is not None:
                if not raw.is_file() or not log.is_file() or not reset_bank.is_file():
                    raise FileNotFoundError(
                        "reusable baseline lacks rollout/log/reset bank for "
                        f"{bucket.bucket_id}"
                    )
                continue
            if raw.is_file() and log.is_file():
                continue
            command = order9_pi_l_collector_command(
                python_executable=args.python_executable,
                repository_root=repository,
                config_path=config_path,
                stage_id=args.stage,
                parent_checkpoint_path=checkpoint["path"],
                parent_checkpoint_sha256=str(checkpoint["sha256"]),
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
                    str(args.num_envs),
                    "--rollout-steps",
                    str(args.rollout_steps),
                    "--diagnostic-deterministic-policy",
                    "--no-tensorboard",
                ]
            )
            pending_commands[label] = command
            pending_logs[label] = log
        if len(pending_commands) == 2:
            if reset_bank.is_file():
                run_parallel_order9_collectors(
                    pending_commands,
                    repository_root=repository,
                    log_paths=pending_logs,
                )
            else:
                # The first process creates the immutable reset bank with the
                # frozen initializer; the second then consumes those exact
                # physical states.  Avoid two writers racing the same sidecar.
                for label in ("initializer", candidate_label):
                    run_logged_order9_command(
                        pending_commands[label],
                        repository_root=repository,
                        log_path=pending_logs[label],
                    )
        elif len(pending_commands) == 1:
            label = next(iter(pending_commands))
            run_logged_order9_command(
                pending_commands[label],
                repository_root=repository,
                log_path=pending_logs[label],
            )

        for label, checkpoint in checkpoints.items():
            raw, log, generation_id = paths[label]
            payload = load_order9_rollout_result(log)
            artifact = load_order9_tensor_rollout_artifact(
                raw, expected_sha256=str(payload["raw_artifact_sha256"])
            )
            _validate_comparison_rollout(
                artifact,
                payload=payload,
                checkpoint_sha256=str(checkpoint["sha256"]),
                generation_id=generation_id,
                bucket_id=bucket.bucket_id,
                environment_count=args.num_envs,
                rollout_steps=args.rollout_steps,
                reset_bank_sha256=hash_file(reset_bank),
                c3_action_contract=args.c3_action_contract,
            )
            artifacts[label] = artifact
            results[label] = payload

        baseline_metrics = _rollout_metrics(artifacts["initializer"])
        candidate_metrics = _rollout_metrics(artifacts[candidate_label])
        bucket_rows.append(
            {
                "bucket_id": bucket.bucket_id,
                "sample_index": bucket.sample_index,
                "seed": bucket.seed,
                "module_count": bucket.module_count,
                "structural_hash": bucket.structural_hash,
                "initializer": baseline_metrics,
                candidate_label: candidate_metrics,
                f"delta_{candidate_label}_minus_initializer": _metric_delta(
                    candidate_metrics, baseline_metrics
                ),
                "rollout_sha256": {
                    label: str(results[label]["raw_artifact_sha256"])
                    for label in results
                },
            }
        )
        _write_json_atomic(
            output / "progress.json",
            {
                "comparison_version": (
                    ORDER9_PI_L_FIXED_CHECKPOINT_COMPARISON_VERSION
                ),
                "completed_bucket_count": ordinal + 1,
                "total_bucket_count": len(validation),
                "last_bucket_id": bucket.bucket_id,
            },
        )
        del artifacts
        gc.collect()

    aggregate = {
        label: _aggregate_metrics([row[label] for row in bucket_rows])
        for label in checkpoints
    }
    delta = _metric_delta(
        aggregate[candidate_label], aggregate["initializer"]
    )
    summary = {
        "bucket_count": len(bucket_rows),
        "module_counts": sorted({row["module_count"] for row in bucket_rows}),
        "environment_count_per_run": args.num_envs,
        "rollout_steps_per_environment": args.rollout_steps,
        "paired_run_count": 2 * len(bucket_rows),
        "deterministic_policy": True,
        "all_phase_reset_distribution": True,
        "promotion_eligible": False,
        "initializer_reward_mean": aggregate["initializer"]["reward_mean"],
        f"{candidate_label}_reward_mean": aggregate[candidate_label]["reward_mean"],
        "reward_mean_delta": delta["reward_mean"],
        "initializer_qp_feasible_rate": aggregate["initializer"][
            "qp_feasible_rate"
        ],
        f"{candidate_label}_qp_feasible_rate": aggregate[candidate_label][
            "qp_feasible_rate"
        ],
        "qp_feasible_rate_delta": delta["qp_feasible_rate"],
        "initializer_collision_rate": aggregate["initializer"][
            "collision_rate"
        ],
        f"{candidate_label}_collision_rate": aggregate[candidate_label][
            "collision_rate"
        ],
        "collision_rate_delta": delta["collision_rate"],
        "wall_elapsed_s": time.perf_counter() - started,
    }
    report = {
        "comparison_version": ORDER9_PI_L_FIXED_CHECKPOINT_COMPARISON_VERSION,
        "candidate_label": candidate_label,
        "stage_id": args.stage,
        "config_path": str(config_path),
        "config_sha256": hash_file(config_path),
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": hash_file(manifest_path),
        "checkpoint": {
            label: {
                "path": str(value["path"]),
                "sha256": value["sha256"],
                "ppo_update_index": value["ppo_update_index"],
            }
            for label, value in checkpoints.items()
        },
        "evaluation_contract": {
            "split": DatasetSplit.VALIDATION.value,
            "paired_bucket_seed_and_physics": True,
            "deterministic_policy_mean_action": True,
            "phase_initialization": "round_robin_all_eight_runtime_phases",
            "runtime_override_used": True,
            "training_performed": False,
            "promotion_eligible": False,
            "reused_initializer_rollouts": reusable_baseline is not None,
            "reused_initializer_evaluation_dir": (
                str(reusable_baseline)
                if reusable_baseline is not None
                else None
            ),
            "c3_action_contract": args.c3_action_contract,
        },
        "aggregate": aggregate,
        f"delta_{candidate_label}_minus_initializer": delta,
        "buckets": bucket_rows,
        "summary": summary,
    }
    _write_json_atomic(report_path, report)
    print("ORDER9_PI_L_COMPARISON=" + json.dumps(summary, sort_keys=True))
    print(f"report: {report_path}")
    return 0


def _validate_reusable_baseline_report(
    report: Mapping[str, Any],
    *,
    config_sha256: str,
    bucket_manifest_sha256: str,
    baseline_checkpoint_sha256: str,
    bucket_ids: list[str],
    environment_count: int,
    rollout_steps: int,
    c3_action_contract: str | None,
) -> None:
    """Reject baseline reuse when any reward or rollout contract differs."""

    checkpoint = report.get("checkpoint")
    initializer = (
        checkpoint.get("initializer")
        if isinstance(checkpoint, Mapping)
        else None
    )
    summary = report.get("summary")
    rows = report.get("buckets")
    evaluation_contract = report.get("evaluation_contract")
    expected = {
        "config_sha256": config_sha256,
        "bucket_manifest_sha256": bucket_manifest_sha256,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise SchemaValidationError(
                f"reusable baseline differs at {name}"
            )
    if (
        not isinstance(initializer, Mapping)
        or initializer.get("sha256") != baseline_checkpoint_sha256
    ):
        raise SchemaValidationError(
            "reusable baseline checkpoint differs from requested initializer"
        )
    if (
        not isinstance(summary, Mapping)
        or int(summary.get("environment_count_per_run", -1))
        != environment_count
        or int(summary.get("rollout_steps_per_environment", -1))
        != rollout_steps
    ):
        raise SchemaValidationError(
            "reusable baseline rollout dimensions differ"
        )
    if (
        not isinstance(evaluation_contract, Mapping)
        or evaluation_contract.get("c3_action_contract") != c3_action_contract
    ):
        raise SchemaValidationError(
            "reusable baseline C3 action contract differs"
        )
    reusable_bucket_ids = (
        [str(row.get("bucket_id")) for row in rows]
        if isinstance(rows, list)
        and all(isinstance(row, Mapping) for row in rows)
        else []
    )
    if (
        len(reusable_bucket_ids) != len(set(reusable_bucket_ids))
        or any(bucket_id not in reusable_bucket_ids for bucket_id in bucket_ids)
    ):
        raise SchemaValidationError(
            "reusable baseline lacks requested validation bucket subset"
        )


def _validate_comparison_rollout(
    artifact: Order9TensorRolloutArtifact,
    *,
    payload: Mapping[str, Any],
    checkpoint_sha256: str,
    generation_id: str,
    bucket_id: str,
    environment_count: int,
    rollout_steps: int,
    reset_bank_sha256: str,
    c3_action_contract: str | None,
) -> None:
    artifact.validate()
    expected_payload = {
        "passed": True,
        "finite_state": True,
        "generation_id": generation_id,
        "environment_count": environment_count,
        "rollout_steps": rollout_steps,
        "runtime_override_used": True,
        "deterministic_policy": True,
        "initial_phase_zero": False,
        "phase_specific_resets_available": True,
        "phase_reset_stratum_count": (
            ORDER9_C3_FIXED_VALIDATION_PHASE_RESET_STRATUM_COUNT
        ),
        "diagnostic_deterministic_all_phases": True,
    }
    for name, value in expected_payload.items():
        if payload.get(name) != value:
            raise SchemaValidationError(
                f"comparison rollout differs at {name}: {bucket_id}"
            )
    metadata = artifact.metadata
    expected_metadata = {
        "generation_id": generation_id,
        "pi_l_checkpoint_sha256": checkpoint_sha256,
        "runtime_override_used": True,
        "deterministic_policy": True,
        "initial_phase_zero": False,
        "phase_specific_resets_available": True,
        "phase_reset_stratum_count": (
            ORDER9_C3_FIXED_VALIDATION_PHASE_RESET_STRATUM_COUNT
        ),
        "phase_progress_semantics": "exact_policy_actor_input",
        "diagnostic_deterministic_all_phases": True,
        "c3_reset_stabilization_checkpoint_sha256": None,
        "c3_reset_bank_sha256": reset_bank_sha256,
        "c3_action_contract": c3_action_contract,
    }
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise SchemaValidationError(
                f"comparison artifact differs at {name}: {bucket_id}"
            )
    if artifact.artifact_version not in {
        ORDER9_CONTACT_SPACE_TENSOR_ROLLOUT_ARTIFACT_VERSION,
        ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
        ORDER9_MORPHOLOGY_INVARIANT_TENSOR_ROLLOUT_ARTIFACT_VERSION,
    }:
        raise SchemaValidationError("comparison rollout artifact version differs")
    stabilization = metadata.get("c3_contact_reset_stabilization")
    if not isinstance(stabilization, dict) or stabilization.get("passed") is not True:
        raise SchemaValidationError("comparison rollout lacks stabilized contact resets")
    task_bucket_ids = {
        task["metadata"].get("order9_rollout_bucket_id")
        for task in metadata["task_specs"]
    }
    if task_bucket_ids != {bucket_id}:
        raise SchemaValidationError("comparison rollout bucket identity differs")
    observed = set(
        int(value)
        for value in artifact.tensors["phase_index"][
            artifact.tensors["valid"]
        ].unique()
    )
    if observed != set(ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME):
        raise SchemaValidationError("comparison rollout lacks all task phases")


def _rollout_metrics(artifact: Order9TensorRolloutArtifact) -> dict[str, Any]:
    tensors = artifact.tensors
    valid = tensors["valid"]
    count = int(valid.sum())
    reward_names = tuple(str(value) for value in artifact.metadata["reward_term_names"])
    actor_labels = tuple(str(value) for value in artifact.metadata["actor_phase_labels"])
    metrics: dict[str, Any] = {
        "sample_count": count,
        "reward_mean": float(tensors["reward"][valid].mean()),
        "phase_success_rate": float(tensors["phase_success"][valid].float().mean()),
        "terminal_rate": float(tensors["terminal"][valid].float().mean()),
        "successful_terminal_count": int(
            (tensors["terminal"] & tensors["phase_success"] & valid).sum()
        ),
        "qp_feasible_rate": float(tensors["qp_feasible"][valid].float().mean()),
        "collision_rate": float(
            tensors["prohibited_collision"][valid].float().mean()
        ),
        "reward_terms": {
            name: float(tensors["reward_terms"][..., index][valid].mean())
            for index, name in enumerate(reward_names)
        },
        "phases": {},
    }
    for phase in sorted(
        int(value) for value in tensors["phase_index"][valid].unique()
    ):
        mask = valid & (tensors["phase_index"] == phase)
        metrics["phases"][actor_labels[phase]] = {
            "sample_count": int(mask.sum()),
            "reward_mean": float(tensors["reward"][mask].mean()),
            "phase_success_rate": float(
                tensors["phase_success"][mask].float().mean()
            ),
            "terminal_rate": float(tensors["terminal"][mask].float().mean()),
            "qp_feasible_rate": float(
                tensors["qp_feasible"][mask].float().mean()
            ),
            "collision_rate": float(
                tensors["prohibited_collision"][mask].float().mean()
            ),
        }
    return metrics


def _aggregate_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("comparison aggregate is empty")
    count = sum(int(row["sample_count"]) for row in rows)

    def weighted(name: str) -> float:
        return sum(
            float(row[name]) * int(row["sample_count"]) for row in rows
        ) / count

    reward_names = tuple(rows[0]["reward_terms"])
    phases = sorted({name for row in rows for name in row["phases"]})
    result = {
        "sample_count": count,
        "reward_mean": weighted("reward_mean"),
        "phase_success_rate": weighted("phase_success_rate"),
        "terminal_rate": weighted("terminal_rate"),
        "successful_terminal_count": sum(
            int(row["successful_terminal_count"]) for row in rows
        ),
        "qp_feasible_rate": weighted("qp_feasible_rate"),
        "collision_rate": weighted("collision_rate"),
        "reward_terms": {
            name: sum(
                float(row["reward_terms"][name]) * int(row["sample_count"])
                for row in rows
            )
            / count
            for name in reward_names
        },
        "phases": {},
    }
    for phase in phases:
        phase_rows = [row["phases"][phase] for row in rows if phase in row["phases"]]
        phase_count = sum(int(row["sample_count"]) for row in phase_rows)
        result["phases"][phase] = {
            "sample_count": phase_count,
            **{
                name: sum(
                    float(row[name]) * int(row["sample_count"])
                    for row in phase_rows
                )
                / phase_count
                for name in (
                    "reward_mean",
                    "phase_success_rate",
                    "terminal_rate",
                    "qp_feasible_rate",
                    "collision_rate",
                )
            },
        }
    return result


def _metric_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    scalar_names = (
        "reward_mean",
        "phase_success_rate",
        "terminal_rate",
        "successful_terminal_count",
        "qp_feasible_rate",
        "collision_rate",
    )
    return {
        **{
            name: float(candidate[name]) - float(baseline[name])
            for name in scalar_names
        },
        "reward_terms": {
            name: float(candidate["reward_terms"][name])
            - float(baseline["reward_terms"][name])
            for name in baseline["reward_terms"]
        },
        "phases": {
            phase: {
                name: float(candidate["phases"][phase][name])
                - float(baseline["phases"][phase][name])
                for name in (
                    "reward_mean",
                    "phase_success_rate",
                    "terminal_rate",
                    "qp_feasible_rate",
                    "collision_rate",
                )
            }
            for phase in baseline["phases"]
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
