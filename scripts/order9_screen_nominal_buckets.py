#!/usr/bin/env python3
from __future__ import annotations

"""Resumable nominal-IK + QPID-only screening for selected C3 buckets."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_pi_l_stage_runner import (
    order9_pi_l_collector_command,
    run_parallel_order9_collectors,
)
from amsrr.training.order9_c3_action_contract import ORDER9_C3_ACTION_CONTRACTS
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
)
from amsrr.utils.hashing import hash_file


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
RESULT_PREFIX = "ORDER9_ROLLOUT_JSON="


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reset-bank-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation"), required=True
    )
    parser.add_argument("--module-count", action="append", type=int, required=True)
    parser.add_argument("--bucket-id", action="append")
    parser.add_argument("--episode-count", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=3200)
    parser.add_argument("--cpu-list", default="0-31")
    parser.add_argument(
        "--maximum-parallel-process-count",
        type=int,
        choices=(1, 2),
        default=1,
        help="Run one or two independent Isaac bucket collectors concurrently.",
    )
    parser.add_argument("--diagnostic-collision-evidence", action="store_true")
    parser.add_argument(
        "--diagnostic-nominal-qpid-only",
        action="store_true",
        help=(
            "Force the nominal-QPID diagnostic while still allowing an "
            "explicit C3 action contract. Contact-space v7 checkpoints need "
            "the contract to construct their morphology-specific action basis."
        ),
    )
    parser.add_argument(
        "--formal-phase-zero-start",
        action="store_true",
        help="Start every evaluation episode from the accepted approach t0 state.",
    )
    parser.add_argument(
        "--training-only-continuous-teacher-rollout",
        action="store_true",
        help=(
            "Mark train-split continuous episodes as privileged warm-start "
            "targets; these artifacts are never promotion evidence."
        ),
    )
    parser.add_argument(
        "--learned-policy-continuous-rollout",
        action="store_true",
        help=(
            "Run the configured production pi_L path without nominal-QPID "
            "or diagnostic action ablation."
        ),
    )
    parser.add_argument(
        "--diagnostic-action-ablation",
        choices=(
            "zero_global",
            "zero_joint",
            "contact_compression_only",
            "contact_compression_plus_centroidal",
            "contact_compression_plus_global",
        ),
        help=(
            "Apply the selected diagnostic policy-action mask instead of the "
            "nominal-QPID-only baseline."
        ),
    )
    parser.add_argument(
        "--c3-action-contract",
        choices=ORDER9_C3_ACTION_CONTRACTS,
        help=(
            "Apply the explicit C3 train/evaluation action contract. This "
            "keeps the contract active continuously across release/retreat."
        ),
    )
    parser.add_argument("--initial-phase-index", type=int, choices=range(8))
    parser.add_argument(
        "--initial-phase-stratum-index", type=int, choices=range(4)
    )
    parser.add_argument(
        "--initial-phase-strata",
        help="Comma-separated PHASE:STRATUM pair for every environment.",
    )
    parser.add_argument("--collision-contact-offset-mm", type=float)
    parser.add_argument("--collision-rest-offset-mm", type=float)
    parser.add_argument(
        "--virtual-contact-lead-mm",
        type=float,
        help=(
            "Diagnostic override for the bounded inward virtual-contact "
            "lead; production defaults remain unchanged."
        ),
    )
    parser.add_argument(
        "--contact-compression-action-span-mm",
        type=float,
        help=(
            "Diagnostic override for the pi_L inward contact-compression "
            "action span."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.episode_count < 1 or args.rollout_steps < 1:
        raise ValueError("episode count and rollout steps must be positive")
    if args.training_only_continuous_teacher_rollout and args.split != "train":
        raise ValueError("continuous teacher rollout requires the train split")
    if args.diagnostic_nominal_qpid_only and any(
        (
            args.training_only_continuous_teacher_rollout,
            args.learned_policy_continuous_rollout,
            args.diagnostic_action_ablation is not None,
        )
    ):
        raise ValueError(
            "explicit nominal-QPID screening cannot be combined with a learned "
            "or training-only rollout mode"
        )
    if not args.diagnostic_nominal_qpid_only and sum(
        (
            bool(args.training_only_continuous_teacher_rollout),
            bool(args.learned_policy_continuous_rollout),
            args.diagnostic_action_ablation is not None,
            args.c3_action_contract is not None,
        )
    ) > 1:
        raise ValueError(
            "training-only, production learned-policy, and diagnostic action "
            "and explicit C3 action-contract rollout modes are mutually exclusive"
        )
    nominal_qpid_only = bool(
        args.diagnostic_nominal_qpid_only
        or (
            not args.training_only_continuous_teacher_rollout
            and not args.learned_policy_continuous_rollout
            and args.diagnostic_action_ablation is None
            and args.c3_action_contract is None
        )
    )
    if (
        args.virtual_contact_lead_mm is not None
        and (
            not math.isfinite(args.virtual_contact_lead_mm)
            or not 0.0 <= args.virtual_contact_lead_mm <= 20.0
        )
    ):
        raise ValueError("virtual contact lead must lie in [0, 20] mm")
    if (
        args.contact_compression_action_span_mm is not None
        and (
            not math.isfinite(args.contact_compression_action_span_mm)
            or not 0.0 <= args.contact_compression_action_span_mm <= 20.0
        )
    ):
        raise ValueError("contact compression action span must lie in [0, 20] mm")
    if (
        args.initial_phase_stratum_index is not None
        and args.initial_phase_index is None
    ):
        raise ValueError("initial phase stratum requires an initial phase")
    if args.initial_phase_strata is not None and any(
        value is not None
        for value in (
            args.initial_phase_index,
            args.initial_phase_stratum_index,
        )
    ):
        raise ValueError(
            "initial phase strata are mutually exclusive with scalar overrides"
        )
    module_counts = tuple(sorted(set(args.module_count)))
    if any(count < 2 or count > 8 for count in module_counts):
        raise ValueError("module counts must lie in [2, 8]")

    repository = REPOSITORY_ROOT
    manifest_path = _resolve(args.bucket_manifest, repository)
    checkpoint = _resolve(args.checkpoint, repository)
    reset_root = _resolve(args.reset_bank_root, repository)
    output = _resolve(args.output_dir, repository)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = hash_file(checkpoint)
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    buckets = tuple(
        bucket
        for bucket in manifest.buckets
        if bucket.split.value == args.split
        and bucket.module_count in module_counts
        and (args.bucket_id is None or bucket.bucket_id in args.bucket_id)
    )
    if not buckets:
        raise ValueError("nominal screening selected no buckets")

    completed: dict[str, dict[str, object]] = {}
    pending_commands: dict[str, list[str]] = {}
    pending_paths: dict[str, tuple[Path, Path, Path]] = {}
    for bucket in buckets:
        bucket_root = output / bucket.bucket_id
        raw = bucket_root / "evaluation_rollout.pt"
        episodes = bucket_root / "evaluation_episodes.jsonl"
        log = bucket_root / "evaluation.log"
        cached = _completed_evidence(
            log=log,
            raw=raw,
            episodes=episodes,
            expected_episode_count=args.episode_count,
            expected_action_ablation=args.diagnostic_action_ablation,
            expected_c3_action_contract=args.c3_action_contract,
            expected_nominal_qpid_only=nominal_qpid_only,
            expected_training_only=args.training_only_continuous_teacher_rollout,
            expected_learned_policy=args.learned_policy_continuous_rollout,
            expected_formal_phase_zero_start=args.formal_phase_zero_start,
        )
        if cached is None:
            command = order9_pi_l_collector_command(
                python_executable=args.python_executable,
                repository_root=repository,
                config_path=_resolve(args.config, repository),
                stage_id=STAGE_ID,
                parent_checkpoint_path=checkpoint,
                parent_checkpoint_sha256=checkpoint_sha256,
                c3_reset_bank_path=reset_root / f"{bucket.bucket_id}.pt",
                generation_id=f"c3_nominal_screen_v2:{bucket.bucket_id}",
                output_raw_path=raw,
                bucket=bucket,
                bucket_manifest_path=manifest_path,
            )
            command.extend(
                [
                    "--num-envs",
                    str(args.episode_count),
                    "--rollout-steps",
                    str(args.rollout_steps),
                    "--evaluation-jsonl",
                    str(episodes),
                    "--evaluation-episode-count",
                    str(args.episode_count),
                    "--no-tensorboard",
                ]
            )
            if nominal_qpid_only:
                command.append("--diagnostic-nominal-qpid-only")
            if args.diagnostic_action_ablation is not None:
                command.extend(
                    [
                        "--diagnostic-action-ablation",
                        args.diagnostic_action_ablation,
                    ]
                )
            if args.c3_action_contract is not None:
                command.extend(
                    [
                        "--c3-action-contract",
                        args.c3_action_contract,
                    ]
                )
            if args.diagnostic_collision_evidence:
                command.append("--diagnostic-collision-evidence")
            if args.formal_phase_zero_start:
                command.append("--formal-phase-zero-start")
            if args.training_only_continuous_teacher_rollout:
                command.append("--training-only-continuous-teacher-rollout")
            if args.virtual_contact_lead_mm is not None:
                command.extend(
                    [
                        "--virtual-contact-lead-mm",
                        str(args.virtual_contact_lead_mm),
                    ]
                )
            if args.contact_compression_action_span_mm is not None:
                command.extend(
                    [
                        "--contact-compression-action-span-mm",
                        str(args.contact_compression_action_span_mm),
                    ]
                )
            if args.initial_phase_index is not None:
                command.extend(
                    [
                        "--diagnostic-initial-phase-index",
                        str(args.initial_phase_index),
                    ]
                )
            if args.initial_phase_stratum_index is not None:
                command.extend(
                    [
                        "--diagnostic-initial-phase-stratum-index",
                        str(args.initial_phase_stratum_index),
                    ]
                )
            if args.initial_phase_strata is not None:
                command.extend(
                    [
                        "--diagnostic-initial-phase-strata",
                        args.initial_phase_strata,
                    ]
                )
                if (
                    not nominal_qpid_only
                    and (
                        args.diagnostic_action_ablation is not None
                        or args.c3_action_contract is not None
                    )
                ):
                    command.append("--diagnostic-learned-phase-strata")
            if args.collision_contact_offset_mm is not None:
                if args.collision_rest_offset_mm is None:
                    raise ValueError("collision rest offset is required")
                command.extend(
                    [
                        "--collision-contact-offset-m",
                        str(1.0e-3 * args.collision_contact_offset_mm),
                        "--collision-rest-offset-m",
                        str(1.0e-3 * args.collision_rest_offset_mm),
                    ]
                )
            if args.cpu_list:
                command = ["taskset", "-c", args.cpu_list, *command]
            pending_commands[bucket.bucket_id] = command
            pending_paths[bucket.bucket_id] = (log, raw, episodes)
            continue
        completed[bucket.bucket_id] = cached
        _print_progress(
            bucket_count=len(buckets),
            completed_bucket_count=len(completed),
            latest_bucket_id=bucket.bucket_id,
        )

    pending_ids = list(pending_commands)
    for start in range(0, len(pending_ids), args.maximum_parallel_process_count):
        group = pending_ids[
            start : start + args.maximum_parallel_process_count
        ]
        run_parallel_order9_collectors(
            {bucket_id: pending_commands[bucket_id] for bucket_id in group},
            repository_root=repository,
            log_paths={
                bucket_id: pending_paths[bucket_id][0] for bucket_id in group
            },
            maximum_parallel_process_count=args.maximum_parallel_process_count,
        )
        for bucket_id in group:
            log, raw, episodes = pending_paths[bucket_id]
            cached = _completed_evidence(
                log=log,
                raw=raw,
                episodes=episodes,
                expected_episode_count=args.episode_count,
                expected_action_ablation=args.diagnostic_action_ablation,
                expected_c3_action_contract=args.c3_action_contract,
                expected_nominal_qpid_only=nominal_qpid_only,
                expected_training_only=(
                    args.training_only_continuous_teacher_rollout
                ),
                expected_learned_policy=args.learned_policy_continuous_rollout,
                expected_formal_phase_zero_start=(
                    args.formal_phase_zero_start
                ),
            )
            if cached is None:
                raise RuntimeError(
                    f"nominal screening produced incomplete evidence: {bucket_id}"
                )
            completed[bucket_id] = cached
            _print_progress(
                bucket_count=len(buckets),
                completed_bucket_count=len(completed),
                latest_bucket_id=bucket_id,
            )

    summary = {
        "checkpoint_sha256": checkpoint_sha256,
        "split": args.split,
        "module_counts": list(module_counts),
        "episode_count_per_bucket": args.episode_count,
        "rollout_steps": args.rollout_steps,
        "maximum_parallel_process_count": args.maximum_parallel_process_count,
        "diagnostic_collision_evidence": bool(
            args.diagnostic_collision_evidence
        ),
        "diagnostic_nominal_qpid_only": nominal_qpid_only,
        "formal_phase_zero_start": bool(args.formal_phase_zero_start),
        "diagnostic_action_ablation": args.diagnostic_action_ablation,
        "c3_action_contract": args.c3_action_contract,
        "learned_policy_continuous_rollout": bool(
            args.learned_policy_continuous_rollout
        ),
        "initial_phase_index": args.initial_phase_index,
        "initial_phase_stratum_index": args.initial_phase_stratum_index,
        "initial_phase_strata": args.initial_phase_strata,
        "virtual_contact_lead_mm": args.virtual_contact_lead_mm,
        "contact_compression_action_span_mm": (
            args.contact_compression_action_span_mm
        ),
        "bucket_results": completed,
    }
    summary_path = output / "screening_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ORDER9_NOMINAL_SCREEN_COMPLETE="
        + json.dumps(
            {
                "bucket_count": len(completed),
                "summary_path": str(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _completed_evidence(
    *,
    log: Path,
    raw: Path,
    episodes: Path,
    expected_episode_count: int,
    expected_action_ablation: str | None,
    expected_c3_action_contract: str | None,
    expected_nominal_qpid_only: bool,
    expected_training_only: bool,
    expected_learned_policy: bool,
    expected_formal_phase_zero_start: bool,
) -> dict[str, object] | None:
    if not log.is_file() or not raw.is_file() or not episodes.is_file():
        return None
    rows = [
        json.loads(line)
        for line in episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_episode_count:
        return None
    result = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
    if result is None:
        return None
    if (
        bool(result.get("formal_phase_zero_start"))
        != expected_formal_phase_zero_start
    ):
        return None
    if expected_nominal_qpid_only:
        mode_matches = (
            result.get("diagnostic_nominal_qpid_only") is True
            and result.get("diagnostic_action_ablation") is None
            and (
                expected_c3_action_contract is None
                or result.get("c3_action_contract")
                == expected_c3_action_contract
            )
        )
    elif expected_c3_action_contract is not None:
        contract_metadata = result
        if (
            "c3_nominal_only_action_mask_contract" not in result
            or "c3_nominal_only_action_mask_phase_labels" not in result
        ):
            payload = torch.load(raw, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("metadata"), dict
            ):
                return None
            contract_metadata = payload["metadata"]
        mode_matches = (
            result.get("c3_action_contract") == expected_c3_action_contract
            and result.get("diagnostic_action_ablation") is None
            and result.get("diagnostic_nominal_qpid_only") is False
            and contract_metadata.get("c3_action_contract")
            == expected_c3_action_contract
            and contract_metadata.get("c3_nominal_only_action_mask_contract") is None
            and contract_metadata.get("c3_nominal_only_action_mask_phase_labels") == []
        )
    elif expected_training_only:
        mode_matches = (
            result.get("training_only_continuous_teacher_rollout") is True
            and result.get("diagnostic_nominal_qpid_only") is False
            and result.get("diagnostic_action_ablation") is None
        )
    elif expected_learned_policy:
        mode_matches = (
            result.get("training_only_continuous_teacher_rollout") is False
            and result.get("diagnostic_nominal_qpid_only") is False
            and result.get("diagnostic_action_ablation") is None
        )
    elif expected_action_ablation is None:
        mode_matches = result.get("diagnostic_nominal_qpid_only") is True
    else:
        mode_matches = (
            result.get("diagnostic_action_ablation")
            == expected_action_ablation
        )
    if not mode_matches:
        return None
    return {
        "episode_count": len(rows),
        "successful_episode_count": sum(bool(row["task_success"]) for row in rows),
        "failure_reason_counts": _counts(row["failure_reason"] for row in rows),
        "terminal_phase_counts": _counts(
            str(int(row["metrics"]["terminal_phase_index"])) for row in rows
        ),
        "phase_transition_counts": result["phase_transition_counts"],
        "wall_elapsed_s": result["wall_elapsed_s"],
    }


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _print_progress(
    *, bucket_count: int, completed_bucket_count: int, latest_bucket_id: str
) -> None:
    print(
        "ORDER9_NOMINAL_SCREEN_PROGRESS="
        + json.dumps(
            {
                "bucket_count": bucket_count,
                "completed_bucket_count": completed_bucket_count,
                "latest_bucket_id": latest_bucket_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
