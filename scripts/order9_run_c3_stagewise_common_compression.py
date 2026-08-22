#!/usr/bin/env python3
from __future__ import annotations

"""Resume C3 incrementally with cheap forgetting checks and new-size validation."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.policies.order9_low_level_policy import (
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
    Order9ContactSpacePhaseConditionedActorCritic,
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
    ORDER9_C3_ACTION_CONTRACTS,
    order9_c3_action_contract_global_dimension,
    order9_c3_action_contract_uses_full_policy,
)
from amsrr.training.order9_c3_stagewise_curriculum import (
    ORDER9_C3_CONTINUOUS_VALIDATION_SCHEDULE,
    ORDER9_C3_STAGEWISE_CURRICULUM_VERSION,
    ORDER9_C3_STAGEWISE_MODULE_MAXIMA,
    cleanup_order9_intermediate_raw_artifacts,
    compare_order9_c3_incremental_forgetting,
    select_order9_c3_new_module_validation_bucket_ids,
    should_run_order9_c3_continuous_validation,
    summarize_order9_c3_phase_reset_validation,
    summarize_order9_c3_stagewise_validation,
    write_order9_c3_stagewise_json,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_curriculum_lineage import (
    load_order9_stage_parent_checkpoint,
)
from amsrr.training.order9_pipeline import order9_stage_by_id
from amsrr.training.order9_pi_l_stage_runner import (
    validate_order9_pi_l_stage_runner_inputs,
)
from amsrr.utils.hashing import hash_file


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
ACTION_CONTRACT = ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=(
            "artifacts/p4_full/order9/stages/"
            "c3_pi_l_ppo_arbitrary_morphology/"
            "training_lineages/"
            "contact_space_projected_qpid_physical_feasibility_from_zero_command_v1/"
            "modules_2_3_normal_contact_quality_min2_v1/update_000003/"
            "checkpoint_update_000003.pt"
        ),
    )
    parser.add_argument(
        "--bucket-manifest",
        default=(
            "artifacts/p4_full/order9/stages/"
            "c3_pi_l_ppo_arbitrary_morphology/"
            "rollout_buckets_current_lineage_v5/"
            "manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "artifacts/p4_full/order9/stages/"
            "c3_pi_l_ppo_arbitrary_morphology/training_lineages/"
            "incremental_full_action_min2_clear300_preload_screened_from_update3_v2"
        ),
    )
    parser.add_argument(
        "--action-contract",
        choices=ORDER9_C3_ACTION_CONTRACTS,
        default=ACTION_CONTRACT,
    )
    parser.add_argument(
        "--python-executable",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--validation-episodes-per-bucket", type=int, default=4)
    parser.add_argument("--validation-rollout-steps", type=int, default=15000)
    parser.add_argument("--maximum-updates-per-stage", type=int, default=13)
    parser.add_argument(
        "--continuous-validation-interval-updates", type=int, default=3
    )
    parser.add_argument("--stage-budget-extension-updates", type=int, default=80)
    parser.add_argument("--maximum-parallel-process-count", type=int, default=2)
    parser.add_argument(
        "--validation-process-start-stagger-s", type=float, default=8.0
    )
    parser.add_argument(
        "--maximum-parallel-training-collector-count", type=int, default=2
    )
    parser.add_argument(
        "--training-collector-start-stagger-s", type=float, default=8.0
    )
    parser.add_argument("--cpu-list", default="0-7,9-31")
    parser.add_argument("--initial-maximum-module-count", type=int, default=3)
    parser.add_argument("--forgetting-reward-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--forgetting-reward-absolute-tolerance", type=float, default=0.02)
    parser.add_argument("--forgetting-phase-success-tolerance", type=float, default=0.03)
    parser.add_argument("--stop-after-maximum-module-count", type=int, default=8)
    parser.add_argument("--keep-training-raw", action="store_true")
    parser.add_argument("--keep-intermediate-validation-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.validation_episodes_per_bucket < 1
        or args.validation_rollout_steps < 1
        or args.maximum_updates_per_stage < 1
        or args.continuous_validation_interval_updates < 1
        or args.stage_budget_extension_updates < 1
        or not 1 <= args.maximum_parallel_process_count <= 4
        or not math.isfinite(args.validation_process_start_stagger_s)
        or args.validation_process_start_stagger_s < 0.0
        or not 1 <= args.maximum_parallel_training_collector_count <= 2
        or not math.isfinite(args.training_collector_start_stagger_s)
        or args.training_collector_start_stagger_s < 0.0
        or args.stop_after_maximum_module_count
        not in ORDER9_C3_STAGEWISE_MODULE_MAXIMA
        or not 2 <= args.initial_maximum_module_count < args.stop_after_maximum_module_count
        or not 0.0 <= args.forgetting_reward_relative_tolerance < 1.0
        or args.forgetting_reward_absolute_tolerance < 0.0
        or not 0.0 <= args.forgetting_phase_success_tolerance <= 1.0
    ):
        raise ValueError("invalid C3 stagewise operational budget")

    repository = REPOSITORY_ROOT
    config_path = _resolve(args.config, repository)
    initializer = _resolve(args.initial_checkpoint, repository)
    manifest_path = _resolve(args.bucket_manifest, repository)
    output_root = _resolve(args.output_root, repository)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "stagewise_runner_state.json"
    log_root = output_root / "orchestrator_logs"

    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, STAGE_ID)
    manifest = validate_order9_pi_l_stage_runner_inputs(
        config,
        stage_id=STAGE_ID,
        bucket_manifest_path=manifest_path,
        repository_root=repository,
    )
    _validate_common_contract(config, args.action_contract)
    if not initializer.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("C3 stagewise initializer or bucket manifest is missing")
    initial_loaded = load_order9_stage_parent_checkpoint(
        config,
        stage,
        initializer,
        expected_family=Order9PolicyFamily.PI_L,
        update_index=0,
    )
    initializer_update = int(
        initial_loaded.metadata.metadata.get("ppo_update_index", -1)
    )
    if initializer_update < -1:
        raise ValueError("C3 incremental parent update index is invalid")
    expected_model_types = (
        (
            Order9ContactSpacePhaseConditionedActorCritic,
            Order9CategoricalContactNormalPhaseConditionedActorCritic,
        )
        if args.action_contract == ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
        else (Order9MorphologyInvariantCompressionActorCritic,)
    )
    if type(initial_loaded.model) not in expected_model_types:
        raise ValueError(
            "C3 stagewise action contract and initializer policy differ"
        )

    immutable_state = {
        "runner_version": ORDER9_C3_STAGEWISE_CURRICULUM_VERSION,
        "stage_id": STAGE_ID,
        "action_contract": args.action_contract,
        "config_path": str(config_path),
        "config_sha256": hash_file(config_path),
        "initializer_path": str(initializer),
        "initializer_sha256": initial_loaded.sha256,
        "initializer_update_index": initializer_update,
        "initial_maximum_module_count": args.initial_maximum_module_count,
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": hash_file(manifest_path),
        "module_maxima": list(ORDER9_C3_STAGEWISE_MODULE_MAXIMA),
        "validation_episodes_per_bucket": args.validation_episodes_per_bucket,
        "validation_rollout_steps": args.validation_rollout_steps,
        "maximum_updates_per_stage": args.maximum_updates_per_stage,
        "continuous_validation_schedule": (
            ORDER9_C3_CONTINUOUS_VALIDATION_SCHEDULE
        ),
        "continuous_validation_interval_updates": (
            args.continuous_validation_interval_updates
        ),
        "forgetting_reward_relative_tolerance": (
            args.forgetting_reward_relative_tolerance
        ),
        "forgetting_reward_absolute_tolerance": (
            args.forgetting_reward_absolute_tolerance
        ),
        "forgetting_phase_success_tolerance": (
            args.forgetting_phase_success_tolerance
        ),
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for name, value in immutable_state.items():
            if name not in state and name in {
                "continuous_validation_schedule",
                "continuous_validation_interval_updates",
            }:
                state[name] = value
            elif state.get(name) != value:
                raise ValueError(f"C3 stagewise resume contract differs at {name}")
        write_order9_c3_stagewise_json(state_path, state)
    else:
        state = {
            **immutable_state,
            "status": "running",
            "current_maximum_module_count": args.initial_maximum_module_count,
            "stages": [],
        }
        write_order9_c3_stagewise_json(state_path, state)

    parent_checkpoint = initializer
    parent_checkpoint_sha256 = initial_loaded.sha256
    parent_update_index = initializer_update
    module_maxima = tuple(
        value
        for value in ORDER9_C3_STAGEWISE_MODULE_MAXIMA
        if value > args.initial_maximum_module_count
    )
    for maximum_module_count in module_maxima:
        if maximum_module_count > args.stop_after_maximum_module_count:
            break
        previous = _stage_state(state, maximum_module_count)
        if previous is not None and previous.get("passed") is True:
            parent_checkpoint = Path(str(previous["checkpoint_path"])).resolve()
            parent_checkpoint_sha256 = str(previous["checkpoint_sha256"])
            parent_update_index = int(previous["update_index"])
            _validate_checkpoint(parent_checkpoint, parent_checkpoint_sha256)
            continue

        segment_root = output_root / f"modules_2_{maximum_module_count}"
        if previous is None:
            previous = {
                "maximum_module_count": maximum_module_count,
                "status": "training",
                "passed": False,
                "parent_checkpoint_path": str(parent_checkpoint),
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "parent_update_index": parent_update_index,
                "segment_root": str(segment_root),
                "attempts": [],
            }
            state["stages"].append(previous)
        else:
            parent_checkpoint = Path(
                str(previous["parent_checkpoint_path"])
            ).resolve()
            parent_checkpoint_sha256 = str(previous["parent_checkpoint_sha256"])
            parent_update_index = int(previous["parent_update_index"])
            _validate_checkpoint(parent_checkpoint, parent_checkpoint_sha256)

        _backfill_stage_attempt_evidence(
            previous=previous,
            segment_root=segment_root,
        )

        state["current_maximum_module_count"] = maximum_module_count
        state["status"] = "running"
        write_order9_c3_stagewise_json(state_path, state)
        print(
            "ORDER9_C3_STAGEWISE_STAGE_START="
            + json.dumps(
                {
                    "maximum_module_count": maximum_module_count,
                    "parent_update_index": parent_update_index,
                    "action_contract": args.action_contract,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        while len(previous["attempts"]) < args.maximum_updates_per_stage:
            completed_results = sorted(
                segment_root.glob(
                    "update_*/training_result_update_*.json"
                )
            )
            expected_update_index = parent_update_index + 1 + len(
                previous["attempts"]
            )
            result_path = segment_root / f"update_{expected_update_index:06d}" / (
                f"training_result_update_{expected_update_index:06d}.json"
            )
            if not result_path.is_file():
                if completed_results and int(
                    completed_results[-1].parent.name.split("_")[-1]
                ) >= expected_update_index:
                    raise ValueError(
                        "C3 stagewise segment contains an unrecorded future update"
                    )
                training_command = _training_command(
                    python_executable=args.python_executable,
                    config_path=config_path,
                    initializer=parent_checkpoint,
                    manifest_path=manifest_path,
                    segment_root=segment_root,
                    maximum_module_count=maximum_module_count,
                    branch_parent_update_index=parent_update_index,
                    update_index=expected_update_index,
                    budget_extension=args.stage_budget_extension_updates,
                    maximum_parallel_collector_count=(
                        args.maximum_parallel_training_collector_count
                    ),
                    collector_start_stagger_s=(
                        args.training_collector_start_stagger_s
                    ),
                    cpu_list=args.cpu_list,
                    action_contract=args.action_contract,
                )
                if args.dry_run:
                    print(shlex.join(training_command), flush=True)
                    return 0
                _run_logged(
                    training_command,
                    repository=repository,
                    log_path=log_root
                    / f"modules_2_{maximum_module_count}_update_"
                    f"{expected_update_index:06d}_training.log",
                )

            training_result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = _resolve(str(training_result["checkpoint_path"]), repository)
            checkpoint_sha256 = str(training_result["checkpoint_sha256"])
            _validate_checkpoint(checkpoint, checkpoint_sha256)
            active_manifest = checkpoint.parent / "stage_training_complete.json"
            if not active_manifest.is_file():
                raise FileNotFoundError(active_manifest)

            generation_root = (
                segment_root
                / "generations"
                / f"generation_{expected_update_index:06d}"
            )
            reward_summary_path = generation_root / (
                "training_rollout_reward_summary.json"
            )
            if not reward_summary_path.is_file():
                _run_logged(
                    _reward_summary_command(
                        python_executable=args.python_executable,
                        dataset_manifest=generation_root / "dataset" / "manifest.json",
                        output_path=reward_summary_path,
                    ),
                    repository=repository,
                    log_path=generation_root / "logs" / "reward_summary.log",
                )
            reward_summary = json.loads(
                reward_summary_path.read_text(encoding="utf-8")
            )
            candidate_update_index = expected_update_index - 1
            candidate_checkpoint = (
                parent_checkpoint
                if candidate_update_index == parent_update_index
                else segment_root
                / f"update_{candidate_update_index:06d}"
                / f"checkpoint_update_{candidate_update_index:06d}.pt"
            )
            candidate_checkpoint_sha256 = str(
                reward_summary["evaluated_behavior_checkpoint_sha256"]
            )
            _validate_checkpoint(
                candidate_checkpoint,
                candidate_checkpoint_sha256,
            )
            candidate_active_manifest = (
                candidate_checkpoint.parent / "stage_training_complete.json"
            )
            if not candidate_active_manifest.is_file():
                raise FileNotFoundError(candidate_active_manifest)
            bucket_cycle = str(reward_summary["bucket_cycle_index"])
            baselines = previous.setdefault("metric_baseline_by_bucket_cycle", {})
            if bucket_cycle not in baselines:
                baselines[bucket_cycle] = {
                    "summary_path": str(reward_summary_path),
                    "checkpoint_sha256": candidate_checkpoint_sha256,
                    "update_index": candidate_update_index,
                }
                forgetting_payload = {
                    "contract": "order9_c3_incremental_baseline_v1",
                    "bucket_cycle_index": int(bucket_cycle),
                    "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
                    "existing_maximum_module_count": maximum_module_count - 1,
                    "baseline_only": True,
                    "passed": True,
                    "modules": [],
                }
                new_module_metric_payload = {
                    "module_count": maximum_module_count,
                    "baseline_only": True,
                    "passed": True,
                }
            else:
                baseline_summary = json.loads(
                    Path(str(baselines[bucket_cycle]["summary_path"]))
                    .read_text(encoding="utf-8")
                )
                forgetting_payload = compare_order9_c3_incremental_forgetting(
                    baseline_summary=baseline_summary,
                    candidate_summary=reward_summary,
                    existing_maximum_module_count=maximum_module_count - 1,
                    reward_relative_tolerance=(
                        args.forgetting_reward_relative_tolerance
                    ),
                    reward_absolute_tolerance=(
                        args.forgetting_reward_absolute_tolerance
                    ),
                    phase_success_absolute_tolerance=(
                        args.forgetting_phase_success_tolerance
                    ),
                )
                including_new = compare_order9_c3_incremental_forgetting(
                    baseline_summary=baseline_summary,
                    candidate_summary=reward_summary,
                    existing_maximum_module_count=maximum_module_count,
                    reward_relative_tolerance=(
                        args.forgetting_reward_relative_tolerance
                    ),
                    reward_absolute_tolerance=(
                        args.forgetting_reward_absolute_tolerance
                    ),
                    phase_success_absolute_tolerance=(
                        args.forgetting_phase_success_tolerance
                    ),
                )
                new_module_metric_payload = including_new["modules"][-1]
            forgetting_summary_path = generation_root / "forgetting_summary.json"
            write_order9_c3_stagewise_json(
                forgetting_summary_path,
                {
                    "evaluated_update_index": candidate_update_index,
                    "evaluated_checkpoint_sha256": candidate_checkpoint_sha256,
                    "reward_summary_path": str(reward_summary_path),
                    "forgetting": forgetting_payload,
                    "new_module_metrics": new_module_metric_payload,
                },
            )
            quick_validation_summary_path = (
                generation_root / "quick_validation_summary.json"
            )
            if not quick_validation_summary_path.is_file():
                quick_validation = summarize_order9_c3_phase_reset_validation(
                    dataset_manifest_path=generation_root / "dataset" / "manifest.json",
                    rollout_log_path=generation_root / "logs" / "validation.log",
                )
                write_order9_c3_stagewise_json(
                    quick_validation_summary_path, quick_validation
                )
            quick_validation_payload = json.loads(
                quick_validation_summary_path.read_text(encoding="utf-8")
            )
            print(
                "ORDER9_C3_STAGEWISE_QUICK_VALIDATION="
                + json.dumps(
                    {
                        "maximum_module_count": maximum_module_count,
                        "training_update_index": expected_update_index,
                        "evaluated_checkpoint_sha256": (
                            quick_validation_payload[
                                "evaluated_checkpoint_sha256"
                            ]
                        ),
                        "module_count": quick_validation_payload[
                            "module_count"
                        ],
                        "terminal_success_rate": quick_validation_payload[
                            "terminal_success_rate"
                        ],
                        "promotion_evidence_eligible": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            attempt_ordinal = len(previous["attempts"]) + 1
            run_continuous_validation = (
                should_run_order9_c3_continuous_validation(
                    attempt_ordinal=attempt_ordinal,
                    interval_updates=(
                        args.continuous_validation_interval_updates
                    ),
                    maximum_attempt_count=args.maximum_updates_per_stage,
                )
                and not bool(forgetting_payload.get("baseline_only", False))
                and bool(forgetting_payload["passed"])
                and bool(new_module_metric_payload["passed"])
            )
            evaluation_root = segment_root / "evaluations" / (
                f"update_{candidate_update_index:06d}_new_module_"
                f"{maximum_module_count:02d}_4ep_v1"
            )
            continuous_validation_summary_path = evaluation_root / (
                "stagewise_validation_summary.json"
            )
            continuous_validation_payload = None
            if run_continuous_validation:
                if not continuous_validation_summary_path.is_file():
                    bucket_ids = select_order9_c3_new_module_validation_bucket_ids(
                        manifest,
                        module_count=maximum_module_count,
                    )
                    validation_command = _validation_command(
                        python_executable=args.python_executable,
                        config_path=config_path,
                        manifest_path=manifest_path,
                        checkpoint=candidate_checkpoint,
                        active_manifest=candidate_active_manifest,
                        segment_root=segment_root,
                        reset_bank_root=output_root / "phase_reset_banks",
                        update_index=candidate_update_index,
                        evaluation_root=evaluation_root,
                        bucket_ids=bucket_ids,
                        episode_count=args.validation_episodes_per_bucket,
                        rollout_steps=args.validation_rollout_steps,
                        maximum_parallel_process_count=(
                            args.maximum_parallel_process_count
                        ),
                        process_start_stagger_s=(
                            args.validation_process_start_stagger_s
                        ),
                        cpu_list=args.cpu_list,
                        action_contract=args.action_contract,
                    )
                    if args.dry_run:
                        print(shlex.join(validation_command), flush=True)
                        return 0
                    _run_logged(
                        validation_command,
                        repository=repository,
                        log_path=log_root
                        / f"modules_2_{maximum_module_count}_update_"
                        f"{expected_update_index:06d}_validation.log",
                        allowed_return_codes=(0, 2),
                    )
                    validation = summarize_order9_c3_stagewise_validation(
                        manifest=manifest,
                        maximum_module_count=maximum_module_count,
                        expected_episode_count_per_bucket=(
                            args.validation_episodes_per_bucket
                        ),
                        evaluation_root=evaluation_root,
                        minimum_success_rate=float(stage.minimum_success_rate),
                        new_module_only=True,
                    )
                    write_order9_c3_stagewise_json(
                        continuous_validation_summary_path,
                        validation.to_dict(),
                    )
                continuous_validation_payload = json.loads(
                    continuous_validation_summary_path.read_text(
                        encoding="utf-8"
                    )
                )
            attempt = {
                "update_index": expected_update_index,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "evaluated_candidate_update_index": candidate_update_index,
                "evaluated_candidate_checkpoint_path": str(candidate_checkpoint),
                "evaluated_candidate_checkpoint_sha256": (
                    candidate_checkpoint_sha256
                ),
                "training_result_path": str(result_path),
                "training_health": _training_health_summary(training_result),
                "training_rollout_reward_summary_path": str(reward_summary_path),
                "training_rollout_reward_summary": reward_summary,
                "forgetting_summary_path": str(forgetting_summary_path),
                "forgetting": forgetting_payload,
                "new_module_metrics": new_module_metric_payload,
                "quick_validation_summary_path": str(
                    quick_validation_summary_path
                ),
                "quick_validation": quick_validation_payload,
                "continuous_validation_scheduled": run_continuous_validation,
                "continuous_validation_summary_path": (
                    None
                    if not run_continuous_validation
                    else str(continuous_validation_summary_path)
                ),
                "continuous_validation": continuous_validation_payload,
            }
            previous["attempts"].append(attempt)
            passed = bool(
                continuous_validation_payload is not None
                and continuous_validation_payload["passed"]
            )
            previous["status"] = (
                "passed" if passed else "training"
            )
            previous["passed"] = passed
            if previous["passed"]:
                previous["update_index"] = candidate_update_index
                previous["checkpoint_path"] = str(candidate_checkpoint)
                previous["checkpoint_sha256"] = candidate_checkpoint_sha256
            write_order9_c3_stagewise_json(state_path, state)
            if continuous_validation_payload is not None:
                print(
                    "ORDER9_C3_STAGEWISE_CONTINUOUS_VALIDATION="
                    + json.dumps(
                        {
                            "maximum_module_count": maximum_module_count,
                            "update_index": candidate_update_index,
                            "success_count": continuous_validation_payload[
                                "success_count"
                            ],
                            "episode_count": continuous_validation_payload[
                                "episode_count"
                            ],
                            "safety_failure_count": (
                                continuous_validation_payload[
                                    "safety_failure_count"
                                ]
                            ),
                            "passed": continuous_validation_payload["passed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if not args.keep_training_raw:
                cleanup_order9_intermediate_raw_artifacts(
                    roots=(
                        segment_root
                        / "generations"
                        / f"generation_{expected_update_index:06d}"
                        / "raw",
                    ),
                    allowed_ancestor=output_root,
                    evidence_path=(
                        segment_root
                        / "generations"
                        / f"generation_{expected_update_index:06d}"
                        / "raw_cleanup_manifest.json"
                    ),
                )
            if (
                run_continuous_validation
                and not args.keep_intermediate_validation_raw
            ):
                cleanup_order9_intermediate_raw_artifacts(
                    roots=(evaluation_root / "buckets",),
                    allowed_ancestor=output_root,
                    evidence_path=evaluation_root / "raw_cleanup_manifest.json",
                )
            if previous["passed"]:
                parent_checkpoint = candidate_checkpoint
                parent_checkpoint_sha256 = candidate_checkpoint_sha256
                parent_update_index = candidate_update_index
                break

        if previous.get("passed") is not True:
            previous["status"] = "method_issue"
            state["status"] = "method_issue"
            write_order9_c3_stagewise_json(state_path, state)
            print(
                "ORDER9_C3_STAGEWISE_METHOD_ISSUE="
                + json.dumps(
                    {
                        "maximum_module_count": maximum_module_count,
                        "completed_update_count": len(previous["attempts"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 2

    final_stage = _stage_state(state, args.stop_after_maximum_module_count)
    state["status"] = (
        "training_complete_pending_formal_promotion"
        if args.stop_after_maximum_module_count == 8
        else "operational_stop_reached"
    )
    if final_stage is not None and final_stage.get("passed") is True:
        state["final_checkpoint_path"] = final_stage["checkpoint_path"]
        state["final_checkpoint_sha256"] = final_stage["checkpoint_sha256"]
        state["final_update_index"] = final_stage["update_index"]
    write_order9_c3_stagewise_json(state_path, state)
    print(
        "ORDER9_C3_STAGEWISE_TRAINING_COMPLETE="
        + json.dumps(
            {
                "status": state["status"],
                "final_update_index": state.get("final_update_index"),
                "final_checkpoint_sha256": state.get("final_checkpoint_sha256"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _validate_common_contract(config, action_contract: str) -> None:
    runtime = config.production_runtime
    if not runtime.c3_contact_compression_action_adapter_enabled:
        raise ValueError("C3 coordinated-compression adapter is not active")
    if order9_c3_action_contract_uses_full_policy(action_contract):
        expected_width = (
            12
            if action_contract == ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
            else 18
        )
        if order9_c3_action_contract_global_dimension(action_contract) != expected_width:
            raise ValueError("C3 full action contract has an invalid global width")
        return
    if (
        tuple(runtime.c3_contact_compression_only_module_counts)
        != tuple(range(2, 9))
        or tuple(runtime.c3_joint_only_module_counts)
        or order9_c3_action_contract_global_dimension(action_contract) != 0
        or not config.optimization.c3_boundary_fine_tune.compression_only_actor_objective
        or not config.optimization.c3_boundary_fine_tune.contact_residual_only_actor_update
    ):
        raise ValueError("C3 common coordinated-compression contract is not active")


def _stage_state(state: dict[str, object], maximum_module_count: int):
    matches = [
        value
        for value in state["stages"]
        if int(value["maximum_module_count"]) == maximum_module_count
    ]
    if len(matches) > 1:
        raise ValueError("duplicate C3 stagewise segment state")
    return None if not matches else matches[0]


def _backfill_stage_attempt_evidence(
    *,
    previous: dict[str, object],
    segment_root: Path,
) -> None:
    """Migrate attempts written before periodic-validation scheduling."""

    for attempt in previous["attempts"]:
        update_index = int(attempt["update_index"])
        result_path = Path(str(attempt["training_result_path"]))
        training_result = json.loads(result_path.read_text(encoding="utf-8"))
        attempt.setdefault(
            "training_health", _training_health_summary(training_result)
        )
        generation_root = (
            segment_root / "generations" / f"generation_{update_index:06d}"
        )
        summary_path = generation_root / "quick_validation_summary.json"
        if not summary_path.is_file():
            summary = summarize_order9_c3_phase_reset_validation(
                dataset_manifest_path=generation_root / "dataset" / "manifest.json",
                rollout_log_path=generation_root / "logs" / "validation.log",
            )
            write_order9_c3_stagewise_json(summary_path, summary)
        attempt.setdefault("quick_validation_summary_path", str(summary_path))
        attempt.setdefault(
            "quick_validation",
            json.loads(summary_path.read_text(encoding="utf-8")),
        )
        if "validation" in attempt and "continuous_validation" not in attempt:
            attempt["continuous_validation_scheduled"] = True
            attempt["continuous_validation"] = attempt["validation"]
            attempt["continuous_validation_summary_path"] = attempt.get(
                "validation_summary_path"
            )


def _training_health_summary(training_result: dict[str, object]) -> dict[str, object]:
    update = training_result["ppo_update"]
    metadata = update["metadata"]
    return {
        "actor_loss": float(update["actor_loss"]),
        "value_loss": float(update["value_loss"]),
        "approximate_kl": float(update["approximate_kl"]),
        "maximum_phase_kl_observed": float(
            metadata["maximum_phase_kl_observed"]
        ),
        "entropy": float(update["entropy"]),
        "clipped_fraction": float(update["clipped_fraction"]),
        "early_stopped_for_kl": bool(update["early_stopped_for_kl"]),
        "optimizer_rollback_count": int(metadata["optimizer_rollback_count"]),
    }


def _training_command(
    *,
    python_executable: str,
    config_path: Path,
    initializer: Path,
    manifest_path: Path,
    segment_root: Path,
    maximum_module_count: int,
    branch_parent_update_index: int,
    update_index: int,
    budget_extension: int,
    maximum_parallel_collector_count: int,
    collector_start_stagger_s: float,
    cpu_list: str,
    action_contract: str,
) -> list[str]:
    command = [
        "taskset",
        "-c",
        cpu_list,
        python_executable,
        str(REPOSITORY_ROOT / "scripts/order9_run_pi_l_ppo_stage.py"),
        "--config",
        str(config_path),
        "--stage",
        STAGE_ID,
        "--initial-checkpoint",
        str(initializer),
        "--bucket-manifest",
        str(manifest_path),
        "--stage-root",
        str(segment_root),
        "--training-min-module-count",
        "2",
        "--training-max-module-count",
        str(maximum_module_count),
        "--c3-action-contract",
        action_contract,
        "--additional-update-count",
        str(budget_extension),
        "--stop-after-update-index",
        str(update_index),
        "--maximum-parallel-collector-process-count",
        str(maximum_parallel_collector_count),
        "--collector-process-start-stagger-s",
        str(collector_start_stagger_s),
    ]
    if branch_parent_update_index >= 0:
        command.extend(
            ["--branch-parent-update-index", str(branch_parent_update_index)]
        )
    for manifest in (
        "artifacts/p4_full/order9/stages/c0_order8_teacher_collection/stage_promoted.json",
        "artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/stage_promoted.json",
        "artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/"
        "evaluations/extension_update_000049_final_200/stage_promoted.json",
    ):
        command.extend(["--prior-stage-manifest", manifest])
    return command


def _reward_summary_command(
    *,
    python_executable: str,
    dataset_manifest: Path,
    output_path: Path,
) -> list[str]:
    return [
        python_executable,
        str(REPOSITORY_ROOT / "scripts/order9_summarize_c3_generation_rewards.py"),
        "--dataset-manifest",
        str(dataset_manifest),
        "--output",
        str(output_path),
    ]


def _validation_command(
    *,
    python_executable: str,
    config_path: Path,
    manifest_path: Path,
    checkpoint: Path,
    active_manifest: Path,
    segment_root: Path,
    reset_bank_root: Path,
    update_index: int,
    evaluation_root: Path,
    bucket_ids: tuple[str, ...],
    episode_count: int,
    rollout_steps: int,
    maximum_parallel_process_count: int,
    process_start_stagger_s: float,
    cpu_list: str,
    action_contract: str,
) -> list[str]:
    command = [
        python_executable,
        str(REPOSITORY_ROOT / "scripts/order9_run_c3_promotion.py"),
        "--config",
        str(config_path),
        "--bucket-manifest",
        str(manifest_path),
        "--checkpoint",
        str(checkpoint),
        "--active-manifest",
        str(active_manifest),
        "--training-lineage-root",
        str(segment_root),
        "--reset-bank-root",
        str(reset_bank_root),
        "--through-update",
        str(update_index),
        "--output-dir",
        str(evaluation_root),
        "--episode-count-per-bucket",
        str(episode_count),
        "--rollout-steps",
        str(rollout_steps),
        "--maximum-parallel-process-count",
        str(maximum_parallel_process_count),
        "--process-start-stagger-s",
        str(process_start_stagger_s),
        "--c3-action-contract",
        action_contract,
        "--disable-fail-fast",
        "--cpu-list",
        cpu_list,
    ]
    for bucket_id in bucket_ids:
        command.extend(["--bucket-id", bucket_id])
    return command


def _run_logged(
    command: list[str],
    *,
    repository: Path,
    log_path: Path,
    allowed_return_codes: tuple[int, ...] = (0,),
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("ORDER9_C3_STAGEWISE_COMMAND=" + shlex.join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "4"),
                "OPENBLAS_NUM_THREADS": os.environ.get(
                    "OPENBLAS_NUM_THREADS", "4"
                ),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "4"),
                "PYTHONFAULTHANDLER": "1",
            },
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("ORDER9_"):
                    print(line, end="", flush=True)
            return_code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGINT)
            process.wait()
            raise
    if return_code not in allowed_return_codes:
        raise RuntimeError(
            f"C3 stagewise subprocess failed with {return_code}; log={log_path}"
        )


def _validate_checkpoint(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or hash_file(path) != expected_sha256:
        raise ValueError(f"C3 stagewise checkpoint binding differs: {path}")


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
