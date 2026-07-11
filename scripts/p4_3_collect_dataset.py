from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.training.p4_3_rollout_runner import (
    P4_3RolloutRunner,
    load_p4_3_rollout_runner_config,
)
from amsrr.utils.hashing import hash_file


def _hash_existing_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    return hash_file(candidate) if candidate.is_file() else None


def _rollout_metric_sum(result, name: str) -> int:
    return int(
        sum(
            float(item.rollout_result.metrics.get(name, 0.0))
            for item in result.candidate_results
            if item.rollout_result is not None
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect deterministic Isaac rollouts for P4.3a.")
    parser.add_argument(
        "--config",
        default="configs/training/p4_3_learning_bootstrap.yaml",
    )
    parser.add_argument("--real", action="store_true", help="Run the real Isaac collection gate.")
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--task-start-index", type=int)
    parser.add_argument("--candidates-per-task", type=int)
    parser.add_argument("--candidate-offset", type=int)
    parser.add_argument("--archive-path")
    parser.add_argument("--pi-l-checkpoint")
    parser.add_argument("--online-evaluation-path", default="artifacts/p4_3/pi_l/online_rollout_evaluation.json")
    args = parser.parse_args(argv)

    config = load_p4_3_rollout_runner_config(args.config)
    config.dry_run = not args.real
    if args.task_count is not None:
        config.task_count = args.task_count
    if args.task_start_index is not None:
        config.task_start_index = args.task_start_index
    if args.candidates_per_task is not None:
        config.candidates_per_task = args.candidates_per_task
    if args.candidate_offset is not None:
        config.candidate_offset = args.candidate_offset
    if args.pi_l_checkpoint is not None:
        config.learned_pi_l_checkpoint_path = args.pi_l_checkpoint
    config.validate()
    resolved_archive_path = args.archive_path
    if resolved_archive_path is None:
        resolved_archive_path = (
            "artifacts/p4_3/pi_l/online_rollout_archive.jsonl"
            if args.pi_l_checkpoint is not None
            else config.archive_path
        )
    result = P4_3RolloutRunner(config).run(archive_path=resolved_archive_path)
    online_evaluation: dict[str, object] | None = None
    if args.pi_l_checkpoint is not None:
        learned_count = sum(
            int(item.rollout_result.metrics.get("p4_3_pi_l_learned_decision_count", 0.0))
            for item in result.candidate_results
            if item.rollout_result is not None
        )
        fallback_count = sum(
            int(item.rollout_result.metrics.get("p4_3_pi_l_fallback_count", 0.0))
            for item in result.candidate_results
            if item.rollout_result is not None
        )
        rollout_count = len(result.candidate_results)
        rollout_passed_count = int(result.metrics.get("success_count", 0.0))
        qp_terminal_count = _rollout_metric_sum(
            result, "controller_qp_infeasible_terminal"
        )
        hard_collision_count = _rollout_metric_sum(result, "hard_collision")
        object_drop_count = _rollout_metric_sum(result, "object_drop")
        checkpoint_load_failed_count = _rollout_metric_sum(
            result, "p4_3_pi_l_checkpoint_load_failed"
        )
        overlay_nonzero_count = _rollout_metric_sum(
            result, "p4_3_pi_l_overlay_nonzero_count"
        )
        overlay_delta_norm_sum = sum(
            float(item.rollout_result.metrics.get("p4_3_pi_l_overlay_delta_norm_sum", 0.0))
            for item in result.candidate_results
            if item.rollout_result is not None
        )
        overlay_delta_norm_max = max(
            (
                float(item.rollout_result.metrics.get("p4_3_pi_l_overlay_delta_norm_max", 0.0))
                for item in result.candidate_results
                if item.rollout_result is not None
            ),
            default=0.0,
        )
        blend_factors = {
            float(item.rollout_result.metrics.get("p4_3_pi_l_runtime_blend_factor", 0.0))
            for item in result.candidate_results
            if item.rollout_result is not None
        }
        checkpoint_loaded = all(
            item.rollout_result is not None
            and item.rollout_result.metrics.get("p4_3_pi_l_checkpoint_loaded", 0.0) > 0.5
            for item in result.candidate_results
        )
        isaac_backed = (
            not result.dry_run
            and rollout_count > 0
            and result.metrics.get("isaac_backed_count", 0.0) == rollout_count
        )
        all_rollouts_passed = rollout_count > 0 and rollout_passed_count == rollout_count
        checkpoint_sha256 = _hash_existing_file(args.pi_l_checkpoint)
        archive_sha256 = _hash_existing_file(resolved_archive_path)
        safety_violation_count = (
            qp_terminal_count + hard_collision_count + object_drop_count
        )
        online_evaluation = {
            "schema_version": "p4_3_pi_l_online_evaluation_v1",
            "evaluation_type": "learned_pi_l_online_isaac_rollout",
            "source_is_real_isaac": isaac_backed,
            "isaac_backed": isaac_backed,
            "checkpoint_path": str(args.pi_l_checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_load_failed_count": checkpoint_load_failed_count,
            "learned_decision_count": learned_count,
            "fallback_count": fallback_count,
            "overlay_nonzero_count": overlay_nonzero_count,
            "overlay_delta_norm_sum": overlay_delta_norm_sum,
            "overlay_delta_norm_max": overlay_delta_norm_max,
            "runtime_blend_factor": (
                next(iter(blend_factors)) if len(blend_factors) == 1 else None
            ),
            "rollout_passed_count": rollout_passed_count,
            "rollout_count": rollout_count,
            "task_ids": sorted({archive.task_spec.task_id for archive in result.archives}),
            "all_rollouts_passed": all_rollouts_passed,
            "controller_qp_infeasible_terminal_count": qp_terminal_count,
            "hard_collision_count": hard_collision_count,
            "object_drop_count": object_drop_count,
            "safety_violation_count": safety_violation_count,
            "controller_qp_safety_layer_used": True,
            "controller_authority_preserved": True,
            "controller_active_knot_preserved": True,
            "learned_policy_command_fields": [
                "desired_body_twist",
                "desired_body_position",
                "residual_wrench_body",
            ],
            "nonlearned_command_fields_source": "p4_2_deterministic_command",
            "deterministic_fallback_available": True,
            "learned_policy_deployed_in_isaac": checkpoint_loaded and learned_count > 0,
            "p4_full_completion_claim": False,
            "natural_contact_success_claim": False,
            "archive_path": str(resolved_archive_path),
            "archive_sha256": archive_sha256,
        }
        evaluation_path = Path(args.online_evaluation_path)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(json.dumps(online_evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dry_run": result.dry_run,
                "metrics": result.metrics,
                "episode_ids": [archive.episode_id for archive in result.archives],
                "candidate_results": [
                    {
                        "task_index": item.task_index,
                        "candidate_id": item.candidate_id,
                        "variant": item.variant,
                        "isaac_backed": bool(
                            item.rollout_result is not None and item.rollout_result.isaac_backed
                        ),
                        "passed": bool(
                            item.rollout_result is not None and item.rollout_result.passed
                        ),
                    }
                    for item in result.candidate_results
                ],
            },
            sort_keys=True,
        )
    )
    if not args.real:
        return 0
    expected = config.task_count * config.candidates_per_task
    real_count = int(result.metrics.get("isaac_backed_count", 0.0))
    collection_complete = len(result.archives) == expected and real_count == expected
    if online_evaluation is None:
        return 0 if collection_complete else 1
    online_gate = (
        collection_complete
        and bool(online_evaluation["checkpoint_loaded"])
        and int(online_evaluation["learned_decision_count"]) > 0
        and int(online_evaluation["overlay_nonzero_count"]) > 0
        and bool(online_evaluation["all_rollouts_passed"])
        and int(online_evaluation["safety_violation_count"]) == 0
    )
    return 0 if online_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
