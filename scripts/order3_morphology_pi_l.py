from __future__ import annotations

"""CLI for independently resumable Order-3 morphology-conditioned pi_L stages."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.training.order3_pipeline_runner import (
    DEFAULT_ORDER3_PIPELINE_CONFIG_PATH,
    Order3PipelineMode,
    Order3PipelineRunner,
)
from amsrr.schemas.order3_rollout_condition import Order3RolloutCondition


def _mode(value: str) -> Order3PipelineMode:
    try:
        return Order3PipelineMode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"mode must be one of {Order3PipelineMode.values()}"
        ) from exc


def _add_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        type=_mode,
        choices=Order3PipelineMode.values(),
        required=True,
        help="full enforces exact pool-wide coverage; smoke requires explicit paths.",
    )


def _add_checkpoint(parser: argparse.ArgumentParser, *, parent: bool = False) -> None:
    prefix = "parent-" if parent else ""
    parser.add_argument(f"--{prefix}checkpoint-path", required=True)
    parser.add_argument(f"--{prefix}checkpoint-sha256", required=True)


def _add_condition_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Expand the configured YAML curriculum into rollout conditions.",
    )
    parser.add_argument("--curriculum-stage", action="append", default=[])
    parser.add_argument("--replicates-per-stage", type=int, default=1)
    parser.add_argument("--rollout-condition-json", action="append", default=[])


def _add_visualization(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--viewer",
        choices=("kit",),
        default=None,
        help="Launch the Isaac Lab Kit viewer for a real learned-policy rollout.",
    )
    parser.add_argument(
        "--realtime-playback",
        action="store_true",
        help="Advance each Isaac step at the configured physics dt for inspection.",
    )
    parser.add_argument(
        "--keep-open-after-rollout-s",
        type=float,
        default=0.0,
        help="Keep the Kit viewer open for this many seconds after the rollout.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run staged Order-3 morphology-conditioned pi_L workflows."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_ORDER3_PIPELINE_CONFIG_PATH,
        help="Order-3 pipeline YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    pool = subparsers.add_parser("build-pool", help="Build the deterministic split pool.")
    pool.add_argument("--output-path", default=None)
    pool.add_argument("--overwrite", action="store_true")

    bc_rollouts = subparsers.add_parser(
        "bc-rollouts",
        help="Plan or execute deterministic-v2 Isaac takeoff sources for BC.",
    )
    _add_mode(bc_rollouts)
    bc_rollouts.add_argument("--pool-manifest-path", default=None)
    bc_rollouts.add_argument("--graph-path", action="append", default=[])
    bc_rollouts.add_argument("--real", action="store_true")
    bc_rollouts.add_argument("--plan-path", default=None)

    collect_bc = subparsers.add_parser(
        "collect-bc",
        help="Convert deterministic-v2 takeoff reports into a BC-only dataset.",
    )
    _add_mode(collect_bc)
    collect_bc.add_argument("--pool-manifest-path", default=None)
    collect_bc.add_argument("--report-path", action="append", default=[])
    collect_bc.add_argument("--output-dir", default=None)
    collect_bc.add_argument("--overwrite", action="store_true")

    train_bc = subparsers.add_parser("train-bc", help="Train the BC warm-start stage.")
    train_bc.add_argument("--dataset-path", default=None)
    train_bc.add_argument("--output-root", default=None)
    train_bc.add_argument("--git-revision", default=None)

    learned = subparsers.add_parser(
        "learned-rollouts",
        help="Plan or execute learned stochastic Isaac rollouts for PPO.",
    )
    _add_mode(learned)
    _add_checkpoint(learned)
    learned.add_argument("--pool-manifest-path", default=None)
    learned.add_argument("--graph-path", action="append", default=[])
    learned.add_argument("--real", action="store_true")
    learned.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=True)
    _add_condition_selection(learned)
    learned.add_argument("--external-wrench-body", type=float, nargs=6, default=[0.0] * 6)
    learned.add_argument("--disturbance-start-s", type=float, default=3.0)
    learned.add_argument("--disturbance-duration-s", type=float, default=0.0)
    learned.add_argument("--plan-path", default=None)

    learned_one = subparsers.add_parser(
        "learned-rollout-one",
        help=argparse.SUPPRESS,
    )
    _add_mode(learned_one)
    _add_checkpoint(learned_one)
    learned_one.add_argument("--graph-path", required=True)
    learned_one.add_argument("--report-path", required=True)
    learned_one.add_argument("--real", action="store_true")
    learned_one.add_argument("--stochastic", action="store_true")
    learned_one.add_argument("--rollout-condition-json", default=None)
    learned_one.add_argument("--raw-report", action="store_true")
    learned_one.add_argument("--external-wrench-body", type=float, nargs=6, default=[0.0] * 6)
    learned_one.add_argument("--disturbance-start-s", type=float, default=3.0)
    learned_one.add_argument("--disturbance-duration-s", type=float, default=0.0)
    _add_visualization(learned_one)

    collect_ppo = subparsers.add_parser(
        "collect-ppo",
        help="Convert learned online reports into a PPO-only dataset.",
    )
    _add_mode(collect_ppo)
    _add_checkpoint(collect_ppo)
    collect_ppo.add_argument("--pool-manifest-path", default=None)
    collect_ppo.add_argument("--report-path", action="append", default=[])
    collect_ppo.add_argument("--output-dir", default=None)
    collect_ppo.add_argument("--overwrite", action="store_true")

    train_ppo = subparsers.add_parser("train-ppo", help="Train PPO from online traces.")
    _add_checkpoint(train_ppo, parent=True)
    train_ppo.add_argument("--dataset-path", default=None)
    train_ppo.add_argument("--output-root", default=None)
    train_ppo.add_argument("--git-revision", default=None)
    train_ppo.add_argument("--update-index", type=int, required=True)

    ppo_cycle = subparsers.add_parser(
        "ppo-cycle",
        help="Alternate fresh real-Isaac rollouts and one PPO update per cycle.",
    )
    _add_mode(ppo_cycle)
    _add_checkpoint(ppo_cycle)
    _add_condition_selection(ppo_cycle)
    ppo_cycle.add_argument("--pool-manifest-path", default=None)
    ppo_cycle.add_argument("--graph-path", action="append", default=[])
    ppo_cycle.add_argument("--start-update-index", type=int, default=0)
    ppo_cycle.add_argument("--update-count", type=int, default=None)
    ppo_cycle.add_argument("--git-revision", default=None)
    ppo_cycle.add_argument("--real", action="store_true", required=True)

    evaluate_learned = subparsers.add_parser(
        "evaluate-learned",
        help="Plan or execute deterministic learned-policy evaluation rollouts.",
    )
    _add_mode(evaluate_learned)
    _add_checkpoint(evaluate_learned)
    _add_condition_selection(evaluate_learned)
    evaluate_learned.add_argument("--pool-manifest-path", default=None)
    evaluate_learned.add_argument("--graph-path", action="append", default=[])
    evaluate_learned.add_argument("--ood-graph-path", action="append", default=[])
    evaluate_learned.add_argument("--real", action="store_true")
    evaluate_learned.add_argument("--plan-path", default=None)
    _add_visualization(evaluate_learned)

    evaluate_baseline = subparsers.add_parser(
        "evaluate-baseline",
        help="Plan or execute paired deterministic-v2 baseline evaluation rollouts.",
    )
    _add_mode(evaluate_baseline)
    _add_condition_selection(evaluate_baseline)
    evaluate_baseline.add_argument("--pool-manifest-path", default=None)
    evaluate_baseline.add_argument("--graph-path", action="append", default=[])
    evaluate_baseline.add_argument("--ood-graph-path", action="append", default=[])
    evaluate_baseline.add_argument("--real", action="store_true")
    evaluate_baseline.add_argument("--plan-path", default=None)

    baseline_one = subparsers.add_parser("baseline-rollout-one", help=argparse.SUPPRESS)
    _add_mode(baseline_one)
    baseline_one.add_argument("--graph-path", required=True)
    baseline_one.add_argument("--report-path", required=True)
    baseline_one.add_argument("--rollout-condition-json", required=True)
    baseline_one.add_argument("--raw-report", action="store_true")
    baseline_one.add_argument("--real", action="store_true")

    evaluation_episodes = subparsers.add_parser(
        "build-evaluation-episodes",
        help="Pair learned/baseline raw reports into typed evaluation episodes.",
    )
    _add_mode(evaluation_episodes)
    evaluation_episodes.add_argument("--pool-manifest-path", default=None)
    evaluation_episodes.add_argument("--learned-report-path", action="append", required=True)
    evaluation_episodes.add_argument("--baseline-report-path", action="append", required=True)
    evaluation_episodes.add_argument("--checkpoint-sha256", required=True)
    evaluation_episodes.add_argument("--output-path", default=None)

    acceptance_artifact = subparsers.add_parser(
        "build-acceptance-artifact",
        help="Bind evaluation episodes and raw reports into acceptance metadata.",
    )
    _add_checkpoint(acceptance_artifact)
    acceptance_artifact.add_argument("--pool-manifest-path", default=None)
    acceptance_artifact.add_argument("--dataset-manifest-path", required=True)
    acceptance_artifact.add_argument("--episodes-path", required=True)
    acceptance_artifact.add_argument("--output-path", default=None)

    accept = subparsers.add_parser(
        "accept",
        help="Recompute the full Order-3 statistical acceptance gate.",
    )
    _add_mode(accept)
    _add_checkpoint(accept)
    accept.add_argument("--pool-manifest-path", default=None)
    accept.add_argument("--dataset-manifest-path", required=True)
    accept.add_argument("--episodes-path", required=True)
    accept.add_argument("--artifact-metadata-path", default=None)
    accept.add_argument("--output-path", default=None)
    return parser


def _discovered_report_paths(runner: Order3PipelineRunner, kind: str) -> list[str]:
    root = Path(runner.config.pipeline.report_dir) / kind
    return sorted(
        str(path)
        for path in root.glob("**/*.json")
        if path.is_file() and not path.name.startswith(".")
    )


def _plan_path(
    runner: Order3PipelineRunner,
    stage: str,
    requested: str | None,
) -> str:
    if requested is not None:
        return requested
    return str(Path(runner.config.pipeline.artifact_root) / "plans" / f"{stage}.json")


def _conditions_from_args(
    runner: Order3PipelineRunner,
    arguments: argparse.Namespace,
    *,
    default_to_curriculum: bool,
    default_to_evaluation: bool = False,
) -> list[Order3RolloutCondition]:
    if arguments.curriculum and arguments.rollout_condition_json:
        raise ValueError(
            "--curriculum and --rollout-condition-json are mutually exclusive"
        )
    if default_to_evaluation and not (
        arguments.curriculum or arguments.rollout_condition_json
    ):
        return runner.evaluation_conditions(
            replicates_per_cell=arguments.replicates_per_stage
        )
    if arguments.curriculum or (default_to_curriculum and not arguments.rollout_condition_json):
        return runner.curriculum_conditions(
            replicates_per_stage=arguments.replicates_per_stage,
            stage_ids=arguments.curriculum_stage,
        )
    return [
        Order3RolloutCondition.from_json(value)
        for value in arguments.rollout_condition_json
    ]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    runner = Order3PipelineRunner.from_config_path(arguments.config)
    stage = arguments.stage
    output: dict[str, Any]
    exit_code = 0

    if stage == "build-pool":
        result = runner.build_pool(
            output_path=arguments.output_path,
            overwrite=arguments.overwrite,
        )
        output = {
            "stage": stage,
            "pool_hash": result.stable_hash(),
            "entry_count": len(result.entries),
            "split_counts": result.split_counts,
        }
    elif stage == "bc-rollouts":
        plan = runner.plan_bc_rollouts(
            mode=arguments.mode,
            pool_manifest_path=arguments.pool_manifest_path,
            graph_paths=arguments.graph_path,
            real=arguments.real,
        )
        execution = runner.execute_plan(
            plan,
            plan_path=_plan_path(runner, stage, arguments.plan_path),
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "plan_path": _plan_path(runner, stage, arguments.plan_path),
            "command_count": len(plan.commands),
            "full_pool_coverage": plan.full_pool_coverage,
            **execution,
        }
    elif stage == "collect-bc":
        report_paths = list(arguments.report_path)
        if arguments.mode == Order3PipelineMode.FULL and not report_paths:
            report_paths = _discovered_report_paths(runner, "bc")
        result = runner.collect_bc_dataset(
            report_paths=report_paths,
            mode=arguments.mode,
            pool_manifest_path=arguments.pool_manifest_path,
            output_dir=arguments.output_dir,
            overwrite=arguments.overwrite,
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "manifest_path": result.manifest_path,
            "transition_counts": result.manifest.transition_counts,
            "dataset_manifest_sha256": _hash_file(result.manifest_path),
        }
    elif stage == "train-bc":
        result = runner.train_bc(
            dataset_path=arguments.dataset_path,
            output_root=arguments.output_root,
            git_revision=arguments.git_revision,
        )
        output = {
            "stage": stage,
            "checkpoint_path": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "metrics_path": result.metrics_path,
        }
    elif stage == "learned-rollouts":
        conditions = _conditions_from_args(
            runner, arguments, default_to_curriculum=True
        )
        plan = runner.plan_learned_rollouts(
            mode=arguments.mode,
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            pool_manifest_path=arguments.pool_manifest_path,
            graph_paths=arguments.graph_path,
            real=arguments.real,
            external_wrench_body=arguments.external_wrench_body,
            disturbance_start_s=arguments.disturbance_start_s,
            disturbance_duration_s=arguments.disturbance_duration_s,
            stochastic=arguments.stochastic,
            conditions=conditions,
        )
        execution = runner.execute_plan(
            plan,
            plan_path=_plan_path(runner, stage, arguments.plan_path),
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "plan_path": _plan_path(runner, stage, arguments.plan_path),
            "command_count": len(plan.commands),
            "full_pool_coverage": plan.full_pool_coverage,
            **execution,
        }
    elif stage == "learned-rollout-one":
        result = runner.run_learned_rollout_one(
            graph_path=arguments.graph_path,
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            report_path=arguments.report_path,
            real=arguments.real,
            external_wrench_body=arguments.external_wrench_body,
            disturbance_start_s=arguments.disturbance_start_s,
            disturbance_duration_s=arguments.disturbance_duration_s,
            stochastic=arguments.stochastic,
            rollout_condition=(
                None
                if arguments.rollout_condition_json is None
                else Order3RolloutCondition.from_json(
                    arguments.rollout_condition_json
                )
            ),
            raw_report=arguments.raw_report,
            viewer=arguments.viewer,
            realtime_playback=arguments.realtime_playback,
            keep_open_after_rollout_s=arguments.keep_open_after_rollout_s,
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "report_path": arguments.report_path,
            "real_isaac_passed": result.takeoff_result.real_isaac_passed,
            "report_validation_failures": result.report_validation_failures,
        }
        if arguments.real and not result.takeoff_result.real_isaac_passed:
            exit_code = 1
    elif stage == "collect-ppo":
        report_paths = list(arguments.report_path)
        if arguments.mode == Order3PipelineMode.FULL and not report_paths:
            report_paths = [
                path
                for path in _discovered_report_paths(runner, "ppo")
                if arguments.checkpoint_sha256 in Path(path).name
            ]
        result = runner.collect_ppo_dataset(
            report_paths=report_paths,
            mode=arguments.mode,
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            pool_manifest_path=arguments.pool_manifest_path,
            output_dir=arguments.output_dir,
            overwrite=arguments.overwrite,
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "manifest_path": result.manifest_path,
            "transition_counts": result.manifest.transition_counts,
            "dataset_manifest_sha256": _hash_file(result.manifest_path),
        }
    elif stage == "train-ppo":
        result = runner.train_ppo(
            dataset_path=arguments.dataset_path,
            parent_checkpoint_path=arguments.parent_checkpoint_path,
            parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
            update_index=arguments.update_index,
            output_root=arguments.output_root,
            git_revision=arguments.git_revision,
        )
        output = {
            "stage": stage,
            "checkpoint_path": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "metrics_path": result.metrics_path,
        }
    elif stage == "ppo-cycle":
        conditions = _conditions_from_args(
            runner, arguments, default_to_curriculum=True
        )
        result = runner.run_ppo_orchestration(
            mode=arguments.mode,
            initial_checkpoint_path=arguments.checkpoint_path,
            initial_checkpoint_sha256=arguments.checkpoint_sha256,
            start_update_index=arguments.start_update_index,
            update_count=arguments.update_count,
            pool_manifest_path=arguments.pool_manifest_path,
            graph_paths=arguments.graph_path,
            conditions=conditions,
            git_revision=arguments.git_revision,
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "completed_update_count": result.completed_update_count,
            "final_checkpoint_path": result.final_checkpoint_path,
            "final_checkpoint_sha256": result.final_checkpoint_sha256,
            "updates": result.updates,
        }
    elif stage == "evaluate-learned":
        conditions = _conditions_from_args(
            runner,
            arguments,
            default_to_curriculum=False,
            default_to_evaluation=True,
        )
        plan = runner.plan_learned_evaluation_rollouts(
            mode=arguments.mode,
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            pool_manifest_path=arguments.pool_manifest_path,
            graph_paths=arguments.graph_path,
            ood_graph_paths=arguments.ood_graph_path,
            conditions=conditions,
            real=arguments.real,
            viewer=arguments.viewer,
            realtime_playback=arguments.realtime_playback,
            keep_open_after_rollout_s=arguments.keep_open_after_rollout_s,
        )
        execution = runner.execute_plan(
            plan,
            plan_path=_plan_path(runner, stage, arguments.plan_path),
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "plan_path": _plan_path(runner, stage, arguments.plan_path),
            "command_count": len(plan.commands),
            "condition_hashes": plan.condition_hashes,
            **execution,
        }
    elif stage == "evaluate-baseline":
        conditions = _conditions_from_args(
            runner,
            arguments,
            default_to_curriculum=False,
            default_to_evaluation=True,
        )
        plan = runner.plan_baseline_evaluation_rollouts(
            mode=arguments.mode,
            pool_manifest_path=arguments.pool_manifest_path,
            graph_paths=arguments.graph_path,
            ood_graph_paths=arguments.ood_graph_path,
            conditions=conditions,
            real=arguments.real,
        )
        execution = runner.execute_plan(
            plan,
            plan_path=_plan_path(runner, stage, arguments.plan_path),
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "plan_path": _plan_path(runner, stage, arguments.plan_path),
            "command_count": len(plan.commands),
            "condition_hashes": plan.condition_hashes,
            **execution,
        }
    elif stage == "baseline-rollout-one":
        condition = Order3RolloutCondition.from_json(
            arguments.rollout_condition_json
        )
        result = runner.run_baseline_rollout_one(
            graph_path=arguments.graph_path,
            report_path=arguments.report_path,
            real=arguments.real,
            rollout_condition=condition,
            raw_report=arguments.raw_report,
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "report_path": arguments.report_path,
            "real_isaac_passed": result.real_isaac_passed,
            "condition_hash": condition.condition_hash,
        }
        if arguments.real and not result.real_isaac_passed:
            exit_code = 1
    elif stage == "build-evaluation-episodes":
        episodes, episodes_path = runner.build_evaluation_episodes(
            mode=arguments.mode,
            learned_report_paths=arguments.learned_report_path,
            baseline_report_paths=arguments.baseline_report_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            pool_manifest_path=arguments.pool_manifest_path,
            output_path=arguments.output_path,
        )
        output = {
            "stage": stage,
            "mode": arguments.mode.value,
            "episodes_path": episodes_path,
            "episode_count": len(episodes),
        }
    elif stage == "build-acceptance-artifact":
        result, metadata_path = runner.build_acceptance_artifact_metadata(
            pool_manifest_path=arguments.pool_manifest_path,
            dataset_manifest_path=arguments.dataset_manifest_path,
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            episodes_path=arguments.episodes_path,
            output_path=arguments.output_path,
        )
        output = {
            "stage": stage,
            "artifact_metadata_path": metadata_path,
            "evaluation_episode_set_hash": result.evaluation_episode_set_hash,
            "pool_hash": result.pool_hash,
        }
    elif stage == "accept":
        result = runner.evaluate_acceptance(
            mode=arguments.mode,
            pool_manifest_path=arguments.pool_manifest_path,
            dataset_manifest_path=arguments.dataset_manifest_path,
            checkpoint_path=arguments.checkpoint_path,
            checkpoint_sha256=arguments.checkpoint_sha256,
            episodes_path=arguments.episodes_path,
            artifact_metadata_path=arguments.artifact_metadata_path,
            output_path=arguments.output_path,
        )
        output = {
            "stage": stage,
            "completion_passed": result.completion_passed,
            "failures": result.failures,
        }
        if not result.completion_passed:
            exit_code = 1
    else:  # pragma: no cover - argparse makes this unreachable.
        raise RuntimeError(f"unsupported Order3 stage: {stage}")

    output["p4_full_completion_claim"] = False
    print(json.dumps(output, sort_keys=True))
    return exit_code


def _hash_file(path: str | Path) -> str:
    from amsrr.utils.hashing import hash_file

    return hash_file(path)


if __name__ == "__main__":
    raise SystemExit(main())
