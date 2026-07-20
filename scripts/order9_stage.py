#!/usr/bin/env python3
from __future__ import annotations

"""Inspect, preflight, evaluate, and finalize hash-bound Order 9 stages."""

import argparse
import json
from pathlib import Path

from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_evaluation import (
    Order9EvaluationEpisode,
    build_order9_stage_evaluation_report,
    write_order9_stage_evaluation_report,
)
from amsrr.training.order9_pipeline import (
    finalize_order9_stage_from_evaluation,
    load_order9_stage_manifest,
    order9_schedule_hash,
    order9_stage_by_id,
    preflight_order9_stage,
)
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="Print the validated curriculum stages.")

    preflight = commands.add_parser(
        "preflight", help="Create one fail-closed prepared-stage manifest."
    )
    preflight.add_argument("--stage", required=True)
    preflight.add_argument("--input", action="append", default=[])
    preflight.add_argument("--prior-stage-manifest", action="append", default=[])
    preflight.add_argument("--dataset")
    preflight.add_argument("--behavior-checkpoint", action="append", default=[])
    preflight.add_argument("--output", required=True)

    evaluation = commands.add_parser(
        "build-evaluation",
        help="Build aggregate promotion evidence from typed episode JSONL rows.",
    )
    evaluation.add_argument("--stage", required=True)
    evaluation.add_argument("--episode-jsonl", action="append", required=True)
    evaluation.add_argument("--checkpoint", action="append", default=[])
    evaluation.add_argument("--training-env-steps", type=int, default=0)
    evaluation.add_argument("--training-wall-time-s", type=float, default=0.0)
    evaluation.add_argument("--output", required=True)

    finalize = commands.add_parser(
        "finalize",
        help="Promote/reject a stage from a verified evaluation report.",
    )
    finalize.add_argument("--active-manifest", required=True)
    finalize.add_argument("--evaluation", required=True)
    finalize.add_argument("--checkpoint", action="append", default=[])
    finalize.add_argument("--artifact", action="append", default=[])
    finalize.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_order9_learning_config(args.config)
    config.validate()
    if args.command == "list":
        for stage in config.curriculum.stages:
            print(
                f"{stage.stage_index:02d} {stage.stage_id} "
                f"{stage.learning_mode.value} {stage.learning_target.value}"
            )
        return 0
    if args.command == "preflight":
        inputs = _key_paths(args.input, label="input")
        if not inputs:
            raise SystemExit("preflight requires at least one --input KIND=PATH")
        prior = [
            load_order9_stage_manifest(path)
            for path in args.prior_stage_manifest
        ]
        behavior = _key_paths(
            args.behavior_checkpoint, label="behavior-checkpoint"
        )
        behavior_hashes = {
            Order9PolicyFamily(family).value: hash_file(path)
            for family, path in behavior.items()
        }
        prepared, validation = preflight_order9_stage(
            config,
            stage_id=args.stage,
            input_artifact_paths=inputs,
            prior_stage_manifests=prior,
            dataset_manifest_path=args.dataset,
            behavior_checkpoint_sha256_by_family=(
                behavior_hashes if behavior_hashes else None
            ),
            output_path=args.output,
        )
        print(f"run_id: {prepared.run_id}")
        print(f"status: {prepared.status.value}")
        if validation is not None:
            print(f"dataset_valid: {str(validation.valid).lower()}")
        print(f"manifest: {args.output}")
        return 0
    if args.command == "build-evaluation":
        stage = order9_stage_by_id(config, args.stage)
        episodes = _episode_rows(args.episode_jsonl)
        checkpoints = _key_paths(args.checkpoint, label="checkpoint")
        checkpoint_hashes = {
            Order9PolicyFamily(family): hash_file(path)
            for family, path in checkpoints.items()
        }
        report = build_order9_stage_evaluation_report(
            stage=stage,
            schedule_hash=order9_schedule_hash(config),
            episodes=episodes,
            policy_checkpoint_sha256_by_family=checkpoint_hashes,
            training_rollout_environment_step_count=args.training_env_steps,
            training_rollout_wall_elapsed_s=args.training_wall_time_s,
            metadata={
                "episode_jsonl_paths": [str(path) for path in args.episode_jsonl],
                "episode_jsonl_sha256": {
                    str(path): hash_file(path) for path in args.episode_jsonl
                },
            },
        )
        write_order9_stage_evaluation_report(args.output, report)
        metrics = report.stage_metrics()
        print(f"episodes: {metrics.episode_count}")
        print(f"success_rate: {metrics.success_rate:.6f}")
        print(f"fallback_rate: {metrics.fallback_rate:.6f}")
        print(f"report: {args.output}")
        return 0
    if args.command == "finalize":
        active = load_order9_stage_manifest(args.active_manifest)
        checkpoints = _key_paths(args.checkpoint, label="checkpoint")
        artifacts = _key_paths(args.artifact, label="artifact")
        result = finalize_order9_stage_from_evaluation(
            active,
            config,
            evaluation_report_path=args.evaluation,
            output_artifact_paths=artifacts,
            checkpoint_paths_by_family=checkpoints,
            output_path=args.output,
        )
        print(f"stage: {result.stage_id}")
        print(f"status: {result.status.value}")
        print(f"promoted: {str(result.promoted).lower()}")
        print(f"failed_gates: {','.join(result.promotion_failed_gates)}")
        print(f"manifest: {args.output}")
        return 0
    raise AssertionError(args.command)


def _key_paths(values: list[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, path = raw.partition("=")
        if not separator or not key or not path:
            raise SystemExit(f"--{label} values must use KEY=PATH")
        if key in result:
            raise SystemExit(f"duplicate --{label} key {key!r}")
        if not Path(path).is_file():
            raise SystemExit(f"--{label} path does not exist: {path}")
        result[key] = path
    return result


def _episode_rows(paths: list[str]) -> list[Order9EvaluationEpisode]:
    rows: list[Order9EvaluationEpisode] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise SystemExit(f"{path}:{line_number} is not a JSON object")
                rows.append(Order9EvaluationEpisode.from_dict(payload))
    if not rows:
        raise SystemExit("evaluation episode JSONL is empty")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
