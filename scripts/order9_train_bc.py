#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import train_order9_behavior_cloning
from amsrr.training.order9_pipeline import (
    load_order9_stage_manifest,
    preflight_order9_stage,
    record_order9_stage_training_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one hash-bound Order 9 behavior-cloning stage."
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--source-order3-checkpoint")
    parser.add_argument("--prior-stage-manifest", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--git-revision")
    args = parser.parse_args()

    config = load_order9_learning_config(args.config)
    output = Path(
        args.output_dir
        or Path(config.production_runtime.artifact_root) / "stages" / args.stage
    )
    output.mkdir(parents=True, exist_ok=True)
    physical_model_path = config.production_runtime.robot_model_config_path
    physical_model = build_physical_model_from_config(physical_model_path)
    prior = [
        load_order9_stage_manifest(path) for path in args.prior_stage_manifest
    ]
    inputs: dict[str, str | Path] = {
        "curriculum_config": args.config,
        "robot_model_config": physical_model_path,
    }
    if args.parent_checkpoint:
        inputs["parent_checkpoint"] = args.parent_checkpoint
    if args.source_order3_checkpoint:
        inputs["source_order3_checkpoint"] = args.source_order3_checkpoint
    prepared_path = output / "stage_prepared.json"
    prepared, _ = preflight_order9_stage(
        config,
        stage_id=args.stage,
        input_artifact_paths=inputs,
        prior_stage_manifests=prior,
        dataset_manifest_path=args.dataset,
        output_path=prepared_path,
    )
    result = train_order9_behavior_cloning(
        config,
        stage_id=args.stage,
        dataset_manifest_path=args.dataset,
        physical_model=physical_model,
        output_dir=output,
        git_revision=args.git_revision or _git_revision(),
        device=args.device,
        parent_checkpoint_path=args.parent_checkpoint,
        source_order3_checkpoint_path=args.source_order3_checkpoint,
        additional_input_artifact_paths={
            "curriculum_config": args.config,
            "robot_model_config": physical_model_path,
        },
    )
    output_artifacts = {
        "policy_checkpoint": result.checkpoint_path,
        "training_metrics": result.metrics_path,
        "loss_curve": result.loss_curve_path,
        "training_result": output / "training_result.json",
    }
    running_path = output / "stage_training_complete.json"
    record_order9_stage_training_outputs(
        prepared,
        config,
        output_artifact_paths=output_artifacts,
        checkpoint_paths_by_family={
            Order9PolicyFamily(result.policy_family): result.checkpoint_path
        },
        output_path=running_path,
    )
    print(f"stage: {result.stage_id}")
    print(f"family: {result.policy_family.value}")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"checkpoint_sha256: {result.checkpoint_sha256}")
    print(f"training_manifest: {running_path}")
    print("promotion_evaluation_completed: false")
    return 0


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
