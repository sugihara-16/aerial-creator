#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_online_training import train_order9_ppo_update
from amsrr.training.order9_pipeline import (
    load_order9_stage_manifest,
    order9_schedule_hash,
    preflight_order9_stage,
    record_order9_stage_training_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply one exact-replay Order 9 PPO update to one fresh rollout."
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--rollout-dataset", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--update-index", required=True, type=int)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--prior-stage-manifest", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--git-revision")
    args = parser.parse_args()

    config = load_order9_learning_config(args.config)
    stage_root = Path(
        args.output_dir
        or Path(config.production_runtime.artifact_root) / "stages" / args.stage
    )
    output = stage_root / f"update_{args.update_index:06d}"
    output.mkdir(parents=True, exist_ok=True)
    physical_model_path = config.production_runtime.robot_model_config_path
    physical_model = build_physical_model_from_config(physical_model_path)
    parent = load_order9_policy_checkpoint(
        args.parent_checkpoint,
        expected_schedule_hash=order9_schedule_hash(config),
    )
    prior = [
        load_order9_stage_manifest(path) for path in args.prior_stage_manifest
    ]
    inputs = {
        "curriculum_config": args.config,
        "robot_model_config": physical_model_path,
        "parent_checkpoint": args.parent_checkpoint,
    }
    prepared_path = output / "stage_prepared.json"
    prepared, _ = preflight_order9_stage(
        config,
        stage_id=args.stage,
        input_artifact_paths=inputs,
        prior_stage_manifests=prior,
        dataset_manifest_path=args.rollout_dataset,
        behavior_checkpoint_sha256=parent.sha256,
        output_path=prepared_path,
    )
    result = train_order9_ppo_update(
        config,
        stage_id=args.stage,
        rollout_manifest_path=args.rollout_dataset,
        parent_checkpoint_path=args.parent_checkpoint,
        physical_model=physical_model,
        output_dir=output,
        git_revision=args.git_revision or _git_revision(),
        update_index=args.update_index,
        device=args.device,
        additional_input_artifact_paths={
            "curriculum_config": args.config,
            "robot_model_config": physical_model_path,
        },
    )
    result_path = output / f"training_result_update_{args.update_index:06d}.json"
    output_artifacts = {
        "policy_checkpoint": result.checkpoint_path,
        "training_metrics": result.metrics_path,
        "training_result": result_path,
    }
    running_path = output / "stage_training_complete.json"
    record_order9_stage_training_outputs(
        prepared,
        config,
        output_artifact_paths=output_artifacts,
        checkpoint_paths_by_family={
            result.policy_family: result.checkpoint_path
        },
        output_path=running_path,
    )
    print(f"stage: {result.stage_id}")
    print(f"update_index: {result.update_index}")
    print(f"family: {result.policy_family.value}")
    print(f"consumed_environment_steps: {result.consumed_environment_steps}")
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
