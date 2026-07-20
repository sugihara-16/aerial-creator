#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_online_training import train_order9_joint_ppo_update
from amsrr.training.order9_pipeline import (
    load_order9_stage_manifest,
    order9_schedule_hash,
    preflight_order9_stage,
    record_order9_stage_training_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply one factorized C9 PPO update to pi_L, pi_H, and pi_D."
    )
    parser.add_argument("--stage", default="c9_joint_object_task_ppo")
    parser.add_argument("--rollout-dataset", required=True)
    parser.add_argument("--pi-l-parent", required=True)
    parser.add_argument("--pi-h-parent", required=True)
    parser.add_argument("--pi-d-parent", required=True)
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
    parent_paths = {
        Order9PolicyFamily.PI_L: args.pi_l_parent,
        Order9PolicyFamily.PI_H: args.pi_h_parent,
        Order9PolicyFamily.PI_D: args.pi_d_parent,
    }
    schedule_hash = order9_schedule_hash(config)
    parents = {
        family: load_order9_policy_checkpoint(
            path,
            expected_family=family,
            expected_schedule_hash=schedule_hash,
        )
        for family, path in parent_paths.items()
    }
    prior = [
        load_order9_stage_manifest(path) for path in args.prior_stage_manifest
    ]
    inputs = {
        "curriculum_config": args.config,
        "robot_model_config": physical_model_path,
        **{
            f"parent_{family.value}": path
            for family, path in parent_paths.items()
        },
    }
    prepared_path = output / "stage_prepared.json"
    prepared, _ = preflight_order9_stage(
        config,
        stage_id=args.stage,
        input_artifact_paths=inputs,
        prior_stage_manifests=prior,
        dataset_manifest_path=args.rollout_dataset,
        behavior_checkpoint_sha256_by_family={
            family.value: parent.sha256 for family, parent in parents.items()
        },
        output_path=prepared_path,
    )
    result = train_order9_joint_ppo_update(
        config,
        stage_id=args.stage,
        rollout_manifest_path=args.rollout_dataset,
        parent_checkpoint_paths_by_family=parent_paths,
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
    result_path = output / (
        f"joint_training_result_update_{args.update_index:06d}.json"
    )
    output_artifacts = {
        **{
            f"policy_checkpoint_{family}": path
            for family, path in result.checkpoint_path_by_family.items()
        },
        "training_metrics": result.metrics_path,
        "training_result": result_path,
    }
    running_path = output / "stage_training_complete.json"
    record_order9_stage_training_outputs(
        prepared,
        config,
        output_artifact_paths=output_artifacts,
        checkpoint_paths_by_family=result.checkpoint_path_by_family,
        output_path=running_path,
    )
    print(f"stage: {result.stage_id}")
    print(f"update_index: {result.update_index}")
    print(f"consumed_environment_steps: {result.consumed_environment_steps}")
    for family in Order9PolicyFamily:
        print(
            f"{family.value}_checkpoint: "
            f"{result.checkpoint_path_by_family[family.value]}"
        )
        print(
            f"{family.value}_checkpoint_sha256: "
            f"{result.checkpoint_sha256_by_family[family.value]}"
        )
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
