#!/usr/bin/env python3
from __future__ import annotations

"""Prepare split-safe randomized object/topology buckets for pi_L PPO."""

import argparse
import json
from pathlib import Path

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_rollout_buckets import (
    order9_pi_l_collector_arguments,
    prepare_order9_pi_l_rollout_buckets,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", default="c2_pi_l_ppo_fixed_conservative")
    parser.add_argument("--train-buckets", type=int, default=8)
    parser.add_argument("--validation-buckets", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--fixed-robot-usd",
        default=(
            "artifacts/isaac/robots/holon/holon_p4_2_graph/"
            "holon_p4_2_graph.usda"
        ),
    )
    parser.add_argument(
        "--morphology-pool",
        default="artifacts/p4_full/order9/morphology_pool.json",
    )
    parser.add_argument(
        "--morphology-assets",
        default="artifacts/p4_full/order9/morphology_assets/manifest.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-collector-arguments", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    config_path = _resolve(args.config, repository)
    config = load_order9_learning_config(config_path)
    model = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path, repository)
    )
    canonical = load_order9_canonical_reset(
        _resolve(
            config.production_runtime.canonical_order8_report_path,
            repository,
        ),
        expected_sha256=(
            config.production_runtime.canonical_order8_report_sha256
        ),
    )
    base_task = build_order8_grasp_carry_task_spec(
        object_pose_world=tuple(canonical.object_pose_world),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=canonical.transport_distance_m,
        support_height_m=config.randomization.support_top_z_m,
        max_contact_force_n=config.hard_checker.qp_force_scale_n,
        max_contact_torque_nm=config.hard_checker.qp_torque_scale_nm,
        selected_gripper_friction=(
            config.randomization.nominal_selected_gripper_friction
        ),
        task_id="order9-vectorized-base",
    )
    output = Path(args.output).resolve()
    manifest = prepare_order9_pi_l_rollout_buckets(
        output,
        config=config,
        stage_id=args.stage,
        physical_model=model,
        base_task_spec=base_task,
        train_bucket_count=args.train_buckets,
        validation_bucket_count=args.validation_buckets,
        repository_root=repository,
        fixed_robot_usd_path=args.fixed_robot_usd,
        morphology_pool_path=args.morphology_pool,
        morphology_asset_manifest_path=args.morphology_assets,
        seed=args.seed,
    )
    manifest_path = output / "manifest.json"
    validate_order9_pi_l_rollout_bucket_bytes(
        manifest_path,
        repository_root=repository,
    )
    print(
        "ORDER9_ROLLOUT_BUCKETS="
        + json.dumps(
            {
                "stage_id": manifest.stage_id,
                "bucket_count": len(manifest.buckets),
                "train_bucket_count": sum(
                    bucket.split.value == "train" for bucket in manifest.buckets
                ),
                "validation_bucket_count": sum(
                    bucket.split.value == "validation"
                    for bucket in manifest.buckets
                ),
                "manifest": str(manifest_path),
                "manifest_sha256": hash_file(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.print_collector_arguments:
        for bucket in manifest.buckets:
            print(
                "ORDER9_BUCKET_ARGUMENTS="
                + json.dumps(
                    {
                        "bucket_id": bucket.bucket_id,
                        "arguments": order9_pi_l_collector_arguments(
                            bucket,
                            bucket_manifest_path=manifest_path,
                            repository_root=repository,
                        ),
                    },
                    sort_keys=True,
                )
            )
    return 0


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
