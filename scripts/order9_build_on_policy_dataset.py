#!/usr/bin/env python3
from __future__ import annotations

"""Build one hash-bound pi_L PPO generation from real-Isaac raw shards."""

import argparse
from pathlib import Path

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_tensor_dataset_builder import (
    build_order9_pi_l_on_policy_dataset,
)
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--pi-l-checkpoint", required=True)
    parser.add_argument("--raw-artifact", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repository / config_path
    config = load_order9_learning_config(config_path)
    model_path = Path(config.production_runtime.robot_model_config_path)
    if not model_path.is_absolute():
        model_path = repository / model_path
    model = build_physical_model_from_config(model_path)
    manifest = build_order9_pi_l_on_policy_dataset(
        args.output,
        raw_artifact_paths=args.raw_artifact,
        generation_id=args.generation_id,
        stage_id=args.stage,
        pi_l_checkpoint_path=args.pi_l_checkpoint,
        config=config,
        physical_model=model,
    )
    manifest_path = Path(args.output).resolve() / "manifest.json"
    print(f"dataset_id: {manifest.dataset_id}")
    print(
        "environment_steps: "
        f"{manifest.metadata['on_policy_environment_step_count']}"
    )
    print(f"low_level_records: {manifest.record_counts['low_level_control']}")
    print(f"manifest_sha256: {hash_file(manifest_path)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
