#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_design_teacher_dataset import (
    Order9PiDTeacherDatasetConfig,
    build_order9_pi_d_teacher_dataset,
)
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the split-safe deterministic grammar teacher dataset for "
            "Order 9 C7 pi_D behavior cloning."
        )
    )
    parser.add_argument("--morphology-pool", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/p4_full/order9/datasets/c7_pi_d_teacher",
    )
    parser.add_argument(
        "--robot-model-config", default="configs/robot/robot_model.yaml"
    )
    parser.add_argument("--base-task")
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--train-records", type=int, default=400)
    parser.add_argument("--validation-records", type=int, default=50)
    parser.add_argument("--held-out-records", type=int, default=50)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--max-modules", type=int, default=8)
    args = parser.parse_args()
    base_task = (
        TaskSpec.from_dict(load_config(args.base_task))
        if args.base_task
        else default_grasp_carry_task_spec()
    )
    model = build_physical_model_from_config(args.robot_model_config)
    config = Order9PiDTeacherDatasetConfig(
        seed=args.seed,
        train_record_count=args.train_records,
        validation_record_count=args.validation_records,
        held_out_record_count=args.held_out_records,
        min_modules=args.min_modules,
        max_modules=args.max_modules,
    )
    manifest = build_order9_pi_d_teacher_dataset(
        args.morphology_pool,
        Path(args.output_dir),
        config=config,
        base_task_spec=base_task,
        physical_model=model,
        robot_model_config_path=args.robot_model_config,
    )
    manifest_path = Path(args.output_dir) / "manifest.json"
    print(f"dataset_id: {manifest.dataset_id}")
    print(
        "design_action_trajectory_records: "
        f"{manifest.record_counts['design_action_trajectory']}"
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
