#!/usr/bin/env python3
from __future__ import annotations

"""Create an acceptance-ineligible C3 initializer for a wider action span."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from amsrr.policies.order9_low_level_policy import (
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_contact_compression_span_migration import (
    ORDER9_CONTACT_COMPRESSION_SPAN_MIGRATION_VERSION,
    rescale_module_count_bias_for_span,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import build_order9_checkpoint_metadata
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.utils.hashing import hash_file, stable_hash


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-span-mm", type=float, required=True)
    parser.add_argument("--new-span-mm", type=float, required=True)
    parser.add_argument(
        "--preserve-module-count", action="append", type=int, required=True
    )
    parser.add_argument("--seed", type=int, default=390_207)
    return parser


def _git_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return revision + "-span-migration-uncommitted"


def main() -> int:
    args = _parser().parse_args()
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    parent_path = (REPOSITORY_ROOT / args.parent_checkpoint).resolve()
    output_dir = (REPOSITORY_ROOT / args.output_dir).resolve()
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "span_migration_report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("Order9 span-migration output already exists")
    old_span_m = 1.0e-3 * float(args.old_span_mm)
    new_span_m = 1.0e-3 * float(args.new_span_mm)
    counts = tuple(sorted(set(args.preserve_module_count)))
    if any(count < 2 or count > 8 for count in counts):
        raise ValueError("preserved module counts must lie in [2, 8]")

    learning = load_order9_learning_config(config_path)
    configured_span = float(
        learning.production_runtime.c3_contact_compression_action_span_m
    )
    if abs(configured_span - new_span_m) > 1.0e-12:
        raise SchemaValidationError(
            "Order9 configured contact-compression span differs from migration target"
        )
    stage = order9_stage_by_id(learning, STAGE_ID)
    physical = build_physical_model_from_config(
        REPOSITORY_ROOT / learning.production_runtime.robot_model_config_path
    )
    parent = load_order9_policy_checkpoint(
        parent_path, device="cpu", expected_family=Order9PolicyFamily.PI_L
    )
    if not isinstance(parent.model, Order9MorphologyInvariantCompressionActorCritic):
        raise SchemaValidationError("Order9 span migration requires the v6 pi_L actor")

    bias = parent.model.contact_compression_module_count_bias
    before = bias.detach().cpu().clone()
    with torch.no_grad():
        for count in counts:
            bias[count] = rescale_module_count_bias_for_span(
                float(before[count].item()),
                old_span_m=old_span_m,
                new_span_m=new_span_m,
            )
    after = bias.detach().cpu().clone()
    parent_update = parent.metadata.metadata.get("ppo_update_index")
    if not isinstance(parent_update, int) or parent_update < -1:
        raise SchemaValidationError("Order9 span-migration parent update is invalid")
    metrics = {
        f"module_{count}_bias_before": float(before[count].item())
        for count in counts
    }
    metrics.update(
        {
            f"module_{count}_bias_after": float(after[count].item())
            for count in counts
        }
    )
    metadata = build_order9_checkpoint_metadata(
        parent.model,
        stage=stage,
        schedule_hash=order9_schedule_hash(learning),
        physical_model_hash=physical.stable_hash(),
        git_revision=_git_revision(),
        random_seed=args.seed,
        input_artifact_hashes={
            "parent_checkpoint": parent.sha256,
            "training_config": hash_file(config_path),
        },
        parent_checkpoint_sha256=parent.sha256,
        source_order3_checkpoint_sha256=(
            parent.metadata.source_order3_checkpoint_sha256
        ),
        metrics=metrics,
        trainer_version=ORDER9_CONTACT_COMPRESSION_SPAN_MIGRATION_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": -1,
            "warm_start_source_ppo_update_index": parent_update,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
            "old_contact_compression_action_span_m": old_span_m,
            "new_contact_compression_action_span_m": new_span_m,
            "behavior_preserved_module_counts": list(counts),
            "stronger_unscaled_module_counts": [
                count for count in range(2, 9) if count not in counts
            ],
            "raw_contact_actor_input": False,
            "config_hash": stable_hash(learning.to_dict()),
        },
    )
    checkpoint_sha = save_order9_policy_checkpoint(
        checkpoint_path, model=parent.model, metadata=metadata
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "migration_version": ORDER9_CONTACT_COMPRESSION_SPAN_MIGRATION_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent.sha256,
        "old_span_m": old_span_m,
        "new_span_m": new_span_m,
        "preserved_module_counts": list(counts),
        "bias_before": before.tolist(),
        "bias_after": after.tolist(),
        "acceptance_eligible": False,
        "requires_fresh_on_policy_ppo": True,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
