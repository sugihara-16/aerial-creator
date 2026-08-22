#!/usr/bin/env python3
from __future__ import annotations

"""Rebind exact C3 pi_L tensors to a runtime-only curriculum config edit."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.policies.order9_low_level_policy import (
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import build_order9_checkpoint_metadata
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.utils.hashing import hash_file, stable_hash


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
REBIND_VERSION = "order9_c3_runtime_config_checkpoint_rebind_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--seed", type=int, default=390_208)
    return parser


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _git_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return revision + "-runtime-config-rebind-uncommitted"


def main() -> int:
    args = _parser().parse_args()
    config_path = _resolve(args.config)
    parent_path = _resolve(args.parent_checkpoint)
    output_dir = _resolve(args.output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "runtime_config_rebind_report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("Order9 runtime-config rebind output already exists")

    learning = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(learning, STAGE_ID)
    physical = build_physical_model_from_config(
        _resolve(learning.production_runtime.robot_model_config_path)
    )
    parent = load_order9_policy_checkpoint(
        parent_path,
        device="cpu",
        expected_family=Order9PolicyFamily.PI_L,
    )
    if not isinstance(parent.model, Order9MorphologyInvariantCompressionActorCritic):
        raise SchemaValidationError("C3 runtime-config rebind requires the v6 pi_L actor")
    if parent.metadata.curriculum_stage_id != STAGE_ID:
        raise SchemaValidationError("C3 runtime-config rebind parent stage differs")
    if parent.metadata.physical_model_hash != physical.stable_hash():
        raise SchemaValidationError("C3 runtime-config rebind physical model differs")
    parent_update = parent.metadata.metadata.get("ppo_update_index")
    if not isinstance(parent_update, int) or parent_update < -1:
        raise SchemaValidationError("C3 runtime-config rebind parent update is invalid")

    source_state_dict_hash = parent.metadata.state_dict_hash
    schedule_hash = order9_schedule_hash(learning)
    metadata = build_order9_checkpoint_metadata(
        parent.model,
        stage=stage,
        schedule_hash=schedule_hash,
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
        metrics={"changed_parameter_count": 0.0},
        trainer_version=REBIND_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": -1,
            "warm_start_source_ppo_update_index": parent_update,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
            "runtime_config_rebind_reason": args.reason,
            "source_curriculum_schedule_hash": (
                parent.metadata.curriculum_schedule_hash
            ),
            "exact_parameter_copy": True,
            "raw_contact_actor_input": False,
            "config_hash": stable_hash(learning.to_dict()),
        },
    )
    if metadata.state_dict_hash != source_state_dict_hash:
        raise SchemaValidationError("runtime-config rebind changed pi_L parameters")
    checkpoint_sha256 = save_order9_policy_checkpoint(
        checkpoint_path,
        model=parent.model,
        metadata=metadata,
    )
    rebound = load_order9_policy_checkpoint(
        checkpoint_path,
        device="cpu",
        expected_sha256=checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if rebound.metadata.state_dict_hash != source_state_dict_hash:
        raise SchemaValidationError("saved runtime-config rebind changed pi_L parameters")

    report = {
        "rebind_version": REBIND_VERSION,
        "reason": args.reason,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent.sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_curriculum_schedule_hash": parent.metadata.curriculum_schedule_hash,
        "target_curriculum_schedule_hash": schedule_hash,
        "source_state_dict_hash": source_state_dict_hash,
        "target_state_dict_hash": rebound.metadata.state_dict_hash,
        "exact_parameter_copy": True,
        "acceptance_eligible": False,
        "requires_fresh_on_policy_ppo": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
