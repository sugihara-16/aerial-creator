#!/usr/bin/env python3
from __future__ import annotations

"""Create a fresh-on-policy C3 initializer with wider normal exploration."""

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
    Order9ContactSpacePhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_contact_normal_exploration_migration import (
    ORDER9_CONTACT_NORMAL_ACTION_INDEX,
    ORDER9_CONTACT_NORMAL_EXPLORATION_MIGRATION_VERSION,
    widen_order9_contact_normal_exploration,
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
    parser.add_argument("--seed", type=int, default=390_209)
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
    return revision + "-contact-normal-exploration-uncommitted"


def main() -> int:
    args = _parser().parse_args()
    config_path = _resolve(args.config)
    parent_path = _resolve(args.parent_checkpoint)
    output_dir = _resolve(args.output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "contact_normal_exploration_report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError(
            "Order9 contact-normal exploration initializer already exists"
        )

    learning = load_order9_learning_config(config_path)
    configured_std = (
        learning.optimization.c3_boundary_fine_tune.
        contact_normal_exploration_initial_std
    )
    if configured_std is None:
        raise SchemaValidationError(
            "Order9 C3 contact-normal exploration initial std is not configured"
        )
    stage = order9_stage_by_id(learning, STAGE_ID)
    physical = build_physical_model_from_config(
        _resolve(learning.production_runtime.robot_model_config_path)
    )
    parent = load_order9_policy_checkpoint(
        parent_path,
        device="cpu",
        expected_family=Order9PolicyFamily.PI_L,
    )
    if not isinstance(parent.model, Order9ContactSpacePhaseConditionedActorCritic):
        raise SchemaValidationError(
            "Order9 contact-normal exploration initializer requires v7 pi_L"
        )
    if parent.metadata.curriculum_stage_id != STAGE_ID:
        raise SchemaValidationError(
            "Order9 contact-normal exploration parent stage differs"
        )
    if parent.metadata.physical_model_hash != physical.stable_hash():
        raise SchemaValidationError(
            "Order9 contact-normal exploration physical model differs"
        )
    parent_update = parent.metadata.metadata.get("ppo_update_index")
    if not isinstance(parent_update, int) or parent_update < 0:
        raise SchemaValidationError(
            "Order9 contact-normal exploration parent update is invalid"
        )

    before = {
        key: value.detach().cpu().clone()
        for key, value in parent.model.state_dict().items()
    }
    migration = widen_order9_contact_normal_exploration(
        parent.model,
        target_std=float(configured_std),
    )
    after = parent.model.state_dict()
    changed_keys = [
        key for key in before if not torch.equal(before[key], after[key].cpu())
    ]
    if changed_keys != ["contact_space_actor_log_std"]:
        raise SchemaValidationError(
            "Order9 contact-normal exploration changed unexpected tensors"
        )
    delta = before["contact_space_actor_log_std"] != after[
        "contact_space_actor_log_std"
    ].cpu()
    changed_indices = torch.nonzero(delta, as_tuple=False).flatten().tolist()
    if changed_indices != [ORDER9_CONTACT_NORMAL_ACTION_INDEX]:
        raise SchemaValidationError(
            "Order9 contact-normal exploration changed unexpected coordinates"
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
        metrics={
            "changed_scalar_count": float(migration.changed_scalar_count),
            "source_contact_normal_std": migration.source_std,
            "target_contact_normal_std": migration.target_std,
        },
        trainer_version=ORDER9_CONTACT_NORMAL_EXPLORATION_MIGRATION_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": -1,
            "warm_start_source_ppo_update_index": parent_update,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
            "deterministic_actor_mean_preserved": True,
            "changed_parameter_names": [
                "contact_space_actor_log_std[translation.inward_normal]"
            ],
            "source_contact_normal_log_std": migration.source_log_std,
            "source_contact_normal_std": migration.source_std,
            "target_contact_normal_log_std": migration.target_log_std,
            "target_contact_normal_std": migration.target_std,
            "c3_action_contract": "contact_space_projected_policy_command",
            "raw_contact_actor_input": False,
            "config_hash": stable_hash(learning.to_dict()),
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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
        expected_schedule_hash=order9_schedule_hash(learning),
    )
    if not isinstance(rebound.model, Order9ContactSpacePhaseConditionedActorCritic):
        raise SchemaValidationError("saved exploration initializer type differs")
    observed_std = float(
        rebound.model.contact_space_actor_log_std[
            ORDER9_CONTACT_NORMAL_ACTION_INDEX
        ].exp().item()
    )
    if abs(observed_std - migration.target_std) > 1.0e-6:
        raise SchemaValidationError("saved contact-normal exploration std differs")

    report = {
        "migration_version": ORDER9_CONTACT_NORMAL_EXPLORATION_MIGRATION_VERSION,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent.sha256,
        "parent_update_index": parent_update,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_contact_normal_log_std": migration.source_log_std,
        "source_contact_normal_std": migration.source_std,
        "target_contact_normal_log_std": migration.target_log_std,
        "target_contact_normal_std": migration.target_std,
        "changed_scalar_count": migration.changed_scalar_count,
        "changed_parameter_names": [
            "contact_space_actor_log_std[translation.inward_normal]"
        ],
        "deterministic_actor_mean_preserved": True,
        "acceptance_eligible": False,
        "requires_fresh_on_policy_ppo": True,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
