#!/usr/bin/env python3
from __future__ import annotations

"""Warm-start the C3 contact residual head from privileged Isaac rollouts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

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
from amsrr.training.order9_contact_residual_warm_start import (
    ORDER9_CONTACT_RESIDUAL_WARM_START_VERSION,
    Order9ContactResidualWarmStartConfig,
    warm_start_order9_contact_residual,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import (
    build_order9_checkpoint_metadata,
    order9_checkpoint_input_hashes,
)
from amsrr.training.order9_pipeline import (
    order9_schedule_hash,
    order9_stage_by_id,
)
from amsrr.training.order9_tensor_on_policy_dataset import (
    load_order9_tensor_pi_l_dataset,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash

STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--rollout-dataset", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument(
        "--target-artifact",
        action="append",
        default=[],
        help=(
            "Optional real continuous rollout for the entering morphology. "
            "When supplied, dataset target-morphology shards are replaced."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-module-count", type=int, default=5)
    parser.add_argument("--compression-action-step", type=float, default=0.40)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--maximum-samples-per-artifact", type=int, default=8192)
    parser.add_argument("--anchor-loss-weight", type=float, default=1.0)
    parser.add_argument("--anchor-control-loss-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=390_005)
    return parser


def _git_revision() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return value + "-warm-start-uncommitted"


def main() -> int:
    args = _parser().parse_args()
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    dataset_path = (REPOSITORY_ROOT / args.rollout_dataset).resolve()
    parent_path = (REPOSITORY_ROOT / args.parent_checkpoint).resolve()
    output_dir = (REPOSITORY_ROOT / args.output_dir).resolve()
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "warm_start_report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("Order9 warm-start output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)

    learning = load_order9_learning_config(config_path)
    schedule_hash = order9_schedule_hash(learning)
    stage = order9_stage_by_id(learning, STAGE_ID)
    physical_model = build_physical_model_from_config(
        REPOSITORY_ROOT / learning.production_runtime.robot_model_config_path
    )
    bundle = load_order9_tensor_pi_l_dataset(dataset_path)
    parent = load_order9_policy_checkpoint(
        parent_path,
        device=args.device,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if not isinstance(parent.model, Order9MorphologyInvariantCompressionActorCritic):
        raise SchemaValidationError(
            "Order9 contact residual warm-start requires the v6 actor"
        )
    if bundle.manifest.stage_id != STAGE_ID:
        raise SchemaValidationError("Order9 warm-start dataset stage differs")
    if bundle.manifest.curriculum_schedule_hash != schedule_hash:
        raise SchemaValidationError("Order9 warm-start schedule differs")
    if bundle.manifest.physical_model_hash != physical_model.stable_hash():
        raise SchemaValidationError("Order9 warm-start physical model differs")
    if any(
        artifact.metadata.get("raw_contact_actor_input") is not False
        for artifact in bundle.train_artifacts
    ):
        raise SchemaValidationError(
            "Order9 warm-start requires contact-free actor observations"
        )

    warm_config = Order9ContactResidualWarmStartConfig(
        target_module_count=args.target_module_count,
        compression_action_step=args.compression_action_step,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        maximum_samples_per_artifact=args.maximum_samples_per_artifact,
        anchor_loss_weight=args.anchor_loss_weight,
        anchor_control_loss_weight=args.anchor_control_loss_weight,
        seed=args.seed,
    )
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    explicit_target_paths = tuple(
        (REPOSITORY_ROOT / value).resolve() for value in args.target_artifact
    )
    explicit_targets = tuple(
        load_order9_tensor_rollout_artifact(path) for path in explicit_target_paths
    )
    behavior_equivalent_v5_sha = parent.metadata.parent_checkpoint_sha256
    allowed_teacher_behavior_checkpoints = {
        parent.sha256,
        *(
            [behavior_equivalent_v5_sha]
            if behavior_equivalent_v5_sha is not None
            else []
        ),
    }
    training_artifacts = tuple(
        artifact
        for artifact in bundle.train_artifacts
        if len(artifact.metadata["module_ids"]) < args.target_module_count
    ) + (
        explicit_targets
        if explicit_targets
        else tuple(
            artifact
            for artifact in bundle.train_artifacts
            if len(artifact.metadata["module_ids"]) == args.target_module_count
        )
    )
    if any(
        len(artifact.metadata["module_ids"]) != args.target_module_count
        for artifact in explicit_targets
    ):
        raise SchemaValidationError(
            "Order9 explicit warm-start target morphology differs"
        )
    if any(
        artifact.metadata.get("raw_contact_actor_input") is not False
        or artifact.metadata.get("pi_l_checkpoint_sha256")
        not in allowed_teacher_behavior_checkpoints
        or (
            artifact.metadata.get("split") == "train"
            and artifact.metadata.get("training_only_continuous_teacher_rollout")
            is not True
        )
        for artifact in explicit_targets
    ):
        raise SchemaValidationError(
            "Order9 explicit warm-start target is not a contact-free parent rollout"
        )
    with torch.autograd.set_multithreading_enabled(False):
        result = warm_start_order9_contact_residual(
            parent.model,
            training_artifacts,
            physical_model=physical_model,
            config=warm_config,
        )
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    input_hashes = order9_checkpoint_input_hashes(
        bundle,
        parent_checkpoint_path=parent_path,
        source_order3_checkpoint_path=None,
        additional_paths={
            f"continuous_target_{index:03d}": path
            for index, path in enumerate(explicit_target_paths)
        },
    )
    parent_update = parent.metadata.metadata.get("ppo_update_index")
    if not isinstance(parent_update, int) or parent_update < -1:
        raise SchemaValidationError("Order9 warm-start parent update is invalid")
    metadata = build_order9_checkpoint_metadata(
        parent.model,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=physical_model.stable_hash(),
        git_revision=_git_revision(),
        random_seed=args.seed,
        input_artifact_hashes=input_hashes,
        parent_checkpoint_sha256=parent.sha256,
        source_order3_checkpoint_sha256=(
            parent.metadata.source_order3_checkpoint_sha256
        ),
        metrics={
            "target_rmse": result.final_target_rmse,
            "anchor_rmse": result.final_anchor_rmse,
            "optimizer_step_count": float(result.optimizer_step_count),
        },
        trainer_version=ORDER9_CONTACT_RESIDUAL_WARM_START_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": parent_update,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
            "source_rollout_behavior_checkpoint_sha256": (
                bundle.manifest.behavior_checkpoint_sha256
            ),
            "source_rollout_is_on_policy_for_parent": (
                bundle.manifest.behavior_checkpoint_sha256 == parent.sha256
            ),
            "source_rollout_behavior_equivalent_v5_checkpoint_sha256": (
                behavior_equivalent_v5_sha
            ),
            "explicit_continuous_target_artifact_sha256s": [
                hash_file(path) for path in explicit_target_paths
            ],
            "target_module_count": args.target_module_count,
            "preserved_module_counts": list(range(2, args.target_module_count)),
            "compression_action_step": args.compression_action_step,
            "privileged_teacher_training_only": True,
            "raw_contact_actor_input": False,
            "config_hash": stable_hash(learning.to_dict()),
        },
    )
    checkpoint_sha = save_order9_policy_checkpoint(
        checkpoint_path, model=parent.model, metadata=metadata
    )
    report = {
        **result.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent.sha256,
        "rollout_dataset_path": str(dataset_path),
        "rollout_dataset_sha256": bundle.manifest_sha256,
        "rollout_behavior_checkpoint_sha256": (
            bundle.manifest.behavior_checkpoint_sha256
        ),
        "explicit_continuous_target_artifact_paths": [
            str(path) for path in explicit_target_paths
        ],
        "explicit_continuous_target_artifact_sha256s": [
            hash_file(path) for path in explicit_target_paths
        ],
        "wall_elapsed_s": elapsed,
        "configuration": warm_config.__dict__,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
