#!/usr/bin/env python3
from __future__ import annotations

"""Compare C3 PPO trust-region and wrench-weight methods on one rollout.

This is an acceptance-ineligible diagnostic: every candidate starts from the
same behavior checkpoint and consumes the same immutable on-policy rollout.
Promising settings must subsequently collect a fresh rollout and pass the
normal fixed-bucket / promotion path.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import (
    load_order9_learning_config,
)
from amsrr.training.order9_offline_training import (
    build_order9_checkpoint_metadata,
    order9_checkpoint_input_hashes,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_tensor_on_policy_dataset import (
    load_order9_tensor_pi_l_dataset,
    validate_order9_tensor_pi_l_dataset_for_stage,
)
from amsrr.training.order9_tensor_pi_l_ppo import (
    update_order9_tensor_pi_l_ppo,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    Order9TensorRolloutArtifact,
)
from amsrr.policies.order9_low_level_policy import (
    Order9ContactResidualPhaseConditionedActorCritic,
)
from amsrr.utils.hashing import hash_file


DIAGNOSTIC_VERSION = (
    "order9_c3_ppo_method_comparison_v3_compression_credit_assignment"
)
STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
WRENCH_TERM_NAME = "weighted_wrench_range_violation_penalty"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--rollout-dataset", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--update-index", type=int, default=4)
    parser.add_argument(
        "--allow-config-hash-mismatch",
        action="store_true",
        help=(
            "Diagnostic only: permit an immutable rollout when the sole "
            "contract difference is config_hash (for optimizer-only A/B)."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Run only the named built-in variant; may be repeated.",
    )
    return parser


def _variants() -> dict[
    str, tuple[float, float, float, float, float, tuple[str, ...]]
]:
    # name: (non-target hard KL limit, KL penalty weight,
    #        wrench reward weight, learning-rate multiplier,
    #        entropy-bonus multiplier)
    variants = {
        "baseline_kl005_w100": (0.005, 1.0, 1.0, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w100": (0.010, 1.0, 1.0, 1.0, 1.0, ("establish_contact", "lift")),
        "kl020_w100": (0.020, 1.0, 1.0, 1.0, 1.0, ("establish_contact", "lift")),
        "soft_kl039_w100": (0.039, 1.0, 1.0, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w025": (0.010, 1.0, 0.25, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w050": (0.010, 1.0, 0.50, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w200": (0.010, 1.0, 2.00, 1.0, 1.0, ("establish_contact", "lift")),
        "kl020_w200": (0.020, 1.0, 2.00, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w500": (0.010, 1.0, 5.00, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w1000": (0.010, 1.0, 10.00, 1.0, 1.0, ("establish_contact", "lift")),
        "kl010_w100_entropy0": (0.010, 1.0, 1.00, 1.0, 0.0, ("establish_contact", "lift")),
        "kl010_lr2": (0.010, 1.0, 1.0, 2.0, 1.0, ("establish_contact", "lift")),
        "kl010_lr4": (0.010, 1.0, 1.0, 4.0, 1.0, ("establish_contact", "lift")),
        "kl010_lr8": (0.010, 1.0, 1.0, 8.0, 1.0, ("establish_contact", "lift")),
        "kl010_lr4_entropy0": (0.010, 1.0, 1.0, 4.0, 0.0, ("establish_contact", "lift")),
        "kl010_lr4_entropy010": (0.010, 1.0, 1.0, 4.0, 0.1, ("establish_contact", "lift")),
        "kl010_w1000_contact_only": (0.010, 1.0, 10.0, 1.0, 1.0, ("establish_contact",)),
        "kl010_w1000_lift_only": (0.010, 1.0, 10.0, 1.0, 1.0, ("lift",)),
    }
    # These variants keep the exact rollout reward unchanged and isolate the
    # training-only privileged compression target.  The PhysX wrench remains
    # absent from actor observations and deployment.
    variants.update(
        {
            "kl010_w100_no_teacher": (
                0.010,
                1.0,
                1.0,
                1.0,
                1.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w010_no_teacher": (
                0.010,
                1.0,
                0.10,
                1.0,
                1.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w010_lr025_no_teacher": (
                0.010,
                1.0,
                0.10,
                0.25,
                1.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w050_no_teacher": (
                0.010,
                1.0,
                0.50,
                1.0,
                1.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w100_teacher_step001": (
                0.010,
                1.0,
                1.0,
                1.0,
                1.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w100_teacher_step0025": (
                0.010,
                1.0,
                1.0,
                1.0,
                1.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w100_lr025_compression_only": (
                0.010,
                1.0,
                1.0,
                0.25,
                0.0,
                ("establish_contact", "lift"),
            ),
            "kl010_w100_lr025_contact_compression_only": (
                0.010,
                1.0,
                1.0,
                0.25,
                0.0,
                ("establish_contact",),
            ),
            "kl010_w100_lr025_joint_head_compression_only": (
                0.010,
                1.0,
                1.0,
                0.25,
                0.0,
                ("establish_contact", "lift"),
            ),
            "contact_residual_teacher_w1_compression_only": (
                0.010,
                1.0,
                1.0,
                1.0,
                0.0,
                ("establish_contact", "lift", "transport", "place"),
            ),
            "contact_residual_teacher_w5_compression_only": (
                0.010,
                1.0,
                1.0,
                1.0,
                0.0,
                ("establish_contact", "lift", "transport", "place"),
            ),
            "contact_residual_teacher_w5_step050_compression_only": (
                0.010,
                1.0,
                1.0,
                1.0,
                0.0,
                ("establish_contact", "lift", "transport", "place"),
            ),
        }
    )
    return variants


def _privileged_teacher_override(name: str) -> tuple[float, float] | None:
    return {
        "kl010_w100_no_teacher": (0.0, 0.10),
        "kl010_w010_no_teacher": (0.0, 0.10),
        "kl010_w010_lr025_no_teacher": (0.0, 0.10),
        "kl010_w050_no_teacher": (0.0, 0.10),
        "kl010_w100_teacher_step001": (0.10, 0.01),
        "kl010_w100_teacher_step0025": (0.10, 0.025),
        "contact_residual_teacher_w1_compression_only": (1.0, 0.10),
        "contact_residual_teacher_w5_compression_only": (5.0, 0.10),
        "contact_residual_teacher_w5_step050_compression_only": (5.0, 0.50),
    }.get(name)


def _compression_only_actor_override(name: str) -> bool:
    return name.endswith("_compression_only")


def _freeze_non_joint_actor_parameters(name: str, model: torch.nn.Module) -> None:
    if isinstance(model, Order9ContactResidualPhaseConditionedActorCritic):
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad = parameter_name.startswith(
                ("contact_residual_decoder.", "critic.")
            )
        return
    if "joint_head" not in name:
        return
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = parameter_name.startswith(
            ("joint_decoder.", "critic.")
        )


def _reweight_wrench(
    artifact: Order9TensorRolloutArtifact,
    multiplier: float,
) -> Order9TensorRolloutArtifact:
    names = list(artifact.metadata["reward_term_names"])
    index = names.index(WRENCH_TERM_NAME)
    tensors = dict(artifact.tensors)
    if multiplier != 1.0:
        tensors["reward"] = (
            artifact.tensors["reward"]
            + (multiplier - 1.0) * artifact.tensors["reward_terms"][..., index]
        )
        reward_terms = artifact.tensors["reward_terms"].clone()
        reward_terms[..., index].mul_(multiplier)
        tensors["reward_terms"] = reward_terms
    metadata = dict(artifact.metadata)
    metadata["diagnostic_wrench_reward_multiplier"] = multiplier
    return Order9TensorRolloutArtifact(
        metadata=metadata,
        tensors=tensors,
        artifact_version=artifact.artifact_version,
    )


def _git_revision() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return value + "-diagnostic-uncommitted"


def main() -> int:
    args = _parser().parse_args()
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    dataset_path = (REPOSITORY_ROOT / args.rollout_dataset).resolve()
    parent_path = (REPOSITORY_ROOT / args.parent_checkpoint).resolve()
    output = (REPOSITORY_ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, STAGE_ID)
    physical_model = build_physical_model_from_config(
        REPOSITORY_ROOT / config.production_runtime.robot_model_config_path
    )
    bundle = load_order9_tensor_pi_l_dataset(dataset_path)
    parent = load_order9_policy_checkpoint(
        parent_path,
        device=args.device,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=order9_schedule_hash(config),
    )
    validation = validate_order9_tensor_pi_l_dataset_for_stage(
        bundle,
        config=config,
        stage_id=STAGE_ID,
        behavior_checkpoint_sha256=parent.sha256,
    )
    permitted_config_hash_mismatch = (
        args.allow_config_hash_mismatch
        and validation.failures == ["config_hash_mismatch"]
    )
    if not validation.valid and not permitted_config_hash_mismatch:
        raise RuntimeError("dataset contract failed: " + ",".join(validation.failures))

    variants = _variants()
    selected = list(args.variant or variants)
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    seed = config.production_runtime.seed + stage.stage_index * 100_000 + args.update_index
    base_optimization = replace(
        config.optimization.pi_l_ppo,
        epochs_per_update=config.optimization.c3_boundary_fine_tune.epochs_per_update,
        learning_rate=(
            config.optimization.pi_l_ppo.learning_rate
            * config.optimization.c3_boundary_fine_tune.learning_rate_scale
        ),
    )
    input_hashes = order9_checkpoint_input_hashes(
        bundle,
        parent_checkpoint_path=parent_path,
        source_order3_checkpoint_path=None,
        additional_paths={},
    )
    rows = []
    for ordinal, name in enumerate(selected):
        (
            hard_kl,
            kl_weight,
            wrench_weight,
            learning_rate_multiplier,
            entropy_bonus_multiplier,
            target_actor_phase_labels,
        ) = variants[name]
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / "checkpoint.pt"
        metrics_path = destination / "metrics.json"
        if checkpoint_path.is_file() and metrics_path.is_file():
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            print(f"SKIP {name}: complete", flush=True)
            continue

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = load_order9_policy_checkpoint(
            parent_path,
            device=args.device,
            expected_family=Order9PolicyFamily.PI_L,
            expected_schedule_hash=order9_schedule_hash(config),
        ).model
        model.train()
        _freeze_non_joint_actor_parameters(name, model)
        optimization = replace(
            base_optimization,
            learning_rate=(
                base_optimization.learning_rate * learning_rate_multiplier
            ),
            entropy_bonus_weight=(
                base_optimization.entropy_bonus_weight
                * entropy_bonus_multiplier
            ),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=optimization.learning_rate
        )
        fine_tune = replace(
            config.optimization.c3_boundary_fine_tune,
            non_target_parent_kl_limit=hard_kl,
            non_target_parent_kl_weight=kl_weight,
            target_actor_phase_labels=list(target_actor_phase_labels),
        )
        teacher_override = _privileged_teacher_override(name)
        if teacher_override is not None:
            fine_tune = replace(
                fine_tune,
                privileged_compression_teacher_weight=teacher_override[0],
                privileged_compression_teacher_action_step=teacher_override[1],
            )
        fine_tune.validate()
        artifacts = tuple(
            _reweight_wrench(value, wrench_weight)
            for value in bundle.train_artifacts
        )
        started = time.perf_counter()
        with torch.autograd.set_multithreading_enabled(False):
            result = update_order9_tensor_pi_l_ppo(
                model,
                artifacts,
                physical_model=physical_model,
                optimizer=optimizer,
                config=optimization,
                behavior_checkpoint_sha256=parent.sha256,
                seed=seed,
                boundary_fine_tune=fine_tune,
                compression_only_actor_objective=(
                    _compression_only_actor_override(name)
                ),
                joint_head_only_actor_update="joint_head" in name,
                progress_callback=lambda step, values, label=name: print(
                    f"{label} step={step} kl={values['approximate_kl']:.6f} "
                    f"max_nt={values['maximum_non_target_topology_phase_kl']:.6f} "
                    f"applied={int(values['optimizer_step_applied'])}",
                    flush=True,
                ),
            )
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metadata = build_order9_checkpoint_metadata(
            model,
            stage=stage,
            schedule_hash=order9_schedule_hash(config),
            physical_model_hash=physical_model.stable_hash(),
            git_revision=_git_revision(),
            random_seed=seed,
            input_artifact_hashes=input_hashes,
            parent_checkpoint_sha256=parent.sha256,
            source_order3_checkpoint_sha256=(
                parent.metadata.source_order3_checkpoint_sha256
            ),
            metrics={
                "approximate_kl": result.approximate_kl,
                "optimizer_step_count": float(result.optimizer_step_count),
            },
            trainer_version=DIAGNOSTIC_VERSION,
            extra_metadata={
                "acceptance_eligible": False,
                "same_rollout_method_comparison": True,
                "ppo_update_index": args.update_index,
                "variant": name,
                "wrench_reward_multiplier": wrench_weight,
                "learning_rate_multiplier": learning_rate_multiplier,
                "entropy_bonus_multiplier": entropy_bonus_multiplier,
                "target_actor_phase_labels": list(target_actor_phase_labels),
                "compression_only_actor_objective": (
                    _compression_only_actor_override(name)
                ),
                "joint_head_only_actor_update": "joint_head" in name,
                "non_target_parent_kl_limit": hard_kl,
                "non_target_parent_kl_weight": kl_weight,
                "privileged_compression_teacher_weight": (
                    fine_tune.privileged_compression_teacher_weight
                ),
                "privileged_compression_teacher_action_step": (
                    fine_tune.privileged_compression_teacher_action_step
                ),
                "privileged_compression_teacher_training_only": True,
                "privileged_compression_teacher_underforce_only": (
                    fine_tune.privileged_compression_teacher_underforce_only
                ),
                "privileged_wrench_satisfied_parent_kl_weight": (
                    fine_tune.privileged_wrench_satisfied_parent_kl_weight
                ),
                "topology_gradient_surgery_enabled": (
                    fine_tune.topology_gradient_surgery_enabled
                ),
                "raw_contact_actor_input": False,
                "diagnostic_config_hash_mismatch_permitted": (
                    permitted_config_hash_mismatch
                ),
            },
        )
        checkpoint_sha = save_order9_policy_checkpoint(
            checkpoint_path, model=model, metadata=metadata
        )
        row = {
            "variant": name,
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "acceptance_eligible": False,
            "parent_checkpoint_sha256": parent.sha256,
            "rollout_manifest_sha256": bundle.manifest_sha256,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "wall_elapsed_s": elapsed,
            "non_target_parent_kl_limit": hard_kl,
            "non_target_parent_kl_weight": kl_weight,
            "wrench_reward_multiplier": wrench_weight,
            "learning_rate_multiplier": learning_rate_multiplier,
            "entropy_bonus_multiplier": entropy_bonus_multiplier,
            "target_actor_phase_labels": list(target_actor_phase_labels),
            "privileged_compression_teacher_weight": (
                fine_tune.privileged_compression_teacher_weight
            ),
            "privileged_compression_teacher_action_step": (
                fine_tune.privileged_compression_teacher_action_step
            ),
            "privileged_compression_teacher_training_only": True,
            "privileged_compression_teacher_underforce_only": (
                fine_tune.privileged_compression_teacher_underforce_only
            ),
            "privileged_wrench_satisfied_parent_kl_weight": (
                fine_tune.privileged_wrench_satisfied_parent_kl_weight
            ),
            "topology_gradient_surgery_enabled": (
                fine_tune.topology_gradient_surgery_enabled
            ),
            "raw_contact_actor_input": False,
            "diagnostic_config_hash_mismatch_permitted": (
                permitted_config_hash_mismatch
            ),
            "ppo_update": result.to_dict(),
        }
        metrics_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        rows.append(row)
        print(
            f"DONE {name}: steps={result.optimizer_step_count} "
            f"kl={result.approximate_kl:.6f} sha={checkpoint_sha}",
            flush=True,
        )
        del optimizer, model, artifacts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "acceptance_eligible": False,
        "dataset_path": str(dataset_path),
        "dataset_sha256": hash_file(dataset_path),
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent.sha256,
        "diagnostic_config_hash_mismatch_permitted": (
            permitted_config_hash_mismatch
        ),
        "variants": rows,
    }
    report_path = output / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"REPORT {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
