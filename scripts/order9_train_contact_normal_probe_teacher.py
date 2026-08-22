#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.policies.order9_low_level_policy import (
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    order9_state_dict_hash,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_contact_normal_probe_teacher import (
    ORDER9_CONTACT_NORMAL_PROBE_TEACHER_VERSION,
    train_order9_contact_normal_probe_teacher_head,
    validate_order9_contact_normal_probe_teacher_dataset,
    validate_order9_contact_normal_probe_teacher_target_lineage,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument(
        "--probe-source-checkpoint-sha256",
        help=(
            "Checkpoint that generated the probe. Defaults to the training "
            "target itself; a direct on-policy child is also accepted."
        ),
    )
    parser.add_argument("--probe-rollout", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--physical-model-config", default="configs/robot/robot_model.yaml"
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--maximum-optimizer-steps", type=int, default=128)
    parser.add_argument("--retention-kl-weight", type=float, default=1.0)
    return parser


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = _parser().parse_args()
    parent_path = Path(args.parent_checkpoint).resolve()
    rollout_path = Path(args.probe_rollout).resolve()
    teacher_path = Path(args.teacher_dataset).resolve()
    output_path = Path(args.output_checkpoint).resolve()
    report_path = Path(args.output_report).resolve()
    loaded = load_order9_policy_checkpoint(
        parent_path, expected_sha256=args.parent_checkpoint_sha256
    )
    if not isinstance(
        loaded.model, Order9CategoricalContactNormalPhaseConditionedActorCritic
    ):
        raise TypeError("probe teacher requires the categorical contact-normal policy")
    artifact = load_order9_tensor_rollout_artifact(rollout_path)
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    validate_order9_contact_normal_probe_teacher_dataset(teacher)
    rollout_sha256 = hash_file(rollout_path)
    teacher_sha256 = hash_file(teacher_path)
    if teacher["source_rollout_sha256"] != rollout_sha256:
        raise ValueError("probe teacher rollout hash differs")
    probe_source_sha256 = (
        loaded.sha256
        if args.probe_source_checkpoint_sha256 is None
        else args.probe_source_checkpoint_sha256
    )
    if teacher["source_checkpoint_sha256"] != probe_source_sha256:
        raise ValueError("probe teacher source checkpoint differs")
    validate_order9_contact_normal_probe_teacher_target_lineage(
        probe_source_checkpoint_sha256=probe_source_sha256,
        target_checkpoint_sha256=loaded.sha256,
        target_parent_checkpoint_sha256=loaded.metadata.parent_checkpoint_sha256,
    )
    physical_model = build_physical_model_from_config(args.physical_model_config)
    if physical_model.stable_hash() != loaded.metadata.physical_model_hash:
        raise ValueError("probe teacher physical model differs")

    result = train_order9_contact_normal_probe_teacher_head(
        loaded.model,
        artifact,
        teacher,
        physical_model=physical_model,
        learning_rate=args.learning_rate,
        maximum_optimizer_steps=args.maximum_optimizer_steps,
        retention_kl_weight=args.retention_kl_weight,
    )
    metrics = dict(loaded.metadata.metrics)
    metrics.update(
        {
            "contact_normal_probe_teacher_example_count": float(
                result.example_count
            ),
            "contact_normal_probe_teacher_optimizer_steps": float(
                result.optimizer_step_count
            ),
            "contact_normal_probe_teacher_initial_loss": (
                result.initial_teacher_loss
            ),
            "contact_normal_probe_teacher_final_loss": result.final_teacher_loss,
            "contact_normal_probe_teacher_final_match_fraction": (
                result.final_target_match_fraction
            ),
            "contact_normal_probe_teacher_retention_kl": result.retention_kl,
        }
    )
    input_hashes = dict(loaded.metadata.input_artifact_hashes)
    input_hashes.update(
        {
            "contact_normal_probe_rollout": rollout_sha256,
            "contact_normal_probe_teacher": teacher_sha256,
        }
    )
    auxiliary_metadata = dict(loaded.metadata.metadata)
    auxiliary_metadata["contact_normal_probe_teacher"] = {
        "version": ORDER9_CONTACT_NORMAL_PROBE_TEACHER_VERSION,
        "parent_checkpoint_sha256": loaded.sha256,
        "probe_source_checkpoint_sha256": probe_source_sha256,
        "source_rollout_sha256": rollout_sha256,
        "teacher_dataset_sha256": teacher_sha256,
        "trainable_parameter_prefix": "contact_normal_category_logits.",
        "other_parameters_bitwise_frozen": True,
        "result": result.to_dict(),
    }
    checkpoint_metadata = replace(
        loaded.metadata,
        learning_mode=(
            loaded.metadata.learning_mode + "+contact_normal_probe_teacher_head_only"
        ),
        state_dict_hash=order9_state_dict_hash(loaded.model.state_dict()),
        parent_checkpoint_sha256=loaded.sha256,
        input_artifact_hashes=input_hashes,
        metrics=metrics,
        metadata=auxiliary_metadata,
    )
    output_sha256 = save_order9_policy_checkpoint(
        output_path, model=loaded.model, metadata=checkpoint_metadata
    )
    report = {
        "version": ORDER9_CONTACT_NORMAL_PROBE_TEACHER_VERSION,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": loaded.sha256,
        "probe_source_checkpoint_sha256": probe_source_sha256,
        "probe_rollout": str(rollout_path),
        "probe_rollout_sha256": rollout_sha256,
        "teacher_dataset": str(teacher_path),
        "teacher_dataset_sha256": teacher_sha256,
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": output_sha256,
        "result": result.to_dict(),
    }
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
