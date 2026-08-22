#!/usr/bin/env python3
from __future__ import annotations

"""Pretrain the categorical contact-normal head from authored deficits."""

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
    ORDER9_CONTACT_NORMAL_PRELOAD_DEFICIT_TEACHER_VERSION,
    build_order9_contact_normal_preload_deficit_teacher_dataset,
    train_order9_contact_normal_probe_teacher_head,
    validate_order9_contact_normal_probe_teacher_dataset,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--deficit-rollout", action="append", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--physical-model-config", default="configs/robot/robot_model.yaml"
    )
    parser.add_argument(
        "--target-phase-labels",
        default="establish_contact,lift,transport,place",
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
    rollout_paths = [Path(value).resolve() for value in args.deficit_rollout]
    output_path = Path(args.output_checkpoint).resolve()
    report_path = Path(args.output_report).resolve()
    phase_labels = tuple(
        value.strip()
        for value in args.target_phase_labels.split(",")
        if value.strip()
    )
    if not phase_labels:
        raise ValueError("preload-deficit teacher target phases are empty")

    loaded = load_order9_policy_checkpoint(
        parent_path, expected_sha256=args.parent_checkpoint_sha256
    )
    if not isinstance(
        loaded.model, Order9CategoricalContactNormalPhaseConditionedActorCritic
    ):
        raise TypeError(
            "preload-deficit teacher requires the categorical contact-normal policy"
        )
    physical_model = build_physical_model_from_config(
        args.physical_model_config
    )
    if physical_model.stable_hash() != loaded.metadata.physical_model_hash:
        raise ValueError("preload-deficit teacher physical model differs")

    rollout_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    input_hashes = dict(loaded.metadata.input_artifact_hashes)
    for rollout_index, rollout_path in enumerate(rollout_paths):
        rollout_sha256 = hash_file(rollout_path)
        artifact = load_order9_tensor_rollout_artifact(rollout_path)
        if artifact.metadata["pi_l_checkpoint_sha256"] != loaded.sha256:
            raise ValueError(
                "preload-deficit rollout was not collected by the parent checkpoint"
            )
        dataset = build_order9_contact_normal_preload_deficit_teacher_dataset(
            artifact,
            source_rollout_sha256=rollout_sha256,
            target_phase_labels=phase_labels,
        )
        validate_order9_contact_normal_probe_teacher_dataset(dataset)
        dataset_path = report_path.with_name(
            f"{report_path.stem}_dataset_{rollout_index:02d}.json"
        )
        _atomic_write(
            dataset_path,
            json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        )
        dataset_sha256 = hash_file(dataset_path)
        result = train_order9_contact_normal_probe_teacher_head(
            loaded.model,
            artifact,
            dataset,
            physical_model=physical_model,
            learning_rate=args.learning_rate,
            maximum_optimizer_steps=args.maximum_optimizer_steps,
            retention_kl_weight=args.retention_kl_weight,
        )
        key = f"contact_normal_preload_deficit_rollout_{rollout_index:02d}"
        input_hashes[key] = rollout_sha256
        input_hashes[f"{key}_teacher_dataset"] = dataset_sha256
        rollout_rows.append(
            {
                "rollout": str(rollout_path),
                "rollout_sha256": rollout_sha256,
                "teacher_dataset": str(dataset_path),
                "teacher_dataset_sha256": dataset_sha256,
                "example_count": dataset["confident_example_count"],
                "retention_example_count": dataset["retention_example_count"],
            }
        )
        result_rows.append(result.to_dict())

    metrics = dict(loaded.metadata.metrics)
    metrics.update(
        {
            "contact_normal_preload_deficit_pretrain_rollout_count": float(
                len(rollout_rows)
            ),
            "contact_normal_preload_deficit_pretrain_example_count": float(
                sum(int(value["example_count"]) for value in rollout_rows)
            ),
            "contact_normal_preload_deficit_pretrain_final_match_fraction": float(
                sum(float(value["final_target_match_fraction"]) for value in result_rows)
                / len(result_rows)
            ),
        }
    )
    auxiliary_metadata = dict(loaded.metadata.metadata)
    auxiliary_metadata["contact_normal_preload_deficit_pretrain"] = {
        "version": ORDER9_CONTACT_NORMAL_PRELOAD_DEFICIT_TEACHER_VERSION,
        "parent_checkpoint_sha256": loaded.sha256,
        "target_phase_labels": list(phase_labels),
        "trainable_parameter_prefix": "contact_normal_category_logits.",
        "other_parameters_bitwise_frozen": True,
        "rollouts": rollout_rows,
        "results": result_rows,
    }
    checkpoint_metadata = replace(
        loaded.metadata,
        learning_mode=(
            loaded.metadata.learning_mode
            + "+contact_normal_preload_deficit_teacher_head_only"
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
        "version": ORDER9_CONTACT_NORMAL_PRELOAD_DEFICIT_TEACHER_VERSION,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": loaded.sha256,
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": output_sha256,
        "target_phase_labels": list(phase_labels),
        "rollouts": rollout_rows,
        "results": result_rows,
    }
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
