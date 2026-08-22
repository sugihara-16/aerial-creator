#!/usr/bin/env python3
from __future__ import annotations

"""Build one immutable C3 checkpoint/nominal/QPID execution bundle."""

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
)
from amsrr.training.order9_c3_execution_bundle import (
    ORDER9_C3_EXECUTION_BUNDLE_VERSION,
    load_order9_c3_execution_bundle,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--nominal-set-manifest", required=True)
    parser.add_argument("--reset-bank-directory", required=True)
    parser.add_argument("--minimum-module-count", type=int, required=True)
    parser.add_argument("--maximum-module-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--action-contract",
        default=ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    bucket_path = _resolve(args.bucket_manifest)
    nominal_path = _resolve(args.nominal_set_manifest)
    reset_root = _resolve(args.reset_bank_directory)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not 2 <= args.minimum_module_count <= args.maximum_module_count <= 8:
        raise ValueError("module count range must be within [2, 8]")

    learning = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(learning, STAGE_ID)
    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_path, repository_root=REPOSITORY_ROOT
    )
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(bucket_path)
    nominal = validate_order9_c3_nominal_trajectory_set_bytes(
        nominal_path,
        repository_root=REPOSITORY_ROOT,
    )
    nominal_entries = {entry.bucket_id: entry for entry in nominal.entries}

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = checkpoint.get("metadata", {})
    required_checkpoint_metadata = {
        key: checkpoint_metadata[key]
        for key in (
            "policy_family",
            "policy_version",
            "curriculum_stage_id",
            "actor_observation_contract",
            "critic_observation_contract",
            "action_contract",
            "physical_model_hash",
        )
    }

    bindings = []
    counts: dict[str, int] = {}
    for bucket in bucket_manifest.buckets:
        if not (
            args.minimum_module_count
            <= int(bucket.module_count)
            <= args.maximum_module_count
        ):
            continue
        entry = nominal_entries.get(bucket.bucket_id)
        if entry is None:
            raise ValueError(f"nominal set lacks bucket: {bucket.bucket_id}")
        accepted = bucket.metadata.get("accepted_nominal_trajectory")
        if not isinstance(accepted, dict):
            raise ValueError(f"bucket is not nominal-bound: {bucket.bucket_id}")
        collision_path = _resolve(str(accepted["collision_validation_path"]))
        reset_path = reset_root / f"{bucket.bucket_id}.pt"
        if not reset_path.is_file():
            raise FileNotFoundError(reset_path)
        bindings.append(
            {
                "bucket_id": bucket.bucket_id,
                "module_count": int(bucket.module_count),
                "artifact_sha256": entry.artifact_sha256,
                "reset_bank": {
                    "path": _portable(reset_path),
                    "sha256": hash_file(reset_path),
                },
                "collision_validation": {
                    "path": _portable(collision_path),
                    "sha256": hash_file(collision_path),
                },
            }
        )
        key = str(bucket.module_count)
        counts[key] = counts.get(key, 0) + 1

    runtime_sources = [
        "scripts/order9_vectorized_isaac_rollout.py",
        "amsrr/training/order9_c3_execution_bundle.py",
        "amsrr/controllers/qpid_controller.py",
        "amsrr/controllers/batched_qpid_controller.py",
        "amsrr/controllers/batched_virtual_thrust_qp.py",
        "amsrr/training/order9_contact_space_action.py",
        "amsrr/training/order9_tensor_pi_l_runtime.py",
    ]
    payload = {
        "bundle_version": ORDER9_C3_EXECUTION_BUNDLE_VERSION,
        "stage_id": STAGE_ID,
        "action_contract": args.action_contract,
        "config": {
            "path": _portable(config_path),
            "sha256": hash_file(config_path),
            "semantic_hash": stable_hash(learning.to_dict()),
            "stage_semantic_hash": stable_hash(stage.to_dict()),
            "schedule_hash": order9_schedule_hash(learning),
        },
        "checkpoint": {
            "path": _portable(checkpoint_path),
            "sha256": hash_file(checkpoint_path),
            "required_metadata": required_checkpoint_metadata,
        },
        "bucket_manifest": {
            "path": _portable(bucket_path),
            "sha256": hash_file(bucket_path),
        },
        "nominal_set": {
            "path": _portable(nominal_path),
            "sha256": hash_file(nominal_path),
        },
        "runtime_source_bindings": [
            {"path": value, "sha256": hash_file(REPOSITORY_ROOT / value)}
            for value in runtime_sources
        ],
        "required_nominal_semantics": {
            "phase_sequence": [
                "approach",
                "contact_acquisition",
                "lift",
                "transport",
                "place",
                "release",
                "retreat",
                "settle",
            ],
            "rigid_robot_target_translation": {
                "repair_version": "order9_c3_rigid_robot_target_translation_v1",
                "offset_world_m": [0.0, 0.0, 0.0],
                "phase_z_offset_start_end_m": {
                    "retreat": [0.3, 0.3],
                    "settle": [0.3, 0.3],
                },
                "phase_smooth_z_offset_start_end_m": {
                    "release": [0.0, 0.3]
                },
                "smooth_offset_boundary_velocity_zero": True,
                "joint_targets_changed": False,
                "object_targets_changed": False,
                "contact_assignments_changed": False,
                "runtime_planner_enabled": False,
            },
            "generation_method_tokens_by_phase": {
                "release": [
                    "order9_c3_rigid_robot_target_translation_v1",
                    "order9_c3_smoothstep_robot_target_translation_v1",
                ],
                "retreat": ["order9_c3_rigid_robot_target_translation_v1"],
                "settle": ["order9_c3_rigid_robot_target_translation_v1"],
            },
        },
        "module_count_histogram": counts,
        "bucket_bindings": bindings,
        "allowed_change_contract": {
            "rule": "No runtime input may differ from this bundle. A new bundle is required for every intentional change.",
            "semantic_gate_required_before_isaac_launch": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    verified = load_order9_c3_execution_bundle(
        output, repository_root=REPOSITORY_ROOT
    )
    print(
        "ORDER9_C3_EXECUTION_BUNDLE="
        + json.dumps(
            {
                "path": _portable(output),
                "sha256": verified.source_sha256,
                "bucket_count": len(verified.bucket_bindings),
                "module_count_histogram": counts,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
