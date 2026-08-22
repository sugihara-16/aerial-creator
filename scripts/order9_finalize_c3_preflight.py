#!/usr/bin/env python3
from __future__ import annotations

"""Validate and freeze the no-training C3 production preflight evidence."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_curriculum import (
    load_order9_learning_config,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_pi_l_initializer_rebind import (
    validate_order9_pi_l_initializer_physical_rebind,
)
from amsrr.training.order9_pi_l_stage_runner import (
    resolve_order9_pi_l_stage_plan,
    validate_order9_pi_l_stage_runner_inputs,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


PREFLIGHT_VERSION = "order9_c3_accepted_nominal_preflight_v1"
STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
REQUIRED_TENSORBOARD_TAGS = (
    "reward/step_mean/total_reward",
    "reward/step_mean/weighted_object_goal_progress",
    "reward/step_mean/weighted_object_pose_accuracy",
    "reward/step_mean/weighted_grasp_maintenance",
    "reward/step_mean/weighted_centroidal_stability",
    "reward/step_mean/weighted_energy_penalty",
    "reward/step_mean/weighted_qp_residual_penalty",
    "reward/step_mean/weighted_slip_penalty",
    "reward/step_mean/weighted_collision_penalty",
    "reward/step_mean/weighted_actuator_saturation_penalty",
    "rollout/phase_occupancy/approach",
    "rollout/rate/task_success",
    "rollout/rate/hard_collision",
    "rollout/rate/qp_feasible",
    "performance/rollout_environment_steps_per_s",
    "system/live/gpu_memory_used_mib",
    "system/live/gpu_utilization_percent",
    "system/live/process_rss_mib",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument(
        "--nominal-set",
        default=(
            "artifacts/p4_full/order9/stages/"
            "c3_pi_l_ppo_arbitrary_morphology/"
            "nominal_trajectories_current_lineage_human_accepted_v1_manifest.json"
        ),
    )
    parser.add_argument(
        "--bucket-manifest",
        default=(
            "artifacts/p4_full/order9/stages/"
            "c3_pi_l_ppo_arbitrary_morphology/"
            "rollout_buckets_current_lineage_v5/manifest_c3_nominal_replay_v1.json"
        ),
    )
    parser.add_argument(
        "--review-manifest",
        default=(
            "artifacts/p4_full/order9/stages/"
            "c3_pi_l_ppo_arbitrary_morphology/"
            "nominal_trajectories_current_lineage_human_review_v1.json"
        ),
    )
    parser.add_argument(
        "--initializer-manifest",
        default=(
            "artifacts/p4_full/order9/c3_preparation/"
            "pi_l_active_knot_joint_load_initializer_current_physical_v2_manifest.json"
        ),
    )
    parser.add_argument(
        "--rebound-smoke",
        default=(
            "artifacts/p4_full/order9/c3_preflight_accepted_nominal_v1/"
            "rebound_initializer_2x2.pt"
        ),
    )
    parser.add_argument(
        "--capacity-smoke",
        default=(
            "artifacts/p4_full/order9/c3_preflight_accepted_nominal_v1/"
            "index41_1024x2.pt"
        ),
    )
    parser.add_argument(
        "--tensorboard-event",
        default=(
            "artifacts/p4_full/order9/c3_preflight_accepted_nominal_v1/"
            "tensorboard/train/events.out.tfevents.1785084653.leus.347908.0"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/p4_full/order9/c3_preflight_accepted_nominal_v1/"
            "preflight_manifest_v1.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {name: _resolve(value) for name, value in vars(args).items()}
    output = paths.pop("output")

    config = load_order9_learning_config(paths["config"])
    stage = order9_stage_by_id(config, STAGE_ID)
    runtime = resolve_order9_stage_runtime(config, stage)
    physical = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path)
    )
    physical_hash = physical.stable_hash()
    nominal = validate_order9_c3_nominal_trajectory_set_bytes(
        paths["nominal_set"], repository_root=REPOSITORY_ROOT
    )
    buckets = validate_order9_pi_l_stage_runner_inputs(
        config,
        stage_id=STAGE_ID,
        bucket_manifest_path=paths["bucket_manifest"],
        repository_root=REPOSITORY_ROOT,
    )
    rebind = validate_order9_pi_l_initializer_physical_rebind(
        paths["initializer_manifest"],
        expected_target_physical_model_hash=physical_hash,
    )
    initializer = load_order9_policy_checkpoint(
        rebind.target_checkpoint_path,
        expected_sha256=rebind.target_checkpoint_sha256,
        expected_schedule_hash=order9_schedule_hash(config),
    )
    plan = resolve_order9_pi_l_stage_plan(
        config,
        stage_id=STAGE_ID,
        stage_root=REPOSITORY_ROOT / config.production_runtime.artifact_root / "stages" / STAGE_ID,
        initial_checkpoint_path=rebind.target_checkpoint_path,
        repository_root=REPOSITORY_ROOT,
        expected_physical_model_hash=physical_hash,
    )
    review = json.loads(paths["review_manifest"].read_text(encoding="utf-8"))
    if (
        review.get("accepted_count") != 42
        or review.get("rejected_count") != 0
        or len(review.get("entries", [])) != 42
        or any(entry.get("decision") != "accepted" for entry in review["entries"])
        or review.get("nominal_set_manifest_sha256")
        != hash_file(paths["nominal_set"])
    ):
        raise ValueError("C3 human review evidence is incomplete or stale")
    if len(nominal.entries) != 42 or len(buckets.buckets) != 42:
        raise ValueError("C3 production set must contain exactly 42 buckets")
    if runtime.environment_count != 1024 or runtime.rollout_steps_per_environment != 256:
        raise ValueError("C3 production runtime is not the validated 1024x256 setting")
    if plan.next_update_index != 0 or plan.target_update_count != 14:
        raise ValueError("C3 clean-start PPO update plan differs")

    rebound_smoke = _validate_smoke(
        paths["rebound_smoke"],
        expected_environment_count=2,
        expected_checkpoint_sha256=initializer.sha256,
        expected_physical_model_hash=physical_hash,
        expected_nominal_set_sha256=hash_file(paths["nominal_set"]),
    )
    capacity_smoke = _validate_smoke(
        paths["capacity_smoke"],
        expected_environment_count=1024,
        expected_checkpoint_sha256=rebind.source_checkpoint_sha256,
        expected_physical_model_hash=physical_hash,
        expected_nominal_set_sha256=hash_file(paths["nominal_set"]),
    )
    event = EventAccumulator(str(paths["tensorboard_event"]))
    event.Reload()
    scalar_tags = tuple(event.Tags().get("scalars", ()))
    missing_tags = sorted(set(REQUIRED_TENSORBOARD_TAGS) - set(scalar_tags))
    if missing_tags:
        raise ValueError(f"C3 TensorBoard evidence lacks tags: {missing_tags}")

    payload = {
        "preflight_version": PREFLIGHT_VERSION,
        "status": "passed",
        "scope": "implementation_and_no_training_runtime_preflight",
        "training_started": False,
        "stage_id": STAGE_ID,
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "stage_config_hash": stage.stable_hash(),
        "physical_model_hash": physical_hash,
        "accepted_nominal_set": {
            "path": _portable(paths["nominal_set"]),
            "sha256": hash_file(paths["nominal_set"]),
            "bucket_count": len(nominal.entries),
            "human_review_path": _portable(paths["review_manifest"]),
            "human_review_sha256": hash_file(paths["review_manifest"]),
            "all_human_decisions_accepted": True,
            "reference_rate_hz": 10.0,
            "control_rate_hz": 50.0,
            "configuration_space_planner_runtime_enabled": False,
            "runtime_semantics": "hash_bound_offline_nominal_replay",
        },
        "rollout_buckets": {
            "path": _portable(paths["bucket_manifest"]),
            "sha256": hash_file(paths["bucket_manifest"]),
            "bucket_count": len(buckets.buckets),
        },
        "initializer": {
            "path": _portable(Path(rebind.target_checkpoint_path)),
            "sha256": initializer.sha256,
            "manifest_path": _portable(paths["initializer_manifest"]),
            "manifest_sha256": hash_file(paths["initializer_manifest"]),
            "source_checkpoint_sha256": rebind.source_checkpoint_sha256,
            "source_physical_model_hash": rebind.source_physical_model_hash,
            "target_physical_model_hash": rebind.target_physical_model_hash,
            "state_dict_sha256": rebind.target_state_dict_hash,
            "exact_parameter_copy": rebind.exact_parameter_copy,
        },
        "production_plan": {
            "environment_count_per_process": runtime.environment_count,
            "rollout_steps_per_environment": runtime.rollout_steps_per_environment,
            "generation_environment_steps_train_plus_validation": plan.generation_environment_steps,
            "target_update_count": plan.target_update_count,
            "target_environment_steps": plan.target_environment_steps,
            "next_update_index": plan.next_update_index,
        },
        "real_isaac_smokes": {
            "rebound_initializer": rebound_smoke,
            "accepted_nominal_capacity_1024": capacity_smoke,
            "capacity_equivalence_basis": (
                "the 1024-env smoke used the immutable rebind source; source and "
                "target state_dict hashes are byte-identical"
            ),
        },
        "tensorboard": {
            "event_path": _portable(paths["tensorboard_event"]),
            "event_sha256": hash_file(paths["tensorboard_event"]),
            "scalar_tag_count": len(scalar_tags),
            "required_tag_count": len(REQUIRED_TENSORBOARD_TAGS),
            "required_tags_present": True,
        },
        "claims_not_made": [
            "no C3 PPO update has been executed",
            "short smokes do not establish grasp/transport task success",
            "human review establishes ideal-tracking nominal geometry, not learned-policy success",
        ],
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing C3 preflight manifest differs from validation")
    else:
        _atomic_write_json(output, payload)
    print(
        "ORDER9_C3_PREFLIGHT="
        + json.dumps(
            {
                "status": "passed",
                "output": str(output),
                "sha256": hash_file(output),
                "bucket_count": len(nominal.entries),
                "environment_count_per_process": runtime.environment_count,
                "target_update_count": plan.target_update_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_smoke(
    path: Path,
    *,
    expected_environment_count: int,
    expected_checkpoint_sha256: str,
    expected_physical_model_hash: str,
    expected_nominal_set_sha256: str,
) -> dict[str, Any]:
    artifact = load_order9_tensor_rollout_artifact(
        path, expected_sha256=hash_file(path)
    )
    metadata = artifact.metadata
    nominal = metadata.get("c3_nominal_reference")
    if (
        metadata.get("environment_count") != expected_environment_count
        or metadata.get("pi_l_checkpoint_sha256") != expected_checkpoint_sha256
        or metadata.get("physical_model_hash") != expected_physical_model_hash
        or not isinstance(nominal, dict)
        or nominal.get("set_manifest_sha256") != expected_nominal_set_sha256
        or nominal.get("configuration_space_planner_runtime_enabled") is not False
        or float(nominal.get("nominal_knot_dt_s", 0.0)) != 0.1
    ):
        raise ValueError(f"C3 Isaac smoke provenance differs: {path}")
    runtime_load = metadata.get("runtime_load", {})
    return {
        "path": _portable(path),
        "sha256": hash_file(path),
        "environment_count": int(metadata["environment_count"]),
        "rollout_steps": int(metadata["rollout_steps"]),
        "finite_state": True,
        "terminal_count": int(metadata.get("terminal_count", 0)),
        "aggregate_env_steps_per_s": float(metadata["aggregate_env_steps_per_s"]),
        "setup_wall_elapsed_s": float(metadata["setup_wall_elapsed_s"]),
        "rollout_wall_elapsed_s": float(metadata["rollout_wall_elapsed_s"]),
        "gpu_memory_used_peak_mib": float(
            runtime_load.get("gpu_memory_used_mib_peak", 0.0)
        ),
        "process_rss_peak_mib": float(runtime_load.get("process_rss_mib_peak", 0.0)),
        "nominal_artifact_sha256": str(nominal["artifact_sha256"]),
    }


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
