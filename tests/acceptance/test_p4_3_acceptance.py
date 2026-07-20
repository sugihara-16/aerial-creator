from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from amsrr.acceptance.p4_3_acceptance import run_p4_3_acceptance
from amsrr.policies.learned_low_level_policy import (
    PI_L_FEATURE_NAMES,
    PI_L_OUTPUT_MODE,
    PI_L_POLICY_CHECKPOINT_VERSION,
    PI_L_TARGET_NAMES,
)
from amsrr.schemas.datasets import (
    P4_3_DATASET_KINDS,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    P4_3DatasetManifest,
)
from amsrr.training.p2_learning_dataset import P2_LEARNING_FEATURE_NAMES
from amsrr.training.p4_3_pi_d_training import P4_3_PI_D_CHECKPOINT_TASK
from amsrr.utils.hashing import hash_file


def test_p4_3_minimum_learning_run_requires_bound_safe_artifacts(tmp_path: Path) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)

    report = run_p4_3_acceptance(
        dataset_manifest_path=manifest_path,
        pi_l_dir=policy_dirs["pi_l"],
        pi_h_dir=policy_dirs["pi_h"],
        pi_d_dir=policy_dirs["pi_d"],
    )

    assert report.completion_passed is True
    assert not report.failures


@pytest.mark.parametrize(
    ("policy_name", "metadata_key", "invalid_value"),
    (
        ("pi_l", "output_mode", "actuator_commands"),
        ("pi_h", "deterministic_assignment_feasibility_gate", False),
        ("pi_d", "outcome_target_is_inference_feature", True),
    ),
)
def test_p4_3_acceptance_rejects_head_specific_checkpoint_metadata(
    tmp_path: Path,
    policy_name: str,
    metadata_key: str,
    invalid_value: Any,
) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)
    checkpoint_path = policy_dirs[policy_name] / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint[metadata_key] = invalid_value
    torch.save(checkpoint, checkpoint_path)

    report = _run_acceptance(manifest_path, policy_dirs)

    assert report.completion_passed is False
    assert f"{policy_name}_checkpoint_invalid" in report.failures


def test_p4_3_acceptance_rejects_pi_d_without_within_task_ranking_signal(
    tmp_path: Path,
) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)
    metrics_path = policy_dirs["pi_d"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "train_target_std": 0.0,
            "train_unique_target_count": 1.0,
            "task_with_multiple_candidates_count": 0.0,
            "train_ranking_pair_count": 0.0,
        }
    )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    report = _run_acceptance(manifest_path, policy_dirs)

    assert report.completion_passed is False
    assert "pi_d_outcome_signal_or_target_leakage_evidence_invalid" in report.failures


def test_p4_3_acceptance_rejects_manifest_with_tampered_dataset_shard(
    tmp_path: Path,
) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)
    manifest = P4_3DatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    shard_path = Path(manifest.shards[0].path)
    shard_path.write_text('{"tampered":true}\n', encoding="utf-8")

    report = _run_acceptance(manifest_path, policy_dirs)

    assert report.completion_passed is False
    assert "dataset_shard_integrity_or_provenance_invalid" in report.failures


def test_p4_3_acceptance_cross_checks_bound_pi_l_archive_semantics(
    tmp_path: Path,
) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)
    online_path = policy_dirs["pi_l"] / "online_rollout_evaluation.json"
    online = json.loads(online_path.read_text(encoding="utf-8"))
    archive_path = Path(online["archive_path"])
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["metrics"]["p4_3_pi_l_learned_decision_count"] = 0.0
    archive_path.write_text(json.dumps(archive) + "\n", encoding="utf-8")
    online["archive_sha256"] = hash_file(archive_path)
    online_path.write_text(json.dumps(online), encoding="utf-8")

    report = _run_acceptance(manifest_path, policy_dirs)

    assert report.completion_passed is False
    assert "pi_l_semantic_or_online_isaac_evidence_invalid" in report.failures


def test_p4_3_acceptance_recomputes_pi_l_nonlearned_overlay_provenance(
    tmp_path: Path,
) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)
    online_path = policy_dirs["pi_l"] / "online_rollout_evaluation.json"
    online = json.loads(online_path.read_text(encoding="utf-8"))
    archive_path = Path(online["archive_path"])
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["policy_commands"][0]["priority_weights"] = {"tampered": 1.0}
    archive_path.write_text(json.dumps(archive) + "\n", encoding="utf-8")
    online["archive_sha256"] = hash_file(archive_path)
    online_path.write_text(json.dumps(online), encoding="utf-8")

    report = _run_acceptance(manifest_path, policy_dirs)

    assert report.completion_passed is False
    assert "pi_l_semantic_or_online_isaac_evidence_invalid" in report.failures


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    (
        ("checkpoint_sha256", "0" * 64),
        ("archive_sha256", "0" * 64),
        ("source_is_real_isaac", False),
        ("isaac_backed", False),
        ("checkpoint_loaded", False),
        ("checkpoint_load_failed_count", 1),
        ("learned_decision_count", 0),
        ("overlay_nonzero_count", 0),
        ("rollout_passed_count", 0),
        ("all_rollouts_passed", False),
        ("controller_qp_safety_layer_used", False),
        ("controller_authority_preserved", False),
        ("controller_active_knot_preserved", False),
        ("deterministic_fallback_available", False),
        ("learned_policy_deployed_in_isaac", False),
        ("safety_violation_count", 1),
        ("controller_qp_infeasible_terminal_count", 1),
        ("hard_collision_count", 1),
        ("object_drop_count", 1),
        ("p4_full_completion_claim", True),
    ),
)
def test_p4_3_acceptance_rejects_unbound_or_unsafe_pi_l_online_evidence(
    tmp_path: Path,
    key: str,
    invalid_value: Any,
) -> None:
    manifest_path, policy_dirs = _passing_artifacts(tmp_path)
    online_path = policy_dirs["pi_l"] / "online_rollout_evaluation.json"
    online = json.loads(online_path.read_text(encoding="utf-8"))
    online[key] = invalid_value
    online_path.write_text(json.dumps(online), encoding="utf-8")

    report = _run_acceptance(manifest_path, policy_dirs)

    assert report.completion_passed is False
    assert "pi_l_semantic_or_online_isaac_evidence_invalid" in report.failures


def test_p4_3_acceptance_fails_without_learning_artifacts(tmp_path: Path) -> None:
    report = run_p4_3_acceptance(
        dataset_manifest_path=tmp_path / "missing.json",
        pi_l_dir=tmp_path / "pi_l",
        pi_h_dir=tmp_path / "pi_h",
        pi_d_dir=tmp_path / "pi_d",
    )
    assert report.completion_passed is False
    assert "dataset_manifest_missing" in report.failures


def _passing_artifacts(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    manifest_path = _write_manifest(tmp_path)
    policy_dirs = {name: tmp_path / name for name in ("pi_l", "pi_h", "pi_d")}
    for name, directory in policy_dirs.items():
        directory.mkdir()
        torch.save(_checkpoint_payload(name), directory / "checkpoint.pt")
        (directory / "metrics.json").write_text(
            json.dumps(_metrics_payload(name)),
            encoding="utf-8",
        )
        (directory / "loss_curve.csv").write_text(
            "epoch,loss\n1,1.0\n",
            encoding="utf-8",
        )
        evaluation_name = (
            "rollout_outcome_evaluation.json"
            if name == "pi_d"
            else "rollout_evaluation.json"
        )
        (directory / evaluation_name).write_text(
            json.dumps(_evaluation_payload(name)),
            encoding="utf-8",
        )
        (directory / "fallback_metadata.json").write_text(
            json.dumps({"deterministic_fallback_available": True}),
            encoding="utf-8",
        )

    pi_l_dir = policy_dirs["pi_l"]
    (pi_l_dir / "reward_curve.csv").write_text(
        "episode,mean_return\n1,0.0\n",
        encoding="utf-8",
    )
    rollout_archive = pi_l_dir / "online_rollout_archive.jsonl"
    checkpoint_sha256 = hash_file(pi_l_dir / "checkpoint.pt")
    rollout_archive.write_text(
        json.dumps(
            {
                "episode_id": "online-unit",
                "success": True,
                "task_spec": {"task_id": "task-held"},
                "policy_commands": [
                    {
                        "desired_body_twist": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "desired_body_pose": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                        "desired_anchor_pose_offsets": {},
                        "joint_position_bias": {},
                        "joint_velocity_bias": {},
                        "residual_wrench_body": None,
                        "contact_tracking_bias": {},
                        "priority_weights": {"p4_2_phase_approach": 1.0},
                    }
                ],
                "metrics": {
                    "isaac_backed": 1.0,
                    "success": 1.0,
                    "object_drop": 0.0,
                    "hard_collision": 0.0,
                    "controller_qp_infeasible_terminal": 0.0,
                    "p4_3_pi_l_checkpoint_load_failed": 0.0,
                    "p4_3_pi_l_checkpoint_loaded": 1.0,
                    "p4_3_pi_l_learned_decision_count": 1.0,
                    "p4_3_pi_l_fallback_count": 0.0,
                    "p4_3_pi_l_runtime_blend_factor": 0.1,
                    "p4_3_pi_l_overlay_nonzero_count": 1.0,
                    "p4_3_pi_l_overlay_delta_norm_sum": 0.1,
                    "p4_3_pi_l_overlay_delta_norm_max": 0.1,
                },
                "rollout_artifacts": {
                    "phase": "P4.3b",
                    "archive_type": "p4_3_pi_l_online_isaac_evaluation",
                    "p4_3_learned_evaluation": True,
                    "learning_claim": True,
                    "learned_policy_success_claim": False,
                    "is_p4_full_completion": False,
                    "p4_3_pi_l_pre_overlay_policy_commands": [
                        {
                            "desired_body_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "desired_body_pose": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                            "desired_anchor_pose_offsets": {},
                            "joint_position_bias": {},
                            "joint_velocity_bias": {},
                            "residual_wrench_body": None,
                            "contact_tracking_bias": {},
                            "priority_weights": {"p4_2_phase_approach": 1.0},
                        }
                    ],
                    "p4_3_pi_l_controller_active_knots": [
                        {
                            "guard_conditions": [
                                {"type": "p4_2_phase", "phase": "approach"},
                                {"type": "p4_2_attach_gate"},
                            ]
                        }
                    ],
                },
                "learning_artifacts": {
                    "stage": "P4.3b",
                    "pi_l_checkpoint_sha256": checkpoint_sha256,
                    "pi_l_checkpoint_loaded": True,
                    "pi_l_online_inference": True,
                    "pi_l_learned_decision_count": 1.0,
                    "pi_l_fallback_count": 0.0,
                    "pi_l_runtime_blend_factor": 0.1,
                    "pi_l_overlay_nonzero_count": 1.0,
                    "pi_l_overlay_delta_norm_sum": 0.1,
                    "pi_l_overlay_delta_norm_max": 0.1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pi_l_dir / "online_rollout_evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": "p4_3_pi_l_online_evaluation_v1",
                "evaluation_type": "learned_pi_l_online_isaac_rollout",
                "source_is_real_isaac": True,
                "isaac_backed": True,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_loaded": True,
                "checkpoint_load_failed_count": 0,
                "learned_decision_count": 1,
                "fallback_count": 0,
                "overlay_nonzero_count": 1,
                "overlay_delta_norm_sum": 0.1,
                "overlay_delta_norm_max": 0.1,
                "runtime_blend_factor": 0.1,
                "rollout_count": 1,
                "task_ids": ["task-held"],
                "rollout_passed_count": 1,
                "all_rollouts_passed": True,
                "controller_qp_infeasible_terminal_count": 0,
                "hard_collision_count": 0,
                "object_drop_count": 0,
                "safety_violation_count": 0,
                "controller_qp_safety_layer_used": True,
                "controller_authority_preserved": True,
                "controller_active_knot_preserved": True,
                "learned_policy_command_fields": [
                    "desired_body_twist",
                    "desired_body_position",
                    "residual_wrench_body",
                ],
                "nonlearned_command_fields_source": "p4_2_deterministic_command",
                "deterministic_fallback_available": True,
                "learned_policy_deployed_in_isaac": True,
                "p4_full_completion_claim": False,
                "archive_path": str(rollout_archive),
                "archive_sha256": hash_file(rollout_archive),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, policy_dirs


def _write_manifest(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    task_by_split = {
        DatasetSplit.TRAIN: "task-train",
        DatasetSplit.VALIDATION: "task-validation",
        DatasetSplit.HELD_OUT: "task-held",
    }
    mask_by_kind = {
        DatasetKind.LOW_LEVEL_CONTROL: "low_level_control_mask",
        DatasetKind.INTERACTION_TRAJECTORY: "high_level_decision_mask",
        DatasetKind.DESIGN_OUTCOME: "design_decision_mask",
    }
    shards: list[DatasetShard] = []
    source_episode_ids: list[str] = []
    for kind in P4_3_DATASET_KINDS:
        for split in DatasetSplit:
            task_id = task_by_split[split]
            episode_id = f"episode-{split.value}"
            if kind == DatasetKind.ISAAC_ROLLOUT:
                source_episode_ids.append(episode_id)
                payload = {
                    "episode_id": episode_id,
                    "task_spec": {"task_id": task_id},
                    "metrics": {"isaac_backed": 1.0},
                    "rollout_artifacts": {
                        "p4_3_dataset_collection": True,
                        "is_p4_full_completion": False,
                    },
                }
            else:
                required_mask = mask_by_kind[kind]
                payload = {
                    "record_id": f"{episode_id}:{kind.value}",
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "split": split.value,
                    "stage_masks": {
                        "design_decision_mask": required_mask == "design_decision_mask",
                        "high_level_decision_mask": required_mask == "high_level_decision_mask",
                        "low_level_control_mask": required_mask == "low_level_control_mask",
                        "assembly_execution_mask": False,
                    },
                }
            shard_path = dataset_dir / f"{kind.value}_{split.value}.jsonl"
            shard_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            shards.append(
                DatasetShard(kind, split, str(shard_path), 1, hash_file(shard_path))
            )
    source_archive = tmp_path / "rollout.jsonl"
    source_archive.write_text('{"source":"unit"}\n', encoding="utf-8")
    manifest = P4_3DatasetManifest(
        dataset_id="unit-p4-3",
        schema_version="p4_3_dataset_v1",
        source_archive_paths=[str(source_archive)],
        source_episode_ids=source_episode_ids,
        train_task_ids=["task-train"],
        validation_task_ids=["task-validation"],
        held_out_task_ids=["task-held"],
        shards=shards,
        record_counts={kind.value: 3 for kind in P4_3_DATASET_KINDS},
        source_hash="source",
        config_hash="config",
        robot_model_hash="robot",
        urdf_hash="urdf",
        thrust_model_hash="thrust",
        task_hashes={
            "task-train": "a",
            "task-validation": "b",
            "task-held": "c",
        },
        geometry_hashes={"geometry": "hash"},
        random_seeds=[0, 1, 2],
        simulator_version="isaac_lab",
        simulator_hash="sim",
        metadata={
            "isaac_backed_episode_count": 3,
            "natural_contact_success_claim": False,
            "p4_full_completion_claim": False,
        },
    )
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return manifest_path


def _checkpoint_payload(name: str) -> dict[str, Any]:
    common = {"state_dict": {"dummy.weight": torch.ones(1)}}
    if name == "pi_l":
        return {
            **common,
            "checkpoint_version": PI_L_POLICY_CHECKPOINT_VERSION,
            "model_type": "TinyPiLDeltaMLP",
            "task": "pi_l_bounded_policy_command_delta_imitation",
            "output_mode": PI_L_OUTPUT_MODE,
            "feature_names": list(PI_L_FEATURE_NAMES),
            "target_names": list(PI_L_TARGET_NAMES),
            "config_hash": "a" * 64,
            "dataset_sha256": "b" * 64,
            "controller_command_output": False,
            "actuator_target_output": False,
            "deterministic_fallback": {"fallback_available": True},
        }
    if name == "pi_h":
        return {
            **common,
            "model_type": "P4_3HighLevelRanker",
            "training_version": "p4_3_pi_h_imitation_v1",
            "training_stage": "P4.3c",
            "training_config_hash": "c" * 64,
            "dataset_hash": "d" * 64,
            "policy_config": {"encoder_d_model": 48, "hidden_dim": 8},
            "output_contract": "ContactWrenchTrajectory",
            "actuator_command_output": False,
            "deterministic_assignment_feasibility_gate": True,
            "deterministic_fallback": "P4_2DeterministicGraspCarryPlanner",
        }
    if name == "pi_d":
        return {
            **common,
            "model_type": "TinyP2MLP",
            "task": P4_3_PI_D_CHECKPOINT_TASK,
            "feature_names": list(P2_LEARNING_FEATURE_NAMES),
            "feature_min": [0.0] * len(P2_LEARNING_FEATURE_NAMES),
            "feature_max": [1.0] * len(P2_LEARNING_FEATURE_NAMES),
            "training_config": {"epochs": 1},
            "training_config_hash": "e" * 64,
            "dataset_hash": "f" * 64,
            "source_of_truth": "deterministic FeasibilityChecker hard gate",
            "inference_contract": "design and deterministic feasibility features only",
            "outcome_target_is_inference_feature": False,
        }
    raise AssertionError(f"unsupported policy fixture: {name}")


def _metrics_payload(name: str) -> dict[str, Any]:
    if name == "pi_l":
        return {
            "source_is_real_isaac": True,
            "deterministic_fallback_available": True,
            "controller_authority": "controller_qp_safety_layer_only",
        }
    if name == "pi_h":
        return {
            "validation_assignment_feasible_rate": 1.0,
            "validation_exact_selection_rate": 1.0,
            "validation_fallback_rate": 0.0,
        }
    return {
        "train_target_std": 0.5,
        "train_unique_target_count": 2.0,
        "task_with_multiple_candidates_count": 1.0,
        "train_ranking_pair_count": 1.0,
        "validation_ranking_pair_count": 1.0,
        "validation_pairwise_ranking_accuracy": 1.0,
        "training_config_hash": "e" * 64,
        "dataset_hash": "f" * 64,
    }


def _evaluation_payload(name: str) -> dict[str, Any]:
    if name == "pi_h":
        return {
            "schema_valid_rate": 1.0,
            "assignment_feasible_rate": 1.0,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "deterministic_safety_gate_used": True,
            "p4_full_completion_claim": False,
        }
    if name == "pi_d":
        return {
            "outcome_fields_are_targets_not_inference_features": True,
            "record_count": 1,
            "training_config_hash": "e" * 64,
            "dataset_hash": "f" * 64,
            "p4_full_completion_claim": False,
        }
    return {}


def _run_acceptance(manifest_path: Path, policy_dirs: dict[str, Path]):
    return run_p4_3_acceptance(
        dataset_manifest_path=manifest_path,
        pi_l_dir=policy_dirs["pi_l"],
        pi_h_dir=policy_dirs["pi_h"],
        pi_d_dir=policy_dirs["pi_d"],
    )
