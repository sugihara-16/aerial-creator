from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from amsrr.policies.learned_low_level_policy import (
    PI_L_FEATURE_NAMES,
    PI_L_OUTPUT_MODE,
    PI_L_POLICY_CHECKPOINT_VERSION,
    PI_L_TARGET_NAMES,
)
from amsrr.schemas.common import SchemaBase
from amsrr.schemas.datasets import DatasetKind, DatasetSplit, P4_3DatasetManifest
from amsrr.training.p2_learning_dataset import P2_LEARNING_FEATURE_NAMES
from amsrr.training.p4_3_pi_d_training import P4_3_PI_D_CHECKPOINT_TASK
from amsrr.utils.hashing import hash_file


@dataclass
class P4_3AcceptanceReport(SchemaBase):
    dataset_passed: bool
    pi_l_passed: bool
    pi_h_passed: bool
    pi_d_passed: bool
    deterministic_fallbacks_passed: bool
    no_mislabeling_passed: bool
    completion_passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def run_p4_3_acceptance(
    *,
    dataset_manifest_path: str | Path = "artifacts/p4_3/datasets/manifest.json",
    pi_l_dir: str | Path = "artifacts/p4_3/pi_l",
    pi_h_dir: str | Path = "artifacts/p4_3/pi_h",
    pi_d_dir: str | Path = "artifacts/p4_3/pi_d",
) -> P4_3AcceptanceReport:
    failures: list[str] = []
    manifest_path = Path(dataset_manifest_path)
    manifest = _load_manifest(manifest_path, failures)
    dataset_passed = False
    no_mislabeling = False
    if manifest is not None:
        source_count = len(manifest.source_episode_ids)
        isaac_count = int(manifest.metadata.get("isaac_backed_episode_count", 0))
        splits_nonempty = all(
            (manifest.train_task_ids, manifest.validation_task_ids, manifest.held_out_task_ids)
        )
        manifest_contract_passed = (
            source_count > 0
            and isaac_count == source_count
            and splits_nonempty
            and manifest.record_counts.get("low_level_control", 0) > 0
            and manifest.record_counts.get("interaction_trajectory", 0) > 0
            and manifest.record_counts.get("design_outcome", 0) > 0
        )
        shard_integrity_passed = _validate_dataset_artifacts(
            manifest_path,
            manifest,
            failures,
        )
        dataset_passed = manifest_contract_passed and shard_integrity_passed
        if not dataset_passed:
            failures.append("dataset_manifest_not_real_isaac_task_disjoint_complete")
        no_mislabeling = not bool(manifest.metadata.get("natural_contact_success_claim", False)) and not bool(
            manifest.metadata.get("p4_full_completion_claim", False)
        )
        if not no_mislabeling:
            failures.append("dataset_mislabels_natural_contact_or_p4_full_completion")

    pi_l_passed, pi_l_fallback = _policy_artifacts(
        Path(pi_l_dir),
        required=("checkpoint.pt", "metrics.json", "loss_curve.csv", "reward_curve.csv", "rollout_evaluation.json", "online_rollout_evaluation.json", "fallback_metadata.json"),
        failures=failures,
        label="pi_l",
    )
    pi_h_passed, pi_h_fallback = _policy_artifacts(
        Path(pi_h_dir),
        required=("checkpoint.pt", "metrics.json", "loss_curve.csv", "rollout_evaluation.json", "fallback_metadata.json"),
        failures=failures,
        label="pi_h",
    )
    pi_d_passed, pi_d_fallback = _policy_artifacts(
        Path(pi_d_dir),
        required=("checkpoint.pt", "metrics.json", "loss_curve.csv", "rollout_outcome_evaluation.json", "fallback_metadata.json"),
        failures=failures,
        label="pi_d",
    )
    if pi_l_passed and not _csv_has_data(Path(pi_l_dir) / "reward_curve.csv"):
        pi_l_passed = False
        failures.append("pi_l_reward_curve_has_no_data")
    if pi_l_passed and not _validate_pi_l_semantics(
        Path(pi_l_dir),
        failures,
        allowed_task_ids=set(manifest.held_out_task_ids) if manifest is not None else set(),
    ):
        pi_l_passed = False
    if pi_h_passed and not _validate_pi_h_semantics(Path(pi_h_dir), failures):
        pi_h_passed = False
    if pi_d_passed and not _validate_pi_d_semantics(Path(pi_d_dir), failures):
        pi_d_passed = False
    fallback_passed = pi_l_fallback and pi_h_fallback and pi_d_fallback
    if not fallback_passed:
        failures.append("deterministic_fallback_metadata_incomplete")
    completion = all(
        (dataset_passed, pi_l_passed, pi_h_passed, pi_d_passed, fallback_passed, no_mislabeling)
    )
    return P4_3AcceptanceReport(
        dataset_passed=dataset_passed,
        pi_l_passed=pi_l_passed,
        pi_h_passed=pi_h_passed,
        pi_d_passed=pi_d_passed,
        deterministic_fallbacks_passed=fallback_passed,
        no_mislabeling_passed=no_mislabeling,
        completion_passed=completion,
        failures=failures,
        metrics={
            "dataset_passed": float(dataset_passed),
            "pi_l_passed": float(pi_l_passed),
            "pi_h_passed": float(pi_h_passed),
            "pi_d_passed": float(pi_d_passed),
            "deterministic_fallbacks_passed": float(fallback_passed),
            "completion_passed": float(completion),
            "source_episode_count": float(len(manifest.source_episode_ids) if manifest else 0),
        },
        artifacts={
            "dataset_manifest": str(manifest_path),
            "pi_l_dir": str(pi_l_dir),
            "pi_h_dir": str(pi_h_dir),
            "pi_d_dir": str(pi_d_dir),
        },
    )


def _load_manifest(path: Path, failures: list[str]) -> P4_3DatasetManifest | None:
    if not path.is_file():
        failures.append("dataset_manifest_missing")
        return None
    try:
        return P4_3DatasetManifest.from_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"dataset_manifest_invalid:{type(exc).__name__}")
        return None


def _validate_dataset_artifacts(
    manifest_path: Path,
    manifest: P4_3DatasetManifest,
    failures: list[str],
) -> bool:
    """Bind the manifest to real, split-consistent shard bytes.

    Deep schema behavior is covered by the dataset schema/builder tests.  The
    acceptance gate independently verifies the persisted identities, split
    labels, stage masks, real-Isaac provenance, counts, and hashes so a
    self-declared manifest cannot pass while its shards are absent or swapped.
    """

    valid = True
    seen_pairs: set[tuple[DatasetKind, DatasetSplit]] = set()
    isaac_episode_ids: set[str] = set()
    dataset_record_ids: set[str] = set()
    split_tasks = {
        DatasetSplit.TRAIN: set(manifest.train_task_ids),
        DatasetSplit.VALIDATION: set(manifest.validation_task_ids),
        DatasetSplit.HELD_OUT: set(manifest.held_out_task_ids),
    }
    expected_mask = {
        DatasetKind.LOW_LEVEL_CONTROL: "low_level_control_mask",
        DatasetKind.INTERACTION_TRAJECTORY: "high_level_decision_mask",
        DatasetKind.DESIGN_OUTCOME: "design_decision_mask",
    }
    all_mask_names = {
        "design_decision_mask",
        "high_level_decision_mask",
        "low_level_control_mask",
        "assembly_execution_mask",
    }

    for shard in manifest.shards:
        if shard.split is None:
            valid = False
            continue
        pair = (shard.dataset_kind, shard.split)
        if pair in seen_pairs:
            valid = False
        seen_pairs.add(pair)
        shard_path = _evidence_path(manifest_path.parent, shard.path)
        if shard_path is None or not _file_hash_matches(shard_path, shard.sha256):
            valid = False
            continue
        actual_count = 0
        try:
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    actual_count += 1
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        valid = False
                        continue
                    if shard.dataset_kind == DatasetKind.ISAAC_ROLLOUT:
                        episode_id = value.get("episode_id")
                        task_spec = value.get("task_spec")
                        metrics = value.get("metrics")
                        artifacts = value.get("rollout_artifacts")
                        task_id = task_spec.get("task_id") if isinstance(task_spec, dict) else None
                        if (
                            not isinstance(episode_id, str)
                            or not episode_id
                            or episode_id in isaac_episode_ids
                            or not isinstance(metrics, dict)
                            or float(metrics.get("isaac_backed", 0.0)) <= 0.5
                            or not isinstance(artifacts, dict)
                            or artifacts.get("p4_3_dataset_collection") is not True
                            or bool(artifacts.get("is_p4_full_completion", False))
                        ):
                            valid = False
                        else:
                            isaac_episode_ids.add(episode_id)
                    else:
                        record_id = value.get("record_id")
                        episode_id = value.get("episode_id")
                        task_id = value.get("task_id")
                        masks = value.get("stage_masks")
                        required_mask = expected_mask[shard.dataset_kind]
                        if (
                            not isinstance(record_id, str)
                            or not record_id
                            or record_id in dataset_record_ids
                            or not isinstance(episode_id, str)
                            or not episode_id
                            or episode_id not in set(manifest.source_episode_ids)
                            or value.get("split") != shard.split.value
                            or not isinstance(masks, dict)
                            or masks.get(required_mask) is not True
                            or any(
                                bool(masks.get(name, False))
                                for name in all_mask_names - {required_mask}
                            )
                        ):
                            valid = False
                        else:
                            dataset_record_ids.add(record_id)
                    if task_id not in split_tasks[shard.split]:
                        valid = False
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            valid = False
            continue
        if actual_count != shard.record_count:
            valid = False

    expected_pairs = {
        (kind, split)
        for kind in DatasetKind
        for split in DatasetSplit
    }
    if seen_pairs != expected_pairs:
        valid = False
    if isaac_episode_ids != set(manifest.source_episode_ids):
        valid = False
    for source_value in manifest.source_archive_paths:
        source_path = _evidence_path(manifest_path.parent, source_value)
        if source_path is None or not source_path.is_file() or source_path.stat().st_size == 0:
            valid = False
    if not valid:
        failures.append("dataset_shard_integrity_or_provenance_invalid")
    return valid


def _policy_artifacts(
    directory: Path,
    *,
    required: tuple[str, ...],
    failures: list[str],
    label: str,
) -> tuple[bool, bool]:
    missing = [name for name in required if not (directory / name).is_file()]
    empty = [name for name in required if (directory / name).is_file() and (directory / name).stat().st_size == 0]
    if missing:
        failures.append(f"{label}_artifacts_missing:{','.join(missing)}")
    if empty:
        failures.append(f"{label}_artifacts_empty:{','.join(empty)}")
    checkpoint_valid = _checkpoint_valid(directory / "checkpoint.pt", label=label)
    if not checkpoint_valid and not missing:
        failures.append(f"{label}_checkpoint_invalid")
    fallback = _fallback_available(directory / "fallback_metadata.json")
    return not missing and not empty and checkpoint_valid, fallback


def _checkpoint_valid(path: Path, *, label: str) -> bool:
    if not path.is_file():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            return False
    except Exception:
        return False
    if not isinstance(checkpoint, dict) or not _valid_state_dict(checkpoint.get("state_dict")):
        return False
    if label == "pi_l":
        return _valid_pi_l_checkpoint_metadata(checkpoint)
    if label == "pi_h":
        return _valid_pi_h_checkpoint_metadata(checkpoint)
    if label == "pi_d":
        return _valid_pi_d_checkpoint_metadata(checkpoint)
    return False


def _valid_state_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(key, str) and isinstance(item, torch.Tensor) for key, item in value.items())
    )


def _valid_pi_l_checkpoint_metadata(checkpoint: dict[str, Any]) -> bool:
    fallback = checkpoint.get("deterministic_fallback")
    return (
        checkpoint.get("checkpoint_version") == PI_L_POLICY_CHECKPOINT_VERSION
        and checkpoint.get("model_type") == "TinyPiLDeltaMLP"
        and checkpoint.get("task") == "pi_l_bounded_policy_command_delta_imitation"
        and checkpoint.get("output_mode") == PI_L_OUTPUT_MODE
        and checkpoint.get("feature_names") == list(PI_L_FEATURE_NAMES)
        and checkpoint.get("target_names") == list(PI_L_TARGET_NAMES)
        and _valid_sha256(checkpoint.get("config_hash"))
        and _valid_sha256(checkpoint.get("dataset_sha256"))
        and checkpoint.get("controller_command_output") is False
        and checkpoint.get("actuator_target_output") is False
        and isinstance(fallback, dict)
        and fallback.get("fallback_available") is True
    )


def _valid_pi_h_checkpoint_metadata(checkpoint: dict[str, Any]) -> bool:
    policy_config = checkpoint.get("policy_config")
    return (
        checkpoint.get("model_type") == "P4_3HighLevelRanker"
        and checkpoint.get("training_version") == "p4_3_pi_h_imitation_v1"
        and checkpoint.get("training_stage") == "P4.3c"
        and _valid_sha256(checkpoint.get("training_config_hash"))
        and _valid_sha256(checkpoint.get("dataset_hash"))
        and checkpoint.get("output_contract") == "ContactWrenchTrajectory"
        and checkpoint.get("actuator_command_output") is False
        and checkpoint.get("deterministic_assignment_feasibility_gate") is True
        and checkpoint.get("deterministic_fallback")
        == "P4_2DeterministicGraspCarryPlanner"
        and isinstance(policy_config, dict)
        and _positive_int(policy_config.get("encoder_d_model"))
        and _positive_int(policy_config.get("hidden_dim"))
    )


def _valid_pi_d_checkpoint_metadata(checkpoint: dict[str, Any]) -> bool:
    feature_min = checkpoint.get("feature_min")
    feature_max = checkpoint.get("feature_max")
    expected_width = len(P2_LEARNING_FEATURE_NAMES)
    return (
        checkpoint.get("model_type") == "TinyP2MLP"
        and checkpoint.get("task") == P4_3_PI_D_CHECKPOINT_TASK
        and checkpoint.get("feature_names") == list(P2_LEARNING_FEATURE_NAMES)
        and checkpoint.get("source_of_truth") == "deterministic FeasibilityChecker hard gate"
        and checkpoint.get("inference_contract")
        == "design and deterministic feasibility features only"
        and checkpoint.get("outcome_target_is_inference_feature") is False
        and _valid_sha256(checkpoint.get("training_config_hash"))
        and _valid_sha256(checkpoint.get("dataset_hash"))
        and isinstance(checkpoint.get("training_config"), dict)
        and isinstance(feature_min, list)
        and isinstance(feature_max, list)
        and len(feature_min) == expected_width
        and len(feature_max) == expected_width
        and _valid_feature_bounds(feature_min, feature_max)
    )


def _valid_feature_bounds(lower: list[Any], upper: list[Any]) -> bool:
    pairs = zip(lower, upper, strict=True)
    return all(
        (left_value := _finite_number(left)) is not None
        and (right_value := _finite_number(right)) is not None
        and left_value <= right_value
        for left, right in pairs
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _fallback_available(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    values = [value for key, value in data.items() if "fallback" in key]
    return any(value is True or value == 1 or value == 1.0 for value in values)


def _csv_has_data(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) >= 2


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _validate_pi_l_semantics(
    directory: Path,
    failures: list[str],
    *,
    allowed_task_ids: set[str],
) -> bool:
    metrics = _read_json(directory / "metrics.json") or {}
    online = _read_json(directory / "online_rollout_evaluation.json") or {}
    checkpoint_path = directory / "checkpoint.pt"
    rollout_archive = _evidence_path(
        directory,
        _first_present(online, "archive_path", "rollout_archive_path"),
    )
    checkpoint_hash_matches = _file_hash_matches(
        checkpoint_path,
        online.get("checkpoint_sha256"),
    )
    archive_hash_matches = _file_hash_matches(
        rollout_archive,
        _first_present(online, "archive_sha256", "rollout_archive_sha256"),
    )
    rollout_count = _non_negative_int(online.get("rollout_count"))
    passed_count = _non_negative_int(
        _first_present(online, "rollout_passed_count", "passed_rollout_count")
    )
    archive_evidence_valid = _validate_pi_l_online_archive(
        rollout_archive,
        online,
        allowed_task_ids=allowed_task_ids,
    )
    valid = (
        metrics.get("source_is_real_isaac") is True
        and metrics.get("deterministic_fallback_available") is True
        and metrics.get("controller_authority") == "controller_qp_safety_layer_only"
        and checkpoint_hash_matches
        and archive_hash_matches
        and archive_evidence_valid
        and online.get("schema_version") == "p4_3_pi_l_online_evaluation_v1"
        and online.get("evaluation_type") == "learned_pi_l_online_isaac_rollout"
        and isinstance(online.get("task_ids"), list)
        and bool(online.get("task_ids"))
        and all(isinstance(task_id, str) and task_id for task_id in online["task_ids"])
        and set(online["task_ids"]).issubset(allowed_task_ids)
        and online.get("source_is_real_isaac") is True
        and online.get("isaac_backed") is True
        and online.get("checkpoint_loaded") is True
        and _is_zero(online.get("checkpoint_load_failed_count"))
        and _positive_int(online.get("learned_decision_count"))
        and rollout_count is not None
        and rollout_count > 0
        and passed_count == rollout_count
        and online.get("all_rollouts_passed") is True
        and online.get("controller_qp_safety_layer_used") is True
        and online.get("controller_authority_preserved") is True
        and online.get("controller_active_knot_preserved") is True
        and online.get("learned_policy_command_fields")
        == ["desired_body_twist", "desired_body_position", "residual_wrench_body"]
        and online.get("nonlearned_command_fields_source")
        == "p4_2_deterministic_command"
        and (blend_factor := _finite_number(online.get("runtime_blend_factor"))) is not None
        and 0.0 < blend_factor <= 1.0
        and _positive_int(online.get("overlay_nonzero_count"))
        and (overlay_sum := _finite_number(online.get("overlay_delta_norm_sum"))) is not None
        and overlay_sum > 0.0
        and (overlay_max := _finite_number(online.get("overlay_delta_norm_max"))) is not None
        and overlay_max > 0.0
        and online.get("deterministic_fallback_available") is True
        and online.get("learned_policy_deployed_in_isaac") is True
        and _is_zero(online.get("safety_violation_count"))
        and _is_zero(
            _first_present(online, "object_drop_count", "object_drop_terminal_count")
        )
        and _is_zero(
            _first_present(online, "hard_collision_count", "hard_collision_terminal_count")
        )
        and _is_zero(
            _first_present(
                online,
                "controller_qp_infeasible_terminal_count",
                "qp_infeasible_terminal_count",
                "controller_failure_terminal_count",
            )
        )
        and online.get("p4_full_completion_claim") is False
    )
    if not valid:
        failures.append("pi_l_semantic_or_online_isaac_evidence_invalid")
    return valid


def _validate_pi_h_semantics(directory: Path, failures: list[str]) -> bool:
    metrics = _read_json(directory / "metrics.json") or {}
    evaluation = _read_json(directory / "rollout_evaluation.json") or {}
    valid = (
        _at_least(metrics.get("validation_assignment_feasible_rate"), 1.0)
        and _at_least(metrics.get("validation_exact_selection_rate"), 1.0)
        and _at_least(evaluation.get("schema_valid_rate"), 1.0)
        and _at_least(evaluation.get("assignment_feasible_rate"), 1.0)
        and _is_zero(metrics.get("validation_fallback_rate"))
        and _is_zero(evaluation.get("fallback_count"))
        and _is_zero(evaluation.get("fallback_rate"))
        and evaluation.get("deterministic_safety_gate_used") is True
        and evaluation.get("p4_full_completion_claim") is False
    )
    if not valid:
        failures.append("pi_h_safety_or_evaluation_evidence_invalid")
    return valid


def _validate_pi_l_online_archive(
    path: Path | None,
    online: dict[str, Any],
    *,
    allowed_task_ids: set[str],
) -> bool:
    if path is None or not path.is_file():
        return False
    expected_checkpoint_hash = online.get("checkpoint_sha256")
    expected_rollout_count = _non_negative_int(online.get("rollout_count"))
    expected_passed_count = _non_negative_int(
        _first_present(online, "rollout_passed_count", "passed_rollout_count")
    )
    expected_learned_count = _non_negative_int(online.get("learned_decision_count"))
    expected_fallback_count = _non_negative_int(online.get("fallback_count"))
    expected_overlay_count = _non_negative_int(online.get("overlay_nonzero_count"))
    expected_overlay_sum = _finite_number(online.get("overlay_delta_norm_sum"))
    expected_overlay_max = _finite_number(online.get("overlay_delta_norm_max"))
    expected_blend_factor = _finite_number(online.get("runtime_blend_factor"))
    if any(
        value is None
        for value in (
            expected_rollout_count,
            expected_passed_count,
            expected_learned_count,
            expected_fallback_count,
            expected_overlay_count,
            expected_overlay_sum,
            expected_overlay_max,
            expected_blend_factor,
        )
    ):
        return False
    archive_count = 0
    passed_count = 0
    learned_count = 0
    fallback_count = 0
    overlay_count = 0
    overlay_sum = 0.0
    overlay_max = 0.0
    episode_ids: set[str] = set()
    task_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                archive_count += 1
                value = json.loads(line)
                if not isinstance(value, dict):
                    return False
                episode_id = value.get("episode_id")
                metrics = value.get("metrics")
                task_spec = value.get("task_spec")
                rollout = value.get("rollout_artifacts")
                learning = value.get("learning_artifacts")
                if (
                    not isinstance(episode_id, str)
                    or not episode_id
                    or episode_id in episode_ids
                    or not isinstance(metrics, dict)
                    or not isinstance(task_spec, dict)
                    or not isinstance(rollout, dict)
                    or not isinstance(learning, dict)
                ):
                    return False
                episode_ids.add(episode_id)
                task_id = task_spec.get("task_id")
                if not isinstance(task_id, str) or task_id not in allowed_task_ids:
                    return False
                task_ids.add(task_id)
                metric_learned = _non_negative_int(
                    metrics.get("p4_3_pi_l_learned_decision_count")
                )
                metric_fallback = _non_negative_int(metrics.get("p4_3_pi_l_fallback_count"))
                artifact_learned = _non_negative_int(learning.get("pi_l_learned_decision_count"))
                artifact_fallback = _non_negative_int(learning.get("pi_l_fallback_count"))
                metric_overlay_count = _non_negative_int(
                    metrics.get("p4_3_pi_l_overlay_nonzero_count")
                )
                artifact_overlay_count = _non_negative_int(
                    learning.get("pi_l_overlay_nonzero_count")
                )
                metric_overlay_sum = _finite_number(
                    metrics.get("p4_3_pi_l_overlay_delta_norm_sum")
                )
                artifact_overlay_sum = _finite_number(
                    learning.get("pi_l_overlay_delta_norm_sum")
                )
                metric_overlay_max = _finite_number(
                    metrics.get("p4_3_pi_l_overlay_delta_norm_max")
                )
                artifact_overlay_max = _finite_number(
                    learning.get("pi_l_overlay_delta_norm_max")
                )
                metric_blend = _finite_number(metrics.get("p4_3_pi_l_runtime_blend_factor"))
                artifact_blend = _finite_number(learning.get("pi_l_runtime_blend_factor"))
                overlay_evidence = _recompute_pi_l_overlay_evidence(value)
                if (
                    metric_learned is None
                    or metric_fallback is None
                    or metric_learned != artifact_learned
                    or metric_fallback != artifact_fallback
                    or metric_overlay_count is None
                    or metric_overlay_count != artifact_overlay_count
                    or metric_overlay_sum is None
                    or artifact_overlay_sum is None
                    or not math.isclose(metric_overlay_sum, artifact_overlay_sum)
                    or metric_overlay_max is None
                    or artifact_overlay_max is None
                    or not math.isclose(metric_overlay_max, artifact_overlay_max)
                    or metric_blend is None
                    or artifact_blend is None
                    or not math.isclose(metric_blend, artifact_blend)
                    or not math.isclose(metric_blend, expected_blend_factor)
                    or overlay_evidence is None
                    or metric_overlay_count != overlay_evidence[0]
                    or not math.isclose(metric_overlay_sum, overlay_evidence[1])
                    or not math.isclose(metric_overlay_max, overlay_evidence[2])
                    or float(metrics.get("isaac_backed", 0.0)) <= 0.5
                    or float(metrics.get("object_drop", 0.0)) != 0.0
                    or float(metrics.get("hard_collision", 0.0)) != 0.0
                    or float(metrics.get("controller_qp_infeasible_terminal", 0.0)) != 0.0
                    or float(metrics.get("p4_3_pi_l_checkpoint_load_failed", 0.0)) != 0.0
                    or float(metrics.get("p4_3_pi_l_checkpoint_loaded", 0.0)) <= 0.5
                    or rollout.get("phase") != "P4.3b"
                    or rollout.get("archive_type") != "p4_3_pi_l_online_isaac_evaluation"
                    or rollout.get("p4_3_learned_evaluation") is not True
                    or rollout.get("learning_claim") is not True
                    or rollout.get("learned_policy_success_claim") is not False
                    or bool(rollout.get("is_p4_full_completion", False))
                    or learning.get("stage") != "P4.3b"
                    or learning.get("pi_l_checkpoint_sha256") != expected_checkpoint_hash
                    or learning.get("pi_l_checkpoint_loaded") is not True
                    or learning.get("pi_l_online_inference") is not True
                ):
                    return False
                learned_count += metric_learned
                fallback_count += metric_fallback
                overlay_count += metric_overlay_count
                overlay_sum += metric_overlay_sum
                overlay_max = max(overlay_max, metric_overlay_max)
                if value.get("success") is True and float(metrics.get("success", 0.0)) > 0.5:
                    passed_count += 1
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        archive_count == expected_rollout_count
        and passed_count == expected_passed_count
        and learned_count == expected_learned_count
        and fallback_count == expected_fallback_count
        and overlay_count == expected_overlay_count
        and math.isclose(overlay_sum, expected_overlay_sum)
        and math.isclose(overlay_max, expected_overlay_max)
        and sorted(task_ids) == sorted(online.get("task_ids", []))
    )


def _recompute_pi_l_overlay_evidence(
    archive: dict[str, Any],
) -> tuple[int, float, float] | None:
    rollout = archive.get("rollout_artifacts")
    post_commands = archive.get("policy_commands")
    if not isinstance(rollout, dict) or not isinstance(post_commands, list):
        return None
    pre_commands = rollout.get("p4_3_pi_l_pre_overlay_policy_commands")
    active_knots = rollout.get("p4_3_pi_l_controller_active_knots")
    if (
        not isinstance(pre_commands, list)
        or not isinstance(active_knots, list)
        or not post_commands
        or len(pre_commands) != len(post_commands)
        or len(active_knots) != len(post_commands)
    ):
        return None
    nonlearned_fields = (
        "desired_anchor_pose_offsets",
        "joint_position_bias",
        "joint_velocity_bias",
        "contact_tracking_bias",
        "priority_weights",
    )
    nonzero_count = 0
    norm_sum = 0.0
    norm_max = 0.0
    for pre, post, active_knot in zip(pre_commands, post_commands, active_knots):
        if not isinstance(pre, dict) or not isinstance(post, dict) or not isinstance(active_knot, dict):
            return None
        if any(pre.get(field) != post.get(field) for field in nonlearned_fields):
            return None
        pre_pose = pre.get("desired_body_pose")
        post_pose = post.get("desired_body_pose")
        if (
            isinstance(pre_pose, list)
            and isinstance(post_pose, list)
            and (len(pre_pose) != 7 or len(post_pose) != 7 or pre_pose[3:] != post_pose[3:])
        ):
            return None
        guards = active_knot.get("guard_conditions")
        if not isinstance(guards, list) or not all(isinstance(item, dict) for item in guards):
            return None
        guard_types = {item.get("type") for item in guards}
        if not {"p4_2_phase", "p4_2_attach_gate"}.issubset(guard_types):
            return None
        delta_values: list[float] = []
        for field, width in (("desired_body_twist", 6), ("residual_wrench_body", 6)):
            before = pre.get(field)
            after = post.get(field)
            if before is None and after is None:
                continue
            if not isinstance(before, list) or not isinstance(after, list):
                return None
            if len(before) != width or len(after) != width:
                return None
            delta_values.extend(float(right) - float(left) for left, right in zip(before, after))
        if pre_pose is None and post_pose is None:
            pass
        elif isinstance(pre_pose, list) and isinstance(post_pose, list):
            delta_values.extend(
                float(right) - float(left)
                for left, right in zip(pre_pose[:3], post_pose[:3])
            )
        else:
            return None
        if not all(math.isfinite(value) for value in delta_values):
            return None
        norm = math.sqrt(sum(value * value for value in delta_values))
        norm_sum += norm
        norm_max = max(norm_max, norm)
        if norm > 1.0e-9:
            nonzero_count += 1
    return nonzero_count, norm_sum, norm_max


def _validate_pi_d_semantics(directory: Path, failures: list[str]) -> bool:
    metrics = _read_json(directory / "metrics.json") or {}
    evaluation = _read_json(directory / "rollout_outcome_evaluation.json") or {}
    valid = (
        _at_least(metrics.get("train_unique_target_count"), 2.0)
        and _at_least(metrics.get("train_target_std"), 1.0e-12)
        and _at_least(metrics.get("task_with_multiple_candidates_count"), 1.0)
        and _at_least(metrics.get("train_ranking_pair_count"), 1.0)
        and _at_least(metrics.get("validation_ranking_pair_count"), 1.0)
        and _at_least(metrics.get("validation_pairwise_ranking_accuracy"), 0.5)
        and _valid_sha256(metrics.get("training_config_hash"))
        and _valid_sha256(metrics.get("dataset_hash"))
        and evaluation.get("training_config_hash") == metrics.get("training_config_hash")
        and evaluation.get("dataset_hash") == metrics.get("dataset_hash")
        and evaluation.get("outcome_fields_are_targets_not_inference_features") is True
        and _positive_int(evaluation.get("record_count"))
        and evaluation.get("p4_full_completion_claim") is False
    )
    if not valid:
        failures.append("pi_d_outcome_signal_or_target_leakage_evidence_invalid")
    return valid


def _evidence_path(directory: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for resolved in (candidate, directory / candidate, directory.parent / candidate):
        if resolved.is_file():
            return resolved
    return directory / candidate


def _first_present(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _file_hash_matches(path: Path | None, expected: Any) -> bool:
    if path is None or not path.is_file() or not isinstance(expected, str):
        return False
    if len(expected) != 64:
        return False
    try:
        return hash_file(path) == expected.lower()
    except OSError:
        return False


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _non_negative_int(value: Any) -> int | None:
    converted = _finite_number(value)
    if converted is None or converted < 0.0 or not converted.is_integer():
        return None
    return int(converted)


def _positive_int(value: Any) -> bool:
    converted = _non_negative_int(value)
    return converted is not None and converted > 0


def _is_zero(value: Any) -> bool:
    converted = _finite_number(value)
    return converted is not None and converted == 0.0


def _at_least(value: Any, lower: float) -> bool:
    converted = _finite_number(value)
    return converted is not None and converted >= lower
