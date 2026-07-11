from __future__ import annotations

import json
from pathlib import Path

from amsrr.acceptance.p4_3_acceptance import P4_3AcceptanceReport
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.schemas.datasets import P4_3DatasetManifest
from amsrr.schemas.task_spec import TaskSpec
from amsrr.utils.hashing import hash_file, stable_hash


def write_p4_3_learning_summary_archive(
    *,
    source_rollout_archive_path: str | Path,
    output_path: str | Path,
    dataset_manifest_path: str | Path,
    pi_l_dir: str | Path,
    pi_h_dir: str | Path,
    pi_d_dir: str | Path,
    acceptance: P4_3AcceptanceReport,
) -> EpisodeArchive:
    if not acceptance.completion_passed:
        raise ValueError("P4.3 learning summary requires a passed acceptance report")
    source_path = Path(source_rollout_archive_path)
    with source_path.open("r", encoding="utf-8") as handle:
        first_line = next((line for line in handle if line.strip()), None)
    if first_line is None:
        raise ValueError("P4.3 learning summary requires a source rollout archive")
    archive = EpisodeArchive.from_json(first_line)
    artifact_dirs = {
        "pi_l": Path(pi_l_dir),
        "pi_h": Path(pi_h_dir),
        "pi_d": Path(pi_d_dir),
    }
    checkpoints = {
        name: {
            "path": str(directory / "checkpoint.pt"),
            "sha256": hash_file(directory / "checkpoint.pt"),
        }
        for name, directory in artifact_dirs.items()
    }
    online_evaluation_path = artifact_dirs["pi_l"] / "online_rollout_evaluation.json"
    online_evaluation = json.loads(online_evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(online_evaluation, dict):
        raise ValueError("P4.3 pi_L online evaluation must be a JSON object")
    online_archive_value = online_evaluation.get("archive_path")
    if not isinstance(online_archive_value, str) or not online_archive_value:
        raise ValueError("P4.3 pi_L online evaluation must reference its rollout archive")
    online_archive_path = Path(online_archive_value)
    if not online_archive_path.is_file():
        raise ValueError("P4.3 pi_L online rollout archive is missing")
    manifest_path = Path(dataset_manifest_path)
    manifest = P4_3DatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    if archive.episode_id not in manifest.source_episode_ids:
        raise ValueError("P4.3 learning summary source episode is not in the dataset manifest")
    if str(source_path) not in manifest.source_archive_paths:
        raise ValueError("P4.3 learning summary source archive is not in the dataset manifest")
    source_episode_id = archive.episode_id
    source_config_hash = archive.config_hash
    task_payload = archive.task_spec.to_dict()
    task_metadata = task_payload.setdefault("metadata", {})
    task_metadata.update(
        {
            "p4_phase": "P4.3",
            "p4_3_learning_bootstrap": True,
            "p4_3_minimum_learning_summary": True,
            "source_rollout_task_id": archive.task_spec.task_id,
        }
    )
    archive.task_spec = TaskSpec.from_dict(task_payload)
    archive.task_hash = archive.task_spec.stable_hash()
    archive.episode_id = f"p4-3-learning-summary-{stable_hash(checkpoints)[:12]}"
    artifact_hashes = {
        "pi_l_metrics": hash_file(artifact_dirs["pi_l"] / "metrics.json"),
        "pi_l_reward_curve": hash_file(artifact_dirs["pi_l"] / "reward_curve.csv"),
        "pi_l_fallback": hash_file(artifact_dirs["pi_l"] / "fallback_metadata.json"),
        "pi_h_metrics": hash_file(artifact_dirs["pi_h"] / "metrics.json"),
        "pi_h_evaluation": hash_file(artifact_dirs["pi_h"] / "rollout_evaluation.json"),
        "pi_h_fallback": hash_file(artifact_dirs["pi_h"] / "fallback_metadata.json"),
        "pi_d_metrics": hash_file(artifact_dirs["pi_d"] / "metrics.json"),
        "pi_d_evaluation": hash_file(
            artifact_dirs["pi_d"] / "rollout_outcome_evaluation.json"
        ),
        "pi_d_fallback": hash_file(artifact_dirs["pi_d"] / "fallback_metadata.json"),
    }
    archive.learning_artifacts = {
        "phase": "P4.3",
        "minimum_learning_run": True,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": hash_file(manifest_path),
        "source_rollout_archive_path": str(source_path),
        "source_rollout_archive_sha256": hash_file(source_path),
        "source_rollout_episode_id": source_episode_id,
        "checkpoints": checkpoints,
        "artifact_hashes": artifact_hashes,
        "pi_l_metrics_path": str(artifact_dirs["pi_l"] / "metrics.json"),
        "pi_l_reward_curve_path": str(artifact_dirs["pi_l"] / "reward_curve.csv"),
        "pi_l_online_evaluation_path": str(online_evaluation_path),
        "pi_l_online_evaluation_sha256": hash_file(online_evaluation_path),
        "pi_l_online_rollout_archive_path": str(online_archive_path),
        "pi_l_online_rollout_archive_sha256": hash_file(online_archive_path),
        "pi_h_metrics_path": str(artifact_dirs["pi_h"] / "metrics.json"),
        "pi_h_rollout_evaluation_path": str(artifact_dirs["pi_h"] / "rollout_evaluation.json"),
        "pi_d_metrics_path": str(artifact_dirs["pi_d"] / "metrics.json"),
        "pi_d_rollout_outcome_evaluation_path": str(
            artifact_dirs["pi_d"] / "rollout_outcome_evaluation.json"
        ),
        "deterministic_pi_d_fallback": True,
        "deterministic_pi_h_fallback": True,
        "deterministic_pi_l_fallback": True,
        "learned_policy_online_isaac_evaluation": True,
        "pi_l_deployed_in_isaac": True,
        "pi_h_deployed_in_isaac": False,
        "pi_d_deployed_in_isaac": False,
        "pi_h_evaluation_mode": "offline_teacher_record_decode",
        "pi_d_evaluation_mode": "offline_held_out_outcome_regression",
        "acceptance": acceptance.to_dict(),
    }
    archive.rollout_artifacts = {
        "phase": "P4.3",
        "archive_type": "p4_3_minimum_learning_summary",
        "p4_3_dataset_collection": False,
        "p4_3_learned_evaluation": False,
        "p4_3_minimum_learning_run": True,
        "learning_claim": True,
        "learned_policy_success_claim": False,
        "pi_l_deployed_in_isaac": True,
        "pi_h_deployed_in_isaac": False,
        "pi_d_deployed_in_isaac": False,
        "is_p4_full_completion": False,
        "physical_success_claim": False,
        "high_fidelity_natural_grasp_success_claim": False,
    }
    archive.trajectory_records = []
    archive.policy_commands = []
    archive.controller_commands = []
    archive.runtime_observations = []
    archive.actuator_target_records = []
    archive.rewards = []
    archive.metrics = {
        "p4_3_learning_bootstrap": 1.0,
        "p4_3_minimum_learning_completion": 1.0,
        "source_episode_count": float(len(manifest.source_episode_ids)),
        "p4_3_pi_l_checkpoint_loaded": 1.0,
        "p4_3_pi_l_online_inference": 1.0,
        "p4_3_pi_l_learned_decision_count": float(
            online_evaluation.get("learned_decision_count", 0)
        ),
        "p4_3_pi_l_fallback_count": float(online_evaluation.get("fallback_count", 0)),
        "p4_3_pi_l_overlay_nonzero_count": float(
            online_evaluation.get("overlay_nonzero_count", 0)
        ),
        "object_drop": float(online_evaluation.get("object_drop_count", 0)),
        "hard_collision": float(online_evaluation.get("hard_collision_count", 0)),
        "controller_qp_infeasible_terminal": float(
            online_evaluation.get("controller_qp_infeasible_terminal_count", 0)
        ),
        "learned_policy_success_claim": 0.0,
        "physical_success_claim": 0.0,
        "p4_full_completion": 0.0,
    }
    archive.success = True
    archive.failure_reason = None
    archive.reproducibility = {
        "phase": "P4.3",
        "source_config_hash": source_config_hash,
        "dataset_manifest_sha256": archive.learning_artifacts["dataset_manifest_sha256"],
        "pi_l_online_evaluation_sha256": archive.learning_artifacts[
            "pi_l_online_evaluation_sha256"
        ],
    }
    archive.config_hash = stable_hash(
        {
            "source_config_hash": source_config_hash,
            "dataset_manifest_sha256": archive.learning_artifacts["dataset_manifest_sha256"],
            "checkpoints": checkpoints,
            "artifact_hashes": artifact_hashes,
            "pi_l_online_evaluation_sha256": archive.learning_artifacts[
                "pi_l_online_evaluation_sha256"
            ],
            "source_rollout_archive_sha256": archive.learning_artifacts[
                "source_rollout_archive_sha256"
            ],
        }
    )
    archive.reproducibility["config_hash"] = archive.config_hash
    write_episode_archives_jsonl(output_path, [archive])
    if not validate_p4_3_learning_summary_archive(output_path):
        raise ValueError("P4.3 learning summary failed its post-write integrity check")
    return archive


def validate_p4_3_learning_summary_archive(path: str | Path) -> bool:
    summary_path = Path(path)
    try:
        lines = [line for line in summary_path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 1:
            return False
        archive = EpisodeArchive.from_json(lines[0])
        rollout = archive.rollout_artifacts
        learning = archive.learning_artifacts
        if (
            rollout.get("phase") != "P4.3"
            or rollout.get("archive_type") != "p4_3_minimum_learning_summary"
            or rollout.get("p4_3_minimum_learning_run") is not True
            or rollout.get("p4_3_dataset_collection") is not False
            or rollout.get("is_p4_full_completion") is not False
            or rollout.get("pi_l_deployed_in_isaac") is not True
            or rollout.get("pi_h_deployed_in_isaac") is not False
            or rollout.get("pi_d_deployed_in_isaac") is not False
            or archive.task_spec.metadata.get("p4_phase") != "P4.3"
            or archive.task_spec.metadata.get("p4_3_learning_bootstrap") is not True
            or archive.metrics.get("p4_3_minimum_learning_completion") != 1.0
            or archive.metrics.get("p4_3_pi_l_checkpoint_loaded") != 1.0
            or archive.metrics.get("p4_3_pi_l_online_inference") != 1.0
            or archive.metrics.get("p4_full_completion") != 0.0
            or archive.reproducibility.get("config_hash") != archive.config_hash
            or not archive.success
            or archive.trajectory_records
            or archive.policy_commands
            or archive.controller_commands
            or archive.runtime_observations
            or archive.actuator_target_records
            or archive.rewards
        ):
            return False
        manifest_path = Path(str(learning.get("dataset_manifest_path", "")))
        online_path = Path(str(learning.get("pi_l_online_evaluation_path", "")))
        online_archive_path = Path(str(learning.get("pi_l_online_rollout_archive_path", "")))
        if (
            hash_file(manifest_path) != learning.get("dataset_manifest_sha256")
            or hash_file(online_path) != learning.get("pi_l_online_evaluation_sha256")
            or hash_file(online_archive_path)
            != learning.get("pi_l_online_rollout_archive_sha256")
        ):
            return False
        checkpoints = learning.get("checkpoints")
        if not isinstance(checkpoints, dict) or set(checkpoints) != {"pi_l", "pi_h", "pi_d"}:
            return False
        for checkpoint in checkpoints.values():
            if not isinstance(checkpoint, dict):
                return False
            if hash_file(Path(str(checkpoint.get("path", "")))) != checkpoint.get("sha256"):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True
