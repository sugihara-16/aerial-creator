from __future__ import annotations

"""Materialize a proven R1 release-clearance offset into a teacher case."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any

import torch

from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_r1_complete_task_audit import (
    audit_order9_r1_complete_task_semantics,
)
from amsrr.training.order9_r1_isaac_calibration import load_order9_r1_isaac_case
from amsrr.training.order9_r1_release_clearance_repair import (
    ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION,
    apply_order9_r1_release_clearance_repair,
)
from amsrr.training.order9_r1_yaw_branch_repair import (
    _complete_timeline,
    _portable,
    _write_json_text,
)
from amsrr.utils.hashing import hash_file, stable_hash


MATERIALIZATION_VERSION = "order9_r1_release_clearance_materialization_v1"
_SHIFTED_PHASES = frozenset({"place", "release", "retreat", "settle"})


def _candidate_id(case_root: Path) -> str:
    payload = json.loads((case_root / "case_manifest.json").read_text(encoding="utf-8"))
    candidate_id = payload.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("R1 release-clearance candidate identity differs")
    return candidate_id


def derive_order9_r1_release_clearance_case(
    *,
    source_case_root: str | Path,
    destination_case_root: str | Path,
    repository_root: str | Path,
    height_offset_m: float,
    diagnostic_manifest_path: str | Path,
    diagnostic_episodes_path: str | Path,
) -> dict[str, Any]:
    """Clone one case and bind a successful rigid release-height repair."""

    started = perf_counter()
    repository = Path(repository_root).resolve()
    source_root = Path(source_case_root).resolve()
    destination = Path(destination_case_root).resolve()
    diagnostic_manifest = Path(diagnostic_manifest_path).resolve()
    diagnostic_episodes = Path(diagnostic_episodes_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    for path in (diagnostic_manifest, diagnostic_episodes):
        if repository not in path.parents or not path.is_file():
            raise ValueError("R1 release-clearance diagnostic binding differs")
    episodes = [
        json.loads(line)
        for line in diagnostic_episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_id = _candidate_id(source_root)
    if (
        len(episodes) != 2
        or any(
            value.get("task_success") is not True
            or value.get("safety_failure") is not False
            or int(value.get("fallback_decision_count", -1)) != 0
            for value in episodes
        )
    ):
        raise ValueError("R1 release-clearance diagnostic did not pass 2/2")
    manifest_payload = json.loads(diagnostic_manifest.read_text(encoding="utf-8"))
    matching_jobs = [
        value
        for value in manifest_payload.get("jobs", [])
        if isinstance(value, dict) and value.get("name") == candidate_id
    ]
    if (
        len(matching_jobs) != 1
        or abs(float(manifest_payload.get("r1_release_height_offset_m", -1.0)) - float(height_offset_m))
        > 1.0e-12
    ):
        raise ValueError("R1 release-clearance diagnostic manifest differs")

    source_case = load_order9_r1_isaac_case(
        source_root / "case_manifest.json", repository
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(
            source_root,
            temporary,
            ignore=shutil.ignore_patterns("isaac"),
        )
        task = TaskSpec.from_json(
            (temporary / "task_spec.json").read_text(encoding="utf-8")
        )
        set_path = temporary / "nominal_set/manifest.json"
        nominal_set = Order9C3NominalTrajectorySetManifest.from_json(
            set_path.read_text(encoding="utf-8")
        )
        nominal_set.validate()
        if len(nominal_set.entries) != 1:
            raise ValueError("R1 release-clearance set is not singular")
        artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
        artifact_root = artifact_path.parent
        source_artifact_sha256 = hash_file(artifact_path)
        source_phases: dict[str, ContactWrenchTrajectory] = {}
        entry_by_phase = {}
        for entry in artifact.phase_trajectories:
            path = artifact_root / entry.trajectory_path
            source_phases[entry.phase] = ContactWrenchTrajectory.from_json(
                path.read_text(encoding="utf-8")
            )
            entry_by_phase[entry.phase] = entry
        repaired_phases = apply_order9_r1_release_clearance_repair(
            source_phases,
            height_offset_m=float(height_offset_m),
        )
        phase_entries = []
        for phase, repaired in repaired_phases.items():
            entry = entry_by_phase[phase]
            path = artifact_root / entry.trajectory_path
            if phase in _SHIFTED_PHASES:
                _write_json_text(path, repaired.to_dict())
            elif repaired.to_dict() != source_phases[phase].to_dict():
                raise ValueError("R1 release-clearance changed a pre-grasp phase")
            phase_entries.append(
                Order9C3NominalPhaseArtifact(
                    phase=phase,
                    trajectory_path=entry.trajectory_path,
                    trajectory_sha256=hash_file(path),
                    trajectory_hash=stable_hash(repaired.to_dict()),
                    knot_count=len(repaired.knots),
                    generation_method=(
                        entry.generation_method
                        + (
                            "+" + ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION
                            if phase in _SHIFTED_PHASES
                            else ""
                        )
                    ),
                    collision_validation_status="accepted_offline_admission",
                )
            )
        timeline_path = artifact_root / artifact.timeline_path
        _write_json_text(timeline_path, _complete_timeline(repaired_phases))
        source_semantic = audit_order9_r1_complete_task_semantics(
            source_phases,
            task_spec=task,
            lift_clearance_m=0.30,
            retreat_offset_m=0.10,
            materializer_id=MATERIALIZATION_VERSION,
        )
        pose_goals = [
            value
            for value in task.goals
            if value.goal_type == "object_pose" and value.target_pose_world is not None
        ]
        if len(pose_goals) != 1 or pose_goals[0].tolerance_pos_m is None:
            raise ValueError("R1 release-clearance task pose goal differs")
        goal = pose_goals[0]
        terminal_targets = repaired_phases["settle"].knots[-1].object_targets
        if len(terminal_targets) != 1 or terminal_targets[0].pose_target_world is None:
            raise ValueError("R1 release-clearance terminal object target differs")
        terminal_pose = terminal_targets[0].pose_target_world
        goal_pose = goal.target_pose_world
        terminal_position_error_m = math.sqrt(
            sum(
                (float(terminal_pose[index]) - float(goal_pose[index])) ** 2
                for index in range(3)
            )
        )
        if terminal_position_error_m > float(goal.tolerance_pos_m) + 1.0e-12:
            raise ValueError("R1 release-clearance exceeds the task goal tolerance")
        semantic = {
            "status": "accepted",
            "source_complete_task_semantic_audit": source_semantic,
            "repair_transform": "rigid_vertical_place_ramp_and_release_hold",
            "pre_grasp_phase_bytes_preserved": True,
            "joint_path_changed": False,
            "contact_assignments_changed": False,
            "object_grasp_relative_geometry_changed": False,
            "terminal_position_error_m": terminal_position_error_m,
            "task_goal_position_tolerance_m": float(goal.tolerance_pos_m),
            "terminal_pose_within_original_task_goal_tolerance": True,
        }
        provenance = {
            "repair_version": ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION,
            "materialization_version": MATERIALIZATION_VERSION,
            "candidate_id": candidate_id,
            "height_offset_m": float(height_offset_m),
            "source_case_manifest_sha256": hash_file(
                source_root / "case_manifest.json"
            ),
            "source_artifact_sha256": source_artifact_sha256,
            "diagnostic_manifest": {
                "path": _portable(diagnostic_manifest, repository),
                "sha256": hash_file(diagnostic_manifest),
            },
            "diagnostic_episodes": {
                "path": _portable(diagnostic_episodes, repository),
                "sha256": hash_file(diagnostic_episodes),
                "episode_count": 2,
                "success_count": 2,
                "safety_failure_count": 0,
                "fallback_count": 0,
            },
            "pre_grasp_phase_bytes_preserved": True,
            "joint_path_changed": False,
            "contact_assignments_changed": False,
            "object_grasp_relative_geometry_changed": False,
            "acceptance_or_safety_gate_changed": False,
            "ik_invoked": False,
            "trajectory_optimization_invoked": False,
            "controller_layers_invoked_during_materialization": False,
            "training_eligible": False,
        }
        audit_path = temporary / "release_clearance_materialization_audit_v1.json"
        _write_json_text(
            audit_path,
            {
                "audit_version": MATERIALIZATION_VERSION,
                "status": "accepted",
                "candidate_id": candidate_id,
                "complete_task_semantic_audit": semantic,
                "repair_provenance": provenance,
            },
        )
        provenance["materialization_audit"] = {
            "path": _portable(destination / audit_path.name, repository),
            "sha256": hash_file(audit_path),
        }
        selection_evidence = deepcopy(artifact.selection_evidence)
        selection_evidence["r1_release_clearance_repair"] = provenance
        artifact_payload = artifact.to_dict()
        artifact_payload.update(
            {
                "timeline_sha256": hash_file(timeline_path),
                "phase_trajectories": [value.to_dict() for value in phase_entries],
                "selection_evidence": selection_evidence,
                "proxy_collision_validation_status": "enforced_during_generation",
            }
        )
        repaired_artifact = Order9C3NominalTrajectoryArtifact.from_dict(
            artifact_payload
        )
        repaired_artifact.validate()
        _write_json_text(artifact_path, repaired_artifact.to_dict())
        validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)

        nominal_set.entries[0].artifact_sha256 = hash_file(artifact_path)
        nominal_set.metadata.update(
            {
                "r1_release_clearance_repair": True,
                "r1_release_clearance_repair_version": (
                    ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION
                ),
                "r1_release_clearance_height_offset_m": float(height_offset_m),
                "release_clearance_materialization_audit_sha256": hash_file(
                    audit_path
                ),
                "training_eligible": False,
            }
        )
        nominal_set.validate()
        _write_json_text(set_path, nominal_set.to_dict())
        _write_json_text(
            set_path.parent / "collision_validation.json",
            {
                "version": MATERIALIZATION_VERSION,
                "all_accepted": True,
                "records": [
                    {
                        "bucket_id": candidate_id,
                        "accepted": True,
                        "basis": "rigid_upward_release_shift_and_isaac_2_of_2",
                        "materialization_audit_path": _portable(
                            destination / audit_path.name, repository
                        ),
                        "materialization_audit_sha256": hash_file(audit_path),
                    }
                ],
            },
        )
        case_path = temporary / "case_manifest.json"
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        case_payload["nominal_set"].update(
            {
                "path": _portable(
                    destination / "nominal_set/manifest.json", repository
                ),
                "sha256": hash_file(set_path),
            }
        )
        case_payload["nominal_artifact"].update(
            {
                "path": _portable(
                    destination
                    / "nominal_set"
                    / nominal_set.entries[0].artifact_path,
                    repository,
                ),
                "sha256": hash_file(artifact_path),
            }
        )
        case_payload["reset_bank_path"] = _portable(
            destination / "reset_bank.pt", repository
        )
        _write_json_text(case_path, case_payload)
        reset_path = temporary / "reset_bank.pt"
        if not reset_path.is_file():
            raise ValueError("R1 release-clearance source reset bank is absent")
        reset = torch.load(reset_path, map_location="cpu", weights_only=False)
        reference = (reset.get("contract") or {}).get("c3_nominal_reference")
        if not isinstance(reference, dict):
            raise ValueError("R1 release-clearance reset reference is absent")
        reference.update(
            {
                "set_manifest_path": _portable(
                    destination / "nominal_set/manifest.json", repository
                ),
                "set_manifest_sha256": hash_file(set_path),
                "artifact_path": _portable(
                    destination
                    / "nominal_set"
                    / nominal_set.entries[0].artifact_path,
                    repository,
                ),
                "artifact_sha256": hash_file(artifact_path),
                "timeline_sha256": hash_file(timeline_path),
            }
        )
        torch.save(reset, reset_path)
        admission = {
            "admission_version": MATERIALIZATION_VERSION,
            "status": "accepted",
            "candidate_id": candidate_id,
            "height_offset_m": float(height_offset_m),
            "case_manifest_path": _portable(
                destination / "case_manifest.json", repository
            ),
            "case_manifest_sha256": hash_file(case_path),
            "nominal_set_sha256": hash_file(set_path),
            "nominal_artifact_sha256": hash_file(artifact_path),
            "reset_bank_status": "derived_from_source_phase_zero_reset",
            "reset_bank_sha256": hash_file(reset_path),
            "diagnostic_manifest_sha256": hash_file(diagnostic_manifest),
            "diagnostic_episodes_sha256": hash_file(diagnostic_episodes),
            "spatial_path_changed_after_grasp": True,
            "joint_path_changed": False,
            "acceptance_or_safety_gate_changed": False,
            "formal_teacher_collection_authorized": False,
            "training_eligible": False,
            "wall_time_s": perf_counter() - started,
        }
        _write_json_text(
            temporary / "release_clearance_materialization_admission_v1.json",
            admission,
        )
        os.rename(temporary, destination)
        validate_order9_c3_nominal_trajectory_set_bytes(
            destination / "nominal_set/manifest.json",
            repository_root=repository,
            expected_sha256=admission["nominal_set_sha256"],
        )
        load_order9_r1_isaac_case(destination / "case_manifest.json", repository)
        return admission
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "MATERIALIZATION_VERSION",
    "derive_order9_r1_release_clearance_case",
]
