from __future__ import annotations

"""Persist the R1 30 mm generation proof before a case can reach Isaac."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (
    ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME,
    audit_order9_r1_complete_task_geometry_v12,
    require_order9_r1_complete_task_geometry_v12,
)
from amsrr.training.order9_r1_complete_task_materialization_v12 import (
    finalize_order9_r1_v12_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V1,
    Order9R1SupportClearanceTeacherV12Contract,
    build_order9_r1_support_clearance_generation_certificate,
    require_order9_r1_support_clearance_generation_certificate,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V13_VERSION = (
    "order9_r1_support_clearance_generated_and_admitted_v13"
)
ORDER9_R1_PLANNING_CLEARANCE_AUDIT_FILENAME = (
    "complete_task_planning_clearance_audit_v13.json"
)


def finalize_order9_r1_v13_materialized_case(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    task_spec: TaskSpec,
    phase_time_scales: Mapping[str, float],
    joint_rate_limit_rad_s: float,
    repository_root: str | Path,
    geometry_contract: Order9R1NominalGeometryV11Contract,
    clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
    clearance_generation_certificate: Mapping[str, object] | None,
    lift_clearance_m: float = 0.3,
    retreat_offset_m: float = 0.10,
) -> MaterializedOrder9R1IsaacCase:
    """Fail before materialization when the path was only copied and audited."""

    require_order9_r1_support_clearance_generation_certificate(
        clearance_generation_certificate,
        clearance_contract=clearance_contract,
        candidate_id=materialized.manifest.candidate_id,
    )
    repository = Path(repository_root).resolve()
    completed = finalize_order9_r1_v12_materialized_case(
        materialized,
        task_spec=task_spec,
        phase_time_scales=phase_time_scales,
        joint_rate_limit_rad_s=joint_rate_limit_rad_s,
        repository_root=repository,
        geometry_contract=geometry_contract,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=retreat_offset_m,
    )
    root = completed.manifest_path.parent.resolve()
    if (root / "isaac").exists():
        raise SchemaValidationError("R1 v13 clearance proof must precede Isaac")
    artifact_path = _repository_file(
        repository,
        completed.manifest.nominal_artifact.path,
    )
    nominal_set_path = _repository_file(
        repository,
        completed.manifest.nominal_set.path,
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    artifact_root = artifact_path.parent
    phases = {
        value.phase: ContactWrenchTrajectory.from_json(
            (artifact_root / value.trajectory_path).read_text(encoding="utf-8")
        )
        for value in artifact.phase_trajectories
    }
    morphology = MorphologyGraph.from_json(
        (artifact_root / artifact.task_conditioned_morphology_path).read_text(
            encoding="utf-8"
        )
    )
    candidates = ContactCandidateSet.from_json(
        (artifact_root / artifact.contact_candidate_set_path).read_text(
            encoding="utf-8"
        )
    )
    planning_geometry = replace(
        geometry_contract,
        required_robot_support_clearance_m=(clearance_contract.planning_clearance_m),
    )
    planning_audit = audit_order9_r1_complete_task_geometry_v12(
        phases=phases,
        task_spec=task_spec,
        morphology=morphology,
        contact_candidate_set=candidates,
        physical_model_config_path=(repository / "configs/robot/robot_model.yaml"),
        contract=planning_geometry,
    )
    source_certificate = dict(clearance_generation_certificate or {})
    planning_audit.update(
        {
            "materialization_version": (
                ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V13_VERSION
            ),
            "candidate_id": completed.manifest.candidate_id,
            "acceptance_clearance_m": (clearance_contract.acceptance_clearance_m),
            "planning_clearance_m": clearance_contract.planning_clearance_m,
            "planning_buffer_m": clearance_contract.actual_planning_buffer_m,
            "generation_time_clearance_constraint_applied": True,
            "complete_phase_candidate_selection": True,
            "generation_method": source_certificate["generation_method"],
            "vertical_contact_frame_shift_m": source_certificate[
                "vertical_contact_frame_shift_m"
            ],
            "original_deterministic_ik_solution_rigidly_preserved": (
                source_certificate[
                    "original_deterministic_ik_solution_rigidly_preserved"
                ]
            ),
        }
    )
    require_order9_r1_complete_task_geometry_v12(planning_audit)
    audit_path = root / ORDER9_R1_PLANNING_CLEARANCE_AUDIT_FILENAME
    _write_json(audit_path, planning_audit)

    persisted_certificate = build_order9_r1_support_clearance_generation_certificate(
        planning_audit,
        clearance_contract=clearance_contract,
        candidate_id=completed.manifest.candidate_id,
        candidate_rank=int(source_certificate["candidate_rank"]),
        evaluated_candidate_count=int(source_certificate["evaluated_candidate_count"]),
    )
    persisted_certificate.update(
        {
            "planning_audit_path": audit_path.name,
            "planning_audit_sha256": hash_file(audit_path),
            "persisted_artifact_checked": True,
        }
    )
    certificate_path = root / "support_clearance_generation_certificate_v13.json"
    _write_json(certificate_path, persisted_certificate)

    _bind_certificate_to_artifact(
        artifact_path,
        artifact=artifact,
        certificate=persisted_certificate,
        certificate_path=certificate_path,
    )
    _rebind_nominal_set(
        nominal_set_path,
        artifact_path=artifact_path,
        audit_path=audit_path,
        certificate_path=certificate_path,
        contract=clearance_contract,
    )
    _rebind_collision_validation(
        nominal_set_path.parent / "collision_validation.json",
        artifact_path=artifact_path,
    )
    _rebind_case_manifest(
        completed.manifest_path,
        nominal_set_path=nominal_set_path,
        artifact_path=artifact_path,
    )
    return load_order9_r1_isaac_case(completed.manifest_path, repository)


def _bind_certificate_to_artifact(
    path: Path,
    *,
    artifact: Order9C3NominalTrajectoryArtifact,
    certificate: Mapping[str, object],
    certificate_path: Path,
) -> None:
    phase_entries = []
    for entry in artifact.phase_trajectories:
        payload = entry.to_dict()
        payload.update(
            {
                "generation_method": (
                    entry.generation_method
                    + "+"
                    + ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V1
                ),
                "collision_validation_status": "accepted_offline_admission",
            }
        )
        phase_entries.append(Order9C3NominalPhaseArtifact.from_dict(payload))
    payload = artifact.to_dict()
    evidence = dict(payload.get("selection_evidence", {}))
    evidence["r1_support_clearance_generation"] = {
        **dict(certificate),
        "certificate_path": certificate_path.name,
        "certificate_sha256": hash_file(certificate_path),
    }
    payload.update(
        {
            "phase_trajectories": [value.to_dict() for value in phase_entries],
            "selection_evidence": evidence,
            "proxy_collision_validation_status": "enforced_during_generation",
        }
    )
    value = Order9C3NominalTrajectoryArtifact.from_dict(payload)
    value.validate()
    path.write_text(value.to_json(indent=2) + "\n", encoding="utf-8")
    validate_order9_c3_nominal_trajectory_artifact_bytes(path)


def _rebind_nominal_set(
    path: Path,
    *,
    artifact_path: Path,
    audit_path: Path,
    certificate_path: Path,
    contract: Order9R1SupportClearanceTeacherV12Contract,
) -> None:
    value = Order9C3NominalTrajectorySetManifest.from_json(
        path.read_text(encoding="utf-8")
    )
    value.validate()
    if len(value.entries) != 1:
        raise SchemaValidationError("R1 v13 nominal set must contain one entry")
    artifact_sha256 = hash_file(artifact_path)
    value.entries[0].artifact_sha256 = artifact_sha256
    retime = value.metadata.get("nominal_retime_admission")
    if isinstance(retime, dict):
        retime["retimed_artifact_sha256"] = artifact_sha256
    value.metadata["r1_support_clearance_generation"] = {
        "materialization_version": (
            ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V13_VERSION
        ),
        "teacher_contract_sha256": contract.config_sha256,
        "acceptance_clearance_m": contract.acceptance_clearance_m,
        "planning_clearance_m": contract.planning_clearance_m,
        "planning_buffer_m": contract.actual_planning_buffer_m,
        "planning_audit_path": audit_path.name,
        "planning_audit_sha256": hash_file(audit_path),
        "certificate_path": certificate_path.name,
        "certificate_sha256": hash_file(certificate_path),
        "all_eight_phases_generated_under_constraint": True,
        "unsupported_rigid_pose_copy": False,
        "training_eligible": False,
    }
    value.validate()
    path.write_text(value.to_json(indent=2) + "\n", encoding="utf-8")


def _rebind_collision_validation(path: Path, *, artifact_path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_sha256 = hash_file(artifact_path)
    for record in payload.get("records", []):
        retime = record.get("nominal_retime_admission")
        if isinstance(retime, dict):
            retime["retimed_artifact_sha256"] = artifact_sha256
    _write_json(path, payload)


def _rebind_case_manifest(
    path: Path,
    *,
    nominal_set_path: Path,
    artifact_path: Path,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["nominal_artifact"]["sha256"] = hash_file(artifact_path)
    payload["nominal_set"]["sha256"] = hash_file(nominal_set_path)
    _write_json(path, payload)


def _repository_file(repository: Path, relative: str) -> Path:
    path = (repository / relative).resolve()
    if repository not in path.parents or not path.is_file():
        raise SchemaValidationError("R1 v13 bound path is invalid")
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V13_VERSION",
    "ORDER9_R1_PLANNING_CLEARANCE_AUDIT_FILENAME",
    "finalize_order9_r1_v13_materialized_case",
]
