from __future__ import annotations

"""Persist the corrected exact-margin R1 proof before Isaac."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import Mapping
from unittest.mock import patch

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_complete_task_geometry_v13 import (
    audit_order9_r1_complete_task_geometry_v13,
    require_order9_r1_complete_task_geometry_v13,
)
from amsrr.training import order9_r1_complete_task_materialization_v13 as material_v13
from amsrr.training.order9_r1_complete_task_materialization_v7 import (
    finalize_order9_r1_v7_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    Order9R1SupportClearanceTeacherV12Contract,
    build_order9_r1_support_clearance_generation_certificate,
    require_order9_r1_support_clearance_generation_certificate,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V14_VERSION = (
    "order9_r1_exact_margin_generated_and_admitted_v14"
)
ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME = (
    "complete_task_planning_clearance_audit_v14.json"
)
ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V14_FILENAME = (
    "support_clearance_generation_certificate_v14.json"
)


def promote_order9_r1_v12_materialized_case_v14(
    source: MaterializedOrder9R1IsaacCase,
    *,
    destination: str | Path,
    task_spec: TaskSpec,
    repository_root: str | Path,
    geometry_contract: Order9R1NominalGeometryV11Contract,
    clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
) -> MaterializedOrder9R1IsaacCase:
    """Re-audit an immutable v12 case without repeating trajectory generation."""

    repository = Path(repository_root).resolve()
    target = Path(destination).resolve()
    source_root = source.manifest_path.parent.resolve()
    if repository not in target.parents or repository not in source_root.parents:
        raise SchemaValidationError("R1 v14 reuse path is outside the repository")
    source_certificate_path = (
        source_root / "support_clearance_generation_certificate_v13.json"
    )
    source_certificate = json.loads(source_certificate_path.read_text(encoding="utf-8"))
    require_order9_r1_support_clearance_generation_certificate(
        source_certificate,
        clearance_contract=clearance_contract,
        candidate_id=source.manifest.candidate_id,
    )
    if source_certificate.get("persisted_artifact_checked") is not True:
        raise SchemaValidationError("R1 v12 reuse lacks persisted-artifact proof")

    artifact_path = material_v13._repository_file(
        repository,
        source.manifest.nominal_artifact.path,
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    artifact_root = artifact_path.parent
    phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_root / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
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
        required_robot_support_clearance_m=clearance_contract.planning_clearance_m,
    )
    planning_audit = audit_order9_r1_complete_task_geometry_v13(
        phases=phases,
        task_spec=task_spec,
        morphology=morphology,
        contact_candidate_set=candidates,
        physical_model_config_path=repository / "configs/robot/robot_model.yaml",
        contract=planning_geometry,
    )
    planning_audit.update(
        {
            "materialization_version": (
                ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V14_VERSION
            ),
            "candidate_id": source.manifest.candidate_id,
            "acceptance_clearance_m": clearance_contract.acceptance_clearance_m,
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
            "reused_immutable_v12_materialization": True,
            "source_case_manifest_path": source.manifest_path.relative_to(
                repository
            ).as_posix(),
            "source_case_manifest_sha256": hash_file(source.manifest_path),
            "source_nominal_artifact_path": (source.manifest.nominal_artifact.path),
            "source_nominal_artifact_sha256": hash_file(artifact_path),
            "source_generation_certificate_path": (
                source_certificate_path.relative_to(repository).as_posix()
            ),
            "source_generation_certificate_sha256": hash_file(source_certificate_path),
        }
    )
    require_order9_r1_complete_task_geometry_v13(planning_audit)

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in (
        "case_manifest.json",
        "fast_screen.json",
        "complete_task_semantic_audit.json",
        "complete_task_parameter_effect_audit.json",
        "grasp_rotation_stability_audit.json",
    ):
        source_path = source_root / name
        if source_path.is_file():
            shutil.copy2(source_path, target / name)
    audit_path = target / ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME
    material_v13._write_json(audit_path, planning_audit)
    certificate = build_order9_r1_support_clearance_generation_certificate(
        planning_audit,
        clearance_contract=clearance_contract,
        candidate_id=source.manifest.candidate_id,
        candidate_rank=int(source_certificate["candidate_rank"]),
        evaluated_candidate_count=int(source_certificate["evaluated_candidate_count"]),
    )
    certificate.update(
        {
            "planning_audit_path": audit_path.name,
            "planning_audit_sha256": hash_file(audit_path),
            "planning_geometry_audit_version": planning_audit["audit_version"],
            "persisted_artifact_checked": True,
            "reused_immutable_v12_materialization": True,
            "source_nominal_artifact_path": source.manifest.nominal_artifact.path,
            "source_nominal_artifact_sha256": hash_file(artifact_path),
        }
    )
    certificate_path = target / ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V14_FILENAME
    material_v13._write_json(certificate_path, certificate)
    return load_order9_r1_isaac_case(target / "case_manifest.json", repository)


def finalize_order9_r1_v14_materialized_case(
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
    """Finalize once, then prove the persisted path with v13 geometry."""

    require_order9_r1_support_clearance_generation_certificate(
        clearance_generation_certificate,
        clearance_contract=clearance_contract,
        candidate_id=materialized.manifest.candidate_id,
    )
    repository = Path(repository_root).resolve()
    completed = finalize_order9_r1_v7_materialized_case(
        materialized,
        task_spec=task_spec,
        phase_time_scales=phase_time_scales,
        joint_rate_limit_rad_s=joint_rate_limit_rad_s,
        repository_root=repository,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=retreat_offset_m,
    )
    root = completed.manifest_path.parent.resolve()
    if (root / "isaac").exists():
        raise SchemaValidationError("R1 v14 geometry proof must precede Isaac")
    artifact_path = material_v13._repository_file(
        repository,
        completed.manifest.nominal_artifact.path,
    )
    nominal_set_path = material_v13._repository_file(
        repository,
        completed.manifest.nominal_set.path,
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    artifact_root = artifact_path.parent
    phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_root / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
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
        required_robot_support_clearance_m=clearance_contract.planning_clearance_m,
    )
    planning_audit = audit_order9_r1_complete_task_geometry_v13(
        phases=phases,
        task_spec=task_spec,
        morphology=morphology,
        contact_candidate_set=candidates,
        physical_model_config_path=repository / "configs/robot/robot_model.yaml",
        contract=planning_geometry,
    )
    source_certificate = dict(clearance_generation_certificate or {})
    planning_audit.update(
        {
            "materialization_version": (
                ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V14_VERSION
            ),
            "candidate_id": completed.manifest.candidate_id,
            "acceptance_clearance_m": clearance_contract.acceptance_clearance_m,
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
    require_order9_r1_complete_task_geometry_v13(planning_audit)
    audit_path = root / ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME
    material_v13._write_json(audit_path, planning_audit)

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
            "planning_geometry_audit_version": planning_audit["audit_version"],
            "persisted_artifact_checked": True,
        }
    )
    certificate_path = root / ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V14_FILENAME
    material_v13._write_json(certificate_path, persisted_certificate)

    with patch.object(
        material_v13,
        "ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V13_VERSION",
        ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V14_VERSION,
    ):
        material_v13._bind_certificate_to_artifact(
            artifact_path,
            artifact=artifact,
            certificate=persisted_certificate,
            certificate_path=certificate_path,
        )
        material_v13._rebind_nominal_set(
            nominal_set_path,
            artifact_path=artifact_path,
            audit_path=audit_path,
            certificate_path=certificate_path,
            contract=clearance_contract,
        )
    material_v13._rebind_collision_validation(
        nominal_set_path.parent / "collision_validation.json",
        artifact_path=artifact_path,
    )
    material_v13._rebind_case_manifest(
        completed.manifest_path,
        nominal_set_path=nominal_set_path,
        artifact_path=artifact_path,
    )
    return load_order9_r1_isaac_case(completed.manifest_path, repository)


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V14_VERSION",
    "ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME",
    "ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V14_FILENAME",
    "finalize_order9_r1_v14_materialized_case",
    "promote_order9_r1_v12_materialized_case_v14",
]
