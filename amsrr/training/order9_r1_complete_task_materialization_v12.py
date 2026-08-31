from __future__ import annotations

"""Finalize R1 and require the accelerated conservative geometry proof."""

from pathlib import Path
from typing import Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_clearance_diagnostic import (
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (
    ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME,
    audit_order9_r1_complete_task_geometry_v12,
    require_order9_r1_complete_task_geometry_v12,
)
from amsrr.training.order9_r1_complete_task_materialization_v7 import (
    finalize_order9_r1_v7_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
    require_order9_r1_support_identity_v11,
)

ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V12_VERSION = (
    "order9_r1_complete_task_geometry_admitted_v12"
)


def finalize_order9_r1_v12_materialized_case(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    task_spec: TaskSpec,
    phase_time_scales: Mapping[str, float],
    joint_rate_limit_rad_s: float,
    repository_root: str | Path,
    geometry_contract: Order9R1NominalGeometryV11Contract,
    lift_clearance_m: float = 0.3,
    retreat_offset_m: float = 0.10,
) -> MaterializedOrder9R1IsaacCase:
    repository = Path(repository_root).resolve()
    require_order9_r1_support_identity_v11(
        task_spec,
        contract=geometry_contract,
    )
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
        raise SchemaValidationError("R1 v12 geometry audit must precede Isaac")
    artifact_path = _repository_path(
        repository,
        completed.manifest.nominal_artifact.path,
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
    audit = audit_order9_r1_complete_task_geometry_v12(
        phases=phases,
        task_spec=task_spec,
        morphology=morphology,
        contact_candidate_set=candidates,
        physical_model_config_path=(repository / "configs/robot/robot_model.yaml"),
        contract=geometry_contract,
    )
    audit.update(
        {
            "materialization_version": (
                ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V12_VERSION
            ),
            "candidate_id": completed.manifest.candidate_id,
            "source_bucket_id": completed.manifest.source_bucket_id,
            "geometry_config_sha256": geometry_contract.config_sha256,
        }
    )
    write_order9_r1_clearance_diagnostic(
        audit,
        root / ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME,
    )
    require_order9_r1_complete_task_geometry_v12(audit)
    return completed


def _repository_path(repository: Path, relative: str) -> Path:
    path = (repository / relative).resolve()
    if repository not in path.parents or not path.is_file():
        raise SchemaValidationError("R1 v12 bound path is invalid")
    return path


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V12_VERSION",
    "finalize_order9_r1_v12_materialized_case",
]
