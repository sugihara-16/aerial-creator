from __future__ import annotations

"""R1 v7 pre-Isaac completion with the payload-rotation admission."""

from pathlib import Path
from typing import Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_clearance_diagnostic import (
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_complete_task_materialization_v6 import (
    finalize_order9_r1_v6_materialized_case,
)
from amsrr.training.order9_r1_grasp_rotation import (
    audit_order9_r1_grasp_rotation_margin,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
)
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (
    ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE,
    load_order9_r1_rotation_stability_rules,
)

ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V7_VERSION = (
    "order9_r1_complete_task_rotation_admitted_v7"
)
ORDER9_R1_GRASP_ROTATION_AUDIT_FILENAME = "grasp_rotation_stability_audit.json"


def finalize_order9_r1_v7_materialized_case(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    task_spec: TaskSpec,
    phase_time_scales: Mapping[str, float],
    joint_rate_limit_rad_s: float,
    repository_root: str | Path,
    lift_clearance_m: float = 0.3,
    retreat_offset_m: float = 0.10,
) -> MaterializedOrder9R1IsaacCase:
    """Apply v6 completion, then fail closed on configured rotation geometry."""

    repository = Path(repository_root).resolve()
    completed = finalize_order9_r1_v6_materialized_case(
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
        raise SchemaValidationError("R1 v7 rotation audit must precede Isaac")
    _, rules = load_order9_r1_rotation_stability_rules(
        repository / ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE,
        repository_root=repository,
    )
    rule = rules.get(completed.manifest.source_bucket_id)
    if rule is None:
        return completed

    artifact_path = _repository_path(
        repository,
        completed.manifest.nominal_artifact.path,
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    artifact_root = artifact_path.parent
    candidates = ContactCandidateSet.from_json(
        (artifact_root / artifact.contact_candidate_set_path).read_text(
            encoding="utf-8"
        )
    )
    lift = next(
        (value for value in artifact.phase_trajectories if value.phase == "lift"),
        None,
    )
    if lift is None:
        raise SchemaValidationError("R1 v7 rotation audit lacks lift trajectory")
    trajectory = ContactWrenchTrajectory.from_json(
        (artifact_root / lift.trajectory_path).read_text(encoding="utf-8")
    )
    audit = audit_order9_r1_grasp_rotation_margin(
        task_spec=task_spec,
        contact_candidate_set=candidates,
        trajectory=trajectory,
        selected_candidate_group_id=artifact.selected_candidate_group_id,
        required_rotation_axis_object=rule.required_rotation_axis_object,
        minimum_force_moment_arm_m=rule.minimum_force_moment_arm_m,
    )
    audit.update(
        {
            "materialization_version": (
                ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V7_VERSION
            ),
            "candidate_id": completed.manifest.candidate_id,
            "source_bucket_id": completed.manifest.source_bucket_id,
            "override_reason": rule.reason,
            "training_eligible": False,
        }
    )
    write_order9_r1_clearance_diagnostic(
        audit,
        root / ORDER9_R1_GRASP_ROTATION_AUDIT_FILENAME,
    )
    if not audit["accepted"]:
        raise SchemaValidationError(
            "R1 grasp rotation admission rejected before Isaac: "
            + ",".join(audit["violation_codes"])
        )
    return completed


def _repository_path(repository: Path, relative: str) -> Path:
    path = (repository / relative).resolve()
    if repository not in path.parents or not path.is_file():
        raise SchemaValidationError("R1 v7 bound path is invalid")
    return path


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V7_VERSION",
    "ORDER9_R1_GRASP_ROTATION_AUDIT_FILENAME",
    "finalize_order9_r1_v7_materialized_case",
]
