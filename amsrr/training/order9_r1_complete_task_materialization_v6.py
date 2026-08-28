from __future__ import annotations

"""Finalize a private R1 v6 case and reject semantic drift before Isaac.

The C3 artifact writer is hash-bound historical code.  This R1-only layer
therefore operates on a newly materialized, not-yet-executed case: it selects
the R1 complete-task materializer explicitly, persists all eight phases,
proves that the retreat input affects the retreat output, retimes the repaired
artifact, and binds both audits into the case manifests.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_complete_task_audit import (
    ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION,
    ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION,
    ORDER9_R1_COMPLETE_TASK_PHASES,
    audit_order9_r1_complete_task_artifact,
    audit_order9_r1_materializer_parameter_effect,
    write_order9_r1_complete_task_audit,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
    Order9R1IsaacCaseManifest,
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_retime import (
    retime_order9_r1_nominal_artifact,
)
from amsrr.training.order9_r1_nominal_retreat import (
    ORDER9_R1_NOMINAL_RETREAT_VERSION,
    materialize_order9_r1_complete_task_phases,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V6_VERSION = (
    "order9_r1_explicit_complete_task_materialization_v6"
)
_AUDIT_FILENAME = "complete_task_semantic_audit.json"
_PARAMETER_AUDIT_FILENAME = "complete_task_parameter_effect_audit.json"


def finalize_order9_r1_v6_materialized_case(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    task_spec: TaskSpec,
    phase_time_scales: Mapping[str, float],
    joint_rate_limit_rad_s: float,
    repository_root: str | Path,
    lift_clearance_m: float = 0.3,
    retreat_offset_m: float = 0.10,
) -> MaterializedOrder9R1IsaacCase:
    """Repair, audit, retime, and rebind one new R1 case before Isaac."""

    repository = Path(repository_root).resolve()
    root = materialized.manifest_path.parent.resolve()
    if repository not in root.parents or root == repository:
        raise SchemaValidationError("R1 v6 case is outside the repository")
    if (root / "isaac").exists():
        raise SchemaValidationError("R1 v6 complete-task audit must precede Isaac")

    artifact_path = _repository_path(
        repository,
        materialized.manifest.nominal_artifact.path,
    )
    nominal_set_path = _repository_path(
        repository,
        materialized.manifest.nominal_set.path,
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    nominal_set = Order9C3NominalTrajectorySetManifest.from_json(
        nominal_set_path.read_text(encoding="utf-8")
    )
    nominal_set.validate()
    if nominal_set.metadata.get("nominal_retime_admission") is not None:
        raise SchemaValidationError("R1 v6 case was already retimed")

    persisted = _load_phase_trajectories(artifact_path, artifact)
    grasp_phases = {
        phase: persisted[phase] for phase in ORDER9_R1_COMPLETE_TASK_PHASES[:2]
    }
    phase_duration_s = {
        phase: float(persisted[phase].horizon_s)
        for phase in ORDER9_R1_COMPLETE_TASK_PHASES
    }
    parameter_audit = audit_order9_r1_materializer_parameter_effect(
        materialize_order9_r1_complete_task_phases,
        materializer_id=ORDER9_R1_NOMINAL_RETREAT_VERSION,
        phase_trajectories=grasp_phases,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offsets_m=(0.05, float(retreat_offset_m)),
        phase_duration_s=phase_duration_s,
    )
    repaired = materialize_order9_r1_complete_task_phases(
        phase_trajectories=grasp_phases,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=retreat_offset_m,
        phase_duration_s=phase_duration_s,
    )
    _persist_complete_task(
        artifact_path,
        artifact=artifact,
        phases=repaired,
        retreat_offset_m=retreat_offset_m,
    )
    retime_admission = retime_order9_r1_nominal_artifact(
        artifact_path,
        phase_time_scales=phase_time_scales,
        joint_rate_limit_rad_s=joint_rate_limit_rad_s,
    )
    semantic_audit = audit_order9_r1_complete_task_artifact(
        artifact_path,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=retreat_offset_m,
        materializer_id=ORDER9_R1_NOMINAL_RETREAT_VERSION,
    )
    semantic_path = write_order9_r1_complete_task_audit(
        semantic_audit,
        root / _AUDIT_FILENAME,
    )
    parameter_path = write_order9_r1_complete_task_audit(
        parameter_audit,
        root / _PARAMETER_AUDIT_FILENAME,
    )

    _rewrite_collision_admission(
        nominal_set_path.parent / "collision_validation.json",
        candidate_id=materialized.manifest.candidate_id,
        retime_admission=retime_admission.to_dict(),
    )
    _rewrite_nominal_set(
        nominal_set_path,
        nominal_set=nominal_set,
        artifact_path=artifact_path,
        phase_time_scales=phase_time_scales,
        retime_admission=retime_admission.to_dict(),
        semantic_path=semantic_path,
        parameter_path=parameter_path,
    )
    _rewrite_case_manifest(
        materialized.manifest_path,
        manifest=materialized.manifest,
        artifact_path=artifact_path,
        nominal_set_path=nominal_set_path,
    )
    return load_order9_r1_isaac_case(materialized.manifest_path, repository)


def _persist_complete_task(
    manifest_path: Path,
    *,
    artifact: Order9C3NominalTrajectoryArtifact,
    phases: Mapping[str, ContactWrenchTrajectory],
    retreat_offset_m: float,
) -> None:
    if tuple(phases) != ORDER9_R1_COMPLETE_TASK_PHASES:
        raise SchemaValidationError("R1 v6 complete-task phase order changed")
    root = manifest_path.parent
    old_entries = {entry.phase: entry for entry in artifact.phase_trajectories}
    phase_entries = []
    for phase in ORDER9_R1_COMPLETE_TASK_PHASES:
        trajectory = phases[phase]
        trajectory.validate()
        old = old_entries[phase]
        path = root / old.trajectory_path
        path.write_text(trajectory.to_json(indent=2) + "\n", encoding="utf-8")
        phase_entries.append(
            Order9C3NominalPhaseArtifact(
                phase=phase,
                trajectory_path=old.trajectory_path,
                trajectory_sha256=hash_file(path),
                trajectory_hash=stable_hash(trajectory.to_dict()),
                knot_count=len(trajectory.knots),
                generation_method=(
                    f"{old.generation_method}+"
                    f"{ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V6_VERSION}"
                ),
                collision_validation_status=old.collision_validation_status,
            )
        )

    duration_s = sum(
        float(phases[phase].horizon_s) for phase in ORDER9_R1_COMPLETE_TASK_PHASES
    )
    timeline_path = root / artifact.timeline_path
    timeline_path.write_text(
        json.dumps(
            {
                "timeline_version": ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
                "semantic_scope": (
                    "R1 v6 explicit complete eight-phase nominal; checked for "
                    "task semantics and critical-parameter effect before Isaac"
                ),
                "proxy_collision_validation_status": (
                    artifact.proxy_collision_validation_status
                ),
                "duration_s": duration_s,
                "records": _complete_timeline(phases),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = dict(artifact.selection_evidence)
    evidence["r1_complete_task_materialization"] = {
        "materialization_version": (ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V6_VERSION),
        "materializer_id": ORDER9_R1_NOMINAL_RETREAT_VERSION,
        "selection": "explicit_callable_and_identity_v1",
        "retreat_offset_m": float(retreat_offset_m),
        "complete_task_semantic_audit_required_before_isaac": True,
        "parameter_effect_audit_required_before_isaac": True,
        "complete_task_materializer_monkey_patch_used": False,
    }
    payload = artifact.to_dict()
    payload.update(
        {
            "timeline_sha256": hash_file(timeline_path),
            "duration_s": duration_s,
            "phase_trajectories": [entry.to_dict() for entry in phase_entries],
            "selection_evidence": evidence,
        }
    )
    updated = Order9C3NominalTrajectoryArtifact.from_dict(payload)
    updated.validate()
    manifest_path.write_text(updated.to_json(indent=2) + "\n", encoding="utf-8")
    validate_order9_c3_nominal_trajectory_artifact_bytes(manifest_path)


def _rewrite_nominal_set(
    path: Path,
    *,
    nominal_set: Order9C3NominalTrajectorySetManifest,
    artifact_path: Path,
    phase_time_scales: Mapping[str, float],
    retime_admission: Mapping[str, Any],
    semantic_path: Path,
    parameter_path: Path,
) -> None:
    payload = nominal_set.to_dict()
    if len(payload["entries"]) != 1:
        raise SchemaValidationError("R1 v6 case must contain one nominal entry")
    payload["entries"][0]["artifact_sha256"] = hash_file(artifact_path)
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "nominal_phase_time_scales": {
                str(key): float(value) for key, value in phase_time_scales.items()
            },
            "nominal_retime_admission": dict(retime_admission),
            "complete_task_materializer": {
                "materialization_version": (
                    ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V6_VERSION
                ),
                "materializer_id": ORDER9_R1_NOMINAL_RETREAT_VERSION,
                "selection": "explicit_callable_and_identity_v1",
            },
            "complete_task_semantic_audit": _audit_binding(
                semantic_path,
                version=ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION,
            ),
            "complete_task_parameter_effect_audit": _audit_binding(
                parameter_path,
                version=ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION,
            ),
        }
    )
    payload["metadata"] = metadata
    updated = Order9C3NominalTrajectorySetManifest.from_dict(payload)
    updated.validate()
    path.write_text(updated.to_json(indent=2) + "\n", encoding="utf-8")


def _rewrite_case_manifest(
    path: Path,
    *,
    manifest: Order9R1IsaacCaseManifest,
    artifact_path: Path,
    nominal_set_path: Path,
) -> None:
    payload = manifest.to_dict()
    payload["nominal_artifact"]["sha256"] = hash_file(artifact_path)
    payload["nominal_set"]["sha256"] = hash_file(nominal_set_path)
    updated = Order9R1IsaacCaseManifest.from_dict(payload)
    updated.validate()
    path.write_text(updated.to_json(indent=2) + "\n", encoding="utf-8")


def _rewrite_collision_admission(
    path: Path,
    *,
    candidate_id: str,
    retime_admission: Mapping[str, Any],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    matches = (
        [
            record
            for record in records
            if isinstance(record, dict) and record.get("bucket_id") == candidate_id
        ]
        if isinstance(records, list)
        else []
    )
    if len(matches) != 1 or matches[0].get("accepted") is not True:
        raise SchemaValidationError("R1 v6 collision admission is incomplete")
    matches[0]["nominal_retime_admission"] = dict(retime_admission)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_phase_trajectories(
    manifest_path: Path,
    artifact: Order9C3NominalTrajectoryArtifact,
) -> dict[str, ContactWrenchTrajectory]:
    root = manifest_path.parent
    return {
        entry.phase: ContactWrenchTrajectory.from_json(
            (root / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }


def _complete_timeline(
    phases: Mapping[str, ContactWrenchTrajectory],
) -> list[dict[str, Any]]:
    records = []
    offset_s = 0.0
    for phase in ORDER9_R1_COMPLETE_TASK_PHASES:
        trajectory = phases[phase]
        for knot_index, knot in enumerate(trajectory.knots):
            if records and knot_index == 0:
                continue
            records.append(
                {
                    "sample_index": len(records),
                    "global_time_s": offset_s + float(knot.t_rel_s),
                    "phase": phase,
                    "phase_local_time_s": float(knot.t_rel_s),
                    "phase_target_reached": True,
                    "knot": knot.to_dict(),
                }
            )
        offset_s += float(trajectory.horizon_s)
    return records


def _audit_binding(path: Path, *, version: str) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hash_file(path),
        "audit_version": version,
        "status": "accepted",
    }


def _repository_path(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repository / path).resolve()
    if resolved != repository and repository not in resolved.parents:
        raise SchemaValidationError("R1 v6 artifact path escapes repository")
    return resolved


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_MATERIALIZATION_V6_VERSION",
    "finalize_order9_r1_v6_materialized_case",
]
