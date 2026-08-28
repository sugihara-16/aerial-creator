from __future__ import annotations

"""R1 v7 teacher admission for payload rotation stability."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_calibration_runner import (
    Order9R1CalibrationCase,
    Order9R1PreparedCandidate,
)
from amsrr.training.order9_r1_early_tilt_pipeline import (
    Order9R1EarlyTiltTeacherScreenPipeline,
    _load_teacher_surface_overrides,
)
from amsrr.training.order9_r1_grasp_rotation import (
    audit_order9_r1_grasp_rotation_margin,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_ROTATIONAL_TEACHER_PIPELINE_V7_VERSION = (
    "order9_r1_rotational_teacher_pipeline_v7"
)
ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE = Path(
    "configs/training/order9_r1_teacher_overrides_v3.json"
)


@dataclass(frozen=True)
class Order9R1RotationStabilityRule:
    source_bucket_id: str
    selected_surface_port_ids: tuple[int, int]
    preferred_candidate_group_ids: tuple[str, ...]
    pregrasp_clearance_m: float
    collision_margin_m: float
    grasp_contact_height_offset_m: float
    required_rotation_axis_object: tuple[float, float, float]
    minimum_force_moment_arm_m: float
    reason: str

    @property
    def candidate_options(
        self,
    ) -> tuple[tuple[tuple[int, int], str, float, float], ...]:
        return tuple(
            (
                self.selected_surface_port_ids,
                group_id,
                self.pregrasp_clearance_m,
                self.collision_margin_m,
                self.grasp_contact_height_offset_m,
            )
            for group_id in self.preferred_candidate_group_ids
        )


class Order9R1RotationalTeacherScreenPipelineV7(Order9R1EarlyTiltTeacherScreenPipeline):
    """Prefer moment-bearing grasps and reject the known zero-arm geometry."""

    pipeline_version = ORDER9_R1_ROTATIONAL_TEACHER_PIPELINE_V7_VERSION

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config_path = (
            self.repository / ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE
        )
        base_path, self.rotation_stability_rules = (
            load_order9_r1_rotation_stability_rules(
                config_path,
                repository_root=self.repository,
            )
        )
        (
            self.teacher_surface_overrides,
            self.teacher_candidate_overrides,
        ) = _load_teacher_surface_overrides(base_path)
        self.rotation_audits: dict[str, dict[str, Any]] = {}

    def prepare(self, case: Order9R1CalibrationCase) -> Order9R1PreparedCandidate:
        rule = self.rotation_stability_rules.get(case.source_bucket.bucket_id)
        if rule is None:
            return super().prepare(case)

        previous = self.teacher_candidate_overrides.get(case.candidate_id)
        inherited = () if previous is None else previous
        self.teacher_candidate_overrides[case.candidate_id] = (
            rule.candidate_options + inherited
        )
        try:
            prepared = super().prepare(case)
        finally:
            if previous is None:
                self.teacher_candidate_overrides.pop(case.candidate_id, None)
            else:
                self.teacher_candidate_overrides[case.candidate_id] = previous
        if not prepared.teacher_trajectory_complete:
            return prepared

        nominal = prepared.payload
        selection = nominal.selection_bundle
        audit = audit_order9_r1_grasp_rotation_margin(
            task_spec=case.task_spec,
            contact_candidate_set=selection.contact_candidate_set,
            trajectory=nominal.windows[-1].plan.trajectory,
            selected_candidate_group_id=(selection.trajectory_plan.candidate_group_id),
            required_rotation_axis_object=rule.required_rotation_axis_object,
            minimum_force_moment_arm_m=rule.minimum_force_moment_arm_m,
        )
        audit.update(
            {
                "candidate_id": case.candidate_id,
                "source_bucket_id": case.source_bucket.bucket_id,
                "teacher_pipeline_version": self.pipeline_version,
                "override_reason": rule.reason,
            }
        )
        self.rotation_audits[case.candidate_id] = audit
        if audit["accepted"]:
            return prepared
        return Order9R1PreparedCandidate(
            candidate_id=prepared.candidate_id,
            candidate_task_hash=prepared.candidate_task_hash,
            morphology_hash=prepared.morphology_hash,
            physical_model_hash=prepared.physical_model_hash,
            teacher_trajectory_complete=False,
            failure_reason=(
                "R1 grasp rotation admission rejected: "
                + ",".join(audit["violation_codes"])
            ),
        )


def load_order9_r1_rotation_stability_rules(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[Path, dict[str, Order9R1RotationStabilityRule]]:
    repository = Path(repository_root).resolve()
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("config_version") != "order9_r1_rotational_teacher_overrides_v3":
        raise SchemaValidationError("R1 rotation override version is invalid")
    if set(payload) != {
        "config_version",
        "base_overrides",
        "rotation_stability_overrides",
        "failure_evidence",
    }:
        raise SchemaValidationError("R1 rotation override fields changed")
    base_path = _validate_binding(
        payload.get("base_overrides"),
        repository=repository,
        label="base overrides",
    )
    evidence = payload.get("failure_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "case_result",
        "episode_records",
        "raw_rollout",
    }:
        raise SchemaValidationError("R1 rotation failure evidence is invalid")
    for label, binding in evidence.items():
        _validate_binding(binding, repository=repository, label=label)

    raw_rules = payload.get("rotation_stability_overrides")
    if not isinstance(raw_rules, dict) or not raw_rules:
        raise SchemaValidationError("R1 rotation overrides must be non-empty")
    rules = {
        source_bucket_id: _parse_rule(source_bucket_id, value)
        for source_bucket_id, value in raw_rules.items()
    }
    return base_path, rules


def _parse_rule(source_bucket_id: str, value: Any) -> Order9R1RotationStabilityRule:
    if not isinstance(source_bucket_id, str) or not source_bucket_id:
        raise SchemaValidationError("R1 rotation source bucket id is invalid")
    required_fields = {
        "selected_surface_port_ids",
        "preferred_candidate_group_ids",
        "pregrasp_clearance_m",
        "collision_margin_m",
        "required_rotation_axis_object",
        "minimum_force_moment_arm_m",
        "reason",
    }
    allowed_fields = required_fields | {"grasp_contact_height_offset_m"}
    if (
        not isinstance(value, dict)
        or not required_fields.issubset(value)
        or not set(value).issubset(allowed_fields)
    ):
        raise SchemaValidationError("R1 rotation rule fields changed")
    ports = value["selected_surface_port_ids"]
    groups = value["preferred_candidate_group_ids"]
    axis = value["required_rotation_axis_object"]
    clearance = float(value["pregrasp_clearance_m"])
    collision = float(value["collision_margin_m"])
    height_offset = float(value.get("grasp_contact_height_offset_m", 0.0))
    minimum_arm = float(value["minimum_force_moment_arm_m"])
    reason = value["reason"]
    if (
        not isinstance(ports, list)
        or len(ports) != 2
        or len({int(item) for item in ports}) != 2
        or min(int(item) for item in ports) < 0
        or not isinstance(groups, list)
        or not groups
        or any(not isinstance(item, str) or not item for item in groups)
        or len(set(groups)) != len(groups)
        or not isinstance(axis, list)
        or len(axis) != 3
        or not all(math.isfinite(float(item)) for item in axis)
        or sum(float(item) ** 2 for item in axis) <= 0.0
        or not 0.08 <= clearance <= 0.30
        or not 0.005 <= collision <= 0.030
        or not 0.0 <= height_offset <= 0.10
        or not math.isfinite(minimum_arm)
        or minimum_arm <= 0.0
        or not isinstance(reason, str)
        or not reason
    ):
        raise SchemaValidationError("R1 rotation rule is invalid")
    return Order9R1RotationStabilityRule(
        source_bucket_id=source_bucket_id,
        selected_surface_port_ids=tuple(int(item) for item in ports),
        preferred_candidate_group_ids=tuple(groups),
        pregrasp_clearance_m=clearance,
        collision_margin_m=collision,
        grasp_contact_height_offset_m=height_offset,
        required_rotation_axis_object=tuple(float(item) for item in axis),
        minimum_force_moment_arm_m=minimum_arm,
        reason=reason,
    )


def _validate_binding(
    value: Any,
    *,
    repository: Path,
    label: str,
) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise SchemaValidationError(f"R1 rotation {label} binding is invalid")
    relative = value.get("path")
    expected = value.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(expected, str)
        or len(expected) != 64
    ):
        raise SchemaValidationError(f"R1 rotation {label} binding is invalid")
    path = (repository / relative).resolve()
    if repository not in path.parents or not path.is_file():
        raise SchemaValidationError(f"R1 rotation {label} path is invalid")
    if hash_file(path) != expected:
        raise SchemaValidationError(f"R1 rotation {label} bytes changed")
    return path


__all__ = [
    "ORDER9_R1_ROTATIONAL_TEACHER_PIPELINE_V7_VERSION",
    "ORDER9_R1_ROTATIONAL_TEACHER_OVERRIDES_V3_RELATIVE",
    "Order9R1RotationStabilityRule",
    "Order9R1RotationalTeacherScreenPipelineV7",
    "load_order9_r1_rotation_stability_rules",
]
