from __future__ import annotations

"""R1 teacher repair for payload rotation observed only under physics."""

import json
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (
    Order9R1RotationStabilityRule,
    Order9R1RotationalTeacherScreenPipelineV7,
    _parse_rule,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_ROTATION_STABILITY_REPAIR_VERSION = "order9_r1_rotation_stability_repair_v1"
ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE = Path(
    "configs/training/order9_r1_rotation_stability_overrides_v4.json"
)


def load_order9_r1_rotation_stability_repairs(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Order9R1RotationStabilityRule]:
    """Load additional rules without mutating the hash-bound v7 rules."""

    repository = Path(repository_root).resolve()
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get(
        "config_version"
    ) != "order9_r1_rotation_stability_overrides_v4" or set(payload) != {
        "config_version",
        "base_overrides",
        "rotation_stability_overrides",
        "failure_evidence",
    }:
        raise SchemaValidationError("R1 rotation repair config is invalid")
    _validate_binding(payload["base_overrides"], repository=repository)
    evidence = payload.get("failure_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "raw_rollout",
        "episode_records",
    }:
        raise SchemaValidationError("R1 rotation repair evidence is invalid")
    for binding in evidence.values():
        _validate_binding(binding, repository=repository)
    raw = payload.get("rotation_stability_overrides")
    if not isinstance(raw, dict) or not raw:
        raise SchemaValidationError("R1 rotation repair rules are missing")
    return {
        source_bucket_id: _parse_rule(source_bucket_id, value)
        for source_bucket_id, value in raw.items()
    }


class Order9R1RotationStabilityRepairPipeline(
    Order9R1RotationalTeacherScreenPipelineV7
):
    """Add evidence-bound, morphology-local grasp choices before heavy work."""

    pipeline_version = ORDER9_R1_ROTATION_STABILITY_REPAIR_VERSION

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        additions = load_order9_r1_rotation_stability_repairs(
            self.repository / ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE,
            repository_root=self.repository,
        )
        overlap = set(additions).intersection(self.rotation_stability_rules)
        if overlap:
            raise SchemaValidationError(
                "R1 rotation repair must not replace an existing v7 rule"
            )
        self.rotation_stability_rules.update(additions)


def _validate_binding(value: Any, *, repository: Path) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise SchemaValidationError("R1 rotation repair binding is invalid")
    path = (repository / str(value.get("path", ""))).resolve()
    if (
        repository not in path.parents
        or not path.is_file()
        or hash_file(path) != value.get("sha256")
    ):
        raise SchemaValidationError("R1 rotation repair binding changed")
    return path


__all__ = [
    "ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE",
    "ORDER9_R1_ROTATION_STABILITY_REPAIR_VERSION",
    "Order9R1RotationStabilityRepairPipeline",
    "load_order9_r1_rotation_stability_repairs",
]
