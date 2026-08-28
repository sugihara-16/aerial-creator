from __future__ import annotations

"""Hashable contact-geometry rules for R1 nominal-preload collision repair."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from amsrr.schemas.common import SchemaValidationError


ORDER9_R1_PRELOAD_COLLISION_REPAIR_VERSION = (
    "order9_r1_preload_collision_repair_v1"
)

Order9R1PreloadCollisionTeacherOption = tuple[
    tuple[int, int],
    str,
    float,
    float,
    float,
    tuple[float, float, float],
]


@dataclass(frozen=True)
class Order9R1PreloadCollisionRepairRule:
    candidate_id: str
    selected_surface_port_ids: tuple[int, int]
    selected_candidate_group_id: str
    pregrasp_clearance_m: float
    collision_margin_m: float
    grasp_contact_height_offset_m: float
    grasp_contact_tangent_offset_world_m: tuple[float, float, float]
    fallback_teacher_options: tuple[
        Order9R1PreloadCollisionTeacherOption, ...
    ] = ()

    @property
    def teacher_option(
        self,
    ) -> Order9R1PreloadCollisionTeacherOption:
        return (
            self.selected_surface_port_ids,
            self.selected_candidate_group_id,
            self.pregrasp_clearance_m,
            self.collision_margin_m,
            self.grasp_contact_height_offset_m,
            self.grasp_contact_tangent_offset_world_m,
        )

    @property
    def teacher_options(
        self,
    ) -> tuple[Order9R1PreloadCollisionTeacherOption, ...]:
        return (self.teacher_option, *self.fallback_teacher_options)


def load_order9_r1_preload_collision_repair_rules(
    path: str | Path,
) -> tuple[str, dict[str, Order9R1PreloadCollisionRepairRule]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"config_version", "reason", "candidates"}
        or payload.get("config_version")
        != ORDER9_R1_PRELOAD_COLLISION_REPAIR_VERSION
    ):
        raise SchemaValidationError("R1 preload-collision repair fields changed")
    reason = payload.get("reason")
    candidates = payload.get("candidates")
    if (
        not isinstance(reason, str)
        or not reason
        or not isinstance(candidates, dict)
        or not candidates
    ):
        raise SchemaValidationError("R1 preload-collision repair config is empty")
    rules = {
        candidate_id: _parse_rule(candidate_id, value)
        for candidate_id, value in candidates.items()
    }
    if list(candidates) != sorted(candidates):
        raise SchemaValidationError("R1 preload-collision repair order differs")
    return reason, rules


def _parse_rule(
    candidate_id: str, value: object
) -> Order9R1PreloadCollisionRepairRule:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SchemaValidationError("R1 preload-collision candidate id is invalid")
    required = {
        "selected_surface_port_ids",
        "selected_candidate_group_id",
        "pregrasp_clearance_m",
        "collision_margin_m",
    }
    allowed = required | {
        "grasp_contact_height_offset_m",
        "grasp_contact_tangent_offset_world_m",
        "fallback_contact_options",
    }
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or not set(value).issubset(allowed)
    ):
        raise SchemaValidationError("R1 preload-collision candidate fields changed")
    ports = value.get("selected_surface_port_ids")
    group = value.get("selected_candidate_group_id")
    clearance = value.get("pregrasp_clearance_m")
    collision = value.get("collision_margin_m")
    height_offset = value.get("grasp_contact_height_offset_m", 0.0)
    tangent_offset = _parse_tangent_offset(
        value.get("grasp_contact_tangent_offset_world_m", [0.0, 0.0, 0.0])
    )
    if (
        not isinstance(ports, list)
        or len(ports) != 2
        or len({int(item) for item in ports}) != 2
        or min(int(item) for item in ports) < 0
        or not isinstance(group, str)
        or not group
        or isinstance(clearance, bool)
        or not isinstance(clearance, (int, float))
        or not math.isfinite(float(clearance))
        or not 0.08 <= float(clearance) <= 0.30
        or isinstance(collision, bool)
        or not isinstance(collision, (int, float))
        or not math.isfinite(float(collision))
        or not 0.005 <= float(collision) <= 0.030
        or isinstance(height_offset, bool)
        or not isinstance(height_offset, (int, float))
        or not math.isfinite(float(height_offset))
        or not 0.0 <= float(height_offset) <= 0.030
        or math.sqrt(
            float(height_offset) * float(height_offset)
            + sum(item * item for item in tangent_offset)
        )
        > 0.030 + 1.0e-12
    ):
        raise SchemaValidationError("R1 preload-collision candidate rule is invalid")
    primary = (
        tuple(int(item) for item in ports),
        group,
        float(clearance),
        float(collision),
        float(height_offset),
        tangent_offset,
    )
    fallback_raw = value.get("fallback_contact_options", [])
    if not isinstance(fallback_raw, list):
        raise SchemaValidationError(
            "R1 preload-collision fallback options are invalid"
        )
    fallback = tuple(_parse_teacher_option(option) for option in fallback_raw)
    options = (primary, *fallback)
    if len(options) != len(set(options)):
        raise SchemaValidationError(
            "R1 preload-collision teacher options repeat"
        )
    return Order9R1PreloadCollisionRepairRule(
        candidate_id=candidate_id,
        selected_surface_port_ids=primary[0],
        selected_candidate_group_id=group,
        pregrasp_clearance_m=float(clearance),
        collision_margin_m=float(collision),
        grasp_contact_height_offset_m=float(height_offset),
        grasp_contact_tangent_offset_world_m=tangent_offset,
        fallback_teacher_options=fallback,
    )


def _parse_teacher_option(
    value: object,
) -> Order9R1PreloadCollisionTeacherOption:
    required = {
        "selected_surface_port_ids",
        "selected_candidate_group_id",
        "pregrasp_clearance_m",
        "collision_margin_m",
        "grasp_contact_height_offset_m",
    }
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or not set(value).issubset(
            required | {"grasp_contact_tangent_offset_world_m"}
        )
    ):
        raise SchemaValidationError(
            "R1 preload-collision fallback option fields changed"
        )
    ports = value["selected_surface_port_ids"]
    group = value["selected_candidate_group_id"]
    clearance = value["pregrasp_clearance_m"]
    collision = value["collision_margin_m"]
    height_offset = value["grasp_contact_height_offset_m"]
    tangent_offset = _parse_tangent_offset(
        value.get("grasp_contact_tangent_offset_world_m", [0.0, 0.0, 0.0])
    )
    if (
        not isinstance(ports, list)
        or len(ports) != 2
        or len({int(item) for item in ports}) != 2
        or min(int(item) for item in ports) < 0
        or not isinstance(group, str)
        or not group
        or isinstance(clearance, bool)
        or not isinstance(clearance, (int, float))
        or not math.isfinite(float(clearance))
        or not 0.08 <= float(clearance) <= 0.30
        or isinstance(collision, bool)
        or not isinstance(collision, (int, float))
        or not math.isfinite(float(collision))
        or not 0.005 <= float(collision) <= 0.030
        or isinstance(height_offset, bool)
        or not isinstance(height_offset, (int, float))
        or not math.isfinite(float(height_offset))
        or not 0.0 <= float(height_offset) <= 0.030
        or math.sqrt(
            float(height_offset) * float(height_offset)
            + sum(item * item for item in tangent_offset)
        )
        > 0.030 + 1.0e-12
    ):
        raise SchemaValidationError(
            "R1 preload-collision fallback option is invalid"
        )
    return (
        tuple(int(item) for item in ports),
        group,
        float(clearance),
        float(collision),
        float(height_offset),
        tangent_offset,
    )


def _parse_tangent_offset(value: object) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise SchemaValidationError(
            "R1 preload-collision tangent offset is invalid"
        )
    result = tuple(float(item) for item in value)
    if math.sqrt(sum(item * item for item in result)) > 0.030 + 1.0e-12:
        raise SchemaValidationError(
            "R1 preload-collision tangent offset exceeds 30 mm"
        )
    return result  # type: ignore[return-value]


__all__ = [
    "ORDER9_R1_PRELOAD_COLLISION_REPAIR_VERSION",
    "Order9R1PreloadCollisionTeacherOption",
    "Order9R1PreloadCollisionRepairRule",
    "load_order9_r1_preload_collision_repair_rules",
]
