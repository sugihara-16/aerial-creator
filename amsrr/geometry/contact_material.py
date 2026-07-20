from __future__ import annotations

"""Versioned contact-material semantics shared by TaskSpec and candidates.

PhysX combines the two colliding materials.  A contact candidate therefore
must not silently treat the target object's coefficient as the effective
coefficient when the selected robot surface deliberately uses a different
material.  The task metadata below archives the simulator contract without
making candidate generation task-name-specific.
"""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from amsrr.schemas.common import ContactMode, SchemaValidationError


CONTACT_MATERIAL_SEMANTICS_KEY = "contact_material_semantics"
CONTACT_MATERIAL_SEMANTICS_VERSION = "physx_contact_material_semantics_v1"
SUPPORTED_FRICTION_COMBINE_MODES = frozenset(
    {"average", "min", "multiply", "max"}
)
_COMBINE_MODE_CODE = {
    "average": 0.0,
    "min": 1.0,
    "multiply": 2.0,
    "max": 3.0,
}


@dataclass(frozen=True)
class ContactFrictionResolution:
    target_surface_friction: float | None
    robot_surface_friction: float | None
    effective_friction: float | None
    combine_mode: str | None
    combine_mode_code: float
    task_material_applied: bool


def with_selected_robot_contact_material(
    metadata: Mapping[str, Any] | None,
    *,
    target_entity_ids: Sequence[str],
    contact_modes: Sequence[ContactMode | str],
    robot_static_friction: float,
    robot_dynamic_friction: float | None = None,
    friction_combine_mode: str = "max",
    robot_surface_scope: str = "selected_grasp_anchor_surfaces",
) -> dict[str, Any]:
    """Return metadata containing one explicit selected-surface contract."""

    static = _non_negative_friction(
        robot_static_friction,
        "robot_static_friction",
    )
    dynamic = _non_negative_friction(
        static if robot_dynamic_friction is None else robot_dynamic_friction,
        "robot_dynamic_friction",
    )
    mode = _validated_combine_mode(friction_combine_mode)
    entities = sorted({str(value) for value in target_entity_ids if str(value)})
    modes = sorted(
        {
            value.value if isinstance(value, ContactMode) else ContactMode(value).value
            for value in contact_modes
        }
    )
    if not entities:
        raise SchemaValidationError(
            "contact material semantics require at least one target entity"
        )
    if not modes:
        raise SchemaValidationError(
            "contact material semantics require at least one contact mode"
        )
    if not robot_surface_scope:
        raise SchemaValidationError(
            "contact material robot_surface_scope must be non-empty"
        )
    output = dict(metadata or {})
    output[CONTACT_MATERIAL_SEMANTICS_KEY] = {
        "version": CONTACT_MATERIAL_SEMANTICS_VERSION,
        "target_entity_ids": entities,
        "contact_modes": modes,
        "robot_surface_scope": str(robot_surface_scope),
        "robot_static_friction": static,
        "robot_dynamic_friction": dynamic,
        "friction_combine_mode": mode,
    }
    return output


def resolve_contact_friction(
    metadata: Mapping[str, Any] | None,
    *,
    target_entity_id: str,
    contact_mode: ContactMode | str,
    target_surface_friction: float | None,
) -> ContactFrictionResolution:
    """Resolve the effective static coefficient for one candidate.

    Missing metadata preserves the generic pre-Order-9 behavior and uses the
    target surface coefficient.  Once the versioned block is present it is
    validated strictly; malformed physics metadata is not silently ignored.
    """

    target = (
        None
        if target_surface_friction is None
        else _non_negative_friction(
            target_surface_friction,
            "target_surface_friction",
        )
    )
    block = dict(metadata or {}).get(CONTACT_MATERIAL_SEMANTICS_KEY)
    if block is None:
        return ContactFrictionResolution(
            target_surface_friction=target,
            robot_surface_friction=None,
            effective_friction=target,
            combine_mode=None,
            combine_mode_code=-1.0,
            task_material_applied=False,
        )
    if not isinstance(block, Mapping):
        raise SchemaValidationError(
            "TaskSpec contact_material_semantics must be a mapping"
        )
    if block.get("version") != CONTACT_MATERIAL_SEMANTICS_VERSION:
        raise SchemaValidationError(
            "TaskSpec contact material semantics version is unsupported"
        )
    entities = block.get("target_entity_ids")
    modes = block.get("contact_modes")
    if not isinstance(entities, list) or not all(
        isinstance(value, str) and value for value in entities
    ):
        raise SchemaValidationError(
            "contact material target_entity_ids must be non-empty strings"
        )
    if not isinstance(modes, list):
        raise SchemaValidationError(
            "contact material contact_modes must be a list"
        )
    try:
        normalized_modes = {ContactMode(value).value for value in modes}
        normalized_mode = (
            contact_mode.value
            if isinstance(contact_mode, ContactMode)
            else ContactMode(contact_mode).value
        )
    except ValueError as exc:
        raise SchemaValidationError(
            "contact material semantics contain an unsupported contact mode"
        ) from exc
    applies = (
        target_entity_id in set(entities)
        and normalized_mode in normalized_modes
    )
    if not applies:
        return ContactFrictionResolution(
            target_surface_friction=target,
            robot_surface_friction=None,
            effective_friction=target,
            combine_mode=None,
            combine_mode_code=-1.0,
            task_material_applied=False,
        )
    robot = _non_negative_friction(
        block.get("robot_static_friction"),
        "robot_static_friction",
    )
    mode = _validated_combine_mode(block.get("friction_combine_mode"))
    effective = None if target is None else combine_friction(target, robot, mode)
    return ContactFrictionResolution(
        target_surface_friction=target,
        robot_surface_friction=robot,
        effective_friction=effective,
        combine_mode=mode,
        combine_mode_code=_COMBINE_MODE_CODE[mode],
        task_material_applied=True,
    )


def combine_friction(left: float, right: float, mode: str) -> float:
    left_value = _non_negative_friction(left, "left friction")
    right_value = _non_negative_friction(right, "right friction")
    normalized = _validated_combine_mode(mode)
    if normalized == "average":
        return 0.5 * (left_value + right_value)
    if normalized == "min":
        return min(left_value, right_value)
    if normalized == "multiply":
        return left_value * right_value
    return max(left_value, right_value)


def _validated_combine_mode(value: object) -> str:
    mode = str(value)
    if mode not in SUPPORTED_FRICTION_COMBINE_MODES:
        raise SchemaValidationError(
            f"unsupported friction combine mode: {mode!r}"
        )
    return mode


def _non_negative_friction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise SchemaValidationError(f"{name} must be finite and non-negative")
    return parsed


__all__ = [
    "CONTACT_MATERIAL_SEMANTICS_KEY",
    "CONTACT_MATERIAL_SEMANTICS_VERSION",
    "ContactFrictionResolution",
    "SUPPORTED_FRICTION_COMBINE_MODES",
    "combine_friction",
    "resolve_contact_friction",
    "with_selected_robot_contact_material",
]
