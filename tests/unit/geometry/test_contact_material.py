from __future__ import annotations

import pytest

from amsrr.geometry.contact_material import (
    CONTACT_MATERIAL_SEMANTICS_KEY,
    CONTACT_MATERIAL_SEMANTICS_VERSION,
    combine_friction,
    resolve_contact_friction,
    with_selected_robot_contact_material,
)
from amsrr.schemas.common import ContactMode, SchemaValidationError


def test_selected_surface_material_round_trips_and_resolves_max_combine() -> None:
    metadata = with_selected_robot_contact_material(
        {"owner": "unit-test"},
        target_entity_ids=["payload"],
        contact_modes=[ContactMode.GRASP],
        robot_static_friction=4.5,
        friction_combine_mode="max",
    )

    resolution = resolve_contact_friction(
        metadata,
        target_entity_id="payload",
        contact_mode=ContactMode.GRASP,
        target_surface_friction=0.6,
    )

    assert metadata[CONTACT_MATERIAL_SEMANTICS_KEY]["version"] == (
        CONTACT_MATERIAL_SEMANTICS_VERSION
    )
    assert resolution.task_material_applied is True
    assert resolution.effective_friction == pytest.approx(4.5)
    assert resolution.combine_mode == "max"
    assert resolution.combine_mode_code == 3.0


def test_material_contract_is_scoped_by_entity_and_contact_mode() -> None:
    metadata = with_selected_robot_contact_material(
        {},
        target_entity_ids=["payload"],
        contact_modes=[ContactMode.GRASP],
        robot_static_friction=4.5,
    )

    support = resolve_contact_friction(
        metadata,
        target_entity_id="payload",
        contact_mode=ContactMode.SUPPORT,
        target_surface_friction=0.6,
    )
    other_object = resolve_contact_friction(
        metadata,
        target_entity_id="other",
        contact_mode=ContactMode.GRASP,
        target_surface_friction=0.7,
    )

    assert support.task_material_applied is False
    assert support.effective_friction == pytest.approx(0.6)
    assert other_object.task_material_applied is False
    assert other_object.effective_friction == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("average", 0.5), ("min", 0.2), ("multiply", 0.16), ("max", 0.8)],
)
def test_supported_physx_friction_combine_modes(mode: str, expected: float) -> None:
    assert combine_friction(0.2, 0.8, mode) == pytest.approx(expected)


def test_malformed_versioned_material_contract_fails_closed() -> None:
    with pytest.raises(SchemaValidationError):
        resolve_contact_friction(
            {CONTACT_MATERIAL_SEMANTICS_KEY: {"version": "unknown"}},
            target_entity_id="payload",
            contact_mode=ContactMode.GRASP,
            target_surface_friction=0.6,
        )
