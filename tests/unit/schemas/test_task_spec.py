from __future__ import annotations

import pytest

from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec, TaskType


def test_task_spec_parse_grasp_carry_yaml(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)

    assert task.task_id == "grasp_carry_box_001"
    assert task.task_type == TaskType.OBJECT_GRASP_CARRY
    assert task.scene.objects[0].allowed_contact_modes == [
        ContactMode.GRASP,
        ContactMode.SUPPORT,
        ContactMode.PUSH,
    ]
    assert task.robot_constraints.min_modules == 2
    assert task.safety.max_contact_torque_nm == 5.0


def test_task_spec_rejects_missing_grasp_carry_mass(grasp_carry_dict: dict) -> None:
    grasp_carry_dict["scene"]["objects"][0]["mass_kg"] = None

    with pytest.raises(SchemaValidationError, match="mass_kg"):
        TaskSpec.from_dict(grasp_carry_dict)

