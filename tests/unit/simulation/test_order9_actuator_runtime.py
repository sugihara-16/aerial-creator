from __future__ import annotations

import pytest

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.order9_actuator_runtime import (
    order9_actuator_runtime_values,
    validate_order9_actuator_readback,
)


def test_order9_actuator_runtime_uses_physical_model_and_validates_readback() -> None:
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    values = order9_actuator_runtime_values(physical)
    names = (
        "module_0__gimbal1",
        "module_0__pitch_dock_mech_joint1",
    )

    readback = validate_order9_actuator_readback(
        names,
        (values.gimbal_effort_limit, values.dock_effort_limit),
        (values.gimbal_velocity_limit, values.dock_velocity_limit),
        expected=values,
    )

    assert values.gimbal_effort_limit == pytest.approx(0.76)
    assert values.gimbal_velocity_limit == pytest.approx(3.0)
    assert values.dock_effort_limit == pytest.approx(4.1)
    assert values.dock_velocity_limit == pytest.approx(3.0)
    assert readback["matches_physical_model"] is True


def test_order9_actuator_runtime_rejects_stale_usd_limit() -> None:
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    values = order9_actuator_runtime_values(physical)

    with pytest.raises(RuntimeError, match="gimbal effort limits"):
        validate_order9_actuator_readback(
            ("module_0__gimbal1", "module_0__pitch_dock_mech_joint1"),
            (6.6, values.dock_effort_limit),
            (values.gimbal_velocity_limit, values.dock_velocity_limit),
            expected=values,
        )
