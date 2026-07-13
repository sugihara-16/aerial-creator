from __future__ import annotations

import pytest

from amsrr.controllers.controller_handover import (
    blend_controller_commands,
    merge_disjoint_controller_commands,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerCommand,
    ControllerStatus,
)


def _command(module_id: int, thrust: float) -> ControllerCommand:
    prefix = f"module_{module_id}:"
    return ControllerCommand(
        rotor_thrusts_n={prefix + "rotor": thrust},
        vectoring_joint_targets={prefix + "gimbal": 0.1 * thrust},
        joint_torque_commands={},
        dock_mechanism_commands={prefix + "dock": 0.0},
        joint_position_targets={},
        joint_velocity_targets={},
        joint_torque_bias={prefix + "dock": 0.0},
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
    )


def test_merge_and_blend_preserve_complete_actuator_domain() -> None:
    source = merge_disjoint_controller_commands([_command(0, 2.0), _command(1, 4.0)])
    target = merge_disjoint_controller_commands([_command(0, 6.0), _command(1, 8.0)])

    halfway = blend_controller_commands(source, target, 0.5)

    assert halfway.rotor_thrusts_n == {
        "module_0:rotor": pytest.approx(4.0),
        "module_1:rotor": pytest.approx(6.0),
    }
    assert halfway.controller_status.metrics["handover_alpha"] == pytest.approx(0.5)
    assert blend_controller_commands(source, target, 0.0).to_dict()["rotor_thrusts_n"] == source.rotor_thrusts_n
    assert blend_controller_commands(source, target, 1.0).to_dict()["rotor_thrusts_n"] == target.rotor_thrusts_n


def test_handover_rejects_duplicate_or_changed_actuator_domain() -> None:
    with pytest.raises(SchemaValidationError, match="duplicate"):
        merge_disjoint_controller_commands([_command(0, 1.0), _command(0, 2.0)])

    source = _command(0, 1.0)
    target = _command(0, 2.0)
    target.rotor_thrusts_n["module_0:extra"] = 3.0
    with pytest.raises(SchemaValidationError, match="actuator domain mismatch"):
        blend_controller_commands(source, target, 0.5)


def test_handover_preserves_warning_and_worst_residual() -> None:
    source = _command(0, 1.0)
    target = _command(0, 2.0)
    source.controller_status = ControllerStatus(
        status="warning",
        qp_feasible=True,
        metrics={"allocation_residual_norm": 0.25},
    )
    target.controller_status.metrics["allocation_residual_norm"] = 0.10

    blended = blend_controller_commands(source, target, 0.5)

    assert blended.controller_status.status == "warning"
    assert blended.controller_status.qp_feasible is True
    assert blended.controller_status.metrics["allocation_residual_norm"] == pytest.approx(0.25)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_handover_rejects_invalid_alpha(alpha: float) -> None:
    command = _command(0, 1.0)
    with pytest.raises(SchemaValidationError, match="alpha"):
        blend_controller_commands(command, command, alpha)
