from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    POLICY_COMMAND_CONTRACT_LEGACY,
    PolicyCommand,
)


def test_centroidal_policy_command_round_trip_and_legacy_default() -> None:
    command = PolicyCommand(
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        desired_body_pose=(0.2, -0.1, 1.0, 0.0, 0.0, 0.0, 1.0),
        desired_body_twist=[0.0] * 6,
        residual_wrench_body=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        joint_position_targets={"module_1:pitch_dock_mech_joint1": 0.3},
        joint_velocity_targets={"module_1:pitch_dock_mech_joint1": 0.1},
        joint_torque_bias={"module_1:pitch_dock_mech_joint1": 0.4},
    )

    assert PolicyCommand.from_json(command.to_json()).to_dict() == command.to_dict()
    assert PolicyCommand.from_dict({}).control_contract_version == POLICY_COMMAND_CONTRACT_LEGACY


def test_policy_command_rejects_non_finite_local_joint_target() -> None:
    with pytest.raises(SchemaValidationError, match="must be finite"):
        PolicyCommand(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_torque_bias={"module_0:pitch_dock_mech_joint1": float("nan")},
        )
