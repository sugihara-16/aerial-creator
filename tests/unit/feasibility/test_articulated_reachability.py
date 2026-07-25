from __future__ import annotations

import pytest

from amsrr.feasibility.articulated_reachability import (
    ArticulatedIKConfig,
    _joint_regularization_weights,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)


def test_pitch_dock_joints_receive_additional_ik_regularization() -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    joint_ids = (
        "module_0:pitch_dock_mech_joint1",
        "module_0:pitch_dock_mech_joint2",
        "module_0:yaw_dock_mech_joint1",
        "module_0:yaw_dock_mech_joint2",
    )
    config = ArticulatedIKConfig()

    weights = _joint_regularization_weights(
        joint_ids,
        physical_model,
        config,
    )

    pitch_weight = (
        config.joint_regularization_weight
        + config.pitch_joint_regularization_weight
    )
    assert weights["module_0:pitch_dock_mech_joint1"] == pytest.approx(
        pitch_weight
    )
    assert weights["module_0:pitch_dock_mech_joint2"] == pytest.approx(
        pitch_weight
    )
    assert weights["module_0:yaw_dock_mech_joint1"] == pytest.approx(
        config.joint_regularization_weight
    )
    assert weights["module_0:yaw_dock_mech_joint2"] == pytest.approx(
        config.joint_regularization_weight
    )


def test_pitch_joint_regularization_can_be_disabled_but_not_negative() -> None:
    assert (
        ArticulatedIKConfig(
            pitch_joint_regularization_weight=0.0
        ).pitch_joint_regularization_weight
        == 0.0
    )
    with pytest.raises(
        ValueError,
        match="pitch_joint_regularization_weight",
    ):
        ArticulatedIKConfig(pitch_joint_regularization_weight=-1.0)
