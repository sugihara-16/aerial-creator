from __future__ import annotations

import pytest

from amsrr.feasibility.articulated_reachability import (
    ArticulatedIKConfig,
    CentroidalPostureIKConfig,
    _joint_limit_branch_seeds,
    _joint_limits_with_normalized_reserve,
    _joint_regularization_weights,
    _minimum_normalized_joint_limit_reserve,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)


def test_pitch_dock_joints_receive_additional_ik_regularization() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
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
        config.joint_regularization_weight + config.pitch_joint_regularization_weight
    )
    assert weights["module_0:pitch_dock_mech_joint1"] == pytest.approx(pitch_weight)
    assert weights["module_0:pitch_dock_mech_joint2"] == pytest.approx(pitch_weight)
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


def test_joint_limit_reserve_insets_bounds_with_numeric_headroom() -> None:
    physical = {"module_0:yaw": (-2.0, 2.0)}

    unchanged = _joint_limits_with_normalized_reserve(physical, 0.0)
    inset = _joint_limits_with_normalized_reserve(physical, 0.01)

    assert unchanged == physical
    assert inset["module_0:yaw"][0] > -1.96
    assert inset["module_0:yaw"][1] < 1.96
    assert (
        _minimum_normalized_joint_limit_reserve(
            {"module_0:yaw": inset["module_0:yaw"][0]},
            physical,
        )
        > 0.01
    )


def test_joint_limit_branch_seeds_are_center_safe_and_deterministic() -> None:
    limits = {
        "module_0:yaw_1": (-2.0, 2.0),
        "module_0:yaw_2": (-1.0, 3.0),
    }

    first, second = _joint_limit_branch_seeds(limits)

    assert first == {
        "module_0:yaw_1": pytest.approx(-1.0),
        "module_0:yaw_2": pytest.approx(2.0),
    }
    assert second == {
        "module_0:yaw_1": pytest.approx(1.0),
        "module_0:yaw_2": pytest.approx(0.0),
    }


@pytest.mark.parametrize(
    "config_type", [ArticulatedIKConfig, CentroidalPostureIKConfig]
)
def test_ik_configs_reject_invalid_joint_limit_reserve(config_type) -> None:
    with pytest.raises(ValueError, match="minimum_normalized_joint_limit_reserve"):
        config_type(minimum_normalized_joint_limit_reserve=0.5)
