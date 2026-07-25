from __future__ import annotations

import pytest

from amsrr.controllers.load_limited_contact_preload import (
    LoadLimitedContactPreload,
    LoadLimitedContactPreloadConfig,
    load_limited_velocity_targets,
)


def _controller() -> LoadLimitedContactPreload:
    controller = LoadLimitedContactPreload(
        LoadLimitedContactPreloadConfig(
            maximum_speed_rad_s=0.002,
            load_threshold_nm=1.2,
            load_dwell_s=0.10,
        )
    )
    controller.start(
        ordered_joint_ids=("shared", "left", "right"),
        closure_velocity_targets_rad_s={
            "shared": 1.0,
            "left": -2.0,
            "right": 2.0,
        },
        joint_ids_by_anchor={
            0: ("shared", "left"),
            1: ("shared", "right"),
        },
        initial_position_targets_rad={
            "shared": 0.10,
            "left": 0.20,
            "right": -0.20,
        },
    )
    return controller


def test_preload_integrates_previous_absolute_target_and_freezes_each_side() -> None:
    controller = _controller()

    for _ in range(5):
        output = controller.step(
            applied_joint_load_nm={
                "shared": 0.2,
                "left": 1.3,
                "right": 0.4,
            },
            dt_s=0.02,
        )

    assert output.complete is False
    assert output.frozen_anchor_ids == (0,)
    assert output.velocity_targets_rad_s == {
        "shared": 0.0,
        "left": 0.0,
        "right": 0.002,
    }
    # The target lead accumulates from the prior target; it is not rebased to
    # a measured joint position on each controller step.
    assert output.position_targets_rad["left"] == pytest.approx(0.19984)
    assert output.position_targets_rad["right"] == pytest.approx(-0.1998)

    for _ in range(5):
        output = controller.step(
            applied_joint_load_nm={
                "shared": 0.2,
                "left": 1.3,
                "right": 1.3,
            },
            dt_s=0.02,
        )

    assert output.complete is True
    assert output.frozen_anchor_ids == (0, 1)
    assert output.velocity_targets_rad_s == {
        "shared": 0.0,
        "left": 0.0,
        "right": 0.0,
    }
    assert controller.hold().position_targets_rad == output.position_targets_rad


def test_preload_load_dwell_is_contiguous() -> None:
    controller = _controller()
    for load in (1.3, 1.3, 0.5, 1.3):
        output = controller.step(
            applied_joint_load_nm={
                "shared": 0.2,
                "left": load,
                "right": 0.2,
            },
            dt_s=0.02,
        )

    assert output.load_dwell_s_by_anchor[0] == pytest.approx(0.02)
    assert output.frozen_anchor_ids == ()


def test_shared_joint_freezes_with_first_completed_owner() -> None:
    velocities = load_limited_velocity_targets(
        ordered_joint_ids=("shared", "left", "right"),
        closure_velocity_targets_rad_s={
            "shared": 1.0,
            "left": -2.0,
            "right": 2.0,
        },
        joint_ids_by_anchor={
            0: ("shared", "left"),
            1: ("shared", "right"),
        },
        frozen_anchor_ids={0},
        maximum_speed_rad_s=0.002,
    )

    assert velocities == {
        "shared": 0.0,
        "left": 0.0,
        "right": 0.002,
    }
