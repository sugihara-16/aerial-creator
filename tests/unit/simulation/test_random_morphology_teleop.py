from __future__ import annotations

import math

import pytest

from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.random_morphology_takeoff import (
    RandomMorphologyTakeoffConfig,
    RandomMorphologyTakeoffEnv,
)
from amsrr.simulation.random_morphology_teleop import (
    RandomMorphologyTeleopConfig,
    RandomMorphologyTeleopTarget,
    build_random_morphology_teleop_probe_command,
)


def test_teleop_target_applies_body_yaw_relative_translation_and_rotation() -> None:
    config = RandomMorphologyTeleopConfig(
        translation_step_m=0.10,
        rotation_step_rad=math.radians(5.0),
    )
    hover_pose = (0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0)
    target = RandomMorphologyTeleopTarget.from_hover_pose(
        hover_pose,
        settled_height_m=0.1,
        config=config,
    )

    forward = target.apply_key("w", current_pose_world=hover_pose)
    yaw_left = target.apply_key("j", current_pose_world=hover_pose)
    left_after_yaw = target.apply_key("a", current_pose_world=hover_pose)

    assert forward.action == "forward"
    assert forward.target_pose_world[:3] == pytest.approx((0.1, 0.0, 0.6))
    assert yaw_left.action == "yaw_left"
    assert left_after_yaw.target_pose_world[1] > yaw_left.target_pose_world[1]


def test_teleop_target_bounds_height_attitude_and_position_lead() -> None:
    config = RandomMorphologyTeleopConfig(
        translation_step_m=1.0,
        rotation_step_rad=math.radians(20.0),
        max_roll_pitch_rad=math.radians(25.0),
        max_position_lead_m=0.20,
        minimum_height_above_settled_m=0.15,
    )
    hover_pose = (0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0)
    target = RandomMorphologyTeleopTarget.from_hover_pose(
        hover_pose,
        settled_height_m=0.1,
        config=config,
    )

    moved = target.apply_key("w", current_pose_world=hover_pose)
    for _ in range(3):
        pitched = target.apply_key("i", current_pose_world=hover_pose)
    target.target_pose_world = (0.0, 0.0, 0.26, *pitched.target_pose_world[3:])
    down = target.apply_key("f", current_pose_world=hover_pose)

    assert moved.target_pose_world[0] == pytest.approx(0.20)
    assert abs(pitched.target_pose_world[4]) <= math.sin(math.radians(25.0) / 2.0) + 1.0e-6
    assert down.target_pose_world[2] >= 0.25


def test_teleop_hold_reset_and_quit_controls() -> None:
    hover_pose = (0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0)
    current_pose = (0.2, -0.1, 0.7, 0.0, 0.0, 0.0, 1.0)
    target = RandomMorphologyTeleopTarget.from_hover_pose(
        hover_pose,
        settled_height_m=0.1,
    )

    hold = target.apply_key("h", current_pose_world=current_pose)
    reset = target.apply_key("0", current_pose_world=current_pose)
    quit_update = target.apply_key("q", current_pose_world=current_pose)

    assert hold.target_pose_world == current_pose
    assert reset.target_pose_world == hover_pose
    assert quit_update.quit_requested is True


def test_teleop_probe_command_uses_gui_and_no_learning_flags() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=3,
        module_count=8,
    )
    takeoff_config = RandomMorphologyTakeoffConfig()
    env = RandomMorphologyTakeoffEnv(
        config=takeoff_config,
        backend=IsaacLabBackend(
            load_isaac_lab_backend_config(takeoff_config.backend_config_path)
        ),
        physical_model=physical_model,
    )

    command = build_random_morphology_teleop_probe_command(
        env,
        morphology,
        config=RandomMorphologyTeleopConfig(),
    )

    assert "--random-morphology-takeoff" in command
    assert "--random-morphology-teleop" in command
    assert command[command.index("--viz") + 1] == "kit"
    assert "--realtime-playback" in command
    assert not any("checkpoint" in value or "learned" in value for value in command)
