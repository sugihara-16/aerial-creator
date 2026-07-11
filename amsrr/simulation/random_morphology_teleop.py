from __future__ import annotations

import math
from dataclasses import dataclass

from amsrr.geometry.pose_math import (
    pose_from_transform,
    pose_to_xyz_rpy,
    transform_from_xyz_rpy,
)
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffEnv


RANDOM_MORPHOLOGY_TELEOP_VERSION = "random_morphology_teleop_v1"

TELEOP_HELP = """\
Terminal controls (body/yaw-relative target increments):
  W / S : forward / backward       A / D : left / right
  R / F : up / down                J / L : yaw left / right
  I / K : pitch up / down          U / O : roll left / right
  H/space: hold the current pose    0     : reset the initial hover target
  P     : print target pose         ?     : print this help
  Q     : quit
"""


@dataclass
class RandomMorphologyTeleopConfig(SchemaBase):
    translation_step_m: float = 0.05
    rotation_step_rad: float = math.radians(5.0)
    max_roll_pitch_rad: float = math.radians(30.0)
    max_position_lead_m: float = 0.50
    minimum_height_above_settled_m: float = 0.15

    def validate(self) -> None:
        for name in (
            "translation_step_m",
            "rotation_step_rad",
            "max_roll_pitch_rad",
            "max_position_lead_m",
            "minimum_height_above_settled_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"RandomMorphologyTeleopConfig.{name} must be finite and positive"
                )
        if self.max_roll_pitch_rad >= math.pi / 2.0:
            raise SchemaValidationError(
                "RandomMorphologyTeleopConfig.max_roll_pitch_rad must be below pi/2"
            )


@dataclass(frozen=True)
class TeleopTargetUpdate:
    key: str
    action: str
    target_pose_world: Pose7D
    recognized: bool = True
    quit_requested: bool = False
    print_help: bool = False
    print_pose: bool = False


@dataclass
class RandomMorphologyTeleopTarget:
    initial_hover_pose_world: Pose7D
    target_pose_world: Pose7D
    minimum_target_height_m: float
    config: RandomMorphologyTeleopConfig

    @classmethod
    def from_hover_pose(
        cls,
        hover_pose_world: Pose7D,
        *,
        settled_height_m: float,
        config: RandomMorphologyTeleopConfig | None = None,
    ) -> "RandomMorphologyTeleopTarget":
        resolved_config = config or RandomMorphologyTeleopConfig()
        resolved_config.validate()
        return cls(
            initial_hover_pose_world=hover_pose_world,
            target_pose_world=hover_pose_world,
            minimum_target_height_m=(
                float(settled_height_m)
                + resolved_config.minimum_height_above_settled_m
            ),
            config=resolved_config,
        )

    def apply_key(self, key: str, *, current_pose_world: Pose7D) -> TeleopTargetUpdate:
        normalized = key.lower()
        if normalized == "q":
            return self._update(key, "quit", quit_requested=True)
        if normalized == "?":
            return self._update(key, "help", print_help=True)
        if normalized == "p":
            return self._update(key, "print_target", print_pose=True)
        if normalized in {"h", " "}:
            self.target_pose_world = current_pose_world
            return self._update(key, "hold_current_pose")
        if normalized == "0":
            self.target_pose_world = self.initial_hover_pose_world
            return self._update(key, "reset_hover_target")

        position, rpy = pose_to_xyz_rpy(self.target_pose_world)
        roll, pitch, yaw = rpy
        dx = dy = dz = 0.0
        action = ""
        step = self.config.translation_step_m
        if normalized == "w":
            dx, dy, action = math.cos(yaw) * step, math.sin(yaw) * step, "forward"
        elif normalized == "s":
            dx, dy, action = -math.cos(yaw) * step, -math.sin(yaw) * step, "backward"
        elif normalized == "a":
            dx, dy, action = -math.sin(yaw) * step, math.cos(yaw) * step, "left"
        elif normalized == "d":
            dx, dy, action = math.sin(yaw) * step, -math.cos(yaw) * step, "right"
        elif normalized == "r":
            dz, action = step, "up"
        elif normalized == "f":
            dz, action = -step, "down"
        elif normalized == "j":
            yaw, action = yaw + self.config.rotation_step_rad, "yaw_left"
        elif normalized == "l":
            yaw, action = yaw - self.config.rotation_step_rad, "yaw_right"
        elif normalized == "i":
            pitch, action = pitch + self.config.rotation_step_rad, "pitch_up"
        elif normalized == "k":
            pitch, action = pitch - self.config.rotation_step_rad, "pitch_down"
        elif normalized == "u":
            roll, action = roll + self.config.rotation_step_rad, "roll_left"
        elif normalized == "o":
            roll, action = roll - self.config.rotation_step_rad, "roll_right"
        else:
            return TeleopTargetUpdate(
                key=key,
                action="ignored",
                target_pose_world=self.target_pose_world,
                recognized=False,
            )

        max_attitude = self.config.max_roll_pitch_rad
        roll = _clamp(roll, -max_attitude, max_attitude)
        pitch = _clamp(pitch, -max_attitude, max_attitude)
        yaw = _wrap_angle(yaw)
        proposed_position = (
            position[0] + dx,
            position[1] + dy,
            max(position[2] + dz, self.minimum_target_height_m),
        )
        bounded_position = _bound_position_lead(
            proposed_position,
            current_pose_world[:3],
            self.config.max_position_lead_m,
        )
        self.target_pose_world = pose_from_transform(
            transform_from_xyz_rpy(bounded_position, (roll, pitch, yaw))
        )
        return self._update(key, action)

    def _update(
        self,
        key: str,
        action: str,
        *,
        quit_requested: bool = False,
        print_help: bool = False,
        print_pose: bool = False,
    ) -> TeleopTargetUpdate:
        return TeleopTargetUpdate(
            key=key,
            action=action,
            target_pose_world=self.target_pose_world,
            quit_requested=quit_requested,
            print_help=print_help,
            print_pose=print_pose,
        )


def build_random_morphology_teleop_probe_command(
    env: RandomMorphologyTakeoffEnv,
    morphology_graph: MorphologyGraph,
    *,
    config: RandomMorphologyTeleopConfig,
) -> list[str]:
    config.validate()
    command = env.build_probe_command(morphology_graph)
    command.extend(
        [
            "--random-morphology-teleop",
            "--teleop-translation-step-m",
            str(config.translation_step_m),
            "--teleop-rotation-step-rad",
            str(config.rotation_step_rad),
            "--teleop-max-roll-pitch-rad",
            str(config.max_roll_pitch_rad),
            "--teleop-max-position-lead-m",
            str(config.max_position_lead_m),
            "--teleop-minimum-height-above-settled-m",
            str(config.minimum_height_above_settled_m),
            "--hover-stop-on-hold",
            "--realtime-playback",
            "--viz",
            "kit",
        ]
    )
    return command


def format_teleop_pose(pose_world: Pose7D) -> str:
    position, rpy = pose_to_xyz_rpy(pose_world)
    return (
        "position_m=(%.3f, %.3f, %.3f) rpy_deg=(%.1f, %.1f, %.1f)"
        % (
            position[0],
            position[1],
            position[2],
            math.degrees(rpy[0]),
            math.degrees(rpy[1]),
            math.degrees(rpy[2]),
        )
    )


def _bound_position_lead(
    proposed: tuple[float, float, float],
    current: tuple[float, float, float],
    maximum_lead_m: float,
) -> tuple[float, float, float]:
    delta = tuple(proposed[index] - float(current[index]) for index in range(3))
    norm = math.sqrt(sum(value * value for value in delta))
    if norm <= maximum_lead_m:
        return proposed
    scale = maximum_lead_m / max(norm, 1.0e-12)
    return tuple(float(current[index]) + delta[index] * scale for index in range(3))  # type: ignore[return-value]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
