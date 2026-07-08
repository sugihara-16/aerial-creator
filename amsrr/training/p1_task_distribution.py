from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len
from amsrr.schemas.task_spec import TaskSpec
from amsrr.utils.config import load_config


@dataclass
class P1TaskDistributionConfig(SchemaBase):
    object_size_x_m: tuple[float, float] = (0.22, 0.38)
    object_size_y_m: tuple[float, float] = (0.14, 0.28)
    object_size_z_m: tuple[float, float] = (0.10, 0.22)
    object_mass_kg: tuple[float, float] = (0.5, 2.0)
    object_friction: tuple[float, float] = (0.35, 0.9)
    initial_x_m: tuple[float, float] = (0.65, 0.95)
    initial_y_m: tuple[float, float] = (-0.25, 0.25)
    initial_z_m: tuple[float, float] = (0.35, 0.55)
    target_x_m: tuple[float, float] = (1.6, 2.4)
    target_y_m: tuple[float, float] = (-0.35, 0.35)
    target_z_m: tuple[float, float] = (0.35, 0.55)
    target_tolerance_pos_m: float = 0.05

    def validate(self) -> None:
        for name in (
            "object_size_x_m",
            "object_size_y_m",
            "object_size_z_m",
            "object_mass_kg",
            "object_friction",
            "initial_x_m",
            "initial_y_m",
            "initial_z_m",
            "target_x_m",
            "target_y_m",
            "target_z_m",
        ):
            _validate_range(getattr(self, name), f"P1TaskDistributionConfig.{name}")
        if self.target_tolerance_pos_m < 0.0:
            raise SchemaValidationError("P1TaskDistributionConfig.target_tolerance_pos_m must be non-negative")


@dataclass
class P1TaskSample(SchemaBase):
    task_spec: TaskSpec
    seed: int
    sample_index: int
    sampled_values: dict[str, float]


def load_p1_task_distribution_config(path: str) -> P1TaskDistributionConfig:
    data = load_config(path)
    return P1TaskDistributionConfig.from_dict(data.get("distribution", data))


class P1GraspCarryTaskDistribution:
    def __init__(
        self,
        base_task_spec: TaskSpec,
        config: P1TaskDistributionConfig | None = None,
    ) -> None:
        self.base_task_spec = base_task_spec
        self.config = config or P1TaskDistributionConfig()

    def sample(self, *, seed: int, sample_index: int = 0) -> P1TaskSample:
        rng = random.Random(seed)
        task_data = self.base_task_spec.to_dict()
        target_object_id = _target_object_id(task_data)
        sampled = _sample_values(self.config, rng)
        _apply_object_randomization(task_data, target_object_id, sampled)
        _apply_target_randomization(task_data, target_object_id, sampled, self.config)
        task_data["task_id"] = f"{self.base_task_spec.task_id}_p1_{sample_index:04d}"
        metadata = dict(task_data.get("metadata", {}) or {})
        metadata["randomization_seed"] = seed
        metadata["randomization_sample_index"] = sample_index
        metadata["randomization_family"] = "p1_grasp_carry"
        task_data["metadata"] = metadata
        return P1TaskSample(
            task_spec=TaskSpec.from_dict(task_data),
            seed=seed,
            sample_index=sample_index,
            sampled_values=sampled,
        )


def _sample_values(config: P1TaskDistributionConfig, rng: random.Random) -> dict[str, float]:
    return {
        "object_size_x_m": _sample_range(config.object_size_x_m, rng),
        "object_size_y_m": _sample_range(config.object_size_y_m, rng),
        "object_size_z_m": _sample_range(config.object_size_z_m, rng),
        "object_mass_kg": _sample_range(config.object_mass_kg, rng),
        "object_friction": _sample_range(config.object_friction, rng),
        "initial_x_m": _sample_range(config.initial_x_m, rng),
        "initial_y_m": _sample_range(config.initial_y_m, rng),
        "initial_z_m": _sample_range(config.initial_z_m, rng),
        "target_x_m": _sample_range(config.target_x_m, rng),
        "target_y_m": _sample_range(config.target_y_m, rng),
        "target_z_m": _sample_range(config.target_z_m, rng),
    }


def _apply_object_randomization(
    task_data: dict[str, Any],
    target_object_id: str,
    sampled: dict[str, float],
) -> None:
    geometry_id = None
    for obj in task_data["scene"]["objects"]:
        if obj["object_id"] != target_object_id:
            continue
        geometry_id = obj["geometry_id"]
        obj["pose_world"] = _pose_with_xyz(
            obj["pose_world"],
            sampled["initial_x_m"],
            sampled["initial_y_m"],
            sampled["initial_z_m"],
        )
        obj["mass_kg"] = sampled["object_mass_kg"]
        obj["friction"] = sampled["object_friction"]
        break
    if geometry_id is None:
        raise SchemaValidationError(f"Target object {target_object_id!r} not found in task scene")
    for geometry in task_data["scene"]["geometry_library"]:
        if geometry["geometry_id"] != geometry_id:
            continue
        params = dict(geometry.get("primitive_params") or {})
        params["size_m"] = [
            sampled["object_size_x_m"],
            sampled["object_size_y_m"],
            sampled["object_size_z_m"],
        ]
        geometry["primitive_params"] = params
        return
    raise SchemaValidationError(f"Target object geometry {geometry_id!r} not found in task scene")


def _apply_target_randomization(
    task_data: dict[str, Any],
    target_object_id: str,
    sampled: dict[str, float],
    config: P1TaskDistributionConfig,
) -> None:
    for goal in task_data["goals"]:
        if goal.get("goal_type") != "object_pose" or goal.get("target_entity_id") != target_object_id:
            continue
        goal["target_pose_world"] = _pose_with_xyz(
            goal["target_pose_world"],
            sampled["target_x_m"],
            sampled["target_y_m"],
            sampled["target_z_m"],
        )
        goal["tolerance_pos_m"] = config.target_tolerance_pos_m
        return
    raise SchemaValidationError(f"Object pose goal for {target_object_id!r} not found in task")


def _target_object_id(task_data: dict[str, Any]) -> str:
    for goal in task_data["goals"]:
        if goal.get("goal_type") == "object_pose" and goal.get("target_entity_id"):
            return str(goal["target_entity_id"])
    for obj in task_data["scene"]["objects"]:
        if obj.get("movable"):
            return str(obj["object_id"])
    raise SchemaValidationError("P1 grasp/carry distribution requires a movable target object")


def _pose_with_xyz(pose: Pose7D | list[float], x: float, y: float, z: float) -> list[float]:
    require_len(pose, 7, "pose")
    return [float(x), float(y), float(z), float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])]


def _sample_range(value_range: tuple[float, float], rng: random.Random) -> float:
    lower, upper = value_range
    return rng.uniform(lower, upper)


def _validate_range(value_range: tuple[float, float], path: str) -> None:
    require_len(value_range, 2, path)
    lower, upper = value_range
    if lower > upper:
        raise SchemaValidationError(f"{path} lower bound must be <= upper bound")
