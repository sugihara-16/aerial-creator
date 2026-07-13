from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

from amsrr.geometry.pose_math import compose_pose
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len


ASSEMBLY_MOTION_PLAN_VERSION = "assembly_motion_plan_v1"


class AssemblyMotionPlanningError(RuntimeError):
    """Raised when the bounded deterministic assembly planner finds no safe path."""


@dataclass(frozen=True)
class AssemblyMotionPlannerConfig:
    sample_spacing_m: float = 0.025
    angular_sample_spacing_rad: float = math.radians(3.0)
    via_offset_m: float = 0.30
    maximum_via_depth: int = 2

    def __post_init__(self) -> None:
        if self.sample_spacing_m <= 0.0:
            raise SchemaValidationError("AssemblyMotionPlannerConfig.sample_spacing_m must be positive")
        if self.angular_sample_spacing_rad <= 0.0:
            raise SchemaValidationError(
                "AssemblyMotionPlannerConfig.angular_sample_spacing_rad must be positive"
            )
        if self.via_offset_m <= 0.0:
            raise SchemaValidationError("AssemblyMotionPlannerConfig.via_offset_m must be positive")
        if self.maximum_via_depth not in {1, 2}:
            raise SchemaValidationError(
                "AssemblyMotionPlannerConfig.maximum_via_depth must be 1 or 2"
            )


@dataclass
class AssemblyMotionPlan(SchemaBase):
    start_pose_world: Pose7D
    goal_pose_world: Pose7D
    waypoints_world: list[Pose7D]
    method: str
    collision_check_count: int
    version: str = ASSEMBLY_MOTION_PLAN_VERSION

    def validate(self) -> None:
        require_len(self.start_pose_world, 7, "AssemblyMotionPlan.start_pose_world")
        require_len(self.goal_pose_world, 7, "AssemblyMotionPlan.goal_pose_world")
        if not self.waypoints_world:
            raise SchemaValidationError("AssemblyMotionPlan.waypoints_world must not be empty")
        for index, pose in enumerate(self.waypoints_world):
            require_len(pose, 7, f"AssemblyMotionPlan.waypoints_world[{index}]")
        if tuple(self.waypoints_world[-1]) != tuple(self.goal_pose_world):
            raise SchemaValidationError("AssemblyMotionPlan must end at goal_pose_world")
        if self.collision_check_count <= 0:
            raise SchemaValidationError("AssemblyMotionPlan.collision_check_count must be positive")
        if self.version != ASSEMBLY_MOTION_PLAN_VERSION:
            raise SchemaValidationError("AssemblyMotionPlan.version is unsupported")


class DeterministicAssemblyMotionPlanner:
    """Bounded collision-aware SE(3) planner retained as the pi_A fallback.

    Collision geometry and scene ownership stay outside this module.  The
    caller supplies a fail-closed pose oracle for the complete moving
    component.  The planner first checks the direct segment, then stable
    one- and two-via alternatives around the midpoint.
    """

    def __init__(self, config: AssemblyMotionPlannerConfig | None = None) -> None:
        self.config = config or AssemblyMotionPlannerConfig()

    def plan(
        self,
        start_pose_world: Pose7D,
        goal_pose_world: Pose7D,
        *,
        is_pose_collision_free: Callable[[Pose7D], bool],
    ) -> AssemblyMotionPlan:
        _validate_pose(start_pose_world, "start_pose_world")
        _validate_pose(goal_pose_world, "goal_pose_world")
        checks = 0

        def path_is_free(waypoints: list[Pose7D]) -> bool:
            nonlocal checks
            poses = [start_pose_world, *waypoints]
            for left, right in zip(poses, poses[1:]):
                for pose in _segment_samples(left, right, self.config):
                    checks += 1
                    try:
                        free = is_pose_collision_free(pose)
                    except Exception as exc:  # noqa: BLE001 - collision evidence must fail closed
                        raise AssemblyMotionPlanningError(
                            "collision oracle failed closed: "
                            f"{type(exc).__name__}:{exc}"
                        ) from exc
                    if free is not True:
                        return False
            return True

        direct = [goal_pose_world]
        if path_is_free(direct):
            return AssemblyMotionPlan(
                start_pose_world=start_pose_world,
                goal_pose_world=goal_pose_world,
                waypoints_world=direct,
                method="direct_se3",
                collision_check_count=checks,
            )

        for waypoints, method in _via_candidates(
            start_pose_world,
            goal_pose_world,
            offset_m=self.config.via_offset_m,
            maximum_depth=self.config.maximum_via_depth,
        ):
            if path_is_free(waypoints):
                return AssemblyMotionPlan(
                    start_pose_world=start_pose_world,
                    goal_pose_world=goal_pose_world,
                    waypoints_world=waypoints,
                    method=method,
                    collision_check_count=checks,
                )
        raise AssemblyMotionPlanningError(
            "no collision-free direct or bounded deterministic via-point path"
        )


def _via_candidates(
    start: Pose7D,
    goal: Pose7D,
    *,
    offset_m: float,
    maximum_depth: int,
) -> Iterable[tuple[list[Pose7D], str]]:
    midpoint = _interpolate_pose(start, goal, 0.5)
    directions = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    for direction in directions:
        via = compose_pose(
            midpoint,
            (
                offset_m * direction[0],
                offset_m * direction[1],
                offset_m * direction[2],
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        )
        yield [via, goal], "single_via_se3"
    if maximum_depth < 2:
        return
    quarter = _interpolate_pose(start, goal, 0.25)
    three_quarter = _interpolate_pose(start, goal, 0.75)
    for direction in directions:
        delta = (
            offset_m * direction[0],
            offset_m * direction[1],
            offset_m * direction[2],
            0.0,
            0.0,
            0.0,
            1.0,
        )
        yield [compose_pose(quarter, delta), compose_pose(three_quarter, delta), goal], "double_via_se3"


def _segment_samples(
    start: Pose7D,
    goal: Pose7D,
    config: AssemblyMotionPlannerConfig,
) -> Iterable[Pose7D]:
    distance = math.sqrt(sum((float(goal[index]) - float(start[index])) ** 2 for index in range(3)))
    angular_distance = _quaternion_angular_distance(start[3:7], goal[3:7])
    intervals = max(
        1,
        int(math.ceil(distance / config.sample_spacing_m)),
        int(math.ceil(angular_distance / config.angular_sample_spacing_rad)),
    )
    for index in range(intervals + 1):
        yield _interpolate_pose(start, goal, float(index) / float(intervals))


def _interpolate_pose(start: Pose7D, goal: Pose7D, ratio: float) -> Pose7D:
    ratio = min(max(float(ratio), 0.0), 1.0)
    xyz = tuple(
        float(start[index]) + ratio * (float(goal[index]) - float(start[index]))
        for index in range(3)
    )
    quat = _quaternion_slerp(start[3:7], goal[3:7], ratio)
    return (*xyz, *quat)  # type: ignore[return-value]


def _quaternion_slerp(
    left: tuple[float, ...],
    right: tuple[float, ...],
    ratio: float,
) -> tuple[float, float, float, float]:
    q0 = _normalized_quaternion(left)
    q1 = _normalized_quaternion(right)
    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)  # type: ignore[assignment]
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalized_quaternion(
            tuple(q0[index] + ratio * (q1[index] - q0[index]) for index in range(4))
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left_weight = math.sin((1.0 - ratio) * theta) / sin_theta
    right_weight = math.sin(ratio * theta) / sin_theta
    return _normalized_quaternion(
        tuple(left_weight * q0[index] + right_weight * q1[index] for index in range(4))
    )


def _normalized_quaternion(values: tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise SchemaValidationError("pose quaternion must contain four values")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 0.0:
        raise SchemaValidationError("pose quaternion norm must be positive")
    normalized = tuple(float(value) / norm for value in values)
    if normalized[3] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def _quaternion_angular_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    q0 = _normalized_quaternion(left)
    q1 = _normalized_quaternion(right)
    dot = min(1.0, abs(sum(a * b for a, b in zip(q0, q1, strict=True))))
    return 2.0 * math.acos(dot)


def _validate_pose(pose: Pose7D, path: str) -> None:
    require_len(pose, 7, path)
    if not all(math.isfinite(float(value)) for value in pose):
        raise SchemaValidationError(f"{path} must contain finite values")
    _normalized_quaternion(tuple(float(value) for value in pose[3:7]))
