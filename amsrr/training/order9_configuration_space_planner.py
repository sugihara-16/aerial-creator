from __future__ import annotations

"""Deterministic configuration-space planning for the Order 9 pi_H teacher."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

from amsrr.schemas.common import Pose7D, SchemaValidationError

ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION = (
    "order9_configuration_space_overhead_birrt_v6_local_contact"
)
_LIGHTWEIGHT_PLANNING_ONLY = ContextVar(
    "order9_lightweight_configuration_planning_only", default=False
)


@contextmanager
def order9_lightweight_configuration_planning():
    """Forbid sampled searches inside an explicitly lightweight admission pass."""

    token = _LIGHTWEIGHT_PLANNING_ONLY.set(True)
    try:
        yield
    finally:
        _LIGHTWEIGHT_PLANNING_ONLY.reset(token)


class Order9ConfigurationSpacePlanningError(RuntimeError):
    """Raised when the bounded deterministic planner finds no valid path."""


@dataclass(frozen=True)
class Order9ConfigurationState:
    base_pose_world: Pose7D
    joint_positions_rad: dict[str, float]

    def __post_init__(self) -> None:
        _validate_pose(self.base_pose_world)
        if not self.joint_positions_rad:
            raise SchemaValidationError(
                "configuration state requires at least one Dock joint"
            )
        if any(
            not joint_id or not math.isfinite(float(value))
            for joint_id, value in self.joint_positions_rad.items()
        ):
            raise SchemaValidationError(
                "configuration state joints must have non-empty finite values"
            )


@dataclass(frozen=True)
class Order9ConfigurationSpacePlannerConfig:
    maximum_iterations: int = 1200
    maximum_connect_extensions: int = 128
    base_extension_step_m: float = 0.15
    attitude_extension_step_rad: float = 0.25
    joint_extension_step_rad: float = 0.30
    base_collision_sample_spacing_m: float = 0.035
    attitude_collision_sample_spacing_rad: float = 0.075
    joint_collision_sample_spacing_rad: float = 0.075
    sampling_margin_m: float = 0.65
    # Receding-horizon IK need not reproduce an intermediate overhead
    # waypoint bit-for-bit before the fully checked final descent edge can be
    # attempted.  These tolerances only select that edge; its exact goal and
    # all interpolated collision checks remain unchanged.
    waypoint_translation_tolerance_m: float = 0.005
    waypoint_attitude_tolerance_rad: float = 0.001
    waypoint_joint_tolerance_rad: float = 0.035
    goal_bias_period: int = 7
    deterministic_seed: int = 19
    local_contact_maximum_iterations: int = 160
    local_contact_joint_jitter_rad: float = 0.18

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if self.maximum_connect_extensions < 1:
            raise ValueError("maximum_connect_extensions must be positive")
        if self.goal_bias_period < 2:
            raise ValueError("goal_bias_period must be at least two")
        if self.deterministic_seed < 0:
            raise ValueError("deterministic_seed must be non-negative")
        if self.local_contact_maximum_iterations < 1:
            raise ValueError("local_contact_maximum_iterations must be positive")
        for name in (
            "base_extension_step_m",
            "attitude_extension_step_rad",
            "joint_extension_step_rad",
            "base_collision_sample_spacing_m",
            "attitude_collision_sample_spacing_rad",
            "joint_collision_sample_spacing_rad",
            "sampling_margin_m",
            "waypoint_translation_tolerance_m",
            "waypoint_attitude_tolerance_rad",
            "waypoint_joint_tolerance_rad",
            "local_contact_joint_jitter_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Order9ConfigurationSpacePlan:
    states: tuple[Order9ConfigurationState, ...]
    method: str
    collision_check_count: int
    sampled_state_count: int
    tree_node_count: int
    planner_version: str = ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION

    def __post_init__(self) -> None:
        if len(self.states) < 2:
            raise SchemaValidationError(
                "configuration-space plan requires start and goal states"
            )
        if self.collision_check_count < 1:
            raise SchemaValidationError(
                "configuration-space plan requires collision evidence"
            )
        if self.sampled_state_count < 0 or self.tree_node_count < 2:
            raise SchemaValidationError(
                "configuration-space planner counts are invalid"
            )
        if not self.method:
            raise SchemaValidationError(
                "configuration-space plan method must be non-empty"
            )


@dataclass
class _Tree:
    states: list[Order9ConfigurationState]
    parents: list[int | None]


class DeterministicOrder9ConfigurationSpacePlanner:
    """Seed-fixed bidirectional RRT-Connect over base SE(3) and Dock q."""

    planner_version = ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION

    def __init__(
        self,
        config: Order9ConfigurationSpacePlannerConfig | None = None,
    ) -> None:
        self.config = config or Order9ConfigurationSpacePlannerConfig()

    def validate_direct_edge(
        self,
        *,
        start: Order9ConfigurationState,
        goal: Order9ConfigurationState,
        is_collision_free: Callable[[Order9ConfigurationState], bool],
    ) -> tuple[bool, int]:
        """Revalidate one cached route edge without launching a new search.

        A rolling teacher can resume a previously certified path from an
        ideal or measured intermediate state.  The remaining edge is checked
        again because a real controller may not reproduce the previous
        nominal endpoint exactly.  Failure is reported to the caller so it
        can discard the cache and run a complete configuration-space search.
        """

        ordered_ids = tuple(sorted(start.joint_positions_rad))
        if set(goal.joint_positions_rad) != set(ordered_ids):
            raise SchemaValidationError("configuration edge joint identities differ")
        collision_checks = 0

        def checked(state: Order9ConfigurationState) -> bool:
            nonlocal collision_checks
            collision_checks += 1
            try:
                return is_collision_free(state) is True
            except Exception as error:  # noqa: BLE001 - fail closed
                raise Order9ConfigurationSpacePlanningError(
                    "configuration collision oracle failed closed: "
                    f"{type(error).__name__}:{error}"
                ) from error

        if not checked(start) or not checked(goal):
            return False, collision_checks
        accepted = _edge_is_free(
            start,
            goal,
            ordered_ids=ordered_ids,
            config=self.config,
            checked=checked,
        )
        return bool(accepted), collision_checks

    def plan_local_contact(
        self,
        *,
        start: Order9ConfigurationState,
        goal: Order9ConfigurationState,
        joint_limits_rad: Mapping[str, tuple[float, float]],
        is_collision_free: Callable[[Order9ConfigurationState], bool],
        corridor_radius_m: float,
    ) -> Order9ConfigurationSpacePlan:
        """Plan a bounded local pregrasp-to-contact configuration path.

        Contact acquisition is not another global transit phase.  Search is
        therefore restricted to a tube around the start-goal base segment
        and uses no overhead bridge or full-joint-limit samples.
        """

        radius = float(corridor_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("corridor_radius_m must be finite and positive")
        ordered_ids, limits = _validated_joint_domain(
            start=start,
            goal=goal,
            joint_limits_rad=joint_limits_rad,
        )
        collision_checks = 0

        def checked(state: Order9ConfigurationState) -> bool:
            nonlocal collision_checks
            collision_checks += 1
            if (
                _point_to_segment_distance(
                    state.base_pose_world[:3],
                    start.base_pose_world[:3],
                    goal.base_pose_world[:3],
                )
                > radius + 1.0e-9
            ):
                return False
            try:
                return is_collision_free(state) is True
            except Exception as error:  # noqa: BLE001 - fail closed
                raise Order9ConfigurationSpacePlanningError(
                    "configuration collision oracle failed closed: "
                    f"{type(error).__name__}:{error}"
                ) from error

        if not checked(start):
            raise Order9ConfigurationSpacePlanningError(
                "local contact start state is in collision"
            )
        if not checked(goal):
            raise Order9ConfigurationSpacePlanningError(
                "local contact goal state is in collision"
            )
        if _edge_is_free(
            start,
            goal,
            ordered_ids=ordered_ids,
            config=self.config,
            checked=checked,
        ):
            return Order9ConfigurationSpacePlan(
                states=(start, goal),
                method="direct_configuration_edge",
                collision_check_count=collision_checks,
                sampled_state_count=0,
                tree_node_count=2,
            )

        coordinate_path = _deterministic_local_coordinate_bridge(
            start,
            goal,
            corridor_radius_m=radius,
            ordered_ids=ordered_ids,
            config=self.config,
            checked=checked,
        )
        if coordinate_path is not None:
            return Order9ConfigurationSpacePlan(
                states=tuple(coordinate_path),
                method="deterministic_local_coordinate_bridge",
                collision_check_count=collision_checks,
                sampled_state_count=0,
                tree_node_count=len(coordinate_path),
            )

        if _LIGHTWEIGHT_PLANNING_ONLY.get():
            raise Order9ConfigurationSpacePlanningError(
                "sampled local contact search is disabled in lightweight admission"
            )

        start_tree = _Tree(states=[start], parents=[None])
        goal_tree = _Tree(states=[goal], parents=[None])
        active = start_tree
        other = goal_tree
        active_is_start = True
        sampled = 0
        for iteration in range(self.config.local_contact_maximum_iterations):
            target = (
                other.states[0]
                if iteration % self.config.goal_bias_period == 0
                else _sample_local_contact_state(
                    iteration=iteration,
                    start=start,
                    goal=goal,
                    ordered_ids=ordered_ids,
                    limits=limits,
                    corridor_radius_m=radius,
                    config=self.config,
                )
            )
            sampled += 1
            new_index = _extend_once(
                active,
                target,
                ordered_ids=ordered_ids,
                config=self.config,
                checked=checked,
            )
            if new_index is not None:
                connected_index = _connect(
                    other,
                    active.states[new_index],
                    ordered_ids=ordered_ids,
                    config=self.config,
                    checked=checked,
                )
                if connected_index is not None:
                    if active_is_start:
                        start_path = _root_path(active, new_index)
                        goal_path = _root_path(other, connected_index)
                    else:
                        start_path = _root_path(other, connected_index)
                        goal_path = _root_path(active, new_index)
                    path = [
                        *start_path,
                        *list(reversed(goal_path))[1:],
                    ]
                    path = _shortcut_path(
                        path,
                        ordered_ids=ordered_ids,
                        config=self.config,
                        checked=checked,
                    )
                    return Order9ConfigurationSpacePlan(
                        states=tuple(path),
                        method="deterministic_local_birrt_connect",
                        collision_check_count=collision_checks,
                        sampled_state_count=sampled,
                        tree_node_count=(
                            len(start_tree.states) + len(goal_tree.states)
                        ),
                    )
            active, other = other, active
            active_is_start = not active_is_start
        raise Order9ConfigurationSpacePlanningError(
            "bounded deterministic local contact search exhausted "
            f"{self.config.local_contact_maximum_iterations} samples"
        )

    def plan(
        self,
        *,
        start: Order9ConfigurationState,
        goal: Order9ConfigurationState,
        joint_limits_rad: Mapping[str, tuple[float, float]],
        is_collision_free: Callable[[Order9ConfigurationState], bool],
        sampling_center_world: Sequence[float] | None = None,
        overhead_required: bool = False,
    ) -> Order9ConfigurationSpacePlan:
        ordered_ids = tuple(sorted(start.joint_positions_rad))
        if set(goal.joint_positions_rad) != set(ordered_ids) or set(
            joint_limits_rad
        ) != set(ordered_ids):
            raise SchemaValidationError("configuration planner joint identities differ")
        limits = {
            joint_id: (
                float(joint_limits_rad[joint_id][0]),
                float(joint_limits_rad[joint_id][1]),
            )
            for joint_id in ordered_ids
        }
        for joint_id, (lower, upper) in limits.items():
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower >= upper
                or float(start.joint_positions_rad[joint_id]) < lower
                or float(start.joint_positions_rad[joint_id]) > upper
                or float(goal.joint_positions_rad[joint_id]) < lower
                or float(goal.joint_positions_rad[joint_id]) > upper
            ):
                raise SchemaValidationError(
                    f"configuration planner has invalid limits for {joint_id}"
                )

        collision_checks = 0

        def checked(state: Order9ConfigurationState) -> bool:
            nonlocal collision_checks
            collision_checks += 1
            try:
                return is_collision_free(state) is True
            except Exception as error:  # noqa: BLE001 - fail closed
                raise Order9ConfigurationSpacePlanningError(
                    "configuration collision oracle failed closed: "
                    f"{type(error).__name__}:{error}"
                ) from error

        if not checked(start):
            raise Order9ConfigurationSpacePlanningError(
                "configuration-space start state is in collision"
            )
        if not checked(goal):
            raise Order9ConfigurationSpacePlanningError(
                "configuration-space goal state is in collision"
            )
        direct_edge_allowed = not overhead_required or _aligned_for_overhead_descent(
            start,
            goal,
            ordered_ids=ordered_ids,
            config=self.config,
        )
        if direct_edge_allowed and _edge_is_free(
            start,
            goal,
            ordered_ids=ordered_ids,
            config=self.config,
            checked=checked,
        ):
            return Order9ConfigurationSpacePlan(
                states=(start, goal),
                method="direct_configuration_edge",
                collision_check_count=collision_checks,
                sampled_state_count=0,
                tree_node_count=2,
            )

        center = _sampling_center(
            start,
            goal,
            sampling_center_world=sampling_center_world,
        )
        overhead_path = _deterministic_overhead_bridge(
            start,
            goal,
            sampling_center=center,
            ordered_ids=ordered_ids,
            config=self.config,
            checked=checked,
        )
        if overhead_path is not None:
            return Order9ConfigurationSpacePlan(
                states=tuple(overhead_path),
                method="deterministic_overhead_configuration_bridge",
                collision_check_count=collision_checks,
                sampled_state_count=0,
                tree_node_count=len(overhead_path),
            )
        if overhead_required:
            coordinate_path = _deterministic_overhead_coordinate_bridge(
                start,
                goal,
                sampling_center=center,
                ordered_ids=ordered_ids,
                config=self.config,
                checked=checked,
            )
            if coordinate_path is not None:
                return Order9ConfigurationSpacePlan(
                    states=tuple(coordinate_path),
                    method="deterministic_overhead_coordinate_bridge",
                    collision_check_count=collision_checks,
                    sampled_state_count=0,
                    tree_node_count=len(coordinate_path),
                )
            if _LIGHTWEIGHT_PLANNING_ONLY.get():
                raise Order9ConfigurationSpacePlanningError(
                    "sampled overhead search is disabled in lightweight admission"
                )
            overhead_result = _deterministic_overhead_birrt_bridge(
                start,
                goal,
                sampling_center=center,
                ordered_ids=ordered_ids,
                limits=limits,
                config=self.config,
                checked=checked,
            )
            if overhead_result is not None:
                path, sampled, tree_nodes = overhead_result
                return Order9ConfigurationSpacePlan(
                    states=tuple(path),
                    method="deterministic_overhead_birrt_bridge",
                    collision_check_count=collision_checks,
                    sampled_state_count=sampled,
                    tree_node_count=tree_nodes,
                )
            raise Order9ConfigurationSpacePlanningError(
                "deterministic overhead configuration-space search is " "blocked"
            )

        if _LIGHTWEIGHT_PLANNING_ONLY.get():
            raise Order9ConfigurationSpacePlanningError(
                "sampled configuration search is disabled in lightweight admission"
            )
        start_tree = _Tree(states=[start], parents=[None])
        goal_tree = _Tree(states=[goal], parents=[None])
        active = start_tree
        other = goal_tree
        active_is_start = True
        sampled = 0
        for iteration in range(self.config.maximum_iterations):
            target = (
                other.states[0]
                if iteration % self.config.goal_bias_period == 0
                else _sample_state(
                    iteration=iteration,
                    start=start,
                    goal=goal,
                    ordered_ids=ordered_ids,
                    limits=limits,
                    sampling_center=center,
                    config=self.config,
                )
            )
            sampled += 1
            new_index = _extend_once(
                active,
                target,
                ordered_ids=ordered_ids,
                config=self.config,
                checked=checked,
            )
            if new_index is not None:
                connected_index = _connect(
                    other,
                    active.states[new_index],
                    ordered_ids=ordered_ids,
                    config=self.config,
                    checked=checked,
                )
                if connected_index is not None:
                    if active_is_start:
                        start_path = _root_path(active, new_index)
                        goal_path = _root_path(other, connected_index)
                    else:
                        start_path = _root_path(other, connected_index)
                        goal_path = _root_path(active, new_index)
                    path = [
                        *start_path,
                        *list(reversed(goal_path))[1:],
                    ]
                    path = _shortcut_path(
                        path,
                        ordered_ids=ordered_ids,
                        config=self.config,
                        checked=checked,
                    )
                    return Order9ConfigurationSpacePlan(
                        states=tuple(path),
                        method="deterministic_birrt_connect",
                        collision_check_count=collision_checks,
                        sampled_state_count=sampled,
                        tree_node_count=(
                            len(start_tree.states) + len(goal_tree.states)
                        ),
                    )
            active, other = other, active
            active_is_start = not active_is_start
        raise Order9ConfigurationSpacePlanningError(
            "bounded deterministic configuration-space search exhausted "
            f"{self.config.maximum_iterations} samples"
        )


def _validated_joint_domain(
    *,
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    joint_limits_rad: Mapping[str, tuple[float, float]],
) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    ordered_ids = tuple(sorted(start.joint_positions_rad))
    if set(goal.joint_positions_rad) != set(ordered_ids) or set(
        joint_limits_rad
    ) != set(ordered_ids):
        raise SchemaValidationError("configuration planner joint identities differ")
    limits = {
        joint_id: (
            float(joint_limits_rad[joint_id][0]),
            float(joint_limits_rad[joint_id][1]),
        )
        for joint_id in ordered_ids
    }
    for joint_id, (lower, upper) in limits.items():
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            or float(start.joint_positions_rad[joint_id]) < lower
            or float(start.joint_positions_rad[joint_id]) > upper
            or float(goal.joint_positions_rad[joint_id]) < lower
            or float(goal.joint_positions_rad[joint_id]) > upper
        ):
            raise SchemaValidationError(
                f"configuration planner has invalid limits for {joint_id}"
            )
    return ordered_ids, limits


def _deterministic_local_coordinate_bridge(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    corridor_radius_m: float,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> list[Order9ConfigurationState] | None:
    """Try inexpensive local base and one-joint-at-a-time detours."""

    midpoint = _interpolate_state(
        start,
        goal,
        0.5,
        ordered_ids=ordered_ids,
    )
    radius = 0.85 * float(corridor_radius_m)
    offset_directions = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    for direction in offset_directions:
        detour = Order9ConfigurationState(
            base_pose_world=(
                *(
                    float(midpoint.base_pose_world[index])
                    + radius * float(direction[index])
                    for index in range(3)
                ),
                *midpoint.base_pose_world[3:],
            ),
            joint_positions_rad=dict(midpoint.joint_positions_rad),
        )
        if all(
            _edge_is_free(
                left,
                right,
                ordered_ids=ordered_ids,
                config=config,
                checked=checked,
            )
            for left, right in ((start, detour), (detour, goal))
        ):
            return [start, detour, goal]

    deltas = {
        joint_id: abs(
            float(goal.joint_positions_rad[joint_id])
            - float(start.joint_positions_rad[joint_id])
        )
        for joint_id in ordered_ids
    }
    changing_ids = tuple(
        joint_id for joint_id in ordered_ids if deltas[joint_id] > 1.0e-9
    )
    raw_orders = (
        changing_ids,
        tuple(reversed(changing_ids)),
        tuple(sorted(changing_ids, key=lambda value: (-deltas[value], value))),
    )
    orders: list[tuple[str, ...]] = []
    for order in raw_orders:
        if order and order not in orders:
            orders.append(order)
    for move_base_first in (False, True):
        for order in orders:
            path = [start]
            current_pose = (
                goal.base_pose_world if move_base_first else start.base_pose_world
            )
            if move_base_first:
                path.append(
                    Order9ConfigurationState(
                        base_pose_world=current_pose,
                        joint_positions_rad=dict(start.joint_positions_rad),
                    )
                )
            q = dict(start.joint_positions_rad)
            for joint_id in order:
                q[joint_id] = float(goal.joint_positions_rad[joint_id])
                path.append(
                    Order9ConfigurationState(
                        base_pose_world=current_pose,
                        joint_positions_rad=dict(q),
                    )
                )
            if not move_base_first:
                path.append(goal)
            compact = _join_configuration_paths(
                path,
                ordered_ids=ordered_ids,
            )
            if all(
                _edge_is_free(
                    left,
                    right,
                    ordered_ids=ordered_ids,
                    config=config,
                    checked=checked,
                )
                for left, right in zip(compact, compact[1:])
            ):
                return _shortcut_path(
                    compact,
                    ordered_ids=ordered_ids,
                    config=config,
                    checked=checked,
                )
    return None


def _sample_local_contact_state(
    *,
    iteration: int,
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    ordered_ids: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
    corridor_radius_m: float,
    config: Order9ConfigurationSpacePlannerConfig,
) -> Order9ConfigurationState:
    sample_index = (
        iteration
        + 1
        + config.deterministic_seed * config.local_contact_maximum_iterations
    )
    blend = _halton(sample_index, _prime(0))
    center = tuple(
        (1.0 - blend) * float(start.base_pose_world[index])
        + blend * float(goal.base_pose_world[index])
        for index in range(3)
    )
    raw_direction = tuple(
        2.0 * _halton(sample_index, _prime(index + 1)) - 1.0 for index in range(3)
    )
    direction_norm = math.sqrt(sum(value * value for value in raw_direction))
    if direction_norm <= 1.0e-12:
        direction = (1.0, 0.0, 0.0)
    else:
        direction = tuple(value / direction_norm for value in raw_direction)
    radial_fraction = _halton(sample_index, _prime(4)) ** (1.0 / 3.0)
    xyz = tuple(
        center[index] + float(corridor_radius_m) * radial_fraction * direction[index]
        for index in range(3)
    )
    orientation = _quaternion_slerp(
        start.base_pose_world[3:],
        goal.base_pose_world[3:],
        blend,
    )
    jitter_scale = (0.0, 0.33, 0.66, 1.0)[iteration % 4]
    joints = {}
    for joint_index, joint_id in enumerate(ordered_ids):
        lower, upper = limits[joint_id]
        reference = (1.0 - blend) * float(
            start.joint_positions_rad[joint_id]
        ) + blend * float(goal.joint_positions_rad[joint_id])
        unit = _halton(sample_index, _prime(joint_index + 5))
        value = reference + (
            (2.0 * unit - 1.0) * jitter_scale * config.local_contact_joint_jitter_rad
        )
        joints[joint_id] = min(max(value, lower), upper)
    return Order9ConfigurationState(
        base_pose_world=(*xyz, *orientation),
        joint_positions_rad=joints,
    )


def _point_to_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    goal: Sequence[float],
) -> float:
    segment = tuple(float(goal[i]) - float(start[i]) for i in range(3))
    offset = tuple(float(point[i]) - float(start[i]) for i in range(3))
    length_squared = sum(value * value for value in segment)
    if length_squared <= 1.0e-18:
        return math.sqrt(sum(value * value for value in offset))
    fraction = min(
        1.0,
        max(
            0.0,
            sum(offset[i] * segment[i] for i in range(3)) / length_squared,
        ),
    )
    return math.sqrt(
        sum(
            (float(point[i]) - (float(start[i]) + fraction * segment[i])) ** 2
            for i in range(3)
        )
    )


def _deterministic_overhead_bridge(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    sampling_center: tuple[float, float, float],
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> list[Order9ConfigurationState] | None:
    """Try lift, translate, preform, and descend before blind sampling.

    This bridge encodes the C3 support-surface assumption explicitly: the
    mechanism first reaches clear overhead space, moves laterally and changes
    Dock posture there, then approaches the final configuration vertically.
    Every edge still uses the same convex collision oracle as the RRT.
    """

    overhead_z = max(
        float(start.base_pose_world[2]),
        float(goal.base_pose_world[2]),
        float(sampling_center[2]) + float(config.sampling_margin_m),
    )
    start_q = dict(start.joint_positions_rad)
    goal_q = dict(goal.joint_positions_rad)
    candidates = [
        start,
        Order9ConfigurationState(
            base_pose_world=(
                float(start.base_pose_world[0]),
                float(start.base_pose_world[1]),
                overhead_z,
                *start.base_pose_world[3:],
            ),
            joint_positions_rad=start_q,
        ),
        Order9ConfigurationState(
            base_pose_world=(
                float(goal.base_pose_world[0]),
                float(goal.base_pose_world[1]),
                overhead_z,
                *goal.base_pose_world[3:],
            ),
            joint_positions_rad=start_q,
        ),
        Order9ConfigurationState(
            base_pose_world=(
                float(goal.base_pose_world[0]),
                float(goal.base_pose_world[1]),
                overhead_z,
                *goal.base_pose_world[3:],
            ),
            joint_positions_rad=goal_q,
        ),
        goal,
    ]
    path = [candidates[0]]
    for index, state in enumerate(candidates[1:], start=1):
        final_goal = index == len(candidates) - 1
        if final_goal:
            if not _states_close(path[-1], state, ordered_ids=ordered_ids):
                path.append(state)
        elif not _waypoints_equivalent(
            path[-1],
            state,
            ordered_ids=ordered_ids,
            config=config,
        ):
            path.append(state)
    for left, right in zip(path, path[1:]):
        if not _edge_is_free(
            left,
            right,
            ordered_ids=ordered_ids,
            config=config,
            checked=checked,
        ):
            return None
    return path


def _deterministic_overhead_birrt_bridge(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    sampling_center: tuple[float, float, float],
    ordered_ids: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> tuple[list[Order9ConfigurationState], int, int] | None:
    """Find a joint-space detour while retaining a top-down approach.

    The simple overhead bridge changes all Dock joints along one straight
    edge.  Larger morphologies can have collision-free endpoints whose
    straight joint interpolation self-collides.  This bounded deterministic
    fallback lifts first, runs BiRRT entirely in the overhead half-space, and
    descends only after matching the exact final XY/orientation/joint state.
    """

    overhead_z = max(
        float(start.base_pose_world[2]),
        float(goal.base_pose_world[2]),
        float(sampling_center[2]) + float(config.sampling_margin_m),
    )
    overhead_start = Order9ConfigurationState(
        base_pose_world=(
            float(start.base_pose_world[0]),
            float(start.base_pose_world[1]),
            overhead_z,
            *start.base_pose_world[3:],
        ),
        joint_positions_rad=dict(start.joint_positions_rad),
    )
    overhead_goal = Order9ConfigurationState(
        base_pose_world=(
            float(goal.base_pose_world[0]),
            float(goal.base_pose_world[1]),
            overhead_z,
            *goal.base_pose_world[3:],
        ),
        joint_positions_rad=dict(goal.joint_positions_rad),
    )
    if not _edge_is_free(
        start,
        overhead_start,
        ordered_ids=ordered_ids,
        config=config,
        checked=checked,
    ):
        return None
    if not _edge_is_free(
        overhead_goal,
        goal,
        ordered_ids=ordered_ids,
        config=config,
        checked=checked,
    ):
        return None

    def overhead_checked(state: Order9ConfigurationState) -> bool:
        return bool(
            float(state.base_pose_world[2]) >= overhead_z - 1.0e-9 and checked(state)
        )

    if _edge_is_free(
        overhead_start,
        overhead_goal,
        ordered_ids=ordered_ids,
        config=config,
        checked=overhead_checked,
    ):
        overhead_path = [overhead_start, overhead_goal]
        return (
            _join_configuration_paths(
                (start,), overhead_path, (goal,), ordered_ids=ordered_ids
            ),
            0,
            2,
        )

    start_tree = _Tree(states=[overhead_start], parents=[None])
    goal_tree = _Tree(states=[overhead_goal], parents=[None])
    active = start_tree
    other = goal_tree
    active_is_start = True
    sampled = 0
    for iteration in range(config.maximum_iterations):
        target = (
            other.states[0]
            if iteration % config.goal_bias_period == 0
            else _sample_state(
                iteration=iteration,
                start=overhead_start,
                goal=overhead_goal,
                ordered_ids=ordered_ids,
                limits=limits,
                sampling_center=(
                    float(sampling_center[0]),
                    float(sampling_center[1]),
                    overhead_z,
                ),
                config=config,
                minimum_base_z=overhead_z,
            )
        )
        sampled += 1
        new_index = _extend_once(
            active,
            target,
            ordered_ids=ordered_ids,
            config=config,
            checked=overhead_checked,
        )
        if new_index is not None:
            connected_index = _connect(
                other,
                active.states[new_index],
                ordered_ids=ordered_ids,
                config=config,
                checked=overhead_checked,
            )
            if connected_index is not None:
                if active_is_start:
                    start_path = _root_path(active, new_index)
                    goal_path = _root_path(other, connected_index)
                else:
                    start_path = _root_path(other, connected_index)
                    goal_path = _root_path(active, new_index)
                overhead_path = [
                    *start_path,
                    *list(reversed(goal_path))[1:],
                ]
                overhead_path = _shortcut_path(
                    overhead_path,
                    ordered_ids=ordered_ids,
                    config=config,
                    checked=overhead_checked,
                )
                combined = _join_configuration_paths(
                    (start,),
                    overhead_path,
                    (goal,),
                    ordered_ids=ordered_ids,
                )
                return (
                    combined,
                    sampled,
                    max(
                        len(combined),
                        len(start_tree.states) + len(goal_tree.states),
                    ),
                )
        active, other = other, active
        active_is_start = not active_is_start
    return None


def _deterministic_overhead_coordinate_bridge(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    sampling_center: tuple[float, float, float],
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> list[Order9ConfigurationState] | None:
    """Try inexpensive one-joint-at-a-time overhead detours first."""

    overhead_z = max(
        float(start.base_pose_world[2]),
        float(goal.base_pose_world[2]),
        float(sampling_center[2]) + float(config.sampling_margin_m),
    )
    start_overhead = Order9ConfigurationState(
        base_pose_world=(
            float(start.base_pose_world[0]),
            float(start.base_pose_world[1]),
            overhead_z,
            *start.base_pose_world[3:],
        ),
        joint_positions_rad=dict(start.joint_positions_rad),
    )
    goal_overhead_with_start_q = Order9ConfigurationState(
        base_pose_world=(
            float(goal.base_pose_world[0]),
            float(goal.base_pose_world[1]),
            overhead_z,
            *goal.base_pose_world[3:],
        ),
        joint_positions_rad=dict(start.joint_positions_rad),
    )
    goal_overhead = Order9ConfigurationState(
        base_pose_world=goal_overhead_with_start_q.base_pose_world,
        joint_positions_rad=dict(goal.joint_positions_rad),
    )
    deltas = {
        joint_id: abs(
            float(goal.joint_positions_rad[joint_id])
            - float(start.joint_positions_rad[joint_id])
        )
        for joint_id in ordered_ids
    }
    raw_orders = (
        tuple(ordered_ids),
        tuple(reversed(ordered_ids)),
        tuple(sorted(ordered_ids, key=lambda value: (-deltas[value], value))),
        tuple(sorted(ordered_ids, key=lambda value: (deltas[value], value))),
    )
    orders: list[tuple[str, ...]] = []
    for order in raw_orders:
        if order not in orders:
            orders.append(order)

    for move_before_articulation in (True, False):
        for order in orders:
            path = [start, start_overhead]
            current = start_overhead
            if move_before_articulation:
                path.append(goal_overhead_with_start_q)
                current = goal_overhead_with_start_q
            q = dict(current.joint_positions_rad)
            for joint_id in order:
                q[joint_id] = float(goal.joint_positions_rad[joint_id])
                next_state = Order9ConfigurationState(
                    base_pose_world=current.base_pose_world,
                    joint_positions_rad=dict(q),
                )
                if not _states_close(current, next_state, ordered_ids=ordered_ids):
                    path.append(next_state)
                    current = next_state
            if not move_before_articulation:
                path.append(goal_overhead)
            path.append(goal)
            compact = _join_configuration_paths(path, ordered_ids=ordered_ids)
            if all(
                _edge_is_free(
                    left,
                    right,
                    ordered_ids=ordered_ids,
                    config=config,
                    checked=checked,
                )
                for left, right in zip(compact, compact[1:])
            ):
                # The one-axis construction is only a deterministic witness.
                # Collapse any non-adjacent waypoints whose complete direct
                # edge passes the same oracle.  This preserves collision
                # guarantees while avoiding dozens of unnecessary rolling
                # windows for seven- and eight-module morphologies.
                overhead = _shortcut_path(
                    compact[1:-1],
                    ordered_ids=ordered_ids,
                    config=config,
                    checked=checked,
                )
                return _join_configuration_paths(
                    (compact[0],),
                    overhead,
                    (compact[-1],),
                    ordered_ids=ordered_ids,
                )
    return None


def _join_configuration_paths(
    *paths: Sequence[Order9ConfigurationState],
    ordered_ids: Sequence[str],
) -> list[Order9ConfigurationState]:
    output: list[Order9ConfigurationState] = []
    for path in paths:
        for state in path:
            if not output or not _states_close(
                output[-1], state, ordered_ids=ordered_ids
            ):
                output.append(state)
    return output


def _aligned_for_overhead_descent(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
) -> bool:
    """Allow only the already-aligned final vertical approach to stay direct."""

    lateral_distance = math.hypot(
        float(goal.base_pose_world[0]) - float(start.base_pose_world[0]),
        float(goal.base_pose_world[1]) - float(start.base_pose_world[1]),
    )
    return bool(
        float(start.base_pose_world[2])
        >= float(goal.base_pose_world[2]) - config.waypoint_translation_tolerance_m
        and lateral_distance <= config.waypoint_translation_tolerance_m
        and _quaternion_distance(
            start.base_pose_world[3:],
            goal.base_pose_world[3:],
        )
        <= config.waypoint_attitude_tolerance_rad
        and max(
            (
                abs(
                    float(goal.joint_positions_rad[joint_id])
                    - float(start.joint_positions_rad[joint_id])
                )
                for joint_id in ordered_ids
            ),
            default=0.0,
        )
        <= config.waypoint_joint_tolerance_rad
    )


def _extend_once(
    tree: _Tree,
    target: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> int | None:
    nearest_index = min(
        range(len(tree.states)),
        key=lambda index: (
            _distance_units(
                tree.states[index],
                target,
                ordered_ids=ordered_ids,
                config=config,
            ),
            index,
        ),
    )
    nearest = tree.states[nearest_index]
    candidate = _steer(
        nearest,
        target,
        ordered_ids=ordered_ids,
        config=config,
    )
    if _states_close(nearest, candidate, ordered_ids=ordered_ids):
        return None
    if not _edge_is_free(
        nearest,
        candidate,
        ordered_ids=ordered_ids,
        config=config,
        checked=checked,
    ):
        return None
    tree.states.append(candidate)
    tree.parents.append(nearest_index)
    return len(tree.states) - 1


def _connect(
    tree: _Tree,
    target: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> int | None:
    for _ in range(config.maximum_connect_extensions):
        index = _extend_once(
            tree,
            target,
            ordered_ids=ordered_ids,
            config=config,
            checked=checked,
        )
        if index is None:
            return None
        if _states_close(
            tree.states[index],
            target,
            ordered_ids=ordered_ids,
        ):
            return index
    return None


def _root_path(tree: _Tree, index: int) -> list[Order9ConfigurationState]:
    path = []
    current: int | None = index
    while current is not None:
        path.append(tree.states[current])
        current = tree.parents[current]
    path.reverse()
    return path


def _shortcut_path(
    path: Sequence[Order9ConfigurationState],
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> list[Order9ConfigurationState]:
    if len(path) <= 2:
        return list(path)
    shortened = [path[0]]
    left = 0
    while left < len(path) - 1:
        accepted = left + 1
        for right in range(len(path) - 1, left, -1):
            if _edge_is_free(
                path[left],
                path[right],
                ordered_ids=ordered_ids,
                config=config,
                checked=checked,
            ):
                accepted = right
                break
        shortened.append(path[accepted])
        left = accepted
    return shortened


def _edge_is_free(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
    checked: Callable[[Order9ConfigurationState], bool],
) -> bool:
    intervals = max(
        1,
        int(
            math.ceil(
                _translation_distance(start, goal)
                / config.base_collision_sample_spacing_m
            )
        ),
        int(
            math.ceil(
                _quaternion_distance(
                    start.base_pose_world[3:],
                    goal.base_pose_world[3:],
                )
                / config.attitude_collision_sample_spacing_rad
            )
        ),
        int(
            math.ceil(
                max(
                    (
                        abs(
                            float(goal.joint_positions_rad[joint_id])
                            - float(start.joint_positions_rad[joint_id])
                        )
                        for joint_id in ordered_ids
                    ),
                    default=0.0,
                )
                / config.joint_collision_sample_spacing_rad
            )
        ),
    )
    for index in range(1, intervals + 1):
        state = _interpolate_state(
            start,
            goal,
            float(index) / float(intervals),
            ordered_ids=ordered_ids,
        )
        if not checked(state):
            return False
    return True


def _steer(
    start: Order9ConfigurationState,
    target: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
) -> Order9ConfigurationState:
    units = _distance_units(
        start,
        target,
        ordered_ids=ordered_ids,
        config=config,
    )
    ratio = 1.0 if units <= 1.0 else 1.0 / units
    return _interpolate_state(
        start,
        target,
        ratio,
        ordered_ids=ordered_ids,
    )


def _distance_units(
    left: Order9ConfigurationState,
    right: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
) -> float:
    return max(
        _translation_distance(left, right) / config.base_extension_step_m,
        _quaternion_distance(
            left.base_pose_world[3:],
            right.base_pose_world[3:],
        )
        / config.attitude_extension_step_rad,
        max(
            (
                abs(
                    float(right.joint_positions_rad[joint_id])
                    - float(left.joint_positions_rad[joint_id])
                )
                / config.joint_extension_step_rad
                for joint_id in ordered_ids
            ),
            default=0.0,
        ),
    )


def _sample_state(
    *,
    iteration: int,
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    ordered_ids: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
    sampling_center: tuple[float, float, float],
    config: Order9ConfigurationSpacePlannerConfig,
    minimum_base_z: float | None = None,
) -> Order9ConfigurationState:
    sample_index = iteration + 1 + config.deterministic_seed * config.maximum_iterations
    margin = float(config.sampling_margin_m)
    lower_xyz = tuple(
        min(
            float(start.base_pose_world[index]),
            float(goal.base_pose_world[index]),
            float(sampling_center[index]),
        )
        - margin
        for index in range(3)
    )
    upper_xyz = tuple(
        max(
            float(start.base_pose_world[index]),
            float(goal.base_pose_world[index]),
            float(sampling_center[index]),
        )
        + margin
        for index in range(3)
    )
    if minimum_base_z is not None:
        minimum_z = float(minimum_base_z)
        if not math.isfinite(minimum_z):
            raise ValueError("minimum_base_z must be finite")
        lower_xyz = (
            lower_xyz[0],
            lower_xyz[1],
            max(lower_xyz[2], minimum_z),
        )
        upper_xyz = (
            upper_xyz[0],
            upper_xyz[1],
            max(upper_xyz[2], minimum_z),
        )
    xyz = tuple(
        lower_xyz[index]
        + _halton(sample_index, _prime(index)) * (upper_xyz[index] - lower_xyz[index])
        for index in range(3)
    )
    blend = _halton(sample_index, _prime(3))
    orientation = _quaternion_slerp(
        start.base_pose_world[3:],
        goal.base_pose_world[3:],
        blend,
    )
    full_limit_sample = iteration % 5 == 4
    jitter_scale = (0.0, 0.12, 0.24, 0.36)[iteration % 4]
    joints = {}
    for joint_index, joint_id in enumerate(ordered_ids):
        lower, upper = limits[joint_id]
        unit = _halton(sample_index, _prime(joint_index + 4))
        if full_limit_sample:
            value = lower + unit * (upper - lower)
        else:
            reference = (1.0 - blend) * float(
                start.joint_positions_rad[joint_id]
            ) + blend * float(goal.joint_positions_rad[joint_id])
            value = reference + (2.0 * unit - 1.0) * jitter_scale * (upper - lower)
        joints[joint_id] = min(max(value, lower), upper)
    return Order9ConfigurationState(
        base_pose_world=(*xyz, *orientation),
        joint_positions_rad=joints,
    )


def _sampling_center(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    *,
    sampling_center_world: Sequence[float] | None,
) -> tuple[float, float, float]:
    if sampling_center_world is None:
        return tuple(
            0.5
            * (float(start.base_pose_world[index]) + float(goal.base_pose_world[index]))
            for index in range(3)
        )
    if len(sampling_center_world) < 3 or any(
        not math.isfinite(float(value)) for value in sampling_center_world[:3]
    ):
        raise SchemaValidationError(
            "configuration-space sampling center must contain finite xyz"
        )
    return tuple(float(value) for value in sampling_center_world[:3])


def _interpolate_state(
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    ratio: float,
    *,
    ordered_ids: Sequence[str],
) -> Order9ConfigurationState:
    value = min(max(float(ratio), 0.0), 1.0)
    pose = (
        *(
            float(start.base_pose_world[index])
            + value
            * (float(goal.base_pose_world[index]) - float(start.base_pose_world[index]))
            for index in range(3)
        ),
        *_quaternion_slerp(
            start.base_pose_world[3:],
            goal.base_pose_world[3:],
            value,
        ),
    )
    joints = {
        joint_id: (
            float(start.joint_positions_rad[joint_id])
            + value
            * (
                float(goal.joint_positions_rad[joint_id])
                - float(start.joint_positions_rad[joint_id])
            )
        )
        for joint_id in ordered_ids
    }
    return Order9ConfigurationState(
        base_pose_world=pose,
        joint_positions_rad=joints,
    )


def _translation_distance(
    left: Order9ConfigurationState,
    right: Order9ConfigurationState,
) -> float:
    return math.sqrt(
        sum(
            (float(right.base_pose_world[index]) - float(left.base_pose_world[index]))
            ** 2
            for index in range(3)
        )
    )


def _states_close(
    left: Order9ConfigurationState,
    right: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
) -> bool:
    return bool(
        _translation_distance(left, right) <= 1.0e-9
        and _quaternion_distance(
            left.base_pose_world[3:],
            right.base_pose_world[3:],
        )
        <= 1.0e-9
        and max(
            (
                abs(
                    float(right.joint_positions_rad[joint_id])
                    - float(left.joint_positions_rad[joint_id])
                )
                for joint_id in ordered_ids
            ),
            default=0.0,
        )
        <= 1.0e-9
    )


def _waypoints_equivalent(
    left: Order9ConfigurationState,
    right: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    config: Order9ConfigurationSpacePlannerConfig,
) -> bool:
    """Bound receding-horizon completion without weakening edge checks."""

    return bool(
        _translation_distance(left, right) <= config.waypoint_translation_tolerance_m
        and _quaternion_distance(
            left.base_pose_world[3:],
            right.base_pose_world[3:],
        )
        <= config.waypoint_attitude_tolerance_rad
        and max(
            (
                abs(
                    float(right.joint_positions_rad[joint_id])
                    - float(left.joint_positions_rad[joint_id])
                )
                for joint_id in ordered_ids
            ),
            default=0.0,
        )
        <= config.waypoint_joint_tolerance_rad
    )


def _quaternion_slerp(
    left: Sequence[float],
    right: Sequence[float],
    ratio: float,
) -> tuple[float, float, float, float]:
    q0 = _normalized_quaternion(left)
    q1 = _normalized_quaternion(right)
    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalized_quaternion(
            tuple(q0[index] + ratio * (q1[index] - q0[index]) for index in range(4))
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return _normalized_quaternion(
        tuple(
            math.sin((1.0 - ratio) * theta) / sin_theta * q0[index]
            + math.sin(ratio * theta) / sin_theta * q1[index]
            for index in range(4)
        )
    )


def _quaternion_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    q0 = _normalized_quaternion(left)
    q1 = _normalized_quaternion(right)
    dot = min(
        1.0,
        abs(sum(a * b for a, b in zip(q0, q1, strict=True))),
    )
    return 2.0 * math.acos(dot)


def _normalized_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise SchemaValidationError("configuration quaternion must contain four values")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 1.0e-12:
        raise SchemaValidationError("configuration quaternion norm must be positive")
    normalized = tuple(float(value) / norm for value in values)
    if normalized[3] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def _validate_pose(pose: Pose7D) -> None:
    if len(pose) != 7 or any(not math.isfinite(float(value)) for value in pose):
        raise SchemaValidationError("configuration base pose must be a finite Pose7D")
    _normalized_quaternion(pose[3:])


_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
)


def _prime(index: int) -> int:
    if index >= len(_PRIMES):
        raise SchemaValidationError(
            "configuration planner exceeds its deterministic Halton dimension"
        )
    return _PRIMES[index]


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    value = int(index)
    while value > 0:
        fraction /= float(base)
        result += fraction * float(value % base)
        value //= base
    return result


__all__ = [
    "ORDER9_CONFIGURATION_SPACE_PLANNER_VERSION",
    "DeterministicOrder9ConfigurationSpacePlanner",
    "Order9ConfigurationSpacePlan",
    "Order9ConfigurationSpacePlannerConfig",
    "Order9ConfigurationSpacePlanningError",
    "Order9ConfigurationState",
    "order9_lightweight_configuration_planning",
]
