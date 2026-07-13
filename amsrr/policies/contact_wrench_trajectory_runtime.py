from __future__ import annotations

"""Policy-agnostic rolling runtime for ``ContactWrenchTrajectory`` plans."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
)


class ContactWrenchTrajectoryRuntimeError(RuntimeError):
    """Raised when a trajectory cannot be installed or sampled safely."""


@dataclass(frozen=True)
class ContactWrenchTrajectorySample:
    active_knot: InteractionKnot
    plan_start_time_s: float
    plan_elapsed_s: float
    active_knot_index: int
    next_knot_index: int
    interpolation_ratio: float
    plan_sequence: int


class ContactWrenchTrajectoryExecutor:
    """Own the relative-time semantics between a pi_H plan and pi_L.

    The persisted trajectory remains relative to its own rolling-plan origin.
    This executor deliberately receives episode-absolute time only at install
    and sample boundaries, so downstream code never compares ``t_rel_s`` with
    ``RuntimeObservation.time_s`` directly.
    """

    def __init__(self, *, expiry_grace_s: float = 0.25) -> None:
        if not math.isfinite(float(expiry_grace_s)) or expiry_grace_s < 0.0:
            raise ValueError("expiry_grace_s must be finite and non-negative")
        self.expiry_grace_s = float(expiry_grace_s)
        self._trajectory: ContactWrenchTrajectory | None = None
        self._plan_start_time_s: float | None = None
        self._plan_sequence = 0

    @property
    def has_plan(self) -> bool:
        return self._trajectory is not None

    @property
    def trajectory(self) -> ContactWrenchTrajectory | None:
        return self._trajectory

    @property
    def plan_start_time_s(self) -> float | None:
        return self._plan_start_time_s

    @property
    def plan_sequence(self) -> int:
        return self._plan_sequence

    def reset(self) -> None:
        self._trajectory = None
        self._plan_start_time_s = None
        self._plan_sequence = 0

    def install(
        self,
        trajectory: ContactWrenchTrajectory,
        *,
        plan_start_time_s: float,
    ) -> None:
        _validate_executable_trajectory(trajectory)
        if not math.isfinite(float(plan_start_time_s)) or plan_start_time_s < 0.0:
            raise ContactWrenchTrajectoryRuntimeError(
                "plan_start_time_s must be finite and non-negative"
            )
        # Round-trip through the schema to detach runtime state from a mutable
        # policy-owned object before it is exposed to a lower-level consumer.
        self._trajectory = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
        self._plan_start_time_s = float(plan_start_time_s)
        self._plan_sequence += 1

    def sample(self, *, time_s: float) -> ContactWrenchTrajectorySample:
        trajectory = self._trajectory
        start_s = self._plan_start_time_s
        if trajectory is None or start_s is None:
            raise ContactWrenchTrajectoryRuntimeError("no trajectory is installed")
        if not math.isfinite(float(time_s)):
            raise ContactWrenchTrajectoryRuntimeError("sample time must be finite")
        elapsed_s = float(time_s) - start_s
        if elapsed_s < -1.0e-9:
            raise ContactWrenchTrajectoryRuntimeError(
                "sample time precedes the rolling-plan time origin"
            )
        elapsed_s = max(0.0, elapsed_s)
        if elapsed_s > trajectory.horizon_s + self.expiry_grace_s + 1.0e-9:
            raise ContactWrenchTrajectoryRuntimeError(
                "installed ContactWrenchTrajectory expired before replanning"
            )
        clamped_s = min(elapsed_s, trajectory.horizon_s)
        knots = trajectory.knots
        left_index = 0
        for index, knot in enumerate(knots):
            if knot.t_rel_s <= clamped_s + 1.0e-12:
                left_index = index
            else:
                break
        right_index = min(left_index + 1, len(knots) - 1)
        left = knots[left_index]
        right = knots[right_index]
        if left_index == right_index:
            ratio = 0.0
        else:
            duration = right.t_rel_s - left.t_rel_s
            ratio = min(max((clamped_s - left.t_rel_s) / duration, 0.0), 1.0)
        active = _interpolate_interaction_knot(left, right, ratio, clamped_s)
        return ContactWrenchTrajectorySample(
            active_knot=active,
            plan_start_time_s=start_s,
            plan_elapsed_s=clamped_s,
            active_knot_index=left_index,
            next_knot_index=right_index,
            interpolation_ratio=ratio,
            plan_sequence=self._plan_sequence,
        )


def _validate_executable_trajectory(trajectory: ContactWrenchTrajectory) -> None:
    try:
        trajectory.validate()
    except SchemaValidationError as exc:
        raise ContactWrenchTrajectoryRuntimeError(str(exc)) from exc
    if not trajectory.knots:
        raise ContactWrenchTrajectoryRuntimeError(
            "ContactWrenchTrajectory must contain at least one knot"
        )
    times = [float(knot.t_rel_s) for knot in trajectory.knots]
    if not all(math.isfinite(value) for value in times):
        raise ContactWrenchTrajectoryRuntimeError("trajectory knot times must be finite")
    if abs(times[0]) > 1.0e-9:
        raise ContactWrenchTrajectoryRuntimeError(
            "rolling ContactWrenchTrajectory must start at t_rel_s=0"
        )
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ContactWrenchTrajectoryRuntimeError(
            "trajectory knot times must be strictly increasing"
        )
    if times[-1] > trajectory.horizon_s + 1.0e-9:
        raise ContactWrenchTrajectoryRuntimeError(
            "trajectory knot time exceeds horizon_s"
        )
    if trajectory.horizon_s - times[-1] > trajectory.dt_s + 1.0e-9:
        raise ContactWrenchTrajectoryRuntimeError(
            "trajectory leaves more than one dt_s uncovered at the end of its horizon"
        )
    _require_finite_payload(trajectory.to_dict(), "ContactWrenchTrajectory")
    for knot in trajectory.knots:
        target = knot.centroidal_target
        if target is None or target.body_orientation_world is None:
            continue
        norm = math.sqrt(
            sum(float(value) ** 2 for value in target.body_orientation_world)
        )
        if norm <= 1.0e-9:
            raise ContactWrenchTrajectoryRuntimeError(
                "centroidal target quaternion must have non-zero norm"
            )


def _require_finite_payload(value: object, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ContactWrenchTrajectoryRuntimeError(f"{path} contains non-finite data")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_payload(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_payload(item, f"{path}[{index}]")


def _interpolate_interaction_knot(
    left: InteractionKnot,
    right: InteractionKnot,
    ratio: float,
    sample_t_rel_s: float,
) -> InteractionKnot:
    use_right_discrete = ratio >= 1.0 - 1.0e-12
    discrete = right if use_right_discrete else left
    priority_keys = set(left.priority_weights) | set(right.priority_weights)
    priorities = {
        key: _lerp(
            float(left.priority_weights.get(key, 0.0)),
            float(right.priority_weights.get(key, 0.0)),
            ratio,
        )
        for key in sorted(priority_keys)
    }
    return InteractionKnot(
        t_rel_s=float(sample_t_rel_s),
        contact_assignments=[
            type(assignment).from_dict(assignment.to_dict())
            for assignment in discrete.contact_assignments
        ],
        centroidal_target=_interpolate_centroidal_target(
            left.centroidal_target,
            right.centroidal_target,
            ratio,
        ),
        posture_target=(
            None
            if discrete.posture_target is None
            else type(discrete.posture_target).from_dict(
                discrete.posture_target.to_dict()
            )
        ),
        object_targets=[
            type(target).from_dict(target.to_dict())
            for target in discrete.object_targets
        ],
        priority_weights=priorities,
        guard_conditions=[dict(condition) for condition in discrete.guard_conditions],
    )


def _interpolate_centroidal_target(
    left: CentroidalTarget | None,
    right: CentroidalTarget | None,
    ratio: float,
) -> CentroidalTarget | None:
    if left is None and right is None:
        return None
    if left is None:
        return CentroidalTarget.from_dict(right.to_dict())  # type: ignore[union-attr]
    if right is None:
        return CentroidalTarget.from_dict(left.to_dict())
    return CentroidalTarget(
        com_pos_world=_interpolate_optional_tuple(
            left.com_pos_world,
            right.com_pos_world,
            ratio,
        ),
        com_vel_world=_interpolate_optional_tuple(
            left.com_vel_world,
            right.com_vel_world,
            ratio,
        ),
        body_orientation_world=_interpolate_optional_quaternion(
            left.body_orientation_world,
            right.body_orientation_world,
            ratio,
        ),
        centroidal_wrench_preference=_interpolate_optional_list(
            left.centroidal_wrench_preference,
            right.centroidal_wrench_preference,
            ratio,
        ),
    )


def _interpolate_optional_tuple(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    ratio: float,
) -> tuple[float, ...] | None:
    values = _interpolate_optional_sequence(left, right, ratio)
    return None if values is None else tuple(values)


def _interpolate_optional_list(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    ratio: float,
) -> list[float] | None:
    values = _interpolate_optional_sequence(left, right, ratio)
    return None if values is None else list(values)


def _interpolate_optional_sequence(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    ratio: float,
) -> list[float] | None:
    if left is None and right is None:
        return None
    if left is None:
        return [float(value) for value in right or ()]
    if right is None:
        return [float(value) for value in left]
    if len(left) != len(right):
        raise ContactWrenchTrajectoryRuntimeError(
            "cannot interpolate target sequences with different lengths"
        )
    return [
        _lerp(float(left_value), float(right_value), ratio)
        for left_value, right_value in zip(left, right, strict=True)
    ]


def _interpolate_optional_quaternion(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
    ratio: float,
) -> tuple[float, float, float, float] | None:
    if left is None and right is None:
        return None
    if left is None:
        return _normalize_quaternion(right or ())
    if right is None:
        return _normalize_quaternion(left)
    return _quaternion_slerp(left, right, ratio)


def _quaternion_slerp(
    left: Sequence[float],
    right: Sequence[float],
    ratio: float,
) -> tuple[float, float, float, float]:
    start = _normalize_quaternion(left)
    end = _normalize_quaternion(right)
    dot = sum(a * b for a, b in zip(start, end, strict=True))
    if dot < 0.0:
        end = tuple(-value for value in end)
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(_lerp(a, b, ratio) for a, b in zip(start, end, strict=True))
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    start_scale = math.sin((1.0 - ratio) * theta) / sin_theta
    end_scale = math.sin(ratio * theta) / sin_theta
    return _normalize_quaternion(
        tuple(
            start_scale * a + end_scale * b
            for a, b in zip(start, end, strict=True)
        )
    )


def _normalize_quaternion(
    values: Iterable[float],
) -> tuple[float, float, float, float]:
    data = tuple(float(value) for value in values)
    if len(data) != 4:
        raise ContactWrenchTrajectoryRuntimeError(
            "quaternion interpolation requires four values"
        )
    norm = math.sqrt(sum(value * value for value in data))
    if norm <= 1.0e-12:
        raise ContactWrenchTrajectoryRuntimeError(
            "quaternion interpolation received zero norm"
        )
    return tuple(value / norm for value in data)  # type: ignore[return-value]


def _lerp(left: float, right: float, ratio: float) -> float:
    return left + (right - left) * ratio
