from __future__ import annotations

"""Task-adapter-owned phase/reset schedule for Order 9 grasp-and-carry.

This is the deterministic fallback/reference used while ``pi_L`` is trained
and when ``pi_H`` is unavailable.  It is not the learned ``pi_H`` policy and
does not inspect privileged contact truth.  A learned, hard-checked trajectory
can replace the returned target at execution time without changing phase or
reset semantics.
"""

import math
from dataclasses import dataclass, field
from typing import Mapping

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, StrEnum
from amsrr.simulation.order9_object_task_state import (
    Order9CanonicalObjectTaskReset,
)


ORDER9_OBJECT_TASK_RUNTIME_VERSION = "order9_object_grasp_carry_runtime_v1"
ORDER9_OBJECT_TASK_ADAPTER_ID = "object_grasp_carry_v1"

# Actor-visible phase identity is the task-adapter contract used by C0 teacher
# records.  The deterministic target generator below has fewer executable
# phases because ``apply_wrench`` is folded into contact acquisition and
# ``complete``/``safe_hold`` are terminal states.  Never feed the compact
# runtime index directly to a learned policy.
ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS = (
    "approach",
    "establish_contact",
    "apply_wrench",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
    "complete",
    "safe_hold",
)
ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT = len(
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS
)


class Order9ObjectTaskPhase(StrEnum):
    APPROACH = "approach"
    CONTACT_ACQUISITION = "contact_acquisition"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    SETTLE = "settle"


ORDER9_OBJECT_TASK_PHASES = tuple(Order9ObjectTaskPhase)
ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME = (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("approach"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("establish_contact"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("lift"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("transport"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("place"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("release"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("retreat"),
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS.index("settle"),
)


def order9_object_task_actor_phase_index(runtime_phase_index: int) -> int:
    """Map compact target-generator phase identity to the actor contract."""

    if not 0 <= int(runtime_phase_index) < len(
        ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
    ):
        raise SchemaValidationError(
            "Order9 object-task runtime phase index is invalid"
        )
    return ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME[
        int(runtime_phase_index)
    ]


@dataclass
class Order9ObjectTaskRuntimeConfig(SchemaBase):
    phase_duration_s: dict[str, float] = field(
        default_factory=lambda: {
            # Match the normalized phase clocks used by the real-Isaac C0
            # Order 8 teacher: 30 s normally and 90 s while acquiring contact.
            # These are timeout/progress horizons, not forced dwell times;
            # phase success still advances immediately when its physical gates
            # pass.
            Order9ObjectTaskPhase.APPROACH.value: 30.0,
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value: 90.0,
            Order9ObjectTaskPhase.LIFT.value: 30.0,
            Order9ObjectTaskPhase.TRANSPORT.value: 30.0,
            Order9ObjectTaskPhase.PLACE.value: 30.0,
            Order9ObjectTaskPhase.RELEASE.value: 30.0,
            Order9ObjectTaskPhase.RETREAT.value: 30.0,
            Order9ObjectTaskPhase.SETTLE.value: 30.0,
        }
    )
    approach_offset_m: float = 0.30
    retreat_offset_m: float = 0.10
    command_translation_speed_limit_mps: float = 0.10
    contact_joint_velocity_limit_radps: float = 0.12
    release_joint_velocity_limit_radps: float = 0.12
    phase_specific_resets: bool = True
    raw_contact_actor_input: bool = False
    runtime_version: str = ORDER9_OBJECT_TASK_RUNTIME_VERSION

    def validate(self) -> None:
        if self.runtime_version != ORDER9_OBJECT_TASK_RUNTIME_VERSION:
            raise SchemaValidationError("Order9 object-task runtime version mismatch")
        expected = {phase.value for phase in ORDER9_OBJECT_TASK_PHASES}
        if set(self.phase_duration_s) != expected:
            raise SchemaValidationError(
                "Order9 object-task durations must cover every phase exactly"
            )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.phase_duration_s.values()
        ):
            raise SchemaValidationError(
                "Order9 object-task phase durations must be positive"
            )
        for name in (
            "approach_offset_m",
            "retreat_offset_m",
            "command_translation_speed_limit_mps",
            "contact_joint_velocity_limit_radps",
            "release_joint_velocity_limit_radps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9 object-task {name} must be positive"
                )
        if not self.phase_specific_resets:
            raise SchemaValidationError(
                "Order9 production training requires phase-specific resets"
            )
        if self.raw_contact_actor_input:
            raise SchemaValidationError(
                "Order9 object-task actor must not consume raw contact truth"
            )


@dataclass(frozen=True)
class Order9ObjectTaskTarget:
    phase: Order9ObjectTaskPhase
    phase_index: int
    phase_count: int
    phase_elapsed_s: float
    phase_progress: float
    desired_robot_root_pose_world: Pose7D
    desired_robot_root_twist_world: tuple[float, float, float, float, float, float]
    nominal_joint_positions_rad: dict[str, float]
    nominal_joint_velocities_radps: dict[str, float]
    desired_object_pose_world: Pose7D
    contact_schedule_state: str


@dataclass(frozen=True)
class Order9ObjectTaskPhaseReset:
    phase: Order9ObjectTaskPhase
    phase_index: int
    robot_root_pose_world: Pose7D
    robot_root_twist_world: tuple[float, float, float, float, float, float]
    joint_positions_rad: dict[str, float]
    joint_velocities_radps: dict[str, float]
    object_pose_world: Pose7D
    object_twist_world: tuple[float, float, float, float, float, float]
    reset_labels_reused: bool = False


class Order9ObjectTaskRuntime:
    """Pure phase schedule shared by tensor training and the shadow worker."""

    def __init__(
        self,
        canonical: Order9CanonicalObjectTaskReset,
        *,
        config: Order9ObjectTaskRuntimeConfig | None = None,
    ) -> None:
        canonical.validate()
        self.canonical = canonical
        self.config = config or Order9ObjectTaskRuntimeConfig()
        self.config.validate()

    @property
    def phase_count(self) -> int:
        return len(ORDER9_OBJECT_TASK_PHASES)

    def phase(self, phase_index: int) -> Order9ObjectTaskPhase:
        if not 0 <= phase_index < self.phase_count:
            raise SchemaValidationError("Order9 object-task phase index is invalid")
        return ORDER9_OBJECT_TASK_PHASES[phase_index]

    def duration_s(self, phase_index: int) -> float:
        return float(self.config.phase_duration_s[self.phase(phase_index).value])

    def reset_for_phase(
        self,
        phase_index: int,
        *,
        object_position_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
        object_yaw_offset_rad: float = 0.0,
    ) -> Order9ObjectTaskPhaseReset:
        phase = self.phase(phase_index)
        _finite_vector(object_position_offset_world, 3, "object_position_offset_world")
        if not math.isfinite(float(object_yaw_offset_rad)):
            raise SchemaValidationError("Order9 reset yaw offset must be finite")
        robot_pose, object_pose, joints = self._phase_start_state(phase)
        object_pose = _offset_pose(
            object_pose,
            object_position_offset_world,
            yaw_offset_rad=object_yaw_offset_rad,
        )
        # Preserve the physical relative reset for phases that begin in grasp.
        # Pose offsets represent randomized fixture placement, not a teleport of
        # the payload alone after contact has already been established.
        if phase in {
            Order9ObjectTaskPhase.LIFT,
            Order9ObjectTaskPhase.TRANSPORT,
            Order9ObjectTaskPhase.PLACE,
            Order9ObjectTaskPhase.RELEASE,
        }:
            robot_pose = _offset_pose(
                robot_pose,
                object_position_offset_world,
                yaw_offset_rad=object_yaw_offset_rad,
            )
        return Order9ObjectTaskPhaseReset(
            phase=phase,
            phase_index=phase_index,
            robot_root_pose_world=robot_pose,
            robot_root_twist_world=(0.0,) * 6,
            joint_positions_rad=joints,
            joint_velocities_radps={key: 0.0 for key in joints},
            object_pose_world=object_pose,
            object_twist_world=(0.0,) * 6,
            reset_labels_reused=False,
        )

    def target(
        self,
        phase_index: int,
        phase_elapsed_s: float,
        *,
        reset: Order9ObjectTaskPhaseReset | None = None,
    ) -> Order9ObjectTaskTarget:
        phase = self.phase(phase_index)
        if not math.isfinite(float(phase_elapsed_s)) or phase_elapsed_s < 0.0:
            raise SchemaValidationError(
                "Order9 object-task phase elapsed time must be non-negative"
            )
        duration = self.duration_s(phase_index)
        progress = min(float(phase_elapsed_s) / duration, 1.0)
        smooth = _smoothstep(progress)
        start = reset or self.reset_for_phase(phase_index)
        root_start = start.robot_root_pose_world
        object_start = start.object_pose_world
        root_end, object_end, q_end = self._phase_end_state(
            phase,
            root_start=root_start,
            object_start=object_start,
            joint_start=start.joint_positions_rad,
        )
        root_pose = _interpolate_pose(root_start, root_end, smooth)
        object_pose = _interpolate_pose(object_start, object_end, smooth)
        q = _interpolate_map(start.joint_positions_rad, q_end, smooth)
        root_linear = tuple(
            _bounded_phase_velocity(
                float(root_end[index]) - float(root_start[index]),
                duration,
                progress,
                self.config.command_translation_speed_limit_mps,
            )
            for index in range(3)
        )
        qdot_limit = (
            self.config.release_joint_velocity_limit_radps
            if phase == Order9ObjectTaskPhase.RELEASE
            else self.config.contact_joint_velocity_limit_radps
        )
        qdot = {
            key: _bounded_phase_velocity(
                float(q_end[key]) - float(start.joint_positions_rad[key]),
                duration,
                progress,
                qdot_limit,
            )
            for key in q
        }
        return Order9ObjectTaskTarget(
            phase=phase,
            phase_index=phase_index,
            phase_count=self.phase_count,
            phase_elapsed_s=float(phase_elapsed_s),
            phase_progress=progress,
            desired_robot_root_pose_world=root_pose,
            desired_robot_root_twist_world=(*root_linear, 0.0, 0.0, 0.0),
            nominal_joint_positions_rad=q,
            nominal_joint_velocities_radps=qdot,
            desired_object_pose_world=object_pose,
            contact_schedule_state=_contact_schedule_state(phase),
        )

    def next_phase_index(self, phase_index: int) -> int | None:
        self.phase(phase_index)
        next_index = phase_index + 1
        return next_index if next_index < self.phase_count else None

    def _phase_start_state(
        self,
        phase: Order9ObjectTaskPhase,
    ) -> tuple[Pose7D, Pose7D, dict[str, float]]:
        base = tuple(float(value) for value in self.canonical.robot_root_pose_world)
        obj = tuple(float(value) for value in self.canonical.object_pose_world)
        q_close = dict(self.canonical.joint_positions_rad)
        q_open = dict(self.canonical.open_joint_positions_rad)
        lift = self.canonical.lift_clearance_m
        carry = self.canonical.transport_distance_m
        if phase == Order9ObjectTaskPhase.APPROACH:
            return _translated(base, -self.config.approach_offset_m, 0.0, 0.0), obj, q_open
        if phase == Order9ObjectTaskPhase.CONTACT_ACQUISITION:
            return base, obj, q_open
        if phase == Order9ObjectTaskPhase.LIFT:
            return base, obj, q_close
        if phase == Order9ObjectTaskPhase.TRANSPORT:
            return _translated(base, 0.0, 0.0, lift), _translated(obj, 0.0, 0.0, lift), q_close
        if phase == Order9ObjectTaskPhase.PLACE:
            return _translated(base, carry, 0.0, lift), _translated(obj, carry, 0.0, lift), q_close
        if phase == Order9ObjectTaskPhase.RELEASE:
            return _translated(base, carry, 0.0, 0.0), _translated(obj, carry, 0.0, 0.0), q_close
        if phase == Order9ObjectTaskPhase.RETREAT:
            return _translated(base, carry, 0.0, 0.0), _translated(obj, carry, 0.0, 0.0), q_open
        return (
            _translated(base, carry - self.config.retreat_offset_m, 0.0, 0.0),
            _translated(obj, carry, 0.0, 0.0),
            q_open,
        )

    def _phase_end_state(
        self,
        phase: Order9ObjectTaskPhase,
        *,
        root_start: Pose7D,
        object_start: Pose7D,
        joint_start: Mapping[str, float],
    ) -> tuple[Pose7D, Pose7D, dict[str, float]]:
        if phase == Order9ObjectTaskPhase.APPROACH:
            return (
                _translated(root_start, self.config.approach_offset_m, 0.0, 0.0),
                object_start,
                dict(self.canonical.open_joint_positions_rad),
            )
        if phase == Order9ObjectTaskPhase.CONTACT_ACQUISITION:
            return root_start, object_start, dict(self.canonical.joint_positions_rad)
        if phase == Order9ObjectTaskPhase.LIFT:
            return (
                _translated(root_start, 0.0, 0.0, self.canonical.lift_clearance_m),
                _translated(object_start, 0.0, 0.0, self.canonical.lift_clearance_m),
                dict(self.canonical.joint_positions_rad),
            )
        if phase == Order9ObjectTaskPhase.TRANSPORT:
            return (
                _translated(root_start, self.canonical.transport_distance_m, 0.0, 0.0),
                _translated(object_start, self.canonical.transport_distance_m, 0.0, 0.0),
                dict(self.canonical.joint_positions_rad),
            )
        if phase == Order9ObjectTaskPhase.PLACE:
            return (
                _translated(root_start, 0.0, 0.0, -self.canonical.lift_clearance_m),
                _translated(object_start, 0.0, 0.0, -self.canonical.lift_clearance_m),
                dict(self.canonical.joint_positions_rad),
            )
        if phase == Order9ObjectTaskPhase.RELEASE:
            return root_start, object_start, dict(self.canonical.open_joint_positions_rad)
        if phase == Order9ObjectTaskPhase.RETREAT:
            return (
                _translated(root_start, -self.config.retreat_offset_m, 0.0, 0.0),
                object_start,
                dict(self.canonical.open_joint_positions_rad),
            )
        return root_start, object_start, dict(joint_start)


def _contact_schedule_state(phase: Order9ObjectTaskPhase) -> str:
    if phase == Order9ObjectTaskPhase.APPROACH:
        return "approach"
    if phase == Order9ObjectTaskPhase.CONTACT_ACQUISITION:
        return "attach"
    if phase in {
        Order9ObjectTaskPhase.LIFT,
        Order9ObjectTaskPhase.TRANSPORT,
        Order9ObjectTaskPhase.PLACE,
    }:
        return "maintain"
    if phase == Order9ObjectTaskPhase.RELEASE:
        return "release"
    return "inactive"


def _smoothstep(value: float) -> float:
    clipped = min(max(float(value), 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _smoothstep_derivative(value: float) -> float:
    clipped = min(max(float(value), 0.0), 1.0)
    return 6.0 * clipped * (1.0 - clipped)


def _bounded_phase_velocity(
    displacement: float,
    duration_s: float,
    progress: float,
    limit: float,
) -> float:
    value = displacement / duration_s * _smoothstep_derivative(progress)
    return min(max(value, -float(limit)), float(limit))


def _translated(pose: Pose7D, x: float, y: float, z: float) -> Pose7D:
    return (
        float(pose[0]) + float(x),
        float(pose[1]) + float(y),
        float(pose[2]) + float(z),
        *tuple(float(value) for value in pose[3:7]),
    )


def _offset_pose(
    pose: Pose7D,
    offset: tuple[float, float, float],
    *,
    yaw_offset_rad: float,
) -> Pose7D:
    translated = _translated(pose, *offset)
    if math.isclose(yaw_offset_rad, 0.0, abs_tol=1.0e-15):
        return translated
    half = 0.5 * float(yaw_offset_rad)
    yaw = (0.0, 0.0, math.sin(half), math.cos(half))
    x1, y1, z1, w1 = translated[3:7]
    x2, y2, z2, w2 = yaw
    return (
        *translated[:3],
        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
    )


def _interpolate_pose(start: Pose7D, end: Pose7D, alpha: float) -> Pose7D:
    # Curriculum reference poses retain the canonical orientation.  Normalized
    # linear quaternion interpolation is still used so randomized yaw resets
    # remain well-defined without introducing an Euler convention.
    values = [
        (1.0 - alpha) * float(left) + alpha * float(right)
        for left, right in zip(start, end)
    ]
    norm = math.sqrt(sum(value * value for value in values[3:7]))
    if norm <= 1.0e-12:
        raise SchemaValidationError("Order9 interpolated pose quaternion is singular")
    values[3:7] = [value / norm for value in values[3:7]]
    return tuple(values)  # type: ignore[return-value]


def _interpolate_map(
    start: Mapping[str, float],
    end: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    if set(start) != set(end):
        raise SchemaValidationError("Order9 phase joint target ids changed")
    return {
        key: (1.0 - alpha) * float(start[key]) + alpha * float(end[key])
        for key in sorted(start)
    }


def _finite_vector(value: tuple[float, ...], length: int, label: str) -> None:
    if len(value) != length or any(not math.isfinite(float(item)) for item in value):
        raise SchemaValidationError(f"Order9 {label} must contain finite values")


__all__ = [
    "ORDER9_OBJECT_TASK_ADAPTER_ID",
    "ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT",
    "ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME",
    "ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS",
    "ORDER9_OBJECT_TASK_PHASES",
    "ORDER9_OBJECT_TASK_RUNTIME_VERSION",
    "Order9ObjectTaskPhase",
    "Order9ObjectTaskPhaseReset",
    "Order9ObjectTaskRuntime",
    "Order9ObjectTaskRuntimeConfig",
    "Order9ObjectTaskTarget",
    "order9_object_task_actor_phase_index",
]
