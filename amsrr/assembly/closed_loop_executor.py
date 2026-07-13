from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from amsrr.assembly.assembly_control_bridge import (
    AssemblyComponentCommandBundle,
    AssemblyControlBridge,
    AssemblyControlObservation,
    AssemblyControlStepOutput,
)
from amsrr.assembly.assembly_motion_planner import (
    DeterministicAssemblyMotionPlanner,
    AssemblyMotionPlanningError,
)
from amsrr.assembly.construction_state import AssemblyStep, ConstructionState
from amsrr.assembly.control_handoff import ControlHandoffManager
from amsrr.assembly.executor_interface import AssemblyExecutionResult
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.feasibility import Violation, ViolationSeverity
from amsrr.schemas.morphology import MorphologyGraph


class AssemblyControlRuntime(Protocol):
    """Backend boundary used by the stateful Order-5 executor."""

    @property
    def control_dt_s(self) -> float:
        ...

    def observe(self) -> AssemblyControlObservation:
        ...

    def apply_and_step(self, commands: AssemblyComponentCommandBundle) -> None:
        ...

    def is_component_pose_collision_free(
        self,
        component_id: str,
        pose_world: Pose7D,
    ) -> bool:
        ...


@dataclass(frozen=True)
class ClosedLoopAssemblyExecutorConfig:
    waypoint_position_tolerance_m: float = 0.03
    waypoint_attitude_tolerance_rad: float = math.radians(5.0)
    waypoint_speed_tolerance_mps: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise SchemaValidationError(
                    f"ClosedLoopAssemblyExecutorConfig.{name} must be finite and positive"
                )


class ClosedLoopAssemblyExecutor:
    """Adapt the stateful AssemblyControlBridge to AssemblyRunner's step API.

    One bridge session spans the four legacy attach steps.  Each call returns
    only after its physical boundary is reached, while the session and trace
    remain live for the next `AssemblyStep`.
    """

    def __init__(
        self,
        *,
        target_graph: MorphologyGraph,
        bridge: AssemblyControlBridge,
        runtime: AssemblyControlRuntime,
        handoff_manager: ControlHandoffManager | None = None,
        motion_planner: DeterministicAssemblyMotionPlanner | None = None,
        config: ClosedLoopAssemblyExecutorConfig | None = None,
    ) -> None:
        if not math.isfinite(float(runtime.control_dt_s)) or runtime.control_dt_s <= 0.0:
            raise SchemaValidationError("AssemblyControlRuntime.control_dt_s must be positive")
        self.target_graph = target_graph
        self.bridge = bridge
        self.runtime = runtime
        self.handoff_manager = handoff_manager or ControlHandoffManager()
        self.motion_planner = motion_planner or DeterministicAssemblyMotionPlanner()
        self.config = config or ClosedLoopAssemblyExecutorConfig()
        self.trace: list[AssemblyControlStepOutput] = []
        self._output: AssemblyControlStepOutput | None = None
        self._active_edge_ports: tuple[int, int] | None = None
        self._staging_waypoints: list[Pose7D] = []
        self._staging_waypoint_index = 0

    def execute_step(
        self,
        step: AssemblyStep,
        state: ConstructionState,
    ) -> AssemblyExecutionResult:
        if step.step_type in {"retry", "abort"}:
            return self._execute_recovery_step(step)
        if step.step_type == "detach":
            return _failure_result(
                step,
                "detach_requires_order7_release_executor",
            )
        if step.step_type not in {
            "move_to_staging",
            "align_ports",
            "dock",
            "verify_attach",
        }:
            return _failure_result(step, "unsupported_closed_loop_assembly_step")

        observation = self.runtime.observe()
        edge_ports = _step_ports(step)
        if self._output is None or self._active_edge_ports != edge_ports:
            request = self.handoff_manager.build_assembly_control_request(
                step,
                state,
                self.target_graph,
            )
            self._output = self.bridge.begin(request, observation)
            self._active_edge_ports = edge_ports
            try:
                self._prepare_staging_path(self._output, observation)
            except AssemblyMotionPlanningError as exc:
                self._output = self.bridge.enter_safe_hold(
                    observation,
                    reason="staging_motion_plan_failed",
                )
                self.trace.append(self._output)
                return _failure_result(
                    step,
                    f"staging_motion_plan_failed:{exc}",
                    phase=self._output.progress.phase,
                )
            self.trace.append(self._output)

        max_ticks = max(
            1,
            int(math.ceil(float(step.timeout_s) / float(self.runtime.control_dt_s))),
        )
        for tick_index in range(max_ticks):
            if self._output is None:
                raise RuntimeError("ClosedLoopAssemblyExecutor lost its active bridge output")
            commands = self._commands_for_current_path(self._output)
            self.runtime.apply_and_step(commands)
            observation = self.runtime.observe()
            self._advance_staging_path(observation)
            self._output = self.bridge.tick(observation)
            self.trace.append(self._output)
            if self._output.progress.failed:
                return _failure_result(
                    step,
                    self._output.progress.failure_reason or "assembly_control_bridge_failed",
                    control_tick_count=tick_index + 1,
                    phase=self._output.progress.phase,
                )
            if _step_boundary_reached(step.step_type, self._output):
                return AssemblyExecutionResult(
                    step_id=step.step_id,
                    success=True,
                    metrics={
                        "closed_loop_control_tick_count": float(tick_index + 1),
                        "assembly_phase_index": float(_phase_index(self._output.progress.phase)),
                        "assembly_constraint_create_intent": (
                            1.0
                            if self._output.commands.constraint_intent.action == "create"
                            else 0.0
                        ),
                    },
                    message=f"closed-loop boundary reached: {self._output.progress.phase}",
                )
        return _failure_result(
            step,
            f"closed_loop_step_timeout:{step.step_type}",
            control_tick_count=max_ticks,
            phase=self._output.progress.phase if self._output is not None else "unknown",
        )

    def _prepare_staging_path(
        self,
        output: AssemblyControlStepOutput,
        observation: AssemblyControlObservation,
    ) -> None:
        follower_target = next(
            target
            for target in output.commands.component_targets
            if target.role == "follower"
        )
        follower_observation = next(
            component
            for component in observation.components
            if component.component_id == follower_target.component_id
        )
        goal = follower_target.policy_command.desired_body_pose
        if goal is None:
            raise SchemaValidationError("Assembly follower staging target has no desired_body_pose")
        plan = self.motion_planner.plan(
            follower_observation.body_pose_world,
            goal,
            is_pose_collision_free=lambda pose: self.runtime.is_component_pose_collision_free(
                follower_target.component_id,
                pose,
            ),
        )
        self._staging_waypoints = list(plan.waypoints_world)
        self._staging_waypoint_index = 0

    def _commands_for_current_path(
        self,
        output: AssemblyControlStepOutput,
    ) -> AssemblyComponentCommandBundle:
        if (
            output.progress.phase != "staging"
            or self._staging_waypoint_index >= len(self._staging_waypoints)
        ):
            return output.commands
        waypoint = self._staging_waypoints[self._staging_waypoint_index]
        targets = []
        for target in output.commands.component_targets:
            if target.role == "follower":
                targets.append(
                    replace(
                        target,
                        policy_command=replace(
                            target.policy_command,
                            desired_body_pose=waypoint,
                        ),
                    )
                )
            else:
                targets.append(target)
        return replace(output.commands, component_targets=targets)

    def _advance_staging_path(self, observation: AssemblyControlObservation) -> None:
        if self._staging_waypoint_index >= len(self._staging_waypoints):
            return
        if self._output is None:
            return
        follower_target = next(
            target
            for target in self._output.commands.component_targets
            if target.role == "follower"
        )
        follower = next(
            component
            for component in observation.components
            if component.component_id == follower_target.component_id
        )
        waypoint = self._staging_waypoints[self._staging_waypoint_index]
        if (
            _position_error(follower.body_pose_world, waypoint)
            <= self.config.waypoint_position_tolerance_m
            and _attitude_error(follower.body_pose_world, waypoint)
            <= self.config.waypoint_attitude_tolerance_rad
            and _norm(follower.selected_connect_linear_velocity_world)
            <= self.config.waypoint_speed_tolerance_mps
        ):
            self._staging_waypoint_index += 1

    def _execute_recovery_step(self, step: AssemblyStep) -> AssemblyExecutionResult:
        if self._output is not None:
            observation = self.runtime.observe()
            self._output = self.bridge.enter_safe_hold(
                observation,
                reason=f"assembly_{step.step_type}",
            )
            self.trace.append(self._output)
            self.runtime.apply_and_step(self._output.commands)
        self._output = None
        self._active_edge_ports = None
        self._staging_waypoints = []
        self._staging_waypoint_index = 0
        return AssemblyExecutionResult(
            step_id=step.step_id,
            success=True,
            metrics={"closed_loop_recovery": 1.0},
            message=f"{step.step_type} entered deterministic safe hold",
        )


def _step_boundary_reached(step_type: str, output: AssemblyControlStepOutput) -> bool:
    phase = output.progress.phase
    if step_type == "move_to_staging":
        return phase == "prealign_dwell"
    if step_type == "align_ports":
        return phase == "axial_approach"
    if step_type == "dock":
        return phase == "verify"
    if step_type == "verify_attach":
        return output.progress.completed
    return False


def _step_ports(step: AssemblyStep) -> tuple[int, int]:
    if step.src_port_id is None or step.dst_port_id is None:
        raise SchemaValidationError("Closed-loop attach step requires both port ids")
    return tuple(sorted((int(step.src_port_id), int(step.dst_port_id))))


def _failure_result(
    step: AssemblyStep,
    reason: str,
    *,
    control_tick_count: int = 0,
    phase: str = "unknown",
) -> AssemblyExecutionResult:
    return AssemblyExecutionResult(
        step_id=step.step_id,
        success=False,
        violations=[
            Violation(
                code="E_ASSEMBLY_CLOSED_LOOP",
                severity=ViolationSeverity.HARD,
                message=f"{reason} (phase={phase})",
            )
        ],
        metrics={"closed_loop_control_tick_count": float(control_tick_count)},
        message=reason,
    )


def _phase_index(phase: str) -> int:
    phases = (
        "staging",
        "prealign_dwell",
        "axial_approach",
        "fix_ready",
        "verify",
        "safe_hold",
    )
    return phases.index(phase) if phase in phases else -1


def _position_error(left: Pose7D, right: Pose7D) -> float:
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _attitude_error(left: Pose7D, right: Pose7D) -> float:
    q0 = _normalize_quat(left[3:7])
    q1 = _normalize_quat(right[3:7])
    dot = min(1.0, abs(sum(a * b for a, b in zip(q0, q1, strict=True))))
    return 2.0 * math.acos(dot)


def _normalize_quat(values) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 0.0:
        raise SchemaValidationError("pose quaternion norm must be positive")
    return tuple(float(value) / norm for value in values)  # type: ignore[return-value]


def _norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


__all__ = [
    "AssemblyControlRuntime",
    "ClosedLoopAssemblyExecutor",
    "ClosedLoopAssemblyExecutorConfig",
]
