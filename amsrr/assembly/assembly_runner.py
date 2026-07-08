from __future__ import annotations

from dataclasses import dataclass, field, replace

from amsrr.assembly.construction_state import (
    AssemblyPlan,
    AssemblyStep,
    ConstructionState,
    initial_construction_state,
    mark_edge_attached,
)
from amsrr.assembly.executor_interface import AssemblyExecutionResult, AssemblyExecutorInterface
from amsrr.assembly.graph_edit_planner import GraphEditAssemblyPlanner
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.feasibility import Violation
from amsrr.schemas.morphology import DockEdge, MorphologyGraph


@dataclass(frozen=True)
class AssemblyRunnerConfig:
    max_step_count: int = 64
    max_retries_per_step: int = 1
    retry_timeout_s: float = 1.0
    abort_timeout_s: float = 1.0
    stop_on_first_failure: bool = True

    def __post_init__(self) -> None:
        if self.max_step_count <= 0:
            raise SchemaValidationError("AssemblyRunnerConfig.max_step_count must be positive")
        if self.max_retries_per_step < 0:
            raise SchemaValidationError("AssemblyRunnerConfig.max_retries_per_step must be non-negative")
        if self.retry_timeout_s <= 0.0 or self.abort_timeout_s <= 0.0:
            raise SchemaValidationError("AssemblyRunnerConfig retry/abort timeouts must be positive")


@dataclass
class AssemblyRunReport(SchemaBase):
    plan: AssemblyPlan
    success: bool
    final_state: ConstructionState
    step_results: list[AssemblyExecutionResult]
    completed_step_count: int
    attached_edge_count: int
    target_edge_count: int
    state_matches_target: bool
    retry_count: int
    abort_count: int
    aborted: bool
    executed_step_types: list[str]
    failure_reason: str | None = None
    failures: list[Violation] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.completed_step_count < 0:
            raise SchemaValidationError("AssemblyRunReport.completed_step_count must be non-negative")
        if self.attached_edge_count < 0 or self.target_edge_count < 0:
            raise SchemaValidationError("AssemblyRunReport edge counts must be non-negative")
        if self.attached_edge_count > self.target_edge_count:
            raise SchemaValidationError("AssemblyRunReport attached_edge_count cannot exceed target_edge_count")
        if self.retry_count < 0 or self.abort_count < 0:
            raise SchemaValidationError("AssemblyRunReport retry/abort counts must be non-negative")
        if self.success and self.failure_reason is not None:
            raise SchemaValidationError("AssemblyRunReport cannot pass with failure_reason")
        if self.success and not self.state_matches_target:
            raise SchemaValidationError("AssemblyRunReport cannot pass unless final state matches target")


class AssemblyRunner:
    """Run a deterministic AssemblyPlan through an AssemblyExecutorInterface."""

    def __init__(
        self,
        *,
        planner: GraphEditAssemblyPlanner | None = None,
        config: AssemblyRunnerConfig | None = None,
    ) -> None:
        self.planner = planner or GraphEditAssemblyPlanner()
        self.config = config or AssemblyRunnerConfig()

    def run(
        self,
        target_graph: MorphologyGraph,
        executor: AssemblyExecutorInterface,
        *,
        construction_state: ConstructionState | None = None,
    ) -> AssemblyRunReport:
        state = construction_state or initial_construction_state(target_graph)
        plan = self.planner.build_plan(target_graph, construction_state=state)
        if len(plan.steps) > self.config.max_step_count:
            raise SchemaValidationError("AssemblyPlan exceeds AssemblyRunnerConfig.max_step_count")

        step_results: list[AssemblyExecutionResult] = []
        executed_step_types: list[str] = []
        failures: list[Violation] = []
        failure_reason: str | None = None
        completed_planned_step_count = 0
        retry_count = 0
        abort_count = 0
        aborted = False

        for step in plan.steps:
            failed_result: AssemblyExecutionResult | None = None
            for attempt_index in range(self.config.max_retries_per_step + 1):
                state = replace(state, active_step_id=step.step_id)
                result = executor.execute_step(step, state)
                state = _state_after_execution_result(
                    target_graph,
                    previous_state=state,
                    step=step,
                    result=result,
                )
                step_results.append(result)
                executed_step_types.append(step.step_type)
                if result.success:
                    failed_result = None
                    completed_planned_step_count += 1
                    break

                failed_result = result
                failures.extend(result.violations)
                failure_reason = result.message or f"assembly step {step.step_id} failed"
                if attempt_index < self.config.max_retries_per_step:
                    retry_step = _retry_step_for(
                        step,
                        synthetic_step_id=_synthetic_step_id(plan, step_results),
                        attempt_index=attempt_index,
                        timeout_s=self.config.retry_timeout_s,
                    )
                    retry_result = _execute_synthetic_step(
                        target_graph,
                        executor,
                        state=state,
                        step=retry_step,
                    )
                    state = retry_result.updated_state or replace(state, active_step_id=None)
                    step_results.append(retry_result)
                    executed_step_types.append(retry_step.step_type)
                    retry_count += 1
                    if not retry_result.success:
                        failures.extend(retry_result.violations)
                        failure_reason = retry_result.message or f"retry step {retry_step.step_id} failed"
                        failed_result = retry_result
                        break

            if failed_result is not None:
                abort_step = _abort_step_for(
                    step,
                    synthetic_step_id=_synthetic_step_id(plan, step_results),
                    timeout_s=self.config.abort_timeout_s,
                )
                abort_result = _execute_synthetic_step(
                    target_graph,
                    executor,
                    state=state,
                    step=abort_step,
                )
                state = abort_result.updated_state or replace(state, active_step_id=None)
                step_results.append(abort_result)
                executed_step_types.append(abort_step.step_type)
                abort_count += 1
                aborted = True
                if not abort_result.success:
                    failures.extend(abort_result.violations)
                    failure_reason = abort_result.message or f"abort step {abort_step.step_id} failed"
                if self.config.stop_on_first_failure:
                    break

        state = replace(state, active_step_id=None, failures=[*state.failures, *failures])
        metrics = assembly_state_metrics(state, target_graph)
        success = (
            completed_planned_step_count == len(plan.steps)
            and not aborted
            and metrics["state_matches_target"] == 1.0
        )
        return AssemblyRunReport(
            plan=plan,
            success=success,
            final_state=state,
            step_results=step_results,
            completed_step_count=completed_planned_step_count,
            attached_edge_count=int(metrics["attached_edge_count"]),
            target_edge_count=int(metrics["target_edge_count"]),
            state_matches_target=metrics["state_matches_target"] == 1.0,
            retry_count=retry_count,
            abort_count=abort_count,
            aborted=aborted,
            executed_step_types=executed_step_types,
            failure_reason=None if success else failure_reason or _default_failure_reason(metrics),
            failures=failures,
            metrics={
                **metrics,
                "retry_count": float(retry_count),
                "abort_count": float(abort_count),
                "aborted": 1.0 if aborted else 0.0,
            },
        )


def assembly_state_metrics(state: ConstructionState, target_graph: MorphologyGraph) -> dict[str, float]:
    target_module_ids = {module.module_id for module in target_graph.modules}
    state_module_ids = {module.module_id for module in state.physical_graph.modules}
    target_edge_keys = {_edge_key(edge) for edge in target_graph.dock_edges}
    state_edge_keys = {_edge_key(edge) for edge in state.physical_graph.dock_edges}
    target_port_ids = {edge.src_port_id for edge in target_graph.dock_edges} | {
        edge.dst_port_id for edge in target_graph.dock_edges
    }
    occupied_state_port_ids = {
        port.port_global_id
        for port in state.physical_graph.ports
        if port.occupied and port.port_global_id in target_port_ids
    }
    module_match = target_module_ids == state_module_ids
    edge_match = target_edge_keys == state_edge_keys
    port_match = target_port_ids == occupied_state_port_ids
    return {
        "target_module_count": float(len(target_module_ids)),
        "assembled_module_count": float(len(state_module_ids)),
        "target_edge_count": float(len(target_edge_keys)),
        "attached_edge_count": float(len(state_edge_keys)),
        "target_occupied_port_count": float(len(target_port_ids)),
        "occupied_target_port_count": float(len(occupied_state_port_ids)),
        "module_set_matches_target": 1.0 if module_match else 0.0,
        "dock_edge_set_matches_target": 1.0 if edge_match else 0.0,
        "port_occupancy_matches_target": 1.0 if port_match else 0.0,
        "state_matches_target": 1.0 if module_match and edge_match and port_match else 0.0,
    }


def _state_after_execution_result(
    target_graph: MorphologyGraph,
    *,
    previous_state: ConstructionState,
    step: AssemblyStep,
    result: AssemblyExecutionResult,
) -> ConstructionState:
    if not result.success:
        return replace(previous_state, active_step_id=None)
    if result.updated_state is not None:
        return replace(result.updated_state, active_step_id=None)
    if step.step_type == "verify_attach":
        edge_id = _edge_id_from_verify_step(step)
        return replace(mark_edge_attached(previous_state, target_graph, edge_id), active_step_id=None)
    return replace(previous_state, active_step_id=None)


def _execute_synthetic_step(
    target_graph: MorphologyGraph,
    executor: AssemblyExecutorInterface,
    *,
    state: ConstructionState,
    step: AssemblyStep,
) -> AssemblyExecutionResult:
    state = replace(state, active_step_id=step.step_id)
    result = executor.execute_step(step, state)
    updated_state = _state_after_execution_result(
        target_graph,
        previous_state=state,
        step=step,
        result=result,
    )
    if result.updated_state is None and updated_state is not state:
        return replace(result, updated_state=updated_state)
    return result


def _retry_step_for(step: AssemblyStep, *, synthetic_step_id: int, attempt_index: int, timeout_s: float) -> AssemblyStep:
    return AssemblyStep(
        step_id=synthetic_step_id,
        step_type="retry",
        leader_module_id=step.leader_module_id,
        follower_module_id=step.follower_module_id,
        src_port_id=step.src_port_id,
        dst_port_id=step.dst_port_id,
        target_relative_pose=step.target_relative_pose,
        preconditions=[
            {
                "type": "previous_step_failed",
                "failed_step_id": step.step_id,
                "attempt_index": attempt_index,
            }
        ],
        success_conditions=[
            {
                "type": "retry_ready",
                "failed_step_id": step.step_id,
                "next_attempt_index": attempt_index + 1,
            }
        ],
        timeout_s=timeout_s,
    )


def _abort_step_for(step: AssemblyStep, *, synthetic_step_id: int, timeout_s: float) -> AssemblyStep:
    return AssemblyStep(
        step_id=synthetic_step_id,
        step_type="abort",
        leader_module_id=step.leader_module_id,
        follower_module_id=step.follower_module_id,
        src_port_id=step.src_port_id,
        dst_port_id=step.dst_port_id,
        target_relative_pose=step.target_relative_pose,
        preconditions=[{"type": "step_failed_after_retries", "failed_step_id": step.step_id}],
        success_conditions=[{"type": "assembly_aborted", "failed_step_id": step.step_id}],
        timeout_s=timeout_s,
    )


def _synthetic_step_id(plan: AssemblyPlan, step_results: list[AssemblyExecutionResult]) -> int:
    return len(plan.steps) + len(step_results)


def _edge_id_from_verify_step(step: AssemblyStep) -> int:
    for condition in step.success_conditions:
        if condition.get("type") == "edge_attached":
            edge_id = condition.get("edge_id")
            if isinstance(edge_id, int):
                return edge_id
    raise SchemaValidationError("verify_attach AssemblyStep is missing edge_attached success condition")


def _edge_key(edge: DockEdge) -> tuple[int, int, int, int]:
    return (edge.src_module_id, edge.src_port_id, edge.dst_module_id, edge.dst_port_id)


def _default_failure_reason(metrics: dict[str, float]) -> str:
    if metrics["state_matches_target"] != 1.0:
        return "final construction state does not match target graph"
    return "assembly execution failed"
