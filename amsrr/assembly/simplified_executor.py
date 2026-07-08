from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.assembly.construction_state import AssemblyStep, ConstructionState, mark_edge_attached
from amsrr.assembly.executor_interface import AssemblyExecutionResult
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.feasibility import Violation, ViolationSeverity
from amsrr.schemas.morphology import MorphologyGraph


SimplifiedFailureMode = Literal["none", "fail_matching_steps"]


@dataclass(frozen=True)
class SimplifiedAssemblyExecutorConfig:
    failure_mode: SimplifiedFailureMode = "none"
    fail_step_ids: tuple[int, ...] = ()
    fail_step_types: tuple[str, ...] = ()
    fail_once_step_ids: tuple[int, ...] = ()
    fail_once_step_types: tuple[str, ...] = ()
    failure_code: str = "E_DOCK_VERIFY_FAIL"
    failure_message: str = "simplified assembly executor injected failure"
    default_step_duration_s: float = 0.1

    def __post_init__(self) -> None:
        if self.default_step_duration_s <= 0.0:
            raise SchemaValidationError("SimplifiedAssemblyExecutorConfig.default_step_duration_s must be positive")
        if not self.failure_code:
            raise SchemaValidationError("SimplifiedAssemblyExecutorConfig.failure_code must be non-empty")
        if not self.failure_message:
            raise SchemaValidationError("SimplifiedAssemblyExecutorConfig.failure_message must be non-empty")


class SimplifiedAssemblyExecutor:
    """Deterministic smoke executor for P3 assembly integration tests."""

    def __init__(
        self,
        *,
        target_graph: MorphologyGraph | None = None,
        config: SimplifiedAssemblyExecutorConfig | None = None,
    ) -> None:
        self.target_graph = target_graph
        self.config = config or SimplifiedAssemblyExecutorConfig()
        self.executed_step_ids: list[int] = []
        self._failed_once_keys: set[tuple[int, str]] = set()

    def execute_step(self, step: AssemblyStep, state: ConstructionState) -> AssemblyExecutionResult:
        self.executed_step_ids.append(step.step_id)
        if self._should_fail(step):
            violation = Violation(
                code=self.config.failure_code,
                severity=ViolationSeverity.HARD,
                message=self.config.failure_message,
                node_or_edge_ref=_step_ref(step),
            )
            return AssemblyExecutionResult(
                step_id=step.step_id,
                success=False,
                violations=[violation],
                metrics=_step_metrics(step, success=False, duration_s=self.config.default_step_duration_s),
                message=self.config.failure_message,
            )

        updated_state = None
        if step.step_type == "verify_attach" and self.target_graph is not None:
            updated_state = mark_edge_attached(state, self.target_graph, _edge_id_from_verify_step(step))
        return AssemblyExecutionResult(
            step_id=step.step_id,
            success=True,
            updated_state=updated_state,
            metrics=_step_metrics(step, success=True, duration_s=self.config.default_step_duration_s),
            message=None,
        )

    def _should_fail(self, step: AssemblyStep) -> bool:
        if self.config.failure_mode == "none":
            return False
        step_key = (step.step_id, step.step_type)
        if step.step_id in self.config.fail_once_step_ids or step.step_type in self.config.fail_once_step_types:
            if step_key not in self._failed_once_keys:
                self._failed_once_keys.add(step_key)
                return True
        return step.step_id in self.config.fail_step_ids or step.step_type in self.config.fail_step_types


def _step_metrics(step: AssemblyStep, *, success: bool, duration_s: float) -> dict[str, float]:
    return {
        "success": 1.0 if success else 0.0,
        "simulated_duration_s": duration_s,
        "step_id": float(step.step_id),
        "has_follower": 1.0 if step.follower_module_id is not None else 0.0,
    }


def _edge_id_from_verify_step(step: AssemblyStep) -> int:
    for condition in step.success_conditions:
        if condition.get("type") == "edge_attached":
            edge_id = condition.get("edge_id")
            if isinstance(edge_id, int):
                return edge_id
    raise SchemaValidationError("verify_attach AssemblyStep is missing edge_attached success condition")


def _step_ref(step: AssemblyStep) -> str:
    return f"assembly_step:{step.step_id}:{step.step_type}"
