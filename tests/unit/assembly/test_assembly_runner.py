from __future__ import annotations

from amsrr.assembly import (
    AssemblyExecutionResult,
    AssemblyRunner,
    AssemblyRunnerConfig,
    assembly_state_metrics,
    initial_construction_state,
)
from amsrr.assembly.construction_state import AssemblyStep, ConstructionState
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.feasibility import Violation, ViolationSeverity
from amsrr.schemas.task_spec import TaskSpec


class _AlwaysSuccessExecutor:
    def execute_step(self, step: AssemblyStep, state: ConstructionState) -> AssemblyExecutionResult:
        return AssemblyExecutionResult(step_id=step.step_id, success=True)


class _FailingDockExecutor:
    def execute_step(self, step: AssemblyStep, state: ConstructionState) -> AssemblyExecutionResult:
        if step.step_type == "dock":
            return AssemblyExecutionResult(
                step_id=step.step_id,
                success=False,
                violations=[
                    Violation(
                        code="E_DOCK_VERIFY_FAIL",
                        severity=ViolationSeverity.HARD,
                        message="injected dock failure",
                    )
                ],
                message="injected dock failure",
            )
        return AssemblyExecutionResult(step_id=step.step_id, success=True)


def _target_graph(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_grasp_carry_variant_design_output(
        task,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP,
    )
    return design.target_morphology


def test_assembly_runner_completes_plan_and_updates_construction_state(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    report = AssemblyRunner().run(target_graph, _AlwaysSuccessExecutor())

    assert report.success is True
    assert report.failure_reason is None
    assert len(report.step_results) == len(report.plan.steps)
    assert report.completed_step_count == len(report.plan.steps)
    assert report.attached_edge_count == len(target_graph.dock_edges)
    assert report.state_matches_target is True
    assert report.final_state.active_step_id is None
    assert report.final_state.unattached_modules == []
    assert assembly_state_metrics(report.final_state, target_graph)["state_matches_target"] == 1.0
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()


def test_assembly_runner_stops_on_failed_step_without_completing_graph(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    report = AssemblyRunner().run(target_graph, _FailingDockExecutor())

    assert report.success is False
    assert report.failure_reason == "injected dock failure"
    assert len(report.step_results) == 6
    assert report.step_results[-2].success is False
    assert report.failures[0].code == "E_DOCK_VERIFY_FAIL"
    assert report.retry_count == 1
    assert report.abort_count == 1
    assert report.aborted is True
    assert report.executed_step_types == ["move_to_staging", "align_ports", "dock", "retry", "dock", "abort"]
    assert report.attached_edge_count == 0
    assert report.state_matches_target is False


def test_assembly_runner_resumes_from_partial_construction_state(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    initial_state = initial_construction_state(target_graph)
    first_edge_report = AssemblyRunner().run(
        target_graph,
        _AlwaysSuccessExecutor(),
        construction_state=initial_state,
    )
    partial_state = first_edge_report.final_state

    resumed_report = AssemblyRunner().run(
        target_graph,
        _AlwaysSuccessExecutor(),
        construction_state=partial_state,
    )

    assert resumed_report.success is True
    assert resumed_report.plan.steps == []
    assert resumed_report.completed_step_count == 0
    assert resumed_report.state_matches_target is True


def test_assembly_runner_can_disable_retry_for_single_failure_stop(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    report = AssemblyRunner(config=AssemblyRunnerConfig(max_retries_per_step=0)).run(
        target_graph,
        _FailingDockExecutor(),
    )

    assert report.success is False
    assert report.retry_count == 0
    assert report.abort_count == 1
    assert report.executed_step_types == ["move_to_staging", "align_ports", "dock", "abort"]
