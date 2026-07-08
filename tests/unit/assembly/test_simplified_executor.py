from __future__ import annotations

from amsrr.assembly import (
    AssemblyRunner,
    GraphEditAssemblyPlanner,
    SimplifiedAssemblyExecutor,
    SimplifiedAssemblyExecutorConfig,
    initial_construction_state,
)
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.task_spec import TaskSpec


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


def test_simplified_executor_runs_full_assembly_and_returns_updated_state(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    executor = SimplifiedAssemblyExecutor(target_graph=target_graph)

    report = AssemblyRunner().run(target_graph, executor)

    assert report.success is True
    assert report.state_matches_target is True
    assert executor.executed_step_ids == [step.step_id for step in report.plan.steps]
    assert all(result.metrics["success"] == 1.0 for result in report.step_results)
    assert report.final_state.docking_attempts == {
        str(edge.edge_id): 1 for edge in target_graph.dock_edges
    }


def test_simplified_executor_can_inject_step_type_failure(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    executor = SimplifiedAssemblyExecutor(
        target_graph=target_graph,
        config=SimplifiedAssemblyExecutorConfig(
            failure_mode="fail_matching_steps",
            fail_step_types=("align_ports",),
            failure_code="E_ASSEMBLY_TIMEOUT",
            failure_message="alignment timeout",
        ),
    )

    report = AssemblyRunner().run(target_graph, executor)

    assert report.success is False
    assert report.failure_reason == "alignment timeout"
    assert report.step_results[-1].violations[0].code == "E_ASSEMBLY_TIMEOUT"
    assert report.step_results[-1].violations[0].node_or_edge_ref == "assembly_step:1:align_ports"
    assert executor.executed_step_ids == [0, 1]


def test_simplified_executor_success_without_target_graph_uses_runner_state_transition(grasp_carry_dict: dict) -> None:
    target_graph = _target_graph(grasp_carry_dict)
    state = initial_construction_state(target_graph)
    plan = GraphEditAssemblyPlanner().build_plan(target_graph, construction_state=state)
    verify_step = plan.steps[3]
    executor = SimplifiedAssemblyExecutor()

    result = executor.execute_step(verify_step, state)

    assert result.success is True
    assert result.updated_state is None
