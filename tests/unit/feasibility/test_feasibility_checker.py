from __future__ import annotations

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.feasibility.violation_codes import F_CLOSED_LOOP_REJECT_V1, F_REQUIRED_SLOT_COVERAGE
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.graph import build_minimal_design_output
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec


def _minimal_design(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_minimal_design_output(task, irg, physical_model)
    return task, irg, physical_model, design


def test_feasibility_checker_accepts_minimal_design(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, design = _minimal_design(grasp_carry_dict)
    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is True
    assert result.hard_violations == []
    assert result.margins["required_slot_coverage_ratio"] == 1.0
    assert result.margins["thrust_margin_ratio"] >= task.safety.min_thrust_margin_ratio
    assert result.checker_version


def test_feasibility_checker_rejects_missing_required_slot_coverage(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, design = _minimal_design(grasp_carry_dict)
    morphology = design.target_morphology
    design.target_morphology = MorphologyGraph(
        graph_id=morphology.graph_id,
        modules=morphology.modules,
        ports=morphology.ports,
        dock_edges=morphology.dock_edges,
        robot_anchors=[],
        control_groups=morphology.control_groups,
        base_module_id=morphology.base_module_id,
        is_closed_loop=morphology.is_closed_loop,
    )

    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is False
    assert F_REQUIRED_SLOT_COVERAGE in {violation.code for violation in result.hard_violations}


def test_feasibility_checker_rejects_closed_loop_v1(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, design = _minimal_design(grasp_carry_dict)
    morphology = design.target_morphology
    design.target_morphology = MorphologyGraph(
        graph_id=morphology.graph_id,
        modules=morphology.modules,
        ports=morphology.ports,
        dock_edges=morphology.dock_edges,
        robot_anchors=morphology.robot_anchors,
        control_groups=morphology.control_groups,
        base_module_id=morphology.base_module_id,
        is_closed_loop=True,
    )

    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is False
    assert F_CLOSED_LOOP_REJECT_V1 in {violation.code for violation in result.hard_violations}
