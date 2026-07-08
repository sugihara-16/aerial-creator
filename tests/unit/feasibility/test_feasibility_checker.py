from __future__ import annotations

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.feasibility.violation_codes import (
    F_CLOSED_LOOP_REJECT_V1,
    F_COARSE_REACHABILITY,
    F_PORT_OCCUPANCY,
    F_REQUIRED_SLOT_COVERAGE,
    F_ROBOT_ANCHOR_CAPABILITY,
)
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.graph import build_minimal_design_output
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.irg import IRGNodeType
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
    assert result.proxy_scores["L_FEASIBLE"] == 1.0
    assert result.proxy_scores[f"L_{F_CLOSED_LOOP_REJECT_V1}"] == 0.0
    assert result.metadata["level"] == "design"


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
    assert result.proxy_scores["L_FEASIBLE"] == 0.0
    assert result.proxy_scores[f"L_{F_REQUIRED_SLOT_COVERAGE}"] == 1.0


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
    assert result.margins["closed_loop_rejected"] == 1.0
    assert result.proxy_scores[f"L_{F_CLOSED_LOOP_REJECT_V1}"] == 1.0


def test_p2_feasibility_checker_records_acceptance_margins_for_variant(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, _ = _minimal_design(grasp_carry_dict)
    design = build_grasp_carry_variant_design_output(
        task,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP,
    )

    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is True
    assert result.proxy_scores["L_FEASIBLE"] == 1.0
    assert result.proxy_scores["L_HARD_VIOLATION"] == 0.0
    assert result.margins["required_slot_count"] == 1.0
    assert result.margins["required_slot_anchor_required_count"] == 2.0
    assert result.margins["required_slot_anchor_coverage_ratio"] == 1.0
    assert result.margins["required_slot_anchor_capability_coverage_ratio"] == 1.0
    assert result.margins["anchor_capability_valid_ratio"] == 1.0
    assert result.margins["coarse_reachability_ratio"] == 1.0
    assert result.margins["port_conflict_count"] == 0.0
    assert result.margins["payload_required_force_n"] > 0.0
    assert result.margins["payload_anchor_force_n"] >= result.margins["payload_required_force_n"]
    assert result.margins["available_total_vertical_force_n"] > result.margins["required_total_vertical_force_n"]


def test_p2_feasibility_checker_uses_capability_requirement_force_label(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, design = _minimal_design(grasp_carry_dict)
    for anchor in design.target_morphology.robot_anchors:
        if anchor.anchor_type == "grasp":
            anchor.capability["max_force_n"] = 0.01

    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is False
    assert F_ROBOT_ANCHOR_CAPABILITY in {violation.code for violation in result.hard_violations}
    assert result.proxy_scores[f"L_{F_ROBOT_ANCHOR_CAPABILITY}"] == 1.0
    assert result.margins["required_slot_anchor_capability_coverage_ratio"] == 0.0
    assert result.margins["anchor_capability_valid_ratio"] == 0.0


def test_p2_feasibility_checker_records_port_conflict_margins(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, design = _minimal_design(grasp_carry_dict)
    morphology = design.target_morphology
    morphology.dock_edges[1].src_port_id = morphology.dock_edges[0].src_port_id

    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is False
    assert F_PORT_OCCUPANCY in {violation.code for violation in result.hard_violations}
    assert result.proxy_scores[f"L_{F_PORT_OCCUPANCY}"] == 1.0
    assert result.margins["port_duplicate_use_conflict_count"] > 0.0
    assert result.margins["port_conflict_count"] > 0.0


def test_p2_feasibility_checker_records_reachability_margins(grasp_carry_dict: dict) -> None:
    task, irg, physical_model, design = _minimal_design(grasp_carry_dict)
    required_slot = next(
        node
        for node in irg.nodes
        if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature["required"]
    )
    required_slot.feature["allowed_region_ids"] = []

    result = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )

    assert result.feasible is False
    assert F_COARSE_REACHABILITY in {violation.code for violation in result.hard_violations}
    assert result.proxy_scores[f"L_{F_COARSE_REACHABILITY}"] == 1.0
    assert result.margins["coarse_reachability_ratio"] == 0.0
    assert result.margins["coarse_reachability_required_slot_count"] == 1.0
