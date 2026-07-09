from __future__ import annotations

import pytest

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION, compose_pose
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GRASP_CARRY_VARIANT_ORDER,
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.task_spec import TaskSpec


def _inputs(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return task, irg, physical_model


def _signature(design) -> tuple:
    morphology = design.target_morphology
    return (
        len(morphology.modules),
        tuple((edge.src_module_id, edge.dst_module_id, edge.edge_role) for edge in morphology.dock_edges),
        tuple((anchor.module_id, anchor.anchor_type) for anchor in morphology.robot_anchors),
        tuple(sorted(group.group_id for group in morphology.control_groups)),
    )


def test_grasp_carry_variants_build_distinct_feasible_morphologies(grasp_carry_dict: dict) -> None:
    task, irg, physical_model = _inputs(grasp_carry_dict)
    designs = [
        build_grasp_carry_variant_design_output(task, irg, physical_model, variant=variant)
        for variant in GRASP_CARRY_VARIANT_ORDER
    ]

    signatures = {_signature(design) for design in designs}
    assert len(signatures) == len(GRASP_CARRY_VARIANT_ORDER)

    for variant, design in zip(GRASP_CARRY_VARIANT_ORDER, designs):
        result = FeasibilityChecker().check_design(
            design,
            task_spec=task,
            irg=irg,
            physical_model=physical_model,
        )
        assert result.feasible, [violation.code for violation in result.hard_violations]
        assert result.margins["required_slot_coverage_ratio"] == 1.0
        assert design.target_morphology.graph_id.endswith(variant.value)
        _assert_dock_edges_are_port_aligned(design.target_morphology)
        assert design.design_scores["p2_grasp_carry_variant_builder"] == 1.0
        assert design.design_actions[-1].params["variant"] == variant.value
        assert type(design).from_json(design.to_json()).to_dict() == design.to_dict()


def test_grasp_carry_variant_topology_shapes(grasp_carry_dict: dict) -> None:
    task, irg, physical_model = _inputs(grasp_carry_dict)
    by_variant = {
        variant: build_grasp_carry_variant_design_output(task, irg, physical_model, variant=variant)
        for variant in GRASP_CARRY_VARIANT_ORDER
    }

    chain = by_variant[GraspCarryMorphologyVariant.CHAIN_GRASP].target_morphology
    assert len(chain.modules) == 2
    assert [(edge.src_module_id, edge.dst_module_id) for edge in chain.dock_edges] == [(0, 1)]
    assert {anchor.anchor_type for anchor in chain.robot_anchors} == {"grasp"}

    symmetric = by_variant[GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP].target_morphology
    assert len(symmetric.modules) == 3
    assert {(edge.src_module_id, edge.dst_module_id) for edge in symmetric.dock_edges} == {(0, 1), (0, 2)}
    assert [anchor.module_id for anchor in symmetric.robot_anchors] == [1, 2]

    tri_anchor = by_variant[GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP].target_morphology
    assert len(tri_anchor.modules) == 3
    assert {anchor.anchor_type for anchor in tri_anchor.robot_anchors} == {"grasp", "support"}
    assert any(anchor.module_id == 0 and anchor.anchor_type == "support" for anchor in tri_anchor.robot_anchors)
    assert "support_group" in {group.group_id for group in tri_anchor.control_groups}

    central = by_variant[GraspCarryMorphologyVariant.CENTRAL_BASE_PLUS_TWO_GRASP_ARMS].target_morphology
    assert len(central.modules) == 5
    assert {(edge.src_module_id, edge.dst_module_id) for edge in central.dock_edges} == {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
    }
    assert [anchor.module_id for anchor in central.robot_anchors] == [3, 4]


def test_grasp_carry_variants_cover_required_slot_min_count(grasp_carry_dict: dict) -> None:
    task, irg, physical_model = _inputs(grasp_carry_dict)
    required_slot = next(
        node
        for node in irg.nodes
        if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature["required"]
    )

    for variant in GRASP_CARRY_VARIANT_ORDER:
        design = build_grasp_carry_variant_design_output(task, irg, physical_model, variant=variant)
        anchors_for_required_slot = [
            anchor
            for anchor in design.target_morphology.robot_anchors
            if required_slot.feature["slot_id"] in anchor.associated_contact_slot_ids
        ]
        assert len(anchors_for_required_slot) == required_slot.feature["min_count_group"]


def _assert_dock_edges_are_port_aligned(morphology) -> None:
    ports_by_id = {port.port_global_id: port for port in morphology.ports}
    modules_by_id = {module.module_id: module for module in morphology.modules}
    for edge in morphology.dock_edges:
        src_port = ports_by_id[edge.src_port_id]
        dst_port = ports_by_id[edge.dst_port_id]
        src_module = modules_by_id[edge.src_module_id]
        dst_module = modules_by_id[edge.dst_module_id]
        src_port_world = compose_pose(
            src_module.pose_in_design_frame,
            compose_pose(src_port.local_pose, FACE_TO_FACE_DOCK_RELATION),
        )
        dst_port_world = compose_pose(dst_module.pose_in_design_frame, dst_port.local_pose)
        assert dst_port_world == pytest.approx(src_port_world)
