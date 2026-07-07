from __future__ import annotations

from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.graph import build_minimal_design_output
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.morphology import DesignActionType
from amsrr.schemas.task_spec import TaskSpec


def test_minimal_morphology_builder_grasp_carry_design_output(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_minimal_design_output(task, irg, physical_model)

    morphology = design.target_morphology
    assert design.task_id == task.task_id
    assert design.irg_id == irg.irg_id
    assert len(morphology.modules) == 3
    assert morphology.base_module_id == 0
    assert sum(module.is_base for module in morphology.modules) == 1
    assert len(morphology.dock_edges) == 2
    assert morphology.is_closed_loop is False
    assert {anchor.anchor_type for anchor in morphology.robot_anchors} == {"grasp", "support"}

    required_slot = next(
        node
        for node in irg.nodes
        if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature["required"]
    )
    anchors_for_required_slot = [
        anchor
        for anchor in morphology.robot_anchors
        if required_slot.feature["slot_id"] in anchor.associated_contact_slot_ids
    ]
    assert len(anchors_for_required_slot) == required_slot.feature["min_count_group"]
    assert all(anchor.capability["max_force_n"] == task.safety.max_contact_force_n for anchor in morphology.robot_anchors)
    assert design.slot_anchor_binding_prior
    assert design.design_actions[-1].action_type == DesignActionType.STOP


def test_minimal_morphology_design_output_roundtrip(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_minimal_design_output(task, irg, physical_model)

    assert type(design).from_json(design.to_json()).to_dict() == design.to_dict()
