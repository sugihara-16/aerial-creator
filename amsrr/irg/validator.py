from __future__ import annotations

from collections import defaultdict

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.task_spec import TaskSpec, TaskType


class IRGValidator:
    """Deterministic structural validator for v1 IRGs."""

    def validate(self, irg: InteractionRequirementGraph, task_spec: TaskSpec) -> None:
        nodes_by_id = {node.node_id: node for node in irg.nodes}
        nodes_by_type: dict[IRGNodeType, list[int]] = defaultdict(list)
        incoming: dict[int, list[IRGEdgeType]] = defaultdict(list)
        outgoing: dict[int, list[IRGEdgeType]] = defaultdict(list)
        for node in irg.nodes:
            nodes_by_type[node.node_type].append(node.node_id)
        for edge in irg.edges:
            incoming[edge.dst_id].append(edge.edge_type)
            outgoing[edge.src_id].append(edge.edge_type)

        if len(nodes_by_type.get(IRGNodeType.TASK, [])) != 1:
            raise SchemaValidationError("IRG requires exactly one TaskNode")
        phase_ids = nodes_by_type.get(IRGNodeType.PHASE, [])
        if not phase_ids:
            raise SchemaValidationError("IRG requires at least one PhaseNode")
        if len(phase_ids) > 1:
            temporal_edges = [edge for edge in irg.edges if edge.edge_type == IRGEdgeType.TEMPORAL_NEXT]
            if len(temporal_edges) < len(phase_ids) - 1:
                raise SchemaValidationError("IRG phase sequence is missing temporal_next edges")

        for slot_id in nodes_by_type.get(IRGNodeType.CONTACT_SLOT, []):
            if IRGEdgeType.ALLOWS not in incoming[slot_id]:
                raise SchemaValidationError(f"ContactSlot node {slot_id} has no ContactRegion allows edge")

        for wrench_id in nodes_by_type.get(IRGNodeType.WRENCH_REQUIREMENT, []):
            incident = incoming[wrench_id] + outgoing[wrench_id]
            if IRGEdgeType.REQUIRES not in incident and IRGEdgeType.APPLIES_TO not in incident:
                raise SchemaValidationError(f"WrenchRequirement node {wrench_id} has no applies_to/requires relation")

        for state_id in nodes_by_type.get(IRGNodeType.STATE_TARGET, []):
            incident = incoming[state_id] + outgoing[state_id]
            if IRGEdgeType.REQUIRES not in incident and IRGEdgeType.SUPPORTS not in incident:
                raise SchemaValidationError(f"StateTarget node {state_id} has no phase or requirement relation")

        for constraint_id in nodes_by_type.get(IRGNodeType.CONSTRAINT, []):
            if IRGEdgeType.CONSTRAINS not in outgoing[constraint_id]:
                raise SchemaValidationError(f"Constraint node {constraint_id} has no constrains target")

        if task_spec.task_type == TaskType.OBJECT_GRASP_CARRY:
            movable_targets = [obj for obj in task_spec.scene.objects if obj.movable]
            if not movable_targets or any(obj.mass_kg is None for obj in movable_targets):
                raise SchemaValidationError("object_grasp_carry requires movable target object mass")
        if task_spec.task_type == TaskType.VALVE_OPERATION:
            if not any(obj.kinematic_model is not None for obj in task_spec.scene.objects):
                raise SchemaValidationError("valve_operation requires object kinematic model / axis")
        if task_spec.task_type == TaskType.CONTACT_MEDIATED_LOCOMOTION:
            if not task_spec.scene.environment.support_surfaces:
                raise SchemaValidationError("contact_mediated_locomotion requires support surface")
        if task_spec.task_type == TaskType.PERCHING_MANIPULATION:
            allowed = {
                mode.value
                for surface in task_spec.scene.environment.support_surfaces
                for mode in surface.allowed_contact_modes
            }
            if not allowed.intersection({"perch", "latch", "support"}):
                raise SchemaValidationError("perching_manipulation requires perch/latch/support region")

        # Force schema-level node validation by touching nodes_by_id, making mypy/linters happy too.
        if set(nodes_by_id) != {node.node_id for node in irg.nodes}:
            raise SchemaValidationError("IRG node_id map is inconsistent")

