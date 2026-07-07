from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.geometry import ContactRegion, GeometryDescriptor
from amsrr.schemas.irg import IRGEdge, IRGEdgeType, IRGNode, IRGNodeType, PhaseType
from amsrr.schemas.task_spec import ObjectSpec, SurfaceSpec, TaskSpec, TaskType


PHASE_LABEL_TO_TYPE: dict[str, PhaseType] = {
    "takeoff_or_stabilize": PhaseType.FREE_MOTION,
    "navigate": PhaseType.FREE_MOTION,
    "hold_or_land": PhaseType.FREE_MOTION,
    "approach_object": PhaseType.APPROACH,
    "establish_object_contacts": PhaseType.ESTABLISH_CONTACT,
    "apply_grasp_wrench": PhaseType.APPLY_WRENCH,
    "lift_object": PhaseType.TRANSPORT,
    "transport_object": PhaseType.TRANSPORT,
    "place_object": PhaseType.PLACE,
    "release_contacts": PhaseType.RELEASE_CONTACT,
    "approach_valve": PhaseType.APPROACH,
    "establish_valve_contact": PhaseType.ESTABLISH_CONTACT,
    "apply_tangential_wrench": PhaseType.APPLY_WRENCH,
    "rotate_valve": PhaseType.APPLY_WRENCH,
    "release_contact": PhaseType.RELEASE_CONTACT,
    "navigate_to_perch_region": PhaseType.APPROACH,
    "establish_perch_contact": PhaseType.ESTABLISH_CONTACT,
    "hold_perch_wrench": PhaseType.MAINTAIN_CONTACT,
    "optional_manipulation": PhaseType.APPLY_WRENCH,
    "release_perch": PhaseType.RELEASE_CONTACT,
    "approach_support_region": PhaseType.APPROACH,
    "establish_support_contact": PhaseType.ESTABLISH_CONTACT,
    "maintain_support": PhaseType.MAINTAIN_CONTACT,
    "shift_centroidal_state": PhaseType.SHIFT_SUPPORT,
    "reposition_free_anchor": PhaseType.FREE_MOTION,
    "reanchor_support": PhaseType.ESTABLISH_CONTACT,
    "release_or_continue": PhaseType.RELEASE_CONTACT,
}


def phase_type_for_label(phase_label: str) -> PhaseType:
    try:
        return PHASE_LABEL_TO_TYPE[phase_label]
    except KeyError as exc:
        raise SchemaValidationError(f"No phase_type mapping for phase_label {phase_label!r}") from exc


@dataclass(frozen=True)
class SceneEntity:
    entity_id: str
    entity_type: str
    geometry_id: str
    contact_allowed: bool
    allowed_contact_modes: list[ContactMode]
    object_spec: ObjectSpec | None = None
    surface_spec: SurfaceSpec | None = None


@dataclass
class SceneGraph:
    entities: list[SceneEntity]
    geometry_descriptors: dict[str, GeometryDescriptor]
    entity_edges: list[dict[str, Any]] = field(default_factory=list)

    def entity(self, entity_id: str) -> SceneEntity:
        for item in self.entities:
            if item.entity_id == entity_id:
                return item
        raise SchemaValidationError(f"Scene entity {entity_id!r} not found")

    def descriptor_for_entity(self, entity: SceneEntity) -> GeometryDescriptor:
        try:
            return self.geometry_descriptors[entity.geometry_id]
        except KeyError as exc:
            raise SchemaValidationError(
                f"GeometryDescriptor for geometry_id {entity.geometry_id!r} is missing"
            ) from exc

    def contact_regions_for_entity(self, entity_id: str) -> list[ContactRegion]:
        entity = self.entity(entity_id)
        descriptor = self.descriptor_for_entity(entity)
        return descriptor.contact_region_graph.nodes

    def first_contactable_object(self) -> SceneEntity:
        for entity in self.entities:
            if entity.entity_type == "object" and entity.contact_allowed:
                return entity
        raise SchemaValidationError("No contactable object in scene")

    def first_support_surface(self) -> SceneEntity:
        for entity in self.entities:
            if entity.entity_type in {"support_surface", "floor", "wall"} and entity.contact_allowed:
                return entity
        raise SchemaValidationError("No contactable support surface in scene")


@dataclass
class IRGBuilderContext:
    task_spec: TaskSpec
    scene_graph: SceneGraph
    nodes: list[IRGNode] = field(default_factory=list)
    edges: list[IRGEdge] = field(default_factory=list)
    _next_slot_id: int = 0

    def add_node(
        self,
        node_type: IRGNodeType,
        ref_id: str | None,
        feature: dict[str, Any],
        *,
        priority: float = 1.0,
        is_hard: bool = True,
        active_phase_id: int | None = None,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(IRGNode(node_id, node_type, ref_id, priority, is_hard, active_phase_id, feature))
        return node_id

    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        edge_type: IRGEdgeType,
        *,
        priority: float = 1.0,
        condition: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.edges.append(IRGEdge(src_id, dst_id, edge_type, priority, condition, params or {}))

    def add_task_node(self) -> int:
        time_limit = max(goal.time_limit_s for goal in self.task_spec.goals)
        return self.add_node(
            IRGNodeType.TASK,
            self.task_spec.task_id,
            {
                "task_id": self.task_spec.task_id,
                "task_type": self.task_spec.task_type.value,
                "success_conditions": [],
                "failure_conditions": [],
                "time_limit_s": time_limit,
            },
        )

    def add_phase(self, phase_label: str, phase_index: int) -> int:
        phase_type = phase_type_for_label(phase_label)
        return self.add_node(
            IRGNodeType.PHASE,
            phase_label,
            {
                "phase_type": phase_type.value,
                "phase_label": phase_label,
                "phase_index": phase_index,
                "entry_condition": None,
                "exit_condition": None,
                "failure_condition": None,
                "nominal_duration_s": None,
                "max_duration_s": None,
            },
        )

    def add_phase_sequence(self, task_node_id: int, phase_labels: list[str]) -> dict[str, int]:
        phase_ids: dict[str, int] = {}
        previous_id: int | None = None
        for idx, label in enumerate(phase_labels):
            phase_id = self.add_phase(label, idx)
            phase_ids[label] = phase_id
            self.add_edge(task_node_id, phase_id, IRGEdgeType.CONTAINS)
            if previous_id is not None:
                self.add_edge(previous_id, phase_id, IRGEdgeType.TEMPORAL_NEXT)
            previous_id = phase_id
        return phase_ids

    def add_contact_region_nodes(self, entity_id: str) -> dict[str, int]:
        region_nodes: dict[str, int] = {}
        for region in self.scene_graph.contact_regions_for_entity(entity_id):
            compiled_region_id = region.region_id if region.entity_id == entity_id else f"{entity_id}:{region.region_id}"
            node_id = self.add_node(
                IRGNodeType.CONTACT_REGION,
                compiled_region_id,
                {
                    "region_id": compiled_region_id,
                    "target_entity_id": entity_id,
                    "region_type": region.region_type,
                    "allowed_contact_modes": [mode.value for mode in region.allowed_contact_modes],
                    "area_m2": region.area_m2,
                    "normal_summary": list(region.normal_summary_object),
                    "curvature_summary": region.curvature_summary,
                    "friction": region.friction,
                },
            )
            region_nodes[compiled_region_id] = node_id
        return region_nodes

    def next_slot_id(self) -> int:
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        return slot_id

    def add_contact_slot(
        self,
        *,
        target_entity_type: str,
        target_entity_id: str,
        allowed_region_ids: list[str],
        contact_mode: ContactMode,
        required: bool,
        min_count_group: int,
        max_count_group: int,
        required_anchor_capability: dict[str, Any] | None = None,
    ) -> int:
        slot_id = self.next_slot_id()
        return self.add_node(
            IRGNodeType.CONTACT_SLOT,
            f"slot_{slot_id}",
            {
                "slot_id": slot_id,
                "target_entity_type": target_entity_type,
                "target_entity_id": target_entity_id,
                "allowed_region_ids": allowed_region_ids,
                "contact_mode": contact_mode.value,
                "required": required,
                "min_count_group": min_count_group,
                "max_count_group": max_count_group,
                "normal_constraint": None,
                "approach_direction_constraint": None,
                "separation_constraint": None,
                "required_anchor_capability": required_anchor_capability or {},
            },
        )

    def add_wrench_requirement(
        self,
        requirement_id: str,
        *,
        applies_to: str,
        frame: str,
        required_effect: str,
        wrench_lower: list[float] | None = None,
        wrench_upper: list[float] | None = None,
        target_wrench: list[float] | None = None,
        slack_weight: float = 1.0,
        hard_or_soft: str = "hard",
    ) -> int:
        return self.add_node(
            IRGNodeType.WRENCH_REQUIREMENT,
            requirement_id,
            {
                "requirement_id": requirement_id,
                "applies_to": applies_to,
                "frame": frame,
                "required_effect": required_effect,
                "wrench_lower": wrench_lower,
                "wrench_upper": wrench_upper,
                "target_wrench": target_wrench,
                "slack_weight": slack_weight,
                "hard_or_soft": hard_or_soft,
            },
        )

    def add_state_target(
        self,
        target_id: str,
        *,
        target_type: str,
        target_entity_id: str | None = None,
        pose_target_world: Pose7D | None = None,
        twist_target_world: list[float] | None = None,
        q_target: list[float] | None = None,
        tolerance: dict[str, Any] | None = None,
    ) -> int:
        return self.add_node(
            IRGNodeType.STATE_TARGET,
            target_id,
            {
                "target_type": target_type,
                "target_entity_id": target_entity_id,
                "pose_target_world": list(pose_target_world) if pose_target_world is not None else None,
                "twist_target_world": twist_target_world,
                "q_target": q_target,
                "tolerance": tolerance or {},
            },
        )

    def add_constraint(
        self,
        constraint_id: str,
        *,
        constraint_type: str,
        parameters: dict[str, Any] | None = None,
        violation_code: str,
    ) -> int:
        return self.add_node(
            IRGNodeType.CONSTRAINT,
            constraint_id,
            {
                "constraint_type": constraint_type,
                "parameters": parameters or {},
                "violation_code": violation_code,
            },
        )

    def add_capability_requirement(
        self,
        capability_id: str,
        *,
        capability_type: str,
        min_force_n: float | None = None,
        min_torque_nm: float | None = None,
        pose_accuracy_m: float | None = None,
        pose_accuracy_rad: float | None = None,
        stiffness_requirement: float | None = None,
    ) -> int:
        return self.add_node(
            IRGNodeType.CAPABILITY_REQUIREMENT,
            capability_id,
            {
                "capability_type": capability_type,
                "min_force_n": min_force_n,
                "min_torque_nm": min_torque_nm,
                "pose_accuracy_m": pose_accuracy_m,
                "pose_accuracy_rad": pose_accuracy_rad,
                "stiffness_requirement": stiffness_requirement,
            },
        )

    def require_regions(self, entity_id: str) -> tuple[dict[str, int], list[str]]:
        region_nodes = self.add_contact_region_nodes(entity_id)
        if not region_nodes:
            raise SchemaValidationError(f"No contact regions available for entity {entity_id!r}")
        return region_nodes, list(region_nodes.keys())


class InteractionTemplate(Protocol):
    task_type: TaskType

    def validate_required_fields(self, task_spec: TaskSpec, scene_graph: SceneGraph) -> None:
        ...

    def build(self, context: IRGBuilderContext, task_node_id: int) -> None:
        ...
