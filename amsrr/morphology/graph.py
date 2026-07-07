from __future__ import annotations

from dataclasses import dataclass

from amsrr.robot_model.physical_model_builder import build_module_capability_token
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import (
    ControlGroup,
    DesignAction,
    DesignActionType,
    DesignOutput,
    DockEdge,
    ModuleNode,
    MorphologyGraph,
    PortNode,
    RobotAnchor,
    SlotAnchorBindingPrior,
)
from amsrr.schemas.physical_model import DockPortSpec, ModuleCapabilityToken, PhysicalModel
from amsrr.schemas.task_spec import TaskSpec


IDENTITY_POSE: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
PORT_TYPE_ORDER = ("pitch_dock", "yaw_dock", "generic_dock")
CONTACT_MODE_TO_ANCHOR_TYPE = {
    ContactMode.GRASP: "grasp",
    ContactMode.SUPPORT: "support",
    ContactMode.PUSH: "push",
    ContactMode.LATCH: "latch",
    ContactMode.PERCH: "perch",
    ContactMode.TOOL: "tool",
    ContactMode.BODY_CONTACT: "body_contact",
}


@dataclass(frozen=True)
class _SlotRequirement:
    slot_id: int
    contact_mode: ContactMode
    required: bool
    min_count: int
    max_count: int
    target_entity_id: str
    required_anchor_capability: dict


class MinimalMorphologyBuilder:
    """Deterministic P0 morphology seed builder.

    The builder creates a connected tree and enough RobotAnchors to cover IRG
    ContactSlots. It is scaffolding for design-policy work, not an optimizer.
    """

    def build_design_output(
        self,
        task_spec: TaskSpec,
        irg: InteractionRequirementGraph,
        physical_model: PhysicalModel,
    ) -> DesignOutput:
        capability = build_module_capability_token(physical_model, module_type=physical_model.model_id)
        slot_requirements = self._slot_requirements(irg)
        anchor_plan = self._anchor_plan(slot_requirements)
        module_count = max(task_spec.robot_constraints.min_modules, len(anchor_plan), 1)
        if module_count > task_spec.robot_constraints.max_modules:
            raise SchemaValidationError(
                f"minimal morphology requires {module_count} modules, exceeds max_modules={task_spec.robot_constraints.max_modules}"
            )

        modules = self._build_modules(module_count, capability)
        ports, dock_edges = self._build_ports_and_edges(module_count, physical_model.dock_ports)
        anchors = self._build_anchors(anchor_plan, physical_model, task_spec)
        control_groups = [
            ControlGroup(group_id="all_modules", module_ids=[module.module_id for module in modules], role="whole_body")
        ]
        morphology = MorphologyGraph(
            graph_id=f"morphology:{task_spec.task_id}:{irg.stable_hash()[:12]}",
            modules=modules,
            ports=ports,
            dock_edges=dock_edges,
            robot_anchors=anchors,
            control_groups=control_groups,
            base_module_id=0,
            is_closed_loop=False,
        )
        actions = self._design_actions(modules, dock_edges, anchors)
        binding_priors = [
            SlotAnchorBindingPrior(
                slot_id=slot_id,
                anchor_id=anchor.anchor_id,
                score=1.0 if required else 0.5,
                reason_code="required_slot_coverage" if required else "optional_slot_prior",
            )
            for anchor, slot_id, required in zip(anchors, [item[0].slot_id for item in anchor_plan], [item[0].required for item in anchor_plan])
        ]
        return DesignOutput(
            task_id=task_spec.task_id,
            irg_id=irg.irg_id,
            target_morphology=morphology,
            module_roles={module.module_id: module.role_id for module in modules},
            slot_anchor_binding_prior=binding_priors,
            design_actions=actions,
            design_scores={"minimal_seed": 1.0},
        )

    @staticmethod
    def _slot_requirements(irg: InteractionRequirementGraph) -> list[_SlotRequirement]:
        requirements: list[_SlotRequirement] = []
        for node in sorted(irg.nodes, key=lambda item: item.node_id):
            if node.node_type != IRGNodeType.CONTACT_SLOT:
                continue
            requirements.append(_slot_requirement_from_node(node))
        return requirements

    @staticmethod
    def _anchor_plan(slot_requirements: list[_SlotRequirement]) -> list[tuple[_SlotRequirement, int]]:
        plan: list[tuple[_SlotRequirement, int]] = []
        for slot in slot_requirements:
            count = slot.min_count if slot.required else min(1, slot.max_count)
            for idx in range(count):
                plan.append((slot, idx))
        return plan

    @staticmethod
    def _build_modules(module_count: int, capability: ModuleCapabilityToken) -> list[ModuleNode]:
        return [
            ModuleNode(
                module_id=idx,
                module_type=capability.module_type,
                pose_in_design_frame=(float(idx) * 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base" if idx == 0 else "anchor_carrier",
                is_base=idx == 0,
                capability_token=capability,
            )
            for idx in range(module_count)
        ]

    @staticmethod
    def _build_ports_and_edges(module_count: int, dock_ports: list[DockPortSpec]) -> tuple[list[PortNode], list[DockEdge]]:
        if module_count > 1 and len(dock_ports) < 2:
            raise SchemaValidationError("minimal morphology requires at least two dock ports for multi-module graph")
        ports: list[PortNode] = []
        used_port_ids: set[int] = set()
        for module_id in range(module_count):
            for local_idx, dock_port in enumerate(dock_ports):
                port_global_id = module_id * len(dock_ports) + local_idx
                ports.append(
                    PortNode(
                        port_global_id=port_global_id,
                        module_id=module_id,
                        port_local_id=dock_port.port_id,
                        local_pose=dock_port.local_pose,
                        port_type=dock_port.port_type,
                        occupied=False,
                        compatible_port_type_mask=_compatible_mask(dock_port.compatible_port_types),
                    )
                )

        dock_edges: list[DockEdge] = []
        ports_by_module: dict[int, list[PortNode]] = {}
        for port in ports:
            ports_by_module.setdefault(port.module_id, []).append(port)
        port_by_id = {port.port_global_id: port for port in ports}
        for module_id in range(1, module_count):
            src, dst = _first_compatible_free_pair(
                ports_by_module[module_id - 1],
                ports_by_module[module_id],
                used_port_ids,
            )
            used_port_ids.update({src.port_global_id, dst.port_global_id})
            dock_edges.append(
                DockEdge(
                    edge_id=len(dock_edges),
                    src_module_id=src.module_id,
                    src_port_id=src.port_global_id,
                    dst_module_id=dst.module_id,
                    dst_port_id=dst.port_global_id,
                    relative_pose_src_to_dst=IDENTITY_POSE,
                    edge_role="structural",
                    estimated_stiffness=[1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0],
                    latch_state="planned",
                )
            )

        if used_port_ids:
            ports = [
                PortNode(
                    port_global_id=port.port_global_id,
                    module_id=port.module_id,
                    port_local_id=port.port_local_id,
                    local_pose=port.local_pose,
                    port_type=port.port_type,
                    occupied=port.port_global_id in used_port_ids,
                    compatible_port_type_mask=port.compatible_port_type_mask,
                )
                for port in ports
            ]
            port_by_id = {port.port_global_id: port for port in ports}
            for edge in dock_edges:
                if not port_by_id[edge.src_port_id].occupied or not port_by_id[edge.dst_port_id].occupied:
                    raise SchemaValidationError("internal dock port occupancy construction failed")
        return ports, dock_edges

    @staticmethod
    def _build_anchors(
        anchor_plan: list[tuple[_SlotRequirement, int]],
        physical_model: PhysicalModel,
        task_spec: TaskSpec,
    ) -> list[RobotAnchor]:
        link_id = _default_anchor_link(physical_model)
        anchors: list[RobotAnchor] = []
        for idx, (slot, local_idx) in enumerate(anchor_plan):
            module_id = min(idx, max(len(anchor_plan) - 1, 0))
            anchors.append(
                RobotAnchor(
                    anchor_id=idx,
                    module_id=module_id,
                    link_id=link_id,
                    local_pose=(0.0, 0.05 * float(local_idx), 0.0, 0.0, 0.0, 0.0, 1.0),
                    anchor_type=CONTACT_MODE_TO_ANCHOR_TYPE[slot.contact_mode],  # type: ignore[arg-type]
                    capability={
                        **slot.required_anchor_capability,
                        "max_force_n": task_spec.safety.max_contact_force_n,
                        "max_torque_nm": task_spec.safety.max_contact_torque_nm,
                        "target_entity_id": slot.target_entity_id,
                        "contact_mode": slot.contact_mode.value,
                    },
                    associated_contact_slot_ids=[slot.slot_id],
                )
            )
        return anchors

    @staticmethod
    def _design_actions(
        modules: list[ModuleNode],
        dock_edges: list[DockEdge],
        anchors: list[RobotAnchor],
    ) -> list[DesignAction]:
        actions: list[DesignAction] = [DesignAction(DesignActionType.SET_BASE_MODULE, {"module_id": 0})]
        for module in modules:
            actions.append(DesignAction(DesignActionType.ADD_MODULE, {"module_id": module.module_id, "module_type": module.module_type}))
            actions.append(DesignAction(DesignActionType.ASSIGN_ROLE, {"module_id": module.module_id, "role_id": module.role_id}))
        for edge in dock_edges:
            actions.append(
                DesignAction(
                    DesignActionType.CONNECT_PORT,
                    {
                        "edge_id": edge.edge_id,
                        "src_port_id": edge.src_port_id,
                        "dst_port_id": edge.dst_port_id,
                    },
                )
            )
        for anchor in anchors:
            actions.append(DesignAction(DesignActionType.CREATE_ANCHOR, {"anchor_id": anchor.anchor_id, "module_id": anchor.module_id}))
            for slot_id in anchor.associated_contact_slot_ids:
                actions.append(DesignAction(DesignActionType.BIND_ANCHOR_TO_SLOT, {"anchor_id": anchor.anchor_id, "slot_id": slot_id}))
        actions.append(DesignAction(DesignActionType.SET_CONTROL_GROUP, {"group_id": "all_modules"}))
        actions.append(DesignAction(DesignActionType.STOP, {}))
        return actions


def build_minimal_design_output(
    task_spec: TaskSpec,
    irg: InteractionRequirementGraph,
    physical_model: PhysicalModel,
) -> DesignOutput:
    return MinimalMorphologyBuilder().build_design_output(task_spec, irg, physical_model)


def _slot_requirement_from_node(node: IRGNode) -> _SlotRequirement:
    raw_mode = node.feature.get("contact_mode")
    try:
        mode = ContactMode(raw_mode)
    except ValueError as exc:
        raise SchemaValidationError(f"ContactSlot {node.node_id} has unsupported contact_mode {raw_mode!r}") from exc
    if mode not in CONTACT_MODE_TO_ANCHOR_TYPE:
        raise SchemaValidationError(f"ContactSlot {node.node_id} contact_mode {mode.value!r} is not anchor-compatible in P0")
    return _SlotRequirement(
        slot_id=int(node.feature.get("slot_id", node.node_id)),
        contact_mode=mode,
        required=bool(node.feature.get("required", True)),
        min_count=int(node.feature.get("min_count_group", 1)),
        max_count=int(node.feature.get("max_count_group", 1)),
        target_entity_id=str(node.feature.get("target_entity_id", "")),
        required_anchor_capability=dict(node.feature.get("required_anchor_capability", {}) or {}),
    )


def _compatible_mask(compatible_port_types: list[str]) -> list[int]:
    return [1 if port_type in compatible_port_types else 0 for port_type in PORT_TYPE_ORDER]


def _port_type_index(port_type: str) -> int | None:
    try:
        return PORT_TYPE_ORDER.index(port_type)
    except ValueError:
        return None


def _ports_compatible(src: PortNode, dst: PortNode) -> bool:
    dst_idx = _port_type_index(dst.port_type)
    src_idx = _port_type_index(src.port_type)
    if dst_idx is None or src_idx is None:
        return False
    return bool(src.compatible_port_type_mask[dst_idx]) and bool(dst.compatible_port_type_mask[src_idx])


def _first_compatible_free_pair(
    src_ports: list[PortNode],
    dst_ports: list[PortNode],
    used_port_ids: set[int],
) -> tuple[PortNode, PortNode]:
    for src in src_ports:
        if src.port_global_id in used_port_ids:
            continue
        for dst in dst_ports:
            if dst.port_global_id in used_port_ids:
                continue
            if _ports_compatible(src, dst):
                return src, dst
    raise SchemaValidationError("No compatible free dock port pair available")


def _default_anchor_link(physical_model: PhysicalModel) -> str | None:
    if physical_model.dock_ports:
        return physical_model.dock_ports[0].parent_link
    if physical_model.links:
        return physical_model.links[0].link_id
    return None
