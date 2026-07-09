from __future__ import annotations

from dataclasses import dataclass

from amsrr.morphology.dock_geometry import modules_with_dock_aligned_poses, relative_pose_for_dock_ports
from amsrr.robot_model.physical_model_builder import build_module_capability_token
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError, StrEnum
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
from amsrr.schemas.task_spec import TaskSpec, TaskType


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


class GraspCarryMorphologyVariant(StrEnum):
    CHAIN_GRASP = "chain_grasp"
    SYMMETRIC_TWO_ANCHOR_GRASP = "symmetric_two_anchor_grasp"
    TRI_ANCHOR_SUPPORT_GRASP = "tri_anchor_support_grasp"
    CENTRAL_BASE_PLUS_TWO_GRASP_ARMS = "central_base_plus_two_grasp_arms"


GRASP_CARRY_VARIANT_ORDER: tuple[GraspCarryMorphologyVariant, ...] = (
    GraspCarryMorphologyVariant.CHAIN_GRASP,
    GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP,
    GraspCarryMorphologyVariant.CENTRAL_BASE_PLUS_TWO_GRASP_ARMS,
)


@dataclass(frozen=True)
class _SlotRequirement:
    slot_id: int
    contact_mode: ContactMode
    required: bool
    min_count: int
    max_count: int
    target_entity_id: str
    required_anchor_capability: dict


@dataclass(frozen=True)
class _AnchorPlanItem:
    slot: _SlotRequirement
    module_id: int
    local_pose: Pose7D
    role_label: str


@dataclass(frozen=True)
class _VariantLayout:
    module_roles: dict[int, str]
    module_poses: dict[int, Pose7D]
    edge_specs: list[tuple[int, int, str]]
    required_anchor_modules: list[int]
    optional_support_module: int | None = None


class GraspCarryMorphologyVariantBuilder:
    """P2 deterministic grasp/carry morphology variant builder.

    The variants are schema-compatible design-policy scaffolds. They create
    distinct connected-tree topologies and RobotAnchor placements for π_D
    bootstrapping, without claiming optimized morphology search.
    """

    def build_design_output(
        self,
        task_spec: TaskSpec,
        irg: InteractionRequirementGraph,
        physical_model: PhysicalModel,
        *,
        variant: GraspCarryMorphologyVariant | str,
    ) -> DesignOutput:
        if task_spec.task_type != TaskType.OBJECT_GRASP_CARRY:
            raise SchemaValidationError("GraspCarryMorphologyVariantBuilder only supports object_grasp_carry")
        selected_variant = GraspCarryMorphologyVariant(variant)
        slot_requirements = _slot_requirements(irg)
        required_items = _required_anchor_items(slot_requirements)
        if not required_items:
            raise SchemaValidationError("grasp/carry variant builder requires at least one required ContactSlot")

        layout = _layout_for_variant(
            selected_variant,
            task_spec=task_spec,
            required_anchor_count=len(required_items),
        )
        if len(layout.required_anchor_modules) < len(required_items):
            raise SchemaValidationError(
                f"{selected_variant.value} provides {len(layout.required_anchor_modules)} required anchor modules "
                f"for {len(required_items)} required anchors"
            )

        capability = build_module_capability_token(physical_model, module_type=physical_model.model_id)
        modules = _build_modules(layout, capability)
        ports, dock_edges = _build_ports_and_edges(layout, physical_model.dock_ports, modules)
        modules = modules_with_dock_aligned_poses(modules, dock_edges, base_module_id=0)
        anchor_plan = _anchor_plan(
            required_items,
            slot_requirements,
            layout=layout,
            variant=selected_variant,
        )
        anchors = _build_anchors(anchor_plan, physical_model, task_spec, selected_variant)
        control_groups = _build_control_groups(modules, selected_variant, layout)
        morphology = MorphologyGraph(
            graph_id=f"morphology:{task_spec.task_id}:{irg.stable_hash()[:12]}:{selected_variant.value}",
            modules=modules,
            ports=ports,
            dock_edges=dock_edges,
            robot_anchors=anchors,
            control_groups=control_groups,
            base_module_id=0,
            is_closed_loop=False,
        )
        actions = _design_actions(modules, dock_edges, anchors, control_groups, selected_variant)
        binding_priors = _binding_priors(anchors, anchor_plan)
        variant_index = float(GRASP_CARRY_VARIANT_ORDER.index(selected_variant))
        return DesignOutput(
            task_id=task_spec.task_id,
            irg_id=irg.irg_id,
            target_morphology=morphology,
            module_roles={module.module_id: module.role_id for module in modules},
            slot_anchor_binding_prior=binding_priors,
            design_actions=actions,
            design_scores={
                "p2_grasp_carry_variant_id": variant_index,
                "p2_grasp_carry_variant_builder": 1.0,
                "p2_module_count": float(len(modules)),
                "p2_anchor_count": float(len(anchors)),
            },
        )


def build_grasp_carry_variant_design_output(
    task_spec: TaskSpec,
    irg: InteractionRequirementGraph,
    physical_model: PhysicalModel,
    *,
    variant: GraspCarryMorphologyVariant | str,
) -> DesignOutput:
    return GraspCarryMorphologyVariantBuilder().build_design_output(
        task_spec,
        irg,
        physical_model,
        variant=variant,
    )


def _layout_for_variant(
    variant: GraspCarryMorphologyVariant,
    *,
    task_spec: TaskSpec,
    required_anchor_count: int,
) -> _VariantLayout:
    if variant == GraspCarryMorphologyVariant.CHAIN_GRASP:
        desired_count = max(required_anchor_count, task_spec.robot_constraints.min_modules, 2)
        _require_module_budget(desired_count, task_spec, variant)
        roles = {idx: ("base" if idx == 0 else "chain_grasp_link") for idx in range(desired_count)}
        poses = {idx: (0.28 * float(idx), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) for idx in range(desired_count)}
        edges = [(idx - 1, idx, "grasp_arm") for idx in range(1, desired_count)]
        anchors = _distributed_module_ids(desired_count, required_anchor_count)
        return _VariantLayout(roles, poses, edges, anchors)

    if variant == GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP:
        desired_count = max(required_anchor_count + 1, task_spec.robot_constraints.min_modules, 3)
        _require_module_budget(desired_count, task_spec, variant)
        roles = {0: "base"}
        poses = {0: IDENTITY_POSE}
        edges: list[tuple[int, int, str]] = []
        for module_id in range(1, desired_count):
            side = -1.0 if module_id % 2 else 1.0
            arm_index = (module_id + 1) // 2
            roles[module_id] = "left_grasp_arm" if side < 0.0 else "right_grasp_arm"
            poses[module_id] = (0.25 * float(arm_index), 0.22 * side, 0.0, 0.0, 0.0, 0.0, 1.0)
            edges.append((0, module_id, "grasp_arm"))
        anchors = list(range(1, 1 + required_anchor_count))
        return _VariantLayout(roles, poses, edges, anchors)

    if variant == GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP:
        desired_count = max(required_anchor_count + 1, task_spec.robot_constraints.min_modules, 3)
        _require_module_budget(desired_count, task_spec, variant)
        roles = {0: "base_support"}
        poses = {0: IDENTITY_POSE}
        edges = []
        for module_id in range(1, desired_count):
            side = -1.0 if module_id % 2 else 1.0
            arm_index = (module_id + 1) // 2
            roles[module_id] = "left_grasp_arm" if side < 0.0 else "right_grasp_arm"
            poses[module_id] = (0.24 * float(arm_index), 0.24 * side, 0.0, 0.0, 0.0, 0.0, 1.0)
            edges.append((0, module_id, "grasp_arm"))
        anchors = list(range(1, 1 + required_anchor_count))
        return _VariantLayout(roles, poses, edges, anchors, optional_support_module=0)

    if variant == GraspCarryMorphologyVariant.CENTRAL_BASE_PLUS_TWO_GRASP_ARMS:
        desired_count = max(5, 3 + required_anchor_count, task_spec.robot_constraints.min_modules)
        _require_module_budget(desired_count, task_spec, variant)
        roles = {
            0: "central_base",
            1: "left_grasp_arm_root",
            2: "right_grasp_arm_root",
            3: "left_grasp_tip",
            4: "right_grasp_tip",
        }
        poses = {
            0: IDENTITY_POSE,
            1: (0.22, -0.16, 0.0, 0.0, 0.0, 0.0, 1.0),
            2: (0.22, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0),
            3: (0.48, -0.32, 0.0, 0.0, 0.0, 0.0, 1.0),
            4: (0.48, 0.32, 0.0, 0.0, 0.0, 0.0, 1.0),
        }
        edges = [(0, 1, "structural"), (0, 2, "structural"), (1, 3, "grasp_arm"), (2, 4, "grasp_arm")]
        for module_id in range(5, desired_count):
            roles[module_id] = "stabilizer"
            poses[module_id] = (0.22 * float(module_id), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            edges.append((module_id - 1, module_id, "structural"))
        anchors = [3, 4]
        if required_anchor_count > len(anchors):
            anchors.extend(range(5, 5 + required_anchor_count - len(anchors)))
        return _VariantLayout(roles, poses, edges, anchors[:required_anchor_count])

    raise SchemaValidationError(f"Unsupported grasp/carry morphology variant: {variant!r}")


def _require_module_budget(
    desired_count: int,
    task_spec: TaskSpec,
    variant: GraspCarryMorphologyVariant,
) -> None:
    if desired_count > task_spec.robot_constraints.max_modules:
        raise SchemaValidationError(
            f"{variant.value} requires {desired_count} modules, exceeds max_modules="
            f"{task_spec.robot_constraints.max_modules}"
        )


def _distributed_module_ids(module_count: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [module_count - 1]
    if count == 2:
        return [0, module_count - 1]
    return [min(module_count - 1, round(idx * (module_count - 1) / max(1, count - 1))) for idx in range(count)]


def _build_modules(layout: _VariantLayout, capability: ModuleCapabilityToken) -> list[ModuleNode]:
    modules: list[ModuleNode] = []
    for module_id in sorted(layout.module_roles):
        modules.append(
            ModuleNode(
                module_id=module_id,
                module_type=capability.module_type,
                pose_in_design_frame=layout.module_poses[module_id],
                role_id=layout.module_roles[module_id],
                is_base=module_id == 0,
                capability_token=capability,
            )
        )
    return modules


def _build_ports_and_edges(
    layout: _VariantLayout,
    dock_ports: list[DockPortSpec],
    modules: list[ModuleNode],
) -> tuple[list[PortNode], list[DockEdge]]:
    if layout.edge_specs and len(dock_ports) < 2:
        raise SchemaValidationError("grasp/carry morphology variants require at least two dock ports")
    ports = _build_ports(len(modules), dock_ports)
    ports_by_module: dict[int, list[PortNode]] = {}
    for port in ports:
        ports_by_module.setdefault(port.module_id, []).append(port)
    used_port_ids: set[int] = set()
    dock_edges: list[DockEdge] = []
    for edge_id, (src_module_id, dst_module_id, edge_role) in enumerate(layout.edge_specs):
        src_port, dst_port = _first_compatible_free_pair(
            ports_by_module[src_module_id],
            ports_by_module[dst_module_id],
            used_port_ids,
        )
        used_port_ids.update({src_port.port_global_id, dst_port.port_global_id})
        dock_edges.append(
            DockEdge(
                edge_id=edge_id,
                src_module_id=src_module_id,
                src_port_id=src_port.port_global_id,
                dst_module_id=dst_module_id,
                dst_port_id=dst_port.port_global_id,
                relative_pose_src_to_dst=relative_pose_for_dock_ports(src_port, dst_port),
                edge_role=edge_role,  # type: ignore[arg-type]
                estimated_stiffness=[1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0],
                latch_state="planned",
            )
        )
    if not used_port_ids:
        return ports, dock_edges
    return [
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
    ], dock_edges


def _build_ports(module_count: int, dock_ports: list[DockPortSpec]) -> list[PortNode]:
    ports: list[PortNode] = []
    for module_id in range(module_count):
        for local_idx, dock_port in enumerate(dock_ports):
            ports.append(
                PortNode(
                    port_global_id=module_id * len(dock_ports) + local_idx,
                    module_id=module_id,
                    port_local_id=dock_port.port_id,
                    local_pose=dock_port.local_pose,
                    port_type=dock_port.port_type,
                    occupied=False,
                    compatible_port_type_mask=_compatible_mask(dock_port.compatible_port_types),
                )
            )
    return ports


def _anchor_plan(
    required_items: list[_SlotRequirement],
    slot_requirements: list[_SlotRequirement],
    *,
    layout: _VariantLayout,
    variant: GraspCarryMorphologyVariant,
) -> list[_AnchorPlanItem]:
    plan: list[_AnchorPlanItem] = []
    for idx, slot in enumerate(required_items):
        module_id = layout.required_anchor_modules[idx]
        plan.append(
            _AnchorPlanItem(
                slot=slot,
                module_id=module_id,
                local_pose=_anchor_local_pose(slot.contact_mode, idx, len(required_items), variant),
                role_label="required_slot_coverage",
            )
        )
    support_slot = _optional_support_slot(slot_requirements)
    if support_slot is not None and layout.optional_support_module is not None:
        plan.append(
            _AnchorPlanItem(
                slot=support_slot,
                module_id=layout.optional_support_module,
                local_pose=(0.0, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0),
                role_label="optional_support_prior",
            )
        )
    return plan


def _build_anchors(
    anchor_plan: list[_AnchorPlanItem],
    physical_model: PhysicalModel,
    task_spec: TaskSpec,
    variant: GraspCarryMorphologyVariant,
) -> list[RobotAnchor]:
    link_id = _default_anchor_link(physical_model)
    anchors: list[RobotAnchor] = []
    for anchor_id, item in enumerate(anchor_plan):
        anchors.append(
            RobotAnchor(
                anchor_id=anchor_id,
                module_id=item.module_id,
                link_id=link_id,
                local_pose=item.local_pose,
                anchor_type=CONTACT_MODE_TO_ANCHOR_TYPE[item.slot.contact_mode],  # type: ignore[arg-type]
                capability={
                    **item.slot.required_anchor_capability,
                    "max_force_n": task_spec.safety.max_contact_force_n,
                    "max_torque_nm": task_spec.safety.max_contact_torque_nm,
                    "target_entity_id": item.slot.target_entity_id,
                    "contact_mode": item.slot.contact_mode.value,
                    "morphology_variant": variant.value,
                    "anchor_role": item.role_label,
                },
                associated_contact_slot_ids=[item.slot.slot_id],
            )
        )
    return anchors


def _binding_priors(
    anchors: list[RobotAnchor],
    anchor_plan: list[_AnchorPlanItem],
) -> list[SlotAnchorBindingPrior]:
    priors: list[SlotAnchorBindingPrior] = []
    for anchor, item in zip(anchors, anchor_plan):
        priors.append(
            SlotAnchorBindingPrior(
                slot_id=item.slot.slot_id,
                anchor_id=anchor.anchor_id,
                score=1.0 if item.slot.required else 0.65,
                reason_code=item.role_label,
            )
        )
    return priors


def _build_control_groups(
    modules: list[ModuleNode],
    variant: GraspCarryMorphologyVariant,
    layout: _VariantLayout,
) -> list[ControlGroup]:
    groups = [
        ControlGroup(
            group_id="all_modules",
            module_ids=[module.module_id for module in modules],
            role="whole_body",
            metadata={"morphology_variant": variant.value},
        )
    ]
    left_modules = [module_id for module_id, role in layout.module_roles.items() if role.startswith("left_")]
    right_modules = [module_id for module_id, role in layout.module_roles.items() if role.startswith("right_")]
    if left_modules:
        groups.append(ControlGroup(group_id="left_grasp_group", module_ids=sorted(left_modules), role="grasp_arm"))
    if right_modules:
        groups.append(ControlGroup(group_id="right_grasp_group", module_ids=sorted(right_modules), role="grasp_arm"))
    if layout.optional_support_module is not None:
        groups.append(ControlGroup(group_id="support_group", module_ids=[layout.optional_support_module], role="support"))
    return groups


def _design_actions(
    modules: list[ModuleNode],
    dock_edges: list[DockEdge],
    anchors: list[RobotAnchor],
    control_groups: list[ControlGroup],
    variant: GraspCarryMorphologyVariant,
) -> list[DesignAction]:
    actions: list[DesignAction] = [
        DesignAction(DesignActionType.SET_BASE_MODULE, {"module_id": 0, "variant": variant.value})
    ]
    for module in modules:
        actions.append(
            DesignAction(
                DesignActionType.ADD_MODULE,
                {"module_id": module.module_id, "module_type": module.module_type, "variant": variant.value},
            )
        )
        actions.append(
            DesignAction(
                DesignActionType.ASSIGN_ROLE,
                {"module_id": module.module_id, "role_id": module.role_id},
            )
        )
    for edge in dock_edges:
        actions.append(
            DesignAction(
                DesignActionType.CONNECT_PORT,
                {
                    "edge_id": edge.edge_id,
                    "src_port_id": edge.src_port_id,
                    "dst_port_id": edge.dst_port_id,
                    "edge_role": edge.edge_role,
                },
            )
        )
    for anchor in anchors:
        actions.append(
            DesignAction(
                DesignActionType.CREATE_ANCHOR,
                {
                    "anchor_id": anchor.anchor_id,
                    "module_id": anchor.module_id,
                    "anchor_type": anchor.anchor_type,
                },
            )
        )
        for slot_id in anchor.associated_contact_slot_ids:
            actions.append(
                DesignAction(
                    DesignActionType.BIND_ANCHOR_TO_SLOT,
                    {"anchor_id": anchor.anchor_id, "slot_id": slot_id},
                )
            )
    for group in control_groups:
        actions.append(DesignAction(DesignActionType.SET_CONTROL_GROUP, {"group_id": group.group_id}))
    actions.append(DesignAction(DesignActionType.STOP, {"variant": variant.value}))
    return actions


def _slot_requirements(irg: InteractionRequirementGraph) -> list[_SlotRequirement]:
    requirements: list[_SlotRequirement] = []
    for node in sorted(irg.nodes, key=lambda item: item.node_id):
        if node.node_type == IRGNodeType.CONTACT_SLOT:
            requirements.append(_slot_requirement_from_node(node))
    return requirements


def _slot_requirement_from_node(node: IRGNode) -> _SlotRequirement:
    raw_mode = node.feature.get("contact_mode")
    try:
        mode = ContactMode(raw_mode)
    except ValueError as exc:
        raise SchemaValidationError(f"ContactSlot {node.node_id} has unsupported contact_mode {raw_mode!r}") from exc
    if mode not in CONTACT_MODE_TO_ANCHOR_TYPE:
        raise SchemaValidationError(f"ContactSlot {node.node_id} contact_mode {mode.value!r} is not anchor-compatible")
    return _SlotRequirement(
        slot_id=int(node.feature.get("slot_id", node.node_id)),
        contact_mode=mode,
        required=bool(node.feature.get("required", True)),
        min_count=int(node.feature.get("min_count_group", 1)),
        max_count=int(node.feature.get("max_count_group", 1)),
        target_entity_id=str(node.feature.get("target_entity_id", "")),
        required_anchor_capability=dict(node.feature.get("required_anchor_capability", {}) or {}),
    )


def _required_anchor_items(slot_requirements: list[_SlotRequirement]) -> list[_SlotRequirement]:
    items: list[_SlotRequirement] = []
    for slot in slot_requirements:
        if not slot.required:
            continue
        for _ in range(slot.min_count):
            items.append(slot)
    return items


def _optional_support_slot(slot_requirements: list[_SlotRequirement]) -> _SlotRequirement | None:
    for slot in slot_requirements:
        if not slot.required and slot.contact_mode == ContactMode.SUPPORT:
            return slot
    return None


def _anchor_local_pose(
    contact_mode: ContactMode,
    index: int,
    count: int,
    variant: GraspCarryMorphologyVariant,
) -> Pose7D:
    if contact_mode == ContactMode.SUPPORT:
        return (0.0, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0)
    if count <= 1:
        offset = 0.0
    else:
        offset = -0.12 + 0.24 * float(index) / float(count - 1)
    if variant == GraspCarryMorphologyVariant.CHAIN_GRASP:
        return (0.08, offset, 0.0, 0.0, 0.0, 0.0, 1.0)
    return (0.12, offset, 0.0, 0.0, 0.0, 0.0, 1.0)


def _compatible_mask(compatible_port_types: list[str]) -> list[int]:
    return [1 if port_type in compatible_port_types else 0 for port_type in PORT_TYPE_ORDER]


def _ports_compatible(src: PortNode, dst: PortNode) -> bool:
    try:
        dst_idx = PORT_TYPE_ORDER.index(dst.port_type)
        src_idx = PORT_TYPE_ORDER.index(src.port_type)
    except ValueError:
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
    raise SchemaValidationError("No compatible free dock port pair available for grasp/carry variant")


def _default_anchor_link(physical_model: PhysicalModel) -> str | None:
    if physical_model.dock_ports:
        return physical_model.dock_ports[0].parent_link
    if physical_model.links:
        return physical_model.links[0].link_id
    return None
