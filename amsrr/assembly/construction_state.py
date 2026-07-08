from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from amsrr.schemas.common import Condition, Pose7D, SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.schemas.feasibility import Violation
from amsrr.schemas.morphology import ControlGroup, DockEdge, MorphologyGraph, PortNode


AssemblyStepType = Literal["move_to_staging", "align_ports", "dock", "verify_attach", "detach", "retry", "abort"]


@dataclass
class AssemblyStep(SchemaBase):
    step_id: int
    step_type: AssemblyStepType
    leader_module_id: int
    follower_module_id: int | None
    src_port_id: int | None
    dst_port_id: int | None
    target_relative_pose: Pose7D | None
    preconditions: list[Condition]
    success_conditions: list[Condition]
    timeout_s: float

    def validate(self) -> None:
        if self.step_id < 0 or self.leader_module_id < 0:
            raise SchemaValidationError("AssemblyStep ids must be non-negative")
        if self.follower_module_id is not None and self.follower_module_id < 0:
            raise SchemaValidationError("AssemblyStep.follower_module_id must be non-negative")
        if self.src_port_id is not None and self.src_port_id < 0:
            raise SchemaValidationError("AssemblyStep.src_port_id must be non-negative")
        if self.dst_port_id is not None and self.dst_port_id < 0:
            raise SchemaValidationError("AssemblyStep.dst_port_id must be non-negative")
        if self.target_relative_pose is not None:
            require_len(self.target_relative_pose, 7, "AssemblyStep.target_relative_pose")
        if self.timeout_s <= 0.0:
            raise SchemaValidationError("AssemblyStep.timeout_s must be positive")


@dataclass
class AssemblyPlan(SchemaBase):
    plan_id: str
    target_graph_id: str
    steps: list[AssemblyStep]
    estimated_duration_s: float
    fallback_policy: str

    def validate(self) -> None:
        require_non_empty(self.plan_id, "AssemblyPlan.plan_id")
        require_non_empty(self.target_graph_id, "AssemblyPlan.target_graph_id")
        require_non_empty(self.fallback_policy, "AssemblyPlan.fallback_policy")
        if self.estimated_duration_s < 0.0:
            raise SchemaValidationError("AssemblyPlan.estimated_duration_s must be non-negative")
        expected_step_ids = list(range(len(self.steps)))
        actual_step_ids = [step.step_id for step in self.steps]
        if actual_step_ids != expected_step_ids:
            raise SchemaValidationError("AssemblyPlan.steps must have contiguous step_id values")


@dataclass
class ConstructionState(SchemaBase):
    physical_graph: MorphologyGraph
    control_graph: MorphologyGraph
    unattached_modules: list[int]
    attached_components: list[list[int]]
    active_step_id: int | None
    docking_attempts: dict[str, int] = field(default_factory=dict)
    failures: list[Violation] = field(default_factory=list)

    def validate(self) -> None:
        if self.active_step_id is not None and self.active_step_id < 0:
            raise SchemaValidationError("ConstructionState.active_step_id must be non-negative")
        if len(self.unattached_modules) != len(set(self.unattached_modules)):
            raise SchemaValidationError("ConstructionState.unattached_modules must be unique")


def initial_construction_state(target_graph: MorphologyGraph) -> ConstructionState:
    """Create a P1/P3 initial state with only the base component assembled."""

    if not target_graph.modules:
        raise SchemaValidationError("Cannot initialize construction state from an empty target graph")
    base_id = target_graph.base_module_id
    module_ids = sorted(module.module_id for module in target_graph.modules)
    if base_id not in module_ids:
        raise SchemaValidationError("Target graph base_module_id must reference a module")
    graph = induced_morphology_subgraph(target_graph, {base_id}, set(), graph_id=f"{target_graph.graph_id}:construction:base")
    unattached = [module_id for module_id in module_ids if module_id != base_id]
    components = [[base_id], *[[module_id] for module_id in unattached]]
    return ConstructionState(
        physical_graph=graph,
        control_graph=graph,
        unattached_modules=unattached,
        attached_components=components,
        active_step_id=None,
        docking_attempts={},
        failures=[],
    )


def construction_state_from_current_graph(current_graph: MorphologyGraph, target_graph: MorphologyGraph) -> ConstructionState:
    attached_module_ids = {module.module_id for module in current_graph.modules}
    target_module_ids = {module.module_id for module in target_graph.modules}
    if not attached_module_ids <= target_module_ids:
        raise SchemaValidationError("Current graph contains modules outside target graph")
    attached_edge_ids = {
        edge.edge_id
        for edge in target_graph.dock_edges
        if _edge_key(edge) in {_edge_key(current_edge) for current_edge in current_graph.dock_edges}
    }
    graph = induced_morphology_subgraph(
        target_graph,
        attached_module_ids,
        attached_edge_ids,
        graph_id=f"{target_graph.graph_id}:construction:current",
    )
    components = connected_components(target_graph, attached_module_ids, attached_edge_ids)
    unattached = sorted(target_module_ids - attached_module_ids)
    components.extend([[module_id] for module_id in unattached])
    return ConstructionState(
        physical_graph=graph,
        control_graph=graph,
        unattached_modules=unattached,
        attached_components=components,
        active_step_id=None,
        docking_attempts={},
        failures=[],
    )


def mark_edge_attached(state: ConstructionState, target_graph: MorphologyGraph, edge_id: int) -> ConstructionState:
    edge_ids = _attached_edge_ids_from_graph(state.physical_graph, target_graph)
    edge_ids.add(edge_id)
    module_ids = {module.module_id for module in state.physical_graph.modules}
    edge = _edge_by_id(target_graph, edge_id)
    module_ids.update({edge.src_module_id, edge.dst_module_id})
    graph = induced_morphology_subgraph(target_graph, module_ids, edge_ids, graph_id=f"{target_graph.graph_id}:construction:{len(edge_ids)}")
    target_module_ids = {module.module_id for module in target_graph.modules}
    unattached = sorted(target_module_ids - module_ids)
    components = connected_components(target_graph, module_ids, edge_ids)
    components.extend([[module_id] for module_id in unattached])
    docking_attempts = dict(state.docking_attempts)
    docking_attempts[str(edge_id)] = docking_attempts.get(str(edge_id), 0) + 1
    return ConstructionState(
        physical_graph=graph,
        control_graph=graph,
        unattached_modules=unattached,
        attached_components=components,
        active_step_id=None,
        docking_attempts=docking_attempts,
        failures=list(state.failures),
    )


def induced_morphology_subgraph(
    target_graph: MorphologyGraph,
    module_ids: set[int],
    dock_edge_ids: set[int],
    *,
    graph_id: str,
) -> MorphologyGraph:
    if target_graph.base_module_id not in module_ids:
        raise SchemaValidationError("Construction subgraph must include the target base module")

    modules = [module for module in target_graph.modules if module.module_id in module_ids]
    included_edges = [
        replace(edge, latch_state="attached")
        for edge in target_graph.dock_edges
        if edge.edge_id in dock_edge_ids and edge.src_module_id in module_ids and edge.dst_module_id in module_ids
    ]
    occupied_port_ids = {edge.src_port_id for edge in included_edges} | {edge.dst_port_id for edge in included_edges}
    ports = [
        replace(port, occupied=port.port_global_id in occupied_port_ids)
        for port in target_graph.ports
        if port.module_id in module_ids
    ]
    anchors = [anchor for anchor in target_graph.robot_anchors if anchor.module_id in module_ids]
    control_groups = _filtered_control_groups(target_graph.control_groups, module_ids)
    return MorphologyGraph(
        graph_id=graph_id,
        modules=modules,
        ports=ports,
        dock_edges=included_edges,
        robot_anchors=anchors,
        control_groups=control_groups,
        base_module_id=target_graph.base_module_id,
        is_closed_loop=False,
    )


def connected_components(target_graph: MorphologyGraph, module_ids: set[int], dock_edge_ids: set[int]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {module_id: set() for module_id in module_ids}
    for edge in target_graph.dock_edges:
        if edge.edge_id not in dock_edge_ids:
            continue
        if edge.src_module_id in module_ids and edge.dst_module_id in module_ids:
            adjacency[edge.src_module_id].add(edge.dst_module_id)
            adjacency[edge.dst_module_id].add(edge.src_module_id)

    components: list[list[int]] = []
    seen: set[int] = set()
    for module_id in sorted(module_ids):
        if module_id in seen:
            continue
        stack = [module_id]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        components.append(sorted(component))
    return components


def _filtered_control_groups(control_groups: list[ControlGroup], module_ids: set[int]) -> list[ControlGroup]:
    filtered: list[ControlGroup] = []
    for group in control_groups:
        group_modules = [module_id for module_id in group.module_ids if module_id in module_ids]
        if group_modules:
            filtered.append(replace(group, module_ids=group_modules))
    if filtered:
        return filtered
    return [ControlGroup(group_id="assembled_component", module_ids=sorted(module_ids), role="assembly_component")]


def _edge_key(edge: DockEdge) -> tuple[int, int, int, int]:
    return (edge.src_module_id, edge.src_port_id, edge.dst_module_id, edge.dst_port_id)


def _edge_by_id(target_graph: MorphologyGraph, edge_id: int) -> DockEdge:
    for edge in target_graph.dock_edges:
        if edge.edge_id == edge_id:
            return edge
    raise SchemaValidationError(f"Unknown target dock edge_id {edge_id}")


def _attached_edge_ids_from_graph(current_graph: MorphologyGraph, target_graph: MorphologyGraph) -> set[int]:
    current_keys = {_edge_key(edge) for edge in current_graph.dock_edges}
    return {edge.edge_id for edge in target_graph.dock_edges if _edge_key(edge) in current_keys}
