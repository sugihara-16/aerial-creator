from __future__ import annotations

from dataclasses import dataclass

from amsrr.assembly.construction_state import (
    AssemblyPlan,
    AssemblyStep,
    ConstructionState,
    construction_state_from_current_graph,
    initial_construction_state,
    mark_edge_attached,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import DockEdge, MorphologyGraph


@dataclass(frozen=True)
class AssemblyPlannerConfig:
    move_timeout_s: float = 5.0
    align_timeout_s: float = 4.0
    dock_timeout_s: float = 3.0
    verify_timeout_s: float = 2.0
    fallback_policy: str = "retry_once_then_abort"


class GraphEditAssemblyPlanner:
    """Deterministic v1 π_A planner over MorphologyGraph dock edges."""

    def __init__(self, config: AssemblyPlannerConfig | None = None) -> None:
        self.config = config or AssemblyPlannerConfig()

    def build_plan(
        self,
        target_graph: MorphologyGraph,
        *,
        current_graph: MorphologyGraph | None = None,
        construction_state: ConstructionState | None = None,
    ) -> AssemblyPlan:
        state = self._initial_state(target_graph, current_graph=current_graph, construction_state=construction_state)
        steps: list[AssemblyStep] = []
        attached_edges = _attached_edge_ids(state.physical_graph, target_graph)
        attached_modules = {module.module_id for module in state.physical_graph.modules}

        while len(attached_edges) < len(target_graph.dock_edges):
            edge, leader, follower = self._next_attach_edge(target_graph, attached_modules, attached_edges)
            edge_steps = self._steps_for_edge(edge, leader_module_id=leader, follower_module_id=follower, start_step_id=len(steps))
            steps.extend(edge_steps)
            attached_edges.add(edge.edge_id)
            attached_modules.update({edge.src_module_id, edge.dst_module_id})
            state = mark_edge_attached(state, target_graph, edge.edge_id)

        return AssemblyPlan(
            plan_id=f"assembly:{target_graph.graph_id}:{target_graph.stable_hash()[:12]}",
            target_graph_id=target_graph.graph_id,
            steps=steps,
            estimated_duration_s=sum(step.timeout_s for step in steps),
            fallback_policy=self.config.fallback_policy,
        )

    def next_step(
        self,
        target_graph: MorphologyGraph,
        *,
        current_graph: MorphologyGraph | None = None,
        construction_state: ConstructionState | None = None,
    ) -> AssemblyStep | None:
        plan = self.build_plan(target_graph, current_graph=current_graph, construction_state=construction_state)
        return plan.steps[0] if plan.steps else None

    @staticmethod
    def _initial_state(
        target_graph: MorphologyGraph,
        *,
        current_graph: MorphologyGraph | None,
        construction_state: ConstructionState | None,
    ) -> ConstructionState:
        if construction_state is not None:
            return construction_state
        if current_graph is not None:
            return construction_state_from_current_graph(current_graph, target_graph)
        return initial_construction_state(target_graph)

    @staticmethod
    def _next_attach_edge(
        target_graph: MorphologyGraph,
        attached_modules: set[int],
        attached_edge_ids: set[int],
    ) -> tuple[DockEdge, int, int]:
        for edge in sorted(target_graph.dock_edges, key=lambda item: item.edge_id):
            if edge.edge_id in attached_edge_ids:
                continue
            src_attached = edge.src_module_id in attached_modules
            dst_attached = edge.dst_module_id in attached_modules
            if src_attached and not dst_attached:
                return edge, edge.src_module_id, edge.dst_module_id
            if dst_attached and not src_attached:
                return edge, edge.dst_module_id, edge.src_module_id
        raise SchemaValidationError("No attachable dock edge found from current construction state")

    def _steps_for_edge(
        self,
        edge: DockEdge,
        *,
        leader_module_id: int,
        follower_module_id: int,
        start_step_id: int,
    ) -> list[AssemblyStep]:
        src_port_id = edge.src_port_id
        dst_port_id = edge.dst_port_id
        return [
            AssemblyStep(
                step_id=start_step_id,
                step_type="move_to_staging",
                leader_module_id=leader_module_id,
                follower_module_id=follower_module_id,
                src_port_id=src_port_id,
                dst_port_id=dst_port_id,
                target_relative_pose=None,
                preconditions=[
                    {"type": "module_unattached", "module_id": follower_module_id},
                    {"type": "leader_component_stable", "module_id": leader_module_id},
                ],
                success_conditions=[{"type": "at_staging", "module_id": follower_module_id}],
                timeout_s=self.config.move_timeout_s,
            ),
            AssemblyStep(
                step_id=start_step_id + 1,
                step_type="align_ports",
                leader_module_id=leader_module_id,
                follower_module_id=follower_module_id,
                src_port_id=src_port_id,
                dst_port_id=dst_port_id,
                target_relative_pose=edge.relative_pose_src_to_dst,
                preconditions=[{"type": "at_staging", "module_id": follower_module_id}],
                success_conditions=[
                    {
                        "type": "ports_aligned",
                        "src_port_id": src_port_id,
                        "dst_port_id": dst_port_id,
                    }
                ],
                timeout_s=self.config.align_timeout_s,
            ),
            AssemblyStep(
                step_id=start_step_id + 2,
                step_type="dock",
                leader_module_id=leader_module_id,
                follower_module_id=follower_module_id,
                src_port_id=src_port_id,
                dst_port_id=dst_port_id,
                target_relative_pose=edge.relative_pose_src_to_dst,
                preconditions=[
                    {
                        "type": "ports_aligned",
                        "src_port_id": src_port_id,
                        "dst_port_id": dst_port_id,
                    }
                ],
                success_conditions=[{"type": "dock_latch_commanded", "edge_id": edge.edge_id}],
                timeout_s=self.config.dock_timeout_s,
            ),
            AssemblyStep(
                step_id=start_step_id + 3,
                step_type="verify_attach",
                leader_module_id=leader_module_id,
                follower_module_id=follower_module_id,
                src_port_id=src_port_id,
                dst_port_id=dst_port_id,
                target_relative_pose=edge.relative_pose_src_to_dst,
                preconditions=[{"type": "dock_latch_commanded", "edge_id": edge.edge_id}],
                success_conditions=[
                    {
                        "type": "edge_attached",
                        "edge_id": edge.edge_id,
                        "src_module_id": edge.src_module_id,
                        "dst_module_id": edge.dst_module_id,
                    }
                ],
                timeout_s=self.config.verify_timeout_s,
            ),
        ]


def _attached_edge_ids(current_graph: MorphologyGraph, target_graph: MorphologyGraph) -> set[int]:
    current_keys = {
        (edge.src_module_id, edge.src_port_id, edge.dst_module_id, edge.dst_port_id)
        for edge in current_graph.dock_edges
    }
    return {
        edge.edge_id
        for edge in target_graph.dock_edges
        if (edge.src_module_id, edge.src_port_id, edge.dst_module_id, edge.dst_port_id) in current_keys
    }
