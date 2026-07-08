from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.irg import IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import DesignAction, DesignActionType, DesignOutput, MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec


@dataclass(frozen=True)
class DesignActionCandidate:
    candidate_id: int
    action: DesignAction
    valid: bool
    reason_code: str
    score_prior: float = 0.0


@dataclass(frozen=True)
class DesignCandidateStep:
    step_index: int
    selected_action: DesignAction
    candidates: list[DesignActionCandidate]


class DesignCandidateGenerator:
    """Small deterministic action-mask scaffold for π_D teacher traces."""

    def build_teacher_trace(self, design_output: DesignOutput) -> list[DesignCandidateStep]:
        steps: list[DesignCandidateStep] = []
        for step_index, action in enumerate(design_output.design_actions):
            if action.action_type == DesignActionType.STOP:
                candidates = [
                    DesignActionCandidate(
                        candidate_id=step_index * 2,
                        action=action,
                        valid=True,
                        reason_code="teacher_stop_valid",
                        score_prior=1.0,
                    )
                ]
            else:
                candidates = [
                    DesignActionCandidate(
                        candidate_id=step_index * 2,
                        action=action,
                        valid=True,
                        reason_code="teacher_selected",
                        score_prior=1.0,
                    ),
                    DesignActionCandidate(
                        candidate_id=step_index * 2 + 1,
                        action=DesignAction(DesignActionType.STOP, {}),
                        valid=False,
                        reason_code="stop_masked_until_teacher_complete",
                        score_prior=0.0,
                    ),
                ]
            steps.append(DesignCandidateStep(step_index=step_index, selected_action=action, candidates=candidates))
        return steps

    def final_stop_candidate(
        self,
        design_output: DesignOutput,
        *,
        task_spec: TaskSpec | None = None,
        irg: InteractionRequirementGraph | None = None,
        feasibility_result: FeasibilityResult | None = None,
    ) -> DesignActionCandidate:
        valid, reason = self.stop_validity(
            design_output,
            task_spec=task_spec,
            irg=irg,
            feasibility_result=feasibility_result,
        )
        return DesignActionCandidate(
            candidate_id=0,
            action=DesignAction(DesignActionType.STOP, {}),
            valid=valid,
            reason_code=reason,
            score_prior=1.0 if valid else 0.0,
        )

    def stop_validity(
        self,
        design_output: DesignOutput,
        *,
        task_spec: TaskSpec | None = None,
        irg: InteractionRequirementGraph | None = None,
        feasibility_result: FeasibilityResult | None = None,
    ) -> tuple[bool, str]:
        morphology = design_output.target_morphology
        if task_spec is not None:
            module_count = len(morphology.modules)
            if module_count < task_spec.robot_constraints.min_modules:
                return False, "below_min_module_count"
            if module_count > task_spec.robot_constraints.max_modules:
                return False, "above_max_module_count"
            if morphology.is_closed_loop and not task_spec.robot_constraints.allow_closed_loop:
                return False, "closed_loop_rejected"
        if not _has_single_base(morphology):
            return False, "base_module_not_assigned"
        if not _connected(morphology):
            return False, "morphology_not_connected"
        if _has_port_conflict(morphology):
            return False, "port_occupancy_conflict"
        if irg is not None and not _required_slots_covered(design_output, irg):
            return False, "required_slot_uncovered"
        if feasibility_result is not None and not feasibility_result.feasible:
            return False, "feasibility_rejected"
        return True, "stop_valid"


def _has_single_base(morphology: MorphologyGraph) -> bool:
    module_ids = {module.module_id for module in morphology.modules}
    return morphology.base_module_id in module_ids and sum(1 for module in morphology.modules if module.is_base) == 1


def _connected(morphology: MorphologyGraph) -> bool:
    module_ids = {module.module_id for module in morphology.modules}
    if not module_ids:
        return False
    adjacency: dict[int, set[int]] = {module_id: set() for module_id in module_ids}
    for edge in morphology.dock_edges:
        if edge.src_module_id not in module_ids or edge.dst_module_id not in module_ids:
            return False
        adjacency[edge.src_module_id].add(edge.dst_module_id)
        adjacency[edge.dst_module_id].add(edge.src_module_id)
    visited: set[int] = set()
    queue = deque([morphology.base_module_id])
    while queue:
        module_id = queue.popleft()
        if module_id in visited:
            continue
        visited.add(module_id)
        queue.extend(adjacency[module_id] - visited)
    return visited == module_ids


def _has_port_conflict(morphology: MorphologyGraph) -> bool:
    ports_by_id = {port.port_global_id: port for port in morphology.ports}
    uses: dict[int, int] = defaultdict(int)
    for edge in morphology.dock_edges:
        if edge.src_port_id not in ports_by_id or edge.dst_port_id not in ports_by_id:
            return True
        uses[edge.src_port_id] += 1
        uses[edge.dst_port_id] += 1
        if not ports_by_id[edge.src_port_id].occupied or not ports_by_id[edge.dst_port_id].occupied:
            return True
    return any(count > 1 for count in uses.values())


def _required_slots_covered(design_output: DesignOutput, irg: InteractionRequirementGraph) -> bool:
    required_slot_counts: dict[int, int] = {}
    for node in irg.nodes:
        if node.node_type != IRGNodeType.CONTACT_SLOT or not node.feature.get("required", True):
            continue
        slot_id = int(node.feature.get("slot_id", node.node_id))
        required_slot_counts[slot_id] = int(node.feature.get("min_count_group", 1))

    anchor_counts: dict[int, int] = defaultdict(int)
    for prior in design_output.slot_anchor_binding_prior:
        anchor_counts[int(prior.slot_id)] += 1
    return all(anchor_counts.get(slot_id, 0) >= min_count for slot_id, min_count in required_slot_counts.items())
