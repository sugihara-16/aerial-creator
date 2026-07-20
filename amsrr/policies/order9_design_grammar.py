from __future__ import annotations

"""Deterministic candidate grammar and transition system for learned pi_D."""

from dataclasses import dataclass, replace
from typing import Iterable

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.morphology.dock_geometry import (
    modules_with_dock_aligned_poses,
    relative_pose_for_dock_ports,
)
from amsrr.morphology.graph import CONTACT_MODE_TO_ANCHOR_TYPE, PORT_TYPE_ORDER
from amsrr.policies.design_candidate_generator import (
    DesignActionCandidate,
    DesignCandidateStep,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.robot_model.physical_model_builder import build_module_capability_token
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.irg import IRGNodeType
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
from amsrr.utils.hashing import stable_hash


ORDER9_DESIGN_GRAMMAR_VERSION = "order9_holon_sequential_design_grammar_v1"


@dataclass(frozen=True)
class PartialDesignAnchor:
    anchor_id: int
    module_id: int
    surface_port_id: int
    link_id: str | None
    local_pose: Pose7D
    anchor_type: str
    capability_items: tuple[tuple[str, object], ...]
    bound_slot_ids: tuple[int, ...] = ()

    @property
    def capability(self) -> dict[str, object]:
        return dict(self.capability_items)


@dataclass(frozen=True)
class Order9PartialDesignState:
    module_ids: tuple[int, ...] = ()
    base_module_id: int | None = None
    connected_port_pairs: tuple[tuple[int, int], ...] = ()
    module_roles: tuple[tuple[int, str], ...] = ()
    anchors: tuple[PartialDesignAnchor, ...] = ()
    control_group_assigned: bool = False
    stopped: bool = False
    action_history: tuple[DesignAction, ...] = ()

    @property
    def role_by_module(self) -> dict[int, str]:
        return dict(self.module_roles)


@dataclass(frozen=True)
class Order9DesignTeacherStep:
    state: Order9PartialDesignState
    candidate_step: DesignCandidateStep


class Order9DesignGrammar:
    """Enumerate masked graph-edit actions over homogeneous Holon inventory.

    The grammar owns syntax only.  Learned pi_D chooses among valid candidates;
    the deterministic ``FeasibilityChecker`` remains the STOP authority.
    """

    def __init__(
        self,
        context: DesignPolicyContext,
        *,
        checker: FeasibilityChecker | None = None,
    ) -> None:
        self.context = context
        self.checker = checker or FeasibilityChecker()
        self._capability = build_module_capability_token(
            context.physical_model,
            module_type=context.physical_model.model_id,
        )
        if not context.physical_model.dock_ports:
            raise SchemaValidationError("Order 9 design grammar requires docking ports")

    @staticmethod
    def initial_state() -> Order9PartialDesignState:
        return Order9PartialDesignState()

    def candidates(
        self,
        state: Order9PartialDesignState,
    ) -> list[DesignActionCandidate]:
        if state.stopped:
            return []
        actions: list[tuple[DesignAction, str, float]] = []
        constraints = self.context.task_spec.robot_constraints
        if state.base_module_id is None:
            if len(state.module_ids) < constraints.max_modules:
                module_id = len(state.module_ids)
                actions.append(
                    (
                        DesignAction(
                            DesignActionType.ADD_MODULE,
                            {
                                "module_id": module_id,
                                "module_type": self.context.physical_model.model_id,
                            },
                        ),
                        "inventory_module_available",
                        0.0,
                    )
                )
            if len(state.module_ids) >= constraints.min_modules:
                for module_id in state.module_ids:
                    actions.append(
                        (
                            DesignAction(
                                DesignActionType.SET_BASE_MODULE,
                                {"module_id": module_id},
                            ),
                            "base_candidate",
                            0.0,
                        )
                    )
        elif not self._connected(state):
            for src, dst in self._connection_candidates(state):
                actions.append(
                    (
                        DesignAction(
                            DesignActionType.CONNECT_PORT,
                            {
                                "edge_id": len(state.connected_port_pairs),
                                "src_module_id": src.module_id,
                                "src_port_id": src.port_global_id,
                                "dst_module_id": dst.module_id,
                                "dst_port_id": dst.port_global_id,
                                "edge_role": "structural",
                            },
                        ),
                        "compatible_free_ports_between_components",
                        0.0,
                    )
                )
        elif len(state.module_roles) < len(state.module_ids):
            unassigned = sorted(set(state.module_ids) - set(state.role_by_module))[0]
            for role in self._role_vocabulary(unassigned == state.base_module_id):
                actions.append(
                    (
                        DesignAction(
                            DesignActionType.ASSIGN_ROLE,
                            {"module_id": unassigned, "role_id": role},
                        ),
                        "role_vocabulary_candidate",
                        0.0,
                    )
                )
        else:
            unbound = [anchor for anchor in state.anchors if not anchor.bound_slot_ids]
            if unbound:
                anchor = unbound[0]
                for slot in self._compatible_unfilled_slots(state, anchor.anchor_type):
                    actions.append(
                        (
                            DesignAction(
                                DesignActionType.BIND_ANCHOR_TO_SLOT,
                                {
                                    "anchor_id": anchor.anchor_id,
                                    "slot_id": slot["slot_id"],
                                },
                            ),
                            "anchor_slot_mode_compatible",
                            0.0,
                        )
                    )
            else:
                required_slot = self._next_unfilled_required_slot(state)
                if required_slot is not None:
                    actions.extend(self._anchor_creation_actions(state, required_slot))
                elif not state.control_group_assigned:
                    for optional_slot in self._unfilled_optional_slots(state):
                        actions.extend(self._anchor_creation_actions(state, optional_slot))
                    actions.append(
                        (
                            DesignAction(
                                DesignActionType.SET_CONTROL_GROUP,
                                {
                                    "group_id": "all_modules",
                                    "module_ids": list(state.module_ids),
                                    "role": "whole_body",
                                },
                            ),
                            "required_slot_coverage_complete",
                            0.0,
                        )
                    )

        candidates = [
            DesignActionCandidate(
                candidate_id=index,
                action=action,
                valid=True,
                reason_code=reason,
                score_prior=prior,
            )
            for index, (action, reason, prior) in enumerate(actions)
        ]
        stop_valid, stop_reason = self.stop_validity(state)
        candidates.append(
            DesignActionCandidate(
                candidate_id=len(candidates),
                action=DesignAction(DesignActionType.STOP, {}),
                valid=stop_valid,
                reason_code=stop_reason,
                score_prior=0.0,
            )
        )
        return candidates

    def apply(
        self,
        state: Order9PartialDesignState,
        candidate: DesignActionCandidate,
    ) -> Order9PartialDesignState:
        if state.stopped:
            raise SchemaValidationError("cannot apply a design action after STOP")
        current = self.candidates(state)
        matching = [
            item
            for item in current
            if item.candidate_id == candidate.candidate_id
            and item.action.to_dict() == candidate.action.to_dict()
        ]
        if not matching or not matching[0].valid:
            raise SchemaValidationError("selected pi_D action is absent or masked")
        action = matching[0].action
        next_state = state
        if action.action_type == DesignActionType.ADD_MODULE:
            next_state = replace(
                state,
                module_ids=(*state.module_ids, int(action.params["module_id"])),
            )
        elif action.action_type == DesignActionType.SET_BASE_MODULE:
            next_state = replace(state, base_module_id=int(action.params["module_id"]))
        elif action.action_type == DesignActionType.CONNECT_PORT:
            next_state = replace(
                state,
                connected_port_pairs=(
                    *state.connected_port_pairs,
                    (
                        int(action.params["src_port_id"]),
                        int(action.params["dst_port_id"]),
                    ),
                ),
            )
        elif action.action_type == DesignActionType.ASSIGN_ROLE:
            next_state = replace(
                state,
                module_roles=tuple(
                    sorted(
                        (*state.module_roles, (int(action.params["module_id"]), str(action.params["role_id"]))),
                    )
                ),
            )
        elif action.action_type == DesignActionType.CREATE_ANCHOR:
            capability = action.params["capability"]
            if not isinstance(capability, dict):
                raise SchemaValidationError("CREATE_ANCHOR capability must be a mapping")
            anchor = PartialDesignAnchor(
                anchor_id=int(action.params["anchor_id"]),
                module_id=int(action.params["module_id"]),
                surface_port_id=int(action.params["surface_port_id"]),
                link_id=(
                    None
                    if action.params.get("link_id") is None
                    else str(action.params["link_id"])
                ),
                local_pose=tuple(float(value) for value in action.params["local_pose"]),  # type: ignore[arg-type]
                anchor_type=str(action.params["anchor_type"]),
                capability_items=tuple(sorted(capability.items())),
            )
            next_state = replace(state, anchors=(*state.anchors, anchor))
        elif action.action_type == DesignActionType.BIND_ANCHOR_TO_SLOT:
            anchor_id = int(action.params["anchor_id"])
            slot_id = int(action.params["slot_id"])
            next_state = replace(
                state,
                anchors=tuple(
                    replace(anchor, bound_slot_ids=(*anchor.bound_slot_ids, slot_id))
                    if anchor.anchor_id == anchor_id
                    else anchor
                    for anchor in state.anchors
                ),
            )
        elif action.action_type == DesignActionType.SET_CONTROL_GROUP:
            next_state = replace(state, control_group_assigned=True)
        elif action.action_type == DesignActionType.STOP:
            next_state = replace(state, stopped=True)
        else:
            raise SchemaValidationError(
                f"Order 9 grammar does not support transition {action.action_type.value!r}"
            )
        return replace(
            next_state,
            action_history=(*state.action_history, action),
        )

    def stop_validity(self, state: Order9PartialDesignState) -> tuple[bool, str]:
        if state.base_module_id is None:
            return False, "stop_masked_base_unassigned"
        if not self._connected(state):
            return False, "stop_masked_graph_disconnected"
        if len(state.module_roles) != len(state.module_ids):
            return False, "stop_masked_roles_incomplete"
        if any(not anchor.bound_slot_ids for anchor in state.anchors):
            return False, "stop_masked_unbound_anchor"
        if self._next_unfilled_required_slot(state) is not None:
            return False, "stop_masked_required_slot_uncovered"
        if not state.control_group_assigned:
            return False, "stop_masked_control_group_missing"
        design = self.build_design_output(state, include_stop=False)
        result = self.checker.check_design(
            design,
            task_spec=self.context.task_spec,
            irg=self.context.irg,
            physical_model=self.context.physical_model,
        )
        if not result.feasible:
            codes = sorted({violation.code for violation in result.hard_violations})
            return False, "stop_masked_feasibility:" + ",".join(codes)
        return True, "stop_valid_hard_feasible"

    def build_design_output(
        self,
        state: Order9PartialDesignState,
        *,
        include_stop: bool = True,
    ) -> DesignOutput:
        if state.base_module_id is None or not state.module_ids:
            raise SchemaValidationError("cannot materialize design without modules and base")
        roles = state.role_by_module
        ports = self._ports(state)
        port_by_id = {port.port_global_id: port for port in ports}
        edges = []
        for edge_id, (src_id, dst_id) in enumerate(state.connected_port_pairs):
            src = port_by_id[src_id]
            dst = port_by_id[dst_id]
            edges.append(
                DockEdge(
                    edge_id=edge_id,
                    src_module_id=src.module_id,
                    src_port_id=src_id,
                    dst_module_id=dst.module_id,
                    dst_port_id=dst_id,
                    relative_pose_src_to_dst=relative_pose_for_dock_ports(src, dst),
                    edge_role="structural",
                    estimated_stiffness=[1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0],
                    latch_state="planned",
                )
            )
        modules = [
            ModuleNode(
                module_id=module_id,
                module_type=self.context.physical_model.model_id,
                pose_in_design_frame=(
                    0.25 * float(module_id),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                role_id=roles.get(module_id, "unassigned"),
                is_base=module_id == state.base_module_id,
                capability_token=self._capability,
            )
            for module_id in state.module_ids
        ]
        if self._connected(state):
            modules = modules_with_dock_aligned_poses(
                modules, edges, base_module_id=state.base_module_id
            )
        anchors = [
            RobotAnchor(
                anchor_id=anchor.anchor_id,
                module_id=anchor.module_id,
                link_id=anchor.link_id,
                local_pose=anchor.local_pose,
                anchor_type=anchor.anchor_type,  # type: ignore[arg-type]
                capability=anchor.capability,
                associated_contact_slot_ids=list(anchor.bound_slot_ids),
            )
            for anchor in state.anchors
        ]
        groups = (
            [
                ControlGroup(
                    group_id="all_modules",
                    module_ids=list(state.module_ids),
                    role="whole_body",
                    metadata={"grammar_version": ORDER9_DESIGN_GRAMMAR_VERSION},
                )
            ]
            if state.control_group_assigned
            else []
        )
        actions = list(state.action_history)
        if include_stop and (not actions or actions[-1].action_type != DesignActionType.STOP):
            actions.append(DesignAction(DesignActionType.STOP, {}))
        graph_fingerprint = stable_hash(
            {
                "task_id": self.context.task_spec.task_id,
                "actions": [action.to_dict() for action in actions],
                "grammar_version": ORDER9_DESIGN_GRAMMAR_VERSION,
            }
        )[:16]
        morphology = MorphologyGraph(
            graph_id=f"order9-design:{self.context.task_spec.task_id}:{graph_fingerprint}",
            modules=modules,
            ports=ports,
            dock_edges=edges,
            robot_anchors=anchors,
            control_groups=groups,
            base_module_id=state.base_module_id,
            is_closed_loop=False,
        )
        priors = [
            SlotAnchorBindingPrior(
                slot_id=slot_id,
                anchor_id=anchor.anchor_id,
                score=1.0,
                reason_code="order9_pi_d_selected_binding",
            )
            for anchor in state.anchors
            for slot_id in anchor.bound_slot_ids
        ]
        return DesignOutput(
            task_id=self.context.task_spec.task_id,
            irg_id=self.context.irg.irg_id,
            target_morphology=morphology,
            module_roles=roles,
            slot_anchor_binding_prior=priors,
            design_actions=actions,
            design_scores={
                "order9_sequential_pi_d": 1.0,
                "module_count": float(len(modules)),
                "action_count": float(len(actions)),
            },
        )

    def teacher_trace(
        self,
        target: DesignOutput,
        *,
        maximum_steps: int = 256,
    ) -> list[Order9DesignTeacherStep]:
        """Replay a deterministic design through the same runtime action masks."""

        state = self.initial_state()
        trace: list[Order9DesignTeacherStep] = []
        target_edges = {
            frozenset((edge.src_port_id, edge.dst_port_id)): edge
            for edge in target.target_morphology.dock_edges
        }
        target_roles = target.module_roles
        target_bindings = {
            anchor.anchor_id: list(anchor.associated_contact_slot_ids)
            for anchor in target.target_morphology.robot_anchors
        }
        while not state.stopped:
            if len(trace) >= maximum_steps:
                raise SchemaValidationError("teacher design trace exceeded maximum_steps")
            candidates = self.candidates(state)
            selected = self._select_teacher_candidate(
                state,
                candidates,
                target=target,
                target_edges=target_edges,
                target_roles=target_roles,
                target_bindings=target_bindings,
            )
            trace.append(
                Order9DesignTeacherStep(
                    state=state,
                    candidate_step=DesignCandidateStep(
                        step_index=len(trace),
                        selected_action=selected.action,
                        candidates=candidates,
                    ),
                )
            )
            state = self.apply(state, selected)
        return trace

    def _select_teacher_candidate(
        self,
        state: Order9PartialDesignState,
        candidates: list[DesignActionCandidate],
        *,
        target: DesignOutput,
        target_edges: dict[frozenset[int], DockEdge],
        target_roles: dict[int, str],
        target_bindings: dict[int, list[int]],
    ) -> DesignActionCandidate:
        valid = [candidate for candidate in candidates if candidate.valid]
        by_type: dict[DesignActionType, list[DesignActionCandidate]] = {}
        for candidate in valid:
            by_type.setdefault(candidate.action.action_type, []).append(candidate)
        if DesignActionType.ADD_MODULE in by_type:
            if len(state.module_ids) < len(target.target_morphology.modules):
                return by_type[DesignActionType.ADD_MODULE][0]
        if DesignActionType.SET_BASE_MODULE in by_type:
            for candidate in by_type[DesignActionType.SET_BASE_MODULE]:
                if int(candidate.action.params["module_id"]) == target.target_morphology.base_module_id:
                    return candidate
        if DesignActionType.CONNECT_PORT in by_type:
            used = {frozenset(pair) for pair in state.connected_port_pairs}
            for candidate in by_type[DesignActionType.CONNECT_PORT]:
                pair = frozenset(
                    (
                        int(candidate.action.params["src_port_id"]),
                        int(candidate.action.params["dst_port_id"]),
                    )
                )
                if pair in target_edges and pair not in used:
                    return candidate
        if DesignActionType.ASSIGN_ROLE in by_type:
            module_id = int(by_type[DesignActionType.ASSIGN_ROLE][0].action.params["module_id"])
            desired = target_roles.get(module_id)
            for candidate in by_type[DesignActionType.ASSIGN_ROLE]:
                if candidate.action.params["role_id"] == desired:
                    return candidate
            return by_type[DesignActionType.ASSIGN_ROLE][0]
        if DesignActionType.BIND_ANCHOR_TO_SLOT in by_type:
            anchor_id = int(by_type[DesignActionType.BIND_ANCHOR_TO_SLOT][0].action.params["anchor_id"])
            desired_slots = target_bindings.get(anchor_id, [])
            for candidate in by_type[DesignActionType.BIND_ANCHOR_TO_SLOT]:
                if int(candidate.action.params["slot_id"]) in desired_slots:
                    return candidate
            return by_type[DesignActionType.BIND_ANCHOR_TO_SLOT][0]
        if DesignActionType.CREATE_ANCHOR in by_type:
            next_anchor_id = len(state.anchors)
            desired_anchor = next(
                (
                    anchor
                    for anchor in target.target_morphology.robot_anchors
                    if anchor.anchor_id == next_anchor_id
                ),
                None,
            )
            if desired_anchor is not None:
                desired_slots = set(desired_anchor.associated_contact_slot_ids)
                for candidate in by_type[DesignActionType.CREATE_ANCHOR]:
                    if (
                        int(candidate.action.params["module_id"]) == desired_anchor.module_id
                        and int(candidate.action.params["suggested_slot_id"]) in desired_slots
                    ):
                        return candidate
            return by_type[DesignActionType.CREATE_ANCHOR][0]
        if DesignActionType.SET_CONTROL_GROUP in by_type:
            return by_type[DesignActionType.SET_CONTROL_GROUP][0]
        if DesignActionType.STOP in by_type:
            return by_type[DesignActionType.STOP][0]
        raise SchemaValidationError("target design is not expressible by the Order 9 grammar")

    def _ports(self, state: Order9PartialDesignState) -> list[PortNode]:
        used = {port_id for pair in state.connected_port_pairs for port_id in pair}
        count = len(self.context.physical_model.dock_ports)
        return [
            PortNode(
                port_global_id=module_id * count + local_index,
                module_id=module_id,
                port_local_id=port.port_id,
                local_pose=port.local_pose,
                port_type=port.port_type,
                occupied=module_id * count + local_index in used,
                compatible_port_type_mask=[
                    1 if name in port.compatible_port_types else 0
                    for name in PORT_TYPE_ORDER
                ],
            )
            for module_id in state.module_ids
            for local_index, port in enumerate(self.context.physical_model.dock_ports)
        ]

    def _connection_candidates(
        self, state: Order9PartialDesignState
    ) -> list[tuple[PortNode, PortNode]]:
        ports = [port for port in self._ports(state) if not port.occupied]
        component = self._component_by_module(state)
        candidates = []
        for left_index, left in enumerate(ports):
            for right in ports[left_index + 1 :]:
                if left.module_id == right.module_id:
                    continue
                if component[left.module_id] == component[right.module_id]:
                    continue
                if _ports_compatible(left, right):
                    src, dst = sorted(
                        (left, right), key=lambda port: (port.module_id, port.port_global_id)
                    )
                    candidates.append((src, dst))
        return sorted(
            candidates,
            key=lambda pair: (
                pair[0].module_id,
                pair[1].module_id,
                pair[0].port_global_id,
                pair[1].port_global_id,
            ),
        )

    def _anchor_creation_actions(
        self,
        state: Order9PartialDesignState,
        slot: dict[str, object],
    ) -> list[tuple[DesignAction, str, float]]:
        used_surfaces = {anchor.surface_port_id for anchor in state.anchors}
        same_slot_modules = {
            anchor.module_id
            for anchor in state.anchors
            if int(slot["slot_id"]) in anchor.bound_slot_ids
        }
        physical_by_id = {
            item.port_id: item for item in self.context.physical_model.dock_ports
        }
        actions = []
        for port in self._ports(state):
            if port.occupied or port.port_global_id in used_surfaces:
                continue
            if port.module_id in same_slot_modules:
                continue
            physical = physical_by_id[port.port_local_id]
            capability = {
                **dict(slot.get("required_anchor_capability", {})),
                "max_force_n": self.context.task_spec.safety.max_contact_force_n,
                "max_torque_nm": self.context.task_spec.safety.max_contact_torque_nm,
                "target_entity_id": slot["target_entity_id"],
                "contact_mode": str(slot["contact_mode"]),
                "surface_port_id": port.port_global_id,
            }
            mode = ContactMode(str(slot["contact_mode"]))
            actions.append(
                (
                    DesignAction(
                        DesignActionType.CREATE_ANCHOR,
                        {
                            "anchor_id": len(state.anchors),
                            "module_id": port.module_id,
                            "surface_port_id": port.port_global_id,
                            "link_id": physical.parent_link,
                            "local_pose": list(physical.local_pose),
                            "anchor_type": CONTACT_MODE_TO_ANCHOR_TYPE[mode],
                            "capability": capability,
                            "suggested_slot_id": int(slot["slot_id"]),
                        },
                    ),
                    "free_surface_anchor_candidate",
                    0.0,
                )
            )
        return actions

    def _slot_requirements(self) -> list[dict[str, object]]:
        requirements = []
        for node in sorted(self.context.irg.nodes, key=lambda item: item.node_id):
            if node.node_type != IRGNodeType.CONTACT_SLOT:
                continue
            requirements.append(
                {
                    "slot_id": int(node.feature.get("slot_id", node.node_id)),
                    "contact_mode": str(node.feature["contact_mode"]),
                    "required": bool(node.feature.get("required", True)),
                    "min_count": int(node.feature.get("min_count_group", 1)),
                    "max_count": int(node.feature.get("max_count_group", 1)),
                    "target_entity_id": str(node.feature.get("target_entity_id", "")),
                    "required_anchor_capability": dict(
                        node.feature.get("required_anchor_capability", {})
                    ),
                }
            )
        return requirements

    def _coverage(self, state: Order9PartialDesignState) -> dict[int, int]:
        coverage: dict[int, int] = {}
        for anchor in state.anchors:
            for slot_id in anchor.bound_slot_ids:
                coverage[slot_id] = coverage.get(slot_id, 0) + 1
        return coverage

    def _next_unfilled_required_slot(
        self, state: Order9PartialDesignState
    ) -> dict[str, object] | None:
        coverage = self._coverage(state)
        return next(
            (
                slot
                for slot in self._slot_requirements()
                if bool(slot["required"])
                and coverage.get(int(slot["slot_id"]), 0) < int(slot["min_count"])
            ),
            None,
        )

    def _unfilled_optional_slots(
        self, state: Order9PartialDesignState
    ) -> list[dict[str, object]]:
        coverage = self._coverage(state)
        return [
            slot
            for slot in self._slot_requirements()
            if not bool(slot["required"])
            and coverage.get(int(slot["slot_id"]), 0) < int(slot["max_count"])
        ]

    def _compatible_unfilled_slots(
        self,
        state: Order9PartialDesignState,
        anchor_type: str,
    ) -> list[dict[str, object]]:
        coverage = self._coverage(state)
        return [
            slot
            for slot in self._slot_requirements()
            if CONTACT_MODE_TO_ANCHOR_TYPE[ContactMode(str(slot["contact_mode"]))]
            == anchor_type
            and coverage.get(int(slot["slot_id"]), 0) < int(slot["max_count"])
        ]

    def _connected(self, state: Order9PartialDesignState) -> bool:
        if not state.module_ids:
            return False
        components = set(self._component_by_module(state).values())
        return len(components) == 1

    def _component_by_module(
        self, state: Order9PartialDesignState
    ) -> dict[int, int]:
        ports = {port.port_global_id: port for port in self._ports(state)}
        adjacency = {module_id: set() for module_id in state.module_ids}
        for left_id, right_id in state.connected_port_pairs:
            left = ports[left_id].module_id
            right = ports[right_id].module_id
            adjacency[left].add(right)
            adjacency[right].add(left)
        result: dict[int, int] = {}
        for module_id in state.module_ids:
            if module_id in result:
                continue
            component_id = module_id
            stack = [module_id]
            while stack:
                current = stack.pop()
                if current in result:
                    continue
                result[current] = component_id
                stack.extend(adjacency[current] - set(result))
        return result

    def _role_vocabulary(self, is_base: bool) -> tuple[str, ...]:
        modes = {
            str(slot["contact_mode"])
            for slot in self._slot_requirements()
        }
        common = ["base", "central_base", "base_support"] if is_base else [
            "member",
            "structural",
            "anchor_carrier",
            "chain_grasp_link",
            "left_grasp_arm",
            "right_grasp_arm",
            "left_grasp_arm_root",
            "right_grasp_arm_root",
            "left_grasp_tip",
            "right_grasp_tip",
            "stabilizer",
        ]
        common.extend(f"{mode}_anchor_carrier" for mode in sorted(modes))
        return tuple(dict.fromkeys(common))


def _ports_compatible(left: PortNode, right: PortNode) -> bool:
    try:
        left_accepts = bool(
            left.compatible_port_type_mask[PORT_TYPE_ORDER.index(right.port_type)]
        )
        right_accepts = bool(
            right.compatible_port_type_mask[PORT_TYPE_ORDER.index(left.port_type)]
        )
    except (ValueError, IndexError):
        return False
    return left_accepts and right_accepts
