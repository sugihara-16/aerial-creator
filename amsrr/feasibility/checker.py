from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from amsrr.feasibility import violation_codes as codes
from amsrr.morphology.graph import PORT_TYPE_ORDER
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.feasibility import FeasibilityResult, Violation, ViolationSeverity
from amsrr.schemas.irg import IRGEdgeType, IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph, PortNode, RobotAnchor
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec


CHECKER_VERSION = "p2_agent_f_design_v1"
GRAVITY = 9.80665
DESIGN_HARD_CHECK_CODES = (
    codes.F_SCHEMA_VALID,
    codes.F_CONNECTED_GRAPH,
    codes.F_MODULE_COUNT,
    codes.F_PORT_OCCUPANCY,
    codes.F_COMPATIBLE_PORT_TYPES,
    codes.F_CLOSED_LOOP_REJECT_V1,
    codes.F_BASE_MODULE_ASSIGNED,
    codes.F_REQUIRED_SLOT_COVERAGE,
    codes.F_ROBOT_ANCHOR_CAPABILITY,
    codes.F_COARSE_REACHABILITY,
    codes.F_COARSE_COLLISION,
    codes.F_THRUST_MARGIN,
    codes.F_PAYLOAD_MARGIN,
    codes.F_QP_HOVER_FEASIBILITY,
)


@dataclass(frozen=True)
class _CapabilityRequirement:
    capability_type: str
    min_force_n: float | None = None
    min_torque_nm: float | None = None
    pose_accuracy_m: float | None = None
    pose_accuracy_rad: float | None = None
    stiffness_requirement: float | None = None


class FeasibilityChecker:
    """Deterministic hard-check scaffold for design-level feasibility."""

    def check_design(
        self,
        design_output: DesignOutput,
        *,
        task_spec: TaskSpec | None = None,
        irg: InteractionRequirementGraph | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> FeasibilityResult:
        violations: list[Violation] = []
        margins: dict[str, float] = {}
        proxy_scores: dict[str, float] = {}

        morphology = design_output.target_morphology
        self._schema_check(design_output, violations)
        self._base_module_check(morphology, violations, margins)
        self._module_count_check(morphology, task_spec, violations, margins)
        self._connected_graph_check(morphology, violations, margins)
        self._port_checks(morphology, violations, margins)
        self._closed_loop_check(morphology, task_spec, violations, margins)
        if irg is not None:
            self._slot_coverage_check(morphology, irg, violations, margins)
            self._coarse_reachability_check(morphology, irg, violations, margins)
        if task_spec is not None and physical_model is not None:
            self._thrust_and_payload_checks(morphology, task_spec, physical_model, violations, margins, proxy_scores)
        proxy_scores.setdefault("S_ASSEMBLY_COMPLEXITY", 1.0 / max(1, len(morphology.modules) + len(morphology.dock_edges)))
        proxy_scores.setdefault("S_COMPACTNESS", 1.0 / max(1, len(morphology.modules)))
        proxy_scores.setdefault("S_CONTACT_REGION_COVERAGE", margins.get("required_slot_coverage_ratio", 1.0))
        proxy_scores.setdefault("S_REACHABILITY_SCORE", margins.get("coarse_reachability_ratio", 1.0))
        proxy_scores.setdefault("S_GRASP_QUALITY_PROXY", margins.get("required_slot_anchor_coverage_ratio", 1.0))

        hard_violations = [violation for violation in violations if violation.severity == ViolationSeverity.HARD]
        _add_label_scores(proxy_scores, hard_violations)
        return FeasibilityResult(
            feasible=not hard_violations,
            hard_violations=hard_violations,
            soft_violations=[violation for violation in violations if violation.severity != ViolationSeverity.HARD],
            margins=margins,
            proxy_scores=proxy_scores,
            checker_version=CHECKER_VERSION,
            metadata={
                "level": "design",
                "module_count": len(morphology.modules),
                "dock_edge_count": len(morphology.dock_edges),
                "anchor_count": len(morphology.robot_anchors),
                "hard_violation_count": len(hard_violations),
                "soft_violation_count": sum(1 for violation in violations if violation.severity != ViolationSeverity.HARD),
            },
        )

    @staticmethod
    def _schema_check(design_output: DesignOutput, violations: list[Violation]) -> None:
        try:
            design_output.validate()
            design_output.target_morphology.validate()
        except SchemaValidationError as exc:
            violations.append(_hard(codes.F_SCHEMA_VALID, f"design-level schema validation failed: {exc}"))

    @staticmethod
    def _base_module_check(
        morphology: MorphologyGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        module_ids = {module.module_id for module in morphology.modules}
        base_count = sum(1 for module in morphology.modules if module.is_base)
        assigned = morphology.base_module_id in module_ids and base_count == 1
        margins["base_module_assigned"] = 1.0 if assigned else 0.0
        margins["base_module_marker_count"] = float(base_count)
        if not assigned:
            violations.append(
                _hard(
                    codes.F_BASE_MODULE_ASSIGNED,
                    "design-level base module must exist and exactly one ModuleNode must be marked base",
                    node_or_edge_ref=f"base:{morphology.base_module_id}",
                )
            )

    @staticmethod
    def _module_count_check(
        morphology: MorphologyGraph,
        task_spec: TaskSpec | None,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        count = len(morphology.modules)
        if task_spec is None:
            margins["module_count"] = float(count)
            return
        min_modules = task_spec.robot_constraints.min_modules
        max_modules = task_spec.robot_constraints.max_modules
        margins["module_count_min_margin"] = float(count - min_modules)
        margins["module_count_max_margin"] = float(max_modules - count)
        if count < min_modules or count > max_modules:
            violations.append(
                _hard(
                    codes.F_MODULE_COUNT,
                    f"design-level module_count={count} outside [{min_modules}, {max_modules}]",
                    margin=min(count - min_modules, max_modules - count),
                    threshold=float(min_modules),
                )
            )

    @staticmethod
    def _connected_graph_check(
        morphology: MorphologyGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        module_ids = {module.module_id for module in morphology.modules}
        margins["connected_component_module_count"] = 0.0
        margins["connected_module_coverage_ratio"] = 0.0
        if not module_ids:
            violations.append(_hard(codes.F_CONNECTED_GRAPH, "design-level morphology has no modules"))
            return
        adjacency: dict[int, set[int]] = {module_id: set() for module_id in module_ids}
        for edge in morphology.dock_edges:
            if edge.src_module_id not in module_ids or edge.dst_module_id not in module_ids:
                violations.append(
                    _hard(codes.F_CONNECTED_GRAPH, "design-level dock edge references unknown module", node_or_edge_ref=f"edge:{edge.edge_id}")
                )
                continue
            adjacency[edge.src_module_id].add(edge.dst_module_id)
            adjacency[edge.dst_module_id].add(edge.src_module_id)
        if morphology.base_module_id not in adjacency:
            violations.append(
                _hard(
                    codes.F_CONNECTED_GRAPH,
                    "design-level connected graph check cannot start from missing base module",
                    node_or_edge_ref=f"base:{morphology.base_module_id}",
                )
            )
            return
        visited = set()
        queue = deque([morphology.base_module_id])
        while queue:
            module_id = queue.popleft()
            if module_id in visited:
                continue
            visited.add(module_id)
            queue.extend(adjacency[module_id] - visited)
        margins["connected_component_module_count"] = float(len(visited))
        margins["connected_module_coverage_ratio"] = float(len(visited)) / float(len(module_ids))
        if visited != module_ids:
            violations.append(
                _hard(
                    codes.F_CONNECTED_GRAPH,
                    "design-level morphology dock graph is not connected",
                    margin=float(len(visited) - len(module_ids)),
                    threshold=0.0,
                )
            )

    @staticmethod
    def _port_checks(
        morphology: MorphologyGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        ports_by_id = {port.port_global_id: port for port in morphology.ports}
        edge_port_uses: dict[int, int] = defaultdict(int)
        unknown_port_ref_count = 0
        unoccupied_port_ref_count = 0
        incompatible_edge_count = 0
        valid_edge_count = 0
        for edge in morphology.dock_edges:
            src = ports_by_id.get(edge.src_port_id)
            dst = ports_by_id.get(edge.dst_port_id)
            if src is None or dst is None:
                unknown_port_ref_count += 1
                violations.append(_hard(codes.F_PORT_OCCUPANCY, "design-level dock edge references unknown port", node_or_edge_ref=f"edge:{edge.edge_id}"))
                continue
            edge_port_uses[src.port_global_id] += 1
            edge_port_uses[dst.port_global_id] += 1
            edge_valid = True
            if not src.occupied or not dst.occupied:
                unoccupied_port_ref_count += 1
                edge_valid = False
                violations.append(_hard(codes.F_PORT_OCCUPANCY, "design-level dock edge uses a port not marked occupied", node_or_edge_ref=f"edge:{edge.edge_id}"))
            if not _ports_compatible(src, dst):
                incompatible_edge_count += 1
                edge_valid = False
                violations.append(
                    _hard(codes.F_COMPATIBLE_PORT_TYPES, "design-level dock edge uses incompatible port types", node_or_edge_ref=f"edge:{edge.edge_id}")
                )
            if edge_valid:
                valid_edge_count += 1
        duplicate_port_use_conflict_count = 0
        for port_id, use_count in edge_port_uses.items():
            if use_count > 1:
                duplicate_port_use_conflict_count += use_count - 1
                violations.append(
                    _hard(
                        codes.F_PORT_OCCUPANCY,
                        f"design-level port {port_id} is used by multiple dock edges",
                        node_or_edge_ref=f"port:{port_id}",
                        margin=float(1 - use_count),
                        threshold=1.0,
                    )
                )
        port_conflict_count = (
            unknown_port_ref_count
            + unoccupied_port_ref_count
            + incompatible_edge_count
            + duplicate_port_use_conflict_count
        )
        margins["port_unknown_ref_count"] = float(unknown_port_ref_count)
        margins["port_unoccupied_ref_count"] = float(unoccupied_port_ref_count)
        margins["port_incompatible_edge_count"] = float(incompatible_edge_count)
        margins["port_duplicate_use_conflict_count"] = float(duplicate_port_use_conflict_count)
        margins["port_conflict_count"] = float(port_conflict_count)
        margins["port_valid_edge_ratio"] = float(valid_edge_count) / max(1.0, float(len(morphology.dock_edges)))

    @staticmethod
    def _closed_loop_check(
        morphology: MorphologyGraph,
        task_spec: TaskSpec | None,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        allow_closed_loop = task_spec.robot_constraints.allow_closed_loop if task_spec is not None else False
        edge_cycle_flag = len(morphology.modules) > 0 and len(morphology.dock_edges) >= len(morphology.modules)
        closed_loop_flag = morphology.is_closed_loop or edge_cycle_flag
        margins["closed_loop_declared"] = 1.0 if morphology.is_closed_loop else 0.0
        margins["closed_loop_edge_cycle_flag"] = 1.0 if edge_cycle_flag else 0.0
        margins["closed_loop_rejected"] = 1.0 if closed_loop_flag and not allow_closed_loop else 0.0
        margins["tree_edge_count_margin"] = float((len(morphology.modules) - 1) - len(morphology.dock_edges))
        if morphology.is_closed_loop and not allow_closed_loop:
            violations.append(_hard(codes.F_CLOSED_LOOP_REJECT_V1, "design-level closed-loop morphology rejected in v1"))
        if edge_cycle_flag and not allow_closed_loop:
            violations.append(_hard(codes.F_CLOSED_LOOP_REJECT_V1, "design-level dock graph appears to contain a closed loop"))

    def _slot_coverage_check(
        self,
        morphology: MorphologyGraph,
        irg: InteractionRequirementGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        capability_requirements_by_slot = _capability_requirements_by_slot(irg)
        required_slots = [_slot_info(node) for node in irg.nodes if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature.get("required", True)]
        anchors_by_slot: dict[int, list[RobotAnchor]] = defaultdict(list)
        for anchor in morphology.robot_anchors:
            for slot_id in anchor.associated_contact_slot_ids:
                anchors_by_slot[int(slot_id)].append(anchor)
        covered_slots = 0
        capability_covered_slots = 0
        required_anchor_count = 0
        type_covered_anchor_count = 0
        capability_covered_anchor_count = 0
        capability_checked_anchor_count = 0
        capability_valid_anchor_count = 0
        for slot in required_slots:
            anchors = anchors_by_slot.get(slot["slot_id"], [])
            type_compatible = [anchor for anchor in anchors if anchor.anchor_type == slot["anchor_type"]]
            required_anchor_count += slot["min_count"]
            type_covered_anchor_count += min(len(type_compatible), slot["min_count"])
            capability_requirements = capability_requirements_by_slot.get(slot["slot_id"], [])
            capability_valid: list[RobotAnchor] = []
            for anchor in type_compatible:
                capability_checked_anchor_count += 1
                violations_before = len(violations)
                self._anchor_capability_check(anchor, slot, capability_requirements, violations)
                if len(violations) == violations_before:
                    capability_valid.append(anchor)
                    capability_valid_anchor_count += 1
            capability_covered_anchor_count += min(len(capability_valid), slot["min_count"])
            if len(type_compatible) < slot["min_count"]:
                violations.append(
                    _hard(
                        codes.F_REQUIRED_SLOT_COVERAGE,
                        f"design-level ContactSlot {slot['slot_id']} needs {slot['min_count']} compatible anchors, got {len(type_compatible)}",
                        node_or_edge_ref=f"slot:{slot['slot_id']}",
                        margin=float(len(type_compatible) - slot["min_count"]),
                        threshold=float(slot["min_count"]),
                    )
                )
                continue
            covered_slots += 1
            if len(capability_valid) >= slot["min_count"]:
                capability_covered_slots += 1
            else:
                violations.append(
                    _hard(
                        codes.F_ROBOT_ANCHOR_CAPABILITY,
                        f"design-level ContactSlot {slot['slot_id']} needs {slot['min_count']} capability-valid anchors, got {len(capability_valid)}",
                        node_or_edge_ref=f"slot:{slot['slot_id']}",
                        margin=float(len(capability_valid) - slot["min_count"]),
                        threshold=float(slot["min_count"]),
                    )
                )
        margins["required_slot_count"] = float(len(required_slots))
        margins["required_slot_covered_count"] = float(covered_slots)
        margins["required_slot_capability_covered_count"] = float(capability_covered_slots)
        margins["required_slot_coverage_ratio"] = covered_slots / max(1, len(required_slots))
        margins["required_slot_capability_coverage_ratio"] = capability_covered_slots / max(1, len(required_slots))
        margins["required_slot_anchor_required_count"] = float(required_anchor_count)
        margins["required_slot_anchor_covered_count"] = float(type_covered_anchor_count)
        margins["required_slot_anchor_coverage_ratio"] = type_covered_anchor_count / max(1, required_anchor_count)
        margins["required_slot_anchor_capability_covered_count"] = float(capability_covered_anchor_count)
        margins["required_slot_anchor_capability_coverage_ratio"] = capability_covered_anchor_count / max(1, required_anchor_count)
        margins["anchor_capability_checked_count"] = float(capability_checked_anchor_count)
        margins["anchor_capability_valid_count"] = float(capability_valid_anchor_count)
        margins["anchor_capability_valid_ratio"] = capability_valid_anchor_count / max(1, capability_checked_anchor_count)

    @staticmethod
    def _anchor_capability_check(
        anchor: RobotAnchor,
        slot: dict[str, Any],
        capability_requirements: list[_CapabilityRequirement],
        violations: list[Violation],
    ) -> None:
        required_capability = slot["required_anchor_capability"].get("capability_type")
        if required_capability and required_capability != anchor.anchor_type:
            violations.append(
                _hard(
                    codes.F_ROBOT_ANCHOR_CAPABILITY,
                    f"design-level anchor {anchor.anchor_id} capability {anchor.anchor_type!r} does not match required {required_capability!r}",
                    node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                )
            )
        slot_min_force = _optional_float(slot["required_anchor_capability"].get("min_force_n"))
        slot_min_torque = _optional_float(slot["required_anchor_capability"].get("min_torque_nm"))
        requirements = list(capability_requirements)
        if slot_min_force is not None or slot_min_torque is not None:
            requirements.append(
                _CapabilityRequirement(
                    capability_type=required_capability or anchor.anchor_type,
                    min_force_n=slot_min_force,
                    min_torque_nm=slot_min_torque,
                )
            )
        for requirement in requirements:
            if requirement.capability_type and requirement.capability_type != anchor.anchor_type:
                violations.append(
                    _hard(
                        codes.F_ROBOT_ANCHOR_CAPABILITY,
                        f"design-level anchor {anchor.anchor_id} capability {anchor.anchor_type!r} does not match required {requirement.capability_type!r}",
                        node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                    )
                )
            _check_minimum_capability(
                anchor,
                value_key="max_force_n",
                minimum=requirement.min_force_n,
                unit_label="force",
                violations=violations,
            )
            _check_minimum_capability(
                anchor,
                value_key="max_torque_nm",
                minimum=requirement.min_torque_nm,
                unit_label="torque",
                violations=violations,
            )

    @staticmethod
    def _coarse_reachability_check(
        morphology: MorphologyGraph,
        irg: InteractionRequirementGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        slots = [_slot_info(node) for node in irg.nodes if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature.get("required", True)]
        anchors_by_slot: dict[int, list[RobotAnchor]] = defaultdict(list)
        module_ids = {module.module_id for module in morphology.modules}
        for anchor in morphology.robot_anchors:
            for slot_id in anchor.associated_contact_slot_ids:
                anchors_by_slot[int(slot_id)].append(anchor)
        reachable_slot_count = 0
        checked_anchor_count = 0
        reachable_anchor_count = 0
        for slot in slots:
            slot_reachable = bool(slot["allowed_region_ids"])
            if not slot["allowed_region_ids"]:
                slot_reachable = False
                violations.append(
                    _hard(
                        codes.F_COARSE_REACHABILITY,
                        f"design-level ContactSlot {slot['slot_id']} has no allowed regions for coarse reachability",
                        node_or_edge_ref=f"slot:{slot['slot_id']}",
                    )
                )
            for anchor in anchors_by_slot.get(slot["slot_id"], []):
                checked_anchor_count += 1
                if anchor.module_id not in module_ids or len(anchor.local_pose) != 7:
                    slot_reachable = False
                    violations.append(
                        _hard(
                            codes.F_COARSE_REACHABILITY,
                            f"design-level anchor {anchor.anchor_id} cannot be resolved to a module/local pose",
                            node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                        )
                    )
                else:
                    reachable_anchor_count += 1
            if slot_reachable:
                reachable_slot_count += 1
        margins["coarse_reachability_required_slot_count"] = float(len(slots))
        margins["coarse_reachability_slot_valid_count"] = float(reachable_slot_count)
        margins["coarse_reachability_ratio"] = reachable_slot_count / max(1, len(slots))
        margins["coarse_reachability_checked_anchor_count"] = float(checked_anchor_count)
        margins["coarse_reachability_valid_anchor_count"] = float(reachable_anchor_count)
        margins["coarse_reachability_anchor_ratio"] = reachable_anchor_count / max(1, checked_anchor_count)

    @staticmethod
    def _thrust_and_payload_checks(
        morphology: MorphologyGraph,
        task_spec: TaskSpec,
        physical_model: PhysicalModel,
        violations: list[Violation],
        margins: dict[str, float],
        proxy_scores: dict[str, float],
    ) -> None:
        module_count = len(morphology.modules)
        payload_mass = sum(obj.mass_kg or 0.0 for obj in task_spec.scene.objects if obj.movable)
        robot_mass = module_count * physical_model.aggregate_mass_kg
        required_force = (robot_mass + payload_mass) * abs(task_spec.scene.environment.gravity[2])
        available_force_per_module = sum(abs(rotor.thrust_axis_local[2]) * rotor.thrust_max_n for rotor in physical_model.rotors)
        available_force = module_count * available_force_per_module
        thrust_margin = (available_force - required_force) / max(required_force, 1.0e-9)
        margins["robot_mass_kg"] = robot_mass
        margins["payload_mass_kg"] = payload_mass
        margins["total_mass_kg"] = robot_mass + payload_mass
        margins["available_vertical_force_per_module_n"] = available_force_per_module
        margins["available_total_vertical_force_n"] = available_force
        margins["required_total_vertical_force_n"] = required_force
        margins["thrust_margin_ratio"] = thrust_margin
        margins["thrust_margin_threshold"] = task_spec.safety.min_thrust_margin_ratio
        margins["thrust_margin_absolute_n"] = available_force - required_force
        proxy_scores["S_WRENCH_MARGIN"] = thrust_margin
        proxy_scores["S_ENERGY_PROXY"] = required_force / max(available_force, 1.0e-9)
        if thrust_margin < task_spec.safety.min_thrust_margin_ratio:
            violations.append(
                _hard(
                    codes.F_THRUST_MARGIN,
                    "design-level hover thrust margin is below safety minimum",
                    margin=thrust_margin,
                    threshold=task_spec.safety.min_thrust_margin_ratio,
                )
            )
        if thrust_margin < 0.0:
            violations.append(
                _hard(
                    codes.F_QP_HOVER_FEASIBILITY,
                    "design-level coarse hover QP proxy is infeasible because thrust is below weight",
                    margin=thrust_margin,
                    threshold=0.0,
                )
            )
        required_payload_force = payload_mass * GRAVITY
        payload_anchor_force = sum(
            float(anchor.capability.get("max_force_n", 0.0))
            for anchor in morphology.robot_anchors
            if anchor.anchor_type in {"grasp", "support"}
        )
        payload_margin = payload_anchor_force - required_payload_force
        margins["payload_required_force_n"] = required_payload_force
        margins["payload_anchor_force_n"] = payload_anchor_force
        margins["payload_force_margin_n"] = payload_margin
        margins["payload_margin_ratio"] = payload_margin / max(required_payload_force, 1.0e-9)
        if required_payload_force > 0.0 and payload_margin < 0.0:
            violations.append(
                _hard(
                    codes.F_PAYLOAD_MARGIN,
                    "design-level payload contact force margin is negative",
                    margin=payload_margin,
                    threshold=0.0,
                )
            )


def _hard(
    code: str,
    message: str,
    *,
    node_or_edge_ref: str | None = None,
    margin: float | None = None,
    threshold: float | None = None,
) -> Violation:
    return Violation(
        code=code,
        severity=ViolationSeverity.HARD,
        message=message,
        node_or_edge_ref=node_or_edge_ref,
        margin=margin,
        threshold=threshold,
    )


def _ports_compatible(src: PortNode, dst: PortNode) -> bool:
    try:
        dst_idx = PORT_TYPE_ORDER.index(dst.port_type)
        src_idx = PORT_TYPE_ORDER.index(src.port_type)
    except ValueError:
        return False
    return bool(src.compatible_port_type_mask[dst_idx]) and bool(dst.compatible_port_type_mask[src_idx])


def _capability_requirements_by_slot(irg: InteractionRequirementGraph) -> dict[int, list[_CapabilityRequirement]]:
    nodes_by_id = {node.node_id: node for node in irg.nodes}
    slot_requirements: dict[int, list[_CapabilityRequirement]] = defaultdict(list)
    for edge in irg.edges:
        if edge.edge_type != IRGEdgeType.APPLIES_TO:
            continue
        src = nodes_by_id.get(edge.src_id)
        dst = nodes_by_id.get(edge.dst_id)
        if src is None or dst is None:
            continue
        if src.node_type != IRGNodeType.CAPABILITY_REQUIREMENT or dst.node_type != IRGNodeType.CONTACT_SLOT:
            continue
        slot_id = int(dst.feature.get("slot_id", dst.node_id))
        slot_requirements[slot_id].append(_capability_requirement_from_node(src))
    return slot_requirements


def _capability_requirement_from_node(node: IRGNode) -> _CapabilityRequirement:
    feature = node.feature
    return _CapabilityRequirement(
        capability_type=str(feature.get("capability_type", node.ref_id or "")),
        min_force_n=_optional_float(feature.get("min_force_n")),
        min_torque_nm=_optional_float(feature.get("min_torque_nm")),
        pose_accuracy_m=_optional_float(feature.get("pose_accuracy_m")),
        pose_accuracy_rad=_optional_float(feature.get("pose_accuracy_rad")),
        stiffness_requirement=_optional_float(feature.get("stiffness_requirement")),
    )


def _check_minimum_capability(
    anchor: RobotAnchor,
    *,
    value_key: str,
    minimum: float | None,
    unit_label: str,
    violations: list[Violation],
) -> None:
    if minimum is None:
        return
    available = _optional_float(anchor.capability.get(value_key))
    if available is None:
        violations.append(
            _hard(
                codes.F_ROBOT_ANCHOR_CAPABILITY,
                f"design-level anchor {anchor.anchor_id} has no declared max {unit_label} capability",
                node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                margin=None,
                threshold=minimum,
            )
        )
        return
    if available < minimum:
        violations.append(
            _hard(
                codes.F_ROBOT_ANCHOR_CAPABILITY,
                f"design-level anchor {anchor.anchor_id} max {unit_label} is below requirement",
                node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                margin=available - minimum,
                threshold=minimum,
            )
        )


def _add_label_scores(proxy_scores: dict[str, float], hard_violations: list[Violation]) -> None:
    violated_codes = {violation.code for violation in hard_violations}
    proxy_scores["L_FEASIBLE"] = 0.0 if hard_violations else 1.0
    proxy_scores["L_HARD_VIOLATION"] = 1.0 if hard_violations else 0.0
    for code in DESIGN_HARD_CHECK_CODES:
        proxy_scores[f"L_{code}"] = 1.0 if code in violated_codes else 0.0


def _slot_info(node: IRGNode) -> dict[str, Any]:
    raw_mode = node.feature.get("contact_mode")
    mode = ContactMode(raw_mode)
    anchor_type = {
        ContactMode.GRASP: "grasp",
        ContactMode.SUPPORT: "support",
        ContactMode.PUSH: "push",
        ContactMode.LATCH: "latch",
        ContactMode.PERCH: "perch",
        ContactMode.TOOL: "tool",
        ContactMode.BODY_CONTACT: "body_contact",
    }.get(mode)
    if anchor_type is None:
        raise SchemaValidationError(f"ContactSlot {node.node_id} contact_mode {mode.value!r} is unsupported by P0 anchors")
    return {
        "slot_id": int(node.feature.get("slot_id", node.node_id)),
        "contact_mode": mode,
        "anchor_type": anchor_type,
        "required": bool(node.feature.get("required", True)),
        "min_count": int(node.feature.get("min_count_group", 1)),
        "allowed_region_ids": list(node.feature.get("allowed_region_ids", [])),
        "required_anchor_capability": dict(node.feature.get("required_anchor_capability", {}) or {}),
    }


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
