from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from amsrr.feasibility import violation_codes as codes
from amsrr.morphology.graph import PORT_TYPE_ORDER
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.feasibility import FeasibilityResult, Violation, ViolationSeverity
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph, PortNode, RobotAnchor
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec


CHECKER_VERSION = "p0_agent_ef_v1"
GRAVITY = 9.80665


class FeasibilityChecker:
    """P0 deterministic hard-check scaffold for design-level feasibility."""

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
        self._base_module_check(morphology, violations)
        self._module_count_check(morphology, task_spec, violations, margins)
        self._connected_graph_check(morphology, violations)
        self._port_checks(morphology, violations)
        self._closed_loop_check(morphology, task_spec, violations)
        if irg is not None:
            self._slot_coverage_check(morphology, irg, violations, margins)
            self._coarse_reachability_check(morphology, irg, violations)
        if task_spec is not None and physical_model is not None:
            self._thrust_and_payload_checks(morphology, task_spec, physical_model, violations, margins, proxy_scores)
        proxy_scores.setdefault("S_ASSEMBLY_COMPLEXITY", 1.0 / max(1, len(morphology.modules) + len(morphology.dock_edges)))
        proxy_scores.setdefault("S_COMPACTNESS", 1.0 / max(1, len(morphology.modules)))
        proxy_scores.setdefault("S_CONTACT_REGION_COVERAGE", margins.get("required_slot_coverage_ratio", 1.0))

        hard_violations = [violation for violation in violations if violation.severity == ViolationSeverity.HARD]
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
    def _base_module_check(morphology: MorphologyGraph, violations: list[Violation]) -> None:
        module_ids = {module.module_id for module in morphology.modules}
        base_count = sum(1 for module in morphology.modules if module.is_base)
        if morphology.base_module_id not in module_ids or base_count != 1:
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
    def _connected_graph_check(morphology: MorphologyGraph, violations: list[Violation]) -> None:
        module_ids = {module.module_id for module in morphology.modules}
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
        visited = set()
        queue = deque([morphology.base_module_id])
        while queue:
            module_id = queue.popleft()
            if module_id in visited:
                continue
            visited.add(module_id)
            queue.extend(adjacency[module_id] - visited)
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
    def _port_checks(morphology: MorphologyGraph, violations: list[Violation]) -> None:
        ports_by_id = {port.port_global_id: port for port in morphology.ports}
        edge_port_uses: dict[int, int] = defaultdict(int)
        for edge in morphology.dock_edges:
            src = ports_by_id.get(edge.src_port_id)
            dst = ports_by_id.get(edge.dst_port_id)
            if src is None or dst is None:
                violations.append(_hard(codes.F_PORT_OCCUPANCY, "design-level dock edge references unknown port", node_or_edge_ref=f"edge:{edge.edge_id}"))
                continue
            edge_port_uses[src.port_global_id] += 1
            edge_port_uses[dst.port_global_id] += 1
            if not src.occupied or not dst.occupied:
                violations.append(_hard(codes.F_PORT_OCCUPANCY, "design-level dock edge uses a port not marked occupied", node_or_edge_ref=f"edge:{edge.edge_id}"))
            if not _ports_compatible(src, dst):
                violations.append(
                    _hard(codes.F_COMPATIBLE_PORT_TYPES, "design-level dock edge uses incompatible port types", node_or_edge_ref=f"edge:{edge.edge_id}")
                )
        for port_id, use_count in edge_port_uses.items():
            if use_count > 1:
                violations.append(
                    _hard(
                        codes.F_PORT_OCCUPANCY,
                        f"design-level port {port_id} is used by multiple dock edges",
                        node_or_edge_ref=f"port:{port_id}",
                        margin=float(1 - use_count),
                        threshold=1.0,
                    )
                )

    @staticmethod
    def _closed_loop_check(
        morphology: MorphologyGraph,
        task_spec: TaskSpec | None,
        violations: list[Violation],
    ) -> None:
        allow_closed_loop = task_spec.robot_constraints.allow_closed_loop if task_spec is not None else False
        if morphology.is_closed_loop and not allow_closed_loop:
            violations.append(_hard(codes.F_CLOSED_LOOP_REJECT_V1, "design-level closed-loop morphology rejected in v1"))
        if len(morphology.modules) > 0 and len(morphology.dock_edges) >= len(morphology.modules) and not allow_closed_loop:
            violations.append(_hard(codes.F_CLOSED_LOOP_REJECT_V1, "design-level dock graph appears to contain a closed loop"))

    @staticmethod
    def _slot_coverage_check(
        morphology: MorphologyGraph,
        irg: InteractionRequirementGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        required_slots = [_slot_info(node) for node in irg.nodes if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature.get("required", True)]
        anchors_by_slot: dict[int, list[RobotAnchor]] = defaultdict(list)
        for anchor in morphology.robot_anchors:
            for slot_id in anchor.associated_contact_slot_ids:
                anchors_by_slot[int(slot_id)].append(anchor)
        covered = 0
        for slot in required_slots:
            anchors = anchors_by_slot.get(slot["slot_id"], [])
            compatible = [anchor for anchor in anchors if anchor.anchor_type == slot["anchor_type"]]
            if len(compatible) < slot["min_count"]:
                violations.append(
                    _hard(
                        codes.F_REQUIRED_SLOT_COVERAGE,
                        f"design-level ContactSlot {slot['slot_id']} needs {slot['min_count']} compatible anchors, got {len(compatible)}",
                        node_or_edge_ref=f"slot:{slot['slot_id']}",
                        margin=float(len(compatible) - slot["min_count"]),
                        threshold=float(slot["min_count"]),
                    )
                )
                continue
            covered += 1
            for anchor in compatible:
                required_capability = slot["required_anchor_capability"].get("capability_type")
                if required_capability and required_capability != anchor.anchor_type:
                    violations.append(
                        _hard(
                            codes.F_ROBOT_ANCHOR_CAPABILITY,
                            f"design-level anchor {anchor.anchor_id} capability {anchor.anchor_type!r} does not match required {required_capability!r}",
                            node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                        )
                    )
                min_force = _optional_float(slot["required_anchor_capability"].get("min_force_n"))
                max_force = _optional_float(anchor.capability.get("max_force_n"))
                if min_force is not None and max_force is not None and max_force < min_force:
                    violations.append(
                        _hard(
                            codes.F_ROBOT_ANCHOR_CAPABILITY,
                            f"design-level anchor {anchor.anchor_id} max force is below requirement",
                            node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                            margin=max_force - min_force,
                            threshold=min_force,
                        )
                    )
        margins["required_slot_coverage_ratio"] = covered / max(1, len(required_slots))

    @staticmethod
    def _coarse_reachability_check(
        morphology: MorphologyGraph,
        irg: InteractionRequirementGraph,
        violations: list[Violation],
    ) -> None:
        slots = [_slot_info(node) for node in irg.nodes if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature.get("required", True)]
        anchors_by_slot: dict[int, list[RobotAnchor]] = defaultdict(list)
        module_ids = {module.module_id for module in morphology.modules}
        for anchor in morphology.robot_anchors:
            for slot_id in anchor.associated_contact_slot_ids:
                anchors_by_slot[int(slot_id)].append(anchor)
        for slot in slots:
            if not slot["allowed_region_ids"]:
                violations.append(
                    _hard(
                        codes.F_COARSE_REACHABILITY,
                        f"design-level ContactSlot {slot['slot_id']} has no allowed regions for coarse reachability",
                        node_or_edge_ref=f"slot:{slot['slot_id']}",
                    )
                )
            for anchor in anchors_by_slot.get(slot["slot_id"], []):
                if anchor.module_id not in module_ids or len(anchor.local_pose) != 7:
                    violations.append(
                        _hard(
                            codes.F_COARSE_REACHABILITY,
                            f"design-level anchor {anchor.anchor_id} cannot be resolved to a module/local pose",
                            node_or_edge_ref=f"anchor:{anchor.anchor_id}",
                        )
                    )

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
        margins["thrust_margin_ratio"] = thrust_margin
        proxy_scores["S_WRENCH_MARGIN"] = thrust_margin
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
        margins["payload_force_margin_n"] = payload_margin
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
