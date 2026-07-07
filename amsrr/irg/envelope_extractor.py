from __future__ import annotations

from collections import defaultdict
from typing import Any

from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.interaction_envelope import (
    CapabilityRequirement,
    DurationRequirement,
    EnvelopeBranchOption,
    InteractionEnvelope,
    PrecisionRequirement,
    SupportRatioRequirement,
    TargetRegionSet,
    WrenchSpaceRequirement,
)
from amsrr.schemas.irg import IRGEdgeType, IRGNode, IRGNodeType, InteractionRequirementGraph


class InteractionEnvelopeExtractor:
    """Deterministically aggregate compact requirements from a valid IRG."""

    def extract(self, irg: InteractionRequirementGraph) -> InteractionEnvelope:
        nodes_by_id = {node.node_id: node for node in irg.nodes}
        nodes_by_type: dict[IRGNodeType, list[IRGNode]] = defaultdict(list)
        for node in irg.nodes:
            nodes_by_type[node.node_type].append(node)

        contact_slots = sorted(nodes_by_type[IRGNodeType.CONTACT_SLOT], key=lambda node: node.node_id)
        contact_regions = sorted(nodes_by_type[IRGNodeType.CONTACT_REGION], key=lambda node: node.node_id)
        wrench_nodes = sorted(nodes_by_type[IRGNodeType.WRENCH_REQUIREMENT], key=lambda node: node.node_id)
        state_nodes = sorted(nodes_by_type[IRGNodeType.STATE_TARGET], key=lambda node: node.node_id)
        constraint_nodes = sorted(nodes_by_type[IRGNodeType.CONSTRAINT], key=lambda node: node.node_id)
        capability_nodes = sorted(nodes_by_type[IRGNodeType.CAPABILITY_REQUIREMENT], key=lambda node: node.node_id)
        phase_nodes = sorted(nodes_by_type[IRGNodeType.PHASE], key=lambda node: node.node_id)

        return InteractionEnvelope(
            envelope_id=f"envelope:{irg.task_id}:{irg.stable_hash()[:12]}",
            task_id=irg.task_id,
            required_contact_count_range=self._required_contact_count_range(contact_slots),
            required_contact_modes=self._contact_modes(contact_slots),
            target_region_sets=self._target_region_sets(contact_slots, contact_regions),
            wrench_space_requirements=[self._wrench_requirement(node) for node in wrench_nodes],
            support_ratio_requirements=self._support_ratio_requirement(constraint_nodes, wrench_nodes),
            vertical_thrust_ratio_limit=self._vertical_thrust_ratio_limit(constraint_nodes),
            precision_requirements=[item for node in state_nodes if (item := self._precision_requirement(node)) is not None],
            duration_requirements=self._duration_requirements(nodes_by_type[IRGNodeType.TASK], phase_nodes),
            capability_requirements=[self._capability_requirement(node) for node in capability_nodes],
            branch_options=self._branch_options(irg, nodes_by_id),
        )

    @staticmethod
    def _required_contact_count_range(contact_slots: list[IRGNode]) -> tuple[int, int]:
        required_slots = [node for node in contact_slots if bool(node.feature.get("required", True))]
        if not required_slots:
            return (0, 0)
        min_count = sum(int(node.feature.get("min_count_group", 0)) for node in required_slots)
        max_count = sum(int(node.feature.get("max_count_group", 0)) for node in required_slots)
        return (min_count, max_count)

    @staticmethod
    def _contact_modes(contact_slots: list[IRGNode]) -> list[ContactMode]:
        modes: list[ContactMode] = []
        seen: set[ContactMode] = set()
        for node in contact_slots:
            raw_mode = node.feature.get("contact_mode")
            if raw_mode is None:
                continue
            try:
                mode = ContactMode(raw_mode)
            except ValueError as exc:
                raise SchemaValidationError(f"ContactSlot {node.node_id} has invalid contact_mode {raw_mode!r}") from exc
            if mode not in seen:
                modes.append(mode)
                seen.add(mode)
        return modes

    @staticmethod
    def _target_region_sets(contact_slots: list[IRGNode], contact_regions: list[IRGNode]) -> list[TargetRegionSet]:
        region_by_id: dict[str, IRGNode] = {}
        for node in contact_regions:
            region_id = str(node.feature.get("region_id", node.ref_id))
            region_by_id[region_id] = node

        region_ids_by_entity: dict[str, list[str]] = defaultdict(list)
        region_types_by_entity: dict[str, list[str]] = defaultdict(list)
        for slot in contact_slots:
            for region_id in slot.feature.get("allowed_region_ids", []):
                region = region_by_id.get(str(region_id))
                if region is None:
                    continue
                entity_id = str(region.feature.get("target_entity_id", region.feature.get("entity_id", "")))
                if not entity_id:
                    continue
                compiled_region_id = str(region.feature.get("region_id", region.ref_id))
                region_type = str(region.feature.get("region_type", ""))
                if compiled_region_id not in region_ids_by_entity[entity_id]:
                    region_ids_by_entity[entity_id].append(compiled_region_id)
                if region_type and region_type not in region_types_by_entity[entity_id]:
                    region_types_by_entity[entity_id].append(region_type)

        return [
            TargetRegionSet(
                entity_id=entity_id,
                region_ids=region_ids_by_entity[entity_id],
                region_types=region_types_by_entity[entity_id],
            )
            for entity_id in sorted(region_ids_by_entity)
        ]

    @staticmethod
    def _wrench_requirement(node: IRGNode) -> WrenchSpaceRequirement:
        feature = node.feature
        return WrenchSpaceRequirement(
            applies_to=str(feature.get("applies_to", node.ref_id or "unknown")),
            effect=str(feature.get("required_effect", node.ref_id or "unknown")),
            lower_bound_description=feature.get("lower_bound_description"),
            wrench_lower=feature.get("wrench_lower"),
            wrench_upper=feature.get("wrench_upper"),
            target_wrench=feature.get("target_wrench"),
            priority=float(node.priority),
            metadata={
                "source_node_id": node.node_id,
                "source_ref_id": node.ref_id,
                "frame": feature.get("frame"),
                "hard_or_soft": feature.get("hard_or_soft"),
                "slack_weight": feature.get("slack_weight"),
            },
        )

    @staticmethod
    def _support_ratio_requirement(
        constraint_nodes: list[IRGNode],
        wrench_nodes: list[IRGNode],
    ) -> SupportRatioRequirement | None:
        min_contact_support_ratio: float | None = None
        max_vertical_thrust_ratio: float | None = None
        found = False
        for node in constraint_nodes:
            constraint_type = node.feature.get("constraint_type")
            params = node.feature.get("parameters", {}) or {}
            if constraint_type == "support_ratio":
                found = True
                min_contact_support_ratio = _first_float(params, "min_contact_support_ratio", "min_ratio", "support_ratio")
            if constraint_type == "vertical_thrust_ratio":
                found = True
                max_vertical_thrust_ratio = _first_float(params, "max_vertical_thrust_ratio", "max_ratio", "limit")
        for node in wrench_nodes:
            effect = str(node.feature.get("required_effect", ""))
            if "support_ratio" in effect or "vertical_thrust_ratio" in effect:
                found = True
        if not found:
            return None
        return SupportRatioRequirement(
            min_contact_support_ratio=min_contact_support_ratio,
            max_vertical_thrust_ratio=max_vertical_thrust_ratio,
            allow_thrust_for_stabilization=True,
        )

    @staticmethod
    def _vertical_thrust_ratio_limit(constraint_nodes: list[IRGNode]) -> float | None:
        for node in constraint_nodes:
            if node.feature.get("constraint_type") != "vertical_thrust_ratio":
                continue
            params = node.feature.get("parameters", {}) or {}
            value = _first_float(params, "max_vertical_thrust_ratio", "max_ratio", "limit")
            if value is not None:
                return value
        return None

    @staticmethod
    def _precision_requirement(node: IRGNode) -> PrecisionRequirement | None:
        tolerance = node.feature.get("tolerance", {}) or {}
        pos = _first_float(tolerance, "pos_m", "tolerance_pos_m")
        rot = _first_float(tolerance, "rot_rad", "tolerance_rot_rad")
        q_raw = tolerance.get("q", tolerance.get("tolerance_q"))
        q = [float(item) for item in q_raw] if isinstance(q_raw, list) else None
        if pos is None and rot is None and q is None:
            return None
        return PrecisionRequirement(
            target=str(node.feature.get("target_type", node.ref_id or "state_target")),
            tolerance_pos_m=pos,
            tolerance_rot_rad=rot,
            tolerance_q=q,
        )

    @staticmethod
    def _duration_requirements(task_nodes: list[IRGNode], phase_nodes: list[IRGNode]) -> list[DurationRequirement]:
        durations: list[DurationRequirement] = []
        for node in sorted(task_nodes, key=lambda item: item.node_id):
            max_duration = _first_float(node.feature, "time_limit_s")
            if max_duration is not None:
                durations.append(DurationRequirement(phase_label=None, max_duration_s=max_duration))
        for node in phase_nodes:
            min_duration = _first_float(node.feature, "min_duration_s")
            max_duration = _first_float(node.feature, "max_duration_s", "nominal_duration_s")
            if min_duration is not None or max_duration is not None:
                durations.append(
                    DurationRequirement(
                        phase_label=str(node.feature.get("phase_label", node.ref_id or "")),
                        min_duration_s=min_duration,
                        max_duration_s=max_duration,
                    )
                )
        return durations

    @staticmethod
    def _capability_requirement(node: IRGNode) -> CapabilityRequirement:
        feature = node.feature
        return CapabilityRequirement(
            capability_type=str(feature.get("capability_type", node.ref_id or "unknown")),
            min_force_n=_optional_float(feature.get("min_force_n")),
            min_torque_nm=_optional_float(feature.get("min_torque_nm")),
            pose_accuracy_m=_optional_float(feature.get("pose_accuracy_m")),
            pose_accuracy_rad=_optional_float(feature.get("pose_accuracy_rad")),
            stiffness_requirement=_optional_float(feature.get("stiffness_requirement")),
        )

    @staticmethod
    def _branch_options(
        irg: InteractionRequirementGraph,
        nodes_by_id: dict[int, IRGNode],
    ) -> list[EnvelopeBranchOption]:
        branches: list[EnvelopeBranchOption] = []
        branch_edges = {
            IRGEdgeType.MUTUALLY_EXCLUSIVE,
            IRGEdgeType.FALLBACK,
        }
        for edge in irg.edges:
            if edge.edge_type not in branch_edges:
                continue
            src = nodes_by_id.get(edge.src_id)
            dst = nodes_by_id.get(edge.dst_id)
            if src is None or dst is None:
                continue
            branch_id = f"{edge.edge_type.value}:{edge.src_id}:{edge.dst_id}"
            modes = []
            for node in (src, dst):
                raw_mode = node.feature.get("contact_mode")
                if raw_mode is not None:
                    modes.append(ContactMode(raw_mode))
            branches.append(
                EnvelopeBranchOption(
                    branch_id=branch_id,
                    label=f"{src.ref_id or src.node_id}->{dst.ref_id or dst.node_id}",
                    required_contact_modes=modes,
                    metadata={"edge_type": edge.edge_type.value},
                )
            )
        return branches


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _first_float(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None
