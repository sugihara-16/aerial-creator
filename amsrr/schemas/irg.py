from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import Condition, SchemaBase, SchemaValidationError, StrEnum, require_non_empty


class IRGNodeType(StrEnum):
    TASK = "task"
    PHASE = "phase"
    CONTACT_REGION = "contact_region"
    CONTACT_SLOT = "contact_slot"
    WRENCH_REQUIREMENT = "wrench_requirement"
    STATE_TARGET = "state_target"
    CONSTRAINT = "constraint"
    CAPABILITY_REQUIREMENT = "capability_requirement"


class IRGEdgeType(StrEnum):
    TEMPORAL_NEXT = "temporal_next"
    CONTAINS = "contains"
    ACTIVATES = "activates"
    REQUIRES = "requires"
    SUPPORTS = "supports"
    CONSTRAINS = "constrains"
    APPLIES_TO = "applies_to"
    ALLOWS = "allows"
    SIMULTANEOUS = "simultaneous"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    SPATIAL_RELATION = "spatial_relation"
    WRENCH_COMPOSITION = "wrench_composition"
    KINEMATIC_COUPLING = "kinematic_coupling"
    GUARD_TRANSITION = "guard_transition"
    FALLBACK = "fallback"


class PhaseType(StrEnum):
    FREE_MOTION = "free_motion"
    APPROACH = "approach"
    ESTABLISH_CONTACT = "establish_contact"
    MAINTAIN_CONTACT = "maintain_contact"
    APPLY_WRENCH = "apply_wrench"
    SHIFT_SUPPORT = "shift_support"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE_CONTACT = "release_contact"
    RECOVERY = "recovery"


class ConstraintType(StrEnum):
    FRICTION_CONE = "friction_cone"
    NO_SLIP = "no_slip"
    COLLISION_MARGIN = "collision_margin"
    MAX_CONTACT_FORCE = "max_contact_force"
    THRUST_MARGIN = "thrust_margin"
    PAYLOAD_MARGIN = "payload_margin"
    SUPPORT_RATIO = "support_ratio"
    VERTICAL_THRUST_RATIO = "vertical_thrust_ratio"
    TIME_LIMIT = "time_limit"
    JOINT_LIMIT = "joint_limit"
    WORKSPACE = "workspace"
    CLOSED_LOOP_REJECT = "closed_loop_reject"


class CapabilityType(StrEnum):
    GRASP = "grasp"
    SUPPORT = "support"
    PUSH = "push"
    LATCH = "latch"
    PERCH = "perch"
    SLIDE = "slide"
    FREE_FLIGHT = "free_flight"


@dataclass
class IRGNode(SchemaBase):
    node_id: int
    node_type: IRGNodeType
    ref_id: str | None
    priority: float
    is_hard: bool
    active_phase_id: int | None
    feature: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.node_id < 0:
            raise SchemaValidationError("IRGNode.node_id must be non-negative")
        if self.node_type == IRGNodeType.PHASE:
            phase_type = self.feature.get("phase_type")
            try:
                PhaseType(phase_type)
            except ValueError as exc:
                raise SchemaValidationError(f"PhaseNode.feature.phase_type is invalid: {phase_type!r}") from exc
            phase_label = self.feature.get("phase_label")
            if phase_label is not None and not isinstance(phase_label, str):
                raise SchemaValidationError("PhaseNode.feature.phase_label must be str or null")
        if self.node_type == IRGNodeType.CONSTRAINT and "constraint_type" in self.feature:
            try:
                ConstraintType(self.feature["constraint_type"])
            except ValueError as exc:
                raise SchemaValidationError(
                    f"ConstraintNode.feature.constraint_type is invalid: {self.feature['constraint_type']!r}"
                ) from exc
        if self.node_type == IRGNodeType.CONTACT_SLOT:
            forbidden = {"contact_point_world", "contact_pose_world", "candidate_id"}
            present = forbidden.intersection(self.feature)
            if present:
                raise SchemaValidationError(
                    f"ContactSlotNode must be abstract and cannot include final contact fields: {sorted(present)}"
                )


@dataclass
class IRGEdge(SchemaBase):
    src_id: int
    dst_id: int
    edge_type: IRGEdgeType
    priority: float = 1.0
    condition: Condition | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.src_id < 0 or self.dst_id < 0:
            raise SchemaValidationError("IRGEdge src_id and dst_id must be non-negative")


@dataclass
class InteractionRequirementGraph(SchemaBase):
    irg_id: str
    task_id: str
    nodes: list[IRGNode]
    edges: list[IRGEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.irg_id, "InteractionRequirementGraph.irg_id")
        require_non_empty(self.task_id, "InteractionRequirementGraph.task_id")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise SchemaValidationError("InteractionRequirementGraph.nodes has duplicate node_id values")
        node_id_set = set(node_ids)
        for edge in self.edges:
            if edge.src_id not in node_id_set or edge.dst_id not in node_id_set:
                raise SchemaValidationError("IRGEdge references a missing node_id")
        if self.nodes and not any(node.node_type == IRGNodeType.TASK for node in self.nodes):
            raise SchemaValidationError("InteractionRequirementGraph requires a task node")

