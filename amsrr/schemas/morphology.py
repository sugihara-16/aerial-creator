from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, StrEnum, require_len, require_non_empty
from amsrr.schemas.physical_model import ModuleCapabilityToken


@dataclass
class ModuleNode(SchemaBase):
    module_id: int
    module_type: str
    pose_in_design_frame: Pose7D
    role_id: str
    is_base: bool
    capability_token: ModuleCapabilityToken
    health: float = 1.0

    def validate(self) -> None:
        if self.module_id < 0:
            raise SchemaValidationError("ModuleNode.module_id must be non-negative")
        require_non_empty(self.module_type, "ModuleNode.module_type")
        require_len(self.pose_in_design_frame, 7, "ModuleNode.pose_in_design_frame")
        require_non_empty(self.role_id, "ModuleNode.role_id")
        if not 0.0 <= self.health <= 1.0:
            raise SchemaValidationError("ModuleNode.health must be in [0, 1]")


@dataclass
class PortNode(SchemaBase):
    port_global_id: int
    module_id: int
    port_local_id: str
    local_pose: Pose7D
    port_type: str
    occupied: bool
    compatible_port_type_mask: list[int]

    def validate(self) -> None:
        if self.port_global_id < 0:
            raise SchemaValidationError("PortNode.port_global_id must be non-negative")
        if self.module_id < 0:
            raise SchemaValidationError("PortNode.module_id must be non-negative")
        require_non_empty(self.port_local_id, "PortNode.port_local_id")
        require_len(self.local_pose, 7, "PortNode.local_pose")
        require_non_empty(self.port_type, "PortNode.port_type")


@dataclass
class DockEdge(SchemaBase):
    edge_id: int
    src_module_id: int
    src_port_id: int
    dst_module_id: int
    dst_port_id: int
    relative_pose_src_to_dst: Pose7D
    edge_role: Literal["structural", "grasp_arm", "support", "perch_anchor", "locomotion_support"]
    estimated_stiffness: list[float]
    latch_state: Literal["planned", "attached", "detached"]

    def validate(self) -> None:
        if min(self.edge_id, self.src_module_id, self.src_port_id, self.dst_module_id, self.dst_port_id) < 0:
            raise SchemaValidationError("DockEdge ids must be non-negative")
        require_len(self.relative_pose_src_to_dst, 7, "DockEdge.relative_pose_src_to_dst")
        require_len(self.estimated_stiffness, 6, "DockEdge.estimated_stiffness")


@dataclass
class RobotAnchor(SchemaBase):
    anchor_id: int
    module_id: int
    link_id: str | None
    local_pose: Pose7D
    anchor_type: Literal["grasp", "support", "push", "latch", "perch", "tool", "body_contact"]
    capability: dict[str, Any]
    associated_contact_slot_ids: list[int]

    def validate(self) -> None:
        if self.anchor_id < 0 or self.module_id < 0:
            raise SchemaValidationError("RobotAnchor ids must be non-negative")
        require_len(self.local_pose, 7, "RobotAnchor.local_pose")


@dataclass
class ControlGroup(SchemaBase):
    group_id: str
    module_ids: list[int]
    role: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.group_id, "ControlGroup.group_id")
        require_non_empty(self.role, "ControlGroup.role")


@dataclass
class MorphologyGraph(SchemaBase):
    graph_id: str
    modules: list[ModuleNode]
    ports: list[PortNode]
    dock_edges: list[DockEdge]
    robot_anchors: list[RobotAnchor]
    control_groups: list[ControlGroup]
    base_module_id: int
    is_closed_loop: bool

    def validate(self) -> None:
        require_non_empty(self.graph_id, "MorphologyGraph.graph_id")
        module_ids = [module.module_id for module in self.modules]
        if len(module_ids) != len(set(module_ids)):
            raise SchemaValidationError("MorphologyGraph.modules has duplicate module_id values")
        if self.modules and self.base_module_id not in set(module_ids):
            raise SchemaValidationError("MorphologyGraph.base_module_id must reference a module")
        base_count = sum(1 for module in self.modules if module.is_base)
        if self.modules and base_count != 1:
            raise SchemaValidationError("MorphologyGraph requires exactly one base module")


class DesignActionType(StrEnum):
    ADD_MODULE = "add_module"
    CONNECT_PORT = "connect_port"
    DISCONNECT_PORT = "disconnect_port"
    ASSIGN_ROLE = "assign_role"
    CREATE_ANCHOR = "create_anchor"
    BIND_ANCHOR_TO_SLOT = "bind_anchor_to_slot"
    SET_CONTROL_GROUP = "set_control_group"
    SET_BASE_MODULE = "set_base_module"
    STOP = "stop"


@dataclass
class DesignAction(SchemaBase):
    action_type: DesignActionType
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotAnchorBindingPrior(SchemaBase):
    slot_id: int
    anchor_id: int
    score: float
    reason_code: str | None = None

    def validate(self) -> None:
        if self.slot_id < 0 or self.anchor_id < 0:
            raise SchemaValidationError("SlotAnchorBindingPrior ids must be non-negative")


@dataclass
class DesignOutput(SchemaBase):
    task_id: str
    irg_id: str
    target_morphology: MorphologyGraph
    module_roles: dict[int, str]
    slot_anchor_binding_prior: list[SlotAnchorBindingPrior]
    design_actions: list[DesignAction]
    design_logprobs: list[float] | None = None
    design_scores: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.task_id, "DesignOutput.task_id")
        require_non_empty(self.irg_id, "DesignOutput.irg_id")
        if self.design_logprobs is not None and len(self.design_logprobs) != len(self.design_actions):
            raise SchemaValidationError("DesignOutput.design_logprobs must match design_actions length")

