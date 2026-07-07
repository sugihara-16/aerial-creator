from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import Pose7D, SchemaBase, Vector3, require_len, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerStatus


@dataclass
class ModuleRuntimeState(SchemaBase):
    module_id: int
    pose_world: Pose7D
    twist_world: list[float]
    joint_positions: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    health: float = 1.0

    def validate(self) -> None:
        require_len(self.pose_world, 7, "ModuleRuntimeState.pose_world")
        require_len(self.twist_world, 6, "ModuleRuntimeState.twist_world")


@dataclass
class ObjectRuntimeState(SchemaBase):
    object_id: str
    pose_world: Pose7D
    twist_world: list[float]
    generalized_q: list[float] | None = None
    generalized_qdot: list[float] | None = None

    def validate(self) -> None:
        require_non_empty(self.object_id, "ObjectRuntimeState.object_id")
        require_len(self.pose_world, 7, "ObjectRuntimeState.pose_world")
        require_len(self.twist_world, 6, "ObjectRuntimeState.twist_world")


@dataclass
class ContactState(SchemaBase):
    contact_id: str
    entity_a: str
    entity_b: str
    contact_pose_world: Pose7D | None = None
    normal_world: Vector3 | None = None
    wrench_world: list[float] | None = None
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.contact_id, "ContactState.contact_id")
        require_non_empty(self.entity_a, "ContactState.entity_a")
        require_non_empty(self.entity_b, "ContactState.entity_b")
        if self.contact_pose_world is not None:
            require_len(self.contact_pose_world, 7, "ContactState.contact_pose_world")
        if self.normal_world is not None:
            require_len(self.normal_world, 3, "ContactState.normal_world")
        if self.wrench_world is not None:
            require_len(self.wrench_world, 6, "ContactState.wrench_world")


@dataclass
class TaskProgressState(SchemaBase):
    phase_label: str | None = None
    progress_ratio: float = 0.0
    success: bool = False
    failure_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class RuntimeObservation(SchemaBase):
    time_s: float
    morphology_graph: MorphologyGraph
    module_states: list[ModuleRuntimeState]
    object_states: list[ObjectRuntimeState]
    contact_states: list[ContactState]
    controller_status: ControllerStatus
    task_progress: TaskProgressState

    def validate(self) -> None:
        if self.time_s < 0.0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("RuntimeObservation.time_s must be non-negative")

