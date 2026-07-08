from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from amsrr.controllers.policy_command_builder import DesiredBiasReferences
from amsrr.schemas.common import SchemaBase
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerCommand, InteractionKnot, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation


@dataclass
class ControllerContext(SchemaBase):
    runtime_observation: RuntimeObservation
    morphology_graph: MorphologyGraph
    physical_model: PhysicalModel
    active_knot: InteractionKnot
    policy_command: PolicyCommand
    desired_references: DesiredBiasReferences | None = None
    previous_command: ControllerCommand | None = None
    control_dt_s: float = 0.005


class ControllerBase(Protocol):
    """Controller interface: converts pi_L intent into actuator-level commands."""

    def compute(self, context: ControllerContext) -> ControllerCommand:
        ...
