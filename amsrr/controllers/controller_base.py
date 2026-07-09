from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from amsrr.controllers.policy_command_builder import DesiredBiasReferences
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerCommand, InteractionKnot, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation


@dataclass
class PayloadCoupling(SchemaBase):
    """Explicit payload load used by payload-coupled kinematic attach rollouts."""

    payload_id: str
    contact_model: str
    mass_kg: float
    inertia_body: list[float]
    com_offset_body: tuple[float, float, float]
    gravity_mps2: float = 9.80665
    coupling_mode: str = "kinematic_payload_coupled_attach_v1"

    def validate(self) -> None:
        require_non_empty(self.payload_id, "PayloadCoupling.payload_id")
        require_non_empty(self.contact_model, "PayloadCoupling.contact_model")
        require_len(self.inertia_body, 6, "PayloadCoupling.inertia_body")
        require_len(self.com_offset_body, 3, "PayloadCoupling.com_offset_body")
        if self.mass_kg <= 0.0:
            raise SchemaValidationError("PayloadCoupling.mass_kg must be positive")
        if self.gravity_mps2 <= 0.0:
            raise SchemaValidationError("PayloadCoupling.gravity_mps2 must be positive")


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
    payload_coupling: PayloadCoupling | None = None


class ControllerBase(Protocol):
    """Controller interface: converts pi_L intent into actuator-level commands."""

    def compute(self, context: ControllerContext) -> ControllerCommand:
        ...
