from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.runtime import RuntimeObservation


@dataclass(frozen=True)
class HighLevelPolicyContext:
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    morphology_graph: MorphologyGraph
    contact_candidate_set: ContactCandidateSet
    runtime_observation: RuntimeObservation | None = None


class HighLevelPolicyBase(Protocol):
    """pi_H interface: plans contact-wrench trajectories, not actuator commands."""

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        ...
