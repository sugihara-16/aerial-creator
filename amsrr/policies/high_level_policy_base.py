from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from amsrr.policies.high_level_requests import HighLevelDecisionContext
    from amsrr.schemas.high_level import HighLevelRequest

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
    """Current pi_H interface: rank finite requests, never generate a CWT."""

    @property
    def policy_version(self) -> str: ...

    def rank(self, context: HighLevelDecisionContext) -> list[HighLevelRequest]: ...


class LegacyHighLevelTrajectoryPlanner(Protocol):
    """Explicit pre-redesign teacher/replay interface, retained for C3 history."""

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory: ...
