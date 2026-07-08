from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec


@dataclass(frozen=True)
class DesignPolicyContext:
    task_spec: TaskSpec
    irg: InteractionRequirementGraph
    physical_model: PhysicalModel
    interaction_envelope: InteractionEnvelope | None = None


class DesignPolicyBase(Protocol):
    """π_D policy interface: produces morphology design, never actuator commands."""

    def design(self, context: DesignPolicyContext) -> DesignOutput:
        ...


class FixedSimpleDesignPolicy:
    """P1 deterministic π_D baseline for fixed/simple morphology experiments."""

    def __init__(self, teacher: object | None = None) -> None:
        if teacher is None:
            from amsrr.policies.design_teacher import DeterministicDesignTeacher

            teacher = DeterministicDesignTeacher()
        self._teacher = teacher

    def design(self, context: DesignPolicyContext) -> DesignOutput:
        return self._teacher.generate(context).design_output  # type: ignore[attr-defined]
