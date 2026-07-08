from __future__ import annotations

from typing import Protocol

from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec


class SimulationEnvBase(Protocol):
    """Simulator boundary used before binding to Isaac Lab or another backend."""

    def reset(
        self,
        task_spec: TaskSpec | None = None,
        morphology: MorphologyGraph | None = None,
        *,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> RuntimeObservation:
        ...

    def step(self, controller_command: ControllerCommand) -> RuntimeObservation:
        ...

    def get_runtime_observation(self) -> RuntimeObservation:
        ...
