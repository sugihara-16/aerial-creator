from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaBase, require_non_empty
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput
from amsrr.schemas.policies import ContactWrenchTrajectory, ControllerCommand, PolicyCommand
from amsrr.schemas.task_spec import TaskSpec


@dataclass
class EpisodeArchive(SchemaBase):
    episode_id: str
    task_spec: TaskSpec
    task_hash: str
    geometry_hashes: dict[str, str]
    robot_model_hash: str
    config_hash: str
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    design_output: DesignOutput | None
    feasibility_result: FeasibilityResult | None
    assembly_plan: dict[str, Any] | None
    trajectory_records: list[ContactWrenchTrajectory]
    policy_commands: list[PolicyCommand]
    controller_commands: list[ControllerCommand]
    rewards: list[dict[str, float]]
    metrics: dict[str, float]
    success: bool
    failure_reason: str | None
    reproducibility: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.episode_id, "EpisodeArchive.episode_id")
        require_non_empty(self.task_hash, "EpisodeArchive.task_hash")
        require_non_empty(self.robot_model_hash, "EpisodeArchive.robot_model_hash")
        require_non_empty(self.config_hash, "EpisodeArchive.config_hash")


def write_episode_archives_jsonl(path: str | Path, archives: list[EpisodeArchive]) -> None:
    archive_path = Path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("w", encoding="utf-8") as handle:
        for archive in archives:
            handle.write(archive.to_json())
            handle.write("\n")


def read_episode_archives_jsonl(path: str | Path) -> list[EpisodeArchive]:
    archive_path = Path(path)
    archives: list[EpisodeArchive] = []
    with archive_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            archives.append(EpisodeArchive.from_dict(json.loads(line)))
    return archives
