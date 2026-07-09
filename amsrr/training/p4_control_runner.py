from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend
from amsrr.simulation.p4_control_isaac_env import (
    P4ControlIsaacEnv,
    P4ControlLowLevelEnvConfig,
    load_p4_control_low_level_env_config,
)
from amsrr.simulation.p4_control_smoke import P4ControlSmokeResult
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


P4_CONTROL_LOW_LEVEL_RUNNER_VERSION = "p4_control_low_level_runner_v1"


@dataclass
class P4ControlLowLevelRunnerConfig(SchemaBase):
    seed: int = 0
    source_hash: str = "p4_control_low_level"
    runner_version: str = P4_CONTROL_LOW_LEVEL_RUNNER_VERSION
    dry_run: bool = True
    archive_path: str | None = "artifacts/p4_control/p4_control_smoke.jsonl"

    def validate(self) -> None:
        require_non_empty(self.source_hash, "P4ControlLowLevelRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P4ControlLowLevelRunnerConfig.runner_version")


@dataclass
class P4ControlLowLevelRunnerResult(SchemaBase):
    dry_run: bool
    smoke_results: list[P4ControlSmokeResult]
    acceptance_report: Any
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_p4_control_low_level_runner_config(
    path: str | Path,
) -> tuple[P4ControlLowLevelRunnerConfig, P4ControlLowLevelEnvConfig]:
    data = load_config(path)
    _, env_config = load_p4_control_low_level_env_config(path)
    return (
        P4ControlLowLevelRunnerConfig.from_dict(data.get("runner", {})),
        env_config,
    )


class P4ControlLowLevelRunner:
    """Run P4-control low-level smoke cases through the configured Isaac boundary."""

    def __init__(
        self,
        *,
        runner_config: P4ControlLowLevelRunnerConfig | None = None,
        env_config: P4ControlLowLevelEnvConfig | None = None,
        env: P4ControlIsaacEnv | None = None,
        archives: list[EpisodeArchive] | None = None,
    ) -> None:
        self.runner_config = runner_config or P4ControlLowLevelRunnerConfig()
        self.env_config = env_config or P4ControlLowLevelEnvConfig()
        self.env = env or P4ControlIsaacEnv(
            config=self.env_config,
            backend=IsaacLabBackend(),
        )
        self.archives = archives or []

    def run(self, *, archive_path: str | Path | None = None) -> P4ControlLowLevelRunnerResult:
        from amsrr.acceptance.p4_control_acceptance import run_p4_control_acceptance

        smoke_results = self.env.run_smokes(dry_run=self.runner_config.dry_run)
        acceptance_report = run_p4_control_acceptance(self.archives, smoke_results=smoke_results)
        output_path = archive_path
        if output_path is None and self.runner_config.archive_path is not None:
            output_path = self.runner_config.archive_path
        if output_path is not None and self.archives:
            write_episode_archives_jsonl(output_path, self.archives)
        return P4ControlLowLevelRunnerResult(
            dry_run=self.runner_config.dry_run,
            smoke_results=smoke_results,
            acceptance_report=acceptance_report,
            archives=self.archives,
            metrics={
                **acceptance_report.metrics,
                "dry_run": 1.0 if self.runner_config.dry_run else 0.0,
                "smoke_pass_count": float(sum(1 for result in smoke_results if result.passed)),
                "smoke_skip_count": float(sum(1 for result in smoke_results if result.skipped)),
                "config_hash": float(int(stable_hash(self.runner_config)[:8], 16)),
            },
        )


def ensure_real_smoke_requested(config: P4ControlLowLevelRunnerConfig) -> None:
    if config.dry_run:
        raise SchemaValidationError("P4-control real smoke requires runner.dry_run=false")
