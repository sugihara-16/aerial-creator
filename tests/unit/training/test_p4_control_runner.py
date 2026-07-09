from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training import (
    P4ControlLowLevelRunner,
    P4ControlLowLevelRunnerConfig,
    ensure_real_smoke_requested,
    load_p4_control_low_level_runner_config,
)


def test_p4_control_runner_config_loader() -> None:
    runner_config, env_config = load_p4_control_low_level_runner_config("configs/training/p4_control_low_level.yaml")

    assert runner_config.runner_version == "p4_control_low_level_runner_v1"
    assert runner_config.dry_run is True
    assert env_config.config_path == "configs/env/isaac_lab.yaml"
    assert env_config.max_episode_steps == 600


def test_p4_control_runner_dry_run_records_skipped_smokes() -> None:
    runner_config, env_config = load_p4_control_low_level_runner_config("configs/training/p4_control_low_level.yaml")
    runner = P4ControlLowLevelRunner(runner_config=runner_config, env_config=env_config)

    result = runner.run(archive_path=None)

    assert result.dry_run is True
    assert len(result.smoke_results) == 3
    assert result.metrics["smoke_skip_count"] == 3.0
    assert result.acceptance_report.fast_gate_passed is False
    assert result.acceptance_report.real_isaac_smoke_passed is False
    assert result.acceptance_report.completion_passed is False
    assert "P4-control produced no archives" in result.acceptance_report.failure_reasons


def test_p4_control_runner_real_smoke_guard() -> None:
    ensure_real_smoke_requested(P4ControlLowLevelRunnerConfig(dry_run=False))
    with pytest.raises(SchemaValidationError):
        ensure_real_smoke_requested(P4ControlLowLevelRunnerConfig(dry_run=True))
