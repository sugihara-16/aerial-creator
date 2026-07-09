from __future__ import annotations

import pytest

from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation import P4ControlSmokeResult
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
    assert runner_config.robot_model_config_path == "configs/robot/robot_model.yaml"
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


def test_p4_control_runner_real_smoke_builds_summary_archives(tmp_path) -> None:
    runner = P4ControlLowLevelRunner(
        runner_config=P4ControlLowLevelRunnerConfig(dry_run=False, archive_path=None),
        env=_FakeP4ControlEnv(),
    )
    archive_path = tmp_path / "p4_control_smoke.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.dry_run is False
    assert len(result.smoke_results) == 3
    assert len(result.archives) == 3
    assert result.acceptance_report.fast_gate_passed is True
    assert result.acceptance_report.real_isaac_smoke_passed is True
    assert result.acceptance_report.completion_passed is True
    assert result.acceptance_report.failure_reasons == []
    assert all(archive.runtime_observations for archive in result.archives)
    assert all(archive.controller_commands for archive in result.archives)
    assert all(archive.actuator_target_records for archive in result.archives)
    assert all(archive.rollout_artifacts["archive_type"] == "smoke_summary" for archive in result.archives)
    assert all(archive.rollout_artifacts["is_p4_full_completion"] is False for archive in result.archives)
    assert all(archive.rollout_artifacts["physical_success_claim"] is False for archive in result.archives)
    assert result.archives[0].controller_commands[0].controller_status.metrics["allocation_residual_norm"] == 0.0
    assert result.archives[0].actuator_target_records[0]["metrics"]["clipped_target_count"] == 0.0
    loaded = read_episode_archives_jsonl(archive_path)
    assert [archive.episode_id for archive in loaded] == [archive.episode_id for archive in result.archives]


class _FakeP4ControlEnv:
    def run_smokes(self, *, dry_run: bool) -> list[P4ControlSmokeResult]:
        assert dry_run is False
        return [
            _passed_smoke("single_module_hover", module_count=1, final_position_error_m=0.014),
            _passed_smoke("fixed_morphology_hover", module_count=2, final_position_error_m=0.014),
            _passed_smoke(
                "fixed_morphology_waypoint",
                module_count=2,
                final_position_error_m=0.018,
                ramp_duration_s=0.1,
            ),
        ]


def _passed_smoke(
    smoke_name: str,
    *,
    module_count: int,
    final_position_error_m: float,
    ramp_duration_s: float | None = None,
) -> P4ControlSmokeResult:
    metrics = {
        "module_count": float(module_count),
        f"{smoke_name}_smoke_passed": 1.0,
        f"{smoke_name}_module_count": float(module_count),
        f"{smoke_name}_steps": 200.0,
        f"{smoke_name}_duration_s": 1.0,
        f"{smoke_name}_final_position_error_m": final_position_error_m,
        f"{smoke_name}_final_attitude_error_rad": 0.01,
        f"{smoke_name}_qp_infeasible_count": 0.0,
        f"{smoke_name}_clipped_target_count": 0.0,
        f"{smoke_name}_missing_actuator_count": 0.0,
        f"{smoke_name}_unsupported_actuator_count": 0.0,
        f"{smoke_name}_last_bridge_allocation_residual_norm": 0.0,
    }
    if ramp_duration_s is not None:
        metrics[f"{smoke_name}_ramp_duration_s"] = ramp_duration_s
    return P4ControlSmokeResult(
        smoke_name,
        attempted=True,
        passed=True,
        skipped=False,
        isaac_backed=True,
        metrics=metrics,
    )
