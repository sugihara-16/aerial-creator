from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, IsaacLabBackendConfig, load_isaac_lab_backend_config
from amsrr.simulation.p4_control_smoke import P4_CONTROL_REQUIRED_SMOKES, P4ControlSmokeResult
from amsrr.utils.config import load_config


P4_CONTROL_ISAAC_ENV_VERSION = "p4_control_isaac_env_v1"


@dataclass
class P4ControlLowLevelEnvConfig(SchemaBase):
    config_path: str = "configs/env/isaac_lab.yaml"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    control_dt_s: float = 0.005
    smoke_duration_s: float = 3.0
    hold_duration_s: float = 1.0
    hover_target_height_m: float = 0.5
    position_error_threshold_m: float = 0.20
    attitude_error_threshold_rad: float = 0.25
    max_episode_steps: int = 600
    fixed_morphology_module_count: int = 2
    fixed_morphology_module_spacing_m: float = 0.45
    waypoint_target_position_m: tuple[float, float, float] = (0.25, 0.0, 0.2)
    waypoint_target_yaw_rad: float = 0.0

    def validate(self) -> None:
        require_non_empty(self.config_path, "P4ControlLowLevelEnvConfig.config_path")
        require_non_empty(self.robot_model_config_path, "P4ControlLowLevelEnvConfig.robot_model_config_path")
        for name in (
            "control_dt_s",
            "smoke_duration_s",
            "hold_duration_s",
            "hover_target_height_m",
            "position_error_threshold_m",
            "attitude_error_threshold_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise SchemaValidationError(f"P4ControlLowLevelEnvConfig.{name} must be positive")
        if self.max_episode_steps <= 0:
            raise SchemaValidationError("P4ControlLowLevelEnvConfig.max_episode_steps must be positive")
        if self.fixed_morphology_module_count < 1:
            raise SchemaValidationError("P4ControlLowLevelEnvConfig.fixed_morphology_module_count must be >= 1")
        if self.fixed_morphology_module_spacing_m <= 0.0:
            raise SchemaValidationError("P4ControlLowLevelEnvConfig.fixed_morphology_module_spacing_m must be positive")
        require_len(self.waypoint_target_position_m, 3, "P4ControlLowLevelEnvConfig.waypoint_target_position_m")


@dataclass
class P4ControlSmokeScenario(SchemaBase):
    smoke_name: str
    module_count: int
    target_pose_world: Pose7D
    duration_s: float
    hold_duration_s: float
    position_error_threshold_m: float
    attitude_error_threshold_rad: float
    waypoint_tracking: bool = False
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.smoke_name, "P4ControlSmokeScenario.smoke_name")
        if self.module_count < 1:
            raise SchemaValidationError("P4ControlSmokeScenario.module_count must be >= 1")
        require_len(self.target_pose_world, 7, "P4ControlSmokeScenario.target_pose_world")
        for name in (
            "duration_s",
            "hold_duration_s",
            "position_error_threshold_m",
            "attitude_error_threshold_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise SchemaValidationError(f"P4ControlSmokeScenario.{name} must be positive")


def load_p4_control_low_level_env_config(path: str | Path) -> tuple[IsaacLabBackendConfig, P4ControlLowLevelEnvConfig]:
    data = load_config(path)
    env_config = P4ControlLowLevelEnvConfig.from_dict(data.get("env", {}))
    backend_config = load_isaac_lab_backend_config(env_config.config_path)
    return backend_config, env_config


class P4ControlIsaacEnv:
    """P4-control smoke boundary.

    The class owns scenario definitions and backend availability checks. Real
    Isaac physics execution is intentionally left to the next order.
    """

    def __init__(
        self,
        *,
        backend: IsaacLabBackend | None = None,
        config: P4ControlLowLevelEnvConfig | None = None,
    ) -> None:
        self.config = config or P4ControlLowLevelEnvConfig()
        self.backend = backend or IsaacLabBackend()

    def smoke_scenarios(self) -> list[P4ControlSmokeScenario]:
        hover_pose: Pose7D = (0.0, 0.0, self.config.hover_target_height_m, 0.0, 0.0, 0.0, 1.0)
        waypoint_pose: Pose7D = (
            self.config.waypoint_target_position_m[0],
            self.config.waypoint_target_position_m[1],
            self.config.waypoint_target_position_m[2],
            0.0,
            0.0,
            0.0,
            1.0,
        )
        return [
            self._scenario("single_module_hover", module_count=1, target_pose_world=hover_pose),
            self._scenario(
                "fixed_morphology_hover",
                module_count=self.config.fixed_morphology_module_count,
                target_pose_world=hover_pose,
            ),
            self._scenario(
                "fixed_morphology_waypoint",
                module_count=self.config.fixed_morphology_module_count,
                target_pose_world=waypoint_pose,
                waypoint_tracking=True,
            ),
        ]

    def run_smokes(self, *, dry_run: bool = True) -> list[P4ControlSmokeResult]:
        scenarios = self.smoke_scenarios()
        if dry_run:
            return [
                P4ControlSmokeResult(
                    scenario.smoke_name,
                    attempted=False,
                    passed=False,
                    skipped=True,
                    isaac_backed=False,
                    skip_reason="dry_run",
                    metrics=_scenario_metrics(scenario),
                )
                for scenario in scenarios
            ]

        availability = self.backend.availability()
        if not availability.available:
            reason = ",".join(availability.missing_reasons)
            return [
                P4ControlSmokeResult(
                    scenario.smoke_name,
                    attempted=False,
                    passed=False,
                    skipped=True,
                    isaac_backed=False,
                    skip_reason=reason,
                    metrics={**_scenario_metrics(scenario), "isaac_backend_available": 0.0},
                )
                for scenario in scenarios
            ]

        results: list[P4ControlSmokeResult] = []
        for scenario in scenarios:
            if scenario.smoke_name == "single_module_hover":
                results.append(self._run_single_module_hover_smoke(scenario))
            elif scenario.smoke_name == "fixed_morphology_hover":
                results.append(self._run_fixed_morphology_hover_smoke(scenario))
            else:
                results.append(
                    P4ControlSmokeResult(
                        scenario.smoke_name,
                        attempted=False,
                        passed=False,
                        skipped=True,
                        isaac_backed=False,
                        skip_reason="real_isaac_execution_not_implemented",
                        metrics={**_scenario_metrics(scenario), "isaac_backend_available": 1.0},
                    )
            )
        return results

    def _run_single_module_hover_smoke(self, scenario: P4ControlSmokeScenario) -> P4ControlSmokeResult:
        metrics = {**_scenario_metrics(scenario), "isaac_backend_available": 1.0}
        try:
            report = self.backend.run_holon_single_module_hover_smoke(
                config_path=self.config.config_path,
                force_convert=True,
                steps=self.config.max_episode_steps,
                hover_target_height=scenario.target_pose_world[2],
                position_tolerance_m=scenario.position_error_threshold_m,
                attitude_tolerance_rad=scenario.attitude_error_threshold_rad,
                hold_duration_s=scenario.hold_duration_s,
            )
        except Exception as exc:  # pragma: no cover - real subprocess failures are environment-specific.
            return P4ControlSmokeResult(
                scenario.smoke_name,
                attempted=True,
                passed=False,
                skipped=False,
                isaac_backed=True,
                skip_reason=str(exc),
                metrics={**metrics, "runner_exception": 1.0},
            )
        metrics.update(_single_module_hover_report_metrics(report))
        passed = bool(report.get("single_module_hover_smoke_passed"))
        return P4ControlSmokeResult(
            scenario.smoke_name,
            attempted=True,
            passed=passed,
            skipped=False,
            isaac_backed=bool(report.get("isaac_backed", True)),
            skip_reason=None if passed else str(report.get("error", "single_module_hover_failed")),
            metrics=metrics,
        )

    def _run_fixed_morphology_hover_smoke(self, scenario: P4ControlSmokeScenario) -> P4ControlSmokeResult:
        metrics = {**_scenario_metrics(scenario), "isaac_backend_available": 1.0}
        try:
            report = self.backend.run_holon_fixed_morphology_hover_smoke(
                config_path=self.config.config_path,
                force_convert=True,
                steps=self.config.max_episode_steps,
                module_count=scenario.module_count,
                module_spacing_m=self.config.fixed_morphology_module_spacing_m,
                hover_target_height=scenario.target_pose_world[2],
                position_tolerance_m=scenario.position_error_threshold_m,
                attitude_tolerance_rad=scenario.attitude_error_threshold_rad,
                hold_duration_s=scenario.hold_duration_s,
            )
        except Exception as exc:  # pragma: no cover - real subprocess failures are environment-specific.
            return P4ControlSmokeResult(
                scenario.smoke_name,
                attempted=True,
                passed=False,
                skipped=False,
                isaac_backed=True,
                skip_reason=str(exc),
                metrics={**metrics, "runner_exception": 1.0},
            )
        metrics.update(_prefixed_smoke_report_metrics(report, "fixed_morphology_hover"))
        passed = bool(report.get("fixed_morphology_hover_smoke_passed"))
        return P4ControlSmokeResult(
            scenario.smoke_name,
            attempted=True,
            passed=passed,
            skipped=False,
            isaac_backed=bool(report.get("isaac_backed", True)),
            skip_reason=None if passed else str(report.get("error", "fixed_morphology_hover_failed")),
            metrics=metrics,
        )

    def _scenario(
        self,
        smoke_name: str,
        *,
        module_count: int,
        target_pose_world: Pose7D,
        waypoint_tracking: bool = False,
    ) -> P4ControlSmokeScenario:
        return P4ControlSmokeScenario(
            smoke_name=smoke_name,
            module_count=module_count,
            target_pose_world=target_pose_world,
            duration_s=self.config.smoke_duration_s,
            hold_duration_s=self.config.hold_duration_s,
            position_error_threshold_m=self.config.position_error_threshold_m,
            attitude_error_threshold_rad=self.config.attitude_error_threshold_rad,
            waypoint_tracking=waypoint_tracking,
            metadata={
                "env_version": P4_CONTROL_ISAAC_ENV_VERSION,
                "required_by_acceptance": smoke_name in P4_CONTROL_REQUIRED_SMOKES,
            },
        )


def _scenario_metrics(scenario: P4ControlSmokeScenario) -> dict[str, float]:
    return {
        "module_count": float(scenario.module_count),
        "duration_s": scenario.duration_s,
        "hold_duration_s": scenario.hold_duration_s,
        "position_error_threshold_m": scenario.position_error_threshold_m,
        "attitude_error_threshold_rad": scenario.attitude_error_threshold_rad,
        "waypoint_tracking": 1.0 if scenario.waypoint_tracking else 0.0,
    }


def _single_module_hover_report_metrics(report: dict[str, Any]) -> dict[str, float]:
    keys = (
        "command_returncode",
        "spawn_passed",
        "command_probe_passed",
        "single_module_hover_smoke",
        "single_module_hover_smoke_passed",
        "single_module_hover_steps",
        "single_module_hover_requested_steps",
        "single_module_hover_duration_s",
        "single_module_hover_hold_time_s",
        "single_module_hover_hold_required_s",
        "single_module_hover_stopped_on_hold",
        "single_module_hover_position_tolerance_m",
        "single_module_hover_attitude_tolerance_rad",
        "single_module_hover_final_position_error_m",
        "single_module_hover_final_attitude_error_rad",
        "single_module_hover_max_position_error_m",
        "single_module_hover_max_attitude_error_rad",
        "single_module_hover_min_height_m",
        "single_module_hover_max_height_m",
        "single_module_hover_finite_state",
        "single_module_hover_qp_infeasible_count",
        "single_module_hover_controller_clipped_count",
        "single_module_hover_missing_actuator_count",
        "single_module_hover_unsupported_actuator_count",
        "single_module_hover_clipped_target_count",
    )
    metrics: dict[str, float] = {}
    for key in keys:
        value = report.get(key)
        if isinstance(value, bool):
            metrics[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _prefixed_smoke_report_metrics(report: dict[str, Any], prefix: str) -> dict[str, float]:
    suffixes = (
        "smoke",
        "smoke_passed",
        "module_count",
        "module_spacing_m",
        "steps",
        "requested_steps",
        "duration_s",
        "hold_time_s",
        "hold_required_s",
        "stopped_on_hold",
        "position_tolerance_m",
        "attitude_tolerance_rad",
        "final_position_error_m",
        "final_attitude_error_rad",
        "max_position_error_m",
        "max_attitude_error_rad",
        "min_height_m",
        "max_height_m",
        "finite_state",
        "qp_infeasible_count",
        "controller_clipped_count",
        "missing_actuator_count",
        "unsupported_actuator_count",
        "clipped_target_count",
    )
    metrics: dict[str, float] = {}
    for suffix in suffixes:
        key = f"{prefix}_{suffix}"
        value = report.get(key)
        if isinstance(value, bool):
            metrics[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    for key in ("command_returncode", "spawn_passed", "command_probe_passed"):
        value = report.get(key)
        if isinstance(value, bool):
            metrics[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics
