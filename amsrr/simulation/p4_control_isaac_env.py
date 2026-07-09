from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    position_error_threshold_m: float = 0.20
    attitude_error_threshold_rad: float = 0.25
    max_episode_steps: int = 600
    fixed_morphology_module_count: int = 2
    waypoint_target_position_m: tuple[float, float, float] = (0.25, 0.0, 0.2)
    waypoint_target_yaw_rad: float = 0.0

    def validate(self) -> None:
        require_non_empty(self.config_path, "P4ControlLowLevelEnvConfig.config_path")
        require_non_empty(self.robot_model_config_path, "P4ControlLowLevelEnvConfig.robot_model_config_path")
        for name in (
            "control_dt_s",
            "smoke_duration_s",
            "hold_duration_s",
            "position_error_threshold_m",
            "attitude_error_threshold_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise SchemaValidationError(f"P4ControlLowLevelEnvConfig.{name} must be positive")
        if self.max_episode_steps <= 0:
            raise SchemaValidationError("P4ControlLowLevelEnvConfig.max_episode_steps must be positive")
        if self.fixed_morphology_module_count < 1:
            raise SchemaValidationError("P4ControlLowLevelEnvConfig.fixed_morphology_module_count must be >= 1")
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
        hover_pose: Pose7D = (0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
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

        return [
            P4ControlSmokeResult(
                scenario.smoke_name,
                attempted=True,
                passed=False,
                skipped=False,
                isaac_backed=True,
                skip_reason="real_isaac_execution_not_implemented",
                metrics={**_scenario_metrics(scenario), "isaac_backend_available": 1.0},
            )
            for scenario in scenarios
        ]

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
