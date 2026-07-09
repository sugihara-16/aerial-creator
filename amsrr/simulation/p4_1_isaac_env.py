from __future__ import annotations

from pathlib import Path
from typing import Any

from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, IsaacLabBackendConfig, load_isaac_lab_backend_config
from amsrr.simulation.p4_1_backend_smoke import (
    P4_1BackendSmokeResult,
    P4_1FullSceneBackendConfig,
    evaluate_runtime_observation_joint_state,
)
from amsrr.utils.config import load_config


P4_1_ISAAC_BACKEND_ENV_VERSION = "p4_1_isaac_backend_env_v1"


def load_p4_1_full_scene_backend_config(
    path: str | Path,
) -> tuple[IsaacLabBackendConfig, P4_1FullSceneBackendConfig]:
    data = load_config(path)
    env_config = P4_1FullSceneBackendConfig.from_dict(data.get("env", {}))
    backend_config = load_isaac_lab_backend_config(env_config.config_path)
    return backend_config, env_config


class P4_1IsaacBackendEnv:
    """P4.1 full-scene Isaac backend smoke boundary.

    This is intentionally separate from P4-control hover acceptance. It checks
    that a robot, an object, and a floor can coexist in one Isaac stage and
    that per-step observations/commands/actuator records can be returned.
    """

    def __init__(
        self,
        *,
        backend: IsaacLabBackend | None = None,
        config: P4_1FullSceneBackendConfig | None = None,
    ) -> None:
        self.config = config or P4_1FullSceneBackendConfig()
        self.backend = backend or IsaacLabBackend()

    def run_smoke(
        self,
        *,
        dry_run: bool = True,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        uses_p2_selected_design: bool = True,
        uses_p3_assembled_morphology: bool = True,
    ) -> P4_1BackendSmokeResult:
        if dry_run:
            return P4_1BackendSmokeResult(
                smoke_name=self.config.smoke_name,
                attempted=False,
                passed=False,
                skipped=True,
                isaac_backed=False,
                skip_reason="dry_run",
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                metrics={
                    "dry_run": 1.0,
                    "module_count": float(module_count),
                    "full_scene_spawned": 0.0,
                },
            )

        availability = self.backend.availability()
        if not availability.available:
            return P4_1BackendSmokeResult(
                smoke_name=self.config.smoke_name,
                attempted=False,
                passed=False,
                skipped=True,
                isaac_backed=False,
                skip_reason=",".join(availability.missing_reasons),
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                metrics={
                    "isaac_backend_available": 0.0,
                    "module_count": float(module_count),
                },
            )

        try:
            report = self.backend.run_p4_1_full_scene_backend_smoke(
                config_path=self.config.config_path,
                force_convert=True,
                steps=self.config.max_episode_steps,
                module_count=module_count,
                module_spacing_m=module_spacing_m,
                object_size_m=self.config.object_size_m,
                object_mass_kg=self.config.object_mass_kg,
                object_pose_world=self.config.object_initial_pose_world,
                uses_p2_p3_design=uses_p2_selected_design and uses_p3_assembled_morphology,
            )
        except Exception as exc:  # pragma: no cover - real subprocess failures are environment-specific.
            return P4_1BackendSmokeResult(
                smoke_name=self.config.smoke_name,
                attempted=True,
                passed=False,
                skipped=False,
                isaac_backed=True,
                skip_reason=str(exc),
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                metrics={"runner_exception": 1.0, "module_count": float(module_count)},
            )
        return p4_1_result_from_report(
            report,
            smoke_name=self.config.smoke_name,
            uses_p2_selected_design=uses_p2_selected_design,
            uses_p3_assembled_morphology=uses_p3_assembled_morphology,
        )


def p4_1_result_from_report(
    report: dict[str, Any],
    *,
    smoke_name: str,
    uses_p2_selected_design: bool,
    uses_p3_assembled_morphology: bool,
) -> P4_1BackendSmokeResult:
    runtime_observations = [
        RuntimeObservation.from_dict(item)
        for item in report.get("p4_1_runtime_observations", [])
        if isinstance(item, dict)
    ]
    controller_commands = [
        ControllerCommand.from_dict(item)
        for item in report.get("p4_1_controller_commands", [])
        if isinstance(item, dict)
    ]
    actuator_target_records = [
        item for item in report.get("p4_1_actuator_target_records", []) if isinstance(item, dict)
    ]
    object_pose_history = [
        tuple(float(value) for value in item)
        for item in report.get("p4_1_object_pose_history", [])
        if isinstance(item, list | tuple) and len(item) == 7
    ]
    articulated = bool(report.get("p4_1_articulated_morphology", False))
    joint_metrics = evaluate_runtime_observation_joint_state(
        runtime_observations,
        articulated_morphology=articulated,
        articulated_model_update_metrics={
            "max_model_rotor_origin_change_m": _float_report_value(report, "p4_1_max_model_rotor_origin_change_m"),
            "max_model_allocation_change": _float_report_value(report, "p4_1_max_model_allocation_change"),
        },
    )
    metrics = _numeric_report_metrics(report)
    metrics.update(joint_metrics.metrics)
    passed = bool(report.get("p4_1_full_scene_backend_smoke_passed")) and joint_metrics.passed
    return P4_1BackendSmokeResult(
        smoke_name=smoke_name,
        attempted=True,
        passed=passed,
        skipped=False,
        isaac_backed=bool(report.get("isaac_backed", True)),
        uses_p2_selected_design=uses_p2_selected_design and bool(report.get("p4_1_uses_p2_p3", True)),
        uses_p3_assembled_morphology=uses_p3_assembled_morphology and bool(report.get("p4_1_uses_p2_p3", True)),
        full_scene_spawned=bool(report.get("p4_1_full_scene_spawned", False)),
        robot_spawned=bool(report.get("p4_1_robot_spawned", False)),
        object_spawned=bool(report.get("p4_1_object_spawned", False)),
        floor_spawned=bool(report.get("p4_1_floor_spawned", False)),
        articulated_morphology=articulated,
        skip_reason=None if passed else str(report.get("error", "p4_1_full_scene_backend_failed")),
        runtime_observations=runtime_observations,
        controller_commands=controller_commands,
        actuator_target_records=actuator_target_records,
        object_pose_history=object_pose_history,  # type: ignore[arg-type]
        joint_state_metrics=joint_metrics,
        metrics=metrics,
    )


def _numeric_report_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in report.items():
        if isinstance(value, bool):
            metrics[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _float_report_value(report: dict[str, Any], key: str) -> float:
    value = report.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0
