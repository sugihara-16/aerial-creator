from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.utils.config import load_config


ISAAC_LAB_BACKEND_VERSION = "isaac_lab_backend_v1"
DEFAULT_WARP_CACHE_PATH = "/tmp/amsrr_warp_cache"


@dataclass
class IsaacLabBackendConfig(SchemaBase):
    isaaclab_path: str = "${HOME}/IsaacLab"
    micromamba_env: str = "isaaclab3"
    launch_script: str = "isaaclab.sh"
    headless: bool = True
    device: str = "cuda:0"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    holon_urdf_path: str = "assets/robots/holon/holon.urdf"
    generated_usd_dir: str = "artifacts/isaac/robots/holon"
    generated_usd_path: str = "artifacts/isaac/robots/holon/holon/holon.usda"
    rotor_force_application: str = "wrench_composer"
    per_thruster_target_record: bool = True

    def validate(self) -> None:
        for name in (
            "isaaclab_path",
            "micromamba_env",
            "launch_script",
            "device",
            "robot_model_config_path",
            "holon_urdf_path",
            "generated_usd_dir",
            "generated_usd_path",
            "rotor_force_application",
        ):
            require_non_empty(getattr(self, name), f"IsaacLabBackendConfig.{name}")


@dataclass
class IsaacLabAvailability(SchemaBase):
    available: bool
    isaaclab_path_exists: bool
    launch_script_exists: bool
    urdf_exists: bool
    generated_usd_exists: bool
    python_modules_available: bool
    missing_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class IsaacLabBackendUnavailable(RuntimeError):
    """Raised when a real Isaac smoke path is requested without Isaac availability."""


def load_isaac_lab_backend_config(path: str | Path) -> IsaacLabBackendConfig:
    data = load_config(path)
    return IsaacLabBackendConfig.from_dict(data.get("isaac_lab", data))


class IsaacLabBackend:
    """Thin availability/config boundary for later Isaac Lab smoke execution."""

    def __init__(self, config: IsaacLabBackendConfig | None = None) -> None:
        self.config = config or IsaacLabBackendConfig()

    def availability(self) -> IsaacLabAvailability:
        isaaclab_path = self._expanded_path(self.config.isaaclab_path)
        launch_script = isaaclab_path / self.config.launch_script
        urdf_path = self._expanded_path(self.config.holon_urdf_path)
        generated_usd_path = self._expanded_path(self.config.generated_usd_path)
        python_modules_available = _isaac_python_modules_available()
        effective_device = self._effective_device()
        missing_reasons: list[str] = []
        if not isaaclab_path.exists():
            missing_reasons.append("isaaclab_path_missing")
        if not launch_script.exists():
            missing_reasons.append("launch_script_missing")
        if not urdf_path.exists():
            missing_reasons.append("holon_urdf_missing")
        if not python_modules_available:
            missing_reasons.append("isaac_python_modules_unavailable_in_current_interpreter")
        return IsaacLabAvailability(
            available=not missing_reasons,
            isaaclab_path_exists=isaaclab_path.exists(),
            launch_script_exists=launch_script.exists(),
            urdf_exists=urdf_path.exists(),
            generated_usd_exists=generated_usd_path.exists(),
            python_modules_available=python_modules_available,
            missing_reasons=missing_reasons,
            metadata={
                "backend_version": ISAAC_LAB_BACKEND_VERSION,
                "isaaclab_path": str(isaaclab_path),
                "launch_script": str(launch_script),
                "holon_urdf_path": str(urdf_path),
                "generated_usd_path": str(generated_usd_path),
                "rotor_force_application": self.config.rotor_force_application,
                "configured_device": self.config.device,
                "effective_device": effective_device,
            },
        )

    def require_available(self) -> None:
        availability = self.availability()
        if not availability.available:
            raise IsaacLabBackendUnavailable(", ".join(availability.missing_reasons))

    def urdf_conversion_command(self) -> list[str]:
        isaaclab_path = self._expanded_path(self.config.isaaclab_path)
        return [
            str(isaaclab_path / self.config.launch_script),
            "-p",
            str(isaaclab_path / "scripts" / "tools" / "convert_urdf.py"),
            str(self._expanded_path(self.config.holon_urdf_path)),
            str(self._expanded_path(self.config.generated_usd_dir)),
        ]

    def holon_spawn_probe_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 5,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        isaaclab_path = self._expanded_path(self.config.isaaclab_path)
        repo_root = Path(__file__).resolve().parents[2]
        device = self._effective_device()
        command = [
            str(isaaclab_path / self.config.launch_script),
            "-p",
            str(repo_root / "scripts" / "p4_control_holon_spawn_probe.py"),
            "--config",
            str(config_path),
            "--steps",
            str(steps),
            "--device",
            device,
        ]
        if force_convert:
            command.append("--force-convert")
        elif convert_if_missing:
            command.append("--convert-if-missing")
        if generated_usd_dir is not None:
            command.extend(["--generated-usd-dir", str(generated_usd_dir)])
        if generated_usd_path is not None:
            command.extend(["--generated-usd-path", str(generated_usd_path)])
        return command

    def holon_command_probe_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 80,
        hover_force_scale: float = 0.5,
        gimbal_target_rad: float = 0.1,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.extend(
            [
                "--hover-force-scale",
                str(hover_force_scale),
                "--gimbal-target-rad",
                str(gimbal_target_rad),
            ]
        )
        return command

    def holon_controller_command_probe_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 80,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.append("--controller-command-smoke")
        return command

    def holon_single_module_hover_smoke_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 600,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.extend(
            [
                "--single-module-hover-smoke",
                "--hover-target-height",
                str(hover_target_height),
                "--hover-position-tolerance-m",
                str(position_tolerance_m),
                "--hover-attitude-tolerance-rad",
                str(attitude_tolerance_rad),
                "--hover-hold-duration-s",
                str(hold_duration_s),
            ]
        )
        return command

    def holon_single_module_articulated_hover_smoke_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 600,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        articulated_joint_amplitude_rad: float = 0.12,
        articulated_joint_period_s: float = 8.0,
        articulated_joint_warmup_s: float = 1.0,
        articulated_joint_tracking_tolerance_rad: float = 0.20,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_single_module_hover_smoke_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            hover_target_height=hover_target_height,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command[command.index("--single-module-hover-smoke")] = "--single-module-articulated-hover-smoke"
        command.extend(
            [
                "--articulated-joint-amplitude-rad",
                str(articulated_joint_amplitude_rad),
                "--articulated-joint-period-s",
                str(articulated_joint_period_s),
                "--articulated-joint-warmup-s",
                str(articulated_joint_warmup_s),
                "--articulated-joint-tracking-tolerance-rad",
                str(articulated_joint_tracking_tolerance_rad),
            ]
        )
        return command

    def holon_fixed_morphology_hover_smoke_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 600,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.extend(
            [
                "--fixed-morphology-hover-smoke",
                "--fixed-module-count",
                str(module_count),
                "--fixed-module-spacing-m",
                str(module_spacing_m),
                "--hover-target-height",
                str(hover_target_height),
                "--hover-position-tolerance-m",
                str(position_tolerance_m),
                "--hover-attitude-tolerance-rad",
                str(attitude_tolerance_rad),
                "--hover-hold-duration-s",
                str(hold_duration_s),
            ]
        )
        return command

    def holon_fixed_morphology_articulated_hover_smoke_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 600,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        articulated_joint_amplitude_rad: float = 0.12,
        articulated_joint_period_s: float = 8.0,
        articulated_joint_warmup_s: float = 1.0,
        articulated_joint_tracking_tolerance_rad: float = 0.20,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_fixed_morphology_hover_smoke_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            module_count=module_count,
            module_spacing_m=module_spacing_m,
            hover_target_height=hover_target_height,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command[command.index("--fixed-morphology-hover-smoke")] = "--fixed-morphology-articulated-hover-smoke"
        command.extend(
            [
                "--articulated-joint-amplitude-rad",
                str(articulated_joint_amplitude_rad),
                "--articulated-joint-period-s",
                str(articulated_joint_period_s),
                "--articulated-joint-warmup-s",
                str(articulated_joint_warmup_s),
                "--articulated-joint-tracking-tolerance-rad",
                str(articulated_joint_tracking_tolerance_rad),
            ]
        )
        return command

    def holon_fixed_morphology_waypoint_smoke_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 600,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        waypoint_target_position_m: tuple[float, float, float] = (0.05, 0.0, 0.5),
        waypoint_target_yaw_rad: float = 0.0,
        waypoint_ramp_duration_s: float = 0.1,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.extend(
            [
                "--fixed-morphology-waypoint-smoke",
                "--fixed-module-count",
                str(module_count),
                "--fixed-module-spacing-m",
                str(module_spacing_m),
                "--waypoint-target-position-m",
                str(waypoint_target_position_m[0]),
                str(waypoint_target_position_m[1]),
                str(waypoint_target_position_m[2]),
                "--waypoint-target-yaw-rad",
                str(waypoint_target_yaw_rad),
                "--waypoint-ramp-duration-s",
                str(waypoint_ramp_duration_s),
                "--hover-position-tolerance-m",
                str(position_tolerance_m),
                "--hover-attitude-tolerance-rad",
                str(attitude_tolerance_rad),
                "--hover-hold-duration-s",
                str(hold_duration_s),
            ]
        )
        return command

    def p4_1_full_scene_backend_smoke_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 80,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        object_size_m: tuple[float, float, float] = (0.30, 0.20, 0.15),
        object_mass_kg: float = 1.0,
        object_pose_world: tuple[float, float, float, float, float, float, float] = (
            0.8,
            0.0,
            0.4,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        uses_p2_p3_design: bool = True,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.extend(
            [
                "--p4-1-full-scene-backend-smoke",
                "--fixed-module-count",
                str(module_count),
                "--fixed-module-spacing-m",
                str(module_spacing_m),
                "--p4-1-object-size-m",
                str(object_size_m[0]),
                str(object_size_m[1]),
                str(object_size_m[2]),
                "--p4-1-object-mass-kg",
                str(object_mass_kg),
                "--p4-1-object-pose-world",
                str(object_pose_world[0]),
                str(object_pose_world[1]),
                str(object_pose_world[2]),
                str(object_pose_world[3]),
                str(object_pose_world[4]),
                str(object_pose_world[5]),
                str(object_pose_world[6]),
            ]
        )
        if uses_p2_p3_design:
            command.append("--p4-1-uses-p2-p3")
        return command

    def p4_2_deterministic_rollout_command(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        convert_if_missing: bool = True,
        force_convert: bool = False,
        steps: int = 1200,
        morphology_graph: MorphologyGraph,
        contact_candidate_set_json: str | None = None,
        contact_candidate_set_json_path: str | Path | None = None,
        contact_wrench_trajectory_json: str | None = None,
        contact_wrench_trajectory_json_path: str | Path | None = None,
        object_size_m: tuple[float, float, float] = (0.30, 0.20, 0.15),
        object_mass_kg: float = 1.0,
        object_pose_world: tuple[float, float, float, float, float, float, float] = (
            0.8,
            0.0,
            0.4,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        contact_model: str = "kinematic_payload_coupled_attach_v1",
        attach_distance_threshold_m: float = 0.06,
        attach_relative_velocity_threshold_mps: float = 0.20,
        attach_snap_distance_threshold_m: float = 0.03,
        pregrasp_alignment_distance_m: float = 0.12,
        uses_p2_p3_design: bool = True,
        generated_usd_dir: str | Path | None = None,
        generated_usd_path: str | Path | None = None,
    ) -> list[str]:
        command = self.holon_spawn_probe_command(
            config_path=config_path,
            convert_if_missing=convert_if_missing,
            force_convert=force_convert,
            steps=steps,
            generated_usd_dir=generated_usd_dir,
            generated_usd_path=generated_usd_path,
        )
        command.extend(
            [
                "--p4-2-deterministic-rollout",
                "--p4-2-morphology-graph-json",
                morphology_graph.to_json(),
                "--p4-2-object-size-m",
                str(object_size_m[0]),
                str(object_size_m[1]),
                str(object_size_m[2]),
                "--p4-2-object-mass-kg",
                str(object_mass_kg),
                "--p4-2-object-pose-world",
                str(object_pose_world[0]),
                str(object_pose_world[1]),
                str(object_pose_world[2]),
                str(object_pose_world[3]),
                str(object_pose_world[4]),
                str(object_pose_world[5]),
                str(object_pose_world[6]),
                "--p4-2-contact-model",
                str(contact_model),
                "--p4-2-attach-distance-threshold-m",
                str(attach_distance_threshold_m),
                "--p4-2-attach-relative-velocity-threshold-mps",
                str(attach_relative_velocity_threshold_mps),
                "--p4-2-attach-snap-distance-threshold-m",
                str(attach_snap_distance_threshold_m),
                "--p4-2-pregrasp-alignment-distance-m",
                str(pregrasp_alignment_distance_m),
            ]
        )
        if uses_p2_p3_design:
            command.append("--p4-2-uses-p2-p3")
        if contact_candidate_set_json is not None:
            command.extend(["--p4-2-contact-candidate-set-json", contact_candidate_set_json])
        if contact_candidate_set_json_path is not None:
            command.extend(["--p4-2-contact-candidate-set-json-path", str(contact_candidate_set_json_path)])
        if contact_wrench_trajectory_json is not None:
            command.extend(["--p4-2-contact-wrench-trajectory-json", contact_wrench_trajectory_json])
        if contact_wrench_trajectory_json_path is not None:
            command.extend(["--p4-2-contact-wrench-trajectory-json-path", str(contact_wrench_trajectory_json_path)])
        return command

    def run_holon_single_module_hover_smoke(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 600,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        command = self.holon_single_module_hover_smoke_command(
            config_path=config_path,
            convert_if_missing=not force_convert,
            force_convert=force_convert,
            steps=steps,
            hover_target_height=hover_target_height,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            generated_usd_dir=self.config.generated_usd_dir,
            generated_usd_path=self.config.generated_usd_path,
        )
        return _run_json_command(command, timeout_s=timeout_s)

    def run_holon_single_module_articulated_hover_smoke(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 600,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        articulated_joint_amplitude_rad: float = 0.12,
        articulated_joint_period_s: float = 8.0,
        articulated_joint_warmup_s: float = 1.0,
        articulated_joint_tracking_tolerance_rad: float = 0.20,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        command = self.holon_single_module_articulated_hover_smoke_command(
            config_path=config_path,
            convert_if_missing=not force_convert,
            force_convert=force_convert,
            steps=steps,
            hover_target_height=hover_target_height,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            articulated_joint_amplitude_rad=articulated_joint_amplitude_rad,
            articulated_joint_period_s=articulated_joint_period_s,
            articulated_joint_warmup_s=articulated_joint_warmup_s,
            articulated_joint_tracking_tolerance_rad=articulated_joint_tracking_tolerance_rad,
            generated_usd_dir=self.config.generated_usd_dir,
            generated_usd_path=self.config.generated_usd_path,
        )
        return _run_json_command(command, timeout_s=timeout_s)

    def run_holon_fixed_morphology_hover_smoke(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 600,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        command = self.holon_fixed_morphology_hover_smoke_command(
            config_path=config_path,
            convert_if_missing=not force_convert,
            force_convert=force_convert,
            steps=steps,
            module_count=module_count,
            module_spacing_m=module_spacing_m,
            hover_target_height=hover_target_height,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            generated_usd_dir=self.config.generated_usd_dir,
            generated_usd_path=self.config.generated_usd_path,
        )
        return _run_json_command(command, timeout_s=timeout_s)

    def run_holon_fixed_morphology_articulated_hover_smoke(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 600,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        hover_target_height: float = 0.5,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        articulated_joint_amplitude_rad: float = 0.12,
        articulated_joint_period_s: float = 8.0,
        articulated_joint_warmup_s: float = 1.0,
        articulated_joint_tracking_tolerance_rad: float = 0.20,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        command = self.holon_fixed_morphology_articulated_hover_smoke_command(
            config_path=config_path,
            convert_if_missing=not force_convert,
            force_convert=force_convert,
            steps=steps,
            module_count=module_count,
            module_spacing_m=module_spacing_m,
            hover_target_height=hover_target_height,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            articulated_joint_amplitude_rad=articulated_joint_amplitude_rad,
            articulated_joint_period_s=articulated_joint_period_s,
            articulated_joint_warmup_s=articulated_joint_warmup_s,
            articulated_joint_tracking_tolerance_rad=articulated_joint_tracking_tolerance_rad,
            generated_usd_dir=self.config.generated_usd_dir,
            generated_usd_path=self.config.generated_usd_path,
        )
        return _run_json_command(command, timeout_s=timeout_s)

    def run_holon_fixed_morphology_waypoint_smoke(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 600,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        waypoint_target_position_m: tuple[float, float, float] = (0.05, 0.0, 0.5),
        waypoint_target_yaw_rad: float = 0.0,
        waypoint_ramp_duration_s: float = 0.1,
        position_tolerance_m: float = 0.20,
        attitude_tolerance_rad: float = 0.25,
        hold_duration_s: float = 1.0,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        command = self.holon_fixed_morphology_waypoint_smoke_command(
            config_path=config_path,
            convert_if_missing=not force_convert,
            force_convert=force_convert,
            steps=steps,
            module_count=module_count,
            module_spacing_m=module_spacing_m,
            waypoint_target_position_m=waypoint_target_position_m,
            waypoint_target_yaw_rad=waypoint_target_yaw_rad,
            waypoint_ramp_duration_s=waypoint_ramp_duration_s,
            position_tolerance_m=position_tolerance_m,
            attitude_tolerance_rad=attitude_tolerance_rad,
            hold_duration_s=hold_duration_s,
            generated_usd_dir=self.config.generated_usd_dir,
            generated_usd_path=self.config.generated_usd_path,
        )
        return _run_json_command(command, timeout_s=timeout_s)

    def run_p4_1_full_scene_backend_smoke(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 80,
        module_count: int = 2,
        module_spacing_m: float = 0.45,
        object_size_m: tuple[float, float, float] = (0.30, 0.20, 0.15),
        object_mass_kg: float = 1.0,
        object_pose_world: tuple[float, float, float, float, float, float, float] = (
            0.8,
            0.0,
            0.4,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        uses_p2_p3_design: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        command = self.p4_1_full_scene_backend_smoke_command(
            config_path=config_path,
            convert_if_missing=not force_convert,
            force_convert=force_convert,
            steps=steps,
            module_count=module_count,
            module_spacing_m=module_spacing_m,
            object_size_m=object_size_m,
            object_mass_kg=object_mass_kg,
            object_pose_world=object_pose_world,
            uses_p2_p3_design=uses_p2_p3_design,
            generated_usd_dir=self.config.generated_usd_dir,
            generated_usd_path=self.config.generated_usd_path,
        )
        return _run_json_command(command, timeout_s=timeout_s)

    def run_p4_2_deterministic_rollout(
        self,
        *,
        config_path: str | Path = "configs/env/isaac_lab.yaml",
        force_convert: bool = True,
        steps: int = 1200,
        morphology_graph: MorphologyGraph,
        contact_candidate_set_json: str | None = None,
        contact_wrench_trajectory_json: str | None = None,
        object_size_m: tuple[float, float, float] = (0.30, 0.20, 0.15),
        object_mass_kg: float = 1.0,
        object_pose_world: tuple[float, float, float, float, float, float, float] = (
            0.8,
            0.0,
            0.4,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        contact_model: str = "kinematic_payload_coupled_attach_v1",
        attach_distance_threshold_m: float = 0.06,
        attach_relative_velocity_threshold_mps: float = 0.20,
        attach_snap_distance_threshold_m: float = 0.03,
        pregrasp_alignment_distance_m: float = 0.12,
        uses_p2_p3_design: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="amsrr_p4_2_inputs_") as input_dir:
            candidate_path = None
            trajectory_path = None
            if contact_candidate_set_json is not None:
                candidate_path = Path(input_dir) / "contact_candidate_set.json"
                candidate_path.write_text(contact_candidate_set_json, encoding="utf-8")
            if contact_wrench_trajectory_json is not None:
                trajectory_path = Path(input_dir) / "contact_wrench_trajectory.json"
                trajectory_path.write_text(contact_wrench_trajectory_json, encoding="utf-8")
            command = self.p4_2_deterministic_rollout_command(
                config_path=config_path,
                convert_if_missing=not force_convert,
                force_convert=force_convert,
                steps=steps,
                morphology_graph=morphology_graph,
                contact_candidate_set_json_path=candidate_path,
                contact_wrench_trajectory_json_path=trajectory_path,
                object_size_m=object_size_m,
                object_mass_kg=object_mass_kg,
                object_pose_world=object_pose_world,
                contact_model=contact_model,
                attach_distance_threshold_m=attach_distance_threshold_m,
                attach_relative_velocity_threshold_mps=attach_relative_velocity_threshold_mps,
                attach_snap_distance_threshold_m=attach_snap_distance_threshold_m,
                pregrasp_alignment_distance_m=pregrasp_alignment_distance_m,
                uses_p2_p3_design=uses_p2_p3_design,
                generated_usd_dir=self.config.generated_usd_dir,
                generated_usd_path=self.config.generated_usd_path,
            )
            return _run_json_command(command, timeout_s=timeout_s)

    @staticmethod
    def _expanded_path(path: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(path)))

    def _effective_device(self) -> str:
        if self.config.device.startswith("cuda") and not _torch_cuda_available():
            return "cpu"
        return self.config.device


def _isaac_python_modules_available() -> bool:
    for module_name in ("isaaclab", "omni.isaac.lab"):
        try:
            if find_spec(module_name) is not None:
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return False


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run_json_command(command: list[str], *, timeout_s: float | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["WARP_CACHE_PATH"] = env.get("WARP_CACHE_PATH", DEFAULT_WARP_CACHE_PATH)
    Path(env["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    report = _parse_last_json_line(completed.stdout)
    if report is None:
        report = {
            "spawn_passed": False,
            "isaac_backed": True,
            "error": "isaac_command_did_not_emit_json",
        }
        if completed.stdout:
            report["stdout_tail"] = completed.stdout[-2000:]
    report["command_returncode"] = completed.returncode
    if completed.stderr:
        report["stderr_tail"] = completed.stderr[-2000:]
    if completed.returncode != 0 and "error" not in report:
        report["error"] = "isaac_command_failed"
    return report


def _parse_last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate or not candidate.startswith("{"):
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None
