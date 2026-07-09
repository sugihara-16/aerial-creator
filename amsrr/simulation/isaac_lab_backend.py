from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.utils.config import load_config


ISAAC_LAB_BACKEND_VERSION = "isaac_lab_backend_v1"


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
    generated_usd_path: str = "artifacts/isaac/robots/holon/holon.usd"
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
            str(self._expanded_path(self.config.generated_usd_path)),
        ]

    @staticmethod
    def _expanded_path(path: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(path)))


def _isaac_python_modules_available() -> bool:
    for module_name in ("isaaclab", "omni.isaac.lab"):
        try:
            if find_spec(module_name) is not None:
                return True
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return False
