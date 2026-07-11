from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.config import load_config


@dataclass(frozen=True)
class SimulationJointDrive:
    stiffness: float
    damping: float
    safe_velocity_limit_rad_s: float
    provenance: str


@dataclass(frozen=True)
class JointActuatorSpec:
    role: str
    manufacturer: str
    model: str
    user_reported_model: str
    model_resolution_note: str
    joint_name_patterns: tuple[str, ...]
    command_interface: str
    nominal_voltage_v: float
    voltage_range_v: tuple[float, float] | None
    continuous_torque_limit_nm: float
    continuous_torque_basis: str
    peak_torque_nm: float
    peak_torque_basis: str
    rated_speed_rad_s: float | None
    rated_speed_basis: str | None
    no_load_speed_rad_s: float
    no_load_speed_basis: str
    protocol_velocity_limit_rad_s: float | None
    protocol_velocity_limit_basis: str | None
    rated_current_a: float | None
    peak_current_a: float
    gear_ratio: float
    encoder_resolution_counts: int
    operating_temperature_c: tuple[float, float]
    actuator_mass_kg: float
    simulation_drive: SimulationJointDrive
    supported_control_modes: tuple[str, ...]
    sources: tuple[str, ...]
    protocol_torque_limit_nm: float | None = None
    protocol_torque_limit_basis: str | None = None
    backlash_rad: float | None = None
    backdrive_torque_nm: float | None = None
    mechanical_time_constant_s: float | None = None

    def matches(self, joint_id: str) -> bool:
        return any(fnmatch.fnmatchcase(joint_id, pattern) for pattern in self.joint_name_patterns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointActuatorModel:
    version: str
    actuator_roles: dict[str, JointActuatorSpec]

    def spec_for_joint(self, joint_id: str) -> JointActuatorSpec | None:
        matches = [spec for spec in self.actuator_roles.values() if spec.matches(joint_id)]
        if len(matches) > 1:
            roles = sorted(spec.role for spec in matches)
            raise SchemaValidationError(f"Joint {joint_id!r} matches multiple actuator roles: {roles}")
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "actuator_roles": {
                role: spec.to_dict()
                for role, spec in sorted(self.actuator_roles.items())
            },
        }


def load_joint_actuator_model(path: str | Path) -> JointActuatorModel:
    data = load_config(path)
    root = data.get("joint_actuator_model", data)
    if not isinstance(root, dict):
        raise SchemaValidationError("joint_actuator_model config root must be a mapping")
    version = _non_empty_string(root.get("version"), "joint_actuator_model.version")
    raw_roles = root.get("actuator_roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise SchemaValidationError("joint_actuator_model.actuator_roles must be a non-empty mapping")
    roles = {
        str(role): _parse_spec(str(role), raw)
        for role, raw in sorted(raw_roles.items())
    }
    model = JointActuatorModel(version=version, actuator_roles=roles)
    _validate_patterns_do_not_overlap(model)
    return model


def _parse_spec(role: str, raw: object) -> JointActuatorSpec:
    if not isinstance(raw, dict):
        raise SchemaValidationError(f"joint actuator role {role!r} must be a mapping")
    drive_raw = raw.get("simulation_drive")
    if not isinstance(drive_raw, dict):
        raise SchemaValidationError(f"joint actuator role {role!r} is missing simulation_drive")
    drive = SimulationJointDrive(
        stiffness=_non_negative_float(drive_raw.get("stiffness"), f"{role}.simulation_drive.stiffness"),
        damping=_non_negative_float(drive_raw.get("damping"), f"{role}.simulation_drive.damping"),
        safe_velocity_limit_rad_s=_positive_float(
            drive_raw.get("safe_velocity_limit_rad_s"),
            f"{role}.simulation_drive.safe_velocity_limit_rad_s",
        ),
        provenance=_non_empty_string(drive_raw.get("provenance"), f"{role}.simulation_drive.provenance"),
    )
    nominal_voltage = _positive_float(raw.get("nominal_voltage_v"), f"{role}.nominal_voltage_v")
    voltage_range = _optional_pair(raw.get("voltage_range_v"), f"{role}.voltage_range_v")
    temperature_range = _pair(raw.get("operating_temperature_c"), f"{role}.operating_temperature_c")
    if voltage_range is not None and not voltage_range[0] <= nominal_voltage <= voltage_range[1]:
        raise SchemaValidationError(f"{role}.nominal_voltage_v must be inside voltage_range_v")
    spec = JointActuatorSpec(
        role=role,
        manufacturer=_non_empty_string(raw.get("manufacturer"), f"{role}.manufacturer"),
        model=_non_empty_string(raw.get("model"), f"{role}.model"),
        user_reported_model=_non_empty_string(raw.get("user_reported_model"), f"{role}.user_reported_model"),
        model_resolution_note=_non_empty_string(raw.get("model_resolution_note"), f"{role}.model_resolution_note"),
        joint_name_patterns=_string_tuple(raw.get("joint_name_patterns"), f"{role}.joint_name_patterns"),
        command_interface=_non_empty_string(raw.get("command_interface"), f"{role}.command_interface"),
        nominal_voltage_v=nominal_voltage,
        voltage_range_v=voltage_range,
        continuous_torque_limit_nm=_positive_float(
            raw.get("continuous_torque_limit_nm"), f"{role}.continuous_torque_limit_nm"
        ),
        continuous_torque_basis=_non_empty_string(
            raw.get("continuous_torque_basis"), f"{role}.continuous_torque_basis"
        ),
        peak_torque_nm=_positive_float(raw.get("peak_torque_nm"), f"{role}.peak_torque_nm"),
        peak_torque_basis=_non_empty_string(raw.get("peak_torque_basis"), f"{role}.peak_torque_basis"),
        rated_speed_rad_s=_optional_positive_float(raw.get("rated_speed_rad_s"), f"{role}.rated_speed_rad_s"),
        rated_speed_basis=_optional_non_empty_string(raw.get("rated_speed_basis"), f"{role}.rated_speed_basis"),
        no_load_speed_rad_s=_positive_float(raw.get("no_load_speed_rad_s"), f"{role}.no_load_speed_rad_s"),
        no_load_speed_basis=_non_empty_string(raw.get("no_load_speed_basis"), f"{role}.no_load_speed_basis"),
        protocol_velocity_limit_rad_s=_optional_positive_float(
            raw.get("protocol_velocity_limit_rad_s"), f"{role}.protocol_velocity_limit_rad_s"
        ),
        protocol_velocity_limit_basis=_optional_non_empty_string(
            raw.get("protocol_velocity_limit_basis"), f"{role}.protocol_velocity_limit_basis"
        ),
        rated_current_a=_optional_positive_float(raw.get("rated_current_a"), f"{role}.rated_current_a"),
        peak_current_a=_positive_float(raw.get("peak_current_a"), f"{role}.peak_current_a"),
        gear_ratio=_positive_float(raw.get("gear_ratio"), f"{role}.gear_ratio"),
        encoder_resolution_counts=_positive_int(
            raw.get("encoder_resolution_counts"), f"{role}.encoder_resolution_counts"
        ),
        operating_temperature_c=temperature_range,
        actuator_mass_kg=_positive_float(raw.get("actuator_mass_kg"), f"{role}.actuator_mass_kg"),
        simulation_drive=drive,
        supported_control_modes=_string_tuple(
            raw.get("supported_control_modes"), f"{role}.supported_control_modes"
        ),
        sources=_string_tuple(raw.get("sources"), f"{role}.sources"),
        protocol_torque_limit_nm=_optional_positive_float(
            raw.get("protocol_torque_limit_nm"), f"{role}.protocol_torque_limit_nm"
        ),
        protocol_torque_limit_basis=_optional_non_empty_string(
            raw.get("protocol_torque_limit_basis"), f"{role}.protocol_torque_limit_basis"
        ),
        backlash_rad=_optional_non_negative_float(raw.get("backlash_rad"), f"{role}.backlash_rad"),
        backdrive_torque_nm=_optional_non_negative_float(
            raw.get("backdrive_torque_nm"), f"{role}.backdrive_torque_nm"
        ),
        mechanical_time_constant_s=_optional_positive_float(
            raw.get("mechanical_time_constant_s"), f"{role}.mechanical_time_constant_s"
        ),
    )
    if spec.continuous_torque_limit_nm > spec.peak_torque_nm:
        raise SchemaValidationError(f"{role}.continuous_torque_limit_nm must not exceed peak_torque_nm")
    if spec.rated_speed_rad_s is not None and spec.rated_speed_rad_s > spec.no_load_speed_rad_s:
        raise SchemaValidationError(f"{role}.rated_speed_rad_s must not exceed no_load_speed_rad_s")
    if spec.simulation_drive.safe_velocity_limit_rad_s > spec.no_load_speed_rad_s:
        raise SchemaValidationError(f"{role}.simulation safe velocity must not exceed no-load speed")
    return spec


def _validate_patterns_do_not_overlap(model: JointActuatorModel) -> None:
    representative_ids = ("gimbal1", "pitch_dock_mech_joint1", "yaw_dock_mech_joint2")
    for joint_id in representative_ids:
        model.spec_for_joint(joint_id)


def _non_empty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_non_empty_string(value: object, path: str) -> str | None:
    return None if value is None else _non_empty_string(value, path)


def _positive_float(value: object, path: str) -> float:
    result = _number(value, path)
    if result <= 0.0:
        raise SchemaValidationError(f"{path} must be positive")
    return result


def _non_negative_float(value: object, path: str) -> float:
    result = _number(value, path)
    if result < 0.0:
        raise SchemaValidationError(f"{path} must be non-negative")
    return result


def _optional_positive_float(value: object, path: str) -> float | None:
    return None if value is None else _positive_float(value, path)


def _optional_non_negative_float(value: object, path: str) -> float | None:
    return None if value is None else _non_negative_float(value, path)


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{path} must be numeric")
    return float(value)


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaValidationError(f"{path} must be a positive integer")
    return value


def _pair(value: object, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SchemaValidationError(f"{path} must contain two numeric values")
    lower, upper = _number(value[0], f"{path}[0]"), _number(value[1], f"{path}[1]")
    if lower > upper:
        raise SchemaValidationError(f"{path} lower bound must not exceed upper bound")
    return lower, upper


def _optional_pair(value: object, path: str) -> tuple[float, float] | None:
    return None if value is None else _pair(value, path)


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SchemaValidationError(f"{path} must be a non-empty list")
    return tuple(_non_empty_string(item, f"{path}[]") for item in value)
