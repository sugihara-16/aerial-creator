from __future__ import annotations

"""PhysicalModel-bound Isaac joint-drive values for Order 9 runtimes."""

import math
from dataclasses import dataclass
from typing import Sequence

from amsrr.schemas.physical_model import PhysicalModel


@dataclass(frozen=True)
class Order9ActuatorRuntimeValues:
    gimbal_stiffness: float
    gimbal_damping: float
    gimbal_armature: float
    gimbal_effort_limit: float
    gimbal_velocity_limit: float
    dock_stiffness: float
    dock_damping: float
    dock_armature: float
    dock_effort_limit: float
    dock_velocity_limit: float


def order9_actuator_runtime_values(
    physical_model: PhysicalModel,
) -> Order9ActuatorRuntimeValues:
    """Resolve both joint roles from hash-bound PhysicalModel provenance."""

    physical_model.validate()
    specs = physical_model.metadata.get("joint_actuator_specs")
    if not isinstance(specs, dict):
        raise RuntimeError("Order9 PhysicalModel lacks joint-actuator provenance")

    def role_values(role: str) -> tuple[float, float, float, float, float]:
        spec = specs.get(role)
        if not isinstance(spec, dict):
            raise RuntimeError(f"Order9 PhysicalModel lacks {role} actuator spec")
        drive = spec.get("simulation_drive")
        if not isinstance(drive, dict):
            raise RuntimeError(
                f"Order9 PhysicalModel lacks {role} simulation-drive spec"
            )
        try:
            values = (
                float(drive["stiffness"]),
                float(drive["damping"]),
                float(drive.get("armature_kg_m2", 0.0)),
                float(spec["peak_torque_nm"]),
                float(drive["safe_velocity_limit_rad_s"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Order9 {role} actuator runtime fields are incomplete"
            ) from error
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Order9 {role} actuator runtime fields must be finite"
            )
        if values[0] <= 0.0 or values[1] < 0.0 or values[2] < 0.0:
            raise RuntimeError(f"Order9 {role} drive gains/armature are invalid")
        if values[3] <= 0.0 or values[4] <= 0.0:
            raise RuntimeError(f"Order9 {role} effort/velocity limits are invalid")
        return values

    gimbal = role_values("vectoring")
    dock = role_values("dock")
    return Order9ActuatorRuntimeValues(
        gimbal_stiffness=gimbal[0],
        gimbal_damping=gimbal[1],
        gimbal_armature=gimbal[2],
        gimbal_effort_limit=gimbal[3],
        gimbal_velocity_limit=gimbal[4],
        dock_stiffness=dock[0],
        dock_damping=dock[1],
        dock_armature=dock[2],
        dock_effort_limit=dock[3],
        dock_velocity_limit=dock[4],
    )


def validate_order9_actuator_readback(
    joint_names: Sequence[str],
    effort_limits_nm: Sequence[float],
    velocity_limits_rad_s: Sequence[float],
    *,
    expected: Order9ActuatorRuntimeValues,
    absolute_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    """Fail closed unless Isaac applied the same limits to every matched joint."""

    names = tuple(str(value) for value in joint_names)
    effort = tuple(float(value) for value in effort_limits_nm)
    velocity = tuple(float(value) for value in velocity_limits_rad_s)
    if len(names) != len(effort) or len(names) != len(velocity):
        raise RuntimeError("Order9 actuator readback widths differ")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("Order9 actuator readback tolerance must be positive")

    def validate_role(
        label: str,
        pattern: str,
        expected_effort: float,
        expected_velocity: float,
    ) -> dict[str, object]:
        indices = [index for index, name in enumerate(names) if pattern in name]
        if not indices:
            raise RuntimeError(f"Order9 Isaac robot has no {label} joints")
        if any(
            not math.isclose(
                effort[index],
                expected_effort,
                rel_tol=1.0e-5,
                abs_tol=absolute_tolerance,
            )
            for index in indices
        ):
            raise RuntimeError(
                f"Order9 Isaac {label} effort limits differ from PhysicalModel"
            )
        if any(
            not math.isclose(
                velocity[index],
                expected_velocity,
                rel_tol=1.0e-5,
                abs_tol=absolute_tolerance,
            )
            for index in indices
        ):
            raise RuntimeError(
                f"Order9 Isaac {label} velocity limits differ from PhysicalModel"
            )
        first = indices[0]
        return {
            "joint_count": len(indices),
            "effort_limit_nm": effort[first],
            "velocity_limit_rad_s": velocity[first],
        }

    return {
        "gimbal": validate_role(
            "gimbal",
            "gimbal",
            expected.gimbal_effort_limit,
            expected.gimbal_velocity_limit,
        ),
        "dock": validate_role(
            "Dock",
            "dock_mech",
            expected.dock_effort_limit,
            expected.dock_velocity_limit,
        ),
        "matches_physical_model": True,
    }


__all__ = [
    "Order9ActuatorRuntimeValues",
    "order9_actuator_runtime_values",
    "validate_order9_actuator_readback",
]
