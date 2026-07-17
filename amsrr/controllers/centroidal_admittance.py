from __future__ import annotations

"""Aggregate centroidal disturbance estimation and bounded admittance.

This module deliberately estimates only the net external wrench acting on the
assembled morphology.  It does not attempt to decompose per-contact or
internal Dock wrenches, and it does not add either quantity to PolicyCommand.
"""

import math
from dataclasses import dataclass, field

from amsrr.controllers.rigid_body_model import RigidBodyControlModel
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.policies import ControllerCommand


CENTROIDAL_EXTERNAL_WRENCH_ESTIMATOR_VERSION = (
    "aggregate_centroidal_momentum_balance_v1"
)
CENTROIDAL_ADMITTANCE_VERSION = "bounded_contact_acquisition_admittance_v1"


@dataclass(frozen=True)
class CentroidalExternalWrenchEstimatorConfig:
    gravity_mps2: float = 9.80665
    wrench_filter_time_constant_s: float = 0.05
    bias_filter_time_constant_s: float = 0.50

    def validate(self) -> None:
        for name in (
            "gravity_mps2",
            "wrench_filter_time_constant_s",
            "bias_filter_time_constant_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"CentroidalExternalWrenchEstimatorConfig.{name} must be "
                    "finite and positive"
                )


@dataclass(frozen=True)
class CentroidalExternalWrenchEstimate:
    valid: bool
    wrench_body: tuple[float, float, float, float, float, float]
    raw_wrench_body: tuple[float, float, float, float, float, float]
    bias_wrench_body: tuple[float, float, float, float, float, float]
    force_norm_n: float
    torque_norm_nm: float
    estimator_version: str = CENTROIDAL_EXTERNAL_WRENCH_ESTIMATOR_VERSION
    failure_reason: str | None = None


class CentroidalExternalWrenchEstimator:
    """Estimate aggregate external wrench from centroidal momentum balance."""

    def __init__(
        self,
        config: CentroidalExternalWrenchEstimatorConfig | None = None,
    ) -> None:
        self.config = config or CentroidalExternalWrenchEstimatorConfig()
        self.config.validate()
        self._filtered_raw_wrench_body = [0.0] * 6
        self._bias_wrench_body = [0.0] * 6
        self._initialized = False

    def reset(self) -> None:
        self._filtered_raw_wrench_body = [0.0] * 6
        self._bias_wrench_body = [0.0] * 6
        self._initialized = False

    def estimate(
        self,
        *,
        previous_model: RigidBodyControlModel,
        current_model: RigidBodyControlModel,
        applied_controller_command: ControllerCommand | None,
        dt_s: float,
        calibrate_bias: bool,
    ) -> CentroidalExternalWrenchEstimate:
        if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
            return _invalid_estimate("invalid_estimator_dt")
        if applied_controller_command is None:
            return _invalid_estimate("applied_controller_command_missing")
        if (
            previous_model.graph_id != current_model.graph_id
            or previous_model.base_module_id != current_model.base_module_id
        ):
            return _invalid_estimate("centroidal_model_identity_changed")

        momentum_rate = _momentum_rate_wrench_body(
            previous_model,
            current_model,
            float(dt_s),
        )
        actuator_wrench = _known_rotor_wrench_body(
            current_model,
            applied_controller_command,
        )
        gravity_wrench = _gravity_wrench_body(
            current_model,
            float(self.config.gravity_mps2),
        )
        raw = [
            momentum_rate[index]
            - actuator_wrench[index]
            - gravity_wrench[index]
            for index in range(6)
        ]
        if not all(math.isfinite(value) for value in raw):
            return _invalid_estimate("non_finite_momentum_balance")

        if not self._initialized:
            self._filtered_raw_wrench_body = list(raw)
            if calibrate_bias:
                self._bias_wrench_body = list(raw)
            self._initialized = True
        else:
            wrench_alpha = _low_pass_alpha(
                float(dt_s),
                float(self.config.wrench_filter_time_constant_s),
            )
            self._filtered_raw_wrench_body = [
                previous + wrench_alpha * (sample - previous)
                for previous, sample in zip(
                    self._filtered_raw_wrench_body,
                    raw,
                    strict=True,
                )
            ]
            if calibrate_bias:
                bias_alpha = _low_pass_alpha(
                    float(dt_s),
                    float(self.config.bias_filter_time_constant_s),
                )
                self._bias_wrench_body = [
                    previous + bias_alpha * (sample - previous)
                    for previous, sample in zip(
                        self._bias_wrench_body,
                        self._filtered_raw_wrench_body,
                        strict=True,
                    )
                ]

        estimate = tuple(
            filtered - bias
            for filtered, bias in zip(
                self._filtered_raw_wrench_body,
                self._bias_wrench_body,
                strict=True,
            )
        )
        return CentroidalExternalWrenchEstimate(
            valid=True,
            wrench_body=estimate,  # type: ignore[arg-type]
            raw_wrench_body=tuple(raw),  # type: ignore[arg-type]
            bias_wrench_body=tuple(self._bias_wrench_body),  # type: ignore[arg-type]
            force_norm_n=_norm(estimate[:3]),
            torque_norm_nm=_norm(estimate[3:]),
        )


@dataclass(frozen=True)
class CentroidalAdmittanceConfig:
    force_deadband_n: float = 0.5
    torque_deadband_nm: float = 0.05
    linear_admittance_mps_per_n: float = 0.0015
    angular_admittance_radps_per_nm: float = 0.03
    maximum_linear_speed_mps: float = 0.020
    maximum_angular_speed_radps: float = 0.15
    maximum_translation_offset_m: float = 0.030

    def validate(self) -> None:
        for name in (
            "force_deadband_n",
            "torque_deadband_nm",
            "linear_admittance_mps_per_n",
            "angular_admittance_radps_per_nm",
            "maximum_linear_speed_mps",
            "maximum_angular_speed_radps",
            "maximum_translation_offset_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"CentroidalAdmittanceConfig.{name} must be finite and "
                    "non-negative"
                )
        for name in (
            "linear_admittance_mps_per_n",
            "angular_admittance_radps_per_nm",
            "maximum_linear_speed_mps",
            "maximum_angular_speed_radps",
            "maximum_translation_offset_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise SchemaValidationError(
                    f"CentroidalAdmittanceConfig.{name} must be positive"
                )


@dataclass(frozen=True)
class CentroidalAdmittanceCommand:
    desired_body_pose: Pose7D
    desired_body_twist: tuple[float, float, float, float, float, float]
    translation_offset_world: tuple[float, float, float]
    active: bool
    version: str = CENTROIDAL_ADMITTANCE_VERSION


class CentroidalAdmittanceController:
    """Convert aggregate external wrench into a bounded yielding body intent."""

    def __init__(self, config: CentroidalAdmittanceConfig | None = None) -> None:
        self.config = config or CentroidalAdmittanceConfig()
        self.config.validate()
        self._translation_offset_world = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        self._translation_offset_world = [0.0, 0.0, 0.0]

    def update(
        self,
        *,
        nominal_pose_world: Pose7D,
        current_pose_world: Pose7D,
        estimate: CentroidalExternalWrenchEstimate,
        dt_s: float,
        active: bool,
        linear_projection_axis_world: tuple[float, float, float] | None = None,
        angular_admittance_enabled: bool = True,
    ) -> CentroidalAdmittanceCommand:
        if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
            raise SchemaValidationError(
                "CentroidalAdmittanceController.dt_s must be finite and positive"
            )
        _validate_pose(nominal_pose_world, "nominal_pose_world")
        _validate_pose(current_pose_world, "current_pose_world")

        if not active:
            self.reset()
            return CentroidalAdmittanceCommand(
                desired_body_pose=nominal_pose_world,
                desired_body_twist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                translation_offset_world=(0.0, 0.0, 0.0),
                active=False,
            )

        wrench = estimate.wrench_body if estimate.valid else (0.0,) * 6
        force_body = _deadband_vector(
            wrench[:3],
            float(self.config.force_deadband_n),
        )
        torque_body = _deadband_vector(
            wrench[3:],
            float(self.config.torque_deadband_nm),
        )
        world_from_body = _quat_to_matrix(_pose_quat(current_pose_world))
        force_world = _matvec(world_from_body, force_body)
        if linear_projection_axis_world is not None:
            projection_axis = _unit_vector(
                linear_projection_axis_world,
                "linear_projection_axis_world",
            )
            projected_force = _dot(force_world, projection_axis)
            force_world = tuple(
                projected_force * value for value in projection_axis
            )
        linear_velocity_world = _limit_norm(
            tuple(
                float(self.config.linear_admittance_mps_per_n) * value
                for value in force_world
            ),
            float(self.config.maximum_linear_speed_mps),
        )
        angular_velocity_body = (
            _limit_norm(
                tuple(
                    float(self.config.angular_admittance_radps_per_nm) * value
                    for value in torque_body
                ),
                float(self.config.maximum_angular_speed_radps),
            )
            if angular_admittance_enabled
            else (0.0, 0.0, 0.0)
        )
        self._translation_offset_world = list(
            _limit_norm(
                tuple(
                    self._translation_offset_world[index]
                    + linear_velocity_world[index] * float(dt_s)
                    for index in range(3)
                ),
                float(self.config.maximum_translation_offset_m),
            )
        )
        desired_pose = (
            float(nominal_pose_world[0]) + self._translation_offset_world[0],
            float(nominal_pose_world[1]) + self._translation_offset_world[1],
            float(nominal_pose_world[2]) + self._translation_offset_world[2],
            *tuple(float(value) for value in nominal_pose_world[3:7]),
        )
        return CentroidalAdmittanceCommand(
            desired_body_pose=desired_pose,
            desired_body_twist=(
                *linear_velocity_world,
                *angular_velocity_body,
            ),
            translation_offset_world=tuple(self._translation_offset_world),
            active=True,
        )


def _invalid_estimate(reason: str) -> CentroidalExternalWrenchEstimate:
    return CentroidalExternalWrenchEstimate(
        valid=False,
        wrench_body=(0.0,) * 6,
        raw_wrench_body=(0.0,) * 6,
        bias_wrench_body=(0.0,) * 6,
        force_norm_n=0.0,
        torque_norm_nm=0.0,
        failure_reason=reason,
    )


def _momentum_rate_wrench_body(
    previous: RigidBodyControlModel,
    current: RigidBodyControlModel,
    dt_s: float,
) -> list[float]:
    current_rotation = _quat_to_matrix(_pose_quat(current.body_pose_world))
    previous_rotation = _quat_to_matrix(_pose_quat(previous.body_pose_world))
    current_body_from_world = _transpose(current_rotation)
    previous_body_from_world = _transpose(previous_rotation)
    linear_acceleration_world = tuple(
        (
            float(current.body_twist_world[index])
            - float(previous.body_twist_world[index])
        )
        / dt_s
        for index in range(3)
    )
    force_body = _scale(
        _matvec(current_body_from_world, linear_acceleration_world),
        float(current.total_mass_kg),
    )
    omega_body = _matvec(
        current_body_from_world,
        tuple(float(value) for value in current.body_twist_world[3:6]),
    )
    previous_omega_body = _matvec(
        previous_body_from_world,
        tuple(float(value) for value in previous.body_twist_world[3:6]),
    )
    alpha_body = tuple(
        (omega_body[index] - previous_omega_body[index]) / dt_s
        for index in range(3)
    )
    inertia = _inertia_matrix(current.inertia_body)
    angular_momentum = _matvec(inertia, omega_body)
    torque_body = _add(
        _matvec(inertia, alpha_body),
        _cross(omega_body, angular_momentum),
    )
    return [*force_body, *torque_body]


def _known_rotor_wrench_body(
    model: RigidBodyControlModel,
    command: ControllerCommand,
) -> list[float]:
    wrench = [0.0] * 6
    for rotor in model.rotor_elements:
        thrust = float(command.rotor_thrusts_n.get(rotor.global_rotor_id, 0.0))
        for index, coefficient in enumerate(rotor.allocation_column_body):
            wrench[index] += float(coefficient) * thrust
    return wrench


def _gravity_wrench_body(
    model: RigidBodyControlModel,
    gravity_mps2: float,
) -> list[float]:
    body_from_world = _transpose(_quat_to_matrix(_pose_quat(model.body_pose_world)))
    force_body = _matvec(
        body_from_world,
        (0.0, 0.0, -float(model.total_mass_kg) * gravity_mps2),
    )
    return [*force_body, 0.0, 0.0, 0.0]


def _deadband_vector(
    values: tuple[float, ...],
    deadband: float,
) -> tuple[float, ...]:
    magnitude = _norm(values)
    if magnitude <= deadband or magnitude <= 1.0e-12:
        return tuple(0.0 for _ in values)
    scale = (magnitude - deadband) / magnitude
    return tuple(scale * float(value) for value in values)


def _limit_norm(
    values: tuple[float, ...],
    maximum: float,
) -> tuple[float, ...]:
    magnitude = _norm(values)
    if magnitude <= maximum or magnitude <= 1.0e-12:
        return tuple(float(value) for value in values)
    scale = maximum / magnitude
    return tuple(scale * float(value) for value in values)


def _low_pass_alpha(dt_s: float, time_constant_s: float) -> float:
    return 1.0 - math.exp(-dt_s / time_constant_s)


def _validate_pose(pose: Pose7D, label: str) -> None:
    if len(pose) != 7 or not all(math.isfinite(float(value)) for value in pose):
        raise SchemaValidationError(f"{label} must contain seven finite values")


def _pose_quat(pose: Pose7D) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in pose[3:7])  # type: ignore[return-value]


def _quat_to_matrix(
    quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise SchemaValidationError("Quaternion norm must be positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _transpose(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(matrix[row][column] for row in range(3)) for column in range(3)
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, ...],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _inertia_matrix(
    values: list[float],
) -> tuple[tuple[float, float, float], ...]:
    ixx, ixy, ixz, iyy, iyz, izz = (float(value) for value in values)
    return (
        (ixx, ixy, ixz),
        (ixy, iyy, iyz),
        (ixz, iyz, izz),
    )


def _scale(
    values: tuple[float, ...],
    scalar: float,
) -> tuple[float, ...]:
    return tuple(float(scalar) * value for value in values)


def _add(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _cross(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _dot(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    return sum(
        float(first[index]) * float(second[index])
        for index in range(len(first))
    )


def _unit_vector(
    values: tuple[float, ...],
    label: str,
) -> tuple[float, float, float]:
    if len(values) != 3 or any(not math.isfinite(float(value)) for value in values):
        raise SchemaValidationError(f"{label} must contain three finite values")
    magnitude = _norm(values)
    if magnitude <= 1.0e-12:
        raise SchemaValidationError(f"{label} must have non-zero norm")
    return tuple(float(value) / magnitude for value in values)  # type: ignore[return-value]


__all__ = [
    "CENTROIDAL_ADMITTANCE_VERSION",
    "CENTROIDAL_EXTERNAL_WRENCH_ESTIMATOR_VERSION",
    "CentroidalAdmittanceCommand",
    "CentroidalAdmittanceConfig",
    "CentroidalAdmittanceController",
    "CentroidalExternalWrenchEstimate",
    "CentroidalExternalWrenchEstimator",
    "CentroidalExternalWrenchEstimatorConfig",
]
