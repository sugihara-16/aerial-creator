from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from amsrr.schemas.common import SchemaBase, SchemaValidationError, Vector3, require_len, require_non_empty, require_non_negative
from amsrr.controllers.rigid_body_model import RigidBodyControlModel, RotorControlElement


QP_INFEASIBLE_CODE = "E_QP_INFEASIBLE"
QP_RIGID_BODY_MODEL_REQUIRED_CODE = "E_QP_RIGID_BODY_MODEL_REQUIRED"
QP_SOLVER_UNAVAILABLE_CODE = "E_QP_SOLVER_UNAVAILABLE"
QP_THRUST_CLIPPED_CODE = "E_QP_THRUST_CLIPPED"
QP_UNSUPPORTED_WRENCH_CODE = "E_QP_UNSUPPORTED_WRENCH"
QP_VECTORING_CLIPPED_CODE = "E_QP_VECTORING_CLIPPED"


@dataclass
class RotorAllocationSpec(SchemaBase):
    rotor_id: str
    thrust_axis_body: Vector3
    thrust_min_n: float
    thrust_max_n: float

    def validate(self) -> None:
        require_non_empty(self.rotor_id, "RotorAllocationSpec.rotor_id")
        require_len(self.thrust_axis_body, 3, "RotorAllocationSpec.thrust_axis_body")
        require_non_negative(self.thrust_min_n, "RotorAllocationSpec.thrust_min_n")
        require_non_negative(self.thrust_max_n, "RotorAllocationSpec.thrust_max_n")
        if self.thrust_max_n < self.thrust_min_n:
            raise SchemaValidationError("RotorAllocationSpec.thrust_max_n must be >= thrust_min_n")


@dataclass
class QPAllocationProblem(SchemaBase):
    desired_wrench_body: list[float] | None
    rotors: list[RotorAllocationSpec]
    rigid_body_model: RigidBodyControlModel | None = None
    previous_rotor_thrusts_n: dict[str, float] = field(default_factory=dict)
    previous_vectoring_joint_targets: dict[str, float] = field(default_factory=dict)
    control_dt_s: float = 0.005
    vertical_tolerance_n: float = 1.0e-6
    unsupported_wrench_tolerance: float = 1.0e-6

    def validate(self) -> None:
        if self.desired_wrench_body is not None:
            require_len(self.desired_wrench_body, 6, "QPAllocationProblem.desired_wrench_body")
        if self.control_dt_s <= 0.0:
            raise SchemaValidationError("QPAllocationProblem.control_dt_s must be positive")
        require_non_negative(self.vertical_tolerance_n, "QPAllocationProblem.vertical_tolerance_n")
        require_non_negative(self.unsupported_wrench_tolerance, "QPAllocationProblem.unsupported_wrench_tolerance")


@dataclass
class QPAllocationResult(SchemaBase):
    rotor_thrusts_n: dict[str, float]
    feasible: bool
    residual_wrench_body: list[float]
    residual_norm: float
    clipped: bool = False
    violation_codes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    vectoring_joint_targets: dict[str, float] = field(default_factory=dict)
    achieved_wrench_body: list[float] = field(default_factory=lambda: [0.0] * 6)

    def validate(self) -> None:
        require_len(self.residual_wrench_body, 6, "QPAllocationResult.residual_wrench_body")
        require_len(self.achieved_wrench_body, 6, "QPAllocationResult.achieved_wrench_body")
        require_non_negative(self.residual_norm, "QPAllocationResult.residual_norm")


class QPAllocatorInterface(Protocol):
    """Backend boundary for exact QP allocators such as OSQP/C++ implementations."""

    def allocate(self, problem: QPAllocationProblem) -> QPAllocationResult:
        ...


class BoundedVerticalRotorAllocator:
    """Dependency-free P1 allocator for vertical wrench support.

    It solves only the scalar vertical thrust allocation with rotor bounds. Other
    wrench components are reported as unsupported residual, leaving exact
    multi-axis/vectoring allocation to a later backend.
    """

    def allocate(self, problem: QPAllocationProblem) -> QPAllocationResult:
        desired = list(problem.desired_wrench_body or [0.0] * 6)
        target_fz = max(0.0, desired[2])
        specs = sorted(problem.rotors, key=lambda rotor: rotor.rotor_id)
        thrusts = {spec.rotor_id: float(spec.thrust_min_n) for spec in specs}
        effectiveness = {spec.rotor_id: abs(float(spec.thrust_axis_body[2])) for spec in specs}
        min_force = sum(thrusts[spec.rotor_id] * effectiveness[spec.rotor_id] for spec in specs)
        max_force = sum(float(spec.thrust_max_n) * effectiveness[spec.rotor_id] for spec in specs)

        clipped = False
        if target_fz > min_force and max_force > min_force:
            remaining = min(target_fz, max_force) - min_force
            total_capacity = sum(
                (float(spec.thrust_max_n) - float(spec.thrust_min_n)) * effectiveness[spec.rotor_id]
                for spec in specs
            )
            if total_capacity > 0.0:
                for spec in specs:
                    rotor_capacity_force = (float(spec.thrust_max_n) - float(spec.thrust_min_n)) * effectiveness[spec.rotor_id]
                    if rotor_capacity_force <= 0.0:
                        continue
                    added_force = remaining * rotor_capacity_force / total_capacity
                    thrusts[spec.rotor_id] += added_force / effectiveness[spec.rotor_id]
        elif target_fz > max_force:
            clipped = True

        if target_fz >= max_force and specs:
            thrusts = {spec.rotor_id: float(spec.thrust_max_n) for spec in specs}
            clipped = target_fz > max_force

        achieved_fz = sum(thrusts[spec.rotor_id] * effectiveness[spec.rotor_id] for spec in specs)
        residual = list(desired)
        residual[2] = target_fz - achieved_fz
        residual_norm = math.sqrt(sum(value * value for value in residual))
        unsupported_norm = math.sqrt(
            residual[0] * residual[0]
            + residual[1] * residual[1]
            + residual[3] * residual[3]
            + residual[4] * residual[4]
            + residual[5] * residual[5]
        )
        feasible = abs(residual[2]) <= problem.vertical_tolerance_n
        violation_codes: list[str] = []
        if not feasible:
            violation_codes.append(QP_INFEASIBLE_CODE)
        if clipped:
            violation_codes.append(QP_THRUST_CLIPPED_CODE)
        if unsupported_norm > problem.unsupported_wrench_tolerance:
            violation_codes.append(QP_UNSUPPORTED_WRENCH_CODE)
        return QPAllocationResult(
            rotor_thrusts_n=thrusts,
            feasible=feasible,
            residual_wrench_body=residual,
            residual_norm=residual_norm,
            clipped=clipped,
            violation_codes=violation_codes,
            metrics={
                "degraded_fallback": 1.0,
                "qp_primary_path": 0.0,
                "target_fz_n": target_fz,
                "achieved_fz_n": achieved_fz,
                "min_fz_n": min_force,
                "max_fz_n": max_force,
                "unsupported_wrench_norm": unsupported_norm,
            },
            achieved_wrench_body=[
                0.0,
                0.0,
                achieved_fz,
                0.0,
                0.0,
                0.0,
            ],
        )


class VirtualThrustQPAllocator:
    """Primary P4-control allocator using virtual x/z thrust channels.

    Vectoring rotors are expanded into fixed rotor-arm x/z force channels during
    optimization, then converted back to non-negative rotor thrust and absolute
    vectoring joint targets.
    """

    regularization_weight: float = 1.0e-5
    previous_command_weight: float = 1.0e-4

    def allocate(self, problem: QPAllocationProblem) -> QPAllocationResult:
        desired = [float(value) for value in (problem.desired_wrench_body or [0.0] * 6)]
        if problem.rigid_body_model is None:
            return _failed_allocation(
                desired,
                [QP_RIGID_BODY_MODEL_REQUIRED_CODE],
                "rigid_body_model_required",
            )
        if not _all_finite(desired):
            return _failed_allocation(desired, [QP_INFEASIBLE_CODE], "non_finite_desired_wrench")

        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, minimize
        except Exception:
            return _failed_allocation(desired, [QP_SOLVER_UNAVAILABLE_CODE], "solver_unavailable")

        variables: list[dict[str, object]] = []
        rotor_channels: dict[str, dict[str, int]] = {}
        columns: list[list[float]] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        previous_values: list[float] = []
        linear_rows: list[list[float]] = []
        linear_lower: list[float] = []
        linear_upper: list[float] = []

        for rotor in sorted(problem.rigid_body_model.rotor_elements, key=lambda item: item.global_rotor_id):
            if _is_vectoring_rotor(rotor):
                joint_id = rotor.vectoring_joint_ids[0]
                angle_bounds = _vectoring_angle_bounds(
                    problem.rigid_body_model,
                    joint_id,
                    problem.control_dt_s,
                )
                fx_idx = len(variables)
                _append_variable(
                    variables,
                    columns,
                    lower_bounds,
                    upper_bounds,
                    previous_values,
                    rotor=rotor,
                    kind="virtual_x",
                    axis_body=rotor.virtual_x_axis_body,
                    lower=-rotor.thrust_max_n,
                    upper=rotor.thrust_max_n,
                    previous_value=_previous_virtual_x(problem, rotor, joint_id),
                )
                fz_idx = len(variables)
                _append_variable(
                    variables,
                    columns,
                    lower_bounds,
                    upper_bounds,
                    previous_values,
                    rotor=rotor,
                    kind="virtual_z",
                    axis_body=rotor.virtual_z_axis_body,
                    lower=_minimum_virtual_z_from_thrust(rotor, angle_bounds),
                    upper=rotor.thrust_max_n,
                    previous_value=_previous_virtual_z(problem, rotor, joint_id),
                )
                rotor_channels[rotor.global_rotor_id] = {"x": fx_idx, "z": fz_idx}
                _append_angle_linear_constraints(
                    linear_rows,
                    linear_lower,
                    linear_upper,
                    variable_count=len(variables),
                    fx_idx=fx_idx,
                    fz_idx=fz_idx,
                    angle_lower=angle_bounds[0],
                    angle_upper=angle_bounds[1],
                )
                continue

            idx = len(variables)
            _append_variable(
                variables,
                columns,
                lower_bounds,
                upper_bounds,
                previous_values,
                rotor=rotor,
                kind="scalar",
                axis_body=rotor.axis_body,
                lower=rotor.thrust_min_n,
                upper=rotor.thrust_max_n,
                previous_value=problem.previous_rotor_thrusts_n.get(rotor.global_rotor_id, rotor.thrust_min_n),
            )
            rotor_channels[rotor.global_rotor_id] = {"scalar": idx}

        if not variables:
            return _failed_allocation(desired, [QP_INFEASIBLE_CODE], "no_rotor_variables")

        allocation_matrix = np.asarray(_columns_to_rows(columns, row_count=6), dtype=float)
        desired_vector = np.asarray(desired, dtype=float)
        previous_vector = np.asarray(previous_values, dtype=float)
        bounds = Bounds(np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float))
        constraints = []
        if linear_rows:
            for row in linear_rows:
                row.extend([0.0] * (len(variables) - len(row)))
            constraints.append(
                LinearConstraint(
                    np.asarray(linear_rows, dtype=float),
                    np.asarray(linear_lower, dtype=float),
                    np.asarray(linear_upper, dtype=float),
                )
            )
        x0 = np.asarray(
            [
                min(max(previous_values[idx], lower_bounds[idx]), upper_bounds[idx])
                for idx in range(len(previous_values))
            ],
            dtype=float,
        )

        regularization = self.regularization_weight
        previous_weight = self.previous_command_weight

        def objective(values) -> float:
            residual = allocation_matrix @ values - desired_vector
            smooth = values - previous_vector
            return float(
                0.5 * (residual @ residual)
                + 0.5 * regularization * (values @ values)
                + 0.5 * previous_weight * (smooth @ smooth)
            )

        def gradient(values):
            residual = allocation_matrix @ values - desired_vector
            return (
                allocation_matrix.T @ residual
                + regularization * values
                + previous_weight * (values - previous_vector)
            )

        solution = minimize(
            objective,
            x0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1.0e-9, "maxiter": 200, "disp": False},
        )
        if not bool(solution.success):
            result = _failed_allocation(
                desired,
                [QP_INFEASIBLE_CODE],
                "solver_failed",
                metrics={
                    "qp_primary_path": 1.0,
                    "degraded_fallback": 0.0,
                    "qp_solver_success": 0.0,
                    "qp_solver_status": float(getattr(solution, "status", -1)),
                },
            )
            return result

        values = [float(value) for value in solution.x]
        rotor_thrusts, vectoring_targets, clipped_count, thrust_clipped, vectoring_clipped = _back_convert_solution(
            problem.rigid_body_model,
            problem,
            values,
            rotor_channels,
        )
        achieved = _achieved_wrench(problem.rigid_body_model, rotor_thrusts, vectoring_targets)
        residual = [desired[idx] - achieved[idx] for idx in range(6)]
        residual_norm = math.sqrt(sum(value * value for value in residual))
        force_residual_norm = math.sqrt(sum(value * value for value in residual[:3]))
        torque_residual_norm = math.sqrt(sum(value * value for value in residual[3:]))
        violation_codes: list[str] = []
        if residual_norm > problem.unsupported_wrench_tolerance:
            violation_codes.append(QP_UNSUPPORTED_WRENCH_CODE)
        if thrust_clipped:
            violation_codes.append(QP_THRUST_CLIPPED_CODE)
        if vectoring_clipped:
            violation_codes.append(QP_VECTORING_CLIPPED_CODE)
        feasible = residual_norm <= problem.unsupported_wrench_tolerance
        if not feasible:
            violation_codes.insert(0, QP_INFEASIBLE_CODE)
        return QPAllocationResult(
            rotor_thrusts_n=rotor_thrusts,
            feasible=feasible,
            residual_wrench_body=residual,
            residual_norm=residual_norm,
            clipped=clipped_count > 0,
            violation_codes=violation_codes,
            metrics={
                "qp_primary_path": 1.0,
                "degraded_fallback": 0.0,
                "qp_solver_success": 1.0,
                "qp_solver_status": float(getattr(solution, "status", 0)),
                "qp_objective": float(solution.fun),
                "virtual_channel_count": float(sum(1 for item in variables if str(item["kind"]).startswith("virtual_"))),
                "allocation_variable_count": float(len(variables)),
                "allocation_residual_norm": residual_norm,
                "force_residual_norm": force_residual_norm,
                "torque_residual_norm": torque_residual_norm,
                "clipped_target_count": float(clipped_count),
                "rotor_saturation_ratio": _rotor_saturation_ratio(problem.rigid_body_model, rotor_thrusts),
                "min_rotor_thrust_margin": _min_rotor_margin(problem.rigid_body_model, rotor_thrusts),
                "min_vectoring_joint_margin": _min_vectoring_margin(problem.rigid_body_model, vectoring_targets),
            },
            vectoring_joint_targets=vectoring_targets,
            achieved_wrench_body=achieved,
        )


def _failed_allocation(
    desired: list[float],
    violation_codes: list[str],
    reason: str,
    metrics: dict[str, float] | None = None,
) -> QPAllocationResult:
    residual = list(desired)
    return QPAllocationResult(
        rotor_thrusts_n={},
        feasible=False,
        residual_wrench_body=residual,
        residual_norm=math.sqrt(sum(value * value for value in residual if math.isfinite(value))),
        clipped=False,
        violation_codes=violation_codes,
        metrics={
            "qp_primary_path": 1.0,
            "degraded_fallback": 0.0,
            "allocation_failure": 1.0,
            "allocation_failure_reason_hash": float(sum(ord(char) for char in reason)),
            **(metrics or {}),
        },
    )


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _is_vectoring_rotor(rotor: RotorControlElement) -> bool:
    return bool(rotor.vectoring_joint_ids and rotor.virtual_x_axis_body is not None and rotor.virtual_z_axis_body is not None)


def _append_variable(
    variables: list[dict[str, object]],
    columns: list[list[float]],
    lower_bounds: list[float],
    upper_bounds: list[float],
    previous_values: list[float],
    *,
    rotor: RotorControlElement,
    kind: str,
    axis_body: Vector3 | None,
    lower: float,
    upper: float,
    previous_value: float,
) -> None:
    if axis_body is None:
        raise SchemaValidationError(f"Rotor {rotor.global_rotor_id!r} is missing axis for {kind}")
    variables.append({"rotor_id": rotor.global_rotor_id, "kind": kind})
    columns.append(_wrench_column(rotor.origin_body, _normalize(axis_body), rotor.reaction_torque_coeff_nm_per_n))
    lower_bounds.append(float(lower))
    upper_bounds.append(float(upper))
    previous_values.append(float(previous_value))


def _append_angle_linear_constraints(
    linear_rows: list[list[float]],
    linear_lower: list[float],
    linear_upper: list[float],
    *,
    variable_count: int,
    fx_idx: int,
    fz_idx: int,
    angle_lower: float,
    angle_upper: float,
) -> None:
    max_angle = math.pi / 2.0 - 1.0e-4
    angle_lower = max(angle_lower, -max_angle)
    angle_upper = min(angle_upper, max_angle)
    if angle_lower > angle_upper:
        mid = 0.5 * (angle_lower + angle_upper)
        angle_lower = angle_upper = min(max(mid, -max_angle), max_angle)

    upper_row = [0.0] * variable_count
    upper_row[fx_idx] = 1.0
    upper_row[fz_idx] = -math.tan(angle_upper)
    linear_rows.append(upper_row)
    linear_lower.append(-math.inf)
    linear_upper.append(0.0)

    lower_row = [0.0] * variable_count
    lower_row[fx_idx] = -1.0
    lower_row[fz_idx] = math.tan(angle_lower)
    linear_rows.append(lower_row)
    linear_lower.append(-math.inf)
    linear_upper.append(0.0)


def _minimum_virtual_z_from_thrust(
    rotor: RotorControlElement,
    angle_bounds: tuple[float, float],
) -> float:
    max_angle = math.pi / 2.0 - 1.0e-4
    max_abs_angle = min(max(abs(angle_bounds[0]), abs(angle_bounds[1])), max_angle)
    return max(0.0, float(rotor.thrust_min_n) * math.cos(max_abs_angle))


def _vectoring_angle_bounds(
    model: RigidBodyControlModel,
    joint_id: str,
    control_dt_s: float,
) -> tuple[float, float]:
    limits = model.active_actuator_limits.get(joint_id, {})
    lower = limits.get("lower")
    upper = limits.get("upper")
    angle_lower = float(lower) if lower is not None else -math.pi / 2.0
    angle_upper = float(upper) if upper is not None else math.pi / 2.0
    velocity = limits.get("velocity")
    if velocity is not None:
        current = float(model.current_joint_positions.get(joint_id, 0.0))
        delta = abs(float(velocity)) * control_dt_s
        angle_lower = max(angle_lower, current - delta)
        angle_upper = min(angle_upper, current + delta)
    if angle_lower > angle_upper:
        current = float(model.current_joint_positions.get(joint_id, 0.0))
        angle_lower = angle_upper = _clip(current, angle_upper, angle_lower)
    return angle_lower, angle_upper


def _previous_virtual_x(problem: QPAllocationProblem, rotor: RotorControlElement, joint_id: str) -> float:
    thrust = float(problem.previous_rotor_thrusts_n.get(rotor.global_rotor_id, rotor.thrust_min_n))
    angle = float(problem.previous_vectoring_joint_targets.get(joint_id, problem.rigid_body_model.current_joint_positions.get(joint_id, 0.0)))  # type: ignore[union-attr]
    return thrust * math.sin(angle)


def _previous_virtual_z(problem: QPAllocationProblem, rotor: RotorControlElement, joint_id: str) -> float:
    thrust = float(problem.previous_rotor_thrusts_n.get(rotor.global_rotor_id, rotor.thrust_min_n))
    angle = float(problem.previous_vectoring_joint_targets.get(joint_id, problem.rigid_body_model.current_joint_positions.get(joint_id, 0.0)))  # type: ignore[union-attr]
    return thrust * math.cos(angle)


def _back_convert_solution(
    model: RigidBodyControlModel,
    problem: QPAllocationProblem,
    values: list[float],
    rotor_channels: dict[str, dict[str, int]],
) -> tuple[dict[str, float], dict[str, float], int, bool, bool]:
    rotor_thrusts: dict[str, float] = {}
    vectoring_targets: dict[str, float] = {}
    clipped_count = 0
    thrust_clipped = False
    vectoring_clipped = False
    rotors_by_id = {rotor.global_rotor_id: rotor for rotor in model.rotor_elements}
    for rotor_id, channels in rotor_channels.items():
        rotor = rotors_by_id[rotor_id]
        if "scalar" in channels:
            raw_thrust = values[channels["scalar"]]
            thrust = _clip(raw_thrust, rotor.thrust_min_n, rotor.thrust_max_n)
            if thrust != raw_thrust:
                clipped_count += 1
                thrust_clipped = True
            rotor_thrusts[rotor_id] = thrust
            continue

        fx = values[channels["x"]]
        fz = max(0.0, values[channels["z"]])
        raw_thrust = math.hypot(fx, fz)
        raw_target = math.atan2(fx, fz)
        thrust = _clip(raw_thrust, rotor.thrust_min_n, rotor.thrust_max_n)
        if thrust != raw_thrust:
            clipped_count += 1
            thrust_clipped = True
        joint_id = rotor.vectoring_joint_ids[0]
        angle_lower, angle_upper = _vectoring_angle_bounds(model, joint_id, problem.control_dt_s)
        target = _clip(raw_target, angle_lower, angle_upper)
        if target != raw_target:
            clipped_count += 1
            vectoring_clipped = True
        rotor_thrusts[rotor_id] = thrust
        vectoring_targets[joint_id] = target
    return rotor_thrusts, vectoring_targets, clipped_count, thrust_clipped, vectoring_clipped


def _achieved_wrench(
    model: RigidBodyControlModel,
    rotor_thrusts: dict[str, float],
    vectoring_targets: dict[str, float],
) -> list[float]:
    wrench = [0.0] * 6
    for rotor in model.rotor_elements:
        thrust = float(rotor_thrusts.get(rotor.global_rotor_id, 0.0))
        if _is_vectoring_rotor(rotor):
            target = float(vectoring_targets.get(rotor.vectoring_joint_ids[0], 0.0))
            axis = _normalize(
                _add(
                    _scale(rotor.virtual_x_axis_body, math.sin(target)),  # type: ignore[arg-type]
                    _scale(rotor.virtual_z_axis_body, math.cos(target)),  # type: ignore[arg-type]
                )
            )
        else:
            axis = rotor.axis_body
        column = _wrench_column(rotor.origin_body, axis, rotor.reaction_torque_coeff_nm_per_n)
        for idx in range(6):
            wrench[idx] += column[idx] * thrust
    return wrench


def _wrench_column(origin_body: Vector3, axis_body: Vector3, reaction_torque_coeff: float) -> list[float]:
    axis = _normalize(axis_body)
    moment = _cross(origin_body, axis)
    reaction = _scale(axis, reaction_torque_coeff)
    torque = _add(moment, reaction)
    return [axis[0], axis[1], axis[2], torque[0], torque[1], torque[2]]


def _rotor_saturation_ratio(model: RigidBodyControlModel, thrusts: dict[str, float]) -> float:
    if not model.rotor_elements:
        return 0.0
    saturated = 0
    for rotor in model.rotor_elements:
        thrust = thrusts.get(rotor.global_rotor_id, 0.0)
        if abs(thrust - rotor.thrust_min_n) <= 1.0e-9 or abs(thrust - rotor.thrust_max_n) <= 1.0e-9:
            saturated += 1
    return saturated / len(model.rotor_elements)


def _min_rotor_margin(model: RigidBodyControlModel, thrusts: dict[str, float]) -> float:
    margins = []
    for rotor in model.rotor_elements:
        thrust = thrusts.get(rotor.global_rotor_id, 0.0)
        margins.append(min(thrust - rotor.thrust_min_n, rotor.thrust_max_n - thrust))
    return min(margins) if margins else 0.0


def _min_vectoring_margin(model: RigidBodyControlModel, targets: dict[str, float]) -> float:
    margins = []
    for joint_id, target in targets.items():
        limits = model.active_actuator_limits.get(joint_id, {})
        lower = limits.get("lower")
        upper = limits.get("upper")
        if lower is None or upper is None:
            continue
        margins.append(min(float(target) - float(lower), float(upper) - float(target)))
    return min(margins) if margins else 0.0


def _columns_to_rows(columns: list[list[float]], *, row_count: int) -> list[list[float]]:
    return [[float(column[row]) for column in columns] for row in range(row_count)]


def _clip(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        lower, upper = upper, lower
    return min(max(float(value), float(lower)), float(upper))


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _normalize(vector: Vector3) -> Vector3:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 0.0:
        raise SchemaValidationError(f"Cannot normalize zero vector {vector!r}")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)
