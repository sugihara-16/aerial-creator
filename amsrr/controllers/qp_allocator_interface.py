from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol

from amsrr.schemas.common import SchemaBase, SchemaValidationError, Vector3, require_len, require_non_empty, require_non_negative


QP_INFEASIBLE_CODE = "E_QP_INFEASIBLE"
QP_THRUST_CLIPPED_CODE = "E_QP_THRUST_CLIPPED"
QP_UNSUPPORTED_WRENCH_CODE = "E_QP_UNSUPPORTED_WRENCH"


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
    previous_rotor_thrusts_n: dict[str, float] = field(default_factory=dict)
    vertical_tolerance_n: float = 1.0e-6
    unsupported_wrench_tolerance: float = 1.0e-6

    def validate(self) -> None:
        if self.desired_wrench_body is not None:
            require_len(self.desired_wrench_body, 6, "QPAllocationProblem.desired_wrench_body")
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

    def validate(self) -> None:
        require_len(self.residual_wrench_body, 6, "QPAllocationResult.residual_wrench_body")
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
        residual_norm = sqrt(sum(value * value for value in residual))
        unsupported_norm = sqrt(
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
                "target_fz_n": target_fz,
                "achieved_fz_n": achieved_fz,
                "min_fz_n": min_force,
                "max_fz_n": max_force,
                "unsupported_wrench_norm": unsupported_norm,
            },
        )
