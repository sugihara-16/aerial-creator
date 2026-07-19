from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL, PolicyCommand


_TASK_DIM = 6
_KNOWN_VECTORING_LOCAL_JOINT_IDS = frozenset({"gimbal1", "gimbal2"})


@dataclass(frozen=True)
class DockJointLimit:
    """Position, rate, and torque limits for one articulated Dock joint."""

    position_lower_rad: float
    position_upper_rad: float
    max_velocity_radps: float
    max_torque_nm: float


@dataclass(frozen=True)
class DockJointVector:
    """Full ordered global Dock-joint state used as the IK variable vector."""

    joint_ids: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_radps: tuple[float, ...]
    neutral_positions_rad: tuple[float, ...]
    limits: tuple[DockJointLimit, ...]
    vectoring_joint_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class AnchorTaskLinearization:
    """A desired-minus-measured 6D anchor error and its full-joint Jacobian."""

    anchor_id: int
    task_error: tuple[float, ...]
    jacobian: tuple[tuple[float, ...], ...]
    wrench_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    task_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class NaturalContactJointControllerConfig:
    control_dt_s: float = 0.02
    task_error_gain_per_s: float = 1.0
    dls_damping: float = 0.05
    neutral_posture_gain_per_s: float = 0.25
    nullspace_velocity_damping: float = 0.05
    max_position_command_lead_rad: float = 0.0065
    reachability_absolute_tolerance: float = 1e-3
    reachability_relative_tolerance: float = 0.10
    minimum_simultaneous_anchor_count: int = 2


@dataclass(frozen=True)
class SimultaneousAnchorReachability:
    passed: bool
    status: str
    anchor_ids: tuple[int, ...]
    minimum_anchor_count: int
    desired_task_rate_norm: float
    residual_norm: float
    relative_residual: float
    tolerance: float
    per_anchor_residual_norm: dict[int, float]


@dataclass(frozen=True)
class JacobianTransposeTorqueMapping:
    unclipped_joint_torque_bias: dict[str, float]
    joint_torque_bias: dict[str, float]
    clipped_joint_ids: tuple[str, ...]


@dataclass(frozen=True)
class NaturalContactJointDiagnostics:
    structural_joint_ids: tuple[str, ...]
    structural_variable_count: int
    jacobian_column_count: int
    task_influential_joint_ids: tuple[str, ...]
    neutral_regularized_joint_ids: tuple[str, ...]
    velocity_clipped_joint_ids: tuple[str, ...]
    position_limited_joint_ids: tuple[str, ...]
    position_command_lead_limited_joint_ids: tuple[str, ...]
    torque_clipped_joint_ids: tuple[str, ...]
    debug_masked_joint_ids: tuple[str, ...]
    debug_mask_applied: bool
    debug_mask_is_non_structural: bool = True


@dataclass(frozen=True)
class NaturalContactJointControlResult:
    policy_command: PolicyCommand
    reachability: SimultaneousAnchorReachability
    diagnostics: NaturalContactJointDiagnostics
    torque_mapping: JacobianTransposeTorqueMapping


class NaturalContactJointController:
    """Deterministic whole-structure Dock IK and local torque-bias mapper.

    All columns of :class:`DockJointVector` remain in the damped least-squares
    problem.  A zero Jacobian column is handled by nullspace neutral-posture
    regularization; it is never removed or structurally locked.
    """

    def __init__(self, config: NaturalContactJointControllerConfig | None = None) -> None:
        self.config = config or NaturalContactJointControllerConfig()
        _validate_config(self.config)

    def compute(
        self,
        joint_vector: DockJointVector,
        anchor_tasks: Sequence[AnchorTaskLinearization],
        *,
        desired_body_pose: Pose7D | None = None,
        residual_wrench_body: Sequence[float] | None = None,
        priority_weights: Mapping[str, float] | None = None,
        debug_command_mask: Iterable[str] | None = None,
        position_reference_rad: Mapping[str, float] | None = None,
    ) -> NaturalContactJointControlResult:
        """Build an absolute v2 local-joint command and reachability gate.

        ``debug_command_mask`` is deliberately command-only: masked joints emit
        a neutral position hold with zero velocity and torque bias while staying
        in the structural variable set and every output mapping.
        """

        tasks = tuple(sorted(anchor_tasks, key=lambda task: task.anchor_id))
        mask = frozenset(debug_command_mask or ())
        _validate_inputs(joint_vector, tasks, mask)
        np = _numpy()

        joint_count = len(joint_vector.joint_ids)
        q = np.asarray(joint_vector.positions_rad, dtype=float)
        qdot = np.asarray(joint_vector.velocities_radps, dtype=float)
        q_neutral = np.asarray(joint_vector.neutral_positions_rad, dtype=float)
        if position_reference_rad is None:
            q_reference = q.copy()
        else:
            if set(position_reference_rad) != set(joint_vector.joint_ids):
                raise SchemaValidationError(
                    "position_reference_rad must cover exactly the Dock joint vector"
                )
            reference_values = tuple(
                float(position_reference_rad[joint_id])
                for joint_id in joint_vector.joint_ids
            )
            _require_finite(reference_values, "position_reference_rad")
            q_reference = np.asarray(reference_values, dtype=float)

        stacked_jacobian, desired_task_rate = _stack_weighted_tasks(
            np,
            tasks,
            joint_count=joint_count,
            task_gain=self.config.task_error_gain_per_s,
        )
        if tasks:
            normal = stacked_jacobian.T @ stacked_jacobian
            normal += (self.config.dls_damping**2) * np.eye(joint_count)
            damped_inverse = np.linalg.solve(normal, stacked_jacobian.T)
            task_joint_rate = damped_inverse @ desired_task_rate
            # The damped inverse is deliberately used for the primary task,
            # but it is not a valid null-space projector: ``J @ (I - J#J)``
            # remains non-zero when ``J#`` contains damping.  With a slowly
            # advancing contact target that leakage allowed the neutral-
            # posture term to reverse the requested surface motion.  Build
            # the secondary projector from the Moore-Penrose inverse instead
            # so posture regularization cannot alter an achievable primary
            # anchor velocity.
            nullspace = np.eye(joint_count) - np.linalg.pinv(
                stacked_jacobian
            ) @ stacked_jacobian
        else:
            task_joint_rate = np.zeros(joint_count, dtype=float)
            nullspace = np.eye(joint_count)

        neutral_joint_rate = self.config.neutral_posture_gain_per_s * (q_neutral - q)
        neutral_joint_rate -= self.config.nullspace_velocity_damping * qdot
        unconstrained_joint_rate = task_joint_rate + nullspace @ neutral_joint_rate

        (
            bounded_joint_rate,
            position_targets,
            velocity_clipped,
            position_limited,
            position_lead_limited,
        ) = (
            _bound_joint_motion(
                np,
                joint_vector,
                unconstrained_joint_rate,
                position_reference=q_reference,
                dt_s=self.config.control_dt_s,
                max_position_command_lead_rad=(
                    self.config.max_position_command_lead_rad
                ),
            )
        )
        torque_mapping = self._map_anchor_wrenches_validated(
            np,
            joint_vector,
            tasks,
        )

        masked_indices = {index for index, joint_id in enumerate(joint_vector.joint_ids) if joint_id in mask}
        torque_values = [
            torque_mapping.joint_torque_bias[joint_id]
            for joint_id in joint_vector.joint_ids
        ]
        unclipped_torque_values = dict(
            torque_mapping.unclipped_joint_torque_bias
        )
        for index in masked_indices:
            limit = joint_vector.limits[index]
            position_targets[index] = _clip(
                joint_vector.neutral_positions_rad[index],
                limit.position_lower_rad,
                limit.position_upper_rad,
            )
            bounded_joint_rate[index] = 0.0
            torque_values[index] = 0.0
            unclipped_torque_values[joint_vector.joint_ids[index]] = 0.0
        torque_mapping = JacobianTransposeTorqueMapping(
            unclipped_joint_torque_bias=unclipped_torque_values,
            joint_torque_bias={
                joint_id: float(torque_values[index])
                for index, joint_id in enumerate(joint_vector.joint_ids)
            },
            clipped_joint_ids=tuple(
                joint_id
                for joint_id in torque_mapping.clipped_joint_ids
                if joint_id not in mask
            ),
        )

        reachability = _evaluate_reachability(
            np,
            tasks,
            stacked_jacobian,
            desired_task_rate,
            bounded_joint_rate,
            config=self.config,
        )
        position_map = {
            joint_id: float(position_targets[index])
            for index, joint_id in enumerate(joint_vector.joint_ids)
        }
        velocity_map = {
            joint_id: float(bounded_joint_rate[index])
            for index, joint_id in enumerate(joint_vector.joint_ids)
        }
        torque_map = {
            joint_id: float(torque_values[index])
            for index, joint_id in enumerate(joint_vector.joint_ids)
        }

        residual = None
        if residual_wrench_body is not None:
            residual = [float(value) for value in residual_wrench_body]
            _require_vector(residual, _TASK_DIM, "residual_wrench_body")
        weights = {str(key): float(value) for key, value in (priority_weights or {}).items()}
        _require_finite(weights.values(), "priority_weights")
        command = PolicyCommand(
            desired_body_pose=desired_body_pose,
            residual_wrench_body=residual,
            priority_weights=weights,
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=position_map,
            joint_velocity_targets=velocity_map,
            joint_torque_bias=torque_map,
        )
        command.validate()

        influential_indices = {
            column
            for task in tasks
            for row in task.jacobian
            for column, value in enumerate(row)
            if abs(float(value)) > 1e-12
        }
        structural_ids = joint_vector.joint_ids
        diagnostics = NaturalContactJointDiagnostics(
            structural_joint_ids=structural_ids,
            structural_variable_count=joint_count,
            jacobian_column_count=joint_count,
            task_influential_joint_ids=tuple(
                structural_ids[index] for index in sorted(influential_indices)
            ),
            neutral_regularized_joint_ids=tuple(
                joint_id
                for index, joint_id in enumerate(structural_ids)
                if index not in influential_indices
            ),
            velocity_clipped_joint_ids=tuple(
                structural_ids[index] for index in velocity_clipped
            ),
            position_limited_joint_ids=tuple(
                structural_ids[index] for index in position_limited
            ),
            position_command_lead_limited_joint_ids=tuple(
                structural_ids[index] for index in position_lead_limited
            ),
            torque_clipped_joint_ids=torque_mapping.clipped_joint_ids,
            debug_masked_joint_ids=tuple(sorted(mask)),
            debug_mask_applied=bool(mask),
        )
        return NaturalContactJointControlResult(
            policy_command=command,
            reachability=reachability,
            diagnostics=diagnostics,
            torque_mapping=torque_mapping,
        )

    def map_anchor_wrenches(
        self,
        joint_vector: DockJointVector,
        anchor_tasks: Sequence[AnchorTaskLinearization],
        *,
        debug_command_mask: Iterable[str] | None = None,
    ) -> JacobianTransposeTorqueMapping:
        """Map anchor wrench biases through ``sum(J_anchor.T @ wrench)``."""

        tasks = tuple(sorted(anchor_tasks, key=lambda task: task.anchor_id))
        mask = frozenset(debug_command_mask or ())
        _validate_inputs(joint_vector, tasks, mask)
        mapping = self._map_anchor_wrenches_validated(_numpy(), joint_vector, tasks)
        if not mask:
            return mapping
        unclipped_values = dict(mapping.unclipped_joint_torque_bias)
        values = dict(mapping.joint_torque_bias)
        for joint_id in mask:
            unclipped_values[joint_id] = 0.0
            values[joint_id] = 0.0
        return JacobianTransposeTorqueMapping(
            unclipped_joint_torque_bias=unclipped_values,
            joint_torque_bias=values,
            clipped_joint_ids=mapping.clipped_joint_ids,
        )

    @staticmethod
    def _map_anchor_wrenches_validated(
        np: object,
        joint_vector: DockJointVector,
        tasks: Sequence[AnchorTaskLinearization],
    ) -> JacobianTransposeTorqueMapping:
        joint_count = len(joint_vector.joint_ids)
        torque = np.zeros(joint_count, dtype=float)
        for task in tasks:
            jacobian = np.asarray(task.jacobian, dtype=float)
            wrench = np.asarray(task.wrench_bias, dtype=float)
            torque += jacobian.T @ wrench

        clipped: list[str] = []
        unclipped_result: dict[str, float] = {}
        result: dict[str, float] = {}
        for index, joint_id in enumerate(joint_vector.joint_ids):
            limit = joint_vector.limits[index].max_torque_nm
            unclipped_result[joint_id] = float(torque[index])
            value = _clip(float(torque[index]), -limit, limit)
            if not math.isclose(value, float(torque[index]), rel_tol=0.0, abs_tol=1e-12):
                clipped.append(joint_id)
            result[joint_id] = value
        return JacobianTransposeTorqueMapping(
            unclipped_joint_torque_bias=unclipped_result,
            joint_torque_bias=result,
            clipped_joint_ids=tuple(clipped),
        )


def position_drive_peak_effort_lead_rad(
    *,
    stiffness_nm_per_rad: float,
    peak_effort_nm: float,
) -> float:
    """Return the largest static position error inside the drive hard limit.

    This is a simulator/local-servo tuning bound, not a manufacturer stiffness
    claim.  The articulation effort/current-equivalent clamp remains the final
    authority when damping and offset torque are present.
    """

    stiffness = float(stiffness_nm_per_rad)
    peak = float(peak_effort_nm)
    if not math.isfinite(stiffness) or stiffness <= 0.0:
        raise SchemaValidationError(
            "position-drive stiffness must be finite and positive"
        )
    if not math.isfinite(peak) or peak <= 0.0:
        raise SchemaValidationError(
            "position-drive peak effort must be finite and positive"
        )
    return peak / stiffness


def _stack_weighted_tasks(
    np: object,
    tasks: Sequence[AnchorTaskLinearization],
    *,
    joint_count: int,
    task_gain: float,
) -> tuple[object, object]:
    if not tasks:
        return np.zeros((0, joint_count), dtype=float), np.zeros(0, dtype=float)
    jacobian_blocks: list[object] = []
    target_blocks: list[object] = []
    for task in tasks:
        weights = np.sqrt(np.asarray(task.task_weights, dtype=float))
        jacobian_blocks.append(weights[:, None] * np.asarray(task.jacobian, dtype=float))
        target_blocks.append(weights * task_gain * np.asarray(task.task_error, dtype=float))
    return np.vstack(jacobian_blocks), np.concatenate(target_blocks)


def _bound_joint_motion(
    np: object,
    joint_vector: DockJointVector,
    unconstrained_joint_rate: object,
    *,
    position_reference: object,
    dt_s: float,
    max_position_command_lead_rad: float,
) -> tuple[
    object,
    object,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    bounded = np.asarray(unconstrained_joint_rate, dtype=float).copy()
    measured_positions = np.asarray(joint_vector.positions_rad, dtype=float)
    reference_positions = np.asarray(position_reference, dtype=float).copy()
    if reference_positions.shape != measured_positions.shape:
        raise SchemaValidationError(
            "position reference must match the Dock joint vector shape"
        )
    velocity_clipped: list[int] = []
    position_limited: list[int] = []
    position_lead_limited: list[int] = []
    for index, limit in enumerate(joint_vector.limits):
        reference_positions[index] = _clip(
            float(reference_positions[index]),
            limit.position_lower_rad,
            limit.position_upper_rad,
        )
        lower_from_position = (
            limit.position_lower_rad - reference_positions[index]
        ) / dt_s
        upper_from_position = (
            limit.position_upper_rad - reference_positions[index]
        ) / dt_s
        lower = max(-limit.max_velocity_radps, lower_from_position)
        upper = min(limit.max_velocity_radps, upper_from_position)
        raw = float(bounded[index])
        value = _clip(raw, lower, upper)
        if not math.isclose(value, raw, rel_tol=0.0, abs_tol=1e-12):
            velocity_clipped.append(index)
            if raw < lower_from_position or raw > upper_from_position:
                position_limited.append(index)
        bounded[index] = value
    targets = reference_positions + dt_s * bounded
    for index, limit in enumerate(joint_vector.limits):
        lower = max(
            limit.position_lower_rad,
            float(measured_positions[index])
            - float(max_position_command_lead_rad),
        )
        upper = min(
            limit.position_upper_rad,
            float(measured_positions[index])
            + float(max_position_command_lead_rad),
        )
        raw_target = float(targets[index])
        targets[index] = _clip(raw_target, lower, upper)
        if not math.isclose(
            float(targets[index]), raw_target, rel_tol=0.0, abs_tol=1e-12
        ):
            position_lead_limited.append(index)
            # The absolute position reference may lag the measured joint after
            # contact has passively moved the mechanism.  Correcting the
            # position target back into the measured-position lead envelope is
            # a position-servo safety operation; it must not be reinterpreted
            # as a velocity command.  Doing so can exceed max_velocity_radps
            # whenever the two admissible envelopes do not overlap.
            #
            # Keep the independently bounded velocity feed-forward here.  The
            # simulator enforces the position target through its effort-limited
            # drive, while this channel remains inside the motor speed limit.
    return (
        bounded,
        targets,
        tuple(velocity_clipped),
        tuple(position_limited),
        tuple(position_lead_limited),
    )


def _evaluate_reachability(
    np: object,
    tasks: Sequence[AnchorTaskLinearization],
    stacked_jacobian: object,
    desired_task_rate: object,
    bounded_joint_rate: object,
    *,
    config: NaturalContactJointControllerConfig,
) -> SimultaneousAnchorReachability:
    anchor_ids = tuple(task.anchor_id for task in tasks)
    desired_norm = float(np.linalg.norm(desired_task_rate))
    residual = desired_task_rate - stacked_jacobian @ bounded_joint_rate
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / desired_norm if desired_norm > 1e-12 else (0.0 if residual_norm <= 1e-12 else math.inf)
    tolerance = (
        config.reachability_absolute_tolerance
        + config.reachability_relative_tolerance * desired_norm
    )
    enough_anchors = len(tasks) >= config.minimum_simultaneous_anchor_count
    passed = enough_anchors and residual_norm <= tolerance
    if not enough_anchors:
        status = "insufficient_anchor_count"
    elif passed:
        status = "reachable"
    else:
        status = "unreachable_residual"

    per_anchor: dict[int, float] = {}
    for index, task in enumerate(tasks):
        predicted = np.asarray(task.jacobian, dtype=float) @ bounded_joint_rate
        desired = config.task_error_gain_per_s * np.asarray(task.task_error, dtype=float)
        per_anchor[task.anchor_id] = float(np.linalg.norm(desired - predicted))
    return SimultaneousAnchorReachability(
        passed=passed,
        status=status,
        anchor_ids=anchor_ids,
        minimum_anchor_count=config.minimum_simultaneous_anchor_count,
        desired_task_rate_norm=desired_norm,
        residual_norm=residual_norm,
        relative_residual=relative,
        tolerance=tolerance,
        per_anchor_residual_norm=per_anchor,
    )


def _validate_config(config: NaturalContactJointControllerConfig) -> None:
    positive = {
        "control_dt_s": config.control_dt_s,
        "task_error_gain_per_s": config.task_error_gain_per_s,
        "dls_damping": config.dls_damping,
        "max_position_command_lead_rad": (
            config.max_position_command_lead_rad
        ),
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise SchemaValidationError(f"{name} must be finite and positive")
    nonnegative = {
        "neutral_posture_gain_per_s": config.neutral_posture_gain_per_s,
        "nullspace_velocity_damping": config.nullspace_velocity_damping,
        "reachability_absolute_tolerance": config.reachability_absolute_tolerance,
        "reachability_relative_tolerance": config.reachability_relative_tolerance,
    }
    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise SchemaValidationError(f"{name} must be finite and non-negative")
    if config.minimum_simultaneous_anchor_count < 2:
        raise SchemaValidationError("minimum_simultaneous_anchor_count must be at least two")


def _validate_inputs(
    joint_vector: DockJointVector,
    tasks: Sequence[AnchorTaskLinearization],
    debug_mask: frozenset[str],
) -> None:
    joint_count = len(joint_vector.joint_ids)
    if joint_count == 0:
        raise SchemaValidationError("DockJointVector must contain at least one Dock joint")
    if len(set(joint_vector.joint_ids)) != joint_count:
        raise SchemaValidationError("DockJointVector.joint_ids must be unique")
    if any(not joint_id for joint_id in joint_vector.joint_ids):
        raise SchemaValidationError("DockJointVector.joint_ids must be non-empty")
    for name, values in (
        ("positions_rad", joint_vector.positions_rad),
        ("velocities_radps", joint_vector.velocities_radps),
        ("neutral_positions_rad", joint_vector.neutral_positions_rad),
        ("limits", joint_vector.limits),
    ):
        if len(values) != joint_count:
            raise SchemaValidationError(f"DockJointVector.{name} must have {joint_count} entries")
    _require_finite(joint_vector.positions_rad, "DockJointVector.positions_rad")
    _require_finite(joint_vector.velocities_radps, "DockJointVector.velocities_radps")
    _require_finite(joint_vector.neutral_positions_rad, "DockJointVector.neutral_positions_rad")

    declared_vectoring = set(joint_vector.vectoring_joint_ids)
    for joint_id in joint_vector.joint_ids:
        local_id = _local_joint_id(joint_id)
        if (
            joint_id in declared_vectoring
            or local_id in declared_vectoring
            or local_id in _KNOWN_VECTORING_LOCAL_JOINT_IDS
            or "vectoring" in local_id.lower()
        ):
            raise SchemaValidationError(
                f"Vectoring joint {joint_id!r} is not a Dock IK variable"
            )

    for index, limit in enumerate(joint_vector.limits):
        values = (
            limit.position_lower_rad,
            limit.position_upper_rad,
            limit.max_velocity_radps,
            limit.max_torque_nm,
        )
        _require_finite(values, f"DockJointVector.limits[{index}]")
        if limit.position_lower_rad >= limit.position_upper_rad:
            raise SchemaValidationError("Dock joint position range must be non-empty")
        if limit.max_velocity_radps <= 0.0 or limit.max_torque_nm <= 0.0:
            raise SchemaValidationError("Dock joints must remain physically movable with positive limits")
        q = joint_vector.positions_rad[index]
        neutral = joint_vector.neutral_positions_rad[index]
        if not limit.position_lower_rad <= q <= limit.position_upper_rad:
            raise SchemaValidationError(f"Current position for {joint_vector.joint_ids[index]!r} is outside limits")
        if not limit.position_lower_rad <= neutral <= limit.position_upper_rad:
            raise SchemaValidationError(f"Neutral position for {joint_vector.joint_ids[index]!r} is outside limits")

    unknown_mask = debug_mask - set(joint_vector.joint_ids)
    if unknown_mask:
        raise SchemaValidationError(f"Debug command mask contains unknown joints: {sorted(unknown_mask)}")

    anchor_ids = [task.anchor_id for task in tasks]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise SchemaValidationError("AnchorTaskLinearization.anchor_id values must be unique")
    for task in tasks:
        if task.anchor_id < 0:
            raise SchemaValidationError("AnchorTaskLinearization.anchor_id must be non-negative")
        _require_vector(task.task_error, _TASK_DIM, f"anchor[{task.anchor_id}].task_error")
        _require_vector(task.wrench_bias, _TASK_DIM, f"anchor[{task.anchor_id}].wrench_bias")
        _require_vector(task.task_weights, _TASK_DIM, f"anchor[{task.anchor_id}].task_weights")
        if any(weight <= 0.0 for weight in task.task_weights):
            raise SchemaValidationError("Anchor task weights must be positive")
        if len(task.jacobian) != _TASK_DIM:
            raise SchemaValidationError(f"anchor[{task.anchor_id}].jacobian must have six rows")
        for row in task.jacobian:
            if len(row) != joint_count:
                raise SchemaValidationError(
                    f"anchor[{task.anchor_id}].jacobian rows must have {joint_count} columns"
                )
            _require_finite(row, f"anchor[{task.anchor_id}].jacobian")


def _require_vector(values: Sequence[float], length: int, name: str) -> None:
    if len(values) != length:
        raise SchemaValidationError(f"{name} must have length {length}")
    _require_finite(values, name)


def _require_finite(values: Iterable[float], name: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise SchemaValidationError(f"{name} must contain finite values")


def _local_joint_id(global_joint_id: str) -> str:
    return global_joint_id.rsplit(":", 1)[-1]


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _numpy() -> object:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - environment contract guard
        raise RuntimeError("NaturalContactJointController requires numpy at compute time") from exc
    return np
