from __future__ import annotations

import math
from collections.abc import Iterable

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ControllerCommand, ControllerStatus


_COMMAND_FIELDS = (
    "rotor_thrusts_n",
    "vectoring_joint_targets",
    "joint_torque_commands",
    "dock_mechanism_commands",
    "joint_position_targets",
    "joint_velocity_targets",
    "joint_torque_bias",
)
_STATUS_SEVERITY = {
    "ok": 0,
    "warning": 1,
    "infeasible": 2,
    "fault": 3,
}


def merge_disjoint_controller_commands(
    commands: Iterable[ControllerCommand],
) -> ControllerCommand:
    """Merge component commands that address disjoint global actuators.

    This is used only at an already-verified physical control handover.  A
    duplicate command key is rejected instead of selecting one controller
    implicitly.
    """

    items = list(commands)
    if not items:
        raise SchemaValidationError("controller handover requires at least one command")
    contract = items[0].control_contract_version
    if any(item.control_contract_version != contract for item in items[1:]):
        raise SchemaValidationError("controller handover contract versions do not match")
    merged: dict[str, dict[str, float]] = {field: {} for field in _COMMAND_FIELDS}
    for item in items:
        for field in _COMMAND_FIELDS:
            destination = merged[field]
            for key, value in getattr(item, field).items():
                if key in destination:
                    raise SchemaValidationError(
                        f"controller handover has duplicate {field} key {key!r}"
                    )
                destination[key] = float(value)
    feasible = all(item.controller_status.qp_feasible for item in items)
    status_name = max(
        (item.controller_status.status for item in items),
        key=_STATUS_SEVERITY.__getitem__,
    )
    if not feasible and _STATUS_SEVERITY[status_name] < _STATUS_SEVERITY["infeasible"]:
        status_name = "infeasible"
    residuals = [_allocation_residual(item) for item in items]
    status = ControllerStatus(
        status=status_name,
        qp_feasible=feasible,
        active_mode="component_command_merge",
        message=(
            None
            if status_name == "ok"
            else "merged component controller status: " + status_name
        ),
        metrics={
            "merged_component_count": float(len(items)),
            "allocation_residual_norm": max(residuals, default=0.0),
            "warning_endpoint_count": float(
                sum(item.controller_status.status == "warning" for item in items)
            ),
            "non_ok_endpoint_count": float(
                sum(item.controller_status.status != "ok" for item in items)
            ),
        },
    )
    result = ControllerCommand(
        **merged,
        controller_status=status,
        control_contract_version=contract,
    )
    result.validate()
    return result


def blend_controller_commands(
    source: ControllerCommand,
    target: ControllerCommand,
    alpha: float,
) -> ControllerCommand:
    """Linearly blend two complete commands over the same actuator domain.

    Both sides must command exactly the same keys in every channel.  This
    makes a controller-topology change explicit and prevents an actuator from
    silently retaining a stale target during the handover.
    """

    if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise SchemaValidationError("controller handover alpha must be finite in [0, 1]")
    if source.control_contract_version != target.control_contract_version:
        raise SchemaValidationError("controller handover contract versions do not match")
    blended: dict[str, dict[str, float]] = {}
    ratio = float(alpha)
    for field in _COMMAND_FIELDS:
        source_values = getattr(source, field)
        target_values = getattr(target, field)
        if set(source_values) != set(target_values):
            raise SchemaValidationError(
                f"controller handover actuator domain mismatch in {field}"
            )
        blended[field] = {
            key: (1.0 - ratio) * float(source_values[key])
            + ratio * float(target_values[key])
            for key in sorted(source_values)
        }
    feasible = source.controller_status.qp_feasible and target.controller_status.qp_feasible
    status_name = max(
        (source.controller_status.status, target.controller_status.status),
        key=_STATUS_SEVERITY.__getitem__,
    )
    if not feasible and _STATUS_SEVERITY[status_name] < _STATUS_SEVERITY["infeasible"]:
        status_name = "infeasible"
    source_residual = _allocation_residual(source)
    target_residual = _allocation_residual(target)
    status = ControllerStatus(
        status=status_name,
        qp_feasible=feasible,
        active_mode="controller_command_blend",
        message=(
            None
            if status_name == "ok"
            else "handover endpoint controller status: " + status_name
        ),
        metrics={
            "handover_alpha": ratio,
            "source_allocation_residual_norm": source_residual,
            "target_allocation_residual_norm": target_residual,
            "allocation_residual_norm": max(source_residual, target_residual),
            "non_ok_endpoint_count": float(
                (source.controller_status.status != "ok")
                + (target.controller_status.status != "ok")
            ),
        },
    )
    result = ControllerCommand(
        **blended,
        controller_status=status,
        control_contract_version=source.control_contract_version,
    )
    result.validate()
    return result


def _allocation_residual(command: ControllerCommand) -> float:
    metrics = command.controller_status.metrics
    return float(
        metrics.get(
            "allocation_residual_norm",
            metrics.get("residual_norm", 0.0),
        )
    )


__all__ = [
    "blend_controller_commands",
    "merge_disjoint_controller_commands",
]
