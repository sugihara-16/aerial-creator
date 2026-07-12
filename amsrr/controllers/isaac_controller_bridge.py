from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amsrr.controllers.actuator_mapping import ActuatorChannel, ActuatorMapping, clip_to_channel
from amsrr.schemas.common import SchemaBase, require_len, require_non_empty
from amsrr.schemas.policies import ControllerCommand


SUPPORTED_COMMAND_TYPES = {
    "rotor_thrust": "rotor_thrust",
    "vectoring_joint_position": "vectoring_joint_position",
    "dock_joint_position": "dock_joint_position",
    "joint_effort": "joint_effort",
    "joint_position": "joint_position",
    "joint_velocity": "joint_velocity",
    "joint_effort_bias": "joint_effort_bias",
}


@dataclass(frozen=True)
class IsaacControllerBridgeConfig:
    backend: str = "isaac_lab"
    bridge_version: str = "isaac_controller_bridge_v2"


@dataclass
class IsaacActuatorTarget(SchemaBase):
    actuator_id: str
    isaac_target_name: str
    actuator_type: str
    command_key: str
    target_value: float
    unclipped_value: float
    clipped: bool = False
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.actuator_id, "IsaacActuatorTarget.actuator_id")
        require_non_empty(self.isaac_target_name, "IsaacActuatorTarget.isaac_target_name")
        require_non_empty(self.actuator_type, "IsaacActuatorTarget.actuator_type")
        require_non_empty(self.command_key, "IsaacActuatorTarget.command_key")


@dataclass
class IsaacActuatorTargetRecord(SchemaBase):
    time_s: float
    backend: str
    morphology_graph_id: str
    command_index: int
    actuator_targets: list[IsaacActuatorTarget]
    clipped_targets: list[str] = field(default_factory=list)
    missing_actuators: list[str] = field(default_factory=list)
    unsupported_actuators: list[str] = field(default_factory=list)
    allocation_residual_norm: float = 0.0
    qp_status: str = "unknown"
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.backend, "IsaacActuatorTargetRecord.backend")
        require_non_empty(self.morphology_graph_id, "IsaacActuatorTargetRecord.morphology_graph_id")
        if self.time_s < 0.0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("IsaacActuatorTargetRecord.time_s must be non-negative")
        if self.command_index < 0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("IsaacActuatorTargetRecord.command_index must be non-negative")
        require_len([self.allocation_residual_norm], 1, "IsaacActuatorTargetRecord.allocation_residual_norm")


class IsaacControllerBridge:
    """Convert controller-owned commands into Isaac actuator target records."""

    def __init__(self, config: IsaacControllerBridgeConfig | None = None) -> None:
        self.config = config or IsaacControllerBridgeConfig()

    def convert(
        self,
        controller_command: ControllerCommand,
        actuator_mapping: ActuatorMapping,
        *,
        time_s: float,
        command_index: int,
    ) -> IsaacActuatorTargetRecord:
        actuator_targets: list[IsaacActuatorTarget] = []
        clipped_targets: list[str] = []
        missing_actuators: list[str] = []
        unsupported_actuators: list[str] = []

        self._append_command_targets(
            controller_command.rotor_thrusts_n,
            actuator_mapping,
            expected_type="rotor_thrust",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )
        self._append_command_targets(
            controller_command.vectoring_joint_targets,
            actuator_mapping,
            expected_type="vectoring_joint_position",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )
        self._append_command_targets(
            controller_command.dock_mechanism_commands,
            actuator_mapping,
            expected_type="dock_joint_position",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )
        self._append_command_targets(
            controller_command.joint_torque_commands,
            actuator_mapping,
            expected_type="joint_effort",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )
        self._append_command_targets(
            controller_command.joint_position_targets,
            actuator_mapping,
            expected_type="joint_position",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )
        self._append_command_targets(
            controller_command.joint_velocity_targets,
            actuator_mapping,
            expected_type="joint_velocity",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )
        self._append_command_targets(
            controller_command.joint_torque_bias,
            actuator_mapping,
            expected_type="joint_effort_bias",
            actuator_targets=actuator_targets,
            clipped_targets=clipped_targets,
            missing_actuators=missing_actuators,
            unsupported_actuators=unsupported_actuators,
        )

        status_metrics = controller_command.controller_status.metrics
        allocation_residual_norm = float(
            status_metrics.get("allocation_residual_norm", status_metrics.get("residual_norm", 0.0))
        )
        commanded_count = (
            len(controller_command.rotor_thrusts_n)
            + len(controller_command.vectoring_joint_targets)
            + len(controller_command.dock_mechanism_commands)
            + len(controller_command.joint_torque_commands)
            + len(controller_command.joint_position_targets)
            + len(controller_command.joint_velocity_targets)
            + len(controller_command.joint_torque_bias)
        )
        return IsaacActuatorTargetRecord(
            time_s=time_s,
            backend=self.config.backend,
            morphology_graph_id=actuator_mapping.graph_id,
            command_index=command_index,
            actuator_targets=actuator_targets,
            clipped_targets=sorted(clipped_targets),
            missing_actuators=sorted(missing_actuators),
            unsupported_actuators=sorted(unsupported_actuators),
            allocation_residual_norm=allocation_residual_norm,
            qp_status=controller_command.controller_status.status,
            metrics={
                "active_actuator_count": float(len(actuator_mapping.channels)),
                "commanded_actuator_count": float(commanded_count),
                "actuator_target_count": float(len(actuator_targets)),
                "clipped_target_count": float(len(clipped_targets)),
                "missing_actuator_count": float(len(missing_actuators)),
                "unsupported_actuator_count": float(len(unsupported_actuators)),
                "controller_infeasible": 0.0 if controller_command.controller_status.qp_feasible else 1.0,
                "allocation_residual_norm": allocation_residual_norm,
            },
            metadata={
                "bridge_version": self.config.bridge_version,
                "controller_active_mode": controller_command.controller_status.active_mode,
                "control_contract_version": controller_command.control_contract_version,
            },
        )

    @staticmethod
    def _append_command_targets(
        command_values: dict[str, float],
        actuator_mapping: ActuatorMapping,
        *,
        expected_type: str,
        actuator_targets: list[IsaacActuatorTarget],
        clipped_targets: list[str],
        missing_actuators: list[str],
        unsupported_actuators: list[str],
    ) -> None:
        for command_key, value in sorted(command_values.items()):
            channel = actuator_mapping.channel_for_command(command_key)
            if channel is None:
                missing_actuators.append(command_key)
                continue
            if (
                channel.actuator_type != expected_type
                and expected_type not in channel.supported_command_types
            ):
                unsupported_actuators.append(command_key)
                continue
            target = _target_from_channel(channel, command_key, float(value), expected_type=expected_type)
            actuator_targets.append(target)
            if target.clipped:
                clipped_targets.append(channel.actuator_id)


def actuator_target_record_to_dict(record: IsaacActuatorTargetRecord) -> dict[str, Any]:
    return record.to_dict()


def _target_from_channel(
    channel: ActuatorChannel,
    command_key: str,
    value: float,
    *,
    expected_type: str,
) -> IsaacActuatorTarget:
    clipped_value, clipped = clip_to_channel(value, channel, expected_type)
    return IsaacActuatorTarget(
        actuator_id=channel.actuator_id,
        isaac_target_name=channel.isaac_target_name,
        actuator_type=SUPPORTED_COMMAND_TYPES[expected_type],
        command_key=command_key,
        target_value=clipped_value,
        unclipped_value=value,
        clipped=clipped,
        metadata={
            "module_id": channel.module_id,
            "local_id": channel.local_id,
            "channel_primary_type": channel.actuator_type,
        },
    )
