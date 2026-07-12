from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import JointModel, PhysicalModel


ActuatorType = Literal[
    "rotor_thrust",
    "vectoring_joint_position",
    "dock_joint_position",
    "joint_effort",
    "joint_position",
    "joint_velocity",
    "joint_effort_bias",
]


@dataclass
class ActuatorChannel(SchemaBase):
    actuator_id: str
    module_id: int
    local_id: str
    actuator_type: ActuatorType
    isaac_target_name: str
    lower: float | None = None
    upper: float | None = None
    velocity: float | None = None
    effort: float | None = None
    aliases: list[str] = field(default_factory=list)
    supported_command_types: list[ActuatorType] = field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.actuator_id, "ActuatorChannel.actuator_id")
        require_non_empty(self.local_id, "ActuatorChannel.local_id")
        require_non_empty(self.isaac_target_name, "ActuatorChannel.isaac_target_name")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise SchemaValidationError(f"ActuatorChannel {self.actuator_id!r} has lower > upper")


@dataclass
class ActuatorMapping(SchemaBase):
    graph_id: str
    module_ids: list[int]
    channels: list[ActuatorChannel]
    command_key_aliases: dict[str, str]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.graph_id, "ActuatorMapping.graph_id")
        actuator_ids = [channel.actuator_id for channel in self.channels]
        if len(actuator_ids) != len(set(actuator_ids)):
            raise SchemaValidationError("ActuatorMapping.channels contains duplicate actuator_id values")
        known_ids = set(actuator_ids)
        unknown_alias_targets = sorted(set(self.command_key_aliases.values()) - known_ids)
        if unknown_alias_targets:
            raise SchemaValidationError(f"ActuatorMapping has aliases to unknown channels: {unknown_alias_targets}")

    def channel_for_command(
        self,
        command_key: str,
        expected_type: str | None = None,
    ) -> ActuatorChannel | None:
        actuator_id = self.command_key_aliases.get(command_key, command_key)
        channels_by_id = {channel.actuator_id: channel for channel in self.channels}
        channel = channels_by_id.get(actuator_id)
        if channel is None or expected_type is None:
            return channel
        if expected_type == channel.actuator_type or expected_type in channel.supported_command_types:
            return channel
        return None


class ActuatorMappingBuilder:
    """Build deterministic controller-to-Isaac actuator channel mappings."""

    def build(self, morphology_graph: MorphologyGraph, physical_model: PhysicalModel) -> ActuatorMapping:
        module_ids = sorted(module.module_id for module in morphology_graph.modules)
        if not module_ids:
            raise SchemaValidationError("ActuatorMapping requires at least one active module")

        vectoring_joint_ids = sorted(
            {
                joint_id
                for rotor in physical_model.rotors
                for joint_id in rotor.vectoring_joint_ids
            }
        )
        dock_joint_ids = sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in physical_model.dock_ports
                if port.mechanical_limits.get("mechanism_joint_id")
            }
        )
        joints_by_id = {joint.joint_id: joint for joint in physical_model.joints}
        channels: list[ActuatorChannel] = []
        aliases: dict[str, str] = {}
        single_module = len(module_ids) == 1

        for module_id in module_ids:
            for rotor in sorted(physical_model.rotors, key=lambda item: item.rotor_id):
                self._add_channel(
                    channels,
                    aliases,
                    module_id=module_id,
                    local_id=rotor.rotor_id,
                    actuator_type="rotor_thrust",
                    lower=rotor.thrust_min_n,
                    upper=rotor.thrust_max_n,
                    single_module=single_module,
                    metadata={"source": "PhysicalModel.rotors"},
                )

            for joint_id in vectoring_joint_ids:
                joint = _require_joint(joints_by_id, joint_id)
                self._add_channel(
                    channels,
                    aliases,
                    module_id=module_id,
                    local_id=joint_id,
                    actuator_type="vectoring_joint_position",
                    lower=joint.limit_lower,
                    upper=joint.limit_upper,
                    velocity=joint.velocity_limit,
                    effort=joint.effort_limit,
                    single_module=single_module,
                    metadata=_joint_actuator_channel_metadata(
                        physical_model,
                        joint_id,
                        source="RotorModel.vectoring_joint_ids",
                    ),
                )

            for joint_id in dock_joint_ids:
                joint = _require_joint(joints_by_id, joint_id)
                self._add_channel(
                    channels,
                    aliases,
                    module_id=module_id,
                    local_id=joint_id,
                    actuator_type="dock_joint_position",
                    lower=joint.limit_lower,
                    upper=joint.limit_upper,
                    velocity=joint.velocity_limit,
                    effort=joint.effort_limit,
                    single_module=single_module,
                    metadata=_joint_actuator_channel_metadata(
                        physical_model,
                        joint_id,
                        source="DockPortSpec.mechanical_limits",
                    ),
                    supported_command_types=[
                        "joint_position",
                        "joint_velocity",
                        "joint_effort_bias",
                    ],
                )

            for joint in sorted(physical_model.joints, key=lambda item: item.joint_id):
                if joint.joint_id in vectoring_joint_ids or joint.joint_id in dock_joint_ids:
                    continue
                if joint.effort_limit is None:
                    continue
                self._add_channel(
                    channels,
                    aliases,
                    module_id=module_id,
                    local_id=joint.joint_id,
                    actuator_type="joint_effort",
                    lower=-abs(float(joint.effort_limit)),
                    upper=abs(float(joint.effort_limit)),
                    velocity=joint.velocity_limit,
                    effort=joint.effort_limit,
                    single_module=single_module,
                    metadata={"source": "JointModel.effort_limit"},
                )

        return ActuatorMapping(
            graph_id=morphology_graph.graph_id,
            module_ids=module_ids,
            channels=channels,
            command_key_aliases=aliases,
            metadata={
                "active_module_count": len(module_ids),
                "channel_count": len(channels),
                "builder_version": "actuator_mapping_v2",
            },
        )

    @staticmethod
    def _add_channel(
        channels: list[ActuatorChannel],
        aliases: dict[str, str],
        *,
        module_id: int,
        local_id: str,
        actuator_type: ActuatorType,
        lower: float | None = None,
        upper: float | None = None,
        velocity: float | None = None,
        effort: float | None = None,
        single_module: bool,
        metadata: dict[str, str | int | float | bool] | None = None,
        supported_command_types: list[ActuatorType] | None = None,
    ) -> None:
        actuator_id = _global_id(module_id, local_id)
        isaac_target_name = f"module_{module_id}/{local_id}"
        channel_aliases = [actuator_id]
        if single_module:
            channel_aliases.append(local_id)
        channels.append(
            ActuatorChannel(
                actuator_id=actuator_id,
                module_id=module_id,
                local_id=local_id,
                actuator_type=actuator_type,
                isaac_target_name=isaac_target_name,
                lower=lower,
                upper=upper,
                velocity=velocity,
                effort=effort,
                aliases=channel_aliases,
                supported_command_types=supported_command_types or [],
                metadata=metadata or {},
            )
        )
        for alias in channel_aliases:
            aliases[alias] = actuator_id


def build_actuator_mapping(morphology_graph: MorphologyGraph, physical_model: PhysicalModel) -> ActuatorMapping:
    return ActuatorMappingBuilder().build(morphology_graph, physical_model)


def _joint_actuator_channel_metadata(
    physical_model: PhysicalModel,
    joint_id: str,
    *,
    source: str,
) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {"source": source}
    assignments = physical_model.metadata.get("joint_actuator_assignments")
    specs = physical_model.metadata.get("joint_actuator_specs")
    if not isinstance(assignments, dict) or not isinstance(specs, dict):
        return metadata
    role = assignments.get(joint_id)
    spec = specs.get(role) if isinstance(role, str) else None
    if not isinstance(spec, dict):
        return metadata
    metadata.update(
        {
            "actuator_role": role,
            "actuator_manufacturer": str(spec.get("manufacturer", "")),
            "actuator_model": str(spec.get("model", "")),
            "continuous_torque_limit_nm": float(spec.get("continuous_torque_limit_nm", 0.0)),
            "peak_torque_limit_nm": float(spec.get("peak_torque_nm", 0.0)),
            "no_load_speed_rad_s": float(spec.get("no_load_speed_rad_s", 0.0)),
        }
    )
    return metadata


def clip_to_channel(
    value: float,
    channel: ActuatorChannel,
    command_type: str | None = None,
) -> tuple[float, bool]:
    clipped_value = float(value)
    clipped = False
    lower, upper = _command_limits(channel, command_type or channel.actuator_type)
    if lower is not None and clipped_value < lower:
        clipped_value = float(lower)
        clipped = True
    if upper is not None and clipped_value > upper:
        clipped_value = float(upper)
        clipped = True
    return clipped_value, clipped


def _command_limits(channel: ActuatorChannel, command_type: str) -> tuple[float | None, float | None]:
    if command_type == "joint_velocity":
        if channel.velocity is None:
            return None, None
        limit = abs(float(channel.velocity))
        return -limit, limit
    if command_type == "joint_effort_bias":
        continuous_limit = channel.metadata.get("continuous_torque_limit_nm")
        if isinstance(continuous_limit, (int, float)) and float(continuous_limit) > 0.0:
            limit = abs(float(continuous_limit))
            return -limit, limit
        if channel.effort is None:
            return None, None
        limit = abs(float(channel.effort))
        return -limit, limit
    return channel.lower, channel.upper


def _require_joint(joints_by_id: dict[str, JointModel], joint_id: str) -> JointModel:
    joint = joints_by_id.get(joint_id)
    if joint is None:
        raise SchemaValidationError(f"Actuator mapping references unknown joint {joint_id!r}")
    return joint


def _global_id(module_id: int, local_id: str) -> str:
    return f"module_{module_id}:{local_id}"
