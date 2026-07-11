from __future__ import annotations

import pytest

from amsrr.controllers.actuator_mapping import ActuatorMappingBuilder, build_actuator_mapping, clip_to_channel
from amsrr.robot_model.physical_model_builder import build_module_capability_token, build_physical_model_from_config
from amsrr.schemas.morphology import ControlGroup, ModuleNode, MorphologyGraph


def _physical_model():
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


def _morphology(module_count: int = 1) -> MorphologyGraph:
    physical_model = _physical_model()
    capability = build_module_capability_token(physical_model)
    return MorphologyGraph(
        graph_id=f"actuator-mapping-test-{module_count}",
        modules=[
            ModuleNode(
                module_id=module_id,
                module_type="holon",
                pose_in_design_frame=(0.2 * module_id, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base" if module_id == 0 else "attached",
                is_base=module_id == 0,
                capability_token=capability,
            )
            for module_id in range(module_count)
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[ControlGroup(group_id="all", module_ids=list(range(module_count)), role="whole_body")],
        base_module_id=0,
        is_closed_loop=False,
    )


def test_actuator_mapping_builds_single_module_aliases_and_limits() -> None:
    physical_model = _physical_model()
    mapping = build_actuator_mapping(_morphology(), physical_model)

    rotor = mapping.channel_for_command("thrust_1")
    assert rotor is not None
    assert rotor.actuator_id == "module_0:thrust_1"
    assert rotor.actuator_type == "rotor_thrust"
    assert rotor.lower == pytest.approx(0.0)
    assert rotor.upper == pytest.approx(20.0)
    assert mapping.channel_for_command("module_0:gimbal1").actuator_type == "vectoring_joint_position"  # type: ignore[union-attr]
    assert mapping.channel_for_command("pitch_dock_mech_joint1").actuator_type == "dock_joint_position"  # type: ignore[union-attr]
    gimbal = mapping.channel_for_command("gimbal1")
    dock = mapping.channel_for_command("pitch_dock_mech_joint1")
    assert gimbal is not None and gimbal.metadata["actuator_model"] == "XC330-T181-T"
    assert dock is not None and dock.metadata["actuator_model"] == "AK40-10 KV170"
    assert gimbal.effort == pytest.approx(0.76)
    assert dock.effort == pytest.approx(4.1)
    assert mapping.metadata["builder_version"] == "actuator_mapping_v1"


def test_actuator_mapping_uses_global_keys_for_multiple_modules() -> None:
    physical_model = _physical_model()
    mapping = ActuatorMappingBuilder().build(_morphology(module_count=2), physical_model)

    assert mapping.channel_for_command("module_0:thrust_1") is not None
    assert mapping.channel_for_command("module_1:thrust_1") is not None
    assert mapping.channel_for_command("thrust_1") is None
    assert len({channel.actuator_id for channel in mapping.channels}) == len(mapping.channels)


def test_clip_to_channel_reports_clipped_value() -> None:
    mapping = build_actuator_mapping(_morphology(), _physical_model())
    channel = mapping.channel_for_command("gimbal1")
    assert channel is not None

    clipped_value, clipped = clip_to_channel(3.0, channel)

    assert clipped is True
    assert clipped_value == pytest.approx(2.0)
