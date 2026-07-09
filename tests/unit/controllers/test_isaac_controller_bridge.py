from __future__ import annotations

import pytest

from amsrr.controllers.actuator_mapping import build_actuator_mapping
from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge, actuator_target_record_to_dict
from amsrr.robot_model.physical_model_builder import build_module_capability_token, build_physical_model_from_config
from amsrr.schemas.morphology import ModuleNode, MorphologyGraph
from amsrr.schemas.policies import ControllerCommand, ControllerStatus


def _physical_model():
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


def _morphology_graph() -> MorphologyGraph:
    physical_model = _physical_model()
    capability = build_module_capability_token(physical_model)
    return MorphologyGraph(
        graph_id="bridge-test",
        modules=[
            ModuleNode(
                module_id=0,
                module_type="holon",
                pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base",
                is_base=True,
                capability_token=capability,
            )
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )


def _controller_command() -> ControllerCommand:
    return ControllerCommand(
        rotor_thrusts_n={"thrust_1": 25.0, "module_0:thrust_2": 5.0},
        vectoring_joint_targets={"gimbal1": 3.0},
        joint_torque_commands={"payload_joint": 1.0},
        dock_mechanism_commands={"pitch_dock_mech_joint1": 2.0},
        controller_status=ControllerStatus(
            status="warning",
            qp_feasible=True,
            active_mode="qpid_rigid_body_qp",
            metrics={"allocation_residual_norm": 0.125},
        ),
    )


def test_isaac_controller_bridge_converts_and_clips_targets() -> None:
    mapping = build_actuator_mapping(_morphology_graph(), _physical_model())
    record = IsaacControllerBridge().convert(
        _controller_command(),
        mapping,
        time_s=0.25,
        command_index=7,
    )

    targets_by_id = {target.actuator_id: target for target in record.actuator_targets}
    assert targets_by_id["module_0:thrust_1"].target_value == pytest.approx(20.0)
    assert targets_by_id["module_0:thrust_1"].unclipped_value == pytest.approx(25.0)
    assert targets_by_id["module_0:gimbal1"].target_value == pytest.approx(2.0)
    assert targets_by_id["module_0:pitch_dock_mech_joint1"].target_value == pytest.approx(1.5708)
    assert "payload_joint" in record.missing_actuators
    assert record.unsupported_actuators == []
    assert record.allocation_residual_norm == pytest.approx(0.125)
    assert record.metrics["clipped_target_count"] == 3.0
    assert record.metrics["missing_actuator_count"] == 1.0
    assert record.metadata["controller_active_mode"] == "qpid_rigid_body_qp"


def test_isaac_controller_bridge_record_round_trips_as_archive_dict() -> None:
    mapping = build_actuator_mapping(_morphology_graph(), _physical_model())
    record = IsaacControllerBridge().convert(
        _controller_command(),
        mapping,
        time_s=0.0,
        command_index=0,
    )

    data = actuator_target_record_to_dict(record)
    restored = type(record).from_dict(data)

    assert restored.to_dict() == data
    assert data["backend"] == "isaac_lab"
    assert data["morphology_graph_id"] == "bridge-test"
    assert data["metrics"]["actuator_target_count"] == 4.0
