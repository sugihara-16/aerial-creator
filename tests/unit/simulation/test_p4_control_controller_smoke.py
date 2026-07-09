from __future__ import annotations

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.p4_control_controller_smoke import build_single_module_controller_command_smoke


def test_single_module_controller_command_smoke_builds_bridge_record() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    joint_positions = {joint.joint_id: 0.0 for joint in physical_model.joints}
    joint_velocities = {joint.joint_id: 0.0 for joint in physical_model.joints}

    bundle = build_single_module_controller_command_smoke(
        physical_model,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
    )

    assert bundle.controller_command.controller_status.active_mode == "qpid_rigid_body_qp"
    assert bundle.controller_command.controller_status.qp_feasible is True
    assert bundle.controller_command.controller_status.status == "ok"
    assert bundle.controller_command.controller_status.metrics["qp_primary_path"] == 1.0
    assert bundle.controller_command.controller_status.metrics["allocation_residual_norm"] < 1.0e-5
    assert bundle.controller_command.controller_status.metrics["clipped"] == 0.0
    assert bundle.controller_command.rotor_thrusts_n
    assert bundle.controller_command.joint_torque_commands == {}
    assert bundle.actuator_target_record.missing_actuators == []
    assert bundle.actuator_target_record.unsupported_actuators == []
    assert bundle.metrics["bridge_target_count"] >= 4.0
