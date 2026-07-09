from __future__ import annotations

from amsrr.controllers.actuator_mapping import build_actuator_mapping
from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.controllers.controller_base import ControllerContext
from amsrr.robot_model.fixed_morphology_urdf import fixed_morphology_module_poses
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import InteractionKnot, PolicyCommand
from amsrr.simulation.p4_control_controller_smoke import (
    bridge_supported_controller_command,
    build_fixed_morphology,
    build_runtime_observation,
    build_single_module_controller_command_smoke,
)


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
    assert bundle.controller_command.controller_status.metrics["allocation_residual_norm"] < 1.0e-4
    assert bundle.controller_command.controller_status.metrics["clipped"] == 0.0
    assert bundle.controller_command.rotor_thrusts_n
    assert bundle.controller_command.joint_torque_commands == {}
    assert bundle.actuator_target_record.missing_actuators == []
    assert bundle.actuator_target_record.unsupported_actuators == []
    assert bundle.metrics["bridge_target_count"] >= 4.0


def test_fixed_morphology_controller_command_smoke_builds_multi_module_bridge_record() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    module_poses = fixed_morphology_module_poses(physical_model.urdf_path, module_count=2, module_spacing_m=0.45)
    morphology_graph = build_fixed_morphology(
        physical_model,
        graph_id="fixed-morphology-unit-smoke",
        module_count=2,
        module_spacing_m=0.45,
        module_poses=module_poses,
    )
    joint_positions = {joint.joint_id: 0.0 for joint in physical_model.joints}
    runtime_observation = build_runtime_observation(
        morphology_graph,
        time_s=0.0,
        pose_world=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        twist_world=[0.0] * 6,
        joint_positions=joint_positions,
        joint_velocities={},
    )
    # build_runtime_observation is single-module oriented; expand the fixed
    # morphology state for this controller-only unit check.
    runtime_observation.module_states.append(
        type(runtime_observation.module_states[0])(
            module_id=1,
            pose_world=(
                module_poses[1][0],
                module_poses[1][1],
                0.5 + module_poses[1][2],
                module_poses[1][3],
                module_poses[1][4],
                module_poses[1][5],
                module_poses[1][6],
            ),
            twist_world=[0.0] * 6,
            joint_positions=joint_positions,
            joint_velocities={},
        )
    )

    controller_command = QPIDController(
        config=QPIDControllerConfig(allocation_mode="rigid_body_qp")
    ).compute(
        ControllerContext(
            runtime_observation=runtime_observation,
            morphology_graph=morphology_graph,
            physical_model=physical_model,
            active_knot=InteractionKnot(t_rel_s=0.0, contact_assignments=[]),
            policy_command=PolicyCommand(
                desired_body_pose=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                desired_body_twist=[0.0] * 6,
            ),
        )
    )
    bridged_command = bridge_supported_controller_command(controller_command)
    actuator_record = IsaacControllerBridge().convert(
        bridged_command,
        build_actuator_mapping(morphology_graph, physical_model),
        time_s=0.0,
        command_index=0,
    )

    assert len(bridged_command.rotor_thrusts_n) == 8
    assert bridged_command.controller_status.metrics["qp_primary_path"] == 1.0
    assert actuator_record.metrics["missing_actuator_count"] == 0.0
    assert actuator_record.metrics["unsupported_actuator_count"] == 0.0
    assert actuator_record.metrics["actuator_target_count"] >= 8.0
