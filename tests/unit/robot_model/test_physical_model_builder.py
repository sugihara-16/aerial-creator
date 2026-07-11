from __future__ import annotations

import math

import pytest

from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION, compose_pose, dock_module_relative_pose
from amsrr.robot_model.physical_model_builder import (
    build_module_capability_token,
    build_physical_model_from_config,
)


def test_physical_model_total_mass_positive() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")

    assert physical_model.model_id == "holon"
    assert physical_model.urdf_path.endswith("assets/robots/holon/holon.urdf")
    assert len(physical_model.links) == 29
    assert len(physical_model.joints) == 28
    assert physical_model.aggregate_mass_kg == math.fsum(
        link.mass_kg for link in physical_model.links
    )
    assert physical_model.metadata["frame_tree_valid"] is True
    assert physical_model.metadata["root_links"] == ["root"]


def test_physical_model_rotors_and_dock_ports() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")

    assert [rotor.rotor_id for rotor in physical_model.rotors] == ["thrust_1", "thrust_2", "thrust_3", "thrust_4"]
    assert [rotor.thrust_frame_link for rotor in physical_model.rotors] == [
        "thrust_1",
        "thrust_2",
        "thrust_3",
        "thrust_4",
    ]
    assert all(rotor.thrust_axis_local == (0.0, 0.0, 1.0) for rotor in physical_model.rotors)
    assert [rotor.reaction_torque_coeff_nm_per_n for rotor in physical_model.rotors] == pytest.approx(
        [-0.0172, 0.0172, -0.0172, 0.0172]
    )
    assert physical_model.rotors[0].vectoring_joint_ids == ["gimbal1"]
    gimbal = next(joint for joint in physical_model.joints if joint.joint_id == "gimbal1")
    assert gimbal.effort_limit == pytest.approx(0.76)
    assert gimbal.velocity_limit == pytest.approx(10.890854)
    assert len(physical_model.dock_ports) == 4
    assert sorted(port.port_type for port in physical_model.dock_ports) == [
        "pitch_dock",
        "pitch_dock",
        "yaw_dock",
        "yaw_dock",
    ]
    assert all(port.compatible_port_types for port in physical_model.dock_ports)
    pitch = next(port for port in physical_model.dock_ports if port.port_id == "pitch_connect_point_1")
    yaw = next(port for port in physical_model.dock_ports if port.port_id == "yaw_connect_point_1")
    relative = dock_module_relative_pose(pitch.local_pose, yaw.local_pose)
    assert pitch.local_pose[0] > 0.2
    assert yaw.local_pose[0] < -0.2
    assert compose_pose(relative, yaw.local_pose) == pytest.approx(compose_pose(pitch.local_pose, FACE_TO_FACE_DOCK_RELATION))
    assert pitch.mechanical_limits["actuator_model"] == "AK40-10 KV170"
    assert pitch.mechanical_limits["continuous_torque_limit_nm"] == pytest.approx(1.3)
    assert pitch.mechanical_limits["peak_torque_limit_nm"] == pytest.approx(4.1)


def test_physical_model_records_joint_actuator_provenance() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")

    assert physical_model.metadata["joint_actuator_model_version"] == "joint_actuator_model_v1"
    assert physical_model.metadata["joint_actuator_model_hash"]
    assignments = physical_model.metadata["joint_actuator_assignments"]
    assert assignments["gimbal1"] == "vectoring"
    assert assignments["pitch_dock_mech_joint1"] == "dock"
    specs = physical_model.metadata["joint_actuator_specs"]
    assert specs["vectoring"]["model"] == "XC330-T181-T"
    assert specs["dock"]["model"] == "AK40-10 KV170"


def test_module_capability_token_from_physical_model() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    capability = build_module_capability_token(physical_model)

    assert capability.module_type == "holon"
    assert capability.rotor_count == 4
    assert capability.port_count == 4
    assert capability.thrust_to_weight_ratio_est > 0.0
    assert capability.dock_port_type_counts == [2, 2, 0]
    assert capability.has_vectoring is True
    assert capability.has_dock_mechanism is True
