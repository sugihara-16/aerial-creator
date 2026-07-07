from __future__ import annotations

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
    assert physical_model.aggregate_mass_kg > 0.0
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
    assert physical_model.rotors[0].thrust_axis_local == (0.0, 0.0, 1.0)
    assert physical_model.rotors[1].thrust_axis_local == (0.0, 0.0, -1.0)
    assert physical_model.rotors[0].vectoring_joint_ids == ["gimbal1"]
    assert len(physical_model.dock_ports) == 4
    assert sorted(port.port_type for port in physical_model.dock_ports) == [
        "pitch_dock",
        "pitch_dock",
        "yaw_dock",
        "yaw_dock",
    ]
    assert all(port.compatible_port_types for port in physical_model.dock_ports)


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

