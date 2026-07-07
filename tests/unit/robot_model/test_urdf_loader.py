from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.robot_model.urdf_loader import load_urdf


def test_urdf_parse_holon_if_present() -> None:
    path = Path("module_urdf/holon.urdf")
    if not path.exists():
        pytest.skip("module_urdf/holon.urdf is not present in this checkout")

    model = load_urdf(path)
    assert model.links
    assert model.joints
    assert model.frame_tree_valid


def test_urdf_parse_holon_xacro_reference() -> None:
    model = load_urdf("module_urdf/holon.urdf.xacro", rotor_link_patterns=["thrust"])

    assert len(model.links) == 29
    assert len(model.joints) == 28
    assert model.joint_type_counts == {"fixed": 16, "revolute": 8, "continuous": 4}
    assert model.total_mass_kg > 0.0
    assert model.root_links == ["root"]
    assert model.frame_tree_valid
    assert model.candidate_rotor_links == ["thrust1", "thrust2", "thrust3", "thrust4"]
    assert "pitch_connect_point_1" in model.candidate_connect_joints
    assert "yaw_connect_point_2" in model.candidate_connect_joints
    assert model.metadata["baselink"]["name"] == "fc"


def test_asset_urdf_uses_config_thrust_link_names() -> None:
    model = load_urdf("assets/robots/holon/holon.urdf", rotor_joint_patterns=["rotor_"])

    assert model.frame_tree_valid
    assert model.candidate_rotor_links == ["thrust_1", "thrust_2", "thrust_3", "thrust_4"]
    assert model.candidate_rotor_joints == ["rotor1", "rotor2", "rotor3", "rotor4"]
    assert not any(link.name in {"thrust1", "thrust2", "thrust3", "thrust4"} for link in model.links)
