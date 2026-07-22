from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.geometry.pose_math import FACE_TO_FACE_DOCK_RELATION, compose_pose, inverse_pose
from amsrr.robot_model.fixed_morphology_urdf import (
    articulated_morphology_graph_connections,
    articulated_morphology_connections,
    fixed_module_joint_name,
    fixed_module_link_name,
    fixed_morphology_module_poses,
    morphology_graph_module_poses,
    morphology_graph_module_root_poses,
    split_fixed_module_name,
    write_articulated_morphology_graph_urdf,
    write_articulated_morphology_urdf,
    write_fixed_morphology_urdf,
    write_fixed_morphology_graph_urdf,
    write_joint_velocity_override_urdf,
    write_resolved_mesh_urdf,
)
from amsrr.robot_model.physical_model_builder import build_module_capability_token, build_physical_model_from_config
from amsrr.schemas.morphology import ControlGroup, DockEdge, ModuleNode, MorphologyGraph
from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_in_root_frame
from amsrr.simulation.order8_natural_contact import build_representative_order8_morphology


MESH_SEARCH_DIRS = [Path("module_urdf"), Path("module_urdf/mesh")]


def test_fixed_morphology_urdf_prefixes_modules_and_keeps_single_tree(tmp_path: Path) -> None:
    output_path = write_fixed_morphology_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "holon_fixed_2.urdf",
        module_count=2,
        module_spacing_m=0.45,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )

    model = load_urdf(output_path)

    assert model.frame_tree_valid is True
    assert model.root_links == [fixed_module_link_name(0, "root")]
    link_names = {link.name for link in model.links}
    joint_names = {joint.name for joint in model.joints}
    assert fixed_module_link_name(0, "thrust_1") in link_names
    assert fixed_module_link_name(1, "thrust_1") in link_names
    assert fixed_module_joint_name(0, "gimbal1") in joint_names
    assert fixed_module_joint_name(1, "gimbal1") in joint_names
    assert "fixed_module_1_to_module_0" in joint_names
    assert split_fixed_module_name(fixed_module_joint_name(1, "gimbal1")) == (1, "gimbal1")
    assert split_fixed_module_name("gimbal1") is None
    assert len([name for name in link_names if name.endswith("__thrust_1")]) == 2
    module_poses = fixed_morphology_module_poses(
        "assets/robots/holon/holon.urdf",
        module_count=2,
        module_spacing_m=0.45,
    )
    assert module_poses[1][0] == pytest.approx(0.5346, abs=1.0e-3)
    assert module_poses[1][1] == pytest.approx(0.5346, abs=1.0e-3)
    link_poses = link_poses_in_root_frame(model)
    src_port = link_poses[fixed_module_link_name(0, "pitch_connect_dummy_1")]
    dst_port = link_poses[fixed_module_link_name(1, "yaw_connect_dummy_1")]
    assert dst_port == pytest.approx(compose_pose(src_port, FACE_TO_FACE_DOCK_RELATION), abs=1.0e-6)
    mesh_refs = [ref for link in model.links for ref in link.visual_mesh_refs + link.collision_mesh_refs]
    assert mesh_refs
    assert all(Path(ref).is_absolute() for ref in mesh_refs)
    assert all(Path(ref).exists() for ref in mesh_refs)


def test_fixed_morphology_graph_urdf_reflects_graph_module_poses_and_edges(tmp_path: Path) -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    capability = build_module_capability_token(physical_model)
    morphology = MorphologyGraph(
        graph_id="p4_2_graph_specific_asset",
        modules=[
            ModuleNode(0, "holon", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), "base", True, capability),
            ModuleNode(1, "holon", (0.42, -0.18, 0.03, 0.0, 0.0, 0.0, 1.0), "left_grasp", False, capability),
            ModuleNode(2, "holon", (0.36, 0.22, 0.02, 0.0, 0.0, 0.0, 1.0), "right_grasp", False, capability),
        ],
        ports=[],
        dock_edges=[
            DockEdge(0, 0, 0, 1, 1, (0.42, -0.18, 0.03, 0.0, 0.0, 0.0, 1.0), "grasp_arm", [1.0] * 6, "attached"),
            DockEdge(1, 0, 2, 2, 3, (0.36, 0.22, 0.02, 0.0, 0.0, 0.0, 1.0), "grasp_arm", [1.0] * 6, "attached"),
        ],
        robot_anchors=[],
        control_groups=[ControlGroup("all", [0, 1, 2], "whole_body")],
        base_module_id=0,
        is_closed_loop=False,
    )

    output_path = write_fixed_morphology_graph_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "holon_graph.urdf",
        morphology_graph=morphology,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )
    model = load_urdf(output_path)
    link_poses = link_poses_in_root_frame(model)

    assert model.frame_tree_valid is True
    assert model.root_links == [fixed_module_link_name(0, "root")]
    assert "graph_edge_0_module_1_to_module_0" in {joint.name for joint in model.joints}
    assert "graph_edge_1_module_2_to_module_0" in {joint.name for joint in model.joints}
    assert link_poses[fixed_module_link_name(1, "root")] == pytest.approx(morphology.modules[1].pose_in_design_frame)
    assert link_poses[fixed_module_link_name(2, "root")] == pytest.approx(morphology.modules[2].pose_in_design_frame)
    assert morphology_graph_module_poses(morphology)[2] == pytest.approx(morphology.modules[2].pose_in_design_frame)


def test_fixed_morphology_graph_urdf_converts_rotated_fc_poses_to_root_poses(tmp_path: Path) -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    capability = build_module_capability_token(physical_model)
    quarter_turn_z = (0.0, 0.0, 2.0**-0.5, 2.0**-0.5)
    child_fc_pose = (0.42, -0.18, 0.12, *quarter_turn_z)
    morphology = MorphologyGraph(
        graph_id="rotated-fc-frame-graph",
        modules=[
            ModuleNode(0, "holon", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), "base", True, capability),
            ModuleNode(1, "holon", child_fc_pose, "member", False, capability),
        ],
        ports=[],
        dock_edges=[
            DockEdge(0, 0, 0, 1, 1, child_fc_pose, "structural", [1.0] * 6, "attached"),
        ],
        robot_anchors=[],
        control_groups=[ControlGroup("all", [0, 1], "whole_body")],
        base_module_id=0,
        is_closed_loop=False,
    )

    output_path = write_fixed_morphology_graph_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "rotated_graph.urdf",
        morphology_graph=morphology,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )
    link_poses = link_poses_in_root_frame(load_urdf(output_path))
    base_fc_pose = link_poses[fixed_module_link_name(0, "fc")]
    child_fc_pose_in_output = link_poses[fixed_module_link_name(1, "fc")]
    observed_base_fc_to_child_fc = compose_pose(inverse_pose(base_fc_pose), child_fc_pose_in_output)
    root_poses = morphology_graph_module_root_poses(
        morphology,
        source_urdf_path="assets/robots/holon/holon.urdf",
    )

    assert observed_base_fc_to_child_fc == pytest.approx(child_fc_pose, abs=1.0e-6)
    assert link_poses[fixed_module_link_name(1, "root")] == pytest.approx(root_poses[1], abs=1.0e-6)
    assert root_poses[1] != pytest.approx(child_fc_pose, abs=1.0e-6)


def test_articulated_morphology_urdf_connects_child_root_to_parent_dock_port(tmp_path: Path) -> None:
    output_path = write_articulated_morphology_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "holon_articulated_2.urdf",
        module_count=2,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )

    model = load_urdf(output_path)
    connections = articulated_morphology_connections("assets/robots/holon/holon.urdf", module_count=2)

    assert model.frame_tree_valid is True
    assert model.root_links == [fixed_module_link_name(0, "root")]
    assert len(connections) == 1
    connection = connections[0]
    parent_joint = next(joint for joint in model.joints if joint.name == "articulated_module_1_to_module_0")
    assert parent_joint.parent_link == fixed_module_link_name(0, connection.parent_connect_link)
    assert parent_joint.child_link == fixed_module_link_name(1, "root")
    assert connection.parent_mechanism_joint_id == "pitch_dock_mech_joint1"
    assert connection.child_mechanism_joint_id == "yaw_dock_mech_joint1"

    link_poses = link_poses_in_root_frame(model)
    src_port = link_poses[fixed_module_link_name(0, connection.parent_connect_link)]
    dst_port = link_poses[fixed_module_link_name(1, connection.child_connect_link)]
    assert dst_port == pytest.approx(compose_pose(src_port, FACE_TO_FACE_DOCK_RELATION), abs=1.0e-6)


def test_articulated_graph_urdf_preserves_both_dock_dofs_in_structure(
    tmp_path: Path,
) -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    morphology = build_representative_order8_morphology(physical_model)
    output_path = write_articulated_morphology_graph_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "holon_articulated_graph.urdf",
        morphology_graph=morphology,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )

    model = load_urdf(output_path)
    connections = articulated_morphology_graph_connections(
        "assets/robots/holon/holon.urdf",
        morphology_graph=morphology,
    )
    joints = {joint.name: joint for joint in model.joints}
    parent_joint_by_child = {joint.child_link: joint for joint in model.joints}

    assert model.frame_tree_valid is True
    assert model.root_links == [fixed_module_link_name(0, "root")]
    assert [(item.parent_module_id, item.child_module_id) for item in connections] == [
        (0, 1),
        (0, 2),
    ]
    for connection in connections:
        edge_joint = joints[
            f"articulated_graph_edge_{connection.edge_id}_module_"
            f"{connection.child_module_id}_to_module_{connection.parent_module_id}"
        ]
        assert edge_joint.parent_link == fixed_module_link_name(
            connection.parent_module_id, connection.parent_connect_link
        )
        assert edge_joint.child_link == fixed_module_link_name(
            connection.child_module_id, connection.child_connect_link
        )
        structural_joint_name = fixed_module_joint_name(
            connection.child_module_id,
            str(connection.child_mechanism_joint_id),
        )
        assert structural_joint_name in joints
        current = fixed_module_link_name(connection.child_module_id, "fc")
        ancestor_joint_names: set[str] = set()
        while current in parent_joint_by_child:
            ancestor = parent_joint_by_child[current]
            ancestor_joint_names.add(ancestor.name)
            current = ancestor.parent_link
        assert structural_joint_name in ancestor_joint_names

    link_poses = link_poses_in_root_frame(model)
    base_fc = link_poses[fixed_module_link_name(0, "fc")]
    expected = morphology_graph_module_poses(morphology)
    for module_id in (1, 2):
        child_fc = link_poses[fixed_module_link_name(module_id, "fc")]
        observed = compose_pose(inverse_pose(base_fc), child_fc)
        assert observed == pytest.approx(expected[module_id], abs=1.0e-6)


def test_resolved_mesh_urdf_points_asset_meshes_to_existing_files(tmp_path: Path) -> None:
    output_path = write_resolved_mesh_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "holon_resolved.urdf",
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )

    model = load_urdf(output_path)
    mesh_refs = [ref for link in model.links for ref in link.visual_mesh_refs + link.collision_mesh_refs]

    assert mesh_refs
    assert all(Path(ref).is_absolute() for ref in mesh_refs)
    assert all(Path(ref).exists() for ref in mesh_refs)


def test_joint_velocity_override_matches_fixed_module_local_names(tmp_path: Path) -> None:
    fixed_path = write_fixed_morphology_urdf(
        "assets/robots/holon/holon.urdf",
        tmp_path / "holon_fixed_2.urdf",
        module_count=2,
        module_spacing_m=0.45,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )
    output_path = write_joint_velocity_override_urdf(
        fixed_path,
        tmp_path / "holon_fixed_2_fast_gimbals.urdf",
        joint_velocity_overrides={"gimbal1": 20.0, "gimbal2": 20.0},
    )

    text = output_path.read_text()

    assert 'name="module_0__gimbal1"' in text
    assert 'name="module_1__gimbal1"' in text
    assert text.count('velocity="20"') >= 4
