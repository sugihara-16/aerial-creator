from __future__ import annotations

from pathlib import Path

from amsrr.robot_model.fixed_morphology_urdf import (
    fixed_module_joint_name,
    fixed_module_link_name,
    split_fixed_module_name,
    write_fixed_morphology_urdf,
    write_resolved_mesh_urdf,
)
from amsrr.robot_model.urdf_loader import load_urdf


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
    mesh_refs = [ref for link in model.links for ref in link.visual_mesh_refs + link.collision_mesh_refs]
    assert mesh_refs
    assert all(Path(ref).is_absolute() for ref in mesh_refs)
    assert all(Path(ref).exists() for ref in mesh_refs)


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
