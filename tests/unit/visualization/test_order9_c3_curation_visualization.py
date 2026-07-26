from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.visualization.order9_c3_curation import (
    ORDER9_C3_CURATION_VIEWER_VERSION,
    Order9C3ViewerBox,
    Order9C3ViewerMarker,
    build_order9_c3_urdf_mesh_scene,
    order9_c3_urdf_root_pose_from_baselink_pose,
    render_order9_c3_mesh_viewer,
    write_order9_c3_stl_mesh_library,
    write_order9_c3_viewer_javascript,
)


def _binary_triangle_stl() -> bytes:
    header = b"unit triangle" + b"\0" * (80 - len("unit triangle"))
    triangle = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
    return header + struct.pack("<I", 1) + triangle


def _urdf(mesh_path: Path) -> str:
    return f"""<?xml version="1.0"?>
<robot name="viewer_unit">
  <baselink name="module_0__fc"/>
  <link name="module_0__root">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_path}" scale="1 2 3"/></geometry>
      <material name="unit"><color rgba="0.2 0.3 0.4 1"/></material>
    </visual>
    <collision>
      <geometry><mesh filename="{mesh_path}"/></geometry>
    </collision>
  </link>
  <link name="module_0__fc"/>
  <joint name="module_0__fc_joint" type="fixed">
    <parent link="module_0__root"/>
    <child link="module_0__fc"/>
    <origin xyz="0 0.0001 0.060935" rpy="0 0 0"/>
  </joint>
  <link name="module_0__tip">
    <visual><geometry><mesh filename="{mesh_path}"/></geometry></visual>
  </link>
  <joint name="module_0__yaw_dock_mech_joint1" type="revolute">
    <parent link="module_0__root"/>
    <child link="module_0__tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def test_curation_viewer_converts_baselink_world_pose_to_urdf_root(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "triangle.STL"
    mesh_path.write_bytes(_binary_triangle_stl())
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(_urdf(mesh_path), encoding="utf-8")
    baselink_pose_world = (
        2.0,
        3.0,
        4.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    root_pose_world = order9_c3_urdf_root_pose_from_baselink_pose(
        urdf_path,
        baselink_pose_world,
    )
    scene = build_order9_c3_urdf_mesh_scene(
        urdf_path,
        root_pose_world=root_pose_world,
    )

    assert root_pose_world[:3] == pytest.approx(
        (2.0, 2.9999, 3.939065)
    )
    assert scene.link_poses_world["module_0__fc"] == pytest.approx(
        baselink_pose_world
    )


def test_curation_viewer_resolves_exact_stl_and_global_joint_ids(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "triangle.STL"
    mesh_path.write_bytes(_binary_triangle_stl())
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(_urdf(mesh_path), encoding="utf-8")

    scene = build_order9_c3_urdf_mesh_scene(
        urdf_path,
        joint_positions_rad={"module_0:yaw_dock_mech_joint1": 0.5},
        root_pose_world=(2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0),
    )

    assert scene.root_link_id == "module_0__root"
    assert scene.link_poses_world["module_0__root"][:3] == (2.0, 3.0, 4.0)
    assert scene.link_poses_world["module_0__tip"][:3] == pytest.approx(
        (3.0, 3.0, 4.0)
    )
    assert scene.link_poses_world["module_0__tip"][5] == pytest.approx(
        0.2474039593
    )
    assert {value["layer"] for value in scene.instances} == {
        "visual",
        "collision",
    }
    root_visual = next(
        value
        for value in scene.instances
        if value["instance_id"] == "visual:module_0__root:0"
    )
    assert root_visual["mesh_key"]
    assert root_visual["model_matrix"][12:15] == [2.0, 3.0, 4.0]
    assert root_visual["model_matrix"][0] == pytest.approx(1.0)
    assert root_visual["model_matrix"][5] == pytest.approx(2.0)
    assert root_visual["model_matrix"][10] == pytest.approx(3.0)


def test_curation_viewer_writes_shared_library_html_and_scene(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "triangle.STL"
    mesh_path.write_bytes(_binary_triangle_stl())
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(_urdf(mesh_path), encoding="utf-8")
    scene = build_order9_c3_urdf_mesh_scene(urdf_path)
    library_path = write_order9_c3_stl_mesh_library(
        scene.mesh_paths,
        tmp_path / "shared/mesh_library.js",
    )
    viewer_js = write_order9_c3_viewer_javascript(
        tmp_path / "shared/viewer.js"
    )

    artifacts = render_order9_c3_mesh_viewer(
        urdf_scene=scene,
        html_path=tmp_path / "case/final.html",
        mesh_library_path=library_path,
        viewer_javascript_path=viewer_js,
        title="unit final pose",
        subtitle="unit subtitle",
        markers=(
            Order9C3ViewerMarker(
                marker_id="p2",
                label="P2",
                position_world=(0.0, 0.0, 0.0),
                direction_world=(1.0, 0.0, 0.0),
                selected=True,
            ),
        ),
        boxes=(
            Order9C3ViewerBox(
                object_id="object",
                pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                size_m=(0.3, 0.4, 0.15),
            ),
        ),
        metadata={
            "evidence": "unit",
            "review": {
                "case_id": "unit-case",
                "api_path": "/api/order9-c3-review",
            },
        },
    )

    assert artifacts.viewer_version == ORDER9_C3_CURATION_VIEWER_VERSION
    assert artifacts.html_path.is_file()
    assert artifacts.scene_path.is_file()
    assert artifacts.mesh_library_sha256
    page = artifacts.html_path.read_text(encoding="utf-8")
    assert "../shared/mesh_library.js" in page
    assert "../shared/viewer.js" in page
    assert "not Isaac collision" in page
    assert "data-review-action=\"accept\"" in page
    assert "data-review-action=\"reject\"" in page
    assert "data-review-action=\"recalculate\"" in page
    javascript = viewer_js.read_text(encoding="utf-8")
    assert "/api/order9-c3-review" in javascript
    assert "downloadReviewRequest" in javascript
    archived = json.loads(artifacts.scene_path.read_text(encoding="utf-8"))
    assert archived["metadata"]["evidence"] == "unit"
    assert archived["markers"][0]["selected"] is True
    assert archived["boxes"][0]["size_m"] == [0.3, 0.4, 0.15]
    assert archived["metadata"]["review"]["case_id"] == "unit-case"

    animation = {
        "animation_version": "unit-animation-v1",
        "frames_per_second": 10,
        "frame_count": 2,
        "duration_s": 0.1,
        "frames": [
            {
                "frame_index": index,
                "time_s": 0.1 * index,
                "phase": "approach",
                "window_index": 0,
                "window_local_time_s": 0.1 * index,
                "model_matrices": [
                    list(value["model_matrix"])
                    for value in scene.instances
                ],
                "label_positions": [
                    list(value["position_world"])
                    for value in scene.module_labels
                ],
                "box_poses": [
                    [0.0, 0.0, 0.1 * index, 0.0, 0.0, 0.0, 1.0]
                ],
            }
            for index in range(2)
        ],
    }
    animated = render_order9_c3_mesh_viewer(
        urdf_scene=scene,
        html_path=tmp_path / "animated/nominal.html",
        mesh_library_path=library_path,
        viewer_javascript_path=viewer_js,
        title="unit animation",
        subtitle="ideal tracking",
        boxes=(
            Order9C3ViewerBox(
                object_id="object",
                pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                size_m=(0.3, 0.4, 0.15),
            ),
        ),
        animation=animation,
        semantic_scope="unit nominal animation",
    )
    animated_page = animated.html_path.read_text(encoding="utf-8")
    animated_scene = json.loads(
        animated.scene_path.read_text(encoding="utf-8")
    )
    assert 'id="animation-play"' in animated_page
    assert 'id="animation-slider"' in animated_page
    assert animated_scene["animation"]["frame_count"] == 2
    assert len(animated_scene["animation"]["frames"]) == 2
    assert animated_scene["semantic_scope"] == "unit nominal animation"


def test_curation_viewer_rejects_ascii_stl(tmp_path: Path) -> None:
    mesh_path = tmp_path / "not_binary.STL"
    mesh_path.write_text(
        "solid unit\nendsolid unit\n",
        encoding="utf-8",
    )
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(_urdf(mesh_path), encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="STL"):
        build_order9_c3_urdf_mesh_scene(urdf_path)
