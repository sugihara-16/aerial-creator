from __future__ import annotations

import math

import pytest

from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import (
    link_poses_at_joint_positions,
)
from amsrr.schemas.common import SchemaValidationError


def test_link_poses_at_joint_positions_applies_root_and_joint_motion(
    tmp_path,
) -> None:
    urdf = tmp_path / "articulated.urdf"
    urdf.write_text(
        """\
<robot name="articulated">
  <link name="root"/>
  <link name="revolute_child"/>
  <link name="prismatic_child"/>
  <joint name="yaw" type="revolute">
    <parent link="root"/>
    <child link="revolute_child"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="revolute_child"/>
    <child link="prismatic_child"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 2"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    poses = link_poses_at_joint_positions(
        load_urdf(urdf),
        {"yaw": math.pi / 2.0, "slide": 0.5},
        root_pose_world=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )

    assert poses["revolute_child"][:3] == pytest.approx((11.0, 0.0, 0.0))
    assert poses["revolute_child"][3:] == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    assert poses["prismatic_child"][:3] == pytest.approx((11.0, 1.0, 0.5))


def test_link_poses_at_joint_positions_rejects_unknown_joint(tmp_path) -> None:
    urdf = tmp_path / "fixed.urdf"
    urdf.write_text(
        """\
<robot name="fixed">
  <link name="root"/>
</robot>
""",
        encoding="utf-8",
    )

    with pytest.raises(SchemaValidationError, match="unknown joints"):
        link_poses_at_joint_positions(load_urdf(urdf), {"other": 0.0})
