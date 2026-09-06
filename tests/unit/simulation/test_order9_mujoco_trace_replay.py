from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from amsrr.simulation.order9_mujoco_trace_replay import (
    hash_file,
    load_isaac_episode_trace,
    materialize_convex_hull_urdf,
)


def _write_tetrahedron_stl(path: Path) -> None:
    vertices = np.asarray(
        ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]),
        dtype=np.float32,
    )
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    payload = bytearray(84 + len(faces) * 50)
    struct.pack_into("<I", payload, 80, len(faces))
    for index, face in enumerate(faces):
        triangle = vertices[list(face)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        struct.pack_into(
            "<12fH",
            payload,
            84 + 50 * index,
            *(normal.tolist() + triangle.reshape(-1).tolist()),
            0,
        )
    path.write_bytes(payload)


def test_materialize_convex_hull_urdf_is_deterministic(tmp_path: Path) -> None:
    mesh = tmp_path / "part.stl"
    _write_tetrahedron_stl(mesh)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<?xml version='1.0'?>
<robot name='test'>
  <link name='base'>
    <inertial><mass value='1'/><inertia ixx='.1' ixy='0' ixz='0' iyy='.1' iyz='0' izz='.1'/></inertial>
    <collision><geometry><mesh filename='part.stl'/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    first, first_manifest = materialize_convex_hull_urdf(urdf, tmp_path / "out")
    first_hash = hash_file(first)
    second, second_manifest = materialize_convex_hull_urdf(urdf, tmp_path / "out")

    assert first == second
    assert hash_file(second) == first_hash
    assert first_manifest == second_manifest
    assert first_manifest["meshes"][0]["convex_vertex_count"] == 4
    assert first_manifest["meshes"][0]["convex_triangle_count"] == 4


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="MuJoCo is an optional diagnostic dependency",
)
def test_materialized_convex_hull_urdf_loads_in_mujoco(tmp_path: Path) -> None:
    import mujoco

    mesh = tmp_path / "part.stl"
    _write_tetrahedron_stl(mesh)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<?xml version='1.0'?>
<robot name='test'>
  <link name='base'>
    <inertial><mass value='1'/><inertia ixx='.1' ixy='0' ixz='0' iyy='.1' iyz='0' izz='.1'/></inertial>
    <collision><geometry><mesh filename='part.stl'/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    generated, _ = materialize_convex_hull_urdf(urdf, tmp_path / "out")

    model = mujoco.MjModel.from_xml_path(str(generated))

    assert model.nmesh == 1
    assert model.ngeom == 1


def test_load_trace_selects_one_contiguous_terminal_episode(tmp_path: Path) -> None:
    path = tmp_path / "rollout.pt"
    torch.save(
        {
            "artifact_version": "unit",
            "metadata": {},
            "tensors": {
                "valid": torch.tensor([[True], [True], [True], [True]]),
                "episode_serial": torch.tensor([[0], [0], [0], [1]]),
                "terminal": torch.tensor([[False], [False], [True], [False]]),
            },
        },
        path,
    )

    trace = load_isaac_episode_trace(path)

    assert trace.trace_indices.tolist() == [0, 1, 2]
    assert trace.control_step_count == 2
