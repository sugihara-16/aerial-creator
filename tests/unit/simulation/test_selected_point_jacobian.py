from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from pxr import Gf, Usd, UsdPhysics

from amsrr.simulation.selected_contact_jacobian import SelectedPointJacobian


def fixture():
    wp.init()
    stage = Usd.Stage.CreateInMemory()
    paths = [
        "/Robot/root",
        "/Robot/root/base",
        "/Robot/root/arm",
        "/Robot/root/arm/tip",
    ]
    for p in ["/Robot", "/Robot/Physics", *paths]:
        stage.DefinePrim(p, "Xform")

    def joint(cls, name, a, b, position=(0.0, 0.0, 0.0)):
        q = cls.Define(stage, "/Robot/Physics/" + name)
        q.GetBody0Rel().SetTargets([a])
        q.GetBody1Rel().SetTargets([b])
        q.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
        q.GetLocalRot0Attr().Set(Gf.Quatf(1.0))
        if cls is UsdPhysics.RevoluteJoint:
            q.GetAxisAttr().Set("Z")
        return q

    joint(UsdPhysics.FixedJoint, "fixed", paths[0], paths[1])
    # Authored body0 lies downstream: the DOF's sign must be reversed.
    joint(UsdPhysics.RevoluteJoint, "reverse", paths[2], paths[1], (0.0, -1.0, 0.0))
    joint(UsdPhysics.RevoluteJoint, "distal", paths[2], paths[3])
    poses = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    data = SimpleNamespace(
        _root_view=SimpleNamespace(
            link_paths=[paths],
            dof_paths=[["/Robot/Physics/reverse", "/Robot/Physics/distal"]],
        ),
        body_com_jacobian_w=torch.zeros(1, 4, 6, 8),
        body_link_pose_w=SimpleNamespace(
            shape=poses.shape, warp=wp.from_torch(poses, dtype=wp.transformf)
        ),
    )
    return stage, data, poses


def test_selected_joint_point_jacobian_matches_independent_finite_difference():
    stage, data, poses = fixture()
    calc = SelectedPointJacobian(
        data, torch.tensor([2, 3]), torch.tensor([6, 7]), stage=stage
    )
    points = poses[:, [2, 3], :3].contiguous()
    actual = calc(data, points)[0].numpy()

    def position(q):
        def rz(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        arm = np.array([1.0, 0.0, 0.0]) + rz(-q[0]) @ np.array([0.0, 1.0, 0.0])
        tip = arm + rz(-q[0]) @ rz(q[1]) @ np.array([1.0, 0.0, 0.0])
        return np.stack([arm, tip])

    eps = 1e-5
    expected = np.stack(
        [
            (position(np.eye(2)[i] * eps) - position(-np.eye(2)[i] * eps)) / (2 * eps)
            for i in range(2)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    saved = actual.copy()
    # Fresh measured points must be consumed without mutating previous output.
    shifted = calc(data, points + torch.tensor([0.0, 0.1, 0.0]))
    assert not np.array_equal(shifted[0].numpy(), saved)
    np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize(
    "bodies,columns", [([9], [6]), ([2], [0]), ([2], [8]), ([], [6]), ([2], [])]
)
def test_selected_point_rejects_invalid_or_floating_root_columns(bodies, columns):
    stage, data, _ = fixture()
    with pytest.raises(ValueError):
        SelectedPointJacobian(
            data,
            torch.tensor(bodies, dtype=torch.int64),
            torch.tensor(columns, dtype=torch.int64),
            stage=stage,
        )


def test_selected_point_rejects_disconnected_tree_and_wrong_point_shape():
    stage, data, _ = fixture()
    calc = SelectedPointJacobian(
        data, torch.tensor([2]), torch.tensor([6]), stage=stage
    )
    with pytest.raises(ValueError):
        calc(data, torch.zeros(1, 2, 3))
    stage.RemovePrim("/Robot/Physics/fixed")
    with pytest.raises(ValueError, match="connected"):
        SelectedPointJacobian(data, torch.tensor([2]), torch.tensor([6]), stage=stage)


@pytest.mark.parametrize(
    "bad",
    [torch.tensor([1.5]), torch.tensor([True]), torch.tensor([1], dtype=torch.int32)],
)
def test_selected_point_rejects_non_int64_indices_before_cached_access(bad):
    from amsrr.simulation.selected_contact_jacobian import selected_point_jacobians

    stage, data, _ = fixture()
    for bodies, columns in [(bad, torch.tensor([6])), (torch.tensor([2]), bad)]:
        with pytest.raises(ValueError, match="int64"):
            SelectedPointJacobian(data, bodies, columns, stage=stage)
        with pytest.raises(ValueError, match="int64"):
            selected_point_jacobians(data, bodies, columns, torch.zeros(1, 1, 3))
