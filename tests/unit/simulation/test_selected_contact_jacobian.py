from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize("offset", [0, 1])
def test_selected_jacobians_match_com_shift_and_own_outputs(offset):
    wp = pytest.importorskip("warp")
    from amsrr.simulation.selected_contact_jacobian import selected_link_jacobians

    wp.init()
    torch.manual_seed(17)
    pose = torch.randn(2, 6, 7)
    pose[..., 3:] /= pose[..., 3:].norm(dim=-1, keepdim=True)
    com = torch.randn(2, 6, 3)
    source = torch.randn(2, 6 - offset, 6, 9)

    def proxy(tensor, dtype):
        return SimpleNamespace(
            shape=tensor.shape, warp=wp.from_torch(tensor, dtype=dtype)
        )

    data = SimpleNamespace(
        body_link_pose_w=proxy(pose, wp.transformf),
        body_com_pos_b=proxy(com, wp.vec3f),
        body_com_jacobian_w=proxy(source, wp.float32),
    )
    ids = torch.tensor([5, 2, 5])
    actual = selected_link_jacobians(data, ids)
    q = pose[:, ids, 3:]
    xyz, w = q[..., :3], q[..., 3:]
    c = com[:, ids]
    rotated = c + 2 * (
        w * torch.cross(xyz, c, dim=-1)
        + torch.cross(xyz, torch.cross(xyz, c, dim=-1), dim=-1)
    )
    expected = source[:, ids - offset].clone()
    expected[..., :3, :] -= torch.cross(
        expected[..., 3:, :].transpose(-1, -2), rotated[..., None, :], dim=-1
    ).transpose(-1, -2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    saved = actual.clone()
    source.add_(0.7)
    assert not torch.equal(actual, selected_link_jacobians(data, ids))
    assert torch.equal(actual, saved)
    with pytest.raises(ValueError):
        selected_link_jacobians(data, torch.tensor([6]))
    if offset:
        with pytest.raises(ValueError):
            selected_link_jacobians(data, torch.tensor([0]))
