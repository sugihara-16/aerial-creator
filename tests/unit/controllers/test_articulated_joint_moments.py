import pytest
import torch

from amsrr.controllers.articulated_joint_load import _joint_mass_moments


def direct_link_sum(coms, inertias, masses, signs, origins, axes, omega):
    jac = torch.cross(axes[:, :, None], coms[:, None]-origins[:, :, None], dim=-1)*signs[None, :, :, None]
    mass = (jac*masses[None, None, :, None]).sum(2)
    center = (coms*masses[None, :, None]).sum(1)/masses.sum()
    lever = coms-center[:, None]
    angular = (torch.cross(lever[:, None].expand_as(jac), jac, dim=-1)*masses[None, None, :, None]).sum(2)
    angular += ((inertias[:, None]@axes[:, :, None, :, None]).squeeze(-1)*signs[None, :, :, None]).sum(2)
    centrifugal = torch.zeros_like(mass[..., 0])
    if omega is not None:
        w = omega[:, None].expand_as(lever)
        force = torch.cross(w, torch.cross(w, lever, dim=-1), dim=-1)*masses[None, :, None]
        torque = torch.cross(w, (inertias@w[..., None]).squeeze(-1), dim=-1)
        centrifugal = (jac*force[:, None]).sum((-1,-2))+(axes[:, :, None]*torque[:, None]*signs[None, :, :, None]).sum((-1,-2))
    return mass, angular, centrifugal


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('moving', [False, True])
def test_moments_equal_independent_per_link_force_and_torque_sum(dtype, moving):
    torch.manual_seed(719)
    b,j,n=5,9,27
    coms=torch.randn(b,n,3,dtype=dtype)
    q=torch.randn(b,n,3,3,dtype=dtype)
    inertias=q@q.transpose(-1,-2)*.03
    masses=torch.rand(n,dtype=dtype)
    masses[::3]=0
    signs=torch.randint(-1,2,(j,n)).to(dtype)
    signs[0]=0
    origins=torch.randn(b,j,3,dtype=dtype)
    axes=torch.randn(b,j,3,dtype=dtype)
    axes=axes/axes.norm(dim=-1,keepdim=True)
    omega=torch.randn(b,3,dtype=dtype) if moving else None
    expected=direct_link_sum(coms,inertias,masses,signs,origins,axes,omega)
    actual=_joint_mass_moments(coms,inertias,masses,signs,origins,axes,omega)
    tolerance=2e-5 if dtype==torch.float32 else 2e-12
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e,atol=tolerance,rtol=tolerance)
        assert torch.count_nonzero(a[:,0]) == 0
    if dtype==torch.float64 and moving:
        args=[coms,inertias,masses,signs,origins,axes,omega]
        indices=(0,1,4,5,6)
        for i in indices: args[i]=args[i].clone().requires_grad_()
        variables=[args[i] for i in indices]
        left=torch.autograd.grad(sum(x.square().sum() for x in _joint_mass_moments(*args)),variables)
        right=torch.autograd.grad(sum(x.square().sum() for x in direct_link_sum(*args)),variables)
        for a,e in zip(left,right):torch.testing.assert_close(a,e,atol=2e-9,rtol=2e-10)
