from dataclasses import fields
import pytest
import torch

from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.controllers.articulated_joint_load import ArticulatedJointLoadModel
from amsrr.controllers.native_cpu_dynamics import load_native
from amsrr.feasibility.articulated_reachability import (
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)


def builders(contact_count=2):
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    refs = resolve_mesh_backed_anchor_references(
        morphology,
        physical,
        [a.anchor_id for a in morphology.robot_anchors[:contact_count]],
    )
    a, b = [BatchedRigidBodyControlModelBuilder(morphology, physical) for _ in range(2)]
    for obj in (a, b):
        obj._internal_load_model = ArticulatedJointLoadModel(morphology, physical, refs)
    b._use_cpu_compile = b._use_cpu_native = True
    return a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("contacts", [0, 2, 3])
def test_native_model_preserves_all_fields_fresh_state_and_output_ownership(
    dtype, contacts
):
    try:
        load_native()
    except RuntimeError as error:
        pytest.skip(str(error))
    eager, native = builders(contacts)
    torch.manual_seed(271)
    previous = None
    for batch in (1, 3, 1):
        pose = torch.randn(batch, eager.module_count, 7, dtype=dtype) * 0.3
        pose[..., 6] += 1.0
        twist = torch.randn(batch, eager.module_count, 6, dtype=dtype) * 0.2
        q = (
            torch.randn(batch, eager.module_count, eager.local_joint_count, dtype=dtype)
            * 0.2
        )
        rate = torch.randn_like(q) * 0.3
        inputs = dict(
            module_pose_world=pose,
            module_twist_world=twist,
            local_joint_positions_rad=q,
            local_joint_velocities_rad=rate,
        )
        with torch.no_grad():
            expected, actual = eager.build(**inputs), native.build(**inputs)
        for f in fields(expected):
            left, right = getattr(expected, f.name), getattr(actual, f.name)
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(
                    left,
                    right,
                    atol=3e-5 if dtype == torch.float32 else 2e-11,
                    rtol=3e-5 if dtype == torch.float32 else 2e-11,
                )
            else:
                assert left == right
        if previous:
            old, snapshots = previous
            for name, value in snapshots.items():
                assert torch.equal(getattr(old, name), value)
        previous = actual, {
            f.name: getattr(actual, f.name).clone()
            for f in fields(actual)
            if isinstance(getattr(actual, f.name), torch.Tensor)
        }
    pose[..., 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        native.build(**inputs)


def test_native_opt_in_retains_autograd_tensor_path(monkeypatch):
    import amsrr.controllers.native_cpu_dynamics as implementation

    monkeypatch.setattr(
        implementation,
        "load_native",
        lambda: pytest.fail("autograd must not enter native arrays"),
    )
    eager, native = builders()
    q = torch.full(
        (1, eager.module_count, eager.local_joint_count),
        0.13,
        dtype=torch.float64,
        requires_grad=True,
    )
    pose = torch.zeros(1, eager.module_count, 7, dtype=torch.float64)
    pose[..., 6] = 1.0
    inputs = dict(
        module_pose_world=pose,
        module_twist_world=torch.zeros(1, eager.module_count, 6, dtype=torch.float64),
        local_joint_positions_rad=q,
        local_joint_velocities_rad=torch.zeros_like(q),
    )
    a = eager.build(**inputs)
    b = native.build(**inputs)
    expected = torch.autograd.grad(a.joint_gravity_load_nm.square().sum(), q)[0]
    actual = torch.autograd.grad(b.joint_gravity_load_nm.square().sum(), q)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_post_step_native_cache_does_not_hide_new_gradient_requirement():
    try:
        load_native()
    except RuntimeError as error:
        pytest.skip(str(error))
    _, native = builders()
    native._reuse_post_step_model = True
    q = torch.full((1, native.module_count, native.local_joint_count), .13, dtype=torch.float64)
    pose = torch.zeros(1, native.module_count, 7, dtype=torch.float64);pose[..., 6]=1.
    inputs = dict(module_pose_world=pose, module_twist_world=torch.zeros(1,native.module_count,6,dtype=torch.float64),
        local_joint_positions_rad=q, local_joint_velocities_rad=torch.zeros_like(q))
    native.build_kinematics(**inputs)
    q.requires_grad_()
    result = native.build(**inputs)
    gradient = torch.autograd.grad(result.joint_gravity_load_nm.square().sum(), q)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
