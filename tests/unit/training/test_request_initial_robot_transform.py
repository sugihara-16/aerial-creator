from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from amsrr.training.request_object_conditions import transform_initial_robot


def observation():
    return SimpleNamespace(module_states=[SimpleNamespace(
        pose_world=[x, y, 2., 0., 0., 0., 1.],
        twist_world=[1., 0., 0., 0., 1., 0.], joint_positions={'joint':.3})
        for x,y in [(1.,0.),(2.,0.)]], object_states=[{'pose':[5.,0.,1.]}])


def test_planar_transform_preserves_robot_structure_and_object():
    before=observation();after=transform_initial_robot(deepcopy(before),
        dict(yaw_rad=np.pi/2,translation_world_m=[3.,4.,0.]))
    np.testing.assert_allclose(after.module_states[0].pose_world[:3],[3.,5.,2.],atol=1e-12)
    np.testing.assert_allclose(after.module_states[1].pose_world[:3],[3.,6.,2.],atol=1e-12)
    np.testing.assert_allclose(after.module_states[0].twist_world,[0.,1.,0.,-1.,0.,0.],atol=1e-12)
    assert after.object_states==before.object_states
    assert after.module_states[0].joint_positions==before.module_states[0].joint_positions


@pytest.mark.parametrize('yaw,offset',[(float('nan'),[0.,0.,0.]),(0.,[0.,0.,1.]),(0.,[0.,0.])])
def test_invalid_transform_rejected(yaw,offset):
    with pytest.raises(ValueError):
        transform_initial_robot(observation(),dict(yaw_rad=yaw,translation_world_m=offset))
