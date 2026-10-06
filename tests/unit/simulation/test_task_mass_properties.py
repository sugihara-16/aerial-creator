from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from amsrr.simulation.task_mass_properties import inertia_matrix, principal_mass_properties, validate_task_mass_properties, adapt_task_mass_properties_source
from amsrr.training.request_object_conditions import ballasted_box_mass_properties, _preserve_bottom


def test_ballast_matches_independent_weighted_cuboid_corners():
    size = np.array([.3,.4,.2]); mass=1.2; com=np.array([.01,-.014,.008])
    p=ballasted_box_mass_properties(size,mass,com)
    # Eight points at half extents/sqrt(3) exactly reproduce a uniform cuboid's second moments.
    import itertools
    inertia=np.zeros((3,3)); center=np.zeros(3)
    for fraction,scale,origin in ((.8,1.,np.zeros(3)),(.2,.2,com/.2)):
        for sign in itertools.product((-1,1),repeat=3):
            x=origin+np.array(sign)*size*scale/(2*np.sqrt(3)); m=mass*fraction/8
            center+=m*x; d=x-com; inertia+=m*(d@d*np.eye(3)-np.outer(d,d))
    np.testing.assert_allclose(center/mass,com,atol=1e-15)
    np.testing.assert_allclose(inertia_matrix(p.inertia_kgm2),inertia,atol=1e-15)
    with pytest.raises(ValueError):ballasted_box_mass_properties(size,mass,[.1,0,0])


def test_full_tensor_readback_catches_wrong_principal_frame():
    p=ballasted_box_mass_properties([.3,.4,.2],1.2,[.01,-.014,.008])
    eigen,q=principal_mass_properties(p.inertia_kgm2)
    matrix=inertia_matrix(p.inertia_kgm2)
    data=SimpleNamespace(body_mass=torch.tensor([[p.mass_kg]]),body_inertia=torch.tensor(matrix.reshape(1,1,9)),body_com_pose_b=torch.tensor([[p.center_of_mass_object+q.tolist()]]))
    kw=dict(expected_mass_kg=p.mass_kg,expected_inertia_kgm2=p.inertia_kgm2,expected_com_object=p.center_of_mass_object)
    assert validate_task_mass_properties(SimpleNamespace(data=data),**kw)['matches_task_spec']
    # Same eigenvalues, incorrectly expressed in principal rather than object axes.
    data.body_inertia=torch.tensor(np.diag(eigen).reshape(1,1,9))
    with pytest.raises(RuntimeError,match='full inertia'):validate_task_mass_properties(SimpleNamespace(data=data),**kw)


def test_bottom_preserved_for_tilted_geometry():
    pose=[1,2,3,*Rotation.from_euler('xyz',[.1,.2,.3]).as_quat()]
    old=np.array([.3,.4,.2]); new=np.array([.35,.32,.24])
    after=_preserve_bottom(pose,old,new)
    row=np.abs(Rotation.from_quat(pose[3:]).as_matrix()[2])
    assert after[2]-row@new/2==pytest.approx(pose[2]-row@old/2)
    assert after[:2]+after[3:]==pose[:2]+pose[3:]


def test_harness_boundary_is_exact_and_not_written():
    from pathlib import Path
    source=Path('scripts/order9_vectorized_isaac_rollout.py').read_text()
    updated=adapt_task_mass_properties_source(source)
    assert updated.count('obj.write_root_link_velocity_to_sim_index(')==3
    assert 'return _torch(obj.data.root_link_vel_w)' in updated
    assert source==Path('scripts/order9_vectorized_isaac_rollout.py').read_text()
