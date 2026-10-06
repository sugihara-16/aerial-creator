"""Apply TaskSpec mass properties without conflating link and inertial frames."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def inertia_matrix(values):
    a, b, c, d, e, f = map(float, values)
    return np.array([[a, b, c], [b, d, e], [c, e, f]])


def principal_mass_properties(inertia):
    matrix = inertia_matrix(inertia)
    values, axes = np.linalg.eigh(matrix)
    if not np.isfinite(matrix).all() or min(values) <= 0 or max(values) > sum(values) / 2 + 1e-9:
        raise ValueError('invalid physical inertia about object COM')
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    return values, Rotation.from_matrix(axes).as_quat()


def apply_task_mass_properties(stage, object_spec, *, environment_count, environment_prim_paths=None):
    """Author before simulation reset; Task inertia is about COM in object axes."""
    from pxr import Gf, UsdPhysics
    values, quaternion = principal_mass_properties(object_spec.inertia_kgm2)
    paths = ([f'/World/envs/env_{index}' for index in range(environment_count)]
             if environment_prim_paths is None else list(environment_prim_paths))
    if len(paths) != environment_count or len(set(paths)) != environment_count:
        raise ValueError('object mass-property environment paths differ')
    for path in paths:
        prim = stage.GetPrimAtPath(path + '/Object')
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('object rigid-body prim missing at mass-property boundary')
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(object_spec.mass_kg))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, object_spec.center_of_mass_object or (0, 0, 0))))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*map(float, quaternion[:3]))))


def validate_task_mass_properties(obj, *, expected_mass_kg, expected_inertia_kgm2, expected_com_object):
    from amsrr.simulation.order9_tensor_isaac_io import _torch
    mass = _torch(obj.data.body_mass)[:, 0].detach().cpu().numpy()
    local = _torch(obj.data.body_inertia)[:, 0].detach().cpu().numpy().reshape(-1, 3, 3)
    pose = _torch(obj.data.body_com_pose_b)[:, 0].detach().cpu().numpy()
    # PhysX RigidBodyView.get_inertias(): about COM, expressed in the
    # rigid-body-prim frame (not the principal frame returned by get_coms).
    actual = local
    expected = inertia_matrix(expected_inertia_kgm2)
    if not np.allclose(mass, expected_mass_kg, rtol=5e-4, atol=1e-6):
        raise RuntimeError('Isaac object mass differs from TaskSpec')
    if not np.allclose(pose[:, :3], expected_com_object, rtol=0, atol=1e-5):
        raise RuntimeError('Isaac object COM differs from TaskSpec')
    if not np.allclose(actual, expected, rtol=5e-3, atol=1e-6):
        raise RuntimeError(f'Isaac object full inertia in object axes differs: {actual[0]} vs {expected}')
    return dict(mass_kg=float(mass[0]), center_of_mass_object=pose[0, :3].tolist(),
                inertia_eigenvalues_kgm2=np.linalg.eigvalsh(actual[0]).tolist(),
                inertia_object_axes_kgm2=actual[0].tolist(),
                principal_axes_xyzw=pose[0, 3:].tolist(), matches_task_spec=True)


def adapt_task_mass_properties_source(source):
    marker = '    _activate_nested_contact_reports(sim.stage)\n    sim.reset()'
    if source.count(marker) != 1:
        raise ValueError('task mass-property initialization boundary changed')
    source = source.replace(marker, '    apply_task_mass_properties(sim.stage, target_object, environment_count=scene.num_envs, environment_prim_paths=scene.env_prim_paths)\n' + marker)
    old = '''    return torch.cat(
        (_torch(obj.data.root_lin_vel_w), _torch(obj.data.root_ang_vel_w)),
        dim=-1,
    )'''
    if source.count(old) != 1 or source.count('obj.write_root_velocity_to_sim_index(') != 3:
        raise ValueError('object link twist boundary changed')
    source = source.replace(old, '    return _torch(obj.data.root_link_vel_w)')
    return source.replace('obj.write_root_velocity_to_sim_index(', 'obj.write_root_link_velocity_to_sim_index(')
