from __future__ import annotations

import numpy as np
import pytest

from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
    _minimum_peak_utilization_force_allocation,
    solve_contact_force_equilibrium,
    solve_order9_actuator_aware_nominal_preload_from_jacobian,
)


def _unequal_lever_force_allocation(wrench):
    return _minimum_peak_utilization_force_allocation(
        jacobian=np.array([[2., 0.], [0., .5]]), friction=np.array([.6, .6]),
        peak_effort_limit_nm=np.array([4.1, 4.1]), required_support_force_n=2.4,
        minimum_anchor_force_fraction=.5, internal_wrench=wrench)


def test_two_contact_squeeze_balances_force_and_moment_with_unequal_levers():
    from scipy.spatial.transform import Rotation
    positions = np.array([[-.2, 0., .3], [.2, 0., .3]])
    normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
    def wrench(p, n):
        return np.vstack((n.T, np.cross(p - p.mean(0), n).T))
    old = _unequal_lever_force_allocation(None)
    assert abs(old[0] - old[1]) > 1.
    matrix = wrench(positions, normals)
    forces = _unequal_lever_force_allocation(matrix)
    np.testing.assert_allclose(forces, [2., 2.], atol=1e-8)
    np.testing.assert_allclose(matrix @ forces, 0., atol=1e-8)
    rotation = Rotation.from_rotvec([.3, .7, -.4]).as_matrix()
    moved = wrench(positions @ rotation.T + [3., 2., 1.], normals @ rotation.T)
    np.testing.assert_allclose(_unequal_lever_force_allocation(moved), forces, atol=1e-8)


@pytest.mark.parametrize('couple', [False, True])
def test_non_equilibrated_normal_squeeze_is_rejected(couple):
    positions = np.array([[-.2, 0., 0.], [.2, .1, 0.]])
    normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
    matrix = (np.vstack((normals.T, np.cross(positions - positions.mean(0), normals).T))
              if couple else np.array([[1., 0.], [0., 1.], [0., 0.]]))
    with pytest.raises(ValueError, match='infeasible'):
        _unequal_lever_force_allocation(matrix)


@pytest.mark.parametrize('matrix', [np.ones((6, 3)), np.full((6, 2), np.nan), np.ones(2)])
def test_invalid_internal_squeeze_matrix_is_rejected(matrix):
    with pytest.raises(ValueError, match='invalid internal squeeze'):
        _unequal_lever_force_allocation(matrix)


@pytest.mark.parametrize('count', [3, 4])
def test_multicontact_support_balances_force_and_com_moment(count):
    angle = np.arange(count) * 2*np.pi/count
    positions = np.c_[.15*np.cos(angle), .15*np.sin(angle), np.zeros(count)]
    normals = -positions / .15
    center = np.array([.025, -.015, .04])
    jac = np.zeros((count, 3, 3*count))
    for i in range(count):
        jac[i, :, 3*i:3*i+3] = .02*np.eye(3)
    forces, residual = solve_contact_force_equilibrium(
        positions=positions, inward_normals=normals, center_of_mass=center,
        friction=np.full(count, .6/1.25), required_force=[0,0,9.80665],
        position_jacobian=jac, effort_limits=np.full(3*count,4.1),
        maximum_utilization=.8, minimum_normal_force=1.)
    assert residual < 1e-8
    assert forces.sum(0) == pytest.approx([0,0,9.80665],abs=1e-8)
    assert np.cross(positions-center,forces).sum(0) == pytest.approx([0,0,0],abs=1e-8)
    normal_force = (forces*normals).sum(-1)
    tangent = forces-normal_force[:,None]*normals
    assert np.all(normal_force >= 1.-1e-8)
    assert np.all(np.linalg.norm(tangent,axis=-1) <= .6/1.25*normal_force+1e-8)
    torque = np.einsum('aij,ai->j',jac,forces)
    assert np.max(np.abs(torque)) <= .8*4.1+1e-8


def test_multicontact_rejects_unresistable_contact_line_moment():
    # Three collinear points cannot counter gravity about their joining axis
    # when the COM lies off that axis, regardless of normal squeeze strength.
    with pytest.raises(ValueError,match='force/moment equilibrium'):
        solve_contact_force_equilibrium(positions=[[-.1,0,0],[0,0,0],[.1,0,0]],
            inward_normals=[[1,0,0],[-1,0,0],[-1,0,0]], center_of_mass=[0,.03,0],
            friction=[.6]*3, required_force=[0,0,9.8],
            position_jacobian=np.zeros((3,3,1)),effort_limits=[4.1],
            maximum_utilization=.8,minimum_normal_force=1.)


def test_multi_preload_requires_geometry_instead_of_silently_using_pair_model():
    with pytest.raises(ValueError,match='requires contact geometry'):
        solve_order9_actuator_aware_nominal_preload_from_jacobian(
            inward_normal_joint_jacobian_m=np.eye(3),joint_ids=('a','b','c'),
            joint_stiffness_nm_per_rad=[200]*3,joint_peak_effort_limit_nm=[4.1]*3,
            joint_continuous_effort_limit_nm=[1.3]*3,object_mass_kg=1.,
            contact_friction=.6,contact_stiffness_n_per_m=8000.)


def test_preload_resolver_accounts_for_joint_and_contact_compliance() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=float
        ),
        joint_ids=("left", "right"),
        joint_stiffness_nm_per_rad=(200.0, 200.0),
        joint_peak_effort_limit_nm=(4.1, 4.1),
        joint_continuous_effort_limit_nm=(1.3, 1.3),
        object_mass_kg=1.0,
        contact_friction=4.5,
        contact_stiffness_n_per_m=8000.0,
    )

    # 1 kg * g * 1.25, shared by two mu=4.5 contacts.
    assert result.target_normal_force_n_by_anchor == pytest.approx(
        (1.3620347, 1.3620347), rel=1.0e-5
    )
    assert result.inward_lead_m == pytest.approx(0.007)
    assert result.feasible
    assert result.peak_effort_utilization < 0.8
    assert result.continuous_effort_utilization > 1.0


def test_preload_resolver_adds_common_calibrated_model_error_margin() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=float
        ),
        joint_ids=("left", "right"),
        joint_stiffness_nm_per_rad=(200.0, 200.0),
        joint_peak_effort_limit_nm=(4.1, 4.1),
        joint_continuous_effort_limit_nm=(1.3, 1.3),
        object_mass_kg=1.0,
        contact_friction=4.5,
        contact_stiffness_n_per_m=8000.0,
        config=Order9ActuatorAwareNominalPreloadConfig(
            model_error_margin_m=0.007
        ),
    )

    assert result.compliance_predicted_inward_lead_m == pytest.approx(
        0.0069804, rel=1.0e-4
    )
    assert result.model_error_margin_m == pytest.approx(0.007)
    assert result.inward_lead_m == pytest.approx(0.014)
    assert result.feasible


def test_preload_resolver_rejects_peak_effort_infeasibility() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[2.0], [2.0]], dtype=float
        ),
        joint_ids=("shared",),
        joint_stiffness_nm_per_rad=(200.0,),
        joint_peak_effort_limit_nm=(4.1,),
        joint_continuous_effort_limit_nm=(1.3,),
        object_mass_kg=1.0,
        contact_friction=1.0,
        contact_stiffness_n_per_m=8000.0,
    )

    assert not result.feasible
    assert result.rejection_reason == "peak_effort_utilization"


def test_preload_resolver_rejects_excessive_servo_lead() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[1.8, 0.0], [0.0, 1.8]], dtype=float
        ),
        joint_ids=("left", "right"),
        joint_stiffness_nm_per_rad=(50.0, 50.0),
        joint_peak_effort_limit_nm=(20.0, 20.0),
        joint_continuous_effort_limit_nm=(10.0, 10.0),
        object_mass_kg=1.0,
        contact_friction=4.5,
        contact_stiffness_n_per_m=8000.0,
        config=Order9ActuatorAwareNominalPreloadConfig(
            maximum_inward_lead_m=0.040
        ),
    )

    assert not result.feasible
    assert result.rejection_reason == "nominal_inward_lead"
    assert result.inward_lead_m == pytest.approx(0.040)
