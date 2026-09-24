import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
import amsrr.controllers.grasp_slip_compensation as m


@pytest.mark.parametrize('size', [2, 7, 34])
@pytest.mark.parametrize('conditioning', ['full_rank', 'rank_deficient', 'scaled'])
def test_native_box_solver_matches_independent_bvls_objective(size, conditioning):
    from scipy.optimize import lsq_linear
    from amsrr.feasibility.order9_native_loader import load_order9_posture_native

    rng = np.random.default_rng(840 + size)
    a = rng.normal(size=(size + 20, size))
    b = rng.normal(size=len(a)) * 8
    if conditioning == 'rank_deficient':
        a[:, -1] = a[:, 0]
    elif conditioning == 'scaled':
        a = np.vstack((a * 1e3, np.eye(size) * .1))
        b = np.r_[b * 1e3, np.zeros(size)]
    lo, hi = rng.uniform(-.03, 0, size), rng.uniform(.01, .03, size)
    x, _, iterations, _ = load_order9_posture_native().bounded_least_squares(
        a, b, lo, hi, 180, 1e-9)
    reference = lsq_linear(a, b, bounds=(lo, hi), method='bvls',
                           tol=1e-10, max_iter=180)
    assert np.all(x >= lo) and np.all(x <= hi)
    assert 1 <= iterations <= 180
    cost = np.linalg.norm(a @ x - b) ** 2
    reference_cost = np.linalg.norm(a @ reference.x - b) ** 2
    assert abs(cost - reference_cost) <= 1e-8 * (1 + reference_cost)


@pytest.mark.parametrize('invalid', ['nan_matrix', 'reversed_bounds', 'zero_budget', 'nan_tolerance'])
def test_native_box_solver_rejects_invalid_problem(invalid):
    from amsrr.feasibility.order9_native_loader import load_order9_posture_native

    a, b, lo, hi = np.eye(2), np.ones(2), -np.ones(2), np.ones(2)
    budget, tolerance = 180, 1e-9
    if invalid == 'nan_matrix':
        a[0, 0] = np.nan
    elif invalid == 'reversed_bounds':
        lo[0] = 2
    elif invalid == 'zero_budget':
        budget = 0
    else:
        tolerance = np.nan
    with pytest.raises(ValueError):
        load_order9_posture_native().bounded_least_squares(a, b, lo, hi, budget, tolerance)


def pose(p, r=None):
    return np.r_[p, (r or R.identity()).as_quat()]


def rigid_grasp_fk(q):
    # Analytic articulated fixture: a joint rotates both hands about their
    # common grasp centre, unlike a body rotation about the distant CoM.
    r = R.from_rotvec(np.column_stack((q[:, 0] * 0, q[:, 0] * 0, q[:, 0]))).as_matrix()
    p = np.einsum('bij,aj->bai', r, np.array([[0., -.2, 0.], [0., .2, 0.]])) + [0.8, 0., 0.]
    return p, np.repeat(r[:, None], 2, axis=1)


def servo():
    return m.ObservedGraspOrientationServo(anchor_fk=rigid_grasp_fk,
        joint_lower=[-.4], joint_upper=[.4], joint_speeds=[.2])


def test_zero_error_does_not_feed_measured_body_drift_into_command():
    c = servo();b = pose([0, 0, 1]);obj = pose([.8, 0, 1])
    x, req = c.propose(b, np.zeros(1), np.zeros(1), obj, obj, weight=1, dt=.02)
    np.testing.assert_allclose(x, 0, atol=1e-12)
    np.testing.assert_allclose(c.command(b, x, .02)[0], b, atol=1e-12)


@pytest.mark.parametrize('dt', [.01, .02, .05])
def test_combined_correction_uses_joint_motion_and_preserves_rigid_grasp(dt):
    c = servo();b = pose([0, 0, 1]);obj = pose([.8, 0, 1]);measured = pose([.8, 0, 1], R.from_euler('z', .12))
    for _ in range(round(3 / dt)):
        x, req = c.propose(b, np.zeros(1), np.zeros(1), obj, measured, weight=1, dt=dt)
        _, v, qd = c.command(b, x, dt)
        assert np.linalg.norm(x[:3]) <= .03 + 1e-10
        assert np.linalg.norm(x[3:6]) <= .2 + 1e-10
        assert np.max(abs(x[6:])) <= .1 + 1e-10
        assert np.linalg.norm(v[:3]) <= .02 + 1e-10
        assert np.linalg.norm(v[3:]) <= .15 + 1e-10
        assert np.max(abs(qd)) <= .05 + 1e-10
        c.accept(x, req)
    assert abs(x[6]) > .07
    assert c.diagnostics['target_orientation_error_rad'] < .003
    assert c.diagnostics['relative_grasp_position_error_m'] < 1e-10
    assert c.diagnostics['relative_grasp_angle_error_rad'] < 1e-10


def test_world_translation_and_rotation_covariance_below_active_bounds():
    c1, c2 = servo(), servo();b = pose([0, 0, 1]);goal = pose([.8, 0, 1]);measured = pose([.8, 0, 1], R.from_euler('z', .005))
    # Quarter turn preserves the conservative axis-aligned bound boxes.
    turn = R.from_euler('z', np.pi / 2);shift = np.array([5., -3., 2.])
    def tf(p): return pose(turn.apply(p[:3]) + shift, turn * R.from_quat(p[3:]))
    x1, _ = c1.propose(b, np.zeros(1), np.zeros(1), goal, measured, weight=1, dt=.02)
    x2, _ = c2.propose(tf(b), np.zeros(1), np.zeros(1), tf(goal), tf(measured), weight=1, dt=.02)
    np.testing.assert_allclose(x2[:3], turn.apply(x1[:3]), atol=2e-7)
    np.testing.assert_allclose(x2[3:6], turn.apply(x1[3:6]), atol=2e-7)
    np.testing.assert_allclose(x2[6:], x1[6:], atol=2e-7)


def test_joint_limits_total_velocity_and_continuous_release():
    c = servo();b = pose([0, 0, 1]);goal = pose([.8, 0, 1]);measured = pose([.8, 0, 1], R.from_euler('z', -.2));q = np.array([.399])
    x, req = c.propose(b, q, np.array([.19]), goal, measured, weight=1, dt=.02)
    _, _, qd = c.command(b, x, .02)
    assert (q + x[6:] <= .4 + 1e-10).all()
    assert abs(.19 + qd[0]) <= .2 + 1e-10
    c.accept(x, req)
    for _ in range(500):
        x, req = c.propose(b, q, np.zeros(1), goal, measured, weight=0, dt=.02)
        _, v, qd = c.command(b, x, .02)
        assert np.linalg.norm(v[:3]) <= .02 + 1e-10
        assert np.linalg.norm(v[3:]) <= .15 + 1e-10
        assert max(abs(qd)) <= .05 + 1e-10
        c.accept(x, req)
    assert np.linalg.norm(x) < 1e-9


def test_rejected_proposal_does_not_advance_state_and_binding_resets():
    c = servo();c.set_binding('one');b = pose([0, 0, 1]);goal = pose([.8, 0, 1]);measured = pose([.8, 0, 1], R.from_euler('z', .2))
    x, req = c.propose(b, np.zeros(1), np.zeros(1), goal, measured, weight=1, dt=.02)
    np.testing.assert_array_equal(c.offset, 0);np.testing.assert_array_equal(c.request_rotation, 0)
    c.accept(x, req);c.set_binding('one');assert np.linalg.norm(c.offset) > 0
    c.set_binding('two');np.testing.assert_array_equal(c.offset, 0)


@pytest.mark.parametrize('bad', [0, -1, float('nan')])
def test_invalid_interval_rejected(bad):
    p = pose([0, 0, 0])
    with pytest.raises(ValueError):
        servo().propose(p, np.zeros(1), np.zeros(1), p, p, weight=1, dt=bad)


@pytest.mark.parametrize('progress,body_speed,joint_speed,object_speed,object_spin,active', [
    (.5,0.,0.,0.,0.,False),(1.,.01,0.,0.,0.,False),(1.,0.,.01,0.,0.,False),(1.,0.,0.,0.,0.,True),
    (1.,0.,0.,.051,0.,False),(1.,0.,0.,0.,.101,False)])
def test_feedback_only_regulates_a_stationary_endpoint(progress,body_speed,joint_speed,object_speed,object_spin,active):
    from types import SimpleNamespace
    import torch
    from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
    c = m.CheckedGraspSlipController.__new__(m.CheckedGraspSlipController)
    c.dt=.02;c.reserve={'span_m':0.};c.direction=np.zeros(1);c.speeds=np.array([.2])
    c.normals=np.array([[1.,0.,0.]]);c.controllers=[m.MaterialSlipClosure(span_m=0.)]
    c.previous_offsets=np.zeros((1,1));c.trace=[];c.pose_controllers=[servo()]
    c._safe=lambda *args:True
    nominal=torch.tensor([[0.,0.,1.,0.,0.,0.,1.]])
    goal=torch.tensor([[.8,0.,1.,0.,0.,0.,1.]])
    measured=torch.tensor(np.array([pose([.8,0,1],R.from_euler('z',.25))]),dtype=torch.float32)
    twist=torch.zeros(1,6);twist[0,0]=body_speed
    q=torch.zeros(1,1,1);qd=torch.full_like(q,joint_speed)
    target=Order9TensorObjectTaskTarget(nominal,twist,q,qd,goal,nominal,goal,torch.tensor([progress]),torch.tensor([3]))
    supervisor=SimpleNamespace(contact_velocity_observer=SimpleNamespace(previous=None),expected_group='g',
        grip_contact_present=True,environment_origin=torch.zeros(3),time_s=1.,phase=4,final_goal_orientation_tolerance_rad=.2)
    state=SimpleNamespace(object_pose_world=measured,object_twist_world=torch.tensor([[object_speed,0.,0.,0.,0.,object_spin]]))
    result=c.apply(target,state,[supervisor])
    assert (np.linalg.norm(c.pose_controllers[0].offset)>0)==active
    if not active:torch.testing.assert_close(result.desired_robot_root_pose_world,nominal)
    if active:
        supervisor.phase=5
        state.object_pose_world=goal
        c.apply(target,state,[supervisor])
        assert not c.pose_controllers[0].regulating


def test_hold_preserves_the_established_correction_until_release():
    c=servo();b=pose([0,0,1]);goal=pose([.8,0,1]);measured=pose([.8,0,1],R.from_euler('z',.2))
    x,req=c.propose(b,np.zeros(1),np.zeros(1),goal,measured,weight=1,dt=.02)
    c.accept(x,req)
    moved=b.copy();moved[2]-=.01
    held,request=c.propose(moved,np.zeros(1),np.zeros(1),goal,measured,weight=0,dt=.02,hold=True)
    np.testing.assert_array_equal(held,x)
    np.testing.assert_array_equal(request,req)
    _,vel,qd=c.command(moved,held,.02)
    np.testing.assert_allclose(vel,0,atol=1e-12);np.testing.assert_allclose(qd,0,atol=1e-12)
    released,_=c.propose(moved,np.zeros(1),np.zeros(1),goal,measured,weight=0,dt=.02)
    _, before=c.poses(moved,np.zeros(1),held)
    _, after=c.poses(moved,np.zeros(1),released)
    assert np.max(R.from_matrix(after[0]).magnitude()) < np.max(R.from_matrix(before[0]).magnitude())
