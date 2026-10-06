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


def test_translation_feedback_recovers_a_slipped_rigid_object_without_windup():
    c = servo(); b = pose([0, 0, 1]); goal = pose([.8, 0, 1])
    q = np.zeros(1)
    nominal_p, _ = c.poses(b, q, np.zeros_like(c.offset))
    initial_slip = np.array([0., 0., -.012])
    for _ in range(250):
        hand_p, _ = c.poses(b, q, c.offset)
        measured = goal.copy()
        measured[:3] += initial_slip + (hand_p[0] - nominal_p[0]).mean(axis=0)
        x, req = c.propose(b, q, q, goal, measured, weight=1, dt=.02)
        _, v, qd = c.command(b, x, .02)
        assert np.linalg.norm(x[:3]) <= .03 + 1e-10
        assert np.max(abs(x[6:])) <= .1 + 1e-10
        assert np.linalg.norm(v[:3]) <= .02 + 1e-10
        assert np.max(abs(qd)) <= .05 + 1e-10
        c.accept(x, req)
    assert np.linalg.norm(goal[:3] - measured[:3]) < .001
    retained = c.offset.copy()
    unreachable = goal.copy(); unreachable[2] -= 10.
    for _ in range(10):
        c.propose(b, q, q, goal, unreachable, weight=1, dt=.02)
    np.testing.assert_array_equal(c.offset, retained)


def test_translation_activation_preserves_an_already_acceptable_attitude():
    c = servo(); b = pose([0, 0, 1]); goal = pose([.8, 0, 1])
    measured = pose([.8, 0, .99], R.from_euler('z', .065))
    q = np.zeros(1)
    p0, r0 = c.poses(b, q, c.offset)
    for _ in range(100):
        hand_p, _ = c.poses(b, q, c.offset)
        measured[:3] = goal[:3] + np.array([0., 0., -.01]) + (hand_p[0] - p0[0]).mean(axis=0)
        x, req = c.propose(b, q, q, goal, measured, weight=1, dt=.02,
                           regulate_orientation=False)
        c.accept(x, req)
    p, r = c.poses(b, q, x)
    assert (p[0] - p0[0]).mean(axis=0)[2] > .005
    # Match the existing regularized solver's 3 mrad orientation accuracy,
    # rather than requiring an exact equality from its soft objective.
    assert R.from_matrix((r @ r0.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).magnitude().max() < .003
    np.testing.assert_array_equal(req, np.zeros(3))


def test_translation_preserves_achieved_hand_rotation_not_unachieved_request():
    c = servo(); b = pose([0,0,1]); q = np.zeros(1)
    c.offset[4] = .01
    c.request_rotation[1] = .10
    goal = pose([.8,0,1]); measured = pose([.8,0,.99],R.from_euler('y',.065))
    p0,r0 = c.poses(b,q,c.offset)
    x,_ = c.propose(b,q,q,goal,measured,weight=1,dt=.02,regulate_orientation=False)
    p,r = c.poses(b,q,x)
    assert (p[0]-p0[0]).mean(axis=0)[2] > 0
    assert R.from_matrix((r @ r0.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().max() < .001


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
    c.safety_rejections = {}
    c.terminate_on_rejection = False
    c.collision_object=SimpleNamespace(ground_plane_z_m=None)
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
        grip_contact_present=True,environment_origin=torch.zeros(3),time_s=1.,phase=4,final_goal_orientation_tolerance_rad=.2,final_goal_position_tolerance_m=.05)
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


def test_rejected_rotation_does_not_latch_after_attitude_recovers():
    from types import SimpleNamespace
    import torch
    from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
    c = m.CheckedGraspSlipController.__new__(m.CheckedGraspSlipController)
    c.safety_rejections = {}
    c.terminate_on_rejection = False
    c.collision_object=SimpleNamespace(ground_plane_z_m=None)
    c.dt=.02;c.reserve={'span_m':0.};c.direction=np.zeros(1);c.speeds=np.array([.2])
    c.normals=np.array([[1.,0.,0.]]);c.controllers=[m.MaterialSlipClosure(span_m=0.)]
    c.previous_offsets=np.zeros((1,1));c.trace=[];c.pose_controllers=[servo()]
    nominal=torch.tensor([[0.,0.,1.,0.,0.,0.,1.]])
    goal=torch.tensor([[.8,0.,1.,0.,0.,0.,1.]])
    q=torch.zeros(1,1,1)
    target=Order9TensorObjectTaskTarget(nominal,torch.zeros(1,6),q,q,goal,nominal,goal,torch.ones(1),torch.tensor([3]))
    supervisor=SimpleNamespace(contact_velocity_observer=SimpleNamespace(previous=None),expected_group='g',
        grip_contact_present=True,environment_origin=torch.zeros(3),time_s=1.,phase=3,
        final_goal_orientation_tolerance_rad=.2,final_goal_position_tolerance_m=.05)
    state=SimpleNamespace(object_pose_world=torch.tensor([pose([.8,0,.91],R.from_euler('z',.25))]),object_twist_world=torch.zeros(1,6))
    c._safe=lambda joint,body,obj: np.linalg.norm(body-nominal[0].numpy())<1e-12 and np.linalg.norm(joint)<1e-12
    c.apply(target,state,[supervisor])
    np.testing.assert_array_equal(c.pose_controllers[0].offset,0)
    state.object_pose_world=torch.tensor([pose([.8,0,.91],R.from_euler('z',.065))])
    c._safe=lambda *args:True
    c.apply(target,state,[supervisor])
    assert np.linalg.norm(c.pose_controllers[0].offset)>0
    np.testing.assert_array_equal(c.pose_controllers[0].request_rotation,0)


def test_ground_forecast_starts_bounded_lift_before_descending_endpoint():
    from types import SimpleNamespace
    import torch
    from amsrr.training.request_imitation import PHASES
    c = m.CheckedGraspSlipController.__new__(m.CheckedGraspSlipController)
    c.safety_rejections = {}
    c.terminate_on_rejection = False
    c.dt=.02;c.collision_object=SimpleNamespace(ground_plane_z_m=0.)
    c.phase_goal_joints={PHASES[5]:np.zeros(1)}
    c.bundle=SimpleNamespace(morphology=None);c.ids=['joint'];c._ground_forecast_cache=None
    # An independent rigid geometry: its lowest point is 0.26 m below the body.
    c.solver=SimpleNamespace(check_configuration=lambda **kw:{
        'minimum_ground_clearance_m':kw['centroidal_pose_world'][2]-.26})
    control=servo();control.offset[2]=.007
    supervisor=SimpleNamespace(phase=5,environment_origin=torch.tensor([3.,2.,1.]))
    goal=pose([3.,2.,1.25])
    required=.011
    for _ in range(25):
        previous=control.offset.copy()
        candidate=c._anticipate_ground_clearance(control,previous,supervisor,goal,np.zeros(1))
        assert 0 <= candidate[2]-previous[2] <= control.rates[2]*c.dt+1e-12
        assert candidate[2] <= control.scales[2]
        np.testing.assert_array_equal(candidate[3:],previous[3:])
        control.accept(candidate,control.request_rotation)
    assert control.offset[2] == pytest.approx(required)
    assert control.diagnostics['ground_forecast_clearance_m'] == pytest.approx(.001)
    goal[2]=1.20
    with pytest.raises(RuntimeError,match='exceeds bounded'):
        c._anticipate_ground_clearance(control,control.offset,supervisor,goal,np.zeros(1))


@pytest.mark.parametrize('composed_safe',[True,False])
def test_closure_checks_composed_body_command_when_pose_feedback_is_active(composed_safe):
    from types import SimpleNamespace
    import torch
    from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
    c=m.CheckedGraspSlipController.__new__(m.CheckedGraspSlipController)
    c.safety_rejections = {}
    c.terminate_on_rejection = False
    c.collision_object=SimpleNamespace(ground_plane_z_m=None)
    c.dt=.02;c.reserve={'span_m':.006};c.direction=np.array([.1]);c.speeds=np.array([1.])
    c.normals=np.array([[1.,0.,0.]]);c.controllers=[m.MaterialSlipClosure(span_m=.006)]
    c.controllers[0].closure_m=.001;c.previous_offsets=np.array([[.1/6]])
    c.trace=[];control=servo();c.pose_controllers=[control]
    control.set_binding('g');control.offset[2]=.01
    nominal=torch.tensor([[0.,0.,1.,0.,0.,0.,1.]])
    q=torch.zeros(1,1,1)
    target=Order9TensorObjectTaskTarget(nominal,torch.zeros(1,6),q,q,nominal,nominal,nominal,torch.tensor([.5]),torch.tensor([3]))
    supervisor=SimpleNamespace(contact_velocity_observer=SimpleNamespace(previous=None),expected_group='g',
        grip_contact_present=True,environment_origin=torch.zeros(3),time_s=1.,phase=3)
    state=SimpleNamespace(object_pose_world=nominal,object_twist_world=torch.zeros(1,6))
    checked=[]
    def safe(joint,body,obj):
        checked.append((joint.copy(),body.copy()))
        return composed_safe and body[2]>1.005
    c._safe=safe
    if composed_safe:
        result=c.apply(target,state,[supervisor])
        assert result.desired_robot_root_pose_world[0,2] == pytest.approx(1.01)
        assert result.nominal_joint_positions_rad[0,0,0] == pytest.approx(.1/6)
    else:
        with pytest.raises(RuntimeError,match='no safe continuous grasp posture'):
            c.apply(target,state,[supervisor])
    assert checked and all(body[2]>1.005 for joint,body in checked)


def test_release_reference_waits_for_actual_feedback_clear_without_resetting_elapsed():
    from types import SimpleNamespace
    import torch
    c=m.CheckedGraspSlipController.__new__(m.CheckedGraspSlipController)
    c.safety_rejections = {}
    c.terminate_on_rejection = False
    c._release_clock=[None,None];c.release_decay_progress=[None,None];c.release_payload_scale=[1.,1.]
    c.pose_controllers=[servo(),servo()];c.pose_controllers[0].offset[2]=.01
    c.previous_offsets=np.array([[.01],[0.]])
    c.bundle=SimpleNamespace(phase_trajectories={'release':SimpleNamespace(knots=[SimpleNamespace(t_rel_s=3.)])})
    c.trace=[]
    from amsrr.training.request_imitation import PHASES
    phase=torch.tensor([PHASES.index('release')]*2)
    for time_s in (0.,1.,3.,6.):
        elapsed=torch.tensor([time_s,time_s])
        conditioned=c.reference_elapsed(phase,elapsed)
        torch.testing.assert_close(elapsed,torch.full((2,),time_s))
        assert conditioned[0] == 0.
        if time_s<=3.: assert conditioned[1] == 0.
        else: assert conditioned[1] == time_s-3.
        assert c.release_payload_scale[1] == pytest.approx(1.-min(1.,time_s/1.5)**2*(3.-2*min(1.,time_s/1.5)))
        assert c.release_decay_progress[0] == min(1.,time_s/3.)
    c.pose_controllers[0].offset.fill(0.)
    assert c.reference_elapsed(phase,torch.tensor([6.5,6.5]))[0] == 0.
    c.previous_offsets[0].fill(0.)
    assert c.reference_elapsed(phase,torch.tensor([7.,7.]))[0] == 0.
    assert c.reference_elapsed(phase,torch.tensor([7.5,7.5]))[0] == .5
    assert c.trace[-1]['reference_hold_s'] == 7.
    c.reference_elapsed(torch.tensor([0,0]),torch.zeros(2))
    assert c._release_clock == [None,None] and c.release_decay_progress == [None,None]
    assert c.reference_elapsed(phase,torch.ones(2)).tolist() == [0.,0.]
