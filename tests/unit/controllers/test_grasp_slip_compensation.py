import numpy as np
import pytest
from amsrr.controllers.grasp_slip_compensation import MaterialSlipClosure


def test_reserve_bounds_actual_nonlinear_fk_after_ik_quantization():
    from amsrr.controllers.grasp_slip_compensation import _bounded_fk_reserve_scale
    # A one-joint curved FK response: scaling the oversized endpoint linearly
    # would still overshoot the allowed normal displacement.
    def positions(scale):
        return np.array([[-.008 * (2. * scale - scale * scale), 0., 0.]])
    normals = np.array([[1., 0., 0.]])
    naive_scale = .006 / .008
    assert -positions(naive_scale)[0, 0] > .006
    checked = _bounded_fk_reserve_scale(positions, normals, .006)
    assert checked == pytest.approx(.5, abs=1 / 4096)
    assert -positions(checked)[0, 0] <= .006
    assert _bounded_fk_reserve_scale(positions, normals, .008) == 1.
    assert _bounded_fk_reserve_scale(lambda s: -positions(s), normals, .006) == 0.


def test_slip_closure_is_tangential_bounded_and_does_not_wind_up():
    controller = MaterialSlipClosure(span_m=.006, rate_mps=.004)
    p = np.array([[0., 0., 0.], [1., 0., 0.]])
    normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
    def update(points, maintain=True, contact=True):
        return controller.update(points, normals, binding='grasp', maintain=maintain, contact_present=contact, dt=.1)
    assert update(p, False) == 0.
    assert update(p + [0.03, 0., 0.]) == 0.  # normal compression is not slip
    assert update(p + [0., 0., .001]) == 0.
    assert update(p + [0., 0., .005]) == pytest.approx(.0004)
    for _ in range(100):
        actual = update(p + [0., 0., .005])
    assert actual == pytest.approx(.003)
    assert update(p) == pytest.approx(.003)  # retain the established grip
    assert update(p + [0., 0., 1.], contact=False) == pytest.approx(.003)
    for _ in range(100):
        actual = update(p + [0., 0., 1.])
    assert actual == pytest.approx(.006)
    controller.reset()
    assert update(p) == 0.


def test_binding_change_resets_closure_without_future_targets():
    controller = MaterialSlipClosure(span_m=.006)
    n = np.array([[0., 0., 1.]])
    for p in [[[0., 0., 0.]], [[.01, 0., 0.]], [[.02, 0., 0.]]]:
        controller.update(p, n, binding='first', maintain=True, contact_present=True, dt=1.)
    assert controller.closure_m > 0
    assert controller.update([[.02, 0., 0.]], n, binding='second', maintain=True, contact_present=True, dt=1.) == 0


def test_slip_metric_invariant_to_object_frame_rotation():
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('xyz', [.2, -.5, 1.1]).as_matrix()
    p = np.array([[.2, .1, 0.], [-.2, .1, 0.]])
    n = np.array([[1., 0., 0.], [-1., 0., 0.]])
    values = []
    for r in (np.eye(3), rotation):
        c = MaterialSlipClosure(span_m=.006)
        c.update(p @ r.T, n @ r.T, binding='x', maintain=True, contact_present=True, dt=1.)
        values.append(c.update((p + [0., .005, 0.]) @ r.T, n @ r.T, binding='x', maintain=True, contact_present=True, dt=1.))
    assert values == pytest.approx([.003, .003])


def test_modified_commands_checked_with_continuous_release_and_reset():
    from types import SimpleNamespace
    import torch
    from amsrr.controllers.grasp_slip_compensation import CheckedGraspSlipController
    from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
    controller = CheckedGraspSlipController.__new__(CheckedGraspSlipController)
    controller.safety_rejections = {}
    controller.terminate_on_rejection = False
    controller.collision_object = SimpleNamespace(ground_plane_z_m=None)
    controller.dt = .1
    controller.reserve = {'span_m': .006}
    controller.direction = np.array([.1, -.1])
    controller.speeds = np.array([1., 1.])
    controller.normals = np.array([[1., 0., 0.]])
    controller.controllers = [MaterialSlipClosure(span_m=.006)]
    controller.previous_offsets = np.zeros((1, 2))
    controller.trace = []
    from amsrr.controllers.grasp_slip_compensation import ObservedGraspOrientationServo
    controller.pose_controllers = [ObservedGraspOrientationServo(
        anchor_fk=lambda q: (np.zeros((len(q), 1, 3)), np.broadcast_to(np.eye(3), (len(q), 1, 3, 3))),
        joint_lower=[-1., -1.], joint_upper=[1., 1.], joint_speeds=controller.speeds)]
    checked = []
    controller._safe = lambda q, body, obj: checked.append(q.copy()) is None
    pose = torch.tensor([[0., 0., 1., 0., 0., 0., 1.]])
    supervisor = SimpleNamespace(contact_velocity_observer=SimpleNamespace(previous=torch.zeros(1, 1, 3)),
        expected_group='grasp', grip_contact_present=True, environment_origin=torch.zeros(3), time_s=0., phase=2, final_goal_orientation_tolerance_rad=.2)
    state = SimpleNamespace(object_pose_world=pose, object_twist_world=torch.zeros(1,6), module_pose_world=pose[:, None])
    def target(schedule, progress=0.):
        return Order9TensorObjectTaskTarget(pose, torch.zeros(1, 6), torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2), pose, pose, pose, torch.tensor([progress]), torch.tensor([schedule]))
    assert controller.apply(target(2), state, [supervisor]).nominal_joint_positions_rad.abs().max() == 0.
    supervisor.contact_velocity_observer.previous[0, 0, 2] = .005
    previous = np.zeros(2)
    for _ in range(20):
        result = controller.apply(target(3), state, [supervisor])
        q = result.nominal_joint_positions_rad.numpy().reshape(-1)
        np.testing.assert_allclose(result.nominal_joint_velocities_radps.numpy().reshape(-1), (q - previous) / .1, atol=1e-6)
        previous = q.copy()
    assert q[0] == pytest.approx(.05)
    assert checked
    for progress in np.linspace(0., 1., 11):
        result = controller.apply(target(4, progress), state, [supervisor])
        assert result.nominal_joint_velocities_radps.abs().max() <= 1.
    assert result.nominal_joint_positions_rad.abs().max() == 0.
    result = controller.apply(target(0), state, [supervisor])
    assert result.nominal_joint_positions_rad.abs().max() == 0.
    assert controller.controllers[0].closure_m == 0.
    # Reject an unsafe increment while the previous command remains safe.
    controller.apply(target(2), state, [supervisor])
    supervisor.contact_velocity_observer.previous[0, 0, 2] += .005
    controller._safe = lambda q, body, obj: np.linalg.norm(q) < 1e-12
    result = controller.apply(target(3), state, [supervisor])
    assert result.nominal_joint_positions_rad.abs().max() == 0.
    assert controller.controllers[0].closure_m == 0.
    # No unsafe command may be emitted if even retaining the old increment fails.
    controller._safe = lambda q, body, obj: False
    with pytest.raises(RuntimeError, match='no safe continuous'):
        controller.apply(target(3), state, [supervisor])
    controller._safe = lambda q, body, obj: True
    controller.speeds[:] = .01
    result = controller.apply(target(3), state, [supervisor])
    assert result.nominal_joint_velocities_radps.abs().max() <= .010001


def test_partial_reserve_preserves_absolute_surface_tangent_limit():
    from amsrr.controllers.grasp_slip_compensation import _bounded_fk_reserve_scale
    normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
    surfaces = np.array([[0., 0., 0.], [1., 0., 0.]])
    origin = surfaces + np.array([[0., .0015, 0.], [0., 0., 0.]])
    def positions(scale):
        return origin + scale * np.array([[-.005, .010, 0.], [.005, 0., 0.]])
    scale = _bounded_fk_reserve_scale(positions, normals, .005, surface_points=surfaces)
    assert 0.049 < scale <= .05
    displacement = positions(scale) - surfaces
    tangent = displacement - (displacement * normals).sum(1, keepdims=True) * normals
    assert np.linalg.norm(tangent, axis=1).max() <= .002
    assert _bounded_fk_reserve_scale(positions, normals, .005) > scale


def _unloaded_fixture(count):
    from types import SimpleNamespace
    from amsrr.controllers.grasp_slip_compensation import CheckedGraspSlipController, ObservedGraspOrientationServo
    import torch
    from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
    c = CheckedGraspSlipController.__new__(CheckedGraspSlipController)
    c.safety_rejections = {}
    c.terminate_on_rejection = False
    c.collision_object = SimpleNamespace(ground_plane_z_m=None)
    c.dt, c.reserve = .02, {'span_m': .006}
    c.direction, c.speeds = np.array([.1, -.1]), np.ones(2)
    c.normals = np.array([[1., 0., 0.]])
    c.controllers = [MaterialSlipClosure(span_m=.006) for _ in range(count)]
    c.previous_offsets = np.zeros((count, 2))
    c.trace = []
    c.pose_controllers = [ObservedGraspOrientationServo(
        anchor_fk=lambda q: (np.zeros((len(q), 1, 3)), np.broadcast_to(np.eye(3), (len(q), 1, 3, 3))),
        joint_lower=[-1., -1.], joint_upper=[1., 1.], joint_speeds=c.speeds) for _ in range(count)]
    c._safe = lambda *args: True
    torch.manual_seed(103)
    pose = torch.randn(count, 7); pose[:, :3] += 1.
    body = torch.randn(count, 7)
    q = torch.rand(count, 1, 2) - .5
    qdot = torch.rand_like(q) * .1
    twist = torch.randn(count, 6) * .1
    target = Order9TensorObjectTaskTarget(body, twist, q, qdot, pose, pose, body,
        torch.zeros(count), torch.arange(count) % 2)
    state = SimpleNamespace(object_pose_world=pose.clone(), object_twist_world=torch.zeros(count, 6))
    supervisors = [SimpleNamespace(contact_velocity_observer=SimpleNamespace(previous=None),
        expected_group='grasp', grip_contact_present=False, environment_origin=torch.zeros(3),
        time_s=0., phase=1, final_goal_orientation_tolerance_rad=.2,
        final_goal_position_tolerance_m=.052) for _ in range(count)]
    return c, target, state, supervisors


@pytest.mark.parametrize('count', [1, 16])
def test_unloaded_batched_command_matches_scalar_path_and_state(count):
    from copy import deepcopy
    from dataclasses import fields
    import torch
    c, target, state, supervisors = _unloaded_fixture(count)
    reference = deepcopy(c)
    reference._unloaded_batch = lambda *args: False
    for step in range(3):
        for supervisor in supervisors:
            supervisor.time_s = float(step)
            supervisor.phase = step % 2
        if step == 2:
            # Held endpoint can set regulation state even though weight is zero.
            target.phase_progress.fill_(1.)
            target.nominal_joint_velocities_radps.zero_()
            target.desired_robot_root_twist_world.zero_()
            state.object_pose_world[:, 0] += .1
        a, b = c.apply(target, state, supervisors), reference.apply(target, state, supervisors)
        for f in fields(a):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if isinstance(x, torch.Tensor):
                assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)), f.name
        assert c.trace == reference.trace
        np.testing.assert_array_equal(c.previous_offsets, reference.previous_offsets)
        for actual, expected in zip(c.pose_controllers, reference.pose_controllers):
            for name in ('offset', 'request_rotation'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            for name in ('binding', 'regulating', 'regulation_phase', 'diagnostics'):
                assert getattr(actual, name) == getattr(expected, name)


@pytest.mark.parametrize('fault', ['joint_limit', 'speed_limit', 'zero_quaternion', 'nan'])
def test_unloaded_batch_keeps_scalar_rejections(fault):
    from copy import deepcopy
    c, target, state, supervisors = _unloaded_fixture(2)
    if fault == 'joint_limit':
        target.nominal_joint_positions_rad.fill_(2.)
    elif fault == 'speed_limit':
        target.nominal_joint_velocities_radps.fill_(2.)
    elif fault == 'zero_quaternion':
        target.desired_object_pose_world[:, 3:] = 0.
    else:
        target.nominal_joint_positions_rad.fill_(float('nan'))
    reference = deepcopy(c)
    reference._unloaded_batch = lambda *args: False
    with pytest.raises((ValueError, RuntimeError)) as expected:
        reference.apply(target, state, supervisors)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        c.apply(target, state, supervisors)


@pytest.mark.parametrize('progress', [np.float32(1. - 1e-6), np.nextafter(np.float32(1. - 1e-6), np.float32(1.))])
def test_unloaded_endpoint_threshold_matches_python_float_comparison(progress):
    from copy import deepcopy
    c, target, state, supervisors = _unloaded_fixture(2)
    reference = deepcopy(c)
    reference._unloaded_batch = lambda *args: False
    target.phase_progress.fill_(float(progress))
    target.nominal_joint_velocities_radps.zero_()
    target.desired_robot_root_twist_world.zero_()
    state.object_pose_world[:, 0] += .1
    c.apply(target, state, supervisors)
    reference.apply(target, state, supervisors)
    assert [s.regulating for s in c.pose_controllers] == [s.regulating for s in reference.pose_controllers]
    assert c.pose_controllers[0].regulating == (float(progress) >= 1. - 1e-6)


def test_unloaded_rejects_broadcastable_wrong_joint_width():
    from copy import deepcopy
    from dataclasses import replace
    c, target, state, supervisors = _unloaded_fixture(2)
    reference = deepcopy(c)
    reference._unloaded_batch = lambda *args: False
    target = replace(target, nominal_joint_positions_rad=target.nominal_joint_positions_rad[:, :, :1],
        nominal_joint_velocities_radps=target.nominal_joint_velocities_radps[:, :, :1])
    for controller in (reference, c):
        with pytest.raises((ValueError, RuntimeError)):
            controller.apply(target, state, supervisors)


def test_rejected_closure_is_latched_only_for_failed_environment_when_opted_in():
    from copy import deepcopy
    import torch
    c, target, state, supervisors = _unloaded_fixture(2)
    target.contact_schedule_index.fill_(3)
    c.controllers[0].closure_m = .002
    c._safe = lambda *args: False
    ordinary = deepcopy(c)
    with pytest.raises(RuntimeError, match='no safe continuous grasp closure'):
        ordinary.apply(target, state, supervisors)
    c.terminate_on_rejection = True
    result = c.apply(target, state, supervisors)
    assert c.safety_rejections == {0: 'no safe continuous grasp closure command'}
    assert torch.isfinite(result.nominal_joint_positions_rad).all()
    # The healthy row still runs; the rejected row no longer re-enters safety checks.
    c._safe = lambda *args: (_ for _ in ()).throw(AssertionError('rejected row retried'))
    c.apply(target, state, supervisors)
    assert len(c.safety_rejections) == 1


def test_inactive_environment_cannot_generate_late_safety_rejection():
    c, target, state, supervisors = _unloaded_fixture(2)
    target.contact_schedule_index.fill_(3)
    c.controllers[0].closure_m = .002
    c._safe = lambda *args: (_ for _ in ()).throw(AssertionError('inactive row checked'))
    c.active_environments = [False, True]
    c.terminate_on_rejection = True
    c.apply(target, state, supervisors)
    assert c.safety_rejections == {}


@pytest.mark.parametrize('nominal_q,reason', [
    (.9995, 'no admissible held grasp posture'),
    (1.002, 'no admissible grasp posture velocity interval'),
])
def test_infeasible_carried_posture_terminates_only_affected_environment(nominal_q, reason):
    from copy import deepcopy
    import torch
    c, target, state, supervisors = _unloaded_fixture(2)
    target.contact_schedule_index.fill_(3)
    target.nominal_joint_velocities_radps.zero_()
    target.nominal_joint_positions_rad.zero_()
    target.nominal_joint_positions_rad[0, 0, 0] = nominal_q
    for supervisor, servo in zip(supervisors, c.pose_controllers):
        supervisor.time_s = 1.
        supervisor.grip_contact_present = True
        servo.set_binding('grasp')
    # A previously safe +1 mrad correction cannot be held after the nominal
    # trajectory moves toward the upper joint stop. Exercise the real bounds.
    c.pose_controllers[0].offset[6] = .001
    ordinary = deepcopy(c)
    with pytest.raises(RuntimeError, match=reason):
        ordinary.apply(target, state, supervisors)
    before = c.pose_controllers[0].offset.copy()
    c.terminate_on_rejection = True
    result = c.apply(target, state, supervisors)
    assert c.safety_rejections == {0: reason}
    np.testing.assert_array_equal(c.pose_controllers[0].offset, before)
    assert torch.isfinite(result.nominal_joint_positions_rad).all()
    torch.testing.assert_close(result.nominal_joint_positions_rad[1], target.nominal_joint_positions_rad[1])
    rejected = [row for row in c.trace if row.get('error') == 'infeasible_grasp_posture']
    assert len(rejected) == 1 and rejected[0]['environment'] == 0 and rejected[0]['reason'] == reason
    c.apply(target, state, supervisors)
    assert c.safety_rejections == {0: reason}


def test_unexpected_posture_error_is_not_converted_to_episode_failure():
    c, target, state, supervisors = _unloaded_fixture(2)
    target.contact_schedule_index.fill_(3)
    c.terminate_on_rejection = True
    def broken(*args, **kwargs):
        raise RuntimeError('unexpected solver defect')
    c.pose_controllers[0].propose = broken
    with pytest.raises(RuntimeError, match='unexpected solver defect'):
        c.apply(target, state, supervisors)
    assert c.safety_rejections == {}
