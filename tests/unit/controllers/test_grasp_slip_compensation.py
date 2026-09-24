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
