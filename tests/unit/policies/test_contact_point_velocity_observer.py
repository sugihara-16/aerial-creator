import math

import pytest
import torch

from amsrr.simulation.request_event_execution import ContactPointVelocityObserver


def pose(x=0., angle=0.):
    return torch.tensor([[x, 0., 0., 0., 0., math.sin(angle / 2),
                          math.cos(angle / 2)]], dtype=torch.float64)


def test_common_rigid_motion_has_no_relative_slip():
    observer = ContactPointVelocityObserver()
    points = torch.tensor([[[1., 0., 0.], [0., 2., 0.]]], dtype=torch.float64)
    _, valid = observer.update(points, pose(), time_s=0., binding='pair')
    assert not valid
    # Both bodies translate by 3 m and rotate 90 degrees about the object origin.
    transformed = torch.tensor([[[3., 1., 0.], [1., 0., 0.]]], dtype=torch.float64)
    speed, valid = observer.update(transformed, pose(3., math.pi / 2), time_s=.02, binding='pair')
    assert valid
    torch.testing.assert_close(speed, torch.zeros_like(speed), atol=1e-12, rtol=0.)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_actual_motion_and_jumps_cannot_pass_speed_limit(axis):
    observer = ContactPointVelocityObserver()
    points = torch.zeros(1, 2, 3, dtype=torch.float64)
    observer.update(points, pose(), time_s=0., binding='pair')
    points[:, 0, axis] = .002
    speed, valid = observer.update(points, pose(), time_s=.02, binding='pair')
    assert valid and float(speed[0, 0]) == pytest.approx(.1)
    assert not bool((speed <= .05).all())
    points[:, 0, axis] += 1.
    speed, _ = observer.update(points, pose(), time_s=.04, binding='pair')
    assert float(speed[0, 0]) == pytest.approx(50.)


def test_reset_binding_change_and_time_checks():
    observer = ContactPointVelocityObserver()
    points = torch.zeros(1, 2, 3, dtype=torch.float64)
    observer.update(points, pose(), time_s=1., binding='pair')
    with pytest.raises(ValueError, match='monotonic'):
        observer.update(points, pose(), time_s=1., binding='pair')
    _, valid = observer.update(points + 100., pose(), time_s=2., binding='other')
    assert not valid
    observer.reset()
    _, valid = observer.update(points, pose(), time_s=0., binding='pair')
    assert not valid


def test_unequal_intervals_use_actual_elapsed_time_and_instances_are_independent():
    first, second = ContactPointVelocityObserver(), ContactPointVelocityObserver()
    points = torch.zeros(1, 2, 3, dtype=torch.float64)
    first.update(points, pose(), time_s=0., binding='pair')
    second.update(points + 5., pose(), time_s=0., binding='pair')
    moved = points.clone(); moved[:, :, 0] += .001
    speed, valid = first.update(moved, pose(), time_s=.1, binding='pair')
    assert valid
    torch.testing.assert_close(speed, torch.full_like(speed, .01))
    speed, _ = second.update(points + 5., pose(), time_s=.02, binding='pair')
    assert not speed.any()
