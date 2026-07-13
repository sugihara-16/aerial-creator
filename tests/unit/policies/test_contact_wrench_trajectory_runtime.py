from __future__ import annotations

import pytest

from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
    ContactWrenchTrajectoryRuntimeError,
)
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
)


def _trajectory() -> ContactWrenchTrajectory:
    return ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.5,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(0.0, 0.0, 1.0),
                    com_vel_world=(1.0, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                priority_weights={"tracking": 0.0},
                guard_conditions=[{"type": "left"}],
            ),
            InteractionKnot(
                t_rel_s=0.5,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(0.5, 0.0, 1.0),
                    com_vel_world=(1.0, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.3826834324, 0.9238795325),
                ),
                priority_weights={"tracking": 0.5},
                guard_conditions=[{"type": "middle"}],
            ),
            InteractionKnot(
                t_rel_s=1.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(1.0, 0.0, 1.0),
                    com_vel_world=(0.0, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.7071067812, 0.7071067812),
                ),
                priority_weights={"tracking": 1.0},
                guard_conditions=[{"type": "right"}],
            ),
        ],
    )


def test_executor_uses_explicit_rolling_plan_origin_and_interpolates() -> None:
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.1)
    executor.install(_trajectory(), plan_start_time_s=10.0)

    sample = executor.sample(time_s=10.25)

    assert sample.plan_elapsed_s == pytest.approx(0.25)
    assert sample.active_knot_index == 0
    assert sample.next_knot_index == 1
    assert sample.interpolation_ratio == pytest.approx(0.5)
    assert sample.active_knot.centroidal_target.com_pos_world == pytest.approx(
        (0.25, 0.0, 1.0)
    )
    assert sample.active_knot.priority_weights["tracking"] == pytest.approx(0.25)
    assert sample.active_knot.guard_conditions == [{"type": "left"}]


def test_executor_rejects_absolute_relative_time_confusion_and_expiry() -> None:
    executor = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.1)
    executor.install(_trajectory(), plan_start_time_s=10.0)

    with pytest.raises(ContactWrenchTrajectoryRuntimeError, match="precedes"):
        executor.sample(time_s=9.0)
    with pytest.raises(ContactWrenchTrajectoryRuntimeError, match="expired"):
        executor.sample(time_s=11.2)


def test_executor_rejects_unsorted_or_uncovered_trajectory() -> None:
    executor = ContactWrenchTrajectoryExecutor()
    invalid = _trajectory()
    invalid.knots[1].t_rel_s = 0.0
    with pytest.raises(ContactWrenchTrajectoryRuntimeError, match="strictly increasing"):
        executor.install(invalid, plan_start_time_s=0.0)
