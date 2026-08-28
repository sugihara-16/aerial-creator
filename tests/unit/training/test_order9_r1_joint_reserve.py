from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.training.order9_r1_joint_reserve import (
    ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M,
    _project_q,
    _trajectory_with_projected_q,
)


def test_r1_anchor_position_tolerance_is_three_dimensional_30_mm() -> None:
    assert ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M == pytest.approx(0.030)


def _trajectory() -> ContactWrenchTrajectory:
    return ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.5,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=[],
                posture_target=PostureTarget(
                    joint_pos_target={"joint_a": -1.0},
                    joint_vel_target={"joint_a": 0.0},
                ),
            ),
            InteractionKnot(
                t_rel_s=0.5,
                contact_assignments=[],
                posture_target=PostureTarget(
                    joint_pos_target={"joint_a": 0.0},
                    joint_vel_target={"joint_a": 0.0},
                ),
            ),
            InteractionKnot(
                t_rel_s=1.0,
                contact_assignments=[],
                posture_target=PostureTarget(
                    joint_pos_target={"joint_a": 1.0},
                    joint_vel_target={"joint_a": 0.0},
                ),
            ),
        ],
    )


def test_projection_clamps_only_outside_values() -> None:
    projected, changed = _project_q({"joint_a": -1.0}, {"joint_a": (-0.8, 0.8)})
    unchanged, unchanged_flag = _project_q({"joint_a": 0.25}, {"joint_a": (-0.8, 0.8)})

    assert projected == {"joint_a": -0.8}
    assert changed is True
    assert unchanged == {"joint_a": 0.25}
    assert unchanged_flag is False


def test_trajectory_projection_recomputes_joint_velocity() -> None:
    trajectory, changed, q_samples = _trajectory_with_projected_q(
        _trajectory(),
        ordered_ids=("joint_a",),
        reserve_limits={"joint_a": (-0.8, 0.8)},
    )

    assert changed == (0, 2)
    assert [sample["joint_a"] for sample in q_samples] == [-0.8, 0.0, 0.8]
    assert [
        knot.posture_target.joint_vel_target["joint_a"] for knot in trajectory.knots
    ] == pytest.approx([1.6, 1.6, 1.6])


def test_projection_rejects_joint_identity_mismatch() -> None:
    with pytest.raises(SchemaValidationError, match="identity mismatch"):
        _project_q({"joint_a": 0.0}, {"joint_b": (-1.0, 1.0)})
