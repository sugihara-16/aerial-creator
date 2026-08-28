from __future__ import annotations

from copy import deepcopy

import pytest

from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskPhase
from amsrr.training.order9_r1_release_clearance_repair import (
    ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION,
    apply_order9_r1_release_clearance_repair,
)


def _trajectory(phase: str) -> ContactWrenchTrajectory:
    knots = []
    for time_s in (0.0, 1.0, 2.0):
        knots.append(
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(1.0, 2.0, 3.0),
                    com_vel_world=(0.0, 0.0, 0.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"joint": 0.25},
                    joint_vel_target={"joint": 0.0},
                    free_anchor_pose_targets={4: (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)},
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(
                            4.0,
                            5.0,
                            6.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),
                        twist_target_world=[0.0] * 6,
                    )
                ],
            )
        )
    return ContactWrenchTrajectory(
        horizon_s=2.0,
        dt_s=1.0,
        knots=knots,
        derived_mode_label=phase,
    )


def _phases() -> dict[str, ContactWrenchTrajectory]:
    return {phase.value: _trajectory(phase.value) for phase in Order9ObjectTaskPhase}


def test_release_clearance_is_rigid_and_pregrasp_is_byte_equivalent() -> None:
    source = _phases()
    before = deepcopy(source)
    repaired = apply_order9_r1_release_clearance_repair(source, height_offset_m=0.03)

    for phase in (
        Order9ObjectTaskPhase.APPROACH,
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        Order9ObjectTaskPhase.LIFT,
        Order9ObjectTaskPhase.TRANSPORT,
    ):
        assert repaired[phase.value].to_dict() == before[phase.value].to_dict()
    place = repaired[Order9ObjectTaskPhase.PLACE.value]
    assert (
        place.knots[0].to_dict()
        == before[Order9ObjectTaskPhase.PLACE.value].knots[0].to_dict()
    )
    assert place.knots[-1].centroidal_target.com_pos_world[2] == pytest.approx(3.03)
    assert place.knots[-1].posture_target.free_anchor_pose_targets[4][
        2
    ] == pytest.approx(3.03)
    assert place.knots[-1].object_targets[0].pose_target_world[2] == pytest.approx(6.03)
    assert place.knots[1].centroidal_target.com_vel_world[2] > 0.0
    for phase in (
        Order9ObjectTaskPhase.RELEASE,
        Order9ObjectTaskPhase.RETREAT,
        Order9ObjectTaskPhase.SETTLE,
    ):
        trajectory = repaired[phase.value]
        assert trajectory.knots[0].centroidal_target.com_pos_world[2] == pytest.approx(
            3.03
        )
        assert trajectory.knots[0].posture_target.joint_pos_target == {"joint": 0.25}
        assert trajectory.derived_mode_label == (
            f"{ORDER9_R1_RELEASE_CLEARANCE_REPAIR_VERSION}:{phase.value}"
        )


@pytest.mark.parametrize("offset", [0.0, -0.001, 0.030001])
def test_release_clearance_rejects_invalid_offset(offset: float) -> None:
    with pytest.raises(ValueError):
        apply_order9_r1_release_clearance_repair(_phases(), height_offset_m=offset)
