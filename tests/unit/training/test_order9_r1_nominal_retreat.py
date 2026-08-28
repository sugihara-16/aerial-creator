from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    materialize_order9_c3_complete_task_phases,
)
from amsrr.training.order9_r1_nominal_retreat import (
    ORDER9_R1_NOMINAL_RETREAT_VERSION,
    materialize_order9_r1_complete_task_phases,
)

REPOSITORY = Path(__file__).resolve().parents[3]
TASK = REPOSITORY / (
    "artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/"
    "rollout_buckets_current_lineage_v5/buckets/"
    "train-000000-5fecfad4f44f/task_spec.json"
)
JOINT = "module_0:yaw_dock_mech_joint1"


def _trajectory(label: str, offset: float) -> ContactWrenchTrajectory:
    knots = []
    for index, time_s in enumerate((0.0, 0.1, 0.2)):
        position = offset + float(index)
        knots.append(
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=[position, 2.0, 3.0],
                    com_vel_world=[10.0, 0.0, 0.0],
                    body_orientation_world=[0.0, 0.0, 0.0, 1.0],
                ),
                posture_target=PostureTarget(
                    joint_pos_target={JOINT: position},
                    joint_vel_target={JOINT: 10.0},
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="payload",
                        pose_target_world=[
                            position,
                            5.0,
                            6.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        twist_target_world=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    )
                ],
                guard_conditions=[{"phase": label}],
            )
        )
    return ContactWrenchTrajectory(
        horizon_s=0.2,
        dt_s=0.1,
        knots=knots,
        derived_mode_label=label,
    )


def test_r1_retreat_uses_task_offset_instead_of_reversing_full_approach() -> None:
    task = TaskSpec.from_json(TASK.read_text(encoding="utf-8"))
    inputs = {
        "approach": _trajectory("approach", 0.0),
        "contact_acquisition": _trajectory("contact_acquisition", 10.0),
    }
    durations = {phase.value: 1.0 for phase in ORDER9_OBJECT_TASK_PHASES}
    historical = materialize_order9_c3_complete_task_phases(
        phase_trajectories=inputs,
        task_spec=task,
        lift_clearance_m=0.1,
        retreat_offset_m=0.1,
        phase_duration_s=durations,
    )
    repaired = materialize_order9_r1_complete_task_phases(
        phase_trajectories=inputs,
        task_spec=task,
        lift_clearance_m=0.1,
        retreat_offset_m=0.1,
        phase_duration_s=durations,
    )

    historical_retreat = historical["retreat"]
    historical_dx = (
        historical_retreat.knots[0].centroidal_target.com_pos_world[0]
        - historical_retreat.knots[-1].centroidal_target.com_pos_world[0]
    )
    assert historical_dx != pytest.approx(0.1)

    release_end = repaired["release"].knots[-1]
    retreat = repaired["retreat"]
    settle = repaired["settle"]
    assert retreat.derived_mode_label == (
        f"{ORDER9_R1_NOMINAL_RETREAT_VERSION}:retreat"
    )
    assert retreat.knots[0].centroidal_target.com_pos_world == pytest.approx(
        release_end.centroidal_target.com_pos_world
    )
    assert (
        retreat.knots[0].centroidal_target.com_pos_world[0]
        - retreat.knots[-1].centroidal_target.com_pos_world[0]
    ) == pytest.approx(0.1)
    assert any(
        abs(knot.centroidal_target.com_vel_world[0]) > 1.0e-12
        for knot in retreat.knots[1:-1]
    )
    assert settle.knots[0].centroidal_target.com_pos_world == pytest.approx(
        retreat.knots[-1].centroidal_target.com_pos_world
    )
    assert settle.knots[-1].centroidal_target.com_pos_world == pytest.approx(
        settle.knots[0].centroidal_target.com_pos_world
    )
