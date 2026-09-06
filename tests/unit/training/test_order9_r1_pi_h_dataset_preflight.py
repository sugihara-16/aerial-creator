from __future__ import annotations

import torch

from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_r1_pi_h_dataset_preflight import (
    build_order9_r1_pi_h_teacher_window,
    build_order9_r1_pi_h_teacher_window_from_trace,
    select_order9_r1_pi_h_decision_rows,
)


def test_pi_h_window_crosses_phase_boundaries_and_removes_joint_targets() -> None:
    phases = {
        phase.value: _trajectory(index)
        for index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES)
    }

    result = build_order9_r1_pi_h_teacher_window(
        phases,
        actor_phase_label="approach",
        phase_progress=0.5,
    )

    assert result.horizon_s == 2.0
    assert result.dt_s == 0.25
    assert len(result.knots) == 9
    assert [knot.t_rel_s for knot in result.knots] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ]
    # The window begins halfway through approach, crosses contact acquisition,
    # and reaches lift.  The source marker is the centroidal x coordinate.
    assert result.knots[0].centroidal_target.com_pos_world[0] == 0.0
    assert result.knots[3].centroidal_target.com_pos_world[0] == 1.0
    assert result.knots[-1].centroidal_target.com_pos_world[0] == 2.0
    assert all(
        knot.posture_target.joint_pos_target is None
        and knot.posture_target.joint_vel_target is None
        for knot in result.knots
    )


def test_decision_selection_keeps_fixed_rate_phase_change_and_tail() -> None:
    valid = torch.ones(8, dtype=torch.bool)
    times = torch.tensor([0.0, 0.2, 0.5, 0.7, 0.8, 1.0, 1.3, 1.4])
    phases = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    selected = select_order9_r1_pi_h_decision_rows(
        valid=valid,
        time_s=times,
        phase_index=phases,
        interval_s=0.5,
    )

    assert selected == (0, 2, 4, 6, 7)


def test_pi_h_window_uses_the_reference_actually_executed_in_isaac() -> None:
    phases = {
        phase.value: _trajectory(index)
        for index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES)
    }
    step_count = 5
    body_pose = torch.tensor(
        [
            [[10.0 + index, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0]]
            for index in range(step_count)
        ]
    )
    object_pose = torch.tensor(
        [
            [[40.0 + index, 50.0, 60.0, 0.0, 0.0, 0.0, 1.0]]
            for index in range(step_count)
        ]
    )
    tensors = {
        "time_s": torch.arange(step_count, dtype=torch.float32).reshape(-1, 1) * 0.5,
        "phase_index": torch.zeros((step_count, 1), dtype=torch.long),
        "phase_progress": torch.linspace(0.0, 1.0, step_count).reshape(-1, 1),
        "desired_body_pose_world": body_pose,
        "desired_body_twist_reference": torch.zeros((step_count, 1, 6)),
        "desired_object_pose_world": object_pose,
    }

    result = build_order9_r1_pi_h_teacher_window_from_trace(
        phases,
        tensors=tensors,
        raw_index=0,
    )

    assert result.knots[2].centroidal_target.com_pos_world == (11.0, 20.0, 30.0)
    assert result.knots[2].object_targets[0].pose_target_world == (
        41.0,
        50.0,
        60.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert result.knots[2].posture_target.joint_pos_target is None


def _trajectory(phase_index: int) -> ContactWrenchTrajectory:
    return ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.5,
        knots=[
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(float(phase_index), time_s, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:joint": float(phase_index)},
                    joint_vel_target={"module_0:joint": 0.0},
                    free_anchor_pose_targets={
                        0: (float(phase_index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                    },
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    )
                ],
                priority_weights={"phase": 1.0},
            )
            for time_s in (0.0, 0.5, 1.0)
        ],
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )
