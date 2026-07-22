from __future__ import annotations

import pytest
import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
    Order9ObjectTaskRuntime,
)
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_APPROACH,
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_INACTIVE,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
    Order9TensorObjectTaskRuntime,
)


_REPORT = (
    "artifacts/p4_full/order8_natural_contact/"
    "order8_mu4p5_dt20ms_full_v406.json"
)
_REPORT_HASH = "d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49"


def test_tensor_object_task_targets_match_scalar_phase_runtime() -> None:
    canonical = load_order9_canonical_reset(
        _REPORT, expected_sha256=_REPORT_HASH
    )
    scalar = Order9ObjectTaskRuntime(canonical)
    tensor = Order9TensorObjectTaskRuntime(scalar.config)
    resets = [scalar.reset_for_phase(index) for index in range(scalar.phase_count)]
    joint_ids = tuple(sorted(canonical.joint_positions_rad))
    phase_ends = [
        scalar.target(index, scalar.duration_s(index), reset=resets[index])
        for index in range(scalar.phase_count)
    ]
    elapsed = torch.tensor(
        [0.37 * scalar.duration_s(index) for index in range(scalar.phase_count)],
        dtype=torch.float64,
    )
    output = tensor.target(
        phase_index=torch.arange(scalar.phase_count),
        phase_elapsed_s=elapsed,
        reset_robot_root_pose_world=torch.tensor(
            [reset.robot_root_pose_world for reset in resets], dtype=torch.float64
        ),
        reset_object_pose_world=torch.tensor(
            [reset.object_pose_world for reset in resets], dtype=torch.float64
        ),
        reset_joint_positions_rad=torch.tensor(
            [
                [[reset.joint_positions_rad[joint_id] for joint_id in joint_ids]]
                for reset in resets
            ],
            dtype=torch.float64,
        ),
        phase_end_joint_positions_rad=torch.tensor(
            [
                [
                    [
                        phase_end.nominal_joint_positions_rad[joint_id]
                        for joint_id in joint_ids
                    ]
                ]
                for phase_end in phase_ends
            ],
            dtype=torch.float64,
        ),
        lift_clearance_m=torch.full(
            (scalar.phase_count,), canonical.lift_clearance_m, dtype=torch.float64
        ),
        transport_distance_m=torch.full(
            (scalar.phase_count,),
            canonical.transport_distance_m,
            dtype=torch.float64,
        ),
    )

    schedule_by_name = {
        "approach": ORDER9_CONTACT_SCHEDULE_APPROACH,
        "attach": ORDER9_CONTACT_SCHEDULE_ATTACH,
        "maintain": ORDER9_CONTACT_SCHEDULE_MAINTAIN,
        "release": ORDER9_CONTACT_SCHEDULE_RELEASE,
        "inactive": ORDER9_CONTACT_SCHEDULE_INACTIVE,
    }
    for index, reset in enumerate(resets):
        expected = scalar.target(index, elapsed[index].item(), reset=reset)
        assert output.desired_robot_root_pose_world[index].tolist() == pytest.approx(
            expected.desired_robot_root_pose_world
        )
        assert output.desired_robot_root_twist_world[index].tolist() == pytest.approx(
            expected.desired_robot_root_twist_world
        )
        assert output.desired_object_pose_world[index].tolist() == pytest.approx(
            expected.desired_object_pose_world
        )
        assert output.nominal_joint_positions_rad[index, 0].tolist() == pytest.approx(
            [expected.nominal_joint_positions_rad[joint_id] for joint_id in joint_ids]
        )
        assert output.nominal_joint_velocities_radps[index, 0].tolist() == pytest.approx(
            [expected.nominal_joint_velocities_radps[joint_id] for joint_id in joint_ids]
        )
        assert output.phase_progress[index].item() == pytest.approx(
            expected.phase_progress
        )
        assert output.contact_schedule_index[index].item() == schedule_by_name[
            expected.contact_schedule_state
        ]


def test_successor_phase_references_follow_planned_endpoints_without_error_drift() -> None:
    runtime = Order9TensorObjectTaskRuntime()
    phase = torch.tensor(
        [ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.LIFT)],
        dtype=torch.long,
    )
    elapsed = torch.tensor([30.0])
    body_start = torch.tensor([[0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 1.0]])
    object_start = torch.tensor([[0.0, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0]])
    joint_start = torch.zeros((1, 1, 1))
    joint_end = torch.zeros_like(joint_start)
    common = {
        "phase_elapsed_s": elapsed,
        "reset_joint_positions_rad": joint_start,
        "phase_end_joint_positions_rad": joint_end,
        "lift_clearance_m": torch.tensor([0.25]),
        "transport_distance_m": torch.tensor([0.20]),
    }

    lift = runtime.target(
        phase_index=phase,
        reset_robot_root_pose_world=body_start,
        reset_object_pose_world=object_start,
        **common,
    )
    body_reference, object_reference = lift.planned_successor_start(
        torch.tensor([0])
    )
    # A physical lift is allowed to pass 40 mm short of its goal.  That
    # observed error must not become part of the successor plan.
    observed_object = object_reference.clone()
    observed_object[:, 2] -= 0.04
    assert object_reference[:, 2].tolist() == pytest.approx([0.475])
    assert observed_object[:, 2].tolist() == pytest.approx([0.435])

    transport = runtime.target(
        phase_index=torch.tensor(
            [ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.TRANSPORT)]
        ),
        reset_robot_root_pose_world=body_reference,
        reset_object_pose_world=object_reference,
        **common,
    )
    body_reference, object_reference = transport.planned_successor_start(
        torch.tensor([0])
    )
    place = runtime.target(
        phase_index=torch.tensor(
            [ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.PLACE)]
        ),
        reset_robot_root_pose_world=body_reference,
        reset_object_pose_world=object_reference,
        **common,
    )
    _, final_object_reference = place.planned_successor_start(torch.tensor([0]))

    assert final_object_reference[0, :3].tolist() == pytest.approx(
        [0.20, 0.0, 0.225]
    )
