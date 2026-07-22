from __future__ import annotations

import pytest
import torch

from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskRuntime
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
