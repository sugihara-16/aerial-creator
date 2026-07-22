from __future__ import annotations

import pytest
import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
)
from amsrr.training.order9_tensor_teacher_reference import (
    Order9TensorTeacherReference,
)


def _reference() -> Order9TensorTeacherReference:
    phase_count = len(ORDER9_OBJECT_TASK_PHASES)
    progress = torch.tensor([[0.0, 0.5]] * phase_count)
    body_pose = torch.zeros((phase_count, 2, 7))
    body_pose[..., 6] = 1.0
    body_pose[:, 1, 0] = 2.0
    body_twist = torch.zeros((phase_count, 2, 6))
    body_twist[:, 1, 0] = 0.2
    joint_position = torch.zeros((phase_count, 2, 1, 1))
    joint_position[:, 1, 0, 0] = 1.0
    joint_velocity = torch.zeros_like(joint_position)
    joint_velocity[:, 1, 0, 0] = 0.1
    object_pose = body_pose.clone()
    object_pose[:, 1, 1] = 1.0
    return Order9TensorTeacherReference(
        phase_progress=progress,
        phase_lengths=torch.full((phase_count,), 2, dtype=torch.long),
        desired_body_pose_world=body_pose,
        desired_body_twist=body_twist,
        nominal_joint_positions_rad=joint_position,
        nominal_joint_velocities_radps=joint_velocity,
        desired_object_pose_world=object_pose,
        phase_goal_body_pose_world=body_pose[:, 1],
        phase_goal_object_pose_world=object_pose[:, 1],
        initial_module_pose_world=body_pose[0, :1].clone(),
        initial_module_twist_world=body_twist[0, :1].clone(),
        initial_object_pose_world=object_pose[0, 0].clone(),
        initial_object_twist_world=body_twist[0, 0].clone(),
        initial_joint_positions_rad={"module_0__yaw_dock_mech_joint1": 0.1},
        initial_joint_velocities_radps={
            "module_0__yaw_dock_mech_joint1": 0.2
        },
        module_ids=(0,),
        joint_ids=("yaw_dock_mech_joint1",),
        provenance={"source": "unit"},
    )


def test_tensor_teacher_reference_interpolates_and_offsets() -> None:
    reference = _reference()

    sample = reference.sample(
        phase_index=torch.tensor([0, 1]),
        phase_progress=torch.tensor([0.25, 0.75]),
        position_offset_world=torch.tensor([[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]]),
    )

    assert sample.desired_body_pose_world[0, :3].tolist() == pytest.approx(
        [4.0, 4.0, 5.0]
    )
    assert sample.desired_body_pose_world[1, :3].tolist() == pytest.approx(
        [3.0, 2.0, 3.0]
    )
    assert sample.nominal_joint_positions_rad[:, 0, 0].tolist() == pytest.approx(
        [0.5, 1.0]
    )
    assert sample.phase_goal_body_pose_world[:, 0].tolist() == pytest.approx(
        [5.0, 3.0]
    )
    assert torch.allclose(
        sample.desired_body_pose_world[:, 3:7].norm(dim=-1), torch.ones(2)
    )


def test_tensor_teacher_reference_rejects_invalid_phase() -> None:
    reference = _reference()

    with pytest.raises(ValueError, match="phase index is invalid"):
        reference.sample(
            phase_index=torch.tensor([len(ORDER9_OBJECT_TASK_PHASES)]),
            phase_progress=torch.tensor([0.0]),
        )
