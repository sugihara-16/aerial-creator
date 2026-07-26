from __future__ import annotations

import pytest
import torch

from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.simulation.order9_tensor_object_task import (
    Order9TensorObjectTaskTarget,
)
from amsrr.training.order9_c3_nominal_runtime import (
    Order9C3NominalTensorReference,
)


JOINT_ID = "yaw_dock_mech_joint1"


def _trajectory(phase: str, *, position_offset: float) -> ContactWrenchTrajectory:
    knots = []
    for index, time_s in enumerate((0.0, 0.1, 0.2)):
        value = position_offset + float(index)
        knots.append(
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=[value, 2.0, 3.0],
                    com_vel_world=[10.0, 0.0, 0.0],
                    body_orientation_world=[0.0, 0.0, 0.0, 1.0],
                ),
                posture_target=PostureTarget(
                    joint_pos_target={f"module_0:{JOINT_ID}": value},
                    joint_vel_target={f"module_0:{JOINT_ID}": 10.0},
                ),
                guard_conditions=[{"phase": phase}],
            )
        )
    return ContactWrenchTrajectory(
        horizon_s=0.2,
        dt_s=0.1,
        knots=knots,
        derived_mode_label=phase,
    )


def _target(batch: int) -> Order9TensorObjectTaskTarget:
    return Order9TensorObjectTaskTarget(
        desired_robot_root_pose_world=torch.zeros(batch, 7),
        desired_robot_root_twist_world=torch.zeros(batch, 6),
        nominal_joint_positions_rad=torch.zeros(batch, 1, 1),
        nominal_joint_velocities_radps=torch.zeros(batch, 1, 1),
        desired_object_pose_world=torch.zeros(batch, 7),
        phase_goal_robot_root_pose_world=torch.zeros(batch, 7),
        phase_goal_object_pose_world=torch.zeros(batch, 7),
        phase_progress=torch.zeros(batch),
        contact_schedule_index=torch.zeros(batch, dtype=torch.long),
    )


def test_c3_nominal_reference_interpolates_dense_teacher_without_planner() -> None:
    reference = Order9C3NominalTensorReference(
        phase_trajectories={
            "approach": _trajectory("approach", position_offset=0.0),
            "contact_acquisition": _trajectory(
                "contact_acquisition", position_offset=10.0
            ),
        },
        module_ids=(0,),
        joint_ids=(JOINT_ID,),
        device="cpu",
        provenance={"configuration_space_planner_runtime_enabled": False},
    )
    conditioned = reference.condition(
        _target(3),
        phase_index=torch.tensor([0, 1, 2]),
        phase_elapsed_s=torch.tensor([0.05, 0.15, 0.05]),
        scene_origins=torch.tensor(
            [[100.0, 0.0, 0.0], [200.0, 0.0, 0.0], [300.0, 0.0, 0.0]]
        ),
    )

    assert reference.nominal_rate_hz == pytest.approx(10.0)
    assert conditioned.nominal_joint_positions_rad[:, 0, 0].tolist() == pytest.approx(
        [0.5, 11.5, 0.0]
    )
    assert conditioned.desired_robot_root_pose_world[:, 0].tolist() == pytest.approx(
        [100.5, 211.5, 0.0]
    )
    assert conditioned.phase_goal_robot_root_pose_world[:, 0].tolist() == pytest.approx(
        [102.0, 212.0, 0.0]
    )
    assert conditioned.phase_progress.tolist() == pytest.approx([0.25, 0.75, 0.0])
    assert reference.provenance[
        "configuration_space_planner_runtime_enabled"
    ] is False


def test_c3_nominal_reference_clamps_after_phase_end() -> None:
    reference = Order9C3NominalTensorReference(
        phase_trajectories={
            "approach": _trajectory("approach", position_offset=0.0),
            "contact_acquisition": _trajectory(
                "contact_acquisition", position_offset=10.0
            ),
        },
        module_ids=(0,),
        joint_ids=(JOINT_ID,),
        device="cpu",
    )
    conditioned = reference.condition(
        _target(1),
        phase_index=torch.tensor([0]),
        phase_elapsed_s=torch.tensor([9.0]),
        scene_origins=torch.zeros(1, 3),
    )
    assert conditioned.nominal_joint_positions_rad[0, 0, 0].item() == pytest.approx(
        2.0
    )
    assert conditioned.phase_progress.item() == pytest.approx(1.0)
