from __future__ import annotations

from pathlib import Path

import pytest
import torch

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.simulation.order9_tensor_object_task import (
    Order9TensorObjectTaskTarget,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    materialize_order9_c3_complete_task_phases,
)
from amsrr.training.order9_c3_nominal_runtime import (
    Order9C3NominalTensorReference,
    gate_order9_c3_nominal_boundary_completion,
    parse_order9_c3_diagnostic_phase_strata,
)


JOINT_ID = "yaw_dock_mech_joint1"
TASK = Path(
    "artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/"
    "rollout_buckets_current_lineage_v5/buckets/"
    "train-000000-5fecfad4f44f/task_spec.json"
)


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
                object_targets=[
                    ObjectTarget(
                        object_id="payload",
                        pose_target_world=[
                            value,
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
                guard_conditions=[{"phase": phase}],
            )
        )
    return ContactWrenchTrajectory(
        horizon_s=0.2,
        dt_s=0.1,
        knots=knots,
        derived_mode_label=phase,
    )


def _all_phase_trajectories() -> dict[str, ContactWrenchTrajectory]:
    return {
        phase.value: _trajectory(
            phase.value, position_offset=10.0 * float(index)
        )
        for index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES)
    }


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
        phase_trajectories=_all_phase_trajectories(),
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
        [0.5, 11.5, 20.5]
    )
    assert conditioned.desired_robot_root_pose_world[:, 0].tolist() == pytest.approx(
        [100.5, 211.5, 320.5]
    )
    assert conditioned.phase_goal_robot_root_pose_world[:, 0].tolist() == pytest.approx(
        [102.0, 212.0, 322.0]
    )
    assert conditioned.phase_progress.tolist() == pytest.approx([0.25, 0.75, 0.25])
    assert reference.provenance[
        "configuration_space_planner_runtime_enabled"
    ] is False


def test_c3_nominal_reference_clamps_after_phase_end() -> None:
    reference = Order9C3NominalTensorReference(
        phase_trajectories=_all_phase_trajectories(),
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


def test_c3_nominal_reference_exposes_phase_specific_goal_postures() -> None:
    reference = Order9C3NominalTensorReference(
        phase_trajectories=_all_phase_trajectories(),
        module_ids=(0,),
        joint_ids=(JOINT_ID,),
        device="cpu",
    )

    goals = reference.phase_goal_joint_positions(torch.tensor([0, 5, 7]))

    assert goals[:, 0, 0].tolist() == pytest.approx([2.0, 52.0, 72.0])
    with pytest.raises(ValueError, match="out of range"):
        reference.phase_goal_joint_positions(torch.tensor([8]))


def test_c3_nominal_phase_start_is_exact_t_zero_not_first_reset_stratum() -> None:
    reference = Order9C3NominalTensorReference(
        phase_trajectories=_all_phase_trajectories(),
        module_ids=(0,),
        joint_ids=(JOINT_ID,),
        device="cpu",
    )
    start = reference.phase_start_reference()
    resets = reference.phase_reset_references(
        object_pose_local=torch.tensor(
            [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        ),
        lift_clearance_m=0.4,
        transport_distance_m=0.7,
        retreat_offset_m=0.2,
        phase_duration_s={phase.value: 3.0 for phase in ORDER9_OBJECT_TASK_PHASES},
        progress_fractions=(1.0 / 6.0, 1.0 / 2.0, 2.0 / 3.0, 0.9),
    )

    start.validate()
    assert start.body_pose_local[0].item() == pytest.approx(0.0)
    assert start.joint_positions_rad[0, 0].item() == pytest.approx(0.0)
    assert start.object_pose_local[0].item() == pytest.approx(0.0)
    assert resets.phase_elapsed_s[0, 0].item() == pytest.approx(1.0 / 30.0)
    assert resets.body_pose_local[0, 0, 0].item() > start.body_pose_local[0].item()


def test_c3_nominal_boundary_success_waits_for_reference_endpoint() -> None:
    success = gate_order9_c3_nominal_boundary_completion(
        torch.tensor([True, True, True, False, True, True, True]),
        phase_index=torch.tensor([0, 0, 1, 0, 6, 6, 7]),
        phase_progress=torch.tensor([0.953, 1.0, 0.1, 1.0, 0.54, 1.0, 0.1]),
    )

    assert success.tolist() == [False, True, False, False, False, True, True]


def test_parse_c3_diagnostic_phase_strata_requires_one_pair_per_env() -> None:
    assert parse_order9_c3_diagnostic_phase_strata(
        "3:2,4:2,5:3,6:3",
        environment_count=4,
    ) == ((3, 2), (4, 2), (5, 3), (6, 3))
    with pytest.raises(ValueError, match="environment count"):
        parse_order9_c3_diagnostic_phase_strata(
            "3:2,4:2",
            environment_count=4,
        )


def test_c3_nominal_reference_builds_all_morphology_specific_phase_resets() -> None:
    reference = Order9C3NominalTensorReference(
        phase_trajectories=_all_phase_trajectories(),
        module_ids=(0,),
        joint_ids=(JOINT_ID,),
        device="cpu",
    )
    resets = reference.phase_reset_references(
        object_pose_local=torch.tensor(
            [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        ),
        lift_clearance_m=0.4,
        transport_distance_m=0.7,
        retreat_offset_m=0.2,
        phase_duration_s={phase.value: 3.0 for phase in ORDER9_OBJECT_TASK_PHASES},
        progress_fractions=(1.0 / 6.0, 1.0 / 2.0, 2.0 / 3.0, 0.9),
    )

    resets.validate()
    assert resets.stratum_count == 4
    assert resets.body_pose_local[:, 0, 0].tolist() == pytest.approx(
        [
            1.0 / 3.0,
            10.0 + 1.0 / 3.0,
            20.0 + 1.0 / 3.0,
            30.0 + 1.0 / 3.0,
            40.0 + 1.0 / 3.0,
            50.0 + 1.0 / 3.0,
            60.0 + 1.0 / 3.0,
            70.0 + 1.0 / 3.0,
        ]
    )
    assert resets.body_pose_local[:, 0, 2].tolist() == pytest.approx(
        [
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ]
    )
    assert resets.joint_positions_rad[:, 0, 0, 0].tolist() == pytest.approx(
        [
            1.0 / 3.0,
            10.0 + 1.0 / 3.0,
            20.0 + 1.0 / 3.0,
            30.0 + 1.0 / 3.0,
            40.0 + 1.0 / 3.0,
            50.0 + 1.0 / 3.0,
            60.0 + 1.0 / 3.0,
            70.0 + 1.0 / 3.0,
        ]
    )
    assert resets.object_pose_local[:, 0, 0].tolist() == pytest.approx(
        [
            1.0 / 3.0,
            10.0 + 1.0 / 3.0,
            20.0 + 1.0 / 3.0,
            30.0 + 1.0 / 3.0,
            40.0 + 1.0 / 3.0,
            50.0 + 1.0 / 3.0,
            60.0 + 1.0 / 3.0,
            70.0 + 1.0 / 3.0,
        ]
    )
    assert resets.object_pose_local[:, 0, 2].tolist() == pytest.approx(
        [
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
            6.0,
        ]
    )
    assert resets.phase_progress[3].tolist() == pytest.approx(
        [1.0 / 6.0, 1.0 / 2.0, 2.0 / 3.0, 0.9]
    )
    assert resets.body_pose_local[3, 1, 0].item() > resets.body_pose_local[
        3, 0, 0
    ].item()


def test_complete_task_materialization_preserves_grasp_and_reverses_safe_paths() -> None:
    task = TaskSpec.from_json(TASK.read_text(encoding="utf-8"))
    contact = _trajectory("contact_acquisition", position_offset=10.0)
    phases = materialize_order9_c3_complete_task_phases(
        phase_trajectories={
            "approach": _trajectory("approach", position_offset=0.0),
            "contact_acquisition": contact,
        },
        task_spec=task,
        lift_clearance_m=0.1,
        retreat_offset_m=0.1,
        phase_duration_s={phase.value: 1.0 for phase in ORDER9_OBJECT_TASK_PHASES},
    )

    assert tuple(phases) == tuple(
        phase.value for phase in ORDER9_OBJECT_TASK_PHASES
    )
    release = phases["release"]
    assert release.knots[0].posture_target.joint_pos_target[
        f"module_0:{JOINT_ID}"
    ] == pytest.approx(12.0)
    assert release.knots[-1].posture_target.joint_pos_target[
        f"module_0:{JOINT_ID}"
    ] == pytest.approx(10.0)
    assert (
        release.knots[-1].centroidal_target.com_pos_world[2]
        > release.knots[0].centroidal_target.com_pos_world[2]
    )

    transport = phases["transport"]
    object_start = transport.knots[0].object_targets[0].pose_target_world
    object_end = transport.knots[-1].object_targets[0].pose_target_world
    body_start = (
        *transport.knots[0].centroidal_target.com_pos_world,
        *transport.knots[0].centroidal_target.body_orientation_world,
    )
    body_end = (
        *transport.knots[-1].centroidal_target.com_pos_world,
        *transport.knots[-1].centroidal_target.body_orientation_world,
    )
    relative_start = compose_pose(inverse_pose(object_start), body_start)
    relative_end = compose_pose(inverse_pose(object_end), body_end)
    assert relative_end == pytest.approx(relative_start, abs=1.0e-8)
