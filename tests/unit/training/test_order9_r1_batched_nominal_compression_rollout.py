from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_APPROACH,
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    Order9TensorObjectTaskTarget,
)

REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"


def _module():
    spec = importlib.util.spec_from_file_location("order9_r1_batched_wrapper", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_environment_spacing_is_origin_packed() -> None:
    module = _module()
    arguments = ["--env-spacing", "3.0", "--other", "value"]

    module._configure_batched_environment_spacing(arguments)

    assert arguments == [
        "--env-spacing",
        str(module.BATCH_ENV_SPACING_M),
        "--other",
        "value",
    ]


def test_batched_environment_spacing_is_added_when_absent() -> None:
    module = _module()
    arguments = ["--other", "value"]

    module._configure_batched_environment_spacing(arguments)

    assert arguments[-2:] == ["--env-spacing", str(module.BATCH_ENV_SPACING_M)]


def test_batched_scene_rejects_more_than_four_candidates(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": f"candidate-{index}", "argv": ["--placeholder"]}
                    for index in range(5)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SchemaValidationError,
        match="exceeds the verified four-candidate limit",
    ):
        module._load_jobs(manifest, offset=0, limit=None)


def _target(progress: float, schedule: int) -> Order9TensorObjectTaskTarget:
    return Order9TensorObjectTaskTarget(
        desired_robot_root_pose_world=torch.zeros((1, 7)),
        desired_robot_root_twist_world=torch.zeros((1, 6)),
        nominal_joint_positions_rad=torch.full((1, 2, 2), 1.2),
        nominal_joint_velocities_radps=torch.ones((1, 2, 2)),
        desired_object_pose_world=torch.zeros((1, 7)),
        phase_goal_robot_root_pose_world=torch.zeros((1, 7)),
        phase_goal_object_pose_world=torch.zeros((1, 7)),
        phase_progress=torch.tensor([progress]),
        contact_schedule_index=torch.tensor([schedule]),
    )


def test_diagnostic_free_joint_unfold_is_bounded_to_late_approach() -> None:
    module = _module()
    profile = {
        "global_joint_id": "module_2:yaw_dock_mech_joint1",
        "approach_unfold_start_fraction": 0.6,
        "approach_unfold_end_fraction": 0.7,
        "final_target_rad": 0.0,
    }
    before = module._diagnostic_free_joint_unfold_target(
        _target(0.5, ORDER9_CONTACT_SCHEDULE_APPROACH),
        profile=profile,
        module_ids=(0, 2),
        joint_ids=("pitch_dock_mech_joint1", "yaw_dock_mech_joint1"),
    )
    after = module._diagnostic_free_joint_unfold_target(
        _target(0.8, ORDER9_CONTACT_SCHEDULE_APPROACH),
        profile=profile,
        module_ids=(0, 2),
        joint_ids=("pitch_dock_mech_joint1", "yaw_dock_mech_joint1"),
    )
    contact = module._diagnostic_free_joint_unfold_target(
        _target(0.0, ORDER9_CONTACT_SCHEDULE_ATTACH),
        profile=profile,
        module_ids=(0, 2),
        joint_ids=("pitch_dock_mech_joint1", "yaw_dock_mech_joint1"),
    )

    assert before.nominal_joint_positions_rad[0, 1, 1].item() == pytest.approx(1.2)
    assert after.nominal_joint_positions_rad[0, 1, 1].item() == pytest.approx(0.0)
    assert contact.nominal_joint_positions_rad[0, 1, 1].item() == pytest.approx(0.0)
    assert after.nominal_joint_positions_rad[0, 0, 0].item() == pytest.approx(1.2)


def test_diagnostic_late_approach_clearance_returns_to_endpoint() -> None:
    module = _module()
    profile = {
        "translation_world_m": [0.04, 0.0, 0.0],
        "rise_start_fraction": 0.5,
        "rise_end_fraction": 0.6,
        "fall_start_fraction": 0.9,
        "fall_end_fraction": 1.0,
    }
    raised = module._diagnostic_late_approach_clearance_target(
        _target(0.7, ORDER9_CONTACT_SCHEDULE_APPROACH), profile=profile
    )
    endpoint = module._diagnostic_late_approach_clearance_target(
        _target(1.0, ORDER9_CONTACT_SCHEDULE_APPROACH), profile=profile
    )
    contact = module._diagnostic_late_approach_clearance_target(
        _target(0.5, ORDER9_CONTACT_SCHEDULE_ATTACH), profile=profile
    )

    assert raised.desired_robot_root_pose_world[0, 0].item() == pytest.approx(0.04)
    assert raised.desired_robot_root_pose_world[0, 2].item() == pytest.approx(0.0)
    assert endpoint.desired_robot_root_pose_world[0, 0].item() == pytest.approx(0.0)
    assert contact.desired_robot_root_pose_world[0, 0].item() == pytest.approx(0.0)


def test_diagnostic_grasp_centering_ramps_then_holds() -> None:
    module = _module()
    profile = {"translation_world_m": [0.0, 0.02, 0.0]}
    attaching = module._diagnostic_grasp_centering_target(
        _target(0.5, ORDER9_CONTACT_SCHEDULE_ATTACH), profile=profile
    )
    maintained = module._diagnostic_grasp_centering_target(
        _target(0.0, ORDER9_CONTACT_SCHEDULE_MAINTAIN), profile=profile
    )

    assert attaching.desired_robot_root_pose_world[0, 1].item() == pytest.approx(0.01)
    assert attaching.phase_goal_robot_root_pose_world[0, 1].item() == pytest.approx(
        0.01
    )
    assert maintained.desired_robot_root_pose_world[0, 1].item() == pytest.approx(0.02)


def test_diagnostic_grasp_torque_bias_is_directional_and_bounded() -> None:
    module = _module()

    @dataclass(frozen=True)
    class Command:
        joint_torque_bias_nm: torch.Tensor

    @dataclass(frozen=True)
    class Result:
        policy_command: Command

    result = Result(policy_command=Command(torch.zeros((1, 1, 2))))
    biased = module._diagnostic_grasp_torque_bias(
        result,
        target=_target(0.0, ORDER9_CONTACT_SCHEDULE_MAINTAIN),
        compression_direction=torch.tensor([[[2.0, -1.0]]]),
        effort_limits_nm=torch.tensor([4.0, 2.0]),
        fraction=0.20,
    )

    assert torch.allclose(
        biased.policy_command.joint_torque_bias_nm,
        torch.tensor([[[0.8, -0.2]]]),
    )


def test_contact_compression_values_follow_active_slot_order() -> None:
    module = _module()
    mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]],
        dtype=torch.bool,
    )

    values = module._contact_compression_values_by_active_slot(mask, (30.0, 10.0))

    assert torch.equal(
        values,
        torch.tensor([[0.0, 30.0, 0.0, 10.0], [30.0, 0.0, 10.0, 0.0]]),
    )


def test_contact_compression_rejects_wrong_active_count() -> None:
    module = _module()
    mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    with pytest.raises(
        SchemaValidationError,
        match="count differs from active contacts",
    ):
        module._contact_compression_values_by_active_slot(mask, (10.0,))


def test_diagnostic_joint_clearance_pulse_returns_before_contact() -> None:
    module = _module()
    profile = {
        "global_joint_id": "module_2:yaw_dock_mech_joint1",
        "offset_rad": 0.25,
        "rise_start_fraction": 0.55,
        "rise_end_fraction": 0.65,
        "fall_start_fraction": 0.85,
        "fall_end_fraction": 0.98,
    }
    active = module._diagnostic_joint_clearance_pulse_target(
        _target(0.75, ORDER9_CONTACT_SCHEDULE_APPROACH),
        profile=profile,
        module_ids=(0, 2),
        joint_ids=("pitch_dock_mech_joint1", "yaw_dock_mech_joint1"),
    )
    endpoint = module._diagnostic_joint_clearance_pulse_target(
        _target(1.0, ORDER9_CONTACT_SCHEDULE_APPROACH),
        profile=profile,
        module_ids=(0, 2),
        joint_ids=("pitch_dock_mech_joint1", "yaw_dock_mech_joint1"),
    )

    assert active.nominal_joint_positions_rad[0, 1, 1].item() == pytest.approx(1.45)
    assert endpoint.nominal_joint_positions_rad[0, 1, 1].item() == pytest.approx(1.2)


def test_diagnostic_two_stage_approach_preserves_start_goal_and_clearance() -> None:
    module = _module()
    profile = {
        "start_position_offset_from_goal_m": [-0.8, -0.2, 0.1],
        "clearance_height_above_goal_m": 0.4,
        "rise_end_fraction": 0.2,
        "transit_end_fraction": 0.6,
        "descent_start_fraction": 0.8,
        "phase_duration_s": 50.0,
        "joint_motion_start_fraction": 0.1,
        "joint_motion_end_fraction": 0.4,
    }
    start_joints = torch.zeros((1, 2, 2))
    goal_joints = torch.ones((1, 2, 2))
    target = _target(0.0, ORDER9_CONTACT_SCHEDULE_APPROACH)
    target.phase_goal_robot_root_pose_world[:, :3] = torch.tensor([[1.0, 0.5, 0.6]])
    start = module._diagnostic_two_stage_approach_target(
        target,
        profile=profile,
        start_joint_positions_rad=start_joints,
        goal_joint_positions_rad=goal_joints,
    )
    overhead_start = module._diagnostic_two_stage_approach_target(
        _target(0.2, ORDER9_CONTACT_SCHEDULE_APPROACH),
        profile=profile,
        start_joint_positions_rad=start_joints,
        goal_joint_positions_rad=goal_joints,
    )
    overhead_start.phase_goal_robot_root_pose_world[:, :3] = torch.tensor(
        [[1.0, 0.5, 0.6]]
    )
    overhead_start = module._diagnostic_two_stage_approach_target(
        overhead_start,
        profile=profile,
        start_joint_positions_rad=start_joints,
        goal_joint_positions_rad=goal_joints,
    )
    endpoint_target = _target(1.0, ORDER9_CONTACT_SCHEDULE_APPROACH)
    endpoint_target.phase_goal_robot_root_pose_world[:, :3] = torch.tensor(
        [[1.0, 0.5, 0.6]]
    )
    endpoint = module._diagnostic_two_stage_approach_target(
        endpoint_target,
        profile=profile,
        start_joint_positions_rad=start_joints,
        goal_joint_positions_rad=goal_joints,
    )

    assert start.desired_robot_root_pose_world[0, :3].tolist() == pytest.approx(
        [0.2, 0.3, 0.7]
    )
    assert overhead_start.desired_robot_root_pose_world[
        0, :3
    ].tolist() == pytest.approx([0.2, 0.3, 1.0])
    assert endpoint.desired_robot_root_pose_world[0, :3].tolist() == pytest.approx(
        [1.0, 0.5, 0.6]
    )
    assert endpoint.desired_robot_root_twist_world[0, :3].tolist() == pytest.approx(
        [0.0, 0.0, 0.0], abs=1.0e-7
    )
