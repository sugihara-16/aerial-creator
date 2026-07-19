from __future__ import annotations

import pytest

from amsrr.simulation.dock_joint_drive_tuning import (
    DOCK_JOINT_DRIVE_TUNING_DEPLOYMENT_GATE,
    DOCK_JOINT_DRIVE_TUNING_SELECTION_SCOPE,
    DockJointDriveSample,
    DockJointDriveTuningConfig,
    coarse_gain_candidates,
    evaluate_gain_candidate,
    fine_gain_candidates,
    select_best_gain_candidate,
)


def test_bench_selection_requires_separate_contact_validation() -> None:
    assert DOCK_JOINT_DRIVE_TUNING_SELECTION_SCOPE == (
        "contact_free_bench_candidate_only"
    )
    assert DOCK_JOINT_DRIVE_TUNING_DEPLOYMENT_GATE == (
        "separate_representative_contact_task_validation_required"
    )


def test_coarse_and_fine_candidates_are_deterministic_and_unique() -> None:
    config = DockJointDriveTuningConfig(
        coarse_kp_values=(100.0, 200.0),
        coarse_kd_values=(2.0, 5.0),
        fine_multipliers=(0.5, 1.0, 1.5),
    )
    coarse = coarse_gain_candidates(config)
    assert coarse == ((100.0, 2.0), (100.0, 5.0), (200.0, 2.0), (200.0, 5.0))
    fine = fine_gain_candidates(
        config,
        center_kp=200.0,
        center_kd=5.0,
        excluded=coarse,
    )
    assert len(fine) == len(set(fine))
    assert (200.0, 5.0) not in fine
    assert (300.0, 7.5) in fine


def test_candidate_evaluation_prefers_fast_damped_trace() -> None:
    config = DockJointDriveTuningConfig(
        simulation_dt_s=0.02,
        step_hold_s=0.08,
        return_hold_s=0.08,
        disturbance_hold_s=0.08,
        recovery_hold_s=0.08,
    )
    joint_names = ("dock_joint",)
    fast = _trace(
        joint_names,
        step_positions=(0.008, 0.010, 0.010, 0.010),
        return_positions=(0.002, 0.0, 0.0, 0.0),
        disturbance_positions=(0.001, 0.002, 0.002, 0.002),
        recovery_positions=(0.001, 0.0, 0.0, 0.0),
    )
    slow = _trace(
        joint_names,
        step_positions=(0.002, 0.004, 0.006, 0.007),
        return_positions=(0.006, 0.005, 0.004, 0.003),
        disturbance_positions=(0.004, 0.006, 0.007, 0.008),
        recovery_positions=(0.006, 0.005, 0.004, 0.003),
    )
    fast_result = evaluate_gain_candidate(
        kp=200.0,
        kd=5.0,
        joint_names=joint_names,
        samples=fast,
        config=config,
    )
    slow_result = evaluate_gain_candidate(
        kp=100.0,
        kd=1.0,
        joint_names=joint_names,
        samples=slow,
        config=config,
    )
    assert fast_result["feasible"] is True
    assert float(fast_result["score"]) < float(slow_result["score"])
    assert select_best_gain_candidate([slow_result, fast_result]) is fast_result


def test_candidate_evaluation_rejects_speed_limit_violation() -> None:
    config = DockJointDriveTuningConfig(
        simulation_dt_s=0.02,
        step_hold_s=0.02,
        return_hold_s=0.02,
        disturbance_hold_s=0.02,
        recovery_hold_s=0.02,
        velocity_limit_rad_s=3.0,
    )
    sample = DockJointDriveSample(
        phase="step",
        phase_time_s=0.02,
        position_rad_by_joint={"dock_joint": 0.01},
        velocity_rad_s_by_joint={"dock_joint": 3.1},
        target_rad_by_joint={"dock_joint": 0.01},
        applied_torque_nm_by_joint={"dock_joint": 4.1},
    )
    phases = []
    for phase in ("step", "return", "disturbance", "recovery"):
        phases.append(
            DockJointDriveSample(
                phase=phase,
                phase_time_s=sample.phase_time_s,
                position_rad_by_joint=sample.position_rad_by_joint,
                velocity_rad_s_by_joint=sample.velocity_rad_s_by_joint,
                target_rad_by_joint=sample.target_rad_by_joint,
                applied_torque_nm_by_joint=sample.applied_torque_nm_by_joint,
            )
        )
    result = evaluate_gain_candidate(
        kp=800.0,
        kd=0.25,
        joint_names=("dock_joint",),
        samples=phases,
        config=config,
    )
    assert result["feasible"] is False
    with pytest.raises(ValueError, match="no feasible"):
        select_best_gain_candidate([result])


def _trace(
    joint_names: tuple[str, ...],
    *,
    step_positions: tuple[float, ...],
    return_positions: tuple[float, ...],
    disturbance_positions: tuple[float, ...],
    recovery_positions: tuple[float, ...],
) -> list[DockJointDriveSample]:
    result: list[DockJointDriveSample] = []
    for phase, positions in (
        ("step", step_positions),
        ("return", return_positions),
        ("disturbance", disturbance_positions),
        ("recovery", recovery_positions),
    ):
        for index, position in enumerate(positions, start=1):
            target = 0.01 if phase == "step" else 0.0
            result.append(
                DockJointDriveSample(
                    phase=phase,
                    phase_time_s=0.02 * index,
                    position_rad_by_joint={name: position for name in joint_names},
                    velocity_rad_s_by_joint={name: 0.0 for name in joint_names},
                    target_rad_by_joint={name: target for name in joint_names},
                    applied_torque_nm_by_joint={name: 1.0 for name in joint_names},
                )
            )
    return result
