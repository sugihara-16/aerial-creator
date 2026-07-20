from __future__ import annotations

import math
from pathlib import Path

import pytest

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
    Order9ObjectTaskRuntime,
)
from amsrr.simulation.order9_object_task_state import (
    Order9IsaacStateSnapshot,
    load_order9_canonical_reset,
)


ROOT = Path(__file__).resolve().parents[3]
REPORT = (
    ROOT
    / "artifacts/p4_full/order8_natural_contact/"
    "order8_mu4p5_dt20ms_full_v406.json"
)
REPORT_SHA256 = "d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49"


def _runtime() -> Order9ObjectTaskRuntime:
    reset = load_order9_canonical_reset(REPORT, expected_sha256=REPORT_SHA256)
    return Order9ObjectTaskRuntime(reset)


def test_canonical_reset_is_bound_to_passing_order8_bytes() -> None:
    reset = load_order9_canonical_reset(REPORT, expected_sha256=REPORT_SHA256)
    assert reset.source_report_sha256 == REPORT_SHA256
    assert reset.transport_distance_m == pytest.approx(0.2)
    assert reset.lift_clearance_m == pytest.approx(0.1)
    assert reset.metadata["object_support_size_m"] == pytest.approx(
        [0.55, 0.36, 0.15]
    )
    assert reset.metadata["object_support_pose_world"][:3] == pytest.approx(
        [1.4266090350716598, 1.9909655707805166e-06, 0.075]
    )
    assert len(reset.joint_positions_rad) == 12
    assert set(reset.joint_positions_rad) == set(reset.open_joint_positions_rad)
    assert len(reset.reset_hash) == 64
    with pytest.raises(Exception, match="hash mismatch"):
        load_order9_canonical_reset(REPORT, expected_sha256="0" * 64)


def test_phase_schedule_covers_complete_object_task_and_is_continuous() -> None:
    runtime = _runtime()
    assert tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES) == (
        "approach",
        "contact_acquisition",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
        "settle",
    )
    for index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES):
        reset = runtime.reset_for_phase(index)
        start = runtime.target(index, 0.0, reset=reset)
        end = runtime.target(index, runtime.duration_s(index), reset=reset)
        assert start.phase == phase
        assert end.phase_progress == pytest.approx(1.0)
        assert all(math.isfinite(value) for value in end.desired_robot_root_pose_world)
        assert set(end.nominal_joint_positions_rad) == set(
            runtime.canonical.joint_positions_rad
        )


def test_randomized_grasp_phase_moves_robot_and_object_together() -> None:
    runtime = _runtime()
    lift_index = ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.LIFT)
    nominal = runtime.reset_for_phase(lift_index)
    shifted = runtime.reset_for_phase(
        lift_index,
        object_position_offset_world=(0.004, -0.003, 0.002),
        object_yaw_offset_rad=0.01,
    )
    for axis, expected in enumerate((0.004, -0.003, 0.002)):
        assert shifted.robot_root_pose_world[axis] - nominal.robot_root_pose_world[axis] == pytest.approx(expected)
        assert shifted.object_pose_world[axis] - nominal.object_pose_world[axis] == pytest.approx(expected)
    assert shifted.reset_labels_reused is False


def test_release_moves_qclose_to_open_without_actor_contact_truth() -> None:
    runtime = _runtime()
    index = ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.RELEASE)
    reset = runtime.reset_for_phase(index)
    target = runtime.target(index, runtime.duration_s(index), reset=reset)
    assert target.contact_schedule_state == "release"
    assert target.nominal_joint_positions_rad == pytest.approx(
        runtime.canonical.open_joint_positions_rad
    )
    assert runtime.config.raw_contact_actor_input is False


def test_portable_isaac_snapshot_validates_exact_joint_identity() -> None:
    snapshot = Order9IsaacStateSnapshot(
        simulation_time_s=1.0,
        robot_root_pose_world=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        robot_root_twist_world=[0.0] * 6,
        joint_names=["module_0__joint"],
        joint_positions_rad=[0.1],
        joint_velocities_radps=[0.0],
        object_id="object",
        object_pose_world=[1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
        object_twist_world=[0.0] * 6,
        phase_index=2,
        phase_elapsed_s=0.5,
        command_index=4,
    )
    snapshot.validate()
    assert len(snapshot.snapshot_hash) == 64
    snapshot.joint_names.append("module_0__joint")
    with pytest.raises(Exception, match="unique"):
        snapshot.validate()
