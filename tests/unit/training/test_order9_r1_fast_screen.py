from __future__ import annotations

from dataclasses import replace
import math

import pytest

from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolutionEvidence,
)
from amsrr.training.order9_r1_fast_screen import (
    ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
    R1_FAST_SCREEN_BODY_TILT_CODE,
    R1_FAST_SCREEN_COLLISION_CODE,
    R1_FAST_SCREEN_GRASP_POSE_CODE,
    R1_FAST_SCREEN_JOINT_LIMIT_CODE,
    Order9R1FastScreenWindow,
    require_order9_r1_fast_screen_pass,
    screen_order9_r1_resolved_grasp_path,
)
from amsrr.utils.hashing import stable_hash

_JOINTS = ("module_0:yaw", "module_0:pitch")
_LIMITS = {joint_id: (-1.0, 1.0) for joint_id in _JOINTS}


def test_r1_fast_screen_admits_exact_resolved_path_without_heavy_layers() -> None:
    result = _screen(_windows())

    assert result.accepted
    assert result.eligible_for_full_control_test
    assert result.execution_model == ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL
    assert result.checked_phases == ("approach", "contact_acquisition")
    assert result.grasp_pose_reached
    assert result.final_maintained_contact_count == 2
    assert result.minimum_normalized_joint_limit_reserve > 0.01
    assert result.maximum_collision_violating_pair_count == 0
    assert result.screen_wall_time_s >= 0.0
    assert result.isaac_invoked is False
    assert result.controller_layers_invoked is False
    assert result.ik_resolve_invoked is False
    assert result.trajectory_optimization_invoked is False
    require_order9_r1_fast_screen_pass(result)


def test_r1_fast_screen_rejects_near_limit_posture_before_full_test() -> None:
    windows = list(_windows())
    contact = _trajectory(0.2, 0.995, maintained=True)
    windows[-1] = _window(
        phase="contact_acquisition",
        reached=True,
        trajectory=contact,
    )

    result = _screen(tuple(windows))

    assert not result.accepted
    assert R1_FAST_SCREEN_JOINT_LIMIT_CODE in result.violation_codes
    with pytest.raises(SchemaValidationError, match="not eligible"):
        require_order9_r1_fast_screen_pass(result)


def test_r1_fast_screen_rejects_missing_collision_or_grasp_evidence() -> None:
    windows = list(_windows())
    last = windows[-1]
    windows[-1] = replace(
        last,
        phase_target_reached=False,
        resolution_evidence=replace(
            last.resolution_evidence,
            collision_gate_status="not_configured",
            collision_gate_version=None,
            minimum_collision_clearance_m=None,
        ),
    )

    result = _screen(tuple(windows))

    assert not result.accepted
    assert R1_FAST_SCREEN_COLLISION_CODE in result.violation_codes
    assert R1_FAST_SCREEN_GRASP_POSE_CODE in result.violation_codes
    assert not result.eligible_for_full_control_test


def test_r1_fast_screen_rejects_extreme_body_tilt() -> None:
    windows = list(_windows())
    tilted = _trajectory(
        0.1,
        0.2,
        maintained=True,
        body_orientation=(2**-0.5, 0.0, 0.0, 2**-0.5),
    )
    windows[-1] = _window(
        phase="contact_acquisition",
        reached=True,
        trajectory=tilted,
    )

    result = _screen(tuple(windows))

    assert not result.accepted
    assert R1_FAST_SCREEN_BODY_TILT_CODE in result.violation_codes
    assert result.maximum_body_tilt_rad == pytest.approx(math.pi / 2.0)


def _screen(windows):
    return screen_order9_r1_resolved_grasp_path(
        candidate_id="r1-l1-train-bucket-000-sample-000",
        candidate_task_hash="a" * 64,
        morphology_hash="b" * 64,
        physical_model_hash="c" * 64,
        ordered_joint_limits_rad=_LIMITS,
        windows=windows,
        proxy_collision_validation_status="enforced_during_generation",
    )


def _windows() -> tuple[Order9R1FastScreenWindow, ...]:
    approach = _trajectory(0.0, 0.1, maintained=False)
    contact = _trajectory(0.1, 0.2, maintained=True)
    return (
        _window(phase="approach", reached=True, trajectory=approach),
        _window(
            phase="contact_acquisition",
            reached=True,
            trajectory=contact,
        ),
    )


def _window(
    *,
    phase: str,
    reached: bool,
    trajectory: ContactWrenchTrajectory,
) -> Order9R1FastScreenWindow:
    raw = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    evidence = Order9PostureResolutionEvidence(
        resolver_version="unit-resolver-v1",
        solver_version="unit-ik-v1",
        raw_trajectory_hash=stable_hash(raw.to_dict()),
        resolved_trajectory_hash=stable_hash(trajectory.to_dict()),
        output_rate_hz=20.0,
        raw_knot_count=len(raw.knots),
        resolved_knot_count=len(trajectory.knots),
        solved_knot_count=len(trajectory.knots),
        solver_iterations=(1,) * len(trajectory.knots),
        solver_cache_hits=(False,) * len(trajectory.knots),
        maximum_anchor_position_error_m=0.0,
        maximum_anchor_attitude_error_rad=0.0,
        maximum_joint_rate_rad_s=0.1,
        maximum_joint_rate_segment_index=0,
        minimum_required_segment_duration_s=0.1,
        joint_rate_limit_rad_s=1.0,
        minimum_joint_rate_margin_rad_s=0.9,
        solve_wall_time_s=1.0,
        collision_gate_status="accepted",
        collision_gate_version="unit-collision-v1",
        minimum_collision_clearance_m=0.005,
        maximum_collision_violating_pair_count=0,
        external_validation_status="not_run",
        external_validator_version=None,
        external_violation_codes=(),
        external_margins={},
        initial_joint_state_hash="d" * 64,
        nominal_joint_seed_trajectory_hash=None,
    )
    return Order9R1FastScreenWindow(
        phase=phase,
        phase_target_reached=reached,
        raw_trajectory=raw,
        resolved_trajectory=trajectory,
        resolution_evidence=evidence,
        configuration_plan_collision_check_count=8,
    )


def _trajectory(
    start: float,
    end: float,
    *,
    maintained: bool,
    body_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> ContactWrenchTrajectory:
    assignments = [
        ContactAssignment(
            slot_id=index,
            anchor_id=index,
            candidate_id=index,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain" if maintained else "approach",
        )
        for index in range(2)
    ]
    return ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.5,
        knots=[
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=list(assignments),
                centroidal_target=CentroidalTarget(
                    com_pos_world=(0.0, 0.0, 0.5),
                    body_orientation_world=body_orientation,
                ),
                posture_target=PostureTarget(
                    joint_pos_target={joint_id: value for joint_id in _JOINTS},
                    joint_vel_target={joint_id: 0.1 for joint_id in _JOINTS},
                ),
            )
            for time_s, value in ((0.0, start), (1.0, end))
        ],
    )
