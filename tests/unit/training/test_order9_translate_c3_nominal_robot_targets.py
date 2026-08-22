from __future__ import annotations

import pytest

from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from scripts.order9_translate_c3_nominal_robot_targets import (
    _hold_posture_from_reference,
    _open_at_fixed_body_then_lift,
    _parse_phase_hold_postures,
    _parse_phase_open_fractions,
    _parse_phase_open_then_lifts,
    _parse_phase_z_offsets,
    _parse_phase_prelifts,
    _parse_phase_time_scales,
    _prepend_vertical_prelift,
    _time_stretch_trajectory,
    _translate_robot_targets,
)


def _trajectory() -> ContactWrenchTrajectory:
    knots = []
    for time_s in (0.0, 2.0):
        knots.append(
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(1.0, 2.0, 3.0),
                    com_vel_world=(0.0, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:q": 0.4},
                    joint_vel_target={"module_0:q": -0.2},
                    free_anchor_pose_targets={
                        3: (4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0)
                    },
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="payload",
                        pose_target_world=(
                            7.0,
                            8.0,
                            9.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),
                    )
                ],
            )
        )
    return ContactWrenchTrajectory(horizon_s=2.0, dt_s=2.0, knots=knots)


def test_translate_robot_targets_preserves_joint_and_object_targets() -> None:
    trajectory = _trajectory()

    _translate_robot_targets(
        trajectory,
        start_offset=(0.0, 0.0, 0.04),
        end_offset=(0.0, 0.0, 0.01),
    )

    assert [
        knot.centroidal_target.com_pos_world[2] for knot in trajectory.knots
    ] == pytest.approx([3.04, 3.01])
    assert [
        knot.centroidal_target.com_vel_world[2] for knot in trajectory.knots
    ] == pytest.approx([-0.015, -0.015])
    assert [
        knot.posture_target.free_anchor_pose_targets[3][2]
        for knot in trajectory.knots
    ] == pytest.approx([6.04, 6.01])
    assert all(
        knot.posture_target.joint_pos_target == {"module_0:q": 0.4}
        for knot in trajectory.knots
    )
    assert all(
        knot.object_targets[0].pose_target_world[2] == pytest.approx(9.0)
        for knot in trajectory.knots
    )


def test_smooth_translate_has_zero_boundary_velocity() -> None:
    trajectory = _trajectory()
    middle = InteractionKnot.from_dict(
        {**trajectory.knots[0].to_dict(), "t_rel_s": 1.0}
    )
    trajectory.knots.insert(1, middle)

    _translate_robot_targets(
        trajectory,
        start_offset=(0.0, 0.0, 0.0),
        end_offset=(0.0, 0.0, 0.3),
        smooth=True,
    )

    assert [
        knot.centroidal_target.com_pos_world[2]
        for knot in trajectory.knots
    ] == pytest.approx([3.0, 3.15, 3.3])
    assert [
        knot.centroidal_target.com_vel_world[2]
        for knot in trajectory.knots
    ] == pytest.approx([0.0, 0.225, 0.0])
    assert [
        knot.posture_target.free_anchor_pose_targets[3][2]
        for knot in trajectory.knots
    ] == pytest.approx([6.0, 6.15, 6.3])


def test_parse_phase_z_offsets_accepts_constant_and_linear_values() -> None:
    assert _parse_phase_z_offsets(
        ["approach=40", "contact_acquisition=40:10"]
    ) == {
        "approach": pytest.approx((0.04, 0.04)),
        "contact_acquisition": pytest.approx((0.04, 0.01)),
    }

    with pytest.raises(ValueError, match="invalid or repeated"):
        _parse_phase_z_offsets(["approach=40", "approach=10"])


def test_time_stretch_preserves_path_rate_and_scales_velocities() -> None:
    trajectory = _trajectory()
    trajectory.knots[0].object_targets[0].twist_target_world = [
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    trajectory.knots[1].object_targets[0].twist_target_world = [
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    stretched = _time_stretch_trajectory(trajectory, scale=2.0)

    assert stretched.horizon_s == pytest.approx(4.0)
    assert stretched.dt_s == pytest.approx(2.0)
    assert [knot.t_rel_s for knot in stretched.knots] == pytest.approx(
        [0.0, 2.0, 4.0]
    )
    assert stretched.knots[1].posture_target.joint_pos_target == pytest.approx(
        {"module_0:q": 0.4}
    )
    assert stretched.knots[1].posture_target.joint_vel_target == pytest.approx(
        {"module_0:q": -0.1}
    )
    assert stretched.knots[1].object_targets[0].twist_target_world == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert stretched.knots[-1].object_targets[0].pose_target_world == pytest.approx(
        trajectory.knots[-1].object_targets[0].pose_target_world
    )


def test_parse_phase_time_scales_accepts_shortening_and_lengthening() -> None:
    assert _parse_phase_time_scales(["approach=2.0"]) == {"approach": 2.0}
    assert _parse_phase_time_scales(["release=0.1"]) == {"release": 0.1}
    with pytest.raises(ValueError, match="positive"):
        _parse_phase_time_scales(["approach=0.0"])


def test_uniform_phase_retime_can_shorten_path() -> None:
    trajectory = _trajectory()

    shortened = _time_stretch_trajectory(trajectory, scale=0.5)

    assert shortened.horizon_s == pytest.approx(1.0)
    assert shortened.knots[-1].t_rel_s == pytest.approx(1.0)
    assert shortened.knots[-1].posture_target.joint_pos_target == pytest.approx(
        trajectory.knots[-1].posture_target.joint_pos_target
    )
    assert shortened.knots[-1].posture_target.joint_vel_target == pytest.approx(
        {"module_0:q": -0.4}
    )


def test_vertical_prelift_holds_posture_then_replays_source_path() -> None:
    trajectory = _trajectory()
    trajectory.dt_s = 1.0
    trajectory.knots.insert(
        1,
        InteractionKnot.from_dict(
            {
                **trajectory.knots[0].to_dict(),
                "t_rel_s": 1.0,
            }
        ),
    )

    lifted = _prepend_vertical_prelift(
        trajectory,
        lift_m=1.0,
        duration_s=2.0,
    )

    assert lifted.horizon_s == pytest.approx(4.0)
    assert [knot.t_rel_s for knot in lifted.knots] == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )
    assert [
        knot.centroidal_target.com_pos_world[2]
        for knot in lifted.knots[:3]
    ] == pytest.approx([3.0, 3.5, 4.0])
    assert lifted.knots[1].centroidal_target.com_vel_world[2] == pytest.approx(
        0.75
    )
    assert lifted.knots[2].centroidal_target.com_vel_world[2] == pytest.approx(
        0.0
    )
    assert lifted.knots[2].posture_target.joint_pos_target == pytest.approx(
        trajectory.knots[0].posture_target.joint_pos_target
    )
    assert lifted.knots[-1].posture_target.joint_pos_target == pytest.approx(
        trajectory.knots[-1].posture_target.joint_pos_target
    )
    assert lifted.knots[-1].centroidal_target.com_pos_world[2] == pytest.approx(
        trajectory.knots[-1].centroidal_target.com_pos_world[2] + 1.0
    )
    assert lifted.knots[-1].object_targets[0].pose_target_world == pytest.approx(
        trajectory.knots[-1].object_targets[0].pose_target_world
    )


def test_parse_phase_prelifts_requires_positive_distance_and_duration() -> None:
    assert _parse_phase_prelifts(["release=150:30"]) == {
        "release": pytest.approx((0.15, 30.0))
    }
    with pytest.raises(ValueError, match="must be positive"):
        _parse_phase_prelifts(["release=0:30"])


def test_open_then_lift_finishes_joint_path_before_body_motion() -> None:
    trajectory = _trajectory()
    trajectory.dt_s = 1.0
    middle = InteractionKnot.from_dict(
        {**trajectory.knots[0].to_dict(), "t_rel_s": 1.0}
    )
    middle.posture_target.joint_pos_target["module_0:q"] = 0.3
    trajectory.knots.insert(1, middle)
    trajectory.knots[-1].posture_target.joint_pos_target[
        "module_0:q"
    ] = 0.2
    trajectory.knots[-1].centroidal_target.com_pos_world = (1.0, 2.0, 3.5)

    result = _open_at_fixed_body_then_lift(
        trajectory,
        extra_lift_m=0.5,
        lift_duration_s=2.0,
    )

    assert result.horizon_s == pytest.approx(4.0)
    assert result.knots[2].posture_target.joint_pos_target == pytest.approx(
        {"module_0:q": 0.2}
    )
    assert result.knots[2].centroidal_target.com_pos_world == pytest.approx(
        (1.0, 2.0, 3.0)
    )
    assert result.knots[-1].posture_target.joint_pos_target == pytest.approx(
        {"module_0:q": 0.2}
    )
    assert result.knots[-1].centroidal_target.com_pos_world == pytest.approx(
        (1.0, 2.0, 4.0)
    )
    assert result.knots[-1].object_targets[0].pose_target_world == pytest.approx(
        trajectory.knots[-1].object_targets[0].pose_target_world
    )


def test_parse_phase_open_then_lifts_allows_zero_extra_clearance() -> None:
    assert _parse_phase_open_then_lifts(["release=0:30"]) == {
        "release": pytest.approx((0.0, 30.0))
    }
    with pytest.raises(ValueError, match="invalid"):
        _parse_phase_open_then_lifts(["release=10:0"])


def test_open_then_lift_can_stop_at_partial_open_posture() -> None:
    trajectory = _trajectory()
    trajectory.dt_s = 1.0
    middle = InteractionKnot.from_dict(
        {**trajectory.knots[0].to_dict(), "t_rel_s": 1.0}
    )
    middle.posture_target.joint_pos_target["module_0:q"] = 0.3
    trajectory.knots.insert(1, middle)
    trajectory.knots[-1].posture_target.joint_pos_target["module_0:q"] = 0.2
    trajectory.knots[-1].centroidal_target.com_pos_world = (1.0, 2.0, 4.0)

    result = _open_at_fixed_body_then_lift(
        trajectory,
        extra_lift_m=0.0,
        lift_duration_s=2.0,
        open_fraction=0.5,
    )

    assert result.horizon_s == pytest.approx(3.0)
    assert result.knots[1].posture_target.joint_pos_target == pytest.approx(
        {"module_0:q": 0.3}
    )
    assert result.knots[-1].posture_target.joint_pos_target == pytest.approx(
        {"module_0:q": 0.3}
    )
    assert result.knots[-1].centroidal_target.com_pos_world == pytest.approx(
        (1.0, 2.0, 4.0)
    )


def test_hold_posture_from_reference_preserves_target_body_path() -> None:
    reference = _trajectory()
    reference.knots[-1].posture_target.joint_pos_target["module_0:q"] = 0.25
    reference.knots[-1].posture_target.free_anchor_pose_targets[3] = (
        4.0,
        5.0,
        6.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    target = _trajectory()
    for knot in target.knots:
        knot.centroidal_target.com_pos_world = (2.0, 2.0, 3.0)

    _hold_posture_from_reference(target, reference=reference)

    assert target.knots[-1].centroidal_target.com_pos_world == pytest.approx(
        (2.0, 2.0, 3.0)
    )
    assert target.knots[-1].posture_target.joint_pos_target == pytest.approx(
        {"module_0:q": 0.25}
    )
    assert target.knots[-1].posture_target.free_anchor_pose_targets[3] == pytest.approx(
        (5.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0)
    )


def test_parse_partial_open_and_held_posture_contracts() -> None:
    assert _parse_phase_open_fractions(["release=0.25"]) == {
        "release": 0.25
    }
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        _parse_phase_open_fractions(["release=1.1"])
    assert _parse_phase_hold_postures(
        ["retreat=release", "settle=release"]
    ) == {"retreat": "release", "settle": "release"}
    with pytest.raises(ValueError, match="invalid or repeated"):
        _parse_phase_hold_postures(["release=retreat"])
