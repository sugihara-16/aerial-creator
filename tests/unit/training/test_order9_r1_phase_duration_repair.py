from __future__ import annotations

import pytest

from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.training.order9_r1_phase_duration_repair import (
    _move_approach_tail_to_contact_acquisition,
    _extend_phase_endpoint_hold,
    _validated_endpoint_holds,
    _validated_handoff_fraction,
    _validated_multipliers,
)


def _values(multiplier: float = 2.0) -> dict[str, float]:
    return {
        "approach": 1.0,
        "contact_acquisition": 1.0,
        "lift": multiplier,
        "transport": multiplier,
        "place": multiplier,
        "release": multiplier,
        "retreat": multiplier,
        "settle": multiplier,
    }


def test_phase_duration_repair_accepts_bounded_slowdown() -> None:
    assert _validated_multipliers(_values())["transport"] == 2.0


def test_phase_duration_repair_accepts_contact_only_slowdown() -> None:
    values = _values(1.0)
    values["contact_acquisition"] = 2.0
    assert _validated_multipliers(values)["contact_acquisition"] == 2.0


def test_phase_duration_repair_rejects_speedup_and_noop() -> None:
    with pytest.raises(ValueError):
        _validated_multipliers(_values(0.5))
    with pytest.raises(ValueError):
        _validated_multipliers(_values(1.0))


def _trajectory(*, attach: bool) -> ContactWrenchTrajectory:
    knots = []
    for index in range(6):
        assignments = [
            ContactAssignment(
                slot_id=0,
                anchor_id=0,
                candidate_id=0,
                contact_mode="grasp",
                schedule_state="attach" if attach else "approach",
                wrench_target=[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0] if attach else None,
                wrench_lower=[-6.0] * 6 if attach else None,
                wrench_upper=[6.0] * 6 if attach else None,
                wrench_frame="contact",
            )
        ]
        knots.append(
            InteractionKnot(
                t_rel_s=float(index),
                contact_assignments=assignments,
                centroidal_target=CentroidalTarget(
                    com_pos_world=(float(index), 0.0, 1.0)
                ),
                posture_target=PostureTarget(joint_pos_target={"joint": float(index)}),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    )
                ],
                guard_conditions=[
                    {
                        "type": "order9_phase_local_rolling_teacher",
                        "phase": "contact_acquisition" if attach else "approach",
                        "phase_target_reached": "true" if attach else "false",
                    }
                ],
            )
        )
    return ContactWrenchTrajectory(
        horizon_s=5.0,
        dt_s=1.0,
        knots=knots,
        contract_version="contact_frame_robot_on_target_v2",
    )


def test_contact_handoff_preserves_concatenated_joint_and_spatial_path() -> None:
    approach = _trajectory(attach=False)
    contact = _trajectory(attach=True)
    repaired_approach, repaired_contact, evidence = (
        _move_approach_tail_to_contact_acquisition(
            approach,
            contact,
            handoff_fraction=0.6,
        )
    )
    original = approach.knots + contact.knots[1:]
    repaired = repaired_approach.knots + repaired_contact.knots[1:]
    assert [k.centroidal_target.com_pos_world for k in repaired] == [
        k.centroidal_target.com_pos_world for k in original
    ]
    assert [k.posture_target.joint_pos_target for k in repaired] == [
        k.posture_target.joint_pos_target for k in original
    ]
    assert all(
        a.schedule_state == "attach"
        for knot in repaired_contact.knots
        for a in knot.contact_assignments
    )
    assert evidence["joint_and_spatial_samples_reordered"] is False


def test_contact_handoff_rejects_out_of_range_fraction() -> None:
    assert _validated_handoff_fraction(0.88) == 0.88
    with pytest.raises(ValueError):
        _validated_handoff_fraction(0.99)


def test_endpoint_hold_repeats_only_the_stationary_endpoint() -> None:
    source = _trajectory(attach=True)
    repaired = _extend_phase_endpoint_hold(source, hold_s=2.0)
    assert repaired.horizon_s == 7.0
    assert [k.centroidal_target.com_pos_world for k in repaired.knots[:6]] == [
        k.centroidal_target.com_pos_world for k in source.knots
    ]
    assert all(
        knot.centroidal_target.com_pos_world
        == source.knots[-1].centroidal_target.com_pos_world
        for knot in repaired.knots[6:]
    )
    assert all(
        value == 0.0
        for knot in repaired.knots[6:]
        for value in knot.posture_target.joint_vel_target.values()
    )


def test_endpoint_hold_validation_is_bounded_and_phase_complete() -> None:
    result = _validated_endpoint_holds({"transport": 10.0})
    assert result["transport"] == 10.0
    assert result["approach"] == 0.0
    with pytest.raises(ValueError):
        _validated_endpoint_holds({"transport": 31.0})
