from __future__ import annotations

import pytest

from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION,
    Order8NaturalContactConfig,
    Order8NaturalContactObservation,
    Order8NaturalContactPhase,
    Order8RawContactPatch,
)
from amsrr.simulation.natural_contact_evidence import (
    NaturalContactEvidenceMonitor,
)


def test_complete_natural_contact_lifecycle_passes_only_after_resettable_settle_dwell() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = 0.0

    for _ in range(5):
        time_s += 0.05
        step = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                contacts=_selected_contacts(),
            )
        )
    assert step.grasp_acquired is True
    assert step.contact_dwell_elapsed_s == pytest.approx(0.25)

    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.110,
            object_floor_contact=False,
        )
    )
    assert step.lift_acquired is True

    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.TRANSPORT,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.110,
            object_floor_contact=False,
            transport_distance_m=0.200,
        )
    )
    assert step.transport_acquired is True

    # Floor contact is expected after the explicit place phase begins.
    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.PLACE,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.0,
            object_floor_contact=True,
            transport_distance_m=0.200,
        )
    )
    assert step.object_dropped is False

    for _ in range(2):
        time_s += 0.05
        step = monitor.observe(
            _observation(
                Order8NaturalContactPhase.RELEASE,
                time_s=time_s,
                contacts=[],
                object_floor_contact=True,
                transport_distance_m=0.200,
            )
        )
    assert step.release_contact_free_acquired is True

    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.RETREAT,
            time_s=time_s,
            contacts=[],
            object_floor_contact=True,
            transport_distance_m=0.200,
            gripper_object_clearance_m=0.050,
        )
    )
    assert step.retreat_clearance_acquired is True

    for _ in range(10):
        time_s += 0.05
        step = monitor.observe(
            _observation(
                Order8NaturalContactPhase.SETTLE,
                time_s=time_s,
                contacts=[],
                object_floor_contact=True,
                transport_distance_m=0.200,
                gripper_object_clearance_m=0.050,
                object_linear_speed_mps=0.05,
                object_angular_speed_rad_s=0.10,
            )
        )
    assert step.settle_dwell_elapsed_s == pytest.approx(0.50)

    # One unstable sample resets, rather than merely pausing, the dwell.
    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.SETTLE,
            time_s=time_s,
            contacts=[],
            object_floor_contact=True,
            transport_distance_m=0.200,
            gripper_object_clearance_m=0.050,
            object_linear_speed_mps=0.051,
        )
    )
    assert step.settle_dwell_elapsed_s == 0.0

    for _ in range(20):
        time_s += 0.05
        step = monitor.observe(
            _observation(
                Order8NaturalContactPhase.SETTLE,
                time_s=time_s,
                contacts=[],
                object_floor_contact=True,
                transport_distance_m=0.200,
                gripper_object_clearance_m=0.050,
                object_linear_speed_mps=0.05,
                object_angular_speed_rad_s=0.10,
            )
        )

    result = monitor.finalize()
    assert step.settle_acquired is True
    assert result.passed is True
    assert result.failure_reasons == []
    assert result.raw_contact_truth_role == "privileged_diagnostic_only"
    assert result.raw_contact_truth_actor_input is False
    assert result.raw_contact_truth_qpid_command is False


def test_contact_dwell_resets_and_requires_distinct_dock_links() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())

    one_distinct = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.10,
            dt_s=0.10,
            contacts=[
                _patch("a0", "dock_a", normal_force_n=0.3),
                _patch("a1", "dock_a", normal_force_n=0.3),
            ],
        )
    )
    assert one_distinct.selected_distinct_contact_count == 1
    assert one_distinct.selected_contact_exists is False

    two_links = _selected_contacts()
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.20,
            dt_s=0.10,
            contacts=two_links,
        )
    )
    assert step.contact_dwell_elapsed_s == pytest.approx(0.10)

    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.30,
            dt_s=0.10,
            contacts=[],
        )
    )
    assert step.contact_dwell_elapsed_s == 0.0

    for time_s in (0.40, 0.50, 0.60):
        step = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                dt_s=0.10,
                contacts=two_links,
            )
        )
    assert step.grasp_acquired is True


def test_contact_dwell_bridges_only_the_configured_break_grace() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())

    first = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            dt_s=0.05,
            contacts=_selected_contacts(),
        )
    )
    brief_gap = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.10,
            dt_s=0.05,
            contacts=[],
        )
    )

    assert first.contact_dwell_elapsed_s == pytest.approx(0.05)
    assert brief_gap.contact_dwell_elapsed_s == pytest.approx(0.05)
    assert brief_gap.grasp_acquired is False

    for time_s in (0.15, 0.20, 0.25, 0.30):
        acquired = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                dt_s=0.05,
                contacts=_selected_contacts(),
            )
        )

    assert acquired.contact_dwell_elapsed_s == pytest.approx(0.25)
    assert acquired.grasp_acquired is True


def test_penetration_alone_does_not_satisfy_selected_contact_existence() -> None:
    step = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.25,
            dt_s=0.25,
            contacts=[
                _patch(
                    "a",
                    "dock_a",
                    normal_force_n=0.0,
                    penetration_m=0.0002,
                ),
                _patch(
                    "b",
                    "dock_b",
                    normal_force_n=0.0,
                    penetration_m=0.0002,
                ),
            ],
        )
    )
    assert step.selected_contact_link_ids == []
    assert step.selected_contact_exists is False
    assert step.contact_dwell_elapsed_s == 0.0
    assert step.grasp_acquired is False


def test_zero_force_zero_penetration_proximity_patch_is_not_slip_evidence() -> None:
    step = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.APPROACH,
            time_s=0.05,
            contacts=[
                _patch(
                    "proximity",
                    "dock_a",
                    normal_force_n=0.0,
                    penetration_m=0.0,
                    slip_mps=0.5,
                )
            ],
        )
    )

    assert step.selected_contact_exists is False
    assert step.max_tangential_slip_speed_mps == 0.0
    assert "selected_contact_slip_speed_limit_exceeded" not in step.failure_reasons


def test_penetration_noise_floor_defines_selected_release_contact() -> None:
    below = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.RELEASE,
            time_s=0.05,
            contacts=[
                _patch(
                    "a",
                    "dock_a",
                    normal_force_n=0.0,
                    penetration_m=0.000099,
                ),
            ],
        )
    )
    assert below.release_contact_free_elapsed_s == pytest.approx(0.05)

    at_floor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.RELEASE,
            time_s=0.05,
            contacts=[
                _patch(
                    "a",
                    "dock_a",
                    normal_force_n=0.0,
                    penetration_m=0.0001,
                ),
            ],
        )
    )
    assert at_floor.selected_contact_link_ids == []
    assert at_floor.release_contact_free_elapsed_s == 0.0


def test_penetration_noise_floor_filters_unintended_contact() -> None:
    below = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=[
                *_selected_contacts(),
                _patch(
                    "noise",
                    "body_shell",
                    normal_force_n=0.0,
                    penetration_m=0.000099,
                ),
            ],
        )
    )
    assert below.unintended_contact_link_ids == []
    assert "unintended_robot_object_contact" not in below.failure_reasons

    at_floor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=[
                *_selected_contacts(),
                _patch(
                    "contact",
                    "body_shell",
                    normal_force_n=0.0,
                    penetration_m=0.0001,
                ),
            ],
        )
    )
    assert at_floor.unintended_contact_link_ids == ["body_shell"]
    assert "unintended_robot_object_contact" in at_floor.failure_reasons


def test_patch_magnitudes_are_summed_without_opposing_contact_cancellation() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=[
                _patch("a0", "dock_a", normal_force_n=4.0, force_n=8.0),
                _patch("a1", "dock_a", normal_force_n=4.0, force_n=7.0),
                _patch("b0", "dock_b", normal_force_n=6.0, force_n=11.0),
            ],
        )
    )

    assert step.total_selected_force_magnitude_n == pytest.approx(26.0)
    assert step.max_force_per_selected_contact_n == pytest.approx(15.0)
    assert step.selected_distinct_contact_count == 2
    assert step.hard_failure is False


def test_break_grace_is_inclusive_then_required_contact_loss_becomes_drop() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = _acquire_and_lift(monitor)

    time_s += 0.05
    grace = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=time_s,
            contacts=[],
            object_bottom_clearance_m=0.110,
            object_floor_contact=False,
        )
    )
    assert grace.contact_break_elapsed_s == pytest.approx(0.05)
    assert grace.hard_failure is False

    time_s += 0.001
    dropped = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=time_s,
            dt_s=0.001,
            contacts=[],
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
        )
    )
    assert dropped.object_dropped is True
    assert "required_contact_break_grace_exceeded" in dropped.failure_reasons
    assert "object_drop_required_contact_loss" in dropped.failure_reasons


@pytest.mark.parametrize(
    ("patch_kwargs", "failure_reason"),
    [
        ({"force_n": 30.001, "normal_force_n": 11.0}, "selected_contact_force_limit_exceeded"),
        ({"torque_nm": 5.001}, "selected_contact_torque_limit_exceeded"),
        ({"penetration_m": 0.002001}, "selected_contact_penetration_limit_exceeded"),
    ],
)
def test_hard_per_contact_safety_limits_fail_closed(
    patch_kwargs: dict[str, float], failure_reason: str
) -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    contacts = [
        _patch("a", "dock_a", **patch_kwargs),
        _patch("b", "dock_b"),
    ]
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=contacts,
        )
    )

    assert step.hard_failure is True
    assert failure_reason in step.failure_reasons


def test_provisional_slip_and_contact_loss_are_allowed_until_verified_grasp() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())

    provisional = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=_selected_contacts(slip_mps=0.030),
            simultaneous_qclose_acquired=False,
        )
    )
    assert provisional.hard_failure is False
    assert provisional.contact_dwell_elapsed_s == 0.0
    assert provisional.max_tangential_slip_speed_mps == 0.0
    assert provisional.provisional_acquisition_slip_speed_mps == pytest.approx(
        0.030
    )

    separated = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.10,
            contacts=[],
            simultaneous_qclose_acquired=True,
        )
    )
    assert separated.hard_failure is False
    assert separated.contact_dwell_elapsed_s == 0.0

    for time_s in (0.15, 0.20, 0.25, 0.30, 0.35):
        acquired = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                contacts=_selected_contacts(),
                simultaneous_qclose_acquired=True,
            )
        )
    assert acquired.grasp_acquired is True

    maintained_slip = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=0.40,
            contacts=_selected_contacts(slip_mps=0.020001),
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
            simultaneous_qclose_acquired=True,
        )
    )
    assert maintained_slip.max_tangential_slip_speed_mps == pytest.approx(
        0.020001
    )
    assert maintained_slip.gate_results["selected_slip_speed_gate_enabled"] is False
    assert maintained_slip.hard_failure is False
    result = monitor.finalize()
    assert result.max_provisional_acquisition_slip_speed_mps == pytest.approx(
        0.030
    )
    assert result.max_tangential_slip_speed_mps == pytest.approx(0.020001)


def test_diagnostic_can_record_instantaneous_slip_without_speed_failure() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = 0.0
    for _ in range(5):
        time_s += 0.05
        acquired = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                contacts=_selected_contacts(),
            )
        )
    assert acquired.grasp_acquired is True

    time_s += 0.05
    high_slip = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=time_s,
            contacts=_selected_contacts(slip_mps=0.050),
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
        )
    )

    assert high_slip.max_tangential_slip_speed_mps == pytest.approx(0.050)
    assert "selected_contact_slip_speed_limit_exceeded" not in (
        high_slip.failure_reasons
    )
    assert high_slip.hard_failure is False


def test_slip_speed_is_telemetry_while_contact_point_displacement_is_gated() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = 0.0
    for _ in range(5):
        time_s += 0.05
        monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                contacts=_selected_contacts(),
            )
        )

    for _ in range(3):
        time_s += 0.05
        step = monitor.observe(
            _observation(
                Order8NaturalContactPhase.LIFT,
                time_s=time_s,
                contacts=_selected_contacts(slip_mps=0.10),
                selected_contact_points={
                    "dock_a": [0.131, 0.0, 0.0],
                    "dock_b": [-0.10, 0.0, 0.0],
                },
                object_bottom_clearance_m=0.100,
                object_floor_contact=False,
            )
        )

    assert "selected_contact_slip_speed_limit_exceeded" not in step.failure_reasons
    assert "selected_contact_point_slip_displacement_limit_exceeded" in (
        step.failure_reasons
    )


def test_provisional_slip_speed_does_not_gate_contact_confirmation_dwell() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())

    for index in range(5):
        provisional = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=0.05 * (index + 1),
                contacts=_selected_contacts(slip_mps=0.030),
                simultaneous_qclose_acquired=True,
                grasp_confirmation_ready=True,
            )
        )

    assert provisional.contact_dwell_elapsed_s == pytest.approx(0.25)
    assert provisional.grasp_acquired is True
    assert provisional.hard_failure is False
    assert provisional.gate_results["provisional_acquisition_slip_record_only"] is True


def test_force_ramp_remains_acquisition_until_confirmation_ready() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())

    for index in range(10):
        ramping = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=0.05 * (index + 1),
                contacts=_selected_contacts(slip_mps=0.030),
                simultaneous_qclose_acquired=True,
                grasp_confirmation_ready=False,
            )
        )

    assert ramping.grasp_acquired is False
    assert ramping.contact_dwell_elapsed_s == 0.0
    assert ramping.hard_failure is False

    for index in range(5):
        confirmed = monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=0.55 + 0.05 * index,
                contacts=_selected_contacts(),
                simultaneous_qclose_acquired=True,
                grasp_confirmation_ready=True,
            )
        )

    assert confirmed.grasp_confirmation_ready is True
    assert confirmed.grasp_acquired is True


def test_contact_point_slip_is_displacement_from_fixed_grasp_reference() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    contacts = _selected_contacts(slip_mps=2.0)
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.25,
            dt_s=0.25,
            contacts=contacts,
            selected_contact_points={
                "dock_a": [0.10, 0.0, 0.0],
                "dock_b": [-0.10, 0.0, 0.0],
            },
        )
    )
    assert step.grasp_acquired is True
    assert step.max_contact_point_slip_displacement_m == 0.0
    assert step.gate_results["selected_slip_speed_gate_enabled"] is False
    assert step.hard_failure is False

    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=0.50,
            dt_s=0.25,
            contacts=contacts,
            selected_contact_points={
                "dock_a": [0.105, 0.0, 0.0],
                "dock_b": [-0.10, 0.0, 0.0],
            },
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
        )
    )
    assert step.max_contact_point_slip_displacement_m == pytest.approx(0.005)
    assert step.hard_failure is False

    # Remaining in CONTACT_ACQUISITION after grasp must not re-latch the
    # reference.  The displacement therefore grows from the original point.
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.75,
            dt_s=0.25,
            contacts=contacts,
            selected_contact_points={
                "dock_a": [0.125, 0.0, 0.0],
                "dock_b": [-0.10, 0.0, 0.0],
            },
        )
    )
    assert step.max_contact_point_slip_displacement_m == pytest.approx(0.025)
    assert step.hard_failure is False

    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.TRANSPORT,
            time_s=1.0,
            dt_s=0.25,
            contacts=contacts,
            selected_contact_points={
                "dock_a": [0.131, 0.0, 0.0],
                "dock_b": [-0.10, 0.0, 0.0],
            },
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
            transport_distance_m=0.200,
        )
    )
    assert (
        "selected_contact_point_slip_displacement_limit_exceeded"
        in step.failure_reasons
    )

    monitor.reset()
    result = monitor.finalize()
    assert result.attempted is False
    assert result.step_count == 0
    assert result.max_contact_point_slip_displacement_m_by_link == {}
    assert result.failure_reasons == []


def test_lift_clearance_loss_floor_recontact_unintended_contact_and_bad_control_fail_closed() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = _acquire_and_lift(monitor)

    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.TRANSPORT,
            time_s=time_s,
            contacts=[
                *_selected_contacts(),
                _patch("wrong", "body_shell", normal_force_n=1.0),
            ],
            object_bottom_clearance_m=0.099,
            object_floor_contact=True,
            transport_distance_m=0.200,
            controller_qp_feasible=False,
            missing_actuator_target_count=1,
        )
    )

    assert step.object_dropped is True
    assert step.unintended_contact_link_ids == ["body_shell"]
    assert "object_drop_floor_recontact" in step.failure_reasons
    assert "object_drop_lift_clearance_lost" in step.failure_reasons
    assert "unintended_robot_object_contact" in step.failure_reasons
    assert "controller_qp_infeasible" in step.failure_reasons
    assert "missing_actuator_targets" in step.failure_reasons


@pytest.mark.parametrize(
    "phase",
    [Order8NaturalContactPhase.LIFT, Order8NaturalContactPhase.TRANSPORT],
)
def test_downward_velocity_drop_threshold_is_inclusive_after_lift(
    phase: Order8NaturalContactPhase,
) -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = _acquire_and_lift(monitor)

    time_s += 0.05
    safe = monitor.observe(
        _observation(
            phase,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
            object_vertical_velocity_world_mps=-0.249,
            transport_distance_m=0.200,
        )
    )
    assert safe.object_dropped is False

    time_s += 0.05
    dropped = monitor.observe(
        _observation(
            phase,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
            object_vertical_velocity_world_mps=-0.25,
            transport_distance_m=0.200,
        )
    )
    assert dropped.object_dropped is True
    assert (
        "object_drop_downward_velocity_threshold_exceeded"
        in dropped.failure_reasons
    )


def test_downward_velocity_detector_is_inactive_until_lift_is_acquired() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    monitor.observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.25,
            dt_s=0.25,
            contacts=_selected_contacts(),
        )
    )
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=0.30,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.099,
            object_floor_contact=False,
            object_vertical_velocity_world_mps=-1.0,
        )
    )

    assert step.lift_acquired is False
    assert step.object_dropped is False
    assert (
        "object_drop_downward_velocity_threshold_exceeded"
        not in step.failure_reasons
    )


def test_invalid_or_saturated_privileged_contact_truth_fails_closed() -> None:
    invalid = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=_selected_contacts(),
            raw_contact_valid=False,
        )
    )
    saturated = NaturalContactEvidenceMonitor(Order8NaturalContactConfig()).observe(
        _observation(
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            time_s=0.05,
            contacts=_selected_contacts(),
            raw_contact_saturated=True,
        )
    )

    assert "raw_contact_truth_invalid" in invalid.failure_reasons
    assert "raw_contact_truth_saturated" in saturated.failure_reasons


def test_selected_link_order_is_irrelevant_but_settle_cannot_skip_retreat() -> None:
    monitor = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    first = _observation(
        Order8NaturalContactPhase.APPROACH,
        time_s=0.05,
        contacts=[],
    )
    monitor.observe(first)
    reordered = _observation(
        Order8NaturalContactPhase.APPROACH,
        time_s=0.10,
        contacts=[],
    )
    reordered.selected_dock_link_ids = ["dock_b", "dock_a"]
    step = monitor.observe(reordered)
    assert "selected_dock_link_identity_changed" not in step.failure_reasons

    lifecycle = NaturalContactEvidenceMonitor(Order8NaturalContactConfig())
    time_s = _acquire_and_lift(lifecycle)
    time_s += 0.05
    lifecycle.observe(
        _observation(
            Order8NaturalContactPhase.TRANSPORT,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.100,
            object_floor_contact=False,
            transport_distance_m=0.200,
        )
    )
    time_s += 0.05
    lifecycle.observe(
        _observation(
            Order8NaturalContactPhase.PLACE,
            time_s=time_s,
            contacts=_selected_contacts(),
            transport_distance_m=0.200,
        )
    )
    for _ in range(2):
        time_s += 0.05
        lifecycle.observe(
            _observation(
                Order8NaturalContactPhase.RELEASE,
                time_s=time_s,
                contacts=[],
                transport_distance_m=0.200,
            )
        )
    time_s += 0.05
    skipped = lifecycle.observe(
        _observation(
            Order8NaturalContactPhase.SETTLE,
            time_s=time_s,
            contacts=[],
            transport_distance_m=0.200,
            gripper_object_clearance_m=0.050,
        )
    )
    assert "retreat_clearance_not_acquired" in skipped.failure_reasons


def _acquire_and_lift(monitor: NaturalContactEvidenceMonitor) -> float:
    time_s = 0.0
    for _ in range(5):
        time_s += 0.05
        monitor.observe(
            _observation(
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                time_s=time_s,
                contacts=_selected_contacts(),
            )
        )
    time_s += 0.05
    step = monitor.observe(
        _observation(
            Order8NaturalContactPhase.LIFT,
            time_s=time_s,
            contacts=_selected_contacts(),
            object_bottom_clearance_m=0.110,
            object_floor_contact=False,
        )
    )
    assert step.lift_acquired is True
    return time_s


def _selected_contacts(*, slip_mps: float = 0.001) -> list[Order8RawContactPatch]:
    return [
        _patch("selected_a", "dock_a", slip_mps=slip_mps),
        _patch("selected_b", "dock_b", slip_mps=slip_mps),
    ]


def _patch(
    patch_id: str,
    link_id: str,
    *,
    normal_force_n: float = 11.0,
    force_n: float | None = None,
    torque_nm: float = 0.1,
    penetration_m: float = 0.0005,
    slip_mps: float = 0.001,
) -> Order8RawContactPatch:
    return Order8RawContactPatch(
        patch_id=patch_id,
        robot_link_id=link_id,
        other_body_id="payload",
        normal_force_n=normal_force_n,
        force_magnitude_n=(normal_force_n if force_n is None else force_n),
        torque_magnitude_nm=torque_nm,
        penetration_m=penetration_m,
        tangential_slip_speed_mps=slip_mps,
    )


def _observation(
    phase: Order8NaturalContactPhase,
    *,
    time_s: float,
    contacts: list[Order8RawContactPatch],
    dt_s: float = 0.05,
    object_bottom_clearance_m: float = 0.0,
    object_floor_contact: bool = True,
    object_linear_speed_mps: float = 0.0,
    object_vertical_velocity_world_mps: float = 0.0,
    object_angular_speed_rad_s: float = 0.0,
    transport_distance_m: float = 0.0,
    gripper_object_clearance_m: float = 0.0,
    controller_qp_feasible: bool = True,
    missing_actuator_target_count: int = 0,
    raw_contact_valid: bool = True,
    raw_contact_saturated: bool = False,
    simultaneous_qclose_acquired: bool | None = None,
    grasp_confirmation_ready: bool | None = None,
    selected_contact_points: dict[str, list[float]] | None = None,
) -> Order8NaturalContactObservation:
    if simultaneous_qclose_acquired is None:
        simultaneous_qclose_acquired = phase in {
            Order8NaturalContactPhase.CONTACT_ACQUISITION,
            Order8NaturalContactPhase.LIFT,
            Order8NaturalContactPhase.TRANSPORT,
            Order8NaturalContactPhase.PLACE,
            Order8NaturalContactPhase.RELEASE,
            Order8NaturalContactPhase.RETREAT,
            Order8NaturalContactPhase.SETTLE,
            Order8NaturalContactPhase.COMPLETE,
        }
    if grasp_confirmation_ready is None:
        grasp_confirmation_ready = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and simultaneous_qclose_acquired
        )
    return Order8NaturalContactObservation(
        observation_version=ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION,
        phase=phase,
        time_s=time_s,
        step_dt_s=dt_s,
        object_id="payload",
        selected_dock_link_ids=["dock_a", "dock_b"],
        raw_contact_patches=contacts,
        selected_contact_point_object_m_by_link=(
            {
                "dock_a": [0.10, 0.0, 0.0],
                "dock_b": [-0.10, 0.0, 0.0],
            }
            if selected_contact_points is None
            else selected_contact_points
        ),
        raw_contact_valid=raw_contact_valid,
        raw_contact_saturated=raw_contact_saturated,
        object_bottom_clearance_m=object_bottom_clearance_m,
        object_floor_contact=object_floor_contact,
        object_linear_speed_mps=object_linear_speed_mps,
        object_vertical_velocity_world_mps=object_vertical_velocity_world_mps,
        object_angular_speed_rad_s=object_angular_speed_rad_s,
        transport_distance_m=transport_distance_m,
        gripper_object_clearance_m=gripper_object_clearance_m,
        controller_qp_feasible=controller_qp_feasible,
        simultaneous_qclose_acquired=simultaneous_qclose_acquired,
        grasp_confirmation_ready=grasp_confirmation_ready,
        missing_actuator_target_count=missing_actuator_target_count,
    )
