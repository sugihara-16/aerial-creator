from __future__ import annotations

"""Deterministic, Isaac-independent Order 8 natural-contact evidence monitor."""

from dataclasses import dataclass, field
import math

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
    ORDER8_NATURAL_CONTACT_MODEL,
    ORDER8_NATURAL_CONTACT_RESULT_VERSION,
    ORDER8_NATURAL_CONTACT_STEP_EVIDENCE_VERSION,
    Order8NaturalContactConfig,
    Order8NaturalContactObservation,
    Order8NaturalContactPhase,
    Order8NaturalContactResult,
    Order8NaturalContactStepEvidence,
)


_EPS = 1.0e-12
_CONTACT_REQUIRED_PHASES = {
    Order8NaturalContactPhase.LIFT,
    Order8NaturalContactPhase.TRANSPORT,
    Order8NaturalContactPhase.PLACE,
}
_MAINTAINED_CONTACT_PHASES = {
    Order8NaturalContactPhase.CONTACT_ACQUISITION,
    *_CONTACT_REQUIRED_PHASES,
}
_DROP_SENSITIVE_PHASES = {
    Order8NaturalContactPhase.LIFT,
    Order8NaturalContactPhase.TRANSPORT,
}
_POST_RELEASE_PHASES = {
    Order8NaturalContactPhase.RETREAT,
    Order8NaturalContactPhase.SETTLE,
    Order8NaturalContactPhase.COMPLETE,
}


@dataclass
class _LinkContactMetrics:
    normal_force_n: float = 0.0
    force_magnitude_n: float = 0.0
    torque_magnitude_nm: float = 0.0
    max_penetration_m: float = 0.0
    max_slip_speed_mps: float = 0.0
    patch_count: int = 0

    def add(
        self,
        *,
        normal_force_n: float,
        force_magnitude_n: float,
        torque_magnitude_nm: float,
        penetration_m: float,
        slip_speed_mps: float,
    ) -> None:
        # The magnitudes are deliberately summed patch-by-patch.  No vector
        # resultant is formed, so opposing contact forces cannot cancel.
        self.normal_force_n += float(normal_force_n)
        self.force_magnitude_n += float(force_magnitude_n)
        self.torque_magnitude_nm += float(torque_magnitude_nm)
        self.max_penetration_m = max(
            self.max_penetration_m, float(penetration_m)
        )
        self.max_slip_speed_mps = max(
            self.max_slip_speed_mps, float(slip_speed_mps)
        )
        self.patch_count += 1


@dataclass
class NaturalContactEvidenceMonitor:
    """Resettable phase-sensitive monitor for one Order 8 episode.

    The monitor consumes already-resolved non-aggregated contact patches and
    ordinary scalar state/control diagnostics.  It has no Isaac dependency and
    emits no policy or actuator command.
    """

    config: Order8NaturalContactConfig
    _selected_dock_link_ids: tuple[str, ...] | None = field(
        default=None, init=False
    )
    _last_time_s: float | None = field(default=None, init=False)
    _last_phase: Order8NaturalContactPhase | None = field(default=None, init=False)
    _final_phase: Order8NaturalContactPhase = field(
        default=Order8NaturalContactPhase.RESET, init=False
    )
    _step_count: int = field(default=0, init=False)
    _duration_s: float = field(default=0.0, init=False)
    _contact_dwell_s: float = field(default=0.0, init=False)
    _contact_dwell_gap_s: float = field(default=0.0, init=False)
    _contact_break_s: float = field(default=0.0, init=False)
    _release_contact_free_dwell_s: float = field(default=0.0, init=False)
    _settle_dwell_s: float = field(default=0.0, init=False)
    _grasp_acquired: bool = field(default=False, init=False)
    _lift_acquired: bool = field(default=False, init=False)
    _transport_acquired: bool = field(default=False, init=False)
    _place_started: bool = field(default=False, init=False)
    _release_contact_free_acquired: bool = field(default=False, init=False)
    _retreat_clearance_acquired: bool = field(default=False, init=False)
    _settle_acquired: bool = field(default=False, init=False)
    _object_dropped: bool = field(default=False, init=False)
    _simultaneous_qclose_seen: bool = field(default=False, init=False)
    _grasp_contact_point_object_m_by_link: dict[str, tuple[float, float, float]] = field(
        default_factory=dict, init=False
    )
    _contact_point_slip_displacement_m_by_link: dict[str, float] = field(
        default_factory=dict, init=False
    )
    _max_contact_point_slip_displacement_m_by_link: dict[str, float] = field(
        default_factory=dict, init=False
    )
    _max_force_per_contact_n: float = field(default=0.0, init=False)
    _max_torque_per_contact_nm: float = field(default=0.0, init=False)
    _max_penetration_m: float = field(default=0.0, init=False)
    _max_slip_speed_mps: float = field(default=0.0, init=False)
    _max_provisional_acquisition_slip_speed_mps: float = field(
        default=0.0, init=False
    )
    _unintended_contact_count: int = field(default=0, init=False)
    _failure_reasons: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        """Clear all dwell, slip, terminal, and identity state."""

        self._selected_dock_link_ids = None
        self._last_time_s = None
        self._last_phase = None
        self._final_phase = Order8NaturalContactPhase.RESET
        self._step_count = 0
        self._duration_s = 0.0
        self._contact_dwell_s = 0.0
        self._contact_dwell_gap_s = 0.0
        self._contact_break_s = 0.0
        self._release_contact_free_dwell_s = 0.0
        self._settle_dwell_s = 0.0
        self._grasp_acquired = False
        self._lift_acquired = False
        self._transport_acquired = False
        self._place_started = False
        self._release_contact_free_acquired = False
        self._retreat_clearance_acquired = False
        self._settle_acquired = False
        self._object_dropped = False
        self._simultaneous_qclose_seen = False
        self._grasp_contact_point_object_m_by_link = {}
        self._contact_point_slip_displacement_m_by_link = {}
        self._max_contact_point_slip_displacement_m_by_link = {}
        self._max_force_per_contact_n = 0.0
        self._max_torque_per_contact_nm = 0.0
        self._max_penetration_m = 0.0
        self._max_slip_speed_mps = 0.0
        self._max_provisional_acquisition_slip_speed_mps = 0.0
        self._unintended_contact_count = 0
        self._failure_reasons = []

    def observe(
        self, observation: Order8NaturalContactObservation
    ) -> Order8NaturalContactStepEvidence:
        observation.validate()
        if (
            self._last_time_s is not None
            and observation.time_s <= self._last_time_s + _EPS
        ):
            raise SchemaValidationError(
                "Order 8 observation time must increase strictly; reset the monitor for a new episode"
            )

        phase = observation.phase
        dt_s = float(observation.step_dt_s)
        # Selection identity is set-like.  Sensor/view ordering may change
        # without changing which physical Dock links were selected.
        selected_ids = tuple(sorted(observation.selected_dock_link_ids))
        if self._selected_dock_link_ids is None:
            self._selected_dock_link_ids = selected_ids
            self._contact_point_slip_displacement_m_by_link = {
                link_id: 0.0 for link_id in selected_ids
            }
            self._max_contact_point_slip_displacement_m_by_link = {
                link_id: 0.0 for link_id in selected_ids
            }
        elif selected_ids != self._selected_dock_link_ids:
            self._record_failure("selected_dock_link_identity_changed")

        if len(selected_ids) < self.config.required_distinct_dock_links:
            self._record_failure("insufficient_selected_distinct_dock_links")

        if self._last_phase is not None and phase == Order8NaturalContactPhase.RESET:
            self._record_failure("reset_phase_requires_monitor_reset")

        self._step_count += 1
        self._duration_s += dt_s
        self._last_time_s = float(observation.time_s)
        self._final_phase = phase

        phase_changed = self._last_phase is not None and phase != self._last_phase
        if phase_changed and phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            self._contact_dwell_s = 0.0
            self._contact_dwell_gap_s = 0.0
        if phase_changed and phase in _CONTACT_REQUIRED_PHASES:
            self._contact_break_s = 0.0
        if phase_changed and phase == Order8NaturalContactPhase.RELEASE:
            self._release_contact_free_dwell_s = 0.0
        if phase_changed and phase == Order8NaturalContactPhase.SETTLE:
            self._settle_dwell_s = 0.0
        self._last_phase = phase

        if observation.simultaneous_qclose_acquired:
            self._simultaneous_qclose_seen = True
        elif self._simultaneous_qclose_seen:
            self._record_failure("simultaneous_qclose_acquisition_regressed")
        maintained_contact_active = bool(
            self._grasp_acquired and phase in _MAINTAINED_CONTACT_PHASES
        )
        provisional_acquisition_active = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and not self._grasp_acquired
        )

        selected, other_robot_object = _group_robot_object_contacts(
            observation=observation,
            selected_link_ids=set(selected_ids),
        )
        selected_contact_link_ids = sorted(
            link_id
            for link_id, metrics in selected.items()
            if metrics.normal_force_n
            >= self.config.contact_normal_force_threshold_n
        )
        selected_physical_link_ids = sorted(
            link_id
            for link_id, metrics in selected.items()
            if _is_physical_contact(metrics, self.config)
        )
        selected_physical = {
            link_id: selected[link_id]
            for link_id in selected_physical_link_ids
        }
        unintended_link_ids = sorted(
            link_id
            for link_id, metrics in other_robot_object.items()
            if _is_physical_contact(metrics, self.config)
        )
        self._unintended_contact_count += len(unintended_link_ids)

        force_by_selected_link = {
            link_id: metrics.force_magnitude_n
            for link_id, metrics in selected_physical.items()
        }
        torque_by_selected_link = {
            link_id: metrics.torque_magnitude_nm
            for link_id, metrics in selected_physical.items()
        }
        step_total_force = sum(force_by_selected_link.values())
        step_max_force = max(force_by_selected_link.values(), default=0.0)
        step_max_torque = max(torque_by_selected_link.values(), default=0.0)
        step_max_penetration = max(
            (
                metrics.max_penetration_m
                for metrics in selected_physical.values()
            ),
            default=0.0,
        )
        step_max_slip = max(
            (
                metrics.max_slip_speed_mps
                for metrics in selected_physical.values()
            ),
            default=0.0,
        )
        self._max_force_per_contact_n = max(
            self._max_force_per_contact_n, step_max_force
        )
        self._max_torque_per_contact_nm = max(
            self._max_torque_per_contact_nm, step_max_torque
        )
        self._max_penetration_m = max(
            self._max_penetration_m, step_max_penetration
        )
        maintained_step_max_slip = (
            step_max_slip if maintained_contact_active else 0.0
        )
        provisional_step_max_slip = (
            step_max_slip if provisional_acquisition_active else 0.0
        )
        self._max_slip_speed_mps = max(
            self._max_slip_speed_mps, maintained_step_max_slip
        )
        self._max_provisional_acquisition_slip_speed_mps = max(
            self._max_provisional_acquisition_slip_speed_mps,
            provisional_step_max_slip,
        )

        contact_point_coverage_valid = True
        if maintained_contact_active:
            for link_id in selected_contact_link_ids:
                point = observation.selected_contact_point_object_m_by_link.get(
                    link_id
                )
                reference = self._grasp_contact_point_object_m_by_link.get(link_id)
                if point is None or reference is None:
                    contact_point_coverage_valid = False
                    continue
                displacement = math.sqrt(
                    sum(
                        (float(point[axis]) - float(reference[axis])) ** 2
                        for axis in range(3)
                    )
                )
                self._contact_point_slip_displacement_m_by_link[link_id] = (
                    displacement
                )
                self._max_contact_point_slip_displacement_m_by_link[link_id] = max(
                    self._max_contact_point_slip_displacement_m_by_link.get(
                        link_id, 0.0
                    ),
                    displacement,
                )
        max_contact_point_slip_displacement = max(
            self._contact_point_slip_displacement_m_by_link.values(), default=0.0
        )

        if not observation.raw_contact_valid:
            self._record_failure("raw_contact_truth_invalid")
        if observation.raw_contact_saturated:
            self._record_failure("raw_contact_truth_saturated")
        if unintended_link_ids:
            self._record_failure("unintended_robot_object_contact")
        if step_max_force > self.config.max_force_per_contact_n + _EPS:
            self._record_failure("selected_contact_force_limit_exceeded")
        if step_max_torque > self.config.max_torque_per_contact_nm + _EPS:
            self._record_failure("selected_contact_torque_limit_exceeded")
        if step_max_penetration > self.config.max_penetration_m + _EPS:
            self._record_failure("selected_contact_penetration_limit_exceeded")
        if maintained_contact_active and not contact_point_coverage_valid:
            self._record_failure("selected_contact_point_observation_missing")
        if (
            max_contact_point_slip_displacement
            > self.config.max_contact_point_slip_displacement_m + _EPS
        ):
            self._record_failure(
                "selected_contact_point_slip_displacement_limit_exceeded"
            )

        controller_safe = _controller_safe(observation)
        if not observation.controller_qp_feasible:
            self._record_failure("controller_qp_infeasible")
        if observation.missing_actuator_target_count:
            self._record_failure("missing_actuator_targets")
        if observation.unsupported_actuator_target_count:
            self._record_failure("unsupported_actuator_targets")
        if observation.clipped_actuator_target_count:
            self._record_failure("clipped_actuator_targets")
        if observation.unresolved_actuator_target_count:
            self._record_failure("unresolved_actuator_targets")

        selected_contact_exists = (
            len(selected_contact_link_ids)
            >= self.config.required_distinct_dock_links
        )
        contact_step_safe = bool(
            observation.raw_contact_valid
            and not observation.raw_contact_saturated
            and not unintended_link_ids
            and controller_safe
            and step_max_force <= self.config.max_force_per_contact_n + _EPS
            and step_max_torque <= self.config.max_torque_per_contact_nm + _EPS
            and step_max_penetration <= self.config.max_penetration_m + _EPS
            and contact_point_coverage_valid
            and max_contact_point_slip_displacement
            <= self.config.max_contact_point_slip_displacement_m + _EPS
        )
        required_contact_gate = bool(
            self._simultaneous_qclose_seen
            and selected_contact_exists
            and contact_step_safe
        )

        if phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            if not observation.grasp_confirmation_ready:
                self._contact_dwell_s = 0.0
                self._contact_dwell_gap_s = 0.0
            elif required_contact_gate:
                self._contact_dwell_s += dt_s
                self._contact_dwell_gap_s = 0.0
            elif (
                self._simultaneous_qclose_seen
                and contact_step_safe
                and not selected_contact_exists
                and self._contact_dwell_s > 0.0
            ):
                # Use the same brief-contact-loss hysteresis while verifying
                # the grasp that is already enforced after verification.  A
                # missing sample does not add dwell; it only preserves the
                # accumulated true-contact time inside the configured grace.
                self._contact_dwell_gap_s += dt_s
                if (
                    self._contact_dwell_gap_s
                    > self.config.contact_break_grace_s + _EPS
                ):
                    self._contact_dwell_s = 0.0
            else:
                self._contact_dwell_s = 0.0
                self._contact_dwell_gap_s = 0.0
            if (
                not self._grasp_acquired
                and self._contact_dwell_s + _EPS >= self.config.contact_dwell_s
            ):
                self._grasp_acquired = True
                missing_reference_ids = [
                    link_id
                    for link_id in selected_contact_link_ids
                    if link_id
                    not in observation.selected_contact_point_object_m_by_link
                ]
                if missing_reference_ids:
                    self._record_failure(
                        "selected_contact_point_reference_missing_at_grasp"
                    )
                else:
                    self._grasp_contact_point_object_m_by_link = {
                        link_id: tuple(
                            float(value)
                            for value in observation.selected_contact_point_object_m_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_contact_link_ids
                    }

        if phase in _CONTACT_REQUIRED_PHASES:
            if not self._grasp_acquired:
                self._record_failure("contact_dwell_not_acquired_before_motion")
            if required_contact_gate:
                self._contact_break_s = 0.0
            else:
                self._contact_break_s += dt_s
                if self._contact_break_s > self.config.contact_break_grace_s + _EPS:
                    self._record_failure("required_contact_break_grace_exceeded")
                    if self._lift_acquired and phase in _DROP_SENSITIVE_PHASES:
                        self._mark_drop("object_drop_required_contact_loss")
        else:
            self._contact_break_s = 0.0

        if phase == Order8NaturalContactPhase.LIFT and self._grasp_acquired:
            if (
                observation.object_bottom_clearance_m + _EPS
                >= self.config.minimum_lift_clearance_m
                and not observation.object_floor_contact
            ):
                self._lift_acquired = True

        if self._lift_acquired and phase in _DROP_SENSITIVE_PHASES:
            if (
                observation.object_vertical_velocity_world_mps
                <= -self.config.downward_drop_velocity_threshold_mps
            ):
                self._mark_drop(
                    "object_drop_downward_velocity_threshold_exceeded"
                )
            if observation.object_floor_contact:
                self._mark_drop("object_drop_floor_recontact")
            if (
                observation.object_bottom_clearance_m + _EPS
                < self.config.minimum_lift_clearance_m
            ):
                self._mark_drop("object_drop_lift_clearance_lost")

        if phase == Order8NaturalContactPhase.TRANSPORT:
            if not self._lift_acquired:
                self._record_failure("lift_clearance_not_acquired_before_transport")
            if (
                observation.transport_distance_m + _EPS
                >= self.config.required_transport_distance_m
            ):
                self._transport_acquired = True

        if phase == Order8NaturalContactPhase.PLACE:
            self._place_started = True
            if not self._transport_acquired:
                self._record_failure("transport_distance_not_acquired_before_place")

        release_contact_free = bool(
            not selected_physical_link_ids
            and not unintended_link_ids
            and observation.raw_contact_valid
            and not observation.raw_contact_saturated
            and controller_safe
        )
        if phase == Order8NaturalContactPhase.RELEASE:
            if not self._place_started:
                self._record_failure("release_started_before_intended_place")
            if release_contact_free:
                self._release_contact_free_dwell_s += dt_s
            else:
                self._release_contact_free_dwell_s = 0.0
            if (
                self._release_contact_free_dwell_s + _EPS
                >= self.config.release_contact_free_dwell_s
            ):
                self._release_contact_free_acquired = True

        if phase in _POST_RELEASE_PHASES:
            if not self._release_contact_free_acquired:
                self._record_failure("release_contact_free_dwell_not_acquired")
            if selected_physical_link_ids:
                self._record_failure("post_release_selected_contact")
        if phase == Order8NaturalContactPhase.RETREAT:
            if (
                observation.gripper_object_clearance_m + _EPS
                >= self.config.gripper_retreat_clearance_m
                and release_contact_free
            ):
                self._retreat_clearance_acquired = True
        if phase in {
            Order8NaturalContactPhase.SETTLE,
            Order8NaturalContactPhase.COMPLETE,
        } and not self._retreat_clearance_acquired:
            self._record_failure("retreat_clearance_not_acquired")

        settle_stable = bool(
            phase == Order8NaturalContactPhase.SETTLE
            and self._release_contact_free_acquired
            and self._retreat_clearance_acquired
            and release_contact_free
            and observation.gripper_object_clearance_m + _EPS
            >= self.config.gripper_retreat_clearance_m
            and observation.object_linear_speed_mps
            <= self.config.settle_linear_speed_mps + _EPS
            and observation.object_angular_speed_rad_s
            <= self.config.settle_angular_speed_rad_s + _EPS
            and not self._failure_reasons
        )
        if phase == Order8NaturalContactPhase.SETTLE:
            if settle_stable:
                self._settle_dwell_s += dt_s
            else:
                self._settle_dwell_s = 0.0
            if (
                self._settle_dwell_s + _EPS
                >= self.config.post_release_settle_dwell_s
            ):
                self._settle_acquired = True

        if phase == Order8NaturalContactPhase.COMPLETE and not self._all_gates_passed():
            self._record_failure("complete_phase_entered_before_all_gates")
        if phase == Order8NaturalContactPhase.SAFE_HOLD:
            self._record_failure("safe_hold_entered")

        gate_results = {
            "raw_contact_truth_valid": bool(
                observation.raw_contact_valid and not observation.raw_contact_saturated
            ),
            "controller_safe": controller_safe,
            "no_unintended_robot_object_contact": not unintended_link_ids,
            "required_distinct_selected_contacts": selected_contact_exists,
            "simultaneous_qclose_acquired": self._simultaneous_qclose_seen,
            "grasp_confirmation_ready": observation.grasp_confirmation_ready,
            "maintained_contact_slip_enforcement_active": (
                maintained_contact_active
            ),
            "provisional_acquisition_slip_record_only": (
                provisional_acquisition_active
            ),
            "selected_force_safe": (
                step_max_force <= self.config.max_force_per_contact_n + _EPS
            ),
            "selected_torque_safe": (
                step_max_torque <= self.config.max_torque_per_contact_nm + _EPS
            ),
            "selected_penetration_safe": (
                step_max_penetration <= self.config.max_penetration_m + _EPS
            ),
            "selected_slip_speed_safe": True,
            "selected_slip_speed_gate_enabled": False,
            "selected_contact_point_observation_valid": (
                contact_point_coverage_valid
            ),
            "selected_contact_point_slip_safe": (
                max_contact_point_slip_displacement
                <= self.config.max_contact_point_slip_displacement_m + _EPS
            ),
            "grasp_acquired": self._grasp_acquired,
            "lift_acquired": self._lift_acquired,
            "transport_acquired": self._transport_acquired,
            "release_contact_free_acquired": self._release_contact_free_acquired,
            "retreat_clearance_acquired": self._retreat_clearance_acquired,
            "settle_acquired": self._settle_acquired,
            "object_not_dropped": not self._object_dropped,
        }
        return Order8NaturalContactStepEvidence(
            evidence_version=ORDER8_NATURAL_CONTACT_STEP_EVIDENCE_VERSION,
            phase=phase,
            time_s=float(observation.time_s),
            selected_dock_link_ids=list(selected_ids),
            selected_contact_link_ids=selected_contact_link_ids,
            selected_distinct_contact_count=len(selected_contact_link_ids),
            selected_contact_exists=selected_contact_exists,
            simultaneous_qclose_acquired=self._simultaneous_qclose_seen,
            grasp_confirmation_ready=observation.grasp_confirmation_ready,
            contact_dwell_elapsed_s=self._contact_dwell_s,
            grasp_acquired=self._grasp_acquired,
            contact_break_elapsed_s=self._contact_break_s,
            lift_acquired=self._lift_acquired,
            transport_acquired=self._transport_acquired,
            release_contact_free_elapsed_s=self._release_contact_free_dwell_s,
            release_contact_free_acquired=self._release_contact_free_acquired,
            retreat_clearance_acquired=self._retreat_clearance_acquired,
            settle_dwell_elapsed_s=self._settle_dwell_s,
            settle_acquired=self._settle_acquired,
            object_dropped=self._object_dropped,
            controller_safe=controller_safe,
            unintended_contact_link_ids=unintended_link_ids,
            total_selected_force_magnitude_n=step_total_force,
            max_force_per_selected_contact_n=step_max_force,
            max_torque_per_selected_contact_nm=step_max_torque,
            max_penetration_m=step_max_penetration,
            max_tangential_slip_speed_mps=maintained_step_max_slip,
            provisional_acquisition_slip_speed_mps=(
                provisional_step_max_slip
            ),
            contact_point_slip_displacement_m_by_link=dict(
                self._contact_point_slip_displacement_m_by_link
            ),
            max_contact_point_slip_displacement_m=(
                max_contact_point_slip_displacement
            ),
            object_bottom_clearance_m=float(
                observation.object_bottom_clearance_m
            ),
            transport_distance_m=float(observation.transport_distance_m),
            gripper_object_clearance_m=float(
                observation.gripper_object_clearance_m
            ),
            hard_failure=bool(self._failure_reasons),
            failure_reasons=list(self._failure_reasons),
            gate_results=gate_results,
        )

    def finalize(self) -> Order8NaturalContactResult:
        selected_ids = list(self._selected_dock_link_ids or ())
        return Order8NaturalContactResult(
            result_version=ORDER8_NATURAL_CONTACT_RESULT_VERSION,
            config_version=ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
            config_hash=self.config.stable_hash(),
            contact_model=ORDER8_NATURAL_CONTACT_MODEL,
            final_phase=self._final_phase,
            attempted=self._step_count > 0,
            passed=self._all_gates_passed(),
            step_count=self._step_count,
            duration_s=self._duration_s,
            selected_dock_link_ids=selected_ids,
            grasp_acquired=self._grasp_acquired,
            lift_acquired=self._lift_acquired,
            transport_acquired=self._transport_acquired,
            release_contact_free_acquired=self._release_contact_free_acquired,
            retreat_clearance_acquired=self._retreat_clearance_acquired,
            settle_acquired=self._settle_acquired,
            object_dropped=self._object_dropped,
            unintended_contact_count=self._unintended_contact_count,
            max_force_per_selected_contact_n=self._max_force_per_contact_n,
            max_torque_per_selected_contact_nm=self._max_torque_per_contact_nm,
            max_penetration_m=self._max_penetration_m,
            max_tangential_slip_speed_mps=self._max_slip_speed_mps,
            max_provisional_acquisition_slip_speed_mps=(
                self._max_provisional_acquisition_slip_speed_mps
            ),
            max_contact_point_slip_displacement_m_by_link=dict(
                self._max_contact_point_slip_displacement_m_by_link
            ),
            failure_reasons=list(self._failure_reasons),
        )

    def _all_gates_passed(self) -> bool:
        return bool(
            self._step_count > 0
            and self._grasp_acquired
            and self._lift_acquired
            and self._transport_acquired
            and self._release_contact_free_acquired
            and self._retreat_clearance_acquired
            and self._settle_acquired
            and not self._object_dropped
            and self._unintended_contact_count == 0
            and not self._failure_reasons
            and self._final_phase
            in {Order8NaturalContactPhase.SETTLE, Order8NaturalContactPhase.COMPLETE}
        )

    def _record_failure(self, reason: str) -> None:
        if reason not in self._failure_reasons:
            self._failure_reasons.append(reason)

    def _mark_drop(self, reason: str) -> None:
        self._object_dropped = True
        self._record_failure(reason)


def _group_robot_object_contacts(
    *,
    observation: Order8NaturalContactObservation,
    selected_link_ids: set[str],
) -> tuple[dict[str, _LinkContactMetrics], dict[str, _LinkContactMetrics]]:
    selected: dict[str, _LinkContactMetrics] = {}
    other: dict[str, _LinkContactMetrics] = {}
    for patch in observation.raw_contact_patches:
        if patch.other_body_id != observation.object_id:
            continue
        target = selected if patch.robot_link_id in selected_link_ids else other
        metrics = target.setdefault(patch.robot_link_id, _LinkContactMetrics())
        metrics.add(
            normal_force_n=patch.normal_force_n,
            force_magnitude_n=patch.force_magnitude_n,
            torque_magnitude_nm=patch.torque_magnitude_nm,
            penetration_m=patch.penetration_m,
            slip_speed_mps=patch.tangential_slip_speed_mps,
        )
    return selected, other


def _is_physical_contact(
    metrics: _LinkContactMetrics,
    config: Order8NaturalContactConfig,
) -> bool:
    return bool(
        metrics.normal_force_n >= config.contact_normal_force_threshold_n
        or metrics.max_penetration_m
        >= config.contact_penetration_noise_floor_m
    )


def _controller_safe(observation: Order8NaturalContactObservation) -> bool:
    return bool(
        observation.controller_qp_feasible
        and observation.missing_actuator_target_count == 0
        and observation.unsupported_actuator_target_count == 0
        and observation.clipped_actuator_target_count == 0
        and observation.unresolved_actuator_target_count == 0
    )
