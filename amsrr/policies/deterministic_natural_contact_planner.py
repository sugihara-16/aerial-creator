from __future__ import annotations

"""Deterministic contact-aware pi_H fallback for P4-full Order 8.

The planner owns contact scheduling and task-space references only.  It never
emits actuator commands.  Exact Isaac contact truth remains in the separate
Order-8 evidence/safety path; this class consumes only the resulting Boolean
gates and public task progress needed by the deterministic fallback.
"""

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.order8 import (
    Order8NaturalContactPhase,
)
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)


ORDER8_DETERMINISTIC_PI_H_VERSION = "order8_deterministic_natural_contact_pi_h_v1"
# Free-space Dock morphing is position-primary.  The representative grasp has
# one selected surface whose inward translation and fixed orientation are not
# simultaneously first-order reachable at the neutral posture.  Retaining a
# small orientation weight regularizes the mesh attitude without suppressing
# the motion needed to discover q_close from physical arrest.
ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT = 0.05


@dataclass(frozen=True)
class NaturalContactAnchorSelection:
    anchor_id: int
    slot_id: int
    candidate_id: int
    dock_link_id: str
    inward_normal_world: tuple[float, float, float]


@dataclass(frozen=True)
class NaturalContactPlannerConfig:
    horizon_s: float = 1.0
    knot_dt_s: float = 0.1
    phase_timeout_s: float = 15.0
    contact_acquisition_timeout_s: float = 60.0
    normal_force_target_per_contact_n: float = 11.0


@dataclass(frozen=True)
class NaturalContactPlannerFeedback:
    time_s: float
    hover_ready: bool
    simultaneous_reachability_passed: bool
    pregrasp_aligned: bool
    contact_command_dwell_complete: bool
    lift_clearance_reached: bool
    transport_distance_reached: bool
    intended_place_pose_reached: bool
    release_command_dwell_complete: bool
    retreat_clearance_reached: bool
    post_release_settle_complete: bool
    desired_body_pose_by_phase: Mapping[Order8NaturalContactPhase, Pose7D]
    desired_anchor_pose_by_id: Mapping[int, Pose7D]
    desired_body_linear_velocity_by_phase: Mapping[
        Order8NaturalContactPhase, tuple[float, float, float]
    ] = field(default_factory=dict)
    desired_object_pose_by_phase: Mapping[Order8NaturalContactPhase, Pose7D] = field(
        default_factory=dict
    )
    contact_force_scale: float = 1.0
    contact_force_scale_by_anchor_id: Mapping[int, float] = field(
        default_factory=dict
    )
    anchor_pose_priority_by_id: Mapping[int, float] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class NaturalContactPlannerTransition:
    from_phase: Order8NaturalContactPhase
    to_phase: Order8NaturalContactPhase
    time_s: float
    reason: str


class DeterministicNaturalContactPlanner:
    """State-dependent Order-8 planner behind the standard pi_H output.

    The runtime first calls :meth:`observe` with hardware-observable progress and
    fail-closed evidence gates, then calls :meth:`plan` with the unchanged high-
    level policy context.  Contact wrench requirements remain in
    ``ContactWrenchTrajectory``; the low-level deterministic joint controller
    maps them to absolute local-joint targets and bounded torque bias.
    """

    def __init__(
        self,
        selections: Sequence[NaturalContactAnchorSelection],
        *,
        config: NaturalContactPlannerConfig | None = None,
    ) -> None:
        self.selections = tuple(sorted(selections, key=lambda item: item.anchor_id))
        self.config = config or NaturalContactPlannerConfig()
        _validate_config(self.config)
        _validate_selections(self.selections)
        self.reset()

    def reset(self) -> None:
        self._phase = Order8NaturalContactPhase.RESET
        self._phase_start_time_s: float | None = None
        self._feedback: NaturalContactPlannerFeedback | None = None
        self._transitions: list[NaturalContactPlannerTransition] = []
        self._failure_reason: str | None = None

    @property
    def phase(self) -> Order8NaturalContactPhase:
        return self._phase

    @property
    def transitions(self) -> tuple[NaturalContactPlannerTransition, ...]:
        return tuple(self._transitions)

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def observe(self, feedback: NaturalContactPlannerFeedback) -> None:
        _validate_feedback(feedback, self.selections)
        if self._feedback is not None and feedback.time_s < self._feedback.time_s:
            raise SchemaValidationError("Order8 planner feedback time cannot move backwards")
        if self._phase_start_time_s is None:
            self._phase_start_time_s = float(feedback.time_s)
        self._feedback = feedback
        if self._phase in {
            Order8NaturalContactPhase.COMPLETE,
            Order8NaturalContactPhase.SAFE_HOLD,
        }:
            return
        timeout_s = (
            self.config.contact_acquisition_timeout_s
            if self._phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            else self.config.phase_timeout_s
        )
        if self._phase_elapsed(feedback.time_s) > timeout_s:
            self._enter_safe_hold(feedback.time_s, f"phase_timeout:{self._phase.value}")
            return

        next_phase: Order8NaturalContactPhase | None = None
        reason = ""
        if self._phase == Order8NaturalContactPhase.RESET and feedback.hover_ready:
            next_phase, reason = Order8NaturalContactPhase.APPROACH, "hover_ready"
        elif (
            self._phase == Order8NaturalContactPhase.APPROACH
            and feedback.simultaneous_reachability_passed
            and feedback.pregrasp_aligned
        ):
            next_phase, reason = (
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                "simultaneous_reachability_and_prealign",
            )
        elif (
            self._phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and feedback.contact_command_dwell_complete
        ):
            next_phase, reason = (
                Order8NaturalContactPhase.LIFT,
                "nonprivileged_contact_command_dwell_complete",
            )
        elif (
            self._phase == Order8NaturalContactPhase.LIFT
            and feedback.lift_clearance_reached
        ):
            next_phase, reason = Order8NaturalContactPhase.TRANSPORT, "lift_clearance_acquired"
        elif (
            self._phase == Order8NaturalContactPhase.TRANSPORT
            and feedback.transport_distance_reached
        ):
            next_phase, reason = Order8NaturalContactPhase.PLACE, "transport_distance_acquired"
        elif (
            self._phase == Order8NaturalContactPhase.PLACE
            and feedback.intended_place_pose_reached
        ):
            next_phase, reason = (
                Order8NaturalContactPhase.RELEASE,
                "intended_place_pose_reached",
            )
        elif (
            self._phase == Order8NaturalContactPhase.RELEASE
            and feedback.release_command_dwell_complete
        ):
            next_phase, reason = (
                Order8NaturalContactPhase.RETREAT,
                "nonprivileged_release_command_dwell_complete",
            )
        elif (
            self._phase == Order8NaturalContactPhase.RETREAT
            and feedback.retreat_clearance_reached
        ):
            next_phase, reason = Order8NaturalContactPhase.SETTLE, "retreat_clearance_acquired"
        elif (
            self._phase == Order8NaturalContactPhase.SETTLE
            and feedback.post_release_settle_complete
        ):
            next_phase, reason = Order8NaturalContactPhase.COMPLETE, "post_release_settle_acquired"
        if next_phase is not None:
            self._transition(next_phase, time_s=feedback.time_s, reason=reason)

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        feedback = self._feedback
        if feedback is None:
            raise SchemaValidationError("Order8 planner requires observe() before plan()")
        _validate_context(context, self.selections)
        phase = self._phase
        assignments = self._assignments_for_phase(
            phase,
            contact_force_scale=float(feedback.contact_force_scale),
            contact_force_scale_by_anchor_id=(
                feedback.contact_force_scale_by_anchor_id
            ),
            anchor_pose_priority_by_id=feedback.anchor_pose_priority_by_id,
        )
        body_pose = feedback.desired_body_pose_by_phase.get(phase)
        body_linear_velocity = feedback.desired_body_linear_velocity_by_phase.get(
            phase
        )
        centroidal_target = None
        if body_pose is not None:
            centroidal_target = CentroidalTarget(
                com_pos_world=tuple(float(value) for value in body_pose[:3]),
                com_vel_world=(
                    tuple(float(value) for value in body_linear_velocity)
                    if body_linear_velocity is not None
                    else None
                ),
                body_orientation_world=tuple(float(value) for value in body_pose[3:7]),
            )
        anchor_targets = {
            selection.anchor_id: feedback.desired_anchor_pose_by_id[selection.anchor_id]
            for selection in self.selections
            if phase
            in {
                Order8NaturalContactPhase.APPROACH,
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
                Order8NaturalContactPhase.RELEASE,
            }
        }
        object_targets: list[ObjectTarget] = []
        object_pose = feedback.desired_object_pose_by_phase.get(phase)
        if object_pose is not None:
            object_id = _target_object_id(context)
            object_targets.append(
                ObjectTarget(object_id=object_id, pose_target_world=object_pose)
            )
        knot = InteractionKnot(
            t_rel_s=0.0,
            contact_assignments=assignments,
            centroidal_target=centroidal_target,
            posture_target=(
                PostureTarget(free_anchor_pose_targets=anchor_targets)
                if anchor_targets
                else None
            ),
            object_targets=object_targets,
            priority_weights={
                "centroidal_pose": 1.0,
                "anchor_pose": 1.0 if anchor_targets else 0.0,
                "anchor_orientation": (
                    ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
                    if anchor_targets
                    and phase
                    in {
                        Order8NaturalContactPhase.APPROACH,
                        Order8NaturalContactPhase.CONTACT_ACQUISITION,
                        Order8NaturalContactPhase.RELEASE,
                    }
                    else 1.0 if anchor_targets else 0.0
                ),
                "contact_schedule": 1.0 if assignments else 0.0,
            },
        )
        trajectory = ContactWrenchTrajectory(
            horizon_s=self.config.horizon_s,
            dt_s=self.config.knot_dt_s,
            knots=[knot],
            derived_mode_label=(
                f"{ORDER8_DETERMINISTIC_PI_H_VERSION}:{phase.value}"
            ),
        )
        trajectory.validate()
        return trajectory

    def request_safe_hold(self, *, time_s: float, reason: str) -> None:
        """Let an external safety supervisor stop execution without policy leakage.

        Raw simulator contact truth may be used by the Order 8 acceptance/safety
        monitor to abort a rollout, but it is intentionally absent from
        :class:`NaturalContactPlannerFeedback` and therefore cannot select the
        planner's nominal actions or phase transitions.
        """

        if not math.isfinite(float(time_s)) or time_s < 0.0:
            raise SchemaValidationError("Order8 safe-hold time must be non-negative")
        if not isinstance(reason, str) or not reason:
            raise SchemaValidationError("Order8 safe-hold reason must be non-empty")
        self._enter_safe_hold(float(time_s), f"safety_supervisor:{reason}")

    def _assignments_for_phase(
        self,
        phase: Order8NaturalContactPhase,
        *,
        contact_force_scale: float,
        contact_force_scale_by_anchor_id: Mapping[int, float],
        anchor_pose_priority_by_id: Mapping[int, float],
    ) -> list[ContactAssignment]:
        if phase in {
            Order8NaturalContactPhase.RESET,
            Order8NaturalContactPhase.RETREAT,
            Order8NaturalContactPhase.SETTLE,
            Order8NaturalContactPhase.COMPLETE,
            Order8NaturalContactPhase.SAFE_HOLD,
        }:
            return []
        if phase == Order8NaturalContactPhase.APPROACH:
            schedule_state = "approach"
            force = 0.0
        elif phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            schedule_state = "attach"
            force = None
        elif phase == Order8NaturalContactPhase.RELEASE:
            schedule_state = "release"
            force = 0.0
        else:
            schedule_state = "maintain"
            force = self.config.normal_force_target_per_contact_n
        assignments: list[ContactAssignment] = []
        for selection in self.selections:
            selected_force = (
                self.config.normal_force_target_per_contact_n
                * float(
                    contact_force_scale_by_anchor_id.get(
                        selection.anchor_id,
                        contact_force_scale,
                    )
                )
                if force is None
                else force
            )
            wrench = [
                selected_force * float(selection.inward_normal_world[index])
                for index in range(3)
            ] + [0.0, 0.0, 0.0]
            assignments.append(
                ContactAssignment(
                    slot_id=selection.slot_id,
                    anchor_id=selection.anchor_id,
                    candidate_id=selection.candidate_id,
                    contact_mode=ContactMode.GRASP,
                    schedule_state=schedule_state,  # type: ignore[arg-type]
                    wrench_target=wrench,
                    wrench_lower=None,
                    wrench_upper=None,
                    priority=float(
                        anchor_pose_priority_by_id.get(
                            selection.anchor_id,
                            1.0,
                        )
                    ),
                )
            )
        return assignments

    def _phase_elapsed(self, now_s: float) -> float:
        start = self._phase_start_time_s
        return 0.0 if start is None else max(0.0, float(now_s) - start)

    def _transition(
        self,
        next_phase: Order8NaturalContactPhase,
        *,
        time_s: float,
        reason: str,
    ) -> None:
        previous = self._phase
        self._phase = next_phase
        self._phase_start_time_s = float(time_s)
        self._transitions.append(
            NaturalContactPlannerTransition(previous, next_phase, float(time_s), reason)
        )

    def _enter_safe_hold(self, time_s: float, reason: str) -> None:
        if self._phase == Order8NaturalContactPhase.SAFE_HOLD:
            return
        self._failure_reason = reason
        self._transition(
            Order8NaturalContactPhase.SAFE_HOLD,
            time_s=time_s,
            reason=reason,
        )


def _validate_config(config: NaturalContactPlannerConfig) -> None:
    for name in (
        "horizon_s",
        "knot_dt_s",
        "phase_timeout_s",
        "contact_acquisition_timeout_s",
        "normal_force_target_per_contact_n",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise SchemaValidationError(f"NaturalContactPlannerConfig.{name} must be positive")
    if config.knot_dt_s > config.horizon_s:
        raise SchemaValidationError("Order8 planner knot_dt_s must not exceed horizon_s")


def _validate_selections(
    selections: Sequence[NaturalContactAnchorSelection],
) -> None:
    if len(selections) < 2:
        raise SchemaValidationError("Order8 planner requires at least two selected anchors")
    # Multiple anchors may satisfy one ContactSlot with min_count_group > 1;
    # slot ids therefore need not be distinct.  Physical anchors, candidates,
    # and selected Dock links must remain individually identifiable.
    for field_name in ("anchor_id", "candidate_id", "dock_link_id"):
        values = [getattr(selection, field_name) for selection in selections]
        if len(values) != len(set(values)):
            raise SchemaValidationError(f"Order8 selections require distinct {field_name} values")
    for selection in selections:
        if min(selection.anchor_id, selection.slot_id, selection.candidate_id) < 0:
            raise SchemaValidationError("Order8 selection ids must be non-negative")
        if not selection.dock_link_id:
            raise SchemaValidationError("Order8 selected Dock link id must be non-empty")
        norm = math.sqrt(sum(float(value) ** 2 for value in selection.inward_normal_world))
        if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1.0e-6):
            raise SchemaValidationError("Order8 selected inward normals must be unit vectors")


def _validate_feedback(
    feedback: NaturalContactPlannerFeedback,
    selections: Sequence[NaturalContactAnchorSelection],
) -> None:
    if not math.isfinite(float(feedback.time_s)) or feedback.time_s < 0.0:
        raise SchemaValidationError("Order8 planner feedback time must be non-negative")
    for name in (
        "hover_ready",
        "simultaneous_reachability_passed",
        "pregrasp_aligned",
        "contact_command_dwell_complete",
        "lift_clearance_reached",
        "transport_distance_reached",
        "intended_place_pose_reached",
        "release_command_dwell_complete",
        "retreat_clearance_reached",
        "post_release_settle_complete",
    ):
        if type(getattr(feedback, name)) is not bool:
            raise SchemaValidationError(f"Order8 planner feedback {name} must be bool")
    missing = {
        selection.anchor_id for selection in selections
    } - set(feedback.desired_anchor_pose_by_id)
    if missing:
        raise SchemaValidationError(f"Order8 feedback missing anchor targets: {sorted(missing)}")
    for phase, velocity in feedback.desired_body_linear_velocity_by_phase.items():
        if phase not in feedback.desired_body_pose_by_phase:
            raise SchemaValidationError(
                "Order8 body velocity target requires a pose target for the same phase"
            )
        if (
            not isinstance(velocity, Sequence)
            or isinstance(velocity, (str, bytes))
            or len(velocity) != 3
            or any(not math.isfinite(float(value)) for value in velocity)
        ):
            raise SchemaValidationError(
                "Order8 body linear velocity target must be a finite Vector3"
            )
    if (
        not math.isfinite(float(feedback.contact_force_scale))
        or not 0.0 <= float(feedback.contact_force_scale) <= 1.0
    ):
        raise SchemaValidationError(
            "Order8 contact force scale must be finite and in [0, 1]"
        )
    scale_by_anchor = feedback.contact_force_scale_by_anchor_id
    if scale_by_anchor:
        expected_anchor_ids = {selection.anchor_id for selection in selections}
        if set(scale_by_anchor) != expected_anchor_ids:
            raise SchemaValidationError(
                "Order8 per-anchor contact force scales must cover exactly the "
                "selected anchors"
            )
        if any(
            not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in scale_by_anchor.values()
        ):
            raise SchemaValidationError(
                "Order8 per-anchor contact force scales must be finite and in [0, 1]"
            )
    priority_by_anchor = feedback.anchor_pose_priority_by_id
    if priority_by_anchor:
        expected_anchor_ids = {selection.anchor_id for selection in selections}
        if set(priority_by_anchor) != expected_anchor_ids:
            raise SchemaValidationError(
                "Order8 anchor pose priorities must cover exactly the selected anchors"
            )
        if any(
            not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
            for value in priority_by_anchor.values()
        ):
            raise SchemaValidationError(
                "Order8 anchor pose priorities must be finite and in (0, 1]"
            )


def _validate_context(
    context: HighLevelPolicyContext,
    selections: Sequence[NaturalContactAnchorSelection],
) -> None:
    if context.contact_candidate_set.morphology_graph_id != context.morphology_graph.graph_id:
        raise SchemaValidationError("Order8 candidate set/morphology graph mismatch")
    known_anchor_ids = {anchor.anchor_id for anchor in context.morphology_graph.robot_anchors}
    known_candidate_ids = {
        candidate.candidate_id
        for candidate, enabled in zip(
            context.contact_candidate_set.candidates,
            context.contact_candidate_set.candidate_mask,
        )
        if enabled and candidate.unary_valid
    }
    for selection in selections:
        if selection.anchor_id not in known_anchor_ids:
            raise SchemaValidationError(f"Order8 selected unknown anchor {selection.anchor_id}")
        if selection.candidate_id not in known_candidate_ids:
            raise SchemaValidationError(f"Order8 selected unavailable candidate {selection.candidate_id}")


def _target_object_id(context: HighLevelPolicyContext) -> str:
    entity_ids = {
        candidate.target_entity_id
        for candidate in context.contact_candidate_set.candidates
    }
    if len(entity_ids) != 1:
        raise SchemaValidationError("Order8 candidate set must target exactly one object")
    return next(iter(entity_ids))


__all__ = [
    "ORDER8_DETERMINISTIC_PI_H_VERSION",
    "DeterministicNaturalContactPlanner",
    "NaturalContactAnchorSelection",
    "NaturalContactPlannerConfig",
    "NaturalContactPlannerFeedback",
    "NaturalContactPlannerTransition",
]
