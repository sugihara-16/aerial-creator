"""Versioned request-selection contracts; legacy CWT actions are not accepted."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.irg import IRGEdge, IRGNode
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState
from amsrr.schemas.policies import ControllerStatus

HIGH_LEVEL_REQUEST_CONTRACT = "high_level_request_planning_v1"
REQUEST_OUTCOME_STATUSES = (
    "accepted",
    "rejected",
    "no_solution_found",
    "timeout",
    "invalid_request",
    "execution_failed",
    "execution_succeeded",
)


def _finite(value: float, name: str, *, minimum: float = 0.0) -> None:
    if not math.isfinite(value) or value < minimum:
        raise SchemaValidationError(f"{name} must be finite and >= {minimum}")


@dataclass
class HighLevelRequest(SchemaBase):
    contact_group_id: str | None
    transition_id: str | None
    subgoal_id: str

    def validate(self) -> None:
        require_non_empty(self.subgoal_id, "subgoal_id")
        for name in ("contact_group_id", "transition_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise SchemaValidationError(f"{name} must be a nonempty ID or None")


@dataclass
class PreviousRequestOutcome(SchemaBase):
    decision_id: str
    observed_time_s: float
    status: str

    def validate(self) -> None:
        require_non_empty(self.decision_id, "previous decision ID")
        _finite(self.observed_time_s, "previous outcome time")
        if self.status not in REQUEST_OUTCOME_STATUSES:
            raise SchemaValidationError("invalid previous request outcome")


@dataclass
class ContactLoadEstimate(SchemaBase):
    """Deployable estimator output, never an arbitrary ContactState.metadata dict."""

    candidate_id: int
    time_s: float
    signed_distance_m: float
    normal_load_n: float
    slip_speed_mps: float
    confidence: float
    source: Literal["fk_object_estimate_motor_current"] = (
        "fk_object_estimate_motor_current"
    )

    def validate(self) -> None:
        if self.candidate_id < 0 or self.source != "fk_object_estimate_motor_current":
            raise SchemaValidationError("invalid contact estimator identity")
        _finite(self.time_s, "contact time")
        _finite(self.slip_speed_mps, "slip speed")
        if not all(
            math.isfinite(x) for x in (self.signed_distance_m, self.normal_load_n)
        ):
            raise SchemaValidationError("contact estimate must be finite")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise SchemaValidationError("contact confidence must be in [0, 1]")


@dataclass
class ContactMotionEstimate(SchemaBase):
    """Observed geometry and motor torque; no inferred contact wrench."""

    candidate_id: int
    time_s: float
    signed_distance_m: float
    slip_speed_mps: float
    motor_load_nm: float
    velocity_valid: bool
    source: Literal["observed_pose_fk_motor_v1"] = "observed_pose_fk_motor_v1"

    def validate(self) -> None:
        if self.candidate_id < 0 or self.source != "observed_pose_fk_motor_v1":
            raise SchemaValidationError("invalid contact motion identity")
        _finite(self.time_s, "contact motion time")
        _finite(self.slip_speed_mps, "contact motion speed")
        _finite(self.motor_load_nm, "contact motor torque")
        if not math.isfinite(self.signed_distance_m) or not isinstance(self.velocity_valid, bool):
            raise SchemaValidationError("invalid contact motion observation")


@dataclass
class ExecutionGuardSample(SchemaBase):
    """Named, timestamped predicate from a deployable deterministic observer."""

    guard_id: str
    time_s: float
    satisfied: bool
    confidence: float
    observer_version: str
    held_for_s: float = 0.0

    def validate(self) -> None:
        require_non_empty(self.guard_id, "guard_id")
        require_non_empty(self.observer_version, "observer_version")
        _finite(self.time_s, "guard time")
        _finite(self.held_for_s, "guard duration")
        if not isinstance(self.satisfied, bool) or not 0 <= self.confidence <= 1:
            raise SchemaValidationError("invalid guard result")


@dataclass
class ActiveExecutionState(SchemaBase):
    phase_id: int
    phase_started_s: float
    contact_group_id: str | None = None
    # Stable semantic bindings: candidate -> (slot, anchor, target entity).
    contact_bindings: dict[int, tuple[int, int, str]] = field(default_factory=dict)
    plan_id: str | None = None
    revision: int = 0
    contact_surface_hashes: dict[int, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.phase_id < 0 or self.revision < 0:
            raise SchemaValidationError("invalid execution phase/revision")
        _finite(self.phase_started_s, "phase start")
        if self.contact_bindings and not self.contact_group_id:
            raise SchemaValidationError("active contacts require their group identity")
        if set(self.contact_surface_hashes) - set(self.contact_bindings):
            raise SchemaValidationError("surface identity has no active binding")


@dataclass
class HighLevelObservation(SchemaBase):
    """Whitelist excludes raw contacts, teacher phase, success and future metrics."""

    time_s: float
    module_states: list[ModuleRuntimeState]
    object_states: list[ObjectRuntimeState]
    controller_status: ControllerStatus
    contact_estimates: list[ContactLoadEstimate] = field(default_factory=list)
    guard_samples: list[ExecutionGuardSample] = field(default_factory=list)
    contact_motion: list[ContactMotionEstimate] = field(default_factory=list)

    def validate(self) -> None:
        _finite(self.time_s, "observation time")
        for values, key in (
            (self.module_states, "module_id"),
            (self.object_states, "object_id"),
            (self.contact_estimates, "candidate_id"),
            (self.guard_samples, "guard_id"),
            (self.contact_motion, "candidate_id"),
        ):
            ids = [getattr(x, key) for x in values]
            if len(ids) != len(set(ids)):
                raise SchemaValidationError(f"duplicate {key}")
        for item in (*self.contact_estimates, *self.guard_samples, *self.contact_motion):
            item.validate()
            if item.time_s > self.time_s:
                raise SchemaValidationError("future estimator result in observation")

        # Existing runtime classes only check shapes; reject nonfinite payloads here.
        def visit(value: object) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise SchemaValidationError("nonfinite high-level observation")
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)

        visit(self.to_dict())


@dataclass
class RequestCatalogEntry(SchemaBase):
    request: HighLevelRequest
    phase_id: int
    candidate_ids: list[int]
    target_nodes: list[IRGNode]
    transition: IRGEdge | None
    phase_type: str
    # Empty guards do NOT authorize a transition.
    required_guard_ids: list[str]
    heuristic_score: float

    def validate(self) -> None:
        self.request.validate()
        if self.phase_id < 0 or len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise SchemaValidationError("invalid request phase/candidates")
        if bool(self.candidate_ids) != (self.request.contact_group_id is not None):
            raise SchemaValidationError("contact-free request/group mismatch")
        if (self.transition is None) != (self.request.transition_id is None):
            raise SchemaValidationError("transition ID/edge mismatch")
        if self.transition is not None and self.transition.dst_id != self.phase_id:
            raise SchemaValidationError("transition destination mismatch")
        if not math.isfinite(self.heuristic_score):
            raise SchemaValidationError("nonfinite request score")


@dataclass
class HighLevelRequestCatalog(SchemaBase):
    snapshot_hash: str
    entries: list[RequestCatalogEntry]
    contract_version: str = HIGH_LEVEL_REQUEST_CONTRACT

    def validate(self) -> None:
        if self.contract_version != HIGH_LEVEL_REQUEST_CONTRACT:
            raise SchemaValidationError("unsupported high-level request contract")
        require_non_empty(self.snapshot_hash, "snapshot_hash")
        keys = [x.request.stable_hash() for x in self.entries]
        if len(keys) != len(set(keys)):
            raise SchemaValidationError("duplicate request tuple")
        for entry in self.entries:
            entry.validate()

    def resolve(self, request: HighLevelRequest) -> RequestCatalogEntry:
        request.validate()
        key = request.stable_hash()
        for entry in self.entries:
            if entry.request.stable_hash() == key:
                return entry
        raise SchemaValidationError("request tuple is absent from this catalog")


@dataclass
class HighLevelDecisionRecord(SchemaBase):
    observation: HighLevelObservation
    execution_state: ActiveExecutionState
    catalog: HighLevelRequestCatalog
    ranked_requests: list[HighLevelRequest]
    policy_version: str
    contract_version: str = HIGH_LEVEL_REQUEST_CONTRACT
    ranking_scores: list[float] | None = None

    def validate(self) -> None:
        if self.contract_version != HIGH_LEVEL_REQUEST_CONTRACT:
            raise SchemaValidationError(
                "legacy trajectory decision is not a request decision"
            )
        require_non_empty(self.policy_version, "policy_version")
        keys = [x.stable_hash() for x in self.ranked_requests]
        if len(keys) != len(set(keys)):
            raise SchemaValidationError("ranking contains duplicate requests")
        if self.ranking_scores is not None and (
            len(self.ranking_scores) != len(keys)
            or any(not math.isfinite(s) for s in self.ranking_scores)
        ):
            raise SchemaValidationError("invalid archived ranking scores")
        for request in self.ranked_requests:
            self.catalog.resolve(request)


@dataclass
class GeneratedExecutionPlan(SchemaBase):
    request: HighLevelRequest
    snapshot_hash: str
    model_hash: str
    planner_version: str
    planner_config_hash: str
    observation_time_s: float
    valid_until_s: float
    trajectory: ContactWrenchTrajectory
    # Canonical execution interpolation must also be used by C_H.
    interpolation: Literal["joint_quintic_fk_v1"] = "joint_quintic_fk_v1"
    contract_version: str = HIGH_LEVEL_REQUEST_CONTRACT
    solver_constraint_margins: dict[str, list[float]] = field(default_factory=dict)

    def validate(self) -> None:
        if (
            self.contract_version != HIGH_LEVEL_REQUEST_CONTRACT
            or self.interpolation != "joint_quintic_fk_v1"
        ):
            raise SchemaValidationError(
                "unsupported execution plan contract/interpolation"
            )
        for name in (
            "snapshot_hash",
            "model_hash",
            "planner_version",
            "planner_config_hash",
        ):
            require_non_empty(getattr(self, name), name)
        _finite(self.observation_time_s, "plan observation time")
        _finite(self.valid_until_s, "plan validity")
        for values in self.solver_constraint_margins.values():
            if not values or any(not math.isfinite(v) for v in values):
                raise SchemaValidationError("empty/nonfinite solver constraint record")
        self.trajectory.validate()
        if (
            self.valid_until_s <= self.observation_time_s
            or self.valid_until_s > self.observation_time_s + self.trajectory.horizon_s
        ):
            raise SchemaValidationError("invalid plan validity interval")
        if len(self.trajectory.knots) < 2:
            raise SchemaValidationError("plan requires an execution interval")
        times = [k.t_rel_s for k in self.trajectory.knots]
        if any(not math.isfinite(t) for t in times):
            raise SchemaValidationError("nonfinite plan knot time")
        if (
            times[0] != 0
            or times[-1] != self.trajectory.horizon_s
            or any(b <= a for a, b in zip(times, times[1:]))
        ):
            raise SchemaValidationError("plan knots must span the complete horizon")
        joint_ids = None
        object_ids = None
        for knot in self.trajectory.knots:
            p = knot.posture_target
            if p is None or p.joint_pos_target is None or p.joint_vel_target is None:
                raise SchemaValidationError(
                    "execution plan requires solver-produced joints"
                )
            keys = set(p.joint_pos_target)
            if set(p.joint_vel_target) != keys or (
                joint_ids is not None and keys != joint_ids
            ):
                raise SchemaValidationError("inconsistent plan joint set")
            joint_ids = keys
            if any(
                not math.isfinite(v)
                for v in (*p.joint_pos_target.values(), *p.joint_vel_target.values())
            ):
                raise SchemaValidationError("nonfinite planned joint value")
            if any(abs(v) > 1.0e-12 for v in p.joint_vel_target.values()):
                raise SchemaValidationError(
                    "quintic-rest plan requires zero knot velocity"
                )
            c = knot.centroidal_target
            if c is None or c.com_pos_world is None or c.body_orientation_world is None:
                raise SchemaValidationError("plan requires complete centroidal target")
            if not all(
                math.isfinite(x) for x in (*c.com_pos_world, *c.body_orientation_world)
            ):
                raise SchemaValidationError("nonfinite centroidal target")
            if abs(sum(x * x for x in c.body_orientation_world) - 1.0) > 1.0e-6:
                raise SchemaValidationError("plan quaternion must be normalized")
            if (
                c.com_vel_world is None
                or len(c.com_vel_world) != 3
                or any(
                    not math.isfinite(v) or abs(v) > 1.0e-12 for v in c.com_vel_world
                )
            ):
                raise SchemaValidationError(
                    "quintic-rest plan requires zero knot CoM velocity"
                )
            ids = [o.object_id for o in knot.object_targets]
            if len(ids) != len(set(ids)) or (
                object_ids is not None and set(ids) != object_ids
            ):
                raise SchemaValidationError("inconsistent plan object set")
            object_ids = set(ids)
            for obj in knot.object_targets:
                if (
                    obj.twist_target_world is None
                    or len(obj.twist_target_world) != 6
                    or any(
                        not math.isfinite(v) or abs(v) > 1.0e-12
                        for v in obj.twist_target_world
                    )
                ):
                    raise SchemaValidationError(
                        "quintic-rest plan requires zero knot object twist"
                    )
                if obj.pose_target_world is None or not all(
                    math.isfinite(x) for x in obj.pose_target_world
                ):
                    raise SchemaValidationError("plan requires finite object poses")
                if abs(sum(x * x for x in obj.pose_target_world[3:]) - 1.0) > 1.0e-6:
                    raise SchemaValidationError("object quaternion must be normalized")
            for assignment in knot.contact_assignments:
                assignment.validate()


@dataclass
class PlanCheckRecord(SchemaBase):
    plan_hash: str
    snapshot_hash: str
    accepted: bool
    checker_version: str
    violation_codes: list[str] = field(default_factory=list)
    margins: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("plan_hash", "snapshot_hash", "checker_version"):
            require_non_empty(getattr(self, name), name)
        if self.accepted and self.violation_codes:
            raise SchemaValidationError("accepted plan cannot contain hard violations")
        if any(not math.isfinite(x) for x in self.margins.values()):
            raise SchemaValidationError("nonfinite check margin")


@dataclass
class RequestRankingLabel(SchemaBase):
    """Post-execution supervision, stored separately from the decision inputs."""

    request: HighLevelRequest
    decision_snapshot_hash: str
    checked_plan_hash: str
    execution_outcome: Literal[
        "first_choice_success",
        "searched_candidate_success",
        "fallback_success",
        "failed",
    ]

    def validate(self) -> None:
        self.request.validate()
        require_non_empty(self.decision_snapshot_hash, "label decision snapshot")
        require_non_empty(self.checked_plan_hash, "label checked plan identity")
        if self.execution_outcome not in {
            "first_choice_success",
            "searched_candidate_success",
            "fallback_success",
            "failed",
        }:
            raise SchemaValidationError("invalid request execution outcome")
