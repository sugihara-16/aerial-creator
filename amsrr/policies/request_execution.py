"""Executor-owned transitions and nominal references for checked joint plans.

This module produces the input to the nominal controller. Contact/load feedback
and actuator allocation remain controller responsibilities; no pi_L or IK is
called here. A returned reference is not a task-success observation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.policies.high_level_requests import (
    HighLevelDecisionContext,
    contact_surface_hash,
)
from amsrr.policies.request_trajectory_planner import JointPlanEvaluator
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import (
    ActiveExecutionState,
    GeneratedExecutionPlan,
    PlanCheckRecord,
)
from amsrr.schemas.policies import PolicyCommand, POLICY_COMMAND_CONTRACT_CENTROIDAL
from amsrr.schemas.irg import IRGNodeType, IRGEdgeType


@dataclass(frozen=True)
class RequestExecutionConfig:
    maximum_observation_age_s: float = 0.1
    minimum_guard_confidence: float = 0.95
    minimum_guard_dwell_s: float = 0.1
    maximum_joint_tracking_error_rad: float = 0.03
    maximum_position_tracking_error_m: float = 0.005
    maximum_orientation_tracking_error_rad: float = 0.03
    minimum_contact_load_n: float = 0.1
    maximum_slip_speed_mps: float = 0.01
    maximum_contact_distance_m: float = 0.005

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in self.__dict__.values()):
            raise ValueError("execution limits must be finite and positive")
        if self.minimum_guard_confidence > 1:
            raise ValueError("invalid guard confidence")


def guards_ready(entry, observation, config: RequestExecutionConfig) -> bool:
    if entry.transition is None:
        return True
    guards = {g.guard_id: g for g in observation.guard_samples}
    return bool(entry.required_guard_ids) and all(
        gid in guards
        and guards[gid].satisfied
        and 0
        <= observation.time_s - guards[gid].time_s
        <= config.maximum_observation_age_s
        and guards[gid].confidence >= config.minimum_guard_confidence
        and guards[gid].held_for_s >= config.minimum_guard_dwell_s
        for gid in entry.required_guard_ids
    )


def initial_execution_state(
    irg, observation, config: RequestExecutionConfig | None = None
) -> ActiveExecutionState:
    """Initialize the unique declared IRG start; never copy task_progress.phase."""
    config = config or RequestExecutionConfig()
    observation.validate()
    phases = {n.node_id: n for n in irg.nodes if n.node_type == IRGNodeType.PHASE}
    incoming = {
        e.dst_id
        for e in irg.edges
        if e.edge_type
        in {
            IRGEdgeType.TEMPORAL_NEXT,
            IRGEdgeType.GUARD_TRANSITION,
            IRGEdgeType.FALLBACK,
        }
    }
    starts = [n for nid, n in phases.items() if nid not in incoming]
    if len(starts) != 1:
        raise SchemaValidationError("IRG needs one unambiguous initial phase")
    phase = starts[0]
    condition = phase.feature.get("entry_condition")
    if condition is not None:
        if not isinstance(condition, dict) or set(condition) != {"guard_id"}:
            raise SchemaValidationError("initial phase requires a declared guard_id")
        sample = next(
            (
                g
                for g in observation.guard_samples
                if g.guard_id == condition["guard_id"]
            ),
            None,
        )
        if (
            sample is None
            or not sample.satisfied
            or sample.confidence < config.minimum_guard_confidence
            or sample.held_for_s < config.minimum_guard_dwell_s
            or not 0
            <= observation.time_s - sample.time_s
            <= config.maximum_observation_age_s
        ):
            raise SchemaValidationError("initial phase entry guard not satisfied")
    return ActiveExecutionState(phase.node_id, observation.time_s)


class RequestPlanExecutor:
    """Install only an unchanged accepted plan; fail closed on stale/unsafe state."""

    def __init__(
        self, state: ActiveExecutionState, config: RequestExecutionConfig | None = None
    ):
        self.state = deepcopy(state)
        self.config = config or RequestExecutionConfig()
        self._plan: GeneratedExecutionPlan | None = None
        self._context: HighLevelDecisionContext | None = None
        self._evaluator: JointPlanEvaluator | None = None
        self._static_hashes: tuple[str, ...] = ()

    def install(
        self,
        plan: GeneratedExecutionPlan,
        check: PlanCheckRecord,
        origin: HighLevelDecisionContext,
        current: HighLevelDecisionContext,
    ) -> None:
        origin.validate_snapshot()
        current.validate_snapshot()
        plan.validate()
        check.validate()
        if (
            not check.accepted
            or check.plan_hash != plan.stable_hash()
            or check.snapshot_hash != origin.catalog.snapshot_hash
        ):
            raise SchemaValidationError("plan has no matching independent acceptance")
        if plan.snapshot_hash != origin.catalog.snapshot_hash:
            raise SchemaValidationError("plan snapshot mismatch")
        if (
            self.state.stable_hash() != origin.execution_state.stable_hash()
            or self.state.stable_hash() != current.execution_state.stable_hash()
        ):
            raise SchemaValidationError("execution state changed while planning")
        self._check_static_inputs(origin, current)
        entry = origin.catalog.resolve(plan.request)
        if not guards_ready(entry, current.observation, self.config):
            raise SchemaValidationError("requested transition guard is not satisfied")
        evaluator = JointPlanEvaluator(origin)
        knot = evaluator.sample(plan, time_s=current.observation.time_s)
        self._check_tracking(knot, current)
        self._check_contacts(knot, current, self.state.contact_bindings)
        # Only this executor commits the phase, after acceptance and observed guards.
        state = deepcopy(self.state)
        if entry.transition is not None:
            state.phase_id = entry.phase_id
            state.phase_started_s = current.observation.time_s
        state.plan_id = check.plan_hash
        state.revision += 1
        self.state = state
        self._plan, self._context, self._evaluator = (
            deepcopy(plan),
            deepcopy(origin),
            None,
        )
        self._evaluator = JointPlanEvaluator(self._context)
        self._static_hashes = self._static_input_hashes(self._context)

    def command(
        self, current: HighLevelDecisionContext, *, time_s: float
    ) -> PolicyCommand:
        if self._plan is None or self._context is None or self._evaluator is None:
            raise SchemaValidationError(
                "no independently checked plan; recovery is required"
            )
        current.validate_snapshot()
        if self.state.stable_hash() != current.execution_state.stable_hash():
            raise SchemaValidationError("observation has a different execution state")
        if self._plan.stable_hash() != self.state.plan_id:
            raise SchemaValidationError("installed plan was modified")
        if self._static_hashes != self._static_input_hashes(current):
            raise SchemaValidationError("model/geometry/task changed; replan required")
        age = time_s - current.observation.time_s
        if (
            not math.isfinite(age)
            or not 0 <= age <= self.config.maximum_observation_age_s
        ):
            raise SchemaValidationError(
                "stale control observation; recovery is required"
            )
        knot = self._evaluator.sample(self._plan, time_s=time_s)
        observed_knot = (
            knot
            if time_s == current.observation.time_s
            else self._evaluator.sample(self._plan, time_s=current.observation.time_s)
        )
        self._check_tracking(observed_knot, current)
        self._check_contacts(knot, current, self.state.contact_bindings)
        # Contact bindings are established from estimates, never future schedule alone.
        candidates = {
            c.candidate_id: c
            for c in self._context.scene.contact_candidate_set.candidates
        }
        for a in observed_knot.contact_assignments:
            if a.schedule_state in {"attach", "maintain", "slide"}:
                c = candidates[a.candidate_id]
                self.state.contact_bindings[a.candidate_id] = (
                    c.slot_id,
                    c.anchor_id,
                    c.target_entity_id,
                )
                self.state.contact_surface_hashes[a.candidate_id] = (
                    contact_surface_hash(c, self._context.observation)
                )
                self.state.contact_group_id = self._plan.request.contact_group_id
        # Binding removal requires an explicit observed release guard.
        guards = {g.guard_id: g for g in current.observation.guard_samples}
        for a in observed_knot.contact_assignments:
            g = guards.get(f"contact:{a.candidate_id}:released")
            if (
                a.schedule_state == "release"
                and g
                and g.satisfied
                and g.confidence >= self.config.minimum_guard_confidence
                and g.held_for_s >= self.config.minimum_guard_dwell_s
                and 0
                <= current.observation.time_s - g.time_s
                <= self.config.maximum_observation_age_s
            ):
                self.state.contact_bindings.pop(a.candidate_id, None)
                self.state.contact_surface_hashes.pop(a.candidate_id, None)
        if not self.state.contact_bindings:
            self.state.contact_group_id = None
        c = knot.centroidal_target
        return PolicyCommand(
            desired_body_pose=(*c.com_pos_world, *c.body_orientation_world),
            desired_body_twist=[
                *c.com_vel_world,
                *self._evaluator.body_angular_velocity(self._plan, time_s=time_s),
            ],
            joint_position_targets=dict(knot.posture_target.joint_pos_target),
            joint_velocity_targets=dict(knot.posture_target.joint_vel_target),
            priority_weights=dict(knot.priority_weights),
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        )

    @staticmethod
    def _check_static_inputs(origin, current):
        if RequestPlanExecutor._static_input_hashes(
            origin
        ) != RequestPlanExecutor._static_input_hashes(current):
            raise SchemaValidationError("model/geometry/task changed; replan required")

    @staticmethod
    def _static_input_hashes(context):
        return tuple(
            value.stable_hash()
            for value in (
                context.physical_model,
                context.task_spec,
                context.scene.irg,
                context.scene.interaction_envelope,
                context.scene.morphology_graph,
            )
        )

    def _check_tracking(self, knot, current):
        status = current.observation.controller_status
        if not status.qp_feasible or status.status != "ok":
            raise SchemaValidationError("controller is unhealthy; recovery is required")
        q = {
            f"module_{m.module_id}:{key}": value
            for m in current.observation.module_states
            for key, value in m.joint_positions.items()
        }
        for key, target in knot.posture_target.joint_pos_target.items():
            if (
                key not in q
                or abs(q[key] - target) > self.config.maximum_joint_tracking_error_rad
            ):
                raise SchemaValidationError("joint tracking error; replan required")
        rigid = RigidBodyControlModelBuilder().build(
            current.scene.morphology_graph,
            current.physical_model,
            current.scene.runtime_observation,
        )
        target = (
            *knot.centroidal_target.com_pos_world,
            *knot.centroidal_target.body_orientation_world,
        )
        self._check_pose(rigid.body_pose_world, target)
        objects = {o.object_id: o for o in current.observation.object_states}
        for target in knot.object_targets:
            if target.object_id not in objects:
                raise SchemaValidationError("missing object observation")
            self._check_pose(
                objects[target.object_id].pose_world, target.pose_target_world
            )

    def _check_pose(self, actual, target):
        if (
            np.linalg.norm(np.array(actual[:3]) - target[:3])
            > self.config.maximum_position_tracking_error_m
        ):
            raise SchemaValidationError("position tracking error; replan required")
        if (
            Rotation.from_quat(actual[3:]).inv() * Rotation.from_quat(target[3:])
        ).magnitude() > self.config.maximum_orientation_tracking_error_rad:
            raise SchemaValidationError("orientation tracking error; replan required")

    def _check_contacts(self, knot, current, bindings):
        estimates = {e.candidate_id: e for e in current.observation.contact_estimates}
        guards = {g.guard_id: g for g in current.observation.guard_samples}
        required = set(bindings) | {
            a.candidate_id
            for a in knot.contact_assignments
            if a.schedule_state in {"maintain", "slide"}
        }
        # Attach begins at observed touch. It cannot serve as evidence for lift.
        required |= {
            a.candidate_id
            for a in knot.contact_assignments
            if a.schedule_state == "attach"
        }
        for a in knot.contact_assignments:
            g = guards.get(f"contact:{a.candidate_id}:released")
            if (
                a.schedule_state == "release"
                and g
                and g.satisfied
                and g.confidence >= self.config.minimum_guard_confidence
                and g.held_for_s >= self.config.minimum_guard_dwell_s
                and 0
                <= current.observation.time_s - g.time_s
                <= self.config.maximum_observation_age_s
            ):
                required.discard(a.candidate_id)
        for cid in required:
            e = estimates.get(cid)
            if (
                e is None
                or not 0
                <= current.observation.time_s - e.time_s
                <= self.config.maximum_observation_age_s
                or e.confidence < self.config.minimum_guard_confidence
                or e.normal_load_n < self.config.minimum_contact_load_n
                or abs(e.signed_distance_m) > self.config.maximum_contact_distance_m
                or e.slip_speed_mps > self.config.maximum_slip_speed_mps
            ):
                raise SchemaValidationError(
                    "contact support unconfirmed/lost; replan required"
                )
