"""Independent validation of the SAME joint-authoritative generated plan."""

from __future__ import annotations

import math
from copy import deepcopy
from time import monotonic
import numpy as np

from amsrr.feasibility.articulated_reachability import _global_joint_limits
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_requests import HighLevelDecisionContext
from amsrr.policies.request_trajectory_planner import (
    JointPlanEvaluator,
    REQUIRED_PLANNING_CONSTRAINTS,
)
from amsrr.robot_model.whole_structure_kinematics import ordered_global_dock_joint_ids
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import GeneratedExecutionPlan, PlanCheckRecord
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactWrenchTrajectory,
)


class RequestPlanChecker:
    """A production C_H is required; proxy checkers/resolvers cannot be smuggled in.

    The supplied production physics evaluator retains its existing shadow gate.
    Sampling here does not authorize replacing shadow by an unvalidated profile.
    """

    checker_version = "same_joint_plan_c_h_v1"

    def __init__(
        self,
        checker: ContactWrenchTrajectoryFeasibilityChecker,
        *,
        sample_dt_s: float,
        maximum_samples: int,
        maximum_joint_acceleration_radps2: float,
        constraint_evaluator=None,
    ):
        if type(checker) is not ContactWrenchTrajectoryFeasibilityChecker:
            raise ValueError(
                "request C_H requires an independent checker, never a posture resolver"
            )
        cfg = checker.config
        if cfg.evaluation_mode != "production" or not all(
            (
                cfg.require_qp_evaluation,
                cfg.require_collision_evaluation,
                cfg.require_wrench_evaluation,
                cfg.require_reachability_evaluation,
            )
        ):
            raise ValueError(
                "request C_H requires all production physics and reachability checks"
            )
        if checker.physics_evaluator is None or checker.reachability_evaluator is None:
            raise ValueError(
                "request C_H requires physical and reachability evaluators"
            )
        # The existing production evaluator is hybrid QP + independent shadow.
        from amsrr.feasibility.contact_wrench_hybrid import (
            HybridContactWrenchPhysicsEvaluator,
        )

        if not isinstance(
            checker.physics_evaluator, HybridContactWrenchPhysicsEvaluator
        ):
            raise ValueError(
                "online validation has no approved shadow-backed physics evaluator"
            )
        if not math.isfinite(sample_dt_s) or sample_dt_s <= 0 or maximum_samples < 2:
            raise ValueError("invalid independent plan sampling budget")
        if (
            not math.isfinite(maximum_joint_acceleration_radps2)
            or maximum_joint_acceleration_radps2 <= 0
        ):
            raise ValueError("joint acceleration limit is required")
        self.checker = checker
        self.sample_dt_s = sample_dt_s
        self.maximum_samples = maximum_samples
        self.maximum_joint_acceleration_radps2 = maximum_joint_acceleration_radps2
        # Legacy C_H alone does not check the entire new planning contract
        # (notably joint effort, uncertainty and the remaining task). Missing
        # independent model checks reject, even when its shadow gate passes.
        self.constraint_evaluator = constraint_evaluator

    def check(
        self,
        plan: GeneratedExecutionPlan,
        context: HighLevelDecisionContext,
        *,
        deadline: float,
    ) -> PlanCheckRecord:
        original_hash = plan.stable_hash()
        violations = []
        margins = {}
        try:
            context.validate_snapshot()
            plan.validate()
            entry = context.catalog.resolve(plan.request)
            if (
                plan.snapshot_hash != context.catalog.snapshot_hash
                or plan.model_hash != context.physical_model.stable_hash()
            ):
                raise SchemaValidationError("plan input identity mismatch")
            if plan.observation_time_s != context.observation.time_s:
                raise SchemaValidationError("plan observation origin mismatch")
            if (
                plan.trajectory.contract_version
                != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
            ):
                raise SchemaValidationError(
                    "new plan requires explicit contact-frame wrench semantics"
                )
            allowed = {
                c.candidate_id: c
                for c in context.scene.contact_candidate_set.candidates
                if c.candidate_id in entry.candidate_ids
            }
            phase = next(
                n for n in context.scene.irg.nodes if n.node_id == entry.phase_id
            )
            phase_label = phase.feature.get("phase_label")
            if not phase_label:
                raise SchemaValidationError(
                    "existing C_H requires a declared IRG phase label"
                )
            for knot in plan.trajectory.knots:
                for guard in knot.guard_conditions:
                    if (
                        guard.get("type") == "order9_task_phase"
                        and guard.get("phase_label") != phase_label
                    ):
                        raise SchemaValidationError(
                            "plan phase metadata differs from requested IRG phase"
                        )
                if {a.candidate_id for a in knot.contact_assignments} != set(allowed):
                    raise SchemaValidationError(
                        "generated plan omits requested contact binding"
                    )
                for a in knot.contact_assignments:
                    c = allowed.get(a.candidate_id)
                    if c is None or (a.slot_id, a.anchor_id, a.contact_mode) != (
                        c.slot_id,
                        c.anchor_id,
                        c.contact_mode,
                    ):
                        raise SchemaValidationError(
                            "generated plan changes requested contact binding"
                        )
            ids = ordered_global_dock_joint_ids(
                context.scene.morphology_graph, context.physical_model
            )
            limits = _global_joint_limits(
                context.scene.morphology_graph, context.physical_model, ids
            )
            joint_by_id = {j.joint_id: j for j in context.physical_model.joints}
            initial = plan.trajectory.knots[0]
            observed_q = {
                f"module_{m.module_id}:{key}": value
                for m in context.observation.module_states
                for key, value in m.joint_positions.items()
            }
            observed_qdot = {
                f"module_{m.module_id}:{key}": value
                for m in context.observation.module_states
                for key, value in m.joint_velocities.items()
            }
            if set(initial.posture_target.joint_pos_target) != set(ids):
                raise SchemaValidationError("plan joint IDs differ from PhysicalModel")
            if any(
                key not in observed_q
                or abs(initial.posture_target.joint_pos_target[key] - observed_q[key])
                > 1.0e-8
                for key in ids
            ):
                raise SchemaValidationError(
                    "plan initial joint state differs from observation"
                )
            if any(
                key not in observed_qdot or abs(observed_qdot[key]) > 1.0e-8
                for key in ids
            ):
                raise SchemaValidationError(
                    "quintic-rest plan requires observed zero joint velocities"
                )
            rigid = RigidBodyControlModelBuilder().build(
                context.scene.morphology_graph,
                context.physical_model,
                context.scene.runtime_observation,
            )
            if not np.allclose(
                initial.centroidal_target.com_pos_world,
                rigid.body_pose_world[:3],
                atol=1.0e-8,
                rtol=0,
            ):
                raise SchemaValidationError("plan initial CoM differs from observation")
            if np.linalg.norm(rigid.body_twist_world) > 1.0e-8:
                raise SchemaValidationError(
                    "quintic-rest plan requires observed zero body velocity"
                )
            actual_q = np.array(rigid.body_pose_world[3:])
            planned_q = np.array(initial.centroidal_target.body_orientation_world)
            if (
                min(
                    np.linalg.norm(actual_q - planned_q),
                    np.linalg.norm(actual_q + planned_q),
                )
                > 1.0e-8
            ):
                raise SchemaValidationError(
                    "plan initial body orientation differs from observation"
                )
            objects = {o.object_id: o for o in context.observation.object_states}
            if {o.object_id for o in initial.object_targets} != set(objects):
                raise SchemaValidationError("plan must include all observed objects")
            for obj in initial.object_targets:
                if (
                    not np.allclose(
                        obj.pose_target_world,
                        objects[obj.object_id].pose_world,
                        atol=1.0e-8,
                        rtol=0,
                    )
                    or np.linalg.norm(objects[obj.object_id].twist_world) > 1.0e-8
                ):
                    raise SchemaValidationError(
                        "plan initial object state differs from observation"
                    )
            for left, right in zip(plan.trajectory.knots, plan.trajectory.knots[1:]):
                dt = right.t_rel_s - left.t_rel_s
                for key in ids:
                    a, b = (
                        left.posture_target.joint_pos_target[key],
                        right.posture_target.joint_pos_target[key],
                    )
                    lo, hi = limits[key]
                    if min(a, b) < lo or max(a, b) > hi:
                        violations.append("joint_position_limit")
                    local_id = key.split(":", 1)[1]
                    velocity = joint_by_id[local_id].velocity_limit
                    if velocity is None or 1.875 * abs(b - a) / dt > velocity:
                        violations.append("joint_velocity_limit")
                    if (10 / math.sqrt(3)) * abs(
                        b - a
                    ) / dt**2 > self.maximum_joint_acceleration_radps2:
                        violations.append("joint_acceleration_limit")
            count = math.ceil(plan.trajectory.horizon_s / self.sample_dt_s) + 1
            if count > self.maximum_samples:
                raise SchemaValidationError(
                    "independent plan validation sample budget exceeded"
                )
            times = sorted(
                set(
                    np.linspace(0.0, plan.trajectory.horizon_s, count).tolist()
                    + [k.t_rel_s for k in plan.trajectory.knots]
                )
            )
            if len(times) > self.maximum_samples:
                raise SchemaValidationError(
                    "independent plan validation sample budget exceeded"
                )
            evaluator = JointPlanEvaluator(context)
            for original in plan.trajectory.knots:
                evaluated = evaluator.sample(
                    plan, time_s=plan.observation_time_s + original.t_rel_s
                )
                authored = original.posture_target.free_anchor_pose_targets or {}
                actual = evaluated.posture_target.free_anchor_pose_targets or {}
                if set(authored) != set(actual) or any(
                    not np.allclose(authored[k], actual[k], atol=1.0e-8, rtol=0)
                    for k in actual
                ):
                    raise SchemaValidationError(
                        "stored anchor pose does not match the solved joints"
                    )
            dense = []
            for t in times:
                if monotonic() >= deadline:
                    raise TimeoutError("C_H sampling deadline exceeded")
                dense.append(evaluator.sample(plan, time_s=plan.observation_time_s + t))
            trajectory = ContactWrenchTrajectory(
                plan.trajectory.horizon_s,
                self.sample_dt_s,
                dense,
                plan.trajectory.derived_mode_label,
                plan.trajectory.contract_version,
            )
            # Compatibility input for the existing C_H, after the decision only.
            # This phase comes from the validated request catalog, never a teacher.
            check_context = deepcopy(context.scene)
            check_context.runtime_observation.task_progress.phase_label = phase_label
            dense_hash = trajectory.stable_hash()
            result = self.checker.check(trajectory, check_context)
            if trajectory.stable_hash() != dense_hash:
                violations.append("checker_mutated_validation_trajectory")
            if len(result.knot_results) != len(dense):
                violations.append("incomplete_knot_validation")
            violations.extend(v.code for v in result.hard_violations)
            margins.update(result.margins)
            if not result.feasible and not violations:
                violations.append("physics_rejected")
            if self.constraint_evaluator is None:
                violations.append("new_contract_constraints_not_evaluated")
            else:
                terms = self.constraint_evaluator(context, plan, deadline=deadline)
                if set(terms) != REQUIRED_PLANNING_CONSTRAINTS:
                    raise SchemaValidationError(
                        "incomplete independent planning constraints"
                    )
                for name, values in terms.items():
                    values = np.asarray(values, dtype=float).reshape(-1)
                    if not values.size or not np.isfinite(values).all():
                        raise SchemaValidationError(
                            "empty/nonfinite independent constraint"
                        )
                    margins["request_" + name] = float(values.min())
                    if values.min() < 0:
                        violations.append("request_" + name)
            if monotonic() >= deadline:
                violations.append("validation_timeout")
            context.validate_snapshot()
            if plan.stable_hash() != original_hash:
                violations.append("checker_mutated_plan")
        except (
            SchemaValidationError,
            KeyError,
            ValueError,
            TypeError,
            TimeoutError,
        ) as error:
            violations.append(str(error))
        return PlanCheckRecord(
            original_hash,
            context.catalog.snapshot_hash,
            not violations,
            self.checker_version,
            list(dict.fromkeys(violations)),
            margins,
        )
