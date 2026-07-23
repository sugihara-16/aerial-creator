from __future__ import annotations

"""Deterministic articulated trajectory teacher used before learned pi_H."""

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from amsrr.feasibility.articulated_reachability import (
    ARTICULATED_IK_TEACHER_VERSION,
    ArticulatedContactIKSolver,
    ArticulatedIKSolution,
)
from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    PostureTarget,
)
from amsrr.training.order9_teacher import upgrade_teacher_trajectory_to_v2


ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION = (
    "order9_articulated_trajectory_teacher_v1"
)


@dataclass(frozen=True)
class Order9ArticulatedTeacherConfig:
    maximum_candidate_group_attempts: int = 8
    joint_velocity_fraction: float = 0.90
    minimum_pregrasp_duration_s: float = 0.25

    def __post_init__(self) -> None:
        if self.maximum_candidate_group_attempts < 1:
            raise ValueError("maximum_candidate_group_attempts must be positive")
        if not 0.0 < self.joint_velocity_fraction <= 1.0:
            raise ValueError("joint_velocity_fraction must be in (0, 1]")
        if (
            not math.isfinite(self.minimum_pregrasp_duration_s)
            or self.minimum_pregrasp_duration_s <= 0.0
        ):
            raise ValueError("minimum_pregrasp_duration_s must be positive")


@dataclass(frozen=True)
class Order9ArticulatedTeacherPlan:
    trajectory: ContactWrenchTrajectory
    ik_solution: ArticulatedIKSolution
    candidate_group_id: str | None
    teacher_version: str = ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION


class Order9ArticulatedTrajectoryTeacher:
    """Generate complete joint/CoM/anchor targets and retain hard-check separation."""

    teacher_version = ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        config: Order9ArticulatedTeacherConfig | None = None,
        ik_solver: ArticulatedContactIKSolver | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.config = config or Order9ArticulatedTeacherConfig()
        self.ik_solver = ik_solver or ArticulatedContactIKSolver(physical_model)

    def plan(
        self,
        context: HighLevelPolicyContext,
        *,
        initial_object_poses_world: Mapping[str, Pose7D] | None = None,
    ) -> Order9ArticulatedTeacherPlan:
        candidate_groups = _candidate_group_attempts(
            context.contact_candidate_set,
            maximum=self.config.maximum_candidate_group_attempts,
        )
        failures: list[str] = []
        for group_id, candidate_set in candidate_groups:
            attempt_context = replace(
                context,
                contact_candidate_set=candidate_set,
            )
            try:
                baseline = upgrade_teacher_trajectory_to_v2(
                    GraspCarryBaselinePlanner().plan(attempt_context),
                    attempt_context,
                )
                active_knot = next(
                    knot
                    for knot in baseline.knots
                    if any(
                        assignment.schedule_state == "maintain"
                        for assignment in knot.contact_assignments
                    )
                )
                solution = self.ik_solver.solve(
                    morphology=attempt_context.morphology_graph,
                    assignments=active_knot.contact_assignments,
                    candidates={
                        candidate.candidate_id: candidate
                        for candidate in candidate_set.candidates
                    },
                    initial_joint_positions_rad=_runtime_joint_positions(
                        attempt_context
                    ),
                )
            except (SchemaValidationError, StopIteration, ValueError) as error:
                failures.append(f"{group_id or 'fallback'}:{error}")
                continue
            if not solution.feasible:
                failures.append(
                    f"{group_id or 'fallback'}:"
                    f"position={solution.maximum_position_error_m:.6g},"
                    f"normal={solution.maximum_normal_error_rad:.6g}"
                )
                continue
            trajectory = _decorate_complete_trajectory(
                baseline,
                context=attempt_context,
                physical_model=self.physical_model,
                solution=solution,
                initial_object_poses_world=(
                    initial_object_poses_world
                    or _runtime_object_poses(attempt_context)
                ),
                config=self.config,
            )
            return Order9ArticulatedTeacherPlan(
                trajectory=trajectory,
                ik_solution=solution,
                candidate_group_id=group_id,
            )
        detail = "; ".join(failures[:8])
        raise SchemaValidationError(
            "articulated trajectory teacher found no joint-reachable candidate "
            f"group ({detail})"
        )


def _candidate_group_attempts(
    candidate_set: ContactCandidateSet,
    *,
    maximum: int,
) -> tuple[tuple[str | None, ContactCandidateSet], ...]:
    proposals = sorted(
        (
            proposal
            for proposal in candidate_set.group_proposals
            if not proposal.group_violation_codes
        ),
        key=lambda value: (
            _group_priority(value.group_type),
            -float(value.group_score),
            value.group_id,
        ),
    )
    if not proposals:
        return ((None, candidate_set),)
    attempts = []
    for proposal in proposals[:maximum]:
        isolated = ContactCandidateSet.from_dict(candidate_set.to_dict())
        isolated.group_proposals = [
            value
            for value in isolated.group_proposals
            if value.group_id == proposal.group_id
        ]
        attempts.append((proposal.group_id, isolated))
    return tuple(attempts)


def _decorate_complete_trajectory(
    baseline: ContactWrenchTrajectory,
    *,
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
    solution: ArticulatedIKSolution,
    initial_object_poses_world: Mapping[str, Pose7D],
    config: Order9ArticulatedTeacherConfig,
) -> ContactWrenchTrajectory:
    trajectory = ContactWrenchTrajectory.from_dict(baseline.to_dict())
    ordered_ids = tuple(solution.joint_positions_rad)
    runtime_q = _runtime_joint_positions(context)
    start_q = (
        {
            joint_id: float(runtime_q[joint_id])
            for joint_id in ordered_ids
        }
        if runtime_q is not None
        and all(joint_id in runtime_q for joint_id in ordered_ids)
        else {joint_id: 0.0 for joint_id in ordered_ids}
    )
    velocity_limit = (
        _dock_velocity_limit(physical_model) * config.joint_velocity_fraction
    )
    maximum_delta = max(
        (
            abs(solution.joint_positions_rad[joint_id] - start_q[joint_id])
            for joint_id in ordered_ids
        ),
        default=0.0,
    )
    pregrasp_duration = max(
        config.minimum_pregrasp_duration_s,
        maximum_delta / velocity_limit,
    )
    _retime_for_pregrasp(trajectory, pregrasp_duration)

    active_indices = [
        index
        for index, knot in enumerate(trajectory.knots)
        if any(
            assignment.schedule_state in {"attach", "maintain", "slide"}
            for assignment in knot.contact_assignments
        )
    ]
    if not active_indices:
        raise SchemaValidationError(
            "articulated teacher trajectory contains no active contact knot"
        )
    first_active = min(active_indices)
    for index, knot in enumerate(trajectory.knots):
        active = index in active_indices
        q = (
            dict(solution.joint_positions_rad)
            if index >= first_active
            else dict(start_q)
        )
        previous_q = (
            dict(start_q)
            if index == 0
            else (
                dict(solution.joint_positions_rad)
                if index - 1 >= first_active
                else dict(start_q)
            )
        )
        elapsed = (
            pregrasp_duration
            if index == 0
            else max(
                float(knot.t_rel_s)
                - float(trajectory.knots[index - 1].t_rel_s),
                trajectory.dt_s,
            )
        )
        qdot = {
            joint_id: (q[joint_id] - previous_q[joint_id]) / elapsed
            for joint_id in ordered_ids
        }
        object_delta = _object_motion_delta(
            knot,
            initial_object_poses_world,
        )
        centroidal_pose = (
            solution.centroidal_pose_world
            if object_delta is None
            else compose_pose(
                object_delta, solution.centroidal_pose_world
            )
        )
        anchor_poses = (
            {}
            if not active
            else {
                anchor_id: (
                    pose
                    if object_delta is None
                    else compose_pose(object_delta, pose)
                )
                for anchor_id, pose in solution.anchor_poses_world.items()
                if any(
                    assignment.anchor_id == anchor_id
                    and assignment.schedule_state
                    in {"attach", "maintain", "slide"}
                    for assignment in knot.contact_assignments
                )
            }
        )
        existing_wrench = (
            None
            if knot.centroidal_target is None
            else knot.centroidal_target.centroidal_wrench_preference
        )
        knot.centroidal_target = CentroidalTarget(
            com_pos_world=centroidal_pose[:3],
            com_vel_world=(0.0, 0.0, 0.0),
            body_orientation_world=centroidal_pose[3:7],
            centroidal_wrench_preference=existing_wrench,
        )
        knot.posture_target = PostureTarget(
            joint_pos_target=q,
            joint_vel_target=qdot,
            free_anchor_pose_targets=anchor_poses,
        )
    _fill_centroidal_velocities(trajectory)
    trajectory.derived_mode_label = (
        f"{ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION}:"
        f"ik={ARTICULATED_IK_TEACHER_VERSION}:"
        f"source={baseline.derived_mode_label or 'unspecified'}"
    )
    return ContactWrenchTrajectory.from_dict(trajectory.to_dict())


def _retime_for_pregrasp(
    trajectory: ContactWrenchTrajectory,
    pregrasp_duration_s: float,
) -> None:
    first_active = next(
        (
            index
            for index, knot in enumerate(trajectory.knots)
            if any(
                assignment.schedule_state in {"attach", "maintain", "slide"}
                for assignment in knot.contact_assignments
            )
        ),
        None,
    )
    if first_active is None:
        return
    current = float(trajectory.knots[first_active].t_rel_s)
    shift = max(0.0, pregrasp_duration_s - current)
    if shift <= 0.0:
        return
    for index in range(first_active, len(trajectory.knots)):
        trajectory.knots[index].t_rel_s = (
            float(trajectory.knots[index].t_rel_s) + shift
        )
    trajectory.horizon_s = float(trajectory.horizon_s) + shift


def _fill_centroidal_velocities(
    trajectory: ContactWrenchTrajectory,
) -> None:
    for index, knot in enumerate(trajectory.knots):
        target = knot.centroidal_target
        if target is None or target.com_pos_world is None:
            continue
        if index + 1 < len(trajectory.knots):
            next_knot = trajectory.knots[index + 1]
            next_target = next_knot.centroidal_target
            elapsed = float(next_knot.t_rel_s) - float(knot.t_rel_s)
            if (
                next_target is not None
                and next_target.com_pos_world is not None
                and elapsed > 0.0
            ):
                target.com_vel_world = tuple(
                    (
                        float(next_target.com_pos_world[axis])
                        - float(target.com_pos_world[axis])
                    )
                    / elapsed
                    for axis in range(3)
                )
                continue
        target.com_vel_world = (0.0, 0.0, 0.0)


def _object_motion_delta(
    knot: object,
    initial_object_poses_world: Mapping[str, Pose7D],
) -> Pose7D | None:
    targets = tuple(getattr(knot, "object_targets"))
    for target in targets:
        initial = initial_object_poses_world.get(target.object_id)
        if initial is None or target.pose_target_world is None:
            continue
        return compose_pose(target.pose_target_world, inverse_pose(initial))
    return None


def _runtime_joint_positions(
    context: HighLevelPolicyContext,
) -> dict[str, float] | None:
    observation = context.runtime_observation
    if observation is None:
        return None
    values: dict[str, float] = {}
    for state in observation.module_states:
        for local_id, value in state.joint_positions.items():
            values[f"module_{state.module_id}:{local_id}"] = float(value)
    return values or None


def _runtime_object_poses(
    context: HighLevelPolicyContext,
) -> dict[str, Pose7D]:
    observation = context.runtime_observation
    if observation is None:
        return {}
    return {
        state.object_id: state.pose_world for state in observation.object_states
    }


def _dock_velocity_limit(physical_model: PhysicalModel) -> float:
    mechanism_ids = {
        str(port.mechanical_limits.get("mechanism_joint_id"))
        for port in physical_model.dock_ports
    }
    values = [
        float(joint.velocity_limit)
        for joint in physical_model.joints
        if joint.joint_id in mechanism_ids
        and joint.velocity_limit is not None
        and float(joint.velocity_limit) > 0.0
    ]
    specs = physical_model.metadata.get("joint_actuator_specs")
    dock = specs.get("dock") if isinstance(specs, dict) else None
    drive = dock.get("simulation_drive") if isinstance(dock, dict) else None
    safe = (
        drive.get("safe_velocity_limit_rad_s")
        if isinstance(drive, dict)
        else None
    )
    if isinstance(safe, (int, float)) and not isinstance(safe, bool):
        values.append(float(safe))
    if not values or min(values) <= 0.0:
        raise SchemaValidationError(
            "articulated teacher requires a positive Dock velocity limit"
        )
    return min(values)


def _group_priority(group_type: str) -> int:
    return {
        "grasp_pair": 0,
        "multi_grasp": 1,
        "support_set": 2,
    }.get(group_type, 3)


__all__ = [
    "ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION",
    "Order9ArticulatedTeacherConfig",
    "Order9ArticulatedTeacherPlan",
    "Order9ArticulatedTrajectoryTeacher",
]
