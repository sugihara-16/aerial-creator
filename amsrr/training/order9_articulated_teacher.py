from __future__ import annotations

"""Deterministic articulated trajectory teacher used before learned pi_H."""

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.articulated_reachability import (
    ARTICULATED_IK_TEACHER_VERSION,
    ArticulatedContactIKSolver,
    ArticulatedIKSolution,
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.policies import ControllerStatus
from amsrr.training.order9_teacher import upgrade_teacher_trajectory_to_v2
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
    Order9ResolvedPostureTrajectory,
)


ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION = (
    "order9_articulated_trajectory_teacher_v8_release_reference"
)


@dataclass(frozen=True)
class Order9ArticulatedTeacherConfig:
    maximum_candidate_group_attempts: int = 8
    joint_velocity_fraction: float = 0.90
    posture_retime_reserve_fraction: float = 0.95
    minimum_pregrasp_duration_s: float = 0.25
    posture_output_rate_hz: float = 10.0
    rolling_horizon_s: float = 3.0
    rolling_sparse_knot_count: int = 3
    maximum_base_translation_speed_mps: float = 0.10
    maximum_base_rotation_speed_rad_s: float = 0.50
    lift_height_m: float = 0.05
    retreat_distance_m: float = 0.10
    pregrasp_clearance_m: float = 0.03
    approach_articulation_clearance_m: float = 0.50
    approach_articulation_vertical_clearance_m: float = 0.40
    approach_staging_position_tolerance_m: float = 0.02
    # Staging evaluates the realized pi_L posture, not nominal IK tracking.
    # Match pi_L's configured 0.15 rad position-correction authority so its
    # valid bounded correction cannot deadlock the upstream teacher.  Contact
    # acquisition still retains the exact anchor/contact physical gates.
    approach_staging_joint_tolerance_rad: float = 0.15
    approach_staging_attitude_tolerance_rad: float = 0.05
    preferred_candidate_group_id: str | None = None
    contact_joint_speed_limit_rad_s: float | None = None
    posture_anchor_position_tolerance_m: float = 0.005

    def __post_init__(self) -> None:
        if self.maximum_candidate_group_attempts < 1:
            raise ValueError("maximum_candidate_group_attempts must be positive")
        if not 0.0 < self.joint_velocity_fraction <= 1.0:
            raise ValueError("joint_velocity_fraction must be in (0, 1]")
        if not 0.0 < self.posture_retime_reserve_fraction <= 1.0:
            raise ValueError(
                "posture_retime_reserve_fraction must be in (0, 1]"
            )
        if (
            not math.isfinite(self.minimum_pregrasp_duration_s)
            or self.minimum_pregrasp_duration_s <= 0.0
        ):
            raise ValueError("minimum_pregrasp_duration_s must be positive")
        if (
            not math.isfinite(self.posture_output_rate_hz)
            or self.posture_output_rate_hz <= 0.0
        ):
            raise ValueError("posture_output_rate_hz must be positive")
        for name in (
            "rolling_horizon_s",
            "maximum_base_translation_speed_mps",
            "maximum_base_rotation_speed_rad_s",
            "lift_height_m",
            "retreat_distance_m",
            "pregrasp_clearance_m",
            "approach_articulation_clearance_m",
            "approach_articulation_vertical_clearance_m",
            "approach_staging_position_tolerance_m",
            "approach_staging_joint_tolerance_rad",
            "approach_staging_attitude_tolerance_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1.0 <= float(self.rolling_horizon_s) <= 3.0:
            raise ValueError("rolling_horizon_s must be in [1, 3]")
        if self.rolling_sparse_knot_count < 2:
            raise ValueError("rolling_sparse_knot_count must be at least two")
        if (
            self.preferred_candidate_group_id is not None
            and not self.preferred_candidate_group_id
        ):
            raise ValueError(
                "preferred_candidate_group_id must be non-empty when provided"
            )
        if self.contact_joint_speed_limit_rad_s is not None and (
            not math.isfinite(self.contact_joint_speed_limit_rad_s)
            or self.contact_joint_speed_limit_rad_s <= 0.0
        ):
            raise ValueError(
                "contact_joint_speed_limit_rad_s must be positive when provided"
            )
        if (
            not math.isfinite(self.posture_anchor_position_tolerance_m)
            or self.posture_anchor_position_tolerance_m <= 0.0
        ):
            raise ValueError(
                "posture_anchor_position_tolerance_m must be positive"
            )


@dataclass(frozen=True)
class Order9ArticulatedTeacherPlan:
    raw_trajectory: ContactWrenchTrajectory
    trajectory: ContactWrenchTrajectory
    ik_solution: ArticulatedIKSolution
    posture_resolution: Order9ResolvedPostureTrajectory
    candidate_group_id: str | None
    task_phase: str
    phase_target_reached: bool
    teacher_version: str = ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION


class Order9ArticulatedTrajectoryTeacher:
    """Generate one measured-state-anchored, phase-local rolling plan."""

    teacher_version = ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        config: Order9ArticulatedTeacherConfig | None = None,
        ik_solver: ArticulatedContactIKSolver | None = None,
        collision_object: Order9PostureCollisionObject | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.config = config or Order9ArticulatedTeacherConfig()
        self.ik_solver = ik_solver or ArticulatedContactIKSolver(physical_model)
        self.collision_object = collision_object

    def plan(
        self,
        context: HighLevelPolicyContext,
        *,
        initial_object_poses_world: Mapping[str, Pose7D] | None = None,
        nominal_start_joint_positions_rad: Mapping[str, float] | None = None,
        release_joint_positions_rad: Mapping[str, float] | None = None,
    ) -> Order9ArticulatedTeacherPlan:
        candidate_groups = _candidate_group_attempts(
            context.contact_candidate_set,
            maximum=self.config.maximum_candidate_group_attempts,
            preferred_group_id=self.config.preferred_candidate_group_id,
        )
        failures: list[str] = []
        feasible_attempts: list[tuple] = []
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
                    # The contact IK is a deterministic teacher reference for
                    # one assignment, not a state estimator.  A measured-q
                    # seed can switch solution branches between rolling
                    # replans.  The downstream posture resolver still starts
                    # from the measured joints on every window.
                    initial_joint_positions_rad={
                        joint_id: 0.0
                        for joint_id in ordered_global_dock_joint_ids(
                            attempt_context.morphology_graph,
                            self.physical_model,
                        )
                    },
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
            feasible_attempts.append(
                (
                    _candidate_attempt_cost(attempt_context, solution),
                    "" if group_id is None else group_id,
                    group_id,
                    candidate_set,
                    attempt_context,
                    baseline,
                    active_knot,
                    solution,
                )
            )
        for (
            _cost,
            _group_sort_key,
            group_id,
            candidate_set,
            attempt_context,
            baseline,
            active_knot,
            solution,
        ) in sorted(feasible_attempts, key=lambda value: (value[0], value[1])):
            try:
                initial_q = _initial_joint_positions(
                    attempt_context,
                    self.physical_model,
                )
                resolver_initial_q = _nominal_start_joint_positions(
                    initial_q,
                    nominal_start_joint_positions_rad,
                )
                raw_trajectory = _decorate_raw_trajectory(
                    baseline,
                    context=attempt_context,
                    physical_model=self.physical_model,
                    solution=solution,
                    initial_object_poses_world=(
                        initial_object_poses_world
                        or _runtime_object_poses(attempt_context)
                    ),
                    config=self.config,
                    nominal_start_joint_positions_rad=(
                        nominal_start_joint_positions_rad
                    ),
                    release_joint_positions_rad=release_joint_positions_rad,
                )
                task_phase = _task_phase(attempt_context)
                resolution = _resolve_teacher_posture_trajectory(
                    physical_model=self.physical_model,
                    config=self.config,
                    context=attempt_context,
                    raw_trajectory=raw_trajectory,
                    initial_joint_positions_rad=resolver_initial_q,
                    collision_object=self.collision_object,
                )
            except (SchemaValidationError, ValueError) as error:
                failures.append(
                    f"{group_id or 'fallback'}:resolved:{error}"
                )
                continue
            return Order9ArticulatedTeacherPlan(
                raw_trajectory=resolution.raw_trajectory,
                trajectory=resolution.trajectory,
                ik_solution=solution,
                posture_resolution=resolution,
                candidate_group_id=group_id,
                task_phase=task_phase,
                phase_target_reached=_phase_target_reached(
                    resolution.raw_trajectory
                ),
            )
        detail = "; ".join(failures[:8])
        raise SchemaValidationError(
            "articulated trajectory teacher found no joint-reachable candidate "
            f"group ({detail})"
        )


def _candidate_attempt_cost(
    context: HighLevelPolicyContext,
    solution: ArticulatedIKSolution,
) -> tuple[float, float, float]:
    """Rank deterministic teacher contacts by measured approach effort."""

    observation = context.runtime_observation
    if observation is None:
        raise SchemaValidationError(
            "articulated teacher candidate ranking requires runtime state"
        )
    base_module_id = context.morphology_graph.base_module_id
    matches = [
        state
        for state in observation.module_states
        if state.module_id == base_module_id
    ]
    if len(matches) != 1:
        raise SchemaValidationError(
            "articulated teacher candidate ranking lacks one base module"
        )
    current_base_pose = matches[0].pose_world
    translation_distance = math.sqrt(
        sum(
            (
                float(solution.base_pose_world[index])
                - float(current_base_pose[index])
            )
            ** 2
            for index in range(3)
        )
    )
    orientation_distance = _quaternion_distance(
        current_base_pose[3:],
        solution.base_pose_world[3:],
    )
    joint_norm = math.sqrt(
        sum(
            float(value) ** 2
            for value in solution.joint_positions_rad.values()
        )
    )
    return (
        translation_distance + 0.10 * orientation_distance + 0.02 * joint_norm,
        translation_distance,
        joint_norm,
    )


def _candidate_group_attempts(
    candidate_set: ContactCandidateSet,
    *,
    maximum: int,
    preferred_group_id: str | None = None,
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
    if preferred_group_id is not None:
        proposals = [
            proposal
            for proposal in proposals
            if proposal.group_id == preferred_group_id
        ]
        if not proposals:
            raise SchemaValidationError(
                "preferred articulated-teacher candidate group is unavailable"
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


def _decorate_raw_trajectory(
    baseline: ContactWrenchTrajectory,
    *,
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
    solution: ArticulatedIKSolution,
    initial_object_poses_world: Mapping[str, Pose7D],
    config: Order9ArticulatedTeacherConfig,
    nominal_start_joint_positions_rad: Mapping[str, float] | None = None,
    release_joint_positions_rad: Mapping[str, float] | None = None,
) -> ContactWrenchTrajectory:
    observation = context.runtime_observation
    if observation is None:
        raise SchemaValidationError(
            "phase-local articulated teacher requires a runtime observation"
        )
    phase = _task_phase(context)
    ordered_ids = tuple(solution.joint_positions_rad)
    measured_start_q = _initial_joint_positions(context, physical_model)
    if set(measured_start_q) != set(ordered_ids):
        raise SchemaValidationError(
            "articulated teacher IK/runtime Dock joint identities differ"
        )
    start_q = _nominal_start_joint_positions(
        measured_start_q,
        nominal_start_joint_positions_rad,
    )
    module_states = {
        state.module_id: state for state in observation.module_states
    }
    base_state = module_states.get(context.morphology_graph.base_module_id)
    if base_state is None:
        raise SchemaValidationError(
            "articulated teacher runtime observation lacks its base module"
        )
    start_base_pose = tuple(base_state.pose_world)
    selected_anchor_ids = tuple(sorted(solution.anchor_poses_world))
    references = resolve_mesh_backed_anchor_references(
        context.morphology_graph,
        physical_model,
        selected_anchor_ids,
    )
    kinematics = WholeStructureKinematics()
    rigid_body_builder = RigidBodyControlModelBuilder()
    measured_centroidal_pose = rigid_body_builder.build(
        context.morphology_graph,
        physical_model,
        observation,
    ).body_pose_world
    solution_fk = kinematics.forward(
        context.morphology_graph,
        physical_model,
        solution.joint_positions_rad,
        solution.base_pose_world,
        references,
    )
    solution_centroidal_pose = _centroidal_pose_from_fk(
        context=context,
        physical_model=physical_model,
        q=solution.joint_positions_rad,
        module_root_poses_world=solution_fk.module_root_poses_world,
        source_observation=observation,
        builder=rigid_body_builder,
    )
    object_id = _active_object_id(baseline, context)
    current_object_pose = _object_pose(
        observation,
        object_id=object_id,
    )
    initial_object_pose = initial_object_poses_world.get(object_id)
    if initial_object_pose is None:
        raise SchemaValidationError(
            "articulated teacher lacks the selected object's initial pose"
        )
    object_goal_pose = _baseline_object_goal(baseline, object_id=object_id)
    if phase == "approach":
        # C2/Order-8 ordering: first move the assembled CoM to the final grasp
        # pose without changing Dock joints.  Contact articulation begins only
        # after the approach phase has physically completed.
        target_q = dict(start_q)
        target_base_pose = base_pose_for_centroidal_target(
            context.morphology_graph,
            physical_model,
            target_q,
            tuple(solution_centroidal_pose[:3]),
            tuple(solution_centroidal_pose[3:7]),
            kinematics=kinematics,
        )
        target_object_pose = tuple(current_object_pose)
        terminal_phase_target = True
    else:
        (
            target_q,
            target_base_pose,
            target_object_pose,
            terminal_phase_target,
        ) = _phase_target_configuration(
            phase=phase,
            start_q=start_q,
            start_base_pose=start_base_pose,
            current_object_pose=current_object_pose,
            initial_object_pose=initial_object_pose,
            object_goal_pose=object_goal_pose,
            solution=solution,
            config=config,
            release_joint_positions_rad=release_joint_positions_rad,
        )
    target_fk = kinematics.forward(
        context.morphology_graph,
        physical_model,
        target_q,
        target_base_pose,
        references,
    )
    target_centroidal_pose = _centroidal_pose_from_fk(
        context=context,
        physical_model=physical_model,
        q=target_q,
        module_root_poses_world=target_fk.module_root_poses_world,
        source_observation=observation,
        builder=rigid_body_builder,
    )
    velocity_limit = (
        _dock_velocity_limit(physical_model) * config.joint_velocity_fraction
    )
    if (
        phase == "contact_acquisition"
        and config.contact_joint_speed_limit_rad_s is not None
    ):
        velocity_limit = min(
            velocity_limit,
            config.contact_joint_speed_limit_rad_s,
        )
    progress = _rolling_progress_fraction(
        start_q=start_q,
        target_q=target_q,
        start_base_pose=start_base_pose,
        target_base_pose=target_base_pose,
        horizon_s=config.rolling_horizon_s,
        joint_velocity_limit_rad_s=(
            velocity_limit * config.posture_retime_reserve_fraction
        ),
        base_translation_speed_limit_mps=(
            config.maximum_base_translation_speed_mps
        ),
        base_rotation_speed_limit_rad_s=(
            config.maximum_base_rotation_speed_rad_s
        ),
    )
    end_q = _interpolate_mapping(start_q, target_q, progress)
    end_base_pose = _interpolate_pose(
        start_base_pose,
        target_base_pose,
        progress,
    )
    end_object_pose = _interpolate_pose(
        current_object_pose,
        target_object_pose,
        progress,
    )
    assignment_states = _phase_assignment_states(
        phase,
        knot_count=config.rolling_sparse_knot_count,
        target_reached=(
            terminal_phase_target and progress >= 1.0 - 1.0e-9
        ),
    )
    state_templates = _assignment_templates(baseline)
    knots: list[InteractionKnot] = []
    denominator = float(config.rolling_sparse_knot_count - 1)
    for index in range(config.rolling_sparse_knot_count):
        local_progress = _smoothstep(float(index) / denominator)
        q = _interpolate_mapping(start_q, end_q, local_progress)
        if index == 0:
            # A receding-horizon replan retains the previous nominal endpoint
            # as pi_L's joint reference while the runtime observation remains
            # the measured state.  Realize that nominal posture at the current
            # measured centroidal pose so neither the body target nor the
            # actor-relative joint correction is applied twice at a boundary.
            base_pose = base_pose_for_centroidal_target(
                context.morphology_graph,
                physical_model,
                q,
                tuple(measured_centroidal_pose[:3]),
                tuple(measured_centroidal_pose[3:7]),
                kinematics=kinematics,
            )
        elif phase == "contact_acquisition":
            # Keep the already-reached final CoM pose fixed while IK closes
            # the Dock joints onto the assigned contact points.
            base_pose = base_pose_for_centroidal_target(
                context.morphology_graph,
                physical_model,
                q,
                tuple(target_centroidal_pose[:3]),
                tuple(target_centroidal_pose[3:7]),
                kinematics=kinematics,
            )
        elif phase == "release":
            # Opening the free anchors must not move the assembled body target.
            # The object is already supported at the place pose and is no
            # longer part of the payload coupling in this phase.
            base_pose = base_pose_for_centroidal_target(
                context.morphology_graph,
                physical_model,
                q,
                tuple(measured_centroidal_pose[:3]),
                tuple(measured_centroidal_pose[3:7]),
                kinematics=kinematics,
            )
        else:
            base_pose = _interpolate_pose(
                start_base_pose,
                end_base_pose,
                local_progress,
            )
        object_pose = _interpolate_pose(
            current_object_pose,
            end_object_pose,
            local_progress,
        )
        state = assignment_states[index]
        assignments = [
            ContactAssignment.from_dict(value.to_dict())
            for value in state_templates[state]
        ]
        fk = kinematics.forward(
            context.morphology_graph,
            physical_model,
            q,
            base_pose,
            references,
        )
        centroidal_pose = _centroidal_pose_from_fk(
            context=context,
            physical_model=physical_model,
            q=q,
            module_root_poses_world=fk.module_root_poses_world,
            source_observation=observation,
            builder=rigid_body_builder,
        )
        if index == 0:
            centroidal_pose = measured_centroidal_pose
        # A release assignment removes contact/wrench authority, but pi_H
        # still owns the free-anchor separation targets that let trajectory IK
        # generate the opening posture.  Retreat/settle keep them unconstrained.
        anchors_enabled = state != "release" or phase == "release"
        anchor_targets = (
            {
                anchor_id: fk.anchor_poses_world[anchor_id]
                for anchor_id in selected_anchor_ids
            }
            if anchors_enabled
            else {}
        )
        template = _knot_template_for_state(baseline, state)
        object_targets = _phase_object_targets(
            phase=phase,
            state=state,
            object_id=object_id,
            object_pose=object_pose,
        )
        existing_wrench = (
            None
            if template.centroidal_target is None
            else template.centroidal_target.centroidal_wrench_preference
        )
        knots.append(
            InteractionKnot(
                t_rel_s=(
                    config.rolling_horizon_s * float(index) / denominator
                ),
                contact_assignments=assignments,
                centroidal_target=CentroidalTarget(
                    com_pos_world=centroidal_pose[:3],
                    com_vel_world=(0.0, 0.0, 0.0),
                    body_orientation_world=centroidal_pose[3:7],
                    centroidal_wrench_preference=(
                        None
                        if existing_wrench is None
                        else list(existing_wrench)
                    ),
                ),
                posture_target=PostureTarget(
                    free_anchor_pose_targets=anchor_targets,
                ),
                object_targets=object_targets,
                priority_weights={
                    **dict(template.priority_weights),
                    "phase_local_rolling_teacher": 1.0,
                },
                guard_conditions=[
                    {
                        "type": "order9_phase_local_rolling_teacher",
                        "phase": phase,
                        "phase_target_reached": (
                            "true"
                            if (
                                terminal_phase_target
                                and progress >= 1.0 - 1.0e-9
                            )
                            else "false"
                        ),
                    }
                ],
            )
        )
    trajectory = ContactWrenchTrajectory(
        horizon_s=float(config.rolling_horizon_s),
        dt_s=float(config.rolling_horizon_s) / denominator,
        knots=knots,
        contract_version=baseline.contract_version,
    )
    _fill_centroidal_velocities(trajectory)
    trajectory.derived_mode_label = (
        f"{ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION}:"
        f"ik={ARTICULATED_IK_TEACHER_VERSION}:"
        f"phase={phase}:"
        "target_reached="
        f"{terminal_phase_target and progress >= 1.0 - 1.0e-9}:"
        "raw_pi_h_no_joint_targets:"
        f"source={baseline.derived_mode_label or 'unspecified'}"
    )
    return ContactWrenchTrajectory.from_dict(trajectory.to_dict())


def _task_phase(context: HighLevelPolicyContext) -> str:
    observation = context.runtime_observation
    raw = (
        None
        if observation is None
        else observation.task_progress.phase_label
    )
    phase = "approach" if raw is None else str(raw)
    aliases = {
        "establish_contact": "contact_acquisition",
        "apply_wrench": "contact_acquisition",
        "complete": "settle",
        "safe_hold": "settle",
    }
    phase = aliases.get(phase, phase)
    supported = {
        "approach",
        "contact_acquisition",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
        "settle",
    }
    if phase not in supported:
        raise SchemaValidationError(
            f"articulated teacher does not support task phase {phase!r}"
        )
    return phase


def _phase_target_reached(trajectory: ContactWrenchTrajectory) -> bool:
    for guard in trajectory.knots[-1].guard_conditions:
        if guard.get("type") != "order9_phase_local_rolling_teacher":
            continue
        return guard.get("phase_target_reached") == "true"
    raise SchemaValidationError(
        "articulated teacher trajectory lacks rolling completion evidence"
    )


def _active_object_id(
    baseline: ContactWrenchTrajectory,
    context: HighLevelPolicyContext,
) -> str:
    candidates = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    object_ids = {
        candidates[assignment.candidate_id].target_entity_id
        for knot in baseline.knots
        for assignment in knot.contact_assignments
        if assignment.schedule_state in {"attach", "maintain", "slide"}
        and assignment.candidate_id in candidates
    }
    if len(object_ids) != 1:
        raise SchemaValidationError(
            "articulated teacher must resolve exactly one active object"
        )
    return next(iter(object_ids))


def _object_pose(
    observation: RuntimeObservation,
    *,
    object_id: str,
) -> Pose7D:
    matches = [
        state.pose_world
        for state in observation.object_states
        if state.object_id == object_id
    ]
    if len(matches) != 1:
        raise SchemaValidationError(
            "articulated teacher runtime observation must contain its "
            "selected object exactly once"
        )
    return tuple(matches[0])


def _baseline_object_goal(
    baseline: ContactWrenchTrajectory,
    *,
    object_id: str,
) -> Pose7D | None:
    values = [
        target.pose_target_world
        for knot in baseline.knots
        for target in knot.object_targets
        if target.object_id == object_id
        and target.pose_target_world is not None
    ]
    return None if not values else tuple(values[-1])


def _phase_target_configuration(
    *,
    phase: str,
    start_q: Mapping[str, float],
    start_base_pose: Pose7D,
    current_object_pose: Pose7D,
    initial_object_pose: Pose7D,
    object_goal_pose: Pose7D | None,
    solution: ArticulatedIKSolution,
    config: Order9ArticulatedTeacherConfig,
    release_joint_positions_rad: Mapping[str, float] | None,
) -> tuple[dict[str, float], Pose7D, Pose7D, bool]:
    if phase == "release":
        if release_joint_positions_rad is None:
            raise SchemaValidationError(
                "articulated release teacher requires an explicit open "
                "Dock reference"
            )
        target_q = _nominal_start_joint_positions(
            start_q,
            release_joint_positions_rad,
        )
        return (
            target_q,
            tuple(start_base_pose),
            tuple(current_object_pose),
            True,
        )
    if phase == "settle":
        return (
            dict(start_q),
            tuple(start_base_pose),
            tuple(current_object_pose),
            True,
        )
    if phase == "retreat":
        return (
            dict(start_q),
            (
                float(start_base_pose[0]) - config.retreat_distance_m,
                *start_base_pose[1:],
            ),
            tuple(current_object_pose),
            True,
        )
    if phase == "lift":
        target_object = (
            float(initial_object_pose[0]),
            float(initial_object_pose[1]),
            float(initial_object_pose[2]) + config.lift_height_m,
            *initial_object_pose[3:],
        )
    elif phase == "transport":
        target_object = (
            tuple(current_object_pose)
            if object_goal_pose is None
            else (
                float(object_goal_pose[0]),
                float(object_goal_pose[1]),
                float(current_object_pose[2]),
                *object_goal_pose[3:],
            )
        )
    elif phase == "place":
        target_object = (
            tuple(current_object_pose)
            if object_goal_pose is None
            else tuple(object_goal_pose)
        )
    else:
        target_object = tuple(current_object_pose)
    if phase in {"lift", "transport", "place"}:
        # Once physical contact is established, preserve the measured grasp
        # posture and move the coupled robot/object state together.  Returning
        # to the detached canonical IK solution can unload one contact before
        # the payload starts moving.
        object_delta = compose_pose(
            target_object,
            inverse_pose(current_object_pose),
        )
        target_base_pose = compose_pose(
            object_delta,
            start_base_pose,
        )
        target_q = dict(start_q)
    else:
        object_delta = compose_pose(
            target_object,
            inverse_pose(initial_object_pose),
        )
        target_base_pose = compose_pose(
            object_delta,
            solution.base_pose_world,
        )
        target_q = dict(solution.joint_positions_rad)
    return (
        target_q,
        target_base_pose,
        target_object,
        True,
    )


def _pregrasp_candidate_mapping(
    assignments: Sequence[ContactAssignment],
    candidate_set: ContactCandidateSet,
    *,
    clearance_m: float,
) -> dict[int, ContactCandidate]:
    """Move each assigned object-surface target outward for pre-contact IK."""

    active_candidate_ids = {
        int(assignment.candidate_id)
        for assignment in assignments
        if assignment.schedule_state in {"attach", "maintain", "slide"}
    }
    candidates = {
        int(candidate.candidate_id): ContactCandidate.from_dict(
            candidate.to_dict()
        )
        for candidate in candidate_set.candidates
        if int(candidate.candidate_id) in active_candidate_ids
    }
    if set(candidates) != active_candidate_ids:
        raise SchemaValidationError(
            "articulated teacher pregrasp references an unknown candidate"
        )
    for candidate in candidates.values():
        normal_norm = math.sqrt(
            sum(float(value) ** 2 for value in candidate.normal_world)
        )
        if normal_norm <= 1.0e-12:
            raise SchemaValidationError(
                "articulated teacher pregrasp candidate has zero normal"
            )
        offset = tuple(
            float(clearance_m) * float(value) / normal_norm
            for value in candidate.normal_world
        )
        pose = list(candidate.contact_pose_world)
        frame = list(candidate.contact_frame_world)
        for index in range(3):
            pose[index] = float(pose[index]) + offset[index]
            frame[index] = float(frame[index]) + offset[index]
        candidate.contact_pose_world = tuple(pose)  # type: ignore[assignment]
        candidate.contact_frame_world = tuple(frame)  # type: ignore[assignment]
        candidate.validate()
    return candidates


def _pregrasp_base_pose(
    contact_base_pose: Pose7D,
    *,
    object_pose: Pose7D,
    clearance_m: float,
) -> Pose7D:
    radial = (
        float(contact_base_pose[0]) - float(object_pose[0]),
        float(contact_base_pose[1]) - float(object_pose[1]),
        float(contact_base_pose[2]) - float(object_pose[2]),
    )
    norm = math.sqrt(sum(value * value for value in radial))
    if norm <= 1.0e-9:
        radial = (1.0, 0.0, 0.0)
        norm = 1.0
    return (
        *(
            float(contact_base_pose[index])
            + float(clearance_m) * radial[index] / norm
            for index in range(3)
        ),
        *contact_base_pose[3:],
    )


def _approach_staging_target(
    *,
    start_q: Mapping[str, float],
    start_base_pose: Pose7D,
    contact_q: Mapping[str, float],
    contact_base_pose: Pose7D,
    object_pose: Pose7D,
    config: Order9ArticulatedTeacherConfig,
) -> tuple[dict[str, float], Pose7D] | None:
    """Keep articulation away from the object before the final approach.

    The free-base contact IK often places the assembled base to one lateral
    side of the object.  First translate the unchanged current posture along
    that lateral axis, then realize the contact posture/orientation while
    staying at that clearance, and only then translate to the contact pose.
    """

    maximum_joint_error = max(
        (
            abs(float(contact_q[joint_id]) - float(start_q[joint_id]))
            for joint_id in start_q
        ),
        default=0.0,
    )
    attitude_error = _quaternion_distance(
        start_base_pose[3:],
        contact_base_pose[3:],
    )
    staging_base_pose = _pregrasp_base_pose(
        contact_base_pose,
        object_pose=object_pose,
        clearance_m=config.approach_articulation_clearance_m,
    )
    staging_base_pose = (
        float(staging_base_pose[0]),
        float(staging_base_pose[1]),
        float(contact_base_pose[2])
        + config.approach_articulation_vertical_clearance_m,
        *staging_base_pose[3:],
    )
    contact_distance_sq = sum(
        (
            float(contact_q[joint_id]) - float(start_q[joint_id])
        )
        ** 2
        for joint_id in start_q
    )
    neutral_distance_sq = sum(
        float(start_q[joint_id]) ** 2 for joint_id in start_q
    )
    articulation_committed = bool(
        neutral_distance_sq > 1.0e-12
        and contact_distance_sq <= neutral_distance_sq
    )
    posture_established = bool(
        maximum_joint_error
        <= config.approach_staging_joint_tolerance_rad
        and attitude_error
        <= config.approach_staging_attitude_tolerance_rad
    )
    if posture_established:
        # Descend at the collision-clear lateral staging position before the
        # final horizontal approach.  A direct high-to-pregrasp diagonal cuts
        # through the object even when both endpoints are collision-free.
        if (
            abs(float(start_base_pose[2]) - float(contact_base_pose[2]))
            > config.approach_staging_position_tolerance_m
        ):
            values = list(start_base_pose)
            values[0:2] = list(staging_base_pose[0:2])
            values[2] = float(contact_base_pose[2])
            values[3:7] = list(contact_base_pose[3:7])
            return (
                dict(contact_q),
                tuple(values),  # type: ignore[arg-type]
            )
        # Once posture and approach height are established, continue toward
        # the independently opened pregrasp anchor targets.  Never re-enter
        # the neutral transit stages because of bounded tracking residual.
        return None
    if (
        articulation_committed
    ):
        # The articulated posture is established, or the measured posture has
        # crossed the contact/neutral midpoint.  Keep moving toward contact
        # while the slower assembled-body attitude converges; a receding-
        # horizon tracking residual must never reopen it to neutral.
        values = list(start_base_pose)
        values[0:2] = list(staging_base_pose[0:2])
        values[2] = float(staging_base_pose[2])
        values[3:7] = list(contact_base_pose[3:7])
        return (
            dict(contact_q),
            tuple(values),  # type: ignore[arg-type]
        )
    vertical_error = (
        float(staging_base_pose[2]) - float(start_base_pose[2])
    )
    if (
        abs(vertical_error)
        > config.approach_staging_position_tolerance_m
    ):
        return (
            {joint_id: 0.0 for joint_id in start_q},
            (
                float(start_base_pose[0]),
                float(start_base_pose[1]),
                float(staging_base_pose[2]),
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        )
    lateral_error = math.sqrt(
        sum(
            (
                float(staging_base_pose[index])
                - float(start_base_pose[index])
            )
            ** 2
            for index in (0, 1)
        )
    )
    if (
        lateral_error
        > config.approach_staging_position_tolerance_m
    ):
        values = list(start_base_pose)
        values[0:2] = list(staging_base_pose[0:2])
        # Do not re-anchor the nominal posture/height to each measured
        # endpoint.  Otherwise a bounded pi_L steady-state correction becomes
        # cumulative across rolling replans.  The collision-clear transit
        # keeps a stable neutral posture and the absolute pregrasp height.
        values[2] = float(staging_base_pose[2])
        values[3:7] = [0.0, 0.0, 0.0, 1.0]
        return (
            {joint_id: 0.0 for joint_id in start_q},
            tuple(values),  # type: ignore[arg-type]
        )

    return (
        dict(contact_q),
        (
            float(start_base_pose[0]),
            float(start_base_pose[1]),
            float(staging_base_pose[2]),
            *contact_base_pose[3:],
        ),
    )


def _rolling_progress_fraction(
    *,
    start_q: Mapping[str, float],
    target_q: Mapping[str, float],
    start_base_pose: Pose7D,
    target_base_pose: Pose7D,
    horizon_s: float,
    joint_velocity_limit_rad_s: float,
    base_translation_speed_limit_mps: float,
    base_rotation_speed_limit_rad_s: float,
) -> float:
    requirements = [
        max(
            (
                abs(float(target_q[joint_id]) - float(start_q[joint_id]))
                / (joint_velocity_limit_rad_s * horizon_s)
                for joint_id in start_q
            ),
            default=0.0,
        ),
        math.sqrt(
            sum(
                (
                    float(target_base_pose[index])
                    - float(start_base_pose[index])
                )
                ** 2
                for index in range(3)
            )
        )
        / (base_translation_speed_limit_mps * horizon_s),
        _quaternion_distance(
            start_base_pose[3:],
            target_base_pose[3:],
        )
        / (base_rotation_speed_limit_rad_s * horizon_s),
    ]
    required_windows = max(requirements, default=0.0)
    if required_windows <= 1.0:
        return 1.0
    return 1.0 / required_windows


def _phase_assignment_states(
    phase: str,
    *,
    knot_count: int,
    target_reached: bool,
) -> tuple[str, ...]:
    if phase == "approach":
        return ("approach",) * knot_count
    if phase == "contact_acquisition":
        if not target_reached:
            return ("attach",) * knot_count
        return (*(("attach",) * max(0, knot_count - 1)), "maintain")
    if phase in {"lift", "transport", "place"}:
        return ("maintain",) * knot_count
    return ("release",) * knot_count


def _assignment_templates(
    baseline: ContactWrenchTrajectory,
) -> dict[str, tuple[ContactAssignment, ...]]:
    templates: dict[str, tuple[ContactAssignment, ...]] = {}
    for state in ("approach", "attach", "maintain", "release"):
        knot = _knot_template_for_state(baseline, state)
        templates[state] = tuple(knot.contact_assignments)
    return templates


def _knot_template_for_state(
    baseline: ContactWrenchTrajectory,
    state: str,
) -> InteractionKnot:
    return next(
        (
            knot
            for knot in baseline.knots
            if knot.contact_assignments
            and all(
                assignment.schedule_state == state
                for assignment in knot.contact_assignments
            )
        ),
        baseline.knots[0],
    )


def _phase_object_targets(
    *,
    phase: str,
    state: str,
    object_id: str,
    object_pose: Pose7D,
) -> list[ObjectTarget]:
    if phase == "approach" and state == "approach":
        return []
    return [
        ObjectTarget(
            object_id=object_id,
            pose_target_world=tuple(object_pose),
            twist_target_world=[0.0] * 6,
        )
    ]


def _centroidal_pose_from_fk(
    *,
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
    q: Mapping[str, float],
    module_root_poses_world: Mapping[int, Pose7D],
    source_observation: RuntimeObservation,
    builder: RigidBodyControlModelBuilder,
) -> Pose7D:
    local_q: dict[int, dict[str, float]] = {
        module.module_id: {} for module in context.morphology_graph.modules
    }
    for global_id, value in q.items():
        module_label, local_id = global_id.split(":", 1)
        module_id = int(module_label.removeprefix("module_"))
        local_q[module_id][local_id] = float(value)
    observation = RuntimeObservation(
        time_s=float(source_observation.time_s),
        morphology_graph=context.morphology_graph,
        module_states=[
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=module_root_poses_world[module_id],
                twist_world=[0.0] * 6,
                joint_positions=local_q[module_id],
                joint_velocities={
                    local_id: 0.0 for local_id in local_q[module_id]
                },
            )
            for module_id in sorted(module_root_poses_world)
        ],
        object_states=[
            ObjectRuntimeState.from_dict(state.to_dict())
            for state in source_observation.object_states
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState.from_dict(
            source_observation.task_progress.to_dict()
        ),
    )
    return builder.build(
        context.morphology_graph,
        physical_model,
        observation,
    ).body_pose_world


def _interpolate_mapping(
    start: Mapping[str, float],
    target: Mapping[str, float],
    ratio: float,
) -> dict[str, float]:
    return {
        key: float(start[key])
        + (float(target[key]) - float(start[key])) * float(ratio)
        for key in start
    }


def _interpolate_pose(
    start: Pose7D,
    target: Pose7D,
    ratio: float,
) -> Pose7D:
    value = min(max(float(ratio), 0.0), 1.0)
    return (
        *(
            float(start[index])
            + (float(target[index]) - float(start[index])) * value
            for index in range(3)
        ),
        *_quaternion_slerp(start[3:], target[3:], value),
    )


def _quaternion_slerp(
    start: Sequence[float],
    target: Sequence[float],
    ratio: float,
) -> tuple[float, float, float, float]:
    left = _normalized_quaternion(start)
    right = _normalized_quaternion(target)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    if dot < 0.0:
        right = tuple(-value for value in right)
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalized_quaternion(
            tuple(
                left[index]
                + (right[index] - left[index]) * float(ratio)
                for index in range(4)
            )
        )
    angle = math.acos(dot)
    sine = math.sin(angle)
    left_weight = math.sin((1.0 - float(ratio)) * angle) / sine
    right_weight = math.sin(float(ratio) * angle) / sine
    return _normalized_quaternion(
        tuple(
            left_weight * left[index] + right_weight * right[index]
            for index in range(4)
        )
    )


def _normalized_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    resolved = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in resolved))
    if len(resolved) != 4 or norm <= 1.0e-12:
        raise SchemaValidationError(
            "articulated teacher quaternion is invalid"
        )
    return tuple(value / norm for value in resolved)  # type: ignore[return-value]


def _quaternion_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    first = _normalized_quaternion(left)
    second = _normalized_quaternion(right)
    dot = abs(
        sum(a * b for a, b in zip(first, second, strict=True))
    )
    return 2.0 * math.acos(min(max(dot, -1.0), 1.0))


def _smoothstep(value: float) -> float:
    clipped = min(max(float(value), 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


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


def _resolve_teacher_posture_trajectory(
    *,
    physical_model: PhysicalModel,
    config: Order9ArticulatedTeacherConfig,
    context: HighLevelPolicyContext,
    raw_trajectory: ContactWrenchTrajectory,
    initial_joint_positions_rad: Mapping[str, float],
    collision_object: Order9PostureCollisionObject | None,
) -> Order9ResolvedPostureTrajectory:
    """Author sufficient teacher timing, then export one immutable raw plan."""

    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        config=Order9PostureResolverConfig(
            output_rate_hz=config.posture_output_rate_hz,
            joint_velocity_fraction=config.joint_velocity_fraction,
            enforce_joint_rate=False,
            anchor_position_tolerance_m=(
                config.posture_anchor_position_tolerance_m
            ),
        ),
        collision_object=collision_object,
    )
    for _attempt in range(len(raw_trajectory.knots) + 1):
        resolution = resolver.resolve(
            context=context,
            raw_trajectory=raw_trajectory,
            initial_joint_positions_rad=initial_joint_positions_rad,
        )
        evidence = resolution.evidence
        if evidence.minimum_joint_rate_margin_rad_s >= -1.0e-9:
            return resolution
        segment_index = evidence.maximum_joint_rate_segment_index
        if segment_index is None or segment_index <= 0:
            break
        required_duration = (
            evidence.minimum_required_segment_duration_s
            / config.posture_retime_reserve_fraction
        )
        _retime_segment(
            raw_trajectory,
            segment_index=segment_index,
            minimum_duration_s=required_duration,
        )
        _fill_centroidal_velocities(raw_trajectory)
    raise SchemaValidationError(
        "articulated teacher could not author a Dock-rate-feasible raw "
        "trajectory timing"
    )


def _retime_segment(
    trajectory: ContactWrenchTrajectory,
    *,
    segment_index: int,
    minimum_duration_s: float,
) -> None:
    if not 0 < segment_index < len(trajectory.knots):
        raise SchemaValidationError(
            "articulated teacher retime segment index is invalid"
        )
    current_duration = (
        float(trajectory.knots[segment_index].t_rel_s)
        - float(trajectory.knots[segment_index - 1].t_rel_s)
    )
    shift = max(0.0, float(minimum_duration_s) - current_duration)
    if shift <= 0.0:
        raise SchemaValidationError(
            "articulated teacher rate retime made no progress"
        )
    for index in range(segment_index, len(trajectory.knots)):
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


def _ensure_active_object_target(
    knot: object,
    *,
    context: HighLevelPolicyContext,
    initial_object_poses_world: Mapping[str, Pose7D],
) -> None:
    """Keep attached-object motion explicit across interpolation intervals."""

    candidates = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    object_ids = {
        candidates[assignment.candidate_id].target_entity_id
        for assignment in getattr(knot, "contact_assignments")
        if assignment.schedule_state in {"attach", "maintain", "slide"}
        and assignment.candidate_id in candidates
    }
    if len(object_ids) != 1:
        raise SchemaValidationError(
            "articulated teacher active knot must resolve one target object"
        )
    object_id = next(iter(object_ids))
    targets = list(getattr(knot, "object_targets"))
    if any(target.object_id == object_id for target in targets):
        return
    initial = initial_object_poses_world.get(object_id)
    if initial is None:
        raise SchemaValidationError(
            "articulated teacher lacks the active object's initial pose"
        )
    targets.append(
        ObjectTarget(
            object_id=object_id,
            pose_target_world=initial,
        )
    )
    setattr(knot, "object_targets", targets)


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


def _initial_joint_positions(
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
) -> dict[str, float]:
    ordered_ids = ordered_global_dock_joint_ids(
        context.morphology_graph,
        physical_model,
    )
    runtime = _runtime_joint_positions(context)
    if runtime is None:
        return {joint_id: 0.0 for joint_id in ordered_ids}
    missing = [joint_id for joint_id in ordered_ids if joint_id not in runtime]
    if missing:
        raise SchemaValidationError(
            "articulated teacher runtime observation lacks initial Dock "
            f"joint state for {missing}"
        )
    return {
        joint_id: float(runtime[joint_id])
        for joint_id in ordered_ids
    }


def _nominal_start_joint_positions(
    measured: Mapping[str, float],
    requested: Mapping[str, float] | None,
) -> dict[str, float]:
    if requested is None:
        return {joint_id: float(value) for joint_id, value in measured.items()}
    if set(requested) != set(measured):
        raise SchemaValidationError(
            "articulated teacher nominal start Dock identities differ from "
            "the measured state"
        )
    values = {
        joint_id: float(requested[joint_id])
        for joint_id in measured
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise SchemaValidationError(
            "articulated teacher nominal start Dock state must be finite"
        )
    return values


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
