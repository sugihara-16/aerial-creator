from __future__ import annotations

"""Deterministic articulated trajectory teacher used before learned pi_H."""

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.articulated_reachability import (
    ARTICULATED_IK_TEACHER_VERSION,
    ArticulatedContactIKSolver,
    ArticulatedIKSolution,
    _global_joint_limits,
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
from amsrr.training.order9_configuration_space_planner import (
    DeterministicOrder9ConfigurationSpacePlanner,
    Order9ConfigurationSpacePlan,
    Order9ConfigurationSpacePlanningError,
    Order9ConfigurationState,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
    Order9ResolvedPostureTrajectory,
)
from amsrr.utils.hashing import stable_hash


ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION = (
    "order9_articulated_trajectory_teacher_v17_local_contact_planner"
)


# The free-base contact IK has no environment obstacle input.  These bounded,
# deterministic CoM alternatives let the downstream collision-aware posture
# solver realize the same immutable anchor targets while moving the rest of a
# larger morphology above the bucket support.  Negative-Z candidates are
# deliberately absent: C3 approaches the supported object from above.
_CONFIGURATION_GOAL_CENTROIDAL_OFFSETS_M = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.02),
    (0.0, 0.0, 0.04),
    (0.0, 0.0, 0.06),
    (0.0, 0.02, 0.02),
    (0.0, -0.02, 0.02),
    (0.02, 0.0, 0.02),
    (-0.02, 0.0, 0.02),
)


@dataclass(frozen=True)
class Order9ArticulatedTeacherConfig:
    maximum_candidate_group_attempts: int = 8
    joint_velocity_fraction: float = 0.90
    posture_retime_reserve_fraction: float = 0.95
    minimum_pregrasp_duration_s: float = 0.25
    posture_output_rate_hz: float = 10.0
    rolling_horizon_s: float = 3.0
    # A three-second rolling pi_H window is emitted at 2 Hz.  The previous
    # three-knot form was only 0.67 Hz and its straight task-space
    # interpolation could leave the articulated reachability manifold even
    # though both endpoint postures were feasible.
    rolling_sparse_knot_count: int = 7
    maximum_base_translation_speed_mps: float = 0.10
    maximum_base_rotation_speed_rad_s: float = 0.50
    lift_height_m: float = 0.05
    retreat_distance_m: float = 0.10
    # Mesh-backed anchors are bulkier than their contact-frame point.  Keep
    # enough pregrasp separation for the convex proxy of the complete docking
    # mechanism, not merely the mathematical frame origin.
    pregrasp_clearance_m: float = 0.08
    approach_articulation_clearance_m: float = 0.50
    approach_articulation_vertical_clearance_m: float = 0.40
    approach_staging_position_tolerance_m: float = 0.02
    # Staging evaluates the realized pi_L posture, not nominal IK tracking.
    # Match pi_L's configured 0.15 rad position-correction authority so its
    # valid bounded correction cannot deadlock the upstream teacher.  Contact
    # acquisition still retains the exact anchor/contact physical gates.
    approach_staging_joint_tolerance_rad: float = 0.15
    approach_staging_attitude_tolerance_rad: float = 0.05
    # Once the detached approach has reached pregrasp, contact acquisition
    # must remain local.  In particular, the generic configuration-space
    # planner must not solve a blocked closing edge by retreating to the
    # overhead transit altitude and descending a second time.
    contact_acquisition_ceiling_margin_m: float = 0.02
    contact_acquisition_corridor_margin_m: float = 0.03
    preferred_candidate_group_id: str | None = None
    contact_joint_speed_limit_rad_s: float | None = None
    # Keep each 2 Hz detached-approach segment close enough to the nonlinear
    # articulated manifold for the downstream 20 Hz trajectory IK to track
    # without branch/stopping-tolerance oscillation.
    approach_joint_speed_limit_rad_s: float = 0.20
    # This is an intermediate nominal-path tolerance, not the independent
    # final-contact admission gate.  Keep one millimetre of pilot headroom
    # over the 10 mm contact-path target so sub-millimetre interpolation
    # residuals do not discard an otherwise collision-clear configuration
    # edge.
    posture_anchor_position_tolerance_m: float = 0.011

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
            "contact_acquisition_ceiling_margin_m",
            "contact_acquisition_corridor_margin_m",
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
            not math.isfinite(self.approach_joint_speed_limit_rad_s)
            or self.approach_joint_speed_limit_rad_s <= 0.0
        ):
            raise ValueError(
                "approach_joint_speed_limit_rad_s must be positive"
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
    configuration_space_plan: Order9ConfigurationSpacePlan | None = None
    teacher_version: str = ORDER9_ARTICULATED_TRAJECTORY_TEACHER_VERSION


@dataclass
class _ConfigurationRouteCache:
    identity: tuple[str, str, tuple[int, ...], str, str] | None = None
    goal: Order9ConfigurationState | None = None
    remaining_states: tuple[Order9ConfigurationState, ...] = ()
    source_method: str = ""

    def clear(self) -> None:
        self.identity = None
        self.goal = None
        self.remaining_states = ()
        self.source_method = ""


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
        configuration_space_planner: (
            DeterministicOrder9ConfigurationSpacePlanner | None
        ) = None,
    ) -> None:
        self.physical_model = physical_model
        self.config = config or Order9ArticulatedTeacherConfig()
        self.ik_solver = ik_solver or ArticulatedContactIKSolver(physical_model)
        self.collision_object = collision_object
        self.configuration_space_planner = (
            configuration_space_planner
            or DeterministicOrder9ConfigurationSpacePlanner()
        )
        self._planner_collision_resolver = None
        self._configuration_route_cache = _ConfigurationRouteCache()

    def plan(
        self,
        context: HighLevelPolicyContext,
        *,
        initial_object_poses_world: Mapping[str, Pose7D] | None = None,
        nominal_start_joint_positions_rad: Mapping[str, float] | None = None,
        release_joint_positions_rad: Mapping[str, float] | None = None,
        contact_goal_joint_seed_positions_rad: (
            Mapping[str, float] | None
        ) = None,
    ) -> Order9ArticulatedTeacherPlan:
        contact_goal_joint_seed = _validated_contact_goal_joint_seed(
            context,
            self.physical_model,
            contact_goal_joint_seed_positions_rad,
        )
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
                    # one assignment, not a state estimator.  A fixed,
                    # hash-bound reviewed seed preserves an accepted solution
                    # branch across rolling replans; without one, the
                    # deterministic zero seed retains the prior behavior.
                    # The downstream posture resolver still starts from the
                    # measured joints on every window.
                    initial_joint_positions_rad=contact_goal_joint_seed,
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
                (
                    raw_trajectory,
                    configuration_space_plan,
                    nominal_joint_seed_positions,
                ) = _decorate_raw_trajectory(
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
                    contact_goal_joint_seed_positions_rad=(
                        contact_goal_joint_seed
                    ),
                    collision_object=self.collision_object,
                    configuration_space_planner=(
                        self.configuration_space_planner
                    ),
                    configuration_route_cache=(
                        self._configuration_route_cache
                    ),
                    planner_collision_resolver=(
                        self._configuration_space_collision_resolver()
                        if self.collision_object is not None
                        else None
                    ),
                )
                task_phase = _task_phase(attempt_context)
                resolution = _resolve_teacher_posture_trajectory(
                    physical_model=self.physical_model,
                    config=self.config,
                    context=attempt_context,
                    raw_trajectory=raw_trajectory,
                    initial_joint_positions_rad=resolver_initial_q,
                    collision_object=self.collision_object,
                    nominal_joint_seed_positions_by_raw_knot=(
                        nominal_joint_seed_positions
                    ),
                )
            except (
                Order9ConfigurationSpacePlanningError,
                SchemaValidationError,
                ValueError,
            ) as error:
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
                configuration_space_plan=configuration_space_plan,
            )
        detail = "; ".join(failures[:8])
        raise SchemaValidationError(
            "articulated trajectory teacher found no joint-reachable candidate "
            f"group ({detail})"
        )

    def _configuration_space_collision_resolver(
        self,
    ) -> Order9PostureTrajectoryResolver:
        if self.collision_object is None:
            raise SchemaValidationError(
                "configuration-space planning requires a collision object"
            )
        if self._planner_collision_resolver is None:
            self._planner_collision_resolver = (
                Order9PostureTrajectoryResolver(
                    self.physical_model,
                    config=Order9PostureResolverConfig(
                        output_rate_hz=self.config.posture_output_rate_hz,
                        joint_velocity_fraction=(
                            self.config.joint_velocity_fraction
                        ),
                        enforce_joint_rate=False,
                        anchor_position_tolerance_m=(
                            self.config.posture_anchor_position_tolerance_m
                        ),
                    ),
                    collision_object=self.collision_object,
                )
            )
        return self._planner_collision_resolver


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


def _validated_contact_goal_joint_seed(
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
    seed: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return one fixed, morphology-bound contact IK branch seed."""

    ordered_ids = ordered_global_dock_joint_ids(
        context.morphology_graph,
        physical_model,
    )
    if seed is None:
        return {joint_id: 0.0 for joint_id in ordered_ids}
    if set(seed) != set(ordered_ids):
        raise SchemaValidationError(
            "contact goal joint seed identities differ from the morphology"
        )
    values = {
        joint_id: float(seed[joint_id]) for joint_id in ordered_ids
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise SchemaValidationError(
            "contact goal joint seed values must be finite"
        )
    return values


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
    contact_goal_joint_seed_positions_rad: (
        Mapping[str, float] | None
    ) = None,
    collision_object: Order9PostureCollisionObject | None = None,
    configuration_space_planner: (
        DeterministicOrder9ConfigurationSpacePlanner | None
    ) = None,
    configuration_route_cache: _ConfigurationRouteCache | None = None,
    planner_collision_resolver: (
        Order9PostureTrajectoryResolver | None
    ) = None,
) -> tuple[
    ContactWrenchTrajectory,
    Order9ConfigurationSpacePlan | None,
    tuple[dict[str, float], ...],
]:
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
    configuration_space_plan = None
    if phase == "approach":
        # Resolve a genuine pregrasp configuration whose selected contact
        # frames remain outside the object.  The approach phase reaches this
        # configuration through the existing high/lateral collision-clear
        # staging sequence; contact acquisition alone closes the remaining
        # clearance onto the final contact IK solution.
        attach_assignments = tuple(
            _knot_template_for_state(
                baseline,
                "attach",
            ).contact_assignments
        )
        pregrasp_solution = ArticulatedContactIKSolver(
            physical_model,
            kinematics=kinematics,
        ).solve(
            morphology=context.morphology_graph,
            assignments=attach_assignments,
            candidates=_pregrasp_candidate_mapping(
                attach_assignments,
                context.contact_candidate_set,
                clearance_m=config.pregrasp_clearance_m,
            ),
            initial_joint_positions_rad=solution.joint_positions_rad,
            initial_base_pose_world=solution.base_pose_world,
        )
        if not pregrasp_solution.feasible:
            raise SchemaValidationError(
                "articulated teacher could not resolve its collision-clear "
                "pregrasp configuration"
            )
        if (
            collision_object is None
            or configuration_space_planner is None
            or planner_collision_resolver is None
        ):
            raise SchemaValidationError(
                "Order 9 approach teacher requires deterministic "
                "configuration-space planning with convex collision"
            )
        (
            target_q,
            target_base_pose,
            terminal_phase_target,
            configuration_space_plan,
        ) = _configuration_space_phase_target(
            phase=phase,
            context=context,
            physical_model=physical_model,
            source_observation=observation,
            kinematics=kinematics,
            rigid_body_builder=rigid_body_builder,
            references=references,
            start_q=start_q,
            start_base_pose=start_base_pose,
            desired_q=pregrasp_solution.joint_positions_rad,
            desired_base_pose=pregrasp_solution.base_pose_world,
            current_object_pose=current_object_pose,
            allowed_anchor_ids=(),
            collision_object=collision_object,
            planner=configuration_space_planner,
            collision_resolver=planner_collision_resolver,
            route_cache=configuration_route_cache,
            config=config,
        )
        target_object_pose = tuple(current_object_pose)
    elif phase == "contact_acquisition":
        if (
            collision_object is None
            or configuration_space_planner is None
            or planner_collision_resolver is None
        ):
            raise SchemaValidationError(
                "Order 9 contact-acquisition teacher requires deterministic "
                "configuration-space planning with convex collision"
            )
        (
            target_q,
            target_base_pose,
            terminal_phase_target,
            configuration_space_plan,
        ) = _configuration_space_phase_target(
            phase=phase,
            context=context,
            physical_model=physical_model,
            source_observation=observation,
            kinematics=kinematics,
            rigid_body_builder=rigid_body_builder,
            references=references,
            start_q=start_q,
            start_base_pose=start_base_pose,
            desired_q=solution.joint_positions_rad,
            desired_base_pose=solution.base_pose_world,
            current_object_pose=current_object_pose,
            allowed_anchor_ids=selected_anchor_ids,
            collision_object=collision_object,
            planner=configuration_space_planner,
            collision_resolver=planner_collision_resolver,
            route_cache=configuration_route_cache,
            config=config,
            preferred_goal_joint_seed_positions_rad=(
                contact_goal_joint_seed_positions_rad
            ),
        )
        target_object_pose = tuple(current_object_pose)
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
    elif phase == "approach":
        velocity_limit = min(
            velocity_limit,
            config.approach_joint_speed_limit_rad_s,
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
    nominal_joint_seed_positions: list[dict[str, float]] = []
    denominator = float(config.rolling_sparse_knot_count - 1)
    for index in range(config.rolling_sparse_knot_count):
        local_progress = _smoothstep(float(index) / denominator)
        q = _interpolate_mapping(start_q, end_q, local_progress)
        nominal_joint_seed_positions.append(dict(q))
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
        elif configuration_space_plan is not None:
            # The planner certified this complete base/q edge.  Preserve that
            # edge rather than imposing the old fixed-CoM closure heuristic.
            base_pose = _interpolate_pose(
                start_base_pose,
                end_base_pose,
                local_progress,
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
                    "configuration_space_teacher": (
                        1.0
                        if configuration_space_plan is not None
                        else 0.0
                    ),
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
                    },
                    *(
                        [
                            {
                                "type": (
                                    "order9_configuration_space_teacher"
                                ),
                                "planner_version": (
                                    configuration_space_plan.planner_version
                                ),
                                "method": configuration_space_plan.method,
                                "state_count": str(
                                    len(configuration_space_plan.states)
                                ),
                                "collision_check_count": str(
                                    configuration_space_plan
                                    .collision_check_count
                                ),
                            }
                        ]
                        if configuration_space_plan is not None
                        else []
                    ),
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
    return (
        ContactWrenchTrajectory.from_dict(trajectory.to_dict()),
        configuration_space_plan,
        tuple(nominal_joint_seed_positions),
    )


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


def _configuration_space_phase_target(
    *,
    phase: str,
    context: HighLevelPolicyContext,
    physical_model: PhysicalModel,
    source_observation: RuntimeObservation,
    kinematics: WholeStructureKinematics,
    rigid_body_builder: RigidBodyControlModelBuilder,
    references,
    start_q: Mapping[str, float],
    start_base_pose: Pose7D,
    desired_q: Mapping[str, float],
    desired_base_pose: Pose7D,
    current_object_pose: Pose7D,
    allowed_anchor_ids: Sequence[int],
    collision_object: Order9PostureCollisionObject,
    planner: DeterministicOrder9ConfigurationSpacePlanner,
    collision_resolver: Order9PostureTrajectoryResolver,
    route_cache: _ConfigurationRouteCache | None,
    config: Order9ArticulatedTeacherConfig,
    preferred_goal_joint_seed_positions_rad: (
        Mapping[str, float] | None
    ) = None,
) -> tuple[
    dict[str, float],
    Pose7D,
    bool,
    Order9ConfigurationSpacePlan,
]:
    """Plan one collision-certified receding-horizon configuration edge."""

    collision_solver = collision_resolver.ik_solver
    if not (
        hasattr(collision_solver, "set_collision_scene")
        and hasattr(collision_solver, "check_configuration")
    ):
        raise SchemaValidationError(
            "configuration-space teacher requires the native convex solver"
        )
    desired_fk = kinematics.forward(
        context.morphology_graph,
        physical_model,
        desired_q,
        desired_base_pose,
        references,
    )
    desired_centroidal_pose = _centroidal_pose_from_fk(
        context=context,
        physical_model=physical_model,
        q=desired_q,
        module_root_poses_world=desired_fk.module_root_poses_world,
        source_observation=source_observation,
        builder=rigid_body_builder,
    )
    start = Order9ConfigurationState(
        base_pose_world=tuple(start_base_pose),
        joint_positions_rad={
            joint_id: float(value) for joint_id, value in start_q.items()
        },
    )
    scenes = [
        (
            "object",
            current_object_pose,
            collision_object.size_m,
            tuple(int(value) for value in allowed_anchor_ids),
        ),
        *[
            (box.box_id, box.pose_world, box.size_m, ())
            for box in collision_object.environment_boxes
        ],
    ]
    route_identity = (
        phase,
        stable_hash(context.morphology_graph.to_dict()),
        tuple(sorted(int(value) for value in allowed_anchor_ids)),
        stable_hash(
            {
                "object_pose_world": list(current_object_pose),
                "object_size_m": list(collision_object.size_m),
                "environment_boxes": [
                    {
                        "box_id": box.box_id,
                        "pose_world": list(box.pose_world),
                        "size_m": list(box.size_m),
                    }
                    for box in collision_object.environment_boxes
                ],
                "ground_plane_z_m": collision_object.ground_plane_z_m,
            }
        ),
        stable_hash(
            {
                "desired_base_pose_world": list(desired_base_pose),
                "desired_joint_positions_rad": {
                    joint_id: float(desired_q[joint_id])
                    for joint_id in sorted(desired_q)
                },
            }
        ),
    )
    contact_ceiling_z = (
        max(
            float(start_base_pose[2]),
            float(desired_base_pose[2]),
        )
        + float(config.contact_acquisition_ceiling_margin_m)
    )
    contact_corridor_start = tuple(
        float(value) for value in start_base_pose[:3]
    )
    contact_corridor_goal = tuple(
        float(value) for value in desired_base_pose[:3]
    )

    def collision_free(
        state: Order9ConfigurationState,
        diagnostics: list[str] | None = None,
        *,
        enforce_contact_corridor: bool = True,
    ) -> bool:
        if (
            phase == "contact_acquisition"
            and float(state.base_pose_world[2])
            > contact_ceiling_z
        ):
            if diagnostics is not None:
                diagnostics.append(
                    "contact_local_ceiling:z="
                    f"{float(state.base_pose_world[2]):.6g}:limit="
                    f"{contact_ceiling_z:.6g}"
                )
            return False
        if phase == "contact_acquisition" and enforce_contact_corridor:
            corridor_distance = _point_to_segment_distance(
                tuple(float(value) for value in state.base_pose_world[:3]),
                contact_corridor_start,
                contact_corridor_goal,
            )
            if corridor_distance > float(
                config.contact_acquisition_corridor_margin_m
            ):
                if diagnostics is not None:
                    diagnostics.append(
                        "contact_local_corridor:distance="
                        f"{corridor_distance:.6g}:limit="
                        f"{config.contact_acquisition_corridor_margin_m:.6g}"
                    )
                return False
        fk = kinematics.forward(
            context.morphology_graph,
            physical_model,
            state.joint_positions_rad,
            state.base_pose_world,
            (),
        )
        centroidal_pose = _centroidal_pose_from_fk(
            context=context,
            physical_model=physical_model,
            q=state.joint_positions_rad,
            module_root_poses_world=fk.module_root_poses_world,
            source_observation=source_observation,
            builder=rigid_body_builder,
        )
        for (
            scene_id,
            obstacle_pose,
            obstacle_size,
            scene_allowed_anchors,
        ) in scenes:
            collision_solver.set_collision_scene(
                morphology=context.morphology_graph,
                object_pose_world=obstacle_pose,
                object_size_m=obstacle_size,
                allowed_anchor_ids=scene_allowed_anchors,
            )
            result = collision_solver.check_configuration(
                morphology=context.morphology_graph,
                centroidal_pose_world=centroidal_pose,
                joint_positions_rad=state.joint_positions_rad,
                exact=False,
                margin_m=float(
                    collision_solver.collision_config.collision_margin_m
                ),
                ground_plane_z_m=collision_object.ground_plane_z_m,
            )
            if result.get("accepted") is not True:
                if diagnostics is not None:
                    diagnostics.append(
                        f"{scene_id}:clearance="
                        f"{float(result.get('minimum_clearance_m', math.nan)):.6g}:"
                        f"pairs={int(result.get('violating_pair_count', 0))}:"
                        "ground="
                        f"{int(result.get('ground_violating_proxy_count', 0))}"
                    )
                return False
        return True

    anchor_targets = {
        reference.anchor.anchor_id: (
            desired_fk.anchor_poses_world[reference.anchor.anchor_id]
        )
        for reference in references
    }
    desired_seed = {
        joint_id: float(value) for joint_id, value in desired_q.items()
    }
    start_seed = {
        joint_id: float(value) for joint_id, value in start_q.items()
    }
    zero_seed = {joint_id: 0.0 for joint_id in desired_seed}
    joint_seeds = [desired_seed]
    preferred_seed = (
        None
        if preferred_goal_joint_seed_positions_rad is None
        else {
            joint_id: float(
                preferred_goal_joint_seed_positions_rad[joint_id]
            )
            for joint_id in desired_seed
        }
    )
    for seed in (preferred_seed, start_seed, zero_seed):
        if seed is None:
            continue
        if all(
            any(
                abs(float(seed[joint_id]) - float(existing[joint_id]))
                > 1.0e-12
                for joint_id in seed
            )
            for existing in joint_seeds
        ):
            joint_seeds.append(seed)
    goal = None
    goal_failures: list[str] = []
    cached_goal = (
        None
        if route_cache is None or route_cache.identity != route_identity
        else route_cache.goal
    )
    if cached_goal is not None:
        contact_corridor_goal = tuple(
            float(value) for value in cached_goal.base_pose_world[:3]
        )
    if (
        route_cache is not None
        and route_cache.identity == route_identity
        and cached_goal is not None
        and collision_free(cached_goal)
    ):
        goal = cached_goal
    else:
        contact_corridor_goal = tuple(
            float(value) for value in desired_base_pose[:3]
        )
        for offset_index, offset in enumerate(
            _CONFIGURATION_GOAL_CENTROIDAL_OFFSETS_M
        ):
            centroidal_target = (
                float(desired_centroidal_pose[0]) + float(offset[0]),
                float(desired_centroidal_pose[1]) + float(offset[1]),
                float(desired_centroidal_pose[2]) + float(offset[2]),
                *tuple(
                    float(value) for value in desired_centroidal_pose[3:7]
                ),
            )
            for seed_index, seed in enumerate(joint_seeds):
                for (
                    scene_id,
                    obstacle_pose,
                    obstacle_size,
                    scene_allowed_anchors,
                ) in scenes:
                    collision_solver.set_collision_scene(
                        morphology=context.morphology_graph,
                        object_pose_world=obstacle_pose,
                        object_size_m=obstacle_size,
                        allowed_anchor_ids=scene_allowed_anchors,
                    )
                    refined_goal = collision_solver.solve(
                        morphology=context.morphology_graph,
                        centroidal_pose_world=centroidal_target,
                        anchor_pose_targets_world=anchor_targets,
                        initial_joint_positions_rad=seed,
                    )
                    if not refined_goal.feasible:
                        goal_failures.append(
                            f"offset={offset_index}:seed={seed_index}:"
                            f"scene={scene_id}:ik="
                            f"{refined_goal.maximum_position_error_m:.6g}/"
                            f"{refined_goal.maximum_attitude_error_rad:.6g}"
                        )
                        continue
                    candidate_goal = Order9ConfigurationState(
                        base_pose_world=tuple(refined_goal.base_pose_world),
                        joint_positions_rad={
                            joint_id: float(value)
                            for joint_id, value in (
                                refined_goal.joint_positions_rad.items()
                            )
                        },
                    )
                    collision_diagnostics: list[str] = []
                    if not collision_free(
                        candidate_goal,
                        diagnostics=collision_diagnostics,
                        # A refined redundant-IK solution may shift its base
                        # slightly from the unrefined desired base.  Once it
                        # passes every physical collision check, that actual
                        # feasible base becomes the endpoint of the local
                        # contact corridor below.
                        enforce_contact_corridor=False,
                    ):
                        goal_failures.append(
                            f"offset={offset_index}:seed={seed_index}:"
                            f"scene={scene_id}:"
                            + ",".join(collision_diagnostics)
                        )
                        continue
                    goal = candidate_goal
                    break
                if goal is not None:
                    break
            if goal is not None:
                break
    if goal is None:
        raise SchemaValidationError(
            "configuration-space teacher has no multi-obstacle "
            "collision-free goal (" + "; ".join(goal_failures[:8]) + ")"
        )
    contact_corridor_goal = tuple(
        float(value) for value in goal.base_pose_world[:3]
    )

    ordered_ids = tuple(sorted(start_q))
    plan = _resume_configuration_route(
        cache=route_cache,
        identity=route_identity,
        start=start,
        goal=goal,
        planner=planner,
        collision_free=collision_free,
        ordered_ids=ordered_ids,
    )
    if plan is None:
        joint_limits = _global_joint_limits(
            context.morphology_graph,
            physical_model,
            ordered_ids,
        )
        if phase == "contact_acquisition":
            plan = planner.plan_local_contact(
                start=start,
                goal=goal,
                joint_limits_rad=joint_limits,
                is_collision_free=collision_free,
                corridor_radius_m=(
                    config.contact_acquisition_corridor_margin_m
                ),
            )
        else:
            plan = planner.plan(
                start=start,
                goal=goal,
                joint_limits_rad=joint_limits,
                is_collision_free=collision_free,
                sampling_center_world=current_object_pose[:3],
                overhead_required=not bool(allowed_anchor_ids),
            )
        if route_cache is not None:
            route_cache.identity = route_identity
            route_cache.goal = goal
            route_cache.remaining_states = tuple(plan.states[1:])
            route_cache.source_method = plan.method
    next_state = plan.states[1]
    return (
        dict(next_state.joint_positions_rad),
        tuple(next_state.base_pose_world),
        len(plan.states) == 2,
        plan,
    )


def _resume_configuration_route(
    *,
    cache: _ConfigurationRouteCache | None,
    identity: tuple[str, str, tuple[int, ...], str, str],
    start: Order9ConfigurationState,
    goal: Order9ConfigurationState,
    planner: DeterministicOrder9ConfigurationSpacePlanner,
    collision_free: Callable[[Order9ConfigurationState], bool],
    ordered_ids: Sequence[str],
) -> Order9ConfigurationSpacePlan | None:
    """Resume a certified route after rechecking only its active edge."""

    if (
        cache is None
        or cache.identity != identity
        or cache.goal is None
        or not _configuration_states_close(
            cache.goal,
            goal,
            ordered_ids=ordered_ids,
            translation_tolerance_m=1.0e-4,
            attitude_tolerance_rad=1.0e-4,
            joint_tolerance_rad=1.0e-3,
        )
    ):
        if cache is not None:
            cache.clear()
        return None

    remaining = list(cache.remaining_states)
    while remaining and _configuration_states_close(
        start,
        remaining[0],
        ordered_ids=ordered_ids,
        translation_tolerance_m=(
            planner.config.waypoint_translation_tolerance_m
        ),
        attitude_tolerance_rad=(
            planner.config.waypoint_attitude_tolerance_rad
        ),
        joint_tolerance_rad=planner.config.waypoint_joint_tolerance_rad,
    ):
        remaining.pop(0)
    if not remaining:
        remaining = [goal]
    else:
        # The goal was recomputed from the current observation and may differ
        # by harmless solver roundoff.  The active edge is revalidated below,
        # so retaining the current exact goal is safe and deterministic.
        remaining[-1] = goal

    edge_is_free, collision_checks = planner.validate_direct_edge(
        start=start,
        goal=remaining[0],
        is_collision_free=collision_free,
    )
    if not edge_is_free:
        cache.clear()
        return None

    states = (start, *remaining)
    cache.goal = goal
    cache.remaining_states = tuple(remaining)
    return Order9ConfigurationSpacePlan(
        states=states,
        method=f"cached_route:{cache.source_method}",
        collision_check_count=collision_checks,
        sampled_state_count=0,
        tree_node_count=len(states),
    )


def _configuration_states_close(
    left: Order9ConfigurationState,
    right: Order9ConfigurationState,
    *,
    ordered_ids: Sequence[str],
    translation_tolerance_m: float,
    attitude_tolerance_rad: float,
    joint_tolerance_rad: float,
) -> bool:
    return bool(
        math.sqrt(
            sum(
                (
                    float(left.base_pose_world[index])
                    - float(right.base_pose_world[index])
                )
                ** 2
                for index in range(3)
            )
        )
        <= translation_tolerance_m
        and _quaternion_distance(
            left.base_pose_world[3:], right.base_pose_world[3:]
        )
        <= attitude_tolerance_rad
        and max(
            (
                abs(
                    float(left.joint_positions_rad[joint_id])
                    - float(right.joint_positions_rad[joint_id])
                )
                for joint_id in ordered_ids
            ),
            default=0.0,
        )
        <= joint_tolerance_rad
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


def _point_to_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    goal: Sequence[float],
) -> float:
    """Return the Euclidean distance from a point to a 3-D segment."""

    segment = tuple(float(goal[i]) - float(start[i]) for i in range(3))
    offset = tuple(float(point[i]) - float(start[i]) for i in range(3))
    length_squared = sum(value * value for value in segment)
    if length_squared <= 1.0e-18:
        return math.sqrt(sum(value * value for value in offset))
    fraction = min(
        1.0,
        max(
            0.0,
            sum(offset[i] * segment[i] for i in range(3))
            / length_squared,
        ),
    )
    return math.sqrt(
        sum(
            (
                float(point[i])
                - (float(start[i]) + fraction * segment[i])
            )
            ** 2
            for i in range(3)
        )
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
    nominal_joint_seed_positions_by_raw_knot: Sequence[
        Mapping[str, float]
    ],
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
    # Dense IK can identify a rate peak in more than one raw 2 Hz segment.
    # Bound the deterministic retiming loop generously enough to repair each
    # segment and a second-order peak without changing any geometric target.
    for _attempt in range(4 * len(raw_trajectory.knots) + 1):
        resolution = resolver.resolve(
            context=context,
            raw_trajectory=raw_trajectory,
            initial_joint_positions_rad=initial_joint_positions_rad,
            nominal_joint_seed_positions_by_raw_knot=(
                nominal_joint_seed_positions_by_raw_knot
            ),
        )
        evidence = resolution.evidence
        if evidence.minimum_joint_rate_margin_rad_s >= -1.0e-9:
            return resolution
        dense_segment_index = evidence.maximum_joint_rate_segment_index
        if dense_segment_index is None or dense_segment_index <= 0:
            break
        dense_knots = resolution.trajectory.knots
        if dense_segment_index >= len(dense_knots):
            raise SchemaValidationError(
                "articulated teacher dense retime evidence is invalid"
            )
        dense_segment_midpoint_s = 0.5 * (
            float(dense_knots[dense_segment_index - 1].t_rel_s)
            + float(dense_knots[dense_segment_index].t_rel_s)
        )
        segment_index = _raw_segment_for_time(
            raw_trajectory,
            dense_segment_midpoint_s,
        )
        current_duration = (
            float(raw_trajectory.knots[segment_index].t_rel_s)
            - float(raw_trajectory.knots[segment_index - 1].t_rel_s)
        )
        required_duration = current_duration * (
            evidence.maximum_joint_rate_rad_s
            / evidence.joint_rate_limit_rad_s
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


def _raw_segment_for_time(
    trajectory: ContactWrenchTrajectory,
    time_s: float,
) -> int:
    """Map one dense IK segment midpoint back to its owning raw pi_H segment."""

    if not math.isfinite(float(time_s)):
        raise SchemaValidationError(
            "articulated teacher dense retime time is invalid"
        )
    for index in range(1, len(trajectory.knots)):
        if float(time_s) <= float(trajectory.knots[index].t_rel_s) + 1.0e-12:
            return index
    return len(trajectory.knots) - 1


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
