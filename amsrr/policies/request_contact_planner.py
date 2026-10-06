"""Bounded measured-state planning for the provisional grasp/carry profile.

Reuses deterministic contact IK and configuration-space planning, or reviewed
geometry for the selected binding. Nominal preload and executable-path checks
are recomputed after selection; geometric seeds are never actor inputs.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import time

from amsrr.feasibility.articulated_reachability import (
    ArticulatedContactIKSolver, ArticulatedIKConfig,
    _global_joint_limits, _joint_limit_branch_seeds, ordered_global_dock_joint_ids,
)
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTrajectoryTeacher,
    Order9ArticulatedTeacherConfig,
)
from amsrr.training.order9_c3_teacher import build_order9_c3_posture_collision_object
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalWindow,
    _ideal_endpoint_observation,
    _phase_trajectories_from_nominal_windows,
    _build_complete_task_phase_trajectory,
    _complete_task_knot_state,
    _build_reversed_accepted_phase_trajectory,
)
from amsrr.training.order9_r1_nominal_retreat import (
    materialize_order9_r1_complete_task_phases,
)
from amsrr.training.order9_teacher import compile_high_level_context
from amsrr.schemas.runtime import TaskProgressState
from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import stable_hash


def reuse_reviewed_request_geometry(context, request, reviewed_bundle):
    """Reuse an available route only AFTER selection of its exact contact group.

    This is a geometric seed library for the imitation/PPO grasp profile, not
    evidence of arbitrary online planning. Nominal force, compression and the
    complete interpolated-path checks are recomputed by the caller.
    """
    from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
    from amsrr.training.order9_articulated_teacher import _initial_joint_positions

    entry = context.catalog.resolve(request)
    if context.execution_state.plan_id is not None or request.transition_id is not None:
        raise ValueError("reviewed geometric seed requires an initial request")
    contact = reviewed_bundle.phase_trajectories["contact_acquisition"].knots[-1]
    if set(entry.candidate_ids) != {
        a.candidate_id for a in contact.contact_assignments
    }:
        raise ValueError("reviewed geometric seed has a different contact binding")
    if reviewed_bundle.morphology.to_dict() != context.scene.morphology_graph.to_dict():
        raise ValueError("reviewed geometric seed has a different morphology")
    phases = deepcopy(reviewed_bundle.phase_trajectories)
    first = phases["approach"].knots[0]
    measured = RigidBodyControlModelBuilder().build(
        context.scene.morphology_graph,
        context.physical_model,
        context.scene.runtime_observation,
    )
    q = _initial_joint_positions(context.scene, context.physical_model)
    old_q = first.posture_target.joint_pos_target
    if max(abs(q[k] - old_q[k]) for k in old_q) > 0.01:
        raise ValueError("reviewed route is not valid for this initial joint state")
    old_position = first.centroidal_target.com_pos_world
    if (
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(old_position, measured.body_pose_world[:3])
        )
        > 0.01**2
    ):
        raise ValueError("reviewed route is not valid for this initial body state")
    first.centroidal_target.com_pos_world = tuple(measured.body_pose_world[:3])
    first.centroidal_target.body_orientation_world = tuple(measured.body_pose_world[3:])
    first.posture_target.joint_pos_target = q
    observed = {o.object_id: o.pose_world for o in context.observation.object_states}
    for target in first.object_targets:
        if target.object_id in observed:
            target.pose_target_world = observed[target.object_id]
    return SimpleNamespace(
        task=deepcopy(context.task_spec),
        morphology=deepcopy(reviewed_bundle.morphology),
        contact_candidate_set=deepcopy(reviewed_bundle.contact_candidate_set),
        phase_trajectories=phases,
        provenance=dict(
            execution_semantics="request_selected_reviewed_geometric_seed",
            request=request.to_dict(),
            observation_snapshot=context.catalog.snapshot_hash,
            reviewed_phase_hash=stable_hash(
                {k: v.to_dict() for k, v in reviewed_bundle.phase_trajectories.items()}
            ),
            reviewed_provenance=deepcopy(reviewed_bundle.provenance),
            teacher_geometry_reused=True,
            teacher_phase_or_action_supervision=False,
            measured_initial_boundary_applied=True,
        ),
    )


def vertical_clearance_retreat(phases, *, offset_m=0.1):
    """After release, clear the payload upwards before holding position.

    A fixed world-X retreat can move a rotated assembly through the payload.
    This tabletop profile has open space above the released object; the full
    collision checker still admits or rejects the resulting path.
    """
    end = phases["release"].knots[-1]
    start, _, q, _ = _complete_task_knot_state(end)
    goal = (*start[:2], start[2] + offset_m, *start[3:])
    obj = next(o for o in end.object_targets if o.pose_target_world is not None)
    for phase, a, b in [("retreat", start, goal), ("settle", goal, goal)]:
        phases[phase] = _build_complete_task_phase_trajectory(
            template=end,
            phase=phase,
            duration_s=phases[phase].horizon_s,
            dt_s=0.1,
            body_start=a,
            body_end=b,
            q_start=q,
            q_end=q,
            object_id=obj.object_id,
            object_start=obj.pose_target_world,
            object_end=obj.pose_target_world,
            contact_schedule_state="inactive",
            anchor_pose_translation_origin=obj.pose_target_world,
            contract_version=phases["release"].contract_version,
        )
    return phases


def complete_request_phases(grasp_phases, task):
    phases = materialize_order9_r1_complete_task_phases(
        phase_trajectories=grasp_phases,
        task_spec=task,
        lift_clearance_m=0.1,
        retreat_offset_m=0.1,
        phase_duration_s={
            "approach": grasp_phases["approach"].horizon_s,
            "contact_acquisition": grasp_phases["contact_acquisition"].horizon_s,
            **{phase: 3.0 for phase in ("lift", "transport", "place", "release", "retreat", "settle")},
        },
    )
    contact = grasp_phases["contact_acquisition"]
    source = contact.knots[-1].object_targets[0].pose_target_world
    goal = phases["place"].knots[-1].object_targets[0].pose_target_world
    # R1's admitted release rises 300 mm, independently of the payload's
    # 100 mm lift. Using the lift height here loses clearance for the moving
    # rotor/gimbal bodies while the grasp joints open.
    phases["release"] = _build_reversed_accepted_phase_trajectory(
        source=contact,
        phase="release",
        duration_s=phases["release"].horizon_s,
        dt_s=0.1,
        source_object_pose=source,
        target_object_pose=goal,
        added_clearance_start_m=0.0,
        added_clearance_end_m=0.3,
        contact_schedule_state="release",
    )
    return vertical_clearance_retreat(phases)


class _ContactGoalInitializationError(RuntimeError):
    """The selected group's initial contact IK exhausted its iteration budget."""


class _PregraspGoalInitializationError(ValueError):
    """The nominal open-grasp intermediate target has no IK solution."""


class _PregraspLocalRouteError(_PregraspGoalInitializationError):
    """A safe open-grasp pose has no certified local route to contact."""


def _plan_with_pregrasp_clearance(context, request, **kwargs):
    """Shorten an unreachable or locally disconnected open-grasp target.

    Every candidate still traverses the unchanged full collision/path checks.
    This neither changes the selected contact nor relaxes any safety margin.
    """
    started = time.monotonic()
    deadline = kwargs["deadline_s"]
    nominal = Order9ArticulatedTeacherConfig().pregrasp_clearance_m
    failure = None
    for index in range(3):
        remaining = deadline - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("request geometric planning deadline")
        attempt = dict(kwargs, deadline_s=remaining)
        if index:
            attempt["pregrasp_clearance_m"] = nominal / (2 ** index)
        try:
            result = _plan_request_geometry(context, request, **attempt)
        except _PregraspGoalInitializationError as error:
            failure = error
            continue
        if index:
            result.provenance["pregrasp_clearance_retry"] = dict(
                version="bounded_pregrasp_clearance_v2", attempts=index + 1,
                nominal_clearance_m=nominal, accepted_clearance_m=attempt["pregrasp_clearance_m"],
                maximum_attempts=3, collision_margin_unchanged=True,
            )
            result.provenance["planning_seconds"] = time.monotonic() - started
        return result
    raise failure


def plan_request_geometry(
    context, request, *, joint_seed=None, base_seed=None, deadline_s=480.0
):
    """Plan one selected binding, with bounded retries of its contact posture."""
    started = time.monotonic()
    try:
        return _plan_with_pregrasp_clearance(
            context, request, joint_seed=joint_seed, base_seed=base_seed,
            deadline_s=deadline_s,
        )
    except _ContactGoalInitializationError as error:
        failure = error
    morphology = context.scene.morphology_graph
    physical = context.physical_model
    joint_ids = ordered_global_dock_joint_ids(morphology, physical)
    seeds = _joint_limit_branch_seeds(
        _global_joint_limits(morphology, physical, joint_ids)
    )
    measured_base = next(
        state.pose_world for state in context.scene.runtime_observation.module_states
        if state.module_id == morphology.base_module_id
    )
    config = ArticulatedIKConfig()
    retry_config = replace(
        config, maximum_iterations=4 * config.maximum_iterations, preserve_base_tilt=True
    )
    for index, seed in enumerate(seeds):
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 0.0:
            raise TimeoutError("request geometric planning deadline")
        try:
            # The same seed reaches contact IK AND configuration-space planning.
            result = _plan_with_pregrasp_clearance(
                context, request, joint_seed=seed, base_seed=measured_base,
                deadline_s=remaining, initial_ik_config=retry_config,
            )
        except (_ContactGoalInitializationError, ValueError) as error:
            failure = error
            continue
        result.provenance["planning_seconds"] = time.monotonic() - started
        result.provenance["contact_goal_initialization_retry"] = dict(
            version="bounded_flying_contact_goal_v2", branch_index=index,
            maximum_branches=len(seeds), maximum_iterations=retry_config.maximum_iterations,
            joint_seed_hash=stable_hash(seed), base_seed_hash=stable_hash(measured_base),
        )
        return result
    raise ValueError(str(failure)) from failure


def _plan_request_geometry(
    context, request, *, joint_seed=None, base_seed=None, deadline_s=480.0,
    initial_ik_config=None, pregrasp_clearance_m=None,
):
    context.catalog.resolve(request)
    if context.execution_state.plan_id is not None or request.transition_id is not None:
        raise ValueError("initial geometry planner requires a new contact request")
    physical = context.physical_model
    task = context.task_spec
    start = time.monotonic()
    candidates = deepcopy(context.scene.contact_candidate_set)
    # Catalog geometry is grounded in the actual observed object pose. Planner
    # TaskSpec source poses must agree; actor/task inputs remain immutable.
    task = deepcopy(task)
    observed = {o.object_id: o.pose_world for o in context.observation.object_states}
    for obj in task.scene.objects:
        if obj.object_id in observed:
            obj.pose_world = observed[obj.object_id]
    scene = compile_high_level_context(task, context.scene.morphology_graph, candidates)
    observation = deepcopy(context.scene.runtime_observation)

    # WholeStructureKinematics calls these "module roots", but its internal
    # _module_link_poses already re-roots URDF transforms at baselink (fc).
    # Measured module poses therefore pass through unchanged.
    class CompletePoseSeedIK(ArticulatedContactIKSolver):
        def solve(self, **kwargs):
            if (
                base_seed is not None
                and kwargs.get("initial_joint_positions_rad") is not None
                and kwargs.get("initial_base_pose_world") is None
            ):
                kwargs["initial_base_pose_world"] = base_seed
            solution = super().solve(**kwargs)
            if not solution.feasible:
                raise _ContactGoalInitializationError(
                    "selected contact goal IK did not converge: "
                    f"position={solution.maximum_position_error_m:.6g},"
                    f"normal={solution.maximum_normal_error_rad:.6g}"
                )
            # Resolve the contact endpoint before constructing its pregrasp.
            # Refining it only after approach can select a different redundant
            # posture branch and leave no short collision-free acquisition.
            from amsrr.training.order9_posture_resolver import (
                Order9PostureTrajectoryResolver, Order9PostureResolverConfig)
            from amsrr.feasibility.articulated_reachability import resolve_mesh_backed_anchor_references
            from amsrr.training.order9_articulated_teacher import _free_pose_contact_collision_goal
            if not hasattr(self, "_collision_resolver"):
                self._collision_resolver = Order9PostureTrajectoryResolver(physical,
                    config=Order9PostureResolverConfig(collision_margin_m=.005),
                    collision_object=build_order9_c3_posture_collision_object(task),
                    prefer_native_solver=True, require_native_solver=True)
            solver = self._collision_resolver.ik_solver
            obstacle = build_order9_c3_posture_collision_object(task)
            anchor_ids = tuple(sorted(solution.anchor_poses_world))
            refs = resolve_mesh_backed_anchor_references(scene.morphology_graph, physical, anchor_ids)
            objects = [o for o in task.scene.objects if o.object_id in observed]
            object_pose = objects[0].pose_world
            scenes = [("object", object_pose, obstacle.size_m, anchor_ids),
                *[(b.box_id, b.pose_world, b.size_m, ()) for b in obstacle.environment_boxes]]
            accepted = True
            for _, pose, size, allowed in scenes:
                solver.set_collision_scene(morphology=scene.morphology_graph,
                    object_pose_world=pose, object_size_m=size, allowed_anchor_ids=allowed)
                accepted &= solver.check_configuration(morphology=scene.morphology_graph,
                    centroidal_pose_world=solution.centroidal_pose_world,
                    joint_positions_rad=solution.joint_positions_rad, exact=False,
                    margin_m=.005, ground_plane_z_m=obstacle.ground_plane_z_m)["accepted"]
            if not accepted and not self.config.preserve_base_tilt:
                goal = _free_pose_contact_collision_goal(morphology=scene.morphology_graph,
                    physical_model=physical, solver=solver, references=refs,
                    desired_q=solution.joint_positions_rad,
                    desired_centroidal_pose=solution.centroidal_pose_world,
                    anchor_targets=solution.anchor_poses_world, scenes=scenes,
                    ground_plane_z_m=obstacle.ground_plane_z_m)
                if goal is not None:
                    fk = solver.kinematics.forward(scene.morphology_graph, physical,
                        goal.joint_positions_rad, goal.base_pose_world, refs)
                    from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
                    from amsrr.training.order9_articulated_teacher import _centroidal_pose_from_fk
                    centroidal = _centroidal_pose_from_fk(context=scene, physical_model=physical,
                        q=goal.joint_positions_rad, module_root_poses_world=fk.module_root_poses_world,
                        source_observation=observation, builder=RigidBodyControlModelBuilder())
                    import numpy as np
                    from scipy.spatial.transform import Rotation
                    position_errors, normal_errors = [], []
                    for assignment in kwargs["assignments"]:
                        if assignment.anchor_id not in fk.anchor_poses_world:
                            continue
                        candidate = kwargs["candidates"][assignment.candidate_id]
                        actual = fk.anchor_poses_world[assignment.anchor_id]
                        position_errors.append(float(np.linalg.norm(np.array(actual[:3]) - candidate.contact_pose_world[:3])))
                        target_normal = -np.asarray(candidate.normal_world)
                        target_normal /= np.linalg.norm(target_normal)
                        actual_normal = Rotation.from_quat(actual[3:]).as_matrix()[:, 0]
                        normal_errors.append(float(np.arccos(np.clip(actual_normal @ target_normal, -1., 1.))))
                    position_error, normal_error = max(position_errors), max(normal_errors)
                    if (position_error > self.config.contact_position_tolerance_m
                            or normal_error > self.config.contact_normal_tolerance_rad):
                        raise _ContactGoalInitializationError("collision refinement violated original candidate point/normal tolerance")
                    solution = replace(solution, maximum_position_error_m=position_error,
                        maximum_normal_error_rad=normal_error, joint_positions_rad=goal.joint_positions_rad,
                        base_pose_world=goal.base_pose_world, centroidal_pose_world=centroidal,
                        anchor_poses_world=fk.anchor_poses_world,
                        solver_version=solution.solver_version + ":collision_clear_free_pose_v1")
            return solution

    teacher_config = Order9ArticulatedTeacherConfig(
        preferred_candidate_group_id=request.contact_group_id,
        maximum_candidate_group_attempts=1,
        collision_margin_m=0.005,
    )
    if pregrasp_clearance_m is not None:
        teacher_config = replace(teacher_config, pregrasp_clearance_m=pregrasp_clearance_m)
    planner = Order9ArticulatedTrajectoryTeacher(
        physical,
        config=teacher_config,
        ik_solver=CompletePoseSeedIK(physical, config=initial_ik_config),
        collision_object=build_order9_c3_posture_collision_object(task),
    )
    windows = []
    elapsed = 0.0
    for phase in ("approach", "contact_acquisition"):
        observation.task_progress = TaskProgressState(phase_label=phase)
        reached = False
        for _ in range(64):
            if time.monotonic() - start > deadline_s:
                raise TimeoutError("request geometric planning deadline")
            current = replace(scene, runtime_observation=observation)
            try:
                plan = planner.plan(
                    current,
                    initial_object_poses_world=observed,
                    contact_goal_joint_seed_positions_rad=joint_seed,
                    configuration_goal_joint_seed_positions_rad=joint_seed,
                )
            except SchemaValidationError as error:
                if "bounded deterministic local contact search exhausted" in str(error):
                    raise _PregraspLocalRouteError(str(error)) from error
                if "could not resolve its collision-clear pregrasp configuration" not in str(error):
                    raise
                raise _PregraspGoalInitializationError(str(error)) from error
            if plan.candidate_group_id != request.contact_group_id:
                raise ValueError("planner replaced selected group")
            windows.append(Order9C3NominalWindow(len(windows), phase, elapsed, plan))
            observation = _ideal_endpoint_observation(
                context=current, plan=plan, physical_model=physical, next_phase=phase
            )
            elapsed += plan.trajectory.horizon_s
            print(
                "REQUEST_PLAN_WINDOW",
                phase,
                len(windows),
                elapsed,
                plan.phase_target_reached,
                flush=True,
            )
            if plan.phase_target_reached:
                reached = True
                break
        if not reached:
            raise ValueError("request planner exhausted finite horizon windows")
    phases = complete_request_phases(
        _phase_trajectories_from_nominal_windows(windows), task
    )
    return SimpleNamespace(
        task=task,
        morphology=scene.morphology_graph,
        contact_candidate_set=candidates,
        phase_trajectories=phases,
        provenance=dict(
            execution_semantics="request_selected_fresh_geometry",
            request=request.to_dict(),
            observation_snapshot=context.catalog.snapshot_hash,
            joint_seed_hash=None if joint_seed is None else stable_hash(joint_seed),
            base_seed_hash=None if base_seed is None else stable_hash(base_seed),
            teacher_trajectory_replay=False,
            planning_seconds=time.monotonic() - start,
        ),
    )
