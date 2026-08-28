from __future__ import annotations

import pytest

from amsrr.feasibility.articulated_reachability import (
    REACHABILITY_JOINT_LIMIT_CODE,
    ArticulatedIKSolution,
    ArticulatedTrajectoryReachabilityEvaluator,
)
from amsrr.training import order9_articulated_teacher as teacher_module
from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.runtime import RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherConfig,
    Order9ArticulatedTrajectoryTeacher,
    _ConfigurationRouteCache,
    _approach_vertical_speed_limit_mps,
    _candidate_group_attempts,
    _contact_solution_body_tilt_admitted,
    _point_to_segment_distance,
    _phase_assignment_states,
    _resume_configuration_route,
    _rolling_progress_fraction,
    _solve_contact_with_joint_limit_reserve_fallback,
    _validated_contact_goal_joint_seed,
)
from amsrr.training.order9_configuration_space_planner import (
    DeterministicOrder9ConfigurationSpacePlanner,
    Order9ConfigurationState,
)
from amsrr.training.order9_c3_teacher import _neutral_runtime_observation
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_posture_hard_gate import (
    Order9PostureResolvingHardChecker,
)
from amsrr.utils.hashing import stable_hash


def test_contact_configuration_corridor_distance_uses_finite_segment() -> None:
    assert _point_to_segment_distance(
        (0.5, 0.02, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ) == pytest.approx(0.02)
    assert _point_to_segment_distance(
        (1.2, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ) == pytest.approx(0.2)


def test_rolling_progress_accounts_for_cubic_peak_and_vertical_limit() -> None:
    progress = _rolling_progress_fraction(
        start_q={"module_0:yaw": 0.0},
        target_q={"module_0:yaw": 0.0},
        start_base_pose=(0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0),
        target_base_pose=(0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
        horizon_s=3.0,
        joint_velocity_limit_rad_s=1.0,
        base_translation_speed_limit_mps=1.0,
        base_vertical_speed_limit_mps=0.06,
        base_rotation_speed_limit_rad_s=1.0,
        interpolation_peak_rate_scale=1.5,
    )

    # Limit the sparse endpoint delta to 0.12 m so the cubic blend's 1.5x
    # derivative peak remains at 0.06 m/s.
    assert progress == pytest.approx(0.4)


def test_articulated_teacher_rejects_subunit_interpolation_peak_scale() -> None:
    with pytest.raises(ValueError, match="peak_rate_scale"):
        Order9ArticulatedTeacherConfig(trajectory_interpolation_peak_rate_scale=0.99)


def test_contact_solution_tilt_gate_runs_on_ik_solution_pose() -> None:
    upright = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ninety_degrees = (
        0.0,
        0.0,
        0.0,
        2.0**-0.5,
        0.0,
        0.0,
        2.0**-0.5,
    )
    solution = ArticulatedIKSolution(
        feasible=True,
        joint_positions_rad={},
        base_pose_world=ninety_degrees,
        centroidal_pose_world=ninety_degrees,
        anchor_poses_world={},
        maximum_position_error_m=0.0,
        maximum_normal_error_rad=0.0,
        iterations=1,
    )

    assert _contact_solution_body_tilt_admitted(solution, None) is True
    assert _contact_solution_body_tilt_admitted(solution, 1.0) is False
    upright_solution = ArticulatedIKSolution(
        feasible=True,
        joint_positions_rad={},
        base_pose_world=upright,
        centroidal_pose_world=upright,
        anchor_poses_world={},
        maximum_position_error_m=0.0,
        maximum_normal_error_rad=0.0,
        iterations=1,
    )
    assert _contact_solution_body_tilt_admitted(upright_solution, 1.0) is True


@pytest.mark.parametrize("value", [0.0, 3.2, float("nan")])
def test_articulated_teacher_rejects_invalid_contact_solution_tilt(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="maximum_contact_solution_body_tilt"):
        Order9ArticulatedTeacherConfig(maximum_contact_solution_body_tilt_rad=value)


def test_approach_vertical_speed_transitions_into_terminal_zone() -> None:
    config = Order9ArticulatedTeacherConfig(
        maximum_approach_vertical_speed_mps=0.06,
        maximum_terminal_approach_vertical_speed_mps=0.03,
        terminal_approach_slowdown_height_m=0.24,
        rolling_horizon_s=3.0,
        trajectory_interpolation_peak_rate_scale=1.5,
    )

    assert _approach_vertical_speed_limit_mps(
        start_base_z=0.80,
        target_base_z=0.22,
        terminal_base_z=0.22,
        horizon_s=3.0,
        config=config,
    ) == pytest.approx(0.06)
    assert _approach_vertical_speed_limit_mps(
        start_base_z=0.50,
        target_base_z=0.22,
        terminal_base_z=0.22,
        horizon_s=3.0,
        config=config,
    ) == pytest.approx(0.05)
    assert _approach_vertical_speed_limit_mps(
        start_base_z=0.46,
        target_base_z=0.22,
        terminal_base_z=0.22,
        horizon_s=3.0,
        config=config,
    ) == pytest.approx(0.03)


def test_approach_vertical_speed_keeps_cruise_for_ascent() -> None:
    config = Order9ArticulatedTeacherConfig()
    assert _approach_vertical_speed_limit_mps(
        start_base_z=0.22,
        target_base_z=0.50,
        terminal_base_z=0.22,
        horizon_s=3.0,
        config=config,
    ) == pytest.approx(0.06)


def test_articulated_teacher_rejects_terminal_speed_above_cruise() -> None:
    with pytest.raises(ValueError, match="terminal_approach"):
        Order9ArticulatedTeacherConfig(
            maximum_approach_vertical_speed_mps=0.06,
            maximum_terminal_approach_vertical_speed_mps=0.07,
        )


def test_joint_reserve_fallback_does_not_retry_a_feasible_primary(
    monkeypatch,
) -> None:
    calls = 0
    primary = ArticulatedIKSolution(
        feasible=True,
        joint_positions_rad={"module_0:yaw": -0.96},
        base_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        centroidal_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        anchor_poses_world={},
        maximum_position_error_m=0.0,
        maximum_normal_error_rad=0.0,
        iterations=1,
    )

    class Solver:
        def solve(self, **_kwargs):
            nonlocal calls
            calls += 1
            return primary

    monkeypatch.setattr(
        teacher_module,
        "ordered_global_dock_joint_ids",
        lambda _morphology, _physical: ("module_0:yaw",),
    )
    monkeypatch.setattr(
        teacher_module,
        "_global_joint_limits",
        lambda _morphology, _physical, _ordered: {"module_0:yaw": (-1.0, 1.0)},
    )

    result = _solve_contact_with_joint_limit_reserve_fallback(
        solver=Solver(),
        morphology=object(),
        physical_model=object(),
        assignments=(),
        candidates={},
        initial_joint_positions_rad=None,
        minimum_normalized_joint_limit_reserve=0.01,
    )

    assert result is primary
    assert calls == 1


def test_joint_reserve_fallback_selects_largest_reserve_after_failure(
    monkeypatch,
) -> None:
    pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def solution(feasible: bool, q: float) -> ArticulatedIKSolution:
        return ArticulatedIKSolution(
            feasible=feasible,
            joint_positions_rad={"module_0:yaw": q},
            base_pose_world=pose,
            centroidal_pose_world=pose,
            anchor_poses_world={},
            maximum_position_error_m=0.0 if feasible else 1.0,
            maximum_normal_error_rad=0.0 if feasible else 1.0,
            iterations=1,
        )

    attempts = iter(
        (
            solution(False, 0.0),
            solution(True, -0.6),
            solution(True, 0.2),
        )
    )
    calls = 0

    class Solver:
        def solve(self, **_kwargs):
            nonlocal calls
            calls += 1
            return next(attempts)

    monkeypatch.setattr(
        teacher_module,
        "ordered_global_dock_joint_ids",
        lambda _morphology, _physical: ("module_0:yaw",),
    )
    monkeypatch.setattr(
        teacher_module,
        "_global_joint_limits",
        lambda _morphology, _physical, _ordered: {"module_0:yaw": (-1.0, 1.0)},
    )

    result = _solve_contact_with_joint_limit_reserve_fallback(
        solver=Solver(),
        morphology=object(),
        physical_model=object(),
        assignments=(),
        candidates={},
        initial_joint_positions_rad=None,
        minimum_normalized_joint_limit_reserve=0.01,
    )

    assert calls == 3
    assert result.joint_positions_rad["module_0:yaw"] == pytest.approx(0.2)


def _system(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = build_grasp_carry_variant_design_output(
        task,
        built.irg,
        physical,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    morphology = design.target_morphology
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    context = HighLevelPolicyContext(
        built.irg,
        envelope,
        morphology,
        candidates,
        runtime_observation=_neutral_runtime_observation(
            morphology,
            physical,
            task,
        ),
    )
    return task, physical, context


def _collision_object(task: TaskSpec) -> Order9PostureCollisionObject:
    target = task.scene.objects[0]
    geometry = next(
        item
        for item in task.scene.geometry_library
        if item.geometry_id == target.geometry_id
    )
    return Order9PostureCollisionObject(
        object_id=target.object_id,
        size_m=tuple(geometry.primitive_params["size_m"]),
        initial_pose_world=tuple(target.pose_world),
    )


def test_contact_goal_joint_seed_is_morphology_bound(
    grasp_carry_dict: dict,
) -> None:
    _task, physical, context = _system(grasp_carry_dict)
    zero_seed = _validated_contact_goal_joint_seed(
        context,
        physical,
        None,
    )

    assert zero_seed
    assert set(zero_seed.values()) == {0.0}
    reviewed_seed = {joint_id: 0.125 for joint_id in zero_seed}
    assert (
        _validated_contact_goal_joint_seed(
            context,
            physical,
            reviewed_seed,
        )
        == reviewed_seed
    )

    with pytest.raises(
        SchemaValidationError,
        match="seed identities differ",
    ):
        _validated_contact_goal_joint_seed(
            context,
            physical,
            {"unknown_joint": 0.0},
        )


def _production_teacher(
    task: TaskSpec,
    physical,
) -> Order9ArticulatedTrajectoryTeacher:
    return Order9ArticulatedTrajectoryTeacher(
        physical,
        collision_object=_collision_object(task),
    )


def test_articulated_teacher_resumes_and_advances_cached_route() -> None:
    planner = DeterministicOrder9ConfigurationSpacePlanner()
    start = Order9ConfigurationState(
        base_pose_world=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"module_0:yaw": 0.0},
    )
    waypoint = Order9ConfigurationState(
        base_pose_world=(0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"module_0:yaw": 0.5},
    )
    goal = Order9ConfigurationState(
        base_pose_world=(1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
        joint_positions_rad={"module_0:yaw": 1.0},
    )
    identity = ("approach", "morphology", (), "scene", "desired")
    cache = _ConfigurationRouteCache(
        identity=identity,
        goal=goal,
        remaining_states=(waypoint, goal),
        source_method="deterministic_overhead_coordinate_bridge",
    )

    resumed = _resume_configuration_route(
        cache=cache,
        identity=identity,
        start=start,
        goal=goal,
        planner=planner,
        collision_free=lambda _state: True,
        ordered_ids=("module_0:yaw",),
    )

    assert resumed is not None
    assert resumed.method.startswith("cached_route:")
    assert resumed.states == (start, waypoint, goal)

    advanced = _resume_configuration_route(
        cache=cache,
        identity=identity,
        start=waypoint,
        goal=goal,
        planner=planner,
        collision_free=lambda _state: True,
        ordered_ids=("module_0:yaw",),
    )

    assert advanced is not None
    assert advanced.states == (waypoint, goal)


def test_articulated_teacher_emits_complete_joint_com_and_anchor_targets(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    plan = _production_teacher(task, physical).plan(
        context,
        initial_object_poses_world={
            obj.object_id: obj.pose_world for obj in task.scene.objects
        },
    )

    assert plan.ik_solution.feasible
    assert plan.ik_solution.maximum_position_error_m <= 0.005
    assert plan.task_phase == "approach"
    assert plan.raw_trajectory.horizon_s <= 3.0
    assert all(
        assignment.schedule_state == "approach"
        for knot in plan.raw_trajectory.knots
        for assignment in knot.contact_assignments
    )
    assert len(plan.trajectory.knots) > len(plan.raw_trajectory.knots)
    assert all(
        knot.posture_target is not None
        and knot.posture_target.joint_pos_target is None
        and knot.posture_target.joint_vel_target is None
        for knot in plan.raw_trajectory.knots
    )
    assert plan.posture_resolution.evidence.raw_trajectory_hash == stable_hash(
        plan.raw_trajectory.to_dict()
    )
    assert plan.posture_resolution.evidence.resolved_trajectory_hash == stable_hash(
        plan.trajectory.to_dict()
    )
    assert (
        plan.posture_resolution.evidence.maximum_joint_rate_rad_s
        <= plan.posture_resolution.evidence.joint_rate_limit_rad_s
    )
    assert (
        plan.posture_resolution.evidence.nominal_joint_seed_trajectory_hash is not None
    )
    expected_joint_ids = set(plan.ik_solution.joint_positions_rad)
    approach_joint_reference = dict(
        plan.trajectory.knots[0].posture_target.joint_pos_target
    )
    for knot in plan.trajectory.knots:
        assert knot.centroidal_target is not None
        assert knot.centroidal_target.com_pos_world is not None
        assert knot.centroidal_target.body_orientation_world is not None
        assert knot.posture_target is not None
        assert set(knot.posture_target.joint_pos_target or {}) == expected_joint_ids
        assert set(knot.posture_target.joint_vel_target or {}) == expected_joint_ids
    assert plan.configuration_space_plan is not None
    assert plan.configuration_space_plan.collision_check_count > 0
    assert plan.posture_resolution.evidence.collision_gate_status == "accepted"
    joint_configuration_changed = any(
        knot.posture_target.joint_pos_target
        != pytest.approx(approach_joint_reference, abs=1.0e-9)
        for knot in plan.trajectory.knots[1:]
    )
    planned_states = plan.configuration_space_plan.states
    base_configuration_changed = any(
        abs(
            float(planned_states[-1].base_pose_world[index])
            - float(planned_states[0].base_pose_world[index])
        )
        > 1.0e-9
        for index in range(7)
    )
    assert joint_configuration_changed or base_configuration_changed

    checked_context = HighLevelPolicyContext(
        context.irg,
        context.interaction_envelope,
        context.morphology_graph,
        context.contact_candidate_set,
        runtime_observation=_neutral_runtime_observation(
            context.morphology_graph,
            physical,
            task,
        ),
    )
    evaluations = ArticulatedTrajectoryReachabilityEvaluator(
        physical
    ).evaluate_trajectory(
        context=checked_context,
        trajectory=plan.trajectory,
    )

    assert all(value.feasible for value in evaluations)
    assert all(value.evaluator_version for value in evaluations)

    hard_gate = Order9PostureResolvingHardChecker(
        physical_model=physical,
        resolved_checker=ContactWrenchTrajectoryFeasibilityChecker(
            config=ContactWrenchTrajectoryCheckerConfig(
                evaluation_mode="warmup_proxy",
                require_qp_evaluation=False,
                require_collision_evaluation=False,
                require_wrench_evaluation=False,
                require_reachability_evaluation=True,
            ),
            reachability_evaluator=(
                ArticulatedTrajectoryReachabilityEvaluator(physical)
            ),
        ),
    )
    raw_hash = stable_hash(plan.raw_trajectory.to_dict())
    hard_gate_result = hard_gate.check(
        plan.raw_trajectory,
        checked_context,
    )
    execution = hard_gate.execution_trajectory(
        plan.raw_trajectory,
        checked_context,
    )
    assert hard_gate_result.feasible
    assert len(hard_gate_result.knot_results) == len(plan.raw_trajectory.knots)
    assert len(execution.knots) > len(plan.raw_trajectory.knots)
    assert stable_hash(plan.raw_trajectory.to_dict()) == raw_hash
    assert hard_gate_result.metadata["raw_proposal_mutated"] is False
    assert hard_gate_result.metadata["posture_resolution_succeeded"] is True
    repeated_hard_gate_result = hard_gate.check(
        plan.raw_trajectory,
        checked_context,
    )
    assert repeated_hard_gate_result.to_dict() == hard_gate_result.to_dict()
    assert all(
        hard_gate.resolution_evidence(
            plan.raw_trajectory,
            checked_context,
        ).solver_cache_hits
    )

    invalid_raw = ContactWrenchTrajectory.from_dict(plan.raw_trajectory.to_dict())
    invalid_raw.knots[0].posture_target.joint_pos_target = {
        joint_id: 0.0 for joint_id in expected_joint_ids
    }
    with pytest.raises(
        SchemaValidationError,
        match="raw pi_H trajectory must not contain",
    ):
        Order9PostureTrajectoryResolver(physical).resolve(
            context=context,
            raw_trajectory=invalid_raw,
            initial_joint_positions_rad=dict(
                plan.trajectory.knots[0].posture_target.joint_pos_target
            ),
        )


def test_reachability_checker_rejects_mutated_teacher_joint_target(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    plan = _production_teacher(task, physical).plan(
        context,
        initial_object_poses_world={
            obj.object_id: obj.pose_world for obj in task.scene.objects
        },
    )
    mutated = ContactWrenchTrajectory.from_dict(plan.trajectory.to_dict())
    active = mutated.knots[-1]
    joint_id = next(iter(active.posture_target.joint_pos_target))
    active.posture_target.joint_pos_target[joint_id] = 10.0
    checked_context = HighLevelPolicyContext(
        context.irg,
        context.interaction_envelope,
        context.morphology_graph,
        context.contact_candidate_set,
        runtime_observation=_neutral_runtime_observation(
            context.morphology_graph,
            physical,
            task,
        ),
    )

    evaluations = ArticulatedTrajectoryReachabilityEvaluator(
        physical
    ).evaluate_trajectory(
        context=checked_context,
        trajectory=mutated,
    )

    assert any(
        REACHABILITY_JOINT_LIMIT_CODE in value.violation_codes for value in evaluations
    )


def test_articulated_teacher_can_pin_one_candidate_group(
    grasp_carry_dict: dict,
) -> None:
    _task, _physical, context = _system(grasp_carry_dict)
    available = sorted(
        proposal.group_id
        for proposal in context.contact_candidate_set.group_proposals
        if not proposal.group_violation_codes
    )
    assert available

    attempts = _candidate_group_attempts(
        context.contact_candidate_set,
        maximum=8,
        preferred_group_id=available[0],
    )

    assert len(attempts) == 1
    assert attempts[0][0] == available[0]
    assert [proposal.group_id for proposal in attempts[0][1].group_proposals] == [
        available[0]
    ]
    with pytest.raises(
        SchemaValidationError,
        match="candidate group is unavailable",
    ):
        _candidate_group_attempts(
            context.contact_candidate_set,
            maximum=8,
            preferred_group_id="missing-group",
        )
    with pytest.raises(ValueError, match="must be non-empty"):
        Order9ArticulatedTeacherConfig(preferred_candidate_group_id="")


def test_contact_acquisition_uses_attach_schedule_during_articulation() -> None:
    assert _phase_assignment_states(
        "approach",
        knot_count=3,
        target_reached=True,
    ) == ("approach", "approach", "approach")
    assert _phase_assignment_states(
        "contact_acquisition",
        knot_count=3,
        target_reached=False,
    ) == ("attach", "attach", "attach")
    assert _phase_assignment_states(
        "contact_acquisition",
        knot_count=3,
        target_reached=True,
    ) == ("attach", "attach", "maintain")


def test_rolling_teacher_preserves_previous_nominal_reference(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    teacher = _production_teacher(task, physical)
    initial_object_poses = {obj.object_id: obj.pose_world for obj in task.scene.objects}
    initial = teacher.plan(
        context,
        initial_object_poses_world=initial_object_poses,
    )
    measured = dict(initial.trajectory.knots[0].posture_target.joint_pos_target)
    nominal = dict(measured)
    changed_joint = max(
        initial.ik_solution.joint_positions_rad,
        key=lambda joint_id: abs(initial.ik_solution.joint_positions_rad[joint_id]),
    )
    nominal[changed_joint] += 0.01

    replanned = teacher.plan(
        context,
        initial_object_poses_world=initial_object_poses,
        nominal_start_joint_positions_rad=nominal,
    )

    assert replanned.trajectory.knots[
        0
    ].posture_target.joint_pos_target == pytest.approx(nominal, abs=1.0e-6)
    assert (
        replanned.posture_resolution.evidence.initial_joint_state_hash
        == stable_hash(nominal)
    )
    assert (
        replanned.posture_resolution.evidence.initial_joint_state_hash
        != stable_hash(measured)
    )


def test_release_teacher_resolves_an_explicit_open_anchor_reference(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    target = task.scene.objects[0]
    geometry = next(
        item
        for item in task.scene.geometry_library
        if item.geometry_id == target.geometry_id
    )
    teacher = Order9ArticulatedTrajectoryTeacher(
        physical,
        collision_object=Order9PostureCollisionObject(
            object_id=target.object_id,
            size_m=tuple(geometry.primitive_params["size_m"]),
            initial_pose_world=tuple(target.pose_world),
        ),
    )
    initial_object_poses = {obj.object_id: obj.pose_world for obj in task.scene.objects}
    approach = teacher.plan(
        context,
        initial_object_poses_world=initial_object_poses,
    )
    closed = dict(approach.ik_solution.joint_positions_rad)
    opened = {joint_id: 0.0 for joint_id in closed}
    observation = RuntimeObservation.from_dict(context.runtime_observation.to_dict())
    for module_state in observation.module_states:
        prefix = f"module_{module_state.module_id}:"
        module_state.joint_positions = {
            joint_id.removeprefix(prefix): float(value)
            for joint_id, value in closed.items()
            if joint_id.startswith(prefix)
        }
        module_state.joint_velocities = {
            joint_id: 0.0 for joint_id in module_state.joint_positions
        }
    observation.task_progress = TaskProgressState(
        phase_label="release",
        progress_ratio=0.0,
    )
    release_context = HighLevelPolicyContext(
        context.irg,
        context.interaction_envelope,
        context.morphology_graph,
        context.contact_candidate_set,
        runtime_observation=observation,
    )

    release = teacher.plan(
        release_context,
        initial_object_poses_world=initial_object_poses,
        nominal_start_joint_positions_rad=closed,
        release_joint_positions_rad=opened,
    )

    assert release.task_phase == "release"
    assert all(
        assignment.schedule_state == "release"
        for knot in release.raw_trajectory.knots
        for assignment in knot.contact_assignments
    )
    assert all(
        knot.posture_target is not None
        and knot.posture_target.joint_pos_target is None
        and knot.posture_target.joint_vel_target is None
        and set(knot.posture_target.free_anchor_pose_targets)
        == set(release.ik_solution.anchor_poses_world)
        for knot in release.raw_trajectory.knots
    )
    terminal = release.trajectory.knots[-1].posture_target
    assert terminal is not None
    assert set(terminal.joint_pos_target) == set(opened)
    assert sum(abs(float(value)) for value in terminal.joint_pos_target.values()) < sum(
        abs(float(value)) for value in closed.values()
    )
    assert (
        release.posture_resolution.evidence.maximum_anchor_position_error_m
        <= teacher.config.posture_anchor_position_tolerance_m
    )
