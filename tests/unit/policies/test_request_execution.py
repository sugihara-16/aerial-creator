"""Real Holon FK and controller-reference tests; shadow fixtures are NOT physics acceptance."""

from copy import deepcopy
from time import monotonic, sleep
import numpy as np
import pytest

from tests.unit.policies.test_high_level_requests import request_scene, decision
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.qpid_controller import QPIDController
from amsrr.controllers.qp_allocator_interface import VirtualThrustQPAllocator
from amsrr.feasibility.articulated_reachability import (
    ArticulatedTrajectoryReachabilityEvaluator,
)
from amsrr.feasibility.contact_wrench_hybrid import (
    HybridContactWrenchPhysicsEvaluator,
    ShadowKnotObservation,
)
from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
    ContactWrenchTrajectoryCheckerConfig,
)
from amsrr.feasibility.request_plan import RequestPlanChecker
from amsrr.policies.high_level_requests import RequestCatalogBuilder
from amsrr.policies.request_execution import (
    RequestPlanExecutor,
    initial_execution_state,
)
from amsrr.policies.request_runtime import HighLevelRequestRuntime, RequestRuntimeConfig
from amsrr.policies.request_trajectory_planner import (
    ConstrainedRequestPlanner,
    RequestPlanningProblem,
    REQUIRED_PLANNING_CONSTRAINTS,
    JointPlanEvaluator,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import (
    ExecutionGuardSample,
    GeneratedExecutionPlan,
    PlanCheckRecord,
)
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    CentroidalTarget,
    PostureTarget,
    ObjectTarget,
    InteractionKnot,
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
)


def physical_context(parts):
    scene, task, model, state = parts
    q = {
        key: 0.0 for key in ordered_global_dock_joint_ids(scene.morphology_graph, model)
    }
    base = next(
        m.pose_world
        for m in scene.runtime_observation.module_states
        if m.module_id == scene.morphology_graph.base_module_id
    )
    fk = WholeStructureKinematics().forward(scene.morphology_graph, model, q, base, ())
    for m in scene.runtime_observation.module_states:
        m.pose_world = fk.module_root_poses_world[m.module_id]
        m.joint_velocities = dict(m.joint_positions)
    return decision(parts)


def fixture_plan(context, request=None):
    entry = (
        next(e for e in context.catalog.entries if e.transition is None)
        if request is None
        else context.catalog.resolve(request)
    )
    model = RigidBodyControlModelBuilder().build(
        context.scene.morphology_graph,
        context.physical_model,
        context.scene.runtime_observation,
    )
    ids = ordered_global_dock_joint_ids(
        context.scene.morphology_graph, context.physical_model
    )
    observed = {
        f"module_{m.module_id}:{j}": value
        for m in context.observation.module_states
        for j, value in m.joint_positions.items()
    }
    q = {j: observed[j] for j in ids}
    candidates = {
        c.candidate_id: c for c in context.scene.contact_candidate_set.candidates
    }
    assignments = [
        ContactAssignment(
            candidates[cid].slot_id,
            candidates[cid].anchor_id,
            cid,
            candidates[cid].contact_mode,
            "approach",
            [0.0] * 6,
            [0.0] * 6,
            [0.0] * 6,
            wrench_frame="contact",
        )
        for cid in entry.candidate_ids
    ]
    knot = InteractionKnot(
        0.0,
        assignments,
        CentroidalTarget(
            tuple(model.body_pose_world[:3]),
            (0.0, 0.0, 0.0),
            tuple(model.body_pose_world[3:]),
        ),
        PostureTarget(q, {j: 0.0 for j in ids}, {}),
        [
            ObjectTarget(o.object_id, o.pose_world, [0.0] * 6)
            for o in context.observation.object_states
        ],
    )
    end = deepcopy(knot)
    end.t_rel_s = 2.0
    plan = GeneratedExecutionPlan(
        deepcopy(entry.request),
        context.catalog.snapshot_hash,
        context.physical_model.stable_hash(),
        "test_model",
        "test_config",
        context.observation.time_s,
        context.observation.time_s + 2.0,
        ContactWrenchTrajectory(
            2.0,
            0.5,
            [knot, end],
            contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
        ),
    )
    evaluator = JointPlanEvaluator(context)
    for k in plan.trajectory.knots:
        k.posture_target.free_anchor_pose_targets = evaluator.sample(
            plan, time_s=plan.observation_time_s + k.t_rel_s
        ).posture_target.free_anchor_pose_targets
    return plan


class ShadowFixture:
    backend_version = "TEST_ONLY_not_a_physics_backend"

    def rollout(self, *, context, trajectory):
        return [
            ShadowKnotObservation(0.0, 0.0, collision_free_clearance_m=1.0)
            for _ in trajectory.knots
        ]


def fixture_checker(context):
    checker = ContactWrenchTrajectoryFeasibilityChecker(
        config=ContactWrenchTrajectoryCheckerConfig(
            evaluation_mode="production", require_reachability_evaluation=True
        ),
        physics_evaluator=HybridContactWrenchPhysicsEvaluator(
            shadow_backend=ShadowFixture()
        ),
        reachability_evaluator=ArticulatedTrajectoryReachabilityEvaluator(
            context.physical_model
        ),
    )
    return RequestPlanChecker(
        checker,
        sample_dt_s=0.5,
        maximum_samples=12,
        maximum_joint_acceleration_radps2=1.0,
        constraint_evaluator=fixture_extra_constraints,
    )


def fixture_extra_constraints(context, plan, *, deadline):
    """Interface fixture only; not independent physical validation."""
    return {name: [1.0] for name in REQUIRED_PLANNING_CONSTRAINTS}


class NumericalFixtureModel:
    """Tiny feasible/infeasible solver surface, not a Holon motion planner."""

    def __init__(self, *, feasible=True, complete=True):
        self.feasible, self.complete = feasible, complete

    def build_problem(self, context, request, *, deadline):
        plan = fixture_plan(context, request)

        def margins(x):
            result = {name: np.array([1.0]) for name in REQUIRED_PLANNING_CONSTRAINTS}
            result["tracking_margin"] = np.array(
                [x[0] - 0.25, (0.75 if self.feasible else -0.25) - x[0]]
            )
            if not self.complete:
                del result["joint_effort"]
            return result

        return RequestPlanningProblem(
            np.array([0.5]),
            np.array([0.0]),
            np.array([1.0]),
            lambda x: (x[0] - 0.6) ** 2,
            margins,
            lambda x: deepcopy(plan.trajectory),
            "test_surface",
            "test_config",
        )


def valid_factory(context):
    return ConstrainedRequestPlanner(NumericalFixtureModel()), fixture_checker(context)


def blocked_factory(context):
    sleep(30.0)


def test_real_fk_plan_check_and_nominal_controller_references(request_scene):
    context = physical_context(request_scene)
    plan = fixture_plan(context)
    key = next(iter(plan.trajectory.knots[-1].posture_target.joint_pos_target))
    plan.trajectory.knots[-1].posture_target.joint_pos_target[key] += 0.01
    evaluator = JointPlanEvaluator(context)
    plan.trajectory.knots[-1].posture_target.free_anchor_pose_targets = (
        evaluator.sample(plan, time_s=2.0).posture_target.free_anchor_pose_targets
    )
    check = fixture_checker(context).check(plan, context, deadline=monotonic() + 5.0)
    assert check.accepted, check.violation_codes
    executor = RequestPlanExecutor(context.execution_state)
    executor.install(plan, check, context, context)
    request_scene[3].__dict__.update(deepcopy(executor.state.__dict__))
    current = decision(request_scene)
    command = executor.command(current, time_s=current.observation.time_s)
    assert (
        command.joint_position_targets
        == plan.trajectory.knots[0].posture_target.joint_pos_target
    )
    assert (
        command.joint_velocity_targets
        == plan.trajectory.knots[0].posture_target.joint_vel_target
    )
    assert command.desired_body_pose == (
        *plan.trajectory.knots[0].centroidal_target.com_pos_world,
        *plan.trajectory.knots[0].centroidal_target.body_orientation_world,
    )
    assert command.control_contract_version == "centroidal_local_joint_v2"
    assert not command.joint_position_bias and not command.desired_anchor_pose_offsets
    request_scene[0].runtime_observation.time_s = 0.5
    current = decision(request_scene)
    moving_command = executor.command(current, time_s=0.5)
    assert moving_command.joint_position_targets[key] > 0
    output = QPIDController(allocator=VirtualThrustQPAllocator()).compute(
        ControllerContext(
            current.scene.runtime_observation,
            current.scene.morphology_graph,
            current.physical_model,
            evaluator.sample(plan, time_s=0.5),
            moving_command,
        )
    )
    for joint, target in moving_command.joint_position_targets.items():
        assert output.joint_position_targets[joint] == pytest.approx(target)
    # Mutating the caller's copy cannot change what was installed.
    plan.trajectory.knots[0].posture_target.joint_pos_target[
        next(iter(command.joint_position_targets))
    ] += 1.0
    assert (
        executor.command(current, time_s=0.5).joint_position_targets
        == moving_command.joint_position_targets
    )


def test_same_joint_interpolation_has_correct_quintic_derivative(request_scene):
    context = physical_context(request_scene)
    plan = fixture_plan(context)
    key = next(iter(plan.trajectory.knots[0].posture_target.joint_pos_target))
    plan.trajectory.knots[-1].posture_target.joint_pos_target[key] += 0.1
    evaluator = JointPlanEvaluator(context)
    mid = evaluator.sample(plan, time_s=1.0)
    assert mid.posture_target.joint_pos_target[key] == pytest.approx(0.05)
    assert mid.posture_target.joint_vel_target[key] == pytest.approx(0.1 * 1.875 / 2.0)
    eps = 1.0e-5
    q1 = evaluator.sample(plan, time_s=1.0 - eps).posture_target.joint_pos_target[key]
    q2 = evaluator.sample(plan, time_s=1.0 + eps).posture_target.joint_pos_target[key]
    assert (q2 - q1) / (2 * eps) == pytest.approx(
        mid.posture_target.joint_vel_target[key]
    )


def test_transition_cannot_advance_without_fresh_observed_dwell(request_scene):
    context = physical_context(request_scene)
    entry = next(e for e in context.catalog.entries if e.transition is not None)
    plan = fixture_plan(context, entry.request)
    check = PlanCheckRecord(
        plan.stable_hash(), context.catalog.snapshot_hash, True, "TEST_ONLY"
    )
    executor = RequestPlanExecutor(context.execution_state)
    with pytest.raises(SchemaValidationError, match="guard"):
        executor.install(plan, check, context, context)
    assert executor.state.phase_id == context.execution_state.phase_id
    scene, task, model, state = request_scene
    guards = [
        ExecutionGuardSample(gid, 0.0, True, 1.0, "observed_fixture", 0.2)
        for gid in entry.required_guard_ids
    ]
    current = RequestCatalogBuilder().build(
        scene,
        task_spec=task,
        physical_model=model,
        execution_state=state,
        guard_samples=guards,
    )
    executor.install(plan, check, context, current)
    assert executor.state.phase_id == entry.phase_id


def test_plan_mutation_expiry_and_object_only_motion_rejected(request_scene):
    context = physical_context(request_scene)
    plan = fixture_plan(context)
    check = fixture_checker(context).check(plan, context, deadline=monotonic() + 5.0)
    changed = deepcopy(plan)
    changed.trajectory.knots[-1].posture_target.joint_pos_target[
        next(iter(changed.trajectory.knots[-1].posture_target.joint_pos_target))
    ] += 0.01
    executor = RequestPlanExecutor(context.execution_state)
    with pytest.raises(SchemaValidationError, match="acceptance"):
        executor.install(changed, check, context, context)
    obj = request_scene[0].runtime_observation.object_states[0]
    obj.pose_world = (obj.pose_world[0] + 0.01, *obj.pose_world[1:])
    with pytest.raises(SchemaValidationError, match="tracking"):
        executor.install(plan, check, context, decision(request_scene))
    with pytest.raises(SchemaValidationError, match="expired"):
        JointPlanEvaluator(context).sample(plan, time_s=2.01)


def test_solver_distinguishes_found_not_found_missing_constraints_and_timeout(
    request_scene,
):
    context = physical_context(request_scene)
    request = next(e.request for e in context.catalog.entries if e.transition is None)
    for model, expected in [
        (NumericalFixtureModel(), "solution_found"),
        (NumericalFixtureModel(feasible=False), "no_solution_found"),
        (NumericalFixtureModel(complete=False), "invalid_request"),
    ]:
        result = ConstrainedRequestPlanner(model).plan(
            context, request, deadline=monotonic() + 5.0
        )
        assert result.status == expected, result.reason
    assert (
        ConstrainedRequestPlanner(NumericalFixtureModel())
        .plan(context, request, deadline=monotonic() - 1.0)
        .status
        == "timeout"
    )


def test_production_checker_rejects_proxy_and_wrong_initial_joints(request_scene):
    context = physical_context(request_scene)
    with pytest.raises(ValueError, match="production"):
        RequestPlanChecker(
            ContactWrenchTrajectoryFeasibilityChecker(),
            sample_dt_s=0.1,
            maximum_samples=30,
            maximum_joint_acceleration_radps2=1.0,
        )
    plan = fixture_plan(context)
    key = next(iter(plan.trajectory.knots[0].posture_target.joint_pos_target))
    plan.trajectory.knots[0].posture_target.joint_pos_target[key] += 0.01
    result = fixture_checker(context).check(plan, context, deadline=monotonic() + 5.0)
    assert not result.accepted
    assert any("initial joint" in v for v in result.violation_codes)


def test_legacy_shadow_gate_alone_cannot_authorize_new_contract(request_scene):
    context = physical_context(request_scene)
    checker = fixture_checker(context)
    checker.constraint_evaluator = None
    result = checker.check(fixture_plan(context), context, deadline=monotonic() + 5.0)
    assert not result.accepted
    assert "new_contract_constraints_not_evaluated" in result.violation_codes


def test_validation_sample_budget_is_checked_before_allocation(
    request_scene, monkeypatch
):
    context = physical_context(request_scene)
    plan = fixture_plan(context)
    plan.trajectory.horizon_s = plan.trajectory.knots[-1].t_rel_s = (
        plan.valid_until_s
    ) = 1.0e12

    def forbidden(*args, **kwargs):
        pytest.fail("must reject before allocating an unbounded sample grid")

    monkeypatch.setattr("amsrr.feasibility.request_plan.np.linspace", forbidden)
    result = fixture_checker(context).check(plan, context, deadline=monotonic() + 5.0)
    assert not result.accepted
    assert any("sample budget" in code for code in result.violation_codes)


def test_runtime_archives_request_before_worker_and_kills_timeout(
    request_scene, tmp_path
):
    context = physical_context(request_scene)

    # Select only continuing requests so this test exercises the hung worker.
    class ContinuePolicy:
        policy_version = "TEST_ONLY"

        def rank(self, context):
            return [e.request for e in context.catalog.entries if e.transition is None]

    runtime = HighLevelRequestRuntime(
        worker_factory=blocked_factory,
        executor=RequestPlanExecutor(context.execution_state),
        archive_directory=tmp_path,
        policy=ContinuePolicy(),
        config=RequestRuntimeConfig(wall_budget_s=0.1, maximum_candidates=1),
    )
    with runtime:
        identity = runtime.begin(context)
        assert (tmp_path / f"{identity}.request.json").exists()
        started = monotonic()
        while (result := runtime.poll(context)) is None:
            sleep(0.01)
        assert result.status == "timeout"
        assert monotonic() - started < 1.0
        assert result.execution_success is None
        assert runtime._process is None


def test_real_process_route_checks_installs_and_archives_search_separately(
    request_scene, tmp_path
):
    context = physical_context(request_scene)
    with HighLevelRequestRuntime(
        worker_factory=valid_factory,
        executor=RequestPlanExecutor(context.execution_state),
        archive_directory=tmp_path,
        config=RequestRuntimeConfig(wall_budget_s=10.0, maximum_candidates=1),
    ) as runtime:
        identity = runtime.begin(context)
        while (result := runtime.poll(context)) is None:
            sleep(0.01)
        assert result.status == "searched_candidate_accepted", result.to_dict()
        assert (
            result.accepted_rank > 0
        )  # Earlier requests had unobserved transition guards.
        assert result.execution_success is None
        attempt = next(a for a in result.attempts if a.rank == result.accepted_rank)
        assert attempt.check.accepted
        assert runtime.executor.state.plan_id == attempt.plan.stable_hash()
        assert (tmp_path / f"{identity}.candidate-{result.accepted_rank}.json").exists()


def test_checker_cannot_modify_a_validated_trajectory(request_scene):
    context = physical_context(request_scene)
    checker = fixture_checker(context)
    original = checker.checker.check

    def mutating(trajectory, scene):
        result = original(trajectory, scene)
        trajectory.knots[-1].centroidal_target.com_pos_world = (100.0, 0.0, 0.0)
        return result

    checker.checker.check = mutating
    result = checker.check(fixture_plan(context), context, deadline=monotonic() + 5.0)
    assert not result.accepted
    assert "checker_mutated_validation_trajectory" in result.violation_codes


def test_initial_phase_comes_from_irg_and_declared_entry_guard(request_scene):
    context = physical_context(request_scene)
    state = initial_execution_state(context.scene.irg, context.observation)
    assert state.phase_id == context.execution_state.phase_id
    irg = deepcopy(context.scene.irg)
    phase = next(n for n in irg.nodes if n.node_id == state.phase_id)
    phase.feature["entry_condition"] = {"guard_id": "ready"}
    with pytest.raises(SchemaValidationError, match="entry guard"):
        initial_execution_state(irg, context.observation)
    observed = deepcopy(context.observation)
    observed.guard_samples = [
        ExecutionGuardSample("ready", 0.0, True, 1.0, "fixture", 0.01)
    ]
    with pytest.raises(SchemaValidationError, match="entry guard"):
        initial_execution_state(irg, observed)
    observed.guard_samples[0].held_for_s = 0.2
    assert initial_execution_state(irg, observed).phase_id == state.phase_id


def test_legacy_pi_h_cli_is_explicit_and_rejects_before_dataset_io(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/order9_train_bc.py",
            "--stage",
            "r1_pi_h_assignment_bc",
            "--dataset",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--legacy-full-cwt-pi-h" in result.stderr
    assert not output.exists()
