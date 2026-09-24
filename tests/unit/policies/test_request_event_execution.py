from types import SimpleNamespace
import torch
import pytest
from amsrr.simulation.request_event_execution import RequestEventSupervisor
from amsrr.simulation.request_event_execution import object_pose_goal_deadlines
from amsrr.simulation.request_event_execution import ObjectPoseGoalDeadlineMonitor
from amsrr.simulation.request_event_execution import apply_object_goal_outcomes
from amsrr.simulation.request_event_execution import BatchedRequestEventSupervisor
from amsrr.simulation.request_event_execution import validate_initial_encoding
from amsrr.schemas.task_spec import GoalSpec
from tests.unit.policies.test_high_level_requests import request_scene, decision


@pytest.mark.parametrize('field', ['features', 'owners', 'graphs'])
def test_initial_input_audit_rejects_changed_targets_ownership_and_state(field):
    from copy import deepcopy
    expected = dict(features=torch.zeros(2, 3), owners=torch.tensor([0, 1]),
                    graphs={'nodes': torch.zeros(2, 4)})
    actual = deepcopy(expected)
    validate_initial_encoding(expected, actual)
    if field == 'graphs':
        actual[field]['nodes'][0, 0] = .006
    else:
        actual[field][0] += 1
    with pytest.raises(ValueError, match='initial encoding changed'):
        validate_initial_encoding(expected, actual)


def test_planned_candidate_grounding_cannot_change_initial_actor_inputs(request_scene):
    from copy import deepcopy
    from amsrr.policies.request_actor_critic import RequestActorCritic
    from amsrr.schemas.policies import ObjectTarget
    from amsrr.training.request_imitation import PHASES
    from amsrr.training.request_ppo import serialize_encoding
    scene, task, physical, _ = request_scene
    model = RequestActorCritic()
    candidates = deepcopy(scene.contact_candidate_set)
    target = ObjectTarget(task.scene.objects[0].object_id, task.scene.objects[0].pose_world)
    bundle = SimpleNamespace(morphology=scene.morphology_graph,
        contact_candidate_set=deepcopy(candidates), provenance={'compiled_plan_id':'plan'},
        phase_trajectories={p:SimpleNamespace(horizon_s=1.,knots=[SimpleNamespace(object_targets=[target])]) for p in PHASES})
    def initial():
        supervisor = RequestEventSupervisor(model=model,bundle=bundle,task=task,
            initial_task=task,initial_candidates=candidates,physical=physical,
            expected_group=candidates.group_proposals[0].group_id)
        context = supervisor.context(scene.runtime_observation,0,initial=True)
        return model.decide(context,deterministic=True)
    before = initial()
    for candidate in bundle.contact_candidate_set.candidates:
        for name in ('contact_pose_world','contact_frame_world'):
            pose=list(getattr(candidate,name));pose[2]+=.004;setattr(candidate,name,tuple(pose))
    after = initial()
    validate_initial_encoding(serialize_encoding(before['encoded']),serialize_encoding(after['encoded']))
    assert before['requests'] == after['requests']


def setup():
    s = RequestEventSupervisor.__new__(RequestEventSupervisor)
    s.started = True
    s.phase = 2
    s.dt = 0.02
    s.time_s = 1.0
    s.pending = object()
    s.dwell = 0.0
    s.release_latched = False
    s.final_goal_position_tolerance_m = 0.05
    s.final_goal_orientation_tolerance_rad = 0.2
    s.initial_object = torch.zeros(3)
    s.maximum_lift = 0.0
    s.maximum_transport = 0.0
    s.events = [{"reward": 0.0, "reward_events": [], "reward_timing": "observed_reward_time_v1"}]
    s.evaluator_phase_exits = []
    s.trace = []
    pose = torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]])
    state = SimpleNamespace(
        object_pose_world=pose, object_twist_world=torch.zeros(1, 6)
    )
    model = SimpleNamespace(body_pose_world=pose, body_twist_world=torch.zeros(1, 6))
    target = SimpleNamespace(
        phase_goal_robot_root_pose_world=pose,
        phase_goal_object_pose_world=pose,
        phase_progress=torch.ones(1),
    )
    args = dict(
        phase_index=torch.tensor([2]),
        phase_elapsed_s=torch.tensor([30.0]),
        target=target,
        state=state,
        control_model=model,
        surface_distance=torch.zeros(1, 2),
        relative_speed=torch.zeros(1, 2),
        motor_load=torch.ones(1, 2),
        qp_feasible=torch.ones(1, dtype=torch.bool),
        privileged_reward=SimpleNamespace(
            phase_success=torch.ones(1, dtype=torch.bool)
        ),
    )
    return s, args


@pytest.mark.parametrize('phase', [4, 5, 7])
@pytest.mark.parametrize('tolerance,error,expected', [(.05,.0503,False),(.05,.0499,True),(.02,.0203,False),(.02,.0199,True)])
def test_final_phase_guard_uses_authored_position_tolerance(phase,tolerance,error,expected):
    s,a=setup();s.phase=phase;s.dwell=2.;s.release_latched=True
    s.final_goal_position_tolerance_m=tolerance
    a['phase_index'].fill_(phase)
    a['state'].object_pose_world=a['state'].object_pose_world.clone()
    a['state'].object_pose_world[0,0]=error
    if phase==5:a['surface_distance'].fill_(.006)
    assert bool(s.step(**a).item()) is expected


@pytest.mark.parametrize('phase', [4, 5, 7])
def test_final_phase_guard_uses_authored_orientation_tolerance(phase):
    import math
    s,a=setup();s.phase=phase;s.dwell=2.;s.release_latched=True
    s.final_goal_orientation_tolerance_rad=.03
    a['phase_index'].fill_(phase)
    a['state'].object_pose_world=a['state'].object_pose_world.clone()
    a['state'].object_pose_world[0,3:]=torch.tensor([0.,0.,math.sin(.04/2),math.cos(.04/2)])
    if phase==5:a['surface_distance'].fill_(.006)
    assert not s.step(**a).item()


def test_privileged_success_cannot_advance_unloaded_contact():
    s, a = setup()
    a["motor_load"].zero_()
    assert not s.step(**a).item()


def test_false_privileged_label_cannot_override_observed_guard():
    s, a = setup()
    a["privileged_reward"].phase_success.zero_()
    assert s.step(**a).item()
    assert not s.evaluator_phase_exits[-1]["privileged_phase_success"]


@pytest.fixture
def contact_velocity_path():
    """Execute the actual adapted harness math without launching Isaac."""
    import ast
    from scripts.run_request_policy import (
        PROTECTED_HARNESS, PROTECTED_SHA256, _with_contact_point_velocities,
    )
    from amsrr.utils.hashing import hash_file

    assert hash_file(PROTECTED_HARNESS) == PROTECTED_SHA256
    tree = ast.parse(_with_contact_point_velocities(PROTECTED_HARNESS.read_text()))
    helpers = {"_selected_grasp_frame_positions", "_quaternion_rotate_vector",
               "_deployable_selected_contact_feedback_inputs", "_torch"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<harness helpers>", "exec"), namespace)
    assignments = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in {"pre_selected_body_twist", "selected_body_twist", "selected_relative_speed_mps"}:
                assignments[n.targets[0].id] = compile(ast.Module(body=[n], type_ignores=[]), "<harness input>", "exec")
    assert len(assignments) == 3
    return namespace, assignments


@pytest.mark.parametrize("slip,expected_advance", [(0.0, True), (0.049, True), (0.051, False)])
@pytest.mark.parametrize("origin_shift", [0.0, 0.2])
@pytest.mark.parametrize("frame_quaternion", [(0., 0., 0., 1.), (0., 0., 2**-0.5, 2**-0.5)])
def test_contact_speed_uses_same_point_and_preserves_slip_threshold(
    contact_velocity_path, slip, expected_advance, origin_shift, frame_quaternion,
):
    ns, assignments = contact_velocity_path
    # Two bodies share a rigid rotation around the object's COM. Their origin
    # velocities differ, but velocity at their common point must agree.
    # Additional x translation is known physical slip, independent of frame.
    link_velocity = torch.tensor([[[slip, 1.0 + origin_shift, 0., 0., 0., 1.]]])
    ns.update(robot=SimpleNamespace(data=SimpleNamespace(
        body_link_vel_w=link_velocity,
        body_lin_vel_w=torch.tensor([[[slip, 1.4, 0.]]]),
        body_ang_vel_w=torch.tensor([[[0., 0., 1.]]]),
    )), selected_anchor_body_indices_tensor=torch.tensor([0]))
    for name in ("pre_selected_body_twist", "selected_body_twist"):
        exec(assignments[name], ns)
        assert torch.equal(ns[name], link_velocity)
    distance, twist = ns["_deployable_selected_contact_feedback_inputs"](
        selected_body_pose_world=torch.tensor([[[1.+origin_shift, 0., 0., 0., 0., 0., 1.]]]),
        selected_body_twist_world=ns["selected_body_twist"],
        selected_anchor_local_pose=torch.tensor([[-.5-origin_shift, 0., 0., 0., 0., 0., 1.]]),
        object_pose_world=torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]),
        object_twist_world=torch.tensor([[0., 0., 0., 0., 0., 1.]]),
        contact_frame_pose_world=torch.tensor([[[.5, 0., 0., *frame_quaternion]]]),
        contact_normal_world=torch.tensor([[[1., 0., 0.]]]),
    )
    ns["relative_twist_contact"] = twist
    exec(assignments["selected_relative_speed_mps"], ns)
    speed = ns["selected_relative_speed_mps"]
    torch.testing.assert_close(speed, torch.tensor([[slip]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(distance, torch.zeros_like(distance), atol=1e-6, rtol=0)
    supervisor, args = setup()
    args.update(surface_distance=distance, relative_speed=speed, motor_load=torch.ones_like(speed))
    assert supervisor.step(**args).item() is expected_advance


def test_contact_velocity_adapter_rejects_changed_harness_boundary():
    from scripts.run_request_policy import PROTECTED_HARNESS, _with_contact_point_velocities

    with pytest.raises(ValueError, match="velocity boundary changed"):
        _with_contact_point_velocities(PROTECTED_HARNESS.read_text().replace(
            "pre_selected_body_twist = torch.cat(", "renamed_twist = torch.cat("
        ))


@pytest.mark.parametrize("mode", ["refresh", "lazy", "reinitialize"])
def test_sensor_container_reuse_reads_each_fresh_step_and_fails_closed(mode):
    from scripts.run_request_policy import _with_refreshed_contact_buffers
    import time

    class Sensor:
        def __init__(self):
            self._data = SimpleNamespace(value=0)
            self.reads = 0

        @property
        def data(self):
            self.reads += 1
            return self._data

    robot, obj = Sensor(), Sensor()
    def update(dt):
        for sensor in (robot, obj):
            sensor._data.value += 1
        if mode == "reinitialize":
            obj._data = SimpleNamespace(value=100)

    source = '''def run():
    values = []
    rollout_started = time.perf_counter()
    for rollout_index in range(int(args_cli.rollout_steps)):
        scene.update(float(args_cli.dt))
        values.append((robot_sensor.data.value, object_sensor.data.value, object_sensor.data.value))
    rollout_elapsed = time.perf_counter() - rollout_started
    return values
'''
    ns = dict(time=time, robot_sensor=robot, object_sensor=obj,
              scene=SimpleNamespace(cfg=SimpleNamespace(lazy_sensor_update=mode == "lazy"), update=update),
              args_cli=SimpleNamespace(rollout_steps=3, dt=.02))
    exec(_with_refreshed_contact_buffers(source), ns)
    if mode != "refresh":
        with pytest.raises(RuntimeError):
            ns["run"]()
    else:
        assert ns["run"]() == [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        assert robot.reads == obj.reads == 1


@pytest.mark.parametrize("collision", [False, True])
def test_lazy_collision_details_preserve_events_without_unused_pose_reads(collision):
    import ast
    import json
    from scripts.run_request_policy import PROTECTED_HARNESS, _with_lazy_collision_details

    original = PROTECTED_HARNESS.read_text()
    outputs = []
    reads = []
    for source in (original, _with_lazy_collision_details(original)):
        class BodyData:
            count = 0
            @property
            def body_pose_w(self):
                self.count += 1
                return torch.tensor([[[0., 0., 0., 0., 0., 0., 1.]]])
        data = BodyData()
        tree = ast.parse(source)
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "args_cli.diagnostic_collision_evidence"]
        assert len(nodes) == 1
        logs = []
        ns = dict(torch=torch, json=json, print=lambda text, **kwargs: logs.append(text),
            _torch=lambda value: value, robot=SimpleNamespace(data=data),
            args_cli=SimpleNamespace(diagnostic_collision_evidence=True),
            contact=SimpleNamespace(prohibited_collision=torch.tensor([collision]),
                prohibited_object_contact=torch.tensor([collision]),
                prohibited_environment_contact=torch.tensor([False]),
                object_contact_forces_by_robot_body_world=torch.tensor([[[1., 0., 0.]]])),
            robot_net=torch.zeros(1, 1, 3), diagnostic_collision_reported=torch.tensor([False]),
            io=SimpleNamespace(selected_anchor_body_indices=(0,), contact_force_threshold_n=.5,
                               robot_body_names=("anchor",)),
            rollout_index=3, pre_phase=torch.tensor([1]), pre_elapsed=torch.tensor([.2]))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<collision diagnostic>", "exec"), ns)
        assert ns["diagnostic_collision_reported"].item() == collision
        outputs.append(logs)
        reads.append(data.count)
    assert outputs[0] == outputs[1]
    assert reads == [1, int(collision)]


def test_reviewed_route_cannot_replace_the_policy_selected_contact_binding(
    request_scene,
):
    from amsrr.policies.request_contact_planner import reuse_reviewed_request_geometry

    context = decision(request_scene)
    entry = next(
        e
        for e in context.catalog.entries
        if e.request.contact_group_id is not None and e.transition is None
    )
    wrong_binding = SimpleNamespace(
        phase_trajectories={
            "contact_acquisition": SimpleNamespace(
                knots=[SimpleNamespace(contact_assignments=[])]
            )
        }
    )
    with pytest.raises(ValueError, match="different contact binding"):
        reuse_reviewed_request_geometry(context, entry.request, wrong_binding)


def test_progress_and_geometric_contact_are_required():
    s, a = setup()
    a["target"].phase_progress.fill_(0.8)
    assert not s.step(**a).item()
    a["target"].phase_progress.fill_(1.0)
    a["surface_distance"].fill_(0.03)
    assert not s.step(**a).item()


def test_retreat_clears_vertically_without_changing_release_or_object_pose():
    from amsrr.schemas.policies import (
        InteractionKnot,
        ContactWrenchTrajectory,
        CentroidalTarget,
        PostureTarget,
        ObjectTarget,
    )
    from amsrr.policies.request_contact_planner import vertical_clearance_retreat

    k = InteractionKnot(
        0.0,
        [],
        centroidal_target=CentroidalTarget(
            com_pos_world=(0.2, 0.3, 0.4), body_orientation_world=(0.0, 0.0, 0.0, 1.0)
        ),
        posture_target=PostureTarget(
            joint_pos_target={"j": 0.1}, joint_vel_target={"j": 0.0}
        ),
        object_targets=[ObjectTarget("o", (0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 1.0))],
    )
    end = InteractionKnot.from_dict({**k.to_dict(), "t_rel_s": 1.0})
    phase = ContactWrenchTrajectory(1.0, 1.0, [k, end])
    before = phase.to_dict()
    phases = vertical_clearance_retreat(
        {"release": phase, "retreat": phase, "settle": phase}
    )
    assert phases["release"].to_dict() == before
    assert tuple(phases["retreat"].knots[-1].centroidal_target.com_pos_world) == (
        0.2,
        0.3,
        0.5,
    )
    assert phases["retreat"].knots[-1].posture_target.joint_pos_target == {"j": 0.1}
    assert (
        phases["settle"].knots[-1].object_targets[0].pose_target_world
        == end.object_targets[0].pose_target_world
    )


def test_boundary_matches_executor_and_does_not_double_reward_early_completion():
    s, a = setup()
    a["target"].phase_progress.fill_(0.9995)
    assert not s.step(**a).item()
    assert not s.evaluator_phase_exits
    a["target"].phase_progress.fill_(1.0)
    assert s.step(**a).item()
    assert len(s.evaluator_phase_exits) == 1


def test_object_orientation_is_a_deployable_guard():
    s, a = setup()
    a["state"].object_pose_world = a["state"].object_pose_world.clone()
    a["state"].object_pose_world[0, 3:] = torch.tensor([0.0, 0.0, 0.7071068, 0.7071068])
    assert not s.step(**a).item()


def deadline_task():
    return SimpleNamespace(
        scene=SimpleNamespace(objects=[SimpleNamespace(object_id="object")]),
        goals=[
            GoalSpec(
                goal_id="place",
                goal_type="object_pose",
                time_limit_s=150.0,
                target_entity_id="object",
                target_pose_world=(1, 0, 0, 0, 0, 0, 1),
                tolerance_pos_m=0.05,
                tolerance_rot_rad=0.2,
            )
        ],
    )


@pytest.mark.parametrize(
    "arrival,expected", [(149.0, True), (150.0, True), (150.01, False)]
)
def test_goal_deadline_uses_first_arrival_not_end_of_release_and_retreat(
    arrival, expected
):
    poses = [[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]
    result = object_pose_goal_deadlines(deadline_task(), [0, arrival, 200], poses)[0]
    assert result["passed"] is expected
    assert result["first_pose_goal_reach_s"] == arrival
    supervisor, _ = setup()
    supervisor.initial_runtime_check = {}
    supervisor.finish(result["passed"])
    assert supervisor.events[-1]["reward"] == (5.0 if expected else -5.0)
    assert supervisor.events[-1]["done"]


def test_goal_deadline_requires_authored_position_and_orientation_tolerances():
    # A 51 mm position error must not pass a 50 mm goal, nor may a reversed
    # orientation count as on-time arrival just because the position matches.
    poses = [[1.051, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1]]
    result = object_pose_goal_deadlines(deadline_task(), [149, 150, 151], poses)[0]
    assert not result["passed"]
    assert result["first_pose_goal_reach_s"] == 151


@pytest.mark.parametrize("times", [[2, 1], [0, float("nan")], [0, float("inf")]])
def test_goal_deadline_rejects_invalid_observation_clock(times):
    with pytest.raises(ValueError, match="invalid physical"):
        object_pose_goal_deadlines(deadline_task(), times, [[1, 0, 0, 0, 0, 0, 1]] * 2)


def deadline_reward():
    from dataclasses import fields
    from amsrr.training.order9_tensor_reward import Order9TensorRewardResult

    values = {f.name: torch.zeros(1) for f in fields(Order9TensorRewardResult)}
    for key in ("phase_success", "terminal_failure", "timeout"):
        values[key] = torch.zeros(1, dtype=torch.bool)
    values["phase_success"].fill_(True)
    values["terms"] = {}
    values["next_state"] = None
    return Order9TensorRewardResult(**values)


def test_unmet_goal_terminates_even_when_legacy_phase_success_prevents_timeout():
    monitor = ObjectPoseGoalDeadlineMonitor(deadline_task())
    reward = deadline_reward()
    pose = torch.tensor([0, 0, 0, 0, 0, 0, 1.0])
    assert monitor.apply(reward, time_s=150.0, object_pose_world=pose) is reward
    stopped = monitor.apply(reward, time_s=150.02, object_pose_world=pose)
    assert stopped.timeout.item() and stopped.terminal_failure.item()
    assert not stopped.phase_success.item()
    assert reward.phase_success.item() and not reward.terminal_failure.item()
    assert monitor.termination_time_s == 150.02


@pytest.mark.parametrize("arrival", [149.0, 150.0])
def test_deadline_monitor_preserves_on_time_arrival_and_later_release(arrival):
    monitor = ObjectPoseGoalDeadlineMonitor(deadline_task())
    reward = deadline_reward()
    assert (
        monitor.apply(reward, time_s=arrival, object_pose_world=[1, 0, 0, 0, 0, 0, 1])
        is reward
    )
    assert (
        monitor.apply(reward, time_s=200, object_pose_world=[1.1, 0, 0, 0, 0, 0, 1])
        is reward
    )
    assert not monitor.deadline_missed
    assert monitor.first_arrivals == {"place": arrival}


@pytest.mark.parametrize("pose", [[1.051, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0]])
def test_deadline_monitor_cannot_latch_inaccurate_position_or_orientation(pose):
    monitor = ObjectPoseGoalDeadlineMonitor(deadline_task())
    monitor.apply(deadline_reward(), time_s=149, object_pose_world=pose)
    result = monitor.apply(
        deadline_reward(), time_s=151, object_pose_world=[1, 0, 0, 0, 0, 0, 1]
    )
    assert result.terminal_failure.item()
    assert not monitor.first_arrivals


def test_deadline_monitor_uses_the_archived_post_step_clock():
    # Float32 episode accumulation differs from the supervisor's Python clock.
    # Both live termination and the completed-trajectory checker use pre_time+dt.
    pose = torch.tensor([1, 0, 0, 0, 0, 0, 1.0])
    time_s = float(torch.tensor(149.98).double()) + 0.02
    monitor = ObjectPoseGoalDeadlineMonitor(deadline_task())
    monitor.apply(deadline_reward(), time_s=time_s, object_pose_world=pose)
    monitor.apply(deadline_reward(), time_s=151, object_pose_world=pose)
    result = object_pose_goal_deadlines(deadline_task(), [time_s], pose[None])[0]
    assert result["passed"]
    assert not monitor.deadline_missed
    assert monitor.first_arrivals["place"] == result["first_pose_goal_reach_s"]


def test_deadline_monitor_rejects_backwards_clock():
    monitor = ObjectPoseGoalDeadlineMonitor(deadline_task())
    pose = [0, 0, 0, 0, 0, 0, 1]
    monitor.apply(deadline_reward(), time_s=2, object_pose_world=pose)
    with pytest.raises(ValueError, match="clock moved backwards"):
        monitor.apply(deadline_reward(), time_s=1, object_pose_world=pose)


@pytest.mark.parametrize(
    "passed,deadline,safety,original,expected,timeout",
    [
        (True, False, False, None, None, 0),
        (False, False, False, None, "object_pose_goal_not_reached", 0),
        (False, True, False, "phase_timeout", "object_pose_goal_deadline_missed", 1),
        (False, False, False, "phase_timeout", "phase_timeout", 1),
        (False, True, True, "object_dropped", "object_dropped", 1),
    ],
)
def test_goal_outcome_distinguishes_tolerance_deadline_and_physical_failures(
    passed, deadline, safety, original, expected, timeout
):
    record = dict(
        task_success=original is None,
        no_fallback_success=original is None,
        safety_failure=safety,
        failure_reason=original,
        metrics={"timeout": float(original == "phase_timeout")},
    )
    apply_object_goal_outcomes(
        record, [{"passed": passed, "final_pose_within_tolerance": passed}],
        deadline_terminated=deadline
    )
    assert record["failure_reason"] == expected
    assert record["metrics"]["timeout"] == timeout
    assert record["task_success"] == passed
    assert record["no_fallback_success"] == passed


@pytest.mark.parametrize("final_error,expected", [(0.0499, True), (0.0501, False)])
def test_transient_goal_arrival_does_not_reward_final_pose_outside_tolerance(
    final_error, expected
):
    # Preserve the on-time arrival, but use the authored final tolerance rather
    # than the protected legacy harness's extra 2 mm numerical margin.
    poses = [[1, 0, 0, 0, 0, 0, 1], [1 + final_error, 0, 0, 0, 0, 0, 1]]
    outcomes = object_pose_goal_deadlines(deadline_task(), [140, 160], poses)
    assert outcomes[0]["passed"]
    assert outcomes[0]["final_pose_within_tolerance"] is expected
    record = dict(task_success=True, no_fallback_success=True,
                  safety_failure=False, failure_reason=None, metrics={"timeout": 0.0})
    apply_object_goal_outcomes(record, outcomes, deadline_terminated=False)
    assert record["task_success"] is expected
    assert record["metrics"]["timeout"] == 0.0
    assert record["failure_reason"] == (None if expected else "object_pose_goal_not_maintained")
    supervisor, _ = setup()
    supervisor.initial_runtime_check = {}
    supervisor.finish(record["task_success"])
    assert supervisor.events[-1]["reward"] == (5.0 if expected else -5.0)


def batch_values(*rows):
    from dataclasses import fields, is_dataclass, replace

    first = rows[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(rows)
    if isinstance(first, SimpleNamespace):
        return SimpleNamespace(**{k: batch_values(*(getattr(r, k) for r in rows))
                                  for k in vars(first)})
    if is_dataclass(first):
        return replace(first, **{f.name: batch_values(*(getattr(r, f.name) for r in rows))
                                 for f in fields(first)})
    return first


def test_batch_guards_and_retired_environment_events_are_independent():
    first, a = setup()
    second, b = setup()
    first.task = second.task = deadline_task()
    b["motor_load"].zero_()
    batch = BatchedRequestEventSupervisor([first, second])
    args = {k: batch_values(a[k], b[k]) for k in a}
    assert batch.step(active=torch.tensor([True, True]), **args).tolist() == [True, False]
    saved = (first.time_s, first.events[0]["reward"], len(first.trace))
    args["motor_load"][1].fill_(1)
    assert batch.step(active=torch.tensor([False, True]), **args).tolist() == [False, True]
    assert saved == (first.time_s, first.events[0]["reward"], len(first.trace))
    assert second.time_s > first.time_s
    with pytest.raises(ValueError, match="distinct"):
        BatchedRequestEventSupervisor([first, first])


def test_actor_observation_removes_replica_translation(request_scene):
    scene, task, physical, _ = request_scene
    supervisor, _ = setup()
    supervisor.bundle = SimpleNamespace(morphology=scene.morphology_graph)
    supervisor.task = task
    supervisor.module_ids = [m.module_id for m in scene.morphology_graph.modules]
    supervisor.joint_ids = [j.joint_id for j in physical.joints]
    modules = torch.tensor([[m.pose_world for m in scene.runtime_observation.module_states]], dtype=torch.float64)
    objects = torch.tensor([scene.runtime_observation.object_states[0].pose_world], dtype=torch.float64)
    state = SimpleNamespace(
        module_pose_world=modules, module_twist_world=torch.zeros(1, len(supervisor.module_ids), 6),
        local_joint_positions_rad=torch.zeros(1, len(supervisor.module_ids), len(supervisor.joint_ids)),
        local_joint_velocities_radps=torch.zeros(1, len(supervisor.module_ids), len(supervisor.joint_ids)),
        object_pose_world=objects, object_twist_world=torch.zeros(1, 6),
    )
    supervisor.environment_origin = torch.zeros(3)
    original = supervisor.observation(state)
    supervisor.environment_origin = torch.tensor([3., -3., 0.])
    state.module_pose_world = modules.clone()
    state.object_pose_world = objects.clone()
    state.module_pose_world[..., :3] += supervisor.environment_origin
    state.object_pose_world[..., :3] += supervisor.environment_origin
    translated = supervisor.observation(state)
    for before, after in zip(original.module_states + original.object_states,
                             translated.module_states + translated.object_states):
        torch.testing.assert_close(torch.tensor(before.pose_world), torch.tensor(after.pose_world))


def test_batch_deadlines_use_each_origin_and_retire_without_reset():
    batch = BatchedRequestEventSupervisor([SimpleNamespace(task=deadline_task()) for _ in range(3)])
    batch.origins = torch.tensor([[0., 0., 0.], [3., -3., 0.], [-3., 3., 0.]])
    poses = torch.tensor([[1., 0., 0., 0., 0., 0., 1.],
                          [0., 0., 0., 0., 0., 0., 1.],
                          [1., 0., 0., 0., 0., 0., 1.]])
    poses[:, :3] += batch.origins
    reward = batch_values(*(deadline_reward() for _ in range(3)))
    result = batch.apply_deadlines(reward, time_s=torch.tensor([149., 151., 200.]),
                                   object_pose_world=poses, active=torch.tensor([True, True, False]))
    assert result.terminal_failure.tolist() == [False, True, True]
    assert result.phase_success.tolist() == [True, False, False]
    assert batch.deadlines[0].first_arrivals == {"place": 149.}
    assert batch.deadlines[1].termination_time_s == 151.
    assert batch.deadlines[2].last_time_s is None
    assert not reward.terminal_failure.any()


def test_batch_harness_preserves_only_first_terminal_sample_and_rejects_drift():
    import ast
    from scripts.run_request_policy import PROTECTED_HARNESS, _with_independent_request_environments

    original = PROTECTED_HARNESS.read_text()
    adapted = _with_independent_request_environments(original)
    ast.parse(adapted)
    active = torch.tensor([True, True])
    valid = active.clone()
    terminal = torch.tensor([True, False])
    active &= ~terminal
    reset_line = next(l.strip() for l in adapted.splitlines()
                      if l.strip().startswith("reset_ids = torch.nonzero(terminal"))
    namespace = dict(torch=torch, terminal=terminal, evaluation_active=active)
    exec(reset_line, namespace)
    assert not len(namespace["reset_ids"])
    assert valid.tolist() == [True, True]  # include the terminal step
    assert active.tolist() == [False, True]  # omit subsequent samples
    with pytest.raises(ValueError, match="boundary changed"):
        _with_independent_request_environments(adapted)
    assert PROTECTED_HARNESS.read_text() == original


def test_colocated_harness_requires_explicit_collision_isolation():
    import ast
    from scripts.run_request_policy import PROTECTED_HARNESS, _with_colocated_request_environments

    original = PROTECTED_HARNESS.read_text()
    adapted = _with_colocated_request_environments(original)
    ast.parse(adapted)
    assert "scene_cfg.filter_collisions = True" in adapted
    assert "GetPrimAtPath('/World/collisions')" in adapted
    assert "co-located environment origins differ" in adapted
    with pytest.raises(ValueError, match="boundary changed"):
        _with_colocated_request_environments(adapted)
    assert PROTECTED_HARNESS.read_text() == original
