from types import SimpleNamespace
import torch
import pytest
from amsrr.simulation.request_event_execution import RequestEventSupervisor
from tests.unit.policies.test_high_level_requests import request_scene, decision


def setup():
    s = RequestEventSupervisor.__new__(RequestEventSupervisor)
    s.started = True
    s.phase = 2
    s.dt = 0.02
    s.time_s = 1.0
    s.pending = object()
    s.dwell = 0.0
    s.release_latched = False
    s.initial_object = torch.zeros(3)
    s.maximum_lift = 0.0
    s.maximum_transport = 0.0
    s.events = [{"reward": 0.0}]
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


def test_privileged_success_cannot_advance_unloaded_contact():
    s, a = setup()
    a["motor_load"].zero_()
    assert not s.step(**a).item()


def test_false_privileged_label_cannot_override_observed_guard():
    s, a = setup()
    a["privileged_reward"].phase_success.zero_()
    assert s.step(**a).item()
    assert not s.evaluator_phase_exits[-1]["privileged_phase_success"]


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
