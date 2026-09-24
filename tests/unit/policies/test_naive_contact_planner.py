from types import SimpleNamespace

import pytest
import torch

from amsrr.policies.naive_contact_planner import (
    NaiveContactPlanReference,
    preload_weight,
    bounded_joint_curve,
)
from tests.unit.policies.test_high_level_requests import request_scene


def test_reset_uses_predecision_observation_not_selected_plan(request_scene):
    from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
    scene, _, physical, _ = request_scene
    obs = scene.runtime_observation
    for module in obs.module_states:
        module.twist_world = [0.01, 0.02, 0.03, 0., 0., 0.]
        module.joint_velocities = {j.joint_id: 0.04 for j in physical.joints}
    reference = object.__new__(NaiveContactPlanReference)
    reference.device, reference.dtype = torch.device('cpu'), torch.float64
    reference.module_ids = tuple(m.module_id for m in obs.module_states)
    reference.joint_ids = tuple(j.joint_id for j in physical.joints)
    reference.initial_observation, reference.physical_model = obs, physical
    # No trajectory exists: a selected plan cannot contribute to the reset.
    reference._references = {}
    reset = reference.phase_start_reference()
    measured = RigidBodyControlModelBuilder().build(obs.morphology_graph, physical, obs)
    torch.testing.assert_close(reset.body_twist, torch.tensor(measured.body_twist_world, dtype=torch.float64))
    torch.testing.assert_close(reset.body_pose_local, torch.tensor(measured.body_pose_world, dtype=torch.float64))
    assert torch.all(reset.joint_velocities_radps == 0.04)
    obs.module_states[0].twist_world[2] += 0.1
    assert not torch.equal(reference.phase_start_reference().body_twist, reset.body_twist)


def test_observed_reset_velocity_is_shifted_from_com_to_root():
    import ast
    from scripts.run_request_policy import _with_reset_twist_reference_point, PROTECTED_HARNESS
    tree = ast.parse(_with_reset_twist_reference_point(PROTECTED_HARNESS.read_text()))
    reset = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_restore_c3_formal_phase_zero')
    nodes = [n for n in reset.body if isinstance(n, (ast.Assign, ast.AugAssign)) and
             any(isinstance(x, ast.Name) and x.id == 'root_twist' for x in ast.walk(n))]
    ns = dict(torch=torch, body_twist=torch.tensor([[0., 1., 0., 0., 0., 1.]]),
              desired_body_world=torch.tensor([[1., 0., 0.]]), root_pose_world=torch.tensor([[0., 0., 0.]]))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reset twist>', 'exec'), ns)
    torch.testing.assert_close(ns['root_twist'], torch.tensor([[0., 0., 0., 0., 0., 1.]]))


def test_planned_joint_curve_stays_bounded_between_knots():
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline

    times = [0.0, 0.1, 0.2, 0.3]
    q, v, correction = bounded_joint_curve(
        times, [[0.0], [0.9], [1.001], [0.99]], [-1.0], [1.0]
    )
    assert correction == pytest.approx(0.001)
    curve = CubicHermiteSpline(times, q[:, 0], v[:, 0])
    samples = curve(np.linspace(0, 0.3, 1001))
    assert np.max(samples) <= 1.0 + 1e-12
    assert v[0, 0] == v[-1, 0] == 0.0


def test_cubic_joint_interpolator_preserves_position_and_true_velocity():
    evaluator = object.__new__(NaiveContactPlanReference)
    evaluator.device, evaluator.dtype = torch.device("cpu"), torch.float64
    # q(t)=t^3 is determined exactly by these endpoint positions/derivatives.
    pose = torch.tensor([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=torch.float64)
    reference = SimpleNamespace(
        times_s=torch.tensor([0.0, 1.0], dtype=torch.float64),
        body_pose_world=pose,
        body_twist_world=torch.zeros((2, 6), dtype=torch.float64),
        object_pose_world=pose,
        object_twist_world=torch.zeros((2, 6), dtype=torch.float64),
        joint_positions_rad=torch.tensor([0.0, 1.0], dtype=torch.float64).reshape(
            2, 1, 1
        ),
        joint_velocities_radps=torch.tensor([0.0, 3.0], dtype=torch.float64).reshape(
            2, 1, 1
        ),
    )
    t = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64)
    sampled = evaluator._sample(reference, t)
    torch.testing.assert_close(sampled[2].flatten(), t**3)
    torch.testing.assert_close(sampled[3].flatten(), 3 * t**2)
    after = evaluator._sample(reference, torch.tensor([2.0], dtype=torch.float64))
    assert after[2].item() == 1.0
    assert after[3].item() == 0.0


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("approach", (0.0, 0.0)),
        ("contact_acquisition", (0.5, 1.5)),
        ("lift", (1.0, 0.0)),
        ("transport", (1.0, 0.0)),
        ("place", (1.0, 0.0)),
        ("release", (0.5, -1.5)),
        ("retreat", (0.0, 0.0)),
    ],
)
def test_preload_is_loaded_and_unloaded_continuously(phase, expected):
    assert preload_weight(phase, 0.5) == pytest.approx(expected)
    assert preload_weight(phase, 0.0)[1] == 0.0
    assert preload_weight(phase, 1.0)[1] == 0.0


def test_new_nominal_execution_keeps_planned_velocity_and_never_calls_actor(
    monkeypatch,
):
    from tests.unit.training.test_order9_tensor_pi_l_runtime import _runtime_fixture
    from amsrr.simulation.teacher_contact_execution import TeacherContactController

    runtime, state, target = _runtime_fixture()
    runtime.__class__ = TeacherContactController
    target.desired_robot_root_twist_world[:, 0] = 0.01
    target.nominal_joint_positions_rad[:] = 0.02
    target.nominal_joint_velocities_radps[:] = 0.03

    def forbidden(*args, **kwargs):
        raise AssertionError("nominal execution called the learned actor")

    monkeypatch.setattr(runtime.policy, "forward", forbidden)
    result = runtime.compute_nominal_qpid_hold(
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.ones(2),
        estimated_payload_inertia_body=torch.zeros((2, 6)),
        payload_active=torch.zeros(2),
    )
    command = result.policy_command
    torch.testing.assert_close(
        command.desired_body_twist, target.desired_robot_root_twist_world
    )
    torch.testing.assert_close(
        command.joint_position_targets_rad, target.nominal_joint_positions_rad
    )
    torch.testing.assert_close(
        command.joint_velocity_targets_radps, target.nominal_joint_velocities_radps
    )
    assert torch.isfinite(result.controller_result.allocation.rotor_thrusts_n).all()
