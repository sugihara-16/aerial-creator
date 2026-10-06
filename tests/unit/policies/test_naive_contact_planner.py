from types import SimpleNamespace

import pytest
import torch

from amsrr.policies.naive_contact_planner import (
    NaiveContactPlanReference,
    preload_weight,
    bounded_joint_curve,
    preload_body_curve,
    normalized_lerp_world_angular_velocity,
    retime_bounded_joint_curve,
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


def test_post_step_model_uses_encoder_rates_and_same_compiled_signature():
    import ast
    from scripts.run_request_policy import _with_articulated_centroidal_velocity, PROTECTED_HARNESS
    tree = ast.parse(_with_articulated_centroidal_velocity(PROTECTED_HARNESS.read_text()))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'post_control' for t in n.targets))
    calls = []
    state = SimpleNamespace(module_pose_world=object(), module_twist_world=object(),
        local_joint_positions_rad=object(), local_joint_velocities_radps=object())
    runtime = SimpleNamespace(builder=SimpleNamespace(build_kinematics=lambda **kw: calls.append(kw)))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<post-step model>', 'exec'),
        dict(state=state, policy_runtime=runtime))
    assert calls[0]['local_joint_velocities_rad'] is state.local_joint_velocities_radps
    assert len(calls[0]) == 4


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


def test_executed_joint_rate_retimes_local_intervals_and_exact_cubic_extrema():
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    times = np.array([0., 2., 2.1, 2.2, 4.2, 6.2])
    positions = np.array([[0., 0.], [0., 0.], [.15, -.04], [.3, -.1], [.3, -.1], [.3, -.1]])
    t, q, v, correction, receipt = retime_bounded_joint_curve(times, positions, [-1., -1.], [1., 1.], .2)
    np.testing.assert_array_equal(q, positions)
    assert correction == 0.
    assert t[-1] < (times[-1] * receipt['maximum_speed_before_radps'] / .2)
    np.testing.assert_allclose(np.diff(t)[[0, 3, 4]], np.diff(times)[[0, 3, 4]])
    samples = CubicHermiteSpline(t, q, v)(np.linspace(t[0], t[-1], 20001), 1)
    assert abs(samples).max() <= .2 + 1e-9
    assert receipt['maximum_speed_after_radps'] <= .2 + 1e-9
    assert np.all(v[[0, -1]] == 0.)


def test_executed_joint_rate_keeps_already_admissible_timing():
    import numpy as np
    times = np.arange(5.)
    t, q, v, _, receipt = retime_bounded_joint_curve(times, [[0.], [.01], [.02], [.01], [0.]], [-1.], [1.], .2)
    np.testing.assert_array_equal(t, times)
    assert receipt['iterations'] == 0


def test_body_preload_velocity_includes_rotation_and_smooth_loading():
    import numpy as np
    from scipy.spatial.transform import Rotation
    t = np.linspace(0., 2., 21)
    rotation = Rotation.from_rotvec(np.outer(t, [0., 0., .3]))
    delta = rotation.apply(np.tile([.01, 0., 0.], (len(t), 1)))
    p, v = preload_body_curve('contact_acquisition', t, rotation.as_quat(), [.01, 0., 0.])
    u = t / 2
    weight = u*u*(3-2*u)
    rate = 6*u*(1-u)/2
    expected = rate[:, None]*delta + weight[:, None]*np.cross([0., 0., .3], delta)
    expected[0], expected[-1] = 0., 0.
    np.testing.assert_allclose(p, weight[:, None]*delta, atol=1e-12)
    np.testing.assert_allclose(v, expected, atol=1e-12)
    # Adjacent phases share the same loaded endpoint and angular transport.
    p2, v2 = preload_body_curve('lift', t, rotation.as_quat(), [.01, 0., 0.])
    np.testing.assert_allclose(p[-1], p2[-1], atol=1e-12)
    np.testing.assert_allclose(v[-1], v2[-1], atol=1e-12)
    p3, v3 = preload_body_curve('release', t, rotation.as_quat(), [.01, 0., 0.])
    np.testing.assert_allclose(p3[-1], 0., atol=1e-12)
    np.testing.assert_allclose(v3[-1], 0., atol=1e-12)


def test_normalized_quaternion_velocity_matches_independent_world_rotation_difference():
    import numpy as np
    from scipy.spatial.transform import Rotation
    start = Rotation.from_euler('xyz', [.5, -.3, .2]).as_quat()
    end = (Rotation.from_rotvec([.1, .3, -.2]) * Rotation.from_quat(start)).as_quat()
    tensor = lambda v: torch.tensor(v, dtype=torch.float64)
    alpha = .37
    def at(u):
        q = (1-u)*start + u*end
        return Rotation.from_quat(q / np.linalg.norm(q))
    epsilon = 1e-6
    expected = (at(alpha+epsilon)*at(alpha-epsilon).inv()).as_rotvec() / (2*epsilon*2.5)
    for sign in [-1., 1.]:
        actual = normalized_lerp_world_angular_velocity(tensor(start[None]), tensor(sign*end[None]),
            tensor([alpha]), tensor([2.5]))
        np.testing.assert_allclose(actual[0], expected, atol=1e-9)
    zero = normalized_lerp_world_angular_velocity(tensor(start[None]), tensor(-start[None]),
        tensor([alpha]), tensor([2.5]))
    np.testing.assert_allclose(zero, 0., atol=1e-12)


def test_nominal_world_angular_velocity_is_transformed_at_qpid_boundary():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from amsrr.simulation.teacher_contact_execution import qpid_reference_twist
    rotation = Rotation.from_euler('xyz', [.4, -.6, .8])
    pose = torch.tensor([[1., 2., 3., *rotation.as_quat()]], dtype=torch.float64)
    twist = torch.tensor([[.1, .2, .3, .4, .5, .6]], dtype=torch.float64)
    result = qpid_reference_twist(twist, pose)
    np.testing.assert_allclose(result[0, 3:], rotation.inv().apply(twist[0, 3:]), atol=1e-12)
    torch.testing.assert_close(result[:, :3], twist[:, :3])
    torch.testing.assert_close(twist, torch.tensor([[.1, .2, .3, .4, .5, .6]], dtype=torch.float64))


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
    reference.body_twist_world[-1] = 1.
    after = evaluator._sample(reference, torch.tensor([2.0], dtype=torch.float64))
    assert torch.all(after[1] == 0.)
    reference.body_pose_world = reference.body_pose_world.clone()
    reference.body_pose_world[-1, :3] = torch.tensor([.7, -.4, .2])
    reference.body_twist_world[:] = 0.
    inside = t[1:-1]
    sample = evaluator._sample(reference, inside)
    plus = evaluator._sample(reference, inside + 1e-6)[0][:, :3]
    minus = evaluator._sample(reference, inside - 1e-6)[0][:, :3]
    torch.testing.assert_close(sample[1][:, :3], (plus-minus)/2e-6, atol=1e-9, rtol=0.)
    endpoints = evaluator._sample(reference, t[[0, -1]])
    assert torch.all(endpoints[1] == 0.)


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
    runtime._joint_load_contacts = None
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


def _compression_problem(monkeypatch, *, normal_error=0., clearance=.01,
                         initial_clearance=None):
    """Two opposing contacts with one symmetric closure joint."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    import amsrr.policies.naive_contact_planner as n
    ids = ['closure']
    monkeypatch.setattr(n, 'ordered_global_dock_joint_ids', lambda *a: ids)
    monkeypatch.setattr(n, '_global_joint_limits', lambda *a: {'closure': (-1., 1.)})
    monkeypatch.setattr(n, 'resolve_mesh_backed_anchor_references', lambda *a: ())
    def fk(q):
        value = q[:, 0]
        positions = np.zeros((len(q), 2, 3))
        positions[:, 0, 0], positions[:, 1, 0] = -1.+value, 1.-value
        rotation = Rotation.from_euler('y', normal_error).as_matrix()
        rotations = np.array([rotation, rotation @ Rotation.from_euler('z', np.pi).as_matrix()])
        return positions, np.broadcast_to(rotations, (len(q), 2, 3, 3)).copy()
    kin = SimpleNamespace(bind_centroidal_anchor_fk=lambda *a: fk)
    monkeypatch.setattr(n, 'Order9PostureTrajectoryResolver',
        lambda *a, **kw: SimpleNamespace(ik_solver=SimpleNamespace(kinematics=kin)))
    knot = SimpleNamespace(posture_target=SimpleNamespace(joint_pos_target={'closure': 0.}),
        centroidal_target=SimpleNamespace(com_pos_world=[0., 0., 0.],
                                         body_orientation_world=[0., 0., 0., 1.]),
        contact_assignments=[SimpleNamespace(anchor_id=i, candidate_id=i) for i in range(2)])
    candidates = SimpleNamespace(candidates=[SimpleNamespace(candidate_id=i, normal_world=[x, 0., 0.])
        for i, x in enumerate([-1., 1.])])
    calls = []
    def check(**kw):
        calls.append(kw)
        return {'minimum_clearance_m': initial_clearance if len(calls)==1 and initial_clearance is not None else clearance}
    solver = SimpleNamespace(set_collision_scene=lambda **kw: None, check_configuration=check)
    obstacle = SimpleNamespace(environment_boxes=[SimpleNamespace(pose_world=[0.,0.,-2.,0.,0.,0.,1.],size_m=[1.,1.,1.])],ground_plane_z_m=None)
    return dict(morphology=object(),physical_model=object(),contact_knot=knot,
        candidate_set=candidates,minimum_lead_m=.002,maximum_lead_m=.04), dict(
        carried_knots=(knot,),collision_object=obstacle,collision_solver=solver)


def test_carried_clearance_preserves_safe_nominal_solution(monkeypatch):
    import numpy as np
    from amsrr.policies.naive_contact_planner import solve_nominal_contact_compression
    args, constraints = _compression_problem(monkeypatch)
    original = solve_nominal_contact_compression(**args)
    checked = solve_nominal_contact_compression(**args, **constraints)
    assert original[0].joint_delta_rad == checked[0].joint_delta_rad
    np.testing.assert_array_equal(original[1], checked[1])
    np.testing.assert_array_equal(checked[2], np.zeros(3))
    assert not checked[3]['required']


def test_preload_uses_same_clearance_as_final_check(monkeypatch):
    """A feasible press must not be sacrificed to an extra 5mm objective."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from amsrr.policies.naive_contact_planner import solve_nominal_contact_compression
    args, constraints = _compression_problem(monkeypatch)
    args['minimum_lead_m'] = .012
    def coupled_clearance(**kw):
        q = kw['joint_positions_rad']['closure']
        rotation = Rotation.from_quat(kw['centroidal_pose_world'][3:])
        # Opposing contacts close by q. Rolling about their common normal
        # clears a nearby arm obstacle without rotating either contact normal.
        roll = rotation.as_rotvec()[0]
        inward = 1. - (1.-q)*rotation.as_matrix()[0, 0]
        return {'minimum_clearance_m': .0133-inward-.006+.15*roll}
    constraints['collision_solver'].check_configuration = coupled_clearance
    result, translation, rotation, evidence = solve_nominal_contact_compression(**args, **constraints)
    assert evidence['required']
    assert min(result.achieved_inward_displacement_m.values()) >= .012
    # Match the production final check's existing 0.2mm numerical tolerance.
    assert min(evidence['after_m']) >= .001 - .0002
    assert max(evidence['after_m']) < .005
    assert np.linalg.norm(translation) <= .030
    assert np.linalg.norm(rotation) <= .1
    assert result.maximum_joint_delta_rad <= .15


@pytest.mark.parametrize('unsafe', ['clearance', 'normal'])
def test_carried_pose_solver_rejects_remaining_unsafe_constraint(monkeypatch, unsafe):
    from amsrr.policies.naive_contact_planner import solve_nominal_contact_compression
    args, constraints = _compression_problem(monkeypatch,
        normal_error=.2 if unsafe=='normal' else 0.,
        clearance=.01 if unsafe=='normal' else -.001, initial_clearance=-.001)
    with pytest.raises(ValueError, match='carried-goal clearance or normal error'):
        solve_nominal_contact_compression(**args, **constraints)


def test_centroidal_curve_has_continuous_velocity_and_matches_independent_spline():
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    times = np.array([0., .1, .3, .6])
    positions = np.array([[0., 0., 0.], [.01, -.02, .003], [.04, -.01, .005], [.05, 0., .01]])
    _, slopes, _ = bounded_joint_curve(times, positions, [-np.inf]*3, [np.inf]*3)
    pose = torch.tensor(np.c_[positions, np.tile([0., 0., 0., 1.], (4, 1))], dtype=torch.float64)
    reference = SimpleNamespace(times_s=torch.tensor(times, dtype=torch.float64),
        body_pose_world=pose, body_twist_world=torch.tensor(np.c_[slopes, np.zeros((4,3))]),
        object_pose_world=pose, object_twist_world=torch.zeros((4,6),dtype=torch.float64),
        joint_positions_rad=torch.zeros((4,1,1),dtype=torch.float64),
        joint_velocities_radps=torch.zeros((4,1,1),dtype=torch.float64))
    evaluator=object.__new__(NaiveContactPlanReference)
    evaluator.device,evaluator.dtype=torch.device('cpu'),torch.float64
    points=np.linspace(0.,.6,301);sample=evaluator._sample(reference,torch.tensor(points))
    independent=CubicHermiteSpline(times,positions,slopes)
    np.testing.assert_allclose(sample[0][:,:3],independent(points),atol=1e-12)
    np.testing.assert_allclose(sample[1][:,:3],independent(points,1),atol=1e-12)
    for knot in times[1:-1]:
        near=evaluator._sample(reference,torch.tensor([knot-1e-9,knot+1e-9]))[1][:,:3]
        torch.testing.assert_close(near[0],near[1],atol=2e-8,rtol=0.)
    outside=evaluator._sample(reference,torch.tensor([-.1,.7]))
    torch.testing.assert_close(outside[0][:,:3],pose[[0,-1],:3])
    assert torch.all(outside[1]==0.)


def test_rest_to_rest_retiming_preserves_points_and_reduces_endpoint_acceleration():
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    from amsrr.policies.naive_contact_planner import rest_to_rest_knot_times
    t=np.linspace(0.,3.,31); points=np.c_[np.zeros(31),np.zeros(31),.1*t]
    retimed=rest_to_rest_knot_times(t)
    assert retimed[0]==0. and retimed[-1]==5.625
    assert np.all(np.diff(retimed)>=np.diff(t)-1e-12)
    curves=[]
    for tt in [t,retimed]:
        q,v,_=bounded_joint_curve(tt,points,[-np.inf]*3,[np.inf]*3)
        np.testing.assert_array_equal(q,points)
        curves.append(CubicHermiteSpline(tt,q,v))
    a0=np.linalg.norm(curves[0](t[0],2))
    a1=np.linalg.norm(curves[1](retimed[0],2))
    assert a1 < a0 / 10
    speed=np.linalg.norm(curves[1](np.linspace(0.,retimed[-1],1000),1),axis=-1)
    assert speed.max()<=.101
    np.testing.assert_allclose(curves[1]([0.,retimed[-1]],1),0.,atol=1e-12)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_reference_matches_scalar_curves_across_mixed_phases_and_holds(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from dataclasses import make_dataclass, fields
    from amsrr.training.order9_c3_nominal_runtime import Order9C3NominalTensorReference
    from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
    from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
    reference = object.__new__(NaiveContactPlanReference)
    reference.device, reference.dtype = torch.device(device), torch.float64
    reference.feedback_reference_clock = None
    reference._references = {}
    gen = torch.Generator().manual_seed(391)
    def random(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64).to(device)
    for index, phase in enumerate(ORDER9_OBJECT_TASK_PHASES):
        count = 2+index%3
        pose = random(count, 7)
        pose[:, 3:] /= pose[:, 3:].norm(dim=-1, keepdim=True)
        obj = random(count, 7)
        obj[:, 3:] /= obj[:, 3:].norm(dim=-1, keepdim=True)
        reference._references[phase.value] = SimpleNamespace(
            times_s=torch.linspace(0., 1+index*.1, count, device=device, dtype=torch.float64),
            body_pose_world=pose, body_twist_world=random(count, 6),
            object_pose_world=obj, object_twist_world=random(count, 6),
            joint_positions_rad=random(count, 2, 3), joint_velocities_radps=random(count, 2, 3))
    batch = 32
    # Only the reference-owned fields are needed by the scalar method.
    shapes = dict(desired_robot_root_pose_world=(7,), desired_robot_root_twist_world=(6,),
        nominal_joint_positions_rad=(2,3), nominal_joint_velocities_radps=(2,3),
        desired_object_pose_world=(7,), phase_goal_robot_root_pose_world=(7,),
        phase_goal_object_pose_world=(7,), phase_progress=())
    Target = make_dataclass('ReferenceTestTarget', [(name, torch.Tensor) for name in shapes])
    target = Target(**{name:random(batch, *shape) for name,shape in shapes.items()})
    phases = torch.arange(batch, device=device)%len(ORDER9_OBJECT_TASK_PHASES)
    phases[-2:] = torch.tensor([-1,99],device=device)
    elapsed = torch.tensor([-1., 0., .37, 2.]*8,device=device,dtype=torch.float64)
    origins = random(batch, 3)
    for shift in (0., .1):
        expected = Order9C3NominalTensorReference.condition(reference, target,
            phase_index=phases, phase_elapsed_s=elapsed+shift, scene_origins=origins)
        actual = reference.condition(target,phase_index=phases,phase_elapsed_s=elapsed+shift,scene_origins=origins)
        for field in fields(target):
            torch.testing.assert_close(getattr(actual,field.name),getattr(expected,field.name),rtol=1e-12,atol=1e-12)
