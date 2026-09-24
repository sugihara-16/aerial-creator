"""Exact recurrent math and output-lifetime checks for rollout CUDA replay."""
from dataclasses import fields, is_dataclass, replace
import importlib.util
from pathlib import Path

import pytest
import torch

from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_tensor_reward import Order9TensorRewardEngine
from amsrr.training.order9_deployable_phase_gate import Order9DeployablePhaseGate
from amsrr.training.order9_anchor_normal_force_estimator import Order9AnchorNormalForceEstimator
from amsrr.utils.tensor_dataclass_graph import TensorDataclassGraph


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(value, device='cuda'):
    if isinstance(value, torch.Tensor):
        return value.to(device).clone()
    if is_dataclass(value):
        return type(value)(**{f.name: convert(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: convert(v, device) for k, v in value.items()}
    return value


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif is_dataclass(left):
        for field in fields(left):
            same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    else:
        assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA replay')
@pytest.mark.parametrize('kind', ['reward', 'gate'])
def test_replay_preserves_all_phases_recurrence_and_previous_outputs(kind):
    fixture = fixture_module('test_order9_tensor_reward' if kind == 'reward'
                             else 'test_order9_deployable_phase_gate')
    cls = Order9TensorRewardEngine if kind == 'reward' else Order9DeployablePhaseGate
    eager, graph = cls(), cls()
    graph._use_cuda_graph = True
    initial = convert(fixture._evidence(ORDER9_OBJECT_TASK_PHASES[0]))
    if kind == 'reward':
        left = eager.initial_state(object_pose_world=initial.object_pose_world,
                                   desired_object_pose_world=initial.desired_object_pose_world)
    else:
        left = eager.initial_state(batch_size=1, device='cuda')
    right = convert(left)
    previous = None
    for repeat in range(3):
        for phase in ORDER9_OBJECT_TASK_PHASES:
            evidence = convert(fixture._evidence(phase))
            if kind == 'reward':
                evidence = replace(evidence, supported_placement=(None if repeat == 0 else
                    torch.tensor([repeat == 2], device='cuda')))
            evidence.object_pose_world[:, 0] += .001 * repeat
            a, b = eager.step(evidence, left), graph.step(evidence, right)
            same(a, b)
            if previous is not None:
                same(*previous)
            previous = b, convert(b)
            left, right = a.next_state, b.next_state
    evidence.local_joint_positions_rad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        graph.step(evidence, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA replay')
def test_estimator_replay_refreshes_inputs_shapes_and_retained_state():
    fixture = fixture_module('test_order9_anchor_normal_force_estimator')
    eager, graph = Order9AnchorNormalForceEstimator(), Order9AnchorNormalForceEstimator()
    graph._use_cuda_graph = True
    previous = None
    for batch in (1, 2, 1):
        jacobian, normals, owner = [v.cuda() for v in fixture._inputs(batch)]
        left = eager.initial_state(batch_size=batch, anchor_count=2, joint_count=4, device='cuda')
        right = convert(left)
        for step in range(8):
            args = dict(applied_joint_torque_nm=torch.full((batch, 4), .03 * step, device='cuda'),
                        gravity_joint_torque_nm=torch.zeros(batch, 4, device='cuda'),
                        grasp_point_linear_jacobian_world=jacobian, contact_normal_world=normals,
                        anchor_joint_owner_mask=owner,
                        selected_anchor_mask=torch.ones(batch, 2, device='cuda', dtype=torch.bool),
                        baseline_update_mask=torch.full((batch,), step == 0, device='cuda', dtype=torch.bool),
                        estimation_active_mask=torch.full((batch,), step != 0, device='cuda', dtype=torch.bool))
            a, b = eager.step(**args, state=left), graph.step(**args, state=right)
            same(a, b)
            if previous is not None:
                same(*previous)
            previous = b, convert(b)
            left, right = a.next_state, b.next_state
        args['applied_joint_torque_nm'][0, 0] = torch.nan
        with pytest.raises(ValueError, match='non-finite'):
            graph.step(**args, state=right)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_graph_does_not_detach_differentiable_input(device):
    x = torch.ones(3, device=device, requires_grad=True)
    output = TensorDataclassGraph().call(lambda a: a * a, (x,), configuration=None)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
