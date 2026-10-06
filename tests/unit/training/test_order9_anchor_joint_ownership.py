"""Exercise the Isaac harness helper without importing SimulationApp."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _helper(jacobian):
    from scripts.run_request_policy import _with_unobservable_contact_failure
    path = Path(__file__).resolve().parents[3] / 'scripts/order9_vectorized_isaac_rollout.py'
    node = next(n for n in ast.parse(_with_unobservable_contact_failure(path.read_text())).body
                if isinstance(n, ast.FunctionDef) and n.name == '_deployable_anchor_joint_owner_mask')
    namespace = dict(torch=torch, math=math, MorphologyGraph=object,
        resolve_mesh_backed_anchor_references=lambda *args: (),
        WholeStructureKinematics=lambda: SimpleNamespace(compute=lambda *args: SimpleNamespace(
            ordered_global_dock_joint_ids=('module_0:a', 'module_0:b'),
            anchor_jacobians={0: jacobian})))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name]


@pytest.mark.parametrize('closure,expected', [([0., 0.], [False, False]),
                                               ([.1, 0.], [True, False]),
                                               ([0., .1], [False, True])])
def test_owner_preserves_motion_rule_and_represents_unobservable_anchor(closure, expected):
    helper = _helper([[0., 1.]] + [[0., 0.]] * 5)
    graph = SimpleNamespace(robot_anchors=[SimpleNamespace(anchor_id=0, module_id=0,
                            capability={'dock_mechanism_joint_id': 'a'})])
    mask = helper(morphology=graph, physical_model=None, selected_anchor_ids=[0],
                  module_ids=[0], local_joint_ids=['a', 'b'],
                  contact_joint_positions_rad=torch.zeros(1, 2),
                  closure_direction_rad=torch.tensor([closure]))
    assert mask.tolist() == [[expected]]
    # A missing owner cannot fabricate contact-load evidence, even if unrelated
    # motors report large loads. The actual request gate requires every anchor.
    loads = (torch.tensor([[[100., 100.]]])[:, None] * mask[None]).sum(dim=(-1, -2))
    assert bool((loads >= .05).all()) == any(expected)
