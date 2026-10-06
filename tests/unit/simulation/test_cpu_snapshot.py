from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from amsrr.utils.tensor_snapshot import cpu_snapshot


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_snapshot_preserves_nested_values_and_owns_storage(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    @dataclass(frozen=True)
    class Observation:
        pose: torch.Tensor
        metadata: str
    original = dict(observation=Observation(torch.arange(12., device=device).reshape(3, 4).T, 'pose'),
        nested=SimpleNamespace(mask=torch.tensor([True, False], device=device),
            index=torch.tensor(7, device=device)), empty=torch.empty(0, device=device))
    result = cpu_snapshot(original)
    assert result['observation'].metadata == 'pose'
    assert torch.equal(result['observation'].pose, original['observation'].pose.cpu())
    assert result['nested'].mask.dtype == torch.bool
    assert result['nested'].index.item() == 7
    assert result['empty'].shape == (0,)
    result['observation'].pose.zero_()
    assert original['observation'].pose.sum().item() == 66


def test_cpu_tensor_graph_updates_inputs_keeps_owned_outputs_and_autograd():
    from amsrr.utils.tensor_dataclass_graph import TensorDataclassGraph
    @dataclass(frozen=True)
    class Result:
        values: torch.Tensor
        identity: str
    graph=TensorDataclassGraph()
    def function(x, scale):
        return dict(result=Result(torch.sin(x)*scale,'fixed'),state=(x+1,None))
    first=None
    for count,shift,scale in [(2,.1,2.),(2,.5,2.),(3,-.2,2.),(3,.4,3.)]:
        x=torch.arange(count,dtype=torch.float64)+shift
        expected=function(x,scale)
        actual=graph.call(function,args=(x,scale),configuration=scale)
        torch.testing.assert_close(actual['result'].values,expected['result'].values,rtol=1e-14,atol=1e-14)
        assert actual['result'].identity=='fixed' and actual['state'][1] is None
        if first is None:first=(actual,actual['result'].values.clone())
        x.fill_(99.)
        torch.testing.assert_close(first[0]['result'].values,first[1],rtol=0,atol=0)
    x=torch.tensor([.3],requires_grad=True)
    graph.call(function,args=(x,2.),configuration=2.)['result'].values.sum().backward()
    torch.testing.assert_close(x.grad,2*torch.cos(x))
