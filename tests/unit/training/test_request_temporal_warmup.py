import pytest
import torch
from amsrr.training.request_temporal_warmup import (
    STATE_START, STATE_END, TIME_COLUMNS, retimed_head_inputs,
)


def test_retiming_preserves_progress_feedback_and_source_data():
    raw = torch.ones(2, 3, STATE_END)
    raw[:, :, TIME_COLUMNS[0]] = 3.
    raw[:, :, TIME_COLUMNS[1]] = 4.
    mean, scale = torch.zeros(STATE_END), torch.ones(STATE_END)
    encoded = torch.cat((raw, torch.ones(2, 3, 12)), dim=-1)
    saved = encoded.clone()
    result = retimed_head_inputs(encoded, raw, torch.tensor([2., .5]), mean, scale)
    assert torch.equal(encoded, saved)
    assert torch.equal(result[:, 0, TIME_COLUMNS[0]], torch.tensor([6., 1.5]))
    assert torch.equal(result[:, 0, TIME_COLUMNS[1]], torch.tensor([8., 2.]))
    keep = [i for i in range(STATE_START, STATE_END) if i not in TIME_COLUMNS]
    assert torch.equal(result[:, :, keep], encoded[:, :, keep])
    assert not result[:, :, :STATE_START].count_nonzero()
    assert not result[:, :, STATE_END:].count_nonzero()


@pytest.mark.parametrize('factor', [0., -1., float('nan'), float('inf')])
def test_invalid_retiming_rejected(factor):
    with pytest.raises(ValueError, match='retiming'):
        retimed_head_inputs(torch.zeros(1, 2, STATE_END + 4),
            torch.zeros(1, 2, STATE_END), torch.tensor([factor]),
            torch.zeros(STATE_END), torch.ones(STATE_END))
