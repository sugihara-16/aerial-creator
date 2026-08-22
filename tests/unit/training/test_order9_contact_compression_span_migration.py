import math

import pytest

from amsrr.training.order9_contact_compression_span_migration import (
    rescale_module_count_bias_for_span,
)


def test_rescale_bias_preserves_bias_only_physical_displacement() -> None:
    before = 0.8569126725
    after = rescale_module_count_bias_for_span(
        before, old_span_m=0.01, new_span_m=0.02
    )
    assert 0.02 * math.tanh(after) == pytest.approx(
        0.01 * math.tanh(before), abs=1.0e-12
    )


@pytest.mark.parametrize(
    "old_span,new_span",
    [(0.0, 0.02), (0.01, 0.0), (0.02, 0.01), (math.nan, 0.02)],
)
def test_rescale_bias_rejects_invalid_span(old_span: float, new_span: float) -> None:
    with pytest.raises(ValueError):
        rescale_module_count_bias_for_span(
            0.5, old_span_m=old_span, new_span_m=new_span
        )
