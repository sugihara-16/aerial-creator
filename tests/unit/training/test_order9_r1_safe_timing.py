from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_safe_timing import (
    ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT,
    order9_r1_safe_phase_time_scales,
)


def test_safe_timing_uses_measured_profiles_for_every_supported_count() -> None:
    assert order9_r1_safe_phase_time_scales(2)["lift"] == 0.1
    assert order9_r1_safe_phase_time_scales(3)["approach"] == 0.75
    assert order9_r1_safe_phase_time_scales(3)["retreat"] == 0.1
    assert order9_r1_safe_phase_time_scales(4)["approach"] == 0.75
    assert order9_r1_safe_phase_time_scales(5)["settle"] == 0.1
    assert order9_r1_safe_phase_time_scales(6)["contact_acquisition"] == 0.75
    assert order9_r1_safe_phase_time_scales(7)["settle"] == 0.1
    assert order9_r1_safe_phase_time_scales(8)["transport"] == 0.1


def test_safe_timing_is_fail_closed_and_returns_a_copy() -> None:
    with pytest.raises(SchemaValidationError, match="no admitted profile"):
        order9_r1_safe_phase_time_scales(9)
    value = order9_r1_safe_phase_time_scales(2)
    value["approach"] = 1.0
    assert ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT[2]["approach"] == 0.5
