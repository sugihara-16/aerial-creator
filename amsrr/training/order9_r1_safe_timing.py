from __future__ import annotations

"""Fail-closed, morphology-aware physical timing for R1 teacher replay.

The R1 v5 experiment showed that a geometric-path and joint-rate check is not
enough to admit time compression: the closed-loop controller can still hit a
support or lose QP feasibility.  This module therefore exposes only a small,
fixed table of physical-time profiles.  Every admitted module count has direct
two-replay Isaac evidence; unknown module counts fail closed.
"""

import math
from collections.abc import Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES

ORDER9_R1_SAFE_TIMING_VERSION = "order9_r1_safe_morphology_timing_v1"
ORDER9_R1_SAFE_TIMING_BASIS = "module_count"

_PHASES = tuple(value.value for value in ORDER9_OBJECT_TASK_PHASES)
_FAST = {
    "approach": 0.5,
    "contact_acquisition": 0.5,
    "lift": 0.1,
    "transport": 0.1,
    "place": 0.1,
    "release": 0.1,
    "retreat": 0.1,
    "settle": 0.1,
}
_CONTACT_SAFE = {
    "approach": 0.75,
    "contact_acquisition": 0.75,
    "lift": 0.1,
    "transport": 0.1,
    "place": 0.1,
    "release": 0.1,
    "retreat": 0.1,
    "settle": 0.1,
}

# Module counts 2 and 5 did not need extra contact-acquisition time.  Counts 3
# and above receive the contact-safe lead-in.  Post-release timing can remain
# short because v6 replaces the erroneous whole-approach reversal with the
# task's actual 0.10 m retreat before this time parameterization is applied.
# Formal selection/confirmation still evaluates every candidate.
ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT = {
    2: dict(_FAST),
    3: dict(_CONTACT_SAFE),
    4: dict(_CONTACT_SAFE),
    5: dict(_FAST),
    6: dict(_CONTACT_SAFE),
    7: dict(_CONTACT_SAFE),
    8: dict(_CONTACT_SAFE),
}


def order9_r1_safe_phase_time_scales(module_count: int) -> dict[str, float]:
    """Return a defensive copy of the admitted R1 physical-time profile."""

    if isinstance(module_count, bool) or not isinstance(module_count, int):
        raise SchemaValidationError("R1 safe timing module count must be integral")
    try:
        profile = ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT[module_count]
    except KeyError as error:
        raise SchemaValidationError(
            f"R1 safe timing has no admitted profile for {module_count} modules"
        ) from error
    validate_order9_r1_safe_phase_time_scales(profile)
    return dict(profile)


def validate_order9_r1_safe_phase_time_scales(
    phase_time_scales: Mapping[str, float],
) -> None:
    """Reject missing, expanded, non-finite, or non-positive phase timing."""

    if set(phase_time_scales) != set(_PHASES):
        raise SchemaValidationError("R1 safe timing phases are incomplete")
    if any(
        not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0
        for value in phase_time_scales.values()
    ):
        raise SchemaValidationError("R1 safe timing scale is outside (0, 1]")


__all__ = [
    "ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT",
    "ORDER9_R1_SAFE_TIMING_BASIS",
    "ORDER9_R1_SAFE_TIMING_VERSION",
    "order9_r1_safe_phase_time_scales",
    "validate_order9_r1_safe_phase_time_scales",
]
