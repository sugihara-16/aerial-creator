from __future__ import annotations

"""C3 reset-stratum contract for late predecessor-phase exposure."""

import math
from typing import Sequence


ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT = (
    "order9_c3_boundary_tail_sampling_v2_place_release_retreat"
)
ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT = (
    "order9_c3_transition_backward_curriculum_v1_fixed_progress"
)


def order9_c3_fixed_reset_stratum_index(
    *,
    progress_fractions: Sequence[float],
    reset_progress_fraction: float,
) -> int:
    """Resolve one exact persisted reset stratum for boundary curricula.

    The backward curriculum must start from a real state stored in the
    accepted-nominal reset bank.  Nearest-neighbour selection would silently
    change the requested physical starting state when the bank changes, so the
    configured fraction must identify exactly one persisted stratum.
    """

    fractions = tuple(float(value) for value in progress_fractions)
    target = float(reset_progress_fraction)
    if (
        not fractions
        or fractions != tuple(sorted(set(fractions)))
        or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in fractions
        )
        or not math.isfinite(target)
        or not 0.0 < target < 1.0
    ):
        raise ValueError("Order9 C3 fixed reset progress is invalid")
    matching = [
        index
        for index, value in enumerate(fractions)
        if math.isclose(value, target, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(matching) != 1:
        raise ValueError(
            "Order9 C3 fixed reset progress must identify one persisted stratum"
        )
    return matching[0]


def order9_c3_reset_strata_by_phase(
    *,
    phase_labels: Sequence[str],
    progress_fractions: Sequence[float],
    boundary_tail_phase_labels: Sequence[str],
    boundary_tail_progress_fraction: float,
) -> tuple[tuple[int, ...], ...]:
    """Return selectable reset strata without broadening unrelated phases."""

    phases = tuple(str(value) for value in phase_labels)
    fractions = tuple(float(value) for value in progress_fractions)
    boundary_phases = tuple(str(value) for value in boundary_tail_phase_labels)
    tail = float(boundary_tail_progress_fraction)
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("Order9 C3 phase labels must be non-empty and unique")
    if (
        len(fractions) < 4
        or fractions != tuple(sorted(set(fractions)))
        or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in fractions
        )
    ):
        raise ValueError("Order9 C3 reset fractions are invalid")
    if (
        not boundary_phases
        or len(set(boundary_phases)) != len(boundary_phases)
        or any(value not in phases for value in boundary_phases)
    ):
        raise ValueError("Order9 C3 boundary-tail phase labels are invalid")
    matching = [
        index
        for index, value in enumerate(fractions)
        if math.isclose(value, tail, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(matching) != 1:
        raise ValueError(
            "Order9 C3 boundary-tail fraction must identify one reset stratum"
        )
    tail_index = matching[0]
    base = tuple(index for index in range(len(fractions)) if index != tail_index)
    if len(base) < 3:
        raise ValueError("Order9 C3 requires at least three base reset strata")
    full = tuple(range(len(fractions)))
    boundary_set = set(boundary_phases)
    return tuple(full if phase in boundary_set else base for phase in phases)


__all__ = [
    "ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT",
    "ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT",
    "order9_c3_fixed_reset_stratum_index",
    "order9_c3_reset_strata_by_phase",
]
