from __future__ import annotations

import pytest

from amsrr.training.order9_c3_boundary_sampling import (
    ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT,
    ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT,
    order9_c3_fixed_reset_stratum_index,
    order9_c3_reset_strata_by_phase,
)


PHASES = (
    "approach",
    "contact_acquisition",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
)
FRACTIONS = (1.0 / 6.0, 0.5, 2.0 / 3.0, 0.9)


def test_boundary_tail_is_selectable_for_all_late_phase_boundaries() -> None:
    strata = order9_c3_reset_strata_by_phase(
        phase_labels=PHASES,
        progress_fractions=FRACTIONS,
        boundary_tail_phase_labels=("place", "release", "retreat"),
        boundary_tail_progress_fraction=0.9,
    )

    assert ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT.endswith(
        "place_release_retreat"
    )
    assert strata[PHASES.index("place")] == (0, 1, 2, 3)
    assert strata[PHASES.index("release")] == (0, 1, 2, 3)
    assert strata[PHASES.index("retreat")] == (0, 1, 2, 3)
    assert all(
        indices == (0, 1, 2)
        for phase, indices in zip(PHASES, strata, strict=True)
        if phase not in {"place", "release", "retreat"}
    )


def test_boundary_tail_requires_a_configured_fraction_and_known_phase() -> None:
    with pytest.raises(ValueError, match="identify one reset stratum"):
        order9_c3_reset_strata_by_phase(
            phase_labels=PHASES,
            progress_fractions=FRACTIONS,
            boundary_tail_phase_labels=("place", "release", "retreat"),
            boundary_tail_progress_fraction=0.85,
        )
    with pytest.raises(ValueError, match="phase labels"):
        order9_c3_reset_strata_by_phase(
            phase_labels=PHASES,
            progress_fractions=FRACTIONS,
            boundary_tail_phase_labels=("place", "release", "unknown"),
            boundary_tail_progress_fraction=0.9,
        )


def test_transition_backward_curriculum_resolves_an_exact_persisted_state() -> None:
    assert ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT.endswith(
        "fixed_progress"
    )
    assert (
        order9_c3_fixed_reset_stratum_index(
            progress_fractions=FRACTIONS,
            reset_progress_fraction=0.9,
        )
        == 3
    )
    with pytest.raises(ValueError, match="identify one persisted stratum"):
        order9_c3_fixed_reset_stratum_index(
            progress_fractions=FRACTIONS,
            reset_progress_fraction=0.85,
        )
