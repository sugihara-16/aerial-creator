from __future__ import annotations

import pytest

from amsrr.feasibility.morphology_flight import (
    MorphologyFlightFeasibilityChecker,
    MorphologyFlightFeasibilityConfig,
)
from amsrr.morphology.random_feasible import (
    RandomFeasibleConnectedMorphologyDistribution,
    RandomFeasibleMorphologyConfig,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError


MESH_SEARCH_DIRS = ("module_urdf", "module_urdf/mesh")


def _distribution(
    *,
    feasibility_config: MorphologyFlightFeasibilityConfig | None = None,
    rejection_config: RandomFeasibleMorphologyConfig | None = None,
) -> RandomFeasibleConnectedMorphologyDistribution:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    checker = MorphologyFlightFeasibilityChecker(
        feasibility_config
        or MorphologyFlightFeasibilityConfig(mesh_search_dirs=MESH_SEARCH_DIRS)
    )
    return RandomFeasibleConnectedMorphologyDistribution(
        physical_model,
        feasibility_checker=checker,
        config=rejection_config,
    )


def test_random_feasible_sample_is_deterministic_and_fail_closed() -> None:
    distribution = _distribution()

    first = distribution.sample_with_report(seed=17)
    second = distribution.sample_with_report(seed=17)

    assert first.to_dict() == second.to_dict()
    assert first.feasibility_result.feasible is True
    assert 2 <= len(first.morphology_graph.modules) <= 8
    assert first.attempt_count >= 1


@pytest.mark.parametrize("module_count", range(2, 9))
def test_random_feasible_distribution_supplies_every_supported_size(module_count: int) -> None:
    report = _distribution().sample_with_report(seed=100 + module_count, module_count=module_count)

    assert len(report.morphology_graph.modules) == module_count
    assert report.feasibility_result.feasible is True


def test_random_feasible_batch_rejects_canonical_duplicates() -> None:
    reports = _distribution().sample_reports(seed=22, count=12)
    hashes = [report.structural_hash for report in reports]

    assert len(hashes) == len(set(hashes))
    assert all(report.feasibility_result.feasible for report in reports)


def test_random_feasible_exclusion_returns_a_different_structure() -> None:
    distribution = _distribution()
    first = distribution.sample_with_report(seed=9)
    second = distribution.sample_with_report(
        seed=9,
        excluded_structural_hashes={first.structural_hash},
    )

    assert second.structural_hash != first.structural_hash
    assert second.duplicate_rejection_count >= 1


def test_rejection_sampling_keeps_the_once_sampled_module_count() -> None:
    baseline = _distribution()
    first = baseline.sample_with_report(seed=9)

    class RecordingProposal:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.config = delegate.config
            self.module_counts: list[int | None] = []

        def sample(self, *, seed: int, module_count: int | None = None):
            self.module_counts.append(module_count)
            return self.delegate.sample(seed=seed, module_count=module_count)

    recorder = RecordingProposal(baseline.proposal)
    distribution = RandomFeasibleConnectedMorphologyDistribution(
        baseline.physical_model,
        proposal=recorder,  # type: ignore[arg-type]
        feasibility_checker=baseline.feasibility_checker,
    )
    second = distribution.sample_with_report(
        seed=9,
        excluded_structural_hashes={first.structural_hash},
    )

    assert second.duplicate_rejection_count >= 1
    assert len(recorder.module_counts) >= 2
    assert set(recorder.module_counts) == {len(second.morphology_graph.modules)}


def test_random_feasible_distribution_reports_exhausted_rejection_budget() -> None:
    impossible_checker_config = MorphologyFlightFeasibilityConfig(
        mesh_search_dirs=MESH_SEARCH_DIRS,
        min_thrust_margin_ratio=100.0,
    )
    distribution = _distribution(
        feasibility_config=impossible_checker_config,
        rejection_config=RandomFeasibleMorphologyConfig(max_attempts_per_sample=3),
    )

    with pytest.raises(SchemaValidationError, match="rejection budget exhausted"):
        distribution.sample(seed=0)


def test_random_feasible_config_and_arguments_validate() -> None:
    with pytest.raises(SchemaValidationError, match="must be positive"):
        RandomFeasibleMorphologyConfig(max_attempts_per_sample=0)
    with pytest.raises(SchemaValidationError, match="non-negative integer"):
        _distribution().sample(seed=-1)
    with pytest.raises(SchemaValidationError, match="sample count must be positive"):
        _distribution().samples(seed=0, count=0)
