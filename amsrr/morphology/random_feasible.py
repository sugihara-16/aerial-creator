from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from amsrr.feasibility.morphology_flight import MorphologyFlightFeasibilityChecker
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel


@dataclass(frozen=True)
class RandomFeasibleMorphologyConfig:
    """Bounded rejection policy for the Order-1 feasible distribution."""

    max_attempts_per_sample: int = 256

    def __post_init__(self) -> None:
        if self.max_attempts_per_sample <= 0:
            raise SchemaValidationError(
                "RandomFeasibleMorphologyConfig.max_attempts_per_sample must be positive"
            )


@dataclass
class RandomFeasibleMorphologySample:
    morphology_graph: MorphologyGraph
    feasibility_result: FeasibilityResult
    requested_seed: int
    accepted_proposal_seed: int
    attempt_count: int
    duplicate_rejection_count: int
    rejected_violation_counts: dict[str, int] = field(default_factory=dict)

    @property
    def structural_hash(self) -> str:
        return morphology_structural_hash(self.morphology_graph)

    def to_dict(self) -> dict[str, object]:
        return {
            "morphology_graph": self.morphology_graph.to_dict(),
            "feasibility_result": self.feasibility_result.to_dict(),
            "requested_seed": self.requested_seed,
            "accepted_proposal_seed": self.accepted_proposal_seed,
            "attempt_count": self.attempt_count,
            "duplicate_rejection_count": self.duplicate_rejection_count,
            "rejected_violation_counts": dict(self.rejected_violation_counts),
            "structural_hash": self.structural_hash,
        }


class RandomFeasibleConnectedMorphologyDistribution:
    """Task-independent feasible distribution over connected Holon trees.

    The constructive proposal owns the probability measure. This wrapper
    conditions it on deterministic morphology-flight feasibility and, for a
    batch, on a unique canonical structural hash. A bounded failure raises
    instead of returning an unchecked graph.
    """

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        proposal: RandomConnectedMorphologyDistribution | None = None,
        feasibility_checker: MorphologyFlightFeasibilityChecker | None = None,
        config: RandomFeasibleMorphologyConfig | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.proposal = proposal or RandomConnectedMorphologyDistribution(physical_model)
        self.feasibility_checker = feasibility_checker or MorphologyFlightFeasibilityChecker()
        self.config = config or RandomFeasibleMorphologyConfig()

    def sample(
        self,
        *,
        seed: int,
        module_count: int | None = None,
        excluded_structural_hashes: set[str] | None = None,
    ) -> MorphologyGraph:
        return self.sample_with_report(
            seed=seed,
            module_count=module_count,
            excluded_structural_hashes=excluded_structural_hashes,
        ).morphology_graph

    def sample_with_report(
        self,
        *,
        seed: int,
        module_count: int | None = None,
        excluded_structural_hashes: set[str] | None = None,
    ) -> RandomFeasibleMorphologySample:
        rng = _rng_for_seed(seed)
        return self._sample_from_rng(
            rng,
            requested_seed=seed,
            module_count=module_count,
            excluded_structural_hashes=set(excluded_structural_hashes or ()),
        )

    def samples(
        self,
        *,
        seed: int,
        count: int,
        module_count: int | None = None,
    ) -> list[MorphologyGraph]:
        return [
            sample.morphology_graph
            for sample in self.sample_reports(seed=seed, count=count, module_count=module_count)
        ]

    def sample_reports(
        self,
        *,
        seed: int,
        count: int,
        module_count: int | None = None,
    ) -> list[RandomFeasibleMorphologySample]:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SchemaValidationError("random feasible morphology sample count must be positive")
        rng = _rng_for_seed(seed)
        seen: set[str] = set()
        reports: list[RandomFeasibleMorphologySample] = []
        for _ in range(count):
            report = self._sample_from_rng(
                rng,
                requested_seed=seed,
                module_count=module_count,
                excluded_structural_hashes=seen,
            )
            seen.add(report.structural_hash)
            reports.append(report)
        return reports

    def _sample_from_rng(
        self,
        rng: random.Random,
        *,
        requested_seed: int,
        module_count: int | None,
        excluded_structural_hashes: set[str],
    ) -> RandomFeasibleMorphologySample:
        rejected_codes: Counter[str] = Counter()
        duplicate_rejections = 0
        selected_module_count = module_count
        if selected_module_count is None:
            selected_module_count = rng.randint(
                self.proposal.config.min_modules,
                self.proposal.config.max_modules,
            )
        for attempt in range(1, self.config.max_attempts_per_sample + 1):
            proposal_seed = rng.getrandbits(63)
            morphology = self.proposal.sample(
                seed=proposal_seed,
                module_count=selected_module_count,
            )
            structural_hash = morphology_structural_hash(morphology)
            if structural_hash in excluded_structural_hashes:
                duplicate_rejections += 1
                continue
            feasibility = self.feasibility_checker.check(morphology, self.physical_model)
            if feasibility.feasible:
                return RandomFeasibleMorphologySample(
                    morphology_graph=morphology,
                    feasibility_result=feasibility,
                    requested_seed=requested_seed,
                    accepted_proposal_seed=proposal_seed,
                    attempt_count=attempt,
                    duplicate_rejection_count=duplicate_rejections,
                    rejected_violation_counts=dict(sorted(rejected_codes.items())),
                )
            rejected_codes.update(violation.code for violation in feasibility.hard_violations)
        detail = ", ".join(f"{code}={count}" for code, count in sorted(rejected_codes.items()))
        if duplicate_rejections:
            detail = f"{detail}, duplicates={duplicate_rejections}" if detail else f"duplicates={duplicate_rejections}"
        raise SchemaValidationError(
            "random feasible morphology rejection budget exhausted "
            f"after {self.config.max_attempts_per_sample} attempts"
            + (f" ({detail})" if detail else "")
        )


def _rng_for_seed(seed: int) -> random.Random:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SchemaValidationError("random feasible morphology seed must be a non-negative integer")
    return random.Random(seed)
