from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation import SimplifiedGraspCarryEnv, SimplifiedGraspCarryEnvConfig
from amsrr.training import (
    P1GraspCarryTaskDistribution,
    P1RunnerConfig,
    P1SimplifiedRunner,
    load_p1_runner_config,
)


@dataclass
class P1AcceptanceCriteria(SchemaBase):
    episode_count: int = 1000
    min_success_rate: float = 0.60
    candidate_sample_count: int = 16
    seed: int = 0
    candidate_seed_offset: int = 100_000
    config_path: str = "configs/training/p1_grasp_carry_distribution.yaml"
    source_hash: str = "p1_acceptance"

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P1AcceptanceCriteria.episode_count must be positive")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise SchemaValidationError("P1AcceptanceCriteria.min_success_rate must be in [0, 1]")
        if self.candidate_sample_count <= 0:
            raise SchemaValidationError("P1AcceptanceCriteria.candidate_sample_count must be positive")
        if self.candidate_seed_offset < 0:
            raise SchemaValidationError("P1AcceptanceCriteria.candidate_seed_offset must be non-negative")
        if not self.config_path:
            raise SchemaValidationError("P1AcceptanceCriteria.config_path must be non-empty")
        if not self.source_hash:
            raise SchemaValidationError("P1AcceptanceCriteria.source_hash must be non-empty")


@dataclass
class P1AcceptanceReport(SchemaBase):
    passed: bool
    episode_count: int
    success_count: int
    crash_count: int
    failure_count: int
    success_rate: float
    min_success_rate: float
    candidate_sample_count: int
    non_empty_candidate_sample_count: int
    contact_candidate_counts: list[int] = field(default_factory=list)
    archive_count: int = 0
    archive_roundtrip_count: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P1AcceptanceReport.episode_count must be positive")
        if self.success_count < 0 or self.crash_count < 0 or self.failure_count < 0:
            raise SchemaValidationError("P1AcceptanceReport counts must be non-negative")
        if not 0.0 <= self.success_rate <= 1.0:
            raise SchemaValidationError("P1AcceptanceReport.success_rate must be in [0, 1]")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise SchemaValidationError("P1AcceptanceReport.min_success_rate must be in [0, 1]")
        if self.candidate_sample_count <= 0:
            raise SchemaValidationError("P1AcceptanceReport.candidate_sample_count must be positive")
        if self.non_empty_candidate_sample_count < 0:
            raise SchemaValidationError("P1AcceptanceReport.non_empty_candidate_sample_count must be non-negative")
        if self.non_empty_candidate_sample_count > self.candidate_sample_count:
            raise SchemaValidationError(
                "P1AcceptanceReport.non_empty_candidate_sample_count cannot exceed candidate_sample_count"
            )
        if len(self.contact_candidate_counts) != self.candidate_sample_count:
            raise SchemaValidationError(
                "P1AcceptanceReport.contact_candidate_counts length must match candidate_sample_count"
            )
        if self.archive_count < 0:
            raise SchemaValidationError("P1AcceptanceReport.archive_count must be non-negative")
        if self.archive_roundtrip_count is not None and self.archive_roundtrip_count < 0:
            raise SchemaValidationError("P1AcceptanceReport.archive_roundtrip_count must be non-negative")


def run_p1_acceptance(
    base_task_spec: TaskSpec,
    *,
    criteria: P1AcceptanceCriteria | None = None,
    archive_path: str | Path | None = None,
) -> P1AcceptanceReport:
    """Run the v0.4 Section 24.2 P1 acceptance gate on the simplified backend."""

    criteria = criteria or P1AcceptanceCriteria()
    loaded_runner_config, distribution_config, env_config = load_p1_runner_config(criteria.config_path)
    runner_config = P1RunnerConfig(
        episode_count=criteria.episode_count,
        seed=criteria.seed,
        source_hash=criteria.source_hash,
        simulator_version=loaded_runner_config.simulator_version,
        archive_success_only=loaded_runner_config.archive_success_only,
    )
    runner = P1SimplifiedRunner(
        base_task_spec,
        runner_config=runner_config,
        distribution_config=distribution_config,
        env_config=env_config,
    )
    runner_result = runner.run(archive_path=archive_path)

    distribution = P1GraspCarryTaskDistribution(base_task_spec, distribution_config)
    contact_candidate_counts = _contact_candidate_counts(
        distribution,
        criteria=criteria,
        env_config=env_config,
    )
    non_empty_count = sum(1 for count in contact_candidate_counts if count > 0)
    success_rate = runner_result.metrics.get(
        "success_rate",
        float(runner_result.success_count) / float(runner_result.episode_count),
    )
    failure_reasons = _failure_reasons(
        success_rate=success_rate,
        crash_count=runner_result.crash_count,
        candidate_sample_count=criteria.candidate_sample_count,
        non_empty_candidate_sample_count=non_empty_count,
        min_success_rate=criteria.min_success_rate,
    )
    archive_roundtrip_count = None
    if archive_path is not None:
        archive_roundtrip_count = len(read_episode_archives_jsonl(archive_path))

    return P1AcceptanceReport(
        passed=not failure_reasons,
        episode_count=runner_result.episode_count,
        success_count=runner_result.success_count,
        crash_count=runner_result.crash_count,
        failure_count=runner_result.failure_count,
        success_rate=success_rate,
        min_success_rate=criteria.min_success_rate,
        candidate_sample_count=criteria.candidate_sample_count,
        non_empty_candidate_sample_count=non_empty_count,
        contact_candidate_counts=contact_candidate_counts,
        archive_count=len(runner_result.archives),
        archive_roundtrip_count=archive_roundtrip_count,
        metrics=dict(runner_result.metrics),
        failure_reasons=failure_reasons,
    )


def _contact_candidate_counts(
    distribution: P1GraspCarryTaskDistribution,
    *,
    criteria: P1AcceptanceCriteria,
    env_config: SimplifiedGraspCarryEnvConfig,
) -> list[int]:
    counts: list[int] = []
    for index in range(criteria.candidate_sample_count):
        seed = criteria.seed + criteria.candidate_seed_offset + index
        sample = distribution.sample(seed=seed, sample_index=index)
        env = SimplifiedGraspCarryEnv(sample.task_spec, config=env_config)
        counts.append(len(env.artifacts.contact_candidate_set.candidates))
    return counts


def _failure_reasons(
    *,
    success_rate: float,
    crash_count: int,
    candidate_sample_count: int,
    non_empty_candidate_sample_count: int,
    min_success_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if success_rate < min_success_rate:
        reasons.append(f"success_rate {success_rate:.3f} < required {min_success_rate:.3f}")
    if crash_count > 0:
        reasons.append(f"schema/checker/controller crash_count {crash_count} > 0")
    if non_empty_candidate_sample_count < candidate_sample_count:
        reasons.append(
            "contact candidate sampler returned empty candidates for "
            f"{candidate_sample_count - non_empty_candidate_sample_count} valid samples"
        )
    return reasons
