from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from amsrr.feasibility.checker import DESIGN_HARD_CHECK_CODES
from amsrr.feasibility.violation_codes import F_CLOSED_LOOP_REJECT_V1
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging import EpisodeArchive, read_episode_archives_jsonl
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignPolicyConfig
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P2DesignDistributionConfig,
    P2DesignEvaluationRunner,
    P2DesignRunnerConfig,
    P2GraspCarryDesignDistribution,
    load_p2_design_runner_config,
)


P2_REQUIRED_FEASIBILITY_MARGIN_KEYS = (
    "required_slot_coverage_ratio",
    "required_slot_anchor_capability_coverage_ratio",
    "closed_loop_rejected",
    "port_conflict_count",
    "thrust_margin_ratio",
    "payload_margin_ratio",
    "coarse_reachability_ratio",
)


@dataclass
class P2AcceptanceCriteria(SchemaBase):
    episode_count: int = 1000
    min_valid_design_rate: float = 0.70
    min_required_slot_coverage: float = 0.90
    seed: int = 0
    config_path: str = "configs/training/p2_design_grasp_carry.yaml"
    source_hash: str = "p2_acceptance"
    closed_loop_probe_seed_offset: int = 200_000

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P2AcceptanceCriteria.episode_count must be positive")
        if not 0.0 <= self.min_valid_design_rate <= 1.0:
            raise SchemaValidationError("P2AcceptanceCriteria.min_valid_design_rate must be in [0, 1]")
        if not 0.0 <= self.min_required_slot_coverage <= 1.0:
            raise SchemaValidationError("P2AcceptanceCriteria.min_required_slot_coverage must be in [0, 1]")
        if not self.config_path:
            raise SchemaValidationError("P2AcceptanceCriteria.config_path must be non-empty")
        if not self.source_hash:
            raise SchemaValidationError("P2AcceptanceCriteria.source_hash must be non-empty")
        if self.closed_loop_probe_seed_offset < 0:
            raise SchemaValidationError("P2AcceptanceCriteria.closed_loop_probe_seed_offset must be non-negative")


@dataclass
class P2AcceptanceReport(SchemaBase):
    passed: bool
    episode_count: int
    valid_design_count: int
    invalid_design_count: int
    crash_count: int
    valid_design_rate: float
    min_valid_design_rate: float
    accepted_design_count: int
    accepted_required_slot_coverage_mean: float
    accepted_required_slot_coverage_min: float
    min_required_slot_coverage: float
    closed_loop_invalid_rejected: bool
    closed_loop_feasibility_label_stored: bool
    feasibility_label_archive_count: int
    feasibility_label_valid_count: int
    archive_count: int
    archive_roundtrip_count: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P2AcceptanceReport.episode_count must be positive")
        for name in ("valid_design_count", "invalid_design_count", "crash_count", "accepted_design_count"):
            if getattr(self, name) < 0:
                raise SchemaValidationError(f"P2AcceptanceReport.{name} must be non-negative")
        if not 0.0 <= self.valid_design_rate <= 1.0:
            raise SchemaValidationError("P2AcceptanceReport.valid_design_rate must be in [0, 1]")
        if not 0.0 <= self.min_valid_design_rate <= 1.0:
            raise SchemaValidationError("P2AcceptanceReport.min_valid_design_rate must be in [0, 1]")
        if not 0.0 <= self.accepted_required_slot_coverage_mean <= 1.0:
            raise SchemaValidationError("P2AcceptanceReport.accepted_required_slot_coverage_mean must be in [0, 1]")
        if not 0.0 <= self.accepted_required_slot_coverage_min <= 1.0:
            raise SchemaValidationError("P2AcceptanceReport.accepted_required_slot_coverage_min must be in [0, 1]")
        if not 0.0 <= self.min_required_slot_coverage <= 1.0:
            raise SchemaValidationError("P2AcceptanceReport.min_required_slot_coverage must be in [0, 1]")
        if self.feasibility_label_archive_count < 0 or self.feasibility_label_valid_count < 0:
            raise SchemaValidationError("P2AcceptanceReport label counts must be non-negative")
        if self.feasibility_label_valid_count > self.feasibility_label_archive_count:
            raise SchemaValidationError(
                "P2AcceptanceReport.feasibility_label_valid_count cannot exceed feasibility_label_archive_count"
            )
        if self.archive_count < 0:
            raise SchemaValidationError("P2AcceptanceReport.archive_count must be non-negative")
        if self.archive_roundtrip_count is not None and self.archive_roundtrip_count < 0:
            raise SchemaValidationError("P2AcceptanceReport.archive_roundtrip_count must be non-negative")


def run_p2_acceptance(
    base_task_spec: TaskSpec,
    *,
    criteria: P2AcceptanceCriteria | None = None,
    archive_path: str | Path | None = None,
) -> P2AcceptanceReport:
    """Run the v0.4 Section 24.3 P2 design acceptance gate."""

    criteria = criteria or P2AcceptanceCriteria()
    loaded_runner_config, distribution_config, policy_config = load_p2_design_runner_config(criteria.config_path)
    runner_config = P2DesignRunnerConfig(
        episode_count=criteria.episode_count,
        seed=criteria.seed,
        source_hash=criteria.source_hash,
        runner_version=loaded_runner_config.runner_version,
        robot_model_config_path=loaded_runner_config.robot_model_config_path,
        archive_success_only=loaded_runner_config.archive_success_only,
    )
    runner = P2DesignEvaluationRunner(
        base_task_spec,
        runner_config=runner_config,
        distribution_config=distribution_config,
        policy_config=policy_config,
    )
    runner_result = runner.run(archive_path=archive_path)

    archives = runner_result.archives
    archive_roundtrip_count = None
    if archive_path is not None:
        archives = read_episode_archives_jsonl(archive_path)
        archive_roundtrip_count = len(archives)

    accepted_coverage = _accepted_slot_coverages(archives)
    coverage_mean = sum(accepted_coverage) / max(1.0, float(len(accepted_coverage)))
    coverage_min = min(accepted_coverage) if accepted_coverage else 0.0
    label_valid_count = sum(1 for archive in archives if _feasibility_labels_are_stored_correctly(archive))
    closed_loop_result = _closed_loop_invalid_design_result(
        base_task_spec,
        criteria=criteria,
        distribution_config=distribution_config,
        policy_config=policy_config,
        robot_model_config_path=runner_config.robot_model_config_path,
    )
    closed_loop_rejected = _closed_loop_invalid_rejected(closed_loop_result)
    closed_loop_label_stored = _closed_loop_label_stored(closed_loop_result)
    valid_design_rate = runner_result.metrics.get(
        "valid_design_rate",
        float(runner_result.valid_design_count) / float(runner_result.episode_count),
    )
    failure_reasons = _failure_reasons(
        valid_design_rate=valid_design_rate,
        min_valid_design_rate=criteria.min_valid_design_rate,
        accepted_design_count=len(accepted_coverage),
        accepted_required_slot_coverage_min=coverage_min,
        min_required_slot_coverage=criteria.min_required_slot_coverage,
        closed_loop_invalid_rejected=closed_loop_rejected,
        closed_loop_feasibility_label_stored=closed_loop_label_stored,
        feasibility_label_archive_count=len(archives),
        feasibility_label_valid_count=label_valid_count,
    )

    return P2AcceptanceReport(
        passed=not failure_reasons,
        episode_count=runner_result.episode_count,
        valid_design_count=runner_result.valid_design_count,
        invalid_design_count=runner_result.invalid_design_count,
        crash_count=runner_result.crash_count,
        valid_design_rate=valid_design_rate,
        min_valid_design_rate=criteria.min_valid_design_rate,
        accepted_design_count=len(accepted_coverage),
        accepted_required_slot_coverage_mean=coverage_mean,
        accepted_required_slot_coverage_min=coverage_min,
        min_required_slot_coverage=criteria.min_required_slot_coverage,
        closed_loop_invalid_rejected=closed_loop_rejected,
        closed_loop_feasibility_label_stored=closed_loop_label_stored,
        feasibility_label_archive_count=len(archives),
        feasibility_label_valid_count=label_valid_count,
        archive_count=len(runner_result.archives),
        archive_roundtrip_count=archive_roundtrip_count,
        metrics=dict(runner_result.metrics),
        failure_reasons=failure_reasons,
    )


def _closed_loop_invalid_design_result(
    base_task_spec: TaskSpec,
    *,
    criteria: P2AcceptanceCriteria,
    distribution_config: P2DesignDistributionConfig,
    policy_config: P2DesignPolicyConfig,
    robot_model_config_path: str,
) -> FeasibilityResult:
    seed = criteria.seed + criteria.closed_loop_probe_seed_offset
    sample = P2GraspCarryDesignDistribution(base_task_spec, distribution_config).sample(seed=seed, sample_index=0)
    physical_model = build_physical_model_from_config(robot_model_config_path)
    irg = IRGBuilder().build(sample.task_spec)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    context = DesignPolicyContext(
        task_spec=sample.task_spec,
        irg=irg,
        physical_model=physical_model,
        interaction_envelope=envelope,
    )
    good_design = build_grasp_carry_variant_design_output(
        sample.task_spec,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP,
    )
    invalid_design = replace(
        good_design,
        target_morphology=replace(good_design.target_morphology, is_closed_loop=True),
    )
    selection = P2DesignPolicy(config=policy_config).evaluate_design_outputs(
        context,
        [(GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP, invalid_design)],
    )
    return selection.selected_candidate.feasibility_result


def _accepted_slot_coverages(archives: list[EpisodeArchive]) -> list[float]:
    coverages: list[float] = []
    for archive in archives:
        feasibility = archive.feasibility_result
        if feasibility is None or not archive.success or not feasibility.feasible:
            continue
        coverages.append(float(feasibility.margins.get("required_slot_coverage_ratio", 0.0)))
    return coverages


def _feasibility_labels_are_stored_correctly(archive: EpisodeArchive) -> bool:
    feasibility = archive.feasibility_result
    if feasibility is None:
        return False
    proxy_scores = feasibility.proxy_scores
    if proxy_scores.get("L_FEASIBLE") != (1.0 if feasibility.feasible else 0.0):
        return False
    if proxy_scores.get("L_HARD_VIOLATION") != (1.0 if feasibility.hard_violations else 0.0):
        return False
    if archive.metrics.get("label_L_FEASIBLE") != proxy_scores.get("L_FEASIBLE"):
        return False
    if archive.metrics.get("label_L_HARD_VIOLATION") != proxy_scores.get("L_HARD_VIOLATION"):
        return False
    for code in DESIGN_HARD_CHECK_CODES:
        key = f"L_{code}"
        if key not in proxy_scores:
            return False
    hard_violation_codes = {violation.code for violation in feasibility.hard_violations}
    for code in DESIGN_HARD_CHECK_CODES:
        expected = 1.0 if code in hard_violation_codes else 0.0
        if proxy_scores.get(f"L_{code}") != expected:
            return False
    for key in P2_REQUIRED_FEASIBILITY_MARGIN_KEYS:
        if key not in feasibility.margins:
            return False
    return True


def _closed_loop_invalid_rejected(feasibility: FeasibilityResult) -> bool:
    return (
        not feasibility.feasible
        and any(violation.code == F_CLOSED_LOOP_REJECT_V1 for violation in feasibility.hard_violations)
        and feasibility.margins.get("closed_loop_rejected") == 1.0
    )


def _closed_loop_label_stored(feasibility: FeasibilityResult) -> bool:
    return (
        feasibility.proxy_scores.get("L_FEASIBLE") == 0.0
        and feasibility.proxy_scores.get("L_HARD_VIOLATION") == 1.0
        and feasibility.proxy_scores.get(f"L_{F_CLOSED_LOOP_REJECT_V1}") == 1.0
    )


def _failure_reasons(
    *,
    valid_design_rate: float,
    min_valid_design_rate: float,
    accepted_design_count: int,
    accepted_required_slot_coverage_min: float,
    min_required_slot_coverage: float,
    closed_loop_invalid_rejected: bool,
    closed_loop_feasibility_label_stored: bool,
    feasibility_label_archive_count: int,
    feasibility_label_valid_count: int,
) -> list[str]:
    reasons: list[str] = []
    if valid_design_rate < min_valid_design_rate:
        reasons.append(f"valid_design_rate {valid_design_rate:.3f} < required {min_valid_design_rate:.3f}")
    if accepted_design_count <= 0:
        reasons.append("no accepted designs available for required_slot_coverage check")
    elif accepted_required_slot_coverage_min < min_required_slot_coverage:
        reasons.append(
            "accepted required_slot_coverage_min "
            f"{accepted_required_slot_coverage_min:.3f} < required {min_required_slot_coverage:.3f}"
        )
    if not closed_loop_invalid_rejected:
        reasons.append("closed_loop_invalid design was not rejected")
    if not closed_loop_feasibility_label_stored:
        reasons.append("closed_loop_invalid feasibility labels were not stored correctly")
    if feasibility_label_archive_count <= 0:
        reasons.append("no archives available for feasibility label storage check")
    elif feasibility_label_valid_count != feasibility_label_archive_count:
        reasons.append(
            "feasibility labels stored correctly for "
            f"{feasibility_label_valid_count}/{feasibility_label_archive_count} archives"
        )
    return reasons
