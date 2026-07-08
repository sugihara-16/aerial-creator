from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amsrr.assembly import (
    AssemblyRunner,
    AssemblyRunnerConfig,
    SimplifiedAssemblyExecutor,
    SimplifiedAssemblyExecutorConfig,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging import read_episode_archives_jsonl
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignPolicyConfig
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P2DesignDistributionConfig,
    P2GraspCarryDesignDistribution,
    P3AssemblyEvaluationRunner,
    P3AssemblyRunnerConfig,
    load_p3_assembly_runner_config,
)


@dataclass
class P3AcceptanceCriteria(SchemaBase):
    episode_count: int = 1000
    min_assembly_success_rate: float = 0.70
    seed: int = 0
    config_path: str = "configs/training/p3_assembly_grasp_carry.yaml"
    source_hash: str = "p3_acceptance"
    retry_probe_seed_offset: int = 300_000
    abort_probe_seed_offset: int = 400_000

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P3AcceptanceCriteria.episode_count must be positive")
        if not 0.0 <= self.min_assembly_success_rate <= 1.0:
            raise SchemaValidationError("P3AcceptanceCriteria.min_assembly_success_rate must be in [0, 1]")
        if not self.config_path:
            raise SchemaValidationError("P3AcceptanceCriteria.config_path must be non-empty")
        if not self.source_hash:
            raise SchemaValidationError("P3AcceptanceCriteria.source_hash must be non-empty")
        if self.retry_probe_seed_offset < 0 or self.abort_probe_seed_offset < 0:
            raise SchemaValidationError("P3AcceptanceCriteria probe seed offsets must be non-negative")


@dataclass
class P3AcceptanceReport(SchemaBase):
    passed: bool
    episode_count: int
    assembly_success_count: int
    assembly_failure_count: int
    crash_count: int
    assembly_success_rate: float
    min_assembly_success_rate: float
    state_match_count: int
    state_match_rate: float
    retry_path_tested: bool
    abort_path_tested: bool
    retry_probe_success: bool
    retry_probe_retry_count: int
    abort_probe_aborted: bool
    abort_probe_abort_count: int
    archive_count: int
    archive_roundtrip_count: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P3AcceptanceReport.episode_count must be positive")
        for name in (
            "assembly_success_count",
            "assembly_failure_count",
            "crash_count",
            "state_match_count",
            "retry_probe_retry_count",
            "abort_probe_abort_count",
            "archive_count",
        ):
            if getattr(self, name) < 0:
                raise SchemaValidationError(f"P3AcceptanceReport.{name} must be non-negative")
        if not 0.0 <= self.assembly_success_rate <= 1.0:
            raise SchemaValidationError("P3AcceptanceReport.assembly_success_rate must be in [0, 1]")
        if not 0.0 <= self.min_assembly_success_rate <= 1.0:
            raise SchemaValidationError("P3AcceptanceReport.min_assembly_success_rate must be in [0, 1]")
        if not 0.0 <= self.state_match_rate <= 1.0:
            raise SchemaValidationError("P3AcceptanceReport.state_match_rate must be in [0, 1]")
        if self.archive_roundtrip_count is not None and self.archive_roundtrip_count < 0:
            raise SchemaValidationError("P3AcceptanceReport.archive_roundtrip_count must be non-negative")


def run_p3_acceptance(
    base_task_spec: TaskSpec,
    *,
    criteria: P3AcceptanceCriteria | None = None,
    archive_path: str | Path | None = None,
) -> P3AcceptanceReport:
    """Run the v0.4 Section 24.4 P3 deterministic assembly acceptance gate."""

    criteria = criteria or P3AcceptanceCriteria()
    loaded_runner_config, distribution_config, policy_config = load_p3_assembly_runner_config(criteria.config_path)
    runner_config = P3AssemblyRunnerConfig(
        episode_count=criteria.episode_count,
        seed=criteria.seed,
        source_hash=criteria.source_hash,
        runner_version=loaded_runner_config.runner_version,
        robot_model_config_path=loaded_runner_config.robot_model_config_path,
        archive_success_only=loaded_runner_config.archive_success_only,
        max_retries_per_step=loaded_runner_config.max_retries_per_step,
    )
    runner = P3AssemblyEvaluationRunner(
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

    state_match_count = sum(
        1
        for archive in archives
        if archive.success and archive.metrics.get("assembly_state_matches_target") == 1.0
    )
    assembly_success_rate = runner_result.metrics.get(
        "assembly_success_rate",
        float(runner_result.assembly_success_count) / float(runner_result.episode_count),
    )
    state_match_rate = float(state_match_count) / max(1.0, float(runner_result.assembly_success_count))
    retry_probe = _run_retry_probe(
        base_task_spec,
        seed=criteria.seed + criteria.retry_probe_seed_offset,
        distribution_config=distribution_config,
        policy_config=policy_config,
        robot_model_config_path=runner_config.robot_model_config_path,
    )
    abort_probe = _run_abort_probe(
        base_task_spec,
        seed=criteria.seed + criteria.abort_probe_seed_offset,
        distribution_config=distribution_config,
        policy_config=policy_config,
        robot_model_config_path=runner_config.robot_model_config_path,
    )
    retry_path_tested = retry_probe.success and retry_probe.retry_count > 0 and retry_probe.abort_count == 0
    abort_path_tested = (not abort_probe.success) and abort_probe.aborted and abort_probe.abort_count > 0
    failure_reasons = _failure_reasons(
        assembly_success_rate=assembly_success_rate,
        min_assembly_success_rate=criteria.min_assembly_success_rate,
        crash_count=runner_result.crash_count,
        assembly_success_count=runner_result.assembly_success_count,
        state_match_count=state_match_count,
        retry_path_tested=retry_path_tested,
        abort_path_tested=abort_path_tested,
        archive_count=len(archives),
        archive_roundtrip_count=archive_roundtrip_count,
        expected_archive_count=len(runner_result.archives),
    )

    return P3AcceptanceReport(
        passed=not failure_reasons,
        episode_count=runner_result.episode_count,
        assembly_success_count=runner_result.assembly_success_count,
        assembly_failure_count=runner_result.assembly_failure_count,
        crash_count=runner_result.crash_count,
        assembly_success_rate=assembly_success_rate,
        min_assembly_success_rate=criteria.min_assembly_success_rate,
        state_match_count=state_match_count,
        state_match_rate=state_match_rate,
        retry_path_tested=retry_path_tested,
        abort_path_tested=abort_path_tested,
        retry_probe_success=retry_probe.success,
        retry_probe_retry_count=retry_probe.retry_count,
        abort_probe_aborted=abort_probe.aborted,
        abort_probe_abort_count=abort_probe.abort_count,
        archive_count=len(runner_result.archives),
        archive_roundtrip_count=archive_roundtrip_count,
        metrics={
            **runner_result.metrics,
            "state_match_count": float(state_match_count),
            "state_match_rate": state_match_rate,
            "retry_probe_retry_count": float(retry_probe.retry_count),
            "abort_probe_abort_count": float(abort_probe.abort_count),
        },
        failure_reasons=failure_reasons,
    )


def _run_retry_probe(
    base_task_spec: TaskSpec,
    *,
    seed: int,
    distribution_config: P2DesignDistributionConfig,
    policy_config: P2DesignPolicyConfig,
    robot_model_config_path: str,
):
    target_graph = _probe_target_graph(
        base_task_spec,
        seed=seed,
        distribution_config=distribution_config,
        policy_config=policy_config,
        robot_model_config_path=robot_model_config_path,
    )
    return AssemblyRunner(config=AssemblyRunnerConfig(max_retries_per_step=1)).run(
        target_graph,
        SimplifiedAssemblyExecutor(
            target_graph=target_graph,
            config=SimplifiedAssemblyExecutorConfig(
                failure_mode="fail_matching_steps",
                fail_once_step_types=("dock",),
            ),
        ),
    )


def _run_abort_probe(
    base_task_spec: TaskSpec,
    *,
    seed: int,
    distribution_config: P2DesignDistributionConfig,
    policy_config: P2DesignPolicyConfig,
    robot_model_config_path: str,
):
    target_graph = _probe_target_graph(
        base_task_spec,
        seed=seed,
        distribution_config=distribution_config,
        policy_config=policy_config,
        robot_model_config_path=robot_model_config_path,
    )
    return AssemblyRunner(config=AssemblyRunnerConfig(max_retries_per_step=1)).run(
        target_graph,
        SimplifiedAssemblyExecutor(
            target_graph=target_graph,
            config=SimplifiedAssemblyExecutorConfig(
                failure_mode="fail_matching_steps",
                fail_step_types=("dock",),
            ),
        ),
    )


def _probe_target_graph(
    base_task_spec: TaskSpec,
    *,
    seed: int,
    distribution_config: P2DesignDistributionConfig,
    policy_config: P2DesignPolicyConfig,
    robot_model_config_path: str,
) -> MorphologyGraph:
    sample = P2GraspCarryDesignDistribution(base_task_spec, distribution_config).sample(seed=seed, sample_index=0)
    physical_model = build_physical_model_from_config(robot_model_config_path)
    irg = IRGBuilder().build(sample.task_spec)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    selection = P2DesignPolicy(config=policy_config).evaluate_candidates(
        DesignPolicyContext(
            task_spec=sample.task_spec,
            irg=irg,
            physical_model=physical_model,
            interaction_envelope=envelope,
        )
    )
    return selection.selected_candidate.design_output.target_morphology


def _failure_reasons(
    *,
    assembly_success_rate: float,
    min_assembly_success_rate: float,
    crash_count: int,
    assembly_success_count: int,
    state_match_count: int,
    retry_path_tested: bool,
    abort_path_tested: bool,
    archive_count: int,
    archive_roundtrip_count: int | None,
    expected_archive_count: int,
) -> list[str]:
    reasons: list[str] = []
    if assembly_success_rate < min_assembly_success_rate:
        reasons.append(
            f"assembly_success_rate {assembly_success_rate:.3f} < required {min_assembly_success_rate:.3f}"
        )
    if crash_count > 0:
        reasons.append(f"assembly crash_count {crash_count} > 0")
    if assembly_success_count <= 0:
        reasons.append("no successful assembly episodes available for state consistency check")
    elif state_match_count != assembly_success_count:
        reasons.append(
            f"state_match_count {state_match_count} != assembly_success_count {assembly_success_count}"
        )
    if not retry_path_tested:
        reasons.append("retry path was not tested successfully")
    if not abort_path_tested:
        reasons.append("abort path was not tested successfully")
    if archive_count <= 0:
        reasons.append("no P3 archives were produced")
    if archive_roundtrip_count is not None and archive_roundtrip_count != expected_archive_count:
        reasons.append(
            f"archive_roundtrip_count {archive_roundtrip_count} != expected {expected_archive_count}"
        )
    return reasons
