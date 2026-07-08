from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amsrr.logging import EpisodeArchive, read_episode_archives_jsonl
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P4_0_SIMPLIFIED_BACKEND_NOTE,
    P4_0FullPipelineRunner,
    P4_0FullPipelineRunnerConfig,
    load_p4_0_full_pipeline_runner_config,
)


@dataclass
class P4_0AcceptanceCriteria(SchemaBase):
    episode_count: int = 1000
    seed: int = 0
    config_path: str = "configs/training/p4_0_grasp_carry.yaml"
    source_hash: str = "p4_0_acceptance"

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P4_0AcceptanceCriteria.episode_count must be positive")
        require_non_empty(self.config_path, "P4_0AcceptanceCriteria.config_path")
        require_non_empty(self.source_hash, "P4_0AcceptanceCriteria.source_hash")


@dataclass
class P4_0AcceptanceReport(SchemaBase):
    passed: bool
    episode_count: int
    success_count: int
    failure_count: int
    crash_count: int
    success_rate: float
    object_drop_rate: float
    collision_rate: float
    qp_infeasible_rate: float
    archive_count: int
    archive_roundtrip_count: int | None
    p2_selected_design_used: bool
    p3_assembly_result_used: bool
    fixed_simple_design_policy_absent: bool
    contact_candidates_generated: bool
    trajectory_generated: bool
    policy_commands_generated: bool
    controller_commands_generated: bool
    archive_complete: bool
    simplified_metrics_recorded: bool
    no_mislabeling_passed: bool
    report_declares_simplified_backend: bool
    backend_note: str
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P4_0AcceptanceReport.episode_count must be positive")
        for name in ("success_count", "failure_count", "crash_count", "archive_count"):
            if getattr(self, name) < 0:
                raise SchemaValidationError(f"P4_0AcceptanceReport.{name} must be non-negative")
        if self.archive_roundtrip_count is not None and self.archive_roundtrip_count < 0:
            raise SchemaValidationError("P4_0AcceptanceReport.archive_roundtrip_count must be non-negative")
        for name in ("success_rate", "object_drop_rate", "collision_rate", "qp_infeasible_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise SchemaValidationError(f"P4_0AcceptanceReport.{name} must be in [0, 1]")
        require_non_empty(self.backend_note, "P4_0AcceptanceReport.backend_note")


def run_p4_0_acceptance(
    base_task_spec: TaskSpec,
    *,
    criteria: P4_0AcceptanceCriteria | None = None,
    archive_path: str | Path | None = None,
) -> P4_0AcceptanceReport:
    """Run the v0.4 Section 24.5.1 P4.0 simplified full-pipeline gate."""

    criteria = criteria or P4_0AcceptanceCriteria()
    loaded_runner_config, distribution_config, policy_config, env_config = load_p4_0_full_pipeline_runner_config(
        criteria.config_path
    )
    runner_config = P4_0FullPipelineRunnerConfig(
        episode_count=criteria.episode_count,
        seed=criteria.seed,
        source_hash=criteria.source_hash,
        runner_version=loaded_runner_config.runner_version,
        robot_model_config_path=loaded_runner_config.robot_model_config_path,
        archive_success_only=loaded_runner_config.archive_success_only,
        max_retries_per_step=loaded_runner_config.max_retries_per_step,
        simulator_version=loaded_runner_config.simulator_version,
    )
    runner = P4_0FullPipelineRunner(
        base_task_spec,
        runner_config=runner_config,
        distribution_config=distribution_config,
        policy_config=policy_config,
        env_config=env_config,
    )
    runner_result = runner.run(archive_path=archive_path)

    archives = runner_result.archives
    archive_roundtrip_count = None
    if archive_path is not None:
        archives = read_episode_archives_jsonl(archive_path)
        archive_roundtrip_count = len(archives)

    p2_selected_design_used = _all_archives(archives, _p2_selected_design_used)
    p3_assembly_result_used = _all_archives(archives, _p3_assembly_result_used)
    fixed_simple_design_policy_absent = _all_archives(
        archives,
        lambda archive: archive.metrics.get("fixed_simple_design_policy_used") == 0.0,
    )
    contact_candidates_generated = _all_archives(
        archives,
        lambda archive: archive.metrics.get("contact_candidate_count", 0.0) > 0.0,
    )
    trajectory_generated = _all_archives(archives, lambda archive: bool(archive.trajectory_records))
    policy_commands_generated = _all_archives(archives, lambda archive: bool(archive.policy_commands))
    controller_commands_generated = _all_archives(archives, lambda archive: bool(archive.controller_commands))
    archive_complete = _all_archives(archives, _archive_complete)
    simplified_metrics_recorded = _metrics_recorded(runner_result.metrics)
    no_mislabeling_passed = _all_archives(archives, _no_mislabeling_passed) and (
        runner_result.metrics.get("simplified_backend") == 1.0
        and runner_result.metrics.get("isaac_backed") == 0.0
        and runner_result.metrics.get("p4_full_completion") == 0.0
    )
    report_declares_simplified_backend = "not Isaac-backed physical success rates" in P4_0_SIMPLIFIED_BACKEND_NOTE
    failure_reasons = _failure_reasons(
        runner_result=runner_result,
        archives=archives,
        archive_roundtrip_count=archive_roundtrip_count,
        p2_selected_design_used=p2_selected_design_used,
        p3_assembly_result_used=p3_assembly_result_used,
        fixed_simple_design_policy_absent=fixed_simple_design_policy_absent,
        contact_candidates_generated=contact_candidates_generated,
        trajectory_generated=trajectory_generated,
        policy_commands_generated=policy_commands_generated,
        controller_commands_generated=controller_commands_generated,
        archive_complete=archive_complete,
        simplified_metrics_recorded=simplified_metrics_recorded,
        no_mislabeling_passed=no_mislabeling_passed,
        report_declares_simplified_backend=report_declares_simplified_backend,
    )
    return P4_0AcceptanceReport(
        passed=not failure_reasons,
        episode_count=runner_result.episode_count,
        success_count=runner_result.success_count,
        failure_count=runner_result.failure_count,
        crash_count=runner_result.crash_count,
        success_rate=runner_result.metrics.get("success_rate", 0.0),
        object_drop_rate=runner_result.metrics.get("object_drop_rate", 0.0),
        collision_rate=runner_result.metrics.get("collision_rate", 0.0),
        qp_infeasible_rate=runner_result.metrics.get("qp_infeasible_rate", 0.0),
        archive_count=len(archives),
        archive_roundtrip_count=archive_roundtrip_count,
        p2_selected_design_used=p2_selected_design_used,
        p3_assembly_result_used=p3_assembly_result_used,
        fixed_simple_design_policy_absent=fixed_simple_design_policy_absent,
        contact_candidates_generated=contact_candidates_generated,
        trajectory_generated=trajectory_generated,
        policy_commands_generated=policy_commands_generated,
        controller_commands_generated=controller_commands_generated,
        archive_complete=archive_complete,
        simplified_metrics_recorded=simplified_metrics_recorded,
        no_mislabeling_passed=no_mislabeling_passed,
        report_declares_simplified_backend=report_declares_simplified_backend,
        backend_note=P4_0_SIMPLIFIED_BACKEND_NOTE,
        metrics=runner_result.metrics,
        failure_reasons=failure_reasons,
    )


def _all_archives(archives: list[EpisodeArchive], predicate) -> bool:
    return bool(archives) and all(predicate(archive) for archive in archives)


def _p2_selected_design_used(archive: EpisodeArchive) -> bool:
    return (
        archive.design_output is not None
        and archive.design_output.design_scores.get("p2_design_policy_selected") == 1.0
        and archive.metrics.get("p2_selected_design_used") == 1.0
    )


def _p3_assembly_result_used(archive: EpisodeArchive) -> bool:
    return (
        archive.assembly_plan is not None
        and archive.metrics.get("p3_assembly_result_used") == 1.0
        and archive.metrics.get("assembly_success") == 1.0
        and archive.metrics.get("assembly_state_matches_target") == 1.0
    )


def _archive_complete(archive: EpisodeArchive) -> bool:
    return (
        archive.design_output is not None
        and archive.feasibility_result is not None
        and archive.assembly_plan is not None
        and bool(archive.trajectory_records)
        and bool(archive.policy_commands)
        and bool(archive.controller_commands)
        and bool(archive.rewards)
        and bool(archive.runtime_observations)
        and archive.metrics.get("contact_candidate_count", 0.0) > 0.0
        and archive.metrics.get("assignment_feasibility_cache_count", 0.0) > 0.0
    )


def _metrics_recorded(metrics: dict[str, float]) -> bool:
    return all(
        key in metrics
        for key in (
            "success_rate",
            "object_drop_rate",
            "collision_rate",
            "qp_infeasible_rate",
        )
    )


def _no_mislabeling_passed(archive: EpisodeArchive) -> bool:
    return (
        archive.metrics.get("simplified_backend") == 1.0
        and archive.metrics.get("isaac_backed") == 0.0
        and archive.metrics.get("p4_full_completion") == 0.0
        and archive.rollout_artifacts.get("phase") == "P4.0"
        and archive.rollout_artifacts.get("backend") == "simplified"
        and archive.rollout_artifacts.get("is_p4_full_completion") is False
        and archive.rollout_artifacts.get("isaac_backed") is False
        and archive.rollout_artifacts.get("physical_success_claim") is False
    )


def _failure_reasons(
    *,
    runner_result,
    archives: list[EpisodeArchive],
    archive_roundtrip_count: int | None,
    p2_selected_design_used: bool,
    p3_assembly_result_used: bool,
    fixed_simple_design_policy_absent: bool,
    contact_candidates_generated: bool,
    trajectory_generated: bool,
    policy_commands_generated: bool,
    controller_commands_generated: bool,
    archive_complete: bool,
    simplified_metrics_recorded: bool,
    no_mislabeling_passed: bool,
    report_declares_simplified_backend: bool,
) -> list[str]:
    reasons: list[str] = []
    if runner_result.crash_count:
        reasons.append("P4.0 runner crashed")
    if not archives:
        reasons.append("P4.0 runner produced no archives")
    if archive_roundtrip_count is not None and archive_roundtrip_count != len(runner_result.archives):
        reasons.append("P4.0 archive roundtrip count mismatch")
    if not p2_selected_design_used:
        reasons.append("P4.0 archives do not consistently use P2 selected DesignOutput")
    if not p3_assembly_result_used:
        reasons.append("P4.0 archives do not consistently use P3 assembly result")
    if not fixed_simple_design_policy_absent:
        reasons.append("P4.0 archives indicate FixedSimpleDesignPolicy fixed path usage")
    if not contact_candidates_generated:
        reasons.append("P4.0 archives are missing contact candidates")
    if not trajectory_generated:
        reasons.append("P4.0 archives are missing pi_H trajectory records")
    if not policy_commands_generated:
        reasons.append("P4.0 archives are missing pi_L PolicyCommand records")
    if not controller_commands_generated:
        reasons.append("P4.0 archives are missing ControllerCommand records")
    if not archive_complete:
        reasons.append("P4.0 archives are incomplete")
    if not simplified_metrics_recorded:
        reasons.append("P4.0 simplified metrics are not recorded")
    if not no_mislabeling_passed:
        reasons.append("P4.0 no-mislabeling checks failed")
    if not report_declares_simplified_backend:
        reasons.append("P4.0 report does not declare simplified backend metric limits")
    return reasons
