from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from amsrr.assembly import (
    AssemblyRunner,
    AssemblyRunnerConfig,
    SimplifiedAssemblyExecutor,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.morphology.grasp_carry_designs import GraspCarryMorphologyVariant
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignPolicyConfig, P2DesignSelection
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.p2_design_distribution import (
    P2DesignDistributionConfig,
    P2DesignTaskSample,
    P2GraspCarryDesignDistribution,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


P3_ASSEMBLY_RUNNER_VERSION = "p3_assembly_eval_runner_v1"


@dataclass
class P3AssemblyRunnerConfig(SchemaBase):
    episode_count: int = 1000
    seed: int = 0
    source_hash: str = "unknown"
    runner_version: str = P3_ASSEMBLY_RUNNER_VERSION
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    archive_success_only: bool = False
    max_retries_per_step: int = 1

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P3AssemblyRunnerConfig.episode_count must be positive")
        if self.max_retries_per_step < 0:
            raise SchemaValidationError("P3AssemblyRunnerConfig.max_retries_per_step must be non-negative")
        require_non_empty(self.source_hash, "P3AssemblyRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P3AssemblyRunnerConfig.runner_version")
        require_non_empty(self.robot_model_config_path, "P3AssemblyRunnerConfig.robot_model_config_path")


@dataclass
class P3AssemblyRunnerResult(SchemaBase):
    episode_count: int
    assembly_success_count: int
    assembly_failure_count: int
    crash_count: int
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P3AssemblyRunnerResult.episode_count must be positive")
        for name in ("assembly_success_count", "assembly_failure_count", "crash_count"):
            if getattr(self, name) < 0:
                raise SchemaValidationError(f"P3AssemblyRunnerResult.{name} must be non-negative")


def load_p3_assembly_runner_config(
    path: str | Path,
) -> tuple[P3AssemblyRunnerConfig, P2DesignDistributionConfig, P2DesignPolicyConfig]:
    data = load_config(path)
    return (
        P3AssemblyRunnerConfig.from_dict(data.get("runner", {})),
        P2DesignDistributionConfig.from_dict(data.get("distribution", {})),
        _p2_design_policy_config_from_dict(data.get("policy", {})),
    )


class P3AssemblyEvaluationRunner:
    """Evaluate deterministic P3 assembly integration over P2-selected morphologies."""

    def __init__(
        self,
        base_task_spec: TaskSpec,
        *,
        runner_config: P3AssemblyRunnerConfig | None = None,
        distribution_config: P2DesignDistributionConfig | None = None,
        policy_config: P2DesignPolicyConfig | None = None,
        design_policy: P2DesignPolicy | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> None:
        self.base_task_spec = base_task_spec
        self.runner_config = runner_config or P3AssemblyRunnerConfig()
        self.distribution_config = distribution_config or P2DesignDistributionConfig()
        self.policy_config = policy_config or (
            design_policy.config if design_policy is not None else P2DesignPolicyConfig()
        )
        self.physical_model = physical_model or build_physical_model_from_config(
            self.runner_config.robot_model_config_path
        )
        self.distribution = P2GraspCarryDesignDistribution(base_task_spec, self.distribution_config)
        self.irg_builder = IRGBuilder()
        self.envelope_extractor = InteractionEnvelopeExtractor()
        self.design_policy = design_policy or P2DesignPolicy(config=self.policy_config)

    def run(self, *, archive_path: str | Path | None = None) -> P3AssemblyRunnerResult:
        assembly_success_count = 0
        assembly_failure_count = 0
        crash_count = 0
        retry_count_sum = 0.0
        abort_count_sum = 0.0
        plan_step_count_sum = 0.0
        executed_step_count_sum = 0.0
        state_match_sum = 0.0
        archives: list[EpisodeArchive] = []

        for index in range(self.runner_config.episode_count):
            seed = self.runner_config.seed + index
            sample = self._sample(seed=seed, index=index)
            try:
                archive = self._run_one(sample, seed=seed, episode_id=f"p3_assembly_{index:04d}")
            except Exception:
                crash_count += 1
                continue

            if archive.success:
                assembly_success_count += 1
            else:
                assembly_failure_count += 1
            retry_count_sum += archive.metrics.get("assembly_retry_count", 0.0)
            abort_count_sum += archive.metrics.get("assembly_abort_count", 0.0)
            plan_step_count_sum += archive.metrics.get("assembly_plan_step_count", 0.0)
            executed_step_count_sum += archive.metrics.get("assembly_executed_step_count", 0.0)
            state_match_sum += archive.metrics.get("assembly_state_matches_target", 0.0)
            if archive.success or not self.runner_config.archive_success_only:
                archives.append(archive)

        evaluated_count = assembly_success_count + assembly_failure_count
        result = P3AssemblyRunnerResult(
            episode_count=self.runner_config.episode_count,
            assembly_success_count=assembly_success_count,
            assembly_failure_count=assembly_failure_count,
            crash_count=crash_count,
            archives=archives,
            metrics={
                "assembly_success_rate": float(assembly_success_count) / float(self.runner_config.episode_count),
                "assembly_failure_rate": float(assembly_failure_count) / float(self.runner_config.episode_count),
                "crash_rate": float(crash_count) / float(self.runner_config.episode_count),
                "archive_count": float(len(archives)),
                "mean_retry_count": retry_count_sum / max(1.0, float(evaluated_count)),
                "mean_abort_count": abort_count_sum / max(1.0, float(evaluated_count)),
                "mean_plan_step_count": plan_step_count_sum / max(1.0, float(evaluated_count)),
                "mean_executed_step_count": executed_step_count_sum / max(1.0, float(evaluated_count)),
                "state_match_rate": state_match_sum / max(1.0, float(evaluated_count)),
            },
        )
        if archive_path is not None:
            write_episode_archives_jsonl(archive_path, archives)
        return result

    def _sample(self, *, seed: int, index: int) -> P2DesignTaskSample:
        sample = self.distribution.sample(seed=seed, sample_index=index)
        task_data = sample.task_spec.to_dict()
        task_data["task_id"] = f"{self.base_task_spec.task_id}_p3_{index:04d}"
        metadata = dict(task_data.get("metadata", {}) or {})
        metadata["assembly_evaluation_phase"] = "P3"
        task_data["metadata"] = metadata
        return P2DesignTaskSample(
            task_spec=TaskSpec.from_dict(task_data),
            seed=sample.seed,
            sample_index=sample.sample_index,
            sampled_values=sample.sampled_values,
        )

    def _run_one(
        self,
        sample: P2DesignTaskSample,
        *,
        seed: int,
        episode_id: str,
    ) -> EpisodeArchive:
        builder_result = self.irg_builder.build_with_scene_graph(sample.task_spec)
        irg = builder_result.irg
        envelope = self.envelope_extractor.extract(irg)
        context = DesignPolicyContext(
            task_spec=sample.task_spec,
            irg=irg,
            physical_model=self.physical_model,
            interaction_envelope=envelope,
        )
        selection = self.design_policy.evaluate_candidates(context)
        selected = selection.selected_candidate
        target_graph = selected.design_output.target_morphology
        assembly_report = AssemblyRunner(
            config=AssemblyRunnerConfig(max_retries_per_step=self.runner_config.max_retries_per_step)
        ).run(
            target_graph,
            SimplifiedAssemblyExecutor(target_graph=target_graph),
        )
        success = selected.feasibility_result.feasible and assembly_report.success
        failure_reason = None
        if not selected.feasibility_result.feasible:
            failure_reason = selected.rejection_reason
        elif not assembly_report.success:
            failure_reason = assembly_report.failure_reason
        return _episode_archive(
            sample,
            irg=irg,
            envelope=envelope,
            selection=selection,
            assembly_report=assembly_report,
            physical_model=self.physical_model,
            episode_id=episode_id,
            seed=seed,
            success=success,
            failure_reason=failure_reason,
            config_hash=self._config_hash(),
            runner_config=self.runner_config,
        )

    def _config_hash(self) -> str:
        return stable_hash(
            {
                "runner": self.runner_config,
                "distribution": self.distribution_config,
                "policy": self.policy_config,
            }
        )


def _episode_archive(
    sample: P2DesignTaskSample,
    *,
    irg: InteractionRequirementGraph,
    envelope: InteractionEnvelope,
    selection: P2DesignSelection,
    assembly_report,
    physical_model: PhysicalModel,
    episode_id: str,
    seed: int,
    success: bool,
    failure_reason: str | None,
    config_hash: str,
    runner_config: P3AssemblyRunnerConfig,
) -> EpisodeArchive:
    selected = selection.selected_candidate
    feasibility = selected.feasibility_result
    metrics = _assembly_metrics(selection, assembly_report)
    return EpisodeArchive(
        episode_id=episode_id,
        task_spec=sample.task_spec,
        task_hash=sample.task_spec.stable_hash(),
        geometry_hashes={
            geometry.geometry_id: stable_hash(geometry)
            for geometry in sample.task_spec.scene.geometry_library
        },
        robot_model_hash=physical_model.stable_hash(),
        config_hash=config_hash,
        irg=irg,
        interaction_envelope=envelope,
        design_output=selected.design_output,
        feasibility_result=feasibility,
        assembly_plan=assembly_report.plan.to_dict(),
        trajectory_records=[],
        policy_commands=[],
        controller_commands=[],
        rewards=[],
        metrics=metrics,
        success=success,
        failure_reason=failure_reason,
        reproducibility={
            "source_hash": runner_config.source_hash,
            "random_seed": seed,
            "runner_version": runner_config.runner_version,
            "policy_version": selection.policy_version,
            "robot_model_config_path": runner_config.robot_model_config_path,
            "urdf_hash": str(physical_model.metadata.get("urdf_hash", "")),
            "thrust_model_hash": str(physical_model.metadata.get("thrust_model_hash", "")),
        },
    )


def _assembly_metrics(selection: P2DesignSelection, assembly_report) -> dict[str, float]:
    selected = selection.selected_candidate
    metrics = {
        "success": 1.0 if assembly_report.success else 0.0,
        "crashed": 0.0,
        "selected_feasible": 1.0 if selected.feasibility_result.feasible else 0.0,
        "selected_candidate_id": float(selected.candidate_id),
        "selected_soft_score": float(selected.soft_score),
        "assembly_success": 1.0 if assembly_report.success else 0.0,
        "assembly_retry_count": float(assembly_report.retry_count),
        "assembly_abort_count": float(assembly_report.abort_count),
        "assembly_aborted": 1.0 if assembly_report.aborted else 0.0,
        "assembly_plan_step_count": float(len(assembly_report.plan.steps)),
        "assembly_executed_step_count": float(len(assembly_report.step_results)),
        "assembly_completed_step_count": float(assembly_report.completed_step_count),
        "assembly_attached_edge_count": float(assembly_report.attached_edge_count),
        "assembly_target_edge_count": float(assembly_report.target_edge_count),
        "assembly_state_matches_target": 1.0 if assembly_report.state_matches_target else 0.0,
    }
    for key, value in assembly_report.metrics.items():
        metrics[f"assembly_{key}"] = float(value)
    return metrics


def _p2_design_policy_config_from_dict(data: dict[str, Any]) -> P2DesignPolicyConfig:
    if not isinstance(data, dict):
        raise SchemaValidationError("P2DesignPolicyConfig expects a dict")
    raw = dict(data)
    policy_fields = {item.name for item in fields(P2DesignPolicyConfig)}
    unknown = set(raw) - policy_fields
    if unknown:
        raise SchemaValidationError(f"P2DesignPolicyConfig got unknown fields: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    if "variants" in raw:
        kwargs["variants"] = tuple(GraspCarryMorphologyVariant(item) for item in raw.pop("variants"))
    for key, value in raw.items():
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        kwargs[key] = value
    return P2DesignPolicyConfig(**kwargs)
