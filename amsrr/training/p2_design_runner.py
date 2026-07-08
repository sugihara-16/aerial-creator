from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

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


P2_DESIGN_RUNNER_VERSION = "p2_design_eval_runner_v1"
P2_DESIGN_ARCHIVE_METRIC_KEYS = (
    "required_slot_coverage_ratio",
    "required_slot_anchor_capability_coverage_ratio",
    "closed_loop_rejected",
    "port_conflict_count",
    "thrust_margin_ratio",
    "payload_margin_ratio",
    "coarse_reachability_ratio",
)


@dataclass
class P2DesignRunnerConfig(SchemaBase):
    episode_count: int = 1000
    seed: int = 0
    source_hash: str = "unknown"
    runner_version: str = P2_DESIGN_RUNNER_VERSION
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    archive_success_only: bool = False

    def validate(self) -> None:
        if self.episode_count <= 0:
            raise SchemaValidationError("P2DesignRunnerConfig.episode_count must be positive")
        require_non_empty(self.source_hash, "P2DesignRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P2DesignRunnerConfig.runner_version")
        require_non_empty(self.robot_model_config_path, "P2DesignRunnerConfig.robot_model_config_path")


@dataclass
class P2DesignRunnerResult(SchemaBase):
    episode_count: int
    valid_design_count: int
    invalid_design_count: int
    crash_count: int
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_p2_design_runner_config(
    path: str | Path,
) -> tuple[P2DesignRunnerConfig, P2DesignDistributionConfig, P2DesignPolicyConfig]:
    data = load_config(path)
    return (
        P2DesignRunnerConfig.from_dict(data.get("runner", {})),
        P2DesignDistributionConfig.from_dict(data.get("distribution", {})),
        _p2_design_policy_config_from_dict(data.get("policy", {})),
    )


class P2DesignEvaluationRunner:
    """Evaluate P2 grasp/carry design candidates over a randomized TaskSpec distribution."""

    def __init__(
        self,
        base_task_spec: TaskSpec,
        *,
        runner_config: P2DesignRunnerConfig | None = None,
        distribution_config: P2DesignDistributionConfig | None = None,
        policy_config: P2DesignPolicyConfig | None = None,
        design_policy: P2DesignPolicy | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> None:
        self.base_task_spec = base_task_spec
        self.runner_config = runner_config or P2DesignRunnerConfig()
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

    def run(self, *, archive_path: str | Path | None = None) -> P2DesignRunnerResult:
        valid_design_count = 0
        invalid_design_count = 0
        crash_count = 0
        selected_soft_score_sum = 0.0
        accepted_candidate_count_sum = 0.0
        required_slot_coverage_sum = 0.0
        archives: list[EpisodeArchive] = []

        for index in range(self.runner_config.episode_count):
            seed = self.runner_config.seed + index
            sample = self.distribution.sample(seed=seed, sample_index=index)
            try:
                archive = self._run_one(sample, seed=seed, episode_id=f"p2_design_{index:04d}")
            except Exception:
                crash_count += 1
                continue

            if archive.success:
                valid_design_count += 1
            else:
                invalid_design_count += 1
            selected_soft_score_sum += archive.metrics.get("selected_soft_score", 0.0)
            accepted_candidate_count_sum += archive.metrics.get("accepted_candidate_count", 0.0)
            required_slot_coverage_sum += archive.metrics.get("required_slot_coverage_ratio", 0.0)
            if archive.success or not self.runner_config.archive_success_only:
                archives.append(archive)

        evaluated_count = valid_design_count + invalid_design_count
        result = P2DesignRunnerResult(
            episode_count=self.runner_config.episode_count,
            valid_design_count=valid_design_count,
            invalid_design_count=invalid_design_count,
            crash_count=crash_count,
            archives=archives,
            metrics={
                "valid_design_rate": float(valid_design_count) / float(self.runner_config.episode_count),
                "invalid_design_rate": float(invalid_design_count) / float(self.runner_config.episode_count),
                "crash_rate": float(crash_count) / float(self.runner_config.episode_count),
                "archive_count": float(len(archives)),
                "mean_selected_soft_score": selected_soft_score_sum / max(1.0, float(evaluated_count)),
                "mean_accepted_candidate_count": accepted_candidate_count_sum / max(1.0, float(evaluated_count)),
                "mean_required_slot_coverage": required_slot_coverage_sum / max(1.0, float(evaluated_count)),
            },
        )
        if archive_path is not None:
            write_episode_archives_jsonl(archive_path, archives)
        return result

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
        return _episode_archive(
            sample,
            irg=irg,
            envelope=envelope,
            selection=selection,
            physical_model=self.physical_model,
            episode_id=episode_id,
            seed=seed,
            success=selected.feasibility_result.feasible,
            failure_reason=None if selected.feasibility_result.feasible else selected.rejection_reason,
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
    physical_model: PhysicalModel,
    episode_id: str,
    seed: int,
    success: bool,
    failure_reason: str | None,
    config_hash: str,
    runner_config: P2DesignRunnerConfig,
) -> EpisodeArchive:
    selected = selection.selected_candidate
    feasibility = selected.feasibility_result
    metrics = _selection_metrics(selection)
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
        assembly_plan=None,
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


def _selection_metrics(selection: P2DesignSelection) -> dict[str, float]:
    selected = selection.selected_candidate
    feasibility = selected.feasibility_result
    metrics = {
        "success": 1.0 if selected.accepted else 0.0,
        "crashed": 0.0,
        "candidate_count": float(len(selection.candidates)),
        "accepted_candidate_count": float(len(selection.accepted_candidates)),
        "rejected_candidate_count": float(len(selection.rejected_candidates)),
        "selected_candidate_id": float(selected.candidate_id),
        "selected_soft_score": float(selected.soft_score),
        "selected_accepted": 1.0 if selected.accepted else 0.0,
        "selected_feasible": 1.0 if feasibility.feasible else 0.0,
    }
    for key in P2_DESIGN_ARCHIVE_METRIC_KEYS:
        metrics[key] = float(feasibility.margins.get(key, 0.0))
    for key, value in feasibility.proxy_scores.items():
        if key.startswith("L_"):
            metrics[f"label_{key}"] = float(value)
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
