from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.assembly import AssemblyRunner, AssemblyRunnerConfig, SimplifiedAssemblyExecutor
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignSelection
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import PolicyCommand
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend
from amsrr.simulation.p4_1_backend_smoke import P4_1BackendSmokeResult
from amsrr.simulation.p4_1_isaac_env import (
    P4_1IsaacBackendEnv,
    load_p4_1_full_scene_backend_config,
)
from amsrr.training.p2_design_distribution import P2DesignTaskSample, P2GraspCarryDesignDistribution
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.training.p3_assembly_runner import load_p3_assembly_runner_config
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


P4_1_BACKEND_SMOKE_RUNNER_VERSION = "p4_1_backend_smoke_runner_v1"


@dataclass
class P4_1BackendSmokeRunnerConfig(SchemaBase):
    seed: int = 0
    sample_index: int = 0
    source_hash: str = "p4_1_backend_smoke"
    runner_version: str = P4_1_BACKEND_SMOKE_RUNNER_VERSION
    dry_run: bool = True
    archive_path: str | None = "artifacts/p4_1/p4_1_backend_smoke.jsonl"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    p3_config_path: str = "configs/training/p3_assembly_grasp_carry.yaml"
    module_spacing_m: float = 0.45

    def validate(self) -> None:
        require_non_empty(self.source_hash, "P4_1BackendSmokeRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P4_1BackendSmokeRunnerConfig.runner_version")
        require_non_empty(self.robot_model_config_path, "P4_1BackendSmokeRunnerConfig.robot_model_config_path")
        require_non_empty(self.p3_config_path, "P4_1BackendSmokeRunnerConfig.p3_config_path")
        if self.sample_index < 0:
            raise SchemaValidationError("P4_1BackendSmokeRunnerConfig.sample_index must be non-negative")
        if self.module_spacing_m <= 0.0:
            raise SchemaValidationError("P4_1BackendSmokeRunnerConfig.module_spacing_m must be positive")


@dataclass
class P4_1BackendSmokeRunnerResult(SchemaBase):
    dry_run: bool
    smoke_result: P4_1BackendSmokeResult
    acceptance_report: Any | None = None
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class P4_1P2P3Case:
    sample: P2DesignTaskSample
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    selection: P2DesignSelection
    assembly_report: object
    assembled_morphology: MorphologyGraph
    physical_model: PhysicalModel

    @property
    def module_count(self) -> int:
        return len(self.assembled_morphology.modules)


def load_p4_1_backend_smoke_runner_config(
    path: str | Path,
) -> tuple[P4_1BackendSmokeRunnerConfig, object]:
    data = load_config(path)
    _, env_config = load_p4_1_full_scene_backend_config(path)
    return P4_1BackendSmokeRunnerConfig.from_dict(data.get("runner", {})), env_config


class P4_1BackendSmokeRunner:
    """Run the P4.1 full-scene backend smoke over a P2/P3 selected morphology."""

    def __init__(
        self,
        *,
        runner_config: P4_1BackendSmokeRunnerConfig | None = None,
        env_config=None,
        env: P4_1IsaacBackendEnv | None = None,
        base_task_spec: TaskSpec | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> None:
        self.runner_config = runner_config or P4_1BackendSmokeRunnerConfig()
        self.env_config = env_config or load_p4_1_full_scene_backend_config(
            "configs/training/p4_1_backend_smoke.yaml"
        )[1]
        self.env = env or P4_1IsaacBackendEnv(
            config=self.env_config,
            backend=IsaacLabBackend(),
        )
        self.base_task_spec = base_task_spec or default_grasp_carry_task_spec()
        self.physical_model = physical_model

    def run(self, *, archive_path: str | Path | None = None) -> P4_1BackendSmokeRunnerResult:
        from amsrr.acceptance.p4_1_acceptance import run_p4_1_acceptance

        case = self.build_p2_p3_case()
        smoke_result = self.env.run_smoke(
            dry_run=self.runner_config.dry_run,
            module_count=case.module_count,
            module_spacing_m=self.runner_config.module_spacing_m,
            uses_p2_selected_design=True,
            uses_p3_assembled_morphology=bool(case.assembly_report.success),
        )
        archives: list[EpisodeArchive] = []
        if smoke_result.attempted and not smoke_result.skipped:
            archives.append(self.build_archive(case, smoke_result))

        output_path = archive_path
        if output_path is None and self.runner_config.archive_path is not None:
            output_path = self.runner_config.archive_path
        if output_path is not None and archives:
            write_episode_archives_jsonl(output_path, archives)
        acceptance_report = run_p4_1_acceptance(archives, smoke_results=[smoke_result])
        return P4_1BackendSmokeRunnerResult(
            dry_run=self.runner_config.dry_run,
            smoke_result=smoke_result,
            acceptance_report=acceptance_report,
            archives=archives,
            metrics={
                **acceptance_report.metrics,
                "dry_run": 1.0 if self.runner_config.dry_run else 0.0,
                "archive_count": float(len(archives)),
                "p2_selected_design_used": 1.0,
                "p3_assembly_result_used": 1.0 if case.assembly_report.success else 0.0,
                "p4_1_smoke_passed": 1.0 if smoke_result.passed else 0.0,
                "p4_1_module_count": float(case.module_count),
                "fixed_two_module_only": 1.0 if case.module_count == 2 else 0.0,
                "p4_full_completion": 0.0,
            },
        )

    def build_p2_p3_case(self) -> P4_1P2P3Case:
        p3_runner_config, distribution_config, policy_config = load_p3_assembly_runner_config(
            self.runner_config.p3_config_path
        )
        physical_model = self._physical_model()
        sample = P2GraspCarryDesignDistribution(self.base_task_spec, distribution_config).sample(
            seed=self.runner_config.seed,
            sample_index=self.runner_config.sample_index,
        )
        sample = _p4_1_sample(sample, self.runner_config.sample_index)
        builder_result = IRGBuilder().build_with_scene_graph(sample.task_spec)
        irg = builder_result.irg
        envelope = InteractionEnvelopeExtractor().extract(irg)
        context = DesignPolicyContext(
            task_spec=sample.task_spec,
            irg=irg,
            physical_model=physical_model,
            interaction_envelope=envelope,
        )
        selection = P2DesignPolicy(config=policy_config).evaluate_candidates(context)
        selected = selection.selected_candidate
        assembly_report = AssemblyRunner(
            config=AssemblyRunnerConfig(max_retries_per_step=p3_runner_config.max_retries_per_step)
        ).run(
            selected.design_output.target_morphology,
            SimplifiedAssemblyExecutor(target_graph=selected.design_output.target_morphology),
        )
        return P4_1P2P3Case(
            sample=sample,
            irg=irg,
            interaction_envelope=envelope,
            selection=selection,
            assembly_report=assembly_report,
            assembled_morphology=assembly_report.final_state.physical_graph,
            physical_model=physical_model,
        )

    def build_archive(self, case: P4_1P2P3Case, smoke_result: P4_1BackendSmokeResult) -> EpisodeArchive:
        selected = case.selection.selected_candidate
        config_hash = self._config_hash()
        metrics = _archive_metrics(case, smoke_result)
        archive_seed = {
            "task_id": case.sample.task_spec.task_id,
            "smoke_name": smoke_result.smoke_name,
            "runner": self.runner_config,
            "passed": smoke_result.passed,
        }
        return EpisodeArchive(
            episode_id=f"p4-1-backend-smoke-{stable_hash(archive_seed)[:8]}",
            task_spec=case.sample.task_spec,
            task_hash=case.sample.task_spec.stable_hash(),
            geometry_hashes={
                geometry.geometry_id: stable_hash(geometry)
                for geometry in case.sample.task_spec.scene.geometry_library
            },
            robot_model_hash=case.physical_model.stable_hash(),
            config_hash=config_hash,
            irg=case.irg,
            interaction_envelope=case.interaction_envelope,
            design_output=selected.design_output,
            feasibility_result=selected.feasibility_result,
            assembly_plan=case.assembly_report.plan.to_dict(),
            trajectory_records=[],
            policy_commands=[
                PolicyCommand(
                    desired_body_pose=(0.0, 0.0, case.sample.task_spec.scene.objects[0].pose_world[2], 0.0, 0.0, 0.0, 1.0),
                    desired_body_twist=[0.0] * 6,
                )
            ],
            controller_commands=smoke_result.controller_commands,
            rewards=[{"p4_1_backend_smoke_passed": 1.0 if smoke_result.passed else 0.0}],
            metrics=metrics,
            success=smoke_result.passed,
            failure_reason=None if smoke_result.passed else smoke_result.skip_reason or "p4_1_backend_smoke_failed",
            runtime_observations=smoke_result.runtime_observations,
            actuator_target_records=smoke_result.actuator_target_records,
            rollout_artifacts={
                "phase": "P4.1",
                "backend": smoke_result.backend,
                "archive_type": "p4_1_backend_smoke_per_step",
                "smoke_name": smoke_result.smoke_name,
                "p4_1_object_pose_history": [list(pose) for pose in smoke_result.object_pose_history],
                "p2_selected_design_used": True,
                "p3_assembled_morphology_used": bool(case.assembly_report.success),
                "assembled_morphology_graph_id": case.assembled_morphology.graph_id,
                "assembled_module_count": case.module_count,
                "is_p4_full_completion": False,
                "p4_2_rollout_claim": False,
                "object_grasp_carry_claim": False,
                "learning_claim": False,
                "physical_success_claim": False,
            },
            learning_artifacts={},
            reproducibility={
                "source_hash": self.runner_config.source_hash,
                "random_seed": self.runner_config.seed,
                "runner_version": self.runner_config.runner_version,
                "p3_config_path": self.runner_config.p3_config_path,
                "robot_model_config_path": self.runner_config.robot_model_config_path,
                "config_hash": config_hash,
                "smoke_result_hash": stable_hash(smoke_result),
                "urdf_hash": str(case.physical_model.metadata.get("urdf_hash", "")),
                "thrust_model_hash": str(case.physical_model.metadata.get("thrust_model_hash", "")),
            },
        )

    def _physical_model(self) -> PhysicalModel:
        if self.physical_model is None:
            self.physical_model = build_physical_model_from_config(self.runner_config.robot_model_config_path)
        return self.physical_model

    def _config_hash(self) -> str:
        return stable_hash({"runner": self.runner_config, "env": self.env_config})


def _p4_1_sample(sample: P2DesignTaskSample, sample_index: int) -> P2DesignTaskSample:
    task_data = sample.task_spec.to_dict()
    task_data["task_id"] = f"grasp_carry_box_001_p4_1_{sample_index:04d}"
    metadata = dict(task_data.get("metadata", {}) or {})
    metadata["p4_phase"] = "P4.1"
    metadata["p4_1_backend_smoke"] = True
    metadata["p4_full_completion"] = False
    task_data["metadata"] = metadata
    return P2DesignTaskSample(
        task_spec=TaskSpec.from_dict(task_data),
        seed=sample.seed,
        sample_index=sample.sample_index,
        sampled_values=sample.sampled_values,
    )


def _archive_metrics(case: P4_1P2P3Case, smoke_result: P4_1BackendSmokeResult) -> dict[str, float]:
    selected = case.selection.selected_candidate
    metrics = {
        **smoke_result.metrics,
        "success": 1.0 if smoke_result.passed else 0.0,
        "p2_selected_design_used": 1.0,
        "p3_assembly_result_used": 1.0 if case.assembly_report.success else 0.0,
        "selected_feasible": 1.0 if selected.feasibility_result.feasible else 0.0,
        "selected_candidate_id": float(selected.candidate_id),
        "selected_soft_score": float(selected.soft_score),
        "assembly_success": 1.0 if case.assembly_report.success else 0.0,
        "assembly_state_matches_target": 1.0 if case.assembly_report.state_matches_target else 0.0,
        "assembly_plan_step_count": float(len(case.assembly_report.plan.steps)),
        "assembly_executed_step_count": float(len(case.assembly_report.step_results)),
        "assembly_retry_count": float(case.assembly_report.retry_count),
        "assembly_abort_count": float(case.assembly_report.abort_count),
        "assembled_module_count": float(case.module_count),
        "fixed_two_module_only": 1.0 if case.module_count == 2 else 0.0,
        "runtime_observation_count": float(len(smoke_result.runtime_observations)),
        "controller_command_count": float(len(smoke_result.controller_commands)),
        "actuator_target_record_count": float(len(smoke_result.actuator_target_records)),
        "object_pose_history_count": float(len(smoke_result.object_pose_history)),
        "isaac_backed": 1.0 if smoke_result.isaac_backed else 0.0,
        "p4_1_backend_smoke_passed": 1.0 if smoke_result.passed else 0.0,
        "p4_full_completion": 0.0,
        "object_grasp_carry_success_claim": 0.0,
        "learned_policy_claim": 0.0,
        "p4_2_rollout_claim": 0.0,
    }
    for key, value in case.assembly_report.metrics.items():
        metrics[f"assembly_{key}"] = float(value)
    return metrics
