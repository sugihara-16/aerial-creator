from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amsrr.assembly import AssemblyRunner, AssemblyRunnerConfig, SimplifiedAssemblyExecutor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import P4_2DeterministicGraspCarryPlanner
from amsrr.policies.design_policy_p2 import P2DesignCandidateEvaluation, P2DesignSelection
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.simulation.p4_2_rollout import P4_2DeterministicRolloutResult
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.p4_2_isaac_env import P4_2IsaacEnv
from amsrr.training.p4_2_deterministic_rollout_runner import (
    P4_2DeterministicRolloutRunner,
    P4_2DeterministicRolloutRunnerConfig,
    P4_2P2P3RolloutCase,
    _initial_runtime_observation,
    _p4_2_rollout_object_params,
    load_p4_2_deterministic_rollout_runner_config,
)
from amsrr.training.p4_3_reward import compute_p4_3_archive_rewards
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file, stable_hash


P4_3_ROLLOUT_RUNNER_VERSION = "p4_3_rollout_runner_v1"


@dataclass
class P4_3RolloutRunnerConfig(SchemaBase):
    seed: int = 0
    task_start_index: int = 0
    task_count: int = 6
    candidates_per_task: int = 2
    candidate_offset: int = 0
    dry_run: bool = True
    p4_2_config_path: str = "configs/training/p4_2_deterministic_rollout.yaml"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    archive_path: str = "artifacts/p4_3/rollouts/deterministic_isaac.jsonl"
    dataset_dir: str = "artifacts/p4_3/datasets"
    split_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "train": 2.0 / 3.0,
            "validation": 1.0 / 6.0,
            "held_out": 1.0 / 6.0,
        }
    )
    contact_model: str = "kinematic_payload_coupled_attach_v1"
    learned_pi_l_checkpoint_path: str | None = None
    learned_pi_l_runtime_blend_factor: float = 0.10
    source_hash: str = "p4_3_deterministic_dataset_collection"
    runner_version: str = P4_3_ROLLOUT_RUNNER_VERSION

    def validate(self) -> None:
        if self.task_count < 1:
            raise SchemaValidationError("P4_3RolloutRunnerConfig.task_count must be positive")
        if self.task_start_index < 0:
            raise SchemaValidationError("P4_3RolloutRunnerConfig.task_start_index must be non-negative")
        if self.candidates_per_task < 1:
            raise SchemaValidationError("P4_3RolloutRunnerConfig.candidates_per_task must be positive")
        if self.candidate_offset < 0:
            raise SchemaValidationError("P4_3RolloutRunnerConfig.candidate_offset must be non-negative")
        if not 0.0 < self.learned_pi_l_runtime_blend_factor <= 1.0:
            raise SchemaValidationError(
                "P4_3RolloutRunnerConfig.learned_pi_l_runtime_blend_factor must be in (0, 1]"
            )
        for name in (
            "p4_2_config_path",
            "robot_model_config_path",
            "archive_path",
            "dataset_dir",
            "source_hash",
            "runner_version",
            "contact_model",
        ):
            require_non_empty(getattr(self, name), f"P4_3RolloutRunnerConfig.{name}")
        required_splits = {"train", "validation", "held_out"}
        if set(self.split_fractions) != required_splits:
            raise SchemaValidationError(
                "P4_3RolloutRunnerConfig.split_fractions must define train, validation, held_out"
            )
        if any(value <= 0.0 for value in self.split_fractions.values()):
            raise SchemaValidationError("P4_3 split fractions must be positive")
        if abs(sum(self.split_fractions.values()) - 1.0) > 1.0e-6:
            raise SchemaValidationError("P4_3 split fractions must sum to 1")


@dataclass
class P4_3CandidateRolloutResult(SchemaBase):
    task_index: int
    candidate_id: int
    variant: str
    deterministic_feasible: bool
    rollout_result: P4_2DeterministicRolloutResult | None
    episode_id: str | None


@dataclass
class P4_3RolloutRunnerResult(SchemaBase):
    dry_run: bool
    candidate_results: list[P4_3CandidateRolloutResult]
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_p4_3_rollout_runner_config(path: str | Path) -> P4_3RolloutRunnerConfig:
    data = load_config(path)
    return P4_3RolloutRunnerConfig.from_dict(data.get("collection", {}))


class P4_3RolloutRunner:
    """Collect deterministic Isaac outcomes without changing the P4.2 gate."""

    def __init__(self, config: P4_3RolloutRunnerConfig | None = None) -> None:
        self.config = config or P4_3RolloutRunnerConfig()

    def run(self, *, archive_path: str | Path | None = None) -> P4_3RolloutRunnerResult:
        archives: list[EpisodeArchive] = []
        candidate_results: list[P4_3CandidateRolloutResult] = []
        for task_index in range(
            self.config.task_start_index,
            self.config.task_start_index + self.config.task_count,
        ):
            p4_2_runner = self._p4_2_runner(task_index)
            base_case = p4_2_runner.build_p2_p3_rollout_case()
            candidates = self._rollout_candidates(base_case.selection)
            for candidate in candidates:
                case = self._candidate_case(base_case, candidate)
                object_pose, object_size, object_mass = _p4_2_rollout_object_params(
                    case.sample.task_spec,
                    p4_2_runner.env_config,
                )
                rollout = p4_2_runner.env.run_rollout(
                    dry_run=self.config.dry_run,
                    morphology_graph=case.assembled_morphology,
                    contact_candidate_set=case.contact_candidate_set,
                    contact_wrench_trajectory=case.trajectory,
                    object_pose_world=object_pose,
                    object_size_m=object_size,
                    object_mass_kg=object_mass,
                    uses_p2_selected_design=True,
                    uses_p3_assembled_morphology=bool(case.assembly_report.success),
                    learned_pi_l_checkpoint_path=self.config.learned_pi_l_checkpoint_path,
                    learned_pi_l_runtime_blend_factor=(
                        self.config.learned_pi_l_runtime_blend_factor
                    ),
                )
                episode_id: str | None = None
                if rollout.attempted and not rollout.skipped:
                    archive = p4_2_runner.build_archive(case, rollout)
                    episode_id = self._annotate_archive(archive, task_index, candidate)
                    archives.append(archive)
                candidate_results.append(
                    P4_3CandidateRolloutResult(
                        task_index=task_index,
                        candidate_id=candidate.candidate_id,
                        variant=candidate.variant,
                        deterministic_feasible=candidate.feasibility_result.feasible,
                        rollout_result=rollout,
                        episode_id=episode_id,
                    )
                )
        output_path = Path(archive_path or self.config.archive_path)
        if archives:
            write_episode_archives_jsonl(output_path, archives)
        return P4_3RolloutRunnerResult(
            dry_run=self.config.dry_run,
            candidate_results=candidate_results,
            archives=archives,
            metrics={
                "task_count": float(self.config.task_count),
                "candidate_rollout_count": float(len(candidate_results)),
                "archive_count": float(len(archives)),
                "isaac_backed_count": float(
                    sum(
                        1
                        for item in candidate_results
                        if item.rollout_result is not None and item.rollout_result.isaac_backed
                    )
                ),
                "success_count": float(
                    sum(
                        1
                        for item in candidate_results
                        if item.rollout_result is not None and item.rollout_result.passed
                    )
                ),
                "learned_pi_l_requested": 1.0 if self.config.learned_pi_l_checkpoint_path else 0.0,
            },
        )

    def _p4_2_runner(self, task_index: int) -> P4_2DeterministicRolloutRunner:
        runner_config, env_config = load_p4_2_deterministic_rollout_runner_config(
            self.config.p4_2_config_path
        )
        if env_config.contact_model != self.config.contact_model:
            raise SchemaValidationError(
                "P4.3 collection contact_model must match the P4.2 Isaac environment contract"
            )
        runner_config.seed = self.config.seed + task_index
        runner_config.sample_index = task_index
        runner_config.dry_run = self.config.dry_run
        runner_config.archive_path = None
        runner_config.robot_model_config_path = self.config.robot_model_config_path
        backend = IsaacLabBackend(load_isaac_lab_backend_config(env_config.config_path))
        env = P4_2IsaacEnv(config=env_config, backend=backend)
        return P4_2DeterministicRolloutRunner(
            runner_config=runner_config,
            env_config=env_config,
            env=env,
        )

    def _rollout_candidates(
        self,
        selection: P2DesignSelection,
    ) -> list[P2DesignCandidateEvaluation]:
        accepted = sorted(
            selection.accepted_candidates,
            key=lambda item: (
                item.candidate_id != selection.selected_candidate.candidate_id,
                -item.soft_score,
                item.candidate_id,
            ),
        )
        start = self.config.candidate_offset
        stop = start + self.config.candidates_per_task
        selected = accepted[start:stop]
        if len(selected) != self.config.candidates_per_task:
            raise SchemaValidationError(
                "P4.3 requested candidate window exceeds deterministic feasible candidates"
            )
        return selected

    def _candidate_case(
        self,
        base_case: P4_2P2P3RolloutCase,
        candidate: P2DesignCandidateEvaluation,
    ) -> P4_2P2P3RolloutCase:
        if not candidate.feasibility_result.feasible:
            raise SchemaValidationError("P4.3 must not execute hard-infeasible designs in Isaac")
        target = candidate.design_output.target_morphology
        assembly_report = AssemblyRunner(config=AssemblyRunnerConfig()).run(
            target,
            SimplifiedAssemblyExecutor(target_graph=target),
        )
        assembled = assembly_report.final_state.physical_graph
        builder_result = IRGBuilder().build_with_scene_graph(base_case.sample.task_spec)
        candidate_set = ContactCandidateSampler().sample(
            task_spec=base_case.sample.task_spec,
            irg=base_case.irg,
            interaction_envelope=base_case.interaction_envelope,
            morphology_graph=assembled,
            geometry_descriptors=builder_result.scene_graph.geometry_descriptors,
        )
        runtime = _initial_runtime_observation(base_case.sample.task_spec, assembled)
        trajectory = P4_2DeterministicGraspCarryPlanner().plan(
            HighLevelPolicyContext(
                irg=base_case.irg,
                interaction_envelope=base_case.interaction_envelope,
                morphology_graph=assembled,
                contact_candidate_set=candidate_set,
                runtime_observation=runtime,
            )
        )
        candidate_selection = P2DesignSelection(
            candidates=base_case.selection.candidates,
            accepted_candidates=base_case.selection.accepted_candidates,
            rejected_candidates=base_case.selection.rejected_candidates,
            selected_candidate=candidate,
            policy_version=base_case.selection.policy_version,
        )
        return P4_2P2P3RolloutCase(
            sample=base_case.sample,
            irg=base_case.irg,
            interaction_envelope=base_case.interaction_envelope,
            selection=candidate_selection,
            assembly_report=assembly_report,
            assembled_morphology=assembled,
            contact_candidate_set=candidate_set,
            trajectory=trajectory,
            physical_model=base_case.physical_model,
        )

    def _annotate_archive(
        self,
        archive: EpisodeArchive,
        task_index: int,
        candidate: P2DesignCandidateEvaluation,
    ) -> str:
        seed = {
            "task_id": archive.task_spec.task_id,
            "task_index": task_index,
            "candidate_id": candidate.candidate_id,
            "variant": candidate.variant,
            "config": self.config,
        }
        archive.episode_id = f"p4-3-dataset-{stable_hash(seed)[:12]}"
        learned_evaluation = self.config.learned_pi_l_checkpoint_path is not None
        learned_checkpoint_loaded = (
            archive.metrics.get("p4_3_pi_l_checkpoint_loaded", 0.0) > 0.5
        )
        learned_decision_count = archive.metrics.get(
            "p4_3_pi_l_learned_decision_count", 0.0
        )
        if learned_evaluation:
            archive.episode_id = f"p4-3-pi-l-eval-{stable_hash(seed)[:12]}"
        archive.rollout_artifacts.update(
            {
                "phase": "P4.3b" if learned_evaluation else "P4.3a",
                "archive_type": (
                    "p4_3_pi_l_online_isaac_evaluation"
                    if learned_evaluation
                    else "p4_3_deterministic_isaac_dataset"
                ),
                "p4_3_dataset_collection": not learned_evaluation,
                "p4_3_learned_evaluation": learned_evaluation,
                "p4_3_task_index": task_index,
                "p4_3_candidate_id": candidate.candidate_id,
                "p4_3_candidate_variant": candidate.variant,
                "learning_claim": (
                    learned_evaluation
                    and learned_checkpoint_loaded
                    and learned_decision_count > 0.0
                ),
                "learned_policy_success_claim": False,
                "is_p4_full_completion": False,
                "high_fidelity_natural_grasp_success_claim": False,
            }
        )
        archive.learning_artifacts = {
            "stage": "P4.3b" if learned_evaluation else "P4.3a",
            "dataset_only": not learned_evaluation,
            "deterministic_pi_d_fallback": True,
            "deterministic_pi_h_fallback": True,
            "deterministic_pi_l_fallback": True,
            "runner_version": self.config.runner_version,
        }
        if learned_evaluation and self.config.learned_pi_l_checkpoint_path is not None:
            checkpoint_path = Path(self.config.learned_pi_l_checkpoint_path)
            archive.learning_artifacts.update(
                {
                    "pi_l_checkpoint_path": self.config.learned_pi_l_checkpoint_path,
                    "pi_l_checkpoint_sha256": (
                        hash_file(checkpoint_path) if checkpoint_path.is_file() else None
                    ),
                    "pi_l_checkpoint_loaded": learned_checkpoint_loaded,
                    "pi_l_online_inference": learned_checkpoint_loaded,
                    "pi_l_learned_decision_count": learned_decision_count,
                    "pi_l_fallback_count": archive.metrics.get("p4_3_pi_l_fallback_count", 0.0),
                    "pi_l_runtime_blend_factor": archive.metrics.get(
                        "p4_3_pi_l_runtime_blend_factor", 0.0
                    ),
                    "pi_l_overlay_nonzero_count": archive.metrics.get(
                        "p4_3_pi_l_overlay_nonzero_count", 0.0
                    ),
                    "pi_l_overlay_delta_norm_sum": archive.metrics.get(
                        "p4_3_pi_l_overlay_delta_norm_sum", 0.0
                    ),
                    "pi_l_overlay_delta_norm_max": archive.metrics.get(
                        "p4_3_pi_l_overlay_delta_norm_max", 0.0
                    ),
                }
            )
        archive.rewards = compute_p4_3_archive_rewards(archive)
        archive.metrics["p4_3_reward_record_count"] = float(len(archive.rewards))
        archive.metrics["p4_3_episode_return"] = float(
            sum(record.get("reward", 0.0) for record in archive.rewards)
        )
        archive.reproducibility.update(
            {
                "p4_3_runner_version": self.config.runner_version,
                "p4_3_source_hash": self.config.source_hash,
                "p4_3_candidate_id": candidate.candidate_id,
                "p4_3_candidate_variant": candidate.variant,
            }
        )
        return archive.episode_id
