from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.assembly import AssemblyRunner, AssemblyRunnerConfig, SimplifiedAssemblyExecutor
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import P4_2DeterministicGraspCarryPlanner, p4_2_phase_from_knot
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignSelection
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactAssignment, ContactWrenchTrajectory, ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation import P4_2DeterministicRolloutConfig, P4_2DeterministicRolloutResult, P4_2IsaacEnv
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend
from amsrr.simulation.p4_2_isaac_env import load_p4_2_deterministic_rollout_env_config
from amsrr.training.p2_design_distribution import P2DesignTaskSample, P2GraspCarryDesignDistribution
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.training.p3_assembly_runner import load_p3_assembly_runner_config
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


P4_2_DETERMINISTIC_ROLLOUT_RUNNER_VERSION = "p4_2_deterministic_rollout_runner_v1"


@dataclass
class P4_2DeterministicRolloutRunnerConfig(SchemaBase):
    seed: int = 0
    sample_index: int = 0
    source_hash: str = "p4_2_deterministic_rollout"
    runner_version: str = P4_2_DETERMINISTIC_ROLLOUT_RUNNER_VERSION
    dry_run: bool = True
    archive_path: str | None = "artifacts/p4_2/p4_2_deterministic_rollout.jsonl"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    p3_config_path: str = "configs/training/p3_assembly_grasp_carry.yaml"

    def validate(self) -> None:
        require_non_empty(self.source_hash, "P4_2DeterministicRolloutRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P4_2DeterministicRolloutRunnerConfig.runner_version")
        require_non_empty(self.robot_model_config_path, "P4_2DeterministicRolloutRunnerConfig.robot_model_config_path")
        require_non_empty(self.p3_config_path, "P4_2DeterministicRolloutRunnerConfig.p3_config_path")
        if self.sample_index < 0:
            raise SchemaValidationError("P4_2DeterministicRolloutRunnerConfig.sample_index must be non-negative")


@dataclass
class P4_2P2P3RolloutCase:
    sample: P2DesignTaskSample
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    selection: P2DesignSelection
    assembly_report: object
    assembled_morphology: MorphologyGraph
    contact_candidate_set: ContactCandidateSet
    trajectory: ContactWrenchTrajectory
    physical_model: PhysicalModel

    @property
    def module_count(self) -> int:
        return len(self.assembled_morphology.modules)


@dataclass
class P4_2DeterministicRolloutRunnerResult(SchemaBase):
    dry_run: bool
    rollout_result: P4_2DeterministicRolloutResult
    acceptance_report: Any | None = None
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_p4_2_deterministic_rollout_runner_config(
    path: str | Path,
) -> tuple[P4_2DeterministicRolloutRunnerConfig, P4_2DeterministicRolloutConfig]:
    data = load_config(path)
    _, env_config = load_p4_2_deterministic_rollout_env_config(path)
    return P4_2DeterministicRolloutRunnerConfig.from_dict(data.get("runner", {})), env_config


class P4_2DeterministicRolloutRunner:
    """Run the P4.2 deterministic grasp/carry rollout over a P2/P3 morphology."""

    def __init__(
        self,
        *,
        runner_config: P4_2DeterministicRolloutRunnerConfig | None = None,
        env_config: P4_2DeterministicRolloutConfig | None = None,
        env: P4_2IsaacEnv | None = None,
        base_task_spec: TaskSpec | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> None:
        self.runner_config = runner_config or P4_2DeterministicRolloutRunnerConfig()
        self.env_config = env_config or load_p4_2_deterministic_rollout_env_config(
            "configs/training/p4_2_deterministic_rollout.yaml"
        )[1]
        self.env = env or P4_2IsaacEnv(
            config=self.env_config,
            backend=IsaacLabBackend(),
        )
        self.base_task_spec = base_task_spec or default_grasp_carry_task_spec()
        self.physical_model = physical_model

    def run(self, *, archive_path: str | Path | None = None) -> P4_2DeterministicRolloutRunnerResult:
        from amsrr.acceptance.p4_2_acceptance import run_p4_2_acceptance

        case = self.build_p2_p3_rollout_case()
        rollout_result = self.env.run_rollout(
            dry_run=self.runner_config.dry_run,
            morphology_graph=case.assembled_morphology,
            uses_p2_selected_design=True,
            uses_p3_assembled_morphology=bool(case.assembly_report.success),
        )
        archives: list[EpisodeArchive] = []
        if rollout_result.attempted and not rollout_result.skipped:
            archives.append(self.build_archive(case, rollout_result))

        output_path = archive_path
        if output_path is None and self.runner_config.archive_path is not None:
            output_path = self.runner_config.archive_path
        if output_path is not None and archives:
            write_episode_archives_jsonl(output_path, archives)

        acceptance_report = run_p4_2_acceptance(archives, rollout_results=[rollout_result])
        return P4_2DeterministicRolloutRunnerResult(
            dry_run=self.runner_config.dry_run,
            rollout_result=rollout_result,
            acceptance_report=acceptance_report,
            archives=archives,
            metrics={
                **acceptance_report.metrics,
                **_case_metrics(case, rollout_result),
                "dry_run": 1.0 if self.runner_config.dry_run else 0.0,
                "archive_count": float(len(archives)),
            },
        )

    def build_p2_p3_rollout_case(self) -> P4_2P2P3RolloutCase:
        p3_runner_config, distribution_config, policy_config = load_p3_assembly_runner_config(
            self.runner_config.p3_config_path
        )
        physical_model = self._physical_model()
        sample = P2GraspCarryDesignDistribution(self.base_task_spec, distribution_config).sample(
            seed=self.runner_config.seed,
            sample_index=self.runner_config.sample_index,
        )
        sample = _p4_2_sample(sample, self.runner_config.sample_index)
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
        assembled_morphology = assembly_report.final_state.physical_graph
        candidate_set = ContactCandidateSampler().sample(
            task_spec=sample.task_spec,
            irg=irg,
            interaction_envelope=envelope,
            morphology_graph=assembled_morphology,
            geometry_descriptors=builder_result.scene_graph.geometry_descriptors,
        )
        runtime = _initial_runtime_observation(sample.task_spec, assembled_morphology)
        trajectory = P4_2DeterministicGraspCarryPlanner().plan(
            HighLevelPolicyContext(
                irg=irg,
                interaction_envelope=envelope,
                morphology_graph=assembled_morphology,
                contact_candidate_set=candidate_set,
                runtime_observation=runtime,
            )
        )
        return P4_2P2P3RolloutCase(
            sample=sample,
            irg=irg,
            interaction_envelope=envelope,
            selection=selection,
            assembly_report=assembly_report,
            assembled_morphology=assembled_morphology,
            contact_candidate_set=candidate_set,
            trajectory=trajectory,
            physical_model=physical_model,
        )

    def build_archive(
        self,
        case: P4_2P2P3RolloutCase,
        rollout_result: P4_2DeterministicRolloutResult,
    ) -> EpisodeArchive:
        selected = case.selection.selected_candidate
        config_hash = self._config_hash()
        metrics = _archive_metrics(case, rollout_result)
        archive_seed = {
            "task_id": case.sample.task_spec.task_id,
            "rollout_name": rollout_result.rollout_name,
            "runner": self.runner_config,
            "passed": rollout_result.passed,
        }
        return EpisodeArchive(
            episode_id=f"p4-2-deterministic-rollout-{stable_hash(archive_seed)[:8]}",
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
            trajectory_records=[case.trajectory],
            policy_commands=rollout_result.policy_commands,
            controller_commands=rollout_result.controller_commands,
            rewards=[{"p4_2_deterministic_rollout_success": 1.0 if rollout_result.passed else 0.0}],
            metrics=metrics,
            success=rollout_result.passed,
            failure_reason=None if rollout_result.passed else rollout_result.skip_reason or "p4_2_rollout_failed",
            runtime_observations=rollout_result.runtime_observations,
            actuator_target_records=rollout_result.actuator_target_records,
            rollout_artifacts={
                "phase": "P4.2",
                "backend": rollout_result.backend,
                "archive_type": "p4_2_deterministic_rollout_per_step",
                "rollout_name": rollout_result.rollout_name,
                "contact_model": rollout_result.contact_model,
                "p4_2_phase_transitions": [transition.to_dict() for transition in rollout_result.phase_transitions],
                "p4_2_attach_events": [event.to_dict() for event in rollout_result.attach_events],
                "contact_candidate_set": case.contact_candidate_set.to_dict(),
                "selected_contact_candidate_ids": _selected_candidate_ids(case.trajectory),
                "selected_contact_assignments": [
                    assignment.to_dict()
                    for assignment in _selected_contact_assignments(case.trajectory)
                ],
                "p4_2_phase_sequence": [
                    phase
                    for phase in (p4_2_phase_from_knot(knot) for knot in case.trajectory.knots)
                    if phase is not None
                ],
                "p2_selected_design_used": True,
                "p3_assembled_morphology_used": bool(case.assembly_report.success),
                "assembled_morphology_graph_id": case.assembled_morphology.graph_id,
                "assembled_module_count": case.module_count,
                "morphology_asset_reflected": rollout_result.morphology_asset_reflected,
                "module_placement_reflected": rollout_result.module_placement_reflected,
                "actuator_mapping_reflected": rollout_result.actuator_mapping_reflected,
                "object_attach_release_only": bool(
                    rollout_result.rollout_artifacts.get("object_attach_release_only", True)
                ),
                "module_attach_detach_claim": False,
                "dynamic_morphology_update_claim": False,
                "asset_generation_semantics": rollout_result.rollout_artifacts.get(
                    "asset_generation_semantics",
                    "reset_time_fixed_morphology_not_pi_a_dynamic_construction",
                ),
                "is_p4_full_completion": False,
                "p4_3_learning_bootstrap": False,
                "learning_claim": False,
                "learned_policy_success_claim": False,
                "high_fidelity_natural_grasp_success_claim": False,
                "checkpoint_claim": False,
                "reward_curve_training_claim": False,
                "real_isaac_completion_claim": False,
            },
            learning_artifacts={},
            reproducibility={
                "source_hash": self.runner_config.source_hash,
                "random_seed": self.runner_config.seed,
                "runner_version": self.runner_config.runner_version,
                "p3_config_path": self.runner_config.p3_config_path,
                "robot_model_config_path": self.runner_config.robot_model_config_path,
                "config_hash": config_hash,
                "rollout_result_hash": stable_hash(rollout_result),
                "trajectory_hash": stable_hash(case.trajectory),
                "contact_candidate_set_hash": stable_hash(case.contact_candidate_set),
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


def _p4_2_sample(sample: P2DesignTaskSample, sample_index: int) -> P2DesignTaskSample:
    task_data = sample.task_spec.to_dict()
    task_data["task_id"] = f"grasp_carry_box_001_p4_2_{sample_index:04d}"
    metadata = dict(task_data.get("metadata", {}) or {})
    metadata["p4_phase"] = "P4.2"
    metadata["p4_2_deterministic_rollout"] = True
    metadata["p4_full_completion"] = False
    metadata["p4_3_learning_bootstrap"] = False
    task_data["metadata"] = metadata
    return P2DesignTaskSample(
        task_spec=TaskSpec.from_dict(task_data),
        seed=sample.seed,
        sample_index=sample.sample_index,
        sampled_values=sample.sampled_values,
    )


def _initial_runtime_observation(task_spec: TaskSpec, morphology_graph: MorphologyGraph) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology_graph,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions={},
                joint_velocities={},
            )
            for module in morphology_graph.modules
        ],
        object_states=[
            ObjectRuntimeState(
                object_id=obj.object_id,
                pose_world=obj.pose_world,
                twist_world=[0.0] * 6,
            )
            for obj in task_spec.scene.objects
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(phase_label="reset", progress_ratio=0.0),
    )


def _selected_contact_assignments(trajectory: ContactWrenchTrajectory) -> list[ContactAssignment]:
    assignments: dict[tuple[int, int, int], ContactAssignment] = {}
    for knot in trajectory.knots:
        for assignment in knot.contact_assignments:
            key = (assignment.slot_id, assignment.anchor_id, assignment.candidate_id)
            assignments[key] = assignment
    return [assignments[key] for key in sorted(assignments)]


def _selected_candidate_ids(trajectory: ContactWrenchTrajectory) -> list[int]:
    return sorted({assignment.candidate_id for assignment in _selected_contact_assignments(trajectory)})


def _case_metrics(
    case: P4_2P2P3RolloutCase,
    rollout_result: P4_2DeterministicRolloutResult,
) -> dict[str, float]:
    selected = case.selection.selected_candidate
    metrics = {
        **rollout_result.metrics,
        "success": 1.0 if rollout_result.passed else 0.0,
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
        "contact_candidate_count": float(len(case.contact_candidate_set.candidates)),
        "selected_contact_candidate_count": float(len(_selected_candidate_ids(case.trajectory))),
        "trajectory_knot_count": float(len(case.trajectory.knots)),
        "runtime_observation_count": float(len(rollout_result.runtime_observations)),
        "controller_command_count": float(len(rollout_result.controller_commands)),
        "actuator_target_record_count": float(len(rollout_result.actuator_target_records)),
        "attach_event_count": float(len(rollout_result.attach_events)),
        "isaac_backed": 1.0 if rollout_result.isaac_backed else 0.0,
        "p4_2_deterministic_rollout_passed": 1.0 if rollout_result.passed else 0.0,
        "p4_full_completion": 0.0,
        "p4_3_learning_bootstrap": 0.0,
        "learned_policy_success_claim": 0.0,
        "high_fidelity_natural_grasp_success_claim": 0.0,
        "checkpoint_claim": 0.0,
        "reward_curve_training_claim": 0.0,
        "real_isaac_completion_claim": 0.0,
        "module_attach_detach_claim": 0.0,
        "dynamic_morphology_update_claim": 0.0,
    }
    for key, value in case.assembly_report.metrics.items():
        metrics[f"assembly_{key}"] = float(value)
    return metrics


def _archive_metrics(
    case: P4_2P2P3RolloutCase,
    rollout_result: P4_2DeterministicRolloutResult,
) -> dict[str, float]:
    return _case_metrics(case, rollout_result)
