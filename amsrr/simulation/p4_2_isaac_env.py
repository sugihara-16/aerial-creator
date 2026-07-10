from __future__ import annotations

from pathlib import Path
from typing import Any

from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory, ControllerCommand, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, IsaacLabBackendConfig, load_isaac_lab_backend_config
from amsrr.simulation.p4_2_rollout import (
    P4_2_CONTACT_MODEL,
    P4_2AttachEvent,
    P4_2DeterministicRolloutConfig,
    P4_2DeterministicRolloutResult,
    P4_2PhaseTransitionRecord,
    P4_2ReleaseEvent,
    P4_2RolloutPhase,
)
from amsrr.utils.config import load_config


P4_2_ISAAC_ENV_VERSION = "p4_2_isaac_env_v1"


def load_p4_2_deterministic_rollout_env_config(
    path: str | Path,
) -> tuple[IsaacLabBackendConfig, P4_2DeterministicRolloutConfig]:
    data = load_config(path)
    env_config = P4_2DeterministicRolloutConfig.from_dict(data.get("env", {}))
    backend_config = load_isaac_lab_backend_config(env_config.config_path)
    return backend_config, env_config


class P4_2IsaacEnv:
    """P4.2 deterministic grasp/carry rollout boundary.

    This boundary consumes a P3 assembled MorphologyGraph as an already-built
    robot structure. Rollout object attach/release events do not edit robot
    morphology and are not pi_A module attach/detach events.
    """

    def __init__(
        self,
        *,
        backend: IsaacLabBackend | None = None,
        config: P4_2DeterministicRolloutConfig | None = None,
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
    ) -> None:
        if keep_open_after_rollout_s < 0.0:
            raise ValueError("keep_open_after_rollout_s must be non-negative")
        self.config = config or P4_2DeterministicRolloutConfig()
        self.backend = backend or IsaacLabBackend()
        self.viewer = viewer
        self.realtime_playback = realtime_playback
        self.keep_open_after_rollout_s = keep_open_after_rollout_s

    def run_rollout(
        self,
        *,
        dry_run: bool = True,
        morphology_graph: MorphologyGraph | None = None,
        contact_candidate_set: ContactCandidateSet | None = None,
        contact_wrench_trajectory: ContactWrenchTrajectory | None = None,
        object_pose_world: tuple[float, float, float, float, float, float, float] | None = None,
        object_size_m: tuple[float, float, float] | None = None,
        object_mass_kg: float | None = None,
        uses_p2_selected_design: bool = True,
        uses_p3_assembled_morphology: bool = True,
    ) -> P4_2DeterministicRolloutResult:
        if dry_run:
            return P4_2DeterministicRolloutResult(
                rollout_name=self.config.rollout_name,
                attempted=False,
                passed=False,
                skipped=True,
                isaac_backed=False,
                skip_reason="dry_run",
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                morphology_asset_reflected=False,
                module_placement_reflected=False,
                actuator_mapping_reflected=False,
                rollout_artifacts={
                    "phase": "P4.2",
                    "dry_run": True,
                    "module_attach_detach_claim": False,
                    "object_attach_release_only": True,
                },
                metrics={"dry_run": 1.0},
            )
        if morphology_graph is None:
            return P4_2DeterministicRolloutResult(
                rollout_name=self.config.rollout_name,
                attempted=False,
                passed=False,
                skipped=True,
                isaac_backed=False,
                skip_reason="missing_morphology_graph",
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                metrics={"missing_morphology_graph": 1.0},
            )

        availability = self.backend.availability()
        if not availability.available:
            return P4_2DeterministicRolloutResult(
                rollout_name=self.config.rollout_name,
                attempted=False,
                passed=False,
                skipped=True,
                isaac_backed=False,
                skip_reason=",".join(availability.missing_reasons),
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                metrics={"isaac_backend_available": 0.0},
            )

        try:
            report = self.backend.run_p4_2_deterministic_rollout(
                config_path=self.config.config_path,
                force_convert=True,
                steps=self.config.max_episode_steps,
                morphology_graph=morphology_graph,
                contact_candidate_set_json=(
                    contact_candidate_set.to_json() if contact_candidate_set is not None else None
                ),
                contact_wrench_trajectory_json=(
                    contact_wrench_trajectory.to_json() if contact_wrench_trajectory is not None else None
                ),
                object_size_m=object_size_m or self.config.object_size_m,
                object_mass_kg=float(object_mass_kg or self.config.object_mass_kg),
                object_pose_world=object_pose_world or self.config.object_initial_pose_world,
                contact_model=self.config.contact_model,
                attach_distance_threshold_m=self.config.attach_distance_threshold_m,
                attach_relative_velocity_threshold_mps=self.config.attach_relative_velocity_threshold_mps,
                attach_snap_distance_threshold_m=self.config.attach_snap_distance_threshold_m,
                pregrasp_alignment_distance_m=self.config.pregrasp_alignment_distance_m,
                uses_p2_p3_design=uses_p2_selected_design and uses_p3_assembled_morphology,
                viewer=self.viewer,
                realtime_playback=self.realtime_playback,
                keep_open_after_smoke_s=self.keep_open_after_rollout_s,
            )
        except Exception as exc:  # pragma: no cover - real subprocess failures are environment-specific.
            return P4_2DeterministicRolloutResult(
                rollout_name=self.config.rollout_name,
                attempted=True,
                passed=False,
                skipped=False,
                isaac_backed=True,
                skip_reason=str(exc),
                uses_p2_selected_design=uses_p2_selected_design,
                uses_p3_assembled_morphology=uses_p3_assembled_morphology,
                metrics={"runner_exception": 1.0},
            )
        return p4_2_result_from_report(
            report,
            rollout_name=self.config.rollout_name,
            uses_p2_selected_design=uses_p2_selected_design,
            uses_p3_assembled_morphology=uses_p3_assembled_morphology,
        )


def p4_2_result_from_report(
    report: dict[str, Any],
    *,
    rollout_name: str,
    uses_p2_selected_design: bool,
    uses_p3_assembled_morphology: bool,
) -> P4_2DeterministicRolloutResult:
    runtime_observations = [
        RuntimeObservation.from_dict(item)
        for item in report.get("p4_2_runtime_observations", [])
        if isinstance(item, dict)
    ]
    policy_commands = [
        PolicyCommand.from_dict(item)
        for item in report.get("p4_2_policy_commands", [])
        if isinstance(item, dict)
    ]
    controller_commands = [
        ControllerCommand.from_dict(item)
        for item in report.get("p4_2_controller_commands", [])
        if isinstance(item, dict)
    ]
    actuator_target_records = [
        item for item in report.get("p4_2_actuator_target_records", []) if isinstance(item, dict)
    ]
    phase_transitions = [
        P4_2PhaseTransitionRecord.from_dict(item)
        for item in report.get("p4_2_phase_transitions", [])
        if isinstance(item, dict)
    ]
    attach_events = [
        P4_2AttachEvent.from_dict(item)
        for item in report.get("p4_2_attach_events", [])
        if isinstance(item, dict)
    ]
    release_events = [
        P4_2ReleaseEvent.from_dict(item)
        for item in report.get("p4_2_release_events", [])
        if isinstance(item, dict)
    ]
    final_phase = _final_phase(report)
    metrics = _numeric_report_metrics(report)
    metrics.setdefault("p4_full_completion", 0.0)
    metrics.setdefault("p4_3_learning_bootstrap", 0.0)
    metrics.setdefault("learned_policy_success_claim", 0.0)
    metrics.setdefault("high_fidelity_natural_grasp_success_claim", 0.0)
    metrics.setdefault("true_fixed_joint_dynamics_success_claim", 0.0)
    morphology_asset_reflected = bool(report.get("p4_2_morphology_asset_reflected", False))
    module_placement_reflected = bool(report.get("p4_2_module_placement_reflected", False))
    actuator_mapping_reflected = bool(report.get("p4_2_actuator_mapping_reflected", False))
    p2_p3 = bool(report.get("p4_2_uses_p2_p3", True))
    passed = (
        bool(report.get("p4_2_deterministic_rollout_passed", False))
        and final_phase == P4_2RolloutPhase.SUCCESS
        and bool(attach_events)
        and bool(release_events)
        and morphology_asset_reflected
        and module_placement_reflected
        and actuator_mapping_reflected
    )
    return P4_2DeterministicRolloutResult(
        rollout_name=rollout_name,
        attempted=True,
        passed=passed,
        skipped=False,
        isaac_backed=bool(report.get("isaac_backed", True)),
        contact_model=str(report.get("p4_2_contact_model", P4_2_CONTACT_MODEL)),
        uses_p2_selected_design=uses_p2_selected_design and p2_p3,
        uses_p3_assembled_morphology=uses_p3_assembled_morphology and p2_p3,
        morphology_asset_reflected=morphology_asset_reflected,
        module_placement_reflected=module_placement_reflected,
        actuator_mapping_reflected=actuator_mapping_reflected,
        final_phase=final_phase,
        skip_reason=None if passed else str(report.get("error", "p4_2_rollout_failed")),
        phase_transitions=phase_transitions,
        attach_events=attach_events,
        release_events=release_events,
        runtime_observations=runtime_observations,
        policy_commands=policy_commands,
        controller_commands=controller_commands,
        actuator_target_records=actuator_target_records,
        metrics=metrics,
        rollout_artifacts={
            "phase": "P4.2",
            "backend": "isaac_lab",
            "contact_model": str(report.get("p4_2_contact_model", P4_2_CONTACT_MODEL)),
            "object_attach_release_only": bool(report.get("p4_2_object_attach_release_only", True)),
            "module_attach_detach_claim": bool(report.get("p4_2_module_attach_detach_claim", False)),
            "dynamic_morphology_update_claim": bool(
                report.get("p4_2_dynamic_morphology_update_claim", False)
            ),
            "asset_generation_semantics": str(
                report.get(
                    "p4_2_asset_generation_semantics",
                    "reset_time_fixed_morphology_not_pi_a_dynamic_construction",
                )
            ),
            "link_backed_anchor_pose_used": bool(report.get("p4_2_link_backed_anchor_pose_used", False)),
            "anchor_pose_source": str(report.get("p4_2_anchor_pose_source", "")),
            "anchor_link_id": str(report.get("p4_2_anchor_link_id", "")),
            "anchor_resolved_body_name": str(report.get("p4_2_anchor_resolved_body_name", "")),
            "anchor_debug_samples": list(report.get("p4_2_anchor_debug_samples", [])),
            "is_p4_full_completion": False,
            "p4_3_learning_bootstrap": False,
            "learning_claim": False,
            "learned_policy_success_claim": False,
            "high_fidelity_natural_grasp_success_claim": False,
            "true_fixed_joint_dynamics_success_claim": False,
            "checkpoint_claim": False,
            "reward_curve_training_claim": False,
            "p4_4_natural_contact_grasp_remaining": True,
        },
    )


def _final_phase(report: dict[str, Any]) -> P4_2RolloutPhase:
    raw = report.get("p4_2_final_phase", P4_2RolloutPhase.TIMEOUT_FAILURE.value)
    return P4_2RolloutPhase(str(raw))


def _numeric_report_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in report.items():
        if isinstance(value, bool):
            metrics[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics
