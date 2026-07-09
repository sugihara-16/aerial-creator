from __future__ import annotations

from dataclasses import dataclass, field

from amsrr.logging import EpisodeArchive
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.simulation import (
    P4_2_REQUIRED_REAL_ROLLOUTS,
    P4_2DeterministicRolloutResult,
    P4_2RolloutPhase,
)


P4_2_PHASE_SEQUENCE = [
    "approach",
    "pregrasp_align",
    "attach_attempt",
    "attached_maintain",
    "transport",
    "release",
]


@dataclass
class P4_2AcceptanceReport(SchemaBase):
    fast_gate_passed: bool
    real_isaac_rollout_passed: bool
    completion_passed: bool
    archive_count: int
    rollout_result_count: int
    p2_selected_design_used: bool
    p3_assembly_result_used: bool
    trajectory_records_saved: bool
    selected_contact_candidates_saved: bool
    phase_sequence_saved: bool
    per_step_runtime_observations_saved: bool
    per_step_policy_commands_saved: bool
    per_step_controller_commands_saved: bool
    per_step_actuator_target_records_saved: bool
    attach_events_saved: bool
    release_events_saved: bool
    payload_coupling_saved: bool
    morphology_reflection_saved: bool
    no_mislabeling_passed: bool
    required_rollout_names: list[str]
    passed_rollout_names: list[str]
    skipped_rollout_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.archive_count < 0:
            raise SchemaValidationError("P4_2AcceptanceReport.archive_count must be non-negative")
        if self.rollout_result_count < 0:
            raise SchemaValidationError("P4_2AcceptanceReport.rollout_result_count must be non-negative")
        for rollout_name in self.required_rollout_names:
            require_non_empty(rollout_name, "P4_2AcceptanceReport.required_rollout_names")


def run_p4_2_acceptance(
    archives: list[EpisodeArchive],
    *,
    rollout_results: list[P4_2DeterministicRolloutResult] | None = None,
) -> P4_2AcceptanceReport:
    """Evaluate P4.2 acceptance without running Isaac inside the gate."""

    rollout_results = rollout_results or []
    p2_selected_design_used = _all_archives(archives, _p2_selected_design_used)
    p3_assembly_result_used = _all_archives(archives, _p3_assembly_result_used)
    trajectory_records_saved = _all_archives(archives, _trajectory_records_saved)
    selected_contact_candidates_saved = _all_archives(archives, _selected_contact_candidates_saved)
    phase_sequence_saved = _all_archives(archives, _phase_sequence_saved)
    per_step_runtime_observations_saved = _all_archives(archives, _runtime_observations_saved)
    per_step_policy_commands_saved = _all_archives(archives, _policy_commands_saved)
    per_step_controller_commands_saved = _all_archives(archives, _controller_commands_saved)
    per_step_actuator_target_records_saved = _all_archives(archives, _actuator_target_records_saved)
    attach_events_saved = _all_archives(archives, _attach_events_saved)
    release_events_saved = _all_archives(archives, _release_events_saved)
    payload_coupling_saved = _all_archives(archives, _payload_coupling_saved)
    morphology_reflection_saved = _all_archives(archives, _morphology_reflection_saved)
    no_mislabeling_passed = _all_archives(archives, _no_mislabeling_passed)
    fast_failure_reasons = _fast_failure_reasons(
        archives=archives,
        p2_selected_design_used=p2_selected_design_used,
        p3_assembly_result_used=p3_assembly_result_used,
        trajectory_records_saved=trajectory_records_saved,
        selected_contact_candidates_saved=selected_contact_candidates_saved,
        phase_sequence_saved=phase_sequence_saved,
        per_step_runtime_observations_saved=per_step_runtime_observations_saved,
        per_step_policy_commands_saved=per_step_policy_commands_saved,
        per_step_controller_commands_saved=per_step_controller_commands_saved,
        per_step_actuator_target_records_saved=per_step_actuator_target_records_saved,
        attach_events_saved=attach_events_saved,
        release_events_saved=release_events_saved,
        payload_coupling_saved=payload_coupling_saved,
        morphology_reflection_saved=morphology_reflection_saved,
        no_mislabeling_passed=no_mislabeling_passed,
    )
    fast_gate_passed = not fast_failure_reasons
    real_isaac_rollout_passed = _real_isaac_rollout_passed(rollout_results)
    rollout_failure_reasons = _rollout_failure_reasons(rollout_results)
    completion_passed = fast_gate_passed and real_isaac_rollout_passed
    failure_reasons = list(fast_failure_reasons)
    if fast_gate_passed and not real_isaac_rollout_passed:
        failure_reasons.append("P4.2 real Isaac rollout gate has not passed")
    failure_reasons.extend(rollout_failure_reasons)
    passed_rollout_names = sorted(
        result.rollout_name
        for result in rollout_results
        if result.passed and result.attempted and result.isaac_backed and not result.skipped
    )
    skipped_rollout_names = sorted(result.rollout_name for result in rollout_results if result.skipped)
    return P4_2AcceptanceReport(
        fast_gate_passed=fast_gate_passed,
        real_isaac_rollout_passed=real_isaac_rollout_passed,
        completion_passed=completion_passed,
        archive_count=len(archives),
        rollout_result_count=len(rollout_results),
        p2_selected_design_used=p2_selected_design_used,
        p3_assembly_result_used=p3_assembly_result_used,
        trajectory_records_saved=trajectory_records_saved,
        selected_contact_candidates_saved=selected_contact_candidates_saved,
        phase_sequence_saved=phase_sequence_saved,
        per_step_runtime_observations_saved=per_step_runtime_observations_saved,
        per_step_policy_commands_saved=per_step_policy_commands_saved,
        per_step_controller_commands_saved=per_step_controller_commands_saved,
        per_step_actuator_target_records_saved=per_step_actuator_target_records_saved,
        attach_events_saved=attach_events_saved,
        release_events_saved=release_events_saved,
        payload_coupling_saved=payload_coupling_saved,
        morphology_reflection_saved=morphology_reflection_saved,
        no_mislabeling_passed=no_mislabeling_passed,
        required_rollout_names=list(P4_2_REQUIRED_REAL_ROLLOUTS),
        passed_rollout_names=passed_rollout_names,
        skipped_rollout_names=skipped_rollout_names,
        metrics={
            "fast_gate_passed": 1.0 if fast_gate_passed else 0.0,
            "real_isaac_rollout_passed": 1.0 if real_isaac_rollout_passed else 0.0,
            "completion_passed": 1.0 if completion_passed else 0.0,
            "archive_count": float(len(archives)),
            "rollout_result_count": float(len(rollout_results)),
        },
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


def _trajectory_records_saved(archive: EpisodeArchive) -> bool:
    return (
        bool(archive.trajectory_records)
        and archive.trajectory_records[0].derived_mode_label == "p4_2_deterministic_grasp_carry"
        and bool(archive.trajectory_records[0].knots)
    )


def _selected_contact_candidates_saved(archive: EpisodeArchive) -> bool:
    candidate_set = archive.rollout_artifacts.get("contact_candidate_set")
    selected_ids = archive.rollout_artifacts.get("selected_contact_candidate_ids")
    selected_assignments = archive.rollout_artifacts.get("selected_contact_assignments")
    return (
        isinstance(candidate_set, dict)
        and isinstance(candidate_set.get("candidates"), list)
        and bool(candidate_set.get("candidates"))
        and isinstance(selected_ids, list)
        and bool(selected_ids)
        and isinstance(selected_assignments, list)
        and bool(selected_assignments)
    )


def _phase_sequence_saved(archive: EpisodeArchive) -> bool:
    phases = archive.rollout_artifacts.get("p4_2_phase_sequence")
    return phases == P4_2_PHASE_SEQUENCE


def _runtime_observations_saved(archive: EpisodeArchive) -> bool:
    return bool(archive.runtime_observations) and all(
        bool(observation.object_states) and bool(observation.module_states)
        for observation in archive.runtime_observations
    )


def _policy_commands_saved(archive: EpisodeArchive) -> bool:
    return bool(archive.policy_commands) and len(archive.policy_commands) == len(archive.runtime_observations)


def _controller_commands_saved(archive: EpisodeArchive) -> bool:
    return bool(archive.controller_commands) and len(archive.controller_commands) == len(archive.runtime_observations)


def _actuator_target_records_saved(archive: EpisodeArchive) -> bool:
    return (
        bool(archive.actuator_target_records)
        and len(archive.actuator_target_records) == len(archive.runtime_observations)
        and all(isinstance(record, dict) for record in archive.actuator_target_records)
    )


def _attach_events_saved(archive: EpisodeArchive) -> bool:
    events = archive.rollout_artifacts.get("p4_2_attach_events")
    return isinstance(events, list) and bool(events) and archive.metrics.get("attach_event_count", 0.0) > 0.0


def _release_events_saved(archive: EpisodeArchive) -> bool:
    events = archive.rollout_artifacts.get("p4_2_release_events")
    return isinstance(events, list) and bool(events) and archive.metrics.get("release_event_count", 0.0) > 0.0


def _payload_coupling_saved(archive: EpisodeArchive) -> bool:
    before_after_delta_found = False
    for command in archive.controller_commands:
        metrics = command.controller_status.metrics
        if metrics.get("payload_coupled", 0.0) != 1.0:
            continue
        required = (
            "payload_mass_kg",
            "payload_inertia_body_ixx",
            "payload_inertia_body_iyy",
            "payload_inertia_body_izz",
            "payload_com_offset_body_x",
            "payload_com_offset_body_y",
            "payload_com_offset_body_z",
            "payload_gravity_wrench_body_fz",
            "target_wrench_body_before_payload_fz",
            "target_wrench_body_after_payload_fz",
            "achieved_wrench_body_fz",
            "allocation_residual_norm",
            "clipped",
            "clipped_target_count",
        )
        if any(key not in metrics for key in required):
            continue
        if metrics["payload_mass_kg"] <= 0.0:
            continue
        before = float(metrics["target_wrench_body_before_payload_fz"])
        after = float(metrics["target_wrench_body_after_payload_fz"])
        if abs(after - before) > 1.0e-6:
            before_after_delta_found = True
    actuator_records_ok = bool(archive.actuator_target_records) and all(
        isinstance(record.get("metrics"), dict)
        and "allocation_residual_norm" in record.get("metrics", {})
        and "missing_actuator_count" in record.get("metrics", {})
        and "unsupported_actuator_count" in record.get("metrics", {})
        and "clipped_target_count" in record.get("metrics", {})
        for record in archive.actuator_target_records
    )
    return before_after_delta_found and actuator_records_ok


def _morphology_reflection_saved(archive: EpisodeArchive) -> bool:
    artifacts = archive.rollout_artifacts
    return (
        artifacts.get("morphology_asset_reflected") is True
        and artifacts.get("module_placement_reflected") is True
        and artifacts.get("actuator_mapping_reflected") is True
        and artifacts.get("assembled_morphology_graph_id")
        and archive.metrics.get("assembled_module_count", 0.0) > 0.0
    )


def _no_mislabeling_passed(archive: EpisodeArchive) -> bool:
    artifacts = archive.rollout_artifacts
    return (
        artifacts.get("phase") == "P4.2"
        and artifacts.get("object_attach_release_only") is True
        and artifacts.get("module_attach_detach_claim") is False
        and artifacts.get("dynamic_morphology_update_claim") is False
        and artifacts.get("is_p4_full_completion") is False
        and artifacts.get("p4_3_learning_bootstrap") is False
        and artifacts.get("learning_claim") is False
        and artifacts.get("learned_policy_success_claim") is False
        and artifacts.get("high_fidelity_natural_grasp_success_claim") is False
        and artifacts.get("true_fixed_joint_dynamics_success_claim") is False
        and artifacts.get("checkpoint_claim") is False
        and artifacts.get("reward_curve_training_claim") is False
        and archive.metrics.get("p4_full_completion", 0.0) == 0.0
        and archive.metrics.get("p4_3_learning_bootstrap", 0.0) == 0.0
        and archive.metrics.get("learned_policy_success_claim", 0.0) == 0.0
        and archive.metrics.get("high_fidelity_natural_grasp_success_claim", 0.0) == 0.0
        and archive.metrics.get("true_fixed_joint_dynamics_success_claim", 0.0) == 0.0
        and archive.metrics.get("checkpoint_claim", 0.0) == 0.0
        and archive.metrics.get("reward_curve_training_claim", 0.0) == 0.0
        and archive.metrics.get("module_attach_detach_claim", 0.0) == 0.0
        and archive.metrics.get("dynamic_morphology_update_claim", 0.0) == 0.0
    )


def _real_isaac_rollout_passed(rollout_results: list[P4_2DeterministicRolloutResult]) -> bool:
    results_by_name = {result.rollout_name: result for result in rollout_results}
    for rollout_name in P4_2_REQUIRED_REAL_ROLLOUTS:
        result = results_by_name.get(rollout_name)
        if result is None:
            return False
        if not (
            result.attempted
            and result.passed
            and result.isaac_backed
            and not result.skipped
            and result.uses_p2_selected_design
            and result.uses_p3_assembled_morphology
            and result.morphology_asset_reflected
            and result.module_placement_reflected
            and result.actuator_mapping_reflected
            and result.final_phase == P4_2RolloutPhase.SUCCESS
            and bool(result.attach_events)
            and bool(result.release_events)
            and bool(result.runtime_observations)
            and bool(result.policy_commands)
            and bool(result.controller_commands)
            and bool(result.actuator_target_records)
        ):
            return False
    return True


def _fast_failure_reasons(
    *,
    archives: list[EpisodeArchive],
    p2_selected_design_used: bool,
    p3_assembly_result_used: bool,
    trajectory_records_saved: bool,
    selected_contact_candidates_saved: bool,
    phase_sequence_saved: bool,
    per_step_runtime_observations_saved: bool,
    per_step_policy_commands_saved: bool,
    per_step_controller_commands_saved: bool,
    per_step_actuator_target_records_saved: bool,
    attach_events_saved: bool,
    release_events_saved: bool,
    payload_coupling_saved: bool,
    morphology_reflection_saved: bool,
    no_mislabeling_passed: bool,
) -> list[str]:
    reasons: list[str] = []
    if not archives:
        reasons.append("P4.2 produced no archives")
    if not p2_selected_design_used:
        reasons.append("P4.2 archives do not use P2 selected DesignOutput")
    if not p3_assembly_result_used:
        reasons.append("P4.2 archives do not use P3 assembled morphology")
    if not trajectory_records_saved:
        reasons.append("P4.2 archives are missing deterministic trajectory records")
    if not selected_contact_candidates_saved:
        reasons.append("P4.2 archives are missing selected contact candidates")
    if not phase_sequence_saved:
        reasons.append("P4.2 archives are missing the deterministic rollout phase sequence")
    if not per_step_runtime_observations_saved:
        reasons.append("P4.2 archives are missing per-step RuntimeObservation records")
    if not per_step_policy_commands_saved:
        reasons.append("P4.2 archives are missing per-step PolicyCommand records")
    if not per_step_controller_commands_saved:
        reasons.append("P4.2 archives are missing per-step ControllerCommand records")
    if not per_step_actuator_target_records_saved:
        reasons.append("P4.2 archives are missing per-step actuator target records")
    if not attach_events_saved:
        reasons.append("P4.2 archives are missing gated object attach events")
    if not release_events_saved:
        reasons.append("P4.2 archives are missing intended object release events")
    if not payload_coupling_saved:
        reasons.append("P4.2 archives do not prove payload-coupled controller computation")
    if not morphology_reflection_saved:
        reasons.append("P4.2 archives do not prove graph-specific asset/module/mapping reflection")
    if not no_mislabeling_passed:
        reasons.append("P4.2 no-mislabeling checks failed")
    return reasons


def _rollout_failure_reasons(rollout_results: list[P4_2DeterministicRolloutResult]) -> list[str]:
    reasons: list[str] = []
    results_by_name = {result.rollout_name: result for result in rollout_results}
    for rollout_name in P4_2_REQUIRED_REAL_ROLLOUTS:
        result = results_by_name.get(rollout_name)
        if result is None:
            reasons.append(f"P4.2 missing real Isaac rollout result: {rollout_name}")
            continue
        if result.skipped:
            reasons.append(f"P4.2 real Isaac rollout skipped: {rollout_name}")
        elif not result.attempted:
            reasons.append(f"P4.2 real Isaac rollout not attempted: {rollout_name}")
        elif not result.isaac_backed:
            reasons.append(f"P4.2 rollout was not Isaac-backed: {rollout_name}")
        elif not result.passed:
            reasons.append(f"P4.2 real Isaac rollout failed: {rollout_name}")
        elif not (result.uses_p2_selected_design and result.uses_p3_assembled_morphology):
            reasons.append(f"P4.2 real Isaac rollout did not use P2/P3 case: {rollout_name}")
        elif not (
            result.morphology_asset_reflected
            and result.module_placement_reflected
            and result.actuator_mapping_reflected
        ):
            reasons.append(f"P4.2 real Isaac rollout did not reflect graph-specific morphology: {rollout_name}")
        elif not result.attach_events:
            reasons.append(f"P4.2 real Isaac rollout has no gated attach event: {rollout_name}")
        elif not result.release_events:
            reasons.append(f"P4.2 real Isaac rollout has no release event: {rollout_name}")
    return reasons
