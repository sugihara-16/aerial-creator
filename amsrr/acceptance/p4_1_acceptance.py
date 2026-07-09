from __future__ import annotations

from dataclasses import dataclass, field

from amsrr.logging import EpisodeArchive
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.simulation.p4_1_backend_smoke import (
    P4_1_REQUIRED_REAL_SMOKES,
    P4_1BackendSmokeResult,
    evaluate_runtime_observation_joint_state,
)


@dataclass
class P4_1AcceptanceReport(SchemaBase):
    fast_gate_passed: bool
    real_isaac_smoke_passed: bool
    completion_passed: bool
    archive_count: int
    smoke_result_count: int
    p2_selected_design_used: bool
    p3_assembly_result_used: bool
    not_fixed_two_module_only: bool
    per_step_runtime_observations_saved: bool
    per_step_controller_commands_saved: bool
    per_step_actuator_target_records_saved: bool
    object_pose_history_saved: bool
    joint_state_preserved: bool
    full_scene_spawned: bool
    no_mislabeling_passed: bool
    required_smoke_names: list[str]
    passed_smoke_names: list[str]
    skipped_smoke_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.archive_count < 0:
            raise SchemaValidationError("P4_1AcceptanceReport.archive_count must be non-negative")
        if self.smoke_result_count < 0:
            raise SchemaValidationError("P4_1AcceptanceReport.smoke_result_count must be non-negative")
        for smoke_name in self.required_smoke_names:
            require_non_empty(smoke_name, "P4_1AcceptanceReport.required_smoke_names")


def run_p4_1_acceptance(
    archives: list[EpisodeArchive],
    *,
    smoke_results: list[P4_1BackendSmokeResult] | None = None,
) -> P4_1AcceptanceReport:
    """Evaluate P4.1 without running Isaac inside the acceptance function."""

    smoke_results = smoke_results or []
    p2_selected_design_used = _all_archives(archives, _p2_selected_design_used)
    p3_assembly_result_used = _all_archives(archives, _p3_assembly_result_used)
    not_fixed_two_module_only = _all_archives(archives, lambda archive: archive.metrics.get("fixed_two_module_only") == 0.0)
    per_step_runtime_observations_saved = _all_archives(archives, _runtime_observations_saved)
    per_step_controller_commands_saved = _all_archives(archives, _controller_commands_saved)
    per_step_actuator_target_records_saved = _all_archives(archives, _actuator_target_records_saved)
    object_pose_history_saved = _all_archives(archives, _object_pose_history_saved)
    joint_state_preserved = _all_archives(archives, _joint_state_preserved)
    full_scene_spawned = _all_archives(archives, _full_scene_spawned)
    no_mislabeling_passed = _all_archives(archives, _no_mislabeling_passed)
    fast_failure_reasons = _fast_failure_reasons(
        archives=archives,
        p2_selected_design_used=p2_selected_design_used,
        p3_assembly_result_used=p3_assembly_result_used,
        not_fixed_two_module_only=not_fixed_two_module_only,
        per_step_runtime_observations_saved=per_step_runtime_observations_saved,
        per_step_controller_commands_saved=per_step_controller_commands_saved,
        per_step_actuator_target_records_saved=per_step_actuator_target_records_saved,
        object_pose_history_saved=object_pose_history_saved,
        joint_state_preserved=joint_state_preserved,
        full_scene_spawned=full_scene_spawned,
        no_mislabeling_passed=no_mislabeling_passed,
    )
    fast_gate_passed = not fast_failure_reasons
    real_isaac_smoke_passed = _real_isaac_smoke_passed(smoke_results)
    smoke_failure_reasons = _smoke_failure_reasons(smoke_results)
    completion_passed = fast_gate_passed and real_isaac_smoke_passed
    failure_reasons = list(fast_failure_reasons)
    if fast_gate_passed and not real_isaac_smoke_passed:
        failure_reasons.append("P4.1 real Isaac smoke gate has not passed")
    failure_reasons.extend(smoke_failure_reasons)
    passed_smoke_names = sorted(
        result.smoke_name
        for result in smoke_results
        if result.passed and result.attempted and result.isaac_backed and not result.skipped
    )
    skipped_smoke_names = sorted(result.smoke_name for result in smoke_results if result.skipped)
    return P4_1AcceptanceReport(
        fast_gate_passed=fast_gate_passed,
        real_isaac_smoke_passed=real_isaac_smoke_passed,
        completion_passed=completion_passed,
        archive_count=len(archives),
        smoke_result_count=len(smoke_results),
        p2_selected_design_used=p2_selected_design_used,
        p3_assembly_result_used=p3_assembly_result_used,
        not_fixed_two_module_only=not_fixed_two_module_only,
        per_step_runtime_observations_saved=per_step_runtime_observations_saved,
        per_step_controller_commands_saved=per_step_controller_commands_saved,
        per_step_actuator_target_records_saved=per_step_actuator_target_records_saved,
        object_pose_history_saved=object_pose_history_saved,
        joint_state_preserved=joint_state_preserved,
        full_scene_spawned=full_scene_spawned,
        no_mislabeling_passed=no_mislabeling_passed,
        required_smoke_names=list(P4_1_REQUIRED_REAL_SMOKES),
        passed_smoke_names=passed_smoke_names,
        skipped_smoke_names=skipped_smoke_names,
        metrics={
            "fast_gate_passed": 1.0 if fast_gate_passed else 0.0,
            "real_isaac_smoke_passed": 1.0 if real_isaac_smoke_passed else 0.0,
            "completion_passed": 1.0 if completion_passed else 0.0,
            "archive_count": float(len(archives)),
            "smoke_result_count": float(len(smoke_results)),
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


def _runtime_observations_saved(archive: EpisodeArchive) -> bool:
    return bool(archive.runtime_observations) and all(
        bool(observation.object_states) and bool(observation.module_states)
        for observation in archive.runtime_observations
    )


def _controller_commands_saved(archive: EpisodeArchive) -> bool:
    return bool(archive.controller_commands) and len(archive.controller_commands) == len(archive.runtime_observations)


def _actuator_target_records_saved(archive: EpisodeArchive) -> bool:
    return (
        bool(archive.actuator_target_records)
        and len(archive.actuator_target_records) == len(archive.runtime_observations)
        and all(isinstance(record, dict) for record in archive.actuator_target_records)
    )


def _object_pose_history_saved(archive: EpisodeArchive) -> bool:
    history = archive.rollout_artifacts.get("p4_1_object_pose_history")
    return (
        isinstance(history, list)
        and len(history) == len(archive.runtime_observations)
        and bool(history)
        and all(isinstance(pose, list) and len(pose) == 7 for pose in history)
    )


def _joint_state_preserved(archive: EpisodeArchive) -> bool:
    metrics = evaluate_runtime_observation_joint_state(
        archive.runtime_observations,
        articulated_morphology=archive.metrics.get("p4_1_articulated_morphology", 0.0) == 1.0,
        articulated_model_update_metrics={
            "max_model_rotor_origin_change_m": archive.metrics.get("p4_1_max_model_rotor_origin_change_m", 0.0),
            "max_model_allocation_change": archive.metrics.get("p4_1_max_model_allocation_change", 0.0),
        },
    )
    return metrics.passed


def _full_scene_spawned(archive: EpisodeArchive) -> bool:
    artifacts = archive.rollout_artifacts
    return (
        archive.metrics.get("p4_1_full_scene_spawned", 0.0) == 1.0
        and archive.metrics.get("p4_1_robot_spawned", 0.0) == 1.0
        and archive.metrics.get("p4_1_object_spawned", 0.0) == 1.0
        and archive.metrics.get("p4_1_floor_spawned", 0.0) == 1.0
        and artifacts.get("phase") == "P4.1"
    )


def _no_mislabeling_passed(archive: EpisodeArchive) -> bool:
    artifacts = archive.rollout_artifacts
    return (
        artifacts.get("phase") == "P4.1"
        and artifacts.get("is_p4_full_completion") is False
        and artifacts.get("p4_2_rollout_claim") is False
        and artifacts.get("object_grasp_carry_claim") is False
        and artifacts.get("learning_claim") is False
        and artifacts.get("physical_success_claim") is False
        and archive.metrics.get("p4_full_completion", 0.0) == 0.0
        and archive.metrics.get("object_grasp_carry_success_claim", 0.0) == 0.0
        and archive.metrics.get("learned_policy_claim", 0.0) == 0.0
        and archive.metrics.get("p4_2_rollout_claim", 0.0) == 0.0
    )


def _real_isaac_smoke_passed(smoke_results: list[P4_1BackendSmokeResult]) -> bool:
    results_by_name = {result.smoke_name: result for result in smoke_results}
    for smoke_name in P4_1_REQUIRED_REAL_SMOKES:
        result = results_by_name.get(smoke_name)
        if result is None:
            return False
        if not (
            result.attempted
            and result.passed
            and result.isaac_backed
            and not result.skipped
            and result.uses_p2_selected_design
            and result.uses_p3_assembled_morphology
            and result.full_scene_spawned
            and result.robot_spawned
            and result.object_spawned
            and result.floor_spawned
        ):
            return False
        if result.joint_state_metrics is None or not result.joint_state_metrics.passed:
            return False
    return True


def _fast_failure_reasons(
    *,
    archives: list[EpisodeArchive],
    p2_selected_design_used: bool,
    p3_assembly_result_used: bool,
    not_fixed_two_module_only: bool,
    per_step_runtime_observations_saved: bool,
    per_step_controller_commands_saved: bool,
    per_step_actuator_target_records_saved: bool,
    object_pose_history_saved: bool,
    joint_state_preserved: bool,
    full_scene_spawned: bool,
    no_mislabeling_passed: bool,
) -> list[str]:
    reasons: list[str] = []
    if not archives:
        reasons.append("P4.1 produced no archives")
    if not p2_selected_design_used:
        reasons.append("P4.1 archives do not use P2 selected DesignOutput")
    if not p3_assembly_result_used:
        reasons.append("P4.1 archives do not use P3 assembled morphology")
    if not not_fixed_two_module_only:
        reasons.append("P4.1 archive is only a fixed 2-module case")
    if not per_step_runtime_observations_saved:
        reasons.append("P4.1 archives are missing per-step RuntimeObservation records")
    if not per_step_controller_commands_saved:
        reasons.append("P4.1 archives are missing per-step ControllerCommand records")
    if not per_step_actuator_target_records_saved:
        reasons.append("P4.1 archives are missing per-step actuator target records")
    if not object_pose_history_saved:
        reasons.append("P4.1 archives are missing object pose history")
    if not joint_state_preserved:
        reasons.append("P4.1 RuntimeObservation joint-state preservation failed")
    if not full_scene_spawned:
        reasons.append("P4.1 full-scene robot/object/floor spawn evidence is missing")
    if not no_mislabeling_passed:
        reasons.append("P4.1 no-mislabeling checks failed")
    return reasons


def _smoke_failure_reasons(smoke_results: list[P4_1BackendSmokeResult]) -> list[str]:
    reasons: list[str] = []
    results_by_name = {result.smoke_name: result for result in smoke_results}
    for smoke_name in P4_1_REQUIRED_REAL_SMOKES:
        result = results_by_name.get(smoke_name)
        if result is None:
            reasons.append(f"P4.1 missing real Isaac smoke result: {smoke_name}")
            continue
        if result.skipped:
            reasons.append(f"P4.1 real Isaac smoke skipped: {smoke_name}")
        elif not result.attempted:
            reasons.append(f"P4.1 real Isaac smoke not attempted: {smoke_name}")
        elif not result.isaac_backed:
            reasons.append(f"P4.1 smoke was not Isaac-backed: {smoke_name}")
        elif not result.passed:
            reasons.append(f"P4.1 real Isaac smoke failed: {smoke_name}")
        elif not (result.uses_p2_selected_design and result.uses_p3_assembled_morphology):
            reasons.append(f"P4.1 real Isaac smoke did not use P2/P3 case: {smoke_name}")
    return reasons
