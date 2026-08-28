from __future__ import annotations

"""Hash-bound R1 calibration with closed-loop-safe physical timing."""

import json
from pathlib import Path
from typing import Any, Sequence

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_clearance_diagnostic import (
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_complete_task_audit import (
    ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION,
    ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION,
    ORDER9_R1_COMPLETE_TASK_PHASES,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
)
from amsrr.training.order9_r1_nominal_calibration import (
    Order9R1NominalReplayResult,
)
from amsrr.training.order9_r1_nominal_calibration_v3 import (
    run_order9_r1_nominal_v3_isaac_cases,
    write_order9_r1_nominal_v3_case_contract,
)
from amsrr.training.order9_r1_nominal_retime import (
    ORDER9_R1_NOMINAL_RETIME_VERSION,
)
from amsrr.training.order9_r1_nominal_retreat import (
    ORDER9_R1_NOMINAL_RETREAT_VERSION,
    validate_order9_r1_short_retreat,
)
from amsrr.training.order9_r1_safe_timing import (
    ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT,
    ORDER9_R1_SAFE_TIMING_BASIS,
    ORDER9_R1_SAFE_TIMING_VERSION,
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_V6_PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v6"
ORDER9_R1_NOMINAL_V6_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v6"
)
ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT = (
    "deterministic_teacher_closed_loop_safe_morphology_timing_early_tilt_"
    "short_retreat_alternative_surface_joint_reserve_30mm_nominal_qpid_qp_"
    "local_servo_isaac_v6"
)
ORDER9_R1_NOMINAL_V6_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v6.yaml"
)
ORDER9_R1_NOMINAL_V6_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V6_APPROVAL.json"
)
ORDER9_R1_NOMINAL_V6_CASE_CONTRACT_VERSION = (
    "order9_r1_nominal_calibration_case_contract_v6"
)
ORDER9_R1_NOMINAL_V6_CASE_RESULT_VERSION = (
    "order9_r1_nominal_calibration_case_result_v6"
)
_EXPECTED_PROTOCOL_SHA256 = (
    "271c2f6791761ce608dc0160acafb0d7daa2b1c538e0da538f18bda70c6d5614"
)
_EXPECTED_APPROVAL_SHA256 = (
    "8e42cf53209982afc83315741bbe1245f7085385bd9575306dac666987d4265c"
)
_BINDING_KEYS = (
    "base_v5_protocol",
    "base_v5_approval",
    "base_v5_result_ledger",
    "base_numeric_protocol",
    "base_numeric_approval",
    "protected_c3_curriculum",
    "protected_c3_checkpoint",
    "protected_c3_rollout",
    "protected_c3_release_ledger",
    "r1_early_tilt_pipeline_implementation",
    "r1_teacher_overrides",
    "r1_retime_implementation",
    "r1_nominal_retreat_implementation",
    "r1_complete_task_semantic_audit_implementation",
    "r1_safe_timing_implementation",
    "r1_isaac_materialization_implementation",
    "r1_complete_task_materialization_v6_implementation",
)
# The digest remains part of the immutable v6 record, while this live
# implementation path is allowed to contain its v9 successor.
_SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS = frozenset(
    {"r1_early_tilt_pipeline_implementation"}
)
_REQUIRED_APPROVAL_SCOPE = {
    "replace_unsafe_universal_retime",
    "module_count_conditioned_physical_timing",
    "closed_loop_isaac_evidence_for_each_timing_profile",
    "short_task_offset_retreat_after_release",
    "grasp_path_preserved",
    "retreat_reauthored_and_isaac_validated",
    "complete_eight_phase_semantic_audit_before_isaac",
    "retreat_parameter_effect_test",
    "explicit_materializer_selection",
    "same_controller_free_screen_before_isaac",
    "same_contact_assignments",
    "joint_rate_recheck_after_retime",
    "nominal_qpid_qp_local_servo_standard_path",
    "pi_l_retained_but_not_applied",
    "completion_within_ten_hours",
    "reuse_four_level_numeric_ladder",
    "selection_on_22_train_buckets",
    "single_confirmation_on_14_validation_buckets",
    "two_isaac_replays_per_screen_accepted_candidate",
    "stop_after_first_failed_level",
    "no_learning_by_protocol_alone",
}


def load_order9_r1_nominal_calibration_v6_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_V6_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_V6_APPROVAL_RELATIVE
    if hash_file(protocol_path) != _EXPECTED_PROTOCOL_SHA256:
        raise SchemaValidationError("R1 nominal v6 protocol bytes changed")
    if hash_file(approval_path) != _EXPECTED_APPROVAL_SHA256:
        raise SchemaValidationError("R1 nominal v6 approval bytes changed")
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    expected_profiles = {
        int(module_count): dict(profile)
        for module_count, profile in (
            ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT.items()
        )
    }
    actual_profiles = {
        int(module_count): dict(profile)
        for module_count, profile in dict(
            protocol.get("phase_time_scales_by_module_count", {})
        ).items()
    }
    if (
        protocol.get("protocol_version") != ORDER9_R1_NOMINAL_V6_PROTOCOL_VERSION
        or protocol.get("calibration_gate_id")
        != "r1-nominal-closed-loop-safe-timing-gate-v6"
        or protocol.get("status") != "approved"
        or protocol.get("execution_contract") != ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT
        or protocol.get("safe_timing_version") != ORDER9_R1_SAFE_TIMING_VERSION
        or protocol.get("timing_selection_basis") != ORDER9_R1_SAFE_TIMING_BASIS
        or actual_profiles != expected_profiles
        or protocol.get("grasp_geometric_path_changed") is not False
        or protocol.get("post_release_retreat_path_changed") is not True
        or float(protocol.get("retreat_offset_m", -1.0)) != 0.10
        or protocol.get("grasp_endpoint_configuration_changed") is not False
        or protocol.get("final_endpoint_configuration_changed") is not True
        or protocol.get("contact_assignments_changed") is not False
        or protocol.get("retreat_isaac_validated") is not True
        or protocol.get("complete_task_semantic_audit_required") is not True
        or protocol.get("complete_task_materializer_selection")
        != "explicit_callable_and_identity_v1"
        or protocol.get("complete_task_materializer_monkey_patch_forbidden") is not True
        or protocol.get("joint_rate_limit_rechecked") is not True
        or protocol.get("isaac_required_after_fast_screen") is not True
        or protocol.get("pi_l_actor_command_applied") is not False
        or float(protocol.get("maximum_wall_time_s", -1.0)) != 36000.0
    ):
        raise SchemaValidationError("R1 nominal v6 protocol contract is invalid")
    if (
        approval.get("record_version") != ORDER9_R1_NOMINAL_V6_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("calibration_gate_id") != protocol["calibration_gate_id"]
        or set(approval.get("approval_scope", ())) != _REQUIRED_APPROVAL_SCOPE
        or approval.get("approved_protocol", {}).get("sha256")
        != _EXPECTED_PROTOCOL_SHA256
    ):
        raise SchemaValidationError("R1 nominal v6 approval record is invalid")
    for key in _BINDING_KEYS:
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v6 binding missing: {key}")
        path = repository / str(binding.get("path", ""))
        if not path.is_file() or (
            key not in _SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS
            and hash_file(path) != binding.get("sha256")
        ):
            raise SchemaValidationError(f"R1 nominal v6 binding changed: {key}")
    _validate_profile_evidence(protocol, repository)
    return protocol, approval


def write_order9_r1_nominal_v6_case_contract(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_v6_contract(repository)
    root = materialized.manifest_path.parent
    base_contract = write_order9_r1_nominal_v3_case_contract(
        materialized,
        repository_root=repository,
    )
    fast_screen_path = root / "fast_screen.json"
    fast_screen = json.loads(fast_screen_path.read_text(encoding="utf-8"))
    nominal_set_path = repository / materialized.manifest.nominal_set.path
    nominal_set = json.loads(nominal_set_path.read_text(encoding="utf-8"))
    admission = nominal_set.get("metadata", {}).get("nominal_retime_admission")
    audit_binding = nominal_set.get("metadata", {}).get("complete_task_semantic_audit")
    parameter_binding = nominal_set.get("metadata", {}).get(
        "complete_task_parameter_effect_audit"
    )
    if not isinstance(audit_binding, dict) or not isinstance(parameter_binding, dict):
        raise SchemaValidationError("R1 nominal v6 complete-task audits are missing")
    audit_path = root / str(audit_binding.get("path", ""))
    parameter_path = root / str(parameter_binding.get("path", ""))
    if (
        not audit_path.is_file()
        or hash_file(audit_path) != audit_binding.get("sha256")
        or not parameter_path.is_file()
        or hash_file(parameter_path) != parameter_binding.get("sha256")
    ):
        raise SchemaValidationError("R1 nominal v6 complete-task audit bytes changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    parameter_audit = json.loads(parameter_path.read_text(encoding="utf-8"))
    artifact_path = repository / materialized.manifest.nominal_artifact.path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    phase_trajectories = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_path.parent / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }
    validate_order9_r1_short_retreat(
        phase_trajectories,
        retreat_offset_m=0.10,
    )
    expected_scales = order9_r1_safe_phase_time_scales(
        materialized.manifest.module_count
    )
    if (
        fast_screen.get("accepted") is not True
        or fast_screen.get("isaac_invoked") is not False
        or fast_screen.get("controller_layers_invoked") is not False
        or not isinstance(admission, dict)
        or admission.get("admission_version") != ORDER9_R1_NOMINAL_RETIME_VERSION
        or admission.get("phase_time_scales") != expected_scales
        or admission.get("geometric_path_unchanged") is not True
        or admission.get("endpoint_configuration_unchanged") is not True
        or admission.get("contact_assignments_unchanged") is not True
        or admission.get("collision_admission_inherited_from_identical_geometry")
        is not True
        or admission.get("accepted") is not True
        or float(admission.get("minimum_joint_rate_margin_rad_s", -1.0)) < 0.0
        or audit_binding.get("audit_version") != ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION
        or audit_binding.get("status") != "accepted"
        or audit.get("audit_version") != ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION
        or audit.get("status") != "accepted"
        or tuple(audit.get("checked_phases", ())) != ORDER9_R1_COMPLETE_TASK_PHASES
        or audit.get("materializer_id") != ORDER9_R1_NOMINAL_RETREAT_VERSION
        or audit.get("isaac_invoked") is not False
        or audit.get("controller_layers_invoked") is not False
        or audit.get("trajectory_optimization_invoked") is not False
        or parameter_binding.get("audit_version")
        != ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION
        or parameter_binding.get("status") != "accepted"
        or parameter_audit.get("audit_version")
        != ORDER9_R1_COMPLETE_TASK_PARAMETER_EFFECT_VERSION
        or parameter_audit.get("status") != "accepted"
        or parameter_audit.get("materializer_id") != ORDER9_R1_NOMINAL_RETREAT_VERSION
        or parameter_audit.get("retreat_offsets_m") != [0.05, 0.10]
        or parameter_audit.get("isaac_invoked") is not False
        or parameter_audit.get("controller_layers_invoked") is not False
        or parameter_audit.get("trajectory_optimization_invoked") is not False
    ):
        raise SchemaValidationError("R1 nominal v6 case lacks safe timing admission")
    payload = {
        "contract_version": ORDER9_R1_NOMINAL_V6_CASE_CONTRACT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT,
        "safe_timing_version": ORDER9_R1_SAFE_TIMING_VERSION,
        "nominal_retreat_version": ORDER9_R1_NOMINAL_RETREAT_VERSION,
        "timing_selection_basis": ORDER9_R1_SAFE_TIMING_BASIS,
        "module_count": materialized.manifest.module_count,
        "phase_time_scales": expected_scales,
        "grasp_geometric_path_changed": False,
        "post_release_retreat_path_changed": True,
        "retreat_offset_m": 0.10,
        "grasp_endpoint_configuration_changed": False,
        "final_endpoint_configuration_changed": True,
        "contact_assignments_changed": False,
        "retreat_isaac_validated": True,
        "complete_task_semantic_audit_required": True,
        "complete_task_materializer_selection": "explicit_callable_and_identity_v1",
        "complete_task_materializer_monkey_patch_forbidden": True,
        "joint_rate_limit_rechecked": True,
        "controller_free_screen_completed_before_isaac": True,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "nominal_preload_applied": True,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "isaac_required": True,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v6_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V6_PROTOCOL_RELATIVE
            ),
            "v6_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V6_APPROVAL_RELATIVE
            ),
            "base_v3_contract": _binding(base_contract),
            "case_manifest": _binding(materialized.manifest_path),
            "fast_screen": _binding(fast_screen_path),
            "nominal_set": _binding(nominal_set_path),
            "complete_task_semantic_audit": _binding(audit_path),
            "complete_task_parameter_effect_audit": _binding(parameter_path),
            "safe_timing_implementation": _binding(
                repository / protocol["r1_safe_timing_implementation"]["path"]
            ),
            "nominal_retreat_implementation": _binding(
                repository / protocol["r1_nominal_retreat_implementation"]["path"]
            ),
        },
        "retime_admission": admission,
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_contract_v6.json",
    )


def run_order9_r1_nominal_v6_isaac_cases(
    cases: Sequence[MaterializedOrder9R1IsaacCase],
    *,
    repository_root: str | Path,
    python_executable: str | Path = (
        "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
    ),
    rollout_steps: int = 15000,
    maximum_parallel_process_count: int = 4,
    persistent_morphology_coalescing: bool = True,
) -> dict[str, tuple[Order9R1NominalReplayResult, ...]]:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v6_contract(repository)
    for case in cases:
        write_order9_r1_nominal_v6_case_contract(case, repository_root=repository)
    results = run_order9_r1_nominal_v3_isaac_cases(
        cases,
        repository_root=repository,
        python_executable=python_executable,
        rollout_steps=rollout_steps,
        maximum_parallel_process_count=maximum_parallel_process_count,
        persistent_morphology_coalescing=persistent_morphology_coalescing,
    )
    for case in cases:
        write_order9_r1_nominal_v6_case_result(
            case,
            results[case.manifest.candidate_id],
            repository_root=repository,
        )
    return results


def write_order9_r1_nominal_v6_case_result(
    materialized: MaterializedOrder9R1IsaacCase,
    results: Sequence[Order9R1NominalReplayResult],
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v6_contract(repository)
    root = materialized.manifest_path.parent
    payload = {
        "result_version": ORDER9_R1_NOMINAL_V6_CASE_RESULT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT,
        "safe_timing_version": ORDER9_R1_SAFE_TIMING_VERSION,
        "timing_selection_basis": ORDER9_R1_SAFE_TIMING_BASIS,
        "module_count": materialized.manifest.module_count,
        "phase_time_scales": order9_r1_safe_phase_time_scales(
            materialized.manifest.module_count
        ),
        "status": "accepted" if all(value.success for value in results) else "rejected",
        "episode_count": len(results),
        "success_count": sum(value.success for value in results),
        "safety_failure_count": sum(value.safety_failure for value in results),
        "fallback_count": sum(value.fallback_used for value in results),
        "pi_l_actor_command_applied": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v6_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V6_PROTOCOL_RELATIVE
            ),
            "v6_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V6_APPROVAL_RELATIVE
            ),
            "v6_case_contract": _binding(root / "nominal_contract_v6.json"),
            "base_v3_result": _binding(root / "nominal_result_v3.json"),
            "raw_rollout": _binding(root / "isaac/evaluation_rollout.pt"),
            "episodes": _binding(root / "isaac/evaluation_episodes.jsonl"),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_result_v6.json",
    )


def _validate_profile_evidence(protocol: dict[str, Any], repository: Path) -> None:
    evidence = protocol.get("profile_isaac_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise SchemaValidationError("R1 nominal v6 profile evidence is missing")
    covered: set[int] = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            raise SchemaValidationError("R1 nominal v6 profile evidence is invalid")
        module_count = int(entry.get("module_count", -1))
        result_binding = entry.get("result")
        episodes_binding = entry.get("episodes")
        if (
            module_count not in ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT
            or entry.get("phase_time_scales")
            != order9_r1_safe_phase_time_scales(module_count)
            or not isinstance(result_binding, dict)
            or not isinstance(episodes_binding, dict)
        ):
            raise SchemaValidationError("R1 nominal v6 profile evidence differs")
        result_path = repository / str(result_binding.get("path", ""))
        episodes_path = repository / str(episodes_binding.get("path", ""))
        if (
            not result_path.is_file()
            or hash_file(result_path) != result_binding.get("sha256")
            or not episodes_path.is_file()
            or hash_file(episodes_path) != episodes_binding.get("sha256")
        ):
            raise SchemaValidationError("R1 nominal v6 profile evidence bytes changed")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            result.get("status") != "accepted"
            or int(result.get("episode_count", -1)) != 2
            or int(result.get("success_count", -1)) != 2
            or int(result.get("safety_failure_count", -1)) != 0
            or int(result.get("fallback_count", -1)) != 0
            or len(episodes) != 2
            or not all(value.get("task_success") is True for value in episodes)
            or any(value.get("safety_failure") is True for value in episodes)
            or any(
                int(value.get("fallback_decision_count", -1)) != 0 for value in episodes
            )
        ):
            raise SchemaValidationError("R1 nominal v6 profile Isaac gate failed")
        covered.add(module_count)
    if covered != set(ORDER9_R1_SAFE_PHASE_TIME_SCALES_BY_MODULE_COUNT):
        raise SchemaValidationError("R1 nominal v6 profile evidence coverage differs")


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": hash_file(source)}


__all__ = [
    "ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT",
    "load_order9_r1_nominal_calibration_v6_contract",
    "run_order9_r1_nominal_v6_isaac_cases",
    "write_order9_r1_nominal_v6_case_contract",
    "write_order9_r1_nominal_v6_case_result",
]
