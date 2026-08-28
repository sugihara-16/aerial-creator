from __future__ import annotations

"""Hash-bound R1 calibration with admitted identical-path retiming."""

import json
from pathlib import Path
from typing import Any, Sequence

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_clearance_diagnostic import (
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
)
from amsrr.training.order9_r1_nominal_calibration import (
    Order9R1NominalReplayResult,
)
from amsrr.training.order9_r1_nominal_calibration_v3 import (
    run_order9_r1_nominal_v3_isaac_cases,
)
from amsrr.training.order9_r1_nominal_retime import (
    ORDER9_R1_NOMINAL_RETIME_VERSION,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_V5_PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v5"
ORDER9_R1_NOMINAL_V5_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v5"
)
ORDER9_R1_NOMINAL_V5_EXECUTION_CONTRACT = (
    "deterministic_teacher_identical_path_retime_early_tilt_alternative_surface_"
    "joint_reserve_30mm_nominal_qpid_qp_local_servo_isaac_v4"
)
ORDER9_R1_NOMINAL_V5_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v5.yaml"
)
ORDER9_R1_NOMINAL_V5_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V5_APPROVAL.json"
)
ORDER9_R1_NOMINAL_V5_CASE_CONTRACT_VERSION = (
    "order9_r1_nominal_calibration_case_contract_v5"
)
ORDER9_R1_NOMINAL_V5_CASE_RESULT_VERSION = (
    "order9_r1_nominal_calibration_case_result_v5"
)
_EXPECTED_PROTOCOL_SHA256 = (
    "713c69a31017f5280e06e9a35eb3793cfbb12d1cd4f3d9a81fa756e7b213948e"
)
_EXPECTED_APPROVAL_SHA256 = (
    "0583371f4553d13a4ba9311aae480e497e44ae028253be37f29b67f05f4e2da0"
)
_PHASES = (
    "approach",
    "contact_acquisition",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
)
_EXPECTED_PHASE_TIME_SCALES = {
    "approach": 0.5,
    "contact_acquisition": 0.5,
    "lift": 0.1,
    "transport": 0.1,
    "place": 0.1,
    "release": 0.1,
    "retreat": 0.1,
    "settle": 0.1,
}
_BINDING_KEYS = (
    "base_v4_protocol",
    "base_v4_approval",
    "base_v4_result_ledger",
    "base_numeric_protocol",
    "base_numeric_approval",
    "protected_c3_curriculum",
    "protected_c3_checkpoint",
    "protected_c3_rollout",
    "protected_c3_release_ledger",
    "r1_early_tilt_pipeline_implementation",
    "r1_teacher_overrides",
    "r1_retime_implementation",
    "r1_isaac_materialization_implementation",
    "representative_retime_isaac_result",
    "representative_retime_isaac_episodes",
)
# The digest remains part of the immutable v5 record, while this live
# implementation path is allowed to contain its v9 successor.
_SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS = frozenset(
    {"r1_early_tilt_pipeline_implementation"}
)
_REQUIRED_APPROVAL_SCOPE = {
    "completion_within_ten_hours",
    "same_controller_free_screen_before_isaac",
    "identical_geometric_path_time_reparameterization",
    "same_endpoint_configuration",
    "same_contact_assignments",
    "joint_rate_recheck_after_retime",
    "collision_admission_only_for_identical_geometry",
    "representative_two_replay_isaac_admission",
    "nominal_qpid_qp_local_servo_standard_path",
    "pi_l_retained_but_not_applied",
    "reuse_four_level_numeric_ladder",
    "selection_on_22_train_buckets",
    "single_confirmation_on_14_validation_buckets",
    "two_isaac_replays_per_screen_accepted_candidate",
    "stop_after_first_failed_level",
    "no_learning_by_protocol_alone",
}


def load_order9_r1_nominal_calibration_v5_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_V5_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_V5_APPROVAL_RELATIVE
    if hash_file(protocol_path) != _EXPECTED_PROTOCOL_SHA256:
        raise SchemaValidationError("R1 nominal v5 protocol bytes changed")
    if hash_file(approval_path) != _EXPECTED_APPROVAL_SHA256:
        raise SchemaValidationError("R1 nominal v5 approval bytes changed")
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version") != ORDER9_R1_NOMINAL_V5_PROTOCOL_VERSION
        or protocol.get("calibration_gate_id")
        != "r1-nominal-identical-path-retime-gate-v5"
        or protocol.get("status") != "approved"
        or protocol.get("execution_contract") != ORDER9_R1_NOMINAL_V5_EXECUTION_CONTRACT
        or protocol.get("phase_time_scales") != _EXPECTED_PHASE_TIME_SCALES
        or protocol.get("geometric_path_changed") is not False
        or protocol.get("endpoint_configuration_changed") is not False
        or protocol.get("contact_assignments_changed") is not False
        or protocol.get("joint_rate_limit_rechecked") is not True
        or protocol.get("isaac_required_after_fast_screen") is not True
        or protocol.get("pi_l_actor_command_applied") is not False
        or int(protocol.get("representative_isaac_success_count", -1)) != 2
        or int(protocol.get("representative_isaac_safety_failure_count", -1)) != 0
        or int(protocol.get("representative_isaac_fallback_count", -1)) != 0
        or float(protocol.get("maximum_wall_time_s", -1.0)) != 36000.0
    ):
        raise SchemaValidationError("R1 nominal v5 protocol contract is invalid")
    if (
        approval.get("record_version") != ORDER9_R1_NOMINAL_V5_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("calibration_gate_id") != protocol["calibration_gate_id"]
        or set(approval.get("approval_scope", ())) != _REQUIRED_APPROVAL_SCOPE
        or approval.get("approved_protocol", {}).get("sha256")
        != _EXPECTED_PROTOCOL_SHA256
    ):
        raise SchemaValidationError("R1 nominal v5 approval record is invalid")
    for key in _BINDING_KEYS:
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v5 binding missing: {key}")
        path = repository / str(binding.get("path", ""))
        if not path.is_file() or (
            key not in _SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS
            and hash_file(path) != binding.get("sha256")
        ):
            raise SchemaValidationError(f"R1 nominal v5 binding changed: {key}")
    episodes_path = (
        repository / protocol["representative_retime_isaac_episodes"]["path"]
    )
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(episodes) != 2
        or not all(value.get("task_success") is True for value in episodes)
        or any(value.get("safety_failure") is True for value in episodes)
        or any(int(value.get("fallback_decision_count", -1)) != 0 for value in episodes)
    ):
        raise SchemaValidationError("R1 nominal v5 representative Isaac gate failed")
    return protocol, approval


def write_order9_r1_nominal_v5_case_contract(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_v5_contract(repository)
    root = materialized.manifest_path.parent
    fast_screen_path = root / "fast_screen.json"
    fast_screen = json.loads(fast_screen_path.read_text(encoding="utf-8"))
    nominal_set_path = repository / materialized.manifest.nominal_set.path
    nominal_set = json.loads(nominal_set_path.read_text(encoding="utf-8"))
    admission = nominal_set.get("metadata", {}).get("nominal_retime_admission")
    if (
        fast_screen.get("accepted") is not True
        or fast_screen.get("isaac_invoked") is not False
        or fast_screen.get("controller_layers_invoked") is not False
        or not isinstance(admission, dict)
        or admission.get("admission_version") != ORDER9_R1_NOMINAL_RETIME_VERSION
        or admission.get("phase_time_scales") != _EXPECTED_PHASE_TIME_SCALES
        or admission.get("geometric_path_unchanged") is not True
        or admission.get("endpoint_configuration_unchanged") is not True
        or admission.get("contact_assignments_unchanged") is not True
        or admission.get("collision_admission_inherited_from_identical_geometry")
        is not True
        or admission.get("accepted") is not True
        or float(admission.get("minimum_joint_rate_margin_rad_s", -1.0)) < 0.0
    ):
        raise SchemaValidationError("R1 nominal v5 case lacks retime admission")
    payload = {
        "contract_version": ORDER9_R1_NOMINAL_V5_CASE_CONTRACT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V5_EXECUTION_CONTRACT,
        "phase_time_scales": _EXPECTED_PHASE_TIME_SCALES,
        "geometric_path_changed": False,
        "endpoint_configuration_changed": False,
        "contact_assignments_changed": False,
        "joint_rate_limit_rechecked": True,
        "collision_admission_inherited_from_identical_geometry": True,
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
            "v5_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V5_PROTOCOL_RELATIVE
            ),
            "v5_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V5_APPROVAL_RELATIVE
            ),
            "case_manifest": _binding(materialized.manifest_path),
            "fast_screen": _binding(fast_screen_path),
            "nominal_set": _binding(nominal_set_path),
            "retime_implementation": _binding(
                repository / protocol["r1_retime_implementation"]["path"]
            ),
        },
        "retime_admission": admission,
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_contract_v5.json",
    )


def run_order9_r1_nominal_v5_isaac_cases(
    cases: Sequence[MaterializedOrder9R1IsaacCase],
    *,
    repository_root: str | Path,
    python_executable: str | Path = (
        "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
    ),
    rollout_steps: int = 7000,
    maximum_parallel_process_count: int = 3,
    persistent_morphology_coalescing: bool = True,
) -> dict[str, tuple[Order9R1NominalReplayResult, ...]]:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v5_contract(repository)
    for case in cases:
        write_order9_r1_nominal_v5_case_contract(case, repository_root=repository)
    results = run_order9_r1_nominal_v3_isaac_cases(
        cases,
        repository_root=repository,
        python_executable=python_executable,
        rollout_steps=rollout_steps,
        maximum_parallel_process_count=maximum_parallel_process_count,
        persistent_morphology_coalescing=persistent_morphology_coalescing,
    )
    for case in cases:
        write_order9_r1_nominal_v5_case_result(
            case,
            results[case.manifest.candidate_id],
            repository_root=repository,
        )
    return results


def write_order9_r1_nominal_v5_case_result(
    materialized: MaterializedOrder9R1IsaacCase,
    results: Sequence[Order9R1NominalReplayResult],
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v5_contract(repository)
    root = materialized.manifest_path.parent
    payload = {
        "result_version": ORDER9_R1_NOMINAL_V5_CASE_RESULT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V5_EXECUTION_CONTRACT,
        "status": "accepted" if all(value.success for value in results) else "rejected",
        "episode_count": len(results),
        "success_count": sum(value.success for value in results),
        "safety_failure_count": sum(value.safety_failure for value in results),
        "fallback_count": sum(value.fallback_used for value in results),
        "phase_time_scales": _EXPECTED_PHASE_TIME_SCALES,
        "pi_l_actor_command_applied": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v5_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V5_PROTOCOL_RELATIVE
            ),
            "v5_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V5_APPROVAL_RELATIVE
            ),
            "v5_case_contract": _binding(root / "nominal_contract_v5.json"),
            "base_v3_result": _binding(root / "nominal_result_v3.json"),
            "raw_rollout": _binding(root / "isaac/evaluation_rollout.pt"),
            "episodes": _binding(root / "isaac/evaluation_episodes.jsonl"),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_result_v5.json",
    )


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": hash_file(source)}


__all__ = [
    "ORDER9_R1_NOMINAL_V5_EXECUTION_CONTRACT",
    "load_order9_r1_nominal_calibration_v5_contract",
    "run_order9_r1_nominal_v5_isaac_cases",
    "write_order9_r1_nominal_v5_case_contract",
    "write_order9_r1_nominal_v5_case_result",
]
