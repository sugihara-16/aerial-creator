from __future__ import annotations

"""Hash-bound R1 calibration with early safe teacher construction."""

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
    ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
    load_order9_r1_nominal_calibration_v3_contract,
    run_order9_r1_nominal_v3_isaac_cases,
    write_order9_r1_nominal_v3_case_contract,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_V4_PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v4"
ORDER9_R1_NOMINAL_V4_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v4"
)
ORDER9_R1_NOMINAL_V4_EXECUTION_CONTRACT = (
    "deterministic_teacher_early_tilt_alternative_surface_joint_reserve_"
    "30mm_nominal_qpid_qp_local_servo_isaac_v3"
)
ORDER9_R1_NOMINAL_V4_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v4.yaml"
)
ORDER9_R1_NOMINAL_V4_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V4_APPROVAL.json"
)
ORDER9_R1_NOMINAL_V4_CASE_CONTRACT_VERSION = (
    "order9_r1_nominal_calibration_case_contract_v4"
)
ORDER9_R1_NOMINAL_V4_CASE_RESULT_VERSION = (
    "order9_r1_nominal_calibration_case_result_v4"
)
_EXPECTED_PROTOCOL_SHA256 = (
    "59b5f9478b6f559b04eb7f378acb458d918aff16a3d656e851aa3ae6d33184ef"
)
_EXPECTED_APPROVAL_SHA256 = (
    "ca81c4c79a6821b74d77d72d1255f9c21ee36c5694907a45ff4258b37207eced"
)
_REQUIRED_APPROVAL_SCOPE = {
    "teacher_failures_handled_internally",
    "deterministic_ad_hoc_teacher_surface_selection_allowed",
    "contact_solution_body_tilt_60deg_hard_boundary",
    "extreme_contact_solution_removed_before_dense_path",
    "alternative_unoccupied_surface_pair_allowed",
    "three_dimensional_grasp_point_distance_30mm",
    "one_percent_joint_limit_reserve_projection",
    "fast_screen_before_isaac",
    "nominal_qpid_qp_local_servo_standard_path",
    "pi_l_retained_but_not_applied",
    "reuse_four_level_numeric_ladder",
    "selection_on_22_train_buckets",
    "single_confirmation_on_14_validation_buckets",
    "two_isaac_replays_per_screen_accepted_candidate",
    "stop_after_first_failed_level",
    "no_learning_by_protocol_alone",
}
_BINDING_KEYS = (
    "base_v3_protocol",
    "base_v3_approval",
    "base_v3_result_ledger",
    "base_numeric_protocol",
    "base_numeric_approval",
    "protected_c3_curriculum",
    "protected_c3_checkpoint",
    "protected_c3_rollout",
    "protected_c3_release_ledger",
    "articulated_teacher_implementation",
    "c3_teacher_adapter_implementation",
    "nominal_teacher_implementation",
    "r1_early_tilt_pipeline_implementation",
    "r1_teacher_overrides",
    "joint_reserve_projection_implementation",
    "teacher_screen_pipeline_implementation",
    "nominal_v3_runtime_overlay",
    "nominal_isaac_runtime_implementation",
)
# v9 is authoritative. These paths are live implementation locations that
# later R1 revisions intentionally replaced; their historical digests remain
# immutable in the v4 protocol. Persisted evidence and protected C3 bindings
# below still require an exact current-byte match.
_SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS = frozenset(
    {
        "c3_teacher_adapter_implementation",
        "nominal_teacher_implementation",
        "r1_early_tilt_pipeline_implementation",
    }
)


def load_order9_r1_nominal_calibration_v4_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_V4_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_V4_APPROVAL_RELATIVE
    if hash_file(protocol_path) != _EXPECTED_PROTOCOL_SHA256:
        raise SchemaValidationError("R1 nominal v4 protocol bytes changed")
    if hash_file(approval_path) != _EXPECTED_APPROVAL_SHA256:
        raise SchemaValidationError("R1 nominal v4 approval bytes changed")
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version") != ORDER9_R1_NOMINAL_V4_PROTOCOL_VERSION
        or protocol.get("calibration_gate_id")
        != "r1-nominal-early-safe-teacher-gate-v4"
        or protocol.get("status") != "approved"
        or protocol.get("execution_contract") != ORDER9_R1_NOMINAL_V4_EXECUTION_CONTRACT
        or protocol.get("base_v3_execution_contract")
        != ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT
        or protocol.get("contact_solution_tilt_checked_before_dense_path") is not True
        or protocol.get(
            "extreme_contact_solution_removed_before_configuration_space_planning"
        )
        is not True
        or float(protocol.get("maximum_body_tilt_rad", -1.0)) != 1.0471975511965976
        or protocol.get("alternative_unoccupied_surface_pair_allowed") is not True
        or float(protocol.get("minimum_normalized_joint_limit_reserve", -1.0)) != 0.01
        or protocol.get("grasp_point_position_error_metric")
        != "three_dimensional_euclidean_distance"
        or float(protocol.get("maximum_grasp_point_position_error_m", -1.0)) != 0.030
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("isaac_required_after_fast_screen") is not True
    ):
        raise SchemaValidationError("R1 nominal v4 protocol contract is invalid")
    if (
        approval.get("record_version") != ORDER9_R1_NOMINAL_V4_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("calibration_gate_id") != protocol["calibration_gate_id"]
        or set(approval.get("approval_scope", ())) != _REQUIRED_APPROVAL_SCOPE
        or approval.get("approved_protocol", {}).get("sha256")
        != _EXPECTED_PROTOCOL_SHA256
    ):
        raise SchemaValidationError("R1 nominal v4 approval record is invalid")
    for key in _BINDING_KEYS:
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v4 binding missing: {key}")
        source = repository / str(binding.get("path", ""))
        if not source.is_file() or (
            key not in _SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS
            and hash_file(source) != binding.get("sha256")
        ):
            raise SchemaValidationError(f"R1 nominal v4 binding changed: {key}")
    load_order9_r1_nominal_calibration_v3_contract(repository)
    return protocol, approval


def write_order9_r1_nominal_v4_case_contract(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_v4_contract(repository)
    base_contract = write_order9_r1_nominal_v3_case_contract(
        materialized,
        repository_root=repository,
    )
    root = materialized.manifest_path.parent
    fast_screen_path = root / "fast_screen.json"
    fast_screen = json.loads(fast_screen_path.read_text(encoding="utf-8"))
    if (
        fast_screen.get("accepted") is not True
        or float(fast_screen.get("maximum_body_tilt_rad", float("inf")))
        > float(protocol["maximum_body_tilt_rad"]) + 1.0e-9
        or float(fast_screen.get("minimum_normalized_joint_limit_reserve", -1.0))
        < float(protocol["minimum_normalized_joint_limit_reserve"])
        or fast_screen.get("maximum_collision_violating_pair_count") != 0
    ):
        raise SchemaValidationError(
            "R1 nominal v4 case lacks accepted tilt/reserve/collision evidence"
        )
    payload = {
        "contract_version": ORDER9_R1_NOMINAL_V4_CASE_CONTRACT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V4_EXECUTION_CONTRACT,
        "contact_solution_tilt_checked_before_dense_path": True,
        "maximum_body_tilt_rad": protocol["maximum_body_tilt_rad"],
        "alternative_unoccupied_surface_pair_allowed": True,
        "joint_limit_reserve_projection_enabled": True,
        "minimum_normalized_joint_limit_reserve": 0.01,
        "grasp_point_position_error_metric": ("three_dimensional_euclidean_distance"),
        "maximum_grasp_point_position_error_m": 0.030,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "isaac_required": True,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v4_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V4_PROTOCOL_RELATIVE
            ),
            "v4_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V4_APPROVAL_RELATIVE
            ),
            "base_v3_contract": _binding(base_contract),
            "case_manifest": _binding(materialized.manifest_path),
            "fast_screen": _binding(fast_screen_path),
            "early_tilt_pipeline": _binding(
                repository / protocol["r1_early_tilt_pipeline_implementation"]["path"]
            ),
            "teacher_overrides": _binding(
                repository / protocol["r1_teacher_overrides"]["path"]
            ),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_contract_v4.json",
    )


def run_order9_r1_nominal_v4_isaac_cases(
    cases: Sequence[MaterializedOrder9R1IsaacCase],
    *,
    repository_root: str | Path,
    python_executable: str | Path = (
        "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
    ),
    rollout_steps: int = 15000,
    maximum_parallel_process_count: int = 2,
    persistent_morphology_coalescing: bool = True,
) -> dict[str, tuple[Order9R1NominalReplayResult, ...]]:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v4_contract(repository)
    for case in cases:
        write_order9_r1_nominal_v4_case_contract(
            case,
            repository_root=repository,
        )
    results = run_order9_r1_nominal_v3_isaac_cases(
        cases,
        repository_root=repository,
        python_executable=python_executable,
        rollout_steps=rollout_steps,
        maximum_parallel_process_count=maximum_parallel_process_count,
        persistent_morphology_coalescing=persistent_morphology_coalescing,
    )
    for case in cases:
        write_order9_r1_nominal_v4_case_result(
            case,
            results[case.manifest.candidate_id],
            repository_root=repository,
        )
    return results


def write_order9_r1_nominal_v4_case_result(
    materialized: MaterializedOrder9R1IsaacCase,
    results: Sequence[Order9R1NominalReplayResult],
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v4_contract(repository)
    root = materialized.manifest_path.parent
    payload = {
        "result_version": ORDER9_R1_NOMINAL_V4_CASE_RESULT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V4_EXECUTION_CONTRACT,
        "status": (
            "accepted" if all(value.success for value in results) else "rejected"
        ),
        "episode_count": len(results),
        "success_count": sum(value.success for value in results),
        "safety_failure_count": sum(value.safety_failure for value in results),
        "fallback_count": sum(value.fallback_used for value in results),
        "pi_l_actor_command_applied": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v4_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V4_PROTOCOL_RELATIVE
            ),
            "v4_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V4_APPROVAL_RELATIVE
            ),
            "v4_case_contract": _binding(root / "nominal_contract_v4.json"),
            "base_v3_result": _binding(root / "nominal_result_v3.json"),
            "raw_rollout": _binding(root / "isaac/evaluation_rollout.pt"),
            "episodes": _binding(root / "isaac/evaluation_episodes.jsonl"),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_result_v4.json",
    )


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": hash_file(source)}


__all__ = [
    "ORDER9_R1_NOMINAL_V4_EXECUTION_CONTRACT",
    "load_order9_r1_nominal_calibration_v4_contract",
    "run_order9_r1_nominal_v4_isaac_cases",
    "write_order9_r1_nominal_v4_case_contract",
    "write_order9_r1_nominal_v4_case_result",
]
