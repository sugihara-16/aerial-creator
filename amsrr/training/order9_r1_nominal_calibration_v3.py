from __future__ import annotations

"""Hash-bound R1 nominal calibration overlay for joint reserve and 30 mm.

The Isaac execution semantics are deliberately reused from the approved v2
nominal runtime.  This overlay owns candidate admission, its additional
contract/evidence bindings, and the v3 result wrapper without mutating v2.
"""

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
    ORDER9_R1_NOMINAL_EXECUTION_CONTRACT as BASE_EXECUTION_CONTRACT,
    Order9R1NominalReplayResult,
    load_order9_r1_nominal_calibration_contract as load_base_contract,
    run_order9_r1_nominal_isaac_cases as run_base_isaac_cases,
    write_order9_r1_nominal_case_contract as write_base_case_contract,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_V3_PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v3"
ORDER9_R1_NOMINAL_V3_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v3"
)
ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT = (
    "deterministic_teacher_ik_joint_reserve_30mm_nominal_"
    "qpid_qp_local_servo_isaac_v2"
)
ORDER9_R1_NOMINAL_V3_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v3.yaml"
)
ORDER9_R1_NOMINAL_V3_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V3_APPROVAL.json"
)
ORDER9_R1_NOMINAL_V3_CASE_CONTRACT_VERSION = (
    "order9_r1_nominal_calibration_case_contract_v3"
)
ORDER9_R1_NOMINAL_V3_CASE_RESULT_VERSION = (
    "order9_r1_nominal_calibration_case_result_v3"
)
_EXPECTED_PROTOCOL_SHA256 = (
    "e0dfee539548a508426848a855122a95b9b45456c459f8ee4bc7102e7edae63c"
)
_EXPECTED_APPROVAL_SHA256 = (
    "b589fc3db1f90e94f4c8a6e77a055c2759839e7701be24d023e94906f4cbb27d"
)
_REQUIRED_APPROVAL_SCOPE = {
    "three_dimensional_grasp_point_distance_30mm",
    "not_per_axis_plus_minus_30mm",
    "one_percent_joint_limit_reserve_projection",
    "fail_closed_anchor_collision_and_joint_rate_recheck",
    "reuse_four_level_numeric_ladder",
    "selection_on_22_train_buckets",
    "single_confirmation_on_14_validation_buckets",
    "fast_screen_before_isaac",
    "nominal_qpid_qp_local_servo_standard_path",
    "pi_l_retained_but_not_applied",
    "reuse_v2_nominal_isaac_runtime_contract",
    "two_isaac_replays_per_screen_accepted_candidate",
    "stop_after_first_failed_level",
    "no_teacher_collection_or_learning_by_protocol_alone",
}


def load_order9_r1_nominal_calibration_v3_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_V3_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_V3_APPROVAL_RELATIVE
    if hash_file(protocol_path) != _EXPECTED_PROTOCOL_SHA256:
        raise SchemaValidationError("R1 nominal v3 protocol bytes changed")
    if hash_file(approval_path) != _EXPECTED_APPROVAL_SHA256:
        raise SchemaValidationError("R1 nominal v3 approval bytes changed")
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version") != ORDER9_R1_NOMINAL_V3_PROTOCOL_VERSION
        or protocol.get("calibration_gate_id")
        != "r1-nominal-joint-reserve-30mm-gate-v3"
        or protocol.get("status") != "approved"
        or protocol.get("execution_contract") != ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT
        or protocol.get("base_nominal_isaac_execution_contract")
        != BASE_EXECUTION_CONTRACT
        or protocol.get("joint_limit_reserve_projection_enabled") is not True
        or float(protocol.get("minimum_normalized_joint_limit_reserve", -1.0)) != 0.01
        or protocol.get("grasp_point_position_error_metric")
        != "three_dimensional_euclidean_distance"
        or float(protocol.get("maximum_grasp_point_position_error_m", -1.0)) != 0.030
        or protocol.get("per_axis_plus_minus_30mm_contract") is not False
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("isaac_required_after_fast_screen") is not True
    ):
        raise SchemaValidationError("R1 nominal v3 protocol contract is invalid")
    if (
        approval.get("record_version") != ORDER9_R1_NOMINAL_V3_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("calibration_gate_id") != protocol["calibration_gate_id"]
        or set(approval.get("approval_scope", ())) != _REQUIRED_APPROVAL_SCOPE
        or approval.get("approved_protocol", {}).get("sha256")
        != _EXPECTED_PROTOCOL_SHA256
    ):
        raise SchemaValidationError("R1 nominal v3 approval record is invalid")
    for key in (
        "base_nominal_protocol",
        "base_nominal_approval",
        "base_numeric_protocol",
        "base_numeric_approval",
        "protected_c3_curriculum",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_release_ledger",
        "joint_reserve_projection_implementation",
        "teacher_screen_pipeline_implementation",
        "nominal_isaac_runtime_implementation",
    ):
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v3 binding missing: {key}")
        source = repository / str(binding.get("path", ""))
        if not source.is_file() or hash_file(source) != binding.get("sha256"):
            raise SchemaValidationError(f"R1 nominal v3 binding changed: {key}")
    if hash_file(approval_path) != hash_file(
        repository / str(protocol["approval_record"])
    ):
        raise SchemaValidationError("R1 nominal v3 approval path differs")
    load_base_contract(repository)
    return protocol, approval


def write_order9_r1_nominal_v3_case_contract(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_v3_contract(repository)
    base_contract = write_base_case_contract(
        materialized,
        repository_root=repository,
    )
    root = materialized.manifest_path.parent
    fast_screen_path = root / "fast_screen.json"
    fast_screen = json.loads(fast_screen_path.read_text(encoding="utf-8"))
    if (
        fast_screen.get("accepted") is not True
        or float(fast_screen.get("minimum_normalized_joint_limit_reserve", -1.0))
        < float(protocol["minimum_normalized_joint_limit_reserve"])
        or fast_screen.get("maximum_collision_violating_pair_count") != 0
    ):
        raise SchemaValidationError(
            "R1 nominal v3 case lacks an accepted reserve/collision screen"
        )
    payload = {
        "contract_version": ORDER9_R1_NOMINAL_V3_CASE_CONTRACT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
        "base_nominal_isaac_execution_contract": BASE_EXECUTION_CONTRACT,
        "joint_limit_reserve_projection_enabled": True,
        "minimum_normalized_joint_limit_reserve": 0.01,
        "grasp_point_position_error_metric": ("three_dimensional_euclidean_distance"),
        "maximum_grasp_point_position_error_m": 0.030,
        "per_axis_plus_minus_30mm_contract": False,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "isaac_required": True,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v3_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V3_PROTOCOL_RELATIVE
            ),
            "v3_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V3_APPROVAL_RELATIVE
            ),
            "base_nominal_contract": _binding(base_contract),
            "case_manifest": _binding(materialized.manifest_path),
            "fast_screen": _binding(fast_screen_path),
            "joint_reserve_projection_implementation": _binding(
                repository / protocol["joint_reserve_projection_implementation"]["path"]
            ),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_contract_v3.json",
    )


def run_order9_r1_nominal_v3_isaac_cases(
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
    load_order9_r1_nominal_calibration_v3_contract(repository)
    for case in cases:
        write_order9_r1_nominal_v3_case_contract(
            case,
            repository_root=repository,
        )
    results = run_base_isaac_cases(
        cases,
        repository_root=repository,
        python_executable=python_executable,
        rollout_steps=rollout_steps,
        maximum_parallel_process_count=maximum_parallel_process_count,
        persistent_morphology_coalescing=persistent_morphology_coalescing,
    )
    for case in cases:
        write_order9_r1_nominal_v3_case_result(
            case,
            results[case.manifest.candidate_id],
            repository_root=repository,
        )
    return results


def write_order9_r1_nominal_v3_case_result(
    materialized: MaterializedOrder9R1IsaacCase,
    results: Sequence[Order9R1NominalReplayResult],
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_v3_contract(repository)
    root = materialized.manifest_path.parent
    payload = {
        "result_version": ORDER9_R1_NOMINAL_V3_CASE_RESULT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
        "base_nominal_isaac_execution_contract": BASE_EXECUTION_CONTRACT,
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
            "v3_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V3_PROTOCOL_RELATIVE
            ),
            "v3_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V3_APPROVAL_RELATIVE
            ),
            "v3_case_contract": _binding(root / "nominal_contract_v3.json"),
            "base_nominal_result": _binding(root / "nominal_result.json"),
            "raw_rollout": _binding(root / "isaac/evaluation_rollout.pt"),
            "episodes": _binding(root / "isaac/evaluation_episodes.jsonl"),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        root / "nominal_result_v3.json",
    )


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": hash_file(source)}


__all__ = [
    "ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT",
    "load_order9_r1_nominal_calibration_v3_contract",
    "run_order9_r1_nominal_v3_isaac_cases",
    "write_order9_r1_nominal_v3_case_contract",
    "write_order9_r1_nominal_v3_case_result",
]
