#!/usr/bin/env python3
from __future__ import annotations

"""Run hash-bound R1 nominal calibration with 1% reserve and 30 mm radius."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import order9_run_r1_nominal_calibration_v2 as base_runner  # noqa: E402

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    Order9R1CalibrationCaseOutcome,
    Order9R1DeterministicTeacherScreenPipeline,
)
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_fast_screen import (  # noqa: E402
    Order9R1FastScreenResult,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v3 import (  # noqa: E402
    ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
    load_order9_r1_nominal_calibration_v3_contract,
    run_order9_r1_nominal_v3_isaac_cases,
    write_order9_r1_nominal_v3_case_contract,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

_PREPARATION_PIPELINE = None
_PREPARATION_PIPELINE_KEY = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--maximum-parallel-process-count", type=int, default=2)
    parser.add_argument("--maximum-preparation-process-count", type=int, default=8)
    parser.add_argument("--isaac-case-batch-size", type=int, default=2)
    parser.add_argument(
        "--no-persistent-morphology-coalescing",
        action="store_true",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.rollout_steps < 1
        or args.maximum_parallel_process_count < 1
        or args.maximum_preparation_process_count < 1
        or args.isaac_case_batch_size < 1
    ):
        raise ValueError("R1 nominal v3 runtime limits are invalid")
    repository = REPOSITORY.resolve()
    protocol, approval = load_order9_r1_nominal_calibration_v3_contract(repository)
    base_path = repository / protocol["base_numeric_protocol"]["path"]
    base = load_order9_r1_calibration_protocol(
        base_path,
        repository_root=repository,
    )
    output_root = repository / protocol["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    # Reuse the reviewed v2 ladder/early-stop implementation while replacing
    # only its candidate preparation and nominal case-evidence adapters.
    base_runner._prepare_and_screen_case = _prepare_and_screen_case_v3
    base_runner.run_order9_r1_nominal_isaac_cases = run_order9_r1_nominal_v3_isaac_cases
    base_runner.write_order9_r1_nominal_case_contract = (
        write_order9_r1_nominal_v3_case_contract
    )
    base_runner.ORDER9_R1_NOMINAL_EXECUTION_CONTRACT = (
        ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT
    )

    started = perf_counter()
    selection_results = []
    selected_level = None
    total_case_evidence_count = 0
    for level in base.calibration_levels:
        result, evidence_count = base_runner._run_level(
            split=base.selection_split,
            level_id=level.level_id,
            base=base,
            base_path=base_path,
            output_root=output_root / "selection" / level.level_id,
            repository=repository,
            isaac_python=args.isaac_python,
            rollout_steps=args.rollout_steps,
            maximum_parallel_process_count=(args.maximum_parallel_process_count),
            maximum_preparation_process_count=(args.maximum_preparation_process_count),
            isaac_case_batch_size=args.isaac_case_batch_size,
            persistent_morphology_coalescing=(
                not args.no_persistent_morphology_coalescing
            ),
        )
        _write_level_v3(
            output_root / "selection" / level.level_id,
            result,
            evidence_count,
            protocol,
            approval,
        )
        selection_results.append(result)
        total_case_evidence_count += evidence_count
        _write_run_state_v3(
            output_root=output_root,
            status="selection_in_progress",
            selection_results=selection_results,
            confirmation_result=None,
            selected_level_id=(level.level_id if result.passed else selected_level),
            accepted_level_id=None,
            total_case_evidence_count=total_case_evidence_count,
            measured_wall_time_s=perf_counter() - started,
            protocol=protocol,
            approval=approval,
        )
        if not result.passed:
            break
        selected_level = level.level_id

    confirmation = None
    if selected_level is not None:
        confirmation, evidence_count = base_runner._run_level(
            split=base.confirmation_split,
            level_id=selected_level,
            base=base,
            base_path=base_path,
            output_root=output_root / "confirmation" / selected_level,
            repository=repository,
            isaac_python=args.isaac_python,
            rollout_steps=args.rollout_steps,
            maximum_parallel_process_count=(args.maximum_parallel_process_count),
            maximum_preparation_process_count=(args.maximum_preparation_process_count),
            isaac_case_batch_size=args.isaac_case_batch_size,
            persistent_morphology_coalescing=(
                not args.no_persistent_morphology_coalescing
            ),
        )
        _write_level_v3(
            output_root / "confirmation" / selected_level,
            confirmation,
            evidence_count,
            protocol,
            approval,
        )
        total_case_evidence_count += evidence_count
    accepted_level = (
        selected_level if confirmation is not None and confirmation.passed else None
    )
    destination = _write_run_state_v3(
        output_root=output_root,
        status=("accepted" if accepted_level is not None else "rejected"),
        selection_results=selection_results,
        confirmation_result=confirmation,
        selected_level_id=selected_level,
        accepted_level_id=accepted_level,
        total_case_evidence_count=total_case_evidence_count,
        measured_wall_time_s=perf_counter() - started,
        protocol=protocol,
        approval=approval,
    )
    print(f"ORDER9_R1_NOMINAL_V3_CALIBRATION_RESULT={destination}", flush=True)
    return 0


def _prepare_and_screen_case_v3(arguments):
    (
        case,
        repository_text,
        source_manifest_text,
        minimum_joint_reserve,
        maximum_body_tilt,
        minimum_support_margin,
        base_path_text,
        output_root_text,
    ) = arguments
    repository = Path(repository_text)
    output_root = Path(output_root_text)
    protocol, _approval = load_order9_r1_nominal_calibration_v3_contract(repository)
    destination = output_root / case.candidate_id
    manifest_path = destination / "case_manifest.json"
    if manifest_path.is_file():
        persisted = load_order9_r1_isaac_case(manifest_path, repository)
        screen_payload = json.loads(
            (destination / "fast_screen.json").read_text(encoding="utf-8")
        )
        screen_payload["checked_phases"] = tuple(screen_payload["checked_phases"])
        screen_payload["violation_codes"] = tuple(screen_payload["violation_codes"])
        screen = Order9R1FastScreenResult(**screen_payload)
        write_order9_r1_nominal_v3_case_contract(
            persisted,
            repository_root=repository,
        )
        return (
            Order9R1CalibrationCaseOutcome(
                case=case,
                support_gate_passed=True,
                teacher_feasible=True,
                fast_screen=screen,
                full_layer_replays=(),
                failure_reason=None,
            ),
            manifest_path,
        )
    support_passed = (
        case.support_audit.minimum_projected_com_margin_m >= minimum_support_margin
    )
    if not support_passed:
        outcome = Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=False,
            teacher_feasible=False,
            fast_screen=None,
            full_layer_replays=(),
            failure_reason="projected center of mass is outside support",
        )
        _write_cpu_rejection_v3(output_root, outcome)
        return outcome, None

    global _PREPARATION_PIPELINE, _PREPARATION_PIPELINE_KEY
    pipeline_key = (
        repository_text,
        source_manifest_text,
        minimum_joint_reserve,
        maximum_body_tilt,
        float(protocol["maximum_grasp_point_position_error_m"]),
    )
    if _PREPARATION_PIPELINE_KEY != pipeline_key:
        _PREPARATION_PIPELINE = Order9R1DeterministicTeacherScreenPipeline(
            repository_root=repository,
            source_bucket_manifest_path=(repository / source_manifest_text),
            minimum_normalized_joint_limit_reserve=minimum_joint_reserve,
            maximum_body_tilt_rad=maximum_body_tilt,
            anchor_position_tolerance_m=float(
                protocol["maximum_grasp_point_position_error_m"]
            ),
            enforce_joint_limit_reserve_during_ik=True,
        )
        _PREPARATION_PIPELINE_KEY = pipeline_key
    prepared = _PREPARATION_PIPELINE.prepare(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete:
        outcome = Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=True,
            teacher_feasible=False,
            fast_screen=None,
            full_layer_replays=(),
            failure_reason=prepared.failure_reason,
        )
        _write_cpu_rejection_v3(output_root, outcome)
        return outcome, None
    screen = _PREPARATION_PIPELINE.screen(case, prepared)
    if not screen.accepted:
        outcome = Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=True,
            teacher_feasible=True,
            fast_screen=screen,
            full_layer_replays=(),
            failure_reason="fast exact-tracking screen rejected the candidate",
        )
        _write_cpu_rejection_v3(output_root, outcome)
        return outcome, None
    persisted = materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=destination,
        repository_root=repository,
        approved_protocol_path=Path(base_path_text),
    )
    write_order9_r1_nominal_v3_case_contract(
        persisted,
        repository_root=repository,
    )
    outcome = Order9R1CalibrationCaseOutcome(
        case=case,
        support_gate_passed=True,
        teacher_feasible=True,
        fast_screen=screen,
        full_layer_replays=(),
        failure_reason=None,
    )
    return outcome, persisted.manifest_path


def _write_cpu_rejection_v3(output_root, outcome) -> None:
    write_order9_r1_clearance_diagnostic(
        {
            "result_version": "order9_r1_nominal_cpu_rejection_v3",
            "execution_contract": ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
            "candidate_id": outcome.case.candidate_id,
            "support_gate_passed": outcome.support_gate_passed,
            "teacher_feasible": outcome.teacher_feasible,
            "fast_screen": (
                None if outcome.fast_screen is None else outcome.fast_screen.to_dict()
            ),
            "failure_reason": outcome.failure_reason,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "training_eligible": False,
        },
        output_root / "cpu_rejections" / f"{outcome.case.candidate_id}.json",
    )


def _write_level_v3(
    output_root,
    result,
    evidence_count,
    protocol,
    approval,
) -> Path:
    return write_order9_r1_clearance_diagnostic(
        {
            "result_version": "order9_r1_nominal_calibration_level_result_v3",
            "execution_contract": ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
            "joint_limit_reserve_projection_enabled": True,
            "minimum_normalized_joint_limit_reserve": 0.01,
            "maximum_grasp_point_position_error_m": 0.030,
            "grasp_point_position_error_metric": (
                "three_dimensional_euclidean_distance"
            ),
            "pi_l_actor_command_applied": False,
            "case_evidence_count": evidence_count,
            "result": result.to_dict(),
            "bindings": _run_bindings(protocol, approval),
        },
        output_root / "level_result_v3.json",
    )


def _write_run_state_v3(
    *,
    output_root,
    status,
    selection_results,
    confirmation_result,
    selected_level_id,
    accepted_level_id,
    total_case_evidence_count,
    measured_wall_time_s,
    protocol,
    approval,
) -> Path:
    payload = {
        "result_version": "order9_r1_nominal_calibration_run_result_v3",
        "status": status,
        "execution_contract": ORDER9_R1_NOMINAL_V3_EXECUTION_CONTRACT,
        "joint_limit_reserve_projection_enabled": True,
        "minimum_normalized_joint_limit_reserve": 0.01,
        "maximum_grasp_point_position_error_m": 0.030,
        "grasp_point_position_error_metric": ("three_dimensional_euclidean_distance"),
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "selection_results": [value.to_dict() for value in selection_results],
        "selected_level_id": selected_level_id,
        "confirmation_result": (
            None if confirmation_result is None else confirmation_result.to_dict()
        ),
        "confirmation_attempt_count": 0 if confirmation_result is None else 1,
        "accepted_level_id": accepted_level_id,
        "case_evidence_count": total_case_evidence_count,
        "planned_case_count": sum(value.case_count for value in selection_results)
        + (0 if confirmation_result is None else confirmation_result.case_count),
        "measured_wall_time_s": max(measured_wall_time_s, 1.0e-12),
        "c3_promotion_evidence_eligible": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": accepted_level_id is not None,
        "bindings": {
            **_run_bindings(protocol, approval),
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": hash_file(Path(__file__).resolve()),
            },
        },
        "approval_record_version": approval["record_version"],
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        output_root / "calibration_result_v3.json",
    )


def _run_bindings(protocol, approval) -> dict[str, object]:
    protocol_path = REPOSITORY / (
        "configs/training/order9_r1_nominal_calibration_protocol_v3.yaml"
    )
    approval_path = REPOSITORY / protocol["approval_record"]
    if approval.get("approved_protocol", {}).get("sha256") != hash_file(protocol_path):
        raise SchemaValidationError("R1 nominal v3 run approval differs")
    return {
        "v3_protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": hash_file(protocol_path),
        },
        "v3_approval": {
            "path": str(approval_path.resolve()),
            "sha256": hash_file(approval_path),
        },
        "base_nominal_protocol": protocol["base_nominal_protocol"],
        "base_numeric_protocol": protocol["base_numeric_protocol"],
        "joint_reserve_projection_implementation": protocol[
            "joint_reserve_projection_implementation"
        ],
        "teacher_screen_pipeline_implementation": protocol[
            "teacher_screen_pipeline_implementation"
        ],
        "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
        "protected_c3_rollout": protocol["protected_c3_rollout"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
