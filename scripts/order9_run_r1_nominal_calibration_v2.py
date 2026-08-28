#!/usr/bin/env python3
from __future__ import annotations

"""Run the approved four-level R1 calibration through nominal control."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    Order9R1CalibrationCaseOutcome,
    Order9R1DeterministicTeacherScreenPipeline,
    enumerate_order9_r1_calibration_level_cases,
    _summarize_level,
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
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    ORDER9_R1_NOMINAL_EXECUTION_CONTRACT,
    load_order9_r1_nominal_calibration_contract,
    run_order9_r1_nominal_isaac_cases,
    write_order9_r1_nominal_case_contract,
)
from amsrr.utils.hashing import hash_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/training/order9_r1_nominal_calibration_protocol_v2.yaml",
    )
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
        raise ValueError("R1 nominal calibration runtime limits are invalid")
    repository = REPOSITORY.resolve()
    protocol, approval = load_order9_r1_nominal_calibration_contract(repository)
    base_path = repository / protocol["base_numeric_protocol"]["path"]
    base = load_order9_r1_calibration_protocol(
        base_path,
        repository_root=repository,
    )
    output_root = repository / protocol["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    selection_results = []
    selected_level = None
    total_case_evidence_count = 0
    for level in base.calibration_levels:
        result, evidence_count = _run_level(
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
        selection_results.append(result)
        total_case_evidence_count += evidence_count
        _write_run_state(
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
        confirmation, evidence_count = _run_level(
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
        total_case_evidence_count += evidence_count
    accepted_level = (
        selected_level if confirmation is not None and confirmation.passed else None
    )
    status = "accepted" if accepted_level is not None else "rejected"
    destination = _write_run_state(
        output_root=output_root,
        status=status,
        selection_results=selection_results,
        confirmation_result=confirmation,
        selected_level_id=selected_level,
        accepted_level_id=accepted_level,
        total_case_evidence_count=total_case_evidence_count,
        measured_wall_time_s=perf_counter() - started,
        protocol=protocol,
        approval=approval,
    )
    print(f"ORDER9_R1_NOMINAL_CALIBRATION_RESULT={destination}", flush=True)
    return 0


def _run_level(
    *,
    split,
    level_id,
    base,
    base_path,
    output_root,
    repository,
    isaac_python,
    rollout_steps,
    maximum_parallel_process_count,
    maximum_preparation_process_count,
    isaac_case_batch_size,
    persistent_morphology_coalescing,
):
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=base_path,
        repository_root=repository,
        level_id=level_id,
        split=split,
    )
    level = next(
        value for value in base.calibration_levels if value.level_id == level_id
    )
    print(
        "ORDER9_R1_NOMINAL_LEVEL_START="
        + json.dumps(
            {
                "level_id": level_id,
                "split": split,
                "case_count": len(cases),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    execution_cases = sorted(cases, key=_execution_case_priority)
    worker_arguments = [
        (
            case,
            str(repository),
            str(base.source_bucket_manifest.path),
            base.minimum_normalized_joint_limit_reserve,
            base.maximum_body_tilt_rad,
            base.minimum_support_projected_com_margin_m,
            str(base_path),
            str(output_root),
        )
        for case in execution_cases
    ]
    outcomes_by_id = {}
    materialized = []
    completed_screen_count = 0
    cpu_early_stop_reason = None
    with ProcessPoolExecutor(max_workers=maximum_preparation_process_count) as executor:
        arguments_iterator = iter(worker_arguments)
        pending = {
            executor.submit(_prepare_and_screen_case, arguments): arguments
            for arguments in _take(
                arguments_iterator, maximum_preparation_process_count
            )
        }
        while pending:
            done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                outcome, manifest_path = future.result()
                outcomes_by_id[outcome.case.candidate_id] = outcome
                completed_screen_count += 1
                if manifest_path is not None:
                    materialized.append(
                        load_order9_r1_isaac_case(manifest_path, repository)
                    )
                if outcome.case.sample_kind == "lattice" and not _cpu_outcome_passed(
                    outcome
                ):
                    cpu_early_stop_reason = "lattice_cpu_gate_cannot_reach_100_percent"
                if completed_screen_count % 10 == 0:
                    print(
                        "ORDER9_R1_NOMINAL_SCREEN_PROGRESS="
                        + json.dumps(
                            {
                                "level_id": level_id,
                                "split": split,
                                "completed": completed_screen_count,
                                "total": len(cases),
                                "admitted_for_isaac": len(materialized),
                                "early_stop_reason": cpu_early_stop_reason,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if cpu_early_stop_reason is None:
                for arguments in _take(arguments_iterator, len(done)):
                    pending[executor.submit(_prepare_and_screen_case, arguments)] = (
                        arguments
                    )
    if cpu_early_stop_reason is not None:
        for case in cases:
            if case.candidate_id in outcomes_by_id:
                continue
            outcomes_by_id[case.candidate_id] = Order9R1CalibrationCaseOutcome(
                case=case,
                support_gate_passed=(
                    case.support_audit.minimum_projected_com_margin_m
                    >= base.minimum_support_projected_com_margin_m
                ),
                teacher_feasible=False,
                fast_screen=None,
                full_layer_replays=(),
                failure_reason=("not attempted after a decisive lattice CPU failure"),
            )
    if completed_screen_count % 10 != 0:
        print(
            "ORDER9_R1_NOMINAL_SCREEN_PROGRESS="
            + json.dumps(
                {
                    "level_id": level_id,
                    "split": split,
                    "completed": completed_screen_count,
                    "total": len(cases),
                    "admitted_for_isaac": len(materialized),
                    "early_stop_reason": cpu_early_stop_reason,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    execution_priority = {
        case.candidate_id: index for index, case in enumerate(execution_cases)
    }
    materialized.sort(key=lambda value: execution_priority[value.manifest.candidate_id])
    outcomes = [outcomes_by_id[case.candidate_id] for case in cases]
    termination_reason = cpu_early_stop_reason or _decisive_cpu_failure(base, outcomes)
    replay_results = {}
    print(
        "ORDER9_R1_NOMINAL_ISAAC_START="
        + json.dumps(
            {
                "level_id": level_id,
                "split": split,
                "case_count": (
                    0 if termination_reason is not None else len(materialized)
                ),
                "skipped_reason": termination_reason,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if termination_reason is None:
        for start in range(0, len(materialized), isaac_case_batch_size):
            batch = materialized[start : start + isaac_case_batch_size]
            batch_results = run_order9_r1_nominal_isaac_cases(
                batch,
                repository_root=repository,
                python_executable=isaac_python,
                rollout_steps=rollout_steps,
                maximum_parallel_process_count=(maximum_parallel_process_count),
                persistent_morphology_coalescing=(persistent_morphology_coalescing),
            )
            replay_results.update(batch_results)
            _apply_replay_results(outcomes_by_id, batch_results)
            outcomes = [outcomes_by_id[case.candidate_id] for case in cases]
            print(
                "ORDER9_R1_NOMINAL_ISAAC_PROGRESS="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "split": split,
                        "completed_cases": min(start + len(batch), len(materialized)),
                        "total_cases": len(materialized),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            termination_reason = _decisive_isaac_failure(
                base,
                outcomes,
            )
            if termination_reason is not None:
                print(
                    "ORDER9_R1_NOMINAL_ISAAC_EARLY_STOP="
                    + json.dumps(
                        {
                            "level_id": level_id,
                            "split": split,
                            "reason": termination_reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    outcomes = [outcomes_by_id[case.candidate_id] for case in cases]
    result = _summarize_level(
        base,
        level,
        split,
        len({case.source_bucket.bucket_id for case in cases}),
        outcomes,
    )
    write_order9_r1_clearance_diagnostic(
        {
            "result_version": "order9_r1_nominal_calibration_level_result_v2",
            "execution_contract": ORDER9_R1_NOMINAL_EXECUTION_CONTRACT,
            "pi_l_actor_command_applied": False,
            "c3_promotion_evidence_eligible": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
            "termination_reason": termination_reason,
            "cpu_preparation_attempt_count": completed_screen_count,
            "cpu_preparation_unattempted_count": (len(cases) - completed_screen_count),
            "result": result.to_dict(),
        },
        output_root / "level_result.json",
    )
    print(
        "ORDER9_R1_NOMINAL_LEVEL_RESULT="
        + json.dumps(result.to_dict(), sort_keys=True),
        flush=True,
    )
    return result, completed_screen_count


_PREPARATION_PIPELINE = None
_PREPARATION_PIPELINE_KEY = None


def _take(iterator, count):
    values = []
    for _index in range(count):
        try:
            values.append(next(iterator))
        except StopIteration:
            break
    return values


def _execution_case_priority(case):
    return (
        0 if case.sample_kind == "lattice" else 1,
        case.sample_index,
        case.source_bucket.bucket_id,
    )


def _cpu_outcome_passed(outcome):
    return bool(
        outcome.support_gate_passed
        and outcome.teacher_feasible
        and outcome.fast_screen is not None
        and outcome.fast_screen.accepted
    )


def _prepare_and_screen_case(arguments):
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
        write_order9_r1_nominal_case_contract(
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
        _write_cpu_rejection(output_root, outcome)
        return outcome, None

    global _PREPARATION_PIPELINE, _PREPARATION_PIPELINE_KEY
    pipeline_key = (
        repository_text,
        source_manifest_text,
        minimum_joint_reserve,
        maximum_body_tilt,
    )
    if _PREPARATION_PIPELINE_KEY != pipeline_key:
        _PREPARATION_PIPELINE = Order9R1DeterministicTeacherScreenPipeline(
            repository_root=repository,
            source_bucket_manifest_path=(repository / source_manifest_text),
            minimum_normalized_joint_limit_reserve=minimum_joint_reserve,
            maximum_body_tilt_rad=maximum_body_tilt,
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
        _write_cpu_rejection(output_root, outcome)
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
        _write_cpu_rejection(output_root, outcome)
        return outcome, None
    persisted = materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=destination,
        repository_root=repository,
        approved_protocol_path=Path(base_path_text),
    )
    write_order9_r1_nominal_case_contract(
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


def _apply_replay_results(outcomes_by_id, replay_results):
    for candidate_id, replays in replay_results.items():
        outcome = outcomes_by_id[candidate_id]
        outcomes_by_id[candidate_id] = Order9R1CalibrationCaseOutcome(
            case=outcome.case,
            support_gate_passed=True,
            teacher_feasible=True,
            fast_screen=outcome.fast_screen,
            full_layer_replays=replays,
            failure_reason=next(
                (value.failure_reason for value in replays if not value.success),
                None,
            ),
        )


def _decisive_cpu_failure(protocol, outcomes):
    lattice = [value for value in outcomes if value.case.sample_kind == "lattice"]
    interior = [value for value in outcomes if value.case.sample_kind == "interior"]
    if any(
        not (
            value.support_gate_passed
            and value.teacher_feasible
            and value.fast_screen is not None
            and value.fast_screen.accepted
        )
        for value in lattice
    ):
        return "lattice_cpu_gate_cannot_reach_100_percent"

    def rate(successes, attempts):
        return 0.0 if attempts == 0 else successes / attempts

    teacher = sum(value.teacher_feasible for value in outcomes)
    screens = [value for value in outcomes if value.teacher_feasible]
    screen_success = sum(
        value.fast_screen is not None and value.fast_screen.accepted
        for value in screens
    )
    interior_teacher = sum(value.teacher_feasible for value in interior)
    interior_screens = [value for value in interior if value.teacher_feasible]
    interior_screen_success = sum(
        value.fast_screen is not None and value.fast_screen.accepted
        for value in interior_screens
    )
    gates = (
        (rate(teacher, len(outcomes)), protocol.minimum_teacher_feasibility_rate),
        (rate(screen_success, len(screens)), protocol.minimum_fast_screen_pass_rate),
        (
            rate(interior_teacher, len(interior)),
            protocol.minimum_interior_teacher_feasibility_rate,
        ),
        (
            rate(interior_screen_success, len(interior_screens)),
            protocol.minimum_interior_fast_screen_pass_rate,
        ),
    )
    if any(actual < minimum for actual, minimum in gates):
        return "cpu_rate_gate_failed"
    return None


def _decisive_isaac_failure(protocol, outcomes):
    attempted = [value for value in outcomes if value.full_layer_replays]
    if any(
        value.case.sample_kind == "lattice" and not value.passed for value in attempted
    ):
        return "lattice_isaac_gate_cannot_reach_100_percent"
    replays = [replay for value in attempted for replay in value.full_layer_replays]
    if any(replay.safety_failure for replay in replays):
        return "safety_failure_gate_exceeded"
    if any(replay.fallback_used for replay in replays):
        return "fallback_gate_exceeded"
    expected_replay_count = sum(
        (
            protocol.isaac_replays_per_lattice_case
            if value.case.sample_kind == "lattice"
            else protocol.isaac_replays_per_interior_case
        )
        for value in outcomes
    )
    maximum_success = sum(replay.success for replay in replays) + (
        expected_replay_count - len(replays)
    )
    if maximum_success / expected_replay_count < protocol.minimum_isaac_success_rate:
        return "isaac_success_rate_cannot_recover"
    interior_replays = [
        replay
        for value in attempted
        if value.case.sample_kind == "interior"
        for replay in value.full_layer_replays
    ]
    expected_interior_replays = (
        len([value for value in outcomes if value.case.sample_kind == "interior"])
        * protocol.isaac_replays_per_interior_case
    )
    maximum_interior_success = sum(replay.success for replay in interior_replays) + (
        expected_interior_replays - len(interior_replays)
    )
    if (
        maximum_interior_success / expected_interior_replays
        < protocol.minimum_interior_isaac_success_rate
    ):
        return "interior_isaac_success_rate_cannot_recover"
    return None


def _write_cpu_rejection(output_root, outcome):
    destination = output_root / "cpu_rejections" / f"{outcome.case.candidate_id}.json"
    write_order9_r1_clearance_diagnostic(
        {
            "result_version": "order9_r1_nominal_cpu_rejection_v2",
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
        destination,
    )


def _write_run_state(
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
):
    protocol_path = REPOSITORY / (
        "configs/training/order9_r1_nominal_calibration_protocol_v2.yaml"
    )
    approval_path = REPOSITORY / protocol["approval_record"]
    payload = {
        "result_version": "order9_r1_nominal_calibration_run_result_v2",
        "status": status,
        "execution_contract": ORDER9_R1_NOMINAL_EXECUTION_CONTRACT,
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
            "nominal_protocol": {
                "path": str(protocol_path.resolve()),
                "sha256": hash_file(protocol_path),
            },
            "nominal_approval": {
                "path": str(approval_path.resolve()),
                "sha256": hash_file(approval_path),
            },
            "base_numeric_protocol": protocol["base_numeric_protocol"],
            "protected_c3_curriculum": protocol["protected_c3_curriculum"],
            "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
            "protected_c3_rollout": protocol["protected_c3_rollout"],
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": hash_file(Path(__file__).resolve()),
            },
        },
        "approval_record_version": approval["record_version"],
    }
    return write_order9_r1_clearance_diagnostic(
        payload,
        output_root / "calibration_result.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
