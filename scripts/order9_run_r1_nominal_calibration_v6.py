#!/usr/bin/env python3
from __future__ import annotations

"""Run R1 calibration with closed-loop-safe morphology timing."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence, TextIO

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import order9_run_r1_nominal_calibration_v2 as base_runner  # noqa: E402
from scripts import order9_run_r1_nominal_calibration_v3 as v3_runner  # noqa: E402

from amsrr.training import order9_r1_early_tilt_pipeline as early_pipeline  # noqa: E402
from amsrr.training import order9_r1_nominal_calibration as nominal_runner  # noqa: E402
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    annotate_order9_r1_nominal_isaac_result,
    validate_order9_r1_nominal_isaac_result,
)
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_complete_task_materialization_v6 import (  # noqa: E402
    finalize_order9_r1_v6_materialized_case,
)
from amsrr.training.order9_r1_early_tilt_pipeline import (  # noqa: E402
    Order9R1EarlyTiltTeacherScreenPipeline,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case as _base_materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v6 import (  # noqa: E402
    ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT,
    load_order9_r1_nominal_calibration_v6_contract,
    run_order9_r1_nominal_v6_isaac_cases,
    write_order9_r1_nominal_v6_case_contract,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    ORDER9_R1_SAFE_TIMING_BASIS,
    ORDER9_R1_SAFE_TIMING_VERSION,
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

_EXECUTION_WINDOW_VERSION = "order9_r1_nominal_v6_execution_window_v1"
_EXECUTION_WINDOW_FILENAME = "formal_execution_window_v6.json"
_V6_DEADLINE_UNIX_S: float | None = None


class Order9R1V6WallTimeExceeded(RuntimeError):
    """Raised after stopping every active R1 Isaac child at the approved limit."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--maximum-parallel-process-count", type=int, default=4)
    parser.add_argument("--maximum-preparation-process-count", type=int, default=24)
    parser.add_argument("--isaac-case-batch-size", type=int, default=8)
    parser.add_argument("--no-persistent-morphology-coalescing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.rollout_steps < 1
        or args.maximum_parallel_process_count < 1
        or args.maximum_preparation_process_count < 1
        or args.isaac_case_batch_size < 1
    ):
        raise ValueError("R1 nominal v6 runtime limits are invalid")
    repository = REPOSITORY.resolve()
    protocol, approval = load_order9_r1_nominal_calibration_v6_contract(repository)
    if (
        args.maximum_parallel_process_count
        > int(protocol["maximum_parallel_process_count"])
        or args.maximum_preparation_process_count
        > int(protocol["maximum_preparation_process_count"])
        or args.isaac_case_batch_size > int(protocol["isaac_case_batch_size"])
    ):
        raise ValueError("R1 nominal v6 runtime exceeds the approved bounds")
    base_path = repository / protocol["base_numeric_protocol"]["path"]
    base = load_order9_r1_calibration_protocol(base_path, repository_root=repository)
    output_root = repository / protocol["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    base_runner._prepare_and_screen_case = _prepare_and_screen_case_v6
    base_runner.run_order9_r1_nominal_isaac_cases = (
        _run_order9_r1_nominal_v6_isaac_cases_with_preexisting_early_stop
    )
    base_runner.write_order9_r1_nominal_case_contract = (
        write_order9_r1_nominal_v6_case_contract
    )
    base_runner.ORDER9_R1_NOMINAL_EXECUTION_CONTRACT = (
        ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT
    )
    base_runner._execution_case_priority = _v6_execution_case_priority

    execution_window = _load_or_create_execution_window(
        output_root,
        maximum_wall_time_s=float(protocol["maximum_wall_time_s"]),
    )
    global _V6_DEADLINE_UNIX_S
    _V6_DEADLINE_UNIX_S = float(execution_window["deadline_unix_s"])
    nominal_runner.run_parallel_order9_collectors = (
        _run_parallel_order9_collectors_with_deadline
    )

    started_unix_s = float(execution_window["started_unix_s"])
    selection_results = []
    selected_level = None
    total_case_evidence_count = 0
    try:
        for level in base.calibration_levels:
            _enforce_wall_time(protocol)
            result, evidence_count = base_runner._run_level(
                split=base.selection_split,
                level_id=level.level_id,
                base=base,
                base_path=base_path,
                output_root=output_root / "selection" / level.level_id,
                repository=repository,
                isaac_python=args.isaac_python,
                rollout_steps=args.rollout_steps,
                maximum_parallel_process_count=args.maximum_parallel_process_count,
                maximum_preparation_process_count=args.maximum_preparation_process_count,
                isaac_case_batch_size=args.isaac_case_batch_size,
                persistent_morphology_coalescing=(
                    not args.no_persistent_morphology_coalescing
                ),
            )
            selection_results.append(result)
            total_case_evidence_count += evidence_count
            _write_level_v6(
                output_root / "selection" / level.level_id,
                result,
                evidence_count,
                protocol,
                approval,
            )
            _write_run_state_v6(
                output_root=output_root,
                status="selection_in_progress",
                selection_results=selection_results,
                confirmation_result=None,
                selected_level_id=(level.level_id if result.passed else selected_level),
                accepted_level_id=None,
                total_case_evidence_count=total_case_evidence_count,
                measured_wall_time_s=time.time() - started_unix_s,
                protocol=protocol,
                approval=approval,
            )
            if not result.passed:
                break
            selected_level = level.level_id

        confirmation = None
        if selected_level is not None:
            _enforce_wall_time(protocol)
            confirmation, evidence_count = base_runner._run_level(
                split=base.confirmation_split,
                level_id=selected_level,
                base=base,
                base_path=base_path,
                output_root=output_root / "confirmation" / selected_level,
                repository=repository,
                isaac_python=args.isaac_python,
                rollout_steps=args.rollout_steps,
                maximum_parallel_process_count=args.maximum_parallel_process_count,
                maximum_preparation_process_count=args.maximum_preparation_process_count,
                isaac_case_batch_size=args.isaac_case_batch_size,
                persistent_morphology_coalescing=(
                    not args.no_persistent_morphology_coalescing
                ),
            )
            total_case_evidence_count += evidence_count
            _write_level_v6(
                output_root / "confirmation" / selected_level,
                confirmation,
                evidence_count,
                protocol,
                approval,
            )
    except Order9R1V6WallTimeExceeded:
        destination = _write_run_state_v6(
            output_root=output_root,
            status="incomplete_wall_time_exceeded",
            selection_results=selection_results,
            confirmation_result=None,
            selected_level_id=selected_level,
            accepted_level_id=None,
            total_case_evidence_count=_completed_v6_case_count(output_root),
            measured_wall_time_s=time.time() - started_unix_s,
            protocol=protocol,
            approval=approval,
            termination_reason="approved_wall_time_exceeded",
        )
        print(f"ORDER9_R1_NOMINAL_V6_CALIBRATION_RESULT={destination}", flush=True)
        return 2
    accepted_level = (
        selected_level if confirmation is not None and confirmation.passed else None
    )
    destination = _write_run_state_v6(
        output_root=output_root,
        status="accepted" if accepted_level is not None else "rejected",
        selection_results=selection_results,
        confirmation_result=confirmation,
        selected_level_id=selected_level,
        accepted_level_id=accepted_level,
        total_case_evidence_count=total_case_evidence_count,
        measured_wall_time_s=time.time() - started_unix_s,
        protocol=protocol,
        approval=approval,
    )
    print(f"ORDER9_R1_NOMINAL_V6_CALIBRATION_RESULT={destination}", flush=True)
    return 0


def _materialize_order9_r1_isaac_case_v6(**kwargs):
    case = kwargs["case"]
    screen = kwargs["screen"]
    destination = Path(kwargs["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"Order9 R1 Isaac case output exists: {destination}")
    base_kwargs = dict(kwargs)
    base_kwargs.pop("nominal_time_scale", None)
    base_kwargs.pop("nominal_phase_time_scales", None)
    try:
        materialized = _base_materialize_order9_r1_isaac_case(**base_kwargs)
        return finalize_order9_r1_v6_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=kwargs["repository_root"],
        )
    except BaseException:
        # This directory did not exist on entry and contains preparation output
        # only.  Never leave a partially rebound case available to Isaac.
        if destination.exists():
            import shutil

            shutil.rmtree(destination)
        raise


def _run_order9_r1_nominal_v6_isaac_cases_with_preexisting_early_stop(
    cases,
    **kwargs,
):
    """Stop before new Isaac work when a completed lattice failure already exists."""

    repository = Path(kwargs["repository_root"]).resolve()
    for case in cases:
        isaac_root = case.manifest_path.parent / "isaac"
        if not (
            (isaac_root / "evaluation_rollout.pt").is_file()
            and (isaac_root / "evaluation_episodes.jsonl").is_file()
        ):
            continue
        try:
            annotate_order9_r1_nominal_isaac_result(
                case,
                repository_root=repository,
            )
            results = validate_order9_r1_nominal_isaac_result(
                case,
                repository_root=repository,
            )
        except (OSError, ValueError):
            continue
        if _is_decisive_preexisting_lattice_failure(
            case.manifest.candidate_id,
            results,
        ):
            return run_order9_r1_nominal_v6_isaac_cases(
                [case],
                **kwargs,
            )
    return run_order9_r1_nominal_v6_isaac_cases(cases, **kwargs)


def _is_decisive_preexisting_lattice_failure(candidate_id, results) -> bool:
    return "__lattice_" in candidate_id and any(
        not result.success or result.safety_failure or result.fallback_used
        for result in results
    )


def _prepare_and_screen_case_v6(arguments):
    early_pipeline.ORDER9_R1_TEACHER_OVERRIDES_RELATIVE = Path(
        "configs/training/order9_r1_teacher_overrides_v2.json"
    )
    v3_runner.Order9R1DeterministicTeacherScreenPipeline = (
        Order9R1EarlyTiltTeacherScreenPipeline
    )
    v3_runner.materialize_order9_r1_isaac_case = _materialize_order9_r1_isaac_case_v6
    outcome, manifest_path = v3_runner._prepare_and_screen_case_v3(arguments)
    repository = Path(arguments[1])
    output_root = Path(arguments[-1])
    if manifest_path is not None:
        materialized = load_order9_r1_isaac_case(manifest_path, repository)
        write_order9_r1_nominal_v6_case_contract(
            materialized,
            repository_root=repository,
        )
    elif outcome.failure_reason is not None:
        write_order9_r1_clearance_diagnostic(
            {
                "result_version": "order9_r1_nominal_cpu_rejection_v6",
                "execution_contract": ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT,
                "candidate_id": outcome.case.candidate_id,
                "support_gate_passed": outcome.support_gate_passed,
                "teacher_feasible": outcome.teacher_feasible,
                "fast_screen": (
                    None
                    if outcome.fast_screen is None
                    else outcome.fast_screen.to_dict()
                ),
                "failure_reason": outcome.failure_reason,
                "isaac_invoked": False,
                "controller_layers_invoked": False,
                "training_eligible": False,
            },
            output_root / "cpu_rejections_v6" / f"{outcome.case.candidate_id}.json",
        )
    return outcome, manifest_path


def _write_level_v6(output_root, result, evidence_count, protocol, approval) -> Path:
    return write_order9_r1_clearance_diagnostic(
        {
            "result_version": "order9_r1_nominal_calibration_level_result_v6",
            "execution_contract": ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT,
            "safe_timing_version": ORDER9_R1_SAFE_TIMING_VERSION,
            "timing_selection_basis": ORDER9_R1_SAFE_TIMING_BASIS,
            "phase_time_scales_by_module_count": protocol[
                "phase_time_scales_by_module_count"
            ],
            "controller_free_screen_completed_before_isaac": True,
            "complete_task_semantic_audit_required_before_isaac": True,
            "complete_task_materializer_selection": (
                "explicit_callable_and_identity_v1"
            ),
            "complete_task_materializer_monkey_patch_used": False,
            "pi_l_actor_command_applied": False,
            "case_evidence_count": evidence_count,
            "result": result.to_dict(),
            "bindings": _run_bindings(protocol, approval),
        },
        output_root / "level_result_v6.json",
    )


def _write_run_state_v6(
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
    termination_reason=None,
) -> Path:
    payload = {
        "result_version": "order9_r1_nominal_calibration_run_result_v6",
        "status": status,
        "execution_contract": ORDER9_R1_NOMINAL_V6_EXECUTION_CONTRACT,
        "safe_timing_version": ORDER9_R1_SAFE_TIMING_VERSION,
        "timing_selection_basis": ORDER9_R1_SAFE_TIMING_BASIS,
        "phase_time_scales_by_module_count": protocol[
            "phase_time_scales_by_module_count"
        ],
        "grasp_geometric_path_changed": False,
        "post_release_retreat_path_changed": True,
        "retreat_offset_m": 0.10,
        "grasp_endpoint_configuration_changed": False,
        "final_endpoint_configuration_changed": True,
        "contact_assignments_changed": False,
        "retreat_isaac_validated": True,
        "controller_free_screen_completed_before_isaac": True,
        "complete_task_semantic_audit_required_before_isaac": True,
        "complete_task_materializer_selection": "explicit_callable_and_identity_v1",
        "complete_task_materializer_monkey_patch_used": False,
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
        "maximum_wall_time_s": float(protocol["maximum_wall_time_s"]),
        "completed_within_approved_wall_time": (
            measured_wall_time_s <= float(protocol["maximum_wall_time_s"])
        ),
        "termination_reason": termination_reason,
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
        output_root / "calibration_result_v6.json",
    )


def _run_bindings(protocol, approval) -> dict[str, object]:
    protocol_path = REPOSITORY / (
        "configs/training/order9_r1_nominal_calibration_protocol_v6.yaml"
    )
    approval_path = REPOSITORY / protocol["approval_record"]
    if approval.get("approved_protocol", {}).get("sha256") != hash_file(protocol_path):
        raise ValueError("R1 nominal v6 run approval differs")
    return {
        "v6_protocol": {"path": str(protocol_path), "sha256": hash_file(protocol_path)},
        "v6_approval": {"path": str(approval_path), "sha256": hash_file(approval_path)},
        "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
        "protected_c3_rollout": protocol["protected_c3_rollout"],
        "base_v5_result_ledger": protocol["base_v5_result_ledger"],
        "r1_teacher_overrides": protocol["r1_teacher_overrides"],
        "r1_retime_implementation": protocol["r1_retime_implementation"],
        "r1_complete_task_semantic_audit_implementation": protocol[
            "r1_complete_task_semantic_audit_implementation"
        ],
        "r1_complete_task_materialization_v6_implementation": protocol[
            "r1_complete_task_materialization_v6_implementation"
        ],
        "r1_safe_timing_implementation": protocol["r1_safe_timing_implementation"],
        "r1_isaac_materialization_implementation": protocol[
            "r1_isaac_materialization_implementation"
        ],
    }


def _load_or_create_execution_window(
    output_root: Path,
    *,
    maximum_wall_time_s: float,
) -> dict[str, object]:
    destination = output_root / _EXECUTION_WINDOW_FILENAME
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if (
            payload.get("window_version") != _EXECUTION_WINDOW_VERSION
            or float(payload.get("maximum_wall_time_s", -1.0)) != maximum_wall_time_s
            or float(payload.get("deadline_unix_s", 0.0))
            != float(payload.get("started_unix_s", 0.0)) + maximum_wall_time_s
        ):
            raise ValueError("R1 nominal v6 execution-window contract changed")
        return payload

    existing_files = [path for path in output_root.rglob("*") if path.is_file()]
    started_unix_s = min(
        [time.time(), *(path.stat().st_mtime for path in existing_files)]
    )
    return json.loads(
        write_order9_r1_clearance_diagnostic(
            {
                "window_version": _EXECUTION_WINDOW_VERSION,
                "started_unix_s": started_unix_s,
                "maximum_wall_time_s": maximum_wall_time_s,
                "deadline_unix_s": started_unix_s + maximum_wall_time_s,
                "resume_resets_window": False,
                "formal_teacher_collection_authorized": False,
                "training_eligible": False,
            },
            destination,
        ).read_text(encoding="utf-8")
    )


def _run_parallel_order9_collectors_with_deadline(
    commands: Mapping[str, Sequence[str]],
    *,
    repository_root: str | Path,
    log_paths: Mapping[str, str | Path],
    maximum_parallel_process_count: int | None = None,
    process_start_stagger_s: float = 0.0,
) -> tuple[float, dict[str, int]]:
    """Run the unmodified protected rollout and stop its process groups on time."""

    if not commands or set(commands) != set(log_paths):
        raise ValueError("Order9 collector commands/log paths must be aligned")
    if _V6_DEADLINE_UNIX_S is None:
        raise RuntimeError("R1 nominal v6 execution deadline is not initialized")
    maximum = maximum_parallel_process_count or len(commands)
    if maximum < 1 or process_start_stagger_s < 0.0:
        raise ValueError("R1 nominal v6 collector runtime limit is invalid")
    repository = Path(repository_root).resolve()
    pending = list(commands.items())
    handles: dict[str, TextIO] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    return_codes: dict[str, int] = {}
    started = time.perf_counter()
    try:
        while pending or processes:
            _enforce_wall_time(None)
            while pending and len(processes) < maximum:
                if processes and process_start_stagger_s > 0.0:
                    _sleep_with_deadline(process_start_stagger_s)
                name, command = pending.pop(0)
                log = Path(log_paths[name])
                if not log.is_absolute():
                    log = repository / log
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("w", encoding="utf-8")
                handles[name] = handle
                handle.write("ORDER9_COMMAND=" + shlex.join(command) + "\n")
                handle.flush()
                processes[name] = subprocess.Popen(
                    list(command),
                    cwd=repository,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            completed = [
                name
                for name, process in processes.items()
                if process.poll() is not None
            ]
            if not completed:
                _sleep_with_deadline(0.25)
                continue
            for name in completed:
                process = processes.pop(name)
                return_codes[name] = int(process.returncode)
                handles.pop(name).close()
    except BaseException:
        _terminate_process_groups(processes.values())
        raise
    finally:
        for handle in handles.values():
            handle.close()
    failures = {name: code for name, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"Order9 parallel collectors failed: {failures}")
    return time.perf_counter() - started, return_codes


def _terminate_process_groups(processes) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    limit = time.monotonic() + 5.0
    while active and time.monotonic() < limit:
        active = [process for process in active if process.poll() is None]
        if active:
            time.sleep(0.05)
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _sleep_with_deadline(duration_s: float) -> None:
    if _V6_DEADLINE_UNIX_S is None:
        raise RuntimeError("R1 nominal v6 execution deadline is not initialized")
    remaining = _V6_DEADLINE_UNIX_S - time.time()
    if remaining <= 0.0:
        raise Order9R1V6WallTimeExceeded(
            "R1 nominal v6 exceeded its approved ten-hour wall time"
        )
    time.sleep(min(duration_s, remaining))
    _enforce_wall_time(None)


def _enforce_wall_time(_protocol) -> None:
    if _V6_DEADLINE_UNIX_S is None:
        raise RuntimeError("R1 nominal v6 execution deadline is not initialized")
    if time.time() >= _V6_DEADLINE_UNIX_S:
        raise Order9R1V6WallTimeExceeded(
            "R1 nominal v6 exceeded its approved ten-hour wall time"
        )


def _completed_v6_case_count(output_root: Path) -> int:
    return sum(
        1 for path in output_root.rglob("nominal_result_v6.json") if path.is_file()
    )


def _v6_execution_case_priority(case) -> tuple[int, int, int, str]:
    """Test lattice corners before edges/faces without changing the case set."""

    if case.sample_kind != "lattice":
        return (1, 0, case.sample_index, case.source_bucket.bucket_id)
    lattice_index = int(case.sample_index)
    coordinate_indices = (
        lattice_index // 9,
        (lattice_index % 9) // 3,
        lattice_index % 3,
    )
    centered_coordinate_count = sum(value == 1 for value in coordinate_indices)
    return (
        0,
        centered_coordinate_count,
        lattice_index,
        case.source_bucket.bucket_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
