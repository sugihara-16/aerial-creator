#!/usr/bin/env python3
from __future__ import annotations

"""Replay all 682 R1 minimum-level cases with the approved v7 repair."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    Order9R1CalibrationCaseOutcome,
    _summarize_level,
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_fast_screen import (  # noqa: E402
    Order9R1FastScreenResult,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v7 import (  # noqa: E402
    ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT,
    load_order9_r1_nominal_calibration_v7_contract,
    run_order9_r1_nominal_v7_isaac_cases,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

_RESULT_VERSION = "order9_r1_nominal_calibration_minimum_level_result_v7"
_RUN_VERSION = "order9_r1_nominal_calibration_minimum_level_run_v7"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--maximum-parallel-process-count", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.rollout_steps < 1 or not 1 <= args.maximum_parallel_process_count <= 4:
        raise ValueError("R1 nominal v7 runtime limits are invalid")
    repository = REPOSITORY.resolve()
    protocol, approval = load_order9_r1_nominal_calibration_v7_contract(repository)
    base_path = repository / protocol["base_numeric_protocol"]["path"]
    base = load_order9_r1_calibration_protocol(base_path, repository_root=repository)
    level_id = str(protocol["base_level_id"])
    split = str(protocol["base_split"])
    level = next(
        value for value in base.calibration_levels if value.level_id == level_id
    )
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=base_path,
        repository_root=repository,
        level_id=level_id,
        split=split,
    )
    expected_count = int(protocol["source_bucket_count"]) * int(
        protocol["candidate_count_per_bucket"]
    )
    if len(cases) != expected_count:
        raise SchemaValidationError("R1 nominal v7 case enumeration count differs")
    output_root = repository / protocol["output_root"]
    source_root = repository / protocol["source_v6_level_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    materialized = []
    outcomes_by_id = {}
    source_bindings = []
    for index, case in enumerate(cases, start=1):
        source_case_root = source_root / case.candidate_id
        source_manifest = source_case_root / "case_manifest.json"
        source_screen = source_case_root / "fast_screen.json"
        source_contract = source_case_root / "nominal_contract_v6.json"
        for required in (source_manifest, source_screen, source_contract):
            if not required.is_file():
                raise FileNotFoundError(required)
        destination_root = output_root / case.candidate_id
        destination_manifest = destination_root / "case_manifest.json"
        if not destination_root.exists():
            destination_root.mkdir(parents=True)
            shutil.copy2(source_manifest, destination_manifest)
        elif not destination_manifest.is_file():
            raise SchemaValidationError(
                f"R1 nominal v7 partial case directory exists: {destination_root}"
            )
        elif destination_manifest.read_bytes() != source_manifest.read_bytes():
            raise SchemaValidationError(
                f"R1 nominal v7 manifest copy changed: {case.candidate_id}"
            )
        loaded = load_order9_r1_isaac_case(destination_manifest, repository)
        if (
            loaded.manifest.candidate_id != case.candidate_id
            or loaded.manifest.source_bucket_id != case.source_bucket.bucket_id
            or loaded.manifest.morphology_hash != case.source_bucket.morphology_hash
        ):
            raise SchemaValidationError("R1 nominal v7 case identity differs")
        screen = _load_fast_screen(source_screen)
        if (
            not screen.accepted
            or not screen.eligible_for_full_control_test
            or screen.candidate_id != case.candidate_id
            or screen.isaac_invoked
            or screen.controller_layers_invoked
        ):
            raise SchemaValidationError("R1 nominal v7 reused screen is not admitted")
        outcomes_by_id[case.candidate_id] = Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=True,
            teacher_feasible=True,
            fast_screen=screen,
            full_layer_replays=(),
            failure_reason=None,
        )
        materialized.append(loaded)
        source_bindings.append(
            {
                "candidate_id": case.candidate_id,
                "source_manifest_path": str(source_manifest.relative_to(repository)),
                "source_manifest_sha256": hash_file(source_manifest),
                "v7_manifest_path": str(destination_manifest.relative_to(repository)),
                "v7_manifest_sha256": hash_file(destination_manifest),
                "fast_screen_path": str(source_screen.relative_to(repository)),
                "fast_screen_sha256": hash_file(source_screen),
            }
        )
        if index % 100 == 0 or index == len(cases):
            print(
                "ORDER9_R1_NOMINAL_V7_REUSE_PROGRESS="
                + json.dumps({"completed": index, "total": len(cases)}),
                flush=True,
            )
    reuse_manifest = write_order9_r1_clearance_diagnostic(
        {
            "manifest_version": "order9_r1_nominal_v7_v6_input_reuse_manifest_v1",
            "case_count": len(source_bindings),
            "v6_teacher_trajectory_bytes_reused": True,
            "controller_free_screen_success_count": len(source_bindings),
            "isaac_invoked_while_reusing_inputs": False,
            "entries": source_bindings,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
        },
        output_root / "v6_input_reuse_manifest_v7.json",
    )
    print(
        "ORDER9_R1_NOMINAL_V7_ISAAC_START="
        + json.dumps(
            {
                "case_count": len(materialized),
                "episode_count": len(materialized) * 2,
                "morphology_count": len(
                    {value.manifest.morphology_hash for value in materialized}
                ),
                "maximum_parallel_process_count": args.maximum_parallel_process_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    replay_results = run_order9_r1_nominal_v7_isaac_cases(
        materialized,
        repository_root=repository,
        python_executable=args.isaac_python,
        rollout_steps=args.rollout_steps,
        maximum_parallel_process_count=args.maximum_parallel_process_count,
    )
    for case in cases:
        replays = replay_results[case.candidate_id]
        outcomes_by_id[case.candidate_id] = Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=True,
            teacher_feasible=True,
            fast_screen=outcomes_by_id[case.candidate_id].fast_screen,
            full_layer_replays=replays,
            failure_reason=(
                None
                if all(value.success for value in replays)
                else next(
                    (value.failure_reason for value in replays if not value.success),
                    "task failed",
                )
            ),
        )
    outcomes = [outcomes_by_id[case.candidate_id] for case in cases]
    result = _summarize_level(
        base,
        level,
        split,
        len({case.source_bucket.bucket_id for case in cases}),
        outcomes,
    )
    measured = max(perf_counter() - started, 1.0e-12)
    level_path = write_order9_r1_clearance_diagnostic(
        {
            "result_version": _RESULT_VERSION,
            "status": "passed" if result.passed else "rejected",
            "execution_contract": ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT,
            "result": result.to_dict(),
            "v6_teacher_trajectory_bytes_reused": True,
            "controller_free_screen_case_count": len(cases),
            "pi_l_actor_command_applied": False,
            "minimum_level_only": True,
            "higher_calibration_levels_executed": False,
            "held_out_confirmation_executed": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
            "measured_wall_time_s": measured,
            "bindings": {
                **_run_bindings(protocol, approval, repository),
                "v6_input_reuse_manifest": _binding(reuse_manifest),
            },
        },
        output_root / "level_result_v7.json",
    )
    run_path = write_order9_r1_clearance_diagnostic(
        {
            "run_version": _RUN_VERSION,
            "status": "minimum_level_passed" if result.passed else "rejected",
            "execution_contract": ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT,
            "minimum_level_id": level_id,
            "minimum_level_passed": result.passed,
            "completed_case_count": len(cases),
            "completed_episode_count": sum(
                len(value.full_layer_replays) for value in outcomes
            ),
            "success_episode_count": sum(
                replay.success
                for value in outcomes
                for replay in value.full_layer_replays
            ),
            "safety_failure_count": sum(
                replay.safety_failure
                for value in outcomes
                for replay in value.full_layer_replays
            ),
            "fallback_count": sum(
                replay.fallback_used
                for value in outcomes
                for replay in value.full_layer_replays
            ),
            "additional_compression_mm_by_source_bucket": protocol[
                "additional_compression_mm_by_source_bucket"
            ],
            "default_additional_compression_mm": protocol[
                "default_additional_compression_mm"
            ],
            "v6_teacher_trajectory_bytes_reused": True,
            "pi_l_actor_command_applied": False,
            "minimum_level_only": True,
            "higher_calibration_levels_executed": False,
            "held_out_confirmation_executed": False,
            "accepted_distribution_published": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
            "measured_wall_time_s": measured,
            "maximum_wall_time_s": float(protocol["maximum_wall_time_s"]),
            "completed_within_approved_wall_time": measured
            <= float(protocol["maximum_wall_time_s"]),
            "bindings": {
                **_run_bindings(protocol, approval, repository),
                "level_result": _binding(level_path),
                "v6_input_reuse_manifest": _binding(reuse_manifest),
            },
        },
        output_root / "calibration_minimum_level_result_v7.json",
    )
    print(f"ORDER9_R1_NOMINAL_V7_RESULT={run_path}", flush=True)
    return 0 if result.passed else 2


def _load_fast_screen(path: Path) -> Order9R1FastScreenResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checked_phases"] = tuple(payload["checked_phases"])
    payload["violation_codes"] = tuple(payload["violation_codes"])
    return Order9R1FastScreenResult(**payload)


def _run_bindings(protocol, approval, repository: Path) -> dict[str, object]:
    protocol_path = repository / (
        "configs/training/order9_r1_nominal_calibration_protocol_v7.yaml"
    )
    approval_path = repository / protocol["approval_record"]
    if approval["approved_protocol"]["sha256"] != hash_file(protocol_path):
        raise SchemaValidationError("R1 nominal v7 approval binding changed")
    return {
        "v7_protocol": _binding(protocol_path),
        "v7_approval": _binding(approval_path),
        "base_v6_result_ledger": protocol["base_v6_result_ledger"],
        "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
        "protected_c3_rollout": protocol["protected_c3_rollout"],
        "selected_repair_raw_evidence": protocol["selected_repair_raw_evidence"],
        "selected_repair_episode_evidence": protocol[
            "selected_repair_episode_evidence"
        ],
        "compression_wrapper_implementation": protocol[
            "compression_wrapper_implementation"
        ],
    }


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    return {"path": str(source), "sha256": hash_file(source)}


if __name__ == "__main__":
    raise SystemExit(main())
