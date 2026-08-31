#!/usr/bin/env python3
from __future__ import annotations

"""Run the lightweight and bounded-control gates for one R1 range's corners."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_range_selection_v10 import (  # noqa: E402
    ORDER9_R1_RANGE_LEVEL_IDS,
    range_case_priority,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts.order9_run_r1_range_selection_v10 import (  # noqa: E402
    BASE_PROTOCOL,
    OUTPUT_ROOT,
    _atomic_json,
    _binding,
    _execute_level,
    _ensure_python_hash_seed_zero,
    _load_contract,
    _prepare_case,
    _validate_prepared_case,
)

PROBE_VERSION = "order9_r1_range_corner_probe_v10"
EXPECTED_SOURCE_COUNT = 22
EXPECTED_CORNER_COUNT_PER_SOURCE = 8
EXPECTED_CANDIDATE_COUNT = EXPECTED_SOURCE_COUNT * EXPECTED_CORNER_COUNT_PER_SOURCE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-id", required=True, choices=ORDER9_R1_RANGE_LEVEL_IDS)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-preparation-process-count", type=int, default=24)
    parser.add_argument("--maximum-parallel-batches", type=int, default=4)
    parser.add_argument("--maximum-wall-time-s", type=float, default=36000.0)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return parser


def _archive_previous(path: Path) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = hash_file(path)[:16]
    _atomic_json(path.parent / "history" / f"{path.stem}_{digest}.json", payload)


def _corner_cases(level_id: str):
    cases = list(
        enumerate_order9_r1_calibration_level_cases(
            protocol_path=BASE_PROTOCOL,
            repository_root=REPOSITORY,
            level_id=level_id,
            split="train",
        )
    )
    corners = [
        case
        for case in cases
        if range_case_priority(case.candidate_id, case.sample_kind, case.sample_index)[
            0
        ]
        == 0
    ]
    corners.sort(
        key=lambda case: range_case_priority(
            case.candidate_id, case.sample_kind, case.sample_index
        )
    )
    if (
        len(corners) != EXPECTED_CANDIDATE_COUNT
        or len({case.source_bucket.bucket_id for case in corners})
        != EXPECTED_SOURCE_COUNT
    ):
        raise RuntimeError("R1 range corner candidate set differs")
    return corners


def _prepare_corners(
    level_id: str,
    *,
    output_root: Path,
    maximum_process_count: int,
):
    cases = _corner_cases(level_id)
    preparation_root = output_root / "prepared" / level_id
    records = {}
    pending = []
    for case in cases:
        destination = preparation_root / case.candidate_id
        try:
            records[case.candidate_id] = _validate_prepared_case(case, destination)
        except (OSError, RuntimeError, ValueError):
            if destination.exists():
                shutil.rmtree(destination)
            pending.append(case)

    arguments = [(case, str(BASE_PROTOCOL), str(preparation_root)) for case in pending]
    with ProcessPoolExecutor(max_workers=maximum_process_count) as executor:
        futures = {
            executor.submit(_prepare_case, value): value[0] for value in arguments
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            records[case.candidate_id] = future.result()
            print(
                "ORDER9_R1_RANGE_CORNER_PREPARATION_PROGRESS="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "completed_pending": completed,
                        "pending_total": len(pending),
                        "total_resolved": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    ordered = [records[case.candidate_id] for case in cases]
    path = output_root / "corner_probe" / level_id / "preparation.json"
    _archive_previous(path)
    path = _atomic_json(
        path,
        {
            "record_version": "order9_r1_range_corner_preparation_v10",
            "level_id": level_id,
            "candidate_count": len(ordered),
            "prepared_count": sum(bool(record["prepared"]) for record in ordered),
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "entries": ordered,
        },
    )
    return cases, ordered, path


def main() -> int:
    _ensure_python_hash_seed_zero()
    args = _parser().parse_args()
    if (
        not 1 <= args.maximum_preparation_process_count <= 24
        or not 1 <= args.maximum_parallel_batches <= 4
        or args.maximum_wall_time_s <= 0.0
        or args.rollout_steps < 1
    ):
        raise ValueError("R1 range corner runtime limits are invalid")
    protocol, approval = _load_contract()
    output_root = Path(args.output_root).resolve()
    started = time.time()
    deadline = started + min(
        float(args.maximum_wall_time_s), float(protocol["maximum_wall_time_s"])
    )
    cases, preparation, preparation_path = _prepare_corners(
        args.level_id,
        output_root=output_root,
        maximum_process_count=args.maximum_preparation_process_count,
    )
    lightweight_passed = all(bool(record["prepared"]) for record in preparation)
    if lightweight_passed:
        entries = _execute_level(
            args.level_id,
            cases=cases,
            preparation_records=preparation,
            output_root=output_root,
            isaac_python=args.isaac_python,
            rollout_steps=args.rollout_steps,
            maximum_parallel_batches=args.maximum_parallel_batches,
            deadline=deadline,
        )
    else:
        entries = [dict(record) for record in preparation]
    control_passed = bool(
        lightweight_passed
        and all(
            entry.get("valid_evidence") is True
            and entry.get("candidate_passed") is True
            and int(entry.get("episode_count", -1)) == 2
            and int(entry.get("success_count", -1)) == 2
            and int(entry.get("safety_failure_count", -1)) == 0
            and int(entry.get("fallback_count", -1)) == 0
            for entry in entries
        )
    )
    result_path = output_root / "corner_probe" / args.level_id / "result.json"
    _archive_previous(result_path)
    result_path = _atomic_json(
        result_path,
        {
            "result_version": "order9_r1_range_corner_probe_result_v10",
            "probe_version": PROBE_VERSION,
            "level_id": args.level_id,
            "status": "passed" if control_passed else "failed",
            "candidate_count": len(entries),
            "episode_count": sum(
                int(entry.get("episode_count", 0)) for entry in entries
            ),
            "success_count": sum(
                int(entry.get("success_count", 0)) for entry in entries
            ),
            "safety_failure_count": sum(
                int(entry.get("safety_failure_count", 0)) for entry in entries
            ),
            "fallback_count": sum(
                int(entry.get("fallback_count", 0)) for entry in entries
            ),
            "lightweight_passed": lightweight_passed,
            "control_passed": control_passed,
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "elapsed_s": time.time() - started,
            "bindings": {
                "protocol": _binding(
                    REPOSITORY
                    / "configs/training/order9_r1_range_selection_protocol_v10.json"
                ),
                "approval": _binding(
                    REPOSITORY
                    / "for_codex/R1_RANGE_SELECTION_PROTOCOL_V10_APPROVAL.json"
                ),
                "preparation": _binding(preparation_path),
                "probe_implementation": _binding(Path(__file__)),
            },
            "approval_record_version": approval["record_version"],
            "entries": entries,
        },
    )
    print(
        "ORDER9_R1_RANGE_CORNER_PROBE_COMPLETE="
        + json.dumps(
            {
                "level_id": args.level_id,
                "status": "passed" if control_passed else "failed",
                "result": str(result_path),
                "sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
