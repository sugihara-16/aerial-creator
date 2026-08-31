#!/usr/bin/env python3
from __future__ import annotations

"""Prepare missing R1 v13 candidates while named repairs run separately."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import sys

for _thread_environment_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import order9_run_r1_range_selection_v10 as runtime_v10
from scripts import order9_run_r1_range_selection_v11 as runtime_v11
from scripts import order9_run_r1_range_selection_v13 as runtime_v13


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="r1_l2_20mm_10deg")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--maximum-process-count", type=int, default=24)
    parser.add_argument("--exclude-candidate-id", action="append", default=[])
    parser.add_argument(
        "--prioritize-isaac-chunks",
        action="store_true",
        help=(
            "Prepare missing members of the most-complete four-candidate "
            "Isaac chunks first. This changes scheduling only."
        ),
    )
    return parser


def _isaac_chunk_priority(cases, valid_candidate_ids: frozenset[str]):
    """Return a scheduling key that closes partially prepared Isaac chunks."""

    chunk_by_candidate_id = {}
    by_source = {}
    for case in cases:
        by_source.setdefault(case.source_bucket.bucket_id, []).append(case)
    for source_id, source_cases in sorted(by_source.items()):
        source_cases.sort(
            key=lambda case: runtime_v10.range_case_priority(
                case.candidate_id,
                case.sample_kind,
                case.sample_index,
            )
        )
        for chunk_index, candidate_ids in enumerate(
            runtime_v10.chunk_candidate_ids(
                tuple(case.candidate_id for case in source_cases)
            )
        ):
            prepared_count = sum(
                candidate_id in valid_candidate_ids for candidate_id in candidate_ids
            )
            for within_chunk_index, candidate_id in enumerate(candidate_ids):
                chunk_by_candidate_id[candidate_id] = (
                    -prepared_count,
                    source_id,
                    chunk_index,
                    within_chunk_index,
                )
    return lambda case: chunk_by_candidate_id[case.candidate_id]


def main() -> int:
    arguments = _parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "R1 v13 preparation requires PYTHONHASHSEED=0 before interpreter start"
        )
    if not 1 <= arguments.maximum_process_count <= 24:
        raise ValueError("R1 remaining preparation process count differs")
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    output_root = runtime_v13.OUTPUT_ROOT
    prepared_root = output_root / "prepared" / arguments.level
    excluded = frozenset(arguments.exclude_candidate_id)
    cases = list(runtime_v11._corrected_cases(arguments.level, arguments.split))
    cases.sort(
        key=lambda case: runtime_v10.range_case_priority(
            case.candidate_id,
            case.sample_kind,
            case.sample_index,
        )
    )
    missing = []
    valid_count = 0
    valid_candidate_ids = set()
    for case in cases:
        if case.candidate_id in excluded:
            continue
        destination = prepared_root / case.candidate_id
        if destination.is_dir():
            try:
                runtime_v13._validate_prepared_case(case, destination)
                valid_count += 1
                valid_candidate_ids.add(case.candidate_id)
                continue
            except (OSError, RuntimeError, ValueError):
                pass
        missing.append(case)

    if arguments.prioritize_isaac_chunks:
        missing.sort(
            key=_isaac_chunk_priority(cases, frozenset(valid_candidate_ids))
        )

    records = []
    decisive = None
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(arguments.maximum_process_count, len(missing)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    runtime_v13._prepare_case, (case, str(output_root))
                ): case
                for case in missing
            }
            while futures:
                completed, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    case = futures.pop(future)
                    record = future.result()
                    records.append(record)
                    if case.sample_kind == "lattice" and not record["prepared"]:
                        decisive = record
                if decisive is not None:
                    for future in futures:
                        future.cancel()
                    break
                if (
                    len(records) % arguments.maximum_process_count == 0
                    or decisive is not None
                    or not futures
                ):
                    print(
                        "ORDER9_R1_V13_REMAINING_PREPARATION_PROGRESS="
                        + json.dumps(
                            {
                                "level_id": arguments.level,
                                "split": arguments.split,
                                "previously_valid": valid_count,
                                "completed_missing": len(records),
                                "total_missing": len(missing),
                                "decisive_candidate_id": (
                                    None
                                    if decisive is None
                                    else decisive["candidate_id"]
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    print(
        "ORDER9_R1_V13_REMAINING_PREPARATION_COMPLETE="
        + json.dumps(
            {
                "level_id": arguments.level,
                "split": arguments.split,
                "previously_valid": valid_count,
                "prepared_now": sum(bool(value["prepared"]) for value in records),
                "decisive_record": decisive,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if decisive is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
