#!/usr/bin/env python3
from __future__ import annotations

"""Run ready four-candidate R1 v13 Isaac chunks while CPU preparation continues."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import order9_run_r1_range_selection_v13 as runtime_v13

runtime_v10 = runtime_v13.runtime_v10
runtime_v11 = runtime_v13.runtime_v11


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="r1_l2_20mm_10deg")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--maximum-parallel-batches", type=int, default=4)
    parser.add_argument("--maximum-wall-time-s", type=float, default=7200.0)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        default=[],
        help="Source buckets handled by a separate confirmation run.",
    )
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    return parser


def _chunk_specs(cases):
    by_source = {}
    for case in cases:
        by_source.setdefault(case.source_bucket.bucket_id, []).append(case)
    result = []
    for source_id, source_cases in sorted(by_source.items()):
        source_cases.sort(
            key=lambda case: runtime_v10.range_case_priority(
                case.candidate_id, case.sample_kind, case.sample_index
            )
        )
        for chunk_index, candidate_ids in enumerate(
            runtime_v10.chunk_candidate_ids(
                tuple(case.candidate_id for case in source_cases)
            )
        ):
            digest = hashlib.sha256("\n".join(candidate_ids).encode()).hexdigest()[:12]
            result.append((source_id, chunk_index, digest, tuple(candidate_ids)))
    return result


def main() -> int:
    arguments = _parser().parse_args()
    if not 1 <= arguments.maximum_parallel_batches <= 4:
        raise ValueError("R1 streaming batch count differs")
    runtime_v13._load_contract()
    runtime_v13._install_v13_preparation_hooks()
    excluded_sources = frozenset(arguments.exclude_source_id)
    cases = [
        case
        for case in runtime_v11._corrected_cases(arguments.level, arguments.split)
        if case.source_bucket.bucket_id not in excluded_sources
    ]
    case_by_id = {case.candidate_id: case for case in cases}
    specs = _chunk_specs(cases)
    preparation_root = runtime_v13.OUTPUT_ROOT / "prepared" / arguments.level
    isaac_root = runtime_v13.OUTPUT_ROOT / "isaac" / arguments.level
    deadline = time.time() + arguments.maximum_wall_time_s
    completed = set()
    decisive = None

    while time.time() < deadline and len(completed) < len(specs):
        ready = []
        restart_confirmation_candidates = {}
        for source_id, chunk_index, digest, candidate_ids in specs:
            key = (source_id, chunk_index, digest)
            if key in completed:
                continue
            candidate_root = (
                isaac_root / "batches" / source_id / f"chunk_{chunk_index:03d}_{digest}"
            )
            chunk_complete = True
            chunk_failed = []
            for candidate_id in candidate_ids:
                single = runtime_v10._inspect_candidate_output(
                    candidate_id,
                    isaac_root / "single" / candidate_id,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                bounded = runtime_v10._inspect_candidate_output(
                    candidate_id,
                    candidate_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                evidence = single if single.get("valid_evidence") else bounded
                if not evidence.get("valid_evidence"):
                    chunk_complete = False
                elif not evidence.get("candidate_passed"):
                    chunk_failed.append((candidate_id, evidence, single, bounded))
            if chunk_complete and not chunk_failed:
                completed.add(key)
                continue
            if chunk_failed:
                # A bounded failure is not decisive until independently replayed.
                needs_single = [
                    value
                    for value in chunk_failed
                    if not value[2].get("valid_evidence")
                ]
                for candidate_id, _evidence, _single, _bounded in needs_single:
                    restart_confirmation_candidates[candidate_id] = source_id
                if not needs_single:
                    for candidate_id, evidence, _single, _bounded in chunk_failed:
                        case = case_by_id[candidate_id]
                        if (
                            case.sample_kind == "lattice"
                            or evidence.get("safety_failure_count", 0)
                            or evidence.get("fallback_count", 0)
                        ):
                            decisive = evidence
                            break
                    if decisive is not None:
                        break
            prepared = []
            for candidate_id in candidate_ids:
                case = case_by_id[candidate_id]
                destination = preparation_root / candidate_id
                try:
                    runtime_v13._validate_prepared_case(case, destination)
                    prepared.append(candidate_id)
                except (OSError, RuntimeError, ValueError):
                    break
            if len(prepared) == len(candidate_ids):
                ready.append((source_id, chunk_index, digest, candidate_ids))
        if decisive is not None:
            break
        if restart_confirmation_candidates:
            confirmation_work = []
            for candidate_id, source_id in restart_confirmation_candidates.items():
                single_root = isaac_root / "single" / candidate_id
                job = runtime_v10._job_for_case(
                    case_by_id[candidate_id],
                    preparation_root=preparation_root,
                    isaac_python=arguments.isaac_python,
                    rollout_steps=arguments.rollout_steps,
                )
                manifest = runtime_v10._write_batch_manifest(
                    path=(
                        isaac_root
                        / "single_manifests"
                        / f"{candidate_id}.json"
                    ),
                    source_id=source_id,
                    jobs_by_id={candidate_id: job},
                    candidate_ids=(candidate_id,),
                    execution_kind="single_candidate_confirmation",
                )
                confirmation_work.append((candidate_id, manifest, single_root))
            runtime_v10._run_processes(
                confirmation_work,
                isaac_python=arguments.isaac_python,
                maximum_parallel=arguments.maximum_parallel_batches,
                deadline=deadline,
            )
            continue
        if not ready:
            time.sleep(2.0)
            continue

        # Pass the complete ready queue to the bounded process runner.  The
        # runner itself enforces ``maximum_parallel_batches`` and starts the
        # next queued chunk as soon as any slower morphology frees a slot.
        # Truncating here left GPUs idle until every process in a four-chunk
        # wave had finished.
        wave = ready
        work = []
        roots = {}
        for source_id, chunk_index, digest, candidate_ids in wave:
            candidate_root = (
                isaac_root / "batches" / source_id / f"chunk_{chunk_index:03d}_{digest}"
            )
            pending = [
                candidate_id
                for candidate_id in candidate_ids
                if not runtime_v10._inspect_candidate_output(
                    candidate_id,
                    candidate_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                ).get("valid_evidence")
                and not runtime_v10._inspect_candidate_output(
                    candidate_id,
                    isaac_root / "single" / candidate_id,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                ).get("valid_evidence")
            ]
            if not pending:
                continue
            jobs = {
                candidate_id: runtime_v10._job_for_case(
                    case_by_id[candidate_id],
                    preparation_root=preparation_root,
                    isaac_python=arguments.isaac_python,
                    rollout_steps=arguments.rollout_steps,
                )
                for candidate_id in pending
            }
            manifest = runtime_v10._write_batch_manifest(
                path=(
                    isaac_root
                    / "manifests"
                    / source_id
                    / f"chunk_{chunk_index:03d}_{digest}.json"
                ),
                source_id=source_id,
                jobs_by_id=jobs,
                candidate_ids=pending,
                execution_kind="bounded_four_range_selection",
            )
            work_id = f"{source_id}:chunk_{chunk_index:03d}_{digest}"
            roots[work_id] = (candidate_root, source_id, candidate_ids)
            work.append((work_id, manifest, candidate_root))
        runtime_v10._run_processes(
            work,
            isaac_python=arguments.isaac_python,
            maximum_parallel=arguments.maximum_parallel_batches,
            deadline=deadline,
        )

        confirmation_work = []
        for work_id, (candidate_root, source_id, candidate_ids) in roots.items():
            for candidate_id in candidate_ids:
                bounded = runtime_v10._inspect_candidate_output(
                    candidate_id,
                    candidate_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                if not bounded.get("valid_evidence") or bounded.get("candidate_passed"):
                    continue
                single_root = isaac_root / "single" / candidate_id
                single = runtime_v10._inspect_candidate_output(
                    candidate_id,
                    single_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                if not single.get("valid_evidence"):
                    job = runtime_v10._job_for_case(
                        case_by_id[candidate_id],
                        preparation_root=preparation_root,
                        isaac_python=arguments.isaac_python,
                        rollout_steps=arguments.rollout_steps,
                    )
                    manifest = runtime_v10._write_batch_manifest(
                        path=isaac_root / "single_manifests" / f"{candidate_id}.json",
                        source_id=source_id,
                        jobs_by_id={candidate_id: job},
                        candidate_ids=(candidate_id,),
                        execution_kind="single_candidate_confirmation",
                    )
                    confirmation_work.append(
                        (candidate_id, manifest, single_root)
                    )

        # Independent confirmations have no shared mutable state.  Submit the
        # complete queue to the same bounded process runner so up to four
        # failed batch members are checked concurrently.  The previous
        # candidate-by-candidate call left three Isaac slots idle.
        if confirmation_work:
            runtime_v10._run_processes(
                confirmation_work,
                isaac_python=arguments.isaac_python,
                maximum_parallel=arguments.maximum_parallel_batches,
                deadline=deadline,
            )

        for work_id, (candidate_root, _source_id, candidate_ids) in roots.items():
            for candidate_id in candidate_ids:
                bounded = runtime_v10._inspect_candidate_output(
                    candidate_id,
                    candidate_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                if not bounded.get("valid_evidence") or bounded.get("candidate_passed"):
                    continue
                single_root = isaac_root / "single" / candidate_id
                single = runtime_v10._inspect_candidate_output(
                    candidate_id,
                    single_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                case = case_by_id[candidate_id]
                if (
                    single.get("valid_evidence")
                    and not single.get("candidate_passed")
                    and (
                        case.sample_kind == "lattice"
                        or single.get("safety_failure_count", 0)
                        or single.get("fallback_count", 0)
                    )
                ):
                    decisive = single
                    break
            if decisive is not None:
                break
        print(
            "ORDER9_R1_V13_STREAMING_ISAAC_PROGRESS="
            + json.dumps(
                {
                    "level_id": arguments.level,
                    "split": arguments.split,
                    "completed_chunks": len(completed),
                    "total_chunks": len(specs),
                    "decisive_candidate_id": (
                        None if decisive is None else decisive["candidate_id"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if decisive is not None:
            break

    print(
        "ORDER9_R1_V13_STREAMING_ISAAC_COMPLETE="
        + json.dumps(
            {
                "level_id": arguments.level,
                "split": arguments.split,
                "completed_chunks": len(completed),
                "total_chunks": len(specs),
                "decisive_evidence": decisive,
                "timed_out": time.time() >= deadline,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if decisive is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
