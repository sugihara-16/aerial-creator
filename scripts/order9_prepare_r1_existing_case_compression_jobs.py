#!/usr/bin/env python3
from __future__ import annotations

"""Prepare a redirected R1 nominal-compression test from an existing case."""

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

DEFAULT_BATCH_ROOT = Path(
    "artifacts/p4_full/order9/r1_teacher/calibration_v9_nominal_compression_v7/"
    "selection/r1_l1_10mm_5deg/persistent_process_batches"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--candidate-id")
    target.add_argument("--source-bucket-id")
    parser.add_argument("--additional-compression-mm", required=True, type=float)
    parser.add_argument("--release-height-offset-mm", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.additional_compression_mm <= 10.0:
        raise ValueError("additional compression must be in [0, 10] mm")
    if not 0.0 <= args.release_height_offset_mm <= 30.0:
        raise ValueError("release height offset must be in [0, 30] mm")
    repository = REPOSITORY.resolve()
    root = (repository / args.batch_root).resolve()
    matches: list[tuple[Path, list[dict]]] = []
    for path in sorted(root.glob("*/persistent_bucket_jobs.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list):
            continue
        if args.candidate_id is not None:
            selected = [
                value
                for value in jobs
                if isinstance(value, dict) and value.get("name") == args.candidate_id
            ]
        else:
            token = f"__train__{args.source_bucket_id}__"
            selected = [
                value
                for value in jobs
                if isinstance(value, dict) and token in str(value.get("name", ""))
            ]
            if selected and len(selected) != len(jobs):
                raise SchemaValidationError("R1 source-bucket manifest is mixed")
        if selected:
            matches.append((path, selected))
    preferred = [value for value in matches if "_chunk_" in value[0].parent.name]
    if len(preferred) == 1:
        matches = preferred
    if len(matches) != 1:
        raise SchemaValidationError("R1 existing candidate job is not unique")
    source, jobs = matches[0]
    compression = str(float(args.additional_compression_mm))
    output = write_order9_r1_clearance_diagnostic(
        {
            "manifest_version": "order9_r1_existing_case_compression_jobs_v1",
            "source_manifest": {
                "path": str(source.relative_to(repository)),
                "sha256": hash_file(source),
            },
            "candidate_count": len(jobs),
            "r1_nominal_additional_compression_sweep_mm": (
                f"{compression},{compression}"
            ),
            "r1_release_height_offset_m": (
                float(args.release_height_offset_mm) * 1.0e-3
            ),
            "pi_l_actor_command_applied": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
            "jobs": jobs,
        },
        (repository / args.output).resolve(),
    )
    print(
        "ORDER9_R1_EXISTING_COMPRESSION_JOB="
        + json.dumps(
            {
                "candidate_id": args.candidate_id,
                "source_bucket_id": args.source_bucket_id,
                "candidate_count": len(jobs),
                "additional_compression_mm": args.additional_compression_mm,
                "release_height_offset_mm": args.release_height_offset_mm,
                "jobs": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
