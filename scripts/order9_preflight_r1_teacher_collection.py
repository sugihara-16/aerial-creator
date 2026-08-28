#!/usr/bin/env python3
from __future__ import annotations

"""Audit readiness for R1 calibration or teacher trajectory collection."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_r1_preflight import (
    ORDER9_R1_MINIMUM_FREE_GIB,
    preflight_order9_r1_teacher_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("calibration", "collection"),
        default="calibration",
        help=(
            "calibration verifies the immutable C3/environment base; collection "
            "also requires accepted R1 distribution evidence"
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--calibration-protocol",
        type=Path,
        default=Path("configs/training/order9_r1_calibration_protocol_v1.yaml"),
    )
    parser.add_argument("--distribution-manifest", type=Path)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=ORDER9_R1_MINIMUM_FREE_GIB,
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-stage-output", type=Path)
    args = parser.parse_args()

    repository = args.repository_root.resolve()
    output = args.output
    if output is None:
        output = repository / (
            "artifacts/p4_full/order9/r1_teacher/preflight/"
            f"r1_{args.mode}_preflight_v1.json"
        )
    elif not output.is_absolute():
        output = repository / output

    report = preflight_order9_r1_teacher_collection(
        repository_root=repository,
        mode=args.mode,
        distribution_manifest_path=args.distribution_manifest,
        calibration_protocol_path=args.calibration_protocol,
        minimum_free_gib=args.minimum_free_gib,
        require_cuda=not args.allow_cpu,
        prepared_stage_output_path=args.prepared_stage_output,
    )
    _atomic_write_text(output, report.to_json(indent=2) + "\n")
    print(f"report: {output}")
    print(f"passed: {str(report.passed).lower()}")
    print(f"ready_for_calibration: {str(report.ready_for_calibration).lower()}")
    print(f"ready_for_collection: {str(report.ready_for_collection).lower()}")
    if report.failures:
        for failure in report.failures:
            print(f"failure: {failure}")
    return 0 if report.passed else 1


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
