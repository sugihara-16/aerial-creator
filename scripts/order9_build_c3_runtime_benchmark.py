#!/usr/bin/env python3
from __future__ import annotations

"""Build the C3 runtime gate from single and concurrent collector evidence."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_c3_runtime_benchmark import (
    build_order9_c3_runtime_benchmark_report,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument(
        "--stage",
        default="c3_pi_l_ppo_arbitrary_morphology",
    )
    parser.add_argument("--pi-l-checkpoint-sha256", required=True)
    parser.add_argument("--throughput-artifact", action="append", required=True)
    parser.add_argument("--concurrent-artifact", action="append", required=True)
    parser.add_argument(
        "--rejected-capacity-artifact",
        action="append",
        default=[],
    )
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--maximum-gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    return parser


def _resolve(value: str) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


def main() -> int:
    args = _parser().parse_args()
    config = load_order9_learning_config(_resolve(args.config))
    output = _resolve(args.output)
    report = build_order9_c3_runtime_benchmark_report(
        config,
        stage_id=args.stage,
        pi_l_checkpoint_sha256=args.pi_l_checkpoint_sha256,
        throughput_artifact_paths=[
            _resolve(value) for value in args.throughput_artifact
        ],
        concurrent_artifact_paths=[
            _resolve(value) for value in args.concurrent_artifact
        ],
        rejected_capacity_artifact_paths=[
            _resolve(value) for value in args.rejected_capacity_artifact
        ],
        bucket_manifest_path=_resolve(args.bucket_manifest),
        order8_report_path=_resolve(
            config.production_runtime.canonical_order8_report_path
        ),
        output_path=output,
        maximum_gpu_memory_fraction=args.maximum_gpu_memory_fraction,
    )
    print(f"passed: {str(report.passed).lower()}")
    print(f"selected_environment_count: {report.selected_environment_count}")
    print(f"report_sha256: {hash_file(output)}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
