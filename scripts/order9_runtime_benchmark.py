#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_isaac_benchmark_runner import (
    run_order9_isaac_runtime_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the real vectorized Isaac substrate for Order 9."
    )
    parser.add_argument(
        "--config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/p4_full/order9/runtime_benchmark.json",
    )
    args = parser.parse_args()
    learning = load_order9_learning_config(args.config)
    report = run_order9_isaac_runtime_benchmark(
        learning.runtime_benchmark,
        output_path=Path(args.output),
    )
    print(f"passed: {report.passed}")
    print(f"selected_environment_count: {report.selected_environment_count}")
    for sample in report.samples:
        print(
            f"envs={sample.environment_count} "
            f"aggregate_env_steps_per_s={sample.aggregate_env_steps_per_s:.1f} "
            f"passed={sample.passed_throughput_gate}"
        )
    print(f"report: {args.output}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
