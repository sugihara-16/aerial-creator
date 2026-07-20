#!/usr/bin/env python3
from __future__ import annotations

"""Build the Order 9 throughput gate from production rollout artifacts."""

import argparse
from pathlib import Path

from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_production_benchmark import (
    build_order9_production_benchmark_report,
)
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--pi-l-checkpoint-sha256", required=True)
    parser.add_argument("--raw-artifact", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    config_path = _resolve(args.config, repository)
    config = load_order9_learning_config(config_path)
    output = Path(args.output).resolve()
    report = build_order9_production_benchmark_report(
        config.runtime_benchmark,
        raw_artifact_paths=args.raw_artifact,
        expected_stage_id=args.stage,
        expected_checkpoint_sha256=args.pi_l_checkpoint_sha256,
        order8_report_path=_resolve(
            config.production_runtime.canonical_order8_report_path,
            repository,
        ),
        output_path=output,
    )
    print(f"passed: {str(report.passed).lower()}")
    print(f"selected_environment_count: {report.selected_environment_count}")
    for sample in report.samples:
        print(
            f"{sample.environment_count}:"
            f"{sample.aggregate_env_steps_per_s:.6f}:"
            f"{str(sample.passed_throughput_gate).lower()}"
        )
    print(f"report_sha256: {hash_file(output)}")
    print(f"report: {output}")
    return 0


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
