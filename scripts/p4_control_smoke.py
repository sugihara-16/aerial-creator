from __future__ import annotations

import argparse
import json
from pathlib import Path

from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.training.p4_control_runner import (
    P4ControlLowLevelRunner,
    P4ControlLowLevelRunnerConfig,
    load_p4_control_low_level_runner_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or probe P4-control low-level smoke cases.")
    parser.add_argument(
        "--config",
        default="configs/training/p4_control_low_level.yaml",
        help="P4-control low-level runner config path.",
    )
    parser.add_argument("--real", action="store_true", help="Request real Isaac smoke execution.")
    parser.add_argument("--probe", action="store_true", help="Only report IsaacLab backend availability.")
    parser.add_argument("--archive-path", default=None, help="Optional archive JSONL output path.")
    args = parser.parse_args()

    runner_config, env_config = load_p4_control_low_level_runner_config(args.config)
    backend_config = load_isaac_lab_backend_config(env_config.config_path)
    backend = IsaacLabBackend(backend_config)
    if args.probe:
        print(json.dumps(backend.availability().to_dict(), sort_keys=True))
        return 0

    runner_config = P4ControlLowLevelRunnerConfig(
        seed=runner_config.seed,
        source_hash=runner_config.source_hash,
        runner_version=runner_config.runner_version,
        dry_run=not args.real,
        archive_path=runner_config.archive_path,
    )
    runner = P4ControlLowLevelRunner(runner_config=runner_config, env_config=env_config)
    result = runner.run(archive_path=Path(args.archive_path) if args.archive_path else None)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.acceptance_report.fast_gate_passed or result.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
