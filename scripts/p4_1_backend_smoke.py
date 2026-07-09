from __future__ import annotations

import argparse
import json
from pathlib import Path

from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.p4_1_isaac_env import P4_1IsaacBackendEnv
from amsrr.training.p4_1_backend_smoke_runner import (
    P4_1BackendSmokeRunner,
    P4_1BackendSmokeRunnerConfig,
    load_p4_1_backend_smoke_runner_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or probe the P4.1 full-scene backend smoke.")
    parser.add_argument(
        "--config",
        default="configs/training/p4_1_backend_smoke.yaml",
        help="P4.1 backend smoke runner config path.",
    )
    parser.add_argument("--real", action="store_true", help="Request real Isaac smoke execution.")
    parser.add_argument("--probe", action="store_true", help="Only report IsaacLab backend availability.")
    parser.add_argument("--archive-path", default=None, help="Optional archive JSONL output path.")
    args = parser.parse_args()

    runner_config, env_config = load_p4_1_backend_smoke_runner_config(args.config)
    backend_config = load_isaac_lab_backend_config(env_config.config_path)
    backend = IsaacLabBackend(backend_config)
    if args.probe:
        print(json.dumps(backend.availability().to_dict(), sort_keys=True))
        return 0

    runner_config = P4_1BackendSmokeRunnerConfig(
        seed=runner_config.seed,
        sample_index=runner_config.sample_index,
        source_hash=runner_config.source_hash,
        runner_version=runner_config.runner_version,
        dry_run=not args.real,
        archive_path=runner_config.archive_path,
        robot_model_config_path=runner_config.robot_model_config_path,
        p3_config_path=runner_config.p3_config_path,
        module_spacing_m=runner_config.module_spacing_m,
    )
    env = P4_1IsaacBackendEnv(config=env_config, backend=backend)
    runner = P4_1BackendSmokeRunner(runner_config=runner_config, env_config=env_config, env=env)
    result = runner.run(archive_path=Path(args.archive_path) if args.archive_path else None)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.dry_run or result.acceptance_report.completion_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
