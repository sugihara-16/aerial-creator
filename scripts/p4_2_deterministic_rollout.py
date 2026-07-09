from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.simulation import P4_2IsaacEnv
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.training.p4_2_deterministic_rollout_runner import (
    P4_2DeterministicRolloutRunner,
    P4_2DeterministicRolloutRunnerConfig,
    load_p4_2_deterministic_rollout_runner_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or probe the P4.2 deterministic grasp/carry rollout.")
    parser.add_argument(
        "--config",
        default="configs/training/p4_2_deterministic_rollout.yaml",
        help="P4.2 deterministic rollout runner config path.",
    )
    parser.add_argument("--real", action="store_true", help="Request real Isaac rollout execution.")
    parser.add_argument("--probe", action="store_true", help="Only report IsaacLab backend availability.")
    parser.add_argument("--archive-path", default=None, help="Optional archive JSONL output path.")
    args = parser.parse_args()

    runner_config, env_config = load_p4_2_deterministic_rollout_runner_config(args.config)
    backend_config = load_isaac_lab_backend_config(env_config.config_path)
    backend = IsaacLabBackend(backend_config)
    if args.probe:
        print(json.dumps(backend.availability().to_dict(), sort_keys=True))
        return 0

    runner_config = P4_2DeterministicRolloutRunnerConfig(
        seed=runner_config.seed,
        sample_index=runner_config.sample_index,
        source_hash=runner_config.source_hash,
        runner_version=runner_config.runner_version,
        dry_run=not args.real,
        archive_path=runner_config.archive_path,
        robot_model_config_path=runner_config.robot_model_config_path,
        p3_config_path=runner_config.p3_config_path,
    )
    env = P4_2IsaacEnv(config=env_config, backend=backend)
    runner = P4_2DeterministicRolloutRunner(runner_config=runner_config, env_config=env_config, env=env)
    result = runner.run(archive_path=Path(args.archive_path) if args.archive_path else None)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.dry_run or result.rollout_result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
