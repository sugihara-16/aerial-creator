from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    build_order9_r1_clearance_diagnostic,
    write_order9_r1_clearance_diagnostic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare learned-pi_L and nominal-QPID R1 support clearance"
    )
    for name in (
        "learned-raw",
        "learned-episodes",
        "learned-log",
        "nominal-raw",
        "nominal-episodes",
        "nominal-log",
        "task-spec",
        "physical-model-config",
        "canonical-order8-report",
        "canonical-order8-report-sha256",
        "protected-rollout",
        "promoted-checkpoint",
        "promoted-checkpoint-sha256",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = build_order9_r1_clearance_diagnostic(
        learned_raw_path=args.learned_raw,
        learned_episode_path=args.learned_episodes,
        learned_log_path=args.learned_log,
        nominal_raw_path=args.nominal_raw,
        nominal_episode_path=args.nominal_episodes,
        nominal_log_path=args.nominal_log,
        task_spec_path=args.task_spec,
        physical_model_config_path=args.physical_model_config,
        canonical_order8_report_path=args.canonical_order8_report,
        canonical_order8_report_sha256=args.canonical_order8_report_sha256,
        protected_rollout_path=args.protected_rollout,
        promoted_checkpoint_path=args.promoted_checkpoint,
        promoted_checkpoint_sha256=args.promoted_checkpoint_sha256,
    )
    destination = write_order9_r1_clearance_diagnostic(payload, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
