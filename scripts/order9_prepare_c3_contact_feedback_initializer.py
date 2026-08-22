#!/usr/bin/env python3
from __future__ import annotations

"""Create the hash-bound v8 contact-feedback C3 initializer."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_pi_l_contact_feedback_migration import (
    prepare_order9_pi_l_contact_feedback_initializer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-inward-residual-mm", type=float, default=6.0)
    parser.add_argument("--initial-normal-exploration-std-mm", type=float, default=1.0)
    return parser


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def main() -> int:
    args = _parser().parse_args()
    output = _resolve(args.output_dir)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    manifest = prepare_order9_pi_l_contact_feedback_initializer(
        config_path=_resolve(args.config),
        source_checkpoint_path=_resolve(args.parent_checkpoint),
        output_checkpoint_path=output / "checkpoint.pt",
        output_manifest_path=output / "migration_manifest.json",
        git_revision=revision + "-contact-feedback-v8-uncommitted",
        initial_inward_residual_m=float(args.initial_inward_residual_mm) * 1.0e-3,
        initial_normal_exploration_std_m=(
            float(args.initial_normal_exploration_std_mm) * 1.0e-3
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
