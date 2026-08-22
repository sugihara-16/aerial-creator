#!/usr/bin/env python3
from __future__ import annotations

"""Create the hash-bound v9 81-bin contact-normal C3 initializer."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_pi_l_categorical_contact_normal_migration import (
    prepare_order9_pi_l_categorical_contact_normal_initializer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-std-normalized", type=float, default=0.30)
    parser.add_argument(
        "--prior-mode",
        choices=("gaussian_centered", "uniform_window"),
        default="gaussian_centered",
    )
    parser.add_argument("--uniform-window-min-mm", type=float, default=10.0)
    parser.add_argument("--uniform-window-max-mm", type=float, default=20.0)
    parser.add_argument("--uniform-outside-logit-penalty", type=float, default=4.0)
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
    manifest = prepare_order9_pi_l_categorical_contact_normal_initializer(
        config_path=_resolve(args.config),
        source_checkpoint_path=_resolve(args.parent_checkpoint),
        output_checkpoint_path=output / "checkpoint.pt",
        output_manifest_path=output / "migration_manifest.json",
        git_revision=revision + "-categorical-contact-normal-v9-uncommitted",
        initial_std_normalized=float(args.initial_std_normalized),
        prior_mode=str(args.prior_mode),
        uniform_window_min_normalized=float(args.uniform_window_min_mm) / 20.0,
        uniform_window_max_normalized=float(args.uniform_window_max_mm) / 20.0,
        uniform_outside_logit_penalty=float(
            args.uniform_outside_logit_penalty
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
