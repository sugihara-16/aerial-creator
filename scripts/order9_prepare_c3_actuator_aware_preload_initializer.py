#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_actuator_aware_preload_migration import (
    prepare_order9_actuator_aware_preload_initializer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebase the C3 categorical contact residual around zero preload"
    )
    parser.add_argument("--config", default="configs/training/order9_learning_curriculum.yaml")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neutral-residual-std-normalized", type=float, default=0.15)
    args = parser.parse_args()
    output = _resolve(args.output_dir)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() + "-actuator-aware-preload-rebase-uncommitted"
    manifest = prepare_order9_actuator_aware_preload_initializer(
        config_path=_resolve(args.config),
        source_checkpoint_path=_resolve(args.source_checkpoint),
        output_checkpoint_path=output / "checkpoint.pt",
        output_manifest_path=output / "migration_manifest.json",
        git_revision=revision,
        neutral_residual_std_normalized=args.neutral_residual_std_normalized,
    )
    import json

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
