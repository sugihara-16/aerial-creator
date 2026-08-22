#!/usr/bin/env python3
from __future__ import annotations

"""Create the explicit, unpromoted active-knot pi_L initializer for C3."""

import argparse
import json
import subprocess
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_pi_l_active_knot_migration import (
    prepare_order9_pi_l_active_knot_initializer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument(
        "--lineage-import",
        default="configs/training/order9_curriculum_lineage_v3.yaml",
    )
    parser.add_argument(
        "--output-checkpoint",
        default=(
            "artifacts/p4_full/order9/c3_preparation/"
            "pi_l_active_knot_joint_load_initializer_v2.pt"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        default=(
            "artifacts/p4_full/order9/c3_preparation/"
            "pi_l_active_knot_joint_load_initializer_v2_manifest.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    return parser


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return (
        (REPOSITORY_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def main() -> int:
    args = _parser().parse_args()
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = prepare_order9_pi_l_active_knot_initializer(
        config_path=_resolve(args.config),
        lineage_import_path=_resolve(args.lineage_import),
        output_checkpoint_path=_resolve(args.output_checkpoint),
        output_manifest_path=_resolve(args.output_manifest),
        git_revision=git_revision,
        device=args.device,
    )
    print(
        "ORDER9_C3_INITIALIZER="
        + json.dumps(
            {
                "checkpoint_path": str(result.checkpoint_path),
                "checkpoint_sha256": result.checkpoint_sha256,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "promoted_checkpoint": False,
            },
            sort_keys=True,
        )
    )
    return 0


raise SystemExit(main())
