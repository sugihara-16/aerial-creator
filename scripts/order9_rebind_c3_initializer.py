#!/usr/bin/env python3
from __future__ import annotations

"""Bind the exact C3 pi_L initializer tensors to the current PhysicalModel."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pi_l_initializer_rebind import (
    prepare_order9_pi_l_initializer_physical_rebind,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument(
        "--source-checkpoint",
        default=(
            "artifacts/p4_full/order9/c3_preparation/"
            "pi_l_active_knot_joint_load_initializer_v2.pt"
        ),
    )
    parser.add_argument(
        "--output-checkpoint",
        default=(
            "artifacts/p4_full/order9/c3_preparation/"
            "pi_l_active_knot_joint_load_initializer_current_physical_v2.pt"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        default=(
            "artifacts/p4_full/order9/c3_preparation/"
            "pi_l_active_knot_joint_load_initializer_current_physical_v2_manifest.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    return parser


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def main() -> int:
    args = _parser().parse_args()
    config = load_order9_learning_config(_resolve(args.config))
    physical = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path)
    )
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = prepare_order9_pi_l_initializer_physical_rebind(
        source_checkpoint_path=_resolve(args.source_checkpoint),
        output_checkpoint_path=_resolve(args.output_checkpoint),
        output_manifest_path=_resolve(args.output_manifest),
        target_physical_model_hash=physical.stable_hash(),
        git_revision=git_revision,
        device=args.device,
    )
    print(
        "ORDER9_C3_INITIALIZER_PHYSICAL_REBIND="
        + json.dumps(
            {
                "checkpoint_path": str(result.checkpoint_path),
                "checkpoint_sha256": result.checkpoint_sha256,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "source_physical_model_hash": (
                    result.manifest.source_physical_model_hash
                ),
                "target_physical_model_hash": (
                    result.manifest.target_physical_model_hash
                ),
                "exact_parameter_copy": result.manifest.exact_parameter_copy,
            },
            sort_keys=True,
        )
    )
    return 0


raise SystemExit(main())
