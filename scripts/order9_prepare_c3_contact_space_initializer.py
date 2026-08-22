#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_pi_l_contact_space_migration import (
    prepare_order9_pi_l_contact_space_initializer,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = prepare_order9_pi_l_contact_space_initializer(
        config_path=REPOSITORY_ROOT / args.config,
        source_checkpoint_path=REPOSITORY_ROOT / args.source_checkpoint,
        output_checkpoint_path=REPOSITORY_ROOT / args.output_checkpoint,
        output_manifest_path=REPOSITORY_ROOT / args.output_manifest,
        git_revision=revision + "-v7-contact-space-uncommitted",
        device=args.device,
    )
    print("ORDER9_C3_CONTACT_SPACE_INITIALIZER=" + json.dumps(result, sort_keys=True))
    return 0


raise SystemExit(main())
