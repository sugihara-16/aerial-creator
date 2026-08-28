#!/usr/bin/env python3
from __future__ import annotations

"""Derive and lightweight-audit one R1 yaw-branch repair case."""

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_yaw_branch_repair import (  # noqa: E402
    derive_order9_r1_yaw_branch_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-case-root", required=True)
    parser.add_argument("--target-case-root", required=True)
    parser.add_argument("--output-case-root", required=True)
    args = parser.parse_args()
    result = derive_order9_r1_yaw_branch_case(
        reference_case_root=args.reference_case_root,
        target_case_root=args.target_case_root,
        destination_case_root=args.output_case_root,
        repository_root=REPOSITORY,
    )
    print("ORDER9_R1_YAW_BRANCH_REPAIR=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
