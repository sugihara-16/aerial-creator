#!/usr/bin/env python3
from __future__ import annotations

"""Materialize a successful R1 release-height diagnostic into its teacher."""

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_release_clearance_materialization import (  # noqa: E402
    derive_order9_r1_release_clearance_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", required=True)
    parser.add_argument("--destination-case", required=True)
    parser.add_argument("--height-offset-mm", type=float, required=True)
    parser.add_argument("--diagnostic-manifest", required=True)
    parser.add_argument("--diagnostic-episodes", required=True)
    args = parser.parse_args()
    result = derive_order9_r1_release_clearance_case(
        source_case_root=args.source_case,
        destination_case_root=args.destination_case,
        repository_root=REPOSITORY,
        height_offset_m=float(args.height_offset_mm) * 1.0e-3,
        diagnostic_manifest_path=args.diagnostic_manifest,
        diagnostic_episodes_path=args.diagnostic_episodes,
    )
    print(
        "ORDER9_R1_RELEASE_CLEARANCE_MATERIALIZED="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
