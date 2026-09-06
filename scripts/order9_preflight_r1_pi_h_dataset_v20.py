#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_r1_pi_h_dataset_preflight import (
    preflight_order9_r1_pi_h_dataset,
)

DEFAULT_SUMMARY = (
    "artifacts/p4_full/order9/r1_teacher/formal_collection_v19/"
    "collection_summary_identity_v2.json"
)
DEFAULT_OUTPUT = (
    "artifacts/p4_full/order9/r1_teacher/formal_collection_v19/"
    "pi_h_dataset_preflight_v20.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-materialize every formal R1 pi_H training row without "
            "starting training."
        )
    )
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = _resolve(args.summary)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"R1 pi_H preflight output exists: {output}")
    result = preflight_order9_r1_pi_h_dataset(
        summary,
        repository_root=REPOSITORY_ROOT,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(result, sort_keys=True))
    print(f"output: {output}")
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
