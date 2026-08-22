#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_force_observer_replay import (
    compare_order9_force_observer_baseline,
    load_order9_force_observer_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    comparison = compare_order9_force_observer_baseline(
        load_order9_force_observer_trace(args.trace)
    )
    rendered = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print("ORDER9_FORCE_OBSERVER_BASELINE_COMPARISON=" + rendered.strip())


if __name__ == "__main__":
    main()
