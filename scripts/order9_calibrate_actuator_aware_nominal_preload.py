#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_actuator_aware_preload_calibration import (
    Order9ActuatorAwarePreloadCalibrationConfig,
    calibrate_order9_actuator_aware_preload_margin,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--quantization-mm", type=float, default=1.0)
    args = parser.parse_args()
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = calibrate_order9_actuator_aware_preload_margin(
        dataset_manifest_path=_resolve(args.dataset_manifest),
        config=Order9ActuatorAwarePreloadCalibrationConfig(
            calibration_quantile=args.quantile,
            quantization_m=args.quantization_mm * 1.0e-3,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
