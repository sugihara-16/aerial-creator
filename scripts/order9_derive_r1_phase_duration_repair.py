#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_phase_duration_repair import (  # noqa: E402
    derive_order9_r1_phase_duration_repair_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", required=True)
    parser.add_argument("--destination-case", required=True)
    parser.add_argument("--contact-acquisition-multiplier", type=float, default=1.0)
    parser.add_argument("--post-grasp-multiplier", type=float, default=2.0)
    parser.add_argument("--place-multiplier", type=float)
    parser.add_argument("--approach-contact-handoff-fraction", type=float)
    parser.add_argument(
        "--phase-endpoint-hold",
        action="append",
        default=[],
        metavar="PHASE=SECONDS",
    )
    parser.add_argument(
        "--phase-multiplier",
        action="append",
        default=[],
        metavar="PHASE=VALUE",
    )
    args = parser.parse_args()
    phases = (
        "approach",
        "contact_acquisition",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
        "settle",
    )
    multipliers = {phase: args.post_grasp_multiplier for phase in phases}
    multipliers["approach"] = 1.0
    multipliers["contact_acquisition"] = args.contact_acquisition_multiplier
    if args.place_multiplier is not None:
        multipliers["place"] = args.place_multiplier
    for item in args.phase_multiplier:
        phase, separator, value = item.partition("=")
        if not separator or phase not in multipliers:
            parser.error("--phase-multiplier must be PHASE=VALUE for a known phase")
        try:
            multipliers[phase] = float(value)
        except ValueError:
            parser.error("--phase-multiplier VALUE must be numeric")
    endpoint_holds = {}
    for item in args.phase_endpoint_hold:
        phase, separator, value = item.partition("=")
        if not separator or phase not in multipliers:
            parser.error("--phase-endpoint-hold must be PHASE=SECONDS for a known phase")
        try:
            endpoint_holds[phase] = float(value)
        except ValueError:
            parser.error("--phase-endpoint-hold SECONDS must be numeric")
    result = derive_order9_r1_phase_duration_repair_case(
        source_case_root=args.source_case,
        destination_case_root=args.destination_case,
        repository_root=REPOSITORY,
        phase_duration_multipliers=multipliers,
        approach_contact_handoff_fraction=(
            args.approach_contact_handoff_fraction
        ),
        phase_endpoint_holds_s=endpoint_holds,
    )
    print("ORDER9_R1_PHASE_DURATION_REPAIR=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
