#!/usr/bin/env python3
from __future__ import annotations

"""Build a batched-Isaac job manifest for already admitted R1 repair cases."""

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--additional-compression-mm", type=float, default=0.0)
    parser.add_argument("--release-height-offset-mm", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.additional_compression_mm <= 10.0:
        raise SchemaValidationError("R1 repaired-case compression is invalid")
    if not 0.0 <= args.release_height_offset_mm <= 30.0:
        raise SchemaValidationError("R1 repaired-case release height is invalid")

    repository = REPOSITORY.resolve()
    jobs = []
    for raw_manifest in args.case_manifest:
        case = load_order9_r1_isaac_case(Path(raw_manifest), repository)
        command, log_path = order9_r1_nominal_case_command(
            case,
            repository_root=repository,
            python_executable=args.isaac_python,
            rollout_steps=args.rollout_steps,
        )
        if (
            len(command) < 3
            or Path(command[1]).name != "order9_vectorized_isaac_rollout.py"
        ):
            raise SchemaValidationError("R1 repaired-case command differs")
        jobs.append(
            {
                "name": case.manifest.candidate_id,
                "argv": command[2:],
                "log_path": str(log_path),
            }
        )
    if len({job["name"] for job in jobs}) != len(jobs):
        raise SchemaValidationError("R1 repaired-case jobs contain duplicate IDs")
    value = f"{args.additional_compression_mm:.1f}"
    output = write_order9_r1_clearance_diagnostic(
        {
            "manifest_version": "order9_r1_repaired_case_isaac_jobs_v1",
            "pi_l_actor_command_applied": False,
            "r1_nominal_additional_compression_sweep_mm": ",".join(
                value for _ in range(2)
            ),
            "r1_release_height_offset_m": (
                float(args.release_height_offset_mm) * 1.0e-3
            ),
            "jobs": jobs,
        },
        Path(args.output),
    )
    print(f"ORDER9_R1_REPAIRED_CASE_JOBS={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
