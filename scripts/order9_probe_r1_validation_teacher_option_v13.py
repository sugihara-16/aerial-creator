#!/usr/bin/env python3
from __future__ import annotations

"""Probe one validation-side R1 v13 option without altering bound evidence."""

from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import order9_probe_r1_range_teacher_option_v13 as probe


def main() -> int:
    original = probe.runtime_v13.runtime_v11._corrected_cases

    def validation_cases(level_id: str, _split: str):
        return original(level_id, "validation")

    probe.runtime_v13.runtime_v11._corrected_cases = validation_cases
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
