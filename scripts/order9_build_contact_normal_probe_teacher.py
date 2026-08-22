#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_contact_normal_probe_teacher import (
    build_order9_contact_normal_probe_teacher_dataset,
    validate_order9_contact_normal_probe_teacher_dataset,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-rollout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-probe-steps", type=int, default=64)
    return parser


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = _parser().parse_args()
    probe = Path(args.probe_rollout).resolve()
    output = Path(args.output).resolve()
    artifact = load_order9_tensor_rollout_artifact(probe)
    payload = build_order9_contact_normal_probe_teacher_dataset(
        artifact,
        source_rollout_sha256=hash_file(probe),
        maximum_probe_steps=args.maximum_probe_steps,
    )
    validate_order9_contact_normal_probe_teacher_dataset(payload)
    _atomic_write(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": hash_file(output), **{
        key: payload[key]
        for key in ("state_count", "confident_example_count", "excluded_state_count")
    }}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
