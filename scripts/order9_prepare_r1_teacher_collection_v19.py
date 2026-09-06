#!/usr/bin/env python3
from __future__ import annotations

"""Prepare and verify the formal R1 teacher collection without starting it."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_r1_teacher_collection_preparation import (  # noqa: E402
    build_order9_r1_teacher_collection_plan,
    build_order9_r1_teacher_collection_preflight,
    load_order9_r1_teacher_collection_protocol,
    protocol_with_path,
)

DEFAULT_PROTOCOL = REPOSITORY / (
    "configs/training/order9_r1_teacher_collection_protocol_v19.json"
)
DEFAULT_OUTPUT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/formal_collection_v19"
)
DEFAULT_ISAAC_PYTHON = Path("/home/leus/.local/share/mamba/envs/isaaclab3/bin/python")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--isaac-python", default=str(DEFAULT_ISAAC_PYTHON))
    return parser


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _active_collection_process_count() -> int:
    completed = subprocess.run(
        ["pgrep", "-af", "order9.*r1.*teacher.*collect"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        return 1
    own_pid = str(os.getpid())
    return sum(
        own_pid not in line and "pgrep -af" not in line
        for line in completed.stdout.splitlines()
        if line.strip()
    )


def _isaac_import_ok(python: Path) -> bool:
    if not python.is_file():
        return False
    completed = subprocess.run(
        [str(python), "-c", "import torch; assert torch.cuda.is_available()"],
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    return completed.returncode == 0


def main() -> int:
    args = _parser().parse_args()
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output_root).resolve()
    protocol = protocol_with_path(
        load_order9_r1_teacher_collection_protocol(
            protocol_path, repository_root=REPOSITORY
        ),
        protocol_path,
    )
    plan = build_order9_r1_teacher_collection_plan(protocol, repository_root=REPOSITORY)
    plan_path = output / "collection_plan.json"
    _atomic_json(plan_path, plan)
    free = shutil.disk_usage(
        output.parent if output.parent.exists() else REPOSITORY
    ).free
    preflight = build_order9_r1_teacher_collection_preflight(
        protocol=protocol,
        plan_path=plan_path,
        repository_root=REPOSITORY,
        free_storage_bytes=free,
        isaac_python=args.isaac_python,
        isaac_import_ok=_isaac_import_ok(Path(args.isaac_python).resolve()),
        active_collection_process_count=_active_collection_process_count(),
        existing_collection_record_count=len(list((output / "records").glob("*.json"))),
    )
    preflight_path = output / "preflight.json"
    _atomic_json(preflight_path, preflight)
    launch_template = {
        "authorization_version": "order9_r1_teacher_collection_launch_v19",
        "status": (
            "awaiting_explicit_user_instruction"
            if preflight["ready_for_explicit_launch"]
            else "not_ready"
        ),
        "collection_launch_authorized": False,
        "training_authorized": False,
        "plan": preflight["plan"],
        "preflight": {
            "path": preflight_path.relative_to(REPOSITORY).as_posix(),
            "sha256": __import__("hashlib")
            .sha256(preflight_path.read_bytes())
            .hexdigest(),
        },
        "note": "Do not edit this template into an approval; create a separately hash-bound authorization only after an explicit user start instruction.",
    }
    _atomic_json(output / "launch_authorization_template.json", launch_template)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    return 0 if preflight["ready_for_explicit_launch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
