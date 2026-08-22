#!/usr/bin/env python3
from __future__ import annotations

"""Run the complete C3 nominal collision audit in CPU subprocess shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPOSITORY_ROOT / "scripts/order9_validate_c3_nominal_collision.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal_manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--margin-m", type=float, default=0.005)
    parser.add_argument("--numerical-tolerance-m", type=float, default=0.0005)
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest = _resolve(args.nominal_manifest)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nominal manifest has no entries")
    worker_count = min(int(args.workers), len(entries))
    chunks = [
        list(range(start, len(entries), worker_count))
        for start in range(worker_count)
    ]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="order9_c3_collision_") as raw:
        temporary = Path(raw)

        def run(worker_index: int) -> Path:
            path = temporary / f"worker_{worker_index:02d}.json"
            command = [
                sys.executable,
                str(VALIDATOR),
                str(manifest),
                "--curriculum-config",
                str(_resolve(args.curriculum_config)),
                "--bucket-indices",
                ",".join(str(value) for value in chunks[worker_index]),
                "--output",
                str(path),
                "--margin-m",
                str(float(args.margin_m)),
                "--numerical-tolerance-m",
                str(float(args.numerical_tolerance_m)),
            ]
            subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=True,
            )
            return path

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            paths = list(executor.map(run, range(worker_count)))
        worker_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        ]
    records_by_id = {
        record["bucket_id"]: record
        for worker in worker_payloads
        for record in worker["records"]
    }
    ordered_ids = [str(entry["bucket_id"]) for entry in entries]
    if set(records_by_id) != set(ordered_ids):
        raise RuntimeError("parallel collision audit coverage differs")
    records = [records_by_id[bucket_id] for bucket_id in ordered_ids]
    result = {
        "validator_version": (
            "order9_c3_saved_nominal_convex_collision_recheck_v2:"
            "convex_proxy:parallel_merge_v1"
        ),
        "collision_geometry_mode": "convex_proxy",
        "requested_collision_margin_m": float(args.margin_m),
        "numerical_tolerance_m": float(args.numerical_tolerance_m),
        "nominal_manifest_path": _portable(manifest),
        "nominal_manifest_sha256": worker_payloads[0][
            "nominal_manifest_sha256"
        ],
        "bucket_count": len(records),
        "all_accepted": all(value["accepted"] for value in records),
        "records": records,
        "worker_count": worker_count,
        "wall_time_s": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"ORDER9_C3_PARALLEL_COLLISION_RECHECK=path={output} "
        f"accepted={result['all_accepted']} workers={worker_count} "
        f"wall_s={result['wall_time_s']:.3f}"
    )
    return 0


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
