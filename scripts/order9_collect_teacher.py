#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from amsrr.training.order9_teacher_collection import (
    build_order9_teacher_dataset,
    load_order9_teacher_episode,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable real-Isaac C0 Order-8 teacher collection and "
            "assemble its verified dataset."
        )
    )
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--held-out-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=9009)
    parser.add_argument(
        "--output-root", default="artifacts/p4_full/order9/c0_teacher"
    )
    parser.add_argument("--config", default="configs/training/order8_natural_contact.yaml")
    parser.add_argument("--backend-config", default="configs/env/isaac_lab.yaml")
    parser.add_argument("--low-level-stride", type=int, default=1)
    parser.add_argument("--high-level-stride", type=int, default=5)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--force-recollect", action="store_true")
    args = parser.parse_args()
    if args.episode_count < 3:
        parser.error("--episode-count must be at least three")
    if min(args.validation_count, args.held_out_count) < 1:
        parser.error("validation and held-out counts must be positive")
    if args.validation_count + args.held_out_count >= args.episode_count:
        parser.error("validation + held-out counts must leave training episodes")
    if min(args.low_level_stride, args.high_level_stride) < 1:
        parser.error("teacher strides must be positive")

    root = Path(args.output_root)
    episodes_root = root / "episodes"
    reports_root = root / "reports"
    episodes_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    episode_manifests: list[Path] = []
    failures: list[dict[str, object]] = []
    collection_started = time.monotonic()
    for index in range(args.episode_count):
        seed = args.seed_start + index
        split = _split(
            index,
            count=args.episode_count,
            validation_count=args.validation_count,
            held_out_count=args.held_out_count,
        )
        episode_id = f"order9-c0-episode-{seed:06d}"
        task_id = f"order9-c0-task-{seed:06d}"
        episode_dir = episodes_root / episode_id
        manifest_path = episode_dir / "episode_manifest.json"
        if manifest_path.is_file() and not args.force_recollect:
            manifest, _, _ = load_order9_teacher_episode(manifest_path)
            if manifest.success:
                episode_manifests.append(manifest_path)
                print(f"reuse {index + 1}/{args.episode_count}: {episode_id}")
                continue
        report_path = reports_root / f"{episode_id}.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/order8_natural_contact.py"),
            "--real",
            "--config",
            args.config,
            "--backend-config",
            args.backend_config,
            "--seed",
            str(seed),
            "--report-path",
            str(report_path),
            "--order9-teacher-output",
            str(episode_dir),
            "--order9-teacher-episode-id",
            episode_id,
            "--order9-teacher-task-id",
            task_id,
            "--order9-teacher-split",
            split,
            "--order9-teacher-low-level-stride",
            str(args.low_level_stride),
            "--order9-teacher-high-level-stride",
            str(args.high_level_stride),
        ]
        if index > 0 or any(episodes_root.glob("*/episode_manifest.json")):
            command.append("--reuse-generated-asset")
        started = time.monotonic()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        elapsed = time.monotonic() - started
        if completed.returncode == 0 and manifest_path.is_file():
            manifest, _, _ = load_order9_teacher_episode(manifest_path)
            if manifest.success:
                episode_manifests.append(manifest_path)
                print(
                    f"complete {index + 1}/{args.episode_count}: "
                    f"{episode_id} wall_s={elapsed:.1f}"
                )
                continue
        failure = {
            "episode_id": episode_id,
            "seed": seed,
            "split": split,
            "returncode": completed.returncode,
            "wall_time_s": elapsed,
            "report_path": str(report_path),
        }
        failures.append(failure)
        print(f"failed {index + 1}/{args.episode_count}: {episode_id}")
        if args.stop_on_failure:
            break

    summary = {
        "requested_episode_count": args.episode_count,
        "successful_episode_count": len(episode_manifests),
        "failure_count": len(failures),
        "failures": failures,
        "wall_time_s": time.monotonic() - collection_started,
    }
    summary_path = root / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if len(episode_manifests) != args.episode_count:
        print(f"summary: {summary_path}")
        return 1
    dataset_dir = root / "dataset"
    manifest = build_order9_teacher_dataset(episode_manifests, dataset_dir)
    print(f"dataset: {dataset_dir / 'manifest.json'}")
    print(f"dataset_id: {manifest.dataset_id}")
    print(f"summary: {summary_path}")
    return 0


def _split(
    index: int,
    *,
    count: int,
    validation_count: int,
    held_out_count: int,
) -> str:
    if index >= count - held_out_count:
        return "held_out"
    if index >= count - held_out_count - validation_count:
        return "validation"
    return "train"


if __name__ == "__main__":
    raise SystemExit(main())
