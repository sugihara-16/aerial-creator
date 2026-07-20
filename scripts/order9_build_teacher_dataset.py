#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from amsrr.training.order9_dataset import load_order9_dataset
from amsrr.training.order9_teacher_collection import build_order9_teacher_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a hash-verified Order 9 C0 dataset from episode bundles."
    )
    parser.add_argument("--episode-manifest", action="append", default=[])
    parser.add_argument("--episode-root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = [Path(value) for value in args.episode_manifest]
    if args.episode_root:
        paths.extend(
            sorted(Path(args.episode_root).glob("*/episode_manifest.json"))
        )
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        parser.error("provide --episode-manifest or --episode-root")
    manifest = build_order9_teacher_dataset(unique, args.output_dir)
    bundle = load_order9_dataset(args.output_dir)
    print(f"dataset_id: {manifest.dataset_id}")
    print(f"manifest: {Path(args.output_dir) / 'manifest.json'}")
    print(f"episodes: {len(manifest.source_episode_ids)}")
    print(f"low_level_records: {len(bundle.low_level_records)}")
    print(f"trajectory_records: {len(bundle.trajectory_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
