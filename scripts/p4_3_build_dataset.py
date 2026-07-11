from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.training.p4_3_dataset_builder import build_p4_3_dataset
from amsrr.training.p4_3_reward import P4_3RewardConfig
from amsrr.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build P4.3 learning shards from Isaac archives.")
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--output-dir", default="artifacts/p4_3/datasets")
    parser.add_argument("--low-level-stride", type=int, default=4)
    parser.add_argument("--config", default="configs/training/p4_3_learning_bootstrap.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = build_p4_3_dataset(
        archive_paths=args.archives,
        output_dir=args.output_dir,
        low_level_stride=args.low_level_stride,
        reward_config=P4_3RewardConfig(**config.get("reward", {})),
        split_fractions=config.get("collection", {}).get("split_fractions"),
    )
    print(
        json.dumps(
            {
                "manifest": result.manifest_path,
                "dataset_id": result.manifest.dataset_id,
                "record_counts": result.manifest.record_counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
