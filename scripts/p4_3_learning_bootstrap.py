from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.acceptance.p4_3_acceptance import run_p4_3_acceptance
from amsrr.training.p4_3_learning_archive import write_p4_3_learning_summary_archive
from amsrr.training.p4_3_pi_d_training import train_p4_3_pi_d
from amsrr.training.p4_3_pi_h_training import train_p4_3_pi_h
from amsrr.training.p4_3_pi_l_training import train_p4_3_pi_l
from amsrr.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run staged P4.3b-d minimum learning bootstrap.")
    parser.add_argument("--config", default="configs/training/p4_3_learning_bootstrap.yaml")
    parser.add_argument("--dataset-dir", default="artifacts/p4_3/datasets")
    parser.add_argument("--output-root", default="artifacts/p4_3")
    parser.add_argument(
        "--acceptance-only",
        action="store_true",
        help="Validate existing artifacts after the separate online pi_L Isaac evaluation.",
    )
    parser.add_argument(
        "--source-rollout-archive",
        default="artifacts/p4_3/rollouts/deterministic_isaac.jsonl",
    )
    parser.add_argument(
        "--summary-archive",
        default="artifacts/p4_3/p4_3_minimum_learning_summary.jsonl",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_root)
    config = load_config(args.config)
    pi_l_cfg = config.get("pi_l", {})
    pi_h_cfg = config.get("pi_h", {})
    pi_d_cfg = config.get("pi_d", {})

    if not args.acceptance_only:
        pi_l = train_p4_3_pi_l(
            dataset_path=dataset_dir,
            output_dir=output_root / "pi_l",
            config_path=args.config,
        )
        trajectory_shards = [
            dataset_dir / f"interaction_trajectory_{split}.jsonl"
            for split in ("train", "validation", "held_out")
        ]
        pi_h = train_p4_3_pi_h(
            shard_paths=trajectory_shards,
            output_dir=output_root / "pi_h",
            epochs=int(pi_h_cfg.get("epochs", 20)),
            learning_rate=float(pi_h_cfg.get("learning_rate", 0.001)),
            seed=int(pi_h_cfg.get("seed", 13)),
            hidden_dim=int(pi_h_cfg.get("hidden_dim", 64)),
        )
        design_shards = [
            dataset_dir / f"design_outcome_{split}.jsonl"
            for split in ("train", "validation", "held_out")
        ]
        pi_d = train_p4_3_pi_d(
            dataset_paths=design_shards,
            output_dir=output_root / "pi_d",
            p2_checkpoint_path=pi_d_cfg.get("initializer_checkpoint"),
            epochs=int(pi_d_cfg.get("epochs", 20)),
            lr=float(pi_d_cfg.get("learning_rate", 0.001)),
            seed=int(pi_d_cfg.get("seed", 17)),
            hidden_dim=int(pi_d_cfg.get("hidden_dim", 24)),
        )
        print(
            json.dumps(
                {
                    "pi_l_checkpoint": pi_l.checkpoint_path,
                    "pi_h_checkpoint": pi_h.checkpoint_path,
                    "pi_d_checkpoint": pi_d.checkpoint_path,
                    "next_gate": "run learned pi_L online Isaac evaluation, then --acceptance-only",
                    "pi_l_config": pi_l_cfg,
                },
                sort_keys=True,
            )
        )
        return 0

    acceptance = run_p4_3_acceptance(
        dataset_manifest_path=dataset_dir / "manifest.json",
        pi_l_dir=output_root / "pi_l",
        pi_h_dir=output_root / "pi_h",
        pi_d_dir=output_root / "pi_d",
    )
    summary_archive = None
    if acceptance.completion_passed:
        summary_archive = write_p4_3_learning_summary_archive(
            source_rollout_archive_path=args.source_rollout_archive,
            output_path=args.summary_archive,
            dataset_manifest_path=dataset_dir / "manifest.json",
            pi_l_dir=output_root / "pi_l",
            pi_h_dir=output_root / "pi_h",
            pi_d_dir=output_root / "pi_d",
            acceptance=acceptance,
        )
    print(
        json.dumps(
            {
                "acceptance": acceptance.to_dict(),
                "summary_archive": args.summary_archive if summary_archive is not None else None,
            },
            sort_keys=True,
        )
    )
    return 0 if acceptance.completion_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
