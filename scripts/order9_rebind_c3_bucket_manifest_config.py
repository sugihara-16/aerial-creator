#!/usr/bin/env python3
from __future__ import annotations

"""Rebind an unchanged C3 bucket set to a compatible runtime-only config edit."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucketManifest,
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash


STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
REBIND_VERSION = "order9_c3_runtime_config_bucket_rebind_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--reason",
        required=True,
        help="Concise provenance reason for the runtime-only config rebind.",
    )
    args = parser.parse_args()
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    source_path = (REPOSITORY_ROOT / args.source_manifest).resolve()
    output_path = (REPOSITORY_ROOT / args.output_manifest).resolve()
    if source_path.parent != output_path.parent:
        raise ValueError("rebound manifest must remain beside its source")
    if output_path.exists():
        raise FileExistsError(output_path)
    learning = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(learning, STAGE_ID)
    source = load_order9_pi_l_rollout_bucket_manifest(source_path)
    if source.stage_id != STAGE_ID:
        raise ValueError("runtime config rebind requires the C3 manifest")
    metadata = dict(source.metadata)
    metadata["runtime_config_rebind"] = {
        "version": REBIND_VERSION,
        "source_manifest_path": str(source_path.relative_to(REPOSITORY_ROOT)),
        "source_manifest_sha256": hash_file(source_path),
        "source_stage_config_hash": source.stage_config_hash,
        "source_curriculum_schedule_hash": source.curriculum_schedule_hash,
        "source_config_hash": source.config_hash,
        "reason": args.reason,
        "bucket_geometry_or_randomization_changed": False,
    }
    rebound = Order9PiLRolloutBucketManifest(
        stage_id=source.stage_id,
        stage_config_hash=stable_hash(stage.to_dict()),
        curriculum_schedule_hash=order9_schedule_hash(learning),
        config_hash=stable_hash(learning.to_dict()),
        physical_model_hash=source.physical_model_hash,
        topology_randomized=source.topology_randomized,
        buckets=source.buckets,
        source_pool_path=source.source_pool_path,
        source_pool_sha256=source.source_pool_sha256,
        source_asset_manifest_path=source.source_asset_manifest_path,
        source_asset_manifest_sha256=source.source_asset_manifest_sha256,
        manifest_version=source.manifest_version,
        metadata=metadata,
    )
    rebound.validate()
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(rebound.to_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    validate_order9_pi_l_rollout_bucket_bytes(
        output_path, repository_root=REPOSITORY_ROOT
    )
    print(
        "ORDER9_REBOUND_BUCKET_MANIFEST="
        + json.dumps(
            {"path": str(output_path), "sha256": hash_file(output_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
