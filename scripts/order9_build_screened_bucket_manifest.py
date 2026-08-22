#!/usr/bin/env python3
from __future__ import annotations

"""Build an immutable Order 9 bucket manifest after Isaac screening."""

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucketManifest,
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file


SCREENING_FILTER_VERSION = "order9_pi_l_bucket_screening_filter_v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = REPOSITORY_ROOT / value
    return value.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-review-manifest", required=True)
    parser.add_argument("--exclude-bucket-id", action="append", default=[])
    parser.add_argument("--screening-evidence", action="append", required=True)
    parser.add_argument(
        "--required-module-count",
        action="append",
        type=int,
        default=[],
        help="Require both train and validation coverage for this module count.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_path = _resolve(args.source_manifest)
    output_path = _resolve(args.output_manifest)
    output_review_path = _resolve(args.output_review_manifest)
    if source_path.parent != output_path.parent:
        raise ValueError(
            "screened manifest must remain beside its source so relative bucket "
            "paths retain their meaning"
        )
    source = load_order9_pi_l_rollout_bucket_manifest(source_path)
    source_ids = {bucket.bucket_id for bucket in source.buckets}
    excluded = tuple(dict.fromkeys(args.exclude_bucket_id))
    missing = sorted(set(excluded) - source_ids)
    if missing:
        raise SchemaValidationError(
            f"screening exclusions are absent from source manifest: {missing}"
        )
    evidence_records: list[dict[str, str]] = []
    for value in args.screening_evidence:
        evidence_path = _resolve(value)
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        # Evidence must be parseable JSON or JSONL; its hash is the immutable
        # provenance binding used by the derived manifest.
        text = evidence_path.read_text(encoding="utf-8")
        if evidence_path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        else:
            json.loads(text)
        evidence_records.append(
            {
                "path": str(evidence_path.relative_to(REPOSITORY_ROOT)),
                "sha256": hash_file(evidence_path),
            }
        )
    retained = [bucket for bucket in source.buckets if bucket.bucket_id not in excluded]
    required_counts = tuple(sorted(set(args.required_module_count)))
    for module_count in required_counts:
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION):
            if not any(
                bucket.module_count == module_count and bucket.split == split
                for bucket in retained
            ):
                raise SchemaValidationError(
                    "screened manifest lacks required coverage for "
                    f"module_count={module_count}, split={split.value}"
                )
    screening_filter = {
        "version": SCREENING_FILTER_VERSION,
        "source_manifest_path": str(source_path.relative_to(REPOSITORY_ROOT)),
        "source_manifest_sha256": hash_file(source_path),
        "excluded_bucket_ids": list(excluded),
        "retained_bucket_count": len(retained),
        "required_module_counts": list(required_counts),
        "evidence": evidence_records,
    }
    source_review_value = source.metadata.get("c3_human_review_manifest_path")
    source_review_sha256 = source.metadata.get("c3_human_review_manifest_sha256")
    if not isinstance(source_review_value, str) or not isinstance(
        source_review_sha256, str
    ):
        raise SchemaValidationError("source manifest lacks C3 human-review binding")
    source_review_path = _resolve(source_review_value)
    if hash_file(source_review_path) != source_review_sha256:
        raise SchemaValidationError("source C3 human-review bytes changed")
    review = json.loads(source_review_path.read_text(encoding="utf-8"))
    entries = review.get("entries")
    if not isinstance(entries, list):
        raise SchemaValidationError("source C3 human-review entries are invalid")
    retained_ids = {bucket.bucket_id for bucket in retained}
    retained_entries = [
        entry for entry in entries if entry.get("bucket_id") in retained_ids
    ]
    if {entry.get("bucket_id") for entry in retained_entries} != retained_ids:
        raise SchemaValidationError("source C3 human review does not cover retained buckets")
    derived_review = dict(review)
    derived_review["entries"] = retained_entries
    derived_review["accepted_count"] = len(retained_entries)
    derived_review["rejected_count"] = 0
    derived_review["screening_filter"] = screening_filter
    output_review_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_review = output_review_path.with_name(
        f".{output_review_path.name}.{os.getpid()}.tmp"
    )
    temporary_review.write_text(
        json.dumps(derived_review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_review, output_review_path)
    metadata = dict(source.metadata)
    metadata["screening_filter"] = screening_filter
    metadata["c3_human_review_manifest_path"] = str(
        output_review_path.relative_to(REPOSITORY_ROOT)
    )
    metadata["c3_human_review_manifest_sha256"] = hash_file(output_review_path)
    screened = Order9PiLRolloutBucketManifest(
        stage_id=source.stage_id,
        stage_config_hash=source.stage_config_hash,
        curriculum_schedule_hash=source.curriculum_schedule_hash,
        config_hash=source.config_hash,
        physical_model_hash=source.physical_model_hash,
        topology_randomized=source.topology_randomized,
        buckets=retained,
        source_pool_path=source.source_pool_path,
        source_pool_sha256=source.source_pool_sha256,
        source_asset_manifest_path=source.source_asset_manifest_path,
        source_asset_manifest_sha256=source.source_asset_manifest_sha256,
        manifest_version=source.manifest_version,
        metadata=metadata,
    )
    screened.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(screened.to_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    validate_order9_pi_l_rollout_bucket_bytes(
        output_path,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        "ORDER9_SCREENED_BUCKET_MANIFEST="
        + json.dumps(
            {
                "path": str(output_path),
                "sha256": hash_file(output_path),
                "excluded_bucket_ids": list(excluded),
                "retained_bucket_count": len(retained),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
