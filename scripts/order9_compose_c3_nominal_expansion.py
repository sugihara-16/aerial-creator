#!/usr/bin/env python3
from __future__ import annotations

"""Compose one or more new buckets into an accepted complete C3 nominal set."""

import argparse
import html
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetEntry,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file


COMPOSER_VERSION = "order9_c3_nominal_expansion_composer_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--expanded-bucket-manifest", required=True)
    parser.add_argument(
        "--addition-manifest", action="append", required=True
    )
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-index", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_path = _resolve(args.base_manifest)
    bucket_path = _resolve(args.expanded_bucket_manifest)
    addition_paths = [_resolve(value) for value in args.addition_manifest]
    output_path = _resolve(args.output_manifest)
    output_index = _resolve(args.output_index)
    if output_path.exists() or output_index.exists():
        raise FileExistsError(output_path if output_path.exists() else output_index)
    if output_path.parent != output_index.parent:
        raise ValueError("output manifest and index must share a directory")

    base = _load_set(base_path)
    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_path, repository_root=REPOSITORY_ROOT
    )
    buckets = load_order9_pi_l_rollout_bucket_manifest(bucket_path)
    if buckets.physical_model_hash != base.physical_model_hash:
        raise ValueError("expanded bucket and nominal PhysicalModel differ")
    base_entries = {entry.bucket_id: (entry, base_path.parent) for entry in base.entries}
    additions: dict[str, tuple[Order9C3NominalTrajectorySetEntry, Path, Path]] = {}
    evidence: dict[str, dict[str, object]] = {}
    for addition_path in addition_paths:
        addition = _load_set(addition_path)
        if addition.physical_model_hash != base.physical_model_hash:
            raise ValueError("addition PhysicalModel differs from base")
        validation_path = addition_path.parent / "collision_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not bool(validation.get("all_accepted")):
            raise ValueError(f"addition collision admission failed: {addition_path}")
        if hash_file(_resolve(addition.bucket_manifest_path)) != addition.bucket_manifest_sha256:
            raise ValueError("addition bucket manifest bytes changed")
        for entry in addition.entries:
            if entry.bucket_id in additions:
                raise ValueError(f"addition bucket repeats: {entry.bucket_id}")
            _validate_entry(addition_path.parent, entry)
            additions[entry.bucket_id] = (entry, addition_path.parent, validation_path)

    expanded_ids = [bucket.bucket_id for bucket in buckets.buckets]
    base_ids = set(base_entries)
    expected_additions = set(expanded_ids) - base_ids
    if set(additions) != expected_additions:
        raise ValueError(
            "addition manifests must cover exactly the expanded bucket ids: "
            f"expected={sorted(expected_additions)}, got={sorted(additions)}"
        )
    if not base_ids.issubset(expanded_ids):
        raise ValueError("expanded bucket manifest removed a base bucket")

    entries: list[Order9C3NominalTrajectorySetEntry] = []
    addition_indices: set[int] = set()
    for index, bucket in enumerate(buckets.buckets):
        if bucket.bucket_id in base_entries:
            incoming, root = base_entries[bucket.bucket_id]
        else:
            incoming, root, validation_path = additions[bucket.bucket_id]
            addition_indices.add(index)
            evidence[str(index)] = {
                "bucket_id": bucket.bucket_id,
                "component_manifest_path": _portable(
                    next(
                        path
                        for path in addition_paths
                        if path.parent == root
                    )
                ),
                "component_artifact_sha256": incoming.artifact_sha256,
                "collision_validation_path": _portable(validation_path),
                "collision_validation_sha256": hash_file(validation_path),
            }
        rewritten = _rewrite_entry(
            incoming, source_root=root, destination_root=output_path.parent
        )
        if (
            rewritten.split != bucket.split
            or rewritten.module_count != bucket.module_count
            or rewritten.structural_hash != bucket.structural_hash
        ):
            raise ValueError(f"nominal identity differs for {bucket.bucket_id}")
        _validate_entry(output_path.parent, rewritten)
        entries.append(rewritten)

    manifest = Order9C3NominalTrajectorySetManifest(
        bucket_manifest_path=_portable(bucket_path),
        bucket_manifest_sha256=hash_file(bucket_path),
        physical_model_hash=base.physical_model_hash,
        entries=entries,
        index_html_path=os.path.relpath(output_index, output_path.parent),
        metadata={
            **base.metadata,
            "composer_version": COMPOSER_VERSION,
            "base_manifest_path": _portable(base_path),
            "base_manifest_sha256": hash_file(base_path),
            "expanded_bucket_manifest_path": _portable(bucket_path),
            "expanded_bucket_manifest_sha256": hash_file(bucket_path),
            "addition_count": len(additions),
            "addition_evidence_by_index": evidence,
            "independent_convex_collision_replay": "all_additions_accepted",
        },
        manifest_version=base.manifest_version,
    )
    manifest.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_path, manifest.to_json(indent=2) + "\n")
    _write_index(output_index, entries, addition_indices)
    validate_order9_c3_nominal_trajectory_set_bytes(
        output_path, repository_root=REPOSITORY_ROOT
    )
    print(
        "ORDER9_C3_NOMINAL_EXPANDED="
        + json.dumps(
            {
                "path": str(output_path),
                "sha256": hash_file(output_path),
                "bucket_count": len(entries),
                "addition_count": len(additions),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _load_set(path: Path) -> Order9C3NominalTrajectorySetManifest:
    value = Order9C3NominalTrajectorySetManifest.from_json(
        path.read_text(encoding="utf-8")
    )
    value.validate()
    return value


def _rewrite_entry(
    entry: Order9C3NominalTrajectorySetEntry,
    *,
    source_root: Path,
    destination_root: Path,
) -> Order9C3NominalTrajectorySetEntry:
    def relative(value: str | None) -> str | None:
        if value is None:
            return None
        return os.path.relpath((source_root / value).resolve(), destination_root)

    return Order9C3NominalTrajectorySetEntry(
        bucket_id=entry.bucket_id,
        split=entry.split,
        module_count=entry.module_count,
        structural_hash=entry.structural_hash,
        artifact_path=str(relative(entry.artifact_path)),
        artifact_sha256=entry.artifact_sha256,
        animation_html_path=relative(entry.animation_html_path),
        animation_scene_path=relative(entry.animation_scene_path),
        animation_scene_sha256=entry.animation_scene_sha256,
    )


def _validate_entry(root: Path, entry: Order9C3NominalTrajectorySetEntry) -> None:
    artifact = root / entry.artifact_path
    if hash_file(artifact) != entry.artifact_sha256:
        raise ValueError(f"nominal artifact bytes changed: {entry.bucket_id}")
    validate_order9_c3_nominal_trajectory_artifact_bytes(artifact)
    if entry.animation_scene_path is not None:
        scene = root / entry.animation_scene_path
        if hash_file(scene) != entry.animation_scene_sha256:
            raise ValueError(f"animation scene bytes changed: {entry.bucket_id}")


def _write_index(
    path: Path,
    entries: list[Order9C3NominalTrajectorySetEntry],
    addition_indices: set[int],
) -> None:
    rows = []
    for index, entry in enumerate(entries):
        animation = (
            ""
            if entry.animation_html_path is None
            else f'<a href="{html.escape(entry.animation_html_path)}">open</a>'
        )
        rows.append(
            f'<tr class="{"addition" if index in addition_indices else ""}">'
            f"<td>{index}</td><td>{html.escape(entry.bucket_id)}</td>"
            f"<td>{entry.module_count}</td><td>{entry.split.value}</td>"
            f"<td>{animation}</td></tr>"
        )
    payload = """<!doctype html><html><head><meta charset="utf-8">
<title>C3 expanded nominal set</title><style>
body{font-family:sans-serif}table{border-collapse:collapse}td,th{padding:.35rem;border:1px solid #bbb}
tr.addition{background:#e8fff0}</style></head><body><h1>C3 expanded nominal set</h1>
<table><thead><tr><th>index</th><th>bucket</th><th>modules</th><th>split</th><th>viewer</th></tr></thead>
<tbody>""" + "\n".join(rows) + "</tbody></table></body></html>\n"
    _atomic_write(path, payload)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


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
