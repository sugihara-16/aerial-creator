#!/usr/bin/env python3
from __future__ import annotations

"""Compose reviewed C3 nominal replacements into a complete set."""

import argparse
import html
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetEntry,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.utils.hashing import hash_file


COMPOSER_VERSION = "order9_c3_nominal_replacement_composer_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument(
        "--replacement",
        action="append",
        required=True,
        metavar="INDEX=MANIFEST",
        help=(
            "Replace one source bucket index with the matching entry from "
            "MANIFEST. May be repeated."
        ),
    )
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-index", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_path = _resolve(args.base_manifest)
    output_manifest_path = _resolve(args.output_manifest)
    output_index_path = _resolve(args.output_index)
    if output_manifest_path.exists():
        raise FileExistsError(output_manifest_path)
    if output_index_path.exists():
        raise FileExistsError(output_index_path)
    if output_manifest_path.parent != output_index_path.parent:
        raise ValueError("output manifest and index must share one directory")

    base = _load_manifest(base_path)
    bucket_manifest_path = _resolve(base.bucket_manifest_path)
    if hash_file(bucket_manifest_path) != base.bucket_manifest_sha256:
        raise ValueError("base bucket manifest hash mismatch")
    bucket_payload = json.loads(
        bucket_manifest_path.read_text(encoding="utf-8")
    )
    buckets = bucket_payload.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != len(base.entries):
        raise ValueError("base entries do not cover the bucket manifest")
    for index, (entry, bucket) in enumerate(zip(base.entries, buckets)):
        if entry.bucket_id != bucket.get("bucket_id"):
            raise ValueError(f"base entry order mismatch at index {index}")
        _validate_entry_files(base_path.parent, entry)

    replacement_specs = _parse_replacements(args.replacement, len(buckets))
    entries = [
        Order9C3NominalTrajectorySetEntry.from_dict(entry.to_dict())
        for entry in base.entries
    ]
    replacement_evidence: dict[str, dict[str, Any]] = {}
    for index, component_path in replacement_specs:
        component = _load_manifest(component_path)
        _require_compatible(base, component)
        bucket = buckets[index]
        bucket_id = str(bucket["bucket_id"])
        matches = [
            entry for entry in component.entries
            if entry.bucket_id == bucket_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"replacement manifest must contain bucket {bucket_id} once"
            )
        incoming = matches[0]
        current = entries[index]
        if (
            incoming.split != current.split
            or incoming.module_count != current.module_count
            or incoming.structural_hash != current.structural_hash
        ):
            raise ValueError(f"replacement identity mismatch at index {index}")
        _validate_entry_files(component_path.parent, incoming)
        validation_path = component_path.parent / "collision_validation.json"
        validation = _load_collision_validation(
            validation_path,
            component_path=component_path,
            bucket_id=bucket_id,
        )
        rewritten = _rewrite_entry_paths(
            incoming,
            source_root=component_path.parent,
            destination_root=output_manifest_path.parent,
        )
        _validate_entry_files(output_manifest_path.parent, rewritten)
        entries[index] = rewritten
        replacement_evidence[str(index)] = {
            "bucket_id": bucket_id,
            "base_artifact_sha256": current.artifact_sha256,
            "replacement_artifact_sha256": incoming.artifact_sha256,
            "component_manifest_path": _portable(component_path),
            "component_manifest_sha256": hash_file(component_path),
            "collision_validation_path": _portable(validation_path),
            "collision_validation_sha256": hash_file(validation_path),
            "collision_validation_frame_count": int(
                validation["selected_record"]["frame_count"]
            ),
        }

    output = Order9C3NominalTrajectorySetManifest(
        bucket_manifest_path=base.bucket_manifest_path,
        bucket_manifest_sha256=base.bucket_manifest_sha256,
        physical_model_hash=base.physical_model_hash,
        entries=entries,
        index_html_path=os.path.relpath(
            output_index_path,
            output_manifest_path.parent,
        ),
        metadata={
            **base.metadata,
            "composer_version": COMPOSER_VERSION,
            "base_manifest_path": _portable(base_path),
            "base_manifest_sha256": hash_file(base_path),
            "replacement_count": len(replacement_evidence),
            "replacement_evidence_by_index": replacement_evidence,
            "independent_convex_collision_replay": (
                "replacement_entries_accepted; aggregate_pending"
            ),
        },
        manifest_version=base.manifest_version,
    )
    output.validate()
    _atomic_write_text(
        output_manifest_path,
        output.to_json(indent=2) + "\n",
    )
    _write_index(
        output_index_path,
        entries=entries,
        replacement_indices=set(dict(replacement_specs)),
    )
    print(
        "ORDER9_C3_REPLACEMENT_SET="
        f"path={output_manifest_path} sha256={hash_file(output_manifest_path)} "
        f"bucket_count={len(entries)} replacements={len(replacement_specs)}"
    )
    return 0


def _load_manifest(path: Path) -> Order9C3NominalTrajectorySetManifest:
    manifest = Order9C3NominalTrajectorySetManifest.from_json(
        path.read_text(encoding="utf-8")
    )
    manifest.validate()
    return manifest


def _parse_replacements(
    values: list[str],
    bucket_count: int,
) -> list[tuple[int, Path]]:
    parsed: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for raw in values:
        index_raw, separator, path_raw = raw.partition("=")
        if not separator or not path_raw:
            raise ValueError("--replacement must use INDEX=MANIFEST")
        index = int(index_raw)
        if index < 0 or index >= bucket_count:
            raise ValueError(f"replacement index out of range: {index}")
        if index in seen:
            raise ValueError(f"replacement index repeats: {index}")
        seen.add(index)
        parsed.append((index, _resolve(path_raw)))
    return sorted(parsed)


def _require_compatible(
    base: Order9C3NominalTrajectorySetManifest,
    component: Order9C3NominalTrajectorySetManifest,
) -> None:
    for name in (
        "manifest_version",
        "bucket_manifest_sha256",
        "physical_model_hash",
    ):
        if getattr(base, name) != getattr(component, name):
            raise ValueError(f"replacement {name} differs from base")


def _validate_entry_files(
    root: Path,
    entry: Order9C3NominalTrajectorySetEntry,
) -> None:
    artifact_path = root / entry.artifact_path
    if hash_file(artifact_path) != entry.artifact_sha256:
        raise ValueError(f"artifact hash mismatch: {entry.bucket_id}")
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
        artifact_path
    )
    if (
        artifact.bucket_id != entry.bucket_id
        or artifact.structural_hash != entry.structural_hash
    ):
        raise ValueError(f"artifact identity mismatch: {entry.bucket_id}")
    if entry.animation_scene_path is not None:
        scene_path = root / entry.animation_scene_path
        if hash_file(scene_path) != entry.animation_scene_sha256:
            raise ValueError(f"scene hash mismatch: {entry.bucket_id}")
        if not (root / str(entry.animation_html_path)).is_file():
            raise ValueError(f"viewer is missing: {entry.bucket_id}")


def _load_collision_validation(
    path: Path,
    *,
    component_path: Path,
    bucket_id: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    matching_records = (
        [
            record for record in records
            if isinstance(record, dict)
            and record.get("bucket_id") == bucket_id
        ]
        if isinstance(records, list)
        else []
    )
    if (
        payload.get("all_accepted") is not True
        or payload.get("nominal_manifest_sha256") != hash_file(component_path)
        or not isinstance(records, list)
        or len(matching_records) != 1
        or matching_records[0].get("accepted") is not True
    ):
        raise ValueError(f"invalid collision validation for {bucket_id}")
    payload["selected_record"] = matching_records[0]
    return payload


def _rewrite_entry_paths(
    entry: Order9C3NominalTrajectorySetEntry,
    *,
    source_root: Path,
    destination_root: Path,
) -> Order9C3NominalTrajectorySetEntry:
    payload = entry.to_dict()
    for name in (
        "artifact_path",
        "animation_html_path",
        "animation_scene_path",
    ):
        value = payload.get(name)
        if value is not None:
            payload[name] = os.path.relpath(
                source_root / str(value),
                destination_root,
            )
    return Order9C3NominalTrajectorySetEntry.from_dict(payload)


def _write_index(
    path: Path,
    *,
    entries: list[Order9C3NominalTrajectorySetEntry],
    replacement_indices: set[int],
) -> None:
    rows = []
    for index, entry in enumerate(entries):
        viewer_path = path.parent / str(entry.animation_html_path)
        viewer_href = os.path.relpath(viewer_path, path.parent)
        replaced = index in replacement_indices
        rows.append(
            f'<tr class="{"replacement" if replaced else ""}">'
            f"<td>{index:02d}</td>"
            f"<td>{html.escape(entry.bucket_id)}</td>"
            f"<td>{entry.module_count}</td>"
            f"<td>{html.escape(str(entry.split))}</td>"
            f"<td>{'new' if replaced else 'retained'}</td>"
            f'<td><a href="{html.escape(viewer_href)}">再生</a></td>'
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C3 nominal trajectories — posture replacements</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1120px;margin:28px auto;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3db;padding:6px 8px}}
th{{background:#edf2f7;position:sticky;top:0}}tr.replacement{{background:#fff4cc}}
code{{background:#eee;padding:2px 4px}}</style></head>
<body><h1>C3 nominal trajectories — posture replacements</h1>
<p>黄色の3件は、人間レビューで棄却されたanchor組を除外し、完全な
<code>pi_H teacher -&gt; deterministic collision-aware IK</code>軌道を再生成した差し替えです。</p>
<table><thead><tr><th>index</th><th>bucket</th><th>modules</th><th>split</th><th>source</th><th>viewer</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    _atomic_write_text(path, document)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
