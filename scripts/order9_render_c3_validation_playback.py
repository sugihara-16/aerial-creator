#!/usr/bin/env python3
from __future__ import annotations

"""Render saved C3 validation artifacts without rerunning Isaac or control."""

import argparse
import gc
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.simulation.order9_morphology_assets import (  # noqa: E402
    load_order9_morphology_asset_manifest,
)
from amsrr.training.order9_rollout_buckets import (  # noqa: E402
    load_order9_pi_l_rollout_bucket_manifest,
)
from amsrr.training.order9_tensor_rollout_artifact import (  # noqa: E402
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from amsrr.visualization.order9_c3_curation import (  # noqa: E402
    build_order9_c3_urdf_mesh_scene,
    write_order9_c3_stl_mesh_library,
    write_order9_c3_viewer_javascript,
)
from amsrr.visualization.order9_c3_rollout_playback import (  # noqa: E402
    load_order9_c3_evaluation_episode_rows,
    render_order9_c3_rollout_artifact_environment,
    write_order9_c3_rollout_playback_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument(
        "--validation-root",
        required=True,
        help="Directory containing buckets/<bucket-id>/evaluation_rollout.pt.",
    )
    parser.add_argument(
        "--bucket-id",
        action="append",
        required=True,
        help="Repeat to add multiple saved validation buckets.",
    )
    parser.add_argument(
        "--environment-index",
        action="append",
        type=int,
        help="Repeat to render selected environments; defaults to every one.",
    )
    parser.add_argument("--frames-per-second", type=int, default=10)
    parser.add_argument("--output-directory", required=True)
    return parser


def _resolve_repository_path(value: str | Path) -> Path:
    source = Path(value)
    return (
        source.resolve()
        if source.is_absolute()
        else (REPOSITORY_ROOT / source).resolve()
    )


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.frames_per_second <= 20:
        raise ValueError("--frames-per-second must lie within [1, 20]")
    manifest_path = _resolve_repository_path(args.bucket_manifest)
    validation_root = _resolve_repository_path(args.validation_root)
    output = _resolve_repository_path(args.output_directory)
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    by_bucket = {bucket.bucket_id: bucket for bucket in manifest.buckets}
    requested = []
    for bucket_id in args.bucket_id:
        if bucket_id not in by_bucket:
            raise ValueError(f"bucket {bucket_id!r} is absent from the manifest")
        if bucket_id not in requested:
            requested.append(bucket_id)
    if not manifest.source_asset_manifest_path:
        raise ValueError("bucket manifest lacks source_asset_manifest_path")
    asset_manifest_path = _resolve_repository_path(manifest.source_asset_manifest_path)
    asset_manifest = load_order9_morphology_asset_manifest(asset_manifest_path)
    asset_manifest.validate()
    asset_by_structural_hash = {
        entry.structural_hash: entry for entry in asset_manifest.entries
    }
    urdf_by_bucket: dict[str, Path] = {}
    asset_entry_by_bucket = {}
    mesh_paths = set()
    for bucket_id in requested:
        bucket = by_bucket[bucket_id]
        entry = asset_by_structural_hash.get(bucket.structural_hash)
        if entry is None:
            raise ValueError(f"bucket {bucket_id!r} lacks a morphology URDF")
        asset_entry_by_bucket[bucket_id] = entry
        urdf = _resolve_repository_path(entry.urdf_path)
        if hash_file(urdf) != entry.urdf_sha256:
            raise ValueError(f"bucket {bucket_id!r} morphology URDF hash differs")
        urdf_by_bucket[bucket_id] = urdf
        mesh_paths.update(build_order9_c3_urdf_mesh_scene(urdf).mesh_paths)

    shared = output / "shared"
    mesh_library = write_order9_c3_stl_mesh_library(
        mesh_paths, shared / "mesh_library.js"
    )
    viewer_javascript = write_order9_c3_viewer_javascript(shared / "viewer.js")
    cases = []
    for bucket_id in requested:
        bucket_root = validation_root / "buckets" / bucket_id
        artifact_path = bucket_root / "evaluation_rollout.pt"
        episode_path = bucket_root / "evaluation_episodes.jsonl"
        print(f"loading saved rollout: {bucket_id}", flush=True)
        artifact = load_order9_tensor_rollout_artifact(artifact_path)
        artifact_urdf_hash = artifact.metadata.get("urdf_hash")
        if (
            isinstance(artifact_urdf_hash, str)
            and artifact_urdf_hash != asset_manifest.source_urdf_sha256
        ):
            raise ValueError(
                f"bucket {bucket_id!r} playback asset uses source URDF "
                f"{asset_manifest.source_urdf_sha256}, but the saved Isaac "
                f"rollout used {artifact_urdf_hash}; select the hash-matched "
                "bucket/asset manifest"
            )
        artifact_usd_hash = artifact.metadata.get("robot_usd_sha256")
        expected_usd_hash = asset_entry_by_bucket[bucket_id].usd_sha256
        if (
            isinstance(artifact_usd_hash, str)
            and artifact_usd_hash != expected_usd_hash
        ):
            raise ValueError(
                f"bucket {bucket_id!r} playback asset uses USD "
                f"{expected_usd_hash}, but the saved Isaac rollout used "
                f"{artifact_usd_hash}; select the hash-matched bucket/asset "
                "manifest"
            )
        rows = load_order9_c3_evaluation_episode_rows(episode_path)
        environments = (
            list(range(artifact.environment_count))
            if args.environment_index is None
            else list(dict.fromkeys(args.environment_index))
        )
        for environment in environments:
            if environment not in rows:
                raise ValueError(
                    f"bucket {bucket_id!r} lacks episode row for env {environment}"
                )
            print(f"rendering {bucket_id} environment {environment}", flush=True)
            case, _viewer = render_order9_c3_rollout_artifact_environment(
                artifact=artifact,
                artifact_path=artifact_path,
                episode_row=rows[environment],
                robot_urdf_path=urdf_by_bucket[bucket_id],
                html_path=output / bucket_id / f"environment_{environment}.html",
                mesh_library_path=mesh_library,
                viewer_javascript_path=viewer_javascript,
                bucket_id=bucket_id,
                environment_index=environment,
                frames_per_second=args.frames_per_second,
            )
            cases.append(case)
        del artifact
        gc.collect()
    index_path = write_order9_c3_rollout_playback_index(output / "index.html", cases)
    summary = {
        "index_path": str(index_path),
        "case_count": len(cases),
        "bucket_ids": requested,
        "frames_per_second": args.frames_per_second,
        "physics_executed_during_playback": False,
        "revalidation_performed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
