#!/usr/bin/env python3
from __future__ import annotations

"""Freeze human-reviewed C3 nominal trajectories into production buckets."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file


FINALIZER_VERSION = "order9_c3_nominal_human_acceptance_finalizer_v1"
HUMAN_REVIEW_VERSION = "order9_c3_nominal_human_review_v1"
BINDING_VERSION = "order9_c3_accepted_nominal_binding_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-set-manifest", required=True)
    parser.add_argument("--base-bucket-manifest", required=True)
    parser.add_argument("--human-review-output", required=True)
    parser.add_argument("--bound-bucket-output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    nominal_path = _resolve(args.nominal_set_manifest)
    bucket_path = _resolve(args.base_bucket_manifest)
    review_path = _resolve(args.human_review_output)
    bound_path = _resolve(args.bound_bucket_output)
    for output in (review_path, bound_path):
        if output.exists():
            raise FileExistsError(output)
    if bound_path.parent != bucket_path.parent:
        raise ValueError(
            "bound bucket output must share the base manifest directory"
        )
    nominal = validate_order9_c3_nominal_trajectory_set_bytes(
        nominal_path,
        repository_root=REPOSITORY_ROOT,
    )
    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_path,
        repository_root=REPOSITORY_ROOT,
    )
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(bucket_path)
    if hash_file(bucket_path) != nominal.bucket_manifest_sha256:
        raise ValueError("nominal set and base bucket manifest differ")
    if [entry.bucket_id for entry in nominal.entries] != [
        bucket.bucket_id for bucket in bucket_manifest.buckets
    ]:
        raise ValueError("nominal set and base bucket order differ")

    review_entries = []
    bindings: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(nominal.entries):
        artifact_path = nominal_path.parent / entry.artifact_path
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            artifact_path
        )
        scene_path = nominal_path.parent / str(entry.animation_scene_path)
        collision_path = artifact_path.parent.parent.parent / "collision_validation.json"
        binding = {
            "binding_version": BINDING_VERSION,
            "source_index": index,
            "set_manifest_path": _portable(nominal_path),
            "set_manifest_sha256": hash_file(nominal_path),
            "artifact_path": _portable(artifact_path),
            "artifact_sha256": entry.artifact_sha256,
            "timeline_path": _portable(artifact_path.parent / artifact.timeline_path),
            "timeline_sha256": artifact.timeline_sha256,
            "animation_scene_path": _portable(scene_path),
            "animation_scene_sha256": entry.animation_scene_sha256,
            "collision_validation_path": _portable(collision_path),
            "collision_validation_sha256": hash_file(collision_path),
            "selected_surface_port_ids": list(
                artifact.selected_surface_port_ids
            ),
            "nominal_duration_s": float(artifact.duration_s),
            "nominal_reference_rate_hz": 10.0,
            "configuration_space_planner_runtime_enabled": False,
        }
        bindings[entry.bucket_id] = binding
        review_entries.append(
            {
                "source_index": index,
                "bucket_id": entry.bucket_id,
                "split": entry.split.value,
                "module_count": entry.module_count,
                "structural_hash": entry.structural_hash,
                "decision": "accepted",
                "artifact_path": binding["artifact_path"],
                "artifact_sha256": binding["artifact_sha256"],
                "animation_scene_path": binding["animation_scene_path"],
                "animation_scene_sha256": binding[
                    "animation_scene_sha256"
                ],
                "collision_validation_path": binding[
                    "collision_validation_path"
                ],
                "collision_validation_sha256": binding[
                    "collision_validation_sha256"
                ],
                "selected_surface_port_ids": binding[
                    "selected_surface_port_ids"
                ],
            }
        )

    review = {
        "review_version": HUMAN_REVIEW_VERSION,
        "review_authority": "repository_user",
        "review_assertion": (
            "all final mesh animations were visually inspected and accepted"
        ),
        "decision_scope": (
            "ideal-tracking nominal approach/contact trajectory geometry; "
            "not learned-policy or dynamics success"
        ),
        "nominal_set_manifest_path": _portable(nominal_path),
        "nominal_set_manifest_sha256": hash_file(nominal_path),
        "accepted_count": len(review_entries),
        "rejected_count": 0,
        "entries": review_entries,
    }
    _atomic_write_json(review_path, review)

    payload = json.loads(bucket_path.read_text(encoding="utf-8"))
    for bucket in payload["buckets"]:
        bucket["metadata"] = {
            **bucket.get("metadata", {}),
            "accepted_nominal_trajectory": bindings[str(bucket["bucket_id"])],
        }
    payload["metadata"] = {
        **payload.get("metadata", {}),
        "c3_accepted_nominal_trajectory_bound": True,
        "c3_accepted_nominal_binding_version": BINDING_VERSION,
        "c3_nominal_set_manifest_path": _portable(nominal_path),
        "c3_nominal_set_manifest_sha256": hash_file(nominal_path),
        "c3_human_review_manifest_path": _portable(review_path),
        "c3_human_review_manifest_sha256": hash_file(review_path),
        "c3_human_review_accepted_count": len(review_entries),
        "c3_nominal_reference_rate_hz": 10.0,
        "c3_configuration_space_planner_runtime_enabled": False,
        "c3_runtime_teacher_semantics": "offline_precomputed_nominal_replay",
        "c3_nominal_finalizer_version": FINALIZER_VERSION,
        "source_bucket_manifest_path": _portable(bucket_path),
        "source_bucket_manifest_sha256": hash_file(bucket_path),
    }
    _atomic_write_json(bound_path, payload)
    validate_order9_pi_l_rollout_bucket_bytes(
        bound_path,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        "ORDER9_C3_NOMINAL_FINALIZED="
        f"count={len(review_entries)} nominal_sha256={hash_file(nominal_path)} "
        f"review={review_path} review_sha256={hash_file(review_path)} "
        f"buckets={bound_path} bucket_sha256={hash_file(bound_path)}"
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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
