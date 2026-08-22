#!/usr/bin/env python3
from __future__ import annotations

"""Upgrade accepted C3 grasp artifacts to complete eight-phase nominals."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskRuntimeConfig,
)
from amsrr.simulation.order9_object_task_state import (
    load_order9_canonical_reset,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION,
    ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetEntry,
    Order9C3NominalTrajectorySetManifest,
    materialize_order9_c3_complete_task_phases,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
)
from amsrr.utils.hashing import hash_file, stable_hash


LEGACY_ARTIFACT_VERSION = (
    "order9_c3_nominal_configuration_space_v5_local_contact_corridor"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest")
    parser.add_argument("output")
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Upgrade only the entries present in the source nominal set. "
            "The source entries must still belong to its hash-bound bucket "
            "manifest. This is intended for independently collision-validated "
            "replacement components."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = _resolve(args.source_manifest)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    entries = source_payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source nominal set has no entries")
    bucket_manifest_path = _resolve(source_payload["bucket_manifest_path"])
    if hash_file(bucket_manifest_path) != source_payload["bucket_manifest_sha256"]:
        raise ValueError("source nominal bucket manifest changed")
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(
        bucket_manifest_path
    )
    buckets = {value.bucket_id: value for value in bucket_manifest.buckets}
    entry_ids = {str(value["bucket_id"]) for value in entries}
    if args.allow_subset:
        unknown_ids = entry_ids - set(buckets)
        if unknown_ids:
            raise ValueError(
                "source nominal set contains buckets outside the bucket "
                f"manifest: {sorted(unknown_ids)}"
            )
    elif set(buckets) != entry_ids:
        raise ValueError("source nominal set does not cover the bucket manifest")
    curriculum = load_order9_learning_config(
        _resolve(args.curriculum_config)
    )
    canonical = load_order9_canonical_reset(
        _resolve(curriculum.production_runtime.canonical_order8_report_path),
        expected_sha256=(
            curriculum.production_runtime.canonical_order8_report_sha256
        ),
    )
    runtime_config = Order9ObjectTaskRuntimeConfig()
    output.mkdir(parents=True)
    upgraded_entries = []
    for index, source_entry in enumerate(entries):
        bucket_id = str(source_entry["bucket_id"])
        bucket = buckets[bucket_id]
        source_artifact_path = source.parent / source_entry["artifact_path"]
        if hash_file(source_artifact_path) != source_entry["artifact_sha256"]:
            raise ValueError(f"source artifact changed: {bucket_id}")
        source_artifact = json.loads(
            source_artifact_path.read_text(encoding="utf-8")
        )
        if source_artifact.get("artifact_version") != LEGACY_ARTIFACT_VERSION:
            raise ValueError(
                f"unexpected source artifact version: {bucket_id}"
            )
        destination = output / "buckets" / bucket_id
        shutil.copytree(source_artifact_path.parent, destination)
        grasp_phases = _load_legacy_grasp_phases(
            destination, source_artifact
        )
        task = TaskSpec.from_json(
            (bucket_manifest_path.parent / bucket.task_spec_path).read_text(
                encoding="utf-8"
            )
        )
        phases = materialize_order9_c3_complete_task_phases(
            phase_trajectories=grasp_phases,
            task_spec=task,
            lift_clearance_m=float(canonical.lift_clearance_m),
            retreat_offset_m=float(runtime_config.retreat_offset_m),
            phase_duration_s=runtime_config.phase_duration_s,
        )
        phase_artifacts = []
        for phase in (value.value for value in ORDER9_OBJECT_TASK_PHASES):
            trajectory = phases[phase]
            path = destination / "phases" / f"{phase}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                trajectory.to_json(indent=2) + "\n", encoding="utf-8"
            )
            phase_artifacts.append(
                Order9C3NominalPhaseArtifact(
                    phase=phase,
                    trajectory_path=str(path.relative_to(destination)),
                    trajectory_sha256=hash_file(path),
                    trajectory_hash=stable_hash(trajectory.to_dict()),
                    knot_count=len(trajectory.knots),
                    generation_method=(
                        "configuration_space_planner_resolved"
                        if phase in {"approach", "contact_acquisition"}
                        else "deterministic_complete_task_edge"
                    ),
                )
            )
        timeline = _complete_timeline(phases)
        duration_s = sum(
            float(value.horizon_s) for value in phases.values()
        )
        timeline_path = destination / "nominal_timeline.json"
        timeline_path.write_text(
            json.dumps(
                {
                    "timeline_version": ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
                    "semantic_scope": (
                        "offline precomputed complete eight-phase nominal; "
                        "configuration-space planning is never executed by "
                        "the training or inference runtime"
                    ),
                    "proxy_collision_validation_status": (
                        "pending_offline_admission"
                    ),
                    "duration_s": duration_s,
                    "records": timeline,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        source_artifact.update(
            {
                "artifact_version": ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
                "timeline_sha256": hash_file(timeline_path),
                "duration_s": duration_s,
                "final_phase": "settle",
                "final_phase_target_reached": True,
                "proxy_collision_validation_status": (
                    "pending_offline_admission"
                ),
                "phase_trajectories": [
                    value.to_dict() for value in phase_artifacts
                ],
            }
        )
        evidence = dict(source_artifact.get("selection_evidence") or {})
        evidence["complete_task_materialization"] = {
            "phase_count": len(phases),
            "lift_clearance_m": float(canonical.lift_clearance_m),
            "retreat_offset_m": float(runtime_config.retreat_offset_m),
            "release_semantics": "open_while_ascending_then_elevated_retreat",
            "configuration_space_planner_runtime_enabled": False,
        }
        source_artifact["selection_evidence"] = evidence
        artifact = Order9C3NominalTrajectoryArtifact.from_dict(
            source_artifact
        )
        artifact.validate()
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            artifact.to_json(indent=2) + "\n", encoding="utf-8"
        )
        validate_order9_c3_nominal_trajectory_artifact_bytes(manifest_path)
        upgraded_entries.append(
            Order9C3NominalTrajectorySetEntry(
                bucket_id=bucket_id,
                split=DatasetSplit(str(source_entry["split"])),
                module_count=int(source_entry["module_count"]),
                structural_hash=str(source_entry["structural_hash"]),
                artifact_path=str(manifest_path.relative_to(output)),
                artifact_sha256=hash_file(manifest_path),
            )
        )
        print(
            f"ORDER9_C3_COMPLETE_PHASE_UPGRADE={index + 1}/{len(entries)} "
            f"bucket={bucket_id}",
            flush=True,
        )
    manifest = Order9C3NominalTrajectorySetManifest(
        bucket_manifest_path=_portable(bucket_manifest_path),
        bucket_manifest_sha256=hash_file(bucket_manifest_path),
        physical_model_hash=str(source_payload["physical_model_hash"]),
        entries=upgraded_entries,
        metadata={
            "source_manifest_path": _portable(source),
            "source_manifest_sha256": hash_file(source),
            "source_subset_upgrade": bool(args.allow_subset),
            "complete_task_phase_count": len(ORDER9_OBJECT_TASK_PHASES),
            "configuration_space_planner_runtime_enabled": False,
            "release_semantics": "open_while_ascending_then_elevated_retreat",
        },
        manifest_version=ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION,
    )
    manifest.validate()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        manifest.to_json(indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"ORDER9_C3_COMPLETE_PHASE_SET=path={manifest_path} "
        f"sha256={hash_file(manifest_path)} buckets={len(upgraded_entries)}"
    )
    return 0


def _load_legacy_grasp_phases(
    root: Path,
    artifact: dict[str, object],
) -> dict[str, ContactWrenchTrajectory]:
    grouped: dict[str, list[tuple[float, ContactWrenchTrajectory]]] = {}
    for window in artifact["windows"]:
        path = root / window["resolved_trajectory_path"]
        if hash_file(path) != window["resolved_trajectory_sha256"]:
            raise ValueError(f"legacy resolved trajectory changed: {path}")
        trajectory = ContactWrenchTrajectory.from_json(
            path.read_text(encoding="utf-8")
        )
        trajectory.validate()
        grouped.setdefault(str(window["phase"]), []).append(
            (float(window["global_start_time_s"]), trajectory)
        )
    result = {}
    for phase in ("approach", "contact_acquisition"):
        windows = grouped.get(phase)
        if not windows:
            raise ValueError(f"legacy phase missing: {phase}")
        origin = windows[0][0]
        knots = []
        for global_start, trajectory in windows:
            for knot_index, knot in enumerate(trajectory.knots):
                time_s = global_start - origin + float(knot.t_rel_s)
                if knots and knot_index == 0 and math.isclose(
                    float(knots[-1].t_rel_s), time_s, abs_tol=1.0e-9
                ):
                    continue
                payload = knot.to_dict()
                payload["t_rel_s"] = time_s
                knots.append(InteractionKnot.from_dict(payload))
        result[phase] = ContactWrenchTrajectory(
            horizon_s=float(knots[-1].t_rel_s),
            dt_s=min(float(value.dt_s) for _, value in windows),
            knots=knots,
            derived_mode_label=f"order9_c3_accepted_nominal_phase_replay:{phase}",
            contract_version=windows[0][1].contract_version,
        )
        result[phase].validate()
    return result


def _complete_timeline(
    phases: dict[str, ContactWrenchTrajectory],
) -> list[dict[str, object]]:
    records = []
    offset = 0.0
    for phase in (value.value for value in ORDER9_OBJECT_TASK_PHASES):
        trajectory = phases[phase]
        for knot_index, knot in enumerate(trajectory.knots):
            if records and knot_index == 0:
                continue
            records.append(
                {
                    "sample_index": len(records),
                    "global_time_s": offset + float(knot.t_rel_s),
                    "phase": phase,
                    "phase_local_time_s": float(knot.t_rel_s),
                    "phase_target_reached": True,
                    "knot": knot.to_dict(),
                }
            )
        offset += float(trajectory.horizon_s)
    return records


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
