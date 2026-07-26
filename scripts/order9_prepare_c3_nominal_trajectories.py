#!/usr/bin/env python3
from __future__ import annotations

"""Generate and visualize ideal-tracking nominal trajectories for C3 buckets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
    validate_order9_morphology_asset_manifest_bytes,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetEntry,
    Order9C3NominalTrajectorySetManifest,
    generate_order9_c3_nominal_grasp_trajectory,
    write_order9_c3_nominal_trajectory_artifact,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash
from amsrr.visualization.order9_c3_curation import (
    build_order9_c3_urdf_mesh_scene,
    write_order9_c3_stl_mesh_library,
    write_order9_c3_viewer_javascript,
)
from amsrr.visualization.order9_c3_nominal_mesh_animation import (
    render_order9_c3_nominal_mesh_animation,
)


DEFAULT_BUCKET_MANIFEST = (
    "artifacts/p4_full/order9/stages/"
    "c3_pi_l_ppo_arbitrary_morphology/"
    "rollout_buckets_current_lineage_v5/manifest.json"
)
DEFAULT_ASSET_MANIFEST = (
    "artifacts/p4_full/order9/c3_preparation/"
    "morphology_assets_grasp_frame_mesh_v2/manifest.json"
)
DEFAULT_OUTPUT = (
    "artifacts/p4_full/order9/stages/"
    "c3_pi_l_ppo_arbitrary_morphology/"
    "nominal_trajectories_bucket_support_ground_v4"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-manifest", default=DEFAULT_BUCKET_MANIFEST)
    parser.add_argument("--asset-manifest", default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument("--robot-model-config")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-second", type=int, default=10)
    parser.add_argument("--maximum-windows-per-phase", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic only: process the first N buckets.",
    )
    parser.add_argument(
        "--bucket-indices",
        help=(
            "Comma-separated source-manifest bucket indices. This is "
            "mutually exclusive with --limit."
        ),
    )
    parser.add_argument(
        "--reselect-contacts",
        action="store_true",
        help=(
            "Ignore the archived precheck contact pair and select contacts "
            "under the current collision/support semantics."
        ),
    )
    parser.add_argument(
        "--preferred-surface-port-ids",
        help=(
            "Force one unordered pair of surface port ids as A,B while "
            "still searching contact groups and the complete trajectory."
        ),
    )
    parser.add_argument(
        "--reviewed-contact-overrides",
        help=(
            "Hash-bound accepted C3a surface pairs keyed by bucket id. "
            "The current task geometry is still re-solved from scratch."
        ),
    )
    parser.add_argument(
        "--posture-rejection-overrides",
        help=(
            "Hash-bound human-rejected surface pairs keyed by bucket id. "
            "Requires --reselect-contacts."
        ),
    )
    parser.set_defaults(enforce_proxy_collision=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Independent bucket workers (default: up to 4 CPU processes).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    preferred_surface_port_ids = (
        None
        if args.preferred_surface_port_ids is None
        else _parse_surface_port_pair(args.preferred_surface_port_ids)
    )
    if preferred_surface_port_ids is not None and (
        args.reselect_contacts
        or args.reviewed_contact_overrides
        or args.posture_rejection_overrides
    ):
        raise ValueError(
            "--preferred-surface-port-ids is mutually exclusive with "
            "contact reselection, reviewed overrides, and posture "
            "rejection overrides"
        )
    if args.reselect_contacts and args.reviewed_contact_overrides:
        raise ValueError(
            "--reselect-contacts and --reviewed-contact-overrides are "
            "mutually exclusive"
        )
    if args.posture_rejection_overrides and not args.reselect_contacts:
        raise ValueError(
            "--posture-rejection-overrides requires --reselect-contacts"
        )
    if args.posture_rejection_overrides and args.reviewed_contact_overrides:
        raise ValueError(
            "--posture-rejection-overrides and --reviewed-contact-overrides "
            "are mutually exclusive"
        )
    bucket_manifest_path = _resolve(args.bucket_manifest)
    if bucket_manifest_path.is_dir():
        bucket_manifest_path = bucket_manifest_path / "manifest.json"
    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_manifest_path,
        repository_root=REPOSITORY_ROOT,
    )
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(
        bucket_manifest_path
    )
    asset_manifest_path = _resolve(args.asset_manifest)
    asset_manifest = Order9MorphologyAssetManifest.from_json(
        asset_manifest_path.read_text(encoding="utf-8")
    )
    validate_order9_morphology_asset_manifest_bytes(
        asset_manifest,
        repository_root=REPOSITORY_ROOT,
        expected_pool_sha256=bucket_manifest.source_pool_sha256,
    )
    if (
        bucket_manifest.source_asset_manifest_sha256
        != hash_file(asset_manifest_path)
    ):
        raise SchemaValidationError(
            "C3 nominal asset manifest differs from the bucket lineage"
        )
    curriculum = load_order9_learning_config(
        _resolve(args.curriculum_config)
    )
    physical_model = build_physical_model_from_config(
        _resolve(
            args.robot_model_config
            or curriculum.production_runtime.robot_model_config_path
        )
    )
    if physical_model.stable_hash() != bucket_manifest.physical_model_hash:
        raise SchemaValidationError(
            "C3 nominal PhysicalModel differs from the bucket lineage"
        )
    buckets = list(bucket_manifest.buckets)
    selected_source_indices = list(range(len(buckets)))
    if args.bucket_indices is not None and args.limit is not None:
        raise ValueError("--bucket-indices and --limit are mutually exclusive")
    if args.bucket_indices is not None:
        selected_source_indices = _parse_bucket_indices(
            args.bucket_indices,
            bucket_count=len(buckets),
        )
        buckets = [buckets[index] for index in selected_source_indices]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        buckets = buckets[: args.limit]
        selected_source_indices = selected_source_indices[: args.limit]
    reviewed_override_path = (
        None
        if args.reviewed_contact_overrides is None
        else _resolve(args.reviewed_contact_overrides)
    )
    reviewed_overrides = (
        {}
        if reviewed_override_path is None
        else _load_reviewed_contact_overrides(
            reviewed_override_path,
            buckets=buckets,
        )
    )
    posture_rejection_path = (
        None
        if args.posture_rejection_overrides is None
        else _resolve(args.posture_rejection_overrides)
    )
    posture_rejections = (
        {}
        if posture_rejection_path is None
        else _load_posture_rejection_overrides(
            posture_rejection_path,
            buckets=buckets,
            bucket_manifest_path=bucket_manifest_path,
        )
    )
    destination = _resolve(args.output)
    if destination.exists():
        raise FileExistsError(
            f"C3 nominal trajectory output exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    started = time.perf_counter()
    try:
        bucket_inputs = []
        mesh_paths: set[Path] = set()
        for bucket in buckets:
            task_path = bucket_manifest_path.parent / bucket.task_spec_path
            graph_path = bucket_manifest_path.parent / bucket.morphology_graph_path
            task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
            graph = MorphologyGraph.from_json(
                graph_path.read_text(encoding="utf-8")
            )
            asset = asset_manifest.entry_for(graph)
            urdf_path = _resolve(asset.urdf_path)
            neutral_scene = build_order9_c3_urdf_mesh_scene(urdf_path)
            mesh_paths.update(neutral_scene.mesh_paths)
            bucket_inputs.append(
                (bucket, task, graph, urdf_path)
            )
        shared_dir = temporary / "shared"
        mesh_library_path = write_order9_c3_stl_mesh_library(
            mesh_paths,
            shared_dir / "mesh_library.js",
        )
        viewer_javascript_path = write_order9_c3_viewer_javascript(
            shared_dir / "mesh_viewer.js"
        )
        jobs = [
            (
                index,
                bucket,
                task,
                graph,
                urdf_path,
                physical_model,
                temporary,
                mesh_library_path,
                viewer_javascript_path,
                args.frames_per_second,
                args.maximum_windows_per_phase,
                args.enforce_proxy_collision,
                args.reselect_contacts,
                preferred_surface_port_ids,
                reviewed_overrides.get(bucket.bucket_id),
                posture_rejections.get(bucket.bucket_id),
                bucket_manifest_path,
                hash_file(bucket_manifest_path),
                bucket_manifest.physical_model_hash,
            )
            for index, (bucket, task, graph, urdf_path) in enumerate(
                bucket_inputs
            )
        ]
        completed: dict[
            int,
            tuple[
                Order9C3NominalTrajectorySetEntry,
                dict[str, object],
                float,
            ],
        ] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_jobs = {
                executor.submit(_prepare_bucket, job): job[0]
                for job in jobs
            }
            for completed_count, future in enumerate(
                as_completed(future_jobs),
                start=1,
            ):
                index, entry_dict, row, wall_s = future.result()
                entry = Order9C3NominalTrajectorySetEntry.from_dict(
                    entry_dict
                )
                completed[index] = (entry, row, wall_s)
                print(
                    "ORDER9_C3_NOMINAL_PROGRESS="
                    f"{completed_count}/{len(jobs)} "
                    f"{entry.bucket_id} wall_s={wall_s:.3f}",
                    flush=True,
                )
        entries = [completed[index][0] for index in range(len(jobs))]
        index_rows = [completed[index][1] for index in range(len(jobs))]
        index_path = temporary / "index.html"
        _write_index(index_path, index_rows)
        manifest = Order9C3NominalTrajectorySetManifest(
            bucket_manifest_path=_portable(
                bucket_manifest_path,
                REPOSITORY_ROOT,
            ),
            bucket_manifest_sha256=hash_file(bucket_manifest_path),
            physical_model_hash=bucket_manifest.physical_model_hash,
            entries=entries,
            index_html_path=_relative(index_path, temporary),
            metadata={
                "bucket_count": len(entries),
                "train_bucket_count": sum(
                    value.split.value == "train" for value in entries
                ),
                "validation_bucket_count": sum(
                    value.split.value == "validation" for value in entries
                ),
                "frames_per_second": args.frames_per_second,
                "maximum_windows_per_phase": (
                    args.maximum_windows_per_phase
                ),
                "worker_process_count": args.workers,
                "source_bucket_indices": selected_source_indices,
                "contact_assignment_reselected": bool(
                    args.reselect_contacts
                    or preferred_surface_port_ids is not None
                ),
                "preferred_surface_port_ids": (
                    None
                    if preferred_surface_port_ids is None
                    else list(preferred_surface_port_ids)
                ),
                "reviewed_contact_override_path": (
                    None
                    if reviewed_override_path is None
                    else _portable(
                        reviewed_override_path,
                        REPOSITORY_ROOT,
                    )
                ),
                "reviewed_contact_override_sha256": (
                    None
                    if reviewed_override_path is None
                    else hash_file(reviewed_override_path)
                ),
                "reviewed_contact_override_count": len(
                    reviewed_overrides
                ),
                "posture_rejection_override_path": (
                    None
                    if posture_rejection_path is None
                    else _portable(
                        posture_rejection_path,
                        REPOSITORY_ROOT,
                    )
                ),
                "posture_rejection_override_sha256": (
                    None
                    if posture_rejection_path is None
                    else hash_file(posture_rejection_path)
                ),
                "posture_rejection_override_count": len(
                    posture_rejections
                ),
                "asset_manifest_path": _portable(
                    asset_manifest_path,
                    REPOSITORY_ROOT,
                ),
                "asset_manifest_sha256": hash_file(asset_manifest_path),
                "ideal_tracking_only": True,
                "isaac_dynamics": False,
                "overhead_approach_required": True,
                "bucket_support_collision": True,
                "ground_halfspace_collision": True,
                "proxy_collision_validation_status": (
                    "enforced_during_generation"
                    if args.enforce_proxy_collision
                    else "pending_offline_admission"
                ),
            },
        )
        manifest.validate()
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            manifest.to_json(indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    final_manifest = destination / "manifest.json"
    print(
        "ORDER9_C3_NOMINAL_SET="
        f"path={final_manifest} sha256={hash_file(final_manifest)} "
        f"bucket_count={len(buckets)} "
        f"wall_s={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return 0


def _prepare_bucket(
    job: tuple,
) -> tuple[int, dict[str, object], dict[str, object], float]:
    (
        index,
        bucket,
        task,
        graph,
        urdf_path,
        physical_model,
        temporary,
        mesh_library_path,
        viewer_javascript_path,
        frames_per_second,
        maximum_windows_per_phase,
        enforce_proxy_collision,
        reselect_contacts,
        preferred_surface_port_ids,
        reviewed_contact_override,
        posture_rejection_override,
        bucket_manifest_path,
        bucket_manifest_sha256,
        physical_model_hash,
    ) = job
    reviewed_surface_ids = (
        None
        if reviewed_contact_override is None
        else tuple(
            int(value)
            for value in reviewed_contact_override["surface_port_ids"]
        )
    )
    reviewed_group_id = (
        None
        if reviewed_contact_override is None
        else str(reviewed_contact_override["candidate_group_id"])
    )
    bucket_started = time.perf_counter()
    if preferred_surface_port_ids is not None:
        print(
            "ORDER9_C3_NOMINAL_SELECTION_REQUEST="
            + json.dumps(
                {
                    "bucket_id": bucket.bucket_id,
                    "preferred_surface_port_ids": list(
                        preferred_surface_port_ids
                    ),
                    "candidate_group_id": None,
                    "fallback_to_other_surface_pair": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    contact_goal_joint_seed = (
        None
        if reviewed_contact_override is None
        else reviewed_contact_override[
            "contact_goal_joint_seed_positions_rad"
        ]
    )

    def report_window_progress(record: dict[str, object]) -> None:
        print(
            "ORDER9_C3_NOMINAL_WINDOW="
            + json.dumps(
                {
                    "bucket_id": bucket.bucket_id,
                    **record,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    result = generate_order9_c3_nominal_grasp_trajectory(
        task_spec=task,
        structural_target=graph,
        physical_model=physical_model,
        maximum_windows_per_phase=maximum_windows_per_phase,
        enforce_proxy_collision_during_generation=enforce_proxy_collision,
        preferred_surface_port_ids=(
            preferred_surface_port_ids
            if preferred_surface_port_ids is not None
            else (
                reviewed_surface_ids
                if reviewed_surface_ids is not None
                else (
                    None
                    if reselect_contacts
                    else _prechecked_surface_ids(task)
                )
            )
        ),
        preferred_candidate_group_id=(
            None
            if preferred_surface_port_ids is not None
            else (
                reviewed_group_id
                if reviewed_group_id is not None
                else (
                    None
                    if reselect_contacts
                    else _prechecked_group_id(task)
                )
            )
        ),
        excluded_surface_port_id_pairs=(
            ()
            if posture_rejection_override is None
            else tuple(
                tuple(int(value) for value in pair)
                for pair in posture_rejection_override[
                    "excluded_surface_port_id_pairs"
                ]
            )
        ),
        contact_goal_joint_seed_positions_rad=(
            contact_goal_joint_seed
        ),
        progress_callback=report_window_progress,
    )
    artifact_dir = temporary / "buckets" / bucket.bucket_id
    artifact = write_order9_c3_nominal_trajectory_artifact(
        result,
        output_dir=artifact_dir,
        bucket_id=bucket.bucket_id,
        split=bucket.split,
        task_spec_sha256=bucket.task_spec_sha256,
        structural_hash=bucket.structural_hash,
        physical_model_hash=physical_model_hash,
        robot_urdf_path=urdf_path,
    )
    viewer_path = (
        temporary
        / "viewers"
        / bucket.bucket_id
        / "nominal_mesh_animation.html"
    )
    viewer = render_order9_c3_nominal_mesh_animation(
        result=result,
        task_spec=task,
        physical_model=physical_model,
        robot_urdf_path=urdf_path,
        html_path=viewer_path,
        mesh_library_path=mesh_library_path,
        viewer_javascript_path=viewer_javascript_path,
        bucket_id=bucket.bucket_id,
        frames_per_second=frames_per_second,
        source_metadata={
            "bucket_manifest_path": str(bucket_manifest_path),
            "bucket_manifest_sha256": bucket_manifest_sha256,
            "nominal_artifact_path": str(
                artifact_dir / "manifest.json"
            ),
            "nominal_artifact_sha256": hash_file(
                artifact_dir / "manifest.json"
            ),
            "physical_model_hash": physical_model_hash,
            "structural_hash": bucket.structural_hash,
            "reviewed_contact_override": reviewed_contact_override,
            "posture_rejection_override": posture_rejection_override,
        },
    )
    entry = Order9C3NominalTrajectorySetEntry(
        bucket_id=bucket.bucket_id,
        split=bucket.split,
        module_count=bucket.module_count,
        structural_hash=bucket.structural_hash,
        artifact_path=_relative(
            artifact_dir / "manifest.json",
            temporary,
        ),
        artifact_sha256=hash_file(
            artifact_dir / "manifest.json"
        ),
        animation_html_path=_relative(
            viewer.html_path,
            temporary,
        ),
        animation_scene_path=_relative(
            viewer.scene_path,
            temporary,
        ),
        animation_scene_sha256=viewer.scene_sha256,
    )
    row = {
        "bucket_id": bucket.bucket_id,
        "split": bucket.split.value,
        "module_count": bucket.module_count,
        "window_count": len(result.windows),
        "duration_s": artifact.duration_s,
        "surface_ids": artifact.selected_surface_port_ids,
        "viewer_path": _relative(viewer.html_path, temporary),
    }
    return (
        int(index),
        entry.to_dict(),
        row,
        time.perf_counter() - bucket_started,
    )


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    body = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['bucket_id']))}</td>"
        f"<td>{html.escape(str(row['split']))}</td>"
        f"<td>{int(row['module_count'])}</td>"
        f"<td>{int(row['window_count'])}</td>"
        f"<td>{float(row['duration_s']):.2f}</td>"
        f"<td>{html.escape(str(row['surface_ids']))}</td>"
        f"<td><a href=\"{html.escape(str(row['viewer_path']))}\">再生</a></td>"
        "</tr>"
        for row in rows
    )
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Order 9 C3 nominal trajectories</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f7fa;color:#17202a}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #d8dee4;padding:7px 9px;text-align:left}}
th{{background:#eef2f6;position:sticky;top:0}} a{{font-weight:650}}
</style></head><body>
<h1>Order 9 C3 nominal trajectories</h1>
<p>π_H teacher + deterministic IKの理想追従mesh animationです。
Isaac dynamics、学習済みπ_L、offline collision admissionの
成功証跡ではありません。</p>
<table><thead><tr><th>bucket</th><th>split</th><th>modules</th>
<th>windows</th><th>duration [s]</th><th>surfaces</th><th>viewer</th>
</tr></thead><tbody>{body}</tbody></table></body></html>
""",
        encoding="utf-8",
    )


def _prechecked_teacher_evidence(task: TaskSpec) -> dict[str, object]:
    value = task.metadata.get("order9_c3_articulated_teacher_precheck")
    if not isinstance(value, dict):
        raise SchemaValidationError(
            "C3 nominal bucket lacks articulated-teacher precheck evidence"
        )
    return value


def _prechecked_surface_ids(task: TaskSpec) -> tuple[int, int]:
    raw = _prechecked_teacher_evidence(task).get(
        "selected_surface_port_ids"
    )
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or len({int(value) for value in raw}) != 2
    ):
        raise SchemaValidationError(
            "C3 nominal bucket precheck surface identity is invalid"
        )
    return int(raw[0]), int(raw[1])


def _prechecked_group_id(task: TaskSpec) -> str:
    raw = _prechecked_teacher_evidence(task).get("candidate_group_id")
    if not isinstance(raw, str) or not raw:
        raise SchemaValidationError(
            "C3 nominal bucket precheck candidate group is invalid"
        )
    return raw


def _load_reviewed_contact_overrides(
    path: Path,
    *,
    buckets: list[object],
) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("override_version")
        != "order9_c3_reviewed_contact_override_v1"
        or not isinstance(payload.get("entries"), list)
    ):
        raise SchemaValidationError(
            "reviewed contact override has an invalid version or entries"
        )
    bucket_by_id = {str(bucket.bucket_id): bucket for bucket in buckets}
    entries = payload["entries"]
    entry_ids = [
        str(value.get("bucket_id"))
        for value in entries
        if isinstance(value, dict)
    ]
    if (
        len(entry_ids) != len(entries)
        or len(entry_ids) != len(set(entry_ids))
        or not set(bucket_by_id).issubset(set(entry_ids))
    ):
        raise SchemaValidationError(
            "reviewed contact overrides must cover every selected bucket "
            "and must not repeat bucket ids"
        )
    output: dict[str, dict[str, object]] = {}
    for raw in entries:
        bucket_id = str(raw["bucket_id"])
        if bucket_id not in bucket_by_id:
            continue
        bucket = bucket_by_id[bucket_id]
        structural_hash = str(raw.get("structural_hash", ""))
        if structural_hash != str(bucket.structural_hash):
            raise SchemaValidationError(
                f"reviewed contact structural hash differs for {bucket_id}"
            )
        surface_ids_raw = raw.get("surface_port_ids")
        if (
            not isinstance(surface_ids_raw, list)
            or len(surface_ids_raw) != 2
            or len({int(value) for value in surface_ids_raw}) != 2
            or min(int(value) for value in surface_ids_raw) < 0
        ):
            raise SchemaValidationError(
                f"reviewed contact surface pair is invalid for {bucket_id}"
            )
        surface_ids = [int(value) for value in surface_ids_raw]
        candidate_group_id = raw.get("candidate_group_id")
        review_case_id = raw.get("review_case_id")
        if (
            not isinstance(candidate_group_id, str)
            or not candidate_group_id
            or not isinstance(review_case_id, str)
            or not review_case_id
        ):
            raise SchemaValidationError(
                f"reviewed contact identities are invalid for {bucket_id}"
            )
        review_manifest_path = _resolve(
            str(raw.get("review_manifest_path", ""))
        )
        expected_review_sha = str(
            raw.get("review_manifest_sha256", "")
        )
        if (
            len(expected_review_sha) != 64
            or hash_file(review_manifest_path) != expected_review_sha
        ):
            raise SchemaValidationError(
                f"review manifest identity differs for {bucket_id}"
            )
        review_manifest = json.loads(
            review_manifest_path.read_text(encoding="utf-8")
        )
        cases = review_manifest.get("cases")
        if not isinstance(cases, list):
            raise SchemaValidationError(
                f"review manifest lacks cases for {bucket_id}"
            )
        matches = [
            value
            for value in cases
            if isinstance(value, dict)
            and value.get("case_id") == review_case_id
        ]
        if len(matches) != 1:
            raise SchemaValidationError(
                f"review case identity is not unique for {bucket_id}"
            )
        case = matches[0]
        if (
            case.get("review_status") != "accepted_by_user"
            or case.get("structural_hash") != structural_hash
            or [int(value) for value in case.get("surface_port_ids", [])]
            != surface_ids
            or case.get("candidate_group_id") != candidate_group_id
        ):
            raise SchemaValidationError(
                f"reviewed contact evidence disagrees for {bucket_id}"
            )
        final_pose = case.get("final_pose")
        final_ik = (
            final_pose.get("ik") if isinstance(final_pose, dict) else None
        )
        joint_seed = (
            final_ik.get("joint_positions_rad")
            if isinstance(final_ik, dict)
            else None
        )
        collision_aware = (
            final_pose.get("collision_aware")
            if isinstance(final_pose, dict)
            else None
        )
        expected_joint_seed_hash = (
            collision_aware.get("joint_solution_hash")
            if isinstance(collision_aware, dict)
            else None
        )
        if (
            not isinstance(joint_seed, dict)
            or not joint_seed
            or any(
                not isinstance(joint_id, str)
                or not joint_id
                or not isinstance(value, (int, float))
                for joint_id, value in joint_seed.items()
            )
            or not isinstance(expected_joint_seed_hash, str)
            or stable_hash(joint_seed) != expected_joint_seed_hash
        ):
            raise SchemaValidationError(
                f"reviewed contact joint seed is invalid for {bucket_id}"
            )
        output[bucket_id] = {
            "candidate_group_id": candidate_group_id,
            "review_case_id": review_case_id,
            "review_manifest_path": _portable(
                review_manifest_path,
                REPOSITORY_ROOT,
            ),
            "review_manifest_sha256": expected_review_sha,
            "structural_hash": structural_hash,
            "surface_port_ids": surface_ids,
            "contact_goal_joint_seed_positions_rad": {
                str(joint_id): float(value)
                for joint_id, value in joint_seed.items()
            },
            "contact_goal_joint_seed_hash": expected_joint_seed_hash,
        }
    return output


def _load_posture_rejection_overrides(
    path: Path,
    *,
    buckets: list[object],
    bucket_manifest_path: Path,
) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("rejection_version")
        != "order9_c3_human_posture_rejection_v1"
        or not isinstance(payload.get("entries"), list)
    ):
        raise SchemaValidationError(
            "posture rejection override has an invalid version or entries"
        )
    if (
        payload.get("bucket_manifest_path")
        != _portable(bucket_manifest_path, REPOSITORY_ROOT)
        or payload.get("bucket_manifest_sha256")
        != hash_file(bucket_manifest_path)
    ):
        raise SchemaValidationError(
            "posture rejection override differs from the bucket lineage"
        )
    bucket_by_id = {str(bucket.bucket_id): bucket for bucket in buckets}
    raw_entries = payload["entries"]
    entry_ids = [
        str(value.get("bucket_id"))
        for value in raw_entries
        if isinstance(value, dict)
    ]
    if (
        len(entry_ids) != len(raw_entries)
        or len(entry_ids) != len(set(entry_ids))
        or not set(bucket_by_id).issubset(set(entry_ids))
    ):
        raise SchemaValidationError(
            "posture rejection overrides must cover every selected bucket "
            "and must not repeat bucket ids"
        )
    output: dict[str, dict[str, object]] = {}
    for raw in raw_entries:
        bucket_id = str(raw["bucket_id"])
        if bucket_id not in bucket_by_id:
            continue
        bucket = bucket_by_id[bucket_id]
        structural_hash = str(raw.get("structural_hash", ""))
        if structural_hash != str(bucket.structural_hash):
            raise SchemaValidationError(
                f"posture rejection structural hash differs for {bucket_id}"
            )
        pairs_raw = raw.get("excluded_surface_port_id_pairs")
        if not isinstance(pairs_raw, list) or not pairs_raw:
            raise SchemaValidationError(
                f"posture rejection pairs are missing for {bucket_id}"
            )
        pairs: list[list[int]] = []
        identities: list[frozenset[int]] = []
        for pair_raw in pairs_raw:
            if (
                not isinstance(pair_raw, list)
                or len(pair_raw) != 2
                or len({int(value) for value in pair_raw}) != 2
                or min(int(value) for value in pair_raw) < 0
            ):
                raise SchemaValidationError(
                    f"posture rejection pair is invalid for {bucket_id}"
                )
            pair = [int(value) for value in pair_raw]
            pairs.append(pair)
            identities.append(frozenset(pair))
        if len(identities) != len(set(identities)):
            raise SchemaValidationError(
                f"posture rejection pairs repeat for {bucket_id}"
            )
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason:
            raise SchemaValidationError(
                f"posture rejection reason is missing for {bucket_id}"
            )
        output[bucket_id] = {
            "structural_hash": structural_hash,
            "excluded_surface_port_id_pairs": pairs,
            "reason": reason,
            "review_authority": "human_visual_review",
        }
    return output


def _parse_bucket_indices(
    raw: str,
    *,
    bucket_count: int,
) -> list[int]:
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--bucket-indices must be a comma-separated list")
    try:
        values = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError(
            "--bucket-indices must contain only integers"
        ) from error
    if len(values) != len(set(values)):
        raise ValueError("--bucket-indices must not contain duplicates")
    invalid = [
        value
        for value in values
        if value < 0 or value >= bucket_count
    ]
    if invalid:
        raise ValueError(
            "--bucket-indices contains out-of-range values: "
            f"{invalid} for bucket_count={bucket_count}"
        )
    return values


def _parse_surface_port_pair(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(
            "--preferred-surface-port-ids must contain exactly A,B"
        )
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            "--preferred-surface-port-ids must contain integer ids"
        ) from error
    if len(set(values)) != 2 or min(values) < 0:
        raise ValueError(
            "--preferred-surface-port-ids must contain two distinct "
            "non-negative ids"
        )
    return values


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _portable(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
