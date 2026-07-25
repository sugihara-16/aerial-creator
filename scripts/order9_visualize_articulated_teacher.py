#!/usr/bin/env python3
from __future__ import annotations

"""Render an offline animation of an Order 9 articulated teacher trajectory."""

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.robot_model.physical_model_builder import (  # noqa: E402
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_c3_teacher import (  # noqa: E402
    build_order9_c3_articulated_teacher,
    order9_c3_teacher_evidence,
)
from amsrr.training.order9_curriculum import (  # noqa: E402
    load_order9_learning_config,
)
from amsrr.training.order9_rollout_buckets import (  # noqa: E402
    Order9PiLRolloutBucket,
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from amsrr.visualization.order9_articulated_teacher import (  # noqa: E402
    render_order9_c3_teacher_animation,
)


DEFAULT_BUCKET_MANIFEST = (
    "artifacts/p4_full/order9/stages/"
    "c3_pi_l_ppo_arbitrary_morphology/rollout_buckets/manifest.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-manifest",
        default=DEFAULT_BUCKET_MANIFEST,
        help="C3 rollout-bucket manifest or its containing directory.",
    )
    parser.add_argument(
        "--bucket-id",
        help="Exact bucket id. Overrides --split/--module-count/--bucket-index.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
    )
    parser.add_argument("--module-count", type=int, default=8)
    parser.add_argument(
        "--bucket-index",
        type=int,
        default=0,
        help="Zero-based index among buckets matching split/module count.",
    )
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument(
        "--robot-model-config",
        help=(
            "Optional PhysicalModel config override. By default the curriculum "
            "production-runtime path is used."
        ),
    )
    parser.add_argument(
        "--frames-per-second",
        type=int,
        default=20,
        help="Offline FK samples per teacher second, in [1, 60].",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output HTML. Default: artifacts/p4_full/order9/"
            "teacher_visualization/<bucket-id>/teacher_kinematics.html"
        ),
    )
    parser.add_argument(
        "--summary-output",
        help="Optional JSON summary path. Defaults beside the HTML.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = _resolve(args.bucket_manifest)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    validate_order9_pi_l_rollout_bucket_bytes(
        manifest_path,
        repository_root=REPOSITORY_ROOT,
    )
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    bucket = _select_bucket(
        manifest.buckets,
        bucket_id=args.bucket_id,
        split=args.split,
        module_count=args.module_count,
        bucket_index=args.bucket_index,
    )
    task_path = manifest_path.parent / bucket.task_spec_path
    graph_path = manifest_path.parent / bucket.morphology_graph_path
    task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
    structural_graph = MorphologyGraph.from_json(
        graph_path.read_text(encoding="utf-8")
    )

    curriculum_path = _resolve(args.curriculum_config)
    curriculum = load_order9_learning_config(curriculum_path)
    robot_config = _resolve(
        args.robot_model_config
        or curriculum.production_runtime.robot_model_config_path
    )
    physical_model = build_physical_model_from_config(robot_config)
    if physical_model.stable_hash() != manifest.physical_model_hash:
        raise SchemaValidationError(
            "visualization PhysicalModel hash differs from the bucket manifest"
        )

    bundle = build_order9_c3_articulated_teacher(
        task_spec=task,
        structural_target=structural_graph,
        physical_model=physical_model,
    )
    actual_evidence = order9_c3_teacher_evidence(bundle)
    expected_evidence = task.metadata.get(
        "order9_c3_articulated_teacher_precheck"
    )
    if not isinstance(expected_evidence, dict):
        raise SchemaValidationError(
            "selected bucket lacks articulated-teacher precheck evidence"
        )
    if actual_evidence != expected_evidence:
        raise SchemaValidationError(
            "recomputed teacher evidence differs from the selected bucket"
        )

    html_path = (
        _resolve(args.output)
        if args.output
        else (
            REPOSITORY_ROOT
            / "artifacts/p4_full/order9/teacher_visualization"
            / bucket.bucket_id
            / "teacher_kinematics.html"
        )
    )
    if html_path.suffix.lower() != ".html":
        raise ValueError("--output must end in .html")
    summary_path = (
        _resolve(args.summary_output) if args.summary_output else None
    )
    artifacts = render_order9_c3_teacher_animation(
        task_spec=task,
        bundle=bundle,
        physical_model=physical_model,
        html_path=html_path,
        summary_path=summary_path,
        frames_per_second=args.frames_per_second,
        source_metadata={
            "bucket_id": bucket.bucket_id,
            "bucket_manifest_path": str(manifest_path),
            "bucket_manifest_sha256": hash_file(manifest_path),
            "dataset_split": bucket.split.value,
            "seed": bucket.seed,
            "sample_index": bucket.sample_index,
            "task_spec_path": str(task_path),
            "task_spec_sha256": bucket.task_spec_sha256,
            "morphology_graph_path": str(graph_path),
            "morphology_graph_sha256": bucket.morphology_graph_sha256,
            "structural_hash": bucket.structural_hash,
            "physical_model_config_path": str(robot_config),
            "physical_model_hash": manifest.physical_model_hash,
            "teacher_evidence": actual_evidence,
        },
    )
    active_assignment = next(
        assignment
        for knot in bundle.trajectory.knots
        for assignment in knot.contact_assignments
        if assignment.schedule_state in {"attach", "maintain", "slide"}
    )
    selected_candidate = next(
        candidate
        for candidate in bundle.contact_candidate_set.candidates
        if candidate.candidate_id == active_assignment.candidate_id
    )
    object_spec = next(
        value
        for value in task.scene.objects
        if value.object_id == selected_candidate.target_entity_id
    )
    print(
        "ORDER9_TEACHER_ANIMATION="
        + json.dumps(
            {
                "bucket_id": bucket.bucket_id,
                "module_count": bucket.module_count,
                "object_id": object_spec.object_id,
                "object_mass_kg": object_spec.mass_kg,
                "selected_surface_port_ids": list(
                    bundle.selected_surface_port_ids
                ),
                "ik_iterations": bundle.trajectory_plan.ik_solution.iterations,
                "ik_maximum_position_error_m": (
                    bundle.trajectory_plan.ik_solution.maximum_position_error_m
                ),
                "ik_maximum_normal_error_rad": (
                    bundle.trajectory_plan.ik_solution.maximum_normal_error_rad
                ),
                "frame_count": artifacts.frame_count,
                "horizon_s": artifacts.horizon_s,
                "html_path": str(artifacts.html_path),
                "summary_path": str(artifacts.summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _select_bucket(
    buckets: list[Order9PiLRolloutBucket],
    *,
    bucket_id: str | None,
    split: str,
    module_count: int,
    bucket_index: int,
) -> Order9PiLRolloutBucket:
    if bucket_index < 0:
        raise ValueError("--bucket-index must be non-negative")
    if not 2 <= module_count <= 8:
        raise ValueError("--module-count must be in [2, 8]")
    if bucket_id is not None:
        matches = [
            bucket
            for bucket in buckets
            if getattr(bucket, "bucket_id") == bucket_id
        ]
        if len(matches) != 1:
            raise ValueError(f"bucket id {bucket_id!r} was not found uniquely")
        return matches[0]
    matches = [
        bucket
        for bucket in buckets
        if getattr(bucket, "split").value == split
        and int(getattr(bucket, "module_count")) == module_count
    ]
    if bucket_index >= len(matches):
        raise ValueError(
            f"only {len(matches)} buckets match split={split!r}, "
            f"module_count={module_count}"
        )
    return matches[bucket_index]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


if __name__ == "__main__":
    raise SystemExit(main())
