#!/usr/bin/env python3
from __future__ import annotations

"""Independently recheck saved C3 nominal knots against convex obstacles."""

import argparse
import json
import math
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.feasibility.order9_posture_collision import (
    CollisionAwareIKConfig,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import InteractionKnot
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_posture_resolver import (
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
)
from amsrr.utils.hashing import hash_file


VALIDATOR_VERSION = "order9_c3_saved_nominal_convex_collision_recheck_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal_manifest")
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--margin-m",
        type=float,
        help=(
            "Requested non-contact margin. Defaults to the runtime proxy "
            "margin."
        ),
    )
    parser.add_argument(
        "--numerical-tolerance-m",
        type=float,
        default=5.0e-4,
        help="Tolerance applied to proxy and selected-contact boundaries.",
    )
    parser.add_argument(
        "--bucket-indices",
        help="Optional comma-separated source entry indices for diagnostics.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    nominal_path = _resolve(args.nominal_manifest)
    if nominal_path.is_dir():
        nominal_path = nominal_path / "manifest.json"
    nominal = Order9C3NominalTrajectorySetManifest.from_json(
        nominal_path.read_text(encoding="utf-8")
    )
    nominal.validate()
    bucket_manifest_path = _resolve(nominal.bucket_manifest_path)
    buckets = load_order9_pi_l_rollout_bucket_manifest(bucket_manifest_path)
    bucket_by_id = {value.bucket_id: value for value in buckets.buckets}
    curriculum = load_order9_learning_config(
        _resolve(args.curriculum_config)
    )
    physical_model = build_physical_model_from_config(
        _resolve(curriculum.production_runtime.robot_model_config_path)
    )
    requested_margin_m = (
        float(args.margin_m)
        if args.margin_m is not None
        else float(CollisionAwareIKConfig().collision_margin_m)
    )
    if (
        not math.isfinite(requested_margin_m)
        or requested_margin_m < 0.0
    ):
        raise ValueError("--margin-m must be finite and non-negative")
    numerical_tolerance_m = float(args.numerical_tolerance_m)
    if (
        not math.isfinite(numerical_tolerance_m)
        or numerical_tolerance_m < 0.0
    ):
        raise ValueError(
            "--numerical-tolerance-m must be finite and non-negative"
        )
    started = time.perf_counter()
    records = []
    selected_entries = list(nominal.entries)
    if args.bucket_indices:
        indices = [
            int(value.strip())
            for value in args.bucket_indices.split(",")
            if value.strip()
        ]
        if (
            not indices
            or len(indices) != len(set(indices))
            or any(index < 0 or index >= len(selected_entries) for index in indices)
        ):
            raise ValueError("--bucket-indices are invalid")
        selected_entries = [selected_entries[index] for index in indices]
    for entry in selected_entries:
        bucket = bucket_by_id[entry.bucket_id]
        task = TaskSpec.from_json(
            (
                bucket_manifest_path.parent / bucket.task_spec_path
            ).read_text(encoding="utf-8")
        )
        artifact_path = nominal_path.parent / entry.artifact_path
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            artifact_path
        )
        morphology = MorphologyGraph.from_json(
            (
                artifact_path.parent
                / artifact.task_conditioned_morphology_path
            ).read_text(encoding="utf-8")
        )
        timeline = json.loads(
            (artifact_path.parent / artifact.timeline_path).read_text(
                encoding="utf-8"
            )
        )["records"]
        collision_object = build_order9_c3_posture_collision_object(task)
        resolver = Order9PostureTrajectoryResolver(
            physical_model,
            collision_object=collision_object,
            require_native_solver=True,
        )
        solver = resolver.ik_solver
        object_pose = tuple(collision_object.initial_pose_world)
        minimum_object_clearance = math.inf
        minimum_support_clearance = math.inf
        minimum_ground_clearance = math.inf
        maximum_violating_pairs = 0
        maximum_ground_violations = 0
        failure_count = 0
        failure_examples = []
        tolerated_selected_contact_count = 0
        for frame_index, record in enumerate(timeline):
            knot = InteractionKnot.from_dict(record["knot"])
            for target in knot.object_targets:
                if (
                    target.object_id == collision_object.object_id
                    and target.pose_target_world is not None
                ):
                    object_pose = tuple(target.pose_target_world)
            posture = knot.posture_target
            centroidal = knot.centroidal_target
            if (
                posture is None
                or posture.joint_pos_target is None
                or centroidal is None
                or centroidal.com_pos_world is None
                or centroidal.body_orientation_world is None
            ):
                raise RuntimeError(
                    f"{entry.bucket_id} frame {frame_index} lacks posture"
                )
            centroidal_pose = (
                *centroidal.com_pos_world,
                *centroidal.body_orientation_world,
            )
            allowed = tuple(
                sorted(
                    int(value.anchor_id)
                    for value in knot.contact_assignments
                    if value.schedule_state
                    in {"attach", "maintain", "slide", "release"}
                )
            )
            scenes = [
                (
                    "object",
                    object_pose,
                    collision_object.size_m,
                    allowed,
                ),
                *[
                    ("support", box.pose_world, box.size_m, ())
                    for box in collision_object.environment_boxes
                ],
            ]
            for kind, pose, size, allowed_anchor_ids in scenes:
                solver.set_collision_scene(
                    morphology=morphology,
                    object_pose_world=pose,
                    object_size_m=size,
                    allowed_anchor_ids=allowed_anchor_ids,
                )
                result = solver.check_configuration(
                    morphology=morphology,
                    centroidal_pose_world=centroidal_pose,
                    joint_positions_rad=posture.joint_pos_target,
                    exact=False,
                    margin_m=requested_margin_m,
                    ground_plane_z_m=collision_object.ground_plane_z_m,
                )
                accepted = result.get("accepted") is True
                selected_limit = result.get(
                    "selected_contact_penetration_limit_m"
                )
                selected_penetration = result.get(
                    "maximum_selected_contact_penetration_m"
                )
                tolerated_selected = (
                    not accepted
                    and int(result.get("ground_violating_proxy_count", 0)) == 0
                    and int(result.get("colliding_pair_count", 0)) == 0
                    and selected_limit is not None
                    and selected_penetration is not None
                    and float(selected_penetration)
                    <= float(selected_limit) + numerical_tolerance_m
                    and all(
                        value.get("colliding") is not True
                        and float(value.get("clearance_m", -math.inf))
                        >= requested_margin_m - numerical_tolerance_m
                        for value in result.get("worst_pairs", [])
                    )
                )
                if tolerated_selected:
                    accepted = True
                    tolerated_selected_contact_count += 1
                if not accepted:
                    failure_count += 1
                    if len(failure_examples) < 16:
                        failure_examples.append(
                            {
                                "frame_index": frame_index,
                                "phase": record.get("phase"),
                                "scene_kind": kind,
                                "result": result,
                            }
                        )
                clearance = float(result["minimum_clearance_m"])
                if kind == "object":
                    minimum_object_clearance = min(
                        minimum_object_clearance, clearance
                    )
                else:
                    minimum_support_clearance = min(
                        minimum_support_clearance, clearance
                    )
                ground_clearance = result.get(
                    "minimum_ground_clearance_m"
                )
                if ground_clearance is not None:
                    minimum_ground_clearance = min(
                        minimum_ground_clearance,
                        float(ground_clearance),
                    )
                maximum_violating_pairs = max(
                    maximum_violating_pairs,
                    int(result["violating_pair_count"]),
                )
                maximum_ground_violations = max(
                    maximum_ground_violations,
                    int(result["ground_violating_proxy_count"]),
                )
        records.append(
            {
                "bucket_id": entry.bucket_id,
                "module_count": entry.module_count,
                "frame_count": len(timeline),
                "accepted": failure_count == 0,
                "minimum_object_self_clearance_m": (
                    minimum_object_clearance
                ),
                "minimum_support_self_clearance_m": (
                    minimum_support_clearance
                ),
                "minimum_ground_clearance_m": minimum_ground_clearance,
                "maximum_violating_pair_count": maximum_violating_pairs,
                "maximum_ground_violating_proxy_count": (
                    maximum_ground_violations
                ),
                "failure_count": failure_count,
                "failure_examples": failure_examples,
                "tolerated_selected_contact_count": (
                    tolerated_selected_contact_count
                ),
            }
        )
    payload = {
        "validator_version": f"{VALIDATOR_VERSION}:convex_proxy",
        "collision_geometry_mode": "convex_proxy",
        "requested_collision_margin_m": requested_margin_m,
        "numerical_tolerance_m": numerical_tolerance_m,
        "nominal_manifest_path": _portable(nominal_path),
        "nominal_manifest_sha256": hash_file(nominal_path),
        "bucket_count": len(records),
        "all_accepted": all(value["accepted"] for value in records),
        "records": records,
        "wall_time_s": time.perf_counter() - started,
    }
    output = (
        _resolve(args.output)
        if args.output
        else nominal_path.parent / "collision_validation.json"
    )
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ORDER9_C3_COLLISION_RECHECK="
        f"path={output} sha256={hash_file(output)} "
        f"accepted={payload['all_accepted']} "
        f"wall_s={payload['wall_time_s']:.3f}"
    )
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
