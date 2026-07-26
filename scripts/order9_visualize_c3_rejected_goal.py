#!/usr/bin/env python3
from __future__ import annotations

"""Reproduce and visualize one rejected C3 configuration-space goal."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.articulated_reachability import (
    resolve_mesh_backed_anchor_references,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
)
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherConfig,
    Order9ArticulatedTrajectoryTeacher,
    _centroidal_pose_from_fk,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_neutral_runtime_observation,
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_configuration_space_planner import (
    Order9ConfigurationState,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_design_teacher_dataset import (
    build_order9_task_conditioned_design_teacher,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
)
from amsrr.utils.hashing import hash_file
from amsrr.visualization.order9_c3_curation import (
    Order9C3UrdfMeshScene,
    Order9C3ViewerBox,
    Order9C3ViewerMarker,
    build_order9_c3_urdf_mesh_scene,
    order9_c3_urdf_root_pose_from_baselink_pose,
    render_order9_c3_mesh_viewer,
    transformed_link_axis,
    write_order9_c3_stl_mesh_library,
    write_order9_c3_viewer_javascript,
)
from amsrr.geometry.pose_math import compose_pose


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
    "artifacts/p4_full/order9/diagnostics/"
    "rejected_goal_support_visualization"
)


class _GoalCaptured(RuntimeError):
    pass


class _CapturingConfigurationPlanner:
    planner_version = "order9_diagnostic_goal_capture_v1"

    def __init__(self) -> None:
        self.start: Order9ConfigurationState | None = None
        self.goal: Order9ConfigurationState | None = None
        self.aggregate_goal_collision_free: bool | None = None

    def plan(self, **kwargs):
        self.start = kwargs["start"]
        self.goal = kwargs["goal"]
        self.aggregate_goal_collision_free = bool(
            kwargs["is_collision_free"](self.goal)
        )
        raise _GoalCaptured("configuration-space goal captured")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-manifest", default=DEFAULT_BUCKET_MANIFEST)
    parser.add_argument("--asset-manifest", default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--bucket-index", type=int, default=6)
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    bucket_manifest_path = _resolve(args.bucket_manifest)
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(
        bucket_manifest_path
    )
    if args.bucket_index < 0 or args.bucket_index >= len(
        bucket_manifest.buckets
    ):
        raise ValueError("--bucket-index is outside the source manifest")
    bucket = bucket_manifest.buckets[args.bucket_index]
    bucket_root = bucket_manifest_path.parent
    task = TaskSpec.from_json(
        (bucket_root / bucket.task_spec_path).read_text(encoding="utf-8")
    )
    source_graph = MorphologyGraph.from_json(
        (bucket_root / bucket.morphology_graph_path).read_text(
            encoding="utf-8"
        )
    )
    curriculum = load_order9_learning_config(
        _resolve(args.curriculum_config)
    )
    physical_model = build_physical_model_from_config(
        _resolve(curriculum.production_runtime.robot_model_config_path)
    )
    archived = task.metadata["order9_c3_articulated_teacher_precheck"]
    surface_ids = tuple(
        int(value) for value in archived["selected_surface_port_ids"]
    )
    candidate_group_id = str(archived["candidate_group_id"])

    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    _trace, design, _feasibility = (
        build_order9_task_conditioned_design_teacher(
            DesignPolicyContext(
                task,
                built.irg,
                physical_model,
                envelope,
            ),
            source_graph,
            preferred_anchor_surface_ids=surface_ids,
        )
    )
    morphology = design.target_morphology
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    observation = build_order9_c3_neutral_runtime_observation(
        morphology,
        physical_model,
        task,
        phase_label="approach",
    )
    context = HighLevelPolicyContext(
        built.irg,
        envelope,
        morphology,
        candidates,
        runtime_observation=observation,
    )
    collision_object = build_order9_c3_posture_collision_object(task)
    capture = _CapturingConfigurationPlanner()
    teacher = Order9ArticulatedTrajectoryTeacher(
        physical_model,
        config=Order9ArticulatedTeacherConfig(
            preferred_candidate_group_id=candidate_group_id
        ),
        collision_object=collision_object,
        configuration_space_planner=capture,  # type: ignore[arg-type]
    )
    try:
        teacher.plan(
            context,
            initial_object_poses_world={
                value.object_id: value.pose_world
                for value in task.scene.objects
            },
        )
    except _GoalCaptured:
        pass
    if capture.goal is None:
        raise SchemaValidationError(
            "diagnostic did not reach a configuration-space goal"
        )

    kinematics = WholeStructureKinematics()
    goal_fk = kinematics.forward(
        morphology,
        physical_model,
        capture.goal.joint_positions_rad,
        capture.goal.base_pose_world,
        (),
    )
    centroidal_pose = _centroidal_pose_from_fk(
        context=context,
        physical_model=physical_model,
        q=capture.goal.joint_positions_rad,
        module_root_poses_world=goal_fk.module_root_poses_world,
        source_observation=observation,
        builder=RigidBodyControlModelBuilder(),
    )
    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        collision_object=collision_object,
        require_native_solver=True,
    )
    solver = resolver.ik_solver
    margin_m = float(solver.collision_config.collision_margin_m)
    checks = {
        "self_only": _check_scene(
            solver,
            morphology=morphology,
            centroidal_pose=centroidal_pose,
            q=capture.goal.joint_positions_rad,
            pose=None,
            size=None,
            margin_m=margin_m,
        ),
        "object": _check_scene(
            solver,
            morphology=morphology,
            centroidal_pose=centroidal_pose,
            q=capture.goal.joint_positions_rad,
            pose=collision_object.initial_pose_world,
            size=collision_object.size_m,
            margin_m=margin_m,
        ),
        "virtual_support": _check_scene(
            solver,
            morphology=morphology,
            centroidal_pose=centroidal_pose,
            q=capture.goal.joint_positions_rad,
            pose=collision_object.environment_boxes[0].pose_world,
            size=collision_object.environment_boxes[0].size_m,
            margin_m=margin_m,
        ),
    }
    actual_support = _support_box(task)
    checks["task_support"] = _check_scene(
        solver,
        morphology=morphology,
        centroidal_pose=centroidal_pose,
        q=capture.goal.joint_positions_rad,
        pose=actual_support.pose_world,
        size=actual_support.size_m,
        margin_m=margin_m,
    )

    asset_manifest = Order9MorphologyAssetManifest.from_json(
        _resolve(args.asset_manifest).read_text(encoding="utf-8")
    )
    urdf_path = _resolve(asset_manifest.entry_for(source_graph).urdf_path)
    root_pose = order9_c3_urdf_root_pose_from_baselink_pose(
        urdf_path,
        capture.goal.base_pose_world,
    )
    urdf_scene = build_order9_c3_urdf_mesh_scene(
        urdf_path,
        joint_positions_rad=capture.goal.joint_positions_rad,
        root_pose_world=root_pose,
    )
    highlighted_scene = _highlight_collision_links(urdf_scene, checks)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shared = output / "shared"
    mesh_library = write_order9_c3_stl_mesh_library(
        highlighted_scene.mesh_paths,
        shared / "mesh_library.js",
    )
    viewer_js = write_order9_c3_viewer_javascript(
        shared / "mesh_viewer.js"
    )
    object_box = Order9C3ViewerBox(
        object_id=collision_object.object_id,
        pose_world=collision_object.initial_pose_world,
        size_m=collision_object.size_m,
        color_rgba=(0.15, 0.68, 0.95, 0.32),
    )
    virtual_support = collision_object.environment_boxes[0]
    virtual_support_top_z = (
        float(virtual_support.pose_world[2])
        + 0.5 * float(virtual_support.size_m[2])
    )
    boxes = (
        object_box,
        actual_support,
        Order9C3ViewerBox(
            object_id="virtual_support_top_plane_visual_clip",
            pose_world=(
                float(virtual_support.pose_world[0]),
                float(virtual_support.pose_world[1]),
                virtual_support_top_z - 0.005,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            # The collision volume is 20 m square.  Clip only its visual
            # representation so that the robot remains legible in the viewer.
            size_m=(4.0, 4.0, 0.01),
            color_rgba=(1.0, 0.58, 0.05, 0.18),
        ),
        Order9C3ViewerBox(
            object_id="ground_z0",
            pose_world=(0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0),
            size_m=(6.0, 6.0, 0.02),
            color_rgba=(0.22, 0.25, 0.28, 0.55),
        ),
    )
    markers = _collision_markers(highlighted_scene, checks)
    artifacts = render_order9_c3_mesh_viewer(
        urdf_scene=highlighted_scene,
        html_path=output / "rejected_goal.html",
        mesh_library_path=mesh_library,
        viewer_javascript_path=viewer_js,
        title=f"{bucket.bucket_id} — rejected approach/pregrasp goal",
        subtitle=(
            "Red/magenta links are reported inside the 5 mm convex margin. "
            "Blue: object, gray: TaskSpec support, amber: virtual support "
            "collision volume, dark gray: z=0 ground."
        ),
        markers=markers,
        boxes=boxes,
        metadata={
            "diagnostic_version": "order9_rejected_goal_support_view_v1",
            "bucket_id": bucket.bucket_id,
            "source_bucket_index": args.bucket_index,
            "source_bucket_manifest": str(bucket_manifest_path),
            "source_bucket_manifest_sha256": hash_file(
                bucket_manifest_path
            ),
            "structural_hash": bucket.structural_hash,
            "selected_surface_port_ids": list(surface_ids),
            "candidate_group_id": candidate_group_id,
            "captured_phase": "approach",
            "captured_target_semantics": "pregrasp_configuration_goal",
            "aggregate_goal_collision_free": (
                capture.aggregate_goal_collision_free
            ),
            "centroidal_pose_world": list(centroidal_pose),
            "base_pose_world": list(capture.goal.base_pose_world),
            "task_support_top_z": (
                float(actual_support.pose_world[2])
                + 0.5 * float(actual_support.size_m[2])
            ),
            "virtual_support_top_z": virtual_support_top_z,
            "virtual_support_collision_size_m": list(
                virtual_support.size_m
            ),
            "joint_positions_rad": dict(
                sorted(capture.goal.joint_positions_rad.items())
            ),
            "collision_checks": checks,
        },
        default_view="iso",
        semantic_scope=(
            "diagnostic reproduction of the current convex gate's rejected "
            "approach/pregrasp goal; actual mesh is visual only"
        ),
    )
    report = {
        "bucket_id": bucket.bucket_id,
        "bucket_index": args.bucket_index,
        "html_path": str(artifacts.html_path),
        "scene_path": str(artifacts.scene_path),
        "scene_sha256": artifacts.scene_sha256,
        "selected_surface_port_ids": list(surface_ids),
        "candidate_group_id": candidate_group_id,
        "captured_target_semantics": "approach_pregrasp_goal",
        "aggregate_goal_collision_free": (
            capture.aggregate_goal_collision_free
        ),
        "base_pose_world": list(capture.goal.base_pose_world),
        "centroidal_pose_world": list(centroidal_pose),
        "task_support_top_z": (
            float(actual_support.pose_world[2])
            + 0.5 * float(actual_support.size_m[2])
        ),
        "virtual_support_top_z": virtual_support_top_z,
        "collision_checks": checks,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ORDER9_REJECTED_GOAL_VIEW="
        f"{artifacts.html_path} report={report_path} "
        f"scene_sha256={artifacts.scene_sha256}"
    )
    return 0


def _check_scene(
    solver,
    *,
    morphology: MorphologyGraph,
    centroidal_pose: Pose7D,
    q: dict[str, float],
    pose: Pose7D | None,
    size: tuple[float, float, float] | None,
    margin_m: float,
) -> dict[str, object]:
    solver.set_collision_scene(
        morphology=morphology,
        object_pose_world=pose,
        object_size_m=size,
        allowed_anchor_ids=(),
    )
    return dict(
        solver.check_configuration(
            morphology=morphology,
            centroidal_pose_world=centroidal_pose,
            joint_positions_rad=q,
            exact=False,
            margin_m=margin_m,
        )
    )


def _support_box(task: TaskSpec) -> Order9C3ViewerBox:
    support = task.scene.environment.support_surfaces[0]
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == support.geometry_id
    )
    size = tuple(
        float(geometry.primitive_params["size_m"][index])
        * float(geometry.scale[index])
        for index in range(3)
    )
    return Order9C3ViewerBox(
        object_id=support.surface_id,
        pose_world=tuple(support.pose_world),
        size_m=size,  # type: ignore[arg-type]
        color_rgba=(0.42, 0.46, 0.52, 0.45),
    )


def _highlight_collision_links(
    scene: Order9C3UrdfMeshScene,
    checks: dict[str, dict[str, object]],
) -> Order9C3UrdfMeshScene:
    link_kinds: dict[tuple[int, str], set[str]] = {}
    for kind, result in checks.items():
        effective_margin = float(result["effective_collision_margin_m"])
        for pair in result.get("worst_pairs", []):
            if float(pair["clearance_m"]) >= effective_margin:
                continue
            first = (
                int(pair["first_module_index"]),
                str(pair["first_link_id"]),
            )
            link_kinds.setdefault(first, set()).add(kind)
            if pair.get("second_kind") == "robot":
                second = (
                    int(pair["second_module_index"]),
                    str(pair["second_link_id"]),
                )
                link_kinds.setdefault(second, set()).add(kind)
    instances = []
    for instance in scene.instances:
        key = (
            int(instance["module_id"])
            if instance["module_id"] is not None
            else -1,
            str(instance["local_link_id"]),
        )
        updated = dict(instance)
        kinds = link_kinds.get(key, set())
        if kinds:
            updated["color_rgba"] = (
                [0.95, 0.05, 0.55, 0.96]
                if "self_only" in kinds
                else [1.0, 0.08, 0.03, 0.96]
            )
        instances.append(updated)
    return replace(scene, instances=tuple(instances))


def _collision_markers(
    scene: Order9C3UrdfMeshScene,
    checks: dict[str, dict[str, object]],
) -> tuple[Order9C3ViewerMarker, ...]:
    output = []
    seen = set()
    for kind, result in checks.items():
        effective_margin = float(result["effective_collision_margin_m"])
        for pair in result.get("worst_pairs", []):
            if float(pair["clearance_m"]) >= effective_margin:
                continue
            key = (
                kind,
                int(pair["first_module_index"]),
                str(pair["first_link_id"]),
            )
            if key in seen:
                continue
            seen.add(key)
            link_id = f"module_{key[1]}__{key[2]}"
            pose = scene.link_poses_world.get(link_id)
            if pose is None:
                continue
            output.append(
                Order9C3ViewerMarker(
                    marker_id=f"{kind}-{key[1]}-{key[2]}",
                    label=(
                        f"{kind}: M{key[1]} {key[2]} "
                        f"clearance={float(pair['clearance_m']):.4f}m"
                    ),
                    position_world=tuple(pose[:3]),
                    direction_world=transformed_link_axis(pose),
                    color_rgba=(1.0, 0.05, 0.05, 1.0),
                    selected=True,
                    kind="collision",
                )
            )
    return tuple(output)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


if __name__ == "__main__":
    raise SystemExit(main())
