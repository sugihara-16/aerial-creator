#!/usr/bin/env python3
from __future__ import annotations

"""Visualize C3 approach goals rejected after adding the support plane."""

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.articulated_reachability import (
    ArticulatedContactIKSolver,
    resolve_mesh_backed_anchor_references,
)
from amsrr.geometry.pose_math import compose_pose
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import (
    GraspCarryBaselinePlanner,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
)
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherConfig,
    _candidate_group_attempts,
    _centroidal_pose_from_fk,
    _knot_template_for_state,
    _pregrasp_candidate_mapping,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_neutral_runtime_observation,
    build_order9_c3_posture_collision_object,
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
from amsrr.training.order9_teacher import (
    upgrade_teacher_trajectory_to_v2,
)
from amsrr.utils.hashing import hash_file, stable_hash
from amsrr.visualization.order9_c3_curation import (
    Order9C3ViewerBox,
    Order9C3ViewerMarker,
    build_order9_c3_urdf_mesh_scene,
    order9_c3_urdf_root_pose_from_baselink_pose,
    render_order9_c3_mesh_viewer,
    write_order9_c3_stl_mesh_library,
    write_order9_c3_viewer_javascript,
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
    "rejected_pregrasp_support_diagnostics_v1"
)
DEFAULT_INDICES = (4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26, 27)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-manifest", default=DEFAULT_BUCKET_MANIFEST)
    parser.add_argument("--asset-manifest", default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bucket-indices",
        default=",".join(str(value) for value in DEFAULT_INDICES),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    destination = _resolve(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    bucket_manifest_path = _resolve(args.bucket_manifest)
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(
        bucket_manifest_path
    )
    assets = Order9MorphologyAssetManifest.from_json(
        _resolve(args.asset_manifest).read_text(encoding="utf-8")
    )
    curriculum = load_order9_learning_config(
        _resolve(args.curriculum_config)
    )
    physical_model = build_physical_model_from_config(
        _resolve(curriculum.production_runtime.robot_model_config_path)
    )
    indices = _indices(args.bucket_indices, len(bucket_manifest.buckets))
    destination.mkdir(parents=True)

    prepared = []
    mesh_paths: set[Path] = set()
    for index in indices:
        bucket = bucket_manifest.buckets[index]
        task = TaskSpec.from_json(
            (
                bucket_manifest_path.parent / bucket.task_spec_path
            ).read_text(encoding="utf-8")
        )
        graph = MorphologyGraph.from_json(
            (
                bucket_manifest_path.parent / bucket.morphology_graph_path
            ).read_text(encoding="utf-8")
        )
        urdf_path = _resolve(assets.entry_for(graph).urdf_path)
        diagnostic = _reconstruct_rejected_pregrasp(
            task=task,
            graph=graph,
            physical_model=physical_model,
        )
        scene = build_order9_c3_urdf_mesh_scene(
            urdf_path,
            joint_positions_rad=diagnostic["joint_positions_rad"],
            root_pose_world=order9_c3_urdf_root_pose_from_baselink_pose(
                urdf_path,
                diagnostic["base_pose_world"],
            ),
        )
        mesh_paths.update(scene.mesh_paths)
        prepared.append(
            (index, bucket, task, graph, urdf_path, diagnostic, scene)
        )

    shared = destination / "shared"
    mesh_library = write_order9_c3_stl_mesh_library(
        mesh_paths,
        shared / "mesh_library.js",
    )
    viewer_js = write_order9_c3_viewer_javascript(
        shared / "mesh_viewer.js"
    )
    rows = []
    for (
        index,
        bucket,
        task,
        graph,
        urdf_path,
        diagnostic,
        scene,
    ) in prepared:
        case_dir = destination / "buckets" / bucket.bucket_id
        markers = _markers(
            diagnostic=diagnostic,
            graph=diagnostic["morphology"],
            physical_model=physical_model,
            scene=scene,
        )
        boxes, box_metadata = _viewer_boxes(task)
        viewer = render_order9_c3_mesh_viewer(
            urdf_scene=scene,
            html_path=case_dir / "rejected_pregrasp.html",
            mesh_library_path=mesh_library,
            viewer_javascript_path=viewer_js,
            title=f"{bucket.bucket_id} — rejected approach pregrasp",
            subtitle=(
                "This is the new teacher's rejected 8 cm pregrasp goal, "
                "not the user-accepted final grasp pose."
            ),
            markers=markers,
            boxes=boxes,
            metadata={
                "diagnostic_version": (
                    "order9_c3_rejected_pregrasp_support_diagnostic_v1"
                ),
                "source_bucket_index": index,
                "source_bucket_id": bucket.bucket_id,
                "module_count": bucket.module_count,
                "structural_hash": bucket.structural_hash,
                "selected_surface_port_ids": list(
                    diagnostic["surface_port_ids"]
                ),
                "candidate_group_id": diagnostic["candidate_group_id"],
                "goal_kind": "approach_pregrasp_8cm_outward",
                "not_final_grasp": True,
                "refinement_status": diagnostic["refinement_status"],
                "collision_checks": diagnostic["collision_checks"],
                "full_virtual_support": box_metadata,
                "joint_solution_hash": stable_hash(
                    diagnostic["joint_positions_rad"]
                ),
                "source_urdf_sha256": hash_file(urdf_path),
            },
            default_view="iso",
            semantic_scope=(
                "diagnostic visualization of the approach pregrasp goal "
                "rejected by convex object/self/support checks; actual URDF "
                "visual meshes are shown, while collision decisions use "
                "convex proxy geometry"
            ),
        )
        rows.append(
            {
                "index": index,
                "bucket_id": bucket.bucket_id,
                "module_count": bucket.module_count,
                "surface_ids": diagnostic["surface_port_ids"],
                "refinement": diagnostic["refinement_status"],
                "object": _summary(diagnostic["collision_checks"]["object"]),
                "support": _summary(
                    diagnostic["collision_checks"]["virtual_support"]
                ),
                "viewer": viewer.html_path.relative_to(destination),
            }
        )
    _write_index(destination / "index.html", rows)
    manifest = {
        "diagnostic_version": (
            "order9_c3_rejected_pregrasp_support_diagnostic_v1"
        ),
        "bucket_manifest_path": str(bucket_manifest_path),
        "bucket_manifest_sha256": hash_file(bucket_manifest_path),
        "bucket_indices": indices,
        "case_count": len(rows),
        "rows": [
            {
                **row,
                "viewer": str(row["viewer"]),
            }
            for row in rows
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ORDER9_C3_REJECTED_PREGRASP_DIAGNOSTICS="
        f"{destination / 'index.html'}"
    )
    return 0


def _reconstruct_rejected_pregrasp(
    *,
    task: TaskSpec,
    graph: MorphologyGraph,
    physical_model,
) -> dict[str, Any]:
    evidence = task.metadata["order9_c3_articulated_teacher_precheck"]
    surface_ids = tuple(
        int(value) for value in evidence["selected_surface_port_ids"]
    )
    group_id = str(evidence["candidate_group_id"])
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    design_context = DesignPolicyContext(
        task,
        built.irg,
        physical_model,
        envelope,
    )
    _trace, design, _feasibility = (
        build_order9_task_conditioned_design_teacher(
            design_context,
            graph,
            preferred_anchor_surface_ids=surface_ids,
        )
    )
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    attempts = _candidate_group_attempts(
        candidates,
        maximum=8,
        preferred_group_id=group_id,
    )
    if len(attempts) != 1:
        raise SchemaValidationError(
            "diagnostic did not resolve exactly one prechecked group"
        )
    _resolved_group_id, selected_candidates = attempts[0]
    observation = build_order9_c3_neutral_runtime_observation(
        design.target_morphology,
        physical_model,
        task,
        phase_label="approach",
    )
    context = HighLevelPolicyContext(
        built.irg,
        envelope,
        design.target_morphology,
        selected_candidates,
        runtime_observation=observation,
    )
    baseline = upgrade_teacher_trajectory_to_v2(
        GraspCarryBaselinePlanner().plan(context),
        context,
    )
    active_knot = next(
        knot
        for knot in baseline.knots
        if any(
            assignment.schedule_state == "maintain"
            for assignment in knot.contact_assignments
        )
    )
    contact_solver = ArticulatedContactIKSolver(physical_model)
    zero_q = {
        joint_id: 0.0
        for joint_id in ordered_global_dock_joint_ids(
            design.target_morphology,
            physical_model,
        )
    }
    final_contact = contact_solver.solve(
        morphology=design.target_morphology,
        assignments=active_knot.contact_assignments,
        candidates={
            candidate.candidate_id: candidate
            for candidate in selected_candidates.candidates
        },
        initial_joint_positions_rad=zero_q,
    )
    if not final_contact.feasible:
        raise SchemaValidationError(
            "diagnostic could not reconstruct the prechecked contact IK"
        )
    attach_assignments = tuple(
        _knot_template_for_state(baseline, "attach").contact_assignments
    )
    pregrasp_targets = _pregrasp_candidate_mapping(
        attach_assignments,
        selected_candidates,
        clearance_m=Order9ArticulatedTeacherConfig().pregrasp_clearance_m,
    )
    pregrasp = ArticulatedContactIKSolver(
        physical_model,
        kinematics=WholeStructureKinematics(),
    ).solve(
        morphology=design.target_morphology,
        assignments=attach_assignments,
        candidates=pregrasp_targets,
        initial_joint_positions_rad=final_contact.joint_positions_rad,
        initial_base_pose_world=final_contact.base_pose_world,
    )
    if not pregrasp.feasible:
        raise SchemaValidationError(
            "diagnostic could not reconstruct the rejected pregrasp IK"
        )

    selected_anchor_ids = tuple(sorted(pregrasp.anchor_poses_world))
    references = resolve_mesh_backed_anchor_references(
        design.target_morphology,
        physical_model,
        selected_anchor_ids,
    )
    kinematics = WholeStructureKinematics()
    desired_fk = kinematics.forward(
        design.target_morphology,
        physical_model,
        pregrasp.joint_positions_rad,
        pregrasp.base_pose_world,
        references,
    )
    desired_centroidal = _centroidal_pose_from_fk(
        context=context,
        physical_model=physical_model,
        q=pregrasp.joint_positions_rad,
        module_root_poses_world=desired_fk.module_root_poses_world,
        source_observation=observation,
        builder=RigidBodyControlModelBuilder(),
    )
    collision_object = build_order9_c3_posture_collision_object(task)
    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        collision_object=collision_object,
        require_native_solver=True,
    )
    solver = resolver.ik_solver
    object_pose = collision_object.initial_pose_world
    assert object_pose is not None
    solver.set_collision_scene(
        morphology=design.target_morphology,
        object_pose_world=object_pose,
        object_size_m=collision_object.size_m,
        allowed_anchor_ids=(),
    )
    refined = solver.solve(
        morphology=design.target_morphology,
        centroidal_pose_world=desired_centroidal,
        anchor_pose_targets_world={
            reference.anchor.anchor_id: (
                desired_fk.anchor_poses_world[
                    reference.anchor.anchor_id
                ]
            )
            for reference in references
        },
        initial_joint_positions_rad=pregrasp.joint_positions_rad,
    )
    if refined.feasible:
        q = dict(refined.joint_positions_rad)
        base_pose = tuple(refined.base_pose_world)
        centroidal = tuple(refined.centroidal_pose_world)
        refinement_status = "feasible_then_collision_checked"
    else:
        q = dict(pregrasp.joint_positions_rad)
        base_pose = tuple(pregrasp.base_pose_world)
        centroidal = tuple(desired_centroidal)
        refinement_status = (
            "collision_aware_refinement_infeasible; "
            "displaying_requested_pregrasp"
        )
    collision_checks = {
        "object": _check_scene(
            solver,
            morphology=design.target_morphology,
            pose=object_pose,
            size=collision_object.size_m,
            centroidal=centroidal,
            q=q,
        )
    }
    support = collision_object.environment_boxes[0]
    collision_checks["virtual_support"] = _check_scene(
        solver,
        morphology=design.target_morphology,
        pose=support.pose_world,
        size=support.size_m,
        centroidal=centroidal,
        q=q,
    )
    return {
        "morphology": design.target_morphology,
        "surface_port_ids": surface_ids,
        "candidate_group_id": group_id,
        "joint_positions_rad": q,
        "base_pose_world": base_pose,
        "centroidal_pose_world": centroidal,
        "refinement_status": refinement_status,
        "collision_checks": collision_checks,
        "pregrasp_targets": pregrasp_targets,
        "references": references,
    }


def _check_scene(
    solver,
    *,
    morphology,
    pose,
    size,
    centroidal,
    q,
) -> dict[str, Any]:
    solver.set_collision_scene(
        morphology=morphology,
        object_pose_world=pose,
        object_size_m=size,
        allowed_anchor_ids=(),
    )
    return dict(
        solver.check_configuration(
            morphology=morphology,
            centroidal_pose_world=centroidal,
            joint_positions_rad=q,
            exact=False,
            margin_m=float(solver.collision_config.collision_margin_m),
        )
    )


def _markers(
    *,
    diagnostic: dict[str, Any],
    graph,
    physical_model,
    scene,
) -> tuple[Order9C3ViewerMarker, ...]:
    markers = []
    target_by_anchor = {}
    for assignment in _active_assignments(
        diagnostic["pregrasp_targets"],
        diagnostic["candidate_group_id"],
    ):
        target_by_anchor[assignment[0]] = assignment[1]
    for reference in diagnostic["references"]:
        anchor_id = reference.anchor.anchor_id
        link_id = (
            f"module_{reference.surface.module_id}__"
            f"{reference.surface.mechanism_link_id}"
        )
        actual = compose_pose(
            scene.link_poses_world[link_id],
            reference.surface.grasp_contact_frame_link,
        )
        markers.append(
            Order9C3ViewerMarker(
                marker_id=f"actual-anchor-{anchor_id}",
                label=f"A{anchor_id} actual",
                position_world=tuple(actual[:3]),
                color_rgba=(1.0, 0.76, 0.04, 1.0),
                selected=True,
                kind="surface",
            )
        )
    for candidate in diagnostic["pregrasp_targets"].values():
        markers.append(
            Order9C3ViewerMarker(
                marker_id=f"pregrasp-target-{candidate.candidate_id}",
                label=f"C{candidate.candidate_id} pregrasp target",
                position_world=tuple(candidate.contact_pose_world[:3]),
                direction_world=tuple(candidate.normal_world),
                color_rgba=(0.05, 0.72, 0.28, 1.0),
                selected=True,
                kind="contact",
            )
        )
    return tuple(markers)


def _active_assignments(
    candidates,
    group_id: str,
) -> tuple[tuple[int, Any], ...]:
    # Marker construction only needs deterministic candidate identity; anchor
    # IDs are already carried by each candidate.
    del group_id
    return tuple(
        (candidate.anchor_id, candidate)
        for candidate in sorted(
            candidates.values(),
            key=lambda value: value.candidate_id,
        )
    )


def _viewer_boxes(
    task: TaskSpec,
) -> tuple[tuple[Order9C3ViewerBox, ...], dict[str, Any]]:
    geometries = {
        value.geometry_id: value for value in task.scene.geometry_library
    }
    movable = next(value for value in task.scene.objects if value.movable)
    object_geometry = geometries[movable.geometry_id]
    object_size = _size(object_geometry)
    support_surface = task.scene.environment.support_surfaces[0]
    support_geometry = geometries[support_surface.geometry_id]
    support_size = _size(support_geometry)
    support_top = (
        float(support_surface.pose_world[2]) + 0.5 * support_size[2]
    )
    # The collision volume is 20 m square.  A 4 m clipped display keeps the
    # robot visible while preserving the same top plane and thickness.
    clipped_ground_size = (4.0, 4.0, 0.20)
    clipped_ground_pose = (
        float(movable.pose_world[0]),
        float(movable.pose_world[1]),
        support_top - 0.10,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return (
        (
            Order9C3ViewerBox(
                object_id=movable.object_id,
                pose_world=tuple(movable.pose_world),
                size_m=object_size,
                color_rgba=(0.95, 0.66, 0.12, 0.45),
            ),
            Order9C3ViewerBox(
                object_id=f"{support_surface.surface_id}:physical_box",
                pose_world=tuple(support_surface.pose_world),
                size_m=support_size,
                color_rgba=(0.34, 0.34, 0.38, 0.34),
            ),
            Order9C3ViewerBox(
                object_id="virtual_ground_clipped_display",
                pose_world=clipped_ground_pose,
                size_m=clipped_ground_size,
                color_rgba=(0.20, 0.48, 0.80, 0.14),
            ),
        ),
        {
            "collision_size_m": [20.0, 20.0, 0.20],
            "display_size_m": list(clipped_ground_size),
            "top_z_m": support_top,
            "display_is_xy_clipped": True,
        },
    )


def _size(geometry) -> tuple[float, float, float]:
    raw = geometry.primitive_params["size_m"]
    return tuple(
        float(raw[index]) * float(geometry.scale[index])
        for index in range(3)
    )


def _summary(result: dict[str, Any]) -> str:
    return (
        f"{'accept' if result.get('accepted') else 'reject'}; "
        f"pairs={result.get('violating_pair_count')}; "
        f"clearance={float(result.get('minimum_clearance_m', 0.0)):.4f}m"
    )


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "\n".join(
        "<tr>"
        f"<td>{row['index']}</td>"
        f"<td>{html.escape(row['bucket_id'])}</td>"
        f"<td>{row['module_count']}</td>"
        f"<td>{html.escape(str(row['surface_ids']))}</td>"
        f"<td>{html.escape(row['refinement'])}</td>"
        f"<td>{html.escape(row['object'])}</td>"
        f"<td>{html.escape(row['support'])}</td>"
        f"<td><a href=\"{html.escape(str(row['viewer']))}\">3D表示</a></td>"
        "</tr>"
        for row in rows
    )
    path.write_text(
        f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>C3 rejected pregrasp support diagnostics</title>
<style>body{{font-family:system-ui;margin:24px;background:#f5f7fa}}
table{{border-collapse:collapse;background:white;width:100%}}
th,td{{border:1px solid #ccd4dd;padding:7px;text-align:left}}
th{{background:#e9eef4}}</style></head><body>
<h1>C3 rejected approach-pregrasp diagnostics</h1>
<p>表示対象は新teacherが棄却した8 cm外側のpregrasp目標です。
ユーザー確認済みの最終把持姿勢ではありません。青は20 m仮想支持面を
4 mに切って表示、灰色はTaskSpecの2 m支持台、橙は把持物体です。</p>
<table><thead><tr><th>index</th><th>bucket</th><th>modules</th>
<th>surfaces</th><th>goal refinement</th><th>object/self</th>
<th>support/self</th><th>viewer</th></tr></thead>
<tbody>{body}</tbody></table></body></html>""",
        encoding="utf-8",
    )


def _indices(raw: str, count: int) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",")]
    if len(values) != len(set(values)) or any(
        value < 0 or value >= count for value in values
    ):
        raise ValueError("bucket indices are duplicate or out of range")
    return values


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (REPOSITORY_ROOT / path).resolve()
    )


if __name__ == "__main__":
    raise SystemExit(main())
