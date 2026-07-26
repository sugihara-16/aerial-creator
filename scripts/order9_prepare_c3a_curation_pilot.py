#!/usr/bin/env python3
from __future__ import annotations

"""Prepare the 12-case human mesh-curation pilot for Order 9 C3a."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.feasibility.articulated_reachability import (  # noqa: E402
    _global_joint_limits,
    resolve_mesh_backed_anchor_references,
)
from amsrr.feasibility.morphology_flight import (  # noqa: E402
    MorphologyFlightFeasibilityChecker,
    MorphologyFlightFeasibilityConfig,
)
from amsrr.feasibility.order9_posture_collision import (  # noqa: E402
    CollisionAwareIKConfig,
)
from amsrr.geometry.pose_math import compose_pose, transform_from_pose  # noqa: E402
from amsrr.morphology.random_connected import (  # noqa: E402
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.robot_model.fixed_morphology_urdf import (  # noqa: E402
    write_articulated_morphology_graph_urdf,
)
from amsrr.robot_model.gripper_surfaces import (  # noqa: E402
    resolve_unoccupied_gripper_surfaces,
)
from amsrr.robot_model.whole_structure_kinematics import (  # noqa: E402
    ordered_global_dock_joint_ids,
)
from amsrr.robot_model.physical_model_builder import (  # noqa: E402
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.order3 import Order3MorphologyPoolManifest  # noqa: E402
from amsrr.schemas.task_spec import GeometryType, TaskSpec  # noqa: E402
from amsrr.simulation.order9_morphology_assets import (  # noqa: E402
    load_order9_morphology_asset_manifest,
    validate_order9_morphology_asset_manifest_bytes,
)
from amsrr.training.order9_c3_teacher import (  # noqa: E402
    Order9C3TeacherConfig,
    build_order9_c3_articulated_teacher,
    order9_c3_teacher_evidence,
)
from amsrr.training.order9_posture_resolver import (  # noqa: E402
    Order9PostureCollisionObject,
    Order9PostureTrajectoryResolver,
)
from amsrr.utils.hashing import hash_file, stable_hash  # noqa: E402
from amsrr.visualization.order9_c3_curation import (  # noqa: E402
    ORDER9_C3_CURATION_VIEWER_VERSION,
    Order9C3ViewerBox,
    Order9C3ViewerMarker,
    build_order9_c3_urdf_mesh_scene,
    order9_c3_urdf_root_pose_from_baselink_pose,
    render_order9_c3_mesh_viewer,
    transformed_link_axis,
    write_order9_c3_stl_mesh_library,
    write_order9_c3_viewer_javascript,
)
from amsrr.visualization.order9_c3_review import (  # noqa: E402
    reset_order9_c3_review_after_recalculation,
)


DEFAULT_CONFIG = "configs/training/order9_c3a_curation_pilot.yaml"
DEFAULT_OUTPUT = "artifacts/p4_full/order9/c3a_curation_pilot_v2"
PILOT_VERSION = "order9_c3a_collision_aware_curation_pilot_v2"
BATCH_VERSION = "order9_c3a_contact_penetration_review_batch_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--neutral-only",
        action="store_true",
        help="Render structure/anchor selection views even when assignments exist.",
    )
    parser.add_argument(
        "--skip-screenshots",
        action="store_true",
        help="Write interactive viewers without invoking headless Chrome.",
    )
    parser.add_argument("--chrome", default="google-chrome")
    parser.add_argument("--screenshot-width", type=int, default=1800)
    parser.add_argument("--screenshot-height", type=int, default=1300)
    parser.add_argument(
        "--recompute-case",
        help="Regenerate one existing case with the next deterministic IK seed.",
    )
    parser.add_argument(
        "--recompute-index",
        type=int,
        default=0,
        help="First deterministic collision-aware IK attempt for recomputation.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = _resolve(args.config)
    output = _resolve(args.output)
    config = _load_config(config_path)
    sources = config["sources"]
    pool_path = _resolve(sources["morphology_pool"])
    asset_manifest_path = _resolve(sources["morphology_assets"])
    robot_model_path = _resolve(sources["robot_model_config"])
    source_urdf = _resolve(sources["source_urdf"])
    task_path = _resolve(sources["canonical_order8_task_bucket"])
    physical_model = build_physical_model_from_config(robot_model_path)
    pool = Order3MorphologyPoolManifest.from_json(
        pool_path.read_text(encoding="utf-8")
    )
    if pool.physical_model_hash != physical_model.stable_hash():
        raise SchemaValidationError("curation pool PhysicalModel hash is stale")
    assets = load_order9_morphology_asset_manifest(asset_manifest_path)
    validate_order9_morphology_asset_manifest_bytes(
        assets,
        repository_root=REPOSITORY_ROOT,
        expected_pool_sha256=hash_file(pool_path),
    )
    curation_version = str(config["version"])
    task = _pilot_task(
        task_path,
        curation_version=curation_version,
    )
    output.mkdir(parents=True, exist_ok=True)
    cases_root = output / "cases"
    shared = output / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    if args.recompute_case:
        return _recompute_existing_case(
            case_id=str(args.recompute_case),
            recompute_index=int(args.recompute_index),
            config=config,
            output=output,
            cases_root=cases_root,
            shared=shared,
            pool=pool,
            assets=assets,
            physical_model=physical_model,
            source_urdf=source_urdf,
            task=task,
            chrome=None if args.skip_screenshots else args.chrome,
            screenshot_width=args.screenshot_width,
            screenshot_height=args.screenshot_height,
        )

    prepared: list[dict[str, Any]] = []
    all_mesh_paths: set[Path] = set()
    for index, case in enumerate(config["cases"], start=1):
        graph, graph_source = _case_graph(
            case,
            config=config,
            pool=pool,
            physical_model=physical_model,
        )
        urdf_path = _case_urdf(
            case,
            graph=graph,
            assets=assets,
            source_urdf=source_urdf,
            cases_root=cases_root,
        )
        neutral_scene = build_order9_c3_urdf_mesh_scene(urdf_path)
        all_mesh_paths.update(neutral_scene.mesh_paths)
        prepared.append(
            {
                "case_config": case,
                "graph": graph,
                "graph_source": graph_source,
                "urdf_path": urdf_path,
                "neutral_scene": neutral_scene,
            }
        )
        print(
            f"ORDER9_C3A_PREPARE={index}/{len(config['cases'])}:"
            f"{case['case_id']}:{case['structural_hash']}",
            flush=True,
        )

    mesh_library = write_order9_c3_stl_mesh_library(
        all_mesh_paths,
        shared / "mesh_library.js",
    )
    viewer_js = write_order9_c3_viewer_javascript(
        shared / "order9_c3_mesh_viewer.js"
    )
    print(
        "ORDER9_C3A_SHARED_MESH_LIBRARY="
        f"{mesh_library}:{mesh_library.stat().st_size}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for index, item in enumerate(prepared, start=1):
        case = item["case_config"]
        graph = item["graph"]
        case_dir = cases_root / case["case_id"]
        neutral_markers = _neutral_surface_markers(graph, physical_model)
        neutral_html = case_dir / "neutral_mesh.html"
        neutral_artifacts = render_order9_c3_mesh_viewer(
            urdf_scene=item["neutral_scene"],
            html_path=neutral_html,
            mesh_library_path=mesh_library,
            viewer_javascript_path=viewer_js,
            title=(
                f"{case['case_id']} — {case['module_count']} modules "
                f"{case['topology']}"
            ),
            subtitle=(
                "Neutral joint pose; P labels are free mesh-backed Dock "
                "surface port IDs"
            ),
            markers=neutral_markers,
            metadata={
                "pilot_version": curation_version,
                "inspection_stage": "neutral_anchor_selection",
                "structural_hash": case["structural_hash"],
                "structural_source": case["structural_source"],
                "graph_source": item["graph_source"],
            },
            default_view="top",
        )
        neutral_png = case_dir / "neutral_top.png"
        if not args.skip_screenshots:
            _screenshot(
                args.chrome,
                neutral_html,
                neutral_png,
                view="top",
                detail="shape",
                width=args.screenshot_width,
                height=args.screenshot_height,
            )

        record: dict[str, Any] = {
            "case_id": case["case_id"],
            "module_count": case["module_count"],
            "topology": case["topology"],
            "structural_hash": case["structural_hash"],
            "structural_source": case["structural_source"],
            "graph_id": graph.graph_id,
            "morphology_hash": graph.stable_hash(),
            "graph_path": _portable(case_dir / "source/morphology_graph.json"),
            "graph_sha256": hash_file(case_dir / "source/morphology_graph.json"),
            "urdf_path": _portable(item["urdf_path"]),
            "urdf_sha256": hash_file(item["urdf_path"]),
            "free_surface_port_ids": [
                surface.port_global_id
                for surface in resolve_unoccupied_gripper_surfaces(
                    graph, physical_model
                )
            ],
            "neutral_viewer_path": _portable(neutral_artifacts.html_path),
            "neutral_scene_path": _portable(neutral_artifacts.scene_path),
            "neutral_top_png_path": (
                _portable(neutral_png) if neutral_png.is_file() else None
            ),
            "surface_port_ids": case.get("surface_port_ids"),
            "candidate_group_id": case.get("candidate_group_id"),
            "review_status": "pending_anchor_selection",
            "final_pose": None,
        }
        surface_ids = case.get("surface_port_ids")
        if not args.neutral_only and surface_ids is not None:
            final = _solve_and_render_final(
                case=case,
                graph=graph,
                urdf_path=item["urdf_path"],
                task=task,
                physical_model=physical_model,
                mesh_library=mesh_library,
                viewer_js=viewer_js,
                case_dir=case_dir,
                chrome=None if args.skip_screenshots else args.chrome,
                screenshot_width=args.screenshot_width,
                screenshot_height=args.screenshot_height,
                recompute_index=0,
                curation_version=curation_version,
            )
            record["review_status"] = "pending_final_pose_review"
            record["surface_port_ids"] = list(final["surface_port_ids"])
            record["candidate_group_id"] = final["candidate_group_id"]
            record["final_pose"] = final
        records.append(record)
        print(
            f"ORDER9_C3A_RENDER={index}/{len(prepared)}:"
            f"{case['case_id']}:{record['review_status']}",
            flush=True,
        )

    manifest = {
        "pilot_version": curation_version,
        "viewer_version": ORDER9_C3_CURATION_VIEWER_VERSION,
        "semantic_scope": (
            "two-contact human actual-mesh curation only; not production "
            "bucket, Isaac collision/path, dynamics, or task-success evidence"
        ),
        "review_status": (
            "pending_final_pose_review"
            if all(record["final_pose"] is not None for record in records)
            else "pending_anchor_selection"
        ),
        "source_config_path": _portable(config_path),
        "source_config_sha256": hash_file(config_path),
        "source_pool_path": _portable(pool_path),
        "source_pool_sha256": hash_file(pool_path),
        "source_asset_manifest_path": _portable(asset_manifest_path),
        "source_asset_manifest_sha256": hash_file(asset_manifest_path),
        "physical_model_hash": physical_model.stable_hash(),
        "task_spec_path": _portable(task_path),
        "pilot_task_hash": task.stable_hash(),
        "mesh_library_path": _portable(mesh_library),
        "mesh_library_sha256": hash_file(mesh_library),
        "case_count": len(records),
        "cases": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_index(output / "index.html", manifest)
    print(
        "ORDER9_C3A_CURATION="
        + json.dumps(
            {
                "case_count": len(records),
                "review_status": manifest["review_status"],
                "manifest": _portable(manifest_path),
                "manifest_sha256": hash_file(manifest_path),
                "index": _portable(output / "index.html"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("version") not in {PILOT_VERSION, BATCH_VERSION}
    ):
        raise SchemaValidationError("C3a curation config version mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise SchemaValidationError("C3a curation cases must be a list")
    if payload["version"] == PILOT_VERSION:
        if len(cases) != 12:
            raise SchemaValidationError(
                "C3a pilot requires exactly 12 cases"
            )
        expected = [
            (2, "chain"),
            (3, "chain"),
            (4, "chain"),
            (4, "branched"),
            (5, "chain"),
            (5, "branched"),
            (6, "chain"),
            (6, "branched"),
            (7, "chain"),
            (7, "branched"),
            (8, "chain"),
            (8, "branched"),
        ]
        actual = [
            (
                int(case.get("module_count", -1)),
                str(case.get("topology", "")),
            )
            for case in cases
        ]
        if actual != expected:
            raise SchemaValidationError(
                "C3a pilot cases must preserve the approved "
                "module/topology order"
            )
    else:
        selection = payload.get("selection", {})
        required_count = int(
            selection.get(
                "required_case_count",
                -1,
            )
        )
        if required_count <= 0 or len(cases) != required_count:
            raise SchemaValidationError(
                "C3a review batch case count does not match "
                "selection.required_case_count"
            )
        gate = CollisionAwareIKConfig()
        if not math.isclose(
            float(selection.get(
                "selected_contact_penetration_limit_m",
                math.nan,
            )),
            gate.max_selected_contact_penetration_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SchemaValidationError(
                "C3a review batch selected-contact penetration limit "
                "does not match the production collision gate"
            )
        if not math.isclose(
            float(selection.get(
                "non_contact_collision_margin_m",
                math.nan,
            )),
            gate.collision_margin_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SchemaValidationError(
                "C3a review batch non-contact collision margin does not "
                "match the production collision gate"
            )
    case_ids = [str(case.get("case_id", "")) for case in cases]
    hashes = [str(case.get("structural_hash", "")) for case in cases]
    if (
        any(not value for value in case_ids + hashes)
        or len(case_ids) != len(set(case_ids))
        or len(hashes) != len(set(hashes))
    ):
        raise SchemaValidationError("C3a pilot case ids/hashes must be unique")
    for case in cases:
        surfaces = case.get("surface_port_ids")
        if surfaces is not None and (
            not isinstance(surfaces, list)
            or len(surfaces) != 2
            or len(set(int(value) for value in surfaces)) != 2
        ):
            raise SchemaValidationError(
                "surface_port_ids must be null or two distinct ids"
            )
        group_id = case.get("candidate_group_id")
        if group_id is not None and not str(group_id):
            raise SchemaValidationError(
                "candidate_group_id must be null or non-empty"
            )
    return payload


def _recompute_existing_case(
    *,
    case_id: str,
    recompute_index: int,
    config: dict[str, Any],
    output: Path,
    cases_root: Path,
    shared: Path,
    pool: Order3MorphologyPoolManifest,
    assets: object,
    physical_model: object,
    source_urdf: Path,
    task: TaskSpec,
    chrome: str | None,
    screenshot_width: int,
    screenshot_height: int,
) -> int:
    manifest_path = output / "manifest.json"
    mesh_library = shared / "mesh_library.js"
    viewer_js = shared / "order9_c3_mesh_viewer.js"
    for required in (manifest_path, mesh_library, viewer_js):
        if not required.is_file():
            raise FileNotFoundError(required)
    matches = [
        value for value in config["cases"] if value["case_id"] == case_id
    ]
    if len(matches) != 1:
        raise SchemaValidationError(
            f"recompute case {case_id!r} is not in the pilot config"
        )
    case = matches[0]
    graph, _graph_source = _case_graph(
        case,
        config=config,
        pool=pool,
        physical_model=physical_model,
    )
    urdf_path = _case_urdf(
        case,
        graph=graph,
        assets=assets,
        source_urdf=source_urdf,
        cases_root=cases_root,
    )
    final = _solve_and_render_final(
        case=case,
        graph=graph,
        urdf_path=urdf_path,
        task=task,
        physical_model=physical_model,
        mesh_library=mesh_library,
        viewer_js=viewer_js,
        case_dir=cases_root / case_id,
        chrome=chrome,
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        recompute_index=recompute_index,
        curation_version=str(config["version"]),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        value for value in manifest["cases"] if value["case_id"] == case_id
    ]
    if len(records) != 1:
        raise SchemaValidationError(
            f"recompute manifest lacks one case {case_id!r}"
        )
    records[0]["final_pose"] = final
    records[0]["review_status"] = "pending_user_accept_or_reject"
    manifest["viewer_version"] = ORDER9_C3_CURATION_VIEWER_VERSION
    manifest["review_status"] = "pending_final_pose_review"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reset_order9_c3_review_after_recalculation(
        output,
        case_id=case_id,
        note="",
        recompute_index=int(final["collision_aware"]["attempt_index"]),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_index(output / "index.html", manifest)
    print(
        "ORDER9_C3A_RECOMPUTE="
        + json.dumps(
            {
                "case_id": case_id,
                "attempt_index": final["collision_aware"]["attempt_index"],
                "viewer": final["viewer_path"],
                "manifest": _portable(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _pilot_task(
    path: Path,
    *,
    curation_version: str = PILOT_VERSION,
) -> TaskSpec:
    task = TaskSpec.from_json(path.read_text(encoding="utf-8"))
    task.task_id = "order9-c3a-manual-curation-pilot"
    task.metadata = {
        key: value
        for key, value in task.metadata.items()
        if key
        not in {
            "dataset_split",
            "order9_c3_articulated_teacher_precheck",
            "order9_rollout_bucket_id",
        }
    }
    task.metadata = {
        **task.metadata,
        "dataset_split": "train",
        "curation_pilot_version": str(curation_version),
        "curation_only": True,
    }
    task.validate()
    return task


def _case_graph(
    case: dict[str, Any],
    *,
    config: dict[str, Any],
    pool: Order3MorphologyPoolManifest,
    physical_model: object,
) -> tuple[MorphologyGraph, dict[str, object]]:
    requested_hash = str(case["structural_hash"])
    if case["structural_source"] == "existing_train_pool":
        matches = [
            entry
            for entry in pool.entries
            if entry.structural_hash == requested_hash
        ]
        if len(matches) != 1 or matches[0].split.value != "train":
            raise SchemaValidationError(
                f"{case['case_id']} does not resolve to one train-pool graph"
            )
        graph = matches[0].morphology_graph
        source = {
            "pool_split": matches[0].split.value,
            "requested_seed": matches[0].requested_seed,
            "accepted_proposal_seed": matches[0].accepted_proposal_seed,
        }
    elif case["structural_source"] == "generated_train_candidate":
        seed = int(config["selection"]["generated_eight_module_chain_seed"])
        graph = RandomConnectedMorphologyDistribution(physical_model).sample(
            seed=seed,
            module_count=8,
        )
        checker = MorphologyFlightFeasibilityChecker(
            MorphologyFlightFeasibilityConfig(
                mesh_search_dirs=("module_urdf", "module_urdf/mesh")
            )
        )
        feasibility = checker.check(graph, physical_model)
        if not feasibility.feasible:
            codes = sorted(
                violation.code for violation in feasibility.hard_violations
            )
            raise SchemaValidationError(
                "generated eight-module chain is not morphology-flight "
                "feasible: " + ",".join(codes)
            )
        source = {
            "pool_split": "train_candidate_not_in_source_pool",
            "proposal_seed": seed,
            "morphology_flight_feasibility_hash": stable_hash(
                feasibility.to_dict()
            ),
        }
    else:
        raise SchemaValidationError(
            f"unsupported structural_source {case['structural_source']!r}"
        )
    actual_hash = morphology_structural_hash(graph)
    if actual_hash != requested_hash:
        raise SchemaValidationError(
            f"{case['case_id']} structural hash mismatch"
        )
    degrees = {module.module_id: 0 for module in graph.modules}
    for edge in graph.dock_edges:
        degrees[edge.src_module_id] += 1
        degrees[edge.dst_module_id] += 1
    topology = "branched" if max(degrees.values()) >= 3 else "chain"
    if len(graph.modules) != int(case["module_count"]) or topology != case["topology"]:
        raise SchemaValidationError(
            f"{case['case_id']} module/topology class mismatch"
        )
    return graph, {**source, "module_degrees": degrees}


def _case_urdf(
    case: dict[str, Any],
    *,
    graph: MorphologyGraph,
    assets: object,
    source_urdf: Path,
    cases_root: Path,
) -> Path:
    case_dir = cases_root / case["case_id"] / "source"
    case_dir.mkdir(parents=True, exist_ok=True)
    graph_path = case_dir / "morphology_graph.json"
    graph_path.write_text(graph.to_json(indent=2) + "\n", encoding="utf-8")
    if case["structural_source"] == "existing_train_pool":
        entry = assets.entry_for(graph)
        urdf = _resolve(entry.urdf_path)
    else:
        urdf = case_dir / f"holon_order9_articulated_v2_{case['structural_hash'][:12]}.urdf"
        write_articulated_morphology_graph_urdf(
            source_urdf,
            urdf,
            morphology_graph=graph,
            mesh_search_dirs=[
                REPOSITORY_ROOT / "module_urdf",
                REPOSITORY_ROOT / "module_urdf/mesh",
            ],
        )
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    return urdf


def _neutral_surface_markers(
    graph: MorphologyGraph,
    physical_model: object,
) -> tuple[Order9C3ViewerMarker, ...]:
    markers = []
    for surface in resolve_unoccupied_gripper_surfaces(graph, physical_model):
        markers.append(
            Order9C3ViewerMarker(
                marker_id=f"surface-{surface.port_global_id}",
                label=(
                    f"P{surface.port_global_id} "
                    f"M{surface.module_id} "
                    f"{surface.port_local_id}"
                ),
                position_world=tuple(
                    surface.grasp_contact_frame_design[:3]
                ),
                direction_world=tuple(surface.neutral_outward_axis_design),
                color_rgba=(0.96, 0.22, 0.08, 1.0),
                kind="surface",
            )
        )
    return tuple(markers)


def _solve_and_render_final(
    *,
    case: dict[str, Any],
    graph: MorphologyGraph,
    urdf_path: Path,
    task: TaskSpec,
    physical_model: object,
    mesh_library: Path,
    viewer_js: Path,
    case_dir: Path,
    chrome: str | None,
    screenshot_width: int,
    screenshot_height: int,
    recompute_index: int,
    curation_version: str = PILOT_VERSION,
) -> dict[str, Any]:
    requested_group = case.get("candidate_group_id")
    bundle = build_order9_c3_articulated_teacher(
        task_spec=task,
        structural_target=graph,
        physical_model=physical_model,
        config=Order9C3TeacherConfig(
            maximum_surface_pair_attempts=1,
            preferred_surface_port_ids=tuple(
                int(value) for value in case["surface_port_ids"]
            ),
            preferred_candidate_group_id=(
                None if requested_group is None else str(requested_group)
            ),
            require_full_trajectory_recheck=False,
        ),
    )
    active_knot = next(
        knot
        for knot in reversed(bundle.trajectory.knots)
        if any(
            assignment.schedule_state in {"attach", "maintain", "slide"}
            for assignment in knot.contact_assignments
        )
    )
    assignments = [
        assignment
        for assignment in active_knot.contact_assignments
        if assignment.schedule_state in {"attach", "maintain", "slide"}
    ]
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in bundle.contact_candidate_set.candidates
    }
    object_ids = {
        candidate_by_id[assignment.candidate_id].target_entity_id
        for assignment in assignments
    }
    if len(object_ids) != 1:
        raise SchemaValidationError(
            "C3 curation collision solve requires one assigned object"
        )
    object_box = _task_box(task, next(iter(object_ids)))
    solution, collision_evidence = _collision_aware_final_solution(
        bundle=bundle,
        physical_model=physical_model,
        object_box=object_box,
        assignments=assignments,
        recompute_index=recompute_index,
    )
    final_scene = build_order9_c3_urdf_mesh_scene(
        urdf_path,
        joint_positions_rad=solution.joint_positions_rad,
        root_pose_world=order9_c3_urdf_root_pose_from_baselink_pose(
            urdf_path,
            solution.base_pose_world,
        ),
    )
    selected_anchor_ids = [assignment.anchor_id for assignment in assignments]
    references = resolve_mesh_backed_anchor_references(
        bundle.design_output.target_morphology,
        physical_model,
        selected_anchor_ids,
    )
    surface_by_anchor = {
        reference.anchor.anchor_id: reference.surface
        for reference in references
    }
    assignment_by_surface = {
        surface_by_anchor[assignment.anchor_id].port_global_id: assignment
        for assignment in assignments
    }
    _validate_rendered_anchor_poses(
        final_scene=final_scene,
        references=references,
        solution=solution,
    )
    markers: list[Order9C3ViewerMarker] = []
    for surface in resolve_unoccupied_gripper_surfaces(graph, physical_model):
        link_id = (
            f"module_{surface.module_id}__"
            f"{surface.mechanism_link_id}"
        )
        pose = compose_pose(
            final_scene.link_poses_world[link_id],
            surface.grasp_contact_frame_link,
        )
        assignment = assignment_by_surface.get(surface.port_global_id)
        label = f"P{surface.port_global_id} M{surface.module_id}"
        selected = assignment is not None
        if assignment is not None:
            candidate = candidate_by_id[assignment.candidate_id]
            label = (
                f"A{assignment.anchor_id}/P{surface.port_global_id} → "
                f"C{candidate.candidate_id} {candidate.region_id}"
            )
        markers.append(
            Order9C3ViewerMarker(
                marker_id=f"surface-{surface.port_global_id}",
                label=label,
                position_world=tuple(pose[:3]),
                direction_world=transformed_link_axis(pose),
                color_rgba=(
                    (1.0, 0.78, 0.05, 1.0)
                    if selected
                    else (0.84, 0.16, 0.12, 0.82)
                ),
                selected=selected,
                kind="surface",
            )
        )
    contact_records = []
    for assignment in assignments:
        candidate = candidate_by_id[assignment.candidate_id]
        markers.append(
            Order9C3ViewerMarker(
                marker_id=f"contact-{candidate.candidate_id}",
                label=f"C{candidate.candidate_id} {candidate.region_id}",
                position_world=tuple(candidate.contact_pose_world[:3]),
                direction_world=tuple(candidate.normal_world),
                color_rgba=(0.05, 0.72, 0.28, 1.0),
                selected=True,
                kind="contact",
            )
        )
        contact_records.append(
            {
                "anchor_id": assignment.anchor_id,
                "surface_port_id": (
                    surface_by_anchor[assignment.anchor_id].port_global_id
                ),
                "module_id": (
                    surface_by_anchor[assignment.anchor_id].module_id
                ),
                "candidate_id": candidate.candidate_id,
                "region_id": candidate.region_id,
                "contact_pose_world": list(candidate.contact_pose_world),
                "normal_world": list(candidate.normal_world),
            }
        )
    boxes = (object_box,)
    final_html = case_dir / "final_grasp_mesh.html"
    evidence = order9_c3_teacher_evidence(bundle)
    final_artifacts = render_order9_c3_mesh_viewer(
        urdf_scene=final_scene,
        html_path=final_html,
        mesh_library_path=mesh_library,
        viewer_javascript_path=viewer_js,
        title=f"{case['case_id']} — IK final grasp pose",
        subtitle=(
            "Yellow: selected robot anchors; green: object contact points; "
            "arrows show outward normals"
        ),
        markers=markers,
        boxes=boxes,
        metadata={
            "pilot_version": str(curation_version),
            "inspection_stage": "final_ik_pose",
            "structural_hash": case["structural_hash"],
            "surface_port_ids": list(bundle.selected_surface_port_ids),
            "candidate_group_id": bundle.trajectory_plan.candidate_group_id,
            "teacher_evidence": evidence,
            "collision_aware_ik": collision_evidence,
            "production_trajectory_recheck_status": (
                "not_run_static_curation_final_pose_only"
            ),
            "review": {
                "case_id": case["case_id"],
                "api_path": "/api/order9-c3-review",
            },
        },
        default_view="iso",
    )
    final_png = case_dir / "final_grasp_iso.png"
    if chrome is not None:
        _screenshot(
            chrome,
            final_html,
            final_png,
            view="iso",
            detail="full",
            width=screenshot_width,
            height=screenshot_height,
        )
    pitch_values = [
        abs(float(value))
        for joint_id, value in solution.joint_positions_rad.items()
        if "pitch_dock_mech_joint" in joint_id
    ]
    return {
        "surface_port_ids": list(bundle.selected_surface_port_ids),
        "candidate_group_id": bundle.trajectory_plan.candidate_group_id,
        "contacts": contact_records,
        "ik": {
            "solver_version": solution.solver_version,
            "iterations": solution.iterations,
            "maximum_position_error_m": solution.maximum_position_error_m,
            "maximum_attitude_error_rad": solution.maximum_attitude_error_rad,
            "maximum_absolute_pitch_deg": math.degrees(
                max(pitch_values, default=0.0)
            ),
            "base_pose_world": list(solution.base_pose_world),
            "centroidal_pose_world": list(solution.centroidal_pose_world),
            "joint_positions_rad": dict(
                sorted(solution.joint_positions_rad.items())
            ),
        },
        "collision_aware": collision_evidence,
        "teacher_ik": {
            "solver_version": bundle.trajectory_plan.ik_solution.solver_version,
            "maximum_position_error_m": (
                bundle.trajectory_plan.ik_solution.maximum_position_error_m
            ),
            "maximum_normal_error_rad": (
                bundle.trajectory_plan.ik_solution.maximum_normal_error_rad
            ),
        },
        "teacher_evidence": evidence,
        "production_trajectory_recheck_status": (
            "not_run_static_curation_final_pose_only"
        ),
        "viewer_path": _portable(final_artifacts.html_path),
        "scene_path": _portable(final_artifacts.scene_path),
        "iso_png_path": (
            _portable(final_png)
            if chrome is not None and final_png.is_file()
            else None
        ),
        "review_status": "pending_user_accept_or_reject",
        "isaac_admission_status": "not_run",
    }


def _collision_aware_final_solution(
    *,
    bundle: object,
    physical_model: object,
    object_box: Order9C3ViewerBox,
    assignments: list[object],
    recompute_index: int,
) -> tuple[object, dict[str, Any]]:
    if recompute_index < 0:
        raise ValueError("recompute_index must be non-negative")
    morphology = bundle.design_output.target_morphology
    teacher_solution = bundle.trajectory_plan.ik_solution
    anchor_ids = tuple(
        sorted(int(assignment.anchor_id) for assignment in assignments)
    )
    anchor_targets = {
        anchor_id: teacher_solution.anchor_poses_world[anchor_id]
        for anchor_id in anchor_ids
    }
    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        collision_object=Order9PostureCollisionObject(
            object_id=object_box.object_id,
            size_m=object_box.size_m,
            initial_pose_world=object_box.pose_world,
        ),
        require_native_solver=True,
    )
    solver = resolver.ik_solver
    solver.set_collision_scene(
        morphology=morphology,
        object_pose_world=object_box.pose_world,
        object_size_m=object_box.size_m,
        allowed_anchor_ids=anchor_ids,
    )
    failures = []
    for attempt_index in range(recompute_index, recompute_index + 16):
        centroidal_target = _curation_centroidal_target(
            teacher_solution.centroidal_pose_world,
            attempt_index,
        )
        initial_q = _curation_joint_seed(
            teacher_solution.joint_positions_rad,
            morphology=morphology,
            physical_model=physical_model,
            attempt_index=attempt_index,
        )
        candidate = solver.solve(
            morphology=morphology,
            centroidal_pose_world=centroidal_target,
            anchor_pose_targets_world=anchor_targets,
            initial_joint_positions_rad=initial_q,
        )
        if not candidate.feasible:
            failures.append(
                f"{attempt_index}:ik:"
                f"{candidate.maximum_position_error_m:.6g}/"
                f"{candidate.maximum_attitude_error_rad:.6g}"
            )
            continue
        admission = solver.check_configuration(
            morphology=morphology,
            centroidal_pose_world=candidate.centroidal_pose_world,
            joint_positions_rad=candidate.joint_positions_rad,
            exact=False,
            margin_m=float(solver.collision_config.collision_margin_m),
        )
        if not bool(admission.get("accepted")):
            failures.append(
                f"{attempt_index}:collision:"
                f"{admission.get('minimum_clearance_m')}:"
                "selected_contact_penetration:"
                f"{admission.get('maximum_selected_contact_penetration_m')}"
            )
            continue
        return candidate, {
            "attempt_index": attempt_index,
            "solver_version": candidate.solver_version,
            "collision_gate_version": candidate.solver_version,
            "collision_margin_m": float(
                solver.collision_config.collision_margin_m
            ),
            "minimum_clearance_m": float(
                admission["minimum_clearance_m"]
            ),
            "violating_pair_count": int(
                admission["violating_pair_count"]
            ),
            "selected_contact_penetration_limit_m": float(
                admission["selected_contact_penetration_limit_m"]
            ),
            "maximum_selected_contact_penetration_m": float(
                admission["maximum_selected_contact_penetration_m"]
            ),
            "selected_contact_penetration_violating_pair_count": int(
                admission[
                    "selected_contact_penetration_violating_pair_count"
                ]
            ),
            "selected_contact_pairs": list(
                admission["selected_contact_pairs"]
            ),
            "centroidal_offset_m": [
                float(centroidal_target[index])
                - float(teacher_solution.centroidal_pose_world[index])
                for index in range(3)
            ],
            "joint_solution_hash": stable_hash(
                dict(sorted(candidate.joint_positions_rad.items()))
            ),
            "admission_exact_mesh": False,
            "admission_status": "accepted",
        }
    raise SchemaValidationError(
        "C3 curation exhausted collision-aware final-pose attempts: "
        + "; ".join(failures[:8])
    )


def _validate_rendered_anchor_poses(
    *,
    final_scene: object,
    references: tuple[object, ...],
    solution: object,
) -> None:
    for reference in references:
        surface = reference.surface
        link_id = (
            f"module_{surface.module_id}__"
            f"{surface.mechanism_link_id}"
        )
        rendered = compose_pose(
            final_scene.link_poses_world[link_id],
            surface.grasp_contact_frame_link,
        )
        expected = solution.anchor_poses_world[
            reference.anchor.anchor_id
        ]
        position_error = math.sqrt(
            sum(
                (
                    float(rendered[index])
                    - float(expected[index])
                )
                ** 2
                for index in range(3)
            )
        )
        if position_error > 1.0e-6:
            raise SchemaValidationError(
                "C3 viewer URDF frame conversion disagrees with posture IK "
                f"for anchor {reference.anchor.anchor_id}: "
                f"{position_error:.6g} m"
            )


def _curation_centroidal_target(
    reference: tuple[float, ...],
    attempt_index: int,
) -> tuple[float, ...]:
    offsets = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.02),
        (0.0, 0.0, -0.02),
        (0.02, 0.0, 0.0),
        (-0.02, 0.0, 0.0),
        (0.0, 0.02, 0.0),
        (0.0, -0.02, 0.0),
        (0.03, 0.03, 0.0),
        (0.03, -0.03, 0.0),
        (-0.03, 0.03, 0.0),
        (-0.03, -0.03, 0.0),
        (0.0, 0.0, 0.04),
        (0.0, 0.0, -0.04),
    )
    ring = attempt_index // len(offsets)
    scale = 1.0 + 0.5 * float(ring)
    offset = offsets[attempt_index % len(offsets)]
    return (
        float(reference[0]) + scale * offset[0],
        float(reference[1]) + scale * offset[1],
        float(reference[2]) + scale * offset[2],
        *tuple(float(value) for value in reference[3:7]),
    )


def _curation_joint_seed(
    reference: dict[str, float],
    *,
    morphology: MorphologyGraph,
    physical_model: object,
    attempt_index: int,
) -> dict[str, float]:
    joint_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    limits = _global_joint_limits(morphology, physical_model, joint_ids)
    if attempt_index == 0:
        return dict(reference)
    if attempt_index == 1:
        return {joint_id: 0.0 for joint_id in joint_ids}
    output = {}
    for index, joint_id in enumerate(joint_ids):
        amplitude = (
            0.04 if "pitch_dock_mech_joint" in joint_id else 0.12
        )
        value = float(reference[joint_id]) + amplitude * math.sin(
            float((attempt_index + 1) * (index + 1))
        )
        lower, upper = limits[joint_id]
        output[joint_id] = min(max(value, lower), upper)
    return output


def _task_box(task: TaskSpec, object_id: str) -> Order9C3ViewerBox:
    obj = next(value for value in task.scene.objects if value.object_id == object_id)
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == obj.geometry_id
    )
    if geometry.geometry_type != GeometryType.BOX:
        raise SchemaValidationError(
            "C3a curation pilot currently expects the conservative box object"
        )
    size = tuple(
        float(value) * float(geometry.scale[index])
        for index, value in enumerate(geometry.primitive_params["size_m"])
    )
    return Order9C3ViewerBox(
        object_id=obj.object_id,
        pose_world=obj.pose_world,
        size_m=size,  # type: ignore[arg-type]
    )


def _screenshot(
    chrome: str,
    html_path: Path,
    output_path: Path,
    *,
    view: str,
    detail: str,
    width: int,
    height: int,
) -> None:
    if width < 800 or height < 600:
        raise ValueError("curation screenshots must be at least 800x600")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        html_path.resolve().as_uri()
        + f"?view={view}&detail={detail}&capture=1"
    )
    with tempfile.TemporaryDirectory(prefix="amsrr-c3a-chrome-") as profile:
        command = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--allow-file-access-from-files",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            "--force-device-scale-factor=1",
            "--hide-scrollbars",
            "--virtual-time-budget=12000",
            f"--screenshot={output_path.resolve()}",
            url,
        ]
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90.0,
            check=False,
        )
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            "headless Chrome screenshot failed: "
            + result.stdout[-2000:]
        )


def _write_index(path: Path, manifest: dict[str, Any]) -> None:
    rows = []
    root = path.parent
    for case in manifest["cases"]:
        neutral = Path(case["neutral_viewer_path"])
        neutral_png = case["neutral_top_png_path"]
        final = case["final_pose"]
        final_link = (
            "pending"
            if final is None
            else (
                f'<a href="{_relative(Path(final["viewer_path"]), root)}">'
                "interactive final grasp</a>"
            )
        )
        image = (
            ""
            if neutral_png is None
            else (
                f'<a href="{_relative(Path(neutral_png), root)}">'
                '<img loading="lazy" '
                f'src="{_relative(Path(neutral_png), root)}"></a>'
            )
        )
        rows.append(
            "<tr>"
            f"<td>{case['case_id']}</td>"
            f"<td>{case['module_count']}</td>"
            f"<td>{case['topology']}</td>"
            f'<td><a href="{_relative(neutral, root)}">interactive neutral</a></td>'
            f"<td>{final_link}</td>"
            f'<td data-review-case="{case["case_id"]}">'
            f"{case['review_status']}</td>"
            f"<td>{image}</td>"
            "</tr>"
        )
    path.write_text(
        """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>Order 9 C3a curation pilot</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#17202a}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd1d1;padding:7px;vertical-align:top}
th{background:#edf2f7}img{width:300px;height:auto}code{font-size:12px}</style></head><body>
<h1>Order 9 C3a — """
        + str(int(manifest["case_count"]))
        + """-case mesh curation</h1>
<p><code>"""
        + str(manifest["pilot_version"])
        + """</code></p>
<p>This is human-inspection evidence only. It is not Isaac collision/path or grasp-success evidence.</p>
<table><thead><tr><th>case</th><th>modules</th><th>topology</th><th>neutral</th>
<th>final</th><th>status</th><th>top view</th></tr></thead><tbody>
"""
        + "\n".join(rows)
        + """</tbody></table>
<script>
for (const cell of document.querySelectorAll("[data-review-case]")) {
  fetch(`/api/order9-c3-review?case_id=${encodeURIComponent(cell.dataset.reviewCase)}`)
    .then(response => response.json())
    .then(payload => { if (payload.ok) cell.textContent = payload.review_status; })
    .catch(() => {});
}
</script></body></html>
""",
        encoding="utf-8",
    )


def _relative(target: Path, origin: Path) -> str:
    import os

    return Path(os.path.relpath(target.resolve(), origin.resolve())).as_posix()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (
        value.resolve()
        if value.is_absolute()
        else (REPOSITORY_ROOT / value).resolve()
    )


def _portable(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
