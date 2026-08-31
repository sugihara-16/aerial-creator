#!/usr/bin/env python3
from __future__ import annotations

"""Replay one persisted R1 geometry audit to test deterministic results."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.feasibility.order9_native_loader_v13 import (  # noqa: E402
    activate_order9_posture_native_v13,
)

activate_order9_posture_native_v13()

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.contact_candidates import ContactCandidateSet  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.policies import ContactWrenchTrajectory  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_complete_task_geometry_v13 import (  # noqa: E402
    audit_order9_r1_complete_task_geometry_v13,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)

DEFAULT_CASE = (
    REPOSITORY
    / "artifacts/p4_full/order9/r1_teacher/diagnostics"
    / "v12_persistence_drift_lattice01_case"
)
DEFAULT_OUTPUT = (
    REPOSITORY
    / "artifacts/p4_full/order9/r1_teacher/diagnostics"
    / "v13_exact_margin_geometry_parallel_replay.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", default=str(DEFAULT_CASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--required-clearance-m", type=float, default=0.0195)
    parser.add_argument("--repeat-count", type=int, default=8)
    parser.add_argument("--process-count", type=int, default=4)
    return parser


def _repository_file(value: str) -> Path:
    path = (REPOSITORY / value).resolve()
    if REPOSITORY not in path.parents or not path.is_file():
        raise SchemaValidationError("R1 geometry replay binding is invalid")
    return path


def _run_once(arguments: tuple[str, float, int]) -> dict[str, Any]:
    case_root_text, required_clearance_m, repetition_index = arguments
    case_root = Path(case_root_text).resolve()
    manifest = json.loads((case_root / "case_manifest.json").read_text())
    artifact_path = _repository_file(manifest["nominal_artifact"]["path"])
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    artifact_root = artifact_path.parent
    phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_root / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }
    task_spec = TaskSpec.from_json((case_root / "task_spec.json").read_text())
    morphology = MorphologyGraph.from_json(
        (artifact_root / artifact.task_conditioned_morphology_path).read_text()
    )
    candidates = ContactCandidateSet.from_json(
        (artifact_root / artifact.contact_candidate_set_path).read_text()
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    audit = audit_order9_r1_complete_task_geometry_v13(
        phases=phases,
        task_spec=task_spec,
        morphology=morphology,
        contact_candidate_set=candidates,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
        contract=replace(
            geometry,
            required_robot_support_clearance_m=required_clearance_m,
        ),
        fail_fast=True,
    )
    return {
        "repetition_index": repetition_index,
        "accepted": audit["accepted"],
        "minimum_robot_support_clearance_lower_bound_m": audit[
            "minimum_robot_support_clearance_lower_bound_m"
        ],
        "failure_examples": audit["failure_examples"],
        "checked_knot_count": audit["checked_knot_count"],
        "computed_collision_scene_count": audit["computed_collision_scene_count"],
        "convex_proof_scene_count": audit.get("convex_proof_scene_count"),
        "exact_fallback_scene_count": audit.get("exact_fallback_scene_count"),
        "exact_mesh_scene_count": audit.get("exact_mesh_scene_count"),
        "exact_mesh_aabb_proof_scene_count": audit.get(
            "exact_mesh_aabb_proof_scene_count"
        ),
        "exact_mesh_narrow_phase_scene_count": audit.get(
            "exact_mesh_narrow_phase_scene_count"
        ),
    }


def main() -> int:
    arguments = _parser().parse_args()
    case_root = Path(arguments.case_root).resolve()
    output = Path(arguments.output).resolve()
    repeat_count = int(arguments.repeat_count)
    process_count = int(arguments.process_count)
    if (
        REPOSITORY not in case_root.parents
        or not (case_root / "case_manifest.json").is_file()
        or REPOSITORY not in output.parents
        or repeat_count < 1
        or process_count < 1
        or process_count > repeat_count
    ):
        raise SchemaValidationError("R1 geometry replay arguments are invalid")
    jobs = [
        (str(case_root), float(arguments.required_clearance_m), index)
        for index in range(repeat_count)
    ]
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        records = list(executor.map(_run_once, jobs))
    records.sort(key=lambda value: value["repetition_index"])
    payload = {
        "diagnostic_version": "order9_r1_exact_margin_geometry_parallel_replay_v13",
        "case_root": case_root.relative_to(REPOSITORY).as_posix(),
        "required_clearance_m": float(arguments.required_clearance_m),
        "repeat_count": repeat_count,
        "process_count": process_count,
        "accepted_count": sum(bool(value["accepted"]) for value in records),
        "unique_minimum_clearances_m": sorted(
            {
                float(value["minimum_robot_support_clearance_lower_bound_m"])
                for value in records
            }
        ),
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "accepted_count": payload["accepted_count"],
                "repeat_count": repeat_count,
                "unique_minimum_clearances_m": payload["unique_minimum_clearances_m"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
