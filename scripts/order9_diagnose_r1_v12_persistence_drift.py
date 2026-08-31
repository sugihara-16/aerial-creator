#!/usr/bin/env python3
from __future__ import annotations

"""Diagnose R1 v12 drift between candidate admission and persisted timing."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.policies import ContactWrenchTrajectory  # noqa: E402
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (  # noqa: E402
    audit_order9_r1_complete_task_geometry_v12,
)
from amsrr.training.order9_r1_complete_task_materialization_v6 import (  # noqa: E402
    finalize_order9_r1_v6_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_nominal_retime import (  # noqa: E402
    _retime_trajectory,
)
from amsrr.training.order9_r1_range_selection_v12 import (  # noqa: E402
    Order9R1ClearanceConstrainedTeacherScreenPipelineV12,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (  # noqa: E402
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
    materialize_order9_r1_v12_candidate_phases,
)
from amsrr.utils.hashing import stable_hash  # noqa: E402
from scripts import order9_run_r1_range_selection_v10 as runtime_v10  # noqa: E402
from scripts import order9_run_r1_range_selection_v11 as runtime_v11  # noqa: E402

DEFAULT_CANDIDATE_ID = (
    "r1_l1_10mm_5deg__validation__validation-000038-e6065f5b3c3e__lattice_01"
)
DEFAULT_OUTPUT = (
    REPOSITORY
    / "artifacts/p4_full/order9/r1_teacher/diagnostics"
    / "v12_persistence_drift_lattice01.json"
)
DEFAULT_CASE_OUTPUT = (
    REPOSITORY
    / "artifacts/p4_full/order9/r1_teacher/diagnostics"
    / "v12_persistence_drift_lattice01_case"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--case-output", default=str(DEFAULT_CASE_OUTPUT))
    return parser


def _audit(phases, *, case, nominal, geometry, planning_clearance_m: float):
    return audit_order9_r1_complete_task_geometry_v12(
        phases=phases,
        task_spec=case.task_spec,
        morphology=nominal.selection_bundle.design_output.target_morphology,
        contact_candidate_set=nominal.selection_bundle.contact_candidate_set,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
        contract=replace(
            geometry,
            required_robot_support_clearance_m=planning_clearance_m,
        ),
        fail_fast=True,
    )


def _load_materialized_phases(materialized):
    artifact_path = REPOSITORY / materialized.manifest.nominal_artifact.path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    root = artifact_path.parent
    return {
        entry.phase: ContactWrenchTrajectory.from_json(
            (root / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }


def main() -> int:
    arguments = _parser().parse_args()
    started = perf_counter()
    base = load_order9_r1_calibration_protocol(
        runtime_v11.BASE_PROTOCOL,
        repository_root=REPOSITORY,
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    cases = runtime_v11._corrected_cases("r1_l1_10mm_5deg", "validation")
    matches = [case for case in cases if case.candidate_id == arguments.candidate_id]
    if len(matches) != 1:
        raise SchemaValidationError("R1 v12 diagnostic candidate identity differs")
    case = matches[0]
    pipeline = Order9R1ClearanceConstrainedTeacherScreenPipelineV12(
        repository_root=REPOSITORY,
        source_bucket_manifest_path=REPOSITORY / base.source_bucket_manifest.path,
        minimum_normalized_joint_limit_reserve=(
            base.minimum_normalized_joint_limit_reserve
        ),
        maximum_body_tilt_rad=base.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
        geometry_contract=geometry,
        clearance_contract=clearance,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
    )
    prepared = pipeline.prepare(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete or prepared.payload is None:
        raise SchemaValidationError(
            f"R1 v12 diagnostic teacher failed: {prepared.failure_reason}"
        )
    screen = pipeline.screen(case, prepared)
    if not screen.accepted or not screen.eligible_for_full_control_test:
        raise SchemaValidationError("R1 v12 diagnostic fast screen rejected")
    phases = materialize_order9_r1_v12_candidate_phases(
        prepared.payload,
        task_spec=case.task_spec,
    )
    planning_clearance_m = clearance.planning_clearance_m
    before = _audit(
        phases,
        case=case,
        nominal=prepared.payload,
        geometry=geometry,
        planning_clearance_m=planning_clearance_m,
    )
    scales = order9_r1_safe_phase_time_scales(case.source_bucket.module_count)
    retimed = {
        phase: _retime_trajectory(trajectory, scale=scales[phase])
        for phase, trajectory in phases.items()
    }
    after = _audit(
        retimed,
        case=case,
        nominal=prepared.payload,
        geometry=geometry,
        planning_clearance_m=planning_clearance_m,
    )
    case_output = Path(arguments.case_output).resolve()
    if REPOSITORY not in case_output.parents or case_output.exists():
        raise SchemaValidationError("R1 v12 diagnostic case output is invalid")
    materialized = materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=case_output,
        repository_root=REPOSITORY,
        approved_protocol_path=runtime_v11.BASE_PROTOCOL,
    )
    initially_persisted = _load_materialized_phases(materialized)
    initially_persisted_audit = _audit(
        initially_persisted,
        case=case,
        nominal=prepared.payload,
        geometry=geometry,
        planning_clearance_m=planning_clearance_m,
    )
    completed = finalize_order9_r1_v6_materialized_case(
        materialized,
        task_spec=case.task_spec,
        phase_time_scales=scales,
        joint_rate_limit_rad_s=(
            float(screen.maximum_joint_rate_rad_s)
            + float(screen.minimum_joint_rate_margin_rad_s)
        ),
        repository_root=REPOSITORY,
    )
    regenerated = _load_materialized_phases(completed)
    regenerated_audit = _audit(
        regenerated,
        case=case,
        nominal=prepared.payload,
        geometry=geometry,
        planning_clearance_m=planning_clearance_m,
    )
    output = Path(arguments.output).resolve()
    if REPOSITORY not in output.parents:
        raise SchemaValidationError("R1 v12 diagnostic output is outside repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic_version": "order9_r1_v12_persistence_drift_diagnostic_v1",
        "candidate_id": case.candidate_id,
        "module_count": case.source_bucket.module_count,
        "planning_clearance_m": planning_clearance_m,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "phase_time_scales": scales,
        "before_timing": before,
        "after_timing": after,
        "initially_persisted": initially_persisted_audit,
        "after_legacy_regeneration": regenerated_audit,
        "before_phase_hashes": {
            phase: stable_hash(value.to_dict()) for phase, value in phases.items()
        },
        "after_phase_hashes": {
            phase: stable_hash(value.to_dict()) for phase, value in retimed.items()
        },
        "initially_persisted_phase_hashes": {
            phase: stable_hash(value.to_dict())
            for phase, value in initially_persisted.items()
        },
        "after_legacy_regeneration_phase_hashes": {
            phase: stable_hash(value.to_dict()) for phase, value in regenerated.items()
        },
        "elapsed_s": perf_counter() - started,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "before_accepted": before["accepted"],
                "after_accepted": after["accepted"],
                "after_failure": (after.get("failure_examples") or [None])[0],
                "initially_persisted_accepted": initially_persisted_audit["accepted"],
                "legacy_regeneration_accepted": regenerated_audit["accepted"],
                "legacy_regeneration_failure": (
                    regenerated_audit.get("failure_examples") or [None]
                )[0],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
