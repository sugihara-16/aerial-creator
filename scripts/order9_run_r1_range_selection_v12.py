#!/usr/bin/env python3
from __future__ import annotations

"""Run formal R1 range selection with the 19.5/30 mm teacher contract."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (  # noqa: E402
    require_order9_r1_complete_task_geometry_v12,
)
from amsrr.training.order9_r1_complete_task_materialization_v13 import (  # noqa: E402
    ORDER9_R1_PLANNING_CLEARANCE_AUDIT_FILENAME,
    finalize_order9_r1_v13_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v10 import (  # noqa: E402
    ORDER9_R1_MINIMUM_LEVEL_ID,
    ORDER9_R1_RANGE_LEVEL_IDS,
)
from amsrr.training.order9_r1_range_selection_v12 import (  # noqa: E402
    Order9R1ClearanceConstrainedTeacherScreenPipelineV12,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (  # noqa: E402
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
    require_order9_r1_support_clearance_generation_certificate,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_range_selection_v10 as runtime_v10  # noqa: E402
from scripts import order9_run_r1_range_selection_v11 as runtime_v11  # noqa: E402

RUNNER_VERSION = "order9_r1_range_selection_and_confirmation_v12"
PROTOCOL = REPOSITORY / "configs/training/order9_r1_range_selection_protocol_v12.json"
APPROVAL = REPOSITORY / "for_codex/R1_RANGE_SELECTION_PROTOCOL_V12_APPROVAL.json"
BASE_PROTOCOL = runtime_v11.BASE_PROTOCOL
OUTPUT_ROOT = REPOSITORY / "artifacts/p4_full/order9/r1_teacher/range_selection_v12"
RESULT_LEDGER_NAME = "range_selection_and_confirmation_result.json"
LEVEL_IDS = ORDER9_R1_RANGE_LEVEL_IDS
_PREPARATION_PIPELINE = None
_PREPARATION_PIPELINE_KEY = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-preparation-process-count", type=int, default=16)
    parser.add_argument("--maximum-parallel-batches", type=int, default=4)
    parser.add_argument("--maximum-wall-time-s", type=float, default=36000.0)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--preparation-only-level",
        choices=(ORDER9_R1_MINIMUM_LEVEL_ID, *LEVEL_IDS),
    )
    parser.add_argument(
        "--preparation-only-split",
        choices=("train", "validation"),
        default="train",
    )
    return parser


def _binding(path: Path) -> dict[str, str]:
    return runtime_v11._binding(path)


def _validate_binding(value: object, *, label: str) -> Path:
    return runtime_v11._validate_binding(value, label=label)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL)
    approval = _read_json(APPROVAL)
    if (
        protocol.get("protocol_version") != "order9_r1_range_selection_protocol_v12"
        or protocol.get("status") != "approved"
        or tuple(protocol.get("selection_level_ids", ())) != LEVEL_IDS
        or protocol.get("inherited_minimum_level_id") != ORDER9_R1_MINIMUM_LEVEL_ID
        or protocol.get("selection_split") != "train"
        or int(protocol.get("selection_bucket_count", -1)) != 22
        or protocol.get("confirmation_split") != "validation"
        or int(protocol.get("confirmation_bucket_count", -1)) != 14
        or float(protocol.get("acceptance_clearance_m", -1.0)) != 0.0195
        or float(protocol.get("planning_clearance_m", -1.0)) != 0.0300
        or float(protocol.get("minimum_planning_buffer_m", -1.0)) != 0.0105
        or protocol.get("all_eight_phases_generated_under_constraint") is not True
        or protocol.get("uncertified_rigid_pose_copy_rejected") is not True
        or protocol.get("reuse_pre_clearance_v11_teacher_rejection") is not True
        or protocol.get("stop_on_first_failed_level") is not True
        or protocol.get("confirmation_failure_stepback_forbidden") is not True
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_applied") is not True
        or protocol.get("local_servo_applied") is not True
        or protocol.get("formal_teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v12 formal range protocol differs")
    if (
        approval.get("record_version")
        != "order9_r1_range_selection_approval_record_v12"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path")
        != PROTOCOL.relative_to(REPOSITORY).as_posix()
        or approval.get("approved_protocol", {}).get("sha256") != hash_file(PROTOCOL)
        or approval.get("learning_authorized") is not False
        or approval.get("teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v12 formal approval differs")
    for label in (
        "base_numeric_protocol",
        "parent_minimum_level_result",
        "parent_v11_rejection",
        "source_bucket_manifest",
        "geometry_config",
        "geometry_implementation",
        "complete_task_geometry_audit",
        "support_clearance_config",
        "support_clearance_teacher",
        "teacher_pipeline",
        "materializer",
        "native_collision_implementation",
        "runner_implementation",
        "batched_wrapper_implementation",
        "batched_runtime_implementation",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
    ):
        _validate_binding(protocol.get(label), label=label)
    return protocol, approval


def _reuse_parent_v11_selection_rejection(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the deterministic L2 failure that precedes the changed v12 gate."""

    parent_path = _validate_binding(
        protocol["parent_v11_rejection"],
        label="parent_v11_rejection",
    )
    parent = _read_json(parent_path)
    selection = parent.get("selection", {})
    source_binding = parent.get("bindings", {}).get("selection_result")
    source_path = _validate_binding(source_binding, label="v11_selection_result")
    source = _read_json(source_path)
    if (
        parent.get("decision") != "rejected_before_teacher_collection"
        or selection.get("first_attempted_level_id") != LEVEL_IDS[0]
        or selection.get("passed") is not False
        or selection.get("isaac_invoked") is not False
        or selection.get("controller_layers_invoked") is not False
        or "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM"
        not in str(selection.get("decisive_reason", ""))
        or source.get("level_id") != LEVEL_IDS[0]
        or source.get("split") != "train"
        or source.get("gate", {}).get("passed") is not False
        or source.get("isaac_invoked") is not False
    ):
        raise SchemaValidationError("R1 v11 pre-clearance rejection differs")
    return {
        "level_id": LEVEL_IDS[0],
        "split": "train",
        "passed": False,
        "gate": source["gate"],
        "result": dict(source_binding),
        "reused_parent_evidence": True,
        "rejection_precedes_support_clearance_candidate_selection": True,
        "decisive_candidate_id": selection["decisive_candidate_id"],
        "decisive_reason": selection["decisive_reason"],
        "isaac_invoked": False,
        "controller_layers_invoked": False,
    }


def _validate_prepared_case(case, destination: Path) -> dict[str, Any]:
    record = runtime_v10._validate_prepared_case(case, destination)
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    planning_path = destination / ORDER9_R1_PLANNING_CLEARANCE_AUDIT_FILENAME
    certificate_path = destination / "support_clearance_generation_certificate_v13.json"
    planning = _read_json(planning_path)
    certificate = _read_json(certificate_path)
    require_order9_r1_complete_task_geometry_v12(planning)
    require_order9_r1_support_clearance_generation_certificate(
        certificate,
        clearance_contract=clearance,
        candidate_id=case.candidate_id,
    )
    if (
        planning.get("candidate_id") != case.candidate_id
        or float(planning.get("planning_clearance_m", -1.0)) != 0.0300
        or planning.get("generation_time_clearance_constraint_applied") is not True
        or certificate.get("persisted_artifact_checked") is not True
    ):
        raise SchemaValidationError("R1 v12 persisted planning proof differs")
    evidence = dict(record["preparation_evidence"])
    evidence.update(
        {
            "complete_task_planning_clearance_audit_v13": _binding(planning_path),
            "support_clearance_generation_certificate_v13": _binding(certificate_path),
        }
    )
    return {
        **record,
        "fast_screen_passed": True,
        "pre_isaac_geometry_passed": True,
        "preparation_evidence": evidence,
    }


def _write_rejection(case, output_root: Path, pipeline) -> dict[str, str]:
    audits = pipeline.clearance_candidate_audits.get(case.candidate_id)
    if not audits:
        return {}
    path = runtime_v10._atomic_json(
        output_root
        / "rejections"
        / case.level_id
        / f"{case.candidate_id}__support_clearance_v12.json",
        {
            "record_version": "order9_r1_support_clearance_rejection_v12",
            "candidate_id": case.candidate_id,
            "accepted": False,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "candidate_audits": list(audits),
        },
    )
    return {"support_clearance_rejection_v12": _binding(path)}


def _prepare_case(arguments) -> dict[str, Any]:
    case, output_root_text = arguments
    output_root = Path(output_root_text)
    destination = output_root / "prepared" / case.level_id / case.candidate_id
    if destination.is_dir():
        try:
            return _validate_prepared_case(case, destination)
        except (OSError, RuntimeError, ValueError, SchemaValidationError):
            shutil.rmtree(destination)

    base = load_order9_r1_calibration_protocol(
        BASE_PROTOCOL,
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
    global _PREPARATION_PIPELINE, _PREPARATION_PIPELINE_KEY
    pipeline_key = (
        base.source_bucket_manifest.sha256,
        base.minimum_normalized_joint_limit_reserve,
        base.maximum_body_tilt_rad,
        geometry.config_sha256,
        clearance.config_sha256,
    )
    if _PREPARATION_PIPELINE_KEY != pipeline_key:
        _PREPARATION_PIPELINE = Order9R1ClearanceConstrainedTeacherScreenPipelineV12(
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
        _PREPARATION_PIPELINE_KEY = pipeline_key

    record = {
        "candidate_id": case.candidate_id,
        "source_bucket_id": case.source_bucket.bucket_id,
        "module_count": case.source_bucket.module_count,
        "sample_kind": case.sample_kind,
        "sample_index": case.sample_index,
        "teacher_feasible": False,
        "fast_screen_passed": False,
        "screen_passed": False,
        "pre_isaac_geometry_passed": False,
        "prepared": False,
        "preparation_failure_reason": None,
        "preparation_evidence": {},
    }
    try:
        prepared = _PREPARATION_PIPELINE.prepare(case)
        prepared.validate_for(case)
        if not prepared.teacher_trajectory_complete:
            record["preparation_failure_reason"] = prepared.failure_reason
            record["preparation_evidence"] = _write_rejection(
                case, output_root, _PREPARATION_PIPELINE
            )
            return record
        record["teacher_feasible"] = True
        screen = _PREPARATION_PIPELINE.screen(case, prepared)
        record["screen_passed"] = bool(screen.accepted)
        if not screen.accepted or not screen.eligible_for_full_control_test:
            record["preparation_failure_reason"] = ",".join(screen.violation_codes)
            return record
        record["fast_screen_passed"] = True
        materialized = materialize_order9_r1_isaac_case(
            case=case,
            prepared=prepared,
            screen=screen,
            output_dir=destination,
            repository_root=REPOSITORY,
            approved_protocol_path=BASE_PROTOCOL,
        )
        certificate = _PREPARATION_PIPELINE.clearance_generation_certificates.get(
            case.candidate_id
        )
        finalize_order9_r1_v13_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=runtime_v11.order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=REPOSITORY,
            geometry_contract=geometry,
            clearance_contract=clearance,
            clearance_generation_certificate=certificate,
        )
        return _validate_prepared_case(case, destination)
    except (OSError, RuntimeError, ValueError, SchemaValidationError) as error:
        evidence = _write_rejection(case, output_root, _PREPARATION_PIPELINE)
        if destination.exists():
            shutil.rmtree(destination)
        record["preparation_failure_reason"] = f"{type(error).__name__}: {error}"
        record["preparation_evidence"] = evidence
        return record


def _install_v12_preparation_hooks() -> None:
    runtime_v11._prepare_case = _prepare_case
    runtime_v11._validate_prepared_case = _validate_prepared_case
    runtime_v11.RUNNER_VERSION = RUNNER_VERSION
    runtime_v10.RUNNER_VERSION = RUNNER_VERSION


def main() -> int:
    runtime_v10._ensure_python_hash_seed_zero()
    args = _parser().parse_args()
    if (
        not 1 <= args.maximum_preparation_process_count <= 24
        or not 1 <= args.maximum_parallel_batches <= 4
        or args.maximum_wall_time_s <= 0.0
        or args.rollout_steps < 1
    ):
        raise ValueError("R1 v12 runtime limits are invalid")
    protocol, approval = _load_contract()
    _install_v12_preparation_hooks()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = load_order9_r1_calibration_protocol(
        BASE_PROTOCOL,
        repository_root=REPOSITORY,
    )
    started = time.time()
    deadline = started + min(
        float(args.maximum_wall_time_s),
        float(protocol["maximum_wall_time_s"]),
    )
    if args.preparation_only_level:
        runtime_v11._prepare_level(
            args.preparation_only_level,
            args.preparation_only_split,
            output_root=output_root,
            maximum_process_count=args.maximum_preparation_process_count,
        )
        return 0

    selection_results = [_reuse_parent_v11_selection_rejection(protocol)]
    selected_level = ORDER9_R1_MINIMUM_LEVEL_ID
    confirmation = None
    try:
        confirmation, _path = runtime_v11._run_level(
            level_id=selected_level,
            split="validation",
            output_root=output_root,
            base=base,
            args=args,
            deadline=deadline,
        )
        status = "accepted" if confirmation["passed"] else "rejected"
    except TimeoutError as error:
        runtime_v10._atomic_json(
            output_root / "incomplete_timeout.json",
            {
                "runner_version": RUNNER_VERSION,
                "status": "incomplete_timeout",
                "selected_level_before_timeout": selected_level,
                "selection_results": selection_results,
                "confirmation_result": confirmation,
                "elapsed_s": time.time() - started,
                "error": str(error),
            },
        )
        return 2

    ledger = {
        "ledger_version": "order9_r1_range_selection_result_ledger_v12",
        "runner_version": RUNNER_VERSION,
        "status": status,
        "selected_level_id": selected_level,
        "accepted_level_id": selected_level if status == "accepted" else None,
        "selection_split": "train",
        "selection_bucket_count": 22,
        "confirmation_split": "validation",
        "confirmation_bucket_count": 14,
        "selection_results": selection_results,
        "confirmation_result": confirmation,
        "acceptance_clearance_m": 0.0195,
        "planning_clearance_m": 0.0300,
        "minimum_planning_buffer_m": 0.0105,
        "generation_certificate_required": True,
        "confirmation_failure_stepback_forbidden": True,
        "formal_teacher_collection_authorized": False,
        "training_eligible": False,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "elapsed_s": time.time() - started,
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "approval": _binding(APPROVAL),
            "base_numeric_protocol": _binding(BASE_PROTOCOL),
            "parent_minimum_level_result": protocol["parent_minimum_level_result"],
            "parent_v11_rejection": protocol["parent_v11_rejection"],
            "support_clearance_config": _binding(
                REPOSITORY / ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE
            ),
            "support_clearance_teacher": _binding(
                REPOSITORY / "amsrr/training/order9_r1_support_clearance_teacher_v12.py"
            ),
            "teacher_pipeline": _binding(
                REPOSITORY / "amsrr/training/order9_r1_range_selection_v12.py"
            ),
            "materializer": _binding(
                REPOSITORY
                / "amsrr/training/order9_r1_complete_task_materialization_v13.py"
            ),
            "runner_implementation": _binding(Path(__file__)),
            "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
            "protected_c3_rollout": protocol["protected_c3_rollout"],
            "protected_c3_curriculum": protocol["protected_c3_curriculum"],
            "protected_c3_release_ledger": protocol["protected_c3_release_ledger"],
        },
        "approval_record_version": approval["record_version"],
    }
    result_path = runtime_v10._atomic_json(
        output_root / RESULT_LEDGER_NAME,
        ledger,
    )
    print(
        "ORDER9_R1_V12_COMPLETE="
        + json.dumps(
            {
                "status": status,
                "selected_level_id": selected_level,
                "result_ledger": str(result_path),
                "result_sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
