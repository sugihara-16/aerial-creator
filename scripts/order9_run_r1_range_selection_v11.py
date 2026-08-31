#!/usr/bin/env python3
from __future__ import annotations

"""Select and confirm the R1 range with the corrected v11 nominal geometry."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training import order9_c3_nominal_trajectory as c3_nominal  # noqa: E402
from amsrr.training import order9_c3_teacher as c3_teacher  # noqa: E402
from amsrr.training.order9_configuration_space_planner import (  # noqa: E402
    order9_lightweight_configuration_planning,
)
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (  # noqa: E402
    ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME,
    require_order9_r1_complete_task_geometry_v12,
)
from amsrr.training.order9_r1_complete_task_materialization_v12 import (  # noqa: E402
    finalize_order9_r1_v12_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    build_order9_r1_explicit_support_collision_object_v11,
    build_order9_r1_geometry_variants_v11,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v10 import (  # noqa: E402
    ORDER9_R1_MINIMUM_LEVEL_ID,
    ORDER9_R1_RANGE_LEVEL_IDS,
    evaluate_range_level_gate,
    range_case_priority,
)
from amsrr.training.order9_r1_range_selection_v11 import (  # noqa: E402
    Order9R1RangeTeacherScreenPipelineV11,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (  # noqa: E402
    Order9R1RotationalTeacherScreenPipelineV7,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_range_selection_v10 as runtime_v10  # noqa: E402

RUNNER_VERSION = "order9_r1_range_selection_and_confirmation_v11"
PROTOCOL = REPOSITORY / "configs/training/order9_r1_range_selection_protocol_v11.json"
APPROVAL = REPOSITORY / "for_codex/R1_RANGE_SELECTION_PROTOCOL_V11_APPROVAL.json"
TEACHER_REPAIRS = (
    REPOSITORY / "configs/training/order9_r1_range_teacher_repairs_v11.json"
)
BASE_PROTOCOL = REPOSITORY / "configs/training/order9_r1_calibration_protocol_v1.yaml"
OUTPUT_ROOT = REPOSITORY / "artifacts/p4_full/order9/r1_teacher/range_selection_v11"
RESULT_LEDGER_NAME = "range_selection_and_confirmation_result.json"
LEVEL_IDS = ORDER9_R1_RANGE_LEVEL_IDS
EXPECTED_BUCKET_COUNTS = {"train": 22, "validation": 14}
SAMPLES_PER_BUCKET = 31
_PREPARATION_PIPELINE = None
_PREPARATION_PIPELINE_KEY = None


class Order9R1FormalRangeTeacherScreenPipelineV11(
    Order9R1RangeTeacherScreenPipelineV11
):
    """Also rebuild level one after the nominal task geometry changed."""

    def prepare(self, case):
        if case.level_id != ORDER9_R1_MINIMUM_LEVEL_ID:
            return super().prepare(case)
        collision = build_order9_r1_explicit_support_collision_object_v11(
            case.task_spec,
            contract=self.geometry_contract,
        )
        original_offset = c3_teacher._offset_horizontal_grasp_candidates

        def raised_offset(candidate_set, *, height_offset_m, tangent_offset_world_m):
            repair = _load_teacher_repairs().get(case.candidate_id)
            repair_offset = (
                None if repair is None else repair["grasp_contact_height_offset_m"]
            )
            return original_offset(
                candidate_set,
                height_offset_m=max(
                    float(height_offset_m),
                    (
                        self.geometry_contract.grasp_contact_height_offset_m
                        if repair_offset is None
                        else repair_offset
                    ),
                ),
                tangent_offset_world_m=tangent_offset_world_m,
            )

        with patch.object(
            c3_nominal,
            "build_order9_c3_posture_collision_object",
            return_value=collision,
        ), patch.object(
            c3_teacher,
            "_offset_horizontal_grasp_candidates",
            side_effect=raised_offset,
        ), order9_lightweight_configuration_planning():
            hints = self.teacher_fast_paths.get(case.candidate_id, ())
            previous = self.teacher_candidate_overrides.get(case.candidate_id)
            previous_strict = self.strict_preferred_candidate_options
            try:
                for option in hints:
                    self.teacher_candidate_overrides[case.candidate_id] = (option,)
                    self.strict_preferred_candidate_options = True
                    prepared = Order9R1RotationalTeacherScreenPipelineV7.prepare(
                        self,
                        case,
                    )
                    if prepared.teacher_trajectory_complete:
                        return prepared
            finally:
                self.strict_preferred_candidate_options = previous_strict
                if previous is None:
                    self.teacher_candidate_overrides.pop(case.candidate_id, None)
                else:
                    self.teacher_candidate_overrides[case.candidate_id] = previous
            return Order9R1RotationalTeacherScreenPipelineV7.prepare(self, case)


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
        help="Run only the pre-Isaac v11 admission for one level.",
    )
    parser.add_argument(
        "--preparation-only-split",
        choices=("train", "validation"),
        default="train",
    )
    return parser


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": source.relative_to(REPOSITORY).as_posix(),
        "sha256": hash_file(source),
    }


def _validate_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 v11 binding is missing: {label}")
    path = (REPOSITORY / str(value.get("path", ""))).resolve()
    if (
        REPOSITORY not in path.parents
        or not path.is_file()
        or value.get("sha256") != hash_file(path)
    ):
        raise SchemaValidationError(f"R1 v11 binding changed: {label}")
    return path


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version") != "order9_r1_range_selection_protocol_v11"
        or protocol.get("status") != "approved"
        or tuple(protocol.get("selection_level_ids", ())) != LEVEL_IDS
        or protocol.get("selection_split") != "train"
        or int(protocol.get("selection_bucket_count", -1)) != 22
        or protocol.get("confirmation_split") != "validation"
        or int(protocol.get("confirmation_bucket_count", -1)) != 14
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_applied") is not True
        or protocol.get("local_servo_applied") is not True
        or protocol.get("full_eight_phase_geometry_admission_required") is not True
        or abs(float(protocol.get("anchor_position_tolerance_m", -1.0)) - 0.030)
        > 1.0e-12
        or abs(
            float(protocol.get("anchor_attitude_tolerance_rad", -1.0))
            - 0.05235987755982989
        )
        > 1.0e-12
        or protocol.get("stop_on_first_failed_level") is not True
        or protocol.get("confirmation_failure_stepback_forbidden") is not True
        or protocol.get("formal_teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v11 range protocol differs")
    if (
        approval.get("record_version")
        != "order9_r1_range_selection_approval_record_v11"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path")
        != PROTOCOL.relative_to(REPOSITORY).as_posix()
        or approval.get("approved_protocol", {}).get("sha256") != hash_file(PROTOCOL)
    ):
        raise SchemaValidationError("R1 v11 range approval differs")
    for label in (
        "base_numeric_protocol",
        "parent_minimum_level_result",
        "source_bucket_manifest",
        "geometry_config",
        "geometry_implementation",
        "teacher_pipeline",
        "complete_task_geometry_audit",
        "materializer",
        "native_collision_implementation",
        "teacher_repair_config",
        "runner_implementation",
        "batched_wrapper_implementation",
        "batched_runtime_implementation",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
    ):
        _validate_binding(protocol.get(label), label=label)
    parent_path = _validate_binding(
        protocol.get("parent_minimum_level_result"),
        label="parent_minimum_level_result",
    )
    parent = _read_json(parent_path)
    if (
        parent.get("decision") != "accepted_minimum_level"
        or parent.get("level_id") != ORDER9_R1_MINIMUM_LEVEL_ID
        or int(parent.get("candidate_count", -1)) != 682
        or int(parent.get("episode_count", -1)) != 1364
        or int(parent.get("success_count", -1)) != 1364
        or int(parent.get("safety_failure_count", -1)) != 0
        or int(parent.get("fallback_count", -1)) != 0
    ):
        raise SchemaValidationError("R1 v11 inherited minimum result differs")
    return protocol, approval


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 v11 JSON object expected: {path}")
    return value


def _load_teacher_repairs() -> dict[str, dict[str, float]]:
    payload = _read_json(TEACHER_REPAIRS)
    values = payload.get("candidate_contact_height_repairs")
    maximum = float(payload.get("maximum_contact_height_offset_m", -1.0))
    if (
        payload.get("repair_version") != "order9_r1_range_teacher_repairs_v11"
        or not math.isclose(maximum, 0.030, abs_tol=1.0e-12)
        or not isinstance(values, dict)
    ):
        raise SchemaValidationError("R1 v11 teacher-repair config differs")
    result = {}
    for candidate_id, repair in values.items():
        if not isinstance(candidate_id, str) or not isinstance(repair, dict):
            raise SchemaValidationError("R1 v11 teacher-repair entry differs")
        source = (
            REPOSITORY / str(repair.get("source_decisive_checkpoint_path", ""))
        ).resolve()
        offset = float(repair.get("grasp_contact_height_offset_m", -1.0))
        height_increment = float(repair.get("nominal_object_height_increment_m", -1.0))
        observed = float(repair.get("observed_robot_support_clearance_m", -1.0))
        required = float(repair.get("required_robot_support_clearance_m", -1.0))
        repair_required = float(
            repair.get("repair_required_robot_support_clearance_m", -1.0)
        )
        if (
            REPOSITORY not in source.parents
            or not source.is_file()
            or hash_file(source) != repair.get("source_decisive_checkpoint_sha256")
            or repair.get("failure_phase") != "place"
            or int(repair.get("failure_knot_index", -1)) < 0
            or not 0.0 < offset <= maximum
            or not 0.04 <= height_increment <= 0.08
            or not 0.0 < observed < required
            or not 0.0 < repair_required <= observed
            or not isinstance(repair.get("reason"), str)
            or not repair["reason"]
        ):
            raise SchemaValidationError("R1 v11 teacher-repair evidence differs")
        source_payload = _read_json(source)
        decisive = source_payload.get("decisive_record", {})
        if (
            decisive.get("candidate_id") != candidate_id
            or decisive.get("prepared") is not False
            or "support_or_self"
            not in str(decisive.get("preparation_failure_reason", ""))
        ):
            raise SchemaValidationError("R1 v11 teacher-repair source differs")
        result[candidate_id] = {
            "grasp_contact_height_offset_m": offset,
            "nominal_object_height_increment_m": height_increment,
            "required_robot_support_clearance_m": repair_required,
        }
    return result


def _corrected_cases(level_id: str, split: str):
    contract = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    source = enumerate_order9_r1_calibration_level_cases(
        protocol_path=BASE_PROTOCOL,
        repository_root=REPOSITORY,
        level_id=level_id,
        split=split,
    )
    repairs = _load_teacher_repairs()
    result = []
    for case in source:
        repair = repairs.get(case.candidate_id)
        case_contract = (
            contract
            if repair is None
            else replace(
                contract,
                object_height_increment_m=repair["nominal_object_height_increment_m"],
            )
        )
        result.append(
            build_order9_r1_geometry_variants_v11(
                case,
                contract=case_contract,
            ).nominal_case
        )
    return tuple(result)


def _validate_prepared_case(case, destination: Path) -> dict[str, Any]:
    record = runtime_v10._validate_prepared_case(case, destination)
    geometry_path = destination / ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME
    geometry = _read_json(geometry_path)
    require_order9_r1_complete_task_geometry_v12(geometry)
    if (
        geometry.get("candidate_id") != case.candidate_id
        or geometry.get("source_bucket_id") != case.source_bucket.bucket_id
        or geometry.get("isaac_invoked") is not False
        or geometry.get("controller_layers_invoked") is not False
    ):
        raise SchemaValidationError("R1 v11 geometry evidence identity differs")
    evidence = dict(record["preparation_evidence"])
    evidence["complete_task_geometry_audit_v12"] = _binding(geometry_path)
    if case.candidate_id in _load_teacher_repairs():
        evidence["teacher_repair_config_v11"] = _binding(TEACHER_REPAIRS)
    return {
        **record,
        "fast_screen_passed": True,
        "pre_isaac_geometry_passed": True,
        "preparation_evidence": evidence,
    }


def _rejection_record(destination: Path, error: BaseException) -> Path | None:
    geometry_path = destination / ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME
    if not geometry_path.is_file():
        return None
    geometry = _read_json(geometry_path)
    rejection_path = (
        destination.parents[2]
        / "rejections"
        / destination.parent.name
        / f"{destination.name}.json"
    )
    return runtime_v10._atomic_json(
        rejection_path,
        {
            "record_version": "order9_r1_pre_isaac_geometry_rejection_v11",
            "candidate_id": destination.name,
            "accepted": False,
            "error": f"{type(error).__name__}: {error}",
            "geometry_audit": geometry,
        },
    )


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
    contract = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    repair = _load_teacher_repairs().get(case.candidate_id)
    materialization_contract = (
        contract
        if repair is None
        else replace(
            contract,
            required_robot_support_clearance_m=repair[
                "required_robot_support_clearance_m"
            ],
        )
    )
    global _PREPARATION_PIPELINE, _PREPARATION_PIPELINE_KEY
    pipeline_key = (
        base.source_bucket_manifest.sha256,
        base.minimum_normalized_joint_limit_reserve,
        base.maximum_body_tilt_rad,
        contract.config_sha256,
    )
    if _PREPARATION_PIPELINE_KEY != pipeline_key:
        _PREPARATION_PIPELINE = Order9R1FormalRangeTeacherScreenPipelineV11(
            repository_root=REPOSITORY,
            source_bucket_manifest_path=(REPOSITORY / base.source_bucket_manifest.path),
            minimum_normalized_joint_limit_reserve=(
                base.minimum_normalized_joint_limit_reserve
            ),
            maximum_body_tilt_rad=base.maximum_body_tilt_rad,
            anchor_position_tolerance_m=0.030,
            enforce_joint_limit_reserve_during_ik=True,
            geometry_contract=contract,
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
            return record
        record["teacher_feasible"] = True
        screen = _PREPARATION_PIPELINE.screen(case, prepared)
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
        finalize_order9_r1_v12_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=REPOSITORY,
            geometry_contract=materialization_contract,
        )
        return _validate_prepared_case(case, destination)
    except (OSError, RuntimeError, ValueError, SchemaValidationError) as error:
        rejection_path = _rejection_record(destination, error)
        if destination.exists():
            shutil.rmtree(destination)
        record["preparation_failure_reason"] = f"{type(error).__name__}: {error}"
        if rejection_path is not None:
            record["preparation_evidence"] = {
                "pre_isaac_geometry_rejection": _binding(rejection_path)
            }
        return record


def _unattempted_record(case, reason: str) -> dict[str, Any]:
    return {
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
        "preparation_failure_reason": reason,
        "preparation_evidence": {},
    }


def _prepare_level(
    level_id: str,
    split: str,
    *,
    output_root: Path,
    maximum_process_count: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    cases = list(_corrected_cases(level_id, split))
    expected = EXPECTED_BUCKET_COUNTS[split] * SAMPLES_PER_BUCKET
    if len(cases) != expected:
        raise SchemaValidationError("R1 v11 candidate count differs")
    cases.sort(
        key=lambda case: range_case_priority(
            case.candidate_id,
            case.sample_kind,
            case.sample_index,
        )
    )
    records: dict[str, dict[str, Any]] = {}
    decisive = False
    teacher_repairs = _load_teacher_repairs()
    decisive_path = (
        output_root / "preparation" / f"{split}__{level_id}__decisive_in_progress.json"
    )
    if decisive_path.is_file():
        cached = _read_json(decisive_path)
        record = cached.get("decisive_record")
        case_by_id = {case.candidate_id: case for case in cases}
        candidate_id = record.get("candidate_id") if isinstance(record, dict) else None
        case = case_by_id.get(candidate_id)
        if candidate_id in teacher_repairs:
            print(
                "ORDER9_R1_V11_DECISIVE_PREPARATION_SUPERSEDED="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "split": split,
                        "candidate_id": candidate_id,
                        "teacher_repair_config": str(TEACHER_REPAIRS),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            if (
                cached.get("record_version") != "order9_r1_range_decisive_pre_isaac_v11"
                or cached.get("level_id") != level_id
                or cached.get("split") != split
                or cached.get("isaac_invoked") is not False
                or cached.get("controller_layers_invoked") is not False
                or case is None
                or case.sample_kind != "lattice"
                or record.get("source_bucket_id") != case.source_bucket.bucket_id
                or record.get("sample_kind") != case.sample_kind
                or int(record.get("sample_index", -1)) != case.sample_index
                or record.get("prepared") is not False
                or not record.get("preparation_failure_reason")
            ):
                raise SchemaValidationError(
                    "R1 v11 decisive pre-Isaac checkpoint differs"
                )
            decisive_record = dict(record)
            evidence = dict(decisive_record.get("preparation_evidence", {}))
            evidence["decisive_pre_isaac_checkpoint"] = _binding(decisive_path)
            decisive_record["preparation_evidence"] = evidence
            records[case.candidate_id] = decisive_record
            decisive = True
            print(
                "ORDER9_R1_V11_DECISIVE_PREPARATION_REUSED="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "split": split,
                        "candidate_id": case.candidate_id,
                        "checkpoint": str(decisive_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if decisive:
        for case in cases:
            records.setdefault(
                case.candidate_id,
                _unattempted_record(
                    case, "not attempted after decisive pre-Isaac rejection"
                ),
            )
        ordered = [records[case.candidate_id] for case in cases]
        path = runtime_v10._atomic_json(
            output_root / "preparation" / f"{split}__{level_id}.json",
            {
                "record_version": "order9_r1_range_preparation_v11",
                "level_id": level_id,
                "split": split,
                "candidate_count": len(ordered),
                "attempted_count": 1,
                "prepared_count": 0,
                "decisive_pre_isaac_rejection": True,
                "isaac_invoked": False,
                "controller_layers_invoked": False,
                "entries": ordered,
            },
        )
        print(f"ORDER9_R1_V11_PREPARATION_RESULT={path}", flush=True)
        return cases, ordered
    arguments = [(case, str(output_root)) for case in cases]
    maximum_pending = 2 * maximum_process_count
    next_index = 0
    with ProcessPoolExecutor(max_workers=maximum_process_count) as executor:
        futures = {}

        def fill_pending() -> None:
            nonlocal next_index
            while next_index < len(arguments) and len(futures) < maximum_pending:
                value = arguments[next_index]
                next_index += 1
                futures[executor.submit(_prepare_case, value)] = value[0]

        fill_pending()
        last_reported = 0
        while futures:
            completed, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                case = futures.pop(future)
                record = future.result()
                records[case.candidate_id] = record
                if case.sample_kind == "lattice" and not record["prepared"]:
                    decisive = True
                    runtime_v10._atomic_json(
                        output_root
                        / "preparation"
                        / f"{split}__{level_id}__decisive_in_progress.json",
                        {
                            "record_version": (
                                "order9_r1_range_decisive_pre_isaac_v11"
                            ),
                            "level_id": level_id,
                            "split": split,
                            "completed_count_at_detection": len(records),
                            "isaac_invoked": False,
                            "controller_layers_invoked": False,
                            "decisive_record": record,
                        },
                    )
            if decisive:
                for future in futures:
                    future.cancel()
                break
            fill_pending()
            if len(records) - last_reported >= maximum_process_count or not futures:
                last_reported = len(records)
                print(
                    "ORDER9_R1_V11_PREPARATION_PROGRESS="
                    + json.dumps(
                        {
                            "level_id": level_id,
                            "split": split,
                            "completed": len(records),
                            "total": len(cases),
                            "admitted": sum(
                                bool(value["prepared"]) for value in records.values()
                            ),
                            "decisive_pre_isaac_rejection": decisive,
                            "maximum_pending": maximum_pending,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if decisive:
            print(
                "ORDER9_R1_V11_PREPARATION_PROGRESS="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "split": split,
                        "completed": len(records),
                        "total": len(cases),
                        "admitted": sum(
                            bool(value["prepared"]) for value in records.values()
                        ),
                        "decisive_pre_isaac_rejection": True,
                        "maximum_pending": maximum_pending,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    for case in cases:
        records.setdefault(
            case.candidate_id,
            _unattempted_record(
                case, "not attempted after decisive pre-Isaac rejection"
            ),
        )
    ordered = [records[case.candidate_id] for case in cases]
    path = runtime_v10._atomic_json(
        output_root / "preparation" / f"{split}__{level_id}.json",
        {
            "record_version": "order9_r1_range_preparation_v11",
            "level_id": level_id,
            "split": split,
            "candidate_count": len(ordered),
            "attempted_count": sum(
                value["preparation_failure_reason"]
                != "not attempted after decisive pre-Isaac rejection"
                for value in ordered
            ),
            "prepared_count": sum(bool(value["prepared"]) for value in ordered),
            "decisive_pre_isaac_rejection": decisive,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "entries": ordered,
        },
    )
    print(f"ORDER9_R1_V11_PREPARATION_RESULT={path}", flush=True)
    return cases, ordered


def _pre_isaac_gate_failed(records: Sequence[Mapping[str, Any]]) -> bool:
    lattice = [value for value in records if value["sample_kind"] == "lattice"]
    interior = [value for value in records if value["sample_kind"] == "interior"]

    def rate(key: str, values: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(value[key]) for value in values) / len(values)

    return bool(
        not all(value["prepared"] for value in lattice)
        or rate("teacher_feasible", records) < 0.95
        or rate("screen_passed", records) < 0.95
        or rate("teacher_feasible", interior) < 0.95
        or rate("screen_passed", interior) < 0.95
    )


def _entries_without_isaac(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(record),
            "valid_evidence": False,
            "candidate_passed": False,
            "episode_count": 0,
            "success_count": 0,
            "safety_failure_count": 0,
            "fallback_count": 0,
            "failure_reason": (
                record.get("preparation_failure_reason")
                or "Isaac skipped after decisive pre-Isaac gate failure"
            ),
        }
        for record in records
    ]


def _gate(base, level_id: str, entries: Sequence[Mapping[str, Any]]):
    gate_level_id = (
        ORDER9_R1_RANGE_LEVEL_IDS[0]
        if level_id == ORDER9_R1_MINIMUM_LEVEL_ID
        else level_id
    )
    result = evaluate_range_level_gate(
        level_id=gate_level_id,
        entries=entries,
        minimum_lattice_pass_rate=base.minimum_lattice_pass_rate,
        minimum_teacher_feasibility_rate=base.minimum_teacher_feasibility_rate,
        minimum_fast_screen_pass_rate=base.minimum_fast_screen_pass_rate,
        minimum_isaac_success_rate=base.minimum_isaac_success_rate,
        minimum_interior_teacher_feasibility_rate=(
            base.minimum_interior_teacher_feasibility_rate
        ),
        minimum_interior_fast_screen_pass_rate=(
            base.minimum_interior_fast_screen_pass_rate
        ),
        minimum_interior_isaac_success_rate=(base.minimum_interior_isaac_success_rate),
        maximum_safety_failure_count=base.maximum_safety_failure_count,
        maximum_fallback_rate=base.maximum_fallback_rate,
    )
    return result if gate_level_id == level_id else replace(result, level_id=level_id)


def _run_level(
    *,
    level_id: str,
    split: str,
    output_root: Path,
    base,
    args,
    deadline: float,
) -> tuple[dict[str, Any], Path]:
    cases, preparation = _prepare_level(
        level_id,
        split,
        output_root=output_root,
        maximum_process_count=args.maximum_preparation_process_count,
    )
    if _pre_isaac_gate_failed(preparation):
        entries = _entries_without_isaac(preparation)
        isaac_invoked = False
    else:
        entries = runtime_v10._execute_level(
            level_id,
            cases=cases,
            preparation_records=preparation,
            output_root=output_root,
            isaac_python=args.isaac_python,
            rollout_steps=args.rollout_steps,
            maximum_parallel_batches=args.maximum_parallel_batches,
            deadline=deadline,
        )
        isaac_invoked = True
    gate = _gate(base, level_id, entries)
    path = runtime_v10._atomic_json(
        output_root / "results" / f"{split}__{level_id}.json",
        {
            "result_version": "order9_r1_range_level_result_v11",
            "runner_version": RUNNER_VERSION,
            "level_id": level_id,
            "split": split,
            "gate": gate.to_dict(),
            "candidate_count": len(entries),
            "isaac_invoked": isaac_invoked,
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "training_eligible": False,
            "entries": entries,
        },
    )
    return {
        "level_id": level_id,
        "split": split,
        "passed": gate.passed,
        "gate": gate.to_dict(),
        "result": _binding(path),
    }, path


def main() -> int:
    runtime_v10._ensure_python_hash_seed_zero()
    args = _parser().parse_args()
    if (
        not 1 <= args.maximum_preparation_process_count <= 24
        or not 1 <= args.maximum_parallel_batches <= 4
        or args.maximum_wall_time_s <= 0.0
        or args.rollout_steps < 1
    ):
        raise ValueError("R1 v11 runtime limits are invalid")
    protocol, approval = _load_contract()
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
    runtime_v10.RUNNER_VERSION = RUNNER_VERSION
    if args.preparation_only_level:
        _prepare_level(
            args.preparation_only_level,
            args.preparation_only_split,
            output_root=output_root,
            maximum_process_count=args.maximum_preparation_process_count,
        )
        return 0

    selection_results = []
    selected_level = ORDER9_R1_MINIMUM_LEVEL_ID
    confirmation = None
    status = "incomplete"
    try:
        for level_id in LEVEL_IDS:
            if time.time() >= deadline:
                raise TimeoutError("R1 v11 exceeded its wall-time bound")
            result, _path = _run_level(
                level_id=level_id,
                split="train",
                output_root=output_root,
                base=base,
                args=args,
                deadline=deadline,
            )
            selection_results.append(result)
            if not result["passed"]:
                break
            selected_level = level_id
        if selected_level is not None:
            confirmation, _path = _run_level(
                level_id=selected_level,
                split="validation",
                output_root=output_root,
                base=base,
                args=args,
                deadline=deadline,
            )
        status = (
            "accepted"
            if confirmation is not None and confirmation["passed"]
            else "rejected"
        )
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
        "ledger_version": "order9_r1_range_selection_result_ledger_v11",
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
        "confirmation_failure_stepback_forbidden": True,
        "full_eight_phase_geometry_admission_required": True,
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
            "geometry_config": _binding(
                REPOSITORY / ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE
            ),
            "teacher_repair_config": _binding(TEACHER_REPAIRS),
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
        "ORDER9_R1_V11_COMPLETE="
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
