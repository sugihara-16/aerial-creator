from __future__ import annotations

"""R1 range-selection helpers layered on the accepted minimum-level path."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_configuration_space_planner import (
    order9_lightweight_configuration_planning,
)
from amsrr.training.order9_r1_calibration_runner import Order9R1CalibrationCase
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (
    Order9R1RotationalTeacherScreenPipelineV7,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_RANGE_SELECTION_V10_VERSION = "order9_r1_range_selection_v10"
ORDER9_R1_RANGE_LEVEL_IDS = (
    "r1_l2_20mm_10deg",
    "r1_l3_30mm_15deg",
    "r1_l4_40mm_20deg",
)
ORDER9_R1_MINIMUM_LEVEL_ID = "r1_l1_10mm_5deg"
ORDER9_R1_MAXIMUM_CANDIDATES_PER_SCENE = 4
ORDER9_R1_RANGE_TEACHER_HINTS_RELATIVE = Path(
    "configs/training/order9_r1_range_teacher_hints_v10.json"
)
Order9R1TeacherHint = (
    tuple[tuple[int, int], str, float, float]
    | tuple[tuple[int, int], str, float, float, float]
    | tuple[
        tuple[int, int],
        str,
        float,
        float,
        float,
        tuple[float, float, float],
    ]
)


def corresponding_minimum_level_candidate_id(candidate_id: str) -> str:
    """Map one higher-level sample to the same source/sample at level one."""

    parts = str(candidate_id).split("__")
    if len(parts) != 4 or parts[0] not in ORDER9_R1_RANGE_LEVEL_IDS:
        raise SchemaValidationError("R1 range candidate identity is invalid")
    return "__".join((ORDER9_R1_MINIMUM_LEVEL_ID, *parts[1:]))


class Order9R1RangeTeacherScreenPipelineV10(Order9R1RotationalTeacherScreenPipelineV7):
    """Reuse minimum-level contact preferences as non-strict higher-level hints."""

    pipeline_version = ORDER9_R1_RANGE_SELECTION_V10_VERSION

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.teacher_fast_paths = load_minimum_level_teacher_hints(
            self.repository / ORDER9_R1_RANGE_TEACHER_HINTS_RELATIVE,
            repository_root=self.repository,
        )

    def prepare(self, case: Order9R1CalibrationCase):
        previous = self.teacher_candidate_overrides.get(case.candidate_id)
        previous_strict = self.strict_preferred_candidate_options
        minimum_id = corresponding_minimum_level_candidate_id(case.candidate_id)
        mapped = self.teacher_candidate_overrides.get(minimum_id, ())
        range_hint = self.teacher_fast_paths.get(case.candidate_id, ())
        minimum_hint = self.teacher_fast_paths.get(minimum_id, ())
        # An exact range hint is an intentionally finite ad-hoc teacher repair.
        # Do not silently append lower-range/general candidates: one rejected
        # candidate otherwise consumes several minutes per extra option.
        combined = (
            range_hint if range_hint else tuple(dict.fromkeys((*minimum_hint, *mapped)))
        )
        prepared = None
        try:
            if previous is None:
                for option in combined:
                    self.teacher_candidate_overrides[case.candidate_id] = (option,)
                    self.strict_preferred_candidate_options = True
                    with order9_lightweight_configuration_planning():
                        prepared = super().prepare(case)
                    if prepared.teacher_trajectory_complete:
                        return prepared
                if range_hint and prepared is not None:
                    return prepared
                self.teacher_candidate_overrides.pop(case.candidate_id, None)
                self.strict_preferred_candidate_options = previous_strict
            with order9_lightweight_configuration_planning():
                return super().prepare(case)
        finally:
            self.strict_preferred_candidate_options = previous_strict
            if previous is None:
                self.teacher_candidate_overrides.pop(case.candidate_id, None)
            else:
                self.teacher_candidate_overrides[case.candidate_id] = previous


def load_minimum_level_teacher_hints(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, tuple[Order9R1TeacherHint, ...]]:
    repository = Path(repository_root).resolve()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    minimum_values = payload.get("minimum_level_hints")
    range_values = payload.get("candidate_fast_paths")
    if (
        payload.get("config_version") != "order9_r1_range_teacher_hints_v10"
        or payload.get("strict") is not False
        or not isinstance(minimum_values, dict)
        or not minimum_values
        or not isinstance(range_values, dict)
    ):
        raise SchemaValidationError("R1 range teacher hints differ")
    result = {}
    for candidate_id, value in {**minimum_values, **range_values}.items():
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.startswith(
                (ORDER9_R1_MINIMUM_LEVEL_ID, *ORDER9_R1_RANGE_LEVEL_IDS)
            )
            or not isinstance(value, dict)
        ):
            raise SchemaValidationError("R1 range teacher hint identity differs")
        additional = value.get("additional_contact_options", [])
        if not isinstance(additional, list):
            raise SchemaValidationError("R1 range additional hints differ")
        options = []
        for option in (value, *additional):
            if not isinstance(option, dict):
                raise SchemaValidationError("R1 range teacher hint differs")
            binding = option.get("source_nominal_artifact")
            if not isinstance(binding, dict):
                raise SchemaValidationError("R1 range teacher hint source is missing")
            source = repository / str(binding.get("path", ""))
            if not source.is_file() or hash_file(source) != binding.get("sha256"):
                raise SchemaValidationError("R1 range teacher hint source changed")
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            ports = option.get("selected_surface_port_ids")
            group_id = option.get("candidate_group_id")
            clearance = float(option.get("pregrasp_clearance_m", -1.0))
            margin = float(option.get("collision_margin_m", -1.0))
            height = option.get("grasp_contact_height_offset_m")
            tangent = option.get("grasp_contact_tangent_offset_world_m")
            height_value = 0.0 if height is None else float(height)
            tangent_values = (
                (0.0, 0.0, 0.0)
                if tangent is None or not isinstance(tangent, list) or len(tangent) != 3
                else tuple(float(value) for value in tangent)
            )
            if (
                not isinstance(ports, list)
                or len(ports) != 2
                or len({int(port) for port in ports}) != 2
                or min(int(port) for port in ports) < 0
                or not isinstance(group_id, str)
                or not group_id
                or tuple(int(port) for port in ports)
                != tuple(
                    int(port)
                    for port in source_payload.get("selected_surface_port_ids", ())
                )
                or group_id != source_payload.get("selected_candidate_group_id")
                or not 0.08 <= clearance <= 0.30
                or not 0.005 <= margin <= 0.030
                or not 0.0 <= height_value <= 0.030
                or (
                    tangent is not None
                    and (
                        not isinstance(tangent, list)
                        or len(tangent) != 3
                        or not all(math.isfinite(float(value)) for value in tangent)
                        or math.sqrt(sum(float(value) ** 2 for value in tangent))
                        > 0.030 + 1.0e-12
                    )
                )
                or math.sqrt(
                    height_value * height_value
                    + sum(value * value for value in tangent_values)
                )
                > 0.030 + 1.0e-12
            ):
                raise SchemaValidationError("R1 range teacher hint content differs")
            base_option = (
                tuple(int(port) for port in ports),
                group_id,
                clearance,
                margin,
            )
            if tangent is not None and height is None:
                raise SchemaValidationError(
                    "R1 range tangent hint requires an explicit height"
                )
            if tangent is not None:
                options.append(
                    (
                        *base_option,
                        height_value,
                        tangent_values,
                    )
                )
            else:
                options.append(
                    base_option if height is None else (*base_option, float(height))
                )
        prefer_additional = value.get("prefer_additional_options_first", False)
        if not isinstance(prefer_additional, bool):
            raise SchemaValidationError("R1 range teacher hint order differs")
        if prefer_additional:
            options = [*options[1:], options[0]]
        result[candidate_id] = tuple(options)
    return result


def range_case_priority(candidate_id: str, sample_kind: str, sample_index: int):
    """Put lattice corners first and random interior samples last."""

    if sample_kind == "interior":
        return (4, int(sample_index), str(candidate_id))
    if sample_kind != "lattice" or not 0 <= int(sample_index) < 27:
        raise SchemaValidationError("R1 range sample identity is invalid")
    index = int(sample_index)
    coordinates = (index // 9, (index % 9) // 3, index % 3)
    centered = sum(value == 1 for value in coordinates)
    return (centered, index, str(candidate_id))


def chunk_candidate_ids(
    candidate_ids: Sequence[str], *, maximum_size: int = 4
) -> tuple[tuple[str, ...], ...]:
    if not 1 <= int(maximum_size) <= ORDER9_R1_MAXIMUM_CANDIDATES_PER_SCENE:
        raise ValueError("R1 range chunk size must be in [1, 4]")
    values = tuple(str(value) for value in candidate_ids)
    if (
        not values
        or any(not value for value in values)
        or len(values) != len(set(values))
    ):
        raise SchemaValidationError("R1 range candidate set is empty or repeats")
    return tuple(
        values[start : start + maximum_size]
        for start in range(0, len(values), maximum_size)
    )


@dataclass(frozen=True)
class Order9R1RangeLevelGate:
    level_id: str
    passed: bool
    lattice_candidate_count: int
    lattice_pass_count: int
    interior_candidate_count: int
    interior_teacher_pass_count: int
    interior_screen_pass_count: int
    interior_episode_count: int
    interior_success_count: int
    total_episode_count: int
    total_success_count: int
    safety_failure_count: int
    fallback_count: int
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "passed": self.passed,
            "lattice_candidate_count": self.lattice_candidate_count,
            "lattice_pass_count": self.lattice_pass_count,
            "interior_candidate_count": self.interior_candidate_count,
            "interior_teacher_pass_count": self.interior_teacher_pass_count,
            "interior_screen_pass_count": self.interior_screen_pass_count,
            "interior_episode_count": self.interior_episode_count,
            "interior_success_count": self.interior_success_count,
            "total_episode_count": self.total_episode_count,
            "total_success_count": self.total_success_count,
            "safety_failure_count": self.safety_failure_count,
            "fallback_count": self.fallback_count,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_range_level_gate(
    *,
    level_id: str,
    entries: Sequence[Mapping[str, Any]],
    minimum_lattice_pass_rate: float,
    minimum_teacher_feasibility_rate: float,
    minimum_fast_screen_pass_rate: float,
    minimum_isaac_success_rate: float,
    minimum_interior_teacher_feasibility_rate: float,
    minimum_interior_fast_screen_pass_rate: float,
    minimum_interior_isaac_success_rate: float,
    maximum_safety_failure_count: int,
    maximum_fallback_rate: float,
) -> Order9R1RangeLevelGate:
    """Apply the approved v1 gates without treating calibration as training data."""

    if level_id not in ORDER9_R1_RANGE_LEVEL_IDS or not entries:
        raise SchemaValidationError("R1 range level entries are invalid")
    lattice = [entry for entry in entries if entry.get("sample_kind") == "lattice"]
    interior = [entry for entry in entries if entry.get("sample_kind") == "interior"]
    if not lattice or not interior or len(lattice) + len(interior) != len(entries):
        raise SchemaValidationError("R1 range sample partition is invalid")

    lattice_pass = sum(bool(entry.get("candidate_passed")) for entry in lattice)
    interior_teacher = sum(bool(entry.get("teacher_feasible")) for entry in interior)
    interior_screen = sum(bool(entry.get("screen_passed")) for entry in interior)
    total_teacher = sum(bool(entry.get("teacher_feasible")) for entry in entries)
    total_screen = sum(bool(entry.get("screen_passed")) for entry in entries)
    interior_episodes = sum(int(entry.get("episode_count", 0)) for entry in interior)
    interior_success = sum(int(entry.get("success_count", 0)) for entry in interior)
    total_episodes = sum(int(entry.get("episode_count", 0)) for entry in entries)
    total_success = sum(int(entry.get("success_count", 0)) for entry in entries)
    safety = sum(int(entry.get("safety_failure_count", 0)) for entry in entries)
    fallback = sum(int(entry.get("fallback_count", 0)) for entry in entries)

    def rate(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else float(numerator) / float(denominator)

    checks = {
        "lattice_pass_rate": rate(lattice_pass, len(lattice))
        >= float(minimum_lattice_pass_rate),
        "teacher_feasibility_rate": rate(total_teacher, len(entries))
        >= float(minimum_teacher_feasibility_rate),
        "fast_screen_pass_rate": rate(total_screen, len(entries))
        >= float(minimum_fast_screen_pass_rate),
        "isaac_success_rate": rate(total_success, total_episodes)
        >= float(minimum_isaac_success_rate),
        "interior_teacher_feasibility_rate": rate(interior_teacher, len(interior))
        >= float(minimum_interior_teacher_feasibility_rate),
        "interior_fast_screen_pass_rate": rate(interior_screen, len(interior))
        >= float(minimum_interior_fast_screen_pass_rate),
        "interior_isaac_success_rate": rate(interior_success, interior_episodes)
        >= float(minimum_interior_isaac_success_rate),
        "maximum_safety_failure_count": safety <= int(maximum_safety_failure_count),
        "maximum_fallback_rate": rate(fallback, total_episodes)
        <= float(maximum_fallback_rate),
    }
    reasons = tuple(key for key, passed in checks.items() if not passed)
    return Order9R1RangeLevelGate(
        level_id=level_id,
        passed=not reasons,
        lattice_candidate_count=len(lattice),
        lattice_pass_count=lattice_pass,
        interior_candidate_count=len(interior),
        interior_teacher_pass_count=interior_teacher,
        interior_screen_pass_count=interior_screen,
        interior_episode_count=interior_episodes,
        interior_success_count=interior_success,
        total_episode_count=total_episodes,
        total_success_count=total_success,
        safety_failure_count=safety,
        fallback_count=fallback,
        failure_reasons=reasons,
    )


def load_range_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("protocol_version") != "order9_r1_range_selection_protocol_v10"
        or payload.get("status") != "approved"
        or tuple(payload.get("selection_level_ids", ())) != ORDER9_R1_RANGE_LEVEL_IDS
        or int(payload.get("maximum_candidates_per_isaac_scene", -1)) != 4
        or not math.isclose(float(payload.get("isaac_environment_spacing_m", -1)), 3.0)
        or int(payload.get("preparation_numeric_library_threads_per_worker", -1)) != 2
        or int(payload.get("python_hash_seed", -1)) != 0
        or payload.get("pi_l_actor_command_applied") is not False
        or payload.get("qpid_qp_applied") is not True
        or payload.get("local_servo_applied") is not True
        or payload.get("stop_on_first_failed_level") is not True
        or payload.get("accept_largest_consecutive_passing_level") is not True
        or payload.get("restart_reuses_validated_evidence") is not True
        or payload.get("single_candidate_confirmation_on_bounded_failure") is not True
        or payload.get("calibration_evidence_training_eligible") is not False
        or payload.get("held_out_confirmation_in_scope") is not False
        or payload.get("formal_teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 range-selection protocol differs")
    return payload


__all__ = [
    "ORDER9_R1_MAXIMUM_CANDIDATES_PER_SCENE",
    "ORDER9_R1_MINIMUM_LEVEL_ID",
    "ORDER9_R1_RANGE_LEVEL_IDS",
    "ORDER9_R1_RANGE_SELECTION_V10_VERSION",
    "Order9R1RangeLevelGate",
    "Order9R1RangeTeacherScreenPipelineV10",
    "chunk_candidate_ids",
    "corresponding_minimum_level_candidate_id",
    "evaluate_range_level_gate",
    "load_minimum_level_teacher_hints",
    "load_range_protocol",
    "range_case_priority",
]
