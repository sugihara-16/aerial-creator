from __future__ import annotations

"""R1 teacher preparation with contact-attitude admission before path search."""

import json
from pathlib import Path

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.training.order9_c3_nominal_trajectory import (
    generate_order9_c3_nominal_grasp_trajectory,
)
from amsrr.training.order9_r1_calibration_runner import (
    Order9R1CalibrationCase,
    Order9R1DeterministicTeacherScreenPipeline,
    Order9R1PreparedCandidate,
)
from amsrr.training.order9_r1_joint_reserve import (
    project_order9_r1_nominal_joint_reserve,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_EARLY_TILT_PIPELINE_VERSION = (
    "order9_r1_early_contact_solution_tilt_pipeline_v1"
)
ORDER9_R1_TEACHER_OVERRIDES_RELATIVE = Path(
    "configs/training/order9_r1_teacher_overrides_v1.json"
)


class Order9R1EarlyTiltTeacherScreenPipeline(
    Order9R1DeterministicTeacherScreenPipeline
):
    """Reject/replace extreme contact IK solutions before dense path work."""

    pipeline_version = ORDER9_R1_EARLY_TILT_PIPELINE_VERSION

    def __init__(
        self, *, strict_preferred_candidate_options: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.strict_preferred_candidate_options = bool(
            strict_preferred_candidate_options
        )
        (
            self.teacher_surface_overrides,
            self.teacher_candidate_overrides,
        ) = _load_teacher_surface_overrides(
            self.repository / ORDER9_R1_TEACHER_OVERRIDES_RELATIVE
        )

    def prepare(self, case: Order9R1CalibrationCase) -> Order9R1PreparedCandidate:
        graph_path = (
            self.bucket_manifest_path.parent / case.source_bucket.morphology_graph_path
        )
        if hash_file(graph_path) != case.source_bucket.morphology_graph_sha256:
            raise SchemaValidationError("Order9 R1 source morphology bytes changed")
        graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
        if graph.stable_hash() != case.source_bucket.morphology_hash:
            raise SchemaValidationError("Order9 R1 source morphology hash changed")
        binding = case.source_bucket.metadata.get("accepted_nominal_trajectory", {})
        preferred = binding.get("selected_surface_port_ids")
        preferred_ports = (
            tuple(int(value) for value in preferred)
            if isinstance(preferred, list) and len(preferred) == 2
            else None
        )
        preferred_port_options = self.teacher_surface_overrides.get(
            case.source_bucket.bucket_id,
            (() if preferred_ports is None else (preferred_ports,)),
        )
        preferred_candidate_options = self.teacher_candidate_overrides.get(
            case.candidate_id,
            (),
        )
        excluded_surface_port_id_pairs = tuple(
            getattr(
                self,
                "excluded_surface_port_id_pairs_by_source",
                {},
            ).get(case.source_bucket.bucket_id, ())
        )
        try:
            nominal = _generate_with_safe_surface_fallback(
                task_spec=case.task_spec,
                structural_target=graph,
                physical_model=self.physical_model,
                preferred_surface_port_id_options=preferred_port_options,
                preferred_candidate_options=preferred_candidate_options,
                maximum_body_tilt_rad=self.screen_config.maximum_body_tilt_rad,
                strict_preferred_candidate_options=(
                    self.strict_preferred_candidate_options
                ),
                excluded_surface_port_id_pairs=(
                    excluded_surface_port_id_pairs
                ),
            )
            if self.enforce_joint_limit_reserve_during_ik:
                nominal = project_order9_r1_nominal_joint_reserve(
                    nominal,
                    task_spec=case.task_spec,
                    physical_model=self.physical_model,
                    minimum_normalized_joint_limit_reserve=(
                        self.screen_config.minimum_normalized_joint_limit_reserve
                    ),
                    anchor_position_tolerance_m=(self.anchor_position_tolerance_m),
                )
        except SchemaValidationError as exc:
            return Order9R1PreparedCandidate(
                candidate_id=case.candidate_id,
                candidate_task_hash=case.task_spec.stable_hash(),
                morphology_hash=case.source_bucket.morphology_hash,
                physical_model_hash=case.physical_model_hash,
                teacher_trajectory_complete=False,
                failure_reason=str(exc),
            )
        return Order9R1PreparedCandidate(
            candidate_id=case.candidate_id,
            candidate_task_hash=case.task_spec.stable_hash(),
            morphology_hash=case.source_bucket.morphology_hash,
            physical_model_hash=case.physical_model_hash,
            teacher_trajectory_complete=True,
            failure_reason=None,
            payload=nominal,
        )


def _generate_with_safe_surface_fallback(
    *,
    task_spec,
    structural_target,
    physical_model,
    preferred_surface_port_id_options: tuple[tuple[int, int], ...],
    preferred_candidate_options: tuple[
        tuple[tuple[int, int], str, float, float]
        | tuple[tuple[int, int], str, float, float, float]
        | tuple[
            tuple[int, int],
            str,
            float,
            float,
            float,
            tuple[float, float, float],
        ],
        ...,
    ] = (),
    maximum_body_tilt_rad: float,
    strict_preferred_candidate_options: bool = False,
    excluded_surface_port_id_pairs: tuple[tuple[int, int], ...] = (),
):
    arguments = {
        "task_spec": task_spec,
        "structural_target": structural_target,
        "physical_model": physical_model,
        "maximum_contact_solution_body_tilt_rad": maximum_body_tilt_rad,
        "excluded_surface_port_id_pairs": excluded_surface_port_id_pairs,
    }
    candidate_failures: list[str] = []
    for option in preferred_candidate_options:
        if len(option) == 4:
            (
                preferred_surface_port_ids,
                preferred_candidate_group_id,
                pregrasp_clearance_m,
                collision_margin_m,
            ) = option
            grasp_contact_height_offset_m = None
            grasp_contact_tangent_offset_world_m = None
        elif len(option) == 5:
            (
                preferred_surface_port_ids,
                preferred_candidate_group_id,
                pregrasp_clearance_m,
                collision_margin_m,
                grasp_contact_height_offset_m,
            ) = option
            grasp_contact_tangent_offset_world_m = None
        elif len(option) == 6:
            (
                preferred_surface_port_ids,
                preferred_candidate_group_id,
                pregrasp_clearance_m,
                collision_margin_m,
                grasp_contact_height_offset_m,
                grasp_contact_tangent_offset_world_m,
            ) = option
        else:
            raise SchemaValidationError(
                "R1 preferred contact option width is invalid"
            )
        try:
            height_argument = (
                {}
                if grasp_contact_height_offset_m is None
                else {
                    "grasp_contact_height_offset_m": (
                        grasp_contact_height_offset_m
                    )
                }
            )
            tangent_argument = (
                {}
                if grasp_contact_tangent_offset_world_m is None
                else {
                    "grasp_contact_tangent_offset_world_m": (
                        grasp_contact_tangent_offset_world_m
                    )
                }
            )
            return generate_order9_c3_nominal_grasp_trajectory(
                **arguments,
                preferred_surface_port_ids=preferred_surface_port_ids,
                preferred_candidate_group_id=preferred_candidate_group_id,
                pregrasp_clearance_m=pregrasp_clearance_m,
                collision_margin_m=collision_margin_m,
                **height_argument,
                **tangent_argument,
            )
        except SchemaValidationError as error:
            candidate_failures.append(str(error))
            continue
    if preferred_candidate_options and strict_preferred_candidate_options:
        raise SchemaValidationError(
            "R1 bound preferred contact options are infeasible: "
            + " | ".join(candidate_failures[:4])
        )
    if not preferred_surface_port_id_options:
        return generate_order9_c3_nominal_grasp_trajectory(**arguments)
    for preferred_surface_port_ids in preferred_surface_port_id_options:
        try:
            return generate_order9_c3_nominal_grasp_trajectory(
                **arguments,
                preferred_surface_port_ids=preferred_surface_port_ids,
            )
        except SchemaValidationError:
            continue
    # The listed pairs are deterministic fast paths, not an R1 teacher-data
    # requirement.  A final unpinned attempt preserves coverage for poses
    # that need another unoccupied, non-extreme, collision-clear pair.
    return generate_order9_c3_nominal_grasp_trajectory(**arguments)


def _load_teacher_surface_overrides(
    path: Path,
) -> tuple[
    dict[str, tuple[tuple[int, int], ...]],
    dict[str, tuple[tuple[tuple[int, int], str, float, float], ...]],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_version") != "order9_r1_teacher_overrides_v1":
        raise SchemaValidationError("R1 teacher override version is invalid")
    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        raise SchemaValidationError("R1 teacher overrides must be an object")
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for bucket_id, value in overrides.items():
        if not isinstance(bucket_id, str) or not bucket_id:
            raise SchemaValidationError("R1 teacher override bucket id is invalid")
        if not isinstance(value, dict):
            raise SchemaValidationError("R1 teacher override entry is invalid")
        options = value.get("preferred_surface_port_id_options")
        if options is None:
            options = [value.get("selected_surface_port_ids")]
        if not isinstance(options, list) or not options:
            raise SchemaValidationError("R1 teacher override ports are invalid")
        validated = []
        for ports in options:
            if (
                not isinstance(ports, list)
                or len(ports) != 2
                or len({int(item) for item in ports}) != 2
                or min(int(item) for item in ports) < 0
            ):
                raise SchemaValidationError("R1 teacher override ports are invalid")
            pair = tuple(int(item) for item in ports)
            if pair not in validated:
                validated.append(pair)
        result[bucket_id] = tuple(validated)
    candidate_overrides = payload.get("candidate_overrides", {})
    if not isinstance(candidate_overrides, dict):
        raise SchemaValidationError("R1 teacher candidate overrides must be an object")
    candidate_result: dict[
        str, tuple[tuple[tuple[int, int], str, float, float], ...]
    ] = {}
    for candidate_id, value in candidate_overrides.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise SchemaValidationError("R1 teacher candidate id is invalid")
        if not isinstance(value, dict):
            raise SchemaValidationError("R1 teacher candidate override is invalid")
        options = value.get("preferred_contact_options")
        if not isinstance(options, list) or not options:
            raise SchemaValidationError("R1 teacher contact options are invalid")
        validated_options = []
        for option in options:
            if not isinstance(option, dict):
                raise SchemaValidationError("R1 teacher contact option is invalid")
            ports = option.get("selected_surface_port_ids")
            group_id = option.get("candidate_group_id")
            clearance = float(option.get("pregrasp_clearance_m", 0.08))
            collision_margin = float(option.get("collision_margin_m", 0.007))
            if (
                not isinstance(ports, list)
                or len(ports) != 2
                or len({int(item) for item in ports}) != 2
                or min(int(item) for item in ports) < 0
                or not isinstance(group_id, str)
                or not group_id
                or not 0.08 <= clearance <= 0.30
                or not 0.005 <= collision_margin <= 0.030
            ):
                raise SchemaValidationError("R1 teacher contact option is invalid")
            validated_options.append(
                (
                    tuple(int(item) for item in ports),
                    group_id,
                    clearance,
                    collision_margin,
                )
            )
        candidate_result[candidate_id] = tuple(validated_options)
    return result, candidate_result


__all__ = [
    "ORDER9_R1_EARLY_TILT_PIPELINE_VERSION",
    "Order9R1EarlyTiltTeacherScreenPipeline",
]
