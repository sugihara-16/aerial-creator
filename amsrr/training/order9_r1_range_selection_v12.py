from __future__ import annotations

"""R1 range teacher with a hard 30 mm complete-path support constraint."""

from dataclasses import replace
from threading import RLock
from unittest.mock import patch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training import order9_c3_nominal_trajectory as c3_nominal
from amsrr.training import order9_c3_teacher as c3_teacher
from amsrr.training.order9_configuration_space_planner import (
    order9_lightweight_configuration_planning,
)
from amsrr.training.order9_r1_calibration_runner import (
    Order9R1CalibrationCase,
    Order9R1PreparedCandidate,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    build_order9_r1_explicit_support_collision_object_v11,
)
from amsrr.training.order9_r1_range_selection_v11 import (
    Order9R1RangeTeacherScreenPipelineV11,
)
from amsrr.training.order9_r1_range_selection_v10 import (
    ORDER9_R1_MINIMUM_LEVEL_ID,
)
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (
    Order9R1RotationalTeacherScreenPipelineV7,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    Order9R1SupportClearanceTeacherV12Contract,
    audit_order9_r1_v12_candidate_at_planning_clearance,
    build_order9_r1_support_clearance_generation_certificate,
    derive_order9_r1_vertical_clearance_nominal,
    order9_r1_vertical_clearance_shift_candidates,
    select_order9_r1_maximin_clearance_candidate,
)

ORDER9_R1_RANGE_SELECTION_V12_VERSION = "order9_r1_range_selection_v12"
_R1_V12_GENERATION_LOCK = RLock()


class Order9R1ClearanceConstrainedTeacherScreenPipelineV12(
    Order9R1RangeTeacherScreenPipelineV11
):
    """Select only complete paths generated under the 30 mm hard target."""

    pipeline_version = ORDER9_R1_RANGE_SELECTION_V12_VERSION

    def __init__(
        self,
        *,
        clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
        physical_model_config_path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.clearance_contract = clearance_contract
        self.physical_model_config_path = physical_model_config_path
        self.clearance_generation_certificates: dict[str, dict] = {}
        self.clearance_candidate_audits: dict[str, tuple[dict, ...]] = {}

    def prepare(self, case: Order9R1CalibrationCase) -> Order9R1PreparedCandidate:
        with _R1_V12_GENERATION_LOCK:
            first = self._prepare_once(case)
            if not first.teacher_trajectory_complete:
                return first
            evaluated = []
            self._append_complete_path_audit(case, first, evaluated)
            if evaluated[-1][1].get("accepted") is not True:
                self._try_vertical_clearance_shifts(case, first, evaluated)
            if not self._has_admitted_candidate(evaluated):
                self._try_alternate_contact_groups(case, first, evaluated)

        self.clearance_candidate_audits[case.candidate_id] = tuple(
            dict(audit) for _prepared, audit in evaluated
        )
        try:
            selected_index, selected, selected_audit = (
                select_order9_r1_maximin_clearance_candidate(evaluated)
            )
        except SchemaValidationError:
            first_failure = next(
                (
                    audit.get("failure_examples", [None])[0]
                    for _prepared, audit in evaluated
                    if audit.get("failure_examples")
                ),
                None,
            )
            return Order9R1PreparedCandidate(
                candidate_id=case.candidate_id,
                candidate_task_hash=case.task_spec.stable_hash(),
                morphology_hash=case.source_bucket.morphology_hash,
                physical_model_hash=case.physical_model_hash,
                teacher_trajectory_complete=False,
                failure_reason=(
                    "R1 30 mm complete-path support-clearance generation "
                    f"exhausted: {first_failure}"
                ),
            )
        certificate = build_order9_r1_support_clearance_generation_certificate(
            selected_audit,
            clearance_contract=self.clearance_contract,
            candidate_id=case.candidate_id,
            candidate_rank=selected_index,
            evaluated_candidate_count=len(evaluated),
        )
        self.clearance_generation_certificates[case.candidate_id] = certificate
        return selected

    def _prepare_once(self, case: Order9R1CalibrationCase):
        if case.level_id != ORDER9_R1_MINIMUM_LEVEL_ID:
            return super().prepare(case)
        collision = build_order9_r1_explicit_support_collision_object_v11(
            case.task_spec,
            contract=self.geometry_contract,
        )
        original_offset = c3_teacher._offset_horizontal_grasp_candidates

        def raised_offset(candidate_set, *, height_offset_m, tangent_offset_world_m):
            return original_offset(
                candidate_set,
                height_offset_m=max(
                    float(height_offset_m),
                    self.geometry_contract.grasp_contact_height_offset_m,
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
                if previous is None:
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

    def _append_complete_path_audit(self, case, prepared, evaluated) -> None:
        _phases, audit = audit_order9_r1_v12_candidate_at_planning_clearance(
            prepared.payload,
            task_spec=case.task_spec,
            physical_model_config_path=self.physical_model_config_path,
            geometry_contract=self.geometry_contract,
            clearance_contract=self.clearance_contract,
            fail_fast=True,
        )
        evaluated.append((prepared, audit))

    @staticmethod
    def _has_admitted_candidate(evaluated) -> bool:
        return any(
            audit.get("accepted") is True and audit.get("status") == "accepted"
            for _prepared, audit in evaluated
        )

    def _try_vertical_clearance_shifts(self, case, prepared, evaluated) -> None:
        initial_audit = evaluated[-1][1]
        for shift_m in order9_r1_vertical_clearance_shift_candidates(
            initial_audit,
            clearance_contract=self.clearance_contract,
        ):
            shifted = replace(
                prepared,
                payload=derive_order9_r1_vertical_clearance_nominal(
                    prepared.payload,
                    vertical_shift_m=shift_m,
                ),
            )
            self._append_complete_path_audit(case, shifted, evaluated)
            audit = evaluated[-1][1]
            audit.update(
                {
                    "generation_method": ("analytic_vertical_contact_frame_shift_v1"),
                    "vertical_contact_frame_shift_m": shift_m,
                    "original_deterministic_ik_solution_rigidly_preserved": True,
                }
            )
            if audit.get("accepted") is True:
                return

    def _try_alternate_contact_groups(self, case, first, evaluated) -> None:
        nominal = first.payload
        selection = nominal.selection_bundle
        selected_group = selection.trajectory_plan.candidate_group_id
        group_ids = tuple(
            proposal.group_id
            for proposal in selection.contact_candidate_set.group_proposals
            if proposal.group_id != selected_group
        )[: self.clearance_contract.maximum_alternate_contact_group_count]
        previous = self.teacher_candidate_overrides.get(case.candidate_id)
        previous_strict = self.strict_preferred_candidate_options
        try:
            self.strict_preferred_candidate_options = True
            for group_id in group_ids:
                self.teacher_candidate_overrides[case.candidate_id] = (
                    (
                        tuple(selection.selected_surface_port_ids),
                        group_id,
                        0.08,
                        0.005,
                        self.geometry_contract.grasp_contact_height_offset_m,
                    ),
                )
                prepared = self._prepare_once(case)
                if not prepared.teacher_trajectory_complete:
                    continue
                self._append_complete_path_audit(case, prepared, evaluated)
                if evaluated[-1][1].get("accepted") is not True:
                    self._try_vertical_clearance_shifts(case, prepared, evaluated)
                if self._has_admitted_candidate(evaluated):
                    return
        finally:
            self.strict_preferred_candidate_options = previous_strict
            if previous is None:
                self.teacher_candidate_overrides.pop(case.candidate_id, None)
            else:
                self.teacher_candidate_overrides[case.candidate_id] = previous


__all__ = [
    "ORDER9_R1_RANGE_SELECTION_V12_VERSION",
    "Order9R1ClearanceConstrainedTeacherScreenPipelineV12",
]
