from __future__ import annotations

"""R1 range teacher using the v13 exact requested-margin geometry gate."""

import copy
from contextlib import nullcontext
from dataclasses import replace
import json
import math
from pathlib import Path
from unittest.mock import patch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training import order9_r1_range_selection_v10 as range_v10
from amsrr.training import order9_r1_range_selection_v12 as range_v12
from amsrr.training import order9_r1_early_tilt_pipeline as early_tilt
from amsrr.training.order9_c3_nominal_trajectory import _flatten_nominal_windows
from amsrr.training.order9_r1_complete_task_geometry_v13 import (
    audit_order9_r1_complete_task_geometry_v13,
)
from amsrr.training.order9_r1_range_selection_v12 import (
    Order9R1ClearanceConstrainedTeacherScreenPipelineV12,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_VERSION,
    derive_order9_r1_vertical_clearance_nominal,
    materialize_order9_r1_v12_candidate_phases,
    _selected_group_candidate_ids,
)
from amsrr.utils.hashing import stable_hash

ORDER9_R1_RANGE_SELECTION_V13_VERSION = "order9_r1_range_selection_v13"
ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE = Path(
    "configs/training/order9_r1_range_teacher_repairs_v13.json"
)
ORDER9_C3_HUMAN_POSTURE_REJECTIONS_V2_RELATIVE = Path(
    "configs/training/order9_c3_human_posture_rejections_v2.json"
)
ORDER9_R1_TRAIN027_SOURCE_BUCKET_ID = "train-000027-7fe33c662d36"
ORDER9_R1_TRAIN027_CONFIGURATION_SEED_SOURCE_CANDIDATE_ID = (
    "r1_l2_20mm_10deg__train__"
    "train-000027-7fe33c662d36__lattice_06"
)


class Order9R1ExactMarginTeacherScreenPipelineV13(
    Order9R1ClearanceConstrainedTeacherScreenPipelineV12
):
    """Select the same finite teachers with the corrected 30 mm proof."""

    pipeline_version = ORDER9_R1_RANGE_SELECTION_V13_VERSION

    def __init__(self, **kwargs) -> None:
        repository = Path(kwargs["repository_root"]).resolve()
        self.human_posture_rejections = _load_human_posture_rejections(
            repository / ORDER9_C3_HUMAN_POSTURE_REJECTIONS_V2_RELATIVE,
            repository=repository,
        )
        self.excluded_surface_port_id_pairs_by_source = {
            source_bucket_id: value["excluded_surface_port_id_pairs"]
            for source_bucket_id, value in self.human_posture_rejections.items()
        }
        self.bounded_local_contact_repairs = _load_bounded_local_contact_repairs(
            repository / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
            repository=repository,
        )
        self.configuration_goal_seeds = _load_configuration_goal_seeds(
            repository / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
            repository=repository,
        )
        self.configuration_goal_seed_evidence = (
            _load_configuration_goal_seed_evidence(
                repository / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
                repository=repository,
            )
        )
        default_seed_id = (
            ORDER9_R1_TRAIN027_CONFIGURATION_SEED_SOURCE_CANDIDATE_ID
        )
        if (
            default_seed_id not in self.configuration_goal_seeds
            or default_seed_id not in self.configuration_goal_seed_evidence
        ):
            raise SchemaValidationError(
                "R1 v13 train027 source configuration seed differs"
            )
        self.source_default_configuration_goal_seeds = {
            ORDER9_R1_TRAIN027_SOURCE_BUCKET_ID: (
                self.configuration_goal_seeds[default_seed_id],
                self.configuration_goal_seed_evidence[default_seed_id],
                default_seed_id,
            )
        }
        self.confirmed_teacher_option_overrides = (
            _load_confirmed_teacher_option_overrides(
                repository / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
            )
        )
        self.automatic_rotation_contact_retries: dict[str, dict] = {}
        self.automatic_lightweight_contact_retries: dict[str, dict] = {}
        self.contact_acquisition_lead_windows = _load_contact_acquisition_leads(
            repository / range_v10.ORDER9_R1_RANGE_TEACHER_HINTS_RELATIVE
        )
        super().__init__(**kwargs)
        _reject_forbidden_fast_paths(
            self.teacher_fast_paths,
            exclusions=self.excluded_surface_port_id_pairs_by_source,
        )
        self.teacher_surface_overrides = _filter_forbidden_surface_overrides(
            self.teacher_surface_overrides,
            exclusions=self.excluded_surface_port_id_pairs_by_source,
        )
        self.teacher_candidate_overrides = _filter_forbidden_candidate_overrides(
            self.teacher_candidate_overrides,
            exclusions=self.excluded_surface_port_id_pairs_by_source,
        )
        for source_bucket_id, rule in self.rotation_stability_rules.items():
            _assert_pair_allowed(
                source_bucket_id,
                rule.selected_surface_port_ids,
                exclusions=self.excluded_surface_port_id_pairs_by_source,
                label="rotation stability rule",
            )
        train027_prefix = "r1_l2_20mm_10deg__train__train-000027-7fe33c662d36"
        default_keys = (
            f"{train027_prefix}__lattice_06",
            f"{train027_prefix}__lattice_24",
        )
        if any(key not in self.teacher_fast_paths for key in default_keys):
            raise SchemaValidationError("R1 v13 train027 default teacher hint differs")
        self.source_default_fast_paths = {
            "train-000027-7fe33c662d36": tuple(
                option
                for key in default_keys
                for option in self.teacher_fast_paths[key]
            )
        }

    def _prepare_once(self, case):
        rejection = self.human_posture_rejections.get(
            case.source_bucket.bucket_id
        )
        if rejection is not None and str(case.source_bucket.structural_hash) != str(
            rejection["structural_hash"]
        ):
            raise SchemaValidationError(
                "R1 v13 human posture rejection structural hash differs"
            )
        confirmed_option = self.confirmed_teacher_option_overrides.get(
            case.candidate_id
        )
        had_fast_path = case.candidate_id in self.teacher_fast_paths
        previous_fast_path = self.teacher_fast_paths.get(case.candidate_id)
        had_candidate_override = case.candidate_id in self.teacher_candidate_overrides
        previous_candidate_override = self.teacher_candidate_overrides.get(
            case.candidate_id
        )
        previous_strict = self.strict_preferred_candidate_options
        if confirmed_option is not None:
            self.teacher_fast_paths[case.candidate_id] = (confirmed_option,)
            self.teacher_candidate_overrides[case.candidate_id] = (
                confirmed_option,
            )
            self.strict_preferred_candidate_options = True
        injected_default = False
        if (
            case.candidate_id not in self.teacher_fast_paths
            and case.source_bucket.bucket_id in self.source_default_fast_paths
        ):
            self.teacher_fast_paths[case.candidate_id] = self.source_default_fast_paths[
                case.source_bucket.bucket_id
            ]
            injected_default = True
        try:
            prepared = self._prepare_once_with_bounded_repair(case)
        finally:
            if injected_default:
                self.teacher_fast_paths.pop(case.candidate_id, None)
            if confirmed_option is not None:
                if had_fast_path:
                    self.teacher_fast_paths[case.candidate_id] = previous_fast_path
                else:
                    self.teacher_fast_paths.pop(case.candidate_id, None)
                if had_candidate_override:
                    self.teacher_candidate_overrides[case.candidate_id] = (
                        previous_candidate_override
                    )
                else:
                    self.teacher_candidate_overrides.pop(case.candidate_id, None)
                self.strict_preferred_candidate_options = previous_strict
        lead = self.contact_acquisition_lead_windows.get(case.candidate_id, 0)
        if prepared.teacher_trajectory_complete and lead:
            prepared = replace(
                prepared,
                payload=_resegment_order9_r1_contact_acquisition(
                    prepared.payload,
                    lead_window_count=lead,
                ),
            )
        return prepared

    def _prepare_once_with_bounded_repair(self, case):
        if (
            case.candidate_id in self.bounded_local_contact_repairs
            or case.candidate_id in self.configuration_goal_seeds
        ):
            return self._prepare_with_bounded_contact_search(case)

        prepared = super()._prepare_once(case)
        failure_reason = str(prepared.failure_reason)
        lightweight_trigger = next(
            (
                trigger
                for trigger in (
                    "sampled local contact search is disabled in lightweight admission",
                    "sampled overhead search is disabled in lightweight admission",
                )
                if trigger in failure_reason
            ),
            None,
        )
        if not prepared.teacher_trajectory_complete and lightweight_trigger:
            retried = self._prepare_with_bounded_contact_search(case)
            self.automatic_lightweight_contact_retries[case.candidate_id] = {
                "retry_version": (
                    "order9_r1_automatic_lightweight_contact_retry_v1"
                ),
                "trigger": lightweight_trigger,
                "bounded_local_contact_maximum_iterations": 160,
                "original_failure_reason": prepared.failure_reason,
                "teacher_trajectory_complete": bool(
                    retried.teacher_trajectory_complete
                ),
                "failure_reason": retried.failure_reason,
                "isaac_invoked": False,
                "controller_layers_invoked": False,
                "acceptance_or_safety_gate_changed": False,
            }
            return retried
        if prepared.teacher_trajectory_complete or (
            "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM" not in failure_reason
        ):
            return prepared

        # A failed moment-arm audit means that one reachable contact pair was
        # unsuitable; it is not evidence that the pose or morphology is
        # infeasible.  Retry every finite, rotation-bearing option from the
        # bound rule.  Only the path search changes; all IK, geometry, support,
        # joint-reserve, and rotation gates remain unchanged.
        rule = self.rotation_stability_rules.get(case.source_bucket.bucket_id)
        if rule is None:
            return prepared
        attempts = []
        for option in rule.candidate_options:
            candidate_rule = replace(
                rule,
                selected_surface_port_ids=tuple(option[0]),
                preferred_candidate_group_ids=(str(option[1]),),
                pregrasp_clearance_m=float(option[2]),
                collision_margin_m=float(option[3]),
                grasp_contact_height_offset_m=float(option[4]),
            )
            previous_rule = self.rotation_stability_rules[case.source_bucket.bucket_id]
            previous_strict = self.strict_preferred_candidate_options
            self.rotation_stability_rules[case.source_bucket.bucket_id] = candidate_rule
            self.strict_preferred_candidate_options = True
            try:
                retried = self._prepare_with_bounded_contact_search(case)
            finally:
                self.strict_preferred_candidate_options = previous_strict
                self.rotation_stability_rules[case.source_bucket.bucket_id] = (
                    previous_rule
                )
            attempts.append(
                {
                    "selected_surface_port_ids": list(option[0]),
                    "candidate_group_id": str(option[1]),
                    "teacher_trajectory_complete": bool(
                        retried.teacher_trajectory_complete
                    ),
                    "failure_reason": retried.failure_reason,
                }
            )
            if retried.teacher_trajectory_complete:
                self.automatic_rotation_contact_retries[case.candidate_id] = {
                    "retry_version": ("order9_r1_automatic_rotation_contact_retry_v1"),
                    "trigger": "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM",
                    "bounded_local_contact_maximum_iterations": 160,
                    "attempts": attempts,
                    "selected_surface_port_ids": list(option[0]),
                    "selected_candidate_group_id": str(option[1]),
                    "isaac_invoked": False,
                    "controller_layers_invoked": False,
                }
                return retried
        self.automatic_rotation_contact_retries[case.candidate_id] = {
            "retry_version": "order9_r1_automatic_rotation_contact_retry_v1",
            "trigger": "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM",
            "bounded_local_contact_maximum_iterations": 160,
            "attempts": attempts,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
        }
        return prepared

    def _prepare_with_bounded_contact_search(self, case):
        # Higher range levels also enter the lightweight context in v10.  Both
        # nested contexts must be disabled or the bounded repair silently
        # remains a lightweight-only attempt.
        configuration_seed = self.configuration_goal_seeds.get(case.candidate_id)
        if configuration_seed is None:
            source_default = self.source_default_configuration_goal_seeds.get(
                case.source_bucket.bucket_id
            )
            if source_default is not None:
                configuration_seed, source_evidence, source_candidate_id = (
                    source_default
                )
                self.configuration_goal_seed_evidence[case.candidate_id] = {
                    "record_version": (
                        "order9_r1_configuration_goal_seed_evidence_v1"
                    ),
                    "candidate_id": case.candidate_id,
                    "scope": "configuration_space_goal_initialization_only",
                    "source_trajectory": dict(
                        source_evidence["source_trajectory"]
                    ),
                    "joint_seed_hash": stable_hash(configuration_seed),
                    "contact_goal_seed_applied": False,
                    "contact_ik_constraints_changed": False,
                    "acceptance_or_safety_gate_changed": False,
                    "isaac_invoked_during_seed_selection": False,
                    "controller_layers_invoked_during_seed_selection": False,
                    "automatic_source_configuration_seed_reuse": True,
                    "source_seed_candidate_id": source_candidate_id,
                }
        original_generate = early_tilt.generate_order9_c3_nominal_grasp_trajectory

        def generate_with_configuration_seed(*values, **keywords):
            if configuration_seed is not None:
                keywords["configuration_goal_joint_seed_positions_rad"] = (
                    configuration_seed
                )
            return original_generate(*values, **keywords)

        with patch.object(
            range_v12,
            "order9_lightweight_configuration_planning",
            return_value=nullcontext(),
        ), patch.object(
            range_v10,
            "order9_lightweight_configuration_planning",
            return_value=nullcontext(),
        ), patch.object(
            early_tilt,
            "generate_order9_c3_nominal_grasp_trajectory",
            side_effect=generate_with_configuration_seed,
        ):
            return super()._prepare_once(case)

    def _append_complete_path_audit(self, case, prepared, evaluated) -> None:
        phases = materialize_order9_r1_v12_candidate_phases(
            prepared.payload,
            task_spec=case.task_spec,
        )
        planning_geometry = replace(
            self.geometry_contract,
            required_robot_support_clearance_m=(
                self.clearance_contract.planning_clearance_m
            ),
        )
        audit = audit_order9_r1_complete_task_geometry_v13(
            phases=phases,
            task_spec=case.task_spec,
            morphology=(
                prepared.payload.selection_bundle.design_output.target_morphology
            ),
            contact_candidate_set=(
                prepared.payload.selection_bundle.contact_candidate_set
            ),
            physical_model_config_path=self.physical_model_config_path,
            contract=planning_geometry,
            fail_fast=True,
        )
        audit.update(
            {
                "teacher_contract_version": (
                    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_VERSION
                ),
                "teacher_contract_sha256": self.clearance_contract.config_sha256,
                "acceptance_clearance_m": (
                    self.clearance_contract.acceptance_clearance_m
                ),
                "planning_clearance_m": self.clearance_contract.planning_clearance_m,
                "planning_buffer_m": (self.clearance_contract.actual_planning_buffer_m),
                "generation_time_clearance_constraint_applied": True,
                "complete_phase_candidate_selection": True,
                "generation_method": (
                    "complete_path_hard_constraint_candidate_selection_v1"
                ),
                "vertical_contact_frame_shift_m": 0.0,
                "original_deterministic_ik_solution_rigidly_preserved": True,
                "isaac_invoked": False,
                "controller_layers_invoked": False,
            }
        )
        evaluated.append((prepared, audit))

    def _try_vertical_clearance_shifts(self, case, prepared, evaluated) -> None:
        # Most support misses are solved by the directly computed 5 mm
        # vertical candidate.  Evaluate that bounded sequence first: trying
        # eight planar directions and every 0.5 mm vertical increment before
        # it multiplied one complete exact-mesh audit into tens of audits.
        super()._try_vertical_clearance_shifts(case, prepared, evaluated)
        if self._has_admitted_candidate(evaluated):
            return
        # The planar fallback exists for the one recorded near-boundary case
        # whose support distance increases under a sub-millimetre XY shift.
        # It is not a general search over every teacher trajectory.
        if case.candidate_id.endswith("train-000027-7fe33c662d36__lattice_24"):
            self._try_planar_clearance_shifts(case, prepared, evaluated)

    def _try_planar_clearance_shifts(self, case, prepared, evaluated) -> None:
        """Move the rigid contact frame tangentially to recover mesh reserve.

        The selected surface region already permits substantially more
        tangential freedom than this bounded search.  First evaluate the eight
        0.5 mm directions.  If none is admitted, continue only along the
        direction that produced the largest complete-path clearance.  This
        avoids another IK solve and bounds the worst case to 17 audits.
        """

        reserve = self.clearance_contract.solver_discretization_reserve_m
        maximum = min(
            0.005, self.clearance_contract.maximum_vertical_contact_frame_shift_m
        )
        diagonal = 1.0 / math.sqrt(2.0)
        unit_directions = (
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (diagonal, diagonal),
            (diagonal, -diagonal),
            (-diagonal, diagonal),
            (-diagonal, -diagonal),
        )
        directional_results = []
        for direction_x, direction_y in unit_directions:
            shifted = replace(
                prepared,
                payload=_derive_order9_r1_planar_contact_frame_nominal(
                    prepared.payload,
                    translation_world_m=(
                        reserve * direction_x,
                        reserve * direction_y,
                    ),
                ),
            )
            self._append_complete_path_audit(case, shifted, evaluated)
            audit = evaluated[-1][1]
            translation = (reserve * direction_x, reserve * direction_y)
            audit.update(
                {
                    "generation_method": "analytic_planar_contact_frame_shift_v13",
                    "planar_contact_frame_shift_world_m": list(translation),
                    "original_deterministic_ik_solution_rigidly_preserved": True,
                }
            )
            if audit.get("accepted") is True:
                return
            directional_results.append(
                (
                    float(
                        audit.get(
                            "minimum_robot_support_clearance_lower_bound_m",
                            -math.inf,
                        )
                    ),
                    direction_x,
                    direction_y,
                )
            )

        _clearance, best_x, best_y = max(directional_results)
        step_count = int(math.floor((maximum + 1.0e-12) / reserve))
        for index in range(2, step_count + 1):
            magnitude = index * reserve
            translation = (magnitude * best_x, magnitude * best_y)
            shifted = replace(
                prepared,
                payload=_derive_order9_r1_planar_contact_frame_nominal(
                    prepared.payload,
                    translation_world_m=translation,
                ),
            )
            self._append_complete_path_audit(case, shifted, evaluated)
            audit = evaluated[-1][1]
            audit.update(
                {
                    "generation_method": "analytic_planar_contact_frame_shift_v13",
                    "planar_contact_frame_shift_world_m": list(translation),
                    "original_deterministic_ik_solution_rigidly_preserved": True,
                }
            )
            if audit.get("accepted") is True:
                return


def _derive_order9_r1_planar_contact_frame_nominal(
    nominal,
    *,
    translation_world_m: tuple[float, float],
):
    """Rigidly translate the robot/contact frame in the world XY plane."""

    dx, dy = (float(value) for value in translation_world_m)
    if (
        not all(math.isfinite(value) for value in (dx, dy))
        or math.hypot(dx, dy) <= 0.0
        or math.hypot(dx, dy) > 0.005 + 1.0e-12
    ):
        raise ValueError("R1 planar contact-frame shift must be in (0, 5 mm]")
    contact_windows = tuple(
        value for value in nominal.windows if value.phase == "contact_acquisition"
    )
    if not contact_windows:
        raise SchemaValidationError("R1 planar shift lacks contact acquisition")
    phase_start = float(contact_windows[0].global_start_time_s)
    phase_end = float(
        contact_windows[-1].global_start_time_s
        + contact_windows[-1].plan.trajectory.horizon_s
    )
    phase_span = max(phase_end - phase_start, 1.0e-12)

    windows = []
    for window in nominal.windows:
        window_start = float(window.global_start_time_s)
        window_end = window_start + float(window.plan.trajectory.horizon_s)
        if window_end <= phase_start + 1.0e-12:
            windows.append(window)
            continue
        if window_start >= phase_end - 1.0e-12:
            start = end = 1.0
        else:
            start = min(1.0, max(0.0, (window_start - phase_start) / phase_span))
            end = min(1.0, max(0.0, (window_end - phase_start) / phase_span))
        windows.append(
            replace(
                window,
                plan=_translate_order9_r1_plan_xy(
                    window.plan,
                    dx=dx,
                    dy=dy,
                    start_fraction=start,
                    end_fraction=end,
                ),
            )
        )

    selection = nominal.selection_bundle
    selected_ids = _selected_group_candidate_ids(selection)
    candidates_payload = selection.contact_candidate_set.to_dict()
    shifted_count = 0
    for candidate in candidates_payload["candidates"]:
        if int(candidate["candidate_id"]) not in selected_ids:
            continue
        for key in ("contact_pose_world", "contact_frame_world"):
            candidate[key][0] += dx
            candidate[key][1] += dy
        scores = dict(candidate.get("candidate_scores", {}))
        scores["r1_planar_contact_frame_shift_world_x_m"] = dx
        scores["r1_planar_contact_frame_shift_world_y_m"] = dy
        candidate["candidate_scores"] = scores
        shifted_count += 1
    if shifted_count != len(selected_ids):
        raise SchemaValidationError("R1 planar shift candidate identity differs")
    shifted_candidates = type(selection.contact_candidate_set).from_dict(
        candidates_payload
    )
    shifted_candidates.validate()
    shifted_selection = replace(
        selection,
        contact_candidate_set=shifted_candidates,
        trajectory_plan=_translate_order9_r1_plan_xy(
            selection.trajectory_plan,
            dx=dx,
            dy=dy,
            start_fraction=0.0,
            end_fraction=1.0,
        ),
    )
    final_payload = nominal.final_observation.to_dict()
    for state in final_payload.get("module_states", []):
        state["pose_world"][0] += dx
        state["pose_world"][1] += dy
    final_observation = type(nominal.final_observation).from_dict(final_payload)
    final_observation.validate()
    shifted_windows = tuple(windows)
    return replace(
        nominal,
        selection_bundle=shifted_selection,
        windows=shifted_windows,
        timeline=_flatten_nominal_windows(shifted_windows),
        final_observation=final_observation,
    )


def _translate_order9_r1_plan_xy(
    plan,
    *,
    dx: float,
    dy: float,
    start_fraction: float,
    end_fraction: float,
):
    def translate_trajectory(trajectory):
        payload = trajectory.to_dict()
        horizon = max(float(trajectory.horizon_s), 1.0e-12)
        for knot in payload["knots"]:
            local = min(1.0, max(0.0, float(knot["t_rel_s"]) / horizon))
            smooth = local * local * (3.0 - 2.0 * local)
            fraction = start_fraction + (end_fraction - start_fraction) * smooth
            centroidal = knot.get("centroidal_target")
            if isinstance(centroidal, dict) and centroidal.get("com_pos_world"):
                centroidal["com_pos_world"][0] += dx * fraction
                centroidal["com_pos_world"][1] += dy * fraction
            posture = knot.get("posture_target")
            targets = (
                posture.get("free_anchor_pose_targets")
                if isinstance(posture, dict)
                else None
            )
            if isinstance(targets, dict):
                for pose in targets.values():
                    pose[0] += dx * fraction
                    pose[1] += dy * fraction
        translated = type(trajectory).from_dict(payload)
        translated.validate()
        return translated

    raw = translate_trajectory(plan.raw_trajectory)
    resolved = translate_trajectory(plan.trajectory)
    resolution = plan.posture_resolution
    evidence = replace(
        resolution.evidence,
        resolver_version=(
            resolution.evidence.resolver_version + "+r1_planar_contact_frame_shift_v13"
        ),
        raw_trajectory_hash=stable_hash(raw.to_dict()),
        resolved_trajectory_hash=stable_hash(resolved.to_dict()),
    )
    shifted_resolution = replace(
        resolution,
        raw_trajectory=raw,
        trajectory=resolved,
        evidence=evidence,
    )
    final_dx = dx * end_fraction
    final_dy = dy * end_fraction

    def translate_pose(pose, x, y):
        return (float(pose[0]) + x, float(pose[1]) + y, *pose[2:])

    ik = plan.ik_solution
    shifted_ik = replace(
        ik,
        base_pose_world=translate_pose(ik.base_pose_world, final_dx, final_dy),
        centroidal_pose_world=translate_pose(
            ik.centroidal_pose_world, final_dx, final_dy
        ),
        anchor_poses_world={
            anchor_id: translate_pose(pose, final_dx, final_dy)
            for anchor_id, pose in ik.anchor_poses_world.items()
        },
        solver_version=ik.solver_version + "+rigid_planar_translation_preserved_v13",
    )
    configuration_plan = plan.configuration_space_plan
    if configuration_plan is not None:
        states = []
        denominator = max(len(configuration_plan.states) - 1, 1)
        for index, state in enumerate(configuration_plan.states):
            fraction = start_fraction + (
                (end_fraction - start_fraction) * index / denominator
            )
            states.append(
                replace(
                    state,
                    base_pose_world=translate_pose(
                        state.base_pose_world,
                        dx * fraction,
                        dy * fraction,
                    ),
                )
            )
        configuration_plan = replace(configuration_plan, states=tuple(states))
    return replace(
        plan,
        raw_trajectory=raw,
        trajectory=resolved,
        ik_solution=shifted_ik,
        posture_resolution=shifted_resolution,
        configuration_space_plan=configuration_plan,
        teacher_version=plan.teacher_version + "+r1_planar_contact_frame_shift_v13",
    )


def _load_human_posture_rejections(
    path: Path,
    *,
    repository: Path,
) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    manifest = (repository / str(payload.get("bucket_manifest_path", ""))).resolve()
    if (
        payload.get("rejection_version")
        != "order9_c3_human_posture_rejection_v1"
        or not isinstance(entries, list)
        or repository not in manifest.parents
        or not manifest.is_file()
        or payload.get("bucket_manifest_sha256") != _sha256(manifest)
    ):
        raise SchemaValidationError("R1 v13 human posture rejection binding differs")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    bucket_hashes = {
        str(value.get("bucket_id")): str(value.get("structural_hash"))
        for value in manifest_payload.get("buckets", ())
        if isinstance(value, dict)
    }
    result: dict[str, dict] = {}
    for value in entries:
        if not isinstance(value, dict):
            raise SchemaValidationError("R1 v13 human posture rejection entry differs")
        source_bucket_id = str(value.get("bucket_id", ""))
        structural_hash = str(value.get("structural_hash", ""))
        raw_pairs = value.get("excluded_surface_port_id_pairs")
        reason = value.get("reason")
        if (
            not source_bucket_id
            or source_bucket_id in result
            or bucket_hashes.get(source_bucket_id) != structural_hash
            or not isinstance(raw_pairs, list)
            or not raw_pairs
            or not isinstance(reason, str)
            or not reason
        ):
            raise SchemaValidationError("R1 v13 human posture rejection entry differs")
        pairs = []
        identities = set()
        for raw_pair in raw_pairs:
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or len({int(port) for port in raw_pair}) != 2
                or min(int(port) for port in raw_pair) < 0
            ):
                raise SchemaValidationError("R1 v13 human posture rejection pair differs")
            pair = tuple(int(port) for port in raw_pair)
            identity = frozenset(pair)
            if identity in identities:
                raise SchemaValidationError("R1 v13 human posture rejection pair repeats")
            identities.add(identity)
            pairs.append(pair)
        result[source_bucket_id] = {
            "structural_hash": structural_hash,
            "excluded_surface_port_id_pairs": tuple(pairs),
            "reason": reason,
        }
    return result


def _candidate_source_bucket_id(
    candidate_id: str,
    exclusions: dict[str, tuple[tuple[int, int], ...]],
) -> str | None:
    matches = tuple(
        source_bucket_id
        for source_bucket_id in exclusions
        if f"__{source_bucket_id}__" in str(candidate_id)
    )
    if len(matches) > 1:
        raise SchemaValidationError("R1 v13 candidate source identity is ambiguous")
    return None if not matches else matches[0]


def _assert_pair_allowed(
    source_bucket_id: str,
    pair,
    *,
    exclusions: dict[str, tuple[tuple[int, int], ...]],
    label: str,
) -> None:
    identity = frozenset(int(port) for port in pair)
    forbidden = {
        frozenset(int(port) for port in value)
        for value in exclusions.get(source_bucket_id, ())
    }
    if identity in forbidden:
        raise SchemaValidationError(
            f"R1 v13 {label} reintroduces a human-rejected contact pair: "
            f"{source_bucket_id}:{sorted(identity)}"
        )


def _reject_forbidden_fast_paths(
    values: dict,
    *,
    exclusions: dict[str, tuple[tuple[int, int], ...]],
) -> None:
    for candidate_id, options in values.items():
        source_bucket_id = _candidate_source_bucket_id(candidate_id, exclusions)
        if source_bucket_id is None:
            continue
        for option in options:
            _assert_pair_allowed(
                source_bucket_id,
                option[0],
                exclusions=exclusions,
                label="teacher fast path",
            )


def _filter_forbidden_surface_overrides(
    values: dict,
    *,
    exclusions: dict[str, tuple[tuple[int, int], ...]],
) -> dict:
    result = {}
    for source_bucket_id, pairs in values.items():
        forbidden = {
            frozenset(int(port) for port in value)
            for value in exclusions.get(source_bucket_id, ())
        }
        allowed = tuple(
            pair
            for pair in pairs
            if frozenset(int(port) for port in pair) not in forbidden
        )
        if allowed:
            result[source_bucket_id] = allowed
    return result


def _filter_forbidden_candidate_overrides(
    values: dict,
    *,
    exclusions: dict[str, tuple[tuple[int, int], ...]],
) -> dict:
    result = {}
    for candidate_id, options in values.items():
        source_bucket_id = _candidate_source_bucket_id(candidate_id, exclusions)
        if source_bucket_id is None:
            result[candidate_id] = options
            continue
        forbidden = {
            frozenset(int(port) for port in value)
            for value in exclusions[source_bucket_id]
        }
        allowed = tuple(
            option
            for option in options
            if frozenset(int(port) for port in option[0]) not in forbidden
        )
        if allowed:
            result[candidate_id] = allowed
    return result


def _load_configuration_goal_seeds(
    path: Path,
    *,
    repository: Path,
) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repairs = payload.get("candidate_repairs")
    automatic = payload.get("automatic_configuration_goal_seed_overrides")
    if not isinstance(repairs, dict) or not isinstance(automatic, dict):
        raise SchemaValidationError("R1 v13 configuration-goal seed config differs")
    if set(repairs).intersection(automatic):
        raise SchemaValidationError(
            "R1 v13 automatic configuration-goal seed duplicates a repair"
        )
    result = {}
    for candidate_id, value in {**repairs, **automatic}.items():
        if not isinstance(value, dict):
            raise SchemaValidationError("R1 v13 configuration-goal seed entry differs")
        binding = value.get("configuration_goal_seed_trajectory")
        if binding is None:
            continue
        source = (repository / str((binding or {}).get("path", ""))).resolve()
        if candidate_id in automatic:
            source_candidate_ids_value = value.get("source_candidate_ids")
            if source_candidate_ids_value is None:
                source_candidate_ids = (value.get("source_candidate_id"),)
            elif (
                isinstance(source_candidate_ids_value, list)
                and len(source_candidate_ids_value) == 2
            ):
                source_candidate_ids = tuple(source_candidate_ids_value)
            else:
                source_candidate_ids = ()
            if (
                not source_candidate_ids
                or len(set(source_candidate_ids)) != len(source_candidate_ids)
                or any(
                    not isinstance(source_candidate_id, str)
                    or source_candidate_id == candidate_id
                    or (
                        f"__train__{ORDER9_R1_TRAIN027_SOURCE_BUCKET_ID}__"
                        not in source_candidate_id
                    )
                    for source_candidate_id in source_candidate_ids
                )
                or (
                    f"__train__{ORDER9_R1_TRAIN027_SOURCE_BUCKET_ID}__"
                    not in candidate_id
                )
                or (
                    len(source_candidate_ids) == 1
                    and f"/{source_candidate_ids[0]}/" not in source.as_posix()
                )
            ):
                raise SchemaValidationError(
                    "R1 v13 automatic configuration-goal seed source differs"
                )
        if (
            value.get("configuration_goal_seed_scope")
            != "configuration_space_goal_initialization_only"
            or repository not in source.parents
            or not source.is_file()
            or (binding or {}).get("sha256") != _sha256(source)
        ):
            raise SchemaValidationError("R1 v13 configuration-goal seed binding differs")
        trajectory = json.loads(source.read_text(encoding="utf-8"))
        if candidate_id in automatic and len(source_candidate_ids) == 2:
            source_bindings = (
                trajectory.get("source_trajectories")
                if isinstance(trajectory, dict)
                else None
            )
            if (
                trajectory.get("seed_version")
                != "order9_r1_interpolated_configuration_goal_seed_v1"
                or trajectory.get("candidate_id") != candidate_id
                or trajectory.get("configuration_goal_seed_scope")
                != "configuration_space_goal_initialization_only"
                or trajectory.get("contact_goal_seed_applied") is not False
                or trajectory.get("contact_ik_constraints_changed") is not False
                or trajectory.get("acceptance_or_safety_gate_changed") is not False
                or not isinstance(source_bindings, list)
                or len(source_bindings) != 2
            ):
                raise SchemaValidationError(
                    "R1 v13 interpolated configuration-goal seed differs"
                )
            for source_candidate_id, source_binding in zip(
                source_candidate_ids, source_bindings, strict=True
            ):
                source_trajectory = (
                    repository / str((source_binding or {}).get("path", ""))
                ).resolve()
                if (
                    repository not in source_trajectory.parents
                    or not source_trajectory.is_file()
                    or f"/{source_candidate_id}/"
                    not in source_trajectory.as_posix()
                    or (source_binding or {}).get("sha256")
                    != _sha256(source_trajectory)
                ):
                    raise SchemaValidationError(
                        "R1 v13 interpolated seed source binding differs"
                    )
        knots = trajectory.get("knots") if isinstance(trajectory, dict) else None
        posture = (
            knots[-1].get("posture_target")
            if isinstance(knots, list) and knots and isinstance(knots[-1], dict)
            else None
        )
        positions = (
            posture.get("joint_pos_target") if isinstance(posture, dict) else None
        )
        if (
            not isinstance(positions, dict)
            or not positions
            or any(
                not isinstance(joint_id, str)
                or not joint_id
                or isinstance(position, bool)
                or not isinstance(position, (int, float))
                or not math.isfinite(float(position))
                for joint_id, position in positions.items()
            )
        ):
            raise SchemaValidationError("R1 v13 configuration-goal seed content differs")
        result[str(candidate_id)] = {
            joint_id: float(position) for joint_id, position in positions.items()
        }
    return result


def _load_configuration_goal_seed_evidence(
    path: Path,
    *,
    repository: Path,
) -> dict[str, dict]:
    seeds = _load_configuration_goal_seeds(path, repository=repository)
    payload = json.loads(path.read_text(encoding="utf-8"))
    repairs = payload["candidate_repairs"]
    automatic = payload["automatic_configuration_goal_seed_overrides"]
    result = {}
    for candidate_id, seed in seeds.items():
        repair = repairs.get(candidate_id)
        automatic_seed = automatic.get(candidate_id)
        if repair is None and automatic_seed is None:
            raise SchemaValidationError("R1 v13 configuration-goal seed source differs")
        value = repair if repair is not None else automatic_seed
        confirmed_isaac_reseed = (
            repair is not None
            and repair.get("repair_kind")
            == "confirmed_isaac_collision_teacher_option_reselection"
        )
        evidence = {
            "record_version": "order9_r1_configuration_goal_seed_evidence_v1",
            "candidate_id": candidate_id,
            "scope": "configuration_space_goal_initialization_only",
            "source_trajectory": dict(
                value["configuration_goal_seed_trajectory"]
            ),
            "joint_seed_hash": stable_hash(seed),
            "contact_goal_seed_applied": False,
            "contact_ik_constraints_changed": False,
            "acceptance_or_safety_gate_changed": False,
            "isaac_invoked_during_seed_selection": confirmed_isaac_reseed,
            "controller_layers_invoked_during_seed_selection": (
                confirmed_isaac_reseed
            ),
        }
        if automatic_seed is not None:
            evidence["automatic_source_configuration_seed_reuse"] = True
            if "source_candidate_ids" in automatic_seed:
                evidence["source_seed_candidate_ids"] = list(
                    automatic_seed["source_candidate_ids"]
                )
            else:
                evidence["source_seed_candidate_id"] = automatic_seed[
                    "source_candidate_id"
                ]
        if confirmed_isaac_reseed:
            evidence["source_confirmed_isaac_failure"] = dict(
                repair["source_confirmed_isaac_failure"]
            )
        result[candidate_id] = evidence
    return result


def _load_bounded_local_contact_repairs(
    path: Path,
    *,
    repository: Path,
) -> frozenset[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repairs = payload.get("candidate_repairs")
    if (
        payload.get("repair_version") != "order9_r1_range_teacher_repairs_v13"
        or int(payload.get("bounded_local_contact_maximum_iterations", -1)) != 160
        or not isinstance(repairs, dict)
        or not repairs
    ):
        raise SchemaValidationError("R1 v13 teacher repair config differs")
    result = set()
    for candidate_id, value in repairs.items():
        if not isinstance(value, dict):
            raise SchemaValidationError("R1 v13 teacher repair entry differs")
        repair_kind = value.get("repair_kind")
        binding_name = {
            "confirmed_isaac_collision_teacher_option_reselection": (
                "source_confirmed_isaac_failure"
            ),
            "confirmed_cpu_teacher_option_selection": "source_confirmed_cpu_success",
        }.get(repair_kind, "source_decisive_checkpoint")
        binding = value.get(binding_name)
        source = (repository / str((binding or {}).get("path", ""))).resolve()
        if (
            not candidate_id
            or repair_kind
            not in {
                "bounded_deterministic_local_contact_search",
                "confirmed_isaac_collision_teacher_option_reselection",
                "confirmed_cpu_teacher_option_selection",
            }
            or repository not in source.parents
            or not source.is_file()
            or (binding or {}).get("sha256") != _sha256(source)
            or float(value.get("acceptance_clearance_m", -1.0)) != 0.0195
            or float(value.get("planning_clearance_m", -1.0)) != 0.0300
        ):
            raise SchemaValidationError("R1 v13 teacher repair evidence differs")
        if repair_kind == "confirmed_isaac_collision_teacher_option_reselection":
            _validate_confirmed_isaac_collision_repair(
                candidate_id,
                value,
                source=source,
                repository=repository,
            )
            result.add(candidate_id)
            continue
        if repair_kind == "confirmed_cpu_teacher_option_selection":
            _validate_confirmed_cpu_teacher_option(
                candidate_id,
                value,
                source=source,
                repository=repository,
            )
            result.add(candidate_id)
            continue
        if (
            value.get("isaac_invoked_during_probe") is not False
            or value.get("controller_layers_invoked_during_probe") is not False
        ):
            raise SchemaValidationError("R1 v13 teacher repair evidence differs")
        decisive = json.loads(source.read_text(encoding="utf-8"))
        record = decisive.get("decisive_record", {})
        failure_reason = str(record.get("preparation_failure_reason", ""))
        if (
            record.get("candidate_id") != candidate_id
            or record.get("prepared") is not False
            or not (
                "sampled local contact search is disabled" in failure_reason
                or "sampled overhead search is disabled" in failure_reason
                or "joint-reserve projection failed collision recheck" in failure_reason
                or "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM" in failure_reason
                or "configuration-space teacher has no multi-obstacle collision-free goal"
                in failure_reason
            )
        ):
            raise SchemaValidationError("R1 v13 teacher repair source differs")
        result.add(candidate_id)
    return frozenset(result)


def _validate_confirmed_cpu_teacher_option(
    candidate_id: str,
    value: dict,
    *,
    source: Path,
    repository: Path,
) -> None:
    """Bind a finite teacher option that passed the complete CPU screen."""

    option = value.get("teacher_candidate_override")
    if (
        value.get("isaac_invoked_during_probe") is not False
        or value.get("controller_layers_invoked_during_probe") is not False
        or value.get("configuration_goal_seed_scope")
        != "configuration_space_goal_initialization_only"
        or value.get("contact_goal_seed_applied") is not False
        or value.get("contact_ik_constraints_changed") is not False
        or value.get("acceptance_or_safety_gate_changed") is not False
        or not isinstance(option, dict)
    ):
        raise SchemaValidationError("R1 confirmed CPU teacher option scope differs")

    evidence = json.loads(source.read_text(encoding="utf-8"))
    probe_result = evidence.get("probe_result")
    seed_binding = evidence.get("configuration_goal_seed_trajectory")
    seed_source = (
        repository / str((seed_binding or {}).get("path", ""))
    ).resolve()
    if (
        evidence.get("record_version")
        != "order9_r1_confirmed_cpu_teacher_option_v1"
        or evidence.get("candidate_id") != candidate_id
        or evidence.get("teacher_candidate_override") != option
        or evidence.get("configuration_goal_seed_scope")
        != "configuration_space_goal_initialization_only"
        or evidence.get("isaac_invoked_during_probe") is not False
        or evidence.get("controller_layers_invoked_during_probe") is not False
        or evidence.get("contact_goal_seed_applied") is not False
        or evidence.get("contact_ik_constraints_changed") is not False
        or evidence.get("acceptance_or_safety_gate_changed") is not False
        or float(evidence.get("acceptance_clearance_m", -1.0)) != 0.0195
        or float(evidence.get("planning_clearance_m", -1.0)) != 0.0300
        or not isinstance(probe_result, dict)
        or probe_result.get("teacher_trajectory_complete") is not True
        or probe_result.get("screen_accepted") is not True
        or probe_result.get("eligible_for_full_control_test") is not True
        or probe_result.get("violation_codes") != []
        or probe_result.get("failure_reason") is not None
        or repository not in seed_source.parents
        or not seed_source.is_file()
        or (seed_binding or {}).get("sha256") != _sha256(seed_source)
    ):
        raise SchemaValidationError("R1 confirmed CPU teacher option evidence differs")

    for binding_name in ("probe_runtime",):
        binding = evidence.get(binding_name)
        bound_source = (
            repository / str((binding or {}).get("path", ""))
        ).resolve()
        if (
            repository not in bound_source.parents
            or not bound_source.is_file()
            or (binding or {}).get("sha256") != _sha256(bound_source)
        ):
            raise SchemaValidationError(
                f"R1 confirmed CPU teacher option {binding_name} differs"
            )
    if evidence.get("teacher_pipeline") not in (
        {
            "path": "amsrr/training/order9_r1_range_selection_v13.py",
            "sha256": "a4b7c1bc4eb264296b734d6505d36b410002e648126298243bad74c0086b24af",
        },
        {
            "path": "amsrr/training/order9_r1_range_selection_v13.py",
            "sha256": "878972b8246e3a4f0f4678c2b8c419c1928d70120f7a2a2529965f840f9e42b2",
        },
        {
            "path": "amsrr/training/order9_r1_range_selection_v13.py",
            "sha256": "d7f4556a5bc4cf1907d4254f79c8d78ace3d4d005ee8f448183ae4d1d4bc6a08",
        },
    ):
        raise SchemaValidationError(
            "R1 confirmed CPU teacher option historical pipeline differs"
        )


def _validate_confirmed_isaac_collision_repair(
    candidate_id: str,
    value: dict,
    *,
    source: Path,
    repository: Path,
) -> None:
    """Require two independent, controller-backed collision observations.

    A trajectory that passed the analytic screen may only be reseeded after a
    bounded Isaac batch and a separate single-candidate replay both report the
    same hard collision.  The reseed changes configuration-space goal
    initialization only; contact IK and every acceptance/safety gate remain
    unchanged.
    """

    if (
        value.get("isaac_invoked_during_probe") is not True
        or value.get("controller_layers_invoked_during_probe") is not True
        or value.get("configuration_goal_seed_scope")
        != "configuration_space_goal_initialization_only"
        or not isinstance(value.get("teacher_candidate_override"), dict)
        or value.get("contact_goal_seed_applied") is not False
        or value.get("contact_ik_constraints_changed") is not False
        or value.get("acceptance_or_safety_gate_changed") is not False
    ):
        raise SchemaValidationError("R1 confirmed Isaac repair scope differs")

    evidence = json.loads(source.read_text(encoding="utf-8"))
    contract = evidence.get("controller_contract")
    if (
        evidence.get("record_version")
        != "order9_r1_confirmed_isaac_teacher_collision_v1"
        or evidence.get("candidate_id") != candidate_id
        or evidence.get("failure_kind")
        != "repeated_non_grasp_body_object_hard_collision"
        or evidence.get("independent_single_candidate_replay") is not True
        or evidence.get("repair_scope")
        != "teacher_candidate_reselection_and_configuration_space_goal_initialization"
        or evidence.get("contact_goal_seed_applied") is not False
        or evidence.get("contact_ik_constraints_changed") is not False
        or evidence.get("acceptance_or_safety_gate_changed") is not False
        or not isinstance(contract, dict)
        or contract.get("pi_l_actor_command_applied") is not False
        or contract.get("qpid_qp_applied") is not True
        or contract.get("local_servo_applied") is not True
    ):
        raise SchemaValidationError("R1 confirmed Isaac collision evidence differs")

    option = value["teacher_candidate_override"]
    ports = option.get("selected_surface_port_ids")
    tangent = option.get("tangent_offset_world_m")
    if (
        not isinstance(ports, list)
        or len(ports) != 2
        or any(isinstance(port, bool) or not isinstance(port, int) for port in ports)
        or not isinstance(option.get("candidate_group_id"), str)
        or not option["candidate_group_id"]
        or option.get("candidate_group_id")
        != evidence.get("replacement_candidate_group_id")
        or ports != evidence.get("selected_surface_port_ids")
        or not isinstance(tangent, list)
        or len(tangent) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in tangent
        )
        or any(
            isinstance(option.get(name), bool)
            or not isinstance(option.get(name), (int, float))
            or not math.isfinite(float(option[name]))
            or float(option[name]) <= 0.0
            for name in (
                "pregrasp_clearance_m",
                "collision_margin_m",
                "grasp_contact_height_offset_m",
            )
        )
    ):
        raise SchemaValidationError("R1 confirmed Isaac teacher option differs")

    for evidence_name in ("bounded_batch_failure", "single_replay_failure"):
        binding = evidence.get(evidence_name)
        path = (repository / str((binding or {}).get("path", ""))).resolve()
        if (
            repository not in path.parents
            or not path.is_file()
            or (binding or {}).get("sha256") != _sha256(path)
        ):
            raise SchemaValidationError("R1 confirmed Isaac episode binding differs")
        try:
            episodes = tuple(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (json.JSONDecodeError, OSError) as error:
            raise SchemaValidationError(
                "R1 confirmed Isaac episode evidence is unreadable"
            ) from error
        generation_id = f"r1_nominal_calibration:{candidate_id}"
        if len(episodes) != 2 or any(
            episode.get("task_success") is not False
            or episode.get("safety_failure") is not True
            or episode.get("failure_reason") != "hard_collision"
            or int(episode.get("fallback_decision_count", -1)) != 0
            or episode.get("isaac_backed") is not True
            or (episode.get("metadata") or {}).get("deterministic_policy") is not True
            or (episode.get("metadata") or {}).get("generation_id") != generation_id
            or float((episode.get("metrics") or {}).get("hard_collision", 0.0))
            != 1.0
            for episode in episodes
        ):
            raise SchemaValidationError("R1 confirmed Isaac collision replay differs")

    for manifest_name in ("bounded_batch_manifest", "single_replay_manifest"):
        binding = evidence.get(manifest_name)
        path = (repository / str((binding or {}).get("path", ""))).resolve()
        if (
            repository not in path.parents
            or not path.is_file()
            or (binding or {}).get("sha256") != _sha256(path)
        ):
            raise SchemaValidationError("R1 confirmed Isaac manifest binding differs")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        matching_jobs = tuple(
            job
            for job in manifest.get("jobs", ())
            if isinstance(job, dict) and job.get("name") == candidate_id
        )
        argv = matching_jobs[0].get("argv") if len(matching_jobs) == 1 else None
        episode_count_index = (
            argv.index("--evaluation-episode-count")
            if isinstance(argv, list) and "--evaluation-episode-count" in argv
            else -1
        )
        if (
            not isinstance(argv, list)
            or "--diagnostic-nominal-qpid-only" not in argv
            or episode_count_index < 0
            or episode_count_index + 1 >= len(argv)
            or argv[episode_count_index + 1] != "2"
        ):
            raise SchemaValidationError("R1 confirmed Isaac controller manifest differs")


def _load_confirmed_teacher_option_overrides(
    path: Path,
) -> dict[str, tuple]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repairs = payload.get("candidate_repairs")
    if not isinstance(repairs, dict):
        raise SchemaValidationError("R1 confirmed teacher options differ")
    result = {}
    for candidate_id, value in repairs.items():
        if not isinstance(value, dict) or value.get("repair_kind") not in {
            "confirmed_isaac_collision_teacher_option_reselection",
            "confirmed_cpu_teacher_option_selection",
        }:
            continue
        option = value.get("teacher_candidate_override")
        if not isinstance(option, dict):
            raise SchemaValidationError("R1 confirmed teacher option differs")
        result[candidate_id] = (
            tuple(int(port) for port in option["selected_surface_port_ids"]),
            str(option["candidate_group_id"]),
            float(option["pregrasp_clearance_m"]),
            float(option["collision_margin_m"]),
            float(option["grasp_contact_height_offset_m"]),
            tuple(float(item) for item in option["tangent_offset_world_m"]),
        )
    return result


def _load_contact_acquisition_leads(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("candidate_fast_paths")
    if not isinstance(values, dict):
        raise SchemaValidationError("R1 range contact phase hints differ")
    result = {}
    for candidate_id, value in values.items():
        if not isinstance(value, dict):
            continue
        lead = value.get("contact_acquisition_lead_windows")
        if lead is None:
            continue
        if isinstance(lead, bool) or int(lead) != 1:
            raise SchemaValidationError("R1 range contact phase lead differs")
        result[str(candidate_id)] = int(lead)
    return result


def _resegment_order9_r1_contact_acquisition(
    nominal,
    *,
    lead_window_count: int,
):
    """Move a bounded approach suffix into contact acquisition unchanged."""

    approach_indices = [
        index
        for index, window in enumerate(nominal.windows)
        if window.phase == "approach"
    ]
    contact_indices = [
        index
        for index, window in enumerate(nominal.windows)
        if window.phase == "contact_acquisition"
    ]
    if (
        len(approach_indices) <= lead_window_count
        or not contact_indices
        or approach_indices[-1] + 1 != contact_indices[0]
    ):
        raise SchemaValidationError("R1 contact phase resegmentation differs")
    moved = frozenset(approach_indices[-lead_window_count:])
    contact_template = nominal.windows[contact_indices[0]].plan
    windows = tuple(
        (
            replace(
                window,
                phase="contact_acquisition",
                plan=_relabel_order9_r1_plan_contact_phase(
                    window.plan,
                    contact_template=contact_template,
                ),
            )
            if index in moved
            else window
        )
        for index, window in enumerate(nominal.windows)
    )
    return replace(
        nominal,
        windows=windows,
        timeline=_flatten_nominal_windows(windows),
    )


def _relabel_order9_r1_plan_contact_phase(plan, *, contact_template):
    def relabel(trajectory, template_trajectory):
        payload = trajectory.to_dict()
        template_assignments = copy.deepcopy(
            template_trajectory.to_dict()["knots"][0]["contact_assignments"]
        )
        # Carry the complete contact-frame bounds required by the runtime.
        # The configured lead is restricted to the final approach window so
        # that the selected gripper contact becomes legal immediately before
        # physical acquisition without moving the rest of approach into the
        # active-contact controller state.
        for assignment in template_assignments:
            assignment["schedule_state"] = "attach"
        for knot in payload["knots"]:
            knot["contact_assignments"] = copy.deepcopy(template_assignments)
            for guard in knot.get("guard_conditions", []):
                if guard.get("phase") == "approach":
                    guard["phase"] = "contact_acquisition"
        value = type(trajectory).from_dict(payload)
        value.derived_mode_label = (
            f"{trajectory.derived_mode_label or 'unspecified'}:"
            "r1_contact_phase_lead_v13"
        )
        value.validate()
        return value

    raw = relabel(plan.raw_trajectory, contact_template.raw_trajectory)
    resolved = relabel(plan.trajectory, contact_template.trajectory)
    resolution = plan.posture_resolution
    evidence = replace(
        resolution.evidence,
        resolver_version=(
            resolution.evidence.resolver_version + "+r1_contact_phase_lead_v13"
        ),
        raw_trajectory_hash=stable_hash(raw.to_dict()),
        resolved_trajectory_hash=stable_hash(resolved.to_dict()),
    )
    return replace(
        plan,
        raw_trajectory=raw,
        trajectory=resolved,
        posture_resolution=replace(
            resolution,
            raw_trajectory=raw,
            trajectory=resolved,
            evidence=evidence,
        ),
        teacher_version=plan.teacher_version + "+r1_contact_phase_lead_v13",
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE",
    "ORDER9_R1_RANGE_SELECTION_V13_VERSION",
    "Order9R1ExactMarginTeacherScreenPipelineV13",
]
