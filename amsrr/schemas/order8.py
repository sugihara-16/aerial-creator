from __future__ import annotations

"""Versioned contracts for the Order 8 natural-contact evidence boundary.

The raw-contact records in this module are privileged simulator truth used for
acceptance diagnostics (and, later, critic/reward construction).  They are not
part of the normal actor observation and are never a QPID command target.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    StrEnum,
    require_len,
    require_non_empty,
)
from amsrr.utils.config import load_config

ORDER8_NATURAL_CONTACT_CONFIG_VERSION = "order8_natural_contact_config_v12"
ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION = "order8_natural_contact_observation_v4"
ORDER8_NATURAL_CONTACT_STEP_EVIDENCE_VERSION = "order8_natural_contact_step_evidence_v4"
ORDER8_NATURAL_CONTACT_RESULT_VERSION = "order8_natural_contact_result_v3"
ORDER8_NATURAL_CONTACT_MODEL = "natural_contact_grasp_v1"
ORDER8_RAW_CONTACT_TRUTH_ROLE = "privileged_diagnostic_only"


class Order8NaturalContactPhase(StrEnum):
    RESET = "reset"
    APPROACH = "approach"
    CONTACT_ACQUISITION = "contact_acquisition"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    SETTLE = "settle"
    COMPLETE = "complete"
    SAFE_HOLD = "safe_hold"


@dataclass
class Order8NaturalContactConfig(SchemaBase):
    config_version: str = ORDER8_NATURAL_CONTACT_CONFIG_VERSION
    contact_model: str = ORDER8_NATURAL_CONTACT_MODEL
    object_mass_kg: float = 1.0
    object_size_m: list[float] = field(default_factory=lambda: [0.30, 0.40, 0.15])
    object_support_height_m: float = 0.15
    initial_object_standoff_m: float = 0.50
    object_friction: float = 0.6
    floor_friction: float = 0.8
    selected_gripper_friction: float = 4.5
    selected_gripper_compliant_contact_stiffness_n_per_m: float = 7500.0
    selected_gripper_compliant_contact_damping_n_s_per_m: float = 75.0
    required_distinct_dock_links: int = 2
    contact_normal_force_threshold_n: float = 0.5
    contact_penetration_noise_floor_m: float = 0.0001
    contact_dwell_s: float = 0.25
    normal_force_target_per_contact_n: float = 11.0
    max_force_per_contact_n: float = 30.0
    max_torque_per_contact_nm: float = 5.0
    max_penetration_m: float = 0.002
    max_tangential_slip_speed_mps: float = 0.02
    max_contact_point_slip_displacement_m: float = 0.030
    contact_break_grace_s: float = 0.05
    minimum_lift_clearance_m: float = 0.100
    downward_drop_velocity_threshold_mps: float = 0.25
    required_transport_distance_m: float = 0.200
    base_translation_speed_limit_mps: float = 0.10
    contact_base_translation_speed_limit_mps: float = 0.010
    payload_load_transfer_s: float = 1.0
    lift_payload_acceleration_mps2: float = 1.0
    lift_acceleration_bias_removal_s: float = 0.5
    hover_dwell_s: float = 2.0
    pregrasp_mesh_clearance_m: float = 0.050
    pregrasp_position_tolerance_m: float = 0.080
    pregrasp_linear_speed_tolerance_mps: float = 0.020
    contact_axial_min_mesh_overlap_m: float = 0.050
    anchor_translation_speed_limit_mps: float = 0.010
    anchor_reference_terminal_tolerance_m: float = 0.001
    anchor_command_tracking_tolerance_m: float = 0.020
    contact_near_surface_slowdown_m: float = 0.015
    contact_surface_arm_clearance_m: float = 0.003
    contact_tangential_tolerance_m: float = 0.050
    contact_surface_creep_speed_limit_mps: float = 0.001
    contact_clearance_sync_deadband_m: float = 0.0005
    contact_clearance_sync_full_slowdown_m: float = 0.0015
    contact_clearance_sync_minimum_speed_scale: float = 0.05
    contact_closure_inward_overtravel_m: float = 0.004
    contact_stall_anchor_speed_threshold_mps: float = 0.0015
    contact_stall_dwell_s: float = 0.10
    contact_centering_max_offset_m: float = 0.030
    contact_centering_max_tilt_rad: float = 0.020
    contact_centering_xy_p_gain: float = 12.0
    contact_centering_xy_d_gain: float = 4.0
    contact_centering_roll_pitch_p_gain: float = 60.0
    contact_centering_roll_pitch_d_gain: float = 24.0
    # The non-privileged mesh/load trigger leads the first raw PhysX contact.
    # These times smooth the contact-admittance state transition; normal
    # centroidal P/I gains remain active so height and attitude are preserved.
    contact_yield_ramp_down_s: float = 0.05
    contact_yield_ramp_up_s: float = 0.50
    contact_yield_integrator_decay_rate_per_s: float = 12.0
    contact_external_wrench_filter_time_constant_s: float = 0.05
    contact_external_wrench_bias_time_constant_s: float = 0.50
    contact_admittance_force_deadband_n: float = 0.5
    contact_admittance_torque_deadband_nm: float = 0.05
    contact_admittance_linear_gain_mps_per_n: float = 0.0015
    contact_admittance_angular_gain_radps_per_nm: float = 0.03
    contact_admittance_max_linear_speed_mps: float = 0.020
    contact_admittance_max_angular_speed_radps: float = 0.15
    contact_admittance_max_translation_offset_m: float = 0.030
    # Retained for backward-compatible report/config parsing.  The active
    # Order-8 path preserves nominal Dock implicit-drive gains and reports this
    # contact-yield drive schedule as disabled.
    contact_yield_joint_drive_stiffness_scale: float = 0.25
    contact_yield_joint_drive_damping_nms_per_rad: float = 8.0
    contact_joint_velocity_limit_radps: float = 0.10
    contact_position_preload_joint_speed_radps: float = 0.002
    contact_position_preload_load_threshold_nm: float = 1.2
    contact_force_ramp_s: float = 40.0
    contact_joint_drive_damping_multiplier: float = 1.0
    contact_acquisition_timeout_s: float = 90.0
    simultaneous_reachability_absolute_tolerance: float = 0.010
    joint_limit_state_tolerance_rad: float = 0.005
    release_contact_free_dwell_s: float = 0.10
    gripper_retreat_clearance_m: float = 0.050
    settle_linear_speed_mps: float = 0.05
    settle_angular_speed_rad_s: float = 0.10
    post_release_settle_dwell_s: float = 1.0
    raw_contact_truth_role: Literal["privileged_diagnostic_only"] = (
        ORDER8_RAW_CONTACT_TRUTH_ROLE
    )
    raw_contact_truth_actor_input: Literal[False] = False
    raw_contact_truth_qpid_command: Literal[False] = False

    def validate(self) -> None:
        if self.config_version != ORDER8_NATURAL_CONTACT_CONFIG_VERSION:
            raise SchemaValidationError(
                "Order8NaturalContactConfig.config_version mismatch"
            )
        if self.contact_model != ORDER8_NATURAL_CONTACT_MODEL:
            raise SchemaValidationError(
                "Order8NaturalContactConfig.contact_model mismatch"
            )
        require_len(self.object_size_m, 3, "Order8NaturalContactConfig.object_size_m")
        if not all(_finite_positive(value) for value in self.object_size_m):
            raise SchemaValidationError(
                "Order8NaturalContactConfig.object_size_m must be finite and positive"
            )
        for name in (
            "object_mass_kg",
            "object_support_height_m",
            "initial_object_standoff_m",
            "object_friction",
            "floor_friction",
            "selected_gripper_friction",
            "selected_gripper_compliant_contact_stiffness_n_per_m",
            "selected_gripper_compliant_contact_damping_n_s_per_m",
            "contact_normal_force_threshold_n",
            "contact_penetration_noise_floor_m",
            "contact_dwell_s",
            "normal_force_target_per_contact_n",
            "max_force_per_contact_n",
            "max_torque_per_contact_nm",
            "max_penetration_m",
            "max_tangential_slip_speed_mps",
            "max_contact_point_slip_displacement_m",
            "contact_break_grace_s",
            "minimum_lift_clearance_m",
            "downward_drop_velocity_threshold_mps",
            "required_transport_distance_m",
            "base_translation_speed_limit_mps",
            "contact_base_translation_speed_limit_mps",
            "payload_load_transfer_s",
            "lift_payload_acceleration_mps2",
            "lift_acceleration_bias_removal_s",
            "hover_dwell_s",
            "pregrasp_mesh_clearance_m",
            "pregrasp_position_tolerance_m",
            "pregrasp_linear_speed_tolerance_mps",
            "contact_axial_min_mesh_overlap_m",
            "anchor_translation_speed_limit_mps",
            "anchor_reference_terminal_tolerance_m",
            "anchor_command_tracking_tolerance_m",
            "contact_near_surface_slowdown_m",
            "contact_surface_arm_clearance_m",
            "contact_tangential_tolerance_m",
            "contact_surface_creep_speed_limit_mps",
            "contact_clearance_sync_deadband_m",
            "contact_clearance_sync_full_slowdown_m",
            "contact_clearance_sync_minimum_speed_scale",
            "contact_closure_inward_overtravel_m",
            "contact_stall_anchor_speed_threshold_mps",
            "contact_stall_dwell_s",
            "contact_centering_max_offset_m",
            "contact_centering_max_tilt_rad",
            "contact_centering_xy_p_gain",
            "contact_centering_xy_d_gain",
            "contact_centering_roll_pitch_p_gain",
            "contact_centering_roll_pitch_d_gain",
            "contact_yield_ramp_down_s",
            "contact_yield_ramp_up_s",
            "contact_yield_integrator_decay_rate_per_s",
            "contact_external_wrench_filter_time_constant_s",
            "contact_external_wrench_bias_time_constant_s",
            "contact_admittance_force_deadband_n",
            "contact_admittance_torque_deadband_nm",
            "contact_admittance_linear_gain_mps_per_n",
            "contact_admittance_angular_gain_radps_per_nm",
            "contact_admittance_max_linear_speed_mps",
            "contact_admittance_max_angular_speed_radps",
            "contact_admittance_max_translation_offset_m",
            "contact_yield_joint_drive_stiffness_scale",
            "contact_yield_joint_drive_damping_nms_per_rad",
            "contact_joint_velocity_limit_radps",
            "contact_position_preload_joint_speed_radps",
            "contact_position_preload_load_threshold_nm",
            "contact_force_ramp_s",
            "contact_joint_drive_damping_multiplier",
            "contact_acquisition_timeout_s",
            "simultaneous_reachability_absolute_tolerance",
            "joint_limit_state_tolerance_rad",
            "release_contact_free_dwell_s",
            "gripper_retreat_clearance_m",
            "settle_linear_speed_mps",
            "settle_angular_speed_rad_s",
            "post_release_settle_dwell_s",
        ):
            if not _finite_positive(getattr(self, name)):
                raise SchemaValidationError(
                    f"Order8NaturalContactConfig.{name} must be finite and positive"
                )
        if self.selected_gripper_friction < self.object_friction:
            raise SchemaValidationError(
                "Order8 selected-gripper friction must be at least the object friction"
            )
        if self.contact_joint_drive_damping_multiplier < 1.0:
            raise SchemaValidationError(
                "Order8 contact joint drive damping multiplier must be at least one"
            )
        if (
            self.contact_position_preload_joint_speed_radps
            > self.contact_joint_velocity_limit_radps
        ):
            raise SchemaValidationError(
                "Order8 position-preload speed must not exceed the contact "
                "joint velocity limit"
            )
        if self.contact_yield_joint_drive_stiffness_scale > 1.0:
            raise SchemaValidationError(
                "Order8 contact-yield joint drive stiffness scale must not exceed one"
            )
        if (
            self.contact_base_translation_speed_limit_mps
            > self.base_translation_speed_limit_mps
        ):
            raise SchemaValidationError(
                "Order8 contact base speed limit must not exceed the general base speed limit"
            )
        if self.pregrasp_mesh_clearance_m >= self.initial_object_standoff_m:
            raise SchemaValidationError(
                "Order8 pregrasp mesh clearance must be smaller than the initial object standoff"
            )
        if (
            not isinstance(self.required_distinct_dock_links, int)
            or isinstance(self.required_distinct_dock_links, bool)
            or self.required_distinct_dock_links < 2
        ):
            raise SchemaValidationError(
                "Order8NaturalContactConfig.required_distinct_dock_links must be an integer >= 2"
            )
        if self.normal_force_target_per_contact_n > self.max_force_per_contact_n:
            raise SchemaValidationError(
                "Order8 normal-force target must not exceed the hard per-contact force limit"
            )
        if self.contact_penetration_noise_floor_m > self.max_penetration_m:
            raise SchemaValidationError(
                "Order8 contact-penetration noise floor must not exceed the hard penetration limit"
            )
        if not (
            self.contact_penetration_noise_floor_m
            < self.contact_surface_arm_clearance_m
            <= self.contact_near_surface_slowdown_m
        ):
            raise SchemaValidationError(
                "Order8 sampled-surface arm clearance must be above the "
                "penetration noise floor and no larger than the near-surface boundary"
            )
        if self.contact_tangential_tolerance_m >= 0.5 * min(self.object_size_m):
            raise SchemaValidationError(
                "Order8 tangential contact tolerance must remain inside every "
                "object-face half extent"
            )
        if (
            self.contact_surface_creep_speed_limit_mps
            > 0.2 * self.anchor_translation_speed_limit_mps
        ):
            raise SchemaValidationError(
                "Order8 surface creep speed must be no larger than the "
                "near-surface speed"
            )
        if (
            self.contact_stall_anchor_speed_threshold_mps
            > self.max_tangential_slip_speed_mps
        ):
            raise SchemaValidationError(
                "Order8 provisional contact-stall speed threshold must not "
                "exceed the raw maintained-contact slip safety limit"
            )
        if not (
            self.contact_clearance_sync_deadband_m
            < self.contact_clearance_sync_full_slowdown_m
            <= self.contact_near_surface_slowdown_m
        ):
            raise SchemaValidationError(
                "Order8 clearance synchronization requires deadband < full "
                "slowdown <= near-surface boundary"
            )
        if self.contact_clearance_sync_minimum_speed_scale > 1.0:
            raise SchemaValidationError(
                "Order8 clearance synchronization minimum speed scale must "
                "not exceed one"
            )
        if (
            self.contact_admittance_max_linear_speed_mps
            > self.max_tangential_slip_speed_mps
        ):
            raise SchemaValidationError(
                "Order8 contact admittance linear speed must not exceed the "
                "maintained-contact slip-speed limit"
            )
        if (
            self.contact_admittance_max_translation_offset_m
            > self.contact_tangential_tolerance_m
        ):
            raise SchemaValidationError(
                "Order8 contact admittance translation offset must remain "
                "inside the approved contact-surface region"
            )
        if self.raw_contact_truth_role != ORDER8_RAW_CONTACT_TRUTH_ROLE:
            raise SchemaValidationError(
                "Order8 raw contact truth must remain privileged diagnostic only"
            )
        if self.raw_contact_truth_actor_input is not False:
            raise SchemaValidationError(
                "Order8 raw contact truth must not be an actor input"
            )
        if self.raw_contact_truth_qpid_command is not False:
            raise SchemaValidationError(
                "Order8 raw contact truth must not be a QPID command"
            )


@dataclass
class Order8RawContactPatch(SchemaBase):
    """One non-aggregated robot/other-body contact patch.

    Force and torque fields are magnitudes.  Keeping non-negative per-patch
    magnitudes makes cancellation by summing opposing vector contacts
    impossible at this interface.
    """

    patch_id: str
    robot_link_id: str
    other_body_id: str
    normal_force_n: float
    force_magnitude_n: float
    torque_magnitude_nm: float
    penetration_m: float
    tangential_slip_speed_mps: float

    def validate(self) -> None:
        for name in ("patch_id", "robot_link_id", "other_body_id"):
            require_non_empty(getattr(self, name), f"Order8RawContactPatch.{name}")
        for name in (
            "normal_force_n",
            "force_magnitude_n",
            "torque_magnitude_nm",
            "penetration_m",
            "tangential_slip_speed_mps",
        ):
            if not _finite_non_negative(getattr(self, name)):
                raise SchemaValidationError(
                    f"Order8RawContactPatch.{name} must be finite and non-negative"
                )
        if self.normal_force_n > self.force_magnitude_n + 1.0e-12:
            raise SchemaValidationError(
                "Order8RawContactPatch.normal_force_n must not exceed force_magnitude_n"
            )


@dataclass
class Order8NaturalContactObservation(SchemaBase):
    observation_version: str
    phase: Order8NaturalContactPhase
    time_s: float
    step_dt_s: float
    object_id: str
    selected_dock_link_ids: list[str]
    raw_contact_patches: list[Order8RawContactPatch]
    selected_contact_point_object_m_by_link: dict[str, list[float]]
    raw_contact_valid: bool
    raw_contact_saturated: bool
    object_bottom_clearance_m: float
    object_floor_contact: bool
    object_linear_speed_mps: float
    object_vertical_velocity_world_mps: float
    object_angular_speed_rad_s: float
    transport_distance_m: float
    gripper_object_clearance_m: float
    controller_qp_feasible: bool
    simultaneous_qclose_acquired: bool
    grasp_confirmation_ready: bool
    missing_actuator_target_count: int = 0
    unsupported_actuator_target_count: int = 0
    clipped_actuator_target_count: int = 0
    unresolved_actuator_target_count: int = 0
    raw_contact_truth_role: Literal["privileged_diagnostic_only"] = (
        ORDER8_RAW_CONTACT_TRUTH_ROLE
    )
    raw_contact_truth_actor_input: Literal[False] = False
    raw_contact_truth_qpid_command: Literal[False] = False

    def validate(self) -> None:
        if self.observation_version != ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION:
            raise SchemaValidationError(
                "Order8NaturalContactObservation.observation_version mismatch"
            )
        if not _finite_non_negative(self.time_s):
            raise SchemaValidationError(
                "Order8NaturalContactObservation.time_s must be finite and non-negative"
            )
        if not _finite_positive(self.step_dt_s):
            raise SchemaValidationError(
                "Order8NaturalContactObservation.step_dt_s must be finite and positive"
            )
        require_non_empty(self.object_id, "Order8NaturalContactObservation.object_id")
        if len(self.selected_dock_link_ids) < 2:
            raise SchemaValidationError(
                "Order8NaturalContactObservation requires at least two selected Dock links"
            )
        if any(not value for value in self.selected_dock_link_ids):
            raise SchemaValidationError(
                "Order8NaturalContactObservation selected Dock link ids must be non-empty"
            )
        if len(set(self.selected_dock_link_ids)) != len(self.selected_dock_link_ids):
            raise SchemaValidationError(
                "Order8NaturalContactObservation selected Dock link ids must be distinct"
            )
        patch_ids = [patch.patch_id for patch in self.raw_contact_patches]
        if len(patch_ids) != len(set(patch_ids)):
            raise SchemaValidationError(
                "Order8NaturalContactObservation raw patch ids must be unique"
            )
        for name in (
            "object_bottom_clearance_m",
            "object_linear_speed_mps",
            "object_angular_speed_rad_s",
            "transport_distance_m",
            "gripper_object_clearance_m",
        ):
            if not _finite_non_negative(getattr(self, name)):
                raise SchemaValidationError(
                    f"Order8NaturalContactObservation.{name} must be finite and non-negative"
                )
        if not set(self.selected_contact_point_object_m_by_link).issubset(
            set(self.selected_dock_link_ids)
        ):
            raise SchemaValidationError(
                "Order8 contact-point map must contain only selected Dock links"
            )
        for point in self.selected_contact_point_object_m_by_link.values():
            if len(point) != 3 or not all(_finite(value) for value in point):
                raise SchemaValidationError(
                    "Order8 contact points must be finite object-frame Vector3 values"
                )
        if not _finite(self.object_vertical_velocity_world_mps):
            raise SchemaValidationError(
                "Order8NaturalContactObservation.object_vertical_velocity_world_mps must be finite"
            )
        for name in (
            "missing_actuator_target_count",
            "unsupported_actuator_target_count",
            "clipped_actuator_target_count",
            "unresolved_actuator_target_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SchemaValidationError(
                    f"Order8NaturalContactObservation.{name} must be a non-negative integer"
                )
        for name in (
            "raw_contact_valid",
            "raw_contact_saturated",
            "object_floor_contact",
            "controller_qp_feasible",
            "simultaneous_qclose_acquired",
            "grasp_confirmation_ready",
        ):
            if type(getattr(self, name)) is not bool:
                raise SchemaValidationError(
                    f"Order8NaturalContactObservation.{name} must be bool"
                )
        if self.raw_contact_truth_role != ORDER8_RAW_CONTACT_TRUTH_ROLE:
            raise SchemaValidationError(
                "Order8 observation raw contact truth must remain privileged diagnostic only"
            )
        if self.raw_contact_truth_actor_input is not False:
            raise SchemaValidationError(
                "Order8 observation raw contact truth must not be an actor input"
            )
        if self.raw_contact_truth_qpid_command is not False:
            raise SchemaValidationError(
                "Order8 observation raw contact truth must not be a QPID command"
            )


@dataclass
class Order8NaturalContactStepEvidence(SchemaBase):
    evidence_version: str
    phase: Order8NaturalContactPhase
    time_s: float
    selected_dock_link_ids: list[str]
    selected_contact_link_ids: list[str]
    selected_distinct_contact_count: int
    selected_contact_exists: bool
    simultaneous_qclose_acquired: bool
    grasp_confirmation_ready: bool
    contact_dwell_elapsed_s: float
    grasp_acquired: bool
    contact_break_elapsed_s: float
    lift_acquired: bool
    transport_acquired: bool
    release_contact_free_elapsed_s: float
    release_contact_free_acquired: bool
    retreat_clearance_acquired: bool
    settle_dwell_elapsed_s: float
    settle_acquired: bool
    object_dropped: bool
    controller_safe: bool
    unintended_contact_link_ids: list[str]
    total_selected_force_magnitude_n: float
    max_force_per_selected_contact_n: float
    max_torque_per_selected_contact_nm: float
    max_penetration_m: float
    max_tangential_slip_speed_mps: float
    provisional_acquisition_slip_speed_mps: float
    contact_point_slip_displacement_m_by_link: dict[str, float]
    max_contact_point_slip_displacement_m: float
    object_bottom_clearance_m: float
    transport_distance_m: float
    gripper_object_clearance_m: float
    hard_failure: bool
    failure_reasons: list[str]
    gate_results: dict[str, bool]
    raw_contact_truth_role: Literal["privileged_diagnostic_only"] = (
        ORDER8_RAW_CONTACT_TRUTH_ROLE
    )
    raw_contact_truth_actor_input: Literal[False] = False
    raw_contact_truth_qpid_command: Literal[False] = False

    def validate(self) -> None:
        if self.evidence_version != ORDER8_NATURAL_CONTACT_STEP_EVIDENCE_VERSION:
            raise SchemaValidationError(
                "Order8NaturalContactStepEvidence.evidence_version mismatch"
            )
        if self.selected_distinct_contact_count != len(self.selected_contact_link_ids):
            raise SchemaValidationError(
                "Order8 step selected contact count does not match distinct link ids"
            )
        if len(set(self.selected_contact_link_ids)) != len(
            self.selected_contact_link_ids
        ):
            raise SchemaValidationError(
                "Order8 step selected contact link ids must be distinct"
            )
        for name in (
            "time_s",
            "contact_dwell_elapsed_s",
            "contact_break_elapsed_s",
            "release_contact_free_elapsed_s",
            "settle_dwell_elapsed_s",
            "total_selected_force_magnitude_n",
            "max_force_per_selected_contact_n",
            "max_torque_per_selected_contact_nm",
            "max_penetration_m",
            "max_tangential_slip_speed_mps",
            "provisional_acquisition_slip_speed_mps",
            "max_contact_point_slip_displacement_m",
            "object_bottom_clearance_m",
            "transport_distance_m",
            "gripper_object_clearance_m",
        ):
            if not _finite_non_negative(getattr(self, name)):
                raise SchemaValidationError(
                    f"Order8NaturalContactStepEvidence.{name} must be finite and non-negative"
                )
        if any(
            not _finite_non_negative(value)
            for value in self.contact_point_slip_displacement_m_by_link.values()
        ):
            raise SchemaValidationError(
                "Order8 contact-point slip values must be finite and non-negative"
            )
        if self.hard_failure != bool(self.failure_reasons):
            raise SchemaValidationError(
                "Order8 step hard_failure must match failure_reasons"
            )
        if self.object_dropped and not self.hard_failure:
            raise SchemaValidationError("Order8 object drop must be a hard failure")
        if type(self.grasp_confirmation_ready) is not bool:
            raise SchemaValidationError(
                "Order8NaturalContactStepEvidence.grasp_confirmation_ready must be bool"
            )
        _validate_truth_boundary(
            self.raw_contact_truth_role,
            self.raw_contact_truth_actor_input,
            self.raw_contact_truth_qpid_command,
        )


@dataclass
class Order8NaturalContactResult(SchemaBase):
    result_version: str
    config_version: str
    config_hash: str
    contact_model: str
    final_phase: Order8NaturalContactPhase
    attempted: bool
    passed: bool
    step_count: int
    duration_s: float
    selected_dock_link_ids: list[str]
    grasp_acquired: bool
    lift_acquired: bool
    transport_acquired: bool
    release_contact_free_acquired: bool
    retreat_clearance_acquired: bool
    settle_acquired: bool
    object_dropped: bool
    unintended_contact_count: int
    max_force_per_selected_contact_n: float
    max_torque_per_selected_contact_nm: float
    max_penetration_m: float
    max_tangential_slip_speed_mps: float
    max_provisional_acquisition_slip_speed_mps: float
    max_contact_point_slip_displacement_m_by_link: dict[str, float]
    failure_reasons: list[str]
    raw_contact_truth_role: Literal["privileged_diagnostic_only"] = (
        ORDER8_RAW_CONTACT_TRUTH_ROLE
    )
    raw_contact_truth_actor_input: Literal[False] = False
    raw_contact_truth_qpid_command: Literal[False] = False

    def validate(self) -> None:
        if self.result_version != ORDER8_NATURAL_CONTACT_RESULT_VERSION:
            raise SchemaValidationError(
                "Order8NaturalContactResult.result_version mismatch"
            )
        if self.config_version != ORDER8_NATURAL_CONTACT_CONFIG_VERSION:
            raise SchemaValidationError(
                "Order8NaturalContactResult.config_version mismatch"
            )
        if self.contact_model != ORDER8_NATURAL_CONTACT_MODEL:
            raise SchemaValidationError(
                "Order8NaturalContactResult.contact_model mismatch"
            )
        if not _is_sha256(self.config_hash):
            raise SchemaValidationError(
                "Order8NaturalContactResult.config_hash must be sha256"
            )
        if (
            not isinstance(self.step_count, int)
            or isinstance(self.step_count, bool)
            or self.step_count < 0
        ):
            raise SchemaValidationError(
                "Order8NaturalContactResult.step_count must be a non-negative integer"
            )
        if not _finite_non_negative(self.duration_s):
            raise SchemaValidationError(
                "Order8NaturalContactResult.duration_s must be finite and non-negative"
            )
        if (
            not isinstance(self.unintended_contact_count, int)
            or isinstance(self.unintended_contact_count, bool)
            or self.unintended_contact_count < 0
        ):
            raise SchemaValidationError(
                "Order8NaturalContactResult.unintended_contact_count must be non-negative"
            )
        if len(set(self.selected_dock_link_ids)) != len(self.selected_dock_link_ids):
            raise SchemaValidationError(
                "Order8 result selected Dock link ids must be distinct"
            )
        if self.attempted and len(self.selected_dock_link_ids) < 2:
            raise SchemaValidationError(
                "An attempted Order8 result requires at least two selected Dock links"
            )
        for name in (
            "max_force_per_selected_contact_n",
            "max_torque_per_selected_contact_nm",
            "max_penetration_m",
            "max_tangential_slip_speed_mps",
            "max_provisional_acquisition_slip_speed_mps",
        ):
            if not _finite_non_negative(getattr(self, name)):
                raise SchemaValidationError(
                    f"Order8NaturalContactResult.{name} must be finite and non-negative"
                )
        if any(
            not _finite_non_negative(value)
            for value in self.max_contact_point_slip_displacement_m_by_link.values()
        ):
            raise SchemaValidationError(
                "Order8 result contact-point slip values must be finite and non-negative"
            )
        required_pass_gates = (
            self.attempted
            and self.grasp_acquired
            and self.lift_acquired
            and self.transport_acquired
            and self.release_contact_free_acquired
            and self.retreat_clearance_acquired
            and self.settle_acquired
            and not self.object_dropped
            and self.unintended_contact_count == 0
            and not self.failure_reasons
            and self.final_phase
            in {Order8NaturalContactPhase.SETTLE, Order8NaturalContactPhase.COMPLETE}
        )
        if self.passed != required_pass_gates:
            raise SchemaValidationError(
                "Order8NaturalContactResult.passed does not match acceptance gates"
            )
        _validate_truth_boundary(
            self.raw_contact_truth_role,
            self.raw_contact_truth_actor_input,
            self.raw_contact_truth_qpid_command,
        )


def load_order8_natural_contact_config(
    path: str | Path,
) -> Order8NaturalContactConfig:
    data = load_config(path)
    return Order8NaturalContactConfig.from_dict(data.get("order8", data))


def _validate_truth_boundary(role: str, actor_input: bool, qpid_command: bool) -> None:
    if role != ORDER8_RAW_CONTACT_TRUTH_ROLE:
        raise SchemaValidationError(
            "Order8 raw contact truth must remain privileged diagnostic only"
        )
    if actor_input is not False:
        raise SchemaValidationError(
            "Order8 raw contact truth must not be an actor input"
        )
    if qpid_command is not False:
        raise SchemaValidationError(
            "Order8 raw contact truth must not be a QPID command"
        )


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_non_negative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
