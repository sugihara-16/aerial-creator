from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from amsrr.assembly.assembly_control_bridge import AssemblyControlBridgeConfig
from amsrr.assembly.assembly_motion_planner import AssemblyMotionPlannerConfig
from amsrr.controllers.detach_wrench_estimator import DetachUnloadGateConfig
from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError, canonical_json, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.simulation.dynamic_dock_constraint import (
    DYNAMIC_DOCK_CONSTRAINT_VERSION,
    DynamicDockConstraintSpec,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend
from amsrr.simulation.isaac_usd_collision import (
    DEFAULT_DOCK_CONVEX_DECOMPOSITION_MAX_HULLS,
    DEFAULT_DOCK_CONVEX_DECOMPOSITION_SHRINK_WRAP,
)
from amsrr.utils.hashing import hash_directory_manifest, hash_file


DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION = "dynamic_assembly_roundtrip_v1"
DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE = "attach_only"
DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE = "roundtrip"
DYNAMIC_ASSEMBLY_ACCEPTANCE_GATES = {
    DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE,
    DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE,
}
DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE = "physical_funnel_contact"
DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE = (
    "selected_pair_collision_filter_fallback"
)
DYNAMIC_ASSEMBLY_MATING_MODES = {
    DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
    DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE,
}
DYNAMIC_ASSEMBLY_PHYSICAL_ACCEPTANCE_CONTRACT = "physical_funnel_contact_v1"
DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT = (
    "selected_pair_collision_filter_fallback_v1"
)
DYNAMIC_ASSEMBLY_PROGRESS_PREFIX = "[dynamic-assembly]"
DYNAMIC_ASSEMBLY_PROGRESS_INTERVAL_S = 1.0
DYNAMIC_ASSEMBLY_LIVE_PHASE_LABELS = {
    "axial_approach": "axial",
    "constraint_enabled": "fixed",
    "unload_dwell": "unload",
}


def dynamic_assembly_acceptance_contract(mating_contact_mode: str) -> str:
    if mating_contact_mode == DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE:
        return DYNAMIC_ASSEMBLY_PHYSICAL_ACCEPTANCE_CONTRACT
    if mating_contact_mode == DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE:
        return DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT
    raise SchemaValidationError(
        f"unsupported dynamic assembly mating mode: {mating_contact_mode!r}"
    )


def format_dynamic_assembly_progress(phase: str, simulation_time_s: float) -> str:
    require_non_empty(phase, "dynamic assembly progress phase")
    if not math.isfinite(float(simulation_time_s)) or float(simulation_time_s) < 0.0:
        raise SchemaValidationError(
            "dynamic assembly progress simulation_time_s must be finite and non-negative"
        )
    live_phase = DYNAMIC_ASSEMBLY_LIVE_PHASE_LABELS.get(phase, phase)
    return (
        f"{DYNAMIC_ASSEMBLY_PROGRESS_PREFIX} "
        f"simulation_time={float(simulation_time_s):.3f}s "
        f"phase={live_phase} event={phase}"
    )


def dynamic_assembly_progress_due(
    last_emit_time_s: float | None,
    simulation_time_s: float,
    *,
    interval_s: float = DYNAMIC_ASSEMBLY_PROGRESS_INTERVAL_S,
) -> bool:
    """Return whether a same-phase live-progress heartbeat is due."""

    current = float(simulation_time_s)
    interval = float(interval_s)
    if not math.isfinite(current) or current < 0.0:
        raise SchemaValidationError(
            "dynamic assembly progress simulation_time_s must be finite and non-negative"
        )
    if not math.isfinite(interval) or interval <= 0.0:
        raise SchemaValidationError(
            "dynamic assembly progress interval_s must be finite and positive"
        )
    if last_emit_time_s is None:
        return True
    previous = float(last_emit_time_s)
    if not math.isfinite(previous) or previous < 0.0:
        raise SchemaValidationError(
            "dynamic assembly progress last_emit_time_s must be finite and non-negative"
        )
    if current + 1.0e-12 < previous:
        raise SchemaValidationError(
            "dynamic assembly progress simulation time cannot move backwards"
        )
    return current - previous + 1.0e-12 >= interval


@dataclass
class DynamicSeparationLifecycle:
    """Bounded separation/filter-removal/post-release lifecycle.

    The caller owns the physics operations and supplies one observation after each
    completed simulation step.  This helper only owns lifecycle state and gate
    ordering.  In particular, a gate satisfied on the final budget step succeeds
    instead of being misclassified as a timeout.
    """

    nominal_separation_steps: int
    max_separation_steps: int
    minimum_gap_m: float
    minimum_clearance_m: float
    required_post_release_stable_steps: int
    max_post_release_steps: int
    phase: str = field(default="separation", init=False)
    separation_steps: int = field(default=0, init=False)
    post_release_steps: int = field(default=0, init=False)
    post_release_stable_steps: int = field(default=0, init=False)
    failure_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        for name in (
            "nominal_separation_steps",
            "max_separation_steps",
            "required_post_release_stable_steps",
            "max_post_release_steps",
        ):
            if not _is_positive_int(getattr(self, name)):
                raise SchemaValidationError(
                    f"DynamicSeparationLifecycle.{name} must be a positive integer"
                )
        for name in ("minimum_gap_m", "minimum_clearance_m"):
            if not _is_finite_positive_number(getattr(self, name)):
                raise SchemaValidationError(
                    f"DynamicSeparationLifecycle.{name} must be finite and positive"
                )
        if self.max_separation_steps < self.nominal_separation_steps:
            raise SchemaValidationError(
                "max_separation_steps cannot be smaller than nominal_separation_steps"
            )
        if self.max_separation_steps > 2 * self.nominal_separation_steps:
            raise SchemaValidationError(
                "max_separation_steps cannot exceed twice nominal_separation_steps"
            )
        if self.max_post_release_steps < self.required_post_release_stable_steps:
            raise SchemaValidationError(
                "max_post_release_steps cannot be smaller than "
                "required_post_release_stable_steps"
            )
        if (
            self.max_post_release_steps
            > 2 * self.required_post_release_stable_steps
        ):
            raise SchemaValidationError(
                "max_post_release_steps cannot exceed twice "
                "required_post_release_stable_steps"
            )

    def observe_separation(self, *, gap_m: float, clearance_m: float) -> str:
        """Observe one completed separation step and return the next action."""

        self._require_phase("separation")
        for name, value in (("gap_m", gap_m), ("clearance_m", clearance_m)):
            if not _is_finite_number(value):
                raise SchemaValidationError(f"{name} must be finite")

        self.separation_steps += 1
        gate_satisfied = (
            self.separation_steps >= self.nominal_separation_steps
            and float(gap_m) >= self.minimum_gap_m
            and float(clearance_m) >= self.minimum_clearance_m
        )
        if gate_satisfied:
            self.phase = "awaiting_filter_removal"
            return "request_filter_removal"
        if self.separation_steps >= self.max_separation_steps:
            self.phase = "timed_out"
            self.failure_reason = "separation_timeout"
            return "timeout"
        return "continue"

    def confirm_filter_removal(self, *, verified: bool) -> str:
        """Enter post-release only after selected-pair filter removal is verified."""

        self._require_phase("awaiting_filter_removal")
        if type(verified) is not bool:
            raise SchemaValidationError("verified must be bool")
        if not verified:
            self.phase = "failed"
            self.failure_reason = "filter_removal_verification_failed"
            return "verification_failed"
        self.phase = "post_release"
        return "post_release"

    def observe_post_release(self, *, stable: bool) -> str:
        """Observe one post-release step with a continuous, resettable dwell."""

        self._require_phase("post_release")
        if type(stable) is not bool:
            raise SchemaValidationError("stable must be bool")

        self.post_release_steps += 1
        if stable:
            self.post_release_stable_steps += 1
        else:
            self.post_release_stable_steps = 0

        if (
            self.post_release_stable_steps
            >= self.required_post_release_stable_steps
        ):
            self.phase = "complete"
            return "complete"
        if self.post_release_steps >= self.max_post_release_steps:
            self.phase = "timed_out"
            self.failure_reason = "post_release_timeout"
            return "timeout"
        return "continue"

    def _require_phase(self, expected: str) -> None:
        if self.phase != expected:
            raise SchemaValidationError(
                f"DynamicSeparationLifecycle phase must be {expected!r}, "
                f"got {self.phase!r}"
            )


@dataclass
class DynamicAssemblyIsaacConfig(SchemaBase):
    version: str = DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION
    acceptance_gate: str = DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE
    collision_type: str = "Convex Decomposition"
    dock_convex_decomposition_max_hulls: int = (
        DEFAULT_DOCK_CONVEX_DECOMPOSITION_MAX_HULLS
    )
    dock_convex_decomposition_shrink_wrap: bool = (
        DEFAULT_DOCK_CONVEX_DECOMPOSITION_SHRINK_WRAP
    )
    generated_usd_dir: str = "artifacts/isaac/robots/holon_dynamic_assembly"
    simulation_dt_s: float = 0.005
    solver_position_iteration_count: int = 8
    solver_velocity_iteration_count: int = 8
    command_timeout_s: float = 600.0
    floor_settle_duration_s: float = 1.0
    floor_settle_required_dwell_s: float = 0.25
    floor_contact_force_threshold_n: float = 0.10
    floor_settle_linear_speed_tolerance_mps: float = 0.05
    floor_settle_angular_speed_tolerance_radps: float = 0.10
    floor_settle_joint_position_tolerance_rad: float = 0.05
    floor_settle_joint_speed_tolerance_radps: float = 0.10
    preflight_vectoring_timeout_s: float = 1.0
    preflight_feasible_dwell_s: float = 0.10
    takeoff_duration_s: float = 2.0
    assembly_height_m: float = 1.0
    takeoff_hold_s: float = 1.0
    hover_acquisition_timeout_s: float = 8.0
    hover_acquisition_dwell_s: float = 0.50
    hover_position_tolerance_m: float = 0.10
    hover_attitude_tolerance_rad: float = math.radians(6.0)
    hover_linear_speed_tolerance_mps: float = 0.15
    hover_angular_speed_tolerance_radps: float = 0.20
    controller_handover_blend_s: float = 0.25
    assembly_translation_speed_limit_mps: float = 0.10
    assembly_angular_speed_limit_radps: float = math.radians(20.0)
    assembly_command_lookahead_s: float = 0.30
    constraint_verify_dwell_s: float = 0.25
    attached_hold_s: float = 1.0
    attached_position_tolerance_m: float = 0.05
    attached_attitude_tolerance_rad: float = math.radians(3.0)
    attached_linear_speed_tolerance_mps: float = 0.10
    attached_angular_speed_tolerance_radps: float = 0.10
    attached_joint_position_tolerance_rad: float = 0.05
    attached_joint_speed_tolerance_radps: float = 0.10
    post_release_hold_s: float = 1.0
    post_release_position_tolerance_m: float = 0.05
    post_release_attitude_tolerance_rad: float = math.radians(8.0)
    post_release_linear_speed_tolerance_mps: float = 0.20
    post_release_angular_speed_tolerance_radps: float = 0.25
    post_release_joint_position_tolerance_rad: float = 0.05
    post_release_joint_speed_tolerance_radps: float = 0.10
    separation_distance_m: float = 0.20
    separation_speed_mps: float = 0.05
    release_filter_clearance_m: float = 0.03
    mating_contact_mode: str = DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE
    guidance_contact_max_axial_gap_m: float = 0.060
    guidance_contact_max_transverse_error_m: float = 0.010
    guidance_contact_max_attitude_error_rad: float = math.radians(3.0)
    # Legacy contact-surface diagnostics.  These are not physical-attach gates:
    # funnel-wall guidance contact normally occurs before both connect frames seat.
    selected_surface_axial_tolerance_m: float = 0.015
    selected_surface_radius_m: float = 0.12
    selected_surface_normal_tolerance_rad: float = math.radians(30.0)
    detach_external_contact_force_threshold_n: float = 0.05
    allocation_mode: str = "rigid_body_qp"
    control_bridge: AssemblyControlBridgeConfig = field(
        default_factory=lambda: AssemblyControlBridgeConfig(
            require_selected_pair_contact=False,
        )
    )
    motion_planner: AssemblyMotionPlannerConfig = field(default_factory=AssemblyMotionPlannerConfig)
    detach_unload: DetachUnloadGateConfig = field(default_factory=DetachUnloadGateConfig)

    def validate(self) -> None:
        if self.version != DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION:
            raise SchemaValidationError("DynamicAssemblyIsaacConfig.version mismatch")
        if self.acceptance_gate not in DYNAMIC_ASSEMBLY_ACCEPTANCE_GATES:
            raise SchemaValidationError(
                "DynamicAssemblyIsaacConfig.acceptance_gate must be attach_only or roundtrip"
            )
        if self.collision_type != "Convex Decomposition":
            raise SchemaValidationError(
                "dynamic assembly requires Convex Decomposition for mating Dock meshes"
            )
        if not 1 <= int(self.dock_convex_decomposition_max_hulls) <= 2048:
            raise SchemaValidationError(
                "dock_convex_decomposition_max_hulls must be in [1, 2048]"
            )
        if type(self.dock_convex_decomposition_shrink_wrap) is not bool:
            raise SchemaValidationError(
                "dock_convex_decomposition_shrink_wrap must be bool"
            )
        for name in (
            "solver_position_iteration_count",
            "solver_velocity_iteration_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SchemaValidationError(
                    f"DynamicAssemblyIsaacConfig.{name} must be a positive integer"
                )
        require_non_empty(
            self.generated_usd_dir,
            "DynamicAssemblyIsaacConfig.generated_usd_dir",
        )
        require_non_empty(self.allocation_mode, "DynamicAssemblyIsaacConfig.allocation_mode")
        if self.allocation_mode != "rigid_body_qp":
            raise SchemaValidationError("dynamic assembly requires rigid_body_qp")
        if self.mating_contact_mode not in DYNAMIC_ASSEMBLY_MATING_MODES:
            raise SchemaValidationError(
                "DynamicAssemblyIsaacConfig.mating_contact_mode must be "
                "physical_funnel_contact or selected_pair_collision_filter_fallback"
            )
        for name in (
            "simulation_dt_s",
            "command_timeout_s",
            "floor_settle_duration_s",
            "floor_settle_required_dwell_s",
            "floor_contact_force_threshold_n",
            "floor_settle_linear_speed_tolerance_mps",
            "floor_settle_angular_speed_tolerance_radps",
            "floor_settle_joint_position_tolerance_rad",
            "floor_settle_joint_speed_tolerance_radps",
            "preflight_vectoring_timeout_s",
            "preflight_feasible_dwell_s",
            "takeoff_duration_s",
            "assembly_height_m",
            "takeoff_hold_s",
            "hover_acquisition_timeout_s",
            "hover_acquisition_dwell_s",
            "hover_position_tolerance_m",
            "hover_attitude_tolerance_rad",
            "hover_linear_speed_tolerance_mps",
            "hover_angular_speed_tolerance_radps",
            "controller_handover_blend_s",
            "assembly_translation_speed_limit_mps",
            "assembly_angular_speed_limit_radps",
            "assembly_command_lookahead_s",
            "constraint_verify_dwell_s",
            "attached_hold_s",
            "attached_position_tolerance_m",
            "attached_attitude_tolerance_rad",
            "attached_linear_speed_tolerance_mps",
            "attached_angular_speed_tolerance_radps",
            "attached_joint_position_tolerance_rad",
            "attached_joint_speed_tolerance_radps",
            "post_release_hold_s",
            "post_release_position_tolerance_m",
            "post_release_attitude_tolerance_rad",
            "post_release_linear_speed_tolerance_mps",
            "post_release_angular_speed_tolerance_radps",
            "post_release_joint_position_tolerance_rad",
            "post_release_joint_speed_tolerance_radps",
            "separation_distance_m",
            "separation_speed_mps",
            "release_filter_clearance_m",
            "guidance_contact_max_axial_gap_m",
            "guidance_contact_max_transverse_error_m",
            "guidance_contact_max_attitude_error_rad",
            "selected_surface_axial_tolerance_m",
            "selected_surface_radius_m",
            "selected_surface_normal_tolerance_rad",
            "detach_external_contact_force_threshold_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"DynamicAssemblyIsaacConfig.{name} must be finite and positive"
                )
        if (
            self.mating_contact_mode == DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE
            and not self.control_bridge.require_selected_pair_contact
        ):
            raise SchemaValidationError(
                "dynamic assembly physical attach requires selected-pair contact"
            )
        if (
            self.mating_contact_mode == DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE
            and self.control_bridge.require_selected_pair_contact
        ):
            raise SchemaValidationError(
                "dynamic assembly selected-pair filter fallback requires "
                "require_selected_pair_contact=false"
            )
        if not math.isclose(
            self.guidance_contact_max_transverse_error_m,
            self.control_bridge.transverse_tolerance_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise SchemaValidationError(
                "guidance contact transverse envelope must match the coarse pre-align envelope"
            )
        if not math.isclose(
            self.guidance_contact_max_attitude_error_rad,
            self.control_bridge.attitude_tolerance_rad,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise SchemaValidationError(
                "guidance contact attitude envelope must match the coarse pre-align envelope"
            )
        if self.floor_settle_required_dwell_s > self.floor_settle_duration_s:
            raise SchemaValidationError(
                "floor settle dwell cannot exceed floor settle duration"
            )
        if self.release_filter_clearance_m >= self.separation_distance_m:
            raise SchemaValidationError(
                "release filter clearance must be smaller than the commanded separation"
            )
        if self.post_release_position_tolerance_m >= (
            self.separation_distance_m - self.release_filter_clearance_m
        ):
            raise SchemaValidationError(
                "post-release position tolerance must preserve the release clearance margin"
            )


@dataclass
class DynamicAssemblyIsaacResult(SchemaBase):
    version: str
    acceptance_gate: str
    graph_id: str
    graph_hash: str
    dry_run: bool
    attempted: bool
    isaac_backed: bool
    attach_passed: bool
    detach_passed: bool
    passed: bool
    report_validation_failures: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def validate(self) -> None:
        if self.version != DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION:
            raise SchemaValidationError("DynamicAssemblyIsaacResult.version mismatch")
        if self.acceptance_gate not in DYNAMIC_ASSEMBLY_ACCEPTANCE_GATES:
            raise SchemaValidationError(
                "DynamicAssemblyIsaacResult.acceptance_gate must be attach_only or roundtrip"
            )
        require_non_empty(self.graph_id, "DynamicAssemblyIsaacResult.graph_id")
        if len(self.graph_hash) != 64:
            raise SchemaValidationError("DynamicAssemblyIsaacResult.graph_hash must be sha256")
        if self.passed and (
            self.dry_run
            or not self.isaac_backed
            or not self.attach_passed
            or (
                self.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE
                and not self.detach_passed
            )
            or self.report_validation_failures
        ):
            raise SchemaValidationError(
                "DynamicAssemblyIsaacResult pass requires the selected real acceptance gate "
                "and no report failures"
            )


class DynamicAssemblyIsaacEnv:
    def __init__(
        self,
        *,
        config: DynamicAssemblyIsaacConfig,
        backend: IsaacLabBackend,
        backend_config_path: str = "configs/env/isaac_lab.yaml",
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
        command_executor: Callable[[list[str], float], dict[str, Any]] | None = None,
        verify_local_artifacts: bool = True,
    ) -> None:
        config.validate()
        if viewer not in {None, "kit"}:
            raise SchemaValidationError("dynamic assembly viewer must be None or 'kit'")
        if viewer is None and (realtime_playback or keep_open_after_rollout_s > 0.0):
            raise SchemaValidationError(
                "dynamic assembly real-time/post-rollout viewing requires viewer='kit'"
            )
        if keep_open_after_rollout_s < 0.0:
            raise SchemaValidationError("keep_open_after_rollout_s must be non-negative")
        self.config = config
        self.backend = backend
        self.backend_config_path = backend_config_path
        self.viewer = viewer
        self.realtime_playback = bool(realtime_playback)
        self.keep_open_after_rollout_s = float(keep_open_after_rollout_s)
        self.command_executor = command_executor or _run_json_command
        self.verify_local_artifacts = bool(verify_local_artifacts)

    def build_probe_command(self, morphology_graph: MorphologyGraph) -> list[str]:
        _validate_roundtrip_graph(morphology_graph)
        assembly_executor_timeout_s = (
            self.config.control_bridge.step_timeout_s
            + max(5.0, self.config.control_bridge.prealign_dwell_s + 2.0)
            + self.config.control_bridge.axial_approach_timeout_s
            + max(5.0, self.config.constraint_verify_dwell_s + 2.0)
        )
        requested_steps = max(
            1,
            int(
                math.ceil(
                    (
                        self.config.floor_settle_duration_s
                        + self.config.preflight_vectoring_timeout_s
                        + self.config.takeoff_duration_s
                        + self.config.takeoff_hold_s
                        + self.config.hover_acquisition_timeout_s
                        + assembly_executor_timeout_s
                        + 2.0 * self.config.controller_handover_blend_s
                        + self.config.constraint_verify_dwell_s
                        + self.config.attached_hold_s
                        # The nominal durations remain the acceptance minima.  A
                        # second bounded window lets closed-loop tracking acquire
                        # the measured release clearance and continuous stable
                        # dwell without weakening either gate.
                        + 2.0 * self.config.post_release_hold_s
                        + 2.0
                        * self.config.separation_distance_m
                        / self.config.separation_speed_mps
                        + 5.0
                    )
                    / self.config.simulation_dt_s
                )
            ),
        )
        command = self.backend.holon_spawn_probe_command(
            config_path=self.backend_config_path,
            convert_if_missing=False,
            force_convert=True,
            generated_usd_dir=self.config.generated_usd_dir,
            steps=requested_steps,
            viewer=self.viewer,
            realtime_playback=self.realtime_playback,
            keep_open_after_smoke_s=self.keep_open_after_rollout_s,
        )
        command.extend(
            [
                "--dynamic-assembly-roundtrip",
                "--dynamic-assembly-graph-json",
                canonical_json(morphology_graph),
                "--dynamic-assembly-config-json",
                self.config.to_json(),
                "--dt",
                str(self.config.simulation_dt_s),
                "--allocation-mode",
                self.config.allocation_mode,
            ]
        )
        return command

    def run(
        self,
        morphology_graph: MorphologyGraph,
        *,
        dry_run: bool = True,
        check_availability: bool = True,
    ) -> DynamicAssemblyIsaacResult:
        _validate_roundtrip_graph(morphology_graph)
        if dry_run:
            return DynamicAssemblyIsaacResult(
                version=self.config.version,
                acceptance_gate=self.config.acceptance_gate,
                graph_id=morphology_graph.graph_id,
                graph_hash=morphology_graph.stable_hash(),
                dry_run=True,
                attempted=False,
                isaac_backed=False,
                attach_passed=False,
                detach_passed=False,
                passed=False,
                report={"probe_command": self.build_probe_command(morphology_graph)},
            )
        if check_availability:
            availability = self.backend.availability()
            if not availability.available:
                return DynamicAssemblyIsaacResult(
                    version=self.config.version,
                    acceptance_gate=self.config.acceptance_gate,
                    graph_id=morphology_graph.graph_id,
                    graph_hash=morphology_graph.stable_hash(),
                    dry_run=False,
                    attempted=False,
                    isaac_backed=False,
                    attach_passed=False,
                    detach_passed=False,
                    passed=False,
                    report_validation_failures=list(availability.missing_reasons),
                    failure_reason=",".join(availability.missing_reasons),
                )
        try:
            report = self.command_executor(
                self.build_probe_command(morphology_graph),
                self.config.command_timeout_s,
            )
        except Exception as exc:  # pragma: no cover - subprocess-specific
            return DynamicAssemblyIsaacResult(
                version=self.config.version,
                acceptance_gate=self.config.acceptance_gate,
                graph_id=morphology_graph.graph_id,
                graph_hash=morphology_graph.stable_hash(),
                dry_run=False,
                attempted=True,
                isaac_backed=True,
                attach_passed=False,
                detach_passed=False,
                passed=False,
                report_validation_failures=["probe_execution_failed"],
                failure_reason=str(exc),
            )
        expected_physical_model = build_physical_model_from_config(
            self.backend.config.robot_model_config_path
        )
        failures = dynamic_assembly_report_failures(
            report,
            morphology_graph=morphology_graph,
            config=self.config,
            backend_config_hash=self.backend.config.stable_hash(),
            physical_model_hash=expected_physical_model.stable_hash(),
            collision_geometry_content_hash_expected=collision_geometry_content_hash(
                expected_physical_model,
                mesh_search_dirs=("module_urdf", "module_urdf/mesh"),
            ),
            verify_local_artifacts=self.verify_local_artifacts,
        )
        attach_passed = report.get("dynamic_assembly_attach_passed") is True
        detach_passed = report.get("dynamic_assembly_detach_passed") is True
        return DynamicAssemblyIsaacResult(
            version=self.config.version,
            acceptance_gate=self.config.acceptance_gate,
            graph_id=morphology_graph.graph_id,
            graph_hash=morphology_graph.stable_hash(),
            dry_run=False,
            attempted=True,
            isaac_backed=report.get("isaac_backed") is True,
            attach_passed=attach_passed,
            detach_passed=detach_passed,
            passed=not failures,
            report_validation_failures=failures,
            report=report,
            failure_reason=None if not failures else "dynamic_assembly_report_failed:" + ",".join(failures),
        )


def dynamic_assembly_report_failures(
    report: dict[str, Any],
    *,
    morphology_graph: MorphologyGraph,
    config: DynamicAssemblyIsaacConfig,
    backend_config_hash: str | None = None,
    physical_model_hash: str | None = None,
    collision_geometry_content_hash_expected: str | None = None,
    verify_local_artifacts: bool = False,
) -> list[str]:
    failures: list[str] = []

    def exact(key: str, expected: Any) -> None:
        if key not in report:
            failures.append(f"missing:{key}")
        elif type(report[key]) is not type(expected) or report[key] != expected:
            failures.append(f"mismatch:{key}")

    for key in (
        "spawn_passed",
        "isaac_backed",
        "command_probe_passed",
        "dynamic_assembly_attach_passed",
        "dynamic_assembly_passed",
        "dynamic_assembly_constraint_identity_verified",
        "dynamic_assembly_external_fixed_joint",
        "dynamic_assembly_constraint_excluded_from_articulation",
        "dynamic_assembly_dock_collision_approximation_verified",
        "dynamic_assembly_selected_pair_filter_applied",
        "dynamic_assembly_selected_pair_filter_apply_verified",
        "dynamic_assembly_floor_initialization_verified",
        "dynamic_assembly_attach_handover_completed",
        "dynamic_assembly_attached_stability_verified",
        "dynamic_assembly_attached_selected_pair_contact_free",
        "dynamic_assembly_preflight_vectoring_ready",
        "dynamic_assembly_hover_acquired",
        "dynamic_assembly_finite_state",
    ):
        exact(key, True)
    physical_mode = (
        config.mating_contact_mode == DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE
    )
    exact("dynamic_assembly_selected_pair_contact_observed", physical_mode)
    exact("dynamic_assembly_guidance_contact_observed", physical_mode)
    exact("dynamic_assembly_physical_mating_contact_claimed", physical_mode)
    exact(
        "dynamic_assembly_physical_attach_passed",
        physical_mode,
    )
    exact(
        "dynamic_assembly_filter_fallback_attach_passed",
        not physical_mode,
    )
    exact(
        "dynamic_assembly_roundtrip",
        config.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE,
    )
    if config.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE:
        for key in (
            "dynamic_assembly_detach_passed",
            "dynamic_assembly_constraint_removed",
            "dynamic_assembly_constraint_disabled_verified",
            "dynamic_assembly_selected_pair_filter_removed",
            "dynamic_assembly_selected_pair_filter_remove_verified",
            "dynamic_assembly_unload_ready",
            "dynamic_assembly_control_graph_split_before_release",
            "dynamic_assembly_post_release_stable",
            "dynamic_assembly_split_handover_completed",
            "dynamic_assembly_follower_external_contact_free_during_unload",
        ):
            exact(key, True)
    exact("command_returncode", 0)
    exact("dynamic_assembly_version", config.version)
    exact("dynamic_assembly_acceptance_gate", config.acceptance_gate)
    exact(
        "dynamic_assembly_solver_position_iteration_count",
        config.solver_position_iteration_count,
    )
    exact(
        "dynamic_assembly_solver_velocity_iteration_count",
        config.solver_velocity_iteration_count,
    )
    for key in (
        "dynamic_assembly_dock_drive_stiffness_nm_per_rad",
        "dynamic_assembly_dock_drive_damping_nms_per_rad",
        "dynamic_assembly_dock_effort_limit_sim_nm",
        "dynamic_assembly_dock_velocity_limit_sim_radps",
    ):
        if not _is_finite_positive_number(report.get(key)):
            failures.append(f"invalid:{key}")
    exact("dynamic_assembly_collision_type", config.collision_type)
    exact("dynamic_assembly_mating_contact_mode", config.mating_contact_mode)
    exact(
        "dynamic_assembly_acceptance_contract",
        dynamic_assembly_acceptance_contract(config.mating_contact_mode),
    )
    exact(
        "dynamic_assembly_dock_collision_approximation_token",
        "convexDecomposition",
    )
    dock_collision_prim_count = report.get(
        "dynamic_assembly_dock_collision_composed_prim_count"
    )
    if not _is_positive_int(dock_collision_prim_count):
        failures.append(
            "invalid:dynamic_assembly_dock_collision_composed_prim_count"
        )
    exact("dynamic_assembly_force_usd_conversion", True)
    exact("dynamic_assembly_graph_id", morphology_graph.graph_id)
    exact("dynamic_assembly_graph_hash", morphology_graph.stable_hash())
    exact("dynamic_assembly_config_hash", config.stable_hash())
    config_backend_hash = report.get("dynamic_assembly_backend_config_hash")
    if not _is_sha256(config_backend_hash):
        failures.append("invalid:dynamic_assembly_backend_config_hash")
    elif backend_config_hash is not None and config_backend_hash != backend_config_hash:
        failures.append("mismatch:dynamic_assembly_backend_config_hash")
    reported_physical_model_hash = report.get("dynamic_assembly_physical_model_hash")
    if not _is_sha256(reported_physical_model_hash):
        failures.append("invalid:dynamic_assembly_physical_model_hash")
    elif physical_model_hash is not None and reported_physical_model_hash != physical_model_hash:
        failures.append("mismatch:dynamic_assembly_physical_model_hash")
    collision_hash = report.get("dynamic_assembly_collision_geometry_content_hash")
    if not _is_sha256(collision_hash):
        failures.append("invalid:dynamic_assembly_collision_geometry_content_hash")
    elif (
        collision_geometry_content_hash_expected is not None
        and collision_hash != collision_geometry_content_hash_expected
    ):
        failures.append("mismatch:dynamic_assembly_collision_geometry_content_hash")
    urdf_hash = report.get("generated_urdf_sha256")
    if not _is_sha256(urdf_hash):
        failures.append("invalid:generated_urdf_sha256")
    usd_hash = report.get("generated_usd_sha256")
    if not _is_sha256(usd_hash):
        failures.append("invalid:generated_usd_sha256")
    usd_bundle_hash = report.get("generated_usd_bundle_hash")
    if not _is_sha256(usd_bundle_hash):
        failures.append("invalid:generated_usd_bundle_hash")
    if verify_local_artifacts:
        configured_root = Path(config.generated_usd_dir).resolve()
        urdf_path_raw = report.get("generated_urdf_path")
        if not isinstance(urdf_path_raw, str) or not urdf_path_raw:
            failures.append("invalid:generated_urdf_path")
        else:
            generated_urdf_path = Path(urdf_path_raw).resolve()
            if (
                not generated_urdf_path.is_file()
                or not generated_urdf_path.is_relative_to(configured_root)
            ):
                failures.append("invalid:generated_urdf_path")
            elif hash_file(generated_urdf_path) != urdf_hash:
                failures.append("mismatch:generated_urdf_sha256")
        usd_path_raw = report.get("usd_path")
        if not isinstance(usd_path_raw, str) or not usd_path_raw:
            failures.append("invalid:usd_path")
        else:
            usd_path = Path(usd_path_raw).resolve()
            if not usd_path.is_file() or not usd_path.is_relative_to(configured_root):
                failures.append("invalid:usd_path")
            else:
                if hash_file(usd_path) != usd_hash:
                    failures.append("mismatch:generated_usd_sha256")
                if hash_directory_manifest(usd_path.parent) != usd_bundle_hash:
                    failures.append("mismatch:generated_usd_bundle_hash")
    exact("dynamic_assembly_module_count", 2)
    exact("dynamic_assembly_constraint_version", DYNAMIC_DOCK_CONSTRAINT_VERSION)
    exact("dynamic_assembly_qpid_joint_dynamics_unaware", True)
    exact("dynamic_assembly_dock_joint_latch_semantics", False)
    for key in (
        "dynamic_assembly_qp_infeasible_count",
        "dynamic_assembly_unintended_contact_count",
        "dynamic_assembly_missing_actuator_count",
        "dynamic_assembly_unsupported_actuator_count",
        "dynamic_assembly_application_unresolved_target_count",
        "dynamic_assembly_clipped_target_count",
        "dynamic_assembly_constraint_identity_failure_count",
        "dynamic_assembly_filter_fallback_selected_contact_violation_count",
    ):
        exact(key, 0)
    events = report.get("dynamic_assembly_events")
    event_times: dict[str, float] = {}
    if not isinstance(events, list) or not events:
        failures.append("invalid:dynamic_assembly_events")
    else:
        event_records_valid = True
        previous_time_s = -math.inf
        phases: list[str] = []
        phase_counts: dict[str, int] = {}
        for event in events:
            if not isinstance(event, dict):
                event_records_valid = False
                continue
            phase = event.get("phase")
            time_s = event.get("time_s")
            metrics = event.get("metrics")
            if (
                not isinstance(phase, str)
                or not phase
                or not _is_finite_nonnegative_number(time_s)
                or not isinstance(metrics, dict)
            ):
                event_records_valid = False
                continue
            numeric_time_s = float(time_s)
            if numeric_time_s < previous_time_s:
                failures.append("invalid:dynamic_assembly_event_timestamps")
            previous_time_s = numeric_time_s
            phases.append(phase)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            event_times.setdefault(phase, numeric_time_s)
        if not event_records_valid:
            failures.append("invalid:dynamic_assembly_events")
        required = [
            "floor_settle",
            "preflight_vectoring",
            "preflight_vectoring_ready",
            "takeoff",
            "hover_acquisition",
            "hover_acquired",
            "staging",
            "prealign_dwell",
            "axial_approach",
            "fix_ready",
            "constraint_enabled",
            "verify",
            "constraint_verified",
            "attach_handover",
            "attached_hold",
        ]
        if config.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE:
            required.extend(
                [
                    "control_graph_split",
                    "split_handover",
                    "unload_dwell",
                    "constraint_removed",
                    "separation",
                    "collision_filter_removed",
                    "post_release_hold",
                    "complete",
                ]
            )
        else:
            required.append("complete")
        for phase in required:
            if phase_counts.get(phase, 0) != 1:
                failures.append(f"invalid:dynamic_assembly_phase_cardinality:{phase}")
        required_positions = [
            phases.index(phase) for phase in required if phase in phases
        ]
        if (
            len(required_positions) != len(required)
            or required_positions != sorted(required_positions)
        ):
            cursor = 0
            for phase in phases:
                if cursor < len(required) and phase == required[cursor]:
                    cursor += 1
            failures.append(
                "invalid:dynamic_assembly_phase_order:"
                + ",".join(required[cursor:])
            )
        if phases != required:
            failures.append("invalid:dynamic_assembly_lifecycle_phase_sequence")
        if "aborted" in phases or not phases or phases[-1] != "complete":
            failures.append("invalid:dynamic_assembly_lifecycle_terminal_event")
        if phase_counts.get("complete") == 1:
            complete_event = next(
                event
                for event in events
                if isinstance(event, dict) and event.get("phase") == "complete"
            )
            complete_metrics = complete_event.get("metrics")
            if not isinstance(complete_metrics, dict) or complete_metrics.get("passed") is not True:
                failures.append("invalid:dynamic_assembly_complete_event")

    failures.extend(
        _dynamic_assembly_nested_evidence_failures(
            report,
            morphology_graph=morphology_graph,
            config=config,
            event_times=event_times,
        )
    )
    return sorted(set(failures))


def _dynamic_assembly_nested_evidence_failures(
    report: dict[str, Any],
    *,
    morphology_graph: MorphologyGraph,
    config: DynamicAssemblyIsaacConfig,
    event_times: dict[str, float],
) -> list[str]:
    failures: list[str] = []

    floor_evidence = report.get("dynamic_assembly_floor_initialization_evidence")
    floor_valid = isinstance(floor_evidence, dict)
    if floor_valid:
        dwell_steps = floor_evidence.get("continuous_dwell_steps")
        required_steps = floor_evidence.get("required_dwell_steps")
        forces = floor_evidence.get("max_contact_force_n_by_module")
        expected_floor_steps = max(
            1,
            int(math.ceil(config.floor_settle_required_dwell_s / config.simulation_dt_s)),
        )
        floor_valid = bool(
            floor_evidence.get("explicit_zero_joint_position_target") is True
            and floor_evidence.get("explicit_zero_joint_velocity_target") is True
            and floor_evidence.get("explicit_zero_joint_effort_bias") is True
            and floor_evidence.get("verified") is True
            and _is_positive_int(required_steps)
            and _is_nonnegative_int(dwell_steps)
            and int(required_steps) == expected_floor_steps
            and int(dwell_steps) >= int(required_steps)
            and isinstance(forces, dict)
            and set(forces)
            == {
                str(report.get("dynamic_assembly_leader_module_id")),
                str(report.get("dynamic_assembly_follower_module_id")),
            }
            and all(
                _is_finite_nonnegative_number(value)
                and float(value) >= config.floor_contact_force_threshold_n
                for value in forces.values()
            )
        )
    if not floor_valid:
        failures.append("invalid:dynamic_assembly_floor_initialization_evidence")

    events = report.get("dynamic_assembly_events", [])
    preflight_event = next(
        (
            event
            for event in events
            if isinstance(event, dict)
            and event.get("phase") == "preflight_vectoring_ready"
        ),
        None,
    )
    preflight_metrics = (
        preflight_event.get("metrics")
        if isinstance(preflight_event, dict)
        else None
    )
    if not (
        isinstance(preflight_metrics, dict)
        and _is_finite_nonnegative_number(preflight_metrics.get("dwell_s"))
        and float(preflight_metrics["dwell_s"])
        >= config.preflight_feasible_dwell_s - config.simulation_dt_s - 1.0e-9
    ):
        failures.append("invalid:dynamic_assembly_preflight_evidence")

    hover_event = next(
        (
            event
            for event in events
            if isinstance(event, dict) and event.get("phase") == "hover_acquired"
        ),
        None,
    )
    hover_metrics = (
        hover_event.get("metrics") if isinstance(hover_event, dict) else None
    )
    hover_limits = {
        "max_position_error_m": config.hover_position_tolerance_m,
        "max_attitude_error_rad": config.hover_attitude_tolerance_rad,
        "max_linear_speed_mps": config.hover_linear_speed_tolerance_mps,
        "max_angular_speed_radps": config.hover_angular_speed_tolerance_radps,
        "max_dock_joint_position_rad": config.attached_joint_position_tolerance_rad,
        "max_dock_joint_speed_radps": config.attached_joint_speed_tolerance_radps,
    }
    if not (
        isinstance(hover_metrics, dict)
        and _is_finite_nonnegative_number(hover_metrics.get("dwell_s"))
        and float(hover_metrics["dwell_s"])
        >= config.hover_acquisition_dwell_s - config.simulation_dt_s - 1.0e-9
        and _bounded_metric_dict(hover_metrics, hover_limits)
    ):
        failures.append("invalid:dynamic_assembly_hover_evidence")

    if not _axial_selected_joint_evidence_valid(report):
        failures.append("invalid:dynamic_assembly_axial_selected_joint_evidence")

    physical_mode = (
        config.mating_contact_mode == DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE
    )
    guidance_contact = report.get("dynamic_assembly_first_guidance_contact_evidence")
    if physical_mode:
        if report.get("dynamic_assembly_mating_filter_evidence") is not None:
            failures.append("invalid:dynamic_assembly_mating_filter_evidence")
        if not _guidance_contact_evidence_valid(
            guidance_contact,
            config=config,
            event_times=event_times,
        ):
            failures.append(
                "invalid:dynamic_assembly_first_guidance_contact_evidence"
            )
    else:
        if (
            guidance_contact is not None
            or report.get("dynamic_assembly_first_selected_contact_evidence")
            is not None
        ):
            failures.append("invalid:dynamic_assembly_fallback_contact_evidence")
        if not _mating_filter_fallback_evidence_valid(
            report,
            config=config,
            event_times=event_times,
        ):
            failures.append("invalid:dynamic_assembly_mating_filter_evidence")

    final_seated = report.get("dynamic_assembly_final_seated_evidence")
    if not _final_seated_evidence_valid(
        final_seated,
        config=config,
        event_times=event_times,
    ):
        failures.append("invalid:dynamic_assembly_final_seated_evidence")

    if not _dock_collision_approximation_evidence_valid(report, config=config):
        failures.append(
            "invalid:dynamic_assembly_dock_collision_approximation_evidence"
        )

    if not _constraint_evidence_valid(report, morphology_graph=morphology_graph):
        failures.append("invalid:dynamic_assembly_constraint_evidence")

    if not _assembly_run_report_valid(
        report.get("dynamic_assembly_assembly_run_report"),
        morphology_graph=morphology_graph,
    ):
        failures.append("invalid:dynamic_assembly_assembly_run_report")

    if not _handover_evidence_valid(
        report.get("dynamic_assembly_controller_handover_samples"),
        acceptance_gate=config.acceptance_gate,
        event_times=event_times,
    ):
        failures.append("invalid:dynamic_assembly_controller_handover_samples")

    attached_steps = report.get("dynamic_assembly_attached_stable_steps")
    attached_required = report.get("dynamic_assembly_attached_required_stable_steps")
    attached_metrics = report.get("dynamic_assembly_attached_max_metrics")
    expected_attached_steps = max(
        1,
        int(math.ceil(config.attached_hold_s / config.simulation_dt_s)),
    )
    attached_limits = {
        "position_error_m": config.attached_position_tolerance_m,
        "attitude_error_rad": config.attached_attitude_tolerance_rad,
        "linear_speed_mps": config.attached_linear_speed_tolerance_mps,
        "angular_speed_radps": config.attached_angular_speed_tolerance_radps,
        "connect_position_error_m": config.control_bridge.fix_axial_tolerance_m,
        "connect_axial_error_m": config.control_bridge.fix_axial_tolerance_m,
        "connect_transverse_error_m": (
            config.control_bridge.fix_transverse_tolerance_m
        ),
        "connect_attitude_error_rad": config.control_bridge.fix_attitude_tolerance_rad,
        "connect_relative_linear_speed_mps": (
            config.control_bridge.fix_relative_linear_speed_tolerance_mps
        ),
        "connect_relative_angular_speed_radps": (
            config.control_bridge.fix_relative_angular_speed_tolerance_radps
        ),
        "dock_joint_position_rad": config.attached_joint_position_tolerance_rad,
        "dock_joint_speed_radps": config.attached_joint_speed_tolerance_radps,
    }
    attached_valid = bool(
        _is_positive_int(attached_required)
        and _is_nonnegative_int(attached_steps)
        and int(attached_required) == expected_attached_steps
        and int(attached_steps) >= int(attached_required)
        and _bounded_metric_dict(attached_metrics, attached_limits)
    )
    if not attached_valid:
        failures.append("invalid:dynamic_assembly_attached_stability_evidence")

    timing_checks = [
        (
            "floor_settle",
            "preflight_vectoring",
            config.floor_settle_duration_s,
        ),
        (
            "constraint_enabled",
            "constraint_verified",
            config.constraint_verify_dwell_s,
        ),
        (
            "attach_handover",
            "attached_hold",
            config.controller_handover_blend_s,
        ),
        (
            "attached_hold",
            (
                "control_graph_split"
                if config.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE
                else "complete"
            ),
            config.attached_hold_s,
        ),
    ]
    if config.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE:
        timing_checks.extend(
            [
                (
                    "split_handover",
                    "unload_dwell",
                    config.controller_handover_blend_s,
                ),
                (
                    "unload_dwell",
                    "constraint_removed",
                    config.detach_unload.unload_dwell_steps * config.simulation_dt_s,
                ),
                (
                    "separation",
                    "post_release_hold",
                    config.separation_distance_m / config.separation_speed_mps,
                ),
                (
                    "post_release_hold",
                    "complete",
                    config.post_release_hold_s,
                ),
            ]
        )
    for start_phase, end_phase, minimum_duration_s in timing_checks:
        if (
            start_phase not in event_times
            or end_phase not in event_times
            or event_times[end_phase] - event_times[start_phase]
            < float(minimum_duration_s) - config.simulation_dt_s - 1.0e-9
        ):
            failures.append(
                f"invalid:dynamic_assembly_lifecycle_timing:{start_phase}:{end_phase}"
            )

    if config.acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE:
        unload_estimate = report.get("dynamic_assembly_unload_estimate")
        unload_decision = report.get("dynamic_assembly_unload_decision")
        follower_id = report.get("dynamic_assembly_follower_module_id")
        edge_id = report.get("dynamic_assembly_edge_id")
        unload_valid = bool(
            report.get("dynamic_assembly_follower_external_contact_scope")
            == "follower_component_all_external_contacts"
            and _is_finite_nonnegative_number(
                report.get("dynamic_assembly_follower_external_contact_max_force_n")
            )
            and float(report["dynamic_assembly_follower_external_contact_max_force_n"])
            < config.detach_external_contact_force_threshold_n
            and report.get(
                "dynamic_assembly_follower_external_contact_invalid_during_unload_count"
            )
            == 0
            and report.get(
                "dynamic_assembly_follower_external_contact_raw_patch_count_during_unload"
            )
            == 0
            and _detach_estimate_valid(
                unload_estimate,
                edge_id=edge_id,
                follower_module_id=follower_id,
                config=config,
            )
            and _detach_decision_valid(unload_decision, config=config)
            and math.isclose(
                float(unload_decision["metrics"]["cut_force_norm_n"]),
                float(unload_estimate["force_norm_n"]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                float(unload_decision["metrics"]["cut_torque_norm_nm"]),
                float(unload_estimate["torque_norm_nm"]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        )
        if not unload_valid:
            failures.append("invalid:dynamic_assembly_unload_evidence")

        final_gap = report.get("dynamic_assembly_final_separation_gap_m")
        minimum_gap = report.get(
            "dynamic_assembly_post_release_min_separation_gap_m"
        )
        if not (
            _is_finite_nonnegative_number(final_gap)
            and float(final_gap) >= 0.8 * config.separation_distance_m
            and _is_finite_nonnegative_number(minimum_gap)
            and float(minimum_gap) >= 0.8 * config.separation_distance_m
        ):
            failures.append("invalid:dynamic_assembly_separation_evidence")

        filter_clearance = report.get(
            "dynamic_assembly_selected_pair_filter_removal_clearance_m"
        )
        final_body_clearance = report.get(
            "dynamic_assembly_final_selected_body_clearance_m"
        )
        minimum_post_release_clearance = report.get(
            "dynamic_assembly_post_release_min_selected_body_clearance_m"
        )
        filter_event = next(
            (
                event
                for event in report.get("dynamic_assembly_events", [])
                if isinstance(event, dict)
                and event.get("phase") == "collision_filter_removed"
            ),
            None,
        )
        filter_event_metrics = (
            filter_event.get("metrics") if isinstance(filter_event, dict) else None
        )
        filter_event_clearance = (
            filter_event_metrics.get("selected_body_clearance_m")
            if isinstance(filter_event_metrics, dict)
            else None
        )
        filter_event_gap = (
            filter_event_metrics.get("separation_gap_m")
            if isinstance(filter_event_metrics, dict)
            else None
        )
        filter_event_steps = (
            filter_event_metrics.get("separation_steps")
            if isinstance(filter_event_metrics, dict)
            else None
        )
        filter_event_time = (
            filter_event.get("time_s") if isinstance(filter_event, dict) else None
        )
        separation_event_time = event_times.get("separation")
        nominal_separation_steps = max(
            1,
            int(
                math.ceil(
                    config.separation_distance_m
                    / config.separation_speed_mps
                    / config.simulation_dt_s
                )
            ),
        )
        if not (
            _is_finite_nonnegative_number(filter_clearance)
            and float(filter_clearance) >= config.release_filter_clearance_m
            and _is_finite_nonnegative_number(final_body_clearance)
            and float(final_body_clearance) >= config.release_filter_clearance_m
            and _is_finite_nonnegative_number(minimum_post_release_clearance)
            and float(minimum_post_release_clearance)
            >= config.release_filter_clearance_m
            and _is_finite_nonnegative_number(filter_event_clearance)
            and math.isclose(
                float(filter_event_clearance),
                float(filter_clearance),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
            and _is_finite_nonnegative_number(filter_event_gap)
            and float(filter_event_gap) >= 0.8 * config.separation_distance_m
            and _is_positive_int(filter_event_steps)
            and nominal_separation_steps
            <= int(filter_event_steps)
            <= 2 * nominal_separation_steps
            and _is_finite_nonnegative_number(filter_event_time)
            and _is_finite_nonnegative_number(separation_event_time)
            and float(filter_event_time) - float(separation_event_time)
            >= nominal_separation_steps * config.simulation_dt_s - 1.0e-9
            and math.isclose(
                float(filter_event_time) - float(separation_event_time),
                int(filter_event_steps) * config.simulation_dt_s,
                rel_tol=1.0e-9,
                abs_tol=config.simulation_dt_s + 1.0e-9,
            )
        ):
            failures.append("invalid:dynamic_assembly_filter_clearance_evidence")
        if not (
            report.get("dynamic_assembly_post_unfilter_selected_contact_count")
            == 0
            and report.get("dynamic_assembly_post_unfilter_raw_invalid_count")
            == 0
        ):
            failures.append("invalid:dynamic_assembly_post_unfilter_contact_evidence")

        post_steps = report.get("dynamic_assembly_post_release_stable_dwell_steps")
        post_required = report.get("dynamic_assembly_post_release_required_dwell_steps")
        expected_post_steps = max(
            1,
            int(math.ceil(config.post_release_hold_s / config.simulation_dt_s)),
        )
        post_release_event = next(
            (
                event
                for event in report.get("dynamic_assembly_events", [])
                if isinstance(event, dict)
                and event.get("phase") == "post_release_hold"
            ),
            None,
        )
        post_release_event_metrics = (
            post_release_event.get("metrics")
            if isinstance(post_release_event, dict)
            else None
        )
        stable_dwell_s = (
            post_release_event_metrics.get("stable_dwell_s")
            if isinstance(post_release_event_metrics, dict)
            else None
        )
        observed_post_release_steps = (
            post_release_event_metrics.get("observed_steps")
            if isinstance(post_release_event_metrics, dict)
            else None
        )
        post_release_event_time = (
            post_release_event.get("time_s")
            if isinstance(post_release_event, dict)
            else None
        )
        complete_event_time = event_times.get("complete")
        post_limits = {
            "position_error_m": config.post_release_position_tolerance_m,
            "attitude_error_rad": config.post_release_attitude_tolerance_rad,
            "linear_speed_mps": config.post_release_linear_speed_tolerance_mps,
            "angular_speed_radps": config.post_release_angular_speed_tolerance_radps,
            "dock_joint_position_rad": (
                config.post_release_joint_position_tolerance_rad
            ),
            "dock_joint_speed_radps": (
                config.post_release_joint_speed_tolerance_radps
            ),
        }
        post_valid = bool(
            _is_positive_int(post_required)
            and _is_nonnegative_int(post_steps)
            and int(post_required) == expected_post_steps
            and int(post_steps) >= int(post_required)
            and _is_positive_int(observed_post_release_steps)
            and int(post_steps)
            <= int(observed_post_release_steps)
            <= 2 * expected_post_steps
            and _is_finite_nonnegative_number(stable_dwell_s)
            and math.isclose(
                float(stable_dwell_s),
                int(post_steps) * config.simulation_dt_s,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            and float(stable_dwell_s)
            >= config.post_release_hold_s - 1.0e-9
            and _is_finite_nonnegative_number(post_release_event_time)
            and _is_finite_nonnegative_number(complete_event_time)
            and math.isclose(
                float(complete_event_time) - float(post_release_event_time),
                float(stable_dwell_s),
                rel_tol=1.0e-9,
                abs_tol=config.simulation_dt_s + 1.0e-9,
            )
            and _bounded_metric_dict(
                report.get("dynamic_assembly_post_release_max_metrics"),
                post_limits,
            )
        )
        if not post_valid:
            failures.append("invalid:dynamic_assembly_post_release_stability_evidence")

    return failures


def _axial_selected_joint_evidence_valid(report: dict[str, Any]) -> bool:
    evidence = report.get("dynamic_assembly_axial_selected_joint_evidence")
    if not isinstance(evidence, dict):
        return False
    by_module = evidence.get("by_module")
    expected_roles = {
        str(report.get("dynamic_assembly_leader_module_id")): "leader",
        str(report.get("dynamic_assembly_follower_module_id")): "follower",
    }
    if not (
        evidence.get("evidence_version")
        == "axial_selected_dock_joint_zero_target_v1"
        and _is_positive_int(evidence.get("sample_count"))
        and evidence.get("all_targets_zero") is True
        and isinstance(by_module, dict)
        and set(by_module) == set(expected_roles)
    ):
        return False
    target_maxima = (
        "max_abs_joint_position_target_rad",
        "max_abs_joint_velocity_target_radps",
        "max_abs_joint_effort_bias_target_nm",
    )
    measured_maxima = (
        "max_abs_measured_joint_position_rad",
        "max_abs_measured_joint_velocity_radps",
        "max_abs_joint_computed_torque_nm",
        "max_abs_joint_applied_torque_nm",
        "max_selected_body_minus_root_angular_speed_radps",
    )
    pose_fields = (
        "first_root_pose_world",
        "last_root_pose_world",
        "first_selected_body_pose_in_root",
        "last_selected_body_pose_in_root",
    )
    for module_key, expected_role in expected_roles.items():
        values = by_module.get(module_key)
        if not (
            isinstance(values, dict)
            and values.get("role") == expected_role
            and isinstance(values.get("joint_id"), str)
            and bool(values.get("joint_id"))
            and isinstance(values.get("resolved_joint_name"), str)
            and bool(values.get("resolved_joint_name"))
            and all(
                _is_finite_nonnegative_number(values.get(key))
                and float(values[key]) <= 1.0e-12
                for key in target_maxima
            )
            and all(
                _is_finite_nonnegative_number(values.get(key))
                for key in measured_maxima
            )
            and all(_finite_sequence(values.get(key), length=7) for key in pose_fields)
        ):
            return False
    return True


def _guidance_contact_evidence_valid(
    evidence: object,
    *,
    config: DynamicAssemblyIsaacConfig,
    event_times: dict[str, float],
) -> bool:
    """Validate safe first contact in the funnel-guidance envelope.

    The first physical patch is intentionally *not* required to lie near either
    final connect plane.  For the Holon connector it normally occurs on the
    pitch-funnel wall roughly 46 mm before the connect frames become coincident.
    """

    if not _raw_selected_pair_contact_evidence_valid(evidence, config=config):
        return False
    assert isinstance(evidence, dict)
    time_s = evidence.get("time_s")
    time_valid = _is_finite_nonnegative_number(time_s)
    if time_valid and "axial_approach" in event_times and "fix_ready" in event_times:
        time_valid = bool(
            event_times["axial_approach"] <= float(time_s) <= event_times["fix_ready"]
        )
    return bool(
        evidence.get("guidance_contact_valid") is True
        and _is_finite_nonnegative_number(evidence.get("guidance_axial_gap_m"))
        and float(evidence["guidance_axial_gap_m"])
        <= config.guidance_contact_max_axial_gap_m
        and _is_finite_nonnegative_number(
            evidence.get("guidance_transverse_error_m")
        )
        and float(evidence["guidance_transverse_error_m"])
        <= config.guidance_contact_max_transverse_error_m
        and _is_finite_nonnegative_number(
            evidence.get("guidance_attitude_error_rad")
        )
        and float(evidence["guidance_attitude_error_rad"])
        <= config.guidance_contact_max_attitude_error_rad
        and evidence.get("guidance_contact_max_axial_gap_m")
        == config.guidance_contact_max_axial_gap_m
        and evidence.get("guidance_contact_max_transverse_error_m")
        == config.guidance_contact_max_transverse_error_m
        and evidence.get("guidance_contact_max_attitude_error_rad")
        == config.guidance_contact_max_attitude_error_rad
        and _finite_sequence(evidence.get("leader_connect_pose_world"), length=7)
        and _finite_sequence(evidence.get("follower_connect_pose_world"), length=7)
        and time_valid
    )


def _mating_filter_fallback_evidence_valid(
    report: dict[str, Any],
    *,
    config: DynamicAssemblyIsaacConfig,
    event_times: dict[str, float],
) -> bool:
    """Validate exact-pair filtering before the pre-align propagation dwell."""

    evidence = report.get("dynamic_assembly_mating_filter_evidence")
    constraint = report.get("dynamic_assembly_constraint_spec")
    if not isinstance(evidence, dict) or not isinstance(constraint, dict):
        return False
    leader_path = constraint.get("leader_body_path")
    follower_path = constraint.get("follower_body_path")
    time_s = evidence.get("time_s")
    prealign_time_s = event_times.get("prealign_dwell")
    axial_time_s = event_times.get("axial_approach")
    timing_valid = bool(
        _is_finite_nonnegative_number(time_s)
        and _is_finite_nonnegative_number(prealign_time_s)
        and _is_finite_nonnegative_number(axial_time_s)
        and math.isclose(
            float(time_s),
            float(prealign_time_s),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and float(axial_time_s) - float(prealign_time_s)
        >= config.control_bridge.prealign_dwell_s
        - config.simulation_dt_s
        - 1.0e-9
    )
    before_leader = evidence.get("leader_targets_before")
    before_follower = evidence.get("follower_targets_before")
    after_leader = evidence.get("leader_targets_after")
    after_follower = evidence.get("follower_targets_after")
    target_lists_valid = all(
        isinstance(values, list)
        and all(isinstance(value, str) and value for value in values)
        for values in (
            before_leader,
            before_follower,
            after_leader,
            after_follower,
        )
    )
    if not target_lists_valid or not isinstance(leader_path, str) or not isinstance(
        follower_path, str
    ):
        return False
    assert isinstance(before_leader, list)
    assert isinstance(before_follower, list)
    assert isinstance(after_leader, list)
    assert isinstance(after_follower, list)
    return bool(
        evidence.get("evidence_version")
        == DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT
        and evidence.get("mating_contact_mode")
        == DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE
        and evidence.get("scope") == "selected_dock_body_pair_only"
        and evidence.get("apply_phase") == "prealign_dwell"
        and evidence.get("applied_before_first_axial_physics_step") is True
        and evidence.get("apply_verified") is True
        and evidence.get("environment_collisions_preserved") is True
        and evidence.get("other_body_pair_collisions_preserved") is True
        and evidence.get("leader_body_prim_valid") is True
        and evidence.get("follower_body_prim_valid") is True
        and evidence.get("leader_is_rigid_body") is True
        and evidence.get("follower_is_rigid_body") is True
        and evidence.get("leader_body_path") == leader_path
        and evidence.get("follower_body_path") == follower_path
        and follower_path not in before_leader
        and leader_path not in before_follower
        and sorted(after_leader) == sorted([*before_leader, follower_path])
        and sorted(after_follower) == sorted(before_follower)
        and evidence.get("added_leader_targets") == [follower_path]
        and evidence.get("removed_leader_targets") == []
        and evidence.get("added_follower_targets") == []
        and evidence.get("removed_follower_targets") == []
        and _is_nonnegative_int(evidence.get("command_index"))
        and evidence.get("selected_contact_count_after_filter") == 0
        and evidence.get("selected_contact_violation_count") == 0
        and timing_valid
    )


def _final_seated_evidence_valid(
    evidence: object,
    *,
    config: DynamicAssemblyIsaacConfig,
    event_times: dict[str, float],
) -> bool:
    """Validate the independent strict, continuously-dwelled pre-fix gate."""

    physical_mode = (
        config.mating_contact_mode == DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE
    )
    if physical_mode:
        if not _raw_selected_pair_contact_evidence_valid(evidence, config=config):
            return False
    elif not isinstance(evidence, dict):
        return False
    assert isinstance(evidence, dict)
    if not physical_mode and any(
        key in evidence
        for key in (
            "selected_contact_points_world",
            "selected_contact_normals_world",
            "selected_patch_forces_n",
            "selected_patch_separations_m",
        )
    ):
        return False
    time_s = evidence.get("time_s")
    time_valid = _is_finite_nonnegative_number(time_s)
    if time_valid and "fix_ready" in event_times and "constraint_enabled" in event_times:
        time_valid = bool(
            event_times["fix_ready"] - config.simulation_dt_s - 1.0e-9
            <= float(time_s)
            <= event_times["constraint_enabled"] + config.simulation_dt_s + 1.0e-9
        )
    return bool(
        evidence.get("evidence_version")
        == (
            "final_seated_contact_v1"
            if physical_mode
            else "final_seated_alignment_v1"
        )
        and evidence.get("selected_pair_scope") == "selected_dock_body_pair"
        and evidence.get("selected_pair_contact_required") is physical_mode
        and evidence.get("selected_pair_contact_observed") is physical_mode
        and evidence.get("final_seated_valid") is True
        and evidence.get("leader_qp_feasible") is True
        and evidence.get("follower_qp_feasible") is True
        and _is_finite_nonnegative_number(evidence.get("axial_error_m"))
        and float(evidence["axial_error_m"])
        <= config.control_bridge.fix_axial_tolerance_m
        and _is_finite_nonnegative_number(evidence.get("transverse_error_m"))
        and float(evidence["transverse_error_m"])
        <= config.control_bridge.fix_transverse_tolerance_m
        and _is_finite_nonnegative_number(evidence.get("position_error_m"))
        and float(evidence["position_error_m"])
        <= config.control_bridge.fix_axial_tolerance_m
        and _is_finite_nonnegative_number(evidence.get("attitude_error_rad"))
        and float(evidence["attitude_error_rad"])
        <= config.control_bridge.fix_attitude_tolerance_rad
        and _is_finite_nonnegative_number(
            evidence.get("relative_linear_speed_mps")
        )
        and float(evidence["relative_linear_speed_mps"])
        <= config.control_bridge.fix_relative_linear_speed_tolerance_mps
        and _is_finite_nonnegative_number(
            evidence.get("relative_angular_speed_radps")
        )
        and float(evidence["relative_angular_speed_radps"])
        <= config.control_bridge.fix_relative_angular_speed_tolerance_radps
        and _is_finite_nonnegative_number(evidence.get("continuous_strict_dwell_s"))
        and float(evidence["continuous_strict_dwell_s"])
        >= config.control_bridge.selected_contact_dwell_s
        - config.simulation_dt_s
        - 1.0e-9
        and evidence.get("required_strict_dwell_s")
        == config.control_bridge.selected_contact_dwell_s
        and _finite_sequence(evidence.get("leader_connect_pose_world"), length=7)
        and _finite_sequence(evidence.get("follower_connect_pose_world"), length=7)
        and _finite_sequence(evidence.get("leader_connect_twist_world"), length=6)
        and _finite_sequence(evidence.get("follower_connect_twist_world"), length=6)
        and time_valid
    )


def _raw_selected_pair_contact_evidence_valid(
    evidence: object,
    *,
    config: DynamicAssemblyIsaacConfig,
) -> bool:
    if not isinstance(evidence, dict):
        return False
    raw_count = evidence.get("selected_raw_contact_count")
    points = evidence.get("selected_contact_points_world")
    normals = evidence.get("selected_contact_normals_world")
    forces = evidence.get("selected_patch_forces_n")
    separations = evidence.get("selected_patch_separations_m")
    sequence_lengths_valid = bool(
        _is_positive_int(raw_count)
        and isinstance(points, list)
        and isinstance(normals, list)
        and isinstance(forces, list)
        and isinstance(separations, list)
        and len(points) == len(normals) == len(forces) == len(separations) == int(raw_count)
    )
    if not sequence_lengths_valid:
        return False
    finite_arrays = bool(
        all(_finite_sequence(point, length=3) for point in points)
        and all(_finite_sequence(normal, length=3) for normal in normals)
        and all(_is_finite_number(force) for force in forces)
        and all(_is_finite_number(separation) for separation in separations)
    )
    if not finite_arrays:
        return False
    physical_patch_count = sum(
        abs(float(force)) > 1.0e-3 or float(separation) <= 0.0
        for force, separation in zip(forces, separations, strict=True)
    )
    return bool(
        evidence.get("selected_pair_scope") == "selected_dock_body_pair"
        and evidence.get("selected_pair_exact_body_match") is True
        and evidence.get("selected_contact") is True
        and evidence.get("raw_contact_valid") is True
        and evidence.get("raw_contact_nonfinite") is False
        and evidence.get("raw_contact_layout_invalid") is False
        and evidence.get("raw_contact_saturated") is False
        and evidence.get("unintended_contact") is False
        and evidence.get("unintended_raw_contact_count") == 0
        and physical_patch_count > 0
        and _is_positive_int(evidence.get("selected_physical_patch_count"))
        and int(evidence["selected_physical_patch_count"]) == physical_patch_count
        and _is_nonnegative_int(evidence.get("raw_contact_count"))
        and int(evidence["raw_contact_count"]) >= int(raw_count)
        and _is_positive_int(evidence.get("raw_contact_capacity"))
        and int(evidence["raw_contact_capacity"])
        > int(evidence["raw_contact_count"])
        and _is_finite_number(evidence.get("selected_min_separation_m"))
        and math.isclose(
            float(evidence["selected_min_separation_m"]),
            min(float(value) for value in separations),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
        and _is_finite_nonnegative_number(evidence.get("selected_force_n"))
        and float(evidence["selected_force_n"])
        <= config.control_bridge.max_selected_contact_force_n
        and _is_finite_nonnegative_number(evidence.get("selected_penetration_m"))
        and float(evidence["selected_penetration_m"])
        <= config.control_bridge.max_selected_contact_penetration_m
        and math.isclose(
            float(evidence["selected_penetration_m"]),
            max(0.0, -float(evidence["selected_min_separation_m"])),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
    )


def _dock_collision_approximation_evidence_valid(
    report: dict[str, Any],
    *,
    config: DynamicAssemblyIsaacConfig,
) -> bool:
    """Validate authored-token evidence, not funnel-cavity preservation.

    ``convexDecomposition`` is a necessary generated-USD property, but this
    report evidence alone does not prove that the resulting convex pieces keep
    the pitch-funnel cavity open.  The physical contact/seating gates provide
    the independent runtime evidence for that behavior.
    """

    evidence = report.get("dynamic_assembly_dock_collision_approximation_evidence")
    if not isinstance(evidence, dict):
        return False
    composed_count = report.get("dynamic_assembly_dock_collision_composed_prim_count")
    paths = evidence.get("composed_prim_paths")
    authored_paths = evidence.get("authored_prim_paths")
    return bool(
        report.get("dynamic_assembly_dock_collision_approximation_verified") is True
        and report.get("dynamic_assembly_dock_collision_approximation_token")
        == "convexDecomposition"
        and _is_positive_int(composed_count)
        and evidence.get("verified") is True
        and evidence.get("requested_collision_type") == "Convex Decomposition"
        and evidence.get("requested_approximation_token") == "convexDecomposition"
        and evidence.get("physx_convex_decomposition_api_verified") is True
        and evidence.get("max_convex_hulls")
        == config.dock_convex_decomposition_max_hulls
        and evidence.get("shrink_wrap")
        is config.dock_convex_decomposition_shrink_wrap
        and _is_positive_int(evidence.get("authored_prim_count"))
        and _is_positive_int(evidence.get("composed_prim_count"))
        and int(evidence["composed_prim_count"]) == int(composed_count)
        and isinstance(paths, list)
        and len(paths) == int(composed_count)
        and all(isinstance(path, str) and path for path in paths)
        and len(set(paths)) == len(paths)
        and isinstance(authored_paths, list)
        and len(authored_paths) == int(evidence["authored_prim_count"])
        and all(isinstance(path, str) and path for path in authored_paths)
        and len(set(authored_paths)) == len(authored_paths)
        and isinstance(evidence.get("original_approximation_tokens"), dict)
        and len(evidence["original_approximation_tokens"])
        == int(evidence["authored_prim_count"])
    )


def _constraint_evidence_valid(
    report: dict[str, Any],
    *,
    morphology_graph: MorphologyGraph,
) -> bool:
    failures = report.get("dynamic_assembly_constraint_identity_failures")
    spec_raw = report.get("dynamic_assembly_constraint_spec")
    if failures != [] or not isinstance(spec_raw, dict):
        return False
    try:
        spec = DynamicDockConstraintSpec.from_dict(spec_raw)
        spec.validate()
    except (SchemaValidationError, TypeError, ValueError):
        return False
    edge = morphology_graph.dock_edges[0]
    return bool(
        spec.edge_id == edge.edge_id
        and {spec.leader_module_id, spec.follower_module_id}
        == {edge.src_module_id, edge.dst_module_id}
        and {spec.leader_port_id, spec.follower_port_id}
        == {edge.src_port_id, edge.dst_port_id}
        and spec.leader_module_id == report.get("dynamic_assembly_leader_module_id")
        and spec.follower_module_id == report.get("dynamic_assembly_follower_module_id")
    )


def _assembly_run_report_valid(
    value: object,
    *,
    morphology_graph: MorphologyGraph,
) -> bool:
    if not isinstance(value, dict):
        return False
    plan = value.get("plan")
    plan_steps = plan.get("steps") if isinstance(plan, dict) else None
    step_results = value.get("step_results")
    expected_step_types = ["move_to_staging", "align_ports", "dock", "verify_attach"]
    metrics = value.get("metrics")
    return bool(
        value.get("success") is True
        and value.get("state_matches_target") is True
        and value.get("aborted") is False
        and value.get("failure_reason") is None
        and value.get("failures") == []
        and value.get("completed_step_count") == 4
        and value.get("attached_edge_count") == 1
        and value.get("target_edge_count") == 1
        and value.get("retry_count") == 0
        and value.get("abort_count") == 0
        and value.get("executed_step_types") == expected_step_types
        and isinstance(plan, dict)
        and plan.get("target_graph_id") == morphology_graph.graph_id
        and isinstance(plan_steps, list)
        and [step.get("step_type") for step in plan_steps if isinstance(step, dict)]
        == expected_step_types
        and isinstance(step_results, list)
        and len(step_results) == 4
        and all(
            isinstance(result, dict)
            and result.get("success") is True
            and result.get("step_id") == index
            for index, result in enumerate(step_results)
        )
        and isinstance(metrics, dict)
        and metrics.get("target_module_count") == 2.0
        and metrics.get("assembled_module_count") == 2.0
        and metrics.get("target_edge_count") == 1.0
        and metrics.get("attached_edge_count") == 1.0
        and metrics.get("module_set_matches_target") == 1.0
        and metrics.get("dock_edge_set_matches_target") == 1.0
        and metrics.get("port_occupancy_matches_target") == 1.0
        and metrics.get("state_matches_target") == 1.0
    )


def _handover_evidence_valid(
    value: object,
    *,
    acceptance_gate: str,
    event_times: dict[str, float],
) -> bool:
    if not isinstance(value, list):
        return False
    required = {
        "components_to_assembled": ("attach_handover", "attached_hold"),
    }
    if acceptance_gate == DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE:
        required["assembled_to_components"] = ("split_handover", "unload_dwell")
    if any(
        not isinstance(sample, dict) or sample.get("direction") not in required
        for sample in value
    ):
        return False
    for direction, (start_phase, end_phase) in required.items():
        samples = [sample for sample in value if sample.get("direction") == direction]
        if not samples or start_phase not in event_times or end_phase not in event_times:
            return False
        alphas = [sample.get("alpha") for sample in samples]
        times = [sample.get("time_s") for sample in samples]
        if not (
            all(_is_finite_number(alpha) and 0.0 < float(alpha) <= 1.0 for alpha in alphas)
            and [float(alpha) for alpha in alphas] == sorted(float(alpha) for alpha in alphas)
            and math.isclose(float(alphas[-1]), 1.0, abs_tol=1.0e-9)
            and all(_is_finite_nonnegative_number(time_s) for time_s in times)
            and [float(time_s) for time_s in times] == sorted(float(time_s) for time_s in times)
            and event_times[start_phase] <= float(times[0])
            and float(times[-1]) <= event_times[end_phase] + 1.0e-9
            and all(sample.get("source_qp_feasible") is True for sample in samples)
            and all(sample.get("target_qp_feasible") is True for sample in samples)
        ):
            return False
    return True


def _detach_estimate_valid(
    value: object,
    *,
    edge_id: object,
    follower_module_id: object,
    config: DynamicAssemblyIsaacConfig,
) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("valid") is True
        and value.get("failure_reason") is None
        and _is_nonnegative_int(edge_id)
        and value.get("edge_id") == edge_id
        and _is_nonnegative_int(follower_module_id)
        and value.get("follower_module_ids") == [follower_module_id]
        and _finite_sequence(value.get("wrench_follower_com_body"), length=6)
        and _finite_sequence(value.get("wrench_follower_dock_frame"), length=6)
        and _is_finite_nonnegative_number(value.get("force_norm_n"))
        and float(value["force_norm_n"]) <= config.detach_unload.force_threshold_n
        and _is_finite_nonnegative_number(value.get("torque_norm_nm"))
        and float(value["torque_norm_nm"]) <= config.detach_unload.torque_threshold_nm
    )


def _detach_decision_valid(
    value: object,
    *,
    config: DynamicAssemblyIsaacConfig,
) -> bool:
    if not isinstance(value, dict):
        return False
    metrics = value.get("metrics")
    limits = {
        "cut_force_norm_n": config.detach_unload.force_threshold_n,
        "cut_torque_norm_nm": config.detach_unload.torque_threshold_nm,
        "relative_position_error_m": config.detach_unload.relative_position_error_threshold_m,
        "relative_rotation_error_rad": config.detach_unload.relative_rotation_error_threshold_rad,
        "relative_linear_speed_mps": config.detach_unload.relative_linear_speed_threshold_mps,
        "relative_angular_speed_radps": config.detach_unload.relative_angular_speed_threshold_radps,
    }
    return bool(
        value.get("ready_to_release") is True
        and _is_positive_int(value.get("consecutive_unload_steps"))
        and int(value["consecutive_unload_steps"]) >= config.detach_unload.unload_dwell_steps
        and value.get("failure_reasons") == []
        and _bounded_metric_dict(metrics, limits)
        and metrics.get("required_unload_dwell_steps")
        == float(config.detach_unload.unload_dwell_steps)
    )


def _bounded_metric_dict(value: object, limits: dict[str, float]) -> bool:
    return bool(
        isinstance(value, dict)
        and all(
            _is_finite_nonnegative_number(value.get(key))
            and float(value[key]) <= float(limit)
            for key, limit in limits.items()
        )
    )


def _finite_sequence(value: object, *, length: int) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_is_finite_number(item) for item in value)
    )


def _is_finite_number(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_finite_nonnegative_number(value: object) -> bool:
    return _is_finite_number(value) and float(value) >= 0.0


def _is_finite_positive_number(value: object) -> bool:
    return _is_finite_number(value) and float(value) > 0.0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_nonnegative_int(value) and int(value) > 0


def _validate_roundtrip_graph(graph: MorphologyGraph) -> None:
    if len(graph.modules) != 2 or len(graph.dock_edges) != 1:
        raise SchemaValidationError(
            "first dynamic assembly round-trip gate requires exactly two modules and one DockEdge"
        )
    if graph.dock_edges[0].latch_state not in {"planned", "detached"}:
        raise SchemaValidationError("dynamic assembly input DockEdge must not already be attached")


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_json_command(command: list[str], timeout_s: float) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def drain(stream, sink: list[str], *, forward_progress: bool) -> None:
        try:
            for line in stream:
                sink.append(line)
                if (
                    forward_progress
                    and line.startswith(DYNAMIC_ASSEMBLY_PROGRESS_PREFIX)
                ):
                    print(line.rstrip("\n"), file=sys.stderr, flush=True)
        finally:
            stream.close()

    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        raise RuntimeError("dynamic assembly probe pipes were not created")
    stdout_thread = threading.Thread(
        target=drain,
        args=(process.stdout, stdout_lines),
        kwargs={"forward_progress": False},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, stderr_lines),
        kwargs={"forward_progress": True},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        raise
    stdout_thread.join()
    stderr_thread.join()

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    payload: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        raise RuntimeError(
            "dynamic assembly probe emitted no JSON report: "
            + stderr[-1000:]
        )
    payload["command_returncode"] = returncode
    payload["command_stderr_tail"] = stderr[-4000:]
    return payload


__all__ = [
    "DYNAMIC_ASSEMBLY_ACCEPTANCE_GATES",
    "DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE",
    "DYNAMIC_ASSEMBLY_FILTER_FALLBACK_ACCEPTANCE_CONTRACT",
    "DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE",
    "DYNAMIC_ASSEMBLY_MATING_MODES",
    "DYNAMIC_ASSEMBLY_PHYSICAL_ACCEPTANCE_CONTRACT",
    "DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE",
    "DYNAMIC_ASSEMBLY_PROGRESS_INTERVAL_S",
    "DYNAMIC_ASSEMBLY_PROGRESS_PREFIX",
    "DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE",
    "DYNAMIC_ASSEMBLY_ROUNDTRIP_VERSION",
    "DynamicAssemblyIsaacConfig",
    "DynamicAssemblyIsaacEnv",
    "DynamicAssemblyIsaacResult",
    "DynamicSeparationLifecycle",
    "dynamic_assembly_acceptance_contract",
    "dynamic_assembly_progress_due",
    "dynamic_assembly_report_failures",
    "format_dynamic_assembly_progress",
]
