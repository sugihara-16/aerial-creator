from __future__ import annotations

"""Real-Isaac boundary for the P4-full Order 4 free-flight pi_H runtime."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order4 import (
    ORDER4_FREE_FLIGHT_REPORT_VERSION,
    ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
    Order4DeterministicPlannerConfig,
    Order4FreeFlightMission,
    Order4TrajectoryRuntimeStep,
)
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL, ContactWrenchTrajectory
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffEnv
from amsrr.utils.hashing import hash_file, stable_hash


ORDER4_FREE_FLIGHT_ENV_VERSION = "order4_free_flight_isaac_env_v1"


@dataclass
class Order4IsaacFreeFlightConfig(SchemaBase):
    mission: Order4FreeFlightMission
    planner: Order4DeterministicPlannerConfig = field(
        default_factory=Order4DeterministicPlannerConfig
    )
    pi_l_checkpoint_path: str | None = None
    expected_pi_l_checkpoint_sha256: str | None = None
    record_runtime_steps: bool = True
    command_timeout_s: float = 600.0
    env_version: str = ORDER4_FREE_FLIGHT_ENV_VERSION

    def validate(self) -> None:
        if self.env_version != ORDER4_FREE_FLIGHT_ENV_VERSION:
            raise SchemaValidationError(
                "Order4IsaacFreeFlightConfig.env_version mismatch"
            )
        self.mission.validate()
        self.planner.validate()
        if (self.pi_l_checkpoint_path is None) != (
            self.expected_pi_l_checkpoint_sha256 is None
        ):
            raise SchemaValidationError(
                "Order4 pi_L checkpoint path and sha256 must be provided together"
            )
        if self.expected_pi_l_checkpoint_sha256 is not None and not _is_sha256(
            self.expected_pi_l_checkpoint_sha256
        ):
            raise SchemaValidationError(
                "Order4 expected pi_L checkpoint sha256 is invalid"
            )
        if type(self.record_runtime_steps) is not bool:
            raise SchemaValidationError(
                "Order4IsaacFreeFlightConfig.record_runtime_steps must be boolean"
            )
        if not math.isfinite(float(self.command_timeout_s)) or self.command_timeout_s <= 0.0:
            raise SchemaValidationError(
                "Order4IsaacFreeFlightConfig.command_timeout_s must be finite and positive"
            )


@dataclass
class Order4IsaacFreeFlightResult(SchemaBase):
    env_version: str
    graph_id: str
    mission_hash: str
    dry_run: bool
    attempted: bool
    isaac_backed: bool
    passed: bool
    report_validation_failures: list[str]
    report: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def validate(self) -> None:
        if self.env_version != ORDER4_FREE_FLIGHT_ENV_VERSION:
            raise SchemaValidationError(
                "Order4IsaacFreeFlightResult.env_version mismatch"
            )
        if not self.graph_id:
            raise SchemaValidationError(
                "Order4IsaacFreeFlightResult.graph_id must be non-empty"
            )
        if not _is_sha256(self.mission_hash):
            raise SchemaValidationError(
                "Order4IsaacFreeFlightResult.mission_hash must be sha256"
            )
        if self.passed and (self.dry_run or not self.isaac_backed):
            raise SchemaValidationError(
                "Order4 pass requires a non-dry real-Isaac result"
            )


class Order4IsaacFreeFlightEnv:
    def __init__(
        self,
        *,
        config: Order4IsaacFreeFlightConfig,
        takeoff_env: RandomMorphologyTakeoffEnv,
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
        command_executor: Callable[[list[str], float], dict[str, Any]] | None = None,
    ) -> None:
        config.validate()
        if takeoff_env.config.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "Order4 Isaac execution requires centroidal_local_joint_v2"
            )
        if viewer not in {None, "kit"}:
            raise ValueError("Order4 viewer must be None or 'kit'")
        if keep_open_after_rollout_s < 0.0:
            raise ValueError("Order4 post-rollout hold must be non-negative")
        if viewer is None and (realtime_playback or keep_open_after_rollout_s > 0.0):
            raise ValueError(
                "Order4 real-time playback and post-rollout hold require viewer='kit'"
            )
        self.config = config
        self.takeoff_env = takeoff_env
        self.viewer = viewer
        self.realtime_playback = realtime_playback
        self.keep_open_after_rollout_s = float(keep_open_after_rollout_s)
        self.command_executor = command_executor or takeoff_env.command_executor
        checkpoint_path = self.config.pi_l_checkpoint_path
        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            if hash_file(checkpoint_path) != self.config.expected_pi_l_checkpoint_sha256:
                raise SchemaValidationError("Order4 pi_L checkpoint sha256 mismatch")

    @property
    def requested_steps(self) -> int:
        return max(
            1,
            int(
                math.ceil(
                    self.config.mission.mission_timeout_s
                    / self.takeoff_env.config.simulation_dt_s
                )
            )
            + 1,
        )

    def build_probe_command(self, morphology_graph: MorphologyGraph) -> list[str]:
        command = self.takeoff_env.build_probe_command(morphology_graph)
        replacements = {
            "--steps": str(self.requested_steps),
            "--takeoff-settle-duration-s": str(
                self.config.planner.floor_settle_duration_s
            ),
            "--takeoff-settle-dwell-duration-s": str(
                self.config.planner.floor_settle_dwell_s
            ),
            "--takeoff-ramp-duration-s": str(self.config.planner.takeoff_duration_s),
            "--takeoff-hover-height-delta-m": str(
                self.config.mission.hover_height_delta_m
            ),
            "--hover-hold-duration-s": str(
                self.config.mission.final_hover_hold_s
            ),
            "--takeoff-hover-acquisition-timeout-s": str(
                self.config.planner.hover_acquisition_timeout_s
            ),
            "--hover-position-tolerance-m": str(
                self.config.planner.position_tolerance_m
            ),
            "--hover-attitude-tolerance-rad": str(
                self.config.planner.attitude_tolerance_rad
            ),
            "--takeoff-hover-linear-speed-threshold-mps": str(
                self.config.planner.linear_speed_tolerance_mps
            ),
            "--takeoff-hover-angular-speed-threshold-rad-s": str(
                self.config.planner.angular_speed_tolerance_rad_s
            ),
        }
        for flag, value in replacements.items():
            _replace_command_argument(command, flag, value)
        command.extend(
            [
                "--order4-free-flight-mission-json",
                self.config.mission.to_canonical_json(),
                "--order4-planner-config-json",
                self.config.planner.to_json(),
            ]
        )
        if self.config.pi_l_checkpoint_path is not None:
            command.extend(
                [
                    "--order4-pi-l-checkpoint-path",
                    self.config.pi_l_checkpoint_path,
                ]
            )
        if not self.config.record_runtime_steps:
            command.append("--no-order4-record-runtime-steps")
        if self.viewer is not None:
            command.extend(["--viz", self.viewer])
        if self.realtime_playback:
            command.append("--realtime-playback")
        if self.keep_open_after_rollout_s > 0.0:
            command.extend(
                [
                    "--keep-open-after-smoke-s",
                    str(self.keep_open_after_rollout_s),
                ]
            )
        return command

    def run(
        self,
        morphology_graph: MorphologyGraph,
        *,
        dry_run: bool = True,
        check_availability: bool = True,
    ) -> Order4IsaacFreeFlightResult:
        # Placement is also the current-URDF connect-frame and morphology gate.
        self.takeoff_env.placement_for(morphology_graph)
        if dry_run:
            return Order4IsaacFreeFlightResult(
                env_version=self.config.env_version,
                graph_id=morphology_graph.graph_id,
                mission_hash=self.config.mission.mission_hash,
                dry_run=True,
                attempted=False,
                isaac_backed=False,
                passed=False,
                report_validation_failures=[],
                report={"probe_command": self.build_probe_command(morphology_graph)},
            )
        if check_availability:
            availability = self.takeoff_env.backend.availability()
            if not availability.available:
                return Order4IsaacFreeFlightResult(
                    env_version=self.config.env_version,
                    graph_id=morphology_graph.graph_id,
                    mission_hash=self.config.mission.mission_hash,
                    dry_run=False,
                    attempted=False,
                    isaac_backed=False,
                    passed=False,
                    report_validation_failures=list(availability.missing_reasons),
                    failure_reason=",".join(availability.missing_reasons),
                )
        try:
            report = self.command_executor(
                self.build_probe_command(morphology_graph),
                self.config.command_timeout_s,
            )
        except Exception as exc:  # pragma: no cover - subprocess-specific failure.
            return Order4IsaacFreeFlightResult(
                env_version=self.config.env_version,
                graph_id=morphology_graph.graph_id,
                mission_hash=self.config.mission.mission_hash,
                dry_run=False,
                attempted=True,
                isaac_backed=True,
                passed=False,
                report_validation_failures=["probe_execution_failed"],
                failure_reason=str(exc),
            )
        failures = order4_free_flight_report_failures(
            report,
            morphology_graph=morphology_graph,
            config=self.config,
            takeoff_env=self.takeoff_env,
            requested_steps=self.requested_steps,
        )
        return Order4IsaacFreeFlightResult(
            env_version=self.config.env_version,
            graph_id=morphology_graph.graph_id,
            mission_hash=self.config.mission.mission_hash,
            dry_run=False,
            attempted=True,
            isaac_backed=report.get("isaac_backed") is True,
            passed=not failures,
            report_validation_failures=failures,
            report=report,
            failure_reason=(
                None
                if not failures
                else "order4_report_validation_failed:" + ",".join(failures)
            ),
        )


def order4_free_flight_report_failures(
    report: dict[str, Any],
    *,
    morphology_graph: MorphologyGraph,
    config: Order4IsaacFreeFlightConfig,
    takeoff_env: RandomMorphologyTakeoffEnv,
    requested_steps: int,
) -> list[str]:
    failures: list[str] = []

    def require_exact(key: str, expected: Any) -> None:
        if key not in report:
            failures.append(f"missing:{key}")
        elif type(report[key]) is not type(expected) or report[key] != expected:
            failures.append(f"mismatch:{key}")

    def require_true(key: str) -> None:
        require_exact(key, True)

    def require_false(key: str) -> None:
        require_exact(key, False)

    def require_zero_count(key: str) -> None:
        require_exact(key, 0)

    require_true("spawn_passed")
    require_true("isaac_backed")
    require_true("command_applied")
    require_true("command_probe_passed")
    require_exact("command_returncode", 0)
    require_true("order4_free_flight_enabled")
    require_true("order4_free_flight_passed")
    require_exact("order4_free_flight_report_version", ORDER4_FREE_FLIGHT_REPORT_VERSION)
    require_exact("order4_free_flight_mission", config.mission.to_dict())
    require_exact("order4_free_flight_mission_hash", config.mission.mission_hash)
    require_exact("order4_free_flight_planner_config", config.planner.to_dict())
    require_exact(
        "order4_free_flight_planner_config_hash",
        stable_hash(config.planner),
    )
    require_true("order4_free_flight_deterministic_pi_h")
    require_exact(
        "order4_free_flight_pi_h_scope",
        "free_flight_only_no_contact_planning",
    )
    require_exact(
        "order4_free_flight_trajectory_runtime_version",
        ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
    )
    require_exact("order4_free_flight_final_phase", "complete")
    require_exact("order4_free_flight_progress_ratio", 1.0)
    require_exact(
        "order4_free_flight_waypoint_count",
        len(config.mission.waypoints),
    )
    require_exact(
        "order4_free_flight_completed_waypoint_count",
        len(config.mission.waypoints),
    )
    require_true("order4_free_flight_time_origin_valid")
    require_exact(
        "order4_free_flight_reachability_status",
        "not_applicable_no_active_assignments",
    )
    require_zero_count("order4_free_flight_max_active_assignment_count")
    require_false("order4_free_flight_safe_hold_active")
    require_exact("order4_free_flight_failure_reason", None)
    require_true("order4_free_flight_existing_actor_progress_unchanged")
    require_true("random_morphology_takeoff_smoke_passed")
    require_true("random_morphology_takeoff_settle_passed")
    require_true("random_morphology_takeoff_ramp_passed")
    require_true("random_morphology_takeoff_hover_passed")
    require_true("random_morphology_takeoff_fixed_dock_neutral_hold_passed")
    require_true("random_morphology_takeoff_exact_cross_module_collision_passed")
    require_true("random_morphology_takeoff_finite_state")
    require_true("random_morphology_takeoff_logging_passed")
    require_exact("random_morphology_takeoff_graph_id", morphology_graph.graph_id)
    require_exact(
        "random_morphology_takeoff_morphology_hash",
        morphology_graph.stable_hash(),
    )
    require_exact(
        "random_morphology_takeoff_backend_config_hash",
        takeoff_env.backend.config.stable_hash(),
    )
    require_exact(
        "random_morphology_takeoff_physical_model_hash",
        takeoff_env.physical_model.stable_hash(),
    )
    require_exact(
        "random_morphology_takeoff_collision_geometry_hash",
        collision_geometry_content_hash(
            takeoff_env.physical_model,
            mesh_search_dirs=takeoff_env.config.mesh_search_dirs,
        ),
    )
    require_exact("random_morphology_takeoff_requested_steps", requested_steps)
    require_exact(
        "random_morphology_takeoff_control_contract_version",
        POLICY_COMMAND_CONTRACT_CENTROIDAL,
    )
    for key in (
        "random_morphology_takeoff_qp_infeasible_count",
        "random_morphology_takeoff_controller_clipped_count",
        "random_morphology_takeoff_missing_actuator_count",
        "random_morphology_takeoff_unsupported_actuator_count",
        "random_morphology_takeoff_clipped_target_count",
        "random_morphology_takeoff_application_unresolved_target_count",
        "random_morphology_takeoff_dynamic_exact_contact_violation_step_count",
        "random_morphology_takeoff_dynamic_exact_raw_contact_observation_count",
        "random_morphology_takeoff_dynamic_exact_raw_contact_saturation_step_count",
    ):
        require_zero_count(key)
    for key in (
        "random_morphology_takeoff_max_abs_dock_position_target_rad",
        "random_morphology_takeoff_max_abs_dock_velocity_target_rad_s",
        "random_morphology_takeoff_max_abs_dock_torque_bias_nm",
    ):
        value = report.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            failures.append(f"invalid:{key}")
        elif abs(float(value)) > 1.0e-12:
            failures.append(f"nonzero:{key}")
    final_hold = report.get("order4_free_flight_final_hover_hold_time_s")
    if not isinstance(final_hold, (int, float)) or isinstance(final_hold, bool):
        failures.append("invalid:order4_free_flight_final_hover_hold_time_s")
    elif float(final_hold) + 1.0e-9 < config.mission.final_hover_hold_s:
        failures.append("below:order4_free_flight_final_hover_hold_time_s")
    require_exact(
        "order4_free_flight_final_hover_hold_required_s",
        config.mission.final_hover_hold_s,
    )
    plan_records = report.get("order4_free_flight_plan_records")
    if not isinstance(plan_records, list) or not plan_records:
        failures.append("invalid:order4_free_flight_plan_records")
    else:
        for index, record in enumerate(plan_records):
            try:
                trajectory = ContactWrenchTrajectory.from_dict(record["trajectory"])
            except (KeyError, TypeError, SchemaValidationError):
                failures.append(f"invalid:order4_plan_record:{index}")
                continue
            if len(trajectory.knots) < 2:
                failures.append(f"single_knot:order4_plan_record:{index}")
            if any(knot.contact_assignments for knot in trajectory.knots):
                failures.append(f"contact_assignment:order4_plan_record:{index}")
    runtime_steps = report.get("order4_free_flight_runtime_steps")
    if config.record_runtime_steps:
        if not isinstance(runtime_steps, list) or not runtime_steps:
            failures.append("invalid:order4_free_flight_runtime_steps")
        else:
            for index, payload in enumerate(runtime_steps):
                try:
                    step = Order4TrajectoryRuntimeStep.from_dict(payload)
                except (TypeError, SchemaValidationError):
                    failures.append(f"invalid:order4_runtime_step:{index}")
                    continue
                if step.active_knot.contact_assignments:
                    failures.append(f"contact_assignment:order4_runtime_step:{index}")
                if abs(
                    step.plan_elapsed_s
                    - (step.time_s - step.plan_start_time_s)
                ) > 1.0e-8:
                    failures.append(f"time_origin:order4_runtime_step:{index}")
    transitions = report.get("order4_free_flight_phase_transitions")
    if not isinstance(transitions, list):
        failures.append("invalid:order4_free_flight_phase_transitions")
    else:
        phases = [transition.get("to_phase") for transition in transitions]
        required = [
            "floor_settle",
            "takeoff",
            "hover_acquisition",
            "final_hover",
            "complete",
        ]
        if any(phase not in phases for phase in required):
            failures.append("missing:order4_required_phase_transition")
        waypoint_indices = [
            transition.get("waypoint_index")
            for transition in transitions
            if transition.get("to_phase") == "waypoint"
        ]
        if waypoint_indices != list(range(len(config.mission.waypoints))):
            failures.append("mismatch:order4_waypoint_phase_transitions")
    if config.pi_l_checkpoint_path is None:
        require_exact(
            "order4_free_flight_low_level_source",
            "deterministic_baseline_pi_l",
        )
        require_exact("order4_pi_l_checkpoint_sha256", None)
    else:
        require_exact(
            "order4_free_flight_low_level_source",
            "order3_morphology_conditioned_pi_l",
        )
        require_exact(
            "order4_pi_l_checkpoint_sha256",
            config.expected_pi_l_checkpoint_sha256,
        )
        require_zero_count("order4_pi_l_fallback_count")
    artifacts = report.get("random_morphology_takeoff_artifacts")
    if not isinstance(artifacts, dict):
        failures.append("invalid:random_morphology_takeoff_artifacts")
    else:
        if artifacts.get("order4_learned_pi_h_claim") is not False:
            failures.append("mislabel:order4_learned_pi_h_claim")
        if artifacts.get("order4_contact_planning_claim") is not False:
            failures.append("mislabel:order4_contact_planning_claim")
        if artifacts.get("is_p4_full_completion") is not False:
            failures.append("mislabel:is_p4_full_completion")
    return sorted(set(failures))


def _replace_command_argument(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise SchemaValidationError(f"Order4 base probe command is missing {flag}") from exc
    if index + 1 >= len(command):
        raise SchemaValidationError(f"Order4 base probe command has no value for {flag}")
    command[index + 1] = value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
