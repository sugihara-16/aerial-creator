from __future__ import annotations

"""Fail-closed real-Isaac wrapper for the Order 8 natural-contact smoke."""

from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.controllers.actuator_mapping import build_actuator_mapping
from amsrr.controllers.centroidal_admittance import (
    CentroidalAdmittanceConfig,
    CentroidalExternalWrenchEstimatorConfig,
)
from amsrr.controllers.natural_contact_joint_controller import (
    NaturalContactJointControllerConfig,
    position_drive_peak_effort_lead_rad,
)
from amsrr.controllers.qpid_controller import QPIDControllerConfig
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.deterministic_natural_contact_planner import (
    NaturalContactPlannerConfig,
    ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT,
)
from amsrr.robot_model.gripper_surfaces import (
    select_opposing_gripper_surface_pair,
)
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError, canonical_json
from amsrr.schemas.morphology import ControlGroup, MorphologyGraph
from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_MODEL,
    ORDER8_NATURAL_CONTACT_RESULT_VERSION,
    ORDER8_RAW_CONTACT_TRUTH_ROLE,
    Order8NaturalContactConfig,
    Order8NaturalContactPhase,
    Order8NaturalContactResult,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend
from amsrr.simulation.order8_isaac_runtime import (
    ORDER8_CONTACT_STALL_RATED_TORQUE_FRACTION,
    ORDER8_GRASP_ADDITIONAL_FLOOR_CLEARANCE_M,
    ORDER8_OBJECT_SUPPORT_PATH,
    ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
    ORDER8_SELECTED_GRIPPER_MATERIAL_PATH,
)
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file, stable_hash

ORDER8_NATURAL_CONTACT_ENV_VERSION = "order8_natural_contact_isaac_env_v1"
ORDER8_NATURAL_CONTACT_REPORT_VERSION = "order8_natural_contact_isaac_report_v1"
ORDER8_NATURAL_CONTACT_SCOPE = "deterministic_natural_contact_substrate_only"
ORDER8_NATURAL_CONTACT_PROGRESS_PREFIX = "[order8-natural-contact]"
ORDER8_NATURAL_CONTACT_REQUIRED_PHASES = (
    "reset",
    "approach",
    "contact_acquisition",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
    "complete",
)
ORDER8_DEFAULT_SIMULATION_DT_S = 0.020
ORDER8_DEFAULT_ROLLOUT_BUDGET_S = 150.0
# Convex-decomposition contact on the full articulated three-module model can
# become prohibitively slow at 200 Hz on the reference workstation.  The
# retained 50 Hz period is inside the design's 50--200 Hz pi_L range and has
# passed the same authored-mesh contact path repeatedly.  Keep the wall timeout
# above the complete 150 s simulation budget; phase timeouts inside the planner
# remain the actual deterministic rollout termination mechanism.
ORDER8_DEFAULT_COMMAND_TIMEOUT_S = 2400.0
ORDER8_DEFAULT_GENERATED_USD_DIR = "artifacts/isaac/robots/holon_order8_natural_contact"


@dataclass
class Order8IsaacNaturalContactResult(SchemaBase):
    env_version: str
    graph_id: str
    graph_hash: str
    config_hash: str
    dry_run: bool
    attempted: bool
    isaac_backed: bool
    passed: bool
    report_validation_failures: list[str]
    report: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def validate(self) -> None:
        if self.env_version != ORDER8_NATURAL_CONTACT_ENV_VERSION:
            raise SchemaValidationError(
                "Order8IsaacNaturalContactResult.env_version mismatch"
            )
        if not self.graph_id:
            raise SchemaValidationError(
                "Order8IsaacNaturalContactResult.graph_id must be non-empty"
            )
        for name in ("graph_hash", "config_hash"):
            if not _is_sha256(getattr(self, name)):
                raise SchemaValidationError(
                    f"Order8IsaacNaturalContactResult.{name} must be sha256"
                )
        if self.passed and (
            self.dry_run
            or not self.attempted
            or not self.isaac_backed
            or bool(self.report_validation_failures)
            or self.failure_reason is not None
        ):
            raise SchemaValidationError(
                "Order8 pass requires attempted real-Isaac evidence with no validation failure"
            )


class Order8IsaacNaturalContactEnv:
    def __init__(
        self,
        *,
        config: Order8NaturalContactConfig,
        backend: IsaacLabBackend,
        physical_model: PhysicalModel,
        backend_config_path: str | Path = "configs/env/isaac_lab.yaml",
        simulation_dt_s: float = ORDER8_DEFAULT_SIMULATION_DT_S,
        rollout_budget_s: float = ORDER8_DEFAULT_ROLLOUT_BUDGET_S,
        command_timeout_s: float = ORDER8_DEFAULT_COMMAND_TIMEOUT_S,
        generated_usd_dir: str | Path = ORDER8_DEFAULT_GENERATED_USD_DIR,
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
        seed: int = 0,
        order9_teacher_output: str | Path | None = None,
        order9_teacher_episode_id: str | None = None,
        order9_teacher_task_id: str | None = None,
        order9_teacher_split: str = "train",
        order9_teacher_low_level_stride: int = 1,
        order9_teacher_high_level_stride: int = 5,
        order9_teacher_window_horizon_s: float = 2.0,
        order9_teacher_window_knot_dt_s: float = 0.1,
        force_convert: bool = True,
        command_executor: Callable[[list[str], float], dict[str, Any]] | None = None,
    ) -> None:
        config.validate()
        physical_model.validate()
        if viewer not in {None, "kit"}:
            raise SchemaValidationError("Order8 viewer must be None or 'kit'")
        if viewer is None and (realtime_playback or keep_open_after_rollout_s > 0.0):
            raise SchemaValidationError(
                "Order8 real-time/post-rollout viewing requires viewer='kit'"
            )
        for name, value in (
            ("simulation_dt_s", simulation_dt_s),
            ("rollout_budget_s", rollout_budget_s),
            ("command_timeout_s", command_timeout_s),
        ):
            if not _finite_positive(value):
                raise SchemaValidationError(
                    f"Order8 {name} must be finite and positive"
                )
        if not _finite_non_negative(keep_open_after_rollout_s):
            raise SchemaValidationError(
                "Order8 keep_open_after_rollout_s must be finite and non-negative"
            )
        if not str(generated_usd_dir):
            raise SchemaValidationError("Order8 generated_usd_dir must be non-empty")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise SchemaValidationError("Order8 seed must be a non-negative integer")
        if order9_teacher_split not in {"train", "validation", "held_out"}:
            raise SchemaValidationError("Order9 teacher split is invalid")
        if order9_teacher_low_level_stride < 1 or order9_teacher_high_level_stride < 1:
            raise SchemaValidationError("Order9 teacher strides must be positive")
        if (
            not _finite_positive(order9_teacher_window_horizon_s)
            or not _finite_positive(order9_teacher_window_knot_dt_s)
            or not math.isclose(
                order9_teacher_window_horizon_s
                / order9_teacher_window_knot_dt_s,
                round(
                    order9_teacher_window_horizon_s
                    / order9_teacher_window_knot_dt_s
                ),
                abs_tol=1.0e-9,
            )
        ):
            raise SchemaValidationError(
                "Order9 teacher window horizon must be an integer multiple of knot dt"
            )
        for name, value in (
            ("order9_teacher_episode_id", order9_teacher_episode_id),
            ("order9_teacher_task_id", order9_teacher_task_id),
        ):
            if value is not None and not value:
                raise SchemaValidationError(f"Order8 {name} must be non-empty when set")
        self.config = config
        self.backend = backend
        self.physical_model = physical_model
        self.backend_config_path = str(backend_config_path)
        self.simulation_dt_s = float(simulation_dt_s)
        self.rollout_budget_s = float(rollout_budget_s)
        self.command_timeout_s = float(command_timeout_s)
        self.generated_usd_dir = str(generated_usd_dir)
        self.viewer = viewer
        self.realtime_playback = bool(realtime_playback)
        self.keep_open_after_rollout_s = float(keep_open_after_rollout_s)
        self.seed = int(seed)
        self.order9_teacher_output = (
            None if order9_teacher_output is None else str(order9_teacher_output)
        )
        self.order9_teacher_episode_id = order9_teacher_episode_id
        self.order9_teacher_task_id = order9_teacher_task_id
        self.order9_teacher_split = order9_teacher_split
        self.order9_teacher_low_level_stride = int(order9_teacher_low_level_stride)
        self.order9_teacher_high_level_stride = int(order9_teacher_high_level_stride)
        self.order9_teacher_window_horizon_s = float(
            order9_teacher_window_horizon_s
        )
        self.order9_teacher_window_knot_dt_s = float(
            order9_teacher_window_knot_dt_s
        )
        self.force_convert = bool(force_convert)
        self.command_executor = command_executor or _run_json_command

    @property
    def requested_steps(self) -> int:
        return max(1, int(math.ceil(self.rollout_budget_s / self.simulation_dt_s)))

    @property
    def collision_geometry_hash(self) -> str:
        return collision_geometry_content_hash(
            self.physical_model,
            mesh_search_dirs=("module_urdf", "module_urdf/mesh"),
        )

    @property
    def source_urdf_hash(self) -> str:
        return hash_file(self.physical_model.urdf_path)

    def representative_morphology(self) -> MorphologyGraph:
        return build_representative_order8_morphology(self.physical_model)

    def build_probe_command(
        self, morphology_graph: MorphologyGraph | None = None
    ) -> list[str]:
        graph = morphology_graph or self.representative_morphology()
        validate_representative_order8_morphology(
            graph,
            physical_model=self.physical_model,
        )
        command = self.backend.holon_spawn_probe_command(
            config_path=self.backend_config_path,
            convert_if_missing=not self.force_convert,
            force_convert=self.force_convert,
            generated_usd_dir=self.generated_usd_dir,
            steps=self.requested_steps,
            viewer=self.viewer,
            realtime_playback=self.realtime_playback,
            keep_open_after_smoke_s=self.keep_open_after_rollout_s,
        )
        command.extend(
            [
                "--order8-natural-contact",
                "--order8-morphology-graph-json",
                canonical_json(graph),
                "--order8-config-json",
                self.config.to_json(),
                "--order8-seed",
                str(self.seed),
                "--dt",
                str(self.simulation_dt_s),
                "--control-contract-version",
                "centroidal_local_joint_v2",
            ]
        )
        if self.order9_teacher_output is not None:
            command.extend(
                [
                    "--order9-teacher-output",
                    self.order9_teacher_output,
                    "--order9-teacher-split",
                    self.order9_teacher_split,
                    "--order9-teacher-low-level-stride",
                    str(self.order9_teacher_low_level_stride),
                    "--order9-teacher-high-level-stride",
                    str(self.order9_teacher_high_level_stride),
                    "--order9-teacher-window-horizon-s",
                    str(self.order9_teacher_window_horizon_s),
                    "--order9-teacher-window-knot-dt-s",
                    str(self.order9_teacher_window_knot_dt_s),
                ]
            )
            if self.order9_teacher_episode_id is not None:
                command.extend(
                    ["--order9-teacher-episode-id", self.order9_teacher_episode_id]
                )
            if self.order9_teacher_task_id is not None:
                command.extend(
                    ["--order9-teacher-task-id", self.order9_teacher_task_id]
                )
        return command

    def run(
        self,
        morphology_graph: MorphologyGraph | None = None,
        *,
        dry_run: bool = True,
        check_availability: bool = True,
    ) -> Order8IsaacNaturalContactResult:
        graph = morphology_graph or self.representative_morphology()
        validate_representative_order8_morphology(
            graph,
            physical_model=self.physical_model,
        )
        common = {
            "env_version": ORDER8_NATURAL_CONTACT_ENV_VERSION,
            "graph_id": graph.graph_id,
            "graph_hash": graph.stable_hash(),
            "config_hash": self.config.stable_hash(),
        }
        if dry_run:
            return Order8IsaacNaturalContactResult(
                **common,
                dry_run=True,
                attempted=False,
                isaac_backed=False,
                passed=False,
                report_validation_failures=[],
                report={"probe_command": self.build_probe_command(graph)},
            )
        if check_availability:
            availability = self.backend.availability()
            blocking_reasons = [
                reason
                for reason in availability.missing_reasons
                if reason != "isaac_python_modules_unavailable_in_current_interpreter"
            ]
            if blocking_reasons:
                return Order8IsaacNaturalContactResult(
                    **common,
                    dry_run=False,
                    attempted=False,
                    isaac_backed=False,
                    passed=False,
                    report_validation_failures=blocking_reasons,
                    failure_reason=",".join(blocking_reasons),
                )
        try:
            report = self.command_executor(
                self.build_probe_command(graph), self.command_timeout_s
            )
        except Exception as exc:  # pragma: no cover - subprocess-specific.
            return Order8IsaacNaturalContactResult(
                **common,
                dry_run=False,
                attempted=True,
                isaac_backed=True,
                passed=False,
                report_validation_failures=["probe_execution_failed"],
                failure_reason=str(exc),
            )
        failures = order8_natural_contact_report_failures(
            report,
            morphology_graph=graph,
            config=self.config,
            physical_model=self.physical_model,
            expected_backend_config_hash=self.backend.config.stable_hash(),
            expected_collision_geometry_hash=self.collision_geometry_hash,
            expected_source_urdf_hash=self.source_urdf_hash,
            requested_steps=self.requested_steps,
            expected_seed=self.seed,
            expected_simulation_dt_s=self.simulation_dt_s,
            expected_force_usd_conversion=self.force_convert,
        )
        return Order8IsaacNaturalContactResult(
            **common,
            dry_run=False,
            attempted=True,
            isaac_backed=report.get("isaac_backed") is True,
            passed=not failures,
            report_validation_failures=failures,
            report=report,
            failure_reason=(
                None
                if not failures
                else "order8_report_validation_failed:" + ",".join(failures)
            ),
        )


def build_representative_order8_morphology(
    physical_model: PhysicalModel,
) -> MorphologyGraph:
    """Build the current exact symmetric two-anchor, three-module design."""

    task = default_grasp_carry_task_spec()
    irg = IRGBuilder().build(task)
    design = build_grasp_carry_variant_design_output(
        task,
        irg,
        physical_model,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    graph = design.target_morphology
    validate_representative_order8_morphology(
        graph,
        physical_model=physical_model,
        expected_graph=graph,
    )
    return graph


def validate_representative_order8_morphology(
    morphology_graph: MorphologyGraph,
    *,
    physical_model: PhysicalModel,
    expected_graph: MorphologyGraph | None = None,
) -> None:
    morphology_graph.validate()
    if expected_graph is None:
        task = default_grasp_carry_task_spec()
        irg = IRGBuilder().build(task)
        expected_graph = build_grasp_carry_variant_design_output(
            task,
            irg,
            physical_model,
            variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
        ).target_morphology
    if morphology_graph.stable_hash() != expected_graph.stable_hash():
        raise SchemaValidationError(
            "Order8 representative morphology must exactly match the current symmetric two-anchor design"
        )
    if (
        len(morphology_graph.modules) != 3
        or len(morphology_graph.robot_anchors) != 2
        or len(morphology_graph.dock_edges) != 2
        or morphology_graph.base_module_id != 0
    ):
        raise SchemaValidationError(
            "Order8 representative morphology must have 3 modules, 2 anchors, and 2 tree edges"
        )
    anchor_modules = {anchor.module_id for anchor in morphology_graph.robot_anchors}
    if anchor_modules != {1, 2}:
        raise SchemaValidationError(
            "Order8 representative anchors must belong to the two symmetric arm modules"
        )
    select_opposing_gripper_surface_pair(morphology_graph, physical_model)


def order8_natural_contact_report_failures(
    report: dict[str, Any],
    *,
    morphology_graph: MorphologyGraph,
    config: Order8NaturalContactConfig,
    physical_model: PhysicalModel,
    expected_backend_config_hash: str | None = None,
    expected_collision_geometry_hash: str | None = None,
    expected_source_urdf_hash: str | None = None,
    requested_steps: int | None = None,
    expected_seed: int | None = None,
    expected_simulation_dt_s: float | None = None,
    expected_force_usd_conversion: bool = True,
) -> list[str]:
    failures: list[str] = []

    def exact(key: str, expected: Any) -> None:
        if key not in report:
            failures.append(f"missing:{key}")
        elif type(report[key]) is not type(expected) or report[key] != expected:
            failures.append(f"mismatch:{key}")

    def true(key: str) -> None:
        exact(key, True)

    def false(key: str) -> None:
        exact(key, False)

    def zero(key: str) -> None:
        exact(key, 0)

    true("spawn_passed")
    true("isaac_backed")
    true("command_applied")
    true("command_probe_passed")
    exact("command_returncode", 0)
    true("order8_natural_contact_enabled")
    false("order8_natural_contact_diagnostic_only")
    false("order8_natural_contact_diagnostic_force_fixture")
    false("order8_natural_contact_diagnostic_precontact_fixture")
    exact("order8_natural_contact_diagnostic_precontact_base_pose", None)
    false("order8_natural_contact_diagnostic_world_fixed_base")
    exact("order8_natural_contact_diagnostic_world_fixed_body_path", None)
    exact("order8_natural_contact_diagnostic_world_fixed_pose", None)
    false("order8_natural_contact_diagnostic_world_fixed_object")
    exact("order8_natural_contact_diagnostic_world_fixed_object_pose", None)
    exact("order8_natural_contact_diagnostic_object_width_padding_m", 0.0)
    exact(
        "order8_natural_contact_runtime_object_size_m",
        [float(value) for value in config.object_size_m],
    )
    exact(
        "order8_natural_contact_object_support_method",
        "free_object_on_fixed_raised_platform_without_pose_constraint_v1",
    )
    exact("order8_natural_contact_object_support_path", ORDER8_OBJECT_SUPPORT_PATH)
    exact(
        "order8_natural_contact_object_support_height_m",
        float(config.object_support_height_m),
    )
    true("order8_natural_contact_object_support_covers_planned_place_pose")
    support_size = report.get("order8_natural_contact_object_support_size_m")
    expected_support_size = (
        float(config.object_size_m[0])
        + float(config.required_transport_distance_m)
        + 0.05,
        max(0.05, float(config.object_size_m[1]) - 0.04),
        float(config.object_support_height_m),
    )
    if not (
        isinstance(support_size, list)
        and len(support_size) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in support_size
        )
        and all(
            math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for actual, expected in zip(
                support_size,
                expected_support_size,
                strict=True,
            )
        )
    ):
        failures.append("mismatch:order8_natural_contact_object_support_size_m")
    support_pose = report.get("order8_natural_contact_object_support_pose_world")
    if not (
        isinstance(support_pose, list)
        and len(support_pose) == 7
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in support_pose
        )
        and math.isclose(
            float(support_pose[2]),
            0.5 * float(config.object_support_height_m),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        failures.append("invalid:order8_natural_contact_object_support_pose_world")
    exact(
        "order8_natural_contact_robot_environment_contact_method",
        "all_robot_rigid_bodies_against_floor_and_object_support_v1",
    )
    unsafe_robot_environment_steps = report.get(
        "order8_natural_contact_robot_environment_unsafe_contact_step_count"
    )
    if unsafe_robot_environment_steps != 0:
        failures.append(
            "mismatch:order8_natural_contact_robot_environment_unsafe_contact_step_count"
        )
    exact(
        "order8_natural_contact_robot_environment_first_unsafe_contact_time_s",
        None,
    )
    true("order8_natural_contact_acceptance_eligible")
    exact("order8_natural_contact_diagnostic_mode", "disabled")
    exact("order8_natural_contact_diagnostic_stop_force_scale", None)
    false("order8_natural_contact_diagnostic_stop_reached")
    true("order8_natural_contact_passed")
    exact(
        "order8_natural_contact_report_version",
        ORDER8_NATURAL_CONTACT_REPORT_VERSION,
    )
    exact("order8_natural_contact_contact_model", ORDER8_NATURAL_CONTACT_MODEL)
    exact("order8_natural_contact_scope", ORDER8_NATURAL_CONTACT_SCOPE)
    false("order8_natural_contact_p4_full_completion_claim")
    false("order8_natural_contact_order9_full_taskspec_claim")
    false("order8_natural_contact_learned_policy_success_claim")
    exact("order8_natural_contact_config", config.to_dict())
    exact("order8_natural_contact_config_hash", config.stable_hash())
    exact("order8_natural_contact_graph_id", morphology_graph.graph_id)
    exact("order8_natural_contact_graph_hash", morphology_graph.stable_hash())
    exact("order8_natural_contact_module_count", 3)
    exact("order8_natural_contact_robot_anchor_count", 2)
    if expected_seed is None:
        seed = report.get("order8_natural_contact_seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            failures.append("invalid:order8_natural_contact_seed")
    else:
        exact("order8_natural_contact_seed", expected_seed)
    seed_application = report.get("order8_natural_contact_seed_applied")
    reported_seed = report.get("order8_natural_contact_seed")
    if not (
        isinstance(seed_application, dict)
        and seed_application.get("seed") == reported_seed
        and seed_application.get("python_random") is True
        and seed_application.get("torch") is True
        and seed_application.get("numpy") is True
        and type(seed_application.get("torch_cuda")) is bool
    ):
        failures.append("invalid:order8_natural_contact_seed_applied")
    if expected_backend_config_hash is None:
        _require_sha256(report, "order8_natural_contact_backend_config_hash", failures)
    else:
        exact(
            "order8_natural_contact_backend_config_hash",
            expected_backend_config_hash,
        )
    exact(
        "order8_natural_contact_physical_model_hash",
        physical_model.stable_hash(),
    )
    if expected_collision_geometry_hash is None:
        _require_sha256(
            report,
            "order8_natural_contact_collision_geometry_content_hash",
            failures,
        )
    else:
        exact(
            "order8_natural_contact_collision_geometry_content_hash",
            expected_collision_geometry_hash,
        )
    if expected_source_urdf_hash is None:
        _require_sha256(report, "order8_natural_contact_source_urdf_sha256", failures)
    else:
        exact(
            "order8_natural_contact_source_urdf_sha256",
            expected_source_urdf_hash,
        )
    for key in (
        "order8_natural_contact_generated_usd_sha256",
        "order8_natural_contact_generated_usd_bundle_hash",
    ):
        _require_sha256(report, key, failures)
    exact(
        "order8_natural_contact_force_usd_conversion",
        bool(expected_force_usd_conversion),
    )
    exact(
        "order8_natural_contact_dock_collision_type",
        "Convex Decomposition",
    )
    exact(
        "order8_natural_contact_dock_collision_approximation_token",
        "convexDecomposition",
    )
    true("order8_natural_contact_dock_collision_approximation_verified")
    if not _positive_int(
        report.get("order8_natural_contact_dock_collision_composed_prim_count")
    ):
        failures.append(
            "invalid:order8_natural_contact_dock_collision_composed_prim_count"
        )
    if requested_steps is None:
        if not _positive_int(report.get("order8_natural_contact_requested_steps")):
            failures.append("invalid:order8_natural_contact_requested_steps")
    else:
        exact("order8_natural_contact_requested_steps", requested_steps)
    simulation_dt = report.get("order8_natural_contact_simulation_dt_s")
    actuator_specs = physical_model.metadata.get("joint_actuator_specs", {})
    dock_spec = (
        actuator_specs.get("dock", {})
        if isinstance(actuator_specs, dict)
        else {}
    )
    drive_spec = (
        dock_spec.get("simulation_drive", {})
        if isinstance(dock_spec, dict)
        else {}
    )
    if not _finite_positive(simulation_dt):
        failures.append("invalid:order8_natural_contact_simulation_dt_s")
    elif expected_simulation_dt_s is not None and not math.isclose(
        float(simulation_dt),
        float(expected_simulation_dt_s),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        failures.append("mismatch:order8_natural_contact_simulation_dt_s")
    if _finite_positive(simulation_dt):
        expected_qpid = QPIDControllerConfig(
            allocation_mode="rigid_body_qp",
            control_dt_s=float(simulation_dt),
        )
        expected_contact_centering_qpid = replace(
            expected_qpid,
            xy_p_gain=float(config.contact_centering_xy_p_gain),
            xy_d_gain=float(config.contact_centering_xy_d_gain),
            roll_pitch_p_gain=float(config.contact_centering_roll_pitch_p_gain),
            roll_pitch_d_gain=float(config.contact_centering_roll_pitch_d_gain),
        )
        expected_external_wrench_estimator = (
            CentroidalExternalWrenchEstimatorConfig(
                gravity_mps2=float(expected_qpid.gravity_mps2),
                wrench_filter_time_constant_s=float(
                    config.contact_external_wrench_filter_time_constant_s
                ),
                bias_filter_time_constant_s=float(
                    config.contact_external_wrench_bias_time_constant_s
                ),
            )
        )
        expected_contact_admittance = CentroidalAdmittanceConfig(
            force_deadband_n=float(config.contact_admittance_force_deadband_n),
            torque_deadband_nm=float(config.contact_admittance_torque_deadband_nm),
            linear_admittance_mps_per_n=float(
                config.contact_admittance_linear_gain_mps_per_n
            ),
            angular_admittance_radps_per_nm=float(
                config.contact_admittance_angular_gain_radps_per_nm
            ),
            maximum_linear_speed_mps=float(
                config.contact_admittance_max_linear_speed_mps
            ),
            maximum_angular_speed_radps=float(
                config.contact_admittance_max_angular_speed_radps
            ),
            maximum_translation_offset_m=float(
                config.contact_admittance_max_translation_offset_m
            ),
        )
        expected_joint_controller = NaturalContactJointControllerConfig(
            control_dt_s=float(simulation_dt),
            max_position_command_lead_rad=position_drive_peak_effort_lead_rad(
                stiffness_nm_per_rad=float(drive_spec.get("stiffness", 200.0)),
                peak_effort_nm=float(dock_spec.get("peak_torque_nm", 4.1)),
            ),
            reachability_absolute_tolerance=float(
                config.simultaneous_reachability_absolute_tolerance
            ),
        )
        expected_planner = NaturalContactPlannerConfig(
            contact_acquisition_timeout_s=float(config.contact_acquisition_timeout_s),
            normal_force_target_per_contact_n=float(
                config.normal_force_target_per_contact_n
            ),
        )
        exact("order8_natural_contact_qpid_config", asdict(expected_qpid))
        exact(
            "order8_natural_contact_qpid_config_hash",
            stable_hash(expected_qpid),
        )
        exact(
            "order8_natural_contact_contact_centering_qpid_config",
            asdict(expected_contact_centering_qpid),
        )
        exact(
            "order8_natural_contact_contact_centering_qpid_config_hash",
            stable_hash(expected_contact_centering_qpid),
        )
        exact(
            "order8_natural_contact_external_wrench_estimator_config",
            asdict(expected_external_wrench_estimator),
        )
        exact(
            "order8_natural_contact_external_wrench_estimator_config_hash",
            stable_hash(expected_external_wrench_estimator),
        )
        exact(
            "order8_natural_contact_contact_admittance_config",
            asdict(expected_contact_admittance),
        )
        exact(
            "order8_natural_contact_contact_admittance_config_hash",
            stable_hash(expected_contact_admittance),
        )
        exact(
            "order8_natural_contact_contact_yield_method",
            "first_damping_compensated_terminal_joint_surface_load_enables_"
            "contact_axis_centroidal_admittance_with_full_height_attitude_"
            "pose_tracking_v9",
        )
        exact(
            "order8_natural_contact_contact_yield_trigger_method",
            "any_selected_terminal_joint_damping_compensated_load_plus_mesh_"
            "proximity_after_closure_armed_latched_until_verified_grasp_v7",
        )
        exact("order8_natural_contact_contact_yield_raw_contact_input", False)
        exact(
            "order8_natural_contact_contact_yield_per_contact_wrench_input",
            False,
        )
        exact(
            "order8_natural_contact_contact_yield_external_wrench_scope",
            "aggregate_centroidal_only_v1",
        )
        if not _finite_non_negative(
            report.get("order8_natural_contact_contact_yield_triggered_time_s")
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_triggered_time_s"
            )
        yield_anchor_ids = report.get(
            "order8_natural_contact_contact_yield_trigger_anchor_ids"
        )
        expected_anchor_ids = {
            int(anchor.anchor_id) for anchor in morphology_graph.robot_anchors
        }
        if not (
            isinstance(yield_anchor_ids, list)
            and bool(yield_anchor_ids)
            and len(yield_anchor_ids) == len(set(yield_anchor_ids))
            and all(
                isinstance(anchor_id, int)
                and not isinstance(anchor_id, bool)
                and anchor_id in expected_anchor_ids
                for anchor_id in yield_anchor_ids
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_trigger_anchor_ids"
            )
        for key in (
            "order8_natural_contact_contact_yield_active_step_count",
            "order8_natural_contact_contact_yield_full_step_count",
            "order8_natural_contact_contact_yield_restore_step_count",
            "order8_natural_contact_contact_yield_estimator_valid_step_count",
        ):
            if not _positive_int(report.get(key)):
                failures.append(f"invalid:{key}")
        invalid_estimator_steps = report.get(
            "order8_natural_contact_contact_yield_estimator_invalid_step_count"
        )
        if not (
            isinstance(invalid_estimator_steps, int)
            and not isinstance(invalid_estimator_steps, bool)
            and invalid_estimator_steps >= 0
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_estimator_invalid_step_count"
            )
        final_yield_blend = report.get(
            "order8_natural_contact_contact_yield_final_blend"
        )
        if (
            not _finite_non_negative(final_yield_blend)
            or float(final_yield_blend) > 1.0e-9
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_final_blend"
            )
        minimum_pi_scale = report.get(
            "order8_natural_contact_contact_yield_minimum_pi_scale"
        )
        if (
            not _finite_non_negative(minimum_pi_scale)
            or not math.isclose(
                float(minimum_pi_scale),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_minimum_pi_scale"
            )
        for key in (
            "order8_natural_contact_contact_yield_maximum_external_force_n",
            "order8_natural_contact_contact_yield_maximum_external_torque_nm",
        ):
            if not _finite_non_negative(report.get(key)):
                failures.append(f"invalid:{key}")
        maximum_yield_offset = report.get(
            "order8_natural_contact_contact_yield_maximum_translation_offset_m"
        )
        if (
            not _finite_non_negative(maximum_yield_offset)
            or float(maximum_yield_offset)
            > config.contact_admittance_max_translation_offset_m + 1.0e-9
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_maximum_translation_offset_m"
            )
        for key, length in (
            ("order8_natural_contact_contact_yield_last_admittance_twist", 6),
            (
                "order8_natural_contact_contact_yield_last_translation_offset_world",
                3,
            ),
        ):
            values = report.get(key)
            if not (
                isinstance(values, list)
                and len(values) == length
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                )
            ):
                failures.append(f"invalid:{key}")
        true("order8_natural_contact_contact_yield_grasp_pose_rebased")
        if not _finite_non_negative(
            report.get(
                "order8_natural_contact_contact_yield_grasp_pose_rebase_time_s"
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_grasp_pose_rebase_time_s"
            )
        grasp_centroidal_pose = report.get(
            "order8_natural_contact_contact_yield_grasp_centroidal_pose"
        )
        if not (
            isinstance(grasp_centroidal_pose, list)
            and len(grasp_centroidal_pose) == 7
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in grasp_centroidal_pose
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_grasp_centroidal_pose"
            )
        exact(
            "order8_natural_contact_contact_yield_grasp_rebase_method",
            "nonprivileged_load_limited_position_preload_measured_full_6d_"
            "centroidal_with_zero_offset_torque_then_pi_restore_v6",
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_method",
            "disabled_nominal_dock_implicit_impedance_preserved_v7",
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_trigger_method",
            "disabled_v7",
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_raw_contact_input",
            False,
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_scope",
            "none_v2",
        )
        true("order8_natural_contact_post_qclose_geometric_preload_complete")
        exact(
            "order8_natural_contact_post_qclose_geometric_preload_method",
            "not_applicable_replaced_by_joint_space_load_limited_preload_v5",
        )
        exact(
            "order8_natural_contact_contact_force_position_preload_method",
            "fixed_closure_ratio_previous_target_integration_per_anchor_"
            "damping_compensated_load_dwell_and_freeze_v3",
        )
        true("order8_natural_contact_contact_position_preload_complete")
        exact(
            "order8_natural_contact_contact_position_preload_completion_source",
            "per_anchor_damping_compensated_moving_chain_load_dwell",
        )
        exact(
            "order8_natural_contact_contact_position_preload_joint_speed_radps",
            config.contact_position_preload_joint_speed_radps,
        )
        exact(
            "order8_natural_contact_contact_position_preload_load_threshold_nm",
            config.contact_position_preload_load_threshold_nm,
        )
        expected_preload_anchor_ids = {
            str(anchor.anchor_id) for anchor in morphology_graph.robot_anchors
        }
        expected_preload_joint_ids = set(
            ordered_global_dock_joint_ids(morphology_graph, physical_model)
        )
        preload_joint_ids_by_anchor = report.get(
            "order8_natural_contact_contact_position_preload_joint_ids_by_anchor"
        )
        if not (
            isinstance(preload_joint_ids_by_anchor, dict)
            and set(preload_joint_ids_by_anchor) == expected_preload_anchor_ids
            and all(
                isinstance(joint_ids, list)
                and bool(joint_ids)
                and len(joint_ids) == len(set(joint_ids))
                and set(joint_ids).issubset(expected_preload_joint_ids)
                for joint_ids in preload_joint_ids_by_anchor.values()
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_position_preload_joint_ids_by_anchor"
            )
        for key in (
            "order8_natural_contact_contact_position_preload_load_nm_by_anchor",
            "order8_natural_contact_contact_position_preload_max_load_nm_by_anchor",
            "order8_natural_contact_contact_position_preload_load_dwell_s_by_anchor",
        ):
            values = report.get(key)
            if not (
                isinstance(values, dict)
                and set(values) == expected_preload_anchor_ids
                and all(_finite_non_negative(value) for value in values.values())
            ):
                failures.append(f"invalid:{key}")
        max_loads = report.get(
            "order8_natural_contact_contact_position_preload_max_load_nm_by_anchor"
        )
        if isinstance(max_loads, dict) and any(
            float(value) + 1.0e-9
            < config.contact_position_preload_load_threshold_nm
            for value in max_loads.values()
            if _finite_non_negative(value)
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_position_preload_max_load_nm_by_anchor"
            )
        exact(
            "order8_natural_contact_contact_position_preload_frozen_anchor_ids",
            sorted(int(value) for value in expected_preload_anchor_ids),
        )
        for key in (
            "order8_natural_contact_contact_position_preload_velocity_targets_radps",
            "order8_natural_contact_contact_position_preload_position_targets_rad",
        ):
            values = report.get(key)
            if not (
                isinstance(values, dict)
                and set(values) == expected_preload_joint_ids
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values.values()
                )
            ):
                failures.append(f"invalid:{key}")
        preload_velocities = report.get(
            "order8_natural_contact_contact_position_preload_velocity_targets_radps"
        )
        if isinstance(preload_velocities, dict) and any(
            abs(float(value)) > 1.0e-12
            for value in preload_velocities.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_position_preload_velocity_targets_radps"
            )
        preload_active_steps = report.get(
            "order8_natural_contact_contact_position_preload_active_step_count"
        )
        if (
            not isinstance(preload_active_steps, int)
            or isinstance(preload_active_steps, bool)
            or preload_active_steps <= 0
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_position_preload_active_step_count"
            )
        if (
            report.get(
                "order8_natural_contact_contact_yield_joint_drive_triggered_time_s"
            )
            is not None
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_joint_drive_triggered_time_s"
            )
        final_joint_drive_blend = report.get(
            "order8_natural_contact_contact_yield_joint_drive_final_blend"
        )
        if (
            not _finite_non_negative(final_joint_drive_blend)
            or float(final_joint_drive_blend) > 1.0e-9
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_yield_joint_drive_final_blend"
            )
        expected_nominal_stiffness = float(drive_spec.get("stiffness", 200.0))
        expected_nominal_damping = float(drive_spec.get("damping", 1.0))
        exact(
            "order8_natural_contact_contact_yield_joint_drive_nominal_stiffness_nm_per_rad",
            expected_nominal_stiffness,
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_nominal_damping_nms_per_rad",
            expected_nominal_damping,
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_stiffness_scale",
            config.contact_yield_joint_drive_stiffness_scale,
        )
        exact(
            "order8_natural_contact_contact_yield_joint_drive_target_damping_nms_per_rad",
            config.contact_yield_joint_drive_damping_nms_per_rad,
        )
        for key in (
            "order8_natural_contact_contact_yield_joint_drive_active_step_count",
            "order8_natural_contact_contact_yield_joint_drive_write_count",
            "order8_natural_contact_contact_yield_joint_drive_restore_write_count",
        ):
            if report.get(key) != 0:
                failures.append(f"invalid:{key}")
        for key, expected_value in (
            (
                "order8_natural_contact_contact_yield_joint_drive_minimum_stiffness_nm_per_rad",
                expected_nominal_stiffness,
            ),
            (
                "order8_natural_contact_contact_yield_joint_drive_maximum_damping_nms_per_rad",
                expected_nominal_damping,
            ),
            (
                "order8_natural_contact_contact_yield_joint_drive_final_stiffness_nm_per_rad",
                expected_nominal_stiffness,
            ),
            (
                "order8_natural_contact_contact_yield_joint_drive_final_damping_nms_per_rad",
                expected_nominal_damping,
            ),
        ):
            value = report.get(key)
            if (
                not _finite_positive(value)
                or not math.isclose(
                    float(value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                failures.append(f"mismatch:{key}")
        exact(
            "order8_natural_contact_contact_axial_qpid_gain_schedule",
            "mesh_open_axial_insert_uses_centering_horizontal_gain_bank_v1",
        )
        if not _positive_int(
            report.get("order8_natural_contact_contact_axial_gain_scheduled_step_count")
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_axial_gain_scheduled_step_count"
            )
        exact(
            "order8_natural_contact_joint_controller_config",
            asdict(expected_joint_controller),
        )
        exact(
            "order8_natural_contact_joint_controller_config_hash",
            stable_hash(expected_joint_controller),
        )
        exact(
            "order8_natural_contact_contact_joint_velocity_limit_command_rad_s",
            config.contact_joint_velocity_limit_radps,
        )
        exact(
            "order8_natural_contact_contact_joint_velocity_limit_basis",
            "fixed_whole_structure_previous_target_integrated_velocity_and_"
            "simulator_consistent_below_ak40_10_configured_speed_limit_v2",
        )
        exact(
            "order8_natural_contact_joint_position_reference_mode",
            "one_shot_whole_structure_ik_direction_previous_target_integrated_"
            "fixed_velocity_ratio_with_diagnostic_absolute_pitch_hold_until_"
            "load_qclose_then_slow_load_limited_previous_target_preload_and_"
            "measured_qopen_direct_return_v12",
        )
        maximum_joint_position_lead = report.get(
            "order8_natural_contact_max_joint_position_command_lead_rad"
        )
        # The ordinary differential-IK output is bounded by the joint-
        # controller lead in ``expected_joint_controller``.  The approved
        # q_close/preload path is instead an absolute previous-target position
        # servo (target[k+1] = target[k] + velocity*dt) and intentionally
        # accumulates load-producing lead while remaining inside the URDF hard
        # limits.  Its physical authorities are the independently audited
        # applied torque, measured speed, and current-equivalent envelopes.
        if not _finite_non_negative(maximum_joint_position_lead):
            failures.append(
                "invalid:order8_natural_contact_max_joint_position_command_lead_rad"
            )
        maximum_joint_velocity_command = report.get(
            "order8_natural_contact_max_joint_velocity_command_radps"
        )
        if (
            not _finite_non_negative(maximum_joint_velocity_command)
            or float(maximum_joint_velocity_command)
            > float(config.contact_joint_velocity_limit_radps) + 1.0e-9
        ):
            failures.append(
                "invalid:order8_natural_contact_max_joint_velocity_command_radps"
            )
        exact(
            "order8_natural_contact_planner_config",
            asdict(expected_planner),
        )
        exact(
            "order8_natural_contact_planner_config_hash",
            stable_hash(expected_planner),
        )
        exact(
            "order8_natural_contact_base_target_speed_limit_mps",
            config.base_translation_speed_limit_mps,
        )
        exact(
            "order8_natural_contact_contact_base_target_speed_limit_mps",
            config.contact_base_translation_speed_limit_mps,
        )
        exact(
            "order8_natural_contact_contact_axial_min_mesh_overlap_m",
            config.contact_axial_min_mesh_overlap_m,
        )
        exact(
            "order8_natural_contact_contact_axial_overlap_method",
            "selected_urdf_mesh_world_aabb_approach_axis_projection_v1",
        )
        axial_overlap_at_latch = report.get(
            "order8_natural_contact_contact_axial_overlap_at_latch_m"
        )
        if (
            not _finite_non_negative(axial_overlap_at_latch)
            or float(axial_overlap_at_latch)
            < config.contact_axial_min_mesh_overlap_m - 1.0e-9
        ):
            failures.append(
                "below:order8_natural_contact_contact_axial_overlap_at_latch_m"
            )
        exact(
            "order8_natural_contact_contact_axial_hold_method",
            "measured_free_object_relative_floor_clear_contact_region_base_pose_"
            "with_rate_limited_retarget_v4",
        )
        exact(
            "order8_natural_contact_grasp_base_pose_method",
            "normal_aligned_floor_clear_tangential_contact_region_v1",
        )
        floor_base_pose = report.get("order8_natural_contact_floor_base_pose")
        unconstrained_grasp_base_pose = report.get(
            "order8_natural_contact_unconstrained_grasp_base_pose"
        )
        grasp_base_pose = report.get("order8_natural_contact_grasp_base_pose")
        planned_poses = (
            ("floor_base_pose", floor_base_pose),
            ("unconstrained_grasp_base_pose", unconstrained_grasp_base_pose),
            ("grasp_base_pose", grasp_base_pose),
        )
        for label, pose in planned_poses:
            if not (
                isinstance(pose, list)
                and len(pose) == 7
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in pose
                )
            ):
                failures.append(f"invalid:order8_natural_contact_{label}")
        vertical_correction = report.get(
            "order8_natural_contact_grasp_base_vertical_correction_m"
        )
        if not _finite_non_negative(vertical_correction):
            failures.append(
                "invalid:order8_natural_contact_grasp_base_vertical_correction_m"
            )
        exact(
            "order8_natural_contact_grasp_additional_floor_clearance_m",
            ORDER8_GRASP_ADDITIONAL_FLOOR_CLEARANCE_M,
        )
        if all(
            isinstance(pose, list)
            and len(pose) == 7
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in pose
            )
            for pose in (
                floor_base_pose,
                unconstrained_grasp_base_pose,
                grasp_base_pose,
            )
        ):
            assert isinstance(floor_base_pose, list)
            assert isinstance(unconstrained_grasp_base_pose, list)
            assert isinstance(grasp_base_pose, list)
            expected_z = max(
                float(floor_base_pose[2])
                + ORDER8_GRASP_ADDITIONAL_FLOOR_CLEARANCE_M,
                float(unconstrained_grasp_base_pose[2]),
            )
            if not math.isclose(
                float(grasp_base_pose[2]), expected_z, abs_tol=1.0e-12
            ):
                failures.append(
                    "mismatch:order8_natural_contact_grasp_base_pose_floor_clear_z"
                )
            if any(
                not math.isclose(
                    float(grasp_base_pose[index]),
                    float(unconstrained_grasp_base_pose[index]),
                    abs_tol=1.0e-12,
                )
                for index in (0, 1, 3, 4, 5, 6)
            ):
                failures.append(
                    "mismatch:order8_natural_contact_grasp_base_pose_nonvertical_components"
                )
            if _finite_non_negative(vertical_correction) and not math.isclose(
                float(vertical_correction),
                expected_z - float(unconstrained_grasp_base_pose[2]),
                abs_tol=1.0e-12,
            ):
                failures.append(
                    "mismatch:order8_natural_contact_grasp_base_vertical_correction_m"
                )
        expected_plan_anchor_ids = {
            str(anchor.anchor_id) for anchor in morphology_graph.robot_anchors
        }
        normal_corrections = report.get(
            "order8_natural_contact_grasp_base_normal_correction_m_by_anchor"
        )
        if not (
            isinstance(normal_corrections, dict)
            and set(normal_corrections) == expected_plan_anchor_ids
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and abs(float(value)) <= 1.0e-9
                for value in normal_corrections.values()
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_grasp_base_normal_correction_m_by_anchor"
            )
        tangential_corrections = report.get(
            "order8_natural_contact_grasp_base_tangential_correction_m_by_anchor"
        )
        if not (
            isinstance(tangential_corrections, dict)
            and set(tangential_corrections) == expected_plan_anchor_ids
            and all(
                isinstance(values, list)
                and len(values) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and abs(float(value))
                    <= config.contact_tangential_tolerance_m + 1.0e-12
                    for value in values
                )
                for values in tangential_corrections.values()
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_grasp_base_tangential_correction_m_by_anchor"
            )
        true("order8_natural_contact_contact_side_closure_enabled")
        exact(
            "order8_natural_contact_contact_anchor_target_speed_limit_mps",
            config.anchor_translation_speed_limit_mps,
        )
        exact(
            "order8_natural_contact_contact_near_anchor_target_speed_limit_mps",
            0.2 * config.anchor_translation_speed_limit_mps,
        )
        exact(
            "order8_natural_contact_contact_near_anchor_slowdown_error_m",
            config.contact_near_surface_slowdown_m,
        )
        exact(
            "order8_natural_contact_contact_surface_anchor_target_speed_limit_mps",
            config.contact_surface_creep_speed_limit_mps,
        )
        exact(
            "order8_natural_contact_contact_surface_anchor_speed_boundary_m",
            config.contact_surface_arm_clearance_m,
        )
        exact(
            "order8_natural_contact_contact_anchor_target_speed_schedule",
            "nonprivileged_three_tier_precenter_then_symmetric_creep_close_"
            "with_opposing_clearance_synchronization_v8",
        )
        exact(
            "order8_natural_contact_contact_clearance_sync_method",
            "closer_surface_linear_slowdown_farther_surface_full_tier_speed_v1",
        )
        exact(
            "order8_natural_contact_contact_clearance_sync_deadband_m",
            config.contact_clearance_sync_deadband_m,
        )
        exact(
            "order8_natural_contact_contact_clearance_sync_full_slowdown_m",
            config.contact_clearance_sync_full_slowdown_m,
        )
        exact(
            "order8_natural_contact_contact_clearance_sync_minimum_speed_scale",
            config.contact_clearance_sync_minimum_speed_scale,
        )
        if not _positive_int(
            report.get(
                "order8_natural_contact_contact_clearance_sync_active_step_count"
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_clearance_sync_active_step_count"
            )
        if not _finite_non_negative(
            report.get("order8_natural_contact_max_contact_clearance_imbalance_m")
        ):
            failures.append(
                "invalid:order8_natural_contact_max_contact_clearance_imbalance_m"
            )
        hold_pose = report.get("order8_natural_contact_contact_axial_hold_base_pose")
        if not (
            isinstance(hold_pose, list)
            and len(hold_pose) == 7
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in hold_pose
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_axial_hold_base_pose"
            )
        axial_settle_dwell = report.get(
            "order8_natural_contact_contact_axial_settle_dwell_s"
        )
        if (
            not _finite_non_negative(axial_settle_dwell)
            or float(axial_settle_dwell) < config.contact_stall_dwell_s - 1.0e-9
        ):
            failures.append("below:order8_natural_contact_contact_axial_settle_dwell_s")
        exact(
            "order8_natural_contact_contact_axial_settle_position_tolerance_m",
            min(
                config.pregrasp_position_tolerance_m,
                config.contact_tangential_tolerance_m,
            ),
        )
        exact(
            "order8_natural_contact_contact_axial_settle_base_speed_tolerance_mps",
            config.pregrasp_linear_speed_tolerance_mps,
        )
        exact(
            "order8_natural_contact_pregrasp_staging_method",
            "selected_urdf_mesh_aabb_axial_retreat_bisection_v1",
        )
        exact(
            "order8_natural_contact_pregrasp_mesh_clearance_target_m",
            config.pregrasp_mesh_clearance_m,
        )
        exact(
            "order8_natural_contact_pregrasp_anchor_target_source",
            "selected_urdf_mesh_aabb_outward_opening_in_base_frame_v1",
        )
        exact(
            "order8_natural_contact_contact_motion_sequence",
            "mesh_open_then_floor_clear_object_relative_base_settle_then_"
            "known_grasp_ready_pose_then_one_shot_whole_structure_direction_"
            "fixed_velocity_close_until_simultaneous_load_qclose_then_slow_"
            "per_side_load_limited_position_preload_v25",
        )
        exact(
            "order8_natural_contact_contact_mesh_precenter_method",
            "one_shot_direction_seed_only_not_completion_gate_v3",
        )
        exact(
            "order8_natural_contact_contact_mesh_precenter_clearance_m",
            config.contact_near_surface_slowdown_m,
        )
        exact(
            "order8_natural_contact_contact_mesh_precenter_tangential_tolerance_m",
            config.contact_tangential_tolerance_m,
        )
        exact(
            "order8_natural_contact_mesh_pair_base_centering_method",
            "horizontal_approach_axis_mean_authored_mesh_patch_centering_v1",
        )
        pair_centering_correction = report.get(
            "order8_natural_contact_mesh_pair_base_centering_correction_world"
        )
        if not (
            isinstance(pair_centering_correction, list)
            and len(pair_centering_correction) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in pair_centering_correction
            )
            and math.sqrt(
                sum(float(value) ** 2 for value in pair_centering_correction)
            )
            <= config.contact_tangential_tolerance_m + 1.0e-9
        ):
            failures.append(
                "invalid:order8_natural_contact_mesh_pair_base_centering_correction_world"
            )
        # The approved smoke uses a known grasp-ready fixture and freezes one
        # whole-structure joint-velocity ratio at closure onset.  Mesh samples
        # seed that one-shot direction and remain privileged evidence, but a
        # receding mesh-precenter completion gate would be a grasp planner that
        # Order 8 deliberately does not own.
        false("order8_natural_contact_contact_mesh_precenter_complete")
        precenter_dwell = report.get(
            "order8_natural_contact_contact_mesh_precenter_dwell_s"
        )
        if not (
            _finite_non_negative(precenter_dwell)
            and math.isclose(float(precenter_dwell), 0.0, abs_tol=1.0e-12)
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_mesh_precenter_dwell_s"
            )
        exact("order8_natural_contact_contact_mesh_precenter_completed_time_s", None)
        exact(
            "order8_natural_contact_contact_centering_method",
            "known_object_relative_centroidal_pose_hold_without_closure_mesh_"
            "feedback_v3",
        )
        exact(
            "order8_natural_contact_contact_centering_joint_motion_mode",
            "all_docks_fixed_one_shot_velocity_ratio_without_receding_"
            "geometry_feedback_v4",
        )
        exact(
            "order8_natural_contact_contact_individual_arrest_centroidal_hold",
            "disabled_provisional_contact_may_separate_until_simultaneous_qclose_v1",
        )
        exact(
            "order8_natural_contact_contact_post_arrest_shape_hold_activation",
            "simultaneous_qclose_only_v1",
        )
        exact(
            "order8_natural_contact_contact_centering_settle_gate",
            "object_relative_final_base_pose_and_speed_dwell_before_joint_close_v1",
        )
        false("order8_natural_contact_contact_centering_raw_contact_input")
        exact(
            "order8_natural_contact_contact_centering_max_offset_limit_m",
            config.contact_centering_max_offset_m,
        )
        exact(
            "order8_natural_contact_contact_centering_max_tilt_limit_rad",
            config.contact_centering_max_tilt_rad,
        )
        exact(
            "order8_natural_contact_contact_centering_tilt_source",
            "not_used_in_surface_region_joint_only_close_v1",
        )
        for disabled_counter in (
            "order8_natural_contact_contact_centering_active_step_count",
            "order8_natural_contact_contact_continuous_balance_active_step_count",
            "order8_natural_contact_contact_sequential_reacquire_active_step_count",
            "order8_natural_contact_contact_sequential_centroidal_nudge_active_step_count",
            "order8_natural_contact_contact_sequential_latched_transfer_active_step_count",
            "order8_natural_contact_contact_sequential_joint_position_hold_step_count",
            "order8_natural_contact_contact_centering_cycle_count",
            "order8_natural_contact_post_first_arrest_creep_active_step_count",
            "order8_natural_contact_post_first_arrest_centroidal_transfer_active_step_count",
        ):
            zero(disabled_counter)
        maximum_centering_offset = report.get(
            "order8_natural_contact_contact_centering_max_observed_offset_m"
        )
        if (
            not _finite_non_negative(maximum_centering_offset)
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_centering_max_observed_offset_m"
            )
        maximum_centering_tilt = report.get(
            "order8_natural_contact_contact_centering_max_observed_tilt_rad"
        )
        if (
            not _finite_non_negative(maximum_centering_tilt)
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_centering_max_observed_tilt_rad"
            )
        if not _finite_non_negative(
            report.get("order8_natural_contact_contact_centering_max_measured_tilt_rad")
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_centering_max_measured_tilt_rad"
            )
        latched_centering_offset = report.get(
            "order8_natural_contact_contact_centering_latched_offset_world"
        )
        if not (
            isinstance(latched_centering_offset, list)
            and len(latched_centering_offset) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in latched_centering_offset
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_contact_centering_latched_offset_world"
            )
        exact(
            "order8_natural_contact_anchor_reference_frame",
            "measured_free_object_relative_authored_mesh_contact_rebased_through_"
            "measured_base_v4",
        )
        exact(
            "order8_natural_contact_contact_tangential_region_method",
            "authored_mesh_sample_componentwise_tangential_region_with_"
            "pair_mean_base_centering_v3",
        )
        exact(
            "order8_natural_contact_contact_tangential_tolerance_m",
            config.contact_tangential_tolerance_m,
        )
        true("order8_natural_contact_provisional_contact_separation_allowed")
        exact(
            "order8_natural_contact_contact_slip_enforcement_phase",
            "grasp_latched_object_frame_contact_point_displacement_v1",
        )
        exact(
            "order8_natural_contact_contact_slip_measurement_method",
            "force_weighted_selected_contact_centroid_object_frame_"
            "displacement_norm_from_grasp_confirmation_v1",
        )
        false("order8_natural_contact_contact_slip_speed_safe_hold_enabled")
        exact(
            "order8_natural_contact_contact_break_enforcement_phase",
            "after_verified_two_contact_grasp_dwell_until_planned_release_v2",
        )
        if not _finite_non_negative(
            report.get(
                "order8_natural_contact_max_provisional_acquisition_slip_speed_mps"
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_max_provisional_acquisition_slip_speed_mps"
            )
        true("order8_natural_contact_object_motion_retargeting_enabled")
        exact(
            "order8_natural_contact_object_motion_retarget_source",
            "measured_free_object_pose_read_only_v1",
        )
        if not _positive_int(
            report.get("order8_natural_contact_object_follow_active_step_count")
        ):
            failures.append(
                "invalid:order8_natural_contact_object_follow_active_step_count"
            )
        for retarget_metric in (
            "order8_natural_contact_max_observed_pre_qclose_object_translation_m",
            "order8_natural_contact_max_observed_base_retarget_translation_m",
        ):
            if not _finite_non_negative(report.get(retarget_metric)):
                failures.append(f"invalid:{retarget_metric}")
        zero("order8_natural_contact_object_follow_pose_write_count")
        true("order8_natural_contact_pregrasp_open_configuration_latched")
        true("order8_natural_contact_contact_axial_alignment_latched")
        expected_opening_anchor_ids = {
            str(anchor.anchor_id) for anchor in morphology_graph.robot_anchors
        }
        opening_distances = report.get(
            "order8_natural_contact_pregrasp_opening_distance_m_by_anchor"
        )
        if not (
            isinstance(opening_distances, dict)
            and set(opening_distances) == expected_opening_anchor_ids
            and all(_finite_non_negative(value) for value in opening_distances.values())
        ):
            failures.append(
                "invalid:order8_natural_contact_pregrasp_opening_distance_m_by_anchor"
            )
        opening_clearances = report.get(
            "order8_natural_contact_pregrasp_opening_clearance_m_by_anchor"
        )
        if not (
            isinstance(opening_clearances, dict)
            and set(opening_clearances) == expected_opening_anchor_ids
            and all(
                _finite_non_negative(value)
                and float(value) >= config.pregrasp_mesh_clearance_m - 1.0e-9
                for value in opening_clearances.values()
            )
        ):
            failures.append(
                "below:order8_natural_contact_pregrasp_opening_clearance_m_by_anchor"
            )
        minimum_achieved_clearance = max(
            0.0,
            config.pregrasp_mesh_clearance_m
            - config.anchor_command_tracking_tolerance_m,
        )
        exact(
            "order8_natural_contact_pregrasp_minimum_achieved_mesh_clearance_m",
            minimum_achieved_clearance,
        )
        achieved_clearance = report.get(
            "order8_natural_contact_pregrasp_achieved_mesh_clearance_m"
        )
        if (
            not _finite_non_negative(achieved_clearance)
            or float(achieved_clearance) < minimum_achieved_clearance - 1.0e-9
        ):
            failures.append(
                "below:order8_natural_contact_pregrasp_achieved_mesh_clearance_m"
            )
        true("order8_natural_contact_pregrasp_reachability_gate_passed")
        reachability_source = report.get(
            "order8_natural_contact_pregrasp_reachability_gate_source"
        )
        if reachability_source not in {
            "differential_and_achieved_mesh_clear_endpoint",
            "differential_whole_structure_jacobian",
            "achieved_mesh_clear_endpoint",
        }:
            failures.append(
                "invalid:order8_natural_contact_pregrasp_reachability_gate_source"
            )
        predicted_staging_clearance = report.get(
            "order8_natural_contact_pregrasp_mesh_clearance_predicted_m"
        )
        if (
            not _finite_non_negative(predicted_staging_clearance)
            or float(predicted_staging_clearance)
            < config.pregrasp_mesh_clearance_m - 1.0e-9
        ):
            failures.append(
                "below:order8_natural_contact_pregrasp_mesh_clearance_predicted_m"
            )
        staging_retreat = report.get(
            "order8_natural_contact_pregrasp_staging_retreat_distance_m"
        )
        if (
            not _finite_positive(staging_retreat)
            or float(staging_retreat) > config.initial_object_standoff_m
        ):
            failures.append(
                "invalid:order8_natural_contact_pregrasp_staging_retreat_distance_m"
            )
        approach_axis = report.get(
            "order8_natural_contact_pregrasp_approach_axis_world"
        )
        if not (
            isinstance(approach_axis, list)
            and len(approach_axis) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in approach_axis
            )
            and math.isclose(
                math.sqrt(sum(float(value) ** 2 for value in approach_axis)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_pregrasp_approach_axis_world"
            )
        true("order8_natural_contact_morphology_aware_module_root_targets")
        exact(
            "order8_natural_contact_module_root_target_source",
            "whole_structure_fk_of_measured_absolute_dock_state_and_"
            "planner_base_pose_v4",
        )
        module_frame_metadata = physical_model.metadata.get("baselink", {})
        expected_module_frame_link_id = (
            str(module_frame_metadata.get("name", "fc"))
            if isinstance(module_frame_metadata, dict)
            else "fc"
        )
        exact(
            "order8_natural_contact_module_frame_link_id",
            expected_module_frame_link_id,
        )
        exact(
            "order8_natural_contact_spawn_pose_conversion",
            "graph_module_frame_to_urdf_root_v1",
        )
        exact(
            "order8_natural_contact_runtime_module_pose_source",
            "isaac_named_module_frame_link_pose_and_twist_v1",
        )
        exact(
            "order8_natural_contact_qpid_centroidal_target_source",
            "single_full_morphology_rigid_body_model_from_planner_base_pose_"
            "and_measured_absolute_dock_state_v5",
        )
        exact(
            "order8_natural_contact_qpid_joint_motion_assumption",
            "quasi_static_measured_shape_without_commanded_joint_motion_"
            "compensation_even_during_slow_preload_v2",
        )
        false(
            "order8_natural_contact_qpid_unreached_joint_target_compensation"
        )
        if not _positive_int(
            report.get(
                "order8_natural_contact_morphology_aware_module_root_target_count"
            )
        ):
            failures.append(
                "invalid:order8_natural_contact_morphology_aware_module_root_target_count"
            )
        max_base_target_step = report.get(
            "order8_natural_contact_max_base_target_step_m"
        )
        if not _finite_non_negative(max_base_target_step):
            failures.append("invalid:order8_natural_contact_max_base_target_step_m")
        elif float(max_base_target_step) > (
            config.base_translation_speed_limit_mps * float(simulation_dt) + 1.0e-12
        ):
            failures.append("above:order8_natural_contact_max_base_target_step_m")
        max_contact_base_target_step = report.get(
            "order8_natural_contact_max_contact_base_target_step_m"
        )
        if not _finite_non_negative(max_contact_base_target_step):
            failures.append(
                "invalid:order8_natural_contact_max_contact_base_target_step_m"
            )
        elif float(max_contact_base_target_step) > (
            config.contact_base_translation_speed_limit_mps * float(simulation_dt)
            + 1.0e-12
        ):
            failures.append(
                "above:order8_natural_contact_max_contact_base_target_step_m"
            )
    exact(
        "order8_natural_contact_actuator_mapping_hash",
        build_actuator_mapping(morphology_graph, physical_model).stable_hash(),
    )
    component_mapping_hashes = report.get(
        "order8_natural_contact_component_actuator_mapping_hashes"
    )
    expected_component_mapping_hashes = {
        str(module.module_id): build_actuator_mapping(
            _order8_component_graph(morphology_graph, module.module_id),
            physical_model,
        ).stable_hash()
        for module in morphology_graph.modules
    }
    if component_mapping_hashes != expected_component_mapping_hashes:
        failures.append(
            "mismatch:order8_natural_contact_component_actuator_mapping_hashes"
        )

    true("order8_natural_contact_free_object")
    false("order8_natural_contact_object_kinematic")
    zero("order8_natural_contact_object_root_pose_write_count")
    false("order8_natural_contact_object_constraint_created")
    exact(
        "order8_natural_contact_object_root_pose_write_audit_method",
        "instrumented_post_spawn_object_pose_write_counter_v1",
    )
    exact(
        "order8_natural_contact_object_constraint_stage_audit_method",
        "usd_physics_joint_body_target_scan_v1",
    )
    zero("order8_natural_contact_object_constraint_reference_count")
    exact("order8_natural_contact_object_constraint_prim_paths", [])
    false("order8_natural_contact_pre_contact_object_pose_hold")
    false("order8_natural_contact_kinematic_payload_attach_used")
    false("order8_natural_contact_dynamic_assembly_filter_fallback_used")
    true("order8_natural_contact_selected_surface_actual_dock_mesh")
    false("order8_natural_contact_debug_command_mask_enabled")

    surface_pair = select_opposing_gripper_surface_pair(
        morphology_graph, physical_model
    )
    exact(
        "order8_natural_contact_selected_surface_module_ids",
        sorted((surface_pair.first.module_id, surface_pair.second.module_id)),
    )
    exact(
        "order8_natural_contact_selected_surface_port_global_ids",
        sorted((surface_pair.first.port_global_id, surface_pair.second.port_global_id)),
    )
    expected_geometry_refs = sorted(
        primitive.geometry_ref
        for surface in (surface_pair.first, surface_pair.second)
        for primitive in surface.collision_primitives
        if primitive.geometry_ref is not None
    )
    exact(
        "order8_natural_contact_selected_surface_geometry_refs",
        expected_geometry_refs,
    )
    exact(
        "order8_natural_contact_selected_gripper_material_method",
        "selected_authored_dock_mesh_compliant_material_v3",
    )
    exact(
        "order8_natural_contact_selected_gripper_material_path",
        ORDER8_SELECTED_GRIPPER_MATERIAL_PATH,
    )
    exact(
        "order8_natural_contact_selected_gripper_static_friction",
        config.selected_gripper_friction,
    )
    exact(
        "order8_natural_contact_selected_gripper_dynamic_friction",
        config.selected_gripper_friction,
    )
    exact(
        "order8_natural_contact_selected_gripper_friction_combine_mode",
        ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
    )
    true("order8_natural_contact_selected_gripper_compliant_contact_enabled")
    exact(
        "order8_natural_contact_selected_gripper_compliant_contact_stiffness_n_per_m",
        config.selected_gripper_compliant_contact_stiffness_n_per_m,
    )
    exact(
        "order8_natural_contact_selected_gripper_compliant_contact_damping_n_s_per_m",
        config.selected_gripper_compliant_contact_damping_n_s_per_m,
    )
    true(
        "order8_natural_contact_selected_gripper_compliant_contact_audit_passed"
    )
    exact(
        "order8_natural_contact_selected_gripper_material_binding_strength",
        "strongerThanDescendants",
    )
    true("order8_natural_contact_selected_gripper_material_binding_audit_passed")
    material_body_paths = report.get(
        "order8_natural_contact_selected_gripper_material_body_paths"
    )
    expected_surfaces = (surface_pair.first, surface_pair.second)
    if not (
        isinstance(material_body_paths, list)
        and len(material_body_paths) == len(expected_surfaces)
        and all(isinstance(path, str) and path for path in material_body_paths)
        and len(set(material_body_paths)) == len(expected_surfaces)
        and all(
            sum(
                path.startswith(f"/World/Order8/Module_{surface.module_id}/")
                and (
                    path.rsplit("/", 1)[-1] == surface.mechanism_link_id
                    or path.rsplit("/", 1)[-1].endswith(
                        "__" + surface.mechanism_link_id
                    )
                )
                for path in material_body_paths
            )
            == 1
            for surface in expected_surfaces
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_selected_gripper_material_body_paths"
        )
        material_body_paths = []
    material_collision_paths = report.get(
        "order8_natural_contact_selected_gripper_material_collision_prim_paths"
    )
    if not (
        isinstance(material_collision_paths, list)
        and len(material_collision_paths) >= len(expected_surfaces)
        and all(isinstance(path, str) and path for path in material_collision_paths)
        and len(set(material_collision_paths)) == len(material_collision_paths)
        and material_body_paths
        and all(
            any(
                path == body_path or path.startswith(body_path.rstrip("/") + "/")
                for body_path in material_body_paths
            )
            for path in material_collision_paths
        )
        and all(
            any(
                path == body_path or path.startswith(body_path.rstrip("/") + "/")
                for path in material_collision_paths
            )
            for body_path in material_body_paths
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_selected_gripper_material_collision_prim_paths"
        )
        material_collision_paths = []
    exact(
        "order8_natural_contact_selected_gripper_material_collision_prim_count",
        len(material_collision_paths),
    )
    exact(
        "order8_natural_contact_gripper_clearance_geometry",
        "urdf_collision_mesh_local_aabb_world_aabb_v1",
    )
    exact(
        "order8_natural_contact_gripper_clearance_mesh_aabb_count",
        len(expected_geometry_refs),
    )
    contact_report_body_counts = report.get(
        "order8_natural_contact_contact_report_body_counts"
    )
    if not (
        isinstance(contact_report_body_counts, dict)
        and set(contact_report_body_counts) == {"0", "1", "2"}
        and all(_positive_int(value) for value in contact_report_body_counts.values())
    ):
        failures.append("invalid:order8_natural_contact_contact_report_body_counts")
    if not _positive_int(
        report.get("order8_natural_contact_object_contact_report_body_count")
    ):
        failures.append(
            "invalid:order8_natural_contact_object_contact_report_body_count"
        )
    contact_view_sensor_count = report.get(
        "order8_natural_contact_robot_object_contact_view_sensor_count"
    )
    if not _positive_int(contact_view_sensor_count):
        failures.append(
            "invalid:order8_natural_contact_robot_object_contact_view_sensor_count"
        )
    elif (
        isinstance(contact_report_body_counts, dict)
        and all(_positive_int(value) for value in contact_report_body_counts.values())
        and contact_view_sensor_count != sum(contact_report_body_counts.values())
    ):
        failures.append(
            "mismatch:order8_natural_contact_robot_object_contact_view_sensor_count"
        )
    exact("order8_natural_contact_robot_object_contact_view_filter_count", 1)
    expected_selected_link_ids = sorted(
        f"module_{surface.module_id}:{surface.mechanism_link_id}"
        for surface in (surface_pair.first, surface_pair.second)
    )
    exact(
        "order8_natural_contact_selected_dock_link_ids",
        expected_selected_link_ids,
    )
    selected_link_ids = report.get("order8_natural_contact_selected_dock_link_ids")
    if not isinstance(selected_link_ids, list):
        selected_link_ids = []
    exact("order8_natural_contact_selected_contact_pair_count", 2)
    for key, require_contact_force in (
        ("order8_natural_contact_last_selected_normal_force_n_by_link", False),
        ("order8_natural_contact_max_selected_normal_force_n_by_link", True),
    ):
        force_by_link = report.get(key)
        if not (
            isinstance(force_by_link, dict)
            and set(force_by_link) == set(selected_link_ids)
            and all(_finite_non_negative(value) for value in force_by_link.values())
            and (
                not require_contact_force
                or all(
                    float(value)
                    >= config.contact_normal_force_threshold_n - 1.0e-9
                    for value in force_by_link.values()
                )
            )
        ):
            failures.append(f"invalid:{key}")

    exact(
        "order8_natural_contact_contact_closure_detection",
        "simultaneous_selected_terminal_joint_load_dwell_then_measured_"
        "qclose_and_privileged_contact_validation_v18",
    )
    exact(
        "order8_natural_contact_contact_anchor_orientation_task_weight",
        ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT,
    )
    exact(
        "order8_natural_contact_contact_anchor_task_hierarchy",
        "contact_translation_primary_measured_orientation_zero_error_then_"
        "verified_absolute_joint_hold_v2",
    )
    exact(
        "order8_natural_contact_contact_terminal_inward_overtravel_m",
        config.contact_closure_inward_overtravel_m,
    )
    exact(
        "order8_natural_contact_provisional_surface_load_settle_method",
        "one_sided_contact_may_separate_continuous_bounded_creep_until_"
        "simultaneous_nonprivileged_surface_load_qclose_v3",
    )
    false(
        "order8_natural_contact_provisional_surface_load_settle_raw_contact_input"
    )
    zero("order8_natural_contact_provisional_surface_load_settle_active_step_count")
    false("order8_natural_contact_contact_closure_raw_contact_input")
    true("order8_natural_contact_contact_terminal_target_snapshotted")
    true("order8_natural_contact_release_terminal_target_snapshotted")
    exact(
        "order8_natural_contact_release_terminal_target_source",
        "measured_closure_start_qopen_anchor_poses_base_v2",
    )
    true("order8_natural_contact_contact_configuration_latched")
    closure_reason = report.get("order8_natural_contact_contact_closure_reason")
    if closure_reason != (
        "dynamic_simultaneous_surface_region_arrest_then_"
        "load_limited_position_preload"
    ):
        failures.append("mismatch:order8_natural_contact_contact_closure_reason")
    stall_latched = report.get("order8_natural_contact_contact_stall_latched")
    if type(stall_latched) is not bool:
        failures.append("invalid:order8_natural_contact_contact_stall_latched")
    elif not stall_latched:
        failures.append("false:order8_natural_contact_contact_stall_latched")
    stall_dwell = report.get("order8_natural_contact_contact_stall_dwell_s")
    if (
        not _finite_non_negative(stall_dwell)
    ):
        failures.append("invalid:order8_natural_contact_contact_stall_dwell_s")
    configuration_dwell = report.get(
        "order8_natural_contact_contact_configuration_dwell_s"
    )
    if (
        not _finite_non_negative(configuration_dwell)
        or float(configuration_dwell) < config.contact_stall_dwell_s - 1.0e-9
    ):
        failures.append("invalid:order8_natural_contact_contact_configuration_dwell_s")
    expected_anchor_metric_ids = {
        str(anchor.anchor_id) for anchor in morphology_graph.robot_anchors
    }
    stall_dwell_by_anchor = report.get(
        "order8_natural_contact_contact_stall_dwell_s_by_anchor"
    )
    if not (
        isinstance(stall_dwell_by_anchor, dict)
        and set(stall_dwell_by_anchor) == expected_anchor_metric_ids
        and all(
            _finite_non_negative(value)
            for value in stall_dwell_by_anchor.values()
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_contact_stall_dwell_s_by_anchor"
        )
    expected_latched_anchor_ids = sorted(
        int(anchor_id) for anchor_id in expected_anchor_metric_ids
    )
    if (
        report.get("order8_natural_contact_contact_stall_latched_anchor_ids")
        != expected_latched_anchor_ids
    ):
        failures.append(
            "mismatch:order8_natural_contact_contact_stall_latched_anchor_ids"
        )
    for key in (
        "order8_natural_contact_contact_stall_command_error_m_by_anchor",
        "order8_natural_contact_contact_stall_anchor_speed_mps_by_anchor",
        "order8_natural_contact_contact_stall_selected_joint_load_nm_by_anchor",
    ):
        values = report.get(key)
        if not (
            isinstance(values, dict)
            and set(values) == expected_anchor_metric_ids
            and all(_finite_non_negative(value) for value in values.values())
        ):
            failures.append(f"invalid:{key}")
    stall_speeds = report.get(
        "order8_natural_contact_contact_stall_anchor_speed_mps_by_anchor"
    )
    # q_close is the simultaneous proximity/load arrest event.  These speeds
    # are retained as diagnostics, not reused as a slip-speed or closure gate;
    # the later grasp/contact dwell and grasp-referenced displacement monitor
    # provide the stable-contact proof.
    actuator_specs = physical_model.metadata.get("joint_actuator_specs", {})
    dock_spec = (
        actuator_specs.get("dock", {}) if isinstance(actuator_specs, dict) else {}
    )
    rated_torque_nm = float(
        dock_spec.get(
            "rated_torque_nm",
            dock_spec.get("continuous_torque_limit_nm", 1.3),
        )
    )
    selected_joint_load_threshold_nm = (
        ORDER8_CONTACT_STALL_RATED_TORQUE_FRACTION * rated_torque_nm
    )
    exact(
        "order8_natural_contact_contact_stall_selected_joint_load_threshold_nm",
        selected_joint_load_threshold_nm,
    )
    exact(
        "order8_natural_contact_contact_stall_selected_joint_load_source",
        "absolute_per_anchor_terminal_mechanism_joint_isaac_applied_"
        "torque_minus_estimated_virtual_drive_damping_torque_v4",
    )
    anchor_by_module = {
        int(anchor.module_id): anchor for anchor in morphology_graph.robot_anchors
    }
    expected_selected_joint_id_by_anchor = {
        str(anchor_by_module[surface.module_id].anchor_id): (
            f"module_{surface.module_id}:{surface.mechanism_joint_id}"
        )
        for surface in (surface_pair.first, surface_pair.second)
    }
    exact(
        "order8_natural_contact_contact_stall_selected_joint_id_by_anchor",
        expected_selected_joint_id_by_anchor,
    )
    stall_joint_loads = report.get(
        "order8_natural_contact_contact_stall_selected_joint_load_nm_by_anchor"
    )
    if isinstance(stall_joint_loads, dict) and any(
        float(value) < selected_joint_load_threshold_nm - 1.0e-9
        for value in stall_joint_loads.values()
        if _finite_non_negative(value)
    ):
        failures.append(
            "below:order8_natural_contact_contact_stall_selected_joint_load_nm_by_anchor"
        )
    exact(
        "order8_natural_contact_contact_stall_speed_reference_frame",
        "first_order_low_pass_selected_mesh_sample_point_object_normal_"
        "relative_speed_v2",
    )
    exact(
        "order8_natural_contact_contact_configuration_base_speed_tolerance_mps",
        config.pregrasp_linear_speed_tolerance_mps,
    )
    exact(
        "order8_natural_contact_contact_configuration_base_speed_gate",
        "world_base_linear_speed_with_object_relative_target_follow_v1",
    )
    exact(
        "order8_natural_contact_contact_mesh_clearance_arm_threshold_m",
        config.contact_surface_arm_clearance_m,
    )
    exact(
        "order8_natural_contact_contact_mesh_clearance_reacquire_tolerance_m",
        config.contact_penetration_noise_floor_m,
    )
    exact(
        "order8_natural_contact_contact_mesh_surface_distance_method",
        "sampled_urdf_collision_mesh_surface_to_observed_object_obb_v1",
    )
    exact(
        "order8_natural_contact_contact_wrench_application_mapping",
        "high_level_semantic_only_local_joint_offset_torque_forced_zero_v4",
    )
    false("order8_natural_contact_contact_wrench_application_raw_contact_input")
    mesh_surface_sample_count = report.get(
        "order8_natural_contact_contact_mesh_surface_sample_count"
    )
    if not _positive_int(mesh_surface_sample_count):
        failures.append(
            "invalid:order8_natural_contact_contact_mesh_surface_sample_count"
        )
    mesh_clearance_at_latch = report.get(
        "order8_natural_contact_contact_mesh_surface_clearance_at_latch_m_by_anchor"
    )
    if not (
        isinstance(mesh_clearance_at_latch, dict)
        and set(mesh_clearance_at_latch) == expected_anchor_metric_ids
        and all(
            _finite_non_negative(value)
            and float(value) <= config.contact_surface_arm_clearance_m + 1.0e-9
            for value in mesh_clearance_at_latch.values()
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_contact_mesh_surface_clearance_at_latch_m_by_anchor"
        )
    tangential_offset_at_latch = report.get(
        "order8_natural_contact_contact_tangential_offset_at_latch_m_by_anchor"
    )
    if not (
        isinstance(tangential_offset_at_latch, dict)
        and set(tangential_offset_at_latch) == expected_anchor_metric_ids
        and all(
            isinstance(offsets_m, list)
            and len(offsets_m) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and abs(float(value))
                <= config.contact_tangential_tolerance_m + 1.0e-9
                for value in offsets_m
            )
            for offsets_m in tangential_offset_at_latch.values()
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_contact_tangential_offset_at_latch_m_by_anchor"
        )
    exact(
        "order8_natural_contact_grasp_hold_anchor_target_source",
        "simultaneous_surface_region_qclose_measured_anchor_poses_in_base_frame_v3",
    )
    exact(
        "order8_natural_contact_grasp_hold_anchor_count",
        len(expected_anchor_metric_ids),
    )
    preload_elapsed_by_anchor = report.get(
        "order8_natural_contact_contact_force_ramp_elapsed_s_by_anchor"
    )
    if not (
        isinstance(preload_elapsed_by_anchor, dict)
        and set(preload_elapsed_by_anchor) == expected_anchor_metric_ids
        and all(
            _finite_non_negative(value)
            for value in preload_elapsed_by_anchor.values()
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_contact_force_ramp_elapsed_s_by_anchor"
        )
    max_force_scale_by_anchor = report.get(
        "order8_natural_contact_max_contact_force_scale_by_anchor"
    )
    if not (
        isinstance(max_force_scale_by_anchor, dict)
        and set(max_force_scale_by_anchor) == expected_anchor_metric_ids
        and all(
            _finite_non_negative(value)
            and math.isclose(float(value), 1.0, abs_tol=1.0e-9)
            for value in max_force_scale_by_anchor.values()
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_max_contact_force_scale_by_anchor"
        )

    zero("order8_natural_contact_dock_joint_structural_lock_count")
    true("order8_natural_contact_whole_structure_kinematics_used")
    physical_dof_count = report.get(
        "order8_natural_contact_dock_joint_physical_dof_count"
    )
    expected_joint_ids = list(
        ordered_global_dock_joint_ids(morphology_graph, physical_model)
    )
    expected_dock_joint_count = len(expected_joint_ids)
    if physical_dof_count != expected_dock_joint_count:
        failures.append("mismatch:order8_natural_contact_dock_joint_physical_dof_count")
    exact(
        "order8_natural_contact_anchor_jacobian_column_count",
        expected_dock_joint_count,
    )
    exact(
        "order8_natural_contact_anchor_jacobian_ids",
        sorted(anchor.anchor_id for anchor in morphology_graph.robot_anchors),
    )
    joint_coverage_keys = (
        "order8_natural_contact_dock_joint_expected_ids",
        "order8_natural_contact_dock_joint_observed_ids",
        "order8_natural_contact_dock_joint_position_commanded_ids",
        "order8_natural_contact_dock_joint_velocity_commanded_ids",
        "order8_natural_contact_dock_joint_torque_bias_commanded_ids",
    )
    exact(joint_coverage_keys[0], expected_joint_ids)
    for key in joint_coverage_keys[1:]:
        exact(key, sorted(expected_joint_ids))
    for key, expected_value in (
        (
            "order8_natural_contact_contact_yield_joint_drive_stiffness_targets_nm_per_rad",
            float(drive_spec.get("stiffness", 200.0)),
        ),
        (
            "order8_natural_contact_contact_yield_joint_drive_damping_targets_nms_per_rad",
            float(drive_spec.get("damping", 1.0)),
        ),
    ):
        targets = report.get(key)
        if not (
            isinstance(targets, dict)
            and set(targets) == set(expected_joint_ids)
            and all(
                _finite_positive(value)
                and math.isclose(
                    float(value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                for value in targets.values()
            )
        ):
            failures.append(f"mismatch:{key}")
    exact(
        "order8_natural_contact_dock_torque_bias_limit_nm",
        float(dock_spec.get("continuous_torque_limit_nm", 1.3)),
    )
    exact(
        "order8_natural_contact_dock_torque_bias_limit_basis",
        "ak40_10_continuous_torque_limit_v1",
    )
    exact(
        "order8_natural_contact_dock_continuous_torque_nm",
        float(dock_spec.get("continuous_torque_limit_nm", 1.3)),
    )
    exact(
        "order8_natural_contact_dock_peak_torque_nm",
        float(dock_spec.get("peak_torque_nm", 4.1)),
    )
    exact(
        "order8_natural_contact_dock_peak_current_a",
        float(dock_spec.get("peak_current_a", 7.3)),
    )
    exact(
        "order8_natural_contact_dock_actuator_telemetry_method",
        "requested_unclipped_limited_isaac_target_computed_applied_speed_"
        "and_linear_current_estimate_v2",
    )
    exact(
        "order8_natural_contact_dock_velocity_limit_sim_rad_s",
        float(drive_spec.get("safe_velocity_limit_rad_s", 3.0)),
    )
    true("order8_natural_contact_dock_actuator_envelope_audit_passed")
    zero("order8_natural_contact_dock_actuator_envelope_violation_step_count")
    telemetry_maxima = report.get(
        "order8_natural_contact_dock_actuator_telemetry_maxima"
    )
    peak_torque_nm = float(dock_spec.get("peak_torque_nm", 4.1))
    if not isinstance(telemetry_maxima, dict):
        failures.append(
            "invalid:order8_natural_contact_dock_actuator_telemetry_maxima"
        )
    else:
        for key in (
            "abs_requested_unclipped_torque_bias_nm",
            "abs_requested_limited_torque_bias_nm",
            "abs_isaac_effort_target_nm",
        ):
            value = telemetry_maxima.get(key)
            if (
                not _finite_non_negative(value)
                or not math.isclose(float(value), 0.0, abs_tol=1.0e-12)
            ):
                failures.append(
                    "invalid:order8_natural_contact_dock_actuator_telemetry_"
                    + key
                )
        computed_torque = telemetry_maxima.get("abs_isaac_computed_torque_nm")
        if not _finite_non_negative(computed_torque):
            failures.append(
                "invalid:order8_natural_contact_dock_actuator_telemetry_"
                "abs_isaac_computed_torque_nm"
            )
        applied_torque = telemetry_maxima.get("abs_isaac_applied_torque_nm")
        if (
            not _finite_non_negative(applied_torque)
            or float(applied_torque) > peak_torque_nm + 1.0e-6
        ):
            failures.append(
                "invalid:order8_natural_contact_dock_actuator_telemetry_"
                "abs_isaac_applied_torque_nm"
            )
        measured_velocity = telemetry_maxima.get("abs_measured_velocity_radps")
        if (
            not _finite_non_negative(measured_velocity)
            or float(measured_velocity)
            > float(drive_spec.get("safe_velocity_limit_rad_s", 3.0)) + 1.0e-6
        ):
            failures.append(
                "invalid:order8_natural_contact_dock_actuator_telemetry_"
                "abs_measured_velocity_radps"
            )
        estimated_current = telemetry_maxima.get("estimated_current_a")
        if (
            not _finite_non_negative(estimated_current)
            or float(estimated_current)
            > float(dock_spec.get("peak_current_a", 7.3)) + 1.0e-6
        ):
            failures.append(
                "invalid:order8_natural_contact_dock_actuator_telemetry_"
                "estimated_current_a"
            )

    exact(
        "order8_natural_contact_ordered_phase_trace",
        list(ORDER8_NATURAL_CONTACT_REQUIRED_PHASES),
    )
    exact(
        "order8_natural_contact_raw_contact_truth_role", ORDER8_RAW_CONTACT_TRUTH_ROLE
    )
    false("order8_natural_contact_raw_contact_truth_actor_input")
    false("order8_natural_contact_raw_contact_truth_qpid_command")
    exact("order8_natural_contact_raw_contact_failure_reasons", [])
    for key in (
        "order8_natural_contact_raw_contact_invalid_count",
        "order8_natural_contact_raw_contact_saturation_count",
        "order8_natural_contact_unintended_contact_count",
        "order8_natural_contact_object_drop_count",
        "order8_natural_contact_post_release_selected_contact_count",
        "order8_natural_contact_qp_infeasible_count",
        "order8_natural_contact_controller_failure_count",
        "order8_natural_contact_missing_actuator_target_count",
        "order8_natural_contact_unsupported_actuator_target_count",
        "order8_natural_contact_clipped_actuator_target_count",
        "order8_natural_contact_unresolved_actuator_target_count",
    ):
        zero(key)
    if not _positive_int(
        report.get("order8_natural_contact_payload_feedforward_active_count")
    ):
        failures.append(
            "invalid:order8_natural_contact_payload_feedforward_active_count"
        )
    exact(
        "order8_natural_contact_payload_feedforward_method",
        "verified_grasp_shared_commanded_lift_progress_and_centroidal_"
        "load_observer_known_payload_qpid_coupling_v7",
    )
    exact(
        "order8_natural_contact_payload_load_observer_method",
        "aggregate_centroidal_external_vertical_force_delta_from_lift_start_"
        "normalized_by_known_payload_weight_v1",
    )
    false("order8_natural_contact_payload_load_observer_raw_contact_input")
    exact(
        "order8_natural_contact_payload_load_transfer_driver",
        "slew_limited_max_commanded_lift_progress_observed_load_after_"
        "verified_grasp_v3",
    )
    exact(
        "order8_natural_contact_payload_commanded_lift_progress_method",
        "shared_lift_phase_elapsed_over_payload_transfer_duration_v1",
    )
    exact(
        "order8_natural_contact_contact_motion_entry_speed_ramp_method",
        "immediate_linear_lift_and_maintained_contact_phase_entry_ramp_v6",
    )
    exact(
        "order8_natural_contact_payload_feedforward_transition_duration_s",
        config.payload_load_transfer_s,
    )
    if not _positive_int(
        report.get("order8_natural_contact_payload_load_observer_valid_step_count")
    ):
        failures.append(
            "invalid:order8_natural_contact_payload_load_observer_valid_step_count"
        )
    zero("order8_natural_contact_payload_load_observer_invalid_step_count")
    if not _finite_non_negative(
        report.get(
            "order8_natural_contact_estimated_payload_lift_transfer_peak_scale"
        )
    ) or float(
        report.get(
            "order8_natural_contact_estimated_payload_lift_transfer_peak_scale",
            math.inf,
        )
    ) > 1.0:
        failures.append(
            "invalid:order8_natural_contact_estimated_payload_lift_transfer_peak_scale"
        )
    if not _finite_non_negative(
        report.get("order8_natural_contact_payload_lift_off_confirmed_time_s")
    ):
        failures.append(
            "invalid:order8_natural_contact_payload_lift_off_confirmed_time_s"
        )
    feedforward_lead = report.get(
        "order8_natural_contact_payload_feedforward_max_lead_over_observed_scale"
    )
    if not _finite_non_negative(feedforward_lead) or float(feedforward_lead) > 1.0:
        failures.append(
            "invalid:order8_natural_contact_payload_feedforward_max_lead_over_observed_scale"
        )
    commanded_progress_peak = report.get(
        "order8_natural_contact_payload_commanded_lift_progress_peak_scale"
    )
    if not (
        _finite_non_negative(commanded_progress_peak)
        and math.isclose(
            float(commanded_progress_peak),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        failures.append(
            "invalid:order8_natural_contact_payload_commanded_lift_progress_peak_scale"
        )
    feedforward_lag = report.get(
        "order8_natural_contact_payload_feedforward_max_lag_behind_commanded_"
        "progress_scale"
    )
    if not _finite_non_negative(feedforward_lag) or float(feedforward_lag) > 1.0e-9:
        failures.append(
            "invalid:order8_natural_contact_payload_feedforward_max_lag_behind_"
            "commanded_progress_scale"
        )
    false("order8_natural_contact_payload_feedforward_object_constraint")
    exact(
        "order8_natural_contact_lift_acceleration_bias_method",
        "known_payload_mass_times_shared_lift_progress_world_vertical_"
        "policy_command_residual_wrench_v1",
    )
    exact(
        "order8_natural_contact_lift_acceleration_bias_qpid_application",
        "policy_command_residual_wrench_body_centroidal_only_v1",
    )
    false("order8_natural_contact_lift_acceleration_bias_raw_contact_input")
    exact(
        "order8_natural_contact_lift_acceleration_bias_payload_mass_kg",
        config.object_mass_kg,
    )
    exact(
        "order8_natural_contact_lift_payload_acceleration_mps2",
        config.lift_payload_acceleration_mps2,
    )
    exact(
        "order8_natural_contact_lift_acceleration_bias_removal_s",
        config.lift_acceleration_bias_removal_s,
    )
    exact(
        "order8_natural_contact_lift_acceleration_bias_removal_method",
        "cubic_smoothstep_zero_endpoint_slope_v1",
    )
    lift_bias_active_count = report.get(
        "order8_natural_contact_lift_acceleration_bias_active_count"
    )
    if not _positive_int(lift_bias_active_count):
        failures.append(
            "invalid:order8_natural_contact_lift_acceleration_bias_active_count"
        )
    if report.get(
        "order8_natural_contact_lift_acceleration_bias_policy_command_active_count"
    ) != lift_bias_active_count:
        failures.append(
            "mismatch:order8_natural_contact_lift_acceleration_bias_policy_"
            "command_active_count"
        )
    zero("order8_natural_contact_lift_acceleration_bias_non_lift_active_count")
    lift_bias_peak_scale = report.get(
        "order8_natural_contact_lift_acceleration_bias_peak_scale"
    )
    lift_bias_lift_off_scale = report.get(
        "order8_natural_contact_lift_acceleration_bias_lift_off_scale"
    )
    for key, value in (
        (
            "order8_natural_contact_lift_acceleration_bias_peak_scale",
            lift_bias_peak_scale,
        ),
        (
            "order8_natural_contact_lift_acceleration_bias_lift_off_scale",
            lift_bias_lift_off_scale,
        ),
    ):
        if not (
            _finite_positive(value)
            and float(value) <= 1.0
        ):
            failures.append(f"invalid:{key}")
    expected_peak_lift_bias_force_n = (
        float(config.object_mass_kg)
        * float(config.lift_payload_acceleration_mps2)
        * (
            float(lift_bias_peak_scale)
            if _finite_positive(lift_bias_peak_scale)
            else math.nan
        )
    )
    for key in (
        "order8_natural_contact_lift_acceleration_bias_peak_force_world_z_n",
        "order8_natural_contact_lift_acceleration_bias_peak_residual_force_"
        "body_norm_n",
    ):
        value = report.get(key)
        if not (
            _finite_positive(value)
            and math.isclose(
                float(value),
                expected_peak_lift_bias_force_n,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            failures.append(f"mismatch:{key}")
    for key in (
        "order8_natural_contact_last_lift_acceleration_bias_scale",
        "order8_natural_contact_last_lift_acceleration_bias_force_world_z_n",
    ):
        value = report.get(key)
        if not (
            _finite_non_negative(value)
            and math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            failures.append(f"mismatch:{key}")
    last_lift_bias_wrench = report.get(
        "order8_natural_contact_last_lift_acceleration_residual_wrench_body"
    )
    if not (
        isinstance(last_lift_bias_wrench, list)
        and len(last_lift_bias_wrench) == 6
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            for value in last_lift_bias_wrench
        )
    ):
        failures.append(
            "mismatch:order8_natural_contact_last_lift_acceleration_"
            "residual_wrench_body"
        )
    lift_off_time = report.get(
        "order8_natural_contact_payload_lift_off_confirmed_time_s"
    )
    lift_bias_removal_time = report.get(
        "order8_natural_contact_lift_acceleration_bias_removal_complete_time_s"
    )
    if not (
        _finite_non_negative(lift_off_time)
        and _finite_non_negative(lift_bias_removal_time)
        and float(lift_bias_removal_time)
        >= float(lift_off_time) + float(config.lift_acceleration_bias_removal_s)
        - 1.0e-9
    ):
        failures.append(
            "invalid:order8_natural_contact_lift_acceleration_bias_"
            "removal_complete_time_s"
        )
    exact("order8_natural_contact_constraint_identity_failures", [])
    exact("order8_natural_contact_failure_reason", None)

    monitor_payload = report.get("order8_natural_contact_monitor_result")
    monitor_result: Order8NaturalContactResult | None = None
    if not isinstance(monitor_payload, dict):
        failures.append("invalid:order8_natural_contact_monitor_result")
    else:
        try:
            monitor_result = Order8NaturalContactResult.from_dict(monitor_payload)
        except (SchemaValidationError, TypeError, ValueError):
            failures.append("invalid:order8_natural_contact_monitor_result")
    if monitor_result is not None:
        if not monitor_result.passed:
            failures.append("false:order8_natural_contact_monitor_result.passed")
        if monitor_result.final_phase != Order8NaturalContactPhase.COMPLETE:
            failures.append(
                "mismatch:order8_natural_contact_monitor_result.final_phase"
            )
        if monitor_result.config_hash != config.stable_hash():
            failures.append(
                "mismatch:order8_natural_contact_monitor_result.config_hash"
            )
        if set(monitor_result.selected_dock_link_ids) != set(selected_link_ids):
            failures.append(
                "mismatch:order8_natural_contact_monitor_result.selected_dock_link_ids"
            )
        if monitor_result.result_version != ORDER8_NATURAL_CONTACT_RESULT_VERSION:
            failures.append(
                "mismatch:order8_natural_contact_monitor_result.result_version"
            )
        for name in (
            "grasp_acquired",
            "lift_acquired",
            "transport_acquired",
            "release_contact_free_acquired",
            "retreat_clearance_acquired",
            "settle_acquired",
        ):
            if getattr(monitor_result, name) is not True:
                failures.append(f"false:order8_natural_contact_monitor_result.{name}")
        if not monitor_result.attempted or monitor_result.step_count <= 0:
            failures.append("invalid:order8_natural_contact_monitor_result.attempted")
        if not _finite_positive(monitor_result.duration_s):
            failures.append("invalid:order8_natural_contact_monitor_result.duration_s")
        if monitor_result.object_dropped:
            failures.append("true:order8_natural_contact_monitor_result.object_dropped")
        if monitor_result.unintended_contact_count != 0:
            failures.append(
                "nonzero:order8_natural_contact_monitor_result.unintended_contact_count"
            )
        if monitor_result.failure_reasons:
            failures.append(
                "nonempty:order8_natural_contact_monitor_result.failure_reasons"
            )
        metric_limits = (
            (
                "max_force_per_selected_contact_n",
                config.max_force_per_contact_n,
            ),
            (
                "max_torque_per_selected_contact_nm",
                config.max_torque_per_contact_nm,
            ),
            ("max_penetration_m", config.max_penetration_m),
        )
        for name, limit in metric_limits:
            if getattr(monitor_result, name) > float(limit) + 1.0e-12:
                failures.append(f"above:order8_natural_contact_monitor_result.{name}")
        if not _finite_non_negative(
            monitor_result.max_provisional_acquisition_slip_speed_mps
        ):
            failures.append(
                "invalid:order8_natural_contact_monitor_result."
                "max_provisional_acquisition_slip_speed_mps"
            )
        elif (
            report.get(
                "order8_natural_contact_max_provisional_acquisition_slip_speed_mps"
            )
            != monitor_result.max_provisional_acquisition_slip_speed_mps
        ):
            failures.append(
                "mismatch:order8_natural_contact_max_provisional_acquisition_slip_speed_mps"
            )
        if set(
            monitor_result.max_contact_point_slip_displacement_m_by_link
        ) != set(
            expected_selected_link_ids
        ):
            failures.append(
                "mismatch:order8_natural_contact_monitor_result."
                "contact_point_slip_links"
            )
        elif any(
            value > config.max_contact_point_slip_displacement_m + 1.0e-12
            for value in (
                monitor_result.max_contact_point_slip_displacement_m_by_link.values()
            )
        ):
            failures.append(
                "above:order8_natural_contact_monitor_result."
                "contact_point_slip_displacement"
            )
        exact(
            "order8_natural_contact_monitor_result_hash",
            stable_hash(monitor_result),
        )
    return failures


def _require_sha256(report: dict[str, Any], key: str, failures: list[str]) -> None:
    if key not in report:
        failures.append(f"missing:{key}")
    elif not _is_sha256(report[key]):
        failures.append(f"invalid:{key}")


def _order8_component_graph(
    morphology_graph: MorphologyGraph,
    module_id: int,
) -> MorphologyGraph:
    modules = [
        replace(
            module,
            is_base=True,
            pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        for module in morphology_graph.modules
        if module.module_id == module_id
    ]
    if len(modules) != 1:
        raise SchemaValidationError(
            f"Order8 cannot resolve component module {module_id}"
        )
    return MorphologyGraph(
        graph_id=f"{morphology_graph.graph_id}:order8-component:{module_id}",
        modules=modules,
        ports=[
            replace(port, occupied=False)
            for port in morphology_graph.ports
            if port.module_id == module_id
        ],
        dock_edges=[],
        robot_anchors=[
            anchor
            for anchor in morphology_graph.robot_anchors
            if anchor.module_id == module_id
        ],
        control_groups=[
            ControlGroup(
                f"component:{module_id}",
                [module_id],
                "order8_component",
            )
        ],
        base_module_id=module_id,
        is_closed_loop=False,
    )


def _distinct_nonempty_strings(value: object, *, minimum: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) >= minimum
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_positive(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_non_negative(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _run_json_command(command: list[str], timeout_s: float) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("WARP_CACHE_PATH", "/tmp/amsrr_warp_cache")
    Path(env["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def drain(stream: Any, sink: list[str], *, forward_progress: bool) -> None:
        try:
            for line in stream:
                sink.append(line)
                if forward_progress and line.startswith(
                    ORDER8_NATURAL_CONTACT_PROGRESS_PREFIX
                ):
                    print(line.rstrip("\n"), file=sys.stderr, flush=True)
        finally:
            stream.close()

    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        raise RuntimeError("Order8 probe pipes were not created")
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
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        stderr = "".join(stderr_lines)
        raise RuntimeError(
            f"Order8 probe timed out after {float(timeout_s):.1f}s; "
            f"progress tail: {stderr[-4000:]}"
        ) from exc
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
            "Order8 probe produced no JSON report "
            f"(returncode={returncode}): {stderr[-1000:]}"
        )
    payload["command_returncode"] = returncode
    payload["command_stderr_tail"] = stderr[-4000:]
    return payload
