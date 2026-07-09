from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.interaction_envelope import InteractionEnvelope, PrecisionRequirement
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import (
    EnvironmentSpec,
    GoalSpec,
    RobotConstraints,
    SafetySpec,
    SceneSpec,
    TaskSpec,
    TaskType,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology
from amsrr.simulation.p4_control_isaac_env import (
    P4ControlIsaacEnv,
    P4ControlLowLevelEnvConfig,
    load_p4_control_low_level_env_config,
)
from amsrr.simulation.p4_control_smoke import P4ControlSmokeResult
from amsrr.utils.config import load_config
from amsrr.utils.hashing import stable_hash


P4_CONTROL_LOW_LEVEL_RUNNER_VERSION = "p4_control_low_level_runner_v1"


@dataclass
class P4ControlLowLevelRunnerConfig(SchemaBase):
    seed: int = 0
    source_hash: str = "p4_control_low_level"
    runner_version: str = P4_CONTROL_LOW_LEVEL_RUNNER_VERSION
    dry_run: bool = True
    archive_path: str | None = "artifacts/p4_control/p4_control_smoke.jsonl"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"

    def validate(self) -> None:
        require_non_empty(self.source_hash, "P4ControlLowLevelRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "P4ControlLowLevelRunnerConfig.runner_version")
        require_non_empty(self.robot_model_config_path, "P4ControlLowLevelRunnerConfig.robot_model_config_path")


@dataclass
class P4ControlLowLevelRunnerResult(SchemaBase):
    dry_run: bool
    smoke_results: list[P4ControlSmokeResult]
    acceptance_report: Any
    archives: list[EpisodeArchive] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def load_p4_control_low_level_runner_config(
    path: str | Path,
) -> tuple[P4ControlLowLevelRunnerConfig, P4ControlLowLevelEnvConfig]:
    data = load_config(path)
    _, env_config = load_p4_control_low_level_env_config(path)
    return (
        P4ControlLowLevelRunnerConfig.from_dict(data.get("runner", {})),
        env_config,
    )


class P4ControlLowLevelRunner:
    """Run P4-control low-level smoke cases through the configured Isaac boundary."""

    def __init__(
        self,
        *,
        runner_config: P4ControlLowLevelRunnerConfig | None = None,
        env_config: P4ControlLowLevelEnvConfig | None = None,
        env: P4ControlIsaacEnv | None = None,
        archives: list[EpisodeArchive] | None = None,
        physical_model: PhysicalModel | None = None,
    ) -> None:
        self.runner_config = runner_config or P4ControlLowLevelRunnerConfig()
        self.env_config = env_config or P4ControlLowLevelEnvConfig()
        self.env = env or P4ControlIsaacEnv(
            config=self.env_config,
            backend=IsaacLabBackend(),
        )
        self.archives = archives or []
        self.physical_model = physical_model

    def run(self, *, archive_path: str | Path | None = None) -> P4ControlLowLevelRunnerResult:
        from amsrr.acceptance.p4_control_acceptance import run_p4_control_acceptance

        smoke_results = self.env.run_smokes(dry_run=self.runner_config.dry_run)
        archives = list(self.archives)
        if not archives and not self.runner_config.dry_run:
            archives = build_p4_control_smoke_archives(
                smoke_results,
                runner_config=self.runner_config,
                env_config=self.env_config,
                physical_model=self._physical_model(),
            )
        self.archives = archives
        acceptance_report = run_p4_control_acceptance(archives, smoke_results=smoke_results)
        output_path = archive_path
        if output_path is None and self.runner_config.archive_path is not None:
            output_path = self.runner_config.archive_path
        if output_path is not None and archives:
            write_episode_archives_jsonl(output_path, archives)
        return P4ControlLowLevelRunnerResult(
            dry_run=self.runner_config.dry_run,
            smoke_results=smoke_results,
            acceptance_report=acceptance_report,
            archives=archives,
            metrics={
                **acceptance_report.metrics,
                "dry_run": 1.0 if self.runner_config.dry_run else 0.0,
                "smoke_pass_count": float(sum(1 for result in smoke_results if result.passed)),
                "smoke_skip_count": float(sum(1 for result in smoke_results if result.skipped)),
                "config_hash": float(int(stable_hash(self.runner_config)[:8], 16)),
            },
        )

    def _physical_model(self) -> PhysicalModel:
        if self.physical_model is None:
            self.physical_model = build_physical_model_from_config(self.runner_config.robot_model_config_path)
        return self.physical_model


def ensure_real_smoke_requested(config: P4ControlLowLevelRunnerConfig) -> None:
    if config.dry_run:
        raise SchemaValidationError("P4-control real smoke requires runner.dry_run=false")


def build_p4_control_smoke_archives(
    smoke_results: list[P4ControlSmokeResult],
    *,
    runner_config: P4ControlLowLevelRunnerConfig,
    env_config: P4ControlLowLevelEnvConfig,
    physical_model: PhysicalModel,
) -> list[EpisodeArchive]:
    archives: list[EpisodeArchive] = []
    for command_index, result in enumerate(smoke_results):
        if result.skipped or not result.attempted:
            continue
        archives.append(
            _build_smoke_archive(
                result,
                command_index=command_index,
                runner_config=runner_config,
                env_config=env_config,
                physical_model=physical_model,
            )
        )
    return archives


def _build_smoke_archive(
    result: P4ControlSmokeResult,
    *,
    command_index: int,
    runner_config: P4ControlLowLevelRunnerConfig,
    env_config: P4ControlLowLevelEnvConfig,
    physical_model: PhysicalModel,
) -> EpisodeArchive:
    module_count = int(_smoke_metric(result, "module_count", 1.0))
    if result.smoke_name == "single_module_hover":
        module_count = 1
    module_spacing_m = 0.0 if module_count == 1 else env_config.fixed_morphology_module_spacing_m
    target_pose = _target_pose_for_smoke(result.smoke_name, env_config)
    morphology_graph = build_fixed_morphology(
        physical_model,
        graph_id=f"p4-control-{result.smoke_name}-morphology",
        module_count=module_count,
        module_spacing_m=module_spacing_m,
    )
    task_spec = _smoke_task_spec(result, target_pose, module_count, env_config)
    controller_status = _smoke_controller_status(result)
    controller_command = ControllerCommand(
        rotor_thrusts_n={},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=controller_status,
    )
    duration_s = _smoke_metric(result, "duration_s", env_config.smoke_duration_s)
    metrics = {
        **result.metrics,
        "summary_archive": 1.0,
        "isaac_backed": 1.0 if result.isaac_backed else 0.0,
        "p4_control_smoke_passed": 1.0 if result.passed else 0.0,
        "p4_full_completion": 0.0,
        "physical_success_claim": 0.0,
    }
    archive_seed = {
        "smoke_name": result.smoke_name,
        "command_index": command_index,
        "runner": runner_config,
        "env": env_config,
        "passed": result.passed,
    }
    config_hash = stable_hash({"runner": runner_config, "env": env_config})
    return EpisodeArchive(
        episode_id=f"p4-control-{result.smoke_name}-{stable_hash(archive_seed)[:8]}",
        task_spec=task_spec,
        task_hash=stable_hash(task_spec),
        geometry_hashes={},
        robot_model_hash=str(physical_model.metadata.get("urdf_hash", physical_model.stable_hash())),
        config_hash=config_hash,
        irg=_smoke_irg(result.smoke_name, task_spec.task_id),
        interaction_envelope=_smoke_envelope(result.smoke_name, task_spec.task_id, env_config),
        design_output=None,
        feasibility_result=None,
        assembly_plan=None,
        trajectory_records=[],
        policy_commands=[PolicyCommand(desired_body_pose=target_pose, desired_body_twist=[0.0] * 6)],
        controller_commands=[controller_command],
        rewards=[{"smoke_passed": 1.0 if result.passed else 0.0}],
        metrics=metrics,
        success=result.passed,
        failure_reason=None if result.passed else result.skip_reason or "p4_control_smoke_failed",
        runtime_observations=[
            _smoke_runtime_observation(
                morphology_graph=morphology_graph,
                time_s=duration_s,
                target_pose=target_pose,
                controller_status=controller_status,
                result=result,
            )
        ],
        actuator_target_records=[
            _smoke_actuator_target_record(
                result,
                morphology_graph_id=morphology_graph.graph_id,
                command_index=command_index,
                time_s=duration_s,
                controller_status=controller_status,
            )
        ],
        rollout_artifacts={
            "phase": "P4-control",
            "backend": result.backend,
            "archive_type": "smoke_summary",
            "smoke_name": result.smoke_name,
            "is_p4_full_completion": False,
            "isaac_backed": result.isaac_backed,
            "physical_success_claim": False,
            "object_grasp_carry_claim": False,
            "learning_claim": False,
        },
        learning_artifacts={},
        reproducibility={
            "source_hash": runner_config.source_hash,
            "runner_version": runner_config.runner_version,
            "config_hash": config_hash,
            "smoke_result_hash": stable_hash(result),
        },
    )


def _smoke_task_spec(
    result: P4ControlSmokeResult,
    target_pose: tuple[float, float, float, float, float, float, float],
    module_count: int,
    env_config: P4ControlLowLevelEnvConfig,
) -> TaskSpec:
    task_id = f"p4-control-{result.smoke_name}"
    return TaskSpec(
        task_id=task_id,
        task_type=TaskType.FREE_FLIGHT_NAVIGATION,
        scene=SceneSpec(
            geometry_library=[],
            objects=[],
            environment=EnvironmentSpec(support_surfaces=[], obstacles=[]),
        ),
        goals=[
            GoalSpec(
                goal_id=f"{result.smoke_name}-target",
                goal_type="free_flight_pose",
                time_limit_s=max(env_config.smoke_duration_s, _smoke_metric(result, "duration_s", 0.0)),
                target_entity_id="robot",
                target_pose_world=target_pose,
                target_twist_world=[0.0] * 6,
                tolerance_pos_m=env_config.position_error_threshold_m,
                tolerance_rot_rad=env_config.attitude_error_threshold_rad,
            )
        ],
        robot_constraints=RobotConstraints(min_modules=module_count, max_modules=max(module_count, 1)),
        safety=SafetySpec(),
        curriculum_tags=["P4-control", result.smoke_name],
        metadata={
            "archive_type": "smoke_summary",
            "isaac_backed": result.isaac_backed,
            "object_grasp_carry_claim": False,
        },
    )


def _smoke_irg(smoke_name: str, task_id: str) -> InteractionRequirementGraph:
    return InteractionRequirementGraph(
        irg_id=f"{task_id}-irg",
        task_id=task_id,
        nodes=[
            IRGNode(
                node_id=0,
                node_type=IRGNodeType.TASK,
                ref_id=task_id,
                priority=1.0,
                is_hard=True,
                active_phase_id=None,
                feature={"phase": "P4-control", "smoke_name": smoke_name},
            )
        ],
        edges=[],
        metadata={"archive_type": "smoke_summary"},
    )


def _smoke_envelope(
    smoke_name: str,
    task_id: str,
    env_config: P4ControlLowLevelEnvConfig,
) -> InteractionEnvelope:
    return InteractionEnvelope(
        envelope_id=f"{task_id}-envelope",
        task_id=task_id,
        required_contact_count_range=(0, 0),
        required_contact_modes=[],
        target_region_sets=[],
        wrench_space_requirements=[],
        precision_requirements=[
            PrecisionRequirement(
                target=smoke_name,
                tolerance_pos_m=env_config.position_error_threshold_m,
                tolerance_rot_rad=env_config.attitude_error_threshold_rad,
            )
        ],
    )


def _smoke_runtime_observation(
    *,
    morphology_graph,
    time_s: float,
    target_pose: tuple[float, float, float, float, float, float, float],
    controller_status: ControllerStatus,
    result: P4ControlSmokeResult,
) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=max(0.0, time_s),
        morphology_graph=morphology_graph,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=_module_summary_pose(target_pose, module.pose_in_design_frame),
                twist_world=[0.0] * 6,
                joint_positions={},
                joint_velocities={},
            )
            for module in morphology_graph.modules
        ],
        object_states=[],
        contact_states=[],
        controller_status=controller_status,
        task_progress=TaskProgressState(
            phase_label=result.smoke_name,
            progress_ratio=1.0 if result.attempted else 0.0,
            success=result.passed,
            failure_reason=None if result.passed else result.skip_reason,
            metrics={
                "smoke_passed": 1.0 if result.passed else 0.0,
                "position_error_m": _smoke_metric(result, "final_position_error_m", 0.0),
                "attitude_error_rad": _smoke_metric(result, "final_attitude_error_rad", 0.0),
            },
        ),
    )


def _smoke_controller_status(result: P4ControlSmokeResult) -> ControllerStatus:
    qp_infeasible_count = _smoke_metric(result, "qp_infeasible_count", 0.0)
    clipped_target_count = _smoke_metric(result, "clipped_target_count", 0.0)
    residual_norm = _smoke_metric(
        result,
        "last_bridge_allocation_residual_norm",
        _smoke_metric(result, "last_controller_allocation_residual_norm", 0.0),
    )
    if qp_infeasible_count > 0.0:
        status = "infeasible"
    elif not result.passed or clipped_target_count > 0.0:
        status = "warning"
    else:
        status = "ok"
    return ControllerStatus(
        status=status,
        qp_feasible=qp_infeasible_count == 0.0,
        active_mode="rigid_body_qp",
        message="P4-control smoke summary archive",
        metrics={
            "allocation_residual_norm": residual_norm,
            "residual_norm": residual_norm,
            "clipped_target_count": clipped_target_count,
            "clipped": clipped_target_count,
            "qp_infeasible_count": qp_infeasible_count,
            "missing_actuator_count": _smoke_metric(result, "missing_actuator_count", 0.0),
            "unsupported_actuator_count": _smoke_metric(result, "unsupported_actuator_count", 0.0),
            "smoke_passed": 1.0 if result.passed else 0.0,
            "summary_archive": 1.0,
        },
    )


def _smoke_actuator_target_record(
    result: P4ControlSmokeResult,
    *,
    morphology_graph_id: str,
    command_index: int,
    time_s: float,
    controller_status: ControllerStatus,
) -> dict[str, Any]:
    residual_norm = controller_status.metrics["allocation_residual_norm"]
    clipped_count = _smoke_metric(result, "clipped_target_count", 0.0)
    missing_count = _smoke_metric(result, "missing_actuator_count", 0.0)
    unsupported_count = _smoke_metric(result, "unsupported_actuator_count", 0.0)
    return {
        "time_s": max(0.0, time_s),
        "backend": result.backend,
        "morphology_graph_id": morphology_graph_id,
        "command_index": command_index,
        "actuator_targets": [],
        "clipped_targets": [],
        "missing_actuators": [],
        "unsupported_actuators": [],
        "allocation_residual_norm": residual_norm,
        "qp_status": controller_status.status,
        "metrics": {
            "allocation_residual_norm": residual_norm,
            "clipped_target_count": clipped_count,
            "missing_actuator_count": missing_count,
            "unsupported_actuator_count": unsupported_count,
            "controller_infeasible": 0.0 if controller_status.qp_feasible else 1.0,
            "summary_archive": 1.0,
        },
        "metadata": {
            "archive_type": "smoke_summary",
            "smoke_name": result.smoke_name,
            "detail_level": "summary",
        },
    }


def _target_pose_for_smoke(
    smoke_name: str,
    env_config: P4ControlLowLevelEnvConfig,
) -> tuple[float, float, float, float, float, float, float]:
    if smoke_name == "fixed_morphology_waypoint":
        x, y, z = env_config.waypoint_target_position_m
        qx, qy, qz, qw = _yaw_quat(env_config.waypoint_target_yaw_rad)
        return (float(x), float(y), float(z), qx, qy, qz, qw)
    return (0.0, 0.0, float(env_config.hover_target_height_m), 0.0, 0.0, 0.0, 1.0)


def _module_summary_pose(
    root_pose: tuple[float, float, float, float, float, float, float],
    module_pose: tuple[float, float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    return (
        root_pose[0] + module_pose[0],
        root_pose[1] + module_pose[1],
        root_pose[2] + module_pose[2],
        root_pose[3],
        root_pose[4],
        root_pose[5],
        root_pose[6],
    )


def _yaw_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _smoke_metric(result: P4ControlSmokeResult, suffix: str, default: float) -> float:
    value = result.metrics.get(f"{result.smoke_name}_{suffix}", result.metrics.get(suffix, default))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return default
