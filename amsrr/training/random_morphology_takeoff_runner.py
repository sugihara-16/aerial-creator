from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.feasibility.morphology_flight import (
    MorphologyFlightFeasibilityChecker,
    MorphologyFlightFeasibilityConfig,
    collision_geometry_content_hash,
)
from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.logging.episode_archive import EpisodeArchive, write_episode_archives_jsonl
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import ContactMode, SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerCommand, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import (
    CollisionModel,
    EnvironmentSpec,
    GeometrySpec,
    GeometryType,
    GoalSpec,
    RobotConstraints,
    SafetySpec,
    SceneSpec,
    SurfaceSpec,
    TaskSpec,
    TaskType,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.random_morphology_takeoff import (
    ORDER2_FLOOR_POSE_WORLD,
    ORDER2_FLOOR_SIZE_M,
    RANDOM_MORPHOLOGY_TAKEOFF_VERSION,
    RandomMorphologyTakeoffConfig,
    RandomMorphologyTakeoffEnv,
    RandomMorphologyTakeoffResult,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file, stable_hash


RANDOM_MORPHOLOGY_TAKEOFF_RUNNER_VERSION = "random_morphology_takeoff_runner_v1"


@dataclass
class RandomMorphologyTakeoffRunnerConfig(SchemaBase):
    seed: int = 0
    module_count: int | None = None
    dry_run: bool = True
    source_hash: str = "random_morphology_takeoff"
    runner_version: str = RANDOM_MORPHOLOGY_TAKEOFF_RUNNER_VERSION
    report_path: str | None = "artifacts/p4_full/random_morphology_takeoff.json"
    archive_path: str | None = "artifacts/p4_full/random_morphology_takeoff.jsonl"
    max_sampling_attempts: int = 64

    def validate(self) -> None:
        if self.seed < 0:
            raise SchemaValidationError("RandomMorphologyTakeoffRunnerConfig.seed must be non-negative")
        if self.module_count is not None and not 2 <= self.module_count <= 8:
            raise SchemaValidationError("RandomMorphologyTakeoffRunnerConfig.module_count must be in [2, 8]")
        require_non_empty(self.source_hash, "RandomMorphologyTakeoffRunnerConfig.source_hash")
        require_non_empty(self.runner_version, "RandomMorphologyTakeoffRunnerConfig.runner_version")
        if self.report_path is not None:
            require_non_empty(self.report_path, "RandomMorphologyTakeoffRunnerConfig.report_path")
        if self.archive_path is not None:
            require_non_empty(self.archive_path, "RandomMorphologyTakeoffRunnerConfig.archive_path")
        if self.max_sampling_attempts <= 0:
            raise SchemaValidationError(
                "RandomMorphologyTakeoffRunnerConfig.max_sampling_attempts must be positive"
            )


@dataclass
class RandomMorphologyTakeoffRunnerResult(SchemaBase):
    runner_version: str
    morphology_graph: MorphologyGraph
    feasibility_result: FeasibilityResult
    takeoff_result: RandomMorphologyTakeoffResult
    config_hash: str
    physical_model_hash: str
    robot_urdf_hash: str
    collision_geometry_hash: str
    backend_config_hash: str
    archive_episode_id: str | None = None
    sampling_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.runner_version, "RandomMorphologyTakeoffRunnerResult.runner_version")
        require_non_empty(self.config_hash, "RandomMorphologyTakeoffRunnerResult.config_hash")
        require_non_empty(
            self.physical_model_hash,
            "RandomMorphologyTakeoffRunnerResult.physical_model_hash",
        )
        require_non_empty(self.robot_urdf_hash, "RandomMorphologyTakeoffRunnerResult.robot_urdf_hash")
        require_non_empty(
            self.collision_geometry_hash,
            "RandomMorphologyTakeoffRunnerResult.collision_geometry_hash",
        )
        require_non_empty(
            self.backend_config_hash,
            "RandomMorphologyTakeoffRunnerResult.backend_config_hash",
        )


def load_random_morphology_takeoff_runner_config(
    path: str | Path,
) -> tuple[RandomMorphologyTakeoffRunnerConfig, RandomMorphologyTakeoffConfig]:
    data = load_config(path)
    return (
        RandomMorphologyTakeoffRunnerConfig.from_dict(data.get("runner", {})),
        RandomMorphologyTakeoffConfig.from_dict(data.get("takeoff", {})),
    )


class RandomMorphologyTakeoffRunner:
    """Execute Order-2 against an arbitrary already-feasible MorphologyGraph."""

    def __init__(
        self,
        *,
        runner_config: RandomMorphologyTakeoffRunnerConfig | None = None,
        takeoff_config: RandomMorphologyTakeoffConfig | None = None,
        env: RandomMorphologyTakeoffEnv | None = None,
        feasibility_checker: MorphologyFlightFeasibilityChecker | None = None,
    ) -> None:
        self.runner_config = runner_config or RandomMorphologyTakeoffRunnerConfig()
        self.takeoff_config = takeoff_config or RandomMorphologyTakeoffConfig()
        physical_model = build_physical_model_from_config(self.takeoff_config.robot_model_config_path)
        if env is None:
            backend_config = load_isaac_lab_backend_config(self.takeoff_config.backend_config_path)
            env = RandomMorphologyTakeoffEnv(
                config=self.takeoff_config,
                backend=IsaacLabBackend(backend_config),
                physical_model=physical_model,
            )
        self.env = env
        self.physical_model = getattr(env, "physical_model", physical_model)
        self.feasibility_checker = feasibility_checker or MorphologyFlightFeasibilityChecker(
            MorphologyFlightFeasibilityConfig(
                mesh_search_dirs=tuple(self.takeoff_config.mesh_search_dirs),
            )
        )

    def run(
        self,
        morphology_graph: MorphologyGraph,
        *,
        report_path: str | Path | None = None,
        archive_path: str | Path | None = None,
        sampling_metadata: dict[str, Any] | None = None,
    ) -> RandomMorphologyTakeoffRunnerResult:
        feasibility_result = self.feasibility_checker.check(morphology_graph, self.physical_model)
        if feasibility_result.feasible:
            takeoff_result = self.env.run(morphology_graph, dry_run=self.runner_config.dry_run)
        else:
            violation_codes = [violation.code for violation in feasibility_result.hard_violations]
            takeoff_result = RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=False,
                dry_run=self.runner_config.dry_run,
                isaac_backed=False,
                unit_contract_passed=False,
                real_isaac_passed=False,
                placement={},
                metrics={
                    "feasibility_passed": False,
                    "hard_violation_count": len(violation_codes),
                    "hard_violation_codes": violation_codes,
                },
                failure_reason="morphology_flight_feasibility_failed",
            )
        physical_model_hash = self.physical_model.stable_hash()
        robot_urdf_hash = hash_file(self.physical_model.urdf_path)
        collision_geometry_hash = collision_geometry_content_hash(
            self.physical_model,
            mesh_search_dirs=self.takeoff_config.mesh_search_dirs,
        )
        backend_config = getattr(getattr(self.env, "backend", None), "config", None)
        if backend_config is None:
            backend_config = load_isaac_lab_backend_config(
                self.takeoff_config.backend_config_path
            )
        backend_config_hash = stable_hash(backend_config)
        config_hash = stable_hash(
            {
                "runner": self.runner_config,
                "takeoff": self.takeoff_config,
                "backend_config": backend_config,
                "contract_version": RANDOM_MORPHOLOGY_TAKEOFF_VERSION,
                "physical_model_hash": physical_model_hash,
                "robot_urdf_hash": robot_urdf_hash,
                "collision_geometry_hash": collision_geometry_hash,
            }
        )
        archive = _episode_archive_from_takeoff_report(
            runner_config=self.runner_config,
            takeoff_config=self.takeoff_config,
            morphology_graph=morphology_graph,
            feasibility_result=feasibility_result,
            takeoff_result=takeoff_result,
            physical_model_hash=physical_model_hash,
            robot_urdf_hash=robot_urdf_hash,
            collision_geometry_hash=collision_geometry_hash,
            backend_config_hash=backend_config_hash,
            config_hash=config_hash,
            sampling_metadata=dict(sampling_metadata or {}),
        )
        output_archive_path = archive_path
        if output_archive_path is None:
            output_archive_path = self.runner_config.archive_path
        if output_archive_path is not None:
            # The runner owns this output path.  Always replace it so a failed
            # attempt cannot leave a prior episode masquerading as fresh evidence.
            write_episode_archives_jsonl(
                output_archive_path,
                [] if archive is None else [archive],
            )
        result = RandomMorphologyTakeoffRunnerResult(
            runner_version=self.runner_config.runner_version,
            morphology_graph=morphology_graph,
            feasibility_result=feasibility_result,
            takeoff_result=takeoff_result,
            config_hash=config_hash,
            physical_model_hash=physical_model_hash,
            robot_urdf_hash=robot_urdf_hash,
            collision_geometry_hash=collision_geometry_hash,
            backend_config_hash=backend_config_hash,
            archive_episode_id=None if archive is None else archive.episode_id,
            sampling_metadata=dict(sampling_metadata or {}),
        )
        output_path = report_path
        if output_path is None:
            output_path = self.runner_config.report_path
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(result.to_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
        return result


_REPORT_RUNTIME_OBSERVATIONS = "random_morphology_takeoff_runtime_observations"
_REPORT_POLICY_COMMANDS = "random_morphology_takeoff_policy_commands"
_REPORT_CONTROLLER_COMMANDS = "random_morphology_takeoff_controller_commands"
_REPORT_ACTUATOR_TARGET_RECORDS = "random_morphology_takeoff_actuator_target_records"


def _episode_archive_from_takeoff_report(
    *,
    runner_config: RandomMorphologyTakeoffRunnerConfig,
    takeoff_config: RandomMorphologyTakeoffConfig,
    morphology_graph: MorphologyGraph,
    feasibility_result: FeasibilityResult,
    takeoff_result: RandomMorphologyTakeoffResult,
    physical_model_hash: str,
    robot_urdf_hash: str,
    collision_geometry_hash: str,
    backend_config_hash: str,
    config_hash: str,
    sampling_metadata: dict[str, Any],
) -> EpisodeArchive | None:
    report = takeoff_result.report
    raw_sequences = {
        _REPORT_RUNTIME_OBSERVATIONS: report.get(_REPORT_RUNTIME_OBSERVATIONS),
        _REPORT_POLICY_COMMANDS: report.get(_REPORT_POLICY_COMMANDS),
        _REPORT_CONTROLLER_COMMANDS: report.get(_REPORT_CONTROLLER_COMMANDS),
        _REPORT_ACTUATOR_TARGET_RECORDS: report.get(_REPORT_ACTUATOR_TARGET_RECORDS),
    }
    if all(value is None for value in raw_sequences.values()):
        if takeoff_result.real_isaac_passed:
            raise SchemaValidationError(
                "successful random morphology takeoff report is missing all per-step sequences"
            )
        return None
    missing_or_invalid = [
        key for key, value in raw_sequences.items() if not isinstance(value, list)
    ]
    if missing_or_invalid:
        raise SchemaValidationError(
            "random morphology takeoff per-step report sequences must all be lists; "
            f"missing or invalid: {missing_or_invalid}"
        )

    runtime_observations = [
        RuntimeObservation.from_dict(item)
        for item in raw_sequences[_REPORT_RUNTIME_OBSERVATIONS]
    ]
    policy_commands = [
        PolicyCommand.from_dict(item)
        for item in raw_sequences[_REPORT_POLICY_COMMANDS]
    ]
    controller_commands = [
        ControllerCommand.from_dict(item)
        for item in raw_sequences[_REPORT_CONTROLLER_COMMANDS]
    ]
    actuator_target_records = [
        IsaacActuatorTargetRecord.from_dict(item).to_dict()
        if isinstance(item, dict)
        else _raise_invalid_actuator_record(index)
        for index, item in enumerate(raw_sequences[_REPORT_ACTUATOR_TARGET_RECORDS])
    ]
    sequence_lengths = {
        len(runtime_observations),
        len(policy_commands),
        len(controller_commands),
        len(actuator_target_records),
    }
    if len(sequence_lengths) != 1:
        raise SchemaValidationError(
            "random morphology takeoff per-step report sequences must be aligned, got lengths "
            f"{sorted(sequence_lengths)}"
        )
    if takeoff_result.real_isaac_passed and not runtime_observations:
        raise SchemaValidationError(
            "successful random morphology takeoff report must contain per-step sequences"
        )

    task_spec, floor_geometry = _takeoff_task_spec(
        morphology_graph,
        takeoff_config=takeoff_config,
        takeoff_result=takeoff_result,
        physical_model_hash=physical_model_hash,
        robot_urdf_hash=robot_urdf_hash,
        collision_geometry_hash=collision_geometry_hash,
    )
    builder_result = IRGBuilder().build_with_scene_graph(task_spec)
    irg = builder_result.irg
    interaction_envelope = InteractionEnvelopeExtractor().extract(irg)
    numeric_metrics = _numeric_archive_metrics(takeoff_result.metrics)
    numeric_metrics.update(
        {
            "success": 1.0 if takeoff_result.real_isaac_passed else 0.0,
            "isaac_backed": 1.0 if takeoff_result.isaac_backed else 0.0,
            "runtime_observation_count": float(len(runtime_observations)),
            "policy_command_count": float(len(policy_commands)),
            "controller_command_count": float(len(controller_commands)),
            "actuator_target_record_count": float(len(actuator_target_records)),
            "p4_full_completion": 0.0,
            "object_task_claim": 0.0,
            "learned_policy_claim": 0.0,
        }
    )
    archive_seed = {
        "runner_version": runner_config.runner_version,
        "graph_hash": morphology_graph.stable_hash(),
        "config_hash": config_hash,
        "takeoff_result_hash": takeoff_result.stable_hash(),
    }
    report_artifacts = report.get("random_morphology_takeoff_artifacts")
    if not isinstance(report_artifacts, dict):
        report_artifacts = {}
    return EpisodeArchive(
        episode_id=f"random-morphology-takeoff-{stable_hash(archive_seed)[:12]}",
        task_spec=task_spec,
        task_hash=task_spec.stable_hash(),
        geometry_hashes={floor_geometry.geometry_id: floor_geometry.stable_hash()},
        robot_model_hash=physical_model_hash,
        config_hash=config_hash,
        irg=irg,
        interaction_envelope=interaction_envelope,
        design_output=None,
        feasibility_result=feasibility_result,
        assembly_plan=None,
        trajectory_records=[],
        policy_commands=policy_commands,
        controller_commands=controller_commands,
        rewards=[
            {
                "random_morphology_takeoff_success": (
                    1.0 if takeoff_result.real_isaac_passed else 0.0
                )
            }
        ],
        metrics=numeric_metrics,
        success=takeoff_result.real_isaac_passed,
        failure_reason=takeoff_result.failure_reason,
        runtime_observations=runtime_observations,
        actuator_target_records=actuator_target_records,
        rollout_artifacts={
            "phase": "P4-full-order2",
            "backend": "isaac_lab" if takeoff_result.isaac_backed else "unavailable",
            "archive_type": "random_morphology_takeoff_per_step",
            "morphology_graph": morphology_graph.to_dict(),
            "morphology_hash": morphology_graph.stable_hash(),
            "placement": takeoff_result.placement,
            "phase_transitions": report.get(
                "random_morphology_takeoff_phase_transitions", []
            ),
            "probe_artifacts": report_artifacts,
            "sampling_metadata": sampling_metadata,
            "is_p4_full_completion": False,
            "physical_success_claim": "floor_takeoff_hover_only",
            "object_task_claim": False,
            "learned_policy_claim": False,
        },
        learning_artifacts={},
        reproducibility={
            "source_hash": runner_config.source_hash,
            "random_seed": runner_config.seed,
            "runner_version": runner_config.runner_version,
            "contract_version": RANDOM_MORPHOLOGY_TAKEOFF_VERSION,
            "config_hash": config_hash,
            "physical_model_hash": physical_model_hash,
            "urdf_hash": robot_urdf_hash,
            "collision_geometry_hash": collision_geometry_hash,
            "backend_config_hash": backend_config_hash,
            "morphology_hash": morphology_graph.stable_hash(),
            "takeoff_result_hash": takeoff_result.stable_hash(),
        },
    )


def _takeoff_task_spec(
    morphology_graph: MorphologyGraph,
    *,
    takeoff_config: RandomMorphologyTakeoffConfig,
    takeoff_result: RandomMorphologyTakeoffResult,
    physical_model_hash: str,
    robot_urdf_hash: str,
    collision_geometry_hash: str,
) -> tuple[TaskSpec, GeometrySpec]:
    target_pose_value = takeoff_result.report.get(
        "random_morphology_takeoff_hover_target_pose_world"
    )
    if not isinstance(target_pose_value, (list, tuple)) or len(target_pose_value) != 7:
        root_pose = takeoff_result.placement.get(
            "root_pose_world",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        target_pose_value = [
            float(root_pose[0]),
            float(root_pose[1]),
            float(root_pose[2]) + takeoff_config.hover_height_delta_m,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    target_pose = tuple(float(value) for value in target_pose_value)
    floor_geometry = GeometrySpec(
        geometry_id="order2_floor_geometry",
        geometry_type=GeometryType.BOX,
        primitive_params={"size_m": list(ORDER2_FLOOR_SIZE_M)},
        asset_path=None,
        collision_model=CollisionModel.PRIMITIVE,
    )
    module_types = sorted({module.module_type for module in morphology_graph.modules})
    task_id = f"random-morphology-takeoff-{morphology_graph.graph_id}"
    duration_s = max(
        takeoff_config.total_duration_s,
        float(takeoff_result.report.get("random_morphology_takeoff_duration_s", 0.0)),
    )
    task_spec = TaskSpec(
        task_id=task_id,
        task_type=TaskType.FREE_FLIGHT_NAVIGATION,
        scene=SceneSpec(
            geometry_library=[floor_geometry],
            objects=[],
            environment=EnvironmentSpec(
                support_surfaces=[
                    SurfaceSpec(
                        surface_id="order2_floor",
                        geometry_id=floor_geometry.geometry_id,
                        pose_world=ORDER2_FLOOR_POSE_WORLD,
                        allowed_contact_modes=[ContactMode.SUPPORT, ContactMode.BODY_CONTACT],
                        friction=1.0,
                    )
                ],
                obstacles=[],
            ),
        ),
        goals=[
            GoalSpec(
                goal_id="takeoff-hover-target",
                goal_type="free_flight_pose",
                time_limit_s=duration_s,
                target_entity_id="robot",
                target_pose_world=target_pose,  # type: ignore[arg-type]
                target_twist_world=[0.0] * 6,
                tolerance_pos_m=takeoff_config.position_error_threshold_m,
                tolerance_rot_rad=takeoff_config.attitude_error_threshold_rad,
            )
        ],
        robot_constraints=RobotConstraints(
            min_modules=len(morphology_graph.modules),
            max_modules=len(morphology_graph.modules),
            allowed_module_types=module_types,
            allow_closed_loop=False,
            max_docked_edges=len(morphology_graph.dock_edges),
        ),
        safety=SafetySpec(),
        curriculum_tags=["P4-full", "order2", "random-connected-morphology"],
        metadata={
            "archive_type": "random_morphology_takeoff_per_step",
            "morphology_hash": morphology_graph.stable_hash(),
            "physical_model_hash": physical_model_hash,
            "urdf_hash": robot_urdf_hash,
            "collision_geometry_hash": collision_geometry_hash,
            "p4_full_completion": False,
        },
    )
    return task_spec, floor_geometry


def _numeric_archive_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool))
    }


def _raise_invalid_actuator_record(index: int) -> dict[str, Any]:
    raise SchemaValidationError(
        f"random morphology takeoff actuator target record {index} must be a mapping"
    )
