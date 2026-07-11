from __future__ import annotations

import json
from dataclasses import replace

import pytest

from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.logging.episode_archive import read_episode_archives_jsonl
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.random_morphology_takeoff import (
    RandomMorphologyTakeoffEnv,
    RandomMorphologyTakeoffResult,
)
from amsrr.training.random_morphology_takeoff_runner import (
    RandomMorphologyTakeoffRunnerConfig,
    RandomMorphologyTakeoffRunner,
    load_random_morphology_takeoff_runner_config,
)
from amsrr.utils.hashing import hash_file, stable_hash


def test_random_morphology_takeoff_runner_config_loader() -> None:
    runner_config, takeoff_config = load_random_morphology_takeoff_runner_config(
        "configs/training/random_morphology_takeoff.yaml"
    )
    assert runner_config.runner_version == "random_morphology_takeoff_runner_v1"
    assert runner_config.dry_run is True
    assert runner_config.max_sampling_attempts == 256
    assert runner_config.archive_path == "artifacts/p4_full/random_morphology_takeoff.jsonl"
    assert takeoff_config.floor_clearance_m == 0.002
    assert takeoff_config.floor_contact_force_threshold_n == 0.5
    assert takeoff_config.floor_contact_dwell_duration_s == 0.10
    assert takeoff_config.exact_cross_module_contact_force_threshold_n == 0.001
    assert takeoff_config.exact_cross_module_contact_max_patches_per_body_pair == 8
    assert takeoff_config.initial_root_position_tolerance_m == 0.002
    assert takeoff_config.initial_root_attitude_tolerance_rad == 0.001
    assert takeoff_config.settle_duration_s == 1.0
    assert takeoff_config.settle_dwell_duration_s == 0.25
    assert takeoff_config.takeoff_ramp_duration_s == 2.0
    assert takeoff_config.hover_acquisition_timeout_s == 2.0
    assert takeoff_config.hover_hold_duration_s == 1.0
    assert takeoff_config.hover_linear_speed_threshold_mps == 0.15
    assert takeoff_config.hover_angular_speed_threshold_rad_s == 0.25
    assert takeoff_config.required_steps == 1201


def test_random_morphology_takeoff_runner_accepts_source_spec_maximum() -> None:
    RandomMorphologyTakeoffRunnerConfig(module_count=8).validate()
    with pytest.raises(SchemaValidationError, match=r"\[2, 8\]"):
        RandomMorphologyTakeoffRunnerConfig(module_count=9).validate()


def test_random_morphology_takeoff_runner_persists_dry_contract(tmp_path) -> None:
    runner_config, takeoff_config = load_random_morphology_takeoff_runner_config(
        "configs/training/random_morphology_takeoff.yaml"
    )
    runner_config = replace(runner_config, archive_path=None)
    physical_model = build_physical_model_from_config(takeoff_config.robot_model_config_path)
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(seed=7, module_count=3)
    backend = IsaacLabBackend(load_isaac_lab_backend_config(takeoff_config.backend_config_path))
    env = RandomMorphologyTakeoffEnv(
        config=takeoff_config,
        backend=backend,
        physical_model=physical_model,
    )
    runner = RandomMorphologyTakeoffRunner(
        runner_config=runner_config,
        takeoff_config=takeoff_config,
        env=env,
    )
    report_path = tmp_path / "takeoff.json"

    result = runner.run(
        morphology,
        report_path=report_path,
        sampling_metadata={"source": "unit_test", "requested_seed": 7},
    )
    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.takeoff_result.dry_run is True
    assert result.takeoff_result.unit_contract_passed is True
    assert result.takeoff_result.real_isaac_passed is False
    assert result.feasibility_result.feasible is True
    assert result.morphology_graph.graph_id == morphology.graph_id
    assert persisted["morphology_graph"]["graph_id"] == morphology.graph_id
    assert persisted["feasibility_result"]["feasible"] is True
    assert persisted["takeoff_result"]["placement"]["floor_gap_m"] == pytest.approx(0.002)
    assert persisted["sampling_metadata"] == {"source": "unit_test", "requested_seed": 7}
    assert result.physical_model_hash == physical_model.stable_hash()
    assert result.robot_urdf_hash == hash_file(physical_model.urdf_path)
    assert result.collision_geometry_hash == collision_geometry_content_hash(
        physical_model,
        mesh_search_dirs=takeoff_config.mesh_search_dirs,
    )
    assert persisted["physical_model_hash"] == result.physical_model_hash
    assert persisted["robot_urdf_hash"] == result.robot_urdf_hash
    assert persisted["collision_geometry_hash"] == result.collision_geometry_hash
    assert result.backend_config_hash == stable_hash(backend.config)
    assert persisted["backend_config_hash"] == result.backend_config_hash
    assert result.archive_episode_id is None


def test_random_morphology_takeoff_runner_fails_closed_before_env_for_infeasible_graph(tmp_path) -> None:
    runner_config, takeoff_config = load_random_morphology_takeoff_runner_config(
        "configs/training/random_morphology_takeoff.yaml"
    )
    runner_config = replace(runner_config, archive_path=None)
    physical_model = build_physical_model_from_config(takeoff_config.robot_model_config_path)
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(seed=0, module_count=3)
    disconnected = type(morphology)(
        graph_id="disconnected-before-takeoff",
        modules=morphology.modules,
        ports=morphology.ports,
        dock_edges=morphology.dock_edges[:1],
        robot_anchors=morphology.robot_anchors,
        control_groups=morphology.control_groups,
        base_module_id=morphology.base_module_id,
        is_closed_loop=False,
    )
    runner = RandomMorphologyTakeoffRunner(
        runner_config=runner_config,
        takeoff_config=takeoff_config,
    )

    result = runner.run(disconnected, report_path=tmp_path / "rejected.json")

    assert result.feasibility_result.feasible is False
    assert result.takeoff_result.attempted is False
    assert result.takeoff_result.unit_contract_passed is False
    assert result.takeoff_result.placement == {}
    assert result.takeoff_result.failure_reason == "morphology_flight_feasibility_failed"


def test_random_morphology_takeoff_runner_writes_typed_episode_archive(tmp_path) -> None:
    runner_config, takeoff_config = load_random_morphology_takeoff_runner_config(
        "configs/training/random_morphology_takeoff.yaml"
    )
    runner_config = replace(
        runner_config,
        dry_run=False,
        report_path=None,
        archive_path=None,
    )
    physical_model = build_physical_model_from_config(takeoff_config.robot_model_config_path)
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(seed=7, module_count=3)
    controller_status = ControllerStatus(
        status="ok",
        qp_feasible=True,
        active_mode="rigid_body_qp",
        metrics={"allocation_residual_norm": 0.0},
    )
    runtime_observation = RuntimeObservation(
        time_s=4.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
            )
            for module in morphology.modules
        ],
        object_states=[],
        contact_states=[],
        controller_status=controller_status,
        task_progress=TaskProgressState(
            phase_label="hover_hold",
            progress_ratio=1.0,
            success=True,
        ),
    )
    policy_command = PolicyCommand(
        desired_body_pose=(0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0),
        desired_body_twist=[0.0] * 6,
    )
    controller_command = ControllerCommand(
        rotor_thrusts_n={"0/rotor_0": 4.0},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=controller_status,
    )
    actuator_record = {
        "time_s": 4.0,
        "backend": "isaac_lab",
        "morphology_graph_id": morphology.graph_id,
        "command_index": 0,
        "actuator_targets": [],
        "metrics": {"clipped_target_count": 0.0},
    }
    report = {
        "isaac_backed": True,
        "random_morphology_takeoff_hover_target_pose_world": [
            0.0,
            0.0,
            0.75,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "random_morphology_takeoff_duration_s": 4.0,
        "random_morphology_takeoff_phase_transitions": [
            {"phase": "hover_hold", "time_s": 3.0}
        ],
        "random_morphology_takeoff_runtime_observations": [runtime_observation.to_dict()],
        "random_morphology_takeoff_policy_commands": [policy_command.to_dict()],
        "random_morphology_takeoff_controller_commands": [controller_command.to_dict()],
        "random_morphology_takeoff_actuator_target_records": [actuator_record],
        "random_morphology_takeoff_artifacts": {
            "phase": "P4-full-order2",
            "backend": "isaac_lab",
            "is_p4_full_completion": False,
        },
    }
    takeoff_result = RandomMorphologyTakeoffResult(
        graph_id=morphology.graph_id,
        attempted=True,
        dry_run=False,
        isaac_backed=True,
        unit_contract_passed=True,
        real_isaac_passed=True,
        placement={
            "root_pose_world": [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0],
            "floor_gap_m": takeoff_config.floor_clearance_m,
        },
        metrics={"module_count": 3, "floor_placement_method": "test"},
        report=report,
    )
    env = _FakeTakeoffEnv(physical_model=physical_model, takeoff_result=takeoff_result)
    runner = RandomMorphologyTakeoffRunner(
        runner_config=runner_config,
        takeoff_config=takeoff_config,
        env=env,
    )
    archive_path = tmp_path / "takeoff.jsonl"
    report_path = tmp_path / "takeoff.json"

    result = runner.run(
        morphology,
        report_path=report_path,
        archive_path=archive_path,
        sampling_metadata={"requested_seed": 7, "proposal_seed": 107},
    )

    archives = read_episode_archives_jsonl(archive_path)
    assert len(archives) == 1
    archive = archives[0]
    assert result.archive_episode_id == archive.episode_id
    assert archive.success is True
    assert archive.robot_model_hash == physical_model.stable_hash()
    assert archive.config_hash == result.config_hash
    assert archive.runtime_observations == [runtime_observation]
    assert archive.policy_commands == [policy_command]
    assert archive.controller_commands == [controller_command]
    assert archive.actuator_target_records == [
        IsaacActuatorTargetRecord.from_dict(actuator_record).to_dict()
    ]
    assert archive.rollout_artifacts["morphology_hash"] == morphology.stable_hash()
    assert archive.rollout_artifacts["sampling_metadata"] == {
        "requested_seed": 7,
        "proposal_seed": 107,
    }
    assert archive.rollout_artifacts["is_p4_full_completion"] is False
    assert archive.reproducibility["physical_model_hash"] == result.physical_model_hash
    assert archive.reproducibility["urdf_hash"] == result.robot_urdf_hash
    assert archive.reproducibility["backend_config_hash"] == result.backend_config_hash
    persisted_result = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_result["archive_episode_id"] == archive.episode_id

    del report["random_morphology_takeoff_policy_commands"]
    with pytest.raises(
        ValueError,
        match="per-step report sequences must all be lists",
    ):
        runner.run(morphology, archive_path=tmp_path / "invalid.jsonl")

    for key in (
        "random_morphology_takeoff_runtime_observations",
        "random_morphology_takeoff_controller_commands",
        "random_morphology_takeoff_actuator_target_records",
    ):
        del report[key]
    with pytest.raises(ValueError, match="missing all per-step sequences"):
        runner.run(morphology, archive_path=tmp_path / "missing.jsonl")

    env.takeoff_result = replace(
        takeoff_result,
        real_isaac_passed=False,
        report={},
        failure_reason="probe_failed_before_step_records",
    )
    failed_result = runner.run(morphology, archive_path=archive_path)
    assert failed_result.archive_episode_id is None
    assert archive_path.read_text(encoding="utf-8") == ""


class _FakeTakeoffEnv:
    def __init__(self, *, physical_model, takeoff_result: RandomMorphologyTakeoffResult) -> None:
        self.physical_model = physical_model
        self.takeoff_result = takeoff_result

    def run(self, morphology_graph, *, dry_run: bool) -> RandomMorphologyTakeoffResult:
        assert morphology_graph.graph_id == self.takeoff_result.graph_id
        assert dry_run is False
        return self.takeoff_result
