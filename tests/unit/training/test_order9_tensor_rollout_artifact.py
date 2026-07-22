import copy
from pathlib import Path

import pytest
import torch

from amsrr.policies.order9_low_level_policy import ORDER9_GLOBAL_ACTION_SIZE
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.policies import ContactAssignment
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskRuntimeConfig,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.training.order9_online_dataset import write_order9_on_policy_dataset
from amsrr.training.order9_curriculum import (
    Order9RuntimeBenchmarkConfig,
    load_order9_learning_config,
)
from amsrr.training.order9_dataset import (
    load_order9_dataset,
    validate_order9_dataset_for_stage,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_production_benchmark import (
    build_order9_production_benchmark_report,
)
from amsrr.training.order9_tensor_dataset_builder import (
    build_order9_pi_l_on_policy_dataset,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_PRODUCTION_COLLECTOR_VERSION,
    Order9TensorRolloutArtifact,
    Order9TensorRolloutBuffer,
    load_order9_tensor_rollout_artifact,
    order9_pi_l_records_from_tensor_artifact,
    write_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash


def _artifact() -> Order9TensorRolloutArtifact:
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    modules = tuple(sorted(module.module_id for module in morphology.modules))
    local_joints = tuple(joint.joint_id for joint in physical.joints)
    command_joints = tuple(
        sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in physical.dock_ports
            }
        )
    )
    rotors = tuple(
        f"module_{module_id}:{rotor.rotor_id}"
        for module_id in modules
        for rotor in sorted(physical.rotors, key=lambda value: value.rotor_id)
    )
    vectoring = tuple(
        f"module_{module_id}:{rotor.vectoring_joint_ids[0]}"
        for module_id in modules
        for rotor in sorted(physical.rotors, key=lambda value: value.rotor_id)
    )
    task = build_order8_grasp_carry_task_spec(
        object_pose_world=(1.0, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.3, 0.4, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.2,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
        task_id="order9-raw-task",
    )
    assignments = [
        ContactAssignment(
            slot_id=0,
            anchor_id=anchor.anchor_id,
            candidate_id=index,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            wrench_lower=[-1.0, -1.0, 0.0, -0.1, -0.1, -0.1],
            wrench_upper=[1.0, 1.0, 2.0, 0.1, 0.1, 0.1],
            wrench_frame="contact",
        ).to_dict()
        for index, anchor in enumerate(
            sorted(morphology.robot_anchors, key=lambda value: value.anchor_id)
        )
    ]
    metadata = {
        "generation_id": "unit-generation",
        "pi_l_checkpoint_sha256": "1" * 64,
        "stage_id": "c2_pi_l_ppo_fixed_conservative",
        "stage_config_hash": "2" * 64,
        "curriculum_schedule_hash": "3" * 64,
        "config_hash": "4" * 64,
        "morphology_graph": morphology.to_dict(),
        "physical_model_hash": physical.stable_hash(),
        "urdf_hash": "5" * 64,
        "thrust_model_hash": "6" * 64,
        "robot_usd_sha256": "7" * 64,
        "simulator_version": "unit-isaac",
        "simulator_hash": "8" * 64,
        "device": "cpu",
        "random_seed": 9,
        "topology_randomized": False,
        "estimated_payload_mass_kg": 1.0,
        "estimated_payload_inertia_body": [0.1, 0.0, 0.0, 0.1, 0.0, 0.1],
        "estimated_payload_com_object": [0.0, 0.0, 0.0],
        "task_specs": [task.to_dict()],
        "environment_splits": [DatasetSplit.TRAIN.value],
        "assignment_templates_by_environment": [assignments],
        "object_id": "order8_object",
        "module_ids": list(modules),
        "local_joint_ids": list(local_joints),
        "command_local_joint_ids": list(command_joints),
        "rotor_global_ids": list(rotors),
        "vectoring_global_joint_ids": list(vectoring),
        "reward_term_names": ["weighted_energy"],
        "control_dt_s": 0.02,
        "raw_contact_actor_input": False,
        "runtime_phase_labels": [
            phase.value for phase in ORDER9_OBJECT_TASK_PHASES
        ],
        "actor_phase_labels": list(ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS),
        "actor_phase_index_by_runtime": list(
            ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
        ),
        "phase_duration_s": dict(
            Order9ObjectTaskRuntimeConfig().phase_duration_s
        ),
    }
    batch = 1
    module_count = len(modules)
    local_count = len(local_joints)
    command_count = len(command_joints)
    rotor_count = len(rotors)
    anchor_count = len(assignments)
    hidden = 8
    joint_action_width = 12

    def pose(shape):
        value = torch.zeros(shape)
        value[..., 6] = 1.0
        return value

    buffer = Order9TensorRolloutBuffer(metadata)
    for step in range(2):
        values = {
            "valid": torch.ones((batch,), dtype=torch.bool),
            "time_s": torch.tensor([0.02 * step]),
            "phase_index": torch.zeros((batch,), dtype=torch.long),
            "phase_progress": torch.tensor([0.1 * step]),
            "episode_serial": torch.zeros((batch,), dtype=torch.long),
            "step_index": torch.tensor([step], dtype=torch.long),
            "module_pose_world": pose((batch, module_count, 7)),
            "module_twist_world": torch.zeros((batch, module_count, 6)),
            "local_joint_positions_rad": torch.zeros(
                (batch, module_count, local_count)
            ),
            "local_joint_velocities_radps": torch.zeros(
                (batch, module_count, local_count)
            ),
            "robot_root_pose_world": pose((batch, 7)),
            "robot_root_twist_world": torch.zeros((batch, 6)),
            "object_pose_world": pose((batch, 7)),
            "object_twist_world": torch.zeros((batch, 6)),
            "desired_body_pose_world": pose((batch, 7)),
            "desired_body_twist_reference": torch.zeros((batch, 6)),
            "desired_object_pose_world": pose((batch, 7)),
            "phase_goal_body_pose_world": pose((batch, 7)),
            "phase_goal_object_pose_world": pose((batch, 7)),
            "desired_joint_positions_rad": torch.zeros(
                (batch, module_count, command_count)
            ),
            "desired_joint_velocities_radps": torch.zeros(
                (batch, module_count, command_count)
            ),
            "selected_assignment_mask": torch.ones(
                (batch, anchor_count), dtype=torch.bool
            ),
            "contact_schedule_index": torch.ones((batch,), dtype=torch.long),
            "actor_controller_qp_feasible": torch.ones(
                (batch,), dtype=torch.bool
            ),
            "actor_controller_status_one_hot": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
            "actor_allocation_residual_norm": torch.zeros((batch,)),
            "actor_task_success": torch.zeros((batch,), dtype=torch.bool),
            "global_action": torch.zeros(
                (batch, ORDER9_GLOBAL_ACTION_SIZE)
            ),
            "joint_action": torch.zeros(
                (batch, module_count, joint_action_width)
            ),
            "previous_global_action": torch.zeros(
                (batch, ORDER9_GLOBAL_ACTION_SIZE)
            ),
            "recurrent_state_in": torch.zeros((batch, hidden)),
            "recurrent_state_out": torch.zeros((batch, hidden)),
            "old_log_prob": torch.zeros((batch,)),
            "old_value": torch.full((batch,), 0.25),
            "privileged_disturbance_body": torch.zeros((batch, 6)),
            "command_body_pose_world": pose((batch, 7)),
            "command_body_twist": torch.zeros((batch, 6)),
            "command_residual_wrench_body": torch.zeros((batch, 6)),
            "command_joint_position_targets_rad": torch.zeros(
                (batch, module_count, command_count)
            ),
            "command_joint_velocity_targets_radps": torch.zeros(
                (batch, module_count, command_count)
            ),
            "command_joint_torque_bias_nm": torch.zeros(
                (batch, module_count, command_count)
            ),
            "controller_desired_wrench_body": torch.zeros((batch, 6)),
            "rotor_thrusts_n": torch.ones((batch, rotor_count)),
            "vectoring_joint_targets_rad": torch.zeros((batch, rotor_count)),
            "allocation_residual_norm": torch.zeros((batch,)),
            "qp_feasible": torch.ones((batch,), dtype=torch.bool),
            "rotor_saturation": torch.zeros(
                (batch, rotor_count), dtype=torch.bool
            ),
            "selected_contact_forces_world": torch.zeros(
                (batch, anchor_count, 3)
            ),
            "prohibited_collision": torch.zeros((batch,), dtype=torch.bool),
            "reward": torch.tensor([1.0]),
            "reward_terms": torch.tensor([[0.1]]),
            "phase_success": torch.zeros((batch,), dtype=torch.bool),
            "terminal": torch.zeros((batch,), dtype=torch.bool),
            "truncated": torch.tensor([step == 1]),
            "bootstrap_value": torch.tensor([0.5 if step == 1 else 0.0]),
            "post_robot_root_pose_world": pose((batch, 7)),
            "post_robot_root_twist_world": torch.zeros((batch, 6)),
            "post_local_joint_positions_rad": torch.zeros(
                (batch, module_count, local_count)
            ),
            "post_local_joint_velocities_radps": torch.zeros(
                (batch, module_count, local_count)
            ),
            "post_object_pose_world": pose((batch, 7)),
            "post_object_twist_world": torch.zeros((batch, 6)),
        }
        buffer.append(values)
    return buffer.finalize()


def test_tensor_rollout_roundtrip_and_existing_record_conversion(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / "raw.pt"

    digest = write_order9_tensor_rollout_artifact(path, artifact)
    loaded = load_order9_tensor_rollout_artifact(path, expected_sha256=digest)
    records = order9_pi_l_records_from_tensor_artifact(loaded)

    assert loaded.environment_step_count == 2
    assert len(records) == 2
    assert records[0].runtime_observation.contact_states == []
    assert records[0].behavior_trace is not None
    assert records[0].behavior_trace.policy_checkpoint_sha256 == "1" * 64
    assert records[1].truncated
    assert records[1].bootstrap_value == pytest.approx(0.5)
    assert records[1].episode_id == records[0].episode_id
    assert records[1].step_index == 1


def test_tensor_rollout_rejects_nonfinal_episode_boundary() -> None:
    artifact = _artifact()
    tensors = {name: value.clone() for name, value in artifact.tensors.items()}
    tensors["terminal"][0, 0] = True
    invalid = Order9TensorRolloutArtifact(metadata=artifact.metadata, tensors=tensors)

    with pytest.raises(SchemaValidationError, match="boundary is not final"):
        invalid.validate()


def test_tensor_rollout_detects_byte_tampering(tmp_path: Path) -> None:
    path = tmp_path / "raw.pt"
    digest = write_order9_tensor_rollout_artifact(path, _artifact())
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(SchemaValidationError, match="SHA-256 mismatch"):
        load_order9_tensor_rollout_artifact(path, expected_sha256=digest)


def test_fixed_topology_uses_task_disjoint_not_structural_disjoint_splits(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    raw_path = tmp_path / "raw.pt"
    write_order9_tensor_rollout_artifact(raw_path, artifact)
    source = order9_pi_l_records_from_tensor_artifact(artifact)[0]
    train = copy.deepcopy(source)
    validation = copy.deepcopy(source)
    validation.task_id = "order9-raw-task-validation"
    validation.split = DatasetSplit.VALIDATION
    validation.episode_id = "validation-episode"
    validation.record_id = "validation-record"
    validation.trajectory_record_id = "validation-trajectory"
    train_task = TaskSpec.from_dict(artifact.metadata["task_specs"][0])
    validation_task = TaskSpec.from_dict(train_task.to_dict())
    validation_task.task_id = validation.task_id
    task_specs = {
        train_task.task_id: train_task,
        validation_task.task_id: validation_task,
    }
    common = dict(
        generation_id="fixed-topology-unit",
        low_level_records=(train, validation),
        task_specs=task_specs,
        behavior_checkpoint_sha256_by_family={"pi_l": "1" * 64},
        source_isaac_artifact_paths=(raw_path,),
        on_policy_environment_step_count=2,
        random_seeds=(1, 2),
        config_hash="2" * 64,
        robot_model_hash="3" * 64,
        urdf_hash="4" * 64,
        thrust_model_hash="5" * 64,
        simulator_version="unit",
        simulator_hash="6" * 64,
    )

    manifest = write_order9_on_policy_dataset(
        tmp_path / "fixed",
        metadata={"topology_randomized": False},
        **common,
    )

    assert manifest.metadata["fixed_topology_task_split"] is True
    assert manifest.metadata["structural_hash_disjoint_splits"] is False
    with pytest.raises(SchemaValidationError, match="structural hash crosses"):
        write_order9_on_policy_dataset(
            tmp_path / "randomized",
            metadata={"topology_randomized": True},
            **common,
        )


def test_tensor_artifact_merger_builds_one_namespaced_on_policy_generation(
    tmp_path: Path,
) -> None:
    config = load_order9_learning_config(
        "configs/training/order9_learning_curriculum.yaml"
    )
    stage = order9_stage_by_id(config, "c2_pi_l_ppo_fixed_conservative")
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    checkpoint = tmp_path / "pi_l.pt"
    checkpoint.write_bytes(b"unit checkpoint bytes")
    checkpoint_sha = hash_file(checkpoint)
    raw_paths = []
    for index, split in enumerate(
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION)
    ):
        source = _artifact()
        metadata = copy.deepcopy(source.metadata)
        task = TaskSpec.from_dict(metadata["task_specs"][0])
        task.task_id = f"merge-{split.value}-task"
        metadata.update(
            {
                "generation_id": "merge-generation",
                "pi_l_checkpoint_sha256": checkpoint_sha,
                "stage_id": stage.stage_id,
                "stage_config_hash": stable_hash(stage.to_dict()),
                "curriculum_schedule_hash": order9_schedule_hash(config),
                "config_hash": stable_hash(config.to_dict()),
                "physical_model_hash": physical.stable_hash(),
                "urdf_hash": hash_file(physical.urdf_path),
                "thrust_model_hash": physical.metadata["thrust_model_hash"],
                "robot_usd_sha256": str(index + 7) * 64,
                "simulator_version": "unit-isaac",
                "simulator_hash": "8" * 64,
                "random_seed": 100 + index,
                "topology_randomized": False,
                "task_specs": [task.to_dict()],
                "environment_splits": [split.value],
                "collector_version": "unit-collector",
                "setup_wall_elapsed_s": 2.0 + index,
                "rollout_wall_elapsed_s": 1.0,
                "runtime_load": {
                    "telemetry_version": "order9_runtime_load_v1",
                    "gpu_monitor_available": True,
                    "gpu_memory_used_mib_peak": 1000.0 + index,
                    "samples": [{"elapsed_s": 0.0}],
                },
            }
        )
        artifact = Order9TensorRolloutArtifact(
            metadata=metadata,
            tensors={name: value.clone() for name, value in source.tensors.items()},
        )
        raw_path = tmp_path / f"raw-{split.value}.pt"
        write_order9_tensor_rollout_artifact(raw_path, artifact)
        raw_paths.append(raw_path)

    output = tmp_path / "dataset"
    manifest = build_order9_pi_l_on_policy_dataset(
        output,
        raw_artifact_paths=raw_paths,
        generation_id="merge-generation",
        stage_id=stage.stage_id,
        pi_l_checkpoint_path=checkpoint,
        config=config,
        physical_model=physical,
    )
    bundle = load_order9_dataset(output)
    validation = validate_order9_dataset_for_stage(
        bundle,
        stage,
        behavior_checkpoint_sha256={"pi_l": checkpoint_sha},
    )

    assert manifest.record_counts["low_level_control"] == 4
    assert manifest.metadata["on_policy_environment_step_count"] == 4
    assert manifest.metadata["collection_runtime_complete"] is True
    assert manifest.metadata["aggregate_collection_env_steps_per_s"] == 2.0
    assert manifest.metadata["end_to_end_collection_env_steps_per_s"] == pytest.approx(
        4.0 / 7.0
    )
    assert all(
        "samples" not in value["runtime_load"]
        for value in manifest.metadata["source_collection_runtime"].values()
    )
    assert validation.valid, validation.failures
    assert len({record.episode_id for record in bundle.low_level_records}) == 2
    assert all(":shard:" in record.episode_id for record in bundle.low_level_records)
    with pytest.raises(FileExistsError, match="not empty"):
        build_order9_pi_l_on_policy_dataset(
            output,
            raw_artifact_paths=raw_paths,
            generation_id="merge-generation",
            stage_id=stage.stage_id,
            pi_l_checkpoint_path=checkpoint,
            config=config,
            physical_model=physical,
        )


def test_production_benchmark_is_derived_from_raw_artifact_timing(
    tmp_path: Path,
) -> None:
    source = _artifact()
    metadata = copy.deepcopy(source.metadata)
    metadata.update(
        {
            "environment_count": 1,
            "rollout_steps": 2,
            "rollout_wall_elapsed_s": 0.01,
            "aggregate_env_steps_per_s": 200.0,
            "terminal_count": 0,
            "successful_terminal_count": 0,
            "collector_version": ORDER9_PRODUCTION_COLLECTOR_VERSION,
            "actuator_readback": {"matches_physical_model": True},
            "object_mass_properties_readback": {"matches_task_spec": True},
        }
    )
    artifact = Order9TensorRolloutArtifact(
        metadata=metadata,
        tensors={name: value.clone() for name, value in source.tensors.items()},
    )
    raw = tmp_path / "benchmark.pt"
    write_order9_tensor_rollout_artifact(raw, artifact)
    config = Order9RuntimeBenchmarkConfig(
        environment_count_candidates=[1],
        initial_environment_count=1,
        minimum_aggregate_env_steps_per_s=100.0,
        warmup_steps=0,
        measurement_steps=2,
    )

    report = build_order9_production_benchmark_report(
        config,
        raw_artifact_paths=[raw],
        expected_stage_id="c2_pi_l_ppo_fixed_conservative",
        expected_checkpoint_sha256="1" * 64,
        order8_report_path=(
            "artifacts/p4_full/order8_natural_contact/"
            "order8_mu4p5_dt20ms_full_v406.json"
        ),
    )

    assert report.passed
    assert report.selected_environment_count == 1
    assert report.samples[0].aggregate_env_steps_per_s == pytest.approx(200.0)
    assert report.samples[0].metadata["production_collector"] is True
