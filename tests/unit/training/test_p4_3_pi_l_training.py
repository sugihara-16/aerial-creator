from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from amsrr.policies.learned_low_level_policy import (
    PI_L_FEATURE_NAMES,
    PI_L_OUTPUT_MODE,
    PI_L_TARGET_NAMES,
    LearnedLowLevelPolicy,
)
from amsrr.policies.low_level_policy_base import BaselineLowLevelPolicy, LowLevelPolicyContext
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    LowLevelControlRecord,
    P4_3DatasetManifest,
    StageDecisionMasks,
)
from amsrr.schemas.morphology import ModuleNode, MorphologyGraph
from amsrr.schemas.physical_model import ModuleCapabilityToken, PhysicalModel
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
    ObjectTarget,
    PolicyCommand,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.p4_3_pi_l_training import (
    P4_3PiLTrainingConfig,
    load_low_level_control_records,
    load_p4_3_pi_l_training_config,
    low_level_record_target_delta,
    train_p4_3_pi_l,
)


def test_pi_l_config_matches_p4_3_bootstrap_defaults() -> None:
    config = load_p4_3_pi_l_training_config()
    assert config.epochs == 20
    assert config.learning_rate == 0.001
    assert config.hidden_dim == 64
    assert config.batch_size == 256
    assert config.seed == 11
    assert config.output_mode == PI_L_OUTPUT_MODE
    assert config.checkpoint_dir == "artifacts/p4_3/pi_l"


def test_pi_l_training_writes_bounded_checkpoint_and_offline_isaac_returns(
    tmp_path: Path,
) -> None:
    records = [
        *[_record(DatasetSplit.TRAIN, "task-train", index) for index in range(4)],
        *[_record(DatasetSplit.VALIDATION, "task-validation", index) for index in range(2)],
        *[_record(DatasetSplit.HELD_OUT, "task-held", index) for index in range(2)],
    ]
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    for split in DatasetSplit:
        _write_records(
            dataset_path / f"low_level_control_{split.value}.jsonl",
            [record for record in records if record.split == split],
        )
    _write_manifest(dataset_path, records)
    output_dir = tmp_path / "pi_l"

    manifest = train_p4_3_pi_l(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=P4_3PiLTrainingConfig(
            epochs=3,
            learning_rate=0.01,
            hidden_dim=8,
            batch_size=2,
            seed=11,
        ),
    )

    assert Path(manifest.checkpoint_path).exists()
    assert Path(manifest.metrics_path).exists()
    assert Path(manifest.loss_curve_path).exists()
    assert Path(manifest.reward_curve_path).exists()
    assert Path(manifest.rollout_evaluation_path).name == "rollout_evaluation.json"
    assert Path(manifest.rollout_evaluation_path).exists()
    assert Path(manifest.fallback_metadata_path).name == "fallback_metadata.json"
    assert Path(manifest.fallback_metadata_path).exists()
    metrics = json.loads(Path(manifest.metrics_path).read_text(encoding="utf-8"))
    assert metrics["evaluation_mode"] == "offline_dataset_evaluation"
    assert metrics["online_rollout_evaluation"] is False
    assert metrics["reward_curve_semantics"] == "offline_episode_returns_not_online_training_curve"
    assert metrics["source_is_real_isaac"] is True
    assert metrics["source_evidence"] == "p4_3_dataset_manifest_all_episodes_real_isaac"
    assert metrics["train_task_ids"] == ["task-train"]
    assert metrics["validation_task_ids"] == ["task-validation"]
    assert metrics["held_out_task_ids"] == ["task-held"]
    assert metrics["deterministic_fallback_available"] is True
    assert metrics["controller_authority"] == "controller_qp_safety_layer_only"

    checkpoint = torch.load(manifest.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["feature_names"] == list(PI_L_FEATURE_NAMES)
    assert checkpoint["target_names"] == list(PI_L_TARGET_NAMES)
    assert checkpoint["output_mode"] == PI_L_OUTPUT_MODE
    assert checkpoint["controller_command_output"] is False
    assert checkpoint["actuator_target_output"] is False
    assert checkpoint["deterministic_fallback"]["path"].endswith("BaselineLowLevelPolicy")
    assert len(checkpoint["output_lower_bounds"]) == len(PI_L_TARGET_NAMES)
    assert len(checkpoint["output_upper_bounds"]) == len(PI_L_TARGET_NAMES)
    rollout_evaluation = json.loads(
        Path(manifest.rollout_evaluation_path).read_text(encoding="utf-8")
    )
    assert rollout_evaluation["evaluation_mode"] == "offline_dataset_evaluation"
    assert rollout_evaluation["online_rollout_executed"] is False
    assert rollout_evaluation["learned_policy_deployed_in_isaac"] is False
    fallback_metadata = json.loads(
        Path(manifest.fallback_metadata_path).read_text(encoding="utf-8")
    )
    assert fallback_metadata["fallback_available"] is True
    assert fallback_metadata["controller_command_output"] is False
    assert fallback_metadata["actuator_target_output"] is False

    reward_rows = list(csv.DictReader(Path(manifest.reward_curve_path).open(encoding="utf-8")))
    assert reward_rows
    assert {row["evaluation_mode"] for row in reward_rows} == {
        "offline_dataset_evaluation"
    }
    assert {row["source_is_real_isaac"] for row in reward_rows} == {"1"}
    expected_total_reward = sum(float(record.reward or 0.0) for record in records)
    assert sum(float(row["offline_episode_return"]) for row in reward_rows) == pytest.approx(
        expected_total_reward
    )

    learned_policy = LearnedLowLevelPolicy.from_checkpoint(manifest.checkpoint_path)
    command = learned_policy.command(_context(time_s=0.0))
    assert isinstance(command, PolicyCommand)
    assert learned_policy.last_diagnostics.fallback_reason is None

    target = low_level_record_target_delta(records[0])
    assert target[:6] == pytest.approx([0.1] * 6)
    assert target[6:9] == pytest.approx([0.02] * 3)
    assert target[9:12] == pytest.approx([0.2] * 3)
    assert target[12:] == pytest.approx([0.02] * 3)


def test_pi_l_loader_rejects_task_leakage_between_splits(tmp_path: Path) -> None:
    dataset_path = tmp_path / "leaked.jsonl"
    _write_records(
        dataset_path,
        [
            _record(DatasetSplit.TRAIN, "shared-task", 0),
            _record(DatasetSplit.VALIDATION, "shared-task", 1),
        ],
    )
    with pytest.raises(ValueError, match="task splits must be disjoint"):
        load_low_level_control_records(dataset_path)


def test_pi_l_missing_teacher_residual_cancels_deterministic_baseline() -> None:
    record = _record(DatasetSplit.TRAIN, "task-train", 0)
    context = _context(time_s=record.time_s)
    baseline = BaselineLowLevelPolicy().command(context)
    assert baseline.residual_wrench_body is not None
    teacher = record.policy_command.to_dict()
    teacher["residual_wrench_body"] = None
    record.policy_command = PolicyCommand.from_dict(teacher)

    target = low_level_record_target_delta(record)

    assert target[9:] == pytest.approx(
        [-value for value in baseline.residual_wrench_body]
    )


def _record(split: DatasetSplit, task_id: str, index: int) -> LowLevelControlRecord:
    context = _context(time_s=float(index) * 0.05)
    baseline = BaselineLowLevelPolicy().command(context)
    teacher = baseline.to_dict()
    assert baseline.desired_body_twist is not None
    assert baseline.desired_body_pose is not None
    assert baseline.residual_wrench_body is not None
    teacher["desired_body_twist"] = [value + 0.1 for value in baseline.desired_body_twist]
    pose = list(baseline.desired_body_pose)
    pose[:3] = [value + 0.02 for value in pose[:3]]
    teacher["desired_body_pose"] = pose
    teacher["residual_wrench_body"] = [
        value + (0.2 if axis < 3 else 0.02)
        for axis, value in enumerate(baseline.residual_wrench_body)
    ]
    status = context.runtime_observation.controller_status
    return LowLevelControlRecord(
        record_id=f"{split.value}-{index}",
        episode_id=f"{split.value}-episode-{index // 2}",
        task_id=task_id,
        split=split,
        step_index=index % 2,
        time_s=context.runtime_observation.time_s,
        trajectory_record_id=f"trajectory-{split.value}",
        active_trajectory_index=0,
        active_knot_index=0,
        runtime_observation=context.runtime_observation,
        active_knot=context.active_knot,
        policy_command=PolicyCommand.from_dict(teacher),
        controller_command=ControllerCommand(
            rotor_thrusts_n={"rotor-0": 1.0},
            vectoring_joint_targets={},
            joint_torque_commands={},
            dock_mechanism_commands={},
            controller_status=status,
        ),
        actuator_target_record={"isaac_backed": True},
        reward_terms={"r_tracking": 1.0 + index},
        reward=1.0 + index,
        terminal=index % 2 == 1,
        stage_masks=StageDecisionMasks(low_level_control_mask=True),
    )


def _write_records(path: Path, records: list[LowLevelControlRecord]) -> None:
    path.write_text("".join(f"{record.to_json()}\n" for record in records), encoding="utf-8")


def _write_manifest(path: Path, records: list[LowLevelControlRecord]) -> None:
    task_ids = {split: sorted({record.task_id for record in records if record.split == split}) for split in DatasetSplit}
    shards = [
        DatasetShard(
            dataset_kind=DatasetKind.LOW_LEVEL_CONTROL,
            split=split,
            path=str(path / f"low_level_control_{split.value}.jsonl"),
            record_count=sum(1 for record in records if record.split == split),
            sha256=f"unit-{split.value}",
        )
        for split in DatasetSplit
    ]
    manifest = P4_3DatasetManifest(
        dataset_id="pi-l-unit-dataset",
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=["unit-source.jsonl"],
        source_episode_ids=sorted({record.episode_id for record in records}),
        train_task_ids=task_ids[DatasetSplit.TRAIN],
        validation_task_ids=task_ids[DatasetSplit.VALIDATION],
        held_out_task_ids=task_ids[DatasetSplit.HELD_OUT],
        shards=shards,
        record_counts={DatasetKind.LOW_LEVEL_CONTROL.value: len(records)},
        source_hash="unit-source-hash",
        config_hash="unit-config-hash",
        robot_model_hash="unit-robot-hash",
        urdf_hash="unit-urdf-hash",
        thrust_model_hash="unit-thrust-hash",
        task_hashes={record.task_id: f"hash-{record.task_id}" for record in records},
        geometry_hashes={"unit-geometry": "unit-geometry-hash"},
        random_seeds=[11],
        simulator_version="isaac_lab",
        simulator_hash="unit-simulator-hash",
        metadata={
            "source_episode_count": len({record.episode_id for record in records}),
            "isaac_backed_episode_count": len({record.episode_id for record in records}),
        },
    )
    (path / "manifest.json").write_text(manifest.to_json(indent=2) + "\n", encoding="utf-8")


def _context(*, time_s: float) -> LowLevelPolicyContext:
    status = ControllerStatus(
        status="ok",
        qp_feasible=True,
        metrics={"allocation_residual_norm": 0.05},
    )
    capability = ModuleCapabilityToken(
        module_type="holon",
        aggregate_mass_norm=1.0,
        aggregate_inertia_features=[0.0] * 6,
        rotor_count=4,
        port_count=4,
        thrust_min_features=[0.0] * 4,
        thrust_max_features=[1.0] * 4,
        thrust_to_weight_ratio_est=2.0,
        dock_port_type_counts=[2, 2],
        has_vectoring=True,
        has_dock_mechanism=True,
    )
    morphology = MorphologyGraph(
        graph_id="morphology-1",
        modules=[
            ModuleNode(
                module_id=0,
                module_type="holon",
                pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base",
                is_base=True,
                capability_token=capability,
            )
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    observation = RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=0,
                pose_world=(0.01 * time_s, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        object_states=[
            ObjectRuntimeState(
                object_id="box",
                pose_world=(0.5 + 0.01 * time_s, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=status,
        task_progress=TaskProgressState(progress_ratio=0.1 * time_s),
    )
    knot = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=(0.1, 0.0, 1.0),
            com_vel_world=(0.1, 0.0, 0.0),
            body_orientation_world=(0.0, 0.0, 0.0, 1.0),
        ),
        object_targets=[
            ObjectTarget(
                object_id="box",
                pose_target_world=(0.7, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_target_world=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        ],
    )
    return LowLevelPolicyContext(
        runtime_observation=observation,
        morphology_graph=morphology,
        physical_model=PhysicalModel(
            model_id="physical-model-1",
            urdf_path="module_urdf/holon.urdf",
            links=[],
            joints=[],
            rotors=[],
            dock_ports=[],
            collision_primitives=[],
            aggregate_mass_kg=1.0,
            aggregate_inertia_body=[0.0] * 6,
        ),
        contact_wrench_trajectory=ContactWrenchTrajectory(
            horizon_s=1.0,
            dt_s=0.01,
            knots=[knot],
        ),
        active_knot=knot,
        controller_status=status,
    )
