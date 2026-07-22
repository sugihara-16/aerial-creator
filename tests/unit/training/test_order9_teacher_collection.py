from __future__ import annotations

from pathlib import Path

from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph, ModuleNode, RobotAnchor
from amsrr.schemas.physical_model import ModuleCapabilityToken
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ContactAssignment,
    ContactWrenchTrajectory,
    CentroidalTarget,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.policies.order9_low_level_policy import Order9LowLevelPolicyConfig
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_dataset import (
    iter_order9_low_level_records,
    load_order9_dataset,
    load_order9_dataset_index,
    validate_order9_pi_l_dataset_for_stage_streaming,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_curriculum import ORDER9_C0_COLLECTION_PROFILE_VERSION
from amsrr.training.order9_pipeline import order9_stage_by_id
from amsrr.training.order9_offline_training import train_order9_behavior_cloning
from amsrr.training.order9_evaluation import (
    order9_teacher_evaluation_episodes,
    write_order9_evaluation_episodes_jsonl,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.training.order9_teacher_collection import (
    Order9TeacherCollectionConfig,
    Order9TeacherEpisodeCollector,
    build_order9_teacher_dataset,
    load_order9_teacher_episode,
    load_order9_teacher_episode_manifest,
    write_order9_teacher_episode,
)


def test_teacher_collector_keeps_privileged_metrics_out_of_actor_and_builds_gzip_dataset(
    tmp_path: Path,
) -> None:
    paths = []
    for index, split in enumerate((DatasetSplit.TRAIN, DatasetSplit.VALIDATION)):
        result = _episode(f"task-{index}", f"episode-{index}", split)
        path = write_order9_teacher_episode(
            result,
            tmp_path / f"episode-{index}",
            random_seed=index,
            robot_model_hash="robot-hash",
            urdf_hash="urdf-hash",
            thrust_model_hash="thrust-hash",
            config_hash="config-hash",
            simulator_version="isaac-test",
            simulator_hash="simulator-hash",
            metadata={
                "c0_collection_profile_version": (
                    ORDER9_C0_COLLECTION_PROFILE_VERSION
                ),
                "c0_condition_id": f"unit-condition-{index}",
                "teacher_low_level_stride": 1,
                "teacher_high_level_stride": 1,
            },
        )
        manifest, low, high = load_order9_teacher_episode(path)
        manifest_path, manifest_only = load_order9_teacher_episode_manifest(path)
        assert manifest_path == path
        assert manifest_only.to_dict() == manifest.to_dict()
        assert manifest.success is True
        assert all(not record.runtime_observation.contact_states for record in low)
        assert all(
            "grasp_maintenance" not in record.runtime_observation.task_progress.metrics
            for record in low
        )
        assert low[-1].terminal is True
        assert high[-1].terminal is True
        paths.append(path)

    manifest = build_order9_teacher_dataset(paths, tmp_path / "dataset")
    bundle = load_order9_dataset(tmp_path / "dataset")
    index = load_order9_dataset_index(tmp_path / "dataset")

    assert manifest.metadata["gzip_shards"] is True
    assert len(bundle.low_level_records) == 6
    assert len(list(iter_order9_low_level_records(index))) == 6
    assert len(bundle.trajectory_records) == 6
    assert bundle.manifest.train_task_ids == ["task-0"]
    assert bundle.manifest.validation_task_ids == ["task-1"]

    config = load_order9_learning_config()
    stage = order9_stage_by_id(config, "c1_pi_l_bc_fixed_nominal")
    validation = validate_order9_pi_l_dataset_for_stage_streaming(index, stage)
    assert validation.valid is True
    assert validation.record_count == 6
    assert validation.metadata["streaming_validation"] is True

    evaluation = order9_teacher_evaluation_episodes(paths)
    evaluation_path = tmp_path / "evaluation" / "episodes.jsonl"
    write_order9_evaluation_episodes_jsonl(evaluation_path, evaluation)
    assert len(evaluation) == 2
    assert all(episode.task_success for episode in evaluation)
    assert all(episode.isaac_backed for episode in evaluation)
    assert all(episode.no_fallback_success for episode in evaluation)
    assert len(evaluation_path.read_text(encoding="utf-8").splitlines()) == 2

    config.optimization.pi_l_bc.epochs = 1
    config.optimization.pi_l_bc.batch_size = 2
    config.optimization.pi_l_bc.sequence_length = 2
    config.optimization.pi_l_bc.burn_in_steps = 1
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    trained = train_order9_behavior_cloning(
        config,
        stage_id=stage.stage_id,
        dataset_manifest_path=tmp_path / "dataset" / "manifest.json",
        physical_model=physical_model,
        output_dir=tmp_path / "training",
        git_revision="unit-test",
        device="cpu",
        model_config=Order9LowLevelPolicyConfig(
            graph_hidden_dim=16,
            graph_message_layers=1,
            recurrent_hidden_dim=16,
        ),
    )
    assert trained.training_record_count == 3
    assert trained.validation_record_count == 3
    assert Path(trained.checkpoint_path).is_file()


def _episode(task_id: str, episode_id: str, split: DatasetSplit):
    task = build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
        task_id=task_id,
    )
    morphology = _morphology()
    candidates = _candidate_set(task_id, morphology.graph_id)
    collector = Order9TeacherEpisodeCollector(
        task_spec=task,
        morphology_graph=morphology,
        contact_candidate_set=candidates,
        config=Order9TeacherCollectionConfig(
            episode_id=episode_id,
            split=split,
            high_level_stride=1,
            window_horizon_s=0.2,
            window_knot_dt_s=0.1,
        ),
    )
    phases = ("approach", "contact_acquisition", "lift", "complete")
    xs = (0.50, 0.55, 0.65, 0.70)
    for index, (phase, x) in enumerate(zip(phases, xs, strict=True)):
        collector.observe_state(
            actor_observation=_observation(morphology, index * 0.1, phase, x, privileged=False),
            reward_observation=_observation(morphology, index * 0.1, phase, x, privileged=True),
        )
        if index < len(phases) - 1:
            collector.record_command(
                trajectory=_legacy_trajectory(),
                centroidal_reference_pose_world=(
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                centroidal_reference_twist=[0.0] * 6,
                policy_command=PolicyCommand(
                    desired_body_pose=(
                        0.0,
                        0.0,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    desired_body_twist=[0.0] * 6,
                    residual_wrench_body=[0.0] * 6,
                    control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
                ),
                controller_command=ControllerCommand(
                    rotor_thrusts_n={"rotor": 2.0},
                    vectoring_joint_targets={},
                    joint_torque_commands={},
                    dock_mechanism_commands={},
                    controller_status=ControllerStatus(
                        status="ok",
                        qp_feasible=True,
                        metrics={"residual_norm": 0.0},
                    ),
                ),
                actuator_target_record={"actuator_targets": []},
                decision_dt_s=0.1,
            )
    return collector.finalize(
        success=True,
        failure_reason=None,
        release_valid=True,
        object_dropped=False,
        hard_collision=False,
        timeout=False,
        qp_infeasible_terminal=False,
    )


def _morphology() -> MorphologyGraph:
    capability = ModuleCapabilityToken(
        module_type="holon",
        aggregate_mass_norm=1.0,
        aggregate_inertia_features=[0.0] * 6,
        rotor_count=4,
        port_count=2,
        thrust_min_features=[0.0] * 4,
        thrust_max_features=[1.0] * 4,
        thrust_to_weight_ratio_est=2.0,
        dock_port_type_counts=[2, 0, 0],
        has_vectoring=True,
        has_dock_mechanism=True,
    )
    return MorphologyGraph(
        graph_id="teacher-collection-morphology",
        modules=[
            ModuleNode(
                module_id=index,
                module_type="holon",
                pose_in_design_frame=(
                    0.0,
                    float(index) * 0.4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                role_id="base" if index == 0 else "grasp",
                is_base=index == 0,
                capability_token=capability,
            )
            for index in range(3)
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[
            RobotAnchor(
                anchor_id=index,
                module_id=index,
                link_id=f"gripper-{index}",
                local_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                anchor_type="grasp",
                capability={"max_force_n": 30.0, "max_torque_nm": 5.0},
                associated_contact_slot_ids=[0],
            )
            for index in range(2)
        ],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )


def _candidate_set(task_id: str, graph_id: str) -> ContactCandidateSet:
    candidates = []
    for index, normal in enumerate(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))):
        candidates.append(
            ContactCandidate(
                candidate_id=index,
                slot_id=0,
                anchor_id=index,
                target_entity_id="order8_object",
                region_id=f"side-{index}",
                contact_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
                contact_frame_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
                normal_world=normal,
                tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                contact_mode=ContactMode.GRASP,
                friction=0.6,
                patch_area_m2=0.01,
                candidate_scores={},
                unary_valid=True,
            )
        )
    return ContactCandidateSet(
        set_id=f"{task_id}-candidates",
        task_id=task_id,
        morphology_graph_id=graph_id,
        candidates=candidates,
        candidate_mask=[True, True],
        slot_coverage={0: [0, 1]},
        pairwise_conflict_matrix=[[False, False], [False, False]],
        pairwise_compatibility_score=[[1.0, 1.0], [1.0, 1.0]],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="teacher-collection-test-v1",
    )


def _legacy_trajectory() -> ContactWrenchTrajectory:
    return ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.1,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                centroidal_target=CentroidalTarget(
                    com_pos_world=(0.0, 0.0, 0.5),
                    com_vel_world=(0.0, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                contact_assignments=[
                    ContactAssignment(
                        slot_id=0,
                        anchor_id=index,
                        candidate_id=index,
                        contact_mode=ContactMode.GRASP,
                        schedule_state="maintain",
                        wrench_target=[-5.0 if index == 0 else 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    )
                    for index in range(2)
                ],
            )
        ],
        derived_mode_label="order8-rolling-test",
    )


def _observation(
    morphology: MorphologyGraph,
    time_s: float,
    phase: str,
    x: float,
    *,
    privileged: bool,
) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=index,
                pose_world=(0.0, float(index) * 0.4, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
            for index in range(3)
        ],
        object_states=[
            ObjectRuntimeState(
                object_id="order8_object",
                pose_world=(x, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label=phase,
            progress_ratio=min(1.0, time_s / 0.3),
            success=phase == "complete",
            metrics=(
                {
                    "grasp_data_available": 1.0,
                    "grasp_maintenance": 1.0,
                    "slip_data_available": 1.0,
                    "slip_speed_mps": 0.0,
                    "collision_data_available": 1.0,
                    "hard_collision": 0.0,
                }
                if privileged
                else {}
            ),
        ),
    )
