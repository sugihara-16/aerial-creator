from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_ACTION_SIZE,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_DATASET_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3PolicyCheckpointMetadata,
    Order3PolicyTransition,
)
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution


def _transition() -> Order3PolicyTransition:
    model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(model).sample(seed=7, module_count=2)
    states = [
        ModuleRuntimeState(
            module_id=module.module_id,
            pose_world=module.pose_in_design_frame,
            twist_world=[0.0] * 6,
        )
        for module in morphology.modules
    ]
    observation = RuntimeObservation(
        time_s=0.25,
        morphology_graph=morphology,
        module_states=states,
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
    return Order3PolicyTransition(
        episode_id="episode-0",
        split=DatasetSplit.TRAIN,
        graph_id=morphology.graph_id,
        structural_hash="structural-hash",
        step_index=0,
        time_s=observation.time_s,
        runtime_observation=observation,
        target_pose_world=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        target_twist=[0.0] * 6,
        previous_action=[0.0] * ORDER3_ACTION_SIZE,
        action=[0.0] * ORDER3_ACTION_SIZE,
        recurrent_state_in=[0.0] * 8,
        old_log_prob=0.0,
        old_value=0.0,
        reward=0.0,
        terminal=False,
        policy_applied=True,
    )


def test_order3_transition_round_trip_preserves_v2_contract() -> None:
    transition = _transition()

    restored = Order3PolicyTransition.from_json(transition.to_json())

    assert restored.to_dict() == transition.to_dict()
    assert restored.dataset_version == ORDER3_DATASET_VERSION
    assert restored.policy_contract_version == "centroidal_local_joint_v2"


def test_order3_transition_rejects_unbounded_actions() -> None:
    transition = _transition()
    action = list(transition.action)
    action[0] = 1.01

    with pytest.raises(SchemaValidationError, match="normalized"):
        replace(transition, action=action)


def test_order3_transition_rejects_graph_identity_mismatch() -> None:
    transition = _transition()

    with pytest.raises(SchemaValidationError, match="graph_id"):
        replace(transition, graph_id="different")


def _checkpoint_metadata() -> Order3PolicyCheckpointMetadata:
    return Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version="centroidal_local_joint_v2",
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        training_stage="bc",
        action_names=list(ORDER3_ACTION_NAMES),
        actor_feature_schema_hash="actor-features",
        graph_feature_schema_hash="graph-features",
        config_hash="config",
        pool_hash="pool",
        dataset_hash="dataset",
        physical_model_hash="physical-model",
        urdf_hash="urdf",
        controller_contract_hash="controller",
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash="fallback",
        seed=1,
        git_revision="revision",
    )


def test_order3_checkpoint_metadata_round_trip_and_authority_boundary() -> None:
    metadata = _checkpoint_metadata()

    restored = Order3PolicyCheckpointMetadata.from_json(metadata.to_json())

    assert restored.to_dict() == metadata.to_dict()
    with pytest.raises(SchemaValidationError, match="authority"):
        replace(metadata, outputs_vectoring_joint_targets=True)
    with pytest.raises(SchemaValidationError, match="parent_bc"):
        replace(metadata, training_stage="ppo")
