from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    POLICY_COMMAND_CONTRACT_LEGACY,
    ControllerCommand,
    ControllerStatus,
    PolicyCommand,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffResult
from amsrr.training.order3_free_flight import Order3FreeFlightRewardConfig
from amsrr.training.order3_takeoff_collector import (
    Order3TakeoffBCCollectorConfig,
    collect_order3_takeoff_bc_transitions,
)


def _source(*, contact_wrench: bool = False):
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=5, module_count=2
    )
    builder = RigidBodyControlModelBuilder()
    times = [0.0, 0.005, 0.010]
    root_heights = [0.10, 0.35, 0.60]
    observations = []
    control_models = []
    for index, (time_s, root_height) in enumerate(zip(times, root_heights, strict=True)):
        module_states = [
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=(
                    module.pose_in_design_frame[0],
                    module.pose_in_design_frame[1],
                    module.pose_in_design_frame[2] + root_height,
                    *module.pose_in_design_frame[3:7],
                ),
                twist_world=[0.0] * 6,
            )
            for module in morphology.modules
        ]
        observation = RuntimeObservation(
            time_s=time_s,
            morphology_graph=morphology,
            module_states=module_states,
            object_states=[],
            contact_states=[],
            controller_status=ControllerStatus(
                status="ok", qp_feasible=True, active_mode="rigid_body_qp"
            ),
            task_progress=TaskProgressState(),
        )
        if index == 0 and contact_wrench:
            observation.contact_states = [
                {
                    "contact_id": "floor-contact",
                    "entity_a": morphology.graph_id,
                    "entity_b": "floor",
                    "active": True,
                    "wrench_world": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                    "metadata": {"source": "isaac_lab_contact_sensor"},
                }
            ]
            observation = RuntimeObservation.from_dict(observation.to_dict())
        observations.append(observation)
        control_models.append(builder.build(morphology, physical_model, observation))

    settled_pose = control_models[0].body_pose_world
    hover_target = control_models[-1].body_pose_world
    policy_commands = [
        PolicyCommand(control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL),
        PolicyCommand(
            desired_body_pose=control_models[1].body_pose_world,
            desired_body_twist=[0.0] * 6,
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        ),
        PolicyCommand(
            desired_body_pose=hover_target,
            desired_body_twist=[0.0] * 6,
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        ),
    ]
    controller_commands = [
        ControllerCommand(
            rotor_thrusts_n={} if index == 0 else {"module_0:rotor_1": 5.0},
            vectoring_joint_targets={},
            joint_torque_commands={},
            dock_mechanism_commands={},
            controller_status=ControllerStatus(
                status="ok",
                qp_feasible=True,
                active_mode="rigid_body_qp",
                metrics={"qp_primary_path": 1.0},
            ),
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        )
        for index in range(3)
    ]
    actuator_records = [
        IsaacActuatorTargetRecord(
            time_s=time_s,
            backend="isaac_lab",
            morphology_graph_id=morphology.graph_id,
            command_index=index,
            actuator_targets=[],
            qp_status="ok",
        )
        for index, time_s in enumerate(times)
    ]
    report = {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_returncode": 0,
        "random_morphology_takeoff_smoke": True,
        "random_morphology_takeoff_smoke_passed": True,
        "random_morphology_takeoff_control_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
        "random_morphology_takeoff_tracking_state_source": "true_morphology_centroidal_frame",
        "random_morphology_takeoff_true_centroidal_tracking": True,
        "random_morphology_takeoff_contact_wrench_tracking_claim": False,
        "random_morphology_takeoff_internal_wrench_tracking_claim": False,
        "random_morphology_takeoff_learned_policy_used": False,
        "random_morphology_takeoff_controller": "deterministic_qpid",
        "random_morphology_takeoff_physical_model_hash": physical_model.stable_hash(),
        "random_morphology_takeoff_sim_dt_s": 0.005,
        "random_morphology_takeoff_steps": 3,
        "random_morphology_takeoff_hover_height_delta_m": hover_target[2]
        - settled_pose[2],
        "random_morphology_takeoff_hover_hold_time_s": 0.005,
        "random_morphology_takeoff_hover_hold_required_s": 0.005,
        "random_morphology_takeoff_settled_pose_world": list(settled_pose),
        "random_morphology_takeoff_hover_target_pose_world": list(hover_target),
        "random_morphology_takeoff_phase_transitions": [
            {"time_s": 0.0, "to_phase": "settle"},
            {"time_s": 0.005, "to_phase": "takeoff_ramp"},
            {"time_s": 0.010, "to_phase": "hover_hold"},
        ],
        "random_morphology_takeoff_exact_cross_module_collision_passed": True,
        "random_morphology_takeoff_finite_state": True,
        "random_morphology_takeoff_runtime_observations": [
            observation.to_dict() for observation in observations
        ],
        "random_morphology_takeoff_policy_commands": [
            command.to_dict() for command in policy_commands
        ],
        "random_morphology_takeoff_controller_commands": [
            command.to_dict() for command in controller_commands
        ],
        "random_morphology_takeoff_actuator_target_records": [
            record.to_dict() for record in actuator_records
        ],
        "random_morphology_takeoff_control_pose_history": [
            list(control_models[min(index + 1, len(control_models) - 1)].body_pose_world)
            for index in range(len(control_models))
        ],
        "random_morphology_takeoff_artifacts": {
            "backend": "isaac_lab",
            "isaac_backed": True,
            "dry_run": False,
            "object_task_claim": False,
            "is_p4_full_completion": False,
        },
    }
    result = RandomMorphologyTakeoffResult(
        graph_id=morphology.graph_id,
        attempted=True,
        dry_run=False,
        isaac_backed=True,
        unit_contract_passed=True,
        real_isaac_passed=True,
        placement={},
        report=report,
    )
    config = Order3TakeoffBCCollectorConfig(
        recurrent_state_dim=8,
        reward_config=Order3FreeFlightRewardConfig(
            success_hold_duration_s=0.005,
        ),
    )
    return result, morphology, physical_model, config


def test_collects_aligned_v2_takeoff_as_zero_action_bc_transitions() -> None:
    source, morphology, physical_model, config = _source()
    structural_hash = morphology_structural_hash(morphology)

    collected = collect_order3_takeoff_bc_transitions(
        source,
        split=DatasetSplit.HELD_OUT,
        expected_structural_hash=structural_hash,
        physical_model=physical_model,
        config=config,
    )

    assert len(collected.transitions) == 3
    assert collected.structural_hash == structural_hash
    assert collected.split == DatasetSplit.HELD_OUT
    assert collected.source_is_real_isaac is True
    assert collected.learned_action_trace_available is False
    assert collected.online_ppo_rollout_eligible is False
    assert collected.transitions[0].policy_applied is False
    assert collected.transitions[1].policy_applied is True
    assert collected.transitions[0].metrics["settle_phase"] == 1.0
    assert collected.transitions[1].target_pose_world == tuple(
        source.report["random_morphology_takeoff_policy_commands"][1][
            "desired_body_pose"
        ]
    )
    assert all(transition.action == [0.0] * 12 for transition in collected.transitions)
    assert all(
        transition.previous_action == [0.0] * 12
        for transition in collected.transitions
    )
    assert all(
        transition.recurrent_state_in == [0.0] * 8
        for transition in collected.transitions
    )
    assert [transition.terminal for transition in collected.transitions] == [
        False,
        False,
        True,
    ]
    assert all(
        transition.metrics["isaac_backed"] == 1.0
        for transition in collected.transitions
    )
    assert all(
        transition.metrics["learned_action_trace_available"] == 0.0
        for transition in collected.transitions
    )
    assert collected.metadata["boundary"].startswith("bc_only")


def test_zero_action_teacher_accepts_explicit_neutral_joint_hold_only() -> None:
    source, _, physical_model, config = _source()
    for command in source.report["random_morphology_takeoff_policy_commands"]:
        command["joint_position_targets"] = {"module_0:pitch_dock_mech_joint1": 0.0}
        command["joint_velocity_targets"] = {"module_0:pitch_dock_mech_joint1": 0.0}
        command["joint_torque_bias"] = {"module_0:pitch_dock_mech_joint1": 0.0}

    collected = collect_order3_takeoff_bc_transitions(
        source,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        config=config,
    )

    assert all(transition.action == [0.0] * 12 for transition in collected.transitions)

    source.report["random_morphology_takeoff_policy_commands"][0][
        "joint_position_targets"
    ]["module_0:pitch_dock_mech_joint1"] = 0.01
    with pytest.raises(SchemaValidationError, match="unsupported residual field"):
        collect_order3_takeoff_bc_transitions(
            source,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_explicitly_sanitizes_source_contact_wrench_without_mutating_report() -> None:
    source, morphology, physical_model, config = _source(contact_wrench=True)
    raw_wrench = source.report["random_morphology_takeoff_runtime_observations"][0][
        "contact_states"
    ][0]["wrench_world"]

    collected = collect_order3_takeoff_bc_transitions(
        source,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        config=config,
    )

    assert raw_wrench == [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    assert collected.transitions[0].runtime_observation.contact_states[0].active is True
    assert (
        collected.transitions[0].runtime_observation.contact_states[0].wrench_world
        is None
    )
    assert collected.removed_privileged_contact_wrench_count == 1
    assert collected.metadata["sanitization_enabled"] is True
    assert collected.metadata["removed_privileged_contact_wrench_count"] == 1
    assert (
        collected.transitions[0].metrics[
            "removed_privileged_contact_wrench_count"
        ]
        == 1.0
    )

    strict = replace(config, sanitize_privileged_contact_wrench=False)
    with pytest.raises(SchemaValidationError, match="privileged contact wrench"):
        collect_order3_takeoff_bc_transitions(
            source,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=strict,
        )


def test_rejects_privileged_contact_metadata_even_when_sanitization_is_enabled() -> None:
    source, _, physical_model, config = _source()
    raw = source.report["random_morphology_takeoff_runtime_observations"][0]
    raw["contact_states"] = [
        {
            "contact_id": "floor-contact",
            "entity_a": source.graph_id,
            "entity_b": "floor",
            "active": True,
            "wrench_world": None,
            "metadata": {"privileged_contact_wrench": [0.0] * 6},
        }
    ]

    with pytest.raises(SchemaValidationError, match="privileged field"):
        collect_order3_takeoff_bc_transitions(
            source,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_rejects_legacy_contract_and_missing_alignment() -> None:
    source, _, physical_model, config = _source()
    legacy = copy.deepcopy(source)
    legacy.report[
        "random_morphology_takeoff_control_contract_version"
    ] = POLICY_COMMAND_CONTRACT_LEGACY
    with pytest.raises(SchemaValidationError, match="contract mismatch"):
        collect_order3_takeoff_bc_transitions(
            legacy,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )

    misaligned = copy.deepcopy(source)
    misaligned.report["random_morphology_takeoff_policy_commands"].pop()
    with pytest.raises(SchemaValidationError, match="lengths mismatch"):
        collect_order3_takeoff_bc_transitions(
            misaligned,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_rejects_non_isaac_provenance_and_control_pose_mismatch() -> None:
    source, _, physical_model, config = _source()
    dry = replace(source, attempted=False, dry_run=True, isaac_backed=False)
    with pytest.raises(SchemaValidationError, match="non-dry real-Isaac"):
        collect_order3_takeoff_bc_transitions(
            dry,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )

    mismatch = copy.deepcopy(source)
    mismatch.report["random_morphology_takeoff_control_pose_history"][0][0] += 0.1
    with pytest.raises(SchemaValidationError, match="true-centroidal pose"):
        collect_order3_takeoff_bc_transitions(
            mismatch,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_rejects_wrong_structural_split_and_learned_action_trace() -> None:
    source, _, physical_model, config = _source()
    with pytest.raises(SchemaValidationError, match="structural hash"):
        collect_order3_takeoff_bc_transitions(
            source,
            split=DatasetSplit.VALIDATION,
            expected_structural_hash="not-the-assigned-morphology",
            physical_model=physical_model,
            config=config,
        )

    learned = copy.deepcopy(source)
    learned.report["random_morphology_takeoff_learned_action_trace"] = [[0.0] * 12]
    with pytest.raises(SchemaValidationError, match="learned action trace"):
        collect_order3_takeoff_bc_transitions(
            learned,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )
