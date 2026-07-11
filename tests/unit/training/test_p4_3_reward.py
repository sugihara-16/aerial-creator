from __future__ import annotations

import math

import pytest

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import (
    ContactState,
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology
from amsrr.training.p4_3_reward import (
    P4_3RewardConfig,
    compute_p4_3_reward_records,
    compute_p4_3_step_reward,
    compute_p4_3_terminal_reward,
)


@pytest.fixture(scope="module")
def morphology():
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return build_fixed_morphology(physical_model, module_count=2, module_spacing_m=0.45)


def test_p4_3_step_reward_contains_all_section_22_4_terms(grasp_carry_dict, morphology) -> None:
    task_spec = TaskSpec.from_dict(grasp_carry_dict)
    previous = _observation(task_spec, morphology, time_s=0.0, object_x=1.0)
    current = _observation(
        task_spec,
        morphology,
        time_s=0.1,
        object_x=1.1,
        contacts=[
            ContactState(
                contact_id="grasp:0",
                entity_a="anchor:0",
                entity_b="box_01",
                active=True,
                metadata={"slip_speed_mps": 0.05},
            )
        ],
        progress_metrics={"collision_data_available": 1.0, "hard_collision": 0.25},
    )
    command = ControllerCommand(
        rotor_thrusts_n={"module_0:thrust_1": 10.0},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=ControllerStatus(
            status="warning",
            qp_feasible=True,
            metrics={"allocation_residual_norm": 5.0, "rotor_saturation_ratio": 0.25},
        ),
    )
    actuator_record = {
        "actuator_targets": [
            {"actuator_type": "rotor_thrust", "target_value": 10.0, "clipped": True},
            {"actuator_type": "rotor_thrust", "target_value": 10.0, "clipped": False},
        ],
        "metrics": {"clipped_target_count": 1.0, "actuator_target_count": 2.0},
    }
    config = P4_3RewardConfig(progress_scale_m=0.1, qp_residual_scale=10.0, slip_speed_scale_mps=0.1)

    reward = compute_p4_3_step_reward(
        task_spec=task_spec,
        previous_observation=previous,
        observation=current,
        controller_command=command,
        actuator_target_record=actuator_record,
        config=config,
    )

    assert reward["r_object_goal_progress"] == pytest.approx(1.0)
    assert 0.0 < reward["r_object_pose_accuracy"] < 1.0
    assert reward["r_grasp_maintenance"] == 1.0
    assert reward["r_centroidal_stability"] == 1.0
    assert reward["r_energy"] == pytest.approx(0.25)
    assert reward["r_qp_residual"] == pytest.approx(0.5)
    assert reward["r_slip"] == pytest.approx(0.5)
    assert reward["r_collision"] == pytest.approx(0.25)
    assert reward["r_actuator_saturation"] == pytest.approx(0.5)
    assert reward["grasp_data_available"] == 1.0
    assert reward["slip_data_available"] == 1.0
    assert reward["energy_is_command_effort_proxy"] == 1.0
    expected = sum(
        reward[key]
        for key in (
            "weighted_object_goal_progress",
            "weighted_object_pose_accuracy",
            "weighted_grasp_maintenance",
            "weighted_centroidal_stability",
            "weighted_energy_penalty",
            "weighted_qp_residual_penalty",
            "weighted_slip_penalty",
            "weighted_collision_penalty",
            "weighted_actuator_saturation_penalty",
        )
    )
    assert reward["per_step_reward"] == pytest.approx(expected)
    assert all(isinstance(value, float) and math.isfinite(value) for value in reward.values())


def test_p4_3_missing_p4_2_contact_and_slip_data_are_explicit_neutral(
    grasp_carry_dict, morphology
) -> None:
    task_spec = TaskSpec.from_dict(grasp_carry_dict)
    observation = _observation(
        task_spec,
        morphology,
        time_s=0.0,
        object_x=0.8,
        contacts=[],
        progress_metrics={
            "anchor_object_distance_m": 0.9,
            "selected_assignment_feasible": 1.0,
            "transport_displacement_m": 0.0,
        },
    )
    before = observation.to_dict()

    reward = compute_p4_3_step_reward(task_spec=task_spec, observation=observation)

    assert reward["r_grasp_maintenance"] == 0.0
    assert reward["grasp_data_available"] == 0.0
    assert reward["contact_data_available"] == 0.0
    assert reward["missing_contact_data"] == 1.0
    assert reward["r_slip"] == 0.0
    assert reward["slip_data_available"] == 0.0
    assert reward["missing_slip_data"] == 1.0
    assert reward["r_energy"] == 0.0
    assert reward["energy_data_available"] == 0.0
    assert reward["r_qp_residual"] == 0.0
    assert reward["qp_residual_data_available"] == 0.0
    assert observation.to_dict() == before
    assert not any("rotor_thrust" in key or "actuator_target" in key for key in reward)


def test_p4_3_terminal_success_requires_pose_and_valid_release(grasp_carry_dict, morphology) -> None:
    task_spec = TaskSpec.from_dict(grasp_carry_dict)
    at_goal = _observation(task_spec, morphology, time_s=1.0, object_x=2.0)

    missing_release = compute_p4_3_terminal_reward(task_spec=task_spec, observation=at_goal)
    success = compute_p4_3_terminal_reward(
        task_spec=task_spec,
        observation=at_goal,
        release_valid=True,
        config=P4_3RewardConfig(success_bonus=7.0),
    )
    failed = compute_p4_3_terminal_reward(
        task_spec=task_spec,
        observation=at_goal,
        release_valid=True,
        object_dropped=True,
        hard_collision=True,
        config=P4_3RewardConfig(failure_penalty=4.0),
    )

    assert missing_release["terminal_goal_pose_within_tolerance"] == 1.0
    assert missing_release["terminal_release_data_available"] == 0.0
    assert missing_release["terminal_success"] == 0.0
    assert missing_release["terminal_reward"] == 0.0
    assert success["terminal_success"] == 1.0
    assert success["terminal_reward"] == 7.0
    assert failed["terminal_success"] == 0.0
    assert failed["terminal_failure"] == 1.0
    assert failed["terminal_reward"] == -4.0


def test_p4_3_sequence_rewards_align_to_observations_without_filling_commands(
    grasp_carry_dict, morphology
) -> None:
    task_spec = TaskSpec.from_dict(grasp_carry_dict)
    observations = [
        _observation(task_spec, morphology, time_s=0.0, object_x=1.0),
        _observation(task_spec, morphology, time_s=0.1, object_x=2.0),
    ]
    command = ControllerCommand(
        rotor_thrusts_n={"module_0:thrust_1": 5.0},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
    )

    records = compute_p4_3_reward_records(
        task_spec=task_spec,
        runtime_observations=observations,
        controller_commands=[command],
        actuator_target_records=[],
        release_valid=True,
    )

    assert len(records) == len(observations)
    assert [record["step_index"] for record in records] == [0.0, 1.0]
    assert records[0]["pre_observation_index"] == 0.0
    assert records[0]["post_observation_index"] == 1.0
    assert records[0]["state_transition_data_available"] == 1.0
    assert records[0]["energy_data_available"] == 1.0
    assert records[0]["terminal_success"] == 1.0
    assert records[0]["terminal_reward_data_available"] == 1.0
    assert records[1]["energy_data_available"] == 0.0
    assert records[1]["post_observation_index"] == -1.0
    assert records[1]["state_transition_data_available"] == 0.0
    assert records[1]["terminal_reward_data_available"] == 0.0
    assert records[1]["terminal_reward"] == 0.0
    assert records[0]["reward"] == pytest.approx(
        records[0]["per_step_reward"] + records[0]["terminal_reward"]
    )
    assert len(records) != len([command])


def test_p4_3_sequence_reward_uses_forward_observation_and_same_index_command(
    grasp_carry_dict, morphology
) -> None:
    task_spec = TaskSpec.from_dict(grasp_carry_dict)
    observations = [
        _observation(task_spec, morphology, time_s=0.0, object_x=1.0),
        _observation(task_spec, morphology, time_s=0.1, object_x=1.1),
        _observation(task_spec, morphology, time_s=0.2, object_x=2.0),
    ]
    commands = [
        _command(thrust_n=0.0, qp_residual=1.0, saturation=0.1),
        _command(thrust_n=5.0, qp_residual=2.0, saturation=0.2),
        _command(thrust_n=10.0, qp_residual=3.0, saturation=0.3),
    ]

    records = compute_p4_3_reward_records(
        task_spec=task_spec,
        runtime_observations=observations,
        controller_commands=commands,
        release_valid=True,
        config=P4_3RewardConfig(
            progress_scale_m=1.0,
            rotor_thrust_scale_n=10.0,
            qp_residual_scale=10.0,
        ),
    )

    # command[0] is paired with obs[0] -> obs[1], not obs[-1] -> obs[0].
    assert records[0]["r_object_goal_progress"] == pytest.approx(0.1)
    assert records[0]["r_energy"] == pytest.approx(0.0)
    assert records[0]["r_qp_residual"] == pytest.approx(0.1)
    assert records[0]["r_actuator_saturation"] == pytest.approx(0.1)
    # command[1] is paired with obs[1] -> obs[2] and owns the terminal reward.
    assert records[1]["r_object_goal_progress"] == pytest.approx(0.9)
    assert records[1]["r_energy"] == pytest.approx(0.25)
    assert records[1]["r_qp_residual"] == pytest.approx(0.2)
    assert records[1]["r_actuator_saturation"] == pytest.approx(0.2)
    assert records[1]["terminal_success"] == 1.0
    assert records[1]["terminal_reward_data_available"] == 1.0
    # command[2] remains an imitation sample, but has no post-observation.
    assert records[2]["r_object_goal_progress"] == 0.0
    assert records[2]["object_goal_progress_data_available"] == 0.0
    assert records[2]["r_object_pose_accuracy"] == 0.0
    assert records[2]["r_energy"] == pytest.approx(1.0)
    assert records[2]["r_qp_residual"] == pytest.approx(0.3)
    assert records[2]["r_actuator_saturation"] == pytest.approx(0.3)
    assert records[2]["terminal_reward_data_available"] == 0.0
    assert records[2]["terminal_reward"] == 0.0


def test_p4_3_single_observation_has_command_only_reward_and_no_inferred_terminal(
    grasp_carry_dict, morphology
) -> None:
    task_spec = TaskSpec.from_dict(grasp_carry_dict)
    records = compute_p4_3_reward_records(
        task_spec=task_spec,
        runtime_observations=[
            _observation(task_spec, morphology, time_s=0.0, object_x=2.0)
        ],
        controller_commands=[
            _command(thrust_n=10.0, qp_residual=4.0, saturation=0.5)
        ],
        release_valid=True,
        config=P4_3RewardConfig(
            rotor_thrust_scale_n=10.0,
            qp_residual_scale=10.0,
        ),
    )

    assert len(records) == 1
    assert records[0]["state_transition_data_available"] == 0.0
    assert records[0]["post_observation_data_available"] == 0.0
    assert records[0]["r_object_goal_progress"] == 0.0
    assert records[0]["r_object_pose_accuracy"] == 0.0
    assert records[0]["r_energy"] == pytest.approx(1.0)
    assert records[0]["r_qp_residual"] == pytest.approx(0.4)
    assert records[0]["r_actuator_saturation"] == pytest.approx(0.5)
    assert records[0]["terminal_reward_data_available"] == 0.0
    assert records[0]["terminal_reward"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"w_slip": -1.0},
        {"progress_scale_m": 0.0},
        {"qp_residual_scale": float("nan")},
    ],
)
def test_p4_3_reward_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        P4_3RewardConfig(**kwargs)


def _observation(
    task_spec: TaskSpec,
    morphology,
    *,
    time_s: float,
    object_x: float,
    contacts: list[ContactState] | None = None,
    progress_metrics: dict[str, float] | None = None,
) -> RuntimeObservation:
    initial = task_spec.scene.objects[0].pose_world
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.1, 0.0, 0.0, 0.0, 0.0, 0.1],
            )
            for module in morphology.modules
        ],
        object_states=[
            ObjectRuntimeState(
                object_id="box_01",
                pose_world=(object_x, initial[1], initial[2], 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=list(contacts or []),
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(metrics=dict(progress_metrics or {})),
    )


def _command(*, thrust_n: float, qp_residual: float, saturation: float) -> ControllerCommand:
    return ControllerCommand(
        rotor_thrusts_n={"module_0:thrust_1": thrust_n},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=ControllerStatus(
            status="ok",
            qp_feasible=True,
            metrics={
                "allocation_residual_norm": qp_residual,
                "rotor_saturation_ratio": saturation,
            },
        ),
    )
