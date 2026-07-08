from __future__ import annotations

from dataclasses import replace

from amsrr.irg import IRGBuilder, InteractionEnvelopeExtractor
from amsrr.policies import DesignPolicyContext, P2DesignPolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation import (
    SimplifiedGraspCarryEnv,
    SimplifiedGraspCarryEnvConfig,
    SimulationEnvBase,
    run_crash_free_episodes,
)


def _env(grasp_carry_dict: dict) -> SimplifiedGraspCarryEnv:
    task = TaskSpec.from_dict(grasp_carry_dict)
    return SimplifiedGraspCarryEnv(
        task,
        config=SimplifiedGraspCarryEnvConfig(
            dt_s=0.1,
            max_episode_steps=40,
            object_tracking_speed_mps=10.0,
            initial_position_jitter_m=0.02,
        ),
    )


def test_simplified_grasp_carry_env_matches_base_protocol(grasp_carry_dict: dict) -> None:
    env: SimulationEnvBase = _env(grasp_carry_dict)

    observation = env.reset(seed=0, episode_id="protocol_smoke")

    assert isinstance(observation, RuntimeObservation)
    assert env.get_runtime_observation().task_progress.phase_label == "establish_contact"
    assert callable(env.step)


def test_simplified_grasp_carry_env_runs_policy_controller_episode(grasp_carry_dict: dict) -> None:
    env = _env(grasp_carry_dict)

    result = env.run_episode(seed=3, episode_id="single_episode")

    assert result.episode_id == "single_episode"
    assert result.crashed is False
    assert result.success is True
    assert result.failure_reason is None
    assert result.metrics["goal_distance_m"] <= env.config.goal_tolerance_m
    assert result.metrics["ever_attached"] == 1.0
    assert result.metrics["policy_command_count"] == result.metrics["controller_command_count"]
    assert result.metrics["controller_command_count"] > 0
    assert env.artifacts.contact_candidate_set.candidates
    assert env.artifacts.contact_wrench_trajectory.knots
    assert env.get_runtime_observation().task_progress.success is True
    assert isinstance(env._last_controller_command, ControllerCommand)
    assert type(result).from_json(result.to_json()).to_dict() == result.to_dict()


def test_simplified_grasp_carry_env_accepts_external_design_output(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    builder_result = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(builder_result.irg)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    selection = P2DesignPolicy().evaluate_candidates(
        DesignPolicyContext(
            task_spec=task,
            irg=builder_result.irg,
            interaction_envelope=envelope,
            physical_model=physical_model,
        )
    )
    selected_design = selection.selected_candidate.design_output
    assembled_morphology = replace(
        selected_design.target_morphology,
        graph_id=f"{selected_design.target_morphology.graph_id}:assembled",
    )

    env = SimplifiedGraspCarryEnv(
        task,
        config=SimplifiedGraspCarryEnvConfig(
            dt_s=0.1,
            max_episode_steps=40,
            object_tracking_speed_mps=10.0,
            initial_position_jitter_m=0.0,
        ),
        design_output=selected_design,
        assembled_morphology=assembled_morphology,
    )
    result = env.run_episode(seed=5, episode_id="external_design")

    assert env.artifacts.design_source == "external_design_output_with_assembled_morphology"
    assert env.artifacts.design_output.task_id == selected_design.task_id
    assert env.artifacts.design_output.design_scores == selected_design.design_scores
    assert env.artifacts.design_output.target_morphology.graph_id == assembled_morphology.graph_id
    assert env.artifacts.contact_candidate_set.morphology_graph_id == assembled_morphology.graph_id
    assert env.artifacts.contact_candidate_set.candidates
    assert env.artifacts.contact_wrench_trajectory.knots
    assert result.success is True


def test_simplified_grasp_carry_1000_episodes_crash_free(grasp_carry_dict: dict) -> None:
    env = _env(grasp_carry_dict)

    batch = run_crash_free_episodes(env, episode_count=1000, seed=100)

    assert batch.episode_count == 1000
    assert batch.crash_count == 0
    assert batch.metrics["crash_rate"] == 0.0
    assert batch.metrics["success_rate"] >= 0.60
    assert batch.success_count == 1000
    assert type(batch).from_json(batch.to_json()).to_dict() == batch.to_dict()
