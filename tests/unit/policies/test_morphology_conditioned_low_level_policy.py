from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    BaselineLowLevelPolicyConfig,
    LowLevelPolicyContext,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
    MorphologyConditionedActorCritic,
    MorphologyConditionedLowLevelPolicy,
    Order3MorphologyConditionedPolicyConfig,
    load_order3_policy_checkpoint,
    order3_actor_feature_schema_hash,
    order3_actor_feature_vector,
    order3_graph_feature_schema_hash,
    save_order3_policy_checkpoint,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_ACTION_SIZE,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3PolicyCheckpointMetadata,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.schemas.runtime import (
    ContactState,
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.utils.hashing import hash_file, stable_hash


@pytest.fixture(scope="module")
def physical_model() -> PhysicalModel:
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


@pytest.fixture(scope="module")
def morphology_distribution(
    physical_model: PhysicalModel,
) -> RandomConnectedMorphologyDistribution:
    return RandomConnectedMorphologyDistribution(physical_model)


def test_actor_critic_shapes_masks_and_backpropagate_through_graph_and_gru(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graphs = [
        morphology_distribution.sample(seed=101, module_count=2),
        morphology_distribution.sample(seed=102, module_count=5),
    ]
    observations = [_runtime(graph, physical_model) for graph in graphs]
    config = _small_config()
    torch.manual_seed(123)
    model = MorphologyConditionedActorCritic(config)
    actor_features = torch.zeros((2, len(ORDER3_ACTOR_FEATURE_NAMES)))
    previous_action = torch.zeros((2, ORDER3_ACTION_SIZE))
    recurrent_state = model.initial_state(2)

    step = model.step(
        graphs,
        observations,
        actor_features,
        previous_action,
        recurrent_state,
        privileged_disturbance_body=torch.ones((2, 6)),
        deterministic=False,
    )

    assert step.action.shape == (2, ORDER3_ACTION_SIZE)
    assert step.action_mean.shape == (2, ORDER3_ACTION_SIZE)
    assert step.log_prob.shape == (2,)
    assert step.entropy.shape == (2,)
    assert step.value.shape == (2,)
    assert step.recurrent_state.shape == (2, config.recurrent_hidden_dim)
    assert step.graph_encoding.tokens.shape == (2, 5, config.graph_hidden_dim)
    assert step.graph_encoding.mask.sum(dim=1).tolist() == [2, 5]
    assert step.joint_residuals.shape == (2, 5, 3 * config.max_local_joint_slots)
    assert torch.count_nonzero(step.joint_residuals[0, 2:]).item() == 0
    assert bool(torch.isfinite(step.log_prob).all().item())
    assert bool((step.action.abs() <= 1.0).all().item())

    loss = (
        step.action.square().mean()
        + step.value.square().mean()
        + step.recurrent_state.square().mean()
        + step.joint_residuals.square().mean()
    )
    loss.backward()
    assert model.graph_encoder.node_projection[0].weight.grad is not None
    assert torch.count_nonzero(model.graph_encoder.node_projection[0].weight.grad).item() > 0
    assert model.recurrent.weight_ih.grad is not None
    assert torch.count_nonzero(model.recurrent.weight_ih.grad).item() > 0
    assert model.actor_mean.weight.grad is not None
    assert torch.count_nonzero(model.actor_mean.weight.grad).item() > 0


def test_deterministic_and_stochastic_actions_have_consistent_log_prob_contract(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=110, module_count=3)
    observation = _runtime(graph, physical_model)
    config = _small_config()
    torch.manual_seed(19)
    model = MorphologyConditionedActorCritic(config)
    features = torch.zeros((1, len(ORDER3_ACTOR_FEATURE_NAMES)))
    previous = torch.zeros((1, ORDER3_ACTION_SIZE))
    hidden = model.initial_state(1)

    first = model.step(
        [graph], [observation], features, previous, hidden, deterministic=True
    )
    second = model.step(
        [graph], [observation], features, previous, hidden, deterministic=True
    )
    assert torch.equal(first.action, first.action_mean)
    assert torch.equal(first.action, second.action)
    assert torch.equal(first.log_prob, second.log_prob)

    torch.manual_seed(20)
    stochastic_first = model.step(
        [graph], [observation], features, previous, hidden, deterministic=False
    )
    torch.manual_seed(21)
    stochastic_second = model.step(
        [graph], [observation], features, previous, hidden, deterministic=False
    )
    assert not torch.equal(stochastic_first.action, stochastic_second.action)
    assert bool(torch.isfinite(stochastic_first.log_prob).all().item())
    assert bool((stochastic_first.action.abs() <= 1.0).all().item())

    evaluated = model.step(
        [graph],
        [observation],
        features,
        previous,
        hidden,
        action=stochastic_first.action.detach(),
    )
    assert torch.allclose(evaluated.action, stochastic_first.action)
    assert torch.allclose(evaluated.log_prob, stochastic_first.log_prob, atol=1.0e-5)


def test_deployable_policy_passes_centroidal_pose_exactly_and_bounds_learned_fields(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=120, module_count=2)
    context = _context(graph, physical_model, time_s=0.4)
    config = _small_config(
        trust_region_blend=0.25,
        linear_twist_correction_limit_mps=0.4,
        angular_twist_correction_limit_radps=0.6,
        residual_force_weight_fraction=0.1,
        residual_torque_per_module_nm=0.2,
    )
    model = _constant_action_model(config, twist_sign=1.0, wrench_sign=-1.0)
    policy = MorphologyConditionedLowLevelPolicy(
        model=model,
        physical_model=physical_model,
        config=config,
        deterministic=True,
    )
    control_model = RigidBodyControlModelBuilder().build(
        graph,
        physical_model,
        context.runtime_observation,
    )

    trace = policy.command_with_trace(context)
    command = trace.command

    assert trace.learned_policy_applied is True
    assert command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert context.active_knot is not None
    assert context.active_knot.centroidal_target is not None
    target = context.active_knot.centroidal_target
    assert target.com_pos_world is not None
    assert target.body_orientation_world is not None
    expected_pose = (*target.com_pos_world, *target.body_orientation_world)
    assert command.desired_body_pose == expected_pose

    baseline_twist = [*target.com_vel_world, 0.0, 0.0, 0.0]  # type: ignore[misc]
    assert command.desired_body_twist is not None
    twist_delta = [
        command.desired_body_twist[index] - baseline_twist[index]
        for index in range(6)
    ]
    twist_bounds = [
        *([config.trust_region_blend * config.linear_twist_correction_limit_mps] * 3),
        *([config.trust_region_blend * config.angular_twist_correction_limit_radps] * 3),
    ]
    assert all(abs(delta) <= bound + 1.0e-7 for delta, bound in zip(twist_delta, twist_bounds))
    assert all(delta > 0.99 * bound for delta, bound in zip(twist_delta, twist_bounds))

    assert command.residual_wrench_body is not None
    wrench_bounds = [
        *(
            [
                config.trust_region_blend
                * control_model.total_mass_kg
                * 9.81
                * config.residual_force_weight_fraction
            ]
            * 3
        ),
        *(
            [
                config.trust_region_blend
                * len(graph.modules)
                * config.residual_torque_per_module_nm
            ]
            * 3
        ),
    ]
    assert all(
        abs(value) <= bound + 1.0e-7
        for value, bound in zip(command.residual_wrench_body, wrench_bounds)
    )
    assert all(value < -0.99 * bound for value, bound in zip(command.residual_wrench_body, wrench_bounds))


def test_free_flight_outputs_absolute_dock_hold_only_and_clears_deprecated_contact_fields(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=130, module_count=3)
    context = _context(graph, physical_model)
    config = _small_config(free_flight_joint_residual_enabled=False)
    policy = MorphologyConditionedLowLevelPolicy(
        model=_constant_action_model(config, twist_sign=0.0, wrench_sign=0.0),
        physical_model=physical_model,
        config=config,
    )

    command = policy.command(context)

    dock_joint_ids = _dock_joint_ids(physical_model)
    vectoring_joint_ids = {
        joint_id for rotor in physical_model.rotors for joint_id in rotor.vectoring_joint_ids
    }
    expected_positions = {
        f"module_{state.module_id}:{joint_id}": 0.0
        for state in context.runtime_observation.module_states
        for joint_id in dock_joint_ids
    }
    assert command.joint_position_targets == expected_positions
    assert command.joint_velocity_targets == {key: 0.0 for key in expected_positions}
    assert command.joint_torque_bias == {key: 0.0 for key in expected_positions}
    assert not any(
        global_id.partition(":")[2] in vectoring_joint_ids
        for global_id in command.joint_position_targets
    )
    assert command.desired_anchor_pose_offsets == {}
    assert command.joint_position_bias == {}
    assert command.joint_velocity_bias == {}
    assert command.contact_tracking_bias == {}
    assert not hasattr(command, "internal_wrench_bias")


def test_contact_wrench_is_private_to_actor_but_privileged_critic_input_does_not_change_action(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=140, module_count=4)
    first = _runtime(graph, physical_model, contact_wrench_scale=1.0)
    second = _runtime(graph, physical_model, contact_wrench_scale=1_000.0)
    control_builder = RigidBodyControlModelBuilder()
    first_control = control_builder.build(graph, physical_model, first)
    second_control = control_builder.build(graph, physical_model, second)
    target_pose = (0.4, -0.2, 1.8, 0.0, 0.0, 0.0, 1.0)
    target_twist = [0.0] * 6

    first_features = order3_actor_feature_vector(
        first,
        first_control,
        target_pose_world=target_pose,
        target_twist=target_twist,
    )
    second_features = order3_actor_feature_vector(
        second,
        second_control,
        target_pose_world=target_pose,
        target_twist=target_twist,
    )
    assert first_features == second_features

    config = _small_config()
    torch.manual_seed(44)
    model = MorphologyConditionedActorCritic(config)
    hidden = model.initial_state(1)
    previous = torch.zeros((1, ORDER3_ACTION_SIZE))
    first_step = model.step(
        [graph],
        [first],
        torch.tensor([first_features]),
        previous,
        hidden,
        privileged_disturbance_body=torch.zeros((1, 6)),
        deterministic=True,
    )
    second_step = model.step(
        [graph],
        [second],
        torch.tensor([second_features]),
        previous,
        hidden,
        privileged_disturbance_body=torch.full((1, 6), 100.0),
        deterministic=True,
    )
    assert torch.equal(first_step.action, second_step.action)
    assert torch.equal(first_step.action_mean, second_step.action_mean)
    assert torch.equal(first_step.recurrent_state, second_step.recurrent_state)


def test_gru_state_carries_between_steps_and_resets_when_episode_time_rewinds(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=150, module_count=2)
    config = _small_config()
    torch.manual_seed(51)
    policy = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
    )

    first = policy.command_with_trace(_context(graph, physical_model, time_s=0.5))
    second = policy.command_with_trace(_context(graph, physical_model, time_s=0.6))
    rewind = policy.command_with_trace(_context(graph, physical_model, time_s=0.1))

    assert first.learned_policy_applied and second.learned_policy_applied and rewind.learned_policy_applied
    assert second.recurrent_state_in == pytest.approx(first.recurrent_state_out)
    assert any(abs(value) > 1.0e-8 for value in second.recurrent_state_in)
    assert rewind.recurrent_state_in == pytest.approx([0.0] * config.recurrent_hidden_dim)

    policy.reset()
    after_explicit_reset = policy.command_with_trace(
        _context(graph, physical_model, time_s=0.7)
    )
    assert after_explicit_reset.recurrent_state_in == pytest.approx(
        [0.0] * config.recurrent_hidden_dim
    )


def test_controller_and_actor_feature_ood_use_strict_v2_fallback(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=160, module_count=2)
    config = _small_config()
    policy = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
    )
    infeasible_status = ControllerStatus(status="infeasible", qp_feasible=False)
    infeasible = policy.command_with_trace(
        _context(
            graph,
            physical_model,
            controller_status=infeasible_status,
        )
    )

    assert infeasible.learned_policy_applied is False
    assert infeasible.fallback_reason == "controller_infeasible"
    assert infeasible.command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert infeasible.command.joint_position_targets
    assert set(infeasible.command.joint_position_targets.values()) == {0.0}
    assert infeasible.command.contact_tracking_bias == {}

    ood_config = _small_config(ood_absolute_feature_limit=1.0e-5)
    ood_policy = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(ood_config),
        physical_model=physical_model,
        config=ood_config,
    )
    ood = ood_policy.command_with_trace(_context(graph, physical_model, time_s=0.4))
    assert ood.learned_policy_applied is False
    assert ood.fallback_reason == "actor_feature_ood"
    assert ood.command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert ood.command.joint_position_targets
    assert set(ood.command.joint_position_targets.values()) == {0.0}


def test_checkpoint_rejects_legacy_roundtrips_v2_and_rejects_contract_tampering(
    tmp_path: Path,
    physical_model: PhysicalModel,
) -> None:
    legacy_path = tmp_path / "legacy_v1.pt"
    torch.save(
        {
            "checkpoint_version": "p4_3_pi_l_checkpoint_v1",
            "state_dict": {},
        },
        legacy_path,
    )
    with pytest.raises(SchemaValidationError, match="not the centroidal morphology-conditioned"):
        load_order3_policy_checkpoint(legacy_path)

    config = _small_config()
    torch.manual_seed(71)
    model = MorphologyConditionedActorCritic(config)
    metadata = _checkpoint_metadata(config, physical_model)
    checkpoint_path = tmp_path / "order3_v2.pt"
    saved_hash = save_order3_policy_checkpoint(
        checkpoint_path,
        model=model,
        metadata=metadata,
    )
    loaded = load_order3_policy_checkpoint(checkpoint_path)
    assert saved_hash == hash_file(checkpoint_path)
    assert loaded.sha256 == saved_hash
    assert loaded.config.to_dict() == config.to_dict()
    assert loaded.metadata.to_dict() == metadata.to_dict()
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected)

    verified_policy = MorphologyConditionedLowLevelPolicy.from_checkpoint(
        checkpoint_path,
        physical_model=physical_model,
        expected_sha256=saved_hash,
    )
    assert verified_policy.config.to_dict() == config.to_dict()
    mismatched_fallback = BaselineLowLevelPolicy(
        BaselineLowLevelPolicyConfig(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            residual_force_limit_n=3.5,
        )
    )
    with pytest.raises(SchemaValidationError, match="fallback config hash"):
        MorphologyConditionedLowLevelPolicy.from_checkpoint(
            checkpoint_path,
            physical_model=physical_model,
            expected_sha256=saved_hash,
            baseline_policy=mismatched_fallback,
        )
    with pytest.raises(SchemaValidationError, match="sha256 mismatch"):
        load_order3_policy_checkpoint(
            checkpoint_path,
            expected_sha256="0" * 64,
        )

    weight_tamper = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weight_name = next(iter(weight_tamper["state_dict"]))
    weight_tamper["state_dict"][weight_name] = (
        weight_tamper["state_dict"][weight_name].clone() + 1.0e-3
    )
    weight_tamper_path = tmp_path / "weight-tampered.pt"
    torch.save(weight_tamper, weight_tamper_path)
    with pytest.raises(SchemaValidationError, match="sha256 mismatch"):
        load_order3_policy_checkpoint(
            weight_tamper_path,
            expected_sha256=saved_hash,
        )

    mismatched_model = replace(
        physical_model,
        aggregate_mass_kg=physical_model.aggregate_mass_kg + 0.01,
    )
    with pytest.raises(SchemaValidationError, match="PhysicalModel hash"):
        MorphologyConditionedLowLevelPolicy.from_checkpoint(
            checkpoint_path,
            physical_model=mismatched_model,
            expected_sha256=saved_hash,
        )

    alternate_urdf_path = tmp_path / "alternate.urdf"
    alternate_urdf_path.write_text(
        Path(physical_model.urdf_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    alternate_model = replace(physical_model, urdf_path=str(alternate_urdf_path))
    alternate_metadata = replace(
        metadata,
        physical_model_hash=alternate_model.stable_hash(),
        urdf_hash="intentionally-wrong-urdf-hash",
    )
    alternate_checkpoint_path = tmp_path / "alternate-model.pt"
    alternate_hash = save_order3_policy_checkpoint(
        alternate_checkpoint_path,
        model=model,
        metadata=alternate_metadata,
    )
    with pytest.raises(SchemaValidationError, match="URDF hash"):
        MorphologyConditionedLowLevelPolicy.from_checkpoint(
            alternate_checkpoint_path,
            physical_model=alternate_model,
            expected_sha256=alternate_hash,
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["metadata"]["encoder_version"] = "tampered_encoder"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(payload, tampered_path)
    with pytest.raises(SchemaValidationError, match="encoder_version"):
        load_order3_policy_checkpoint(tampered_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["unexpected"] = True
    extra_key_path = tmp_path / "extra-key.pt"
    torch.save(payload, extra_key_path)
    with pytest.raises(SchemaValidationError, match="keys do not match"):
        load_order3_policy_checkpoint(extra_key_path)


def test_checkpoint_structural_allowlist_falls_back_for_valid_unseen_graph(
    tmp_path: Path,
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    allowed_graph = morphology_distribution.sample(seed=810, module_count=3)
    unseen_graph = morphology_distribution.sample(seed=811, module_count=3)
    assert morphology_structural_hash(allowed_graph) != morphology_structural_hash(
        unseen_graph
    )
    config = _small_config()
    model = _constant_action_model(config, twist_sign=1.0, wrench_sign=1.0)
    metadata = _checkpoint_metadata(
        config,
        physical_model,
        morphology_hashes=[morphology_structural_hash(allowed_graph)],
    )
    path = tmp_path / "allowlisted.pt"
    expected_hash = save_order3_policy_checkpoint(path, model=model, metadata=metadata)
    policy = MorphologyConditionedLowLevelPolicy.from_checkpoint(
        path,
        physical_model=physical_model,
        expected_sha256=expected_hash,
    )

    allowed = policy.command_with_trace(_context(allowed_graph, physical_model))
    assert allowed.learned_policy_applied is True
    policy.reset()
    unseen = policy.command_with_trace(_context(unseen_graph, physical_model))
    assert unseen.learned_policy_applied is False
    assert unseen.fallback_reason == "structural_hash_ood"


def test_privileged_critic_input_changes_value_without_changing_actor_outputs(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=910, module_count=3)
    observation = _runtime(graph, physical_model)
    config = _small_config()
    model = MorphologyConditionedActorCritic(config)
    with torch.no_grad():
        model.critic[0].weight.zero_()
        model.critic[0].bias.zero_()
        model.critic[0].weight[:, -6:] = 0.1
        model.critic[2].weight.fill_(0.1)
        model.critic[2].bias.zero_()
    actor_features = torch.zeros((1, len(ORDER3_ACTOR_FEATURE_NAMES)))
    previous_action = torch.zeros((1, ORDER3_ACTION_SIZE))
    recurrent = model.initial_state(1)
    nominal = model.step(
        [graph],
        [observation],
        actor_features,
        previous_action,
        recurrent,
        privileged_disturbance_body=torch.zeros((1, 6)),
        deterministic=True,
    )
    disturbed = model.step(
        [graph],
        [observation],
        actor_features,
        previous_action,
        recurrent,
        privileged_disturbance_body=torch.ones((1, 6)),
        deterministic=True,
    )

    assert torch.equal(nominal.action, disturbed.action)
    assert torch.equal(nominal.action_mean, disturbed.action_mean)
    assert torch.equal(nominal.recurrent_state, disturbed.recurrent_state)
    assert torch.equal(nominal.log_prob, disturbed.log_prob)
    assert not torch.equal(nominal.value, disturbed.value)


def _small_config(**overrides) -> Order3MorphologyConditionedPolicyConfig:
    config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    return replace(config, **overrides)


def _constant_action_model(
    config: Order3MorphologyConditionedPolicyConfig,
    *,
    twist_sign: float,
    wrench_sign: float,
) -> MorphologyConditionedActorCritic:
    model = MorphologyConditionedActorCritic(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor_mean.bias[:6].fill_(20.0 * twist_sign)
        model.actor_mean.bias[6:].fill_(20.0 * wrench_sign)
    return model


def _context(
    graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    time_s: float = 0.25,
    controller_status: ControllerStatus | None = None,
) -> LowLevelPolicyContext:
    status = controller_status or ControllerStatus(
        status="ok",
        qp_feasible=True,
        metrics={"allocation_residual_norm": 0.01},
    )
    observation = _runtime(
        graph,
        physical_model,
        time_s=time_s,
        controller_status=status,
    )
    knot = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=(0.35, -0.15, 1.75),
            com_vel_world=(0.12, -0.08, 0.05),
            body_orientation_world=(0.0, 0.0, math.sin(0.1), math.cos(0.1)),
        ),
    )
    trajectory = ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.02,
        knots=[knot],
        derived_mode_label="order3_unit_free_flight",
    )
    return LowLevelPolicyContext(
        runtime_observation=observation,
        morphology_graph=graph,
        physical_model=physical_model,
        contact_wrench_trajectory=trajectory,
        active_knot=knot,
        controller_status=status,
    )


def _runtime(
    graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    time_s: float = 0.25,
    controller_status: ControllerStatus | None = None,
    contact_wrench_scale: float | None = None,
) -> RuntimeObservation:
    joint_ids = [joint.joint_id for joint in physical_model.joints]
    dock_ids = set(_dock_joint_ids(physical_model))
    states: list[ModuleRuntimeState] = []
    for module in graph.modules:
        design_pose = module.pose_in_design_frame
        positions = {
            joint_id: (
                0.01 * (module.module_id + 1) * (index + 1)
                if joint_id in dock_ids
                else 0.0
            )
            for index, joint_id in enumerate(joint_ids)
        }
        states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=(
                    0.2 + float(design_pose[0]),
                    -0.1 + float(design_pose[1]),
                    1.3 + float(design_pose[2]),
                    *design_pose[3:],
                ),
                twist_world=[0.01, -0.02, 0.03, 0.01, 0.02, -0.01],
                joint_positions=positions,
                joint_velocities={joint_id: 0.0 for joint_id in joint_ids},
                health=0.98,
            )
        )
    contacts = []
    if contact_wrench_scale is not None:
        contacts.append(
            ContactState(
                contact_id="privileged-contact",
                entity_a="robot",
                entity_b="floor",
                wrench_world=[
                    contact_wrench_scale,
                    -2.0 * contact_wrench_scale,
                    3.0 * contact_wrench_scale,
                    0.1 * contact_wrench_scale,
                    -0.2 * contact_wrench_scale,
                    0.3 * contact_wrench_scale,
                ],
            )
        )
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=graph,
        module_states=states,
        object_states=[],
        contact_states=contacts,
        controller_status=controller_status
        or ControllerStatus(
            status="ok",
            qp_feasible=True,
            metrics={"allocation_residual_norm": 0.01},
        ),
        task_progress=TaskProgressState(progress_ratio=0.2),
    )


def _dock_joint_ids(physical_model: PhysicalModel) -> list[str]:
    return sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
            if port.mechanical_limits.get("mechanism_joint_id")
        }
    )


def _checkpoint_metadata(
    config: Order3MorphologyConditionedPolicyConfig,
    physical_model: PhysicalModel,
    *,
    morphology_hashes: list[str] | None = None,
) -> Order3PolicyCheckpointMetadata:
    return Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        training_stage="bc",
        action_names=list(ORDER3_ACTION_NAMES),
        actor_feature_schema_hash=order3_actor_feature_schema_hash(),
        graph_feature_schema_hash=order3_graph_feature_schema_hash(),
        config_hash=config.stable_hash(),
        pool_hash="unit-pool-hash",
        dataset_hash="unit-dataset-hash",
        physical_model_hash=physical_model.stable_hash(),
        urdf_hash=hash_file(physical_model.urdf_path),
        controller_contract_hash="unit-controller-contract-hash",
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash=stable_hash(
            BaselineLowLevelPolicyConfig(
                control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            )
        ),
        seed=71,
        git_revision="unit-test-revision",
        metadata={
            "morphology_hashes": {
                "train": list(morphology_hashes or ["a" * 64]),
                "validation": [],
                "held_out": [],
            }
        },
    )
