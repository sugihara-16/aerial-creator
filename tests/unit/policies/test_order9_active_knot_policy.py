from __future__ import annotations

import torch

from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
)
from amsrr.policies.order9_active_knot_features import (
    ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES,
    ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9ActiveKnotLowLevelPolicyConfig,
    Order9ActiveKnotPhaseConditionedActorCritic,
    Order9ContactResidualPhaseConditionedActorCritic,
    Order9LowLevelPolicyConfig,
    Order9MorphologyInvariantCompressionActorCritic,
    Order9PhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.task_spec import TaskType


def test_active_knot_migration_preserves_legacy_actor_exactly() -> None:
    torch.manual_seed(9019)
    legacy_config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    legacy = Order9PhaseConditionedActorCritic(legacy_config)
    successor_config = Order9ActiveKnotLowLevelPolicyConfig.from_dict(
        legacy_config.to_dict()
    )
    successor = Order9ActiveKnotPhaseConditionedActorCritic(successor_config)
    missing, unexpected = successor.initialize_from_legacy_order9(legacy)
    assert missing
    assert not unexpected

    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9019, module_count=3
    )
    batch = 2
    actor = torch.randn(batch, len(ORDER3_ACTOR_FEATURE_NAMES))
    phase = torch.randn(batch, legacy_config.phase_feature_dim)
    previous = torch.randn(batch, ORDER9_GLOBAL_ACTION_SIZE).tanh()
    hidden = torch.randn(batch, legacy_config.recurrent_hidden_dim)
    active_global = torch.randn(
        batch, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
    )
    active_nodes = torch.randn(
        batch,
        len(graph.modules),
        len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
    )
    legacy_step = legacy.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        deterministic=True,
    )
    successor_step = successor.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    for actual, expected in (
        (successor_step.action, legacy_step.action),
        (successor_step.action_mean, legacy_step.action_mean),
        (successor_step.joint_action, legacy_step.joint_action),
        (successor_step.joint_action_mean, legacy_step.joint_action_mean),
        (successor_step.log_prob, legacy_step.log_prob),
        (successor_step.value, legacy_step.value),
        (successor_step.recurrent_state, legacy_step.recurrent_state),
    ):
        assert torch.equal(actual, expected)


def test_active_knot_successor_can_learn_context_sensitivity() -> None:
    torch.manual_seed(9020)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    policy = Order9ActiveKnotPhaseConditionedActorCritic(config)
    with torch.no_grad():
        policy.active_knot_encoder[2].weight.fill_(0.01)
        policy.active_assignment_encoder[2].weight.fill_(0.01)
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9020, module_count=2
    )
    actor = torch.zeros(1, len(ORDER3_ACTOR_FEATURE_NAMES))
    phase = torch.zeros(1, config.phase_feature_dim)
    previous = torch.zeros(1, ORDER9_GLOBAL_ACTION_SIZE)
    hidden = torch.zeros(1, config.recurrent_hidden_dim)
    active_global = torch.zeros(
        1, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
    )
    active_nodes = torch.zeros(
        1,
        len(graph.modules),
        len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
    )
    first = policy.step(
        [graph],
        [None],
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    active_global[:, 0] = 1.0
    active_nodes[:, 0, 0] = 1.0
    second = policy.step(
        [graph],
        [None],
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    assert not torch.equal(first.recurrent_state, second.recurrent_state)
    assert not torch.equal(first.action_mean, second.action_mean)


def test_contact_residual_migration_is_exact_and_phase_gated() -> None:
    torch.manual_seed(9022)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    source = Order9ActiveKnotPhaseConditionedActorCritic(config)
    target = Order9ContactResidualPhaseConditionedActorCritic(config)
    missing, unexpected = target.initialize_from_active_knot(source)
    assert missing
    assert not unexpected

    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9022, module_count=2
    )
    batch = 2
    actor = torch.randn(batch, len(ORDER3_ACTOR_FEATURE_NAMES))
    previous = torch.randn(batch, ORDER9_GLOBAL_ACTION_SIZE).tanh()
    hidden = torch.randn(batch, config.recurrent_hidden_dim)
    active_global = torch.randn(
        batch, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
    )
    active_nodes = torch.randn(
        batch,
        len(graph.modules),
        len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
    )
    phase = torch.zeros(batch, config.phase_feature_dim)
    phase[:, 0] = 1.0
    phase[:, len(TaskType)] = 1.0
    source_step = source.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    initial_step = target.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    assert torch.equal(initial_step.joint_action_mean, source_step.joint_action_mean)

    with torch.no_grad():
        target.contact_residual_decoder[-1].bias.fill_(0.2)
    approach_step = target.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    assert torch.equal(approach_step.joint_action_mean, source_step.joint_action_mean)

    contact_phase = phase.clone()
    contact_phase[:, len(TaskType)] = 0.0
    contact_phase[:, len(TaskType) + 1] = 1.0
    source_contact = source.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=contact_phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    target_contact = target.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=contact_phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    assert not torch.equal(
        target_contact.joint_action_mean, source_contact.joint_action_mean
    )


def test_morphology_invariant_compression_migration_preserves_v5_outputs() -> None:
    torch.manual_seed(9023)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    source = Order9ContactResidualPhaseConditionedActorCritic(config)
    target = Order9MorphologyInvariantCompressionActorCritic(config)
    missing, unexpected = target.initialize_from_contact_residual(source)
    assert set(missing) == {
        "contact_compression_actor_log_std",
        "contact_compression_actor_mean.0.weight",
        "contact_compression_actor_mean.0.bias",
        "contact_compression_actor_mean.2.weight",
        "contact_compression_actor_mean.2.bias",
        "contact_compression_module_count_bias",
    }
    assert not unexpected

    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9023, module_count=3
    )
    batch = 2
    actor = torch.randn(batch, len(ORDER3_ACTOR_FEATURE_NAMES))
    previous = torch.randn(batch, ORDER9_GLOBAL_ACTION_SIZE).tanh()
    hidden = torch.randn(batch, config.recurrent_hidden_dim)
    active_global = torch.randn(
        batch, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
    )
    active_nodes = torch.randn(
        batch,
        len(graph.modules),
        len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
    )
    phase = torch.zeros(batch, config.phase_feature_dim)
    phase[:, 0] = 1.0
    phase[:, len(TaskType) + 1] = 1.0
    source_step = source.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    target_step = target.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    for actual, expected in (
        (target_step.action, source_step.action),
        (target_step.joint_action, source_step.joint_action),
        (target_step.value, source_step.value),
        (target_step.recurrent_state, source_step.recurrent_state),
    ):
        assert torch.equal(actual, expected)
    assert torch.count_nonzero(
        target_step.contact_compression_residual_action
    ).item() == 0

    with torch.no_grad():
        target.contact_compression_actor_mean[-1].bias.fill_(0.3)
    changed = target.step(
        [graph] * batch,
        [None] * batch,
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        deterministic=True,
    )
    assert torch.allclose(
        changed.contact_compression_residual_action,
        torch.full((batch,), torch.tanh(torch.tensor(0.3))),
    )
    assert torch.equal(changed.joint_action, source_step.joint_action)


def test_morphology_invariant_compression_action_log_prob_replays_exactly() -> None:
    torch.manual_seed(9024)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    policy = Order9MorphologyInvariantCompressionActorCritic(config)
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9024, module_count=2
    )
    actor = torch.randn(1, len(ORDER3_ACTOR_FEATURE_NAMES))
    previous = torch.randn(1, ORDER9_GLOBAL_ACTION_SIZE).tanh()
    hidden = torch.randn(1, config.recurrent_hidden_dim)
    phase = torch.zeros(1, config.phase_feature_dim)
    phase[:, 0] = 1.0
    phase[:, len(TaskType) + 1] = 1.0
    active_global = torch.randn(
        1, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
    )
    active_nodes = torch.randn(
        1,
        len(graph.modules),
        len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
    )
    sampled = policy.step(
        [graph],
        [None],
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
    )
    replayed = policy.step(
        [graph],
        [None],
        actor,
        previous,
        hidden,
        phase_features=phase,
        active_knot_features=active_global,
        active_assignment_features=active_nodes,
        action=sampled.action,
        joint_action=sampled.joint_action,
        contact_compression_residual_action=(
            sampled.contact_compression_residual_action
        ),
    )

    assert torch.equal(
        replayed.contact_compression_residual_action,
        sampled.contact_compression_residual_action,
    )
    assert torch.allclose(replayed.log_prob, sampled.log_prob, atol=1.0e-6)


def test_v6_full_action_log_prob_reaches_every_output_head() -> None:
    torch.manual_seed(9025)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    policy = Order9MorphologyInvariantCompressionActorCritic(config)
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9025, module_count=2
    )
    phase = torch.zeros(1, config.phase_feature_dim)
    phase[:, 0] = 1.0
    phase[:, len(TaskType) + 1] = 1.0
    step = policy.step(
        [graph],
        [None],
        torch.randn(1, len(ORDER3_ACTOR_FEATURE_NAMES)),
        torch.zeros(1, ORDER9_GLOBAL_ACTION_SIZE),
        torch.zeros(1, config.recurrent_hidden_dim),
        phase_features=phase,
        active_knot_features=torch.randn(
            1, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
        ),
        active_assignment_features=torch.randn(
            1,
            len(graph.modules),
            len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
        ),
        action=torch.full((1, ORDER9_GLOBAL_ACTION_SIZE), 0.2),
        joint_action=torch.full(
            (1, len(graph.modules), 3 * config.max_local_joint_slots), 0.2
        ),
        contact_compression_residual_action=torch.tensor([0.2]),
    )
    (-step.log_prob.mean()).backward()

    for parameter in (
        policy.actor_mean.weight,
        policy.joint_decoder[-1].weight,
        policy.contact_residual_decoder[-1].weight,
        policy.contact_compression_actor_mean[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_morphology_invariant_compression_count_calibration_does_not_leak() -> None:
    torch.manual_seed(9025)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    policy = Order9MorphologyInvariantCompressionActorCritic(config)
    with torch.no_grad():
        policy.contact_compression_module_count_bias[5] = 0.3
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    phase = torch.zeros(1, config.phase_feature_dim)
    phase[:, 0] = 1.0
    phase[:, len(TaskType) + 1] = 1.0

    outputs = {}
    for module_count in range(2, 6):
        graph = RandomConnectedMorphologyDistribution(physical).sample(
            seed=9100 + module_count, module_count=module_count
        )
        step = policy.step(
            [graph],
            [None],
            torch.zeros(1, len(ORDER3_ACTOR_FEATURE_NAMES)),
            torch.zeros(1, ORDER9_GLOBAL_ACTION_SIZE),
            torch.zeros(1, config.recurrent_hidden_dim),
            phase_features=phase,
            active_knot_features=torch.zeros(
                1, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
            ),
            active_assignment_features=torch.zeros(
                1,
                len(graph.modules),
                len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
            ),
            deterministic=True,
        )
        outputs[module_count] = float(
            step.contact_compression_residual_action.item()
        )

    assert outputs[2] == 0.0
    assert outputs[3] == 0.0
    assert outputs[4] == 0.0
    assert outputs[5] == torch.tanh(torch.tensor(0.3)).item()


def test_order9_critic_value_gradient_is_detached_from_actor_trunk() -> None:
    torch.manual_seed(9021)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    policy = Order9ActiveKnotPhaseConditionedActorCritic(config)
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9021, module_count=3
    )
    step = policy.step(
        [graph],
        [None],
        torch.randn(1, len(ORDER3_ACTOR_FEATURE_NAMES)),
        torch.zeros(1, ORDER9_GLOBAL_ACTION_SIZE),
        policy.initial_state(1),
        phase_features=torch.randn(1, config.phase_feature_dim),
        privileged_disturbance_body=torch.randn(1, 6),
        active_knot_features=torch.randn(
            1, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
        ),
        active_assignment_features=torch.randn(
            1,
            len(graph.modules),
            len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
        ),
        deterministic=True,
    )
    step.value.sum().backward()

    assert any(parameter.grad is not None for parameter in policy.critic.parameters())
    detached_modules = (
        policy.graph_encoder,
        policy.actor_feature_encoder,
        policy.phase_encoder,
        policy.active_knot_encoder,
        policy.active_assignment_encoder,
        policy.fusion,
        policy.recurrent,
    )
    assert all(
        parameter.grad is None
        for module in detached_modules
        for parameter in module.parameters()
    )
