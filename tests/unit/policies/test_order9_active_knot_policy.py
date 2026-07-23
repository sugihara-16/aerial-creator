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
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)


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
