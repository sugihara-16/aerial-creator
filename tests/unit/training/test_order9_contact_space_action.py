from __future__ import annotations

import pytest
import torch

from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
)
from amsrr.policies.order9_active_knot_features import (
    ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES,
    ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES,
    ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES,
    ORDER9_CONTACT_SPACE_FEATURE_NAMES,
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9ActiveKnotLowLevelPolicyConfig,
    Order9ContactFeedbackLowLevelPolicyConfig,
    Order9ContactFeedbackPhaseConditionedActorCritic,
    Order9CategoricalContactNormalLowLevelPolicyConfig,
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
    Order9ContactSpacePhaseConditionedActorCritic,
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.task_spec import TaskType
from amsrr.training.order9_contact_space_action import (
    Order9ContactSpaceActionConfig,
    Order9ContactSpaceActionBasis,
    apply_order9_diagnostic_contact_normal_residual,
    apply_order9_contact_space_action,
    build_order9_deployable_contact_feedback_features,
)
from amsrr.training.order9_pi_l_contact_space_migration import (
    _reset_newly_activated_v7_action_outputs,
)


def test_contact_space_default_uses_common_twenty_mm_normal_authority() -> None:
    config = Order9ContactSpaceActionConfig()

    assert config.normal_translation_limit_m == pytest.approx(0.020)
    assert config.normal_translation_quantization_step_m is None


def test_contact_space_adapter_quantizes_only_physical_normal_translation() -> None:
    modules, joints, slots = 2, 2, 4
    contact_map = torch.zeros(modules * joints, slots * 6)
    contact_map[0, 0] = 1.0
    contact_map[1, 1] = 1.0
    basis = Order9ContactSpaceActionBasis(
        module_ids=(0, 1),
        local_joint_ids=("a", "b"),
        contact_slot_features=torch.zeros(
            slots, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
        ),
        contact_slot_owner_module_indices=torch.tensor([0, 1, -1, -1]),
        contact_slot_mask=torch.tensor([True, True, False, False]),
        contact_to_joint_position=contact_map,
        joint_nullspace_projector=torch.eye(modules * joints),
        centroidal_pose_projector=torch.eye(6),
        centroidal_joint_compensation=torch.zeros(modules * joints, 6),
        contact_action_scale=torch.tensor(
            [0.020, 0.002, 0.002, 0.020, 0.020, 0.020]
        ),
        selected_anchor_ids=(0, 1),
        selected_candidate_ids=(10, 11),
        normal_translation_quantization_step_m=0.0005,
    )
    config = Order9ActiveKnotLowLevelPolicyConfig(max_local_joint_slots=4)
    contact_action = torch.zeros(1, slots, 6)
    contact_action[0, 0, 0] = 0.612
    contact_action[0, 0, 1] = 0.625

    result = apply_order9_contact_space_action(
        normalized_global_action=torch.zeros(1, ORDER9_GLOBAL_ACTION_SIZE),
        normalized_joint_action=torch.zeros(
            1, modules, 3 * config.max_local_joint_slots
        ),
        normalized_contact_action=contact_action,
        phase_index=torch.tensor([3]),
        phase_progress=torch.tensor([0.5]),
        basis=basis,
        policy_config=config,
    )

    # 0.612 * 20 mm = 12.24 mm, which is deployed as exactly 12.0 mm.
    assert result.contact_joint_delta_rad[0, 0, 0] == pytest.approx(0.012)
    # Tangential translation remains continuous: 0.625 * 2 mm = 1.25 mm.
    assert result.contact_joint_delta_rad[0, 0, 1] == pytest.approx(0.00125)


def test_v7_migration_preserves_common_actor_and_zeroes_contact_mean() -> None:
    torch.manual_seed(9070)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    source = Order9MorphologyInvariantCompressionActorCritic(config)
    target = Order9ContactSpacePhaseConditionedActorCritic(config)
    fresh, unexpected = target.initialize_from_morphology_invariant_compression(source)
    assert fresh
    assert not unexpected
    source_state = source.state_dict()
    for name, value in target.state_dict().items():
        if name in source_state:
            assert torch.equal(value, source_state[name])

    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9070, module_count=2
    )
    batch = 2
    phase = torch.zeros(batch, config.phase_feature_dim)
    phase[:, list(TaskType).index(TaskType.OBJECT_GRASP_CARRY)] = 1.0
    phase[:, len(TaskType) + 1] = 1.0
    contact_mask = torch.zeros(batch, config.max_contact_slots, dtype=torch.bool)
    contact_mask[:, :2] = True
    owner = torch.full((batch, config.max_contact_slots), -1, dtype=torch.long)
    owner[:, 0] = 0
    owner[:, 1] = 1
    step = target.step(
        [graph] * batch,
        [None] * batch,
        torch.zeros(batch, len(ORDER3_ACTOR_FEATURE_NAMES)),
        torch.zeros(batch, ORDER9_GLOBAL_ACTION_SIZE),
        torch.zeros(batch, config.recurrent_hidden_dim),
        phase_features=phase,
        active_knot_features=torch.zeros(
            batch, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
        ),
        active_assignment_features=torch.zeros(
            batch,
            len(graph.modules),
            len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
        ),
        contact_slot_features=torch.zeros(
            batch, config.max_contact_slots, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
        ),
        contact_slot_owner_module_indices=owner,
        contact_slot_mask=contact_mask,
        deterministic=True,
    )
    assert torch.equal(
        step.contact_space_residual_action,
        torch.zeros_like(step.contact_space_residual_action),
    )
    assert torch.isfinite(step.log_prob).all()
    assert torch.isfinite(step.entropy).all()
    assert torch.allclose(
        step.log_prob,
        step.contact_log_prob
        + step.centroidal_log_prob
        + step.posture_log_prob,
    )
    assert torch.allclose(
        step.entropy,
        step.contact_entropy
        + step.centroidal_entropy
        + step.posture_entropy,
    )


def test_v7_command_boundary_reset_zeroes_previously_masked_outputs() -> None:
    torch.manual_seed(9071)
    config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    source = Order9MorphologyInvariantCompressionActorCritic(config)
    target = Order9ContactSpacePhaseConditionedActorCritic(config)
    target.initialize_from_morphology_invariant_compression(source)
    assert torch.count_nonzero(target.actor_mean.weight) > 0
    assert torch.count_nonzero(target.joint_decoder[2].weight) > 0

    reset_names = _reset_newly_activated_v7_action_outputs(target)

    assert set(reset_names) == {
        "actor_mean.weight",
        "actor_mean.bias",
        "actor_log_std",
        "joint_decoder.2.weight",
        "joint_decoder.2.bias",
        "joint_actor_log_std",
    }
    assert torch.count_nonzero(target.actor_mean.weight) == 0
    assert torch.count_nonzero(target.actor_mean.bias) == 0
    assert torch.count_nonzero(target.joint_decoder[2].weight) == 0
    assert torch.count_nonzero(target.joint_decoder[2].bias) == 0
    assert torch.equal(
        target.actor_log_std,
        torch.full_like(target.actor_log_std, config.joint_action_log_std_init),
    )
    assert torch.equal(
        target.joint_actor_log_std,
        torch.full_like(
            target.joint_actor_log_std, config.joint_action_log_std_init
        ),
    )


def test_diagnostic_contact_normal_residual_overrides_only_active_normals() -> None:
    slots = 4
    basis = Order9ContactSpaceActionBasis(
        module_ids=(0, 1),
        local_joint_ids=("a", "b"),
        contact_slot_features=torch.zeros(
            slots, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
        ),
        contact_slot_owner_module_indices=torch.tensor([0, 1, -1, -1]),
        contact_slot_mask=torch.tensor([True, True, False, False]),
        contact_to_joint_position=torch.zeros(4, slots * 6),
        joint_nullspace_projector=torch.eye(4),
        centroidal_pose_projector=torch.eye(6),
        centroidal_joint_compensation=torch.zeros(4, 6),
        contact_action_scale=torch.tensor(
            [0.01, 0.002, 0.002, 0.02, 0.02, 0.02]
        ),
        selected_anchor_ids=(0, 1),
        selected_candidate_ids=(10, 11),
    )
    action = torch.full((2, slots, 6), 0.25)

    result = apply_order9_diagnostic_contact_normal_residual(
        normalized_contact_action=action,
        inward_residual_m=torch.tensor([-0.002, 0.008]),
        basis=basis,
    )

    assert torch.allclose(
        result[:, :2, 0],
        torch.tensor([[-0.2, -0.2], [0.8, 0.8]]),
    )
    assert torch.equal(result[:, 2:, 0], action[:, 2:, 0])
    assert torch.equal(result[:, :, 1:], action[:, :, 1:])
    with pytest.raises(ValueError, match="exceeds action span"):
        apply_order9_diagnostic_contact_normal_residual(
            normalized_contact_action=action,
            inward_residual_m=torch.tensor([0.002, 0.011]),
            basis=basis,
        )


def test_v8_migration_preserves_v7_columns_and_zeroes_feedback_columns() -> None:
    torch.manual_seed(9072)
    source_config = Order9ActiveKnotLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    source = Order9ContactSpacePhaseConditionedActorCritic(source_config)
    target_config = Order9ContactFeedbackLowLevelPolicyConfig.from_dict(
        {
            **source_config.to_dict(),
            "contact_space_feature_dim": len(
                ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES
            ),
        }
    )
    target = Order9ContactFeedbackPhaseConditionedActorCritic(target_config)

    fresh, unexpected = target.initialize_from_contact_space(source)

    assert not unexpected
    assert fresh == [
        "contact_space_feature_encoder.0.weight[:, 30:38]"
    ]
    source_state = source.state_dict()
    target_state = target.state_dict()
    expanded = "contact_space_feature_encoder.0.weight"
    assert torch.equal(
        target_state[expanded][:, : len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)],
        source_state[expanded],
    )
    assert torch.count_nonzero(
        target_state[expanded][
            :, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES) :
        ]
    ) == 0
    for name, value in source_state.items():
        if name != expanded:
            assert torch.equal(target_state[name], value)


def test_v9_categorical_contact_normal_uses_exact_81_bin_action_and_replay() -> None:
    torch.manual_seed(9073)
    source_config = Order9ContactFeedbackLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    source = Order9ContactFeedbackPhaseConditionedActorCritic(source_config)
    target_config = Order9CategoricalContactNormalLowLevelPolicyConfig.from_dict(
        source_config.to_dict()
    )
    target = Order9CategoricalContactNormalPhaseConditionedActorCritic(
        target_config
    )
    fresh, unexpected = target.initialize_from_contact_feedback(source)
    assert fresh
    assert not unexpected
    assert all(name.startswith("contact_normal_category_logits.") for name in fresh)
    with torch.no_grad():
        target.contact_space_actor_mean[-1].bias[0] = torch.atanh(
            torch.tensor(0.60)
        )

    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    graph = RandomConnectedMorphologyDistribution(physical).sample(
        seed=9073, module_count=2
    )
    batch = 2
    phase = torch.zeros(batch, target_config.phase_feature_dim)
    phase[:, list(TaskType).index(TaskType.OBJECT_GRASP_CARRY)] = 1.0
    phase[:, len(TaskType) + 1] = 1.0
    contact_mask = torch.zeros(
        batch, target_config.max_contact_slots, dtype=torch.bool
    )
    contact_mask[:, :2] = True
    owner = torch.full(
        (batch, target_config.max_contact_slots), -1, dtype=torch.long
    )
    owner[:, 0] = 0
    owner[:, 1] = 1
    kwargs = {
        "phase_features": phase,
        "active_knot_features": torch.zeros(
            batch, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
        ),
        "active_assignment_features": torch.zeros(
            batch,
            len(graph.modules),
            len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
        ),
        "contact_slot_features": torch.zeros(
            batch,
            target_config.max_contact_slots,
            len(ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES),
        ),
        "contact_slot_owner_module_indices": owner,
        "contact_slot_mask": contact_mask,
    }
    inputs = (
        [graph] * batch,
        [None] * batch,
        torch.zeros(batch, len(ORDER3_ACTOR_FEATURE_NAMES)),
        torch.zeros(batch, ORDER9_GLOBAL_ACTION_SIZE),
        torch.zeros(batch, target_config.recurrent_hidden_dim),
    )
    deterministic = target.step(*inputs, **kwargs, deterministic=True)
    assert torch.equal(
        deterministic.contact_space_residual_action[:, :2, 0],
        torch.full((batch, 2), 0.60),
    )
    assert torch.equal(
        deterministic.contact_space_residual_action[:, 2:, 0],
        torch.zeros(batch, target_config.max_contact_slots - 2),
    )
    replay = target.step(
        *inputs,
        **kwargs,
        action=deterministic.action,
        joint_action=deterministic.joint_action,
        contact_space_residual_action=(
            deterministic.contact_space_residual_action
        ),
    )
    assert torch.allclose(replay.log_prob, deterministic.log_prob, atol=1.0e-5)
    assert torch.isfinite(replay.entropy).all()

    torch.manual_seed(9074)
    sampled = target.step(*inputs, **kwargs)
    category_index = torch.round(
        (sampled.contact_space_residual_action[:, :2, 0] + 1.0) / 0.025
    )
    reconstructed = -1.0 + 0.025 * category_index
    assert torch.allclose(
        sampled.contact_space_residual_action[:, :2, 0], reconstructed
    )
    sampled_replay = target.step(
        *inputs,
        **kwargs,
        action=sampled.action,
        joint_action=sampled.joint_action,
        contact_space_residual_action=sampled.contact_space_residual_action,
    )
    assert torch.allclose(sampled_replay.log_prob, sampled.log_prob, atol=1.0e-5)

    with torch.no_grad():
        categorical_final = target.contact_normal_category_logits[-1]
        assert isinstance(categorical_final, torch.nn.Linear)
        categorical_final.bias.zero_()
        categorical_final.bias[0] = 100.0
    saturated = target.step(*inputs, **kwargs, deterministic=True)
    assert torch.equal(
        saturated.contact_space_residual_action[:, :2, 0],
        -torch.ones(batch, 2),
    )
    saturated_replay = target.step(
        *inputs,
        **kwargs,
        action=saturated.action,
        joint_action=saturated.joint_action,
        contact_space_residual_action=saturated.contact_space_residual_action,
    )
    assert torch.allclose(
        saturated_replay.log_prob, saturated.log_prob, atol=1.0e-5
    )


def test_v9_categorical_contact_normal_uniform_window_prior_has_full_support() -> None:
    config = Order9CategoricalContactNormalLowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
        contact_normal_category_prior_mode="uniform_window",
        contact_normal_category_uniform_window_min_normalized=0.50,
        contact_normal_category_uniform_window_max_normalized=1.00,
        contact_normal_category_uniform_outside_logit_penalty=4.0,
    )
    policy = Order9CategoricalContactNormalPhaseConditionedActorCritic(config)
    values = torch.linspace(-1.0, 1.0, 81)
    logits = policy._contact_normal_prior_logits(
        values, torch.tensor([[[0.0]]])
    )
    probabilities = torch.softmax(logits, dim=-1)
    in_window = (values >= 0.50) & (values <= 1.00)

    assert torch.equal(logits[in_window], torch.zeros_like(logits[in_window]))
    assert torch.equal(
        logits[~in_window], torch.full_like(logits[~in_window], -4.0)
    )
    assert torch.all(probabilities > 0.0)
    assert probabilities[in_window].sum() > 0.94


def test_deployable_contact_feedback_projects_signed_joint_load() -> None:
    slots = 4
    contact_map = torch.zeros(4, slots * 6)
    contact_map[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    contact_map[:, 6] = torch.tensor([0.0, 0.0, -2.0, 0.0])
    basis = Order9ContactSpaceActionBasis(
        module_ids=(0, 1),
        local_joint_ids=("a", "b"),
        contact_slot_features=torch.zeros(
            slots, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
        ),
        contact_slot_owner_module_indices=torch.tensor([0, 1, -1, -1]),
        contact_slot_mask=torch.tensor([True, True, False, False]),
        contact_to_joint_position=contact_map,
        joint_nullspace_projector=torch.eye(4),
        centroidal_pose_projector=torch.eye(6),
        centroidal_joint_compensation=torch.zeros(4, 6),
        contact_action_scale=torch.tensor(
            [0.01, 0.002, 0.002, 0.02, 0.02, 0.02]
        ),
        selected_anchor_ids=(0, 1),
        selected_candidate_ids=(10, 11),
    )
    output = build_order9_deployable_contact_feedback_features(
        signed_surface_distance_m=torch.tensor([[0.005, -0.002]]),
        relative_twist_contact=torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.0] * 6]]
        ),
        signed_hardware_joint_load_nm=torch.tensor(
            [[[2.0, 0.0], [-3.0, 0.0]]]
        ),
        basis=basis,
    )

    assert output.shape == (1, slots, len(ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES))
    assert output[0, 0, 0] == pytest.approx(0.5)
    assert output[0, 1, 0] == pytest.approx(-0.2)
    assert output[0, 0, 1:7].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    assert output[0, 0, 7] == pytest.approx(torch.log1p(torch.tensor(2.0)).item())
    assert output[0, 1, 7] == pytest.approx(torch.log1p(torch.tensor(3.0)).item())
    assert torch.equal(output[:, 2:], torch.zeros_like(output[:, 2:]))


def test_contact_space_adapter_separates_contact_com_and_posture_paths() -> None:
    modules, joints, slots = 2, 2, 4
    contact_map = torch.zeros(modules * joints, slots * 6)
    contact_map[0, 0] = 1.0
    nullspace = torch.diag(torch.tensor([0.0, 1.0, 1.0, 1.0]))
    centroidal_projector = torch.diag(
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    centroidal_compensation = torch.zeros(modules * joints, 6)
    centroidal_compensation[0, 1] = 0.5
    basis = Order9ContactSpaceActionBasis(
        module_ids=(0, 1),
        local_joint_ids=("a", "b"),
        contact_slot_features=torch.zeros(
            slots, len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
        ),
        contact_slot_owner_module_indices=torch.tensor([0, 1, -1, -1]),
        contact_slot_mask=torch.tensor([True, True, False, False]),
        contact_to_joint_position=contact_map,
        joint_nullspace_projector=nullspace,
        centroidal_pose_projector=centroidal_projector,
        centroidal_joint_compensation=centroidal_compensation,
        contact_action_scale=torch.tensor([0.01, 0.002, 0.002, 0.02, 0.02, 0.02]),
        selected_anchor_ids=(0, 1),
        selected_candidate_ids=(10, 11),
    )
    config = Order9ActiveKnotLowLevelPolicyConfig(max_local_joint_slots=4)
    global_action = torch.full((1, ORDER9_GLOBAL_ACTION_SIZE), 0.5)
    joint_action = torch.zeros(1, modules, 3 * config.max_local_joint_slots)
    joint_action[0, 0, 0] = 0.8
    joint_action[0, 0, 1] = 0.4
    joint_action[0, 0, config.max_local_joint_slots] = 0.6
    contact_action = torch.zeros(1, slots, 6)
    contact_action[0, 0, 0] = 1.0

    approach = apply_order9_contact_space_action(
        normalized_global_action=global_action,
        normalized_joint_action=joint_action,
        normalized_contact_action=contact_action,
        phase_index=torch.tensor([0]),
        phase_progress=torch.tensor([0.5]),
        basis=basis,
        policy_config=config,
    )
    assert torch.equal(approach.global_action[:, :12], global_action[:, :12])
    assert torch.equal(approach.global_action[:, 12:], torch.zeros(1, 6))
    assert approach.joint_action[0, 0, 0] == joint_action[0, 0, 0]
    assert torch.equal(
        approach.joint_action[:, :, 2 * config.max_local_joint_slots :],
        torch.zeros_like(
            approach.joint_action[:, :, 2 * config.max_local_joint_slots :]
        ),
    )

    contact = apply_order9_contact_space_action(
        normalized_global_action=global_action,
        normalized_joint_action=joint_action,
        normalized_contact_action=contact_action,
        phase_index=torch.tensor([3]),
        phase_progress=torch.tensor([0.5]),
        basis=basis,
        policy_config=config,
    )
    assert contact.contact_constraint_weight.item() == 1.0
    assert contact.global_action[0, 0] == 0.0
    assert contact.posture_nullspace_joint_delta_rad[0, 0, 0] == 0.0
    assert torch.isclose(
        contact.contact_joint_delta_rad[0, 0, 0], torch.tensor(0.01)
    )
    assert torch.equal(contact.global_action[:, 12:], torch.zeros(1, 6))
