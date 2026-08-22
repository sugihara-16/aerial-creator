from __future__ import annotations

from collections import Counter

import pytest
import torch

from amsrr.training.order9_tensor_pi_l_ppo import (
    ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES,
    _SequencePPOTerms,
    _Sequence,
    _aggregate_advantage_metadata,
    _aggregate_exact_replay,
    _boundary_phase_masks,
    _categorical_preload_deficit_teacher_loss,
    _contact_coordinate_group_log_probs,
    _factorized_clipped_actor_loss,
    _maximum_topology_phase_kl,
    _mean_pcgrad,
    _merge_boundary_incomplete_sequence_batches,
    _normalize_advantages,
    _order9_contact_head_named_parameters,
    _phase_balanced_sequences,
    _phase_kl_metrics,
    _squashed_coordinate_log_prob_and_entropy,
    _transition_budget_sequence_batches,
    _topology_stratified_sequence_step,
    _weighted_contact_coordinate_actor_loss,
    order9_privileged_compression_teacher_direction,
)
from amsrr.training.order9_contact_residual_warm_start import (
    Order9ContactResidualWarmStartConfig,
    configure_order9_contact_residual_warm_start_parameters,
    order9_contact_residual_warm_start_target,
)
from amsrr.policies.order9_low_level_policy import (
    Order9ContactSpacePhaseConditionedActorCritic,
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_curriculum import (
    Order9C3BoundaryFineTuneConfig,
    Order9PPOOptimizationConfig,
)
from amsrr.training.order9_factorized_actor_credit import (
    ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS,
)


def test_phase_local_advantage_normalization_is_independent_per_phase() -> None:
    advantages = torch.tensor([[1.0, 100.0], [3.0, 104.0], [5.0, 108.0]])
    valid = torch.ones_like(advantages, dtype=torch.bool)
    phases = torch.tensor([[0, 3], [0, 3], [0, 3]])

    normalized, metadata = _normalize_advantages(
        advantages,
        valid=valid,
        phase_index=phases,
        phase_local=True,
    )

    assert normalized[:, 0].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert normalized[:, 1].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert normalized[:, 0].std(unbiased=False).item() == pytest.approx(1.0)
    assert normalized[:, 1].std(unbiased=False).item() == pytest.approx(1.0)
    assert metadata["advantage_normalization_scope"] == "actor_phase"
    assert metadata["advantage_phase_sample_count"] == {"0": 3, "3": 3}


@pytest.mark.parametrize("dimension", (12, 18))
def test_contracted_global_log_prob_only_moves_enabled_coordinates(
    dimension: int,
) -> None:
    mean = torch.nn.Parameter(torch.zeros(2, 18))
    action = torch.full((2, 18), 0.2)
    log_std = torch.nn.Parameter(torch.full((18,), -2.0))

    log_prob, entropy = _squashed_coordinate_log_prob_and_entropy(
        bounded_action=action[:, :dimension],
        bounded_mean=torch.tanh(mean[:, :dimension]),
        log_std=log_std[:dimension],
    )
    loss = -(log_prob + 0.01 * entropy).mean()
    loss.backward()

    assert torch.count_nonzero(mean.grad[:, :dimension]) > 0
    assert torch.count_nonzero(log_std.grad[:dimension]) > 0
    assert torch.count_nonzero(mean.grad[:, dimension:]) == 0
    assert torch.count_nonzero(log_std.grad[dimension:]) == 0


@pytest.mark.parametrize("credited_channel", ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
def test_factorized_actor_loss_routes_gradient_to_matching_density_only(
    credited_channel: str,
) -> None:
    parameters = {
        name: torch.nn.Parameter(torch.zeros(4))
        for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    }
    advantages = {
        name: torch.ones(4) if name == credited_channel else torch.zeros(4)
        for name in ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    }

    loss, component_losses = _factorized_clipped_actor_loss(
        new_log_probs=parameters,
        old_log_probs={name: torch.zeros(4) for name in parameters},
        advantages=advantages,
        mask=torch.ones(4, dtype=torch.bool),
        clip_ratio=0.2,
    )
    loss.backward()

    assert set(component_losses) == set(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
    for name, parameter in parameters.items():
        assert parameter.grad is not None
        if name == credited_channel:
            assert torch.count_nonzero(parameter.grad) > 0
        else:
            assert torch.count_nonzero(parameter.grad) == 0


def test_contact_coordinate_log_probs_reconstruct_contact_density() -> None:
    mean = torch.nn.Parameter(torch.zeros(3, 2, 6))
    action = torch.full_like(mean, 0.2)
    slot_mask = torch.tensor(
        [[True, True], [True, False], [False, False]], dtype=torch.bool
    )

    groups = _contact_coordinate_group_log_probs(
        bounded_action=action,
        bounded_mean=torch.tanh(mean),
        log_std=torch.full((6,), -2.0),
        slot_mask=slot_mask,
    )

    assert set(groups) == {
        "normal_translation",
        "tangential_translation",
        "rotation",
    }
    total = torch.stack(tuple(groups.values()), dim=0).sum(dim=0)
    all_coordinates, _ = _squashed_coordinate_log_prob_and_entropy(
        bounded_action=action.reshape(-1, 6),
        bounded_mean=torch.tanh(mean).reshape(-1, 6),
        log_std=torch.full((6,), -2.0),
    )
    expected = (all_coordinates.reshape(3, 2) * slot_mask).sum(dim=-1)
    assert torch.allclose(total, expected)

    (-groups["normal_translation"].sum()).backward()
    assert torch.count_nonzero(mean.grad[..., 0]) > 0
    assert torch.count_nonzero(mean.grad[..., 1:]) == 0


def test_contact_normal_translation_actor_loss_weight_scales_only_normal() -> None:
    losses = {
        "normal_translation": torch.nn.Parameter(torch.tensor(1.0)),
        "tangential_translation": torch.nn.Parameter(torch.tensor(1.0)),
        "rotation": torch.nn.Parameter(torch.tensor(1.0)),
    }

    combined = _weighted_contact_coordinate_actor_loss(
        losses,
        normal_translation_weight=3.0,
    )
    combined.backward()

    assert combined.item() == pytest.approx(5.0 / 3.0)
    assert losses["normal_translation"].grad.item() == pytest.approx(1.0)
    assert losses["tangential_translation"].grad.item() == pytest.approx(1.0 / 3.0)
    assert losses["rotation"].grad.item() == pytest.approx(1.0 / 3.0)


def test_categorical_preload_deficit_teacher_routes_only_selected_slots() -> None:
    logits = torch.nn.Parameter(torch.zeros(3, 2, 81))
    category_index = torch.tensor([44, 48, 52])
    environment_mask = torch.tensor([True, False, True])
    contact_slot_mask = torch.tensor(
        [[True, False], [True, True], [True, True]]
    )
    target_phase_mask = torch.tensor([True, True, False])

    loss, sample_count = _categorical_preload_deficit_teacher_loss(
        category_logits=torch.log_softmax(logits, dim=-1),
        category_index=category_index,
        environment_mask=environment_mask,
        contact_slot_mask=contact_slot_mask,
        target_phase_mask=target_phase_mask,
    )
    loss.backward()

    assert sample_count == 1
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[1:]) == 0


def test_contact_head_extra_parameter_boundary_excludes_shared_actor() -> None:
    policy = Order9ContactSpacePhaseConditionedActorCritic()

    selected = dict(_order9_contact_head_named_parameters(policy))

    assert selected
    assert all(
        name.startswith(ORDER9_CONTACT_HEAD_PARAMETER_PREFIXES)
        for name in selected
    )
    assert any(name.startswith("contact_space_feature_encoder.") for name in selected)
    assert any(name.startswith("contact_space_actor_mean.") for name in selected)
    assert "contact_space_actor_log_std" in selected
    assert not any(name.startswith("fusion.") for name in selected)
    assert not any(name.startswith("recurrent.") for name in selected)
    assert not any(name.startswith("actor_mean.") for name in selected)
    assert not any(name.startswith("joint_decoder.") for name in selected)
    assert not any(name.startswith("critic.") for name in selected)


def test_contact_residual_warm_start_target_is_underforce_only() -> None:
    parent = torch.tensor([-0.2, 0.1, 0.8])
    direction = torch.tensor([-0.5, 0.5, 1.0])

    target = order9_contact_residual_warm_start_target(
        parent,
        direction,
        action_step=0.4,
        maximum_action=0.9,
    )

    assert target.tolist() == pytest.approx([-0.2, 0.3, 0.9])


def test_contact_residual_warm_start_freezes_policy_prior() -> None:
    policy = Order9MorphologyInvariantCompressionActorCritic()

    trainable = configure_order9_contact_residual_warm_start_parameters(policy)

    assert trainable
    assert trainable == ("contact_compression_module_count_bias",)
    assert {
        name for name, value in policy.named_parameters() if value.requires_grad
    } == set(trainable)
    Order9ContactResidualWarmStartConfig().validate()


def test_contact_residual_warm_start_keeps_joint_residual_branch_frozen() -> None:
    policy = Order9MorphologyInvariantCompressionActorCritic()

    configure_order9_contact_residual_warm_start_parameters(policy)

    assert not any(
        parameter.requires_grad
        for parameter in policy.contact_residual_decoder.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in policy.contact_compression_actor_mean.parameters()
    )
    assert policy.contact_compression_module_count_bias.requires_grad
    assert not policy.contact_compression_actor_log_std.requires_grad


def test_phase_local_advantage_normalization_zero_centers_singleton_phase() -> None:
    advantages = torch.tensor([[1.0], [3.0], [9.0]])
    valid = torch.ones_like(advantages, dtype=torch.bool)
    phases = torch.tensor([[0], [0], [5]])

    normalized, metadata = _normalize_advantages(
        advantages,
        valid=valid,
        phase_index=phases,
        phase_local=True,
    )

    assert normalized[:2, 0].tolist() == pytest.approx([-1.0, 1.0])
    assert normalized[2, 0].item() == 0.0
    assert metadata["advantage_phase_sample_count"] == {"0": 2, "5": 1}
    assert metadata["advantage_singleton_phases_zero_centered"] == [5]


def test_state_inheritance_target_only_batch_uses_external_anchor_shards() -> None:
    phases = torch.tensor([1, 2, 4, 1])

    target, anchor = _boundary_phase_masks(phases, [1, 2, 3, 4], allow_target_only=True)

    assert target.all()
    assert not anchor.any()
    with pytest.raises(SchemaValidationError, match="target and anchor"):
        _boundary_phase_masks(phases, [1, 2, 3, 4], allow_target_only=False)


def test_phase_balanced_sequence_sampling_oversamples_rare_phases() -> None:
    sequences = [
        _Sequence(environment=index, start=0, length=16, phase_index=0)
        for index in range(6)
    ]
    sequences += [_Sequence(environment=6, start=0, length=16, phase_index=3)]
    sequences += [
        _Sequence(environment=7 + index, start=0, length=16, phase_index=5)
        for index in range(2)
    ]

    first = _phase_balanced_sequences(sequences, seed=17)
    second = _phase_balanced_sequences(sequences, seed=17)

    assert first == second
    counts = Counter(sequence.phase_index for sequence in first)
    assert counts == {0: 3, 3: 3, 5: 3}


def test_transition_budget_sequence_batches_use_transition_count() -> None:
    sequences = [
        _Sequence(environment=0, start=0, length=16, phase_index=1),
        _Sequence(environment=0, start=16, length=3, phase_index=3),
        _Sequence(environment=1, start=0, length=2, phase_index=1),
        _Sequence(environment=1, start=2, length=15, phase_index=3),
    ]

    batches = _transition_budget_sequence_batches(sequences, transition_budget=20)

    assert batches == [sequences[:2], sequences[2:]]
    assert [sum(item.length for item in batch) for batch in batches] == [19, 17]
    assert [item for batch in batches for item in batch] == sequences


def test_transition_budget_sequence_batches_keep_oversized_sequence() -> None:
    sequence = _Sequence(environment=0, start=0, length=16, phase_index=1)

    batches = _transition_budget_sequence_batches([sequence], transition_budget=8)

    assert batches == [[sequence]]


def test_boundary_batch_repair_merges_non_target_tail_without_data_loss() -> None:
    target = _Sequence(environment=0, start=0, length=16, phase_index=1)
    anchor = _Sequence(environment=1, start=0, length=16, phase_index=3)
    tail = _Sequence(environment=2, start=0, length=7, phase_index=3)

    repaired = _merge_boundary_incomplete_sequence_batches(
        [[target, anchor], [tail]],
        target_phase_indices=[1, 4, 5, 6],
        allow_target_only=False,
    )

    assert repaired == [[target, anchor, tail]]


def test_boundary_batch_repair_keeps_continuous_target_only_batch() -> None:
    first = _Sequence(environment=0, start=0, length=16, phase_index=1)
    second = _Sequence(environment=1, start=0, length=9, phase_index=4)

    repaired = _merge_boundary_incomplete_sequence_batches(
        [[first], [second]],
        target_phase_indices=[1, 4, 5, 6],
        allow_target_only=True,
    )

    assert repaired == [[first], [second]]


def test_phase_local_kl_reports_worst_phase() -> None:
    old = torch.zeros(4)
    new = torch.tensor([0.0, 0.0, 0.5, 0.5])
    phases = torch.tensor([0, 0, 7, 7])

    metrics = _phase_kl_metrics(
        new_log_prob=new,
        old_log_prob=old,
        phase_index=phases,
    )

    assert metrics["approximate_kl_phase_0"] == pytest.approx(0.0)
    assert metrics["approximate_kl_phase_7"] > 0.0
    assert metrics["maximum_phase_kl"] == metrics["approximate_kl_phase_7"]


def test_boundary_topology_phase_kl_selects_only_requested_phases() -> None:
    rows = {
        "m02": {
            "approximate_kl_phase_0": 0.003,
            "approximate_kl_phase_6": 0.02,
            "maximum_phase_kl": 0.02,
        },
        "m08": {
            "approximate_kl_phase_0": 0.006,
            "approximate_kl_phase_6": 0.01,
            "maximum_phase_kl": 0.01,
        },
    }

    assert _maximum_topology_phase_kl(rows, [0]) == pytest.approx(0.006)
    assert _maximum_topology_phase_kl(rows, [6]) == pytest.approx(0.02)


def test_boundary_hard_kl_violation_rolls_back_optimizer_step(monkeypatch) -> None:
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    before = policy.weight.detach().clone()

    def fake_terms(*args, **kwargs):
        del args, kwargs
        new = torch.stack((policy.weight.reshape(()), policy.weight.reshape(())))
        old = torch.zeros_like(new)
        phase = torch.tensor([6, 0])
        target = torch.tensor([True, False])
        return _SequencePPOTerms(
            total=policy.weight.sum(),
            new_log_prob=new,
            old_log_prob=old,
            new_values=torch.zeros(2),
            advantages=torch.ones(2),
            returns=torch.zeros(2),
            entropy=torch.ones(2),
            phase_index=phase,
            target_phase_mask=target,
            non_target_parent_kl=policy.weight.sum() * 0.0,
            privileged_compression_teacher_loss=(policy.weight.sum() * 0.0),
            privileged_compression_teacher_sample_count=0,
            privileged_wrench_satisfied_parent_kl=(policy.weight.sum() * 0.0),
            privileged_wrench_satisfied_sample_count=0,
            contact_normal_preload_deficit_teacher_loss=(
                policy.weight.sum() * 0.0
            ),
            contact_normal_preload_deficit_teacher_sample_count=0,
        )

    monkeypatch.setattr(
        "amsrr.training.order9_tensor_pi_l_ppo._sequence_ppo_terms",
        fake_terms,
    )
    monkeypatch.setattr(
        "amsrr.training.order9_tensor_pi_l_ppo._post_step_topology_kl_metrics",
        lambda *args, **kwargs: {
            "approximate_kl": 0.003,
            "approximate_kl_phase_0": 0.011,
            "approximate_kl_phase_6": 0.001,
            "maximum_phase_kl": 0.011,
            "maximum_topology_phase_kl": 0.011,
            "maximum_target_topology_phase_kl": 0.001,
            "maximum_non_target_topology_phase_kl": 0.011,
        },
    )

    metrics = _topology_stratified_sequence_step(
        policy,
        optimizer,
        [("m02", object(), [object()], torch.ones(1), torch.ones(1))],
        config=Order9PPOOptimizationConfig(),
        boundary_fine_tune=Order9C3BoundaryFineTuneConfig(enabled=True),
        target_phase_indices=(6,),
    )

    assert torch.equal(policy.weight.detach(), before)
    assert metrics["optimizer_step_applied"] == 0.0
    assert metrics["optimizer_step_rolled_back"] == 1.0
    assert metrics["boundary_non_target_kl_violation"] == 1.0


def test_privileged_compression_teacher_tracks_wrench_range_violation() -> None:
    lower = (
        torch.tensor(
            [
                [[0.0, -6.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0, 0.0]],
            ]
        )
        .expand(3, -1, -1)
        .clone()
    )
    upper = (
        torch.tensor(
            [
                [[0.0, -4.0, 0.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0, 0.0, 0.0]],
            ]
        )
        .expand(3, -1, -1)
        .clone()
    )
    actual = torch.zeros_like(lower)
    actual[0, :, 1] = torch.tensor([-2.0, 3.0])
    actual[1, :, 1] = torch.tensor([-8.0, 7.0])
    actual[2, :, 1] = torch.tensor([-5.0, 5.0])

    direction, mask = order9_privileged_compression_teacher_direction(
        selected_contact_wrenches_contact=actual,
        wrench_lower_contact=lower,
        wrench_upper_contact=upper,
        wrench_bound_mask=torch.ones((3, 2), dtype=torch.bool),
    )

    assert direction[0].item() > 0.0
    assert direction[1].item() < 0.0
    assert direction[2].item() == pytest.approx(0.0)
    assert mask.tolist() == [True, True, False]


def test_privileged_compression_teacher_ignores_unbounded_anchors() -> None:
    lower = torch.tensor(
        [[[0.0, -6.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    upper = torch.tensor(
        [[[0.0, -4.0, 0.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    actual = torch.zeros_like(lower)

    direction, mask = order9_privileged_compression_teacher_direction(
        selected_contact_wrenches_contact=actual,
        wrench_lower_contact=lower,
        wrench_upper_contact=upper,
        wrench_bound_mask=torch.zeros((1, 2), dtype=torch.bool),
    )

    assert direction.tolist() == [0.0]
    assert mask.tolist() == [False]


def test_pcgrad_removes_pairwise_gradient_conflict() -> None:
    first = (torch.tensor([1.0, 0.0]),)
    second = (torch.tensor([-1.0, 1.0]),)

    projected = _mean_pcgrad((first, second))[0]

    # The raw mean is [0, 0.5].  PCGrad preserves a useful first-axis
    # component instead of allowing the opposing topology to cancel it.
    assert projected[0].item() > 0.0
    assert projected[1].item() > 0.0


def test_pcgrad_projection_mask_prevents_critic_from_hiding_actor_conflict() -> None:
    first = (torch.tensor([1.0]), torch.tensor([100.0]))
    second = (torch.tensor([-1.0]), torch.tensor([100.0]))

    projected = _mean_pcgrad((first, second), projection_parameter_mask=(True, False))

    assert projected[0].item() == pytest.approx(0.0)
    assert projected[1].item() == pytest.approx(100.0)


def test_topology_exact_replay_aggregation_sums_records_and_takes_maxima() -> None:
    common = {
        "maximum_component_log_prob_sum_replay_error": 0.0,
        "maximum_contact_coordinate_log_prob_sum_replay_error": 0.0,
        "maximum_value_replay_error": 0.0,
        "maximum_recurrent_replay_error": 0.0,
        "maximum_stored_recurrent_continuity_error": 0.0,
        "maximum_stored_previous_action_continuity_error": 0.0,
        "exact_replay_absolute_tolerance": 0.0025,
        "exact_replay_log_prob_tolerance": 0.0025,
        "exact_replay_value_tolerance": 5.0e-5,
        "exact_replay_recurrent_tolerance": 5.0e-4,
        "exact_replay_continuity_tolerance": 2.0e-5,
        "exact_replay_collection_shaped_time_slices": True,
        "exact_replay_timestep_batched_active_sequences": False,
        "exact_replay_record_invariant_cache": True,
    }
    aggregate = _aggregate_exact_replay(
        {
            "m02": {
                **common,
                "exact_replay_record_count": 32,
                "maximum_log_prob_replay_error": 1.0e-6,
            },
            "m08": {
                **common,
                "exact_replay_record_count": 64,
                "maximum_log_prob_replay_error": 3.0e-6,
            },
        }
    )

    assert aggregate["exact_replay_record_count"] == 96
    assert aggregate["maximum_log_prob_replay_error"] == pytest.approx(3.0e-6)
    assert aggregate["exact_replay_topology_count"] == 2


def test_topology_advantage_metadata_uses_weighted_phase_fraction() -> None:
    aggregate = _aggregate_advantage_metadata(
        {
            "m02": {
                "advantage_phase_sample_count": {"0": 2},
                "advantage_phase_positive_fraction_before_normalization": {"0": 0.5},
            },
            "m08": {
                "advantage_phase_sample_count": {"0": 6},
                "advantage_phase_positive_fraction_before_normalization": {"0": 1.0},
            },
        }
    )

    assert aggregate["advantage_normalization_scope"] == "topology_actor_phase"
    assert aggregate["advantage_phase_sample_count"] == {"0": 8}
    assert aggregate["advantage_phase_positive_fraction_before_normalization"][
        "0"
    ] == pytest.approx(0.875)
