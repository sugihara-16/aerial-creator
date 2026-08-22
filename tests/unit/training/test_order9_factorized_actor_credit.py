from __future__ import annotations

import pytest
import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_factorized_actor_credit import (
    ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS,
    ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS,
    factorize_order9_contact_coordinate_reward,
    factorize_order9_actor_reward,
)
from amsrr.training.order9_tensor_reward import ORDER9_TENSOR_REWARD_TERM_NAMES


def test_factorized_actor_credit_reconstructs_scalar_reward_exactly() -> None:
    terms = torch.zeros(2, 2, len(ORDER9_TENSOR_REWARD_TERM_NAMES))
    index = {name: offset for offset, name in enumerate(ORDER9_TENSOR_REWARD_TERM_NAMES)}
    terms[..., index["weighted_grasp_maintenance"]] = 0.6
    terms[..., index["weighted_object_goal_progress"]] = 0.9
    terms[..., index["weighted_collision_penalty"]] = -0.7
    terms[..., index["weighted_energy_penalty"]] = -0.4
    terms[..., index["terminal_success_bonus"]] = 3.0
    reward = terms.sum(dim=-1)
    valid = torch.tensor([[True, True], [True, False]])

    channels, maximum_error = factorize_order9_actor_reward(
        reward=reward,
        reward_terms=terms,
        reward_term_names=ORDER9_TENSOR_REWARD_TERM_NAMES,
        valid=valid,
    )

    assert tuple(channels) == ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS
    reconstructed = torch.stack(tuple(channels.values()), dim=-1).sum(dim=-1)
    assert torch.allclose(reconstructed[valid], reward[valid])
    assert reconstructed[~valid].eq(0.0).all()
    assert maximum_error == pytest.approx(0.0, abs=1.0e-6)
    # Shared terminal success contributes 1.0 to every channel.  Dense terms
    # otherwise reach only their physical role, with energy split in half.
    assert channels["contact"][0, 0].item() == pytest.approx(1.6)
    assert channels["centroidal"][0, 0].item() == pytest.approx(1.7)
    assert channels["posture"][0, 0].item() == pytest.approx(0.1)


def test_factorized_actor_credit_rejects_unknown_reward_contract() -> None:
    terms = torch.zeros(1, 1, len(ORDER9_TENSOR_REWARD_TERM_NAMES))

    with pytest.raises(SchemaValidationError, match="reward-term order"):
        factorize_order9_actor_reward(
            reward=torch.zeros(1, 1),
            reward_terms=terms,
            reward_term_names=tuple(reversed(ORDER9_TENSOR_REWARD_TERM_NAMES)),
            valid=torch.ones(1, 1, dtype=torch.bool),
        )


def test_contact_coordinate_credit_reconstructs_contact_reward() -> None:
    terms = torch.zeros(2, 2, len(ORDER9_TENSOR_REWARD_TERM_NAMES))
    index = {name: offset for offset, name in enumerate(ORDER9_TENSOR_REWARD_TERM_NAMES)}
    terms[..., index["weighted_grasp_maintenance"]] = 0.9
    terms[..., index["weighted_slip_penalty"]] = -0.6
    terms[..., index["weighted_wrench_range_violation_penalty"]] = -0.3
    terms[..., index["weighted_normal_contact_quality_penalty"]] = -0.6
    terms[..., index["terminal_failure_penalty"]] = -9.0
    reward = terms.sum(dim=-1)
    valid = torch.tensor([[True, True], [True, False]])
    outer, _ = factorize_order9_actor_reward(
        reward=reward,
        reward_terms=terms,
        reward_term_names=ORDER9_TENSOR_REWARD_TERM_NAMES,
        valid=valid,
    )

    channels, maximum_error = factorize_order9_contact_coordinate_reward(
        contact_reward=outer["contact"],
        reward_terms=terms,
        reward_term_names=ORDER9_TENSOR_REWARD_TERM_NAMES,
        valid=valid,
    )

    assert tuple(channels) == ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS
    reconstructed = torch.stack(tuple(channels.values()), dim=-1).sum(dim=-1)
    assert torch.allclose(reconstructed[valid], outer["contact"][valid])
    assert reconstructed[~valid].eq(0.0).all()
    assert maximum_error == pytest.approx(0.0, abs=1.0e-6)
    # Contact share of terminal failure is -3 and is split evenly.  Grasp is
    # normal-only, slip tangential-only, and wrench diagnostic is shared.
    assert channels["normal_translation"][0, 0].item() == pytest.approx(-0.8)
    assert channels["tangential_translation"][0, 0].item() == pytest.approx(-1.7)
    assert channels["rotation"][0, 0].item() == pytest.approx(-1.1)
