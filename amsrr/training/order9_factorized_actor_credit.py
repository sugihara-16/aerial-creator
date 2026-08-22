from __future__ import annotations

"""Action-role-specific reward routing for the Order 9 contact-space actor.

The deployed ``PolicyCommand`` is unchanged.  This module only partitions the
already-recorded scalar reward into contact, centroidal, and posture channels
for training-time credit assignment.  The partition is conservative: the
three channels sum exactly to the original reward at every transition.
"""

from collections.abc import Mapping, Sequence

import torch

from amsrr.schemas.common import SchemaValidationError


ORDER9_FACTORIZED_ACTOR_CREDIT_VERSION = (
    "order9_contact_centroidal_posture_actor_credit_v2_normal_quality"
)
ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS = (
    "contact",
    "centroidal",
    "posture",
)
ORDER9_CONTACT_COORDINATE_CREDIT_VERSION = (
    "order9_contact_normal_tangential_rotational_credit_v2_normal_quality"
)
ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS = (
    "normal_translation",
    "tangential_translation",
    "rotation",
)
ORDER9_CONTACT_COORDINATE_ACTION_INDICES: Mapping[str, tuple[int, ...]] = {
    "normal_translation": (0,),
    "tangential_translation": (1, 2),
    "rotation": (3, 4, 5),
}

# Fixed dense-term routing.  Terminal task outcome is intentionally shared:
# it is the only part of the objective that genuinely depends on all three
# physical roles.  Energy is split because the stored term is itself the
# equal mean of rotor and joint energy.
ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS: Mapping[
    str, tuple[float, float, float]
] = {
    "weighted_object_goal_progress": (0.0, 1.0, 0.0),
    "weighted_object_pose_accuracy": (0.0, 1.0, 0.0),
    "weighted_grasp_maintenance": (1.0, 0.0, 0.0),
    "weighted_wrench_range_violation_penalty": (1.0, 0.0, 0.0),
    "weighted_normal_contact_quality_penalty": (1.0, 0.0, 0.0),
    "weighted_centroidal_stability": (0.0, 1.0, 0.0),
    "weighted_energy_penalty": (0.0, 0.5, 0.5),
    "weighted_qp_residual_penalty": (0.0, 1.0, 0.0),
    "weighted_slip_penalty": (1.0, 0.0, 0.0),
    "weighted_collision_penalty": (0.0, 0.0, 1.0),
    "weighted_actuator_saturation_penalty": (0.0, 1.0, 0.0),
    "terminal_success_bonus": (1.0 / 3.0,) * 3,
    "terminal_failure_penalty": (1.0 / 3.0,) * 3,
}

# Split only the contact channel one level further.  The rows sum to the
# contact weight above, so this changes credit assignment without changing the
# scalar task objective.  Contact existence is caused most directly by inward
# translation, while the existing linear-slip term belongs to tangential
# translation.  The currently disabled wrench-box diagnostic and sparse task
# outcome are shared because they can depend on all three contact roles.
ORDER9_CONTACT_COORDINATE_CREDIT_TERM_WEIGHTS: Mapping[
    str, tuple[float, float, float]
] = {
    "weighted_object_goal_progress": (0.0, 0.0, 0.0),
    "weighted_object_pose_accuracy": (0.0, 0.0, 0.0),
    "weighted_grasp_maintenance": (1.0, 0.0, 0.0),
    "weighted_wrench_range_violation_penalty": (1.0 / 3.0,) * 3,
    "weighted_normal_contact_quality_penalty": (1.0, 0.0, 0.0),
    "weighted_centroidal_stability": (0.0, 0.0, 0.0),
    "weighted_energy_penalty": (0.0, 0.0, 0.0),
    "weighted_qp_residual_penalty": (0.0, 0.0, 0.0),
    "weighted_slip_penalty": (0.0, 1.0, 0.0),
    "weighted_collision_penalty": (0.0, 0.0, 0.0),
    "weighted_actuator_saturation_penalty": (0.0, 0.0, 0.0),
    # The outer factorization assigns one third of terminal outcome to the
    # contact role.  Divide that share equally among its three physical roles.
    "terminal_success_bonus": (1.0 / 9.0,) * 3,
    "terminal_failure_penalty": (1.0 / 9.0,) * 3,
}


def factorize_order9_actor_reward(
    *,
    reward: torch.Tensor,
    reward_terms: torch.Tensor,
    reward_term_names: Sequence[str],
    valid: torch.Tensor,
    absolute_tolerance: float = 2.0e-5,
) -> tuple[dict[str, torch.Tensor], float]:
    """Partition one rollout reward without changing its scalar objective.

    Returns a mapping in ``ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS`` order and
    the measured maximum reconstruction error.  Invalid padded transitions are
    kept at zero in every channel.
    """

    names = tuple(str(value) for value in reward_term_names)
    expected = tuple(ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS)
    if names != expected:
        raise SchemaValidationError(
            "Order9 factorized actor credit reward-term order differs"
        )
    if (
        reward_terms.shape != (*reward.shape, len(names))
        or valid.shape != reward.shape
        or valid.dtype != torch.bool
    ):
        raise SchemaValidationError(
            "Order9 factorized actor credit tensor shapes differ"
        )
    if not bool(torch.isfinite(reward).all()) or not bool(
        torch.isfinite(reward_terms).all()
    ):
        raise SchemaValidationError(
            "Order9 factorized actor credit tensors are non-finite"
        )
    if absolute_tolerance <= 0.0:
        raise ValueError("Order9 factorized actor credit tolerance must be positive")

    weights = torch.tensor(
        [ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS[name] for name in names],
        device=reward_terms.device,
        dtype=reward_terms.dtype,
    )
    if not torch.allclose(
        weights.sum(dim=-1),
        torch.ones(len(names), device=weights.device, dtype=weights.dtype),
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise RuntimeError("Order9 factorized actor credit weights do not sum to one")
    factorized = torch.einsum("...r,rc->...c", reward_terms, weights)
    factorized = torch.where(valid.unsqueeze(-1), factorized, 0.0)
    reconstructed = factorized.sum(dim=-1)
    valid_error = (reconstructed - reward).abs()[valid]
    maximum_error = (
        float(valid_error.max().item()) if valid_error.numel() else 0.0
    )
    if maximum_error > absolute_tolerance:
        raise SchemaValidationError(
            "Order9 factorized actor reward does not reconstruct scalar reward: "
            f"{maximum_error:.9g} > {absolute_tolerance:.9g}"
        )
    channels = {
        name: factorized[..., index]
        for index, name in enumerate(ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS)
    }
    return channels, maximum_error


def factorize_order9_contact_coordinate_reward(
    *,
    contact_reward: torch.Tensor,
    reward_terms: torch.Tensor,
    reward_term_names: Sequence[str],
    valid: torch.Tensor,
    absolute_tolerance: float = 2.0e-5,
) -> tuple[dict[str, torch.Tensor], float]:
    """Partition the contact reward into normal, tangential, and rotation.

    This is a training-only refinement of ``factorize_order9_actor_reward``.
    It neither adds reward nor changes the deployed six-dimensional contact
    action.  Invalid padded transitions remain zero in every channel.
    """

    names = tuple(str(value) for value in reward_term_names)
    expected = tuple(ORDER9_CONTACT_COORDINATE_CREDIT_TERM_WEIGHTS)
    if names != expected:
        raise SchemaValidationError(
            "Order9 contact-coordinate credit reward-term order differs"
        )
    if (
        reward_terms.shape != (*contact_reward.shape, len(names))
        or valid.shape != contact_reward.shape
        or valid.dtype != torch.bool
    ):
        raise SchemaValidationError(
            "Order9 contact-coordinate credit tensor shapes differ"
        )
    if not bool(torch.isfinite(contact_reward).all()) or not bool(
        torch.isfinite(reward_terms).all()
    ):
        raise SchemaValidationError(
            "Order9 contact-coordinate credit tensors are non-finite"
        )
    if absolute_tolerance <= 0.0:
        raise ValueError(
            "Order9 contact-coordinate credit tolerance must be positive"
        )

    weights = torch.tensor(
        [ORDER9_CONTACT_COORDINATE_CREDIT_TERM_WEIGHTS[name] for name in names],
        device=reward_terms.device,
        dtype=reward_terms.dtype,
    )
    outer_contact_weights = torch.tensor(
        [
            ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS[name][0]
            for name in names
        ],
        device=weights.device,
        dtype=weights.dtype,
    )
    if not torch.allclose(
        weights.sum(dim=-1),
        outer_contact_weights,
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Order9 contact-coordinate weights do not reconstruct contact credit"
        )
    factorized = torch.einsum("...r,rc->...c", reward_terms, weights)
    factorized = torch.where(valid.unsqueeze(-1), factorized, 0.0)
    reconstructed = factorized.sum(dim=-1)
    valid_error = (reconstructed - contact_reward).abs()[valid]
    maximum_error = (
        float(valid_error.max().item()) if valid_error.numel() else 0.0
    )
    if maximum_error > absolute_tolerance:
        raise SchemaValidationError(
            "Order9 contact-coordinate reward does not reconstruct contact reward: "
            f"{maximum_error:.9g} > {absolute_tolerance:.9g}"
        )
    channels = {
        name: factorized[..., index]
        for index, name in enumerate(ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS)
    }
    return channels, maximum_error


__all__ = [
    "ORDER9_CONTACT_COORDINATE_ACTION_INDICES",
    "ORDER9_CONTACT_COORDINATE_CREDIT_CHANNELS",
    "ORDER9_CONTACT_COORDINATE_CREDIT_TERM_WEIGHTS",
    "ORDER9_CONTACT_COORDINATE_CREDIT_VERSION",
    "ORDER9_FACTORIZED_ACTOR_CREDIT_CHANNELS",
    "ORDER9_FACTORIZED_ACTOR_CREDIT_TERM_WEIGHTS",
    "ORDER9_FACTORIZED_ACTOR_CREDIT_VERSION",
    "factorize_order9_contact_coordinate_reward",
    "factorize_order9_actor_reward",
]
