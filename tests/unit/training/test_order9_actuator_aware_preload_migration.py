from pathlib import Path

import torch

from amsrr.policies.order9_low_level_policy import (
    Order9CategoricalContactNormalLowLevelPolicyConfig,
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
)


def test_neutral_rebase_bias_cancels_positive_window_prior() -> None:
    config = Order9CategoricalContactNormalLowLevelPolicyConfig(
        contact_normal_category_prior_mode="uniform_window",
        contact_normal_category_uniform_window_min_normalized=0.5,
        contact_normal_category_uniform_window_max_normalized=1.0,
        contact_normal_category_uniform_outside_logit_penalty=4.0,
    )
    model = Order9CategoricalContactNormalPhaseConditionedActorCritic(config)
    final = model.contact_normal_category_logits[-1]
    assert isinstance(final, torch.nn.Linear)
    categories = torch.linspace(-1.0, 1.0, 81)
    sigma = 0.15
    desired = -0.5 * (categories / sigma).square()
    prior = torch.where(
        (categories >= 0.5) & (categories <= 1.0),
        torch.zeros_like(categories),
        torch.full_like(categories, -4.0),
    )
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(desired - prior)
    combined = prior + final.bias.detach()
    assert int(combined.argmax().item()) == 40
    assert abs(float(categories[combined.argmax()].item())) < 1.0e-7
