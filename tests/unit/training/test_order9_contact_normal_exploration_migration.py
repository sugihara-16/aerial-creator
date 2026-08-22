import math

import pytest
import torch

from amsrr.policies.order9_low_level_policy import (
    Order9ContactSpacePhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_contact_normal_exploration_migration import (
    ORDER9_CONTACT_NORMAL_ACTION_INDEX,
    widen_order9_contact_normal_exploration,
)


def test_widen_contact_normal_exploration_changes_exactly_one_scalar() -> None:
    policy = Order9ContactSpacePhaseConditionedActorCritic()
    with torch.no_grad():
        policy.contact_space_actor_log_std.copy_(
            torch.tensor([-2.36, -1.9, -1.8, -1.7, -1.6, -1.5])
        )
    before = {
        key: value.detach().clone() for key, value in policy.state_dict().items()
    }

    result = widen_order9_contact_normal_exploration(
        policy,
        target_std=0.30,
    )

    assert result.source_std == pytest.approx(math.exp(-2.36))
    assert result.target_std == pytest.approx(0.30)
    assert result.changed_scalar_count == 1
    assert float(
        policy.contact_space_actor_log_std[
            ORDER9_CONTACT_NORMAL_ACTION_INDEX
        ].detach().exp()
    ) == pytest.approx(0.30)
    for key, value in policy.state_dict().items():
        if key != "contact_space_actor_log_std":
            assert torch.equal(value, before[key])
    changed = (
        policy.contact_space_actor_log_std.detach()
        != before["contact_space_actor_log_std"]
    )
    assert torch.nonzero(changed, as_tuple=False).flatten().tolist() == [
        ORDER9_CONTACT_NORMAL_ACTION_INDEX
    ]


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1, math.nan])
def test_widen_contact_normal_exploration_rejects_invalid_target(
    target: float,
) -> None:
    policy = Order9ContactSpacePhaseConditionedActorCritic()
    with pytest.raises(ValueError):
        widen_order9_contact_normal_exploration(policy, target_std=target)


def test_widen_contact_normal_exploration_rejects_narrower_target() -> None:
    policy = Order9ContactSpacePhaseConditionedActorCritic()
    with torch.no_grad():
        policy.contact_space_actor_log_std[
            ORDER9_CONTACT_NORMAL_ACTION_INDEX
        ] = math.log(0.4)
    with pytest.raises(SchemaValidationError, match="must widen"):
        widen_order9_contact_normal_exploration(policy, target_std=0.3)
