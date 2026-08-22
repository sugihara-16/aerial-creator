from __future__ import annotations

"""Provenance-safe C3 contact-normal exploration initializer migration."""

from dataclasses import dataclass
import math

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_SPACE_ACTION_NAMES,
    Order9ContactSpacePhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaValidationError


ORDER9_CONTACT_NORMAL_EXPLORATION_MIGRATION_VERSION = (
    "order9_contact_normal_exploration_initializer_v1"
)
ORDER9_CONTACT_NORMAL_ACTION_NAME = "translation.inward_normal"
ORDER9_CONTACT_NORMAL_ACTION_INDEX = ORDER9_CONTACT_SPACE_ACTION_NAMES.index(
    ORDER9_CONTACT_NORMAL_ACTION_NAME
)


@dataclass(frozen=True)
class Order9ContactNormalExplorationMigration:
    source_log_std: float
    source_std: float
    target_log_std: float
    target_std: float
    changed_scalar_count: int


def widen_order9_contact_normal_exploration(
    policy: Order9ContactSpacePhaseConditionedActorCritic,
    *,
    target_std: float,
) -> Order9ContactNormalExplorationMigration:
    """Widen only the stochastic inward-normal coordinate.

    The actor mean, all deterministic command paths, other exploration
    coordinates, and every action bound remain bit-identical.
    """

    if not isinstance(policy, Order9ContactSpacePhaseConditionedActorCritic):
        raise SchemaValidationError(
            "Order9 contact-normal exploration migration requires the v7 "
            "contact-space pi_L actor"
        )
    target = float(target_std)
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("contact-normal target std must lie in (0, 1]")
    index = ORDER9_CONTACT_NORMAL_ACTION_INDEX
    source_log = float(
        policy.contact_space_actor_log_std[index].detach().cpu().item()
    )
    source = math.exp(source_log)
    target_log = math.log(target)
    if target_log <= source_log + 1.0e-12:
        raise SchemaValidationError(
            "Order9 contact-normal exploration target must widen the source"
        )
    before = policy.contact_space_actor_log_std.detach().clone()
    with torch.no_grad():
        policy.contact_space_actor_log_std[index].copy_(
            torch.as_tensor(
                target_log,
                device=policy.contact_space_actor_log_std.device,
                dtype=policy.contact_space_actor_log_std.dtype,
            )
        )
    after = policy.contact_space_actor_log_std.detach()
    changed = int((before != after).sum().item())
    if changed != 1 or not torch.equal(before[:index], after[:index]) or not torch.equal(
        before[index + 1 :], after[index + 1 :]
    ):
        raise RuntimeError(
            "Order9 contact-normal exploration migration changed extra coordinates"
        )
    return Order9ContactNormalExplorationMigration(
        source_log_std=source_log,
        source_std=source,
        target_log_std=target_log,
        target_std=target,
        changed_scalar_count=changed,
    )


__all__ = [
    "ORDER9_CONTACT_NORMAL_ACTION_INDEX",
    "ORDER9_CONTACT_NORMAL_ACTION_NAME",
    "ORDER9_CONTACT_NORMAL_EXPLORATION_MIGRATION_VERSION",
    "Order9ContactNormalExplorationMigration",
    "widen_order9_contact_normal_exploration",
]
