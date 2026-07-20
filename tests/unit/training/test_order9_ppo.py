from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_ppo import Order9GAETransition, compute_order9_gae


def test_order9_gae_bootstraps_truncation_without_crossing_reset() -> None:
    result = compute_order9_gae(
        [
            Order9GAETransition(
                reward=1.0,
                value=0.5,
                terminal=False,
                truncated=False,
            ),
            Order9GAETransition(
                reward=2.0,
                value=0.7,
                terminal=False,
                truncated=True,
                bootstrap_value=0.9,
            ),
        ],
        gamma=0.9,
        gae_lambda=0.8,
    )

    assert result.advantages == pytest.approx((2.6492, 2.11))
    assert result.returns == pytest.approx((3.1492, 2.81))


def test_order9_gae_rejects_unbounded_fragment_tail() -> None:
    with pytest.raises(SchemaValidationError, match="must end"):
        compute_order9_gae(
            [
                Order9GAETransition(
                    reward=1.0,
                    value=0.0,
                    terminal=False,
                    truncated=False,
                )
            ],
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_order9_gae_does_not_credit_later_execution_to_checker_rejection() -> None:
    result = compute_order9_gae(
        [
            Order9GAETransition(
                reward=-1.0,
                value=0.2,
                terminal=True,
                truncated=False,
            ),
            Order9GAETransition(
                reward=2.0,
                value=0.4,
                terminal=True,
                truncated=False,
            ),
        ],
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert result.advantages == pytest.approx((-1.2, 1.6))
    assert result.returns == pytest.approx((-1.0, 2.0))
