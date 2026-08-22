from __future__ import annotations

"""Behavior-preserving calibration for a wider C3 compression action span."""

import math


ORDER9_CONTACT_COMPRESSION_SPAN_MIGRATION_VERSION = (
    "order9_contact_compression_span_migration_v1"
)


def rescale_module_count_bias_for_span(
    bias: float,
    *,
    old_span_m: float,
    new_span_m: float,
) -> float:
    """Rescale the bias-only tanh action to preserve physical displacement.

    The shared actor contribution is intentionally left unchanged.  This
    migration is therefore followed by an exact Isaac A/B and fresh on-policy
    PPO; it is a safe initializer, not promotion evidence.
    """

    if not all(math.isfinite(value) for value in (bias, old_span_m, new_span_m)):
        raise ValueError("contact-compression span migration inputs must be finite")
    if old_span_m <= 0.0 or new_span_m <= 0.0 or old_span_m > new_span_m:
        raise ValueError("contact-compression span migration requires 0 < old <= new")
    scale = old_span_m / new_span_m
    target = math.tanh(bias) * scale
    target = min(max(target, -1.0 + 1.0e-7), 1.0 - 1.0e-7)
    return math.atanh(target)


__all__ = [
    "ORDER9_CONTACT_COMPRESSION_SPAN_MIGRATION_VERSION",
    "rescale_module_count_bias_for_span",
]
