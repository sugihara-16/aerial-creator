from __future__ import annotations

"""Training-only nominal-preload deficit randomization for Order 9 C3."""

from dataclasses import dataclass
import math
import random
from typing import Sequence


ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION = (
    "order9-training-nominal-preload-deficit-v1"
)


@dataclass(frozen=True)
class Order9TrainingNominalPreloadDeficitPlan:
    requested_deficit_mm: tuple[float, ...]
    deficit_m_by_environment: tuple[float, ...]
    total_lead_m_by_environment: tuple[float, ...]
    scale_by_environment: tuple[float, ...]


def parse_order9_training_nominal_preload_deficit_mm(
    raw: str | None,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "training nominal-preload deficits must be comma-separated numbers"
        ) from error
    if len(values) < 2 or any(
        not math.isfinite(value) or not 0.0 <= value <= 20.0 for value in values
    ):
        raise ValueError(
            "training nominal-preload deficits require at least two values in "
            "[0, 20] mm"
        )
    if len(set(values)) != len(values):
        raise ValueError("training nominal-preload deficit values must be unique")
    if 0.0 not in values:
        raise ValueError(
            "training nominal-preload deficits must include the unmodified 0 mm case"
        )
    return values


def plan_order9_training_nominal_preload_deficit(
    *,
    nominal_lead_m: float,
    requested_deficit_mm: Sequence[float],
    environment_count: int,
    seed: int,
) -> Order9TrainingNominalPreloadDeficitPlan:
    """Assign balanced, shuffled under-preload cases to parallel environments."""

    if not math.isfinite(nominal_lead_m) or nominal_lead_m <= 0.0:
        raise ValueError("nominal preload lead must be finite and positive")
    if environment_count < 1:
        raise ValueError("environment count must be positive")
    if seed < 0:
        raise ValueError("deficit assignment seed must be non-negative")
    parsed = parse_order9_training_nominal_preload_deficit_mm(
        ",".join(str(float(value)) for value in requested_deficit_mm)
    )
    assert parsed is not None

    balanced = [parsed[index % len(parsed)] for index in range(environment_count)]
    random.Random(seed).shuffle(balanced)
    deficit_m = tuple(value * 1.0e-3 for value in balanced)
    total_lead_m = tuple(
        max(0.0, nominal_lead_m - value) for value in deficit_m
    )
    scale = tuple(value / nominal_lead_m for value in total_lead_m)
    return Order9TrainingNominalPreloadDeficitPlan(
        requested_deficit_mm=parsed,
        deficit_m_by_environment=deficit_m,
        total_lead_m_by_environment=total_lead_m,
        scale_by_environment=scale,
    )


def order9_training_preload_deficit_category_indices(
    *,
    nominal_lead_m: float,
    total_lead_m_by_environment: Sequence[float],
    category_step_m: float,
    category_count: int,
) -> tuple[int, ...]:
    """Map each effective lost nominal lead to its positive residual bin."""

    if not math.isfinite(nominal_lead_m) or nominal_lead_m <= 0.0:
        raise ValueError("nominal preload lead must be finite and positive")
    if (
        not math.isfinite(category_step_m)
        or category_step_m <= 0.0
        or category_count < 3
        or category_count % 2 != 1
    ):
        raise ValueError("contact-normal category grid is invalid")
    zero_index = category_count // 2
    result: list[int] = []
    for total_lead_m in total_lead_m_by_environment:
        if (
            not math.isfinite(total_lead_m)
            or not 0.0 <= total_lead_m <= nominal_lead_m + 1.0e-9
        ):
            raise ValueError("training preload total lead is invalid")
        effective_deficit_m = max(0.0, nominal_lead_m - total_lead_m)
        category_offset = int(round(effective_deficit_m / category_step_m))
        if not math.isclose(
            effective_deficit_m,
            category_offset * category_step_m,
            abs_tol=1.0e-7,
        ):
            raise ValueError("training preload deficit is off category grid")
        category_index = zero_index + category_offset
        if category_index >= category_count:
            raise ValueError("training preload deficit exceeds category grid")
        result.append(category_index)
    return tuple(result)


__all__ = [
    "ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION",
    "Order9TrainingNominalPreloadDeficitPlan",
    "parse_order9_training_nominal_preload_deficit_mm",
    "plan_order9_training_nominal_preload_deficit",
    "order9_training_preload_deficit_category_indices",
]
