from __future__ import annotations

"""Reproducible bounded-diversity conditions for high-fidelity Order 9 C0."""

import math
import random
from dataclasses import dataclass, field

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.order8 import Order8NaturalContactConfig
from amsrr.training.order9_curriculum import ORDER9_C0_COLLECTION_PROFILE_VERSION
from amsrr.training.order9_randomization import Order9ConservativeRandomizationConfig
from amsrr.utils.hashing import stable_hash


@dataclass
class Order9C0TeacherCondition(SchemaBase):
    profile_version: str
    condition_id: str
    episode_index: int
    random_seed: int
    sample_kind: str
    order8_config: Order8NaturalContactConfig
    sampled_values: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.profile_version != ORDER9_C0_COLLECTION_PROFILE_VERSION:
            raise SchemaValidationError("Order9 C0 condition profile version mismatch")
        require_non_empty(self.condition_id, "Order9C0TeacherCondition.condition_id")
        require_non_empty(self.sample_kind, "Order9C0TeacherCondition.sample_kind")
        if self.episode_index < 0 or self.random_seed < 0:
            raise SchemaValidationError(
                "Order9 C0 condition index and seed must be non-negative"
            )
        if any(
            not math.isfinite(float(value)) for value in self.sampled_values.values()
        ):
            raise SchemaValidationError(
                "Order9 C0 condition sampled values must be finite"
            )
        self.order8_config.validate()
        expected = order9_c0_condition_id(
            self.order8_config,
            random_seed=self.random_seed,
        )
        if self.condition_id != expected:
            raise SchemaValidationError("Order9 C0 condition identity mismatch")


def build_order9_c0_teacher_conditions(
    base_config: Order8NaturalContactConfig,
    *,
    episode_count: int,
    seed_start: int,
    randomization: Order9ConservativeRandomizationConfig | None = None,
) -> list[Order9C0TeacherCondition]:
    """Return nominal, four boundary corners, then deterministic random samples."""

    base_config.validate()
    bounds = randomization or Order9ConservativeRandomizationConfig()
    bounds.validate()
    if episode_count < 5:
        raise ValueError("Order9 C0 requires at least five teacher conditions")
    if seed_start < 0:
        raise ValueError("Order9 C0 seed_start must be non-negative")

    conditions: list[Order9C0TeacherCondition] = []
    for index in range(episode_count):
        seed = seed_start + index
        sample_kind, values = _condition_values(index, seed=seed, bounds=bounds)
        config = _apply_condition(base_config, values)
        condition = Order9C0TeacherCondition(
            profile_version=ORDER9_C0_COLLECTION_PROFILE_VERSION,
            condition_id=order9_c0_condition_id(config, random_seed=seed),
            episode_index=index,
            random_seed=seed,
            sample_kind=sample_kind,
            order8_config=config,
            sampled_values=values,
        )
        condition.validate()
        conditions.append(condition)
    if len({condition.condition_id for condition in conditions}) != len(conditions):
        raise SchemaValidationError("Order9 C0 generated duplicate condition IDs")
    return conditions


def order9_c0_condition_id(
    config: Order8NaturalContactConfig,
    *,
    random_seed: int,
) -> str:
    return "order9-c0-condition-" + stable_hash(
        {
            "profile_version": ORDER9_C0_COLLECTION_PROFILE_VERSION,
            "random_seed": int(random_seed),
            "order8_config_hash": config.stable_hash(),
        }
    )[:16]


def _condition_values(
    index: int,
    *,
    seed: int,
    bounds: Order9ConservativeRandomizationConfig,
) -> tuple[str, dict[str, float]]:
    if index == 0:
        return "nominal", {
            "dimension_scale_x": 1.0,
            "dimension_scale_y": 1.0,
            "dimension_scale_z": 1.0,
            "mass_scale": 1.0,
            "initial_standoff_offset_m": 0.0,
            "selected_gripper_friction": bounds.nominal_selected_gripper_friction,
            "contact_stiffness_scale": 1.0,
        }
    corners = (
        ("small_light_low_contact", False, False, False, False),
        ("large_heavy_high_contact", True, True, True, True),
        ("small_heavy_low_contact", False, True, False, True),
        ("large_light_high_contact", True, False, True, False),
    )
    if index <= len(corners):
        label, size_high, mass_high, contact_high, position_high = corners[index - 1]
        size = _edge(bounds.dimension_scale, size_high)
        return label, {
            "dimension_scale_x": size,
            "dimension_scale_y": size,
            "dimension_scale_z": size,
            "mass_scale": _edge(bounds.mass_scale, mass_high),
            "initial_standoff_offset_m": _edge(
                bounds.initial_xy_offset_m, position_high
            ),
            "selected_gripper_friction": _edge(
                bounds.selected_gripper_friction, contact_high
            ),
            "contact_stiffness_scale": _edge(
                bounds.contact_stiffness_scale, contact_high
            ),
        }
    rng = random.Random(_derived_seed(seed, index))
    return "seeded_random", {
        "dimension_scale_x": rng.uniform(*bounds.dimension_scale),
        "dimension_scale_y": rng.uniform(*bounds.dimension_scale),
        "dimension_scale_z": rng.uniform(*bounds.dimension_scale),
        "mass_scale": rng.uniform(*bounds.mass_scale),
        "initial_standoff_offset_m": rng.uniform(*bounds.initial_xy_offset_m),
        "selected_gripper_friction": rng.uniform(
            *bounds.selected_gripper_friction
        ),
        "contact_stiffness_scale": rng.uniform(*bounds.contact_stiffness_scale),
    }


def _apply_condition(
    base: Order8NaturalContactConfig,
    values: dict[str, float],
) -> Order8NaturalContactConfig:
    payload = base.to_dict()
    base_size = tuple(float(value) for value in base.object_size_m)
    payload["object_size_m"] = [
        base_size[0] * values["dimension_scale_x"],
        base_size[1] * values["dimension_scale_y"],
        base_size[2] * values["dimension_scale_z"],
    ]
    payload["object_mass_kg"] = float(base.object_mass_kg) * values["mass_scale"]
    payload["initial_object_standoff_m"] = (
        float(base.initial_object_standoff_m)
        + values["initial_standoff_offset_m"]
    )
    payload["selected_gripper_friction"] = values[
        "selected_gripper_friction"
    ]
    stiffness_scale = values["contact_stiffness_scale"]
    payload["selected_gripper_compliant_contact_stiffness_n_per_m"] = (
        float(base.selected_gripper_compliant_contact_stiffness_n_per_m)
        * stiffness_scale
    )
    payload["selected_gripper_compliant_contact_damping_n_s_per_m"] = (
        float(base.selected_gripper_compliant_contact_damping_n_s_per_m)
        * math.sqrt(stiffness_scale * values["mass_scale"])
    )
    return Order8NaturalContactConfig.from_dict(payload)


def _edge(bounds: tuple[float, float], high: bool) -> float:
    return float(bounds[1 if high else 0])


def _derived_seed(seed: int, index: int) -> int:
    return (int(seed) * 0x9E3779B185EBCA87 + int(index)) & ((1 << 64) - 1)


__all__ = [
    "Order9C0TeacherCondition",
    "build_order9_c0_teacher_conditions",
    "order9_c0_condition_id",
]
