from __future__ import annotations

"""Deterministic per-anchor load-limited position preload.

This is the small controller-side mechanism proven by the Order 8 natural
contact run.  It advances the already selected closure direction from the
previous absolute position target, freezes each contact branch after its
damping-compensated joint load dwells above the threshold, and freezes shared
joints conservatively with the first completed branch.
"""

import math
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

from amsrr.schemas.common import SchemaValidationError


LOAD_LIMITED_CONTACT_PRELOAD_VERSION = (
    "per_anchor_damping_compensated_moving_chain_load_dwell_v1"
)


@dataclass(frozen=True)
class LoadLimitedContactPreloadConfig:
    maximum_speed_rad_s: float = 0.002
    load_threshold_nm: float = 1.2
    load_dwell_s: float = 0.10
    contact_start_force_threshold_n: float = 0.5
    contact_start_dwell_s: float = 0.10
    motion_epsilon_rad_s: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "maximum_speed_rad_s",
            "load_threshold_nm",
            "load_dwell_s",
            "contact_start_force_threshold_n",
            "contact_start_dwell_s",
            "motion_epsilon_rad_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Load-limited contact preload {name} must be positive"
                )


@dataclass(frozen=True)
class LoadLimitedContactPreloadOutput:
    position_targets_rad: dict[str, float]
    velocity_targets_rad_s: dict[str, float]
    load_nm_by_anchor: dict[int, float]
    load_dwell_s_by_anchor: dict[int, float]
    frozen_anchor_ids: tuple[int, ...]
    active: bool
    complete: bool


class LoadLimitedContactPreload:
    """Stateful implementation of the Order 8 position-preload rule."""

    def __init__(
        self,
        config: LoadLimitedContactPreloadConfig | None = None,
    ) -> None:
        self.config = config or LoadLimitedContactPreloadConfig()
        self.reset()

    @property
    def initialized(self) -> bool:
        return bool(self._joint_ids)

    @property
    def complete(self) -> bool:
        return bool(self._complete)

    def start(
        self,
        *,
        ordered_joint_ids: Sequence[str],
        closure_velocity_targets_rad_s: Mapping[str, float],
        joint_ids_by_anchor: Mapping[int, Sequence[str]],
        initial_position_targets_rad: Mapping[str, float],
    ) -> None:
        if self.initialized:
            raise RuntimeError("Contact preload is already initialized")
        joint_ids = _validated_joint_ids(ordered_joint_ids)
        velocities = _validated_closure_velocity(
            joint_ids,
            closure_velocity_targets_rad_s,
        )
        anchor_joints = _validated_anchor_joint_ids(
            joint_ids,
            joint_ids_by_anchor,
            velocities,
            motion_epsilon_rad_s=self.config.motion_epsilon_rad_s,
        )
        if set(initial_position_targets_rad) != set(joint_ids):
            raise SchemaValidationError(
                "Contact preload initial positions must cover every ordered joint"
            )
        positions = {
            joint_id: float(initial_position_targets_rad[joint_id])
            for joint_id in joint_ids
        }
        if any(not math.isfinite(value) for value in positions.values()):
            raise SchemaValidationError(
                "Contact preload initial positions must be finite"
            )
        self._joint_ids = joint_ids
        self._closure_velocity_targets_rad_s = velocities
        self._joint_ids_by_anchor = anchor_joints
        self._position_targets_rad = positions
        self._load_nm_by_anchor = {
            anchor_id: 0.0 for anchor_id in anchor_joints
        }
        self._maximum_load_nm_by_anchor = {
            anchor_id: 0.0 for anchor_id in anchor_joints
        }
        self._load_dwell_s_by_anchor = {
            anchor_id: 0.0 for anchor_id in anchor_joints
        }
        self._frozen_anchor_ids = set()
        self._complete = False

    def step(
        self,
        *,
        applied_joint_load_nm: Mapping[str, float],
        dt_s: float,
    ) -> LoadLimitedContactPreloadOutput:
        if not self.initialized:
            raise RuntimeError("Contact preload has not been initialized")
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Contact preload dt must be positive")
        if set(applied_joint_load_nm) != set(self._joint_ids):
            raise SchemaValidationError(
                "Contact preload loads must cover every ordered joint"
            )
        loads = {
            joint_id: float(applied_joint_load_nm[joint_id])
            for joint_id in self._joint_ids
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in loads.values()
        ):
            raise SchemaValidationError(
                "Contact preload loads must be finite and non-negative"
            )

        if not self._complete:
            for anchor_id, joint_ids in self._joint_ids_by_anchor.items():
                side_load_nm = max(loads[joint_id] for joint_id in joint_ids)
                self._load_nm_by_anchor[anchor_id] = side_load_nm
                self._maximum_load_nm_by_anchor[anchor_id] = max(
                    self._maximum_load_nm_by_anchor[anchor_id],
                    side_load_nm,
                )
                if anchor_id in self._frozen_anchor_ids:
                    continue
                if (
                    side_load_nm + 1.0e-12
                    >= self.config.load_threshold_nm
                ):
                    self._load_dwell_s_by_anchor[anchor_id] += dt
                else:
                    self._load_dwell_s_by_anchor[anchor_id] = 0.0
                if (
                    self._load_dwell_s_by_anchor[anchor_id] + 1.0e-12
                    >= self.config.load_dwell_s
                ):
                    self._frozen_anchor_ids.add(anchor_id)

            self._complete = self._frozen_anchor_ids == set(
                self._joint_ids_by_anchor
            )
            velocity_targets = (
                {joint_id: 0.0 for joint_id in self._joint_ids}
                if self._complete
                else load_limited_velocity_targets(
                    ordered_joint_ids=self._joint_ids,
                    closure_velocity_targets_rad_s=(
                        self._closure_velocity_targets_rad_s
                    ),
                    joint_ids_by_anchor=self._joint_ids_by_anchor,
                    frozen_anchor_ids=self._frozen_anchor_ids,
                    maximum_speed_rad_s=self.config.maximum_speed_rad_s,
                )
            )
            self._position_targets_rad = {
                joint_id: (
                    self._position_targets_rad[joint_id]
                    + velocity_targets[joint_id] * dt
                )
                for joint_id in self._joint_ids
            }
        else:
            velocity_targets = {
                joint_id: 0.0 for joint_id in self._joint_ids
            }
        return self.output(velocity_targets_rad_s=velocity_targets)

    def hold(self) -> LoadLimitedContactPreloadOutput:
        if not self.initialized:
            raise RuntimeError("Contact preload has not been initialized")
        return self.output(
            velocity_targets_rad_s={
                joint_id: 0.0 for joint_id in self._joint_ids
            }
        )

    def output(
        self,
        *,
        velocity_targets_rad_s: Mapping[str, float] | None = None,
    ) -> LoadLimitedContactPreloadOutput:
        if not self.initialized:
            raise RuntimeError("Contact preload has not been initialized")
        velocities = (
            {joint_id: 0.0 for joint_id in self._joint_ids}
            if velocity_targets_rad_s is None
            else {
                joint_id: float(velocity_targets_rad_s[joint_id])
                for joint_id in self._joint_ids
            }
        )
        return LoadLimitedContactPreloadOutput(
            position_targets_rad=dict(self._position_targets_rad),
            velocity_targets_rad_s=velocities,
            load_nm_by_anchor=dict(self._load_nm_by_anchor),
            load_dwell_s_by_anchor=dict(self._load_dwell_s_by_anchor),
            frozen_anchor_ids=tuple(sorted(self._frozen_anchor_ids)),
            active=not self._complete,
            complete=self._complete,
        )

    def metrics(self) -> dict[str, float]:
        result = {
            "contact_preload_initialized": 1.0 if self.initialized else 0.0,
            "contact_preload_complete": 1.0 if self.complete else 0.0,
            "contact_preload_frozen_anchor_count": float(
                len(self._frozen_anchor_ids)
            ),
        }
        result.update(
            {
                f"contact_preload_load_nm.anchor_{anchor_id}": float(value)
                for anchor_id, value in self._load_nm_by_anchor.items()
            }
        )
        result.update(
            {
                f"contact_preload_maximum_load_nm.anchor_{anchor_id}": float(value)
                for anchor_id, value in self._maximum_load_nm_by_anchor.items()
            }
        )
        result.update(
            {
                f"contact_preload_load_dwell_s.anchor_{anchor_id}": float(value)
                for anchor_id, value in self._load_dwell_s_by_anchor.items()
            }
        )
        return result

    def reset(self) -> None:
        self._joint_ids: tuple[str, ...] = ()
        self._closure_velocity_targets_rad_s: dict[str, float] = {}
        self._joint_ids_by_anchor: dict[int, tuple[str, ...]] = {}
        self._position_targets_rad: dict[str, float] = {}
        self._load_nm_by_anchor: dict[int, float] = {}
        self._maximum_load_nm_by_anchor: dict[int, float] = {}
        self._load_dwell_s_by_anchor: dict[int, float] = {}
        self._frozen_anchor_ids: set[int] = set()
        self._complete = False


def load_limited_velocity_targets(
    *,
    ordered_joint_ids: Sequence[str],
    closure_velocity_targets_rad_s: Mapping[str, float],
    joint_ids_by_anchor: Mapping[int, Sequence[str]],
    frozen_anchor_ids: Collection[int],
    maximum_speed_rad_s: float,
) -> dict[str, float]:
    """Scale one fixed closure ratio and freeze completed branch owners."""

    joint_ids = _validated_joint_ids(ordered_joint_ids)
    velocities = _validated_closure_velocity(
        joint_ids,
        closure_velocity_targets_rad_s,
    )
    maximum_speed = float(maximum_speed_rad_s)
    if not math.isfinite(maximum_speed) or maximum_speed <= 0.0:
        raise ValueError("Contact preload maximum speed must be positive")
    peak = max(abs(value) for value in velocities.values())
    if peak <= 0.0:
        raise SchemaValidationError(
            "Contact preload closure direction cannot be zero"
        )
    scale = maximum_speed / peak
    scaled = {
        joint_id: value * scale for joint_id, value in velocities.items()
    }
    anchors = {int(anchor_id) for anchor_id in joint_ids_by_anchor}
    frozen = {int(anchor_id) for anchor_id in frozen_anchor_ids}
    if not frozen.issubset(anchors):
        raise SchemaValidationError(
            "Contact preload frozen anchor ids must be known"
        )
    owners_by_joint: dict[str, set[int]] = {
        joint_id: set() for joint_id in joint_ids
    }
    for anchor_id, raw_joint_ids in joint_ids_by_anchor.items():
        selected = {str(joint_id) for joint_id in raw_joint_ids}
        if not selected or not selected.issubset(owners_by_joint):
            raise SchemaValidationError(
                "Contact preload branch joints must be non-empty known subsets"
            )
        for joint_id in selected:
            owners_by_joint[joint_id].add(int(anchor_id))
    return {
        joint_id: (
            scaled[joint_id]
            if owners_by_joint[joint_id]
            and owners_by_joint[joint_id].isdisjoint(frozen)
            else 0.0
        )
        for joint_id in joint_ids
    }


def _validated_joint_ids(ordered_joint_ids: Sequence[str]) -> tuple[str, ...]:
    joint_ids = tuple(str(joint_id) for joint_id in ordered_joint_ids)
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise SchemaValidationError(
            "Contact preload requires unique ordered joint ids"
        )
    return joint_ids


def _validated_closure_velocity(
    joint_ids: Sequence[str],
    raw: Mapping[str, float],
) -> dict[str, float]:
    if set(raw) != set(joint_ids):
        raise SchemaValidationError(
            "Contact preload closure velocity must cover every ordered joint"
        )
    velocities = {joint_id: float(raw[joint_id]) for joint_id in joint_ids}
    if any(not math.isfinite(value) for value in velocities.values()):
        raise SchemaValidationError(
            "Contact preload closure velocity must be finite"
        )
    return velocities


def _validated_anchor_joint_ids(
    joint_ids: Sequence[str],
    raw: Mapping[int, Sequence[str]],
    velocities: Mapping[str, float],
    *,
    motion_epsilon_rad_s: float,
) -> dict[int, tuple[str, ...]]:
    anchor_ids = tuple(int(anchor_id) for anchor_id in raw)
    if len(anchor_ids) < 2 or len(set(anchor_ids)) != len(anchor_ids):
        raise SchemaValidationError(
            "Contact preload requires at least two unique anchors"
        )
    known = set(joint_ids)
    result: dict[int, tuple[str, ...]] = {}
    for anchor_id in anchor_ids:
        selected = tuple(
            joint_id
            for joint_id in joint_ids
            if joint_id in {str(value) for value in raw[anchor_id]}
            and abs(velocities[joint_id]) > motion_epsilon_rad_s
        )
        if not selected or not set(selected).issubset(known):
            raise SchemaValidationError(
                f"Contact preload anchor {anchor_id} has no moving known joint"
            )
        result[anchor_id] = selected
    return result


__all__ = [
    "LOAD_LIMITED_CONTACT_PRELOAD_VERSION",
    "LoadLimitedContactPreload",
    "LoadLimitedContactPreloadConfig",
    "LoadLimitedContactPreloadOutput",
    "load_limited_velocity_targets",
]
