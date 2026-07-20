from __future__ import annotations

"""Compose rolling Order 8 teacher decisions into full pi_H supervision windows."""

import math
from dataclasses import dataclass
from typing import Sequence

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    InteractionTrajectoryRecord,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.utils.hashing import stable_hash


ORDER9_TEACHER_WINDOW_VERSION = "order9_teacher_zero_order_hold_window_v1"


@dataclass(frozen=True)
class Order9TeacherWindowConfig:
    horizon_s: float = 2.0
    knot_dt_s: float = 0.25
    source_knot_semantics: str = "rolling_first_knot_at_decision_time"
    resampling_semantics: str = "latest_decision_zero_order_hold_on_fixed_grid"
    decision_return_semantics: str = "window_start_record_decision_return"
    allow_terminal_tail_hold: bool = True
    skip_incomplete_nonterminal_windows: bool = True

    def validate(self) -> None:
        if (
            not math.isfinite(self.horizon_s)
            or not math.isfinite(self.knot_dt_s)
            or self.horizon_s <= 0.0
            or self.knot_dt_s <= 0.0
        ):
            raise ValueError("teacher window horizon and knot dt must be finite and positive")
        quotient = self.horizon_s / self.knot_dt_s
        if not math.isclose(quotient, round(quotient), abs_tol=1.0e-9):
            raise ValueError("teacher window horizon must be an integer multiple of knot_dt_s")
        if not self.source_knot_semantics or not self.resampling_semantics:
            raise ValueError("teacher window semantics must be explicit")
        if not self.decision_return_semantics:
            raise ValueError("teacher window return semantics must be explicit")

    @property
    def knot_count(self) -> int:
        return int(round(self.horizon_s / self.knot_dt_s)) + 1


def compose_order9_teacher_windows(
    records: Sequence[InteractionTrajectoryRecord],
    *,
    checker: ContactWrenchTrajectoryFeasibilityChecker,
    config: Order9TeacherWindowConfig | None = None,
) -> list[InteractionTrajectoryRecord]:
    """Return checked, provenance-bearing full trajectories for BC.

    A source record represents the teacher plan made at one high-level update.
    The first knot of the most recent source decision is held until the next
    source decision.  A non-terminal episode tail is never silently padded.
    """

    window_config = config or Order9TeacherWindowConfig()
    window_config.validate()
    if not records:
        return []
    episodes: dict[str, list[InteractionTrajectoryRecord]] = {}
    for record in records:
        record.validate()
        if record.trajectory.contract_version != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME:
            raise SchemaValidationError("teacher window source must use the v2 wrench contract")
        if not record.trajectory.knots:
            raise SchemaValidationError("teacher window source has no trajectory knot")
        episodes.setdefault(record.episode_id, []).append(record)

    windows: list[InteractionTrajectoryRecord] = []
    for episode_id, episode_records in sorted(episodes.items()):
        ordered = sorted(
            episode_records,
            key=lambda record: (record.decision_time_s, record.decision_index),
        )
        _validate_episode_sequence(episode_id, ordered)
        for start_index, start in enumerate(ordered):
            window = _compose_one_window(
                ordered,
                start_index=start_index,
                checker=checker,
                config=window_config,
            )
            if window is not None:
                windows.append(window)
    return windows


def _compose_one_window(
    records: list[InteractionTrajectoryRecord],
    *,
    start_index: int,
    checker: ContactWrenchTrajectoryFeasibilityChecker,
    config: Order9TeacherWindowConfig,
) -> InteractionTrajectoryRecord | None:
    start = records[start_index]
    episode_tail = records[-1]
    available_horizon = episode_tail.decision_time_s - start.decision_time_s
    terminal_tail = _is_terminal_record(episode_tail)
    if available_horizon + 1.0e-9 < config.horizon_s and not (
        config.allow_terminal_tail_hold and terminal_tail
    ):
        if config.skip_incomplete_nonterminal_windows:
            return None
        raise SchemaValidationError(
            f"teacher window starting at {start.record_id!r} lacks future horizon"
        )

    grid = [index * config.knot_dt_s for index in range(config.knot_count)]
    source_records: list[InteractionTrajectoryRecord] = []
    source_index = start_index
    for relative_time in grid:
        absolute_time = start.decision_time_s + relative_time
        while (
            source_index + 1 < len(records)
            and records[source_index + 1].decision_time_s <= absolute_time + 1.0e-9
        ):
            source_index += 1
        source = records[source_index]
        _require_stationary_window_context(start, source)
        source_records.append(source)

    knots: list[InteractionKnot] = []
    for relative_time, source in zip(grid, source_records):
        source_knot = InteractionKnot.from_dict(source.trajectory.knots[0].to_dict())
        source_knot.t_rel_s = float(relative_time)
        knots.append(source_knot)
    trajectory = ContactWrenchTrajectory(
        horizon_s=config.horizon_s,
        dt_s=config.knot_dt_s,
        knots=knots,
        derived_mode_label=ORDER9_TEACHER_WINDOW_VERSION,
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )
    trajectory.validate()
    context = HighLevelPolicyContext(
        irg=start.irg,
        interaction_envelope=start.interaction_envelope,
        morphology_graph=start.morphology_graph,
        contact_candidate_set=start.contact_candidate_set,
        runtime_observation=start.runtime_observation,
    )
    feasibility = checker.check(trajectory, context)
    if not feasibility.feasible:
        codes = sorted({item.code for item in feasibility.hard_violations})
        raise SchemaValidationError(
            f"composed teacher trajectory is infeasible under C_H: {codes}"
        )

    source_ids = _ordered_unique(record.record_id for record in source_records)
    selected_candidate_ids = sorted(
        {
            assignment.candidate_id
            for knot in trajectory.knots
            for assignment in knot.contact_assignments
        }
    )
    assignment_results = []
    seen_results: set[str] = set()
    for source in source_records:
        for result in source.assignment_feasibility_results:
            fingerprint = result.stable_hash()
            if fingerprint not in seen_results:
                assignment_results.append(result)
                seen_results.add(fingerprint)
    composed = InteractionTrajectoryRecord(
        record_id=f"order9-teacher-window:{start.record_id}",
        episode_id=start.episode_id,
        task_id=start.task_id,
        split=start.split,
        decision_index=start.decision_index,
        decision_time_s=start.decision_time_s,
        irg=start.irg,
        interaction_envelope=start.interaction_envelope,
        morphology_graph=start.morphology_graph,
        contact_candidate_set=start.contact_candidate_set,
        runtime_observation=start.runtime_observation,
        trajectory=trajectory,
        selected_candidate_ids=selected_candidate_ids,
        assignment_feasibility_results=assignment_results,
        decision_return=start.decision_return,
        decision_reward=start.decision_reward,
        stage_masks=start.stage_masks,
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.DETERMINISTIC_TEACHER,
            source_version=ORDER9_TEACHER_WINDOW_VERSION,
            parent_record_id=start.record_id,
            metadata={
                "source_record_ids": source_ids,
                "source_record_count": len(source_ids),
                "source_knot_semantics": config.source_knot_semantics,
                "resampling_semantics": config.resampling_semantics,
                "decision_return_semantics": config.decision_return_semantics,
                "terminal_tail_hold_used": bool(
                    available_horizon + 1.0e-9 < config.horizon_s
                ),
                "context_semantics": "window_start_context",
                "horizon_s": config.horizon_s,
                "knot_dt_s": config.knot_dt_s,
            },
        ),
        trajectory_feasibility_result=feasibility,
        terminal=start.terminal,
        truncated=start.truncated,
        bootstrap_value=start.bootstrap_value,
    )
    composed.validate()
    return composed


def _validate_episode_sequence(
    episode_id: str,
    records: list[InteractionTrajectoryRecord],
) -> None:
    if len({record.split for record in records}) != 1:
        raise SchemaValidationError(f"teacher episode {episode_id!r} crosses dataset splits")
    if len({record.task_id for record in records}) != 1:
        raise SchemaValidationError(f"teacher episode {episode_id!r} crosses task IDs")
    times = [record.decision_time_s for record in records]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise SchemaValidationError(
            f"teacher episode {episode_id!r} decision times must increase strictly"
        )
    indices = [record.decision_index for record in records]
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise SchemaValidationError(
            f"teacher episode {episode_id!r} decision indices must increase strictly"
        )


def _require_stationary_window_context(
    start: InteractionTrajectoryRecord,
    source: InteractionTrajectoryRecord,
) -> None:
    fields = (
        ("irg", start.irg.stable_hash(), source.irg.stable_hash()),
        (
            "interaction_envelope",
            start.interaction_envelope.stable_hash(),
            source.interaction_envelope.stable_hash(),
        ),
        (
            "morphology_graph",
            start.morphology_graph.stable_hash(),
            source.morphology_graph.stable_hash(),
        ),
        (
            "contact_candidate_set",
            _candidate_structure_hash(start),
            _candidate_structure_hash(source),
        ),
    )
    changed = [name for name, left, right in fields if left != right]
    if changed:
        raise SchemaValidationError(
            "teacher window crosses a context boundary: " + ", ".join(changed)
        )


def _candidate_structure_hash(record: InteractionTrajectoryRecord) -> str:
    data = record.contact_candidate_set.to_dict()
    # The cache may accumulate evaluation labels between decisions; it is not
    # part of the ID/source mapping that the trajectory window must freeze.
    data["assignment_feasibility_cache"] = {}
    return stable_hash(data)


def _is_terminal_record(record: InteractionTrajectoryRecord) -> bool:
    progress = record.runtime_observation.task_progress
    phase = (progress.phase_label or "").lower()
    return bool(
        record.terminal
        or progress.success
        or progress.failure_reason
        or phase in {"complete", "done", "safe_hold", "failed", "abort"}
    )


def _ordered_unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
