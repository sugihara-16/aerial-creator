from __future__ import annotations

"""Offline replay and A/B metrics for the deployable anchor-force observer."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.training.order9_anchor_normal_force_estimator import (
    Order9AnchorNormalForceEstimator,
    Order9AnchorNormalForceEstimatorConfig,
    order9_branch_free_motion_baseline_update_mask,
)
from amsrr.utils.hashing import hash_file


ORDER9_FORCE_OBSERVER_TRACE_VERSION = "order9_force_observer_trace_v2_joint_mask"
ORDER9_FORCE_OBSERVER_REPLAY_VERSION = "order9_force_observer_offline_replay_v2"
_COMPATIBLE_TRACE_VERSIONS = {
    "order9_force_observer_trace_v1",
    ORDER9_FORCE_OBSERVER_TRACE_VERSION,
}

_TRACE_SEQUENCE_KEYS = (
    "applied_joint_torque_nm",
    "gravity_joint_torque_nm",
    "grasp_point_linear_jacobian_world",
    "reaction_normal_world",
    "selected_anchor_mask",
    "baseline_update_mask",
    "estimation_active_mask",
    "privileged_actual_normal_force_n",
    "required_normal_force_n",
    "surface_distance_m",
    "relative_speed_mps",
    "qp_feasible",
    "phase_index",
    "episode_serial",
    "online_normal_force_n",
    "online_confidence",
)


@dataclass(frozen=True)
class Order9ForceObserverReplayMetrics:
    baseline_enabled: bool
    contact_anchor_sample_count: int
    mean_absolute_error_n: float
    root_mean_square_error_n: float
    p95_absolute_error_n: float
    mean_absolute_error_n_by_anchor: tuple[float, ...]
    mean_confidence_by_anchor: tuple[float, ...]
    force_ready_true_positive_rate: float
    force_ready_false_positive_rate: float
    force_ready_environment_step_count: int
    oracle_ready_environment_step_count: int
    contact_dwell_admitted_environment_count: int
    oracle_dwell_admitted_environment_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Order9ForceObserverReplayResult:
    normal_force_n: torch.Tensor
    confidence: torch.Tensor
    fit_residual_nm: torch.Tensor
    baseline_nm: torch.Tensor
    metrics: Order9ForceObserverReplayMetrics


def write_order9_force_observer_trace(
    path: str | Path,
    *,
    metadata: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
) -> str:
    payload = {
        "trace_version": ORDER9_FORCE_OBSERVER_TRACE_VERSION,
        "metadata": dict(metadata),
        "tensors": {name: value.detach().to(device="cpu") for name, value in tensors.items()},
    }
    validate_order9_force_observer_trace(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Order9 force-observer trace exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hash_file(target)


def load_order9_force_observer_trace(path: str | Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_order9_force_observer_trace(payload)
    return payload


def replay_order9_force_observer_trace(
    payload: Mapping[str, object],
    *,
    baseline_enabled: bool,
    geometry_gated_contact_extension: bool = False,
) -> Order9ForceObserverReplayResult:
    validate_order9_force_observer_trace(payload)
    metadata = payload["metadata"]
    tensors = payload["tensors"]
    assert isinstance(metadata, Mapping)
    assert isinstance(tensors, Mapping)
    config_mapping = metadata.get("force_estimator_config")
    if not isinstance(config_mapping, Mapping):
        raise ValueError("Order9 force-observer trace lacks estimator config")
    config = Order9AnchorNormalForceEstimatorConfig(
        **{
            name: config_mapping[name]
            for name in Order9AnchorNormalForceEstimatorConfig.__dataclass_fields__
        }
    )
    estimator = Order9AnchorNormalForceEstimator(config)
    applied = tensors["applied_joint_torque_nm"]
    assert isinstance(applied, torch.Tensor)
    step_count, batch_size, joint_count = applied.shape
    selected = tensors["selected_anchor_mask"]
    assert isinstance(selected, torch.Tensor)
    anchor_count = selected.shape[-1]
    state = estimator.initial_state(
        batch_size=batch_size,
        anchor_count=anchor_count,
        joint_count=joint_count,
        device="cpu",
        dtype=applied.dtype,
    )
    owner = tensors["anchor_joint_owner_mask"]
    assert isinstance(owner, torch.Tensor)
    serial = tensors["episode_serial"]
    assert isinstance(serial, torch.Tensor)
    forces = []
    confidences = []
    residuals = []
    baselines = []
    previous_serial = serial[0]
    for step in range(step_count):
        if step:
            reset_ids = torch.nonzero(
                serial[step] != previous_serial, as_tuple=False
            ).flatten()
            if reset_ids.numel():
                state = estimator.reset_state_subset(state, reset_ids)
        baseline_mask = tensors["baseline_update_mask"][step]
        if not baseline_enabled:
            baseline_mask = torch.zeros_like(baseline_mask)
        elif geometry_gated_contact_extension:
            baseline_mask = order9_branch_free_motion_baseline_update_mask(
                approach_mask=(
                    tensors["phase_index"][step]
                    == ORDER9_OBJECT_TASK_PHASES.index(
                        Order9ObjectTaskPhase.APPROACH
                    )
                ),
                contact_acquisition_mask=(
                    tensors["phase_index"][step]
                    == ORDER9_OBJECT_TASK_PHASES.index(
                        Order9ObjectTaskPhase.CONTACT_ACQUISITION
                    )
                ),
                selected_surface_distance_m=tensors["surface_distance_m"][step],
                selected_anchor_mask=selected[step],
                anchor_joint_owner_mask=owner,
                baseline_initialized=state.baseline_initialized,
                contact_clearance_m=float(
                    metadata["contact_distance_tolerance_m"]
                ),
            )
        estimate = estimator.step(
            applied_joint_torque_nm=applied[step],
            gravity_joint_torque_nm=tensors["gravity_joint_torque_nm"][step],
            grasp_point_linear_jacobian_world=tensors[
                "grasp_point_linear_jacobian_world"
            ][step],
            contact_normal_world=tensors["reaction_normal_world"][step],
            anchor_joint_owner_mask=owner,
            selected_anchor_mask=selected[step],
            baseline_update_mask=baseline_mask,
            estimation_active_mask=tensors["estimation_active_mask"][step],
            state=state,
        )
        state = estimate.next_state
        forces.append(estimate.normal_force_n)
        confidences.append(estimate.confidence)
        residuals.append(estimate.fit_residual_nm)
        baselines.append(state.free_motion_torque_baseline_nm)
        previous_serial = serial[step]
    normal_force = torch.stack(forces)
    confidence = torch.stack(confidences)
    fit_residual = torch.stack(residuals)
    baseline = torch.stack(baselines)
    metrics = _replay_metrics(
        tensors=tensors,
        normal_force_n=normal_force,
        confidence=confidence,
        baseline_enabled=baseline_enabled,
        dt_s=float(metadata["control_dt_s"]),
        confidence_threshold=float(metadata["confidence_threshold"]),
        contact_distance_tolerance_m=float(
            metadata["contact_distance_tolerance_m"]
        ),
        contact_inward_limit_m=float(metadata["contact_inward_limit_m"]),
        relative_speed_tolerance_mps=float(
            metadata["relative_speed_tolerance_mps"]
        ),
        contact_dwell_s=float(metadata["contact_dwell_s"]),
    )
    return Order9ForceObserverReplayResult(
        normal_force_n=normal_force,
        confidence=confidence,
        fit_residual_nm=fit_residual,
        baseline_nm=baseline,
        metrics=metrics,
    )


def compare_order9_force_observer_baseline(
    payload: Mapping[str, object],
) -> dict[str, object]:
    without = replay_order9_force_observer_trace(
        payload, baseline_enabled=False
    )
    with_baseline = replay_order9_force_observer_trace(
        payload, baseline_enabled=True
    )
    geometry_gated = replay_order9_force_observer_trace(
        payload,
        baseline_enabled=True,
        geometry_gated_contact_extension=True,
    )
    without_mae = without.metrics.mean_absolute_error_n
    with_mae = with_baseline.metrics.mean_absolute_error_n
    relative_mae_reduction = (
        (without_mae - with_mae) / without_mae if without_mae > 0.0 else 0.0
    )
    return {
        "replay_version": ORDER9_FORCE_OBSERVER_REPLAY_VERSION,
        "without_baseline": without.metrics.to_dict(),
        "phase_only_baseline": with_baseline.metrics.to_dict(),
        "geometry_gated_branch_baseline": geometry_gated.metrics.to_dict(),
        # Compatibility alias for consumers written against replay v1.
        "with_baseline": with_baseline.metrics.to_dict(),
        "relative_mae_reduction": relative_mae_reduction,
        "dwell_admission_delta": (
            with_baseline.metrics.contact_dwell_admitted_environment_count
            - without.metrics.contact_dwell_admitted_environment_count
        ),
        "geometry_gated_relative_mae_reduction": (
            (without_mae - geometry_gated.metrics.mean_absolute_error_n)
            / without_mae
            if without_mae > 0.0
            else 0.0
        ),
        "geometry_gated_dwell_admission_delta": (
            geometry_gated.metrics.contact_dwell_admitted_environment_count
            - without.metrics.contact_dwell_admitted_environment_count
        ),
    }


def validate_order9_force_observer_trace(payload: Mapping[str, object]) -> None:
    if payload.get("trace_version") not in _COMPATIBLE_TRACE_VERSIONS:
        raise ValueError("Order9 force-observer trace version differs")
    metadata = payload.get("metadata")
    tensors = payload.get("tensors")
    if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
        raise ValueError("Order9 force-observer trace structure differs")
    missing = set(_TRACE_SEQUENCE_KEYS + ("anchor_joint_owner_mask",)) - set(
        tensors
    )
    if missing:
        raise ValueError(f"Order9 force-observer trace missing {sorted(missing)}")
    applied = tensors["applied_joint_torque_nm"]
    if not isinstance(applied, torch.Tensor) or applied.ndim != 3:
        raise ValueError("Order9 force-observer applied torque shape differs")
    sequence_shape = applied.shape[:2]
    for name in _TRACE_SEQUENCE_KEYS:
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or value.shape[:2] != sequence_shape:
            raise ValueError(f"Order9 force-observer trace {name} shape differs")
        if value.dtype != torch.bool and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Order9 force-observer trace {name} is non-finite")
    owner = tensors["anchor_joint_owner_mask"]
    if (
        not isinstance(owner, torch.Tensor)
        or owner.ndim != 2
        or owner.shape[-1] != applied.shape[-1]
    ):
        raise ValueError("Order9 force-observer owner mask shape differs")


def _replay_metrics(
    *,
    tensors: Mapping[str, torch.Tensor],
    normal_force_n: torch.Tensor,
    confidence: torch.Tensor,
    baseline_enabled: bool,
    dt_s: float,
    confidence_threshold: float,
    contact_distance_tolerance_m: float,
    contact_inward_limit_m: float,
    relative_speed_tolerance_mps: float,
    contact_dwell_s: float,
) -> Order9ForceObserverReplayMetrics:
    selected = tensors["selected_anchor_mask"].bool()
    active = tensors["estimation_active_mask"].bool()[..., None] & selected
    actual = tensors["privileged_actual_normal_force_n"]
    required = tensors["required_normal_force_n"]
    actual_contact = active & (actual >= 0.25)
    absolute_error = (normal_force_n - actual).abs()
    contact_errors = absolute_error[actual_contact]
    if contact_errors.numel():
        mae = float(contact_errors.mean().item())
        rmse = float(torch.sqrt((contact_errors.square()).mean()).item())
        p95 = float(torch.quantile(contact_errors, 0.95).item())
    else:
        mae = rmse = p95 = 0.0
    mae_by_anchor = []
    confidence_by_anchor = []
    for anchor in range(selected.shape[-1]):
        mask = actual_contact[..., anchor]
        mae_by_anchor.append(
            float(absolute_error[..., anchor][mask].mean().item())
            if bool(mask.any())
            else 0.0
        )
        confidence_by_anchor.append(
            float(confidence[..., anchor][mask].mean().item())
            if bool(mask.any())
            else 0.0
        )
    geometric = (
        (tensors["surface_distance_m"] <= contact_distance_tolerance_m)
        & (tensors["surface_distance_m"] >= -contact_inward_limit_m)
        & (tensors["relative_speed_mps"] <= relative_speed_tolerance_mps)
    )
    estimated_anchor_ready = (
        (normal_force_n >= required)
        & (confidence >= confidence_threshold)
        & geometric
        & selected
    )
    actual_anchor_ready = (actual >= required) & geometric & selected
    assigned = selected.sum(dim=-1) >= 2
    estimated_ready = (
        assigned
        & (estimated_anchor_ready | ~selected).all(dim=-1)
        & tensors["qp_feasible"].bool()
    )
    oracle_ready = (
        assigned
        & (actual_anchor_ready | ~selected).all(dim=-1)
        & tensors["qp_feasible"].bool()
    )
    true_positive = estimated_ready & oracle_ready
    false_positive = estimated_ready & ~oracle_ready
    true_positive_rate = _safe_rate(true_positive.sum(), oracle_ready.sum())
    false_positive_rate = _safe_rate(
        false_positive.sum(), (~oracle_ready).sum()
    )
    contact_phase = tensors["phase_index"] == ORDER9_OBJECT_TASK_PHASES.index(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION
    )
    dwell_steps = max(1, int(round(contact_dwell_s / dt_s)))
    admitted = _dwell_admitted(estimated_ready & contact_phase, dwell_steps)
    oracle_admitted = _dwell_admitted(oracle_ready & contact_phase, dwell_steps)
    return Order9ForceObserverReplayMetrics(
        baseline_enabled=baseline_enabled,
        contact_anchor_sample_count=int(actual_contact.sum().item()),
        mean_absolute_error_n=mae,
        root_mean_square_error_n=rmse,
        p95_absolute_error_n=p95,
        mean_absolute_error_n_by_anchor=tuple(mae_by_anchor),
        mean_confidence_by_anchor=tuple(confidence_by_anchor),
        force_ready_true_positive_rate=true_positive_rate,
        force_ready_false_positive_rate=false_positive_rate,
        force_ready_environment_step_count=int(estimated_ready.sum().item()),
        oracle_ready_environment_step_count=int(oracle_ready.sum().item()),
        contact_dwell_admitted_environment_count=int(admitted.sum().item()),
        oracle_dwell_admitted_environment_count=int(oracle_admitted.sum().item()),
    )


def _dwell_admitted(ready: torch.Tensor, dwell_steps: int) -> torch.Tensor:
    consecutive = torch.zeros(ready.shape[1], dtype=torch.long)
    admitted = torch.zeros(ready.shape[1], dtype=torch.bool)
    for step in ready:
        consecutive = torch.where(step, consecutive + 1, torch.zeros_like(consecutive))
        admitted |= consecutive >= dwell_steps
    return admitted


def _safe_rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = int(denominator.item())
    return float(numerator.item()) / value if value else 0.0


__all__ = [
    "ORDER9_FORCE_OBSERVER_REPLAY_VERSION",
    "ORDER9_FORCE_OBSERVER_TRACE_VERSION",
    "Order9ForceObserverReplayMetrics",
    "Order9ForceObserverReplayResult",
    "compare_order9_force_observer_baseline",
    "load_order9_force_observer_trace",
    "replay_order9_force_observer_trace",
    "validate_order9_force_observer_trace",
    "write_order9_force_observer_trace",
]
