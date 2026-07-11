from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    LowLevelPolicyContext,
    select_active_knot,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerStatus, InteractionKnot, PolicyCommand
from amsrr.schemas.runtime import ObjectRuntimeState, RuntimeObservation


PI_L_OUTPUT_MODE = "bounded_policy_command_delta"
PI_L_POLICY_CHECKPOINT_VERSION = "p4_3_pi_l_checkpoint_v1"

_TWIST_AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
_POSITION_AXES = ("x", "y", "z")
_WRENCH_AXES = ("fx", "fy", "fz", "tx", "ty", "tz")

PI_L_FEATURE_NAMES: tuple[str, ...] = (
    "observation.time_s",
    "morphology.module_count",
    "morphology.dock_edge_count",
    "morphology.robot_anchor_count",
    "capability.aggregate_mass_norm_sum",
    "capability.rotor_count_sum",
    "capability.port_count_sum",
    "capability.mean_thrust_to_weight_ratio",
    "capability.vectoring_fraction",
    "capability.dock_mechanism_fraction",
    *(f"base.position.{axis}" for axis in _POSITION_AXES),
    *(f"base.twist.{axis}" for axis in _TWIST_AXES),
    "base.health",
    *(f"object.position.{axis}" for axis in _POSITION_AXES),
    *(f"object.twist.{axis}" for axis in _TWIST_AXES),
    "contacts.active_count",
    "controller.qp_feasible",
    "controller.status.ok",
    "controller.status.warning",
    "controller.status.infeasible",
    "controller.status.fault",
    "controller.allocation_residual_norm",
    "task.progress_ratio",
    "task.success",
    "knot.t_rel_s",
    "knot.assignment_count",
    "knot.mean_assignment_priority",
    *(f"knot.mean_wrench_target.{axis}" for axis in _WRENCH_AXES),
    *(f"centroidal.position_error.{axis}" for axis in _POSITION_AXES),
    *(f"centroidal.velocity_error.{axis}" for axis in _POSITION_AXES),
    *(f"object_target.position_error.{axis}" for axis in _POSITION_AXES),
    *(f"object_target.velocity_error.{axis}" for axis in _TWIST_AXES),
)

PI_L_TARGET_NAMES: tuple[str, ...] = (
    *(f"desired_body_twist_delta.{axis}" for axis in _TWIST_AXES),
    *(f"desired_body_position_delta.{axis}" for axis in _POSITION_AXES),
    *(f"residual_wrench_body_delta.{axis}" for axis in _WRENCH_AXES),
)

# The learned head can only add these bounded deltas to an already valid
# BaselineLowLevelPolicy command. It never owns ControllerCommand or actuators.
PI_L_TARGET_LOWER_BOUNDS: tuple[float, ...] = (
    *([-0.5] * 6),
    # Dataset-supported body-position residual envelope.  These limits cover
    # the deterministic P4.2 teacher relative to the pi_H knot baseline while
    # the downstream controller/QP remains the hard authority.
    -1.75,
    -1.35,
    -0.40,
    # Match the deterministic baseline residual limits so imitation can
    # explicitly cancel a baseline residual when the teacher field is absent.
    *([-4.0] * 3),
    *([-0.50] * 3),
)
PI_L_TARGET_UPPER_BOUNDS: tuple[float, ...] = tuple(
    -value for value in PI_L_TARGET_LOWER_BOUNDS
)
PI_L_FEATURE_OOD_MIN_SCALES: tuple[float, ...] = tuple(1.0 for _ in PI_L_FEATURE_NAMES)


class TinyPiLDeltaMLP(nn.Module):
    """Small bounded head for a PolicyCommand delta subset."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_lower_bounds: Sequence[float] = PI_L_TARGET_LOWER_BOUNDS,
        output_upper_bounds: Sequence[float] = PI_L_TARGET_UPPER_BOUNDS,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("TinyPiLDeltaMLP dimensions must be positive")
        lower = torch.tensor(list(output_lower_bounds), dtype=torch.float32)
        upper = torch.tensor(list(output_upper_bounds), dtype=torch.float32)
        if lower.ndim != 1 or lower.shape != upper.shape or lower.numel() < 1:
            raise ValueError("pi_L output bounds must be non-empty vectors of equal shape")
        if not bool(torch.all(upper > lower)):
            raise ValueError("pi_L upper output bounds must be greater than lower bounds")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(lower.numel())),
        )
        self.register_buffer("output_center", (lower + upper) * 0.5)
        self.register_buffer("output_half_range", (upper - lower) * 0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_center + self.output_half_range * torch.tanh(self.net(features))


@dataclass(frozen=True)
class LearnedLowLevelPolicyDiagnostics:
    used_learned_delta: bool
    fallback_reason: str | None
    bounded_delta: dict[str, float]


class LearnedLowLevelPolicy:
    """P4.3b learned pi_L wrapper with a deterministic baseline fallback.

    Only a bounded subset of ``PolicyCommand`` is changed. The baseline owns
    all other intent fields, and the downstream controller remains the sole
    owner of ``ControllerCommand`` and actuator targets.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        feature_ood_scale: Sequence[float],
        output_lower_bounds: Sequence[float] = PI_L_TARGET_LOWER_BOUNDS,
        output_upper_bounds: Sequence[float] = PI_L_TARGET_UPPER_BOUNDS,
        ood_z_score_limit: float = 8.0,
        baseline_policy: BaselineLowLevelPolicy | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.feature_mean = _float_vector(feature_mean, len(PI_L_FEATURE_NAMES), "feature_mean")
        self.feature_std = _positive_float_vector(
            feature_std,
            len(PI_L_FEATURE_NAMES),
            "feature_std",
        )
        self.feature_ood_scale = _positive_float_vector(
            feature_ood_scale,
            len(PI_L_FEATURE_NAMES),
            "feature_ood_scale",
        )
        self.output_lower_bounds = _float_vector(
            output_lower_bounds,
            len(PI_L_TARGET_NAMES),
            "output_lower_bounds",
        )
        self.output_upper_bounds = _float_vector(
            output_upper_bounds,
            len(PI_L_TARGET_NAMES),
            "output_upper_bounds",
        )
        if any(upper <= lower for lower, upper in zip(self.output_lower_bounds, self.output_upper_bounds)):
            raise ValueError("output upper bounds must be greater than lower bounds")
        if not math.isfinite(ood_z_score_limit) or ood_z_score_limit <= 0.0:
            raise ValueError("ood_z_score_limit must be finite and positive")
        self.ood_z_score_limit = float(ood_z_score_limit)
        self.baseline_policy = baseline_policy or BaselineLowLevelPolicy()
        self.last_diagnostics = LearnedLowLevelPolicyDiagnostics(
            used_learned_delta=False,
            fallback_reason="not_run",
            bounded_delta={},
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        baseline_policy: BaselineLowLevelPolicy | None = None,
    ) -> LearnedLowLevelPolicy:
        checkpoint = _load_checkpoint(checkpoint_path)
        _validate_checkpoint_metadata(checkpoint)
        model = TinyPiLDeltaMLP(
            input_dim=len(PI_L_FEATURE_NAMES),
            hidden_dim=int(checkpoint["hidden_dim"]),
            output_lower_bounds=checkpoint["output_lower_bounds"],
            output_upper_bounds=checkpoint["output_upper_bounds"],
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return cls(
            model=model,
            feature_mean=checkpoint["feature_mean"],
            feature_std=checkpoint["feature_std"],
            feature_ood_scale=checkpoint["feature_ood_scale"],
            output_lower_bounds=checkpoint["output_lower_bounds"],
            output_upper_bounds=checkpoint["output_upper_bounds"],
            ood_z_score_limit=float(checkpoint["ood_z_score_limit"]),
            baseline_policy=baseline_policy,
        )

    def command(self, context: LowLevelPolicyContext) -> PolicyCommand:
        baseline = self.baseline_policy.command(context)
        controller_status = context.controller_status or context.runtime_observation.controller_status
        if _controller_is_infeasible(controller_status):
            return self._fallback(baseline, "controller_infeasible")

        try:
            features = pi_l_feature_vector(context)
        except (SchemaValidationError, TypeError, ValueError):
            return self._fallback(baseline, "feature_extraction_error")
        if len(features) != len(PI_L_FEATURE_NAMES):
            return self._fallback(baseline, "feature_shape")
        if not all(math.isfinite(value) for value in features):
            return self._fallback(baseline, "non_finite_features")
        if any(
            abs(value - mean) > self.ood_z_score_limit * scale
            for value, mean, scale in zip(features, self.feature_mean, self.feature_ood_scale)
        ):
            return self._fallback(baseline, "feature_ood")

        normalized = [
            (value - mean) / std
            for value, mean, std in zip(features, self.feature_mean, self.feature_std)
        ]
        if not all(math.isfinite(value) for value in normalized):
            return self._fallback(baseline, "non_finite_normalized_features")
        try:
            with torch.no_grad():
                output = self.model(torch.tensor([normalized], dtype=torch.float32))
        except (RuntimeError, TypeError, ValueError):
            return self._fallback(baseline, "model_inference_error")
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, len(PI_L_TARGET_NAMES)):
            return self._fallback(baseline, "model_output_shape")
        raw_delta = [float(value) for value in output[0].detach().cpu().tolist()]
        if not all(math.isfinite(value) for value in raw_delta):
            return self._fallback(baseline, "non_finite_model_output")
        bounded_delta = [
            min(max(value, lower), upper)
            for value, lower, upper in zip(
                raw_delta,
                self.output_lower_bounds,
                self.output_upper_bounds,
            )
        ]
        command = merge_bounded_pi_l_delta(baseline, bounded_delta)
        if not _learned_fields_are_finite(command):
            return self._fallback(baseline, "non_finite_merged_command")
        self.last_diagnostics = LearnedLowLevelPolicyDiagnostics(
            used_learned_delta=True,
            fallback_reason=None,
            bounded_delta=dict(zip(PI_L_TARGET_NAMES, bounded_delta)),
        )
        return command

    def _fallback(self, baseline: PolicyCommand, reason: str) -> PolicyCommand:
        self.last_diagnostics = LearnedLowLevelPolicyDiagnostics(
            used_learned_delta=False,
            fallback_reason=reason,
            bounded_delta={},
        )
        return baseline


def pi_l_feature_vector(context: LowLevelPolicyContext) -> list[float]:
    observation = context.runtime_observation
    morphology = context.morphology_graph
    active_knot = select_active_knot(context)
    controller_status = context.controller_status or observation.controller_status
    base_state = _base_module_state(observation, morphology)
    object_state, object_target = _primary_object_state_and_target(observation, active_knot)

    module_count = len(morphology.modules)
    capabilities = [module.capability_token for module in morphology.modules]
    aggregate_mass_norm_sum = sum(item.aggregate_mass_norm for item in capabilities)
    rotor_count_sum = sum(item.rotor_count for item in capabilities)
    port_count_sum = sum(item.port_count for item in capabilities)
    mean_thrust_to_weight = _mean(
        [item.thrust_to_weight_ratio_est for item in capabilities]
    )
    vectoring_fraction = _mean([1.0 if item.has_vectoring else 0.0 for item in capabilities])
    dock_fraction = _mean([1.0 if item.has_dock_mechanism else 0.0 for item in capabilities])

    base_position = list(base_state.pose_world[:3]) if base_state is not None else [0.0] * 3
    base_twist = list(base_state.twist_world) if base_state is not None else [0.0] * 6
    base_health = float(base_state.health) if base_state is not None else 0.0
    object_position = list(object_state.pose_world[:3]) if object_state is not None else [0.0] * 3
    object_twist = list(object_state.twist_world) if object_state is not None else [0.0] * 6

    wrench_targets = [
        assignment.wrench_target
        for assignment in active_knot.contact_assignments
        if assignment.wrench_target is not None
    ]
    mean_wrench = [
        _mean([float(wrench[index]) for wrench in wrench_targets])
        for index in range(6)
    ]
    mean_assignment_priority = _mean(
        [float(assignment.priority) for assignment in active_knot.contact_assignments]
    )

    centroidal_position_error = [0.0] * 3
    centroidal_velocity_error = [0.0] * 3
    if active_knot.centroidal_target is not None:
        if active_knot.centroidal_target.com_pos_world is not None:
            centroidal_position_error = [
                float(target) - float(current)
                for target, current in zip(
                    active_knot.centroidal_target.com_pos_world,
                    base_position,
                )
            ]
        if active_knot.centroidal_target.com_vel_world is not None:
            centroidal_velocity_error = [
                float(target) - float(current)
                for target, current in zip(
                    active_knot.centroidal_target.com_vel_world,
                    base_twist[:3],
                )
            ]

    object_position_error = [0.0] * 3
    object_velocity_error = [0.0] * 6
    if object_target is not None and object_state is not None:
        if object_target.pose_target_world is not None:
            object_position_error = [
                float(target) - float(current)
                for target, current in zip(
                    object_target.pose_target_world[:3],
                    object_state.pose_world[:3],
                )
            ]
        if object_target.twist_target_world is not None:
            object_velocity_error = [
                float(target) - float(current)
                for target, current in zip(
                    object_target.twist_target_world,
                    object_state.twist_world,
                )
            ]

    status_one_hot = [
        1.0 if controller_status.status == status else 0.0
        for status in ("ok", "warning", "infeasible", "fault")
    ]
    active_contact_count = sum(1 for contact in observation.contact_states if contact.active)
    allocation_residual = controller_status.metrics.get(
        "allocation_residual_norm",
        controller_status.metrics.get("residual_norm", 0.0),
    )
    features = [
        float(observation.time_s),
        float(module_count),
        float(len(morphology.dock_edges)),
        float(len(morphology.robot_anchors)),
        float(aggregate_mass_norm_sum),
        float(rotor_count_sum),
        float(port_count_sum),
        float(mean_thrust_to_weight),
        float(vectoring_fraction),
        float(dock_fraction),
        *[float(value) for value in base_position],
        *[float(value) for value in base_twist],
        base_health,
        *[float(value) for value in object_position],
        *[float(value) for value in object_twist],
        float(active_contact_count),
        1.0 if controller_status.qp_feasible else 0.0,
        *status_one_hot,
        float(allocation_residual),
        float(observation.task_progress.progress_ratio),
        1.0 if observation.task_progress.success else 0.0,
        float(active_knot.t_rel_s),
        float(len(active_knot.contact_assignments)),
        float(mean_assignment_priority),
        *mean_wrench,
        *centroidal_position_error,
        *centroidal_velocity_error,
        *object_position_error,
        *object_velocity_error,
    ]
    if len(features) != len(PI_L_FEATURE_NAMES):
        raise RuntimeError(
            f"internal pi_L feature layout mismatch: {len(features)} != {len(PI_L_FEATURE_NAMES)}"
        )
    return features


def merge_bounded_pi_l_delta(
    baseline: PolicyCommand,
    delta: Sequence[float],
) -> PolicyCommand:
    values = _float_vector(delta, len(PI_L_TARGET_NAMES), "pi_L delta")
    bounded = [
        min(max(value, lower), upper)
        for value, lower, upper in zip(
            values,
            PI_L_TARGET_LOWER_BOUNDS,
            PI_L_TARGET_UPPER_BOUNDS,
        )
    ]
    merged = baseline.to_dict()
    offset = 0
    if baseline.desired_body_twist is not None:
        merged["desired_body_twist"] = [
            float(value) + bounded[index]
            for index, value in enumerate(baseline.desired_body_twist)
        ]
    offset += 6
    if baseline.desired_body_pose is not None:
        pose = list(baseline.desired_body_pose)
        for index in range(3):
            pose[index] = float(pose[index]) + bounded[offset + index]
        merged["desired_body_pose"] = pose
    offset += 3
    if baseline.residual_wrench_body is not None:
        merged["residual_wrench_body"] = [
            float(value) + bounded[offset + index]
            for index, value in enumerate(baseline.residual_wrench_body)
        ]
    return PolicyCommand.from_dict(merged)


def overlay_learned_pi_l_subset(
    template: PolicyCommand,
    learned_command: PolicyCommand,
    *,
    blend_factor: float = 1.0,
) -> PolicyCommand:
    """Overlay only the learned P4.3 pi_L fields on deterministic intent.

    The runtime adapter uses the existing P4.2 command as ``template`` so
    contact, anchor, joint, and priority intent cannot change merely because a
    checkpoint was enabled.  The controller/QP retains final command authority.
    """

    if not math.isfinite(blend_factor) or not 0.0 < blend_factor <= 1.0:
        raise ValueError("pi_L runtime blend_factor must be in (0, 1]")
    command_data = template.to_dict()
    if template.desired_body_twist is not None and learned_command.desired_body_twist is not None:
        command_data["desired_body_twist"] = _blend_vectors(
            template.desired_body_twist,
            learned_command.desired_body_twist,
            blend_factor,
        )
    if template.desired_body_pose is not None and learned_command.desired_body_pose is not None:
        pose = list(template.desired_body_pose)
        pose[:3] = _blend_vectors(
            template.desired_body_pose[:3],
            learned_command.desired_body_pose[:3],
            blend_factor,
        )
        command_data["desired_body_pose"] = pose
    if template.residual_wrench_body is not None and learned_command.residual_wrench_body is not None:
        command_data["residual_wrench_body"] = _blend_vectors(
            template.residual_wrench_body,
            learned_command.residual_wrench_body,
            blend_factor,
        )
    return PolicyCommand.from_dict(command_data)


def _blend_vectors(
    template: Sequence[float],
    learned: Sequence[float],
    blend_factor: float,
) -> list[float]:
    if len(template) != len(learned):
        raise ValueError("pi_L runtime overlay vector shapes must match")
    return [
        float(base) + blend_factor * (float(candidate) - float(base))
        for base, candidate in zip(template, learned)
    ]


def _base_module_state(observation: RuntimeObservation, morphology: MorphologyGraph):
    by_id = {state.module_id: state for state in observation.module_states}
    if morphology.base_module_id in by_id:
        return by_id[morphology.base_module_id]
    return min(observation.module_states, key=lambda item: item.module_id, default=None)


def _primary_object_state_and_target(
    observation: RuntimeObservation,
    active_knot: InteractionKnot,
) -> tuple[ObjectRuntimeState | None, Any | None]:
    states = {state.object_id: state for state in observation.object_states}
    targets = sorted(active_knot.object_targets, key=lambda item: item.object_id)
    if targets:
        target = targets[0]
        return states.get(target.object_id), target
    state = min(observation.object_states, key=lambda item: item.object_id, default=None)
    return state, None


def _controller_is_infeasible(status: ControllerStatus) -> bool:
    return status.status in {"infeasible", "fault"} or not status.qp_feasible


def _learned_fields_are_finite(command: PolicyCommand) -> bool:
    values: list[float] = []
    if command.desired_body_twist is not None:
        values.extend(command.desired_body_twist)
    if command.desired_body_pose is not None:
        values.extend(command.desired_body_pose)
    if command.residual_wrench_body is not None:
        values.extend(command.residual_wrench_body)
    return all(math.isfinite(float(value)) for value in values)


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _float_vector(values: Sequence[float], expected: int, name: str) -> list[float]:
    output = [float(value) for value in values]
    if len(output) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(output)}")
    if not all(math.isfinite(value) for value in output):
        raise ValueError(f"{name} must contain finite values")
    return output


def _positive_float_vector(values: Sequence[float], expected: int, name: str) -> list[float]:
    output = _float_vector(values, expected, name)
    if any(value <= 0.0 for value in output):
        raise ValueError(f"{name} must contain positive values")
    return output


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("pi_L checkpoint must contain a mapping")
    return checkpoint


def _validate_checkpoint_metadata(checkpoint: dict[str, Any]) -> None:
    required = {
        "checkpoint_version",
        "output_mode",
        "state_dict",
        "hidden_dim",
        "feature_names",
        "target_names",
        "feature_mean",
        "feature_std",
        "feature_ood_scale",
        "output_lower_bounds",
        "output_upper_bounds",
        "ood_z_score_limit",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"pi_L checkpoint is missing metadata: {missing}")
    if checkpoint["checkpoint_version"] != PI_L_POLICY_CHECKPOINT_VERSION:
        raise ValueError("unsupported pi_L checkpoint version")
    if checkpoint["output_mode"] != PI_L_OUTPUT_MODE:
        raise ValueError("unsupported pi_L output mode")
    if tuple(checkpoint["feature_names"]) != PI_L_FEATURE_NAMES:
        raise ValueError("pi_L checkpoint feature layout does not match runtime")
    if tuple(checkpoint["target_names"]) != PI_L_TARGET_NAMES:
        raise ValueError("pi_L checkpoint target layout does not match runtime")
    if tuple(float(value) for value in checkpoint["output_lower_bounds"]) != (
        PI_L_TARGET_LOWER_BOUNDS
    ):
        raise ValueError("pi_L checkpoint lower bounds do not match the runtime safety contract")
    if tuple(float(value) for value in checkpoint["output_upper_bounds"]) != (
        PI_L_TARGET_UPPER_BOUNDS
    ):
        raise ValueError("pi_L checkpoint upper bounds do not match the runtime safety contract")
