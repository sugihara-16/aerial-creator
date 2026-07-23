from __future__ import annotations

"""Deployable active-knot actor features for the Order 9 low-level policy.

The feature contract contains planned task/contact intent and ordinary runtime
state only.  Measured contact forces, simulator-only state, and estimator truth
remain outside the actor boundary.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from amsrr.geometry.pose_math import quat_to_matrix, transpose
from amsrr.policies.low_level_policy_base import (
    LowLevelPolicyContext,
    select_active_knot,
)
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.schemas.runtime import RuntimeObservation


ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION = (
    "order9_deployable_active_contact_wrench_knot_features_v1"
)
_SCHEDULE_LABELS = ("approach", "attach", "maintain", "slide", "release")
_MODE_LABELS = tuple(mode.value for mode in ContactMode)
_WRENCH_AXES = ("fx", "fy", "fz", "tx", "ty", "tz")
_SUMMARY_NAMES = ("count", "mean", "rms", "minimum", "maximum")


ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "trajectory.signed_log_horizon_s",
    "trajectory.signed_log_dt_s",
    "active_knot.time_over_horizon",
    "phase.progress",
    "assignments.signed_log_count",
    *(f"assignments.schedule_fraction.{name}" for name in _SCHEDULE_LABELS),
    *(f"assignments.mode_fraction.{name}" for name in _MODE_LABELS),
    "centroidal.position_present",
    *(f"centroidal.position_error_world.{axis}" for axis in ("x", "y", "z")),
    "centroidal.velocity_present",
    *(f"centroidal.velocity_error_world.{axis}" for axis in ("x", "y", "z")),
    "centroidal.orientation_present",
    *(f"centroidal.orientation_error_body.{axis}" for axis in ("rx", "ry", "rz")),
    "centroidal.wrench_preference_present",
    *(f"centroidal.signed_log_wrench_preference.{axis}" for axis in _WRENCH_AXES),
    "object.pose_present",
    *(f"object.position_error_world.{axis}" for axis in ("x", "y", "z")),
    *(f"object.orientation_error_body.{axis}" for axis in ("rx", "ry", "rz")),
    "object.twist_present",
    *(f"object.twist_error.{axis}" for axis in ("vx", "vy", "vz", "wx", "wy", "wz")),
    "posture.position_present",
    *(f"posture.position_error.{name}" for name in _SUMMARY_NAMES),
    "posture.velocity_present",
    *(f"posture.velocity_error.{name}" for name in _SUMMARY_NAMES),
    "posture.signed_log_free_anchor_count",
    "wrench.target_fraction",
    "wrench.lower_fraction",
    "wrench.upper_fraction",
    *(f"wrench.signed_log_target_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"wrench.signed_log_lower_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"wrench.signed_log_upper_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"wrench.signed_log_width_mean.{axis}" for axis in _WRENCH_AXES),
    "assignments.priority.count",
    "assignments.priority.mean",
    "assignments.priority.minimum",
    "assignments.priority.maximum",
    "priority_weights.count",
    "priority_weights.value_sum",
    "priority_weights.key_hash_sin_sum",
    "priority_weights.key_hash_cos_sum",
    "guards.count",
    "guards.hash_sin_sum",
    "guards.hash_cos_sum",
    "morphology.module_count_over_eight",
    "physical.signed_log_total_mass_kg",
    *(f"physical.signed_log_total_inertia.{axis}" for axis in _WRENCH_AXES),
    "physical.signed_log_rotor_count",
    "physical.signed_log_total_max_thrust_n",
    "physical.signed_log_non_fixed_joint_count",
    "controller.qp_feasible",
    *(f"controller.status.{name}" for name in ("ok", "warning", "infeasible", "fault")),
    "controller.signed_log_allocation_residual",
    "controller.active_mode_hash_sin",
    "controller.active_mode_hash_cos",
)


ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES: tuple[str, ...] = (
    "assignments.signed_log_count",
    *(f"assignments.schedule_fraction.{name}" for name in _SCHEDULE_LABELS),
    *(f"assignments.mode_fraction.{name}" for name in _MODE_LABELS),
    "assignments.priority.mean",
    "assignments.priority.maximum",
    "wrench.target_fraction",
    "wrench.lower_fraction",
    "wrench.upper_fraction",
    *(f"wrench.signed_log_target_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"wrench.signed_log_lower_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"wrench.signed_log_upper_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"wrench.signed_log_width_mean.{axis}" for axis in _WRENCH_AXES),
    *(f"posture.position_error.{name}" for name in _SUMMARY_NAMES),
    *(f"posture.velocity_error.{name}" for name in _SUMMARY_NAMES),
    "free_anchor.present",
    *(f"free_anchor.position_error_world.{axis}" for axis in ("x", "y", "z")),
    *(f"free_anchor.orientation_error_body.{axis}" for axis in ("rx", "ry", "rz")),
    "ids.anchor_hash_sin_mean",
    "ids.anchor_hash_cos_mean",
    "ids.candidate_hash_sin_mean",
    "ids.candidate_hash_cos_mean",
    "ids.slot_hash_sin_mean",
    "ids.slot_hash_cos_mean",
)

_GLOBAL_INDEX = {
    name: index for index, name in enumerate(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
}
_NODE_INDEX = {
    name: index for index, name in enumerate(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES)
}


@dataclass(frozen=True)
class Order9ActiveKnotFeatureVectors:
    global_features: tuple[float, ...]
    assignment_features: tuple[tuple[float, ...], ...]
    module_ids: tuple[int, ...]

    def validate(self) -> None:
        if len(self.global_features) != len(
            ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
        ):
            raise SchemaValidationError("Order9 active-knot global feature width differs")
        if len(self.assignment_features) != len(self.module_ids):
            raise SchemaValidationError("Order9 active-knot node/module widths differ")
        if tuple(sorted(self.module_ids)) != self.module_ids:
            raise SchemaValidationError("Order9 active-knot module order is not canonical")
        if any(
            len(row) != len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES)
            for row in self.assignment_features
        ):
            raise SchemaValidationError("Order9 active-knot assignment feature width differs")
        values = [*self.global_features]
        for row in self.assignment_features:
            values.extend(row)
        if not all(math.isfinite(float(value)) for value in values):
            raise SchemaValidationError("Order9 active-knot features must be finite")


def order9_active_knot_feature_vectors(
    context: LowLevelPolicyContext,
    *,
    body_pose_world: Sequence[float] | None = None,
    body_twist_world: Sequence[float] | None = None,
) -> Order9ActiveKnotFeatureVectors:
    """Encode one schema-level context without simulator-only observations."""

    context.morphology_graph.validate()
    context.physical_model.validate()
    context.contact_wrench_trajectory.validate()
    knot = select_active_knot(context)
    observation = context.runtime_observation
    module_ids = tuple(
        sorted(module.module_id for module in context.morphology_graph.modules)
    )
    state_by_module = {state.module_id: state for state in observation.module_states}
    if set(module_ids) != set(state_by_module):
        raise SchemaValidationError(
            "Order9 active-knot runtime/morphology module identities differ"
        )
    if body_pose_world is None or body_twist_world is None:
        base = state_by_module[context.morphology_graph.base_module_id]
        body_pose_world = base.pose_world
        body_twist_world = base.twist_world
    body_pose = _pose(body_pose_world, "body_pose_world")
    body_twist = _vector(body_twist_world, 6, "body_twist_world")

    global_values = [0.0] * len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
    node_values = [
        [0.0] * len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES)
        for _ in module_ids
    ]
    module_index = {module_id: index for index, module_id in enumerate(module_ids)}
    anchor_by_id = {
        anchor.anchor_id: anchor for anchor in context.morphology_graph.robot_anchors
    }
    assignments_by_module: dict[int, list[ContactAssignment]] = {
        module_id: [] for module_id in module_ids
    }
    for assignment in knot.contact_assignments:
        anchor = anchor_by_id.get(assignment.anchor_id)
        if anchor is None or anchor.module_id not in assignments_by_module:
            raise SchemaValidationError(
                "Order9 active-knot assignment references an unknown robot anchor"
            )
        assignments_by_module[anchor.module_id].append(assignment)

    _set_global_static(
        global_values,
        trajectory=context.contact_wrench_trajectory,
        knot=knot,
        morphology=context.morphology_graph,
        physical_model=context.physical_model,
        progress=float(observation.task_progress.progress_ratio),
        time_s=float(observation.time_s),
    )
    _set_assignment_aggregate(global_values, knot.contact_assignments, _GLOBAL_INDEX)
    for module_id, assignments in assignments_by_module.items():
        _set_assignment_aggregate(
            node_values[module_index[module_id]], assignments, _NODE_INDEX
        )
    _set_scalar_targets(
        global_values,
        node_values,
        knot=knot,
        observation=observation,
        body_pose=body_pose,
        body_twist=body_twist,
        module_ids=module_ids,
        module_index=module_index,
        anchor_by_id=anchor_by_id,
    )
    _set_scalar_controller(global_values, context)
    output = Order9ActiveKnotFeatureVectors(
        global_features=tuple(global_values),
        assignment_features=tuple(tuple(row) for row in node_values),
        module_ids=module_ids,
    )
    output.validate()
    return output


class Order9ActiveKnotTensorTemplate:
    """Cached single-topology active-knot contract for vectorized Isaac."""

    def __init__(
        self,
        *,
        trajectory: ContactWrenchTrajectory,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        module_ids: Sequence[int],
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        trajectory.validate()
        morphology_graph.validate()
        physical_model.validate()
        if batch_size < 1 or not dtype.is_floating_point:
            raise ValueError("Order9 active-knot tensor template shape/dtype is invalid")
        canonical_ids = tuple(
            sorted(module.module_id for module in morphology_graph.modules)
        )
        if tuple(int(value) for value in module_ids) != canonical_ids:
            raise SchemaValidationError(
                "Order9 active-knot tensor module order differs from morphology"
            )
        self.trajectory = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
        self.morphology_graph = MorphologyGraph.from_dict(
            morphology_graph.to_dict()
        )
        self.physical_model = PhysicalModel.from_dict(physical_model.to_dict())
        self.module_ids = canonical_ids
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self._module_index = {
            module_id: index for index, module_id in enumerate(self.module_ids)
        }
        self._anchor_by_id = {
            anchor.anchor_id: anchor
            for anchor in self.morphology_graph.robot_anchors
        }
        self._knots = tuple(
            self._knot_for_schedule(schedule) for schedule in range(5)
        )
        static_global = []
        static_node = []
        for schedule, knot in enumerate(self._knots):
            global_values = [0.0] * len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)
            node_values = [
                [0.0] * len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES)
                for _ in self.module_ids
            ]
            _set_global_static(
                global_values,
                trajectory=self.trajectory,
                knot=knot,
                morphology=self.morphology_graph,
                physical_model=self.physical_model,
                progress=0.0,
                time_s=0.0,
            )
            _set_assignment_aggregate(
                global_values, knot.contact_assignments, _GLOBAL_INDEX
            )
            by_module: dict[int, list[ContactAssignment]] = {
                module_id: [] for module_id in self.module_ids
            }
            for assignment in knot.contact_assignments:
                anchor = self._anchor_by_id.get(assignment.anchor_id)
                if anchor is None:
                    raise SchemaValidationError(
                        "Order9 tensor active-knot assignment anchor is unknown"
                    )
                by_module[anchor.module_id].append(assignment)
            for module_id, assignments in by_module.items():
                _set_assignment_aggregate(
                    node_values[self._module_index[module_id]],
                    assignments,
                    _NODE_INDEX,
                )
            static_global.append(global_values)
            static_node.append(node_values)
        self._static_global = torch.tensor(
            static_global, device=self.device, dtype=self.dtype
        )
        self._static_node = torch.tensor(
            static_node, device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def features(
        self,
        *,
        time_s: torch.Tensor,
        phase_progress: torch.Tensor,
        contact_schedule_index: torch.Tensor,
        body_pose_world: torch.Tensor,
        body_twist_world: torch.Tensor,
        object_pose_world: torch.Tensor,
        object_twist_world: torch.Tensor,
        desired_body_pose_world: torch.Tensor,
        desired_body_twist_world: torch.Tensor,
        desired_object_pose_world: torch.Tensor,
        current_joint_positions_rad: torch.Tensor,
        current_joint_velocities_radps: torch.Tensor,
        desired_joint_positions_rad: torch.Tensor,
        desired_joint_velocities_radps: torch.Tensor,
        controller_qp_feasible: torch.Tensor,
        controller_status_one_hot: torch.Tensor,
        allocation_residual_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch_size
        expected = {
            "time_s": (batch,),
            "phase_progress": (batch,),
            "contact_schedule_index": (batch,),
            "body_pose_world": (batch, 7),
            "body_twist_world": (batch, 6),
            "object_pose_world": (batch, 7),
            "object_twist_world": (batch, 6),
            "desired_body_pose_world": (batch, 7),
            "desired_body_twist_world": (batch, 6),
            "desired_object_pose_world": (batch, 7),
            "controller_qp_feasible": (batch,),
            "controller_status_one_hot": (batch, 4),
            "allocation_residual_norm": (batch,),
        }
        local_values = locals()
        for name, shape in expected.items():
            if tuple(local_values[name].shape) != shape:
                raise ValueError(
                    f"Order9 tensor active-knot {name} must have shape {shape}"
                )
        node_joint_shape = current_joint_positions_rad.shape
        if (
            len(node_joint_shape) != 3
            or node_joint_shape[:2] != (batch, len(self.module_ids))
            or current_joint_velocities_radps.shape != node_joint_shape
            or desired_joint_positions_rad.shape != node_joint_shape
            or desired_joint_velocities_radps.shape != node_joint_shape
        ):
            raise ValueError("Order9 tensor active-knot joint shapes differ")
        schedule = contact_schedule_index.to(device=self.device, dtype=torch.long)
        if bool((schedule < 0).any()) or bool((schedule > 4).any()):
            raise ValueError("Order9 tensor active-knot schedule index is invalid")
        global_values = self._static_global.index_select(0, schedule).clone()
        node_values = self._static_node.index_select(0, schedule).clone()
        self._set_tensor_targets(
            global_values,
            node_values,
            schedule=schedule,
            time_s=time_s,
            phase_progress=phase_progress,
            body_pose_world=body_pose_world,
            body_twist_world=body_twist_world,
            object_pose_world=object_pose_world,
            object_twist_world=object_twist_world,
            desired_body_pose_world=desired_body_pose_world,
            desired_body_twist_world=desired_body_twist_world,
            desired_object_pose_world=desired_object_pose_world,
            current_joint_positions_rad=current_joint_positions_rad,
            current_joint_velocities_radps=current_joint_velocities_radps,
            desired_joint_positions_rad=desired_joint_positions_rad,
            desired_joint_velocities_radps=desired_joint_velocities_radps,
            controller_qp_feasible=controller_qp_feasible,
            controller_status_one_hot=controller_status_one_hot,
            allocation_residual_norm=allocation_residual_norm,
        )
        if not bool(torch.isfinite(global_values).all()) or not bool(
            torch.isfinite(node_values).all()
        ):
            raise ValueError("Order9 tensor active-knot features are non-finite")
        return global_values, node_values

    def _knot_for_schedule(self, schedule_index: int) -> InteractionKnot:
        if schedule_index == 0:
            source = self.trajectory.knots[-1]
            return InteractionKnot(
                t_rel_s=float(source.t_rel_s),
                contact_assignments=[],
                centroidal_target=source.centroidal_target,
                posture_target=source.posture_target,
                object_targets=list(source.object_targets),
                priority_weights=dict(source.priority_weights),
                guard_conditions=list(source.guard_conditions),
            )
        label = {
            1: "approach",
            2: "attach",
            3: "maintain",
            4: "release",
        }[schedule_index]
        source = next(
            (
                knot
                for knot in self.trajectory.knots
                if any(
                    assignment.schedule_state == label
                    for assignment in knot.contact_assignments
                )
            ),
            self.trajectory.knots[-1],
        )
        assignments = []
        for original in source.contact_assignments:
            assignment = ContactAssignment.from_dict(original.to_dict())
            assignment.schedule_state = label
            if label in {"approach", "release"}:
                assignment.wrench_target = None
                assignment.wrench_lower = None
                assignment.wrench_upper = None
            assignment.validate()
            assignments.append(assignment)
        return InteractionKnot(
            t_rel_s=float(source.t_rel_s),
            contact_assignments=assignments,
            centroidal_target=source.centroidal_target,
            posture_target=source.posture_target,
            object_targets=list(source.object_targets),
            priority_weights=dict(source.priority_weights),
            guard_conditions=list(source.guard_conditions),
        )

    def _set_tensor_targets(
        self,
        global_values: torch.Tensor,
        node_values: torch.Tensor,
        **values: torch.Tensor,
    ) -> None:
        time_s = values["time_s"].to(device=self.device, dtype=self.dtype)
        progress = values["phase_progress"].to(
            device=self.device, dtype=self.dtype
        )
        body_pose = values["body_pose_world"].to(
            device=self.device, dtype=self.dtype
        )
        body_twist = values["body_twist_world"].to(
            device=self.device, dtype=self.dtype
        )
        object_pose = values["object_pose_world"].to(
            device=self.device, dtype=self.dtype
        )
        object_twist = values["object_twist_world"].to(
            device=self.device, dtype=self.dtype
        )
        desired_body_pose = values["desired_body_pose_world"].to(
            device=self.device, dtype=self.dtype
        )
        desired_body_twist = values["desired_body_twist_world"].to(
            device=self.device, dtype=self.dtype
        )
        desired_object_pose = values["desired_object_pose_world"].to(
            device=self.device, dtype=self.dtype
        )
        global_values[:, _GLOBAL_INDEX["active_knot.time_over_horizon"]] = (
            time_s / max(float(self.trajectory.horizon_s), 1.0e-6)
        ).clamp(0.0, 4.0)
        global_values[:, _GLOBAL_INDEX["phase.progress"]] = progress.clamp(0.0, 1.0)
        global_values[:, _GLOBAL_INDEX["centroidal.position_present"]] = 1.0
        _set_tensor_named_vector(
            global_values,
            "centroidal.position_error_world",
            desired_body_pose[:, :3] - body_pose[:, :3],
            ("x", "y", "z"),
        )
        global_values[:, _GLOBAL_INDEX["centroidal.velocity_present"]] = 1.0
        _set_tensor_named_vector(
            global_values,
            "centroidal.velocity_error_world",
            desired_body_twist[:, :3] - body_twist[:, :3],
            ("x", "y", "z"),
        )
        global_values[:, _GLOBAL_INDEX["centroidal.orientation_present"]] = 1.0
        _set_tensor_named_vector(
            global_values,
            "centroidal.orientation_error_body",
            _quaternion_error_rotvec_tensor(
                body_pose[:, 3:7], desired_body_pose[:, 3:7]
            ),
            ("rx", "ry", "rz"),
        )
        global_values[:, _GLOBAL_INDEX["object.pose_present"]] = 1.0
        _set_tensor_named_vector(
            global_values,
            "object.position_error_world",
            desired_object_pose[:, :3] - object_pose[:, :3],
            ("x", "y", "z"),
        )
        _set_tensor_named_vector(
            global_values,
            "object.orientation_error_body",
            _quaternion_error_rotvec_tensor(
                object_pose[:, 3:7], desired_object_pose[:, 3:7]
            ),
            ("rx", "ry", "rz"),
        )
        # The current deterministic task target has no planned object twist.
        global_values[:, _GLOBAL_INDEX["object.twist_present"]] = 0.0
        _set_tensor_named_vector(
            global_values,
            "object.twist_error",
            torch.zeros_like(object_twist),
            ("vx", "vy", "vz", "wx", "wy", "wz"),
        )
        q_error = values["desired_joint_positions_rad"].to(
            device=self.device, dtype=self.dtype
        ) - values["current_joint_positions_rad"].to(
            device=self.device, dtype=self.dtype
        )
        qdot_error = values["desired_joint_velocities_radps"].to(
            device=self.device, dtype=self.dtype
        ) - values["current_joint_velocities_radps"].to(
            device=self.device, dtype=self.dtype
        )
        global_values[:, _GLOBAL_INDEX["posture.position_present"]] = 1.0
        global_values[:, _GLOBAL_INDEX["posture.velocity_present"]] = 1.0
        _set_tensor_summary(
            global_values, q_error.flatten(1), "posture.position_error", _GLOBAL_INDEX
        )
        _set_tensor_summary(
            global_values, qdot_error.flatten(1), "posture.velocity_error", _GLOBAL_INDEX
        )
        for module_index in range(len(self.module_ids)):
            _set_tensor_summary(
                node_values[:, module_index],
                q_error[:, module_index],
                "posture.position_error",
                _NODE_INDEX,
            )
            _set_tensor_summary(
                node_values[:, module_index],
                qdot_error[:, module_index],
                "posture.velocity_error",
                _NODE_INDEX,
            )
        qp = values["controller_qp_feasible"].to(
            device=self.device, dtype=self.dtype
        )
        status = values["controller_status_one_hot"].to(
            device=self.device, dtype=self.dtype
        )
        residual = values["allocation_residual_norm"].to(
            device=self.device, dtype=self.dtype
        )
        global_values[:, _GLOBAL_INDEX["controller.qp_feasible"]] = qp
        for index, label in enumerate(("ok", "warning", "infeasible", "fault")):
            global_values[:, _GLOBAL_INDEX[f"controller.status.{label}"]] = status[
                :, index
            ]
        global_values[
            :, _GLOBAL_INDEX["controller.signed_log_allocation_residual"]
        ] = _signed_log1p_tensor(residual)


def _set_global_static(
    output: list[float],
    *,
    trajectory: ContactWrenchTrajectory,
    knot: InteractionKnot,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    progress: float,
    time_s: float,
) -> None:
    output[_GLOBAL_INDEX["trajectory.signed_log_horizon_s"]] = _signed_log1p(
        trajectory.horizon_s
    )
    output[_GLOBAL_INDEX["trajectory.signed_log_dt_s"]] = _signed_log1p(
        trajectory.dt_s
    )
    output[_GLOBAL_INDEX["active_knot.time_over_horizon"]] = min(
        max(float(time_s) / max(float(trajectory.horizon_s), 1.0e-6), 0.0), 4.0
    )
    output[_GLOBAL_INDEX["phase.progress"]] = min(max(float(progress), 0.0), 1.0)
    weights = knot.priority_weights
    output[_GLOBAL_INDEX["priority_weights.count"]] = _signed_log1p(len(weights))
    output[_GLOBAL_INDEX["priority_weights.value_sum"]] = _signed_log1p(
        sum(float(value) for value in weights.values())
    )
    output[_GLOBAL_INDEX["priority_weights.key_hash_sin_sum"]] = sum(
        _stable_pair(key)[0] for key in weights
    )
    output[_GLOBAL_INDEX["priority_weights.key_hash_cos_sum"]] = sum(
        _stable_pair(key)[1] for key in weights
    )
    guards = knot.guard_conditions
    output[_GLOBAL_INDEX["guards.count"]] = _signed_log1p(len(guards))
    guard_pairs = [
        _stable_pair(json.dumps(guard, sort_keys=True, separators=(",", ":")))
        for guard in guards
    ]
    output[_GLOBAL_INDEX["guards.hash_sin_sum"]] = sum(
        pair[0] for pair in guard_pairs
    )
    output[_GLOBAL_INDEX["guards.hash_cos_sum"]] = sum(
        pair[1] for pair in guard_pairs
    )
    module_count = len(morphology.modules)
    output[_GLOBAL_INDEX["morphology.module_count_over_eight"]] = (
        float(module_count) / 8.0
    )
    output[_GLOBAL_INDEX["physical.signed_log_total_mass_kg"]] = _signed_log1p(
        physical_model.aggregate_mass_kg * module_count
    )
    for axis, value in zip(
        _WRENCH_AXES,
        [float(item) * module_count for item in physical_model.aggregate_inertia_body],
    ):
        output[_GLOBAL_INDEX[f"physical.signed_log_total_inertia.{axis}"]] = (
            _signed_log1p(value)
        )
    output[_GLOBAL_INDEX["physical.signed_log_rotor_count"]] = _signed_log1p(
        len(physical_model.rotors) * module_count
    )
    output[_GLOBAL_INDEX["physical.signed_log_total_max_thrust_n"]] = _signed_log1p(
        sum(float(rotor.thrust_max_n) for rotor in physical_model.rotors)
        * module_count
    )
    output[_GLOBAL_INDEX["physical.signed_log_non_fixed_joint_count"]] = (
        _signed_log1p(
            sum(joint.joint_type != "fixed" for joint in physical_model.joints)
            * module_count
        )
    )
    centroidal = knot.centroidal_target
    if centroidal is not None and centroidal.centroidal_wrench_preference is not None:
        output[_GLOBAL_INDEX["centroidal.wrench_preference_present"]] = 1.0
        for axis, value in zip(
            _WRENCH_AXES, centroidal.centroidal_wrench_preference
        ):
            output[
                _GLOBAL_INDEX[f"centroidal.signed_log_wrench_preference.{axis}"]
            ] = _signed_log1p(float(value))


def _set_assignment_aggregate(
    output: list[float],
    assignments: Sequence[ContactAssignment],
    index: dict[str, int],
) -> None:
    count = len(assignments)
    output[index["assignments.signed_log_count"]] = _signed_log1p(count)
    if count == 0:
        return
    inverse = 1.0 / count
    for assignment in assignments:
        output[
            index[f"assignments.schedule_fraction.{assignment.schedule_state}"]
        ] += inverse
        output[
            index[f"assignments.mode_fraction.{assignment.contact_mode.value}"]
        ] += inverse
    priorities = [float(assignment.priority) for assignment in assignments]
    priority_names = (
        ("assignments.priority.count", float(count)),
        ("assignments.priority.mean", sum(priorities) * inverse),
        ("assignments.priority.minimum", min(priorities)),
        ("assignments.priority.maximum", max(priorities)),
    )
    for name, value in priority_names:
        if name in index:
            output[index[name]] = _signed_log1p(value)
    for field, label in (
        ("wrench_target", "target"),
        ("wrench_lower", "lower"),
        ("wrench_upper", "upper"),
    ):
        present = [
            getattr(assignment, field)
            for assignment in assignments
            if getattr(assignment, field) is not None
        ]
        output[index[f"wrench.{label}_fraction"]] = len(present) * inverse
        if present:
            for axis_index, axis in enumerate(_WRENCH_AXES):
                output[index[f"wrench.signed_log_{label}_mean.{axis}"]] = (
                    _signed_log1p(
                        sum(float(row[axis_index]) for row in present) / len(present)
                    )
                )
    bounded = [
        (assignment.wrench_lower, assignment.wrench_upper)
        for assignment in assignments
        if assignment.wrench_lower is not None
        and assignment.wrench_upper is not None
    ]
    if bounded:
        for axis_index, axis in enumerate(_WRENCH_AXES):
            output[index[f"wrench.signed_log_width_mean.{axis}"]] = _signed_log1p(
                sum(
                    float(upper[axis_index]) - float(lower[axis_index])
                    for lower, upper in bounded
                )
                / len(bounded)
            )
    for id_name, getter in (
        ("anchor", lambda assignment: assignment.anchor_id),
        ("candidate", lambda assignment: assignment.candidate_id),
        ("slot", lambda assignment: assignment.slot_id),
    ):
        sin_name = f"ids.{id_name}_hash_sin_mean"
        cos_name = f"ids.{id_name}_hash_cos_mean"
        if sin_name not in index:
            continue
        pairs = [_stable_pair(f"{id_name}:{getter(item)}") for item in assignments]
        output[index[sin_name]] = sum(pair[0] for pair in pairs) * inverse
        output[index[cos_name]] = sum(pair[1] for pair in pairs) * inverse


def _set_scalar_targets(
    global_values: list[float],
    node_values: list[list[float]],
    *,
    knot: InteractionKnot,
    observation: RuntimeObservation,
    body_pose: Pose7D,
    body_twist: list[float],
    module_ids: tuple[int, ...],
    module_index: dict[int, int],
    anchor_by_id: dict[int, object],
) -> None:
    centroidal = knot.centroidal_target
    if centroidal is not None:
        if centroidal.com_pos_world is not None:
            global_values[_GLOBAL_INDEX["centroidal.position_present"]] = 1.0
            _set_named_vector(
                global_values,
                _GLOBAL_INDEX,
                "centroidal.position_error_world",
                [
                    float(target) - float(current)
                    for target, current in zip(
                        centroidal.com_pos_world, body_pose[:3]
                    )
                ],
                ("x", "y", "z"),
            )
        if centroidal.com_vel_world is not None:
            global_values[_GLOBAL_INDEX["centroidal.velocity_present"]] = 1.0
            _set_named_vector(
                global_values,
                _GLOBAL_INDEX,
                "centroidal.velocity_error_world",
                [
                    float(target) - float(current)
                    for target, current in zip(
                        centroidal.com_vel_world, body_twist[:3]
                    )
                ],
                ("x", "y", "z"),
            )
        if centroidal.body_orientation_world is not None:
            global_values[_GLOBAL_INDEX["centroidal.orientation_present"]] = 1.0
            _set_named_vector(
                global_values,
                _GLOBAL_INDEX,
                "centroidal.orientation_error_body",
                _quaternion_error_rotvec(
                    body_pose[3:7], centroidal.body_orientation_world
                ),
                ("rx", "ry", "rz"),
            )
    object_by_id = {state.object_id: state for state in observation.object_states}
    target = next(
        (
            item
            for item in knot.object_targets
            if item.object_id in object_by_id
        ),
        None,
    )
    if target is not None:
        state = object_by_id[target.object_id]
        if target.pose_target_world is not None:
            global_values[_GLOBAL_INDEX["object.pose_present"]] = 1.0
            _set_named_vector(
                global_values,
                _GLOBAL_INDEX,
                "object.position_error_world",
                [
                    float(desired) - float(current)
                    for desired, current in zip(
                        target.pose_target_world[:3], state.pose_world[:3]
                    )
                ],
                ("x", "y", "z"),
            )
            _set_named_vector(
                global_values,
                _GLOBAL_INDEX,
                "object.orientation_error_body",
                _quaternion_error_rotvec(
                    state.pose_world[3:7], target.pose_target_world[3:7]
                ),
                ("rx", "ry", "rz"),
            )
        if target.twist_target_world is not None:
            global_values[_GLOBAL_INDEX["object.twist_present"]] = 1.0
            _set_named_vector(
                global_values,
                _GLOBAL_INDEX,
                "object.twist_error",
                [
                    float(desired) - float(current)
                    for desired, current in zip(
                        target.twist_target_world, state.twist_world
                    )
                ],
                ("vx", "vy", "vz", "wx", "wy", "wz"),
            )
    posture = knot.posture_target
    state_by_module = {state.module_id: state for state in observation.module_states}
    if posture is not None:
        q_errors = _joint_errors(
            posture.joint_pos_target or {}, state_by_module, "joint_positions"
        )
        qdot_errors = _joint_errors(
            posture.joint_vel_target or {}, state_by_module, "joint_velocities"
        )
        if q_errors:
            global_values[_GLOBAL_INDEX["posture.position_present"]] = 1.0
        if qdot_errors:
            global_values[_GLOBAL_INDEX["posture.velocity_present"]] = 1.0
        _set_summary(
            global_values, _GLOBAL_INDEX, "posture.position_error", q_errors.values()
        )
        _set_summary(
            global_values,
            _GLOBAL_INDEX,
            "posture.velocity_error",
            qdot_errors.values(),
        )
        for module_id in module_ids:
            _set_summary(
                node_values[module_index[module_id]],
                _NODE_INDEX,
                "posture.position_error",
                [
                    value
                    for key, value in q_errors.items()
                    if key[0] == module_id
                ],
            )
            _set_summary(
                node_values[module_index[module_id]],
                _NODE_INDEX,
                "posture.velocity_error",
                [
                    value
                    for key, value in qdot_errors.items()
                    if key[0] == module_id
                ],
            )
        free = posture.free_anchor_pose_targets or {}
        global_values[
            _GLOBAL_INDEX["posture.signed_log_free_anchor_count"]
        ] = _signed_log1p(len(free))
        free_by_module: dict[int, list[tuple[Pose7D, Pose7D]]] = {
            module_id: [] for module_id in module_ids
        }
        for anchor_id, target_pose in free.items():
            anchor = anchor_by_id.get(anchor_id)
            if anchor is None:
                raise SchemaValidationError(
                    "Order9 free-anchor posture target references unknown anchor"
                )
            module_id = int(getattr(anchor, "module_id"))
            free_by_module[module_id].append(
                (
                    state_by_module[module_id].pose_world,
                    _pose(target_pose, "free_anchor_pose_target"),
                )
            )
        for module_id, rows in free_by_module.items():
            if not rows:
                continue
            output = node_values[module_index[module_id]]
            output[_NODE_INDEX["free_anchor.present"]] = 1.0
            position = [
                sum(float(target[index]) - float(current[index]) for current, target in rows)
                / len(rows)
                for index in range(3)
            ]
            orientation_rows = [
                _quaternion_error_rotvec(current[3:7], target[3:7])
                for current, target in rows
            ]
            orientation = [
                sum(row[index] for row in orientation_rows) / len(rows)
                for index in range(3)
            ]
            _set_named_vector(
                output,
                _NODE_INDEX,
                "free_anchor.position_error_world",
                position,
                ("x", "y", "z"),
            )
            _set_named_vector(
                output,
                _NODE_INDEX,
                "free_anchor.orientation_error_body",
                orientation,
                ("rx", "ry", "rz"),
            )


def _set_scalar_controller(
    output: list[float], context: LowLevelPolicyContext
) -> None:
    status = context.controller_status or context.runtime_observation.controller_status
    output[_GLOBAL_INDEX["controller.qp_feasible"]] = float(status.qp_feasible)
    for label in ("ok", "warning", "infeasible", "fault"):
        output[_GLOBAL_INDEX[f"controller.status.{label}"]] = float(
            status.status == label
        )
    residual = float(status.metrics.get("allocation_residual_norm", 0.0))
    output[_GLOBAL_INDEX["controller.signed_log_allocation_residual"]] = (
        _signed_log1p(residual)
    )
    mode_sin, mode_cos = _stable_pair(status.active_mode or "none")
    output[_GLOBAL_INDEX["controller.active_mode_hash_sin"]] = mode_sin
    output[_GLOBAL_INDEX["controller.active_mode_hash_cos"]] = mode_cos


def _joint_errors(
    targets: dict[str, float],
    state_by_module: dict[int, object],
    field: str,
) -> dict[tuple[int, str], float]:
    output: dict[tuple[int, str], float] = {}
    for global_id, target in targets.items():
        if not global_id.startswith("module_") or ":" not in global_id:
            raise SchemaValidationError(
                "Order9 posture target joint identity is not module-qualified"
            )
        module_text, joint_id = global_id.split(":", 1)
        try:
            module_id = int(module_text[len("module_") :])
        except ValueError as exc:
            raise SchemaValidationError(
                "Order9 posture target module identity is invalid"
            ) from exc
        state = state_by_module.get(module_id)
        values = None if state is None else getattr(state, field)
        if values is None or joint_id not in values:
            raise SchemaValidationError(
                "Order9 posture target has no matching runtime joint state"
            )
        output[(module_id, joint_id)] = float(target) - float(values[joint_id])
    return output


def _set_summary(
    output: list[float],
    index: dict[str, int],
    prefix: str,
    raw_values: Iterable[float],
) -> None:
    values = [float(value) for value in raw_values]
    if not values:
        return
    count = len(values)
    rows = {
        "count": _signed_log1p(count),
        "mean": _signed_log1p(sum(values) / count),
        "rms": _signed_log1p(math.sqrt(sum(value * value for value in values) / count)),
        "minimum": _signed_log1p(min(values)),
        "maximum": _signed_log1p(max(values)),
    }
    for name, value in rows.items():
        output[index[f"{prefix}.{name}"]] = value


def _set_tensor_summary(
    output: torch.Tensor,
    values: torch.Tensor,
    prefix: str,
    index: dict[str, int],
) -> None:
    count = values.shape[-1]
    output[:, index[f"{prefix}.count"]] = _signed_log1p_tensor(
        torch.full(
            (values.shape[0],),
            float(count),
            device=values.device,
            dtype=values.dtype,
        )
    )
    output[:, index[f"{prefix}.mean"]] = _signed_log1p_tensor(values.mean(dim=-1))
    output[:, index[f"{prefix}.rms"]] = _signed_log1p_tensor(
        values.square().mean(dim=-1).sqrt()
    )
    output[:, index[f"{prefix}.minimum"]] = _signed_log1p_tensor(
        values.min(dim=-1).values
    )
    output[:, index[f"{prefix}.maximum"]] = _signed_log1p_tensor(
        values.max(dim=-1).values
    )


def _set_named_vector(
    output: list[float],
    index: dict[str, int],
    prefix: str,
    values: Sequence[float],
    suffixes: Sequence[str],
) -> None:
    for suffix, value in zip(suffixes, values):
        output[index[f"{prefix}.{suffix}"]] = _signed_log1p(float(value))


def _set_tensor_named_vector(
    output: torch.Tensor,
    prefix: str,
    values: torch.Tensor,
    suffixes: Sequence[str],
) -> None:
    for value_index, suffix in enumerate(suffixes):
        output[:, _GLOBAL_INDEX[f"{prefix}.{suffix}"]] = _signed_log1p_tensor(
            values[:, value_index]
        )


def _quaternion_error_rotvec(
    current_xyzw: Sequence[float], target_xyzw: Sequence[float]
) -> list[float]:
    current = quat_to_matrix(tuple(float(value) for value in current_xyzw))
    target = quat_to_matrix(tuple(float(value) for value in target_xyzw))
    relative = tuple(
        tuple(
            sum(transpose(current)[row][k] * target[k][column] for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    cosine = max(
        -1.0,
        min(1.0, 0.5 * (sum(relative[index][index] for index in range(3)) - 1.0)),
    )
    angle = math.acos(cosine)
    if angle <= 1.0e-8:
        return [0.0, 0.0, 0.0]
    scale = angle / max(2.0 * math.sin(angle), 1.0e-8)
    return [
        scale * (relative[2][1] - relative[1][2]),
        scale * (relative[0][2] - relative[2][0]),
        scale * (relative[1][0] - relative[0][1]),
    ]


def _quaternion_error_rotvec_tensor(
    current_xyzw: torch.Tensor, target_xyzw: torch.Tensor
) -> torch.Tensor:
    current = _quaternion_to_matrix_tensor(current_xyzw)
    target = _quaternion_to_matrix_tensor(target_xyzw)
    relative = current.transpose(-1, -2) @ target
    cosine = (
        0.5
        * (
            relative[..., 0, 0]
            + relative[..., 1, 1]
            + relative[..., 2, 2]
            - 1.0
        )
    ).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    denominator = 2.0 * torch.sin(angle)
    scale = torch.where(
        angle <= 1.0e-8,
        torch.zeros_like(angle),
        angle / denominator.abs().clamp_min(1.0e-8),
    )
    return vector * scale.unsqueeze(-1)


def _quaternion_to_matrix_tensor(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _pose(values: Sequence[float], label: str) -> Pose7D:
    row = _vector(values, 7, label)
    return tuple(row)  # type: ignore[return-value]


def _vector(values: Sequence[float], width: int, label: str) -> list[float]:
    row = [float(value) for value in values]
    if len(row) != width or not all(math.isfinite(value) for value in row):
        raise SchemaValidationError(f"Order9 active-knot {label} is invalid")
    return row


def _stable_pair(value: str) -> tuple[float, float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    angle = 2.0 * math.pi * unit
    return math.sin(angle), math.cos(angle)


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _signed_log1p_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(value.abs())


__all__ = [
    "ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES",
    "ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION",
    "ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES",
    "Order9ActiveKnotFeatureVectors",
    "Order9ActiveKnotTensorTemplate",
    "order9_active_knot_feature_vectors",
]
