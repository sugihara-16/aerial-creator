from __future__ import annotations

"""Tensor-native hot path for topology-bucketed Order 9 ``pi_L`` rollout."""

from dataclasses import dataclass

import torch

from amsrr.encoders.morphology_graph_encoder import (
    MORPHOLOGY_NODE_FEATURE_NAMES,
    MorphologyGraphBatch,
    MorphologyGraphTensorizer,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import ORDER3_ACTION_SIZE
from amsrr.utils.hashing import stable_hash


ORDER9_TENSOR_RUNTIME_VERSION = "order9_topology_bucket_tensor_runtime_v1"


@dataclass(frozen=True)
class Order9CentroidalTensorObservation:
    time_s: torch.Tensor
    module_count: torch.Tensor
    total_mass_kg: torch.Tensor
    inertia_body: torch.Tensor
    body_pose_world: torch.Tensor
    body_twist_world: torch.Tensor
    target_pose_world: torch.Tensor
    target_twist: torch.Tensor
    controller_qp_feasible: torch.Tensor
    controller_status_one_hot: torch.Tensor
    allocation_residual_norm: torch.Tensor
    task_progress_ratio: torch.Tensor
    task_success: torch.Tensor


class Order9TensorizedTopologyBucket:
    """Cached static graph tensors with in-place runtime feature refresh.

    One instance is owned by one rollout worker. Callers must not share it
    across concurrent CUDA streams without external synchronization.
    """

    def __init__(
        self,
        morphology: MorphologyGraph,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Order9 topology bucket batch_size must be positive")
        single = MorphologyGraphTensorizer().tensorize(
            [morphology], device=device, dtype=dtype
        )
        self.batch_size = batch_size
        self.module_count = int(single.node_features.shape[1])
        self.structural_hash = stable_hash(
            {
                "modules": [module.to_dict() for module in morphology.modules],
                "ports": [port.to_dict() for port in morphology.ports],
                "edges": [edge.to_dict() for edge in morphology.dock_edges],
                "anchors": [anchor.to_dict() for anchor in morphology.robot_anchors],
                "control_groups": [group.to_dict() for group in morphology.control_groups],
                "base_module_id": morphology.base_module_id,
            }
        )
        self.batch = MorphologyGraphBatch(
            node_features=single.node_features.repeat(batch_size, 1, 1),
            node_mask=single.node_mask.repeat(batch_size, 1),
            edge_index=single.edge_index.repeat(batch_size, 1, 1),
            edge_features=single.edge_features.repeat(batch_size, 1, 1),
            edge_mask=single.edge_mask.repeat(batch_size, 1),
            module_ids=single.module_ids.repeat(batch_size, 1),
            edge_ids=single.edge_ids.repeat(batch_size, 1),
            graph_ids=tuple(morphology.graph_id for _ in range(batch_size)),
            metadata={
                **single.metadata,
                "tensor_runtime_version": ORDER9_TENSOR_RUNTIME_VERSION,
                "topology_bucketed": True,
                "structural_hash": self.structural_hash,
                "batch_size": batch_size,
            },
        )

    def update_runtime_(
        self,
        *,
        module_pose_world: torch.Tensor,
        module_twist_world: torch.Tensor,
        module_health: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_mask: torch.Tensor,
        strict: bool = False,
    ) -> MorphologyGraphBatch:
        batch, nodes = self.batch_size, self.module_count
        _shape(module_pose_world, (batch, nodes, 7), "module_pose_world")
        _shape(module_twist_world, (batch, nodes, 6), "module_twist_world")
        _shape(module_health, (batch, nodes), "module_health")
        if joint_positions.ndim != 3 or joint_positions.shape[:2] != (batch, nodes):
            raise ValueError("joint_positions must have shape [B, N, J]")
        _shape(joint_velocities, tuple(joint_positions.shape), "joint_velocities")
        _shape(joint_mask, tuple(joint_positions.shape), "joint_mask")
        if strict:
            tensors = (
                module_pose_world,
                module_twist_world,
                module_health,
                joint_positions,
                joint_velocities,
            )
            if any(not bool(torch.isfinite(item).all().item()) for item in tensors):
                raise ValueError("Order9 topology bucket runtime tensors must be finite")
        features = self.batch.node_features
        device, dtype = features.device, features.dtype
        features[..., _NODE_RUNTIME_PRESENT] = self.batch.node_mask.to(dtype)
        features[..., _NODE_RUNTIME_POSE] = module_pose_world.to(device=device, dtype=dtype)
        features[..., _NODE_RUNTIME_TWIST] = module_twist_world.to(device=device, dtype=dtype)
        features[..., _NODE_RUNTIME_HEALTH] = module_health.to(device=device, dtype=dtype)
        mask = joint_mask.to(device=device, dtype=torch.bool)
        features[..., _NODE_RUNTIME_JOINT_POSITION] = _masked_summary(
            joint_positions.to(device=device, dtype=dtype), mask
        )
        features[..., _NODE_RUNTIME_JOINT_VELOCITY] = _masked_summary(
            joint_velocities.to(device=device, dtype=dtype), mask
        )
        return self.batch


def order9_low_level_actor_features_from_tensors(
    observation: Order9CentroidalTensorObservation,
    *,
    max_modules: int = 8,
) -> torch.Tensor:
    """Vectorized equivalent of ``order3_actor_feature_vector``."""

    batch = int(observation.time_s.shape[0])
    _shape(observation.time_s, (batch,), "time_s")
    _shape(observation.module_count, (batch,), "module_count")
    _shape(observation.total_mass_kg, (batch,), "total_mass_kg")
    _shape(observation.inertia_body, (batch, 6), "inertia_body")
    _shape(observation.body_pose_world, (batch, 7), "body_pose_world")
    _shape(observation.body_twist_world, (batch, 6), "body_twist_world")
    _shape(observation.target_pose_world, (batch, 7), "target_pose_world")
    _shape(observation.target_twist, (batch, 6), "target_twist")
    _shape(observation.controller_qp_feasible, (batch,), "controller_qp_feasible")
    _shape(
        observation.controller_status_one_hot,
        (batch, 4),
        "controller_status_one_hot",
    )
    _shape(observation.allocation_residual_norm, (batch,), "allocation_residual_norm")
    _shape(observation.task_progress_ratio, (batch,), "task_progress_ratio")
    _shape(observation.task_success, (batch,), "task_success")
    if max_modules < 1:
        raise ValueError("Order9 actor max_modules must be positive")
    pose = observation.body_pose_world
    rotation = _quaternion_to_matrix(pose[:, 3:7])
    current_angular_body = torch.bmm(
        rotation.transpose(1, 2), observation.body_twist_world[:, 3:6].unsqueeze(-1)
    ).squeeze(-1)
    position_error = observation.target_pose_world[:, :3] - pose[:, :3]
    orientation_error = _orientation_error_body(
        pose[:, 3:7], observation.target_pose_world[:, 3:7]
    )
    linear_error = observation.target_twist[:, :3] - observation.body_twist_world[:, :3]
    angular_error = observation.target_twist[:, 3:6] - current_angular_body
    phase = observation.time_s * 0.2
    features = torch.cat(
        (
            torch.sin(phase).unsqueeze(-1),
            torch.cos(phase).unsqueeze(-1),
            (observation.module_count / float(max_modules)).unsqueeze(-1),
            _signed_log1p(observation.total_mass_kg).unsqueeze(-1),
            _signed_log1p(observation.inertia_body),
            position_error,
            orientation_error,
            observation.body_twist_world[:, :3],
            current_angular_body,
            observation.target_twist,
            linear_error,
            angular_error,
            observation.controller_qp_feasible.to(pose.dtype).unsqueeze(-1),
            observation.controller_status_one_hot.to(pose.dtype),
            _signed_log1p(observation.allocation_residual_norm).unsqueeze(-1),
            observation.task_progress_ratio.unsqueeze(-1),
            observation.task_success.to(pose.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    if features.shape[1] != len(_ORDER3_ACTOR_FEATURE_NAMES):
        raise RuntimeError("Order9 tensor actor feature layout mismatch")
    return features


def initial_order9_policy_state(
    batch_size: int,
    recurrent_hidden_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((batch_size, ORDER3_ACTION_SIZE), device=device, dtype=dtype),
        torch.zeros(
            (batch_size, recurrent_hidden_dim), device=device, dtype=dtype
        ),
    )


def _masked_summary(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1)
    safe_count = count.clamp_min(1.0)
    mean = (values * weights).sum(dim=-1) / safe_count
    rms = torch.sqrt((values.square() * weights).sum(dim=-1) / safe_count)
    minimum = values.masked_fill(~mask, torch.inf).min(dim=-1).values
    maximum = values.masked_fill(~mask, -torch.inf).max(dim=-1).values
    empty = count <= 0.0
    minimum = torch.where(empty, torch.zeros_like(minimum), minimum)
    maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
    return torch.stack((count, mean, rms, minimum, maximum), dim=-1)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
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
    ).reshape(-1, 3, 3)


def _orientation_error_body(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    current_xyz, current_w = current[:, :3], current[:, 3:4]
    target_xyz, target_w = target[:, :3], target[:, 3:4]
    inverse_xyz = -current_xyz
    relative_xyz = (
        current_w * target_xyz
        + target_w * inverse_xyz
        + torch.cross(inverse_xyz, target_xyz, dim=-1)
    )
    relative_w = current_w * target_w - (
        inverse_xyz * target_xyz
    ).sum(dim=-1, keepdim=True)
    sign = torch.where(relative_w < 0.0, -1.0, 1.0)
    relative_xyz = relative_xyz * sign
    relative_w = relative_w * sign
    axis_norm = relative_xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(axis_norm, relative_w.clamp_min(1.0e-8))
    return torch.where(
        axis_norm > 1.0e-8,
        relative_xyz / axis_norm.clamp_min(1.0e-8) * angle,
        torch.zeros_like(relative_xyz),
    )


def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(value.abs())


def _shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


_NODE_RUNTIME_PRESENT = MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.present")
_NODE_RUNTIME_POSE = slice(
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.pose.x"),
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.pose.qw") + 1,
)
_NODE_RUNTIME_TWIST = slice(
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.twist.vx"),
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.twist.wz") + 1,
)
_NODE_RUNTIME_HEALTH = MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.health")
_NODE_RUNTIME_JOINT_POSITION = slice(
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.joint_position.count"),
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.joint_position.max") + 1,
)
_NODE_RUNTIME_JOINT_VELOCITY = slice(
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.joint_velocity.count"),
    MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.joint_velocity.max") + 1,
)

# Imported lazily in the public policy module in normal operation. Keeping only
# the names here makes the hot-path width assertion explicit without schemas.
from amsrr.policies.morphology_conditioned_low_level_policy import (  # noqa: E402
    ORDER3_ACTOR_FEATURE_NAMES as _ORDER3_ACTOR_FEATURE_NAMES,
)
