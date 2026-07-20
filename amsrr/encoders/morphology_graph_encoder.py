from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from amsrr.geometry.pose_math import inverse_pose, normalize_quat
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.morphology import DockEdge, ModuleNode, MorphologyGraph, PortNode
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation


MORPHOLOGY_GRAPH_TENSORIZER_VERSION = "order3_homogeneous_module_graph_tensor_v1"
MORPHOLOGY_GRAPH_ENCODER_VERSION = "order3_module_graph_message_passing_v1"
MORPHOLOGY_MODULE_TOKEN_TYPE_ID = 1
SOURCE_TYPE_MORPHOLOGY_MODULE = 60

_PORT_TYPES = ("pitch_dock", "yaw_dock", "generic_dock", "other")
_EDGE_ROLES = (
    "structural",
    "grasp_arm",
    "support",
    "perch_anchor",
    "locomotion_support",
)
_LATCH_STATES = ("planned", "attached", "detached")

_POSE_FEATURE_SUFFIXES = ("x", "y", "z", "qx", "qy", "qz", "qw")
_TWIST_FEATURE_SUFFIXES = ("vx", "vy", "vz", "wx", "wy", "wz")
_SUMMARY_SUFFIXES = ("count", "mean", "rms", "min", "max")

MORPHOLOGY_NODE_FEATURE_NAMES: tuple[str, ...] = (
    *(f"design_pose.{name}" for name in _POSE_FEATURE_SUFFIXES),
    "module.health",
    "module.is_base",
    "module.type_hash",
    "module.role_hash",
    "capability.aggregate_mass_norm",
    *(f"capability.inertia.{index}" for index in range(6)),
    "capability.rotor_count",
    "capability.port_count",
    "capability.thrust_min_sum",
    "capability.thrust_min_mean",
    "capability.thrust_min_min",
    "capability.thrust_min_max",
    "capability.thrust_max_sum",
    "capability.thrust_max_mean",
    "capability.thrust_max_min",
    "capability.thrust_max_max",
    "capability.thrust_to_weight_ratio",
    *(f"capability.dock_port_count.{index}" for index in range(3)),
    "capability.has_vectoring",
    "capability.has_dock_mechanism",
    "ports.count",
    "ports.occupied_count",
    "ports.free_count",
    *(f"ports.type_count.{name}" for name in _PORT_TYPES),
    *(f"ports.occupied_type_count.{name}" for name in _PORT_TYPES),
    *(f"ports.compatibility_count.{index}" for index in range(3)),
    "anchors.count",
    "control_groups.membership_count",
    "control_groups.whole_body_membership_count",
    "runtime.present",
    *(f"runtime.pose.{name}" for name in _POSE_FEATURE_SUFFIXES),
    *(f"runtime.twist.{name}" for name in _TWIST_FEATURE_SUFFIXES),
    "runtime.health",
    *(f"runtime.joint_position.{name}" for name in _SUMMARY_SUFFIXES),
    *(f"runtime.joint_velocity.{name}" for name in _SUMMARY_SUFFIXES),
)

MORPHOLOGY_EDGE_FEATURE_NAMES: tuple[str, ...] = (
    *(f"relative_pose.{name}" for name in _POSE_FEATURE_SUFFIXES),
    *(f"source_port.pose.{name}" for name in _POSE_FEATURE_SUFFIXES),
    *(f"destination_port.pose.{name}" for name in _POSE_FEATURE_SUFFIXES),
    *(f"source_port.type.{name}" for name in _PORT_TYPES),
    *(f"destination_port.type.{name}" for name in _PORT_TYPES),
    *(f"source_port.compatibility.{index}" for index in range(3)),
    *(f"destination_port.compatibility.{index}" for index in range(3)),
    "source_port.occupied",
    "destination_port.occupied",
    *(f"edge.role.{name}" for name in _EDGE_ROLES),
    *(f"edge.latch_state.{name}" for name in _LATCH_STATES),
    *(f"edge.signed_log_stiffness.{index}" for index in range(6)),
)


@dataclass(frozen=True)
class MorphologyGraphBatch:
    """Padded homogeneous module graph tensors.

    ``edge_index[:, 0]`` contains message sources and ``edge_index[:, 1]``
    contains destinations. Every schema ``DockEdge`` is represented twice so
    that messages flow in both directions. Schema IDs are mapping metadata only;
    they are deliberately absent from numeric node/edge features.
    """

    node_features: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    module_ids: torch.Tensor
    edge_ids: torch.Tensor
    graph_ids: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def to(self, *, device: torch.device | str, dtype: torch.dtype | None = None) -> "MorphologyGraphBatch":
        floating_dtype = dtype or self.node_features.dtype
        return MorphologyGraphBatch(
            node_features=self.node_features.to(device=device, dtype=floating_dtype),
            node_mask=self.node_mask.to(device=device),
            edge_index=self.edge_index.to(device=device),
            edge_features=self.edge_features.to(device=device, dtype=floating_dtype),
            edge_mask=self.edge_mask.to(device=device),
            module_ids=self.module_ids.to(device=device),
            edge_ids=self.edge_ids.to(device=device),
            graph_ids=self.graph_ids,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class MorphologyGraphEncoderOutput:
    """Module tokens plus a permutation-invariant morphology embedding."""

    tokens: torch.Tensor
    mask: torch.Tensor
    token_type_ids: torch.Tensor
    source_type_ids: torch.Tensor
    source_ids: torch.Tensor
    graph_embeddings: torch.Tensor
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def node_embeddings(self) -> torch.Tensor:
        return self.tokens

    @property
    def global_embedding(self) -> torch.Tensor:
        return self.graph_embeddings

    @property
    def module_ids(self) -> torch.Tensor:
        return self.source_ids

    @property
    def group_slice(self) -> slice:
        return slice(0, int(self.tokens.shape[1]))

    @property
    def group_mask(self) -> torch.Tensor:
        return self.mask


class MorphologyGraphTensorizer:
    """Convert MorphologyGraph objects into one padded module graph batch."""

    def __init__(self, *, min_modules: int = 2, max_modules: int = 8) -> None:
        if not 2 <= min_modules <= max_modules <= 8:
            raise ValueError(
                "MorphologyGraphTensorizer requires 2 <= min_modules <= max_modules <= 8"
            )
        self.min_modules = min_modules
        self.max_modules = max_modules

    def tensorize(
        self,
        morphologies: Sequence[MorphologyGraph],
        *,
        runtime_observations: Sequence[RuntimeObservation | None] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MorphologyGraphBatch:
        graphs = list(morphologies)
        if not graphs:
            raise SchemaValidationError("MorphologyGraphTensorizer requires at least one graph")
        if not dtype.is_floating_point:
            raise SchemaValidationError("MorphologyGraphTensorizer dtype must be floating point")
        runtimes = (
            [None] * len(graphs)
            if runtime_observations is None
            else list(runtime_observations)
        )
        if len(runtimes) != len(graphs):
            raise SchemaValidationError(
                "MorphologyGraphTensorizer runtime_observations must match the graph batch size"
            )

        records = [self._graph_records(graph, runtime) for graph, runtime in zip(graphs, runtimes)]
        max_nodes = max(len(record.nodes) for record in records)
        max_edges = max(len(record.directed_edges) for record in records)
        batch_size = len(records)
        target_device = torch.device(device or "cpu")

        node_features = torch.zeros(
            (batch_size, max_nodes, len(MORPHOLOGY_NODE_FEATURE_NAMES)),
            dtype=dtype,
            device=target_device,
        )
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=target_device)
        module_ids = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=target_device)
        edge_index = torch.full(
            (batch_size, 2, max_edges),
            -1,
            dtype=torch.long,
            device=target_device,
        )
        edge_features = torch.zeros(
            (batch_size, max_edges, len(MORPHOLOGY_EDGE_FEATURE_NAMES)),
            dtype=dtype,
            device=target_device,
        )
        edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=target_device)
        edge_ids = torch.full((batch_size, max_edges), -1, dtype=torch.long, device=target_device)

        for batch_index, record in enumerate(records):
            node_count = len(record.nodes)
            node_features[batch_index, :node_count] = torch.tensor(
                record.node_features,
                dtype=dtype,
                device=target_device,
            )
            node_mask[batch_index, :node_count] = True
            module_ids[batch_index, :node_count] = torch.tensor(
                [node.module_id for node in record.nodes],
                dtype=torch.long,
                device=target_device,
            )
            for edge_offset, directed in enumerate(record.directed_edges):
                edge_index[batch_index, 0, edge_offset] = directed.source_index
                edge_index[batch_index, 1, edge_offset] = directed.destination_index
                edge_features[batch_index, edge_offset] = torch.tensor(
                    directed.features,
                    dtype=dtype,
                    device=target_device,
                )
                edge_mask[batch_index, edge_offset] = True
                edge_ids[batch_index, edge_offset] = directed.edge_id

        return MorphologyGraphBatch(
            node_features=node_features,
            node_mask=node_mask,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_mask=edge_mask,
            module_ids=module_ids,
            edge_ids=edge_ids,
            graph_ids=tuple(graph.graph_id for graph in graphs),
            metadata={
                "tensorizer": type(self).__name__,
                "tensorizer_version": MORPHOLOGY_GRAPH_TENSORIZER_VERSION,
                "min_modules": self.min_modules,
                "max_modules": self.max_modules,
                "node_feature_names": MORPHOLOGY_NODE_FEATURE_NAMES,
                "edge_feature_names": MORPHOLOGY_EDGE_FEATURE_NAMES,
                "dock_edge_representation": "two_directed_messages",
                "port_representation": "edge_and_module_aggregate_features",
                "schema_ids_are_numeric_features": False,
                "privileged_contact_wrench_features": False,
            },
        )

    def _graph_records(
        self,
        graph: MorphologyGraph,
        runtime: RuntimeObservation | None,
    ) -> "_GraphRecords":
        graph.validate()
        nodes = sorted(graph.modules, key=lambda module: module.module_id)
        if not self.min_modules <= len(nodes) <= self.max_modules:
            raise SchemaValidationError(
                "MorphologyGraphEncoder module count must be within the configured range "
                f"[{self.min_modules}, {self.max_modules}]"
            )
        module_ids = [module.module_id for module in nodes]
        if len(module_ids) != len(set(module_ids)):
            raise SchemaValidationError("MorphologyGraphEncoder requires unique module IDs")
        module_id_set = set(module_ids)
        module_index = {module_id: index for index, module_id in enumerate(module_ids)}

        ports_by_id: dict[int, PortNode] = {}
        ports_by_module: dict[int, list[PortNode]] = {module_id: [] for module_id in module_ids}
        for port in graph.ports:
            if port.port_global_id in ports_by_id:
                raise SchemaValidationError("MorphologyGraphEncoder requires unique port IDs")
            if port.module_id not in module_id_set:
                raise SchemaValidationError(
                    f"MorphologyGraphEncoder port {port.port_global_id} references an unknown module"
                )
            ports_by_id[port.port_global_id] = port
            ports_by_module[port.module_id].append(port)

        edge_ids = [edge.edge_id for edge in graph.dock_edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise SchemaValidationError("MorphologyGraphEncoder requires unique DockEdge IDs")
        used_port_ids: set[int] = set()
        for edge in graph.dock_edges:
            self._validate_edge(edge, module_id_set, ports_by_id, used_port_ids)

        anchors_by_module = {module_id: 0 for module_id in module_ids}
        for anchor in graph.robot_anchors:
            if anchor.module_id not in module_id_set:
                raise SchemaValidationError(
                    f"MorphologyGraphEncoder anchor {anchor.anchor_id} references an unknown module"
                )
            anchors_by_module[anchor.module_id] += 1
        group_memberships = {module_id: 0 for module_id in module_ids}
        whole_body_memberships = {module_id: 0 for module_id in module_ids}
        for group in graph.control_groups:
            for module_id in group.module_ids:
                if module_id not in module_id_set:
                    raise SchemaValidationError(
                        f"MorphologyGraphEncoder control group {group.group_id!r} references an unknown module"
                    )
                group_memberships[module_id] += 1
                whole_body_memberships[module_id] += int(group.role == "whole_body")

        runtime_by_module = self._runtime_states(graph, module_id_set, runtime)
        node_features = [
            _module_features(
                module,
                ports=ports_by_module[module.module_id],
                anchor_count=anchors_by_module[module.module_id],
                group_membership_count=group_memberships[module.module_id],
                whole_body_membership_count=whole_body_memberships[module.module_id],
                runtime_state=runtime_by_module.get(module.module_id),
            )
            for module in nodes
        ]
        directed_edges: list[_DirectedEdgeRecord] = []
        for edge in sorted(graph.dock_edges, key=lambda item: item.edge_id):
            src_port = ports_by_id[edge.src_port_id]
            dst_port = ports_by_id[edge.dst_port_id]
            directed_edges.append(
                _DirectedEdgeRecord(
                    edge_id=edge.edge_id,
                    source_index=module_index[edge.src_module_id],
                    destination_index=module_index[edge.dst_module_id],
                    features=_edge_features(
                        edge,
                        source_port=src_port,
                        destination_port=dst_port,
                        relative_pose=edge.relative_pose_src_to_dst,
                    ),
                )
            )
            directed_edges.append(
                _DirectedEdgeRecord(
                    edge_id=edge.edge_id,
                    source_index=module_index[edge.dst_module_id],
                    destination_index=module_index[edge.src_module_id],
                    features=_edge_features(
                        edge,
                        source_port=dst_port,
                        destination_port=src_port,
                        relative_pose=inverse_pose(edge.relative_pose_src_to_dst),
                    ),
                )
            )
        return _GraphRecords(
            nodes=nodes,
            node_features=node_features,
            directed_edges=directed_edges,
        )

    @staticmethod
    def _runtime_states(
        graph: MorphologyGraph,
        module_ids: set[int],
        runtime: RuntimeObservation | None,
    ) -> dict[int, ModuleRuntimeState]:
        if runtime is None:
            return {}
        if runtime.morphology_graph.graph_id != graph.graph_id:
            raise SchemaValidationError(
                "MorphologyGraphEncoder runtime observation must reference the encoded graph"
            )
        state_ids = [state.module_id for state in runtime.module_states]
        if len(state_ids) != len(set(state_ids)):
            raise SchemaValidationError(
                "MorphologyGraphEncoder runtime observation has duplicate module states"
            )
        if set(state_ids) != module_ids:
            missing = sorted(module_ids - set(state_ids))
            unknown = sorted(set(state_ids) - module_ids)
            raise SchemaValidationError(
                "MorphologyGraphEncoder runtime module IDs must exactly match morphology modules; "
                f"missing={missing}, unknown={unknown}"
            )
        return {state.module_id: state for state in runtime.module_states}

    @staticmethod
    def _validate_edge(
        edge: DockEdge,
        module_ids: set[int],
        ports_by_id: dict[int, PortNode],
        used_port_ids: set[int],
    ) -> None:
        if edge.src_module_id not in module_ids or edge.dst_module_id not in module_ids:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} references an unknown module"
            )
        if edge.src_module_id == edge.dst_module_id:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} must not be a self edge"
            )
        src_port = ports_by_id.get(edge.src_port_id)
        dst_port = ports_by_id.get(edge.dst_port_id)
        if src_port is None or dst_port is None:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} references an unknown port"
            )
        if src_port.module_id != edge.src_module_id or dst_port.module_id != edge.dst_module_id:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} port/module endpoints do not match"
            )
        if not src_port.occupied or not dst_port.occupied:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} references a port not marked occupied"
            )
        for port_id in (edge.src_port_id, edge.dst_port_id):
            if port_id in used_port_ids:
                raise SchemaValidationError(
                    f"MorphologyGraphEncoder dock port {port_id} is reused by multiple edges"
                )
            used_port_ids.add(port_id)
        if edge.edge_role not in _EDGE_ROLES:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} has an unsupported role"
            )
        if edge.latch_state not in _LATCH_STATES:
            raise SchemaValidationError(
                f"MorphologyGraphEncoder edge {edge.edge_id} has an unsupported latch state"
            )


class _EdgeMessageLayer(nn.Module):
    def __init__(self, d_model: int, *, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Sigmoid())
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_embeddings: torch.Tensor,
        batch: MorphologyGraphBatch,
    ) -> torch.Tensor:
        batch_size, node_width, hidden_width = nodes.shape
        edge_width = edge_embeddings.shape[1]
        safe_sources = batch.edge_index[:, 0].clamp(min=0)
        safe_destinations = batch.edge_index[:, 1].clamp(min=0)
        gather_width = hidden_width
        source_nodes = torch.gather(
            nodes,
            1,
            safe_sources.unsqueeze(-1).expand(-1, -1, gather_width),
        )
        destination_nodes = torch.gather(
            nodes,
            1,
            safe_destinations.unsqueeze(-1).expand(-1, -1, gather_width),
        )
        message_input = torch.cat(
            (source_nodes, destination_nodes, edge_embeddings), dim=-1
        )
        edge_weights = batch.edge_mask.unsqueeze(-1).to(nodes.dtype)
        messages = self.message(message_input) * self.gate(message_input) * edge_weights

        batch_offsets = (
            torch.arange(batch_size, device=nodes.device, dtype=torch.long)
            .reshape(-1, 1)
            .expand(-1, edge_width)
            * node_width
        )
        flat_destinations = (safe_destinations + batch_offsets).reshape(-1)
        aggregate_flat = torch.zeros(
            (batch_size * node_width, hidden_width),
            dtype=nodes.dtype,
            device=nodes.device,
        )
        aggregate_flat.index_add_(0, flat_destinations, messages.reshape(-1, hidden_width))
        degree_flat = torch.zeros(
            (batch_size * node_width, 1),
            dtype=nodes.dtype,
            device=nodes.device,
        )
        degree_flat.index_add_(0, flat_destinations, edge_weights.reshape(-1, 1))
        aggregate = aggregate_flat.reshape(batch_size, node_width, hidden_width)
        degree = degree_flat.reshape(batch_size, node_width, 1)
        aggregate = aggregate / torch.sqrt(torch.clamp(degree, min=1.0))
        delta = self.update(torch.cat((nodes, aggregate), dim=-1))
        next_nodes = self.norm(nodes + self.dropout(delta))
        return next_nodes * batch.node_mask.unsqueeze(-1).to(nodes.dtype)


class MorphologyGraphEncoder(nn.Module):
    """Pure-Torch edge-aware encoder for a homogeneous module graph."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        message_passing_steps: int = 2,
        dropout: float = 0.0,
        tensorizer: MorphologyGraphTensorizer | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("MorphologyGraphEncoder.d_model must be positive")
        if message_passing_steps <= 0:
            raise ValueError(
                "MorphologyGraphEncoder.message_passing_steps must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("MorphologyGraphEncoder.dropout must be in [0, 1)")
        self.d_model = d_model
        self.message_passing_steps = message_passing_steps
        self.tensorizer = tensorizer or MorphologyGraphTensorizer()
        self.node_projection = nn.Sequential(
            nn.Linear(len(MORPHOLOGY_NODE_FEATURE_NAMES), d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(len(MORPHOLOGY_EDGE_FEATURE_NAMES), d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            _EdgeMessageLayer(d_model, dropout=dropout)
            for _ in range(message_passing_steps)
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        morphologies: Sequence[MorphologyGraph] | MorphologyGraphBatch,
        *,
        runtime_observations: Sequence[RuntimeObservation | None] | None = None,
    ) -> MorphologyGraphEncoderOutput:
        device = self.node_projection[0].weight.device
        dtype = self.node_projection[0].weight.dtype
        if isinstance(morphologies, MorphologyGraphBatch):
            if runtime_observations is not None:
                raise ValueError(
                    "runtime_observations cannot be supplied with a pre-tensorized morphology batch"
                )
            batch = morphologies.to(device=device, dtype=dtype)
        else:
            batch = self.tensorizer.tensorize(
                morphologies,
                runtime_observations=runtime_observations,
                device=device,
                dtype=dtype,
            )
        _validate_batch_shapes(batch)

        node_mask_float = batch.node_mask.unsqueeze(-1).to(dtype)
        nodes = self.node_projection(batch.node_features) * node_mask_float
        edge_embeddings = self.edge_projection(batch.edge_features)
        edge_embeddings = edge_embeddings * batch.edge_mask.unsqueeze(-1).to(dtype)
        for layer in self.layers:
            nodes = layer(nodes, edge_embeddings, batch)

        denominator = torch.clamp(batch.node_mask.sum(dim=1, keepdim=True), min=1).to(dtype)
        mean_pool = (nodes * node_mask_float).sum(dim=1) / denominator
        masked_for_max = nodes.masked_fill(~batch.node_mask.unsqueeze(-1), -torch.inf)
        max_pool = masked_for_max.max(dim=1).values
        graph_embeddings = self.graph_projection(torch.cat((mean_pool, max_pool), dim=-1))
        token_type_ids = torch.where(
            batch.node_mask,
            torch.full_like(batch.module_ids, MORPHOLOGY_MODULE_TOKEN_TYPE_ID),
            torch.zeros_like(batch.module_ids),
        )
        source_type_ids = torch.where(
            batch.node_mask,
            torch.full_like(batch.module_ids, SOURCE_TYPE_MORPHOLOGY_MODULE),
            torch.zeros_like(batch.module_ids),
        )
        return MorphologyGraphEncoderOutput(
            tokens=nodes,
            mask=batch.node_mask,
            token_type_ids=token_type_ids,
            source_type_ids=source_type_ids,
            source_ids=batch.module_ids,
            graph_embeddings=graph_embeddings,
            metadata={
                **batch.metadata,
                "encoder": type(self).__name__,
                "encoder_version": MORPHOLOGY_GRAPH_ENCODER_VERSION,
                "architecture": MORPHOLOGY_GRAPH_ENCODER_VERSION,
                "d_model": self.d_model,
                "message_passing_steps": self.message_passing_steps,
                "global_pooling": "masked_mean_max",
            },
        )


@dataclass(frozen=True)
class _DirectedEdgeRecord:
    edge_id: int
    source_index: int
    destination_index: int
    features: list[float]


@dataclass(frozen=True)
class _GraphRecords:
    nodes: list[ModuleNode]
    node_features: list[list[float]]
    directed_edges: list[_DirectedEdgeRecord]


def _module_features(
    module: ModuleNode,
    *,
    ports: list[PortNode],
    anchor_count: int,
    group_membership_count: int,
    whole_body_membership_count: int,
    runtime_state: ModuleRuntimeState | None,
) -> list[float]:
    capability = module.capability_token
    type_counts = [0.0] * len(_PORT_TYPES)
    occupied_type_counts = [0.0] * len(_PORT_TYPES)
    compatibility_counts = [0.0] * 3
    for port in ports:
        type_index = _port_type_index(port.port_type)
        type_counts[type_index] += 1.0
        if port.occupied:
            occupied_type_counts[type_index] += 1.0
        for index, value in enumerate(_fixed_width(port.compatible_port_type_mask, 3)):
            compatibility_counts[index] += value
    occupied_count = sum(1 for port in ports if port.occupied)
    runtime_features = _runtime_features(runtime_state)
    values = [
        *_pose_features(module.pose_in_design_frame),
        float(module.health),
        1.0 if module.is_base else 0.0,
        _stable_scalar(module.module_type),
        _stable_scalar(module.role_id),
        float(capability.aggregate_mass_norm),
        *_fixed_width(capability.aggregate_inertia_features, 6),
        float(capability.rotor_count),
        float(capability.port_count),
        *_sum_mean_min_max(capability.thrust_min_features),
        *_sum_mean_min_max(capability.thrust_max_features),
        float(capability.thrust_to_weight_ratio_est),
        *_fixed_width(capability.dock_port_type_counts, 3),
        1.0 if capability.has_vectoring else 0.0,
        1.0 if capability.has_dock_mechanism else 0.0,
        float(len(ports)),
        float(occupied_count),
        float(len(ports) - occupied_count),
        *type_counts,
        *occupied_type_counts,
        *compatibility_counts,
        float(anchor_count),
        float(group_membership_count),
        float(whole_body_membership_count),
        *runtime_features,
    ]
    return _checked_features(values, MORPHOLOGY_NODE_FEATURE_NAMES, "module node")


def _runtime_features(state: ModuleRuntimeState | None) -> list[float]:
    if state is None:
        return [0.0] * (1 + 7 + 6 + 1 + 5 + 5)
    return [
        1.0,
        *_pose_features(state.pose_world),
        *[float(value) for value in state.twist_world],
        float(state.health),
        *_mapping_summary(state.joint_positions),
        *_mapping_summary(state.joint_velocities),
    ]


def _edge_features(
    edge: DockEdge,
    *,
    source_port: PortNode,
    destination_port: PortNode,
    relative_pose: Pose7D,
) -> list[float]:
    values = [
        *_pose_features(relative_pose),
        *_pose_features(source_port.local_pose),
        *_pose_features(destination_port.local_pose),
        *_one_hot(_port_type_index(source_port.port_type), len(_PORT_TYPES)),
        *_one_hot(_port_type_index(destination_port.port_type), len(_PORT_TYPES)),
        *_fixed_width(source_port.compatible_port_type_mask, 3),
        *_fixed_width(destination_port.compatible_port_type_mask, 3),
        1.0 if source_port.occupied else 0.0,
        1.0 if destination_port.occupied else 0.0,
        *_one_hot(_EDGE_ROLES.index(edge.edge_role), len(_EDGE_ROLES)),
        *_one_hot(_LATCH_STATES.index(edge.latch_state), len(_LATCH_STATES)),
        *[_signed_log1p(float(value)) for value in edge.estimated_stiffness],
    ]
    return _checked_features(values, MORPHOLOGY_EDGE_FEATURE_NAMES, "dock edge")


def _pose_features(pose: Pose7D) -> list[float]:
    if len(pose) != 7:
        raise SchemaValidationError("MorphologyGraphEncoder pose features require length seven")
    quaternion = normalize_quat(
        (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
    )
    return [float(pose[0]), float(pose[1]), float(pose[2]), *quaternion]


def _mapping_summary(values: dict[str, float]) -> list[float]:
    ordered = [float(values[key]) for key in sorted(values)]
    if not ordered:
        return [0.0] * 5
    mean = sum(ordered) / len(ordered)
    rms = math.sqrt(sum(value * value for value in ordered) / len(ordered))
    return [float(len(ordered)), mean, rms, min(ordered), max(ordered)]


def _sum_mean_min_max(values: Sequence[float]) -> list[float]:
    output = [float(value) for value in values]
    if not output:
        return [0.0] * 4
    return [sum(output), sum(output) / len(output), min(output), max(output)]


def _fixed_width(values: Sequence[float | int], width: int) -> list[float]:
    output = [float(value) for value in values[:width]]
    return output + [0.0] * (width - len(output))


def _port_type_index(port_type: str) -> int:
    try:
        return _PORT_TYPES.index(port_type)
    except ValueError:
        return len(_PORT_TYPES) - 1


def _one_hot(index: int, width: int) -> list[float]:
    return [1.0 if item == index else 0.0 for item in range(width)]


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _stable_scalar(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def _checked_features(values: list[float], names: tuple[str, ...], label: str) -> list[float]:
    if len(values) != len(names):
        raise RuntimeError(
            f"MorphologyGraphEncoder internal {label} feature layout mismatch: "
            f"{len(values)} != {len(names)}"
        )
    if not all(math.isfinite(float(value)) for value in values):
        raise SchemaValidationError(
            f"MorphologyGraphEncoder {label} features must contain only finite values"
        )
    return [float(value) for value in values]


def _validate_batch_shapes(batch: MorphologyGraphBatch) -> None:
    if batch.node_features.ndim != 3:
        raise ValueError("MorphologyGraphBatch.node_features must have shape [B, N, F]")
    batch_size, node_width, node_feature_width = batch.node_features.shape
    if node_feature_width != len(MORPHOLOGY_NODE_FEATURE_NAMES):
        raise ValueError("MorphologyGraphBatch node feature width mismatch")
    if tuple(batch.node_mask.shape) != (batch_size, node_width):
        raise ValueError("MorphologyGraphBatch.node_mask shape mismatch")
    if tuple(batch.module_ids.shape) != (batch_size, node_width):
        raise ValueError("MorphologyGraphBatch.module_ids shape mismatch")
    if batch.edge_features.ndim != 3:
        raise ValueError("MorphologyGraphBatch.edge_features must have shape [B, E, F]")
    edge_width = int(batch.edge_features.shape[1])
    if int(batch.edge_features.shape[0]) != batch_size or int(batch.edge_features.shape[2]) != len(
        MORPHOLOGY_EDGE_FEATURE_NAMES
    ):
        raise ValueError("MorphologyGraphBatch edge feature shape mismatch")
    if tuple(batch.edge_index.shape) != (batch_size, 2, edge_width):
        raise ValueError("MorphologyGraphBatch.edge_index shape mismatch")
    if tuple(batch.edge_mask.shape) != (batch_size, edge_width):
        raise ValueError("MorphologyGraphBatch.edge_mask shape mismatch")
    if tuple(batch.edge_ids.shape) != (batch_size, edge_width):
        raise ValueError("MorphologyGraphBatch.edge_ids shape mismatch")
    if not bool(batch.node_mask.any(dim=1).all().item()):
        raise ValueError("MorphologyGraphBatch requires at least one valid node per graph")
    if bool((batch.module_ids[~batch.node_mask] != -1).any().item()):
        raise ValueError("MorphologyGraphBatch padded module IDs must be -1")
    if bool((batch.edge_ids[~batch.edge_mask] != -1).any().item()):
        raise ValueError("MorphologyGraphBatch padded edge IDs must be -1")
    if bool(batch.edge_mask.any().item()):
        valid_sources = batch.edge_index[:, 0][batch.edge_mask]
        valid_destinations = batch.edge_index[:, 1][batch.edge_mask]
        if bool(((valid_sources < 0) | (valid_sources >= node_width)).any().item()):
            raise ValueError("MorphologyGraphBatch contains an invalid edge source index")
        if bool(((valid_destinations < 0) | (valid_destinations >= node_width)).any().item()):
            raise ValueError("MorphologyGraphBatch contains an invalid edge destination index")


__all__ = [
    "MORPHOLOGY_EDGE_FEATURE_NAMES",
    "MORPHOLOGY_GRAPH_ENCODER_VERSION",
    "MORPHOLOGY_GRAPH_TENSORIZER_VERSION",
    "MORPHOLOGY_MODULE_TOKEN_TYPE_ID",
    "MORPHOLOGY_NODE_FEATURE_NAMES",
    "SOURCE_TYPE_MORPHOLOGY_MODULE",
    "MorphologyGraphBatch",
    "MorphologyGraphEncoder",
    "MorphologyGraphEncoderOutput",
    "MorphologyGraphTensorizer",
]
