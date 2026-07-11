from __future__ import annotations

import random
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Iterable

from amsrr.morphology.dock_geometry import (
    IDENTITY_POSE,
    modules_with_dock_aligned_poses,
    relative_pose_for_dock_ports,
)
from amsrr.robot_model.physical_model_builder import build_module_capability_token
from amsrr.schemas.common import SchemaValidationError, canonical_json
from amsrr.schemas.morphology import ControlGroup, DockEdge, ModuleNode, MorphologyGraph, PortNode
from amsrr.schemas.physical_model import DockPortSpec, PhysicalModel


PORT_TYPE_ORDER = ("pitch_dock", "yaw_dock", "generic_dock")
HOLON_DOCK_PAIR = frozenset(("pitch_dock", "yaw_dock"))
DEFAULT_DOCK_STIFFNESS = (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0)


@dataclass(frozen=True)
class RandomConnectedMorphologyConfig:
    """Configuration for the Order-1 connected-Holons distribution.

    Module count is sampled uniformly from the inclusive configured range.
    The Version-1 range follows the source-spec RobotConstraints upper bound
    of eight modules.
    """

    min_modules: int = 2
    max_modules: int = 8
    module_type: str = "holon"
    graph_id_prefix: str = "morphology:random-connected"
    dock_stiffness: tuple[float, float, float, float, float, float] = DEFAULT_DOCK_STIFFNESS

    def __post_init__(self) -> None:
        if not 2 <= self.min_modules <= self.max_modules <= 8:
            raise SchemaValidationError(
                "RandomConnectedMorphologyConfig requires 2 <= min_modules <= max_modules <= 8"
            )
        if not self.module_type:
            raise SchemaValidationError("RandomConnectedMorphologyConfig.module_type must be non-empty")
        if not self.graph_id_prefix:
            raise SchemaValidationError("RandomConnectedMorphologyConfig.graph_id_prefix must be non-empty")
        if len(self.dock_stiffness) != 6 or any(value < 0.0 for value in self.dock_stiffness):
            raise SchemaValidationError(
                "RandomConnectedMorphologyConfig.dock_stiffness must contain six non-negative values"
            )


class RandomConnectedMorphologyDistribution:
    """Seeded constructive distribution over connected tree morphologies.

    A new module is attached to one compatible, unused port on the current
    tree. This makes connectivity and acyclicity construction invariants. The
    distribution does not perform collision, hover-QP, floor, or simulator
    checks; callers can deterministically filter the returned candidates.
    """

    def __init__(
        self,
        physical_model: PhysicalModel,
        config: RandomConnectedMorphologyConfig | None = None,
    ) -> None:
        self._physical_model = physical_model
        self._config = config or RandomConnectedMorphologyConfig()
        self._dock_ports = tuple(sorted(physical_model.dock_ports, key=lambda port: port.port_id))
        self._validate_holon_ports()

    @property
    def config(self) -> RandomConnectedMorphologyConfig:
        return self._config

    def sample(self, *, seed: int, module_count: int | None = None) -> MorphologyGraph:
        """Return one deterministic candidate for ``seed``.

        Passing ``module_count`` fixes the size while retaining seeded topology
        and port-pair sampling. Otherwise size is uniform on the configured
        inclusive range.
        """

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise SchemaValidationError("random morphology seed must be a non-negative integer")
        if module_count is not None:
            self._validate_module_count(module_count)

        rng = random.Random(seed)
        selected_count = (
            module_count
            if module_count is not None
            else rng.randint(self._config.min_modules, self._config.max_modules)
        )
        capability = build_module_capability_token(
            self._physical_model,
            module_type=self._config.module_type,
        )
        modules = [
            ModuleNode(
                module_id=module_id,
                module_type=self._config.module_type,
                pose_in_design_frame=IDENTITY_POSE,
                role_id="base" if module_id == 0 else "member",
                is_base=module_id == 0,
                capability_token=capability,
            )
            for module_id in range(selected_count)
        ]
        ports = self._build_ports(selected_count)
        ports_by_module = _ports_by_module(ports)
        used_port_ids: set[int] = set()
        edges: list[DockEdge] = []

        for child_module_id in range(1, selected_count):
            candidates = _compatible_free_pairs(
                ports_by_module,
                existing_module_ids=range(child_module_id),
                child_module_id=child_module_id,
                used_port_ids=used_port_ids,
            )
            if not candidates:
                raise SchemaValidationError(
                    f"No compatible free pitch-yaw port pair can attach module {child_module_id}"
                )
            src_port, dst_port = rng.choice(candidates)
            used_port_ids.update((src_port.port_global_id, dst_port.port_global_id))
            edges.append(
                DockEdge(
                    edge_id=len(edges),
                    src_module_id=src_port.module_id,
                    src_port_id=src_port.port_global_id,
                    dst_module_id=dst_port.module_id,
                    dst_port_id=dst_port.port_global_id,
                    relative_pose_src_to_dst=relative_pose_for_dock_ports(src_port, dst_port),
                    edge_role="structural",
                    estimated_stiffness=list(self._config.dock_stiffness),
                    latch_state="planned",
                )
            )

        ports = [replace(port, occupied=port.port_global_id in used_port_ids) for port in ports]
        modules = modules_with_dock_aligned_poses(modules, edges, base_module_id=0)
        morphology = MorphologyGraph(
            graph_id="pending-structural-hash",
            modules=modules,
            ports=ports,
            dock_edges=edges,
            robot_anchors=[],
            control_groups=[
                ControlGroup(
                    group_id="all_modules",
                    module_ids=list(range(selected_count)),
                    role="whole_body",
                    metadata={"source": "random_connected_distribution"},
                )
            ],
            base_module_id=0,
            is_closed_loop=False,
        )
        structural_hash = morphology_structural_hash(morphology)
        return replace(
            morphology,
            graph_id=f"{self._config.graph_id_prefix}:{structural_hash}",
        )

    def samples(
        self,
        *,
        seeds: Iterable[int],
        module_count: int | None = None,
    ) -> list[MorphologyGraph]:
        """Return candidates in seed order without deduplicating them."""

        return [self.sample(seed=seed, module_count=module_count) for seed in seeds]

    def _validate_module_count(self, module_count: int) -> None:
        if not isinstance(module_count, int) or isinstance(module_count, bool):
            raise SchemaValidationError("random morphology module_count must be an integer")
        if not self._config.min_modules <= module_count <= self._config.max_modules:
            raise SchemaValidationError(
                "random morphology module_count must be within the configured inclusive range "
                f"[{self._config.min_modules}, {self._config.max_modules}]"
            )

    def _validate_holon_ports(self) -> None:
        port_ids = [port.port_id for port in self._dock_ports]
        if len(port_ids) != len(set(port_ids)):
            raise SchemaValidationError("PhysicalModel.dock_ports has duplicate port_id values")
        for src in self._dock_ports:
            for dst in self._dock_ports:
                if _dock_specs_form_holon_pair(src, dst):
                    return
        raise SchemaValidationError(
            "random connected Holon morphology requires a mutually compatible pitch-yaw dock pair"
        )

    def _build_ports(self, module_count: int) -> list[PortNode]:
        port_count = len(self._dock_ports)
        return [
            PortNode(
                port_global_id=module_id * port_count + local_index,
                module_id=module_id,
                port_local_id=dock_port.port_id,
                local_pose=dock_port.local_pose,
                port_type=dock_port.port_type,
                occupied=False,
                compatible_port_type_mask=_compatible_mask(dock_port.compatible_port_types),
            )
            for module_id in range(module_count)
            for local_index, dock_port in enumerate(self._dock_ports)
        ]


def morphology_structural_hash(morphology: MorphologyGraph) -> str:
    """Hash a rooted port-labelled tree independently of module IDs/list order.

    Runtime state (pose, health, latch state), graph ID, physical parameters,
    anchors, and control groups are intentionally excluded. The duplicate key
    represents topology, module/role labels, dock-port assignments, and edge
    roles only.
    """

    module_by_id = {module.module_id: module for module in morphology.modules}
    port_by_id = {port.port_global_id: port for port in morphology.ports}
    if len(module_by_id) != len(morphology.modules):
        raise SchemaValidationError("structural hash requires unique module ids")
    if len(port_by_id) != len(morphology.ports):
        raise SchemaValidationError("structural hash requires unique port ids")
    if morphology.base_module_id not in module_by_id:
        raise SchemaValidationError("structural hash requires a valid base module")
    if not module_by_id[morphology.base_module_id].is_base:
        raise SchemaValidationError("structural hash requires base_module_id to identify the base module")
    if morphology.is_closed_loop:
        raise SchemaValidationError("structural hash rejects closed-loop morphology")
    if len(morphology.dock_edges) != max(0, len(module_by_id) - 1):
        raise SchemaValidationError("structural hash requires a tree with module_count - 1 edges")
    local_port_keys = [(port.module_id, port.port_local_id) for port in morphology.ports]
    if len(local_port_keys) != len(set(local_port_keys)):
        raise SchemaValidationError("structural hash requires unique local port ids per module")

    adjacency: dict[int, list[tuple[int, PortNode, PortNode, str]]] = {
        module_id: [] for module_id in module_by_id
    }
    used_port_ids: set[int] = set()
    for edge in morphology.dock_edges:
        if edge.src_module_id not in module_by_id or edge.dst_module_id not in module_by_id:
            raise SchemaValidationError("structural hash edge references a missing module")
        src_port = _edge_port(port_by_id, edge.src_port_id, edge.src_module_id)
        dst_port = _edge_port(port_by_id, edge.dst_port_id, edge.dst_module_id)
        for port in (src_port, dst_port):
            if port.port_global_id in used_port_ids:
                raise SchemaValidationError("structural hash rejects dock port reuse")
            used_port_ids.add(port.port_global_id)
        if edge.src_module_id == edge.dst_module_id:
            raise SchemaValidationError("structural hash rejects self edges")
        adjacency[edge.src_module_id].append(
            (edge.dst_module_id, src_port, dst_port, edge.edge_role)
        )
        adjacency[edge.dst_module_id].append(
            (edge.src_module_id, dst_port, src_port, edge.edge_role)
        )

    canonical_ids = {morphology.base_module_id: 0}
    parents: dict[int, int | None] = {morphology.base_module_id: None}
    pending = [morphology.base_module_id]
    module_records: list[tuple] = []
    edge_records: list[tuple] = []
    while pending:
        module_id = pending.pop(0)
        module = module_by_id[module_id]
        module_records.append(
            (
                canonical_ids[module_id],
                module.module_type,
                module.role_id,
                module.is_base,
            )
        )
        neighbors = sorted(
            adjacency[module_id],
            key=lambda item: (
                item[1].port_local_id,
                item[2].port_local_id,
                item[1].port_type,
                item[2].port_type,
                item[3],
            ),
        )
        for neighbor_id, local_port, remote_port, edge_role in neighbors:
            if neighbor_id == parents[module_id]:
                continue
            if neighbor_id in canonical_ids:
                raise SchemaValidationError("structural hash rejects closed-loop morphology")
            canonical_ids[neighbor_id] = len(canonical_ids)
            parents[neighbor_id] = module_id
            pending.append(neighbor_id)
            edge_records.append(
                (
                    canonical_ids[module_id],
                    local_port.port_local_id,
                    local_port.port_type,
                    canonical_ids[neighbor_id],
                    remote_port.port_local_id,
                    remote_port.port_type,
                    edge_role,
                )
            )

    if set(canonical_ids) != set(module_by_id):
        missing = sorted(set(module_by_id) - set(canonical_ids))
        raise SchemaValidationError(f"structural hash requires a connected tree; missing modules {missing}")
    signature = {
        "modules": module_records,
        "edges": edge_records,
    }
    return sha256(canonical_json(signature).encode("utf-8")).hexdigest()


def _compatible_mask(compatible_port_types: list[str]) -> list[int]:
    return [1 if port_type in compatible_port_types else 0 for port_type in PORT_TYPE_ORDER]


def _ports_by_module(ports: list[PortNode]) -> dict[int, list[PortNode]]:
    grouped: dict[int, list[PortNode]] = {}
    for port in ports:
        grouped.setdefault(port.module_id, []).append(port)
    return grouped


def _compatible_free_pairs(
    ports_by_module: dict[int, list[PortNode]],
    *,
    existing_module_ids: Iterable[int],
    child_module_id: int,
    used_port_ids: set[int],
) -> list[tuple[PortNode, PortNode]]:
    pairs: list[tuple[PortNode, PortNode]] = []
    for parent_module_id in existing_module_ids:
        for src_port in ports_by_module[parent_module_id]:
            if src_port.port_global_id in used_port_ids:
                continue
            for dst_port in ports_by_module[child_module_id]:
                if dst_port.port_global_id in used_port_ids:
                    continue
                if _port_nodes_form_holon_pair(src_port, dst_port):
                    pairs.append((src_port, dst_port))
    return sorted(
        pairs,
        key=lambda pair: (
            pair[0].module_id,
            pair[0].port_local_id,
            pair[1].port_local_id,
        ),
    )


def _port_nodes_form_holon_pair(src: PortNode, dst: PortNode) -> bool:
    if frozenset((src.port_type, dst.port_type)) != HOLON_DOCK_PAIR:
        return False
    try:
        src_accepts_dst = bool(src.compatible_port_type_mask[PORT_TYPE_ORDER.index(dst.port_type)])
        dst_accepts_src = bool(dst.compatible_port_type_mask[PORT_TYPE_ORDER.index(src.port_type)])
    except (IndexError, ValueError):
        return False
    return src_accepts_dst and dst_accepts_src


def _dock_specs_form_holon_pair(src: DockPortSpec, dst: DockPortSpec) -> bool:
    return (
        frozenset((src.port_type, dst.port_type)) == HOLON_DOCK_PAIR
        and dst.port_type in src.compatible_port_types
        and src.port_type in dst.compatible_port_types
    )


def _edge_port(
    port_by_id: dict[int, PortNode],
    port_id: int,
    expected_module_id: int,
) -> PortNode:
    port = port_by_id.get(port_id)
    if port is None:
        raise SchemaValidationError(f"structural hash edge references missing port {port_id}")
    if port.module_id != expected_module_id:
        raise SchemaValidationError(
            f"structural hash edge port {port_id} does not belong to module {expected_module_id}"
        )
    return port
