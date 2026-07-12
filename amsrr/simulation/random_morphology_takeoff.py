from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    compose_pose,
    inverse_pose,
    matvec,
    transform_from_pose,
)
from amsrr.feasibility.morphology_flight import (
    collision_geometry_content_hash,
    morphology_collision_aabbs,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_in_root_frame
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, Vector3, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    POLICY_COMMAND_CONTRACT_LEGACY,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, IsaacLabBackendConfig
from amsrr.utils.hashing import stable_hash


RANDOM_MORPHOLOGY_TAKEOFF_VERSION = "random_morphology_takeoff_v1"
FLOOR_PLACEMENT_METHOD = "order1_morphology_collision_aabbs_v1"
ORDER2_FLOOR_SIZE_M: tuple[float, float, float] = (100.0, 100.0, 0.05)
ORDER2_FLOOR_POSE_WORLD: Pose7D = (0.0, 0.0, -0.025, 0.0, 0.0, 0.0, 1.0)
FIXED_DOCK_JOINT_POSITION_TOLERANCE_RAD = 0.0053
FIXED_DOCK_CONNECT_FRAME_POSITION_TOLERANCE_M = 1.0e-6
FIXED_DOCK_CONNECT_FRAME_ATTITUDE_TOLERANCE_RAD = 1.0e-6


class TakeoffPhase(str, Enum):
    SETTLE = "settle"
    TAKEOFF_RAMP = "takeoff_ramp"
    HOVER_HOLD = "hover_hold"
    COMPLETE = "complete"


@dataclass
class RandomMorphologyTakeoffConfig(SchemaBase):
    backend_config_path: str = "configs/env/isaac_lab.yaml"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    mesh_search_dirs: list[str] = field(default_factory=lambda: ["module_urdf"])
    simulation_dt_s: float = 0.005
    floor_clearance_m: float = 0.002
    floor_contact_force_threshold_n: float = 0.5
    floor_contact_dwell_duration_s: float = 0.10
    exact_cross_module_contact_force_threshold_n: float = 1.0e-3
    exact_cross_module_contact_max_patches_per_body_pair: int = 8
    dock_joint_position_tolerance_rad: float = FIXED_DOCK_JOINT_POSITION_TOLERANCE_RAD
    initial_root_position_tolerance_m: float = 0.002
    initial_root_attitude_tolerance_rad: float = 0.001
    settle_duration_s: float = 1.0
    settle_dwell_duration_s: float = 0.25
    takeoff_ramp_duration_s: float = 2.0
    hover_hold_duration_s: float = 1.0
    hover_acquisition_timeout_s: float = 2.0
    hover_height_delta_m: float = 0.5
    position_error_threshold_m: float = 0.20
    attitude_error_threshold_rad: float = 0.25
    settle_linear_speed_threshold_mps: float = 0.20
    settle_angular_speed_threshold_rad_s: float = 0.50
    hover_linear_speed_threshold_mps: float = 0.15
    hover_angular_speed_threshold_rad_s: float = 0.25
    max_vertical_speed_mps: float = 3.0
    min_height_gain_ratio: float = 0.80
    allocation_mode: str = "rigid_body_qp"
    stop_on_hover_hold: bool = True
    command_timeout_s: float = 300.0
    control_contract_version: str = POLICY_COMMAND_CONTRACT_LEGACY

    def validate(self) -> None:
        for name in (
            "backend_config_path",
            "robot_model_config_path",
            "allocation_mode",
        ):
            require_non_empty(getattr(self, name), f"RandomMorphologyTakeoffConfig.{name}")
        for name in (
            "simulation_dt_s",
            "floor_clearance_m",
            "floor_contact_force_threshold_n",
            "floor_contact_dwell_duration_s",
            "exact_cross_module_contact_force_threshold_n",
            "dock_joint_position_tolerance_rad",
            "initial_root_position_tolerance_m",
            "initial_root_attitude_tolerance_rad",
            "settle_duration_s",
            "settle_dwell_duration_s",
            "takeoff_ramp_duration_s",
            "hover_hold_duration_s",
            "hover_acquisition_timeout_s",
            "hover_height_delta_m",
            "position_error_threshold_m",
            "attitude_error_threshold_rad",
            "settle_linear_speed_threshold_mps",
            "settle_angular_speed_threshold_rad_s",
            "hover_linear_speed_threshold_mps",
            "hover_angular_speed_threshold_rad_s",
            "max_vertical_speed_mps",
            "command_timeout_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise SchemaValidationError(f"RandomMorphologyTakeoffConfig.{name} must be positive")
        if not 0.0 < self.min_height_gain_ratio <= 1.0:
            raise SchemaValidationError(
                "RandomMorphologyTakeoffConfig.min_height_gain_ratio must be in (0, 1]"
            )
        if self.exact_cross_module_contact_max_patches_per_body_pair <= 0:
            raise SchemaValidationError(
                "RandomMorphologyTakeoffConfig."
                "exact_cross_module_contact_max_patches_per_body_pair must be positive"
            )
        if self.settle_dwell_duration_s > self.settle_duration_s:
            raise SchemaValidationError(
                "RandomMorphologyTakeoffConfig.settle_dwell_duration_s must not exceed settle_duration_s"
            )
        if self.floor_contact_dwell_duration_s > self.settle_duration_s:
            raise SchemaValidationError(
                "RandomMorphologyTakeoffConfig.floor_contact_dwell_duration_s must not exceed settle_duration_s"
            )
        if self.allocation_mode != "rigid_body_qp":
            raise SchemaValidationError(
                "random morphology takeoff acceptance requires allocation_mode='rigid_body_qp'"
            )
        if self.control_contract_version not in {
            POLICY_COMMAND_CONTRACT_LEGACY,
            POLICY_COMMAND_CONTRACT_CENTROIDAL,
        }:
            raise SchemaValidationError(
                "RandomMorphologyTakeoffConfig.control_contract_version is unsupported"
            )

    @property
    def total_duration_s(self) -> float:
        return (
            self.settle_duration_s
            + self.takeoff_ramp_duration_s
            + self.hover_hold_duration_s
            + self.hover_acquisition_timeout_s
        )

    @property
    def required_steps(self) -> int:
        return max(1, int(math.ceil(self.total_duration_s / self.simulation_dt_s)) + 1)


@dataclass(frozen=True)
class MorphologyCollisionBounds:
    minimum: Vector3
    maximum: Vector3
    collision_geometry_count: int
    mesh_geometry_count: int
    primitive_geometry_count: int
    method: str = FLOOR_PLACEMENT_METHOD


@dataclass(frozen=True)
class FloorContactPlacement:
    root_pose_world: Pose7D
    collision_bounds_root: MorphologyCollisionBounds
    floor_z_m: float
    clearance_m: float
    initial_lowest_collision_z_world: float

    @property
    def floor_gap_m(self) -> float:
        return self.initial_lowest_collision_z_world - self.floor_z_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_pose_world": list(self.root_pose_world),
            "collision_bounds_root": {
                "minimum": list(self.collision_bounds_root.minimum),
                "maximum": list(self.collision_bounds_root.maximum),
                "collision_geometry_count": self.collision_bounds_root.collision_geometry_count,
                "mesh_geometry_count": self.collision_bounds_root.mesh_geometry_count,
                "primitive_geometry_count": self.collision_bounds_root.primitive_geometry_count,
                "method": self.collision_bounds_root.method,
            },
            "floor_z_m": self.floor_z_m,
            "clearance_m": self.clearance_m,
            "initial_lowest_collision_z_world": self.initial_lowest_collision_z_world,
            "floor_gap_m": self.floor_gap_m,
        }


@dataclass(frozen=True)
class FixedDockConnectFrameAlignment:
    edge_count: int
    max_position_error_m: float
    max_attitude_error_rad: float


def fixed_dock_connect_frame_alignment(
    morphology_graph: MorphologyGraph,
    physical_model: PhysicalModel,
) -> FixedDockConnectFrameAlignment:
    """Check graph module poses against the current PhysicalModel dock frames."""

    validate_takeoff_morphology(morphology_graph)
    modules = {
        module.module_id: tuple(module.pose_in_design_frame)
        for module in morphology_graph.modules
    }
    graph_ports = {port.port_global_id: port for port in morphology_graph.ports}
    physical_ports = {port.port_id: port for port in physical_model.dock_ports}
    max_position_error = 0.0
    max_attitude_error = 0.0
    for edge in morphology_graph.dock_edges:
        src = graph_ports.get(edge.src_port_id)
        dst = graph_ports.get(edge.dst_port_id)
        if src is None or dst is None:
            raise SchemaValidationError(
                f"dock edge {edge.edge_id} references a missing graph port"
            )
        src_spec = physical_ports.get(src.port_local_id)
        dst_spec = physical_ports.get(dst.port_local_id)
        if src_spec is None or dst_spec is None:
            raise SchemaValidationError(
                f"dock edge {edge.edge_id} references a missing PhysicalModel port"
            )
        expected_dst = compose_pose(
            compose_pose(modules[edge.src_module_id], tuple(src_spec.local_pose)),
            FACE_TO_FACE_DOCK_RELATION,
        )
        actual_dst = compose_pose(
            modules[edge.dst_module_id],
            tuple(dst_spec.local_pose),
        )
        error_pose = compose_pose(inverse_pose(expected_dst), actual_dst)
        position_error = math.sqrt(
            sum(float(value) ** 2 for value in error_pose[:3])
        )
        orientation_w = min(1.0, abs(float(error_pose[6])))
        attitude_error = 2.0 * math.acos(orientation_w)
        max_position_error = max(max_position_error, position_error)
        max_attitude_error = max(max_attitude_error, attitude_error)
    return FixedDockConnectFrameAlignment(
        edge_count=len(morphology_graph.dock_edges),
        max_position_error_m=max_position_error,
        max_attitude_error_rad=max_attitude_error,
    )


@dataclass(frozen=True)
class TakeoffTarget:
    phase: TakeoffPhase
    phase_elapsed_s: float
    ramp_progress: float
    desired_pose_world: Pose7D | None
    thrust_enabled: bool


class DeterministicTakeoffScheduler:
    """Pure deterministic settle -> takeoff ramp -> hover-hold scheduler."""

    def __init__(self, config: RandomMorphologyTakeoffConfig | None = None) -> None:
        self.config = config or RandomMorphologyTakeoffConfig()

    def target_at(self, elapsed_s: float, *, settled_pose_world: Pose7D) -> TakeoffTarget:
        elapsed = max(0.0, float(elapsed_s))
        if elapsed < self.config.settle_duration_s:
            return TakeoffTarget(TakeoffPhase.SETTLE, elapsed, 0.0, None, False)

        ramp_elapsed = elapsed - self.config.settle_duration_s
        final_pose = self.final_hover_pose(settled_pose_world)
        if ramp_elapsed < self.config.takeoff_ramp_duration_s:
            progress = min(1.0, ramp_elapsed / self.config.takeoff_ramp_duration_s)
            return TakeoffTarget(
                TakeoffPhase.TAKEOFF_RAMP,
                ramp_elapsed,
                progress,
                _interpolate_pose(settled_pose_world, final_pose, progress),
                True,
            )

        hold_elapsed = ramp_elapsed - self.config.takeoff_ramp_duration_s
        if hold_elapsed < self.config.hover_hold_duration_s:
            return TakeoffTarget(TakeoffPhase.HOVER_HOLD, hold_elapsed, 1.0, final_pose, True)
        return TakeoffTarget(TakeoffPhase.COMPLETE, hold_elapsed, 1.0, final_pose, True)

    def final_hover_pose(self, settled_pose_world: Pose7D) -> Pose7D:
        return (
            float(settled_pose_world[0]),
            float(settled_pose_world[1]),
            float(settled_pose_world[2]) + self.config.hover_height_delta_m,
            0.0,
            0.0,
            0.0,
            1.0,
        )


@dataclass
class RandomMorphologyTakeoffResult(SchemaBase):
    graph_id: str
    attempted: bool
    dry_run: bool
    isaac_backed: bool
    unit_contract_passed: bool
    real_isaac_passed: bool
    placement: dict[str, Any]
    metrics: dict[str, float | int | bool | str | list[Any] | dict[str, Any]] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None


def validate_takeoff_morphology(morphology_graph: MorphologyGraph) -> None:
    module_ids = [module.module_id for module in morphology_graph.modules]
    if not module_ids:
        raise SchemaValidationError("random morphology takeoff requires at least one module")
    if len(set(module_ids)) != len(module_ids):
        raise SchemaValidationError("random morphology takeoff module ids must be unique")
    if morphology_graph.base_module_id not in module_ids:
        raise SchemaValidationError("random morphology takeoff base module is missing")
    if morphology_graph.is_closed_loop:
        raise SchemaValidationError("random morphology takeoff requires an open-loop tree")
    if len(module_ids) > 1 and len(morphology_graph.dock_edges) != len(module_ids) - 1:
        raise SchemaValidationError("random morphology takeoff requires exactly N-1 dock edges")
    adjacency: dict[int, set[int]] = {module_id: set() for module_id in module_ids}
    for edge in morphology_graph.dock_edges:
        if edge.src_module_id not in adjacency or edge.dst_module_id not in adjacency:
            raise SchemaValidationError("random morphology takeoff dock edge references a missing module")
        adjacency[edge.src_module_id].add(edge.dst_module_id)
        adjacency[edge.dst_module_id].add(edge.src_module_id)
    visited: set[int] = set()
    frontier = [morphology_graph.base_module_id]
    while frontier:
        module_id = frontier.pop(0)
        if module_id in visited:
            continue
        visited.add(module_id)
        frontier.extend(sorted(adjacency[module_id] - visited))
    if visited != set(module_ids):
        raise SchemaValidationError("random morphology takeoff graph must be connected")


def intended_dock_body_link_pairs(
    morphology_graph: MorphologyGraph,
    physical_model: PhysicalModel,
) -> list[tuple[int, str, int, str]]:
    """Resolve the one intended collision-body exemption for every dock edge."""

    validate_takeoff_morphology(morphology_graph)
    ports_by_id = {port.port_global_id: port for port in morphology_graph.ports}
    if len(ports_by_id) != len(morphology_graph.ports):
        raise SchemaValidationError("morphology graph has duplicate global port ids")
    dock_specs_by_id = {port.port_id: port for port in physical_model.dock_ports}
    if len(dock_specs_by_id) != len(physical_model.dock_ports):
        raise SchemaValidationError("physical model has duplicate dock port ids")

    pairs: list[tuple[int, str, int, str]] = []
    for edge in sorted(morphology_graph.dock_edges, key=lambda item: item.edge_id):
        src_port = ports_by_id.get(edge.src_port_id)
        dst_port = ports_by_id.get(edge.dst_port_id)
        if src_port is None or src_port.module_id != edge.src_module_id:
            raise SchemaValidationError(
                f"dock edge {edge.edge_id} has an invalid source port binding"
            )
        if dst_port is None or dst_port.module_id != edge.dst_module_id:
            raise SchemaValidationError(
                f"dock edge {edge.edge_id} has an invalid destination port binding"
            )
        src_spec = dock_specs_by_id.get(src_port.port_local_id)
        dst_spec = dock_specs_by_id.get(dst_port.port_local_id)
        if src_spec is None or dst_spec is None:
            raise SchemaValidationError(
                f"dock edge {edge.edge_id} references an unknown physical dock port"
            )
        pairs.append(
            (
                edge.src_module_id,
                src_spec.parent_link,
                edge.dst_module_id,
                dst_spec.parent_link,
            )
        )
    if len(set(pairs)) != len(pairs):
        raise SchemaValidationError("dock edges resolve to duplicate collision-body pairs")
    return pairs


def compute_floor_contact_placement(
    morphology_graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    mesh_search_dirs: list[str | Path] | None = None,
    floor_z_m: float = 0.0,
    clearance_m: float = 0.002,
) -> FloorContactPlacement:
    validate_takeoff_morphology(morphology_graph)
    if clearance_m < 0.0:
        raise SchemaValidationError("floor clearance must be non-negative")
    bounds = compute_morphology_collision_bounds(
        morphology_graph,
        physical_model,
        mesh_search_dirs=mesh_search_dirs,
    )
    root_z = float(floor_z_m) + float(clearance_m) - bounds.minimum[2]
    root_pose: Pose7D = (0.0, 0.0, root_z, 0.0, 0.0, 0.0, 1.0)
    return FloorContactPlacement(
        root_pose_world=root_pose,
        collision_bounds_root=bounds,
        floor_z_m=float(floor_z_m),
        clearance_m=float(clearance_m),
        initial_lowest_collision_z_world=bounds.minimum[2] + root_z,
    )


def compute_morphology_collision_bounds(
    morphology_graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    mesh_search_dirs: list[str | Path] | None = None,
) -> MorphologyCollisionBounds:
    validate_takeoff_morphology(morphology_graph)
    module_bounds = morphology_collision_aabbs(
        morphology_graph,
        physical_model,
        mesh_search_dirs=mesh_search_dirs,
    )
    urdf_model = load_urdf(physical_model.urdf_path)
    link_poses_root = link_poses_in_root_frame(urdf_model)
    base_link = urdf_model.metadata.get("baselink", {}).get("name", "fc")
    if base_link not in link_poses_root:
        raise SchemaValidationError(f"Holon module frame {base_link!r} is missing")
    root_to_design = link_poses_root[base_link]
    points = [
        _transform_point(root_to_design, point)
        for module_id in sorted(module_bounds)
        for point in _bounds_corners(*module_bounds[module_id])
    ]
    if not points:
        raise SchemaValidationError("Holon URDF has no supported collision geometry")
    minimum = tuple(min(point[axis] for point in points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in points) for axis in range(3))
    per_module_geometry_count = len(physical_model.collision_primitives)
    mesh_count = sum(
        1 for primitive in physical_model.collision_primitives if primitive.primitive_type == "mesh"
    )
    primitive_count = per_module_geometry_count - mesh_count
    module_count = len(morphology_graph.modules)
    return MorphologyCollisionBounds(
        minimum=minimum,  # type: ignore[arg-type]
        maximum=maximum,  # type: ignore[arg-type]
        collision_geometry_count=per_module_geometry_count * module_count,
        mesh_geometry_count=mesh_count * module_count,
        primitive_geometry_count=primitive_count * module_count,
    )


class RandomMorphologyTakeoffEnv:
    """Graph-specific dry contract and real-Isaac probe boundary."""

    def __init__(
        self,
        *,
        config: RandomMorphologyTakeoffConfig | None = None,
        backend: IsaacLabBackend | None = None,
        physical_model: PhysicalModel | None = None,
        command_executor: Callable[[list[str], float], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or RandomMorphologyTakeoffConfig()
        self.backend = backend or IsaacLabBackend(IsaacLabBackendConfig())
        self.physical_model = physical_model or build_physical_model_from_config(
            self.config.robot_model_config_path
        )
        physical_urdf_path = Path(
            os.path.expandvars(os.path.expanduser(self.physical_model.urdf_path))
        ).resolve()
        backend_urdf_path = Path(
            os.path.expandvars(os.path.expanduser(self.backend.config.holon_urdf_path))
        ).resolve()
        if physical_urdf_path != backend_urdf_path:
            raise SchemaValidationError(
                "random morphology takeoff requires PhysicalModel and Isaac backend to use the same URDF "
                f"({physical_urdf_path} != {backend_urdf_path})"
            )
        self.command_executor = command_executor or _run_json_command

    def placement_for(self, morphology_graph: MorphologyGraph) -> FloorContactPlacement:
        alignment = fixed_dock_connect_frame_alignment(
            morphology_graph,
            self.physical_model,
        )
        if (
            alignment.max_position_error_m
            > FIXED_DOCK_CONNECT_FRAME_POSITION_TOLERANCE_M
            or alignment.max_attitude_error_rad
            > FIXED_DOCK_CONNECT_FRAME_ATTITUDE_TOLERANCE_RAD
        ):
            raise SchemaValidationError(
                "random morphology graph dock frames do not align under the current "
                "PhysicalModel; regenerate the morphology pool after changing the URDF "
                f"(position_error_m={alignment.max_position_error_m:.9g}, "
                f"attitude_error_rad={alignment.max_attitude_error_rad:.9g})"
            )
        return compute_floor_contact_placement(
            morphology_graph,
            self.physical_model,
            mesh_search_dirs=self.config.mesh_search_dirs,
            floor_z_m=0.0,
            clearance_m=self.config.floor_clearance_m,
        )

    def build_probe_command(self, morphology_graph: MorphologyGraph) -> list[str]:
        placement = self.placement_for(morphology_graph)
        command = self.backend.holon_spawn_probe_command(
            config_path=self.config.backend_config_path,
            convert_if_missing=True,
            force_convert=False,
            steps=self.config.required_steps,
        )
        command.extend(
            [
                "--random-morphology-takeoff",
                "--random-morphology-graph-json",
                morphology_graph.to_json(),
                "--dt",
                str(self.config.simulation_dt_s),
                "--spawn-height",
                str(placement.root_pose_world[2]),
                "--floor-clearance-m",
                str(self.config.floor_clearance_m),
                "--takeoff-floor-contact-force-threshold-n",
                str(self.config.floor_contact_force_threshold_n),
                "--takeoff-floor-contact-dwell-duration-s",
                str(self.config.floor_contact_dwell_duration_s),
                "--takeoff-exact-cross-module-contact-force-threshold-n",
                str(self.config.exact_cross_module_contact_force_threshold_n),
                "--takeoff-exact-cross-module-contact-max-patches-per-body-pair",
                str(
                    self.config.exact_cross_module_contact_max_patches_per_body_pair
                ),
                "--takeoff-dock-joint-position-tolerance-rad",
                str(self.config.dock_joint_position_tolerance_rad),
                "--takeoff-initial-root-position-tolerance-m",
                str(self.config.initial_root_position_tolerance_m),
                "--takeoff-initial-root-attitude-tolerance-rad",
                str(self.config.initial_root_attitude_tolerance_rad),
                "--takeoff-settle-duration-s",
                str(self.config.settle_duration_s),
                "--takeoff-settle-dwell-duration-s",
                str(self.config.settle_dwell_duration_s),
                "--takeoff-ramp-duration-s",
                str(self.config.takeoff_ramp_duration_s),
                "--takeoff-hover-height-delta-m",
                str(self.config.hover_height_delta_m),
                "--hover-hold-duration-s",
                str(self.config.hover_hold_duration_s),
                "--takeoff-hover-acquisition-timeout-s",
                str(self.config.hover_acquisition_timeout_s),
                "--hover-position-tolerance-m",
                str(self.config.position_error_threshold_m),
                "--hover-attitude-tolerance-rad",
                str(self.config.attitude_error_threshold_rad),
                "--takeoff-settle-linear-speed-threshold-mps",
                str(self.config.settle_linear_speed_threshold_mps),
                "--takeoff-settle-angular-speed-threshold-rad-s",
                str(self.config.settle_angular_speed_threshold_rad_s),
                "--takeoff-hover-linear-speed-threshold-mps",
                str(self.config.hover_linear_speed_threshold_mps),
                "--takeoff-hover-angular-speed-threshold-rad-s",
                str(self.config.hover_angular_speed_threshold_rad_s),
                "--takeoff-max-vertical-speed-mps",
                str(self.config.max_vertical_speed_mps),
                "--takeoff-min-height-gain-ratio",
                str(self.config.min_height_gain_ratio),
                "--allocation-mode",
                self.config.allocation_mode,
                "--control-contract-version",
                self.config.control_contract_version,
            ]
        )
        for mesh_search_dir in self.config.mesh_search_dirs:
            command.extend(
                ["--random-morphology-mesh-search-dir", str(mesh_search_dir)]
            )
        if not self.config.stop_on_hover_hold:
            command.append("--no-hover-stop-on-hold")
        return command

    def run(self, morphology_graph: MorphologyGraph, *, dry_run: bool = True) -> RandomMorphologyTakeoffResult:
        placement = self.placement_for(morphology_graph)
        collision_geometry_hash = collision_geometry_content_hash(
            self.physical_model,
            mesh_search_dirs=self.config.mesh_search_dirs,
        )
        unit_metrics = {
            "module_count": len(morphology_graph.modules),
            "dock_edge_count": len(morphology_graph.dock_edges),
            "required_steps": self.config.required_steps,
            "total_duration_s": self.config.total_duration_s,
            "spawn_root_height_m": placement.root_pose_world[2],
            "initial_floor_gap_m": placement.floor_gap_m,
            "collision_geometry_count": placement.collision_bounds_root.collision_geometry_count,
            "floor_placement_method": placement.collision_bounds_root.method,
            "morphology_hash": stable_hash(morphology_graph),
            "collision_geometry_hash": collision_geometry_hash,
        }
        if dry_run:
            return RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=False,
                dry_run=True,
                isaac_backed=False,
                unit_contract_passed=True,
                real_isaac_passed=False,
                placement=placement.to_dict(),
                metrics=unit_metrics,
                report={"probe_command": self.build_probe_command(morphology_graph)},
            )

        availability = self.backend.availability()
        if not availability.available:
            reason = ",".join(availability.missing_reasons)
            return RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=False,
                dry_run=False,
                isaac_backed=False,
                unit_contract_passed=True,
                real_isaac_passed=False,
                placement=placement.to_dict(),
                metrics={**unit_metrics, "isaac_backend_available": False},
                failure_reason=reason,
            )
        try:
            report = self.command_executor(self.build_probe_command(morphology_graph), self.config.command_timeout_s)
        except Exception as exc:  # pragma: no cover - environment-specific subprocess failure.
            return RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=True,
                dry_run=False,
                isaac_backed=True,
                unit_contract_passed=True,
                real_isaac_passed=False,
                placement=placement.to_dict(),
                metrics=unit_metrics,
                failure_reason=str(exc),
            )
        return random_morphology_takeoff_result_from_report(
            morphology_graph,
            placement=placement,
            report=report,
            expected_backend_config_hash=self.backend.config.stable_hash(),
            expected_physical_model_hash=self.physical_model.stable_hash(),
            expected_collision_geometry_hash=collision_geometry_hash,
            expected_config=self.config,
            unit_metrics=unit_metrics,
        )


def random_morphology_takeoff_result_from_report(
    morphology_graph: MorphologyGraph,
    *,
    placement: FloorContactPlacement,
    report: dict[str, Any],
    expected_backend_config_hash: str,
    expected_physical_model_hash: str,
    expected_collision_geometry_hash: str,
    expected_config: RandomMorphologyTakeoffConfig,
    unit_metrics: dict[str, Any] | None = None,
    expected_learned_policy: bool = False,
) -> RandomMorphologyTakeoffResult:
    failures = _random_morphology_takeoff_report_failures(
        morphology_graph,
        placement=placement,
        report=report,
        expected_backend_config_hash=expected_backend_config_hash,
        expected_physical_model_hash=expected_physical_model_hash,
        expected_collision_geometry_hash=expected_collision_geometry_hash,
        expected_config=expected_config,
        expected_learned_policy=expected_learned_policy,
    )
    passed = not failures
    metrics = dict(unit_metrics or {})
    for key, value in report.items():
        if key.startswith("random_morphology_takeoff_") and isinstance(value, (str, int, float, bool)):
            metrics[key] = value
    phase_counts = report.get("random_morphology_takeoff_phase_counts")
    if isinstance(phase_counts, dict):
        metrics["random_morphology_takeoff_phase_counts"] = phase_counts
    metrics["random_morphology_takeoff_report_validation_failures"] = failures
    return RandomMorphologyTakeoffResult(
        graph_id=morphology_graph.graph_id,
        attempted=True,
        dry_run=False,
        isaac_backed=report.get("isaac_backed") is True,
        unit_contract_passed=True,
        real_isaac_passed=passed,
        placement=placement.to_dict(),
        metrics=metrics,
        report=report,
        failure_reason=(
            None
            if passed
            else "random_morphology_takeoff_report_validation_failed:"
            + ",".join(failures)
        ),
    )


def _random_morphology_takeoff_report_failures(
    morphology_graph: MorphologyGraph,
    *,
    placement: FloorContactPlacement,
    report: dict[str, Any],
    expected_backend_config_hash: str,
    expected_physical_model_hash: str,
    expected_collision_geometry_hash: str,
    expected_config: RandomMorphologyTakeoffConfig,
    expected_learned_policy: bool = False,
) -> list[str]:
    """Validate independently reported Order-2 evidence without permissive defaults."""

    failures: list[str] = []

    def require_exact(key: str, expected: Any) -> None:
        if key not in report:
            failures.append(f"missing:{key}")
        elif type(report[key]) is not type(expected) or report[key] != expected:
            failures.append(f"mismatch:{key}")

    def require_true(key: str) -> None:
        require_exact(key, True)

    def require_false(key: str) -> None:
        require_exact(key, False)

    def require_int(key: str, *, minimum: int | None = None, expected: int | None = None) -> int | None:
        if key not in report:
            failures.append(f"missing:{key}")
            return None
        value = report[key]
        if not isinstance(value, int) or isinstance(value, bool):
            failures.append(f"invalid_type:{key}")
            return None
        if expected is not None and value != expected:
            failures.append(f"mismatch:{key}")
        if minimum is not None and value < minimum:
            failures.append(f"below_minimum:{key}")
        return value

    def require_number(key: str, *, positive: bool = False) -> float | None:
        if key not in report:
            failures.append(f"missing:{key}")
            return None
        value = report[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            failures.append(f"invalid_type:{key}")
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            failures.append(f"non_finite:{key}")
            return None
        if positive and numeric <= 0.0:
            failures.append(f"not_positive:{key}")
        return numeric

    def require_not_greater(value_key: str, limit_key: str) -> None:
        value = require_number(value_key)
        limit = require_number(limit_key, positive=True)
        if value is not None and limit is not None and value > limit + 1.0e-9:
            failures.append(f"exceeds:{value_key}>{limit_key}")

    def require_not_less(value_key: str, minimum_key: str) -> None:
        value = require_number(value_key)
        minimum = require_number(minimum_key, positive=True)
        if value is not None and minimum is not None and value + 1.0e-9 < minimum:
            failures.append(f"below:{value_key}<{minimum_key}")

    require_true("spawn_passed")
    require_true("isaac_backed")
    require_true("command_applied")
    require_true("command_probe_passed")
    require_int("command_returncode", expected=0)
    require_exact("allocation_mode", "rigid_body_qp")

    require_true("random_morphology_takeoff_smoke")
    require_true("random_morphology_takeoff_smoke_passed")
    require_exact("random_morphology_takeoff_graph_id", morphology_graph.graph_id)
    require_exact(
        "random_morphology_takeoff_morphology_hash",
        morphology_graph.stable_hash(),
    )
    require_exact(
        "random_morphology_takeoff_backend_config_hash",
        expected_backend_config_hash,
    )
    require_exact(
        "random_morphology_takeoff_physical_model_hash",
        expected_physical_model_hash,
    )
    require_exact(
        "random_morphology_takeoff_collision_geometry_hash",
        expected_collision_geometry_hash,
    )
    require_exact("random_morphology_takeoff_allocation_mode", "rigid_body_qp")
    module_count = len(morphology_graph.modules)
    require_int("random_morphology_takeoff_module_count", expected=module_count)
    require_int(
        "random_morphology_takeoff_dock_edge_count",
        expected=len(morphology_graph.dock_edges),
    )
    require_true("random_morphology_takeoff_single_articulation")
    require_exact(
        "random_morphology_takeoff_assembly_representation",
        "reset_time_fixed_dock_tree",
    )
    require_exact(
        "random_morphology_takeoff_learned_policy_used",
        expected_learned_policy,
    )
    require_exact(
        "random_morphology_takeoff_controller",
        (
            "order3_morphology_conditioned_pi_l_plus_deterministic_qpid"
            if expected_learned_policy
            else "deterministic_qpid"
        ),
    )
    require_true("random_morphology_takeoff_fixed_dock_neutral_hold_passed")
    require_int(
        "random_morphology_takeoff_fixed_dock_joint_count",
        expected=module_count * len(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in build_physical_model_from_config(
                    expected_config.robot_model_config_path
                ).dock_ports
                if port.mechanical_limits.get("mechanism_joint_id")
            }
        ),
    )
    require_exact(
        "random_morphology_takeoff_dock_joint_position_tolerance_rad",
        expected_config.dock_joint_position_tolerance_rad,
    )
    require_not_greater(
        "random_morphology_takeoff_max_abs_dock_joint_position_rad",
        "random_morphology_takeoff_dock_joint_position_tolerance_rad",
    )
    require_not_greater(
        "random_morphology_takeoff_final_max_abs_dock_joint_position_rad",
        "random_morphology_takeoff_dock_joint_position_tolerance_rad",
    )
    for key in (
        "random_morphology_takeoff_max_abs_dock_position_target_rad",
        "random_morphology_takeoff_max_abs_dock_velocity_target_rad_s",
        "random_morphology_takeoff_max_abs_dock_torque_bias_nm",
    ):
        value = require_number(key)
        if value is not None and abs(value) > 1.0e-12:
            failures.append(f"nonzero:{key}")
    reported_contract = report.get(
        "random_morphology_takeoff_control_contract_version",
        POLICY_COMMAND_CONTRACT_LEGACY,
    )
    if reported_contract != expected_config.control_contract_version:
        failures.append("mismatch:random_morphology_takeoff_control_contract_version")
    if expected_config.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL:
        require_true("random_morphology_takeoff_true_centroidal_tracking")
        require_false("random_morphology_takeoff_contact_wrench_tracking_claim")
        require_false("random_morphology_takeoff_internal_wrench_tracking_claim")
        require_exact(
            "random_morphology_takeoff_qp_actuator_variable_scope",
            "rotor_thrust_vectoring_and_slack_only",
        )
        require_exact(
            "random_morphology_takeoff_tracking_state_source",
            "true_morphology_centroidal_frame",
        )
        control_pose_history = report.get("random_morphology_takeoff_control_pose_history")
        if not isinstance(control_pose_history, list) or not control_pose_history:
            failures.append("invalid_or_missing:random_morphology_takeoff_control_pose_history")
    require_exact(
        "random_morphology_takeoff_sim_dt_s", expected_config.simulation_dt_s
    )
    require_true("random_morphology_takeoff_sim_dt_matches_config")

    require_true("random_morphology_takeoff_floor_spawned")
    require_true("random_morphology_takeoff_floor_pose_evidenced")
    require_exact("random_morphology_takeoff_floor_placement", placement.to_dict())
    require_exact(
        "random_morphology_takeoff_initial_root_pose_world",
        list(placement.root_pose_world),
    )
    require_not_greater(
        "random_morphology_takeoff_initial_root_position_error_m",
        "random_morphology_takeoff_initial_root_position_tolerance_m",
    )
    require_exact(
        "random_morphology_takeoff_initial_root_position_tolerance_m",
        expected_config.initial_root_position_tolerance_m,
    )
    require_not_greater(
        "random_morphology_takeoff_initial_root_attitude_error_rad",
        "random_morphology_takeoff_initial_root_attitude_tolerance_rad",
    )
    require_exact(
        "random_morphology_takeoff_initial_root_attitude_tolerance_rad",
        expected_config.initial_root_attitude_tolerance_rad,
    )
    require_true("random_morphology_takeoff_floor_contact_evidenced")
    require_not_less(
        "random_morphology_takeoff_floor_contact_max_aggregate_force_n",
        "random_morphology_takeoff_floor_contact_force_threshold_n",
    )
    require_exact(
        "random_morphology_takeoff_floor_contact_force_threshold_n",
        expected_config.floor_contact_force_threshold_n,
    )
    require_not_less(
        "random_morphology_takeoff_floor_contact_dwell_time_s",
        "random_morphology_takeoff_floor_contact_dwell_required_s",
    )
    require_exact(
        "random_morphology_takeoff_floor_contact_dwell_required_s",
        expected_config.floor_contact_dwell_duration_s,
    )
    contact_sensor_body_count = require_int(
        "random_morphology_takeoff_contact_sensor_body_count", minimum=1
    )
    require_exact(
        "random_morphology_takeoff_contact_external_collider_scope",
        "floor_only",
    )
    require_true("random_morphology_takeoff_self_collisions_enabled")
    require_true(
        "random_morphology_takeoff_exact_cross_module_collision_passed"
    )
    require_true(
        "random_morphology_takeoff_exact_nonadjacent_collision_passed"
    )
    exact_rigid_body_count = require_int(
        "random_morphology_takeoff_exact_collision_rigid_body_count",
        minimum=module_count,
    )
    if (
        exact_rigid_body_count is not None
        and contact_sensor_body_count is not None
        and exact_rigid_body_count != contact_sensor_body_count
    ):
        failures.append("mismatch:random_morphology_takeoff_exact_collision_body_counts")
    filtered_body_pair_count = require_int(
        "random_morphology_takeoff_exact_collision_filtered_body_pair_count",
        minimum=1,
    )
    same_module_filtered_body_pair_count = require_int(
        "random_morphology_takeoff_exact_collision_same_module_filtered_body_pair_count",
        minimum=1,
    )
    intended_dock_body_pair_count = require_int(
        "random_morphology_takeoff_exact_collision_intended_dock_body_pair_count",
        expected=len(morphology_graph.dock_edges),
    )
    if (
        filtered_body_pair_count is not None
        and same_module_filtered_body_pair_count is not None
        and intended_dock_body_pair_count is not None
        and filtered_body_pair_count
        != same_module_filtered_body_pair_count + intended_dock_body_pair_count
    ):
        failures.append(
            "mismatch:random_morphology_takeoff_exact_collision_filtered_body_pair_scope"
        )
    expected_dock_link_pairs = [
        list(pair)
        for pair in intended_dock_body_link_pairs(
            morphology_graph,
            physical_model=build_physical_model_from_config(
                expected_config.robot_model_config_path
            ),
        )
    ]
    require_exact(
        "random_morphology_takeoff_exact_collision_intended_dock_body_link_pairs",
        expected_dock_link_pairs,
    )
    intended_dock_path_pairs = report.get(
        "random_morphology_takeoff_exact_collision_intended_dock_body_pairs"
    )
    if (
        not isinstance(intended_dock_path_pairs, list)
        or len(intended_dock_path_pairs) != len(expected_dock_link_pairs)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(path, str) or not path for path in pair)
            for pair in intended_dock_path_pairs
        )
    ):
        failures.append(
            "invalid_or_missing:random_morphology_takeoff_exact_collision_intended_dock_body_pairs"
        )
    else:
        expected_dock_prim_name_pairs = sorted(
            tuple(
                sorted(
                    (
                        f"module_{src_module_id}__{src_link}",
                        f"module_{dst_module_id}__{dst_link}",
                    )
                )
            )
            for src_module_id, src_link, dst_module_id, dst_link in (
                expected_dock_link_pairs
            )
        )
        reported_dock_prim_name_pairs = sorted(
            tuple(sorted(path.rsplit("/", 1)[-1] for path in pair))
            for pair in intended_dock_path_pairs
        )
        if reported_dock_prim_name_pairs != expected_dock_prim_name_pairs:
            failures.append(
                "mismatch:random_morphology_takeoff_exact_collision_intended_dock_body_pairs"
            )
    require_int(
        "random_morphology_takeoff_exact_collision_adjacent_module_pair_count",
        expected=len(morphology_graph.dock_edges),
    )
    require_int(
        "random_morphology_takeoff_exact_collision_nonadjacent_module_pair_count",
        expected=(module_count * (module_count - 1) // 2)
        - len(morphology_graph.dock_edges),
    )
    for key in (
        "random_morphology_takeoff_exact_nonadjacent_contact_count",
        "random_morphology_takeoff_exact_adjacent_unintended_contact_count",
        "random_morphology_takeoff_filtered_scope_contact_count",
        "random_morphology_takeoff_unclassified_robot_contact_count",
    ):
        require_int(key, expected=0)
    require_exact(
        "random_morphology_takeoff_exact_collision_check_method",
        "isaac_physx_get_initial_collider_pairs_v1",
    )
    require_true(
        "random_morphology_takeoff_exact_collision_fixed_module_root_pose_invariant"
    )
    require_int(
        "random_morphology_takeoff_exact_collision_raw_pair_count",
        minimum=0,
    )
    require_int(
        "random_morphology_takeoff_exact_collision_robot_pair_count",
        minimum=0,
    )
    require_exact(
        "random_morphology_takeoff_dynamic_exact_collision_check_method",
        "omni_physics_tensors_force_matrix_and_contact_data_v2",
    )
    require_exact(
        "random_morphology_takeoff_dynamic_exact_contact_scope",
        "all_cross_module_except_intended_dock_body_pairs",
    )
    module_ids = sorted(module.module_id for module in morphology_graph.modules)
    expected_cross_module_pair_keys = {
        f"{src_module_id}-{dst_module_id}"
        for src_index, src_module_id in enumerate(module_ids)
        for dst_module_id in module_ids[src_index + 1 :]
    }
    expected_cross_module_pair_count = len(expected_cross_module_pair_keys)
    dynamic_contact_view_count = require_int(
        "random_morphology_takeoff_dynamic_exact_contact_view_count",
        expected=expected_cross_module_pair_count,
    )
    dynamic_contact_view_update_count = require_int(
        "random_morphology_takeoff_dynamic_exact_contact_view_update_count",
        minimum=0,
    )
    require_exact(
        "random_morphology_takeoff_dynamic_exact_contact_force_threshold_n",
        expected_config.exact_cross_module_contact_force_threshold_n,
    )
    require_not_greater(
        "random_morphology_takeoff_dynamic_exact_contact_max_force_n",
        "random_morphology_takeoff_dynamic_exact_contact_force_threshold_n",
    )
    require_int(
        "random_morphology_takeoff_dynamic_exact_contact_violation_step_count",
        expected=0,
    )
    pair_max_forces = report.get(
        "random_morphology_takeoff_dynamic_exact_pair_max_forces_n"
    )
    if (
        not isinstance(pair_max_forces, dict)
        or set(pair_max_forces) != expected_cross_module_pair_keys
    ):
        failures.append("mismatch:random_morphology_takeoff_dynamic_exact_pair_max_forces_n")
    elif any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) > expected_config.exact_cross_module_contact_force_threshold_n
        for value in pair_max_forces.values()
    ):
        failures.append("exceeds:random_morphology_takeoff_dynamic_exact_pair_max_forces_n")
    require_exact(
        "random_morphology_takeoff_dynamic_exact_raw_contact_method",
        "omni_physics_tensors_get_contact_data_v1",
    )
    require_int(
        "random_morphology_takeoff_dynamic_exact_raw_contact_max_patches_per_body_pair",
        expected=expected_config.exact_cross_module_contact_max_patches_per_body_pair,
    )
    raw_contact_capacity = require_int(
        "random_morphology_takeoff_dynamic_exact_raw_contact_capacity",
        minimum=1,
    )
    if (
        exact_rigid_body_count is not None
        and exact_rigid_body_count % module_count == 0
        and raw_contact_capacity is not None
    ):
        body_count_per_module = exact_rigid_body_count // module_count
        expected_raw_contact_capacity = (
            expected_cross_module_pair_count
            * body_count_per_module
            * body_count_per_module
            * expected_config.exact_cross_module_contact_max_patches_per_body_pair
        )
        if raw_contact_capacity != expected_raw_contact_capacity:
            failures.append(
                "mismatch:random_morphology_takeoff_dynamic_exact_raw_contact_capacity"
            )
    elif exact_rigid_body_count is not None:
        failures.append(
            "mismatch:random_morphology_takeoff_exact_collision_uniform_module_body_count"
        )
    dynamic_raw_contact_view_update_count = require_int(
        "random_morphology_takeoff_dynamic_exact_raw_contact_view_update_count",
        minimum=0,
    )
    require_int(
        "random_morphology_takeoff_dynamic_exact_raw_contact_observation_count",
        expected=0,
    )
    require_int(
        "random_morphology_takeoff_dynamic_exact_raw_contact_observed_step_count",
        expected=0,
    )
    require_not_greater(
        "random_morphology_takeoff_dynamic_exact_raw_contact_max_force_n",
        "random_morphology_takeoff_dynamic_exact_contact_force_threshold_n",
    )
    require_number(
        "random_morphology_takeoff_dynamic_exact_raw_contact_min_separation_m"
    )
    require_int(
        "random_morphology_takeoff_dynamic_exact_raw_contact_saturation_step_count",
        expected=0,
    )
    require_false(
        "random_morphology_takeoff_dynamic_exact_raw_contact_observed"
    )
    require_false(
        "random_morphology_takeoff_dynamic_exact_raw_contact_buffer_saturated"
    )
    pair_raw_contact_counts = report.get(
        "random_morphology_takeoff_dynamic_exact_pair_raw_contact_counts"
    )
    if (
        not isinstance(pair_raw_contact_counts, dict)
        or set(pair_raw_contact_counts) != expected_cross_module_pair_keys
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value != 0
            for value in pair_raw_contact_counts.values()
        )
    ):
        failures.append(
            "mismatch:random_morphology_takeoff_dynamic_exact_pair_raw_contact_counts"
        )
    require_exact(
        "random_morphology_takeoff_exact_nonadjacent_contact_pairs", []
    )
    require_exact(
        "random_morphology_takeoff_exact_adjacent_unintended_contact_pairs", []
    )
    require_int("random_morphology_takeoff_resolved_fc_body_count", expected=module_count)

    require_true("random_morphology_takeoff_settle_zero_thrust")
    require_exact(
        "random_morphology_takeoff_settle_duration_s",
        expected_config.settle_duration_s,
    )
    require_true("random_morphology_takeoff_settle_passed")
    require_not_greater(
        "random_morphology_takeoff_settled_linear_speed_mps",
        "random_morphology_takeoff_settle_linear_speed_threshold_mps",
    )
    require_exact(
        "random_morphology_takeoff_settle_linear_speed_threshold_mps",
        expected_config.settle_linear_speed_threshold_mps,
    )
    require_not_greater(
        "random_morphology_takeoff_settled_angular_speed_rad_s",
        "random_morphology_takeoff_settle_angular_speed_threshold_rad_s",
    )
    require_exact(
        "random_morphology_takeoff_settle_angular_speed_threshold_rad_s",
        expected_config.settle_angular_speed_threshold_rad_s,
    )
    require_not_less(
        "random_morphology_takeoff_settle_low_speed_dwell_time_s",
        "random_morphology_takeoff_settle_low_speed_dwell_required_s",
    )
    require_exact(
        "random_morphology_takeoff_settle_low_speed_dwell_required_s",
        expected_config.settle_dwell_duration_s,
    )
    require_true("random_morphology_takeoff_ramp_passed")
    require_exact(
        "random_morphology_takeoff_takeoff_ramp_duration_s",
        expected_config.takeoff_ramp_duration_s,
    )
    ramp_progress = require_number("random_morphology_takeoff_ramp_max_progress")
    ramp_completion_threshold = 1.0 - (
        2.0
        * expected_config.simulation_dt_s
        / max(
            expected_config.takeoff_ramp_duration_s,
            expected_config.simulation_dt_s,
        )
    )
    if (
        ramp_progress is not None
        and ramp_progress + 1.0e-9 < ramp_completion_threshold
    ):
        failures.append("below:random_morphology_takeoff_ramp_max_progress")
    require_not_less(
        "random_morphology_takeoff_height_gain_ratio",
        "random_morphology_takeoff_min_height_gain_ratio",
    )
    require_exact(
        "random_morphology_takeoff_min_height_gain_ratio",
        expected_config.min_height_gain_ratio,
    )
    require_true("random_morphology_takeoff_hover_passed")
    require_exact(
        "random_morphology_takeoff_hover_height_delta_m",
        expected_config.hover_height_delta_m,
    )
    require_exact(
        "random_morphology_takeoff_stop_on_hover_hold",
        expected_config.stop_on_hover_hold,
    )
    settled_pose = report.get("random_morphology_takeoff_settled_pose_world")
    if (
        not isinstance(settled_pose, list)
        or len(settled_pose) != 7
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in settled_pose
        )
    ):
        failures.append("invalid:random_morphology_takeoff_settled_pose_world")
    else:
        expected_hover_target = [
            float(settled_pose[0]),
            float(settled_pose[1]),
            float(settled_pose[2]) + expected_config.hover_height_delta_m,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        require_exact(
            "random_morphology_takeoff_hover_target_pose_world",
            expected_hover_target,
        )
    require_not_greater(
        "random_morphology_takeoff_final_position_error_m",
        "random_morphology_takeoff_position_error_threshold_m",
    )
    require_exact(
        "random_morphology_takeoff_position_error_threshold_m",
        expected_config.position_error_threshold_m,
    )
    require_not_greater(
        "random_morphology_takeoff_final_attitude_error_rad",
        "random_morphology_takeoff_attitude_error_threshold_rad",
    )
    require_exact(
        "random_morphology_takeoff_attitude_error_threshold_rad",
        expected_config.attitude_error_threshold_rad,
    )
    require_not_greater(
        "random_morphology_takeoff_final_linear_speed_mps",
        "random_morphology_takeoff_hover_linear_speed_threshold_mps",
    )
    require_exact(
        "random_morphology_takeoff_hover_linear_speed_threshold_mps",
        expected_config.hover_linear_speed_threshold_mps,
    )
    require_not_greater(
        "random_morphology_takeoff_final_angular_speed_rad_s",
        "random_morphology_takeoff_hover_angular_speed_threshold_rad_s",
    )
    require_exact(
        "random_morphology_takeoff_hover_angular_speed_threshold_rad_s",
        expected_config.hover_angular_speed_threshold_rad_s,
    )
    require_not_less(
        "random_morphology_takeoff_hover_hold_time_s",
        "random_morphology_takeoff_hover_hold_required_s",
    )
    require_exact(
        "random_morphology_takeoff_hover_hold_required_s",
        expected_config.hover_hold_duration_s,
    )
    require_exact(
        "random_morphology_takeoff_hover_acquisition_timeout_s",
        expected_config.hover_acquisition_timeout_s,
    )
    require_not_greater(
        "random_morphology_takeoff_max_vertical_speed_mps",
        "random_morphology_takeoff_max_vertical_speed_threshold_mps",
    )
    require_exact(
        "random_morphology_takeoff_max_vertical_speed_threshold_mps",
        expected_config.max_vertical_speed_mps,
    )
    require_true("random_morphology_takeoff_finite_state")
    require_true("random_morphology_takeoff_logging_passed")

    for key in (
        "random_morphology_takeoff_qp_infeasible_count",
        "random_morphology_takeoff_controller_clipped_count",
        "random_morphology_takeoff_missing_actuator_count",
        "random_morphology_takeoff_unsupported_actuator_count",
        "random_morphology_takeoff_clipped_target_count",
        "random_morphology_takeoff_application_unresolved_target_count",
    ):
        require_int(key, expected=0)
    requested_count = require_int(
        "random_morphology_takeoff_application_requested_target_count",
        minimum=1,
    )
    applied_count = require_int(
        "random_morphology_takeoff_application_applied_target_count",
        minimum=1,
    )
    if requested_count is not None and applied_count is not None and requested_count != applied_count:
        failures.append("mismatch:random_morphology_takeoff_application_target_counts")
    require_int("random_morphology_takeoff_reaction_torque_target_count", minimum=1)
    require_number("random_morphology_takeoff_reaction_torque_abs_sum_nm", positive=True)

    executed_steps = require_int("random_morphology_takeoff_steps", minimum=1)
    requested_steps = require_int(
        "random_morphology_takeoff_requested_steps",
        expected=expected_config.required_steps,
    )
    if (
        executed_steps is not None
        and requested_steps is not None
        and executed_steps > requested_steps
    ):
        failures.append("exceeds:random_morphology_takeoff_steps>requested_steps")
    if (
        executed_steps is not None
        and dynamic_contact_view_count is not None
        and dynamic_contact_view_update_count is not None
        and dynamic_contact_view_update_count
        != executed_steps * dynamic_contact_view_count
    ):
        failures.append(
            "mismatch:random_morphology_takeoff_dynamic_exact_contact_view_update_count"
        )
    if (
        executed_steps is not None
        and dynamic_contact_view_count is not None
        and dynamic_raw_contact_view_update_count is not None
        and dynamic_raw_contact_view_update_count
        != executed_steps * dynamic_contact_view_count
    ):
        failures.append(
            "mismatch:random_morphology_takeoff_dynamic_exact_raw_contact_view_update_count"
        )
    phase_counts = report.get("random_morphology_takeoff_phase_counts")
    if not isinstance(phase_counts, dict):
        failures.append("invalid_or_missing:random_morphology_takeoff_phase_counts")
    else:
        validated_phase_counts: list[int] = []
        for phase in (
            TakeoffPhase.SETTLE.value,
            TakeoffPhase.TAKEOFF_RAMP.value,
            TakeoffPhase.HOVER_HOLD.value,
        ):
            value = phase_counts.get(phase)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                failures.append(f"invalid_phase_count:{phase}")
            else:
                validated_phase_counts.append(value)
        complete_count = phase_counts.get(TakeoffPhase.COMPLETE.value)
        if not isinstance(complete_count, int) or isinstance(complete_count, bool) or complete_count < 0:
            failures.append(f"invalid_phase_count:{TakeoffPhase.COMPLETE.value}")
        else:
            validated_phase_counts.append(complete_count)
        if executed_steps is not None and sum(validated_phase_counts) != executed_steps:
            failures.append("mismatch:random_morphology_takeoff_phase_count_sum")

    phase_transitions = report.get("random_morphology_takeoff_phase_transitions")
    if not isinstance(phase_transitions, list):
        failures.append("invalid_or_missing:random_morphology_takeoff_phase_transitions")
    else:
        expected_transition_times = {
            TakeoffPhase.SETTLE.value: 0.0,
            TakeoffPhase.TAKEOFF_RAMP.value: expected_config.settle_duration_s,
            TakeoffPhase.HOVER_HOLD.value: (
                expected_config.settle_duration_s
                + expected_config.takeoff_ramp_duration_s
            ),
        }
        for phase, expected_time in expected_transition_times.items():
            matches = [
                item
                for item in phase_transitions
                if isinstance(item, dict) and item.get("to_phase") == phase
            ]
            if len(matches) != 1:
                failures.append(f"invalid_phase_transition:{phase}")
                continue
            transition_time = matches[0].get("time_s")
            if (
                not isinstance(transition_time, (int, float))
                or isinstance(transition_time, bool)
                or not math.isfinite(float(transition_time))
                or abs(float(transition_time) - expected_time)
                > expected_config.simulation_dt_s + 1.0e-12
            ):
                failures.append(f"mismatch_phase_transition_time:{phase}")

    for key in (
        "random_morphology_takeoff_runtime_observations",
        "random_morphology_takeoff_policy_commands",
        "random_morphology_takeoff_controller_commands",
        "random_morphology_takeoff_actuator_target_records",
        "random_morphology_takeoff_root_pose_history",
    ):
        value = report.get(key)
        if not isinstance(value, list):
            failures.append(f"invalid_or_missing:{key}")
        elif executed_steps is not None and len(value) != executed_steps:
            failures.append(f"length_mismatch:{key}")

    raw_observations = report.get("random_morphology_takeoff_runtime_observations")
    raw_policy_commands = report.get("random_morphology_takeoff_policy_commands")
    raw_controller_commands = report.get(
        "random_morphology_takeoff_controller_commands"
    )
    raw_actuator_records = report.get(
        "random_morphology_takeoff_actuator_target_records"
    )
    if all(
        isinstance(value, list)
        for value in (
            raw_observations,
            raw_policy_commands,
            raw_controller_commands,
            raw_actuator_records,
        )
    ):
        from amsrr.controllers.isaac_controller_bridge import (
            IsaacActuatorTargetRecord,
        )
        from amsrr.schemas.policies import ControllerCommand, PolicyCommand
        from amsrr.schemas.runtime import RuntimeObservation

        sequence_length = min(
            len(raw_observations),
            len(raw_policy_commands),
            len(raw_controller_commands),
            len(raw_actuator_records),
        )
        expected_module_ids = sorted(
            module.module_id for module in morphology_graph.modules
        )
        for index in range(sequence_length):
            try:
                observation = RuntimeObservation.from_dict(raw_observations[index])
                PolicyCommand.from_dict(raw_policy_commands[index])
                ControllerCommand.from_dict(raw_controller_commands[index])
                actuator_record = IsaacActuatorTargetRecord.from_dict(
                    raw_actuator_records[index]
                )
            except (SchemaValidationError, TypeError, ValueError) as exc:
                failures.append(f"invalid_typed_step_record:{index}:{type(exc).__name__}")
                continue
            expected_time_s = index * expected_config.simulation_dt_s
            if abs(observation.time_s - expected_time_s) > 1.0e-9:
                failures.append(f"mismatch:runtime_observation_time:{index}")
            if observation.morphology_graph.stable_hash() != morphology_graph.stable_hash():
                failures.append(f"mismatch:runtime_observation_graph:{index}")
            if sorted(state.module_id for state in observation.module_states) != expected_module_ids:
                failures.append(f"mismatch:runtime_observation_modules:{index}")
            if actuator_record.backend != "isaac_lab":
                failures.append(f"mismatch:actuator_backend:{index}")
            if actuator_record.morphology_graph_id != morphology_graph.graph_id:
                failures.append(f"mismatch:actuator_graph:{index}")
            if actuator_record.command_index != index:
                failures.append(f"mismatch:actuator_command_index:{index}")
            if abs(actuator_record.time_s - expected_time_s) > 1.0e-9:
                failures.append(f"mismatch:actuator_time:{index}")

    artifacts = report.get("random_morphology_takeoff_artifacts")
    deterministic_order3_baseline = (
        not expected_learned_policy
        and report.get("order3_deterministic_baseline_rollout") is True
    )
    expected_artifacts = {
        "phase": (
            "P4-full-order3-pi-l"
            if expected_learned_policy
            else (
                "P4-full-order3-deterministic-baseline"
                if deterministic_order3_baseline
                else "P4-full-order2"
            )
        ),
        "backend": "isaac_lab",
        "isaac_backed": True,
        "dry_run": False,
        "is_p4_full_completion": False,
        "physical_success_claim": "floor_takeoff_hover_only",
        "object_task_claim": False,
        "learned_policy_claim": expected_learned_policy,
    }
    if not isinstance(artifacts, dict):
        failures.append("invalid_or_missing:random_morphology_takeoff_artifacts")
    else:
        for key, expected in expected_artifacts.items():
            if key not in artifacts:
                failures.append(f"missing:random_morphology_takeoff_artifacts.{key}")
            elif type(artifacts[key]) is not type(expected) or artifacts[key] != expected:
                failures.append(f"mismatch:random_morphology_takeoff_artifacts.{key}")

    return list(dict.fromkeys(failures))


def _bounds_corners(minimum: Vector3, maximum: Vector3) -> list[Vector3]:
    return [
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]


def _transform_point(pose: Pose7D, point: Vector3) -> Vector3:
    transform = transform_from_pose(pose)
    rotated = matvec(transform.rotation, point)
    return (
        transform.translation[0] + rotated[0],
        transform.translation[1] + rotated[1],
        transform.translation[2] + rotated[2],
    )


def _interpolate_pose(start: Pose7D, end: Pose7D, ratio: float) -> Pose7D:
    alpha = min(max(float(ratio), 0.0), 1.0)
    position = tuple((1.0 - alpha) * start[idx] + alpha * end[idx] for idx in range(3))
    start_quat = tuple(float(value) for value in start[3:7])
    end_quat = tuple(float(value) for value in end[3:7])
    if sum(left * right for left, right in zip(start_quat, end_quat, strict=True)) < 0.0:
        end_quat = tuple(-value for value in end_quat)
    quat = tuple((1.0 - alpha) * start_quat[idx] + alpha * end_quat[idx] for idx in range(4))
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 1.0e-12:
        quat = end_quat
        norm = math.sqrt(sum(value * value for value in quat))
    normalized = tuple(value / norm for value in quat)
    return (*position, *normalized)  # type: ignore[return-value]


def _run_json_command(command: list[str], timeout_s: float) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("WARP_CACHE_PATH", "/tmp/amsrr_warp_cache")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    reports: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            reports.append(value)
    if not reports:
        raise RuntimeError(
            f"random morphology takeoff probe produced no JSON report (returncode={completed.returncode}): "
            f"{completed.stderr[-1000:]}"
        )
    report = reports[-1]
    report.setdefault("command_returncode", completed.returncode)
    if completed.returncode != 0:
        raise RuntimeError(str(report.get("error", completed.stderr[-1000:])))
    return report
