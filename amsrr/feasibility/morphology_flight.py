from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from amsrr.controllers.qp_allocator_interface import (
    QPAllocationProblem,
    QPAllocationResult,
    QPAllocatorInterface,
    RotorAllocationSpec,
    VirtualThrustQPAllocator,
)
from amsrr.controllers.rigid_body_model import (
    RigidBodyControlModel,
    RigidBodyControlModelBuilder,
)
from amsrr.feasibility import violation_codes as codes
from amsrr.geometry.pose_math import (
    Transform3D,
    compose_transform,
    matvec,
    quat_to_matrix,
    transform_from_pose,
    transform_from_xyz_rpy,
    transpose,
)
from amsrr.morphology.dock_geometry import (
    module_poses_from_dock_edges,
    relative_pose_for_dock_ports,
)
from amsrr.morphology.graph import PORT_TYPE_ORDER
from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_in_module_frame
from amsrr.robot_model.physical_model_builder import build_module_capability_token
from amsrr.schemas.common import Pose7D, SchemaValidationError, Vector3
from amsrr.schemas.feasibility import FeasibilityResult, Violation, ViolationSeverity
from amsrr.schemas.morphology import MorphologyGraph, PortNode
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.utils.hashing import hash_file, stable_hash

CHECKER_VERSION = "p4_full_order1_morphology_flight_v1"


@dataclass(frozen=True)
class MorphologyFlightFeasibilityConfig:
    """Task-independent gates for the randomized flight-morphology support set.

    ``mesh_search_dirs`` is deliberately caller supplied.  In particular, this
    keeps the runtime robot/asset layout configurable instead of embedding the
    repository's Holon source-mesh path in the checker.
    """

    min_modules: int = 2
    max_modules: int = 8
    required_base_module_id: int = 0
    collision_margin_m: float = 0.03
    min_thrust_margin_ratio: float = 0.15
    gravity_mps2: float = 9.80665
    qp_residual_tolerance: float = 1.0e-2
    mesh_search_dirs: tuple[str | Path, ...] = ()
    pose_translation_tolerance_m: float = 1.0e-6
    pose_rotation_tolerance_rad: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.min_modules < 1 or self.max_modules < self.min_modules:
            raise ValueError("module-count bounds are invalid")
        if self.required_base_module_id < 0:
            raise ValueError("required_base_module_id must be non-negative")
        if self.collision_margin_m < 0.0:
            raise ValueError("collision_margin_m must be non-negative")
        if self.min_thrust_margin_ratio < 0.0:
            raise ValueError("min_thrust_margin_ratio must be non-negative")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if self.qp_residual_tolerance < 0.0:
            raise ValueError("qp_residual_tolerance must be non-negative")


@dataclass(frozen=True)
class _AABB:
    lower: Vector3
    upper: Vector3


@dataclass(frozen=True)
class _CollisionRecord:
    link_id: str
    local_transform: Transform3D
    geometry_type: str
    params: tuple[float, ...]
    mesh_ref: str | None
    mesh_scale: Vector3


class _CollisionGeometryError(RuntimeError):
    pass


class MorphologyFlightFeasibilityChecker:
    """Evaluate a morphology before it enters the floor/takeoff curriculum.

    Collision is intentionally conservative: nominal-q URDF collision geometry
    is reduced to one module-frame AABB and transformed for each module.  Only
    different, non-dock-adjacent modules are compared.  Exact link/mesh contact,
    floor contact and dynamic takeoff remain simulator gates in Order 2.

    Hover uses the production rigid-body model and ``VirtualThrustQPAllocator``.
    It performs distinct solves for weight support and for the configured thrust
    margin.  Velocity limits are omitted from this *steady-state* reachability
    test, while position and thrust limits remain active.
    """

    def __init__(
        self,
        config: MorphologyFlightFeasibilityConfig | None = None,
        *,
        rigid_body_builder: RigidBodyControlModelBuilder | None = None,
        allocator: QPAllocatorInterface | None = None,
    ) -> None:
        self.config = config or MorphologyFlightFeasibilityConfig()
        self._rigid_body_builder = rigid_body_builder or RigidBodyControlModelBuilder()
        self._allocator = allocator or VirtualThrustQPAllocator()

    def check(
        self, morphology: MorphologyGraph, physical_model: PhysicalModel
    ) -> FeasibilityResult:
        violations: list[Violation] = []
        margins: dict[str, float] = {}
        proxy_scores: dict[str, float] = {}
        metadata: dict[str, str | float | int | bool] = {
            "level": "morphology_flight",
            "module_count": len(morphology.modules),
            "dock_edge_count": len(morphology.dock_edges),
            "collision_method": "nominal_q_urdf_collision_mesh_module_aabb",
            "collision_pair_scope": "nonadjacent_inter_module_only",
            "collision_geometry_frame_source": "urdf_xml_origin_scale_nominal_link_kinematics",
            "collision_same_module_pairs_excluded": True,
            "collision_dock_adjacent_pairs_excluded": True,
            "collision_bounds_are_conservative": True,
            "collision_margin_m": self.config.collision_margin_m,
            "collision_geometry_status": "not_run",
            "hover_allocation_mode": "steady_state_rigid_body_qp",
            "hover_reference_state": "design_pose_zero_twist_zero_q_qdot",
            "hover_velocity_limits_applied": False,
            "hover_payload_mass_kg": 0.0,
            "thrust_margin_semantics": "qp_certified_lower_bound_at_configured_threshold",
        }

        try:
            self._structural_checks(morphology, physical_model, violations, margins)
        except (SchemaValidationError, ValueError, KeyError, IndexError) as exc:
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID,
                    f"structural feasibility evaluation failed: {exc}",
                )
            )
        structural_ok = not violations
        metadata["structural_gate_passed"] = structural_ok

        if structural_ok:
            self._collision_check(
                morphology, physical_model, violations, margins, metadata
            )
            self._hover_checks(
                morphology, physical_model, violations, margins, proxy_scores, metadata
            )
        else:
            metadata["collision_geometry_status"] = "skipped_structural_failure"
            metadata["hover_gate_status"] = "skipped_structural_failure"

        hard_violations = [
            item for item in violations if item.severity == ViolationSeverity.HARD
        ]
        proxy_scores.setdefault(
            "S_WRENCH_MARGIN", margins.get("thrust_margin_ratio", -1.0)
        )
        proxy_scores.setdefault("S_COMPACTNESS", 1.0 / max(1, len(morphology.modules)))
        metadata["hard_violation_count"] = len(hard_violations)
        return FeasibilityResult(
            feasible=not hard_violations,
            hard_violations=hard_violations,
            soft_violations=[
                item for item in violations if item.severity != ViolationSeverity.HARD
            ],
            margins=margins,
            proxy_scores=proxy_scores,
            checker_version=CHECKER_VERSION,
            metadata=metadata,
        )

    def _structural_checks(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        try:
            morphology.validate()
            physical_model.validate()
            for module in morphology.modules:
                module.validate()
                module.capability_token.validate()
            for item in (
                *morphology.ports,
                *morphology.dock_edges,
                *morphology.robot_anchors,
                *morphology.control_groups,
            ):
                item.validate()
            for item in (
                *physical_model.links,
                *physical_model.joints,
                *physical_model.rotors,
                *physical_model.dock_ports,
                *physical_model.collision_primitives,
            ):
                item.validate()
        except (SchemaValidationError, ValueError) as exc:
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID,
                    f"morphology/physical-model schema validation failed: {exc}",
                )
            )
            return

        modules = morphology.modules
        module_ids = [item.module_id for item in modules]
        module_id_set = set(module_ids)
        count = len(modules)
        expected_module_type = physical_model.model_id
        expected_capability = build_module_capability_token(
            physical_model,
            module_type=expected_module_type,
        ).to_dict()
        module_binding_ok = all(
            module.module_type == expected_module_type
            and module.capability_token.to_dict() == expected_capability
            for module in modules
        )
        margins["physical_model_module_binding_valid"] = (
            1.0 if module_binding_ok else 0.0
        )
        if not module_binding_ok:
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID,
                    "every morphology module must match the supplied PhysicalModel type and capability token",
                )
            )
        margins["module_count"] = float(count)
        margins["module_count_min_margin"] = float(count - self.config.min_modules)
        margins["module_count_max_margin"] = float(self.config.max_modules - count)
        if count < self.config.min_modules or count > self.config.max_modules:
            violations.append(
                _hard(
                    codes.F_MODULE_COUNT,
                    "morphology module_count="
                    f"{count} is outside [{self.config.min_modules}, {self.config.max_modules}]",
                    margin=min(
                        count - self.config.min_modules, self.config.max_modules - count
                    ),
                    threshold=float(self.config.min_modules),
                )
            )

        marked_bases = [item.module_id for item in modules if item.is_base]
        base_ok = (
            morphology.base_module_id == self.config.required_base_module_id
            and marked_bases == [self.config.required_base_module_id]
        )
        margins["base_module_assigned"] = 1.0 if base_ok else 0.0
        if not base_ok:
            violations.append(
                _hard(
                    codes.F_BASE_MODULE_ASSIGNED,
                    f"base must be module {self.config.required_base_module_id} and be the only is_base module",
                    node_or_edge_ref=f"base:{morphology.base_module_id}",
                )
            )

        edge_ids = [edge.edge_id for edge in morphology.dock_edges]
        if len(edge_ids) != len(set(edge_ids)):
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID, "dock_edges contain duplicate edge_id values"
                )
            )

        adjacency = {module_id: set() for module_id in module_id_set}
        pair_set: set[tuple[int, int]] = set()
        edge_refs_valid = True
        for edge in morphology.dock_edges:
            src = edge.src_module_id
            dst = edge.dst_module_id
            pair = tuple(sorted((src, dst)))
            if src not in module_id_set or dst not in module_id_set or src == dst:
                edge_refs_valid = False
                violations.append(
                    _hard(
                        codes.F_CONNECTED_GRAPH,
                        f"dock edge {edge.edge_id} has an invalid module endpoint",
                        node_or_edge_ref=f"edge:{edge.edge_id}",
                    )
                )
                continue
            if pair in pair_set:
                edge_refs_valid = False
                violations.append(
                    _hard(
                        codes.F_CLOSED_LOOP_REJECT_V1,
                        f"parallel dock edges are not allowed in the v1 tree: modules {pair}",
                        node_or_edge_ref=f"edge:{edge.edge_id}",
                    )
                )
            pair_set.add(pair)
            adjacency[src].add(dst)
            adjacency[dst].add(src)

        reachable: set[int] = set()
        pending = (
            [morphology.base_module_id]
            if morphology.base_module_id in module_id_set
            else []
        )
        while pending:
            module_id = pending.pop()
            if module_id in reachable:
                continue
            reachable.add(module_id)
            pending.extend(sorted(adjacency.get(module_id, set()) - reachable))
        connected = bool(modules) and reachable == module_id_set
        tree_edge_count = len(morphology.dock_edges) == max(0, count - 1)
        margins["connected_module_ratio"] = len(reachable) / max(1, count)
        margins["tree_edge_count_margin"] = float(
            max(0, count - 1) - len(morphology.dock_edges)
        )
        if not connected:
            violations.append(
                _hard(
                    codes.F_CONNECTED_GRAPH, "morphology dock graph must be connected"
                )
            )
        if morphology.is_closed_loop or not tree_edge_count:
            violations.append(
                _hard(
                    codes.F_CLOSED_LOOP_REJECT_V1,
                    "v1 flight morphology must be an open tree with exactly module_count-1 dock edges",
                )
            )

        self._port_checks(morphology, physical_model, violations, margins)
        self._control_group_check(morphology, module_id_set, violations, margins)
        if connected and tree_edge_count and edge_refs_valid:
            self._dock_pose_checks(morphology, violations, margins)

    @staticmethod
    def _control_group_check(
        morphology: MorphologyGraph,
        module_ids: set[int],
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        whole_body_groups = [
            group for group in morphology.control_groups if group.role == "whole_body"
        ]
        valid = (
            len(whole_body_groups) == 1
            and set(whole_body_groups[0].module_ids) == module_ids
        )
        margins["whole_body_control_group_valid"] = 1.0 if valid else 0.0
        if not valid:
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID,
                    "morphology requires exactly one whole_body control group covering every module",
                )
            )

    def _port_checks(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        module_ids = {item.module_id for item in morphology.modules}
        port_ids = [item.port_global_id for item in morphology.ports]
        if len(port_ids) != len(set(port_ids)):
            violations.append(
                _hard(
                    codes.F_PORT_OCCUPANCY,
                    "ports contain duplicate port_global_id values",
                )
            )
            return

        expected_ports = {
            (item.port_id, item.port_type) for item in physical_model.dock_ports
        }
        expected_by_id = {item.port_id: item for item in physical_model.dock_ports}
        ports_by_module: dict[int, list[PortNode]] = {
            module_id: [] for module_id in module_ids
        }
        port_by_id = {item.port_global_id: item for item in morphology.ports}
        inventory_ok = True
        local_pose_ok = True
        compatibility_mask_ok = True
        for port in morphology.ports:
            if port.module_id not in module_ids:
                inventory_ok = False
                continue
            ports_by_module[port.module_id].append(port)
            expected = expected_by_id.get(port.port_local_id)
            if expected is None:
                inventory_ok = False
                continue
            translation_error, rotation_error = _pose_error(
                port.local_pose, expected.local_pose
            )
            if (
                translation_error > self.config.pose_translation_tolerance_m
                or rotation_error > self.config.pose_rotation_tolerance_rad
            ):
                local_pose_ok = False
            expected_mask = [
                1 if port_type in expected.compatible_port_types else 0
                for port_type in PORT_TYPE_ORDER
            ]
            if port.compatible_port_type_mask != expected_mask:
                compatibility_mask_ok = False
        for module_id in sorted(module_ids):
            actual = {
                (item.port_local_id, item.port_type)
                for item in ports_by_module[module_id]
            }
            if actual != expected_ports or len(ports_by_module[module_id]) != len(
                expected_ports
            ):
                inventory_ok = False
        margins["port_inventory_valid"] = 1.0 if inventory_ok else 0.0
        if not inventory_ok:
            violations.append(
                _hard(
                    codes.F_PORT_OCCUPANCY,
                    "each module must expose the PhysicalModel dock-port inventory exactly once",
                )
            )
        margins["port_local_pose_valid"] = 1.0 if local_pose_ok else 0.0
        if not local_pose_ok:
            violations.append(
                _hard(
                    codes.F_PORT_OCCUPANCY,
                    "PortNode.local_pose values must match their PhysicalModel DockPortSpec frames",
                )
            )
        margins["port_compatibility_mask_valid"] = 1.0 if compatibility_mask_ok else 0.0
        if not compatibility_mask_ok:
            violations.append(
                _hard(
                    codes.F_COMPATIBLE_PORT_TYPES,
                    "PortNode.compatible_port_type_mask values must be derived from PhysicalModel DockPortSpec",
                )
            )

        use_count = {port_id: 0 for port_id in port_by_id}
        compatibility_ok = True
        endpoint_ok = True
        for edge in morphology.dock_edges:
            src = port_by_id.get(edge.src_port_id)
            dst = port_by_id.get(edge.dst_port_id)
            if src is None or dst is None:
                endpoint_ok = False
                violations.append(
                    _hard(
                        codes.F_PORT_OCCUPANCY,
                        f"dock edge {edge.edge_id} references a missing port",
                        node_or_edge_ref=f"edge:{edge.edge_id}",
                    )
                )
                continue
            use_count[src.port_global_id] += 1
            use_count[dst.port_global_id] += 1
            if (
                src.module_id != edge.src_module_id
                or dst.module_id != edge.dst_module_id
            ):
                endpoint_ok = False
            if not _ports_compatible(src, dst):
                compatibility_ok = False
                violations.append(
                    _hard(
                        codes.F_COMPATIBLE_PORT_TYPES,
                        f"dock edge {edge.edge_id} connects incompatible ports {src.port_type}/{dst.port_type}",
                        node_or_edge_ref=f"edge:{edge.edge_id}",
                    )
                )

        occupancy_ok = endpoint_ok
        for port_id, port in port_by_id.items():
            expected_occupied = use_count[port_id] == 1
            if use_count[port_id] > 1 or port.occupied != expected_occupied:
                occupancy_ok = False
        margins["port_occupancy_valid"] = 1.0 if occupancy_ok else 0.0
        margins["compatible_port_edges"] = 1.0 if compatibility_ok else 0.0
        if not occupancy_ok:
            violations.append(
                _hard(
                    codes.F_PORT_OCCUPANCY,
                    "dock edge endpoints and PortNode.occupied flags must define one-to-one port use",
                )
            )

    def _dock_pose_checks(
        self,
        morphology: MorphologyGraph,
        violations: list[Violation],
        margins: dict[str, float],
    ) -> None:
        port_by_id = {item.port_global_id: item for item in morphology.ports}
        relative_ok = True
        max_relative_translation = 0.0
        max_relative_rotation = 0.0
        for edge in morphology.dock_edges:
            if edge.src_port_id not in port_by_id or edge.dst_port_id not in port_by_id:
                continue
            expected = relative_pose_for_dock_ports(
                port_by_id[edge.src_port_id], port_by_id[edge.dst_port_id]
            )
            translation_error, rotation_error = _pose_error(
                edge.relative_pose_src_to_dst, expected
            )
            max_relative_translation = max(max_relative_translation, translation_error)
            max_relative_rotation = max(max_relative_rotation, rotation_error)
            if (
                translation_error > self.config.pose_translation_tolerance_m
                or rotation_error > self.config.pose_rotation_tolerance_rad
            ):
                relative_ok = False

        aligned_ok = relative_ok
        max_module_translation = 0.0
        max_module_rotation = 0.0
        try:
            expected_poses = module_poses_from_dock_edges(
                [item.module_id for item in morphology.modules],
                morphology.dock_edges,
                base_module_id=morphology.base_module_id,
            )
            for module in morphology.modules:
                translation_error, rotation_error = _pose_error(
                    module.pose_in_design_frame,
                    expected_poses[module.module_id],
                )
                max_module_translation = max(max_module_translation, translation_error)
                max_module_rotation = max(max_module_rotation, rotation_error)
                if (
                    translation_error > self.config.pose_translation_tolerance_m
                    or rotation_error > self.config.pose_rotation_tolerance_rad
                ):
                    aligned_ok = False
        except (SchemaValidationError, ValueError) as exc:
            aligned_ok = False
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID,
                    f"dock-aligned module poses cannot be resolved: {exc}",
                )
            )

        margins["dock_relative_pose_max_translation_error_m"] = max_relative_translation
        margins["dock_relative_pose_max_rotation_error_rad"] = max_relative_rotation
        margins["dock_aligned_module_pose_max_translation_error_m"] = (
            max_module_translation
        )
        margins["dock_aligned_module_pose_max_rotation_error_rad"] = max_module_rotation
        if not aligned_ok:
            violations.append(
                _hard(
                    codes.F_SCHEMA_VALID,
                    "dock edge relative poses and module design poses must be derived from the referenced dock frames",
                )
            )

    def _collision_check(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        violations: list[Violation],
        margins: dict[str, float],
        metadata: dict[str, str | float | int | bool],
    ) -> None:
        try:
            module_bounds, geometry_count = _morphology_collision_aabbs(
                morphology,
                physical_model,
                tuple(Path(item) for item in self.config.mesh_search_dirs),
            )
        except (
            OSError,
            ET.ParseError,
            SchemaValidationError,
            ValueError,
            _CollisionGeometryError,
        ) as exc:
            metadata["collision_geometry_status"] = "unavailable"
            metadata["collision_geometry_error"] = str(exc)[:240]
            violations.append(
                _hard(
                    codes.F_COARSE_COLLISION,
                    f"coarse collision geometry could not be constructed deterministically: {exc}",
                )
            )
            return

        adjacent = {
            tuple(sorted((edge.src_module_id, edge.dst_module_id)))
            for edge in morphology.dock_edges
        }
        checked_pairs = 0
        colliding_pairs = 0
        min_clearance = math.inf
        modules = sorted(module_bounds)
        for idx, src in enumerate(modules):
            for dst in modules[idx + 1 :]:
                if (src, dst) in adjacent:
                    continue
                checked_pairs += 1
                clearance = _aabb_clearance(module_bounds[src], module_bounds[dst])
                min_clearance = min(min_clearance, clearance)
                if clearance < self.config.collision_margin_m:
                    colliding_pairs += 1
                    violations.append(
                        _hard(
                            codes.F_COARSE_COLLISION,
                            f"non-adjacent modules {src}/{dst} have coarse clearance {clearance:.6g} m",
                            node_or_edge_ref=f"modules:{src},{dst}",
                            margin=clearance - self.config.collision_margin_m,
                            threshold=self.config.collision_margin_m,
                        )
                    )

        metadata["collision_geometry_status"] = "available"
        metadata["collision_geometry_count"] = geometry_count
        metadata["collision_geometry_content_hash"] = (
            collision_geometry_content_hash(
                physical_model,
                mesh_search_dirs=self.config.mesh_search_dirs,
            )
        )
        metadata["collision_adjacent_pair_exclusion_count"] = len(adjacent)
        margins["coarse_collision_checked_pair_count"] = float(checked_pairs)
        margins["coarse_collision_pair_count"] = float(colliding_pairs)
        if math.isfinite(min_clearance):
            margins["coarse_collision_min_clearance_m"] = min_clearance
        margins["coarse_collision_margin_m"] = self.config.collision_margin_m

    def _hover_checks(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        violations: list[Violation],
        margins: dict[str, float],
        proxy_scores: dict[str, float],
        metadata: dict[str, str | float | int | bool],
    ) -> None:
        try:
            observation = _reference_observation(morphology, physical_model)
            model = self._rigid_body_builder.build(
                morphology, physical_model, observation
            )
            model = _without_velocity_limits(model)
            hover_wrench = _hover_wrench_body(model, self.config.gravity_mps2)
            hover = self._allocate(model, physical_model, hover_wrench)
            margin_scale = 1.0 + self.config.min_thrust_margin_ratio
            margin = self._allocate(
                model, physical_model, [value * margin_scale for value in hover_wrench]
            )
        except (SchemaValidationError, ValueError, ArithmeticError) as exc:
            metadata["hover_gate_status"] = "model_or_allocator_error"
            metadata["hover_gate_error"] = str(exc)[:240]
            violations.append(
                _hard(
                    codes.F_QP_HOVER_FEASIBILITY,
                    f"rigid-body hover QP could not be evaluated: {exc}",
                )
            )
            return

        hover_ok = hover.feasible and not hover.clipped
        margin_ok = margin.feasible and not margin.clipped
        required_force = model.total_mass_kg * self.config.gravity_mps2
        certified_margin = (
            self.config.min_thrust_margin_ratio
            if margin_ok
            else (0.0 if hover_ok else -1.0)
        )
        margins.update(
            {
                "robot_mass_kg": model.total_mass_kg,
                "required_total_vertical_force_n": required_force,
                "qp_hover_feasible": 1.0 if hover_ok else 0.0,
                "qp_hover_residual_norm": hover.residual_norm,
                "qp_hover_clipped": 1.0 if hover.clipped else 0.0,
                "qp_margin_feasible": 1.0 if margin_ok else 0.0,
                "qp_margin_residual_norm": margin.residual_norm,
                "qp_margin_clipped": 1.0 if margin.clipped else 0.0,
                "thrust_margin_ratio": certified_margin,
                "thrust_margin_threshold": self.config.min_thrust_margin_ratio,
            }
        )
        for name, result in (("qp_hover", hover), ("qp_margin", margin)):
            for key, value in result.metrics.items():
                if math.isfinite(float(value)):
                    margins[f"{name}_{key}"] = float(value)
        proxy_scores["S_WRENCH_MARGIN"] = certified_margin
        proxy_scores["S_ENERGY_PROXY"] = required_force / max(
            sum(hover.rotor_thrusts_n.values()),
            1.0e-9,
        )
        metadata["hover_gate_status"] = "passed" if hover_ok else "failed"
        metadata["hover_qp_violation_codes"] = ",".join(hover.violation_codes)
        metadata["margin_qp_violation_codes"] = ",".join(margin.violation_codes)
        if not hover_ok:
            violations.append(
                _hard(
                    codes.F_QP_HOVER_FEASIBILITY,
                    "morphology-aware rigid-body QP cannot reproduce the gravity-support wrench without clipping",
                    margin=self.config.qp_residual_tolerance - hover.residual_norm,
                    threshold=0.0,
                )
            )
        if not margin_ok:
            violations.append(
                _hard(
                    codes.F_THRUST_MARGIN,
                    "rigid-body QP cannot certify the configured gravity-wrench thrust margin without clipping",
                    margin=certified_margin,
                    threshold=self.config.min_thrust_margin_ratio,
                )
            )

    def _allocate(
        self,
        model: RigidBodyControlModel,
        physical_model: PhysicalModel,
        desired_wrench: list[float],
    ) -> QPAllocationResult:
        return self._allocator.allocate(
            QPAllocationProblem(
                desired_wrench_body=desired_wrench,
                rotors=[
                    RotorAllocationSpec(
                        rotor_id=rotor.rotor_id,
                        thrust_axis_body=rotor.thrust_axis_local,
                        thrust_min_n=rotor.thrust_min_n,
                        thrust_max_n=rotor.thrust_max_n,
                    )
                    for rotor in physical_model.rotors
                ],
                rigid_body_model=model,
                previous_rotor_thrusts_n={},
                previous_vectoring_joint_targets={},
                control_dt_s=1.0,
                unsupported_wrench_tolerance=self.config.qp_residual_tolerance,
                vertical_tolerance_n=self.config.qp_residual_tolerance,
            )
        )


def check_morphology_flight_feasibility(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    config: MorphologyFlightFeasibilityConfig | None = None,
) -> FeasibilityResult:
    return MorphologyFlightFeasibilityChecker(config).check(morphology, physical_model)


def morphology_collision_aabbs(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    mesh_search_dirs: Iterable[str | Path] | None = None,
) -> dict[int, tuple[Vector3, Vector3]]:
    """Return nominal-q collision AABBs in the morphology design frame.

    The public tuple form keeps the helper independent of checker-internal
    dataclasses.  Order 2 uses the lower z values to place an arbitrary connected
    morphology on the floor without duplicating URDF/STL interpretation.
    """

    bounds, _ = _morphology_collision_aabbs(
        morphology,
        physical_model,
        tuple(Path(item) for item in (mesh_search_dirs or ())),
    )
    return {module_id: (item.lower, item.upper) for module_id, item in bounds.items()}


def morphology_min_collision_z(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    mesh_search_dirs: Iterable[str | Path] | None = None,
) -> float:
    """Return the minimum nominal collision-geometry z in the design frame."""

    bounds = morphology_collision_aabbs(
        morphology,
        physical_model,
        mesh_search_dirs=mesh_search_dirs,
    )
    if not bounds:
        raise _CollisionGeometryError("morphology has no module collision bounds")
    return min(lower[2] for lower, _ in bounds.values())


def collision_geometry_content_hash(
    physical_model: PhysicalModel,
    *,
    mesh_search_dirs: Iterable[str | Path] | None = None,
) -> str:
    """Hash every byte-level input used by the URDF collision-geometry gate.

    ``PhysicalModel.stable_hash()`` intentionally stores mesh references rather
    than mesh bytes.  This companion hash makes feasibility/floor-placement
    provenance change when an STL is edited in place.
    """

    urdf_path = Path(physical_model.urdf_path)
    if not urdf_path.is_file():
        raise _CollisionGeometryError(
            f"PhysicalModel URDF does not exist: {urdf_path}"
        )
    search_dirs = tuple(Path(item) for item in (mesh_search_dirs or ()))
    records = _urdf_collision_records(urdf_path)
    serialized_records: list[dict[str, object]] = []
    for record in records:
        mesh_content_hash = None
        if record.geometry_type == "mesh":
            assert record.mesh_ref is not None
            mesh_path = _resolve_mesh(record.mesh_ref, urdf_path.parent, search_dirs)
            mesh_content_hash = hash_file(mesh_path)
        serialized_records.append(
            {
                "link_id": record.link_id,
                "local_translation": list(record.local_transform.translation),
                "local_rotation": [list(row) for row in record.local_transform.rotation],
                "geometry_type": record.geometry_type,
                "params": list(record.params),
                "mesh_ref": record.mesh_ref,
                "mesh_scale": list(record.mesh_scale),
                "mesh_content_hash": mesh_content_hash,
            }
        )
    return stable_hash(
        {
            "method": "urdf_collision_records_and_resolved_mesh_bytes_v1",
            "urdf_content_hash": hash_file(urdf_path),
            "records": serialized_records,
        }
    )


def _ports_compatible(src: PortNode, dst: PortNode) -> bool:
    try:
        src_index = PORT_TYPE_ORDER.index(src.port_type)
        dst_index = PORT_TYPE_ORDER.index(dst.port_type)
    except ValueError:
        return False
    if src_index >= len(dst.compatible_port_type_mask) or dst_index >= len(
        src.compatible_port_type_mask
    ):
        return False
    return bool(src.compatible_port_type_mask[dst_index]) and bool(
        dst.compatible_port_type_mask[src_index]
    )


def _pose_error(actual: Pose7D, expected: Pose7D) -> tuple[float, float]:
    translation = math.sqrt(
        sum((float(actual[idx]) - float(expected[idx])) ** 2 for idx in range(3))
    )
    actual_q = _normalised_quat(actual[3:7])
    expected_q = _normalised_quat(expected[3:7])
    dot = min(1.0, abs(sum(actual_q[idx] * expected_q[idx] for idx in range(4))))
    return translation, 2.0 * math.acos(dot)


def _normalised_quat(values: Iterable[float]) -> tuple[float, float, float, float]:
    quat = tuple(float(item) for item in values)
    if len(quat) != 4:
        raise ValueError("quaternion must contain four values")
    norm = math.sqrt(sum(item * item for item in quat))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    return tuple(item / norm for item in quat)  # type: ignore[return-value]


def _reference_observation(
    morphology: MorphologyGraph, physical_model: PhysicalModel
) -> RuntimeObservation:
    joint_positions = {
        joint.joint_id: 0.0
        for joint in physical_model.joints
        if joint.joint_type != "fixed"
    }
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions=dict(joint_positions),
                joint_velocities={joint_id: 0.0 for joint_id in joint_positions},
                health=module.health,
            )
            for module in sorted(morphology.modules, key=lambda item: item.module_id)
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )


def _without_velocity_limits(model: RigidBodyControlModel) -> RigidBodyControlModel:
    return replace(
        model,
        active_actuator_limits={
            actuator_id: {**limits, "velocity": None}
            for actuator_id, limits in model.active_actuator_limits.items()
        },
        metadata={**model.metadata, "steady_state_velocity_limits_omitted": True},
    )


def _hover_wrench_body(
    model: RigidBodyControlModel, gravity_mps2: float
) -> list[float]:
    body_to_world = quat_to_matrix(
        (
            float(model.body_pose_world[3]),
            float(model.body_pose_world[4]),
            float(model.body_pose_world[5]),
            float(model.body_pose_world[6]),
        )
    )
    force_world = (0.0, 0.0, model.total_mass_kg * gravity_mps2)
    force_body = matvec(transpose(body_to_world), force_world)
    return [force_body[0], force_body[1], force_body[2], 0.0, 0.0, 0.0]


def _module_collision_aabb(
    physical_model: PhysicalModel,
    mesh_search_dirs: tuple[Path, ...],
) -> tuple[_AABB, int]:
    urdf_path = Path(physical_model.urdf_path)
    if not urdf_path.exists():
        raise _CollisionGeometryError(f"PhysicalModel URDF does not exist: {urdf_path}")
    if not physical_model.collision_primitives:
        raise _CollisionGeometryError("PhysicalModel has no collision_primitives")

    urdf_model = load_urdf(urdf_path)
    link_poses = link_poses_in_module_frame(urdf_model)
    records = _urdf_collision_records(urdf_path)
    if len(records) != len(physical_model.collision_primitives):
        raise _CollisionGeometryError(
            "URDF collision geometry count does not match PhysicalModel.collision_primitives "
            f"({len(records)} != {len(physical_model.collision_primitives)})"
        )
    primitive_links = sorted(
        item.link_id for item in physical_model.collision_primitives
    )
    record_links = sorted(item.link_id for item in records)
    if primitive_links != record_links:
        raise _CollisionGeometryError(
            "URDF collision links do not match PhysicalModel.collision_primitives"
        )
    primitives_by_link: dict[str, list] = {}
    records_by_link: dict[str, list[_CollisionRecord]] = {}
    for primitive in physical_model.collision_primitives:
        primitives_by_link.setdefault(primitive.link_id, []).append(primitive)
    for record in records:
        records_by_link.setdefault(record.link_id, []).append(record)
    for link_id, link_records in records_by_link.items():
        link_primitives = primitives_by_link[link_id]
        for primitive, record in zip(link_primitives, link_records):
            if primitive.primitive_type != record.geometry_type:
                raise _CollisionGeometryError(
                    f"collision type mismatch on link {link_id!r}: "
                    f"{primitive.primitive_type!r} != {record.geometry_type!r}"
                )
            if (
                record.geometry_type == "mesh"
                and primitive.geometry_ref != record.mesh_ref
            ):
                raise _CollisionGeometryError(
                    f"collision mesh mismatch on link {link_id!r}: "
                    f"{primitive.geometry_ref!r} != {record.mesh_ref!r}"
                )

    module_bounds: _AABB | None = None
    for record in records:
        if record.link_id not in link_poses:
            raise _CollisionGeometryError(
                f"collision link {record.link_id!r} has no module-frame transform"
            )
        geometry_bounds = _record_local_aabb(record, urdf_path, mesh_search_dirs)
        module_from_geometry = compose_transform(
            transform_from_pose(link_poses[record.link_id]),
            record.local_transform,
        )
        transformed = _transform_aabb(geometry_bounds, module_from_geometry)
        module_bounds = (
            transformed
            if module_bounds is None
            else _aabb_union(module_bounds, transformed)
        )
    if module_bounds is None:
        raise _CollisionGeometryError("URDF has no supported collision geometry")
    return module_bounds, len(records)


def _morphology_collision_aabbs(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    mesh_search_dirs: tuple[Path, ...],
) -> tuple[dict[int, _AABB], int]:
    local_bounds, geometry_count = _module_collision_aabb(
        physical_model, mesh_search_dirs
    )
    return (
        {
            module.module_id: _transform_aabb(
                local_bounds, transform_from_pose(module.pose_in_design_frame)
            )
            for module in morphology.modules
        },
        geometry_count,
    )


def _urdf_collision_records(urdf_path: Path) -> list[_CollisionRecord]:
    root = ET.parse(urdf_path).getroot()
    records: list[_CollisionRecord] = []
    for link in _children(root, "link"):
        link_id = link.attrib.get("name", "")
        for collision in _children(link, "collision"):
            origin = _child(collision, "origin")
            xyz = _vector_attr(origin, "xyz", (0.0, 0.0, 0.0))
            rpy = _vector_attr(origin, "rpy", (0.0, 0.0, 0.0))
            geometry = _child(collision, "geometry")
            if geometry is None:
                raise _CollisionGeometryError(
                    f"collision on link {link_id!r} has no geometry"
                )
            mesh = _child(geometry, "mesh")
            box = _child(geometry, "box")
            sphere = _child(geometry, "sphere")
            cylinder = _child(geometry, "cylinder")
            if mesh is not None:
                mesh_ref = mesh.attrib.get("filename")
                if not mesh_ref:
                    raise _CollisionGeometryError(
                        f"mesh collision on link {link_id!r} has no filename"
                    )
                record = _CollisionRecord(
                    link_id=link_id,
                    local_transform=transform_from_xyz_rpy(xyz, rpy),
                    geometry_type="mesh",
                    params=(),
                    mesh_ref=mesh_ref,
                    mesh_scale=_vector_text(mesh.attrib.get("scale"), (1.0, 1.0, 1.0)),
                )
            elif box is not None:
                record = _CollisionRecord(
                    link_id=link_id,
                    local_transform=transform_from_xyz_rpy(xyz, rpy),
                    geometry_type="box",
                    params=_vector_text(box.attrib.get("size"), ()),
                    mesh_ref=None,
                    mesh_scale=(1.0, 1.0, 1.0),
                )
            elif sphere is not None:
                record = _CollisionRecord(
                    link_id=link_id,
                    local_transform=transform_from_xyz_rpy(xyz, rpy),
                    geometry_type="sphere",
                    params=(float(sphere.attrib["radius"]),),
                    mesh_ref=None,
                    mesh_scale=(1.0, 1.0, 1.0),
                )
            elif cylinder is not None:
                record = _CollisionRecord(
                    link_id=link_id,
                    local_transform=transform_from_xyz_rpy(xyz, rpy),
                    geometry_type="cylinder",
                    params=(
                        float(cylinder.attrib["radius"]),
                        float(cylinder.attrib["length"]),
                    ),
                    mesh_ref=None,
                    mesh_scale=(1.0, 1.0, 1.0),
                )
            else:
                raise _CollisionGeometryError(
                    f"unsupported collision geometry on link {link_id!r}"
                )
            records.append(record)
    return records


def _record_local_aabb(
    record: _CollisionRecord,
    urdf_path: Path,
    mesh_search_dirs: tuple[Path, ...],
) -> _AABB:
    if record.geometry_type == "mesh":
        assert record.mesh_ref is not None
        mesh_path = _resolve_mesh(record.mesh_ref, urdf_path.parent, mesh_search_dirs)
        bounds = _stl_bounds(str(mesh_path.resolve()))
        return _scaled_aabb(bounds, record.mesh_scale)
    if record.geometry_type == "box":
        if len(record.params) != 3:
            raise _CollisionGeometryError(
                "box collision size must contain three values"
            )
        half = tuple(0.5 * abs(item) for item in record.params)
        return _AABB((-half[0], -half[1], -half[2]), (half[0], half[1], half[2]))
    if record.geometry_type == "sphere":
        radius = abs(record.params[0])
        return _AABB((-radius, -radius, -radius), (radius, radius, radius))
    if record.geometry_type == "cylinder":
        radius, length = abs(record.params[0]), abs(record.params[1])
        return _AABB((-radius, -radius, -0.5 * length), (radius, radius, 0.5 * length))
    raise _CollisionGeometryError(f"unsupported geometry type {record.geometry_type!r}")


def _resolve_mesh(mesh_ref: str, urdf_dir: Path, search_dirs: tuple[Path, ...]) -> Path:
    raw = Path(mesh_ref)
    candidates = [raw] if raw.is_absolute() else [urdf_dir / raw]
    if not raw.is_absolute():
        for search_dir in search_dirs:
            candidates.extend((search_dir / raw, search_dir / raw.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise _CollisionGeometryError(
        f"collision mesh {mesh_ref!r} was not found; searched: "
        + ", ".join(str(item) for item in candidates)
    )


@lru_cache(maxsize=128)
def _stl_bounds(path_text: str) -> _AABB:
    path = Path(path_text)
    data = path.read_bytes()
    vertices: Iterable[tuple[float, float, float]]
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + triangle_count * 50
    else:
        triangle_count = 0
        expected_size = -1
    if expected_size == len(data):
        vertices = _binary_stl_vertices(data, triangle_count)
    else:
        vertices = _ascii_stl_vertices(data, path)

    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    count = 0
    for vertex in vertices:
        if not all(math.isfinite(item) for item in vertex):
            raise _CollisionGeometryError(f"STL contains non-finite vertices: {path}")
        for idx in range(3):
            lower[idx] = min(lower[idx], vertex[idx])
            upper[idx] = max(upper[idx], vertex[idx])
        count += 1
    if count == 0:
        raise _CollisionGeometryError(f"STL contains no vertices: {path}")
    return _AABB(tuple(lower), tuple(upper))  # type: ignore[arg-type]


def _binary_stl_vertices(
    data: bytes, triangle_count: int
) -> Iterable[tuple[float, float, float]]:
    for triangle_idx in range(triangle_count):
        base = 84 + triangle_idx * 50 + 12
        yield struct.unpack_from("<3f", data, base)
        yield struct.unpack_from("<3f", data, base + 12)
        yield struct.unpack_from("<3f", data, base + 24)


def _ascii_stl_vertices(
    data: bytes, path: Path
) -> Iterable[tuple[float, float, float]]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise _CollisionGeometryError(
            f"STL is neither valid binary nor ASCII: {path}"
        ) from exc
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            yield (float(fields[1]), float(fields[2]), float(fields[3]))


def _scaled_aabb(bounds: _AABB, scale: Vector3) -> _AABB:
    points = [
        (
            bounds.lower[0] if mask & 1 == 0 else bounds.upper[0],
            bounds.lower[1] if mask & 2 == 0 else bounds.upper[1],
            bounds.lower[2] if mask & 4 == 0 else bounds.upper[2],
        )
        for mask in range(8)
    ]
    scaled = [
        (point[0] * scale[0], point[1] * scale[1], point[2] * scale[2])
        for point in points
    ]
    return _aabb_from_points(scaled)


def _transform_aabb(bounds: _AABB, transform: Transform3D) -> _AABB:
    points = [
        (
            bounds.lower[0] if mask & 1 == 0 else bounds.upper[0],
            bounds.lower[1] if mask & 2 == 0 else bounds.upper[1],
            bounds.lower[2] if mask & 4 == 0 else bounds.upper[2],
        )
        for mask in range(8)
    ]
    transformed = [
        tuple(
            transform.translation[idx] + matvec(transform.rotation, point)[idx]
            for idx in range(3)
        )
        for point in points
    ]
    return _aabb_from_points(transformed)


def _aabb_from_points(points: Iterable[Vector3]) -> _AABB:
    values = list(points)
    if not values:
        raise _CollisionGeometryError("cannot build an AABB without points")
    return _AABB(
        tuple(min(point[idx] for point in values) for idx in range(3)),  # type: ignore[arg-type]
        tuple(max(point[idx] for point in values) for idx in range(3)),  # type: ignore[arg-type]
    )


def _aabb_union(left: _AABB, right: _AABB) -> _AABB:
    return _AABB(
        tuple(min(left.lower[idx], right.lower[idx]) for idx in range(3)),  # type: ignore[arg-type]
        tuple(max(left.upper[idx], right.upper[idx]) for idx in range(3)),  # type: ignore[arg-type]
    )


def _aabb_clearance(left: _AABB, right: _AABB) -> float:
    axis_gaps = [
        max(left.lower[idx] - right.upper[idx], right.lower[idx] - left.upper[idx], 0.0)
        for idx in range(3)
    ]
    return math.sqrt(sum(item * item for item in axis_gaps))


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [item for item in list(element) if _tag_name(item) == tag]


def _child(element: ET.Element, tag: str) -> ET.Element | None:
    values = _children(element, tag)
    return values[0] if values else None


def _vector_attr(element: ET.Element | None, name: str, default: Vector3) -> Vector3:
    return _vector_text(
        element.attrib.get(name) if element is not None else None, default
    )


def _vector_text(text: str | None, default) -> tuple[float, ...]:
    if text is None:
        return tuple(default)
    values = tuple(float(item) for item in text.split())
    if len(values) != 3:
        raise _CollisionGeometryError(f"expected three-vector, got {text!r}")
    return values


def _hard(
    code: str,
    message: str,
    *,
    node_or_edge_ref: str | None = None,
    margin: float | None = None,
    threshold: float | None = None,
) -> Violation:
    return Violation(
        code=code,
        severity=ViolationSeverity.HARD,
        message=message,
        node_or_edge_ref=node_or_edge_ref,
        margin=margin,
        threshold=threshold,
    )
