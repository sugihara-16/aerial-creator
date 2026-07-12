from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from amsrr.controllers.rigid_body_model import RigidBodyControlModel, RigidBodyControlModelBuilder
from amsrr.geometry.pose_math import compose_pose
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len
from amsrr.schemas.morphology import ControlGroup, DockEdge, MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation


DETACH_WRENCH_ESTIMATOR_VERSION = "follower_subtree_centroidal_v1"
CUT_WRENCH_SIGN_CONVENTION = "parent_component_on_follower_component"


@dataclass
class DetachWrenchEstimate(SchemaBase):
    edge_id: int
    follower_module_ids: list[int]
    valid: bool
    failure_reason: str | None = None
    wrench_follower_com_body: list[float] = field(default_factory=lambda: [0.0] * 6)
    wrench_follower_dock_frame: list[float] = field(default_factory=lambda: [0.0] * 6)
    force_norm_n: float = 0.0
    torque_norm_nm: float = 0.0
    follower_com_pose_world: Pose7D | None = None
    follower_dock_pose_world: Pose7D | None = None
    estimator_version: str = DETACH_WRENCH_ESTIMATOR_VERSION
    sign_convention: str = CUT_WRENCH_SIGN_CONVENTION
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        require_len(self.wrench_follower_com_body, 6, "DetachWrenchEstimate.wrench_follower_com_body")
        require_len(
            self.wrench_follower_dock_frame,
            6,
            "DetachWrenchEstimate.wrench_follower_dock_frame",
        )
        if self.edge_id < 0 or any(module_id < 0 for module_id in self.follower_module_ids):
            raise SchemaValidationError("DetachWrenchEstimate ids must be non-negative")
        if self.force_norm_n < 0.0 or self.torque_norm_nm < 0.0:
            raise SchemaValidationError("DetachWrenchEstimate norms must be non-negative")


class FollowerSubtreeDetachWrenchEstimator:
    """Estimate one cut-edge wrench from follower-subtree centroidal balance.

    The returned sign is the wrench applied by the parent component to the
    follower component.  External-contact evidence is a mandatory independent
    input; false or unknown evidence fails closed.
    """

    def __init__(
        self,
        *,
        rigid_body_model_builder: RigidBodyControlModelBuilder | None = None,
        gravity_mps2: float = 9.80665,
    ) -> None:
        if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be finite and positive")
        self.rigid_body_model_builder = rigid_body_model_builder or RigidBodyControlModelBuilder()
        self.gravity_mps2 = float(gravity_mps2)

    def estimate(
        self,
        *,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        previous_observation: RuntimeObservation,
        observation: RuntimeObservation,
        controller_command: ControllerCommand,
        edge_id: int,
        follower_module_id: int,
        dt_s: float,
        external_contact_free: bool | None,
        other_known_external_wrench_follower_com_body: list[float] | None = None,
    ) -> DetachWrenchEstimate:
        if external_contact_free is not True:
            return _invalid(edge_id, [], "follower_external_contact_free_evidence_missing")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return _invalid(edge_id, [], "invalid_estimator_dt")
        other_external = list(other_known_external_wrench_follower_com_body or [0.0] * 6)
        if len(other_external) != 6 or not all(math.isfinite(float(value)) for value in other_external):
            return _invalid(edge_id, [], "invalid_other_known_external_wrench")

        edge = next((item for item in morphology_graph.dock_edges if item.edge_id == edge_id), None)
        if edge is None or edge.latch_state != "attached":
            return _invalid(edge_id, [], "attached_edge_not_found")
        if follower_module_id not in {edge.src_module_id, edge.dst_module_id}:
            return _invalid(edge_id, [], "follower_module_must_be_cut_edge_endpoint")

        follower_ids = _component_after_cut(morphology_graph, edge, follower_module_id)
        parent_endpoint = edge.dst_module_id if follower_module_id == edge.src_module_id else edge.src_module_id
        if parent_endpoint in follower_ids:
            return _invalid(edge_id, follower_ids, "cut_edge_does_not_separate_graph")
        try:
            subtree = _subtree_graph(morphology_graph, follower_ids, follower_module_id, edge_id)
            previous_model = self.rigid_body_model_builder.build(
                subtree,
                physical_model,
                previous_observation,
            )
            current_model = self.rigid_body_model_builder.build(
                subtree,
                physical_model,
                observation,
            )
            follower_dock_pose_world = _follower_dock_pose_world(
                morphology_graph,
                observation,
                edge,
                follower_module_id,
            )
        except (SchemaValidationError, KeyError, ValueError) as exc:
            return _invalid(edge_id, follower_ids, f"model_build_failed:{type(exc).__name__}")

        momentum_rate = _momentum_rate_wrench_body(previous_model, current_model, float(dt_s))
        actuator_wrench = _known_rotor_wrench_body(current_model, controller_command)
        gravity_wrench = _gravity_wrench_body(current_model, self.gravity_mps2)
        cut_wrench_com = [
            momentum_rate[index]
            - actuator_wrench[index]
            - gravity_wrench[index]
            - float(other_external[index])
            for index in range(6)
        ]
        cut_wrench_dock = transform_follower_com_wrench_to_dock_frame(
            cut_wrench_com,
            current_model.body_pose_world,
            follower_dock_pose_world,
        )
        force_norm = _norm(cut_wrench_dock[:3])
        torque_norm = _norm(cut_wrench_dock[3:])
        return DetachWrenchEstimate(
            edge_id=edge_id,
            follower_module_ids=follower_ids,
            valid=True,
            wrench_follower_com_body=cut_wrench_com,
            wrench_follower_dock_frame=cut_wrench_dock,
            force_norm_n=force_norm,
            torque_norm_nm=torque_norm,
            follower_com_pose_world=current_model.body_pose_world,
            follower_dock_pose_world=follower_dock_pose_world,
            metrics={
                "dt_s": float(dt_s),
                "follower_mass_kg": float(current_model.total_mass_kg),
                "follower_module_count": float(len(follower_ids)),
                "momentum_rate_norm": _norm(momentum_rate),
                "known_actuator_wrench_norm": _norm(actuator_wrench),
                "gravity_wrench_norm": _norm(gravity_wrench),
                "other_known_external_wrench_norm": _norm(other_external),
            },
        )


@dataclass(frozen=True)
class DetachUnloadGateConfig:
    force_threshold_n: float = 0.5
    torque_threshold_nm: float = 0.05
    relative_position_error_threshold_m: float = 0.005
    relative_rotation_error_threshold_rad: float = math.radians(2.0)
    relative_linear_speed_threshold_mps: float = 0.02
    relative_angular_speed_threshold_radps: float = 0.10
    unload_dwell_steps: int = 20

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "unload_dwell_steps":
                if int(value) < 1:
                    raise ValueError("DetachUnloadGateConfig.unload_dwell_steps must be positive")
            elif not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"DetachUnloadGateConfig.{name} must be finite and non-negative")


@dataclass
class DetachUnloadGateDecision(SchemaBase):
    ready_to_release: bool
    consecutive_unload_steps: int
    failure_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class DetachUnloadGate:
    """Stateful, fail-closed pre-release dwell gate for one candidate edge."""

    def __init__(self, config: DetachUnloadGateConfig | None = None) -> None:
        self.config = config or DetachUnloadGateConfig()
        self._edge_id: int | None = None
        self._consecutive_steps = 0

    def reset(self) -> None:
        self._edge_id = None
        self._consecutive_steps = 0

    def evaluate(
        self,
        *,
        estimate: DetachWrenchEstimate,
        external_contact_free: bool,
        parent_qp_feasible: bool,
        follower_qp_feasible: bool,
        relative_position_error_m: float,
        relative_rotation_error_rad: float,
        relative_linear_speed_mps: float,
        relative_angular_speed_radps: float,
    ) -> DetachUnloadGateDecision:
        if self._edge_id != estimate.edge_id:
            self._edge_id = estimate.edge_id
            self._consecutive_steps = 0
        checks = {
            "external_contact_not_free": external_contact_free is not True,
            "estimator_invalid": not estimate.valid,
            "parent_qp_infeasible": not parent_qp_feasible,
            "follower_qp_infeasible": not follower_qp_feasible,
            "cut_force_above_threshold": estimate.force_norm_n > self.config.force_threshold_n,
            "cut_torque_above_threshold": estimate.torque_norm_nm > self.config.torque_threshold_nm,
            "relative_position_error_above_threshold": (
                relative_position_error_m > self.config.relative_position_error_threshold_m
            ),
            "relative_rotation_error_above_threshold": (
                relative_rotation_error_rad > self.config.relative_rotation_error_threshold_rad
            ),
            "relative_linear_speed_above_threshold": (
                relative_linear_speed_mps > self.config.relative_linear_speed_threshold_mps
            ),
            "relative_angular_speed_above_threshold": (
                relative_angular_speed_radps > self.config.relative_angular_speed_threshold_radps
            ),
        }
        reasons = [name for name, failed in checks.items() if failed]
        finite_metrics = all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in (
                relative_position_error_m,
                relative_rotation_error_rad,
                relative_linear_speed_mps,
                relative_angular_speed_radps,
            )
        )
        if not finite_metrics:
            reasons.append("invalid_relative_state_metric")
        if reasons:
            self._consecutive_steps = 0
        else:
            self._consecutive_steps += 1
        return DetachUnloadGateDecision(
            ready_to_release=self._consecutive_steps >= self.config.unload_dwell_steps,
            consecutive_unload_steps=self._consecutive_steps,
            failure_reasons=reasons,
            metrics={
                "cut_force_norm_n": float(estimate.force_norm_n),
                "cut_torque_norm_nm": float(estimate.torque_norm_nm),
                "relative_position_error_m": float(relative_position_error_m),
                "relative_rotation_error_rad": float(relative_rotation_error_rad),
                "relative_linear_speed_mps": float(relative_linear_speed_mps),
                "relative_angular_speed_radps": float(relative_angular_speed_radps),
                "required_unload_dwell_steps": float(self.config.unload_dwell_steps),
            },
        )


def _invalid(edge_id: int, follower_ids: list[int], reason: str) -> DetachWrenchEstimate:
    return DetachWrenchEstimate(
        edge_id=max(int(edge_id), 0),
        follower_module_ids=sorted(follower_ids),
        valid=False,
        failure_reason=reason,
    )


def _component_after_cut(
    morphology_graph: MorphologyGraph,
    cut_edge: DockEdge,
    start_module_id: int,
) -> list[int]:
    adjacency = {module.module_id: set() for module in morphology_graph.modules}
    for edge in morphology_graph.dock_edges:
        if edge.edge_id == cut_edge.edge_id or edge.latch_state != "attached":
            continue
        adjacency[edge.src_module_id].add(edge.dst_module_id)
        adjacency[edge.dst_module_id].add(edge.src_module_id)
    if start_module_id not in adjacency:
        return []
    visited: set[int] = set()
    pending = [start_module_id]
    while pending:
        module_id = pending.pop()
        if module_id in visited:
            continue
        visited.add(module_id)
        pending.extend(sorted(adjacency[module_id] - visited, reverse=True))
    return sorted(visited)


def _subtree_graph(
    graph: MorphologyGraph,
    follower_ids: list[int],
    follower_module_id: int,
    cut_edge_id: int,
) -> MorphologyGraph:
    follower_set = set(follower_ids)
    modules = [
        replace(module, is_base=module.module_id == follower_module_id)
        for module in graph.modules
        if module.module_id in follower_set
    ]
    if not modules:
        raise SchemaValidationError("Follower subtree is empty")
    return MorphologyGraph(
        graph_id=f"{graph.graph_id}:cut:{cut_edge_id}:follower:{follower_module_id}",
        modules=modules,
        ports=[port for port in graph.ports if port.module_id in follower_set],
        dock_edges=[
            edge
            for edge in graph.dock_edges
            if edge.edge_id != cut_edge_id
            and edge.src_module_id in follower_set
            and edge.dst_module_id in follower_set
        ],
        robot_anchors=[anchor for anchor in graph.robot_anchors if anchor.module_id in follower_set],
        control_groups=[
            ControlGroup(
                group_id="detach_follower_subtree",
                module_ids=sorted(follower_set),
                role="detach_follower",
            )
        ],
        base_module_id=follower_module_id,
        is_closed_loop=False,
    )


def _follower_dock_pose_world(
    graph: MorphologyGraph,
    observation: RuntimeObservation,
    edge: DockEdge,
    follower_module_id: int,
) -> Pose7D:
    port_id = edge.src_port_id if follower_module_id == edge.src_module_id else edge.dst_port_id
    port = next(
        item
        for item in graph.ports
        if item.port_global_id == port_id and item.module_id == follower_module_id
    )
    module_state = next(item for item in observation.module_states if item.module_id == follower_module_id)
    return compose_pose(module_state.pose_world, port.local_pose)


def _momentum_rate_wrench_body(
    previous: RigidBodyControlModel,
    current: RigidBodyControlModel,
    dt_s: float,
) -> list[float]:
    current_rotation = _quat_to_matrix(_pose_quat(current.body_pose_world))
    previous_rotation = _quat_to_matrix(_pose_quat(previous.body_pose_world))
    current_body_from_world = _transpose(current_rotation)
    previous_body_from_world = _transpose(previous_rotation)
    linear_acceleration_world = tuple(
        (float(current.body_twist_world[index]) - float(previous.body_twist_world[index])) / dt_s
        for index in range(3)
    )
    force_body = _scale(
        _matvec(current_body_from_world, linear_acceleration_world),
        current.total_mass_kg,
    )
    omega_body = _matvec(
        current_body_from_world,
        tuple(float(value) for value in current.body_twist_world[3:6]),
    )
    previous_omega_body = _matvec(
        previous_body_from_world,
        tuple(float(value) for value in previous.body_twist_world[3:6]),
    )
    alpha_body = tuple(
        (omega_body[index] - previous_omega_body[index]) / dt_s
        for index in range(3)
    )
    inertia = _inertia_matrix(current.inertia_body)
    angular_momentum = _matvec(inertia, omega_body)
    torque_body = _add(_matvec(inertia, alpha_body), _cross(omega_body, angular_momentum))
    return [*force_body, *torque_body]


def _known_rotor_wrench_body(
    model: RigidBodyControlModel,
    command: ControllerCommand,
) -> list[float]:
    wrench = [0.0] * 6
    for rotor in model.rotor_elements:
        thrust = float(command.rotor_thrusts_n.get(rotor.global_rotor_id, 0.0))
        for index, coefficient in enumerate(rotor.allocation_column_body):
            wrench[index] += float(coefficient) * thrust
    return wrench


def _gravity_wrench_body(model: RigidBodyControlModel, gravity_mps2: float) -> list[float]:
    body_from_world = _transpose(_quat_to_matrix(_pose_quat(model.body_pose_world)))
    force_body = _matvec(
        body_from_world,
        (0.0, 0.0, -model.total_mass_kg * gravity_mps2),
    )
    return [*force_body, 0.0, 0.0, 0.0]


def transform_follower_com_wrench_to_dock_frame(
    wrench_com_body: list[float],
    com_pose_world: Pose7D,
    dock_pose_world: Pose7D,
) -> list[float]:
    """Shift a follower-CoM wrench to the dock origin and rotate into dock axes."""

    require_len(wrench_com_body, 6, "wrench_com_body")
    world_from_body = _quat_to_matrix(_pose_quat(com_pose_world))
    dock_from_world = _transpose(_quat_to_matrix(_pose_quat(dock_pose_world)))
    force_world = _matvec(world_from_body, tuple(float(value) for value in wrench_com_body[:3]))
    torque_com_world = _matvec(world_from_body, tuple(float(value) for value in wrench_com_body[3:]))
    com_to_dock_world = tuple(
        float(dock_pose_world[index]) - float(com_pose_world[index])
        for index in range(3)
    )
    torque_dock_world = _sub(torque_com_world, _cross(com_to_dock_world, force_world))
    return [
        *_matvec(dock_from_world, force_world),
        *_matvec(dock_from_world, torque_dock_world),
    ]


def _pose_quat(pose: Pose7D) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in pose[3:7])  # type: ignore[return-value]


def _quat_to_matrix(quat: tuple[float, float, float, float]):
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise SchemaValidationError("Quaternion norm must be positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _transpose(matrix):
    return tuple(tuple(matrix[row][column] for row in range(3)) for column in range(3))


def _matvec(matrix, vector):
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))


def _inertia_matrix(values: list[float]):
    ixx, ixy, ixz, iyy, iyz, izz = (float(value) for value in values)
    return ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))


def _add(left, right):
    return tuple(float(left[index]) + float(right[index]) for index in range(3))


def _sub(left, right):
    return tuple(float(left[index]) - float(right[index]) for index in range(3))


def _scale(vector, scalar: float):
    return tuple(float(value) * float(scalar) for value in vector)


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(values) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))
