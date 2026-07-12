from __future__ import annotations

import math
from dataclasses import dataclass, field

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, Vector3, require_len, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import JointModel, LinkModel, PhysicalModel
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation


Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass
class RotorControlElement(SchemaBase):
    global_rotor_id: str
    module_id: int
    rotor_id: str
    thrust_frame_link: str
    origin_body: Vector3
    axis_body: Vector3
    thrust_min_n: float
    thrust_max_n: float
    reaction_torque_coeff_nm_per_n: float
    reaction_torque_axis_body: Vector3
    vectoring_joint_ids: list[str] = field(default_factory=list)
    virtual_x_axis_body: Vector3 | None = None
    virtual_z_axis_body: Vector3 | None = None
    allocation_column_body: list[float] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.global_rotor_id, "RotorControlElement.global_rotor_id")
        require_non_empty(self.rotor_id, "RotorControlElement.rotor_id")
        require_non_empty(self.thrust_frame_link, "RotorControlElement.thrust_frame_link")
        require_len(self.origin_body, 3, "RotorControlElement.origin_body")
        require_len(self.axis_body, 3, "RotorControlElement.axis_body")
        require_len(self.reaction_torque_axis_body, 3, "RotorControlElement.reaction_torque_axis_body")
        if self.virtual_x_axis_body is not None:
            require_len(self.virtual_x_axis_body, 3, "RotorControlElement.virtual_x_axis_body")
        if self.virtual_z_axis_body is not None:
            require_len(self.virtual_z_axis_body, 3, "RotorControlElement.virtual_z_axis_body")
        require_len(self.allocation_column_body, 6, "RotorControlElement.allocation_column_body")
        if self.thrust_min_n < 0.0 or self.thrust_max_n < self.thrust_min_n:
            raise SchemaValidationError("RotorControlElement thrust limits are invalid")


@dataclass
class RigidBodyControlModel(SchemaBase):
    model_id: str
    graph_id: str
    base_module_id: int
    body_pose_world: Pose7D
    total_mass_kg: float
    center_of_mass_body: Vector3
    inertia_body: list[float]
    rotor_elements: list[RotorControlElement]
    rotor_origins_body: dict[str, Vector3]
    rotor_axes_body: dict[str, Vector3]
    allocation_matrix_body: list[list[float]]
    vectoring_joint_axes_body: dict[str, Vector3]
    dock_actuator_ids: list[str]
    active_actuator_limits: dict[str, dict[str, float | None]]
    current_joint_positions: dict[str, float] = field(default_factory=dict)
    body_twist_world: list[float] = field(default_factory=lambda: [0.0] * 6)
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.model_id, "RigidBodyControlModel.model_id")
        require_non_empty(self.graph_id, "RigidBodyControlModel.graph_id")
        require_len(self.body_pose_world, 7, "RigidBodyControlModel.body_pose_world")
        require_len(self.center_of_mass_body, 3, "RigidBodyControlModel.center_of_mass_body")
        require_len(self.inertia_body, 6, "RigidBodyControlModel.inertia_body")
        require_len(self.body_twist_world, 6, "RigidBodyControlModel.body_twist_world")
        if self.total_mass_kg <= 0.0:
            raise SchemaValidationError("RigidBodyControlModel.total_mass_kg must be positive")
        require_len(self.allocation_matrix_body, 6, "RigidBodyControlModel.allocation_matrix_body")
        for row_idx, row in enumerate(self.allocation_matrix_body):
            require_len(row, len(self.rotor_elements), f"RigidBodyControlModel.allocation_matrix_body[{row_idx}]")


@dataclass(frozen=True)
class _Transform:
    rotation: Matrix3
    translation: Vector3


@dataclass(frozen=True)
class _LinkKinematics:
    link_id: str
    module_id: int
    mass_kg: float
    com_world: Vector3
    inertia_world_at_com: Matrix3


@dataclass(frozen=True)
class _ModuleKinematics:
    module_id: int
    link_transforms_module: dict[str, _Transform]
    link_transforms_world: dict[str, _Transform]
    joint_axes_module: dict[str, Vector3]
    joint_axes_world: dict[str, Vector3]
    link_kinematics: list[_LinkKinematics]


class RigidBodyControlModelBuilder:
    """Build a quasi-static single-rigid-body control model from current joints."""

    def build(
        self,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        runtime_observation: RuntimeObservation,
    ) -> RigidBodyControlModel:
        module_states = {state.module_id: state for state in runtime_observation.module_states}
        active_module_ids = sorted(module.module_id for module in morphology_graph.modules)
        if not active_module_ids:
            raise SchemaValidationError("RigidBodyControlModel requires at least one active module")
        missing = [module_id for module_id in active_module_ids if module_id not in module_states]
        if missing:
            raise SchemaValidationError(f"RuntimeObservation is missing module states for {missing}")
        if morphology_graph.base_module_id not in module_states:
            raise SchemaValidationError("RuntimeObservation is missing the base module state")

        base_state = module_states[morphology_graph.base_module_id]
        base_rotation_world = _quat_to_matrix(_pose_quat(base_state.pose_world))
        module_kinematics = [
            self._module_kinematics(
                module_states[module_id],
                physical_model,
            )
            for module_id in active_module_ids
        ]
        link_kinematics = [item for module in module_kinematics for item in module.link_kinematics]
        total_mass = sum(item.mass_kg for item in link_kinematics)
        if total_mass <= 0.0:
            raise SchemaValidationError("Cannot build rigid-body model with non-positive total mass")
        com_world = _scale(
            _sum_vectors(_scale(item.com_world, item.mass_kg) for item in link_kinematics),
            1.0 / total_mass,
        )
        world_to_body_rotation = _transpose(base_rotation_world)
        body_pose_world = (
            com_world[0],
            com_world[1],
            com_world[2],
            base_state.pose_world[3],
            base_state.pose_world[4],
            base_state.pose_world[5],
            base_state.pose_world[6],
        )
        base_twist_world = (list(base_state.twist_world) + [0.0] * 6)[:6]
        base_linear_velocity_world = tuple(float(value) for value in base_twist_world[:3])
        angular_velocity_world = tuple(float(value) for value in base_twist_world[3:6])
        base_origin_world = _pose_translation(base_state.pose_world)
        com_linear_velocity_world = _add(
            base_linear_velocity_world,
            _cross(angular_velocity_world, _sub(com_world, base_origin_world)),
        )
        body_twist_world = [*com_linear_velocity_world, *angular_velocity_world]
        inertia_body_matrix = _zero_matrix()
        for item in link_kinematics:
            inertia_body_at_link_com = _matmul(
                _matmul(world_to_body_rotation, item.inertia_world_at_com),
                base_rotation_world,
            )
            link_com_body = _matvec(world_to_body_rotation, _sub(item.com_world, com_world))
            inertia_body_matrix = _matadd(
                inertia_body_matrix,
                _matadd(inertia_body_at_link_com, _parallel_axis(item.mass_kg, link_com_body)),
            )

        rotor_elements = self._rotor_elements(
            physical_model=physical_model,
            module_kinematics=module_kinematics,
            com_world=com_world,
            world_to_body_rotation=world_to_body_rotation,
        )
        allocation_matrix = _columns_to_rows([rotor.allocation_column_body for rotor in rotor_elements], row_count=6)
        vectoring_axes = self._vectoring_joint_axes(
            physical_model=physical_model,
            module_kinematics=module_kinematics,
            world_to_body_rotation=world_to_body_rotation,
        )
        dock_actuator_ids = self._dock_actuator_ids(physical_model, active_module_ids)
        actuator_limits = self._active_actuator_limits(physical_model, rotor_elements, dock_actuator_ids, active_module_ids)
        current_joint_positions = _current_joint_positions(module_states, active_module_ids)
        return RigidBodyControlModel(
            model_id=f"rigid_body:{physical_model.model_id}:{morphology_graph.graph_id}",
            graph_id=morphology_graph.graph_id,
            base_module_id=morphology_graph.base_module_id,
            body_pose_world=body_pose_world,
            total_mass_kg=total_mass,
            center_of_mass_body=(0.0, 0.0, 0.0),
            inertia_body=_matrix_to_inertia6(inertia_body_matrix),
            rotor_elements=rotor_elements,
            rotor_origins_body={rotor.global_rotor_id: rotor.origin_body for rotor in rotor_elements},
            rotor_axes_body={rotor.global_rotor_id: rotor.axis_body for rotor in rotor_elements},
            allocation_matrix_body=allocation_matrix,
            vectoring_joint_axes_body=vectoring_axes,
            dock_actuator_ids=dock_actuator_ids,
            active_actuator_limits=actuator_limits,
            current_joint_positions=current_joint_positions,
            body_twist_world=body_twist_world,
            metadata={
                "active_module_count": len(active_module_ids),
                "active_link_count": len(link_kinematics),
                "active_rotor_count": len(rotor_elements),
                "body_frame_origin": "com",
                "body_frame_orientation_source": f"module:{morphology_graph.base_module_id}",
                "twist_frame_origin": "com",
                "builder_version": "p4_control_rigid_body_model_v2",
            },
        )

    def _module_kinematics(
        self,
        module_state: ModuleRuntimeState,
        physical_model: PhysicalModel,
    ) -> _ModuleKinematics:
        link_transforms_module, joint_axes_module = _link_transforms_in_module_frame(
            physical_model,
            module_state.joint_positions,
        )
        module_rotation_world = _quat_to_matrix(_pose_quat(module_state.pose_world))
        module_translation_world = _pose_translation(module_state.pose_world)
        link_transforms_world = {
            link_id: _compose(
                _Transform(rotation=module_rotation_world, translation=module_translation_world),
                transform,
            )
            for link_id, transform in link_transforms_module.items()
        }
        joint_axes_world = {
            joint_id: _normalize(_matvec(module_rotation_world, axis))
            for joint_id, axis in joint_axes_module.items()
        }
        links_by_id = {link.link_id: link for link in physical_model.links}
        link_kinematics = [
            _link_kinematics(module_state.module_id, links_by_id[link_id], transform)
            for link_id, transform in link_transforms_world.items()
            if link_id in links_by_id
        ]
        return _ModuleKinematics(
            module_id=module_state.module_id,
            link_transforms_module=link_transforms_module,
            link_transforms_world=link_transforms_world,
            joint_axes_module=joint_axes_module,
            joint_axes_world=joint_axes_world,
            link_kinematics=link_kinematics,
        )

    @staticmethod
    def _rotor_elements(
        *,
        physical_model: PhysicalModel,
        module_kinematics: list[_ModuleKinematics],
        com_world: Vector3,
        world_to_body_rotation: Matrix3,
    ) -> list[RotorControlElement]:
        elements: list[RotorControlElement] = []
        joints_by_id = {joint.joint_id: joint for joint in physical_model.joints}
        for module in sorted(module_kinematics, key=lambda item: item.module_id):
            for rotor in sorted(physical_model.rotors, key=lambda item: item.rotor_id):
                if rotor.thrust_frame_link not in module.link_transforms_world:
                    raise SchemaValidationError(
                        f"Rotor {rotor.rotor_id!r} thrust frame {rotor.thrust_frame_link!r} is missing"
                    )
                transform = module.link_transforms_world[rotor.thrust_frame_link]
                origin_body = _matvec(world_to_body_rotation, _sub(transform.translation, com_world))
                axis_world = _normalize(_matvec(transform.rotation, rotor.thrust_axis_local))
                axis_body = _normalize(_matvec(world_to_body_rotation, axis_world))
                moment_body = _cross(origin_body, axis_body)
                reaction_body = _scale(axis_body, rotor.reaction_torque_coeff_nm_per_n)
                torque_column = _add(moment_body, reaction_body)
                global_rotor_id = _global_id(module.module_id, rotor.rotor_id)
                virtual_x_axis_body = None
                virtual_z_axis_body = None
                if rotor.vectoring_joint_ids:
                    vectoring_joint = joints_by_id.get(rotor.vectoring_joint_ids[0])
                    if vectoring_joint is None:
                        raise SchemaValidationError(
                            f"Rotor {rotor.rotor_id!r} references unknown vectoring joint "
                            f"{rotor.vectoring_joint_ids[0]!r}"
                        )
                    if vectoring_joint.parent_link not in module.link_transforms_world:
                        raise SchemaValidationError(
                            f"Vectoring joint parent link {vectoring_joint.parent_link!r} is missing"
                        )
                    arm_transform = module.link_transforms_world[vectoring_joint.parent_link]
                    z_sign = 1.0 if _dot(rotor.thrust_axis_local, (0.0, 0.0, 1.0)) >= 0.0 else -1.0
                    virtual_z_axis_body = _normalize(
                        _matvec(
                            world_to_body_rotation,
                            _matvec(arm_transform.rotation, (0.0, 0.0, z_sign)),
                        )
                    )
                    vectoring_axis_body = _normalize(
                        _matvec(world_to_body_rotation, module.joint_axes_world[vectoring_joint.joint_id])
                    )
                    virtual_x_axis_body = _normalize(_cross(vectoring_axis_body, virtual_z_axis_body))
                elements.append(
                    RotorControlElement(
                        global_rotor_id=global_rotor_id,
                        module_id=module.module_id,
                        rotor_id=rotor.rotor_id,
                        thrust_frame_link=rotor.thrust_frame_link,
                        origin_body=origin_body,
                        axis_body=axis_body,
                        thrust_min_n=rotor.thrust_min_n,
                        thrust_max_n=rotor.thrust_max_n,
                        reaction_torque_coeff_nm_per_n=rotor.reaction_torque_coeff_nm_per_n,
                        reaction_torque_axis_body=axis_body,
                        vectoring_joint_ids=[_global_id(module.module_id, joint_id) for joint_id in rotor.vectoring_joint_ids],
                        virtual_x_axis_body=virtual_x_axis_body,
                        virtual_z_axis_body=virtual_z_axis_body,
                        allocation_column_body=[
                            axis_body[0],
                            axis_body[1],
                            axis_body[2],
                            torque_column[0],
                            torque_column[1],
                            torque_column[2],
                        ],
                    )
                )
        return elements

    @staticmethod
    def _vectoring_joint_axes(
        *,
        physical_model: PhysicalModel,
        module_kinematics: list[_ModuleKinematics],
        world_to_body_rotation: Matrix3,
    ) -> dict[str, Vector3]:
        vectoring_joint_ids = {
            joint_id
            for rotor in physical_model.rotors
            for joint_id in rotor.vectoring_joint_ids
        }
        axes: dict[str, Vector3] = {}
        for module in module_kinematics:
            for joint_id in sorted(vectoring_joint_ids):
                if joint_id not in module.joint_axes_world:
                    continue
                axes[_global_id(module.module_id, joint_id)] = _normalize(
                    _matvec(world_to_body_rotation, module.joint_axes_world[joint_id])
                )
        return axes

    @staticmethod
    def _dock_actuator_ids(physical_model: PhysicalModel, active_module_ids: list[int]) -> list[str]:
        mechanism_ids = sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in physical_model.dock_ports
                if port.mechanical_limits.get("mechanism_joint_id")
            }
        )
        return [
            _global_id(module_id, mechanism_id)
            for module_id in active_module_ids
            for mechanism_id in mechanism_ids
        ]

    @staticmethod
    def _active_actuator_limits(
        physical_model: PhysicalModel,
        rotor_elements: list[RotorControlElement],
        dock_actuator_ids: list[str],
        active_module_ids: list[int],
    ) -> dict[str, dict[str, float | None]]:
        limits: dict[str, dict[str, float | None]] = {}
        for rotor in rotor_elements:
            limits[rotor.global_rotor_id] = {
                "lower": rotor.thrust_min_n,
                "upper": rotor.thrust_max_n,
                "velocity": None,
                "effort": None,
            }
        joints_by_id = {joint.joint_id: joint for joint in physical_model.joints}
        for module_id in active_module_ids:
            for joint_id, joint in joints_by_id.items():
                limits[_global_id(module_id, joint_id)] = _joint_limits(joint)
        for actuator_id in dock_actuator_ids:
            limits.setdefault(actuator_id, {"lower": None, "upper": None, "velocity": None, "effort": None})
        return limits


def _link_transforms_in_module_frame(
    physical_model: PhysicalModel,
    joint_positions: dict[str, float],
) -> tuple[dict[str, _Transform], dict[str, Vector3]]:
    joints_by_parent: dict[str, list[JointModel]] = {}
    child_links = set()
    for joint in physical_model.joints:
        joints_by_parent.setdefault(joint.parent_link, []).append(joint)
        child_links.add(joint.child_link)
    link_ids = {link.link_id for link in physical_model.links}
    roots = sorted(link_ids - child_links)
    if not roots:
        raise SchemaValidationError("PhysicalModel has no root link")

    root_transforms: dict[str, _Transform] = {
        root: _Transform(rotation=_identity_matrix(), translation=(0.0, 0.0, 0.0))
        for root in roots
    }
    joint_axes_root: dict[str, Vector3] = {}
    pending = list(roots)
    while pending:
        parent = pending.pop(0)
        parent_transform = root_transforms[parent]
        for joint in sorted(joints_by_parent.get(parent, []), key=lambda item: item.joint_id):
            origin_transform = _Transform(
                rotation=_rpy_to_matrix(joint.origin_rpy),
                translation=joint.origin_xyz,
            )
            joint_frame_root = _compose(parent_transform, origin_transform)
            axis_root = _safe_normalize(_matvec(joint_frame_root.rotation, joint.axis_xyz))
            if axis_root is not None:
                joint_axes_root[joint.joint_id] = axis_root
            child_transform = _compose(joint_frame_root, _joint_motion_transform(joint, joint_positions.get(joint.joint_id, 0.0)))
            root_transforms[joint.child_link] = child_transform
            pending.append(joint.child_link)

    base_link = _module_base_link(physical_model, link_ids)
    if base_link not in root_transforms:
        raise SchemaValidationError(f"Module base link {base_link!r} is missing from link transforms")
    root_to_base = root_transforms[base_link]
    base_to_root = _inverse(root_to_base)
    module_transforms = {
        link_id: _compose(base_to_root, transform)
        for link_id, transform in root_transforms.items()
    }
    joint_axes_module = {
        joint_id: _normalize(_matvec(base_to_root.rotation, axis_root))
        for joint_id, axis_root in joint_axes_root.items()
    }
    missing_links = sorted(link_ids - set(module_transforms))
    if missing_links:
        raise SchemaValidationError(f"PhysicalModel has unreachable links: {missing_links}")
    return module_transforms, joint_axes_module


def _module_base_link(physical_model: PhysicalModel, link_ids: set[str]) -> str:
    baselink_meta = physical_model.metadata.get("baselink")
    if isinstance(baselink_meta, dict) and isinstance(baselink_meta.get("name"), str):
        name = str(baselink_meta["name"])
        if name in link_ids:
            return name
    if "fc" in link_ids:
        return "fc"
    roots = physical_model.metadata.get("root_links")
    if isinstance(roots, list) and roots and str(roots[0]) in link_ids:
        return str(roots[0])
    return sorted(link_ids)[0]


def _link_kinematics(module_id: int, link: LinkModel, transform_world: _Transform) -> _LinkKinematics:
    com_world = _add(transform_world.translation, _matvec(transform_world.rotation, link.local_com))
    inertia_local = _inertia6_to_matrix(link.inertia_kgm2)
    inertia_world = _matmul(_matmul(transform_world.rotation, inertia_local), _transpose(transform_world.rotation))
    return _LinkKinematics(
        link_id=link.link_id,
        module_id=module_id,
        mass_kg=link.mass_kg,
        com_world=com_world,
        inertia_world_at_com=inertia_world,
    )


def _joint_motion_transform(joint: JointModel, position: float) -> _Transform:
    if joint.joint_type in {"revolute", "continuous"}:
        return _Transform(rotation=_axis_angle_to_matrix(joint.axis_xyz, float(position)), translation=(0.0, 0.0, 0.0))
    if joint.joint_type == "prismatic":
        return _Transform(rotation=_identity_matrix(), translation=_scale(_normalize(joint.axis_xyz), float(position)))
    return _Transform(rotation=_identity_matrix(), translation=(0.0, 0.0, 0.0))


def _joint_limits(joint: JointModel) -> dict[str, float | None]:
    return {
        "lower": joint.limit_lower,
        "upper": joint.limit_upper,
        "velocity": joint.velocity_limit,
        "effort": joint.effort_limit,
    }


def _current_joint_positions(
    module_states: dict[int, ModuleRuntimeState],
    active_module_ids: list[int],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for module_id in active_module_ids:
        module_state = module_states[module_id]
        for joint_id, position in module_state.joint_positions.items():
            values[_global_id(module_id, joint_id)] = float(position)
    return values


def _global_id(module_id: int, local_id: str) -> str:
    return f"module_{module_id}:{local_id}"


def _pose_translation(pose: Pose7D) -> Vector3:
    return (float(pose[0]), float(pose[1]), float(pose[2]))


def _pose_quat(pose: Pose7D) -> tuple[float, float, float, float]:
    return (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))


def _quat_to_matrix(quat_xyzw: tuple[float, float, float, float]) -> Matrix3:
    x, y, z, w = quat_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise SchemaValidationError("Pose quaternion norm must be positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _rpy_to_matrix(rpy: Vector3) -> Matrix3:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _axis_angle_to_matrix(axis: Vector3, angle: float) -> Matrix3:
    x, y, z = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )


def _compose(left: _Transform, right: _Transform) -> _Transform:
    return _Transform(
        rotation=_matmul(left.rotation, right.rotation),
        translation=_add(left.translation, _matvec(left.rotation, right.translation)),
    )


def _inverse(transform: _Transform) -> _Transform:
    rotation_inv = _transpose(transform.rotation)
    return _Transform(
        rotation=rotation_inv,
        translation=_matvec(rotation_inv, _scale(transform.translation, -1.0)),
    )


def _identity_matrix() -> Matrix3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _zero_matrix() -> Matrix3:
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][idx] * right[idx][col] for idx in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _matadd(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(left[row][col] + right[row][col] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _inertia6_to_matrix(values: list[float]) -> Matrix3:
    require_len(values, 6, "inertia_kgm2")
    ixx, ixy, ixz, iyy, iyz, izz = [float(value) for value in values]
    return ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))


def _matrix_to_inertia6(matrix: Matrix3) -> list[float]:
    return [
        float(matrix[0][0]),
        float(matrix[0][1]),
        float(matrix[0][2]),
        float(matrix[1][1]),
        float(matrix[1][2]),
        float(matrix[2][2]),
    ]


def _parallel_axis(mass: float, offset_body: Vector3) -> Matrix3:
    x, y, z = offset_body
    d2 = x * x + y * y + z * z
    return (
        (mass * (d2 - x * x), -mass * x * y, -mass * x * z),
        (-mass * y * x, mass * (d2 - y * y), -mass * y * z),
        (-mass * z * x, -mass * z * y, mass * (d2 - z * z)),
    )


def _columns_to_rows(columns: list[list[float]], *, row_count: int) -> list[list[float]]:
    return [[float(column[row]) for column in columns] for row in range(row_count)]


def _sum_vectors(vectors) -> Vector3:
    x = y = z = 0.0
    for vector in vectors:
        x += vector[0]
        y += vector[1]
        z += vector[2]
    return (x, y, z)


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _normalize(vector: Vector3) -> Vector3:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 0.0:
        raise SchemaValidationError(f"Cannot normalize zero vector {vector!r}")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _safe_normalize(vector: Vector3) -> Vector3 | None:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 0.0:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)
