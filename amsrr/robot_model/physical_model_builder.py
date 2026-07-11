from __future__ import annotations

from pathlib import Path
from typing import Any

from amsrr.robot_model.joint_actuator_model import JointActuatorModel, JointActuatorSpec, load_joint_actuator_model
from amsrr.robot_model.thrust_model import ThrustModel, ThrustModelEntry, load_thrust_model, normalize_rotor_id
from amsrr.robot_model.urdf_loader import URDFJoint, URDFModel, load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_in_module_frame
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.physical_model import (
    CollisionPrimitive,
    DockPortSpec,
    JointModel,
    LinkModel,
    ModuleCapabilityToken,
    PhysicalModel,
    RotorModel,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file


def _first_or_none(items: list[str]) -> str | None:
    return items[0] if items else None


def _link_by_name(urdf_model: URDFModel) -> dict[str, Any]:
    return {link.name: link for link in urdf_model.links}


def _joint_by_child(urdf_model: URDFModel) -> dict[str, URDFJoint]:
    return {joint.child_link: joint for joint in urdf_model.joints}


def _joint_by_name(urdf_model: URDFModel) -> dict[str, URDFJoint]:
    return {joint.name: joint for joint in urdf_model.joints}


def _ancestor_joints(urdf_model: URDFModel, link_name: str) -> list[URDFJoint]:
    joints_by_child = _joint_by_child(urdf_model)
    ancestors: list[URDFJoint] = []
    current = link_name
    while current in joints_by_child:
        joint = joints_by_child[current]
        ancestors.append(joint)
        current = joint.parent_link
    return ancestors


def _find_thrust_link_name(urdf_model: URDFModel, rotor_id: str) -> str:
    link_names = {link.name for link in urdf_model.links}
    if rotor_id in link_names:
        return rotor_id
    normalized = normalize_rotor_id(rotor_id)
    matches = [name for name in link_names if normalize_rotor_id(name) == normalized]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SchemaValidationError(f"Thrust rotor_id {rotor_id!r} does not match any URDF link")
    raise SchemaValidationError(f"Thrust rotor_id {rotor_id!r} matched multiple URDF links: {matches}")


def _rotor_joint_for_link(urdf_model: URDFModel, link_name: str) -> URDFJoint | None:
    joint = _joint_by_child(urdf_model).get(link_name)
    if joint and joint.joint_type == "continuous":
        return joint
    return joint


def _urdf_m_f_rate(urdf_model: URDFModel) -> float:
    metadata = urdf_model.metadata.get("m_f_rate")
    if not isinstance(metadata, dict):
        return 0.0
    try:
        return float(metadata.get("value", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _reaction_torque_coeff(entry: ThrustModelEntry, rotor_joint: URDFJoint | None, m_f_rate: float) -> float:
    if entry.reaction_torque_coeff_nm_per_n != 0.0:
        return float(entry.reaction_torque_coeff_nm_per_n)
    if rotor_joint is None:
        return 0.0
    return float(rotor_joint.axis_xyz[2]) * m_f_rate


def _build_rotor_models(urdf_model: URDFModel, thrust_model: ThrustModel) -> list[RotorModel]:
    rotors: list[RotorModel] = []
    m_f_rate = _urdf_m_f_rate(urdf_model)
    for entry in thrust_model.rotors:
        thrust_link = _find_thrust_link_name(urdf_model, entry.rotor_id)
        rotor_joint = _rotor_joint_for_link(urdf_model, thrust_link)
        vectoring_joint_ids = [
            joint.name for joint in _ancestor_joints(urdf_model, thrust_link) if "gimbal" in joint.name
        ]
        rotors.append(
            RotorModel(
                rotor_id=entry.rotor_id,
                thrust_frame_link=thrust_link,
                thrust_axis_local=(0.0, 0.0, 1.0),
                thrust_min_n=entry.thrust_min_n,
                thrust_max_n=entry.thrust_max_n,
                reaction_torque_coeff_nm_per_n=_reaction_torque_coeff(entry, rotor_joint, m_f_rate),
                vectoring_joint_ids=vectoring_joint_ids,
            )
        )
    return rotors


def _port_type_from_name(name: str) -> str:
    if "pitch" in name:
        return "pitch_dock"
    if "yaw" in name:
        return "yaw_dock"
    return "generic_dock"


def _compatible_port_types(port_type: str) -> list[str]:
    if port_type == "pitch_dock":
        return ["yaw_dock"]
    if port_type == "yaw_dock":
        return ["pitch_dock"]
    return ["generic_dock", "pitch_dock", "yaw_dock"]


def _mechanism_joint_for_port(urdf_model: URDFModel, parent_link: str) -> URDFJoint | None:
    for joint in urdf_model.joints:
        if joint.child_link == parent_link and "dock_mech_joint" in joint.name:
            return joint
    return None


def _build_dock_ports(
    urdf_model: URDFModel,
    joint_actuator_model: JointActuatorModel | None = None,
) -> list[DockPortSpec]:
    ports: list[DockPortSpec] = []
    joints_by_name = _joint_by_name(urdf_model)
    link_poses_module = link_poses_in_module_frame(urdf_model)
    for joint_name in urdf_model.candidate_connect_joints:
        joint = joints_by_name[joint_name]
        port_type = _port_type_from_name(joint_name + "_" + joint.parent_link)
        mechanism_joint = _mechanism_joint_for_port(urdf_model, joint.parent_link)
        mechanical_limits: dict[str, Any] = {}
        if mechanism_joint is not None:
            mechanical_limits = {
                "mechanism_joint_id": mechanism_joint.name,
                "limit_lower": mechanism_joint.limit_lower,
                "limit_upper": mechanism_joint.limit_upper,
                "effort_limit": mechanism_joint.effort_limit,
                "velocity_limit": mechanism_joint.velocity_limit,
            }
            actuator_spec = (
                joint_actuator_model.spec_for_joint(mechanism_joint.name)
                if joint_actuator_model is not None
                else None
            )
            if actuator_spec is not None:
                mechanical_limits.update(_actuator_limit_metadata(actuator_spec))
        ports.append(
            DockPortSpec(
                port_id=joint.name,
                parent_link=joint.parent_link,
                local_pose=link_poses_module[joint.child_link],
                port_type=port_type,  # type: ignore[arg-type]
                compatible_port_types=_compatible_port_types(port_type),
                latch_axis_local=joint.axis_xyz,
                mechanical_limits=mechanical_limits,
            )
        )
    return sorted(ports, key=lambda port: port.port_id)


def _build_collision_primitives(urdf_model: URDFModel) -> list[CollisionPrimitive]:
    primitives: list[CollisionPrimitive] = []
    for link in urdf_model.links:
        for idx, mesh_ref in enumerate(link.collision_mesh_refs):
            primitives.append(
                CollisionPrimitive(
                    primitive_id=f"{link.name}:collision:{idx}",
                    link_id=link.name,
                    primitive_type="mesh",
                    geometry_ref=mesh_ref,
                )
            )
    return primitives


def _aggregate_inertia(urdf_model: URDFModel) -> list[float]:
    aggregate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for link in urdf_model.links:
        for idx, value in enumerate(link.inertia_kgm2):
            aggregate[idx] += value
    return aggregate


def build_physical_model(
    urdf_path: str | Path,
    thrust_model_path: str | Path,
    *,
    model_id: str | None = None,
    module_type: str = "holon",
    rotor_link_patterns: list[str] | None = None,
    rotor_joint_patterns: list[str] | None = None,
    dock_link_patterns: list[str] | None = None,
    dock_joint_patterns: list[str] | None = None,
    joint_actuator_model_path: str | Path | None = None,
) -> PhysicalModel:
    urdf_model = load_urdf(
        urdf_path,
        rotor_link_patterns=rotor_link_patterns,
        rotor_joint_patterns=rotor_joint_patterns,
        dock_link_patterns=dock_link_patterns,
        dock_joint_patterns=dock_joint_patterns,
    )
    thrust_model = load_thrust_model(thrust_model_path)
    joint_actuator_model = (
        load_joint_actuator_model(joint_actuator_model_path)
        if joint_actuator_model_path is not None
        else None
    )
    if joint_actuator_model is not None:
        _validate_joint_actuator_limits(urdf_model, joint_actuator_model)
    links = [
        LinkModel(
            link_id=link.name,
            parent_joint_id=urdf_model.child_to_joint.get(link.name),
            mass_kg=link.mass_kg,
            inertia_kgm2=link.inertia_kgm2,
            local_com=link.local_com,
            visual_geometry_ref=_first_or_none(link.visual_mesh_refs),
            collision_geometry_ref=_first_or_none(link.collision_mesh_refs),
        )
        for link in urdf_model.links
    ]
    joints = [
        JointModel(
            joint_id=joint.name,
            joint_type=joint.joint_type,  # type: ignore[arg-type]
            parent_link=joint.parent_link,
            child_link=joint.child_link,
            origin_xyz=joint.origin_xyz,
            origin_rpy=joint.origin_rpy,
            axis_xyz=joint.axis_xyz,
            limit_lower=joint.limit_lower,
            limit_upper=joint.limit_upper,
            effort_limit=joint.effort_limit,
            velocity_limit=joint.velocity_limit,
        )
        for joint in urdf_model.joints
    ]
    aggregate_mass = sum(link.mass_kg for link in links)
    metadata = {
        **urdf_model.metadata,
        "module_type": module_type,
        "urdf_hash": hash_file(urdf_path),
        "thrust_model_hash": hash_file(thrust_model_path),
        "frame_tree_valid": urdf_model.frame_tree_valid,
        "frame_tree_errors": urdf_model.frame_tree_errors,
        "root_links": urdf_model.root_links,
        "candidate_rotor_links": urdf_model.candidate_rotor_links,
        "candidate_rotor_joints": urdf_model.candidate_rotor_joints,
        "candidate_dock_links": urdf_model.candidate_dock_links,
        "candidate_dock_joints": urdf_model.candidate_dock_joints,
    }
    if joint_actuator_model is not None and joint_actuator_model_path is not None:
        assignments = {
            joint.name: spec.role
            for joint in urdf_model.joints
            if (spec := joint_actuator_model.spec_for_joint(joint.name)) is not None
        }
        metadata.update(
            {
                "joint_actuator_model_path": str(joint_actuator_model_path),
                "joint_actuator_model_hash": hash_file(joint_actuator_model_path),
                "joint_actuator_model_version": joint_actuator_model.version,
                "joint_actuator_assignments": assignments,
                "joint_actuator_specs": {
                    role: spec.to_dict()
                    for role, spec in sorted(joint_actuator_model.actuator_roles.items())
                },
            }
        )
    return PhysicalModel(
        model_id=model_id or urdf_model.robot_name,
        urdf_path=str(urdf_path),
        links=links,
        joints=joints,
        rotors=_build_rotor_models(urdf_model, thrust_model),
        dock_ports=_build_dock_ports(urdf_model, joint_actuator_model),
        collision_primitives=_build_collision_primitives(urdf_model),
        aggregate_mass_kg=aggregate_mass,
        aggregate_inertia_body=_aggregate_inertia(urdf_model),
        metadata=metadata,
    )


def _resolve_project_path(path_value: str, *, config_path: Path, project_root: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = []
    if project_root is not None:
        candidates.append(project_root / path)
    candidates.append(Path.cwd() / path)
    candidates.append(config_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_physical_model_from_config(
    robot_model_config_path: str | Path = "configs/robot/robot_model.yaml",
    *,
    project_root: str | Path | None = None,
    urdf_path_override: str | Path | None = None,
    thrust_model_path_override: str | Path | None = None,
    joint_actuator_model_path_override: str | Path | None = None,
) -> PhysicalModel:
    config_path = Path(robot_model_config_path)
    root = Path(project_root) if project_root is not None else config_path.resolve().parents[2]
    data = load_config(config_path)
    robot_model_config = data.get("robot_model", data)
    urdf_path = Path(urdf_path_override) if urdf_path_override is not None else _resolve_project_path(
        robot_model_config["module_urdf_path"], config_path=config_path, project_root=root
    )
    thrust_model_path = (
        Path(thrust_model_path_override)
        if thrust_model_path_override is not None
        else _resolve_project_path(robot_model_config["thrust_model_path"], config_path=config_path, project_root=root)
    )
    joint_actuator_model_value = robot_model_config.get("joint_actuator_model_path")
    joint_actuator_model_path = (
        Path(joint_actuator_model_path_override)
        if joint_actuator_model_path_override is not None
        else (
            _resolve_project_path(joint_actuator_model_value, config_path=config_path, project_root=root)
            if isinstance(joint_actuator_model_value, str) and joint_actuator_model_value
            else None
        )
    )

    rotor_detection = robot_model_config.get("rotor_detection", {})
    rotor_patterns = rotor_detection.get("patterns", {})
    dock_detection = robot_model_config.get("dock_port_detection", {})
    dock_patterns = dock_detection.get("patterns", {})
    return build_physical_model(
        urdf_path,
        thrust_model_path,
        model_id=robot_model_config.get("module_type", "holon"),
        module_type=robot_model_config.get("module_type", "holon"),
        rotor_link_patterns=[rotor_patterns.get("thrust", "thrust_")],
        rotor_joint_patterns=[rotor_patterns.get("rotor_joint", "rotor")],
        dock_link_patterns=[dock_patterns.get("pitch", "pitch_dock"), dock_patterns.get("yaw", "yaw_dock")],
        dock_joint_patterns=["dock_mech_joint", "connect_point"],
        joint_actuator_model_path=joint_actuator_model_path,
    )


def _actuator_limit_metadata(spec: JointActuatorSpec) -> dict[str, Any]:
    return {
        "actuator_role": spec.role,
        "actuator_manufacturer": spec.manufacturer,
        "actuator_model": spec.model,
        "continuous_torque_limit_nm": spec.continuous_torque_limit_nm,
        "peak_torque_limit_nm": spec.peak_torque_nm,
        "no_load_speed_rad_s": spec.no_load_speed_rad_s,
        "simulation_safe_velocity_limit_rad_s": spec.simulation_drive.safe_velocity_limit_rad_s,
    }


def _validate_joint_actuator_limits(urdf_model: URDFModel, model: JointActuatorModel) -> None:
    matched_roles: set[str] = set()
    for joint in urdf_model.joints:
        spec = model.spec_for_joint(joint.name)
        if spec is None:
            continue
        matched_roles.add(spec.role)
        if joint.effort_limit is None or abs(float(joint.effort_limit) - spec.peak_torque_nm) > 1.0e-6:
            raise SchemaValidationError(
                f"URDF joint {joint.name!r} effort limit must equal {spec.role} peak torque "
                f"{spec.peak_torque_nm} Nm"
            )
        if joint.velocity_limit is None or abs(float(joint.velocity_limit) - spec.no_load_speed_rad_s) > 1.0e-6:
            raise SchemaValidationError(
                f"URDF joint {joint.name!r} velocity limit must equal {spec.role} no-load speed "
                f"{spec.no_load_speed_rad_s} rad/s"
            )
    missing_roles = sorted(set(model.actuator_roles) - matched_roles)
    if missing_roles:
        raise SchemaValidationError(f"Joint actuator roles did not match any URDF joint: {missing_roles}")


def build_module_capability_token(physical_model: PhysicalModel, *, module_type: str = "holon") -> ModuleCapabilityToken:
    thrust_min = [rotor.thrust_min_n for rotor in physical_model.rotors]
    thrust_max = [rotor.thrust_max_n for rotor in physical_model.rotors]
    weight = max(physical_model.aggregate_mass_kg * 9.80665, 1.0e-9)
    port_type_counts = [
        sum(1 for port in physical_model.dock_ports if port.port_type == "pitch_dock"),
        sum(1 for port in physical_model.dock_ports if port.port_type == "yaw_dock"),
        sum(1 for port in physical_model.dock_ports if port.port_type == "generic_dock"),
    ]
    return ModuleCapabilityToken(
        module_type=module_type,
        aggregate_mass_norm=physical_model.aggregate_mass_kg,
        aggregate_inertia_features=physical_model.aggregate_inertia_body,
        rotor_count=len(physical_model.rotors),
        port_count=len(physical_model.dock_ports),
        thrust_min_features=thrust_min,
        thrust_max_features=thrust_max,
        thrust_to_weight_ratio_est=sum(thrust_max) / weight,
        dock_port_type_counts=port_type_counts,
        has_vectoring=any(rotor.vectoring_joint_ids for rotor in physical_model.rotors),
        has_dock_mechanism=bool(physical_model.dock_ports),
    )
