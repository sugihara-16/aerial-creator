from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import (
    ContactMode,
    Pose7D,
    SchemaBase,
    SchemaValidationError,
    StrEnum,
    Vector3,
    require_len,
    require_non_empty,
    require_non_negative,
    require_positive,
)


class TaskType(StrEnum):
    FREE_FLIGHT_NAVIGATION = "free_flight_navigation"
    OBJECT_GRASP_CARRY = "object_grasp_carry"
    VALVE_OPERATION = "valve_operation"
    PERCHING_MANIPULATION = "perching_manipulation"
    CONTACT_MEDIATED_LOCOMOTION = "contact_mediated_locomotion"


class GeometryType(StrEnum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    MESH = "mesh"
    SDF = "sdf"
    POINT_CLOUD = "point_cloud"


class CollisionModel(StrEnum):
    PRIMITIVE = "primitive"
    CONVEX = "convex"
    MESH = "mesh"
    SDF = "sdf"


@dataclass
class GeometrySpec(SchemaBase):
    geometry_id: str
    geometry_type: GeometryType
    primitive_params: dict[str, Any] | None
    asset_path: str | None
    collision_model: CollisionModel
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    units: Literal["m"] = "m"

    def validate(self) -> None:
        require_non_empty(self.geometry_id, "GeometrySpec.geometry_id")
        require_len(self.scale, 3, "GeometrySpec.scale")
        for idx, item in enumerate(self.scale):
            require_positive(item, f"GeometrySpec.scale[{idx}]")
        if self.geometry_type in {GeometryType.BOX, GeometryType.SPHERE, GeometryType.CYLINDER, GeometryType.CAPSULE}:
            if self.primitive_params is None:
                raise SchemaValidationError("primitive GeometrySpec requires primitive_params")
        if self.geometry_type in {GeometryType.MESH, GeometryType.SDF, GeometryType.POINT_CLOUD}:
            if not self.asset_path:
                raise SchemaValidationError(f"{self.geometry_type.value} GeometrySpec requires asset_path")


@dataclass
class ObjectKinematicModel(SchemaBase):
    model_type: str
    joint_type: str | None = None
    axis_world: Vector3 | None = None
    origin_world: Vector3 | None = None
    q_limits: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.model_type, "ObjectKinematicModel.model_type")


@dataclass
class ObjectSpec(SchemaBase):
    object_id: str
    geometry_id: str
    pose_world: Pose7D
    movable: bool
    mass_kg: float | None
    inertia_kgm2: list[float] | None
    friction: float | None
    material_tag: str | None
    allowed_contact_modes: list[ContactMode]
    contact_allowed: bool = True
    semantic_tags: list[str] = field(default_factory=list)
    kinematic_model: ObjectKinematicModel | None = None
    center_of_mass_object: Vector3 | None = None
    density_kg_m3: float | None = None

    def validate(self) -> None:
        require_non_empty(self.object_id, "ObjectSpec.object_id")
        require_non_empty(self.geometry_id, "ObjectSpec.geometry_id")
        require_len(self.pose_world, 7, "ObjectSpec.pose_world")
        if self.mass_kg is not None:
            require_non_negative(self.mass_kg, "ObjectSpec.mass_kg")
        if self.inertia_kgm2 is not None:
            require_len(self.inertia_kgm2, 6, "ObjectSpec.inertia_kgm2")
        if self.friction is not None:
            require_non_negative(self.friction, "ObjectSpec.friction")
        if self.center_of_mass_object is not None:
            require_len(
                self.center_of_mass_object,
                3,
                "ObjectSpec.center_of_mass_object",
            )
        if self.density_kg_m3 is not None:
            require_positive(self.density_kg_m3, "ObjectSpec.density_kg_m3")


@dataclass
class SurfaceSpec(SchemaBase):
    surface_id: str
    geometry_id: str
    pose_world: Pose7D
    allowed_contact_modes: list[ContactMode]
    friction: float | None = None
    contact_allowed: bool = True
    semantic_tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.surface_id, "SurfaceSpec.surface_id")
        require_non_empty(self.geometry_id, "SurfaceSpec.geometry_id")
        require_len(self.pose_world, 7, "SurfaceSpec.pose_world")
        if self.friction is not None:
            require_non_negative(self.friction, "SurfaceSpec.friction")


@dataclass
class ObstacleSpec(SchemaBase):
    obstacle_id: str
    geometry_id: str
    pose_world: Pose7D
    collision_margin_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.obstacle_id, "ObstacleSpec.obstacle_id")
        require_non_empty(self.geometry_id, "ObstacleSpec.geometry_id")
        require_len(self.pose_world, 7, "ObstacleSpec.pose_world")
        if self.collision_margin_m is not None:
            require_non_negative(self.collision_margin_m, "ObstacleSpec.collision_margin_m")


@dataclass
class WindSpec(SchemaBase):
    velocity_world: Vector3 = (0.0, 0.0, 0.0)
    gust_std: float = 0.0

    def validate(self) -> None:
        require_len(self.velocity_world, 3, "WindSpec.velocity_world")
        require_non_negative(self.gust_std, "WindSpec.gust_std")


@dataclass
class EnvironmentSpec(SchemaBase):
    support_surfaces: list[SurfaceSpec]
    obstacles: list[ObstacleSpec]
    wind: WindSpec | None = None
    gravity: Vector3 = (0.0, 0.0, -9.80665)

    def validate(self) -> None:
        require_len(self.gravity, 3, "EnvironmentSpec.gravity")


@dataclass
class SceneSpec(SchemaBase):
    geometry_library: list[GeometrySpec]
    objects: list[ObjectSpec]
    environment: EnvironmentSpec
    world_frame: str = "world"

    def validate(self) -> None:
        require_non_empty(self.world_frame, "SceneSpec.world_frame")
        geometry_ids = [item.geometry_id for item in self.geometry_library]
        if len(geometry_ids) != len(set(geometry_ids)):
            raise SchemaValidationError("SceneSpec.geometry_library has duplicate geometry_id values")
        known = set(geometry_ids)
        for obj in self.objects:
            if obj.geometry_id not in known:
                raise SchemaValidationError(f"ObjectSpec {obj.object_id!r} references missing geometry_id {obj.geometry_id!r}")


@dataclass
class GoalSpec(SchemaBase):
    goal_id: str
    goal_type: Literal[
        "robot_pose",
        "object_pose",
        "object_displacement",
        "object_joint_state",
        "contact_state",
        "centroidal_state",
        "free_flight_pose",
    ]
    time_limit_s: float
    target_entity_id: str | None = None
    target_pose_world: Pose7D | None = None
    target_twist_world: list[float] | None = None
    target_q: list[float] | None = None
    tolerance_pos_m: float | None = None
    tolerance_rot_rad: float | None = None
    tolerance_q: list[float] | None = None

    def validate(self) -> None:
        require_non_empty(self.goal_id, "GoalSpec.goal_id")
        require_positive(self.time_limit_s, "GoalSpec.time_limit_s")
        if self.target_pose_world is not None:
            require_len(self.target_pose_world, 7, "GoalSpec.target_pose_world")
        if self.target_twist_world is not None:
            require_len(self.target_twist_world, 6, "GoalSpec.target_twist_world")
        if self.tolerance_pos_m is not None:
            require_non_negative(self.tolerance_pos_m, "GoalSpec.tolerance_pos_m")
        if self.tolerance_rot_rad is not None:
            require_non_negative(self.tolerance_rot_rad, "GoalSpec.tolerance_rot_rad")


@dataclass
class RobotConstraints(SchemaBase):
    min_modules: int = 1
    max_modules: int = 8
    allowed_module_types: list[str] = field(default_factory=lambda: ["holon"])
    allow_closed_loop: bool = False
    max_docked_edges: int | None = None
    max_robot_anchors: int = 16

    def validate(self) -> None:
        if self.min_modules < 1:
            raise SchemaValidationError("RobotConstraints.min_modules must be >= 1")
        if self.max_modules < self.min_modules:
            raise SchemaValidationError("RobotConstraints.max_modules must be >= min_modules")
        if self.max_docked_edges is not None and self.max_docked_edges < 0:
            raise SchemaValidationError("RobotConstraints.max_docked_edges must be non-negative")
        if self.max_robot_anchors < 1:
            raise SchemaValidationError("RobotConstraints.max_robot_anchors must be >= 1")


@dataclass
class SafetySpec(SchemaBase):
    collision_margin_m: float = 0.03
    max_contact_force_n: float = 30.0
    max_contact_torque_nm: float = 5.0
    max_tilt_rad: float = 1.2
    min_thrust_margin_ratio: float = 0.15
    min_qp_margin: float = 0.0
    allow_object_drop: bool = False

    def validate(self) -> None:
        require_non_negative(self.collision_margin_m, "SafetySpec.collision_margin_m")
        require_positive(self.max_contact_force_n, "SafetySpec.max_contact_force_n")
        require_positive(self.max_contact_torque_nm, "SafetySpec.max_contact_torque_nm")
        require_positive(self.max_tilt_rad, "SafetySpec.max_tilt_rad")
        require_non_negative(self.min_thrust_margin_ratio, "SafetySpec.min_thrust_margin_ratio")


@dataclass
class TaskSpec(SchemaBase):
    task_id: str
    task_type: TaskType
    scene: SceneSpec
    goals: list[GoalSpec]
    robot_constraints: RobotConstraints
    safety: SafetySpec
    curriculum_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.task_id, "TaskSpec.task_id")
        if not self.goals:
            raise SchemaValidationError("TaskSpec.goals must not be empty")
        if self.task_type == TaskType.OBJECT_GRASP_CARRY:
            object_by_id = {obj.object_id: obj for obj in self.scene.objects}
            target_ids = [goal.target_entity_id for goal in self.goals if goal.goal_type == "object_pose" and goal.target_entity_id]
            targets = [object_by_id[target_id] for target_id in target_ids if target_id in object_by_id]
            if not targets:
                targets = [obj for obj in self.scene.objects if obj.movable]
            for obj in targets:
                if obj.movable and obj.mass_kg is None:
                    raise SchemaValidationError("object_grasp_carry target movable object requires mass_kg")
