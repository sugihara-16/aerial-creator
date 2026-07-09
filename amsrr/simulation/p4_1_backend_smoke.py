from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation


P4_1_BACKEND_SMOKE_VERSION = "p4_1_backend_smoke_v1"
P4_1_REQUIRED_REAL_SMOKES = ("p2_p3_full_scene_backend",)


@dataclass
class P4_1FullSceneBackendConfig(SchemaBase):
    config_path: str = "configs/env/isaac_lab.yaml"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    p3_config_path: str = "configs/training/p3_assembly_grasp_carry.yaml"
    control_dt_s: float = 0.005
    max_episode_steps: int = 80
    smoke_name: str = "p2_p3_full_scene_backend"
    object_id: str = "box_01"
    object_size_m: tuple[float, float, float] = (0.30, 0.20, 0.15)
    object_mass_kg: float = 1.0
    object_initial_pose_world: Pose7D = (0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
    floor_prim_path: str = "/World/defaultGroundPlane"
    require_p2_p3_design: bool = True
    require_joint_positions: bool = True
    require_joint_velocities_when_available: bool = True

    def validate(self) -> None:
        for name in (
            "config_path",
            "robot_model_config_path",
            "p3_config_path",
            "smoke_name",
            "object_id",
            "floor_prim_path",
        ):
            require_non_empty(getattr(self, name), f"P4_1FullSceneBackendConfig.{name}")
        if self.control_dt_s <= 0.0:
            raise SchemaValidationError("P4_1FullSceneBackendConfig.control_dt_s must be positive")
        if self.max_episode_steps <= 0:
            raise SchemaValidationError("P4_1FullSceneBackendConfig.max_episode_steps must be positive")
        if self.object_mass_kg <= 0.0:
            raise SchemaValidationError("P4_1FullSceneBackendConfig.object_mass_kg must be positive")
        require_len(self.object_size_m, 3, "P4_1FullSceneBackendConfig.object_size_m")
        require_len(self.object_initial_pose_world, 7, "P4_1FullSceneBackendConfig.object_initial_pose_world")


@dataclass
class P4_1RuntimeJointStateMetrics(SchemaBase):
    module_state_count: int
    modules_with_pose: int
    modules_with_twist: int
    modules_with_joint_positions: int
    modules_with_joint_velocities: int
    vectoring_joint_key_count: int
    dock_joint_key_count: int
    vectoring_joint_value_count: int
    dock_joint_value_count: int
    articulated_model_update_checked: bool = False
    articulated_model_update_passed: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failure_reasons


@dataclass
class P4_1BackendSmokeResult(SchemaBase):
    smoke_name: str
    attempted: bool
    passed: bool
    skipped: bool
    isaac_backed: bool
    backend: str = "isaac_lab"
    uses_p2_selected_design: bool = False
    uses_p3_assembled_morphology: bool = False
    full_scene_spawned: bool = False
    robot_spawned: bool = False
    object_spawned: bool = False
    floor_spawned: bool = False
    articulated_morphology: bool = False
    skip_reason: str | None = None
    runtime_observations: list[RuntimeObservation] = field(default_factory=list)
    controller_commands: list[ControllerCommand] = field(default_factory=list)
    actuator_target_records: list[dict[str, Any]] = field(default_factory=list)
    object_pose_history: list[Pose7D] = field(default_factory=list)
    joint_state_metrics: P4_1RuntimeJointStateMetrics | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.smoke_name, "P4_1BackendSmokeResult.smoke_name")
        if self.skipped and self.skip_reason is None:
            raise SchemaValidationError("P4_1BackendSmokeResult.skip_reason is required when skipped")


def evaluate_runtime_observation_joint_state(
    observations: list[RuntimeObservation],
    *,
    vectoring_joint_substrings: tuple[str, ...] = ("gimbal",),
    dock_joint_substrings: tuple[str, ...] = ("dock_mech",),
    require_joint_velocities_when_available: bool = True,
    articulated_morphology: bool = False,
    articulated_model_update_metrics: dict[str, float] | None = None,
) -> P4_1RuntimeJointStateMetrics:
    """Check that P4.1 observations preserve controller-relevant joint state."""

    module_state_count = 0
    modules_with_pose = 0
    modules_with_twist = 0
    modules_with_joint_positions = 0
    modules_with_joint_velocities = 0
    vectoring_joint_key_count = 0
    dock_joint_key_count = 0
    vectoring_joint_value_count = 0
    dock_joint_value_count = 0
    failure_reasons: list[str] = []

    if not observations:
        failure_reasons.append("P4.1 runtime observations are missing")

    for observation_index, observation in enumerate(observations):
        if not observation.module_states:
            failure_reasons.append(f"P4.1 observation {observation_index} has no module states")
        for module_state in observation.module_states:
            module_state_count += 1
            if len(module_state.pose_world) == 7:
                modules_with_pose += 1
            if len(module_state.twist_world) == 6:
                modules_with_twist += 1
            if module_state.joint_positions:
                modules_with_joint_positions += 1
            if module_state.joint_velocities:
                modules_with_joint_velocities += 1

            vectoring_keys = _matching_joint_keys(module_state.joint_positions, vectoring_joint_substrings)
            dock_keys = _matching_joint_keys(module_state.joint_positions, dock_joint_substrings)
            vectoring_joint_key_count += len(vectoring_keys)
            dock_joint_key_count += len(dock_keys)
            vectoring_joint_value_count += sum(_is_number(module_state.joint_positions[key]) for key in vectoring_keys)
            dock_joint_value_count += sum(_is_number(module_state.joint_positions[key]) for key in dock_keys)

    if module_state_count == 0:
        failure_reasons.append("P4.1 module state count is zero")
    if modules_with_pose != module_state_count:
        failure_reasons.append("P4.1 module pose fields are incomplete")
    if modules_with_twist != module_state_count:
        failure_reasons.append("P4.1 module twist fields are incomplete")
    if modules_with_joint_positions != module_state_count:
        failure_reasons.append("P4.1 module joint_positions are not populated for every module")
    if require_joint_velocities_when_available and modules_with_joint_velocities == 0:
        failure_reasons.append("P4.1 module joint_velocities are missing")
    if vectoring_joint_key_count == 0:
        failure_reasons.append("P4.1 vectoring/gimbal joint positions are missing")
    if dock_joint_key_count == 0:
        failure_reasons.append("P4.1 dock mechanism joint positions are missing")
    if vectoring_joint_value_count != vectoring_joint_key_count:
        failure_reasons.append("P4.1 vectoring/gimbal joint position values are not numeric")
    if dock_joint_value_count != dock_joint_key_count:
        failure_reasons.append("P4.1 dock mechanism joint position values are not numeric")

    update_metrics = articulated_model_update_metrics or {}
    articulated_checked = bool(articulated_morphology)
    articulated_passed = True
    if articulated_morphology:
        rotor_origin_change = float(update_metrics.get("max_model_rotor_origin_change_m", 0.0))
        allocation_change = float(update_metrics.get("max_model_allocation_change", 0.0))
        articulated_passed = rotor_origin_change > 0.0 or allocation_change > 0.0
        if not articulated_passed:
            failure_reasons.append("P4.1 articulated observation did not prove B(q) model update")

    metrics = {
        "module_state_count": float(module_state_count),
        "modules_with_pose": float(modules_with_pose),
        "modules_with_twist": float(modules_with_twist),
        "modules_with_joint_positions": float(modules_with_joint_positions),
        "modules_with_joint_velocities": float(modules_with_joint_velocities),
        "vectoring_joint_key_count": float(vectoring_joint_key_count),
        "dock_joint_key_count": float(dock_joint_key_count),
        "vectoring_joint_value_count": float(vectoring_joint_value_count),
        "dock_joint_value_count": float(dock_joint_value_count),
        "articulated_model_update_checked": 1.0 if articulated_checked else 0.0,
        "articulated_model_update_passed": 1.0 if articulated_passed else 0.0,
    }
    metrics.update({key: float(value) for key, value in update_metrics.items() if isinstance(value, (int, float))})
    return P4_1RuntimeJointStateMetrics(
        module_state_count=module_state_count,
        modules_with_pose=modules_with_pose,
        modules_with_twist=modules_with_twist,
        modules_with_joint_positions=modules_with_joint_positions,
        modules_with_joint_velocities=modules_with_joint_velocities,
        vectoring_joint_key_count=vectoring_joint_key_count,
        dock_joint_key_count=dock_joint_key_count,
        vectoring_joint_value_count=vectoring_joint_value_count,
        dock_joint_value_count=dock_joint_value_count,
        articulated_model_update_checked=articulated_checked,
        articulated_model_update_passed=articulated_passed,
        metrics=metrics,
        failure_reasons=failure_reasons,
    )


def _matching_joint_keys(values: dict[str, float], substrings: tuple[str, ...]) -> list[str]:
    return [key for key in values if any(pattern in key for pattern in substrings)]


def _is_number(value: object) -> int:
    return 1 if isinstance(value, (int, float)) else 0
