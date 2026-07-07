from __future__ import annotations

import pytest

from amsrr.irg.irg_builder import IRGBuilder
from amsrr.irg.templates.base import phase_type_for_label
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType, IRGNodeType, PhaseType
from amsrr.schemas.task_spec import TaskSpec


def _base_robot_and_safety() -> dict:
    return {
        "robot_constraints": {"min_modules": 1, "max_modules": 4, "allow_closed_loop": False},
        "safety": {
            "collision_margin_m": 0.03,
            "max_contact_force_n": 30.0,
            "max_contact_torque_nm": 5.0,
            "max_tilt_rad": 1.2,
            "min_thrust_margin_ratio": 0.15,
        },
    }


def _floor_geometry() -> dict:
    return {
        "geometry_id": "floor_geom",
        "geometry_type": "box",
        "primitive_params": {"size_m": [2.0, 2.0, 0.05]},
        "asset_path": None,
        "scale": [1.0, 1.0, 1.0],
        "collision_model": "primitive",
    }


def _empty_scene() -> dict:
    return {
        "world_frame": "world",
        "geometry_library": [],
        "objects": [],
        "environment": {"support_surfaces": [], "obstacles": [], "wind": None},
    }


def _free_flight_task() -> TaskSpec:
    data = {
        "task_id": "free_flight_001",
        "task_type": "free_flight_navigation",
        "scene": _empty_scene(),
        "goals": [
            {
                "goal_id": "nav_goal",
                "goal_type": "free_flight_pose",
                "target_entity_id": None,
                "target_pose_world": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "tolerance_pos_m": 0.05,
                "tolerance_rot_rad": 0.1,
                "time_limit_s": 10.0,
            }
        ],
        **_base_robot_and_safety(),
    }
    return TaskSpec.from_dict(data)


def _valve_task() -> TaskSpec:
    data = {
        "task_id": "valve_001",
        "task_type": "valve_operation",
        "scene": {
            "world_frame": "world",
            "geometry_library": [
                {
                    "geometry_id": "valve_geom",
                    "geometry_type": "cylinder",
                    "primitive_params": {"radius_m": 0.15, "height_m": 0.04},
                    "asset_path": None,
                    "scale": [1.0, 1.0, 1.0],
                    "collision_model": "primitive",
                }
            ],
            "objects": [
                {
                    "object_id": "valve_01",
                    "geometry_id": "valve_geom",
                    "pose_world": [0.8, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                    "movable": False,
                    "mass_kg": None,
                    "inertia_kgm2": None,
                    "friction": 0.7,
                    "material_tag": "metal",
                    "contact_allowed": True,
                    "allowed_contact_modes": ["push", "stick"],
                    "semantic_tags": ["valve"],
                    "kinematic_model": {
                        "model_type": "revolute",
                        "joint_type": "revolute",
                        "axis_world": [0.0, 0.0, 1.0],
                        "origin_world": [0.8, 0.0, 0.8],
                        "q_limits": [-3.14, 3.14],
                    },
                }
            ],
            "environment": {"support_surfaces": [], "obstacles": [], "wind": None},
        },
        "goals": [
            {
                "goal_id": "turn_valve",
                "target_entity_id": "valve_01",
                "goal_type": "object_joint_state",
                "target_q": [1.57],
                "tolerance_q": [0.05],
                "time_limit_s": 20.0,
            }
        ],
        **_base_robot_and_safety(),
    }
    return TaskSpec.from_dict(data)


def _perching_task() -> TaskSpec:
    data = {
        "task_id": "perch_001",
        "task_type": "perching_manipulation",
        "scene": {
            "world_frame": "world",
            "geometry_library": [_floor_geometry()],
            "objects": [],
            "environment": {
                "support_surfaces": [
                    {
                        "surface_id": "wall_perch",
                        "geometry_id": "floor_geom",
                        "pose_world": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "friction": 0.8,
                        "contact_allowed": True,
                        "allowed_contact_modes": ["perch", "latch", "support"],
                    }
                ],
                "obstacles": [],
                "wind": None,
            },
        },
        "goals": [
            {
                "goal_id": "hold_body",
                "goal_type": "robot_pose",
                "target_pose_world": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "tolerance_pos_m": 0.1,
                "tolerance_rot_rad": 0.2,
                "time_limit_s": 15.0,
            }
        ],
        **_base_robot_and_safety(),
    }
    return TaskSpec.from_dict(data)


def _locomotion_task() -> TaskSpec:
    data = {
        "task_id": "locomotion_001",
        "task_type": "contact_mediated_locomotion",
        "scene": {
            "world_frame": "world",
            "geometry_library": [_floor_geometry()],
            "objects": [],
            "environment": {
                "support_surfaces": [
                    {
                        "surface_id": "floor",
                        "geometry_id": "floor_geom",
                        "pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        "friction": 0.9,
                        "contact_allowed": True,
                        "allowed_contact_modes": ["support"],
                    }
                ],
                "obstacles": [],
                "wind": None,
            },
        },
        "goals": [
            {
                "goal_id": "move_body",
                "goal_type": "robot_pose",
                "target_pose_world": [2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "tolerance_pos_m": 0.1,
                "tolerance_rot_rad": 0.2,
                "time_limit_s": 20.0,
            }
        ],
        **_base_robot_and_safety(),
    }
    return TaskSpec.from_dict(data)


def test_phase_label_to_phase_type_mapping() -> None:
    assert phase_type_for_label("approach_object") == PhaseType.APPROACH
    assert phase_type_for_label("establish_perch_contact") == PhaseType.ESTABLISH_CONTACT
    assert phase_type_for_label("rotate_valve") == PhaseType.APPLY_WRENCH
    with pytest.raises(SchemaValidationError):
        phase_type_for_label("approach_object_but_invalid")


def test_irg_builder_grasp_carry_valid(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)

    node_types = [node.node_type for node in irg.nodes]
    assert irg.task_id == "grasp_carry_box_001"
    assert node_types.count(IRGNodeType.TASK) == 1
    assert node_types.count(IRGNodeType.PHASE) == 7
    assert node_types.count(IRGNodeType.CONTACT_REGION) == 6
    assert node_types.count(IRGNodeType.CONTACT_SLOT) >= 1
    assert any(edge.edge_type == IRGEdgeType.ALLOWS for edge in irg.edges)
    assert any(node.ref_id == "inward_grasp_force" for node in irg.nodes)
    assert all("contact_pose_world" not in node.feature for node in irg.nodes if node.node_type == IRGNodeType.CONTACT_SLOT)


def test_irg_builder_all_task_families_smoke(grasp_carry_dict: dict) -> None:
    tasks = [
        TaskSpec.from_dict(grasp_carry_dict),
        _free_flight_task(),
        _valve_task(),
        _perching_task(),
        _locomotion_task(),
    ]
    builder = IRGBuilder()

    for task in tasks:
        irg = builder.build(task)
        assert irg.task_id == task.task_id
        assert any(node.node_type == IRGNodeType.TASK for node in irg.nodes)
        assert any(node.node_type == IRGNodeType.PHASE for node in irg.nodes)
        assert all(
            node.feature["phase_type"] in {item.value for item in PhaseType}
            for node in irg.nodes
            if node.node_type == IRGNodeType.PHASE
        )
        contact_slots = [node for node in irg.nodes if node.node_type == IRGNodeType.CONTACT_SLOT]
        if task.task_type.value != "free_flight_navigation":
            assert contact_slots
            assert any(edge.edge_type == IRGEdgeType.ALLOWS for edge in irg.edges)

