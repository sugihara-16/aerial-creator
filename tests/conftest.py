from __future__ import annotations

import pytest
import yaml


GRASP_CARRY_YAML = """
task_id: grasp_carry_box_001
task_type: object_grasp_carry
scene:
  world_frame: world
  geometry_library:
    - geometry_id: box_geom
      geometry_type: box
      primitive_params:
        size_m: [0.30, 0.20, 0.15]
      asset_path: null
      scale: [1.0, 1.0, 1.0]
      collision_model: primitive
  objects:
    - object_id: box_01
      geometry_id: box_geom
      pose_world: [0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]
      movable: true
      mass_kg: 1.0
      inertia_kgm2: null
      friction: 0.6
      material_tag: cardboard
      contact_allowed: true
      allowed_contact_modes: [grasp, support, push]
  environment:
    support_surfaces:
      - surface_id: floor
        geometry_id: floor_geom
        pose_world: [0, 0, 0, 0, 0, 0, 1]
        friction: 0.8
        contact_allowed: true
        allowed_contact_modes: [support]
    obstacles: []
    wind: null
goals:
  - goal_id: place_box
    target_entity_id: box_01
    goal_type: object_pose
    target_pose_world: [2.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]
    tolerance_pos_m: 0.05
    tolerance_rot_rad: 0.20
    time_limit_s: 30.0
robot_constraints:
  min_modules: 2
  max_modules: 6
  allow_closed_loop: false
safety:
  collision_margin_m: 0.03
  max_contact_force_n: 30.0
  min_thrust_margin_ratio: 0.15
"""


@pytest.fixture
def grasp_carry_dict() -> dict:
    return yaml.safe_load(GRASP_CARRY_YAML)

