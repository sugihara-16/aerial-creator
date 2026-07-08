from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy, P2DesignPolicyConfig, P2DesignSelection
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.p2_design_distribution import P2DesignTaskSample, P2GraspCarryDesignDistribution
from amsrr.training.p2_design_runner import load_p2_design_runner_config


DEFAULT_GRASP_CARRY_TASK_DICT = {
    "task_id": "grasp_carry_box_001",
    "task_type": "object_grasp_carry",
    "scene": {
        "world_frame": "world",
        "geometry_library": [
            {
                "geometry_id": "box_geom",
                "geometry_type": "box",
                "primitive_params": {"size_m": [0.30, 0.20, 0.15]},
                "asset_path": None,
                "scale": [1.0, 1.0, 1.0],
                "collision_model": "primitive",
            }
        ],
        "objects": [
            {
                "object_id": "box_01",
                "geometry_id": "box_geom",
                "pose_world": [0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
                "movable": True,
                "mass_kg": 1.0,
                "inertia_kgm2": None,
                "friction": 0.6,
                "material_tag": "cardboard",
                "contact_allowed": True,
                "allowed_contact_modes": ["grasp", "support", "push"],
            }
        ],
        "environment": {
            "support_surfaces": [
                {
                    "surface_id": "floor",
                    "geometry_id": "floor_geom",
                    "pose_world": [0, 0, 0, 0, 0, 0, 1],
                    "friction": 0.8,
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
            "goal_id": "place_box",
            "target_entity_id": "box_01",
            "goal_type": "object_pose",
            "target_pose_world": [2.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
            "tolerance_pos_m": 0.05,
            "tolerance_rot_rad": 0.20,
            "time_limit_s": 30.0,
        }
    ],
    "robot_constraints": {
        "min_modules": 2,
        "max_modules": 6,
        "allow_closed_loop": False,
    },
    "safety": {
        "collision_margin_m": 0.03,
        "max_contact_force_n": 30.0,
        "min_thrust_margin_ratio": 0.15,
    },
}


@dataclass(frozen=True)
class P2InspectionContext:
    sample: P2DesignTaskSample
    task_spec: TaskSpec
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    physical_model: PhysicalModel
    design_context: DesignPolicyContext
    selection: P2DesignSelection
    policy_config: P2DesignPolicyConfig


def default_grasp_carry_task_spec() -> TaskSpec:
    return TaskSpec.from_dict(DEFAULT_GRASP_CARRY_TASK_DICT)


def build_p2_inspection_context(
    *,
    config_path: str | Path = "configs/training/p2_design_grasp_carry.yaml",
    seed: int = 0,
    sample_index: int = 0,
    base_task_spec: TaskSpec | None = None,
) -> P2InspectionContext:
    runner_config, distribution_config, policy_config = load_p2_design_runner_config(config_path)
    task_spec = base_task_spec or default_grasp_carry_task_spec()
    sample = P2GraspCarryDesignDistribution(task_spec, distribution_config).sample(
        seed=seed,
        sample_index=sample_index,
    )
    physical_model = build_physical_model_from_config(runner_config.robot_model_config_path)
    irg = IRGBuilder().build(sample.task_spec)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    design_context = DesignPolicyContext(
        task_spec=sample.task_spec,
        irg=irg,
        physical_model=physical_model,
        interaction_envelope=envelope,
    )
    selection = P2DesignPolicy(config=policy_config).evaluate_candidates(design_context)
    return P2InspectionContext(
        sample=sample,
        task_spec=sample.task_spec,
        irg=irg,
        interaction_envelope=envelope,
        physical_model=physical_model,
        design_context=design_context,
        selection=selection,
        policy_config=policy_config,
    )
