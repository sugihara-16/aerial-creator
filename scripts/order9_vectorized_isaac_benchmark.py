#!/usr/bin/env python3
from __future__ import annotations

"""Real-Isaac and tensorized pi_L throughput probe for the Order 9 scene.

This probe measures the low-logging vector physics substrate together with one
full tensorized Order 9 pi_L actor-critic inference per control step.  It uses
the cached three-module Order 8 morphology asset, one free 1 kg box, and a
support collider in every cloned environment.  Its randomly initialized policy
is a throughput workload only: it does not claim learned-policy quality or
replace the unchanged full-mesh Order 8 acceptance rollout.
"""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--measurement-steps", type=int, default=2048)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--env-spacing", type=float, default=2.5)
    parser.add_argument(
        "--robot-usd",
        default=(
            "artifacts/isaac/robots/holon/holon_p4_2_graph/"
            "holon_p4_2_graph.usda"
        ),
    )
    parser.add_argument("--phase-reset-interval", type=int, default=128)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
if args_cli.num_envs < 1:
    raise ValueError("--num-envs must be positive")
if args_cli.warmup_steps < 0 or args_cli.measurement_steps < 1:
    raise ValueError("benchmark step counts are invalid")
if args_cli.dt <= 0.0 or args_cli.phase_reset_interval < 1:
    raise ValueError("benchmark dt/reset interval must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from amsrr.policies.order9_low_level_policy import (
    ORDER9_PI_L_POLICY_VERSION,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.order3 import ORDER3_ACTION_SIZE
from amsrr.schemas.task_spec import TaskType
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.training.order9_tensor_runtime import (
    ORDER9_TENSOR_RUNTIME_VERSION,
    Order9CentroidalTensorObservation,
    Order9TensorizedTopologyBucket,
    order9_low_level_actor_features_from_tensors,
)
from amsrr.utils.hashing import stable_hash


ROBOT_USD = str(Path(args_cli.robot_usd).resolve())
if not Path(ROBOT_USD).is_file():
    raise FileNotFoundError(f"cached Order 8 morphology USD is missing: {ROBOT_USD}")


@configclass
class Order9BenchmarkSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "gimbal_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*gimbal.*"], stiffness=40.0, damping=2.0
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"], stiffness=200.0, damping=5.0
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"], stiffness=0.0, damping=0.0
            ),
        },
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.40, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.38, 0.12)
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.225)),
    )
    support: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.0, 0.30),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0)),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        ),
    )


def _reset_object_subset(
    scene: InteractiveScene, phase_index: int
) -> torch.Tensor:
    object_asset = scene["object"]
    count = scene.num_envs
    phase_count = 4
    env_ids = torch.arange(
        phase_index % phase_count,
        count,
        phase_count,
        device=scene.device,
        dtype=torch.long,
    )
    if env_ids.numel() == 0:
        return env_ids
    root_pose = object_asset.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += scene.env_origins[env_ids]
    root_pose[:, 0] += 0.002 * float((phase_index % 3) - 1)
    object_asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    root_velocity = object_asset.data.default_root_vel.torch[env_ids].clone()
    object_asset.write_root_velocity_to_sim_index(
        root_velocity=root_velocity, env_ids=env_ids
    )
    return env_ids


class _TensorizedPiLRuntime:
    """GPU-resident representative Order 9 rollout state for the benchmark."""

    def __init__(self, scene: InteractiveScene, *, dt_s: float) -> None:
        self.device = torch.device(scene.device)
        self.dtype = torch.float32
        self.batch_size = int(scene.num_envs)
        self.dt_s = float(dt_s)
        physical_model = build_physical_model_from_config(
            "configs/robot/robot_model.yaml"
        )
        morphology = build_representative_order8_morphology(physical_model)
        self.module_count = len(morphology.modules)
        self.bucket = Order9TensorizedTopologyBucket(
            morphology,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.config = Order9LowLevelPolicyConfig()
        torch.manual_seed(9009)
        self.policy = Order9PhaseConditionedActorCritic(self.config).to(
            device=self.device, dtype=self.dtype
        )
        self.policy.eval()
        self.previous_action = torch.zeros(
            (self.batch_size, ORDER3_ACTION_SIZE),
            device=self.device,
            dtype=self.dtype,
        )
        self.recurrent_state = self.policy.initial_state(
            self.batch_size, device=self.device, dtype=self.dtype
        )
        self.time_s = torch.zeros(
            self.batch_size, device=self.device, dtype=self.dtype
        )
        self.phase_index = torch.arange(
            self.batch_size, device=self.device, dtype=torch.long
        ).remainder_(4)
        self.phase_progress = torch.zeros_like(self.time_s)
        self.phase_features = torch.zeros(
            (self.batch_size, self.config.phase_feature_dim),
            device=self.device,
            dtype=self.dtype,
        )
        task_types = list(TaskType)
        task_offset = task_types.index(TaskType.OBJECT_GRASP_CARRY)
        self.phase_features[:, task_offset] = 1.0
        self.phase_offset = len(task_types)
        self.progress_offset = self.phase_offset + self.config.max_phase_count
        # Stable task-adapter coordinates; equivalent to the schema path but
        # computed once rather than hashing inside every control step.
        adapter = int.from_bytes(
            hashlib.sha256(b"object_grasp_carry_v1").digest()[:8], "big"
        ) / float(2**64 - 1)
        self.phase_features[:, self.progress_offset + 1] = math.sin(
            2.0 * math.pi * adapter
        )
        self.phase_features[:, self.progress_offset + 2] = math.cos(
            2.0 * math.pi * adapter
        )
        module_mass = float(physical_model.aggregate_mass_kg)
        self.total_mass = torch.full(
            (self.batch_size,),
            module_mass * self.module_count,
            device=self.device,
            dtype=self.dtype,
        )
        aggregate_inertia = torch.tensor(
            physical_model.aggregate_inertia_body,
            device=self.device,
            dtype=self.dtype,
        )
        self.inertia = aggregate_inertia.reshape(1, 6).repeat(
            self.batch_size, 1
        ) * float(self.module_count)
        self.module_health = torch.ones(
            (self.batch_size, self.module_count),
            device=self.device,
            dtype=self.dtype,
        )
        self.controller_qp_feasible = torch.ones_like(self.time_s)
        self.controller_status = torch.zeros(
            (self.batch_size, 4), device=self.device, dtype=self.dtype
        )
        self.controller_status[:, 0] = 1.0
        self.allocation_residual = torch.zeros_like(self.time_s)
        self.task_success = torch.zeros_like(self.time_s)
        self.target_twist = torch.zeros(
            (self.batch_size, 6), device=self.device, dtype=self.dtype
        )
        self.parameter_count = sum(
            parameter.numel() for parameter in self.policy.parameters()
        )

    def reset_subset_(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.phase_index[env_ids] = (self.phase_index[env_ids] + 1).remainder(4)
        self.phase_progress[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.recurrent_state[env_ids] = 0.0

    @torch.no_grad()
    def infer(self, scene: InteractiveScene) -> None:
        robot = scene["robot"]
        root_position = _torch_tensor(robot.data.root_pos_w)
        root_quaternion = _torch_tensor(robot.data.root_quat_w)
        root_linear_velocity = _torch_tensor(robot.data.root_lin_vel_w)
        root_angular_velocity = _torch_tensor(robot.data.root_ang_vel_w)
        joint_positions = _torch_tensor(robot.data.joint_pos)
        joint_velocities = _torch_tensor(robot.data.joint_vel)
        body_pose = torch.cat((root_position, root_quaternion), dim=-1)
        body_twist = torch.cat(
            (root_linear_velocity, root_angular_velocity), dim=-1
        )
        module_pose = body_pose.unsqueeze(1).expand(
            -1, self.module_count, -1
        )
        module_twist = body_twist.unsqueeze(1).expand(
            -1, self.module_count, -1
        )
        local_joint_positions = joint_positions.unsqueeze(1).expand(
            -1, self.module_count, -1
        )
        local_joint_velocities = joint_velocities.unsqueeze(1).expand(
            -1, self.module_count, -1
        )
        joint_mask = torch.ones_like(local_joint_positions, dtype=torch.bool)
        graph_batch = self.bucket.update_runtime_(
            module_pose_world=module_pose,
            module_twist_world=module_twist,
            module_health=self.module_health,
            joint_positions=local_joint_positions,
            joint_velocities=local_joint_velocities,
            joint_mask=joint_mask,
        )
        target_pose = body_pose.clone()
        target_pose[:, 2].add_(0.10)
        actor_features = order9_low_level_actor_features_from_tensors(
            Order9CentroidalTensorObservation(
                time_s=self.time_s,
                module_count=torch.full_like(
                    self.time_s, float(self.module_count)
                ),
                total_mass_kg=self.total_mass,
                inertia_body=self.inertia,
                body_pose_world=body_pose,
                body_twist_world=body_twist,
                target_pose_world=target_pose,
                target_twist=self.target_twist,
                controller_qp_feasible=self.controller_qp_feasible,
                controller_status_one_hot=self.controller_status,
                allocation_residual_norm=self.allocation_residual,
                task_progress_ratio=self.phase_progress,
                task_success=self.task_success,
            ),
            max_modules=self.config.max_modules,
        )
        phase_slice = self.phase_features[
            :, self.phase_offset : self.progress_offset
        ]
        phase_slice.zero_()
        phase_slice.scatter_(1, self.phase_index.unsqueeze(1), 1.0)
        self.phase_features[:, self.progress_offset] = self.phase_progress
        result = self.policy.step(
            graph_batch,
            None,
            actor_features,
            self.previous_action,
            self.recurrent_state,
            phase_features=self.phase_features,
            deterministic=False,
        )
        self.previous_action.copy_(result.action)
        self.recurrent_state.copy_(result.recurrent_state)
        self.time_s.add_(self.dt_s)
        self.phase_progress.add_(1.0 / float(args_cli.phase_reset_interval)).clamp_(
            max=1.0
        )

    def metadata(self) -> dict[str, object]:
        return {
            "tensorized_pi_l_inference": True,
            "tensor_runtime_version": ORDER9_TENSOR_RUNTIME_VERSION,
            "pi_l_policy_version": ORDER9_PI_L_POLICY_VERSION,
            "pi_l_model_config": self.config.to_dict(),
            "pi_l_model_config_hash": stable_hash(self.config.to_dict()),
            "pi_l_parameter_count": int(self.parameter_count),
            "pi_l_global_action_dimension": ORDER3_ACTION_SIZE,
            "pi_l_joint_action_dimension_per_module": (
                3 * self.config.max_local_joint_slots
            ),
            "pi_l_policy_weights": "randomly_initialized_throughput_only",
            "pi_l_stochastic_actor_and_critic_evaluated": True,
        }


def _torch_tensor(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


@torch.no_grad()
def _step(
    scene: InteractiveScene,
    sim: sim_utils.SimulationContext,
    policy_runtime: _TensorizedPiLRuntime,
) -> None:
    robot = scene["robot"]
    policy_runtime.infer(scene)
    robot.set_joint_position_target_index(
        target=torch.zeros_like(robot.data.joint_pos.torch)
    )
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())


def main() -> dict[str, object]:
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=float(args_cli.dt),
            device=str(args_cli.device),
            use_fabric=True,
        )
    )
    setup_started = time.perf_counter()
    scene = InteractiveScene(
        Order9BenchmarkSceneCfg(
            num_envs=int(args_cli.num_envs),
            env_spacing=float(args_cli.env_spacing),
            replicate_physics=True,
        )
    )
    sim.reset()
    scene.reset()
    policy_runtime = _TensorizedPiLRuntime(scene, dt_s=float(args_cli.dt))
    setup_elapsed = time.perf_counter() - setup_started
    for index in range(int(args_cli.warmup_steps)):
        if index and index % int(args_cli.phase_reset_interval) == 0:
            env_ids = _reset_object_subset(
                scene, index // int(args_cli.phase_reset_interval)
            )
            policy_runtime.reset_subset_(env_ids)
        _step(scene, sim, policy_runtime)
    torch.cuda.synchronize() if str(args_cli.device).startswith("cuda") else None
    reset_count = 0
    started = time.perf_counter()
    for index in range(int(args_cli.measurement_steps)):
        if index and index % int(args_cli.phase_reset_interval) == 0:
            env_ids = _reset_object_subset(
                scene, index // int(args_cli.phase_reset_interval)
            )
            reset_count += int(env_ids.numel())
            policy_runtime.reset_subset_(env_ids)
        _step(scene, sim, policy_runtime)
    torch.cuda.synchronize() if str(args_cli.device).startswith("cuda") else None
    elapsed = time.perf_counter() - started
    robot_positions = scene["robot"].data.root_pos_w.torch
    object_positions = scene["object"].data.root_pos_w.torch
    finite = bool(
        torch.isfinite(robot_positions).all().item()
        and torch.isfinite(object_positions).all().item()
    )
    aggregate = int(args_cli.num_envs) * int(args_cli.measurement_steps) / elapsed
    result: dict[str, object] = {
        "benchmark_version": "order9_vectorized_isaac_child_v2",
        "attempted": True,
        "isaac_backed": True,
        "backend_version": "isaaclab_interactive_scene_physx_v1",
        "device": str(args_cli.device),
        "environment_count": int(args_cli.num_envs),
        "warmup_steps": int(args_cli.warmup_steps),
        "measurement_steps": int(args_cli.measurement_steps),
        "control_dt_s": float(args_cli.dt),
        "wall_elapsed_s": elapsed,
        "setup_wall_elapsed_s": setup_elapsed,
        "aggregate_env_steps_per_s": aggregate,
        "per_environment_steps_per_s": aggregate / int(args_cli.num_envs),
        "topology_bucketed": True,
        "topology_bucket_asset": ROBOT_USD,
        "phase_specific_resets": True,
        "phase_reset_environment_count": reset_count,
        "per_step_json_logging": False,
        "finite_state": finite,
        "object_mass_kg": 1.0,
        "object_size_m": [0.30, 0.40, 0.15],
        "object_friction": 0.6,
        "support_friction": 0.8,
        "selected_gripper_friction_contract": 4.5,
        "selected_gripper_contact_stiffness_contract_n_per_m": 7500.0,
        "selected_gripper_contact_damping_contract_n_s_per_m": 75.0,
        "raw_contact_actor_input": False,
        "unchanged_order8_acceptance_replaced": False,
        **policy_runtime.metadata(),
        "gpu_memory_allocated_bytes": (
            int(torch.cuda.memory_allocated())
            if str(args_cli.device).startswith("cuda")
            else 0
        ),
    }
    if not finite or not math.isfinite(aggregate) or aggregate <= 0.0:
        raise RuntimeError("Order9 vectorized Isaac benchmark produced invalid state")
    sim.stop()
    sim.clear_instance()
    return result


try:
    output = main()
    print("ORDER9_BENCHMARK_JSON=" + json.dumps(output, sort_keys=True))
finally:
    simulation_app.close()
