#!/usr/bin/env python3
from __future__ import annotations

"""Collect one topology/object bucket of real-Isaac Order 9 ``pi_L`` PPO.

The hot path is tensor-only: copied Isaac environments, phase-conditioned
``pi_L``, batched QPID/QP, privileged contact reduction, phase-aware reward,
and a compact raw tensor artifact.  JSONL schema reconstruction is deliberately
left to the post-simulation dataset builder.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", default="c2_pi_l_ppo_fixed_conservative")
    parser.add_argument("--pi-l-checkpoint", required=True)
    parser.add_argument("--pi-l-checkpoint-sha256", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-raw", required=True)
    parser.add_argument(
        "--evaluation-jsonl",
        help=(
            "Write deterministic, phase-zero, first-terminal episode evidence "
            "for BC-stage promotion."
        ),
    )
    parser.add_argument("--evaluation-episode-count", type=int, default=100)
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Diagnostic override; production defaults to the selected stage runtime.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        help="Diagnostic override; production defaults to the selected stage runtime.",
    )
    parser.add_argument("--seed", type=int, default=9009)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--env-spacing", type=float, default=3.0)
    parser.add_argument(
        "--robot-usd",
        help="Explicit topology-bucket USD; fixed C1/C2 must match its manifest.",
    )
    parser.add_argument(
        "--fixed-nominal-asset-manifest",
        default="artifacts/p4_full/order9/fixed_nominal_asset/manifest.json",
        help="Hash-bound fixed-morphology USD used by C1/C2.",
    )
    parser.add_argument("--morphology-graph-json")
    parser.add_argument("--task-spec-json")
    parser.add_argument(
        "--teacher-dataset-manifest",
        default="artifacts/p4_full/order9/c0_teacher/dataset/manifest.json",
        help=(
            "Checkpoint-bound C0 source for the fixed-nominal C1 active-knot "
            "reference."
        ),
    )
    parser.add_argument("--selected-gripper-friction", type=float)
    parser.add_argument("--contact-stiffness", type=float, default=7500.0)
    parser.add_argument("--contact-damping", type=float, default=75.0)
    parser.add_argument("--estimated-mass-kg", type=float)
    parser.add_argument("--estimated-inertia-body", nargs=6, type=float)
    parser.add_argument("--estimated-com-object", nargs=3, type=float)
    parser.add_argument(
        "--tensorboard-log-dir",
        help="Override the shared stage TensorBoard directory.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable live TensorBoard telemetry for diagnostic runs.",
    )
    parser.add_argument(
        "--canonical-phase-resets",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Use hash-bound Order 8 physical phase starts for its fixed topology.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
if (args_cli.num_envs is not None and args_cli.num_envs < 1) or (
    args_cli.rollout_steps is not None and args_cli.rollout_steps < 1
):
    raise ValueError("Order9 rollout environment/step counts must be positive")
if args_cli.seed < 0 or args_cli.dt <= 0.0 or args_cli.env_spacing <= 0.0:
    raise ValueError("Order9 rollout seed/dt/spacing is invalid")
if args_cli.evaluation_episode_count < 1:
    raise ValueError("Order9 evaluation episode count must be positive")
if args_cli.evaluation_jsonl is not None and args_cli.split != "validation":
    raise ValueError("Order9 BC promotion evidence must use the validation split")
for name in ("contact_stiffness", "contact_damping"):
    if not math.isfinite(float(getattr(args_cli, name))) or float(
        getattr(args_cli, name)
    ) <= 0.0:
        raise ValueError(f"--{name.replace('_', '-')} must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
from dataclasses import fields, replace

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

from amsrr.controllers.batched_qpid_controller import BatchedQPIDController
from amsrr.controllers.qpid_controller import QPIDControllerConfig
from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.geometry.contact_material import resolve_contact_friction
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.fixed_morphology_urdf import (
    articulated_morphology_graph_connections,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import ContactMode
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.schemas.task_spec import GeometryType, TaskSpec
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_fixed_nominal_asset import (
    load_order9_fixed_nominal_asset_manifest,
    validate_order9_fixed_nominal_asset_manifest_bytes,
)
from amsrr.simulation.order9_actuator_runtime import (
    Order9ActuatorRuntimeValues,
    order9_actuator_runtime_values,
    validate_order9_actuator_readback,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskRuntime,
    Order9ObjectTaskRuntimeConfig,
)
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.simulation.order9_tensor_isaac_io import Order9TensorIsaacIO
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
    ORDER9_PHASE_SUCCESSOR_REFERENCE_SEMANTICS,
    Order9TensorObjectTaskRuntime,
)
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_curriculum import (
    load_order9_learning_config,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_evaluation import (
    Order9EvaluationEpisode,
    write_order9_evaluation_episodes_jsonl,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_runtime_load import Order9RuntimeLoadMonitor
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    upgrade_teacher_trajectory_to_v2,
)
from amsrr.training.order9_tensor_pi_l_runtime import Order9TensorPiLRuntime
from amsrr.training.order9_tensor_reward import (
    ORDER9_TENSOR_REWARD_TERM_NAMES,
    Order9TensorRewardEngine,
    Order9TensorRewardInput,
    Order9TensorRewardState,
)
from amsrr.training.order9_tensorboard import (
    ORDER9_TENSORBOARD_LOGGER_VERSION,
    Order9TensorBoardLogger,
)
from amsrr.training.order9_tensor_teacher_reference import (
    load_order9_nominal_tensor_teacher_reference,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_PRODUCTION_COLLECTOR_VERSION,
    Order9TensorRolloutBuffer,
    write_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash


_RESULT_PREFIX = "ORDER9_ROLLOUT_JSON="
_COLLECTOR_VERSION = ORDER9_PRODUCTION_COLLECTOR_VERSION
_SIMULATOR_VERSION = "isaaclab3_physx_gpu_tensor_runtime"


@configclass
class _Order9RolloutSceneCfg(InteractiveSceneCfg):
    pass


class _PhaseStateBank:
    """Per-environment physical states; later phases unlock only on success."""

    def __init__(self, scene: InteractiveScene, phase_count: int) -> None:
        robot = scene["robot"]
        obj = scene["object"]
        batch = scene.num_envs
        device = torch.device(scene.device)
        dtype = _torch(robot.data.root_pose_w).dtype
        joint_count = _torch(robot.data.joint_pos).shape[1]
        self.available = torch.zeros(
            (batch, phase_count), device=device, dtype=torch.bool
        )
        self.robot_root_pose_local = torch.zeros(
            (batch, phase_count, 7), device=device, dtype=dtype
        )
        self.robot_root_twist = torch.zeros(
            (batch, phase_count, 6), device=device, dtype=dtype
        )
        self.joint_position = torch.zeros(
            (batch, phase_count, joint_count), device=device, dtype=dtype
        )
        self.joint_velocity = torch.zeros_like(self.joint_position)
        self.object_pose_local = torch.zeros(
            (batch, phase_count, 7), device=device, dtype=dtype
        )
        self.object_twist = torch.zeros(
            (batch, phase_count, 6), device=device, dtype=dtype
        )
        self.origins = scene.env_origins
        self._object = obj

    def capture(
        self,
        scene: InteractiveScene,
        *,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        robot = scene["robot"]
        obj = scene["object"]
        root_pose = _torch(robot.data.root_pose_w)[env_ids].clone()
        object_pose = _object_pose(obj)[env_ids].clone()
        root_pose[:, :3] -= self.origins[env_ids]
        object_pose[:, :3] -= self.origins[env_ids]
        self.robot_root_pose_local[env_ids, phase_indices] = root_pose
        self.robot_root_twist[env_ids, phase_indices] = torch.cat(
            (
                _torch(robot.data.root_lin_vel_w)[env_ids],
                _torch(robot.data.root_ang_vel_w)[env_ids],
            ),
            dim=-1,
        )
        self.joint_position[env_ids, phase_indices] = _torch(
            robot.data.joint_pos
        )[env_ids]
        self.joint_velocity[env_ids, phase_indices] = _torch(
            robot.data.joint_vel
        )[env_ids]
        self.object_pose_local[env_ids, phase_indices] = object_pose
        self.object_twist[env_ids, phase_indices] = _object_twist(obj)[env_ids]
        self.available[env_ids, phase_indices] = True

    def install(
        self,
        *,
        env_id: int,
        phase_index: int,
        robot_root_pose_local: torch.Tensor,
        robot_root_twist: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        object_pose_local: torch.Tensor,
        object_twist: torch.Tensor,
    ) -> None:
        self.robot_root_pose_local[env_id, phase_index] = robot_root_pose_local
        self.robot_root_twist[env_id, phase_index] = robot_root_twist
        self.joint_position[env_id, phase_index] = joint_position
        self.joint_velocity[env_id, phase_index] = joint_velocity
        self.object_pose_local[env_id, phase_index] = object_pose_local
        self.object_twist[env_id, phase_index] = object_twist
        self.available[env_id, phase_index] = True

    def select_reset_phases(
        self, env_ids: torch.Tensor, episode_serial: torch.Tensor
    ) -> torch.Tensor:
        selected = torch.zeros_like(env_ids)
        for output_index, env_id in enumerate(env_ids.tolist()):
            phases = torch.nonzero(
                self.available[env_id], as_tuple=False
            ).flatten()
            if phases.numel() == 0:
                raise RuntimeError("Order9 reset bank has no phase-zero state")
            offset = int(episode_serial[env_id].item()) % int(phases.numel())
            selected[output_index] = phases[offset]
        return selected

    def restore(
        self,
        scene: InteractiveScene,
        *,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        if not bool(self.available[env_ids, phase_indices].all()):
            raise RuntimeError("Order9 attempted an unavailable phase reset")
        scene.reset(env_ids)
        robot = scene["robot"]
        obj = scene["object"]
        root_pose = self.robot_root_pose_local[env_ids, phase_indices].clone()
        object_pose = self.object_pose_local[env_ids, phase_indices].clone()
        root_pose[:, :3] += self.origins[env_ids]
        object_pose[:, :3] += self.origins[env_ids]
        q = self.joint_position[env_ids, phase_indices]
        qdot = self.joint_velocity[env_ids, phase_indices]
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot_root_twist[env_ids, phase_indices],
            env_ids=env_ids,
        )
        robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
        robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        robot.set_joint_velocity_target_index(
            target=torch.zeros_like(qdot), env_ids=env_ids
        )
        robot.set_joint_effort_target_index(
            target=torch.zeros_like(qdot), env_ids=env_ids
        )
        obj.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(
            root_velocity=self.object_twist[env_ids, phase_indices],
            env_ids=env_ids,
        )


def main() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    config_path = (repository / args_cli.config).resolve()
    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, args_cli.stage)
    evaluation_mode = args_cli.evaluation_jsonl is not None
    if stage.learning_target.value != "pi_l" or (
        stage.learning_mode.value != "ppo" and not evaluation_mode
    ):
        raise ValueError(
            "vectorized pi_L rollout requires a pi_L PPO stage or evaluation mode"
        )
    if bool(stage.topology_randomized) and args_cli.morphology_graph_json is None:
        raise ValueError("topology-randomized stage requires --morphology-graph-json")
    configured_runtime = resolve_order9_stage_runtime(config, stage)
    if (
        configured_runtime.rollout_steps_per_environment is None
        and not evaluation_mode
    ):
        raise ValueError("vectorized pi_L rollout requires a PPO stage runtime")
    runtime_override_used = (
        args_cli.num_envs is not None or args_cli.rollout_steps is not None
    )
    if args_cli.num_envs is None:
        args_cli.num_envs = configured_runtime.environment_count
    if args_cli.rollout_steps is None:
        if configured_runtime.rollout_steps_per_environment is None:
            raise ValueError(
                "BC-stage evaluation requires an explicit --rollout-steps"
            )
        args_cli.rollout_steps = configured_runtime.rollout_steps_per_environment
    if evaluation_mode and args_cli.evaluation_episode_count > args_cli.num_envs:
        raise ValueError(
            "evaluation episode count cannot exceed the environment count"
        )
    physical = build_physical_model_from_config(
        repository / config.production_runtime.robot_model_config_path
    )
    actuator_runtime = order9_actuator_runtime_values(physical)
    canonical = load_order9_canonical_reset(
        repository / config.production_runtime.canonical_order8_report_path,
        expected_sha256=config.production_runtime.canonical_order8_report_sha256,
    )
    morphology = (
        MorphologyGraph.from_json(
            Path(args_cli.morphology_graph_json).read_text(encoding="utf-8")
        )
        if args_cli.morphology_graph_json
        else build_representative_order8_morphology(physical)
    )
    morphology.validate()
    robot_asset_manifest = None
    if bool(stage.topology_randomized):
        if args_cli.robot_usd is None:
            raise ValueError(
                "topology-randomized rollout requires --robot-usd"
            )
        robot_usd = Path(args_cli.robot_usd).resolve()
    else:
        robot_asset_manifest_path = (
            repository / args_cli.fixed_nominal_asset_manifest
        ).resolve()
        robot_asset_manifest = load_order9_fixed_nominal_asset_manifest(
            robot_asset_manifest_path
        )
        robot_usd = validate_order9_fixed_nominal_asset_manifest_bytes(
            robot_asset_manifest,
            repository_root=repository,
            expected_morphology=morphology,
            expected_physical_model_hash=physical.stable_hash(),
        )
        if (
            args_cli.robot_usd is not None
            and Path(args_cli.robot_usd).resolve() != robot_usd
        ):
            raise ValueError(
                "fixed rollout --robot-usd differs from its hash-bound manifest"
            )
    if not robot_usd.is_file():
        raise FileNotFoundError(robot_usd)
    task = _load_task(repository, config, canonical)
    target_object, geometry = _target_object_and_geometry(task)
    geometry_kind, geometry_values, object_half_height = _geometry_values(geometry)
    object_mass = float(target_object.mass_kg or 0.0)
    if object_mass <= 0.0 or target_object.inertia_kgm2 is None:
        raise ValueError("Order9 rollout object requires positive mass and inertia")
    object_friction = float(target_object.friction or 0.0)
    friction_resolution = resolve_contact_friction(
        task.metadata,
        target_entity_id=target_object.object_id,
        contact_mode=ContactMode.GRASP,
        target_surface_friction=object_friction,
    )
    selected_friction = (
        float(args_cli.selected_gripper_friction)
        if args_cli.selected_gripper_friction is not None
        else float(friction_resolution.robot_surface_friction or 4.5)
    )
    assignments, candidates = _teacher_assignments(task, morphology)
    selected_anchor_ids = tuple(assignment.anchor_id for assignment in assignments)
    if len(selected_anchor_ids) < 2 or len(set(selected_anchor_ids)) != len(
        selected_anchor_ids
    ):
        raise RuntimeError("Order9 grasp teacher must select unique multi-contact anchors")
    articulated_link_ids = tuple(
        link.link_id for link in physical.links if float(link.mass_kg) > 0.0
    )
    physical_robot_body_names = tuple(
        f"module_{module.module_id}__{link.link_id}"
        for module in sorted(morphology.modules, key=lambda value: value.module_id)
        for link in physical.links
        if link.link_id in articulated_link_ids
    )
    internal_robot_body_names: tuple[str, ...] = ()
    if robot_asset_manifest is not None:
        source_urdf = Path(robot_asset_manifest.source_urdf_path)
        if not source_urdf.is_absolute():
            source_urdf = repository / source_urdf
        internal_robot_body_names = tuple(
            f"module_{connection.child_module_id}__"
            f"{connection.child_mechanism_joint_id}__reroot_offset_link"
            for connection in articulated_morphology_graph_connections(
                source_urdf,
                morphology_graph=morphology,
            )
            if connection.child_mechanism_joint_id is not None
        )
    robot_body_names_expected = (
        *physical_robot_body_names,
        *internal_robot_body_names,
    )
    scene_cfg = _scene_cfg(
        robot_usd=robot_usd,
        object_kind=geometry_kind,
        geometry_values=geometry_values,
        object_pose=target_object.pose_world,
        object_mass=object_mass,
        object_friction=object_friction,
        support_size=tuple(canonical.metadata["object_support_size_m"]),
        support_pose=tuple(canonical.metadata["object_support_pose_world"]),
        robot_body_names=robot_body_names_expected,
        actuator_runtime=actuator_runtime,
    )
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=float(args_cli.dt),
            device=str(args_cli.device),
            use_fabric=True,
        )
    )
    load_monitor = Order9RuntimeLoadMonitor(
        sample_interval_s=config.production_runtime.runtime_load_sample_interval_s,
        device=str(args_cli.device),
    )
    load_monitor.start(torch_module=torch)
    setup_started = time.perf_counter()
    scene_cfg.num_envs = int(args_cli.num_envs)
    scene_cfg.env_spacing = float(args_cli.env_spacing)
    scene_cfg.replicate_physics = True
    scene_cfg.lazy_sensor_update = False
    scene = InteractiveScene(scene_cfg)
    selected_names = tuple(
        f"module_{anchor.module_id}__{anchor.link_id}"
        for anchor_id in selected_anchor_ids
        for anchor in morphology.robot_anchors
        if anchor.anchor_id == anchor_id
    )
    if len(selected_names) != len(selected_anchor_ids):
        raise RuntimeError("Order9 selected anchor body identity is incomplete")
    _bind_selected_material(
        sim.stage,
        selected_body_names=selected_names,
        friction=selected_friction,
        stiffness=float(args_cli.contact_stiffness),
        damping=float(args_cli.contact_damping),
    )
    _activate_nested_contact_reports(sim.stage)
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    obj = scene["object"]
    robot_sensor = scene["robot_contact"]
    object_sensor = scene["object_contact"]
    object_mass_properties_readback = _validate_object_mass_properties(
        obj,
        expected_mass_kg=object_mass,
        expected_inertia_kgm2=tuple(target_object.inertia_kgm2),
        expected_com_object=tuple(
            target_object.center_of_mass_object or (0.0, 0.0, 0.0)
        ),
    )
    actuator_readback = validate_order9_actuator_readback(
        robot.joint_names,
        _torch(robot.data.joint_effort_limits)[0].tolist(),
        _torch(robot.data.joint_velocity_limits)[0].tolist(),
        expected=actuator_runtime,
    )
    if set(robot.body_names) != set(robot_body_names_expected):
        raise RuntimeError(
            "Order9 robot USD body identity differs from morphology: "
            f"missing={sorted(set(robot_body_names_expected) - set(robot.body_names))}, "
            f"extra={sorted(set(robot.body_names) - set(robot_body_names_expected))}"
        )
    if object_sensor.contact_view.filter_count != len(robot.body_names):
        raise RuntimeError("Order9 object contact filter count differs from robot bodies")
    robot_sensor_order = tuple(str(value) for value in robot_sensor.body_names)
    if set(robot_sensor_order) != set(robot.body_names):
        raise RuntimeError("Order9 robot contact sensor body identity differs")
    robot_sensor_reorder = torch.tensor(
        [robot_sensor_order.index(name) for name in robot.body_names],
        device=scene.device,
        dtype=torch.long,
    )
    io = Order9TensorIsaacIO(
        morphology_graph=morphology,
        physical_model=physical,
        robot_body_names=robot.body_names,
        robot_joint_names=robot.joint_names,
        # The object ContactSensor emits one force-matrix column per filter in
        # configuration order, which is the morphology-derived body order.
        object_filter_body_names=robot_body_names_expected,
        selected_anchor_ids=selected_anchor_ids,
    )
    checkpoint = load_order9_policy_checkpoint(
        args_cli.pi_l_checkpoint,
        device=scene.device,
        expected_sha256=args_cli.pi_l_checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=order9_schedule_hash(config),
    )
    controller = BatchedQPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_qp", control_dt_s=float(args_cli.dt)
        )
    )
    policy_runtime = Order9TensorPiLRuntime(
        morphology_graph=morphology,
        physical_model=physical,
        policy=checkpoint.model,
        batch_size=scene.num_envs,
        device=scene.device,
        controller=controller,
        policy_frame_origins_world=scene.env_origins,
    )
    task_runtime = Order9TensorObjectTaskRuntime()
    teacher_reference = None
    if stage.stage_id == "c1_pi_l_bc_fixed_nominal":
        dataset_manifest_sha256 = checkpoint.metadata.input_artifact_hashes.get(
            "dataset_manifest"
        )
        if dataset_manifest_sha256 is None:
            raise ValueError("C1 checkpoint does not bind its C0 dataset manifest")
        teacher_reference = load_order9_nominal_tensor_teacher_reference(
            repository / args_cli.teacher_dataset_manifest,
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
            repository_root=repository,
            module_ids=policy_runtime.builder.module_ids,
            joint_ids=policy_runtime.decoder.local_joint_ids,
            device=scene.device,
            dtype=torch.float32,
        )
        if (
            teacher_reference.provenance["source_graph_hash"]
            != morphology.stable_hash()
        ):
            raise ValueError("C1 teacher reference morphology hash differs")
    scalar_task_runtime = Order9ObjectTaskRuntime(canonical, config=task_runtime.config)
    joint_reference_start, joint_reference_end = _canonical_joint_reference_banks(
        scalar_task_runtime,
        module_ids=policy_runtime.builder.module_ids,
        joint_ids=policy_runtime.decoder.local_joint_ids,
        device=torch.device(scene.device),
        dtype=torch.float32,
    )
    reward_engine = Order9TensorRewardEngine(control_dt_s=float(args_cli.dt))
    phase_count = len(ORDER9_OBJECT_TASK_PHASES)
    bank = _PhaseStateBank(scene, phase_count)
    teacher_phase_zero_expected = None
    canonical_resets = _canonical_resets_enabled(
        morphology, canonical.metadata["source_graph_hash"]
    )
    if canonical_resets:
        _seed_canonical_bank(
            bank,
            scene,
            canonical=canonical,
            task=task,
            robot_joint_names=tuple(robot.joint_names),
        )
        if teacher_reference is not None:
            teacher_phase_zero_expected = _install_teacher_phase_zero(
                bank,
                scene,
                io=io,
                teacher_reference=teacher_reference,
                robot_joint_names=tuple(robot.joint_names),
                task_object_pose_world=tuple(target_object.pose_world),
            )
    else:
        _align_arbitrary_phase_zero(
            scene,
            sim=sim,
            io=io,
            candidate_points=[
                next(
                    candidate.contact_pose_world
                    for candidate in candidates.candidates
                    if candidate.candidate_id == assignment.candidate_id
                )
                for assignment in assignments
            ],
            approach_offset_m=Order9ObjectTaskRuntimeConfig().approach_offset_m,
        )
        all_ids = torch.arange(scene.num_envs, device=scene.device)
        bank.capture(
            scene,
            env_ids=all_ids,
            phase_indices=torch.zeros_like(all_ids),
        )
    all_ids = torch.arange(scene.num_envs, device=scene.device, dtype=torch.long)
    phase_index = (
        torch.zeros_like(all_ids)
        if evaluation_mode
        else all_ids.remainder(phase_count)
        if canonical_resets
        else torch.zeros_like(all_ids)
    )
    bank.restore(scene, env_ids=all_ids, phase_indices=phase_index)
    sim.forward()
    scene.update(0.0)
    state = io.gather_state(robot=robot, object_asset=obj)
    if teacher_phase_zero_expected is not None:
        _validate_teacher_phase_zero_alignment(
            state,
            expected=teacher_phase_zero_expected,
        )
    control = policy_runtime.builder.build(
        module_pose_world=state.module_pose_world,
        module_twist_world=state.module_twist_world,
        local_joint_positions_rad=state.local_joint_positions_rad,
    )
    phase_start_body_pose = control.body_pose_world.clone()
    phase_start_object_pose = state.object_pose_world.clone()
    phase_elapsed = torch.zeros(
        scene.num_envs, device=scene.device, dtype=torch.float32
    )
    episode_time = torch.zeros_like(phase_elapsed)
    episode_serial = torch.zeros(
        scene.num_envs, device=scene.device, dtype=torch.long
    )
    episode_step = torch.zeros_like(episode_serial)
    lift_clearance = torch.full_like(
        phase_elapsed, float(canonical.lift_clearance_m)
    )
    transport_distance = torch.full_like(
        phase_elapsed, _transport_distance(task, target_object.object_id)
    )
    task_object_position_world = torch.tensor(
        target_object.pose_world[:3],
        device=scene.device,
        dtype=phase_elapsed.dtype,
    )
    duration_values = torch.tensor(
        [
            task_runtime.config.phase_duration_s[phase.value]
            for phase in ORDER9_OBJECT_TASK_PHASES
        ],
        device=scene.device,
        dtype=torch.float32,
    )
    target = task_runtime.target(
        phase_index=phase_index,
        phase_elapsed_s=phase_elapsed,
        reset_robot_root_pose_world=phase_start_body_pose,
        reset_object_pose_world=phase_start_object_pose,
        reset_joint_positions_rad=joint_reference_start.index_select(
            0, phase_index
        ),
        phase_end_joint_positions_rad=joint_reference_end.index_select(
            0, phase_index
        ),
        lift_clearance_m=lift_clearance,
        transport_distance_m=transport_distance,
    )
    target = _condition_target_on_teacher_reference(
        target,
        teacher_reference=teacher_reference,
        phase_index=phase_index,
        scene_origins=scene.env_origins,
        task_object_position_world=task_object_position_world,
    )
    reward_state = reward_engine.initial_state(
        object_pose_world=state.object_pose_world,
        desired_object_pose_world=target.phase_goal_object_pose_world,
    )
    estimated_mass = torch.full_like(
        phase_elapsed,
        float(args_cli.estimated_mass_kg or object_mass),
    )
    estimated_inertia = torch.tensor(
        args_cli.estimated_inertia_body or target_object.inertia_kgm2,
        device=scene.device,
        dtype=torch.float32,
    ).reshape(1, 6).expand(scene.num_envs, -1)
    estimated_com = torch.tensor(
        args_cli.estimated_com_object
        or target_object.center_of_mass_object
        or (0.0, 0.0, 0.0),
        device=scene.device,
        dtype=torch.float32,
    ).reshape(1, 3).expand(scene.num_envs, -1)
    support_top = torch.full_like(
        phase_elapsed, float(config.randomization.support_top_z_m)
    )
    half_height = torch.full_like(phase_elapsed, float(object_half_height))
    selected_mask = torch.ones(
        (scene.num_envs, len(assignments)),
        device=scene.device,
        dtype=torch.bool,
    )
    translated_tasks = [
        _translated_task(task, scene.env_origins[index], index)
        for index in range(scene.num_envs)
    ]
    split = DatasetSplit(args_cli.split)
    reward_names = ORDER9_TENSOR_REWARD_TERM_NAMES
    buffer = Order9TensorRolloutBuffer(
        _rollout_metadata(
            config=config,
            stage=stage,
            morphology=morphology,
            physical=physical,
            checkpoint_sha256=checkpoint.sha256,
            tasks=translated_tasks,
            split=split,
            assignments=assignments,
            io=io,
            reward_names=reward_names,
            selected_friction=selected_friction,
            canonical_resets=canonical_resets,
            robot_usd=robot_usd,
            robot_asset_manifest=robot_asset_manifest,
            estimated_mass_kg=float(estimated_mass[0].item()),
            estimated_inertia_body=tuple(
                float(value) for value in estimated_inertia[0].tolist()
            ),
            estimated_com_object=tuple(
                float(value) for value in estimated_com[0].tolist()
            ),
            object_mass_properties_readback=object_mass_properties_readback,
            actuator_readback=actuator_readback,
            teacher_reference=teacher_reference,
        )
    )
    tensorboard_logger = None
    tensorboard_update_index = None
    tensorboard_log_dir = None
    if not evaluation_mode and not args_cli.no_tensorboard:
        tensorboard_update_index = _next_ppo_update_index(
            checkpoint.metadata.metadata
        )
        tensorboard_log_dir = _tensorboard_log_dir(
            repository,
            artifact_root=config.production_runtime.artifact_root,
            stage_id=stage.stage_id,
            override=args_cli.tensorboard_log_dir,
        ) / split.value
        generation_environment_steps = (
            configured_runtime.generation_environment_steps
        )
        if generation_environment_steps is None:
            raise ValueError("Order9 TensorBoard PPO generation size is missing")
        tensorboard_logger = Order9TensorBoardLogger(
            tensorboard_log_dir,
            stage_id=stage.stage_id,
            generation_id=args_cli.generation_id,
            split=split.value,
            update_index=tensorboard_update_index,
            generation_environment_steps=generation_environment_steps,
            phase_labels=tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES),
            reward_term_names=reward_names,
        )
    setup_elapsed = time.perf_counter() - setup_started
    rollout_started = time.perf_counter()
    terminal_count = 0
    success_count = 0
    evaluation_active = torch.ones(
        scene.num_envs, device=scene.device, dtype=torch.bool
    )
    evaluation_step_count = torch.zeros(
        scene.num_envs, device=scene.device, dtype=torch.long
    )
    evaluation_return = torch.zeros_like(phase_elapsed)
    evaluation_outcomes: list[dict[str, object]] = []
    for rollout_index in range(int(args_cli.rollout_steps)):
        pre_state = state
        pre_target = target
        pre_phase = phase_index.clone()
        pre_actor_phase = _actor_phase_indices(pre_phase)
        pre_elapsed = phase_elapsed.clone()
        pre_time = episode_time.clone()
        pre_serial = episode_serial.clone()
        pre_step = episode_step.clone()
        payload_active = (pre_phase >= 2) & (pre_phase <= 5)
        policy_step = policy_runtime.compute(
            time_s=pre_time,
            phase_index=pre_actor_phase,
            task_target=pre_target,
            state=pre_state,
            estimated_payload_mass_kg=estimated_mass,
            estimated_payload_inertia_body=estimated_inertia,
            payload_active=payload_active,
            estimated_payload_com_object=estimated_com,
            deterministic=evaluation_mode,
        )
        io.apply(
            robot=robot,
            policy_command=policy_step.policy_command,
            controller_result=policy_step.controller_result,
        )
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(float(args_cli.dt))
        state = io.gather_state(robot=robot, object_asset=obj)
        post_control = policy_runtime.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
        )
        robot_net = _torch(robot_sensor.data.net_forces_w).index_select(
            1, robot_sensor_reorder
        )
        object_matrix = _torch(object_sensor.data.force_matrix_w)
        allow_contact = (
            (pre_target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_ATTACH)
            | (
                pre_target.contact_schedule_index
                == ORDER9_CONTACT_SCHEDULE_MAINTAIN
            )
            | (
                pre_target.contact_schedule_index
                == ORDER9_CONTACT_SCHEDULE_RELEASE
            )
        )
        contact = io.reduce_contacts(
            robot_net_contact_forces_world=robot_net,
            object_force_matrix_world=object_matrix,
            robot_body_linear_velocity_world=_torch(robot.data.body_lin_vel_w),
            robot_body_angular_velocity_world=_torch(robot.data.body_ang_vel_w),
            selected_assignment_mask=selected_mask,
            allow_selected_object_contact=allow_contact,
        )
        allocation = policy_step.controller_result.allocation
        rotor_saturation = allocation.thrust_clipped | allocation.vectoring_clipped
        reward = reward_engine.step(
            Order9TensorRewardInput(
                phase_index=pre_phase,
                phase_elapsed_s=pre_elapsed + float(args_cli.dt),
                phase_duration_s=duration_values[pre_phase],
                robot_body_pose_world=post_control.body_pose_world,
                robot_body_twist_world=post_control.body_twist_world,
                module_twist_world=state.module_twist_world,
                object_pose_world=state.object_pose_world,
                object_twist_world=state.object_twist_world,
                desired_robot_pose_world=(
                    pre_target.phase_goal_robot_root_pose_world
                ),
                desired_object_pose_world=pre_target.phase_goal_object_pose_world,
                selected_contact_forces_world=contact.selected_contact_forces_world,
                selected_link_twist_world=contact.selected_link_twist_world,
                selected_contact_mask=contact.selected_contact_mask,
                prohibited_collision=contact.prohibited_collision,
                support_top_z_m=support_top,
                object_half_height_m=half_height,
                qp_feasible=allocation.feasible,
                allocation_residual_norm=allocation.residual_norm,
                rotor_thrusts_n=allocation.rotor_thrusts_n,
                rotor_saturation=rotor_saturation,
                joint_torque_bias_nm=(
                    policy_step.policy_command.joint_torque_bias_nm
                ),
            ),
            reward_state,
        )
        last_phase = pre_phase == (phase_count - 1)
        completed_task = reward.phase_success & last_phase
        terminal = reward.terminal_failure | completed_task
        terminal_count += int(terminal.sum().item())
        success_count += int(completed_task.sum().item())
        if tensorboard_logger is not None:
            tensorboard_logger.log_rollout_step(
                rollout_index=rollout_index,
                reward=reward.reward,
                reward_terms=reward.terms,
                phase_index=pre_phase,
                statuses={
                    "phase_success": reward.phase_success,
                    "task_success": completed_task,
                    "terminal": terminal,
                    "hard_collision": reward.hard_collision,
                    "object_dropped": reward.object_dropped,
                    "qp_infeasible_terminal": reward.qp_infeasible_terminal,
                    "timeout": reward.timeout,
                    "qp_feasible": allocation.feasible,
                },
                elapsed_s=max(time.perf_counter() - rollout_started, 1.0e-12),
                runtime_sample=load_monitor.latest_sample(),
            )
        if evaluation_mode:
            evaluation_step_count[evaluation_active] += 1
            evaluation_return[evaluation_active] += reward.reward[evaluation_active]
            evaluation_terminal = terminal & evaluation_active
            for environment in torch.nonzero(
                evaluation_terminal, as_tuple=False
            ).flatten().tolist():
                succeeded = bool(completed_task[environment].item())
                failure_reason = None
                if not succeeded:
                    if bool(reward.hard_collision[environment].item()):
                        failure_reason = "hard_collision"
                    elif bool(reward.object_dropped[environment].item()):
                        failure_reason = "object_dropped"
                    elif bool(
                        reward.qp_infeasible_terminal[environment].item()
                    ):
                        failure_reason = "qp_infeasible_terminal"
                    elif bool(reward.timeout[environment].item()):
                        failure_reason = "phase_timeout"
                    else:  # pragma: no cover - terminal causes are exhaustive.
                        failure_reason = "terminal_failure"
                evaluation_outcomes.append(
                    {
                        "environment": environment,
                        "task_success": succeeded,
                        "safety_failure": bool(
                            reward.hard_collision[environment].item()
                            or reward.object_dropped[environment].item()
                            or reward.qp_infeasible_terminal[environment].item()
                        ),
                        "failure_reason": failure_reason,
                        "environment_step_count": int(
                            evaluation_step_count[environment].item()
                        ),
                        "episode_return": float(
                            evaluation_return[environment].item()
                        ),
                        "terminal_phase_index": int(
                            pre_actor_phase[environment].item()
                        ),
                        "hard_collision": bool(
                            reward.hard_collision[environment].item()
                        ),
                        "object_dropped": bool(
                            reward.object_dropped[environment].item()
                        ),
                        "qp_infeasible_terminal": bool(
                            reward.qp_infeasible_terminal[environment].item()
                        ),
                        "timeout": bool(reward.timeout[environment].item()),
                    }
                )
            evaluation_active &= ~evaluation_terminal
        next_phase_mask = reward.phase_success & ~last_phase
        next_ids = torch.nonzero(next_phase_mask, as_tuple=False).flatten()
        if next_ids.numel():
            next_indices = pre_phase[next_ids] + 1
            bank.capture(scene, env_ids=next_ids, phase_indices=next_indices)
            phase_index[next_ids] = next_indices
            phase_elapsed[next_ids] = 0.0
            planned_body_start, planned_object_start = (
                pre_target.planned_successor_start(next_ids)
            )
            phase_start_body_pose[next_ids] = planned_body_start
            phase_start_object_pose[next_ids] = planned_object_start
        continuing = ~terminal & ~next_phase_mask
        phase_elapsed[continuing] += float(args_cli.dt)
        episode_time[~terminal] += float(args_cli.dt)
        episode_step[~terminal] += 1
        target = task_runtime.target(
            phase_index=phase_index,
            phase_elapsed_s=phase_elapsed,
            reset_robot_root_pose_world=phase_start_body_pose,
            reset_object_pose_world=phase_start_object_pose,
            reset_joint_positions_rad=joint_reference_start.index_select(
                0, phase_index
            ),
            phase_end_joint_positions_rad=joint_reference_end.index_select(
                0, phase_index
            ),
            lift_clearance_m=lift_clearance,
            transport_distance_m=transport_distance,
        )
        target = _condition_target_on_teacher_reference(
            target,
            teacher_reference=teacher_reference,
            phase_index=phase_index,
            scene_origins=scene.env_origins,
            task_object_position_world=task_object_position_world,
        )
        reward_state = _reset_goal_distance_for_phase_transition(
            reward.next_state,
            env_ids=next_ids,
            object_pose_world=state.object_pose_world,
            desired_object_pose_world=target.phase_goal_object_pose_world,
        )
        evaluation_complete = bool(
            evaluation_mode and not bool(evaluation_active.any())
        )
        final_collection_step = (
            rollout_index + 1 == int(args_cli.rollout_steps)
            or evaluation_complete
        )
        truncated = torch.zeros_like(terminal)
        bootstrap = torch.zeros_like(phase_elapsed)
        if final_collection_step:
            truncated = ~terminal
            if bool(truncated.any()):
                bootstrap_value = policy_runtime.evaluate_bootstrap_value(
                    time_s=episode_time,
                    phase_index=_actor_phase_indices(phase_index),
                    task_target=target,
                    state=state,
                    estimated_payload_mass_kg=estimated_mass,
                    estimated_payload_inertia_body=estimated_inertia,
                    payload_active=(phase_index >= 2) & (phase_index <= 5),
                    estimated_payload_com_object=estimated_com,
                )
                bootstrap[truncated] = bootstrap_value[truncated]
        buffer.append(
            _artifact_step(
                valid=torch.ones_like(terminal),
                pre_time=pre_time,
                pre_phase=pre_actor_phase,
                pre_elapsed=pre_elapsed,
                duration=duration_values[pre_phase],
                pre_serial=pre_serial,
                pre_step=pre_step,
                pre_state=pre_state,
                pre_target=pre_target,
                policy_step=policy_step,
                selected_mask=selected_mask,
                contact=contact,
                reward=reward,
                reward_names=reward_names,
                rotor_saturation=rotor_saturation,
                terminal=terminal,
                truncated=truncated,
                bootstrap=bootstrap,
                post_state=state,
            )
        )
        policy_runtime.finish_transition(
            phase_success=reward.phase_success,
            terminal_or_reset=terminal,
            current_vectoring_angles_rad=post_control.current_vectoring_angles_rad,
        )
        if evaluation_complete:
            break
        reset_ids = torch.nonzero(terminal, as_tuple=False).flatten()
        if reset_ids.numel() and not final_collection_step:
            episode_serial[reset_ids] += 1
            reset_phases = bank.select_reset_phases(reset_ids, episode_serial)
            phase_index[reset_ids] = reset_phases
            phase_elapsed[reset_ids] = 0.0
            episode_time[reset_ids] = 0.0
            episode_step[reset_ids] = 0
            bank.restore(
                scene, env_ids=reset_ids, phase_indices=reset_phases
            )
            sim.forward()
            scene.update(0.0)
            state = io.gather_state(robot=robot, object_asset=obj)
            reset_control = policy_runtime.builder.build(
                module_pose_world=state.module_pose_world,
                module_twist_world=state.module_twist_world,
                local_joint_positions_rad=state.local_joint_positions_rad,
            )
            phase_start_body_pose[reset_ids] = reset_control.body_pose_world[
                reset_ids
            ]
            phase_start_object_pose[reset_ids] = state.object_pose_world[reset_ids]
            target = task_runtime.target(
                phase_index=phase_index,
                phase_elapsed_s=phase_elapsed,
                reset_robot_root_pose_world=phase_start_body_pose,
                reset_object_pose_world=phase_start_object_pose,
                reset_joint_positions_rad=joint_reference_start.index_select(
                    0, phase_index
                ),
                phase_end_joint_positions_rad=joint_reference_end.index_select(
                    0, phase_index
                ),
                lift_clearance_m=lift_clearance,
                transport_distance_m=transport_distance,
            )
            target = _condition_target_on_teacher_reference(
                target,
                teacher_reference=teacher_reference,
                phase_index=phase_index,
                scene_origins=scene.env_origins,
                task_object_position_world=task_object_position_world,
            )
            reward_state = reward_engine.reset_state_subset(
                reward_state,
                reset_ids,
                object_pose_world=state.object_pose_world,
                desired_object_pose_world=target.phase_goal_object_pose_world,
            )
    if str(args_cli.device).startswith("cuda"):
        torch.cuda.synchronize()
    rollout_elapsed = time.perf_counter() - rollout_started
    runtime_load = load_monitor.stop(torch_module=torch)
    artifact = buffer.finalize()
    artifact.metadata.update(
        {
            "setup_wall_elapsed_s": float(setup_elapsed),
            "rollout_wall_elapsed_s": float(rollout_elapsed),
            "collection_wall_elapsed_s": float(setup_elapsed + rollout_elapsed),
            "aggregate_env_steps_per_s": (
                artifact.environment_step_count / rollout_elapsed
            ),
            "end_to_end_env_steps_per_s": (
                artifact.environment_step_count / (setup_elapsed + rollout_elapsed)
            ),
            "environment_count": int(scene.num_envs),
            "rollout_steps": int(artifact.step_count),
            "requested_rollout_steps": int(args_cli.rollout_steps),
            "configured_environment_count": configured_runtime.environment_count,
            "configured_rollout_steps_per_environment": (
                configured_runtime.rollout_steps_per_environment
            ),
            "configured_generation_environment_steps": (
                configured_runtime.generation_environment_steps
            ),
            "runtime_override_used": runtime_override_used,
            "runtime_load": runtime_load,
            "terminal_count": int(terminal_count),
            "successful_terminal_count": int(success_count),
            "evaluation_mode": bool(evaluation_mode),
            "deterministic_policy": bool(evaluation_mode),
            "initial_phase_zero": bool(evaluation_mode),
            "tensorboard_enabled": tensorboard_logger is not None,
            "tensorboard_log_dir": (
                None if tensorboard_log_dir is None else str(tensorboard_log_dir)
            ),
            "tensorboard_update_index": tensorboard_update_index,
            "tensorboard_logger_version": (
                ORDER9_TENSORBOARD_LOGGER_VERSION
                if tensorboard_logger is not None
                else None
            ),
        }
    )
    if tensorboard_logger is not None:
        tensorboard_logger.log_rollout_summary(
            environment_steps=artifact.environment_step_count,
            wall_elapsed_s=rollout_elapsed,
            runtime_load=runtime_load,
        )
        tensorboard_logger.close()
    raw_sha = write_order9_tensor_rollout_artifact(args_cli.output_raw, artifact)
    evaluation_episode_count = 0
    if evaluation_mode:
        requested = int(args_cli.evaluation_episode_count)
        if len(evaluation_outcomes) < requested:
            raise RuntimeError(
                "Order9 evaluation rollout produced "
                f"{len(evaluation_outcomes)} first-terminal episodes; "
                f"{requested} required"
            )
        selected_outcomes = sorted(
            evaluation_outcomes, key=lambda item: int(item["environment"])
        )[:requested]
        raw_path = Path(args_cli.output_raw).resolve()
        evaluation_episodes = []
        for outcome in selected_outcomes:
            environment = int(outcome["environment"])
            succeeded = bool(outcome["task_success"])
            evaluation_episodes.append(
                Order9EvaluationEpisode(
                    episode_id=(
                        f"{args_cli.generation_id}:evaluation:env:{environment:04d}"
                    ),
                    task_id=translated_tasks[environment].task_id,
                    split=split,
                    random_seed=int(args_cli.seed) + environment,
                    task_success=succeeded,
                    no_fallback_success=succeeded,
                    safety_failure=bool(outcome["safety_failure"]),
                    high_level_decision_count=0,
                    fallback_decision_count=0,
                    environment_step_count=int(
                        outcome["environment_step_count"]
                    ),
                    isaac_backed=True,
                    full_mesh_evaluation=True,
                    source_artifact_path=str(raw_path),
                    source_artifact_sha256=raw_sha,
                    failure_reason=outcome["failure_reason"],
                    metrics={
                        "episode_return": float(outcome["episode_return"]),
                        "terminal_phase_index": float(
                            outcome["terminal_phase_index"]
                        ),
                        "hard_collision": float(outcome["hard_collision"]),
                        "object_dropped": float(outcome["object_dropped"]),
                        "qp_infeasible_terminal": float(
                            outcome["qp_infeasible_terminal"]
                        ),
                        "timeout": float(outcome["timeout"]),
                    },
                    metadata={
                        "environment_index": environment,
                        "generation_id": args_cli.generation_id,
                        "deterministic_policy": True,
                        "initial_phase_index": 0,
                        "first_terminal_only": True,
                        "raw_contact_actor_input": False,
                    },
                )
            )
        write_order9_evaluation_episodes_jsonl(
            args_cli.evaluation_jsonl, evaluation_episodes
        )
        evaluation_episode_count = len(evaluation_episodes)
    finite = bool(
        torch.isfinite(state.robot_root_pose_world).all()
        and torch.isfinite(state.object_pose_world).all()
    )
    result = {
        "passed": finite,
        "generation_id": args_cli.generation_id,
        "stage_id": stage.stage_id,
        "split": split.value,
        "raw_artifact_path": str(Path(args_cli.output_raw).resolve()),
        "raw_artifact_sha256": raw_sha,
        "environment_count": scene.num_envs,
        "rollout_steps": int(artifact.step_count),
        "requested_rollout_steps": int(args_cli.rollout_steps),
        "environment_steps": artifact.environment_step_count,
        "wall_elapsed_s": rollout_elapsed,
        "setup_wall_elapsed_s": setup_elapsed,
        "collection_wall_elapsed_s": setup_elapsed + rollout_elapsed,
        "aggregate_env_steps_per_s": artifact.environment_step_count
        / rollout_elapsed,
        "end_to_end_env_steps_per_s": artifact.environment_step_count
        / (setup_elapsed + rollout_elapsed),
        "configured_environment_count": configured_runtime.environment_count,
        "configured_rollout_steps_per_environment": (
            configured_runtime.rollout_steps_per_environment
        ),
        "runtime_override_used": runtime_override_used,
        "runtime_load": {
            name: value
            for name, value in runtime_load.items()
            if name != "samples"
        },
        "terminal_count": terminal_count,
        "successful_terminal_count": success_count,
        "evaluation_mode": evaluation_mode,
        "evaluation_episode_count": evaluation_episode_count,
        "evaluation_jsonl": (
            None
            if args_cli.evaluation_jsonl is None
            else str(Path(args_cli.evaluation_jsonl).resolve())
        ),
        "deterministic_policy": evaluation_mode,
        "initial_phase_zero": evaluation_mode,
        "tensorboard_enabled": tensorboard_logger is not None,
        "tensorboard_log_dir": (
            None if tensorboard_log_dir is None else str(tensorboard_log_dir)
        ),
        "tensorboard_update_index": tensorboard_update_index,
        "tensorboard_logger_version": (
            ORDER9_TENSORBOARD_LOGGER_VERSION
            if tensorboard_logger is not None
            else None
        ),
        "canonical_phase_resets": canonical_resets,
        "unlocked_phase_indices": [
            index
            for index in range(phase_count)
            if bool(bank.available[:, index].any())
        ],
        "raw_contact_actor_input": False,
        "finite_state": finite,
    }
    sim.stop()
    sim.clear_instance()
    if not finite:
        raise RuntimeError("Order9 rollout produced non-finite physical state")
    return result


def _next_ppo_update_index(metadata: dict[str, object]) -> int:
    raw = metadata.get("ppo_update_index")
    if raw is None:
        return 0
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ValueError("Order9 parent checkpoint PPO update index is invalid")
    return raw + 1


def _tensorboard_log_dir(
    repository: Path,
    *,
    artifact_root: str,
    stage_id: str,
    override: str | None,
) -> Path:
    value = (
        Path(override)
        if override is not None
        else Path(artifact_root) / "stages" / stage_id / "tensorboard"
    )
    return (repository / value).resolve() if not value.is_absolute() else value.resolve()


def _load_task(repository: Path, config, canonical) -> TaskSpec:
    if args_cli.task_spec_json:
        task = TaskSpec.from_json(
            Path(args_cli.task_spec_json).read_text(encoding="utf-8")
        )
        task.validate()
        return task
    return build_order8_grasp_carry_task_spec(
        object_pose_world=tuple(canonical.object_pose_world),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=canonical.transport_distance_m,
        support_height_m=config.randomization.support_top_z_m,
        max_contact_force_n=config.hard_checker.qp_force_scale_n,
        max_contact_torque_nm=config.hard_checker.qp_torque_scale_nm,
        selected_gripper_friction=(
            config.randomization.nominal_selected_gripper_friction
        ),
        task_id="order9-vectorized-nominal",
    )


def _target_object_and_geometry(task: TaskSpec):
    target_id = next(
        goal.target_entity_id
        for goal in task.goals
        if goal.goal_type == "object_pose"
    )
    obj = next(value for value in task.scene.objects if value.object_id == target_id)
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == obj.geometry_id
    )
    return obj, geometry


def _geometry_values(geometry):
    params = dict(geometry.primitive_params or {})
    if geometry.geometry_type == GeometryType.BOX:
        size = tuple(float(value) for value in params["size_m"])
        return "box", size, 0.5 * size[2]
    if geometry.geometry_type == GeometryType.SPHERE:
        radius = float(params["radius_m"])
        return "sphere", (radius,), radius
    if geometry.geometry_type == GeometryType.CYLINDER:
        radius, height = float(params["radius_m"]), float(params["height_m"])
        return "cylinder", (radius, height), 0.5 * height
    if geometry.geometry_type == GeometryType.CAPSULE:
        radius, height = float(params["radius_m"]), float(params["height_m"])
        return "capsule", (radius, height), 0.5 * height + radius
    raise ValueError("Order9 rollout currently supports primitive object geometries")


def _object_spawn(kind, values, *, mass, friction):
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(mass)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(friction),
            dynamic_friction=float(friction),
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.72, 0.38, 0.12)
        ),
        activate_contact_sensors=True,
    )
    if kind == "box":
        return sim_utils.CuboidCfg(size=values, **common)
    if kind == "sphere":
        return sim_utils.SphereCfg(radius=values[0], **common)
    if kind == "cylinder":
        return sim_utils.CylinderCfg(radius=values[0], height=values[1], **common)
    return sim_utils.CapsuleCfg(radius=values[0], height=values[1], **common)


def _scene_cfg(
    *,
    robot_usd: Path,
    object_kind: str,
    geometry_values: tuple[float, ...],
    object_pose,
    object_mass: float,
    object_friction: float,
    support_size,
    support_pose,
    robot_body_names,
    actuator_runtime: Order9ActuatorRuntimeValues,
):
    cfg = _Order9RolloutSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8,
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
                joint_names_expr=[".*gimbal.*"],
                stiffness=actuator_runtime.gimbal_stiffness,
                damping=actuator_runtime.gimbal_damping,
                armature=actuator_runtime.gimbal_armature,
                effort_limit_sim=actuator_runtime.gimbal_effort_limit,
                velocity_limit_sim=actuator_runtime.gimbal_velocity_limit,
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=actuator_runtime.dock_stiffness,
                damping=actuator_runtime.dock_damping,
                armature=actuator_runtime.dock_armature,
                effort_limit_sim=actuator_runtime.dock_effort_limit,
                velocity_limit_sim=actuator_runtime.dock_velocity_limit,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"], stiffness=0.0, damping=0.0
            ),
        },
    )
    cfg.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_object_spawn(
            object_kind,
            geometry_values,
            mass=object_mass,
            friction=object_friction,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(value) for value in object_pose[:3]),
            rot=tuple(float(value) for value in object_pose[3:7]),
        ),
    )
    cfg.support = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(value) for value in support_size),
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
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in support_pose[:3]),
            rot=tuple(float(value) for value in support_pose[3:7]),
        ),
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        ),
    )
    cfg.robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        debug_vis=False,
        max_contact_data_count_per_prim=16,
    )
    cfg.object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        debug_vis=False,
        max_contact_data_count_per_prim=max(32, 4 * len(robot_body_names)),
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{name}" for name in robot_body_names
        ],
    )
    return cfg


def _teacher_assignments(task: TaskSpec, morphology: MorphologyGraph):
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    context = HighLevelPolicyContext(
        built.irg, envelope, morphology, candidates
    )
    trajectory = upgrade_teacher_trajectory_to_v2(
        GraspCarryBaselinePlanner().plan(context), context
    )
    maintain = next(
        knot
        for knot in trajectory.knots
        if any(
            assignment.schedule_state == "maintain"
            for assignment in knot.contact_assignments
        )
    )
    assignments = sorted(
        (
            assignment
            for assignment in maintain.contact_assignments
            if assignment.schedule_state == "maintain"
        ),
        key=lambda value: value.anchor_id,
    )
    return assignments, candidates


def _bind_selected_material(
    stage,
    *,
    selected_body_names,
    friction: float,
    stiffness: float,
    damping: float,
) -> None:
    material_path = "/World/Order9SelectedGripperMaterial"
    cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
        compliant_contact_stiffness=stiffness,
        compliant_contact_damping=damping,
        friction_combine_mode="max",
    )
    cfg.func(material_path, cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(Sdf.Path(material_path)))
    selected = set(selected_body_names)
    count = 0
    for env_id in range(int(args_cli.num_envs)):
        root = stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{env_id}/Robot"))
        if not root.IsValid():
            raise RuntimeError("Order9 rollout robot prim is invalid")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.GetName() not in selected or not prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                continue
            api = (
                UsdShade.MaterialBindingAPI(prim)
                if prim.HasAPI(UsdShade.MaterialBindingAPI)
                else UsdShade.MaterialBindingAPI.Apply(prim)
            )
            api.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            count += 1
    expected_minimum = int(args_cli.num_envs) * len(selected)
    if count < expected_minimum:
        raise RuntimeError(
            f"Order9 selected material bound {count}, expected at least {expected_minimum}"
        )


def _activate_nested_contact_reports(stage) -> None:
    count = 0
    for env_id in range(int(args_cli.num_envs)):
        root = stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{env_id}/Robot"))
        if not root.IsValid():
            raise RuntimeError("Order9 rollout robot prim is invalid")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            PhysxSchema.PhysxContactReportAPI.Apply(
                prim
            ).CreateThresholdAttr().Set(0.0)
            count += 1
    if count == 0:
        raise RuntimeError("Order9 rollout found no robot contact-report body")


def _canonical_resets_enabled(morphology, source_graph_hash: str) -> bool:
    matches = morphology.stable_hash() == str(source_graph_hash)
    if args_cli.canonical_phase_resets == "yes" and not matches:
        raise ValueError("canonical phase resets cannot be applied to another morphology")
    if args_cli.canonical_phase_resets == "no":
        return False
    return matches


def _seed_canonical_bank(
    bank: _PhaseStateBank,
    scene: InteractiveScene,
    *,
    canonical,
    task: TaskSpec,
    robot_joint_names,
) -> None:
    target_object, _ = _target_object_and_geometry(task)
    canonical_position = canonical.object_pose_world[:3]
    offset = tuple(
        float(target_object.pose_world[index]) - float(canonical_position[index])
        for index in range(3)
    )
    yaw_offset = _yaw(target_object.pose_world[3:7]) - _yaw(
        canonical.object_pose_world[3:7]
    )
    runtime = Order9ObjectTaskRuntime(canonical)
    robot = scene["robot"]
    default_q = _torch(robot.data.default_joint_pos)
    joint_lookup = {name: index for index, name in enumerate(robot_joint_names)}
    device = torch.device(scene.device)
    dtype = default_q.dtype
    for env_id in range(scene.num_envs):
        for phase_index in range(len(ORDER9_OBJECT_TASK_PHASES)):
            reset = runtime.reset_for_phase(
                phase_index,
                object_position_offset_world=offset,
                object_yaw_offset_rad=yaw_offset,
            )
            q = default_q[env_id].clone()
            qdot = torch.zeros_like(q)
            for global_id, value in reset.joint_positions_rad.items():
                name = global_id.replace(":", "__", 1)
                if name in joint_lookup:
                    q[joint_lookup[name]] = float(value)
            bank.install(
                env_id=env_id,
                phase_index=phase_index,
                robot_root_pose_local=torch.tensor(
                    reset.robot_root_pose_world, device=device, dtype=dtype
                ),
                robot_root_twist=torch.zeros(6, device=device, dtype=dtype),
                joint_position=q,
                joint_velocity=qdot,
                object_pose_local=torch.tensor(
                    reset.object_pose_world, device=device, dtype=dtype
                ),
                object_twist=torch.zeros(6, device=device, dtype=dtype),
            )


def _install_teacher_phase_zero(
    bank: _PhaseStateBank,
    scene: InteractiveScene,
    *,
    io: Order9TensorIsaacIO,
    teacher_reference,
    robot_joint_names: tuple[str, ...],
    task_object_pose_world: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Install the exact successful C0 phase-zero physical distribution.

    C0 used one articulation per module, whereas the tensor runtime uses one
    rigid fixed-morphology articulation.  Align its root through module 0's
    physical-model frame, then fail closed later if every module does not land
    on the C0 geometry within the rigid-assembly tolerance.
    """

    robot = scene["robot"]
    device = torch.device(scene.device)
    if tuple(teacher_reference.module_ids) != tuple(io.module_ids):
        raise RuntimeError(
            "Order9 C0 initial module identity differs from the fixed USD"
        )
    default_q = _torch(robot.data.default_joint_pos)
    dtype = default_q.dtype
    source_object_position = teacher_reference.initial_object_pose_world[:3]
    task_offset = torch.tensor(
        task_object_pose_world[:3], device=device, dtype=dtype
    ) - source_object_position.to(device=device, dtype=dtype)

    current_root = _torch(robot.data.root_pose_w)[0].detach().cpu().tolist()
    module_zero_index = io.module_ids.index(0)
    current_module_zero = _torch(robot.data.body_pose_w)[
        0, io.module_body_indices[module_zero_index]
    ]
    # The caller validates body identity before this helper.  Remove the copied
    # environment origin so scalar pose composition remains in local world.
    origin_zero = scene.env_origins[0]
    current_root[:3] = (
        torch.tensor(current_root[:3], device=device, dtype=dtype) - origin_zero
    ).detach().cpu().tolist()
    current_module_zero_local = current_module_zero.clone()
    current_module_zero_local[:3] -= origin_zero
    root_to_module_zero = compose_pose(
        inverse_pose(tuple(float(value) for value in current_root)),
        tuple(float(value) for value in current_module_zero_local.detach().cpu()),
    )

    initial_module_pose = teacher_reference.initial_module_pose_world.to(
        device=device, dtype=dtype
    ).clone()
    initial_module_pose[:, :3] += task_offset
    desired_root_pose = compose_pose(
        tuple(float(value) for value in initial_module_pose[0].detach().cpu()),
        inverse_pose(root_to_module_zero),
    )
    module_zero_twist = teacher_reference.initial_module_twist_world[0].to(
        device=device, dtype=dtype
    )
    root_position = torch.tensor(
        desired_root_pose[:3], device=device, dtype=dtype
    )
    root_to_module_world = initial_module_pose[0, :3] - root_position
    root_linear_velocity = module_zero_twist[:3] - torch.linalg.cross(
        module_zero_twist[3:], root_to_module_world, dim=-1
    )
    root_twist = torch.cat((root_linear_velocity, module_zero_twist[3:]))

    joint_lookup = {name: index for index, name in enumerate(robot_joint_names)}
    missing = sorted(
        set(teacher_reference.initial_joint_positions_rad) - set(joint_lookup)
    )
    if missing:
        raise RuntimeError(
            f"Order9 C0 initial joints are missing from fixed USD: {missing}"
        )
    initial_object_pose = teacher_reference.initial_object_pose_world.to(
        device=device, dtype=dtype
    ).clone()
    initial_object_pose[:3] += task_offset
    initial_object_twist = teacher_reference.initial_object_twist_world.to(
        device=device, dtype=dtype
    )
    for env_id in range(scene.num_envs):
        q = default_q[env_id].clone()
        qdot = torch.zeros_like(q)
        for name, value in teacher_reference.initial_joint_positions_rad.items():
            q[joint_lookup[name]] = float(value)
            qdot[joint_lookup[name]] = float(
                teacher_reference.initial_joint_velocities_radps[name]
            )
        bank.install(
            env_id=env_id,
            phase_index=0,
            robot_root_pose_local=torch.tensor(
                desired_root_pose, device=device, dtype=dtype
            ),
            robot_root_twist=root_twist,
            joint_position=q,
            joint_velocity=qdot,
            object_pose_local=initial_object_pose,
            object_twist=initial_object_twist,
        )

    origins = scene.env_origins.to(dtype=dtype)
    expected_module_pose = initial_module_pose.unsqueeze(0).expand(
        scene.num_envs, -1, -1
    ).clone()
    expected_module_pose[:, :, :3] += origins.unsqueeze(1)
    expected_object_pose = initial_object_pose.unsqueeze(0).expand(
        scene.num_envs, -1
    ).clone()
    expected_object_pose[:, :3] += origins
    return {
        "module_pose_world": expected_module_pose,
        "module_twist_world": teacher_reference.initial_module_twist_world.to(
            device=device, dtype=dtype
        ).unsqueeze(0).expand(scene.num_envs, -1, -1),
        "object_pose_world": expected_object_pose,
        "object_twist_world": initial_object_twist.unsqueeze(0).expand(
            scene.num_envs, -1
        ),
    }


def _validate_teacher_phase_zero_alignment(
    state,
    *,
    expected: dict[str, torch.Tensor],
) -> None:
    module_position_error = torch.linalg.vector_norm(
        state.module_pose_world[..., :3]
        - expected["module_pose_world"][..., :3],
        dim=-1,
    )
    module_orientation_error = _quaternion_distance_rad(
        state.module_pose_world[..., 3:7],
        expected["module_pose_world"][..., 3:7],
    )
    object_position_error = torch.linalg.vector_norm(
        state.object_pose_world[..., :3]
        - expected["object_pose_world"][..., :3],
        dim=-1,
    )
    object_orientation_error = _quaternion_distance_rad(
        state.object_pose_world[..., 3:7],
        expected["object_pose_world"][..., 3:7],
    )
    maxima = {
        "module_position_m": float(module_position_error.max().item()),
        "module_orientation_rad": float(module_orientation_error.max().item()),
        "object_position_m": float(object_position_error.max().item()),
        "object_orientation_rad": float(object_orientation_error.max().item()),
    }
    if (
        maxima["module_position_m"] > 2.0e-3
        or maxima["module_orientation_rad"] > 3.0e-3
        or maxima["object_position_m"] > 1.0e-4
        or maxima["object_orientation_rad"] > 1.0e-4
    ):
        raise RuntimeError(
            "Order9 fixed USD cannot reproduce the exact C0 phase-zero "
            f"physical state: {maxima}"
        )


def _quaternion_distance_rad(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual = actual / actual.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    expected = expected / expected.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    dot = (actual * expected).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _align_arbitrary_phase_zero(
    scene: InteractiveScene,
    *,
    sim: sim_utils.SimulationContext,
    io: Order9TensorIsaacIO,
    candidate_points,
    approach_offset_m: float,
) -> None:
    robot = scene["robot"]
    body_pose = _torch(robot.data.body_pose_w)
    selected = torch.tensor(
        io.selected_anchor_body_indices, device=scene.device, dtype=torch.long
    )
    anchor_centroid = body_pose.index_select(1, selected)[..., :3].mean(dim=1)
    target_local = torch.tensor(
        [pose[:3] for pose in candidate_points],
        device=scene.device,
        dtype=anchor_centroid.dtype,
    ).mean(dim=0)
    target = target_local.unsqueeze(0) + scene.env_origins
    target[:, 0] -= float(approach_offset_m)
    root_pose = _torch(robot.data.root_pose_w).clone()
    root_pose[:, :3] += target - anchor_centroid
    env_ids = torch.arange(scene.num_envs, device=scene.device)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(
            (scene.num_envs, 6), device=scene.device, dtype=root_pose.dtype
        ),
        env_ids=env_ids,
    )
    sim.forward()
    scene.update(0.0)


def _transport_distance(task: TaskSpec, object_id: str) -> float:
    obj = next(value for value in task.scene.objects if value.object_id == object_id)
    goal = next(
        value
        for value in task.goals
        if value.goal_type == "object_pose" and value.target_entity_id == object_id
    )
    if goal.target_pose_world is None:
        raise ValueError("Order9 object goal has no target pose")
    return float(goal.target_pose_world[0]) - float(obj.pose_world[0])


def _translated_task(task: TaskSpec, origin: torch.Tensor, index: int) -> TaskSpec:
    value = TaskSpec.from_dict(task.to_dict())
    translation = tuple(float(item) for item in origin.tolist())
    value.task_id = f"{task.task_id}:env:{index:04d}"
    for obj in value.scene.objects:
        obj.pose_world = _translate_pose(obj.pose_world, translation)
    for surface in value.scene.environment.support_surfaces:
        surface.pose_world = _translate_pose(surface.pose_world, translation)
    for obstacle in value.scene.environment.obstacles:
        obstacle.pose_world = _translate_pose(obstacle.pose_world, translation)
    for goal in value.goals:
        if goal.target_pose_world is not None:
            goal.target_pose_world = _translate_pose(
                goal.target_pose_world, translation
            )
    value.metadata = {
        **value.metadata,
        "isaac_environment_origin_world": list(translation),
        "isaac_environment_index": index,
    }
    value.validate()
    return value


def _rollout_metadata(
    *,
    config,
    stage,
    morphology,
    physical,
    checkpoint_sha256,
    tasks,
    split,
    assignments,
    io,
    reward_names,
    selected_friction,
    canonical_resets,
    robot_usd,
    robot_asset_manifest,
    estimated_mass_kg,
    estimated_inertia_body,
    estimated_com_object,
    object_mass_properties_readback,
    actuator_readback,
    teacher_reference,
):
    thrust_model_hash = str(physical.metadata.get("thrust_model_hash", ""))
    if not thrust_model_hash:
        raise RuntimeError("Order9 PhysicalModel lacks thrust-model provenance")
    return {
        "generation_id": args_cli.generation_id,
        "pi_l_checkpoint_sha256": checkpoint_sha256,
        "stage_id": stage.stage_id,
        "stage_config_hash": stable_hash(stage.to_dict()),
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "config_hash": stable_hash(config.to_dict()),
        "morphology_graph": morphology.to_dict(),
        "physical_model_hash": physical.stable_hash(),
        "urdf_hash": hash_file(physical.urdf_path),
        "thrust_model_hash": thrust_model_hash,
        "robot_usd_sha256": hash_file(robot_usd),
        "robot_asset_manifest": (
            None
            if robot_asset_manifest is None
            else robot_asset_manifest.to_dict()
        ),
        "simulator_version": _SIMULATOR_VERSION,
        "device": str(args_cli.device),
        "simulator_hash": stable_hash(
            {
                "simulator_version": _SIMULATOR_VERSION,
                "collector_version": _COLLECTOR_VERSION,
                "use_fabric": True,
                "quaternion_layout": "xyzw",
                "control_dt_s": float(args_cli.dt),
                "phase_successor_reference_semantics": (
                    ORDER9_PHASE_SUCCESSOR_REFERENCE_SEMANTICS
                ),
            }
        ),
        "random_seed": int(args_cli.seed),
        "estimated_payload_mass_kg": float(estimated_mass_kg),
        "estimated_payload_inertia_body": list(estimated_inertia_body),
        "estimated_payload_com_object": list(estimated_com_object),
        "object_mass_properties_readback": dict(object_mass_properties_readback),
        "actuator_readback": dict(actuator_readback),
        "teacher_reference": (
            None
            if teacher_reference is None
            else dict(teacher_reference.provenance)
        ),
        "task_specs": [task.to_dict() for task in tasks],
        "environment_splits": [split.value for _ in tasks],
        "assignment_templates_by_environment": [
            [assignment.to_dict() for assignment in assignments] for _ in tasks
        ],
        "object_id": _target_object_and_geometry(tasks[0])[0].object_id,
        "module_ids": list(io.module_ids),
        "local_joint_ids": list(io.local_joint_ids),
        "command_local_joint_ids": list(policy_command_joint_ids(physical)),
        "rotor_global_ids": [
            f"module_{module_id}:{rotor.rotor_id}"
            for module_id in io.module_ids
            for rotor in sorted(physical.rotors, key=lambda value: value.rotor_id)
        ],
        "vectoring_global_joint_ids": [
            f"module_{module_id}:{rotor.vectoring_joint_ids[0]}"
            for module_id in io.module_ids
            for rotor in sorted(physical.rotors, key=lambda value: value.rotor_id)
        ],
        "reward_term_names": list(reward_names),
        "control_dt_s": float(args_cli.dt),
        "raw_contact_actor_input": False,
        "topology_randomized": bool(stage.topology_randomized),
        "collector_version": _COLLECTOR_VERSION,
        "selected_anchor_ids": list(io.selected_anchor_ids),
        "selected_gripper_friction": float(selected_friction),
        "contact_stiffness_n_per_m": float(args_cli.contact_stiffness),
        "contact_damping_n_s_per_m": float(args_cli.contact_damping),
        "canonical_phase_resets": bool(canonical_resets),
        "phase_reset_state_labels_reused": False,
        "phase_successor_reference_semantics": (
            ORDER9_PHASE_SUCCESSOR_REFERENCE_SEMANTICS
        ),
        "runtime_phase_labels": [
            phase.value for phase in ORDER9_OBJECT_TASK_PHASES
        ],
        "actor_phase_labels": list(ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS),
        "actor_phase_index_by_runtime": list(
            ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
        ),
        "phase_duration_s": dict(
            Order9ObjectTaskRuntimeConfig().phase_duration_s
        ),
        "evaluation_mode": bool(args_cli.evaluation_jsonl is not None),
        "deterministic_policy": bool(args_cli.evaluation_jsonl is not None),
        "initial_phase_zero": bool(args_cli.evaluation_jsonl is not None),
    }


def _actor_phase_indices(runtime_phase_indices: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(
        ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
        device=runtime_phase_indices.device,
        dtype=runtime_phase_indices.dtype,
    )
    return mapping[runtime_phase_indices.long()]


def _condition_target_on_teacher_reference(
    target,
    *,
    teacher_reference,
    phase_index: torch.Tensor,
    scene_origins: torch.Tensor,
    task_object_position_world: torch.Tensor,
):
    if teacher_reference is None:
        return target
    source_object_position = teacher_reference.desired_object_pose_world[0, 0, :3]
    task_offset = task_object_position_world - source_object_position
    sample = teacher_reference.sample(
        phase_index=phase_index,
        phase_progress=target.phase_progress,
        position_offset_world=scene_origins + task_offset.unsqueeze(0),
    )
    return replace(
        target,
        desired_robot_root_pose_world=sample.desired_body_pose_world,
        desired_robot_root_twist_world=sample.desired_body_twist,
        nominal_joint_positions_rad=sample.nominal_joint_positions_rad,
        nominal_joint_velocities_radps=sample.nominal_joint_velocities_radps,
        desired_object_pose_world=sample.desired_object_pose_world,
        phase_goal_robot_root_pose_world=sample.phase_goal_body_pose_world,
        phase_goal_object_pose_world=sample.phase_goal_object_pose_world,
    )


def _canonical_joint_reference_banks(
    runtime: Order9ObjectTaskRuntime,
    *,
    module_ids,
    joint_ids,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the deterministic active-knot posture reference."""

    starts = []
    ends = []
    for phase_index in range(runtime.phase_count):
        reset = runtime.reset_for_phase(phase_index)
        end = runtime.target(
            phase_index,
            runtime.duration_s(phase_index),
            reset=reset,
        )
        phase_start = []
        phase_end = []
        for module_id in module_ids:
            start_row = []
            end_row = []
            for joint_id in joint_ids:
                global_id = f"module_{module_id}:{joint_id}"
                if (
                    global_id not in reset.joint_positions_rad
                    or global_id not in end.nominal_joint_positions_rad
                ):
                    raise RuntimeError(
                        "Order9 deterministic posture reference does not cover "
                        f"{global_id}; arbitrary-morphology reference generation "
                        "must be supplied before this topology is trained"
                    )
                start_row.append(float(reset.joint_positions_rad[global_id]))
                end_row.append(float(end.nominal_joint_positions_rad[global_id]))
            phase_start.append(start_row)
            phase_end.append(end_row)
        starts.append(phase_start)
        ends.append(phase_end)
    return (
        torch.tensor(starts, device=device, dtype=dtype),
        torch.tensor(ends, device=device, dtype=dtype),
    )


def policy_command_joint_ids(physical) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in physical.dock_ports
            }
        )
    )


def _artifact_step(**values):
    pre_state = values["pre_state"]
    pre_target = values["pre_target"]
    step = values["policy_step"]
    command = step.policy_command
    allocation = step.controller_result.allocation
    reward = values["reward"]
    post = values["post_state"]
    return {
        "valid": values["valid"],
        "time_s": values["pre_time"],
        "phase_index": values["pre_phase"],
        "phase_progress": (
            values["pre_elapsed"] / values["duration"]
        ).clamp(0.0, 1.0),
        "episode_serial": values["pre_serial"],
        "step_index": values["pre_step"],
        "module_pose_world": pre_state.module_pose_world,
        "module_twist_world": pre_state.module_twist_world,
        "local_joint_positions_rad": pre_state.local_joint_positions_rad,
        "local_joint_velocities_radps": pre_state.local_joint_velocities_radps,
        "robot_root_pose_world": pre_state.robot_root_pose_world,
        "robot_root_twist_world": pre_state.robot_root_twist_world,
        "object_pose_world": pre_state.object_pose_world,
        "object_twist_world": pre_state.object_twist_world,
        "desired_body_pose_world": pre_target.desired_robot_root_pose_world,
        "desired_body_twist_reference": pre_target.desired_robot_root_twist_world,
        "desired_object_pose_world": pre_target.desired_object_pose_world,
        "phase_goal_body_pose_world": (
            pre_target.phase_goal_robot_root_pose_world
        ),
        "phase_goal_object_pose_world": pre_target.phase_goal_object_pose_world,
        "desired_joint_positions_rad": pre_target.nominal_joint_positions_rad,
        "desired_joint_velocities_radps": (
            pre_target.nominal_joint_velocities_radps
        ),
        "selected_assignment_mask": values["selected_mask"],
        "contact_schedule_index": pre_target.contact_schedule_index,
        "actor_controller_qp_feasible": step.actor_controller_qp_feasible,
        "actor_controller_status_one_hot": step.actor_controller_status_one_hot,
        "actor_allocation_residual_norm": step.actor_allocation_residual_norm,
        "actor_task_success": step.actor_task_success,
        "global_action": step.policy_step.action,
        "joint_action": step.policy_step.joint_action,
        "previous_global_action": step.previous_global_action,
        "recurrent_state_in": step.recurrent_state_in,
        "recurrent_state_out": step.policy_step.recurrent_state,
        "old_log_prob": step.policy_step.log_prob,
        "old_value": step.policy_step.value,
        "privileged_disturbance_body": step.privileged_disturbance_body,
        "command_body_pose_world": command.desired_body_pose_world,
        "command_body_twist": command.desired_body_twist,
        "command_residual_wrench_body": command.residual_wrench_body,
        "command_joint_position_targets_rad": command.joint_position_targets_rad,
        "command_joint_velocity_targets_radps": command.joint_velocity_targets_radps,
        "command_joint_torque_bias_nm": command.joint_torque_bias_nm,
        "controller_desired_wrench_body": step.controller_result.desired_wrench_body,
        "rotor_thrusts_n": allocation.rotor_thrusts_n,
        "vectoring_joint_targets_rad": allocation.vectoring_joint_targets_rad,
        "allocation_residual_norm": allocation.residual_norm,
        "qp_feasible": allocation.feasible,
        "rotor_saturation": values["rotor_saturation"],
        "selected_contact_forces_world": values[
            "contact"
        ].selected_contact_forces_world,
        "prohibited_collision": values["contact"].prohibited_collision,
        "reward": reward.reward,
        "reward_terms": torch.stack(
            [reward.terms[name] for name in values["reward_names"]], dim=-1
        ),
        "phase_success": reward.phase_success,
        "terminal": values["terminal"],
        "truncated": values["truncated"],
        "bootstrap_value": values["bootstrap"],
        "post_robot_root_pose_world": post.robot_root_pose_world,
        "post_robot_root_twist_world": post.robot_root_twist_world,
        "post_local_joint_positions_rad": post.local_joint_positions_rad,
        "post_local_joint_velocities_radps": post.local_joint_velocities_radps,
        "post_object_pose_world": post.object_pose_world,
        "post_object_twist_world": post.object_twist_world,
    }


def _reset_goal_distance_for_phase_transition(
    state: Order9TensorRewardState,
    *,
    env_ids: torch.Tensor,
    object_pose_world: torch.Tensor,
    desired_object_pose_world: torch.Tensor,
) -> Order9TensorRewardState:
    if env_ids.numel() == 0:
        return state
    values = {field.name: getattr(state, field.name).clone() for field in fields(state)}
    values["previous_object_goal_distance_m"][env_ids] = torch.linalg.vector_norm(
        object_pose_world[env_ids, :3]
        - desired_object_pose_world[env_ids, :3],
        dim=-1,
    )
    return Order9TensorRewardState(**values)


def _object_pose(obj) -> torch.Tensor:
    # Task/object poses are expressed at the object link/geometry frame.  The
    # separately randomized estimator CoM is transformed from this frame in
    # the controller; exposing PhysX's true CoM pose here would leak it.
    return _torch(obj.data.root_pose_w)


def _validate_object_mass_properties(
    obj,
    *,
    expected_mass_kg: float,
    expected_inertia_kgm2: tuple[float, ...],
    expected_com_object: tuple[float, ...],
) -> dict[str, object]:
    mass = _torch(obj.data.body_mass)[:, 0]
    inertia = _torch(obj.data.body_inertia)[:, 0].reshape(-1, 3, 3)
    com = _torch(obj.data.body_com_pose_b)[:, 0, :3]
    expected_inertia = torch.tensor(
        [
            [expected_inertia_kgm2[0], expected_inertia_kgm2[1], expected_inertia_kgm2[2]],
            [expected_inertia_kgm2[1], expected_inertia_kgm2[3], expected_inertia_kgm2[4]],
            [expected_inertia_kgm2[2], expected_inertia_kgm2[4], expected_inertia_kgm2[5]],
        ],
        device=inertia.device,
        dtype=inertia.dtype,
    )
    expected_eigenvalues = torch.linalg.eigvalsh(expected_inertia)
    actual_eigenvalues = torch.linalg.eigvalsh(inertia)
    expected_com = torch.tensor(
        expected_com_object,
        device=com.device,
        dtype=com.dtype,
    )
    if not torch.allclose(
        mass,
        torch.full_like(mass, float(expected_mass_kg)),
        rtol=5.0e-4,
        atol=1.0e-6,
    ):
        raise RuntimeError("Order9 Isaac object mass readback differs from TaskSpec")
    if not torch.allclose(
        actual_eigenvalues,
        expected_eigenvalues.reshape(1, 3).expand_as(actual_eigenvalues),
        rtol=5.0e-3,
        atol=1.0e-6,
    ):
        raise RuntimeError("Order9 Isaac object inertia readback differs from TaskSpec")
    if not torch.allclose(
        com,
        expected_com.reshape(1, 3).expand_as(com),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise RuntimeError("Order9 Isaac object CoM readback differs from TaskSpec")
    return {
        "mass_kg": float(mass[0].item()),
        "center_of_mass_object": [float(value) for value in com[0].tolist()],
        "inertia_eigenvalues_kgm2": [
            float(value) for value in actual_eigenvalues[0].tolist()
        ],
        "matches_task_spec": True,
    }


def _object_twist(obj) -> torch.Tensor:
    return torch.cat(
        (_torch(obj.data.root_lin_vel_w), _torch(obj.data.root_ang_vel_w)),
        dim=-1,
    )


def _torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _translate_pose(pose, translation):
    return (
        float(pose[0]) + translation[0],
        float(pose[1]) + translation[1],
        float(pose[2]) + translation[2],
        *tuple(float(value) for value in pose[3:7]),
    )


def _yaw(quaternion) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


_exit_code = 1
try:
    _result = main()
    print(_RESULT_PREFIX + json.dumps(_result, sort_keys=True), flush=True)
    _exit_code = 0
except BaseException as _error:
    print(
        "ORDER9_ROLLOUT_ERROR="
        + json.dumps(
            {"error_type": type(_error).__name__, "error": str(_error)},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
finally:
    simulation_app.close()
raise SystemExit(_exit_code)
