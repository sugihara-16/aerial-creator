#!/usr/bin/env python3
from __future__ import annotations

"""Exercise the production Order 9 shadow worker through a real PhysX step."""

import argparse
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.order9_low_level_runtime import Order9LowLevelRuntimePolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskRuntime
from amsrr.simulation.order9_object_task_state import (
    Order9IsaacStateSnapshot,
    load_order9_canonical_reset,
)
from amsrr.simulation.order9_shadow_worker import (
    JsonLineSubprocessShadowTransport,
    Order9ShadowStateExport,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pipeline import order9_schedule_hash
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    upgrade_teacher_trajectory_to_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/order9_learning_curriculum.yaml")
    parser.add_argument("--pi-l-checkpoint", required=True)
    parser.add_argument("--pi-l-checkpoint-sha256", required=True)
    parser.add_argument("--micromamba-env", default="isaaclab3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--worker-script", default="scripts/order9_isaac_shadow_worker.py"
    )
    parser.add_argument("--morphology-graph-json")
    parser.add_argument("--robot-usd")
    return parser


def _short_nominal_trajectory(
    context: HighLevelPolicyContext,
    *,
    dt_s: float,
) -> ContactWrenchTrajectory:
    legacy = GraspCarryBaselinePlanner().plan(context)
    upgraded = upgrade_teacher_trajectory_to_v2(legacy, context)
    source = next(
        knot
        for knot in upgraded.knots
        if any(
            assignment.schedule_state == "maintain"
            for assignment in knot.contact_assignments
        )
    )
    first = InteractionKnot.from_dict(source.to_dict())
    second = InteractionKnot.from_dict(source.to_dict())
    first.t_rel_s = 0.0
    second.t_rel_s = float(dt_s)
    trajectory = ContactWrenchTrajectory(
        horizon_s=float(dt_s),
        dt_s=float(dt_s),
        knots=[first, second],
        derived_mode_label="order9_real_isaac_shadow_smoke",
        contract_version=upgraded.contract_version,
    )
    trajectory.validate()
    return trajectory


def _worker_command(args: argparse.Namespace, *, repository: Path) -> list[str]:
    micromamba = shutil.which("micromamba")
    if micromamba is None:
        raise RuntimeError("micromamba executable is unavailable")
    command = [
        micromamba,
        "run",
        "-n",
        str(args.micromamba_env),
        "--",
        "python",
        str((repository / args.worker_script).resolve()),
        "--viz",
        "none",
        "--device",
        str(args.device),
        "--config",
        str((repository / args.config).resolve()),
        "--pi-l-checkpoint",
        str(Path(args.pi_l_checkpoint).resolve()),
        "--pi-l-checkpoint-sha256",
        str(args.pi_l_checkpoint_sha256),
        "--dt",
        str(args.dt),
    ]
    if args.morphology_graph_json:
        command.extend(
            [
                "--morphology-graph-json",
                str(Path(args.morphology_graph_json).resolve()),
            ]
        )
    if args.robot_usd:
        command.extend(["--robot-usd", str(Path(args.robot_usd).resolve())])
    return command


def main() -> int:
    args = _parser().parse_args()
    if args.dt <= 0.0 or args.timeout_s <= 0.0:
        raise ValueError("Order9 shadow smoke dt/timeout must be positive")
    repository = REPO_ROOT
    config = load_order9_learning_config(repository / args.config)
    config.validate()
    reset = load_order9_canonical_reset(
        repository / config.production_runtime.canonical_order8_report_path,
        expected_sha256=config.production_runtime.canonical_order8_report_sha256,
    )
    physical_model = build_physical_model_from_config(
        repository / config.production_runtime.robot_model_config_path
    )
    morphology = (
        MorphologyGraph.from_json(
            Path(args.morphology_graph_json).read_text(encoding="utf-8")
        )
        if args.morphology_graph_json
        else build_representative_order8_morphology(physical_model)
    )
    task = build_order8_grasp_carry_task_spec(
        object_pose_world=tuple(reset.object_pose_world),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=reset.transport_distance_m,
        support_height_m=config.randomization.support_top_z_m,
        max_contact_force_n=config.hard_checker.qp_force_scale_n,
        max_contact_torque_nm=config.hard_checker.qp_torque_scale_nm,
        selected_gripper_friction=(
            config.randomization.nominal_selected_gripper_friction
        ),
    )
    builder = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(builder.irg)
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=builder.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=builder.scene_graph.geometry_descriptors,
    )
    context = HighLevelPolicyContext(
        irg=builder.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        contact_candidate_set=candidates,
    )
    trajectory = _short_nominal_trajectory(context, dt_s=float(args.dt))
    policy = Order9LowLevelRuntimePolicy.from_checkpoint(
        args.pi_l_checkpoint,
        physical_model=physical_model,
        expected_sha256=args.pi_l_checkpoint_sha256,
        expected_schedule_hash=order9_schedule_hash(config),
        deterministic=True,
        device="cpu",
    )
    controller = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_qp",
            control_dt_s=float(args.dt),
        )
    )
    command = _worker_command(args, repository=repository)
    transport = JsonLineSubprocessShadowTransport(
        command,
        cwd=repository,
        timeout_s=float(args.timeout_s),
        environment={"PYTHONPATH": str(repository)},
    )
    try:
        description = dict(transport.request("describe", {}))
        descriptor = description.get("descriptor")
        if not isinstance(descriptor, dict):
            raise RuntimeError("Order9 worker describe response lacks a descriptor")
        scene = descriptor.get("scene")
        if not isinstance(scene, dict) or not isinstance(scene.get("joint_names"), list):
            raise RuntimeError("Order9 worker descriptor lacks joint names")
        joint_names = [str(value) for value in scene["joint_names"]]
        phase_reset = Order9ObjectTaskRuntime(reset).reset_for_phase(2)
        q_by_name = {
            key.replace(":", "__", 1): float(value)
            for key, value in phase_reset.joint_positions_rad.items()
        }
        qdot_by_name = {
            key.replace(":", "__", 1): float(value)
            for key, value in phase_reset.joint_velocities_radps.items()
        }
        snapshot = Order9IsaacStateSnapshot(
            simulation_time_s=0.0,
            robot_root_pose_world=list(phase_reset.robot_root_pose_world),
            robot_root_twist_world=list(phase_reset.robot_root_twist_world),
            joint_names=joint_names,
            joint_positions_rad=[q_by_name.get(name, 0.0) for name in joint_names],
            joint_velocities_radps=[qdot_by_name.get(name, 0.0) for name in joint_names],
            object_id="order8_object",
            object_pose_world=list(phase_reset.object_pose_world),
            object_twist_world=list(phase_reset.object_twist_world),
            phase_index=2,
            phase_elapsed_s=0.0,
            command_index=0,
            metadata={"source": "order9_real_isaac_shadow_smoke"},
        )
        snapshot.validate()
        topology_hash = morphology_structural_hash(morphology)
        if descriptor.get("topology_structural_hash") != topology_hash:
            raise RuntimeError("Order9 worker descriptor topology mismatch")
        state = Order9ShadowStateExport(
            state_id="order9-real-isaac-shadow-smoke-state",
            topology_structural_hash=topology_hash,
            simulation_time_s=0.0,
            simulation_state=snapshot.to_dict(),
            controller_state={
                "qpid": controller.export_runtime_state(),
                "trajectory_execution": {
                    "previous_controller_command": None,
                    "command_index": 0,
                },
            },
            pi_l_state=policy.export_runtime_state(),
            pi_l_checkpoint_sha256=args.pi_l_checkpoint_sha256,
            metadata={"smoke": True},
        )
        state.validate()
        synchronized = dict(
            transport.request(
                "synchronize",
                {
                    "state": state.to_dict(),
                    "state_digest": state.state_digest,
                    "pi_l_checkpoint_sha256": args.pi_l_checkpoint_sha256,
                },
            )
        )
        executed = dict(
            transport.request(
                "execute",
                {
                    "state_digest": state.state_digest,
                    "pi_l_checkpoint_sha256": args.pi_l_checkpoint_sha256,
                    "proposal_hash": trajectory.stable_hash(),
                    "trajectory": trajectory.to_dict(),
                    "context": {
                        "irg": context.irg.to_dict(),
                        "interaction_envelope": context.interaction_envelope.to_dict(),
                        "morphology_graph": context.morphology_graph.to_dict(),
                        "contact_candidate_set": context.contact_candidate_set.to_dict(),
                        "runtime_observation": None,
                    },
                },
            )
        )
        reset_result = dict(
            transport.request("reset", {"state_digest": state.state_digest})
        )
        observations = executed.get("observations")
        passed = bool(
            synchronized.get("accepted") is True
            and executed.get("accepted") is True
            and reset_result.get("accepted") is True
            and isinstance(observations, list)
            and len(observations) == len(trajectory.knots)
            and isinstance(scene.get("actuator_readback"), dict)
            and scene["actuator_readback"].get("matches_physical_model") is True
        )
        report = {
            "passed": passed,
            "checkpoint_sha256": args.pi_l_checkpoint_sha256,
            "state_digest": state.state_digest,
            "proposal_hash": trajectory.stable_hash(),
            "worker_version": description.get("worker_version"),
            "joint_count": len(joint_names),
            "candidate_count": len(candidates.candidates),
            "observation_count": len(observations) if isinstance(observations, list) else 0,
            "observations": observations,
            "actuator_readback": scene.get("actuator_readback"),
        }
        print(json.dumps(report, sort_keys=True))
        return 0 if passed else 1
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
