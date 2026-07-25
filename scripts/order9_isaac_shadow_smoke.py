#!/usr/bin/env python3
from __future__ import annotations

"""Exercise the production Order 9 shadow worker through a real PhysX step."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.articulated_reachability import (
    ArticulatedIKSolution,
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.order9_low_level_runtime import Order9LowLevelRuntimePolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.robot_model.urdf_loader import URDFModel, load_urdf
from amsrr.robot_model.urdf_transforms import (
    link_poses_at_joint_positions,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.task_spec import GeometryType, TaskSpec
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
    Order9ObjectTaskRuntime,
)
from amsrr.simulation.order9_object_task_state import (
    Order9IsaacStateSnapshot,
    load_order9_canonical_reset,
)
from amsrr.simulation.order9_production_hard_checker_runtime import (
    write_order9_shadow_bucket_morphology,
)
from amsrr.simulation.order9_shadow_worker import (
    JsonLineSubprocessShadowTransport,
    Order9ShadowStateExport,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherConfig,
    Order9ArticulatedTrajectoryTeacher,
)
from amsrr.training.order9_pipeline import order9_schedule_hash
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    upgrade_teacher_trajectory_to_v2,
)
from amsrr.utils.hashing import hash_file, stable_hash


_TRAJECTORY_HASH_DECIMAL_PLACES = 9
_GUI_MINIMUM_RPC_TIMEOUT_S = 300.0


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
    parser.add_argument(
        "--posture-fixture",
        help=(
            "Resolve and execute one raw pi_H case from an Order 9 posture "
            "fixture instead of the legacy one-step smoke trajectory."
        ),
    )
    parser.add_argument(
        "--posture-case-id",
        help="Case id selected from --posture-fixture.",
    )
    parser.add_argument(
        "--c2-bucket-manifest",
        help=(
            "Execute the joint-free articulated teacher and posture resolver "
            "from the exact phase-zero state of one immutable C2 bucket."
        ),
    )
    parser.add_argument(
        "--c2-bucket-id",
        help="Validation bucket selected from --c2-bucket-manifest.",
    )
    parser.add_argument(
        "--c2-articulated-urdf",
        help=(
            "Generated articulated URDF matching the C2 bucket USD. It "
            "converts the Isaac articulation root into exact module fc "
            "poses for rolling-teacher observations."
        ),
    )
    parser.add_argument(
        "--c2-rolling-window-count",
        type=int,
        default=1,
        help=(
            "Maximum measured-state replans. Execution stops at the current "
            "phase unless --c2-stop-after-phase requests a sequential task."
        ),
    )
    parser.add_argument(
        "--c2-stop-after-phase",
        choices=tuple(
            phase.value for phase in ORDER9_OBJECT_TASK_PHASES[:6]
        ),
        help=(
            "Advance sequentially from approach using measured real-Isaac "
            "phase gates, and stop after the named phase succeeds."
        ),
    )
    parser.add_argument(
        "--c2-exact-runtime-reference",
        action="store_true",
        help=(
            "Replace the articulated teacher with the original C2 approach "
            "reference, segmented into production-shadow horizons. This is a "
            "scalar execution-path regression diagnostic, not a C2 promotion."
        ),
    )
    parser.add_argument(
        "--c2-batched-rollout",
        help=(
            "Promoted C2 batched evaluation_rollout.pt used to recover the "
            "fixed-USD phase-zero centroidal frame for exact scalar replay."
        ),
    )
    parser.add_argument(
        "--qpid-only-tracking",
        action="store_true",
        help=(
            "Diagnostic only: bypass pi_L and contact/collision evidence so "
            "the C2 centroidal reference is applied directly to QPID."
        ),
    )
    parser.add_argument(
        "--bypass-pi-l",
        action="store_true",
        help=(
            "Diagnostic only: retain the articulated teacher, IK trajectory, "
            "contact evidence, and phase gates while applying its nominal "
            "PolicyCommand directly to QPID without learned pi_L corrections."
        ),
    )
    parser.add_argument(
        "--qpid-test-total-displacement-m",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help=(
            "Optional 30-second smoothstep displacement for the QPID-only "
            "diagnostic; omitted means the exact C2 +0.30 m x approach."
        ),
    )
    parser.add_argument(
        "--skip-mesh-collision-evidence",
        action="store_true",
        help=(
            "Retain Isaac contact evidence but skip the expensive diagnostic "
            "mesh-collision oracle."
        ),
    )
    parser.add_argument(
        "--gui-realtime",
        action="store_true",
        help=(
            "Open the Isaac Sim Kit viewer and play physics no faster than "
            "real time. This changes visualization/pacing only."
        ),
    )
    parser.add_argument(
        "--gui-keep-open-s",
        type=float,
        default=5.0,
        help=(
            "With --gui-realtime, keep the responsive Kit viewer open at the "
            "final state for this many seconds."
        ),
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for the complete machine-readable smoke report.",
    )
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


def _worker_command(
    args: argparse.Namespace,
    *,
    repository: Path,
    scene_arguments: dict[str, object] | None = None,
    morphology_graph_path: Path | None = None,
    robot_usd_path: Path | None = None,
) -> list[str]:
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
        "kit" if args.gui_realtime else "none",
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
    if args.gui_realtime:
        command.extend(
            [
                "--real-time-playback",
                "--keep-open-after-run-s",
                format(float(args.gui_keep_open_s), ".17g"),
            ]
        )
    if morphology_graph_path is not None:
        command.extend(
            ["--morphology-graph-json", str(morphology_graph_path.resolve())]
        )
    elif args.morphology_graph_json:
        command.extend(
            [
                "--morphology-graph-json",
                str(Path(args.morphology_graph_json).resolve()),
            ]
        )
    if robot_usd_path is not None:
        command.extend(["--robot-usd", str(robot_usd_path.resolve())])
    elif args.robot_usd:
        command.extend(["--robot-usd", str(Path(args.robot_usd).resolve())])
    if scene_arguments is not None:
        command.extend(
            [
                "--object-id",
                str(scene_arguments["object_id"]),
                "--object-geometry",
                str(scene_arguments["object_geometry"]),
                "--object-size",
                *(
                    format(float(value), ".17g")
                    for value in scene_arguments["object_size"]
                ),
                "--object-mass-kg",
                format(float(scene_arguments["object_mass_kg"]), ".17g"),
                "--object-friction",
                format(float(scene_arguments["object_friction"]), ".17g"),
                "--selected-gripper-friction",
                format(
                    float(scene_arguments["selected_gripper_friction"]),
                    ".17g",
                ),
                "--contact-stiffness",
                format(float(scene_arguments["contact_stiffness"]), ".17g"),
                "--contact-damping",
                format(float(scene_arguments["contact_damping"]), ".17g"),
                "--support-top-z",
                format(float(scene_arguments["support_top_z"]), ".17g"),
            ]
        )
    if args.qpid_only_tracking:
        command.extend(["--qpid-only", "--skip-contact-evidence"])
    elif args.bypass_pi_l:
        command.append("--qpid-only")
        if args.skip_mesh_collision_evidence:
            command.append("--skip-collision-evidence")
    elif args.skip_mesh_collision_evidence:
        command.append("--skip-collision-evidence")
    return command


def _task_scene_arguments(
    task: TaskSpec,
    *,
    selected_gripper_friction: float,
    contact_stiffness: float,
    contact_damping: float,
) -> dict[str, object]:
    if len(task.scene.objects) != 1:
        raise ValueError("Order9 shadow task requires exactly one object")
    target = task.scene.objects[0]
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == target.geometry_id
    )
    if (
        geometry.geometry_type is not GeometryType.BOX
        or geometry.primitive_params is None
        or target.mass_kg is None
        or target.friction is None
    ):
        raise ValueError("Order9 shadow task currently requires one dynamic box")
    size = geometry.primitive_params.get("size_m")
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError("Order9 shadow task box size is invalid")
    surfaces = task.scene.environment.support_surfaces
    if len(surfaces) != 1:
        raise ValueError("Order9 shadow task requires one support surface")
    support_geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == surfaces[0].geometry_id
    )
    support_size = (
        None
        if support_geometry.primitive_params is None
        else support_geometry.primitive_params.get("size_m")
    )
    if not isinstance(support_size, list) or len(support_size) != 3:
        raise ValueError("Order9 shadow task support size is invalid")
    return {
        "object_id": target.object_id,
        "object_geometry": geometry.geometry_type.value,
        "object_size": tuple(float(value) for value in size),
        "object_mass_kg": float(target.mass_kg),
        "object_friction": float(target.friction),
        "selected_gripper_friction": float(selected_gripper_friction),
        "contact_stiffness": float(contact_stiffness),
        "contact_damping": float(contact_damping),
        "support_top_z": float(surfaces[0].pose_world[2])
        + 0.5 * float(support_size[2]),
    }


def _task_posture_collision_object(
    task: TaskSpec,
) -> Order9PostureCollisionObject:
    """Bind the posture resolver to the task-declared dynamic box."""

    target = task.scene.objects[0]
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == target.geometry_id
    )
    size = (
        None
        if geometry.primitive_params is None
        else geometry.primitive_params.get("size_m")
    )
    if geometry.geometry_type is not GeometryType.BOX or not (
        isinstance(size, list) and len(size) == 3
    ):
        raise ValueError(
            "Order9 posture collision gate currently requires one box object"
        )
    return Order9PostureCollisionObject(
        object_id=target.object_id,
        size_m=tuple(float(value) for value in size),
        initial_pose_world=tuple(float(value) for value in target.pose_world),
    )


def _posture_fixture_inputs(
    path: Path,
    *,
    case_id: str,
    physical_model: object,
    contact_stiffness: float,
    contact_damping: float,
) -> tuple[
    TaskSpec,
    MorphologyGraph,
    HighLevelPolicyContext,
    ContactWrenchTrajectory,
    dict[str, object],
    dict[str, object],
]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("Order9 posture fixture must be a mapping")
    if fixture.get("physical_model_hash") != physical_model.stable_hash():
        raise ValueError("Order9 posture fixture PhysicalModel hash differs")
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Order9 posture fixture cases are missing")
    matches = [
        value
        for value in cases
        if isinstance(value, dict) and value.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Order9 posture fixture does not contain one {case_id!r} case"
        )
    case = matches[0]
    task = TaskSpec.from_dict(fixture["task_spec"])
    morphology = MorphologyGraph.from_dict(case["morphology_graph"])
    candidates = ContactCandidateSet.from_dict(case["contact_candidate_set"])
    raw = ContactWrenchTrajectory.from_dict(case["raw_pi_h_trajectory"])
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    context = HighLevelPolicyContext(
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        contact_candidate_set=candidates,
    )
    resolver_values = fixture["resolver_config"]
    resolution = Order9PostureTrajectoryResolver(
        physical_model,
        config=Order9PostureResolverConfig(
            output_rate_hz=float(resolver_values["output_rate_hz"]),
            joint_velocity_fraction=float(
                resolver_values["joint_velocity_fraction"]
            ),
        ),
    ).resolve(
        context=context,
        raw_trajectory=raw,
        initial_joint_positions_rad=case["initial_joint_positions_rad"],
    )
    if resolution.evidence.raw_trajectory_hash != case[
        "expected_raw_trajectory_hash"
    ]:
        raise RuntimeError("Order9 posture fixture raw pi_H hash differs")
    exact_resolved_hash_match = (
        resolution.evidence.resolved_trajectory_hash
        == case["expected_resolved_trajectory_hash"]
    )
    resolved_quantized_hash = stable_hash(
        _quantize_floats(
            resolution.trajectory.to_dict(),
            decimal_places=_TRAJECTORY_HASH_DECIMAL_PLACES,
        )
    )
    quantized_resolved_hash_match = (
        resolved_quantized_hash
        == case.get("expected_resolved_trajectory_quantized_hash")
    )
    if not exact_resolved_hash_match and not quantized_resolved_hash_match:
        raise RuntimeError(
            "Order9 posture fixture resolved trajectory hash differs"
        )
    material = task.metadata.get("contact_material_semantics", {})
    if not isinstance(material, dict):
        raise ValueError("Order9 posture fixture contact material is invalid")
    selected_friction = material.get("robot_static_friction")
    if selected_friction is None:
        raise ValueError(
            "Order9 posture fixture lacks selected gripper friction"
        )
    scene_arguments = _task_scene_arguments(
        task,
        selected_gripper_friction=float(selected_friction),
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
    )
    evidence = {
        "fixture_path": str(path.resolve()),
        "fixture_sha256": hash_file(path),
        "case_id": case_id,
        "raw_pi_h_trajectory_hash": raw.stable_hash(),
        "resolved_trajectory_hash": resolution.trajectory.stable_hash(),
        "resolved_trajectory_quantized_hash": resolved_quantized_hash,
        "exact_resolved_trajectory_hash_match": exact_resolved_hash_match,
        "quantized_resolved_trajectory_hash_match": (
            quantized_resolved_hash_match
        ),
        "resolver_evidence": resolution.evidence.to_dict(),
        "task_hash": task.stable_hash(),
    }
    return (
        task,
        morphology,
        context,
        resolution.trajectory,
        scene_arguments,
        evidence,
    )


def _yaw_from_pose(pose: tuple[float, ...] | list[float]) -> float:
    qx, qy, qz, qw = (float(value) for value in pose[3:7])
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _c2_phase_zero_reset(reset: object, task: TaskSpec):
    target = task.scene.objects[0]
    position_offset = tuple(
        float(target.pose_world[index])
        - float(reset.object_pose_world[index])
        for index in range(3)
    )
    yaw_offset = _yaw_from_pose(target.pose_world) - _yaw_from_pose(
        reset.object_pose_world
    )
    return Order9ObjectTaskRuntime(reset).reset_for_phase(
        0,
        object_position_offset_world=position_offset,
        object_yaw_offset_rad=yaw_offset,
    )


def _centroidal_translation_delta(
    trajectory: ContactWrenchTrajectory,
) -> list[float] | None:
    first = trajectory.knots[0].centroidal_target
    last = trajectory.knots[-1].centroidal_target
    if (
        first is None
        or last is None
        or first.com_pos_world is None
        or last.com_pos_world is None
    ):
        return None
    return [
        float(last.com_pos_world[index])
        - float(first.com_pos_world[index])
        for index in range(3)
    ]


def _runtime_observation_from_phase_zero(
    *,
    morphology: MorphologyGraph,
    task: TaskSpec,
    phase_reset: object,
    articulated_urdf: URDFModel,
) -> RuntimeObservation:
    module_root_poses = link_poses_at_joint_positions(
        articulated_urdf,
        {
            key.replace(":", "__", 1): float(value)
            for key, value in phase_reset.joint_positions_rad.items()
        },
        root_pose_world=tuple(phase_reset.robot_root_pose_world),
    )
    module_states = []
    for module in sorted(morphology.modules, key=lambda value: value.module_id):
        prefix = f"module_{module.module_id}:"
        joint_positions = {
            key[len(prefix) :]: float(value)
            for key, value in phase_reset.joint_positions_rad.items()
            if key.startswith(prefix)
        }
        joint_velocities = {
            key[len(prefix) :]: float(value)
            for key, value in phase_reset.joint_velocities_radps.items()
            if key.startswith(prefix)
        }
        module_states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module_root_poses[
                    f"module_{module.module_id}__fc"
                ],
                twist_world=[0.0] * 6,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
            )
        )
    observation = RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=module_states,
        object_states=[
            ObjectRuntimeState(
                object_id=task.scene.objects[0].object_id,
                pose_world=tuple(phase_reset.object_pose_world),
                twist_world=list(phase_reset.object_twist_world),
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="approach",
            progress_ratio=0.0,
        ),
    )
    observation.validate()
    return observation


class _C2CanonicalContactPostureSolver:
    """Expose the promoted C2 contact posture as a deterministic teacher target."""

    def __init__(self, solution: ArticulatedIKSolution) -> None:
        self.solution = solution

    def solve(self, **_kwargs: object) -> ArticulatedIKSolution:
        return self.solution


def _c2_canonical_contact_solution(
    *,
    reset: object,
    task: TaskSpec,
    morphology: MorphologyGraph,
    context: HighLevelPolicyContext,
    physical_model: object,
    articulated_urdf: URDFModel,
) -> ArticulatedIKSolution:
    """Reconstruct the physically promoted C2 final grasp posture in world space."""

    target = task.scene.objects[0]
    position_offset = tuple(
        float(target.pose_world[index])
        - float(reset.object_pose_world[index])
        for index in range(3)
    )
    yaw_offset = _yaw_from_pose(target.pose_world) - _yaw_from_pose(
        reset.object_pose_world
    )
    runtime = Order9ObjectTaskRuntime(reset)
    contact_phase_index = tuple(
        phase.value for phase in ORDER9_OBJECT_TASK_PHASES
    ).index(Order9ObjectTaskPhase.CONTACT_ACQUISITION.value)
    contact_reset = runtime.reset_for_phase(
        contact_phase_index,
        object_position_offset_world=position_offset,
        object_yaw_offset_rad=yaw_offset,
    )
    contact_target = runtime.target(
        contact_phase_index,
        runtime.duration_s(contact_phase_index),
        reset=contact_reset,
    )
    q_close = dict(contact_target.nominal_joint_positions_rad)
    module_link_poses = link_poses_at_joint_positions(
        articulated_urdf,
        {
            key.replace(":", "__", 1): float(value)
            for key, value in q_close.items()
        },
        root_pose_world=tuple(contact_target.desired_robot_root_pose_world),
    )
    base_pose = tuple(
        module_link_poses[
            f"module_{morphology.base_module_id}__fc"
        ]
    )
    selected_anchor_ids = tuple(
        sorted(anchor.anchor_id for anchor in morphology.robot_anchors)
    )
    references = resolve_mesh_backed_anchor_references(
        morphology,
        physical_model,
        selected_anchor_ids,
    )
    fk = WholeStructureKinematics().forward(
        morphology,
        physical_model,
        q_close,
        base_pose,
        references,
    )
    source_observation = context.runtime_observation
    if source_observation is None:
        raise RuntimeError(
            "Order9 C2 canonical contact solution requires runtime state"
        )
    module_states = []
    for module in sorted(morphology.modules, key=lambda value: value.module_id):
        prefix = f"module_{module.module_id}:"
        module_states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=fk.module_root_poses_world[module.module_id],
                twist_world=[0.0] * 6,
                joint_positions={
                    key[len(prefix) :]: float(value)
                    for key, value in q_close.items()
                    if key.startswith(prefix)
                },
                joint_velocities={
                    key[len(prefix) :]: 0.0
                    for key in q_close
                    if key.startswith(prefix)
                },
            )
        )
    final_observation = RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=module_states,
        object_states=[
            ObjectRuntimeState.from_dict(value.to_dict())
            for value in source_observation.object_states
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label=Order9ObjectTaskPhase.CONTACT_ACQUISITION.value,
            progress_ratio=1.0,
        ),
    )
    final_observation.validate()
    centroidal_pose = RigidBodyControlModelBuilder().build(
        morphology,
        physical_model,
        final_observation,
    ).body_pose_world
    return ArticulatedIKSolution(
        feasible=True,
        joint_positions_rad=q_close,
        base_pose_world=base_pose,
        centroidal_pose_world=tuple(centroidal_pose),
        anchor_poses_world=dict(fk.anchor_poses_world),
        maximum_position_error_m=0.0,
        maximum_normal_error_rad=0.0,
        iterations=0,
        solver_version="c2_promoted_contact_posture_reference_v1",
    )


def _c2_bucket_inputs(
    manifest_path: Path,
    *,
    bucket_id: str,
    repository: Path,
    physical_model: object,
    reset: object,
    articulated_urdf_path: Path,
    exact_runtime_reference: bool = False,
    batched_rollout_path: Path | None = None,
    qpid_test_total_displacement_m: tuple[float, float, float] | None = None,
) -> tuple[
    TaskSpec,
    MorphologyGraph,
    HighLevelPolicyContext,
    ContactWrenchTrajectory,
    dict[str, object],
    dict[str, object],
    object,
    URDFModel,
    Path,
    Path,
    ArticulatedIKSolution | None,
]:
    validate_order9_pi_l_rollout_bucket_bytes(
        manifest_path,
        repository_root=repository,
    )
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    matches = [
        bucket for bucket in manifest.buckets if bucket.bucket_id == bucket_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Order9 C2 manifest does not contain one {bucket_id!r} bucket"
        )
    bucket = matches[0]
    if bucket.split.value != "validation" or bucket.module_count != 3:
        raise ValueError(
            "Order9 C2 regression requires a fixed three-module validation bucket"
        )
    source = (
        manifest_path / "manifest.json"
        if manifest_path.is_dir()
        else manifest_path
    ).resolve()
    task_path = (source.parent / bucket.task_spec_path).resolve()
    graph_path = (source.parent / bucket.morphology_graph_path).resolve()
    robot_usd = Path(bucket.robot_usd_path)
    if not robot_usd.is_absolute():
        robot_usd = (repository / robot_usd).resolve()
    task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
    morphology = MorphologyGraph.from_json(
        graph_path.read_text(encoding="utf-8")
    )
    articulated_urdf_path = articulated_urdf_path.resolve()
    if not articulated_urdf_path.is_file():
        raise ValueError(
            "Order9 C2 generated articulated URDF is unavailable"
        )
    articulated_urdf = load_urdf(articulated_urdf_path)
    phase_reset = _c2_phase_zero_reset(reset, task)
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
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        contact_candidate_set=candidates,
        runtime_observation=_runtime_observation_from_phase_zero(
            morphology=morphology,
            task=task,
            phase_reset=phase_reset,
            articulated_urdf=articulated_urdf,
        ),
    )
    scene_arguments = _task_scene_arguments(
        task,
        selected_gripper_friction=bucket.selected_gripper_friction,
        contact_stiffness=bucket.contact_stiffness_n_per_m,
        contact_damping=bucket.contact_damping_n_s_per_m,
    )
    current_centroidal_pose = RigidBodyControlModelBuilder().build(
        morphology,
        physical_model,
        context.runtime_observation,
    ).body_pose_world
    if exact_runtime_reference:
        if batched_rollout_path is None:
            raise ValueError(
                "Order9 exact C2 replay requires its batched rollout artifact"
            )
        reference_centroidal_pose = _c2_batched_reference_start_pose(
            batched_rollout_path,
            morphology=morphology,
            phase_reset=phase_reset,
        )
        trajectory = _c2_exact_runtime_reference_segment(
            context,
            reset=reset,
            phase_reset=phase_reset,
            centroidal_start_pose_world=reference_centroidal_pose,
            segment_start_s=0.0,
            segment_horizon_s=3.0,
            dt_s=0.02,
            total_displacement_world_m=qpid_test_total_displacement_m,
        )
        raw_trajectory_hash = None
        posture_resolution_evidence = None
        teacher_phase_target_reached = False
        teacher_task_phase = "approach"
        teacher_candidate_group_id = None
        teacher_terminal_posture_source = None
        c2_contact_solution = None
        reference_semantics = "exact_c2_tensor_runtime_approach_reference"
    else:
        reference_centroidal_pose = current_centroidal_pose
        c2_contact_solution = _c2_canonical_contact_solution(
            reset=reset,
            task=task,
            morphology=morphology,
            context=context,
            physical_model=physical_model,
            articulated_urdf=articulated_urdf,
        )
        plan = Order9ArticulatedTrajectoryTeacher(
            physical_model,
            config=Order9ArticulatedTeacherConfig(
                preferred_candidate_group_id="slot_0:grasp_pair:2",
                contact_joint_speed_limit_rad_s=0.005,
                posture_anchor_position_tolerance_m=0.0005,
            ),
            ik_solver=_C2CanonicalContactPostureSolver(
                c2_contact_solution
            ),
            collision_object=_task_posture_collision_object(task),
        ).plan(
            context,
            initial_object_poses_world={
                target.object_id: target.pose_world
                for target in task.scene.objects
            },
        )
        if any(
            knot.posture_target is not None
            and (
                knot.posture_target.joint_pos_target is not None
                or knot.posture_target.joint_vel_target is not None
            )
            for knot in plan.raw_trajectory.knots
        ):
            raise RuntimeError("Order9 C2 raw pi_H teacher emitted joint targets")
        trajectory = plan.trajectory
        raw_trajectory_hash = plan.raw_trajectory.stable_hash()
        posture_resolution_evidence = (
            plan.posture_resolution.evidence.to_dict()
        )
        teacher_phase_target_reached = plan.phase_target_reached
        teacher_task_phase = plan.task_phase
        teacher_candidate_group_id = plan.candidate_group_id
        teacher_terminal_posture_source = (
            "c2_promoted_contact_posture_then_native_trajectory_ik"
        )
        reference_semantics = "articulated_teacher_receding_horizon"
    first_target = trajectory.knots[0].centroidal_target
    initial_target_distance = None
    if first_target is not None and first_target.com_pos_world is not None:
        initial_target_distance = math.sqrt(
            sum(
                (
                    float(first_target.com_pos_world[index])
                    - float(current_centroidal_pose[index])
                )
                ** 2
                for index in range(3)
            )
        )
    evidence = {
        "manifest_path": str(source),
        "manifest_sha256": hash_file(source),
        "bucket_id": bucket.bucket_id,
        "bucket_seed": bucket.seed,
        "task_hash": task.stable_hash(),
        "morphology_hash": morphology.stable_hash(),
        "raw_pi_h_trajectory_hash": raw_trajectory_hash,
        "resolved_trajectory_hash": trajectory.stable_hash(),
        "resolved_posture_knot_count": len(trajectory.knots),
        "posture_resolution_evidence": posture_resolution_evidence,
        "initial_target_translation_distance_m": initial_target_distance,
        "current_centroidal_pose_world": list(current_centroidal_pose),
        "c2_reference_centroidal_start_pose_world": list(
            reference_centroidal_pose
        ),
        "c2_reference_kinematic_translation_correction_world_m": [
            float(reference_centroidal_pose[index])
            - float(current_centroidal_pose[index])
            for index in range(3)
        ],
        "c2_batched_rollout_path": (
            None
            if batched_rollout_path is None
            else str(batched_rollout_path.resolve())
        ),
        "c2_reference_semantics": reference_semantics,
        "qpid_test_total_displacement_world_m": (
            None
            if qpid_test_total_displacement_m is None
            else list(qpid_test_total_displacement_m)
        ),
        "teacher_task_phase": teacher_task_phase,
        "teacher_candidate_group_id": teacher_candidate_group_id,
        "teacher_terminal_posture_source": teacher_terminal_posture_source,
        "teacher_phase_target_reached": teacher_phase_target_reached,
        "teacher_centroidal_translation_delta_world_m": (
            _centroidal_translation_delta(trajectory)
        ),
        "initial_state_source": "c2_canonical_phase_zero_with_bucket_offset",
        "runtime_observation_source": "generated_articulated_urdf_fk",
        "generated_articulated_urdf_path": str(articulated_urdf_path),
        "generated_articulated_urdf_sha256": hash_file(
            articulated_urdf_path
        ),
    }
    return (
        task,
        morphology,
        context,
        trajectory,
        scene_arguments,
        evidence,
        phase_reset,
        articulated_urdf,
        graph_path,
        robot_usd,
        c2_contact_solution,
    )


def _c2_batched_reference_start_pose(
    rollout_path: Path,
    *,
    morphology: MorphologyGraph,
    phase_reset: object,
) -> tuple[float, ...]:
    """Recover the exact C2 fixed-USD centroidal frame from promotion evidence."""

    import torch

    payload = torch.load(
        rollout_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Order9 C2 batched rollout payload is invalid")
    metadata = payload.get("metadata")
    tensors = payload.get("tensors")
    if not isinstance(metadata, dict) or not isinstance(tensors, dict):
        raise ValueError("Order9 C2 batched rollout lacks metadata/tensors")
    source_graph = metadata.get("morphology_graph")
    if (
        not isinstance(source_graph, dict)
        or MorphologyGraph.from_dict(source_graph).stable_hash()
        != morphology.stable_hash()
    ):
        raise ValueError("Order9 C2 batched rollout morphology differs")
    required = (
        "valid",
        "phase_index",
        "episode_serial",
        "phase_progress",
        "robot_root_pose_world",
        "desired_body_pose_world",
    )
    if any(name not in tensors for name in required):
        raise ValueError("Order9 C2 batched rollout lacks phase-zero tensors")
    valid = tensors["valid"]
    if valid.ndim != 2 or valid.shape[1] < 1:
        raise ValueError("Order9 C2 batched rollout valid mask is invalid")
    mask = (
        valid[:, 0]
        & (tensors["phase_index"][:, 0] == 0)
        & (tensors["episode_serial"][:, 0] == 0)
    )
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() < 1:
        raise ValueError("Order9 C2 batched rollout has no phase-zero row")
    start_index = int(indices[0].item())
    if abs(float(tensors["phase_progress"][start_index, 0].item())) > 1.0e-8:
        raise ValueError("Order9 C2 batched rollout does not begin at zero progress")
    artifact_root = tensors["robot_root_pose_world"][
        start_index, 0
    ].tolist()
    artifact_target = tensors["desired_body_pose_world"][
        start_index, 0
    ].tolist()
    origin = [
        float(artifact_root[index])
        - float(phase_reset.robot_root_pose_world[index])
        for index in range(3)
    ]
    return (
        *tuple(
            float(artifact_target[index]) - origin[index]
            for index in range(3)
        ),
        *tuple(float(value) for value in artifact_target[3:7]),
    )


def _c2_exact_runtime_reference_segment(
    context: HighLevelPolicyContext,
    *,
    reset: object,
    phase_reset: object,
    centroidal_start_pose_world: tuple[float, ...] | list[float],
    segment_start_s: float,
    segment_horizon_s: float,
    dt_s: float,
    total_displacement_world_m: (
        tuple[float, float, float] | None
    ) = None,
) -> ContactWrenchTrajectory:
    """Encode the original C2 phase-zero tensor target as scalar knots."""

    if segment_start_s < 0.0 or segment_horizon_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("Order9 C2 reference segment timing is invalid")
    runtime = Order9ObjectTaskRuntime(reset)
    duration_s = runtime.duration_s(0)
    if segment_start_s + segment_horizon_s > duration_s + 1.0e-9:
        raise ValueError("Order9 C2 reference segment exceeds approach duration")
    upgraded = upgrade_teacher_trajectory_to_v2(
        GraspCarryBaselinePlanner().plan(context),
        context,
    )
    source = next(
        knot
        for knot in upgraded.knots
        if all(
            assignment.schedule_state == "approach"
            for assignment in knot.contact_assignments
        )
    )
    start_pose = tuple(float(value) for value in centroidal_start_pose_world)
    if len(start_pose) != 7:
        raise ValueError("Order9 C2 centroidal start pose must contain seven values")
    step_count = int(round(segment_horizon_s / dt_s))
    if not math.isclose(
        step_count * dt_s,
        segment_horizon_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("Order9 C2 reference horizon must be divisible by dt")
    knots: list[InteractionKnot] = []
    for step_index in range(step_count + 1):
        local_time_s = step_index * dt_s
        phase_time_s = segment_start_s + local_time_s
        target = runtime.target(0, phase_time_s, reset=phase_reset)
        if total_displacement_world_m is None:
            displacement = tuple(
                float(target.desired_robot_root_pose_world[index])
                - float(phase_reset.robot_root_pose_world[index])
                for index in range(3)
            )
            linear_velocity = tuple(
                float(value)
                for value in target.desired_robot_root_twist_world[:3]
            )
        else:
            progress = min(max(phase_time_s / duration_s, 0.0), 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            derivative = 6.0 * progress * (1.0 - progress)
            displacement = tuple(
                float(value) * smooth
                for value in total_displacement_world_m
            )
            linear_velocity = tuple(
                float(value) / duration_s * derivative
                for value in total_displacement_world_m
            )
        knot = InteractionKnot.from_dict(source.to_dict())
        knot.t_rel_s = local_time_s
        knot.centroidal_target = CentroidalTarget(
            com_pos_world=tuple(
                start_pose[index] + displacement[index]
                for index in range(3)
            ),
            com_vel_world=linear_velocity,
            body_orientation_world=tuple(start_pose[3:7]),
            centroidal_wrench_preference=[0.0] * 6,
        )
        source_posture = knot.posture_target
        knot.posture_target = PostureTarget(
            joint_pos_target=dict(target.nominal_joint_positions_rad),
            joint_vel_target=dict(target.nominal_joint_velocities_radps),
            free_anchor_pose_targets=(
                None
                if source_posture is None
                else dict(source_posture.free_anchor_pose_targets or {})
            ),
        )
        knot.object_targets = [
            ObjectTarget(
                object_id=context.runtime_observation.object_states[0].object_id,
                pose_target_world=target.desired_object_pose_world,
            )
        ]
        knot.priority_weights = {
            **knot.priority_weights,
            "c2_exact_runtime_reference": 1.0,
        }
        knot.guard_conditions = [
            {
                "type": "order9_c2_scalar_regression",
                "phase_elapsed_s": phase_time_s,
            }
        ]
        knots.append(knot)
    trajectory = ContactWrenchTrajectory(
        horizon_s=float(segment_horizon_s),
        dt_s=float(dt_s),
        knots=knots,
        derived_mode_label=(
            (
                "order9_qpid_only_3d_tracking_reference:"
                if total_displacement_world_m is not None
                else "order9_c2_exact_runtime_reference:"
            )
            + f"phase_start={segment_start_s:.9g}"
        ),
        contract_version=upgraded.contract_version,
    )
    trajectory.validate()
    return trajectory


def _c2_replan_from_endpoint(
    endpoint_observation: dict[str, object],
    *,
    context: HighLevelPolicyContext,
    task: TaskSpec,
    physical_model: object,
    articulated_urdf: URDFModel,
    contact_solution: ArticulatedIKSolution,
    phase_label: str | None = None,
    nominal_start_joint_positions_rad: Mapping[str, float] | None = None,
    release_joint_positions_rad: Mapping[str, float] | None = None,
) -> tuple[
    HighLevelPolicyContext,
    ContactWrenchTrajectory,
    dict[str, object],
]:
    metrics = endpoint_observation.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("Order9 C2 rolling endpoint lacks metrics")

    def vector(prefix: str, size: int) -> tuple[float, ...]:
        values = []
        for index in range(size):
            value = metrics.get(f"endpoint.{prefix}_{index}")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeError(
                    f"Order9 C2 rolling endpoint lacks {prefix}_{index}"
                )
            values.append(float(value))
        return tuple(values)

    global_q = {
        key.removeprefix("endpoint.joint_position_rad.").replace(
            "__", ":", 1
        ): float(value)
        for key, value in metrics.items()
        if key.startswith("endpoint.joint_position_rad.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    global_qdot = {
        key.removeprefix("endpoint.joint_velocity_radps.").replace(
            "__", ":", 1
        ): float(value)
        for key, value in metrics.items()
        if key.startswith("endpoint.joint_velocity_radps.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    ordered_dock_ids = ordered_global_dock_joint_ids(
        context.morphology_graph,
        physical_model,
    )
    missing = [
        joint_id for joint_id in ordered_dock_ids if joint_id not in global_q
    ]
    if missing:
        raise RuntimeError(
            "Order9 C2 rolling endpoint lacks Dock joints "
            f"{missing}"
        )
    root_pose = vector("robot_root_pose_world", 7)
    root_twist = vector("robot_root_twist_world", 6)
    object_pose = vector("object_pose_world", 7)
    object_twist = vector("object_twist_world", 6)
    exact_module_state_available = all(
        all(
            isinstance(
                metrics.get(
                    "endpoint.module_pose_world."
                    f"module_{module.module_id}_{index}"
                ),
                (int, float),
            )
            and not isinstance(
                metrics.get(
                    "endpoint.module_pose_world."
                    f"module_{module.module_id}_{index}"
                ),
                bool,
            )
            for index in range(7)
        )
        and all(
            isinstance(
                metrics.get(
                    "endpoint.module_twist_world."
                    f"module_{module.module_id}_{index}"
                ),
                (int, float),
            )
            and not isinstance(
                metrics.get(
                    "endpoint.module_twist_world."
                    f"module_{module.module_id}_{index}"
                ),
                bool,
            )
            for index in range(6)
        )
        for module in context.morphology_graph.modules
    )
    urdf_module_poses = (
        None
        if exact_module_state_available
        else link_poses_at_joint_positions(
            articulated_urdf,
            {
                key.replace(":", "__", 1): float(value)
                for key, value in global_q.items()
            },
            root_pose_world=root_pose,
        )
    )
    module_states = []
    for module in sorted(
        context.morphology_graph.modules,
        key=lambda value: value.module_id,
    ):
        prefix = f"module_{module.module_id}:"
        if exact_module_state_available:
            module_pose = tuple(
                float(
                    metrics[
                        "endpoint.module_pose_world."
                        f"module_{module.module_id}_{index}"
                    ]
                )
                for index in range(7)
            )
            module_twist = [
                float(
                    metrics[
                        "endpoint.module_twist_world."
                        f"module_{module.module_id}_{index}"
                    ]
                )
                for index in range(6)
            ]
        else:
            if urdf_module_poses is None:
                raise RuntimeError(
                    "Order9 C2 endpoint URDF FK is unavailable"
                )
            module_pose = urdf_module_poses[
                f"module_{module.module_id}__fc"
            ]
            module_twist = list(root_twist)
        module_states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module_pose,
                twist_world=module_twist,
                joint_positions={
                    key[len(prefix) :]: value
                    for key, value in global_q.items()
                    if key.startswith(prefix)
                },
                joint_velocities={
                    key[len(prefix) :]: value
                    for key, value in global_qdot.items()
                    if key.startswith(prefix)
                },
            )
        )
    phase = phase_label
    if phase is None:
        phase = (
            "approach"
            if context.runtime_observation is None
            or context.runtime_observation.task_progress.phase_label is None
            else str(context.runtime_observation.task_progress.phase_label)
        )
    observation = RuntimeObservation(
        time_s=float(
            metrics.get("endpoint.elapsed_s", 0.0)
        ),
        morphology_graph=context.morphology_graph,
        module_states=module_states,
        object_states=[
            ObjectRuntimeState(
                object_id=task.scene.objects[0].object_id,
                pose_world=object_pose,
                twist_world=list(object_twist),
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label=phase,
            progress_ratio=0.0,
        ),
    )
    observation.validate()
    replanned_context = HighLevelPolicyContext(
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        morphology_graph=context.morphology_graph,
        contact_candidate_set=context.contact_candidate_set,
        runtime_observation=observation,
    )
    plan = Order9ArticulatedTrajectoryTeacher(
        physical_model,
        config=Order9ArticulatedTeacherConfig(
            preferred_candidate_group_id="slot_0:grasp_pair:2",
            contact_joint_speed_limit_rad_s=0.005,
            posture_anchor_position_tolerance_m=0.0005,
        ),
        ik_solver=_C2CanonicalContactPostureSolver(contact_solution),
        collision_object=_task_posture_collision_object(task),
    ).plan(
        replanned_context,
        initial_object_poses_world={
            value.object_id: value.pose_world for value in task.scene.objects
        },
        nominal_start_joint_positions_rad=(
            nominal_start_joint_positions_rad
        ),
        release_joint_positions_rad=release_joint_positions_rad,
    )
    return (
        replanned_context,
        plan.trajectory,
        {
            "teacher_task_phase": plan.task_phase,
            "teacher_candidate_group_id": plan.candidate_group_id,
            "teacher_terminal_posture_source": (
                "c2_promoted_contact_posture_then_native_trajectory_ik"
            ),
            "teacher_phase_target_reached": plan.phase_target_reached,
            "teacher_centroidal_translation_delta_world_m": (
                _centroidal_translation_delta(plan.trajectory)
            ),
            "raw_pi_h_trajectory_hash": (
                plan.raw_trajectory.stable_hash()
            ),
            "resolved_trajectory_hash": plan.trajectory.stable_hash(),
            "resolved_posture_knot_count": len(plan.trajectory.knots),
            "runtime_observation_source": (
                "direct_isaac_module_body_state"
                if exact_module_state_available
                else "generated_articulated_urdf_fk"
            ),
            "posture_resolution_evidence": (
                plan.posture_resolution.evidence.to_dict()
            ),
        },
    )


def _terminal_joint_reference(
    trajectory: ContactWrenchTrajectory,
) -> dict[str, float]:
    posture = trajectory.knots[-1].posture_target
    if posture is None or posture.joint_pos_target is None:
        raise RuntimeError(
            "Order9 rolling trajectory lacks its terminal joint reference"
        )
    values = {
        str(joint_id): float(value)
        for joint_id, value in posture.joint_pos_target.items()
    }
    if not values or any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError(
            "Order9 rolling terminal joint reference is invalid"
        )
    return values


def _quantize_floats(value: Any, *, decimal_places: int) -> Any:
    if isinstance(value, float):
        quantized = round(value, decimal_places)
        return 0.0 if quantized == 0.0 else quantized
    if isinstance(value, list):
        return [
            _quantize_floats(item, decimal_places=decimal_places)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _quantize_floats(item, decimal_places=decimal_places)
            for item in value
        )
    if isinstance(value, dict):
        return {
            key: _quantize_floats(item, decimal_places=decimal_places)
            for key, item in value.items()
        }
    return value


def _posture_physics_summary(
    observations: object,
    *,
    trajectory: ContactWrenchTrajectory,
    context: HighLevelPolicyContext,
    qp_residual_threshold: float,
    wrench_residual_threshold: float,
    qpid_only: bool = False,
    defer_contact_wrench_gate: bool = False,
) -> dict[str, object]:
    if not isinstance(observations, list) or len(observations) != len(
        trajectory.knots
    ):
        return {
            "passed": False,
            "reason": "missing_or_incomplete_observations",
        }
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    maximum_qp_residual = 0.0
    maximum_wrench_residual = 0.0
    prohibited_overlap_count = 0
    first_prohibited_overlap: dict[str, object] | None = None
    finite_count = 0
    unchanged_count = 0
    controller_step_count = 0.0
    learned_pi_l_applied_count = 0.0
    qpid_reference_applied_count = 0.0
    pi_l_fallback_count = 0.0
    unresolved_actuator_target_count = 0.0
    clipped_actuator_count = 0.0
    minimum_prohibited_margin = float("inf")
    for knot_index, (knot, observation) in enumerate(
        zip(trajectory.knots, observations)
    ):
        if not isinstance(observation, dict):
            return {
                "passed": False,
                "reason": "malformed_observation",
            }
        maximum_qp_residual = max(
            maximum_qp_residual,
            float(observation["controller_qp_residual"]),
        )
        maximum_wrench_residual = max(
            maximum_wrench_residual,
            float(observation["contact_wrench_residual"]),
        )
        finite_count += int(bool(observation["finite_state"]))
        unchanged_count += int(bool(observation["main_state_unchanged"]))
        metrics = observation["metrics"]
        controller_step_count += float(
            metrics.get("controller_step_count", 0.0)
        )
        learned_pi_l_applied_count += float(
            metrics.get("learned_pi_l_applied_count", 0.0)
        )
        qpid_reference_applied_count += float(
            metrics.get("qpid_reference_applied_count", 0.0)
        )
        pi_l_fallback_count += float(
            metrics.get("pi_l_fallback_count", 0.0)
        )
        unresolved_actuator_target_count += float(
            metrics.get("unresolved_actuator_target_count", 0.0)
        )
        clipped_actuator_count += float(
            metrics.get("clipped_actuator_count", 0.0)
        )
        allowed_candidates: set[int] = set()
        allowed_identities: set[tuple[int, str]] = set()
        for assignment in knot.contact_assignments:
            if assignment.schedule_state not in {
                "attach",
                "maintain",
                "slide",
                "release",
            }:
                continue
            candidate = candidate_by_id.get(assignment.candidate_id)
            if candidate is None:
                continue
            allowed_candidates.add(candidate.candidate_id)
            allowed_identities.add(
                (assignment.anchor_id, candidate.target_entity_id)
            )
        for sample in observation["collision_samples"]:
            allowed = bool(sample["task_allowed"])
            candidate_id = sample.get("candidate_id")
            anchor_id = sample.get("anchor_id")
            target_entity_id = sample.get("target_entity_id")
            if candidate_id is not None:
                allowed |= int(candidate_id) in allowed_candidates
            elif anchor_id is not None and target_entity_id is not None:
                allowed |= (
                    int(anchor_id),
                    str(target_entity_id),
                ) in allowed_identities
            if allowed:
                continue
            signed_distance = float(sample["signed_distance_m"])
            minimum_prohibited_margin = min(
                minimum_prohibited_margin, signed_distance
            )
            if signed_distance <= 0.0:
                prohibited_overlap_count += 1
                if first_prohibited_overlap is None:
                    first_prohibited_overlap = {
                        "knot_index": knot_index,
                        "knot_time_s": float(knot.t_rel_s),
                        "entity_a": str(sample["entity_a"]),
                        "entity_b": str(sample["entity_b"]),
                        "signed_distance_m": signed_distance,
                    }
    if minimum_prohibited_margin == float("inf"):
        minimum_prohibited_margin = min(
            float(value["collision_free_clearance_m"])
            for value in observations
        )
    passed = bool(
        finite_count == len(observations)
        and unchanged_count == len(observations)
        and maximum_qp_residual <= qp_residual_threshold
        and (
            defer_contact_wrench_gate
            or maximum_wrench_residual <= wrench_residual_threshold
        )
        and prohibited_overlap_count == 0
        and controller_step_count > 0.0
        and (
            qpid_reference_applied_count == controller_step_count
            if qpid_only
            else learned_pi_l_applied_count == controller_step_count
        )
        and pi_l_fallback_count == 0.0
        and unresolved_actuator_target_count == 0.0
        and clipped_actuator_count == 0.0
    )
    return {
        "passed": passed,
        "observation_count": len(observations),
        "finite_state_count": finite_count,
        "main_state_unchanged_count": unchanged_count,
        "controller_step_count": controller_step_count,
        "learned_pi_l_applied_count": learned_pi_l_applied_count,
        "qpid_reference_applied_count": qpid_reference_applied_count,
        "pi_l_fallback_count": pi_l_fallback_count,
        "unresolved_actuator_target_count": (
            unresolved_actuator_target_count
        ),
        "clipped_actuator_count": clipped_actuator_count,
        "maximum_controller_qp_residual": maximum_qp_residual,
        "controller_qp_residual_threshold": float(
            qp_residual_threshold
        ),
        "maximum_contact_wrench_residual": maximum_wrench_residual,
        "contact_wrench_residual_threshold": float(
            wrench_residual_threshold
        ),
        "contact_wrench_gate_deferred_to_phase_gate": bool(
            defer_contact_wrench_gate
        ),
        "minimum_prohibited_collision_margin_m": (
            minimum_prohibited_margin
        ),
        "prohibited_overlap_count": prohibited_overlap_count,
        "first_prohibited_overlap": first_prohibited_overlap,
    }


def _c2_scalar_tracking_summary(
    endpoint: object,
    *,
    trajectory: ContactWrenchTrajectory,
    context: HighLevelPolicyContext,
    physical_model: object,
    centroidal_start_pose_world: tuple[float, ...] | list[float],
    kinematic_translation_correction_world_m: (
        tuple[float, ...] | list[float]
    ),
) -> dict[str, object]:
    if not isinstance(endpoint, dict) or not isinstance(
        endpoint.get("metrics"), dict
    ):
        return {"passed": False, "reason": "missing_endpoint_metrics"}
    metrics = endpoint["metrics"]

    def vector(prefix: str, size: int) -> tuple[float, ...]:
        keys = [f"endpoint.{prefix}_{index}" for index in range(size)]
        if any(key not in metrics for key in keys):
            raise RuntimeError(
                f"Order9 C2 scalar endpoint lacks {prefix}"
            )
        return tuple(float(metrics[key]) for key in keys)

    global_q = {
        key.removeprefix("endpoint.joint_position_rad.").replace(
            "__", ":", 1
        ): float(value)
        for key, value in metrics.items()
        if key.startswith("endpoint.joint_position_rad.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    global_qdot = {
        key.removeprefix("endpoint.joint_velocity_radps.").replace(
            "__", ":", 1
        ): float(value)
        for key, value in metrics.items()
        if key.startswith("endpoint.joint_velocity_radps.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    root_pose = vector("robot_root_pose_world", 7)
    root_twist = vector("robot_root_twist_world", 6)
    ordered_dock_ids = ordered_global_dock_joint_ids(
        context.morphology_graph,
        physical_model,
    )
    missing = [
        joint_id for joint_id in ordered_dock_ids if joint_id not in global_q
    ]
    if missing:
        raise RuntimeError(
            f"Order9 C2 scalar endpoint lacks Dock joints {missing}"
        )
    fk = WholeStructureKinematics().forward(
        context.morphology_graph,
        physical_model,
        {joint_id: global_q[joint_id] for joint_id in ordered_dock_ids},
        root_pose,
        (),
    )
    module_states = [
        ModuleRuntimeState(
            module_id=module.module_id,
            pose_world=fk.module_root_poses_world[module.module_id],
            twist_world=list(root_twist),
            joint_positions={
                key.removeprefix(f"module_{module.module_id}:"): value
                for key, value in global_q.items()
                if key.startswith(f"module_{module.module_id}:")
            },
            joint_velocities={
                key.removeprefix(f"module_{module.module_id}:"): value
                for key, value in global_qdot.items()
                if key.startswith(f"module_{module.module_id}:")
            },
        )
        for module in sorted(
            context.morphology_graph.modules,
            key=lambda value: value.module_id,
        )
    ]
    observation = RuntimeObservation(
        time_s=float(metrics.get("endpoint.elapsed_s", 0.0)),
        morphology_graph=context.morphology_graph,
        module_states=module_states,
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="approach",
            progress_ratio=0.0,
        ),
    )
    reconstructed_pose = RigidBodyControlModelBuilder().build(
        context.morphology_graph,
        physical_model,
        observation,
    ).body_pose_world
    correction = tuple(
        float(value)
        for value in kinematic_translation_correction_world_m
    )
    if len(correction) != 3:
        raise ValueError(
            "Order9 C2 kinematic translation correction must contain three values"
        )
    direct_centroidal_keys = [
        f"endpoint.centroidal_pose_world_{index}" for index in range(7)
    ]
    if all(key in metrics for key in direct_centroidal_keys):
        actual_pose = tuple(
            float(metrics[key]) for key in direct_centroidal_keys
        )
        actual_pose_source = "actual_isaac_module_body_state"
    else:
        actual_pose = (
            *tuple(
                float(reconstructed_pose[index]) + correction[index]
                for index in range(3)
            ),
            *tuple(float(value) for value in reconstructed_pose[3:7]),
        )
        actual_pose_source = "kinematic_reconstruction_with_fixed_usd_offset"
    final_target = trajectory.knots[-1].centroidal_target
    if (
        final_target is None
        or final_target.com_pos_world is None
        or final_target.body_orientation_world is None
    ):
        return {"passed": False, "reason": "missing_centroidal_target"}
    start_pose = tuple(float(value) for value in centroidal_start_pose_world)
    target_pose = (
        *tuple(float(value) for value in final_target.com_pos_world),
        *tuple(float(value) for value in final_target.body_orientation_world),
    )
    position_error = tuple(
        float(actual_pose[index]) - target_pose[index] for index in range(3)
    )
    error_norm = math.sqrt(sum(value * value for value in position_error))
    target_displacement = tuple(
        target_pose[index] - start_pose[index] for index in range(3)
    )
    actual_displacement = tuple(
        float(actual_pose[index]) - start_pose[index] for index in range(3)
    )
    # This is the C2 approach gate's position tolerance.  The diagnostic
    # compares against the instantaneous C2 command, which is stricter than
    # the phase-goal reward only in timing semantics, not a new promotion gate.
    position_tolerance_m = 0.08
    orientation_error = _quaternion_distance_rad(
        actual_pose[3:7],
        target_pose[3:7],
    )
    orientation_tolerance_rad = 0.20
    passed = bool(
        error_norm <= position_tolerance_m
        and orientation_error <= orientation_tolerance_rad
    )
    return {
        "passed": passed,
        "semantics": "instantaneous_exact_c2_reference_tracking",
        "actual_centroidal_pose_world": list(actual_pose),
        "actual_centroidal_pose_source": actual_pose_source,
        "raw_kinematic_centroidal_pose_world": list(reconstructed_pose),
        "kinematic_translation_correction_world_m": list(correction),
        "target_centroidal_pose_world": list(target_pose),
        "centroidal_start_pose_world": list(start_pose),
        "position_error_world_m": list(position_error),
        "position_error_norm_m": error_norm,
        "position_tolerance_m": position_tolerance_m,
        "orientation_error_rad": orientation_error,
        "orientation_tolerance_rad": orientation_tolerance_rad,
        "target_displacement_world_m": list(target_displacement),
        "actual_displacement_world_m": list(actual_displacement),
    }


def _quaternion_distance_rad(
    first: tuple[float, ...] | list[float],
    second: tuple[float, ...] | list[float],
) -> float:
    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        raise ValueError("Order9 task-goal quaternion norm is zero")
    dot = abs(
        sum(
            float(left) * float(right)
            for left, right in zip(first, second)
        )
        / (first_norm * second_norm)
    )
    return 2.0 * math.acos(min(max(dot, -1.0), 1.0))


def _task_goal_summary(
    observations: object,
    *,
    task: TaskSpec,
) -> dict[str, object]:
    if not isinstance(observations, list) or not observations:
        return {"passed": False, "reason": "missing_observations"}
    final = observations[-1]
    if not isinstance(final, dict) or not isinstance(
        final.get("metrics"), dict
    ):
        return {"passed": False, "reason": "missing_final_metrics"}
    metrics = final["metrics"]
    keys = [f"endpoint.object_pose_world_{index}" for index in range(7)]
    if any(key not in metrics for key in keys):
        return {"passed": False, "reason": "missing_final_object_pose"}
    final_pose = tuple(float(metrics[key]) for key in keys)
    target_object_id = task.scene.objects[0].object_id
    matches = [
        goal
        for goal in task.goals
        if goal.target_entity_id == target_object_id
        and goal.target_pose_world is not None
    ]
    if len(matches) != 1:
        return {"passed": False, "reason": "missing_unique_object_pose_goal"}
    goal = matches[0]
    goal_pose = tuple(float(value) for value in goal.target_pose_world)
    position_error = math.sqrt(
        sum(
            (final_pose[index] - goal_pose[index]) ** 2
            for index in range(3)
        )
    )
    orientation_error = _quaternion_distance_rad(
        final_pose[3:7], goal_pose[3:7]
    )
    position_tolerance = float(goal.tolerance_pos_m or 0.0)
    orientation_tolerance = float(goal.tolerance_rot_rad or 0.0)
    passed = bool(
        position_tolerance > 0.0
        and orientation_tolerance > 0.0
        and position_error <= position_tolerance
        and orientation_error <= orientation_tolerance
    )
    return {
        "passed": passed,
        "semantics": "final_object_goal_pose_only",
        "final_object_pose_world": list(final_pose),
        "goal_object_pose_world": list(goal_pose),
        "position_error_m": position_error,
        "position_tolerance_m": position_tolerance,
        "orientation_error_rad": orientation_error,
        "orientation_tolerance_rad": orientation_tolerance,
        "release_and_retreat_success_separately_proven": False,
    }


def _phase_gate_summary(
    observations: object,
    *,
    trajectory: ContactWrenchTrajectory,
    phase_label: str,
    teacher_target_reached: bool,
    support_top_z_m: float,
    object_half_height_m: float,
) -> dict[str, object]:
    """Apply the existing Order 9 physical success gates to one scalar window."""

    if not isinstance(observations, list) or not observations:
        return {
            "passed": False,
            "phase": phase_label,
            "reason": "missing_observations",
        }
    final = observations[-1]
    if not isinstance(final, dict) or not isinstance(
        final.get("metrics"), dict
    ):
        return {
            "passed": False,
            "phase": phase_label,
            "reason": "missing_final_metrics",
        }
    metrics = final["metrics"]

    def vector(prefix: str, size: int) -> tuple[float, ...] | None:
        keys = [f"endpoint.{prefix}_{index}" for index in range(size)]
        if any(key not in metrics for key in keys):
            return None
        return tuple(float(metrics[key]) for key in keys)

    actual_centroidal = vector("centroidal_pose_world", 7)
    centroidal_twist = vector("centroidal_twist_world", 6)
    actual_object = vector("object_pose_world", 7)
    final_knot = trajectory.knots[-1]
    centroidal_target = final_knot.centroidal_target
    desired_centroidal = (
        None
        if centroidal_target is None
        or centroidal_target.com_pos_world is None
        or centroidal_target.body_orientation_world is None
        else (
            *tuple(float(value) for value in centroidal_target.com_pos_world),
            *tuple(
                float(value)
                for value in centroidal_target.body_orientation_world
            ),
        )
    )
    desired_object = next(
        (
            tuple(float(value) for value in target.pose_target_world)
            for target in final_knot.object_targets
            if target.pose_target_world is not None
        ),
        None,
    )
    centroidal_position_error = (
        None
        if actual_centroidal is None or desired_centroidal is None
        else math.sqrt(
            sum(
                (
                    actual_centroidal[index]
                    - desired_centroidal[index]
                )
                ** 2
                for index in range(3)
            )
        )
    )
    centroidal_orientation_error = (
        None
        if actual_centroidal is None or desired_centroidal is None
        else _quaternion_distance_rad(
            actual_centroidal[3:7], desired_centroidal[3:7]
        )
    )
    centroidal_linear_speed = (
        None
        if centroidal_twist is None
        else math.sqrt(sum(value * value for value in centroidal_twist[:3]))
    )
    object_position_error = (
        None
        if actual_object is None or desired_object is None
        else math.sqrt(
            sum(
                (actual_object[index] - desired_object[index]) ** 2
                for index in range(3)
            )
        )
    )
    object_orientation_error = (
        None
        if actual_object is None or desired_object is None
        else _quaternion_distance_rad(
            actual_object[3:7], desired_object[3:7]
        )
    )

    contact_force_threshold_n = 0.5
    required_contact_count = 2
    sufficient_contact_suffix = 0
    contact_free_suffix = 0
    endpoint_active_contact_count = 0
    contact_prefix = "endpoint_contact_force_norm_n."
    for row in reversed(observations):
        if not isinstance(row, dict) or not isinstance(
            row.get("metrics"), dict
        ):
            break
        row_metrics = row["metrics"]
        active_count = sum(
            1
            for key, value in row_metrics.items()
            if key.startswith(contact_prefix)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) >= contact_force_threshold_n
        )
        if sufficient_contact_suffix == 0:
            endpoint_active_contact_count = active_count
        if active_count < required_contact_count:
            break
        sufficient_contact_suffix += 1
    for row in reversed(observations):
        if not isinstance(row, dict) or not isinstance(
            row.get("metrics"), dict
        ):
            break
        row_metrics = row["metrics"]
        active_count = sum(
            1
            for key, value in row_metrics.items()
            if key.startswith(contact_prefix)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) >= contact_force_threshold_n
        )
        if active_count != 0:
            break
        contact_free_suffix += 1
    contact_dwell_s = sufficient_contact_suffix * float(trajectory.dt_s)
    contact_free_dwell_s = contact_free_suffix * float(trajectory.dt_s)
    enough_contact = endpoint_active_contact_count >= required_contact_count
    contact_preload_complete = bool(
        float(metrics.get("endpoint.contact_preload_complete", 0.0)) >= 0.5
    )

    phase = Order9ObjectTaskPhase(phase_label)
    reasons: list[str] = []
    if (
        phase != Order9ObjectTaskPhase.CONTACT_ACQUISITION
        and not teacher_target_reached
    ):
        reasons.append("teacher_target_not_reached")
    if phase == Order9ObjectTaskPhase.APPROACH:
        if centroidal_position_error is None or centroidal_position_error > 0.08:
            reasons.append("centroidal_position")
        if (
            centroidal_orientation_error is None
            or centroidal_orientation_error > 0.20
        ):
            reasons.append("centroidal_orientation")
        if centroidal_linear_speed is None or centroidal_linear_speed > 0.02:
            reasons.append("centroidal_speed")
    elif phase == Order9ObjectTaskPhase.CONTACT_ACQUISITION:
        if contact_dwell_s < 0.25:
            reasons.append("contact_dwell")
        if not contact_preload_complete:
            reasons.append("contact_preload")
    elif phase in {
        Order9ObjectTaskPhase.LIFT,
        Order9ObjectTaskPhase.TRANSPORT,
        Order9ObjectTaskPhase.PLACE,
    }:
        if object_position_error is None or object_position_error > 0.05:
            reasons.append("object_position")
        if object_orientation_error is None or object_orientation_error > 0.20:
            reasons.append("object_orientation")
        if not enough_contact:
            reasons.append("contact_count")
        if (
            phase == Order9ObjectTaskPhase.LIFT
            and (
                actual_object is None
                or actual_object[2]
                - float(object_half_height_m)
                - float(support_top_z_m)
                < 0.001
            )
        ):
            reasons.append("lift_clearance")
    elif phase == Order9ObjectTaskPhase.RELEASE:
        if object_position_error is None or object_position_error > 0.05:
            reasons.append("object_position")
        if object_orientation_error is None or object_orientation_error > 0.20:
            reasons.append("object_orientation")
        if contact_free_dwell_s < 0.10:
            reasons.append("release_contact_free_dwell")
    else:
        reasons.append("phase_gate_not_supported_by_grasp_transport_check")
    return {
        "passed": not reasons,
        "phase": phase.value,
        "teacher_target_reached": bool(teacher_target_reached),
        "failure_reasons": reasons,
        "actual_centroidal_pose_world": (
            None if actual_centroidal is None else list(actual_centroidal)
        ),
        "desired_centroidal_pose_world": (
            None if desired_centroidal is None else list(desired_centroidal)
        ),
        "centroidal_position_error_m": centroidal_position_error,
        "centroidal_orientation_error_rad": centroidal_orientation_error,
        "centroidal_linear_speed_mps": centroidal_linear_speed,
        "actual_object_pose_world": (
            None if actual_object is None else list(actual_object)
        ),
        "desired_object_pose_world": (
            None if desired_object is None else list(desired_object)
        ),
        "object_position_error_m": object_position_error,
        "object_orientation_error_rad": object_orientation_error,
        "active_contact_count": endpoint_active_contact_count,
        "required_contact_count": required_contact_count,
        "contact_force_threshold_n": contact_force_threshold_n,
        "contact_dwell_s": contact_dwell_s,
        "required_contact_dwell_s": 0.25,
        "contact_free_dwell_s": contact_free_dwell_s,
        "required_contact_free_dwell_s": 0.10,
        "release_valid": bool(
            phase == Order9ObjectTaskPhase.RELEASE
            and contact_free_dwell_s >= 0.10
            and object_position_error is not None
            and object_position_error <= 0.05
            and object_orientation_error is not None
            and object_orientation_error <= 0.20
        ),
        "contact_preload_complete": contact_preload_complete,
        "support_clearance_m": (
            None
            if actual_object is None
            else actual_object[2]
            - float(object_half_height_m)
            - float(support_top_z_m)
        ),
        "semantics": "existing_order9_tensor_reward_physical_gate_scalar_replay",
    }


def main() -> int:
    args = _parser().parse_args()
    if (
        not math.isfinite(float(args.gui_keep_open_s))
        or args.gui_keep_open_s < 0.0
    ):
        raise ValueError("--gui-keep-open-s must be finite and non-negative")
    if args.c2_rolling_window_count < 1:
        raise ValueError("--c2-rolling-window-count must be positive")
    if args.c2_stop_after_phase is not None and args.c2_bucket_manifest is None:
        raise ValueError(
            "--c2-stop-after-phase requires --c2-bucket-manifest"
        )
    if args.c2_stop_after_phase is not None and (
        args.c2_exact_runtime_reference or args.qpid_only_tracking
    ):
        raise ValueError(
            "sequential task execution requires the articulated teacher and pi_L"
        )
    if args.c2_exact_runtime_reference and args.c2_bucket_manifest is None:
        raise ValueError(
            "--c2-exact-runtime-reference requires --c2-bucket-manifest"
        )
    if args.c2_exact_runtime_reference != (
        args.c2_batched_rollout is not None
    ):
        raise ValueError(
            "--c2-exact-runtime-reference and --c2-batched-rollout "
            "must be provided together"
        )
    if args.qpid_only_tracking and not args.c2_exact_runtime_reference:
        raise ValueError(
            "--qpid-only-tracking requires --c2-exact-runtime-reference"
        )
    if args.bypass_pi_l and args.qpid_only_tracking:
        raise ValueError(
            "--bypass-pi-l and --qpid-only-tracking are mutually exclusive"
        )
    if (
        args.qpid_test_total_displacement_m is not None
        and not args.qpid_only_tracking
    ):
        raise ValueError(
            "--qpid-test-total-displacement-m requires "
            "--qpid-only-tracking"
        )
    if (
        args.c2_bucket_manifest is None
        and args.c2_rolling_window_count != 1
    ):
        raise ValueError(
            "--c2-rolling-window-count requires --c2-bucket-manifest"
        )
    if args.dt <= 0.0 or args.timeout_s <= 0.0:
        raise ValueError("Order9 shadow smoke dt/timeout must be positive")
    if (args.posture_fixture is None) != (args.posture_case_id is None):
        raise ValueError(
            "--posture-fixture and --posture-case-id must be provided together"
        )
    if (args.c2_bucket_manifest is None) != (args.c2_bucket_id is None):
        raise ValueError(
            "--c2-bucket-manifest and --c2-bucket-id must be provided together"
        )
    if (args.c2_bucket_manifest is None) != (
        args.c2_articulated_urdf is None
    ):
        raise ValueError(
            "--c2-articulated-urdf is required exactly when "
            "--c2-bucket-manifest is used"
        )
    if args.posture_fixture is not None and args.c2_bucket_manifest is not None:
        raise ValueError(
            "posture fixture and C2 bucket modes are mutually exclusive"
        )
    if args.posture_fixture is not None and (
        args.morphology_graph_json is None or args.robot_usd is None
    ):
        raise ValueError(
            "posture fixture execution requires --morphology-graph-json "
            "and --robot-usd"
        )
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
    fixture_evidence: dict[str, object] | None = None
    c2_bucket_evidence: dict[str, object] | None = None
    c2_phase_reset: object | None = None
    c2_articulated_urdf: URDFModel | None = None
    c2_contact_solution: ArticulatedIKSolution | None = None
    scene_arguments: dict[str, object] | None = None
    worker_morphology_path: Path | None = None
    worker_robot_usd_path: Path | None = None
    if args.posture_fixture is not None:
        (
            task,
            morphology,
            context,
            trajectory,
            scene_arguments,
            fixture_evidence,
        ) = _posture_fixture_inputs(
            Path(args.posture_fixture).resolve(),
            case_id=str(args.posture_case_id),
            physical_model=physical_model,
            contact_stiffness=(
                config.randomization.nominal_contact_stiffness_n_per_m
            ),
            contact_damping=(
                config.randomization.nominal_contact_damping_n_s_per_m
            ),
        )
        source_morphology = MorphologyGraph.from_json(
            Path(args.morphology_graph_json).read_text(encoding="utf-8")
        )
        if morphology_structural_hash(
            morphology
        ) != morphology_structural_hash(source_morphology):
            raise ValueError(
                "posture fixture topology differs from worker graph input"
            )
        worker_morphology_path = write_order9_shadow_bucket_morphology(
            repository
            / "artifacts/p4_full/order9/posture_resolver/"
            "shadow_bucket_inputs",
            morphology,
        )
        worker_robot_usd_path = Path(args.robot_usd).resolve()
    elif args.c2_bucket_manifest is not None:
        (
            task,
            morphology,
            context,
            trajectory,
            scene_arguments,
            c2_bucket_evidence,
            c2_phase_reset,
            c2_articulated_urdf,
            worker_morphology_path,
            worker_robot_usd_path,
            c2_contact_solution,
        ) = _c2_bucket_inputs(
            Path(args.c2_bucket_manifest).resolve(),
            bucket_id=str(args.c2_bucket_id),
            repository=repository,
            physical_model=physical_model,
            reset=reset,
            articulated_urdf_path=Path(
                args.c2_articulated_urdf
            ).resolve(),
            exact_runtime_reference=bool(args.c2_exact_runtime_reference),
            batched_rollout_path=(
                None
                if args.c2_batched_rollout is None
                else Path(args.c2_batched_rollout).resolve()
            ),
            qpid_test_total_displacement_m=(
                None
                if args.qpid_test_total_displacement_m is None
                else tuple(
                    float(value)
                    for value in args.qpid_test_total_displacement_m
                )
            ),
        )
    else:
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
        trajectory = _short_nominal_trajectory(
            context, dt_s=float(args.dt)
        )
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
    command = _worker_command(
        args,
        repository=repository,
        scene_arguments=scene_arguments,
        morphology_graph_path=worker_morphology_path,
        robot_usd_path=worker_robot_usd_path,
    )
    effective_timeout_s = (
        max(float(args.timeout_s), _GUI_MINIMUM_RPC_TIMEOUT_S)
        if args.gui_realtime
        else float(args.timeout_s)
    )
    if args.gui_realtime:
        print(
            "ORDER9_GUI_STATUS="
            + json.dumps(
                {
                    "stage": "launching_worker",
                    "rpc_timeout_s": effective_timeout_s,
                    "keep_open_after_run_s": float(args.gui_keep_open_s),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    transport = JsonLineSubprocessShadowTransport(
        command,
        cwd=repository,
        timeout_s=effective_timeout_s,
        environment={"PYTHONPATH": str(repository)},
        status_stream=sys.stderr if args.gui_realtime else None,
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
        if args.posture_fixture is not None:
            first = trajectory.knots[0]
            centroidal = first.centroidal_target
            posture = first.posture_target
            if (
                centroidal is None
                or centroidal.com_pos_world is None
                or centroidal.body_orientation_world is None
                or posture is None
                or posture.joint_pos_target is None
                or posture.joint_vel_target is None
            ):
                raise RuntimeError(
                    "resolved posture fixture lacks its initial references"
                )
            robot_root_pose_world = base_pose_for_centroidal_target(
                morphology,
                physical_model,
                posture.joint_pos_target,
                centroidal.com_pos_world,
                centroidal.body_orientation_world,
            )
            robot_root_twist_world = [0.0] * 6
            q_by_name = {
                key.replace(":", "__", 1): float(value)
                for key, value in posture.joint_pos_target.items()
            }
            qdot_by_name = {
                key.replace(":", "__", 1): float(value)
                for key, value in posture.joint_vel_target.items()
            }
            target_object = task.scene.objects[0]
            object_id = target_object.object_id
            object_pose_world = list(target_object.pose_world)
            object_twist_world = [0.0] * 6
            phase_index = 0
        elif args.c2_bucket_manifest is not None:
            if c2_phase_reset is None:
                raise RuntimeError("Order9 C2 phase-zero reset is missing")
            robot_root_pose_world = list(
                c2_phase_reset.robot_root_pose_world
            )
            robot_root_twist_world = list(
                c2_phase_reset.robot_root_twist_world
            )
            q_by_name = {
                key.replace(":", "__", 1): float(value)
                for key, value in c2_phase_reset.joint_positions_rad.items()
            }
            qdot_by_name = {
                key.replace(":", "__", 1): float(value)
                for key, value in c2_phase_reset.joint_velocities_radps.items()
            }
            object_id = task.scene.objects[0].object_id
            object_pose_world = list(c2_phase_reset.object_pose_world)
            object_twist_world = list(c2_phase_reset.object_twist_world)
            phase_index = 0
        else:
            phase_reset = Order9ObjectTaskRuntime(reset).reset_for_phase(2)
            robot_root_pose_world = list(
                phase_reset.robot_root_pose_world
            )
            robot_root_twist_world = list(
                phase_reset.robot_root_twist_world
            )
            q_by_name = {
                key.replace(":", "__", 1): float(value)
                for key, value in phase_reset.joint_positions_rad.items()
            }
            qdot_by_name = {
                key.replace(":", "__", 1): float(value)
                for key, value in phase_reset.joint_velocities_radps.items()
            }
            object_id = "order8_object"
            object_pose_world = list(phase_reset.object_pose_world)
            object_twist_world = list(phase_reset.object_twist_world)
            phase_index = 2
        snapshot = Order9IsaacStateSnapshot(
            simulation_time_s=0.0,
            robot_root_pose_world=list(robot_root_pose_world),
            robot_root_twist_world=list(robot_root_twist_world),
            joint_names=joint_names,
            joint_positions_rad=[q_by_name.get(name, 0.0) for name in joint_names],
            joint_velocities_radps=[qdot_by_name.get(name, 0.0) for name in joint_names],
            object_id=object_id,
            object_pose_world=object_pose_world,
            object_twist_world=object_twist_world,
            phase_index=phase_index,
            phase_elapsed_s=0.0,
            command_index=0,
            metadata={
                "source": "order9_real_isaac_shadow_smoke",
                "posture_fixture_case_id": args.posture_case_id,
                "c2_bucket_id": args.c2_bucket_id,
            },
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
        current_context = context
        current_trajectory = trajectory
        current_c2_evidence = c2_bucket_evidence
        execution_windows: list[dict[str, object]] = []
        observations: list[object] = []
        phase_transitions: list[dict[str, object]] = []
        successful_phase_labels: list[str] = []
        sequential_target_reached = False
        window_journal_path = (
            None
            if not args.output_json
            else Path(args.output_json).resolve().with_suffix(
                Path(args.output_json).suffix + ".windows.jsonl"
            )
        )
        if window_journal_path is not None:
            window_journal_path.parent.mkdir(parents=True, exist_ok=True)
            window_journal_path.write_text("", encoding="utf-8")
        maximum_windows = (
            args.c2_rolling_window_count
            if args.c2_bucket_manifest is not None
            else 1
        )
        for window_index in range(maximum_windows):
            if args.gui_realtime:
                print(
                    "ORDER9_GUI_STATUS="
                    + json.dumps(
                        {
                            "stage": "executing_window",
                            "window_index": window_index,
                            "maximum_window_count": maximum_windows,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            executed = dict(
                transport.request(
                    "execute",
                    {
                        "state_digest": state.state_digest,
                        "pi_l_checkpoint_sha256": (
                            args.pi_l_checkpoint_sha256
                        ),
                        "proposal_hash": (
                            current_trajectory.stable_hash()
                        ),
                        "trajectory": current_trajectory.to_dict(),
                        "context": {
                            "irg": current_context.irg.to_dict(),
                            "interaction_envelope": (
                                current_context.interaction_envelope.to_dict()
                            ),
                            "morphology_graph": (
                                current_context.morphology_graph.to_dict()
                            ),
                            "contact_candidate_set": (
                                current_context.contact_candidate_set.to_dict()
                            ),
                            "runtime_observation": (
                                None
                                if current_context.runtime_observation is None
                                else current_context.runtime_observation.to_dict()
                            ),
                        },
                    },
                )
            )
            window_observations = executed.get("observations")
            window_runtime_passed = bool(
                executed.get("accepted") is True
                and isinstance(window_observations, list)
                and len(window_observations)
                == len(current_trajectory.knots)
            )
            window_physics = (
                _posture_physics_summary(
                    window_observations,
                    trajectory=current_trajectory,
                    context=current_context,
                    qp_residual_threshold=(
                        config.hard_checker.qp_residual_threshold
                    ),
                    wrench_residual_threshold=(
                        config.hard_checker.wrench_residual_threshold
                    ),
                    qpid_only=bool(
                        args.qpid_only_tracking or args.bypass_pi_l
                    ),
                    defer_contact_wrench_gate=(
                        current_c2_evidence is not None
                        and current_c2_evidence.get(
                            "teacher_task_phase"
                        )
                        in {
                            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value,
                            Order9ObjectTaskPhase.LIFT.value,
                            Order9ObjectTaskPhase.TRANSPORT.value,
                            Order9ObjectTaskPhase.PLACE.value,
                            Order9ObjectTaskPhase.RELEASE.value,
                        }
                    ),
                )
                if (
                    args.posture_fixture is not None
                    or args.c2_bucket_manifest is not None
                )
                else None
            )
            window_tracking = None
            if (
                args.c2_exact_runtime_reference
                and isinstance(window_observations, list)
                and window_observations
                and current_c2_evidence is not None
            ):
                start_pose = current_c2_evidence.get(
                    "c2_reference_centroidal_start_pose_world"
                )
                correction = current_c2_evidence.get(
                    "c2_reference_kinematic_translation_correction_world_m"
                )
                if not isinstance(start_pose, list) or not isinstance(
                    correction, list
                ):
                    raise RuntimeError(
                        "Order9 C2 scalar reference frame evidence is missing"
                    )
                window_tracking = _c2_scalar_tracking_summary(
                    window_observations[-1],
                    trajectory=current_trajectory,
                    context=current_context,
                    physical_model=physical_model,
                    centroidal_start_pose_world=start_pose,
                    kinematic_translation_correction_world_m=correction,
                )
            target_reached = bool(
                current_c2_evidence is not None
                and current_c2_evidence.get(
                    "teacher_phase_target_reached"
                )
            )
            current_phase_label = (
                None
                if current_c2_evidence is None
                else current_c2_evidence.get("teacher_task_phase")
            )
            phase_gate = None
            if args.c2_stop_after_phase is not None:
                if not isinstance(current_phase_label, str):
                    raise RuntimeError(
                        "Order9 sequential task lacks teacher phase identity"
                    )
                if scene_arguments is None:
                    raise RuntimeError(
                        "Order9 sequential task lacks scene arguments"
                    )
                object_size = scene_arguments.get("object_size")
                support_top_z = scene_arguments.get("support_top_z")
                if not (
                    isinstance(object_size, tuple)
                    and len(object_size) == 3
                    and isinstance(support_top_z, (int, float))
                    and not isinstance(support_top_z, bool)
                ):
                    raise RuntimeError(
                        "Order9 sequential task scene geometry is invalid"
                    )
                phase_gate = _phase_gate_summary(
                    window_observations,
                    trajectory=current_trajectory,
                    phase_label=current_phase_label,
                    teacher_target_reached=target_reached,
                    support_top_z_m=float(support_top_z),
                    object_half_height_m=0.5 * float(object_size[2]),
                )
            execution_windows.append(
                {
                    "window_index": window_index,
                    "proposal_hash": current_trajectory.stable_hash(),
                    "trajectory_horizon_s": (
                        current_trajectory.horizon_s
                    ),
                    "trajectory_knot_count": len(
                        current_trajectory.knots
                    ),
                    "runtime_passed": window_runtime_passed,
                    "physics_summary": window_physics,
                    "tracking_summary": window_tracking,
                    "phase_gate_summary": phase_gate,
                    "teacher": (
                        None
                        if current_c2_evidence is None
                        else dict(current_c2_evidence)
                    ),
                    "rpc": {
                        key: value
                        for key, value in executed.items()
                        if key != "observations"
                    },
                }
            )
            if args.c2_stop_after_phase is not None:
                print(
                    "ORDER9_C2_WINDOW="
                    + json.dumps(
                        {
                            "window_index": window_index,
                            "phase": current_phase_label,
                            "teacher_target_reached": target_reached,
                            "phase_gate_passed": (
                                None
                                if phase_gate is None
                                else bool(phase_gate.get("passed"))
                            ),
                            "phase_gate_failure_reasons": (
                                None
                                if phase_gate is None
                                else phase_gate.get("failure_reasons")
                            ),
                            "physics_passed": (
                                None
                                if not isinstance(window_physics, dict)
                                else bool(window_physics.get("passed"))
                            ),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if isinstance(window_observations, list):
                observations.extend(window_observations)
            if window_journal_path is not None:
                with window_journal_path.open(
                    "a", encoding="utf-8"
                ) as journal:
                    journal.write(
                        json.dumps(
                            {
                                "execution_window": execution_windows[-1],
                                "endpoint_observation": (
                                    None
                                    if not isinstance(
                                        window_observations, list
                                    )
                                    or not window_observations
                                    else window_observations[-1]
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if (
                not window_runtime_passed
                or (
                    args.c2_stop_after_phase is not None
                    and (
                        not isinstance(window_physics, dict)
                        or not bool(window_physics.get("passed"))
                    )
                )
            ):
                break
            endpoint = window_observations[-1]
            if not isinstance(endpoint, dict):
                raise RuntimeError(
                    "Order9 C2 rolling endpoint is malformed"
                )
            if args.c2_stop_after_phase is not None:
                if not isinstance(phase_gate, dict):
                    raise RuntimeError(
                        "Order9 sequential task phase gate is missing"
                    )
                if bool(phase_gate.get("passed")):
                    if not isinstance(current_phase_label, str):
                        raise RuntimeError(
                            "Order9 sequential phase identity is missing"
                        )
                    successful_phase_labels.append(current_phase_label)
                    if current_phase_label == args.c2_stop_after_phase:
                        sequential_target_reached = True
                        break
                    phase_values = tuple(
                        phase.value for phase in ORDER9_OBJECT_TASK_PHASES
                    )
                    current_phase_index = phase_values.index(
                        current_phase_label
                    )
                    if current_phase_index + 1 >= len(phase_values):
                        break
                    next_phase_label = phase_values[current_phase_index + 1]
                    phase_transitions.append(
                        {
                            "from_phase": current_phase_label,
                            "to_phase": next_phase_label,
                            "after_window_index": window_index,
                            "simulated_time_s": sum(
                                float(value["trajectory_horizon_s"])
                                for value in execution_windows
                            ),
                        }
                    )
                else:
                    next_phase_label = current_phase_label
                if window_index + 1 >= maximum_windows:
                    break
                if c2_articulated_urdf is None:
                    raise RuntimeError(
                        "Order9 C2 articulated URDF model is missing"
                    )
                if c2_contact_solution is None:
                    raise RuntimeError(
                        "Order9 C2 contact-posture reference is missing"
                    )
                (
                    current_context,
                    current_trajectory,
                    current_c2_evidence,
                ) = _c2_replan_from_endpoint(
                    endpoint,
                    context=current_context,
                    task=task,
                    physical_model=physical_model,
                    articulated_urdf=c2_articulated_urdf,
                    contact_solution=c2_contact_solution,
                    phase_label=next_phase_label,
                    nominal_start_joint_positions_rad=(
                        _terminal_joint_reference(current_trajectory)
                    ),
                    release_joint_positions_rad=(
                        None
                        if c2_phase_reset is None
                        else c2_phase_reset.joint_positions_rad
                    ),
                )
                continue
            if (
                args.c2_bucket_manifest is None
                or target_reached
                or window_index + 1 >= maximum_windows
            ):
                break
            if args.c2_exact_runtime_reference:
                if current_c2_evidence is None:
                    raise RuntimeError(
                        "Order9 C2 scalar reference evidence is missing"
                    )
                start_pose = current_c2_evidence.get(
                    "c2_reference_centroidal_start_pose_world"
                )
                if not isinstance(start_pose, list):
                    raise RuntimeError(
                        "Order9 C2 scalar reference start pose is missing"
                    )
                segment_start_s = (
                    (window_index + 1) * current_trajectory.horizon_s
                )
                current_trajectory = _c2_exact_runtime_reference_segment(
                    current_context,
                    reset=reset,
                    phase_reset=c2_phase_reset,
                    centroidal_start_pose_world=start_pose,
                    segment_start_s=segment_start_s,
                    segment_horizon_s=current_trajectory.horizon_s,
                    dt_s=float(args.dt),
                    total_displacement_world_m=(
                        None
                        if current_c2_evidence.get(
                            "qpid_test_total_displacement_world_m"
                        )
                        is None
                        else tuple(
                            float(value)
                            for value in current_c2_evidence[
                                "qpid_test_total_displacement_world_m"
                            ]
                        )
                    ),
                )
                current_c2_evidence = {
                    **current_c2_evidence,
                    "resolved_trajectory_hash": (
                        current_trajectory.stable_hash()
                    ),
                    "c2_reference_segment_start_s": segment_start_s,
                    "c2_reference_segment_end_s": (
                        segment_start_s + current_trajectory.horizon_s
                    ),
                }
            else:
                if c2_articulated_urdf is None:
                    raise RuntimeError(
                        "Order9 C2 articulated URDF model is missing"
                    )
                if c2_contact_solution is None:
                    raise RuntimeError(
                        "Order9 C2 contact-posture reference is missing"
                    )
                (
                    current_context,
                    current_trajectory,
                    current_c2_evidence,
                ) = _c2_replan_from_endpoint(
                    endpoint,
                    context=current_context,
                    task=task,
                    physical_model=physical_model,
                    articulated_urdf=c2_articulated_urdf,
                    contact_solution=c2_contact_solution,
                    nominal_start_joint_positions_rad=(
                        _terminal_joint_reference(current_trajectory)
                    ),
                    release_joint_positions_rad=(
                        None
                        if c2_phase_reset is None
                        else c2_phase_reset.joint_positions_rad
                    ),
                )
        reset_result = dict(
            transport.request("reset", {"state_digest": state.state_digest})
        )
        context = current_context
        trajectory = current_trajectory
        runtime_passed = bool(
            synchronized.get("accepted") is True
            and reset_result.get("accepted") is True
            and execution_windows
            and all(
                bool(window["runtime_passed"])
                for window in execution_windows
            )
            and isinstance(scene.get("actuator_readback"), dict)
            and scene["actuator_readback"].get("matches_physical_model") is True
        )
        window_physics = [
            window["physics_summary"]
            for window in execution_windows
            if isinstance(window.get("physics_summary"), dict)
        ]
        physics_summary = (
            None
            if not window_physics
            else window_physics[0]
            if len(window_physics) == 1
            else {
                "passed": all(
                    bool(value.get("passed"))
                    for value in window_physics
                ),
                "window_count": len(window_physics),
                "windows": window_physics,
            }
        )
        physical_gate_passed = bool(
            physics_summary is not None and physics_summary.get("passed")
        )
        task_goal_summary = (
            _task_goal_summary(observations, task=task)
            if args.c2_bucket_manifest is not None
            else None
        )
        task_goal_passed = bool(
            task_goal_summary is not None
            and task_goal_summary.get("passed")
        )
        c2_rolling_summary = None
        if args.c2_bucket_manifest is not None:
            tracking_rows = [
                value["tracking_summary"]
                for value in execution_windows
                if isinstance(value.get("tracking_summary"), dict)
            ]
            final_teacher = execution_windows[-1].get("teacher")
            final_target_reached = bool(
                isinstance(final_teacher, dict)
                and final_teacher.get("teacher_phase_target_reached")
            )
            final_phase_gate = execution_windows[-1].get(
                "phase_gate_summary"
            )
            final_phase_gate_passed = bool(
                isinstance(final_phase_gate, dict)
                and final_phase_gate.get("passed")
            )
            exact_reference_passed = bool(
                args.c2_exact_runtime_reference
                and len(tracking_rows) == len(execution_windows)
                and all(bool(value.get("passed")) for value in tracking_rows)
            )
            c2_rolling_summary = {
                "passed": bool(
                    runtime_passed
                    and physical_gate_passed
                    and (
                        exact_reference_passed
                        if args.c2_exact_runtime_reference
                        else sequential_target_reached
                        if args.c2_stop_after_phase is not None
                        else final_target_reached
                    )
                ),
                "requested_maximum_window_count": (
                    args.c2_rolling_window_count
                ),
                "executed_window_count": len(execution_windows),
                "phase_target_reached": (
                    sequential_target_reached
                    if args.c2_stop_after_phase is not None
                    else final_target_reached
                ),
                "requested_stop_after_phase": args.c2_stop_after_phase,
                "successful_phase_labels": successful_phase_labels,
                "phase_transitions": phase_transitions,
                "final_phase_gate_passed": final_phase_gate_passed,
                "semantics": (
                    (
                        "exact_c2_runtime_reference_qpid_only_tracking"
                        if args.qpid_only_tracking
                        else "exact_c2_runtime_reference_scalar_regression"
                    )
                    if args.c2_exact_runtime_reference
                    else (
                        "measured_state_receding_horizon_sequential_physical_gates"
                        if args.c2_stop_after_phase is not None
                        else "measured_state_receding_horizon_current_phase_only"
                    )
                ),
                "tracking_windows": tracking_rows,
                "full_task_success_claimed": False,
                "grasp_transport_success_claimed": bool(
                    sequential_target_reached
                    and args.c2_stop_after_phase
                    in {
                        Order9ObjectTaskPhase.TRANSPORT.value,
                        Order9ObjectTaskPhase.PLACE.value,
                    }
                ),
            }
        passed = bool(
            runtime_passed
            and (
                c2_rolling_summary is not None
                and c2_rolling_summary["passed"]
                if args.c2_bucket_manifest is not None
                else physical_gate_passed
                if args.posture_fixture is not None
                else True
            )
        )
        report = {
            "passed": passed,
            "runtime_passed": runtime_passed,
            "physical_gate_passed": (
                physical_gate_passed
                if (
                    args.posture_fixture is not None
                    or args.c2_bucket_manifest is not None
                )
                else None
            ),
            "task_goal_passed": (
                task_goal_passed
                if args.c2_bucket_manifest is not None
                else None
            ),
            "task_goal_summary": task_goal_summary,
            "c2_rolling_summary": c2_rolling_summary,
            "physics_summary": physics_summary,
            "checkpoint_sha256": args.pi_l_checkpoint_sha256,
            "command_source": (
                "direct_qpid_reference"
                if args.qpid_only_tracking
                else "articulated_teacher_direct_qpid_diagnostic"
                if args.bypass_pi_l
                else "learned_pi_l"
            ),
            "state_digest": state.state_digest,
            "proposal_hash": trajectory.stable_hash(),
            "worker_version": description.get("worker_version"),
            "joint_count": len(joint_names),
            "candidate_count": len(
                context.contact_candidate_set.candidates
            ),
            "observation_count": len(observations) if isinstance(observations, list) else 0,
            "observations": observations,
            "execution_windows": execution_windows,
            "window_journal_path": (
                None
                if window_journal_path is None
                else str(window_journal_path)
            ),
            "actuator_readback": scene.get("actuator_readback"),
            "posture_fixture": fixture_evidence,
            "c2_bucket": c2_bucket_evidence,
            "rpc": {
                "synchronize": synchronized,
                "execute": {
                    "window_count": len(execution_windows),
                    "windows": [
                        window["rpc"] for window in execution_windows
                    ],
                },
                "reset": reset_result,
            },
        }
        if args.output_json:
            output = Path(args.output_json).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.gui_realtime:
            print(
                "ORDER9_GUI_RESULT="
                + json.dumps(
                    {
                        "viewer_execution_completed": bool(runtime_passed),
                        "task_or_phase_passed": bool(passed),
                        "executed_window_count": len(execution_windows),
                        "observation_count": len(observations),
                        "output_json": (
                            None
                            if not args.output_json
                            else str(Path(args.output_json).resolve())
                        ),
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(report, sort_keys=True))
        return 0 if passed else 1
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
