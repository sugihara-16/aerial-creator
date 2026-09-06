from __future__ import annotations

"""MuJoCo replay bridge for one saved Order 9 Isaac trajectory.

The bridge is deliberately diagnostic.  It does not replace the protected
Isaac acceptance runtime.  It reconstructs the same articulated morphology,
task primitives, nominal targets, QPID/QP controller and local joint drives,
then compares the resulting MuJoCo trajectory with one immutable Isaac trace.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch

from amsrr.controllers.batched_qpid_controller import BatchedQPIDController
from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.feasibility.order9_posture_collision import (
    _binary_stl_convex_hull,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.simulation.order9_actuator_runtime import (
    order9_actuator_runtime_values,
)
from amsrr.simulation.order9_tensor_object_task import (
    order9_payload_feedforward_scale,
)
from amsrr.training.order9_tensor_pi_l_runtime import Order9TensorPiLRuntime
from amsrr.training.order9_tensor_reward import (
    Order9TensorRewardEngine,
    Order9TensorRewardInput,
)

ORDER9_MUJOCO_TRACE_REPLAY_VERSION = (
    "order9_mujoco_closed_loop_isaac_trace_comparison_v1"
)


@dataclass(frozen=True)
class IsaacEpisodeTrace:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    environment_index: int
    episode_serial: int
    trace_indices: torch.Tensor

    @property
    def control_step_count(self) -> int:
        # The final trace row is the post-settle terminal marker and has no
        # following reference state against which to compare a transition.
        return int(self.trace_indices.numel()) - 1


@dataclass(frozen=True)
class MujocoModelContext:
    model: Any
    robot_root_body_id: int
    payload_body_id: int
    support_geom_id: int
    payload_geom_id: int
    module_site_ids: tuple[int, ...]
    selected_body_ids: tuple[int, ...]
    selected_site_ids: tuple[int, ...]
    rotor_body_ids: tuple[int, ...]
    rotor_reaction_coefficients: np.ndarray
    gimbal_joint_ids: tuple[int, ...]
    dock_joint_ids: tuple[tuple[int, ...], ...]
    local_joint_ids: tuple[tuple[int, ...], ...]
    robot_free_joint_id: int
    payload_free_joint_id: int
    gimbal_drive: tuple[float, float, float, float]
    dock_drive: tuple[float, float, float, float]
    mesh_bridge_manifest: dict[str, Any]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    payload = bytearray(84 + 50 * len(faces))
    payload[:80] = b"A-MSRR source-mesh convex hull".ljust(80, b" ")
    struct.pack_into("<I", payload, 80, len(faces))
    for index, face in enumerate(faces):
        triangle = np.asarray(vertices[face], dtype=np.float32)
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm > 0.0:
            normal /= norm
        struct.pack_into(
            "<12fH",
            payload,
            84 + 50 * index,
            *(normal.tolist() + triangle.reshape(-1).tolist()),
            0,
        )
    path.write_bytes(payload)


def materialize_convex_hull_urdf(
    source_urdf: Path, output_directory: Path
) -> tuple[Path, dict[str, Any]]:
    """Create the per-source-mesh convex-hull model used by both engines.

    The source files remain untouched.  Isaac's generated USD represents each
    imported collision mesh by its convex hull.  Replacing each MuJoCo mesh by
    the convex hull of the same source vertices therefore removes MuJoCo's STL
    face-count limitation without introducing a decimation tolerance.
    """

    source_urdf = source_urdf.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source_urdf).getroot()
    generated: dict[Path, tuple[Path, dict[str, Any]]] = {}
    for mesh in root.findall(".//mesh"):
        source_mesh = Path(str(mesh.attrib["filename"]))
        if not source_mesh.is_absolute():
            source_mesh = source_urdf.parent / source_mesh
        source_mesh = source_mesh.resolve()
        scale = tuple(
            float(value) for value in mesh.attrib.get("scale", "1 1 1").split()
        )
        if scale != (1.0, 1.0, 1.0):
            raise ValueError("MuJoCo bridge requires unit-scaled source STL meshes")
        if source_mesh not in generated:
            source_sha = hash_file(source_mesh)
            vertices, faces = _binary_stl_convex_hull(
                source_mesh, np.ones(3, dtype=np.float64)
            )
            destination = output_directory / (
                f"{source_mesh.stem}.{source_sha[:12]}.convex.stl"
            )
            if not destination.is_file():
                _write_binary_stl(destination, vertices, faces)
            destination_sha = hash_file(destination)
            generated[source_mesh] = (
                destination,
                {
                    "source_path": str(source_mesh),
                    "source_sha256": source_sha,
                    "convex_hull_path": str(destination.resolve()),
                    "convex_hull_sha256": destination_sha,
                    "convex_vertex_count": int(vertices.shape[0]),
                    "convex_triangle_count": int(faces.shape[0]),
                },
            )
        mesh.attrib["filename"] = str(generated[source_mesh][0].resolve())
    output_urdf = output_directory / "robot_convex_hulls.urdf"
    ET.ElementTree(root).write(output_urdf, encoding="utf-8", xml_declaration=True)
    manifest = {
        "version": "order9_mujoco_source_mesh_convex_hull_bridge_v1",
        "source_urdf_path": str(source_urdf),
        "source_urdf_sha256": hash_file(source_urdf),
        "generated_urdf_path": str(output_urdf.resolve()),
        "generated_urdf_sha256": hash_file(output_urdf),
        "collision_semantics": (
            "one convex hull per authored source mesh; no vertex decimation"
        ),
        "meshes": [value[1] for value in generated.values()],
    }
    return output_urdf, manifest


def load_isaac_episode_trace(
    path: Path, *, environment_index: int = 0, episode_serial: int = 0
) -> IsaacEpisodeTrace:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("artifact_version") is None:
        raise ValueError("Isaac rollout lacks artifact_version")
    metadata = dict(payload["metadata"])
    tensors = dict(payload["tensors"])
    valid = tensors["valid"][:, environment_index]
    serial = tensors["episode_serial"][:, environment_index]
    indices = torch.nonzero(
        valid & (serial == int(episode_serial)), as_tuple=False
    ).flatten()
    if indices.numel() < 2 or not bool(torch.all(indices[1:] == indices[:-1] + 1)):
        raise ValueError("Isaac episode rows are missing or non-contiguous")
    if not bool(tensors["terminal"][indices[-1], environment_index]):
        raise ValueError("Isaac episode does not end in a terminal marker")
    return IsaacEpisodeTrace(
        path=path.resolve(),
        sha256=hash_file(path),
        metadata=metadata,
        tensors=tensors,
        environment_index=int(environment_index),
        episode_serial=int(episode_serial),
        trace_indices=indices,
    )


def _task_spec(trace: IsaacEpisodeTrace) -> dict[str, Any]:
    specs = trace.metadata.get("task_specs")
    if not isinstance(specs, list) or trace.environment_index >= len(specs):
        raise ValueError("Isaac trace lacks its per-environment TaskSpec")
    return dict(specs[trace.environment_index])


def _selected_anchors(trace: IsaacEpisodeTrace) -> tuple[dict[str, Any], ...]:
    morphology = trace.metadata["morphology_graph"]
    by_id = {int(anchor["anchor_id"]): anchor for anchor in morphology["robot_anchors"]}
    selected = tuple(int(value) for value in trace.metadata["selected_anchor_ids"])
    return tuple(dict(by_id[anchor_id]) for anchor_id in selected)


def resolve_source_urdf(case_manifest: Mapping[str, Any]) -> Path:
    usd = Path(str(case_manifest["robot_usd"]["path"]))
    if not usd.is_absolute():
        usd = Path.cwd() / usd
    asset_root = usd.resolve().parents[2]
    candidates = sorted(asset_root.glob("*.urdf"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one source URDF beside the bound USD, found {len(candidates)}"
        )
    return candidates[0]


def build_mujoco_model(
    *,
    trace: IsaacEpisodeTrace,
    source_urdf: Path,
    mesh_cache_directory: Path,
    physics_dt_s: float,
) -> MujocoModelContext:
    import mujoco

    if physics_dt_s <= 0.0:
        raise ValueError("MuJoCo physics dt must be positive")
    generated_urdf, mesh_manifest = materialize_convex_hull_urdf(
        source_urdf, mesh_cache_directory
    )
    spec = mujoco.MjSpec.from_file(str(generated_urdf))
    module_ids = tuple(int(value) for value in trace.metadata["module_ids"])
    for module_id in module_ids:
        spec.body(f"module_{module_id}__fc").add_site(
            name=f"module_{module_id}__fc_site", size=[0.001]
        )
    anchors = _selected_anchors(trace)
    for anchor in anchors:
        local_pose = tuple(float(value) for value in anchor["local_pose"])
        spec.body(f"module_{anchor['module_id']}__{anchor['link_id']}").add_site(
            name=f"selected_anchor_{anchor['anchor_id']}",
            pos=local_pose[:3],
            quat=(
                local_pose[6],
                local_pose[3],
                local_pose[4],
                local_pose[5],
            ),
            size=[0.001],
        )
    spec.body("module_0__root").add_freejoint(name="robot_free")

    task = _task_spec(trace)
    geometry = {
        value["geometry_id"]: value for value in task["scene"]["geometry_library"]
    }
    obj = next(
        value
        for value in task["scene"]["objects"]
        if value["object_id"] == trace.metadata["object_id"]
    )
    if geometry[obj["geometry_id"]]["geometry_type"] != "box":
        raise ValueError("MuJoCo comparison currently requires a box object")
    object_size = geometry[obj["geometry_id"]]["primitive_params"]["size_m"]
    support = task["scene"]["environment"]["support_surfaces"][0]
    support_size = geometry[support["geometry_id"]]["primitive_params"]["size_m"]
    support_pose = tuple(float(value) for value in support["pose_world"])
    spec.worldbody.add_geom(
        name="support",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * np.asarray(support_size)).tolist(),
        pos=support_pose[:3],
        quat=(
            support_pose[6],
            support_pose[3],
            support_pose[4],
            support_pose[5],
        ),
        contype=4,
        conaffinity=3,
        friction=[float(support.get("friction", 0.8)), 0.005, 0.0001],
    )
    payload = spec.worldbody.add_body(name="payload")
    payload.add_freejoint(name="payload_free")
    payload.add_geom(
        name="payload_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * np.asarray(object_size)).tolist(),
        contype=2,
        conaffinity=5,
        friction=[float(obj.get("friction", 0.6)), 0.005, 0.0001],
        mass=float(obj["mass_kg"]),
    )
    for geom in spec.geoms:
        if geom.name not in {"support", "payload_geom"}:
            # robot-object, robot-support and object-support are enabled;
            # robot-robot contacts are disabled like the Isaac articulation.
            geom.contype = 1
            geom.conaffinity = 6
    spec.option.timestep = float(physics_dt_s)
    spec.compiler.discardvisual = True
    model = spec.compile()

    def object_id(kind: Any, name: str) -> int:
        value = mujoco.mj_name2id(model, kind, name)
        if value < 0:
            raise ValueError(f"MuJoCo model lacks {name!r}")
        return int(value)

    root_body_id = object_id(mujoco.mjtObj.mjOBJ_BODY, "module_0__root")
    payload_body_id = object_id(mujoco.mjtObj.mjOBJ_BODY, "payload")
    support_geom_id = object_id(mujoco.mjtObj.mjOBJ_GEOM, "support")
    payload_geom_id = object_id(mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    module_sites = tuple(
        object_id(mujoco.mjtObj.mjOBJ_SITE, f"module_{value}__fc_site")
        for value in module_ids
    )
    selected_bodies = tuple(
        object_id(
            mujoco.mjtObj.mjOBJ_BODY,
            f"module_{anchor['module_id']}__{anchor['link_id']}",
        )
        for anchor in anchors
    )
    selected_sites = tuple(
        object_id(
            mujoco.mjtObj.mjOBJ_SITE,
            f"selected_anchor_{anchor['anchor_id']}",
        )
        for anchor in anchors
    )
    rotor_bodies = tuple(
        object_id(
            mujoco.mjtObj.mjOBJ_BODY,
            global_id.replace(":", "__"),
        )
        for global_id in trace.metadata["rotor_global_ids"]
    )
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    coefficient_by_rotor = {
        rotor.rotor_id: float(rotor.reaction_torque_coeff_nm_per_n)
        for rotor in physical.rotors
    }
    rotor_coefficients = np.asarray(
        [
            coefficient_by_rotor[str(global_id).split(":", 1)[1]]
            for global_id in trace.metadata["rotor_global_ids"]
        ],
        dtype=np.float64,
    )
    gimbal_joints = tuple(
        object_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            global_id.replace(":", "__"),
        )
        for global_id in trace.metadata["vectoring_global_joint_ids"]
    )
    dock_joints = tuple(
        tuple(
            object_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                f"module_{module_id}__{local_id}",
            )
            for local_id in trace.metadata["command_local_joint_ids"]
        )
        for module_id in module_ids
    )
    local_joint_ids = tuple(
        tuple(
            int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"module_{module_id}__{local_id}",
                )
            )
            for local_id in trace.metadata["local_joint_ids"]
        )
        for module_id in module_ids
    )
    robot_free = object_id(mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    payload_free = object_id(mujoco.mjtObj.mjOBJ_JOINT, "payload_free")

    actuator = order9_actuator_runtime_values(physical)
    for joint_id in gimbal_joints:
        dof = int(model.jnt_dofadr[joint_id])
        # Put the derivative term in MuJoCo's passive damping so the implicit
        # integrator treats the stiff local drive like PhysX's implicit drive.
        model.dof_damping[dof] = actuator.gimbal_damping
        model.dof_armature[dof] = actuator.gimbal_armature
    for joint_ids in dock_joints:
        for joint_id in joint_ids:
            dof = int(model.jnt_dofadr[joint_id])
            model.dof_damping[dof] = actuator.dock_damping
            model.dof_armature[dof] = actuator.dock_armature
    model.body_gravcomp[:] = 1.0
    model.body_gravcomp[0] = 0.0
    model.body_gravcomp[payload_body_id] = 0.0
    selected_friction = float(trace.metadata["selected_gripper_friction"])
    stiffness = float(trace.metadata["contact_stiffness_n_per_m"])
    damping = float(trace.metadata["contact_damping_n_s_per_m"])
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) in selected_bodies:
            model.geom_friction[geom_id] = (selected_friction, 0.005, 0.0001)
            model.geom_priority[geom_id] = 2
            # Negative solref selects MuJoCo's direct stiffness/damping form.
            model.geom_solref[geom_id] = (-stiffness, -damping)
    return MujocoModelContext(
        model=model,
        robot_root_body_id=root_body_id,
        payload_body_id=payload_body_id,
        support_geom_id=support_geom_id,
        payload_geom_id=payload_geom_id,
        module_site_ids=module_sites,
        selected_body_ids=selected_bodies,
        selected_site_ids=selected_sites,
        rotor_body_ids=rotor_bodies,
        rotor_reaction_coefficients=rotor_coefficients,
        gimbal_joint_ids=gimbal_joints,
        dock_joint_ids=dock_joints,
        local_joint_ids=local_joint_ids,
        robot_free_joint_id=robot_free,
        payload_free_joint_id=payload_free,
        gimbal_drive=(
            actuator.gimbal_stiffness,
            actuator.gimbal_damping,
            actuator.gimbal_effort_limit,
            actuator.gimbal_velocity_limit,
        ),
        dock_drive=(
            actuator.dock_stiffness,
            actuator.dock_damping,
            actuator.dock_effort_limit,
            actuator.dock_velocity_limit,
        ),
        mesh_bridge_manifest=mesh_manifest,
    )


def _xyzw_to_wxyz(value: Sequence[float]) -> np.ndarray:
    return np.asarray((value[3], value[0], value[1], value[2]), dtype=np.float64)


def _wxyz_to_xyzw(value: Sequence[float]) -> np.ndarray:
    return np.asarray((value[1], value[2], value[3], value[0]), dtype=np.float64)


def _quaternion_error_rad(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1.0e-12)
    right = right / max(float(np.linalg.norm(right)), 1.0e-12)
    return 2.0 * math.acos(min(abs(float(np.dot(left, right))), 1.0))


def _joint_qpos_dof(model: Any, joint_id: int) -> tuple[int, int]:
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def initialize_mujoco_state(
    context: MujocoModelContext, trace: IsaacEpisodeTrace
) -> Any:
    import mujoco

    model = context.model
    data = mujoco.MjData(model)
    tensor = trace.tensors
    env = trace.environment_index
    first = int(trace.trace_indices[0])
    root_pose = tensor["robot_root_pose_world"][first, env].numpy()
    root_twist = tensor["robot_root_twist_world"][first, env].numpy()
    qpos, dof = _joint_qpos_dof(model, context.robot_free_joint_id)
    data.qpos[qpos : qpos + 3] = root_pose[:3]
    data.qpos[qpos + 3 : qpos + 7] = _xyzw_to_wxyz(root_pose[3:7])
    data.qvel[dof : dof + 6] = root_twist

    object_pose = tensor["object_pose_world"][first, env].numpy()
    object_twist = tensor["object_twist_world"][first, env].numpy()
    qpos, dof = _joint_qpos_dof(model, context.payload_free_joint_id)
    data.qpos[qpos : qpos + 3] = object_pose[:3]
    data.qpos[qpos + 3 : qpos + 7] = _xyzw_to_wxyz(object_pose[3:7])
    data.qvel[dof : dof + 6] = object_twist

    for module_index, joint_ids in enumerate(context.local_joint_ids):
        for local_index, joint_id in enumerate(joint_ids):
            if joint_id < 0:
                continue
            qpos, dof = _joint_qpos_dof(model, joint_id)
            data.qpos[qpos] = float(
                tensor["local_joint_positions_rad"][
                    first, env, module_index, local_index
                ]
            )
            data.qvel[dof] = float(
                tensor["local_joint_velocities_radps"][
                    first, env, module_index, local_index
                ]
            )
    mujoco.mj_forward(model, data)
    return data


def _site_pose_and_twist(
    model: Any, data: Any, site_id: int
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    pose = np.concatenate((data.site_xpos[site_id].copy(), _wxyz_to_xyzw(quaternion)))
    spatial = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, spatial, 0)
    return pose, np.concatenate((spatial[3:6], spatial[0:3]))


def _body_pose_and_twist(
    model: Any, data: Any, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    pose = np.concatenate(
        (data.xpos[body_id].copy(), _wxyz_to_xyzw(data.xquat[body_id]))
    )
    spatial = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
    return pose, np.concatenate((spatial[3:6], spatial[0:3]))


def gather_mujoco_state(
    context: MujocoModelContext,
    data: Any,
    trace: IsaacEpisodeTrace,
) -> dict[str, np.ndarray]:
    import mujoco

    module_pose, module_twist = zip(
        *(
            _site_pose_and_twist(context.model, data, site_id)
            for site_id in context.module_site_ids
        ),
        strict=True,
    )
    selected_pose, selected_twist = zip(
        *(
            _site_pose_and_twist(context.model, data, site_id)
            for site_id in context.selected_site_ids
        ),
        strict=True,
    )
    local_ids = tuple(str(value) for value in trace.metadata["local_joint_ids"])
    joint_position = np.zeros(
        (len(context.module_site_ids), len(local_ids)), dtype=np.float64
    )
    joint_velocity = np.zeros_like(joint_position)
    for module_index, joint_ids in enumerate(context.local_joint_ids):
        for local_index, joint_id in enumerate(joint_ids):
            if joint_id < 0:
                continue
            qpos, dof = _joint_qpos_dof(context.model, joint_id)
            joint_position[module_index, local_index] = data.qpos[qpos]
            joint_velocity[module_index, local_index] = data.qvel[dof]
    root_pose, root_twist = _body_pose_and_twist(
        context.model, data, context.robot_root_body_id
    )
    object_pose, object_twist = _body_pose_and_twist(
        context.model, data, context.payload_body_id
    )
    return {
        "module_pose_world": np.asarray(module_pose),
        "module_twist_world": np.asarray(module_twist),
        "selected_pose_world": np.asarray(selected_pose),
        "selected_twist_world": np.asarray(selected_twist),
        "local_joint_positions_rad": joint_position,
        "local_joint_velocities_radps": joint_velocity,
        "robot_root_pose_world": root_pose,
        "robot_root_twist_world": root_twist,
        "object_pose_world": object_pose,
        "object_twist_world": object_twist,
    }


def _contact_evidence(
    context: MujocoModelContext,
    data: Any,
    *,
    contact_schedule_index: int,
    threshold_n: float = 0.5,
) -> tuple[np.ndarray, bool, dict[str, int | float]]:
    import mujoco

    selected_force = np.zeros((len(context.selected_body_ids), 3))
    prohibited = False
    robot_support_count = 0
    nonselected_object_count = 0
    selected_object_count = 0
    selected_by_body = {
        body_id: index for index, body_id in enumerate(context.selected_body_ids)
    }
    selected_contact_allowed = contact_schedule_index in (2, 3)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        force_contact = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(context.model, data, contact_index, force_contact)
        force_norm = float(np.linalg.norm(force_contact[:3]))
        if force_norm < threshold_n:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(context.model.geom_bodyid[geom1])
        body2 = int(context.model.geom_bodyid[geom2])
        bodies = {body1, body2}
        geoms = {geom1, geom2}
        robot_bodies = bodies - {0, context.payload_body_id}
        if context.support_geom_id in geoms and robot_bodies:
            prohibited = True
            robot_support_count += 1
        if context.payload_geom_id in geoms and robot_bodies:
            robot_body = next(iter(robot_bodies))
            if robot_body not in selected_by_body:
                prohibited = True
                nonselected_object_count += 1
            else:
                selected_object_count += 1
                if not selected_contact_allowed:
                    prohibited = True
                frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                force_world = frame.T @ force_contact[:3]
                selected_force[selected_by_body[robot_body]] += force_world
    return (
        selected_force,
        prohibited,
        {
            "selected_object_contact_count": selected_object_count,
            "nonselected_object_contact_count": nonselected_object_count,
            "robot_support_contact_count": robot_support_count,
        },
    )


def _apply_local_drives(
    context: MujocoModelContext,
    data: Any,
    *,
    vectoring_targets: np.ndarray,
    dock_position_targets: np.ndarray,
    dock_velocity_targets: np.ndarray,
    dock_torque_bias: np.ndarray,
) -> None:
    gimbal_stiffness, _, gimbal_effort, _ = context.gimbal_drive
    dock_stiffness, dock_damping, dock_effort, _ = context.dock_drive
    for offset, joint_id in enumerate(context.gimbal_joint_ids):
        qpos, dof = _joint_qpos_dof(context.model, joint_id)
        torque = gimbal_stiffness * (vectoring_targets[offset] - data.qpos[qpos])
        data.qfrc_applied[dof] += np.clip(torque, -gimbal_effort, gimbal_effort)
    for module_index, joint_ids in enumerate(context.dock_joint_ids):
        for local_index, joint_id in enumerate(joint_ids):
            qpos, dof = _joint_qpos_dof(context.model, joint_id)
            torque = (
                dock_stiffness
                * (dock_position_targets[module_index, local_index] - data.qpos[qpos])
                + dock_damping * dock_velocity_targets[module_index, local_index]
                + dock_torque_bias[module_index, local_index]
            )
            data.qfrc_applied[dof] += np.clip(torque, -dock_effort, dock_effort)


def _apply_rotor_wrenches(
    context: MujocoModelContext, data: Any, thrusts_n: np.ndarray
) -> None:
    data.xfrc_applied[:] = 0.0
    for index, body_id in enumerate(context.rotor_body_ids):
        rotation = data.xmat[body_id].reshape(3, 3)
        axis_world = rotation @ np.asarray((0.0, 0.0, 1.0))
        data.xfrc_applied[body_id, :3] = thrusts_n[index] * axis_world
        data.xfrc_applied[body_id, 3:6] = (
            thrusts_n[index] * context.rotor_reaction_coefficients[index] * axis_world
        )


def _count_joint_velocity_limit_exceedances(
    context: MujocoModelContext, data: Any
) -> int:
    gimbal_limit = context.gimbal_drive[3]
    dock_limit = context.dock_drive[3]
    clipped = 0
    for joint_id in context.gimbal_joint_ids:
        _, dof = _joint_qpos_dof(context.model, joint_id)
        before = float(data.qvel[dof])
        clipped += int(abs(before) > gimbal_limit)
    for joint_ids in context.dock_joint_ids:
        for joint_id in joint_ids:
            _, dof = _joint_qpos_dof(context.model, joint_id)
            before = float(data.qvel[dof])
            clipped += int(abs(before) > dock_limit)
    return clipped


def _tensor(value: np.ndarray | Sequence[float], *, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype).unsqueeze(0)


def _payload_offset_body(
    body_pose_world: torch.Tensor,
    object_pose_world: torch.Tensor,
    estimated_com_object: torch.Tensor,
) -> torch.Tensor:
    return Order9TensorPiLRuntime._payload_offset_body(
        body_pose_world,
        object_pose_world,
        estimated_payload_com_object=estimated_com_object,
    )


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "final": float(array[-1]),
    }


def _quaternion_error_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1.0e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1.0e-12)
    dot = np.clip(np.abs(np.sum(left * right, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def replay_saved_actuator_trace_mujoco(
    *,
    trace: IsaacEpisodeTrace,
    context: MujocoModelContext,
    control_dt_s: float = 0.02,
) -> dict[str, Any]:
    """Replay the exact Isaac actuator outputs to isolate physics speed."""

    import mujoco

    ratio = control_dt_s / float(context.model.opt.timestep)
    substeps = int(round(ratio))
    if substeps < 1 or not math.isclose(
        substeps * float(context.model.opt.timestep),
        control_dt_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("control dt must be an integral multiple of physics dt")
    data = initialize_mujoco_state(context, trace)
    tensors = trace.tensors
    env = trace.environment_index
    indices = trace.trace_indices
    initial_index = int(indices[0])
    initial = gather_mujoco_state(context, data, trace)
    initial_reference = tensors["module_pose_world"][initial_index, env].numpy()
    initial_module_position_error = np.linalg.norm(
        initial["module_pose_world"][:, :3] - initial_reference[:, :3], axis=-1
    )
    initial_module_orientation_error = _quaternion_error_array(
        initial["module_pose_world"][:, 3:7], initial_reference[:, 3:7]
    )

    module_pose: list[np.ndarray] = []
    module_twist: list[np.ndarray] = []
    local_joint_position: list[np.ndarray] = []
    root_pose: list[np.ndarray] = []
    object_pose: list[np.ndarray] = []
    object_twist: list[np.ndarray] = []
    selected_twist: list[np.ndarray] = []
    selected_force: list[np.ndarray] = []
    prohibited: list[bool] = []
    contact_counts: list[dict[str, int | float]] = []
    velocity_limit_exceedance_count = 0
    loop_started = time.perf_counter()
    for transition_index in range(trace.control_step_count):
        trace_index = int(indices[transition_index])
        thrusts = tensors["rotor_thrusts_n"][trace_index, env].numpy()
        vectoring = tensors["vectoring_joint_targets_rad"][trace_index, env].numpy()
        dock_q = tensors["command_joint_position_targets_rad"][trace_index, env].numpy()
        dock_qdot = tensors["command_joint_velocity_targets_radps"][
            trace_index, env
        ].numpy()
        dock_bias = tensors["command_joint_torque_bias_nm"][trace_index, env].numpy()
        for _ in range(substeps):
            data.qfrc_applied[:] = 0.0
            _apply_rotor_wrenches(context, data, thrusts)
            _apply_local_drives(
                context,
                data,
                vectoring_targets=vectoring,
                dock_position_targets=dock_q,
                dock_velocity_targets=dock_qdot,
                dock_torque_bias=dock_bias,
            )
            mujoco.mj_step(context.model, data)
            velocity_limit_exceedance_count += _count_joint_velocity_limit_exceedances(
                context, data
            )
        state = gather_mujoco_state(context, data, trace)
        forces, collision, counts = _contact_evidence(
            context,
            data,
            contact_schedule_index=int(
                tensors["contact_schedule_index"][trace_index, env]
            ),
        )
        module_pose.append(state["module_pose_world"])
        module_twist.append(state["module_twist_world"])
        local_joint_position.append(state["local_joint_positions_rad"])
        root_pose.append(state["robot_root_pose_world"])
        object_pose.append(state["object_pose_world"])
        object_twist.append(state["object_twist_world"])
        selected_twist.append(state["selected_twist_world"])
        selected_force.append(forces)
        prohibited.append(collision)
        contact_counts.append(counts)
    physics_loop_wall_s = time.perf_counter() - loop_started

    module_pose_array = np.asarray(module_pose)
    module_twist_array = np.asarray(module_twist)
    local_joint_array = np.asarray(local_joint_position)
    root_pose_array = np.asarray(root_pose)
    object_pose_array = np.asarray(object_pose)
    object_twist_array = np.asarray(object_twist)
    selected_twist_array = np.asarray(selected_twist)
    selected_force_array = np.asarray(selected_force)
    prohibited_array = np.asarray(prohibited, dtype=bool)
    morphology = MorphologyGraph.from_dict(trace.metadata["morphology_graph"])
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    builder = BatchedRigidBodyControlModelBuilder(morphology, physical)
    body = builder.build(
        module_pose_world=torch.as_tensor(module_pose_array, dtype=torch.float32),
        module_twist_world=torch.as_tensor(module_twist_array, dtype=torch.float32),
        local_joint_positions_rad=torch.as_tensor(
            local_joint_array, dtype=torch.float32
        ),
    )
    body_pose = body.body_pose_world.numpy()
    body_twist = body.body_twist_world.numpy()
    next_indices = indices[1:].numpy()
    reference_root = tensors["robot_root_pose_world"][next_indices, env].numpy()
    reference_object = tensors["object_pose_world"][next_indices, env].numpy()
    reference_body = builder.build(
        module_pose_world=tensors["module_pose_world"][next_indices, env],
        module_twist_world=tensors["module_twist_world"][next_indices, env],
        local_joint_positions_rad=tensors["local_joint_positions_rad"][
            next_indices, env
        ],
    ).body_pose_world.numpy()
    command_joint_indices = [
        trace.metadata["local_joint_ids"].index(value)
        for value in trace.metadata["command_local_joint_ids"]
    ]
    reference_joint = tensors["local_joint_positions_rad"][next_indices, env].numpy()[
        :, :, command_joint_indices
    ]
    joint_error = np.max(
        np.abs(local_joint_array[:, :, command_joint_indices] - reference_joint),
        axis=(1, 2),
    )

    task = _task_spec(trace)
    geometry_by_id = {
        value["geometry_id"]: value for value in task["scene"]["geometry_library"]
    }
    obj = next(
        value
        for value in task["scene"]["objects"]
        if value["object_id"] == trace.metadata["object_id"]
    )
    object_half_height = 0.5 * float(
        geometry_by_id[obj["geometry_id"]]["primitive_params"]["size_m"][2]
    )
    support = task["scene"]["environment"]["support_surfaces"][0]
    support_height = float(
        geometry_by_id[support["geometry_id"]]["primitive_params"]["size_m"][2]
    )
    support_top = float(support["pose_world"][2]) + 0.5 * support_height
    phase_success_seen = {index: False for index in range(8)}
    contact_dwell = 0.0
    contact_break = 0.0
    contact_free_dwell = 0.0
    settle_dwell = 0.0
    qp_dwell = 0.0
    grasp_acquired = False
    terminal_failure = False
    first_failure: dict[str, Any] | None = None
    two_contact_tick_count = 0
    previous_phase = -1
    phase_elapsed_s = 0.0
    for step, trace_index_value in enumerate(indices[:-1]):
        trace_index = int(trace_index_value)
        phase = int(tensors["phase_index"][trace_index, env])
        if phase != previous_phase:
            previous_phase = phase
            phase_elapsed_s = 0.0
        duration = float(
            trace.metadata["phase_duration_s"][
                trace.metadata["runtime_phase_labels"][phase]
            ]
        )
        elapsed = phase_elapsed_s + control_dt_s
        phase_elapsed_s = elapsed
        active_count = int(
            np.sum(np.linalg.norm(selected_force_array[step], axis=-1) >= 0.5)
        )
        enough_contact = active_count >= 2
        two_contact_tick_count += int(enough_contact)
        qp_feasible = bool(tensors["qp_feasible"][trace_index, env])
        grasp_ready = enough_contact and qp_feasible
        contact_dwell = contact_dwell + control_dt_s if grasp_ready else 0.0
        grasp_acquired = grasp_acquired or contact_dwell >= 0.25
        contact_break = (
            contact_break + control_dt_s
            if grasp_acquired and not enough_contact
            else 0.0
        )
        no_contact = active_count == 0
        contact_free_dwell = contact_free_dwell + control_dt_s if no_contact else 0.0
        qp_dwell = qp_dwell + control_dt_s if not qp_feasible else 0.0
        desired_object = tensors["phase_goal_object_pose_world"][
            trace_index, env
        ].numpy()
        object_position_error = float(
            np.linalg.norm(object_pose_array[step, :3] - desired_object[:3])
        )
        object_orientation_error = _quaternion_error_rad(
            object_pose_array[step, 3:7], desired_object[3:7]
        )
        object_pose_ok = (
            object_position_error <= 0.052 and object_orientation_error <= 0.2
        )
        desired_body = tensors["phase_goal_body_pose_world"][trace_index, env].numpy()
        robot_position_error = float(
            np.linalg.norm(body_pose[step, :3] - desired_body[:3])
        )
        robot_orientation_error = _quaternion_error_rad(
            body_pose[step, 3:7], desired_body[3:7]
        )
        robot_speed = float(np.linalg.norm(body_twist[step, :3]))
        object_linear_speed = float(np.linalg.norm(object_twist_array[step, :3]))
        object_angular_speed = float(np.linalg.norm(object_twist_array[step, 3:6]))
        settled = (
            object_pose_ok
            and object_linear_speed <= 0.05
            and object_angular_speed <= 0.10
        )
        settle_dwell = settle_dwell + control_dt_s if settled else 0.0
        release_valid = contact_free_dwell >= 0.10 and object_pose_ok
        support_clearance = (
            object_pose_array[step, 2] - object_half_height - support_top
        )
        phase_success = False
        if phase == 0:
            phase_success = (
                robot_position_error <= 0.08
                and robot_orientation_error <= 0.20
                and robot_speed <= 0.02
                and qp_feasible
            )
        elif phase == 1:
            phase_success = contact_dwell >= 0.25
        elif phase == 2:
            phase_success = (
                object_pose_ok and support_clearance >= 0.001 and grasp_ready
            )
        elif phase in (3, 4):
            phase_success = object_pose_ok and grasp_ready
        elif phase == 5:
            phase_success = release_valid
        elif phase == 6:
            phase_success = robot_position_error <= 0.05 and no_contact and qp_feasible
        elif phase == 7:
            phase_success = settle_dwell >= 1.0
        maintain = phase in (2, 3, 4)
        object_dropped = maintain and (
            contact_break >= 0.05
            or object_twist_array[step, 2] < -0.25
            or support_clearance < -0.052
        )
        failure = (
            bool(prohibited_array[step])
            or object_dropped
            or qp_dwell >= 0.10
            or (elapsed >= duration and not phase_success)
        )
        phase_success_seen[phase] |= phase_success and not failure
        if failure and first_failure is None:
            first_failure = {
                "control_step": step,
                "time_s": step * control_dt_s,
                "phase_index": phase,
                "hard_collision": bool(prohibited_array[step]),
                "object_dropped": bool(object_dropped),
                "qp_infeasible_terminal": bool(qp_dwell >= 0.10),
                "timeout": bool(elapsed >= duration and not phase_success),
            }
        terminal_failure |= failure
    evaluation_wall_s = time.perf_counter() - loop_started - physics_loop_wall_s
    terminal_index = int(indices[-1])
    terminal_goal = tensors["desired_object_pose_world"][terminal_index, env].numpy()
    terminal_position_error = float(
        np.linalg.norm(object_pose_array[-1, :3] - terminal_goal[:3])
    )
    terminal_orientation_error = _quaternion_error_rad(
        object_pose_array[-1, 3:7], terminal_goal[3:7]
    )
    present_phases = sorted(
        {
            int(tensors["phase_index"][int(value), env])
            for value in indices[:-1]
            if int(tensors["phase_index"][int(value), env]) < 8
        }
    )
    all_present_phase_gates = all(phase_success_seen[value] for value in present_phases)
    success = (
        not terminal_failure
        and all_present_phase_gates
        and terminal_position_error <= 0.052
        and terminal_orientation_error <= 0.2
    )
    simulated_duration_s = trace.control_step_count * control_dt_s
    maximum_counts = {
        name: max(int(value[name]) for value in contact_counts)
        for name in contact_counts[0]
    }
    return {
        "version": ORDER9_MUJOCO_TRACE_REPLAY_VERSION,
        "comparison_scope": "single-candidate diagnostic; not R1 formal range evidence",
        "physics": {
            "engine": "MuJoCo",
            "version": mujoco.__version__,
            "physics_dt_s": float(context.model.opt.timestep),
            "control_dt_s": float(control_dt_s),
            "physics_substeps_per_control": substeps,
            "self_collision_enabled": False,
            "robot_gravity_enabled": False,
            "payload_gravity_enabled": True,
        },
        "timing": {
            "control_step_count": trace.control_step_count,
            "simulated_duration_s": simulated_duration_s,
            "physics_replay_wall_s": physics_loop_wall_s,
            "acceptance_evaluation_wall_s": evaluation_wall_s,
            "simulation_wall_s": physics_loop_wall_s,
            "realtime_factor": simulated_duration_s / physics_loop_wall_s,
            "control_steps_per_wall_s": trace.control_step_count / physics_loop_wall_s,
            "isaac_saved_batch_rollout_wall_s": float(
                trace.metadata["rollout_wall_elapsed_s"]
            ),
            "isaac_saved_batch_environment_count": int(
                trace.metadata["environment_count"]
            ),
            "isaac_saved_aggregate_env_steps_per_s": float(
                trace.metadata["aggregate_env_steps_per_s"]
            ),
            "throughput_ratio_vs_isaac_aggregate_env_steps": (
                (trace.control_step_count / physics_loop_wall_s)
                / float(trace.metadata["aggregate_env_steps_per_s"])
            ),
        },
        "bridge_identity": {
            "isaac_rollout_path": str(trace.path),
            "isaac_rollout_sha256": trace.sha256,
            "environment_index": trace.environment_index,
            "episode_serial": trace.episode_serial,
            "candidate_generation_id": trace.metadata["generation_id"],
            "pi_l_actor_command_applied": bool(
                trace.metadata["pi_l_actor_command_applied"]
            ),
            "controller_mode": "saved exact Isaac actuator output replay",
            "mesh_bridge": context.mesh_bridge_manifest,
        },
        "initial_state_alignment": {
            "maximum_module_position_error_m": float(
                np.max(initial_module_position_error)
            ),
            "maximum_module_orientation_error_rad": float(
                np.max(initial_module_orientation_error)
            ),
        },
        "trajectory_error_mujoco_vs_isaac": {
            "robot_root_position_m": _metric_summary(
                np.linalg.norm(root_pose_array[:, :3] - reference_root[:, :3], axis=-1)
            ),
            "robot_root_orientation_rad": _metric_summary(
                _quaternion_error_array(root_pose_array[:, 3:7], reference_root[:, 3:7])
            ),
            "centroidal_body_position_m": _metric_summary(
                np.linalg.norm(body_pose[:, :3] - reference_body[:, :3], axis=-1)
            ),
            "centroidal_body_orientation_rad": _metric_summary(
                _quaternion_error_array(body_pose[:, 3:7], reference_body[:, 3:7])
            ),
            "object_position_m": _metric_summary(
                np.linalg.norm(
                    object_pose_array[:, :3] - reference_object[:, :3], axis=-1
                )
            ),
            "object_orientation_rad": _metric_summary(
                _quaternion_error_array(
                    object_pose_array[:, 3:7], reference_object[:, 3:7]
                )
            ),
            "maximum_command_joint_position_rad": _metric_summary(joint_error),
        },
        "acceptance": {
            "isaac_reference_success": bool(
                tensors["actor_task_success"][terminal_index, env]
            ),
            "mujoco_success": success,
            "terminal_failure": terminal_failure,
            "first_failure": first_failure,
            "present_phase_indices": present_phases,
            "phase_success_seen": {
                str(index): bool(phase_success_seen[index]) for index in present_phases
            },
            "all_present_phase_gates_satisfied": all_present_phase_gates,
            "terminal_object_position_error_m": terminal_position_error,
            "terminal_object_orientation_error_rad": terminal_orientation_error,
            "object_position_tolerance_m": 0.052,
            "object_orientation_tolerance_rad": 0.2,
            "prohibited_collision_tick_count": int(np.sum(prohibited_array)),
            "two_selected_contacts_tick_count": two_contact_tick_count,
            "maximum_contact_counts": maximum_counts,
            "joint_velocity_limit_exceedance_substep_count": (
                velocity_limit_exceedance_count
            ),
        },
    }


def replay_closed_loop_mujoco(
    *,
    trace: IsaacEpisodeTrace,
    context: MujocoModelContext,
    control_dt_s: float = 0.02,
) -> dict[str, Any]:
    """Run the saved target trajectory through the production QPID/QP."""

    import mujoco

    ratio = control_dt_s / float(context.model.opt.timestep)
    substeps = int(round(ratio))
    if substeps < 1 or not math.isclose(
        substeps * float(context.model.opt.timestep),
        control_dt_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("control dt must be an integral multiple of physics dt")
    data = initialize_mujoco_state(context, trace)
    tensors = trace.tensors
    env = trace.environment_index
    indices = trace.trace_indices
    morphology = MorphologyGraph.from_dict(trace.metadata["morphology_graph"])
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    builder = BatchedRigidBodyControlModelBuilder(morphology, physical)
    actual_controller = BatchedQPIDController()
    actual_controller_state = actual_controller.initial_state(
        1, builder.rotor_count, device=torch.device("cpu"), dtype=torch.float32
    )
    estimated_mass = torch.tensor(
        [trace.metadata["estimated_payload_mass_kg"]], dtype=torch.float32
    )
    estimated_inertia = torch.tensor(
        [trace.metadata["estimated_payload_inertia_body"]], dtype=torch.float32
    )
    estimated_com = torch.tensor(
        [trace.metadata["estimated_payload_com_object"]], dtype=torch.float32
    )
    reward_engine = Order9TensorRewardEngine(control_dt_s=control_dt_s)
    initial_index = int(indices[0])
    reward_state = reward_engine.initial_state(
        object_pose_world=_tensor(
            gather_mujoco_state(context, data, trace)["object_pose_world"]
        ),
        desired_object_pose_world=tensors["phase_goal_object_pose_world"][
            initial_index, env : env + 1
        ],
    )
    local_index_by_id = {
        value: index for index, value in enumerate(trace.metadata["local_joint_ids"])
    }
    command_joint_indices = [
        local_index_by_id[value] for value in trace.metadata["command_local_joint_ids"]
    ]
    task = _task_spec(trace)
    object_geometry_id = next(
        value["geometry_id"]
        for value in task["scene"]["objects"]
        if value["object_id"] == trace.metadata["object_id"]
    )
    geometry_by_id = {
        value["geometry_id"]: value for value in task["scene"]["geometry_library"]
    }
    object_half_height = 0.5 * float(
        geometry_by_id[object_geometry_id]["primitive_params"]["size_m"][2]
    )
    support = task["scene"]["environment"]["support_surfaces"][0]
    support_height = float(
        geometry_by_id[support["geometry_id"]]["primitive_params"]["size_m"][2]
    )
    support_top = float(support["pose_world"][2]) + 0.5 * support_height

    root_position_errors: list[float] = []
    root_orientation_errors: list[float] = []
    body_position_errors: list[float] = []
    body_orientation_errors: list[float] = []
    object_position_errors: list[float] = []
    object_orientation_errors: list[float] = []
    command_joint_errors: list[float] = []
    phase_success_seen = {index: False for index in range(8)}
    terminal_failure = False
    first_failure: dict[str, Any] | None = None
    prohibited_collision_count = 0
    selected_contact_tick_count = 0
    joint_velocity_limit_exceedance_count = 0
    maximum_contact_counts = {
        "selected_object_contact_count": 0,
        "nonselected_object_contact_count": 0,
        "robot_support_contact_count": 0,
    }
    initial = gather_mujoco_state(context, data, trace)
    initial_module_position_error = np.linalg.norm(
        initial["module_pose_world"][:, :3]
        - tensors["module_pose_world"][initial_index, env, :, :3].numpy(),
        axis=-1,
    )
    initial_module_orientation_error = np.asarray(
        [
            _quaternion_error_rad(left, right)
            for left, right in zip(
                initial["module_pose_world"][:, 3:7],
                tensors["module_pose_world"][initial_index, env, :, 3:7].numpy(),
                strict=True,
            )
        ]
    )

    started = time.perf_counter()
    previous_phase = -1
    phase_elapsed_s = 0.0
    for transition_index in range(trace.control_step_count):
        trace_index = int(indices[transition_index])
        next_trace_index = int(indices[transition_index + 1])
        phase = tensors["phase_index"][trace_index, env : env + 1].long()
        if int(phase.item()) >= 8:
            break
        if int(phase.item()) != previous_phase:
            previous_phase = int(phase.item())
            phase_elapsed_s = 0.0
        post_step_phase_elapsed_s = phase_elapsed_s + control_dt_s
        phase_elapsed_s = post_step_phase_elapsed_s
        phase_progress = tensors["phase_progress"][trace_index, env : env + 1]
        payload_scale = order9_payload_feedforward_scale(phase, phase_progress)

        pre = gather_mujoco_state(context, data, trace)
        actual_model = builder.build(
            module_pose_world=_tensor(pre["module_pose_world"]),
            module_twist_world=_tensor(pre["module_twist_world"]),
            local_joint_positions_rad=_tensor(pre["local_joint_positions_rad"]),
        )
        actual_offset = _payload_offset_body(
            actual_model.body_pose_world,
            _tensor(pre["object_pose_world"]),
            estimated_com,
        )
        actual_result = actual_controller.compute(
            control_model=actual_model,
            desired_body_pose_world=tensors["command_body_pose_world"][
                trace_index, env : env + 1
            ],
            desired_body_twist=tensors["command_body_twist"][
                trace_index, env : env + 1
            ],
            residual_wrench_body=tensors["command_residual_wrench_body"][
                trace_index, env : env + 1
            ],
            state=actual_controller_state,
            payload_active=payload_scale > 0.0,
            payload_mass_kg=estimated_mass * payload_scale,
            payload_inertia_body=estimated_inertia * payload_scale.unsqueeze(-1),
            payload_com_offset_body=actual_offset,
        )
        actual_controller_state = actual_result.next_state
        thrusts = actual_result.allocation.rotor_thrusts_n[0].numpy()
        vectoring = actual_result.allocation.vectoring_joint_targets_rad[0].numpy()
        dock_q = tensors["command_joint_position_targets_rad"][trace_index, env].numpy()
        dock_qdot = tensors["command_joint_velocity_targets_radps"][
            trace_index, env
        ].numpy()
        dock_bias = tensors["command_joint_torque_bias_nm"][trace_index, env].numpy()
        for _ in range(substeps):
            data.qfrc_applied[:] = 0.0
            _apply_rotor_wrenches(context, data, thrusts)
            _apply_local_drives(
                context,
                data,
                vectoring_targets=vectoring,
                dock_position_targets=dock_q,
                dock_velocity_targets=dock_qdot,
                dock_torque_bias=dock_bias,
            )
            mujoco.mj_step(context.model, data)
            joint_velocity_limit_exceedance_count += (
                _count_joint_velocity_limit_exceedances(context, data)
            )
        post = gather_mujoco_state(context, data, trace)
        selected_force, prohibited, contact_counts = _contact_evidence(
            context,
            data,
            contact_schedule_index=int(
                tensors["contact_schedule_index"][trace_index, env]
            ),
        )
        prohibited_collision_count += int(prohibited)
        selected_contact_tick_count += int(
            np.sum(np.linalg.norm(selected_force, axis=-1) >= 0.5) >= 2
        )
        for name, value in contact_counts.items():
            maximum_contact_counts[name] = max(maximum_contact_counts[name], int(value))

        actual_body = builder.build(
            module_pose_world=_tensor(post["module_pose_world"]),
            module_twist_world=_tensor(post["module_twist_world"]),
            local_joint_positions_rad=_tensor(post["local_joint_positions_rad"]),
        )
        reference_root = tensors["robot_root_pose_world"][next_trace_index, env].numpy()
        reference_object = tensors["object_pose_world"][next_trace_index, env].numpy()
        reference_body = (
            builder.build(
                module_pose_world=tensors["module_pose_world"][
                    next_trace_index, env : env + 1
                ],
                module_twist_world=tensors["module_twist_world"][
                    next_trace_index, env : env + 1
                ],
                local_joint_positions_rad=tensors["local_joint_positions_rad"][
                    next_trace_index, env : env + 1
                ],
            )
            .body_pose_world[0]
            .numpy()
        )
        root_position_errors.append(
            float(
                np.linalg.norm(post["robot_root_pose_world"][:3] - reference_root[:3])
            )
        )
        root_orientation_errors.append(
            _quaternion_error_rad(
                post["robot_root_pose_world"][3:7], reference_root[3:7]
            )
        )
        body_position_errors.append(
            float(
                np.linalg.norm(
                    actual_body.body_pose_world[0, :3].numpy() - reference_body[:3]
                )
            )
        )
        body_orientation_errors.append(
            _quaternion_error_rad(
                actual_body.body_pose_world[0, 3:7].numpy(),
                reference_body[3:7],
            )
        )
        object_position_errors.append(
            float(np.linalg.norm(post["object_pose_world"][:3] - reference_object[:3]))
        )
        object_orientation_errors.append(
            _quaternion_error_rad(post["object_pose_world"][3:7], reference_object[3:7])
        )
        command_joint_errors.append(
            float(
                np.max(
                    np.abs(
                        post["local_joint_positions_rad"][:, command_joint_indices]
                        - tensors["local_joint_positions_rad"][
                            next_trace_index,
                            env,
                            :,
                            command_joint_indices,
                        ].numpy()
                    )
                )
            )
        )

        duration = float(
            trace.metadata["phase_duration_s"][
                trace.metadata["runtime_phase_labels"][int(phase.item())]
            ]
        )
        evidence = Order9TensorRewardInput(
            phase_index=phase,
            phase_elapsed_s=torch.tensor(
                [post_step_phase_elapsed_s], dtype=torch.float32
            ),
            phase_duration_s=torch.tensor([duration], dtype=torch.float32),
            robot_body_pose_world=actual_body.body_pose_world,
            robot_body_twist_world=actual_body.body_twist_world,
            module_twist_world=_tensor(post["module_twist_world"]),
            object_pose_world=_tensor(post["object_pose_world"]),
            object_twist_world=_tensor(post["object_twist_world"]),
            desired_robot_pose_world=tensors["phase_goal_body_pose_world"][
                trace_index, env : env + 1
            ],
            desired_object_pose_world=tensors["phase_goal_object_pose_world"][
                trace_index, env : env + 1
            ],
            local_joint_positions_rad=_tensor(
                post["local_joint_positions_rad"][:, command_joint_indices]
            ),
            phase_goal_joint_positions_rad=tensors["desired_joint_positions_rad"][
                trace_index, env : env + 1
            ],
            selected_contact_forces_world=_tensor(selected_force),
            selected_contact_wrenches_contact=tensors[
                "selected_contact_wrenches_contact"
            ][trace_index, env : env + 1],
            wrench_lower_contact=tensors["wrench_lower_contact"][
                trace_index, env : env + 1
            ],
            wrench_upper_contact=tensors["wrench_upper_contact"][
                trace_index, env : env + 1
            ],
            wrench_bound_mask=tensors["wrench_bound_mask"][trace_index, env : env + 1],
            selected_link_twist_world=_tensor(post["selected_twist_world"]),
            selected_contact_mask=torch.ones(
                (1, len(context.selected_body_ids)), dtype=torch.bool
            ),
            selected_grasp_frame_normal_error_m=torch.zeros(
                (1, len(context.selected_body_ids)), dtype=torch.float32
            ),
            selected_relative_normal_velocity_mps=torch.zeros(
                (1, len(context.selected_body_ids)), dtype=torch.float32
            ),
            selected_motor_load_proxy=torch.zeros(
                (1, len(context.selected_body_ids)), dtype=torch.float32
            ),
            target_motor_load_proxy=torch.zeros(
                (1, len(context.selected_body_ids)), dtype=torch.float32
            ),
            contact_preload_complete=torch.ones(1, dtype=torch.bool),
            prohibited_collision=torch.tensor([prohibited], dtype=torch.bool),
            support_top_z_m=torch.tensor([support_top], dtype=torch.float32),
            object_half_height_m=torch.tensor(
                [object_half_height], dtype=torch.float32
            ),
            qp_feasible=actual_result.allocation.feasible,
            allocation_residual_norm=actual_result.allocation.residual_norm,
            rotor_thrusts_n=actual_result.allocation.rotor_thrusts_n,
            rotor_saturation=(
                actual_result.allocation.thrust_clipped
                | actual_result.allocation.vectoring_clipped
            ),
            joint_torque_bias_nm=tensors["command_joint_torque_bias_nm"][
                trace_index, env : env + 1
            ],
        )
        reward = reward_engine.step(evidence, reward_state)
        reward_state = reward.next_state
        phase_success_seen[int(phase.item())] |= bool(reward.phase_success[0])
        if bool(reward.terminal_failure[0]) and first_failure is None:
            first_failure = {
                "control_step": transition_index,
                "time_s": transition_index * control_dt_s,
                "phase_index": int(phase.item()),
                "hard_collision": bool(reward.hard_collision[0]),
                "object_dropped": bool(reward.object_dropped[0]),
                "qp_infeasible_terminal": bool(reward.qp_infeasible_terminal[0]),
                "timeout": bool(reward.timeout[0]),
            }
        terminal_failure |= bool(reward.terminal_failure[0])
    simulation_wall_s = time.perf_counter() - started
    final_state = gather_mujoco_state(context, data, trace)
    terminal_index = int(indices[-1])
    terminal_goal = tensors["desired_object_pose_world"][terminal_index, env].numpy()
    terminal_position_error = float(
        np.linalg.norm(final_state["object_pose_world"][:3] - terminal_goal[:3])
    )
    terminal_orientation_error = _quaternion_error_rad(
        final_state["object_pose_world"][3:7], terminal_goal[3:7]
    )
    present_phases = sorted(
        {
            int(tensors["phase_index"][int(value), env])
            for value in indices[:-1]
            if int(tensors["phase_index"][int(value), env]) < 8
        }
    )
    all_present_phase_gates = all(phase_success_seen[value] for value in present_phases)
    success = (
        not terminal_failure
        and all_present_phase_gates
        and terminal_position_error <= 0.052
        and terminal_orientation_error <= 0.2
    )
    simulated_duration_s = trace.control_step_count * control_dt_s
    return {
        "version": ORDER9_MUJOCO_TRACE_REPLAY_VERSION,
        "comparison_scope": (
            "single-candidate diagnostic; not R1 formal range evidence"
        ),
        "physics": {
            "engine": "MuJoCo",
            "version": mujoco.__version__,
            "physics_dt_s": float(context.model.opt.timestep),
            "control_dt_s": float(control_dt_s),
            "physics_substeps_per_control": substeps,
            "self_collision_enabled": False,
            "robot_gravity_enabled": False,
            "payload_gravity_enabled": True,
            "selected_contact_direct_stiffness_n_per_m": float(
                trace.metadata["contact_stiffness_n_per_m"]
            ),
            "selected_contact_direct_damping_n_s_per_m": float(
                trace.metadata["contact_damping_n_s_per_m"]
            ),
        },
        "timing": {
            "control_step_count": trace.control_step_count,
            "simulated_duration_s": simulated_duration_s,
            "simulation_wall_s": simulation_wall_s,
            "realtime_factor": simulated_duration_s / simulation_wall_s,
            "control_steps_per_wall_s": trace.control_step_count / simulation_wall_s,
            "isaac_saved_batch_rollout_wall_s": float(
                trace.metadata["rollout_wall_elapsed_s"]
            ),
            "isaac_saved_batch_environment_count": int(
                trace.metadata["environment_count"]
            ),
            "isaac_saved_aggregate_env_steps_per_s": float(
                trace.metadata["aggregate_env_steps_per_s"]
            ),
            "throughput_ratio_vs_isaac_aggregate_env_steps": (
                (trace.control_step_count / simulation_wall_s)
                / float(trace.metadata["aggregate_env_steps_per_s"])
            ),
        },
        "bridge_identity": {
            "isaac_rollout_path": str(trace.path),
            "isaac_rollout_sha256": trace.sha256,
            "environment_index": trace.environment_index,
            "episode_serial": trace.episode_serial,
            "candidate_generation_id": trace.metadata["generation_id"],
            "pi_l_actor_command_applied": bool(
                trace.metadata["pi_l_actor_command_applied"]
            ),
            "controller_mode": "production nominal QPID/QP closed loop",
            "mesh_bridge": context.mesh_bridge_manifest,
        },
        "initial_state_alignment": {
            "maximum_module_position_error_m": float(
                np.max(initial_module_position_error)
            ),
            "maximum_module_orientation_error_rad": float(
                np.max(initial_module_orientation_error)
            ),
        },
        "controller_reproduction_on_saved_isaac_states": {
            "evaluated": False,
            "reason": "excluded from timed closed-loop replay",
        },
        "trajectory_error_mujoco_vs_isaac": {
            "robot_root_position_m": _metric_summary(root_position_errors),
            "robot_root_orientation_rad": _metric_summary(root_orientation_errors),
            "centroidal_body_position_m": _metric_summary(body_position_errors),
            "centroidal_body_orientation_rad": _metric_summary(body_orientation_errors),
            "object_position_m": _metric_summary(object_position_errors),
            "object_orientation_rad": _metric_summary(object_orientation_errors),
            "maximum_command_joint_position_rad": _metric_summary(command_joint_errors),
        },
        "acceptance": {
            "isaac_reference_success": bool(
                tensors["actor_task_success"][terminal_index, env]
            ),
            "mujoco_success": success,
            "terminal_failure": terminal_failure,
            "first_failure": first_failure,
            "present_phase_indices": present_phases,
            "phase_success_seen": {
                str(index): bool(phase_success_seen[index]) for index in present_phases
            },
            "all_present_phase_gates_satisfied": all_present_phase_gates,
            "terminal_object_position_error_m": terminal_position_error,
            "terminal_object_orientation_error_rad": terminal_orientation_error,
            "object_position_tolerance_m": 0.052,
            "object_orientation_tolerance_rad": 0.2,
            "prohibited_collision_tick_count": prohibited_collision_count,
            "two_selected_contacts_tick_count": selected_contact_tick_count,
            "maximum_contact_counts": maximum_contact_counts,
            "joint_velocity_limit_exceedance_substep_count": (
                joint_velocity_limit_exceedance_count
            ),
        },
    }


def write_comparison_result(result: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "IsaacEpisodeTrace",
    "MujocoModelContext",
    "ORDER9_MUJOCO_TRACE_REPLAY_VERSION",
    "build_mujoco_model",
    "gather_mujoco_state",
    "hash_file",
    "initialize_mujoco_state",
    "load_isaac_episode_trace",
    "materialize_convex_hull_urdf",
    "replay_closed_loop_mujoco",
    "replay_saved_actuator_trace_mujoco",
    "resolve_source_urdf",
    "write_comparison_result",
]
