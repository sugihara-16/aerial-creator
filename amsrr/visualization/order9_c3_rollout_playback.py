from __future__ import annotations

"""Browser playback of saved Order 9 C3 Isaac rollout state.

The renderer is deliberately read-only.  It does not import Isaac Lab, step
PhysX, run a policy, or invoke QPID.  Recorded articulation-root, joint, and
object states are reconstructed with the hash-bound URDF solely for visual
inspection.
"""

import json
import math
import os
import re
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amsrr.geometry.pose_math import (
    Transform3D,
    transform_from_pose,
    transform_from_xyz_rpy,
)
from amsrr.robot_model.urdf_loader import URDFModel, load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_at_joint_positions
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    Order9TensorRolloutArtifact,
)
from amsrr.training.order9_tensor_reward import Order9TensorRewardGateConfig
from amsrr.utils.hashing import hash_file
from amsrr.visualization.order9_c3_curation import (
    Order9C3MeshViewerArtifacts,
    Order9C3UrdfMeshScene,
    Order9C3ViewerBox,
    build_order9_c3_urdf_mesh_scene,
    render_order9_c3_mesh_viewer,
)

ORDER9_C3_ROLLOUT_PLAYBACK_VERSION = (
    "order9_c3_saved_isaac_rollout_playback_v1_compact_urdf_fk"
)
ORDER9_C3_ROLLOUT_FRAME_ENCODING = "urdf_fk_v1"
_MODULE_JOINT_PATTERN = re.compile(r"^module_(\d+)__(.+)$")
_GROUND_DISPLAY_THICKNESS_M = 0.01


@dataclass(frozen=True)
class Order9C3RolloutPlaybackCase:
    bucket_id: str
    environment_index: int
    task_success: bool
    failure_reason: str | None
    duration_s: float
    frame_count: int
    html_path: Path
    scene_path: Path
    source_artifact_sha256: str


def load_order9_c3_evaluation_episode_rows(
    path: str | Path,
) -> dict[int, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        metadata = value.get("metadata")
        environment = (
            metadata.get("environment_index") if isinstance(metadata, dict) else None
        )
        if not isinstance(environment, int) or environment < 0:
            raise SchemaValidationError(
                f"evaluation episode line {line_number} lacks environment_index"
            )
        if environment in rows:
            raise SchemaValidationError(
                "evaluation episode file repeats an environment"
            )
        rows[environment] = value
    if not rows:
        raise SchemaValidationError("evaluation episode file is empty")
    return rows


def order9_c3_first_episode_indices(
    tensors: Mapping[str, torch.Tensor],
    environment_index: int,
) -> tuple[int, ...]:
    """Return the first recorded episode through its first terminal state."""

    valid = tensors["valid"]
    if valid.ndim != 2 or not 0 <= environment_index < valid.shape[1]:
        raise ValueError("playback environment index is out of range")
    indices = torch.nonzero(valid[:, environment_index], as_tuple=False).flatten()
    if indices.numel() == 0:
        raise SchemaValidationError("playback environment contains no valid states")
    serial_tensor = tensors["episode_serial"][:, environment_index]
    first_serial = int(serial_tensor[int(indices[0])].item())
    selected = [
        int(index)
        for index in indices.tolist()
        if int(serial_tensor[int(index)].item()) == first_serial
    ]
    if not selected:
        raise SchemaValidationError("playback first episode is empty")
    boundary = (
        tensors["terminal"][:, environment_index]
        | tensors["truncated"][:, environment_index]
    )
    terminal_offsets = [index for index in selected if bool(boundary[index])]
    if terminal_offsets:
        selected = selected[: selected.index(terminal_offsets[0]) + 1]
    return tuple(selected)


def sample_order9_c3_playback_indices(
    indices: Sequence[int],
    time_s: torch.Tensor,
    *,
    frames_per_second: int,
) -> tuple[int, ...]:
    if not 1 <= frames_per_second <= 20:
        raise ValueError("frames_per_second must lie within [1, 20]")
    if not indices:
        raise ValueError("playback sample requires source indices")
    values = [float(time_s[index].item()) for index in indices]
    if any(not math.isfinite(value) for value in values) or any(
        current + 1.0e-8 < previous for previous, current in zip(values, values[1:])
    ):
        raise SchemaValidationError("playback episode time is not monotonic")
    period = 1.0 / float(frames_per_second)
    selected = [int(indices[0])]
    next_time = values[0] + period
    for index, value in zip(indices[1:-1], values[1:-1]):
        if value + 1.0e-9 < next_time:
            continue
        selected.append(int(index))
        next_time = value + period
    if int(indices[-1]) != selected[-1]:
        selected.append(int(indices[-1]))
    return tuple(selected)


def render_order9_c3_rollout_artifact_environment(
    *,
    artifact: Order9TensorRolloutArtifact,
    artifact_path: str | Path,
    episode_row: Mapping[str, Any],
    robot_urdf_path: str | Path,
    html_path: str | Path,
    mesh_library_path: str | Path,
    viewer_javascript_path: str | Path,
    bucket_id: str,
    environment_index: int,
    frames_per_second: int = 10,
) -> tuple[Order9C3RolloutPlaybackCase, Order9C3MeshViewerArtifacts]:
    """Render one saved evaluation episode without executing simulation."""

    artifact.validate()
    if not 1 <= frames_per_second <= 20:
        raise ValueError("frames_per_second must lie within [1, 20]")
    if not 0 <= environment_index < artifact.environment_count:
        raise ValueError("playback environment index is out of range")
    artifact_source = Path(artifact_path).resolve()
    urdf_source = Path(robot_urdf_path).resolve()
    if not artifact_source.is_file():
        raise FileNotFoundError(artifact_source)
    if not urdf_source.is_file():
        raise FileNotFoundError(urdf_source)
    row_environment = _episode_environment_index(episode_row)
    if row_environment != environment_index:
        raise SchemaValidationError(
            "playback episode row does not match the selected environment"
        )
    task_values = artifact.metadata.get("task_specs")
    if (
        not isinstance(task_values, list)
        or len(task_values) != artifact.environment_count
    ):
        raise SchemaValidationError("rollout artifact task_specs differ")
    task = TaskSpec.from_dict(task_values[environment_index])
    module_ids = tuple(int(value) for value in artifact.metadata["module_ids"])
    local_joint_ids = tuple(
        str(value) for value in artifact.metadata["local_joint_ids"]
    )
    tensors = artifact.tensors
    episode_indices = order9_c3_first_episode_indices(tensors, environment_index)
    sampled_indices = sample_order9_c3_playback_indices(
        episode_indices,
        tensors["time_s"][:, environment_index],
        frames_per_second=frames_per_second,
    )
    model = load_urdf(urdf_source)
    fk_contract, coordinate_sources = _urdf_fk_contract(
        model,
        module_ids=module_ids,
        local_joint_ids=local_joint_ids,
    )
    first_index = sampled_indices[0]
    first_root = _pose(tensors["robot_root_pose_world"][first_index, environment_index])
    first_joint_positions = _joint_map(
        tensors["local_joint_positions_rad"][first_index, environment_index],
        module_ids=module_ids,
        local_joint_ids=local_joint_ids,
        known_joint_ids={joint.name for joint in model.joints},
    )
    scene = build_order9_c3_urdf_mesh_scene(
        urdf_source,
        joint_positions_rad=first_joint_positions,
        root_pose_world=first_root,
    )
    scene = _bind_scene_to_fk(
        scene,
        model=model,
        fk_contract=fk_contract,
        joint_positions=first_joint_positions,
        root_pose_world=first_root,
    )
    boxes, box_pose_static = _task_boxes(task)
    # ``phase_index`` is the actor-facing phase index.  It is intentionally
    # not contiguous with the runtime phase index (runtime lift maps to actor
    # lift, index 3), so using runtime labels here mislabels lift as transport.
    phase_labels = tuple(
        str(value) for value in artifact.metadata.get("actor_phase_labels", ())
    )
    if not phase_labels:
        phase_labels = tuple(
            str(value)
            for value in artifact.metadata.get("runtime_phase_labels", ())
        )
    contact_feature_names = tuple(
        str(value)
        for value in artifact.metadata.get("contact_space_feature_names", ())
    )
    mass_properties = artifact.metadata.get("object_mass_properties_readback")
    payload_mass_kg = (
        float(mass_properties["mass_kg"])
        if isinstance(mass_properties, Mapping)
        and isinstance(mass_properties.get("mass_kg"), (int, float))
        else float(artifact.metadata.get("estimated_payload_mass_kg", 0.0))
    )
    contact_force_threshold_n = (
        Order9TensorRewardGateConfig().contact_force_threshold_n
    )
    source_dt_s = float(artifact.metadata.get("control_dt_s", 0.02))
    if not math.isfinite(source_dt_s) or source_dt_s <= 0.0:
        raise SchemaValidationError("rollout control_dt_s is invalid")
    frames: list[dict[str, Any]] = []
    bounds_points: list[tuple[float, float, float, float]] = []
    for index in sampled_indices:
        frames.append(
            _playback_frame(
                tensors=tensors,
                source_index=index,
                environment_index=environment_index,
                coordinate_sources=coordinate_sources,
                phase_labels=phase_labels,
                contact_feature_names=contact_feature_names,
                contact_force_threshold_n=contact_force_threshold_n,
                payload_mass_kg=payload_mass_kg,
                box_pose_static=box_pose_static,
                episode_row=episode_row,
                post_state=False,
                time_offset_s=0.0,
            )
        )
        _append_frame_bounds(
            bounds_points,
            tensors=tensors,
            source_index=index,
            environment_index=environment_index,
        )
    last_index = episode_indices[-1]
    if bool(
        tensors["terminal"][last_index, environment_index]
        or tensors["truncated"][last_index, environment_index]
    ):
        frames.append(
            _playback_frame(
                tensors=tensors,
                source_index=last_index,
                environment_index=environment_index,
                coordinate_sources=coordinate_sources,
                phase_labels=phase_labels,
                contact_feature_names=contact_feature_names,
                contact_force_threshold_n=contact_force_threshold_n,
                payload_mass_kg=payload_mass_kg,
                box_pose_static=box_pose_static,
                episode_row=episode_row,
                post_state=True,
                time_offset_s=source_dt_s,
            )
        )
    for frame_index, frame in enumerate(frames):
        frame["frame_index"] = frame_index
    bounds = _playback_bounds(bounds_points, boxes)
    task_success = bool(episode_row.get("task_success"))
    failure_reason_value = episode_row.get("failure_reason")
    failure_reason = (
        str(failure_reason_value) if failure_reason_value is not None else None
    )
    result_label = "SUCCESS" if task_success else f"FAIL: {failure_reason}"
    duration_s = float(frames[-1]["time_s"] - frames[0]["time_s"])
    animation = {
        "animation_version": ORDER9_C3_ROLLOUT_PLAYBACK_VERSION,
        "frame_encoding": ORDER9_C3_ROLLOUT_FRAME_ENCODING,
        "frames_per_second": frames_per_second,
        "source_rate_hz": 1.0 / source_dt_s,
        "frame_count": len(frames),
        "duration_s": duration_s,
        "bounds": bounds,
        "urdf_fk": fk_contract,
        "frames": frames,
    }
    artifact_sha256 = hash_file(artifact_source)
    viewer = render_order9_c3_mesh_viewer(
        urdf_scene=scene,
        html_path=html_path,
        mesh_library_path=mesh_library_path,
        viewer_javascript_path=viewer_javascript_path,
        title=(f"C3 saved Isaac rollout: {bucket_id} / env {environment_index}"),
        subtitle=(
            f"{len(module_ids)} modules | {result_label} | "
            f"recorded {source_dt_s * 1000.0:.0f} ms state, "
            f"display {frames_per_second} Hz"
        ),
        boxes=boxes,
        metadata={
            "playback_version": ORDER9_C3_ROLLOUT_PLAYBACK_VERSION,
            "bucket_id": bucket_id,
            "environment_index": environment_index,
            "task_success": task_success,
            "failure_reason": failure_reason,
            "source_artifact_path": str(artifact_source),
            "source_artifact_sha256": artifact_sha256,
            "source_artifact_version": artifact.artifact_version,
            "pi_l_checkpoint_sha256": artifact.metadata.get("pi_l_checkpoint_sha256"),
            "physics_executed_during_playback": False,
            "policy_executed_during_playback": False,
            "qpid_executed_during_playback": False,
            "revalidation_performed": False,
            "recorded_state_reconstruction": (
                "artifact articulation-root/joint/object readback plus "
                "hash-bound URDF forward kinematics"
            ),
        },
        animation=animation,
        semantic_scope=(
            "exact URDF visual meshes reconstructed from the already-saved "
            "Isaac rollout articulation-root, joint, and object state; this "
            "playback does not rerun PhysX, pi_L, QPID, validation, collision "
            "admission, or success evaluation"
        ),
    )
    case = Order9C3RolloutPlaybackCase(
        bucket_id=bucket_id,
        environment_index=environment_index,
        task_success=task_success,
        failure_reason=failure_reason,
        duration_s=duration_s,
        frame_count=len(frames),
        html_path=viewer.html_path,
        scene_path=viewer.scene_path,
        source_artifact_sha256=artifact_sha256,
    )
    return case, viewer


def write_order9_c3_rollout_playback_index(
    path: str | Path,
    cases: Sequence[Order9C3RolloutPlaybackCase],
) -> Path:
    if not cases:
        raise ValueError("playback index requires at least one case")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        records.append(
            {
                "bucket_id": case.bucket_id,
                "environment_index": case.environment_index,
                "task_success": case.task_success,
                "failure_reason": case.failure_reason,
                "duration_s": case.duration_s,
                "frame_count": case.frame_count,
                "viewer": Path(
                    os.path.relpath(case.html_path, destination.parent)
                ).as_posix(),
            }
        )
    payload = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    title = "Order 9 C3 saved Isaac rollout playback"
    destination.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
html,body{{height:100%;margin:0;background:#eef1f5;color:#18212b;font-family:system-ui,sans-serif}}
body{{display:grid;grid-template-rows:auto 1fr;overflow:hidden}}
header{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:9px 12px;background:#fff;box-shadow:0 2px 8px #0002;z-index:2}}
h1{{font-size:16px;margin:0 12px 0 0}} select,button{{font:14px system-ui;padding:5px 8px}}
#result{{font:600 13px ui-monospace,monospace}} #scope{{font-size:12px;color:#52606d;margin-left:auto}}
iframe{{width:100%;height:100%;border:0;background:white}}
</style></head><body><header><h1>{escape(title)}</h1>
<label>bucket <select id="bucket"></select></label>
<label>environment <select id="environment"></select></label>
<button id="open">別タブで開く</button><span id="result"></span>
<span id="scope">保存stateの再生のみ（再validationなし）</span></header>
<iframe id="viewer" title="rollout playback"></iframe>
<script>const cases={payload};
const bucket=document.getElementById("bucket"),environment=document.getElementById("environment");
const viewer=document.getElementById("viewer"),result=document.getElementById("result");
for(const value of [...new Set(cases.map(x=>x.bucket_id))]){{const option=document.createElement("option");option.value=value;option.textContent=value;bucket.appendChild(option);}}
function environments(){{const previous=environment.value;environment.textContent="";for(const item of cases.filter(x=>x.bucket_id===bucket.value)){{const option=document.createElement("option");option.value=String(item.environment_index);option.textContent=String(item.environment_index);environment.appendChild(option);}}if([...environment.options].some(x=>x.value===previous))environment.value=previous;load();}}
function load(){{const item=cases.find(x=>x.bucket_id===bucket.value&&x.environment_index===Number(environment.value));if(!item)return;viewer.src=item.viewer;result.textContent=item.task_success?`SUCCESS | ${{item.duration_s.toFixed(2)}} s`:`FAIL: ${{item.failure_reason}} | ${{item.duration_s.toFixed(2)}} s`;result.style.color=item.task_success?"#147a39":"#b42318";}}
bucket.addEventListener("change",environments);environment.addEventListener("change",load);document.getElementById("open").addEventListener("click",()=>window.open(viewer.src,"_blank"));environments();
</script></body></html>""",
        encoding="utf-8",
    )
    manifest = {
        "playback_version": ORDER9_C3_ROLLOUT_PLAYBACK_VERSION,
        "physics_executed_during_playback": False,
        "revalidation_performed": False,
        "cases": records,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _episode_environment_index(row: Mapping[str, Any]) -> int:
    metadata = row.get("metadata")
    value = metadata.get("environment_index") if isinstance(metadata, dict) else None
    if not isinstance(value, int) or value < 0:
        raise SchemaValidationError("evaluation episode lacks environment_index")
    return value


def _urdf_fk_contract(
    model: URDFModel,
    *,
    module_ids: Sequence[int],
    local_joint_ids: Sequence[str],
) -> tuple[dict[str, Any], tuple[tuple[int, int], ...]]:
    if len(model.root_links) != 1:
        raise SchemaValidationError("playback URDF requires one root link")
    outgoing: dict[str, list[Any]] = {}
    for joint in model.joints:
        outgoing.setdefault(joint.parent_link, []).append(joint)
    for joints in outgoing.values():
        joints.sort(key=lambda value: value.name)
    links = [model.root_links[0]]
    link_index = {links[0]: 0}
    ordered_joints = []
    pending = [links[0]]
    while pending:
        parent = pending.pop(0)
        for joint in outgoing.get(parent, ()):
            if joint.child_link in link_index:
                raise SchemaValidationError("playback URDF is not a tree")
            link_index[joint.child_link] = len(links)
            links.append(joint.child_link)
            ordered_joints.append(joint)
            pending.append(joint.child_link)
    if len(links) != len(model.links):
        raise SchemaValidationError("playback URDF contains unreachable links")
    module_lookup = {int(value): index for index, value in enumerate(module_ids)}
    local_lookup = {str(value): index for index, value in enumerate(local_joint_ids)}
    coordinate_joint_ids = [
        joint.name for joint in ordered_joints if joint.joint_type != "fixed"
    ]
    coordinate_index = {
        joint_id: index for index, joint_id in enumerate(coordinate_joint_ids)
    }
    coordinate_sources: list[tuple[int, int]] = []
    for joint_id in coordinate_joint_ids:
        match = _MODULE_JOINT_PATTERN.match(joint_id)
        if match is None:
            raise SchemaValidationError(
                f"playback variable joint {joint_id!r} lacks module identity"
            )
        module_id = int(match.group(1))
        local_id = match.group(2)
        if module_id not in module_lookup or local_id not in local_lookup:
            raise SchemaValidationError(
                f"playback joint {joint_id!r} is absent from artifact readback"
            )
        coordinate_sources.append((module_lookup[module_id], local_lookup[local_id]))
    joints_payload = []
    for joint in ordered_joints:
        joints_payload.append(
            {
                "joint_id": joint.name,
                "joint_type": joint.joint_type,
                "parent_link_index": link_index[joint.parent_link],
                "child_link_index": link_index[joint.child_link],
                "origin_matrix": _transform_matrix(
                    transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
                ),
                "axis_xyz": [float(value) for value in joint.axis_xyz],
                "coordinate_index": coordinate_index.get(joint.name, -1),
            }
        )
    return (
        {
            "links": links,
            "root_link_index": 0,
            "coordinate_joint_ids": coordinate_joint_ids,
            "joints": joints_payload,
        },
        tuple(coordinate_sources),
    )


def _bind_scene_to_fk(
    scene: Order9C3UrdfMeshScene,
    *,
    model: URDFModel,
    fk_contract: Mapping[str, Any],
    joint_positions: Mapping[str, float],
    root_pose_world: Pose7D,
) -> Order9C3UrdfMeshScene:
    links = [str(value) for value in fk_contract["links"]]
    link_index = {value: index for index, value in enumerate(links)}
    poses = link_poses_at_joint_positions(
        model,
        joint_positions,
        root_pose_world=root_pose_world,
    )
    instances = []
    for source in scene.instances:
        link_id = str(source["link_id"])
        if link_id not in link_index or link_id not in poses:
            raise SchemaValidationError("playback mesh link is absent from FK")
        link_matrix = _pose_matrix(poses[link_id])
        local_matrix = _matrix_multiply(
            _rigid_inverse_matrix(link_matrix),
            [float(value) for value in source["model_matrix"]],
        )
        value = dict(source)
        value["link_index"] = link_index[link_id]
        value["local_matrix"] = local_matrix
        instances.append(value)
    labels = []
    for source in scene.module_labels:
        value = dict(source)
        match = re.match(r"^module-(\d+)$", str(value["label_id"]))
        link_id = f"module_{int(match.group(1))}__main_body" if match else ""
        if link_id in link_index:
            value["link_index"] = link_index[link_id]
        labels.append(value)
    return replace(
        scene,
        instances=tuple(instances),
        module_labels=tuple(labels),
    )


def _playback_frame(
    *,
    tensors: Mapping[str, torch.Tensor],
    source_index: int,
    environment_index: int,
    coordinate_sources: Sequence[tuple[int, int]],
    phase_labels: Sequence[str],
    contact_feature_names: Sequence[str],
    contact_force_threshold_n: float,
    payload_mass_kg: float,
    box_pose_static: Sequence[Pose7D],
    episode_row: Mapping[str, Any],
    post_state: bool,
    time_offset_s: float,
) -> dict[str, Any]:
    prefix = "post_" if post_state else ""
    root = tensors[f"{prefix}robot_root_pose_world"][source_index, environment_index]
    joints = tensors[f"{prefix}local_joint_positions_rad"][
        source_index, environment_index
    ]
    object_pose = tensors[f"{prefix}object_pose_world"][source_index, environment_index]
    phase_index = int(tensors["phase_index"][source_index, environment_index])
    phase = (
        phase_labels[phase_index]
        if 0 <= phase_index < len(phase_labels)
        else "complete"
    )
    qp_feasible = bool(tensors["qp_feasible"][source_index, environment_index])
    prohibited_collision = bool(
        tensors["prohibited_collision"][source_index, environment_index]
    )
    force = tensors["selected_contact_forces_world"][source_index, environment_index]
    mask = tensors["selected_assignment_mask"][source_index, environment_index]
    active_force = force[mask]
    minimum_force = (
        float(torch.linalg.vector_norm(active_force, dim=-1).min().item())
        if active_force.numel()
        else 0.0
    )
    reward = float(tensors["reward"][source_index, environment_index].item())
    diagnostics = _playback_diagnostics(
        tensors=tensors,
        source_index=source_index,
        environment_index=environment_index,
        prefix=prefix,
        active_mask=mask,
        contact_feature_names=contact_feature_names,
        contact_force_threshold_n=contact_force_threshold_n,
        payload_mass_kg=payload_mass_kg,
    )
    terminal = bool(tensors["terminal"][source_index, environment_index])
    if post_state:
        terminal = True
    status = (
        f"{phase} | QP={'ok' if qp_feasible else 'infeasible'} | "
        f"contact-min={minimum_force:.2f} N | reward={reward:.3f}"
    )
    if prohibited_collision:
        status += " | prohibited-collision"
    if terminal:
        if bool(episode_row.get("task_success")):
            status += " | TERMINAL SUCCESS"
        else:
            status += f" | TERMINAL {episode_row.get('failure_reason')}"
    return {
        "source_step_index": int(source_index),
        "time_s": _compact(
            float(tensors["time_s"][source_index, environment_index].item())
            + time_offset_s
        ),
        "phase": phase,
        "phase_index": phase_index,
        "status_text": status,
        "root_pose_world": _compact_list(root),
        "joint_positions": [
            _compact(float(joints[module, local].item()))
            for module, local in coordinate_sources
        ],
        "box_poses": [
            _compact_list(object_pose),
            *[list(value) for value in box_pose_static],
        ],
        "qp_feasible": qp_feasible,
        "prohibited_collision": prohibited_collision,
        "minimum_selected_contact_force_n": _compact(minimum_force),
        "reward": _compact(reward),
        "terminal": terminal,
        "post_terminal_state": post_state,
        "diagnostics": diagnostics,
    }


def _playback_diagnostics(
    *,
    tensors: Mapping[str, torch.Tensor],
    source_index: int,
    environment_index: int,
    prefix: str,
    active_mask: torch.Tensor,
    contact_feature_names: Sequence[str],
    contact_force_threshold_n: float,
    payload_mass_kg: float,
) -> dict[str, Any]:
    root_pose = tensors[f"{prefix}robot_root_pose_world"][
        source_index, environment_index
    ]
    object_pose = tensors[f"{prefix}object_pose_world"][
        source_index, environment_index
    ]
    command_pose = tensors["command_body_pose_world"][source_index, environment_index]
    desired_object_pose = tensors["desired_object_pose_world"][
        source_index, environment_index
    ]
    force_world = tensors["selected_contact_forces_world"][
        source_index, environment_index
    ]
    force_norms = torch.linalg.vector_norm(force_world, dim=-1)
    active_force_norms = force_norms[active_mask]

    vertical_support_n = 0.0
    relative_vertical_speeds: list[float] = []
    feature_index = {name: index for index, name in enumerate(contact_feature_names)}
    required = (
        "contact.normal_world.z",
        "contact.tangent_1_world.z",
        "contact.tangent_2_world.z",
        "feedback.relative_linear_velocity_contact.x",
        "feedback.relative_linear_velocity_contact.y",
        "feedback.relative_linear_velocity_contact.z",
    )
    if all(name in feature_index for name in required):
        actor_mask = tensors["actor_contact_slot_mask"][source_index, environment_index]
        actor_features = tensors["actor_contact_slot_features"][
            source_index, environment_index
        ]
        active_slots = torch.nonzero(actor_mask, as_tuple=False).flatten().tolist()
        contact_wrenches = tensors["selected_contact_wrenches_contact"][
            source_index, environment_index
        ]
        active_contact_indices = torch.nonzero(
            active_mask, as_tuple=False
        ).flatten().tolist()
        for contact_index, slot_index in zip(active_contact_indices, active_slots):
            features = actor_features[slot_index]
            axis_vertical = torch.tensor(
                [
                    features[feature_index["contact.normal_world.z"]],
                    features[feature_index["contact.tangent_1_world.z"]],
                    features[feature_index["contact.tangent_2_world.z"]],
                ],
                dtype=features.dtype,
                device=features.device,
            )
            vertical_support_n += float(
                torch.dot(contact_wrenches[contact_index, :3], axis_vertical).item()
            )
            relative_velocity_contact = torch.stack(
                [
                    features[
                        feature_index[
                            "feedback.relative_linear_velocity_contact.x"
                        ]
                    ],
                    features[
                        feature_index[
                            "feedback.relative_linear_velocity_contact.y"
                        ]
                    ],
                    features[
                        feature_index[
                            "feedback.relative_linear_velocity_contact.z"
                        ]
                    ],
                ]
            )
            relative_vertical_speeds.append(
                float(torch.dot(relative_velocity_contact, axis_vertical).item())
            )

    object_twist = tensors[f"{prefix}object_twist_world"][
        source_index, environment_index
    ]
    root_twist = tensors[f"{prefix}robot_root_twist_world"][
        source_index, environment_index
    ]
    return {
        "robot_root_height_m": _compact(float(root_pose[2].item())),
        "command_body_height_m": _compact(float(command_pose[2].item())),
        "object_height_m": _compact(float(object_pose[2].item())),
        "desired_object_height_m": _compact(float(desired_object_pose[2].item())),
        "robot_vertical_velocity_mps": _compact(float(root_twist[2].item())),
        "object_vertical_velocity_mps": _compact(float(object_twist[2].item())),
        "selected_contact_force_n": _compact_list(active_force_norms),
        "contact_force_threshold_n": _compact(contact_force_threshold_n),
        "vertical_support_force_n": _compact(vertical_support_n),
        "payload_weight_n": _compact(max(0.0, payload_mass_kg) * 9.80665),
        "contact_relative_vertical_speed_mps": [
            _compact(value) for value in relative_vertical_speeds
        ],
    }


def _task_boxes(
    task: TaskSpec,
) -> tuple[tuple[Order9C3ViewerBox, ...], tuple[Pose7D, ...]]:
    collision = build_order9_c3_posture_collision_object(task)
    if len(collision.environment_boxes) != 1:
        raise SchemaValidationError("playback requires one support box")
    if collision.ground_plane_z_m is None:
        raise SchemaValidationError("playback requires a ground plane")
    object_spec = next(
        (
            value
            for value in task.scene.objects
            if value.object_id == collision.object_id
        ),
        None,
    )
    if object_spec is None:
        raise SchemaValidationError("playback task object is absent")
    object_geometry = next(
        (
            value
            for value in task.scene.geometry_library
            if value.geometry_id == object_spec.geometry_id
        ),
        None,
    )
    if object_geometry is None or object_geometry.geometry_type.value != "box":
        raise SchemaValidationError("playback currently requires a box object")
    size_value = (object_geometry.primitive_params or {}).get("size_m")
    if not isinstance(size_value, (list, tuple)) or len(size_value) != 3:
        raise SchemaValidationError("playback box object lacks size_m")
    object_size = tuple(
        float(size_value[index]) * float(object_geometry.scale[index])
        for index in range(3)
    )
    support = collision.environment_boxes[0]
    ground_pose: Pose7D = (
        float(support.pose_world[0]),
        float(support.pose_world[1]),
        float(collision.ground_plane_z_m) - 0.5 * _GROUND_DISPLAY_THICKNESS_M,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    goal = next(
        (
            value
            for value in task.goals
            if value.target_entity_id == object_spec.object_id
            and value.target_pose_world is not None
        ),
        None,
    )
    goal_pose = (
        tuple(float(value) for value in goal.target_pose_world)
        if goal is not None
        else tuple(float(value) for value in object_spec.pose_world)
    )
    boxes = (
        Order9C3ViewerBox(
            object_id=object_spec.object_id,
            pose_world=tuple(object_spec.pose_world),
            size_m=object_size,
            color_rgba=(0.95, 0.66, 0.12, 0.55),
        ),
        Order9C3ViewerBox(
            object_id=support.box_id,
            pose_world=support.pose_world,
            size_m=support.size_m,
            color_rgba=(0.34, 0.34, 0.38, 0.42),
        ),
        Order9C3ViewerBox(
            object_id="ground_halfspace_display_patch",
            pose_world=ground_pose,
            size_m=(2.4, 2.4, _GROUND_DISPLAY_THICKNESS_M),
            color_rgba=(0.20, 0.48, 0.80, 0.10),
        ),
        Order9C3ViewerBox(
            object_id=f"{object_spec.object_id}:goal",
            pose_world=goal_pose,
            size_m=object_size,
            color_rgba=(0.16, 0.72, 0.32, 0.16),
        ),
    )
    return boxes, (support.pose_world, ground_pose, goal_pose)


def _append_frame_bounds(
    output: list[tuple[float, float, float, float]],
    *,
    tensors: Mapping[str, torch.Tensor],
    source_index: int,
    environment_index: int,
) -> None:
    modules = tensors["module_pose_world"][source_index, environment_index, :, :3]
    for value in modules.tolist():
        output.append((float(value[0]), float(value[1]), float(value[2]), 0.55))
    object_position = tensors["object_pose_world"][source_index, environment_index, :3]
    output.append(
        (
            float(object_position[0]),
            float(object_position[1]),
            float(object_position[2]),
            0.35,
        )
    )


def _playback_bounds(
    points: Sequence[tuple[float, float, float, float]],
    boxes: Sequence[Order9C3ViewerBox],
) -> dict[str, list[float]]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for x, y, z, margin in points:
        for axis, value in enumerate((x, y, z)):
            minimum[axis] = min(minimum[axis], value - margin)
            maximum[axis] = max(maximum[axis], value + margin)
    for box in boxes:
        for axis in range(3):
            half = 0.5 * float(box.size_m[axis])
            minimum[axis] = min(minimum[axis], float(box.pose_world[axis]) - half)
            maximum[axis] = max(maximum[axis], float(box.pose_world[axis]) + half)
    if not all(math.isfinite(value) for value in minimum + maximum):
        raise SchemaValidationError("playback bounds are non-finite")
    return {
        "min": [_compact(value) for value in minimum],
        "max": [_compact(value) for value in maximum],
    }


def _joint_map(
    values: torch.Tensor,
    *,
    module_ids: Sequence[int],
    local_joint_ids: Sequence[str],
    known_joint_ids: set[str] | None = None,
) -> dict[str, float]:
    if tuple(values.shape) != (len(module_ids), len(local_joint_ids)):
        raise SchemaValidationError("playback joint tensor shape differs")
    result = {}
    for module, module_id in enumerate(module_ids):
        for local, local_id in enumerate(local_joint_ids):
            joint_id = f"module_{module_id}__{local_id}"
            # The runtime readback contract may contain fixed diagnostic
            # frames added after a hash-bound morphology asset was composed.
            # They are not generalized coordinates and must not be passed to
            # FK for an older asset which does not contain those frames.
            if known_joint_ids is not None and joint_id not in known_joint_ids:
                continue
            result[joint_id] = float(values[module, local].item())
    return result


def _pose(values: torch.Tensor) -> Pose7D:
    result = tuple(float(value) for value in values.tolist())
    if len(result) != 7 or not all(math.isfinite(value) for value in result):
        raise SchemaValidationError("playback pose is invalid")
    return result  # type: ignore[return-value]


def _pose_matrix(pose: Sequence[float]) -> list[float]:
    return _transform_matrix(transform_from_pose(tuple(pose)))  # type: ignore[arg-type]


def _transform_matrix(transform: Transform3D) -> list[float]:
    rotation = transform.rotation
    return [
        rotation[0][0],
        rotation[1][0],
        rotation[2][0],
        0.0,
        rotation[0][1],
        rotation[1][1],
        rotation[2][1],
        0.0,
        rotation[0][2],
        rotation[1][2],
        rotation[2][2],
        0.0,
        transform.translation[0],
        transform.translation[1],
        transform.translation[2],
        1.0,
    ]


def _rigid_inverse_matrix(matrix: Sequence[float]) -> list[float]:
    if len(matrix) != 16:
        raise ValueError("matrix must contain 16 values")
    result = [0.0] * 16
    result[0], result[1], result[2] = matrix[0], matrix[4], matrix[8]
    result[4], result[5], result[6] = matrix[1], matrix[5], matrix[9]
    result[8], result[9], result[10] = matrix[2], matrix[6], matrix[10]
    result[15] = 1.0
    translation = (matrix[12], matrix[13], matrix[14])
    for row in range(3):
        result[12 + row] = -sum(
            result[4 * column + row] * translation[column] for column in range(3)
        )
    return result


def _matrix_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    if len(left) != 16 or len(right) != 16:
        raise ValueError("matrix must contain 16 values")
    output = [0.0] * 16
    for column in range(4):
        for row in range(4):
            output[4 * column + row] = sum(
                left[4 * inner + row] * right[4 * column + inner] for inner in range(4)
            )
    return [_compact(value) for value in output]


def _compact(value: float) -> float:
    if not math.isfinite(float(value)):
        raise SchemaValidationError("playback numeric value is non-finite")
    return round(float(value), 7)


def _compact_list(values: torch.Tensor) -> list[float]:
    return [_compact(float(value)) for value in values.tolist()]


__all__ = [
    "ORDER9_C3_ROLLOUT_FRAME_ENCODING",
    "ORDER9_C3_ROLLOUT_PLAYBACK_VERSION",
    "Order9C3RolloutPlaybackCase",
    "load_order9_c3_evaluation_episode_rows",
    "order9_c3_first_episode_indices",
    "render_order9_c3_rollout_artifact_environment",
    "sample_order9_c3_playback_indices",
    "write_order9_c3_rollout_playback_index",
]
